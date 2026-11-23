// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 CIRU AI.
// Native RDNA 3.5 (gfx1151) MoE prefill kernels for Inkling 2 Mini:
//   - Unified single-launch M128/M64 K32 (2+ CTAs/CU) WMMA MoE prefill across
//     source NVFP4, mixed INT4/C1, and coda C1 with zero-extra-pass 16-row fragment skip.
//   - Bit-exact deterministic slot-order (0..7) FP32->RTNE-BF16 route reduction.

#include "hip_common.cuh"

// 1-warp coalesced LDS-cached scan of plan64 expert_ids into:
//   UniDesc[0..Counts[2]-1]: (m64_idx << 1) | is_pair (1=128 rows, 0=64 tail rows)
//   Writes unified descriptors UniDesc into TailIdx and counts (pairs, tails, unified) into Counts[0..2].
__global__ void pair_m64_blocks_kernel(
    const int32_t* __restrict__ Expert,
    const int32_t* __restrict__ Padded,
    int32_t* __restrict__ PairIdx,
    int32_t* __restrict__ TailIdx,
    int32_t* __restrict__ Counts,
    int max_blocks
) {
    __shared__ int32_t s_exp[2048];
    int n_blocks = min(*Padded / 64, max_blocks);
    if (n_blocks <= 2048) {
        for (int i = threadIdx.x; i < n_blocks; i += 32) {
            s_exp[i] = Expert[i];
        }
        __syncthreads();
        if (threadIdx.x != 0) return;
        int np = 0, nt = 0, nu = 0, i = 0;
        while (i < n_blocks) {
            int e = s_exp[i];
            if (e < 0) { i += 1; continue; }
            if (i + 1 < n_blocks && s_exp[i + 1] == e) {
                PairIdx[np++] = i;
                TailIdx[nu++] = (i << 1) | 1;
                i += 2;
            } else {
                TailIdx[nu++] = (i << 1) | 0;
                nt += 1;
                i += 1;
            }
        }
        Counts[0] = np;
        Counts[1] = nt;
        Counts[2] = nu;
    } else {
        if (threadIdx.x != 0) return;
        int np = 0, nt = 0, nu = 0, i = 0;
        while (i < n_blocks) {
            int e = Expert[i];
            if (e < 0) { i += 1; continue; }
            if (i + 1 < n_blocks && Expert[i + 1] == e) {
                PairIdx[np++] = i;
                TailIdx[nu++] = (i << 1) | 1;
                i += 2;
            } else {
                TailIdx[nu++] = (i << 1) | 0;
                nt += 1;
                i += 1;
            }
        }
        Counts[0] = np;
        Counts[1] = nt;
        Counts[2] = nu;
    }
}

// Unified K32 2-CTA/CU RDNA 3.5 WMMA MoE prefill kernel:
//   MODE: 0=source NVFP4, 2=coda C1, 10=mixed(0/1 INT4), 20=mixed(0/2 C1)
//   FUSED: 0=w2 (BF16), 1=w13 fused SwiGLU (BF16), 2=w13 fused SwiGLU (FP16)
//   N_TILE=128 (W_M=4, W_N=2 -> 20.5 KiB LDS, 10 waves/SIMD) for w13
//   N_TILE=256 (W_M=2, W_N=4 -> 30.5 KiB LDS, 2 CTAs/CU) for w2
template <int M_TILE, int N_TILE, int W_M, int W_N, int FUSED, int TOKENS_DIV, int MODE, int X_IS_BF16>
__global__ __launch_bounds__(32 * W_M * W_N, 2)
void moe_wmma_k32_unified_kernel(
    const uint16_t* __restrict__ X,
    const uint32_t* __restrict__ Packed,
    const uint32_t* __restrict__ Raw,
    const float* __restrict__ Outer,
    const uint16_t* __restrict__ RowScale,
    const uint32_t* __restrict__ TableU32,
    const int32_t* __restrict__ Kinds,
    const int32_t* __restrict__ Sorted,
    const int32_t* __restrict__ Expert,
    const int32_t* __restrict__ UniDesc,
    const int32_t* __restrict__ Counts,
    uint16_t* __restrict__ Y,
    const float* __restrict__ Gamma,
    int N, int K, int ROUTES, float gate_cap, float up_cap
) {
    const int n_tiles = N / N_TILE;
    const int pid = blockIdx.x;
    const int m_ord = pid / n_tiles;
    const int n_blk = pid % n_tiles;
    if (m_ord >= Counts[2]) return;

    const int raw_desc = UniDesc[m_ord];
    const int m64_idx = raw_desc >> 1;
    const int active_rows = (raw_desc & 1) ? 128 : 64;
    const int expert = Expert[m64_idx];
    if (expert < 0) return;

    int kind = MODE;
    if constexpr (MODE == 10 || MODE == 20) {
        kind = Kinds[expert];
    }

    constexpr int K_STRIDE = 40;  // 32 + 8 f16 pad = 20 dwords -> 16B aligned & bank-conflict-free
    constexpr int N_THREADS = 32 * W_M * W_N;
    constexpr int WM_ROWS = M_TILE / W_M;
    constexpr int WN_COLS = N_TILE / W_N;
    constexpr int BM = WM_ROWS / 16;
    constexpr int BN = WN_COLS / 16;

    __shared__ uint16_t smem_A[M_TILE * K_STRIDE];
    __shared__ uint16_t smem_B[N_TILE * K_STRIDE];
    __shared__ int smem_route[M_TILE];

    const int tid = threadIdx.x;
    const int lane = tid & 31;
    const int warp = tid >> 5;
    const int wm = warp / W_N;
    const int wn = warp % W_N;
    const int lane16 = lane & 15;
    const int half_lane = lane >> 4;

    const int sorted_base = m64_idx * 64;
    for (int i = tid; i < M_TILE; i += N_THREADS) {
        smem_route[i] = (i < active_rows) ? Sorted[sorted_base + i] : -1;
    }
    __syncthreads();

    // Per-warp active 16-row fragment mask (skips all-padding fragments with zero extra weight passes)
    uint32_t frag_mask = 0;
    #pragma unroll
    for (int im = 0; im < BM; ++im) {
        int row = wm * WM_ROWS + im * 16 + lane16;
        int r = smem_route[row];
        if (__ballot(r >= 0 && r < ROUTES) != 0) frag_mask |= (1u << im);
    }

    constexpr int A_CHUNKS_PER_ROW = 4;
    constexpr int A_ITERS = (M_TILE * A_CHUNKS_PER_ROW) / N_THREADS;
    const uint16_t* a_ptr[A_ITERS];
    int a_smem_off[A_ITERS];
    bool a_valid[A_ITERS];
    #pragma unroll
    for (int it = 0; it < A_ITERS; ++it) {
        int idx = it * N_THREADS + tid;
        int r = idx / A_CHUNKS_PER_ROW;
        int c = (idx % A_CHUNKS_PER_ROW) * 8;
        int route = smem_route[r];
        a_valid[it] = (route >= 0 && route < ROUTES);
        a_ptr[it] = a_valid[it] ? (X + (int64_t)(route / TOKENS_DIV) * K + c) : nullptr;
        a_smem_off[it] = r * K_STRIDE + c;
    }

    constexpr int B_CHUNKS_PER_ROW = 2;
    constexpr int B_ITERS = (N_TILE * B_CHUNKS_PER_ROW) / N_THREADS;
    const int n_bank_tiles = N >> 7;
    const int k_bank_tiles = K >> 6;
    const uint32_t* b_pack_base[B_ITERS];
    const uint32_t* b_raw_base[B_ITERS];
    float b_row_scale[B_ITERS];
    int b_kk_sub[B_ITERS];
    int b_smem_off[B_ITERS];
    #pragma unroll
    for (int it = 0; it < B_ITERS; ++it) {
        int idx = it * N_THREADS + tid;
        int r = idx / B_CHUNKS_PER_ROW;
        int kk = idx % B_CHUNKS_PER_ROW;
        int n128_sub = r >> 7;
        int r128 = r & 127;
        int col = n_blk * N_TILE + r;
        int64_t tile0 = ((int64_t)expert * n_bank_tiles + (n_blk * (N_TILE >> 7) + n128_sub)) * k_bank_tiles;
        b_raw_base[it] = Raw + tile0 * 128 + r128;
        b_pack_base[it] = Packed + (tile0 * 128 + r128) * 8 + kk * 2;
        b_kk_sub[it] = kk;
        b_smem_off[it] = r * K_STRIDE + kk * 16;
        if constexpr (MODE == 2 || MODE == 20) {
            b_row_scale[it] = (kind == 2) ? f16_bits_to_f32(RowScale[(int64_t)expert * N + col]) : 1.0f;
        }
    }

    const uint32_t* tab_e = (MODE == 2 || MODE == 20) ? (TableU32 + (int64_t)expert * 256) : nullptr;

    float8 acc[BM][BN];
    #pragma unroll
    for (int im = 0; im < BM; ++im)
        #pragma unroll
        for (int in = 0; in < BN; ++in)
            acc[im][in] = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};

    const int n_k32 = K >> 5;
    uint4 reg_a[A_ITERS];
    uint2 reg_p[B_ITERS];
    uint32_t reg_raw[B_ITERS];

    auto prefetch32 = [&](int k32) {
        int k_off = k32 << 5;
        int kt64 = k32 >> 1;
        int half_k = k32 & 1;
        #pragma unroll
        for (int it = 0; it < A_ITERS; ++it) {
            reg_a[it] = a_valid[it]
                ? *reinterpret_cast<const uint4*>(a_ptr[it] + k_off)
                : make_uint4(0, 0, 0, 0);
        }
        #pragma unroll
        for (int it = 0; it < B_ITERS; ++it) {
            reg_raw[it] = b_raw_base[it][kt64 * 128];
            reg_p[it] = *reinterpret_cast<const uint2*>(b_pack_base[it] + (int64_t)kt64 * (128 * 8) + half_k * 4);
        }
    };

    prefetch32(0);
    for (int k32 = 0; k32 < n_k32; ++k32) {
        int half_k = k32 & 1;
        #pragma unroll
        for (int it = 0; it < A_ITERS; ++it) {
            uint4 val = a_valid[it] ? (X_IS_BF16 ? bf16x8_to_f16x8(reg_a[it]) : reg_a[it]) : make_uint4(0, 0, 0, 0);
            *reinterpret_cast<uint4*>(smem_A + a_smem_off[it]) = val;
        }
        #pragma unroll
        for (int it = 0; it < B_ITERS; ++it) {
            uint4 o0, o1;
            int kk4 = half_k * 2 + b_kk_sub[it];
            if (kind == 0) {
                uint32_t scale_byte = (reg_raw[it] >> (kk4 * 8)) & 0xFFu;
                unpack_nvfp4_16_u4(reg_p[it].x, reg_p[it].y, scale_byte, o0, o1);
            } else if (kind == 1) {
                uint16_t s_bf16 = (half_k == 0)
                    ? (uint16_t)(reg_raw[it] & 0xFFFFu)
                    : (uint16_t)(reg_raw[it] >> 16);
                unpack_int4_16_u4(reg_p[it].x, reg_p[it].y, bf16_to_f32(s_bf16), o0, o1);
            } else {
                uint32_t tag = (reg_raw[it] >> (kk4 * 8)) & 0xFFu;
                uint32_t entry = tab_e[tag];
                float s_val = f16_bits_to_f32((uint16_t)(entry & 0xFFFFu)) * b_row_scale[it];
                float o_val = f16_bits_to_f32((uint16_t)(entry >> 16)) * b_row_scale[it];
                unpack_c1_16_u4(reg_p[it].x, reg_p[it].y, s_val, o_val, o0, o1);
            }
            uint4* sp = reinterpret_cast<uint4*>(smem_B + b_smem_off[it]);
            sp[0] = o0;
            sp[1] = o1;
        }
        __syncthreads();

        if (k32 + 1 < n_k32) {
            prefetch32(k32 + 1);
        }

        if (frag_mask != 0) {
            #pragma unroll
            for (int kk = 0; kk < 32; kk += 16) {
                half16 a_frag[BM];
                #pragma unroll
                for (int im = 0; im < BM; ++im) {
                    if (frag_mask & (1u << im)) {
                        int row = wm * WM_ROWS + im * 16 + lane16;
                        const uint4* ap = reinterpret_cast<const uint4*>(smem_A + row * K_STRIDE + kk);
                        uint4 pair[2] = {ap[0], ap[1]};
                        __builtin_memcpy(&a_frag[im], pair, sizeof(half16));
                    }
                }
                #pragma unroll
                for (int in = 0; in < BN; ++in) {
                    int col = wn * WN_COLS + in * 16 + lane16;
                    const uint4* bp = reinterpret_cast<const uint4*>(smem_B + col * K_STRIDE + kk);
                    uint4 pair[2] = {bp[0], bp[1]};
                    half16 b_frag;
                    __builtin_memcpy(&b_frag, pair, sizeof(half16));
                    #pragma unroll
                    for (int im = 0; im < BM; ++im) {
                        if (frag_mask & (1u << im)) {
                            acc[im][in] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a_frag[im], b_frag, acc[im][in]);
                        }
                    }
                }
            }
        }
        __syncthreads();
    }

    if (frag_mask == 0) return;
    const float outer = (kind == 0) ? Outer[expert] : 1.0f;
    const float inv_g = (FUSED != 0) ? (1.0f / gate_cap) : 0.0f;
    const float inv_u = (FUSED != 0) ? (1.0f / up_cap) : 0.0f;
    const int n_col_base = n_blk * N_TILE + wn * WN_COLS;
    #pragma unroll
    for (int im = 0; im < BM; ++im) {
        if (!(frag_mask & (1u << im))) continue;
        int m_base = wm * WM_ROWS + im * 16;
        #pragma unroll
        for (int i = 0; i < 8; ++i) {
            int row_in_blk = half_lane + 2 * i;
            int r = smem_route[m_base + row_in_blk];
            if (r < 0 || r >= ROUTES) continue;
            if constexpr (FUSED) {
                float gam = Gamma[r];
                #pragma unroll
                for (int in = 0; in < BN; ++in) {
                    float v = bf16_round_f32(acc[im][in][i] * outer);
                    float up_v = __shfl_xor(v, 1, 32);
                    uint16_t h_val = 0;
                    if ((lane16 & 1) == 0) {
                        float mid = gate_cap * fast_tanh_hw(v * inv_g) * fast_sigmoid_hw(v)
                                  * up_cap * fast_tanh_hw(up_v * inv_u) * gam;
                        if constexpr (FUSED == 2) {
                            _Float16 h = (_Float16)bf16_round_f32(mid);
                            h_val = *reinterpret_cast<uint16_t*>(&h);
                        } else {
                            h_val = f32_to_bf16_rtne(mid);
                        }
                    }
                    uint32_t next_h = __shfl_down((uint32_t)h_val, 2, 16);
                    if ((lane16 & 3) == 0) {
                        uint32_t pk = (uint32_t)h_val | (next_h << 16);
                        int col_out = (n_col_base + in * 16 + lane16) >> 1;
                        *reinterpret_cast<uint32_t*>(Y + (int64_t)r * (N >> 1) + col_out) = pk;
                    }
                }
            } else {
                #pragma unroll
                for (int in = 0; in < BN; ++in) {
                    uint16_t h0 = f32_to_bf16_rtne(acc[im][in][i] * outer);
                    uint32_t h1 = __shfl_down((uint32_t)h0, 1, 16);
                    if ((lane16 & 1) == 0) {
                        uint32_t pk = (uint32_t)h0 | (h1 << 16);
                        int col_out = n_col_base + in * 16 + lane16;
                        *reinterpret_cast<uint32_t*>(Y + (int64_t)r * N + col_out) = pk;
                    }
                }
            }
        }
    }
}

// Bit-exact deterministic slot-order (0..7) FP32 route reduction -> RTNE BF16.
// Grid: (cdiv(N/8, blockDim.x), M), Block: 288 for N<=2304 (1 CTA/row, 0 idle lanes), else 256.
__global__ __launch_bounds__(288, 4)
void hip_moe_reduce_kernel(
    const uint16_t* __restrict__ Down,
    uint16_t* __restrict__ Out,
    int M, int N
) {
    int tok = blockIdx.y;
    int col = (blockIdx.x * blockDim.x + threadIdx.x) * 8;
    if (tok >= M || col >= N) return;

    float acc[8] = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
    const uint16_t* base = Down + (int64_t)tok * 8 * N + col;
    #pragma unroll
    for (int slot = 0; slot < 8; ++slot) {
        uint4 v = *reinterpret_cast<const uint4*>(base + (int64_t)slot * N);
        acc[0] += bf16_to_f32((uint16_t)(v.x & 0xFFFFu));
        acc[1] += bf16_to_f32((uint16_t)(v.x >> 16));
        acc[2] += bf16_to_f32((uint16_t)(v.y & 0xFFFFu));
        acc[3] += bf16_to_f32((uint16_t)(v.y >> 16));
        acc[4] += bf16_to_f32((uint16_t)(v.z & 0xFFFFu));
        acc[5] += bf16_to_f32((uint16_t)(v.z >> 16));
        acc[6] += bf16_to_f32((uint16_t)(v.w & 0xFFFFu));
        acc[7] += bf16_to_f32((uint16_t)(v.w >> 16));
    }
    uint4 out_v = make_uint4(
        pack_bf16x2(acc[0], acc[1]),
        pack_bf16x2(acc[2], acc[3]),
        pack_bf16x2(acc[4], acc[5]),
        pack_bf16x2(acc[6], acc[7])
    );
    *reinterpret_cast<uint4*>(Out + (int64_t)tok * N + col) = out_v;
}

extern "C" {

int inkling_hip_abi_version(void) {
    return INKLING_HIP_ABI_VERSION;
}

const char* inkling_hip_src_hash(void) {
    return INKLING_HIP_SRC_HASH;
}

void launch_pair_m64(
    const void* Expert, const void* Padded, void* PairIdx, void* TailIdx, void* Counts,
    int max_blocks, hipStream_t stream
) {
    hipLaunchKernelGGL(
        pair_m64_blocks_kernel, dim3(1), dim3(32), 0, stream,
        (const int32_t*)Expert, (const int32_t*)Padded,
        (int32_t*)PairIdx, (int32_t*)TailIdx, (int32_t*)Counts, max_blocks);
}

#define DEF_W13_LAUNCH(NAME, MODE, FUSED, TDIV, XBF16) \
void NAME( \
    const void* X, const void* Packed, const void* Raw, const void* Outer, \
    const void* RowScale, const void* TableU32, const void* Kinds, \
    const void* Sorted, const void* Expert, \
    const void* PairIdx, const void* UniDesc, const void* Counts, \
    void* Y, const void* Gamma, \
    int N, int K, int ROUTES, int max_m64_blocks, float gate_cap, float up_cap, hipStream_t stream \
) { \
    (void)PairIdx; \
    int max_uni = max_m64_blocks / 2 + min(max_m64_blocks, 128) + 1; \
    dim3 grid(max_uni * (N / 128), 1); \
    hipLaunchKernelGGL( \
        (moe_wmma_k32_unified_kernel<128, 128, 4, 2, FUSED, TDIV, MODE, XBF16>), grid, dim3(256), 0, stream, \
        (const uint16_t*)X, (const uint32_t*)Packed, (const uint32_t*)Raw, (const float*)Outer, \
        (const uint16_t*)RowScale, (const uint32_t*)TableU32, (const int32_t*)Kinds, \
        (const int32_t*)Sorted, (const int32_t*)Expert, (const int32_t*)UniDesc, (const int32_t*)Counts, \
        (uint16_t*)Y, (const float*)Gamma, N, K, ROUTES, gate_cap, up_cap); \
}

#define DEF_W2_LAUNCH(NAME, MODE) \
void NAME( \
    const void* X, const void* Packed, const void* Raw, const void* Outer, \
    const void* RowScale, const void* TableU32, const void* Kinds, \
    const void* Sorted, const void* Expert, \
    const void* PairIdx, const void* UniDesc, const void* Counts, \
    void* Y, const void* Gamma, \
    int N, int K, int ROUTES, int max_m64_blocks, float gate_cap, float up_cap, hipStream_t stream \
) { \
    (void)PairIdx; \
    int max_uni = max_m64_blocks / 2 + min(max_m64_blocks, 128) + 1; \
    if ((N & 255) == 0) { \
        dim3 grid(max_uni * (N / 256), 1); \
        hipLaunchKernelGGL( \
            (moe_wmma_k32_unified_kernel<128, 256, 2, 4, 0, 1, MODE, 0>), grid, dim3(256), 0, stream, \
            (const uint16_t*)X, (const uint32_t*)Packed, (const uint32_t*)Raw, (const float*)Outer, \
            (const uint16_t*)RowScale, (const uint32_t*)TableU32, (const int32_t*)Kinds, \
            (const int32_t*)Sorted, (const int32_t*)Expert, (const int32_t*)UniDesc, (const int32_t*)Counts, \
            (uint16_t*)Y, (const float*)Gamma, N, K, ROUTES, gate_cap, up_cap); \
    } else { \
        dim3 grid(max_uni * (N / 128), 1); \
        hipLaunchKernelGGL( \
            (moe_wmma_k32_unified_kernel<128, 128, 4, 2, 0, 1, MODE, 0>), grid, dim3(256), 0, stream, \
            (const uint16_t*)X, (const uint32_t*)Packed, (const uint32_t*)Raw, (const float*)Outer, \
            (const uint16_t*)RowScale, (const uint32_t*)TableU32, (const int32_t*)Kinds, \
            (const int32_t*)Sorted, (const int32_t*)Expert, (const int32_t*)UniDesc, (const int32_t*)Counts, \
            (uint16_t*)Y, (const float*)Gamma, N, K, ROUTES, gate_cap, up_cap); \
    } \
}

// BF16-direct X input variants for w13
DEF_W13_LAUNCH(launch_source_w13_f16_xbf16,   0, 2, 8, 1)
DEF_W13_LAUNCH(launch_source_w13_bf16_xbf16,  0, 1, 8, 1)
DEF_W13_LAUNCH(launch_mixed1_w13_f16_xbf16,  10, 2, 8, 1)
DEF_W13_LAUNCH(launch_mixed1_w13_bf16_xbf16, 10, 1, 8, 1)
DEF_W13_LAUNCH(launch_mixed2_w13_f16_xbf16,  20, 2, 8, 1)
DEF_W13_LAUNCH(launch_mixed2_w13_bf16_xbf16, 20, 1, 8, 1)
DEF_W13_LAUNCH(launch_coda_w13_f16_xbf16,     2, 2, 8, 1)
DEF_W13_LAUNCH(launch_coda_w13_bf16_xbf16,    2, 1, 8, 1)

// FP16 X input variants for w13 and w2
DEF_W13_LAUNCH(launch_source_w13_f16,   0, 2, 8, 0)
DEF_W13_LAUNCH(launch_source_w13_bf16,  0, 1, 8, 0)
DEF_W2_LAUNCH(launch_source_w2,         0)
DEF_W13_LAUNCH(launch_mixed1_w13_f16,  10, 2, 8, 0)
DEF_W13_LAUNCH(launch_mixed1_w13_bf16, 10, 1, 8, 0)
DEF_W2_LAUNCH(launch_mixed1_w2,        10)
DEF_W13_LAUNCH(launch_mixed2_w13_f16,  20, 2, 8, 0)
DEF_W13_LAUNCH(launch_mixed2_w13_bf16, 20, 1, 8, 0)
DEF_W2_LAUNCH(launch_mixed2_w2,        20)
DEF_W13_LAUNCH(launch_coda_w13_f16,     2, 2, 8, 0)
DEF_W13_LAUNCH(launch_coda_w13_bf16,    2, 1, 8, 0)
DEF_W2_LAUNCH(launch_coda_w2,           2)

void launch_hip_moe_reduce(const void* Down, void* Out, int M, int N, hipStream_t stream) {
    if (M <= 0 || N <= 0) return;
    int n_vec = N / 8;
    int blk = (n_vec <= 288) ? ((n_vec + 31) & ~31) : 256;
    dim3 grid((n_vec + blk - 1) / blk, M);
    hipLaunchKernelGGL(
        hip_moe_reduce_kernel, grid, dim3(blk), 0, stream,
        (const uint16_t*)Down, (uint16_t*)Out, M, N);
}

}
