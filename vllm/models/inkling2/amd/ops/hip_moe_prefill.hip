// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 CIRU AI.
// Native RDNA 3.5 (gfx1151) prefill kernels for Inkling 2 Mini:
//   1. Paired-M128 + M64-tail (1x4) K64 WMMA MoE prefill (source NVFP4, mixed INT4/C1, coda C1)
//      with direct BF16 X-load on w13, hardware exp2f SwiGLU, uint32 coalesced stores, and FP16 handoff.
//   2. 1-warp (zero-barrier, zero-LDS, uint2 coalesced) fused K+V short-conv + L2-clip/RMSNorm prefill.
//   3. Zero-materialization 4-query AttnRes message coefficient kernel (BF16 bank -> [4, T, 2] FP32).

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <stdint.h>

typedef _Float16 half16 __attribute__((ext_vector_type(16)));
typedef float float8 __attribute__((ext_vector_type(8)));

__device__ __forceinline__ uint16_t f32_to_bf16_rtne(float x) {
    uint32_t u = __float_as_uint(x);
    uint32_t lsb = (u >> 16) & 1u;
    u += 0x7FFFu + lsb;
    return (uint16_t)(u >> 16);
}

__device__ __forceinline__ float bf16_round_f32(float x) {
    uint16_t h = f32_to_bf16_rtne(x);
    return __uint_as_float((uint32_t)h << 16);
}

__device__ __forceinline__ float bf16_to_f32(uint16_t h) {
    return __uint_as_float((uint32_t)h << 16);
}

__device__ __forceinline__ uint32_t pack_bf16x2(float a, float b) {
    return (uint32_t)f32_to_bf16_rtne(a) | ((uint32_t)f32_to_bf16_rtne(b) << 16);
}

__device__ __forceinline__ float f16_bits_to_f32(uint16_t h) {
    _Float16 f;
    __builtin_memcpy(&f, &h, sizeof(f));
    return (float)f;
}

__device__ __forceinline__ float fast_tanh_hw(float x) {
    float ax = fabsf(x);
    float e = exp2f(-2.8853900817779268f * ax);
    return copysignf((1.0f - e) / (1.0f + e), x);
}

__device__ __forceinline__ float fast_sigmoid_hw(float x) {
    return 1.0f / (1.0f + exp2f(-1.4426950408889634f * x));
}

__device__ __forceinline__ uint4 bf16x8_to_f16x8(uint4 v) {
    auto cvt2 = [](uint32_t u) -> uint32_t {
        float f0 = bf16_to_f32((uint16_t)(u & 0xFFFFu));
        float f1 = bf16_to_f32((uint16_t)(u >> 16));
        uint32_t pk;
        asm("v_cvt_pkrtz_f16_f32 %0, %1, %2" : "=v"(pk) : "v"(f0), "v"(f1));
        return pk;
    };
    return make_uint4(cvt2(v.x), cvt2(v.y), cvt2(v.z), cvt2(v.w));
}

// ============================================================================
// 1. PAIRED-M128 + M64-TAIL K64 WMMA MOE PREFILL
// ============================================================================

// 1-warp coalesced LDS-cached scan of plan64's expert_ids into:
//   PairIdx[0..Counts[0]-1]: starting m64 index `b` where Expert[b] == Expert[b+1] (128 contiguous rows)
//   TailIdx[0..Counts[1]-1]: leftover single m64 index `b` (64 rows)
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
        int np = 0, nt = 0, i = 0;
        while (i < n_blocks) {
            int e = s_exp[i];
            if (e < 0) { i += 1; continue; }
            if (i + 1 < n_blocks && s_exp[i + 1] == e) {
                PairIdx[np++] = i;
                i += 2;
            } else {
                TailIdx[nt++] = i;
                i += 1;
            }
        }
        Counts[0] = np;
        Counts[1] = nt;
    } else {
        if (threadIdx.x != 0) return;
        int np = 0, nt = 0, i = 0;
        while (i < n_blocks) {
            int e = Expert[i];
            if (e < 0) { i += 1; continue; }
            if (i + 1 < n_blocks && Expert[i + 1] == e) {
                PairIdx[np++] = i;
                i += 2;
            } else {
                TailIdx[nt++] = i;
                i += 1;
            }
        }
        Counts[0] = np;
        Counts[1] = nt;
    }
}

// Unpack 8 packed bytes (= 16 NVFP4 nibbles, 1 E4M3 scale byte) -> 2x uint4 (= 16 f16s)
__device__ __forceinline__ void unpack_nvfp4_16_u4(
    uint32_t p0, uint32_t p1, uint32_t scale_byte, uint4 &o0, uint4 &o1
) {
    uint32_t scale_f16 = (scale_byte << 7) + 0x1C00u;
    uint32_t scale_pk = scale_f16 | (scale_f16 << 16);
    uint32_t dst[8];
    const uint32_t words[2] = {p0, p1};
    #pragma unroll
    for (int w = 0; w < 2; ++w) {
        uint32_t u = words[w];
        #pragma unroll
        for (int b = 0; b < 4; ++b) {
            uint32_t packed = (u >> (b * 8)) & 0xFFu;
            uint32_t sel = 0x000C000Cu | ((packed & 7u) << 8) | ((packed & 0x70u) << 20);
            uint32_t mags;
            asm("v_perm_b32 %0, %1, %2, %3"
                : "=v"(mags) : "v"(0x4A484644u), "v"(0x42403C00u), "v"(sel));
            uint32_t sgn = mags | ((packed & 8u) << 12) | ((packed & 128u) << 24);
            uint32_t scaled;
            asm("v_pk_mul_f16 %0, %1, %2"
                : "=v"(scaled) : "v"(sgn), "v"(scale_pk));
            dst[w * 4 + b] = scaled;
        }
    }
    o0 = make_uint4(dst[0], dst[1], dst[2], dst[3]);
    o1 = make_uint4(dst[4], dst[5], dst[6], dst[7]);
}

// Unpack 8 packed bytes (= 16 signed INT4 values, 1 BF16 group-32 scale) -> 2x uint4 (= 16 f16s)
__device__ __forceinline__ void unpack_int4_16_u4(
    uint32_t p0, uint32_t p1, float scale_f32, uint4 &o0, uint4 &o1
) {
    uint32_t dst[8];
    const uint32_t words[2] = {p0, p1};
    #pragma unroll
    for (int w = 0; w < 2; ++w) {
        uint32_t u = words[w];
        #pragma unroll
        for (int b = 0; b < 4; ++b) {
            uint32_t packed = (u >> (b * 8)) & 0xFFu;
            float f0 = (float)(((int)(packed & 15u) ^ 8) - 8) * scale_f32;
            float f1 = (float)(((int)(packed >> 4) ^ 8) - 8) * scale_f32;
            uint32_t pk;
            asm("v_cvt_pkrtz_f16_f32 %0, %1, %2" : "=v"(pk) : "v"(f0), "v"(f1));
            dst[w * 4 + b] = pk;
        }
    }
    o0 = make_uint4(dst[0], dst[1], dst[2], dst[3]);
    o1 = make_uint4(dst[4], dst[5], dst[6], dst[7]);
}

// Unpack 8 packed bytes (= 16 C1 U4 values, FP32 affine scale & offset) -> 2x uint4 (= 16 f16s)
__device__ __forceinline__ void unpack_c1_16_u4(
    uint32_t p0, uint32_t p1, float s_val, float o_val, uint4 &o0, uint4 &o1
) {
    uint32_t dst[8];
    const uint32_t words[2] = {p0, p1};
    #pragma unroll
    for (int w = 0; w < 2; ++w) {
        uint32_t u = words[w];
        #pragma unroll
        for (int b = 0; b < 4; ++b) {
            uint32_t packed = (u >> (b * 8)) & 0xFFu;
            float f0 = fmaf((float)(packed & 15u), s_val, o_val);
            float f1 = fmaf((float)(packed >> 4), s_val, o_val);
            uint32_t pk;
            asm("v_cvt_pkrtz_f16_f32 %0, %1, %2" : "=v"(pk) : "v"(f0), "v"(f1));
            dst[w * 4 + b] = pk;
        }
    }
    o0 = make_uint4(dst[0], dst[1], dst[2], dst[3]);
    o1 = make_uint4(dst[4], dst[5], dst[6], dst[7]);
}

// Unified K64 RDNA 3.5 WMMA kernel:
//   MODE=0:  source NVFP4
//   MODE=10: mixed NVFP4 (kind==0) + signed INT4 (kind==1)
//   MODE=20: mixed NVFP4 (kind==0) + C1 (kind==2)
//   MODE=2:  coda C1
//   FUSED=0: plain projection -> BF16 [T, top_k, N]
//   FUSED=1: fused softcap SwiGLU -> BF16 [T, top_k, N/2]
//   FUSED=2: fused softcap SwiGLU -> FP16 [T, top_k, N/2] (exact BF16-rounded values for zero-cast w2!)
//   X_IS_BF16=1: reads BF16 X directly and converts to FP16 in registers before LDS store (zero cast kernel!)
template <int M_TILE, int W_M, int W_N, int FUSED, int TOKENS_DIV, int MODE, int X_IS_BF16>
__global__ __launch_bounds__(32 * W_M * W_N, 2)
void moe_wmma_k64_unified_kernel(
    const uint16_t* __restrict__ X,          // BF16 or FP16 [T, K] or [T*top_k, K]
    const uint32_t* __restrict__ Packed,     // [E, N/128, K/64, 128, 8] uint32
    const uint32_t* __restrict__ Raw,        // [E, N/128, K/64, 128] uint32
    const float* __restrict__ Outer,         // [E] fp32 (NVFP4)
    const uint16_t* __restrict__ RowScale,   // [E, N] fp16 (C1)
    const uint32_t* __restrict__ TableU32,   // [E, 256] uint32 = 2x fp16 (scale, offset) (C1)
    const int32_t* __restrict__ Kinds,       // [E] int32 (Mixed)
    const int32_t* __restrict__ Sorted,
    const int32_t* __restrict__ Expert,
    const int32_t* __restrict__ BlockIdxList,
    const int32_t* __restrict__ CountPtr,
    uint16_t* __restrict__ Y,
    const float* __restrict__ Gamma,
    int N, int K, int ROUTES, float gate_cap, float up_cap
) {
    const int active_blocks = *CountPtr;
    if (active_blocks <= 0) return;

    const int n_tiles_grid = N >> 7;
    int pid = blockIdx.x;
    int grp_m = pid / (n_tiles_grid * 4);
    int rem = pid % (n_tiles_grid * 4);
    int n_blk = rem / 4;
    int list_idx = grp_m * 4 + (rem & 3);
    if (list_idx >= active_blocks) return;

    const int m64_idx = BlockIdxList[list_idx];
    const int expert = Expert[m64_idx];
    if (expert < 0) return;

    int kind = MODE;
    if constexpr (MODE == 10 || MODE == 20) {
        kind = Kinds[expert];
    }

    constexpr int N_TILE = 128;
    constexpr int K_STRIDE = 72;  // 64 + 8 f16 pad = 36 dwords -> zero b128 bank conflicts
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
        smem_route[i] = Sorted[sorted_base + i];
    }
    __syncthreads();

    constexpr int A_CHUNKS_PER_ROW = 8;
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

    constexpr int B_CHUNKS_PER_ROW = 4;
    constexpr int B_ITERS = (N_TILE * B_CHUNKS_PER_ROW) / N_THREADS;
    const int n_bank_tiles = N >> 7;
    const int k_bank_tiles = K >> 6;
    const uint32_t* b_pack_base[B_ITERS];
    const uint32_t* b_raw_base[B_ITERS];
    float b_row_scale[B_ITERS];
    int b_kk[B_ITERS];
    int b_smem_off[B_ITERS];
    #pragma unroll
    for (int it = 0; it < B_ITERS; ++it) {
        int idx = it * N_THREADS + tid;
        int r = idx / B_CHUNKS_PER_ROW;
        int kk = idx % B_CHUNKS_PER_ROW;
        int col = n_blk * 128 + r;
        int64_t tile0 = ((int64_t)expert * n_bank_tiles + n_blk) * k_bank_tiles;
        b_raw_base[it] = Raw + tile0 * 128 + r;
        b_pack_base[it] = Packed + (tile0 * 128 + r) * 8 + kk * 2;
        b_kk[it] = kk;
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

    uint4 reg_a[A_ITERS];
    uint2 reg_p[B_ITERS];
    uint32_t reg_raw[B_ITERS];

    auto prefetch = [&](int kt) {
        int k_off = kt << 6;
        #pragma unroll
        for (int it = 0; it < A_ITERS; ++it) {
            reg_a[it] = a_valid[it]
                ? *reinterpret_cast<const uint4*>(a_ptr[it] + k_off)
                : make_uint4(0, 0, 0, 0);
        }
        #pragma unroll
        for (int it = 0; it < B_ITERS; ++it) {
            reg_raw[it] = b_raw_base[it][kt * 128];
            reg_p[it] = *reinterpret_cast<const uint2*>(b_pack_base[it] + (int64_t)kt * (128 * 8));
        }
    };

    prefetch(0);
    for (int kt = 0; kt < k_bank_tiles; ++kt) {
        #pragma unroll
        for (int it = 0; it < A_ITERS; ++it) {
            uint4 val = X_IS_BF16 ? bf16x8_to_f16x8(reg_a[it]) : reg_a[it];
            *reinterpret_cast<uint4*>(smem_A + a_smem_off[it]) = val;
        }
        #pragma unroll
        for (int it = 0; it < B_ITERS; ++it) {
            uint4 o0, o1;
            if (kind == 0) {
                uint32_t scale_byte = (reg_raw[it] >> (b_kk[it] * 8)) & 0xFFu;
                unpack_nvfp4_16_u4(reg_p[it].x, reg_p[it].y, scale_byte, o0, o1);
            } else if (kind == 1) {
                uint16_t s_bf16 = (b_kk[it] < 2)
                    ? (uint16_t)(reg_raw[it] & 0xFFFFu)
                    : (uint16_t)(reg_raw[it] >> 16);
                unpack_int4_16_u4(reg_p[it].x, reg_p[it].y, bf16_to_f32(s_bf16), o0, o1);
            } else {
                uint32_t tag = (reg_raw[it] >> (b_kk[it] * 8)) & 0xFFu;
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

        if (kt + 1 < k_bank_tiles) {
            prefetch(kt + 1);
        }

        #pragma unroll
        for (int kk = 0; kk < 64; kk += 16) {
            half16 a_frag[BM];
            #pragma unroll
            for (int im = 0; im < BM; ++im) {
                int row = wm * WM_ROWS + im * 16 + lane16;
                const uint4* ap = reinterpret_cast<const uint4*>(smem_A + row * K_STRIDE + kk);
                uint4 pair[2] = {ap[0], ap[1]};
                __builtin_memcpy(&a_frag[im], pair, sizeof(half16));
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
                    acc[im][in] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a_frag[im], b_frag, acc[im][in]);
                }
            }
        }
        __syncthreads();
    }

    const float outer = (kind == 0) ? Outer[expert] : 1.0f;
    const float inv_g = (FUSED != 0) ? (1.0f / gate_cap) : 0.0f;
    const float inv_u = (FUSED != 0) ? (1.0f / up_cap) : 0.0f;
    const int n_col_base = n_blk * 128 + wn * WN_COLS;
    #pragma unroll
    for (int im = 0; im < BM; ++im) {
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

// ============================================================================
// 2. 1-WARP (ZERO-BARRIER, ZERO-LDS, uint2 COALESCED) FUSED K+V SHORT-CONV
//    Handles BOTH K and V in ONE launch! D=128 -> 32 lanes x 4 bf16 (uint2 = 8B)
//    Block = (128,) = 4 independent warps (each warp owns 1 (t_tile, head)!)
// ============================================================================
template <int BT, int DYNAMIC>
__global__ __launch_bounds__(128, 4)
void hip_kv_conv_1warp_kernel(
    const uint16_t* __restrict__ K_in,
    const uint16_t* __restrict__ V_in,
    const uint16_t* __restrict__ Wk,
    const uint16_t* __restrict__ Wv,
    const uint16_t* __restrict__ Dk,
    const uint16_t* __restrict__ Dv,
    uint16_t* __restrict__ ConvCache,
    uint16_t* __restrict__ KeyCache,
    uint16_t* __restrict__ ValCache,
    const int64_t* __restrict__ KvSlots,
    const int32_t* __restrict__ Pos,
    const int32_t* __restrict__ SeqIdx,
    const int64_t* __restrict__ ConvSlots,
    const int32_t* __restrict__ BlockTable,
    const int32_t* __restrict__ QStart,
    int T, int KVH, int64_t stride_k, int64_t stride_v,
    int64_t stride_w0, int64_t stride_w1, int64_t stride_dk, int64_t stride_dv,
    int64_t stride_c_blk, int64_t stride_c_h, int64_t stride_c_n,
    int64_t stride_kv_blk, int64_t stride_kv_n, int64_t stride_kv_h,
    int kv_block_size, int64_t stride_bt, int max_blocks, int N_conv,
    float scale, float clip_threshold, float clip_eps, float norm_eps
) {
    const int warp = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    const int tile_idx = blockIdx.x * 4 + warp;
    const int n_tiles = (T + BT - 1) / BT;
    if (tile_idx >= n_tiles) return;
    const int t0 = tile_idx * BT;
    const int head = blockIdx.y;
    const int d0 = lane * 4;
    const int col0 = head * 128 + d0;

    float wk_s[4][4], wv_s[4][4];
    #pragma unroll
    for (int lag = 0; lag < 4; ++lag) {
        if constexpr (DYNAMIC) {
            uint2 pk = *reinterpret_cast<const uint2*>(Wk + lag * stride_w0 + col0 * stride_w1);
            uint2 pv = *reinterpret_cast<const uint2*>(Wv + lag * stride_w0 + col0 * stride_w1);
            wk_s[lag][0] = bf16_to_f32(pk.x & 0xFFFFu); wk_s[lag][1] = bf16_to_f32(pk.x >> 16);
            wk_s[lag][2] = bf16_to_f32(pk.y & 0xFFFFu); wk_s[lag][3] = bf16_to_f32(pk.y >> 16);
            wv_s[lag][0] = bf16_to_f32(pv.x & 0xFFFFu); wv_s[lag][1] = bf16_to_f32(pv.x >> 16);
            wv_s[lag][2] = bf16_to_f32(pv.y & 0xFFFFu); wv_s[lag][3] = bf16_to_f32(pv.y >> 16);
        } else {
            int k_idx = 3 - lag;
            #pragma unroll
            for (int c = 0; c < 4; ++c) {
                wk_s[lag][c] = bf16_to_f32(Wk[(col0 + c) * stride_w0 + k_idx * stride_w1]);
                wv_s[lag][c] = bf16_to_f32(Wv[(col0 + c) * stride_w0 + k_idx * stride_w1]);
            }
        }
    }

    auto load_u2_f32 = [](const uint16_t* p, float out[4]) {
        uint2 u = *reinterpret_cast<const uint2*>(p);
        out[0] = bf16_to_f32(u.x & 0xFFFFu); out[1] = bf16_to_f32(u.x >> 16);
        out[2] = bf16_to_f32(u.y & 0xFFFFu); out[3] = bf16_to_f32(u.y >> 16);
    };

    float kh[3][4] = {{0.f}}, vh[3][4] = {{0.f}};
    const int qst0 = QStart[t0];
    const int pos0 = Pos[t0];
    const int req0 = SeqIdx[t0];
    if (ConvSlots[t0] >= 0) {
        #pragma unroll
        for (int h_i = 0; h_i < 3; ++h_i) {
            int lag = 3 - h_i;
            int row = t0 - lag;
            int src = pos0 - lag;
            if (src >= 0) {
                if (row >= qst0) {
                    load_u2_f32(K_in + (int64_t)row * stride_k + col0, kh[h_i]);
                    load_u2_f32(V_in + (int64_t)row * stride_v + col0, vh[h_i]);
                } else {
                    int blk_i = min(src / N_conv, max_blocks - 1);
                    int64_t pblk = BlockTable[(int64_t)req0 * stride_bt + blk_i];
                    int64_t c_off = pblk * stride_c_blk + (int64_t)(src % N_conv) * stride_c_n + (int64_t)head * stride_c_h + d0;
                    load_u2_f32(ConvCache + c_off, kh[h_i]);
                    load_u2_f32(ConvCache + c_off + 128, vh[h_i]);
                }
            }
        }
    }

    const int t_end = min(t0 + BT, T);
    for (int t = t0; t < t_end; ++t) {
        int64_t cslot = ConvSlots[t];
        int qst = QStart[t];
        int pos = Pos[t];
        int req = SeqIdx[t];
        if (t > t0 && qst == t) {
            #pragma unroll
            for (int h_i = 0; h_i < 3; ++h_i) {
                int lag = 3 - h_i;
                int src = pos - lag;
                if (cslot >= 0 && src >= 0) {
                    int blk_i = min(src / N_conv, max_blocks - 1);
                    int64_t pblk = BlockTable[(int64_t)req * stride_bt + blk_i];
                    int64_t c_off = pblk * stride_c_blk + (int64_t)(src % N_conv) * stride_c_n + (int64_t)head * stride_c_h + d0;
                    load_u2_f32(ConvCache + c_off, kh[h_i]);
                    load_u2_f32(ConvCache + c_off + 128, vh[h_i]);
                } else {
                    #pragma unroll
                    for (int c = 0; c < 4; ++c) { kh[h_i][c] = 0.f; vh[h_i][c] = 0.f; }
                }
            }
        }

        uint2 ku = *reinterpret_cast<const uint2*>(K_in + (int64_t)t * stride_k + col0);
        uint2 vu = *reinterpret_cast<const uint2*>(V_in + (int64_t)t * stride_v + col0);
        float kc[4] = {bf16_to_f32(ku.x & 0xFFFFu), bf16_to_f32(ku.x >> 16), bf16_to_f32(ku.y & 0xFFFFu), bf16_to_f32(ku.y >> 16)};
        float vc[4] = {bf16_to_f32(vu.x & 0xFFFFu), bf16_to_f32(vu.x >> 16), bf16_to_f32(vu.y & 0xFFFFu), bf16_to_f32(vu.y >> 16)};

        if (cslot >= 0) {
            int64_t c_off = (cslot / N_conv) * stride_c_blk + (cslot % N_conv) * stride_c_n + (int64_t)head * stride_c_h + d0;
            *reinterpret_cast<uint2*>(ConvCache + c_off) = ku;
            *reinterpret_cast<uint2*>(ConvCache + c_off + 128) = vu;
        }

        float dk[4], dv[4];
        if constexpr (DYNAMIC) {
            uint2 dku = *reinterpret_cast<const uint2*>(Dk + (int64_t)t * stride_dk + head * 4);
            uint2 dvu = *reinterpret_cast<const uint2*>(Dv + (int64_t)t * stride_dv + head * 4);
            dk[0] = bf16_to_f32(dku.x & 0xFFFFu); dk[1] = bf16_to_f32(dku.x >> 16);
            dk[2] = bf16_to_f32(dku.y & 0xFFFFu); dk[3] = bf16_to_f32(dku.y >> 16);
            dv[0] = bf16_to_f32(dvu.x & 0xFFFFu); dv[1] = bf16_to_f32(dvu.x >> 16);
            dv[2] = bf16_to_f32(dvu.y & 0xFFFFu); dv[3] = bf16_to_f32(dvu.y >> 16);
        }

        float ky[4], vy[4];
        float ksq = 0.f, vsq = 0.f;
        bool valid = (cslot >= 0);
        #pragma unroll
        for (int c = 0; c < 4; ++c) {
            float ka, va;
            if (valid) {
                if constexpr (DYNAMIC) {
                    ka = kc[c] * (wk_s[0][c] + dk[0]);
                    ka = fmaf(kh[2][c], wk_s[1][c] + dk[1], ka);
                    ka = fmaf(kh[1][c], wk_s[2][c] + dk[2], ka);
                    ka = fmaf(kh[0][c], wk_s[3][c] + dk[3], ka);
                    va = vc[c] * (wv_s[0][c] + dv[0]);
                    va = fmaf(vh[2][c], wv_s[1][c] + dv[1], va);
                    va = fmaf(vh[1][c], wv_s[2][c] + dv[2], va);
                    va = fmaf(vh[0][c], wv_s[3][c] + dv[3], va);
                } else {
                    ka = kh[0][c] * wk_s[3][c];
                    ka = fmaf(kh[1][c], wk_s[2][c], ka);
                    ka = fmaf(kh[2][c], wk_s[1][c], ka);
                    ka = fmaf(kc[c], wk_s[0][c], ka);
                    va = vh[0][c] * wv_s[3][c];
                    va = fmaf(vh[1][c], wv_s[2][c], va);
                    va = fmaf(vh[2][c], wv_s[1][c], va);
                    va = fmaf(vc[c], wv_s[0][c], va);
                }
            } else {
                ka = 0.f; va = 0.f;
            }
            ky[c] = bf16_to_f32(f32_to_bf16_rtne((ka + kc[c]) * scale));
            vy[c] = bf16_to_f32(f32_to_bf16_rtne((va + vc[c]) * scale));
            ksq += ky[c] * ky[c];
            vsq += vy[c] * vy[c];
            kh[0][c] = kh[1][c]; kh[1][c] = kh[2][c]; kh[2][c] = kc[c];
            vh[0][c] = vh[1][c]; vh[1][c] = vh[2][c]; vh[2][c] = vc[c];
        }

        #pragma unroll
        for (int off = 16; off > 0; off >>= 1) {
            ksq += __shfl_xor(ksq, off, 32);
            vsq += __shfl_xor(vsq, off, 32);
        }
        float k_norm = (ksq > 0.f) ? (ksq * rsqrtf(ksq)) : 0.f;
        float k_sc = fminf(clip_threshold / (k_norm + clip_eps), 1.0f);
        float v_sc = rsqrtf(vsq * (1.0f / 128.0f) + norm_eps);

        int64_t kv_slot = KvSlots[t];
        if (kv_slot >= 0) {
            int64_t kv_off = (kv_slot / kv_block_size) * stride_kv_blk + (kv_slot % kv_block_size) * stride_kv_n + (int64_t)head * stride_kv_h + d0;
            uint2 ko = make_uint2(pack_bf16x2(ky[0] * k_sc, ky[1] * k_sc), pack_bf16x2(ky[2] * k_sc, ky[3] * k_sc));
            uint2 vo = make_uint2(pack_bf16x2(vy[0] * v_sc, vy[1] * v_sc), pack_bf16x2(vy[2] * v_sc, vy[3] * v_sc));
            *reinterpret_cast<uint2*>(KeyCache + kv_off) = ko;
            *reinterpret_cast<uint2*>(ValCache + kv_off) = vo;
        }
    }
}

// ============================================================================
// 3. ZERO-MATERIALIZATION 4-QUERY ATTN_RES MESSAGE COEFFICIENTS (D=2304, N_STACK=1)
//    Produces Coeffs[4, T, 2] FP32 so loop blend reconstructs exact FP32 messages
//    on the fly from BF16 Bank0/Bank1 without allocating or writing 141 MB FP32!
// ============================================================================
__global__ __launch_bounds__(288, 4)
void hip_attn_res_q4_coeffs_kernel(
    const uint16_t* __restrict__ Bank0,
    const uint16_t* __restrict__ Bank1,
    const float* __restrict__ Queries,
    float* __restrict__ Coeffs,
    int64_t sb0, int64_t sb1, int64_t sq,
    int T, float eps
) {
    const int t = blockIdx.x;
    if (t >= T) return;
    const int tid = threadIdx.x;
    const int c0 = tid * 8;

    uint4 u0 = *reinterpret_cast<const uint4*>(Bank0 + (int64_t)t * sb0 + c0);
    uint4 u1 = *reinterpret_cast<const uint4*>(Bank1 + (int64_t)t * sb1 + c0);
    float b0[8] = {
        bf16_to_f32(u0.x & 0xFFFFu), bf16_to_f32(u0.x >> 16),
        bf16_to_f32(u0.y & 0xFFFFu), bf16_to_f32(u0.y >> 16),
        bf16_to_f32(u0.z & 0xFFFFu), bf16_to_f32(u0.z >> 16),
        bf16_to_f32(u0.w & 0xFFFFu), bf16_to_f32(u0.w >> 16),
    };
    float b1[8] = {
        bf16_to_f32(u1.x & 0xFFFFu), bf16_to_f32(u1.x >> 16),
        bf16_to_f32(u1.y & 0xFFFFu), bf16_to_f32(u1.y >> 16),
        bf16_to_f32(u1.z & 0xFFFFu), bf16_to_f32(u1.z >> 16),
        bf16_to_f32(u1.w & 0xFFFFu), bf16_to_f32(u1.w >> 16),
    };

    float s0 = 0.f, s1 = 0.f;
    float d0[4] = {0.f, 0.f, 0.f, 0.f}, d1[4] = {0.f, 0.f, 0.f, 0.f};
    #pragma unroll
    for (int j = 0; j < 8; ++j) {
        s0 = fmaf(b0[j], b0[j], s0);
        s1 = fmaf(b1[j], b1[j], s1);
    }
    #pragma unroll
    for (int qi = 0; qi < 4; ++qi) {
        float4 qa = *reinterpret_cast<const float4*>(Queries + (int64_t)qi * sq + c0);
        float4 qb = *reinterpret_cast<const float4*>(Queries + (int64_t)qi * sq + c0 + 4);
        d0[qi] = qa.x*b0[0] + qa.y*b0[1] + qa.z*b0[2] + qa.w*b0[3] + qb.x*b0[4] + qb.y*b0[5] + qb.z*b0[6] + qb.w*b0[7];
        d1[qi] = qa.x*b1[0] + qa.y*b1[1] + qa.z*b1[2] + qa.w*b1[3] + qb.x*b1[4] + qb.y*b1[5] + qb.z*b1[6] + qb.w*b1[7];
    }

    #pragma unroll
    for (int off = 16; off > 0; off >>= 1) {
        s0 += __shfl_xor(s0, off, 32);
        s1 += __shfl_xor(s1, off, 32);
        #pragma unroll
        for (int qi = 0; qi < 4; ++qi) {
            d0[qi] += __shfl_xor(d0[qi], off, 32);
            d1[qi] += __shfl_xor(d1[qi], off, 32);
        }
    }

    __shared__ float sm[9][10];
    int warp = tid >> 5, lane = tid & 31;
    if (lane == 0) {
        sm[warp][0] = s0; sm[warp][1] = s1;
        #pragma unroll
        for (int qi = 0; qi < 4; ++qi) {
            sm[warp][2 + qi] = d0[qi];
            sm[warp][6 + qi] = d1[qi];
        }
    }
    __syncthreads();

    if (tid == 0) {
        float ts0 = 0.f, ts1 = 0.f;
        float td0[4] = {0.f, 0.f, 0.f, 0.f}, td1[4] = {0.f, 0.f, 0.f, 0.f};
        #pragma unroll
        for (int w = 0; w < 9; ++w) {
            ts0 += sm[w][0]; ts1 += sm[w][1];
            #pragma unroll
            for (int qi = 0; qi < 4; ++qi) {
                td0[qi] += sm[w][2 + qi];
                td1[qi] += sm[w][6 + qi];
            }
        }
        float r0 = rsqrtf(ts0 * (1.0f / 2304.0f) + eps);
        float r1 = rsqrtf(ts1 * (1.0f / 2304.0f) + eps);
        #pragma unroll
        for (int qi = 0; qi < 4; ++qi) {
            float l0 = td0[qi] * r0;
            float l1 = td1[qi] * r1;
            float m = fmaxf(l0, l1);
            float e0 = expf(l0 - m);
            float e1 = expf(l1 - m);
            float inv = 1.0f / (e0 + e1);
            Coeffs[((int64_t)qi * T + t) * 2 + 0] = (e0 * inv) * r0;
            Coeffs[((int64_t)qi * T + t) * 2 + 1] = (e1 * inv) * r1;
        }
    }
}

extern "C" {

void launch_pair_m64(
    const void* Expert, const void* Padded, void* PairIdx, void* TailIdx, void* Counts,
    int max_blocks, hipStream_t stream
) {
    hipLaunchKernelGGL(
        pair_m64_blocks_kernel, dim3(1), dim3(32), 0, stream,
        (const int32_t*)Expert, (const int32_t*)Padded,
        (int32_t*)PairIdx, (int32_t*)TailIdx, (int32_t*)Counts, max_blocks);
}

#define DEF_PAIRED_LAUNCH(NAME, MODE, FUSED, TDIV, XBF16) \
void NAME( \
    const void* X, const void* Packed, const void* Raw, const void* Outer, \
    const void* RowScale, const void* TableU32, const void* Kinds, \
    const void* Sorted, const void* Expert, \
    const void* PairIdx, const void* TailIdx, const void* Counts, \
    void* Y, const void* Gamma, \
    int N, int K, int ROUTES, int max_m64_blocks, float gate_cap, float up_cap, hipStream_t stream \
) { \
    int max_pairs = max_m64_blocks / 2 + 1; \
    int max_tails = min(max_m64_blocks, 128); \
    dim3 grid_p(((max_pairs + 3) / 4) * 4 * (N / 128), 1); \
    hipLaunchKernelGGL( \
        (moe_wmma_k64_unified_kernel<128, 2, 4, FUSED, TDIV, MODE, XBF16>), grid_p, dim3(256), 0, stream, \
        (const uint16_t*)X, (const uint32_t*)Packed, (const uint32_t*)Raw, (const float*)Outer, \
        (const uint16_t*)RowScale, (const uint32_t*)TableU32, (const int32_t*)Kinds, \
        (const int32_t*)Sorted, (const int32_t*)Expert, (const int32_t*)PairIdx, (const int32_t*)Counts, \
        (uint16_t*)Y, (const float*)Gamma, N, K, ROUTES, gate_cap, up_cap); \
    dim3 grid_t(((max_tails + 3) / 4) * 4 * (N / 128), 1); \
    hipLaunchKernelGGL( \
        (moe_wmma_k64_unified_kernel<64, 1, 4, FUSED, TDIV, MODE, XBF16>), grid_t, dim3(128), 0, stream, \
        (const uint16_t*)X, (const uint32_t*)Packed, (const uint32_t*)Raw, (const float*)Outer, \
        (const uint16_t*)RowScale, (const uint32_t*)TableU32, (const int32_t*)Kinds, \
        (const int32_t*)Sorted, (const int32_t*)Expert, (const int32_t*)TailIdx, ((const int32_t*)Counts) + 1, \
        (uint16_t*)Y, (const float*)Gamma, N, K, ROUTES, gate_cap, up_cap); \
}

// BF16-direct X input variants for w13 (zero x.to(float16) overhead!)
DEF_PAIRED_LAUNCH(launch_source_w13_f16_xbf16,   0, 2, 8, 1)
DEF_PAIRED_LAUNCH(launch_source_w13_bf16_xbf16,  0, 1, 8, 1)
DEF_PAIRED_LAUNCH(launch_mixed1_w13_f16_xbf16,  10, 2, 8, 1)
DEF_PAIRED_LAUNCH(launch_mixed1_w13_bf16_xbf16, 10, 1, 8, 1)
DEF_PAIRED_LAUNCH(launch_mixed2_w13_f16_xbf16,  20, 2, 8, 1)
DEF_PAIRED_LAUNCH(launch_mixed2_w13_bf16_xbf16, 20, 1, 8, 1)
DEF_PAIRED_LAUNCH(launch_coda_w13_f16_xbf16,     2, 2, 8, 1)
DEF_PAIRED_LAUNCH(launch_coda_w13_bf16_xbf16,    2, 1, 8, 1)

// FP16 X input variants (w13 if X already FP16, and w2 consuming w13's FP16 output!)
DEF_PAIRED_LAUNCH(launch_source_w13_f16,   0, 2, 8, 0)
DEF_PAIRED_LAUNCH(launch_source_w13_bf16,  0, 1, 8, 0)
DEF_PAIRED_LAUNCH(launch_source_w2,        0, 0, 1, 0)
DEF_PAIRED_LAUNCH(launch_mixed1_w13_f16,  10, 2, 8, 0)
DEF_PAIRED_LAUNCH(launch_mixed1_w13_bf16, 10, 1, 8, 0)
DEF_PAIRED_LAUNCH(launch_mixed1_w2,       10, 0, 1, 0)
DEF_PAIRED_LAUNCH(launch_mixed2_w13_f16,  20, 2, 8, 0)
DEF_PAIRED_LAUNCH(launch_mixed2_w13_bf16, 20, 1, 8, 0)
DEF_PAIRED_LAUNCH(launch_mixed2_w2,       20, 0, 1, 0)
DEF_PAIRED_LAUNCH(launch_coda_w13_f16,     2, 2, 8, 0)
DEF_PAIRED_LAUNCH(launch_coda_w13_bf16,    2, 1, 8, 0)
DEF_PAIRED_LAUNCH(launch_coda_w2,          2, 0, 1, 0)

void launch_hip_kv_conv_1w(
    const void* K, const void* V, const void* Wk, const void* Wv, const void* Dk, const void* Dv,
    void* CC, void* KC, void* VC, const void* KVS, const void* Pos, const void* Seq, const void* CS,
    const void* BT_ptr, const void* QS,
    int T, int KVH, int64_t sk, int64_t sv, int64_t sw0, int64_t sw1, int64_t sdk, int64_t sdv,
    int64_t scb, int64_t sch, int64_t scn, int64_t skvb, int64_t skvn, int64_t skvh,
    int kv_bs, int64_t sbt, int mb, int nc, float scale, float ct, float ce, float ne,
    int dynamic, hipStream_t stream
) {
    constexpr int BT = 64;
    int n_tiles = (T + BT - 1) / BT;
    dim3 grid((n_tiles + 3) / 4, KVH);
    dim3 block(128);
    if (dynamic) {
        hipLaunchKernelGGL((hip_kv_conv_1warp_kernel<BT, 1>), grid, block, 0, stream,
            (const uint16_t*)K, (const uint16_t*)V, (const uint16_t*)Wk, (const uint16_t*)Wv,
            (const uint16_t*)Dk, (const uint16_t*)Dv, (uint16_t*)CC, (uint16_t*)KC, (uint16_t*)VC,
            (const int64_t*)KVS, (const int32_t*)Pos, (const int32_t*)Seq, (const int64_t*)CS,
            (const int32_t*)BT_ptr, (const int32_t*)QS, T, KVH, sk, sv, sw0, sw1, sdk, sdv,
            scb, sch, scn, skvb, skvn, skvh, kv_bs, sbt, mb, nc, scale, ct, ce, ne);
    } else {
        hipLaunchKernelGGL((hip_kv_conv_1warp_kernel<BT, 0>), grid, block, 0, stream,
            (const uint16_t*)K, (const uint16_t*)V, (const uint16_t*)Wk, (const uint16_t*)Wv,
            (const uint16_t*)Dk, (const uint16_t*)Dv, (uint16_t*)CC, (uint16_t*)KC, (uint16_t*)VC,
            (const int64_t*)KVS, (const int32_t*)Pos, (const int32_t*)Seq, (const int64_t*)CS,
            (const int32_t*)BT_ptr, (const int32_t*)QS, T, KVH, sk, sv, sw0, sw1, sdk, sdv,
            scb, sch, scn, skvb, skvn, skvh, kv_bs, sbt, mb, nc, scale, ct, ce, ne);
    }
}

void launch_q4_coeffs(
    const void* B0, const void* B1, const void* Q, void* C,
    int64_t sb0, int64_t sb1, int64_t sq, int T, float eps, hipStream_t stream
) {
    hipLaunchKernelGGL(
        hip_attn_res_q4_coeffs_kernel, dim3(T), dim3(288), 0, stream,
        (const uint16_t*)B0, (const uint16_t*)B1, (const float*)Q, (float*)C,
        sb0, sb1, sq, T, eps);
}

}
