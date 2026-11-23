// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 CIRU AI.
// Native RDNA 3.5 (gfx1151) single-pass-per-expert M16/M32 decode MoE kernels.
//   - 1-warp greedy adjacent M16->M32 block planner (0 host sync, HIP-graph safe).
//   - Zero-smem_B register-unpack K64 software-pipelined WMMA across all 4 bank modes
//     (source NVFP4, mixed1 INT4, mixed2 C1, coda C1), reusing unpacked weights across
//     both M16 halves of paired M32 blocks.

#include "hip_common.cuh"
#include <type_traits>

// ============================================================================
// 1. M16 -> M32 GREEDY ADJACENT-BLOCK PLANNER (1 WARP, 0 HOST SYNC)
// ============================================================================
__global__ __launch_bounds__(32, 1)
void hip_moe_decode_plan_kernel(
    const int32_t* __restrict__ Experts,
    const int32_t* __restrict__ Padded,
    uint32_t*      __restrict__ UniDesc,
    int32_t*       __restrict__ Counts,
    int max_m16_blocks
) {
    __shared__ int32_t s_exp[512];
    int padded = *Padded;
    int n_blocks = (padded > 0) ? min(padded >> 4, max_m16_blocks) : 0;
    if (n_blocks <= 0) {
        if (threadIdx.x == 0) {
            Counts[0] = 0;
            Counts[1] = 0;
            Counts[2] = 0;
        }
        return;
    }

    if (n_blocks <= 512) {
        for (int i = (int)threadIdx.x; i < n_blocks; i += 32) {
            s_exp[i] = Experts[i];
        }
        __syncthreads();
        if (threadIdx.x != 0) return;
        int np = 0, nt = 0, nu = 0, i = 0;
        while (i < n_blocks) {
            int e = s_exp[i];
            if (e < 0) { i += 1; continue; }
            if (i + 1 < n_blocks && s_exp[i + 1] == e) {
                UniDesc[nu++] = ((uint32_t)i << 1) | 1u;
                np += 1;
                i += 2;
            } else {
                UniDesc[nu++] = ((uint32_t)i << 1) | 0u;
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
            int e = Experts[i];
            if (e < 0) { i += 1; continue; }
            if (i + 1 < n_blocks && Experts[i + 1] == e) {
                UniDesc[nu++] = ((uint32_t)i << 1) | 1u;
                np += 1;
                i += 2;
            } else {
                UniDesc[nu++] = ((uint32_t)i << 1) | 0u;
                nt += 1;
                i += 1;
            }
        }
        Counts[0] = np;
        Counts[1] = nt;
        Counts[2] = nu;
    }
}

__device__ __forceinline__ half16 moe_dec_load_h16_u4(const uint16_t* ptr) {
    uint4 lo = *reinterpret_cast<const uint4*>(ptr);
    uint4 hi = *reinterpret_cast<const uint4*>(ptr + 8);
    half16 out;
    uint32_t words[8] = {lo.x, lo.y, lo.z, lo.w, hi.x, hi.y, hi.z, hi.w};
    __builtin_memcpy(&out, words, 32);
    return out;
}

// ============================================================================
// 2. SINGLE-PASS M16/M32 DECODE MoE KERNEL (ZERO smem_B, SOFTWARE-PIPELINED)
//    BANK_MODE: 0=source NVFP4, 2=coda C1, 10=mixed(0/1 INT4), 20=mixed(0/2 C1)
//    FUSED_ACT: 0=w2 (BF16), 1=w13 fused softcap SwiGLU
//    TOKENS_DIV: 8 for w13 (X is [M, K]), 1 for w2 (X is [M*8, K])
//    USE_F16_X: 1 if X is FP16, 0 if X is BF16 (converted on smem_A store)
//    OUT_F16: 1 if w13 outputs FP16, 0 if BF16
// ============================================================================
template <int BANK_MODE, int FUSED_ACT, int TOKENS_DIV, int USE_F16_X, int OUT_F16>
__global__ __launch_bounds__(128, 6)
void hip_moe_decode_unified_kernel(
    const uint16_t* __restrict__ X,
    const uint32_t* __restrict__ Packed,
    const uint32_t* __restrict__ Raw,
    const float*    __restrict__ Outer,
    const int32_t*  __restrict__ Kinds,
    const uint16_t* __restrict__ RowScale,
    const uint32_t* __restrict__ TableU32,
    const int32_t*  __restrict__ Sorted,
    const int32_t*  __restrict__ Experts,
    const uint32_t* __restrict__ UniDesc,
    const int32_t*  __restrict__ Counts,
    const float*    __restrict__ Gamma,
    uint16_t*       __restrict__ Out,
    int ROUTES, int N, int K, float gate_cap, float up_cap
) {
    const int n_tiles = N >> 7;
    const int pid     = (int)blockIdx.x;
    const int m_ord   = pid / n_tiles;
    const int n_blk   = pid % n_tiles;
    if (m_ord >= Counts[2]) return;

    const uint32_t raw_desc = UniDesc[m_ord];
    const int m16_idx  = (int)(raw_desc >> 1);
    const bool is_pair = ((raw_desc & 1u) != 0u);
    const int expert   = Experts[m16_idx];
    if (expert < 0) return;

    int kind = BANK_MODE;
    if constexpr (BANK_MODE == 10 || BANK_MODE == 20) {
        kind = Kinds[expert];
    }

    const int tid    = (int)threadIdx.x;
    const int warp   = tid >> 5;
    const int lane   = tid & 31;
    const int lane16 = lane & 15;
    const int half   = lane >> 4;

    __shared__ uint16_t smem_A[32][72];
    __shared__ int32_t  smem_route[32];

    const int active_rows = is_pair ? 32 : 16;
    const int sorted_base = m16_idx * 16;
    if (tid < 32) {
        smem_route[tid] = (tid < active_rows) ? Sorted[sorted_base + tid] : -1;
    }
    __syncthreads();

    // Cooperative smem_A loader coordinates (128 thr -> 16 rows x 8 uint4 cols = 1024 f16s)
    const int tx_row = tid >> 3;
    const int tx_col = (tid & 7) * 8;
    const int r0 = smem_route[tx_row];
    const bool v0 = (r0 >= 0 && r0 < ROUTES);
    const uint16_t* a_ptr0 = v0 ? (X + (int64_t)(r0 / TOKENS_DIV) * K + tx_col) : nullptr;
    int r1 = -1;
    bool v1 = false;
    const uint16_t* a_ptr1 = nullptr;
    if (is_pair) {
        r1 = smem_route[tx_row + 16];
        v1 = (r1 >= 0 && r1 < ROUTES);
        a_ptr1 = v1 ? (X + (int64_t)(r1 / TOKENS_DIV) * K + tx_col) : nullptr;
    }

    // Each warp owns 2 consecutive 16-row N-tiles inside the 128-row bank tile
    const int row0_in_128 = warp * 32 + lane16;
    const int row1_in_128 = row0_in_128 + 16;
    const int col0 = n_blk * 128 + row0_in_128;
    const int col1 = col0 + 16;

    const int n_kt64 = K >> 6;
    const int64_t tile0 = ((int64_t)expert * n_tiles + n_blk) * n_kt64;
    const uint32_t* p0_base = Packed + (tile0 * 128 + row0_in_128) * 8 + half * 4;
    const uint32_t* p1_base = Packed + (tile0 * 128 + row1_in_128) * 8 + half * 4;
    const uint32_t* r0_base = Raw    + (tile0 * 128 + row0_in_128);
    const uint32_t* r1_base = Raw    + (tile0 * 128 + row1_in_128);

    float rs0 = 1.0f, rs1 = 1.0f;
    const uint32_t* tab_e = nullptr;
    if constexpr (BANK_MODE == 2 || BANK_MODE == 20) {
        if (kind == 2) {
            rs0 = f16_bits_to_f32(RowScale[(int64_t)expert * N + col0]);
            rs1 = f16_bits_to_f32(RowScale[(int64_t)expert * N + col1]);
            tab_e = TableU32 + (int64_t)expert * 256;
        }
    }

    // Software-pipeline prologue: issue kt=0 weight loads immediately
    uint4    p0_nxt = *reinterpret_cast<const uint4*>(p0_base);
    uint4    p1_nxt = *reinterpret_cast<const uint4*>(p1_base);
    uint32_t r0_nxt = r0_base[0];
    uint32_t r1_nxt = r1_base[0];

    float8 acc0_m0 = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
    float8 acc1_m0 = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
    float8 acc0_m1 = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
    float8 acc1_m1 = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};

    auto unpack_pair_f16 = [&](uint32_t pw0, uint32_t pw1, uint32_t raw_w, int kk_lo, float rs, half16 &b_lo, half16 &b_hi) {
        const int kk4 = (half == 0) ? kk_lo : (kk_lo + 2);
        uint4 o0, o1;
        if constexpr (BANK_MODE == 0) {
            uint32_t sb = (raw_w >> (kk4 * 8)) & 0xFFu;
            unpack_nvfp4_16_u4(pw0, pw1, sb, o0, o1);
        } else if constexpr (BANK_MODE == 2) {
            uint32_t tag = (raw_w >> (kk4 * 8)) & 0xFFu;
            uint32_t ent = tab_e[tag];
            float sv = f16_bits_to_f32((uint16_t)(ent & 0xFFFFu)) * rs;
            float ov = f16_bits_to_f32((uint16_t)(ent >> 16)) * rs;
            unpack_c1_16_u4(pw0, pw1, sv, ov, o0, o1);
        } else if constexpr (BANK_MODE == 10) {
            if (kind == 0) {
                uint32_t sb = (raw_w >> (kk4 * 8)) & 0xFFu;
                unpack_nvfp4_16_u4(pw0, pw1, sb, o0, o1);
            } else {
                uint16_t sb16 = (half == 0) ? (uint16_t)(raw_w & 0xFFFFu) : (uint16_t)(raw_w >> 16);
                unpack_int4_16_u4(pw0, pw1, bf16_to_f32(sb16), o0, o1);
            }
        } else {
            if (kind == 0) {
                uint32_t sb = (raw_w >> (kk4 * 8)) & 0xFFu;
                unpack_nvfp4_16_u4(pw0, pw1, sb, o0, o1);
            } else {
                uint32_t tag = (raw_w >> (kk4 * 8)) & 0xFFu;
                uint32_t ent = tab_e[tag];
                float sv = f16_bits_to_f32((uint16_t)(ent & 0xFFFFu)) * rs;
                float ov = f16_bits_to_f32((uint16_t)(ent >> 16)) * rs;
                unpack_c1_16_u4(pw0, pw1, sv, ov, o0, o1);
            }
        }
        uint32_t w[8] = {o0.x, o0.y, o0.z, o0.w, o1.x, o1.y, o1.z, o1.w};
        uint32_t o[8];
        #pragma unroll
        for (int i = 0; i < 8; ++i) o[i] = __shfl_xor(w[i], 16, 32);
        __builtin_memcpy(&b_lo, (half == 0) ? w : o, 32);
        __builtin_memcpy(&b_hi, (half == 0) ? o : w, 32);
    };

    for (int kt = 0; kt < n_kt64; ++kt) {
        const int k_off = kt << 6;
        // Stage A tile(s) into LDS
        uint4 xv0 = v0 ? *reinterpret_cast<const uint4*>(a_ptr0 + k_off) : make_uint4(0, 0, 0, 0);
        if constexpr (!USE_F16_X) {
            if (v0) xv0 = bf16x8_to_f16x8(xv0);
        }
        *reinterpret_cast<uint4*>(&smem_A[tx_row][tx_col]) = xv0;
        if (is_pair) {
            uint4 xv1 = v1 ? *reinterpret_cast<const uint4*>(a_ptr1 + k_off) : make_uint4(0, 0, 0, 0);
            if constexpr (!USE_F16_X) {
                if (v1) xv1 = bf16x8_to_f16x8(xv1);
            }
            *reinterpret_cast<uint4*>(&smem_A[tx_row + 16][tx_col]) = xv1;
        }
        __syncthreads();

        // Latch current weights and immediately prefetch kt+1
        uint4    p0 = p0_nxt;
        uint4    p1 = p1_nxt;
        uint32_t r0w = r0_nxt;
        uint32_t r1w = r1_nxt;
        if (kt + 1 < n_kt64) {
            const int64_t next_row_off = (int64_t)(kt + 1) * 128;
            p0_nxt = *reinterpret_cast<const uint4*>(p0_base + next_row_off * 8);
            p1_nxt = *reinterpret_cast<const uint4*>(p1_base + next_row_off * 8);
            r0_nxt = r0_base[next_row_off];
            r1_nxt = r1_base[next_row_off];
        }

        // Pass A: ks = 0 & ks = 2
        {
            half16 a0_m0 = moe_dec_load_h16_u4(&smem_A[lane16][0]);
            half16 a2_m0 = moe_dec_load_h16_u4(&smem_A[lane16][32]);
            half16 b0, b2;
            unpack_pair_f16(p0.x, p0.y, r0w, 0, rs0, b0, b2);
            acc0_m0 = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a0_m0, b0, acc0_m0);
            acc0_m0 = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a2_m0, b2, acc0_m0);
            if (is_pair) {
                half16 a0_m1 = moe_dec_load_h16_u4(&smem_A[16 + lane16][0]);
                half16 a2_m1 = moe_dec_load_h16_u4(&smem_A[16 + lane16][32]);
                acc0_m1 = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a0_m1, b0, acc0_m1);
                acc0_m1 = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a2_m1, b2, acc0_m1);
                unpack_pair_f16(p1.x, p1.y, r1w, 0, rs1, b0, b2);
                acc1_m0 = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a0_m0, b0, acc1_m0);
                acc1_m0 = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a2_m0, b2, acc1_m0);
                acc1_m1 = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a0_m1, b0, acc1_m1);
                acc1_m1 = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a2_m1, b2, acc1_m1);
            } else {
                unpack_pair_f16(p1.x, p1.y, r1w, 0, rs1, b0, b2);
                acc1_m0 = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a0_m0, b0, acc1_m0);
                acc1_m0 = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a2_m0, b2, acc1_m0);
            }
        }
        // Pass B: ks = 1 & ks = 3
        {
            half16 a1_m0 = moe_dec_load_h16_u4(&smem_A[lane16][16]);
            half16 a3_m0 = moe_dec_load_h16_u4(&smem_A[lane16][48]);
            half16 b1, b3;
            unpack_pair_f16(p0.z, p0.w, r0w, 1, rs0, b1, b3);
            acc0_m0 = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a1_m0, b1, acc0_m0);
            acc0_m0 = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a3_m0, b3, acc0_m0);
            if (is_pair) {
                half16 a1_m1 = moe_dec_load_h16_u4(&smem_A[16 + lane16][16]);
                half16 a3_m1 = moe_dec_load_h16_u4(&smem_A[16 + lane16][48]);
                acc0_m1 = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a1_m1, b1, acc0_m1);
                acc0_m1 = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a3_m1, b3, acc0_m1);
                unpack_pair_f16(p1.z, p1.w, r1w, 1, rs1, b1, b3);
                acc1_m0 = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a1_m0, b1, acc1_m0);
                acc1_m0 = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a3_m0, b3, acc1_m0);
                acc1_m1 = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a1_m1, b1, acc1_m1);
                acc1_m1 = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a3_m1, b3, acc1_m1);
            } else {
                unpack_pair_f16(p1.z, p1.w, r1w, 1, rs1, b1, b3);
                acc1_m0 = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a1_m0, b1, acc1_m0);
                acc1_m0 = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a3_m0, b3, acc1_m0);
            }
        }
        __syncthreads();
    }

    const float outer = (kind == 0) ? Outer[expert] : 1.0f;
    const float inv_g = FUSED_ACT ? (1.0f / gate_cap) : 0.0f;
    const float inv_u = FUSED_ACT ? (1.0f / up_cap) : 0.0f;

    auto write_tile = [&](int m_base, const float8 &a0, const float8 &a1) {
        #pragma unroll
        for (int i = 0; i < 8; ++i) {
            int r = smem_route[m_base + half + 2 * i];
            if (r < 0 || r >= ROUTES) continue;
            if constexpr (FUSED_ACT) {
                float gam = Gamma[r];
                auto emit_fused = [&](float raw_v, int col_base) {
                    float v = bf16_round_f32(raw_v * outer);
                    float up_v = __shfl_xor(v, 1, 32);
                    uint16_t h_val = 0;
                    if ((lane16 & 1) == 0) {
                        float mid = gate_cap * fast_tanh_hw(v * inv_g) * fast_sigmoid_hw(v)
                                  * up_cap * fast_tanh_hw(up_v * inv_u) * gam;
                        if constexpr (OUT_F16) {
                            _Float16 h = (_Float16)bf16_round_f32(mid);
                            h_val = *reinterpret_cast<uint16_t*>(&h);
                        } else {
                            h_val = f32_to_bf16_rtne(mid);
                        }
                    }
                    uint32_t next_h = __shfl_down((uint32_t)h_val, 2, 16);
                    if ((lane16 & 3) == 0) {
                        uint32_t pk = (uint32_t)h_val | (next_h << 16);
                        int col_out = (col_base + lane16) >> 1;
                        *reinterpret_cast<uint32_t*>(Out + (int64_t)r * (N >> 1) + col_out) = pk;
                    }
                };
                emit_fused(a0[i], n_blk * 128 + warp * 32);
                emit_fused(a1[i], n_blk * 128 + warp * 32 + 16);
            } else {
                auto emit_w2 = [&](float raw_v, int col_base) {
                    uint16_t h0 = f32_to_bf16_rtne(raw_v * outer);
                    uint32_t h1 = __shfl_down((uint32_t)h0, 1, 16);
                    if ((lane16 & 1) == 0) {
                        uint32_t pk = (uint32_t)h0 | (h1 << 16);
                        *reinterpret_cast<uint32_t*>(Out + (int64_t)r * N + col_base + lane16) = pk;
                    }
                };
                emit_w2(a0[i], n_blk * 128 + warp * 32);
                emit_w2(a1[i], n_blk * 128 + warp * 32 + 16);
            }
        }
    };

    write_tile(0, acc0_m0, acc1_m0);
    if (is_pair) {
        write_tile(16, acc0_m1, acc1_m1);
    }
}

extern "C" {

int launch_hip_moe_decode_plan(
    const int32_t* Experts, const int32_t* Padded,
    uint32_t* UniDesc, int32_t* Counts,
    int max_m16_blocks, void* stream_ptr
) {
    if (max_m16_blocks <= 0) return -1;
    hipStream_t stream = (hipStream_t)stream_ptr;
    hipLaunchKernelGGL(
        hip_moe_decode_plan_kernel, dim3(1), dim3(32), 0, stream,
        Experts, Padded, UniDesc, Counts, max_m16_blocks);
    return 0;
}

int launch_hip_moe_decode_unified(
    int bank_mode, int fused_act, int tokens_div, int use_f16_x, int out_f16,
    const void* X, const void* W, const void* Scales, const void* Outer,
    const void* Kinds, const void* RowScales, const void* Table,
    const int32_t* Sorted, const int32_t* Experts,
    const uint32_t* UniDesc, const int32_t* Counts,
    const void* Gamma, void* Out,
    int routes, int N, int K, int max_uni_blocks,
    float gate_cap, float up_cap, void* stream_ptr
) {
    if (routes <= 0 || N <= 0 || K <= 0 || (N & 127) != 0 || (K & 63) != 0 || max_uni_blocks <= 0) return -1;
    if (bank_mode != 0 && bank_mode != 2 && bank_mode != 10 && bank_mode != 20) return -1;
    if (tokens_div != 1 && tokens_div != 8) return -1;
    hipStream_t stream = (hipStream_t)stream_ptr;
    dim3 grid((unsigned int)(max_uni_blocks * (N >> 7)));
    dim3 block(128);

#define LAUNCH_MOE_DEC(BM, FA, TD, XF16, OF16) \
    hipLaunchKernelGGL((hip_moe_decode_unified_kernel<BM, FA, TD, XF16, OF16>), grid, block, 0, stream, \
        (const uint16_t*)X, (const uint32_t*)W, (const uint32_t*)Scales, (const float*)Outer, \
        (const int32_t*)Kinds, (const uint16_t*)RowScales, (const uint32_t*)Table, \
        Sorted, Experts, UniDesc, Counts, (const float*)Gamma, (uint16_t*)Out, \
        routes, N, K, gate_cap, up_cap)

#define DISP_MODE(BM) do { \
    if (fused_act) { \
        if (out_f16) { \
            if (use_f16_x) LAUNCH_MOE_DEC(BM, 1, 8, 1, 1); \
            else           LAUNCH_MOE_DEC(BM, 1, 8, 0, 1); \
        } else { \
            if (use_f16_x) LAUNCH_MOE_DEC(BM, 1, 8, 1, 0); \
            else           LAUNCH_MOE_DEC(BM, 1, 8, 0, 0); \
        } \
    } else { \
        if (use_f16_x) LAUNCH_MOE_DEC(BM, 0, 1, 1, 0); \
        else           LAUNCH_MOE_DEC(BM, 0, 1, 0, 0); \
    } \
} while (0)

    if (bank_mode == 0)       DISP_MODE(0);
    else if (bank_mode == 2)  DISP_MODE(2);
    else if (bank_mode == 10) DISP_MODE(10);
    else                      DISP_MODE(20);

#undef DISP_MODE
#undef LAUNCH_MOE_DEC
    return 0;
}

}  // extern "C"
