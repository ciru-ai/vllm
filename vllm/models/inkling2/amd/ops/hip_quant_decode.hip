// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 CIRU AI.
// Native RDNA 3.5 (gfx1151) W6 & W4 quantized decode linear & Markov argmax kernels.
//   - N_TILE=64 (4 warps x 16 N-rows/warp) with software-pipelined 16B P4 + 8B P2 loads.
//   - M_TILE=16 uses 0-LDS, 0-barrier independent warps; M_TILE=32/48 stages X once per CTA.
//   - Full-MLP prefetched W6 M=1 GEMV (v_dot2_f32_bf16) & 8-lane/row W4 Markov fused argmax.

#include "hip_common.cuh"
#include <type_traits>

__device__ __forceinline__ short16 qdec_load_short16_u4(const uint16_t* ptr) {
    uint4 lo = *reinterpret_cast<const uint4*>(ptr);
    uint4 hi = *reinterpret_cast<const uint4*>(ptr + 8);
    short16 out;
    uint32_t words[8] = {lo.x, lo.y, lo.z, lo.w, hi.x, hi.y, hi.z, hi.w};
    __builtin_memcpy(&out, words, 32);
    return out;
}

// ============================================================================
// 1. SINGLE-PASS W6 WMMA LINEAR (N_TILE=64, M_TILE=16/32/48, SOFTWARE-PIPELINED)
// ============================================================================
template <bool IS_QKV, int M_TILE, int SPLITS, bool OUT_FP32, bool HAS_SOFTCAP>
__global__ __launch_bounds__(128, 6)
void hip_w6_wmma_kernel(
    const uint16_t* __restrict__ XQ,
    const uint16_t* __restrict__ XK,
    const uint16_t* __restrict__ XV,
    const uint8_t*  __restrict__ P4,
    const uint32_t* __restrict__ P2,
    const uint16_t* __restrict__ Scales,
    void*           __restrict__ Out,
    int M, int N_pad, int K, int q_tiles, int k_tiles_end, float soft_cap
) {
    constexpr int M_TILES = M_TILE / 16;
    constexpr bool USE_SMEM = true;
    const int tile_n  = (int)blockIdx.x;   // 64 output rows per CTA (2 CTAs per 128-row tile)
    const int tile_m  = (int)blockIdx.y;
    const int split   = (int)blockIdx.z;

    const int tile128 = tile_n >> 1;
    const int n_half  = tile_n & 1;

    const uint16_t* X = XQ;
    if constexpr (IS_QKV) {
        X = (tile128 < q_tiles) ? XQ : ((tile128 < k_tiles_end) ? XK : XV);
    }

    const int tid    = (int)threadIdx.x;
    const int warp   = tid >> 5;
    const int lane   = tid & 31;
    const int lane16 = lane & 15;
    const int half   = lane >> 4;

    const int row_in_128 = n_half * 64 + warp * 16 + lane16;
    const int out_col    = tile_n * 64 + warp * 16 + lane16;

    const int kt_total  = K >> 6;
    const int kt_per_sp = kt_total / SPLITS;
    const int kt_start  = split * kt_per_sp;
    const int kt_end    = kt_start + kt_per_sp;

    __shared__ uint16_t smem_X[USE_SMEM ? M_TILE : 1][USE_SMEM ? 72 : 1];

    float8 acc[M_TILES];
    #pragma unroll
    for (int mt = 0; mt < M_TILES; ++mt) {
        acc[mt] = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
    }

    const int tx_row = tid >> 3;
    const int tx_col = tid & 7;
    const int mr0 = tile_m * M_TILE + lane16;
    const bool mr0_valid = (mr0 < M);
    const uint16_t* x0_row_ptr = X + (int64_t)(mr0_valid ? mr0 : 0) * K;

    const int64_t base_tile_tidx = (int64_t)tile128 * kt_total * 128 + row_in_128;
    const uint8_t*  p4_base = P4     + base_tile_tidx * 32 + half * 16;
    const uint32_t* p2_base = P2     + base_tile_tidx * 4  + half * 2;
    const uint16_t* sc_base = Scales + base_tile_tidx;

    // Software pipeline prologue: issue kt_start weight loads immediately
    uint4    p4_cur = *reinterpret_cast<const uint4*>(p4_base + (int64_t)kt_start * (128 * 32));
    uint2    p2_cur = *reinterpret_cast<const uint2*>(p2_base + (int64_t)kt_start * (128 * 4));
    uint16_t sc_cur = sc_base[(int64_t)kt_start * 128];

    auto unpack_pair_ks = [&](uint32_t px, uint32_t py, uint32_t p2w, short16 &w_lo, short16 &w_hi) {
        uint32_t w[8], o[8];
        w[0] = unpack_w6_pair_bf16((px      ) & 0xFFu, (p2w      ) & 15u);
        w[1] = unpack_w6_pair_bf16((px >>  8) & 0xFFu, (p2w >>  4) & 15u);
        w[2] = unpack_w6_pair_bf16((px >> 16) & 0xFFu, (p2w >>  8) & 15u);
        w[3] = unpack_w6_pair_bf16((px >> 24) & 0xFFu, (p2w >> 12) & 15u);
        w[4] = unpack_w6_pair_bf16((py      ) & 0xFFu, (p2w >> 16) & 15u);
        w[5] = unpack_w6_pair_bf16((py >>  8) & 0xFFu, (p2w >> 20) & 15u);
        w[6] = unpack_w6_pair_bf16((py >> 16) & 0xFFu, (p2w >> 24) & 15u);
        w[7] = unpack_w6_pair_bf16((py >> 24) & 0xFFu, (p2w >> 28) & 15u);
        #pragma unroll
        for (int i = 0; i < 8; ++i) o[i] = __shfl_xor(w[i], 16, 32);
        __builtin_memcpy(&w_lo, (half == 0) ? w : o, 32);
        __builtin_memcpy(&w_hi, (half == 0) ? o : w, 32);
    };

    for (int kt = kt_start; kt < kt_end; ++kt) {
        const int k_base = kt << 6;
        if constexpr (USE_SMEM) {
            #pragma unroll
            for (int mt = 0; mt < M_TILES; ++mt) {
                const int mr = tile_m * M_TILE + mt * 16 + tx_row;
                uint4 xv = (mr < M)
                    ? *reinterpret_cast<const uint4*>(X + (int64_t)mr * K + k_base + tx_col * 8)
                    : make_uint4(0, 0, 0, 0);
                *reinterpret_cast<uint4*>(&smem_X[mt * 16 + tx_row][tx_col * 8]) = xv;
            }
            __syncthreads();
        }

        // Latch current weights and immediately prefetch kt+1 to overlap HBM latency with unpack+WMMA
        uint4    p4 = p4_cur;
        uint2    p2 = p2_cur;
        float    sc = f16_bits_to_f32(sc_cur);
        if (kt + 1 < kt_end) {
            p4_cur = *reinterpret_cast<const uint4*>(p4_base + (int64_t)(kt + 1) * (128 * 32));
            p2_cur = *reinterpret_cast<const uint2*>(p2_base + (int64_t)(kt + 1) * (128 * 4));
            sc_cur = sc_base[(int64_t)(kt + 1) * 128];
        }

        float8 blk[M_TILES];
        #pragma unroll
        for (int mt = 0; mt < M_TILES; ++mt) {
            blk[mt] = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
        }

        // Pass A: ks = 0 & ks = 2
        {
            short16 w0, w2;
            unpack_pair_ks(p4.x, p4.y, p2.x, w0, w2);
            #pragma unroll
            for (int mt = 0; mt < M_TILES; ++mt) {
                short16 x0, x2;
                if constexpr (USE_SMEM) {
                    x0 = qdec_load_short16_u4(&smem_X[mt * 16 + lane16][0]);
                    x2 = qdec_load_short16_u4(&smem_X[mt * 16 + lane16][32]);
                } else {
                    x0 = mr0_valid ? qdec_load_short16_u4(x0_row_ptr + k_base +  0) : short16{};
                    x2 = mr0_valid ? qdec_load_short16_u4(x0_row_ptr + k_base + 32) : short16{};
                }
                blk[mt] = __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32(x0, w0, blk[mt]);
                blk[mt] = __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32(x2, w2, blk[mt]);
            }
        }
        // Pass B: ks = 1 & ks = 3
        {
            short16 w1, w3;
            unpack_pair_ks(p4.z, p4.w, p2.y, w1, w3);
            #pragma unroll
            for (int mt = 0; mt < M_TILES; ++mt) {
                short16 x1, x3;
                if constexpr (USE_SMEM) {
                    x1 = qdec_load_short16_u4(&smem_X[mt * 16 + lane16][16]);
                    x3 = qdec_load_short16_u4(&smem_X[mt * 16 + lane16][48]);
                } else {
                    x1 = mr0_valid ? qdec_load_short16_u4(x0_row_ptr + k_base + 16) : short16{};
                    x3 = mr0_valid ? qdec_load_short16_u4(x0_row_ptr + k_base + 48) : short16{};
                }
                blk[mt] = __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32(x1, w1, blk[mt]);
                blk[mt] = __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32(x3, w3, blk[mt]);
            }
        }

        #pragma unroll
        for (int mt = 0; mt < M_TILES; ++mt) {
            #pragma unroll
            for (int r = 0; r < 8; ++r) {
                acc[mt][r] = fmaf(blk[mt][r], sc, acc[mt][r]);
            }
        }
        if constexpr (USE_SMEM) __syncthreads();
    }

    const int64_t split_row_base = (SPLITS > 1) ? ((int64_t)split * M) : 0;
    const float inv_cap = HAS_SOFTCAP ? (1.0f / soft_cap) : 0.0f;
    #pragma unroll
    for (int mt = 0; mt < M_TILES; ++mt) {
        #pragma unroll
        for (int r = 0; r < 8; ++r) {
            const int mr = tile_m * M_TILE + mt * 16 + half + 2 * r;
            if (mr < M) {
                float val = acc[mt][r];
                if constexpr (SPLITS == 1 && HAS_SOFTCAP) {
                    val = tanhf(val * inv_cap) * soft_cap;
                }
                const int64_t idx = (split_row_base + mr) * (int64_t)N_pad + out_col;
                if constexpr (SPLITS > 1 || OUT_FP32) {
                    reinterpret_cast<float*>(Out)[idx] = val;
                } else {
                    reinterpret_cast<uint16_t*>(Out)[idx] = f32_to_bf16_rtne(val);
                }
            }
        }
    }
}

// ============================================================================
// 2. SINGLE-PASS W4 WMMA LINEAR (+ OPTIONAL BIAS, N_TILE=64, PIPELINED)
// ============================================================================
template <int M_TILE, int SPLITS, bool HAS_BIAS, bool OUT_FP32>
__global__ __launch_bounds__(128, 6)
void hip_w4_wmma_kernel(
    const uint16_t* __restrict__ X,
    const uint8_t*  __restrict__ Packed,
    const uint16_t* __restrict__ Scales,
    const uint16_t* __restrict__ Bias,
    void*           __restrict__ Out,
    int M, int N_out, int N_pad, int K
) {
    constexpr int M_TILES = M_TILE / 16;
    constexpr bool USE_SMEM = true;
    const int tile_n  = (int)blockIdx.x;
    const int tile_m  = (int)blockIdx.y;
    const int split   = (int)blockIdx.z;

    const int tile128 = tile_n >> 1;
    const int n_half  = tile_n & 1;

    const int tid    = (int)threadIdx.x;
    const int warp   = tid >> 5;
    const int lane   = tid & 31;
    const int lane16 = lane & 15;
    const int half   = lane >> 4;

    const int row_in_128 = n_half * 64 + warp * 16 + lane16;
    const int out_col    = tile_n * 64 + warp * 16 + lane16;

    const int kt_total  = K >> 6;
    const int kt_per_sp = kt_total / SPLITS;
    const int kt_start  = split * kt_per_sp;
    const int kt_end    = kt_start + kt_per_sp;

    __shared__ uint16_t smem_X[USE_SMEM ? M_TILE : 1][USE_SMEM ? 72 : 1];

    float8 acc[M_TILES];
    #pragma unroll
    for (int mt = 0; mt < M_TILES; ++mt) {
        acc[mt] = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
    }

    const int tx_row = tid >> 3;
    const int tx_col = tid & 7;
    const int mr0 = tile_m * M_TILE + lane16;
    const bool mr0_valid = (mr0 < M);
    const uint16_t* x0_row_ptr = X + (int64_t)(mr0_valid ? mr0 : 0) * K;

    const int64_t base_tile_tidx = (int64_t)tile128 * kt_total * 128 + row_in_128;
    const uint8_t*  p4_base = Packed + base_tile_tidx * 32 + half * 16;
    const uint16_t* sc_base = Scales + base_tile_tidx;

    uint4    p4_cur = *reinterpret_cast<const uint4*>(p4_base + (int64_t)kt_start * (128 * 32));
    uint16_t sc_cur = sc_base[(int64_t)kt_start * 128];

    auto unpack_pair_w4 = [&](uint32_t px, uint32_t py, short16 &w_lo, short16 &w_hi) {
        uint32_t w[8], o[8];
        w[0] = unpack_w4_pair_bf16((px      ) & 0xFFu);
        w[1] = unpack_w4_pair_bf16((px >>  8) & 0xFFu);
        w[2] = unpack_w4_pair_bf16((px >> 16) & 0xFFu);
        w[3] = unpack_w4_pair_bf16((px >> 24) & 0xFFu);
        w[4] = unpack_w4_pair_bf16((py      ) & 0xFFu);
        w[5] = unpack_w4_pair_bf16((py >>  8) & 0xFFu);
        w[6] = unpack_w4_pair_bf16((py >> 16) & 0xFFu);
        w[7] = unpack_w4_pair_bf16((py >> 24) & 0xFFu);
        #pragma unroll
        for (int i = 0; i < 8; ++i) o[i] = __shfl_xor(w[i], 16, 32);
        __builtin_memcpy(&w_lo, (half == 0) ? w : o, 32);
        __builtin_memcpy(&w_hi, (half == 0) ? o : w, 32);
    };

    for (int kt = kt_start; kt < kt_end; ++kt) {
        const int k_base = kt << 6;
        if constexpr (USE_SMEM) {
            #pragma unroll
            for (int mt = 0; mt < M_TILES; ++mt) {
                const int mr = tile_m * M_TILE + mt * 16 + tx_row;
                uint4 xv = (mr < M)
                    ? *reinterpret_cast<const uint4*>(X + (int64_t)mr * K + k_base + tx_col * 8)
                    : make_uint4(0, 0, 0, 0);
                *reinterpret_cast<uint4*>(&smem_X[mt * 16 + tx_row][tx_col * 8]) = xv;
            }
            __syncthreads();
        }

        uint4 p4 = p4_cur;
        float sc = f16_bits_to_f32(sc_cur);
        if (kt + 1 < kt_end) {
            p4_cur = *reinterpret_cast<const uint4*>(p4_base + (int64_t)(kt + 1) * (128 * 32));
            sc_cur = sc_base[(int64_t)(kt + 1) * 128];
        }

        float8 blk[M_TILES];
        #pragma unroll
        for (int mt = 0; mt < M_TILES; ++mt) {
            blk[mt] = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
        }

        {
            short16 w0, w2;
            unpack_pair_w4(p4.x, p4.y, w0, w2);
            #pragma unroll
            for (int mt = 0; mt < M_TILES; ++mt) {
                short16 x0, x2;
                if constexpr (USE_SMEM) {
                    x0 = qdec_load_short16_u4(&smem_X[mt * 16 + lane16][0]);
                    x2 = qdec_load_short16_u4(&smem_X[mt * 16 + lane16][32]);
                } else {
                    x0 = mr0_valid ? qdec_load_short16_u4(x0_row_ptr + k_base +  0) : short16{};
                    x2 = mr0_valid ? qdec_load_short16_u4(x0_row_ptr + k_base + 32) : short16{};
                }
                blk[mt] = __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32(x0, w0, blk[mt]);
                blk[mt] = __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32(x2, w2, blk[mt]);
            }
        }
        {
            short16 w1, w3;
            unpack_pair_w4(p4.z, p4.w, w1, w3);
            #pragma unroll
            for (int mt = 0; mt < M_TILES; ++mt) {
                short16 x1, x3;
                if constexpr (USE_SMEM) {
                    x1 = qdec_load_short16_u4(&smem_X[mt * 16 + lane16][16]);
                    x3 = qdec_load_short16_u4(&smem_X[mt * 16 + lane16][48]);
                } else {
                    x1 = mr0_valid ? qdec_load_short16_u4(x0_row_ptr + k_base + 16) : short16{};
                    x3 = mr0_valid ? qdec_load_short16_u4(x0_row_ptr + k_base + 48) : short16{};
                }
                blk[mt] = __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32(x1, w1, blk[mt]);
                blk[mt] = __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32(x3, w3, blk[mt]);
            }
        }

        #pragma unroll
        for (int mt = 0; mt < M_TILES; ++mt) {
            #pragma unroll
            for (int r = 0; r < 8; ++r) {
                acc[mt][r] = fmaf(blk[mt][r], sc, acc[mt][r]);
            }
        }
        if constexpr (USE_SMEM) __syncthreads();
    }

    float bval = 0.0f;
    if constexpr (SPLITS == 1 && HAS_BIAS) {
        bval = (out_col < N_out) ? bf16_to_f32(Bias[out_col]) : 0.0f;
    }

    const int64_t split_row_base = (SPLITS > 1) ? ((int64_t)split * M) : 0;
    #pragma unroll
    for (int mt = 0; mt < M_TILES; ++mt) {
        #pragma unroll
        for (int r = 0; r < 8; ++r) {
            const int mr = tile_m * M_TILE + mt * 16 + half + 2 * r;
            if (mr < M) {
                float val = acc[mt][r];
                if constexpr (SPLITS == 1 && HAS_BIAS) val += bval;
                const int64_t idx = (split_row_base + mr) * (int64_t)N_pad + out_col;
                if constexpr (SPLITS > 1 || OUT_FP32) {
                    reinterpret_cast<float*>(Out)[idx] = val;
                } else {
                    reinterpret_cast<uint16_t*>(Out)[idx] = f32_to_bf16_rtne(val);
                }
            }
        }
    }
}

// ============================================================================
// 3. SPLIT-K REDUCER (SHARED BY W6, W4, AND DSPARK W8)
// ============================================================================
template <bool OUT_FP32, bool HAS_SOFTCAP>
__global__ __launch_bounds__(256, 4)
void hip_reduce_splits_kernel(
    const float* __restrict__ Partial,
    void*        __restrict__ Out,
    int64_t total_elems, int splits, float soft_cap
) {
    const int64_t base = ((int64_t)blockIdx.x * 256 + threadIdx.x) * 4;
    if (base >= total_elems) return;
    const float inv_cap = HAS_SOFTCAP ? (1.0f / soft_cap) : 0.0f;
    if (base + 4 <= total_elems && (total_elems & 3) == 0) {
        float4 s = make_float4(0.f, 0.f, 0.f, 0.f);
        if (splits == 4) {
            float4 a = *reinterpret_cast<const float4*>(Partial + base);
            float4 b = *reinterpret_cast<const float4*>(Partial + total_elems + base);
            float4 c = *reinterpret_cast<const float4*>(Partial + 2 * total_elems + base);
            float4 d = *reinterpret_cast<const float4*>(Partial + 3 * total_elems + base);
            s.x = (a.x + b.x) + (c.x + d.x);
            s.y = (a.y + b.y) + (c.y + d.y);
            s.z = (a.z + b.z) + (c.z + d.z);
            s.w = (a.w + b.w) + (c.w + d.w);
        } else {
            for (int sp = 0; sp < splits; ++sp) {
                float4 v = *reinterpret_cast<const float4*>(Partial + (int64_t)sp * total_elems + base);
                s.x += v.x; s.y += v.y; s.z += v.z; s.w += v.w;
            }
        }
        if constexpr (HAS_SOFTCAP) {
            s.x = tanhf(s.x * inv_cap) * soft_cap;
            s.y = tanhf(s.y * inv_cap) * soft_cap;
            s.z = tanhf(s.z * inv_cap) * soft_cap;
            s.w = tanhf(s.w * inv_cap) * soft_cap;
        }
        if constexpr (OUT_FP32) {
            *reinterpret_cast<float4*>(reinterpret_cast<float*>(Out) + base) = s;
        } else {
            uint2 pk = make_uint2(pack_bf16x2(s.x, s.y), pack_bf16x2(s.z, s.w));
            *reinterpret_cast<uint2*>(reinterpret_cast<uint16_t*>(Out) + base) = pk;
        }
    } else {
        for (int j = 0; j < 4 && base + j < total_elems; ++j) {
            int64_t idx = base + j;
            float s = 0.0f;
            for (int sp = 0; sp < splits; ++sp) s += Partial[(int64_t)sp * total_elems + idx];
            if constexpr (HAS_SOFTCAP) s = tanhf(s * inv_cap) * soft_cap;
            if constexpr (OUT_FP32) reinterpret_cast<float*>(Out)[idx] = s;
            else reinterpret_cast<uint16_t*>(Out)[idx] = f32_to_bf16_rtne(s);
        }
    }
}

// ============================================================================
// 4. GROUPWISE W6 M=1 GEMV (FULL-MLP PREFETCHED + V_DOT2_F32_BF16)
// ============================================================================
template <int GROUP_SIZE, bool OUT_FP32, bool HAS_SOFTCAP>
__global__ __launch_bounds__(128, 8)
void hip_w6_head_gemv_m1_kernel(
    const uint16_t* __restrict__ X,
    const uint64_t* __restrict__ Pair,
    const uint32_t* __restrict__ Third,
    const uint16_t* __restrict__ Scales,
    void*           __restrict__ Out,
    int N, int K, float soft_cap
) {
    const int tid  = (int)threadIdx.x;
    const int warp = tid >> 5;
    const int lane = tid & 31;
    const int row  = (int)blockIdx.x * 4 + warp;

    __shared__ uint16_t smem_X[3072];
    for (int i = tid * 8; i < K; i += 128 * 8) {
        *reinterpret_cast<uint4*>(&smem_X[i]) = *reinterpret_cast<const uint4*>(X + i);
    }
    __syncthreads();

    if (row >= N) return;

    const int records = K >> 4;   // 144 for K=2304, 192 for K=3072 -> <= 6 iters/lane!
    const int groups  = K / GROUP_SIZE;
    constexpr int REC_PER_GRP = GROUP_SIZE / 16;

    const uint64_t* pair_row  = Pair   + (int64_t)row * records;
    const uint32_t* third_row = Third  + (int64_t)row * records;
    const uint16_t* sc_row    = Scales + (int64_t)row * groups;

    // Issue all <= 6 record loads upfront for maximum memory-level parallelism
    uint64_t p_buf[6];
    uint32_t t_buf[6];
    uint16_t s_buf[6];
    #pragma unroll
    for (int it = 0; it < 6; ++it) {
        int rec = it * 32 + lane;
        if (rec < records) {
            p_buf[it] = pair_row[rec];
            t_buf[it] = third_row[rec];
            s_buf[it] = sc_row[rec / REC_PER_GRP];
        }
    }

    auto s6_to_bf16_u16 = [](uint32_t raw6) -> uint32_t {
        int v = ((int)(raw6 << 26)) >> 26;
        return __float_as_uint((float)v) >> 16;
    };

    float acc = 0.0f;
    #pragma unroll
    for (int it = 0; it < 6; ++it) {
        int rec = it * 32 + lane;
        if (rec < records) {
            uint64_t p  = p_buf[it];
            uint32_t w0 = (uint32_t)p;
            uint32_t w1 = (uint32_t)(p >> 32);
            uint32_t w2 = t_buf[it];
            uint32_t last = (w0 >> 30) | ((w1 >> 28) & 12u) | ((w2 >> 26) & 48u);

            uint32_t c_pk[8];
            c_pk[0] = s6_to_bf16_u16(w0      ) | (s6_to_bf16_u16(w0 >>  6) << 16);
            c_pk[1] = s6_to_bf16_u16(w0 >> 12) | (s6_to_bf16_u16(w0 >> 18) << 16);
            c_pk[2] = s6_to_bf16_u16(w0 >> 24) | (s6_to_bf16_u16(w1      ) << 16);
            c_pk[3] = s6_to_bf16_u16(w1 >>  6) | (s6_to_bf16_u16(w1 >> 12) << 16);
            c_pk[4] = s6_to_bf16_u16(w1 >> 18) | (s6_to_bf16_u16(w1 >> 24) << 16);
            c_pk[5] = s6_to_bf16_u16(w2      ) | (s6_to_bf16_u16(w2 >>  6) << 16);
            c_pk[6] = s6_to_bf16_u16(w2 >> 12) | (s6_to_bf16_u16(w2 >> 18) << 16);
            c_pk[7] = s6_to_bf16_u16(w2 >> 24) | (s6_to_bf16_u16(last    ) << 16);

            uint4 x0 = *reinterpret_cast<const uint4*>(&smem_X[rec * 16 + 0]);
            uint4 x1 = *reinterpret_cast<const uint4*>(&smem_X[rec * 16 + 8]);
            float dot = 0.0f;
            asm("v_dot2_f32_bf16 %0, %1, %2, %0" : "+v"(dot) : "v"(c_pk[0]), "v"(x0.x));
            asm("v_dot2_f32_bf16 %0, %1, %2, %0" : "+v"(dot) : "v"(c_pk[1]), "v"(x0.y));
            asm("v_dot2_f32_bf16 %0, %1, %2, %0" : "+v"(dot) : "v"(c_pk[2]), "v"(x0.z));
            asm("v_dot2_f32_bf16 %0, %1, %2, %0" : "+v"(dot) : "v"(c_pk[3]), "v"(x0.w));
            asm("v_dot2_f32_bf16 %0, %1, %2, %0" : "+v"(dot) : "v"(c_pk[4]), "v"(x1.x));
            asm("v_dot2_f32_bf16 %0, %1, %2, %0" : "+v"(dot) : "v"(c_pk[5]), "v"(x1.y));
            asm("v_dot2_f32_bf16 %0, %1, %2, %0" : "+v"(dot) : "v"(c_pk[6]), "v"(x1.z));
            asm("v_dot2_f32_bf16 %0, %1, %2, %0" : "+v"(dot) : "v"(c_pk[7]), "v"(x1.w));

            acc = fmaf(dot, f16_bits_to_f32(s_buf[it]), acc);
        }
    }

    #pragma unroll
    for (int off = 16; off > 0; off >>= 1) {
        acc += __shfl_xor(acc, off, 32);
    }
    if (lane == 0) {
        if constexpr (HAS_SOFTCAP) {
            acc = tanhf(acc * (1.0f / soft_cap)) * soft_cap;
        }
        if constexpr (OUT_FP32) {
            reinterpret_cast<float*>(Out)[row] = acc;
        } else {
            reinterpret_cast<uint16_t*>(Out)[row] = f32_to_bf16_rtne(acc);
        }
    }
}

// ============================================================================
// 5. W4 MARKOV FUSED ARGMAX (8 LANES/ROW, 16B UINT4 + V_DOT2)
// ============================================================================
__device__ __forceinline__ void first_max_pair(float &ls, int32_t &li, float rs, int32_t ri) {
    bool take_r = (rs > ls) || ((rs == ls) && (ri < li));
    if (take_r) {
        ls = rs;
        li = ri;
    }
}

__global__ __launch_bounds__(128, 8)
void hip_w4_markov_argmax_partial_kernel(
    const uint16_t* __restrict__ X,
    const uint8_t*  __restrict__ Packed,
    const uint16_t* __restrict__ Scales,
    const uint16_t* __restrict__ Bias,
    const uint16_t* __restrict__ Base,
    float*          __restrict__ PartVals,
    int32_t*        __restrict__ PartIds,
    int valid, float softcap
) {
    const int bid        = (int)blockIdx.x;
    const int tid        = (int)threadIdx.x;
    const int row_in_cta = tid >> 3;
    const int sub_lane   = tid & 7;
    const int row        = bid * 16 + row_in_cta;

    __shared__ uint32_t smem_X2[128];
    smem_X2[tid] = *reinterpret_cast<const uint32_t*>(X + tid * 2);
    __syncthreads();

    float dot = 0.0f;
    if (row < valid) {
        uint4 p = *reinterpret_cast<const uint4*>(Packed + (int64_t)row * 128 + sub_lane * 16);
        float sc = f16_bits_to_f32(Scales[(int64_t)row * 4 + (sub_lane >> 1)]);
        const uint32_t* x2 = &smem_X2[sub_lane * 16];
        uint32_t pw[4] = {p.x, p.y, p.z, p.w};
        float acc = 0.0f;
        #pragma unroll
        for (int w = 0; w < 4; ++w) {
            uint32_t u = pw[w];
            uint32_t c0 = unpack_w4_pair_bf16((u      ) & 0xFFu);
            uint32_t c1 = unpack_w4_pair_bf16((u >>  8) & 0xFFu);
            uint32_t c2 = unpack_w4_pair_bf16((u >> 16) & 0xFFu);
            uint32_t c3 = unpack_w4_pair_bf16((u >> 24) & 0xFFu);
            asm("v_dot2_f32_bf16 %0, %1, %2, %0" : "+v"(acc) : "v"(c0), "v"(x2[w * 4 + 0]));
            asm("v_dot2_f32_bf16 %0, %1, %2, %0" : "+v"(acc) : "v"(c1), "v"(x2[w * 4 + 1]));
            asm("v_dot2_f32_bf16 %0, %1, %2, %0" : "+v"(acc) : "v"(c2), "v"(x2[w * 4 + 2]));
            asm("v_dot2_f32_bf16 %0, %1, %2, %0" : "+v"(acc) : "v"(c3), "v"(x2[w * 4 + 3]));
        }
        dot = acc * sc;
    }

    dot += __shfl_xor(dot, 4, 32);
    dot += __shfl_xor(dot, 2, 32);
    dot += __shfl_xor(dot, 1, 32);

    __shared__ float smem_s[16];
    __shared__ int32_t smem_i[16];
    if (sub_lane == 0) {
        float score = -INFINITY;
        if (row < valid) {
            float bv = bf16_to_f32(Bias[row]);
            float base_v = bf16_to_f32(Base[row]);
            float corr = bf16_round_f32(dot + bv);
            float comb = bf16_round_f32(base_v + corr);
            score = bf16_round_f32(tanhf(comb * (1.0f / softcap)) * softcap);
            if (!(score == score)) score = -INFINITY;
        }
        smem_s[row_in_cta] = score;
        smem_i[row_in_cta] = row;
    }
    __syncthreads();

    if (tid < 16) {
        float bs = smem_s[tid];
        int32_t bi = smem_i[tid];
        #pragma unroll
        for (int off = 8; off > 0; off >>= 1) {
            float os = __shfl_xor(bs, off, 32);
            int32_t oi = __shfl_xor(bi, off, 32);
            first_max_pair(bs, bi, os, oi);
        }
        if (tid == 0) {
            PartVals[bid] = bs;
            PartIds[bid]  = bi;
        }
    }
}

__global__ __launch_bounds__(256, 1)
void hip_w4_markov_argmax_finish_kernel(
    const float*   __restrict__ PartVals,
    const int32_t* __restrict__ PartIds,
    int64_t*       __restrict__ OutIdx,
    int splits
) {
    const int tid  = (int)threadIdx.x;
    const int warp = tid >> 5;
    const int lane = tid & 31;

    float bs = -INFINITY;
    int32_t bi = 0x7FFFFFFF;
    for (int i = tid; i < splits; i += 256) {
        first_max_pair(bs, bi, PartVals[i], PartIds[i]);
    }
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1) {
        float os = __shfl_xor(bs, off, 32);
        int32_t oi = __shfl_xor(bi, off, 32);
        first_max_pair(bs, bi, os, oi);
    }
    __shared__ float ws[8];
    __shared__ int32_t wi[8];
    if (lane == 0) {
        ws[warp] = bs;
        wi[warp] = bi;
    }
    __syncthreads();
    if (tid == 0) {
        float fs = ws[0];
        int32_t fi = wi[0];
        #pragma unroll
        for (int w = 1; w < 8; ++w) {
            first_max_pair(fs, fi, ws[w], wi[w]);
        }
        OutIdx[0] = (int64_t)fi;
    }
}

extern "C" {

int launch_hip_reduce_splits(
    const void* Partial, void* Out, int64_t total_elems, int splits, int out_fp32, float soft_cap, void* stream_ptr
) {
    if (total_elems <= 0 || splits <= 1) return 0;
    hipStream_t stream = (hipStream_t)stream_ptr;
    dim3 grid((unsigned int)((total_elems + 1023) / 1024));
    dim3 block(256);
    const bool has_cap = (soft_cap > 0.0f);
    if (out_fp32) {
        if (has_cap) hipLaunchKernelGGL((hip_reduce_splits_kernel<true, true>),  grid, block, 0, stream, (const float*)Partial, Out, total_elems, splits, soft_cap);
        else         hipLaunchKernelGGL((hip_reduce_splits_kernel<true, false>), grid, block, 0, stream, (const float*)Partial, Out, total_elems, splits, soft_cap);
    } else {
        if (has_cap) hipLaunchKernelGGL((hip_reduce_splits_kernel<false, true>),  grid, block, 0, stream, (const float*)Partial, Out, total_elems, splits, soft_cap);
        else         hipLaunchKernelGGL((hip_reduce_splits_kernel<false, false>), grid, block, 0, stream, (const float*)Partial, Out, total_elems, splits, soft_cap);
    }
    return 0;
}

int launch_hip_w6_wmma(
    const void* XQ, const void* XK, const void* XV,
    const void* P4, const void* P2, const void* Scales,
    void* Out, void* Partial,
    int M, int N_pad, int K, int q_pad, int k_end,
    int is_qkv, int splits, int out_fp32, float soft_cap, void* stream_ptr
) {
    if (M <= 0 || N_pad <= 0 || K <= 0 || (N_pad & 127) != 0 || (K & 63) != 0) return -1;
    if (splits != 1 && splits != 2 && splits != 4) return -1;
    if (((K >> 6) % splits) != 0) return -1;
    hipStream_t stream = (hipStream_t)stream_ptr;
    const int q_tiles = q_pad >> 7;
    const int k_tiles_end = k_end >> 7;
    const bool has_cap = (soft_cap > 0.0f);
    if (is_qkv) {
        splits = 1;
    }
    void* dst = (splits > 1) ? Partial : Out;

    auto run = [&](auto qkv_t, auto mt_t, auto sp_t, auto f32_t, auto cap_t) {
        constexpr bool QKV = decltype(qkv_t)::value;
        constexpr int  MT  = decltype(mt_t)::value;
        constexpr int  SP  = decltype(sp_t)::value;
        constexpr bool F32 = decltype(f32_t)::value;
        constexpr bool CAP = decltype(cap_t)::value;
        dim3 grid(N_pad / 64, (M + MT - 1) / MT, SP);
        dim3 block(128);
        hipLaunchKernelGGL((hip_w6_wmma_kernel<QKV, MT, SP, F32, CAP>), grid, block, 0, stream,
            (const uint16_t*)XQ, (const uint16_t*)XK, (const uint16_t*)XV,
            (const uint8_t*)P4, (const uint32_t*)P2, (const uint16_t*)Scales,
            dst, M, N_pad, K, q_tiles, k_tiles_end, soft_cap);
    };

    auto disp_mt = [&](auto qkv_t, auto sp_t, auto f32_t, auto cap_t) {
        if (M <= 16)      run(qkv_t, std::integral_constant<int, 16>{}, sp_t, f32_t, cap_t);
        else if (M <= 32) run(qkv_t, std::integral_constant<int, 32>{}, sp_t, f32_t, cap_t);
        else if (M <= 48) run(qkv_t, std::integral_constant<int, 48>{}, sp_t, f32_t, cap_t);
        else              run(qkv_t, std::integral_constant<int, 32>{}, sp_t, f32_t, cap_t);
    };

    if (is_qkv) {
        disp_mt(std::true_type{}, std::integral_constant<int, 1>{}, std::false_type{}, std::false_type{});
    } else if (splits == 1) {
        if (out_fp32) {
            if (has_cap) disp_mt(std::false_type{}, std::integral_constant<int, 1>{}, std::true_type{}, std::true_type{});
            else         disp_mt(std::false_type{}, std::integral_constant<int, 1>{}, std::true_type{}, std::false_type{});
        } else {
            if (has_cap) disp_mt(std::false_type{}, std::integral_constant<int, 1>{}, std::false_type{}, std::true_type{});
            else         disp_mt(std::false_type{}, std::integral_constant<int, 1>{}, std::false_type{}, std::false_type{});
        }
    } else if (splits == 2) {
        disp_mt(std::false_type{}, std::integral_constant<int, 2>{}, std::true_type{}, std::false_type{});
        launch_hip_reduce_splits(Partial, Out, (int64_t)M * N_pad, 2, out_fp32, soft_cap, stream_ptr);
    } else {
        disp_mt(std::false_type{}, std::integral_constant<int, 4>{}, std::true_type{}, std::false_type{});
        launch_hip_reduce_splits(Partial, Out, (int64_t)M * N_pad, 4, out_fp32, soft_cap, stream_ptr);
    }
    return 0;
}

int launch_hip_w6_head_gemv_m1(
    const void* X, const void* Pair, const void* Third, const void* Scales, void* Out,
    int N, int K, int group_size, int out_fp32, float soft_cap, void* stream_ptr
) {
    if (N <= 0 || K <= 0 || K > 3072 || (K & 15) != 0) return -1;
    if (group_size != 32 && group_size != 64) return -1;
    hipStream_t stream = (hipStream_t)stream_ptr;
    dim3 grid((N + 3) / 4);
    dim3 block(128);
    const bool has_cap = (soft_cap > 0.0f);
    auto go = [&](auto gs_t, auto f32_t, auto cap_t) {
        constexpr int  GS  = decltype(gs_t)::value;
        constexpr bool F32 = decltype(f32_t)::value;
        constexpr bool CAP = decltype(cap_t)::value;
        hipLaunchKernelGGL((hip_w6_head_gemv_m1_kernel<GS, F32, CAP>), grid, block, 0, stream,
            (const uint16_t*)X, (const uint64_t*)Pair, (const uint32_t*)Third, (const uint16_t*)Scales,
            Out, N, K, soft_cap);
    };
    auto disp_out = [&](auto gs_t) {
        if (out_fp32) {
            if (has_cap) go(gs_t, std::true_type{}, std::true_type{});
            else         go(gs_t, std::true_type{}, std::false_type{});
        } else {
            if (has_cap) go(gs_t, std::false_type{}, std::true_type{});
            else         go(gs_t, std::false_type{}, std::false_type{});
        }
    };
    if (group_size == 32) disp_out(std::integral_constant<int, 32>{});
    else                  disp_out(std::integral_constant<int, 64>{});
    return 0;
}

int launch_hip_w4_wmma(
    const void* X, const void* Packed, const void* Scales, const void* Bias,
    void* Out, void* Partial,
    int M, int N_out, int N_pad, int K, int splits, int has_bias, int out_fp32, void* stream_ptr
) {
    if (M <= 0 || N_pad <= 0 || K <= 0 || (N_pad & 127) != 0 || (K & 63) != 0) return -1;
    if (has_bias) splits = 1;
    if (splits != 1 && splits != 2 && splits != 4) return -1;
    if (((K >> 6) % splits) != 0) return -1;
    hipStream_t stream = (hipStream_t)stream_ptr;
    void* dst = (splits > 1) ? Partial : Out;

    auto run = [&](auto mt_t, auto sp_t, auto bias_t, auto f32_t) {
        constexpr int  MT  = decltype(mt_t)::value;
        constexpr int  SP  = decltype(sp_t)::value;
        constexpr bool BS  = decltype(bias_t)::value;
        constexpr bool F32 = decltype(f32_t)::value;
        dim3 grid(N_pad / 64, (M + MT - 1) / MT, SP);
        dim3 block(128);
        hipLaunchKernelGGL((hip_w4_wmma_kernel<MT, SP, BS, F32>), grid, block, 0, stream,
            (const uint16_t*)X, (const uint8_t*)Packed, (const uint16_t*)Scales,
            (const uint16_t*)Bias, dst, M, N_out, N_pad, K);
    };
    auto disp_mt = [&](auto sp_t, auto bias_t, auto f32_t) {
        if (M <= 16)      run(std::integral_constant<int, 16>{}, sp_t, bias_t, f32_t);
        else if (M <= 32) run(std::integral_constant<int, 32>{}, sp_t, bias_t, f32_t);
        else if (M <= 48) run(std::integral_constant<int, 48>{}, sp_t, bias_t, f32_t);
        else              run(std::integral_constant<int, 32>{}, sp_t, bias_t, f32_t);
    };

    if (splits == 1) {
        if (has_bias) disp_mt(std::integral_constant<int, 1>{}, std::true_type{}, std::false_type{});
        else if (out_fp32) disp_mt(std::integral_constant<int, 1>{}, std::false_type{}, std::true_type{});
        else disp_mt(std::integral_constant<int, 1>{}, std::false_type{}, std::false_type{});
    } else if (splits == 2) {
        disp_mt(std::integral_constant<int, 2>{}, std::false_type{}, std::true_type{});
        launch_hip_reduce_splits(Partial, Out, (int64_t)M * N_pad, 2, out_fp32, 0.0f, stream_ptr);
    } else {
        disp_mt(std::integral_constant<int, 4>{}, std::false_type{}, std::true_type{});
        launch_hip_reduce_splits(Partial, Out, (int64_t)M * N_pad, 4, out_fp32, 0.0f, stream_ptr);
    }
    return 0;
}

int launch_hip_w4_markov_argmax(
    const void* Latent, const void* Packed, const void* Scales,
    const void* Bias, const void* Base,
    void* PartVals, void* PartIds, void* OutIdx,
    int valid, float softcap, void* stream_ptr
) {
    if (valid <= 0) return -1;
    hipStream_t stream = (hipStream_t)stream_ptr;
    const int splits = (valid + 15) / 16;
    hipLaunchKernelGGL(hip_w4_markov_argmax_partial_kernel, dim3(splits), dim3(128), 0, stream,
        (const uint16_t*)Latent, (const uint8_t*)Packed, (const uint16_t*)Scales,
        (const uint16_t*)Bias, (const uint16_t*)Base,
        (float*)PartVals, (int32_t*)PartIds, valid, softcap);
    hipLaunchKernelGGL(hip_w4_markov_argmax_finish_kernel, dim3(1), dim3(256), 0, stream,
        (const float*)PartVals, (const int32_t*)PartIds, (int64_t*)OutIdx, splits);
    return 0;
}

}  // extern "C"
