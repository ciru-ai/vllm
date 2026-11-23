// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 CIRU AI.
// Native RDNA 3.5 (gfx1151) prefill glue kernels for Inkling 2 Mini.

#include "hip_common.cuh"

// ============================================================================
// 0. ZERO-MATERIALIZATION 4-QUERY ATTN_RES MESSAGE COEFFICIENTS (D=2304, N_STACK=1)
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

// ============================================================================
// C1: Request- & Tile-Boundary-Safe Static ShortConv Prefill (BT=64, W=4, 8 ch/thr)
// ============================================================================

template<bool HAS_ADDEND, bool HAS_RES, bool ADD_F32, bool RES_F32, bool OUT_F32>
__global__ __launch_bounds__(64, 8)
void hip_sconv_prefill_kernel(
    const uint16_t* __restrict__ X,
    const uint16_t* __restrict__ Weight,
    uint16_t*       __restrict__ Cache,
    const void*     __restrict__ Pos,
    const int32_t*  __restrict__ BlockTable,
    const int32_t*  __restrict__ SeqIdx,
    const int64_t*  __restrict__ Slot,
    const int32_t*  __restrict__ QStart,
    const void*     __restrict__ Addend,
    const void*                  Residual,
    void*                        Out,
    int T, int D, int H, int stream_off, int stream_w,
    int64_t sx, int64_t sw0, int64_t sw1,
    int64_t scb, int64_t sch, int64_t scn,
    int64_t sbt, int max_blocks, int N_ring, float scale, int pos_i64
) {
    constexpr int BT = 64;
    const int t0 = blockIdx.x * BT;
    const int c0 = blockIdx.y * 512 + threadIdx.x * 8;
    if (t0 >= T || c0 >= D) return;
    const int t_end = min(t0 + BT, T);
    const bool full8 = (c0 + 8 <= D);

    float w[4][8];
    #pragma unroll
    for (int i = 0; i < 8; ++i) {
        int c = c0 + i;
        if (c < D) {
            #pragma unroll
            for (int k = 0; k < 4; ++k) {
                w[k][i] = bf16_to_f32(Weight[(int64_t)c * sw0 + (int64_t)k * sw1]);
            }
        } else {
            #pragma unroll
            for (int k = 0; k < 4; ++k) w[k][i] = 0.0f;
        }
    }

    int64_t c_col[8];
    #pragma unroll
    for (int i = 0; i < 8; ++i) {
        int c = min(c0 + i, D - 1);
        c_col[i] = (int64_t)(c / stream_w) * sch + stream_off + (c % stream_w);
    }
    const bool contig_cache = full8 && ((c0 % stream_w) + 8 <= stream_w);

    auto load_tap = [&](int t_cur, int pos_cur, int req_cur, int qs_cur, int lag, float dst[8]) {
        int src = pos_cur - lag;
        if (src < 0) {
            #pragma unroll
            for (int i = 0; i < 8; ++i) dst[i] = 0.0f;
            return;
        }
        int row = t_cur - lag;
        if (row >= qs_cur) {
            const uint16_t* xp = X + (int64_t)row * sx + c0;
            if (full8 && (((uintptr_t)xp & 15u) == 0u)) {
                uint4 u = *reinterpret_cast<const uint4*>(xp);
                dst[0] = bf16_to_f32((uint16_t)(u.x & 0xFFFFu)); dst[1] = bf16_to_f32((uint16_t)(u.x >> 16));
                dst[2] = bf16_to_f32((uint16_t)(u.y & 0xFFFFu)); dst[3] = bf16_to_f32((uint16_t)(u.y >> 16));
                dst[4] = bf16_to_f32((uint16_t)(u.z & 0xFFFFu)); dst[5] = bf16_to_f32((uint16_t)(u.z >> 16));
                dst[6] = bf16_to_f32((uint16_t)(u.w & 0xFFFFu)); dst[7] = bf16_to_f32((uint16_t)(u.w >> 16));
            } else {
                #pragma unroll
                for (int i = 0; i < 8; ++i) dst[i] = (c0 + i < D) ? bf16_to_f32(xp[i]) : 0.0f;
            }
        } else {
            int lblk = min(src / N_ring, max_blocks - 1);
            int64_t blk = (int64_t)BlockTable[(int64_t)req_cur * sbt + lblk];
            int64_t base = blk * scb + (int64_t)(src % N_ring) * scn;
            if (contig_cache && (((uintptr_t)(Cache + base + c_col[0]) & 15u) == 0u)) {
                uint4 u = *reinterpret_cast<const uint4*>(Cache + base + c_col[0]);
                dst[0] = bf16_to_f32((uint16_t)(u.x & 0xFFFFu)); dst[1] = bf16_to_f32((uint16_t)(u.x >> 16));
                dst[2] = bf16_to_f32((uint16_t)(u.y & 0xFFFFu)); dst[3] = bf16_to_f32((uint16_t)(u.y >> 16));
                dst[4] = bf16_to_f32((uint16_t)(u.z & 0xFFFFu)); dst[5] = bf16_to_f32((uint16_t)(u.z >> 16));
                dst[6] = bf16_to_f32((uint16_t)(u.w & 0xFFFFu)); dst[7] = bf16_to_f32((uint16_t)(u.w >> 16));
            } else {
                #pragma unroll
                for (int i = 0; i < 8; ++i) dst[i] = (c0 + i < D) ? bf16_to_f32(Cache[base + c_col[i]]) : 0.0f;
            }
        }
    };

    float h0[8] = {0.f,0.f,0.f,0.f,0.f,0.f,0.f,0.f};
    float h1[8] = {0.f,0.f,0.f,0.f,0.f,0.f,0.f,0.f};
    float h2[8] = {0.f,0.f,0.f,0.f,0.f,0.f,0.f,0.f};
    bool need_init = true;
    for (int t = t0; t < t_end; ++t) {
        int qs  = QStart[t];
        int pos = pos_i64 ? (int)((const int64_t*)Pos)[t] : ((const int32_t*)Pos)[t];
        int req = SeqIdx[t];
        int64_t slot = Slot[t];

        if (slot >= 0) {
            if (need_init || t == qs) {
                load_tap(t, pos, req, qs, 3, h0);
                load_tap(t, pos, req, qs, 2, h1);
                load_tap(t, pos, req, qs, 1, h2);
                need_init = false;
            }
        } else {
            need_init = true;
        }

        uint16_t cur_bits[8];
        float cur[8];
        const uint16_t* xp = X + (int64_t)t * sx + c0;
        uint4 x_u4;
        bool x_aligned = full8 && (((uintptr_t)xp & 15u) == 0u);
        if (x_aligned) {
            x_u4 = *reinterpret_cast<const uint4*>(xp);
            cur[0] = bf16_to_f32((uint16_t)(x_u4.x & 0xFFFFu)); cur[1] = bf16_to_f32((uint16_t)(x_u4.x >> 16));
            cur[2] = bf16_to_f32((uint16_t)(x_u4.y & 0xFFFFu)); cur[3] = bf16_to_f32((uint16_t)(x_u4.y >> 16));
            cur[4] = bf16_to_f32((uint16_t)(x_u4.z & 0xFFFFu)); cur[5] = bf16_to_f32((uint16_t)(x_u4.z >> 16));
            cur[6] = bf16_to_f32((uint16_t)(x_u4.w & 0xFFFFu)); cur[7] = bf16_to_f32((uint16_t)(x_u4.w >> 16));
        } else {
            #pragma unroll
            for (int i = 0; i < 8; ++i) {
                cur_bits[i] = (c0 + i < D) ? xp[i] : 0u;
                cur[i] = bf16_to_f32(cur_bits[i]);
            }
            x_u4 = make_uint4(
                (uint32_t)cur_bits[0] | ((uint32_t)cur_bits[1] << 16),
                (uint32_t)cur_bits[2] | ((uint32_t)cur_bits[3] << 16),
                (uint32_t)cur_bits[4] | ((uint32_t)cur_bits[5] << 16),
                (uint32_t)cur_bits[6] | ((uint32_t)cur_bits[7] << 16));
        }

        if (slot >= 0) {
            int64_t cbase = (slot / N_ring) * scb + (slot % N_ring) * scn;
            if (contig_cache && (((uintptr_t)(Cache + cbase + c_col[0]) & 15u) == 0u)) {
                *reinterpret_cast<uint4*>(Cache + cbase + c_col[0]) = x_u4;
            } else {
                uint32_t cw[4] = {x_u4.x, x_u4.y, x_u4.z, x_u4.w};
                #pragma unroll
                for (int i = 0; i < 4; ++i) {
                    if (c0 + 2 * i + 0 < D) Cache[cbase + c_col[2 * i + 0]] = (uint16_t)(cw[i] & 0xFFFFu);
                    if (c0 + 2 * i + 1 < D) Cache[cbase + c_col[2 * i + 1]] = (uint16_t)(cw[i] >> 16);
                }
            }
        }

        float y[8];
        bool valid = (slot >= 0);
        #pragma unroll
        for (int i = 0; i < 8; ++i) {
            float acc = 0.0f;
            if (valid) {
                acc = h0[i] * w[0][i];
                acc = fmaf(h1[i], w[1][i], acc);
                acc = fmaf(h2[i], w[2][i], acc);
                acc = fmaf(cur[i], w[3][i], acc);
            }
            float yi = (acc + cur[i]) * scale;
            if constexpr (!OUT_F32) yi = bf16_round_f32(yi);
            y[i] = yi;
            h0[i] = h1[i]; h1[i] = h2[i]; h2[i] = cur[i];
        }

        int64_t out_off = (int64_t)t * D + c0;
        if constexpr (HAS_ADDEND) {
            float av[8];
            if constexpr (ADD_F32) {
                const float* ap = reinterpret_cast<const float*>(Addend) + out_off;
                if (full8 && (((uintptr_t)ap & 15u) == 0u)) {
                    float4 a0 = *reinterpret_cast<const float4*>(ap);
                    float4 a1 = *reinterpret_cast<const float4*>(ap + 4);
                    av[0]=a0.x; av[1]=a0.y; av[2]=a0.z; av[3]=a0.w;
                    av[4]=a1.x; av[5]=a1.y; av[6]=a1.z; av[7]=a1.w;
                } else {
                    #pragma unroll
                    for (int i = 0; i < 8; ++i) av[i] = (c0 + i < D) ? ap[i] : 0.0f;
                }
            } else {
                const uint16_t* ap = reinterpret_cast<const uint16_t*>(Addend) + out_off;
                if (full8 && (((uintptr_t)ap & 15u) == 0u)) {
                    uint4 au = *reinterpret_cast<const uint4*>(ap);
                    av[0]=bf16_to_f32((uint16_t)(au.x&0xFFFFu)); av[1]=bf16_to_f32((uint16_t)(au.x>>16));
                    av[2]=bf16_to_f32((uint16_t)(au.y&0xFFFFu)); av[3]=bf16_to_f32((uint16_t)(au.y>>16));
                    av[4]=bf16_to_f32((uint16_t)(au.z&0xFFFFu)); av[5]=bf16_to_f32((uint16_t)(au.z>>16));
                    av[6]=bf16_to_f32((uint16_t)(au.w&0xFFFFu)); av[7]=bf16_to_f32((uint16_t)(au.w>>16));
                } else {
                    #pragma unroll
                    for (int i = 0; i < 8; ++i) av[i] = (c0 + i < D) ? bf16_to_f32(ap[i]) : 0.0f;
                }
            }
            if constexpr (HAS_RES) {
                if constexpr (RES_F32) {
                    const float* rp = reinterpret_cast<const float*>(Residual) + out_off;
                    if (full8 && (((uintptr_t)rp & 15u) == 0u)) {
                        float4 r0 = *reinterpret_cast<const float4*>(rp);
                        float4 r1 = *reinterpret_cast<const float4*>(rp + 4);
                        av[0]=r0.x+av[0]; av[1]=r0.y+av[1]; av[2]=r0.z+av[2]; av[3]=r0.w+av[3];
                        av[4]=r1.x+av[4]; av[5]=r1.y+av[5]; av[6]=r1.z+av[6]; av[7]=r1.w+av[7];
                    } else {
                        #pragma unroll
                        for (int i = 0; i < 8; ++i) if (c0 + i < D) av[i] = rp[i] + av[i];
                    }
                } else {
                    const uint16_t* rp = reinterpret_cast<const uint16_t*>(Residual) + out_off;
                    if (full8 && (((uintptr_t)rp & 15u) == 0u)) {
                        uint4 ru = *reinterpret_cast<const uint4*>(rp);
                        av[0]=bf16_to_f32((uint16_t)(ru.x&0xFFFFu))+av[0]; av[1]=bf16_to_f32((uint16_t)(ru.x>>16))+av[1];
                        av[2]=bf16_to_f32((uint16_t)(ru.y&0xFFFFu))+av[2]; av[3]=bf16_to_f32((uint16_t)(ru.y>>16))+av[3];
                        av[4]=bf16_to_f32((uint16_t)(ru.z&0xFFFFu))+av[4]; av[5]=bf16_to_f32((uint16_t)(ru.z>>16))+av[5];
                        av[6]=bf16_to_f32((uint16_t)(ru.w&0xFFFFu))+av[6]; av[7]=bf16_to_f32((uint16_t)(ru.w>>16))+av[7];
                    } else {
                        #pragma unroll
                        for (int i = 0; i < 8; ++i) if (c0 + i < D) av[i] = bf16_to_f32(rp[i]) + av[i];
                    }
                }
            }
            #pragma unroll
            for (int i = 0; i < 8; ++i) y[i] = av[i] + y[i];
        }

        if constexpr (OUT_F32) {
            float* op = reinterpret_cast<float*>(Out) + out_off;
            if (full8 && (((uintptr_t)op & 15u) == 0u)) {
                *reinterpret_cast<float4*>(op)     = make_float4(y[0], y[1], y[2], y[3]);
                *reinterpret_cast<float4*>(op + 4) = make_float4(y[4], y[5], y[6], y[7]);
            } else {
                #pragma unroll
                for (int i = 0; i < 8; ++i) if (c0 + i < D) op[i] = y[i];
            }
        } else {
            uint16_t* op = reinterpret_cast<uint16_t*>(Out) + out_off;
            if (full8 && (((uintptr_t)op & 15u) == 0u)) {
                *reinterpret_cast<uint4*>(op) = make_uint4(
                    pack_bf16x2(y[0], y[1]), pack_bf16x2(y[2], y[3]),
                    pack_bf16x2(y[4], y[5]), pack_bf16x2(y[6], y[7]));
            } else {
                #pragma unroll
                for (int i = 0; i < 8; ++i) if (c0 + i < D) op[i] = f32_to_bf16_rtne(y[i]);
            }
        }
    }
}

// ============================================================================
// C2: Native HIP Loop Zero-Mat Prefill (BT=16, 0 spills, coalesced uint4/float4 Pass 2!)
// ============================================================================

__device__ __forceinline__ short16 glue_load_s16_u4(const uint16_t* ptr) {
    uint4 lo = *reinterpret_cast<const uint4*>(ptr);
    uint4 hi = *reinterpret_cast<const uint4*>(ptr + 8);
    short16 out;
    uint32_t words[8] = {lo.x, lo.y, lo.z, lo.w, hi.x, hi.y, hi.z, hi.w};
    __builtin_memcpy(&out, words, 32);
    return out;
}

__global__ __launch_bounds__(128, 6)
void hip_loop_zeromat_prefill_kernel(
    const float*    __restrict__ Dest,    // [S, T, 2304] FP32 (sd_s may be 0!)
    const uint16_t* __restrict__ Bank0,   // [T, 2304] BF16
    const uint16_t* __restrict__ Bank1,   // [T, 2304] BF16
    const float*    __restrict__ Coeffs,  // [S, T, 2] FP32
    const uint16_t* __restrict__ WdownT,  // [S, 2*2304, 64] BF16
    const uint16_t* __restrict__ Wup,     // [S, 2*2304, 64] BF16
    const uint16_t* __restrict__ NormW,   // [S, 2304] BF16
    uint16_t*       __restrict__ Out,     // [S, T, 2304] BF16
    int64_t sd_s, int64_t sd_t,
    int64_t sb0, int64_t sb1,
    int64_t sc_s, int64_t sn_s,
    int T, float norm_eps
) {
    constexpr int D = 2304;
    constexpr int BT = 16;
    const int64_t s = (int64_t)blockIdx.x;
    const int t0 = blockIdx.y * BT;
    if (t0 >= T) return;

    const int tid = threadIdx.x;
    const int warp = tid >> 5;
    const int lane = tid & 31;
    const int lane16 = lane & 15;
    const int half = lane >> 4;

    __shared__ uint16_t smem_A[16][72];    // 2.25 KiB (D -> M -> H[16][72])
    __shared__ union {
        uint16_t WT[64][72];               // 9.00 KiB (Pass 1 weight transpose)
        struct {
            float G[16][68];               // 4.25 KiB (Pass 2 coalesced gain)
            float F[16][68];               // 4.25 KiB (Pass 2 coalesced forget)
        } gf;                              // 8.50 KiB -> 11.25 KiB total!
    } smem_u;
    __shared__ float smem_c0[16];
    __shared__ float smem_c1[16];
    __shared__ float smem_rstd_d[16];
    __shared__ float smem_rstd_m[16];

    if (tid < 16) {
        int t = t0 + tid;
        if (t < T) {
            smem_c0[tid] = Coeffs[s * sc_s + (int64_t)t * 2 + 0];
            smem_c1[tid] = Coeffs[s * sc_s + (int64_t)t * 2 + 1];
        } else {
            smem_c0[tid] = 0.0f;
            smem_c1[tid] = 0.0f;
        }
    }
    __syncthreads();

    // Pass 1: Down projection [16, 2304] @ [2304, 64]
    float8 acc_d = {0.f,0.f,0.f,0.f,0.f,0.f,0.f,0.f};
    float8 acc_m = {0.f,0.f,0.f,0.f,0.f,0.f,0.f,0.f};

    const int ld_row = tid >> 3;          // 0..15
    const int ld_k   = (tid & 7) * 8;     // 0, 8, ..., 56
    const int ld_tok = t0 + ld_row;
    const bool ld_ok = (ld_tok < T);
    const float my_c0 = smem_c0[ld_row];
    const float my_c1 = smem_c1[ld_row];
    float my_sq_d = 0.0f, my_sq_m = 0.0f;

    const float* dest_row = Dest + s * sd_s + (int64_t)ld_tok * sd_t;
    const uint16_t* b0_row = Bank0 + (int64_t)ld_tok * sb0;
    const uint16_t* b1_row = Bank1 + (int64_t)ld_tok * sb1;
    const uint16_t* wd_base = WdownT + s * (int64_t)(2 * D * 64);

    const int wt_r0 = (tid & 7) * 8;
    const int wt_kp = (tid >> 3);

    auto load_w_trans = [&](int row_base) {
        #pragma unroll
        for (int rep = 0; rep < 2; ++rep) {
            int k_loc = (wt_kp + rep * 16) * 2;
            uint4 wa = *reinterpret_cast<const uint4*>(wd_base + (int64_t)(row_base + k_loc + 0) * 64 + wt_r0);
            uint4 wb = *reinterpret_cast<const uint4*>(wd_base + (int64_t)(row_base + k_loc + 1) * 64 + wt_r0);
            uint32_t aw[4] = {wa.x, wa.y, wa.z, wa.w};
            uint32_t bw[4] = {wb.x, wb.y, wb.z, wb.w};
            #pragma unroll
            for (int j = 0; j < 4; ++j) {
                *reinterpret_cast<uint32_t*>(&smem_u.WT[wt_r0 + 2 * j + 0][k_loc]) = (aw[j] & 0xFFFFu) | ((bw[j] & 0xFFFFu) << 16);
                *reinterpret_cast<uint32_t*>(&smem_u.WT[wt_r0 + 2 * j + 1][k_loc]) = (aw[j] >> 16)     | (bw[j] & 0xFFFF0000u);
            }
        }
    };

    for (int k0 = 0; k0 < D; k0 += 64) {
        uint4 d_u4 = make_uint4(0u, 0u, 0u, 0u);
        if (ld_ok) {
            float4 f0 = *reinterpret_cast<const float4*>(dest_row + k0 + ld_k);
            float4 f1 = *reinterpret_cast<const float4*>(dest_row + k0 + ld_k + 4);
            my_sq_d = fmaf(f0.x, f0.x, fmaf(f0.y, f0.y, fmaf(f0.z, f0.z, fmaf(f0.w, f0.w,
                      fmaf(f1.x, f1.x, fmaf(f1.y, f1.y, fmaf(f1.z, f1.z, fmaf(f1.w, f1.w, my_sq_d))))))));
            d_u4 = make_uint4(pack_bf16x2(f0.x, f0.y), pack_bf16x2(f0.z, f0.w),
                              pack_bf16x2(f1.x, f1.y), pack_bf16x2(f1.z, f1.w));
        }
        *reinterpret_cast<uint4*>(&smem_A[ld_row][ld_k]) = d_u4;
        load_w_trans(k0);
        __syncthreads();

        #pragma unroll
        for (int kb = 0; kb < 64; kb += 16) {
            short16 a_d = glue_load_s16_u4(&smem_A[lane16][kb]);
            short16 b_w = glue_load_s16_u4(&smem_u.WT[warp * 16 + lane16][kb]);
            acc_d = __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32(a_d, b_w, acc_d);
        }
        __syncthreads();

        uint4 m_u4 = make_uint4(0u, 0u, 0u, 0u);
        if (ld_ok) {
            uint4 u0 = *reinterpret_cast<const uint4*>(b0_row + k0 + ld_k);
            uint4 u1 = *reinterpret_cast<const uint4*>(b1_row + k0 + ld_k);
            uint32_t b0w[4] = {u0.x, u0.y, u0.z, u0.w};
            uint32_t b1w[4] = {u1.x, u1.y, u1.z, u1.w};
            uint32_t mw[4];
            #pragma unroll
            for (int j = 0; j < 4; ++j) {
                float m0 = my_c0 * bf16_to_f32((uint16_t)(b0w[j] & 0xFFFFu)) + my_c1 * bf16_to_f32((uint16_t)(b1w[j] & 0xFFFFu));
                float m1 = my_c0 * bf16_to_f32((uint16_t)(b0w[j] >> 16))     + my_c1 * bf16_to_f32((uint16_t)(b1w[j] >> 16));
                my_sq_m = fmaf(m0, m0, fmaf(m1, m1, my_sq_m));
                mw[j] = pack_bf16x2(m0, m1);
            }
            m_u4 = make_uint4(mw[0], mw[1], mw[2], mw[3]);
        }
        *reinterpret_cast<uint4*>(&smem_A[ld_row][ld_k]) = m_u4;
        load_w_trans(D + k0);
        __syncthreads();

        #pragma unroll
        for (int kb = 0; kb < 64; kb += 16) {
            short16 a_m = glue_load_s16_u4(&smem_A[lane16][kb]);
            short16 b_w = glue_load_s16_u4(&smem_u.WT[warp * 16 + lane16][kb]);
            acc_m = __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32(a_m, b_w, acc_m);
        }
        __syncthreads();
    }

    #pragma unroll
    for (int off = 4; off > 0; off >>= 1) {
        my_sq_d += __shfl_xor(my_sq_d, off, 8);
        my_sq_m += __shfl_xor(my_sq_m, off, 8);
    }
    if ((tid & 7) == 0) {
        smem_rstd_d[ld_row] = rsqrtf(my_sq_d * (1.0f / (float)D) + 1e-6f);
        smem_rstd_m[ld_row] = rsqrtf(my_sq_m * (1.0f / (float)D) + 1e-6f);
    }
    __syncthreads();

    #pragma unroll
    for (int r = 0; r < 8; ++r) {
        int t_loc = half + 2 * r;
        smem_A[t_loc][warp * 16 + lane16] =
            f32_to_bf16_rtne(acc_d[r] * smem_rstd_d[t_loc] + acc_m[r] * smem_rstd_m[t_loc]);
    }
    __syncthreads();

    // Pass 2: Up projection + coalesced float4/uint4 blend & store to L2-hot Out!
    const uint16_t* wu_base = Wup + s * (int64_t)(2 * D * 64);
    uint16_t* out_s = Out + s * (int64_t)T * D;
    uint16_t* o_row = out_s + (int64_t)ld_tok * D;
    float my_sq_b = 0.0f;

    for (int k0 = 0; k0 < D; k0 += 64) {
        int k_col = k0 + warp * 16 + lane16;
        const uint16_t* wg_row = wu_base + (int64_t)k_col * 64;
        const uint16_t* wf_row = wu_base + (int64_t)(D + k_col) * 64;

        float8 g_acc = {0.f,0.f,0.f,0.f,0.f,0.f,0.f,0.f};
        float8 f_acc = {0.f,0.f,0.f,0.f,0.f,0.f,0.f,0.f};
        #pragma unroll
        for (int rb = 0; rb < 4; ++rb) {
            short16 ah = glue_load_s16_u4(&smem_A[lane16][rb * 16]);
            short16 bg = glue_load_s16_u4(wg_row + rb * 16);
            short16 bf = glue_load_s16_u4(wf_row + rb * 16);
            g_acc = __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32(ah, bg, g_acc);
            f_acc = __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32(ah, bf, f_acc);
        }

        int sm_col = warp * 16 + lane16;
        #pragma unroll
        for (int r = 0; r < 8; ++r) {
            int t_loc = half + 2 * r;
            smem_u.gf.G[t_loc][sm_col] = fast_tanh_hw(g_acc[r]);
            smem_u.gf.F[t_loc][sm_col] = fast_tanh_hw(f_acc[r]);
        }
        __syncthreads();

        if (ld_ok) {
            float4 df0 = *reinterpret_cast<const float4*>(dest_row + k0 + ld_k);
            float4 df1 = *reinterpret_cast<const float4*>(dest_row + k0 + ld_k + 4);
            uint4 u0 = *reinterpret_cast<const uint4*>(b0_row + k0 + ld_k);
            uint4 u1 = *reinterpret_cast<const uint4*>(b1_row + k0 + ld_k);
            float4 gf0 = *reinterpret_cast<const float4*>(&smem_u.gf.G[ld_row][ld_k]);
            float4 gf1 = *reinterpret_cast<const float4*>(&smem_u.gf.G[ld_row][ld_k + 4]);
            float4 ff0 = *reinterpret_cast<const float4*>(&smem_u.gf.F[ld_row][ld_k]);
            float4 ff1 = *reinterpret_cast<const float4*>(&smem_u.gf.F[ld_row][ld_k + 4]);
            float dv[8] = {df0.x, df0.y, df0.z, df0.w, df1.x, df1.y, df1.z, df1.w};
            float gv[8] = {gf0.x, gf0.y, gf0.z, gf0.w, gf1.x, gf1.y, gf1.z, gf1.w};
            float fv[8] = {ff0.x, ff0.y, ff0.z, ff0.w, ff1.x, ff1.y, ff1.z, ff1.w};
            uint32_t b0w[4] = {u0.x, u0.y, u0.z, u0.w};
            uint32_t b1w[4] = {u1.x, u1.y, u1.z, u1.w};
            uint32_t ow[4];
            #pragma unroll
            for (int j = 0; j < 4; ++j) {
                float m0 = my_c0 * bf16_to_f32((uint16_t)(b0w[j] & 0xFFFFu)) + my_c1 * bf16_to_f32((uint16_t)(b1w[j] & 0xFFFFu));
                float m1 = my_c0 * bf16_to_f32((uint16_t)(b0w[j] >> 16))     + my_c1 * bf16_to_f32((uint16_t)(b1w[j] >> 16));
                float y0 = (1.0f - fv[2 * j + 0]) * dv[2 * j + 0] + gv[2 * j + 0] * m0;
                float y1 = (1.0f - fv[2 * j + 1]) * dv[2 * j + 1] + gv[2 * j + 1] * m1;
                my_sq_b = fmaf(y0, y0, fmaf(y1, y1, my_sq_b));
                ow[j] = pack_bf16x2(y0, y1);
            }
            *reinterpret_cast<uint4*>(o_row + k0 + ld_k) = make_uint4(ow[0], ow[1], ow[2], ow[3]);
        }
        __syncthreads();
    }

    #pragma unroll
    for (int off = 4; off > 0; off >>= 1) my_sq_b += __shfl_xor(my_sq_b, off, 8);
    if ((tid & 7) == 0) smem_rstd_d[ld_row] = rsqrtf(my_sq_b * (1.0f / (float)D) + norm_eps);
    __syncthreads();

    // Pass 3: Reload L2-hot Out and apply RMSNorm * NormW[s, k]
    if (ld_ok) {
        const float rstd = smem_rstd_d[ld_row];
        const uint16_t* nw_row = NormW + s * sn_s;
        for (int k0 = 0; k0 < D; k0 += 64) {
            uint4 bu = *reinterpret_cast<const uint4*>(o_row + k0 + ld_k);
            uint4 nu = *reinterpret_cast<const uint4*>(nw_row + k0 + ld_k);
            uint32_t bw[4] = {bu.x, bu.y, bu.z, bu.w};
            uint32_t nw[4] = {nu.x, nu.y, nu.z, nu.w};
            uint32_t ow[4];
            #pragma unroll
            for (int j = 0; j < 4; ++j) {
                float y0 = bf16_to_f32((uint16_t)(bw[j] & 0xFFFFu)) * rstd * bf16_to_f32((uint16_t)(nw[j] & 0xFFFFu));
                float y1 = bf16_to_f32((uint16_t)(bw[j] >> 16))     * rstd * bf16_to_f32((uint16_t)(nw[j] >> 16));
                ow[j] = pack_bf16x2(y0, y1);
            }
            *reinterpret_cast<uint4*>(o_row + k0 + ld_k) = make_uint4(ow[0], ow[1], ow[2], ow[3]);
        }
    }
}

// ============================================================================
// C3: Native HIP Block Attention Residuals (D=2304, 288 threads x 8 elems)
//     ZERO SCRATCH MEMORY! Explicit scalar registers for 1-pass (N_STACK <= 2)
//     and LDS softmax buffer for 2-pass (N_STACK >= 3).
// ============================================================================

template<bool IS_F32>
__device__ __forceinline__ void load_d8(const void* base, int64_t elem_off, float dst[8]) {
    if constexpr (IS_F32) {
        const float* p = reinterpret_cast<const float*>(base) + elem_off;
        float4 a = *reinterpret_cast<const float4*>(p);
        float4 b = *reinterpret_cast<const float4*>(p + 4);
        dst[0] = a.x; dst[1] = a.y; dst[2] = a.z; dst[3] = a.w;
        dst[4] = b.x; dst[5] = b.y; dst[6] = b.z; dst[7] = b.w;
    } else {
        const uint16_t* p = reinterpret_cast<const uint16_t*>(base) + elem_off;
        uint4 u = *reinterpret_cast<const uint4*>(p);
        dst[0] = bf16_to_f32((uint16_t)(u.x & 0xFFFFu)); dst[1] = bf16_to_f32((uint16_t)(u.x >> 16));
        dst[2] = bf16_to_f32((uint16_t)(u.y & 0xFFFFu)); dst[3] = bf16_to_f32((uint16_t)(u.y >> 16));
        dst[4] = bf16_to_f32((uint16_t)(u.z & 0xFFFFu)); dst[5] = bf16_to_f32((uint16_t)(u.z >> 16));
        dst[6] = bf16_to_f32((uint16_t)(u.w & 0xFFFFu)); dst[7] = bf16_to_f32((uint16_t)(u.w >> 16));
    }
}

template<bool OUT_F32>
__device__ __forceinline__ void store_d8(void* base, int64_t elem_off, const float src[8]) {
    if constexpr (OUT_F32) {
        float* p = reinterpret_cast<float*>(base) + elem_off;
        *reinterpret_cast<float4*>(p)     = make_float4(src[0], src[1], src[2], src[3]);
        *reinterpret_cast<float4*>(p + 4) = make_float4(src[4], src[5], src[6], src[7]);
    } else {
        uint16_t* p = reinterpret_cast<uint16_t*>(base) + elem_off;
        *reinterpret_cast<uint4*>(p) = make_uint4(
            pack_bf16x2(src[0], src[1]), pack_bf16x2(src[2], src[3]),
            pack_bf16x2(src[4], src[5]), pack_bf16x2(src[6], src[7]));
    }
}

template<int Q_NUM, bool HAS_ADDEND, bool NORM_VALS, bool APPLY_NORM, bool SRC_F32, bool OUT_F32, bool NW_F32>
__global__ __launch_bounds__(288, 4)
void hip_attn_res_prefill_kernel(
    const void*  __restrict__ Stack,
    const void*  __restrict__ Last,
    const void*  __restrict__ Addend,
    const float* __restrict__ Queries,
    const void*  __restrict__ NormW,
    void*        __restrict__ Out,
    int64_t s_stk, int64_t s_stk_r, int64_t s_last, int64_t s_add,
    int64_t s_q, int64_t s_nw,
    int T, int N_STACK, float eps, float norm_eps
) {
    constexpr int D = 2304;
    constexpr int MAX_B = 18;
    const int t = blockIdx.x;
    if (t >= T) return;
    const int tid = threadIdx.x;
    const int warp = tid >> 5;
    const int lane = tid & 31;
    const int c0 = tid * 8;
    const int nb = N_STACK + 1;

    float qv[Q_NUM][8];
    #pragma unroll
    for (int qi = 0; qi < Q_NUM; ++qi) {
        load_d8<true>(Queries, (int64_t)qi * s_q + c0, qv[qi]);
    }

    auto load_blk = [&](int b, float dst[8]) {
        if (b < N_STACK) {
            load_d8<SRC_F32>(Stack, (int64_t)b * s_stk + (int64_t)t * s_stk_r + c0, dst);
        } else {
            load_d8<SRC_F32>(Last, (int64_t)t * s_last + c0, dst);
            if constexpr (HAS_ADDEND) {
                float av[8];
                load_d8<SRC_F32>(Addend, (int64_t)t * s_add + c0, av);
                #pragma unroll
                for (int j = 0; j < 8; ++j) dst[j] += av[j];
            }
        }
    };

    __shared__ float sm_red[9][1 + Q_NUM];
    __shared__ float sm_ss[MAX_B];
    __shared__ float sm_dot[Q_NUM][MAX_B];
    __shared__ float sm_w[Q_NUM][MAX_B];
    __shared__ float sm_ex[MAX_B];
    __shared__ float sm_norm[9];
    __shared__ float sm_rn;

    auto reduce_blk = [&](int b, const float v[8]) {
        float ss = 0.0f;
        float dt[Q_NUM];
        #pragma unroll
        for (int qi = 0; qi < Q_NUM; ++qi) dt[qi] = 0.0f;
        #pragma unroll
        for (int j = 0; j < 8; ++j) {
            float x = v[j];
            ss = fmaf(x, x, ss);
            #pragma unroll
            for (int qi = 0; qi < Q_NUM; ++qi) dt[qi] = fmaf(qv[qi][j], x, dt[qi]);
        }
        #pragma unroll
        for (int off = 16; off > 0; off >>= 1) {
            ss += __shfl_xor(ss, off, 32);
            #pragma unroll
            for (int qi = 0; qi < Q_NUM; ++qi) dt[qi] += __shfl_xor(dt[qi], off, 32);
        }
        if (lane == 0) {
            sm_red[warp][0] = ss;
            #pragma unroll
            for (int qi = 0; qi < Q_NUM; ++qi) sm_red[warp][1 + qi] = dt[qi];
        }
        __syncthreads();
        if (tid == 0) {
            float tss = 0.0f;
            float tdt[Q_NUM];
            #pragma unroll
            for (int qi = 0; qi < Q_NUM; ++qi) tdt[qi] = 0.0f;
            #pragma unroll
            for (int w = 0; w < 9; ++w) {
                tss += sm_red[w][0];
                #pragma unroll
                for (int qi = 0; qi < Q_NUM; ++qi) tdt[qi] += sm_red[w][1 + qi];
            }
            float r = rsqrtf(tss * (1.0f / (float)D) + eps);
            sm_ss[b] = r;
            #pragma unroll
            for (int qi = 0; qi < Q_NUM; ++qi) sm_dot[qi][b] = tdt[qi] * r;
        }
        __syncthreads();
    };

    auto finish_weights = [&]() {
        if (tid == 0) {
            #pragma unroll
            for (int qi = 0; qi < Q_NUM; ++qi) {
                float mx = -INFINITY;
                for (int b = 0; b < nb; ++b) mx = fmaxf(mx, sm_dot[qi][b]);
                float sum_e = 0.0f;
                for (int b = 0; b < nb; ++b) {
                    float e = expf(sm_dot[qi][b] - mx);
                    sm_ex[b] = e;
                    sum_e += e;
                }
                float inv_e = 1.0f / sum_e;
                for (int b = 0; b < nb; ++b) {
                    float p = sm_ex[b] * inv_e;
                    if constexpr (NORM_VALS) p *= sm_ss[b];
                    sm_w[qi][b] = p;
                }
            }
        }
        __syncthreads();
    };

    auto apply_norm_and_store = [&](int qi, float mix[8]) {
        if constexpr (APPLY_NORM) {
            float mss = 0.0f;
            #pragma unroll
            for (int j = 0; j < 8; ++j) mss = fmaf(mix[j], mix[j], mss);
            #pragma unroll
            for (int off = 16; off > 0; off >>= 1) mss += __shfl_xor(mss, off, 32);
            if (lane == 0) sm_norm[warp] = mss;
            __syncthreads();
            if (tid == 0) {
                float tot = 0.0f;
                #pragma unroll
                for (int w = 0; w < 9; ++w) tot += sm_norm[w];
                sm_rn = rsqrtf(tot * (1.0f / (float)D) + norm_eps);
            }
            __syncthreads();
            float rn = sm_rn;
            __syncthreads();
            float nw[8];
            load_d8<NW_F32>(NormW, (int64_t)qi * s_nw + c0, nw);
            #pragma unroll
            for (int j = 0; j < 8; ++j) mix[j] = mix[j] * rn * nw[j];
        }
        store_d8<OUT_F32>(Out, ((int64_t)qi * T + t) * D + c0, mix);
    };

    if constexpr (!SRC_F32) {
        // ---- PACKED-UINT4 1-PASS REGISTER CACHE (N_STACK <= 12 -> 100% 1-PASS DRAM READ, 0 SCRATCH!) ----
        auto load_u4_stk = [&](int b) -> uint4 {
            const uint16_t* p = reinterpret_cast<const uint16_t*>(Stack)
                              + (int64_t)b * s_stk + (int64_t)t * s_stk_r + c0;
            return *reinterpret_cast<const uint4*>(p);
        };
        auto unpack_u4 = [](uint4 u, float dst[8]) {
            dst[0] = bf16_to_f32((uint16_t)(u.x & 0xFFFFu)); dst[1] = bf16_to_f32((uint16_t)(u.x >> 16));
            dst[2] = bf16_to_f32((uint16_t)(u.y & 0xFFFFu)); dst[3] = bf16_to_f32((uint16_t)(u.y >> 16));
            dst[4] = bf16_to_f32((uint16_t)(u.z & 0xFFFFu)); dst[5] = bf16_to_f32((uint16_t)(u.z >> 16));
            dst[6] = bf16_to_f32((uint16_t)(u.w & 0xFFFFu)); dst[7] = bf16_to_f32((uint16_t)(u.w >> 16));
        };

        uint4 s0 = {}, s1 = {}, s2 = {}, s3 = {}, s4 = {}, s5 = {};
        uint4 s6 = {}, s7 = {}, s8 = {}, s9 = {}, s10 = {}, s11 = {};
        float tmp[8], lv[8];

        #define P1_STK(IDX, REG) do { if (IDX < N_STACK) { REG = load_u4_stk(IDX); unpack_u4(REG, tmp); reduce_blk(IDX, tmp); } } while (0)
        P1_STK(0, s0); P1_STK(1, s1); P1_STK(2, s2);  P1_STK(3, s3);
        P1_STK(4, s4); P1_STK(5, s5); P1_STK(6, s6);  P1_STK(7, s7);
        P1_STK(8, s8); P1_STK(9, s9); P1_STK(10, s10); P1_STK(11, s11);
        #undef P1_STK
        for (int b = 12; b < N_STACK; ++b) {
            load_blk(b, tmp);
            reduce_blk(b, tmp);
        }
        load_blk(N_STACK, lv);
        reduce_blk(N_STACK, lv);
        finish_weights();

        float mix0[8] = {0,0,0,0,0,0,0,0};
        float mix1[8] = {0,0,0,0,0,0,0,0};
        float mix2[8] = {0,0,0,0,0,0,0,0};
        float mix3[8] = {0,0,0,0,0,0,0,0};
        auto accum_v = [&](int b, const float v[8]) {
            float w0 = sm_w[0][b];
            #pragma unroll
            for (int j = 0; j < 8; ++j) mix0[j] = fmaf(w0, v[j], mix0[j]);
            if constexpr (Q_NUM >= 2) {
                float w1 = sm_w[1][b];
                #pragma unroll
                for (int j = 0; j < 8; ++j) mix1[j] = fmaf(w1, v[j], mix1[j]);
            }
            if constexpr (Q_NUM >= 3) {
                float w2 = sm_w[2][b];
                #pragma unroll
                for (int j = 0; j < 8; ++j) mix2[j] = fmaf(w2, v[j], mix2[j]);
            }
            if constexpr (Q_NUM >= 4) {
                float w3 = sm_w[3][b];
                #pragma unroll
                for (int j = 0; j < 8; ++j) mix3[j] = fmaf(w3, v[j], mix3[j]);
            }
        };
        #define P2_STK(IDX, REG) do { if (IDX < N_STACK) { unpack_u4(REG, tmp); accum_v(IDX, tmp); } } while (0)
        P2_STK(0, s0); P2_STK(1, s1); P2_STK(2, s2);  P2_STK(3, s3);
        P2_STK(4, s4); P2_STK(5, s5); P2_STK(6, s6);  P2_STK(7, s7);
        P2_STK(8, s8); P2_STK(9, s9); P2_STK(10, s10); P2_STK(11, s11);
        #undef P2_STK
        for (int b = 12; b < N_STACK; ++b) {
            load_blk(b, tmp);
            accum_v(b, tmp);
        }
        accum_v(N_STACK, lv);

        apply_norm_and_store(0, mix0);
        if constexpr (Q_NUM >= 2) apply_norm_and_store(1, mix1);
        if constexpr (Q_NUM >= 3) apply_norm_and_store(2, mix2);
        if constexpr (Q_NUM >= 4) apply_norm_and_store(3, mix3);
    } else {
        // ---- FP32 SOURCE STREAMING PATH: 0 SCRATCH! ----
        for (int b = 0; b < nb; ++b) {
            float v[8];
            load_blk(b, v);
            reduce_blk(b, v);
        }
        finish_weights();

        float mix0[8] = {0,0,0,0,0,0,0,0};
        float mix1[8] = {0,0,0,0,0,0,0,0};
        float mix2[8] = {0,0,0,0,0,0,0,0};
        float mix3[8] = {0,0,0,0,0,0,0,0};
        for (int b = 0; b < nb; ++b) {
            float v[8];
            load_blk(b, v);
            float w0 = sm_w[0][b];
            #pragma unroll
            for (int j = 0; j < 8; ++j) mix0[j] = fmaf(w0, v[j], mix0[j]);
            if constexpr (Q_NUM >= 2) {
                float w1 = sm_w[1][b];
                #pragma unroll
                for (int j = 0; j < 8; ++j) mix1[j] = fmaf(w1, v[j], mix1[j]);
            }
            if constexpr (Q_NUM >= 3) {
                float w2 = sm_w[2][b];
                #pragma unroll
                for (int j = 0; j < 8; ++j) mix2[j] = fmaf(w2, v[j], mix2[j]);
            }
            if constexpr (Q_NUM >= 4) {
                float w3 = sm_w[3][b];
                #pragma unroll
                for (int j = 0; j < 8; ++j) mix3[j] = fmaf(w3, v[j], mix3[j]);
            }
        }
        apply_norm_and_store(0, mix0);
        if constexpr (Q_NUM >= 2) apply_norm_and_store(1, mix1);
        if constexpr (Q_NUM >= 3) apply_norm_and_store(2, mix2);
        if constexpr (Q_NUM >= 4) apply_norm_and_store(3, mix3);
    }
}

// ============================================================================
// C4: Two-Output BF16-WMMA Prefill Router (BN=128 single-read X -> FP32)
// ============================================================================

__global__ __launch_bounds__(128, 6)
void hip_router_prefill_kernel(
    const uint16_t* __restrict__ X,
    const uint16_t* __restrict__ W,
    const float*    __restrict__ Bias,
    float*          __restrict__ Scores,
    float*          __restrict__ Biased,
    int T, int64_t stride_x
) {
    constexpr int K = 2304;
    constexpr int N = 128;
    const int t0 = blockIdx.x * 64;
    if (t0 >= T) return;

    const int tid = threadIdx.x;
    const int warp = tid >> 5;
    const int lane = tid & 31;
    const int lane16 = lane & 15;
    const int half = lane >> 4;

    __shared__ uint16_t smem_X[64][40];  //  5.00 KiB
    __shared__ uint16_t smem_W[128][40]; // 10.00 KiB -> 15.00 KiB total

    float8 acc[8];
    #pragma unroll
    for (int nt = 0; nt < 8; ++nt) acc[nt] = {0.f,0.f,0.f,0.f,0.f,0.f,0.f,0.f};

    const int ld_r0 = tid >> 2;         // 0..31
    const int ld_k  = (tid & 3) * 8;    // 0, 8, 16, 24

    for (int k0 = 0; k0 < K; k0 += 32) {
        #pragma unroll
        for (int rep = 0; rep < 2; ++rep) {
            int r = ld_r0 + rep * 32;
            uint4 xu = (t0 + r < T)
                ? *reinterpret_cast<const uint4*>(X + (int64_t)(t0 + r) * stride_x + k0 + ld_k)
                : make_uint4(0u, 0u, 0u, 0u);
            *reinterpret_cast<uint4*>(&smem_X[r][ld_k]) = xu;
        }
        #pragma unroll
        for (int rep = 0; rep < 4; ++rep) {
            int r = ld_r0 + rep * 32;
            *reinterpret_cast<uint4*>(&smem_W[r][ld_k]) =
                *reinterpret_cast<const uint4*>(W + (int64_t)r * K + k0 + ld_k);
        }
        __syncthreads();

        #pragma unroll
        for (int kd = 0; kd < 32; kd += 16) {
            short16 a_x = glue_load_s16_u4(&smem_X[warp * 16 + lane16][kd]);
            #pragma unroll
            for (int nt = 0; nt < 8; ++nt) {
                short16 b_w = glue_load_s16_u4(&smem_W[nt * 16 + lane16][kd]);
                acc[nt] = __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32(a_x, b_w, acc[nt]);
            }
        }
        __syncthreads();
    }

    #pragma unroll
    for (int nt = 0; nt < 8; ++nt) {
        int col = nt * 16 + lane16;
        float b_val = Bias[col];
        #pragma unroll
        for (int r = 0; r < 8; ++r) {
            int tok = t0 + warp * 16 + half + 2 * r;
            if (tok < T) {
                float sc = 1.0f / (1.0f + expf(-acc[nt][r]));
                int64_t off = (int64_t)tok * N + col;
                Scores[off] = sc;
                Biased[off] = sc + b_val;
            }
        }
    }
}

extern "C" {

void launch_q4_coeffs(
    const void* B0, const void* B1, const void* Q, void* C,
    int64_t sb0, int64_t sb1, int64_t sq, int T, float eps, hipStream_t stream
) {
    hipLaunchKernelGGL(
        hip_attn_res_q4_coeffs_kernel, dim3(T), dim3(288), 0, stream,
        (const uint16_t*)B0, (const uint16_t*)B1, (const float*)Q, (float*)C,
        sb0, sb1, sq, T, eps);
}

void launch_hip_sconv_prefill(
    const void* X, const void* W, void* Cache,
    const void* Pos, const void* BT_ptr, const void* Seq, const void* Slot, const void* QStart,
    const void* Addend, const void* Residual, void* Out,
    int T, int D, int H, int stream_off, int stream_w,
    int64_t sx, int64_t sw0, int64_t sw1,
    int64_t scb, int64_t sch, int64_t scn,
    int64_t sbt, int max_blocks, int N_ring, float scale,
    int has_addend, int has_res, int add_f32, int res_f32, int out_f32,
    int pos_i64, hipStream_t stream
) {
    if (T <= 0 || D <= 0) return;
    dim3 grid((T + 63) / 64, (D + 511) / 512);
    dim3 block(64);
#define LAUNCH_SC(HA, HR, AF, RF, OF) \
    hipLaunchKernelGGL((hip_sconv_prefill_kernel<HA, HR, AF, RF, OF>), grid, block, 0, stream, \
        (const uint16_t*)X, (const uint16_t*)W, (uint16_t*)Cache, \
        Pos, (const int32_t*)BT_ptr, (const int32_t*)Seq, \
        (const int64_t*)Slot, (const int32_t*)QStart, Addend, Residual, Out, \
        T, D, H, stream_off, stream_w, sx, sw0, sw1, scb, sch, scn, sbt, max_blocks, N_ring, scale, pos_i64)
    if (!has_addend) {
        if (out_f32) LAUNCH_SC(false, false, false, false, true);
        else         LAUNCH_SC(false, false, false, false, false);
    } else if (!has_res) {
        if (add_f32 && out_f32)       LAUNCH_SC(true, false, true,  false, true);
        else if (add_f32 && !out_f32) LAUNCH_SC(true, false, true,  false, false);
        else if (!add_f32 && out_f32) LAUNCH_SC(true, false, false, false, true);
        else                          LAUNCH_SC(true, false, false, false, false);
    } else {
        switch ((add_f32 ? 4 : 0) | (res_f32 ? 2 : 0) | (out_f32 ? 1 : 0)) {
            case 0: LAUNCH_SC(true, true, false, false, false); break;
            case 1: LAUNCH_SC(true, true, false, false, true);  break;
            case 2: LAUNCH_SC(true, true, false, true,  false); break;
            case 3: LAUNCH_SC(true, true, false, true,  true);  break;
            case 4: LAUNCH_SC(true, true, true,  false, false); break;
            case 5: LAUNCH_SC(true, true, true,  false, true);  break;
            case 6: LAUNCH_SC(true, true, true,  true,  false); break;
            case 7: LAUNCH_SC(true, true, true,  true,  true);  break;
        }
    }
#undef LAUNCH_SC
}

void launch_hip_loop_zeromat_prefill(
    const void* Dest, const void* Bank0, const void* Bank1, const void* Coeffs,
    const void* WdownT, const void* Wup, const void* NormW, void* Out,
    int64_t sd_s, int64_t sd_t, int64_t sb0, int64_t sb1, int64_t sc_s, int64_t sn_s,
    int T, int streams, float norm_eps, hipStream_t stream
) {
    if (T <= 0 || streams <= 0) return;
    dim3 grid(streams, (T + 15) / 16);
    dim3 block(128);
    hipLaunchKernelGGL(hip_loop_zeromat_prefill_kernel, grid, block, 0, stream,
        (const float*)Dest, (const uint16_t*)Bank0, (const uint16_t*)Bank1, (const float*)Coeffs,
        (const uint16_t*)WdownT, (const uint16_t*)Wup, (const uint16_t*)NormW, (uint16_t*)Out,
        sd_s, sd_t, sb0, sb1, sc_s, sn_s, T, norm_eps);
}

void launch_hip_attn_res_prefill(
    const void* Stack, const void* Last, const void* Addend,
    const void* Queries, const void* NormW, void* Out,
    int64_t s_stk, int64_t s_stk_r, int64_t s_last, int64_t s_add,
    int64_t s_q, int64_t s_nw,
    int T, int Q, int N_STACK,
    int has_addend, int norm_vals, int apply_norm, int src_f32, int out_f32, int nw_f32,
    float eps, float norm_eps, hipStream_t stream
) {
    if (T <= 0 || N_STACK < 1 || N_STACK >= 18) return;
    dim3 grid(T);
    dim3 block(288);
#define LAUNCH_AR(QN, HA, NV, AN, SF, OF, NF) \
    hipLaunchKernelGGL((hip_attn_res_prefill_kernel<QN, HA, NV, AN, SF, OF, NF>), grid, block, 0, stream, \
        Stack, Last, Addend, (const float*)Queries, NormW, Out, \
        s_stk, s_stk_r, s_last, s_add, s_q, s_nw, T, N_STACK, eps, norm_eps)
#define DISPATCH_AR_Q(QN) do { \
    int m = (has_addend ? 8 : 0) | (norm_vals ? 4 : 0) | (apply_norm ? 2 : 0) | (out_f32 ? 1 : 0); \
    if (!src_f32 && !nw_f32) { \
        switch (m) { \
            case 0:  LAUNCH_AR(QN, false, false, false, false, false, false); break; \
            case 1:  LAUNCH_AR(QN, false, false, false, false, true,  false); break; \
            case 2:  LAUNCH_AR(QN, false, false, true,  false, false, false); break; \
            case 3:  LAUNCH_AR(QN, false, false, true,  false, true,  false); break; \
            case 4:  LAUNCH_AR(QN, false, true,  false, false, false, false); break; \
            case 5:  LAUNCH_AR(QN, false, true,  false, false, true,  false); break; \
            case 6:  LAUNCH_AR(QN, false, true,  true,  false, false, false); break; \
            case 7:  LAUNCH_AR(QN, false, true,  true,  false, true,  false); break; \
            case 8:  LAUNCH_AR(QN, true,  false, false, false, false, false); break; \
            case 9:  LAUNCH_AR(QN, true,  false, false, false, true,  false); break; \
            case 10: LAUNCH_AR(QN, true,  false, true,  false, false, false); break; \
            case 11: LAUNCH_AR(QN, true,  false, true,  false, true,  false); break; \
            case 12: LAUNCH_AR(QN, true,  true,  false, false, false, false); break; \
            case 13: LAUNCH_AR(QN, true,  true,  false, false, true,  false); break; \
            case 14: LAUNCH_AR(QN, true,  true,  true,  false, false, false); break; \
            case 15: LAUNCH_AR(QN, true,  true,  true,  false, true,  false); break; \
        } \
    } else if (src_f32 && !nw_f32) { \
        switch (m) { \
            case 1:  LAUNCH_AR(QN, false, false, false, true,  true,  false); break; \
            case 3:  LAUNCH_AR(QN, false, false, true,  true,  true,  false); break; \
            case 5:  LAUNCH_AR(QN, false, true,  false, true,  true,  false); break; \
            case 7:  LAUNCH_AR(QN, false, true,  true,  true,  true,  false); break; \
            case 9:  LAUNCH_AR(QN, true,  false, false, true,  true,  false); break; \
            case 11: LAUNCH_AR(QN, true,  false, true,  true,  true,  false); break; \
            case 13: LAUNCH_AR(QN, true,  true,  false, true,  true,  false); break; \
            case 15: LAUNCH_AR(QN, true,  true,  true,  true,  true,  false); break; \
            default: LAUNCH_AR(QN, false, false, false, true,  true,  false); break; \
        } \
    } \
} while (0)
    if (Q == 1)      DISPATCH_AR_Q(1);
    else if (Q == 3) DISPATCH_AR_Q(3);
    else if (Q == 4) DISPATCH_AR_Q(4);
#undef DISPATCH_AR_Q
#undef LAUNCH_AR
}

void launch_hip_router_prefill(
    const void* X, const void* W, const void* Bias,
    void* Scores, void* Biased, int T, int64_t stride_x, hipStream_t stream
) {
    if (T <= 0) return;
    dim3 grid((T + 63) / 64);
    dim3 block(128);
    hipLaunchKernelGGL(hip_router_prefill_kernel, grid, block, 0, stream,
        (const uint16_t*)X, (const uint16_t*)W, (const float*)Bias,
        (float*)Scores, (float*)Biased, T, stride_x);
}

}
