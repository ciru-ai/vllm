// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 CIRU AI.
// Native RDNA 3.5 (gfx1151) decode glue & DSpark draft kernels for Inkling 2 Mini.

#include "hip_common.cuh"
#include <type_traits>

extern "C" int launch_hip_reduce_splits(
    const void* Partial, void* Out, int64_t total_elems, int splits, int out_fp32, float soft_cap, void* stream_ptr
);

// ============================================================================
// 1. SMALL-M STATIC SHORTCONV DECODE (BT=8, W=4, 1 ch/thr, 32-thr warp, aliasing-safe)
// ============================================================================
template<bool HAS_ADDEND, bool HAS_RES, bool ADD_F32, bool RES_F32, bool OUT_F32>
__global__ __launch_bounds__(32, 16)
void hip_sconv_decode_kernel(
    const uint16_t* __restrict__ X,
    const uint16_t* __restrict__ Weight,
    uint16_t*       __restrict__ Cache,
    const void*     __restrict__ Pos,
    const int32_t*  __restrict__ BlockTable,
    const int32_t*  __restrict__ SeqIdx,
    const int64_t*  __restrict__ Slot,
    const int32_t*  __restrict__ QStart,
    const void*     __restrict__ Addend,
    const void*                  Residual,  // may alias Out!
    void*                        Out,       // may alias Residual!
    int T, int D, int H, int stream_off, int stream_w,
    int64_t sx, int64_t sw0, int64_t sw1,
    int64_t scb, int64_t sch, int64_t scn,
    int64_t sbt, int max_blocks, int N_ring, float scale, int pos_i64
) {
    constexpr int BT = 8;
    const int t0 = (int)blockIdx.x * BT;
    const int c  = (int)blockIdx.y * 32 + (int)threadIdx.x;
    if (t0 >= T || c >= D) return;
    const int t_end = min(t0 + BT, T);

    const float w0 = bf16_to_f32(Weight[(int64_t)c * sw0 + 0 * sw1]);
    const float w1 = bf16_to_f32(Weight[(int64_t)c * sw0 + 1 * sw1]);
    const float w2 = bf16_to_f32(Weight[(int64_t)c * sw0 + 2 * sw1]);
    const float w3 = bf16_to_f32(Weight[(int64_t)c * sw0 + 3 * sw1]);
    const int64_t c_col = (int64_t)(c / stream_w) * sch + stream_off + (c % stream_w);

    auto load_tap = [&](int t_cur, int pos_cur, int req_cur, int qs_cur, int lag) -> float {
        int src = pos_cur - lag;
        if (src < 0) return 0.0f;
        int row = t_cur - lag;
        if (row >= qs_cur) {
            return bf16_to_f32(X[(int64_t)row * sx + c]);
        }
        int lblk = min(src / N_ring, max_blocks - 1);
        int64_t blk = (int64_t)BlockTable[(int64_t)req_cur * sbt + lblk];
        int64_t base = blk * scb + (int64_t)(src % N_ring) * scn;
        return bf16_to_f32(Cache[base + c_col]);
    };

    float h0 = 0.0f, h1 = 0.0f, h2 = 0.0f;
    bool need_init = true;
    for (int t = t0; t < t_end; ++t) {
        const int qs  = QStart[t];
        const int pos = pos_i64 ? (int)((const int64_t*)Pos)[t] : ((const int32_t*)Pos)[t];
        const int req = SeqIdx[t];
        const int64_t slot = Slot[t];

        if (slot >= 0) {
            if (need_init || t == qs) {
                h0 = load_tap(t, pos, req, qs, 3);
                h1 = load_tap(t, pos, req, qs, 2);
                h2 = load_tap(t, pos, req, qs, 1);
                need_init = false;
            }
        } else {
            need_init = true;
        }

        const uint16_t cur_bits = X[(int64_t)t * sx + c];
        const float cur = bf16_to_f32(cur_bits);
        if (slot >= 0) {
            const int64_t cbase = (slot / N_ring) * scb + (slot % N_ring) * scn;
            Cache[cbase + c_col] = cur_bits;
        }

        float acc = 0.0f;
        if (slot >= 0) {
            acc = h0 * w0;
            acc = fmaf(h1, w1, acc);
            acc = fmaf(h2, w2, acc);
            acc = fmaf(cur, w3, acc);
        }
        float yi = (acc + cur) * scale;
        if constexpr (!OUT_F32) yi = bf16_round_f32(yi);
        h0 = h1; h1 = h2; h2 = cur;

        const int64_t out_off = (int64_t)t * D + c;
        if constexpr (HAS_ADDEND) {
            float av = ADD_F32
                ? reinterpret_cast<const float*>(Addend)[out_off]
                : bf16_to_f32(reinterpret_cast<const uint16_t*>(Addend)[out_off]);
            if constexpr (HAS_RES) {
                float rv = RES_F32
                    ? reinterpret_cast<const float*>(Residual)[out_off]
                    : bf16_to_f32(reinterpret_cast<const uint16_t*>(Residual)[out_off]);
                av = rv + av;
            }
            yi = av + yi;
        }

        if constexpr (OUT_F32) {
            reinterpret_cast<float*>(Out)[out_off] = yi;
        } else {
            reinterpret_cast<uint16_t*>(Out)[out_off] = f32_to_bf16_rtne(yi);
        }
    }
}

// ============================================================================
// 2. DECODE LOOP CONTROLLER (DOWN, FUSED BLEND+RMS, PAIR_RMS, BLEND_RMS)
// ============================================================================
__global__ __launch_bounds__(128, 8)
void hip_loop_decode_down_kernel(
    const uint32_t* __restrict__ Pair2,  // [S, M, K/2] packed bf16x2
    const uint32_t* __restrict__ W2,     // [S, 64, K/2] packed bf16x2
    uint16_t*       __restrict__ Out,    // [S, M, 64] bf16
    int M, int K_pairs
) {
    const int n = (int)blockIdx.x;    // 0..63
    const int s = (int)blockIdx.y;    // 0..S-1
    const int tid = (int)threadIdx.x;
    const int warp = tid >> 5;
    const int lane = tid & 31;

    const uint32_t* w_row = W2 + ((int64_t)s * 64 + n) * K_pairs;
    const uint32_t* p_s   = Pair2 + (int64_t)s * M * K_pairs;
    const bool use_u4 = ((K_pairs & 3) == 0) && ((((uintptr_t)w_row | (uintptr_t)p_s) & 15u) == 0u);
    const int K_u4 = K_pairs >> 2;
    __shared__ float sm[4][8];

    for (int m0 = 0; m0 < M; m0 += 8) {
        const int m_cnt = min(8, M - m0);
        float acc[8] = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
        if (use_u4) {
            const uint4* w4 = reinterpret_cast<const uint4*>(w_row);
            const uint4* p4 = reinterpret_cast<const uint4*>(p_s + (int64_t)m0 * K_pairs);
            if (m_cnt == 8) {
                for (int p = tid; p < K_u4; p += 128) {
                    uint4 wv = w4[p];
                    #pragma unroll
                    for (int r = 0; r < 8; ++r) {
                        uint4 xv = p4[(int64_t)r * K_u4 + p];
                        asm("v_dot2_f32_bf16 %0, %1, %2, %0" : "+v"(acc[r]) : "v"(wv.x), "v"(xv.x));
                        asm("v_dot2_f32_bf16 %0, %1, %2, %0" : "+v"(acc[r]) : "v"(wv.y), "v"(xv.y));
                        asm("v_dot2_f32_bf16 %0, %1, %2, %0" : "+v"(acc[r]) : "v"(wv.z), "v"(xv.z));
                        asm("v_dot2_f32_bf16 %0, %1, %2, %0" : "+v"(acc[r]) : "v"(wv.w), "v"(xv.w));
                    }
                }
            } else {
                for (int p = tid; p < K_u4; p += 128) {
                    uint4 wv = w4[p];
                    #pragma unroll
                    for (int r = 0; r < 8; ++r) {
                        if (r < m_cnt) {
                            uint4 xv = p4[(int64_t)r * K_u4 + p];
                            asm("v_dot2_f32_bf16 %0, %1, %2, %0" : "+v"(acc[r]) : "v"(wv.x), "v"(xv.x));
                            asm("v_dot2_f32_bf16 %0, %1, %2, %0" : "+v"(acc[r]) : "v"(wv.y), "v"(xv.y));
                            asm("v_dot2_f32_bf16 %0, %1, %2, %0" : "+v"(acc[r]) : "v"(wv.z), "v"(xv.z));
                            asm("v_dot2_f32_bf16 %0, %1, %2, %0" : "+v"(acc[r]) : "v"(wv.w), "v"(xv.w));
                        }
                    }
                }
            }
        } else {
            for (int p = tid; p < K_pairs; p += 128) {
                uint32_t wv = w_row[p];
                #pragma unroll
                for (int r = 0; r < 8; ++r) {
                    if (r < m_cnt) {
                        uint32_t xv = p_s[(int64_t)(m0 + r) * K_pairs + p];
                        asm("v_dot2_f32_bf16 %0, %1, %2, %0" : "+v"(acc[r]) : "v"(wv), "v"(xv));
                    }
                }
            }
        }
        #pragma unroll
        for (int r = 0; r < 8; ++r) {
            if (r < m_cnt) {
                float v = acc[r];
                #pragma unroll
                for (int off = 16; off > 0; off >>= 1) v += __shfl_xor(v, off, 32);
                if (lane == 0) sm[warp][r] = v;
            }
        }
        __syncthreads();
        if (tid < m_cnt) {
            float tot = sm[0][tid] + sm[1][tid] + sm[2][tid] + sm[3][tid];
            Out[((int64_t)s * M + (m0 + tid)) * 64 + n] = f32_to_bf16_rtne(tot);
        }
        __syncthreads();
    }
}

template <bool NW_F32>
__global__ __launch_bounds__(288, 4)
void hip_loop_decode_blend_rms_kernel(
    const uint16_t* __restrict__ Hidden,  // [S, M, 64] bf16
    const uint16_t* __restrict__ Wup,     // [S, 2*2304, 64] bf16
    const float*    __restrict__ Dest,    // [S, M, 2304] fp32
    const float*    __restrict__ Msg,     // [S, M, 2304] fp32
    const void*     __restrict__ NormW,   // [S, 2304] bf16 or fp32
    uint16_t*       __restrict__ Out,     // [S, M, 2304] bf16
    int64_t sd_s, int64_t sd_t, int64_t sm_s, int64_t sm_t, int64_t sn_s,
    int M, float norm_eps
) {
    constexpr int D = 2304;
    const int m = (int)blockIdx.x;
    const int s = (int)blockIdx.y;
    const int tid = (int)threadIdx.x;
    const int warp = tid >> 5;
    const int lane = tid & 31;
    const int c0 = tid * 8;

    __shared__ uint32_t smem_h[32];
    if (tid < 32) {
        smem_h[tid] = *reinterpret_cast<const uint32_t*>(Hidden + ((int64_t)s * M + m) * 64 + tid * 2);
    }
    __syncthreads();

    const uint32_t* w_gain_base   = reinterpret_cast<const uint32_t*>(Wup + ((int64_t)s * (2 * D) + c0) * 64);
    const uint32_t* w_forget_base = reinterpret_cast<const uint32_t*>(Wup + ((int64_t)s * (2 * D) + D + c0) * 64);

    float4 d0 = *reinterpret_cast<const float4*>(Dest + (int64_t)s * sd_s + (int64_t)m * sd_t + c0);
    float4 d1 = *reinterpret_cast<const float4*>(Dest + (int64_t)s * sd_s + (int64_t)m * sd_t + c0 + 4);
    float4 m0 = *reinterpret_cast<const float4*>(Msg  + (int64_t)s * sm_s + (int64_t)m * sm_t + c0);
    float4 m1 = *reinterpret_cast<const float4*>(Msg  + (int64_t)s * sm_s + (int64_t)m * sm_t + c0 + 4);
    float dv[8] = {d0.x, d0.y, d0.z, d0.w, d1.x, d1.y, d1.z, d1.w};
    float mv[8] = {m0.x, m0.y, m0.z, m0.w, m1.x, m1.y, m1.z, m1.w};

    float bl[8];
    float sq = 0.0f;
    #pragma unroll
    for (int j = 0; j < 8; ++j) {
        const uint32_t* wg = w_gain_base   + j * 32;
        const uint32_t* wf = w_forget_base + j * 32;
        float g = 0.0f, f = 0.0f;
        #pragma unroll
        for (int u = 0; u < 8; ++u) {
            uint4 gq = *reinterpret_cast<const uint4*>(wg + u * 4);
            uint4 fq = *reinterpret_cast<const uint4*>(wf + u * 4);
            asm("v_dot2_f32_bf16 %0, %1, %2, %0" : "+v"(g) : "v"(gq.x), "v"(smem_h[u * 4 + 0]));
            asm("v_dot2_f32_bf16 %0, %1, %2, %0" : "+v"(g) : "v"(gq.y), "v"(smem_h[u * 4 + 1]));
            asm("v_dot2_f32_bf16 %0, %1, %2, %0" : "+v"(g) : "v"(gq.z), "v"(smem_h[u * 4 + 2]));
            asm("v_dot2_f32_bf16 %0, %1, %2, %0" : "+v"(g) : "v"(gq.w), "v"(smem_h[u * 4 + 3]));
            asm("v_dot2_f32_bf16 %0, %1, %2, %0" : "+v"(f) : "v"(fq.x), "v"(smem_h[u * 4 + 0]));
            asm("v_dot2_f32_bf16 %0, %1, %2, %0" : "+v"(f) : "v"(fq.y), "v"(smem_h[u * 4 + 1]));
            asm("v_dot2_f32_bf16 %0, %1, %2, %0" : "+v"(f) : "v"(fq.z), "v"(smem_h[u * 4 + 2]));
            asm("v_dot2_f32_bf16 %0, %1, %2, %0" : "+v"(f) : "v"(fq.w), "v"(smem_h[u * 4 + 3]));
        }
        float b = (1.0f - tanhf(f)) * dv[j] + tanhf(g) * mv[j];
        bl[j] = b;
        sq = fmaf(b, b, sq);
    }

    #pragma unroll
    for (int off = 16; off > 0; off >>= 1) sq += __shfl_xor(sq, off, 32);
    __shared__ float sm_sq[9];
    if (lane == 0) sm_sq[warp] = sq;
    __syncthreads();
    if (tid == 0) {
        float tot = 0.0f;
        #pragma unroll
        for (int w = 0; w < 9; ++w) tot += sm_sq[w];
        sm_sq[0] = rsqrtf(tot * (1.0f / (float)D) + norm_eps);
    }
    __syncthreads();
    const float rstd = sm_sq[0];

    float nw[8];
    if constexpr (NW_F32) {
        const float* np = reinterpret_cast<const float*>(NormW) + (int64_t)s * sn_s + c0;
        float4 n0 = *reinterpret_cast<const float4*>(np);
        float4 n1 = *reinterpret_cast<const float4*>(np + 4);
        nw[0]=n0.x; nw[1]=n0.y; nw[2]=n0.z; nw[3]=n0.w;
        nw[4]=n1.x; nw[5]=n1.y; nw[6]=n1.z; nw[7]=n1.w;
    } else {
        uint4 nu = *reinterpret_cast<const uint4*>(reinterpret_cast<const uint16_t*>(NormW) + (int64_t)s * sn_s + c0);
        nw[0]=bf16_to_f32((uint16_t)(nu.x&0xFFFFu)); nw[1]=bf16_to_f32((uint16_t)(nu.x>>16));
        nw[2]=bf16_to_f32((uint16_t)(nu.y&0xFFFFu)); nw[3]=bf16_to_f32((uint16_t)(nu.y>>16));
        nw[4]=bf16_to_f32((uint16_t)(nu.z&0xFFFFu)); nw[5]=bf16_to_f32((uint16_t)(nu.z>>16));
        nw[6]=bf16_to_f32((uint16_t)(nu.w&0xFFFFu)); nw[7]=bf16_to_f32((uint16_t)(nu.w>>16));
    }

    uint4 ov = make_uint4(
        pack_bf16x2(bl[0] * rstd * nw[0], bl[1] * rstd * nw[1]),
        pack_bf16x2(bl[2] * rstd * nw[2], bl[3] * rstd * nw[3]),
        pack_bf16x2(bl[4] * rstd * nw[4], bl[5] * rstd * nw[5]),
        pack_bf16x2(bl[6] * rstd * nw[6], bl[7] * rstd * nw[7])
    );
    *reinterpret_cast<uint4*>(Out + ((int64_t)s * M + m) * D + c0) = ov;
}

__global__ __launch_bounds__(32, 8)
void hip_loop_decode_blend_p1_kernel(
    const uint16_t* __restrict__ Hidden,   // [S, M, 64] bf16 (M <= 64)
    const uint16_t* __restrict__ Wup,      // [S, 2*2304, 64] bf16
    const float*    __restrict__ Dest,     // [S, M, 2304] fp32
    const float*    __restrict__ Msg,      // [S, M, 2304] fp32
    float*          __restrict__ Blended,  // [S, M, 2304] fp32
    float*          __restrict__ PartSq,   // [S, M, 72] fp32
    int64_t sd_s, int64_t sd_t, int64_t sm_s, int64_t sm_t, int M
) {
    constexpr int D = 2304;
    const int blk  = (int)blockIdx.x;  // 0..71
    const int s    = (int)blockIdx.y;  // 0..S-1
    const int lane = (int)threadIdx.x; // 0..31
    const int c_base = blk * 32;
    const int c = c_base + lane;

    __shared__ uint32_t smem_h[64][32];
    const uint32_t* h_base = reinterpret_cast<const uint32_t*>(Hidden + (int64_t)s * M * 64);
    const int h_u4_total = M * 8;
    const bool h_aligned16 = (((uintptr_t)h_base & 15u) == 0u);
    if (h_aligned16) {
        const uint4* h4 = reinterpret_cast<const uint4*>(h_base);
        for (int idx = lane; idx < h_u4_total; idx += 32) {
            int mr = idx >> 3;
            int u4 = idx & 7;
            uint4 v = h4[idx];
            smem_h[mr][u4 * 4 + 0] = v.x;
            smem_h[mr][u4 * 4 + 1] = v.y;
            smem_h[mr][u4 * 4 + 2] = v.z;
            smem_h[mr][u4 * 4 + 3] = v.w;
        }
    } else {
        for (int idx = lane; idx < M * 32; idx += 32) {
            smem_h[idx >> 5][idx & 31] = h_base[idx];
        }
    }

    __shared__ uint32_t sm_w[32][33];
    const uint4* wg_src = reinterpret_cast<const uint4*>(Wup + ((int64_t)s * (2 * D) + c_base) * 64);
    const uint4* wf_src = reinterpret_cast<const uint4*>(Wup + ((int64_t)s * (2 * D) + D + c_base) * 64);

    #pragma unroll
    for (int i = 0; i < 8; ++i) {
        int idx = i * 32 + lane;
        int r = idx >> 3;
        int u4 = idx & 7;
        uint4 v = wg_src[idx];
        sm_w[r][u4 * 4 + 0] = v.x;
        sm_w[r][u4 * 4 + 1] = v.y;
        sm_w[r][u4 * 4 + 2] = v.z;
        sm_w[r][u4 * 4 + 3] = v.w;
    }
    __syncthreads();
    uint32_t wg[32];
    #pragma unroll
    for (int u = 0; u < 32; ++u) wg[u] = sm_w[lane][u];
    __syncthreads();

    #pragma unroll
    for (int i = 0; i < 8; ++i) {
        int idx = i * 32 + lane;
        int r = idx >> 3;
        int u4 = idx & 7;
        uint4 v = wf_src[idx];
        sm_w[r][u4 * 4 + 0] = v.x;
        sm_w[r][u4 * 4 + 1] = v.y;
        sm_w[r][u4 * 4 + 2] = v.z;
        sm_w[r][u4 * 4 + 3] = v.w;
    }
    __syncthreads();
    uint32_t wf[32];
    #pragma unroll
    for (int u = 0; u < 32; ++u) wf[u] = sm_w[lane][u];
    __syncthreads();

    const float* d_s = Dest + (int64_t)s * sd_s + c;
    const float* m_s = Msg  + (int64_t)s * sm_s + c;
    for (int m = 0; m < M; ++m) {
        float g = 0.0f, f = 0.0f;
        #pragma unroll
        for (int u = 0; u < 32; ++u) {
            uint32_t hv = smem_h[m][u];
            asm("v_dot2_f32_bf16 %0, %1, %2, %0" : "+v"(g) : "v"(wg[u]), "v"(hv));
            asm("v_dot2_f32_bf16 %0, %1, %2, %0" : "+v"(f) : "v"(wf[u]), "v"(hv));
        }
        float dv = d_s[(int64_t)m * sd_t];
        float mv = m_s[(int64_t)m * sm_t];
        float b  = (1.0f - tanhf(f)) * dv + tanhf(g) * mv;
        int64_t row = (int64_t)s * M + m;
        Blended[row * D + c] = b;
        float sq = b * b;
        #pragma unroll
        for (int off = 16; off > 0; off >>= 1) sq += __shfl_xor(sq, off, 32);
        if (lane == 0) PartSq[row * 72 + blk] = sq;
    }
}

template <bool NW_F32>
__global__ __launch_bounds__(288, 4)
void hip_loop_decode_blend_p2_kernel(
    const float* __restrict__ Blended,  // [S, M, 2304] fp32
    const float* __restrict__ PartSq,   // [S, M, 72] fp32
    const void*  __restrict__ NormW,    // [S, 2304] bf16 or fp32
    uint16_t*    __restrict__ Out,      // [S, M, 2304] bf16
    int64_t sn_s, int M, float norm_eps
) {
    constexpr int D = 2304;
    const int m    = (int)blockIdx.x;
    const int s    = (int)blockIdx.y;
    const int tid  = (int)threadIdx.x;
    const int warp = tid >> 5;
    const int lane = tid & 31;
    const int c0   = tid * 8;
    const int64_t row = (int64_t)s * M + m;

    float ps = (tid < 72) ? PartSq[row * 72 + tid] : 0.0f;
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1) ps += __shfl_xor(ps, off, 32);
    __shared__ float sm_sq[9];
    if (lane == 0) sm_sq[warp] = ps;
    __syncthreads();
    if (tid == 0) {
        float tot = sm_sq[0] + sm_sq[1] + sm_sq[2];
        sm_sq[0] = rsqrtf(tot * (1.0f / (float)D) + norm_eps);
    }
    __syncthreads();
    const float rstd = sm_sq[0];

    float4 b0 = *reinterpret_cast<const float4*>(Blended + row * D + c0);
    float4 b1 = *reinterpret_cast<const float4*>(Blended + row * D + c0 + 4);
    float bl[8] = {b0.x, b0.y, b0.z, b0.w, b1.x, b1.y, b1.z, b1.w};

    float nw[8];
    if constexpr (NW_F32) {
        const float* np = reinterpret_cast<const float*>(NormW) + (int64_t)s * sn_s + c0;
        float4 n0 = *reinterpret_cast<const float4*>(np);
        float4 n1 = *reinterpret_cast<const float4*>(np + 4);
        nw[0]=n0.x; nw[1]=n0.y; nw[2]=n0.z; nw[3]=n0.w;
        nw[4]=n1.x; nw[5]=n1.y; nw[6]=n1.z; nw[7]=n1.w;
    } else {
        uint4 nu = *reinterpret_cast<const uint4*>(reinterpret_cast<const uint16_t*>(NormW) + (int64_t)s * sn_s + c0);
        nw[0]=bf16_to_f32((uint16_t)(nu.x&0xFFFFu)); nw[1]=bf16_to_f32((uint16_t)(nu.x>>16));
        nw[2]=bf16_to_f32((uint16_t)(nu.y&0xFFFFu)); nw[3]=bf16_to_f32((uint16_t)(nu.y>>16));
        nw[4]=bf16_to_f32((uint16_t)(nu.z&0xFFFFu)); nw[5]=bf16_to_f32((uint16_t)(nu.z>>16));
        nw[6]=bf16_to_f32((uint16_t)(nu.w&0xFFFFu)); nw[7]=bf16_to_f32((uint16_t)(nu.w>>16));
    }

    *reinterpret_cast<uint4*>(Out + row * D + c0) = make_uint4(
        pack_bf16x2(bl[0] * rstd * nw[0], bl[1] * rstd * nw[1]),
        pack_bf16x2(bl[2] * rstd * nw[2], bl[3] * rstd * nw[3]),
        pack_bf16x2(bl[4] * rstd * nw[4], bl[5] * rstd * nw[5]),
        pack_bf16x2(bl[6] * rstd * nw[6], bl[7] * rstd * nw[7])
    );
}

__global__ __launch_bounds__(288, 4)
void hip_pair_rmsnorm_kernel(
    const float* __restrict__ Dest,
    const float* __restrict__ Msg,
    uint16_t*    __restrict__ Out,
    int64_t sd_s, int64_t sd_t, int64_t sm_s, int64_t sm_t, int M
) {
    constexpr int D = 2304;
    const int m = (int)blockIdx.x;
    const int s = (int)blockIdx.y;
    const int tid = (int)threadIdx.x;
    const int warp = tid >> 5;
    const int lane = tid & 31;
    const int c0 = tid * 8;

    float4 d0 = *reinterpret_cast<const float4*>(Dest + (int64_t)s * sd_s + (int64_t)m * sd_t + c0);
    float4 d1 = *reinterpret_cast<const float4*>(Dest + (int64_t)s * sd_s + (int64_t)m * sd_t + c0 + 4);
    float4 m0 = *reinterpret_cast<const float4*>(Msg  + (int64_t)s * sm_s + (int64_t)m * sm_t + c0);
    float4 m1 = *reinterpret_cast<const float4*>(Msg  + (int64_t)s * sm_s + (int64_t)m * sm_t + c0 + 4);
    float sq_d = d0.x*d0.x + d0.y*d0.y + d0.z*d0.z + d0.w*d0.w + d1.x*d1.x + d1.y*d1.y + d1.z*d1.z + d1.w*d1.w;
    float sq_m = m0.x*m0.x + m0.y*m0.y + m0.z*m0.z + m0.w*m0.w + m1.x*m1.x + m1.y*m1.y + m1.z*m1.z + m1.w*m1.w;

    #pragma unroll
    for (int off = 16; off > 0; off >>= 1) {
        sq_d += __shfl_xor(sq_d, off, 32);
        sq_m += __shfl_xor(sq_m, off, 32);
    }
    __shared__ float sd_sm[9], sm_sm[9];
    if (lane == 0) { sd_sm[warp] = sq_d; sm_sm[warp] = sq_m; }
    __syncthreads();
    if (tid == 0) {
        float td = 0.0f, tm = 0.0f;
        #pragma unroll
        for (int w = 0; w < 9; ++w) { td += sd_sm[w]; tm += sm_sm[w]; }
        sd_sm[0] = rsqrtf(td * (1.0f / (float)D) + 1e-6f);
        sm_sm[0] = rsqrtf(tm * (1.0f / (float)D) + 1e-6f);
    }
    __syncthreads();
    const float rd = sd_sm[0], rm = sm_sm[0];

    uint16_t* ob = Out + ((int64_t)s * M + m) * (2 * D) + c0;
    *reinterpret_cast<uint4*>(ob) = make_uint4(
        pack_bf16x2(d0.x * rd, d0.y * rd), pack_bf16x2(d0.z * rd, d0.w * rd),
        pack_bf16x2(d1.x * rd, d1.y * rd), pack_bf16x2(d1.z * rd, d1.w * rd));
    *reinterpret_cast<uint4*>(ob + D) = make_uint4(
        pack_bf16x2(m0.x * rm, m0.y * rm), pack_bf16x2(m0.z * rm, m0.w * rm),
        pack_bf16x2(m1.x * rm, m1.y * rm), pack_bf16x2(m1.z * rm, m1.w * rm));
}

template <bool NW_F32>
__global__ __launch_bounds__(288, 4)
void hip_blend_rmsnorm_kernel(
    const float* __restrict__ Dest,
    const float* __restrict__ Msg,
    const float* __restrict__ GF,
    const void*  __restrict__ NormW,
    uint16_t*    __restrict__ Out,
    int64_t sd_s, int64_t sd_t, int64_t sm_s, int64_t sm_t, int64_t sn_s,
    int M, float norm_eps
) {
    constexpr int D = 2304;
    const int m = (int)blockIdx.x;
    const int s = (int)blockIdx.y;
    const int tid = (int)threadIdx.x;
    const int warp = tid >> 5;
    const int lane = tid & 31;
    const int c0 = tid * 8;
    const int64_t row = (int64_t)s * M + m;

    float4 d0 = *reinterpret_cast<const float4*>(Dest + (int64_t)s * sd_s + (int64_t)m * sd_t + c0);
    float4 d1 = *reinterpret_cast<const float4*>(Dest + (int64_t)s * sd_s + (int64_t)m * sd_t + c0 + 4);
    float4 m0 = *reinterpret_cast<const float4*>(Msg  + (int64_t)s * sm_s + (int64_t)m * sm_t + c0);
    float4 m1 = *reinterpret_cast<const float4*>(Msg  + (int64_t)s * sm_s + (int64_t)m * sm_t + c0 + 4);
    const float* gf_row = GF + row * (2 * D) + c0;
    float4 g0 = *reinterpret_cast<const float4*>(gf_row);
    float4 g1 = *reinterpret_cast<const float4*>(gf_row + 4);
    float4 f0 = *reinterpret_cast<const float4*>(gf_row + D);
    float4 f1 = *reinterpret_cast<const float4*>(gf_row + D + 4);

    float dv[8] = {d0.x, d0.y, d0.z, d0.w, d1.x, d1.y, d1.z, d1.w};
    float mv[8] = {m0.x, m0.y, m0.z, m0.w, m1.x, m1.y, m1.z, m1.w};
    float gv[8] = {g0.x, g0.y, g0.z, g0.w, g1.x, g1.y, g1.z, g1.w};
    float fv[8] = {f0.x, f0.y, f0.z, f0.w, f1.x, f1.y, f1.z, f1.w};

    float bl[8];
    float sq = 0.0f;
    #pragma unroll
    for (int j = 0; j < 8; ++j) {
        float b = (1.0f - tanhf(fv[j])) * dv[j] + tanhf(gv[j]) * mv[j];
        bl[j] = b;
        sq = fmaf(b, b, sq);
    }

    #pragma unroll
    for (int off = 16; off > 0; off >>= 1) sq += __shfl_xor(sq, off, 32);
    __shared__ float sm_sq[9];
    if (lane == 0) sm_sq[warp] = sq;
    __syncthreads();
    if (tid == 0) {
        float tot = 0.0f;
        #pragma unroll
        for (int w = 0; w < 9; ++w) tot += sm_sq[w];
        sm_sq[0] = rsqrtf(tot * (1.0f / (float)D) + norm_eps);
    }
    __syncthreads();
    const float rstd = sm_sq[0];

    float nw[8];
    if constexpr (NW_F32) {
        const float* np = reinterpret_cast<const float*>(NormW) + (int64_t)s * sn_s + c0;
        float4 n0 = *reinterpret_cast<const float4*>(np);
        float4 n1 = *reinterpret_cast<const float4*>(np + 4);
        nw[0]=n0.x; nw[1]=n0.y; nw[2]=n0.z; nw[3]=n0.w;
        nw[4]=n1.x; nw[5]=n1.y; nw[6]=n1.z; nw[7]=n1.w;
    } else {
        uint4 nu = *reinterpret_cast<const uint4*>(reinterpret_cast<const uint16_t*>(NormW) + (int64_t)s * sn_s + c0);
        nw[0]=bf16_to_f32((uint16_t)(nu.x&0xFFFFu)); nw[1]=bf16_to_f32((uint16_t)(nu.x>>16));
        nw[2]=bf16_to_f32((uint16_t)(nu.y&0xFFFFu)); nw[3]=bf16_to_f32((uint16_t)(nu.y>>16));
        nw[4]=bf16_to_f32((uint16_t)(nu.z&0xFFFFu)); nw[5]=bf16_to_f32((uint16_t)(nu.z>>16));
        nw[6]=bf16_to_f32((uint16_t)(nu.w&0xFFFFu)); nw[7]=bf16_to_f32((uint16_t)(nu.w>>16));
    }
    *reinterpret_cast<uint4*>(Out + row * D + c0) = make_uint4(
        pack_bf16x2(bl[0] * rstd * nw[0], bl[1] * rstd * nw[1]),
        pack_bf16x2(bl[2] * rstd * nw[2], bl[3] * rstd * nw[3]),
        pack_bf16x2(bl[4] * rstd * nw[4], bl[5] * rstd * nw[5]),
        pack_bf16x2(bl[6] * rstd * nw[6], bl[7] * rstd * nw[7]));
}

// ============================================================================
// 3. DECODE ROUTER (GEMM M<=8 + BIT-EXACT IEEE-754 ROUTE_M8 & M16 PACKER)
// ============================================================================
template <bool FUSED_SIGMOID_BIAS>
__global__ __launch_bounds__(128, 8)
void hip_router_gemm_m8_kernel(
    const uint32_t* __restrict__ X2,        // [M, 1152] bf16x2
    const uint32_t* __restrict__ W2,        // [128, 1152] bf16x2
    const float*    __restrict__ Bias,      // [128] optional
    float*          __restrict__ Out,       // [M, 128]
    float*          __restrict__ Corrected, // [M, 128] optional
    int M
) {
    const int e = (int)blockIdx.x;
    const int tid = (int)threadIdx.x;
    const int warp = tid >> 5;
    const int lane = tid & 31;
    const uint32_t* wr = W2 + (int64_t)e * 1152;
    const bool use_u4 = ((((uintptr_t)wr | (uintptr_t)X2) & 15u) == 0u);

    float acc[8] = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
    if (use_u4) {
        const uint4* w4 = reinterpret_cast<const uint4*>(wr);
        const uint4* x4 = reinterpret_cast<const uint4*>(X2);
        if (M == 8) {
            for (int p = tid; p < 288; p += 128) {
                uint4 wv = w4[p];
                #pragma unroll
                for (int r = 0; r < 8; ++r) {
                    uint4 xv = x4[(int64_t)r * 288 + p];
                    asm("v_dot2_f32_bf16 %0, %1, %2, %0" : "+v"(acc[r]) : "v"(wv.x), "v"(xv.x));
                    asm("v_dot2_f32_bf16 %0, %1, %2, %0" : "+v"(acc[r]) : "v"(wv.y), "v"(xv.y));
                    asm("v_dot2_f32_bf16 %0, %1, %2, %0" : "+v"(acc[r]) : "v"(wv.z), "v"(xv.z));
                    asm("v_dot2_f32_bf16 %0, %1, %2, %0" : "+v"(acc[r]) : "v"(wv.w), "v"(xv.w));
                }
            }
        } else if (M == 1) {
            for (int p = tid; p < 288; p += 128) {
                uint4 wv = w4[p];
                uint4 xv = x4[p];
                asm("v_dot2_f32_bf16 %0, %1, %2, %0" : "+v"(acc[0]) : "v"(wv.x), "v"(xv.x));
                asm("v_dot2_f32_bf16 %0, %1, %2, %0" : "+v"(acc[0]) : "v"(wv.y), "v"(xv.y));
                asm("v_dot2_f32_bf16 %0, %1, %2, %0" : "+v"(acc[0]) : "v"(wv.z), "v"(xv.z));
                asm("v_dot2_f32_bf16 %0, %1, %2, %0" : "+v"(acc[0]) : "v"(wv.w), "v"(xv.w));
            }
        } else {
            for (int p = tid; p < 288; p += 128) {
                uint4 wv = w4[p];
                #pragma unroll
                for (int r = 0; r < 8; ++r) {
                    if (r < M) {
                        uint4 xv = x4[(int64_t)r * 288 + p];
                        asm("v_dot2_f32_bf16 %0, %1, %2, %0" : "+v"(acc[r]) : "v"(wv.x), "v"(xv.x));
                        asm("v_dot2_f32_bf16 %0, %1, %2, %0" : "+v"(acc[r]) : "v"(wv.y), "v"(xv.y));
                        asm("v_dot2_f32_bf16 %0, %1, %2, %0" : "+v"(acc[r]) : "v"(wv.z), "v"(xv.z));
                        asm("v_dot2_f32_bf16 %0, %1, %2, %0" : "+v"(acc[r]) : "v"(wv.w), "v"(xv.w));
                    }
                }
            }
        }
    } else {
        for (int p = tid; p < 1152; p += 128) {
            uint32_t wv = wr[p];
            #pragma unroll
            for (int r = 0; r < 8; ++r) {
                if (r < M) {
                    uint32_t xv = X2[(int64_t)r * 1152 + p];
                    asm("v_dot2_f32_bf16 %0, %1, %2, %0" : "+v"(acc[r]) : "v"(wv), "v"(xv));
                }
            }
        }
    }
    __shared__ float sm[4][8];
    #pragma unroll
    for (int r = 0; r < 8; ++r) {
        if (r < M) {
            float v = acc[r];
            #pragma unroll
            for (int off = 16; off > 0; off >>= 1) v += __shfl_xor(v, off, 32);
            if (lane == 0) sm[warp][r] = v;
        }
    }
    __syncthreads();
    if (tid < M) {
        float tot = sm[0][tid] + sm[1][tid] + sm[2][tid] + sm[3][tid];
        if constexpr (FUSED_SIGMOID_BIAS) {
            float sc = 1.0f / (1.0f + expf(-tot));
            Out[tid * 128 + e] = sc;
            Corrected[tid * 128 + e] = sc + Bias[e];
        } else {
            Out[tid * 128 + e] = tot;
        }
    }
}

__device__ __forceinline__ uint64_t ieee_desc_stable_key(float val, int expert) {
    uint32_t bits = __float_as_uint(val);
    uint32_t mag = bits & 0x7FFFFFFFu;
    if (mag == 0u) bits = 0u;
    uint32_t ord = (bits & 0x80000000u) ? ~bits : (bits ^ 0x80000000u);
    if (mag > 0x7F800000u) ord = 0xFFFFFFFFu;
    return ((uint64_t)ord << 7) | (uint64_t)(127 - expert);
}

template <bool DO_ALIGN, bool DO_SIGMOID, bool BIAS_IS_CORR>
__global__ __launch_bounds__(256, 2)
void hip_route_m8_kernel(
    const float* __restrict__ Score,
    const float* __restrict__ Bias,
    int64_t*     __restrict__ Ids,
    float*       __restrict__ Gamma,
    int32_t*     __restrict__ Sorted,
    int32_t*     __restrict__ Experts,
    int32_t*     __restrict__ Padded,
    int M, float scale
) {
    const int tid  = (int)threadIdx.x;
    const int row  = tid >> 5;   // 0..7 (1 warp per token row!)
    const int lane = tid & 31;

    __shared__ float    s_src[8][128];
    __shared__ int32_t  s_ids[64];
    __shared__ uint8_t  s_bcnt[128];
    __shared__ uint8_t  s_bucket[128][8];

    if constexpr (DO_ALIGN) {
        if (tid < 128) s_bcnt[tid] = 0;
    }

    if (row < M) {
        uint64_t keys[4];
        bool     used[4] = {false, false, false, false};
        #pragma unroll
        for (int j = 0; j < 4; ++j) {
            int e = lane + j * 32;
            float src = Score[row * 128 + e];
            if constexpr (DO_SIGMOID) {
                src = 1.0f / (1.0f + expf(-src));
            }
            s_src[row][e] = src;
            float corr = BIAS_IS_CORR ? Bias[e] : (src + Bias[e]);
            keys[j] = ieee_desc_stable_key(corr, e);
        }

        int   top_e[8];
        float top_s[8];
        #pragma unroll
        for (int k = 0; k < 8; ++k) {
            uint64_t best_k = 0;
            bool has_any = false;
            #pragma unroll
            for (int j = 0; j < 4; ++j) {
                if (!used[j] && (!has_any || keys[j] > best_k)) {
                    best_k = keys[j];
                    has_any = true;
                }
            }
            // Encode valid flag in bit 63? Keys only use 39 bits (32+7), so valid keys have bit 40 set!
            uint64_t wk = has_any ? (best_k | (1ULL << 40)) : 0ULL;
            #pragma unroll
            for (int off = 16; off > 0; off >>= 1) {
                uint32_t lo = (uint32_t)wk;
                uint32_t hi = (uint32_t)(wk >> 32);
                uint32_t olo = __shfl_xor(lo, off, 32);
                uint32_t ohi = __shfl_xor(hi, off, 32);
                uint64_t owk = ((uint64_t)ohi << 32) | (uint64_t)olo;
                if (owk > wk) wk = owk;
            }
            int win_e = 127 - (int)(wk & 127ULL);
            #pragma unroll
            for (int j = 0; j < 4; ++j) {
                if (lane + j * 32 == win_e) used[j] = true;
            }
            if (lane == 0) {
                top_e[k] = win_e;
                top_s[k] = s_src[row][win_e];
            }
        }

        if (lane == 0) {
            float denom = ((top_s[0] + top_s[1]) + (top_s[2] + top_s[3]))
                        + ((top_s[4] + top_s[5]) + (top_s[6] + top_s[7])) + 1e-20f;
            #pragma unroll
            for (int k = 0; k < 8; ++k) {
                Ids[row * 8 + k] = (int64_t)top_e[k];
                Gamma[row * 8 + k] = (top_s[k] / denom) * scale;
                s_ids[row * 8 + k] = top_e[k];
            }
        }
    }

    if constexpr (DO_ALIGN) {
        __shared__ int32_t s_wtot[4];
        __shared__ int32_t s_woff[4];
        __shared__ uint8_t s_blk_e[64];
        __syncthreads();
        const int n_routes = M * 8;
        int has = 0;
        int ws = 0;
        if (tid < 128) {
            int c = 0;
            for (int rt = 0; rt < n_routes; ++rt) {
                if (s_ids[rt] == tid) {
                    s_bucket[tid][c] = (uint8_t)rt;
                    c += 1;
                }
            }
            s_bcnt[tid] = (uint8_t)c;
            has = (c > 0) ? 1 : 0;
            ws = has;
            #pragma unroll
            for (int off = 1; off < 32; off <<= 1) {
                int v = __shfl_up(ws, off, 32);
                if (lane >= off) ws += v;
            }
            if (lane == 31) s_wtot[row] = ws;
        }
        __syncthreads();
        if (tid == 0) {
            int acc_b = 0;
            #pragma unroll
            for (int w = 0; w < 4; ++w) {
                s_woff[w] = acc_b;
                acc_b += s_wtot[w];
            }
            *Padded = acc_b * 16;
        }
        __syncthreads();
        if (tid < 128 && has) {
            int blk = s_woff[row] + (ws - 1);
            Experts[blk] = tid;
            s_blk_e[blk] = (uint8_t)tid;
        }
        __syncthreads();
        const int total_elems = (s_woff[3] + s_wtot[3]) * 16;
        for (int idx = tid; idx < total_elems; idx += 256) {
            int b = idx >> 4;
            int j = idx & 15;
            int e = (int)s_blk_e[b];
            Sorted[idx] = (j < (int)s_bcnt[e]) ? (int32_t)s_bucket[e][j] : n_routes;
        }
    }
}

// ============================================================================
// 4. DSPARK DRAFT ENGINE KERNELS (MIX_RMS, RMS_NORM, QK_NORM_ROPE, W8, SILU, SOFTCAP, MARKOV_GATE)
// ============================================================================
__global__ __launch_bounds__(256, 4)
void hip_mix_rmsnorm_kernel(
    const uint16_t* __restrict__ X,
    const float*    __restrict__ Alpha,
    const uint16_t* __restrict__ Weight,
    uint16_t*       __restrict__ Out,
    int64_t s_row, int64_t s_tap, int64_t s_hid,
    int taps, int hidden, float eps
) {
    const int64_t row = (int64_t)blockIdx.x;
    const int tid = (int)threadIdx.x;
    const int warp = tid >> 5;
    const int lane = tid & 31;

    __shared__ float s_alp[16];
    if (tid < taps && tid < 16) s_alp[tid] = Alpha[tid];
    __syncthreads();

    float sq = 0.0f;
    for (int h = tid; h < hidden; h += 256) {
        float acc = 0.0f;
        for (int t = 0; t < taps; ++t) {
            float xv = bf16_to_f32(X[row * s_row + (int64_t)t * s_tap + (int64_t)h * s_hid]);
            float av = (t < 16) ? s_alp[t] : Alpha[t];
            acc = fmaf(xv, av, acc);
        }
        sq = fmaf(acc, acc, sq);
    }
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1) sq += __shfl_xor(sq, off, 32);
    __shared__ float sm[8];
    if (lane == 0) sm[warp] = sq;
    __syncthreads();
    if (tid == 0) {
        float tot = 0.0f;
        #pragma unroll
        for (int w = 0; w < 8; ++w) tot += sm[w];
        sm[0] = rsqrtf(tot / (float)hidden + eps);
    }
    __syncthreads();
    const float inv = sm[0];

    for (int h = tid; h < hidden; h += 256) {
        float acc = 0.0f;
        for (int t = 0; t < taps; ++t) {
            float xv = bf16_to_f32(X[row * s_row + (int64_t)t * s_tap + (int64_t)h * s_hid]);
            float av = (t < 16) ? s_alp[t] : Alpha[t];
            acc = fmaf(xv, av, acc);
        }
        float w = bf16_to_f32(Weight[h]);
        Out[row * hidden + h] = f32_to_bf16_rtne(acc * inv * w);
    }
}

template <bool IN_F32, bool OUT_F32>
__global__ __launch_bounds__(256, 4)
void hip_rms_norm_kernel(
    const void*     __restrict__ X,
    const uint16_t* __restrict__ Weight,
    void*           __restrict__ Out,
    int64_t stride_x, int hidden, float eps
) {
    const int64_t row = (int64_t)blockIdx.x;
    const int tid = (int)threadIdx.x;
    const int warp = tid >> 5;
    const int lane = tid & 31;

    float sq = 0.0f;
    for (int h = tid; h < hidden; h += 256) {
        float v = IN_F32
            ? reinterpret_cast<const float*>(X)[row * stride_x + h]
            : bf16_to_f32(reinterpret_cast<const uint16_t*>(X)[row * stride_x + h]);
        sq = fmaf(v, v, sq);
    }
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1) sq += __shfl_xor(sq, off, 32);
    __shared__ float sm[8];
    if (lane == 0) sm[warp] = sq;
    __syncthreads();
    if (tid == 0) {
        float tot = 0.0f;
        #pragma unroll
        for (int w = 0; w < 8; ++w) tot += sm[w];
        sm[0] = rsqrtf(tot / (float)hidden + eps);
    }
    __syncthreads();
    const float inv = sm[0];

    for (int h = tid; h < hidden; h += 256) {
        float v = IN_F32
            ? reinterpret_cast<const float*>(X)[row * stride_x + h]
            : bf16_to_f32(reinterpret_cast<const uint16_t*>(X)[row * stride_x + h]);
        float res = v * inv * bf16_to_f32(Weight[h]);
        if constexpr (OUT_F32) {
            reinterpret_cast<float*>(Out)[row * hidden + h] = res;
        } else {
            reinterpret_cast<uint16_t*>(Out)[row * hidden + h] = f32_to_bf16_rtne(res);
        }
    }
}

template <bool DO_NORM>
__global__ __launch_bounds__(64, 8)
void hip_qk_norm_rope_kernel(
    uint16_t*       __restrict__ X,
    const uint16_t* __restrict__ QW,
    const uint16_t* __restrict__ KW,
    const float*    __restrict__ CosSin,
    const int64_t*  __restrict__ Pos,
    int64_t row_stride, int64_t q_size, int nhq, int D, float eps
) {
    const int64_t t = (int64_t)blockIdx.x;
    const int h = (int)blockIdx.y;
    const int tid = (int)threadIdx.x;
    const int half = D >> 1;
    const bool active = (tid < half);

    const bool is_q = (h < nhq);
    const int64_t col0 = is_q ? ((int64_t)h * D) : (q_size + (int64_t)(h - nhq) * D);
    uint16_t* row = X + t * row_stride + col0;

    float x1 = 0.0f, x2 = 0.0f, c = 0.0f, s = 0.0f;
    if (active) {
        const int64_t pos = Pos[t];
        c = CosSin[pos * D + tid];
        s = CosSin[pos * D + half + tid];
        x1 = bf16_to_f32(row[tid]);
        x2 = bf16_to_f32(row[half + tid]);
    }

    if constexpr (DO_NORM) {
        float sq = active ? (x1 * x1 + x2 * x2) : 0.0f;
        #pragma unroll
        for (int off = 16; off > 0; off >>= 1) sq += __shfl_xor(sq, off, 32);
        __shared__ float sm[2];
        if ((tid & 31) == 0) sm[tid >> 5] = sq;
        __syncthreads();
        if (!active) return;
        float tot = sm[0] + sm[1];
        float inv = 1.0f / sqrtf(tot / (float)D + eps);
        const uint16_t* wp = is_q ? QW : KW;
        x1 = bf16_round_f32(x1 * inv * bf16_to_f32(wp[tid]));
        x2 = bf16_round_f32(x2 * inv * bf16_to_f32(wp[half + tid]));
    } else {
        if (!active) return;
    }

    float o1 = x1 * c - x2 * s;
    float o2 = x2 * c + x1 * s;
    row[tid] = f32_to_bf16_rtne(o1);
    row[half + tid] = f32_to_bf16_rtne(o2);
}

__device__ __forceinline__ uint32_t unpack_s8_pair_bf16(int8_t a, int8_t b) {
    uint32_t lo = __float_as_uint((float)(int)a) >> 16;
    uint32_t hi = __float_as_uint((float)(int)b) & 0xFFFF0000u;
    return lo | hi;
}

__device__ __forceinline__ short16 glue_d_load_s16_u4(const uint16_t* ptr) {
    uint4 lo = *reinterpret_cast<const uint4*>(ptr);
    uint4 hi = *reinterpret_cast<const uint4*>(ptr + 8);
    short16 out;
    uint32_t words[8] = {lo.x, lo.y, lo.z, lo.w, hi.x, hi.y, hi.z, hi.w};
    __builtin_memcpy(&out, words, 32);
    return out;
}

template <int SPLITS>
__global__ __launch_bounds__(128, 6)
void hip_dspark_w8_kernel(
    const uint16_t* __restrict__ X,       // [M, K] bf16 (M <= 16)
    const int8_t*   __restrict__ Codes,   // [N/64, K/64, 64, 64] int8
    const uint16_t* __restrict__ Scales,  // [N/64, K/128, 64] fp16
    void*           __restrict__ Out,     // [M, N] bf16 if SPLITS==1 else [SPLITS, M, N] fp32
    int M, int N, int K
) {
    const int tile_n = (int)blockIdx.x;
    const int split  = (int)blockIdx.y;
    const int tid    = (int)threadIdx.x;
    const int warp   = tid >> 5;
    const int lane   = tid & 31;
    const int lane16 = lane & 15;
    const int half   = lane >> 4;

    const int row_in_64 = warp * 16 + lane16;
    const int out_col   = tile_n * 64 + row_in_64;

    const int groups = K >> 7;
    const int k_tiles = K >> 6;
    const int g_per_sp = groups / SPLITS;
    const int g_start = split * g_per_sp;
    const int g_end   = g_start + g_per_sp;

    __shared__ uint16_t smem_X[16][72];
    const int tx_row = tid >> 3;
    const int tx_col = (tid & 7) * 8;

    float8 acc = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};

    auto unpack_16_s8 = [&](uint4 u, short16 &w_lo, short16 &w_hi) {
        uint32_t words[4] = {u.x, u.y, u.z, u.w};
        uint32_t w[8], o[8];
        #pragma unroll
        for (int i = 0; i < 4; ++i) {
            uint32_t v = words[i];
            w[i * 2 + 0] = unpack_s8_pair_bf16((int8_t)(v & 0xFFu), (int8_t)((v >> 8) & 0xFFu));
            w[i * 2 + 1] = unpack_s8_pair_bf16((int8_t)((v >> 16) & 0xFFu), (int8_t)((v >> 24) & 0xFFu));
        }
        #pragma unroll
        for (int i = 0; i < 8; ++i) o[i] = __shfl_xor(w[i], 16, 32);
        __builtin_memcpy(&w_lo, (half == 0) ? w : o, 32);
        __builtin_memcpy(&w_hi, (half == 0) ? o : w, 32);
    };

    for (int grp = g_start; grp < g_end; ++grp) {
        float8 blk = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
        #pragma unroll
        for (int gh = 0; gh < 2; ++gh) {
            const int kt = grp * 2 + gh;
            const int k_base = kt << 6;
            uint4 xv = (tx_row < M)
                ? *reinterpret_cast<const uint4*>(X + (int64_t)tx_row * K + k_base + tx_col)
                : make_uint4(0, 0, 0, 0);
            *reinterpret_cast<uint4*>(&smem_X[tx_row][tx_col]) = xv;
            __syncthreads();

            const int64_t c_off = (((int64_t)tile_n * k_tiles + kt) * 64 + row_in_64) * 64 + half * 32;
            uint4 q0 = *reinterpret_cast<const uint4*>(Codes + c_off);
            uint4 q1 = *reinterpret_cast<const uint4*>(Codes + c_off + 16);

            short16 w0, w2, w1, w3;
            unpack_16_s8(q0, w0, w2);
            short16 x0 = glue_d_load_s16_u4(&smem_X[lane16][0]);
            short16 x2 = glue_d_load_s16_u4(&smem_X[lane16][32]);
            blk = __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32(x0, w0, blk);
            blk = __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32(x2, w2, blk);

            unpack_16_s8(q1, w1, w3);
            short16 x1 = glue_d_load_s16_u4(&smem_X[lane16][16]);
            short16 x3 = glue_d_load_s16_u4(&smem_X[lane16][48]);
            blk = __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32(x1, w1, blk);
            blk = __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32(x3, w3, blk);
            __syncthreads();
        }
        float sc = f16_bits_to_f32(Scales[((int64_t)tile_n * groups + grp) * 64 + row_in_64]);
        #pragma unroll
        for (int r = 0; r < 8; ++r) acc[r] = fmaf(blk[r], sc, acc[r]);
    }

    const int64_t sp_row = (SPLITS > 1) ? ((int64_t)split * M) : 0;
    #pragma unroll
    for (int r = 0; r < 8; ++r) {
        int mr = half + 2 * r;
        if (mr < M) {
            int64_t idx = (sp_row + mr) * (int64_t)N + out_col;
            if constexpr (SPLITS > 1) {
                reinterpret_cast<float*>(Out)[idx] = acc[r];
            } else {
                reinterpret_cast<uint16_t*>(Out)[idx] = f32_to_bf16_rtne(acc[r]);
            }
        }
    }
}

__global__ __launch_bounds__(256, 4)
void hip_silu_and_mul_kernel(
    const uint16_t* __restrict__ X,
    uint16_t*       __restrict__ Y,
    int64_t stride_x, int N
) {
    const int64_t row = (int64_t)blockIdx.y;
    const int col = ((int)blockIdx.x * 256 + (int)threadIdx.x) * 4;
    if (col >= N) return;
    const uint16_t* gp = X + row * stride_x + col;
    const uint16_t* up = X + row * stride_x + N + col;
    uint16_t* op = Y + row * N + col;
    for (int j = 0; j < 4 && col + j < N; ++j) {
        float g = bf16_to_f32(gp[j]);
        float u = bf16_to_f32(up[j]);
        float s = g / (1.0f + expf(-g));
        op[j] = f32_to_bf16_rtne(s * u);
    }
}

__global__ __launch_bounds__(256, 4)
void hip_add_softcap_kernel(
    const uint16_t* __restrict__ A,
    const uint16_t* __restrict__ B,
    uint16_t*       __restrict__ Out,
    int64_t stride_a, int64_t stride_b, int N, float softcap
) {
    const int64_t row = (int64_t)blockIdx.y;
    const int col = ((int)blockIdx.x * 256 + (int)threadIdx.x) * 4;
    if (col >= N) return;
    const float inv_cap = 1.0f / softcap;
    const uint16_t* ap = A + row * stride_a + col;
    const uint16_t* bp = B + row * stride_b + col;
    uint16_t* op = Out + row * N + col;
    for (int j = 0; j < 4 && col + j < N; ++j) {
        float x = bf16_round_f32(bf16_to_f32(ap[j]) + bf16_to_f32(bp[j]));
        float y = tanhf(x * inv_cap) * softcap;
        op[j] = f32_to_bf16_rtne(y);
    }
}

__global__ __launch_bounds__(256, 4)
void hip_markov_gate_kernel(
    const float*    __restrict__ Raw,
    const uint16_t* __restrict__ State,
    uint16_t*       __restrict__ Out,
    uint16_t*       __restrict__ Latent,
    int rank
) {
    const int64_t row = (int64_t)blockIdx.x;
    const int c = (int)threadIdx.x;
    if (c >= rank) return;
    const float* rp = Raw + row * 3 * rank + c;
    float gate = bf16_round_f32(rp[0]);
    float cand = bf16_round_f32(rp[rank]);
    float out_v = bf16_round_f32(rp[2 * rank]);
    float st = bf16_to_f32(State[row * rank + c]);
    gate = bf16_round_f32(1.0f / (1.0f + expf(-gate)));
    cand = bf16_round_f32(tanhf(cand));
    float keep = bf16_round_f32(gate * st);
    float upd  = bf16_round_f32(bf16_round_f32(1.0f - gate) * cand);
    Out[row * rank + c] = f32_to_bf16_rtne(keep + upd);
    Latent[row * rank + c] = f32_to_bf16_rtne(tanhf(out_v));
}

extern "C" {

int launch_hip_sconv_decode(
    const void* X, const void* Weight, void* Cache,
    const void* Pos, const void* BlockTable, const void* SeqIdx,
    const void* Slot, const void* QStart,
    const void* Addend, const void* Residual, void* Out,
    int T, int D, int H, int stream_off, int stream_w,
    int64_t sx, int64_t sw0, int64_t sw1,
    int64_t scb, int64_t sch, int64_t scn,
    int64_t sbt, int max_blocks, int N_ring, float scale,
    int has_addend, int has_res, int add_f32, int res_f32, int out_f32,
    int pos_i64, void* stream_ptr
) {
    if (T <= 0 || D <= 0) return 0;
    hipStream_t stream = (hipStream_t)stream_ptr;
    dim3 grid((T + 7) / 8, (D + 31) / 32);
    dim3 block(32);
#define LSC(HA, HR, AF, RF, OF) \
    hipLaunchKernelGGL((hip_sconv_decode_kernel<HA, HR, AF, RF, OF>), grid, block, 0, stream, \
        (const uint16_t*)X, (const uint16_t*)Weight, (uint16_t*)Cache, \
        Pos, (const int32_t*)BlockTable, (const int32_t*)SeqIdx, \
        (const int64_t*)Slot, (const int32_t*)QStart, Addend, Residual, Out, \
        T, D, H, stream_off, stream_w, sx, sw0, sw1, scb, sch, scn, sbt, max_blocks, N_ring, scale, pos_i64)
    if (!has_addend) {
        if (out_f32) LSC(false, false, false, false, true);
        else         LSC(false, false, false, false, false);
    } else if (!has_res) {
        int m = (add_f32 ? 2 : 0) | (out_f32 ? 1 : 0);
        switch (m) {
            case 0: LSC(true, false, false, false, false); break;
            case 1: LSC(true, false, false, false, true);  break;
            case 2: LSC(true, false, true,  false, false); break;
            case 3: LSC(true, false, true,  false, true);  break;
        }
    } else {
        int m = (add_f32 ? 4 : 0) | (res_f32 ? 2 : 0) | (out_f32 ? 1 : 0);
        switch (m) {
            case 0: LSC(true, true, false, false, false); break;
            case 1: LSC(true, true, false, false, true);  break;
            case 2: LSC(true, true, false, true,  false); break;
            case 3: LSC(true, true, false, true,  true);  break;
            case 4: LSC(true, true, true,  false, false); break;
            case 5: LSC(true, true, true,  false, true);  break;
            case 6: LSC(true, true, true,  true,  false); break;
            case 7: LSC(true, true, true,  true,  true);  break;
        }
    }
#undef LSC
    return 0;
}

int launch_hip_loop_decode_down(
    const void* Pair, const void* W, void* Out,
    int streams, int M, int K, void* stream_ptr
) {
    if (streams <= 0 || M <= 0 || K <= 0 || (K & 1) != 0) return -1;
    hipLaunchKernelGGL(hip_loop_decode_down_kernel, dim3(64, streams), dim3(128), 0, (hipStream_t)stream_ptr,
        (const uint32_t*)Pair, (const uint32_t*)W, (uint16_t*)Out, M, K >> 1);
    return 0;
}

int launch_hip_loop_decode_blend_rms(
    const void* Hidden, const void* Wup, const void* Dest, const void* Msg,
    const void* NormW, void* Out,
    int64_t sd_s, int64_t sd_t, int64_t sm_s, int64_t sm_t, int64_t sn_s,
    int streams, int M, int nw_f32, float norm_eps, void* stream_ptr
) {
    if (streams <= 0 || M <= 0) return -1;
    dim3 grid(M, streams), block(288);
    if (nw_f32) {
        hipLaunchKernelGGL((hip_loop_decode_blend_rms_kernel<true>), grid, block, 0, (hipStream_t)stream_ptr,
            (const uint16_t*)Hidden, (const uint16_t*)Wup, (const float*)Dest, (const float*)Msg,
            NormW, (uint16_t*)Out, sd_s, sd_t, sm_s, sm_t, sn_s, M, norm_eps);
    } else {
        hipLaunchKernelGGL((hip_loop_decode_blend_rms_kernel<false>), grid, block, 0, (hipStream_t)stream_ptr,
            (const uint16_t*)Hidden, (const uint16_t*)Wup, (const float*)Dest, (const float*)Msg,
            NormW, (uint16_t*)Out, sd_s, sd_t, sm_s, sm_t, sn_s, M, norm_eps);
    }
    return 0;
}

int launch_hip_loop_decode_blend_rms_2pass(
    const void* Hidden, const void* Wup, const void* Dest, const void* Msg,
    const void* NormW, void* Out, void* BlendedTmp, void* PartSq,
    int64_t sd_s, int64_t sd_t, int64_t sm_s, int64_t sm_t, int64_t sn_s,
    int streams, int M, int nw_f32, float norm_eps, void* stream_ptr
) {
    if (streams <= 0 || M <= 0 || M > 64) return -1;
    hipStream_t stream = (hipStream_t)stream_ptr;
    hipLaunchKernelGGL(hip_loop_decode_blend_p1_kernel, dim3(72, streams), dim3(32), 0, stream,
        (const uint16_t*)Hidden, (const uint16_t*)Wup, (const float*)Dest, (const float*)Msg,
        (float*)BlendedTmp, (float*)PartSq, sd_s, sd_t, sm_s, sm_t, M);
    dim3 grid2(M, streams), block2(288);
    if (nw_f32) {
        hipLaunchKernelGGL((hip_loop_decode_blend_p2_kernel<true>), grid2, block2, 0, stream,
            (const float*)BlendedTmp, (const float*)PartSq, NormW, (uint16_t*)Out, sn_s, M, norm_eps);
    } else {
        hipLaunchKernelGGL((hip_loop_decode_blend_p2_kernel<false>), grid2, block2, 0, stream,
            (const float*)BlendedTmp, (const float*)PartSq, NormW, (uint16_t*)Out, sn_s, M, norm_eps);
    }
    return 0;
}

int launch_hip_pair_rmsnorm(
    const void* Dest, const void* Msg, void* Out,
    int64_t sd_s, int64_t sd_t, int64_t sm_s, int64_t sm_t,
    int streams, int M, void* stream_ptr
) {
    if (streams <= 0 || M <= 0) return -1;
    hipLaunchKernelGGL(hip_pair_rmsnorm_kernel, dim3(M, streams), dim3(288), 0, (hipStream_t)stream_ptr,
        (const float*)Dest, (const float*)Msg, (uint16_t*)Out, sd_s, sd_t, sm_s, sm_t, M);
    return 0;
}

int launch_hip_blend_rmsnorm(
    const void* Dest, const void* Msg, const void* GF, const void* NormW, void* Out,
    int64_t sd_s, int64_t sd_t, int64_t sm_s, int64_t sm_t, int64_t sn_s,
    int streams, int M, int nw_f32, float norm_eps, void* stream_ptr
) {
    if (streams <= 0 || M <= 0) return -1;
    dim3 grid(M, streams), block(288);
    if (nw_f32) {
        hipLaunchKernelGGL((hip_blend_rmsnorm_kernel<true>), grid, block, 0, (hipStream_t)stream_ptr,
            (const float*)Dest, (const float*)Msg, (const float*)GF, NormW, (uint16_t*)Out,
            sd_s, sd_t, sm_s, sm_t, sn_s, M, norm_eps);
    } else {
        hipLaunchKernelGGL((hip_blend_rmsnorm_kernel<false>), grid, block, 0, (hipStream_t)stream_ptr,
            (const float*)Dest, (const float*)Msg, (const float*)GF, NormW, (uint16_t*)Out,
            sd_s, sd_t, sm_s, sm_t, sn_s, M, norm_eps);
    }
    return 0;
}

int launch_hip_router_gemm_m8(
    const void* X, const void* W, const void* Bias, void* Out, void* Corrected,
    int M, int fused_sigmoid_bias, void* stream_ptr
) {
    if (M <= 0 || M > 8) return -1;
    if (fused_sigmoid_bias) {
        hipLaunchKernelGGL((hip_router_gemm_m8_kernel<true>), dim3(128), dim3(128), 0, (hipStream_t)stream_ptr,
            (const uint32_t*)X, (const uint32_t*)W, (const float*)Bias, (float*)Out, (float*)Corrected, M);
    } else {
        hipLaunchKernelGGL((hip_router_gemm_m8_kernel<false>), dim3(128), dim3(128), 0, (hipStream_t)stream_ptr,
            (const uint32_t*)X, (const uint32_t*)W, nullptr, (float*)Out, nullptr, M);
    }
    return 0;
}

int launch_hip_route_m8(
    const void* Score, const void* Bias, void* Ids, void* Gamma,
    void* Sorted, void* Experts, void* Padded,
    int M, int do_align, int do_sigmoid, int bias_is_corrected, float scale, void* stream_ptr
) {
    if (M <= 0 || M > 8) return -1;
    hipStream_t s = (hipStream_t)stream_ptr;
    if (bias_is_corrected) {
        hipLaunchKernelGGL((hip_route_m8_kernel<false, false, true>), dim3(1), dim3(256), 0, s,
            (const float*)Score, (const float*)Bias, (int64_t*)Ids, (float*)Gamma, nullptr, nullptr, nullptr, M, scale);
    } else if (do_align) {
        if (do_sigmoid) hipLaunchKernelGGL((hip_route_m8_kernel<true, true, false>),  dim3(1), dim3(256), 0, s, (const float*)Score, (const float*)Bias, (int64_t*)Ids, (float*)Gamma, (int32_t*)Sorted, (int32_t*)Experts, (int32_t*)Padded, M, scale);
        else            hipLaunchKernelGGL((hip_route_m8_kernel<true, false, false>), dim3(1), dim3(256), 0, s, (const float*)Score, (const float*)Bias, (int64_t*)Ids, (float*)Gamma, (int32_t*)Sorted, (int32_t*)Experts, (int32_t*)Padded, M, scale);
    } else {
        if (do_sigmoid) hipLaunchKernelGGL((hip_route_m8_kernel<false, true, false>),  dim3(1), dim3(256), 0, s, (const float*)Score, (const float*)Bias, (int64_t*)Ids, (float*)Gamma, nullptr, nullptr, nullptr, M, scale);
        else            hipLaunchKernelGGL((hip_route_m8_kernel<false, false, false>), dim3(1), dim3(256), 0, s, (const float*)Score, (const float*)Bias, (int64_t*)Ids, (float*)Gamma, nullptr, nullptr, nullptr, M, scale);
    }
    return 0;
}

int launch_hip_mix_rmsnorm(
    const void* X, const void* Alpha, const void* Weight, void* Out,
    int64_t s_row, int64_t s_tap, int64_t s_hid,
    int rows, int taps, int hidden, float eps, void* stream_ptr
) {
    if (rows <= 0 || taps <= 0 || hidden <= 0) return 0;
    hipLaunchKernelGGL(hip_mix_rmsnorm_kernel, dim3(rows), dim3(256), 0, (hipStream_t)stream_ptr,
        (const uint16_t*)X, (const float*)Alpha, (const uint16_t*)Weight, (uint16_t*)Out,
        s_row, s_tap, s_hid, taps, hidden, eps);
    return 0;
}

int launch_hip_rms_norm(
    const void* X, const void* Weight, void* Out,
    int64_t stride_x, int rows, int hidden, int in_f32, int out_f32, float eps, void* stream_ptr
) {
    if (rows <= 0 || hidden <= 0) return 0;
    hipStream_t s = (hipStream_t)stream_ptr;
    if (in_f32 && out_f32)       hipLaunchKernelGGL((hip_rms_norm_kernel<true, true>),   dim3(rows), dim3(256), 0, s, X, (const uint16_t*)Weight, Out, stride_x, hidden, eps);
    else if (in_f32 && !out_f32) hipLaunchKernelGGL((hip_rms_norm_kernel<true, false>),  dim3(rows), dim3(256), 0, s, X, (const uint16_t*)Weight, Out, stride_x, hidden, eps);
    else if (!in_f32 && out_f32) hipLaunchKernelGGL((hip_rms_norm_kernel<false, true>),  dim3(rows), dim3(256), 0, s, X, (const uint16_t*)Weight, Out, stride_x, hidden, eps);
    else                         hipLaunchKernelGGL((hip_rms_norm_kernel<false, false>), dim3(rows), dim3(256), 0, s, X, (const uint16_t*)Weight, Out, stride_x, hidden, eps);
    return 0;
}

int launch_hip_qk_norm_rope(
    void* X, const void* QW, const void* KW, const void* CosSin, const void* Pos,
    int64_t row_stride, int64_t q_size, int T, int nhq, int nhk, int D, int do_norm, float eps, void* stream_ptr
) {
    if (T <= 0 || (nhq + nhk) <= 0 || D <= 0 || D > 128 || (D & 1) != 0) return -1;
    dim3 grid(T, nhq + nhk), block(64);
    if (do_norm) {
        hipLaunchKernelGGL((hip_qk_norm_rope_kernel<true>), grid, block, 0, (hipStream_t)stream_ptr,
            (uint16_t*)X, (const uint16_t*)QW, (const uint16_t*)KW, (const float*)CosSin, (const int64_t*)Pos,
            row_stride, q_size, nhq, D, eps);
    } else {
        hipLaunchKernelGGL((hip_qk_norm_rope_kernel<false>), grid, block, 0, (hipStream_t)stream_ptr,
            (uint16_t*)X, nullptr, nullptr, (const float*)CosSin, (const int64_t*)Pos,
            row_stride, q_size, nhq, D, eps);
    }
    return 0;
}

int launch_hip_dspark_w8(
    const void* X, const void* Codes, const void* Scales, void* Out, void* Partial,
    int M, int N, int K, int splits, void* stream_ptr
) {
    if (M <= 0 || M > 16 || N <= 0 || K <= 0 || (N & 63) != 0 || (K & 127) != 0) return -1;
    if (splits != 1 && splits != 2 && splits != 4 && splits != 8) return -1;
    if (((K >> 7) % splits) != 0) return -1;
    hipStream_t s = (hipStream_t)stream_ptr;
    dim3 grid(N / 64, splits), block(128);
    void* dst = (splits > 1) ? Partial : Out;
    if (splits == 1)      hipLaunchKernelGGL((hip_dspark_w8_kernel<1>), grid, block, 0, s, (const uint16_t*)X, (const int8_t*)Codes, (const uint16_t*)Scales, dst, M, N, K);
    else if (splits == 2) hipLaunchKernelGGL((hip_dspark_w8_kernel<2>), grid, block, 0, s, (const uint16_t*)X, (const int8_t*)Codes, (const uint16_t*)Scales, dst, M, N, K);
    else if (splits == 4) hipLaunchKernelGGL((hip_dspark_w8_kernel<4>), grid, block, 0, s, (const uint16_t*)X, (const int8_t*)Codes, (const uint16_t*)Scales, dst, M, N, K);
    else                  hipLaunchKernelGGL((hip_dspark_w8_kernel<8>), grid, block, 0, s, (const uint16_t*)X, (const int8_t*)Codes, (const uint16_t*)Scales, dst, M, N, K);
    if (splits > 1) {
        return launch_hip_reduce_splits(Partial, Out, (int64_t)M * N, splits, 0, 0.0f, stream_ptr);
    }
    return 0;
}

int launch_hip_silu_and_mul(
    const void* X, void* Y, int64_t stride_x, int rows, int N, void* stream_ptr
) {
    if (rows <= 0 || N <= 0) return 0;
    dim3 grid((N + 1023) / 1024, rows), block(256);
    hipLaunchKernelGGL(hip_silu_and_mul_kernel, grid, block, 0, (hipStream_t)stream_ptr,
        (const uint16_t*)X, (uint16_t*)Y, stride_x, N);
    return 0;
}

int launch_hip_add_softcap(
    const void* A, const void* B, void* Out,
    int64_t stride_a, int64_t stride_b, int rows, int N, float softcap, void* stream_ptr
) {
    if (rows <= 0 || N <= 0) return 0;
    dim3 grid((N + 1023) / 1024, rows), block(256);
    hipLaunchKernelGGL(hip_add_softcap_kernel, grid, block, 0, (hipStream_t)stream_ptr,
        (const uint16_t*)A, (const uint16_t*)B, (uint16_t*)Out, stride_a, stride_b, N, softcap);
    return 0;
}

int launch_hip_markov_gate(
    const void* Raw, const void* State, void* Out, void* Latent,
    int rows, int rank, void* stream_ptr
) {
    if (rows <= 0 || rank <= 0 || rank > 256) return -1;
    hipLaunchKernelGGL(hip_markov_gate_kernel, dim3(rows), dim3(256), 0, (hipStream_t)stream_ptr,
        (const float*)Raw, (const uint16_t*)State, (uint16_t*)Out, (uint16_t*)Latent, rank);
    return 0;
}

}  // extern "C"
