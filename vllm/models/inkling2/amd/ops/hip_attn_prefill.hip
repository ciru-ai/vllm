// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 CIRU AI.
// Native RDNA 3.5 (gfx1151) attention prefill kernels for Inkling 2 Mini.

#include "hip_common.cuh"

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
    const void* __restrict__ Pos,
    const int32_t* __restrict__ SeqIdx,
    const int64_t* __restrict__ ConvSlots,
    const int32_t* __restrict__ BlockTable,
    const int32_t* __restrict__ QStart,
    int T, int KVH, int64_t stride_k, int64_t stride_v,
    int64_t stride_w0, int64_t stride_w1, int64_t stride_dk, int64_t stride_dv,
    int64_t stride_c_blk, int64_t stride_c_h, int64_t stride_c_n,
    int64_t stride_kv_blk, int64_t stride_kv_n, int64_t stride_kv_h,
    int kv_block_size, int64_t stride_bt, int max_blocks, int N_conv,
    float scale, float clip_threshold, float clip_eps, float norm_eps, int pos_i64
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
    const int pos0 = pos_i64 ? (int)((const int64_t*)Pos)[t0] : ((const int32_t*)Pos)[t0];
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
        int pos = pos_i64 ? (int)((const int64_t*)Pos)[t] : ((const int32_t*)Pos)[t];
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


extern "C" {

void launch_hip_kv_conv_1w(
    const void* K, const void* V, const void* Wk, const void* Wv, const void* Dk, const void* Dv,
    void* CC, void* KC, void* VC, const void* KVS, const void* Pos, const void* Seq, const void* CS,
    const void* BT_ptr, const void* QS,
    int T, int KVH, int64_t sk, int64_t sv, int64_t sw0, int64_t sw1, int64_t sdk, int64_t sdv,
    int64_t scb, int64_t sch, int64_t scn, int64_t skvb, int64_t skvn, int64_t skvh,
    int kv_bs, int64_t sbt, int mb, int nc, float scale, float ct, float ce, float ne,
    int dynamic, int pos_i64, hipStream_t stream
) {
    constexpr int BT = 64;
    int n_tiles = (T + BT - 1) / BT;
    dim3 grid((n_tiles + 3) / 4, KVH);
    dim3 block(128);
    if (dynamic) {
        hipLaunchKernelGGL((hip_kv_conv_1warp_kernel<BT, 1>), grid, block, 0, stream,
            (const uint16_t*)K, (const uint16_t*)V, (const uint16_t*)Wk, (const uint16_t*)Wv,
            (const uint16_t*)Dk, (const uint16_t*)Dv, (uint16_t*)CC, (uint16_t*)KC, (uint16_t*)VC,
            (const int64_t*)KVS, Pos, (const int32_t*)Seq, (const int64_t*)CS,
            (const int32_t*)BT_ptr, (const int32_t*)QS, T, KVH, sk, sv, sw0, sw1, sdk, sdv,
            scb, sch, scn, skvb, skvn, skvh, kv_bs, sbt, mb, nc, scale, ct, ce, ne, pos_i64);
    } else {
        hipLaunchKernelGGL((hip_kv_conv_1warp_kernel<BT, 0>), grid, block, 0, stream,
            (const uint16_t*)K, (const uint16_t*)V, (const uint16_t*)Wk, (const uint16_t*)Wv,
            (const uint16_t*)Dk, (const uint16_t*)Dv, (uint16_t*)CC, (uint16_t*)KC, (uint16_t*)VC,
            (const int64_t*)KVS, Pos, (const int32_t*)Seq, (const int64_t*)CS,
            (const int32_t*)BT_ptr, (const int32_t*)QS, T, KVH, sk, sv, sw0, sw1, sdk, sdv,
            scb, sch, scn, skvb, skvn, skvh, kv_bs, sbt, mb, nc, scale, ct, ce, ne, pos_i64);
    }
}


}

// ============================================================================
// B1: Exact Per-Head Per-Channel Q/R L2-Clip (16 lanes/head, uint4 Q, clip before tau)
// ============================================================================

template<bool HAS_TAU, bool R_FP16>
__global__ __launch_bounds__(128, 8)
void hip_qr_clip_kernel(
    const uint16_t* __restrict__ Q,       // [T, H, 128] BF16 (strided)
    const uint16_t* __restrict__ R,       // [T, H, 16]  BF16 (strided)
    const uint16_t* __restrict__ QScale,  // [H, 128]    BF16
    const uint16_t* __restrict__ RScale,  // [H, 16]     BF16
    const float*    __restrict__ Tau,     // [T]         FP32 (optional)
    uint16_t*       __restrict__ QOut,    // [T, H, 128] BF16 contiguous
    uint16_t*       __restrict__ ROut,    // [T, H, 16]  BF16 or FP16 contiguous
    int T, int H,
    int64_t stride_qt, int64_t stride_qh,
    int64_t stride_rt, int64_t stride_rh,
    float q_thresh, float r_thresh, float eps
) {
    const int t = blockIdx.x;
    const int half_id = threadIdx.x >> 4; // 0..7 heads per CTA
    const int lane16  = threadIdx.x & 15;
    const int h = blockIdx.y * 8 + half_id;
    if (t >= T || h >= H) return;

    const float tau = HAS_TAU ? Tau[t] : 1.0f;

    // --- Q [128]: 16 lanes x 8 BF16s (128-byte coalesced uint4) ---
    const int d0 = lane16 * 8;
    const uint16_t* q_ptr  = Q + (int64_t)t * stride_qt + (int64_t)h * stride_qh + d0;
    const uint16_t* qs_ptr = QScale + (int64_t)h * 128 + d0;

    float qv[8];
    if ((((uintptr_t)q_ptr | (uintptr_t)qs_ptr) & 15u) == 0u) {
        uint4 qu  = *reinterpret_cast<const uint4*>(q_ptr);
        uint4 qsu = *reinterpret_cast<const uint4*>(qs_ptr);
        uint32_t qw[4]  = {qu.x, qu.y, qu.z, qu.w};
        uint32_t qsw[4] = {qsu.x, qsu.y, qsu.z, qsu.w};
        #pragma unroll
        for (int i = 0; i < 4; ++i) {
            qv[2 * i + 0] = bf16_to_f32((uint16_t)(qw[i] & 0xFFFFu)) * bf16_to_f32((uint16_t)(qsw[i] & 0xFFFFu));
            qv[2 * i + 1] = bf16_to_f32((uint16_t)(qw[i] >> 16))     * bf16_to_f32((uint16_t)(qsw[i] >> 16));
        }
    } else {
        #pragma unroll
        for (int i = 0; i < 8; ++i) {
            qv[i] = bf16_to_f32(q_ptr[i]) * bf16_to_f32(qs_ptr[i]);
        }
    }

    float q_ss = 0.0f;
    #pragma unroll
    for (int i = 0; i < 8; ++i) q_ss = fmaf(qv[i], qv[i], q_ss);
    #pragma unroll
    for (int off = 8; off > 0; off >>= 1) q_ss += __shfl_xor(q_ss, off, 16);
    float q_norm = (q_ss > 0.0f) ? (q_ss * rsqrtf(q_ss)) : 0.0f;
    float q_sc = fminf(q_thresh / (q_norm + eps), 1.0f);

    uint32_t qo_w[4];
    #pragma unroll
    for (int i = 0; i < 4; ++i) {
        float y0 = qv[2 * i + 0] * q_sc;
        float y1 = qv[2 * i + 1] * q_sc;
        if constexpr (HAS_TAU) {
            y0 = bf16_round_f32(y0) * tau;
            y1 = bf16_round_f32(y1) * tau;
        }
        qo_w[i] = pack_bf16x2(y0, y1);
    }
    *reinterpret_cast<uint4*>(QOut + ((int64_t)t * H + h) * 128 + d0) =
        make_uint4(qo_w[0], qo_w[1], qo_w[2], qo_w[3]);

    // --- R [16]: 16 lanes x 1 element (100% wave utilization) ---
    float rv = bf16_to_f32(R[(int64_t)t * stride_rt + (int64_t)h * stride_rh + lane16])
             * bf16_to_f32(RScale[(int64_t)h * 16 + lane16]);
    float r_ss = rv * rv;
    #pragma unroll
    for (int off = 8; off > 0; off >>= 1) r_ss += __shfl_xor(r_ss, off, 16);
    float r_norm = (r_ss > 0.0f) ? (r_ss * rsqrtf(r_ss)) : 0.0f;
    float r_sc = fminf(r_thresh / (r_norm + eps), 1.0f);
    float ry = rv * r_sc;
    if constexpr (HAS_TAU) {
        ry = bf16_round_f32(ry) * tau;
    }
    uint16_t out_bits;
    if constexpr (R_FP16) {
        _Float16 hf = (_Float16)ry;
        __builtin_memcpy(&out_bits, &hf, 2);
    } else {
        out_bits = f32_to_bf16_rtne(ry);
    }
    ROut[((int64_t)t * H + h) * 16 + lane16] = out_bits;
}

// ============================================================================
// B2 Helper: Paged KV -> Contiguous HND Gather (for Long-Context Hybrid Far)
// ============================================================================

__global__ __launch_bounds__(128, 4)
void hip_gather_paged_kv_hnd_kernel(
    const uint16_t* __restrict__ KCache,
    const uint16_t* __restrict__ VCache,
    const int32_t*  __restrict__ BlockTableRow,
    uint16_t*       __restrict__ KOut,
    uint16_t*       __restrict__ VOut,
    int n_pages, int page_size,
    int64_t skb, int64_t skp, int64_t skh,
    int64_t svb, int64_t svp, int64_t svh,
    int64_t sokh, int64_t sokn,
    int64_t sovh, int64_t sovn
) {
    const int pg_i = blockIdx.x;
    const int kvh = blockIdx.y;
    if (pg_i >= n_pages) return;
    const int64_t page_id = (int64_t)BlockTableRow[pg_i];
    const int tid = threadIdx.x;
    const int d0 = (tid & 15) * 8;
    const int row_base = tid >> 4;
    const uint16_t* k_src_pg = KCache + page_id * skb + (int64_t)kvh * skh + d0;
    const uint16_t* v_src_pg = VCache + page_id * svb + (int64_t)kvh * svh + d0;
    uint16_t* k_dst_pg = KOut + (int64_t)kvh * sokh + (int64_t)(pg_i * page_size) * sokn + d0;
    uint16_t* v_dst_pg = VOut + (int64_t)kvh * sovh + (int64_t)(pg_i * page_size) * sovn + d0;
    for (int p = row_base; p < page_size; p += 8) {
        *reinterpret_cast<uint4*>(k_dst_pg + (int64_t)p * sokn) =
            *reinterpret_cast<const uint4*>(k_src_pg + (int64_t)p * skp);
        *reinterpret_cast<uint4*>(v_dst_pg + (int64_t)p * sovn) =
            *reinterpret_cast<const uint4*>(v_src_pg + (int64_t)p * svp);
    }
}

// ============================================================================
// B2: Zero-Materialization BF16-WMMA Flash-Rel-Attention
//   - Swapped K@Q^T and V^T@P^T: Q[16,128] in registers, P transposed in-reg!
//   - Non-aliasing smem_K[16][136] (4.25 KiB) + zero-conflict smem_VT (6.25 KiB)
//     + smem_R[64][16] (2.00 KiB) = 12.50 KiB LDS total -> 2 barriers/tile!
//   - Local: 191 VGPRs, 8 waves/SIMD, 0 spills; Global: 7-8 waves/SIMD, 0 spills.
// ============================================================================

template<bool R_FP16>
__device__ __forceinline__ float dot16_packed(uint4 r0, uint4 r1, uint4 p0, uint4 p1) {
    float acc = 0.0f;
    if constexpr (R_FP16) {
        asm("v_dot2_f32_f16 %0, %1, %2, %0" : "+v"(acc) : "v"(r0.x), "v"(p0.x));
        asm("v_dot2_f32_f16 %0, %1, %2, %0" : "+v"(acc) : "v"(r0.y), "v"(p0.y));
        asm("v_dot2_f32_f16 %0, %1, %2, %0" : "+v"(acc) : "v"(r0.z), "v"(p0.z));
        asm("v_dot2_f32_f16 %0, %1, %2, %0" : "+v"(acc) : "v"(r0.w), "v"(p0.w));
        asm("v_dot2_f32_f16 %0, %1, %2, %0" : "+v"(acc) : "v"(r1.x), "v"(p1.x));
        asm("v_dot2_f32_f16 %0, %1, %2, %0" : "+v"(acc) : "v"(r1.y), "v"(p1.y));
        asm("v_dot2_f32_f16 %0, %1, %2, %0" : "+v"(acc) : "v"(r1.z), "v"(p1.z));
        asm("v_dot2_f32_f16 %0, %1, %2, %0" : "+v"(acc) : "v"(r1.w), "v"(p1.w));
    } else {
        asm("v_dot2_f32_bf16 %0, %1, %2, %0" : "+v"(acc) : "v"(r0.x), "v"(p0.x));
        asm("v_dot2_f32_bf16 %0, %1, %2, %0" : "+v"(acc) : "v"(r0.y), "v"(p0.y));
        asm("v_dot2_f32_bf16 %0, %1, %2, %0" : "+v"(acc) : "v"(r0.z), "v"(p0.z));
        asm("v_dot2_f32_bf16 %0, %1, %2, %0" : "+v"(acc) : "v"(r0.w), "v"(p0.w));
        asm("v_dot2_f32_bf16 %0, %1, %2, %0" : "+v"(acc) : "v"(r1.x), "v"(p1.x));
        asm("v_dot2_f32_bf16 %0, %1, %2, %0" : "+v"(acc) : "v"(r1.y), "v"(p1.y));
        asm("v_dot2_f32_bf16 %0, %1, %2, %0" : "+v"(acc) : "v"(r1.z), "v"(p1.z));
        asm("v_dot2_f32_bf16 %0, %1, %2, %0" : "+v"(acc) : "v"(r1.w), "v"(p1.w));
    }
    return acc;
}

__device__ __forceinline__ short16 load_short16_u4(const uint16_t* ptr) {
    uint4 lo = *reinterpret_cast<const uint4*>(ptr);
    uint4 hi = *reinterpret_cast<const uint4*>(ptr + 8);
    short16 out;
    uint32_t words[8] = {lo.x, lo.y, lo.z, lo.w, hi.x, hi.y, hi.z, hi.w};
    __builtin_memcpy(&out, words, 32);
    return out;
}

// Zero-bank-conflict VT: 16 groups of 8 d-rows, 100 dwords/group (8*12 + 4 pad = 400 B, 16B aligned).
__device__ __forceinline__ const uint16_t* vt_row_ptr(const uint32_t* vt_base, int d) {
    return reinterpret_cast<const uint16_t*>(vt_base + (d >> 3) * 100 + (d & 7) * 12);
}

__device__ __forceinline__ void store_vt_zconf(
    uint32_t* vt_base, int d_blk, int pair, uint4 va, uint4 vb
) {
    uint32_t* grp = vt_base + d_blk * 100;
    uint32_t a[4] = {va.x, va.y, va.z, va.w};
    uint32_t b[4] = {vb.x, vb.y, vb.z, vb.w};
    #pragma unroll
    for (int j = 0; j < 4; ++j) {
        uint32_t lo = (a[j] & 0xFFFFu) | ((b[j] & 0xFFFFu) << 16);
        uint32_t hi = (a[j] >> 16)     | (b[j] & 0xFFFF0000u);
        grp[(2 * j + 0) * 12 + pair] = lo;
        grp[(2 * j + 1) * 12 + pair] = hi;
    }
}

__device__ __forceinline__ short16 pack_p_in_reg(const float p_f[8], int half) {
    uint32_t my_p[4] = {
        pack_bf16x2(p_f[0], p_f[1]),
        pack_bf16x2(p_f[2], p_f[3]),
        pack_bf16x2(p_f[4], p_f[5]),
        pack_bf16x2(p_f[6], p_f[7]),
    };
    uint32_t oth_p[4] = {
        __shfl_xor(my_p[0], 16, 32),
        __shfl_xor(my_p[1], 16, 32),
        __shfl_xor(my_p[2], 16, 32),
        __shfl_xor(my_p[3], 16, 32),
    };
    uint32_t w[8];
    #pragma unroll
    for (int i = 0; i < 4; ++i) {
        uint32_t ev = (half == 0) ? my_p[i] : oth_p[i];
        uint32_t od = (half == 0) ? oth_p[i] : my_p[i];
        w[2 * i + 0] = (ev & 0xFFFFu) | ((od & 0xFFFFu) << 16);
        w[2 * i + 1] = (ev >> 16)     | (od & 0xFFFF0000u);
    }
    short16 out;
    __builtin_memcpy(&out, w, 32);
    return out;
}

template<bool IS_LOCAL, bool HYBRID_MERGE, bool R_FP16, bool ROUND_BIAS_BF16, int PAGE_SHIFT, int MIN_CTAS>
__global__ __launch_bounds__(128, MIN_CTAS)
void hip_flash_rel_attn_prefill_kernel(
    const uint16_t* __restrict__ Q,
    const uint16_t* __restrict__ R,
    const uint16_t* __restrict__ RelProjT,
    const uint16_t* __restrict__ KCache,
    const uint16_t* __restrict__ VCache,
    const uint16_t* __restrict__ Gate,
    const uint16_t* __restrict__ Sink,
    const uint16_t* __restrict__ OutFar,
    const float*    __restrict__ LseFar,
    uint16_t*       __restrict__ Result,
    const int32_t*  __restrict__ BlockTable,
    const int32_t*  __restrict__ SeqLens,
    const int32_t*  __restrict__ CuSeqlensQ,
    float scale_log2,
    int far_len_arg, int req_idx_arg,
    int64_t sqt, int64_t sqh,
    int64_t srt, int64_t srh,
    int64_t skb, int64_t skp, int64_t skh,
    int64_t svb, int64_t svp, int64_t svh,
    int64_t sgate, int64_t ssink,
    int64_t sofh, int64_t sofq,
    int64_t slfh, int64_t slfq,
    int64_t sres, int64_t sbt,
    int H, int GROUP, int PAGE, int EXTENT, int WINDOW
) {
    constexpr float LOG2E = 1.4426950408889634f;
    constexpr float LN2   = 0.6931471805599453f;
    constexpr int PAGE_MASK = (PAGE_SHIFT > 0) ? ((1 << PAGE_SHIFT) - 1) : 0;

    const int head = blockIdx.x;
    const int q_block = blockIdx.y;
    const int req = HYBRID_MERGE ? req_idx_arg : (int)blockIdx.z;
    const int kv_head = head / GROUP;

    const int q_start = CuSeqlensQ[req];
    const int q_len = CuSeqlensQ[req + 1] - q_start;
    const int q0 = q_block * 64;
    if (q0 >= q_len) return;

    const int seq_len = SeqLens[req];
    const int past = seq_len - q_len;
    const int pos0 = past + q0;
    const int hi = min(pos0 + 64, seq_len);
    const int64_t tok0 = (int64_t)(q_start + q0);

    const int tid = threadIdx.x;
    const int warp = tid >> 5;
    const int lane = tid & 31;
    const int lane16 = lane & 15;
    const int half = lane >> 4;

    const int m_cta = warp * 16 + lane16;
    const int q_idx = q0 + m_cta;
    const bool rvalid = (q_idx < q_len);
    const int pos = past + q_idx;
    const int64_t tok = tok0 + m_cta;

    __shared__ uint16_t smem_K[16][136];    // 4.25 KiB
    __shared__ uint32_t smem_VT[16 * 100];  // 6.25 KiB
    __shared__ uint16_t smem_R[64][16];     // 2.00 KiB -> 12.50 KiB total!

    // 1. Load Q[m_cta, 0..127] scaled by scale_log2 into registers (0 LDS!)
    short16 q_reg[8];
    if (rvalid) {
        const uint16_t* q_row = Q + tok * sqt + (int64_t)head * sqh;
        #pragma unroll
        for (int kd = 0; kd < 8; ++kd) {
            uint4 q0_u4 = *reinterpret_cast<const uint4*>(q_row + kd * 16);
            uint4 q1_u4 = *reinterpret_cast<const uint4*>(q_row + kd * 16 + 8);
            uint32_t w[8] = {q0_u4.x, q0_u4.y, q0_u4.z, q0_u4.w, q1_u4.x, q1_u4.y, q1_u4.z, q1_u4.w};
            #pragma unroll
            for (int j = 0; j < 8; ++j) {
                w[j] = pack_bf16x2(
                    bf16_to_f32((uint16_t)(w[j] & 0xFFFFu)) * scale_log2,
                    bf16_to_f32((uint16_t)(w[j] >> 16)) * scale_log2);
            }
            __builtin_memcpy(&q_reg[kd], w, 32);
        }
    } else {
        #pragma unroll
        for (int kd = 0; kd < 8; ++kd) q_reg[kd] = {};
    }

    // 2. Initialize online softmax state
    float m_i = -INFINITY;
    float l_i = 0.0f;
    float8 acc_o[8];
    if constexpr (HYBRID_MERGE) {
        if (rvalid) {
            float lf = LseFar[(int64_t)head * slfh + (int64_t)q_idx * slfq];
            bool has_far = (lf > -INFINITY);
            m_i = has_far ? (lf * LOG2E) : -INFINITY;
            l_i = has_far ? 1.0f : 0.0f;
            const uint16_t* of_row = OutFar + (int64_t)head * sofh + (int64_t)q_idx * sofq;
            #pragma unroll
            for (int dt = 0; dt < 8; ++dt) {
                #pragma unroll
                for (int r = 0; r < 8; ++r) {
                    acc_o[dt][r] = has_far ? bf16_to_f32(of_row[dt * 16 + half + 2 * r]) : 0.0f;
                }
            }
        } else {
            #pragma unroll
            for (int dt = 0; dt < 8; ++dt) acc_o[dt] = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
        }
    } else {
        #pragma unroll
        for (int dt = 0; dt < 8; ++dt) acc_o[dt] = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
    }

    const int32_t* table = BlockTable + (int64_t)req * sbt;
    const uint16_t* k_hbase = KCache + (int64_t)kv_head * skh;
    const uint16_t* v_hbase = VCache + (int64_t)kv_head * svh;

    const int pair  = tid >> 4;
    const int d_blk = tid & 15;
    const int d0    = d_blk * 8;
    const int r0    = pair * 2;
    const int r1    = pair * 2 + 1;

    int near_start = 0;
    if (hi > 0) {
        if constexpr (!IS_LOCAL) {
            int far_start = HYBRID_MERGE ? far_len_arg : 0;
            int far_end = max(pos0 - EXTENT + 1, 0) & ~15;
            if (far_end < far_start) far_end = far_start;

            // ---- 3A. Global Far Phase (BN = 16, 2 barriers/tile, zero relative bias) ----
            for (int key_begin = far_start; key_begin < far_end; key_begin += 16) {
                if constexpr (PAGE_SHIFT > 0) {
                    int64_t pg = (int64_t)table[key_begin >> PAGE_SHIFT];
                    int sl = key_begin & PAGE_MASK;
                    const uint16_t* kp_pg = k_hbase + pg * skb + d0;
                    *reinterpret_cast<uint4*>(&smem_K[r0][d0]) = *reinterpret_cast<const uint4*>(kp_pg + (int32_t)(sl + r0) * (int32_t)skp);
                    *reinterpret_cast<uint4*>(&smem_K[r1][d0]) = *reinterpret_cast<const uint4*>(kp_pg + (int32_t)(sl + r1) * (int32_t)skp);
                    const uint16_t* vp_pg = v_hbase + pg * svb + d0;
                    uint4 va = *reinterpret_cast<const uint4*>(vp_pg + (int32_t)(sl + r0) * (int32_t)svp);
                    uint4 vb = *reinterpret_cast<const uint4*>(vp_pg + (int32_t)(sl + r1) * (int32_t)svp);
                    store_vt_zconf(smem_VT, d_blk, pair, va, vb);
                } else {
                    int64_t pg0 = (int64_t)table[(key_begin + r0) / PAGE];
                    int64_t pg1 = (int64_t)table[(key_begin + r1) / PAGE];
                    int sl0 = (key_begin + r0) % PAGE;
                    int sl1 = (key_begin + r1) % PAGE;
                    *reinterpret_cast<uint4*>(&smem_K[r0][d0]) = *reinterpret_cast<const uint4*>(k_hbase + pg0 * skb + (int64_t)sl0 * skp + d0);
                    *reinterpret_cast<uint4*>(&smem_K[r1][d0]) = *reinterpret_cast<const uint4*>(k_hbase + pg1 * skb + (int64_t)sl1 * skp + d0);
                    uint4 va = *reinterpret_cast<const uint4*>(v_hbase + pg0 * svb + (int64_t)sl0 * svp + d0);
                    uint4 vb = *reinterpret_cast<const uint4*>(v_hbase + pg1 * svb + (int64_t)sl1 * svp + d0);
                    store_vt_zconf(smem_VT, d_blk, pair, va, vb);
                }
                __syncthreads();

                float8 s = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
                #pragma unroll
                for (int kd = 0; kd < 8; ++kd) {
                    short16 a_k = load_short16_u4(&smem_K[lane16][kd * 16]);
                    s = __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32(a_k, q_reg[kd], s);
                }

                float mx = -INFINITY;
                #pragma unroll
                for (int r = 0; r < 8; ++r) mx = fmaxf(mx, rvalid ? s[r] : -INFINITY);
                mx = fmaxf(mx, __shfl_xor(mx, 16, 32));
                float m_new = fmaxf(m_i, mx);
                float m_ref = (m_new == -INFINITY) ? 0.0f : m_new;
                float alpha = (m_i == -INFINITY) ? 0.0f : exp2f(m_i - m_ref);
                float p_f[8];
                float ps = 0.0f;
                #pragma unroll
                for (int r = 0; r < 8; ++r) {
                    p_f[r] = rvalid ? exp2f(s[r] - m_ref) : 0.0f;
                    ps += p_f[r];
                }
                ps += __shfl_xor(ps, 16, 32);
                l_i = l_i * alpha + ps;
                m_i = m_new;
                #pragma unroll
                for (int dt = 0; dt < 8; ++dt) {
                    #pragma unroll
                    for (int r = 0; r < 8; ++r) acc_o[dt][r] *= alpha;
                }
                short16 b_p = pack_p_in_reg(p_f, half);

                #pragma unroll
                for (int dt = 0; dt < 8; ++dt) {
                    short16 a_v = load_short16_u4(vt_row_ptr(smem_VT, dt * 16 + lane16));
                    acc_o[dt] = __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32(a_v, b_p, acc_o[dt]);
                }
                __syncthreads();
            }
            near_start = far_end;
        } else {
            int max_dist = min(WINDOW, EXTENT - 1);
            near_start = max(pos0 - max_dist, 0) & ~15;
        }

        // ---- 3B. Near Phase (BN = 16, 2 barriers/tile, smem_R + L1 RelProjT) ----
        if (near_start < hi) {
            {
                int rr = tid >> 1;
                int rc = (tid & 1) * 8;
                *reinterpret_cast<uint4*>(&smem_R[rr][rc]) = (q0 + rr < q_len)
                    ? *reinterpret_cast<const uint4*>(R + (tok0 + rr) * srt + (int64_t)head * srh + rc)
                    : make_uint4(0u, 0u, 0u, 0u);
            }

            for (int key_begin = near_start; key_begin < hi; key_begin += 16) {
                bool v0 = (key_begin + r0 < hi);
                bool v1 = (key_begin + r1 < hi);
                if constexpr (PAGE_SHIFT > 0) {
                    int64_t pg = (int64_t)table[key_begin >> PAGE_SHIFT];
                    int sl = key_begin & PAGE_MASK;
                    const uint16_t* kp_pg = k_hbase + pg * skb + d0;
                    *reinterpret_cast<uint4*>(&smem_K[r0][d0]) = v0 ? *reinterpret_cast<const uint4*>(kp_pg + (int32_t)(sl + r0) * (int32_t)skp) : make_uint4(0u,0u,0u,0u);
                    *reinterpret_cast<uint4*>(&smem_K[r1][d0]) = v1 ? *reinterpret_cast<const uint4*>(kp_pg + (int32_t)(sl + r1) * (int32_t)skp) : make_uint4(0u,0u,0u,0u);
                    const uint16_t* vp_pg = v_hbase + pg * svb + d0;
                    uint4 va = v0 ? *reinterpret_cast<const uint4*>(vp_pg + (int32_t)(sl + r0) * (int32_t)svp) : make_uint4(0u,0u,0u,0u);
                    uint4 vb = v1 ? *reinterpret_cast<const uint4*>(vp_pg + (int32_t)(sl + r1) * (int32_t)svp) : make_uint4(0u,0u,0u,0u);
                    store_vt_zconf(smem_VT, d_blk, pair, va, vb);
                } else {
                    int64_t pg0 = v0 ? (int64_t)table[(key_begin + r0) / PAGE] : 0;
                    int64_t pg1 = v1 ? (int64_t)table[(key_begin + r1) / PAGE] : 0;
                    int sl0 = (key_begin + r0) % PAGE;
                    int sl1 = (key_begin + r1) % PAGE;
                    *reinterpret_cast<uint4*>(&smem_K[r0][d0]) = v0 ? *reinterpret_cast<const uint4*>(k_hbase + pg0 * skb + (int64_t)sl0 * skp + d0) : make_uint4(0u,0u,0u,0u);
                    *reinterpret_cast<uint4*>(&smem_K[r1][d0]) = v1 ? *reinterpret_cast<const uint4*>(k_hbase + pg1 * skb + (int64_t)sl1 * skp + d0) : make_uint4(0u,0u,0u,0u);
                    uint4 va = v0 ? *reinterpret_cast<const uint4*>(v_hbase + pg0 * svb + (int64_t)sl0 * svp + d0) : make_uint4(0u,0u,0u,0u);
                    uint4 vb = v1 ? *reinterpret_cast<const uint4*>(v_hbase + pg1 * svb + (int64_t)sl1 * svp + d0) : make_uint4(0u,0u,0u,0u);
                    store_vt_zconf(smem_VT, d_blk, pair, va, vb);
                }
                __syncthreads();

                float8 s = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
                #pragma unroll
                for (int kd = 0; kd < 8; ++kd) {
                    short16 a_k = load_short16_u4(&smem_K[lane16][kd * 16]);
                    s = __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32(a_k, q_reg[kd], s);
                }

                float sv[8] = {s[0], s[1], s[2], s[3], s[4], s[5], s[6], s[7]};
                uint4 r_lo = *reinterpret_cast<const uint4*>(&smem_R[m_cta][0]);
                uint4 r_hi = *reinterpret_cast<const uint4*>(&smem_R[m_cta][8]);
                float mx = -INFINITY;
                bool full_interior = (q0 + 64 <= q_len)
                    && (key_begin + 15 <= pos0)
                    && ((pos0 + 63) - key_begin < EXTENT)
                    && (!IS_LOCAL || ((pos0 + 63) - key_begin <= WINDOW));
                if (full_interior) {
                    int base_dist = pos - (key_begin + half);
                    #pragma unroll 1
                    for (int r = 0; r < 8; ++r) {
                        const uint16_t* rp = RelProjT + (int64_t)(base_dist - 2 * r) * 16;
                        uint4 p0 = *reinterpret_cast<const uint4*>(rp);
                        uint4 p1 = *reinterpret_cast<const uint4*>(rp + 8);
                        float bias = dot16_packed<R_FP16>(r_lo, r_hi, p0, p1);
                        if constexpr (ROUND_BIAS_BF16) bias = bf16_round_f32(bias);
                        sv[r] += bias * LOG2E;
                        mx = fmaxf(mx, sv[r]);
                    }
                } else {
                    #pragma unroll 1
                    for (int r = 0; r < 8; ++r) {
                        int key_idx = key_begin + half + 2 * r;
                        int dist = pos - key_idx;
                        bool valid = rvalid && (key_idx < hi) && (dist >= 0);
                        if constexpr (IS_LOCAL) valid = valid && (dist <= WINDOW) && (dist < EXTENT);
                        if (valid) {
                            if (dist < EXTENT) {
                                const uint16_t* rp = RelProjT + (int64_t)dist * 16;
                                uint4 p0 = *reinterpret_cast<const uint4*>(rp);
                                uint4 p1 = *reinterpret_cast<const uint4*>(rp + 8);
                                float bias = dot16_packed<R_FP16>(r_lo, r_hi, p0, p1);
                                if constexpr (ROUND_BIAS_BF16) bias = bf16_round_f32(bias);
                                sv[r] += bias * LOG2E;
                            }
                        } else {
                            sv[r] = -INFINITY;
                        }
                        mx = fmaxf(mx, sv[r]);
                    }
                }
                mx = fmaxf(mx, __shfl_xor(mx, 16, 32));
                float m_new = fmaxf(m_i, mx);
                float m_ref = (m_new == -INFINITY) ? 0.0f : m_new;
                float alpha = (m_i == -INFINITY) ? 0.0f : exp2f(m_i - m_ref);
                float p_f[8];
                float ps = 0.0f;
                #pragma unroll
                for (int r = 0; r < 8; ++r) {
                    p_f[r] = (sv[r] == -INFINITY) ? 0.0f : exp2f(sv[r] - m_ref);
                    ps += p_f[r];
                }
                ps += __shfl_xor(ps, 16, 32);
                l_i = l_i * alpha + ps;
                m_i = m_new;
                #pragma unroll
                for (int dt = 0; dt < 8; ++dt) {
                    #pragma unroll
                    for (int r = 0; r < 8; ++r) acc_o[dt][r] *= alpha;
                }
                short16 b_p = pack_p_in_reg(p_f, half);

                #pragma unroll
                for (int dt = 0; dt < 8; ++dt) {
                    short16 a_v = load_short16_u4(vt_row_ptr(smem_VT, dt * 16 + lane16));
                    acc_o[dt] = __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32(a_v, b_p, acc_o[dt]);
                }
                __syncthreads();
            }
        }
    }

    // 4. Fused Gate + Sink Epilogue -> Result[tok, head * 128 + d]
    if (rvalid) {
        bool has_keys = (l_i > 0.0f);
        float inv_l = has_keys ? (1.0f / l_i) : 0.0f;
        float sink_sig = 0.0f;
        if (has_keys) {
            float lse_e = (m_i + log2f(l_i)) * LN2;
            float sink_val = bf16_to_f32(Sink[tok * ssink + head]);
            sink_sig = fast_sigmoid_hw(lse_e - sink_val);
        }
        const uint16_t* g_row = Gate + tok * sgate + (int64_t)head * 128;
        uint16_t* o_row = Result + tok * sres + (int64_t)head * 128;
        #pragma unroll
        for (int dt = 0; dt < 8; ++dt) {
            #pragma unroll
            for (int r = 0; r < 8; ++r) {
                int d = dt * 16 + half + 2 * r;
                float g_val = bf16_to_f32(g_row[d]);
                float y = (acc_o[dt][r] * inv_l) * fast_sigmoid_hw(g_val) * sink_sig;
                o_row[d] = f32_to_bf16_rtne(y);
            }
        }
    }
}

extern "C" {

void launch_hip_qr_clip(
    const void* Q, const void* R, const void* QScale, const void* RScale, const void* Tau,
    void* QOut, void* ROut,
    int T, int H,
    int64_t stride_qt, int64_t stride_qh,
    int64_t stride_rt, int64_t stride_rh,
    float q_thresh, float r_thresh, float eps,
    int has_tau, int r_fp16, hipStream_t stream
) {
    if (T <= 0 || H <= 0) return;
    dim3 grid(T, (H + 7) / 8);
    dim3 block(128);
    if (has_tau) {
        if (r_fp16) {
            hipLaunchKernelGGL((hip_qr_clip_kernel<true, true>), grid, block, 0, stream,
                (const uint16_t*)Q, (const uint16_t*)R, (const uint16_t*)QScale, (const uint16_t*)RScale,
                (const float*)Tau, (uint16_t*)QOut, (uint16_t*)ROut,
                T, H, stride_qt, stride_qh, stride_rt, stride_rh, q_thresh, r_thresh, eps);
        } else {
            hipLaunchKernelGGL((hip_qr_clip_kernel<true, false>), grid, block, 0, stream,
                (const uint16_t*)Q, (const uint16_t*)R, (const uint16_t*)QScale, (const uint16_t*)RScale,
                (const float*)Tau, (uint16_t*)QOut, (uint16_t*)ROut,
                T, H, stride_qt, stride_qh, stride_rt, stride_rh, q_thresh, r_thresh, eps);
        }
    } else {
        if (r_fp16) {
            hipLaunchKernelGGL((hip_qr_clip_kernel<false, true>), grid, block, 0, stream,
                (const uint16_t*)Q, (const uint16_t*)R, (const uint16_t*)QScale, (const uint16_t*)RScale,
                (const float*)Tau, (uint16_t*)QOut, (uint16_t*)ROut,
                T, H, stride_qt, stride_qh, stride_rt, stride_rh, q_thresh, r_thresh, eps);
        } else {
            hipLaunchKernelGGL((hip_qr_clip_kernel<false, false>), grid, block, 0, stream,
                (const uint16_t*)Q, (const uint16_t*)R, (const uint16_t*)QScale, (const uint16_t*)RScale,
                (const float*)Tau, (uint16_t*)QOut, (uint16_t*)ROut,
                T, H, stride_qt, stride_qh, stride_rt, stride_rh, q_thresh, r_thresh, eps);
        }
    }
}

void launch_hip_gather_paged_kv_hnd(
    const void* KC, const void* VC, const void* BlockTableRow,
    void* KOut, void* VOut,
    int n_pages, int KVH, int page_size,
    int64_t skb, int64_t skp, int64_t skh,
    int64_t svb, int64_t svp, int64_t svh,
    int64_t sokh, int64_t sokn,
    int64_t sovh, int64_t sovn,
    hipStream_t stream
) {
    if (n_pages <= 0) return;
    dim3 grid(n_pages, KVH);
    dim3 block(128);
    hipLaunchKernelGGL(hip_gather_paged_kv_hnd_kernel, grid, block, 0, stream,
        (const uint16_t*)KC, (const uint16_t*)VC, (const int32_t*)BlockTableRow,
        (uint16_t*)KOut, (uint16_t*)VOut, n_pages, page_size,
        skb, skp, skh, svb, svp, svh, sokh, sokn, sovh, sovn);
}

void launch_hip_flash_rel_attn_prefill(
    const void* Q, const void* R, const void* RelProjT,
    const void* KC, const void* VC, const void* Gate, const void* Sink,
    void* Result, const void* BlockTable, const void* SeqLens, const void* CuSeqlensQ,
    float scale_log2,
    int64_t sqt, int64_t sqh, int64_t srt, int64_t srh,
    int64_t skb, int64_t skp, int64_t skh,
    int64_t svb, int64_t svp, int64_t svh,
    int64_t sgate, int64_t ssink, int64_t sres, int64_t sbt,
    int H, int GROUP, int PAGE, int EXTENT, int WINDOW,
    int max_seqlen_q, int num_reqs,
    int r_fp16, int round_bias_bf16, hipStream_t stream
) {
    if (max_seqlen_q <= 0 || num_reqs <= 0) return;
    dim3 grid(H, (max_seqlen_q + 63) / 64, num_reqs);
    dim3 block(128);
    int ps = (PAGE == 16) ? 4 : ((PAGE == 64) ? 6 : 0);
    int is_loc = (WINDOW >= 0) ? 1 : 0;
#define LAUNCH_P(LOC, RFP, RND, PS, MC) \
    hipLaunchKernelGGL((hip_flash_rel_attn_prefill_kernel<LOC, false, RFP, RND, PS, MC>), grid, block, 0, stream, \
        (const uint16_t*)Q, (const uint16_t*)R, (const uint16_t*)RelProjT, \
        (const uint16_t*)KC, (const uint16_t*)VC, (const uint16_t*)Gate, (const uint16_t*)Sink, \
        nullptr, nullptr, (uint16_t*)Result, \
        (const int32_t*)BlockTable, (const int32_t*)SeqLens, (const int32_t*)CuSeqlensQ, \
        scale_log2, 0, 0, sqt, sqh, srt, srh, skb, skp, skh, svb, svp, svh, \
        sgate, ssink, 0, 0, 0, 0, sres, sbt, H, GROUP, PAGE, EXTENT, WINDOW)
#define DISPATCH_P(LOC, RFP, RND, MC) do { \
    if (ps == 4)      LAUNCH_P(LOC, RFP, RND, 4, MC); \
    else if (ps == 6) LAUNCH_P(LOC, RFP, RND, 6, MC); \
    else              LAUNCH_P(LOC, RFP, RND, 0, 7); \
} while (0)
    if (!r_fp16 && round_bias_bf16) {
        if (is_loc) DISPATCH_P(true,  false, true,  7);
        else        DISPATCH_P(false, false, true,  7);
    } else if (!r_fp16 && !round_bias_bf16) {
        if (is_loc) DISPATCH_P(true,  false, false, 7);
        else        DISPATCH_P(false, false, false, 7);
    } else if (r_fp16 && round_bias_bf16) {
        if (is_loc) DISPATCH_P(true,  true,  true,  7);
        else        DISPATCH_P(false, true,  true,  7);
    } else {
        if (is_loc) DISPATCH_P(true,  true,  false, 7);
        else        DISPATCH_P(false, true,  false, 7);
    }
#undef DISPATCH_P
#undef LAUNCH_P
}

void launch_hip_flash_rel_attn_near_merge(
    const void* Q, const void* R, const void* RelProjT,
    const void* KC, const void* VC, const void* Gate, const void* Sink,
    const void* OutFar, const void* LseFar,
    void* Result, const void* BlockTable, const void* SeqLens, const void* CuSeqlensQ,
    float scale_log2, int far_len, int req_idx,
    int64_t sqt, int64_t sqh, int64_t srt, int64_t srh,
    int64_t skb, int64_t skp, int64_t skh,
    int64_t svb, int64_t svp, int64_t svh,
    int64_t sgate, int64_t ssink,
    int64_t sofh, int64_t sofq, int64_t slfh, int64_t slfq,
    int64_t sres, int64_t sbt,
    int H, int GROUP, int PAGE, int EXTENT,
    int ql, int r_fp16, int round_bias_bf16, hipStream_t stream
) {
    if (ql <= 0) return;
    dim3 grid(H, (ql + 63) / 64, 1);
    dim3 block(128);
    int ps = (PAGE == 16) ? 4 : ((PAGE == 64) ? 6 : 0);
#define LAUNCH_M(RFP, RND, PS) \
    hipLaunchKernelGGL((hip_flash_rel_attn_prefill_kernel<false, true, RFP, RND, PS, 7>), grid, block, 0, stream, \
        (const uint16_t*)Q, (const uint16_t*)R, (const uint16_t*)RelProjT, \
        (const uint16_t*)KC, (const uint16_t*)VC, (const uint16_t*)Gate, (const uint16_t*)Sink, \
        (const uint16_t*)OutFar, (const float*)LseFar, (uint16_t*)Result, \
        (const int32_t*)BlockTable, (const int32_t*)SeqLens, (const int32_t*)CuSeqlensQ, \
        scale_log2, far_len, req_idx, sqt, sqh, srt, srh, skb, skp, skh, svb, svp, svh, \
        sgate, ssink, sofh, sofq, slfh, slfq, sres, sbt, H, GROUP, PAGE, EXTENT, -1)
#define DISPATCH_M(RFP, RND) do { \
    if (ps == 4)      LAUNCH_M(RFP, RND, 4); \
    else if (ps == 6) LAUNCH_M(RFP, RND, 6); \
    else              LAUNCH_M(RFP, RND, 0); \
} while (0)
    if (!r_fp16 && round_bias_bf16) DISPATCH_M(false, true);
    else if (!r_fp16)                 DISPATCH_M(false, false);
    else if (round_bias_bf16)         DISPATCH_M(true,  true);
    else                              DISPATCH_M(true,  false);
#undef DISPATCH_M
#undef LAUNCH_M
}

}
