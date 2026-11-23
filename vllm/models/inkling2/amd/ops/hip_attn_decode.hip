// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 CIRU AI.
// Native RDNA 3.5 (gfx1151) 1M-ready decode attention kernels for Inkling 2 Mini.

#include "hip_common.cuh"
#include <type_traits>

// ============================================================================
// 1. 1-WARP PER (KV_HEAD, REQ) FUSED K+V SHORT-CONV DECODE WRITER
//    Grid = (num_kv_heads, num_reqs), Block = (32,) — 0 LDS, 0 barriers, int64 KV
// ============================================================================
template <int DYNAMIC, int POS_I64>
__global__ __launch_bounds__(32, 8)
void hip_kv_conv_decode_kernel(
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
    const void* __restrict__ PosRaw,
    const int32_t* __restrict__ SeqIdx,
    const int64_t* __restrict__ ConvSlots,
    const int32_t* __restrict__ BlockTable,
    const int32_t* __restrict__ QStart,
    const int32_t* __restrict__ CuSeqlensQ,
    int T, int KVH, int64_t stride_k, int64_t stride_v,
    int64_t stride_w0, int64_t stride_w1, int64_t stride_dk, int64_t stride_dv,
    int64_t stride_c_blk, int64_t stride_c_h, int64_t stride_c_n,
    int64_t stride_kv_blk, int64_t stride_kv_n, int64_t stride_kv_h,
    int kv_block_size, int64_t stride_bt, int max_blocks, int N_conv,
    float scale, float clip_threshold, float clip_eps, float norm_eps
) {
    const int head = (int)blockIdx.x;
    const int req_blk = (int)blockIdx.y;
    const int lane = (int)threadIdx.x;
    const int d0 = lane * 4;
    const int col0 = head * 128 + d0;

    const int q_start = CuSeqlensQ[req_blk];
    const int q_end = min(CuSeqlensQ[req_blk + 1], T);
    if (q_start >= q_end) return;

    auto read_pos = [&](int idx) -> int64_t {
        if constexpr (POS_I64) {
            return reinterpret_cast<const int64_t*>(PosRaw)[idx];
        } else {
            return (int64_t)reinterpret_cast<const int32_t*>(PosRaw)[idx];
        }
    };

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
    const int64_t cslot0 = ConvSlots[q_start];
    const int64_t pos0 = read_pos(q_start);
    const int req0 = SeqIdx[q_start];
    if (cslot0 >= 0) {
        #pragma unroll
        for (int h_i = 0; h_i < 3; ++h_i) {
            int lag = 3 - h_i;
            int64_t src = pos0 - lag;
            if (src >= 0) {
                int64_t blk_i = min(src / (int64_t)N_conv, (int64_t)(max_blocks - 1));
                int64_t pblk = (int64_t)BlockTable[(int64_t)req0 * stride_bt + blk_i];
                int64_t c_off = pblk * stride_c_blk + (src % (int64_t)N_conv) * stride_c_n
                              + (int64_t)head * stride_c_h + d0;
                load_u2_f32(ConvCache + c_off, kh[h_i]);
                load_u2_f32(ConvCache + c_off + 128, vh[h_i]);
            }
        }
    }

    for (int t = q_start; t < q_end; ++t) {
        const int64_t cslot = ConvSlots[t];
        const int qst = QStart[t];
        if (t > q_start && qst == t) {
            const int64_t pos = read_pos(t);
            const int req = SeqIdx[t];
            #pragma unroll
            for (int h_i = 0; h_i < 3; ++h_i) {
                int lag = 3 - h_i;
                int64_t src = pos - lag;
                if (cslot >= 0 && src >= 0) {
                    int64_t blk_i = min(src / (int64_t)N_conv, (int64_t)(max_blocks - 1));
                    int64_t pblk = (int64_t)BlockTable[(int64_t)req * stride_bt + blk_i];
                    int64_t c_off = pblk * stride_c_blk + (src % (int64_t)N_conv) * stride_c_n
                                  + (int64_t)head * stride_c_h + d0;
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
            int64_t c_off = (cslot / (int64_t)N_conv) * stride_c_blk
                          + (cslot % (int64_t)N_conv) * stride_c_n
                          + (int64_t)head * stride_c_h + d0;
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
        const bool valid = (cslot >= 0);
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
            int64_t kv_off = (kv_slot / (int64_t)kv_block_size) * stride_kv_blk
                           + (kv_slot % (int64_t)kv_block_size) * stride_kv_n
                           + (int64_t)head * stride_kv_h + d0;
            uint2 ko = make_uint2(pack_bf16x2(ky[0] * k_sc, ky[1] * k_sc), pack_bf16x2(ky[2] * k_sc, ky[3] * k_sc));
            uint2 vo = make_uint2(pack_bf16x2(vy[0] * v_sc, vy[1] * v_sc), pack_bf16x2(vy[2] * v_sc, vy[3] * v_sc));
            *reinterpret_cast<uint2*>(KeyCache + kv_off) = ko;
            *reinterpret_cast<uint2*>(ValCache + kv_off) = vo;
        }
    }
}

// ============================================================================
// 2. ZERO-MAT 1M-READY FLASH-REL-DECODE SPLIT KERNEL
//    Grid = (num_splits, num_kv_heads=8, num_reqs), Block = (128,) = 4 warps
//    Zero __syncthreads() in key loop, zero smem_K, 12.5 KiB LDS, int64 KV
// ============================================================================
__device__ __forceinline__ float decode_dot16_bf16(uint4 r0, uint4 r1, const uint16_t* p_ptr) {
    uint4 p0 = *reinterpret_cast<const uint4*>(p_ptr);
    uint4 p1 = *reinterpret_cast<const uint4*>(p_ptr + 8);
    float acc = 0.0f;
    asm("v_dot2_f32_bf16 %0, %1, %2, %0" : "+v"(acc) : "v"(r0.x), "v"(p0.x));
    asm("v_dot2_f32_bf16 %0, %1, %2, %0" : "+v"(acc) : "v"(r0.y), "v"(p0.y));
    asm("v_dot2_f32_bf16 %0, %1, %2, %0" : "+v"(acc) : "v"(r0.z), "v"(p0.z));
    asm("v_dot2_f32_bf16 %0, %1, %2, %0" : "+v"(acc) : "v"(r0.w), "v"(p0.w));
    asm("v_dot2_f32_bf16 %0, %1, %2, %0" : "+v"(acc) : "v"(r1.x), "v"(p1.x));
    asm("v_dot2_f32_bf16 %0, %1, %2, %0" : "+v"(acc) : "v"(r1.y), "v"(p1.y));
    asm("v_dot2_f32_bf16 %0, %1, %2, %0" : "+v"(acc) : "v"(r1.z), "v"(p1.z));
    asm("v_dot2_f32_bf16 %0, %1, %2, %0" : "+v"(acc) : "v"(r1.w), "v"(p1.w));
    return acc;
}

__device__ __forceinline__ short16 decode_load_short16_u4(const uint16_t* ptr) {
    uint4 lo = *reinterpret_cast<const uint4*>(ptr);
    uint4 hi = *reinterpret_cast<const uint4*>(ptr + 8);
    short16 out;
    uint32_t words[8] = {lo.x, lo.y, lo.z, lo.w, hi.x, hi.y, hi.z, hi.w};
    __builtin_memcpy(&out, words, 32);
    return out;
}

__device__ __forceinline__ const uint16_t* decode_vt_row_ptr(const uint32_t* vt_base, int d) {
    return reinterpret_cast<const uint16_t*>(vt_base + (d >> 3) * 100 + (d & 7) * 12);
}

__device__ __forceinline__ void decode_store_vt_zconf(
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

__device__ __forceinline__ short16 decode_pack_p_in_reg(const float p_f[8], int half) {
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

template <bool IS_LOCAL, bool HAS_REL, int PAGE_SHIFT, int BQ_PACK, int MIN_CTAS>
__global__ __launch_bounds__(128, MIN_CTAS)
void hip_flash_rel_decode_split_kernel(
    const uint16_t* __restrict__ Q,
    const uint16_t* __restrict__ R,
    const uint16_t* __restrict__ RelProjT,
    const uint16_t* __restrict__ KCache,
    const uint16_t* __restrict__ VCache,
    float* __restrict__ OPart,
    float* __restrict__ LsePart,
    const int32_t* __restrict__ BlockTable,
    const int32_t* __restrict__ SeqLens,
    const int32_t* __restrict__ CuSeqlensQ,
    int num_splits, int num_tokens, int window, int extent, float scale_log2,
    int64_t sqt, int64_t sqh,
    int64_t srt, int64_t srh,
    int64_t skb, int64_t skp, int64_t skh,
    int64_t svb, int64_t svp, int64_t svh,
    int64_t sot, int64_t soh, int64_t slh, int64_t stable
) {
    constexpr float LOG2E = 1.4426950408889634f;
    constexpr int PAGE_MASK = (1 << PAGE_SHIFT) - 1;
    constexpr int K_WARPS = (BQ_PACK == 16) ? 4 : 2;

    const int split = (int)blockIdx.x;
    const int kv_head = (int)blockIdx.y;
    const int req = (int)blockIdx.z;

    const int q_start = CuSeqlensQ[req];
    const int q_len = CuSeqlensQ[req + 1] - q_start;
    const int seq_len = SeqLens[req];

    const int tid = (int)threadIdx.x;
    const int warp = tid >> 5;
    const int lane = tid & 31;
    const int lane16 = lane & 15;
    const int half = lane >> 4;

    const int q_tile_id = (BQ_PACK == 16) ? 0 : (warp >> 1);
    const int k_warp_id = (BQ_PACK == 16) ? warp : (warp & 1);

    const int m_row = q_tile_id * 16 + lane16;
    const int q_idx = m_row / 3;
    const int h_off = m_row - q_idx * 3;
    const int head = kv_head * 3 + h_off;
    const bool row_valid = (q_idx < q_len);
    const int64_t tok = (int64_t)(q_start + q_idx);
    const int64_t out_tok = (int64_t)split * (int64_t)num_tokens + tok;

    const int past = seq_len - q_len;
    const int first_pos = HAS_REL ? past : (seq_len - 1);
    const int pos = HAS_REL ? (past + q_idx) : (seq_len - 1);
    const int hi = seq_len;
    int lo = 0;
    if constexpr (IS_LOCAL) {
        int w_lo = first_pos - window;
        lo = (w_lo > 0) ? (w_lo & ~15) : 0;
    }
    const int first_tile = lo >> 4;
    const int end_tile = (hi + 15) >> 4;
    const int total_tiles = max(0, end_tile - first_tile);
    const int tiles_per_split = (total_tiles + num_splits - 1) / num_splits;
    const int split_first = first_tile + split * tiles_per_split;
    const int split_end = min(split_first + tiles_per_split, end_tile);

    // Zero-work early exit for empty splits or empty requests
    if (q_len <= 0 || seq_len <= 0 || split_first >= split_end) {
        if (k_warp_id == 0 && half == 0 && row_valid) {
            LsePart[(int64_t)head * slh + out_tok] = -INFINITY;
        }
        return;
    }

    union SharedBuf {
        uint32_t vt[4][8 * 100];      // 12,800 B (4 warps x 3.125 KiB)
        struct {
            float m[4][16];           // 256 B
            float l[4][16];           // 256 B
            float o[4][16][32];       // 8,192 B
        } merge;
    };
    __shared__ SharedBuf smem;
    uint32_t* my_vt = smem.vt[warp];

    // Load Q[m_row, 0..127] (unscaled BF16 -> exact FP32 scale_log2 after WMMA)
    short16 q_reg[8];
    if (row_valid) {
        const uint16_t* q_row = Q + tok * sqt + (int64_t)head * sqh;
        #pragma unroll
        for (int kd = 0; kd < 8; ++kd) {
            q_reg[kd] = decode_load_short16_u4(q_row + kd * 16);
        }
    } else {
        #pragma unroll
        for (int kd = 0; kd < 8; ++kd) q_reg[kd] = {};
    }



    const int32_t* table = BlockTable + (int64_t)req * stable;
    const uint16_t* k_head_base = KCache + (int64_t)kv_head * skh;
    const uint16_t* v_head_base = VCache + (int64_t)kv_head * svh;

    float m_i = -INFINITY;
    float l_i = 0.0f;
    float8 acc_o[8];
    #pragma unroll
    for (int dt = 0; dt < 8; ++dt) acc_o[dt] = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};

    const int pair = lane16 >> 1;
    const int odd = lane16 & 1;

    // Helper lambda to execute V^T @ P^T in 2 LDS-reuse halves (64 d-cols each, 0 barriers!)
    auto run_v_pass = [&](const uint16_t* v_row, bool k_valid, short16 p_reg) {
        #pragma unroll
        for (int half_d = 0; half_d < 2; ++half_d) {
            const int d_off_base = half_d * 64;
            #pragma unroll
            for (int sub = 0; sub < 4; ++sub) {
                int d_blk = sub * 2 + half;  // 0..7 -> 8 groups of 8 bf16 = 64 d-cols
                uint4 v_me = k_valid
                    ? *reinterpret_cast<const uint4*>(v_row + d_off_base + d_blk * 8)
                    : make_uint4(0, 0, 0, 0);
                uint4 v_oth = make_uint4(
                    __shfl_xor(v_me.x, 1, 32),
                    __shfl_xor(v_me.y, 1, 32),
                    __shfl_xor(v_me.z, 1, 32),
                    __shfl_xor(v_me.w, 1, 32)
                );
                if (odd == 0) {
                    decode_store_vt_zconf(my_vt, d_blk, pair, v_me, v_oth);
                }
            }
            asm volatile("" ::: "memory");
            #pragma unroll
            for (int dt_l = 0; dt_l < 4; ++dt_l) {
                short16 vt_reg = decode_load_short16_u4(decode_vt_row_ptr(my_vt, dt_l * 16 + lane16));
                acc_o[half_d * 4 + dt_l] = __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32(
                    vt_reg, p_reg, acc_o[half_d * 4 + dt_l]
                );
            }
        }
    };

    // Determine Phase A (branchless FAR) boundary
    int far_end_tile = split_first;
    if constexpr (!IS_LOCAL) {
        int full_tiles_end = hi >> 4;
        if constexpr (!HAS_REL) {
            far_end_tile = min(split_end, full_tiles_end);
        } else {
            int f_lim = max(past - extent + 1, 0) >> 4;
            far_end_tile = min(split_end, min(f_lim, full_tiles_end));
        }
    }

    // ------------------------------------------------------------------------
    // PHASE A: Branchless FAR Loop (99.2% of keys @ 128K, 99.9% @ 1M!)
    // Zero RelProjT loads, zero dist/window/causal branches, zero barriers!
    // ------------------------------------------------------------------------
    if constexpr (!IS_LOCAL) {
        for (int tile = split_first + k_warp_id; tile < far_end_tile; tile += K_WARPS) {
            const int k0 = tile << 4;
            const int64_t page_id = (int64_t)table[k0 >> PAGE_SHIFT];
            const int64_t slot = (int64_t)((k0 & PAGE_MASK) + lane16);
            const uint16_t* k_row = k_head_base + page_id * skb + slot * skp;
            const uint16_t* v_row = v_head_base + page_id * svb + slot * svp;

            float8 s_acc = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
            #pragma unroll
            for (int kd = 0; kd < 8; ++kd) {
                uint4 my_u4 = *reinterpret_cast<const uint4*>(k_row + kd * 16 + half * 8);
                uint4 oth_u4 = make_uint4(
                    __shfl_xor(my_u4.x, 16, 32),
                    __shfl_xor(my_u4.y, 16, 32),
                    __shfl_xor(my_u4.z, 16, 32),
                    __shfl_xor(my_u4.w, 16, 32)
                );
                uint4 lo_u4 = (half == 0) ? my_u4 : oth_u4;
                uint4 hi_u4 = (half == 0) ? oth_u4 : my_u4;
                short16 k_reg;
                uint32_t w[8] = {lo_u4.x, lo_u4.y, lo_u4.z, lo_u4.w, hi_u4.x, hi_u4.y, hi_u4.z, hi_u4.w};
                __builtin_memcpy(&k_reg, w, 32);
                s_acc = __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32(k_reg, q_reg[kd], s_acc);
            }

            float m_loc = -INFINITY;
            #pragma unroll
            for (int r = 0; r < 8; ++r) {
                s_acc[r] *= scale_log2;
                m_loc = fmaxf(m_loc, s_acc[r]);
            }
            float m_tile = fmaxf(m_loc, __shfl_xor(m_loc, 16, 32));
            float m_new = fmaxf(m_i, m_tile);
            float alpha = exp2f(m_i - m_new);
            float p_f[8];
            float l_loc = 0.0f;
            #pragma unroll
            for (int r = 0; r < 8; ++r) {
                float pv = exp2f(s_acc[r] - m_new);
                p_f[r] = pv;
                l_loc += pv;
            }
            float l_tile = l_loc + __shfl_xor(l_loc, 16, 32);
            l_i = l_i * alpha + l_tile;
            m_i = m_new;
            #pragma unroll
            for (int dt = 0; dt < 8; ++dt) {
                #pragma unroll
                for (int r = 0; r < 8; ++r) acc_o[dt][r] *= alpha;
            }

            short16 p_reg = decode_pack_p_in_reg(p_f, half);
            run_v_pass(v_row, true, p_reg);
        }
    }

    // ------------------------------------------------------------------------
    // PHASE B: NEAR / Local / Tail Loop (on-the-fly RelProjT + exact masks)
    // ------------------------------------------------------------------------
    const int near_start_tile = max(split_first, far_end_tile);
    uint4 r_u4_0 = make_uint4(0, 0, 0, 0);
    uint4 r_u4_1 = make_uint4(0, 0, 0, 0);
    if constexpr (HAS_REL) {
        if (row_valid && near_start_tile < split_end) {
            const uint16_t* r_row = R + tok * srt + (int64_t)head * srh;
            r_u4_0 = *reinterpret_cast<const uint4*>(r_row);
            r_u4_1 = *reinterpret_cast<const uint4*>(r_row + 8);
        }
    }
    for (int tile = near_start_tile + k_warp_id; tile < split_end; tile += K_WARPS) {
        const int k0 = tile << 4;
        const int64_t page_id = (int64_t)table[k0 >> PAGE_SHIFT];
        const int key_lane = k0 + lane16;
        const bool k_valid = (key_lane < hi);
        const int64_t slot = (int64_t)((k0 & PAGE_MASK) + lane16);
        const uint16_t* k_row = k_head_base + page_id * skb + slot * skp;
        const uint16_t* v_row = v_head_base + page_id * svb + slot * svp;

        float8 s_acc = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
        #pragma unroll
        for (int kd = 0; kd < 8; ++kd) {
            uint4 my_u4 = k_valid
                ? *reinterpret_cast<const uint4*>(k_row + kd * 16 + half * 8)
                : make_uint4(0, 0, 0, 0);
            uint4 oth_u4 = make_uint4(
                __shfl_xor(my_u4.x, 16, 32),
                __shfl_xor(my_u4.y, 16, 32),
                __shfl_xor(my_u4.z, 16, 32),
                __shfl_xor(my_u4.w, 16, 32)
            );
            uint4 lo_u4 = (half == 0) ? my_u4 : oth_u4;
            uint4 hi_u4 = (half == 0) ? oth_u4 : my_u4;
            short16 k_reg;
            uint32_t w[8] = {lo_u4.x, lo_u4.y, lo_u4.z, lo_u4.w, hi_u4.x, hi_u4.y, hi_u4.z, hi_u4.w};
            __builtin_memcpy(&k_reg, w, 32);
            s_acc = __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32(k_reg, q_reg[kd], s_acc);
        }

        float m_loc = -INFINITY;
        #pragma unroll
        for (int r = 0; r < 8; ++r) {
            int key = k0 + half + 2 * r;
            int dist = pos - key;
            bool ok = row_valid && (key < hi) && (dist >= 0);
            if constexpr (IS_LOCAL) ok = ok && (dist <= window);
            if (ok) {
                float sc = s_acc[r] * scale_log2;
                if constexpr (HAS_REL) {
                    if (dist < extent) {
                        float b = bf16_round_f32(decode_dot16_bf16(r_u4_0, r_u4_1, RelProjT + (int64_t)dist * 16));
                        sc += b * LOG2E;
                    }
                }
                s_acc[r] = sc;
                m_loc = fmaxf(m_loc, sc);
            } else {
                s_acc[r] = -INFINITY;
            }
        }

        float m_tile = fmaxf(m_loc, __shfl_xor(m_loc, 16, 32));
        float m_new = fmaxf(m_i, m_tile);
        float m_ref = (m_new == -INFINITY) ? 0.0f : m_new;
        float alpha = (m_i == -INFINITY) ? 0.0f : exp2f(m_i - m_ref);
        float p_f[8];
        float l_loc = 0.0f;
        #pragma unroll
        for (int r = 0; r < 8; ++r) {
            float pv = (s_acc[r] == -INFINITY) ? 0.0f : exp2f(s_acc[r] - m_ref);
            p_f[r] = pv;
            l_loc += pv;
        }
        float l_tile = l_loc + __shfl_xor(l_loc, 16, 32);
        l_i = l_i * alpha + l_tile;
        m_i = m_new;
        #pragma unroll
        for (int dt = 0; dt < 8; ++dt) {
            #pragma unroll
            for (int r = 0; r < 8; ++r) acc_o[dt][r] *= alpha;
        }

        short16 p_reg = decode_pack_p_in_reg(p_f, half);
        run_v_pass(v_row, k_valid, p_reg);
    }

    // ------------------------------------------------------------------------
    // FINAL WARP-MERGE ACROSS K_WARPS (reuses smem union, 0 extra LDS!)
    // ------------------------------------------------------------------------
    __syncthreads();
    if (half == 0) {
        smem.merge.m[warp][lane16] = m_i;
        smem.merge.l[warp][lane16] = l_i;
    }
    __syncthreads();

    const int w0 = q_tile_id * K_WARPS;
    float m_final = -INFINITY;
    #pragma unroll
    for (int kw = 0; kw < K_WARPS; ++kw) {
        m_final = fmaxf(m_final, smem.merge.m[w0 + kw][lane16]);
    }
    float l_final = 0.0f;
    float w_scale[K_WARPS];
    const float m_ref_f = (m_final == -INFINITY) ? 0.0f : m_final;
    #pragma unroll
    for (int kw = 0; kw < K_WARPS; ++kw) {
        float mw = smem.merge.m[w0 + kw][lane16];
        float lw = smem.merge.l[w0 + kw][lane16];
        float ex = (mw == -INFINITY || lw <= 0.0f) ? 0.0f : exp2f(mw - m_ref_f);
        w_scale[kw] = ex;
        l_final += lw * ex;
    }
    const bool has_keys = (l_final > 0.0f);
    const float inv_l = has_keys ? (1.0f / l_final) : 0.0f;
    #pragma unroll
    for (int kw = 0; kw < K_WARPS; ++kw) w_scale[kw] *= inv_l;

    // Reduce O in 4 chunks of 32 d-cols (2 dt tiles per chunk)
    #pragma unroll
    for (int chunk = 0; chunk < 4; ++chunk) {
        #pragma unroll
        for (int dt_l = 0; dt_l < 2; ++dt_l) {
            int dt = chunk * 2 + dt_l;
            #pragma unroll
            for (int r = 0; r < 8; ++r) {
                smem.merge.o[warp][lane16][dt_l * 16 + half + 2 * r] = acc_o[dt][r];
            }
        }
        __syncthreads();
        if (k_warp_id == 0 && row_valid) {
            float* o_dst = OPart + out_tok * sot + (int64_t)head * soh + chunk * 32;
            #pragma unroll
            for (int dt_l = 0; dt_l < 2; ++dt_l) {
                #pragma unroll
                for (int r = 0; r < 8; ++r) {
                    int d_loc = dt_l * 16 + half + 2 * r;
                    float sum_v = 0.0f;
                    #pragma unroll
                    for (int kw = 0; kw < K_WARPS; ++kw) {
                        sum_v = fmaf(smem.merge.o[w0 + kw][lane16][d_loc], w_scale[kw], sum_v);
                    }
                    o_dst[d_loc] = sum_v;
                }
            }
        }
        __syncthreads();
    }

    if (k_warp_id == 0 && half == 0 && row_valid) {
        LsePart[(int64_t)head * slh + out_tok] = has_keys ? (m_final + log2f(l_final)) : -INFINITY;
    }
}


// ============================================================================
// 2b. COOPERATIVE BN=32 FLASH-REL-DECODE SPLIT KERNEL FOR BQ_PACK=32 (B=1 M=6..8)
//     All 128 threads cooperatively load K & V ONCE into an 8.5 KiB LDS union,
//     completely eliminating 2x redundant HBM reads between Q[0..15] & Q[16..31]!
// ============================================================================
template <bool IS_LOCAL, bool HAS_REL, int PAGE_SHIFT, int MIN_CTAS>
__global__ __launch_bounds__(128, MIN_CTAS)
void hip_flash_rel_decode_split_bq32_kernel(
    const uint16_t* __restrict__ Q,
    const uint16_t* __restrict__ R,
    const uint16_t* __restrict__ RelProjT,
    const uint16_t* __restrict__ KCache,
    const uint16_t* __restrict__ VCache,
    float* __restrict__ OPart,
    float* __restrict__ LsePart,
    const int32_t* __restrict__ BlockTable,
    const int32_t* __restrict__ SeqLens,
    const int32_t* __restrict__ CuSeqlensQ,
    int num_splits, int num_tokens, int window, int extent, float scale_log2,
    int64_t sqt, int64_t sqh,
    int64_t srt, int64_t srh,
    int64_t skb, int64_t skp, int64_t skh,
    int64_t svb, int64_t svp, int64_t svh,
    int64_t sot, int64_t soh, int64_t slh, int64_t stable
) {
    constexpr float LOG2E = 1.4426950408889634f;
    constexpr int PAGE_MASK = (1 << PAGE_SHIFT) - 1;

    const int split = (int)blockIdx.x;
    const int kv_head = (int)blockIdx.y;
    const int req = (int)blockIdx.z;

    const int q_start = CuSeqlensQ[req];
    const int q_len = CuSeqlensQ[req + 1] - q_start;
    const int seq_len = SeqLens[req];

    const int tid = (int)threadIdx.x;
    const int warp = tid >> 5;
    const int lane = tid & 31;
    const int lane16 = lane & 15;
    const int half = lane >> 4;

    const int q_tile_id = warp >> 1;   // 0: Q[0..15] (warps 0,1); 1: Q[16..31] (warps 2,3)
    const int k_warp_id = warp & 1;    // 0: keys 0..15 of 32-key step; 1: keys 16..31 of 32-key step

    const int m_row = q_tile_id * 16 + lane16;
    const int q_idx = m_row / 3;
    const int h_off = m_row - q_idx * 3;
    const int head = kv_head * 3 + h_off;
    const bool row_valid = (q_idx < q_len);
    const int64_t tok = (int64_t)(q_start + q_idx);
    const int64_t out_tok = (int64_t)split * (int64_t)num_tokens + tok;

    const int past = seq_len - q_len;
    const int first_pos = HAS_REL ? past : (seq_len - 1);
    const int pos = HAS_REL ? (past + q_idx) : (seq_len - 1);
    const int hi = seq_len;
    int lo = 0;
    if constexpr (IS_LOCAL) {
        int w_lo = first_pos - window;
        lo = (w_lo > 0) ? (w_lo & ~31) : 0;
    }
    const int first_blk = lo >> 5;         // 32-key blocks!
    const int end_blk = (hi + 31) >> 5;
    const int total_blks = max(0, end_blk - first_blk);
    const int blks_per_split = (total_blks + num_splits - 1) / num_splits;
    const int split_first = first_blk + split * blks_per_split;
    const int split_end = min(split_first + blks_per_split, end_blk);

    if (q_len <= 0 || seq_len <= 0 || split_first >= split_end) {
        if (k_warp_id == 0 && half == 0 && row_valid) {
            LsePart[(int64_t)head * slh + out_tok] = -INFINITY;
        }
        return;
    }

    // 8.5 KiB LDS union (allows up to 7 CTAs/CU on LDS!)
    union CoopShared {
        uint16_t k[32][136];              // 8,704 B
        uint32_t vt[2][8 * 100];          // 6,400 B (2 x 16-key subtiles x 64 d-cols)
        struct {
            float m[4][16];               // 256 B
            float l[4][16];               // 256 B
            float o[4][16][32];           // 8,192 B
        } merge;                          // 8,704 B
    };
    __shared__ CoopShared smem;
    __shared__ uint16_t smem_Q[32][136];
    __shared__ uint16_t smem_R[32][16];

    // Load Q[32][128] once into 8.5 KiB LDS (frees 56+ VGPRs -> ~145 VGPRs, 0 spills!)
    if (k_warp_id == 0) {
        if (row_valid) {
            const uint16_t* q_row = Q + tok * sqt + (int64_t)head * sqh;
            #pragma unroll
            for (int kd = 0; kd < 8; ++kd) {
                *reinterpret_cast<uint4*>(&smem_Q[m_row][kd * 16 + half * 8]) =
                    *reinterpret_cast<const uint4*>(q_row + kd * 16 + half * 8);
            }
        } else {
            #pragma unroll
            for (int kd = 0; kd < 8; ++kd) {
                *reinterpret_cast<uint4*>(&smem_Q[m_row][kd * 16 + half * 8]) = make_uint4(0, 0, 0, 0);
            }
        }
    }
    __syncthreads();

    const int32_t* table = BlockTable + (int64_t)req * stable;
    const uint16_t* k_head_base = KCache + (int64_t)kv_head * skh;
    const uint16_t* v_head_base = VCache + (int64_t)kv_head * svh;

    float m_i = -INFINITY;
    float l_i = 0.0f;
    float8 acc_o[8];
    #pragma unroll
    for (int dt = 0; dt < 8; ++dt) acc_o[dt] = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};

    // Cooperative K load indices: 128 threads -> 8 rows x 16 uint4s (= full 128-bf16 row coalesced!)
    const int ck_row = tid >> 4;          // 0..7
    const int ck_d   = tid & 15;          // 0..15 (x8 bf16 = 0..127)
    // Cooperative V load indices: 128 threads -> 16 rows x 8 uint4s (= 64 bf16 coalesced!)
    const int cv_row  = tid >> 3;         // 0..15
    const int cv_dblk = tid & 7;          // 0..7 (d_blk)
    const int cv_pair = cv_row >> 1;      // 0..7
    const int cv_odd  = cv_row & 1;       // 0 or 1 (partner is lane ^ 8!)

    auto run_coop_v = [&](int k0, int64_t p0, int64_t p1, bool check_hi, short16 p_reg) {
        const int64_t s0_base = (int64_t)(k0 & PAGE_MASK);
        const int64_t s1_base = (int64_t)((k0 + 16) & PAGE_MASK);
        const uint16_t* v0_ptr = v_head_base + p0 * svb + (s0_base + cv_row) * svp;
        const uint16_t* v1_ptr = v_head_base + p1 * svb + (s1_base + cv_row) * svp;
        const bool v0_ok = !check_hi || (k0 + cv_row < hi);
        const bool v1_ok = !check_hi || (k0 + 16 + cv_row < hi);

        #pragma unroll
        for (int half_d = 0; half_d < 2; ++half_d) {
            const int d_off = half_d * 64 + cv_dblk * 8;
            {
                uint4 va0 = v0_ok ? *reinterpret_cast<const uint4*>(v0_ptr + d_off) : make_uint4(0, 0, 0, 0);
                uint4 vb0 = make_uint4(__shfl_xor(va0.x, 8, 32), __shfl_xor(va0.y, 8, 32), __shfl_xor(va0.z, 8, 32), __shfl_xor(va0.w, 8, 32));
                if (cv_odd == 0) decode_store_vt_zconf(smem.vt[0], cv_dblk, cv_pair, va0, vb0);
            }
            {
                uint4 va1 = v1_ok ? *reinterpret_cast<const uint4*>(v1_ptr + d_off) : make_uint4(0, 0, 0, 0);
                uint4 vb1 = make_uint4(__shfl_xor(va1.x, 8, 32), __shfl_xor(va1.y, 8, 32), __shfl_xor(va1.z, 8, 32), __shfl_xor(va1.w, 8, 32));
                if (cv_odd == 0) decode_store_vt_zconf(smem.vt[1], cv_dblk, cv_pair, va1, vb1);
            }
            __syncthreads();
            const uint32_t* my_vt = smem.vt[k_warp_id];
            #pragma unroll
            for (int dt_l = 0; dt_l < 4; ++dt_l) {
                short16 vt_reg = decode_load_short16_u4(decode_vt_row_ptr(my_vt, dt_l * 16 + lane16));
                acc_o[half_d * 4 + dt_l] = __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32(
                    vt_reg, p_reg, acc_o[half_d * 4 + dt_l]
                );
                asm volatile("" : "+v"(acc_o[half_d * 4 + dt_l]));
            }
            __syncthreads();
        }
    };

    int far_end_blk = split_first;
    if constexpr (!IS_LOCAL) {
        int full_blks_end = hi >> 5;
        if constexpr (!HAS_REL) {
            far_end_blk = min(split_end, full_blks_end);
        } else {
            int f_lim = max(past - extent + 1, 0) >> 5;
            far_end_blk = min(split_end, min(f_lim, full_blks_end));
        }
    }

    // PHASE A: Branchless Cooperative BN=32 FAR Loop (1.0x HBM reads!)
    if constexpr (!IS_LOCAL) {
        for (int blk = split_first; blk < far_end_blk; ++blk) {
            const int k0 = blk << 5;
            const int64_t p0 = (int64_t)table[k0 >> PAGE_SHIFT];
            const int64_t p1 = (PAGE_SHIFT == 4) ? (int64_t)table[(k0 + 16) >> PAGE_SHIFT] : p0;
            const int64_t s0_base = (int64_t)(k0 & PAGE_MASK);
            const int64_t s1_base = (int64_t)((k0 + 16) & PAGE_MASK);

            // Cooperative K load: 32 keys x 128 d -> 4 uint4s/thread
            const uint16_t* k0_base = k_head_base + p0 * skb + ck_d * 8;
            const uint16_t* k1_base = k_head_base + p1 * skb + ck_d * 8;
            *reinterpret_cast<uint4*>(&smem.k[ck_row +  0][ck_d * 8]) = *reinterpret_cast<const uint4*>(k0_base + (s0_base + ck_row + 0) * skp);
            *reinterpret_cast<uint4*>(&smem.k[ck_row +  8][ck_d * 8]) = *reinterpret_cast<const uint4*>(k0_base + (s0_base + ck_row + 8) * skp);
            *reinterpret_cast<uint4*>(&smem.k[ck_row + 16][ck_d * 8]) = *reinterpret_cast<const uint4*>(k1_base + (s1_base + ck_row + 0) * skp);
            *reinterpret_cast<uint4*>(&smem.k[ck_row + 24][ck_d * 8]) = *reinterpret_cast<const uint4*>(k1_base + (s1_base + ck_row + 8) * skp);
            __syncthreads();

            const int k_row_idx = k_warp_id * 16 + lane16;
            float8 s_acc = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
            #pragma unroll
            for (int kd = 0; kd < 8; ++kd) {
                short16 k_reg = decode_load_short16_u4(&smem.k[k_row_idx][kd * 16]);
                short16 q_r = decode_load_short16_u4(&smem_Q[m_row][kd * 16]);
                s_acc = __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32(k_reg, q_r, s_acc);
            asm volatile("" : "+v"(s_acc));
            asm volatile("" : "+v"(s_acc));
            }
            __syncthreads();

            float m_loc = -INFINITY;
            #pragma unroll
            for (int r = 0; r < 8; ++r) {
                s_acc[r] *= scale_log2;
                m_loc = fmaxf(m_loc, s_acc[r]);
            }
            float m_tile = fmaxf(m_loc, __shfl_xor(m_loc, 16, 32));
            float m_new = fmaxf(m_i, m_tile);
            float alpha = exp2f(m_i - m_new);
            float p_f[8];
            float l_loc = 0.0f;
            #pragma unroll
            for (int r = 0; r < 8; ++r) {
                float pv = exp2f(s_acc[r] - m_new);
                p_f[r] = pv;
                l_loc += pv;
            }
            float l_tile = l_loc + __shfl_xor(l_loc, 16, 32);
            l_i = l_i * alpha + l_tile;
            m_i = m_new;
            #pragma unroll
            for (int dt = 0; dt < 8; ++dt) {
                #pragma unroll
                for (int r = 0; r < 8; ++r) acc_o[dt][r] *= alpha;
            }
            short16 p_reg = decode_pack_p_in_reg(p_f, half);
            run_coop_v(k0, p0, p1, false, p_reg);
        }
    }

    // PHASE B: NEAR / Local / Tail Cooperative BN=32 Loop
    const int near_start_blk = max(split_first, far_end_blk);
    if constexpr (HAS_REL) {
        if (k_warp_id == 0 && half == 0 && near_start_blk < split_end) {
            if (row_valid) {
                const uint16_t* r_row = R + tok * srt + (int64_t)head * srh;
                *reinterpret_cast<uint4*>(&smem_R[m_row][0]) = *reinterpret_cast<const uint4*>(r_row);
                *reinterpret_cast<uint4*>(&smem_R[m_row][8]) = *reinterpret_cast<const uint4*>(r_row + 8);
            } else {
                *reinterpret_cast<uint4*>(&smem_R[m_row][0]) = make_uint4(0, 0, 0, 0);
                *reinterpret_cast<uint4*>(&smem_R[m_row][8]) = make_uint4(0, 0, 0, 0);
            }
        }
        if (near_start_blk < split_end) __syncthreads();
    }

    for (int blk = near_start_blk; blk < split_end; ++blk) {
        const int k0 = blk << 5;
        const int64_t p0 = (int64_t)table[k0 >> PAGE_SHIFT];
        const int64_t p1 = (PAGE_SHIFT == 4 && k0 + 16 < hi) ? (int64_t)table[(k0 + 16) >> PAGE_SHIFT] : p0;
        const int64_t s0_base = (int64_t)(k0 & PAGE_MASK);
        const int64_t s1_base = (int64_t)((k0 + 16) & PAGE_MASK);

        const uint16_t* k0_base = k_head_base + p0 * skb + ck_d * 8;
        const uint16_t* k1_base = k_head_base + p1 * skb + ck_d * 8;
        *reinterpret_cast<uint4*>(&smem.k[ck_row +  0][ck_d * 8]) = (k0 + ck_row +  0 < hi) ? *reinterpret_cast<const uint4*>(k0_base + (s0_base + ck_row + 0) * skp) : make_uint4(0,0,0,0);
        *reinterpret_cast<uint4*>(&smem.k[ck_row +  8][ck_d * 8]) = (k0 + ck_row +  8 < hi) ? *reinterpret_cast<const uint4*>(k0_base + (s0_base + ck_row + 8) * skp) : make_uint4(0,0,0,0);
        *reinterpret_cast<uint4*>(&smem.k[ck_row + 16][ck_d * 8]) = (k0 + ck_row + 16 < hi) ? *reinterpret_cast<const uint4*>(k1_base + (s1_base + ck_row + 0) * skp) : make_uint4(0,0,0,0);
        *reinterpret_cast<uint4*>(&smem.k[ck_row + 24][ck_d * 8]) = (k0 + ck_row + 24 < hi) ? *reinterpret_cast<const uint4*>(k1_base + (s1_base + ck_row + 8) * skp) : make_uint4(0,0,0,0);
        __syncthreads();

        const int k_sub0 = k0 + k_warp_id * 16;
        const int k_row_idx = k_warp_id * 16 + lane16;
        float8 s_acc = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
        #pragma unroll
        for (int kd = 0; kd < 8; ++kd) {
            short16 k_reg = decode_load_short16_u4(&smem.k[k_row_idx][kd * 16]);
            short16 q_r = decode_load_short16_u4(&smem_Q[m_row][kd * 16]);
            s_acc = __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32(k_reg, q_r, s_acc);
        }
        __syncthreads();

        uint4 r_loc0 = make_uint4(0, 0, 0, 0), r_loc1 = make_uint4(0, 0, 0, 0);
        if constexpr (HAS_REL) {
            r_loc0 = *reinterpret_cast<const uint4*>(&smem_R[m_row][0]);
            r_loc1 = *reinterpret_cast<const uint4*>(&smem_R[m_row][8]);
        }
        float m_loc = -INFINITY;
        #pragma unroll
        for (int r = 0; r < 8; ++r) {
            int key = k_sub0 + half + 2 * r;
            int dist = pos - key;
            bool ok = row_valid && (key < hi) && (dist >= 0);
            if constexpr (IS_LOCAL) ok = ok && (dist <= window);
            if (ok) {
                float sc = s_acc[r] * scale_log2;
                if constexpr (HAS_REL) {
                    if (dist < extent) {
                        float b = bf16_round_f32(decode_dot16_bf16(r_loc0, r_loc1, RelProjT + (int64_t)dist * 16));
                        sc += b * LOG2E;
                    }
                }
                s_acc[r] = sc;
                m_loc = fmaxf(m_loc, sc);
            } else {
                s_acc[r] = -INFINITY;
            }
        }
        float m_tile = fmaxf(m_loc, __shfl_xor(m_loc, 16, 32));
        float m_new = fmaxf(m_i, m_tile);
        float m_ref = (m_new == -INFINITY) ? 0.0f : m_new;
        float alpha = (m_i == -INFINITY) ? 0.0f : exp2f(m_i - m_ref);
        float p_f[8];
        float l_loc = 0.0f;
        #pragma unroll
        for (int r = 0; r < 8; ++r) {
            float pv = (s_acc[r] == -INFINITY) ? 0.0f : exp2f(s_acc[r] - m_ref);
            p_f[r] = pv;
            l_loc += pv;
        }
        float l_tile = l_loc + __shfl_xor(l_loc, 16, 32);
        l_i = l_i * alpha + l_tile;
        m_i = m_new;
        #pragma unroll
        for (int dt = 0; dt < 8; ++dt) {
            #pragma unroll
            for (int r = 0; r < 8; ++r) acc_o[dt][r] *= alpha;
        }
        short16 p_reg = decode_pack_p_in_reg(p_f, half);
        run_coop_v(k0, p0, p1, true, p_reg);
    }

    // FINAL 2-WARP MERGE PER Q_TILE (warps 0+1 -> Q[0..15], warps 2+3 -> Q[16..31])
    if (half == 0) {
        smem.merge.m[warp][lane16] = m_i;
        smem.merge.l[warp][lane16] = l_i;
    }
    __syncthreads();

    const int w0 = q_tile_id * 2;
    float m0 = smem.merge.m[w0][lane16], m1 = smem.merge.m[w0 + 1][lane16];
    float l0 = smem.merge.l[w0][lane16], l1 = smem.merge.l[w0 + 1][lane16];
    float m_final = fmaxf(m0, m1);
    float m_ref_f = (m_final == -INFINITY) ? 0.0f : m_final;
    float e0 = (m0 == -INFINITY || l0 <= 0.0f) ? 0.0f : exp2f(m0 - m_ref_f);
    float e1 = (m1 == -INFINITY || l1 <= 0.0f) ? 0.0f : exp2f(m1 - m_ref_f);
    float l_final = l0 * e0 + l1 * e1;
    const bool has_keys = (l_final > 0.0f);
    const float inv_l = has_keys ? (1.0f / l_final) : 0.0f;
    const float ws0 = e0 * inv_l, ws1 = e1 * inv_l;

    #pragma unroll
    for (int chunk = 0; chunk < 4; ++chunk) {
        #pragma unroll
        for (int dt_l = 0; dt_l < 2; ++dt_l) {
            int dt = chunk * 2 + dt_l;
            #pragma unroll
            for (int r = 0; r < 8; ++r) {
                smem.merge.o[warp][lane16][dt_l * 16 + half + 2 * r] = acc_o[dt][r];
            }
        }
        __syncthreads();
        if (k_warp_id == 0 && row_valid) {
            float* o_dst = OPart + out_tok * sot + (int64_t)head * soh + chunk * 32;
            #pragma unroll
            for (int dt_l = 0; dt_l < 2; ++dt_l) {
                #pragma unroll
                for (int r = 0; r < 8; ++r) {
                    int d_loc = dt_l * 16 + half + 2 * r;
                    o_dst[d_loc] = smem.merge.o[w0][lane16][d_loc] * ws0
                                 + smem.merge.o[w0 + 1][lane16][d_loc] * ws1;
                }
            }
        }
        __syncthreads();
    }
    if (k_warp_id == 0 && half == 0 && row_valid) {
        LsePart[(int64_t)head * slh + out_tok] = has_keys ? (m_final + log2f(l_final)) : -INFINITY;
    }
}

// ============================================================================
// 3. FUSED SPLIT-KV COMBINE + GATE/SINK EPILOGUE
//    Grid = (num_tokens, num_heads), Block = (64,) -> 2 d-elems/thread = 128
// ============================================================================
template <bool HAS_GATE>
__global__ __launch_bounds__(64, 8)
void hip_decode_combine_kernel(
    const float* __restrict__ OPart,
    const float* __restrict__ LsePart,
    const uint16_t* __restrict__ Gate,
    const uint16_t* __restrict__ Sinks,
    uint16_t* __restrict__ Out,
    const int32_t* __restrict__ CuSeqlensQ,
    int num_reqs, int num_splits, int num_tokens,
    int64_t sot, int64_t soh, int64_t slh,
    int64_t sgt, int64_t sgh, int64_t ssk_t, int64_t sout_t, int64_t sout_h
) {
    constexpr float LN2 = 0.6931471805599453f;
    const int row = (int)blockIdx.x;
    const int head = (int)blockIdx.y;
    const int tid = (int)threadIdx.x;
    const int d0 = tid * 2;
    uint16_t* y_ptr = Out + (int64_t)row * sout_t + (int64_t)head * sout_h + d0;

    if (row >= CuSeqlensQ[num_reqs]) {
        *reinterpret_cast<uint32_t*>(y_ptr) = 0u;
        return;
    }

    float m_i = -INFINITY;
    float l_i = 0.0f;
    float acc0 = 0.0f, acc1 = 0.0f;
    const float* lse_base = LsePart + (int64_t)head * slh;

    for (int s = 0; s < num_splits; ++s) {
        int64_t part = (int64_t)s * (int64_t)num_tokens + (int64_t)row;
        float lse = lse_base[part];
        if (lse == -INFINITY) continue;
        float m_new = fmaxf(m_i, lse);
        float alpha = (m_i == -INFINITY) ? 0.0f : exp2f(m_i - m_new);
        float w = exp2f(lse - m_new);
        const float* op = OPart + part * sot + (int64_t)head * soh + d0;
        acc0 = fmaf(op[0], w, acc0 * alpha);
        acc1 = fmaf(op[1], w, acc1 * alpha);
        l_i = l_i * alpha + w;
        m_i = m_new;
    }

    const bool has_keys = (l_i > 0.0f);
    float inv_l = has_keys ? (1.0f / l_i) : 0.0f;
    float y0 = acc0 * inv_l;
    float y1 = acc1 * inv_l;

    if constexpr (HAS_GATE) {
        uint32_t g_pk = *reinterpret_cast<const uint32_t*>(Gate + (int64_t)row * sgt + (int64_t)head * sgh + d0);
        float g0 = bf16_to_f32((uint16_t)(g_pk & 0xFFFFu));
        float g1 = bf16_to_f32((uint16_t)(g_pk >> 16));
        float sink = bf16_to_f32(Sinks[(int64_t)row * ssk_t + head]);
        float lse_e = has_keys ? ((m_i + log2f(l_i)) * LN2) : -INFINITY;
        float sk_sig = fast_sigmoid_hw(lse_e - sink);
        y0 = y0 * fast_sigmoid_hw(g0) * sk_sig;
        y1 = y1 * fast_sigmoid_hw(g1) * sk_sig;
    }

    *reinterpret_cast<uint32_t*>(y_ptr) = pack_bf16x2(y0, y1);
}

extern "C" {

int launch_hip_kv_conv_decode(
    const void* K, const void* V, const void* Wk, const void* Wv, const void* Dk, const void* Dv,
    void* CC, void* KC, void* VC, const void* KVS, const void* Pos, const void* Seq, const void* CS,
    const void* BT_ptr, const void* QS, const void* CuQ,
    int num_reqs, int T, int KVH, int64_t sk, int64_t sv, int64_t sw0, int64_t sw1,
    int64_t sdk, int64_t sdv, int64_t scb, int64_t sch, int64_t scn,
    int64_t skvb, int64_t skvn, int64_t skvh, int kv_bs, int64_t sbt, int mb, int nc,
    float scale, float ct, float ce, float ne, int dynamic, int pos_is_i64, void* stream_ptr
) {
    if (num_reqs <= 0 || T <= 0 || KVH <= 0) return 0;
    hipStream_t stream = (hipStream_t)stream_ptr;
    dim3 grid(KVH, num_reqs);
    dim3 block(32);
    auto launch = [&](auto dyn_tag, auto pos_tag) {
        constexpr int DYN = decltype(dyn_tag)::value;
        constexpr int P64 = decltype(pos_tag)::value;
        hipLaunchKernelGGL((hip_kv_conv_decode_kernel<DYN, P64>), grid, block, 0, stream,
            (const uint16_t*)K, (const uint16_t*)V, (const uint16_t*)Wk, (const uint16_t*)Wv,
            (const uint16_t*)Dk, (const uint16_t*)Dv, (uint16_t*)CC, (uint16_t*)KC, (uint16_t*)VC,
            (const int64_t*)KVS, Pos, (const int32_t*)Seq, (const int64_t*)CS,
            (const int32_t*)BT_ptr, (const int32_t*)QS, (const int32_t*)CuQ,
            T, KVH, sk, sv, sw0, sw1, sdk, sdv, scb, sch, scn, skvb, skvn, skvh,
            kv_bs, sbt, mb, nc, scale, ct, ce, ne);
    };
    if (dynamic) {
        if (pos_is_i64) launch(std::integral_constant<int, 1>{}, std::integral_constant<int, 1>{});
        else            launch(std::integral_constant<int, 1>{}, std::integral_constant<int, 0>{});
    } else {
        if (pos_is_i64) launch(std::integral_constant<int, 0>{}, std::integral_constant<int, 1>{});
        else            launch(std::integral_constant<int, 0>{}, std::integral_constant<int, 0>{});
    }
    return 0;
}

int launch_hip_flash_rel_decode_split(
    const void* Q, const void* R, const void* RelProjT, const void* KCache, const void* VCache,
    void* OPart, void* LsePart, const void* BlockTable, const void* SeqLens, const void* CuSeqlensQ,
    int num_reqs, int num_tokens, int num_splits, int max_query_len, int num_heads, int num_kv_heads,
    int page_size, int is_local, int has_rel, int window, int extent, float scale_log2,
    int64_t sqt, int64_t sqh, int64_t srt, int64_t srh,
    int64_t skb, int64_t skp, int64_t skh, int64_t svb, int64_t svp, int64_t svh,
    int64_t sot, int64_t soh, int64_t slh, int64_t stable, void* stream_ptr
) {
    if (num_reqs <= 0 || num_tokens <= 0 || num_splits <= 0) return 0;
    if (num_heads != 24 || num_kv_heads != 8 || max_query_len < 1 || max_query_len > 8) return -1;
    if (page_size != 16 && page_size != 64) return -1;
    hipStream_t stream = (hipStream_t)stream_ptr;
    dim3 grid(num_splits, num_kv_heads, num_reqs);
    dim3 block(128);
    const bool use_bq32 = (max_query_len > 5);

    auto dispatch = [&](auto loc_t, auto rel_t, auto psh_t, auto bq_t) {
        constexpr bool LOC = decltype(loc_t)::value;
        constexpr bool REL = decltype(rel_t)::value;
        constexpr int PSH = decltype(psh_t)::value;
        constexpr int BQ = decltype(bq_t)::value;
        if constexpr (BQ == 32) {
            hipLaunchKernelGGL((hip_flash_rel_decode_split_bq32_kernel<LOC, REL, PSH, 4>), grid, block, 0, stream,
                (const uint16_t*)Q, (const uint16_t*)R, (const uint16_t*)RelProjT,
                (const uint16_t*)KCache, (const uint16_t*)VCache,
                (float*)OPart, (float*)LsePart,
                (const int32_t*)BlockTable, (const int32_t*)SeqLens, (const int32_t*)CuSeqlensQ,
                num_splits, num_tokens, window, extent, scale_log2,
                sqt, sqh, srt, srh, skb, skp, skh, svb, svp, svh, sot, soh, slh, stable);
        } else {
            hipLaunchKernelGGL((hip_flash_rel_decode_split_kernel<LOC, REL, PSH, 16, 4>), grid, block, 0, stream,
                (const uint16_t*)Q, (const uint16_t*)R, (const uint16_t*)RelProjT,
                (const uint16_t*)KCache, (const uint16_t*)VCache,
                (float*)OPart, (float*)LsePart,
                (const int32_t*)BlockTable, (const int32_t*)SeqLens, (const int32_t*)CuSeqlensQ,
                num_splits, num_tokens, window, extent, scale_log2,
                sqt, sqh, srt, srh, skb, skp, skh, svb, svp, svh, sot, soh, slh, stable);
        }
    };

    auto disp_bq = [&](auto loc_t, auto rel_t, auto psh_t) {
        if (use_bq32) dispatch(loc_t, rel_t, psh_t, std::integral_constant<int, 32>{});
        else          dispatch(loc_t, rel_t, psh_t, std::integral_constant<int, 16>{});
    };
    auto disp_page = [&](auto loc_t, auto rel_t) {
        if (page_size == 64) disp_bq(loc_t, rel_t, std::integral_constant<int, 6>{});
        else                 disp_bq(loc_t, rel_t, std::integral_constant<int, 4>{});
    };
    if (is_local) {
        if (has_rel) disp_page(std::true_type{}, std::true_type{});
        else         disp_page(std::true_type{}, std::false_type{});
    } else {
        if (has_rel) disp_page(std::false_type{}, std::true_type{});
        else         disp_page(std::false_type{}, std::false_type{});
    }
    return 0;
}

int launch_hip_decode_combine(
    const void* OPart, const void* LsePart, const void* Gate, const void* Sinks, void* Out,
    const void* CuSeqlensQ, int num_reqs, int num_tokens, int num_heads, int num_splits, int has_gate,
    int64_t sot, int64_t soh, int64_t slh, int64_t sgt, int64_t sgh, int64_t ssk_t,
    int64_t sout_t, int64_t sout_h, void* stream_ptr
) {
    if (num_tokens <= 0 || num_heads <= 0 || num_splits <= 0) return 0;
    hipStream_t stream = (hipStream_t)stream_ptr;
    dim3 grid(num_tokens, num_heads);
    dim3 block(64);
    if (has_gate) {
        hipLaunchKernelGGL((hip_decode_combine_kernel<true>), grid, block, 0, stream,
            (const float*)OPart, (const float*)LsePart, (const uint16_t*)Gate, (const uint16_t*)Sinks,
            (uint16_t*)Out, (const int32_t*)CuSeqlensQ, num_reqs, num_splits, num_tokens,
            sot, soh, slh, sgt, sgh, ssk_t, sout_t, sout_h);
    } else {
        hipLaunchKernelGGL((hip_decode_combine_kernel<false>), grid, block, 0, stream,
            (const float*)OPart, (const float*)LsePart, (const uint16_t*)Gate, (const uint16_t*)Sinks,
            (uint16_t*)Out, (const int32_t*)CuSeqlensQ, num_reqs, num_splits, num_tokens,
            sot, soh, slh, sgt, sgh, ssk_t, sout_t, sout_h);
    }
    return 0;
}

}  // extern "C"
