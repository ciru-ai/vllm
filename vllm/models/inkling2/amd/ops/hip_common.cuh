// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 CIRU AI.
// Shared device helpers and ABI metadata for Inkling 2 Mini gfx1151 native HIP kernels.
#pragma once

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <stdint.h>

#ifndef INKLING_HIP_ABI_VERSION
#define INKLING_HIP_ABI_VERSION 3
#endif

#define INKLING_STR_(x) #x
#define INKLING_STR(x) INKLING_STR_(x)
#ifdef INKLING_HIP_SRC_HASH_RAW
#define INKLING_HIP_SRC_HASH INKLING_STR(INKLING_HIP_SRC_HASH_RAW)
#elif !defined(INKLING_HIP_SRC_HASH)
#define INKLING_HIP_SRC_HASH "unknown"
#endif

typedef _Float16 half16 __attribute__((ext_vector_type(16)));
typedef short short16 __attribute__((ext_vector_type(16)));
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

// Unpack 1 packed W6 byte (2 low nibbles) + 4-bit hi2 pair -> 2 packed BF16 codes in [-32, 31]
__device__ __forceinline__ uint32_t unpack_w6_pair_bf16(uint32_t p4_byte, uint32_t h2_nibble) {
    int lo_q = (int)((p4_byte & 15u) | ((h2_nibble & 3u) << 4)) - 32;
    int hi_q = (int)(((p4_byte >> 4) & 15u) | (((h2_nibble >> 2) & 3u) << 4)) - 32;
    uint32_t lo_b = __float_as_uint((float)lo_q) >> 16;
    uint32_t hi_b = __float_as_uint((float)hi_q) & 0xFFFF0000u;
    return lo_b | hi_b;
}

// Unpack 1 packed W4 byte (2 offset-binary nibbles) -> 2 packed BF16 codes in [-8, 7]
__device__ __forceinline__ uint32_t unpack_w4_pair_bf16(uint32_t p4_byte) {
    int lo_q = (int)(p4_byte & 15u) - 8;
    int hi_q = (int)((p4_byte >> 4) & 15u) - 8;
    uint32_t lo_b = __float_as_uint((float)lo_q) >> 16;
    uint32_t hi_b = __float_as_uint((float)hi_q) & 0xFFFF0000u;
    return lo_b | hi_b;
}
