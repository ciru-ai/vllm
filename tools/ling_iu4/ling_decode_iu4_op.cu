#include <torch/extension.h>

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>

#include <hip/hip_bf16.h>
#include <hip/hip_runtime.h>

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <cstdio>
#include <mutex>

namespace ling_iu4
{
void launch_quantize(const torch::Tensor& input,
                     torch::Tensor& packed,
                     torch::Tensor& scales,
                     const at::cuda::CUDAStream& stream);
void check_launch(const char* stage);
} // namespace ling_iu4

namespace ling_iu4_decode
{

using bf16_t = __hip_bfloat16;
using int2x32 = int __attribute__((ext_vector_type(2)));
using int8x32 = int __attribute__((ext_vector_type(8)));
using float8x32 = float __attribute__((ext_vector_type(8)));

constexpr int kGroupSize = 32;
constexpr uint32_t kUint4B8ToS4Mask = 0x88888888u;

float clip_ratio_from_env(const char* name)
{
    const char* raw = std::getenv(name);
    if(raw == nullptr || raw[0] == '\0')
    {
        return 1.0f;
    }
    char* end = nullptr;
    const float value = std::strtof(raw, &end);
    TORCH_CHECK(end != raw && *end == '\0' && value >= 0.5f && value <= 1.0f,
                name,
                " must be a float in [0.5, 1.0]");
    return value;
}

float w13_clip_ratio()
{
    static const float value = clip_ratio_from_env("LING_IU4_W13_CLIP");
    return value;
}

float w2_clip_ratio()
{
    static const float value = clip_ratio_from_env("LING_IU4_W2_CLIP");
    return value;
}

__host__ __device__ constexpr int c_fragment_row(const int lane, const int reg)
{
    return 2 * reg + ((lane & 31) >> 4);
}

__global__ void quantize_pack_s4_g32_clipped(
    const bf16_t* __restrict__ input,
    uint32_t* __restrict__ packed,
    float* __restrict__ scales,
    const int rows,
    const int K,
    const float clip_ratio)
{
    const int lane = threadIdx.x & 31;
    const int group = blockIdx.x;
    const int row = blockIdx.y;
    if(row >= rows)
    {
        return;
    }
    const int groups = K / kGroupSize;
    const int k = group * kGroupSize + lane;
    const float value = __bfloat162float(input[static_cast<long>(row) * K + k]);
    float maximum = fabsf(value);
#pragma unroll
    for(int offset = 16; offset > 0; offset >>= 1)
    {
        maximum = fmaxf(maximum, __shfl_down(maximum, offset, 32));
    }
    maximum = __shfl(maximum, 0, 32);
    const float clipped_maximum = maximum * clip_ratio;
    const float scale =
        clipped_maximum > 0.0f ? clipped_maximum * (1.0f / 7.0f) : 0.0f;
    const double normalized = clipped_maximum > 0.0f
                                  ? static_cast<double>(value) * 7.0 /
                                        static_cast<double>(clipped_maximum)
                                  : 0.0;
    int q = clipped_maximum > 0.0f ? __double2int_rn(normalized) : 0;
    q = q < -7 ? -7 : (q > 7 ? 7 : q);
    const int source_base = (lane >> 3) << 3;
    uint32_t word = 0;
#pragma unroll
    for(int i = 0; i < 8; ++i)
    {
        const int code = __shfl(q, source_base + i, 32);
        word |= static_cast<uint32_t>(code & 0xF) << (4 * i);
    }
    if((lane & 7) == 0)
    {
        const int words_per_row = K / 8;
        packed[static_cast<long>(row) * words_per_row + group * 4 + lane / 8] =
            word;
    }
    if(lane == 0)
    {
        scales[static_cast<long>(row) * groups + group] = scale;
    }
}

void launch_quantize_with_clip(const torch::Tensor& input,
                               torch::Tensor& packed,
                               torch::Tensor& scales,
                               const at::cuda::CUDAStream& stream,
                               const float clip_ratio)
{
    if(clip_ratio == 1.0f)
    {
        ling_iu4::launch_quantize(input, packed, scales, stream);
        return;
    }
    const int rows = static_cast<int>(input.size(0));
    const int K = static_cast<int>(input.size(1));
    const dim3 grid(K / kGroupSize, rows);
    quantize_pack_s4_g32_clipped<<<grid, 32, 0, stream>>>(
        reinterpret_cast<const bf16_t*>(input.data_ptr<at::BFloat16>()),
        reinterpret_cast<uint32_t*>(packed.data_ptr<int32_t>()),
        scales.data_ptr<float>(),
        rows,
        K,
        clip_ratio);
    ling_iu4::check_launch("clipped quantize_pack_s4_g32 launch failed");
}

template <bool Residual3, bool Residual4, bool Outlier1 = false>
__global__ void quantize_pack_s4_g32_residual(
    const bf16_t* __restrict__ input,
    uint32_t* __restrict__ packed0,
    float* __restrict__ scales0,
    uint32_t* __restrict__ packed1,
    float* __restrict__ scales1,
    uint32_t* __restrict__ packed2,
    float* __restrict__ scales2,
    uint32_t* __restrict__ packed3,
    float* __restrict__ scales3,
    int32_t* __restrict__ outlier_indices,
    float* __restrict__ outlier_values,
    const int rows,
    const int K)
{
    const int lane = threadIdx.x & 31;
    const int group = blockIdx.x;
    const int row = blockIdx.y;
    if(row >= rows)
    {
        return;
    }
    const int groups = K / kGroupSize;
    const int k = group * kGroupSize + lane;
    const float value = __bfloat162float(input[static_cast<long>(row) * K + k]);
    int outlier_lane = 0;
    float outlier_value = 0.0f;
    if constexpr(Outlier1)
    {
        float largest = fabsf(value);
        int largest_lane = lane;
#pragma unroll
        for(int offset = 16; offset > 0; offset >>= 1)
        {
            const float candidate = __shfl_down(largest, offset, 32);
            const int candidate_lane = __shfl_down(largest_lane, offset, 32);
            if(lane < offset && candidate > largest)
            {
                largest = candidate;
                largest_lane = candidate_lane;
            }
        }
        outlier_lane = __shfl(largest_lane, 0, 32);
        outlier_value = __shfl(value, outlier_lane, 32);
    }
    const float quant_value = Outlier1 && lane == outlier_lane ? 0.0f : value;
    float maximum0 = fabsf(quant_value);
#pragma unroll
    for(int offset = 16; offset > 0; offset >>= 1)
    {
        maximum0 = fmaxf(maximum0, __shfl_down(maximum0, offset, 32));
    }
    maximum0 = __shfl(maximum0, 0, 32);
    const float scale0 = maximum0 > 0.0f ? maximum0 * (1.0f / 7.0f) : 0.0f;
    int q0 = scale0 > 0.0f
                 ? __double2int_rn(static_cast<double>(quant_value) /
                                    static_cast<double>(scale0))
                 : 0;
    q0 = q0 < -7 ? -7 : (q0 > 7 ? 7 : q0);
    const float residual = quant_value - static_cast<float>(q0) * scale0;
    float maximum1 = fabsf(residual);
#pragma unroll
    for(int offset = 16; offset > 0; offset >>= 1)
    {
        maximum1 = fmaxf(maximum1, __shfl_down(maximum1, offset, 32));
    }
    maximum1 = __shfl(maximum1, 0, 32);
    const float scale1 = maximum1 > 0.0f ? maximum1 * (1.0f / 7.0f) : 0.0f;
    int q1 = scale1 > 0.0f
                 ? __double2int_rn(static_cast<double>(residual) /
                                    static_cast<double>(scale1))
                 : 0;
    q1 = q1 < -7 ? -7 : (q1 > 7 ? 7 : q1);
    const float residual_second =
        residual - static_cast<float>(q1) * scale1;
    float maximum2 = fabsf(residual_second);
#pragma unroll
    for(int offset = 16; offset > 0; offset >>= 1)
    {
        maximum2 = fmaxf(maximum2, __shfl_down(maximum2, offset, 32));
    }
    maximum2 = __shfl(maximum2, 0, 32);
    const float scale2 = maximum2 > 0.0f ? maximum2 * (1.0f / 7.0f) : 0.0f;
    int q2 = scale2 > 0.0f
                 ? __double2int_rn(static_cast<double>(residual_second) /
                                    static_cast<double>(scale2))
                 : 0;
    q2 = q2 < -7 ? -7 : (q2 > 7 ? 7 : q2);
    const float residual_third =
        residual_second - static_cast<float>(q2) * scale2;
    float maximum3 = fabsf(residual_third);
#pragma unroll
    for(int offset = 16; offset > 0; offset >>= 1)
    {
        maximum3 = fmaxf(maximum3, __shfl_down(maximum3, offset, 32));
    }
    maximum3 = __shfl(maximum3, 0, 32);
    const float scale3 = maximum3 > 0.0f ? maximum3 * (1.0f / 7.0f) : 0.0f;
    int q3 = scale3 > 0.0f
                 ? __double2int_rn(static_cast<double>(residual_third) /
                                    static_cast<double>(scale3))
                 : 0;
    q3 = q3 < -7 ? -7 : (q3 > 7 ? 7 : q3);
    const int source_base = (lane >> 3) << 3;
    uint32_t word0 = 0;
    uint32_t word1 = 0;
    uint32_t word2 = 0;
    uint32_t word3 = 0;
#pragma unroll
    for(int i = 0; i < 8; ++i)
    {
        word0 |= static_cast<uint32_t>(__shfl(q0, source_base + i, 32) & 0xF)
                 << (4 * i);
        word1 |= static_cast<uint32_t>(__shfl(q1, source_base + i, 32) & 0xF)
                 << (4 * i);
        if constexpr(Residual3)
        {
            word2 |=
                static_cast<uint32_t>(__shfl(q2, source_base + i, 32) & 0xF)
                << (4 * i);
        }
        if constexpr(Residual4)
        {
            word3 |=
                static_cast<uint32_t>(__shfl(q3, source_base + i, 32) & 0xF)
                << (4 * i);
        }
    }
    if((lane & 7) == 0)
    {
        const long word_index = static_cast<long>(row) * (K / 8) +
                                group * 4 + lane / 8;
        packed0[word_index] = word0;
        packed1[word_index] = word1;
        if constexpr(Residual3)
        {
            packed2[word_index] = word2;
        }
        if constexpr(Residual4)
        {
            packed3[word_index] = word3;
        }
    }
    if(lane == 0)
    {
        const long scale_index = static_cast<long>(row) * groups + group;
        scales0[scale_index] = scale0;
        scales1[scale_index] = scale1;
        if constexpr(Residual3)
        {
            scales2[scale_index] = scale2;
        }
        if constexpr(Residual4)
        {
            scales3[scale_index] = scale3;
        }
        if constexpr(Outlier1)
        {
            outlier_indices[scale_index] = outlier_lane;
            outlier_values[scale_index] = outlier_value;
        }
    }
}

void launch_quantize_residual2(const torch::Tensor& input,
                               torch::Tensor& packed0,
                               torch::Tensor& scales0,
                               torch::Tensor& packed1,
                               torch::Tensor& scales1,
                               const at::cuda::CUDAStream& stream)
{
    const int rows = static_cast<int>(input.size(0));
    const int K = static_cast<int>(input.size(1));
    const dim3 grid(K / kGroupSize, rows);
    quantize_pack_s4_g32_residual<false, false><<<grid, 32, 0, stream>>>(
        reinterpret_cast<const bf16_t*>(input.data_ptr<at::BFloat16>()),
        reinterpret_cast<uint32_t*>(packed0.data_ptr<int32_t>()),
        scales0.data_ptr<float>(),
        reinterpret_cast<uint32_t*>(packed1.data_ptr<int32_t>()),
        scales1.data_ptr<float>(),
        nullptr,
        nullptr,
        nullptr,
        nullptr,
        nullptr,
        nullptr,
        rows,
        K);
    ling_iu4::check_launch("residual2 quantize_pack_s4_g32 launch failed");
}

void launch_quantize_residual3(const torch::Tensor& input,
                               torch::Tensor& packed0,
                               torch::Tensor& scales0,
                               torch::Tensor& packed1,
                               torch::Tensor& scales1,
                               torch::Tensor& packed2,
                               torch::Tensor& scales2,
                               const at::cuda::CUDAStream& stream)
{
    const int rows = static_cast<int>(input.size(0));
    const int K = static_cast<int>(input.size(1));
    const dim3 grid(K / kGroupSize, rows);
    quantize_pack_s4_g32_residual<true, false><<<grid, 32, 0, stream>>>(
        reinterpret_cast<const bf16_t*>(input.data_ptr<at::BFloat16>()),
        reinterpret_cast<uint32_t*>(packed0.data_ptr<int32_t>()),
        scales0.data_ptr<float>(),
        reinterpret_cast<uint32_t*>(packed1.data_ptr<int32_t>()),
        scales1.data_ptr<float>(),
        reinterpret_cast<uint32_t*>(packed2.data_ptr<int32_t>()),
        scales2.data_ptr<float>(),
        nullptr,
        nullptr,
        nullptr,
        nullptr,
        rows,
        K);
    ling_iu4::check_launch("residual3 quantize_pack_s4_g32 launch failed");
}

void launch_quantize_residual4(const torch::Tensor& input,
                               torch::Tensor& packed0,
                               torch::Tensor& scales0,
                               torch::Tensor& packed1,
                               torch::Tensor& scales1,
                               torch::Tensor& packed2,
                               torch::Tensor& scales2,
                               torch::Tensor& packed3,
                               torch::Tensor& scales3,
                               const at::cuda::CUDAStream& stream)
{
    const int rows = static_cast<int>(input.size(0));
    const int K = static_cast<int>(input.size(1));
    const dim3 grid(K / kGroupSize, rows);
    quantize_pack_s4_g32_residual<true, true><<<grid, 32, 0, stream>>>(
        reinterpret_cast<const bf16_t*>(input.data_ptr<at::BFloat16>()),
        reinterpret_cast<uint32_t*>(packed0.data_ptr<int32_t>()),
        scales0.data_ptr<float>(),
        reinterpret_cast<uint32_t*>(packed1.data_ptr<int32_t>()),
        scales1.data_ptr<float>(),
        reinterpret_cast<uint32_t*>(packed2.data_ptr<int32_t>()),
        scales2.data_ptr<float>(),
        reinterpret_cast<uint32_t*>(packed3.data_ptr<int32_t>()),
        scales3.data_ptr<float>(),
        nullptr,
        nullptr,
        rows,
        K);
    ling_iu4::check_launch("residual4 quantize_pack_s4_g32 launch failed");
}

void launch_quantize_residual3_outlier1(
    const torch::Tensor& input,
    torch::Tensor& packed0,
    torch::Tensor& scales0,
    torch::Tensor& packed1,
    torch::Tensor& scales1,
    torch::Tensor& packed2,
    torch::Tensor& scales2,
    torch::Tensor& outlier_indices,
    torch::Tensor& outlier_values,
    const at::cuda::CUDAStream& stream)
{
    const int rows = static_cast<int>(input.size(0));
    const int K = static_cast<int>(input.size(1));
    const dim3 grid(K / kGroupSize, rows);
    quantize_pack_s4_g32_residual<true, false, true>
        <<<grid, 32, 0, stream>>>(
        reinterpret_cast<const bf16_t*>(input.data_ptr<at::BFloat16>()),
        reinterpret_cast<uint32_t*>(packed0.data_ptr<int32_t>()),
        scales0.data_ptr<float>(),
        reinterpret_cast<uint32_t*>(packed1.data_ptr<int32_t>()),
        scales1.data_ptr<float>(),
        reinterpret_cast<uint32_t*>(packed2.data_ptr<int32_t>()),
        scales2.data_ptr<float>(),
        nullptr,
        nullptr,
        outlier_indices.data_ptr<int32_t>(),
        outlier_values.data_ptr<float>(),
        rows,
        K);
    ling_iu4::check_launch(
        "residual3-outlier1 quantize_pack_s4_g32 launch failed");
}

__global__ void add_bf16_inplace(bf16_t* __restrict__ destination,
                                 const bf16_t* __restrict__ residual,
                                 const long count)
{
    const long index = static_cast<long>(blockIdx.x) * blockDim.x + threadIdx.x;
    if(index < count)
    {
        destination[index] = __float2bfloat16(
            __bfloat162float(destination[index]) +
            __bfloat162float(residual[index]));
    }
}

// One Wave32 quantizes two independent K16 activation groups. This matches the
// native IU4 WMMA depth, while official Ling weight scales remain G32.
__global__ void quantize_pack_s4_g16(const bf16_t* __restrict__ input,
                                     uint32_t* __restrict__ packed,
                                     float* __restrict__ scales,
                                     const int rows,
                                     const int K)
{
    const int wave_lane = threadIdx.x & 31;
    const int subgroup = wave_lane >> 4;
    const int lane = wave_lane & 15;
    const int group16 = blockIdx.x * 2 + subgroup;
    const int row = blockIdx.y;
    if(row >= rows)
    {
        return;
    }
    const int k = group16 * 16 + lane;
    const float value = __bfloat162float(input[static_cast<long>(row) * K + k]);
    float maximum = fabsf(value);
#pragma unroll
    for(int offset = 8; offset > 0; offset >>= 1)
    {
        maximum = fmaxf(maximum, __shfl_down(maximum, offset, 16));
    }
    maximum = __shfl(maximum, subgroup * 16, 32);
    const float scale = maximum > 0.0f ? maximum * (1.0f / 7.0f) : 0.0f;
    const double normalized = maximum > 0.0f
                                  ? static_cast<double>(value) * 7.0 /
                                        static_cast<double>(maximum)
                                  : 0.0;
    int q = maximum > 0.0f ? __double2int_rn(normalized) : 0;
    q = q < -7 ? -7 : (q > 7 ? 7 : q);
    const int source_base = subgroup * 16 + ((lane >> 3) << 3);
    uint32_t word = 0;
#pragma unroll
    for(int i = 0; i < 8; ++i)
    {
        word |= static_cast<uint32_t>(__shfl(q, source_base + i, 32) & 0xF)
                << (4 * i);
    }
    if((lane & 7) == 0)
    {
        packed[static_cast<long>(row) * (K / 8) + group16 * 2 + lane / 8] = word;
    }
    if(lane == 0)
    {
        scales[static_cast<long>(row) * (K / 16) + group16] = scale;
    }
}

void launch_quantize_s4_g16(const torch::Tensor& input,
                            torch::Tensor& packed,
                            torch::Tensor& scales,
                            const at::cuda::CUDAStream& stream)
{
    const int rows = static_cast<int>(input.size(0));
    const int K = static_cast<int>(input.size(1));
    const dim3 grid(K / 32, rows);
    quantize_pack_s4_g16<<<grid, 32, 0, stream>>>(
        reinterpret_cast<const bf16_t*>(input.data_ptr<at::BFloat16>()),
        reinterpret_cast<uint32_t*>(packed.data_ptr<int32_t>()),
        scales.data_ptr<float>(),
        rows,
        K);
    ling_iu4::check_launch("quantize_pack_s4_g16 launch failed");
}

__device__ __forceinline__ int sum_s4_word(const uint32_t word)
{
    int sum = 0;
#pragma unroll
    for(int index = 0; index < 8; ++index)
    {
        const int nibble = static_cast<int>((word >> (4 * index)) & 0xF);
        sum += nibble >= 8 ? nibble - 16 : nibble;
    }
    return sum;
}

// Expand eight affine-S2 indices into eight signed-S4 nibbles. The two
// mirrored codebooks are {-7,-3,1,5} and {-5,-1,3,7}; the latter is selected
// by a negative stored scale. Both are affine step-4 sets, so expansion is a
// small bit-plane transform rather than a lookup table.
__device__ __forceinline__ uint32_t expand_affine_s2_eight(
    uint32_t packed_indices, const bool positive_tail)
{
    uint32_t dilated = packed_indices & 0xFFFFu;
    dilated = (dilated | (dilated << 8)) & 0x00FF00FFu;
    dilated = (dilated | (dilated << 4)) & 0x0F0F0F0Fu;
    dilated = (dilated | (dilated << 2)) & 0x33333333u;
    return 0x11111111u |
           (positive_tail ? 0x22222222u : 0u) |
           ((dilated & 0x11111111u) << 2) |
           (((~dilated) & 0x22222222u) << 2);
}

__device__ __forceinline__ int2x32 expand_affine_s2_sixteen(
    const uint32_t packed_indices, const bool positive_tail)
{
    int2x32 fragment;
    fragment[0] = static_cast<int>(
        expand_affine_s2_eight(packed_indices, positive_tail));
    fragment[1] = static_cast<int>(
        expand_affine_s2_eight(packed_indices >> 16, positive_tail));
    return fragment;
}

__global__ void quantize_pack_u4_g32_asymmetric(
    const bf16_t* __restrict__ input,
    uint32_t* __restrict__ packed,
    float* __restrict__ scales,
    int32_t* __restrict__ zero_points,
    const int rows,
    const int K)
{
    const int lane = threadIdx.x & 31;
    const int group = blockIdx.x;
    const int row = blockIdx.y;
    if(row >= rows)
    {
        return;
    }
    const int groups = K / kGroupSize;
    const int k = group * kGroupSize + lane;
    const float value = __bfloat162float(input[static_cast<long>(row) * K + k]);
    float minimum = value;
    float maximum = value;
#pragma unroll
    for(int offset = 16; offset > 0; offset >>= 1)
    {
        minimum = fminf(minimum, __shfl_down(minimum, offset, 32));
        maximum = fmaxf(maximum, __shfl_down(maximum, offset, 32));
    }
    minimum = __shfl(minimum, 0, 32);
    maximum = __shfl(maximum, 0, 32);
    const float span = maximum - minimum;
    const float scale = span > 0.0f ? span * (1.0f / 15.0f) : 0.0f;
    int zero = span > 0.0f
                   ? __double2int_rn(-static_cast<double>(minimum) /
                                     static_cast<double>(scale))
                   : 0;
    zero = zero < 0 ? 0 : (zero > 15 ? 15 : zero);
    int q = span > 0.0f
                ? __double2int_rn(static_cast<double>(value) /
                                      static_cast<double>(scale) +
                                  static_cast<double>(zero))
                : 0;
    q = q < 0 ? 0 : (q > 15 ? 15 : q);
    const int source_base = (lane >> 3) << 3;
    uint32_t word = 0;
#pragma unroll
    for(int i = 0; i < 8; ++i)
    {
        word |= static_cast<uint32_t>(__shfl(q, source_base + i, 32) & 0xF)
                << (4 * i);
    }
    if((lane & 7) == 0)
    {
        packed[static_cast<long>(row) * (K / 8) + group * 4 + lane / 8] = word;
    }
    if(lane == 0)
    {
        const long metadata = static_cast<long>(row) * groups + group;
        scales[metadata] = scale;
        zero_points[metadata] = zero;
    }
}

void launch_quantize_u4_asymmetric(const torch::Tensor& input,
                                   torch::Tensor& packed,
                                   torch::Tensor& scales,
                                   torch::Tensor& zero_points,
                                   const at::cuda::CUDAStream& stream)
{
    const int rows = static_cast<int>(input.size(0));
    const int K = static_cast<int>(input.size(1));
    const dim3 grid(K / kGroupSize, rows);
    quantize_pack_u4_g32_asymmetric<<<grid, 32, 0, stream>>>(
        reinterpret_cast<const bf16_t*>(input.data_ptr<at::BFloat16>()),
        reinterpret_cast<uint32_t*>(packed.data_ptr<int32_t>()),
        scales.data_ptr<float>(),
        zero_points.data_ptr<int32_t>(),
        rows,
        K);
    ling_iu4::check_launch("asymmetric quantize_pack_u4_g32 launch failed");
}

__global__ void silu_and_mul_bf16(const bf16_t* __restrict__ gate_up,
                                  bf16_t* __restrict__ output,
                                  const int rows,
                                  const int intermediate)
{
    const long index = static_cast<long>(blockIdx.x) * blockDim.x + threadIdx.x;
    const long count = static_cast<long>(rows) * intermediate;
    if(index >= count)
    {
        return;
    }
    const int row = static_cast<int>(index / intermediate);
    const int col = static_cast<int>(index - static_cast<long>(row) * intermediate);
    const float gate =
        __bfloat162float(gate_up[static_cast<long>(row) * (2 * intermediate) + col]);
    const float up = __bfloat162float(
        gate_up[static_cast<long>(row) * (2 * intermediate) + intermediate + col]);
    const float activated = gate / (1.0f + expf(-gate));
    const bf16_t activated_bf16 = __float2bfloat16(activated);
    output[index] = __float2bfloat16(__bfloat162float(activated_bf16) * up);
}

// One Wave32 computes one routed row by one 16-column output tile. Only row 0
// of the 16x16 WMMA tile is live; routes are independent and run concurrently.
// This deliberately targets Ling decode M1-M3, where routes normally belong
// to different experts and sorting cannot fill a shared-expert M16 tile.
template <bool InputPerToken, bool ApplyRouterWeight>
__global__ void routed_s4s4_g32_wmma(
    const uint32_t* __restrict__ a_packed,
    const uint32_t* __restrict__ b_uint4b8,
    const float* __restrict__ a_scale,
    const bf16_t* __restrict__ b_scale,
    const int32_t* __restrict__ topk_ids,
    const float* __restrict__ topk_weights,
    bf16_t* __restrict__ output,
    const int routes,
    const int top_k,
    const int experts,
    const int N,
    const int K)
{
    const int lane = threadIdx.x & 31;
    const int source_lane = lane & 15;
    const int route = blockIdx.y;
    const int tile_n = blockIdx.x * 16;
    if(route >= routes)
    {
        return;
    }

    const int expert = topk_ids[route];
    if(expert < 0 || expert >= experts)
    {
        return;
    }
    const int input_row = InputPerToken ? route / top_k : route;
    const int words_per_row = K / 8;
    const int groups = K / kGroupSize;
    const int source_col = tile_n + source_lane;

    float8x32 accumulator;
#pragma unroll
    for(int j = 0; j < 8; ++j)
    {
        accumulator[j] = 0.0f;
    }

    for(int group = 0; group < groups; ++group)
    {
        int8x32 group_accumulator;
#pragma unroll
        for(int j = 0; j < 8; ++j)
        {
            group_accumulator[j] = 0;
        }

#pragma unroll
        for(int half = 0; half < 2; ++half)
        {
            int2x32 a_fragment = {0, 0};
            int2x32 b_fragment = {0, 0};
            const int word = group * 4 + half * 2;
            if(source_lane == 0)
            {
                const long base = static_cast<long>(input_row) * words_per_row + word;
                a_fragment[0] = static_cast<int>(a_packed[base]);
                a_fragment[1] = static_cast<int>(a_packed[base + 1]);
            }
            if(source_col < N)
            {
                const long row = static_cast<long>(expert) * N + source_col;
                const long base = row * words_per_row + word;
                b_fragment[0] =
                    static_cast<int>(b_uint4b8[base] ^ kUint4B8ToS4Mask);
                b_fragment[1] =
                    static_cast<int>(b_uint4b8[base + 1] ^ kUint4B8ToS4Mask);
            }
            group_accumulator = __builtin_amdgcn_wmma_i32_16x16x16_iu4_w32(
                true, a_fragment, true, b_fragment, group_accumulator, false);
        }

        if(lane < 16 && source_col < N)
        {
            const long scale_row = static_cast<long>(expert) * N + source_col;
            const float activation_scale =
                a_scale[static_cast<long>(input_row) * groups + group];
            const float weight_scale =
                __bfloat162float(b_scale[scale_row * groups + group]);
            accumulator[0] += static_cast<float>(group_accumulator[0]) *
                              activation_scale * weight_scale;
        }
    }

    if(lane < 16 && source_col < N)
    {
        float value = accumulator[0];
        if constexpr(ApplyRouterWeight)
        {
            value *= topk_weights[route];
        }
        output[static_cast<long>(route) * N + source_col] =
            __float2bfloat16(value);
    }
}

// Sparse-decode specialization for the load-time tiled Ling layout. One wave
// owns one routed expert and two adjacent N16 fragments. It avoids expert
// sorting and its two WMMA products share the activation fragment.
template <bool InputPerToken, bool ApplyRouterWeight, bool WeightS2 = false>
__global__ void routed_s4s4_g32_wmma_n32_tiled(
    const uint32_t* __restrict__ a_packed,
    const uint32_t* __restrict__ b_tiled,
    const float* __restrict__ a_scale,
    const bf16_t* __restrict__ b_scale_tiled,
    const int32_t* __restrict__ topk_ids,
    const float* __restrict__ topk_weights,
    bf16_t* __restrict__ output,
    const int routes,
    const int top_k,
    const int experts,
    const int N,
    const int K)
{
    const int lane = threadIdx.x & 31;
    const int source_lane = lane & 15;
    const int route = blockIdx.y;
    if(route >= routes)
    {
        return;
    }
    const int expert = topk_ids[route];
    if(expert < 0 || expert >= experts)
    {
        return;
    }
    const int input_row = InputPerToken ? route / top_k : route;
    const int groups = K / kGroupSize;
    const int words_per_row = K / 8;
    const int source_col0 = blockIdx.x * 32 + source_lane;
    const int source_col1 = source_col0 + 16;

    float8x32 accumulator0;
    float8x32 accumulator1;
#pragma unroll
    for(int j = 0; j < 8; ++j)
    {
        accumulator0[j] = 0.0f;
        accumulator1[j] = 0.0f;
    }

    for(int group = 0; group < groups; ++group)
    {
        int8x32 group_accumulator0;
        int8x32 group_accumulator1;
#pragma unroll
        for(int j = 0; j < 8; ++j)
        {
            group_accumulator0[j] = 0;
            group_accumulator1[j] = 0;
        }
        float s2_weight_scale0 = 0.0f;
        float s2_weight_scale1 = 0.0f;
        bool s2_positive_tail0 = false;
        bool s2_positive_tail1 = false;
        if constexpr(WeightS2)
        {
            const long scale_base =
                ((static_cast<long>(expert) * (N / 32) + blockIdx.x) * groups +
                 group) *
                32;
            const float raw_scale0 =
                __bfloat162float(b_scale_tiled[scale_base + source_lane]);
            const float raw_scale1 = __bfloat162float(
                b_scale_tiled[scale_base + source_lane + 16]);
            s2_positive_tail0 = raw_scale0 < 0.0f;
            s2_positive_tail1 = raw_scale1 < 0.0f;
            s2_weight_scale0 = fabsf(raw_scale0);
            s2_weight_scale1 = fabsf(raw_scale1);
        }
#pragma unroll
        for(int half = 0; half < 2; ++half)
        {
            int2x32 a_fragment = {0, 0};
            int2x32 b_fragment0 = {0, 0};
            int2x32 b_fragment1 = {0, 0};
            const int word = group * 4 + half * 2;
            if(source_lane == 0)
            {
                const long base =
                    static_cast<long>(input_row) * words_per_row + word;
                a_fragment[0] = static_cast<int>(a_packed[base]);
                a_fragment[1] = static_cast<int>(a_packed[base + 1]);
            }
            if constexpr(WeightS2)
            {
                const long tiled_base =
                    ((((static_cast<long>(expert) * (N / 32) + blockIdx.x) *
                        groups + group) *
                           2 +
                       half) *
                          32 +
                      source_lane);
                b_fragment0 = expand_affine_s2_sixteen(
                    b_tiled[tiled_base], s2_positive_tail0);
                b_fragment1 = expand_affine_s2_sixteen(
                    b_tiled[tiled_base + 16], s2_positive_tail1);
            }
            else
            {
                const long tiled_base =
                    (((((static_cast<long>(expert) * (N / 32) + blockIdx.x) *
                         groups + group) *
                            2 +
                        half) *
                           32 +
                       source_lane) *
                      2);
                b_fragment0[0] =
                    static_cast<int>(b_tiled[tiled_base] ^ kUint4B8ToS4Mask);
                b_fragment0[1] = static_cast<int>(
                    b_tiled[tiled_base + 1] ^ kUint4B8ToS4Mask);
                const long tiled_base1 = tiled_base + 16 * 2;
                b_fragment1[0] =
                    static_cast<int>(b_tiled[tiled_base1] ^ kUint4B8ToS4Mask);
                b_fragment1[1] = static_cast<int>(
                    b_tiled[tiled_base1 + 1] ^ kUint4B8ToS4Mask);
            }
            group_accumulator0 = __builtin_amdgcn_wmma_i32_16x16x16_iu4_w32(
                true, a_fragment, true, b_fragment0, group_accumulator0, false);
            group_accumulator1 = __builtin_amdgcn_wmma_i32_16x16x16_iu4_w32(
                true, a_fragment, true, b_fragment1, group_accumulator1, false);
        }

        if(lane < 16)
        {
            const long scale_base =
                ((static_cast<long>(expert) * (N / 32) + blockIdx.x) * groups +
                 group) *
                32;
            const float activation_scale =
                a_scale[static_cast<long>(input_row) * groups + group];
            const float weight_scale0 = WeightS2
                                            ? s2_weight_scale0
                                            : __bfloat162float(b_scale_tiled[
                                                  scale_base + source_lane]);
            const float weight_scale1 =
                WeightS2
                    ? s2_weight_scale1
                    : __bfloat162float(
                          b_scale_tiled[scale_base + source_lane + 16]);
            accumulator0[0] += static_cast<float>(group_accumulator0[0]) *
                               activation_scale * weight_scale0;
            accumulator1[0] += static_cast<float>(group_accumulator1[0]) *
                               activation_scale * weight_scale1;
        }
    }

    if(lane < 16)
    {
        const float router_weight =
            ApplyRouterWeight ? topk_weights[route] : 1.0f;
        output[static_cast<long>(route) * N + source_col0] =
            __float2bfloat16(accumulator0[0] * router_weight);
        output[static_cast<long>(route) * N + source_col1] =
            __float2bfloat16(accumulator1[0] * router_weight);
    }
}

// Grouped prefill kernel. One wave consumes a complete 16-route expert tile
// and produces 32 output columns. Two N fragments share every activation
// load, while sorted route metadata keeps all 16 WMMA rows useful.
template <bool W2,
          bool TiledWeights,
          bool AsymmetricU4,
          bool Residual2 = false,
          bool Residual3 = false,
          bool Residual4 = false,
          bool Outlier1 = false>
__global__ void grouped_s4s4_g32_wmma_n32(
    const uint32_t* __restrict__ a_packed,
    const uint32_t* __restrict__ b_uint4b8,
    const float* __restrict__ a_scale,
    const uint32_t* __restrict__ a_packed_residual,
    const float* __restrict__ a_scale_residual,
    const uint32_t* __restrict__ a_packed_residual_second,
    const float* __restrict__ a_scale_residual_second,
    const uint32_t* __restrict__ a_packed_residual_third,
    const float* __restrict__ a_scale_residual_third,
    const int32_t* __restrict__ outlier_indices,
    const float* __restrict__ outlier_values,
    const int32_t* __restrict__ a_zero,
    const bf16_t* __restrict__ b_scale,
    const int32_t* __restrict__ sorted_token_ids,
    const int32_t* __restrict__ expert_ids,
    const int32_t* __restrict__ num_tokens_post_padded,
    const float* __restrict__ topk_weights,
    bf16_t* __restrict__ output,
    const int routes,
    const int top_k,
    const int experts,
    const int N,
    const int K)
{
    const int lane = threadIdx.x & 31;
    const int source_lane = lane & 15;
    const int sorted_base = blockIdx.y * 16;
    if(sorted_base >= num_tokens_post_padded[0])
    {
        return;
    }
    const int expert = expert_ids[blockIdx.y];
    if(expert < 0 || expert >= experts)
    {
        return;
    }
    const int tile_n = blockIdx.x * 32;
    const int source_col0 = tile_n + source_lane;
    const int source_col1 = source_col0 + 16;
    const int sorted_source_row = sorted_base + source_lane;
    const int source_route = sorted_token_ids[sorted_source_row];
    const int input_source_row =
        W2 ? sorted_source_row : source_route / top_k;
    const bool source_live = source_route >= 0 && source_route < routes;
    const int words_per_row = K / 8;
    const int groups = K / kGroupSize;

    float8x32 accumulator0;
    float8x32 accumulator1;
#pragma unroll
    for(int j = 0; j < 8; ++j)
    {
        accumulator0[j] = 0.0f;
        accumulator1[j] = 0.0f;
    }

    for(int group = 0; group < groups; ++group)
    {
        int8x32 group_accumulator0;
        int8x32 group_accumulator1;
        int8x32 group_accumulator_residual0;
        int8x32 group_accumulator_residual1;
        int8x32 group_accumulator_residual_second0;
        int8x32 group_accumulator_residual_second1;
        int8x32 group_accumulator_residual_third0;
        int8x32 group_accumulator_residual_third1;
        int weight_sum0 = 0;
        int weight_sum1 = 0;
#pragma unroll
        for(int j = 0; j < 8; ++j)
        {
            group_accumulator0[j] = 0;
            group_accumulator1[j] = 0;
            group_accumulator_residual0[j] = 0;
            group_accumulator_residual1[j] = 0;
            group_accumulator_residual_second0[j] = 0;
            group_accumulator_residual_second1[j] = 0;
            group_accumulator_residual_third0[j] = 0;
            group_accumulator_residual_third1[j] = 0;
        }
#pragma unroll
        for(int half = 0; half < 2; ++half)
        {
            int2x32 a_fragment = {0, 0};
            int2x32 a_fragment_residual = {0, 0};
            int2x32 a_fragment_residual_second = {0, 0};
            int2x32 a_fragment_residual_third = {0, 0};
            int2x32 b_fragment0 = {0, 0};
            int2x32 b_fragment1 = {0, 0};
            const int word = group * 4 + half * 2;
            if(source_live)
            {
                const long base =
                    static_cast<long>(input_source_row) * words_per_row + word;
                a_fragment[0] = static_cast<int>(a_packed[base]);
                a_fragment[1] = static_cast<int>(a_packed[base + 1]);
                if constexpr(Residual2)
                {
                    a_fragment_residual[0] =
                        static_cast<int>(a_packed_residual[base]);
                    a_fragment_residual[1] =
                        static_cast<int>(a_packed_residual[base + 1]);
                }
                if constexpr(Residual3)
                {
                    a_fragment_residual_second[0] =
                        static_cast<int>(a_packed_residual_second[base]);
                    a_fragment_residual_second[1] =
                        static_cast<int>(a_packed_residual_second[base + 1]);
                }
                if constexpr(Residual4)
                {
                    a_fragment_residual_third[0] =
                        static_cast<int>(a_packed_residual_third[base]);
                    a_fragment_residual_third[1] =
                        static_cast<int>(a_packed_residual_third[base + 1]);
                }
            }
            if(source_col0 < N)
            {
                long base;
                if constexpr(TiledWeights)
                {
                    base = (((((static_cast<long>(expert) * (N / 32) + blockIdx.x) *
                                groups + group) *
                                   2 +
                               half) *
                                  32 +
                              source_lane) *
                             2);
                }
                else
                {
                    const long row = static_cast<long>(expert) * N + source_col0;
                    base = row * words_per_row + word;
                }
                b_fragment0[0] =
                    static_cast<int>(b_uint4b8[base] ^ kUint4B8ToS4Mask);
                b_fragment0[1] =
                    static_cast<int>(b_uint4b8[base + 1] ^ kUint4B8ToS4Mask);
                if constexpr(AsymmetricU4)
                {
                    weight_sum0 += sum_s4_word(
                        static_cast<uint32_t>(b_fragment0[0]));
                    weight_sum0 += sum_s4_word(
                        static_cast<uint32_t>(b_fragment0[1]));
                }
            }
            if(source_col1 < N)
            {
                long base;
                if constexpr(TiledWeights)
                {
                    base = (((((static_cast<long>(expert) * (N / 32) + blockIdx.x) *
                                groups + group) *
                                   2 +
                               half) *
                                  32 +
                              source_lane + 16) *
                             2);
                }
                else
                {
                    const long row = static_cast<long>(expert) * N + source_col1;
                    base = row * words_per_row + word;
                }
                b_fragment1[0] =
                    static_cast<int>(b_uint4b8[base] ^ kUint4B8ToS4Mask);
                b_fragment1[1] =
                    static_cast<int>(b_uint4b8[base + 1] ^ kUint4B8ToS4Mask);
                if constexpr(AsymmetricU4)
                {
                    weight_sum1 += sum_s4_word(
                        static_cast<uint32_t>(b_fragment1[0]));
                    weight_sum1 += sum_s4_word(
                        static_cast<uint32_t>(b_fragment1[1]));
                }
            }
            group_accumulator0 = __builtin_amdgcn_wmma_i32_16x16x16_iu4_w32(
                !AsymmetricU4,
                a_fragment,
                true,
                b_fragment0,
                group_accumulator0,
                false);
            group_accumulator1 = __builtin_amdgcn_wmma_i32_16x16x16_iu4_w32(
                !AsymmetricU4,
                a_fragment,
                true,
                b_fragment1,
                group_accumulator1,
                false);
            if constexpr(Residual2)
            {
                group_accumulator_residual0 =
                    __builtin_amdgcn_wmma_i32_16x16x16_iu4_w32(
                        true,
                        a_fragment_residual,
                        true,
                        b_fragment0,
                        group_accumulator_residual0,
                        false);
                group_accumulator_residual1 =
                    __builtin_amdgcn_wmma_i32_16x16x16_iu4_w32(
                        true,
                        a_fragment_residual,
                        true,
                        b_fragment1,
                        group_accumulator_residual1,
                        false);
            }
            if constexpr(Residual3)
            {
                group_accumulator_residual_second0 =
                    __builtin_amdgcn_wmma_i32_16x16x16_iu4_w32(
                        true,
                        a_fragment_residual_second,
                        true,
                        b_fragment0,
                        group_accumulator_residual_second0,
                        false);
                group_accumulator_residual_second1 =
                    __builtin_amdgcn_wmma_i32_16x16x16_iu4_w32(
                        true,
                        a_fragment_residual_second,
                        true,
                        b_fragment1,
                        group_accumulator_residual_second1,
                        false);
            }
            if constexpr(Residual4)
            {
                group_accumulator_residual_third0 =
                    __builtin_amdgcn_wmma_i32_16x16x16_iu4_w32(
                        true,
                        a_fragment_residual_third,
                        true,
                        b_fragment0,
                        group_accumulator_residual_third0,
                        false);
                group_accumulator_residual_third1 =
                    __builtin_amdgcn_wmma_i32_16x16x16_iu4_w32(
                        true,
                        a_fragment_residual_third,
                        true,
                        b_fragment1,
                        group_accumulator_residual_third1,
                        false);
            }
        }

        long scale_index0;
        long scale_index1;
        if constexpr(TiledWeights)
        {
            const long base =
                ((static_cast<long>(expert) * (N / 32) + blockIdx.x) * groups +
                 group) *
                32;
            scale_index0 = base + source_lane;
            scale_index1 = base + source_lane + 16;
        }
        else
        {
            scale_index0 =
                (static_cast<long>(expert) * N + source_col0) * groups + group;
            scale_index1 =
                (static_cast<long>(expert) * N + source_col1) * groups + group;
        }
        const float weight_scale0 =
            source_col0 < N ? __bfloat162float(b_scale[scale_index0]) : 0.0f;
        const float weight_scale1 =
            source_col1 < N ? __bfloat162float(b_scale[scale_index1]) : 0.0f;
#pragma unroll
        for(int j = 0; j < 8; ++j)
        {
            const int sorted_output_row = sorted_base + c_fragment_row(lane, j);
            const int output_route = sorted_token_ids[sorted_output_row];
            if(output_route >= 0 && output_route < routes)
            {
                const int input_output_row =
                    W2 ? sorted_output_row : output_route / top_k;
                const float activation_scale =
                    a_scale[static_cast<long>(input_output_row) * groups + group];
                int dot0 = group_accumulator0[j];
                int dot1 = group_accumulator1[j];
                if constexpr(AsymmetricU4)
                {
                    const int zero = a_zero[
                        static_cast<long>(input_output_row) * groups + group];
                    dot0 -= zero * weight_sum0;
                    dot1 -= zero * weight_sum1;
                }
                accumulator0[j] += static_cast<float>(dot0) *
                                   activation_scale * weight_scale0;
                accumulator1[j] += static_cast<float>(dot1) *
                                   activation_scale * weight_scale1;
                if constexpr(Residual2)
                {
                    const float residual_scale = a_scale_residual[
                        static_cast<long>(input_output_row) * groups + group];
                    accumulator0[j] +=
                        static_cast<float>(group_accumulator_residual0[j]) *
                        residual_scale * weight_scale0;
                    accumulator1[j] +=
                        static_cast<float>(group_accumulator_residual1[j]) *
                        residual_scale * weight_scale1;
                }
                if constexpr(Residual3)
                {
                    const float residual_second_scale =
                        a_scale_residual_second[
                            static_cast<long>(input_output_row) * groups + group];
                    accumulator0[j] +=
                        static_cast<float>(
                            group_accumulator_residual_second0[j]) *
                        residual_second_scale * weight_scale0;
                    accumulator1[j] +=
                        static_cast<float>(
                            group_accumulator_residual_second1[j]) *
                        residual_second_scale * weight_scale1;
                }
                if constexpr(Residual4)
                {
                    const float residual_third_scale = a_scale_residual_third[
                        static_cast<long>(input_output_row) * groups + group];
                    accumulator0[j] +=
                        static_cast<float>(group_accumulator_residual_third0[j]) *
                        residual_third_scale * weight_scale0;
                    accumulator1[j] +=
                        static_cast<float>(group_accumulator_residual_third1[j]) *
                        residual_third_scale * weight_scale1;
                }
                if constexpr(Outlier1)
                {
                    const long activation_scale_index =
                        static_cast<long>(input_output_row) * groups + group;
                    const int outlier_index =
                        outlier_indices[activation_scale_index];
                    const int half = outlier_index / 16;
                    const int word_in_half = (outlier_index & 15) / 8;
                    long packed_index0;
                    long packed_index1;
                    if constexpr(TiledWeights)
                    {
                        const long tile_base =
                            ((((static_cast<long>(expert) * (N / 32) + blockIdx.x) *
                                groups + group) *
                                   2 +
                               half) *
                              32);
                        packed_index0 =
                            (tile_base + source_lane) * 2 + word_in_half;
                        packed_index1 =
                            (tile_base + source_lane + 16) * 2 + word_in_half;
                    }
                    else
                    {
                        packed_index0 =
                            (static_cast<long>(expert) * N + source_col0) *
                                words_per_row +
                            group * 4 + outlier_index / 8;
                        packed_index1 =
                            (static_cast<long>(expert) * N + source_col1) *
                                words_per_row +
                            group * 4 + outlier_index / 8;
                    }
                    const int shift = 4 * (outlier_index & 7);
                    const float outlier_value =
                        outlier_values[activation_scale_index];
                    if(source_col0 < N)
                    {
                        const int weight_code0 = static_cast<int>(
                            (b_uint4b8[packed_index0] >> shift) & 0xF) - 8;
                        accumulator0[j] += outlier_value *
                                           static_cast<float>(weight_code0) *
                                           weight_scale0;
                    }
                    if(source_col1 < N)
                    {
                        const int weight_code1 = static_cast<int>(
                            (b_uint4b8[packed_index1] >> shift) & 0xF) - 8;
                        accumulator1[j] += outlier_value *
                                           static_cast<float>(weight_code1) *
                                           weight_scale1;
                    }
                }
            }
        }
    }

#pragma unroll
    for(int j = 0; j < 8; ++j)
    {
        const int sorted_output_row = sorted_base + c_fragment_row(lane, j);
        const int output_route = sorted_token_ids[sorted_output_row];
        if(output_route >= 0 && output_route < routes)
        {
            const int destination_row = W2 ? output_route : sorted_output_row;
            const float router_weight = W2 ? topk_weights[output_route] : 1.0f;
            if(source_col0 < N)
            {
                output[static_cast<long>(destination_row) * N + source_col0] =
                    __float2bfloat16(accumulator0[j] * router_weight);
            }
            if(source_col1 < N)
            {
                output[static_cast<long>(destination_row) * N + source_col1] =
                    __float2bfloat16(accumulator1[j] * router_weight);
            }
        }
    }
}

// Symmetric S4 activations with one scale per native K16 WMMA half. Official
// Ling weights and their G32 BF16 scales are unchanged.
template <bool W2, bool WeightS2 = false>
__global__ void grouped_s4g16_s4g32_wmma_n32_tiled(
    const uint32_t* __restrict__ a_packed,
    const uint32_t* __restrict__ b_tiled,
    const float* __restrict__ a_scale_g16,
    const bf16_t* __restrict__ b_scale_g32,
    const int32_t* __restrict__ sorted_token_ids,
    const int32_t* __restrict__ expert_ids,
    const int32_t* __restrict__ num_tokens_post_padded,
    const float* __restrict__ topk_weights,
    bf16_t* __restrict__ output,
    const int routes,
    const int top_k,
    const int experts,
    const int N,
    const int K)
{
    const int lane = threadIdx.x & 31;
    const int source_lane = lane & 15;
    const int sorted_base = blockIdx.y * 16;
    if(sorted_base >= num_tokens_post_padded[0])
    {
        return;
    }
    const int expert = expert_ids[blockIdx.y];
    if(expert < 0 || expert >= experts)
    {
        return;
    }
    const int source_col0 = blockIdx.x * 32 + source_lane;
    const int source_col1 = source_col0 + 16;
    const int sorted_source_row = sorted_base + source_lane;
    const int source_route = sorted_token_ids[sorted_source_row];
    const int input_source_row = W2 ? sorted_source_row : source_route / top_k;
    const bool source_live = source_route >= 0 && source_route < routes;
    const int words_per_row = K / 8;
    const int weight_groups = K / 32;
    const int activation_groups = K / 16;

    float8x32 accumulator0;
    float8x32 accumulator1;
#pragma unroll
    for(int j = 0; j < 8; ++j)
    {
        accumulator0[j] = 0.0f;
        accumulator1[j] = 0.0f;
    }

    for(int group = 0; group < weight_groups; ++group)
    {
        const long weight_scale_base =
            ((static_cast<long>(expert) * (N / 32) + blockIdx.x) *
                 weight_groups +
             group) *
            32;
        const float raw_weight_scale0 =
            __bfloat162float(b_scale_g32[weight_scale_base + source_lane]);
        const float raw_weight_scale1 = __bfloat162float(
            b_scale_g32[weight_scale_base + source_lane + 16]);
        const bool s2_positive_tail0 = WeightS2 && raw_weight_scale0 < 0.0f;
        const bool s2_positive_tail1 = WeightS2 && raw_weight_scale1 < 0.0f;
        const float weight_scale0 =
            WeightS2 ? fabsf(raw_weight_scale0) : raw_weight_scale0;
        const float weight_scale1 =
            WeightS2 ? fabsf(raw_weight_scale1) : raw_weight_scale1;

#pragma unroll
        for(int half = 0; half < 2; ++half)
        {
            int2x32 a_fragment = {0, 0};
            int2x32 b_fragment0 = {0, 0};
            int2x32 b_fragment1 = {0, 0};
            int8x32 half_accumulator0;
            int8x32 half_accumulator1;
#pragma unroll
            for(int j = 0; j < 8; ++j)
            {
                half_accumulator0[j] = 0;
                half_accumulator1[j] = 0;
            }
            const int word = group * 4 + half * 2;
            if(source_live)
            {
                const long base =
                    static_cast<long>(input_source_row) * words_per_row + word;
                a_fragment[0] = static_cast<int>(a_packed[base]);
                a_fragment[1] = static_cast<int>(a_packed[base + 1]);
            }
            if constexpr(WeightS2)
            {
                const long tiled_base =
                    ((((static_cast<long>(expert) * (N / 32) + blockIdx.x) *
                        weight_groups + group) *
                           2 +
                       half) *
                          32 +
                      source_lane);
                b_fragment0 = expand_affine_s2_sixteen(
                    b_tiled[tiled_base], s2_positive_tail0);
                b_fragment1 = expand_affine_s2_sixteen(
                    b_tiled[tiled_base + 16], s2_positive_tail1);
            }
            else
            {
                const long tiled_base =
                    (((((static_cast<long>(expert) * (N / 32) + blockIdx.x) *
                         weight_groups + group) *
                            2 +
                        half) *
                           32 +
                       source_lane) *
                      2);
                b_fragment0[0] =
                    static_cast<int>(b_tiled[tiled_base] ^ kUint4B8ToS4Mask);
                b_fragment0[1] = static_cast<int>(
                    b_tiled[tiled_base + 1] ^ kUint4B8ToS4Mask);
                const long tiled_base1 = tiled_base + 16 * 2;
                b_fragment1[0] =
                    static_cast<int>(b_tiled[tiled_base1] ^ kUint4B8ToS4Mask);
                b_fragment1[1] = static_cast<int>(
                    b_tiled[tiled_base1 + 1] ^ kUint4B8ToS4Mask);
            }
            half_accumulator0 = __builtin_amdgcn_wmma_i32_16x16x16_iu4_w32(
                true, a_fragment, true, b_fragment0, half_accumulator0, false);
            half_accumulator1 = __builtin_amdgcn_wmma_i32_16x16x16_iu4_w32(
                true, a_fragment, true, b_fragment1, half_accumulator1, false);

#pragma unroll
            for(int j = 0; j < 8; ++j)
            {
                const int sorted_output_row =
                    sorted_base + c_fragment_row(lane, j);
                const int output_route = sorted_token_ids[sorted_output_row];
                if(output_route >= 0 && output_route < routes)
                {
                    const int input_output_row =
                        W2 ? sorted_output_row : output_route / top_k;
                    const float activation_scale = a_scale_g16[
                        static_cast<long>(input_output_row) * activation_groups +
                        group * 2 + half];
                    accumulator0[j] +=
                        static_cast<float>(half_accumulator0[j]) *
                        activation_scale * weight_scale0;
                    accumulator1[j] +=
                        static_cast<float>(half_accumulator1[j]) *
                        activation_scale * weight_scale1;
                }
            }
        }
    }

#pragma unroll
    for(int j = 0; j < 8; ++j)
    {
        const int sorted_output_row = sorted_base + c_fragment_row(lane, j);
        const int output_route = sorted_token_ids[sorted_output_row];
        if(output_route >= 0 && output_route < routes)
        {
            const int destination_row = W2 ? output_route : sorted_output_row;
            const float router_weight = W2 ? topk_weights[output_route] : 1.0f;
            output[static_cast<long>(destination_row) * N + source_col0] =
                __float2bfloat16(accumulator0[j] * router_weight);
            output[static_cast<long>(destination_row) * N + source_col1] =
                __float2bfloat16(accumulator1[j] * router_weight);
        }
    }
}

// Tiled W13 specialization. Four waves share one transposed 16xK activation
// tile and its G32 scales in LDS, then each wave computes a distinct N32
// fragment. This removes the repeated activation traffic across four adjacent
// output tiles while preserving the exact WMMA and BF16 accumulation order.
__global__ void grouped_w13_s4s4_g32_wmma_n128_shared(
    const uint32_t* __restrict__ a_packed,
    const uint32_t* __restrict__ b_tiled,
    const float* __restrict__ a_scale,
    const bf16_t* __restrict__ b_scale_tiled,
    const int32_t* __restrict__ sorted_token_ids,
    const int32_t* __restrict__ expert_ids,
    const int32_t* __restrict__ num_tokens_post_padded,
    bf16_t* __restrict__ output,
    const int routes,
    const int top_k,
    const int experts,
    const int N,
    const int K)
{
    constexpr int tile_rows = 16;
    constexpr int words_per_row = 2560 / 8;
    constexpr int groups = 2560 / kGroupSize;
    __shared__ uint32_t shared_packed[words_per_row * tile_rows];
    __shared__ float shared_scale[groups * tile_rows];
    __shared__ int32_t shared_routes[tile_rows];
    __shared__ int32_t shared_input_rows[tile_rows];

    const int thread = threadIdx.x;
    const int wave = thread >> 5;
    const int lane = thread & 31;
    const int source_lane = lane & 15;
    const int sorted_base = blockIdx.y * tile_rows;
    if(sorted_base >= num_tokens_post_padded[0])
    {
        return;
    }
    const int expert = expert_ids[blockIdx.y];
    if(expert < 0 || expert >= experts)
    {
        return;
    }
    if(thread < tile_rows)
    {
        const int route = sorted_token_ids[sorted_base + thread];
        shared_routes[thread] = route;
        shared_input_rows[thread] =
            route >= 0 && route < routes ? route / top_k : -1;
    }
    __syncthreads();

    for(int index = thread; index < words_per_row * tile_rows;
        index += blockDim.x)
    {
        const int word = index / tile_rows;
        const int row = index - word * tile_rows;
        const int input_row = shared_input_rows[row];
        shared_packed[index] =
            input_row >= 0
                ? a_packed[static_cast<long>(input_row) * words_per_row + word]
                : 0u;
    }
    for(int index = thread; index < groups * tile_rows; index += blockDim.x)
    {
        const int group = index / tile_rows;
        const int row = index - group * tile_rows;
        const int input_row = shared_input_rows[row];
        shared_scale[index] =
            input_row >= 0
                ? a_scale[static_cast<long>(input_row) * groups + group]
                : 0.0f;
    }
    __syncthreads();

    const int n_tile = blockIdx.x * 4 + wave;
    const int tile_n = n_tile * 32;
    const int source_col0 = tile_n + source_lane;
    const int source_col1 = source_col0 + 16;
    float8x32 accumulator0;
    float8x32 accumulator1;
#pragma unroll
    for(int j = 0; j < 8; ++j)
    {
        accumulator0[j] = 0.0f;
        accumulator1[j] = 0.0f;
    }

    for(int group = 0; group < groups; ++group)
    {
        int8x32 group_accumulator0;
        int8x32 group_accumulator1;
#pragma unroll
        for(int j = 0; j < 8; ++j)
        {
            group_accumulator0[j] = 0;
            group_accumulator1[j] = 0;
        }
#pragma unroll
        for(int half = 0; half < 2; ++half)
        {
            const int word = group * 4 + half * 2;
            int2x32 a_fragment = {
                static_cast<int>(shared_packed[word * tile_rows + source_lane]),
                static_cast<int>(
                    shared_packed[(word + 1) * tile_rows + source_lane])};
            int2x32 b_fragment0 = {0, 0};
            int2x32 b_fragment1 = {0, 0};
            if(source_col0 < N)
            {
                const long base =
                    (((((static_cast<long>(expert) * (N / 32) + n_tile) * groups +
                         group) *
                            2 +
                        half) *
                           32 +
                       source_lane) *
                      2);
                b_fragment0[0] =
                    static_cast<int>(b_tiled[base] ^ kUint4B8ToS4Mask);
                b_fragment0[1] =
                    static_cast<int>(b_tiled[base + 1] ^ kUint4B8ToS4Mask);
            }
            if(source_col1 < N)
            {
                const long base =
                    (((((static_cast<long>(expert) * (N / 32) + n_tile) * groups +
                         group) *
                            2 +
                        half) *
                           32 +
                       source_lane + 16) *
                      2);
                b_fragment1[0] =
                    static_cast<int>(b_tiled[base] ^ kUint4B8ToS4Mask);
                b_fragment1[1] =
                    static_cast<int>(b_tiled[base + 1] ^ kUint4B8ToS4Mask);
            }
            group_accumulator0 = __builtin_amdgcn_wmma_i32_16x16x16_iu4_w32(
                true, a_fragment, true, b_fragment0, group_accumulator0, false);
            group_accumulator1 = __builtin_amdgcn_wmma_i32_16x16x16_iu4_w32(
                true, a_fragment, true, b_fragment1, group_accumulator1, false);
        }

        const long scale_base =
            ((static_cast<long>(expert) * (N / 32) + n_tile) * groups + group) *
            32;
        const float weight_scale0 =
            source_col0 < N
                ? __bfloat162float(b_scale_tiled[scale_base + source_lane])
                : 0.0f;
        const float weight_scale1 =
            source_col1 < N
                ? __bfloat162float(b_scale_tiled[scale_base + source_lane + 16])
                : 0.0f;
#pragma unroll
        for(int j = 0; j < 8; ++j)
        {
            const int row = c_fragment_row(lane, j);
            const float activation_scale = shared_scale[group * tile_rows + row];
            accumulator0[j] += static_cast<float>(group_accumulator0[j]) *
                               activation_scale * weight_scale0;
            accumulator1[j] += static_cast<float>(group_accumulator1[j]) *
                               activation_scale * weight_scale1;
        }
    }

#pragma unroll
    for(int j = 0; j < 8; ++j)
    {
        const int row = c_fragment_row(lane, j);
        const int route = shared_routes[row];
        if(route >= 0 && route < routes)
        {
            const int destination_row = sorted_base + row;
            if(source_col0 < N)
            {
                output[static_cast<long>(destination_row) * N + source_col0] =
                    __float2bfloat16(accumulator0[j]);
            }
            if(source_col1 < N)
            {
                output[static_cast<long>(destination_row) * N + source_col1] =
                    __float2bfloat16(accumulator1[j]);
            }
        }
    }
}

__global__ void grouped_silu_and_mul_bf16(
    const bf16_t* __restrict__ gate_up,
    bf16_t* __restrict__ output,
    const int32_t* __restrict__ num_tokens_post_padded,
    const int capacity,
    const int intermediate,
    const float clamp_limit)
{
    const long index = static_cast<long>(blockIdx.x) * blockDim.x + threadIdx.x;
    const long count = static_cast<long>(capacity) * intermediate;
    if(index >= count)
    {
        return;
    }
    const int row = static_cast<int>(index / intermediate);
    if(row >= num_tokens_post_padded[0])
    {
        output[index] = __float2bfloat16(0.0f);
        return;
    }
    const int col = static_cast<int>(index - static_cast<long>(row) * intermediate);
    float gate =
        __bfloat162float(gate_up[static_cast<long>(row) * (2 * intermediate) + col]);
    float up = __bfloat162float(
        gate_up[static_cast<long>(row) * (2 * intermediate) + intermediate + col]);
    if(clamp_limit > 0.0f)
    {
        gate = fminf(gate, clamp_limit);
        up = fminf(fmaxf(up, -clamp_limit), clamp_limit);
    }
    const float activated = gate / (1.0f + expf(-gate));
    const bf16_t activated_bf16 = __float2bfloat16(activated);
    output[index] = __float2bfloat16(__bfloat162float(activated_bf16) * up);
}

// W2 decode specialization: one Wave32 owns one token/output tile and walks
// its eight routed experts. This keeps the same per-route BF16 rounding as
// the unfused path while eliminating seven waves, routed-output traffic, and
// the separate reduction launch for each output tile.
template <int ResidualPlanes, bool ParallelRoutes = false, bool Outlier1 = false>
__global__ void routed_w2_topk_s4s4_g32_wmma(
    const uint32_t* __restrict__ a_packed,
    const uint32_t* __restrict__ b_uint4b8,
    const float* __restrict__ a_scale,
    const uint32_t* __restrict__ a_packed_residual,
    const float* __restrict__ a_scale_residual,
    const uint32_t* __restrict__ a_packed_residual_second,
    const float* __restrict__ a_scale_residual_second,
    const uint32_t* __restrict__ a_packed_residual_third,
    const float* __restrict__ a_scale_residual_third,
    const int32_t* __restrict__ outlier_indices,
    const float* __restrict__ outlier_values,
    const bf16_t* __restrict__ b_scale,
    const int32_t* __restrict__ topk_ids,
    const float* __restrict__ topk_weights,
    bf16_t* __restrict__ output,
    const int tokens,
    const int experts,
    const int N,
    const int K)
{
    const int lane = threadIdx.x & 31;
    const int source_lane = lane & 15;
    const int grid_row = blockIdx.y;
    const int token = ParallelRoutes ? grid_row / 8 : grid_row;
    const int tile_n = blockIdx.x * 16;
    const int source_col = tile_n + source_lane;
    if(token >= tokens)
    {
        return;
    }
    const int words_per_row = K / 8;
    const int groups = K / kGroupSize;
    float total = 0.0f;

    constexpr int RouteIterations = ParallelRoutes ? 1 : 8;
#pragma unroll
    for(int route_iteration = 0; route_iteration < RouteIterations;
        ++route_iteration)
    {
        const int topk = ParallelRoutes ? grid_row % 8 : route_iteration;
        const int route = token * 8 + topk;
        const int expert = topk_ids[route];
        if(expert < 0 || expert >= experts)
        {
            continue;
        }
        float route_accumulator = 0.0f;
        for(int group = 0; group < groups; ++group)
        {
            int8x32 group_accumulator;
            int8x32 group_accumulator_residual;
            int8x32 group_accumulator_residual_second;
            int8x32 group_accumulator_residual_third;
#pragma unroll
            for(int j = 0; j < 8; ++j)
            {
                group_accumulator[j] = 0;
                group_accumulator_residual[j] = 0;
                group_accumulator_residual_second[j] = 0;
                group_accumulator_residual_third[j] = 0;
            }
#pragma unroll
            for(int half = 0; half < 2; ++half)
            {
                int2x32 a_fragment = {0, 0};
                int2x32 a_fragment_residual = {0, 0};
                int2x32 a_fragment_residual_second = {0, 0};
                int2x32 a_fragment_residual_third = {0, 0};
                int2x32 b_fragment = {0, 0};
                const int word = group * 4 + half * 2;
                if(source_lane == 0)
                {
                    const long base = static_cast<long>(route) * words_per_row + word;
                    a_fragment[0] = static_cast<int>(a_packed[base]);
                    a_fragment[1] = static_cast<int>(a_packed[base + 1]);
                    if constexpr(ResidualPlanes >= 2)
                    {
                        a_fragment_residual[0] =
                            static_cast<int>(a_packed_residual[base]);
                        a_fragment_residual[1] =
                            static_cast<int>(a_packed_residual[base + 1]);
                        if constexpr(ResidualPlanes >= 3)
                        {
                            a_fragment_residual_second[0] =
                                static_cast<int>(a_packed_residual_second[base]);
                            a_fragment_residual_second[1] =
                                static_cast<int>(a_packed_residual_second[base + 1]);
                        }
                        if constexpr(ResidualPlanes >= 4)
                        {
                            a_fragment_residual_third[0] =
                                static_cast<int>(a_packed_residual_third[base]);
                            a_fragment_residual_third[1] =
                                static_cast<int>(a_packed_residual_third[base + 1]);
                        }
                    }
                }
                if(source_col < N)
                {
                    const long row = static_cast<long>(expert) * N + source_col;
                    const long base = row * words_per_row + word;
                    b_fragment[0] =
                        static_cast<int>(b_uint4b8[base] ^ kUint4B8ToS4Mask);
                    b_fragment[1] =
                        static_cast<int>(b_uint4b8[base + 1] ^ kUint4B8ToS4Mask);
                }
                group_accumulator = __builtin_amdgcn_wmma_i32_16x16x16_iu4_w32(
                    true, a_fragment, true, b_fragment, group_accumulator, false);
                if constexpr(ResidualPlanes >= 2)
                {
                    group_accumulator_residual =
                        __builtin_amdgcn_wmma_i32_16x16x16_iu4_w32(
                            true,
                            a_fragment_residual,
                            true,
                            b_fragment,
                            group_accumulator_residual,
                            false);
                    if constexpr(ResidualPlanes >= 3)
                    {
                        group_accumulator_residual_second =
                            __builtin_amdgcn_wmma_i32_16x16x16_iu4_w32(
                                true,
                                a_fragment_residual_second,
                                true,
                                b_fragment,
                                group_accumulator_residual_second,
                                false);
                    }
                    if constexpr(ResidualPlanes >= 4)
                    {
                        group_accumulator_residual_third =
                            __builtin_amdgcn_wmma_i32_16x16x16_iu4_w32(
                                true,
                                a_fragment_residual_third,
                                true,
                                b_fragment,
                                group_accumulator_residual_third,
                                false);
                    }
                }
            }
            if(lane < 16 && source_col < N)
            {
                const long scale_row = static_cast<long>(expert) * N + source_col;
                const long activation_scale_index =
                    static_cast<long>(route) * groups + group;
                float activation_dot =
                    static_cast<float>(group_accumulator[0]) *
                    a_scale[activation_scale_index];
                if constexpr(ResidualPlanes >= 2)
                {
                    activation_dot +=
                        static_cast<float>(group_accumulator_residual[0]) *
                        a_scale_residual[activation_scale_index];
                    if constexpr(ResidualPlanes >= 3)
                    {
                        activation_dot +=
                            static_cast<float>(
                                group_accumulator_residual_second[0]) *
                            a_scale_residual_second[activation_scale_index];
                    }
                    if constexpr(ResidualPlanes >= 4)
                    {
                        activation_dot +=
                            static_cast<float>(
                                group_accumulator_residual_third[0]) *
                            a_scale_residual_third[activation_scale_index];
                    }
                }
                if constexpr(Outlier1)
                {
                    const int outlier_index =
                        outlier_indices[activation_scale_index];
                    const long weight_row =
                        static_cast<long>(expert) * N + source_col;
                    const long packed_index =
                        weight_row * words_per_row + group * 4 +
                        outlier_index / 8;
                    const uint32_t packed_word = b_uint4b8[packed_index];
                    const int shift = 4 * (outlier_index & 7);
                    const int weight_code =
                        static_cast<int>((packed_word >> shift) & 0xF) - 8;
                    activation_dot +=
                        outlier_values[activation_scale_index] *
                        static_cast<float>(weight_code);
                }
                route_accumulator += activation_dot *
                    __bfloat162float(b_scale[scale_row * groups + group]);
            }
        }
        if(lane < 16 && source_col < N)
        {
            const bf16_t rounded_route =
                __float2bfloat16(route_accumulator * topk_weights[route]);
            if constexpr(ParallelRoutes)
            {
                output[static_cast<long>(route) * N + source_col] =
                    rounded_route;
            }
            else
            {
                total += __bfloat162float(rounded_route);
            }
        }
    }
    if constexpr(!ParallelRoutes)
    {
        if(lane < 16 && source_col < N)
        {
            output[static_cast<long>(token) * N + source_col] =
                __float2bfloat16(total);
        }
    }
}

__global__ void reduce_topk_bf16(const bf16_t* __restrict__ routed,
                                 bf16_t* __restrict__ output,
                                 const int tokens,
                                 const int top_k,
                                 const int hidden)
{
    const long index = static_cast<long>(blockIdx.x) * blockDim.x + threadIdx.x;
    const long count = static_cast<long>(tokens) * hidden;
    if(index >= count)
    {
        return;
    }
    const int token = static_cast<int>(index / hidden);
    const int column = static_cast<int>(index - static_cast<long>(token) * hidden);
    float sum = 0.0f;
#pragma unroll
    for(int route = 0; route < 8; ++route)
    {
        if(route < top_k)
        {
            const long routed_index =
                (static_cast<long>(token) * top_k + route) * hidden + column;
            sum += __bfloat162float(routed[routed_index]);
        }
    }
    output[index] = __float2bfloat16(sum);
}

void check_decode_inputs(const torch::Tensor& input,
                         const torch::Tensor& w13_packed,
                         const torch::Tensor& w13_scale,
                         const torch::Tensor& w2_packed,
                         const torch::Tensor& w2_scale,
                         const torch::Tensor& topk_ids,
                         const torch::Tensor& topk_weights)
{
    TORCH_CHECK(input.is_cuda() && w13_packed.is_cuda() && w13_scale.is_cuda() &&
                    w2_packed.is_cuda() && w2_scale.is_cuda() && topk_ids.is_cuda() &&
                    topk_weights.is_cuda(),
                "all tensors must be CUDA/HIP tensors");
    TORCH_CHECK(input.scalar_type() == torch::kBFloat16, "input must be BF16");
    TORCH_CHECK(w13_packed.scalar_type() == torch::kInt32 &&
                    w2_packed.scalar_type() == torch::kInt32,
                "packed weights must be int32 views");
    TORCH_CHECK(w13_scale.scalar_type() == torch::kBFloat16 &&
                    w2_scale.scalar_type() == torch::kBFloat16,
                "weight scales must be BF16");
    TORCH_CHECK(topk_ids.scalar_type() == torch::kInt32,
                "topk_ids must be int32");
    TORCH_CHECK(topk_weights.scalar_type() == torch::kFloat32,
                "topk_weights must be FP32");
    TORCH_CHECK(input.dim() == 2 && input.size(1) == 2560,
                "input must be [M,2560]");
    TORCH_CHECK(input.size(0) >= 1, "input must contain at least one row");
    TORCH_CHECK(w13_packed.sizes() == torch::IntArrayRef({512, 1536, 320}),
                "w13 must be [512,1536,320] int32");
    TORCH_CHECK(w13_scale.sizes() == torch::IntArrayRef({512, 1536, 80}),
                "w13 scales must be [512,1536,80]");
    TORCH_CHECK(w2_packed.sizes() == torch::IntArrayRef({512, 2560, 96}),
                "w2 must be [512,2560,96] int32");
    TORCH_CHECK(w2_scale.sizes() == torch::IntArrayRef({512, 2560, 24}),
                "w2 scales must be [512,2560,24]");
    TORCH_CHECK(topk_ids.sizes() ==
                    torch::IntArrayRef({input.size(0), 8}),
                "topk_ids must be [M,8]");
    TORCH_CHECK(topk_weights.sizes() == topk_ids.sizes(),
                "topk_weights shape mismatch");
    TORCH_CHECK(input.is_contiguous() && w13_packed.is_contiguous() &&
                    w13_scale.is_contiguous() && w2_packed.is_contiguous() &&
                    w2_scale.is_contiguous() && topk_ids.is_contiguous() &&
                    topk_weights.is_contiguous(),
                "all tensors must be contiguous");
}

torch::Tensor moe(torch::Tensor input,
                  torch::Tensor w13_packed,
                  torch::Tensor w13_scale,
                  torch::Tensor w2_packed,
                  torch::Tensor w2_scale,
                  torch::Tensor topk_ids,
                  torch::Tensor topk_weights)
{
    static std::once_flag dispatch_marker;
    std::call_once(dispatch_marker, []() {
        std::fprintf(stderr, "LING_IU4_DECODE_DISPATCH\n");
        std::fflush(stderr);
    });
    check_decode_inputs(input,
                        w13_packed,
                        w13_scale,
                        w2_packed,
                        w2_scale,
                        topk_ids,
                        topk_weights);
    TORCH_CHECK(input.size(0) <= 3, "decode operator supports M1-M3 only");
    const at::cuda::OptionalCUDAGuard device_guard(input.device());
    const at::cuda::CUDAStream stream = at::cuda::getCurrentCUDAStream();
    const int tokens = static_cast<int>(input.size(0));
    constexpr int top_k = 8;
    const int routes = tokens * top_k;

    torch::Tensor input_packed = torch::empty(
        {tokens, 320}, input.options().dtype(torch::kInt32));
    torch::Tensor input_scale = torch::empty(
        {tokens, 80}, input.options().dtype(torch::kFloat32));
    torch::Tensor gate_up = torch::empty({routes, 1536}, input.options());
    torch::Tensor hidden = torch::empty({routes, 768}, input.options());
    torch::Tensor hidden_packed = torch::empty(
        {routes, 96}, input.options().dtype(torch::kInt32));
    torch::Tensor hidden_scale = torch::empty(
        {routes, 24}, input.options().dtype(torch::kFloat32));
    torch::Tensor routed_output = torch::empty({routes, 2560}, input.options());
    torch::Tensor output = torch::empty({tokens, 2560}, input.options());

    launch_quantize_with_clip(
        input, input_packed, input_scale, stream, w13_clip_ratio());
    const dim3 w13_grid(1536 / 16, routes);
    routed_s4s4_g32_wmma<true, false><<<w13_grid, 32, 0, stream>>>(
        reinterpret_cast<const uint32_t*>(input_packed.data_ptr<int32_t>()),
        reinterpret_cast<const uint32_t*>(w13_packed.data_ptr<int32_t>()),
        input_scale.data_ptr<float>(),
        reinterpret_cast<const bf16_t*>(w13_scale.data_ptr<at::BFloat16>()),
        topk_ids.data_ptr<int32_t>(),
        topk_weights.data_ptr<float>(),
        reinterpret_cast<bf16_t*>(gate_up.data_ptr<at::BFloat16>()),
        routes,
        top_k,
        512,
        1536,
        2560);
    ling_iu4::check_launch("decode IU4 W13 launch failed");

    constexpr int threads = 256;
    const long activation_count = static_cast<long>(routes) * 768;
    const int activation_blocks =
        static_cast<int>((activation_count + threads - 1) / threads);
    silu_and_mul_bf16<<<activation_blocks, threads, 0, stream>>>(
        reinterpret_cast<const bf16_t*>(gate_up.data_ptr<at::BFloat16>()),
        reinterpret_cast<bf16_t*>(hidden.data_ptr<at::BFloat16>()),
        routes,
        768);
    ling_iu4::check_launch("decode IU4 SiLU launch failed");

    launch_quantize_with_clip(
        hidden, hidden_packed, hidden_scale, stream, w2_clip_ratio());
    const dim3 w2_grid(2560 / 16, routes);
    routed_s4s4_g32_wmma<false, true><<<w2_grid, 32, 0, stream>>>(
        reinterpret_cast<const uint32_t*>(hidden_packed.data_ptr<int32_t>()),
        reinterpret_cast<const uint32_t*>(w2_packed.data_ptr<int32_t>()),
        hidden_scale.data_ptr<float>(),
        reinterpret_cast<const bf16_t*>(w2_scale.data_ptr<at::BFloat16>()),
        topk_ids.data_ptr<int32_t>(),
        topk_weights.data_ptr<float>(),
        reinterpret_cast<bf16_t*>(routed_output.data_ptr<at::BFloat16>()),
        routes,
        top_k,
        512,
        2560,
        768);
    ling_iu4::check_launch("decode IU4 W2 launch failed");

    const long output_count = static_cast<long>(tokens) * 2560;
    const int output_blocks = static_cast<int>((output_count + threads - 1) / threads);
    reduce_topk_bf16<<<output_blocks, threads, 0, stream>>>(
        reinterpret_cast<const bf16_t*>(routed_output.data_ptr<at::BFloat16>()),
        reinterpret_cast<bf16_t*>(output.data_ptr<at::BFloat16>()),
        tokens,
        top_k,
        2560);
    ling_iu4::check_launch("decode IU4 reduction launch failed");
    return output;
}

template <bool WeightS2>
torch::Tensor moe_tiled_impl(torch::Tensor input,
                             torch::Tensor w13_packed,
                             torch::Tensor w13_scale,
                             torch::Tensor w2_packed,
                             torch::Tensor w2_scale,
                             torch::Tensor topk_ids,
                             torch::Tensor topk_weights,
                             c10::optional<torch::Tensor> output_arg = c10::nullopt)
{
    static std::once_flag dispatch_marker;
    std::call_once(dispatch_marker, []() {
        std::fprintf(stderr,
                     WeightS2 ? "LING_IU4_AFFINE_S2_W13_DECODE_DISPATCH\n"
                              : "LING_IU4_TILED_DECODE_DISPATCH\n");
        std::fflush(stderr);
    });
    check_decode_inputs(input,
                        w13_packed,
                        w13_scale,
                        w2_packed,
                        w2_scale,
                        topk_ids,
                        topk_weights);
    TORCH_CHECK(input.size(0) <= 3, "tiled decode supports M1-M3 only");
    const at::cuda::OptionalCUDAGuard device_guard(input.device());
    const at::cuda::CUDAStream stream = at::cuda::getCurrentCUDAStream();
    const int tokens = static_cast<int>(input.size(0));
    constexpr int top_k = 8;
    const int routes = tokens * top_k;

    torch::Tensor input_packed = torch::empty(
        {tokens, 320}, input.options().dtype(torch::kInt32));
    torch::Tensor input_scale = torch::empty(
        {tokens, 80}, input.options().dtype(torch::kFloat32));
    torch::Tensor gate_up = torch::empty({routes, 1536}, input.options());
    torch::Tensor hidden = torch::empty({routes, 768}, input.options());
    torch::Tensor hidden_packed = torch::empty(
        {routes, 96}, input.options().dtype(torch::kInt32));
    torch::Tensor hidden_scale = torch::empty(
        {routes, 24}, input.options().dtype(torch::kFloat32));
    torch::Tensor routed_output = torch::empty({routes, 2560}, input.options());
    torch::Tensor output = output_arg.has_value()
        ? *output_arg
        : torch::empty({tokens, 2560}, input.options());
    TORCH_CHECK(output.is_cuda() && output.scalar_type() == torch::kBFloat16 &&
                    output.sizes() == torch::IntArrayRef({tokens, 2560}) &&
                    output.is_contiguous(),
                "output must be contiguous BF16 [M,2560] on the GPU");

    launch_quantize_with_clip(
        input, input_packed, input_scale, stream, w13_clip_ratio());
    const dim3 w13_grid(1536 / 32, routes);
    routed_s4s4_g32_wmma_n32_tiled<true, false, WeightS2>
        <<<w13_grid, 32, 0, stream>>>(
        reinterpret_cast<const uint32_t*>(input_packed.data_ptr<int32_t>()),
        reinterpret_cast<const uint32_t*>(w13_packed.data_ptr<int32_t>()),
        input_scale.data_ptr<float>(),
        reinterpret_cast<const bf16_t*>(w13_scale.data_ptr<at::BFloat16>()),
        topk_ids.data_ptr<int32_t>(),
        topk_weights.data_ptr<float>(),
        reinterpret_cast<bf16_t*>(gate_up.data_ptr<at::BFloat16>()),
        routes,
        top_k,
        512,
        1536,
        2560);
    ling_iu4::check_launch("tiled decode IU4 W13 launch failed");

    constexpr int threads = 256;
    const long activation_count = static_cast<long>(routes) * 768;
    const int activation_blocks =
        static_cast<int>((activation_count + threads - 1) / threads);
    silu_and_mul_bf16<<<activation_blocks, threads, 0, stream>>>(
        reinterpret_cast<const bf16_t*>(gate_up.data_ptr<at::BFloat16>()),
        reinterpret_cast<bf16_t*>(hidden.data_ptr<at::BFloat16>()),
        routes,
        768);
    ling_iu4::check_launch("tiled decode IU4 SiLU launch failed");

    launch_quantize_with_clip(
        hidden, hidden_packed, hidden_scale, stream, w2_clip_ratio());
    const dim3 w2_grid(2560 / 32, routes);
    routed_s4s4_g32_wmma_n32_tiled<false, true, false>
        <<<w2_grid, 32, 0, stream>>>(
        reinterpret_cast<const uint32_t*>(hidden_packed.data_ptr<int32_t>()),
        reinterpret_cast<const uint32_t*>(w2_packed.data_ptr<int32_t>()),
        hidden_scale.data_ptr<float>(),
        reinterpret_cast<const bf16_t*>(w2_scale.data_ptr<at::BFloat16>()),
        topk_ids.data_ptr<int32_t>(),
        topk_weights.data_ptr<float>(),
        reinterpret_cast<bf16_t*>(routed_output.data_ptr<at::BFloat16>()),
        routes,
        top_k,
        512,
        2560,
        768);
    ling_iu4::check_launch("tiled decode IU4 W2 launch failed");

    const long output_count = static_cast<long>(tokens) * 2560;
    const int output_blocks =
        static_cast<int>((output_count + threads - 1) / threads);
    reduce_topk_bf16<<<output_blocks, threads, 0, stream>>>(
        reinterpret_cast<const bf16_t*>(routed_output.data_ptr<at::BFloat16>()),
        reinterpret_cast<bf16_t*>(output.data_ptr<at::BFloat16>()),
        tokens,
        top_k,
        2560);
    ling_iu4::check_launch("tiled decode IU4 reduction launch failed");
    return output;
}

torch::Tensor moe_tiled(torch::Tensor input,
                        torch::Tensor w13_packed,
                        torch::Tensor w13_scale,
                        torch::Tensor w2_packed,
                        torch::Tensor w2_scale,
                        torch::Tensor topk_ids,
                        torch::Tensor topk_weights)
{
    return moe_tiled_impl<false>(input,
                                 w13_packed,
                                 w13_scale,
                                 w2_packed,
                                 w2_scale,
                                 topk_ids,
                                 topk_weights);
}

torch::Tensor moe_tiled_out(torch::Tensor input,
                            torch::Tensor w13_packed,
                            torch::Tensor w13_scale,
                            torch::Tensor w2_packed,
                            torch::Tensor w2_scale,
                            torch::Tensor topk_ids,
                            torch::Tensor topk_weights,
                            torch::Tensor output)
{
    return moe_tiled_impl<false>(input,
                                 w13_packed,
                                 w13_scale,
                                 w2_packed,
                                 w2_scale,
                                 topk_ids,
                                 topk_weights,
                                 output);
}

torch::Tensor moe_tiled_s2w13(torch::Tensor input,
                              torch::Tensor w13_packed,
                              torch::Tensor w13_scale,
                              torch::Tensor w2_packed,
                              torch::Tensor w2_scale,
                              torch::Tensor topk_ids,
                              torch::Tensor topk_weights)
{
    return moe_tiled_impl<true>(input,
                                w13_packed,
                                w13_scale,
                                w2_packed,
                                w2_scale,
                                topk_ids,
                                topk_weights);
}

template <int ResidualPlanes, bool Outlier1 = false>
torch::Tensor w2_impl(torch::Tensor hidden,
                      torch::Tensor w2_packed,
                      torch::Tensor w2_scale,
                      torch::Tensor topk_ids,
                      torch::Tensor topk_weights)
{
    static std::once_flag dispatch_marker;
    std::call_once(dispatch_marker, []() {
        std::fprintf(stderr,
                     Outlier1
                         ? "LING_IU4_W2_RESIDUAL3_OUTLIER1_DISPATCH\n"
                         : ResidualPlanes == 4
                         ? "LING_IU4_W2_RESIDUAL4_DISPATCH\n"
                         : ResidualPlanes == 3
                         ? "LING_IU4_W2_RESIDUAL3_DISPATCH\n"
                         : ResidualPlanes == 2
                         ? "LING_IU4_W2_RESIDUAL2_DISPATCH\n"
                         : "LING_IU4_W2_DISPATCH\n");
        std::fflush(stderr);
    });
    TORCH_CHECK(hidden.is_cuda() && w2_packed.is_cuda() && w2_scale.is_cuda() &&
                    topk_ids.is_cuda() && topk_weights.is_cuda(),
                "all tensors must be CUDA/HIP tensors");
    TORCH_CHECK(hidden.scalar_type() == torch::kBFloat16 && hidden.dim() == 2 &&
                    hidden.size(1) == 768 && hidden.size(0) >= 8 &&
                    hidden.size(0) <= 24 && hidden.size(0) % 8 == 0,
                "hidden must be BF16 [M*8,768] for M1-M3");
    TORCH_CHECK(w2_packed.scalar_type() == torch::kInt32 &&
                    w2_packed.sizes() == torch::IntArrayRef({512, 2560, 96}),
                "w2 must be [512,2560,96] int32");
    TORCH_CHECK(w2_scale.scalar_type() == torch::kBFloat16 &&
                    w2_scale.sizes() == torch::IntArrayRef({512, 2560, 24}),
                "w2 scales must be BF16 [512,2560,24]");
    const int tokens = static_cast<int>(hidden.size(0) / 8);
    TORCH_CHECK(topk_ids.scalar_type() == torch::kInt32 &&
                    topk_ids.sizes() == torch::IntArrayRef({tokens, 8}),
                "topk_ids must be int32 [M,8]");
    TORCH_CHECK(topk_weights.scalar_type() == torch::kFloat32 &&
                    topk_weights.sizes() == topk_ids.sizes(),
                "topk_weights must be FP32 [M,8]");
    TORCH_CHECK(hidden.is_contiguous() && w2_packed.is_contiguous() &&
                    w2_scale.is_contiguous() && topk_ids.is_contiguous() &&
                    topk_weights.is_contiguous(),
                "all tensors must be contiguous");

    const at::cuda::OptionalCUDAGuard device_guard(hidden.device());
    const at::cuda::CUDAStream stream = at::cuda::getCurrentCUDAStream();
    const int routes = static_cast<int>(hidden.size(0));
    torch::Tensor hidden_packed = torch::empty(
        {routes, 96}, hidden.options().dtype(torch::kInt32));
    torch::Tensor hidden_scale = torch::empty(
        {routes, 24}, hidden.options().dtype(torch::kFloat32));
    torch::Tensor hidden_packed_residual;
    torch::Tensor hidden_scale_residual;
    torch::Tensor hidden_packed_residual_second;
    torch::Tensor hidden_scale_residual_second;
    torch::Tensor hidden_packed_residual_third;
    torch::Tensor hidden_scale_residual_third;
    torch::Tensor outlier_indices;
    torch::Tensor outlier_values;
    torch::Tensor output = torch::empty({tokens, 2560}, hidden.options());

    if constexpr(ResidualPlanes >= 2)
    {
        hidden_packed_residual = torch::empty_like(hidden_packed);
        hidden_scale_residual = torch::empty_like(hidden_scale);
        if constexpr(ResidualPlanes >= 3)
        {
            hidden_packed_residual_second = torch::empty_like(hidden_packed);
            hidden_scale_residual_second = torch::empty_like(hidden_scale);
        }
        if constexpr(Outlier1)
        {
            outlier_indices = torch::empty(
                {routes, 24}, hidden.options().dtype(torch::kInt32));
            outlier_values = torch::empty(
                {routes, 24}, hidden.options().dtype(torch::kFloat32));
        }
        if constexpr(ResidualPlanes >= 4)
        {
            hidden_packed_residual_third = torch::empty_like(hidden_packed);
            hidden_scale_residual_third = torch::empty_like(hidden_scale);
            launch_quantize_residual4(hidden,
                                      hidden_packed,
                                      hidden_scale,
                                      hidden_packed_residual,
                                      hidden_scale_residual,
                                      hidden_packed_residual_second,
                                      hidden_scale_residual_second,
                                      hidden_packed_residual_third,
                                      hidden_scale_residual_third,
                                      stream);
        }
        else if constexpr(Outlier1)
        {
            launch_quantize_residual3_outlier1(hidden,
                                               hidden_packed,
                                               hidden_scale,
                                               hidden_packed_residual,
                                               hidden_scale_residual,
                                               hidden_packed_residual_second,
                                               hidden_scale_residual_second,
                                               outlier_indices,
                                               outlier_values,
                                               stream);
        }
        else if constexpr(ResidualPlanes == 3)
        {
            launch_quantize_residual3(hidden,
                                      hidden_packed,
                                      hidden_scale,
                                      hidden_packed_residual,
                                      hidden_scale_residual,
                                      hidden_packed_residual_second,
                                      hidden_scale_residual_second,
                                      stream);
        }
        else
        {
            launch_quantize_residual2(hidden,
                                      hidden_packed,
                                      hidden_scale,
                                      hidden_packed_residual,
                                      hidden_scale_residual,
                                      stream);
        }
    }
    else
    {
        launch_quantize_with_clip(
            hidden, hidden_packed, hidden_scale, stream, w2_clip_ratio());
    }
    const dim3 w2_grid(2560 / 16, tokens);
    routed_w2_topk_s4s4_g32_wmma<ResidualPlanes, false, Outlier1>
        <<<w2_grid, 32, 0, stream>>>(
        reinterpret_cast<const uint32_t*>(hidden_packed.data_ptr<int32_t>()),
        reinterpret_cast<const uint32_t*>(w2_packed.data_ptr<int32_t>()),
        hidden_scale.data_ptr<float>(),
        ResidualPlanes >= 2
            ? reinterpret_cast<const uint32_t*>(
                  hidden_packed_residual.data_ptr<int32_t>())
            : nullptr,
        ResidualPlanes >= 2 ? hidden_scale_residual.data_ptr<float>() : nullptr,
        ResidualPlanes >= 3
            ? reinterpret_cast<const uint32_t*>(
                  hidden_packed_residual_second.data_ptr<int32_t>())
            : nullptr,
        ResidualPlanes >= 3
            ? hidden_scale_residual_second.data_ptr<float>()
            : nullptr,
        ResidualPlanes >= 4
            ? reinterpret_cast<const uint32_t*>(
                  hidden_packed_residual_third.data_ptr<int32_t>())
            : nullptr,
        ResidualPlanes >= 4 ? hidden_scale_residual_third.data_ptr<float>()
                            : nullptr,
        Outlier1 ? outlier_indices.data_ptr<int32_t>() : nullptr,
        Outlier1 ? outlier_values.data_ptr<float>() : nullptr,
        reinterpret_cast<const bf16_t*>(w2_scale.data_ptr<at::BFloat16>()),
        topk_ids.data_ptr<int32_t>(),
        topk_weights.data_ptr<float>(),
        reinterpret_cast<bf16_t*>(output.data_ptr<at::BFloat16>()),
        tokens,
        512,
        2560,
        768);
    ling_iu4::check_launch("decode fused IU4 W2 top-k launch failed");
    return output;
}

torch::Tensor w2(torch::Tensor hidden,
                 torch::Tensor w2_packed,
                 torch::Tensor w2_scale,
                 torch::Tensor topk_ids,
                 torch::Tensor topk_weights)
{
    return w2_impl<1>(hidden, w2_packed, w2_scale, topk_ids, topk_weights);
}

torch::Tensor w2_residual2(torch::Tensor hidden,
                           torch::Tensor w2_packed,
                           torch::Tensor w2_scale,
                           torch::Tensor topk_ids,
                           torch::Tensor topk_weights)
{
    return w2_impl<2>(hidden, w2_packed, w2_scale, topk_ids, topk_weights);
}

torch::Tensor w2_residual3(torch::Tensor hidden,
                           torch::Tensor w2_packed,
                           torch::Tensor w2_scale,
                           torch::Tensor topk_ids,
                           torch::Tensor topk_weights)
{
    return w2_impl<3>(hidden, w2_packed, w2_scale, topk_ids, topk_weights);
}

torch::Tensor w2_residual4(torch::Tensor hidden,
                           torch::Tensor w2_packed,
                           torch::Tensor w2_scale,
                           torch::Tensor topk_ids,
                           torch::Tensor topk_weights)
{
    return w2_impl<4>(hidden, w2_packed, w2_scale, topk_ids, topk_weights);
}

torch::Tensor w2_residual3_outlier1(torch::Tensor hidden,
                                    torch::Tensor w2_packed,
                                    torch::Tensor w2_scale,
                                    torch::Tensor topk_ids,
                                    torch::Tensor topk_weights)
{
    return w2_impl<3, true>(
        hidden, w2_packed, w2_scale, topk_ids, topk_weights);
}

template <bool TiledWeights,
          bool AsymmetricU4,
          bool ActivationG16,
          bool WeightS2 = false,
          bool Residual2 = false,
          bool Residual3 = false,
          bool Residual4 = false,
          bool Clamp4 = false,
          bool Outlier1W13 = false,
          bool Outlier1W2 = false>
torch::Tensor grouped_moe_impl(torch::Tensor input,
                               torch::Tensor w13_packed,
                               torch::Tensor w13_scale,
                               torch::Tensor w2_packed,
                               torch::Tensor w2_scale,
                               torch::Tensor topk_ids,
                               torch::Tensor topk_weights,
                               torch::Tensor sorted_token_ids,
                               torch::Tensor expert_ids,
                               torch::Tensor num_tokens_post_padded,
                               c10::optional<torch::Tensor> output_arg = c10::nullopt)
{
    static_assert(!Residual4 || Residual3);
    static_assert(!Residual3 || Residual2);
    static_assert(!(Outlier1W13 || Outlier1W2) || (Residual3 && !Residual4));
    static_assert(!Residual2 || (!AsymmetricU4 && !ActivationG16 && !WeightS2));
    static std::once_flag dispatch_marker;
    std::call_once(dispatch_marker, []() {
        std::fprintf(stderr,
                     WeightS2
                          ? "LING_IU4_AFFINE_S2_W13_PREFILL_DISPATCH\n"
                          : (Outlier1W13 || Outlier1W2)
                          ? "LING_IU4_RESIDUAL3_OUTLIER1_PREFILL_DISPATCH\n"
                          : Residual4
                         ? "LING_IU4_RESIDUAL4_PREFILL_DISPATCH\n"
                         : Residual3
                         ? "LING_IU4_RESIDUAL3_PREFILL_DISPATCH\n"
                         : Residual2
                         ? "LING_IU4_RESIDUAL2_PREFILL_DISPATCH\n"
                         : ActivationG16
                         ? "LING_IU4_S4G16_PREFILL_DISPATCH\n"
                         : (AsymmetricU4
                                ? "LING_IU4_ASYM_U4_PREFILL_DISPATCH\n"
                                : "LING_IU4_GROUPED_DISPATCH\n"));
        std::fflush(stderr);
    });
    check_decode_inputs(input,
                        w13_packed,
                        w13_scale,
                        w2_packed,
                        w2_scale,
                        topk_ids,
                        topk_weights);
    TORCH_CHECK(input.size(0) >= 1,
                "grouped operator requires at least one input token");
    TORCH_CHECK(sorted_token_ids.is_cuda() && expert_ids.is_cuda() &&
                    num_tokens_post_padded.is_cuda(),
                "grouping metadata must be CUDA/HIP tensors");
    TORCH_CHECK(sorted_token_ids.scalar_type() == torch::kInt32 &&
                    expert_ids.scalar_type() == torch::kInt32 &&
                    num_tokens_post_padded.scalar_type() == torch::kInt32,
                "grouping metadata must be int32");
    TORCH_CHECK(sorted_token_ids.dim() == 1 && expert_ids.dim() == 1 &&
                    num_tokens_post_padded.numel() == 1,
                "invalid grouping metadata shapes");
    // vLLM reserves routes + E * (block_size - 1) token-id slots and one
    // expert-id entry for a possible tail.  That reserved capacity is not
    // necessarily block aligned when prompt and decode rows share a step;
    // num_tokens_post_padded is the block-aligned live extent.  Only complete
    // reserved blocks are addressable without synchronizing that scalar back
    // to the host.
    TORCH_CHECK(expert_ids.size(0) >= sorted_token_ids.size(0) / 16,
                "grouping metadata has insufficient expert block capacity");
    TORCH_CHECK(sorted_token_ids.is_contiguous() && expert_ids.is_contiguous() &&
                    num_tokens_post_padded.is_contiguous(),
                "grouping metadata must be contiguous");

    const at::cuda::OptionalCUDAGuard device_guard(input.device());
    const at::cuda::CUDAStream stream = at::cuda::getCurrentCUDAStream();
    const int tokens = static_cast<int>(input.size(0));
    const int routes = tokens * 8;
    const int capacity =
        static_cast<int>(sorted_token_ids.size(0) / 16) * 16;
    // moe_align_block_size reserves padding for every expert, but only experts
    // owning a route can contribute a padded block. Therefore routes * 16 is
    // an exact static upper bound for G=16 and avoids processing thousands of
    // impossible rows during sparse decode without a device-to-host sync.
    const int work_rows = std::min(capacity, routes * 16);
    torch::Tensor input_packed = torch::empty(
        {tokens, 320}, input.options().dtype(torch::kInt32));
    torch::Tensor input_scale = torch::empty(
        {tokens, ActivationG16 ? 160 : 80},
        input.options().dtype(torch::kFloat32));
    torch::Tensor input_zero = torch::empty(
        {tokens, 80}, input.options().dtype(torch::kInt32));
    torch::Tensor input_packed_residual;
    torch::Tensor input_scale_residual;
    torch::Tensor input_packed_residual_second;
    torch::Tensor input_scale_residual_second;
    torch::Tensor input_packed_residual_third;
    torch::Tensor input_scale_residual_third;
    torch::Tensor input_outlier_indices;
    torch::Tensor input_outlier_values;
    torch::Tensor gate_up = torch::empty({work_rows, 1536}, input.options());
    torch::Tensor hidden = torch::empty({work_rows, 768}, input.options());
    torch::Tensor hidden_packed = torch::empty(
        {work_rows, 96}, input.options().dtype(torch::kInt32));
    torch::Tensor hidden_scale = torch::empty(
        {work_rows, ActivationG16 ? 48 : 24},
        input.options().dtype(torch::kFloat32));
    torch::Tensor hidden_zero = torch::empty(
        {work_rows, 24}, input.options().dtype(torch::kInt32));
    torch::Tensor hidden_packed_residual;
    torch::Tensor hidden_scale_residual;
    torch::Tensor hidden_packed_residual_second;
    torch::Tensor hidden_scale_residual_second;
    torch::Tensor hidden_packed_residual_third;
    torch::Tensor hidden_scale_residual_third;
    torch::Tensor hidden_outlier_indices;
    torch::Tensor hidden_outlier_values;
    torch::Tensor routed_output = torch::empty({routes, 2560}, input.options());
    torch::Tensor output = output_arg.has_value()
        ? *output_arg
        : torch::empty({tokens, 2560}, input.options());
    TORCH_CHECK(output.is_cuda() && output.scalar_type() == torch::kBFloat16 &&
                    output.sizes() == torch::IntArrayRef({tokens, 2560}) &&
                    output.is_contiguous(),
                "output must be contiguous BF16 [M,2560] on the GPU");

    if constexpr(Residual2)
    {
        input_packed_residual = torch::empty_like(input_packed);
        input_scale_residual = torch::empty_like(input_scale);
        hidden_packed_residual = torch::empty_like(hidden_packed);
        hidden_scale_residual = torch::empty_like(hidden_scale);
        if constexpr(Residual3)
        {
            input_packed_residual_second = torch::empty_like(input_packed);
            input_scale_residual_second = torch::empty_like(input_scale);
            hidden_packed_residual_second = torch::empty_like(hidden_packed);
            hidden_scale_residual_second = torch::empty_like(hidden_scale);
            if constexpr(Outlier1W13)
            {
                input_outlier_indices = torch::empty(
                    {tokens, 80}, input.options().dtype(torch::kInt32));
                input_outlier_values = torch::empty(
                    {tokens, 80}, input.options().dtype(torch::kFloat32));
                launch_quantize_residual3_outlier1(
                    input,
                    input_packed,
                    input_scale,
                    input_packed_residual,
                    input_scale_residual,
                    input_packed_residual_second,
                    input_scale_residual_second,
                    input_outlier_indices,
                    input_outlier_values,
                    stream);
            }
            else if constexpr(Residual4)
            {
                input_packed_residual_third = torch::empty_like(input_packed);
                input_scale_residual_third = torch::empty_like(input_scale);
                hidden_packed_residual_third = torch::empty_like(hidden_packed);
                hidden_scale_residual_third = torch::empty_like(hidden_scale);
                launch_quantize_residual4(input,
                                          input_packed,
                                          input_scale,
                                          input_packed_residual,
                                          input_scale_residual,
                                          input_packed_residual_second,
                                          input_scale_residual_second,
                                          input_packed_residual_third,
                                          input_scale_residual_third,
                                          stream);
            }
            else
            {
                launch_quantize_residual3(input,
                                          input_packed,
                                          input_scale,
                                          input_packed_residual,
                                          input_scale_residual,
                                          input_packed_residual_second,
                                          input_scale_residual_second,
                                          stream);
            }
            if constexpr(Outlier1W2)
            {
                hidden_outlier_indices = torch::empty(
                    {work_rows, 24}, input.options().dtype(torch::kInt32));
                hidden_outlier_values = torch::empty(
                    {work_rows, 24}, input.options().dtype(torch::kFloat32));
            }
        }
        else
        {
            launch_quantize_residual2(input,
                                      input_packed,
                                      input_scale,
                                      input_packed_residual,
                                      input_scale_residual,
                                      stream);
        }
    }
    else if constexpr(ActivationG16)
    {
        launch_quantize_s4_g16(input, input_packed, input_scale, stream);
    }
    else if constexpr(AsymmetricU4)
    {
        launch_quantize_u4_asymmetric(
            input, input_packed, input_scale, input_zero, stream);
    }
    else
    {
        launch_quantize_with_clip(
            input, input_packed, input_scale, stream, w13_clip_ratio());
    }
    if constexpr(TiledWeights)
    {
        const dim3 w13_grid(1536 / 32, work_rows / 16);
        if constexpr(ActivationG16)
        {
            grouped_s4g16_s4g32_wmma_n32_tiled<false, WeightS2>
                <<<w13_grid, 32, 0, stream>>>(
                reinterpret_cast<const uint32_t*>(
                    input_packed.data_ptr<int32_t>()),
                reinterpret_cast<const uint32_t*>(w13_packed.data_ptr<int32_t>()),
                input_scale.data_ptr<float>(),
                reinterpret_cast<const bf16_t*>(
                    w13_scale.data_ptr<at::BFloat16>()),
                sorted_token_ids.data_ptr<int32_t>(),
                expert_ids.data_ptr<int32_t>(),
                num_tokens_post_padded.data_ptr<int32_t>(),
                topk_weights.data_ptr<float>(),
                reinterpret_cast<bf16_t*>(gate_up.data_ptr<at::BFloat16>()),
                routes,
                8,
                512,
                1536,
                2560);
        }
        else
        {
            grouped_s4s4_g32_wmma_n32<false,
                                      true,
                                      AsymmetricU4,
                                      Residual2,
                                      Residual3,
                                      Residual4,
                                      Outlier1W13>
                <<<w13_grid, 32, 0, stream>>>(
                reinterpret_cast<const uint32_t*>(
                    input_packed.data_ptr<int32_t>()),
                reinterpret_cast<const uint32_t*>(w13_packed.data_ptr<int32_t>()),
                input_scale.data_ptr<float>(),
                Residual2
                    ? reinterpret_cast<const uint32_t*>(
                          input_packed_residual.data_ptr<int32_t>())
                    : nullptr,
                Residual2 ? input_scale_residual.data_ptr<float>() : nullptr,
                Residual3
                    ? reinterpret_cast<const uint32_t*>(
                          input_packed_residual_second.data_ptr<int32_t>())
                    : nullptr,
                Residual3 ? input_scale_residual_second.data_ptr<float>()
                          : nullptr,
                Residual4
                    ? reinterpret_cast<const uint32_t*>(
                          input_packed_residual_third.data_ptr<int32_t>())
                    : nullptr,
                Residual4 ? input_scale_residual_third.data_ptr<float>()
                           : nullptr,
                Outlier1W13 ? input_outlier_indices.data_ptr<int32_t>() : nullptr,
                Outlier1W13 ? input_outlier_values.data_ptr<float>() : nullptr,
                input_zero.data_ptr<int32_t>(),
                reinterpret_cast<const bf16_t*>(
                    w13_scale.data_ptr<at::BFloat16>()),
                sorted_token_ids.data_ptr<int32_t>(),
                expert_ids.data_ptr<int32_t>(),
                num_tokens_post_padded.data_ptr<int32_t>(),
                topk_weights.data_ptr<float>(),
                reinterpret_cast<bf16_t*>(gate_up.data_ptr<at::BFloat16>()),
                routes,
                8,
                512,
                1536,
                2560);
        }
    }
    else
    {
        const dim3 w13_grid(1536 / 32, work_rows / 16);
        grouped_s4s4_g32_wmma_n32<false,
                                  false,
                                  AsymmetricU4,
                                  Residual2,
                                  Residual3,
                                  Residual4,
                                  Outlier1W13>
            <<<w13_grid, 32, 0, stream>>>(
            reinterpret_cast<const uint32_t*>(input_packed.data_ptr<int32_t>()),
            reinterpret_cast<const uint32_t*>(w13_packed.data_ptr<int32_t>()),
            input_scale.data_ptr<float>(),
            Residual2
                ? reinterpret_cast<const uint32_t*>(
                      input_packed_residual.data_ptr<int32_t>())
                : nullptr,
            Residual2 ? input_scale_residual.data_ptr<float>() : nullptr,
            Residual3
                ? reinterpret_cast<const uint32_t*>(
                      input_packed_residual_second.data_ptr<int32_t>())
                : nullptr,
            Residual3 ? input_scale_residual_second.data_ptr<float>() : nullptr,
            Residual4
                ? reinterpret_cast<const uint32_t*>(
                      input_packed_residual_third.data_ptr<int32_t>())
                : nullptr,
            Residual4 ? input_scale_residual_third.data_ptr<float>() : nullptr,
            Outlier1W13 ? input_outlier_indices.data_ptr<int32_t>() : nullptr,
            Outlier1W13 ? input_outlier_values.data_ptr<float>() : nullptr,
            input_zero.data_ptr<int32_t>(),
            reinterpret_cast<const bf16_t*>(w13_scale.data_ptr<at::BFloat16>()),
            sorted_token_ids.data_ptr<int32_t>(),
            expert_ids.data_ptr<int32_t>(),
            num_tokens_post_padded.data_ptr<int32_t>(),
            topk_weights.data_ptr<float>(),
            reinterpret_cast<bf16_t*>(gate_up.data_ptr<at::BFloat16>()),
            routes,
            8,
            512,
            1536,
            2560);
    }
    ling_iu4::check_launch("grouped IU4 W13 launch failed");

    constexpr int threads = 256;
    const long activation_count = static_cast<long>(work_rows) * 768;
    const int activation_blocks =
        static_cast<int>((activation_count + threads - 1) / threads);
    grouped_silu_and_mul_bf16<<<activation_blocks, threads, 0, stream>>>(
        reinterpret_cast<const bf16_t*>(gate_up.data_ptr<at::BFloat16>()),
        reinterpret_cast<bf16_t*>(hidden.data_ptr<at::BFloat16>()),
        num_tokens_post_padded.data_ptr<int32_t>(),
        work_rows,
        768,
        Clamp4 ? 4.0f : 0.0f);
    ling_iu4::check_launch("grouped IU4 SiLU launch failed");

    if constexpr(Residual2)
    {
        if constexpr(Residual3)
        {
            if constexpr(Outlier1W2)
            {
                launch_quantize_residual3_outlier1(
                    hidden,
                    hidden_packed,
                    hidden_scale,
                    hidden_packed_residual,
                    hidden_scale_residual,
                    hidden_packed_residual_second,
                    hidden_scale_residual_second,
                    hidden_outlier_indices,
                    hidden_outlier_values,
                    stream);
            }
            else if constexpr(Residual4)
            {
                launch_quantize_residual4(hidden,
                                          hidden_packed,
                                          hidden_scale,
                                          hidden_packed_residual,
                                          hidden_scale_residual,
                                          hidden_packed_residual_second,
                                          hidden_scale_residual_second,
                                          hidden_packed_residual_third,
                                          hidden_scale_residual_third,
                                          stream);
            }
            else
            {
                launch_quantize_residual3(hidden,
                                          hidden_packed,
                                          hidden_scale,
                                          hidden_packed_residual,
                                          hidden_scale_residual,
                                          hidden_packed_residual_second,
                                          hidden_scale_residual_second,
                                          stream);
            }
        }
        else
        {
            launch_quantize_residual2(hidden,
                                      hidden_packed,
                                      hidden_scale,
                                      hidden_packed_residual,
                                      hidden_scale_residual,
                                      stream);
        }
    }
    else if constexpr(ActivationG16)
    {
        launch_quantize_s4_g16(hidden, hidden_packed, hidden_scale, stream);
    }
    else if constexpr(AsymmetricU4)
    {
        launch_quantize_u4_asymmetric(
            hidden, hidden_packed, hidden_scale, hidden_zero, stream);
    }
    else
    {
        launch_quantize_with_clip(
            hidden, hidden_packed, hidden_scale, stream, w2_clip_ratio());
    }
    const dim3 w2_grid(2560 / 32, work_rows / 16);
    if constexpr(ActivationG16)
    {
        grouped_s4g16_s4g32_wmma_n32_tiled<true, false>
            <<<w2_grid, 32, 0, stream>>>(
            reinterpret_cast<const uint32_t*>(hidden_packed.data_ptr<int32_t>()),
            reinterpret_cast<const uint32_t*>(w2_packed.data_ptr<int32_t>()),
            hidden_scale.data_ptr<float>(),
            reinterpret_cast<const bf16_t*>(w2_scale.data_ptr<at::BFloat16>()),
            sorted_token_ids.data_ptr<int32_t>(),
            expert_ids.data_ptr<int32_t>(),
            num_tokens_post_padded.data_ptr<int32_t>(),
            topk_weights.data_ptr<float>(),
            reinterpret_cast<bf16_t*>(routed_output.data_ptr<at::BFloat16>()),
            routes,
            8,
            512,
            2560,
            768);
    }
    else
    {
        grouped_s4s4_g32_wmma_n32<true,
                                  TiledWeights,
                                  AsymmetricU4,
                                  Residual2,
                                  Residual3,
                                  Residual4,
                                  Outlier1W2>
            <<<w2_grid, 32, 0, stream>>>(
            reinterpret_cast<const uint32_t*>(hidden_packed.data_ptr<int32_t>()),
            reinterpret_cast<const uint32_t*>(w2_packed.data_ptr<int32_t>()),
            hidden_scale.data_ptr<float>(),
            Residual2
                ? reinterpret_cast<const uint32_t*>(
                      hidden_packed_residual.data_ptr<int32_t>())
                : nullptr,
            Residual2 ? hidden_scale_residual.data_ptr<float>() : nullptr,
            Residual3
                ? reinterpret_cast<const uint32_t*>(
                      hidden_packed_residual_second.data_ptr<int32_t>())
                : nullptr,
            Residual3 ? hidden_scale_residual_second.data_ptr<float>() : nullptr,
            Residual4
                ? reinterpret_cast<const uint32_t*>(
                      hidden_packed_residual_third.data_ptr<int32_t>())
                : nullptr,
            Residual4 ? hidden_scale_residual_third.data_ptr<float>() : nullptr,
            Outlier1W2 ? hidden_outlier_indices.data_ptr<int32_t>() : nullptr,
            Outlier1W2 ? hidden_outlier_values.data_ptr<float>() : nullptr,
            hidden_zero.data_ptr<int32_t>(),
            reinterpret_cast<const bf16_t*>(w2_scale.data_ptr<at::BFloat16>()),
            sorted_token_ids.data_ptr<int32_t>(),
            expert_ids.data_ptr<int32_t>(),
            num_tokens_post_padded.data_ptr<int32_t>(),
            topk_weights.data_ptr<float>(),
            reinterpret_cast<bf16_t*>(routed_output.data_ptr<at::BFloat16>()),
            routes,
            8,
            512,
            2560,
            768);
    }
    ling_iu4::check_launch("grouped IU4 W2 launch failed");

    const long output_count = static_cast<long>(tokens) * 2560;
    const int output_blocks = static_cast<int>((output_count + threads - 1) / threads);
    reduce_topk_bf16<<<output_blocks, threads, 0, stream>>>(
        reinterpret_cast<const bf16_t*>(routed_output.data_ptr<at::BFloat16>()),
        reinterpret_cast<bf16_t*>(output.data_ptr<at::BFloat16>()),
        tokens,
        8,
        2560);
    ling_iu4::check_launch("grouped IU4 reduction launch failed");
    return output;
}

torch::Tensor grouped_moe(torch::Tensor input,
                          torch::Tensor w13_packed,
                          torch::Tensor w13_scale,
                          torch::Tensor w2_packed,
                          torch::Tensor w2_scale,
                          torch::Tensor topk_ids,
                          torch::Tensor topk_weights,
                          torch::Tensor sorted_token_ids,
                          torch::Tensor expert_ids,
                          torch::Tensor num_tokens_post_padded)
{
    return grouped_moe_impl<false, false, false>(input,
                                   w13_packed,
                                   w13_scale,
                                   w2_packed,
                                   w2_scale,
                                   topk_ids,
                                   topk_weights,
                                   sorted_token_ids,
                                   expert_ids,
                                   num_tokens_post_padded);
}

#define DEFINE_GROUPED_ROWMAJOR_RESIDUAL(NAME, R2, R3, R4)                 \
torch::Tensor NAME(torch::Tensor input,                                    \
                   torch::Tensor w13_packed, torch::Tensor w13_scale,      \
                   torch::Tensor w2_packed, torch::Tensor w2_scale,        \
                   torch::Tensor topk_ids, torch::Tensor topk_weights,     \
                   torch::Tensor sorted_token_ids,                         \
                   torch::Tensor expert_ids,                               \
                   torch::Tensor num_tokens_post_padded)                   \
{                                                                          \
    return grouped_moe_impl<false, false, false, false, R2, R3, R4>(       \
        input, w13_packed, w13_scale, w2_packed, w2_scale, topk_ids,       \
        topk_weights, sorted_token_ids, expert_ids,                        \
        num_tokens_post_padded);                                           \
}

DEFINE_GROUPED_ROWMAJOR_RESIDUAL(grouped_moe_r2, true, false, false)
DEFINE_GROUPED_ROWMAJOR_RESIDUAL(grouped_moe_r3, true, true, false)
DEFINE_GROUPED_ROWMAJOR_RESIDUAL(grouped_moe_r4, true, true, true)

#undef DEFINE_GROUPED_ROWMAJOR_RESIDUAL

torch::Tensor grouped_moe_tiled(torch::Tensor input,
                                torch::Tensor w13_packed,
                                torch::Tensor w13_scale,
                                torch::Tensor w2_packed,
                                torch::Tensor w2_scale,
                                torch::Tensor topk_ids,
                                torch::Tensor topk_weights,
                                torch::Tensor sorted_token_ids,
                                torch::Tensor expert_ids,
                                torch::Tensor num_tokens_post_padded)
{
    return grouped_moe_impl<true, false, false>(input,
                                  w13_packed,
                                  w13_scale,
                                  w2_packed,
                                  w2_scale,
                                  topk_ids,
                                  topk_weights,
                                  sorted_token_ids,
                                  expert_ids,
                                  num_tokens_post_padded);
}

torch::Tensor grouped_moe_tiled_clamp4(
    torch::Tensor input,
    torch::Tensor w13_packed,
    torch::Tensor w13_scale,
    torch::Tensor w2_packed,
    torch::Tensor w2_scale,
    torch::Tensor topk_ids,
    torch::Tensor topk_weights,
    torch::Tensor sorted_token_ids,
    torch::Tensor expert_ids,
    torch::Tensor num_tokens_post_padded)
{
    return grouped_moe_impl<true, false, false, false, false, false, false, true>(
        input,
        w13_packed,
        w13_scale,
        w2_packed,
        w2_scale,
        topk_ids,
        topk_weights,
        sorted_token_ids,
        expert_ids,
        num_tokens_post_padded);
}

torch::Tensor grouped_moe_tiled_out(
    torch::Tensor input,
    torch::Tensor w13_packed,
    torch::Tensor w13_scale,
    torch::Tensor w2_packed,
    torch::Tensor w2_scale,
    torch::Tensor topk_ids,
    torch::Tensor topk_weights,
    torch::Tensor sorted_token_ids,
    torch::Tensor expert_ids,
    torch::Tensor num_tokens_post_padded,
    torch::Tensor output)
{
    return grouped_moe_impl<true, false, false>(input,
                                                w13_packed,
                                                w13_scale,
                                                w2_packed,
                                                w2_scale,
                                                topk_ids,
                                                topk_weights,
                                                sorted_token_ids,
                                                expert_ids,
                                                num_tokens_post_padded,
                                                output);
}

torch::Tensor grouped_moe_tiled_r2(
    torch::Tensor input,
    torch::Tensor w13_packed,
    torch::Tensor w13_scale,
    torch::Tensor w2_packed,
    torch::Tensor w2_scale,
    torch::Tensor topk_ids,
    torch::Tensor topk_weights,
    torch::Tensor sorted_token_ids,
    torch::Tensor expert_ids,
    torch::Tensor num_tokens_post_padded)
{
    return grouped_moe_impl<true, false, false, false, true>(
        input,
        w13_packed,
        w13_scale,
        w2_packed,
        w2_scale,
        topk_ids,
        topk_weights,
        sorted_token_ids,
        expert_ids,
        num_tokens_post_padded);
}

torch::Tensor grouped_moe_tiled_r2_out(
    torch::Tensor input,
    torch::Tensor w13_packed,
    torch::Tensor w13_scale,
    torch::Tensor w2_packed,
    torch::Tensor w2_scale,
    torch::Tensor topk_ids,
    torch::Tensor topk_weights,
    torch::Tensor sorted_token_ids,
    torch::Tensor expert_ids,
    torch::Tensor num_tokens_post_padded,
    torch::Tensor output)
{
    return grouped_moe_impl<true, false, false, false, true>(
        input,
        w13_packed,
        w13_scale,
        w2_packed,
        w2_scale,
        topk_ids,
        topk_weights,
        sorted_token_ids,
        expert_ids,
        num_tokens_post_padded,
        output);
}

torch::Tensor grouped_moe_tiled_r2_clamp4(
    torch::Tensor input,
    torch::Tensor w13_packed,
    torch::Tensor w13_scale,
    torch::Tensor w2_packed,
    torch::Tensor w2_scale,
    torch::Tensor topk_ids,
    torch::Tensor topk_weights,
    torch::Tensor sorted_token_ids,
    torch::Tensor expert_ids,
    torch::Tensor num_tokens_post_padded)
{
    return grouped_moe_impl<true, false, false, false, true, false, false, true>(
        input,
        w13_packed,
        w13_scale,
        w2_packed,
        w2_scale,
        topk_ids,
        topk_weights,
        sorted_token_ids,
        expert_ids,
        num_tokens_post_padded);
}

torch::Tensor grouped_moe_tiled_r3(
    torch::Tensor input,
    torch::Tensor w13_packed,
    torch::Tensor w13_scale,
    torch::Tensor w2_packed,
    torch::Tensor w2_scale,
    torch::Tensor topk_ids,
    torch::Tensor topk_weights,
    torch::Tensor sorted_token_ids,
    torch::Tensor expert_ids,
    torch::Tensor num_tokens_post_padded)
{
    return grouped_moe_impl<true, false, false, false, true, true>(
        input,
        w13_packed,
        w13_scale,
        w2_packed,
        w2_scale,
        topk_ids,
        topk_weights,
        sorted_token_ids,
        expert_ids,
        num_tokens_post_padded);
}

torch::Tensor grouped_moe_tiled_r3_out(
    torch::Tensor input,
    torch::Tensor w13_packed,
    torch::Tensor w13_scale,
    torch::Tensor w2_packed,
    torch::Tensor w2_scale,
    torch::Tensor topk_ids,
    torch::Tensor topk_weights,
    torch::Tensor sorted_token_ids,
    torch::Tensor expert_ids,
    torch::Tensor num_tokens_post_padded,
    torch::Tensor output)
{
    return grouped_moe_impl<true, false, false, false, true, true>(
        input,
        w13_packed,
        w13_scale,
        w2_packed,
        w2_scale,
        topk_ids,
        topk_weights,
        sorted_token_ids,
        expert_ids,
        num_tokens_post_padded,
        output);
}

torch::Tensor grouped_moe_tiled_r3_clamp4(
    torch::Tensor input,
    torch::Tensor w13_packed,
    torch::Tensor w13_scale,
    torch::Tensor w2_packed,
    torch::Tensor w2_scale,
    torch::Tensor topk_ids,
    torch::Tensor topk_weights,
    torch::Tensor sorted_token_ids,
    torch::Tensor expert_ids,
    torch::Tensor num_tokens_post_padded)
{
    return grouped_moe_impl<true, false, false, false, true, true, false, true>(
        input,
        w13_packed,
        w13_scale,
        w2_packed,
        w2_scale,
        topk_ids,
        topk_weights,
        sorted_token_ids,
        expert_ids,
        num_tokens_post_padded);
}

torch::Tensor grouped_moe_tiled_r3_outlier1(
    torch::Tensor input,
    torch::Tensor w13_packed,
    torch::Tensor w13_scale,
    torch::Tensor w2_packed,
    torch::Tensor w2_scale,
    torch::Tensor topk_ids,
    torch::Tensor topk_weights,
    torch::Tensor sorted_token_ids,
    torch::Tensor expert_ids,
    torch::Tensor num_tokens_post_padded)
{
    return grouped_moe_impl<true,
                            false,
                            false,
                            false,
                            true,
                            true,
                            false,
                            false,
                            false,
                            true>(input,
                                  w13_packed,
                                  w13_scale,
                                  w2_packed,
                                  w2_scale,
                                  topk_ids,
                                  topk_weights,
                                  sorted_token_ids,
                                  expert_ids,
                                  num_tokens_post_padded);
}

torch::Tensor grouped_moe_tiled_r3_outlier1_clamp4(
    torch::Tensor input,
    torch::Tensor w13_packed,
    torch::Tensor w13_scale,
    torch::Tensor w2_packed,
    torch::Tensor w2_scale,
    torch::Tensor topk_ids,
    torch::Tensor topk_weights,
    torch::Tensor sorted_token_ids,
    torch::Tensor expert_ids,
    torch::Tensor num_tokens_post_padded)
{
    return grouped_moe_impl<true,
                            false,
                            false,
                            false,
                            true,
                            true,
                            false,
                            true,
                            false,
                            true>(input,
                                  w13_packed,
                                  w13_scale,
                                  w2_packed,
                                  w2_scale,
                                  topk_ids,
                                  topk_weights,
                                  sorted_token_ids,
                                  expert_ids,
                                  num_tokens_post_padded);
}

torch::Tensor grouped_moe_tiled_r4(
    torch::Tensor input,
    torch::Tensor w13_packed,
    torch::Tensor w13_scale,
    torch::Tensor w2_packed,
    torch::Tensor w2_scale,
    torch::Tensor topk_ids,
    torch::Tensor topk_weights,
    torch::Tensor sorted_token_ids,
    torch::Tensor expert_ids,
    torch::Tensor num_tokens_post_padded)
{
    return grouped_moe_impl<true, false, false, false, true, true, true>(
        input,
        w13_packed,
        w13_scale,
        w2_packed,
        w2_scale,
        topk_ids,
        topk_weights,
        sorted_token_ids,
        expert_ids,
        num_tokens_post_padded);
}

torch::Tensor grouped_moe_tiled_r4_out(
    torch::Tensor input,
    torch::Tensor w13_packed,
    torch::Tensor w13_scale,
    torch::Tensor w2_packed,
    torch::Tensor w2_scale,
    torch::Tensor topk_ids,
    torch::Tensor topk_weights,
    torch::Tensor sorted_token_ids,
    torch::Tensor expert_ids,
    torch::Tensor num_tokens_post_padded,
    torch::Tensor output)
{
    return grouped_moe_impl<true, false, false, false, true, true, true>(
        input,
        w13_packed,
        w13_scale,
        w2_packed,
        w2_scale,
        topk_ids,
        topk_weights,
        sorted_token_ids,
        expert_ids,
        num_tokens_post_padded,
        output);
}

torch::Tensor grouped_moe_tiled_u4(torch::Tensor input,
                                   torch::Tensor w13_packed,
                                   torch::Tensor w13_scale,
                                   torch::Tensor w2_packed,
                                   torch::Tensor w2_scale,
                                   torch::Tensor topk_ids,
                                   torch::Tensor topk_weights,
                                   torch::Tensor sorted_token_ids,
                                   torch::Tensor expert_ids,
                                   torch::Tensor num_tokens_post_padded)
{
    return grouped_moe_impl<true, true, false>(input,
                                        w13_packed,
                                        w13_scale,
                                        w2_packed,
                                        w2_scale,
                                        topk_ids,
                                        topk_weights,
                                        sorted_token_ids,
                                        expert_ids,
                                        num_tokens_post_padded);
}

torch::Tensor grouped_moe_tiled_g16(torch::Tensor input,
                                    torch::Tensor w13_packed,
                                    torch::Tensor w13_scale,
                                    torch::Tensor w2_packed,
                                    torch::Tensor w2_scale,
                                    torch::Tensor topk_ids,
                                    torch::Tensor topk_weights,
                                    torch::Tensor sorted_token_ids,
                                    torch::Tensor expert_ids,
                                    torch::Tensor num_tokens_post_padded)
{
    return grouped_moe_impl<true, false, true>(input,
                                               w13_packed,
                                               w13_scale,
                                               w2_packed,
                                               w2_scale,
                                               topk_ids,
                                               topk_weights,
                                               sorted_token_ids,
                                               expert_ids,
                                               num_tokens_post_padded);
}

torch::Tensor grouped_moe_tiled_g16_out(
    torch::Tensor input,
    torch::Tensor w13_packed,
    torch::Tensor w13_scale,
    torch::Tensor w2_packed,
    torch::Tensor w2_scale,
    torch::Tensor topk_ids,
    torch::Tensor topk_weights,
    torch::Tensor sorted_token_ids,
    torch::Tensor expert_ids,
    torch::Tensor num_tokens_post_padded,
    torch::Tensor output)
{
    return grouped_moe_impl<true, false, true>(input,
                                               w13_packed,
                                               w13_scale,
                                               w2_packed,
                                               w2_scale,
                                               topk_ids,
                                               topk_weights,
                                               sorted_token_ids,
                                               expert_ids,
                                               num_tokens_post_padded,
                                               output);
}

torch::Tensor grouped_moe_tiled_g16_s2w13(
    torch::Tensor input,
    torch::Tensor w13_packed,
    torch::Tensor w13_scale,
    torch::Tensor w2_packed,
    torch::Tensor w2_scale,
    torch::Tensor topk_ids,
    torch::Tensor topk_weights,
    torch::Tensor sorted_token_ids,
    torch::Tensor expert_ids,
    torch::Tensor num_tokens_post_padded)
{
    return grouped_moe_impl<true, false, true, true>(
        input,
        w13_packed,
        w13_scale,
        w2_packed,
        w2_scale,
        topk_ids,
        topk_weights,
        sorted_token_ids,
        expert_ids,
        num_tokens_post_padded);
}

torch::Tensor moe_meta(torch::Tensor input,
                       torch::Tensor,
                       torch::Tensor,
                       torch::Tensor,
                       torch::Tensor,
                       torch::Tensor,
                       torch::Tensor)
{
    return torch::empty({input.size(0), 2560}, input.options());
}

torch::Tensor w2_meta(torch::Tensor hidden,
                      torch::Tensor,
                      torch::Tensor,
                      torch::Tensor,
                      torch::Tensor)
{
    return torch::empty({hidden.size(0) / 8, 2560}, hidden.options());
}

torch::Tensor grouped_moe_meta(torch::Tensor input,
                               torch::Tensor,
                               torch::Tensor,
                               torch::Tensor,
                               torch::Tensor,
                               torch::Tensor,
                               torch::Tensor,
                               torch::Tensor,
                               torch::Tensor,
                               torch::Tensor)
{
    return torch::empty({input.size(0), 2560}, input.options());
}

torch::Tensor moe_out_meta(torch::Tensor,
                           torch::Tensor,
                           torch::Tensor,
                           torch::Tensor,
                           torch::Tensor,
                           torch::Tensor,
                           torch::Tensor,
                           torch::Tensor output)
{
    return output;
}

torch::Tensor grouped_moe_out_meta(torch::Tensor,
                                   torch::Tensor,
                                   torch::Tensor,
                                   torch::Tensor,
                                   torch::Tensor,
                                   torch::Tensor,
                                   torch::Tensor,
                                   torch::Tensor,
                                   torch::Tensor,
                                   torch::Tensor,
                                   torch::Tensor output)
{
    return output;
}

} // namespace ling_iu4_decode

TORCH_LIBRARY(ling_iu4_decode, m)
{
    m.def(
        "moe(Tensor input, Tensor w13_packed, Tensor w13_scale, Tensor w2_packed, Tensor w2_scale, Tensor topk_ids, Tensor topk_weights) -> Tensor");
    m.def(
        "moe_tiled(Tensor input, Tensor w13_packed, Tensor w13_scale, Tensor w2_packed, Tensor w2_scale, Tensor topk_ids, Tensor topk_weights) -> Tensor");
    m.def(
        "moe_tiled_out(Tensor input, Tensor w13_packed, Tensor w13_scale, Tensor w2_packed, Tensor w2_scale, Tensor topk_ids, Tensor topk_weights, Tensor(a!) output) -> Tensor(a!)");
    m.def(
        "moe_tiled_s2w13(Tensor input, Tensor w13_packed, Tensor w13_scale, Tensor w2_packed, Tensor w2_scale, Tensor topk_ids, Tensor topk_weights) -> Tensor");
    m.def(
        "w2(Tensor hidden, Tensor w2_packed, Tensor w2_scale, Tensor topk_ids, Tensor topk_weights) -> Tensor");
    m.def(
        "w2_residual2(Tensor hidden, Tensor w2_packed, Tensor w2_scale, Tensor topk_ids, Tensor topk_weights) -> Tensor");
    m.def(
        "w2_residual3(Tensor hidden, Tensor w2_packed, Tensor w2_scale, Tensor topk_ids, Tensor topk_weights) -> Tensor");
    m.def(
        "w2_residual4(Tensor hidden, Tensor w2_packed, Tensor w2_scale, Tensor topk_ids, Tensor topk_weights) -> Tensor");
    m.def(
        "w2_residual3_outlier1(Tensor hidden, Tensor w2_packed, Tensor w2_scale, Tensor topk_ids, Tensor topk_weights) -> Tensor");
    m.def(
        "grouped_moe(Tensor input, Tensor w13_packed, Tensor w13_scale, Tensor w2_packed, Tensor w2_scale, Tensor topk_ids, Tensor topk_weights, Tensor sorted_token_ids, Tensor expert_ids, Tensor num_tokens_post_padded) -> Tensor");
    m.def(
        "grouped_moe_r2(Tensor input, Tensor w13_packed, Tensor w13_scale, Tensor w2_packed, Tensor w2_scale, Tensor topk_ids, Tensor topk_weights, Tensor sorted_token_ids, Tensor expert_ids, Tensor num_tokens_post_padded) -> Tensor");
    m.def(
        "grouped_moe_r3(Tensor input, Tensor w13_packed, Tensor w13_scale, Tensor w2_packed, Tensor w2_scale, Tensor topk_ids, Tensor topk_weights, Tensor sorted_token_ids, Tensor expert_ids, Tensor num_tokens_post_padded) -> Tensor");
    m.def(
        "grouped_moe_r4(Tensor input, Tensor w13_packed, Tensor w13_scale, Tensor w2_packed, Tensor w2_scale, Tensor topk_ids, Tensor topk_weights, Tensor sorted_token_ids, Tensor expert_ids, Tensor num_tokens_post_padded) -> Tensor");
    m.def(
        "grouped_moe_tiled(Tensor input, Tensor w13_packed, Tensor w13_scale, Tensor w2_packed, Tensor w2_scale, Tensor topk_ids, Tensor topk_weights, Tensor sorted_token_ids, Tensor expert_ids, Tensor num_tokens_post_padded) -> Tensor");
    m.def(
        "grouped_moe_tiled_clamp4(Tensor input, Tensor w13_packed, Tensor w13_scale, Tensor w2_packed, Tensor w2_scale, Tensor topk_ids, Tensor topk_weights, Tensor sorted_token_ids, Tensor expert_ids, Tensor num_tokens_post_padded) -> Tensor");
    m.def(
        "grouped_moe_tiled_out(Tensor input, Tensor w13_packed, Tensor w13_scale, Tensor w2_packed, Tensor w2_scale, Tensor topk_ids, Tensor topk_weights, Tensor sorted_token_ids, Tensor expert_ids, Tensor num_tokens_post_padded, Tensor(a!) output) -> Tensor(a!)");
    m.def(
        "grouped_moe_tiled_r2(Tensor input, Tensor w13_packed, Tensor w13_scale, Tensor w2_packed, Tensor w2_scale, Tensor topk_ids, Tensor topk_weights, Tensor sorted_token_ids, Tensor expert_ids, Tensor num_tokens_post_padded) -> Tensor");
    m.def(
        "grouped_moe_tiled_r2_clamp4(Tensor input, Tensor w13_packed, Tensor w13_scale, Tensor w2_packed, Tensor w2_scale, Tensor topk_ids, Tensor topk_weights, Tensor sorted_token_ids, Tensor expert_ids, Tensor num_tokens_post_padded) -> Tensor");
    m.def(
        "grouped_moe_tiled_r2_out(Tensor input, Tensor w13_packed, Tensor w13_scale, Tensor w2_packed, Tensor w2_scale, Tensor topk_ids, Tensor topk_weights, Tensor sorted_token_ids, Tensor expert_ids, Tensor num_tokens_post_padded, Tensor(a!) output) -> Tensor(a!)");
    m.def(
        "grouped_moe_tiled_r3(Tensor input, Tensor w13_packed, Tensor w13_scale, Tensor w2_packed, Tensor w2_scale, Tensor topk_ids, Tensor topk_weights, Tensor sorted_token_ids, Tensor expert_ids, Tensor num_tokens_post_padded) -> Tensor");
    m.def(
        "grouped_moe_tiled_r3_clamp4(Tensor input, Tensor w13_packed, Tensor w13_scale, Tensor w2_packed, Tensor w2_scale, Tensor topk_ids, Tensor topk_weights, Tensor sorted_token_ids, Tensor expert_ids, Tensor num_tokens_post_padded) -> Tensor");
    m.def(
        "grouped_moe_tiled_r3_outlier1(Tensor input, Tensor w13_packed, Tensor w13_scale, Tensor w2_packed, Tensor w2_scale, Tensor topk_ids, Tensor topk_weights, Tensor sorted_token_ids, Tensor expert_ids, Tensor num_tokens_post_padded) -> Tensor");
    m.def(
        "grouped_moe_tiled_r3_outlier1_clamp4(Tensor input, Tensor w13_packed, Tensor w13_scale, Tensor w2_packed, Tensor w2_scale, Tensor topk_ids, Tensor topk_weights, Tensor sorted_token_ids, Tensor expert_ids, Tensor num_tokens_post_padded) -> Tensor");
    m.def(
        "grouped_moe_tiled_r3_out(Tensor input, Tensor w13_packed, Tensor w13_scale, Tensor w2_packed, Tensor w2_scale, Tensor topk_ids, Tensor topk_weights, Tensor sorted_token_ids, Tensor expert_ids, Tensor num_tokens_post_padded, Tensor(a!) output) -> Tensor(a!)");
    m.def(
        "grouped_moe_tiled_r4(Tensor input, Tensor w13_packed, Tensor w13_scale, Tensor w2_packed, Tensor w2_scale, Tensor topk_ids, Tensor topk_weights, Tensor sorted_token_ids, Tensor expert_ids, Tensor num_tokens_post_padded) -> Tensor");
    m.def(
        "grouped_moe_tiled_r4_out(Tensor input, Tensor w13_packed, Tensor w13_scale, Tensor w2_packed, Tensor w2_scale, Tensor topk_ids, Tensor topk_weights, Tensor sorted_token_ids, Tensor expert_ids, Tensor num_tokens_post_padded, Tensor(a!) output) -> Tensor(a!)");
    m.def(
        "grouped_moe_tiled_u4(Tensor input, Tensor w13_packed, Tensor w13_scale, Tensor w2_packed, Tensor w2_scale, Tensor topk_ids, Tensor topk_weights, Tensor sorted_token_ids, Tensor expert_ids, Tensor num_tokens_post_padded) -> Tensor");
    m.def(
        "grouped_moe_tiled_g16(Tensor input, Tensor w13_packed, Tensor w13_scale, Tensor w2_packed, Tensor w2_scale, Tensor topk_ids, Tensor topk_weights, Tensor sorted_token_ids, Tensor expert_ids, Tensor num_tokens_post_padded) -> Tensor");
    m.def(
        "grouped_moe_tiled_g16_out(Tensor input, Tensor w13_packed, Tensor w13_scale, Tensor w2_packed, Tensor w2_scale, Tensor topk_ids, Tensor topk_weights, Tensor sorted_token_ids, Tensor expert_ids, Tensor num_tokens_post_padded, Tensor(a!) output) -> Tensor(a!)");
    m.def(
        "grouped_moe_tiled_g16_s2w13(Tensor input, Tensor w13_packed, Tensor w13_scale, Tensor w2_packed, Tensor w2_scale, Tensor topk_ids, Tensor topk_weights, Tensor sorted_token_ids, Tensor expert_ids, Tensor num_tokens_post_padded) -> Tensor");
}

TORCH_LIBRARY_IMPL(ling_iu4_decode, CUDA, m)
{
    m.impl("moe", &ling_iu4_decode::moe);
    m.impl("moe_tiled", &ling_iu4_decode::moe_tiled);
    m.impl("moe_tiled_out", &ling_iu4_decode::moe_tiled_out);
    m.impl("moe_tiled_s2w13", &ling_iu4_decode::moe_tiled_s2w13);
    m.impl("w2", &ling_iu4_decode::w2);
    m.impl("w2_residual2", &ling_iu4_decode::w2_residual2);
    m.impl("w2_residual3", &ling_iu4_decode::w2_residual3);
    m.impl("w2_residual4", &ling_iu4_decode::w2_residual4);
    m.impl("w2_residual3_outlier1",
           &ling_iu4_decode::w2_residual3_outlier1);
    m.impl("grouped_moe", &ling_iu4_decode::grouped_moe);
    m.impl("grouped_moe_r2", &ling_iu4_decode::grouped_moe_r2);
    m.impl("grouped_moe_r3", &ling_iu4_decode::grouped_moe_r3);
    m.impl("grouped_moe_r4", &ling_iu4_decode::grouped_moe_r4);
    m.impl("grouped_moe_tiled", &ling_iu4_decode::grouped_moe_tiled);
    m.impl("grouped_moe_tiled_clamp4",
           &ling_iu4_decode::grouped_moe_tiled_clamp4);
    m.impl("grouped_moe_tiled_out", &ling_iu4_decode::grouped_moe_tiled_out);
    m.impl("grouped_moe_tiled_r2", &ling_iu4_decode::grouped_moe_tiled_r2);
    m.impl("grouped_moe_tiled_r2_clamp4",
           &ling_iu4_decode::grouped_moe_tiled_r2_clamp4);
    m.impl("grouped_moe_tiled_r2_out", &ling_iu4_decode::grouped_moe_tiled_r2_out);
    m.impl("grouped_moe_tiled_r3", &ling_iu4_decode::grouped_moe_tiled_r3);
    m.impl("grouped_moe_tiled_r3_clamp4",
           &ling_iu4_decode::grouped_moe_tiled_r3_clamp4);
    m.impl("grouped_moe_tiled_r3_outlier1",
           &ling_iu4_decode::grouped_moe_tiled_r3_outlier1);
    m.impl("grouped_moe_tiled_r3_outlier1_clamp4",
           &ling_iu4_decode::grouped_moe_tiled_r3_outlier1_clamp4);
    m.impl("grouped_moe_tiled_r3_out", &ling_iu4_decode::grouped_moe_tiled_r3_out);
    m.impl("grouped_moe_tiled_r4", &ling_iu4_decode::grouped_moe_tiled_r4);
    m.impl("grouped_moe_tiled_r4_out", &ling_iu4_decode::grouped_moe_tiled_r4_out);
    m.impl("grouped_moe_tiled_u4", &ling_iu4_decode::grouped_moe_tiled_u4);
    m.impl("grouped_moe_tiled_g16", &ling_iu4_decode::grouped_moe_tiled_g16);
    m.impl("grouped_moe_tiled_g16_out",
           &ling_iu4_decode::grouped_moe_tiled_g16_out);
    m.impl("grouped_moe_tiled_g16_s2w13",
           &ling_iu4_decode::grouped_moe_tiled_g16_s2w13);
}

TORCH_LIBRARY_IMPL(ling_iu4_decode, Meta, m)
{
    m.impl("moe", &ling_iu4_decode::moe_meta);
    m.impl("moe_tiled", &ling_iu4_decode::moe_meta);
    m.impl("moe_tiled_out", &ling_iu4_decode::moe_out_meta);
    m.impl("moe_tiled_s2w13", &ling_iu4_decode::moe_meta);
    m.impl("w2", &ling_iu4_decode::w2_meta);
    m.impl("w2_residual2", &ling_iu4_decode::w2_meta);
    m.impl("w2_residual3", &ling_iu4_decode::w2_meta);
    m.impl("w2_residual4", &ling_iu4_decode::w2_meta);
    m.impl("w2_residual3_outlier1", &ling_iu4_decode::w2_meta);
    m.impl("grouped_moe", &ling_iu4_decode::grouped_moe_meta);
    m.impl("grouped_moe_r2", &ling_iu4_decode::grouped_moe_meta);
    m.impl("grouped_moe_r3", &ling_iu4_decode::grouped_moe_meta);
    m.impl("grouped_moe_r4", &ling_iu4_decode::grouped_moe_meta);
    m.impl("grouped_moe_tiled", &ling_iu4_decode::grouped_moe_meta);
    m.impl("grouped_moe_tiled_clamp4", &ling_iu4_decode::grouped_moe_meta);
    m.impl("grouped_moe_tiled_out", &ling_iu4_decode::grouped_moe_out_meta);
    m.impl("grouped_moe_tiled_r2", &ling_iu4_decode::grouped_moe_meta);
    m.impl("grouped_moe_tiled_r2_clamp4",
           &ling_iu4_decode::grouped_moe_meta);
    m.impl("grouped_moe_tiled_r2_out", &ling_iu4_decode::grouped_moe_out_meta);
    m.impl("grouped_moe_tiled_r3", &ling_iu4_decode::grouped_moe_meta);
    m.impl("grouped_moe_tiled_r3_clamp4",
           &ling_iu4_decode::grouped_moe_meta);
    m.impl("grouped_moe_tiled_r3_outlier1",
           &ling_iu4_decode::grouped_moe_meta);
    m.impl("grouped_moe_tiled_r3_outlier1_clamp4",
           &ling_iu4_decode::grouped_moe_meta);
    m.impl("grouped_moe_tiled_r3_out", &ling_iu4_decode::grouped_moe_out_meta);
    m.impl("grouped_moe_tiled_r4", &ling_iu4_decode::grouped_moe_meta);
    m.impl("grouped_moe_tiled_r4_out", &ling_iu4_decode::grouped_moe_out_meta);
    m.impl("grouped_moe_tiled_u4", &ling_iu4_decode::grouped_moe_meta);
    m.impl("grouped_moe_tiled_g16", &ling_iu4_decode::grouped_moe_meta);
    m.impl("grouped_moe_tiled_g16_out",
           &ling_iu4_decode::grouped_moe_out_meta);
    m.impl("grouped_moe_tiled_g16_s2w13",
           &ling_iu4_decode::grouped_moe_meta);
}
