#include <torch/extension.h>

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>

#include <hip/hip_bf16.h>
#include <hip/hip_runtime.h>

#include <algorithm>
#include <cstdint>
#include <tuple>

namespace ling_iu4
{

using bf16_t = __hip_bfloat16;
using int2x32 = int __attribute__((ext_vector_type(2)));
using int8x32 = int __attribute__((ext_vector_type(8)));
using float8x32 = float __attribute__((ext_vector_type(8)));

constexpr int kGroupSize = 32;
constexpr uint32_t kUint4B8ToS4Mask = 0x88888888u;

__host__ __device__ constexpr int c_fragment_row(const int lane, const int reg)
{
    return 2 * reg + ((lane & 31) >> 4);
}

__host__ __device__ constexpr int c_fragment_col(const int lane) { return lane & 15; }

// One Wave32 block quantizes one row/group of 32 BF16 values. Activations use
// the symmetric [-7, 7] set; the -8 S4 code stays unused so max-abs scaling is
// genuinely symmetric. Four lanes write four packed uint32 words.
__global__ void quantize_pack_s4_g32(const bf16_t* __restrict__ input,
                                     uint32_t* __restrict__ packed,
                                     float* __restrict__ scales,
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
    float maximum = fabsf(value);
#pragma unroll
    for(int offset = 16; offset > 0; offset >>= 1)
    {
        maximum = fmaxf(maximum, __shfl_down(maximum, offset, 32));
    }
    maximum = __shfl(maximum, 0, 32);

    const float scale = maximum > 0.0f ? maximum * (1.0f / 7.0f) : 0.0f;
    // BF16 inputs are exactly representable in FP64. Using FP64 only for the
    // ratio makes mathematical half-way cases deterministic across the HIP
    // kernel and independent reference; the scale and GEMM remain FP32/BF16.
    const double normalized = maximum > 0.0f
                                  ? static_cast<double>(value) * 7.0 /
                                        static_cast<double>(maximum)
                                  : 0.0;
    int q = maximum > 0.0f ? __double2int_rn(normalized) : 0;
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
        packed[static_cast<long>(row) * words_per_row + group * 4 + lane / 8] = word;
    }
    if(lane == 0)
    {
        scales[static_cast<long>(row) * groups + group] = scale;
    }
}

// A: two's-complement S4 packed row-major [M, K/8].
// B: official Ling uint4b8 packed row-major [N, K/8].
// Scales: A FP32 [M, K/32], B BF16 [N, K/32].
// Output: BF16 [M, N]. The official B values are converted only in registers.
__global__ void s4s4_g32_wmma(const uint32_t* __restrict__ a_packed,
                              const uint32_t* __restrict__ b_uint4b8,
                              const float* __restrict__ a_scale,
                              const bf16_t* __restrict__ b_scale,
                              bf16_t* __restrict__ output,
                              const int M,
                              const int N,
                              const int K)
{
    const int lane = threadIdx.x & 31;
    const int source_lane = lane & 15;
    const int tile_m = blockIdx.y * 16;
    const int tile_n = blockIdx.x * 16;
    const int source_row = tile_m + source_lane;
    const int source_col = tile_n + source_lane;
    const int words_per_row = K / 8;
    const int groups = K / kGroupSize;

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
            if(source_row < M)
            {
                const long base = static_cast<long>(source_row) * words_per_row + word;
                a_fragment[0] = static_cast<int>(a_packed[base]);
                a_fragment[1] = static_cast<int>(a_packed[base + 1]);
            }
            if(source_col < N)
            {
                const long base = static_cast<long>(source_col) * words_per_row + word;
                b_fragment[0] =
                    static_cast<int>(b_uint4b8[base] ^ kUint4B8ToS4Mask);
                b_fragment[1] =
                    static_cast<int>(b_uint4b8[base + 1] ^ kUint4B8ToS4Mask);
            }

            group_accumulator = __builtin_amdgcn_wmma_i32_16x16x16_iu4_w32(
                true, a_fragment, true, b_fragment, group_accumulator, false);
        }

        const int output_col = tile_n + c_fragment_col(lane);
        const float weight_scale =
            output_col < N
                ? __bfloat162float(b_scale[static_cast<long>(output_col) * groups + group])
                : 0.0f;
#pragma unroll
        for(int j = 0; j < 8; ++j)
        {
            const int output_row = tile_m + c_fragment_row(lane, j);
            if(output_row < M && output_col < N)
            {
                const float activation_scale =
                    a_scale[static_cast<long>(output_row) * groups + group];
                accumulator[j] += static_cast<float>(group_accumulator[j]) *
                                  activation_scale * weight_scale;
            }
        }
    }

    const int output_col = tile_n + c_fragment_col(lane);
#pragma unroll
    for(int j = 0; j < 8; ++j)
    {
        const int output_row = tile_m + c_fragment_row(lane, j);
        if(output_row < M && output_col < N)
        {
            output[static_cast<long>(output_row) * N + output_col] =
                __float2bfloat16(accumulator[j]);
        }
    }
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

void check_launch(const char* stage)
{
    const hipError_t status = hipGetLastError();
    TORCH_CHECK(status == hipSuccess, stage, ": ", hipGetErrorString(status));
}

void check_common(const torch::Tensor& input,
                  const torch::Tensor& weight_packed,
                  const torch::Tensor& weight_scale)
{
    TORCH_CHECK(input.is_cuda(), "input must be a CUDA/HIP tensor");
    TORCH_CHECK(weight_packed.is_cuda(), "weight_packed must be a CUDA/HIP tensor");
    TORCH_CHECK(weight_scale.is_cuda(), "weight_scale must be a CUDA/HIP tensor");
    TORCH_CHECK(input.scalar_type() == torch::kBFloat16, "input must be BF16");
    TORCH_CHECK(weight_packed.scalar_type() == torch::kInt32,
                "weight_packed must be int32");
    TORCH_CHECK(weight_scale.scalar_type() == torch::kBFloat16,
                "weight_scale must be BF16");
    TORCH_CHECK(input.dim() == 2, "input must be [M, K]");
    TORCH_CHECK(weight_packed.dim() == 2, "weight_packed must be [N, K/8]");
    TORCH_CHECK(weight_scale.dim() == 2, "weight_scale must be [N, K/32]");
    TORCH_CHECK(input.is_contiguous(), "input must be contiguous");
    TORCH_CHECK(weight_packed.is_contiguous(), "weight_packed must be contiguous");
    TORCH_CHECK(weight_scale.is_contiguous(), "weight_scale must be contiguous");
    const int64_t K = input.size(1);
    const int64_t N = weight_packed.size(0);
    TORCH_CHECK(K % kGroupSize == 0, "K must be divisible by 32");
    TORCH_CHECK(N % 16 == 0, "N must be divisible by 16");
    TORCH_CHECK(weight_packed.size(1) * 8 == K, "packed K dimension mismatch");
    TORCH_CHECK(weight_scale.size(0) == N, "scale N dimension mismatch");
    TORCH_CHECK(weight_scale.size(1) * kGroupSize == K,
                "scale group dimension mismatch");
    TORCH_CHECK(input.device() == weight_packed.device() &&
                    input.device() == weight_scale.device(),
                "all tensors must share one device");
}

void launch_quantize(const torch::Tensor& input,
                     torch::Tensor& packed,
                     torch::Tensor& scales,
                     const at::cuda::CUDAStream& stream)
{
    const int rows = static_cast<int>(input.size(0));
    const int K = static_cast<int>(input.size(1));
    const dim3 grid(K / kGroupSize, rows);
    quantize_pack_s4_g32<<<grid, 32, 0, stream>>>(
        reinterpret_cast<const bf16_t*>(input.data_ptr<at::BFloat16>()),
        reinterpret_cast<uint32_t*>(packed.data_ptr<int32_t>()),
        scales.data_ptr<float>(),
        rows,
        K);
    check_launch("quantize_pack_s4_g32 launch failed");
}

void launch_gemm(const torch::Tensor& packed,
                 const torch::Tensor& activation_scale,
                 const torch::Tensor& weight_packed,
                 const torch::Tensor& weight_scale,
                 torch::Tensor& output,
                 const int M,
                 const int N,
                 const int K,
                 const at::cuda::CUDAStream& stream)
{
    const dim3 grid((N + 15) / 16, (M + 15) / 16);
    s4s4_g32_wmma<<<grid, 32, 0, stream>>>(
        reinterpret_cast<const uint32_t*>(packed.data_ptr<int32_t>()),
        reinterpret_cast<const uint32_t*>(weight_packed.data_ptr<int32_t>()),
        activation_scale.data_ptr<float>(),
        reinterpret_cast<const bf16_t*>(weight_scale.data_ptr<at::BFloat16>()),
        reinterpret_cast<bf16_t*>(output.data_ptr<at::BFloat16>()),
        M,
        N,
        K);
    check_launch("s4s4_g32_wmma launch failed");
}

torch::Tensor s4s4_linear(torch::Tensor input,
                          torch::Tensor weight_packed,
                          torch::Tensor weight_scale)
{
    check_common(input, weight_packed, weight_scale);
    const at::cuda::OptionalCUDAGuard device_guard(input.device());
    const at::cuda::CUDAStream stream = at::cuda::getCurrentCUDAStream();
    const int M = static_cast<int>(input.size(0));
    const int K = static_cast<int>(input.size(1));
    const int N = static_cast<int>(weight_packed.size(0));

    torch::Tensor packed = torch::empty(
        {M, K / 8}, input.options().dtype(torch::kInt32));
    torch::Tensor scales = torch::empty(
        {M, K / kGroupSize}, input.options().dtype(torch::kFloat32));
    torch::Tensor output = torch::empty({M, N}, input.options());
    launch_quantize(input, packed, scales, stream);
    launch_gemm(packed, scales, weight_packed, weight_scale, output, M, N, K, stream);
    return output;
}

torch::Tensor s4s4_linear_out(torch::Tensor input,
                              torch::Tensor weight_packed,
                              torch::Tensor weight_scale,
                              torch::Tensor packed,
                              torch::Tensor activation_scale,
                              torch::Tensor output)
{
    check_common(input, weight_packed, weight_scale);
    const int M = static_cast<int>(input.size(0));
    const int K = static_cast<int>(input.size(1));
    const int N = static_cast<int>(weight_packed.size(0));
    TORCH_CHECK(packed.is_cuda() && activation_scale.is_cuda() && output.is_cuda(),
                "workspaces must be CUDA/HIP tensors");
    TORCH_CHECK(packed.scalar_type() == torch::kInt32,
                "packed workspace must be int32");
    TORCH_CHECK(activation_scale.scalar_type() == torch::kFloat32,
                "activation scale workspace must be FP32");
    TORCH_CHECK(output.scalar_type() == torch::kBFloat16,
                "output workspace must be BF16");
    TORCH_CHECK(packed.sizes() == torch::IntArrayRef({M, K / 8}),
                "packed workspace shape mismatch");
    TORCH_CHECK(activation_scale.sizes() ==
                    torch::IntArrayRef({M, K / kGroupSize}),
                "activation scale workspace shape mismatch");
    TORCH_CHECK(output.sizes() == torch::IntArrayRef({M, N}),
                "output workspace shape mismatch");
    TORCH_CHECK(packed.is_contiguous() && activation_scale.is_contiguous() &&
                    output.is_contiguous(),
                "workspaces must be contiguous");
    TORCH_CHECK(input.device() == packed.device() &&
                    input.device() == activation_scale.device() &&
                    input.device() == output.device(),
                "workspaces must share the input device");

    const at::cuda::OptionalCUDAGuard device_guard(input.device());
    const at::cuda::CUDAStream stream = at::cuda::getCurrentCUDAStream();
    launch_quantize(input, packed, activation_scale, stream);
    launch_gemm(
        packed, activation_scale, weight_packed, weight_scale, output, M, N, K, stream);
    return output;
}

std::tuple<torch::Tensor, torch::Tensor> quantize_s4(torch::Tensor input)
{
    TORCH_CHECK(input.is_cuda(), "input must be a CUDA/HIP tensor");
    TORCH_CHECK(input.scalar_type() == torch::kBFloat16, "input must be BF16");
    TORCH_CHECK(input.dim() == 2, "input must be [M, K]");
    TORCH_CHECK(input.is_contiguous(), "input must be contiguous");
    TORCH_CHECK(input.size(1) % kGroupSize == 0, "K must be divisible by 32");
    const at::cuda::OptionalCUDAGuard device_guard(input.device());
    const at::cuda::CUDAStream stream = at::cuda::getCurrentCUDAStream();
    const int M = static_cast<int>(input.size(0));
    const int K = static_cast<int>(input.size(1));
    torch::Tensor packed = torch::empty(
        {M, K / 8}, input.options().dtype(torch::kInt32));
    torch::Tensor scales = torch::empty(
        {M, K / kGroupSize}, input.options().dtype(torch::kFloat32));
    launch_quantize(input, packed, scales, stream);
    return std::make_tuple(packed, scales);
}

torch::Tensor s4s4_mlp(torch::Tensor input,
                       torch::Tensor w13_packed,
                       torch::Tensor w13_scale,
                       torch::Tensor w2_packed,
                       torch::Tensor w2_scale)
{
    check_common(input, w13_packed, w13_scale);
    const int M = static_cast<int>(input.size(0));
    const int hidden = static_cast<int>(input.size(1));
    const int gate_up = static_cast<int>(w13_packed.size(0));
    TORCH_CHECK(gate_up % 2 == 0, "w13 output must contain gate and up halves");
    const int intermediate = gate_up / 2;
    TORCH_CHECK(w2_packed.size(0) == hidden, "w2 output must equal hidden size");
    TORCH_CHECK(w2_packed.size(1) * 8 == intermediate,
                "w2 input must equal the gated intermediate size");
    TORCH_CHECK(w2_scale.size(0) == hidden &&
                    w2_scale.size(1) * kGroupSize == intermediate,
                "w2 scale shape mismatch");
    TORCH_CHECK(w2_packed.is_cuda() && w2_scale.is_cuda(), "w2 tensors must be GPU");
    TORCH_CHECK(w2_packed.scalar_type() == torch::kInt32 &&
                    w2_scale.scalar_type() == torch::kBFloat16,
                "w2 tensor dtypes must be int32/BF16");
    TORCH_CHECK(w2_packed.is_contiguous() && w2_scale.is_contiguous(),
                "w2 tensors must be contiguous");

    const at::cuda::OptionalCUDAGuard device_guard(input.device());
    const at::cuda::CUDAStream stream = at::cuda::getCurrentCUDAStream();

    torch::Tensor input_packed = torch::empty(
        {M, hidden / 8}, input.options().dtype(torch::kInt32));
    torch::Tensor input_scale = torch::empty(
        {M, hidden / kGroupSize}, input.options().dtype(torch::kFloat32));
    torch::Tensor gate_up_output = torch::empty({M, gate_up}, input.options());
    torch::Tensor intermediate_output = torch::empty({M, intermediate}, input.options());
    torch::Tensor intermediate_packed = torch::empty(
        {M, intermediate / 8}, input.options().dtype(torch::kInt32));
    torch::Tensor intermediate_scale = torch::empty(
        {M, intermediate / kGroupSize}, input.options().dtype(torch::kFloat32));
    torch::Tensor output = torch::empty({M, hidden}, input.options());

    launch_quantize(input, input_packed, input_scale, stream);
    launch_gemm(input_packed,
                input_scale,
                w13_packed,
                w13_scale,
                gate_up_output,
                M,
                gate_up,
                hidden,
                stream);

    const long activation_count = static_cast<long>(M) * intermediate;
    constexpr int threads = 256;
    const int blocks = static_cast<int>((activation_count + threads - 1) / threads);
    silu_and_mul_bf16<<<blocks, threads, 0, stream>>>(
        reinterpret_cast<const bf16_t*>(gate_up_output.data_ptr<at::BFloat16>()),
        reinterpret_cast<bf16_t*>(intermediate_output.data_ptr<at::BFloat16>()),
        M,
        intermediate);
    check_launch("silu_and_mul_bf16 launch failed");

    launch_quantize(intermediate_output, intermediate_packed, intermediate_scale, stream);
    launch_gemm(intermediate_packed,
                intermediate_scale,
                w2_packed,
                w2_scale,
                output,
                M,
                hidden,
                intermediate,
                stream);
    return output;
}

} // namespace ling_iu4

TORCH_LIBRARY(ling_iu4, m)
{
    m.def("quantize_s4(Tensor input) -> (Tensor, Tensor)");
    m.def("s4s4_linear(Tensor input, Tensor weight_packed, Tensor weight_scale) -> Tensor");
    m.def(
        "s4s4_linear_out(Tensor input, Tensor weight_packed, Tensor weight_scale, Tensor(a!) packed, Tensor(b!) activation_scale, Tensor(c!) output) -> Tensor(c!)");
    m.def(
        "s4s4_mlp(Tensor input, Tensor w13_packed, Tensor w13_scale, Tensor w2_packed, Tensor w2_scale) -> Tensor");
}

TORCH_LIBRARY_IMPL(ling_iu4, CUDA, m)
{
    m.impl("quantize_s4", &ling_iu4::quantize_s4);
    m.impl("s4s4_linear", &ling_iu4::s4s4_linear);
    m.impl("s4s4_linear_out", &ling_iu4::s4s4_linear_out);
    m.impl("s4s4_mlp", &ling_iu4::s4s4_mlp);
}
