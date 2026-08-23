# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Triton-based MoE expert implementations."""

import logging
import os

import torch

import vllm.model_executor.layers.fused_moe.modular_kernel as mk
from vllm import _custom_ops as ops
from vllm.model_executor.layers.fused_moe.activation import (
    MoEActivation,
    apply_moe_activation_supported,
)
from vllm.model_executor.layers.fused_moe.config import (
    FusedMoEConfig,
    FusedMoEParallelConfig,
    FusedMoEQuantConfig,
)
from vllm.model_executor.layers.fused_moe.experts.lora_experts_mixin import (
    LoRAExpertsMixin,
)
from vllm.model_executor.layers.fused_moe.fused_moe import (
    _prepare_expert_assignment,
    invoke_fused_moe_triton_kernel,
    invoke_fused_moe_wna16_triton_kernel,
    try_get_optimal_moe_config,
)
from vllm.model_executor.layers.fused_moe.moe_align_block_size import (
    moe_align_block_size,
)
from vllm.model_executor.layers.fused_moe.moe_fused_mul_sum import moe_sum_decode
from vllm.model_executor.layers.fused_moe.topk_weight_and_reduce import (
    TopKWeightAndReduceNoOP,
)
from vllm.model_executor.layers.fused_moe.utils import (
    _resize_cache,
    moe_kernel_quantize_input,
    swiglu_limit_func,
)
from vllm.model_executor.layers.quantization.utils.fp8_utils import (
    is_deep_gemm_e8m0_used,
)
from vllm.model_executor.layers.quantization.utils.quant_utils import (
    QuantKey,
    kFp8Dynamic128Sym,
    kFp8DynamicTensorSym,
    kFp8DynamicTokenSym,
    kFp8Static128BlockSym,
    kFp8StaticChannelSym,
    kFp8StaticTensorSym,
    kInt4Static,
    kInt4Static32,
    kInt8DynamicTensorSym,
    kInt8DynamicTokenSym,
    kInt8Static,
    kInt8StaticChannelSym,
    kInt8StaticTensorSym,
)
from vllm.platforms import current_platform
from vllm.triton_utils import tl, triton
from vllm.utils.multi_stream_utils import maybe_execute_in_parallel


logger = logging.getLogger(__name__)
_LING_IU4_DIRECT_OUT = os.environ.get("VLLM_LING_IU4_DIRECT_OUT", "") == "1"
_LING_IU4_DECODE_MODE = os.environ.get("VLLM_LING_IU4_DECODE", "")
_LING_IU4_TILED_WEIGHTS = (
    os.environ.get("VLLM_LING_IU4_TILED_WEIGHTS", "0") == "1"
)
_LING_IU4_PREFILL_QUANT = os.environ.get("LING_IU4_PREFILL_QUANT", "s4")
_LING_IU4_W13_CODEBOOK = os.environ.get("LING_IU4_W13_CODEBOOK", "s4")
_LING_IU4_MIN_TOKENS = int(os.environ.get("VLLM_LING_IU4_MIN_TOKENS", "1"))
if _LING_IU4_MIN_TOKENS < 1:
    raise ValueError("VLLM_LING_IU4_MIN_TOKENS must be at least 1")
_LING_IU4_MIN_ROUTE_EFFICIENCY = float(
    os.environ.get("VLLM_LING_IU4_MIN_ROUTE_EFFICIENCY", "0.5")
)
if not 0.0 < _LING_IU4_MIN_ROUTE_EFFICIENCY <= 1.0:
    raise ValueError("VLLM_LING_IU4_MIN_ROUTE_EFFICIENCY must be in (0,1]")
_LING_IU4_LAYER_WINDOW_TEXT = os.environ.get("VLLM_LING_IU4_LAYER_WINDOW", "")
if _LING_IU4_LAYER_WINDOW_TEXT:
    _window_parts = _LING_IU4_LAYER_WINDOW_TEXT.split(":", 1)
    if len(_window_parts) != 2:
        raise ValueError("VLLM_LING_IU4_LAYER_WINDOW must be START:END")
    _LING_IU4_LAYER_WINDOW = (int(_window_parts[0]), int(_window_parts[1]))
    if not 0 <= _LING_IU4_LAYER_WINDOW[0] < _LING_IU4_LAYER_WINDOW[1] <= 41:
        raise ValueError("VLLM_LING_IU4_LAYER_WINDOW must satisfy 0<=START<END<=41")
else:
    _LING_IU4_LAYER_WINDOW = None
if _LING_IU4_TILED_WEIGHTS and _LING_IU4_DECODE_MODE != "grouped_tiled":
    raise ValueError(
        "VLLM_LING_IU4_TILED_WEIGHTS=1 requires grouped_tiled mode"
    )
if _LING_IU4_DECODE_MODE == "grouped_tiled" and not _LING_IU4_TILED_WEIGHTS:
    raise ValueError(
        "grouped_tiled mode requires explicit VLLM_LING_IU4_TILED_WEIGHTS=1"
    )
if _LING_IU4_TILED_WEIGHTS and (
    _LING_IU4_MIN_TOKENS != 1 or _LING_IU4_LAYER_WINDOW is not None
):
    raise ValueError(
        "tiled weights prohibit token thresholds and layer windows because "
        "the W4 fallback requires canonical row-major weights"
    )
if _LING_IU4_DIRECT_OUT and _LING_IU4_DECODE_MODE != "grouped_tiled":
    raise ValueError("VLLM_LING_IU4_DIRECT_OUT=1 requires grouped_tiled mode")
_LING_IU4_WEIGHT_ORDINALS: dict[int, int] = {}
_LING_IU4_LOGGED_ORDINALS: set[int] = set()
_LING_IU4_STEP_ROUTE_DECISION: dict[str, int | float | bool] = {}
_LING_W2_CAPTURE_DIR = os.environ.get("LING_W2_CAPTURE_DIR", "")
_LING_W2_CAPTURE_ORDINALS: dict[int, int] = {}
_LING_W2_CAPTURED_ORDINALS: set[int] = set()
_LING_IU4_DECODE_ENABLED = _LING_IU4_DECODE_MODE in (
    "1",
    "full",
    "w2",
    "w2_residual2",
    "w2_residual3",
    "w2_residual3_outlier1",
    "w2_residual4",
    "grouped",
    "grouped_tiled",
)
if _LING_IU4_DECODE_ENABLED:
    _ling_iu4_library = os.environ.get("VLLM_LING_IU4_LIBRARY")
    if not _ling_iu4_library:
        raise RuntimeError(
            "VLLM_LING_IU4_LIBRARY is required when VLLM_LING_IU4_DECODE=1"
        )
    torch.ops.load_library(_ling_iu4_library)
    logger.warning("Loaded experimental Ling gfx1151 IU4 operator: %s", _ling_iu4_library)
    if _LING_IU4_DIRECT_OUT:
        logger.warning(
            "LING_IU4_DIRECT_OUT enabled: zero-copy native output with "
            "%s activation quantization at M >= %d",
            _LING_IU4_PREFILL_QUANT,
            _LING_IU4_MIN_TOKENS,
        )
        if _LING_IU4_LAYER_WINDOW is not None:
            logger.warning(
                "LING_IU4 layer window enabled: routed-MoE ordinals [%d,%d)",
                *_LING_IU4_LAYER_WINDOW,
            )


def _use_wna16_naive_block_assignment(
    *,
    num_tokens: int,
    top_k: int,
    num_experts: int,
    global_num_experts: int,
    block_size_m: int,
    expert_map: torch.Tensor | None,
    use_int4_w4a16: bool,
    block_shape: list[int] | None,
) -> bool:
    """Select the narrowly validated gfx1151 WNA16 small-token fast path."""
    return (
        num_tokens in (1, 2, 3)
        and top_k == 8
        and num_experts == 512
        and global_num_experts == 512
        and block_size_m == 16
        and expert_map is None
        and use_int4_w4a16
        and block_shape == [0, 32]
    )


_WNA16_SILU_OUTPUT_SIZE = 768
_WNA16_SILU_BLOCK_SIZE = 128


@triton.jit
def _wna16_exact_silu_and_mul_kernel(
    input_ptr,
    output_ptr,
    OUTPUT_SIZE: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    row = tl.program_id(0)
    block = tl.program_id(1)
    offsets = block * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < OUTPUT_SIZE
    row_input = input_ptr + row * (2 * OUTPUT_SIZE)
    gate = tl.load(row_input + offsets, mask=mask, other=0.0).to(tl.float32)
    up = tl.load(row_input + OUTPUT_SIZE + offsets, mask=mask, other=0.0).to(
        tl.float32
    )

    # Match the native packed HIP kernel's intermediate BF16 rounding.
    activated = (gate * tl.sigmoid(gate)).to(tl.bfloat16).to(tl.float32)
    tl.store(
        output_ptr + row * OUTPUT_SIZE + offsets,
        activated * up,
        mask=mask,
    )


def _use_wna16_exact_silu_and_mul(
    output: torch.Tensor, input: torch.Tensor
) -> bool:
    if torch.compiler.is_compiling() or not current_platform.is_rocm():
        return False
    from vllm.platforms.rocm import on_gfx1151

    return (
        on_gfx1151()
        and input.is_cuda
        and output.is_cuda
        and input.dtype == torch.bfloat16
        and output.dtype == torch.bfloat16
        and input.is_contiguous()
        and output.is_contiguous()
        and input.shape in ((8, 1536), (16, 1536))
        and output.shape == (input.shape[0], _WNA16_SILU_OUTPUT_SIZE)
    )


def _invoke_wna16_exact_silu_and_mul(
    output: torch.Tensor, input: torch.Tensor
) -> None:
    grid = (
        input.shape[0],
        triton.cdiv(_WNA16_SILU_OUTPUT_SIZE, _WNA16_SILU_BLOCK_SIZE),
    )
    _wna16_exact_silu_and_mul_kernel[grid](
        input,
        output,
        OUTPUT_SIZE=_WNA16_SILU_OUTPUT_SIZE,
        BLOCK_SIZE=_WNA16_SILU_BLOCK_SIZE,
        num_warps=1,
    )


class TritonExperts(LoRAExpertsMixin, mk.FusedMoEExpertsModular):
    """Triton-based fused MoE expert implementation."""

    @staticmethod
    def is_supported_config(
        cls: type[mk.FusedMoEExperts],
        moe_config: FusedMoEConfig,
        weight_key: QuantKey | None,
        activation_key: QuantKey | None,
        activation_format: mk.FusedMoEActivationFormat,
    ) -> tuple[bool, str | None]:
        supported, reason = mk.FusedMoEExperts.is_supported_config(
            cls,
            moe_config,
            weight_key,
            activation_key,
            activation_format,
        )
        if not supported or not current_platform.is_cuda_alike():
            return supported, reason

        padded_num_experts = (moe_config.num_experts + 31) // 32 * 32
        if padded_num_experts >= 1024:
            return False, (
                "kernel's moe_align_block_size requires fewer than 1024 padded experts"
            )
        return True, None

    def __init__(
        self,
        moe_config: FusedMoEConfig,
        quant_config: FusedMoEQuantConfig,
    ):
        # Whether quantized MOE runs natively, or through
        # higher-precision + activation QDQ.
        self.quantization_emulation = False
        super().__init__(moe_config, quant_config)

    @staticmethod
    def activation_format() -> mk.FusedMoEActivationFormat:
        return mk.FusedMoEActivationFormat.Standard

    @property
    def expects_unquantized_inputs(self) -> bool:
        # Defer activation quantization to apply() only when LoRA is active AND
        # tokens are dispatched across ranks (DP+EP all2all).
        return (
            self._lora_context is not None
            and self.quant_dtype is not None
            and self.moe_config.moe_parallel_config.use_all2all_kernels
        )

    @staticmethod
    def _supports_current_device() -> bool:
        return current_platform.is_cuda_alike() or current_platform.is_xpu()

    @staticmethod
    def _supports_no_act_and_mul() -> bool:
        return True

    @staticmethod
    def _supports_quant_scheme(
        weight_key: QuantKey | None,
        activation_key: QuantKey | None,
    ) -> bool:
        # INT8 requires at least 7.5 (Turing) on CUDA. ROCm CDNA GPUs
        # (e.g. MI2xx/MI3xx/gfx950) provide native INT8 matrix-core support and
        # the Triton int8_w8a8 fused MoE kernel handles them.
        device_supports_int8 = (
            current_platform.is_cuda()
            and current_platform.has_device_capability((7, 5))
        ) or current_platform.is_rocm()

        supported: list[tuple[QuantKey | None, QuantKey | None]] = [(None, None)]
        if device_supports_int8:
            # Activations are consumed as float and quantized to int8
            # dynamically inside the kernel, so only dynamic-activation int8
            # schemes are supported (static-activation int8 is not).
            supported += [
                # per-channel weight + dynamic per-token activation
                (kInt8StaticChannelSym, kInt8DynamicTokenSym),
                # per-tensor weight + dynamic per-tensor activation
                (kInt8StaticTensorSym, kInt8DynamicTensorSym),
            ]
        if current_platform.supports_fp8():
            supported += [
                (kFp8Static128BlockSym, kFp8Dynamic128Sym),
                (kFp8StaticChannelSym, kFp8DynamicTokenSym),
                (kFp8StaticTensorSym, kFp8DynamicTokenSym),
                (kFp8StaticTensorSym, kFp8StaticTensorSym),
                (kFp8StaticTensorSym, kFp8DynamicTensorSym),
            ]
        return (weight_key, activation_key) in supported

    @staticmethod
    def _supports_activation(activation: MoEActivation) -> bool:
        return apply_moe_activation_supported(activation)

    @staticmethod
    def _supports_parallel_config(moe_parallel_config: FusedMoEParallelConfig) -> bool:
        return not (
            moe_parallel_config.use_fi_nvl_two_sided_kernels
            or moe_parallel_config.use_fi_nvl_one_sided_kernels
        )

    @staticmethod
    def _supports_batch_invariance():
        return True

    def finalize_weight_and_reduce_impl(self) -> mk.TopKWeightAndReduce:
        return TopKWeightAndReduceNoOP()

    def activation(
        self,
        activation: MoEActivation,
        output: torch.Tensor,
        input: torch.Tensor,
        **kwargs,
    ) -> None:
        activation_config = self.activation_config
        if (
            activation == MoEActivation.SILU
            and activation_config.clamp_limit is not None
        ):
            swiglu_limit_func(output, input, activation_config.clamp_limit)
            return

        # SWIGLUOAI_UNINTERLEAVE routes to the silu_and_mul_with_clamp kernel and
        # requires a clamp limit. Other activations ignore these parameters.
        if activation == MoEActivation.SWIGLUOAI_UNINTERLEAVE:
            assert activation_config.clamp_limit is not None, (
                "SWIGLUOAI_UNINTERLEAVE requires gemm1_clamp_limit"
            )

        super().activation(
            activation,
            output,
            input,
        )

    def workspace_shapes(
        self,
        M: int,
        N: int,
        K: int,
        topk: int,
        global_num_experts: int,
        local_num_experts: int,
        expert_tokens_meta: mk.ExpertTokensMetadata | None,
        activation: MoEActivation,
    ) -> tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...]]:
        activation_out_dim = self.adjust_N_for_activation(N, activation)
        workspace1 = (M, topk, max(activation_out_dim, K))
        workspace2 = (M, topk, max(N, K))
        output = (M, K)
        return (workspace1, workspace2, output)

    def apply(
        self,
        output: torch.Tensor,
        hidden_states: torch.Tensor,
        w1: torch.Tensor,
        w2: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        activation: MoEActivation,
        global_num_experts: int,
        expert_map: torch.Tensor | None,
        a1q_scale: torch.Tensor | None,
        a2_scale: torch.Tensor | None,
        workspace13: torch.Tensor,
        workspace2: torch.Tensor,
        expert_tokens_meta: mk.ExpertTokensMetadata | None,
        apply_router_weight_on_input: bool,
    ):
        # Check constraints.
        if self.quant_config.use_int4_w4a16:
            assert hidden_states.size(-1) // 2 == w1.size(2), "Hidden size mismatch"
        else:
            assert hidden_states.size(-1) == w1.size(2), (
                f"Hidden size mismatch {hidden_states.size(-1)} != {w1.size(2)}"
            )

        assert hidden_states.is_contiguous(), "Hidden_states must be contiguous"
        assert hidden_states.dim() == 2
        assert w1.stride(-1) == 1, "Stride of last dimension must be 1"
        assert w2.stride(-1) == 1, "Stride of last dimension must be 1"
        assert hidden_states.dtype in [
            torch.float32,
            torch.float16,
            torch.bfloat16,
            torch.float8_e4m3fn,
            torch.float8_e4m3fnuz,
        ]

        # We declared expects_unquantized_inputs (LoRA + DP/EP all2all), so the
        # prepare step deferred activation quantization to this kernel:
        # `hidden_states` arrives unquantized. Keep the unquantized tensor for
        # the LoRA shrink input and quantize a copy here for the base GEMM
        # (mirrors what the prepare step would have done, but after the
        # all-gather so the layout matches the gathered topk_ids / token map).
        lora_unquantized_hidden_states: torch.Tensor | None = None
        if self.expects_unquantized_inputs:
            assert a1q_scale is None
            lora_unquantized_hidden_states = hidden_states
            hidden_states, a1q_scale = moe_kernel_quantize_input(
                hidden_states,
                self.a1_scale,
                self.quant_dtype,
                self.per_act_token_quant,
                self.block_shape,
                quantization_emulation=self.quantization_emulation,
            )

        E, num_tokens, N, K, top_k_num = self.moe_problem_size(
            hidden_states, w1, w2, topk_ids
        )

        if global_num_experts == -1:
            global_num_experts = E

        config = try_get_optimal_moe_config(
            w1.size(),
            w2.size(),
            top_k_num,
            self.quant_config.config_name(hidden_states.dtype),
            num_tokens,
            block_shape=self.block_shape,
        )

        if hidden_states.dtype == torch.bfloat16:
            compute_type = tl.bfloat16
        elif hidden_states.dtype == torch.float16:
            compute_type = tl.float16
        elif hidden_states.dtype == torch.float32:
            compute_type = tl.float32
        elif (
            hidden_states.dtype == torch.float8_e4m3fn
            or hidden_states.dtype == torch.float8_e4m3fnuz
        ):
            compute_type = tl.bfloat16
        else:
            raise ValueError(f"Unsupported compute_type: {hidden_states.dtype}")

        # Note that the output tensor might be in workspace1
        intermediate_cache1 = _resize_cache(workspace2, (num_tokens, top_k_num, N))
        cache2_dim = self.adjust_N_for_activation(N, activation)
        intermediate_cache2 = _resize_cache(
            workspace13, (num_tokens * top_k_num, cache2_dim)
        )
        intermediate_cache3 = _resize_cache(workspace2, (num_tokens, top_k_num, K))

        sorted_token_ids, expert_ids, num_tokens_post_padded = (
            _prepare_expert_assignment(
                topk_ids,
                config,
                num_tokens,
                top_k_num,
                global_num_experts,
                expert_map,
                use_int8_w8a16=self.quant_config.use_int8_w8a16,
                use_int4_w4a16=self.quant_config.use_int4_w4a16,
                block_shape=self.block_shape,
            )
        )

        # LoRA w13: applied to intermediate_cache1 before activation. When
        # the LoRA layer requested a dual-stream schedule, we run base w13
        # GEMM on the default stream and the LoRA fast-path on aux_stream;
        # the LoRA writes its delta into a fresh zero buffer (add_inputs=
        # False) and we sum it into intermediate_cache1 after both finish.
        #
        # The LoRA shrink kernel needs unquantized, gathered-layout
        # activations. When activation quant was deferred to this kernel
        # (expects_unquantized_inputs), the input we quantized above is exactly
        # that, so use it directly. Otherwise fall back to the context stash
        # (e.g. weight-only quant), guarding on a row-count match so a
        # DP-gathered layout never indexes a local stash out of bounds.
        sorted_token_ids_lora = None
        expert_ids_lora = None
        num_tokens_post_padded_lora = None
        token_lora_mapping = None
        lora_context = self._lora_context
        if lora_unquantized_hidden_states is not None:
            lora_x = lora_unquantized_hidden_states
        elif (
            lora_context is not None
            and lora_context.original_hidden_states is not None
            and lora_context.original_hidden_states.shape[0] == hidden_states.shape[0]
        ):
            lora_x = lora_context.original_hidden_states
        else:
            lora_x = hidden_states

        # TODO: The fallback to self.a1_scale was added for deferred static
        # activation quantization in https://github.com/vllm-project/vllm/pull/40857.
        # Activation emulation relies solely on `a1q_scale` output of
        # `moe_kernel_quantize_input` - this should be adapted to
        # always solely rely on `a1q_scale`.
        input_scale = (
            a1q_scale
            if self.quantization_emulation
            else (a1q_scale if a1q_scale is not None else self.a1_scale)
        )

        def _base_w13_fn():
            invoke_fused_moe_triton_kernel(
                hidden_states,
                w1,
                intermediate_cache1,
                input_scale,
                self.w1_scale,
                None,  # topk_weights
                sorted_token_ids,
                expert_ids,
                num_tokens_post_padded,
                False,  # mul_routed_weights
                top_k_num,
                config,
                compute_type=compute_type,
                use_fp8_w8a8=self.quant_config.use_fp8_w8a8,
                use_int8_w8a8=self.quant_config.use_int8_w8a8,
                use_int8_w8a16=self.quant_config.use_int8_w8a16,
                use_int4_w4a16=self.quant_config.use_int4_w4a16,
                per_channel_quant=self.per_act_token_quant,
                block_shape=self.block_shape,
                B_bias=self.w1_bias,
            )

        if lora_context is not None and lora_context.aux_stream is not None:
            # add_inputs=False: kernel overwrites lora_delta_w13. zeros (not
            # empty) so untouched rows -- e.g. blocks where every program
            # early-exits because lora_id<0 -- stay at zero and the trailing
            # add_() is a no-op there.
            lora_delta_w13 = torch.zeros_like(intermediate_cache1)

            def _lora_w13_fn():
                return self.apply_w13_lora(
                    lora_context,
                    y=lora_delta_w13,
                    x=lora_x,
                    topk_ids=topk_ids,
                    topk_weights=topk_weights,
                    expert_map=expert_map,
                    w1=w1,
                    w2=w2,
                    num_tokens=num_tokens,
                    top_k_num=top_k_num,
                    add_inputs=False,
                )

            assert lora_context.events is not None
            _, lora_meta = maybe_execute_in_parallel(
                _base_w13_fn,
                _lora_w13_fn,
                lora_context.events[0],
                lora_context.events[1],
                lora_context.aux_stream,
            )
            (
                sorted_token_ids_lora,
                expert_ids_lora,
                num_tokens_post_padded_lora,
                token_lora_mapping,
            ) = lora_meta
            intermediate_cache1.add_(lora_delta_w13)
        else:
            _base_w13_fn()
            if lora_context is not None:
                (
                    sorted_token_ids_lora,
                    expert_ids_lora,
                    num_tokens_post_padded_lora,
                    token_lora_mapping,
                ) = self.apply_w13_lora(
                    lora_context,
                    y=intermediate_cache1,
                    x=lora_x,
                    topk_ids=topk_ids,
                    topk_weights=topk_weights,
                    expert_map=expert_map,
                    w1=w1,
                    w2=w2,
                    num_tokens=num_tokens,
                    top_k_num=top_k_num,
                )

        a2q_scale: torch.Tensor | None = None

        # Fuse SiLU+Mul + FP8 block quantize into a single kernel
        # when conditions permit (gated SiLU, fp8 block quant with
        # group_size=128, no LoRA requiring the BF16 intermediate).
        if (
            activation == MoEActivation.SILU
            and self.quant_config.use_fp8_w8a8
            and self.block_shape == [128, 128]
            and lora_context is None
            and not is_deep_gemm_e8m0_used()
        ):
            qintermediate_cache2, a2q_scale = ops.silu_and_mul_per_block_quant(
                intermediate_cache1.view(-1, N),
                group_size=128,
                quant_dtype=current_platform.fp8_dtype(),
            )
        else:
            self.activation(
                activation, intermediate_cache2, intermediate_cache1.view(-1, N)
            )

            qintermediate_cache2, a2q_scale = moe_kernel_quantize_input(
                intermediate_cache2,
                a2_scale,
                self.quant_dtype,
                self.per_act_token_quant,
                self.block_shape,
                quantization_emulation=self.quantization_emulation,
            )

        # LoRA w2: applied to intermediate_cache3 before moe_sum, using the
        # unquantized intermediate_cache2 as the lora_a input.  Reuses the
        # sorted_token_ids_lora computed above. Same dual-stream pattern as
        # the w13 pair: base GEMM on default stream, LoRA delta on aux,
        # join via .add_() into intermediate_cache3.
        def _base_w2_fn():
            invoke_fused_moe_triton_kernel(
                qintermediate_cache2,
                w2,
                intermediate_cache3,
                a2q_scale,
                self.w2_scale,
                topk_weights,
                sorted_token_ids,
                expert_ids,
                num_tokens_post_padded,
                not apply_router_weight_on_input,
                1,
                config,
                compute_type=compute_type,
                use_fp8_w8a8=self.quant_config.use_fp8_w8a8,
                use_int8_w8a8=self.quant_config.use_int8_w8a8,
                use_int8_w8a16=self.quant_config.use_int8_w8a16,
                use_int4_w4a16=self.quant_config.use_int4_w4a16,
                per_channel_quant=self.per_act_token_quant,
                block_shape=self.block_shape,
                B_bias=self.w2_bias,
            )

        if lora_context is not None and lora_context.aux_stream is not None:
            lora_delta_w2 = torch.zeros_like(intermediate_cache3)

            def _lora_w2_fn():
                self.apply_w2_lora(
                    lora_context,
                    y=lora_delta_w2,
                    x=intermediate_cache2,
                    topk_weights=topk_weights,
                    sorted_token_ids_lora=sorted_token_ids_lora,
                    expert_ids_lora=expert_ids_lora,
                    num_tokens_post_padded_lora=num_tokens_post_padded_lora,
                    token_lora_mapping=token_lora_mapping,
                    num_tokens=num_tokens,
                    w1=w1,
                    w2=w2,
                    top_k_num=top_k_num,
                    add_inputs=False,
                )

            assert lora_context.events is not None
            maybe_execute_in_parallel(
                _base_w2_fn,
                _lora_w2_fn,
                lora_context.events[2],
                lora_context.events[3],
                lora_context.aux_stream,
            )
            intermediate_cache3.add_(lora_delta_w2)
        else:
            _base_w2_fn()
            if lora_context is not None:
                self.apply_w2_lora(
                    lora_context,
                    y=intermediate_cache3,
                    x=intermediate_cache2,
                    topk_weights=topk_weights,
                    sorted_token_ids_lora=sorted_token_ids_lora,
                    expert_ids_lora=expert_ids_lora,
                    num_tokens_post_padded_lora=num_tokens_post_padded_lora,
                    token_lora_mapping=token_lora_mapping,
                    num_tokens=num_tokens,
                    w1=w1,
                    w2=w2,
                    top_k_num=top_k_num,
                )

        # separate function is required for MoE + LoRA
        self.moe_sum(intermediate_cache3, output)

    def moe_sum(self, input: torch.Tensor, output: torch.Tensor) -> None:
        ops.moe_sum(input, output)


class TritonWNA16Experts(TritonExperts):
    @staticmethod
    def _supports_current_device() -> bool:
        return current_platform.is_cuda_alike() or current_platform.is_xpu()

    @staticmethod
    def _supports_no_act_and_mul() -> bool:
        return True

    @staticmethod
    def _supports_quant_scheme(
        weight_key: QuantKey | None,
        activation_key: QuantKey | None,
    ) -> bool:
        SUPPORTED_W = [
            kInt4Static,
            kInt8Static,
            kInt4Static32,
            # other group sizes?
        ]
        return weight_key in SUPPORTED_W

    @staticmethod
    def _supports_activation(activation: MoEActivation) -> bool:
        return activation in [
            MoEActivation.SILU,
            MoEActivation.GELU,
            MoEActivation.GELU_TANH,
            MoEActivation.SWIGLUOAI,
            MoEActivation.SWIGLUSTEP,
            MoEActivation.SILU_NO_MUL,
            MoEActivation.GELU_NO_MUL,
            MoEActivation.GELU_TANH_NO_MUL,
            MoEActivation.RELU2_NO_MUL,
        ]

    @staticmethod
    def _supports_parallel_config(moe_parallel_config: FusedMoEParallelConfig) -> bool:
        return not (
            moe_parallel_config.use_fi_nvl_two_sided_kernels
            or moe_parallel_config.use_fi_nvl_one_sided_kernels
        )

    def activation(
        self,
        activation: MoEActivation,
        output: torch.Tensor,
        input: torch.Tensor,
        **kwargs,
    ) -> None:
        # Keep compiled MLA-side MoE activations on Inductor's fused pointwise
        # route. The exact KDA-side eager shapes otherwise pay the high fixed
        # latency of the generic HIP activation kernel on gfx1151.
        if (
            activation == MoEActivation.SILU
            and self.activation_config.clamp_limit is None
            and self.quant_config.use_int4_w4a16
            and self.block_shape == [0, 32]
            and _use_wna16_exact_silu_and_mul(output, input)
        ):
            _invoke_wna16_exact_silu_and_mul(output, input)
            return

        super().activation(
            activation,
            output,
            input,
            **kwargs,
        )

    def apply(
        self,
        output: torch.Tensor,
        hidden_states: torch.Tensor,
        w1: torch.Tensor,
        w2: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        activation: MoEActivation,
        global_num_experts: int,
        expert_map: torch.Tensor | None,
        a1q_scale: torch.Tensor | None,
        a2_scale: torch.Tensor | None,
        workspace13: torch.Tensor,
        workspace2: torch.Tensor,
        expert_tokens_meta: mk.ExpertTokensMetadata | None,
        apply_router_weight_on_input: bool,
    ):
        # Check constraints.
        if self.quant_config.use_int4_w4a16:
            assert hidden_states.size(-1) // 2 == w1.size(2), (
                f"Hidden size mismatch {hidden_states.size(-1) // 2} == {w1.size(2)}"
            )
        else:
            assert hidden_states.size(-1) == w1.size(2), (
                f"Hidden size mismatch {hidden_states.size(-1)} != {w1.size(2)}"
            )

        assert hidden_states.is_contiguous(), "Hidden_states must be contiguous"
        assert hidden_states.dim() == 2
        assert w1.stride(-1) == 1, "Stride of last dimension must be 1"
        assert w2.stride(-1) == 1, "Stride of last dimension must be 1"
        assert hidden_states.dtype in [
            torch.float32,
            torch.float16,
            torch.bfloat16,
            torch.float8_e4m3fn,
            torch.float8_e4m3fnuz,
        ]

        E, num_tokens, N, K, top_k_num = self.moe_problem_size(
            hidden_states, w1, w2, topk_ids
        )

        if global_num_experts == -1:
            global_num_experts = E

        ling_iu4_layer_selected = True
        layer_ordinal = None
        if _LING_IU4_DECODE_ENABLED:
            weight_key = w1.data_ptr()
            layer_ordinal = _LING_IU4_WEIGHT_ORDINALS.get(weight_key)
            if layer_ordinal is None:
                layer_ordinal = len(_LING_IU4_WEIGHT_ORDINALS)
                _LING_IU4_WEIGHT_ORDINALS[weight_key] = layer_ordinal
            if _LING_IU4_LAYER_WINDOW is not None:
                ling_iu4_layer_selected = (
                    _LING_IU4_LAYER_WINDOW[0]
                    <= layer_ordinal
                    < _LING_IU4_LAYER_WINDOW[1]
                )
            if (
                ling_iu4_layer_selected
                and layer_ordinal not in _LING_IU4_LOGGED_ORDINALS
            ):
                _LING_IU4_LOGGED_ORDINALS.add(layer_ordinal)
                logger.warning("LING_IU4 selected routed-MoE ordinal %d", layer_ordinal)

        ling_iu4_common_eligible = (
            _LING_IU4_DECODE_ENABLED
            and num_tokens >= _LING_IU4_MIN_TOKENS
            and ling_iu4_layer_selected
            and current_platform.is_rocm()
            and hidden_states.dtype == torch.bfloat16
            and output.dtype == torch.bfloat16
            and E == 512
            and global_num_experts == 512
            and top_k_num == 8
            and N == 1536
            and K == 2560
            and tuple(w1.shape) == (512, 1536, 1280)
            and tuple(w2.shape) == (512, 2560, 384)
            and w1.dtype == torch.uint8
            and w2.dtype == torch.uint8
            and self.w1_scale is not None
            and self.w2_scale is not None
            and tuple(self.w1_scale.shape) == (512, 1536, 80)
            and tuple(self.w2_scale.shape) == (512, 2560, 24)
            and self.w1_scale.dtype == torch.bfloat16
            and self.w2_scale.dtype == torch.bfloat16
            and self.quant_config.use_int4_w4a16
            and self.quant_config.w1_zp is None
            and self.quant_config.w2_zp is None
            and self.block_shape == [0, 32]
            and activation == MoEActivation.SILU
            and self.activation_config.clamp_limit is None
            and expert_map is None
            and a1q_scale is None
            and a2_scale is None
            and expert_tokens_meta is None
            and not apply_router_weight_on_input
            and self._lora_context is None
            and hidden_states.is_contiguous()
            and topk_ids.dtype == torch.int32
            and topk_ids.is_contiguous()
            and topk_weights.dtype == torch.float32
            and topk_weights.is_contiguous()
        )
        ling_iu4_decode_eligible = ling_iu4_common_eligible and num_tokens in (
            1,
            2,
            3,
        ) and (
            _LING_IU4_DECODE_MODE in (
                "w2_residual2",
                "w2_residual3",
                "w2_residual3_outlier1",
                "w2_residual4",
            )
            or _LING_IU4_PREFILL_QUANT not in (
                "asym_u4", "residual2", "residual3", "residual4"
            )
        )
        ling_iu4_tiled_companion = None
        companion_mode = bool(
            os.environ.get("VLLM_LING_IU4_COMPANION_LAYERS", "")
        )
        if ling_iu4_common_eligible and companion_mode:
            from vllm.model_executor.layers.fused_moe.oracle.int_wna16 import (
                get_ling_iu4_tiled_companion,
            )

            ling_iu4_tiled_companion = get_ling_iu4_tiled_companion(w1.data_ptr())
        ling_iu4_grouped_candidate = (
            ling_iu4_common_eligible
            and _LING_IU4_DECODE_MODE in ("grouped", "grouped_tiled")
            and num_tokens >= 1
            and (not companion_mode or ling_iu4_tiled_companion is not None)
        )
        grouped_token_ids = grouped_expert_ids = grouped_post_padded = None
        ling_iu4_grouped_eligible = ling_iu4_grouped_candidate
        if ling_iu4_grouped_candidate:
            grouped_token_ids, grouped_expert_ids, grouped_post_padded = (
                moe_align_block_size(topk_ids, 16, global_num_experts, expert_map)
            )
            # One exact device-to-host occupancy decision at the first routed
            # MoE layer per scheduler step.  Later layers reuse it, avoiding
            # forty stream synchronizations while replacing the old global-M
            # guess with the step's real routed padding efficiency.
            # A full tiled checkpoint has no valid W4 fallback, so occupancy
            # is a scheduling statistic rather than a dispatch veto.  The
            # gate only chooses between IU4 and canonical W4 when both layouts
            # are valid (row-major or selective-companion modes).
            if _LING_IU4_TILED_WEIGHTS:
                ling_iu4_grouped_eligible = True
            elif layer_ordinal == 0:
                post_padded = int(grouped_post_padded.item())
                route_efficiency = (num_tokens * top_k_num) / max(post_padded, 1)
                ling_iu4_grouped_eligible = (
                    route_efficiency >= _LING_IU4_MIN_ROUTE_EFFICIENCY
                )
                _LING_IU4_STEP_ROUTE_DECISION.update(
                    num_tokens=num_tokens,
                    post_padded=post_padded,
                    route_efficiency=route_efficiency,
                    eligible=ling_iu4_grouped_eligible,
                )
                logger.warning(
                    "LING_IU4 route gate M=%d postpad=%d efficiency=%.4f "
                    "eligible=%s",
                    num_tokens,
                    post_padded,
                    route_efficiency,
                    ling_iu4_grouped_eligible,
                )
            else:
                ling_iu4_grouped_eligible = bool(
                    _LING_IU4_STEP_ROUTE_DECISION.get("eligible", False)
                    and _LING_IU4_STEP_ROUTE_DECISION.get("num_tokens")
                    == num_tokens
                )
        if _LING_IU4_TILED_WEIGHTS and not ling_iu4_grouped_eligible:
            raise RuntimeError(
                "tiled IU4 weights cannot fall through to the row-major W4 kernel; "
                f"common={ling_iu4_common_eligible} grouped_candidate="
                f"{ling_iu4_grouped_candidate} enabled={_LING_IU4_DECODE_ENABLED} "
                f"min={_LING_IU4_MIN_TOKENS} M={num_tokens} layer={layer_ordinal} "
                f"selected={ling_iu4_layer_selected} rocm={current_platform.is_rocm()} "
                f"E={E}/{global_num_experts} topk={top_k_num} N={N} K={K} "
                f"hidden={hidden_states.dtype}/{hidden_states.is_contiguous()} "
                f"output={output.dtype} w1={tuple(w1.shape)}/{w1.dtype} "
                f"w2={tuple(w2.shape)}/{w2.dtype} "
                f"s1={None if self.w1_scale is None else (tuple(self.w1_scale.shape), self.w1_scale.dtype)} "
                f"s2={None if self.w2_scale is None else (tuple(self.w2_scale.shape), self.w2_scale.dtype)} "
                f"block={self.block_shape} activation={activation} "
                f"int4={self.quant_config.use_int4_w4a16} "
                f"zp1={self.quant_config.w1_zp is not None} "
                f"zp2={self.quant_config.w2_zp is not None} "
                f"clamp={self.activation_config.clamp_limit} "
                f"expert_map={expert_map is not None} a1={a1q_scale is not None} "
                f"a2={a2_scale is not None} meta={expert_tokens_meta is not None} "
                f"router_input={apply_router_weight_on_input} lora={self._lora_context is not None} "
                f"ids={topk_ids.dtype}/{topk_ids.is_contiguous()} "
                f"weights={topk_weights.dtype}/{topk_weights.is_contiguous()}"
            )
        if ling_iu4_decode_eligible and _LING_IU4_DECODE_MODE == "grouped_tiled":
            if (
                _LING_IU4_DIRECT_OUT
                and _LING_IU4_PREFILL_QUANT in ("s4", "s4_g16")
                and _LING_IU4_W13_CODEBOOK != "affine_s2"
            ):
                torch.ops.ling_iu4_decode.moe_tiled_out(
                    hidden_states,
                    w1.view(torch.int32),
                    self.w1_scale,
                    w2.view(torch.int32),
                    self.w2_scale,
                    topk_ids,
                    topk_weights,
                    output,
                )
                return
            decode_operator = (
                torch.ops.ling_iu4_decode.moe_tiled_s2w13
                if _LING_IU4_W13_CODEBOOK == "affine_s2"
                else torch.ops.ling_iu4_decode.moe_tiled
            )
            candidate = decode_operator(
                hidden_states,
                w1.view(torch.int32),
                self.w1_scale,
                w2.view(torch.int32),
                self.w2_scale,
                topk_ids,
                topk_weights,
            )
            output.copy_(candidate)
            return
        if ling_iu4_decode_eligible and _LING_IU4_DECODE_MODE in ("1", "full"):
            candidate = torch.ops.ling_iu4_decode.moe(
                hidden_states,
                w1.view(torch.int32),
                self.w1_scale,
                w2.view(torch.int32),
                self.w2_scale,
                topk_ids,
                topk_weights,
            )
            output.copy_(candidate)
            return

        config = try_get_optimal_moe_config(
            w1.size(),
            w2.size(),
            top_k_num,
            self.quant_config.config_name(hidden_states.dtype),
            num_tokens,
            block_shape=self.block_shape,
        )

        if hidden_states.dtype == torch.bfloat16:
            compute_type = tl.bfloat16
        elif hidden_states.dtype == torch.float16:
            compute_type = tl.float16
        elif hidden_states.dtype == torch.float32:
            compute_type = tl.float32
        elif (
            hidden_states.dtype == torch.float8_e4m3fn
            or hidden_states.dtype == torch.float8_e4m3fnuz
        ):
            compute_type = tl.bfloat16
        else:
            raise ValueError(f"Unsupported compute_type: {hidden_states.dtype}")

        if ling_iu4_grouped_eligible:
            assert grouped_token_ids is not None
            assert grouped_expert_ids is not None
            assert grouped_post_padded is not None
            operator_w1, operator_w2 = w1, w2
            operator_w1_scale, operator_w2_scale = self.w1_scale, self.w2_scale
            tiled_companion = ling_iu4_tiled_companion
            if tiled_companion is not None:
                grouped_operator = (
                    torch.ops.ling_iu4_decode.grouped_moe_tiled_g16
                )
                operator_w1, operator_w2, operator_w1_scale, operator_w2_scale, _ = (
                    tiled_companion
                )
            elif _LING_IU4_DECODE_MODE != "grouped_tiled":
                if _LING_IU4_PREFILL_QUANT == "residual2":
                    grouped_operator = torch.ops.ling_iu4_decode.grouped_moe_r2
                elif _LING_IU4_PREFILL_QUANT == "residual3":
                    grouped_operator = torch.ops.ling_iu4_decode.grouped_moe_r3
                elif _LING_IU4_PREFILL_QUANT == "residual4":
                    grouped_operator = torch.ops.ling_iu4_decode.grouped_moe_r4
                else:
                    grouped_operator = torch.ops.ling_iu4_decode.grouped_moe
            elif (
                _LING_IU4_PREFILL_QUANT == "s4_g16"
                and _LING_IU4_W13_CODEBOOK == "affine_s2"
            ):
                grouped_operator = (
                    torch.ops.ling_iu4_decode.grouped_moe_tiled_g16_s2w13
                )
            elif _LING_IU4_PREFILL_QUANT == "asym_u4":
                grouped_operator = torch.ops.ling_iu4_decode.grouped_moe_tiled_u4
            elif _LING_IU4_PREFILL_QUANT == "residual2":
                grouped_operator = torch.ops.ling_iu4_decode.grouped_moe_tiled_r2
            elif _LING_IU4_PREFILL_QUANT == "residual3":
                grouped_operator = torch.ops.ling_iu4_decode.grouped_moe_tiled_r3
            elif _LING_IU4_PREFILL_QUANT == "residual4":
                grouped_operator = torch.ops.ling_iu4_decode.grouped_moe_tiled_r4
            elif _LING_IU4_PREFILL_QUANT == "s4_g16":
                grouped_operator = torch.ops.ling_iu4_decode.grouped_moe_tiled_g16
            else:
                grouped_operator = torch.ops.ling_iu4_decode.grouped_moe_tiled
            if (
                _LING_IU4_DIRECT_OUT
                and _LING_IU4_PREFILL_QUANT in (
                    "s4", "s4_g16", "residual2", "residual3", "residual4"
                )
                and _LING_IU4_W13_CODEBOOK != "affine_s2"
            ):
                if _LING_IU4_PREFILL_QUANT == "s4_g16":
                    grouped_out_operator = (
                        torch.ops.ling_iu4_decode.grouped_moe_tiled_g16_out
                    )
                elif _LING_IU4_PREFILL_QUANT == "residual2":
                    grouped_out_operator = (
                        torch.ops.ling_iu4_decode.grouped_moe_tiled_r2_out
                    )
                elif _LING_IU4_PREFILL_QUANT == "residual3":
                    grouped_out_operator = (
                        torch.ops.ling_iu4_decode.grouped_moe_tiled_r3_out
                    )
                elif _LING_IU4_PREFILL_QUANT == "residual4":
                    grouped_out_operator = (
                        torch.ops.ling_iu4_decode.grouped_moe_tiled_r4_out
                    )
                else:
                    grouped_out_operator = (
                        torch.ops.ling_iu4_decode.grouped_moe_tiled_out
                    )
                grouped_out_operator(
                    hidden_states,
                    w1.view(torch.int32),
                    self.w1_scale,
                    w2.view(torch.int32),
                    self.w2_scale,
                    topk_ids,
                    topk_weights,
                    grouped_token_ids,
                    grouped_expert_ids,
                    grouped_post_padded,
                    output,
                )
            else:
                candidate = grouped_operator(
                    hidden_states,
                    operator_w1.view(torch.int32),
                    operator_w1_scale,
                    operator_w2.view(torch.int32),
                    operator_w2_scale,
                    topk_ids,
                    topk_weights,
                    grouped_token_ids,
                    grouped_expert_ids,
                    grouped_post_padded,
                )
                output.copy_(candidate)
            return

        # Note that the output tensor might be in workspace1
        intermediate_cache1 = _resize_cache(workspace2, (num_tokens, top_k_num, N))
        activation_out_dim = self.adjust_N_for_activation(N, activation)
        intermediate_cache2 = _resize_cache(
            workspace13, (num_tokens * top_k_num, activation_out_dim)
        )
        intermediate_cache3 = _resize_cache(workspace2, (num_tokens, top_k_num, K))

        use_naive_block_assignment = _use_wna16_naive_block_assignment(
            num_tokens=num_tokens,
            top_k=top_k_num,
            num_experts=E,
            global_num_experts=global_num_experts,
            block_size_m=config["BLOCK_SIZE_M"],
            expert_map=expert_map,
            use_int4_w4a16=self.quant_config.use_int4_w4a16,
            block_shape=self.block_shape,
        )
        use_naive_block_assignment = (
            use_naive_block_assignment
            and topk_ids.shape == (num_tokens, top_k_num)
            and topk_ids.is_contiguous()
        )
        if use_naive_block_assignment:
            sorted_token_ids = None
            expert_ids = topk_ids.view(-1)
            # Compile-time dead in the naive kernel specialization.
            num_tokens_post_padded = expert_ids
        else:
            sorted_token_ids, expert_ids, num_tokens_post_padded = (
                moe_align_block_size(
                    topk_ids,
                    config["BLOCK_SIZE_M"],
                    global_num_experts,
                    expert_map,
                )
            )

        invoke_fused_moe_wna16_triton_kernel(
            hidden_states,
            w1,
            intermediate_cache1,
            self.w1_scale,
            self.quant_config.w1_zp,
            None,  # topk_weights
            sorted_token_ids,
            expert_ids,
            num_tokens_post_padded,
            False,  # mul_routed_weights
            top_k_num,
            config,
            compute_type=compute_type,
            use_int8_w8a16=self.quant_config.use_int8_w8a16,
            use_int4_w4a16=self.quant_config.use_int4_w4a16,
            block_shape=self.block_shape,
        )

        self.activation(
            activation, intermediate_cache2, intermediate_cache1.view(-1, N)
        )

        if _LING_W2_CAPTURE_DIR and num_tokens == 1:
            weight_key = w1.data_ptr()
            capture_ordinal = _LING_W2_CAPTURE_ORDINALS.get(weight_key)
            if capture_ordinal is None:
                capture_ordinal = len(_LING_W2_CAPTURE_ORDINALS)
                _LING_W2_CAPTURE_ORDINALS[weight_key] = capture_ordinal
            if capture_ordinal not in _LING_W2_CAPTURED_ORDINALS:
                _LING_W2_CAPTURED_ORDINALS.add(capture_ordinal)
                os.makedirs(_LING_W2_CAPTURE_DIR, exist_ok=True)
                selected_experts = topk_ids[0].to(torch.long)
                torch.save(
                    {
                        "ordinal": capture_ordinal,
                        "hidden": intermediate_cache2.detach().cpu(),
                        "topk_ids": topk_ids.detach().cpu(),
                        "topk_weights": topk_weights.detach().cpu(),
                        "w2": w2.index_select(0, selected_experts).detach().cpu(),
                        "w2_scale": self.w2_scale.index_select(
                            0, selected_experts
                        ).detach().cpu(),
                    },
                    os.path.join(
                        _LING_W2_CAPTURE_DIR,
                        f"w2-layer-{capture_ordinal:02d}.pt",
                    ),
                )
                logger.warning(
                    "LING_W2_CAPTURE saved routed-MoE ordinal %d",
                    capture_ordinal,
                )

        if ling_iu4_decode_eligible and _LING_IU4_DECODE_MODE in (
            "w2",
            "w2_residual2",
            "w2_residual3",
            "w2_residual3_outlier1",
            "w2_residual4",
        ):
            w2_operator = {
                "w2": torch.ops.ling_iu4_decode.w2,
                "w2_residual2": torch.ops.ling_iu4_decode.w2_residual2,
                "w2_residual3": torch.ops.ling_iu4_decode.w2_residual3,
                "w2_residual3_outlier1": (
                    torch.ops.ling_iu4_decode.w2_residual3_outlier1
                ),
                "w2_residual4": torch.ops.ling_iu4_decode.w2_residual4,
            }[_LING_IU4_DECODE_MODE]
            candidate = w2_operator(
                intermediate_cache2,
                w2.view(torch.int32),
                self.w2_scale,
                topk_ids,
                topk_weights,
            )
            output.copy_(candidate)
            return

        a2q_scale: torch.Tensor | None = None

        qintermediate_cache2, a2q_scale = moe_kernel_quantize_input(
            intermediate_cache2,
            a2_scale,
            self.quant_dtype,
            self.per_act_token_quant,
            self.block_shape,
        )

        invoke_fused_moe_wna16_triton_kernel(
            qintermediate_cache2,
            w2,
            intermediate_cache3,
            self.w2_scale,
            self.quant_config.w2_zp,
            topk_weights,
            sorted_token_ids,
            expert_ids,
            num_tokens_post_padded,
            not apply_router_weight_on_input,
            1,
            config,
            compute_type=compute_type,
            use_int8_w8a16=self.quant_config.use_int8_w8a16,
            use_int4_w4a16=self.quant_config.use_int4_w4a16,
            block_shape=self.block_shape,
        )

        # The native HIP top-k sum has very high fixed latency on gfx1151 for
        # the exact target-decode and K1/K2 verification shapes. Keep every
        # adjacent shape on the native implementation.
        if (
            use_naive_block_assignment
            and self._lora_context is None
            and intermediate_cache3.shape
            in ((1, 8, 2560), (2, 8, 2560), (3, 8, 2560))
            and output.shape in ((1, 2560), (2, 2560), (3, 2560))
            and output.shape[0] == intermediate_cache3.shape[0]
            and intermediate_cache3.is_contiguous()
            and output.is_contiguous()
        ):
            moe_sum_decode(intermediate_cache3, output)
        else:
            # separate function is required for MoE + LoRA
            self.moe_sum(intermediate_cache3, output)
