# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Triton-based MoE expert implementations."""

from collections.abc import Callable
import math
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
    kInt4Static32Asym,
    kInt4StaticAsym,
    kInt8DynamicTensorSym,
    kInt8DynamicTokenSym,
    kInt8Static,
    kInt8StaticChannelSym,
    kInt8StaticTensorSym,
)
from vllm.logger import init_logger
from vllm.platforms import current_platform
from vllm.triton_utils import tl
from vllm.utils.multi_stream_utils import maybe_execute_in_parallel


logger = init_logger(__name__)

_ENABLE_GLM_M1_DIRECT_ROUTE = os.getenv(
    "VLLM_ROCM_MOE_M1_DIRECT_ROUTE", "0"
).lower() not in {"0", "false", "no"}
_ENABLE_GLM_M8_SINGLE_LAUNCH_ALIGN = os.getenv(
    "VLLM_GLM53_M8_SINGLE_LAUNCH_ALIGN_ENABLED", "0"
).lower() not in {"0", "false", "no"}
_GLM_M8_SINGLE_LAUNCH_ALIGN_LIBRARY = os.getenv(
    "VLLM_GLM53_M8_SINGLE_LAUNCH_ALIGN_LIBRARY", ""
)
_GLM_M8_SINGLE_LAUNCH_ALIGN_LIBRARY_LOADED = False
_ENABLE_GLM_NATIVE_N32_KWORD_LAYOUT = os.getenv(
    "VLLM_ROCM_MOE_NATIVE_N32_KWORD_LAYOUT", "0"
).lower() not in {"0", "false", "no"}
_ENABLE_GLM_RESIDENT_G128 = os.getenv(
    "VLLM_GLM53_RESIDENT_G128_ENABLED", "0"
).lower() not in {"0", "false", "no"}
_GLM_RESIDENT_G128_LIBRARY = os.getenv(
    "VLLM_GLM53_RESIDENT_G128_LIBRARY", ""
)
_GLM_RESIDENT_G128_LIBRARY_LOADED = False
_ENABLE_GLM_M4_RESIDUAL_DOT8 = os.getenv(
    "VLLM_GLM53_M4_RESIDUAL_DOT8_ENABLED", "0"
).lower() not in {"0", "false", "no"}
_GLM_M4_RESIDUAL_DOT8_LIBRARY = os.getenv(
    "VLLM_GLM53_M4_RESIDUAL_DOT8_LIBRARY", ""
)
_GLM_M4_RESIDUAL_DOT8_LIBRARY_LOADED = False
_ENABLE_GLM_M8_RESIDUAL_DOT8 = os.getenv(
    "VLLM_GLM53_M8_RESIDUAL_DOT8_ENABLED", "0"
).lower() not in {"0", "false", "no"}
_GLM_M8_RESIDUAL_DOT8_LIBRARY = os.getenv(
    "VLLM_GLM53_M8_RESIDUAL_DOT8_LIBRARY", ""
)
_GLM_M8_RESIDUAL_DOT8_LIBRARY_LOADED = False
_ENABLE_GLM_EXL3_K3_DUAL20 = os.getenv(
    "VLLM_GLM53_EXL3_K3_DUAL20_ENABLED", "0"
).lower() not in {"0", "false", "no"}
_GLM_EXL3_K3_DUAL20_LIBRARY = os.getenv(
    "VLLM_GLM53_EXL3_K3_DUAL20_LIBRARY", ""
)
_GLM_EXL3_K3_DUAL20_PACKET_ROOT = os.getenv(
    "VLLM_GLM53_EXL3_K3_DUAL20_PACKET_ROOT", ""
)
_GLM_EXL3_K3_DUAL20_LAYERS = frozenset(
    int(value)
    for value in os.getenv(
        "VLLM_GLM53_EXL3_K3_DUAL20_LAYERS",
        "3,6,9,12,15,18,21,23,24,25,27,30,33,34,35,37,39,42,43,44",
    ).split(",")
    if value
)
_GLM_EXL3_K3_DUAL20_LIBRARY_LOADED = False
_GLM_EXL3_K3_NEXT_LAYER = 3


def _load_glm_m8_single_launch_align_library() -> None:
    global _GLM_M8_SINGLE_LAUNCH_ALIGN_LIBRARY_LOADED
    if _GLM_M8_SINGLE_LAUNCH_ALIGN_LIBRARY_LOADED:
        return
    if not _GLM_M8_SINGLE_LAUNCH_ALIGN_LIBRARY:
        raise RuntimeError(
            "VLLM_GLM53_M8_SINGLE_LAUNCH_ALIGN_LIBRARY is required"
        )
    torch.ops.load_library(_GLM_M8_SINGLE_LAUNCH_ALIGN_LIBRARY)
    try:
        torch.ops.glm53_m8_align.align
    except (AttributeError, RuntimeError) as exc:
        raise RuntimeError(
            "M8 alignment library did not register glm53_m8_align::align"
        ) from exc
    _GLM_M8_SINGLE_LAUNCH_ALIGN_LIBRARY_LOADED = True


def _load_glm_resident_g128_library() -> None:
    global _GLM_RESIDENT_G128_LIBRARY_LOADED
    if _GLM_RESIDENT_G128_LIBRARY_LOADED:
        return
    if not _GLM_RESIDENT_G128_LIBRARY:
        raise RuntimeError(
            "VLLM_GLM53_RESIDENT_G128_LIBRARY is required when the resident "
            "G128 MoE path is enabled"
        )
    torch.ops.load_library(_GLM_RESIDENT_G128_LIBRARY)
    try:
        torch.ops.glm53_resident_hybrid_g128.moe_out
    except AttributeError as exc:
        raise RuntimeError(
            "Resident G128 library did not register "
            "glm53_resident_hybrid_g128::moe_out"
        ) from exc
    _GLM_RESIDENT_G128_LIBRARY_LOADED = True


def _load_glm_m4_residual_dot8_library() -> None:
    global _GLM_M4_RESIDUAL_DOT8_LIBRARY_LOADED
    if _GLM_M4_RESIDUAL_DOT8_LIBRARY_LOADED:
        return
    if not _GLM_M4_RESIDUAL_DOT8_LIBRARY:
        raise RuntimeError(
            "VLLM_GLM53_M4_RESIDUAL_DOT8_LIBRARY is required when the M4 "
            "residual-dot8 MoE path is enabled"
        )
    torch.ops.load_library(_GLM_M4_RESIDUAL_DOT8_LIBRARY)
    try:
        torch.ops.glm53_m4_residual_dot8.moe_out
    except AttributeError as exc:
        raise RuntimeError(
            "M4 residual-dot8 library did not register "
            "glm53_m4_residual_dot8::moe_out"
        ) from exc
    _GLM_M4_RESIDUAL_DOT8_LIBRARY_LOADED = True


def _load_glm_m8_residual_dot8_library() -> None:
    global _GLM_M8_RESIDUAL_DOT8_LIBRARY_LOADED
    if _GLM_M8_RESIDUAL_DOT8_LIBRARY_LOADED:
        return
    if not _GLM_M8_RESIDUAL_DOT8_LIBRARY:
        raise RuntimeError(
            "VLLM_GLM53_M8_RESIDUAL_DOT8_LIBRARY is required when the M8 "
            "residual-dot8 MoE path is enabled"
        )
    torch.ops.load_library(_GLM_M8_RESIDUAL_DOT8_LIBRARY)
    try:
        torch.ops.glm53_m8_residual_dot8.moe_out
    except AttributeError as exc:
        raise RuntimeError(
            "M8 residual-dot8 library did not register "
            "glm53_m8_residual_dot8::moe_out"
        ) from exc
    _GLM_M8_RESIDUAL_DOT8_LIBRARY_LOADED = True


def _load_glm_exl3_k3_dual20_library() -> None:
    global _GLM_EXL3_K3_DUAL20_LIBRARY_LOADED
    if _GLM_EXL3_K3_DUAL20_LIBRARY_LOADED:
        return
    if not _GLM_EXL3_K3_DUAL20_LIBRARY:
        raise RuntimeError(
            "VLLM_GLM53_EXL3_K3_DUAL20_LIBRARY is required when K3 is enabled"
        )
    torch.ops.load_library(_GLM_EXL3_K3_DUAL20_LIBRARY)
    try:
        torch.ops.glm53_exl3_mcg_k3_dual20_m8.moe_out
    except AttributeError as exc:
        raise RuntimeError(
            "K3 library did not register "
            "glm53_exl3_mcg_k3_dual20_m8::moe_out"
        ) from exc
    _GLM_EXL3_K3_DUAL20_LIBRARY_LOADED = True


def _glm_exl3_k3_layer_for_instance(instance) -> int:
    layer = getattr(instance, "_glm53_exl3_k3_layer", None)
    if layer is not None:
        return layer
    global _GLM_EXL3_K3_NEXT_LAYER
    layer = _GLM_EXL3_K3_NEXT_LAYER
    _GLM_EXL3_K3_NEXT_LAYER += 1
    instance._glm53_exl3_k3_layer = layer
    return layer


def _load_glm_exl3_k3_packet(instance, layer: int, device: torch.device):
    retained = getattr(instance, "_glm53_exl3_k3_packet", None)
    if retained is not None:
        return retained
    if not _GLM_EXL3_K3_DUAL20_PACKET_ROOT:
        raise RuntimeError(
            "VLLM_GLM53_EXL3_K3_DUAL20_PACKET_ROOT is required when K3 is enabled"
        )
    from safetensors import safe_open

    rank = int(os.getenv("RANK", "0"))
    packet_path = os.path.join(
        _GLM_EXL3_K3_DUAL20_PACKET_ROOT,
        f"rank{rank}-layer{layer}-mcg-k3.safetensors",
    )
    names = (
        "w13.trellis",
        "w13.suh",
        "w13.svh",
        "w2.trellis",
        "w2.suh",
        "w2.svh",
    )
    with safe_open(packet_path, framework="pt", device=str(device)) as handle:
        weights = tuple(handle.get_tensor(name).contiguous() for name in names)
    scratch = (
        torch.empty((64,), dtype=torch.int32, device=device),
        torch.empty((64, 4096), dtype=torch.float16, device=device),
        torch.empty((64, 2048), dtype=torch.float16, device=device),
        torch.empty((64, 1024), dtype=torch.float16, device=device),
        torch.empty((64, 1024), dtype=torch.float16, device=device),
        torch.empty((64, 4096), dtype=torch.float16, device=device),
    )
    retained = (weights, scratch)
    instance._glm53_exl3_k3_packet = retained
    logger.info_once(
        "Loaded the GLM EXL3 MCG K3 M8 packet for rank %d layer %d.",
        rank,
        layer,
    )
    return retained


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
        self._moe_sum_override: (
            Callable[[torch.Tensor, torch.Tensor], bool] | None
        ) = None

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

        # SWIGLUOAI_UNINTERLEAVE routes to torch.ops._C.silu_and_mul_with_clamp
        # via apply_moe_activation() and requires clamp_limit to be set.
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
            torch.int8,
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
            or hidden_states.dtype == torch.int8
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

        # Include fused shared-expert rows while preserving EP remapping.
        num_align_experts = w1.shape[0] if expert_map is None else global_num_experts
        sorted_token_ids, expert_ids, num_tokens_post_padded = (
            _prepare_expert_assignment(
                topk_ids,
                config,
                num_tokens,
                top_k_num,
                num_align_experts,
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

    def set_moe_sum_override(
        self,
        override: Callable[[torch.Tensor, torch.Tensor], bool] | None,
    ) -> None:
        """Install an instance-scoped reducer that returns true when handled."""
        self._moe_sum_override = override

    def moe_sum(self, input: torch.Tensor, output: torch.Tensor) -> None:
        if self._moe_sum_override is not None and self._moe_sum_override(
            input, output
        ):
            return
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
            kInt4StaticAsym,
            kInt4Static32Asym,
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
            MoEActivation.SWIGLUOAI_UNINTERLEAVE,
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

    def moe_problem_size(
        self,
        a1: torch.Tensor,
        w1: torch.Tensor,
        w2: torch.Tensor,
        topk_ids: torch.Tensor,
    ) -> tuple[int, int, int, int, int]:
        if (
            _ENABLE_GLM_NATIVE_N32_KWORD_LAYOUT
            and self.quant_config.use_int4_w4a16
            and w1.ndim == 4
            and w2.ndim == 4
        ):
            E = w1.size(0)
            N = w1.size(1) * 32
            K = a1.size(-1)
            assert a1.dim() == 2 and topk_ids.size(0) == a1.size(0)
            assert topk_ids.dim() == 2
            return E, a1.size(0), N, K, topk_ids.size(1)
        return super().moe_problem_size(a1, w1, w2, topk_ids)

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
        use_native_n32_kword_layout = (
            _ENABLE_GLM_NATIVE_N32_KWORD_LAYOUT
            and self.quant_config.use_int4_w4a16
            and w1.ndim == 4
            and w2.ndim == 4
            and self.w1_scale.ndim == 4
            and self.w2_scale.ndim == 4
        )
        if use_native_n32_kword_layout:
            assert self.quant_config.w1_zp is None
            assert self.quant_config.w2_zp is None
            assert tuple(self.block_shape or ()) == (0, 128)
            assert w1.dtype == torch.int32 and w2.dtype == torch.int32
            assert w1.size(3) == 32 and w2.size(3) == 32
            assert self.w1_scale.size(3) == 32
            assert self.w2_scale.size(3) == 32
            assert hidden_states.size(-1) == w1.size(2) * 8
            assert w2.size(1) * 32 == hidden_states.size(-1)
            assert w1.size(0) == w2.size(0)
            E = w1.size(0)
            num_tokens = hidden_states.size(0)
            N = w1.size(1) * 32
            K = hidden_states.size(-1)
            top_k_num = topk_ids.size(1)
            config_w1_size = torch.Size((E, N, w1.size(2) * 4))
            config_w2_size = torch.Size(
                (E, w2.size(1) * 32, w2.size(2) * 4)
            )
        elif self.quant_config.use_int4_w4a16:
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

        if not use_native_n32_kword_layout:
            E, num_tokens, N, K, top_k_num = self.moe_problem_size(
                hidden_states, w1, w2, topk_ids
            )
            config_w1_size = w1.size()
            config_w2_size = w2.size()

        if global_num_experts == -1:
            global_num_experts = E

        config = try_get_optimal_moe_config(
            config_w1_size,
            config_w2_size,
            top_k_num,
            self.quant_config.config_name(hidden_states.dtype),
            num_tokens,
            block_shape=self.block_shape,
        )
        if use_native_n32_kword_layout:
            logger.info_once(
                "Using canonical N32/K-word G128 weights in Triton W4A16 MoE."
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
        activation_out_dim = self.adjust_N_for_activation(N, activation)
        intermediate_cache2 = _resize_cache(
            workspace13, (num_tokens * top_k_num, activation_out_dim)
        )
        intermediate_cache3 = _resize_cache(workspace2, (num_tokens, top_k_num, K))

        k3_target_shape = (
            _ENABLE_GLM_EXL3_K3_DUAL20
            and use_native_n32_kword_layout
            and E == 288
            and N == 2048
            and K == 4096
            and top_k_num == 8
            and hidden_states.dtype == torch.bfloat16
            and topk_ids.dtype == torch.int32
            and topk_weights.dtype == torch.float32
            and tuple(w1.shape[1:]) == (64, 512, 32)
            and tuple(w2.shape[1:]) == (128, 128, 32)
            and activation == MoEActivation.SILU
            and self.activation_config.clamp_limit == 10.0
            and expert_map is None
            and not apply_router_weight_on_input
        )
        if k3_target_shape:
            layer = _glm_exl3_k3_layer_for_instance(self)
            if layer in _GLM_EXL3_K3_DUAL20_LAYERS:
                _load_glm_exl3_k3_dual20_library()
                weights, scratch = _load_glm_exl3_k3_packet(
                    self, layer, hidden_states.device
                )
                if num_tokens == 8 and tuple(topk_ids.shape) == (8, 8):
                    torch.ops.glm53_exl3_mcg_k3_dual20_m8.moe_out(
                        hidden_states,
                        *weights,
                        topk_ids,
                        topk_weights,
                        *scratch,
                        output,
                    )
                    logger.info_once(
                        "Using GLM EXL3 MCG K3 for selected M8 routed layers."
                    )
                    return

        use_m8_residual_dot8 = (
            _ENABLE_GLM_M8_RESIDUAL_DOT8
            and use_native_n32_kword_layout
            and num_tokens == 8
            and E == 288
            and N == 2048
            and K == 4096
            and top_k_num == 8
            and hidden_states.dtype == torch.bfloat16
            and topk_ids.dtype == torch.int32
            and topk_weights.dtype == torch.float32
            and self.w1_scale.dtype == torch.bfloat16
            and self.w2_scale.dtype == torch.bfloat16
            and tuple(w1.shape[1:]) == (64, 512, 32)
            and tuple(w2.shape[1:]) == (128, 128, 32)
            and tuple(self.w1_scale.shape[1:]) == (64, 32, 32)
            and tuple(self.w2_scale.shape[1:]) == (128, 8, 32)
            and config["BLOCK_SIZE_M"] == 16
            and activation == MoEActivation.SILU
            and self.activation_config.clamp_limit == 10.0
            and expert_map is None
            and not apply_router_weight_on_input
        )
        if use_m8_residual_dot8:
            _load_glm_m8_residual_dot8_library()
            routes = num_tokens * top_k_num
            gate_up = intermediate_cache1.view(routes, N)
            hidden = intermediate_cache2.view(routes, activation_out_dim)
            routed = intermediate_cache3.view(routes, K)

            scratch_bytes = workspace13.view(torch.uint8).flatten()
            scratch_offset = hidden.numel() * hidden.element_size()

            def take_m8_scratch(shape: tuple[int, ...], dtype: torch.dtype):
                nonlocal scratch_offset
                numel = math.prod(shape)
                nbytes = numel * 4
                end = scratch_offset + nbytes
                if end > scratch_bytes.numel():
                    raise RuntimeError(
                        "M8 residual-dot8 scratch exceeds the modular MoE workspace"
                    )
                view = (
                    scratch_bytes.narrow(0, scratch_offset, nbytes)
                    .view(dtype)
                    .view(shape)
                )
                scratch_offset = end
                return view

            input_q0 = take_m8_scratch((num_tokens, K // 8), torch.int32)
            input_s0 = take_m8_scratch((num_tokens, K // 32), torch.float32)
            input_q1 = take_m8_scratch((num_tokens, K // 8), torch.int32)
            input_s1 = take_m8_scratch((num_tokens, K // 32), torch.float32)
            hidden_q0 = take_m8_scratch(
                (routes, activation_out_dim // 8), torch.int32
            )
            hidden_s0 = take_m8_scratch(
                (routes, activation_out_dim // 32), torch.float32
            )
            hidden_q1 = take_m8_scratch(
                (routes, activation_out_dim // 8), torch.int32
            )
            hidden_s1 = take_m8_scratch(
                (routes, activation_out_dim // 32), torch.float32
            )

            torch.ops.glm53_m8_residual_dot8.moe_out(
                hidden_states,
                w1,
                self.w1_scale,
                w2,
                self.w2_scale,
                topk_ids,
                topk_weights,
                input_q0,
                input_s0,
                input_q1,
                input_s1,
                gate_up,
                hidden,
                hidden_q0,
                hidden_s0,
                hidden_q1,
                hidden_s1,
                routed,
            )
            logger.info_once(
                "Using the GLM M8 residual-dot8 W13/W2 path."
            )
            self.moe_sum(intermediate_cache3, output)
            return

        use_m4_residual_dot8 = (
            _ENABLE_GLM_M4_RESIDUAL_DOT8
            and use_native_n32_kword_layout
            and num_tokens == 4
            and E == 288
            and N == 2048
            and K == 4096
            and top_k_num == 8
            and hidden_states.dtype == torch.bfloat16
            and topk_ids.dtype == torch.int32
            and topk_weights.dtype == torch.float32
            and self.w1_scale.dtype == torch.bfloat16
            and self.w2_scale.dtype == torch.bfloat16
            and tuple(w1.shape[1:]) == (64, 512, 32)
            and tuple(w2.shape[1:]) == (128, 128, 32)
            and tuple(self.w1_scale.shape[1:]) == (64, 32, 32)
            and tuple(self.w2_scale.shape[1:]) == (128, 8, 32)
            and config["BLOCK_SIZE_M"] == 16
            and activation == MoEActivation.SILU
            and self.activation_config.clamp_limit == 10.0
            and expert_map is None
            and not apply_router_weight_on_input
        )
        if use_m4_residual_dot8:
            _load_glm_m4_residual_dot8_library()
            routes = num_tokens * top_k_num
            gate_up = intermediate_cache1.view(routes, N)
            hidden = intermediate_cache2.view(routes, activation_out_dim)
            routed = intermediate_cache3.view(routes, K)

            scratch_bytes = workspace13.view(torch.uint8).flatten()
            scratch_offset = hidden.numel() * hidden.element_size()

            def take_m4_scratch(shape: tuple[int, ...], dtype: torch.dtype):
                nonlocal scratch_offset
                numel = math.prod(shape)
                nbytes = numel * 4
                end = scratch_offset + nbytes
                if end > scratch_bytes.numel():
                    raise RuntimeError(
                        "M4 residual-dot8 scratch exceeds the modular MoE workspace"
                    )
                view = (
                    scratch_bytes.narrow(0, scratch_offset, nbytes)
                    .view(dtype)
                    .view(shape)
                )
                scratch_offset = end
                return view

            input_q0 = take_m4_scratch((num_tokens, K // 8), torch.int32)
            input_s0 = take_m4_scratch((num_tokens, K // 32), torch.float32)
            input_q1 = take_m4_scratch((num_tokens, K // 8), torch.int32)
            input_s1 = take_m4_scratch((num_tokens, K // 32), torch.float32)
            hidden_q0 = take_m4_scratch(
                (routes, activation_out_dim // 8), torch.int32
            )
            hidden_s0 = take_m4_scratch(
                (routes, activation_out_dim // 32), torch.float32
            )
            hidden_q1 = take_m4_scratch(
                (routes, activation_out_dim // 8), torch.int32
            )
            hidden_s1 = take_m4_scratch(
                (routes, activation_out_dim // 32), torch.float32
            )

            torch.ops.glm53_m4_residual_dot8.moe_out(
                hidden_states,
                w1,
                self.w1_scale,
                w2,
                self.w2_scale,
                topk_ids,
                topk_weights,
                input_q0,
                input_s0,
                input_q1,
                input_s1,
                gate_up,
                hidden,
                hidden_q0,
                hidden_s0,
                hidden_q1,
                hidden_s1,
                routed,
            )
            logger.info_once(
                "Using the GLM M4 residual-dot8 W13/W2 path."
            )
            self.moe_sum(intermediate_cache3, output)
            return

        # A single top-k=8 token does not need an E-wide expert sort. Preserve
        # the promoted M=16 GEMM tile exactly: each selected expert receives one
        # route id followed by 15 sentinel ids. This removes only the sort/count
        # kernels, without changing the GEMM's numerical implementation.
        # Keep this experimental path narrowly gated until it has broader
        # upstream coverage; all other problems retain generic alignment.
        use_m1_direct_route = (
            _ENABLE_GLM_M1_DIRECT_ROUTE
            and num_tokens == 1
            and E == 288
            and N in (1024, 2048)
            and K == 4096
            and top_k_num == 8
            and hidden_states.dtype == torch.bfloat16
            and topk_ids.dtype == torch.int32
            and tuple(topk_ids.shape) == (1, 8)
            and config["BLOCK_SIZE_M"] == 16
            and self.quant_config.use_int4_w4a16
            and tuple(self.block_shape or ()) == (0, 128)
            and activation == MoEActivation.SILU
            and expert_map is None
        )
        if use_m1_direct_route:
            route_metadata = getattr(self, "_m1_direct_route_metadata", None)
            if route_metadata is None:
                block_size_m = config["BLOCK_SIZE_M"]
                sorted_token_ids = torch.full(
                    (top_k_num * block_size_m,),
                    top_k_num,
                    dtype=torch.int32,
                    device=topk_ids.device,
                )
                sorted_token_ids[::block_size_m] = torch.arange(
                    top_k_num, dtype=torch.int32, device=topk_ids.device
                )
                route_metadata = (
                    sorted_token_ids,
                    torch.tensor(
                        [top_k_num * block_size_m],
                        dtype=torch.int32,
                        device=topk_ids.device,
                    ),
                )
                self._m1_direct_route_metadata = route_metadata
            sorted_token_ids, num_tokens_post_padded = route_metadata
            expert_ids = topk_ids.reshape(-1)
            logger.info_once(
                "Using the GLM M=1 direct-route W4A16 MoE path."
            )
        elif (
            _ENABLE_GLM_M8_SINGLE_LAUNCH_ALIGN
            and use_native_n32_kword_layout
            and num_tokens == 8
            and E == 288
            and N == 2048
            and K == 4096
            and top_k_num == 8
            and hidden_states.dtype == torch.bfloat16
            and topk_ids.dtype == torch.int32
            and topk_ids.is_contiguous()
            and tuple(topk_ids.shape) == (8, 8)
            and config["BLOCK_SIZE_M"] == 16
            and self.quant_config.use_int4_w4a16
            and tuple(self.block_shape or ()) == (0, 128)
            and activation == MoEActivation.SILU
            and self.activation_config.clamp_limit == 10.0
            and expert_map is None
        ):
            # Fixed M8/top-k=8 emits the exact generic grouped metadata in one
            # launch. Both W4A16 GEMMs and all numerical epilogues are unchanged.
            _load_glm_m8_single_launch_align_library()
            route_metadata = getattr(
                self, "_m8_single_launch_align_metadata", None
            )
            if route_metadata is None:
                block_size_m = config["BLOCK_SIZE_M"]
                routes = num_tokens * top_k_num
                route_metadata = (
                    torch.empty(
                        routes * block_size_m,
                        dtype=torch.int32,
                        device=topk_ids.device,
                    ),
                    torch.empty(
                        routes, dtype=torch.int32, device=topk_ids.device
                    ),
                    torch.empty(
                        1, dtype=torch.int32, device=topk_ids.device
                    ),
                )
                self._m8_single_launch_align_metadata = route_metadata
            (
                sorted_token_ids,
                expert_ids,
                num_tokens_post_padded,
            ) = route_metadata
            torch.ops.glm53_m8_align.align(
                topk_ids,
                sorted_token_ids,
                expert_ids,
                num_tokens_post_padded,
            )
            logger.info_once(
                "Using the GLM M=8 single-launch W4A16 alignment path."
            )
        else:
            # Include fused shared-expert rows while preserving EP remapping.
            num_align_experts = (
                w1.shape[0] if expert_map is None else global_num_experts
            )
            sorted_token_ids, expert_ids, num_tokens_post_padded = (
                moe_align_block_size(
                    topk_ids,
                    config["BLOCK_SIZE_M"],
                    num_align_experts,
                    expert_map,
                )
            )

        use_resident_g128 = (
            _ENABLE_GLM_RESIDENT_G128
            and use_native_n32_kword_layout
            and num_tokens >= 64
            and E == 288
            and N == 2048
            and K == 4096
            and top_k_num == 8
            and hidden_states.dtype == torch.bfloat16
            and topk_ids.dtype == torch.int32
            and topk_weights.dtype == torch.float32
            and self.w1_scale.dtype == torch.bfloat16
            and self.w2_scale.dtype == torch.bfloat16
            and tuple(w1.shape[1:]) == (64, 512, 32)
            and tuple(w2.shape[1:]) == (128, 128, 32)
            and tuple(self.w1_scale.shape[1:]) == (64, 32, 32)
            and tuple(self.w2_scale.shape[1:]) == (128, 8, 32)
            and config["BLOCK_SIZE_M"] == 16
            and activation == MoEActivation.SILU
            and self.activation_config.clamp_limit == 10.0
            and expert_map is None
            and not apply_router_weight_on_input
        )
        if use_resident_g128:
            _load_glm_resident_g128_library()
            routes = num_tokens * top_k_num
            gate_up = intermediate_cache1.view(routes, N)
            hidden = intermediate_cache2.view(routes, activation_out_dim)
            routed = intermediate_cache3.view(routes, K)

            scratch_bytes = workspace13.view(torch.uint8).flatten()
            scratch_offset = hidden.numel() * hidden.element_size()

            def take_scratch(shape: tuple[int, ...], dtype: torch.dtype):
                nonlocal scratch_offset
                numel = math.prod(shape)
                nbytes = numel * 4  # All scratch views are int32 or fp32.
                end = scratch_offset + nbytes
                if end > scratch_bytes.numel():
                    raise RuntimeError(
                        "Resident G128 scratch exceeds the modular MoE workspace"
                    )
                view = (
                    scratch_bytes.narrow(0, scratch_offset, nbytes)
                    .view(dtype)
                    .view(shape)
                )
                scratch_offset = end
                return view

            input_q0 = take_scratch((num_tokens, K // 8), torch.int32)
            input_s0 = take_scratch((num_tokens, K // 32), torch.float32)
            input_q1 = take_scratch((num_tokens, K // 8), torch.int32)
            input_s1 = take_scratch((num_tokens, K // 32), torch.float32)
            hidden_q0 = take_scratch(
                (routes, activation_out_dim // 8), torch.int32
            )
            hidden_s0 = take_scratch(
                (routes, activation_out_dim // 32), torch.float32
            )
            hidden_q1 = take_scratch(
                (routes, activation_out_dim // 8), torch.int32
            )
            hidden_s1 = take_scratch(
                (routes, activation_out_dim // 32), torch.float32
            )
            queue_head = take_scratch((1,), torch.int32)

            torch.ops.glm53_resident_hybrid_g128.moe_out(
                hidden_states,
                w1,
                self.w1_scale,
                w2,
                self.w2_scale,
                topk_weights,
                sorted_token_ids,
                expert_ids,
                num_tokens_post_padded,
                input_q0,
                input_s0,
                input_q1,
                input_s1,
                gate_up,
                hidden,
                hidden_q0,
                hidden_s0,
                hidden_q1,
                hidden_s1,
                routed,
                queue_head,
            )
            logger.info_once(
                "Using the GLM resident G128 W13/W2 path for M>=64."
            )
            self.moe_sum(intermediate_cache3, output)
            return

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
            use_native_n32_kword_layout=use_native_n32_kword_layout,
        )

        self.activation(
            activation, intermediate_cache2, intermediate_cache1.view(-1, N)
        )

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
            use_native_n32_kword_layout=use_native_n32_kword_layout,
        )

        # separate function is required for MoE + LoRA
        self.moe_sum(intermediate_cache3, output)
