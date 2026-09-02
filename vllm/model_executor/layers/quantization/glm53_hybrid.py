# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""GLM-5.3 hybrid quantization for the gfx1151 F1 dense experiment.

The checkpoint remains a normal compressed-tensors W4A16 checkpoint.  This
config delegates every module to compressed-tensors except the fused KDA input
projection, which may use the separately forged S4/G32 + Scale8 packet for its
M=1..8 decode/verify shapes.  The feature is deliberately opt-in and retains
the BF16 weight as the fallback for all unsupported shapes.
"""

from __future__ import annotations

import json
import os
from functools import lru_cache
from pathlib import Path
import re
from typing import TYPE_CHECKING, Any

import torch

import vllm.envs as envs
from vllm.logger import init_logger
from vllm.model_executor.layers.fused_moe import (
    FusedMoEMethodBase,
    RoutedExperts,
    SharedExperts,
)
from vllm.model_executor.layers.linear import (
    LinearBase,
    LinearMethodBase,
    UnquantizedLinearMethod,
)
from vllm.model_executor.layers.fused_moe.experts.triton_moe import (
    TritonWNA16Experts,
)
from vllm.model_executor.layers.quantization.compressed_tensors.compressed_tensors import (  # noqa: E501
    CompressedTensorsConfig,
)
from vllm.model_executor.layers.quantization.compressed_tensors.compressed_tensors_moe.compressed_tensors_moe_wna16 import (  # noqa: E501
    CompressedTensorsWNA16MoEMethod,
)
from vllm.model_executor.layers.fused_moe.oracle.int_wna16 import (
    WNA16MoEBackend,
)
from vllm.model_executor.utils import replace_parameter
from .glm53_oproj_capture import capture_enabled, capture_input

if TYPE_CHECKING:
    from transformers import PretrainedConfig

    from vllm.model_executor.layers.quantization import QuantizationMethods
    from vllm.model_executor.layers.quantization.base_config import QuantizeMethodBase

logger = init_logger(__name__)

_F1_PREFIX_SUFFIX = ".self_attn.in_proj_qkvbfg_a"
_F1_SCHEMA = "glm53_hybrid_f1_dense_v1"
_F1_INPUT_SIZE = 4096
_F1_OUTPUT_SIZE = 12576
_F1_MAX_WMMA_M = 8
_F1_M1_WORKSPACE_BYTES = 5120
_F1_WMMA_WORKSPACE_BYTES_PER_ROW = 9216
_KDA_OPROJ_S4_LAYERS = frozenset(
    {0, 1, 2, 4, 5, 6, 8, 9, 10, 12, 13, 14, 16, 17, 18, 20, 21, 22,
     24, 25, 26, 28, 29, 30, 32, 33, 34, 36, 37, 38, 40, 41, 42, 44}
)
_KDA_OPROJ_RE = re.compile(r"\.layers\.(\d+)\.self_attn\.o_proj$")
_KDA_OPROJ_MAX_M = 8
_KDA_OPROJ_WORKSPACE_BYTES = (
    _KDA_OPROJ_MAX_M * _F1_WMMA_WORKSPACE_BYTES_PER_ROW
)
_KDA_OPROJ_W8_WORKSPACE_BYTES = _KDA_OPROJ_MAX_M * (
    4096 + (4096 // 32) * 4
)
_KDA_OPROJ_M4_WORKSPACE_BYTES = 4 * _F1_WMMA_WORKSPACE_BYTES_PER_ROW
_IU4_M1_LAYER_PANEL = frozenset({3, 9, 18, 27, 36, 42})
_IU4_CANONICAL_LAYERS = frozenset(range(3, 45))
_IU4_M1_LAYER_RE = re.compile(r"\.layers\.(\d+)\.mlp\.experts$")
_IU4_CANONICAL_LAYOUT = "glm53_iu4_n32_kword_g128_v1"


def _resolve_path(value: str, manifest_path: Path) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = manifest_path.parent / path
    return path.resolve()


@lru_cache(maxsize=None)
def _load_manifest(path_value: str) -> tuple[Path, dict[str, Any]]:
    manifest_path = Path(path_value).expanduser().resolve()
    with manifest_path.open(encoding="utf-8") as handle:
        manifest = json.load(handle)
    if manifest.get("schema") != _F1_SCHEMA:
        raise ValueError(
            "Unsupported GLM53 hybrid manifest schema: "
            f"{manifest.get('schema')!r}; expected {_F1_SCHEMA!r}"
        )
    if not isinstance(manifest.get("layers"), dict):
        raise ValueError("GLM53 hybrid manifest requires a 'layers' mapping")
    return manifest_path, manifest


@lru_cache(maxsize=None)
def _load_library(path_value: str) -> None:
    path = Path(path_value).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"GLM53 hybrid library not found: {path}")
    torch.ops.load_library(str(path))
    namespace = getattr(torch.ops, "glm53_f1_dense_kda", None)
    if namespace is None or not hasattr(namespace, "linear_out"):
        raise RuntimeError(
            "GLM53 hybrid library did not register glm53_f1_dense_kda::linear_out"
        )


def _kda_oproj_s4_enabled() -> bool:
    return os.getenv("VLLM_GLM53_KDA_OPROJ_S4_ENABLED", "").strip().lower() in {
        "1", "true", "yes", "on"
    }


@lru_cache(maxsize=None)
def _load_kda_oproj_s4_library(path_value: str) -> None:
    path = Path(path_value).expanduser().resolve()
    torch.ops.load_library(str(path))
    namespace = getattr(torch.ops, "glm53_kda_oproj_large_m", None)
    if namespace is None or not hasattr(namespace, "linear_out"):
        raise RuntimeError(
            "KDA o_proj S4 library did not register "
            "glm53_kda_oproj_large_m::linear_out"
        )


def _kda_oproj_w8_r2_enabled() -> bool:
    return os.getenv("VLLM_GLM53_KDA_OPROJ_W8A8R2_ENABLED", "").strip().lower() in {
        "1", "true", "yes", "on"
    }


def _kda_oproj_w8_enabled() -> bool:
    return _kda_oproj_w8_r2_enabled() or os.getenv(
        "VLLM_GLM53_KDA_OPROJ_W8A8_ENABLED", ""
    ).strip().lower() in {"1", "true", "yes", "on"}


@lru_cache(maxsize=None)
def _kda_oproj_w8_layers() -> frozenset[int]:
    value = os.getenv("VLLM_GLM53_KDA_OPROJ_W8A8_LAYERS", "").strip()
    if not value:
        return _KDA_OPROJ_S4_LAYERS
    return frozenset(int(item) for item in value.split(",") if item.strip())


@lru_cache(maxsize=None)
def _load_kda_oproj_w8_library(path_value: str) -> None:
    path = Path(path_value).expanduser().resolve()
    torch.ops.load_library(str(path))
    namespace = getattr(torch.ops, "glm53_kda_oproj_w8a8_m8", None)
    if namespace is None or not hasattr(namespace, "linear_out"):
        raise RuntimeError(
            "KDA o_proj W8 library did not register "
            "glm53_kda_oproj_w8a8_m8::linear_out"
        )


def _kda_oproj_m4_enabled() -> bool:
    return os.getenv("VLLM_GLM53_KDA_OPROJ_M4_ENABLED", "").strip().lower() in {
        "1", "true", "yes", "on"
    }


@lru_cache(maxsize=None)
def _load_kda_oproj_m4_library(path_value: str) -> None:
    path = Path(path_value).expanduser().resolve()
    torch.ops.load_library(str(path))
    namespace = getattr(torch.ops, "glm53_kda_oproj_m4", None)
    if namespace is None or not hasattr(namespace, "linear_out"):
        raise RuntimeError(
            "KDA o_proj M4 library did not register "
            "glm53_kda_oproj_m4::linear_out"
        )


def _set_buffer(layer: torch.nn.Module, name: str, value: torch.Tensor) -> None:
    if name in layer._buffers:
        layer._buffers[name] = value
    else:
        layer.register_buffer(name, value, persistent=False)


@lru_cache(maxsize=None)
def _load_iu4_m1_library(path_value: str) -> None:
    path = Path(path_value).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"GLM53 hybrid IU4 M1 library not found: {path}")
    torch.ops.load_library(str(path))


def _iu4_m1_op(name: str):
    namespace = getattr(torch.ops, "glm53_dot8_m1", None)
    if namespace is None or not hasattr(namespace, name):
        raise RuntimeError(
            "GLM53 hybrid IU4 library did not register "
            f"glm53_dot8_m1::{name}"
        )
    return getattr(namespace, name)


@lru_cache(maxsize=None)
def _load_iu4_r3_library(path_value: str) -> None:
    path = Path(path_value).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"GLM53 hybrid IU4 R3 library not found: {path}")
    torch.ops.load_library(str(path))
    namespace = getattr(torch.ops, "glm53_dot8_r3", None)
    if namespace is None or not hasattr(namespace, "moe_r3"):
        raise RuntimeError(
            "GLM53 hybrid IU4 R3 library did not register "
            "glm53_dot8_r3::moe_r3"
        )


@lru_cache(maxsize=None)
def _load_top8_epilogue_library(path_value: str) -> None:
    path = Path(path_value).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"GLM53 top8 epilogue library not found: {path}")
    torch.ops.load_library(str(path))
    namespace = getattr(torch.ops, "glm53_top8_epilogue", None)
    if namespace is None or not hasattr(namespace, "sum_out"):
        raise RuntimeError(
            "GLM53 top8 epilogue library did not register "
            "glm53_top8_epilogue::sum_out"
        )


def _try_glm53_top8_epilogue(
    input: torch.Tensor,
    output: torch.Tensor,
) -> bool:
    """Handle only canonical GLM prefill; false preserves stock moe_sum."""
    if (
        input.is_cuda
        and output.is_cuda
        and input.dtype == torch.bfloat16
        and output.dtype == torch.bfloat16
        and input.dim() == 3
        and output.dim() == 2
        and input.shape[0] > 1
        and tuple(input.shape[1:]) == (8, 4096)
        and tuple(output.shape) == (input.shape[0], 4096)
        and input.device == output.device
        and input.is_contiguous()
        and output.is_contiguous()
    ):
        torch.ops.glm53_top8_epilogue.sum_out(input, output)
        return True
    return False


def _retile_iu4_weight(packed_u8: torch.Tensor) -> torch.Tensor:
    experts, outputs, packed_bytes = packed_u8.shape
    return (
        packed_u8.contiguous()
        .view(torch.int32)
        .view(experts, outputs // 32, 32, packed_bytes // 4)
        .permute(0, 1, 3, 2)
        .contiguous()
    )


def _retile_iu4_scale(scales: torch.Tensor) -> torch.Tensor:
    experts, outputs, groups = scales.shape
    return (
        scales.view(experts, outputs // 32, 32, groups)
        .permute(0, 1, 3, 2)
        .contiguous()
    )


class Glm53Iu4M1MoEMethod(FusedMoEMethodBase):
    """Native gfx1151 top-8 M=1 routed experts with Triton fallback."""

    def __init__(self, fallback: CompressedTensorsWNA16MoEMethod, prefix: str):
        super().__init__(fallback.moe)
        self.fallback = fallback
        self.prefix = prefix

    def create_weights(self, *args, **kwargs):
        return self.fallback.create_weights(*args, **kwargs)

    def maybe_roundup_sizes(self, *args, **kwargs):
        return self.fallback.maybe_roundup_sizes(*args, **kwargs)

    def uses_weight_scale_2_pattern(self) -> bool:
        return self.fallback.uses_weight_scale_2_pattern()

    def get_fused_moe_quant_config(self, layer: RoutedExperts):
        return self.fallback.get_fused_moe_quant_config(layer)

    @property
    def supports_eplb(self) -> bool:
        return self.fallback.supports_eplb

    @property
    def is_monolithic(self) -> bool:
        return self.fallback.is_monolithic

    @property
    def supports_internal_mk(self) -> bool:
        return self.fallback.supports_internal_mk

    @property
    def mk_can_overlap_shared_experts(self) -> bool:
        return self.fallback.mk_can_overlap_shared_experts

    @property
    def topk_indices_dtype(self) -> torch.dtype | None:
        return torch.int32

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        self.fallback.process_weights_after_loading(layer)
        self.moe_quant_config = self.fallback.moe_quant_config
        self.moe_kernel = self.fallback.moe_kernel
        if self.fallback.wna16_backend != WNA16MoEBackend.TRITON:
            raise RuntimeError("GLM53 hybrid IU4 M1 requires the Triton W4A16 fallback")
        library = envs.VLLM_GLM53_HYBRID_IU4_M1_LIBRARY
        if not library:
            raise ValueError(
                "VLLM_GLM53_HYBRID_IU4_M1_LIBRARY is required when the IU4 M1 "
                "path is enabled"
            )
        _load_iu4_m1_library(library)
        _iu4_m1_op("moe_r2")
        with torch.no_grad():
            _set_buffer(
                layer,
                "iu4_m1_w13_weight",
                _retile_iu4_weight(layer.w13_weight_packed),
            )
            _set_buffer(
                layer,
                "iu4_m1_w2_weight",
                _retile_iu4_weight(layer.w2_weight_packed),
            )
            _set_buffer(
                layer,
                "iu4_m1_w13_scale",
                _retile_iu4_scale(layer.w13_weight_scale),
            )
            _set_buffer(
                layer,
                "iu4_m1_w2_scale",
                _retile_iu4_scale(layer.w2_weight_scale),
            )
        logger.info("GLM53 hybrid IU4 M1 routed panel armed for %s", self.prefix)

    def apply(
        self,
        layer: RoutedExperts,
        x: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        shared_experts: SharedExperts | None,
        shared_experts_input: torch.Tensor | None,
    ) -> torch.Tensor:
        if (
            envs.VLLM_GLM53_HYBRID_IU4_M1_ENABLED
            and x.dtype == torch.bfloat16
            and tuple(x.shape) == (1, 4096)
            and x.is_contiguous()
            and layer.expert_map is None
            and not layer.apply_router_weight_on_input
        ):
            return _iu4_m1_op("moe_r2")(
                x,
                layer.iu4_m1_w13_weight,
                layer.iu4_m1_w13_scale,
                layer.iu4_m1_w2_weight,
                layer.iu4_m1_w2_scale,
                topk_ids,
                topk_weights,
            )
        return self.fallback.apply(
            layer,
            x,
            topk_weights,
            topk_ids,
            shared_experts,
            shared_experts_input,
        )


class Glm53Iu4CanonicalMoEMethod(Glm53Iu4M1MoEMethod):
    """Single-copy native IU4 layout for every routed GLM expert layer.

    Checkpoint loading and compressed-tensors conversion complete first. The
    four converted expert tensors are then replaced, one at a time, by their
    equal-byte N32/K-word-major forms. No compressed-tensors execution path is
    retained because its Triton consumer cannot read this layout.
    """

    @property
    def supports_eplb(self) -> bool:
        return False

    @property
    def supports_internal_mk(self) -> bool:
        return False

    @property
    def mk_can_overlap_shared_experts(self) -> bool:
        return False

    def _arm_canonical_runtime(self, layer: torch.nn.Module) -> None:
        if self.fallback.wna16_backend != WNA16MoEBackend.TRITON:
            raise RuntimeError(
                "GLM53 canonical IU4 layout requires the Triton W4A16 "
                "checkpoint conversion"
            )
        if not envs.VLLM_ROCM_MOE_NATIVE_N32_KWORD_LAYOUT:
            raise ValueError(
                "VLLM_ROCM_MOE_NATIVE_N32_KWORD_LAYOUT=1 is required when "
                "the canonical IU4 layout is enabled"
            )
        library = envs.VLLM_GLM53_HYBRID_IU4_CANONICAL_M1_LIBRARY
        if not library:
            raise ValueError(
                "VLLM_GLM53_HYBRID_IU4_CANONICAL_M1_LIBRARY is required "
                "when the canonical IU4 layout is enabled"
            )
        _load_iu4_r3_library(library)

        self.fallback.moe_kernel = None
        self.fallback.moe_quant_config = None
        self.moe_kernel = None
        self.moe_quant_config = None
        layer.w13_weight = layer.w13_weight_packed
        layer.w2_weight = layer.w2_weight_packed
        layer.iu4_weight_layout = _IU4_CANONICAL_LAYOUT
        self.fallback._setup_kernel(layer)
        self.moe_quant_config = self.fallback.moe_quant_config
        self.moe_kernel = self.fallback.moe_kernel

        if envs.VLLM_GLM53_HYBRID_TOP8_EPILOGUE_ENABLED:
            library = envs.VLLM_GLM53_HYBRID_TOP8_EPILOGUE_LIBRARY
            if not library:
                raise ValueError(
                    "VLLM_GLM53_HYBRID_TOP8_EPILOGUE_LIBRARY is required "
                    "when the top8 epilogue is enabled"
                )
            _load_top8_epilogue_library(library)
            if self.moe_kernel is None:
                raise RuntimeError("GLM53 canonical WNA16 kernel was not created")
            experts = self.moe_kernel.fused_experts
            if not isinstance(experts, TritonWNA16Experts):
                raise RuntimeError(
                    "GLM53 top8 epilogue requires TritonWNA16Experts; got "
                    f"{type(experts).__name__}"
                )
            experts.set_moe_sum_override(_try_glm53_top8_epilogue)
            logger.info(
                "GLM53 Wave32 fixed-top8 prefill epilogue armed for %s",
                self.prefix,
            )
        source = (
            "runtime prepack"
            if getattr(layer, "_vllm_runtime_prepacked", False)
            else "checkpoint"
        )
        logger.info(
            "GLM53 canonical single-copy IU4 layout armed for %s from %s",
            self.prefix,
            source,
        )

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        if getattr(layer, "_vllm_runtime_prepacked", False):
            self._arm_canonical_runtime(layer)
            return

        self.fallback.process_weights_after_loading(layer)

        # The fallback kernel/config retain the old scale tensors. Drop those
        # references and the packed-weight aliases before replacing storage.
        self.fallback.moe_kernel = None
        self.fallback.moe_quant_config = None
        self.moe_kernel = None
        self.moe_quant_config = None
        layer.w13_weight = None
        layer.w2_weight = None

        with torch.no_grad():
            replace_parameter(
                layer,
                "w13_weight_packed",
                _retile_iu4_weight(layer.w13_weight_packed),
            )
            replace_parameter(
                layer,
                "w2_weight_packed",
                _retile_iu4_weight(layer.w2_weight_packed),
            )
            replace_parameter(
                layer,
                "w13_weight_scale",
                _retile_iu4_scale(layer.w13_weight_scale),
            )
            replace_parameter(
                layer,
                "w2_weight_scale",
                _retile_iu4_scale(layer.w2_weight_scale),
            )
        self._arm_canonical_runtime(layer)

    def apply(
        self,
        layer: RoutedExperts,
        x: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        shared_experts: SharedExperts | None,
        shared_experts_input: torch.Tensor | None,
    ) -> torch.Tensor:
        if (
            x.dtype != torch.bfloat16
            or x.dim() != 2
            or x.shape[1] != 4096
            or not x.is_contiguous()
            or layer.expert_map is not None
            or layer.apply_router_weight_on_input
        ):
            raise RuntimeError(
                "GLM53 canonical IU4 consumer requires contiguous BF16 "
                "[M,4096], no EP map, and output-side router weights"
            )

        arguments = (
            x,
            layer.w13_weight_packed,
            layer.w13_weight_scale,
            layer.w2_weight_packed,
            layer.w2_weight_scale,
            topk_ids,
            topk_weights,
        )
        if x.shape[0] == 1:
            return torch.ops.glm53_dot8_r3.moe_r3(*arguments)
        return self.fallback.apply(
            layer,
            x,
            topk_weights,
            topk_ids,
            shared_experts,
            shared_experts_input,
        )


class Glm53F1DenseLinearMethod(LinearMethodBase):
    """M=1..8 F1 KDA projection with the current BF16 method as fallback."""

    def __init__(self, prefix: str) -> None:
        self.prefix = prefix
        self.fallback = UnquantizedLinearMethod()
        self.packet_path: Path | None = None
        self.max_m = 0

    def create_weights(
        self,
        layer: torch.nn.Module,
        input_size_per_partition: int,
        output_partition_sizes: list[int],
        input_size: int,
        output_size: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs,
    ) -> None:
        local_output_size = sum(output_partition_sizes)
        if (
            input_size_per_partition != _F1_INPUT_SIZE
            or input_size != _F1_INPUT_SIZE
            or local_output_size != _F1_OUTPUT_SIZE
            or params_dtype != torch.bfloat16
        ):
            raise ValueError(
                "GLM53 F1 dense currently supports only BF16 at the TP1 KDA "
                f"shape [N,K]=[{_F1_OUTPUT_SIZE},{_F1_INPUT_SIZE}], got "
                f"{params_dtype} [{local_output_size},{input_size_per_partition}]"
            )
        self.fallback.create_weights(
            layer=layer,
            input_size_per_partition=input_size_per_partition,
            output_partition_sizes=output_partition_sizes,
            input_size=input_size,
            output_size=output_size,
            params_dtype=params_dtype,
            **extra_weight_attrs,
        )

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        self.fallback.process_weights_after_loading(layer)

        if getattr(layer, "_vllm_runtime_prepacked", False):
            library_value = envs.VLLM_GLM53_HYBRID_LIBRARY
            if not library_value:
                raise ValueError(
                    "VLLM_GLM53_HYBRID_LIBRARY is required for the GLM53 "
                    "runtime prepack"
                )
            _load_library(library_value)
            required = {
                "f1_weight_s4_dot_i32",
                "f1_scale8_codes_dot",
                "f1_scale8_lut",
            }
            missing = {name for name in required if not hasattr(layer, name)}
            if missing:
                raise ValueError(
                    f"GLM53 runtime prepack is missing F1 buffers {sorted(missing)} "
                    f"for {self.prefix!r}"
                )
            max_m = envs.VLLM_GLM53_HYBRID_MAX_M
            if not 1 <= max_m <= _F1_MAX_WMMA_M:
                raise ValueError(
                    f"VLLM_GLM53_HYBRID_MAX_M must be in [1,8], got {max_m}"
                )
            workspace_bytes = (
                _F1_M1_WORKSPACE_BYTES
                if max_m == 1
                else max_m * _F1_WMMA_WORKSPACE_BYTES_PER_ROW
            )
            device = layer.weight.device
            _set_buffer(
                layer,
                "f1_output_buffer",
                torch.empty(
                    (max_m, _F1_OUTPUT_SIZE),
                    dtype=torch.bfloat16,
                    device=device,
                ),
            )
            _set_buffer(
                layer,
                "f1_workspace",
                torch.empty(workspace_bytes, dtype=torch.uint8, device=device),
            )
            self.packet_path = Path("<glm53-runtime-prepack-v1>")
            self.max_m = max_m
            logger.info_once(
                "GLM53 hybrid F1 dense path armed for %s from runtime prepack.",
                self.prefix,
            )
            return

        manifest_value = envs.VLLM_GLM53_HYBRID_MANIFEST
        if not manifest_value:
            raise ValueError(
                "VLLM_GLM53_HYBRID_MANIFEST is required when "
                "VLLM_GLM53_HYBRID_F1_ENABLED=1"
            )
        manifest_path, manifest = _load_manifest(manifest_value)

        layer_entry = manifest["layers"].get(self.prefix)
        if not isinstance(layer_entry, dict):
            raise KeyError(
                f"GLM53 hybrid manifest has no exact layer entry for {self.prefix!r}"
            )
        ranks = layer_entry.get("ranks")
        if not isinstance(ranks, dict):
            raise ValueError(
                f"GLM53 hybrid layer {self.prefix!r} requires a 'ranks' mapping"
            )
        tp_rank = getattr(layer, "tp_rank", None)
        if tp_rank is None:
            from vllm.distributed import get_tensor_model_parallel_rank

            tp_rank = get_tensor_model_parallel_rank()
        rank = str(tp_rank)
        packet_entry = ranks.get(rank)
        if not isinstance(packet_entry, dict):
            raise KeyError(
                f"GLM53 hybrid manifest has no TP rank {rank} packet for "
                f"{self.prefix!r}"
            )

        library_value = envs.VLLM_GLM53_HYBRID_LIBRARY or manifest.get("library")
        if not library_value:
            raise ValueError(
                "GLM53 hybrid manifest requires 'library' unless "
                "VLLM_GLM53_HYBRID_LIBRARY is set"
            )
        library_path = _resolve_path(str(library_value), manifest_path)
        _load_library(str(library_path))

        packet_value = packet_entry.get("packet")
        if not packet_value:
            raise ValueError(
                f"GLM53 hybrid packet entry for {self.prefix!r} rank {rank} "
                "requires 'packet'"
            )
        packet_path = _resolve_path(str(packet_value), manifest_path)
        if not packet_path.is_file():
            raise FileNotFoundError(f"GLM53 hybrid packet not found: {packet_path}")

        from safetensors.torch import load_file

        packet = load_file(str(packet_path), device="cpu")
        required = {
            "weight_s4_dot_i32",
            "scale8_codes_dot",
            "scale8_offset_i16",
        }
        missing = required.difference(packet)
        if missing:
            raise ValueError(
                f"GLM53 hybrid packet {packet_path} is missing {sorted(missing)}"
            )

        weight = packet["weight_s4_dot_i32"].contiguous()
        scale_codes = packet["scale8_codes_dot"].contiguous()
        offset_tensor = packet["scale8_offset_i16"].contiguous()
        expected_weight_shape = (
            _F1_OUTPUT_SIZE // 32,
            _F1_INPUT_SIZE // 8,
            32,
        )
        expected_scale_shape = (
            _F1_OUTPUT_SIZE // 32,
            _F1_INPUT_SIZE // 32,
            32,
        )
        if weight.dtype != torch.int32 or tuple(weight.shape) != expected_weight_shape:
            raise ValueError(
                "GLM53 hybrid weight must be int32 "
                f"{expected_weight_shape}, got {weight.dtype} {tuple(weight.shape)}"
            )
        if (
            scale_codes.dtype != torch.uint8
            or tuple(scale_codes.shape) != expected_scale_shape
        ):
            raise ValueError(
                "GLM53 hybrid Scale8 codes must be uint8 "
                f"{expected_scale_shape}, got "
                f"{scale_codes.dtype} {tuple(scale_codes.shape)}"
            )
        if offset_tensor.dtype != torch.int16 or offset_tensor.numel() != 1:
            raise ValueError(
                "GLM53 hybrid Scale8 offset must be one int16 value, got "
                f"{offset_tensor.dtype} {tuple(offset_tensor.shape)}"
            )

        device = layer.weight.device
        offset = int(offset_tensor.item())
        lut = torch.zeros(256, dtype=torch.float32)
        indices = torch.arange(1, 256, dtype=torch.float32)
        lut[1:] = torch.exp2((indices - 1.0 + float(offset)) / 16.0)
        _set_buffer(layer, "f1_weight_s4_dot_i32", weight.to(device))
        _set_buffer(layer, "f1_scale8_codes_dot", scale_codes.to(device))
        _set_buffer(layer, "f1_scale8_lut", lut.to(device))
        max_m = envs.VLLM_GLM53_HYBRID_MAX_M
        if not 1 <= max_m <= _F1_MAX_WMMA_M:
            raise ValueError(f"VLLM_GLM53_HYBRID_MAX_M must be in [1,8], got {max_m}")
        workspace_bytes = (
            _F1_M1_WORKSPACE_BYTES
            if max_m == 1
            else max_m * _F1_WMMA_WORKSPACE_BYTES_PER_ROW
        )
        _set_buffer(
            layer,
            "f1_output_buffer",
            torch.empty((max_m, _F1_OUTPUT_SIZE), dtype=torch.bfloat16, device=device),
        )
        _set_buffer(
            layer,
            "f1_workspace",
            torch.empty(workspace_bytes, dtype=torch.uint8, device=device),
        )
        self.packet_path = packet_path
        self.max_m = max_m
        logger.info_once(
            "GLM53 hybrid F1 dense path armed for %s (TP rank %s).",
            self.prefix,
            rank,
        )

    def apply(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if (
            envs.VLLM_GLM53_HYBRID_F1_ENABLED
            and x.dtype == torch.bfloat16
            and x.ndim == 2
            and 1 <= x.shape[0] <= self.max_m
            and x.shape[1] == _F1_INPUT_SIZE
            and x.is_contiguous()
        ):
            if self.packet_path is None:
                raise RuntimeError(
                    f"GLM53 hybrid F1 packet was not loaded for {self.prefix!r}"
                )
            output = layer.f1_output_buffer[: x.shape[0]]
            output = torch.ops.glm53_f1_dense_kda.linear_out(
                x,
                layer.f1_weight_s4_dot_i32,
                layer.f1_scale8_codes_dot,
                layer.f1_scale8_lut,
                output,
                layer.f1_workspace,
            )
            if bias is not None:
                output.add_(bias)
            return output
        return self.fallback.apply(layer, x, bias)


class Glm53KdaOProjCaptureLinearMethod(LinearMethodBase):
    """BF16 OProj fallback plus an exact-M8 calibration capture."""

    def __init__(self, prefix: str, layer_index: int) -> None:
        self.prefix = prefix
        self.layer_index = layer_index
        self.fallback = UnquantizedLinearMethod()

    def create_weights(
        self,
        layer: torch.nn.Module,
        input_size_per_partition: int,
        output_partition_sizes: list[int],
        input_size: int,
        output_size: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs,
    ) -> None:
        self.fallback.create_weights(
            layer=layer,
            input_size_per_partition=input_size_per_partition,
            output_partition_sizes=output_partition_sizes,
            input_size=input_size,
            output_size=output_size,
            params_dtype=params_dtype,
            **extra_weight_attrs,
        )

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        self.fallback.process_weights_after_loading(layer)

    def apply(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        capture_input(self.layer_index, x)
        return self.fallback.apply(layer, x, bias)


class Glm53KdaOProjW8LinearMethod(LinearMethodBase):
    """Exact-M8 W8A8/G32+Scale8 KDA output projection with BF16 fallback."""

    def __init__(self, prefix: str, layer_index: int) -> None:
        self.prefix = prefix
        self.layer_index = layer_index
        self.activation_residual_planes = 2 if _kda_oproj_w8_r2_enabled() else 1
        self.fallback = UnquantizedLinearMethod()

    def create_weights(
        self,
        layer: torch.nn.Module,
        input_size_per_partition: int,
        output_partition_sizes: list[int],
        input_size: int,
        output_size: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs,
    ) -> None:
        if (
            input_size_per_partition != 4096
            or sum(output_partition_sizes) != 4096
            or params_dtype != torch.bfloat16
        ):
            raise ValueError(
                "KDA o_proj W8 requires a BF16 rank-local [4096,4096] weight"
            )
        self.fallback.create_weights(
            layer=layer,
            input_size_per_partition=input_size_per_partition,
            output_partition_sizes=output_partition_sizes,
            input_size=input_size,
            output_size=output_size,
            params_dtype=params_dtype,
            **extra_weight_attrs,
        )

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        self.fallback.process_weights_after_loading(layer)
        library_name = (
            "VLLM_GLM53_KDA_OPROJ_W8A8R2_LIBRARY"
            if self.activation_residual_planes == 2
            else "VLLM_GLM53_KDA_OPROJ_W8A8_LIBRARY"
        )
        library = os.getenv(library_name, "").strip()
        packet_root = os.getenv(
            "VLLM_GLM53_KDA_OPROJ_W8A8_PACKET_ROOT", ""
        ).strip()
        if not library or not packet_root:
            raise ValueError(
                "KDA o_proj W8A8 requires ..._LIBRARY and ..._PACKET_ROOT"
            )
        _load_kda_oproj_w8_library(library)
        if (
            self.activation_residual_planes == 2
            and not hasattr(torch.ops.glm53_kda_oproj_w8a8_m8, "linear_out_r2")
        ):
            raise RuntimeError("KDA o_proj W8 R2 library did not register linear_out_r2")
        from vllm.distributed import get_tensor_model_parallel_rank
        from safetensors.torch import load_file

        rank = get_tensor_model_parallel_rank()
        packet_path = (
            Path(packet_root).expanduser().resolve()
            / f"rank{rank}"
            / f"layer{self.layer_index:02d}-kda-oproj-w8g32-scale8.safetensors"
        )
        packet = load_file(str(packet_path), device="cpu")
        weight = packet["weight_s8_dot_i32"].contiguous()
        codes = packet["scale8_codes_dot"].contiguous()
        offset = int(packet["scale8_offset_i16"].item())
        if weight.dtype != torch.int32 or tuple(weight.shape) != (128, 1024, 32):
            raise ValueError(f"invalid o_proj W8 weight packet {packet_path}")
        if codes.dtype != torch.uint8 or tuple(codes.shape) != (128, 128, 32):
            raise ValueError(f"invalid o_proj W8 Scale8 packet {packet_path}")

        device = layer.weight.device
        lut = torch.zeros(256, dtype=torch.float32)
        indices = torch.arange(1, 256, dtype=torch.float32)
        lut[1:] = torch.exp2((indices - 1.0 + float(offset)) / 16.0)
        _set_buffer(layer, "oproj_w8_weight_s8_dot_i32", weight.to(device))
        _set_buffer(layer, "oproj_w8_scale8_codes_dot", codes.to(device))
        _set_buffer(layer, "oproj_w8_scale8_lut", lut.to(device))
        _set_buffer(
            layer,
            "oproj_w8_output_buffer",
            torch.empty(
                (_KDA_OPROJ_MAX_M, 4096),
                dtype=torch.bfloat16,
                device=device,
            ),
        )
        _set_buffer(
            layer,
            "oproj_w8_workspace",
            torch.empty(
                self.activation_residual_planes * _KDA_OPROJ_W8_WORKSPACE_BYTES,
                dtype=torch.uint8,
                device=device,
            ),
        )

    def apply(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if (
            x.dtype == torch.bfloat16
            and x.ndim == 2
            and tuple(x.shape) == (8, 4096)
            and x.is_contiguous()
        ):
            op = (
                torch.ops.glm53_kda_oproj_w8a8_m8.linear_out_r2
                if self.activation_residual_planes == 2
                else torch.ops.glm53_kda_oproj_w8a8_m8.linear_out
            )
            output = op(
                x,
                layer.oproj_w8_weight_s8_dot_i32,
                layer.oproj_w8_scale8_codes_dot,
                layer.oproj_w8_scale8_lut,
                layer.oproj_w8_output_buffer,
                layer.oproj_w8_workspace,
            )
            if bias is not None:
                output.add_(bias)
            return output
        return self.fallback.apply(layer, x, bias)


class Glm53KdaOProjS4LinearMethod(LinearMethodBase):
    """M8 S4+Scale8 KDA output projection with BF16 fallback."""

    def __init__(self, prefix: str, layer_index: int) -> None:
        self.prefix = prefix
        self.layer_index = layer_index
        self.fallback = UnquantizedLinearMethod()

    def create_weights(
        self,
        layer: torch.nn.Module,
        input_size_per_partition: int,
        output_partition_sizes: list[int],
        input_size: int,
        output_size: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs,
    ) -> None:
        if (
            input_size_per_partition != 4096
            or sum(output_partition_sizes) != 4096
            or params_dtype != torch.bfloat16
        ):
            raise ValueError(
                "KDA o_proj S4 requires a BF16 rank-local [4096,4096] weight"
            )
        self.fallback.create_weights(
            layer=layer,
            input_size_per_partition=input_size_per_partition,
            output_partition_sizes=output_partition_sizes,
            input_size=input_size,
            output_size=output_size,
            params_dtype=params_dtype,
            **extra_weight_attrs,
        )

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        self.fallback.process_weights_after_loading(layer)
        library = os.getenv("VLLM_GLM53_KDA_OPROJ_S4_LIBRARY", "").strip()
        packet_root = os.getenv("VLLM_GLM53_KDA_OPROJ_S4_PACKET_ROOT", "").strip()
        if not library or not packet_root:
            raise ValueError(
                "KDA o_proj S4 requires ..._LIBRARY and ..._PACKET_ROOT"
            )
        _load_kda_oproj_s4_library(library)
        from vllm.distributed import get_tensor_model_parallel_rank
        from safetensors.torch import load_file

        rank = get_tensor_model_parallel_rank()
        packet_path = (
            Path(packet_root).expanduser().resolve()
            / f"rank{rank}"
            / f"layer{self.layer_index:02d}-kda-oproj-s4g32-scale8.safetensors"
        )
        packet = load_file(str(packet_path), device="cpu")
        weight = packet["weight_s4_dot_i32"].contiguous()
        codes = packet["scale8_codes_dot"].contiguous()
        offset = int(packet["scale8_offset_i16"].item())
        if weight.dtype != torch.int32 or tuple(weight.shape) != (128, 512, 32):
            raise ValueError(f"invalid o_proj weight packet {packet_path}")
        if codes.dtype != torch.uint8 or tuple(codes.shape) != (128, 128, 32):
            raise ValueError(f"invalid o_proj Scale8 packet {packet_path}")

        device = layer.weight.device
        lut = torch.zeros(256, dtype=torch.float32)
        indices = torch.arange(1, 256, dtype=torch.float32)
        lut[1:] = torch.exp2((indices - 1.0 + float(offset)) / 16.0)
        _set_buffer(layer, "oproj_weight_s4_dot_i32", weight.to(device))
        _set_buffer(layer, "oproj_scale8_codes_dot", codes.to(device))
        _set_buffer(layer, "oproj_scale8_lut", lut.to(device))
        _set_buffer(
            layer,
            "oproj_output_buffer",
            torch.empty(
                (_KDA_OPROJ_MAX_M, 4096),
                dtype=torch.bfloat16,
                device=device,
            ),
        )
        _set_buffer(
            layer,
            "oproj_workspace",
            torch.empty(
                _KDA_OPROJ_WORKSPACE_BYTES,
                dtype=torch.uint8,
                device=device,
            ),
        )

    def apply(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if (
            x.dtype == torch.bfloat16
            and x.ndim == 2
            and tuple(x.shape) == (8, 4096)
            and x.is_contiguous()
        ):
            output = torch.ops.glm53_kda_oproj_large_m.linear_out(
                x,
                layer.oproj_weight_s4_dot_i32,
                layer.oproj_scale8_codes_dot,
                layer.oproj_scale8_lut,
                layer.oproj_output_buffer[: x.shape[0]],
                layer.oproj_workspace,
            )
            if bias is not None:
                output.add_(bias)
            return output
        return self.fallback.apply(layer, x, bias)


class Glm53KdaOProjM4LinearMethod(LinearMethodBase):
    """Retained fixed-M4 KDA OProj control with BF16 fallback."""

    def __init__(self, prefix: str, layer_index: int) -> None:
        self.prefix = prefix
        self.layer_index = layer_index
        self.fallback = UnquantizedLinearMethod()

    def create_weights(
        self,
        layer: torch.nn.Module,
        input_size_per_partition: int,
        output_partition_sizes: list[int],
        input_size: int,
        output_size: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs,
    ) -> None:
        if (
            input_size_per_partition != 4096
            or sum(output_partition_sizes) != 4096
            or params_dtype != torch.bfloat16
        ):
            raise ValueError(
                "KDA o_proj M4 requires a BF16 rank-local [4096,4096] weight"
            )
        self.fallback.create_weights(
            layer=layer,
            input_size_per_partition=input_size_per_partition,
            output_partition_sizes=output_partition_sizes,
            input_size=input_size,
            output_size=output_size,
            params_dtype=params_dtype,
            **extra_weight_attrs,
        )

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        self.fallback.process_weights_after_loading(layer)
        library = os.getenv("VLLM_GLM53_KDA_OPROJ_M4_LIBRARY", "").strip()
        packet_root = os.getenv("VLLM_GLM53_KDA_OPROJ_M4_PACKET_ROOT", "").strip()
        if not library or not packet_root:
            raise ValueError(
                "KDA o_proj M4 requires ..._LIBRARY and ..._PACKET_ROOT"
            )
        _load_kda_oproj_m4_library(library)
        from vllm.distributed import get_tensor_model_parallel_rank
        from safetensors.torch import load_file

        rank = get_tensor_model_parallel_rank()
        packet_path = (
            Path(packet_root).expanduser().resolve()
            / f"rank{rank}"
            / f"layer{self.layer_index:02d}-kda-oproj-s4g32-scale8.safetensors"
        )
        packet = load_file(str(packet_path), device="cpu")
        weight = packet["weight_s4_dot_i32"].contiguous()
        codes = packet["scale8_codes_dot"].contiguous()
        offset = int(packet["scale8_offset_i16"].item())
        if weight.dtype != torch.int32 or tuple(weight.shape) != (128, 512, 32):
            raise ValueError(f"invalid o_proj weight packet {packet_path}")
        if codes.dtype != torch.uint8 or tuple(codes.shape) != (128, 128, 32):
            raise ValueError(f"invalid o_proj Scale8 packet {packet_path}")

        device = layer.weight.device
        lut = torch.zeros(256, dtype=torch.float32)
        indices = torch.arange(1, 256, dtype=torch.float32)
        lut[1:] = torch.exp2((indices - 1.0 + float(offset)) / 16.0)
        _set_buffer(layer, "oproj_weight_s4_dot_i32", weight.to(device))
        _set_buffer(layer, "oproj_scale8_codes_dot", codes.to(device))
        _set_buffer(layer, "oproj_scale8_lut", lut.to(device))
        _set_buffer(
            layer,
            "oproj_output_buffer",
            torch.empty((4, 4096), dtype=torch.bfloat16, device=device),
        )
        _set_buffer(
            layer,
            "oproj_workspace",
            torch.empty(
                _KDA_OPROJ_M4_WORKSPACE_BYTES,
                dtype=torch.uint8,
                device=device,
            ),
        )

    def apply(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if (
            x.dtype == torch.bfloat16
            and x.ndim == 2
            and tuple(x.shape) == (4, 4096)
            and x.is_contiguous()
        ):
            output = torch.ops.glm53_kda_oproj_m4.linear_out(
                x,
                layer.oproj_weight_s4_dot_i32,
                layer.oproj_scale8_codes_dot,
                layer.oproj_scale8_lut,
                layer.oproj_output_buffer,
                layer.oproj_workspace,
            )
            if bias is not None:
                output.add_(bias)
            return output
        return self.fallback.apply(layer, x, bias)


class Glm53HybridConfig(CompressedTensorsConfig):
    """Compressed-tensors experts plus an opt-in F1 dense KDA projection."""

    @classmethod
    def get_name(cls) -> "QuantizationMethods":
        return "glm53_hybrid"

    @classmethod
    def override_quantization_method(
        cls,
        hf_quant_cfg: dict[str, Any],
        user_quant: str | None,
        hf_config: "PretrainedConfig | None" = None,
    ) -> "QuantizationMethods | None":
        if user_quant != cls.get_name():
            return None
        model_type = getattr(hf_config, "model_type", None)
        checkpoint_method = hf_quant_cfg.get("quant_method")
        if model_type == "glm5_next" and checkpoint_method == "compressed-tensors":
            return cls.get_name()
        return None

    def get_quant_method(
        self,
        layer: torch.nn.Module,
        prefix: str,
    ) -> "QuantizeMethodBase | None":
        oproj_match = _KDA_OPROJ_RE.search(prefix)
        if (
            capture_enabled()
            and isinstance(layer, LinearBase)
            and oproj_match is not None
            and int(oproj_match.group(1)) in _KDA_OPROJ_S4_LAYERS
        ):
            return Glm53KdaOProjCaptureLinearMethod(
                prefix, int(oproj_match.group(1))
            )
        if (
            _kda_oproj_w8_enabled()
            and isinstance(layer, LinearBase)
            and oproj_match is not None
            and int(oproj_match.group(1)) in _kda_oproj_w8_layers()
        ):
            return Glm53KdaOProjW8LinearMethod(
                prefix, int(oproj_match.group(1))
            )
        if (
            _kda_oproj_s4_enabled()
            and isinstance(layer, LinearBase)
            and oproj_match is not None
            and int(oproj_match.group(1)) in _KDA_OPROJ_S4_LAYERS
        ):
            return Glm53KdaOProjS4LinearMethod(
                prefix, int(oproj_match.group(1))
            )
        if (
            _kda_oproj_m4_enabled()
            and isinstance(layer, LinearBase)
            and oproj_match is not None
            and int(oproj_match.group(1)) in _KDA_OPROJ_S4_LAYERS
        ):
            return Glm53KdaOProjM4LinearMethod(
                prefix, int(oproj_match.group(1))
            )
        if (
            envs.VLLM_GLM53_HYBRID_F1_ENABLED
            and isinstance(layer, LinearBase)
            and prefix.endswith(_F1_PREFIX_SUFFIX)
        ):
            return Glm53F1DenseLinearMethod(prefix)
        method = super().get_quant_method(layer, prefix)
        if (
            envs.VLLM_GLM53_HYBRID_IU4_CANONICAL_ENABLED
            and isinstance(layer, RoutedExperts)
            and isinstance(method, CompressedTensorsWNA16MoEMethod)
        ):
            match = _IU4_M1_LAYER_RE.search(prefix)
            if match and int(match.group(1)) in _IU4_CANONICAL_LAYERS:
                return Glm53Iu4CanonicalMoEMethod(method, prefix)
        if (
            envs.VLLM_GLM53_HYBRID_IU4_M1_ENABLED
            and isinstance(layer, RoutedExperts)
            and isinstance(method, CompressedTensorsWNA16MoEMethod)
        ):
            match = _IU4_M1_LAYER_RE.search(prefix)
            if match and int(match.group(1)) in _IU4_M1_LAYER_PANEL:
                return Glm53Iu4M1MoEMethod(method, prefix)
        return method
