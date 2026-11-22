# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Inkling2 attention: relative-bias GQA with short-conv K/V, L2-clipped
q/k/r, gated output and per-token attention sinks."""

from dataclasses import dataclass, replace
import os
from typing import cast

import torch
from torch import nn

from vllm.config import VllmConfig, get_current_vllm_config
from vllm.distributed import (
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
)
from vllm.forward_context import get_forward_context
from vllm.model_executor.layers.attention_layer_base import AttentionLayerBase
from vllm.model_executor.layers.linear import (
    MergedColumnParallelLinear,
    RowParallelLinear,
)
from vllm.model_executor.utils import set_weight_attrs
from vllm.utils.torch_utils import canonicalize_singleton_dim_strides
from vllm.v1.attention.backend import (
    AttentionBackend,
    AttentionCGSupport,
    AttentionMetadata,
    AttentionMetadataBuilder,
    CommonAttentionMetadata,
    MultipleOf,
)
from vllm.v1.core.single_type_kv_cache_manager import (
    FullAttentionManager,
    SlidingWindowManager,
)
from vllm.v1.kv_cache_interface import AttentionSpec, KVCacheLayerConfig


from ..configs import Inkling2ModelConfig
from .conv_state import InklingConvState
from .ops.attention import (
    gated_rel_attention,
    plan_decode_attention,
    rel_decode_attention,
)
from .ops.lossless_bf16 import (
    LosslessBF16Weights,
    lossless_bf16_gemv,
    pack_lossless_bf16,
)
from .ops.qkvr_w8 import pack_qkvr_w8, qkvr_w8_linear
from .ops.sconv import attention_prologue, shift_rel_proj
from .sconv import K_STREAM, V_STREAM, Inkling2ShortConv, conv_metadata


def _can_project_packed_m1(x: torch.Tensor) -> bool:
    return (
        x.shape[0] == 1
        and x.is_cuda
        and x.dtype == torch.bfloat16
        and x.is_contiguous()
        and x.data_ptr() % 4 == 0
    )


def _can_project_w6_batch(x: torch.Tensor, max_m: int = 8) -> bool:
    return (
        x.ndim == 2
        and 2 <= x.shape[0] <= max_m
        and x.shape[1] in (2304, 3072)
        and x.is_cuda
        and x.dtype == torch.bfloat16
        and x.is_contiguous()
        and x.data_ptr() % 4 == 0
    )


def _w6_batch_tile(x: torch.Tensor) -> int:
    return 1 << (int(x.shape[0]) - 1).bit_length()


@dataclass
class Inkling2AttentionMetadata(AttentionMetadata):
    block_table: torch.Tensor
    slot_mapping: torch.Tensor
    query_start_loc: torch.Tensor
    seq_lens: torch.Tensor
    max_query_len: int
    num_actual_tokens: int


class Inkling2AttentionMetadataBuilder(
    AttentionMetadataBuilder[Inkling2AttentionMetadata]
):
    @classmethod
    def get_cudagraph_support(cls, vllm_config, kv_cache_spec):
        if os.getenv("INKLING_AMD_DSPARK_TARGET_GRAPH") == "1":
            return AttentionCGSupport.UNIFORM_BATCH
        if os.getenv("INKLING_AMD_DECODE_GRAPH") == "1":
            return AttentionCGSupport.UNIFORM_SINGLE_TOKEN_DECODE
        return super().get_cudagraph_support(vllm_config, kv_cache_spec)

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._init_reorder_batch_threshold(1)

    def build(
        self,
        common_prefix_len: int,
        common_attn_metadata: CommonAttentionMetadata,
        fast_build: bool = False,
    ) -> Inkling2AttentionMetadata:
        meta = common_attn_metadata
        return Inkling2AttentionMetadata(
            meta.block_table_tensor,
            meta.slot_mapping,
            meta.query_start_loc,
            meta.seq_lens,
            meta.max_query_len,
            meta.num_actual_tokens,
        )


class Inkling2AttentionBackend(AttentionBackend):
    @staticmethod
    def get_name() -> str:
        return "INKLING2_ROCM_REL"

    @staticmethod
    def get_impl_cls():
        raise NotImplementedError("Inkling 2 relative attention runs on its layer")

    @staticmethod
    def get_builder_cls() -> type[Inkling2AttentionMetadataBuilder]:
        return Inkling2AttentionMetadataBuilder

    @staticmethod
    def get_supported_kernel_block_sizes(kv_cache_spec=None) -> list[MultipleOf]:
        return [MultipleOf(16)]


def _kv_cache_layer_config(
    vllm_config: VllmConfig,
    num_kv_heads: int,
    head_dim: int,
    sliding_window: int | None,
) -> KVCacheLayerConfig:
    spec = AttentionSpec(
        block_size=vllm_config.cache_config.block_size,
        num_kv_heads=num_kv_heads,
        head_size=head_dim,
        dtype=torch.bfloat16,
    )
    if sliding_window is not None:
        return SlidingWindowManager.layer_config(spec, sliding_window=sliding_window)
    return FullAttentionManager.layer_config(spec)


class Inkling2AttentionPassCache(nn.Module, AttentionLayerBase):
    """The KV cache of one extra loop pass of a looped `Inkling2Attention`
    (each pass of the loop core attends over its own keys and values)."""

    def __init__(
        self,
        prefix: str,
        num_heads: int,
        num_kv_heads: int,
        head_dim: int,
        sliding_window: int | None,
    ) -> None:
        super().__init__()
        self.prefix = prefix
        # Read by the metadata builders of the layer's attention group.
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.sliding_window = sliding_window
        self.kv_cache = torch.tensor([])
        forward_context = get_current_vllm_config().compilation_config
        if prefix in forward_context.static_forward_context:
            raise ValueError(f"Duplicate layer name: {prefix}")
        forward_context.static_forward_context[prefix] = self

    def get_attn_backend(self) -> type[AttentionBackend]:
        return Inkling2AttentionBackend

    def get_kv_cache_config(self, vllm_config: VllmConfig) -> KVCacheLayerConfig:
        return _kv_cache_layer_config(
            vllm_config, self.num_kv_heads, self.head_dim, self.sliding_window
        )


class Inkling2Attention(nn.Module, AttentionLayerBase):
    """Inkling2 attention for one decoder layer.

    Local layers project all of q/k/v from one input; full layers project the
    query side (q, r, gate, sink) from `x_q`, k from `x_k` and v from
    `x_v`, and use dynamic K/V convolutions and log-scaled q/r.
    """

    def __init__(
        self,
        config: Inkling2ModelConfig,
        is_local: bool,
        conv_state: InklingConvState,
        prefix: str,
        pass_conv_states: tuple[InklingConvState, ...] = (),
    ) -> None:
        super().__init__()
        self.prefix = prefix
        self.is_local = is_local
        split_programs = os.getenv("INKLING_AMD_M1_SPLIT_PROGRAMS_PER_CU")
        self.m1_split_programs_per_cu = int(split_programs) if split_programs else None
        if self.m1_split_programs_per_cu is not None and self.m1_split_programs_per_cu < 1:
            raise ValueError("Inkling split programs per CU must be positive")
        key_tile = os.getenv("INKLING_AMD_M1_KEY_TILE")
        if key_tile:
            self.m1_key_tile = int(key_tile)
            if self.m1_key_tile not in (16, 32, 64, 128):
                raise ValueError("Inkling M1 key tile must be 16, 32, 64 or 128")
        else:
            self.m1_key_tile = 128 if os.getenv("INKLING_AMD_M1_KEY_TILE_128") == "1" else None
        waves = os.getenv("INKLING_AMD_M1_ATTENTION_WAVES")
        if waves:
            self.m1_num_warps = int(waves)
            if self.m1_num_warps not in (1, 2, 4):
                raise ValueError("Inkling M1 attention waves must be 1, 2 or 4")
        else:
            self.m1_num_warps = 2 if os.getenv("INKLING_AMD_M1_ATTENTION_TWO_WAVES") == "1" else None
        self.m1_compact_merge = os.getenv("INKLING_AMD_M1_COMPACT_SPLIT_MERGE") == "1"
        self.m1_contiguous_splits = os.getenv("INKLING_AMD_M1_CONTIGUOUS_SPLITS") == "1"
        num_heads = (
            config.swa_num_attention_heads if is_local else config.num_attention_heads
        )
        num_kv_heads = (
            config.swa_num_key_value_heads if is_local else config.num_key_value_heads
        )
        self.head_dim = config.swa_head_dim if is_local else config.head_dim
        self.d_rel = config.d_rel
        tp_size = get_tensor_model_parallel_world_size()
        tp_rank = get_tensor_model_parallel_rank()
        assert num_heads % tp_size == 0 and num_kv_heads % tp_size == 0
        self.num_heads = num_heads // tp_size
        self.num_kv_heads = num_kv_heads // tp_size
        self.scaling = self.head_dim**-0.5
        self.sliding_window = config.sliding_window_size
        self.rel_extent = config.sliding_window_size if is_local else config.rel_extent
        self.window = config.sliding_window_size - 1 if is_local else -1
        self.qk_clip = config.qk_l2_clip_threshold
        self.rel_clip = config.rel_l2_clip_threshold
        self.eps = config.rms_norm_eps
        assert config.use_value_norm_no_gamma

        q_size = self.num_heads * self.head_dim
        kv_size = self.num_kv_heads * self.head_dim
        # Query-side outputs first so full layers can project them in one GEMM.
        self.split_sizes = [
            q_size,
            self.num_heads * self.d_rel,
            q_size,
            self.num_heads,
            kv_size,
            kv_size,
        ]
        # Full layers project the query side alone; rounding its N up to 8
        # keeps the output rows 16-byte aligned for hipBLASLt (the extra
        # columns, from the K rows, are discarded).
        q_side = sum(self.split_sizes[:4])
        self.q_side_splits = [*self.split_sizes[:4], -q_side % 8]
        k_end = q_side + kv_size
        self._full_qkvr_rows = (
            (0, sum(self.q_side_splits)),
            (q_side, k_end),
            (k_end, k_end + kv_size),
        )
        self._qkv_cta_aligned = q_side % 4 == 0 and kv_size % 4 == 0
        self._fused_w8_qkv = os.getenv("INKLING_AMD_W8_FUSED_QKV") == "1"
        self._w6_verify_batch = (
            os.getenv("INKLING_AMD_W6_VERIFY_ATTN") == "1"
            or os.getenv("INKLING_AMD_TURBO_W6_WMMA") == "1"
        )
        self._w6_waves = int(os.getenv("INKLING_AMD_W6_ATTN_WAVES", "4"))
        pad_size = -sum(self.split_sizes) % 8
        if pad_size:
            self.split_sizes.append(pad_size)
        self.qkvr = MergedColumnParallelLinear(
            config.hidden_size,
            [size * tp_size for size in self.split_sizes],
            bias=False,
            prefix=f"{prefix}.qkvr",
        )
        self.register_buffer("_qkvr_w8", None, persistent=False)
        self.register_buffer("_qkvr_scale", None, persistent=False)
        self._lossless_qkvr: dict[tuple[int, int] | None, LosslessBF16Weights] = {}
        self._lossless_wo: LosslessBF16Weights | None = None
        self._groupwise_w8_qkvr: tuple[torch.Tensor, torch.Tensor] | None = None
        self._groupwise_w8_wo: tuple[torch.Tensor, torch.Tensor] | None = None
        self._groupwise_w6_qkvr: tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None = None
        self._groupwise_w6_wo: tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None = None
        self._w6_wmma_qkvr = None
        self._w6_wmma_qkv_meta = None
        self._w6_wmma_wo = None
        self._turbo_prefill_attn = os.getenv("INKLING_AMD_TURBO_PREFILL_ATTN") == "1" 
        self.wo_ud = RowParallelLinear(
            q_size * tp_size,
            config.hidden_size,
            bias=False,
            reduce_results=False,
            prefix=f"{prefix}.wo_ud",
        )
        self.rel_logits_proj = nn.Parameter(
            torch.empty(self.d_rel, self.rel_extent), requires_grad=False
        )
        self.q_scale = nn.Parameter(
            torch.empty(self.num_heads, self.head_dim), requires_grad=False
        )
        self.r_scale = nn.Parameter(
            torch.empty(self.num_heads, self.d_rel), requires_grad=False
        )
        for param in (self.q_scale, self.r_scale):
            set_weight_attrs(
                param,
                {
                    "weight_loader": lambda p, w: p.data.copy_(
                        w.narrow(0, tp_rank * p.shape[0], p.shape[0])
                    )
                },
            )

        self.conv_state = conv_state
        self.conv_states = (conv_state, *pass_conv_states)
        self.pass_caches = nn.ModuleList(
            Inkling2AttentionPassCache(
                f"{prefix}.pass{p}",
                self.num_heads,
                self.num_kv_heads,
                self.head_dim,
                self.sliding_window if is_local else None,
            )
            for p in range(1, 1 if config.loop_share_cache else len(self.conv_states))
        )
        self.k_sconv, self.v_sconv = (
            Inkling2ShortConv(
                kv_size,
                config.sconv_kernel_size,
                conv_state,
                stream,
                dynamic_input_size=None if is_local else config.hidden_size,
                num_heads=self.num_kv_heads,
            )
            for stream in (K_STREAM, V_STREAM)
        )

        vllm_config = get_current_vllm_config()
        assert vllm_config.cache_config.cache_dtype in ("auto", "bfloat16")
        self.kv_cache = torch.tensor([])
        forward_context = vllm_config.compilation_config.static_forward_context
        if prefix in forward_context:
            raise ValueError(f"Duplicate layer name: {prefix}")
        forward_context[prefix] = self

    def process_weights_after_loading(self) -> None:
        self.rel_proj_shifted = shift_rel_proj(self.rel_logits_proj, self.is_local)
        if os.getenv("INKLING_AMD_QKVR_W8") == "1" and self.qkvr.weight is not None:
            self._qkvr_w8, self._qkvr_scale = pack_qkvr_w8(self.qkvr.weight)
            self.qkvr.weight = None
        if get_tensor_model_parallel_world_size() != 1:
            return
        lossless = os.getenv("INKLING_AMD_LOSSLESS_ATTN", "0")
        groupwise = os.getenv("INKLING_AMD_GROUPWISE_W8_ATTN", "0")
        groupwise_w6 = os.getenv("INKLING_AMD_GROUPWISE_W6_ATTN", "0")
        if lossless not in ("0", "1", "qkvr", "wo"):
            raise ValueError("Select 0, 1, qkvr or wo for INKLING_AMD_LOSSLESS_ATTN")
        if groupwise not in ("0", "1", "qkvr", "wo"):
            raise ValueError("Select 0, 1, qkvr or wo for INKLING_AMD_GROUPWISE_W8_ATTN")
        if groupwise_w6 not in ("0", "1", "qkvr", "wo"):
            raise ValueError("Select 0, 1, qkvr or wo for INKLING_AMD_GROUPWISE_W6_ATTN")

        qkvr_weight = self.qkvr.weight
        # A projection uses one M1 pack. Legacy W8 has already replaced QKVR.
        if groupwise_w6 != "0":
            from .ops.groupwise_w6_head import pack_groupwise_w6_head

            if groupwise_w6 in ("1", "wo"):
                self._groupwise_w6_wo = pack_groupwise_w6_head(self.wo_ud.weight)
            if groupwise_w6 in ("1", "qkvr") and qkvr_weight is not None:
                self._groupwise_w6_qkvr = pack_groupwise_w6_head(qkvr_weight)
            if os.getenv("INKLING_AMD_TURBO_W6_WMMA") == "1":
                from .ops.turbo_kernels import pack_w6_wmma, pack_w6_wmma_qkv

                if groupwise_w6 in ("1", "wo"):
                    self._w6_wmma_wo = pack_w6_wmma(self.wo_ud.weight)
                if groupwise_w6 in ("1", "qkvr") and qkvr_weight is not None:
                    q_side = sum(self.split_sizes[:4])
                    kv_size = self.split_sizes[4]
                    packed_qkv, q_pad, k_end, total = pack_w6_wmma_qkv(
                        qkvr_weight, q_side, kv_size
                    )
                    self._w6_wmma_qkvr = packed_qkv
                    self._w6_wmma_qkv_meta = (q_side, kv_size, q_pad, k_end)

        if groupwise != "0":
            from .ops.groupwise_w8_head import pack_groupwise_w8

            if groupwise in ("1", "wo") and self._groupwise_w6_wo is None:
                self._groupwise_w8_wo = pack_groupwise_w8(self.wo_ud.weight)
            if groupwise in ("1", "qkvr") and qkvr_weight is not None and self._groupwise_w6_qkvr is None:
                self._groupwise_w8_qkvr = pack_groupwise_w8(qkvr_weight)

        if lossless in ("1", "wo") and self._groupwise_w8_wo is None and self._groupwise_w6_wo is None:
            wo_weight = self.wo_ud.weight
            self._lossless_wo = pack_lossless_bf16(wo_weight.detach().cpu()).to(
                wo_weight.device
            )
        if (
            lossless in ("1", "qkvr")
            and qkvr_weight is not None
            and self._groupwise_w8_qkvr is None
            and self._groupwise_w6_qkvr is None
        ):
            packed = pack_lossless_bf16(qkvr_weight.detach().cpu()).to(qkvr_weight.device)
            self._lossless_qkvr[None] = packed
            for rows in self._full_qkvr_rows:
                self._lossless_qkvr[rows] = replace(
                    packed, descriptors=packed.descriptors[slice(*rows)]
                )

    def get_attn_backend(self) -> type[AttentionBackend]:
        return Inkling2AttentionBackend

    def get_kv_cache_config(self, vllm_config: VllmConfig) -> KVCacheLayerConfig:
        return _kv_cache_layer_config(
            vllm_config,
            self.num_kv_heads,
            self.head_dim,
            self.sliding_window if self.is_local else None,
        )

    def _pass_cache(self, pass_idx: int) -> tuple[str, torch.Tensor]:
        if pass_idx == 0 or not self.pass_caches:
            return self.prefix, self.kv_cache
        cache = self.pass_caches[pass_idx - 1]
        return cache.prefix, cache.kv_cache

    def _project_qkvr_rows(
        self, x: torch.Tensor, rows: tuple[int, int] | None = None
    ) -> torch.Tensor:
        row_slice = slice(*rows) if rows is not None else slice(None)
        w6 = self._groupwise_w6_qkvr
        if w6 is not None and _can_project_packed_m1(x):
            from .ops.groupwise_w6_head import groupwise_w6_head_linear

            bank = w6 if rows is None else tuple(tensor[row_slice] for tensor in w6)
            return groupwise_w6_head_linear(
                x, *bank, out_dtype=torch.bfloat16, num_warps=self._w6_waves,
            )
        groupwise = self._groupwise_w8_qkvr
        if groupwise is not None and _can_project_packed_m1(x):
            from .ops.groupwise_w8_head import groupwise_w8_gemv

            codes, scales = groupwise
            if rows is not None:
                codes, scales = codes[row_slice], scales[row_slice]
            return groupwise_w8_gemv(x, codes, scales)
        packed = self._lossless_qkvr.get(rows)
        if packed is not None and _can_project_packed_m1(x):
            return lossless_bf16_gemv(x, packed, variant="paired")
        if self._qkvr_w8 is None:
            weight = self.qkvr.weight
            return nn.functional.linear(x, weight if rows is None else weight[row_slice])
        q, scales = self._qkvr_w8, self._qkvr_scale
        if rows is not None:
            q, scales = q[row_slice], scales[row_slice]
        return qkvr_w8_linear(x, q, scales)

    def _project(
        self, x_q: torch.Tensor, x_k: torch.Tensor, x_v: torch.Tensor
    ) -> list[torch.Tensor]:
        if x_k is x_q and x_v is x_q:
            if (
                self._w6_verify_batch
                and self._w6_wmma_qkvr is not None
                and self._w6_wmma_qkv_meta is not None
                and _can_project_w6_batch(x_q, 64)
            ):
                from .ops.turbo_kernels import w6_wmma_qkv_linear

                q_side, kv_size, q_pad, k_end = self._w6_wmma_qkv_meta
                projected = w6_wmma_qkv_linear(
                    x_q, x_q, x_q, self._w6_wmma_qkvr, q_pad, k_end,
                )
                return [
                    *projected[:, :q_side].split(self.split_sizes[:4], dim=-1),
                    projected[:, q_pad : q_pad + kv_size],
                    projected[:, k_end : k_end + kv_size],
                ]
            w6 = self._groupwise_w6_qkvr
            if self._w6_verify_batch and w6 is not None and _can_project_w6_batch(x_q):
                from .ops.groupwise_w6_head import groupwise_w6_head_batched_linear

                projected = groupwise_w6_head_batched_linear(
                    x_q, *w6, out_dtype=torch.bfloat16, tile_m=_w6_batch_tile(x_q),
                )
                return list(projected.split(self.split_sizes, dim=-1)[:6])
            projected = self._project_qkvr_rows(x_q)
            return list(projected.split(self.split_sizes, dim=-1)[:6])
        q_rows, k_rows, v_rows = self._full_qkvr_rows
        w6 = self._groupwise_w6_qkvr
        if (
            self._w6_verify_batch
            and self._w6_wmma_qkvr is not None
            and self._w6_wmma_qkv_meta is not None
            and _can_project_w6_batch(x_q, 64)
            and _can_project_w6_batch(x_k, 64)
            and _can_project_w6_batch(x_v, 64)
        ):
            from .ops.turbo_kernels import w6_wmma_qkv_linear

            q_side, kv_size, q_pad, k_end = self._w6_wmma_qkv_meta
            projected = w6_wmma_qkv_linear(
                x_q, x_k, x_v, self._w6_wmma_qkvr, q_pad, k_end,
            )
            return [
                *projected[:, :q_side].split(self.split_sizes[:4], dim=-1),
                projected[:, q_pad : q_pad + kv_size],
                projected[:, k_end : k_end + kv_size],
            ]
        if (
            self._w6_verify_batch
            and self._qkv_cta_aligned
            and w6 is not None
            and _can_project_w6_batch(x_q)
            and _can_project_w6_batch(x_k)
            and _can_project_w6_batch(x_v)
        ):
            from .ops.groupwise_w6_head import groupwise_w6_qkv_batched

            ends = (k_rows[0], v_rows[0], v_rows[1])
            projected = groupwise_w6_qkv_batched(
                x_q, x_k, x_v, *w6, ends, tile_m=_w6_batch_tile(x_q),
            )
            return list(projected.split(self.split_sizes[:6], dim=-1))
        groupwise = self._groupwise_w8_qkvr
        if (
            self._qkv_cta_aligned
            and (w6 is not None or (self._fused_w8_qkv and groupwise is not None))
            and _can_project_packed_m1(x_q)
            and _can_project_packed_m1(x_k)
            and _can_project_packed_m1(x_v)
        ):
            ends = (k_rows[0], v_rows[0], v_rows[1])
            if w6 is not None:
                from .ops.groupwise_w6_head import groupwise_w6_qkv_gemv

                projected = groupwise_w6_qkv_gemv(
                    x_q, x_k, x_v, *w6, ends, num_warps=self._w6_waves,
                )
            else:
                from .ops.groupwise_w8_head import groupwise_w8_qkv_gemv

                projected = groupwise_w8_qkv_gemv(x_q, x_k, x_v, *groupwise, ends)
            return list(projected.split(self.split_sizes[:6], dim=-1))
        projected = self._project_qkvr_rows(x_q, q_rows)
        k = self._project_qkvr_rows(x_k, k_rows)
        v = self._project_qkvr_rows(x_v, v_rows)
        return [*projected.split(self.q_side_splits, dim=-1)[:4], k, v]

    def _split_kv_cache(self, cache: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        key_cache, value_cache = cache.transpose(1, 2).split(
            self.head_dim, dim=-1
        )
        return (
            canonicalize_singleton_dim_strides(key_cache),
            canonicalize_singleton_dim_strides(value_cache),
        )

    def forward(
        self,
        positions: torch.Tensor,
        x_q: torch.Tensor,
        x_k: torch.Tensor | None = None,
        x_v: torch.Tensor | None = None,
        tau: torch.Tensor | None = None,
        *,
        pass_idx: int = 0,
    ) -> torch.Tensor:
        """Return this rank's bf16 `(T, hidden)` partial output (to be
        all-reduced over TP).

        Local layers read q/k/v from `x_q`; full layers take `x_k`, `x_v` and
        the fp32 log-scaling `tau` for q/r.
        """
        if self.is_local and x_k is None:
            x_k = x_v = x_q
        assert x_k is not None and x_v is not None
        q, r, gate, sinks, k, v = self._project(x_q, x_k, x_v)
        output = torch.empty(gate.shape, dtype=q.dtype, device=q.device)
        self._attention(
            positions,
            q,
            r,
            gate,
            sinks,
            k,
            v,
            self.k_sconv.dynamic_taps(x_k),
            self.v_sconv.dynamic_taps(x_v),
            tau,
            output,
            pass_idx,
        )
        if (
            self._w6_verify_batch
            and self._w6_wmma_wo is not None
            and _can_project_w6_batch(output, 16)
        ):
            from .ops.turbo_kernels import w6_wmma_linear

            return w6_wmma_linear(output, self._w6_wmma_wo, splits=4)
        if (
            self._w6_verify_batch
            and self._groupwise_w6_wo is not None
            and _can_project_w6_batch(output)
        ):
            from .ops.groupwise_w6_head import groupwise_w6_head_batched_linear

            return groupwise_w6_head_batched_linear(
                output, *self._groupwise_w6_wo, out_dtype=torch.bfloat16,
                tile_m=_w6_batch_tile(output),
            )
        if self._groupwise_w6_wo is not None and _can_project_packed_m1(output):
            from .ops.groupwise_w6_head import groupwise_w6_head_linear

            return groupwise_w6_head_linear(
                output, *self._groupwise_w6_wo, out_dtype=torch.bfloat16, num_warps=self._w6_waves,
            )
        if self._groupwise_w8_wo is not None and _can_project_packed_m1(output):
            from .ops.groupwise_w8_head import groupwise_w8_gemv

            return groupwise_w8_gemv(output, *self._groupwise_w8_wo)
        if self._lossless_wo is not None and _can_project_packed_m1(output):
            return lossless_bf16_gemv(output, self._lossless_wo, variant="paired")
        return nn.functional.linear(output, self.wo_ud.weight)

    def _attention(
        self,
        positions: torch.Tensor,
        q: torch.Tensor,
        r: torch.Tensor,
        gate: torch.Tensor,
        sinks: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        k_dynamic: torch.Tensor | None,
        v_dynamic: torch.Tensor | None,
        tau: torch.Tensor | None,
        output: torch.Tensor,
        pass_idx: int,
    ) -> None:
        """Write the gated attention output to `output`.

        The prologue's layout and the attention plan follow the batch's
        requests, so this runs eagerly in piecewise CUDA graphs, which are
        captured on a dummy batch and replayed for any batch of the same
        token count.
        """
        attn_metadata = get_forward_context().attn_metadata
        key, kv_cache = self._pass_cache(pass_idx)
        if not isinstance(attn_metadata, dict) or kv_cache.numel() == 0:
            output.zero_()
            return
        md = cast(Inkling2AttentionMetadata, attn_metadata[key])
        conv_state = self.conv_states[pass_idx]
        conv_md = conv_metadata(conv_state)
        assert conv_md is not None
        num_tokens = q.shape[0]
        q = q.view(num_tokens, self.num_heads, self.head_dim)
        key_cache, value_cache = self._split_kv_cache(kv_cache)
        clipped_q = q.new_empty(q.shape)
        # Sheared as `rel_attention` reads it (see attention_prologue).
        rel_logits = q.new_empty(
            (num_tokens + 128, self.num_heads, self.rel_extent + 256)
        )
        attention_prologue(
            q,
            r.view(num_tokens, self.num_heads, self.d_rel),
            k,
            v,
            self.q_scale,
            self.r_scale,
            tau,
            self.rel_proj_shifted,
            self.k_sconv.conv_weight,
            self.v_sconv.conv_weight,
            k_dynamic,
            v_dynamic,
            conv_state.kv_cache,
            positions,
            conv_md.block_table,
            conv_md.seq_idx,
            conv_md.slot_mapping,
            conv_md.query_start,
            key_cache,
            value_cache,
            md.slot_mapping,
            clipped_q,
            rel_logits,
            md.query_start_loc,
            md.seq_lens,
            clip_threshold=self.qk_clip,
            rel_threshold=self.rel_clip,
            norm_eps=self.eps,
        )
        # Decode and verify batches split the keys; the split count follows
        # the batch alone, so it holds for every replay of a CUDA graph.
        m1 = md.seq_lens.shape[0] == md.max_query_len == 1
        key_tile = self.m1_key_tile if m1 else None
        num_splits = plan_decode_attention(
            self.is_local,
            md.seq_lens.shape[0],
            md.max_query_len,
            self.num_heads,
            self.num_kv_heads,
            self.rel_extent,
            programs_per_cu=self.m1_split_programs_per_cu if m1 else None,
            key_tile=key_tile,
        )
        if num_splits:
            if (
                self._turbo_prefill_attn
                and (md.seq_lens.shape[0] > 1 or md.max_query_len >= 2)
                and 1 <= md.max_query_len <= 8
                and key_cache.shape[1] in (16, 64)
            ):
                from .ops.turbo_kernels import turbo_rel_decode_m8_attention

                turbo_rel_decode_m8_attention(
                    clipped_q,
                    key_cache,
                    value_cache,
                    rel_logits,
                    gate,
                    sinks,
                    output,
                    block_table=md.block_table,
                    cache_seqlens=md.seq_lens,
                    cu_seqlens_q=md.query_start_loc,
                    max_query_len=md.max_query_len,
                    num_actual_tokens=md.num_actual_tokens,
                    softmax_scale=self.scaling,
                    is_local=self.is_local,
                )
                return
            rel_decode_attention(
                clipped_q,
                key_cache,
                value_cache,
                rel_logits,
                gate,
                sinks,
                output,
                block_table=md.block_table,
                cache_seqlens=md.seq_lens,
                cu_seqlens_q=md.query_start_loc,
                max_query_len=md.max_query_len,
                num_actual_tokens=md.num_actual_tokens,
                softmax_scale=self.scaling,
                is_local=self.is_local,
                num_splits=num_splits,
                key_tile=key_tile,
                num_warps=self.m1_num_warps if m1 else None,
                compact_merge=self.m1_compact_merge and m1,
                contiguous_splits=self.m1_contiguous_splits and m1,
            )
            return
        if (
            self._turbo_prefill_attn
            and md.max_query_len >= 64
            and key_cache.shape[1] in (16, 64)
        ):
            from .ops.turbo_kernels import turbo_gated_rel_attention

            turbo_gated_rel_attention(
                clipped_q,
                key_cache,
                value_cache,
                rel_logits,
                gate,
                sinks,
                output,
                block_table=md.block_table,
                cache_seqlens=md.seq_lens,
                cu_seqlens_q=md.query_start_loc,
                max_seqlen_q=md.max_query_len,
                num_actual_tokens=md.num_actual_tokens,
                softmax_scale=self.scaling,
                window=self.window,
            )
            return
        gated_rel_attention(
            clipped_q,
            key_cache,
            value_cache,
            rel_logits,
            gate,
            sinks,
            output,
            block_table=md.block_table,
            cache_seqlens=md.seq_lens,
            cu_seqlens_q=md.query_start_loc,
            max_seqlen_q=md.max_query_len,
            num_actual_tokens=md.num_actual_tokens,
            softmax_scale=self.scaling,
            window=self.window,
        )
