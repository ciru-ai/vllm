# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Inkling2 DSpark draft (`mtp.safetensors`): an anchor plus mask block run
through a small dense transformer over DFlare-mixed target-feature K/V, then a
recurrent (`rnn_hidden`) Markov head. Numerics follow the reference
implementation's draft (RMSNorms as flashinfer's, bf16 logits)."""

import os
from collections.abc import Iterable
from typing import cast

import torch
from torch import nn

from vllm import _custom_ops as ops
from vllm.config import VllmConfig, get_current_vllm_config
from vllm.distributed import (
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
    tensor_model_parallel_all_gather,
    tensor_model_parallel_all_reduce,
)
from vllm.forward_context import get_forward_context
from vllm.model_executor.layers.linear import (
    MergedColumnParallelLinear,
    QKVParallelLinear,
    ReplicatedLinear,
    RowParallelLinear,
)
from vllm.model_executor.layers.rotary_embedding import get_rope
from vllm.model_executor.model_loader.weight_utils import default_weight_loader
from vllm.utils.torch_utils import canonicalize_singleton_dim_strides
from vllm.v1.core.single_type_kv_cache_manager import DFlashSlidingWindowManager
from vllm.v1.kv_cache_interface import KVCacheLayerConfig

from ..configs import Inkling2DSparkConfig
from ..speculator_config import Inkling2DSparkSpeculatorConfig
from .attention import (
    Inkling2AttentionBackend,
    Inkling2AttentionMetadata,
    Inkling2AttentionPassCache,
)
from .ops.attention import block_decode_attention, plan_decode_attention
from .ops.dspark import (
    add_softcap,
    dspark_argmax,
    markov_gate,
    mix_rmsnorm,
    qk_norm_rope_,
    rope_,
    silu_and_mul,
)
from .ops.dspark_w8_backbone import DraftW8Weight, draft_w8_linear, pack_draft_w8
from .ops.norm import fused_add_rms_norm, rms_norm


def _draft_linear(
    x: torch.Tensor, layer: nn.Module, packed: DraftW8Weight | None
) -> torch.Tensor:
    if (
        packed is not None
        and x.shape == (8, packed.in_features)
        and x.dtype == torch.bfloat16
        and x.device == packed.codes.device
        and x.is_contiguous()
    ):
        return draft_w8_linear(x, packed)
    return nn.functional.linear(x, layer.weight)


class _RMSNorm(nn.Module):
    def __init__(self, hidden_size: int, eps: float) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size), requires_grad=False)
        self.eps = eps

    def forward(
        self, x: torch.Tensor, residual: torch.Tensor | None = None
    ) -> torch.Tensor:
        if residual is None:
            return rms_norm(x, self.weight, self.eps)
        # In place: residual += x; x = norm(residual), from the unrounded sum.
        fused_add_rms_norm(x, residual, self.weight, self.eps)
        return x


def all_reduce_add_rmsnorm(
    x: torch.Tensor, residual: torch.Tensor, norm: _RMSNorm
) -> torch.Tensor:
    """In place: `residual += all_reduce(x); x = norm(residual)`."""
    return norm(tensor_model_parallel_all_reduce(x), residual)


class Inkling2DraftAttentionBackend(Inkling2AttentionBackend):
    """Paged BF16 cache metadata for the draft's bidirectional block kernel."""

    @staticmethod
    def get_name() -> str:
        return "INKLING2_ROCM_DSPARK"

    @classmethod
    def supports_non_causal(cls) -> bool:
        return True


class Inkling2DraftKV(Inkling2AttentionPassCache):
    def get_attn_backend(self) -> type[Inkling2DraftAttentionBackend]:
        return Inkling2DraftAttentionBackend

    def get_kv_cache_config(self, vllm_config: VllmConfig) -> KVCacheLayerConfig:
        layer_config = super().get_kv_cache_config(vllm_config)
        if self.sliding_window is None:
            return layer_config
        return DFlashSlidingWindowManager.layer_config(
            layer_config.spec,
            sliding_window=self.sliding_window,
            num_speculative_tokens=vllm_config.num_speculative_tokens,
            max_in_flight_steps=vllm_config.max_concurrent_batches,
            num_lookahead_tokens=vllm_config.num_lookahead_tokens,
        )


class Inkling2DraftSelfAttention(nn.Module):
    """Bidirectional block attention over context K/V (from target features)
    and the block itself; local layers see `sliding_window_size` positions
    before the block start."""

    def __init__(
        self, config: Inkling2DSparkConfig, is_local: bool, prefix: str
    ) -> None:
        super().__init__()
        tp_size = get_tensor_model_parallel_world_size()
        assert tp_size <= config.num_key_value_heads
        hidden, self.head_dim = config.hidden_size, config.head_dim
        self.num_heads = config.num_attention_heads // tp_size
        self.num_kv_heads = config.num_key_value_heads // tp_size
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim
        self.eps = config.rms_norm_eps
        _vc_spec = get_current_vllm_config().speculative_config
        self.block = (_vc_spec.num_speculative_tokens + 1) if _vc_spec is not None else config.draft_block_size
        self.is_local = is_local
        self.qkv_proj = QKVParallelLinear(
            hidden,
            self.head_dim,
            config.num_attention_heads,
            config.num_key_value_heads,
            bias=False,
            prefix=f"{prefix}.qkv_proj",
        )
        self.kv_proj_ctx = MergedColumnParallelLinear(
            hidden,
            [config.num_key_value_heads * self.head_dim] * 2,
            bias=False,
            prefix=f"{prefix}.kv_proj_ctx",
        )
        self.o_proj = RowParallelLinear(
            config.num_attention_heads * self.head_dim,
            hidden,
            bias=False,
            prefix=f"{prefix}.o_proj",
        )
        self.w8_qkv: DraftW8Weight | None = None
        self.w8_o: DraftW8Weight | None = None
        self.w4_qkv = None
        self.w4_o = None
        self.w4_kv_ctx = None
        self.q_norm = _RMSNorm(self.head_dim, self.eps)
        self.k_norm = _RMSNorm(self.head_dim, self.eps)
        self.k_norm_ctx = _RMSNorm(self.head_dim, self.eps)
        # Owns the paged KV cache only; the same spec as the target's layers
        # (the scheduler keeps the last sliding_window_size positions before
        # the block, which is what the draft reads).
        self.attn = Inkling2DraftKV(
            f"{prefix}.attn",
            self.num_heads,
            self.num_kv_heads,
            self.head_dim,
            config.sliding_window_size if is_local else None,
        )
        self.register_buffer(
            "kv_scale", torch.ones((), dtype=torch.float32), persistent=False
        )
        # Keys a block sees: itself and, if local, the window before it.
        self._sliding_window_size = config.sliding_window_size
        self.window = config.sliding_window_size + self.block if is_local else 0
        vllm_config = get_current_vllm_config()
        max_num_seqs = vllm_config.scheduler_config.max_num_seqs
        for b in range(2, config.draft_block_size + 1):
            self.register_buffer(
                f"_cu_seqlens_b{b}",
                torch.arange(max_num_seqs + 1, dtype=torch.int32) * b,
                persistent=False,
            )
        self.register_buffer(
            "cu_seqlens_q",
            getattr(self, f"_cu_seqlens_b{self.block}"),
            persistent=False,
        )

    def _metadata(self) -> Inkling2AttentionMetadata | None:
        attn_metadata = get_forward_context().attn_metadata
        if not isinstance(attn_metadata, dict) or self.attn.kv_cache.numel() == 0:
            return None
        return cast(Inkling2AttentionMetadata, attn_metadata[self.attn.prefix])

    def _split_kv(self) -> tuple[torch.Tensor, torch.Tensor]:
        key, value = self.attn.kv_cache.transpose(1, 2).split(self.head_dim, dim=-1)
        return (
            canonicalize_singleton_dim_strides(key),
            canonicalize_singleton_dim_strides(value),
        )

    def _write_kv(
        self, k: torch.Tensor, v: torch.Tensor, slot_mapping: torch.Tensor
    ) -> None:
        key, value = self._split_kv()
        ops.reshape_and_cache_flash(
            k, v, key, value, slot_mapping, "auto", self.kv_scale, self.kv_scale
        )

    def forward(
        self, positions: torch.Tensor, x: torch.Tensor, cos_sin_cache: torch.Tensor
    ) -> torch.Tensor:
        if self.w4_qkv is not None and x.shape[0] <= 8 and x.dtype == torch.bfloat16:
            from .ops.turbo_kernels import w4_wmma_linear

            qkv = w4_wmma_linear(x, self.w4_qkv, splits=1)
        else:
            qkv = _draft_linear(x, self.qkv_proj, self.w8_qkv)
        qk_norm_rope_(
            qkv,
            positions,
            self.q_norm.weight,
            self.k_norm.weight,
            cos_sin_cache,
            self.num_heads,
            self.num_kv_heads,
            self.eps,
        )
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
        out = q.new_zeros((q.shape[0], self.q_size))
        md = self._metadata()
        if md is not None:
            self._write_kv(
                k.view(-1, self.num_kv_heads, self.head_dim),
                v.view(-1, self.num_kv_heads, self.head_dim),
                md.slot_mapping[: q.shape[0]],
            )
            key_cache, value_cache = self._split_kv()
            num_reqs = min(q.shape[0] // self.block, md.seq_lens.shape[0])
            num_tokens = num_reqs * self.block
            # The plan's 0 (query tiles, for longer blocks) is one split
            # here.
            num_splits = max(
                1,
                plan_decode_attention(
                    self.is_local,
                    num_reqs,
                    self.block,
                    self.num_heads,
                    self.num_kv_heads,
                    self.window,
                ),
            )
            block_decode_attention(
                q[:num_tokens].view(num_tokens, self.num_heads, self.head_dim),
                key_cache,
                value_cache,
                out[:num_tokens],
                block_table=md.block_table[:num_reqs],
                cache_seqlens=md.seq_lens[:num_reqs],
                cu_seqlens_q=self.cu_seqlens_q[: num_reqs + 1],
                block=self.block,
                window=self.window,
                num_splits=num_splits,
            )
        if self.w4_o is not None and out.shape[0] <= 8 and out.dtype == torch.bfloat16:
            from .ops.turbo_kernels import w4_wmma_linear

            return w4_wmma_linear(out, self.w4_o, splits=4)
        return _draft_linear(out, self.o_proj, self.w8_o)

    def store_context_kv(
        self,
        context: torch.Tensor,
        positions: torch.Tensor,
        cos_sin_cache: torch.Tensor,
        slot_mapping: torch.Tensor | None,
    ) -> None:
        if self.w4_kv_ctx is not None and 1 <= context.shape[0] <= 8 and context.dtype == torch.bfloat16:
            from .ops.turbo_kernels import w4_wmma_linear

            kv_ctx = w4_wmma_linear(context, self.w4_kv_ctx, splits=1)
        else:
            kv_ctx = nn.functional.linear(context, self.kv_proj_ctx.weight)
        k, v = kv_ctx.split([self.kv_size, self.kv_size], -1)
        k = self.k_norm_ctx(k.reshape(-1, self.head_dim))
        k = k.view(-1, self.num_kv_heads, self.head_dim)
        rope_(k, positions, cos_sin_cache)
        if slot_mapping is not None and self.attn.kv_cache.numel() > 0:
            self._write_kv(
                k, v.view(-1, self.num_kv_heads, self.head_dim), slot_mapping
            )


class Inkling2DraftMLP(nn.Module):
    def __init__(self, config: Inkling2DSparkConfig, prefix: str) -> None:
        super().__init__()
        self.gate_up_proj = MergedColumnParallelLinear(
            config.hidden_size,
            [config.intermediate_size] * 2,
            bias=False,
            prefix=f"{prefix}.gate_up_proj",
        )
        self.down_proj = RowParallelLinear(
            config.intermediate_size,
            config.hidden_size,
            bias=False,
            prefix=f"{prefix}.down_proj",
        )

        self.w8_gate_up: DraftW8Weight | None = None
        self.w8_down: DraftW8Weight | None = None
        self.w4_gate_up = None
        self.w4_down = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.w4_gate_up is not None and 1 <= x.shape[0] <= 64 and x.shape[1] == self.w4_gate_up.in_features and x.dtype == torch.bfloat16:
            from .ops.turbo_kernels import w4_wmma_linear

            x = silu_and_mul(w4_wmma_linear(x, self.w4_gate_up, splits=1))
            return w4_wmma_linear(x, self.w4_down, splits=4)
        x = silu_and_mul(_draft_linear(x, self.gate_up_proj, self.w8_gate_up))
        return _draft_linear(x, self.down_proj, self.w8_down)


class Inkling2DraftLayer(nn.Module):
    def __init__(
        self, config: Inkling2DSparkConfig, is_local: bool, prefix: str
    ) -> None:
        super().__init__()
        hidden, eps = config.hidden_size, config.rms_norm_eps
        self.input_layernorm = _RMSNorm(hidden, eps)
        self.post_attention_layernorm = _RMSNorm(hidden, eps)
        self.dflare_norm = _RMSNorm(hidden, eps)
        self.self_attn = Inkling2DraftSelfAttention(
            config, is_local, f"{prefix}.self_attn"
        )
        self.mlp = Inkling2DraftMLP(config, f"{prefix}.mlp")

    def forward(
        self,
        positions: torch.Tensor,
        x: torch.Tensor,
        residual: torch.Tensor | None,
        cos_sin_cache: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Returns this rank's partial MLP output and the residual."""
        if residual is None:
            residual = x
            x = self.input_layernorm(x)
        else:
            x = all_reduce_add_rmsnorm(x, residual, self.input_layernorm)
        x = self.self_attn(positions, x, cos_sin_cache)
        x = all_reduce_add_rmsnorm(x, residual, self.post_attention_layernorm)
        return self.mlp(x), residual


class Inkling2MarkovHead(nn.Module):
    """Recurrent (`rnn_hidden`) correction of the draft's base logits.

    Its rank-`r` latent goes through `hidden_out` and the shared LM head,
    folded once into a `(vocab, r)` projection. The `[hidden, f_rnn]`
    columns of `rnn_proj` (and its bias) are applied to all draft steps at
    once in fp32; each step adds the `[state, previous]` columns and rounds.
    """

    def __init__(self, config: Inkling2DSparkConfig) -> None:
        super().__init__()
        hidden, rank = config.hidden_size, config.markov_rank
        self.markov_rank = rank
        self.markov_w1 = nn.Embedding(
            config.padded_vocab_size,
            rank,
            _weight=torch.empty(config.padded_vocab_size, rank),
            _freeze=True,
        )
        self.rnn_proj = ReplicatedLinear(2 * rank + 2 * hidden, 3 * rank, bias=True)
        self.hidden_out = ReplicatedLinear(rank, hidden, bias=True)
        self.dflare_norm = _RMSNorm(hidden, config.rms_norm_eps)
        self.fold_weight: torch.Tensor | None = None
        self.fold_bias: torch.Tensor | None = None
        self.w8_fold_weight: tuple[torch.Tensor, torch.Tensor] | None = None
        self.w6_fold_weight: tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None = None
        self.w4_fold_weight: tuple[torch.Tensor, torch.Tensor] | None = None
        self.w4_fold_wmma = None
        self.fold_bias_pad: torch.Tensor | None = None
        self.rnn_bias: torch.Tensor | None = None

    @torch.no_grad()
    def fold(self, lm_head_weight: torch.Tensor) -> None:
        """Fold `hidden_out` into the (TP-local) LM head rows in fp64."""
        projection = self.hidden_out.weight.double()
        bias = self.hidden_out.bias.double()
        num_rows = lm_head_weight.shape[0]
        self.fold_weight = lm_head_weight.new_empty((num_rows, self.markov_rank))
        self.fold_bias = lm_head_weight.new_empty((num_rows,))
        for start in range(0, num_rows, 4096):
            rows = slice(start, start + 4096)
            head = lm_head_weight[rows].double()
            self.fold_weight[rows].copy_(head @ projection)
            self.fold_bias[rows].copy_(head @ bias)
        if self.markov_rank == 256 and os.getenv("INKLING_AMD_DSPARK_W8_MARKOV") == "1":
            from .ops.groupwise_w8_head import pack_groupwise_w8

            self.w8_fold_weight = pack_groupwise_w8(self.fold_weight)

    def context(self, hidden: torch.Tensor, f_rnn: torch.Tensor) -> torch.Tensor:
        """fp32 `rnn_proj` bias plus its `[hidden, f_rnn]` term for the
        `(req, step)`-ordered `hidden`; returns `(step, req, 3 * r)`."""
        assert self.rnn_bias is not None
        num_reqs, width = f_rnn.shape
        hidden = hidden.view(num_reqs, -1, width).transpose(0, 1)
        inputs = torch.cat([hidden, f_rnn.expand_as(hidden)], dim=-1)
        weight = self.rnn_proj.weight[:, 2 * self.markov_rank :]
        out = torch.addmm(
            self.rnn_bias, inputs.flatten(0, 1), weight.T, out_dtype=torch.float32
        )
        return out.view(hidden.shape[0], num_reqs, -1)

    def step_latent(
        self, state: torch.Tensor, prev_tokens: torch.Tensor, context: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        inputs = torch.cat([state, self.markov_w1(prev_tokens)], dim=-1)
        weight = self.rnn_proj.weight[:, : 2 * self.markov_rank]
        raw = torch.addmm(context, inputs, weight.T, out_dtype=torch.float32)
        return markov_gate(raw, state)

    def step(
        self, state: torch.Tensor, prev_tokens: torch.Tensor, context: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """One recurrence step; returns the new state and the TP-local bias."""
        assert self.fold_weight is not None and self.fold_bias is not None
        state, latent = self.step_latent(state, prev_tokens, context)
        if (
            self.w4_fold_wmma is not None
            and 1 <= latent.shape[0] <= 16
            and latent.dtype == torch.bfloat16
        ):
            from .ops.turbo_kernels import w4_wmma_linear

            return state, w4_wmma_linear(
                latent, self.w4_fold_wmma, bias=self.fold_bias_pad
            )
        if self.w8_fold_weight is not None and latent.shape[0] == 1:
            from .ops.groupwise_w8_head import groupwise_w8_markov_addmm

            return state, groupwise_w8_markov_addmm(
                latent, *self.w8_fold_weight, self.fold_bias
            )
        return state, torch.addmm(self.fold_bias, latent, self.fold_weight.T)


class Inkling2DSparkDecoder(nn.Module):
    def __init__(
        self, config: Inkling2DSparkConfig, max_model_len: int, prefix: str
    ) -> None:
        super().__init__()
        self.config = config
        hidden = config.hidden_size
        self.layers = nn.ModuleList(
            Inkling2DraftLayer(
                config, i in config.local_layer_ids, f"{prefix}.layers.{i}"
            )
            for i in range(config.num_hidden_layers)
        )
        self.norm = _RMSNorm(hidden, config.rms_norm_eps)
        self.markov_head = Inkling2MarkovHead(config)
        self.fuse_weight = nn.Parameter(
            torch.zeros(config.num_hidden_layers + 1, len(config.target_layer_ids)),
            requires_grad=False,
        )
        self.mask_embed = nn.Parameter(torch.zeros(hidden), requires_grad=False)
        self.register_buffer("fuse_alpha", torch.empty_like(self.fuse_weight))
        # fp32 [cos | sin] table, read directly by the draft's RoPE kernels.
        self.cos_sin_cache = get_rope(
            config.head_dim,
            max_position=max_model_len,
            rope_parameters={"rope_type": "default", "rope_theta": config.rope_theta},
            dtype=torch.float32,
        ).cos_sin_cache
        # Shared with the target model (see Inkling2DSparkModel.attach_target).
        self.embed_tokens: nn.Module | None = None
        self.embed_norm: nn.Module | None = None

    def mix(self, features: torch.Tensor, row: int, norm: _RMSNorm) -> torch.Tensor:
        targets = features.view(features.shape[0], -1, self.config.hidden_size)
        return mix_rmsnorm(targets, self.fuse_alpha[row], norm.weight, norm.eps)

    def embed_block(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Normalized target embedding of each block's anchor, then masks."""
        assert self.embed_tokens is not None and self.embed_norm is not None
        block = self.layers[0].self_attn.block
        ids = input_ids.view(-1, block)
        x = self.mask_embed.expand(ids.shape[0], block, -1).clone()
        x[:, 0] = self.embed_norm(self.embed_tokens(ids[:, 0].contiguous()))
        return x.view(-1, self.config.hidden_size)

    def forward(self, input_ids: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        x = self.embed_block(input_ids)
        residual = None
        for layer in self.layers:
            x, residual = layer(positions, x, residual, self.cos_sin_cache)
        assert residual is not None
        return all_reduce_add_rmsnorm(x, residual, self.norm)


class Inkling2DSparkModel(nn.Module):
    """DSpark draft for Inkling2; plugged in by `Inkling2DSparkSpeculator`."""

    # The speculator this is a draft model of (see SpeculativeDraftModel).
    speculator_config_class = Inkling2DSparkSpeculatorConfig
    allow_patterns_overrides = ["mtp.safetensors"]
    has_own_embed_tokens = False
    has_own_lm_head = False
    draft_id_to_target_id = None

    _stacked = (
        (".q_proj.", ".qkv_proj.", "q"),
        (".k_proj.", ".qkv_proj.", "k"),
        (".v_proj.", ".qkv_proj.", "v"),
        (".k_proj_ctx.", ".kv_proj_ctx.", 0),
        (".v_proj_ctx.", ".kv_proj_ctx.", 1),
        (".gate_proj.", ".gate_up_proj.", 0),
        (".up_proj.", ".gate_up_proj.", 1),
    )

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        super().__init__()
        assert vllm_config.speculative_config is not None
        config = vllm_config.speculative_config.draft_model_config.hf_config
        assert isinstance(config, Inkling2DSparkConfig)
        num_spec = vllm_config.speculative_config.num_speculative_tokens
        if not (1 <= num_spec <= config.draft_block_size - 1):
            raise ValueError(
                "Inkling2 DSpark requires 1 <= num_speculative_tokens <= "
                f"{config.draft_block_size - 1}"
            )
        if vllm_config.model_config.max_model_len > config.max_position_embeddings:
            raise ValueError(
                "Inkling2 DSpark drafts up to max_position_embeddings="
                f"{config.max_position_embeddings} positions; set --max-model-len "
                "to at most that"
            )
        self.config = config
        self.model = Inkling2DSparkDecoder(
            config, vllm_config.model_config.max_model_len, f"{prefix}model"
        )
        self.lm_head: nn.Module | None = None
        self.w6_base_head: tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None = None
        self.w4_base_head = None
        self.markov_topk = int(os.getenv("INKLING_AMD_TURBO_MARKOV_TOPK", "0"))
        self.max_num_reqs = vllm_config.scheduler_config.max_num_seqs
        self.fused_markov_argmax = (
            os.getenv("INKLING_AMD_DSPARK_MARKOV_FUSED_ARGMAX") == "1"
            and vllm_config.parallel_config.tensor_parallel_size == 1
            and vllm_config.speculative_config.draft_sample_method == "greedy"
            and config.markov_rank == 256
            and torch.version.hip is not None
            and torch.cuda.get_device_properties().gcnArchName.split(":")[0] == "gfx1151"
        )

    @torch.no_grad()
    def attach_target(self, target_model: nn.Module) -> None:
        """Finish sharing the target's normalized embeddings and LM head (its
        `embed_tokens` and `lm_head` are aliased by `load_dspark_model`)
        and derive the weights that depend on them."""
        assert self.lm_head is not None and self.lm_head.weight.dtype == torch.bfloat16
        if os.getenv("INKLING_AMD_DSPARK_W6_BASE_HEAD") == "1":
            if (
                self.lm_head is not target_model.lm_head
                or getattr(target_model, "packed_head_format", None) != "w6"
                or target_model.packed_head is None
            ):
                raise ValueError("The DSpark W6 base head needs a shared target W6 head")
            self.w6_base_head = target_model.packed_head
            self.w6_base_head_waves = target_model.w6_head_waves
        self.model.embed_norm = target_model.model.embed_norm
        head = self.model.markov_head
        head.fold(self.lm_head.weight)
        if self.fused_markov_argmax:
            if os.getenv("INKLING_AMD_DSPARK_MARKOV_W6") == "1":
                from .ops.groupwise_w6_head import pack_groupwise_w6_head

                head.w6_fold_weight = pack_groupwise_w6_head(
                    head.fold_weight, group_size=64
                )
            elif head.w8_fold_weight is None and os.getenv("INKLING_AMD_TURBO_DRAFT_W4") != "1":
                raise ValueError("Fused W8 Markov argmax requires the W8 fold bank")
        head.rnn_bias = head.rnn_proj.bias.float()
        num_cols = self.lm_head.weight.shape[0]
        vocab_offset = get_tensor_model_parallel_rank() * num_cols
        self.valid_vocab = min(num_cols, self.config.vocab_size - vocab_offset)
        assert self.valid_vocab > 0
        self.model.fuse_alpha.copy_(
            torch.softmax(self.model.fuse_weight.float(), dim=-1)
        )
        self._prepare_w8_backbone()
        if (
            os.getenv("INKLING_AMD_TURBO_DRAFT_W4") == "1"
            and get_tensor_model_parallel_world_size() == 1
        ):
            from .ops.turbo_kernels import pack_w4_markov, pack_w4_wmma

            self.w4_base_head = pack_w4_wmma(self.lm_head.weight)
            for layer in self.model.layers:
                layer.mlp.w4_gate_up = pack_w4_wmma(layer.mlp.gate_up_proj.weight)
                layer.mlp.w4_down = pack_w4_wmma(layer.mlp.down_proj.weight)
            if head.fold_weight is not None:
                if self.max_num_reqs == 1 or self.fused_markov_argmax:
                    head.w4_fold_weight = pack_w4_markov(head.fold_weight)
                head.w4_fold_wmma = pack_w4_wmma(head.fold_weight)
                bias_pad = torch.zeros(
                    head.w4_fold_wmma.padded_out,
                    dtype=head.fold_bias.dtype,
                    device=head.fold_bias.device,
                )
                bias_pad[: head.fold_bias.shape[0]].copy_(head.fold_bias)
                head.fold_bias_pad = bias_pad

    def _prepare_w8_backbone(self) -> None:
        assert self.lm_head is not None
        mode = os.getenv("INKLING_AMD_DSPARK_W8_BACKBONE", "0")
        if mode not in ("0", "mlp", "all"):
            raise ValueError("INKLING_AMD_DSPARK_W8_BACKBONE expects 0, mlp, or all")
        if (
            mode != "0"
            and get_tensor_model_parallel_world_size() == 1
            and self.config.hidden_size == 2304
            and self.config.intermediate_size == 9216
            and self.config.num_attention_heads == 24
            and self.config.num_key_value_heads == 8
            and self.config.head_dim == 128
            and self.config.draft_block_size == 8
            and torch.version.hip is not None
            and self.lm_head.weight.is_cuda
            and torch.cuda.get_device_properties(
                self.lm_head.weight.device,
            ).gcnArchName.split(":", 1)[0] == "gfx1151"
        ):
            for layer in self.model.layers:
                layer.mlp.w8_gate_up = pack_draft_w8(layer.mlp.gate_up_proj.weight)
                layer.mlp.w8_down = pack_draft_w8(layer.mlp.down_proj.weight)
                if mode == "all":
                    layer.self_attn.w8_qkv = pack_draft_w8(layer.self_attn.qkv_proj.weight)
                    layer.self_attn.w8_o = pack_draft_w8(layer.self_attn.o_proj.weight)

    def set_active_block_size(self, block: int) -> None:
        for layer in self.model.layers:
            attn = layer.self_attn
            attn.block = block
            attn.window = attn._sliding_window_size + block if attn.is_local else 0
            attn.cu_seqlens_q = getattr(attn, f"_cu_seqlens_b{block}")

    def get_draft_kv_cache_layer_names(self) -> list[str]:
        return [layer.self_attn.attn.prefix for layer in self.model.layers]

    def combine_hidden_states(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # Each draft layer mixes the raw target features itself.
        return hidden_states

    def precompute_and_store_context_kv(
        self,
        context_states: torch.Tensor,
        context_positions: torch.Tensor,
        context_slot_mappings: list[torch.Tensor | None] | None = None,
    ) -> None:
        for i, layer in enumerate(self.model.layers):
            context = self.model.mix(context_states, i, layer.dflare_norm)
            layer.self_attn.store_context_kv(
                context,
                context_positions,
                self.model.cos_sin_cache,
                None if context_slot_mappings is None else context_slot_mappings[i],
            )

    def compute_f_rnn(self, features: torch.Tensor) -> torch.Tensor:
        """Recurrent-head context from the last committed token's features."""
        return self.model.mix(features, -1, self.model.markov_head.dflare_norm)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        inputs_embeds: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return self.model(input_ids, positions)

    def compute_base_logits(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """TP-local BF16 base logits before the Markov bias and softcap."""
        assert self.lm_head is not None
        if (
            self.w4_base_head is not None
            and hidden_states.ndim == 2
            and 1 <= hidden_states.shape[0] <= 64
            and hidden_states.dtype == torch.bfloat16
        ):
            from .ops.turbo_kernels import w4_wmma_linear

            return w4_wmma_linear(hidden_states, self.w4_base_head)
        if (
            self.w6_base_head is not None
            and hidden_states.ndim == 2
            and 1 <= hidden_states.shape[0] <= 8
            and hidden_states.dtype == torch.bfloat16
        ):
            from .ops.groupwise_w6_head import groupwise_w6_head_batched_linear

            return groupwise_w6_head_batched_linear(
                hidden_states,
                *self.w6_base_head,
                out_dtype=torch.bfloat16,
                num_warps=self.w6_base_head_waves,
                tile_m=8,
            )
        return nn.functional.linear(hidden_states, self.lm_head.weight)

    def markov_step(
        self, state: torch.Tensor, prev_tokens: torch.Tensor, context: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Advance the recurrent head; return the state and TP-local bias."""
        return self.model.markov_head.step(state, prev_tokens, context)

    def markov_step_greedy(
        self, state: torch.Tensor, prev_tokens: torch.Tensor,
        context: torch.Tensor, base_logits: torch.Tensor,
        cand_ids: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        head = self.model.markov_head
        assert head.fold_bias is not None
        state, latent = head.step_latent(state, prev_tokens, context)
        if head.w4_fold_weight is not None:
            from .ops.turbo_kernels import topk_markov_argmax, w4_markov_argmax

            if cand_ids is not None:
                next_token = topk_markov_argmax(
                    latent, *head.w4_fold_weight, head.fold_bias, base_logits,
                    cand_ids, self.config.final_logit_softcapping,
                )
            else:
                next_token = w4_markov_argmax(
                    latent, *head.w4_fold_weight, head.fold_bias, base_logits,
                    self.config.final_logit_softcapping, self.valid_vocab,
                )
            return state, next_token
        from .ops.markov_argmax import fused_markov_argmax

        weight = head.w6_fold_weight or head.w8_fold_weight
        assert weight is not None
        next_token = fused_markov_argmax(
            latent, weight, head.fold_bias, base_logits,
            self.config.final_logit_softcapping, self.valid_vocab,
        )
        return state, next_token

    def markov_logits(
        self, base_logits: torch.Tensor, bias: torch.Tensor
    ) -> torch.Tensor:
        """Full-vocab softcapped logits."""
        logits = add_softcap(base_logits, bias, self.config.final_logit_softcapping)
        logits = tensor_model_parallel_all_gather(logits)
        return logits[:, : self.config.vocab_size]

    def markov_argmax(
        self, base_logits: torch.Tensor, bias: torch.Tensor
    ) -> torch.Tensor:
        """`markov_logits(...).argmax(-1)`, NaN as -inf."""
        return dspark_argmax(
            base_logits, bias, self.config.final_logit_softcapping, self.valid_vocab
        )

    def map_draft_to_target(self, draft_ids: torch.Tensor) -> torch.Tensor:
        return draft_ids

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        params = dict(self.named_parameters())
        loaded: set[str] = set()
        for name, weight in weights:
            name, shard = f"model.{name}", None
            for old, new, shard_id in self._stacked:
                if old in name:
                    name, shard = name.replace(old, new), shard_id
                    break
            param = params[name]
            if weight.dtype != param.dtype:
                raise ValueError(f"{name}: expected {param.dtype}, got {weight.dtype}")
            if shard is None:
                getattr(param, "weight_loader", default_weight_loader)(param, weight)
            else:
                param.weight_loader(param, weight, shard)
            loaded.add(name)
        return loaded
