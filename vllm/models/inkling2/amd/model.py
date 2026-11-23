# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Eager TP1 text backend for Inkling 2 Mini on ROCm."""

import os
from collections import Counter
from collections.abc import Callable, Iterable, Sequence
from pathlib import Path

import regex as re
import torch
from torch import nn

from vllm.config import VllmConfig
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.vocab_parallel_embedding import (
    ParallelLMHead,
    VocabParallelEmbedding,
)
from vllm.model_executor.model_loader.weight_utils import default_weight_loader
from vllm.model_executor.models.interfaces import (
    MultiModalEmbeddings,
    SupportsEagle3,
    SupportsMultiModal,
)
from vllm.model_executor.models.utils import _merge_multimodal_embeddings
from vllm.multimodal import MULTIMODAL_REGISTRY
from vllm.sequence import IntermediateTensors

from ..configs import Inkling2MMConfig, Inkling2ModelConfig
from ..processing import (
    IMAGE_PLACEHOLDER,
    Inkling2DummyInputsBuilder,
    Inkling2MultiModalProcessor,
    Inkling2ProcessingInfo,
)
from ..quant_config import Inkling2QuantConfig
from .vision import Inkling2VisionEncoder
from .attention import Inkling2Attention
from .conv_state import InklingConvState
from .moe import PROJECTIONS, Inkling2MoE, QuantFormat, coda_c1_paths
from .ops.attention import compute_log_scaling_tau
from .ops.lossless_bf16 import lossless_bf16_gemv, pack_lossless_bf16
from .ops.loop import attn_res, blend_rms_norm, pair_rms_norm
from .ops.norm import rms_norm
from .sconv import ATTN_STREAM, MLP_STREAM, Inkling2ShortConv


class Inkling2RMSNorm(nn.Module):
    """RMSNorm; `num_inputs > 1` stacks the weights of per-input norms (the
    block AttnRes mixes apply them in the same kernel)."""

    def __init__(
        self,
        hidden_size: int,
        eps: float,
        out_dtype: torch.dtype | None = torch.bfloat16,
        num_inputs: int = 1,
    ) -> None:
        super().__init__()
        self.weight = nn.Parameter(
            torch.ones(num_inputs, hidden_size), requires_grad=False
        )
        self.weight.weight_loader = self._load
        self.eps = eps
        self.out_dtype = out_dtype

    @staticmethod
    def _load(param: nn.Parameter, loaded: torch.Tensor, shard_id: int = 0) -> None:
        param.data[shard_id].copy_(loaded)

    def forward(self, x: torch.Tensor, index: int = 0) -> torch.Tensor:
        return rms_norm(x, self.weight[index], self.eps, self.out_dtype)


class Inkling2AttnRes(nn.Module):
    """Pseudo-queries of the block attention residuals (fp32), pre-scaled by
    `hidden_size**-0.5` at load time."""

    def __init__(self, config: Inkling2ModelConfig) -> None:
        super().__init__()
        hidden = config.hidden_size
        local_ids = set(config.local_layer_ids)
        self.attn_queries = nn.ParameterList(
            nn.Parameter(
                torch.zeros(
                    (1, hidden) if i in local_ids else (3, hidden),
                    dtype=torch.float32,
                ),
                requires_grad=False,
            )
            for i in range(1, config.num_hidden_layers)
        )
        self.mlp_queries = nn.ParameterList(
            nn.Parameter(torch.zeros(1, hidden, dtype=torch.float32), False)
            for _ in range(config.num_hidden_layers)
        )
        self.output_query = nn.Parameter(
            torch.zeros(1, hidden, dtype=torch.float32), requires_grad=False
        )
        for param in self.parameters():
            param.weight_loader = self._load_query
        self.scale = hidden**-0.5

    def _load_query(self, param: nn.Parameter, weight: torch.Tensor) -> None:
        param.data.copy_(weight.view(param.shape)).mul_(self.scale)


class Inkling2DecoderLayer(nn.Module):
    def __init__(
        self,
        config: Inkling2ModelConfig,
        layer_id: int,
        quant_config: Inkling2QuantConfig | None,
        prefix: str,
        num_passes: int = 1,
        *,
        coda_c1_path: Path | None = None,
    ) -> None:
        """`num_passes > 1` (a loop core layer) gives each pass its own KV and
        conv-state caches."""
        super().__init__()
        self.is_local = layer_id in config.local_layer_ids
        num_kv_heads = (
            config.swa_num_key_value_heads
            if self.is_local
            else config.num_key_value_heads
        )
        self.conv_states = tuple(
            InklingConvState(
                num_kv_heads=num_kv_heads,
                head_dim=config.swa_head_dim if self.is_local else config.head_dim,
                hidden_size=config.hidden_size,
                kernel_size=config.sconv_kernel_size,
                prefix=f"{prefix}.conv_state" + (f".pass{p}" if p else ""),
                replicate_hidden=True,
            )
            for p in range(num_passes)
        )
        self.conv_state = self.conv_states[0]
        self.attn = Inkling2Attention(
            config,
            self.is_local,
            self.conv_state,
            prefix=f"{prefix}.attn",
            pass_conv_states=self.conv_states[1:],
        )
        eps = config.rms_norm_eps
        # Local layers: attn_norm; full layers: q/k/v_input_norm, stacked.
        self.num_attn_inputs = 1 if self.is_local else 3
        self.attn_norm = Inkling2RMSNorm(
            config.hidden_size, eps, num_inputs=self.num_attn_inputs
        )
        self.mlp_norm = Inkling2RMSNorm(config.hidden_size, eps)
        expert_format: QuantFormat = (
            "mxfp8"
            if quant_config is not None and quant_config.is_mxfp8_layer(layer_id)
            else "nvfp4"
        )
        self.mlp = Inkling2MoE(
            config,
            expert_format,
            config.layer_visit_ids(layer_id),
            layer_idx=layer_id,
            coda_c1_path=coda_c1_path,
        )
        self.attn_sconv, self.mlp_sconv = (
            Inkling2ShortConv(
                config.hidden_size, config.sconv_kernel_size, self.conv_state, stream
            )
            for stream in (ATTN_STREAM, MLP_STREAM)
        )

    def attention(
        self,
        positions: torch.Tensor,
        inputs: Sequence[torch.Tensor],
        tau: torch.Tensor | None,
        pass_idx: int,
    ) -> torch.Tensor:
        if self.is_local and len(inputs) == 1:
            return self.attn(positions, inputs[0], pass_idx=pass_idx)
        return self.attn(
            positions, inputs[0], inputs[1], inputs[2],
            None if self.is_local else tau, pass_idx=pass_idx,
        )


LOOP_STREAMS = ("attn", "k", "v", "mlp")


class Inkling2MessageController(nn.Module):
    """Blends a loop message into a core sublayer input: from the rank-`r`
    projection `h` of the RMS-normalized `[dest, message]`,
    `(1 - tanh(forget(h))) * dest + tanh(gain(h)) * message` in fp32.

    The weights are views of the layer's stream-stacked weights
    (`Inkling2LoopDecoderLayer`), which run every stream at once; `forward`
    is the per-stream reference.
    """

    def __init__(
        self, down: torch.Tensor, gain: torch.Tensor, forget: torch.Tensor
    ) -> None:
        super().__init__()
        self.down_weight = nn.Parameter(down, requires_grad=False)
        self.gain_weight = nn.Parameter(gain, requires_grad=False)
        self.forget_weight = nn.Parameter(forget, requires_grad=False)

class Inkling2LoopDecoderLayer(Inkling2DecoderLayer):
    """A loop core layer: per-pass caches and a message controller per
    stream (`LOOP_STREAMS`), whose weights are stacked over the streams."""

    def __init__(
        self,
        config: Inkling2ModelConfig,
        layer_id: int,
        quant_config: Inkling2QuantConfig | None,
        prefix: str,
    ) -> None:
        super().__init__(
            config, layer_id, quant_config, prefix, num_passes=config.num_loop_passes
        )
        assert config.loop_config is not None
        rank, width = config.loop_config["gate_rank"], config.hidden_size
        streams = len(LOOP_STREAMS)
        self.use_packed_loop = os.getenv("INKLING_AMD_PACKED_LOOP") == "1"
        self.loop_m8_fused = os.getenv("INKLING_AMD_LOOP_M8_FUSED") == "1"
        self.loop_gemv = os.getenv("INKLING_AMD_LOOP_GEMV", "0")
        if self.loop_gemv not in ("0", "gain", "both"):
            raise ValueError("Select 0, gain or both for INKLING_AMD_LOOP_GEMV")
        self.register_buffer("controller_gain_forget_fp32", None, persistent=False)
        # down (S, r, 2D) and [gain; forget] (S, 2D, r), bf16.
        self.register_buffer(
            "controller_down",
            torch.empty(streams, rank, 2 * width, dtype=torch.bfloat16),
            persistent=False,
        )
        self.register_buffer(
            "controller_gain_forget",
            torch.empty(streams, 2 * width, rank, dtype=torch.bfloat16),
            persistent=False,
        )
        self.controllers = nn.ModuleDict(
            {
                stream: Inkling2MessageController(
                    self.controller_down[i],
                    self.controller_gain_forget[i, :width],
                    self.controller_gain_forget[i, width:],
                )
                for i, stream in enumerate(LOOP_STREAMS)
            }
        )

    def process_weights_after_loading(self) -> None:
        # The controllers' weights must still alias the stacked ones.
        for i, controller in enumerate(self.controllers.values()):
            down, gain_forget = self.controller_down, self.controller_gain_forget
            assert controller.down_weight.data_ptr() == down[i].data_ptr()
            assert controller.gain_weight.data_ptr() == gain_forget[i].data_ptr()
        self.controller_down_t = None
        if self.use_packed_loop and self.controller_gain_forget.is_cuda:
            # This is the same FP32 widening that the eager path does per visit.
            self.controller_gain_forget_fp32 = self.controller_gain_forget.float()
            self.controller_down_t = self.controller_down.transpose(1, 2).contiguous()

    def blend_zeromat(
        self,
        streams: slice,
        dest: torch.Tensor,
        bank0: torch.Tensor,
        bank1: torch.Tensor,
        coeffs: torch.Tensor,
        norm_weight: torch.Tensor,
        norm_eps: float,
    ) -> torch.Tensor | None:
        down_t = getattr(self, "controller_down_t", None)
        if down_t is None and self.controller_down.is_cuda:
            down_t = self.controller_down.transpose(1, 2).contiguous()
            self.controller_down_t = down_t
        if down_t is None:
            return None
        from .ops.turbo_kernels import fused_loop_zeromat_prefill

        return fused_loop_zeromat_prefill(
            dest, bank0, bank1, coeffs,
            down_t[streams], self.controller_gain_forget[streams],
            norm_weight, norm_eps,
        )

    def blend(
        self,
        streams: slice,
        dest: torch.Tensor,
        message: torch.Tensor,
        norm_weight: torch.Tensor,
        norm_eps: float,
    ) -> torch.Tensor:
        """The `streams` controllers on fp32 `(S, T, D)` `dest` (may
        broadcast over streams) and `message`, each output RMS-normalized
        with its row of `norm_weight` `(S, D)`: bf16 `(S, T, D)`."""
        if (
            self.use_packed_loop
            and message.shape[1] >= 64
            and os.getenv("INKLING_AMD_TURBO_PREFILL_ATTN") == "1"
            and os.getenv("INKLING_AMD_FUSED_LOOP_ALLINONE", "1") != "0"
        ):
            down_t = getattr(self, "controller_down_t", None)
            if down_t is None and self.controller_down.is_cuda:
                down_t = self.controller_down.transpose(1, 2).contiguous()
                self.controller_down_t = down_t
            if down_t is not None:
                from .ops.turbo_kernels import fused_loop_allinone_prefill

                result = fused_loop_allinone_prefill(
                    dest, message, down_t[streams],
                    self.controller_gain_forget[streams], norm_weight, norm_eps,
                )
                if result is not None:
                    return result
        pair = pair_rms_norm(dest, message, packed=self.use_packed_loop)
        down_weight = self.controller_down[streams]
        gain_forget = None
        if self.loop_gemv != "0" and pair.shape[1] == 1:
            from .ops.loop_controller import loop_controller_decode

            gain_forget = loop_controller_decode(
                pair, down_weight, self.controller_gain_forget[streams],
                packed_down=self.loop_gemv == "both",
            )
        if gain_forget is None:
            if (
                self.loop_m8_fused
                and self.use_packed_loop
                and 2 <= pair.shape[1] <= 8
                and os.getenv("INKLING_AMD_TURBO_ROUTER_M8") == "1"
            ):
                from .ops.turbo_kernels import controller_down_m8

                hidden = controller_down_m8(pair, down_weight)
            else:
                hidden = torch.bmm(pair, down_weight.transpose(1, 2))
            if self.use_packed_loop and 1 <= pair.shape[1] <= 64:
                from .ops.hip_runtime import hip_loop_decode_blend_rms

                result = hip_loop_decode_blend_rms(
                    hidden, self.controller_gain_forget[streams],
                    dest, message, norm_weight, norm_eps,
                )
                if result is not None:
                    return result
            if self.loop_m8_fused and self.use_packed_loop and pair.shape[1] == 8:
                from .ops.loop_m8_fused import fused_loop_m8_blend

                result = fused_loop_m8_blend(
                    hidden, self.controller_gain_forget[streams],
                    dest, message, norm_weight, norm_eps,
                )
                if result is not None:
                    return result
            if (
                self.use_packed_loop
                and pair.shape[1] >= 16
                and os.getenv("INKLING_AMD_TURBO_PREFILL_ATTN") == "1"
            ):
                from .ops.turbo_kernels import fused_loop_prefill_blend

                result = fused_loop_prefill_blend(
                    hidden, self.controller_gain_forget[streams],
                    dest, message, norm_weight, norm_eps,
                )
                if result is not None:
                    return result
            weight_fp32 = self.controller_gain_forget_fp32
            gain_weight = (
                self.controller_gain_forget[streams].float()
                if weight_fp32 is None
                else weight_fp32[streams]
            )
            gain_forget = torch.bmm(hidden.float(), gain_weight.transpose(1, 2))
        return blend_rms_norm(
            dest, message, gain_forget, norm_weight, norm_eps,
            packed=self.use_packed_loop,
        )


class Inkling2LoopModel(nn.Module):

    def __init__(
        self,
        config: Inkling2ModelConfig,
        quant_config: Inkling2QuantConfig | None,
        prefix: str,
    ) -> None:
        # Serving options set through --hf-overrides skip the constructor.
        config.check_loop_options()
        super().__init__()
        coda_banks = coda_c1_paths()
        if any(layer < config.loop_core_layers.stop for layer in coda_banks):
            raise ValueError("Inkling coda C1 requires layers after the loop core")
        self.config = config
        self.block_size = config.n_layers_per_attn_res_block
        self.normalize_values = config.attn_res_normalize_values
        self.embed_tokens = VocabParallelEmbedding(
            config.padded_vocab_size, config.hidden_size
        )
        self.embed_norm = Inkling2RMSNorm(
            config.hidden_size, config.rms_norm_eps, out_dtype=None
        )
        self.layers = nn.ModuleList(
            self._make_layer(
                config, i, quant_config, f"{prefix}.layers.{i}", coda_banks.get(i)
            )
            for i in range(config.num_hidden_layers)
        )
        self.attn_res = Inkling2AttnRes(config)
        self.norm = Inkling2RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.loop_core = config.loop_core_layers
        self.hidden_size = config.hidden_size
        self.aux_hidden_state_layers: tuple[int, ...] = ()
        # Message pseudo-queries (raw), two per core layer: attn/mlp and k/v.
        shape = (2 * len(self.loop_core), config.hidden_size)
        self.middle_cycle_cross_pass_queries = nn.Parameter(
            torch.zeros(shape, dtype=torch.float32), requires_grad=False
        )
        self.middle_cycle_cross_pass_kv_queries = nn.Parameter(
            torch.zeros(shape, dtype=torch.float32), requires_grad=False
        )
        # Per core layer, the `LOOP_STREAMS` queries scaled by D**-0.5.
        self.register_buffer(
            "message_queries",
            torch.zeros(
                len(self.loop_core),
                len(LOOP_STREAMS),
                config.hidden_size,
                dtype=torch.float32,
            ),
            persistent=False,
        )

    def process_weights_after_loading(self) -> None:
        queries = self.middle_cycle_cross_pass_queries.view(-1, 2, self.hidden_size)
        kv_queries = self.middle_cycle_cross_pass_kv_queries.view(
            -1, 2, self.hidden_size
        )
        # (attn, k, v, mlp) = (queries[0], kv[0], kv[1], queries[1]).
        stacked = torch.stack(
            (queries[:, 0], kv_queries[:, 0], kv_queries[:, 1], queries[:, 1]), 1
        )
        self.message_queries.copy_(stacked * self.hidden_size**-0.5)

    def _make_layer(
        self,
        config: Inkling2ModelConfig,
        layer_id: int,
        quant_config: Inkling2QuantConfig | None,
        prefix: str,
        coda_c1_path: Path | None = None,
    ) -> Inkling2DecoderLayer:
        if layer_id in config.loop_core_layers:
            return Inkling2LoopDecoderLayer(config, layer_id, quant_config, prefix)
        return Inkling2DecoderLayer(
            config, layer_id, quant_config, prefix, coda_c1_path=coda_c1_path
        )

    def _max_loop_blocks(self) -> int:
        """Completed AttnRes blocks a loop forward holds at most: each pass
        restarts from the prelude's, and the coda continues the last pass."""
        num_blocks = max_blocks = 0
        core = self.loop_core
        for i in [*range(core.start), *core, *range(core.stop, len(self.layers))]:
            num_blocks += i % self.block_size == 0
            max_blocks = max(max_blocks, num_blocks)
        return max_blocks

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.embed_norm(self.embed_tokens(input_ids))

    def _mix(
        self, stack, last, query, norm, addend=None
    ):
        return attn_res(
            stack,
            last,
            addend,
            query,
            None if norm is None else norm.weight,
            0.0 if norm is None else norm.eps,
            self.normalize_values,
        )

    def _run_layer(
        self, layer_id, pass_idx, stack, num_blocks, inputs, bank,
        next_query, next_norm, positions, tau,
    ):
        layer = self.layers[layer_id]
        partial = stack[num_blocks]
        owner = layer.conv_states[pass_idx]
        messages = None
        msg_coeffs = None
        if bank is not None:
            assert isinstance(layer, Inkling2LoopDecoderLayer) and inputs is not None
            norm = layer.attn_norm
            destinations = inputs.expand(3, -1, -1) if layer.is_local else inputs
            norm_weight = norm.weight.expand(3, -1) if layer.is_local else norm.weight
            attn_inputs = None
            if (
                layer.use_packed_loop
                and partial.shape[0] >= 64
                and os.getenv("INKLING_AMD_TURBO_PREFILL_ATTN") == "1"
                and os.getenv("INKLING_AMD_FUSED_LOOP_ALLINONE", "1") != "0"
            ):
                from .ops.hip_moe_prefill import hip_attn_res_q4_coeffs

                msg_coeffs = hip_attn_res_q4_coeffs(
                    bank[0],
                    bank[1],
                    self.message_queries[layer_id - self.loop_core.start],
                )
                if msg_coeffs is not None:
                    attn_inputs = layer.blend_zeromat(
                        slice(0, 3),
                        destinations,
                        bank[0],
                        bank[1],
                        msg_coeffs[:3],
                        norm_weight,
                        norm.eps,
                    )
                    if attn_inputs is None:
                        msg_coeffs = None
            if attn_inputs is None:
                messages = attn_res(
                    bank[:1],
                    bank[1],
                    None,
                    self.message_queries[layer_id - self.loop_core.start],
                    None,
                    0.0,
                    True,
                )
                attn_inputs = layer.blend(
                    slice(0, 3), destinations, messages[:3], norm_weight, norm.eps
                )
        elif inputs is None:
            assert num_blocks == 0
            attn_inputs = [
                layer.attn_norm(partial, i) for i in range(layer.num_attn_inputs)
            ]
        else:
            attn_inputs = inputs

        attn_out = layer.attn_sconv(
            layer.attention(positions, attn_inputs, tau, pass_idx),
            positions,
            owner=owner,
        )
        first = layer_id % self.block_size == 0
        has_loop_msg = (messages is not None) or (msg_coeffs is not None)
        mlp_norm = layer.mlp_norm if not has_loop_msg else None
        query = self.attn_res.mlp_queries[layer_id]
        if first:
            mlp_inputs = self._mix(stack[:num_blocks + 1], attn_out, query, mlp_norm)
        else:
            mlp_inputs = self._mix(
                stack[:num_blocks], partial, query, mlp_norm, attn_out
            )
        if msg_coeffs is not None:
            norm = layer.mlp_norm
            zm_mlp = layer.blend_zeromat(
                slice(3, 4), mlp_inputs, bank[0], bank[1], msg_coeffs[3:], norm.weight, norm.eps
            )
            if zm_mlp is not None:
                mlp_inputs = zm_mlp
            else:
                messages = attn_res(
                    bank[:1],
                    bank[1],
                    None,
                    self.message_queries[layer_id - self.loop_core.start],
                    None,
                    0.0,
                    True,
                )
                mlp_inputs = layer.blend(
                    slice(3, 4), mlp_inputs, messages[3:], norm.weight, norm.eps
                )
        elif messages is not None:
            norm = layer.mlp_norm
            mlp_inputs = layer.blend(
                slice(3, 4), mlp_inputs, messages[3:], norm.weight, norm.eps
            )
        residual = None if first else partial
        num_blocks += first
        layer.mlp_sconv(
            layer.mlp(mlp_inputs[0], pass_idx=pass_idx),
            positions,
            addend=attn_out,
            residual=residual,
            out=stack[num_blocks],
            owner=owner,
        )
        outputs = self._mix(
            stack[:num_blocks], stack[num_blocks], next_query, next_norm
        )
        return num_blocks, outputs

    def forward(
        self, positions: torch.Tensor, inputs_embeds: torch.Tensor
    ) -> torch.Tensor | tuple[torch.Tensor, list[torch.Tensor]]:
        """Run the prelude, each core pass, and the coda."""
        num_tokens, hidden = inputs_embeds.shape
        core = self.loop_core
        num_passes = self.config.num_loop_passes
        stack = inputs_embeds.new_empty(
            (self._max_loop_blocks() + 1, num_tokens, hidden)
        )
        stack[0].copy_(inputs_embeds)
        tau = None
        if self.config.log_scaling_n_floor is not None:
            tau = compute_log_scaling_tau(
                positions, self.config.log_scaling_n_floor, self.config.log_scaling_alpha
            )
        # (layer, pass) visits; the prelude and coda run as pass 0.
        visits = [(i, 0) for i in range(core.start)]
        visits += [(i, p) for p in range(num_passes) for i in core]
        visits += [(i, 0) for i in range(core.stop, len(self.layers))]
        aux_hidden_states = []
        next_norm: nn.Module | None
        num_blocks = 0
        inputs = bank = prelude_blocks = prelude_partial = prelude_dests = None
        for visit, (i, pass_idx) in enumerate(visits):
            looped = pass_idx > 0
            if i == core.start and not looped:
                prelude_blocks = num_blocks
                prelude_partial = stack[num_blocks].clone()
            elif i == core.start:
                # Every pass restarts from the prelude's state, with the
                # previous pass's blocks (after the prelude's) and partial as
                # the messages' bank.
                assert prelude_blocks is not None and prelude_partial is not None
                bank = stack[prelude_blocks + 1 : num_blocks + 1].clone()
                num_blocks = prelude_blocks
                stack[num_blocks].copy_(prelude_partial)
                if prelude_dests is None:
                    query = self.attn_res.attn_queries[i - 1]
                    prelude_dests = attn_res(
                        stack[:num_blocks],
                        prelude_partial,
                        None,
                        query,
                        None,
                        0.0,
                        True,
                    )
                inputs = prelude_dests
            if visit in self.aux_hidden_state_layers:
                aux_hidden_states.append(stack[num_blocks].clone())
            if visit + 1 == len(visits):
                next_query, next_norm = self.attn_res.output_query, self.norm
            else:
                next_i, next_pass = visits[visit + 1]
                next_query = self.attn_res.attn_queries[next_i - 1]
                # Messages blend into fp32 mixes; a pass restart discards it.
                next_norm = None if next_pass else self.layers[next_i].attn_norm
            num_blocks, inputs = self._run_layer(
                i,
                pass_idx,
                stack,
                num_blocks,
                inputs,
                bank if looped else None,
                next_query,
                next_norm,
                positions,
                tau,
            )
        assert inputs is not None
        out = inputs[0]
        if self.aux_hidden_state_layers:
            return out, [*aux_hidden_states, out]
        return out


@MULTIMODAL_REGISTRY.register_processor(
    Inkling2MultiModalProcessor,
    info=Inkling2ProcessingInfo,
    dummy_inputs=Inkling2DummyInputsBuilder,
)
class Inkling2LoopForConditionalGeneration(
    nn.Module,
    SupportsMultiModal,
    SupportsEagle3,
):
    """Load the released Mini checkpoint through the vLLM text registry."""

    # Weights mirror the checkpoint names after these substitutions, except for
    # the fused per-layer weights in ``_shards``.
    _renames = (
        ("model.llm.unembed.", "lm_head."),
        ("model.llm.embed.", "model.embed_tokens."),
        ("model.llm.", "model."),
        (".q_l2_clip.scale", ".q_scale"),
        (".rel_l2_clip.scale", ".r_scale"),
        (".rel_logits_proj.proj", ".rel_logits_proj"),
    )
    _shards = {
        ".attn.wq_du.": (".attn.qkvr.", 0),
        ".attn.wr_du.": (".attn.qkvr.", 1),
        ".attn.gate_attn.": (".attn.qkvr.", 2),
        ".attn.attn_sink_proj.": (".attn.qkvr.", 3),
        ".attn.wk_dv.": (".attn.qkvr.", 4),
        ".attn.wv_dv.": (".attn.qkvr.", 5),
        ".q_input_norm.": (".attn_norm.", 0),
        ".k_input_norm.": (".attn_norm.", 1),
        ".v_input_norm.": (".attn_norm.", 2),
    }

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        super().__init__()
        model_config = vllm_config.model_config
        parallel = vllm_config.parallel_config
        config: Inkling2MMConfig = model_config.hf_config
        quant_config = vllm_config.quant_config
        if parallel.tensor_parallel_size != 1 or parallel.pipeline_parallel_size != 1:
            raise ValueError("The ROCm Mini reference requires TP1 and PP1")
        spec = vllm_config.speculative_config
        if not model_config.enforce_eager:
            plain_graph = os.getenv("INKLING_AMD_DECODE_GRAPH") == "1" and spec is None
            target_graph = (
                os.getenv("INKLING_AMD_DSPARK_TARGET_GRAPH") == "1"
                and spec is not None
                and spec.method == "dspark"
                and 1 <= spec.num_speculative_tokens <= 7
                and spec.draft_sample_method in ("greedy", "probabilistic")
                and parallel.data_parallel_size == 1
            )
            if not (1 <= vllm_config.scheduler_config.max_num_seqs <= 8) or not (
                plain_graph or target_graph
            ):
                raise ValueError(
                    "The ROCm Mini graph path requires 1..8 sequences with plain "
                    "decode or DSpark speculation"
                )
        if spec is not None and (
            spec.method != "dspark"
            or spec.draft_model_config.architectures != ["Inkling2DSparkModel"]
        ):
            raise ValueError(
                "The ROCm Mini reference only supports the native Inkling DSpark draft"
            )
        if model_config.enable_prompt_embeds:
            raise ValueError("The ROCm Mini reference expects token IDs")
        if config.text_config.loop_config is None or not isinstance(
            quant_config, Inkling2QuantConfig
        ):
            raise ValueError("Expected a quantized Inkling 2 Mini loop checkpoint")
        self.config = config
        self.text_config = text_config = config.text_config
        mm_cfg = model_config.multimodal_config
        self.visual = (
            Inkling2VisionEncoder(config.vision_config, text_config.hidden_size)
            if config.vision_config is not None
            and mm_cfg is not None
            and mm_cfg.get_limit_per_prompt("image")
            else None
        )
        self.model = Inkling2LoopModel(text_config, quant_config, f"{prefix}model")
        self.lm_head = ParallelLMHead(text_config.padded_vocab_size, text_config.hidden_size)
        self.logits_processor = LogitsProcessor(
            text_config.padded_vocab_size,
            org_vocab_size=text_config.vocab_size,
            soft_cap=text_config.final_logit_softcapping,
        )
        self.packed_head = None
        head_formats = (
            ("lossless", "INKLING_AMD_LOSSLESS_HEAD"),
            ("w8", "INKLING_AMD_GROUPWISE_W8_HEAD"),
            ("w6", "INKLING_AMD_GROUPWISE_W6_HEAD"),
        )
        enabled_heads = [name for name, flag in head_formats if os.getenv(flag) == "1"]
        if len(enabled_heads) > 1:
            raise ValueError("Select one Inkling AMD output-head format")
        self.packed_head_format = next(iter(enabled_heads), None)
        self.w6_head_waves = int(os.getenv("INKLING_AMD_W6_HEAD_WAVES", "4"))
        self.w6_head_fused_softcap = os.getenv("INKLING_AMD_W6_HEAD_FUSED_SOFTCAP") == "1"
        self.w6_head_batch_tile = int(os.getenv("INKLING_AMD_W6_HEAD_BATCH_TILE", "0"))
        if self.w6_head_batch_tile not in (0, 2, 4, 8):
            raise ValueError("Inkling batched W6 head tile must be 0, 2, 4 or 8")
        self.packed_logits_processor = (
            LogitsProcessor(
                text_config.padded_vocab_size,
                org_vocab_size=text_config.vocab_size,
                soft_cap=text_config.final_logit_softcapping,
                logits_as_input=True,
            )
            if self.packed_head_format is not None
            else None
        )

    def process_weights_after_loading(self) -> None:
        if self.packed_head_format == "w8":
            from .ops.groupwise_w8_head import pack_groupwise_w8_head

            self.packed_head = pack_groupwise_w8_head(self.lm_head.weight)
        elif self.packed_head_format == "w6":
            from .ops.groupwise_w6_head import pack_groupwise_w6_head

            group_size = int(os.getenv("INKLING_AMD_W6_HEAD_GROUP_SIZE", "64"))
            self.packed_head = pack_groupwise_w6_head(self.lm_head.weight, group_size=group_size)
            if os.getenv("INKLING_AMD_TURBO_W6_WMMA") == "1":
                from .ops.turbo_kernels import pack_w6_wmma

                self.w6_wmma_head = pack_w6_wmma(self.lm_head.weight)
            else:
                self.w6_wmma_head = None
        elif self.packed_head_format == "lossless":
            weight = self.lm_head.weight
            self.packed_head = pack_lossless_bf16(weight.detach().cpu()).to(weight.device)
            print(f"Inkling ROCm lossless BF16 LM head: {self.packed_head.storage_bytes} bytes", flush=True)

    @classmethod
    def get_placeholder_str(cls, modality: str, i: int) -> str | None:
        if modality.startswith("image"):
            return IMAGE_PLACEHOLDER
        raise ValueError("Only image modality is supported")

    def embed_multimodal(self, **kwargs: object) -> MultiModalEmbeddings:
        patches = kwargs.get("vision_pixels")
        grid_thw = kwargs.get("vision_grid_thw")
        if patches is None or grid_thw is None:
            return ()
        assert isinstance(patches, torch.Tensor)
        assert isinstance(grid_thw, torch.Tensor)
        assert self.visual is not None
        embeds = self.visual(patches, grid_thw)
        return embeds.split(grid_thw.prod(-1).tolist())

    def embed_input_ids(
        self,
        input_ids: torch.Tensor,
        multimodal_embeddings: MultiModalEmbeddings | None = None,
        *,
        is_multimodal: torch.Tensor | None = None,
    ) -> torch.Tensor:
        inputs_embeds = self.model.embed_input_ids(input_ids)
        if not multimodal_embeddings:
            return inputs_embeds
        assert is_multimodal is not None
        return _merge_multimodal_embeddings(
            inputs_embeds=inputs_embeds,
            multimodal_embeddings=multimodal_embeddings,
            is_multimodal=is_multimodal,
        )

    def get_language_model(self) -> nn.Module:
        return self

    def set_aux_hidden_state_layers(self, layers: tuple[int, ...]) -> None:
        self.model.aux_hidden_state_layers = tuple(layers)

    def forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
        **kwargs: object,
    ) -> torch.Tensor | tuple[torch.Tensor, list[torch.Tensor]]:
        if inputs_embeds is None:
            assert input_ids is not None
            inputs_embeds = self.embed_input_ids(input_ids)
        return self.model(positions, inputs_embeds)

    def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor | None:
        has_w6_wmma = getattr(self, "w6_wmma_head", None) is not None
        batched_w6 = (
            (self.w6_head_batch_tile > 0 or has_w6_wmma)
            and self.packed_head_format == "w6"
            and hidden_states.ndim == 2
            and hidden_states.shape[1] == self.text_config.hidden_size
            and 2 <= hidden_states.shape[0] <= (64 if has_w6_wmma else 8)
        )
        if (
            self.packed_head is not None
            and (batched_w6 or hidden_states.shape == (1, self.text_config.hidden_size))
            and hidden_states.is_cuda
            and hidden_states.dtype == torch.bfloat16
            and self.logits_processor.head_dtype in (None, torch.bfloat16, torch.float32)
        ):
            out_dtype = self.logits_processor.head_dtype or hidden_states.dtype
            if self.packed_head_format == "w8":
                from .ops.groupwise_w8_head import groupwise_w8_head_linear

                logits = groupwise_w8_head_linear(
                    hidden_states, *self.packed_head, out_dtype=out_dtype
                )
            elif self.packed_head_format == "w6":
                from .ops.groupwise_w6_head import (
                    groupwise_w6_head_batched_linear,
                    groupwise_w6_head_linear,
                )

                soft_cap = 0.0
                if (
                    self.w6_head_fused_softcap
                    and out_dtype == torch.float32
                    and self.packed_logits_processor.scale == 1.0
                ):
                    soft_cap = self.packed_logits_processor.soft_cap or 0.0
                if batched_w6 and getattr(self, "w6_wmma_head", None) is not None:
                    from .ops.turbo_kernels import w6_wmma_linear

                    logits = w6_wmma_linear(
                        hidden_states, self.w6_wmma_head,
                        out_dtype=out_dtype, soft_cap=soft_cap,
                    )
                elif batched_w6:
                    logits = groupwise_w6_head_batched_linear(
                        hidden_states, *self.packed_head, out_dtype=out_dtype,
                        num_warps=self.w6_head_waves, soft_cap=soft_cap,
                        tile_m=min(self.w6_head_batch_tile, 1 << (int(hidden_states.shape[0]) - 1).bit_length()),
                    )
                else:
                    logits = groupwise_w6_head_linear(
                        hidden_states, *self.packed_head, out_dtype=out_dtype,
                        num_warps=self.w6_head_waves, soft_cap=soft_cap,
                    )
                if soft_cap:
                    return logits[..., : self.text_config.vocab_size]
            else:
                logits = lossless_bf16_gemv(
                    hidden_states.contiguous(), self.packed_head, variant="paired", out_dtype=out_dtype
                )
            return self.packed_logits_processor(
                self.lm_head, logits[..., : self.text_config.vocab_size]
            )
        return self.logits_processor(self.lm_head, hidden_states)

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        params = dict(self.named_parameters())
        num_layers = self.text_config.num_hidden_layers
        coda_source_names = {
            f"model.layers.{layer_idx}.mlp.experts.{projection}_{suffix}"
            for layer_idx, layer in enumerate(self.model.layers)
            if layer.mlp.experts.coda_c1_path is not None
            for projection in PROJECTIONS
            for suffix in ("weight", "weight_scale")
        }
        # Truncated models (e.g. fewer layers via hf_overrides) skip the rest.
        layer_limits = {
            "layers": num_layers,
            "attn_queries": num_layers - 1,
            "mlp_queries": num_layers,
        }
        loaded: set[str] = set()
        loaded_shards: set[tuple[str, int]] = set()
        visual_weights: list[tuple[str, torch.Tensor]] = []
        for name, weight in weights:
            if name.startswith("model.visual."):
                if self.visual is not None:
                    visual_weights.append((name.removeprefix("model.visual."), weight))
                continue
            for old, new in self._renames:
                name = name.replace(old, new)
            if name in coda_source_names:
                continue
            shard_id = None
            for old, (new, shard) in self._shards.items():
                if old in name:
                    name = name.replace(old, new)
                    shard_id = shard
                    break
            if name not in params:
                m = re.search(
                    r"\.(layers|attn_queries|mlp_queries)\.(\d+)(?:\.|$)", name
                )
                if m is not None and int(m.group(2)) >= layer_limits[m.group(1)]:
                    continue
                raise ValueError(f"Unexpected checkpoint tensor: {name}")
            param = params[name]
            if param.dtype != weight.dtype:
                if ".mlp.experts." in name and weight.dtype == torch.bfloat16:
                    raise ValueError(
                        f"{name}: BF16 expert weights are not supported; the "
                        "Inkling2 experts are NVFP4 / MXFP8, as declared per "
                        "tensor in the checkpoint's hf_quant_config.json"
                    )
                raise ValueError(
                    f"{name}: checkpoint dtype {weight.dtype} != {param.dtype}"
                )
            loader: Callable[..., None] = getattr(
                param, "weight_loader", default_weight_loader
            )
            if shard_id is None:
                loader(param, weight)
                loaded.add(name)
            else:
                loader(param, weight, shard_id)
                loaded_shards.add((name, shard_id))
        num_shards = Counter(new for new, _ in self._shards.values())
        for name, count in Counter(name for name, _ in loaded_shards).items():
            fused = next(f for f in num_shards if f in name)
            if count == num_shards[fused]:
                loaded.add(name)
        if self.visual is not None:
            loaded |= {f"visual.{n}" for n in self.visual.load_weights(visual_weights)}
        if missing := params.keys() - loaded:
            raise ValueError(f"Missing checkpoint tensors: {sorted(missing)}")
        return loaded
