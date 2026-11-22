# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Modified or added by Ciru for Inkling 2 Mini on AMD gfx1151.
"""DSpark speculator for the Inkling2 draft: stores the raw target features
(each draft layer mixes them itself) and runs its recurrent Markov head."""

import os

import torch
import torch.nn as nn

from vllm.config import VllmConfig
from vllm.v1.worker.gpu.spec_decode.dspark.speculator import DSparkSpeculator
from vllm.v1.worker.gpu.spec_decode.dspark.utils import load_dspark_model


class Inkling2DSparkSpeculator(DSparkSpeculator):
    _speculator_name = "Inkling2 DSpark"

    def __init__(self, vllm_config: VllmConfig, device: torch.device):
        super().__init__(vllm_config, device)
        self._require_full_verify_for_graph = (
            os.getenv("INKLING_AMD_DSPARK_DRAFT_GRAPH") == "1"
        )
        config = self.draft_model_config.hf_config
        assert not self.sample_from_anchor
        assert self._draft_topk is None and not self.enable_adaptive_verification
        self.hidden_states = torch.zeros(
            self.max_num_tokens,
            len(config.target_layer_ids) * config.hidden_size,
            dtype=self.dtype,
            device=device,
        )
        # combine_hidden_states is the identity: store features eagerly.
        self._aux_staging = None
        # Read by prepare_context_anchor inside the FULL draft graph.
        self._num_rejected = torch.zeros(
            self.max_num_reqs, dtype=torch.int32, device=device
        )
        self.f_rnn = torch.zeros(
            self.max_num_reqs, config.hidden_size, dtype=self.dtype, device=device
        )
        self._max_spec_steps = self.num_speculative_steps
        self._active_n_spec = self.num_speculative_steps
        self._adaptive_depth_max_reqs: dict[int, int] = {}
        self._adaptive_anchors: dict[int, torch.Tensor] = {}
        self._adaptive_sample_cols: dict[int, torch.Tensor] = {}
        self._adaptive_step_cols: dict[int, torch.Tensor] = {}
        self._adaptive_managers: dict[int, object] = {}
        spec_cfg = vllm_config.speculative_config
        if spec_cfg is not None and spec_cfg.uses_dynamic_speculative_decoding():
            from vllm.v1.spec_decode.dynamic.utils import build_dynamic_sd_schedule_lookup

            dense = build_dynamic_sd_schedule_lookup(
                spec_cfg.num_speculative_tokens_per_batch_size,
                vllm_max_batch_size=self.max_num_reqs,
                vllm_num_speculative_tokens=self._max_spec_steps,
            )
            for b_sz, k_val in enumerate(dense):
                if b_sz > 0 and k_val > 0:
                    self._adaptive_depth_max_reqs[k_val] = max(
                        self._adaptive_depth_max_reqs.get(k_val, 0), b_sz
                    )
            if len(self._adaptive_depth_max_reqs) > 1:
                for k_val in self._adaptive_depth_max_reqs:
                    q_per_req = 1 + k_val
                    self._adaptive_anchors[k_val] = (
                        torch.arange(self.max_num_reqs, dtype=torch.int64, device=device)
                        * q_per_req
                    )
                    self._adaptive_sample_cols[k_val] = torch.arange(
                        k_val, dtype=torch.int32, device=device
                    ).repeat(self.max_num_reqs)
                    self._adaptive_step_cols[k_val] = torch.arange(
                        k_val, dtype=torch.int32, device=device
                    )

    def draft_logits_spec(self, vllm_config: VllmConfig) -> tuple[torch.dtype, float]:
        # Draft base logits come from a bf16 GEMM, unlike the target's fp32 head.
        return self.dtype, 0.0

    def load_draft_model(
        self, target_model: nn.Module, target_attn_layer_names: set[str]
    ) -> nn.Module:
        model = load_dspark_model(target_model, self.vllm_config)
        model.attach_target(target_model.get_language_model())
        return model

    def prepare_context_anchor(
        self,
        num_reqs: int,
        query_start_loc: torch.Tensor,
        num_rejected: torch.Tensor | None,
    ) -> None:
        assert num_rejected is not None
        ends = query_start_loc[1 : num_reqs + 1] - num_rejected[:num_reqs]
        rows = self.hidden_states[(ends - 1).clamp_min(0).long()]
        self.f_rnn[:num_reqs] = self.model.compute_f_rnn(rows)

    def _sample_sequential_body_c1(self, head_hidden: torch.Tensor) -> None:
        n_spec = self.num_speculative_steps
        hidden = head_hidden[self.sample_indices[:n_spec]]
        base_logits = self.model.compute_base_logits(hidden).view(1, n_spec, -1)
        context = self.model.model.markov_head.context(hidden, self.f_rnn[:1])
        state = hidden.new_zeros(1, self.model.config.markov_rank)
        prev = self.input_buffers.input_ids[self._anchor_idx[:1]]
        for i in range(n_spec):
            state, prev = self.model.markov_step_greedy(
                state, prev, context[i], base_logits[:, i], cand_ids=None,
            )
            self.draft_tokens[:1, i] = prev

    def _select_depth(self, k: int) -> None:
        self.num_speculative_steps = k
        self.num_query_per_req = 1 + k
        self._active_n_spec = k
        if k in self._adaptive_anchors:
            self._anchor_idx = self._adaptive_anchors[k]
            self.sample_col = self._adaptive_sample_cols[k]
            self._step_cols = self._adaptive_step_cols[k]
        if k in self._adaptive_managers:
            self.query_cudagraph_manager = self._adaptive_managers[k]
        if getattr(self, "model", None) is not None and hasattr(self.model, "set_active_block_size"):
            self.model.set_active_block_size(1 + k)

    def init_cudagraph_manager(self, cudagraph_mode) -> None:
        if len(self._adaptive_anchors) <= 1:
            super().init_cudagraph_manager(cudagraph_mode)
            return
        from vllm.config.compilation import CUDAGraphMode
        from vllm.v1.attention.backend import AttentionCGSupport
        from vllm.v1.worker.gpu.spec_decode.dflash.cudagraph import DFlashCudaGraphManager

        wants_full = cudagraph_mode.decode_mode() == CUDAGraphMode.FULL
        supports_full = (
            self.attn_cg_support.min_cg_support.value
            >= AttentionCGSupport.UNIFORM_BATCH.value
        )
        resolved = CUDAGraphMode.FULL_DECODE_ONLY if (wants_full and supports_full) else CUDAGraphMode.NONE
        for k_val in sorted(self._adaptive_depth_max_reqs.keys(), reverse=True):
            max_r = self._adaptive_depth_max_reqs[k_val]
            self._adaptive_managers[k_val] = DFlashCudaGraphManager(
                self.vllm_config,
                self.device,
                resolved,
                decode_query_lens={1 + k_val: max_r},
            )
        self._select_depth(self._max_spec_steps)

    def capture(self) -> None:
        if len(self._adaptive_managers) <= 1:
            super().capture()
            return
        for k_val in sorted(self._adaptive_managers.keys(), reverse=True):
            self._select_depth(k_val)
            super().capture()
        self._select_depth(self._max_spec_steps)

    def propose(self, *args, num_speculative_tokens: int | None = None, **kwargs):
        if len(self._adaptive_anchors) > 1:
            k_sel = (
                self._max_spec_steps
                if (num_speculative_tokens is None or kwargs.get("dummy_run", False))
                else int(num_speculative_tokens)
            )
            if k_sel > 0 and k_sel in self._adaptive_anchors:
                self._select_depth(k_sel)
                return super().propose(*args, num_speculative_tokens=num_speculative_tokens, **kwargs)
            self._select_depth(self._max_spec_steps)
            return super().propose(*args, num_speculative_tokens=num_speculative_tokens, **kwargs)
        self._active_n_spec = (
            self.num_speculative_steps
            if num_speculative_tokens is None
            else min(int(num_speculative_tokens), self.num_speculative_steps)
        )
        try:
            return super().propose(*args, num_speculative_tokens=num_speculative_tokens, **kwargs)
        finally:
            self._active_n_spec = self.num_speculative_steps

    def _sample_sequential(self, num_reqs: int, head_hidden: torch.Tensor) -> None:
        n_spec = self._active_n_spec
        max_n_spec = self.num_speculative_steps
        if (
            self.draft_logits is None
            and num_reqs == 1
            and getattr(self.model, "fused_markov_argmax", False)
            and getattr(self.model, "markov_topk", 0) == 0
            and os.getenv("INKLING_AMD_TURBO_DRAFT_W4") == "1"
            and not self._require_full_verify_for_graph
            and len(self._adaptive_anchors) <= 1
            and not torch.cuda.is_current_stream_capturing()
            and head_hidden.shape[0] == n_spec + 1
        ):
            if not hasattr(self, "_sample_seq_graph"):
                self._sample_seq_graph = None
                self._sample_seq_warmups = 0
                self._sample_seq_hidden_buf = torch.empty_like(head_hidden)
            if self._sample_seq_graph is not None:
                self._sample_seq_hidden_buf.copy_(head_hidden)
                self._sample_seq_graph.replay()
                return
            self._sample_seq_hidden_buf.copy_(head_hidden)
            self._sample_seq_warmups += 1
            if self._sample_seq_warmups < 2:
                self._sample_sequential_body_c1(self._sample_seq_hidden_buf)
                return
            torch.cuda.synchronize()
            g = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g):
                self._sample_sequential_body_c1(self._sample_seq_hidden_buf)
            self._sample_seq_graph = g
            self._sample_seq_graph.replay()
            return
        if n_spec <= 0:
            return
        full_sample = num_reqs * max_n_spec
        sample_idx = self.sample_indices[:full_sample].view(num_reqs, max_n_spec)[:, :n_spec].reshape(-1)
        hidden = head_hidden[sample_idx]
        base_logits = self.model.compute_base_logits(hidden).view(num_reqs, n_spec, -1)
        context = self.model.model.markov_head.context(hidden, self.f_rnn[:num_reqs])
        idx_map = self.sample_idx_mapping[:full_sample].view(num_reqs, max_n_spec)[:, :n_spec]
        sample_pos = self.sample_pos[:full_sample].view(num_reqs, max_n_spec)[:, :n_spec]
        state = hidden.new_zeros(num_reqs, self.model.config.markov_rank)
        prev = self.input_buffers.input_ids[self._anchor_idx[:num_reqs]]
        if (
            self.draft_logits is None and num_reqs == 1
            and getattr(self.model, "fused_markov_argmax", False)
        ):
            topk_k = getattr(self.model, "markov_topk", 0)
            cands = None
            if topk_k > 0:
                cands = torch.topk(
                    base_logits[0, :, : self.model.valid_vocab], topk_k, dim=-1, sorted=False
                ).indices
            for i in range(n_spec):
                state, prev = self.model.markov_step_greedy(
                    state, prev, context[i], base_logits[:, i],
                    cand_ids=None if cands is None else cands[i],
                )
                self.draft_tokens[:num_reqs, i] = prev
            return
        for i in range(n_spec):
            state, bias = self.model.markov_step(state, prev, context[i])
            if self.draft_logits is None:
                prev = self.model.markov_argmax(base_logits[:, i], bias)
            else:
                logits = self.model.markov_logits(base_logits[:, i], bias)
                prev = self._sample_logits(logits, idx_map[:, i], sample_pos[:, i], i)
            self.draft_tokens[:num_reqs, i] = prev
