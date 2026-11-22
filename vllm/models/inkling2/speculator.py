# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
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

    def _sample_sequential(self, num_reqs: int, head_hidden: torch.Tensor) -> None:
        n_spec = self.num_speculative_steps
        if (
            self.draft_logits is None
            and num_reqs == 1
            and getattr(self.model, "fused_markov_argmax", False)
            and getattr(self.model, "markov_topk", 0) == 0
            and os.getenv("INKLING_AMD_TURBO_DRAFT_W4") == "1"
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
        num_sample = num_reqs * n_spec
        hidden = head_hidden[self.sample_indices[:num_sample]]
        base_logits = self.model.compute_base_logits(hidden).view(num_reqs, n_spec, -1)
        context = self.model.model.markov_head.context(hidden, self.f_rnn[:num_reqs])
        idx_map = self.sample_idx_mapping[:num_sample].view(num_reqs, n_spec)
        sample_pos = self.sample_pos[:num_sample].view(num_reqs, n_spec)
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
