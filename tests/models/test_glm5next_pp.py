# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import inspect
from types import SimpleNamespace

import torch
from torch import nn

from vllm.model_executor.models.interfaces import supports_pp
from vllm.models.glm5next.nvidia import model as glm_model
from vllm.models.glm5next.nvidia import mtp as glm_mtp
from vllm.models.glm5next.nvidia.model import (
    Glm5NextForCausalLM,
    Glm5NextForConditionalGeneration,
    Glm5NextModel,
)
from vllm.models.glm5next.nvidia.mtp import Glm5NextMTP
from vllm.v1.attention.backends.mla import indexer as mla_indexer
from vllm.v1.attention.backends.mla.rocm_aiter_mla_sparse import (
    ROCMAiterMLASparseBackend,
)


def _bare_model(hidden_size: int, layer) -> Glm5NextModel:
    model = Glm5NextModel.__new__(Glm5NextModel)
    nn.Module.__init__(model)
    model.config = SimpleNamespace(
        hidden_size=hidden_size,
        mhc=True,
        mhc_num_residual_streams=4,
    )
    model.is_sequence_parallel = False
    model._active_layers = [layer]
    model.norm = nn.Identity()
    return model


class _SenderLayer:
    def __init__(self, materialized: torch.Tensor):
        self.materialized = materialized
        self.hc_post_calls = 0

    def __call__(self, positions, hidden_states, residual, post, comb):
        tokens, hidden_size = hidden_states.shape
        return (
            hidden_states,
            torch.zeros(tokens, 4, hidden_size),
            torch.zeros(tokens, 4),
            torch.zeros(tokens, 4, 4),
        )

    def hc_post(self, hidden_states, residual, post, comb):
        self.hc_post_calls += 1
        return self.materialized


class _ReceiverLayer:
    def __call__(self, positions, hidden_states, residual, post, comb):
        assert hidden_states.ndim == 3
        assert residual is None and post is None and comb is None
        return hidden_states.mean(dim=1), None, None, None


def test_glm5next_pp_classes_are_discoverable():
    assert supports_pp(Glm5NextForCausalLM)
    assert supports_pp(Glm5NextForConditionalGeneration)
    assert supports_pp(Glm5NextMTP)


def test_rocm_sparse_backends_pin_exact_64_token_pages(monkeypatch):
    monkeypatch.setattr(mla_indexer.current_platform, "is_rocm", lambda: True)
    assert (
        mla_indexer.DeepseekV32IndexerBackend.get_supported_kernel_block_sizes()
        == [1, 64]
    )
    assert ROCMAiterMLASparseBackend.get_supported_kernel_block_sizes() == [1, 64]


def test_glm5next_vision_tower_is_owned_only_by_first_pp_rank():
    source = inspect.getsource(Glm5NextForConditionalGeneration.__init__)
    gate = source.index("if get_pp_group().is_first_rank:")
    tower = source.index("self.visual = Glm5NextVisionTransformer(")
    fallback = source.index("self.visual = PPMissingLayer()")
    assert gate < tower < fallback


def test_glm5next_pp_factory_uses_native_four_stream_shape():
    model = _bare_model(4096, _ReceiverLayer())
    tokens = 3
    intermediate = model.make_empty_intermediate_tensors(
        tokens, torch.bfloat16, torch.device("cpu")
    )
    hidden_states = intermediate["hidden_states"]
    assert hidden_states.shape == (tokens, 4, 4096)
    assert hidden_states.numel() * hidden_states.element_size() == tokens * 32768


def test_glm5next_pp_materializes_one_tensor_at_stage_boundary(monkeypatch):
    tokens, hidden_size = 2, 8
    expected = torch.arange(tokens * 4 * hidden_size, dtype=torch.float32).reshape(
        tokens, 4, hidden_size
    )
    sender_layer = _SenderLayer(expected)
    sender = _bare_model(hidden_size, sender_layer)
    monkeypatch.setattr(
        glm_model,
        "get_pp_group",
        lambda: SimpleNamespace(is_first_rank=True, is_last_rank=False),
    )

    sent = sender(
        input_ids=None,
        positions=torch.arange(tokens),
        intermediate_tensors=None,
        inputs_embeds=torch.zeros(tokens, hidden_size),
    )
    assert set(sent.tensors) == {"hidden_states"}
    assert torch.equal(sent["hidden_states"], expected)
    assert sender_layer.hc_post_calls == 1

    receiver = _bare_model(hidden_size, _ReceiverLayer())
    monkeypatch.setattr(
        glm_model,
        "get_pp_group",
        lambda: SimpleNamespace(is_first_rank=False, is_last_rank=True),
    )
    received = receiver(
        input_ids=None,
        positions=torch.arange(tokens),
        intermediate_tensors=sent,
    )
    assert received.shape == (tokens, hidden_size)


class _EmbeddingOnlyMTP(Glm5NextMTP):
    def named_parameters(self, *args, **kwargs):
        return iter((("model.embed_tokens.weight", self.embedding),))


def test_candidate_a_top_level_embedding_loads_into_last_rank_mtp(monkeypatch):
    model = _EmbeddingOnlyMTP.__new__(_EmbeddingOnlyMTP)
    nn.Module.__init__(model)
    model.embedding = nn.Parameter(torch.zeros(2, 3), requires_grad=False)
    model.embedding.weight_loader = lambda param, value: param.data.copy_(value)
    model.config = SimpleNamespace(
        n_routed_experts=1,
        num_nextn_predict_layers=0,
        num_hidden_layers=45,
        is_linear_attn=False,
        mla_nope=False,
        qk_rope_head_dim=0,
    )
    model.model = SimpleNamespace(mtp_start_layer_idx=45, num_mtp_layers=0)
    monkeypatch.setattr(
        glm_mtp, "fused_moe_make_expert_params_mapping", lambda *args, **kwargs: []
    )

    expected = torch.arange(6, dtype=torch.float32).reshape(2, 3)
    loaded = model.load_weights(
        (("model.language_model.embed_tokens.weight", expected),)
    )
    assert loaded == {"model.embed_tokens.weight"}
    assert torch.equal(model.embedding, expected)
