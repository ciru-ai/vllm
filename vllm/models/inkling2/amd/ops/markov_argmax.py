# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Fused rank-256 DSpark fold and greedy selection for gfx1151."""

import torch
from triton.experimental import gluon
from triton.experimental.gluon import language as gl
from triton.experimental.gluon.language.extra import libdevice

from vllm.triton_utils import triton
from vllm.v1.worker.gpu.sample.greedy_argmax import _argmax_finish

from .groupwise_w6_head import _w6_tile


@gluon.jit
def _w8_markov_dot(
    X, Q, S, rows, cols, valid, layout: gl.constexpr,
    GROUP_SIZE: gl.constexpr, ROWS: gl.constexpr,
):
    partial = gl.zeros((ROWS, GROUP_SIZE), gl.float32, layout)
    for group in gl.static_range(2):
        k = group * GROUP_SIZE + cols
        x = gl.load(X + k).to(gl.float32)
        q = gl.load(
            Q + rows[:, None] * (2 * GROUP_SIZE) + k[None, :],
            valid[:, None],
            other=0,
        ).to(gl.float32)
        scale = gl.load(S + rows * 2 + group, valid, other=0).to(gl.float32)
        partial += (q * x[None, :]) * scale[:, None]
    return gl.sum(partial, axis=1)


@gluon.jit
def _first_max(left_score, left_id, right_score, right_id):
    take_left = (left_score > right_score) | (
        (left_score == right_score) & (left_id < right_id)
    )
    return (
        gl.where(take_left, left_score, right_score),
        gl.where(take_left, left_id, right_id),
    )


@gluon.jit
def _markov_argmax_partial(
    X, Codes, Third, Scales, Bias, Base, Values, Indices, softcap,
    VALID: gl.constexpr, ROWS: gl.constexpr, W6: gl.constexpr,
):
    block = gl.program_id(0)
    if W6:
        tile: gl.constexpr = gl.BlockedLayout(
            [1, 1, 16], [1, 32, 1], [4, 1, 1], [2, 1, 0]
        )
        row_lane: gl.constexpr = gl.SliceLayout(2, tile)
        input_layout: gl.constexpr = gl.SliceLayout(0, tile)
        rows = block * ROWS + gl.arange(0, ROWS, gl.SliceLayout(1, row_lane))
        lanes = gl.arange(0, 32, gl.SliceLayout(0, row_lane))
        input_lanes = gl.arange(0, 32, gl.SliceLayout(1, input_layout))
        input_slots = gl.arange(0, 16, gl.SliceLayout(0, input_layout))
        input_cols = input_lanes[:, None] * 16 + input_slots[None, :]
        partial = _w6_tile(
            X, Codes, Third, Scales, rows, lanes, input_cols,
            input_slots[None, :][None, :, :], 0, VALID, 256, 64, ROWS, True,
        )
        folded = gl.sum(partial, axis=1)
    else:
        tile: gl.constexpr = gl.BlockedLayout([1, 4], [1, 32], [4, 1], [1, 0])
        rows = block * ROWS + gl.arange(0, ROWS, gl.SliceLayout(1, tile))
        cols = gl.arange(0, 128, gl.SliceLayout(0, tile))
        folded = _w8_markov_dot(X, Codes, Scales, rows, cols, rows < VALID, tile, 128, ROWS)

    valid = rows < VALID
    bias = gl.load(Bias + rows, valid, other=0).to(gl.float32)
    base = gl.load(Base + rows, valid, other=0).to(gl.float32)
    correction = (folded + bias).to(gl.bfloat16, fp_downcast_rounding="rtne")
    combined = (base + correction.to(gl.float32)).to(
        gl.bfloat16, fp_downcast_rounding="rtne"
    )
    score = (libdevice.tanh(combined / softcap) * softcap).to(
        gl.bfloat16, fp_downcast_rounding="rtne"
    ).to(gl.float32)
    score = gl.where(valid & (score == score), score, float("-inf"))
    value, index = gl.reduce((score, rows), 0, _first_max)
    gl.store(Values + block, value)
    gl.store(Indices + block, index)


def fused_markov_argmax(
    latent: torch.Tensor,
    weight: tuple[torch.Tensor, ...],
    bias: torch.Tensor,
    base: torch.Tensor,
    softcap: float,
    valid: int,
) -> torch.Tensor:
    """Return one token ID after the BF16 fold and greedy softcap epilogue."""
    assert latent.shape == (1, 256) and latent.dtype == torch.bfloat16
    assert base.shape == (1, bias.numel()) and base.dtype == bias.dtype == torch.bfloat16
    assert 0 < valid <= base.shape[1] and base.stride(1) == bias.stride(0) == 1
    w6 = len(weight) == 3
    codes, third, scales = weight if w6 else (weight[0], None, weight[1])
    rows = 16
    splits = triton.cdiv(valid, rows)
    values = torch.empty(splits, dtype=torch.float32, device=base.device)
    indices = torch.empty(splits, dtype=torch.int32, device=base.device)
    out = torch.empty(1, dtype=torch.int64, device=base.device)
    _markov_argmax_partial[(splits,)](
        latent, codes, third, scales, bias, base, values, indices, softcap,
        VALID=valid, ROWS=rows, W6=w6, num_warps=4,
    )
    _argmax_finish[(1,)](
        values, indices, out, splits, triton.next_power_of_2(splits), num_warps=4,
    )
    return out
