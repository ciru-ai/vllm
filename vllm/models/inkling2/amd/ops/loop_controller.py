# SPDX-License-Identifier: Apache-2.0
"""Single-token Inkling 2 Mini loop-controller projections for gfx1151."""

import torch
import triton
import triton.language as tl


_PAIR_WIDTH = 4608
_RANK = 64


@triton.jit
def _controller_down_kernel(Pair, Weight, Hidden):
    row = tl.program_id(0)
    stream = tl.program_id(1)
    pair = Pair + stream * 4608
    weight = Weight + (stream * 64 + row) * 4608

    cols = tl.arange(0, 4096)
    x = tl.load(pair + cols).to(tl.float32)
    w = tl.load(weight + cols).to(tl.float32)
    bulk = tl.sum(x * w, axis=0)

    tail_cols = 4096 + tl.arange(0, 512)
    x_tail = tl.load(pair + tail_cols).to(tl.float32)
    w_tail = tl.load(weight + tail_cols).to(tl.float32)
    tail = tl.sum(x_tail * w_tail, axis=0)
    tl.store(Hidden + stream * 64 + row, (bulk + tail).to(tl.bfloat16))


@triton.jit
def _controller_gain_forget_kernel(Hidden, Weight, Out, BLOCK_ROWS: tl.constexpr):
    rows = tl.program_id(0) * BLOCK_ROWS + tl.arange(0, BLOCK_ROWS)
    stream = tl.program_id(1)
    cols = tl.arange(0, 64)
    x = tl.load(Hidden + stream * 64 + cols).to(tl.float32)
    w = tl.load(Weight + (stream * 4608 + rows[:, None]) * 64 + cols[None, :])
    values = tl.sum(w.to(tl.float32) * x[None, :], axis=1)
    tl.store(Out + stream * 4608 + rows, values)


def _project_down(pair: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    streams = pair.shape[0]
    hidden = pair.new_empty((streams, 1, _RANK))
    _controller_down_kernel[(_RANK, streams)](pair, weight, hidden, num_warps=4)
    return hidden


def _project_gain_forget(hidden: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    streams = hidden.shape[0]
    out = hidden.new_empty((streams, 1, _PAIR_WIDTH), dtype=torch.float32)
    block_rows = 16
    _controller_gain_forget_kernel[(_PAIR_WIDTH // block_rows, streams)](
        hidden, weight, out, block_rows, num_warps=4,
    )
    return out


def loop_controller_decode(
    pair: torch.Tensor,
    down_weight: torch.Tensor,
    gain_forget_weight: torch.Tensor,
    *,
    packed_down: bool = True,
) -> torch.Tensor | None:
    """Preserve BF16 down rounding and widen exact gain/forget weights on chip."""
    if not pair.is_cuda or pair.ndim != 3:
        return None
    streams = pair.shape[0]
    if (
        streams not in (1, 3)
        or pair.shape != (streams, 1, _PAIR_WIDTH)
        or down_weight.shape != (streams, _RANK, _PAIR_WIDTH)
        or gain_forget_weight.shape != (streams, _PAIR_WIDTH, _RANK)
        or any(
            t.device != pair.device or t.dtype != torch.bfloat16 or not t.is_contiguous()
            for t in (pair, down_weight, gain_forget_weight)
        )
    ):
        return None
    hidden = (
        _project_down(pair, down_weight)
        if packed_down else torch.bmm(pair, down_weight.transpose(1, 2))
    )
    return _project_gain_forget(hidden, gain_forget_weight)
