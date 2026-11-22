# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Modified or added by Ciru for Inkling 2 Mini on AMD gfx1151.
"""Direct packed MXFP8 expert projections for gfx1151 decode."""

import torch
import triton
import triton.language as tl

from .ops.expert_nvfp4 import _positive_e4m3fn
from .ops.grouped_decode import projection_activation_layout


@triton.jit
def _projection(X, Weight, Scales, Ids, Out,
                N: tl.constexpr, K: tl.constexpr, X_STRIDE: tl.constexpr, ROUTES_PER_INPUT: tl.constexpr,
                BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr):
    slot = tl.program_id(1)
    expert = tl.load(Ids + slot).to(tl.int64)
    rows = tl.program_id(0) * BLOCK_N + tl.arange(0, BLOCK_N)
    cols = tl.arange(0, BLOCK_K)
    bank_rows = expert * N + rows
    x = tl.load(X + (slot // ROUTES_PER_INPUT) * X_STRIDE + cols, cols < K, other=0).to(tl.float32)
    mask = (rows[:, None] < N) & (cols[None, :] < K)
    raw = tl.load(
        Weight + bank_rows[:, None] * K + cols[None, :], mask, other=0,
    ).to(tl.int32)
    magnitude = _positive_e4m3fn(raw & 127)
    weight = tl.where((raw & 128) != 0, -magnitude, magnitude)
    scale = tl.load(
        Scales + bank_rows[:, None] * (K // 32) + cols[None, :] // 32,
        mask, other=0,
    )
    # The publisher's UE8M0 bytes in these layers are finite FP32 exponents.
    scale = (scale.to(tl.uint32) << 23).to(tl.float32, bitcast=True)
    weight = (weight * scale).to(tl.bfloat16, fp_downcast_rounding="rtne")
    result = tl.sum(weight.to(tl.float32) * x[None, :], axis=1)
    tl.store(
        Out + slot * N + rows,
        result.to(tl.bfloat16, fp_downcast_rounding="rtne"),
        rows < N,
    )


def mxfp8_decode(x, weight, scales, ids):
    _, rows, width = weight.shape
    x_stride, routes_per_input = projection_activation_layout(x, ids, width)
    output = x.new_empty((*ids.shape, rows))
    block_rows = 2
    _projection[(triton.cdiv(rows, block_rows), ids.numel())](
        x, weight.view(torch.uint8), scales, ids, output, rows, width,
        x_stride, routes_per_input,
        block_rows, triton.next_power_of_2(width), num_warps=4,
    )
    return output
