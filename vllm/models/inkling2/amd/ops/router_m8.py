# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""M8 sigmoid, stable top-eight routing, and expert-block packing on ROCm."""

import torch
import triton
import triton.language as tl
from triton.language.extra.hip import libdevice


@triton.jit
def _max(lhs, rhs):
    return tl.maximum(lhs, rhs)


@triton.jit
def _min(lhs, rhs):
    return tl.minimum(lhs, rhs)


@triton.jit
def _pack_routes_m8(ids, Sorted, Experts, Padded):
    # Each expert gets one M16 block: there is at most one route per token.
    routes = tl.arange(0, 64)
    by_expert = tl.sort((tl.reshape(ids, (64,)) << 6) | routes, descending=False)
    expert = by_expert >> 6
    previous = tl.gather(expert, tl.maximum(routes - 1, 0), 0)
    new_group = (routes == 0) | (expert != previous)
    block = tl.cumsum(new_group.to(tl.int32)) - 1
    start = tl.associative_scan(tl.where(new_group, routes, 0), 0, _max)
    tl.store(Sorted + block * 16 + routes - start, by_expert & 63)
    tl.store(Experts + block, expert, new_group)

    # Assign each block's padding to its first live route. The writes are
    # disjoint from all live routes and require no ordering between waves.
    marks = tl.where(new_group, routes, 64)
    next_mark = tl.gather(marks, tl.minimum(routes + 1, 63), 0)
    next_mark = tl.where(routes < 63, next_mark, 64)
    end = tl.associative_scan(next_mark, 0, _min, reverse=True)
    slot = tl.arange(0, 16)
    padding = new_group[:, None] & (slot[None, :] >= (end - routes)[:, None])
    tl.store(Sorted + block[:, None] * 16 + slot[None, :], 64, padding)
    tl.store(Padded, tl.sum(new_group.to(tl.int32)) * 16)


@triton.jit
def _route_m8(Score, Bias, Ids, Gamma, Sorted, Experts, Padded,
              SCALE: tl.constexpr, ALIGN: tl.constexpr, SIGMOID: tl.constexpr):
    row = tl.arange(0, 8)
    expert = tl.arange(0, 128)
    src = tl.load(Score + row[:, None] * 128 + expert[None, :])
    if SIGMOID:
        src = tl.div_rn(1.0, 1.0 + libdevice.exp(-src))
    bias = tl.load(Bias + expert)
    bits = (src + bias[None, :]).to(tl.uint32, bitcast=True)
    magnitude = bits & 0x7fffffff
    # Match the ROCm 128-column stable descending warp merge used by Torch.
    bits = tl.where(magnitude == 0, 0, bits)
    ordered = tl.where((bits & 0x80000000) != 0, ~bits, bits ^ 0x80000000)
    ordered = tl.where(magnitude > 0x7f800000, 0xffffffff, ordered)
    key = (ordered.to(tl.uint64) << 7) | (127 - expert[None, :]).to(tl.uint64)
    ranked = tl.topk(key, 8)
    ids = 127 - (ranked & 127).to(tl.int32)
    selected = tl.gather(src, ids, 1)

    # Preserve the width-eight normalization's FP32 adjacent-pair sum.
    even, odd = tl.split(tl.reshape(selected, (8, 4, 2)))
    low_pair, high_pair = tl.split(tl.reshape(even + odd, (8, 2, 2)))
    low_half, high_half = tl.split(low_pair + high_pair)
    denominator = low_half + high_half + 1e-20
    gamma = tl.div_rn(selected, denominator[:, None]) * SCALE
    slot = row[:, None] * 8 + tl.arange(0, 8)[None, :]
    tl.store(Ids + slot, ids.to(tl.int64))
    tl.store(Gamma + slot, gamma)
    if ALIGN:
        _pack_routes_m8(ids, Sorted, Experts, Padded)


def route_m8(logits, bias, scale, mode):
    """Use the original FP32 GEMM logits and optionally emit a fused M16 plan."""
    if (
        mode not in ("tail", "align", "select")
        or logits.shape != (8, 128)
        or bias.shape != (128,)
        or logits.dtype != torch.float32
        or bias.dtype != torch.float32
        or not logits.is_cuda
        or bias.device != logits.device
        or not logits.is_contiguous()
        or not bias.is_contiguous()
    ):
        raise ValueError("Expected contiguous ROCm FP32 M8-128 logits and 128 biases")

    align = mode != "tail"
    sigmoid = mode != "select"
    ids = torch.empty((8, 8), dtype=torch.int64, device=logits.device)
    gamma = torch.empty((8, 8), dtype=torch.float32, device=logits.device)
    plan = None
    if align:
        sorted_ids = torch.empty((1024,), dtype=torch.int32, device=logits.device)
        experts = torch.empty((64,), dtype=torch.int32, device=logits.device)
        padded = torch.empty((1,), dtype=torch.int32, device=logits.device)
        plan = (sorted_ids, experts, padded)
    _route_m8[(1,)](
        logits if sigmoid else logits.sigmoid(), bias, ids, gamma,
        *(plan if align else (None, None, None)),
        SCALE=scale, ALIGN=align, SIGMOID=sigmoid, num_warps=2, enable_fp_fusion=False,
    )
    return ids, gamma, plan
