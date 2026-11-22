# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""ROCm single-token routing operations for Inkling 2 Mini."""

import torch
import triton
import triton.language as tl
from triton.language.extra.hip import libdevice


@triton.jit
def _router_dot_pairs(X, Weight, pairs):
    x = tl.load(X + pairs)
    weight = tl.load(Weight + pairs)
    dot = tl.inline_asm_elementwise(
        "v_dot2_f32_bf16 $0, $1, $2, 0",
        constraints="=v,v,v",
        args=[weight, x],
        dtype=tl.float32,
        is_pure=True,
        pack=1,
    )
    return tl.sum(dot, axis=0)


@triton.jit
def _router_scores_m1_kernel(X, Weight, Bias, Score, Corrected):
    # One wave per expert uses a warp-local reduction, with no LDS or barrier.
    row = tl.program_id(0)
    x = X.to(tl.pointer_type(tl.uint32))
    weight = Weight.to(tl.pointer_type(tl.uint32)) + row * 1152
    bulk = _router_dot_pairs(x, weight, tl.arange(0, 1024))
    tail = _router_dot_pairs(x, weight, 1024 + tl.arange(0, 128))
    score = tl.div_rn(1.0, 1.0 + libdevice.exp(-(bulk + tail)))
    bias = tl.load(Bias + row)
    tl.store(Score + row, score)
    tl.store(Corrected + row, score + bias)


def router_scores_m1(
    x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor] | None:
    """Return raw and bias-corrected FP32 sigmoid scores for Torch top-k."""
    if (
        not x.is_cuda
        or x.shape != (1, 2304)
        or weight.shape != (128, 2304)
        or bias.shape != (128,)
        or x.dtype != torch.bfloat16
        or weight.dtype != torch.bfloat16
        or bias.dtype != torch.float32
        or not all(t.device == x.device and t.is_contiguous() for t in (x, weight, bias))
        or x.data_ptr() % 4
        or weight.data_ptr() % 4
    ):
        return None
    score = x.new_empty((1, 128), dtype=torch.float32)
    corrected = torch.empty_like(score)
    _router_scores_m1_kernel[(128,)](
        x, weight, bias, score, corrected, num_warps=1, enable_fp_fusion=False,
    )
    return score, corrected


@triton.jit
def _normalize_top8(selected, SCALE: tl.constexpr):
    # ATen's ROCm contiguous width-eight sum shuffles at offsets 1, 2, 4.
    # Adjacent pairs preserve that FP32 denominator and the original slot order.
    even, odd = tl.split(tl.reshape(selected, (4, 2)))
    low_pair, high_pair = tl.split(tl.reshape(even + odd, (2, 2)))
    low_half, high_half = tl.split(low_pair + high_pair)
    denominator = low_half + high_half + 1e-20
    return tl.div_rn(selected, denominator) * SCALE


@triton.jit
def _normalize_top8_routes_kernel(Score, RouteIds, Gamma, SCALE: tl.constexpr):
    slot = tl.arange(0, 8)
    expert = tl.load(RouteIds + slot).to(tl.int32)
    selected = tl.load(Score + expert)
    tl.store(Gamma + slot, _normalize_top8(selected, SCALE))


@triton.jit
def _select_top8_routes_kernel(Score, Corrected, RouteIds, Gamma, SCALE: tl.constexpr):
    expert = tl.arange(0, 128)
    bits = tl.load(Corrected + expert).to(tl.uint32, bitcast=True)
    magnitude = bits & 0x7fffffff
    # The pinned Torch ROCm warp merge is descending and stable, with NaNs
    # first. Treat both signed zeros and all NaNs as tied before adding IDs.
    bits = tl.where(magnitude == 0, 0, bits)
    ordered = tl.where((bits & 0x80000000) != 0, ~bits, bits ^ 0x80000000)
    ordered = tl.where(magnitude > 0x7f800000, 0xffffffff, ordered)
    key = (ordered.to(tl.uint64) << 7) | (127 - expert).to(tl.uint64)
    ranked = tl.topk(key, 8)
    ids = 127 - (ranked & 127).to(tl.int32)
    selected = tl.load(Score + ids)
    slot = tl.arange(0, 8)
    tl.store(RouteIds + slot, ids.to(tl.int64))
    tl.store(Gamma + slot, _normalize_top8(selected, SCALE))


def select_top8_routes_m1(
    score: torch.Tensor, corrected: torch.Tensor, scale: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Fuse the pinned ROCm width-128 stable top-eight order and normalization."""
    if (
        not score.is_cuda
        or corrected.device != score.device
        or any(t.shape != (1, 128) or t.dtype != torch.float32 or not t.is_contiguous()
               for t in (score, corrected))
    ):
        raise ValueError("expected contiguous M1 FP32 raw and corrected scores")

    ids = score.new_empty((1, 8), dtype=torch.int64)
    gamma = score.new_empty((1, 8))
    _select_top8_routes_kernel[(1,)](
        score, corrected, ids, gamma, scale, num_warps=1, enable_fp_fusion=False,
    )
    return ids, gamma


def normalize_top8_routes(
    score: torch.Tensor, route_ids: torch.Tensor, scale: float,
) -> torch.Tensor:
    """Gather uncorrected scores by Torch's top-k IDs and normalize in slot order."""
    if (
        score.shape != (1, 128)
        or score.dtype != torch.float32
        or route_ids.shape != (1, 8)
        or route_ids.dtype != torch.int64
        or not score.is_contiguous()
        or not route_ids.is_contiguous()
        or score.device != route_ids.device
    ):
        raise ValueError("expected contiguous M1 FP32 scores and int64 top-eight IDs")

    gamma = torch.empty_like(route_ids, dtype=score.dtype)
    _normalize_top8_routes_kernel[(1,)](
        score, route_ids, gamma, scale, num_warps=1, enable_fp_fusion=False,
    )
    return gamma
