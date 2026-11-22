"""Wave-local source NVFP4 M1 projections for gfx1151."""

import torch
from triton.experimental import gluon
from triton.experimental.gluon import language as gl

from .grouped_decode import projection_activation_layout


@gluon.jit
def _weight_pair(packed):
    selector = (packed & 7) | ((packed << 12) & 0x00070000)
    fields = gl.inline_asm_elementwise(
        "v_perm_b32 $0, $1, $2, $3", constraints="=v,v,v,v",
        args=[0x85848382, 0x81807E00, selector], dtype=gl.uint32, is_pure=True, pack=1,
    )
    return (((fields & 0x007F007F) << 6) | ((fields & 0x00C000C0) << 7)
            | ((packed & 8) << 12) | ((packed & 128) << 24))


@gluon.jit
def _dot2(p, x):
    return gl.inline_asm_elementwise(
        "v_dot2_f32_bf16 $0, $1, $2, 0", constraints="=v,v,v",
        args=[_weight_pair(p), x], dtype=gl.float32, is_pure=True, pack=1,
    )


@gluon.jit
def _load_tile(X, Packed, Scale, first_row, start,
               GROUPS: gl.constexpr, COUNT: gl.constexpr):
    # Each wave owns one row. Two lanes consume one 16-weight group.
    words: gl.constexpr = gl.BlockedLayout([1, 1], [1, 32], [4, 1], [1, 0])
    pairs: gl.constexpr = gl.BlockedLayout([1, 4], [1, 32], [4, 1], [1, 0])
    fragments: gl.constexpr = gl.BlockedLayout([1, 1, 1], [1, 16, 2], [4, 1, 1], [2, 1, 0])
    group_layout: gl.constexpr = gl.SliceLayout(2, fragments)
    lane = gl.arange(0, COUNT * 2, gl.SliceLayout(0, words))
    rows = first_row + gl.arange(0, 4, gl.SliceLayout(1, words))
    weights = gl.load(Packed + rows[:, None] * (GROUPS * 2) + (start * 2 + lane[None, :]))
    x_idx = gl.arange(0, COUNT * 8, gl.SliceLayout(0, pairs))
    activation = gl.load(X + start * 8 + x_idx)
    groups = start + gl.arange(0, COUNT, gl.SliceLayout(0, group_layout))
    scale_rows = first_row + gl.arange(0, 4, gl.SliceLayout(1, group_layout))
    scale = gl.load(Scale + scale_rows[:, None] * GROUPS + groups[None, :])
    return weights, activation, scale


@gluon.jit
def _scaled_tile(weights, activation, scale, COUNT: gl.constexpr):
    words: gl.constexpr = gl.BlockedLayout([1, 1], [1, 32], [4, 1], [1, 0])
    fragments: gl.constexpr = gl.BlockedLayout([1, 1, 1], [1, 16, 2], [4, 1, 1], [2, 1, 0])
    even, odd = gl.split(gl.reshape(activation, [COUNT * 2, 2, 2]))
    x0, x2 = gl.split(even)
    x1, x3 = gl.split(odd)
    x0 = gl.convert_layout(x0, gl.SliceLayout(0, words), assert_trivial=True)
    x1 = gl.convert_layout(x1, gl.SliceLayout(0, words), assert_trivial=True)
    x2 = gl.convert_layout(x2, gl.SliceLayout(0, words), assert_trivial=True)
    x3 = gl.convert_layout(x3, gl.SliceLayout(0, words), assert_trivial=True)
    # Each word holds four E2M1 pairs. Keep the products and group sum in FP32.
    p0 = _dot2(weights, x0[None, :])
    p1 = _dot2(weights >> 8, x1[None, :])
    p2 = _dot2(weights >> 16, x2[None, :])
    p3 = _dot2(weights >> 24, x3[None, :])
    dots = gl.convert_layout(gl.reshape((p0 + p1) + (p2 + p3), [4, COUNT, 2]),
                             fragments, assert_trivial=True)
    group_dot = gl.sum(dots, 2)
    bits = ((scale.to(gl.uint16) << 4) + (120 << 7)).to(gl.uint16)
    return group_dot * bits.to(gl.bfloat16, bitcast=True).to(gl.float32)


@gluon.jit
def _wave_source_projection(
    X, Packed, Scale, RouteIds, Outer, Out,
    N: gl.constexpr, GROUPS: gl.constexpr, EXPERTS: gl.constexpr, X_STRIDE: gl.constexpr,
):
    slot = gl.program_id(1)
    expert = gl.load(RouteIds + slot)
    row_origin = gl.program_id(0) * 4
    fragments: gl.constexpr = gl.BlockedLayout([1, 1, 1], [1, 16, 2], [4, 1, 1], [2, 1, 0])
    group_layout: gl.constexpr = gl.SliceLayout(2, fragments)
    reduced: gl.constexpr = gl.SliceLayout(1, group_layout)
    rows = row_origin + gl.arange(0, 4, reduced)
    out = Out + slot * N + rows
    if expert < 0 or expert >= EXPERTS:
        gl.store(out, gl.full((4,), float("nan"), gl.bfloat16, reduced))
        return
    first_row = expert.to(gl.int64) * N + row_origin
    x = (X + slot * X_STRIDE).to(gl.pointer_type(gl.uint32))
    packed = Packed.to(gl.pointer_type(gl.uint32))

    acc = gl.zeros((4, 16), gl.float32, group_layout)
    weights, activation, scale = _load_tile(x, packed, Scale, first_row, 0, GROUPS, 16)
    # Fetch neighboring tile operands while scoring the previous tile.
    for chunk in range(1, GROUPS // 16):
        next_weights, next_activation, next_scale = _load_tile(
            x, packed, Scale, first_row, chunk * 16, GROUPS, 16,
        )
        acc += _scaled_tile(weights, activation, scale, 16)
        weights, activation, scale = next_weights, next_activation, next_scale
    acc += _scaled_tile(weights, activation, scale, 16)
    result = gl.sum(acc, 1)
    if GROUPS % 16:
        tail_words, tail_x, tail_scale = _load_tile(
            x, packed, Scale, first_row, GROUPS - GROUPS % 16, GROUPS, GROUPS % 16,
        )
        result += gl.sum(_scaled_tile(tail_words, tail_x, tail_scale, GROUPS % 16), 1)
    factor = 0.5 * gl.load(Outer + expert)
    gl.store(out, (result * factor).to(gl.bfloat16, fp_downcast_rounding="rtne"))


def source_wave_projection(
    x: torch.Tensor, packed: torch.Tensor, scales: torch.Tensor, route_ids: torch.Tensor,
    outer_scales: torch.Tensor,
) -> torch.Tensor:
    """Use validated E4M3 normal scales and the original source packet layout.

    Accept one BF16 activation per token or per route. The result is [8, N].
    W13 and W2 must have their source K2304 and K1152 widths, respectively.
    """
    if packed.ndim != 3 or packed.dtype != torch.uint8:
        raise ValueError("expected packed source E2M1 weights [E, N, K/2]")
    experts, rows, packed_width = packed.shape
    width = 2 * packed_width
    if not experts or not rows or rows % 4 or width not in (1152, 2304) or route_ids.ndim != 1:
        raise ValueError("expected M1 routes, K1152/K2304 and a multiple of four output rows")
    stride, _ = projection_activation_layout(x, route_ids, width)
    groups = width // 16
    if scales.shape != (experts, rows, groups) or scales.dtype != torch.uint8:
        raise ValueError("expected validated positive-normal E4M3 scales [E, N, K/16]")
    if outer_scales.shape != (experts,) or outer_scales.dtype != torch.float32:
        raise ValueError("expected FP32 outer scales [E]")
    if any(not t.is_contiguous() or t.device != x.device for t in (x, packed, scales, route_ids, outer_scales)):
        raise ValueError("expected contiguous operands on the same device")
    if x.data_ptr() % 16 or packed.data_ptr() % 4:
        raise ValueError("expected 16-byte aligned activations and 4-byte aligned packed weights")

    out = x.new_empty((route_ids.numel(), rows))
    _wave_source_projection[(rows // 4, route_ids.numel())](
        x, packed, scales, route_ids, outer_scales, out,
        N=rows, GROUPS=groups, EXPERTS=experts, X_STRIDE=stride, num_warps=4,
    )
    return out
