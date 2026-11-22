# SPDX-License-Identifier: Apache-2.0
# Modified or added by Ciru for Inkling 2 Mini on AMD gfx1151.
"""Direct top-eight W4A16 expert projections for gfx1151."""

import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice

from .expert_e2m1_g16 import _e2m1_doubled_pair
from .expert_nvfp4 import _positive_e4m3fn


TOP_K = 8


def projection_activation_layout(x, route_ids, width):
    """Return the activation row stride and the number of routes sharing a row."""
    if (route_ids.ndim not in (1, 2) or route_ids.shape[-1] != TOP_K or not route_ids.numel()
            or route_ids.dtype not in (torch.int32, torch.int64) or not route_ids.is_contiguous()):
        raise ValueError("routes must be contiguous int32 or int64 [8] or nonempty [M, 8]")
    if x.ndim and x.dtype == torch.bfloat16 and x.stride(-1) == 1:
        if x.shape == (*route_ids.shape[:-1], width):
            return (0, 1) if x.ndim == 1 else (x.stride(0), TOP_K)
        if x.shape == (*route_ids.shape, width) and x.is_contiguous():
            return width, 1
    raise ValueError("activation must be BF16 with one vector per token or per route")


@triton.jit
def _doubled_e2m1_bf16_pair(packed):
    selector = (packed & 7) | ((packed << 12) & 0x00070000)
    fields = tl.inline_asm_elementwise(
        "v_perm_b32 $0, $1, $2, $3",
        constraints="=v,v,v,v",
        args=[0x85848382, 0x81807E00, selector],
        dtype=tl.uint32, is_pure=True, pack=1,
    )
    # The compact table encodes BF16 bits for magnitudes 0,1,2,3,4,6,8,12.
    # Bit 6 supplies both bits 12 and 13 of each BF16 lane.
    mantissa_exponent = ((fields & 0x007F007F) << 6) | ((fields & 0x00C000C0) << 7)
    signs = ((packed & 8) << 12) | ((packed & 128) << 24)
    return mantissa_exponent | signs


@triton.jit
def _softcap(x, CAP: tl.constexpr):
    return CAP * libdevice.tanh(tl.div_rn(x, CAP))


@triton.jit
def _route_softcap(gate, up, gamma, GATE_CAP: tl.constexpr, UP_CAP: tl.constexpr):
    sigmoid = 1.0 / (1.0 + tl.exp(-gate))
    return ((_softcap(gate, GATE_CAP) * sigmoid) * _softcap(up, UP_CAP)) * gamma


@triton.jit
def _grouped_projection(
    X, Packed, Scales, RouteIds, Outer, RowScales, Table, Out,
    N: tl.constexpr, GROUPS: tl.constexpr, GROUP_BYTES: tl.constexpr,
    BLOCK_ROWS: tl.constexpr, BLOCK_GROUPS: tl.constexpr, EXPERTS: tl.constexpr,
    X_STRIDE: tl.constexpr, ROUTES_PER_INPUT: tl.constexpr,
    FORMAT: tl.constexpr, NORMAL_SCALES: tl.constexpr, BF16_DOT2: tl.constexpr = False,
    Gamma=None, FUSED_ACTIVATION: tl.constexpr = False,
    GATE_CAP: tl.constexpr = 0.0, UP_CAP: tl.constexpr = 0.0,
    STREAM_WEIGHTS: tl.constexpr = False,
):
    slot = tl.program_id(1)
    expert = tl.load(RouteIds + slot)
    expert_valid = (expert >= 0) & (expert < EXPERTS)
    rows = tl.program_id(0) * BLOCK_ROWS + tl.arange(0, BLOCK_ROWS)
    byte = tl.arange(0, BLOCK_GROUPS * GROUP_BYTES)
    row_mask = rows < N
    byte_mask = byte < GROUPS * GROUP_BYTES
    weight_mask = row_mask & expert_valid
    bank_rows = expert.to(tl.int64) * N + rows

    x = X + (slot // ROUTES_PER_INPUT) * X_STRIDE
    if BF16_DOT2:
        pair_x = tl.load(x.to(tl.pointer_type(tl.uint32)) + byte, byte_mask, other=0)
    else:
        even = tl.load(x + 2 * byte, byte_mask, other=0).to(tl.float32)
        odd = tl.load(x + 2 * byte + 1, byte_mask, other=0).to(tl.float32)
    codes = tl.load(
        Packed + bank_rows[:, None] * (GROUPS * GROUP_BYTES) + byte[None, :],
        weight_mask[:, None] & byte_mask[None, :], other=0,
        cache_modifier=".cg" if STREAM_WEIGHTS else "",
    ).to(tl.int32)
    if BF16_DOT2:
        pair_w = _doubled_e2m1_bf16_pair(codes.to(tl.uint32))
        pair_dot = tl.inline_asm_elementwise(
            "v_dot2_f32_bf16 $0, $1, $2, 0",
            constraints="=v,v,v",
            args=[pair_w, tl.broadcast_to(pair_x[None, :], (BLOCK_ROWS, BLOCK_GROUPS * GROUP_BYTES))],
            dtype=tl.float32, is_pure=True, pack=1,
        )
    else:
        if FORMAT == "source-e2m1":
            low, high = _e2m1_doubled_pair(codes.to(tl.uint32))
        elif FORMAT == "flash-c1":
            low = (codes & 15).to(tl.float32)
            high = (codes >> 4).to(tl.float32)
        else:
            low = (((codes & 15) ^ 8) - 8).to(tl.float32)
            high = (((codes >> 4) ^ 8) - 8).to(tl.float32)
        pair_dot = low * even[None, :] + high * odd[None, :]
    group_dot = tl.sum(tl.reshape(pair_dot, (BLOCK_ROWS, BLOCK_GROUPS, GROUP_BYTES)), 2)

    groups = tl.arange(0, BLOCK_GROUPS)
    scale_mask = weight_mask[:, None] & (groups[None, :] < GROUPS)
    metadata = tl.load(Scales + bank_rows[:, None] * GROUPS + groups[None, :], scale_mask, other=0)
    if FORMAT == "source-e2m1":
        scale_bytes = metadata.to(tl.int32)
        if NORMAL_SCALES:
            # Finite positive normals map to BF16 by moving the mantissa and rebiasing.
            bits = ((scale_bytes << 4) + (120 << 7)).to(tl.uint16)
            scale = bits.to(tl.bfloat16, bitcast=True).to(tl.float32)
        else:
            scale = _positive_e4m3fn(scale_bytes)
        factor = 0.5 * tl.load(Outer + expert, expert_valid, other=0)
        result = tl.sum(group_dot * scale, 1) * factor
    elif FORMAT == "flash-c1":
        tags = metadata.to(tl.int32)
        entries = Table + expert * 512 + 2 * tags
        scale = tl.load(entries, scale_mask, other=0).to(tl.float32)
        offset = tl.load(entries + 1, scale_mask, other=0).to(tl.float32)
        x_sum = tl.sum(tl.reshape(even + odd, (BLOCK_GROUPS, GROUP_BYTES)), 1)
        row_scale = tl.load(RowScales + bank_rows, weight_mask, other=0).to(tl.float32)
        result = tl.sum(group_dot * scale + x_sum[None, :] * offset, 1) * row_scale
    else:
        result = tl.sum(group_dot * metadata.to(tl.float32), 1)

    # Keep bounds checking on the device so valid routes require no synchronization.
    result = tl.where(expert_valid, result, float("nan"))
    projected = result.to(tl.bfloat16, fp_downcast_rounding="rtne")
    if FUSED_ACTIVATION:
        # A two-row W13 block owns one interleaved gate/up pair.
        gate, up = tl.split(projected)
        gamma = tl.load(Gamma + slot)
        activated = _route_softcap(gate.to(tl.float32), up.to(tl.float32),
                                   gamma, GATE_CAP, UP_CAP)
        col = tl.program_id(0)
        tl.store(Out + slot * (N // 2) + col,
                 activated.to(tl.bfloat16, fp_downcast_rounding="rtne"))
    else:
        tl.store(Out + slot * N + rows, projected, row_mask)


@triton.jit
def _source_scaled_groups(
    X, Packed, Scales, weight_mask, bank_rows, N_GROUPS: tl.constexpr,
    BLOCK_ROWS: tl.constexpr, OFFSET: tl.constexpr, COUNT: tl.constexpr,
):
    groups = OFFSET + tl.arange(0, COUNT)
    byte = 8 * OFFSET + tl.arange(0, 8 * COUNT)
    even = tl.load(X + 2 * byte).to(tl.float32)
    odd = tl.load(X + 2 * byte + 1).to(tl.float32)
    codes = tl.load(
        Packed + bank_rows[:, None] * (8 * N_GROUPS) + byte[None, :],
        weight_mask[:, None], other=0,
    ).to(tl.uint32)
    low, high = _e2m1_doubled_pair(codes)
    pair_dot = low * even[None, :] + high * odd[None, :]
    group_dot = tl.sum(tl.reshape(pair_dot, (BLOCK_ROWS, COUNT, 8)), 2)
    scale_bytes = tl.load(
        Scales + bank_rows[:, None] * N_GROUPS + groups[None, :],
        weight_mask[:, None], other=0,
    ).to(tl.int32)
    bits = ((scale_bytes << 4) + (120 << 7)).to(tl.uint16)
    return group_dot * bits.to(tl.bfloat16, bitcast=True).to(tl.float32)


@triton.jit
def _source_short_k_projection(
    X, Packed, Scales, RouteIds, Outer, Out,
    N: tl.constexpr, GROUPS: tl.constexpr, BLOCK_ROWS: tl.constexpr, EXPERTS: tl.constexpr,
    X_STRIDE: tl.constexpr,
):
    slot = tl.program_id(1)
    expert = tl.load(RouteIds + slot)
    expert_valid = (expert >= 0) & (expert < EXPERTS)
    rows = tl.program_id(0) * BLOCK_ROWS + tl.arange(0, BLOCK_ROWS)
    row_mask = rows < N
    weight_mask = row_mask & expert_valid
    bank_rows = expert.to(tl.int64) * N + rows
    x = X + slot * X_STRIDE

    # Both Inkling inputs have exactly eight times as many full groups as
    # tail groups. Accumulate them within one program without padded K work.
    TAIL: tl.constexpr = GROUPS // 9
    MAIN: tl.constexpr = 8 * TAIL
    main = _source_scaled_groups(x, Packed, Scales, weight_mask, bank_rows, GROUPS, BLOCK_ROWS, 0, MAIN)
    tail = _source_scaled_groups(x, Packed, Scales, weight_mask, bank_rows, GROUPS, BLOCK_ROWS, MAIN, TAIL)
    # Fold the nonzero upper-half terms into the leading group slots before
    # the final reduction. Padding would only add zeros to the other slots.
    main = tl.reshape(main, (BLOCK_ROWS, 8, TAIL))
    merged = main + tl.where(tl.arange(0, 8)[None, :, None] == 0, tail[:, None, :], 0.0)
    dot = tl.sum(tl.reshape(merged, (BLOCK_ROWS, MAIN)), 1)
    factor = 0.5 * tl.load(Outer + expert, expert_valid, other=0)
    result = tl.where(expert_valid, dot * factor, float("nan"))
    tl.store(Out + slot * N + rows, result.to(tl.bfloat16, fp_downcast_rounding="rtne"), row_mask)


def grouped_projection(
    x, packed, scales, route_ids, *, format, outer_scales=None, flash_aux=None,
    normal_scales=False, tile_rows=4, short_k=False, bf16_dot2=False, fused_activation=None,
    stream_weights=False,
):
    """Project BF16 inputs into BF16 route slots.

    With routes [8], accept x[K] or x[8, K] and return [8, N].
    With routes [M, 8], accept x[M, K] or x[M, 8, K] and return [M, 8, N].
    The byte bank is [E, N, K/2], low nibble first. s4-g32 uses BF16 [E, N, K/32] scales and
    two's-complement nibbles. source-e2m1 uses raw positive E4M3FN scales
    [E, N, K/16] and FP32 [E] amax/denominator ratios. Enable normal_scales
    only after checking scale bytes are all in [8, 126] at load time.
    short_k selects an M1 source kernel for normal scales at K=1152 or K=2304.
    bf16_dot2 keeps the baseline reduction geometry and uses a packed BF16 dot.
    stream_weights applies the gfx1151 streaming hint to DOT2 weight bytes.
    Fused M1 source W13 accepts (gamma, gate_cap, up_cap) and returns [8, N/2].
    flash-c1 uses U8 [E, N, K/16] tags and flash_aux=(FP16 [E, N] row
    multipliers, FP16 [E, 256, 2] affine tables).
    """
    if format not in ("s4-g32", "source-e2m1", "flash-c1") or tile_rows not in (2, 4):
        raise ValueError("invalid grouped projection format or output tile")
    if packed.ndim != 3 or packed.dtype != torch.uint8:
        raise ValueError("packed weights must be uint8 [E, N, K/2]")
    experts, rows, packed_width = packed.shape
    width = 2 * packed_width
    group_size = 32 if format == "s4-g32" else 16
    scale_dtype = torch.bfloat16 if format == "s4-g32" else torch.uint8
    groups = width // group_size
    if not experts or not rows or not groups or width % group_size:
        raise ValueError("packed dimensions must be nonempty and aligned to the group size")
    x_stride, routes_per_input = projection_activation_layout(x, route_ids, width)
    if scales.shape != (experts, rows, groups) or scales.dtype != scale_dtype:
        raise ValueError("scales must match the packed format and geometry")
    operands = [x, packed, scales, route_ids]
    row_scales = table = None
    if format == "source-e2m1":
        if outer_scales is None or outer_scales.shape != (experts,) or outer_scales.dtype != torch.float32:
            raise ValueError("source E2M1 requires FP32 outer scales [E]")
        operands.append(outer_scales)
    if format == "flash-c1":
        if flash_aux is None:
            raise ValueError("Flash C1 requires row multipliers and affine tables")
        row_scales, table = flash_aux
        if row_scales.shape != (experts, rows) or table.shape != (experts, 256, 2):
            raise ValueError("Flash C1 expects row multipliers [E, N] and tables [E, 256, 2]")
        if row_scales.dtype != torch.float16 or table.dtype != torch.float16:
            raise ValueError("Flash C1 metadata must be FP16")
        operands.extend((row_scales, table))
    use_m1_source = format == "source-e2m1" and normal_scales and route_ids.ndim == 1 and groups in (72, 144)
    use_dot2 = bf16_dot2 and use_m1_source and x.data_ptr() % 4 == 0
    if stream_weights and not use_dot2:
        raise ValueError("streaming weight loads require M1 source BF16 DOT2")
    gamma, gate_cap, up_cap = None, 0.0, 0.0
    out_rows = rows
    if fused_activation is not None:
        gamma, gate_cap, up_cap = fused_activation
        if not use_dot2 or groups != 144 or tile_rows != 2 or rows % 2:
            raise ValueError("fused activation requires M1 source W13 with a packed BF16 dot and row pairs")
        if gamma.shape != route_ids.shape or gamma.dtype != torch.float32 or gate_cap <= 0 or up_cap <= 0:
            raise ValueError("fused activation requires FP32 route weights and positive softcaps")
        operands.append(gamma)
        out_rows = rows // 2
    if any(not t.is_contiguous() or t.device != x.device for t in operands):
        raise ValueError("all projection operands must be contiguous on one device")

    out = x.new_empty((*route_ids.shape, out_rows))
    grid = (triton.cdiv(rows, tile_rows), route_ids.numel())
    if short_k and not bf16_dot2 and use_m1_source:
        _source_short_k_projection[grid](
            x, packed, scales, route_ids, outer_scales, out,
            N=rows, GROUPS=groups, BLOCK_ROWS=tile_rows, EXPERTS=experts, X_STRIDE=x_stride,
            num_warps=4,
        )
        return out
    _grouped_projection[grid](
        x, packed, scales, route_ids, outer_scales, row_scales, table, out,
        N=rows, GROUPS=groups, GROUP_BYTES=group_size // 2,
        BLOCK_ROWS=tile_rows, BLOCK_GROUPS=triton.next_power_of_2(groups), EXPERTS=experts,
        X_STRIDE=x_stride, ROUTES_PER_INPUT=routes_per_input,
        FORMAT=format, NORMAL_SCALES=normal_scales, BF16_DOT2=use_dot2,
        Gamma=gamma, FUSED_ACTIVATION=fused_activation is not None,
        GATE_CAP=gate_cap, UP_CAP=up_cap,
        STREAM_WEIGHTS=stream_weights,
        num_warps=4,
    )
    return out


@triton.jit
def _inkling_softcap(W13, Gamma, Out, N: tl.constexpr, BLOCK: tl.constexpr,
                     GATE_CAP: tl.constexpr, UP_CAP: tl.constexpr):
    slot = tl.program_id(1)
    cols = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    pair = W13 + 2 * (slot * N + cols)
    gate = tl.load(pair, cols < N, other=0).to(tl.float32)
    up = tl.load(pair + 1, cols < N, other=0).to(tl.float32)
    gamma = tl.load(Gamma + slot)
    activated = _route_softcap(gate, up, gamma, GATE_CAP, UP_CAP)
    tl.store(Out + slot * N + cols, activated.to(tl.bfloat16, fp_downcast_rounding="rtne"), cols < N)


def inkling_softcap(w13, route_gamma, *, gate_cap=4.0, up_cap=25.0):
    """Apply softcapped SwiGLU to interleaved BF16 rows, weighting before BF16 cast."""
    if (w13.ndim not in (2, 3) or w13.shape[-2] != TOP_K
            or not w13.numel() or w13.shape[-1] % 2):
        raise ValueError("gate/up must have shape [8, 2*N] or [M, 8, 2*N] with interleaved pairs")
    if route_gamma.shape != w13.shape[:-1] or route_gamma.dtype != torch.float32:
        raise ValueError("FP32 route gamma must match the gate/up route slots")
    if w13.dtype != torch.bfloat16 or not w13.is_contiguous() or not route_gamma.is_contiguous():
        raise ValueError("the epilogue requires contiguous BF16 gate/up and contiguous gamma")
    if w13.device != route_gamma.device or gate_cap <= 0 or up_cap <= 0:
        raise ValueError("the epilogue requires one device and positive softcaps")

    width = w13.shape[-1] // 2
    out = w13.new_empty((*route_gamma.shape, width))
    block = 128
    _inkling_softcap[(triton.cdiv(width, block), route_gamma.numel())](
        w13, route_gamma, out, width, block, gate_cap, up_cap, num_warps=4,
    )
    return out


@triton.jit
def _inkling_reduce(Down, Out, N: tl.constexpr, BLOCK: tl.constexpr, ROUTES: tl.constexpr):
    token = tl.program_id(1)
    cols = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    acc = tl.full((BLOCK,), 0.0, tl.float32)
    for slot in tl.static_range(ROUTES):
        acc += tl.load(Down + (token * ROUTES + slot) * N + cols, cols < N, other=0).to(tl.float32)
    tl.store(Out + token * N + cols, acc.to(tl.bfloat16, fp_downcast_rounding="rtne"), cols < N)


def inkling_reduce(w2):
    """Sum already weighted BF16 expert rows in route order and round once to BF16."""
    if w2.ndim not in (2, 3) or w2.shape[-2] != TOP_K or not w2.numel():
        raise ValueError("down projections must have shape [8, N] or [M, 8, N]")
    if w2.dtype != torch.bfloat16 or not w2.is_contiguous():
        raise ValueError("down projections must be contiguous BF16")

    width = w2.shape[-1]
    out = w2.new_empty((*w2.shape[:-2], width))
    block = 128
    _inkling_reduce[(triton.cdiv(width, block), out.numel() // width)](
        w2, out, width, block, TOP_K, num_warps=4,
    )
    return out
