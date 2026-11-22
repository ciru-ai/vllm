"""Direct gfx1151 projections with one weight format per expert."""

import torch
import triton
import triton.language as tl

from .expert_e2m1_g16 import _e2m1_doubled_pair
from .expert_nvfp4 import _positive_e4m3fn
from .grouped_decode import projection_activation_layout


@triton.jit
def _project_tile(
    codes, even, odd, RawScales, Outer, RowScales, Table,
    expert, bank_rows, row_mask,
    K_BYTES: tl.constexpr, BLOCK_ROWS: tl.constexpr, BLOCK_BYTES: tl.constexpr,
    KIND: tl.constexpr, NORMAL_SCALES: tl.constexpr,
):
    if KIND == 0:
        low, high = _e2m1_doubled_pair(codes.to(tl.uint32))
    elif KIND == 1:
        low = (((codes & 15) ^ 8) - 8).to(tl.float32)
        high = (((codes >> 4) ^ 8) - 8).to(tl.float32)
    else:
        low = (codes & 15).to(tl.float32)
        high = (codes >> 4).to(tl.float32)

    group_bytes: tl.constexpr = 16 if KIND == 1 else 8
    groups: tl.constexpr = K_BYTES // group_bytes
    padded_groups: tl.constexpr = BLOCK_BYTES // group_bytes
    pairs = low * even[None, :] + high * odd[None, :]
    dots = tl.sum(tl.reshape(pairs, (BLOCK_ROWS, padded_groups, group_bytes)), 2)
    group = tl.arange(0, padded_groups)
    mask = row_mask[:, None] & (group[None, :] < groups)
    base = bank_rows[:, None] * groups + group[None, :]

    if KIND == 0:
        raw = tl.load(RawScales + base, mask, other=0).to(tl.int32)
        if NORMAL_SCALES:
            # Finite positive normals map to BF16 by moving and rebiasing the bits.
            bits = ((raw << 4) + (120 << 7)).to(tl.uint16)
            scale = bits.to(tl.bfloat16, bitcast=True).to(tl.float32)
        else:
            scale = _positive_e4m3fn(raw)
        result = tl.sum(dots * scale, 1) * (0.5 * tl.load(Outer + expert))
    elif KIND == 1:
        scales = RawScales.to(tl.pointer_type(tl.bfloat16))
        scale = tl.load(scales + base, mask, other=0).to(tl.float32)
        result = tl.sum(dots * scale, 1)
    else:
        tags = tl.load(RawScales + base, mask, other=0).to(tl.int32)
        entries = Table + expert * 512 + 2 * tags
        scale = tl.load(entries, mask, other=0).to(tl.float32)
        offset = tl.load(entries + 1, mask, other=0).to(tl.float32)
        x_sum = tl.sum(tl.reshape(even + odd, (padded_groups, group_bytes)), 1)
        row_scale = tl.load(RowScales + bank_rows, row_mask, other=0).to(tl.float32)
        result = tl.sum(dots * scale + x_sum[None, :] * offset, 1) * row_scale
    return result


@triton.jit
def _mixed_grouped_projection(
    X, Packed, RawScales, RouteIds, Kind, Outer, RowScales, Table, Out,
    N: tl.constexpr, K_BYTES: tl.constexpr, BLOCK_ROWS: tl.constexpr, BLOCK_BYTES: tl.constexpr,
    EXPERTS: tl.constexpr, X_STRIDE: tl.constexpr, ROUTES_PER_INPUT: tl.constexpr,
    NORMAL_SCALES: tl.constexpr, HAS_FLASH: tl.constexpr,
):
    slot = tl.program_id(1)
    expert = tl.load(RouteIds + slot).to(tl.int64)
    expert_valid = (expert >= 0) & (expert < EXPERTS)
    kind = tl.load(Kind + expert, expert_valid, other=-1)
    rows = tl.program_id(0) * BLOCK_ROWS + tl.arange(0, BLOCK_ROWS)
    byte = tl.arange(0, BLOCK_BYTES)
    row_mask = rows < N
    byte_mask = byte < K_BYTES
    bank_rows = expert * N + rows

    # Every lane in this workgroup handles the same route slot and expert kind.
    # Invalid IDs/kinds produce a visible NaN without an out-of-bounds read.
    out = tl.full((BLOCK_ROWS,), float("nan"), tl.float32)
    if (kind == 0) | (kind == 1) | ((kind == 2) & HAS_FLASH):
        x = X + (slot // ROUTES_PER_INPUT) * X_STRIDE
        even = tl.load(x + 2 * byte, byte_mask, other=0).to(tl.float32)
        odd = tl.load(x + 2 * byte + 1, byte_mask, other=0).to(tl.float32)
        codes = tl.load(
            Packed + bank_rows[:, None] * K_BYTES + byte[None, :],
            row_mask[:, None] & byte_mask[None, :], other=0,
        ).to(tl.int32)
        if kind == 0:
            out = _project_tile(
                codes, even, odd, RawScales, Outer, RowScales, Table, expert, bank_rows, row_mask,
                K_BYTES, BLOCK_ROWS, BLOCK_BYTES, 0, NORMAL_SCALES,
            )
        elif kind == 1:
            out = _project_tile(
                codes, even, odd, RawScales, Outer, RowScales, Table, expert, bank_rows, row_mask,
                K_BYTES, BLOCK_ROWS, BLOCK_BYTES, 1, NORMAL_SCALES,
            )
        elif HAS_FLASH:
            out = _project_tile(
                codes, even, odd, RawScales, Outer, RowScales, Table, expert, bank_rows, row_mask,
                K_BYTES, BLOCK_ROWS, BLOCK_BYTES, 2, NORMAL_SCALES,
            )
    tl.store(Out + slot * N + rows, out.to(tl.bfloat16, fp_downcast_rounding="rtne"), row_mask)


def mixed_grouped_projection(
    x, packed, raw_scales, route_ids, kind, outer_scales, *,
    flash_aux=None, normal_scales=True, tile_rows=4,
):
    """Project BF16 inputs using the same route and activation shapes as grouped_projection.

    The byte bank is packed[E,N,K/2] (low nibble first), with parallel
    raw_scales[E,N,K/16] U8 storage. kind[E] is int32: 0 selects E2M1 with
    positive E4M3FN scales and FP32 outer_scales[E]; 1 selects signed G32
    with BF16 scales viewed as U8; 2 selects Flash C1 with group tags and
    flash_aux=(FP16 [E,N] row multipliers, FP16 [E,256,2] affine tables).
    Validate E2M1 raw bytes are in [8,126] before using normal_scales=True.
    Returns BF16 [8,N] or [M,8,N] in route order. Invalid routes or kinds produce NaN.
    """
    if tile_rows not in (2, 4) or packed.ndim != 3 or packed.dtype != torch.uint8:
        raise ValueError("expected tile_rows 2 or 4 and packed U8 weights [E,N,K/2]")
    experts, rows, packed_width = packed.shape
    width = 2 * packed_width
    if not experts or not rows or not width or width % 32:
        raise ValueError("nonempty weight dimensions and K divisible by 32 are required")
    x_stride, routes_per_input = projection_activation_layout(x, route_ids, width)
    if raw_scales.shape != (experts, rows, width // 16) or raw_scales.dtype != torch.uint8:
        raise ValueError("raw scales must be U8 [E,N,K/16]")
    if kind.shape != (experts,) or kind.dtype != torch.int32:
        raise ValueError("format kinds must be int32 [E]")
    if outer_scales.shape != (experts,) or outer_scales.dtype != torch.float32:
        raise ValueError("source outer scales must be FP32 [E]")
    operands = [x, packed, raw_scales, route_ids, kind, outer_scales]
    row_scales = table = None
    if flash_aux is not None:
        row_scales, table = flash_aux
        if row_scales.shape != (experts, rows) or table.shape != (experts, 256, 2):
            raise ValueError("Flash C1 expects row multipliers [E,N] and tables [E,256,2]")
        if row_scales.dtype != torch.float16 or table.dtype != torch.float16:
            raise ValueError("Flash C1 metadata must be FP16")
        operands.extend((row_scales, table))
    if any(not t.is_contiguous() or t.device != x.device for t in operands):
        raise ValueError("all projection operands must be contiguous on one device")
    if raw_scales.data_ptr() % 2:
        raise ValueError("raw scales must be aligned for a BF16 view")

    out = x.new_empty((*route_ids.shape, rows))
    _mixed_grouped_projection[(triton.cdiv(rows, tile_rows), route_ids.numel())](
        x, packed, raw_scales, route_ids, kind, outer_scales, row_scales, table, out,
        N=rows, K_BYTES=packed_width, BLOCK_ROWS=tile_rows,
        BLOCK_BYTES=triton.next_power_of_2(packed_width), EXPERTS=experts,
        X_STRIDE=x_stride, ROUTES_PER_INPUT=routes_per_input, NORMAL_SCALES=normal_scales,
        HAS_FLASH=flash_aux is not None, num_warps=4,
    )
    return out
