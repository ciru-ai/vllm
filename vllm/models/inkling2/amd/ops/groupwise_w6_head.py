# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Modified or added by Ciru for Inkling 2 Mini on AMD gfx1151.
"""Standalone gfx1151 signed W6A16 GEMV with K32 or K64 scales."""

import os
from math import isfinite

import torch
from triton.experimental import gluon
from triton.experimental.gluon import language as gl
from triton.experimental.gluon.language.extra import libdevice

from vllm.triton_utils import triton


_WIDTHS = (2304, 3072)
_GROUP_SIZES = (32, 64)
_RECORD_SIZE = 16


@torch.no_grad()
def pack_groupwise_w6_head(
    weight: torch.Tensor, *, group_size: int = 64,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Store 16 signed codes in one 64-bit and one 32-bit word per record."""
    if weight.ndim != 2 or weight.shape[1] not in (*_WIDTHS, 256) or weight.dtype != torch.bfloat16:
        raise ValueError("Expected BF16 weights with width 256, 2304 or 3072")
    if group_size not in _GROUP_SIZES:
        raise ValueError("W6 scale groups must contain 32 or 64 weights")

    n, width = weight.shape
    record_count = width // _RECORD_SIZE
    group_count = width // group_size
    pair = torch.empty((n, record_count), dtype=torch.int64, device=weight.device)
    third = torch.empty((n, record_count), dtype=torch.int32, device=weight.device)
    scales = torch.empty((n, group_count), dtype=torch.float16, device=weight.device)
    min_scale = torch.finfo(torch.float16).tiny * torch.finfo(torch.float16).eps
    chunk_rows = 2048
    shifts = torch.arange(0, 30, 6, device=weight.device)
    for start in range(0, n, chunk_rows):
        stop = min(start + chunk_rows, n)
        values = weight[start:stop].float().reshape(-1, group_count, group_size)
        scale = (values.abs().amax(dim=-1) / 31).clamp_min_(min_scale).half()
        codes = values.div_(scale.float()[:, :, None]).round_().clamp_(-31, 31)
        records = (codes.to(torch.int64) & 63).reshape(-1, record_count, _RECORD_SIZE)
        last = records[:, :, 15]

        # Five six-bit codes per dword, and two bits of the last code in each.
        word0 = (records[:, :, :5] << shifts).sum(dim=2) | ((last & 3) << 30)
        word1 = (records[:, :, 5:10] << shifts).sum(dim=2) | (((last >> 2) & 3) << 30)
        word2 = (records[:, :, 10:15] << shifts).sum(dim=2) | ((last >> 4) << 30)
        pair[start:stop].copy_(word0 | (word1 << 32))
        third[start:stop].copy_(word2)
        scales[start:stop].copy_(scale)
    return pair, third, scales


@gluon.jit
def _w6_dot2_record(codes, x, ROWS: gl.constexpr, ROW_LANE: gl.constexpr):
    # Codes fit exactly in BF16. The dot replaces the first adjacent-pair
    # reduction while keeping the remaining FP32 tree and the scale position.
    pair_shape: gl.constexpr = (ROWS, 32, 8, 2)
    x, _ = gl.broadcast(x[None, :, :], codes)
    c0, c1 = gl.split(gl.reshape(codes.to(gl.bfloat16), pair_shape))
    x0, x1 = gl.split(gl.reshape(x, pair_shape))
    c_pair = c0.to(gl.uint16, bitcast=True).to(gl.uint32) | (
        c1.to(gl.uint16, bitcast=True).to(gl.uint32) << 16
    )
    x_pair = x0.to(gl.uint16, bitcast=True).to(gl.uint32) | (
        x1.to(gl.uint16, bitcast=True).to(gl.uint32) << 16
    )
    sums = gl.inline_asm_elementwise(
        "v_dot2_f32_bf16 $0, $1, $2, 0",
        constraints="=v,v,v", args=[c_pair, x_pair],
        dtype=gl.float32, is_pure=True, pack=1,
    )
    return gl.convert_layout(gl.sum(sums, axis=2), ROW_LANE)


@gluon.jit
def _w6_tile(
    X, Pair, Third, Scales, rows, lanes, input_cols, slots, chunk,
    N: gl.constexpr, K: gl.constexpr, GROUP_SIZE: gl.constexpr, ROWS: gl.constexpr,
    TAIL: gl.constexpr, USE_DOT2: gl.constexpr = False,
):
    records: gl.constexpr = K // 16
    groups: gl.constexpr = K // GROUP_SIZE
    records_per_group: gl.constexpr = GROUP_SIZE // 16
    record = chunk * 32 + lanes
    valid = rows[:, None] < N if N % ROWS else True
    if TAIL:
        valid = valid & (record[None, :] < records)
    pair = gl.load(Pair + rows[:, None] * records + record[None, :], valid, 0)
    word0 = pair.to(gl.uint32)
    word1 = (pair.to(gl.uint64) >> 32).to(gl.uint32)
    word2 = gl.load(Third + rows[:, None] * records + record[None, :], valid, 0)
    word2 = word2.to(gl.uint32)
    scale = gl.load(
        Scales + rows[:, None] * groups + record[None, :] // records_per_group, valid, 0,
    ).to(gl.float32)
    if TAIL:
        x = gl.load(X + chunk * 512 + input_cols, chunk * 512 + input_cols < K, 0)
    else:
        x = gl.load(X + chunk * 512 + input_cols)

    last = (word0 >> 30) | ((word1 >> 28) & 12) | ((word2 >> 26) & 48)
    word = gl.where(slots < 5, word0[:, :, None], gl.where(
        slots < 10, word1[:, :, None], word2[:, :, None],
    ))
    word = gl.where(slots == 15, last[:, :, None], word)
    bits = gl.where(slots == 15, 0, (slots % 5) * 6)
    codes = (word << (26 - bits)).to(gl.int32) >> 26
    if USE_DOT2:
        return _w6_dot2_record(codes, x, ROWS, scale.type.layout) * scale
    return gl.sum(codes.to(gl.float32) * x[None, :, :].to(gl.float32), axis=2) * scale


@gluon.jit
def _groupwise_w6_rows(
    X, Pair, Third, Scales, Y, row_start, N: gl.constexpr, K: gl.constexpr,
    GROUP_SIZE: gl.constexpr, ROWS: gl.constexpr, SOFT_CAP: gl.constexpr,
):
    # One output row per wave and one 16-code record per lane.
    tile: gl.constexpr = gl.BlockedLayout([1, 1, 16], [1, 32, 1], [ROWS, 1, 1], [2, 1, 0])
    row_lane: gl.constexpr = gl.SliceLayout(2, tile)
    input_layout: gl.constexpr = gl.SliceLayout(0, tile)
    rows = row_start + gl.arange(0, ROWS, gl.SliceLayout(1, row_lane))
    lanes = gl.arange(0, 32, gl.SliceLayout(0, row_lane))
    input_lanes = gl.arange(0, 32, gl.SliceLayout(1, input_layout))
    input_slots = gl.arange(0, 16, gl.SliceLayout(0, input_layout))
    input_cols = input_lanes[:, None] * 16 + input_slots[None, :]
    slots = input_slots[None, :][None, :, :]
    partial = gl.zeros((ROWS, 32), gl.float32, row_lane)
    for chunk in range(K // 512):
        partial += _w6_tile(
            X, Pair, Third, Scales, rows, lanes, input_cols, slots, chunk, N, K, GROUP_SIZE, ROWS, False,
        )
    if K % 512:
        partial += _w6_tile(
            X, Pair, Third, Scales, rows, lanes, input_cols, slots, K // 512, N, K, GROUP_SIZE, ROWS, True,
        )
    result = gl.sum(partial, axis=1)
    if SOFT_CAP != 0.0:
        result = libdevice.tanh(gl.div_rn(result, SOFT_CAP)) * SOFT_CAP
    if Y.dtype.element_ty == gl.bfloat16:
        result = result.to(gl.bfloat16, fp_downcast_rounding="rtne")
    gl.store(Y + rows, result, rows < N if N % ROWS else True)


@gluon.jit
def _groupwise_w6_head_gemv(
    X, Pair, Third, Scales, Y, N: gl.constexpr, K: gl.constexpr, GROUP_SIZE: gl.constexpr,
    ROWS: gl.constexpr, SOFT_CAP: gl.constexpr = 0.0,
):
    _groupwise_w6_rows(
        X, Pair, Third, Scales, Y, gl.program_id(0) * ROWS, N, K, GROUP_SIZE, ROWS, SOFT_CAP,
    )


@gluon.jit
def _w6_batched_chunk(
    inputs, partials, Pair, Third, Scales, rows, lanes, input_cols, slots, chunk,
    N: gl.constexpr, K: gl.constexpr, GROUP_SIZE: gl.constexpr, ROWS: gl.constexpr,
    TILE_M: gl.constexpr, TAIL: gl.constexpr, USE_DOT2: gl.constexpr,
):
    updated = ()
    for index in gl.static_range(TILE_M):
        tile = _w6_tile(
            inputs[index], Pair, Third, Scales, rows, lanes, input_cols, slots, chunk,
            N, K, GROUP_SIZE, ROWS, TAIL, USE_DOT2,
        )
        updated += (partials[index] + tile,)
    return updated


@gluon.jit
def _groupwise_w6_batched_rows(
    X, Pair, Third, Scales, Y, row_start, M: gl.constexpr, N: gl.constexpr, K: gl.constexpr,
    GROUP_SIZE: gl.constexpr, ROWS: gl.constexpr, TILE_M: gl.constexpr, SOFT_CAP: gl.constexpr,
    USE_DOT2: gl.constexpr,
):
    # Keep the M1 lane layout. Each output wave owns independent
    # accumulators for the token rows that share its decoded weights.
    tile: gl.constexpr = gl.BlockedLayout([1, 1, 16], [1, 32, 1], [ROWS, 1, 1], [2, 1, 0])
    row_lane: gl.constexpr = gl.SliceLayout(2, tile)
    input_layout: gl.constexpr = gl.SliceLayout(0, tile)
    rows = row_start + gl.arange(0, ROWS, gl.SliceLayout(1, row_lane))
    lanes = gl.arange(0, 32, gl.SliceLayout(0, row_lane))
    input_lanes = gl.arange(0, 32, gl.SliceLayout(1, input_layout))
    input_slots = gl.arange(0, 16, gl.SliceLayout(0, input_layout))
    input_cols = input_lanes[:, None] * 16 + input_slots[None, :]
    slots = input_slots[None, :][None, :, :]
    first_token = gl.program_id(1) * TILE_M
    inputs = ()
    for index in gl.static_range(TILE_M):
        token = first_token + index
        if M % TILE_M:
            token = gl.minimum(token, M - 1)
        inputs += (X + token * K,)

    zero = gl.zeros((ROWS, 32), gl.float32, row_lane)
    partials = (zero,) * TILE_M
    for chunk in range(K // 512):
        partials = _w6_batched_chunk(
            inputs, partials, Pair, Third, Scales, rows, lanes, input_cols, slots, chunk,
            N, K, GROUP_SIZE, ROWS, TILE_M, False, USE_DOT2,
        )
    if K % 512:
        partials = _w6_batched_chunk(
            inputs, partials, Pair, Third, Scales, rows, lanes, input_cols, slots, K // 512,
            N, K, GROUP_SIZE, ROWS, TILE_M, True, USE_DOT2,
        )
    for index in gl.static_range(TILE_M):
        token = first_token + index
        result = gl.sum(partials[index], axis=1)
        if SOFT_CAP != 0.0:
            result = libdevice.tanh(gl.div_rn(result, SOFT_CAP)) * SOFT_CAP
        if Y.dtype.element_ty == gl.bfloat16:
            result = result.to(gl.bfloat16, fp_downcast_rounding="rtne")
        valid = rows < N if N % ROWS else True
        if M % TILE_M:
            valid = valid & (token < M)
        gl.store(Y + token * N + rows, result, valid)


@gluon.jit
def _groupwise_w6_head_batched_gemv(
    X, Pair, Third, Scales, Y, M: gl.constexpr, N: gl.constexpr, K: gl.constexpr,
    GROUP_SIZE: gl.constexpr, ROWS: gl.constexpr, TILE_M: gl.constexpr, SOFT_CAP: gl.constexpr,
    USE_DOT2: gl.constexpr = False,
):
    _groupwise_w6_batched_rows(
        X, Pair, Third, Scales, Y, gl.program_id(0) * ROWS,
        M, N, K, GROUP_SIZE, ROWS, TILE_M, SOFT_CAP, USE_DOT2,
    )


@gluon.jit
def _groupwise_w6_qkv_batched_gemv(
    XQ, XK, XV, Pair, Third, Scales, Y,
    Q_END: gl.constexpr, K_END: gl.constexpr, M: gl.constexpr, N: gl.constexpr, K: gl.constexpr,
    GROUP_SIZE: gl.constexpr, ROWS: gl.constexpr, TILE_M: gl.constexpr,
    USE_DOT2: gl.constexpr = False,
):
    row_start = gl.program_id(0) * ROWS
    x = gl.where(row_start < Q_END, XQ, gl.where(row_start < K_END, XK, XV))
    _groupwise_w6_batched_rows(
        x, Pair, Third, Scales, Y, row_start, M, N, K, GROUP_SIZE, ROWS, TILE_M, 0.0, USE_DOT2,
    )


@gluon.jit
def _groupwise_w6_qkv_gemv(
    XQ, XK, XV, Pair, Third, Scales, Y,
    Q_END: gl.constexpr, K_END: gl.constexpr, N: gl.constexpr, K: gl.constexpr,
    GROUP_SIZE: gl.constexpr, ROWS: gl.constexpr,
):
    row_start = gl.program_id(0) * ROWS
    # Every row in a workgroup belongs to the same activation.
    x = gl.where(row_start < Q_END, XQ, gl.where(row_start < K_END, XK, XV))
    _groupwise_w6_rows(x, Pair, Third, Scales, Y, row_start, N, K, GROUP_SIZE, ROWS, 0.0)


def _packed_group_size(
    x: torch.Tensor, pair: torch.Tensor, third: torch.Tensor, scales: torch.Tensor,
) -> int:
    width = x.shape[1]
    if (
        pair.ndim != 2 or pair.shape[1] != width // _RECORD_SIZE or pair.dtype != torch.int64
        or third.shape != pair.shape or third.dtype != torch.int32
        or scales.ndim != 2 or scales.shape[0] != pair.shape[0] or scales.dtype != torch.float16
        or any(t.device != x.device or not t.is_contiguous() for t in (pair, third, scales))
    ):
        raise ValueError("Expected contiguous packed signed W6 weights and FP16 scales")
    if scales.shape[1] == width // 64:
        return 64
    if scales.shape[1] == width // 32:
        return 32
    raise ValueError("W6 packed scale count must match G32 or G64")


def groupwise_w6_head_linear(
    x: torch.Tensor, pair: torch.Tensor, third: torch.Tensor, scales: torch.Tensor,
    *, out_dtype: torch.dtype = torch.float32, num_warps: int = 4, soft_cap: float = 0.0,
) -> torch.Tensor:
    """Project one BF16 token with FP32 accumulation and optional FP32 softcap."""
    if x.ndim != 2 or x.shape[0] != 1 or x.shape[1] not in _WIDTHS or x.dtype != torch.bfloat16:
        raise ValueError("Expected one BF16 activation row of width 2304 or 3072")
    if out_dtype not in (torch.bfloat16, torch.float32):
        raise ValueError("Projection output dtype must be BF16 or FP32")
    if soft_cap and (out_dtype != torch.float32 or soft_cap < 0 or not isfinite(soft_cap)):
        raise ValueError("Softcap requires a positive finite cap and FP32 output")
    if num_warps not in (2, 4):
        raise ValueError("W6 projection requires two or four waves per workgroup")
    group_size = _packed_group_size(x, pair, third, scales)
    width = x.shape[1]
    n = pair.shape[0]

    if not x.is_cuda:
        group_count = width // group_size
        words = (pair, pair >> 32, third.long())
        codes = [(word >> shift) & 63 for word in words for shift in range(0, 30, 6)]
        last = ((words[0] >> 30) & 3) | ((words[1] >> 28) & 12) | ((words[2] >> 26) & 48)
        codes = torch.stack([*codes, last], dim=-1).reshape(n, group_count, group_size)
        signed = ((codes + 32) % 64 - 32).float()
        products = signed * x.float().reshape(group_count, group_size)
        result = (products.sum(dim=-1) * scales.float()).sum(dim=1, keepdim=True).T
        if soft_cap:
            result = (result / soft_cap).tanh() * soft_cap
        return result.to(out_dtype)

    result = x.new_empty((1, n), dtype=out_dtype)
    if n:
        _groupwise_w6_head_gemv[(triton.cdiv(n, num_warps),)](
            x.contiguous(), pair, third, scales, result, n, width, group_size, num_warps,
            soft_cap, num_warps=num_warps,
        )
    return result


def _use_batched_dot2(x: torch.Tensor, tile_m: int) -> bool:
    return (
        tile_m == 8
        and os.environ.get("INKLING_AMD_W6_BATCH_DOT2") == "1"
        and torch.version.hip is not None
        and torch.cuda.get_device_properties(x.device).gcnArchName.split(":", 1)[0] == "gfx1151"
    )


def groupwise_w6_head_batched_linear(
    x: torch.Tensor, pair: torch.Tensor, third: torch.Tensor, scales: torch.Tensor,
    *, out_dtype: torch.dtype = torch.float32, num_warps: int = 4, soft_cap: float = 0.0,
    tile_m: int = 4,
) -> torch.Tensor:
    """Project up to eight BF16 tokens using the same W6 bank and row arithmetic."""
    if x.ndim != 2 or not 1 <= x.shape[0] <= 8 or x.shape[1] not in _WIDTHS or x.dtype != torch.bfloat16:
        raise ValueError("Expected 1–8 BF16 activation rows of width 2304 or 3072")
    if tile_m not in (2, 4, 8):
        raise ValueError("The batched W6 tile must contain two, four or eight tokens")
    if out_dtype not in (torch.bfloat16, torch.float32):
        raise ValueError("Projection output dtype must be BF16 or FP32")
    if soft_cap and (out_dtype != torch.float32 or soft_cap < 0 or not isfinite(soft_cap)):
        raise ValueError("Softcap requires a positive finite cap and FP32 output")
    if num_warps not in (2, 4):
        raise ValueError("W6 projection requires two or four waves per workgroup")

    m, width = x.shape
    if m == 1 or not x.is_cuda:
        rows = [
            groupwise_w6_head_linear(
                row, pair, third, scales, out_dtype=out_dtype, num_warps=num_warps, soft_cap=soft_cap,
            )
            for row in x.split(1)
        ]
        return rows[0] if m == 1 else torch.cat(rows)

    group_size = _packed_group_size(x, pair, third, scales)
    n = pair.shape[0]
    result = x.new_empty((m, n), dtype=out_dtype)
    if n:
        _groupwise_w6_head_batched_gemv[(triton.cdiv(n, num_warps), triton.cdiv(m, tile_m))](
            x.contiguous(), pair, third, scales, result, m, n, width, group_size, num_warps,
            tile_m, soft_cap, _use_batched_dot2(x, tile_m), num_warps=num_warps,
        )
    return result


def groupwise_w6_qkv_gemv(
    x_q: torch.Tensor, x_k: torch.Tensor, x_v: torch.Tensor,
    pair: torch.Tensor, third: torch.Tensor, scales: torch.Tensor, ends: tuple[int, int, int],
    *, num_warps: int = 4,
) -> torch.Tensor:
    """Project three M1 inputs over workgroup-aligned ranges to BF16."""
    if x_q.ndim != 2 or x_q.shape[1] not in _WIDTHS:
        raise ValueError("Expected W6 activation width 2304 or 3072")
    if num_warps not in (2, 4):
        raise ValueError("W6 projection requires two or four waves per workgroup")
    width = x_q.shape[1]
    activations = (x_q, x_k, x_v)
    if any(
        x.shape != (1, width) or x.dtype != torch.bfloat16
        or x.device != x_q.device or not x.is_contiguous()
        for x in activations
    ):
        raise ValueError("Expected three contiguous BF16 activation rows on the same device")
    group_size = _packed_group_size(x_q, pair, third, scales)
    q_end, k_end, v_end = ends
    if not (0 < q_end < k_end < v_end <= pair.shape[0]) or any(end % num_warps for end in ends):
        raise ValueError("Q/K/V row ranges must end on workgroup row boundaries")

    if not x_q.is_cuda:
        return torch.cat([
            groupwise_w6_head_linear(
                x, pair[start:end], third[start:end], scales[start:end], out_dtype=torch.bfloat16,
                num_warps=num_warps,
            )
            for x, start, end in zip(activations, (0, q_end, k_end), ends)
        ], dim=-1)

    result = x_q.new_empty((1, v_end))
    _groupwise_w6_qkv_gemv[(v_end // num_warps,)](
        x_q, x_k, x_v, pair, third, scales, result, q_end, k_end, v_end, width, group_size,
        num_warps, num_warps=num_warps,
    )
    return result


def groupwise_w6_qkv_batched(
    x_q: torch.Tensor, x_k: torch.Tensor, x_v: torch.Tensor,
    pair: torch.Tensor, third: torch.Tensor, scales: torch.Tensor, ends: tuple[int, int, int],
    *, num_warps: int = 4, tile_m: int = 4,
) -> torch.Tensor:
    """Project the Q/R, K and V workgroup-aligned ranges for up to eight BF16 tokens."""
    if x_q.ndim != 2 or not 1 <= x_q.shape[0] <= 8 or x_q.shape[1] not in _WIDTHS:
        raise ValueError("Expected 1–8 W6 activation rows of width 2304 or 3072")
    if num_warps not in (2, 4):
        raise ValueError("W6 projection requires two or four waves per workgroup")
    if tile_m not in (2, 4, 8):
        raise ValueError("The batched W6 tile must contain two, four or eight tokens")
    activations = (x_q, x_k, x_v)
    if any(x.shape != x_q.shape or x.dtype != torch.bfloat16
           or x.device != x_q.device or not x.is_contiguous() for x in activations):
        raise ValueError("Expected three equal-shaped contiguous BF16 inputs on the same device")
    m, width = x_q.shape
    if m == 1:
        return groupwise_w6_qkv_gemv(*activations, pair, third, scales, ends, num_warps=num_warps)
    group_size = _packed_group_size(x_q, pair, third, scales)
    q_end, k_end, v_end = ends
    if not (0 < q_end < k_end < v_end <= pair.shape[0]) or any(end % num_warps for end in ends):
        raise ValueError("Q/K/V row ranges must end on workgroup row boundaries")
    if not x_q.is_cuda:
        return torch.cat([
            groupwise_w6_qkv_gemv(*rows, pair, third, scales, ends, num_warps=num_warps)
            for rows in zip(*(x.split(1) for x in activations))
        ], dim=0)

    result = x_q.new_empty((m, v_end))
    _groupwise_w6_qkv_batched_gemv[(v_end // num_warps, triton.cdiv(m, tile_m))](
        *activations, pair, third, scales, result, q_end, k_end, m, v_end, width,
        group_size, num_warps, tile_m, _use_batched_dot2(x_q, tile_m), num_warps=num_warps,
    )
    return result
