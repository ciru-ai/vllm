# SPDX-License-Identifier: Apache-2.0
# Modified or added by Ciru for Inkling 2 Mini on AMD gfx1151.
"""Lossless C1 coda packet pre-tiling for M16/N128/K64 gfx1151 WMMA."""

from typing import NamedTuple

import torch
from triton.experimental import gluon
from triton.experimental.gluon import language as gl
from triton.experimental.gluon.language.amd import AMDWMMALayout
from triton.experimental.gluon.language.amd.rdna3 import wmma
from triton.experimental.gluon.language.extra import libdevice


@gluon.jit
def _expand_c1_groups(value, layout: gl.constexpr):
    expanded: gl.constexpr = gl.DistributedLinearLayout(
        [[0, 0, 1], [0, 0, 2], [0, 0, 4], [0, 1, 0], [0, 2, 0]],
        [[1, 0, 0], [2, 0, 0], [4, 0, 0], [8, 0, 0], [64, 0, 0]],
        [[16, 0, 0], [32, 0, 0]], [], [128, 4, 8],
    )
    value = gl.convert_layout(value, gl.SliceLayout(2, expanded), assert_trivial=True)
    shared, _ = gl.broadcast(value[:, :, None], gl.zeros((128, 4, 8), gl.float32, expanded))
    return gl.convert_layout(gl.reshape(shared, (128, 32)), layout, assert_trivial=True)


@gluon.jit
def _c1_bf16_pair(packed, raw, row_scale, Table, expert):
    # One lookup for each original G16 tag. Both FP16 fields travel in one load.
    groups: gl.constexpr = gl.DistributedLinearLayout(
        [[0, 1], [0, 2]],
        [[1, 0], [2, 0], [4, 0], [8, 0], [64, 0]],
        [[16, 0], [32, 0]], [], [128, 4],
    )
    group_rows: gl.constexpr = gl.SliceLayout(1, groups)
    group = gl.arange(0, 4, gl.SliceLayout(0, groups))
    raw = gl.convert_layout(raw, group_rows, assert_trivial=True)
    row = gl.convert_layout(row_scale, group_rows, assert_trivial=True)
    tag = (raw[:, None] >> (group[None, :] * 8)) & 255
    entries = gl.load(Table.to(gl.pointer_type(gl.uint32)) + expert * 256 + tag)
    scale = entries.to(gl.uint16).to(gl.float16, bitcast=True).to(gl.float32)
    offset = (entries >> 16).to(gl.uint16).to(gl.float16, bitcast=True).to(gl.float32)
    scale = _expand_c1_groups(scale * row[:, None], packed.type.layout)
    offset = _expand_c1_groups(offset * row[:, None], packed.type.layout)
    low = (packed & 15).to(gl.float32) * scale + offset
    high = (packed >> 4).to(gl.float32) * scale + offset
    return (
        low.to(gl.bfloat16, fp_downcast_rounding="rtne"),
        high.to(gl.bfloat16, fp_downcast_rounding="rtne"),
    )


@gluon.jit
def _coda_wmma_pretiled(
    X, Packed, Tags, RowScale, Table, Sorted, Expert, Padded, Y, Gamma,
    N: gl.constexpr, K: gl.constexpr, ROUTES: gl.constexpr,
    TOKENS: gl.constexpr, FUSED: gl.constexpr, GATE: gl.constexpr, UP: gl.constexpr,
):
    m = gl.program_id(0)
    n = gl.program_id(1)
    if m * 16 >= gl.load(Padded):
        return
    expert = gl.load(Expert + m)
    if expert < 0:
        return
    mma: gl.constexpr = AMDWMMALayout(1, True, [[0, 1], [0, 2]])
    a_load: gl.constexpr = gl.BlockedLayout([1, 64], [16, 2], [1, 4], [0, 1])
    # Each wave decodes two fragments before exchanging halves for WMMA.
    b_byte: gl.constexpr = gl.DistributedLinearLayout(
        [[0, 1], [0, 2], [0, 4], [0, 8], [0, 16]],
        [[1, 0], [2, 0], [4, 0], [8, 0], [64, 0]],
        [[16, 0], [32, 0]], [], [128, 32],
    )
    a_range = gl.arange(0, 16, gl.SliceLayout(1, a_load))
    a_cols = gl.arange(0, 64, gl.SliceLayout(0, a_load))
    b_row = gl.arange(0, 128, gl.SliceLayout(1, b_byte))
    b_cols = gl.arange(0, 32, gl.SliceLayout(0, b_byte))
    route = gl.load(Sorted + m * 16 + a_range).to(gl.int64)
    valid = (route >= 0) & (route < ROUTES)
    rows = route // TOKENS
    tile = (expert.to(gl.int64) * (N // 128) + n) * (K // 64)
    row_scale = gl.load(RowScale + expert.to(gl.int64) * N + n * 128 + b_row)
    row_scale = row_scale.to(gl.float32)
    acc = gl.zeros((16, 128), gl.float32, mma)
    for k in range(K // 64):
        x = gl.load(X + rows[:, None] * K + k * 64 + a_cols[None, :], valid[:, None], 0)
        packet_row = (tile + k) * 128 + b_row
        offsets = packet_row[:, None] * 32 + b_cols[None, :]
        packed = gl.load(Packed + offsets).to(gl.uint32)
        raw = gl.load(Tags + packet_row).to(gl.uint32)
        low, high = _c1_bf16_pair(packed, raw, row_scale, Table, expert)
        weights = gl.reshape(gl.join(low, high), (128, 64))
        a = gl.convert_layout(x, gl.DotOperandLayout(0, mma, 16), assert_trivial=True)
        b = gl.convert_layout(gl.permute(weights, (1, 0)), gl.DotOperandLayout(1, mma, 16))
        acc = wmma(a, b, acc)
    out_layout: gl.constexpr = gl.BlockedLayout([1, 2], [1, 32], [1, 4], [1, 0])
    out_rows: gl.constexpr = gl.SliceLayout(1, out_layout)
    route = gl.convert_layout(route, out_rows)
    valid = gl.convert_layout(valid, out_rows)
    out = gl.convert_layout(acc.to(gl.bfloat16, fp_downcast_rounding="rtne"), out_layout)
    if FUSED:
        gate, up = gl.split(gl.reshape(out, (16, 64, 2)))
        gate = gate.to(gl.float32)
        up = up.to(gl.float32)
        gamma = gl.load(Gamma + route, valid, other=0)
        sigmoid = gl.div_rn(1.0, 1.0 + libdevice.exp(-gate))
        mid = GATE * libdevice.tanh(gate * (1.0 / GATE))
        mid = ((mid * sigmoid) * UP) * libdevice.tanh(up * (1.0 / UP))
        gamma = gl.convert_layout(gamma, gl.SliceLayout(1, gate.type.layout), assert_trivial=True)
        mid = (mid * gamma[:, None]).to(gl.bfloat16, fp_downcast_rounding="rtne")
        mid = gl.convert_layout(mid, out_layout)
        cols = n * 64 + gl.arange(0, 64, gl.SliceLayout(0, out_layout))
        gl.store(Y + route[:, None] * (N // 2) + cols[None, :], mid, valid[:, None])
    else:
        cols = n * 128 + gl.arange(0, 128, gl.SliceLayout(0, out_layout))
        gl.store(Y + route[:, None] * N + cols[None, :], out, valid[:, None])


class CodaWmmaBank(NamedTuple):
    packed: torch.Tensor
    tags: torch.Tensor
    row_scales: torch.Tensor
    table: torch.Tensor


def _pack_wmma_packets(packed, raw):
    experts, n, half_k = packed.shape
    tiles = (experts, n // 128, 128, half_k // 32)
    tiled_packed = packed.reshape(*tiles, 32).transpose(2, 3).contiguous()
    tiled_raw = raw.reshape(*tiles, 4).transpose(2, 3).contiguous()
    return tiled_packed, tiled_raw.view(torch.int32).squeeze(-1)


def pack_coda_wmma(packed, tags, row_scales, table) -> CodaWmmaBank:
    """Transpose C1 packet and tag bytes into tile order, retaining the metadata."""
    experts, n, half_k = packed.shape
    if (
        packed.dtype != torch.uint8 or tags.dtype != torch.uint8
        or tags.shape != (experts, n, half_k // 8)
        or row_scales.shape != (experts, n) or row_scales.dtype != torch.float16
        or table.shape != (experts, 256, 2) or table.dtype != torch.float16
        or not (experts and n) or n % 128 or half_k not in (576, 1152)
    ):
        raise ValueError("Expected a C1 expert bank aligned to N128/K64")
    if any(not t.is_contiguous() or t.device != packed.device
           for t in (packed, tags, row_scales, table)) or table.data_ptr() % 4:
        raise ValueError("Expected contiguous C1 bank tensors and aligned tables on one device")
    return CodaWmmaBank(*_pack_wmma_packets(packed, tags), row_scales, table)


def coda_wmma_projection(x, bank: CodaWmmaBank, plan, *, fused_activation=None):
    """Project using the existing sorted route plan and BF16 route slots."""
    if (plan.block_m, plan.block_n, plan.block_k) != (16, 128, 64):
        raise ValueError("The pre-tiled C1 WMMA projection requires M16/N128/K64")
    experts, n_tiles, k_tiles, _, _ = bank.packed.shape
    n, k = n_tiles * 128, k_tiles * 64
    route_shape = (plan.num_tokens, plan.top_k)
    if (
        experts != plan.num_experts or x.dtype != torch.bfloat16
        or x.shape not in ((plan.num_tokens, k), (*route_shape, k))
    ):
        raise ValueError("Expected BF16 C1 activations matching the route plan")
    operands = (*bank, x, plan.sorted_ids, plan.expert_ids, plan.padded_tokens)
    gamma, gate_cap, up_cap = None, 0.0, 0.0
    if fused_activation is not None:
        gamma, gate_cap, up_cap = fused_activation
        if (
            x.ndim != 2 or gamma.shape != route_shape or gamma.dtype != torch.float32
            or gate_cap <= 0 or up_cap <= 0
        ):
            raise ValueError("Expected W13 inputs, FP32 route weights and positive softcaps")
        operands = (*operands, gamma)
    if any(not t.is_contiguous() or t.device != x.device for t in operands):
        raise ValueError("Expected contiguous C1 projection operands on one device")
    out = x.new_empty((*route_shape, n // 2 if gamma is not None else n))
    _coda_wmma_pretiled[(plan.expert_ids.numel(), n_tiles)](
        x, *bank, plan.sorted_ids, plan.expert_ids, plan.padded_tokens, out, gamma,
        N=n, K=k, ROUTES=plan.num_tokens * plan.top_k,
        TOKENS=plan.top_k if x.ndim == 2 else 1, FUSED=gamma is not None,
        GATE=gate_cap, UP=up_cap, num_warps=4, num_stages=1, enable_fp_fusion=False,
    )
    return out
