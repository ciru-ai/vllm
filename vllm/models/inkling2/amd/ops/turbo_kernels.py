# SPDX-License-Identifier: Apache-2.0
"""Ciru gfx1151 turbo kernels for Inkling 2 Mini PP & TG on vLLM."""

from typing import NamedTuple
import os

import torch
import triton
import triton.language as tl
from triton.experimental import gluon
from triton.experimental.gluon import language as gl
from triton.experimental.gluon.language.amd import AMDWMMALayout
from triton.experimental.gluon.language.amd.rdna3 import wmma
from triton.experimental.gluon.language.extra import libdevice
from triton.language.extra.hip import libdevice as tl_hip_lib

from vllm.v1.worker.gpu.sample.greedy_argmax import _argmax_finish
from .coda_wmma_pretile import _c1_bf16_pair, _expand_c1_groups
from .source_wmma_pretile import _finite_bf16, _unpack_e2m1

# ============================================================================
# 1. FP16-DIRECT NVFP4 UNPACK + PRE-TILED MoE WMMA (M16 DECODE & M64 PREFILL)
# ============================================================================

@gluon.jit
def _unpack_e2m1_f16x2_scaled(packed, scale_byte):
    sel = 0x000C000C | ((packed & 7) << 8) | ((packed & 0x70) << 20)
    mags = gl.inline_asm_elementwise(
        "v_perm_b32 $0, $1, $2, $3", constraints="=v,v,v,v",
        args=[0x4A484644, 0x42403C00, sel], dtype=gl.uint32, is_pure=True, pack=1,
    )
    signed = mags | ((packed & 8) << 12) | ((packed & 128) << 24)
    scale_f16 = (scale_byte << 7) + 0x1C00
    scale_pk = scale_f16 | (scale_f16 << 16)
    scaled_pk = gl.inline_asm_elementwise(
        "v_pk_mul_f16 $0, $1, $2", constraints="=v,v,v",
        args=[signed, scale_pk], dtype=gl.uint32, is_pure=True, pack=1,
    )
    low = scaled_pk.to(gl.uint16).to(gl.float16, bitcast=True)
    high = (scaled_pk >> 16).to(gl.uint16).to(gl.float16, bitcast=True)
    return low, high


@gluon.jit
def _source_wmma_turbo_kernel(
    X, Packed, Raw, Outer, Sorted, Expert, Padded, Y, Gamma,
    N: gl.constexpr, K: gl.constexpr, ROUTES,
    TOKENS: gl.constexpr, FUSED: gl.constexpr, GATE: gl.constexpr, UP: gl.constexpr,
    M_TILE: gl.constexpr, USE_F16: gl.constexpr,
):
    m = gl.program_id(0)
    n = gl.program_id(1)
    if m * M_TILE >= gl.load(Padded):
        return
    expert = gl.load(Expert + m)
    if expert < 0:
        return
    if M_TILE == 16:
        mma: gl.constexpr = AMDWMMALayout(1, True, [[0, 1], [0, 2]])
        a_load: gl.constexpr = gl.BlockedLayout([1, 64], [16, 2], [1, 4], [0, 1])
        out_layout: gl.constexpr = gl.DistributedLinearLayout(
            [[0, 1], [0, 4], [0, 8], [0, 64]],
            [[1, 0], [2, 0], [4, 0], [8, 0], [0, 2]],
            [[0, 16], [0, 32]], [], [16, 128],
        )
    else:
        mma: gl.constexpr = AMDWMMALayout(1, True, [[1, 0], [0, 1]])
        a_load: gl.constexpr = gl.BlockedLayout([1, 8], [8, 4], [2, 2], [0, 1])
        out_layout: gl.constexpr = gl.BlockedLayout([1, 2], [2, 16], [2, 2], [1, 0])

    b_byte: gl.constexpr = gl.DistributedLinearLayout(
        [[0, 1], [0, 2], [0, 4], [0, 8], [0, 16]],
        [[1, 0], [2, 0], [4, 0], [8, 0], [64, 0]],
        [[16, 0], [32, 0]], [], [128, 32],
    )
    a_range = gl.arange(0, M_TILE, gl.SliceLayout(1, a_load))
    a_cols = gl.arange(0, 64, gl.SliceLayout(0, a_load))
    b_row = gl.arange(0, 128, gl.SliceLayout(1, b_byte))
    byte_cols = gl.arange(0, 32, gl.SliceLayout(0, b_byte))
    route = gl.load(Sorted + m * M_TILE + a_range).to(gl.int64)
    valid = (route >= 0) & (route < ROUTES)
    rows = route // TOKENS
    tile = (expert.to(gl.int64) * (N // 128) + n) * (K // 64)
    outer = gl.load(Outer + expert)
    acc = gl.zeros((M_TILE, 128), gl.float32, mma)
    for k in range(K // 64):
        x = gl.load(
            X + rows[:, None] * K + (k * 64 + a_cols[None, :]), valid[:, None], 0,
        )
        packet_row = (tile + k) * 128 + b_row
        packed = gl.load(Packed + packet_row[:, None] * 32 + byte_cols[None, :]).to(gl.uint32)
        raw = gl.load(Raw + packet_row).to(gl.uint32)
        scale_byte = (raw[:, None] >> ((byte_cols[None, :] // 8) * 8)) & 255
        if USE_F16:
            x = x.to(gl.float16)
            low, high = _unpack_e2m1_f16x2_scaled(packed, scale_byte)
        else:
            scale_bits = ((scale_byte << 4) + (119 << 7)).to(gl.uint16)
            scale = scale_bits.to(gl.bfloat16, bitcast=True).to(gl.float32)
            low, high = _unpack_e2m1(packed)
            low = _finite_bf16((low * scale) * outer)
            high = _finite_bf16((high * scale) * outer)
        weights = gl.reshape(gl.join(low, high), (128, 64))
        a = gl.convert_layout(x, gl.DotOperandLayout(0, mma, 16))
        b = gl.convert_layout(gl.permute(weights, (1, 0)), gl.DotOperandLayout(1, mma, 16))
        acc = wmma(a, b, acc)
    if USE_F16:
        acc = acc * outer
    out_rows: gl.constexpr = gl.SliceLayout(1, out_layout)
    route = gl.convert_layout(route, out_rows)
    valid = gl.convert_layout(valid, out_rows)
    out = gl.convert_layout(acc.to(gl.bfloat16, fp_downcast_rounding="rtne"), out_layout)
    if FUSED:
        gate, up = gl.split(gl.reshape(out, (M_TILE, 64, 2)))
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


def source_wmma_turbo_projection(
    x, bank, plan, *, fused_activation=None, use_f16: bool = True,
):
    experts, n_tiles, k_tiles, _, _ = bank.packed.shape
    n, k = n_tiles * 128, k_tiles * 64
    route_shape = (plan.num_tokens, plan.top_k)
    gamma, gate_cap, up_cap = None, 0.0, 0.0
    if fused_activation is not None:
        gamma, gate_cap, up_cap = fused_activation
    out = x.new_empty((*route_shape, n // 2 if gamma is not None else n))
    bm = plan.block_m
    ns = 2 if bm == 64 else 1
    _source_wmma_turbo_kernel[(plan.expert_ids.numel(), n_tiles)](
        x, bank.packed, bank.raw_scales, bank.outer_scales,
        plan.sorted_ids, plan.expert_ids, plan.padded_tokens, out, gamma,
        N=n, K=k, ROUTES=plan.num_tokens * plan.top_k,
        TOKENS=plan.top_k if x.ndim == 2 else 1, FUSED=gamma is not None,
        GATE=gate_cap, UP=up_cap, M_TILE=bm, USE_F16=use_f16,
        num_warps=4, num_stages=ns, enable_fp_fusion=False,
    )
    return out


@gluon.jit
def _coda_wmma_prefill_m64(
    X, Packed, Tags, RowScale, Table, Sorted, Expert, Padded, Y, Gamma,
    N: gl.constexpr, K: gl.constexpr, ROUTES,
    TOKENS: gl.constexpr, FUSED: gl.constexpr, GATE: gl.constexpr, UP: gl.constexpr,
):
    m = gl.program_id(0)
    n = gl.program_id(1)
    if m * 64 >= gl.load(Padded):
        return
    expert = gl.load(Expert + m)
    if expert < 0:
        return
    mma: gl.constexpr = AMDWMMALayout(1, True, [[1, 0], [0, 1]])
    a_load: gl.constexpr = gl.BlockedLayout([1, 8], [8, 4], [2, 2], [0, 1])
    b_byte: gl.constexpr = gl.DistributedLinearLayout(
        [[0, 1], [0, 2], [0, 4], [0, 8], [0, 16]],
        [[1, 0], [2, 0], [4, 0], [8, 0], [64, 0]],
        [[16, 0], [32, 0]], [], [128, 32],
    )
    a_range = gl.arange(0, 64, gl.SliceLayout(1, a_load))
    a_cols = gl.arange(0, 64, gl.SliceLayout(0, a_load))
    b_row = gl.arange(0, 128, gl.SliceLayout(1, b_byte))
    b_cols = gl.arange(0, 32, gl.SliceLayout(0, b_byte))
    route = gl.load(Sorted + m * 64 + a_range).to(gl.int64)
    valid = (route >= 0) & (route < ROUTES)
    rows = route // TOKENS
    tile = (expert.to(gl.int64) * (N // 128) + n) * (K // 64)
    row_scale = gl.load(RowScale + expert.to(gl.int64) * N + n * 128 + b_row).to(gl.float32)
    acc = gl.zeros((64, 128), gl.float32, mma)
    for k in range(K // 64):
        x = gl.load(X + rows[:, None] * K + k * 64 + a_cols[None, :], valid[:, None], 0)
        packet_row = (tile + k) * 128 + b_row
        packed = gl.load(Packed + packet_row[:, None] * 32 + b_cols[None, :]).to(gl.uint32)
        raw = gl.load(Tags + packet_row).to(gl.uint32)
        low, high = _c1_bf16_pair(packed, raw, row_scale, Table, expert)
        weights = gl.reshape(gl.join(low, high), (128, 64))
        a = gl.convert_layout(x, gl.DotOperandLayout(0, mma, 16))
        b = gl.convert_layout(gl.permute(weights, (1, 0)), gl.DotOperandLayout(1, mma, 16))
        acc = wmma(a, b, acc)
    out_layout: gl.constexpr = gl.BlockedLayout([1, 2], [2, 16], [2, 2], [1, 0])
    out_rows: gl.constexpr = gl.SliceLayout(1, out_layout)
    route = gl.convert_layout(route, out_rows)
    valid = gl.convert_layout(valid, out_rows)
    out = gl.convert_layout(acc.to(gl.bfloat16, fp_downcast_rounding="rtne"), out_layout)
    if FUSED:
        gate, up = gl.split(gl.reshape(out, (64, 64, 2)))
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


def coda_wmma_prefill_m64(x, bank, plan, *, fused_activation=None):
    experts, n_tiles, k_tiles, _, _ = bank.packed.shape
    n, k = n_tiles * 128, k_tiles * 64
    route_shape = (plan.num_tokens, plan.top_k)
    gamma, gate_cap, up_cap = None, 0.0, 0.0
    if fused_activation is not None:
        gamma, gate_cap, up_cap = fused_activation
    out = x.new_empty((*route_shape, n // 2 if gamma is not None else n))
    _coda_wmma_prefill_m64[(plan.expert_ids.numel(), n_tiles)](
        x, bank.packed, bank.tags, bank.row_scales, bank.table,
        plan.sorted_ids, plan.expert_ids, plan.padded_tokens, out, gamma,
        N=n, K=k, ROUTES=plan.num_tokens * plan.top_k,
        TOKENS=plan.top_k if x.ndim == 2 else 1, FUSED=gamma is not None,
        GATE=gate_cap, UP=up_cap, num_warps=4, num_stages=2, enable_fp_fusion=False,
    )
    return out


@gluon.jit
def _mixed_wmma_tile_m64(
    X, Packed, Raw, Outer, RowScale, Table, expert, rows, valid, n,
    N: gl.constexpr, K: gl.constexpr, KIND: gl.constexpr,
    MMA: gl.constexpr, A_LOAD: gl.constexpr,
):
    b_byte: gl.constexpr = gl.DistributedLinearLayout(
        [[0, 1], [0, 2], [0, 4], [0, 8], [0, 16]],
        [[1, 0], [2, 0], [4, 0], [8, 0], [64, 0]],
        [[16, 0], [32, 0]], [], [128, 32],
    )
    b_row = gl.arange(0, 128, gl.SliceLayout(1, b_byte))
    a_cols = gl.arange(0, 64, gl.SliceLayout(0, A_LOAD))
    byte_cols = gl.arange(0, 32, gl.SliceLayout(0, b_byte))
    tile = (expert.to(gl.int64) * (N // 128) + n) * (K // 64)
    if KIND == 0:
        outer = gl.load(Outer + expert)
    elif KIND == 2:
        row = gl.load(RowScale + expert.to(gl.int64) * N + n * 128 + b_row).to(gl.float32)
    acc = gl.zeros((64, 128), gl.float32, MMA)
    for k in range(K // 64):
        x = gl.load(X + rows[:, None] * K + k * 64 + a_cols[None, :], valid[:, None], 0)
        packet_row = (tile + k) * 128 + b_row
        packed = gl.load(Packed + packet_row[:, None] * 32 + byte_cols[None, :]).to(gl.uint32)
        raw = gl.load(Raw + packet_row).to(gl.uint32)
        if KIND == 0:
            scale_byte = (raw[:, None] >> ((byte_cols[None, :] // 8) * 8)) & 255
            scale_bits = ((scale_byte << 4) + (119 << 7)).to(gl.uint16)
            scale = scale_bits.to(gl.bfloat16, bitcast=True).to(gl.float32)
            low, high = _unpack_e2m1(packed)
            low = _finite_bf16((low * scale) * outer)
            high = _finite_bf16((high * scale) * outer)
        elif KIND == 1:
            scale_bits = (raw[:, None] >> ((byte_cols[None, :] // 16) * 16)).to(gl.uint16)
            scale = scale_bits.to(gl.bfloat16, bitcast=True).to(gl.float32)
            low = ((((packed & 15).to(gl.int32) ^ 8) - 8).to(gl.float32) * scale).to(gl.bfloat16, fp_downcast_rounding="rtne")
            high = ((((packed >> 4).to(gl.int32) ^ 8) - 8).to(gl.float32) * scale).to(gl.bfloat16, fp_downcast_rounding="rtne")
        else:
            low, high = _c1_bf16_pair(packed, raw, row, Table, expert)
        weights = gl.reshape(gl.join(low, high), (128, 64))
        a = gl.convert_layout(x, gl.DotOperandLayout(0, MMA, 16))
        b = gl.convert_layout(gl.permute(weights, (1, 0)), gl.DotOperandLayout(1, MMA, 16))
        acc = wmma(a, b, acc)
    return acc


@gluon.jit
def _mixed_wmma_prefill_m64(
    X, Packed, Raw, Outer, Kind, RowScale, Table, Sorted, Expert, Padded, Y, Gamma,
    N: gl.constexpr, K: gl.constexpr, ROUTES, TOKENS: gl.constexpr,
    FITTED_KIND: gl.constexpr, FUSED: gl.constexpr, GATE: gl.constexpr, UP: gl.constexpr,
):
    m = gl.program_id(0)
    n = gl.program_id(1)
    if m * 64 >= gl.load(Padded):
        return
    expert = gl.load(Expert + m)
    if expert < 0:
        return
    mma: gl.constexpr = AMDWMMALayout(1, True, [[1, 0], [0, 1]])
    a_load: gl.constexpr = gl.BlockedLayout([1, 8], [8, 4], [2, 2], [0, 1])
    a_range = gl.arange(0, 64, gl.SliceLayout(1, a_load))
    route = gl.load(Sorted + m * 64 + a_range).to(gl.int64)
    valid = (route >= 0) & (route < ROUTES)
    rows = route // TOKENS
    if gl.load(Kind + expert) == 0:
        acc = _mixed_wmma_tile_m64(
            X, Packed, Raw, Outer, RowScale, Table, expert, rows, valid, n, N, K, 0, mma, a_load,
        )
    else:
        acc = _mixed_wmma_tile_m64(
            X, Packed, Raw, Outer, RowScale, Table, expert, rows, valid, n, N, K, FITTED_KIND, mma, a_load,
        )
    out_layout: gl.constexpr = gl.BlockedLayout([1, 2], [2, 16], [2, 2], [1, 0])
    out_rows: gl.constexpr = gl.SliceLayout(1, out_layout)
    route = gl.convert_layout(route, out_rows)
    valid = gl.convert_layout(valid, out_rows)
    out = gl.convert_layout(acc.to(gl.bfloat16, fp_downcast_rounding="rtne"), out_layout)
    if FUSED:
        gate, up = gl.split(gl.reshape(out, (64, 64, 2)))
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


def mixed_wmma_prefill_m64(x, bank, plan, *, fused_activation=None):
    experts, n_tiles, k_tiles, _, _ = bank.packed.shape
    n, k = n_tiles * 128, k_tiles * 64
    route_shape = (plan.num_tokens, plan.top_k)
    gamma, gate_cap, up_cap = None, 0.0, 0.0
    if fused_activation is not None:
        gamma, gate_cap, up_cap = fused_activation
    out = x.new_empty((*route_shape, n // 2 if gamma is not None else n))
    _mixed_wmma_prefill_m64[(plan.expert_ids.numel(), n_tiles)](
        x, bank.packed, bank.raw_scales, bank.outer_scales, bank.kind,
        bank.row_scales, bank.table,
        plan.sorted_ids, plan.expert_ids, plan.padded_tokens, out, gamma,
        N=n, K=k, ROUTES=plan.num_tokens * plan.top_k,
        TOKENS=plan.top_k if x.ndim == 2 else 1, FITTED_KIND=bank.fitted_kind,
        FUSED=gamma is not None, GATE=gate_cap, UP=up_cap,
        num_warps=4, num_stages=2, enable_fp_fusion=False,
    )
    return out


# ============================================================================
# 2. FAST PREFILL ATTENTION (1 HEAD/CTA, SCALAR PAGE LOAD, FUSED GATE+SINK)
# ============================================================================

@triton.jit
def _turbo_sigmoid(x):
    return 1.0 / (1.0 + tl.math.exp2(-x * 1.4426950408889634))


@triton.jit
def _turbo_prefill_rel_attn_kernel(
    Q, K, V, Bias, Gate, Sink, Result,
    BlockTable, SeqLens, CuSeqlensQ,
    scale_log2,
    stride_qt, stride_qh,
    stride_kb, stride_kp, stride_kh,
    stride_vb, stride_vp, stride_vh,
    stride_bt, stride_bh,
    stride_gate, stride_sink,
    stride_res, stride_table,
    H: tl.constexpr, GROUP: tl.constexpr, D: tl.constexpr, PAGE: tl.constexpr,
    E_PAD: tl.constexpr, EXTENT: tl.constexpr, WINDOW: tl.constexpr,
    BLOCK_Q: tl.constexpr, BLOCK_N: tl.constexpr,
):
    q_block = tl.program_id(0)
    head = tl.program_id(1)
    req = tl.program_id(2)
    kv_head = head // GROUP

    q_start = tl.load(CuSeqlensQ + req)
    q_len = tl.load(CuSeqlensQ + req + 1) - q_start
    q0 = q_block * BLOCK_Q
    if q0 >= q_len:
        return
    seq_len = tl.load(SeqLens + req)
    past = seq_len - q_len

    q_offs = tl.arange(0, BLOCK_Q)
    q_idx = q0 + q_offs
    row_valid = q_idx < q_len
    token = (q_start + q_idx).to(tl.int64)
    d = tl.arange(0, D)

    q = tl.load(
        Q + token[:, None] * stride_qt + head * stride_qh + d[None, :],
        row_valid[:, None], other=0.0,
    )
    pos = past + q_idx
    first_pos = past + q0
    hi = tl.minimum(past + q0 + BLOCK_Q, seq_len)

    table = BlockTable + req * stride_table
    k_head_base = K + kv_head * stride_kh
    v_head_base = V + kv_head * stride_vh
    n_offs = tl.arange(0, BLOCK_N)

    m_i = tl.full([BLOCK_Q], float("-inf"), tl.float32)
    l_i = tl.zeros([BLOCK_Q], tl.float32)
    acc = tl.zeros([BLOCK_Q, D], tl.float32)

    end_block = tl.cdiv(hi, BLOCK_N)

    if WINDOW < 0:
        q_scaled = (q.to(tl.float32) * scale_log2).to(tl.bfloat16)
        first_bias_key = first_pos - E_PAD
        near_start_block = tl.maximum(first_bias_key, 0) // BLOCK_N
        for blk in range(0, near_start_block):
            key_begin = blk * BLOCK_N
            if BLOCK_N <= PAGE:
                page_id = tl.load(table + (key_begin // PAGE)).to(tl.int64)
                slot_base = key_begin % PAGE
                k = tl.load(k_head_base + page_id * stride_kb + (slot_base + n_offs)[:, None] * stride_kp + d[None, :])
            else:
                p0 = tl.load(table + (key_begin // PAGE)).to(tl.int64)
                p1 = tl.load(table + (key_begin // PAGE) + 1).to(tl.int64)
                pg = tl.where(n_offs < PAGE, p0, p1)
                sl = n_offs % PAGE
                k = tl.load(k_head_base + pg[:, None] * stride_kb + sl[:, None] * stride_kp + d[None, :])
            s = tl.dot(q_scaled, tl.trans(k))
            m_new = tl.maximum(m_i, tl.max(s, 1))
            alpha = tl.math.exp2(m_i - m_new)
            p = tl.math.exp2(s - m_new[:, None])
            l_i = l_i * alpha + tl.sum(p, 1)
            if BLOCK_N <= PAGE:
                v = tl.load(v_head_base + page_id * stride_vb + (slot_base + n_offs)[:, None] * stride_vp + d[None, :])
            else:
                v = tl.load(v_head_base + pg[:, None] * stride_vb + sl[:, None] * stride_vp + d[None, :])
            acc = acc * alpha[:, None] + tl.dot(p.to(v.dtype), v)
            m_i = m_new
    else:
        lo = tl.maximum(first_pos - WINDOW, 0) // BLOCK_N * BLOCK_N
        near_start_block = lo // BLOCK_N

    tile_end = tl.minimum((q_idx // 128 + 1) * 128, q_len)
    key0 = tl.cdiv(past + tile_end, 128) * 128 - E_PAD
    bias_rows = Bias + token * stride_bt + head * stride_bh - key0

    for blk in range(near_start_block, end_block):
        key_begin = blk * BLOCK_N
        keys = key_begin + n_offs
        key_valid = keys < hi
        if BLOCK_N <= PAGE:
            page_id = tl.load(table + (key_begin // PAGE)).to(tl.int64)
            slot_base = key_begin % PAGE
            k = tl.load(
                k_head_base + page_id * stride_kb + (slot_base + n_offs)[:, None] * stride_kp + d[None, :],
                key_valid[:, None], other=0.0,
            )
        else:
            p0 = tl.load(table + (key_begin // PAGE)).to(tl.int64)
            p1 = tl.load(table + (key_begin // PAGE) + 1, mask=(key_begin + PAGE) < hi, other=p0).to(tl.int64)
            pg = tl.where(n_offs < PAGE, p0, p1)
            sl = n_offs % PAGE
            k = tl.load(
                k_head_base + pg[:, None] * stride_kb + sl[:, None] * stride_kp + d[None, :],
                key_valid[:, None], other=0.0,
            )
        s = tl.dot(q, tl.trans(k)) * scale_log2
        dist = pos[:, None] - keys[None, :]
        valid = row_valid[:, None] & key_valid[None, :] & (dist >= 0)
        if WINDOW >= 0:
            valid &= dist <= WINDOW
        in_table = valid & (keys[None, :] >= key0[:, None])
        bias = tl.load(bias_rows[:, None] + keys[None, :], in_table, other=0.0)
        s = s + bias.to(tl.float32) * 1.4426950408889634
        s = tl.where(valid, s, float("-inf"))
        m_new = tl.maximum(m_i, tl.max(s, 1))
        m_ref = tl.where(m_new == float("-inf"), 0.0, m_new)
        alpha = tl.math.exp2(m_i - m_ref)
        p = tl.math.exp2(s - m_ref[:, None])
        l_i = l_i * alpha + tl.sum(p, 1)
        if BLOCK_N <= PAGE:
            v = tl.load(
                v_head_base + page_id * stride_vb + (slot_base + n_offs)[:, None] * stride_vp + d[None, :],
                key_valid[:, None], other=0.0,
            )
        else:
            v = tl.load(
                v_head_base + pg[:, None] * stride_vb + sl[:, None] * stride_vp + d[None, :],
                key_valid[:, None], other=0.0,
            )
        acc = acc * alpha[:, None] + tl.dot(p.to(v.dtype), v)
        m_i = m_new

    has_keys = l_i > 0.0
    out = acc / tl.where(has_keys, l_i, 1.0)[:, None]
    lse_e = tl.where(has_keys, (m_i + tl.math.log2(l_i)) * 0.6931471805599453, float("-inf"))
    gate = tl.load(Gate + token[:, None] * stride_gate + head * D + d[None, :], row_valid[:, None], other=0.0).to(tl.float32)
    sink = tl.load(Sink + token * stride_sink + head, row_valid, other=0.0).to(tl.float32)
    y = out * _turbo_sigmoid(gate) * _turbo_sigmoid(lse_e - sink)[:, None]
    tl.store(
        Result + token[:, None] * stride_res + head * D + d[None, :],
        y.to(Result.dtype.element_ty),
        row_valid[:, None],
    )


@triton.jit
def _gather_paged_kv_hnd_kernel(
    K_cache, V_cache, BlockTable, K_out, V_out,
    n_pages,
    stride_kb, stride_kp, stride_kh,
    stride_vb, stride_vp, stride_vh,
    stride_okh, stride_okn,
    stride_ovh, stride_ovn,
    KVH: tl.constexpr, PAGE: tl.constexpr, D: tl.constexpr,
):
    pg_i = tl.program_id(0)
    kvh = tl.program_id(1)
    page_id = tl.load(BlockTable + pg_i).to(tl.int64)
    p = tl.arange(0, PAGE)
    d = tl.arange(0, D)
    k = tl.load(K_cache + page_id * stride_kb + p[:, None] * stride_kp + kvh * stride_kh + d[None, :])
    v = tl.load(V_cache + page_id * stride_vb + p[:, None] * stride_vp + kvh * stride_vh + d[None, :])
    out_n = (pg_i * PAGE + p).to(tl.int64)
    tl.store(K_out + kvh * stride_okh + out_n[:, None] * stride_okn + d[None, :], k)
    tl.store(V_out + kvh * stride_ovh + out_n[:, None] * stride_ovn + d[None, :], v)


@triton.jit
def _turbo_prefill_rel_attn_near_merge_kernel(
    Q, K, V, Bias, Gate, Sink, OutFar, LseFar, Result,
    BlockTable, SeqLens, CuSeqlensQ,
    scale_log2,
    far_len,
    req_idx,
    stride_qt, stride_qh,
    stride_kb, stride_kp, stride_kh,
    stride_vb, stride_vp, stride_vh,
    stride_bt, stride_bh,
    stride_gate, stride_sink,
    stride_ofh, stride_ofq,
    stride_lfh, stride_lfq,
    stride_res, stride_table,
    H: tl.constexpr, GROUP: tl.constexpr, D: tl.constexpr, PAGE: tl.constexpr,
    E_PAD: tl.constexpr,
    BLOCK_Q: tl.constexpr, BLOCK_N: tl.constexpr,
):
    q_block = tl.program_id(0)
    head = tl.program_id(1)
    kv_head = head // GROUP

    q_start = tl.load(CuSeqlensQ + req_idx)
    q_len = tl.load(CuSeqlensQ + req_idx + 1) - q_start
    q0 = q_block * BLOCK_Q
    if q0 >= q_len:
        return
    seq_len = tl.load(SeqLens + req_idx)
    past = seq_len - q_len

    q_offs = tl.arange(0, BLOCK_Q)
    q_idx = q0 + q_offs
    row_valid = q_idx < q_len
    token = (q_start + q_idx).to(tl.int64)
    d = tl.arange(0, D)

    q = tl.load(
        Q + token[:, None] * stride_qt + head * stride_qh + d[None, :],
        row_valid[:, None], other=0.0,
    )
    pos = past + q_idx
    hi = tl.minimum(past + q0 + BLOCK_Q, seq_len)

    table = BlockTable + req_idx * stride_table
    k_head_base = K + kv_head * stride_kh
    v_head_base = V + kv_head * stride_vh
    n_offs = tl.arange(0, BLOCK_N)

    lse_far = tl.load(
        LseFar + head * stride_lfh + q_idx * stride_lfq,
        row_valid, other=float("-inf"),
    ).to(tl.float32)
    out_far = tl.load(
        OutFar + head * stride_ofh + q_idx[:, None] * stride_ofq + d[None, :],
        row_valid[:, None], other=0.0,
    ).to(tl.float32)
    m_i = lse_far * 1.4426950408889634
    l_i = tl.where(row_valid, 1.0, 0.0)
    acc = out_far

    near_start_block = far_len // BLOCK_N
    end_block = tl.cdiv(hi, BLOCK_N)

    tile_end = tl.minimum((q_idx // 128 + 1) * 128, q_len)
    key0 = tl.cdiv(past + tile_end, 128) * 128 - E_PAD
    bias_rows = Bias + token * stride_bt + head * stride_bh - key0

    for blk in range(near_start_block, end_block):
        key_begin = blk * BLOCK_N
        keys = key_begin + n_offs
        key_valid = keys < hi
        if BLOCK_N <= PAGE:
            page_id = tl.load(table + (key_begin // PAGE)).to(tl.int64)
            slot_base = key_begin % PAGE
            k = tl.load(
                k_head_base + page_id * stride_kb + (slot_base + n_offs)[:, None] * stride_kp + d[None, :],
                key_valid[:, None], other=0.0,
            )
        else:
            p0 = tl.load(table + (key_begin // PAGE)).to(tl.int64)
            p1 = tl.load(table + (key_begin // PAGE) + 1, mask=(key_begin + PAGE) < hi, other=p0).to(tl.int64)
            pg = tl.where(n_offs < PAGE, p0, p1)
            sl = n_offs % PAGE
            k = tl.load(
                k_head_base + pg[:, None] * stride_kb + sl[:, None] * stride_kp + d[None, :],
                key_valid[:, None], other=0.0,
            )
        s = tl.dot(q, tl.trans(k)) * scale_log2
        dist = pos[:, None] - keys[None, :]
        valid = row_valid[:, None] & key_valid[None, :] & (dist >= 0)
        in_table = valid & (keys[None, :] >= key0[:, None])
        bias = tl.load(bias_rows[:, None] + keys[None, :], in_table, other=0.0)
        s = s + bias.to(tl.float32) * 1.4426950408889634
        s = tl.where(valid, s, float("-inf"))
        m_new = tl.maximum(m_i, tl.max(s, 1))
        m_ref = tl.where(m_new == float("-inf"), 0.0, m_new)
        alpha = tl.math.exp2(m_i - m_ref)
        p = tl.math.exp2(s - m_ref[:, None])
        l_i = l_i * alpha + tl.sum(p, 1)
        if BLOCK_N <= PAGE:
            v = tl.load(
                v_head_base + page_id * stride_vb + (slot_base + n_offs)[:, None] * stride_vp + d[None, :],
                key_valid[:, None], other=0.0,
            )
        else:
            v = tl.load(
                v_head_base + pg[:, None] * stride_vb + sl[:, None] * stride_vp + d[None, :],
                key_valid[:, None], other=0.0,
            )
        acc = acc * alpha[:, None] + tl.dot(p.to(v.dtype), v)
        m_i = m_new

    has_keys = l_i > 0.0
    out = acc / tl.where(has_keys, l_i, 1.0)[:, None]
    lse_e = tl.where(has_keys, (m_i + tl.math.log2(l_i)) * 0.6931471805599453, float("-inf"))
    gate = tl.load(Gate + token[:, None] * stride_gate + head * D + d[None, :], row_valid[:, None], other=0.0).to(tl.float32)
    sink = tl.load(Sink + token * stride_sink + head, row_valid, other=0.0).to(tl.float32)
    y = out * _turbo_sigmoid(gate) * _turbo_sigmoid(lse_e - sink)[:, None]
    tl.store(
        Result + token[:, None] * stride_res + head * D + d[None, :],
        y.to(Result.dtype.element_ty),
        row_valid[:, None],
    )


def turbo_gated_rel_attention(
    q: torch.Tensor, key_cache: torch.Tensor, value_cache: torch.Tensor,
    sheared_bias: torch.Tensor, gate: torch.Tensor, sinks: torch.Tensor,
    result: torch.Tensor, *, block_table: torch.Tensor, cache_seqlens: torch.Tensor,
    cu_seqlens_q: torch.Tensor, max_seqlen_q: int, num_actual_tokens: int,
    softmax_scale: float, window: int,
) -> None:
    num_tokens, num_heads, head_dim = q.shape
    page, num_kv_heads = key_cache.shape[1:3]
    nt = num_actual_tokens
    if nt < num_tokens:
        result[nt:].zero_()
    if nt == 0:
        return
    num_reqs = cache_seqlens.shape[0]
    e_pad = sheared_bias.shape[2]
    extent = e_pad - 256
    bq, bn, nw = 64, 32, 4
    # Hybrid AOTriton-Far + Triton-Near for long-context global prefill (>=16K past):
    # eliminates the 32K->64K MALL spill cliff (2.40x faster at 64K, 0.0000 BF16 max err).
    if (
        window < 0
        and max_seqlen_q >= 64
        and page in (16, 32, 64)
        and q.dtype in (torch.bfloat16, torch.float16)
        and not torch.cuda.is_current_stream_capturing()
    ):
        seq_cpu = cache_seqlens.tolist()
        cu_cpu = cu_seqlens_q.tolist()
        align = max(bn, page)
        hybrid_reqs = []
        short_reqs = []
        for r in range(num_reqs):
            ql = cu_cpu[r + 1] - cu_cpu[r]
            past = seq_cpu[r] - ql
            far_len = (max(past - e_pad, 0) // align) * align
            if ql >= 64 and far_len >= 2048:
                hybrid_reqs.append((r, cu_cpu[r], ql, far_len))
            elif ql > 0:
                short_reqs.append((r, cu_cpu[r], ql))
        if hybrid_reqs:
            for r, qs, ql in short_reqs:
                qe = qs + ql
                cu1 = cu_seqlens_q.new_tensor([0, ql])
                _turbo_prefill_rel_attn_kernel[(triton.cdiv(ql, bq), num_heads, 1)](
                    q[qs:qe], key_cache, value_cache, sheared_bias[qs:qe],
                    gate[qs:qe], sinks[qs:qe], result[qs:qe],
                    block_table[r : r + 1], cache_seqlens[r : r + 1], cu1,
                    softmax_scale * 1.4426950408889634,
                    q.stride(0), q.stride(1),
                    key_cache.stride(0), key_cache.stride(1), key_cache.stride(2),
                    value_cache.stride(0), value_cache.stride(1), value_cache.stride(2),
                    sheared_bias.stride(0), sheared_bias.stride(1),
                    gate.stride(0), sinks.stride(0),
                    result.stride(0), block_table.stride(0),
                    H=num_heads, GROUP=num_heads // num_kv_heads, D=head_dim, PAGE=page,
                    E_PAD=e_pad, EXTENT=extent, WINDOW=window,
                    BLOCK_Q=bq, BLOCK_N=bn, num_warps=nw, num_stages=1,
                )
            for r, qs, ql, far_len in hybrid_reqs:
                n_pages = far_len // page
                k_far = q.new_empty((1, num_kv_heads, far_len, head_dim))
                v_far = q.new_empty((1, num_kv_heads, far_len, head_dim))
                _gather_paged_kv_hnd_kernel[(n_pages, num_kv_heads)](
                    key_cache, value_cache, block_table[r], k_far[0], v_far[0], n_pages,
                    key_cache.stride(0), key_cache.stride(1), key_cache.stride(2),
                    value_cache.stride(0), value_cache.stride(1), value_cache.stride(2),
                    k_far.stride(1), k_far.stride(2), v_far.stride(1), v_far.stride(2),
                    KVH=num_kv_heads, PAGE=page, D=head_dim, num_warps=4,
                )
                q_hnd = q[qs : qs + ql].transpose(0, 1).unsqueeze(0).contiguous()
                out_far, lse_far = torch.ops.aten._scaled_dot_product_flash_attention(
                    q_hnd, k_far, v_far, 0.0, False, False, scale=softmax_scale
                )[:2]
                out_far_3d = out_far[0]
                lse_far_2d = lse_far[0]
                _turbo_prefill_rel_attn_near_merge_kernel[(triton.cdiv(ql, bq), num_heads)](
                    q, key_cache, value_cache, sheared_bias, gate, sinks,
                    out_far_3d, lse_far_2d, result,
                    block_table, cache_seqlens, cu_seqlens_q,
                    softmax_scale * 1.4426950408889634, far_len, r,
                    q.stride(0), q.stride(1),
                    key_cache.stride(0), key_cache.stride(1), key_cache.stride(2),
                    value_cache.stride(0), value_cache.stride(1), value_cache.stride(2),
                    sheared_bias.stride(0), sheared_bias.stride(1),
                    gate.stride(0), sinks.stride(0),
                    out_far_3d.stride(0), out_far_3d.stride(1),
                    lse_far_2d.stride(0), lse_far_2d.stride(1),
                    result.stride(0), block_table.stride(0),
                    H=num_heads, GROUP=num_heads // num_kv_heads, D=head_dim, PAGE=page,
                    E_PAD=e_pad, BLOCK_Q=bq, BLOCK_N=bn, num_warps=nw, num_stages=1,
                )
            return
    grid = (triton.cdiv(max_seqlen_q, bq), num_heads, num_reqs)
    _turbo_prefill_rel_attn_kernel[grid](
        q, key_cache, value_cache, sheared_bias, gate, sinks, result,
        block_table, cache_seqlens, cu_seqlens_q,
        softmax_scale * 1.4426950408889634,
        q.stride(0), q.stride(1),
        key_cache.stride(0), key_cache.stride(1), key_cache.stride(2),
        value_cache.stride(0), value_cache.stride(1), value_cache.stride(2),
        sheared_bias.stride(0), sheared_bias.stride(1),
        gate.stride(0), sinks.stride(0),
        result.stride(0), block_table.stride(0),
        H=num_heads, GROUP=num_heads // num_kv_heads, D=head_dim, PAGE=page,
        E_PAD=e_pad, EXTENT=extent, WINDOW=window,
        BLOCK_Q=bq, BLOCK_N=bn, num_warps=nw, num_stages=1,
    )


# ============================================================================
# 3. FAST FP32-EXACT M=8 ROUTER GEMM (128 CTAs READING BF16 WEIGHTS!)
# ============================================================================

@triton.jit
def _router_gemm_m8_fp32_kernel(X, W, Out, M: tl.constexpr, K: tl.constexpr, BLOCK_K: tl.constexpr):
    e = tl.program_id(0)
    rows = tl.arange(0, 8)
    r_mask = rows < M
    k_offs = tl.arange(0, BLOCK_K)
    acc = tl.zeros((8,), tl.float32)
    w_base = W + e.to(tl.int64) * K
    for k0 in range(0, K, BLOCK_K):
        k = k0 + k_offs
        mask = k < K
        w = tl.load(w_base + k, mask, other=0.0).to(tl.float32)
        x = tl.load(X + rows[:, None] * K + k[None, :], r_mask[:, None] & mask[None, :], other=0.0).to(tl.float32)
        acc += tl.sum(x * w[None, :], axis=1)
    tl.store(Out + rows * 128 + e, acc, r_mask)


def router_gemm_m8_fp32(x: torch.Tensor, w_bf16: torch.Tensor) -> torch.Tensor:
    m = x.shape[0]
    out = torch.empty((m, 128), dtype=torch.float32, device=x.device)
    _router_gemm_m8_fp32_kernel[(128,)](
        x, w_bf16, out, M=m, K=x.shape[1], BLOCK_K=512, num_warps=4, enable_fp_fusion=False,
    )
    return out


# ============================================================================
# 4. TARGET W6 WMMA (QKVR, WO WITH SPLIT-K, LM_HEAD WITH SOFTCAP)
# ============================================================================

class W6WmmaWeight(NamedTuple):
    p4: torch.Tensor
    p2: torch.Tensor
    scales: torch.Tensor
    out_features: int
    padded_out: int
    in_features: int


@torch.no_grad()
def pack_w6_wmma(weight: torch.Tensor) -> W6WmmaWeight:
    n, k = weight.shape
    n_pad = ((n + 127) // 128) * 128
    groups = k // 64
    nt, kt = n_pad // 128, k // 64
    tp4 = torch.empty((nt, kt, 128, 32), dtype=torch.uint8, device=weight.device)
    tp2 = torch.empty((nt, kt, 128, 4), dtype=torch.uint32, device=weight.device)
    ts = torch.empty((nt, kt, 128), dtype=torch.float16, device=weight.device)
    chunk_tiles = 64  # 8192 rows per chunk to keep peak packing memory small
    for t0 in range(0, nt, chunk_tiles):
        t1 = min(nt, t0 + chunk_tiles)
        r0, r1 = t0 * 128, min(n, t1 * 128)
        rows = (t1 - t0) * 128
        w = torch.zeros((rows, k), dtype=torch.float32, device=weight.device)
        if r1 > r0:
            w[: r1 - r0].copy_(weight[r0:r1].float())
        w = w.reshape(rows, groups, 64)
        scale = (w.abs().amax(dim=-1) / 31.0).clamp_min_(1e-7).half()
        q = (w / scale.float().unsqueeze(-1)).round().clamp(-31, 31).to(torch.int32) + 32
        lo4 = (q & 15).reshape(rows, groups, 32, 2)
        p4 = (lo4[..., 0] | (lo4[..., 1] << 4)).to(torch.uint8)
        hi2 = (q >> 4).reshape(rows, groups, 16, 4)
        p2 = (hi2[..., 0] | (hi2[..., 1] << 2) | (hi2[..., 2] << 4) | (hi2[..., 3] << 6)).to(torch.uint8)
        ct = t1 - t0
        tp4[t0:t1].copy_(p4.reshape(ct, 128, kt, 32).transpose(1, 2))
        tp2[t0:t1].copy_(p2.reshape(ct, 128, kt, 16).transpose(1, 2).contiguous().view(torch.uint32))
        ts[t0:t1].copy_(scale.reshape(ct, 128, kt).transpose(1, 2))
    return W6WmmaWeight(tp4, tp2, ts, n, n_pad, k)


@torch.no_grad()
def pack_w6_wmma_qkv(weight: torch.Tensor, q_side: int, kv_size: int) -> tuple[W6WmmaWeight, int, int, int]:
    """Pack [q_side, kv_size, kv_size] with q_side padded to 128-row boundary."""
    k = weight.shape[1]
    q_pad = ((q_side + 127) // 128) * 128
    k_pad = ((kv_size + 127) // 128) * 128
    v_pad = ((kv_size + 127) // 128) * 128
    total = q_pad + k_pad + v_pad
    buf = torch.zeros((total, k), dtype=weight.dtype, device=weight.device)
    buf[:q_side].copy_(weight[:q_side])
    buf[q_pad : q_pad + kv_size].copy_(weight[q_side : q_side + kv_size])
    buf[q_pad + k_pad : q_pad + k_pad + kv_size].copy_(weight[q_side + kv_size : q_side + 2 * kv_size])
    return pack_w6_wmma(buf), q_pad, q_pad + k_pad, total


@gluon.jit
def _w6_wmma_m8_kernel(
    X, P4, P2, Scales, Y, M,
    N: gl.constexpr, K: gl.constexpr, SPLITS: gl.constexpr,
    OUT_FP32: gl.constexpr, SOFT_CAP: gl.constexpr,
):
    tile_n = gl.program_id(0)
    split = gl.program_id(1)
    k_tiles: gl.constexpr = K // 64
    tps: gl.constexpr = k_tiles // SPLITS
    mma: gl.constexpr = AMDWMMALayout(1, True, [[0, 1], [0, 2]])
    a_load: gl.constexpr = gl.BlockedLayout([1, 64], [16, 2], [1, 4], [0, 1])
    b_byte: gl.constexpr = gl.DistributedLinearLayout(
        [[0, 1], [0, 2], [0, 4], [0, 8], [0, 16]],
        [[1, 0], [2, 0], [4, 0], [8, 0], [64, 0]],
        [[16, 0], [32, 0]], [], [128, 32],
    )
    a_row = gl.arange(0, 16, gl.SliceLayout(1, a_load))
    a_col = gl.arange(0, 64, gl.SliceLayout(0, a_load))
    b_row = gl.arange(0, 128, gl.SliceLayout(1, b_byte))
    b_col = gl.arange(0, 32, gl.SliceLayout(0, b_byte))
    sc_col = gl.arange(0, 128, gl.SliceLayout(0, mma))
    acc = gl.zeros((16, 128), gl.float32, mma)
    for step in range(tps):
        kt = split * tps + step
        x = gl.load(X + a_row[:, None] * K + (kt * 64 + a_col[None, :]), a_row[:, None] < M, 0)
        tidx = (tile_n * k_tiles + kt) * 128 + b_row
        p4 = gl.load(P4 + tidx[:, None] * 32 + b_col[None, :]).to(gl.uint32)
        p2_word = gl.load(P2 + tidx[:, None] * 4 + (b_col[None, :] // 8))
        shift = (b_col[None, :] & 7) * 4
        h2_pair = (p2_word >> shift) & 15
        lo_q = (((p4 & 15) | ((h2_pair & 3) << 4)).to(gl.int32) - 32).to(gl.bfloat16)
        hi_q = (((p4 >> 4) | ((h2_pair >> 2) << 4)).to(gl.int32) - 32).to(gl.bfloat16)
        w = gl.reshape(gl.join(lo_q, hi_q), (128, 64))
        a = gl.convert_layout(x, gl.DotOperandLayout(0, mma, 16))
        b = gl.convert_layout(gl.permute(w, (1, 0)), gl.DotOperandLayout(1, mma, 16))
        blk = wmma(a, b, gl.zeros((16, 128), gl.float32, mma))
        sc = gl.load(Scales + (tile_n * k_tiles + kt) * 128 + sc_col).to(gl.float32)
        acc += blk * sc[None, :]
    out_layout: gl.constexpr = gl.DistributedLinearLayout(
        [[0, 1], [0, 4], [0, 8], [0, 64]],
        [[1, 0], [2, 0], [4, 0], [8, 0], [0, 2]],
        [[0, 16], [0, 32]], [], [16, 128],
    )
    out_row = gl.arange(0, 16, gl.SliceLayout(1, out_layout))
    out_col = gl.arange(0, 128, gl.SliceLayout(0, out_layout))
    val = gl.convert_layout(acc, out_layout)
    if SPLITS == 1:
        if SOFT_CAP > 0.0:
            val = libdevice.tanh(val * (1.0 / SOFT_CAP)) * SOFT_CAP
        if not OUT_FP32:
            val = val.to(gl.bfloat16, fp_downcast_rounding="rtne")
    gl.store(
        Y + (split * M + out_row[:, None]) * N + tile_n * 128 + out_col[None, :],
        val, out_row[:, None] < M,
    )


@gluon.jit
def _w6_wmma_qkv_m8_kernel(
    XQ, XK, XV, P4, P2, Scales, Y, M,
    Q_TILES: gl.constexpr, K_TILES_END: gl.constexpr,
    N: gl.constexpr, K: gl.constexpr,
):
    tile_n = gl.program_id(0)
    X = gl.where(tile_n < Q_TILES, XQ, gl.where(tile_n < K_TILES_END, XK, XV))
    k_tiles: gl.constexpr = K // 64
    mma: gl.constexpr = AMDWMMALayout(1, True, [[0, 1], [0, 2]])
    a_load: gl.constexpr = gl.BlockedLayout([1, 64], [16, 2], [1, 4], [0, 1])
    b_byte: gl.constexpr = gl.DistributedLinearLayout(
        [[0, 1], [0, 2], [0, 4], [0, 8], [0, 16]],
        [[1, 0], [2, 0], [4, 0], [8, 0], [64, 0]],
        [[16, 0], [32, 0]], [], [128, 32],
    )
    a_row = gl.arange(0, 16, gl.SliceLayout(1, a_load))
    a_col = gl.arange(0, 64, gl.SliceLayout(0, a_load))
    b_row = gl.arange(0, 128, gl.SliceLayout(1, b_byte))
    b_col = gl.arange(0, 32, gl.SliceLayout(0, b_byte))
    sc_col = gl.arange(0, 128, gl.SliceLayout(0, mma))
    acc = gl.zeros((16, 128), gl.float32, mma)
    for kt in range(k_tiles):
        x = gl.load(X + a_row[:, None] * K + (kt * 64 + a_col[None, :]), a_row[:, None] < M, 0)
        tidx = (tile_n * k_tiles + kt) * 128 + b_row
        p4 = gl.load(P4 + tidx[:, None] * 32 + b_col[None, :]).to(gl.uint32)
        p2_word = gl.load(P2 + tidx[:, None] * 4 + (b_col[None, :] // 8))
        shift = (b_col[None, :] & 7) * 4
        h2_pair = (p2_word >> shift) & 15
        lo_q = (((p4 & 15) | ((h2_pair & 3) << 4)).to(gl.int32) - 32).to(gl.bfloat16)
        hi_q = (((p4 >> 4) | ((h2_pair >> 2) << 4)).to(gl.int32) - 32).to(gl.bfloat16)
        w = gl.reshape(gl.join(lo_q, hi_q), (128, 64))
        a = gl.convert_layout(x, gl.DotOperandLayout(0, mma, 16))
        b = gl.convert_layout(gl.permute(w, (1, 0)), gl.DotOperandLayout(1, mma, 16))
        blk = wmma(a, b, gl.zeros((16, 128), gl.float32, mma))
        sc = gl.load(Scales + (tile_n * k_tiles + kt) * 128 + sc_col).to(gl.float32)
        acc += blk * sc[None, :]
    out_layout: gl.constexpr = gl.DistributedLinearLayout(
        [[0, 1], [0, 4], [0, 8], [0, 64]],
        [[1, 0], [2, 0], [4, 0], [8, 0], [0, 2]],
        [[0, 16], [0, 32]], [], [16, 128],
    )
    out_row = gl.arange(0, 16, gl.SliceLayout(1, out_layout))
    out_col = gl.arange(0, 128, gl.SliceLayout(0, out_layout))
    val = gl.convert_layout(acc.to(gl.bfloat16, fp_downcast_rounding="rtne"), out_layout)
    gl.store(Y + out_row[:, None] * N + tile_n * 128 + out_col[None, :], val, out_row[:, None] < M)


@triton.jit
def _reduce_splits_bf16(Partial, Y, SIZE, SPLITS: tl.constexpr, BLOCK: tl.constexpr):
    col = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    sp = tl.arange(0, SPLITS)
    part = tl.load(Partial + sp[:, None] * SIZE + col[None, :], col[None, :] < SIZE, 0)
    tl.store(Y + col, tl.sum(part, axis=0).to(tl.bfloat16), col < SIZE)


@gluon.jit
def _w6_wmma_batched_f16_kernel(
    X, P4, P2, Scales, Y, M,
    N: gl.constexpr, K: gl.constexpr, M_TILE: gl.constexpr,
    OUT_FP32: gl.constexpr, SOFT_CAP: gl.constexpr,
):
    tile_n = gl.program_id(0)
    tile_m = gl.program_id(1)
    k_tiles: gl.constexpr = K // 64
    if M_TILE == 16:
        mma: gl.constexpr = AMDWMMALayout(1, True, [[0, 1], [0, 2]])
        a_load: gl.constexpr = gl.BlockedLayout([1, 64], [16, 2], [1, 4], [0, 1])
        out_layout: gl.constexpr = gl.DistributedLinearLayout(
            [[0, 1], [0, 4], [0, 8], [0, 64]],
            [[1, 0], [2, 0], [4, 0], [8, 0], [0, 2]],
            [[0, 16], [0, 32]], [], [16, 128],
        )
    else:
        mma: gl.constexpr = AMDWMMALayout(1, True, [[1, 0], [0, 1]])
        a_load: gl.constexpr = gl.BlockedLayout([1, 8], [8, 4], [2, 2], [0, 1])
        out_layout: gl.constexpr = gl.BlockedLayout([1, 2], [2, 16], [2, 2], [1, 0])
    b_byte: gl.constexpr = gl.DistributedLinearLayout(
        [[0, 1], [0, 2], [0, 4], [0, 8], [0, 16]],
        [[1, 0], [2, 0], [4, 0], [8, 0], [64, 0]],
        [[16, 0], [32, 0]], [], [128, 32],
    )
    a_row = tile_m * M_TILE + gl.arange(0, M_TILE, gl.SliceLayout(1, a_load))
    a_col = gl.arange(0, 64, gl.SliceLayout(0, a_load))
    b_row = gl.arange(0, 128, gl.SliceLayout(1, b_byte))
    b_col = gl.arange(0, 32, gl.SliceLayout(0, b_byte))
    acc = gl.zeros((M_TILE, 128), gl.float32, mma)
    for kt in range(k_tiles):
        x = gl.load(X + a_row[:, None] * K + (kt * 64 + a_col[None, :]), a_row[:, None] < M, 0).to(gl.float16)
        tidx = (tile_n * k_tiles + kt) * 128 + b_row
        p4 = gl.load(P4 + tidx[:, None] * 32 + b_col[None, :]).to(gl.uint32)
        p2_word = gl.load(P2 + tidx[:, None] * 4 + (b_col[None, :] // 8))
        shift = (b_col[None, :] & 7) * 4
        h2_pair = (p2_word >> shift) & 15
        sc = gl.load(Scales + tidx)
        lo_q = (((p4 & 15) | ((h2_pair & 3) << 4)).to(gl.int32) - 32).to(gl.float16) * sc[:, None]
        hi_q = (((p4 >> 4) | ((h2_pair >> 2) << 4)).to(gl.int32) - 32).to(gl.float16) * sc[:, None]
        w = gl.reshape(gl.join(lo_q, hi_q), (128, 64))
        a = gl.convert_layout(x, gl.DotOperandLayout(0, mma, 16))
        b = gl.convert_layout(gl.permute(w, (1, 0)), gl.DotOperandLayout(1, mma, 16))
        acc = wmma(a, b, acc)
    out_row = tile_m * M_TILE + gl.arange(0, M_TILE, gl.SliceLayout(1, out_layout))
    out_col = gl.arange(0, 128, gl.SliceLayout(0, out_layout))
    val = gl.convert_layout(acc, out_layout)
    if SOFT_CAP > 0.0:
        val = libdevice.tanh(val * (1.0 / SOFT_CAP)) * SOFT_CAP
    if not OUT_FP32:
        val = val.to(gl.bfloat16, fp_downcast_rounding="rtne")
    gl.store(Y + out_row[:, None] * N + tile_n * 128 + out_col[None, :], val, out_row[:, None] < M)


@gluon.jit
def _w6_wmma_qkv_batched_f16_kernel(
    XQ, XK, XV, P4, P2, Scales, Y, M,
    Q_TILES: gl.constexpr, K_TILES_END: gl.constexpr,
    N: gl.constexpr, K: gl.constexpr, M_TILE: gl.constexpr,
):
    tile_n = gl.program_id(0)
    tile_m = gl.program_id(1)
    X = gl.where(tile_n < Q_TILES, XQ, gl.where(tile_n < K_TILES_END, XK, XV))
    k_tiles: gl.constexpr = K // 64
    if M_TILE == 16:
        mma: gl.constexpr = AMDWMMALayout(1, True, [[0, 1], [0, 2]])
        a_load: gl.constexpr = gl.BlockedLayout([1, 64], [16, 2], [1, 4], [0, 1])
        out_layout: gl.constexpr = gl.DistributedLinearLayout(
            [[0, 1], [0, 4], [0, 8], [0, 64]],
            [[1, 0], [2, 0], [4, 0], [8, 0], [0, 2]],
            [[0, 16], [0, 32]], [], [16, 128],
        )
    else:
        mma: gl.constexpr = AMDWMMALayout(1, True, [[1, 0], [0, 1]])
        a_load: gl.constexpr = gl.BlockedLayout([1, 8], [8, 4], [2, 2], [0, 1])
        out_layout: gl.constexpr = gl.BlockedLayout([1, 2], [2, 16], [2, 2], [1, 0])
    b_byte: gl.constexpr = gl.DistributedLinearLayout(
        [[0, 1], [0, 2], [0, 4], [0, 8], [0, 16]],
        [[1, 0], [2, 0], [4, 0], [8, 0], [64, 0]],
        [[16, 0], [32, 0]], [], [128, 32],
    )
    a_row = tile_m * M_TILE + gl.arange(0, M_TILE, gl.SliceLayout(1, a_load))
    a_col = gl.arange(0, 64, gl.SliceLayout(0, a_load))
    b_row = gl.arange(0, 128, gl.SliceLayout(1, b_byte))
    b_col = gl.arange(0, 32, gl.SliceLayout(0, b_byte))
    acc = gl.zeros((M_TILE, 128), gl.float32, mma)
    for kt in range(k_tiles):
        x = gl.load(X + a_row[:, None] * K + (kt * 64 + a_col[None, :]), a_row[:, None] < M, 0).to(gl.float16)
        tidx = (tile_n * k_tiles + kt) * 128 + b_row
        p4 = gl.load(P4 + tidx[:, None] * 32 + b_col[None, :]).to(gl.uint32)
        p2_word = gl.load(P2 + tidx[:, None] * 4 + (b_col[None, :] // 8))
        shift = (b_col[None, :] & 7) * 4
        h2_pair = (p2_word >> shift) & 15
        sc = gl.load(Scales + tidx)
        lo_q = (((p4 & 15) | ((h2_pair & 3) << 4)).to(gl.int32) - 32).to(gl.float16) * sc[:, None]
        hi_q = (((p4 >> 4) | ((h2_pair >> 2) << 4)).to(gl.int32) - 32).to(gl.float16) * sc[:, None]
        w = gl.reshape(gl.join(lo_q, hi_q), (128, 64))
        a = gl.convert_layout(x, gl.DotOperandLayout(0, mma, 16))
        b = gl.convert_layout(gl.permute(w, (1, 0)), gl.DotOperandLayout(1, mma, 16))
        acc = wmma(a, b, acc)
    out_row = tile_m * M_TILE + gl.arange(0, M_TILE, gl.SliceLayout(1, out_layout))
    out_col = gl.arange(0, 128, gl.SliceLayout(0, out_layout))
    val = gl.convert_layout(acc, out_layout).to(gl.bfloat16, fp_downcast_rounding="rtne")
    gl.store(Y + out_row[:, None] * N + tile_n * 128 + out_col[None, :], val, out_row[:, None] < M)


def w6_wmma_linear(
    x: torch.Tensor, weight: W6WmmaWeight, *,
    out_dtype: torch.dtype = torch.bfloat16, soft_cap: float = 0.0, splits: int = 1,
) -> torch.Tensor:
    m, k = x.shape
    n_pad = weight.padded_out
    out = x.new_empty((m, n_pad), dtype=out_dtype)
    if m > 16:
        bm = 32 if m <= 32 else 64
        _w6_wmma_batched_f16_kernel[(n_pad // 128, triton.cdiv(m, bm))](
            x.contiguous(), weight.p4, weight.p2, weight.scales, out, m,
            N=n_pad, K=k, M_TILE=bm, OUT_FP32=out_dtype == torch.float32,
            SOFT_CAP=soft_cap, num_warps=4, num_stages=1,
        )
    elif splits == 1:
        _w6_wmma_m8_kernel[(n_pad // 128, 1)](
            x.contiguous(), weight.p4, weight.p2, weight.scales, out, m,
            N=n_pad, K=k, SPLITS=1, OUT_FP32=out_dtype == torch.float32,
            SOFT_CAP=soft_cap, num_warps=4, num_stages=1,
        )
    else:
        partial = torch.empty((splits, m, n_pad), dtype=torch.float32, device=x.device)
        _w6_wmma_m8_kernel[(n_pad // 128, splits)](
            x.contiguous(), weight.p4, weight.p2, weight.scales, partial, m,
            N=n_pad, K=k, SPLITS=splits, OUT_FP32=True, SOFT_CAP=0.0,
            num_warps=4, num_stages=1,
        )
        _reduce_splits_bf16[(triton.cdiv(m * n_pad, 256),)](
            partial, out, m * n_pad, splits, 256, num_warps=4,
        )
    return out[:, : weight.out_features]


def w6_wmma_qkv_linear(
    x_q: torch.Tensor, x_k: torch.Tensor, x_v: torch.Tensor,
    weight: W6WmmaWeight, q_pad: int, k_end: int,
) -> torch.Tensor:
    m, k = x_q.shape
    n_pad = weight.padded_out
    out = x_q.new_empty((m, n_pad), dtype=torch.bfloat16)
    if m > 16:
        bm = 32 if m <= 32 else 64
        _w6_wmma_qkv_batched_f16_kernel[(n_pad // 128, triton.cdiv(m, bm))](
            x_q.contiguous(), x_k.contiguous(), x_v.contiguous(),
            weight.p4, weight.p2, weight.scales, out, m,
            Q_TILES=q_pad // 128, K_TILES_END=k_end // 128,
            N=n_pad, K=k, M_TILE=bm, num_warps=4, num_stages=1,
        )
    else:
        _w6_wmma_qkv_m8_kernel[(n_pad // 128,)](
            x_q.contiguous(), x_k.contiguous(), x_v.contiguous(),
            weight.p4, weight.p2, weight.scales, out, m,
            Q_TILES=q_pad // 128, K_TILES_END=k_end // 128,
            N=n_pad, K=k, num_warps=4, num_stages=1,
        )
    return out


# ============================================================================
# 5. DRAFT W4 WMMA (BASE HEAD + BACKBONE) & W4 MARKOV ARGMAX (ZERO TARGET LOSS)
# ============================================================================

class W4WmmaWeight(NamedTuple):
    packed: torch.Tensor
    scales: torch.Tensor
    out_features: int
    padded_out: int
    in_features: int


@torch.no_grad()
def pack_w4_wmma(weight: torch.Tensor) -> W4WmmaWeight:
    n, k = weight.shape
    n_pad = ((n + 127) // 128) * 128
    groups = k // 64
    nt, kt = n_pad // 128, k // 64
    tp = torch.empty((nt, kt, 128, 32), dtype=torch.uint8, device=weight.device)
    ts = torch.empty((nt, kt, 128), dtype=torch.float16, device=weight.device)
    chunk_tiles = 64
    for t0 in range(0, nt, chunk_tiles):
        t1 = min(nt, t0 + chunk_tiles)
        r0, r1 = t0 * 128, min(n, t1 * 128)
        rows = (t1 - t0) * 128
        w = torch.zeros((rows, k), dtype=torch.float32, device=weight.device)
        if r1 > r0:
            w[: r1 - r0].copy_(weight[r0:r1].float())
        w = w.reshape(rows, groups, 64)
        scale = (w.abs().amax(dim=-1) / 7.0).clamp_min_(1e-7).half()
        q = (w / scale.float().unsqueeze(-1)).round().clamp(-7, 7).to(torch.int32) + 8
        q = q.reshape(rows, k // 2, 2)
        packed = (q[..., 0] | (q[..., 1] << 4)).to(torch.uint8)
        ct = t1 - t0
        tp[t0:t1].copy_(packed.reshape(ct, 128, kt, 32).transpose(1, 2))
        ts[t0:t1].copy_(scale.reshape(ct, 128, kt).transpose(1, 2))
    return W4WmmaWeight(tp, ts, n, n_pad, k)


@gluon.jit
def _w4_wmma_m8_kernel(
    X, Packed, Scales, Y, M,
    N: gl.constexpr, K: gl.constexpr, SPLITS: gl.constexpr,
):
    tile_n = gl.program_id(0)
    split = gl.program_id(1)
    k_tiles: gl.constexpr = K // 64
    tps: gl.constexpr = k_tiles // SPLITS
    mma: gl.constexpr = AMDWMMALayout(1, True, [[0, 1], [0, 2]])
    a_load: gl.constexpr = gl.BlockedLayout([1, 64], [16, 2], [1, 4], [0, 1])
    b_byte: gl.constexpr = gl.DistributedLinearLayout(
        [[0, 1], [0, 2], [0, 4], [0, 8], [0, 16]],
        [[1, 0], [2, 0], [4, 0], [8, 0], [64, 0]],
        [[16, 0], [32, 0]], [], [128, 32],
    )
    a_row = gl.arange(0, 16, gl.SliceLayout(1, a_load))
    a_col = gl.arange(0, 64, gl.SliceLayout(0, a_load))
    b_row = gl.arange(0, 128, gl.SliceLayout(1, b_byte))
    b_col = gl.arange(0, 32, gl.SliceLayout(0, b_byte))
    sc_col = gl.arange(0, 128, gl.SliceLayout(0, mma))
    acc = gl.zeros((16, 128), gl.float32, mma)
    for step in range(tps):
        kt = split * tps + step
        x = gl.load(X + a_row[:, None] * K + (kt * 64 + a_col[None, :]), a_row[:, None] < M, 0)
        tidx = (tile_n * k_tiles + kt) * 128 + b_row
        packed = gl.load(Packed + tidx[:, None] * 32 + b_col[None, :]).to(gl.uint32)
        lo = ((packed & 15).to(gl.int32) - 8).to(gl.bfloat16)
        hi = ((packed >> 4).to(gl.int32) - 8).to(gl.bfloat16)
        w = gl.reshape(gl.join(lo, hi), (128, 64))
        a = gl.convert_layout(x, gl.DotOperandLayout(0, mma, 16))
        b = gl.convert_layout(gl.permute(w, (1, 0)), gl.DotOperandLayout(1, mma, 16))
        blk = wmma(a, b, gl.zeros((16, 128), gl.float32, mma))
        sc = gl.load(Scales + (tile_n * k_tiles + kt) * 128 + sc_col).to(gl.float32)
        acc += blk * sc[None, :]
    out_layout: gl.constexpr = gl.DistributedLinearLayout(
        [[0, 1], [0, 4], [0, 8], [0, 64]],
        [[1, 0], [2, 0], [4, 0], [8, 0], [0, 2]],
        [[0, 16], [0, 32]], [], [16, 128],
    )
    out_row = gl.arange(0, 16, gl.SliceLayout(1, out_layout))
    out_col = gl.arange(0, 128, gl.SliceLayout(0, out_layout))
    val = gl.convert_layout(acc, out_layout)
    if SPLITS == 1:
        val = val.to(gl.bfloat16, fp_downcast_rounding="rtne")
    gl.store(
        Y + (split * M + out_row[:, None]) * N + tile_n * 128 + out_col[None, :],
        val, out_row[:, None] < M,
    )


@gluon.jit
def _w4_wmma_batched_f16_kernel(
    X, Packed, Scales, Bias, Y, M,
    N: gl.constexpr, K: gl.constexpr, M_TILE: gl.constexpr, HAS_BIAS: gl.constexpr,
):
    tile_n = gl.program_id(0)
    tile_m = gl.program_id(1)
    k_tiles: gl.constexpr = K // 64
    if M_TILE == 16:
        mma: gl.constexpr = AMDWMMALayout(1, True, [[0, 1], [0, 2]])
        a_load: gl.constexpr = gl.BlockedLayout([1, 64], [16, 2], [1, 4], [0, 1])
        out_layout: gl.constexpr = gl.DistributedLinearLayout(
            [[0, 1], [0, 4], [0, 8], [0, 64]],
            [[1, 0], [2, 0], [4, 0], [8, 0], [0, 2]],
            [[0, 16], [0, 32]], [], [16, 128],
        )
    else:
        mma: gl.constexpr = AMDWMMALayout(1, True, [[1, 0], [0, 1]])
        a_load: gl.constexpr = gl.BlockedLayout([1, 8], [8, 4], [2, 2], [0, 1])
        out_layout: gl.constexpr = gl.BlockedLayout([1, 2], [2, 16], [2, 2], [1, 0])
    b_byte: gl.constexpr = gl.DistributedLinearLayout(
        [[0, 1], [0, 2], [0, 4], [0, 8], [0, 16]],
        [[1, 0], [2, 0], [4, 0], [8, 0], [64, 0]],
        [[16, 0], [32, 0]], [], [128, 32],
    )
    a_row = tile_m * M_TILE + gl.arange(0, M_TILE, gl.SliceLayout(1, a_load))
    a_col = gl.arange(0, 64, gl.SliceLayout(0, a_load))
    b_row = gl.arange(0, 128, gl.SliceLayout(1, b_byte))
    b_col = gl.arange(0, 32, gl.SliceLayout(0, b_byte))
    acc = gl.zeros((M_TILE, 128), gl.float32, mma)
    for kt in range(k_tiles):
        x = gl.load(X + a_row[:, None] * K + (kt * 64 + a_col[None, :]), a_row[:, None] < M, 0).to(gl.float16)
        tidx = (tile_n * k_tiles + kt) * 128 + b_row
        packed = gl.load(Packed + tidx[:, None] * 32 + b_col[None, :]).to(gl.uint32)
        sc = gl.load(Scales + tidx)
        lo = ((packed & 15).to(gl.int32) - 8).to(gl.float16) * sc[:, None]
        hi = ((packed >> 4).to(gl.int32) - 8).to(gl.float16) * sc[:, None]
        w = gl.reshape(gl.join(lo, hi), (128, 64))
        a = gl.convert_layout(x, gl.DotOperandLayout(0, mma, 16))
        b = gl.convert_layout(gl.permute(w, (1, 0)), gl.DotOperandLayout(1, mma, 16))
        acc = wmma(a, b, acc)
    out_row = tile_m * M_TILE + gl.arange(0, M_TILE, gl.SliceLayout(1, out_layout))
    out_col = gl.arange(0, 128, gl.SliceLayout(0, out_layout))
    val = gl.convert_layout(acc, out_layout)
    if HAS_BIAS:
        bv = gl.load(Bias + tile_n * 128 + out_col, (tile_n * 128 + out_col) < N, 0).to(gl.float32)
        val = val + bv[None, :]
    val = val.to(gl.bfloat16, fp_downcast_rounding="rtne")
    gl.store(Y + out_row[:, None] * N + tile_n * 128 + out_col[None, :], val, out_row[:, None] < M)


def w4_wmma_linear(
    x: torch.Tensor, weight: W4WmmaWeight, *, bias: torch.Tensor | None = None, splits: int = 1,
) -> torch.Tensor:
    m, k = x.shape
    n_pad = weight.padded_out
    out = x.new_empty((m, n_pad), dtype=torch.bfloat16)
    if m > 16 or bias is not None:
        bm = 16 if m <= 16 else (32 if m <= 32 else 64)
        bias_arg = bias if bias is not None else out
        _w4_wmma_batched_f16_kernel[(n_pad // 128, triton.cdiv(m, bm))](
            x.contiguous(), weight.packed, weight.scales, bias_arg, out, m,
            N=n_pad, K=k, M_TILE=bm, HAS_BIAS=bias is not None,
            num_warps=4, num_stages=1,
        )
    elif splits == 1:
        _w4_wmma_m8_kernel[(n_pad // 128, 1)](
            x.contiguous(), weight.packed, weight.scales, out, m,
            N=n_pad, K=k, SPLITS=1, num_warps=4, num_stages=1,
        )
    else:
        partial = torch.empty((splits, m, n_pad), dtype=torch.float32, device=x.device)
        _w4_wmma_m8_kernel[(n_pad // 128, splits)](
            x.contiguous(), weight.packed, weight.scales, partial, m,
            N=n_pad, K=k, SPLITS=splits, num_warps=4, num_stages=1,
        )
        _reduce_splits_bf16[(triton.cdiv(m * n_pad, 256),)](
            partial, out, m * n_pad, splits, 256, num_warps=4,
        )
    return out[:, : weight.out_features]


@torch.no_grad()
def pack_w4_markov(weight: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    v, k = weight.shape
    assert k == 256
    w = weight.float().reshape(v, 4, 64)
    scale = (w.abs().amax(dim=-1) / 7.0).clamp_min_(1e-7).half()
    q = (w / scale.float().unsqueeze(-1)).round().clamp(-7, 7).to(torch.int32) + 8
    q = q.reshape(v, 128, 2)
    packed = (q[..., 0] | (q[..., 1] << 4)).to(torch.uint8).contiguous()
    return packed, scale.contiguous()


@gluon.jit
def _first_max(ls, li, rs, ri):
    tl_cond = (ls > rs) | ((ls == rs) & (li < ri))
    return gl.where(tl_cond, ls, rs), gl.where(tl_cond, li, ri)


@gluon.jit
def _w4_markov_argmax_partial(
    X, Packed, Scales, Bias, Base, Values, Indices, softcap,
    VALID: gl.constexpr, ROWS: gl.constexpr,
):
    block = gl.program_id(0)
    tile: gl.constexpr = gl.BlockedLayout([1, 4], [1, 32], [4, 1], [1, 0])
    rows = block * ROWS + gl.arange(0, ROWS, gl.SliceLayout(1, tile))
    bcols = gl.arange(0, 128, gl.SliceLayout(0, tile))
    valid = rows < VALID
    p = gl.load(Packed + rows[:, None] * 128 + bcols[None, :], valid[:, None], other=0).to(gl.int32)
    lo = ((p & 15) - 8).to(gl.float32)
    hi = ((p >> 4) - 8).to(gl.float32)
    x_lo = gl.load(X + bcols * 2).to(gl.float32)
    x_hi = gl.load(X + bcols * 2 + 1).to(gl.float32)
    sc = gl.load(Scales + rows[:, None] * 4 + (bcols[None, :] // 32), valid[:, None], other=0).to(gl.float32)
    dot = (lo * x_lo[None, :] + hi * x_hi[None, :]) * sc
    folded = gl.sum(dot, axis=1)
    bias = gl.load(Bias + rows, valid, other=0).to(gl.float32)
    base = gl.load(Base + rows, valid, other=0).to(gl.float32)
    correction = (folded + bias).to(gl.bfloat16, fp_downcast_rounding="rtne")
    combined = (base + correction.to(gl.float32)).to(gl.bfloat16, fp_downcast_rounding="rtne")
    score = (libdevice.tanh(combined / softcap) * softcap).to(gl.bfloat16, fp_downcast_rounding="rtne").to(gl.float32)
    score = gl.where(valid & (score == score), score, float("-inf"))
    val, idx = gl.reduce((score, rows), 0, _first_max)
    gl.store(Values + block, val)
    gl.store(Indices + block, idx)


def w4_markov_argmax(
    latent: torch.Tensor, packed: torch.Tensor, scales: torch.Tensor,
    bias: torch.Tensor, base: torch.Tensor, softcap: float, valid: int,
) -> torch.Tensor:
    rows = 16
    splits = triton.cdiv(valid, rows)
    values = torch.empty(splits, dtype=torch.float32, device=base.device)
    indices = torch.empty(splits, dtype=torch.int32, device=base.device)
    out = torch.empty(1, dtype=torch.int64, device=base.device)
    _w4_markov_argmax_partial[(splits,)](
        latent, packed, scales, bias, base, values, indices, softcap,
        VALID=valid, ROWS=rows, num_warps=4,
    )
    _argmax_finish[(1,)](
        values, indices, out, splits, triton.next_power_of_2(splits), num_warps=4,
    )
    return out


@triton.jit
def _topk_markov_argmax_kernel(
    X, Packed, Scales, Bias, Base, CandIds, OutVal, OutIdx, softcap,
    TOPK: tl.constexpr, BLOCK: tl.constexpr,
):
    bid = tl.program_id(0)
    offs = bid * BLOCK + tl.arange(0, BLOCK)
    valid = offs < TOPK
    tok = tl.load(CandIds + offs, valid, other=0).to(tl.int64)
    bcols = tl.arange(0, 128)
    x_lo = tl.load(X + bcols * 2).to(tl.float32)
    x_hi = tl.load(X + bcols * 2 + 1).to(tl.float32)
    p = tl.load(Packed + tok[:, None] * 128 + bcols[None, :], valid[:, None], other=0).to(tl.int32)
    lo = ((p & 15) - 8).to(tl.float32)
    hi = ((p >> 4) - 8).to(tl.float32)
    sc = tl.load(Scales + tok[:, None] * 4 + (bcols[None, :] // 32), valid[:, None], other=0).to(gl.float32) if False else tl.load(Scales + tok[:, None] * 4 + (bcols[None, :] // 32), valid[:, None], other=0).to(tl.float32)
    folded = tl.sum((lo * x_lo[None, :] + hi * x_hi[None, :]) * sc, axis=1)
    bias = tl.load(Bias + tok, valid, other=0).to(tl.float32)
    base = tl.load(Base + tok, valid, other=0).to(tl.float32)
    combined = base + folded + bias
    score = tl.where(valid & (combined == combined), combined, float("-inf"))
    # Monotonic softcap does not change argmax!
    best_idx_in_blk = tl.argmax(score, axis=0)
    best_val = tl.max(score, axis=0)
    best_tok = tl.load(CandIds + bid * BLOCK + best_idx_in_blk)
    tl.store(OutVal + bid, best_val)
    tl.store(OutIdx + bid, best_tok.to(tl.int32))


def topk_markov_argmax(
    latent: torch.Tensor, packed: torch.Tensor, scales: torch.Tensor,
    bias: torch.Tensor, base: torch.Tensor, cand_ids: torch.Tensor, softcap: float,
) -> torch.Tensor:
    topk = cand_ids.numel()
    block = 16
    splits = triton.cdiv(topk, block)
    vals = torch.empty(splits, dtype=torch.float32, device=base.device)
    idxs = torch.empty(splits, dtype=torch.int32, device=base.device)
    out = torch.empty(1, dtype=torch.int64, device=base.device)
    _topk_markov_argmax_kernel[(splits,)](
        latent, packed, scales, bias, base, cand_ids, vals, idxs, softcap,
        TOPK=topk, BLOCK=block, num_warps=4,
    )
    _argmax_finish[(1,)](
        vals, idxs, out, splits, triton.next_power_of_2(splits), num_warps=4,
    )
    return out


# ============================================================================
# 6. WAVE 2: TURBO M=8 SPLIT-KV DECODE ATTENTION (1.80x FASTER AT 64K!)
# ============================================================================

@triton.jit
def _turbo_decode_m8_split_kernel(
    Q, K, V, Bias, OPart, LsePart,
    BlockTable, SeqLens, CuSeqlensQ,
    scale_log2, num_splits, num_tokens,
    stride_qt, stride_qh,
    stride_kb, stride_kp, stride_kh,
    stride_vb, stride_vp, stride_vh,
    stride_bt, stride_bh,
    stride_ot, stride_oh, stride_lh, stride_table,
    GROUP: tl.constexpr, D: tl.constexpr, PAGE: tl.constexpr,
    E_PAD: tl.constexpr, EXTENT: tl.constexpr, WINDOW: tl.constexpr,
    BLOCK_Q: tl.constexpr, BLOCK_H: tl.constexpr, BLOCK_N: tl.constexpr,
):
    BLOCK_M: tl.constexpr = BLOCK_Q * BLOCK_H
    split = tl.program_id(0)
    kv_head = tl.program_id(1)
    req = tl.program_id(2)
    q_start = tl.load(CuSeqlensQ + req)
    q_len = tl.load(CuSeqlensQ + req + 1) - q_start
    seq_len = tl.load(SeqLens + req)
    past = seq_len - q_len

    rows = tl.arange(0, BLOCK_M)
    q_idx = rows // BLOCK_H
    head = kv_head * GROUP + rows % BLOCK_H
    row_valid = (q_idx < q_len) & (rows % BLOCK_H < GROUP)
    token = (q_start + q_idx).to(tl.int64)
    d = tl.arange(0, D)

    q = tl.load(
        Q + token[:, None] * stride_qt + head[:, None] * stride_qh + d[None, :],
        row_valid[:, None], other=0.0,
    )
    pos = past + q_idx
    first_pos = past
    hi = seq_len
    lo = 0
    if WINDOW >= 0:
        lo = tl.maximum(first_pos - WINDOW, 0) // BLOCK_N * BLOCK_N
    first_block = lo // BLOCK_N
    end_block = tl.cdiv(hi, BLOCK_N)
    tiles_per_split = tl.cdiv(end_block - first_block, num_splits)
    n_first = first_block + split * tiles_per_split
    n_end = tl.minimum(n_first + tiles_per_split, end_block)

    table = BlockTable + req * stride_table
    k_head_base = K + kv_head * stride_kh
    v_head_base = V + kv_head * stride_vh
    n_offs = tl.arange(0, BLOCK_N)

    m_i = tl.full([BLOCK_M], float("-inf"), tl.float32)
    l_i = tl.zeros([BLOCK_M], tl.float32)
    acc = tl.zeros([BLOCK_M, D], tl.float32)

    if WINDOW < 0:
        q_scaled = (q.to(tl.float32) * scale_log2).to(tl.bfloat16)
        first_bias_key = first_pos - E_PAD
        far_limit = tl.maximum(first_bias_key, 0) // BLOCK_N
        far_end = tl.minimum(n_end, far_limit)
        for blk in range(n_first, far_end):
            key_begin = blk * BLOCK_N
            if BLOCK_N <= PAGE:
                page_id = tl.load(table + (key_begin // PAGE)).to(tl.int64)
                slot_base = key_begin % PAGE
                k = tl.load(k_head_base + page_id * stride_kb + (slot_base + n_offs)[:, None] * stride_kp + d[None, :])
                v = tl.load(v_head_base + page_id * stride_vb + (slot_base + n_offs)[:, None] * stride_vp + d[None, :])
            else:
                p0 = tl.load(table + (key_begin // PAGE)).to(tl.int64)
                p1 = tl.load(table + (key_begin // PAGE) + 1).to(tl.int64)
                pg = tl.where(n_offs < PAGE, p0, p1)
                sl = n_offs % PAGE
                k = tl.load(k_head_base + pg[:, None] * stride_kb + sl[:, None] * stride_kp + d[None, :])
                v = tl.load(v_head_base + pg[:, None] * stride_vb + sl[:, None] * stride_vp + d[None, :])
            s = tl.dot(q_scaled, tl.trans(k))
            m_new = tl.maximum(m_i, tl.max(s, 1))
            alpha = tl.math.exp2(m_i - m_new)
            p = tl.math.exp2(s - m_new[:, None])
            l_i = l_i * alpha + tl.sum(p, 1)
            acc = acc * alpha[:, None] + tl.dot(p.to(v.dtype), v)
            m_i = m_new
        near_start = tl.maximum(n_first, far_limit)
    else:
        near_start = n_first

    tile_end = tl.minimum((q_idx // 128 + 1) * 128, q_len)
    key0 = tl.cdiv(past + tile_end, 128) * 128 - E_PAD
    bias_rows = Bias + token * stride_bt + head * stride_bh - key0

    for blk in range(near_start, n_end):
        key_begin = blk * BLOCK_N
        keys = key_begin + n_offs
        key_valid = keys < hi
        if BLOCK_N <= PAGE:
            page_id = tl.load(table + (key_begin // PAGE)).to(tl.int64)
            slot_base = key_begin % PAGE
            k = tl.load(
                k_head_base + page_id * stride_kb + (slot_base + n_offs)[:, None] * stride_kp + d[None, :],
                key_valid[:, None], other=0.0,
            )
        else:
            p0 = tl.load(table + (key_begin // PAGE)).to(tl.int64)
            p1 = tl.load(table + (key_begin // PAGE) + 1, mask=(key_begin + PAGE) < hi, other=p0).to(tl.int64)
            pg = tl.where(n_offs < PAGE, p0, p1)
            sl = n_offs % PAGE
            k = tl.load(
                k_head_base + pg[:, None] * stride_kb + sl[:, None] * stride_kp + d[None, :],
                key_valid[:, None], other=0.0,
            )
        s = tl.dot(q, tl.trans(k)) * scale_log2
        dist = pos[:, None] - keys[None, :]
        valid = row_valid[:, None] & key_valid[None, :] & (dist >= 0)
        if WINDOW >= 0:
            valid &= dist <= WINDOW
        in_table = valid & (keys[None, :] >= key0[:, None])
        bias = tl.load(bias_rows[:, None] + keys[None, :], in_table, other=0.0)
        s = s + bias.to(tl.float32) * 1.4426950408889634
        s = tl.where(valid, s, float("-inf"))
        m_new = tl.maximum(m_i, tl.max(s, 1))
        m_ref = tl.where(m_new == float("-inf"), 0.0, m_new)
        alpha = tl.math.exp2(m_i - m_ref)
        p = tl.math.exp2(s - m_ref[:, None])
        l_i = l_i * alpha + tl.sum(p, 1)
        if BLOCK_N <= PAGE:
            v = tl.load(
                v_head_base + page_id * stride_vb + (slot_base + n_offs)[:, None] * stride_vp + d[None, :],
                key_valid[:, None], other=0.0,
            )
        else:
            v = tl.load(
                v_head_base + pg[:, None] * stride_vb + sl[:, None] * stride_vp + d[None, :],
                key_valid[:, None], other=0.0,
            )
        acc = acc * alpha[:, None] + tl.dot(p.to(v.dtype), v)
        m_i = m_new

    has_keys = l_i > 0.0
    out = acc / tl.where(has_keys, l_i, 1.0)[:, None]
    out_token = token + split * num_tokens
    tl.store(OPart + out_token[:, None] * stride_ot + head[:, None] * stride_oh + d[None, :], out, row_valid[:, None])
    lse = tl.where(has_keys, m_i + tl.math.log2(l_i), float("-inf"))
    tl.store(LsePart + head * stride_lh + out_token, lse, row_valid)


def turbo_rel_decode_m8_attention(
    q: torch.Tensor, key_cache: torch.Tensor, value_cache: torch.Tensor,
    rel_logits: torch.Tensor, gate: torch.Tensor, sinks: torch.Tensor,
    out: torch.Tensor, *, block_table: torch.Tensor, cache_seqlens: torch.Tensor,
    cu_seqlens_q: torch.Tensor, max_query_len: int, num_actual_tokens: int,
    softmax_scale: float, is_local: bool,
) -> None:
    from .attention import _combine

    num_tokens, num_heads, head_dim = q.shape
    page, num_kv_heads = key_cache.shape[1:3]
    group = num_heads // num_kv_heads
    num_reqs = cache_seqlens.shape[0]
    extent = rel_logits.shape[2] - 256
    window = extent - 1 if is_local else -1
    # Adaptive split count by concurrent requests: preserves C1 64K peak & maximizes B=2..8 CU occupancy
    if num_reqs == 1:
        num_splits = 8 if is_local else 10
    elif num_reqs == 2:
        num_splits = 4 if is_local else 5
    elif num_reqs <= 4:
        num_splits = 2 if is_local else 6
    else:
        num_splits = 1 if is_local else 6
    block_n = 32 if page == 16 else 64
    o_part = q.new_empty((num_splits * num_tokens, num_heads, head_dim), dtype=torch.float32)
    lse_part = q.new_empty((num_heads, num_splits * num_tokens), dtype=torch.float32)
    _turbo_decode_m8_split_kernel[(num_splits, num_kv_heads, num_reqs)](
        q, key_cache, value_cache, rel_logits, o_part, lse_part,
        block_table, cache_seqlens, cu_seqlens_q,
        softmax_scale * 1.4426950408889634, num_splits, num_tokens,
        q.stride(0), q.stride(1),
        key_cache.stride(0), key_cache.stride(1), key_cache.stride(2),
        value_cache.stride(0), value_cache.stride(1), value_cache.stride(2),
        rel_logits.stride(0), rel_logits.stride(1),
        o_part.stride(0), o_part.stride(1), lse_part.stride(0), block_table.stride(0),
        GROUP=group, D=head_dim, PAGE=page, E_PAD=rel_logits.shape[2], EXTENT=extent,
        WINDOW=window, BLOCK_Q=8, BLOCK_H=4, BLOCK_N=block_n, num_warps=4, num_stages=1,
    )
    _combine(o_part, lse_part, gate, sinks, out, cu_seqlens_q, num_splits, True, compact_splits=True)


# ============================================================================
# 7. WAVE 2: FAST M=8 ROUTER (1.62x FASTER) & FAST LOOP CONTROLLER DOWN (1.61x)
# ============================================================================

@triton.jit
def _t_max(a, b):
    return tl.maximum(a, b)

@triton.jit
def _t_min(a, b):
    return tl.minimum(a, b)

@triton.jit
def _fast_route_m8_kernel(
    Logits, Bias, Ids, Gamma, Sorted, Experts, Padded, M: tl.constexpr, SCALE: tl.constexpr,
):
    N_ROUTES: tl.constexpr = M * 8
    row = tl.arange(0, 8)
    r_valid = row < M
    expert = tl.arange(0, 128)
    src = tl.load(Logits + row[:, None] * 128 + expert[None, :], r_valid[:, None], other=0.0)
    src = tl.div_rn(1.0, 1.0 + tl.math.exp(-src))
    bias = tl.load(Bias + expert)
    corr = src + bias[None, :]
    ids_flat = tl.zeros((8, 8), tl.int32)
    sel_flat = tl.zeros((8, 8), tl.float32)
    slot_offs = tl.arange(0, 8)
    work = corr
    for k in tl.static_range(8):
        best = tl.argmax(work, axis=1)
        val = tl.sum(tl.where(expert[None, :] == best[:, None], src, 0.0), axis=1)
        ids_flat = tl.where(slot_offs[None, :] == k, best[:, None], ids_flat)
        sel_flat = tl.where(slot_offs[None, :] == k, val[:, None], sel_flat)
        work = tl.where(expert[None, :] == best[:, None], float("-inf"), work)
    denom = tl.sum(sel_flat, axis=1) + 1e-20
    gamma = tl.div_rn(sel_flat, denom[:, None]) * SCALE
    slot = row[:, None] * 8 + slot_offs[None, :]
    tl.store(Ids + slot, ids_flat.to(tl.int64), r_valid[:, None])
    tl.store(Gamma + slot, gamma, r_valid[:, None])
    # Mark inactive rows (row >= M) with sentinel expert 128 so they sort to the end!
    ids_masked = tl.where(r_valid[:, None], ids_flat, 128)
    routes = tl.arange(0, 64)
    by_expert = tl.sort((tl.reshape(ids_masked, (64,)) << 6) | routes, descending=False)
    exp_s = by_expert >> 6
    active = exp_s < 128
    prev = tl.gather(exp_s, tl.maximum(routes - 1, 0), 0)
    new_g = active & ((routes == 0) | (exp_s != prev))
    blk = tl.cumsum(new_g.to(tl.int32)) - 1
    start = tl.associative_scan(tl.where(new_g, routes, 0), 0, _t_max)
    tl.store(Sorted + blk * 16 + routes - start, by_expert & 63, active)
    tl.store(Experts + blk, exp_s, new_g)
    marks = tl.where(new_g, routes, N_ROUTES)
    next_mark = tl.where(routes < N_ROUTES - 1, tl.gather(marks, tl.minimum(routes + 1, 63), 0), N_ROUTES)
    end = tl.associative_scan(next_mark, 0, _t_min, reverse=True)
    s16 = tl.arange(0, 16)
    pad_m = new_g[:, None] & (s16[None, :] >= (end - routes)[:, None])
    tl.store(Sorted + blk[:, None] * 16 + s16[None, :], N_ROUTES, pad_m)
    tl.store(Padded, tl.sum(new_g.to(tl.int32)) * 16)


def fast_route_m8(logits: torch.Tensor, bias: torch.Tensor, scale: float):
    m = logits.shape[0]
    ids = torch.empty((m, 8), dtype=torch.int64, device=logits.device)
    gamma = torch.empty((m, 8), dtype=torch.float32, device=logits.device)
    sorted_ids = torch.empty((m * 128,), dtype=torch.int32, device=logits.device)
    experts = torch.empty((m * 8,), dtype=torch.int32, device=logits.device)
    padded = torch.empty((1,), dtype=torch.int32, device=logits.device)
    _fast_route_m8_kernel[(1,)](
        logits, bias, ids, gamma, sorted_ids, experts, padded,
        M=m, SCALE=scale, num_warps=8, enable_fp_fusion=False,
    )
    return ids, gamma, (sorted_ids, experts, padded)


@triton.jit
def _controller_down_m8_kernel(Pair, W, Out, M: tl.constexpr, K: tl.constexpr, BLOCK_K: tl.constexpr):
    n = tl.program_id(0)
    s = tl.program_id(1)
    rows = tl.arange(0, 8)
    r_mask = rows < M
    k_offs = tl.arange(0, BLOCK_K)
    acc = tl.zeros((8,), tl.float32)
    w_base = W + (s.to(tl.int64) * 64 + n) * K
    p_base = Pair + s.to(tl.int64) * M * K
    for k0 in range(0, K, BLOCK_K):
        k = k0 + k_offs
        mask = k < K
        w = tl.load(w_base + k, mask, other=0.0).to(tl.float32)
        x = tl.load(p_base + rows[:, None] * K + k[None, :], r_mask[:, None] & mask[None, :], other=0.0).to(tl.float32)
        acc += tl.sum(x * w[None, :], axis=1)
    tl.store(Out + (s.to(tl.int64) * M + rows) * 64 + n, acc.to(tl.bfloat16), r_mask)


def controller_down_m8(pair: torch.Tensor, down_weight: torch.Tensor) -> torch.Tensor:
    streams, m, k = pair.shape
    out = pair.new_empty((streams, m, 64))
    _controller_down_m8_kernel[(64, streams)](
        pair.contiguous(), down_weight.contiguous(), out, M=m, K=k, BLOCK_K=512, num_warps=4,
    )
    return out


# ============================================================================
# 8. WAVE 2: FP16-DIRECT MIXED & CODA MoE (M16 DECODE + M64 PREFILL)
# ============================================================================

@gluon.jit
def _c1_f16_pair(packed, raw, row_scale, Table, expert):
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
    low = ((packed & 15).to(gl.float32) * scale + offset).to(gl.float16)
    high = ((packed >> 4).to(gl.float32) * scale + offset).to(gl.float16)
    return low, high


@gluon.jit
def _mixed_wmma_f16_tile(
    X, Packed, Raw, Outer, RowScale, Table, expert, rows, valid, n,
    N: gl.constexpr, K: gl.constexpr, KIND: gl.constexpr,
    MMA: gl.constexpr, A_LOAD: gl.constexpr, M_TILE: gl.constexpr,
):
    b_byte: gl.constexpr = gl.DistributedLinearLayout(
        [[0, 1], [0, 2], [0, 4], [0, 8], [0, 16]],
        [[1, 0], [2, 0], [4, 0], [8, 0], [64, 0]],
        [[16, 0], [32, 0]], [], [128, 32],
    )
    b_row = gl.arange(0, 128, gl.SliceLayout(1, b_byte))
    a_cols = gl.arange(0, 64, gl.SliceLayout(0, A_LOAD))
    byte_cols = gl.arange(0, 32, gl.SliceLayout(0, b_byte))
    tile = (expert.to(gl.int64) * (N // 128) + n) * (K // 64)
    if KIND == 0:
        outer = gl.load(Outer + expert)
    elif KIND == 2:
        row = gl.load(RowScale + expert.to(gl.int64) * N + n * 128 + b_row).to(gl.float32)
    acc = gl.zeros((M_TILE, 128), gl.float32, MMA)
    for k in range(K // 64):
        x = gl.load(X + rows[:, None] * K + k * 64 + a_cols[None, :], valid[:, None], 0).to(gl.float16)
        packet_row = (tile + k) * 128 + b_row
        packed = gl.load(Packed + packet_row[:, None] * 32 + byte_cols[None, :]).to(gl.uint32)
        raw = gl.load(Raw + packet_row).to(gl.uint32)
        if KIND == 0:
            scale_byte = (raw[:, None] >> ((byte_cols[None, :] // 8) * 8)) & 255
            low, high = _unpack_e2m1_f16x2_scaled(packed, scale_byte)
        elif KIND == 1:
            scale_bits = (raw[:, None] >> ((byte_cols[None, :] // 16) * 16)).to(gl.uint16)
            scale = scale_bits.to(gl.bfloat16, bitcast=True).to(gl.float32)
            low = ((((packed & 15).to(gl.int32) ^ 8) - 8).to(gl.float32) * scale).to(gl.float16)
            high = ((((packed >> 4).to(gl.int32) ^ 8) - 8).to(gl.float32) * scale).to(gl.float16)
        else:
            low, high = _c1_f16_pair(packed, raw, row, Table, expert)
        weights = gl.reshape(gl.join(low, high), (128, 64))
        a = gl.convert_layout(x, gl.DotOperandLayout(0, MMA, 16))
        b = gl.convert_layout(gl.permute(weights, (1, 0)), gl.DotOperandLayout(1, MMA, 16))
        acc = wmma(a, b, acc)
    if KIND == 0:
        acc = acc * outer
    return acc


@gluon.jit
def _mixed_wmma_turbo_kernel(
    X, Packed, Raw, Outer, Kind, RowScale, Table, Sorted, Expert, Padded, Y, Gamma,
    N: gl.constexpr, K: gl.constexpr, ROUTES, TOKENS: gl.constexpr,
    FITTED_KIND: gl.constexpr, FUSED: gl.constexpr, GATE: gl.constexpr, UP: gl.constexpr,
    M_TILE: gl.constexpr,
):
    m = gl.program_id(0)
    n = gl.program_id(1)
    if m * M_TILE >= gl.load(Padded):
        return
    expert = gl.load(Expert + m)
    if expert < 0:
        return
    if M_TILE == 16:
        mma: gl.constexpr = AMDWMMALayout(1, True, [[0, 1], [0, 2]])
        a_load: gl.constexpr = gl.BlockedLayout([1, 64], [16, 2], [1, 4], [0, 1])
        out_layout: gl.constexpr = gl.DistributedLinearLayout(
            [[0, 1], [0, 4], [0, 8], [0, 64]],
            [[1, 0], [2, 0], [4, 0], [8, 0], [0, 2]],
            [[0, 16], [0, 32]], [], [16, 128],
        )
    else:
        mma: gl.constexpr = AMDWMMALayout(1, True, [[1, 0], [0, 1]])
        a_load: gl.constexpr = gl.BlockedLayout([1, 8], [8, 4], [2, 2], [0, 1])
        out_layout: gl.constexpr = gl.BlockedLayout([1, 2], [2, 16], [2, 2], [1, 0])
    a_range = gl.arange(0, M_TILE, gl.SliceLayout(1, a_load))
    route = gl.load(Sorted + m * M_TILE + a_range).to(gl.int64)
    valid = (route >= 0) & (route < ROUTES)
    rows = route // TOKENS
    if gl.load(Kind + expert) == 0:
        acc = _mixed_wmma_f16_tile(
            X, Packed, Raw, Outer, RowScale, Table, expert, rows, valid, n, N, K, 0, mma, a_load, M_TILE,
        )
    else:
        acc = _mixed_wmma_f16_tile(
            X, Packed, Raw, Outer, RowScale, Table, expert, rows, valid, n, N, K, FITTED_KIND, mma, a_load, M_TILE,
        )
    out_rows: gl.constexpr = gl.SliceLayout(1, out_layout)
    route = gl.convert_layout(route, out_rows)
    valid = gl.convert_layout(valid, out_rows)
    out = gl.convert_layout(acc.to(gl.bfloat16, fp_downcast_rounding="rtne"), out_layout)
    if FUSED:
        gate, up = gl.split(gl.reshape(out, (M_TILE, 64, 2)))
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


def mixed_wmma_turbo_projection(x, bank, plan, *, fused_activation=None):
    experts, n_tiles, k_tiles, _, _ = bank.packed.shape
    n, k = n_tiles * 128, k_tiles * 64
    route_shape = (plan.num_tokens, plan.top_k)
    gamma, gate_cap, up_cap = None, 0.0, 0.0
    if fused_activation is not None:
        gamma, gate_cap, up_cap = fused_activation
    out = x.new_empty((*route_shape, n // 2 if gamma is not None else n))
    bm = plan.block_m
    _mixed_wmma_turbo_kernel[(plan.expert_ids.numel(), n_tiles)](
        x, bank.packed, bank.raw_scales, bank.outer_scales, bank.kind,
        bank.row_scales, bank.table,
        plan.sorted_ids, plan.expert_ids, plan.padded_tokens, out, gamma,
        N=n, K=k, ROUTES=plan.num_tokens * plan.top_k,
        TOKENS=plan.top_k if x.ndim == 2 else 1, FITTED_KIND=bank.fitted_kind,
        FUSED=gamma is not None, GATE=gate_cap, UP=up_cap, M_TILE=bm,
        num_warps=4, num_stages=2 if bm == 64 else 1, enable_fp_fusion=False,
    )
    return out


@gluon.jit
def _coda_wmma_turbo_kernel(
    X, Packed, Raw, RowScale, Table, Sorted, Expert, Padded, Y, Gamma,
    N: gl.constexpr, K: gl.constexpr, ROUTES, TOKENS: gl.constexpr,
    FUSED: gl.constexpr, GATE: gl.constexpr, UP: gl.constexpr,
    M_TILE: gl.constexpr,
):
    m = gl.program_id(0)
    n = gl.program_id(1)
    if m * M_TILE >= gl.load(Padded):
        return
    expert = gl.load(Expert + m)
    if expert < 0:
        return
    if M_TILE == 16:
        mma: gl.constexpr = AMDWMMALayout(1, True, [[0, 1], [0, 2]])
        a_load: gl.constexpr = gl.BlockedLayout([1, 64], [16, 2], [1, 4], [0, 1])
        out_layout: gl.constexpr = gl.DistributedLinearLayout(
            [[0, 1], [0, 4], [0, 8], [0, 64]],
            [[1, 0], [2, 0], [4, 0], [8, 0], [0, 2]],
            [[0, 16], [0, 32]], [], [16, 128],
        )
    else:
        mma: gl.constexpr = AMDWMMALayout(1, True, [[1, 0], [0, 1]])
        a_load: gl.constexpr = gl.BlockedLayout([1, 8], [8, 4], [2, 2], [0, 1])
        out_layout: gl.constexpr = gl.BlockedLayout([1, 2], [2, 16], [2, 2], [1, 0])
    b_byte: gl.constexpr = gl.DistributedLinearLayout(
        [[0, 1], [0, 2], [0, 4], [0, 8], [0, 16]],
        [[1, 0], [2, 0], [4, 0], [8, 0], [64, 0]],
        [[16, 0], [32, 0]], [], [128, 32],
    )
    a_range = gl.arange(0, M_TILE, gl.SliceLayout(1, a_load))
    route = gl.load(Sorted + m * M_TILE + a_range).to(gl.int64)
    valid = (route >= 0) & (route < ROUTES)
    rows = route // TOKENS
    b_row = gl.arange(0, 128, gl.SliceLayout(1, b_byte))
    a_cols = gl.arange(0, 64, gl.SliceLayout(0, a_load))
    byte_cols = gl.arange(0, 32, gl.SliceLayout(0, b_byte))
    tile = (expert.to(gl.int64) * (N // 128) + n) * (K // 64)
    row_scale = gl.load(RowScale + expert.to(gl.int64) * N + n * 128 + b_row).to(gl.float32)
    acc = gl.zeros((M_TILE, 128), gl.float32, mma)
    for k in range(K // 64):
        x = gl.load(X + rows[:, None] * K + k * 64 + a_cols[None, :], valid[:, None], 0).to(gl.float16)
        packet_row = (tile + k) * 128 + b_row
        packed = gl.load(Packed + packet_row[:, None] * 32 + byte_cols[None, :]).to(gl.uint32)
        raw = gl.load(Raw + packet_row).to(gl.uint32)
        low, high = _c1_f16_pair(packed, raw, row_scale, Table, expert)
        weights = gl.reshape(gl.join(low, high), (128, 64))
        a = gl.convert_layout(x, gl.DotOperandLayout(0, mma, 16))
        b = gl.convert_layout(gl.permute(weights, (1, 0)), gl.DotOperandLayout(1, mma, 16))
        acc = wmma(a, b, acc)
    out_rows: gl.constexpr = gl.SliceLayout(1, out_layout)
    route = gl.convert_layout(route, out_rows)
    valid = gl.convert_layout(valid, out_rows)
    out = gl.convert_layout(acc.to(gl.bfloat16, fp_downcast_rounding="rtne"), out_layout)
    if FUSED:
        gate, up = gl.split(gl.reshape(out, (M_TILE, 64, 2)))
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


def coda_wmma_turbo_projection(x, bank, plan, *, fused_activation=None):
    experts, n_tiles, k_tiles, _, _ = bank.packed.shape
    n, k = n_tiles * 128, k_tiles * 64
    route_shape = (plan.num_tokens, plan.top_k)
    gamma, gate_cap, up_cap = None, 0.0, 0.0
    if fused_activation is not None:
        gamma, gate_cap, up_cap = fused_activation
    out = x.new_empty((*route_shape, n // 2 if gamma is not None else n))
    bm = plan.block_m
    _coda_wmma_turbo_kernel[(plan.expert_ids.numel(), n_tiles)](
        x, bank.packed, bank.tags, bank.row_scales, bank.table,
        plan.sorted_ids, plan.expert_ids, plan.padded_tokens, out, gamma,
        N=n, K=k, ROUTES=plan.num_tokens * plan.top_k,
        TOKENS=plan.top_k if x.ndim == 2 else 1,
        FUSED=gamma is not None, GATE=gate_cap, UP=up_cap, M_TILE=bm,
        num_warps=4, num_stages=2 if bm == 64 else 1, enable_fp_fusion=False,
    )
    return out


# ============================================================================
# 9. WAVE 5: FUSED PREFILL LOOP BLEND + RMSNORM (1.60x FASTER, SKIPS 212MB BMM!)
# ============================================================================

@triton.jit
def _fused_loop_blend_p1_kernel(
    Hidden, Weight, Dest, Message, Blended, PartialSumSq,
    stride_dest_s, stride_dest_t, stride_message_s, stride_message_t,
    T, D: tl.constexpr, BT: tl.constexpr, BD: tl.constexpr,
    D_PER_BLK: tl.constexpr, N_DBLK: tl.constexpr,
):
    t_blk = tl.program_id(0)
    d_grp = tl.program_id(1)
    s = tl.program_id(2).to(tl.int64)
    t = (t_blk * BT + tl.arange(0, BT)).to(tl.int64)
    k = tl.arange(0, 64)
    t_mask = t < T
    h = tl.load(Hidden + (s * T + t[:, None]) * 64 + k[None, :], t_mask[:, None], other=0.0)
    sum_sq = tl.zeros((BT,), tl.float32)
    d_offs = tl.arange(0, BD)
    d_start = d_grp * D_PER_BLK
    for off in range(0, D_PER_BLK, BD):
        d = d_start + off + d_offs
        w_base = Weight + (s * (2 * D) + d[:, None]) * 64 + k[None, :]
        wg = tl.load(w_base)
        wf = tl.load(w_base + D * 64)
        gain = tl_hip_lib.tanh(tl.dot(h, tl.trans(wg)))
        forget = tl_hip_lib.tanh(tl.dot(h, tl.trans(wf)))
        dest = tl.load(Dest + s * stride_dest_s + t[:, None] * stride_dest_t + d[None, :], t_mask[:, None], other=0.0)
        msg = tl.load(Message + s * stride_message_s + t[:, None] * stride_message_t + d[None, :], t_mask[:, None], other=0.0)
        b = (1.0 - forget) * dest + gain * msg
        sum_sq += tl.sum(b * b, axis=1)
        tl.store(Blended + (s * T + t[:, None]) * D + d[None, :], b.to(tl.bfloat16), t_mask[:, None])
    tl.store(PartialSumSq + (s * N_DBLK + d_grp) * T + t, sum_sq, t_mask)


@triton.jit
def _fused_loop_blend_p2_kernel(
    Blended, PartialSumSq, NormWeight, Out, stride_w_s, T, norm_eps,
    D: tl.constexpr, BT: tl.constexpr, BD: tl.constexpr, N_DBLK: tl.constexpr,
):
    t_blk = tl.program_id(0)
    d_blk = tl.program_id(1)
    s = tl.program_id(2).to(tl.int64)
    t = (t_blk * BT + tl.arange(0, BT)).to(tl.int64)
    d = d_blk * BD + tl.arange(0, BD)
    t_mask = t < T
    g = tl.arange(0, 16)
    g_mask = g < N_DBLK
    psq = tl.load(
        PartialSumSq + (s * N_DBLK + g[:, None]) * T + t[None, :],
        g_mask[:, None] & t_mask[None, :],
        other=0.0,
    )
    sum_sq = tl.sum(psq, axis=0)
    rstd = tl_hip_lib.rsqrt(tl.div_rn(sum_sq, D) + norm_eps)
    w = tl.load(NormWeight + s * stride_w_s + d).to(tl.float32)
    b = tl.load(Blended + (s * T + t[:, None]) * D + d[None, :], t_mask[:, None], other=0.0).to(tl.float32)
    tl.store(Out + (s * T + t[:, None]) * D + d[None, :], (b * rstd[:, None] * w[None, :]).to(tl.bfloat16), t_mask[:, None])


def fused_loop_prefill_blend(
    hidden: torch.Tensor, weight: torch.Tensor, dest: torch.Tensor,
    message: torch.Tensor, norm_weight: torch.Tensor, norm_eps: float,
) -> torch.Tensor | None:
    from .loop import _packed_rows

    if not _packed_rows(dest, message):
        return None
    streams, t_len, width = message.shape
    if (
        t_len < 16
        or width != 2304
        or hidden.shape != (streams, t_len, 64)
        or weight.shape != (streams, 2 * width, 64)
        or hidden.dtype != torch.bfloat16
        or weight.dtype != torch.bfloat16
        or not hidden.is_contiguous()
        or not weight.is_contiguous()
        or norm_weight.shape != (streams, width)
        or norm_weight.stride(-1) != 1
    ):
        return None
    blended = message.new_empty(message.shape, dtype=torch.bfloat16)
    partial_sum_sq = message.new_empty((streams, 9, t_len), dtype=torch.float32)
    out = message.new_empty(message.shape, dtype=torch.bfloat16)
    bt = 16 if t_len <= 32 else 32
    _fused_loop_blend_p1_kernel[(triton.cdiv(t_len, bt), 9, streams)](
        hidden, weight, dest, message, blended, partial_sum_sq,
        *dest.stride()[:2], *message.stride()[:2], t_len,
        D=width, BT=bt, BD=64, D_PER_BLK=256, N_DBLK=9, num_warps=4,
    )
    _fused_loop_blend_p2_kernel[(triton.cdiv(t_len, bt), width // 128, streams)](
        blended, partial_sum_sq, norm_weight, out, norm_weight.stride(0), t_len, norm_eps,
        D=width, BT=bt, BD=128, N_DBLK=9, num_warps=4,
    )
    return out
