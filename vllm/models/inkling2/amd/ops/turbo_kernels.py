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
from .norm import _l2_clip
from .sconv import _conv
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
    if M_TILE == 64:
        n = gl.program_id(0)
        m = gl.program_id(1)
    else:
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
    grid = (n_tiles, plan.expert_ids.numel()) if bm == 64 else (plan.expert_ids.numel(), n_tiles)
    _source_wmma_turbo_kernel[grid](
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
def _mixed_split_kind_m64_kernel(
    X, Packed, Raw, Outer, Kind, RowScale, Table, Sorted, Expert, Padded, Y, Gamma,
    N: gl.constexpr, K: gl.constexpr, ROUTES, TOKENS: gl.constexpr,
    TARGET_KIND: gl.constexpr, FUSED: gl.constexpr, GATE: gl.constexpr, UP: gl.constexpr,
):
    n = gl.program_id(0)
    m = gl.program_id(1)
    if m * 64 >= gl.load(Padded):
        return
    expert = gl.load(Expert + m)
    if expert < 0:
        return
    if gl.load(Kind + expert) != TARGET_KIND:
        return
    mma: gl.constexpr = AMDWMMALayout(1, True, [[1, 0], [0, 1]])
    a_load: gl.constexpr = gl.BlockedLayout([1, 8], [8, 4], [2, 2], [0, 1])
    out_layout: gl.constexpr = gl.BlockedLayout([1, 2], [2, 16], [2, 2], [1, 0])
    b_byte: gl.constexpr = gl.DistributedLinearLayout(
        [[0, 1], [0, 2], [0, 4], [0, 8], [0, 16]],
        [[1, 0], [2, 0], [4, 0], [8, 0], [64, 0]],
        [[16, 0], [32, 0]], [], [128, 32],
    )
    a_range = gl.arange(0, 64, gl.SliceLayout(1, a_load))
    a_cols = gl.arange(0, 64, gl.SliceLayout(0, a_load))
    b_row = gl.arange(0, 128, gl.SliceLayout(1, b_byte))
    byte_cols = gl.arange(0, 32, gl.SliceLayout(0, b_byte))
    route = gl.load(Sorted + m * 64 + a_range).to(gl.int64)
    valid = (route >= 0) & (route < ROUTES)
    rows = route // TOKENS
    tile = (expert.to(gl.int64) * (N // 128) + n) * (K // 64)
    if TARGET_KIND == 0:
        outer = gl.load(Outer + expert)
    elif TARGET_KIND == 2:
        row_sc = gl.load(RowScale + expert.to(gl.int64) * N + n * 128 + b_row).to(gl.float32)
    acc = gl.zeros((64, 128), gl.float32, mma)
    for k in range(K // 64):
        x = gl.load(X + rows[:, None] * K + k * 64 + a_cols[None, :], valid[:, None], 0).to(gl.float16)
        packet_row = (tile + k) * 128 + b_row
        packed = gl.load(Packed + packet_row[:, None] * 32 + byte_cols[None, :]).to(gl.uint32)
        raw = gl.load(Raw + packet_row).to(gl.uint32)
        if TARGET_KIND == 0:
            scale_byte = (raw[:, None] >> ((byte_cols[None, :] // 8) * 8)) & 255
            low, high = _unpack_e2m1_f16x2_scaled(packed, scale_byte)
        elif TARGET_KIND == 1:
            scale_bits = (raw[:, None] >> ((byte_cols[None, :] // 16) * 16)).to(gl.uint16)
            scale = scale_bits.to(gl.bfloat16, bitcast=True).to(gl.float32)
            low = ((((packed & 15).to(gl.int32) ^ 8) - 8).to(gl.float32) * scale).to(gl.float16)
            high = ((((packed >> 4).to(gl.int32) ^ 8) - 8).to(gl.float32) * scale).to(gl.float16)
        else:
            low, high = _c1_f16_pair(packed, raw, row_sc, Table, expert)
        weights = gl.reshape(gl.join(low, high), (128, 64))
        a = gl.convert_layout(x, gl.DotOperandLayout(0, mma, 16))
        b = gl.convert_layout(gl.permute(weights, (1, 0)), gl.DotOperandLayout(1, mma, 16))
        acc = wmma(a, b, acc)
    if TARGET_KIND == 0:
        acc = acc * outer
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
    grid = (n_tiles, plan.expert_ids.numel())
    routes = plan.num_tokens * plan.top_k
    tokens = plan.top_k if x.ndim == 2 else 1
    _mixed_split_kind_m64_kernel[grid](
        x, bank.packed, bank.raw_scales, bank.outer_scales, bank.kind,
        bank.row_scales, bank.table,
        plan.sorted_ids, plan.expert_ids, plan.padded_tokens, out, gamma,
        N=n, K=k, ROUTES=routes, TOKENS=tokens,
        TARGET_KIND=0, FUSED=gamma is not None, GATE=gate_cap, UP=up_cap,
        num_warps=4, num_stages=2, enable_fp_fusion=False,
    )
    _mixed_split_kind_m64_kernel[grid](
        x, bank.packed, bank.raw_scales, bank.outer_scales, bank.kind,
        bank.row_scales, bank.table,
        plan.sorted_ids, plan.expert_ids, plan.padded_tokens, out, gamma,
        N=n, K=k, ROUTES=routes, TOKENS=tokens,
        TARGET_KIND=bank.fitted_kind, FUSED=gamma is not None, GATE=gate_cap, UP=up_cap,
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
    head = tl.program_id(0)
    q_block = tl.program_id(1)
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
    d = tl.arange(0, D)

    tok0 = (q_start + q0).to(tl.int64)
    q_base = Q + tok0 * stride_qt + head.to(tl.int64) * stride_qh
    q = tl.load(q_base + q_offs[:, None] * stride_qt + d[None, :], row_valid[:, None], other=0.0)
    q_scaled = (q.to(tl.float32) * scale_log2).to(tl.bfloat16)

    pos = past + q_idx
    first_pos = past + q0
    hi = tl.minimum(past + q0 + BLOCK_Q, seq_len)

    table = BlockTable + req * stride_table
    k_head_base = K + kv_head.to(tl.int64) * stride_kh
    v_head_base = V + kv_head.to(tl.int64) * stride_vh
    n_offs = tl.arange(0, BLOCK_N)

    m_i = tl.full([BLOCK_Q], float("-inf"), tl.float32)
    l_i = tl.zeros([BLOCK_Q], tl.float32)
    acc = tl.zeros([BLOCK_Q, D], tl.float32)

    end_block = tl.cdiv(hi, BLOCK_N)

    if WINDOW < 0:
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

    # BLOCK_Q divides 128, so all rows in this tile share one scalar key0 and bias base pointer.
    tile_end_s = tl.minimum((q0 // 128 + 1) * 128, q_len)
    key0_s = tl.cdiv(past + tile_end_s, 128) * 128 - E_PAD
    bias_base = Bias + tok0 * stride_bt + head.to(tl.int64) * stride_bh

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
        s = tl.dot(q_scaled, tl.trans(k))
        dist = pos[:, None] - keys[None, :]
        valid = row_valid[:, None] & key_valid[None, :] & (dist >= 0)
        if WINDOW >= 0:
            valid &= dist <= WINDOW
        b_col = keys - key0_s
        in_table = valid & (b_col[None, :] >= 0)
        bias = tl.load(bias_base + q_offs[:, None] * stride_bt + b_col[None, :], in_table, other=0.0)
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
    gate = tl.load(Gate + tok0 * stride_gate + q_offs[:, None] * stride_gate + head * D + d[None, :], row_valid[:, None], other=0.0).to(tl.float32)
    sink = tl.load(Sink + tok0 * stride_sink + q_offs * stride_sink + head, row_valid, other=0.0).to(tl.float32)
    y = out * _turbo_sigmoid(gate) * _turbo_sigmoid(lse_e - sink)[:, None]
    tl.store(
        Result + tok0 * stride_res + q_offs[:, None] * stride_res + head * D + d[None, :],
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
    head = tl.program_id(0)
    q_block = tl.program_id(1)
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
    d = tl.arange(0, D)

    tok0 = (q_start + q0).to(tl.int64)
    q_base = Q + tok0 * stride_qt + head.to(tl.int64) * stride_qh
    q = tl.load(q_base + q_offs[:, None] * stride_qt + d[None, :], row_valid[:, None], other=0.0)
    q_scaled = (q.to(tl.float32) * scale_log2).to(tl.bfloat16)
    pos = past + q_idx
    hi = tl.minimum(past + q0 + BLOCK_Q, seq_len)

    table = BlockTable + req_idx * stride_table
    k_head_base = K + kv_head.to(tl.int64) * stride_kh
    v_head_base = V + kv_head.to(tl.int64) * stride_vh
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

    tile_end_s = tl.minimum((q0 // 128 + 1) * 128, q_len)
    key0_s = tl.cdiv(past + tile_end_s, 128) * 128 - E_PAD
    bias_base = Bias + tok0 * stride_bt + head.to(tl.int64) * stride_bh

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
        s = tl.dot(q_scaled, tl.trans(k))
        dist = pos[:, None] - keys[None, :]
        valid = row_valid[:, None] & key_valid[None, :] & (dist >= 0)
        b_col = keys - key0_s
        in_table = valid & (b_col[None, :] >= 0)
        bias = tl.load(bias_base + q_offs[:, None] * stride_bt + b_col[None, :], in_table, other=0.0)
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
    gate = tl.load(Gate + tok0 * stride_gate + q_offs[:, None] * stride_gate + head * D + d[None, :], row_valid[:, None], other=0.0).to(tl.float32)
    sink = tl.load(Sink + tok0 * stride_sink + q_offs * stride_sink + head, row_valid, other=0.0).to(tl.float32)
    y = out * _turbo_sigmoid(gate) * _turbo_sigmoid(lse_e - sink)[:, None]
    tl.store(
        Result + tok0 * stride_res + q_offs[:, None] * stride_res + head * D + d[None, :],
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
    # BLOCK_N=16 stays <= PAGE (16/32/64) for scalar page_id + contiguous loads and higher occupancy (1.44-1.70x faster on gfx1151).
    bq, bn, nw = 64, (16 if page >= 16 else 32), 4
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
                _turbo_prefill_rel_attn_kernel[(num_heads, triton.cdiv(ql, bq), 1)](
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
                _turbo_prefill_rel_attn_near_merge_kernel[(num_heads, triton.cdiv(ql, bq))](
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
    grid = (num_heads, triton.cdiv(max_seqlen_q, bq), num_reqs)
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
    from .hip_runtime import hip_router_gemm_m8

    hip_out = hip_router_gemm_m8(x, w_bf16)
    if hip_out is not None:
        return hip_out
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


@gluon.jit
def _w6_wmma_splitk_batched_kernel(
    X, P4, P2, Scales, Partial, M,
    N: gl.constexpr, K: gl.constexpr, M_TILE: gl.constexpr, SPLITS: gl.constexpr,
):
    tile_n = gl.program_id(0)
    tile_m = gl.program_id(1)
    sp = gl.program_id(2)
    k_tiles: gl.constexpr = K // 64
    k_per_sp: gl.constexpr = k_tiles // SPLITS
    k_start = sp * k_per_sp
    mma: gl.constexpr = AMDWMMALayout(1, True, [[0, 1], [0, 2]])
    a_load: gl.constexpr = gl.BlockedLayout([1, 64], [16, 2], [1, 4], [0, 1])
    out_layout: gl.constexpr = gl.DistributedLinearLayout(
        [[0, 1], [0, 4], [0, 8], [0, 64]],
        [[1, 0], [2, 0], [4, 0], [8, 0], [0, 2]],
        [[0, 16], [0, 32]], [], [16, 128],
    )
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
    for dk in range(k_per_sp):
        kt = k_start + dk
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
    gl.store(
        Partial + (sp * M + out_row[:, None]) * N + tile_n * 128 + out_col[None, :],
        val, out_row[:, None] < M,
    )


def w6_wmma_linear(
    x: torch.Tensor, weight: W6WmmaWeight, *,
    out_dtype: torch.dtype = torch.bfloat16, soft_cap: float = 0.0, splits: int = 1,
) -> torch.Tensor:
    from .hip_runtime import hip_w6_wmma_linear
    hip_out = hip_w6_wmma_linear(x, weight, splits=splits, out_dtype=out_dtype, soft_cap=soft_cap)
    if hip_out is not None:
        return hip_out
    m, k = x.shape
    n_pad = weight.padded_out
    out = x.new_empty((m, n_pad), dtype=out_dtype)
    if (
        m > 16
        and splits > 1
        and out_dtype == torch.bfloat16
        and soft_cap == 0.0
        and (k // 64) % splits == 0
    ):
        partial = torch.empty((splits, m, n_pad), dtype=torch.float32, device=x.device)
        _w6_wmma_splitk_batched_kernel[(n_pad // 128, triton.cdiv(m, 16), splits)](
            x.contiguous(), weight.p4, weight.p2, weight.scales, partial, m,
            N=n_pad, K=k, M_TILE=16, SPLITS=splits, num_warps=4, num_stages=2,
        )
        _reduce_splits_bf16[(triton.cdiv(m * n_pad, 256),)](
            partial, out, m * n_pad, splits, 256, num_warps=4,
        )
    elif m > 16:
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
    from .hip_runtime import hip_w6_wmma_qkv_linear
    hip_out = hip_w6_wmma_qkv_linear(x_q, x_k, x_v, weight, q_pad, k_end)
    if hip_out is not None:
        return hip_out
    m, k = x_q.shape
    n_pad = weight.padded_out
    out = x_q.new_empty((m, n_pad), dtype=torch.bfloat16)
    if m > 16:
        bm = 32
        _w6_wmma_qkv_batched_f16_kernel[(n_pad // 128, triton.cdiv(m, bm))](
            x_q.contiguous(), x_k.contiguous(), x_v.contiguous(),
            weight.p4, weight.p2, weight.scales, out, m,
            Q_TILES=q_pad // 128, K_TILES_END=k_end // 128,
            N=n_pad, K=k, M_TILE=bm, num_warps=4, num_stages=2,
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


@gluon.jit
def _w4_wmma_splitk_kernel(
    X, Packed, Scales, Partial, M,
    N: gl.constexpr, K: gl.constexpr, M_TILE: gl.constexpr, SPLITS: gl.constexpr,
):
    tile_n = gl.program_id(0)
    tile_m = gl.program_id(1)
    sp = gl.program_id(2)
    k_tiles: gl.constexpr = K // 64
    k_per_sp: gl.constexpr = k_tiles // SPLITS
    k_start = sp * k_per_sp
    mma: gl.constexpr = AMDWMMALayout(1, True, [[0, 1], [0, 2]])
    a_load: gl.constexpr = gl.BlockedLayout([1, 64], [16, 2], [1, 4], [0, 1])
    out_layout: gl.constexpr = gl.DistributedLinearLayout(
        [[0, 1], [0, 4], [0, 8], [0, 64]],
        [[1, 0], [2, 0], [4, 0], [8, 0], [0, 2]],
        [[0, 16], [0, 32]], [], [16, 128],
    )
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
    for dk in range(k_per_sp):
        kt = k_start + dk
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
    gl.store(
        Partial + (sp * M + out_row[:, None]) * N + tile_n * 128 + out_col[None, :],
        val, out_row[:, None] < M,
    )


def w4_wmma_linear(
    x: torch.Tensor, weight: W4WmmaWeight, *, bias: torch.Tensor | None = None, splits: int = 1,
) -> torch.Tensor:
    from .hip_runtime import hip_w4_wmma_linear
    hip_out = hip_w4_wmma_linear(x, weight, bias=bias, splits=splits)
    if hip_out is not None:
        return hip_out
    m, k = x.shape
    n_pad = weight.padded_out
    out = x.new_empty((m, n_pad), dtype=torch.bfloat16)
    if m > 16 and bias is None and splits > 1 and (k // 64) % splits == 0:
        partial = torch.empty((splits, m, n_pad), dtype=torch.float32, device=x.device)
        _w4_wmma_splitk_kernel[(n_pad // 128, triton.cdiv(m, 16), splits)](
            x.contiguous(), weight.packed, weight.scales, partial, m,
            N=n_pad, K=k, M_TILE=16, SPLITS=splits, num_warps=4, num_stages=2,
        )
        _reduce_splits_bf16[(triton.cdiv(m * n_pad, 256),)](
            partial, out, m * n_pad, splits, 256, num_warps=4,
        )
    elif m > 16 or bias is not None:
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
    from .hip_runtime import hip_w4_markov_argmax
    hip_out = hip_w4_markov_argmax(latent, packed, scales, bias, base, softcap, valid)
    if hip_out is not None:
        return hip_out
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
# 6. TURBO M=8 SPLIT-KV DECODE ATTENTION
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
    bias_rows = Bias + token * stride_bt + head.to(tl.int64) * stride_bh - key0

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
# 7. FAST M=8 ROUTER & LOOP CONTROLLER DOWN
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
    from .hip_runtime import hip_route_m8

    hip_res = hip_route_m8(logits, bias, scale, "align")
    if hip_res is not None:
        return hip_res
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
    from .hip_runtime import hip_loop_decode_down

    hip_out = hip_loop_decode_down(pair, down_weight)
    if hip_out is not None:
        return hip_out
    streams, m, k = pair.shape
    out = pair.new_empty((streams, m, 64))
    _controller_down_m8_kernel[(64, streams)](
        pair.contiguous(), down_weight.contiguous(), out, M=m, K=k, BLOCK_K=512, num_warps=4,
    )
    return out


# ============================================================================
# 8. FP16-DIRECT MIXED & CODA MoE (M16 DECODE + M64 PREFILL)
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
    scale = entries.to(gl.uint16).to(gl.float16, bitcast=True).to(gl.float32) * row[:, None]
    offset = (entries >> 16).to(gl.uint16).to(gl.float16, bitcast=True).to(gl.float32) * row[:, None]
    scale = _expand_c1_groups(scale, packed.type.layout)
    offset = _expand_c1_groups(offset, packed.type.layout)
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
    ns = 2 if bm == 64 else 1
    _mixed_wmma_turbo_kernel[(plan.expert_ids.numel(), n_tiles)](
        x, bank.packed, bank.raw_scales, bank.outer_scales, bank.kind,
        bank.row_scales, bank.table,
        plan.sorted_ids, plan.expert_ids, plan.padded_tokens, out, gamma,
        N=n, K=k, ROUTES=plan.num_tokens * plan.top_k,
        TOKENS=plan.top_k if x.ndim == 2 else 1, FITTED_KIND=bank.fitted_kind,
        FUSED=gamma is not None, GATE=gate_cap, UP=up_cap, M_TILE=bm,
        num_warps=4, num_stages=ns, enable_fp_fusion=False,
    )
    return out


@gluon.jit
def _coda_wmma_turbo_kernel(
    X, Packed, Raw, RowScale, Table, Sorted, Expert, Padded, Y, Gamma,
    N: gl.constexpr, K: gl.constexpr, ROUTES, TOKENS: gl.constexpr,
    FUSED: gl.constexpr, GATE: gl.constexpr, UP: gl.constexpr,
    M_TILE: gl.constexpr,
):
    if M_TILE == 64:
        n = gl.program_id(0)
        m = gl.program_id(1)
    else:
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
    ns = 2 if bm == 64 else 1
    grid = (n_tiles, plan.expert_ids.numel()) if bm == 64 else (plan.expert_ids.numel(), n_tiles)
    _coda_wmma_turbo_kernel[grid](
        x, bank.packed, bank.tags, bank.row_scales, bank.table,
        plan.sorted_ids, plan.expert_ids, plan.padded_tokens, out, gamma,
        N=n, K=k, ROUTES=plan.num_tokens * plan.top_k,
        TOKENS=plan.top_k if x.ndim == 2 else 1,
        FUSED=gamma is not None, GATE=gate_cap, UP=up_cap, M_TILE=bm,
        num_warps=4, num_stages=ns, enable_fp_fusion=False,
    )
    return out


# ============================================================================
# 9. FUSED LOOP BLEND + RMSNORM (BATCHED DECODE M=16..64 & PREFILL)
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
    D: tl.constexpr, BT: tl.constexpr, BD: tl.constexpr,
    N_DBLK: tl.constexpr, G_POW2: tl.constexpr,
):
    t_blk = tl.program_id(0)
    d_blk = tl.program_id(1)
    s = tl.program_id(2).to(tl.int64)
    t = (t_blk * BT + tl.arange(0, BT)).to(tl.int64)
    d = d_blk * BD + tl.arange(0, BD)
    t_mask = t < T
    g = tl.arange(0, G_POW2)
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
    if 1 <= t_len <= 64:
        from .hip_runtime import hip_loop_decode_blend_rms

        hip_out = hip_loop_decode_blend_rms(hidden, weight, dest, message, norm_weight, norm_eps)
        if hip_out is not None:
            return hip_out
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
    out = message.new_empty(message.shape, dtype=torch.bfloat16)
    if t_len <= 64:
        n_dblk, d_per_blk, g_pow2 = 36, 64, 64
        bt = 16 if t_len <= 16 else (32 if t_len <= 32 else 64)
    else:
        n_dblk, d_per_blk, g_pow2 = 9, 256, 16
        bt = 32
    partial_sum_sq = message.new_empty((streams, n_dblk, t_len), dtype=torch.float32)
    _fused_loop_blend_p1_kernel[(triton.cdiv(t_len, bt), n_dblk, streams)](
        hidden, weight, dest, message, blended, partial_sum_sq,
        *dest.stride()[:2], *message.stride()[:2], t_len,
        D=width, BT=bt, BD=64, D_PER_BLK=d_per_blk, N_DBLK=n_dblk, num_warps=4,
    )
    _fused_loop_blend_p2_kernel[(triton.cdiv(t_len, bt), width // 128, streams)](
        blended, partial_sum_sq, norm_weight, out, norm_weight.stride(0), t_len, norm_eps,
        D=width, BT=bt, BD=128, N_DBLK=n_dblk, G_POW2=g_pow2, num_warps=4,
    )
    return out


# ============================================================================
# 10. SPLIT CONTIGUOUS ATTENTION PROLOGUE (STRIDED [H,T+128,E] + SKIP-CACHE KV)
# ============================================================================

@triton.jit
def _fast_qr_rel_strided_kernel(
    Q, R, QScale, RScale, Tau, RelProj, QOut, RelOut,
    SeqIdx, CuSeqlensQ, SeqLens,
    T, stride_q, stride_r, stride_rel_t, stride_rel_h,
    q_threshold, r_threshold, eps,
    H: tl.constexpr, D: tl.constexpr, D_REL: tl.constexpr,
    E: tl.constexpr, W: tl.constexpr, HAS_TAU: tl.constexpr,
    BLOCK_H: tl.constexpr, BLOCK_E: tl.constexpr,
):
    t = tl.program_id(0).to(tl.int64)
    if t >= T:
        return
    E_PAD: tl.constexpr = E + 256
    heads = tl.arange(0, BLOCK_H)[:, None]
    h_mask = heads < H
    tau = tl.load(Tau + t) if HAS_TAU else 1.0

    q_cols = tl.arange(0, D)[None, :]
    q = _l2_clip(Q + t * stride_q, QScale, heads, q_cols, h_mask, q_threshold, eps, tau, HAS_TAU)
    tl.store(QOut + t * H * D + heads * D + q_cols, q, h_mask)

    r_cols = tl.arange(0, D_REL)[None, :]
    r = _l2_clip(R + t * stride_r, RScale, heads, r_cols, h_mask, r_threshold, eps, tau, HAS_TAU)

    seq = tl.load(SeqIdx + t)
    q_start = tl.load(CuSeqlensQ + seq)
    q_len = tl.load(CuSeqlensQ + seq + 1) - q_start
    q_idx = t - q_start
    pos = tl.load(SeqLens + seq) - q_len + q_idx
    tile_end = tl.minimum((q_idx // 128 + 1) * 128, q_len)
    diag = pos + E_PAD - tl.cdiv(pos - q_idx + tile_end, 128) * 128
    diag = tl.minimum(tl.maximum(diag, E + 1), E_PAD - 1)
    diag8 = diag // 8 * 8
    table = RelProj + (diag - diag8) * (D_REL + 1) * W + E + 256 - diag8
    e_idx = tl.arange(0, BLOCK_E)[None, :]
    out_base = RelOut + t * stride_rel_t + heads.to(tl.int64) * stride_rel_h
    for chunk in tl.static_range(E_PAD // BLOCK_E):
        col = chunk * BLOCK_E + e_idx
        proj = tl.load(table + tl.trans(r_cols) * W + col)
        pad = tl.load(table + D_REL * W + col)
        rel = tl.dot(r, proj) + pad.to(tl.float32)
        tl.store(out_base + col, rel.to(RelOut.dtype.element_ty), h_mask)


@triton.jit
def _kv_conv_tile_skip_kernel(
    k_ptr, v_ptr, wk_ptr, wv_ptr, dk_ptr, dv_ptr,
    cache_ptr, key_cache_ptr, value_cache_ptr, kv_slot_ptr,
    pos_ptr, seq_idx_ptr, slot_ptr, block_table_ptr, qstart_ptr,
    T, stride_k, stride_v, stride_w0, stride_w1, stride_dk, stride_dv,
    stride_c_blk, stride_c_h, stride_c_n,
    stride_kv_blk, stride_kv_n, stride_kv_h, kv_block_size,
    stride_bt, max_blocks, N, scale, clip_threshold, clip_eps, norm_eps,
    W: tl.constexpr, D: tl.constexpr, DYNAMIC: tl.constexpr,
    BT: tl.constexpr, SKIP_IDLE_CACHE: tl.constexpr,
):
    first = tl.program_id(0) * BT
    tile_reads_cache = True
    if SKIP_IDLE_CACHE:
        last = tl.minimum(first + BT, T) - 1
        tile_reads_cache = tl.load(qstart_ptr + last) > first - (W - 1)
    toff = first + tl.arange(0, BT)
    head = tl.program_id(1)
    is_v = tl.program_id(2) == 1
    d_idx = tl.arange(0, D)
    coff = head * D + d_idx
    t_mask = toff < T
    if is_v:
        x_ptr, w_ptr, dyn_ptr, stride_x, stride_dyn = v_ptr, wv_ptr, dv_ptr, stride_v, stride_dv
    else:
        x_ptr, w_ptr, dyn_ptr, stride_x, stride_dyn = k_ptr, wk_ptr, dk_ptr, stride_k, stride_dk
    y = _conv(
        x_ptr, stride_x, cache_ptr, head * stride_c_h + tl.program_id(2) * D + d_idx,
        stride_c_blk, stride_c_n, N, w_ptr, stride_w0, stride_w1,
        pos_ptr, seq_idx_ptr, slot_ptr, block_table_ptr, stride_bt, max_blocks,
        qstart_ptr, scale, toff, coff, t_mask, d_idx < D,
        W, dyn_ptr if DYNAMIC else None, stride_dyn, D,
        SKIP_IDLE_CACHE=SKIP_IDLE_CACHE,
        CACHE_FIRST=False,
        tile_reads_cache=tile_reads_cache,
    )
    y = y.to(k_ptr.dtype.element_ty).to(tl.float32)
    sum_sq = tl.sum(y * y, axis=1)
    if is_v:
        y = y * tl.rsqrt(tl.div_rn(sum_sq, D) + norm_eps)[:, None]
        cache = value_cache_ptr
    else:
        norm = tl.where(sum_sq > 0, sum_sq * tl.rsqrt(sum_sq), 0.0)
        y = y * tl.minimum(tl.div_rn(clip_threshold, norm + clip_eps), 1.0)[:, None]
        cache = key_cache_ptr
    kv_slot = tl.load(kv_slot_ptr + toff, mask=t_mask, other=-1)
    safe = tl.maximum(kv_slot, 0)
    dst = (
        cache + (safe // kv_block_size)[:, None] * stride_kv_blk
        + (safe % kv_block_size)[:, None] * stride_kv_n
        + head * stride_kv_h + d_idx[None, :]
    )
    tl.store(dst, y.to(cache.dtype.element_ty), mask=(kv_slot >= 0)[:, None])


def turbo_attention_prologue(
    q: torch.Tensor, r: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
    q_scale: torch.Tensor, r_scale: torch.Tensor, tau: torch.Tensor | None,
    rel_proj: torch.Tensor, k_weight: torch.Tensor, v_weight: torch.Tensor,
    k_dynamic: torch.Tensor | None, v_dynamic: torch.Tensor | None,
    cache: torch.Tensor, positions: torch.Tensor, block_table: torch.Tensor,
    seq_idx: torch.Tensor, slot_mapping: torch.Tensor, query_start: torch.Tensor,
    key_cache: torch.Tensor, value_cache: torch.Tensor, kv_slot_mapping: torch.Tensor,
    q_out: torch.Tensor, rel_out: torch.Tensor, cu_seqlens_q: torch.Tensor,
    seq_lens: torch.Tensor, *,
    clip_threshold: float, rel_threshold: float, norm_eps: float, clip_eps: float = 1e-6,
) -> None:
    num_tokens = kv_slot_mapping.shape[0]
    num_heads, head_dim = key_cache.shape[2:]
    num_q_heads, d_rel = r.shape[1:]
    extent = rel_proj.shape[2] - 512
    dynamic = k_dynamic is not None
    if dynamic:
        width = k_weight.shape[0]
        dyn_strides = (k_dynamic.stride(0), v_dynamic.stride(0))
    else:
        width = k_weight.shape[1]
        dyn_strides = (0, 0)
    block_e = min(extent, 256)
    _fast_qr_rel_strided_kernel[(num_tokens,)](
        q, r, q_scale, r_scale, tau, rel_proj, q_out, rel_out,
        seq_idx, cu_seqlens_q, seq_lens, num_tokens,
        q.stride(0), r.stride(0), rel_out.stride(0), rel_out.stride(1),
        clip_threshold, rel_threshold, clip_eps,
        H=num_q_heads, D=head_dim, D_REL=d_rel, E=extent, W=rel_proj.shape[2],
        HAS_TAU=tau is not None, BLOCK_H=max(triton.next_power_of_2(num_q_heads), 16),
        BLOCK_E=block_e, num_warps=8, enable_fp_fusion=False,
    )
    prefill = num_tokens > 256
    scale = (width * (2 if dynamic else 1) + 1) ** -0.5
    if prefill:
        from .hip_moe_prefill import hip_kv_conv_prefill

        if hip_kv_conv_prefill(
            k, v, k_weight, v_weight, k_dynamic, v_dynamic,
            cache, key_cache, value_cache, kv_slot_mapping,
            positions, seq_idx, slot_mapping, block_table, query_start,
            num_tokens=num_tokens, num_heads=num_heads, head_dim=head_dim,
            width=width, scale=scale, clip_threshold=clip_threshold,
            clip_eps=clip_eps, norm_eps=norm_eps,
        ):
            return
    bt = (64 if not dynamic else 16) if prefill else 4
    _kv_conv_tile_skip_kernel[(triton.cdiv(num_tokens, bt), num_heads, 2)](
        k, v, k_weight, v_weight,
        k_dynamic if dynamic else k_weight,
        v_dynamic if dynamic else v_weight,
        cache, key_cache, value_cache, kv_slot_mapping,
        positions, seq_idx, slot_mapping, block_table, query_start,
        num_tokens, k.stride(0), v.stride(0), k_weight.stride(0), k_weight.stride(1),
        *dyn_strides,
        cache.stride(0), cache.stride(1), cache.stride(2),
        key_cache.stride(0), key_cache.stride(1), key_cache.stride(2),
        key_cache.shape[1], block_table.stride(0), block_table.shape[1], cache.shape[2],
        scale, clip_threshold, clip_eps, norm_eps,
        W=width, D=head_dim, DYNAMIC=dynamic, BT=bt, SKIP_IDLE_CACHE=prefill,
        num_warps=4, enable_fp_fusion=False,
    )


# ============================================================================
# 10. PREFILL ROUTER BF16-WMMA->FP32 & ALL-IN-ONE FUSED LOOP CONTROLLER
# ============================================================================

@triton.jit
def _router_prefill_bf16_to_fp32_kernel(
    X, W, Bias, Score, Corrected,
    T, K: tl.constexpr, N: tl.constexpr, BT: tl.constexpr, BK: tl.constexpr,
):
    t_blk = tl.program_id(0)
    t = t_blk * BT + tl.arange(0, BT)
    t_mask = t < T
    n = tl.arange(0, N)
    acc = tl.zeros((BT, N), tl.float32)
    for k0 in range(0, K, BK):
        k = k0 + tl.arange(0, BK)
        x = tl.load(X + t[:, None].to(tl.int64) * K + k[None, :], t_mask[:, None], other=0)
        w = tl.load(W + n[None, :].to(tl.int64) * K + k[:, None])
        acc = tl.dot(x, w, acc)
    score = tl.div_rn(1.0, 1.0 + tl_hip_lib.exp(-acc))
    bias = tl.load(Bias + n)
    corr = score + bias[None, :]
    tl.store(Score + t[:, None].to(tl.int64) * N + n[None, :], score, t_mask[:, None])
    tl.store(Corrected + t[:, None].to(tl.int64) * N + n[None, :], corr, t_mask[:, None])


def router_scores_prefill(
    x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor] | None:
    if x.ndim == 2 and x.shape[0] >= 64:
        from .hip_runtime import hip_router_prefill

        hip_res = hip_router_prefill(x, weight, bias)
        if hip_res is not None:
            return hip_res
    if (
        not x.is_cuda
        or x.ndim != 2
        or x.shape[0] < 64
        or x.shape[1] != 2304
        or weight.shape != (128, 2304)
        or bias.shape != (128,)
        or x.dtype != torch.bfloat16
        or weight.dtype != torch.bfloat16
        or bias.dtype != torch.float32
        or not x.is_contiguous()
        or not weight.is_contiguous()
        or not bias.is_contiguous()
    ):
        return None
    t_len = x.shape[0]
    score = x.new_empty((t_len, 128), dtype=torch.float32)
    corrected = x.new_empty((t_len, 128), dtype=torch.float32)
    bt = 128 if t_len >= 512 else 64
    nw = 8 if bt == 128 else 4
    _router_prefill_bf16_to_fp32_kernel[(triton.cdiv(t_len, bt),)](
        x, weight, bias, score, corrected, t_len, K=2304, N=128, BT=bt, BK=64, num_warps=nw,
    )
    return score, corrected


@triton.jit
def _fused_loop_allinone_kernel(
    Dest, Msg, WdownT, Wup, NormW, Out,
    sd_s, sd_t, sm_s, sm_t, sw_s,
    T, norm_eps, D: tl.constexpr, BT: tl.constexpr, BK: tl.constexpr,
):
    t_blk = tl.program_id(0)
    s = tl.program_id(1).to(tl.int64)
    t = (t_blk * BT + tl.arange(0, BT)).to(tl.int64)
    t_mask = t < T
    r = tl.arange(0, 64)
    acc_d = tl.zeros((BT, 64), tl.float32)
    acc_m = tl.zeros((BT, 64), tl.float32)
    sq_d = tl.zeros((BT,), tl.float32)
    sq_m = tl.zeros((BT,), tl.float32)
    wd_base = WdownT + s * (2 * D * 64)
    for k0 in range(0, D, BK):
        k = k0 + tl.arange(0, BK)
        d_val = tl.load(Dest + s * sd_s + t[:, None] * sd_t + k[None, :], t_mask[:, None], other=0.0)
        m_val = tl.load(Msg + s * sm_s + t[:, None] * sm_t + k[None, :], t_mask[:, None], other=0.0)
        sq_d += tl.sum(d_val * d_val, axis=1)
        sq_m += tl.sum(m_val * m_val, axis=1)
        wd = tl.load(wd_base + k[:, None] * 64 + r[None, :])
        wm = tl.load(wd_base + (D + k[:, None]) * 64 + r[None, :])
        acc_d += tl.dot(d_val.to(tl.bfloat16), wd)
        acc_m += tl.dot(m_val.to(tl.bfloat16), wm)
    rstd_d = tl_hip_lib.rsqrt(tl.div_rn(sq_d, D) + 1e-6)
    rstd_m = tl_hip_lib.rsqrt(tl.div_rn(sq_m, D) + 1e-6)
    hidden = (acc_d * rstd_d[:, None] + acc_m * rstd_m[:, None]).to(tl.bfloat16)

    wu_base = Wup + s * (2 * D * 64)
    sq_b = tl.zeros((BT,), tl.float32)
    out_base = Out + (s * T + t[:, None]) * D
    for k0 in range(0, D, BK):
        k = k0 + tl.arange(0, BK)
        wg = tl.load(wu_base + k[:, None] * 64 + r[None, :])
        wf = tl.load(wu_base + (D + k[:, None]) * 64 + r[None, :])
        gain = tl_hip_lib.tanh(tl.dot(hidden, tl.trans(wg)))
        forget = tl_hip_lib.tanh(tl.dot(hidden, tl.trans(wf)))
        d_val = tl.load(Dest + s * sd_s + t[:, None] * sd_t + k[None, :], t_mask[:, None], other=0.0)
        m_val = tl.load(Msg + s * sm_s + t[:, None] * sm_t + k[None, :], t_mask[:, None], other=0.0)
        b = (1.0 - forget) * d_val + gain * m_val
        sq_b += tl.sum(b * b, axis=1)
        tl.store(out_base + k[None, :], b.to(tl.bfloat16), t_mask[:, None])
    rstd_b = tl_hip_lib.rsqrt(tl.div_rn(sq_b, D) + norm_eps)
    nw_base = NormW + s * sw_s
    for k0 in range(0, D, BK):
        k = k0 + tl.arange(0, BK)
        b = tl.load(out_base + k[None, :], t_mask[:, None], other=0.0).to(tl.float32)
        nw = tl.load(nw_base + k).to(tl.float32)
        tl.store(out_base + k[None, :], (b * rstd_b[:, None] * nw[None, :]).to(tl.bfloat16), t_mask[:, None])


def fused_loop_allinone_prefill(
    dest: torch.Tensor,
    message: torch.Tensor,
    down_weight_t: torch.Tensor,
    up_weight: torch.Tensor,
    norm_weight: torch.Tensor,
    norm_eps: float,
) -> torch.Tensor | None:
    from .loop import _packed_rows

    if not _packed_rows(dest, message):
        return None
    streams, t_len, width = message.shape
    if (
        t_len < 64
        or width != 2304
        or down_weight_t.shape != (streams, 2 * width, 64)
        or up_weight.shape != (streams, 2 * width, 64)
        or down_weight_t.dtype != torch.bfloat16
        or up_weight.dtype != torch.bfloat16
        or not down_weight_t.is_contiguous()
        or not up_weight.is_contiguous()
        or norm_weight.shape != (streams, width)
        or norm_weight.stride(-1) != 1
    ):
        return None
    out = message.new_empty(message.shape, dtype=torch.bfloat16)
    bt = 16
    _fused_loop_allinone_kernel[(triton.cdiv(t_len, bt), streams)](
        dest, message, down_weight_t, up_weight, norm_weight, out,
        *dest.stride()[:2], *message.stride()[:2], norm_weight.stride(0),
        t_len, norm_eps, D=width, BT=bt, BK=64, num_warps=4,
    )
    return out


@triton.jit
def _turbo_fast_tanh(x):
    ax = tl.abs(x)
    e = tl.math.exp2(-2.8853900817779268 * ax)
    t = (1.0 - e) / (1.0 + e)
    return tl.where(x >= 0.0, t, -t)


@triton.jit
def _fused_loop_zeromat_kernel(
    Dest, Bank0, Bank1, Coeffs, WdownT, Wup, NormW, Out,
    sd_s, sd_t, sb0, sb1, sc_s, sw_s,
    T, norm_eps, D: tl.constexpr, BT: tl.constexpr, BK: tl.constexpr,
):
    # Grid=(streams, cdiv(T, BT)) keeps all S streams for the same token block
    # co-scheduled across CUs for high L2/MALL reuse of Dest, Bank0, and Bank1.
    s = tl.program_id(0).to(tl.int64)
    t_blk = tl.program_id(1)
    t = (t_blk * BT + tl.arange(0, BT)).to(tl.int64)
    t_mask = t < T
    r = tl.arange(0, 64)

    c0 = tl.load(Coeffs + s * sc_s + t * 2 + 0, t_mask, other=0.0)
    c1 = tl.load(Coeffs + s * sc_s + t * 2 + 1, t_mask, other=0.0)

    acc_d = tl.zeros((BT, 64), tl.float32)
    acc_m = tl.zeros((BT, 64), tl.float32)
    sq_d = tl.zeros((BT,), tl.float32)
    sq_m = tl.zeros((BT,), tl.float32)
    wd_base = WdownT + s * (2 * D * 64)
    for k0 in range(0, D, BK):
        k = k0 + tl.arange(0, BK)
        d_val = tl.load(Dest + s * sd_s + t[:, None] * sd_t + k[None, :], t_mask[:, None], other=0.0)
        b0 = tl.load(Bank0 + t[:, None] * sb0 + k[None, :], t_mask[:, None], other=0.0).to(tl.float32)
        b1 = tl.load(Bank1 + t[:, None] * sb1 + k[None, :], t_mask[:, None], other=0.0).to(tl.float32)
        m_val = c0[:, None] * b0 + c1[:, None] * b1
        sq_d += tl.sum(d_val * d_val, axis=1)
        sq_m += tl.sum(m_val * m_val, axis=1)
        wd = tl.load(wd_base + k[:, None] * 64 + r[None, :])
        wm = tl.load(wd_base + (D + k[:, None]) * 64 + r[None, :])
        acc_d += tl.dot(d_val.to(tl.bfloat16), wd)
        acc_m += tl.dot(m_val.to(tl.bfloat16), wm)
    rstd_d = tl_hip_lib.rsqrt(tl.div_rn(sq_d, D) + 1e-6)
    rstd_m = tl_hip_lib.rsqrt(tl.div_rn(sq_m, D) + 1e-6)
    hidden = (acc_d * rstd_d[:, None] + acc_m * rstd_m[:, None]).to(tl.bfloat16)

    wu_base = Wup + s * (2 * D * 64)
    sq_b = tl.zeros((BT,), tl.float32)
    out_base = Out + (s * T + t[:, None]) * D
    for k0 in range(0, D, BK):
        k = k0 + tl.arange(0, BK)
        wg = tl.load(wu_base + k[:, None] * 64 + r[None, :])
        wf = tl.load(wu_base + (D + k[:, None]) * 64 + r[None, :])
        gain = _turbo_fast_tanh(tl.dot(hidden, tl.trans(wg)))
        forget = _turbo_fast_tanh(tl.dot(hidden, tl.trans(wf)))
        d_val = tl.load(Dest + s * sd_s + t[:, None] * sd_t + k[None, :], t_mask[:, None], other=0.0)
        b0 = tl.load(Bank0 + t[:, None] * sb0 + k[None, :], t_mask[:, None], other=0.0).to(tl.float32)
        b1 = tl.load(Bank1 + t[:, None] * sb1 + k[None, :], t_mask[:, None], other=0.0).to(tl.float32)
        m_val = c0[:, None] * b0 + c1[:, None] * b1
        b = (1.0 - forget) * d_val + gain * m_val
        sq_b += tl.sum(b * b, axis=1)
        tl.store(out_base + k[None, :], b.to(tl.bfloat16), t_mask[:, None])
    rstd_b = tl_hip_lib.rsqrt(tl.div_rn(sq_b, D) + norm_eps)
    nw_base = NormW + s * sw_s
    for k0 in range(0, D, BK):
        k = k0 + tl.arange(0, BK)
        b = tl.load(out_base + k[None, :], t_mask[:, None], other=0.0).to(tl.float32)
        nw = tl.load(nw_base + k).to(tl.float32)
        tl.store(out_base + k[None, :], (b * rstd_b[:, None] * nw[None, :]).to(tl.bfloat16), t_mask[:, None])


def fused_loop_zeromat_prefill(
    dest: torch.Tensor,
    bank0: torch.Tensor,
    bank1: torch.Tensor,
    coeffs: torch.Tensor,
    down_weight_t: torch.Tensor,
    up_weight: torch.Tensor,
    norm_weight: torch.Tensor,
    norm_eps: float,
) -> torch.Tensor | None:
    if dest.ndim == 3 and dest.shape[1] >= 64:
        from .hip_runtime import hip_loop_zeromat_prefill

        hip_out = hip_loop_zeromat_prefill(
            dest, bank0, bank1, coeffs, down_weight_t, up_weight, norm_weight, norm_eps,
        )
        if hip_out is not None:
            return hip_out
    if (
        not dest.is_cuda
        or dest.ndim != 3
        or dest.dtype != torch.float32
        or dest.stride(-1) != 1
        or bank0.dtype != torch.bfloat16
        or bank1.dtype != torch.bfloat16
        or coeffs.dtype != torch.float32
        or not coeffs.is_contiguous()
    ):
        return None
    streams, t_len, width = dest.shape
    if (
        t_len < 64
        or width != 2304
        or streams not in (1, 3)
        or bank0.shape != (t_len, width)
        or bank1.shape != (t_len, width)
        or bank0.stride(-1) != 1
        or bank1.stride(-1) != 1
        or coeffs.shape != (streams, t_len, 2)
        or down_weight_t.shape != (streams, 2 * width, 64)
        or up_weight.shape != (streams, 2 * width, 64)
        or down_weight_t.dtype != torch.bfloat16
        or up_weight.dtype != torch.bfloat16
        or not down_weight_t.is_contiguous()
        or not up_weight.is_contiguous()
        or norm_weight.shape != (streams, width)
        or norm_weight.stride(-1) != 1
    ):
        return None
    out = bank0.new_empty((streams, t_len, width), dtype=torch.bfloat16)
    bt = 16
    _fused_loop_zeromat_kernel[(streams, triton.cdiv(t_len, bt))](
        dest, bank0, bank1, coeffs, down_weight_t, up_weight, norm_weight, out,
        *dest.stride()[:2], bank0.stride(0), bank1.stride(0), coeffs.stride(0), norm_weight.stride(0),
        t_len, norm_eps, D=width, BT=bt, BK=64, num_warps=4,
    )
    return out
