# SPDX-License-Identifier: Apache-2.0
# Modified or added by Ciru for Inkling 2 Mini on AMD gfx1151.
"""Packed source NVFP4 W13/W2 WMMA tile replay for gfx1151."""

import os
from typing import NamedTuple

import triton
import torch
from triton.experimental import gluon
from triton.experimental.gluon import language as gl
from triton.experimental.gluon.language.amd import AMDWMMALayout
from triton.experimental.gluon.language.amd.rdna3 import wmma
from triton.experimental.gluon.language.extra import libdevice


@gluon.jit
def _unpack_e2m1(packed):
    selector = (packed & 7) | ((packed << 4) & 0x0700)
    mags = gl.inline_asm_elementwise(
        "v_perm_b32 $0, $1, $2, $3", constraints="=v,v,v,v",
        args=[0x0C080604, 0x03020100, selector], dtype=gl.uint32, is_pure=True, pack=1,
    )
    low = (mags & 255).to(gl.float32).to(gl.uint32, bitcast=True)
    high = ((mags >> 8) & 255).to(gl.float32).to(gl.uint32, bitcast=True)
    return ((low | ((packed & 8) << 28)).to(gl.float32, bitcast=True),
            (high | ((packed & 128) << 24)).to(gl.float32, bitcast=True))


@gluon.jit
def _finite_bf16(value):
    # The validated inputs keep every source weight finite in FP32.
    bits = value.to(gl.uint32, bitcast=True)
    rounded = bits + 0x7FFF + ((bits >> 16) & 1)
    return (rounded >> 16).to(gl.uint16).to(gl.bfloat16, bitcast=True)


@gluon.jit
def _source_wmma_pretiled(
    X, Packed, Raw, Outer, Sorted, Expert, Padded, Y, Gamma,
    N: gl.constexpr, K: gl.constexpr, ROUTES: gl.constexpr,
    TOKENS: gl.constexpr, FUSED: gl.constexpr, GATE: gl.constexpr, UP: gl.constexpr,
    N_FAST: gl.constexpr = False, WAVE_EPILOGUE: gl.constexpr = False,
):
    m = gl.program_id(1 if N_FAST else 0)
    n = gl.program_id(0 if N_FAST else 1)
    if m * 16 >= gl.load(Padded):
        return
    expert = gl.load(Expert + m)
    if expert < 0:
        return
    mma: gl.constexpr = AMDWMMALayout(1, True, [[0, 1], [0, 2]])
    a_load: gl.constexpr = gl.BlockedLayout([1, 64], [16, 2], [1, 4], [0, 1])
    # Each wave holds two output fragments. Each half decodes one fragment
    # before a lane exchange supplies the duplicate registers required by WMMA.
    b_byte: gl.constexpr = gl.DistributedLinearLayout(
        [[0, 1], [0, 2], [0, 4], [0, 8], [0, 16]],
        [[1, 0], [2, 0], [4, 0], [8, 0], [64, 0]],
        [[16, 0], [32, 0]], [], [128, 32],
    )
    b_rows: gl.constexpr = gl.SliceLayout(1, b_byte)
    a_range = gl.arange(0, 16, gl.SliceLayout(1, a_load))
    a_cols = gl.arange(0, 64, gl.SliceLayout(0, a_load))
    b_row = gl.arange(0, 128, b_rows)
    byte_cols = gl.arange(0, 32, gl.SliceLayout(0, b_byte))
    route = gl.load(Sorted + m * 16 + a_range).to(gl.int64)
    valid = (route >= 0) & (route < ROUTES)
    rows = route // TOKENS
    tile = (expert.to(gl.int64) * (N // 128) + n) * (K // 64)
    outer = gl.load(Outer + expert)
    acc = gl.zeros((16, 128), gl.float32, mma)
    for k in range(K // 64):
        x = gl.load(
            X + rows[:, None] * K + (k * 64 + a_cols[None, :]), valid[:, None], 0,
        )
        packet_row = (tile + k) * 128 + b_row
        packed = gl.load(Packed + packet_row[:, None] * 32 + byte_cols[None, :]).to(gl.uint32)
        raw = gl.load(Raw + packet_row).to(gl.uint32)
        scale_byte = (raw[:, None] >> ((byte_cols[None, :] // 8) * 8)) & 255
        scale_bits = ((scale_byte << 4) + (119 << 7)).to(gl.uint16)
        scale = scale_bits.to(gl.bfloat16, bitcast=True).to(gl.float32)
        low, high = _unpack_e2m1(packed)
        low = _finite_bf16((low * scale) * outer)
        high = _finite_bf16((high * scale) * outer)
        weights = gl.reshape(gl.join(low, high), (128, 64))
        a = gl.convert_layout(x, gl.DotOperandLayout(0, mma, 16), assert_trivial=True)
        b = gl.convert_layout(gl.permute(weights, (1, 0)), gl.DotOperandLayout(1, mma, 16))
        acc = wmma(a, b, acc)
    if WAVE_EPILOGUE:
        # Keep each gate/up pair in one lane and the accumulator fragments
        # in their original waves.
        out_layout: gl.constexpr = gl.DistributedLinearLayout(
            [[0, 1], [0, 4], [0, 8], [0, 64]],
            [[1, 0], [2, 0], [4, 0], [8, 0], [0, 2]],
            [[0, 16], [0, 32]], [], [16, 128],
        )
    else:
        out_layout: gl.constexpr = gl.BlockedLayout([1, 2], [1, 32], [1, 4], [1, 0])
    out_rows: gl.constexpr = gl.SliceLayout(1, out_layout)
    route = gl.convert_layout(route, out_rows, assert_trivial=WAVE_EPILOGUE)
    valid = gl.convert_layout(valid, out_rows, assert_trivial=WAVE_EPILOGUE)
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
        if WAVE_EPILOGUE:
            mid_rows: gl.constexpr = gl.SliceLayout(1, mid.type.layout)
            route = gl.convert_layout(route, mid_rows, assert_trivial=True)
            valid = gl.convert_layout(valid, mid_rows, assert_trivial=True)
            cols = n * 64 + gl.arange(0, 64, gl.SliceLayout(0, mid.type.layout))
        else:
            mid = gl.convert_layout(mid, out_layout)
            cols = n * 64 + gl.arange(0, 64, gl.SliceLayout(0, out_layout))
        gl.store(Y + route[:, None] * (N // 2) + cols[None, :], mid, valid[:, None])
    else:
        cols = n * 128 + gl.arange(0, 128, gl.SliceLayout(0, out_layout))
        gl.store(Y + route[:, None] * N + cols[None, :], out, valid[:, None])


class SourceWmmaBank(NamedTuple):
    packed: torch.Tensor
    raw_scales: torch.Tensor
    outer_scales: torch.Tensor


def pack_source_wmma(
    packed: torch.Tensor, raw_scales: torch.Tensor, outer_scales: torch.Tensor,
) -> SourceWmmaBank:
    """Transpose the packet and scale tiles while retaining every original byte."""
    experts, n, bytes_k = packed.shape
    k = bytes_k * 2
    if (packed.dtype != torch.uint8 or raw_scales.dtype != torch.uint8
            or raw_scales.shape != (experts, n, k // 16)
            or outer_scales.shape != (experts,) or outer_scales.dtype != torch.float32
            or k not in (1152, 2304) or n % 128):
        raise ValueError("Expected homogeneous source NVFP4 expert banks aligned to N128/K64")
    if not (packed.device == raw_scales.device == outer_scales.device):
        raise ValueError("Expected source expert banks on one device")
    lo, hi = torch.aminmax(raw_scales)
    if not 8 <= int(lo) <= int(hi) <= 126:
        raise ValueError("The scaled WMMA source path requires normal E4M3 scale bytes")
    max_outer = torch.finfo(torch.float32).max / (6 * 448)
    if not bool(torch.all(outer_scales.abs() <= max_outer)):
        raise ValueError("Expected finite reconstructed FP32 source weights")
    tiles = (experts, n // 128, 128, k // 64)
    tiled_packed = packed.reshape(*tiles, 32).transpose(2, 3).contiguous()
    tiled_scales = raw_scales.reshape(*tiles, 4).transpose(2, 3).contiguous()
    return SourceWmmaBank(tiled_packed, tiled_scales.view(torch.int32).squeeze(-1), outer_scales)


def source_wmma_projection(x, bank: SourceWmmaBank, plan, *, fused_activation=None):
    """Use the existing sorted route plan and emit the same BF16 route slots."""
    if (plan.block_m, plan.block_n, plan.block_k) != (16, 128, 64):
        raise ValueError("The packed source WMMA prototype requires M16/N128/K64")
    experts, n_tiles, k_tiles, _, _ = bank.packed.shape
    n, k = n_tiles * 128, k_tiles * 64
    route_shape = (plan.num_tokens, plan.top_k)
    if (experts != plan.num_experts or x.dtype != torch.bfloat16
            or x.shape not in ((plan.num_tokens, k), (*route_shape, k))
            or not x.is_contiguous()):
        raise ValueError("Expected contiguous BF16 source expert activations matching the plan")
    gamma, gate_cap, up_cap = None, 0.0, 0.0
    if fused_activation is not None:
        gamma, gate_cap, up_cap = fused_activation
        if (x.ndim != 2 or gamma.shape != route_shape or gamma.dtype != torch.float32
                or not gamma.is_contiguous() or gate_cap <= 0 or up_cap <= 0):
            raise ValueError("Expected W13 inputs, FP32 route weights and positive softcaps")
    out = x.new_empty((*route_shape, n // 2 if gamma is not None else n))
    n_fast = plan.num_tokens == 8 and os.getenv("INKLING_AMD_SOURCE_WMMA_N_FAST") == "1"
    wave_epilogue = (
        plan.num_tokens == 8 and os.getenv("INKLING_AMD_SOURCE_WMMA_WAVE_EPILOGUE") == "1"
    )
    m_tiles = plan.expert_ids.numel()
    grid = (n_tiles, m_tiles) if n_fast else (m_tiles, n_tiles)
    _source_wmma_pretiled[grid](
        x, *bank, plan.sorted_ids, plan.expert_ids, plan.padded_tokens, out, gamma,
        N=n, K=k, ROUTES=plan.num_tokens * plan.top_k,
        TOKENS=plan.top_k if x.ndim == 2 else 1, FUSED=gamma is not None,
        GATE=gate_cap, UP=up_cap, N_FAST=n_fast, WAVE_EPILOGUE=wave_epilogue,
        num_warps=4, num_stages=1, enable_fp_fusion=False,
    )
    return out
