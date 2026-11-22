"""Single-launch pre-tiled NVFP4 and fitted expert projections on gfx1151."""

from typing import NamedTuple

import torch
from triton.experimental import gluon
from triton.experimental.gluon import language as gl
from triton.experimental.gluon.language.amd import AMDWMMALayout
from triton.experimental.gluon.language.amd.rdna3 import wmma
from triton.experimental.gluon.language.extra import libdevice

from .coda_wmma_pretile import _c1_bf16_pair, _pack_wmma_packets
from .source_wmma_pretile import _finite_bf16, _unpack_e2m1


@gluon.jit
def _mixed_wmma_tile(
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
    acc = gl.zeros((16, 128), gl.float32, MMA)
    for k in range(K // 64):
        x = gl.load(
            X + rows[:, None] * K + k * 64 + a_cols[None, :], valid[:, None], 0,
        )
        packet_row = (tile + k) * 128 + b_row
        packed = gl.load(Packed + packet_row[:, None] * 32 + byte_cols[None, :]).to(gl.uint32)
        raw = gl.load(Raw + packet_row).to(gl.uint32)
        if KIND == 0:
            scale_byte = (raw[:, None] >> ((byte_cols[None, :] // 8) * 8)) & 255
            scale_bits = ((scale_byte << 4) + (120 << 7)).to(gl.uint16)
            scale = scale_bits.to(gl.bfloat16, bitcast=True).to(gl.float32)
            low, high = _unpack_e2m1(packed)
            low = _finite_bf16(((low * 0.5) * scale) * outer)
            high = _finite_bf16(((high * 0.5) * scale) * outer)
        elif KIND == 1:
            scale_bits = (raw[:, None] >> ((byte_cols[None, :] // 16) * 16)).to(gl.uint16)
            scale = scale_bits.to(gl.bfloat16, bitcast=True).to(gl.float32)
            low = (((packed & 15).to(gl.int32) ^ 8) - 8).to(gl.float32) * scale
            high = (((packed >> 4).to(gl.int32) ^ 8) - 8).to(gl.float32) * scale
            low = low.to(gl.bfloat16, fp_downcast_rounding="rtne")
            high = high.to(gl.bfloat16, fp_downcast_rounding="rtne")
        else:
            low, high = _c1_bf16_pair(packed, raw, row, Table, expert)
        weights = gl.reshape(gl.join(low, high), (128, 64))
        a = gl.convert_layout(x, gl.DotOperandLayout(0, MMA, 16), assert_trivial=True)
        b = gl.convert_layout(gl.permute(weights, (1, 0)), gl.DotOperandLayout(1, MMA, 16))
        acc = wmma(a, b, acc)
    return acc


@gluon.jit
def _mixed_wmma_pretiled(
    X, Packed, Raw, Outer, Kind, RowScale, Table, Sorted, Expert, Padded, Y, Gamma,
    N: gl.constexpr, K: gl.constexpr, ROUTES: gl.constexpr, TOKENS: gl.constexpr,
    FITTED_KIND: gl.constexpr, FUSED: gl.constexpr, GATE: gl.constexpr, UP: gl.constexpr,
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
    a_range = gl.arange(0, 16, gl.SliceLayout(1, a_load))
    route = gl.load(Sorted + m * 16 + a_range).to(gl.int64)
    valid = (route >= 0) & (route < ROUTES)
    rows = route // TOKENS
    # The route plan assigns one expert to each CTA, so every wave takes the
    # same branch before entering its K loop.
    if gl.load(Kind + expert) == 0:
        acc = _mixed_wmma_tile(
            X, Packed, Raw, Outer, RowScale, Table, expert, rows, valid, n,
            N, K, 0, mma, a_load,
        )
    else:
        acc = _mixed_wmma_tile(
            X, Packed, Raw, Outer, RowScale, Table, expert, rows, valid, n,
            N, K, FITTED_KIND, mma, a_load,
        )

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


class MixedWmmaBank(NamedTuple):
    packed: torch.Tensor
    raw_scales: torch.Tensor
    outer_scales: torch.Tensor
    kind: torch.Tensor
    row_scales: torch.Tensor | None
    table: torch.Tensor | None
    fitted_kind: int


def pack_mixed_wmma(packed, raw_scales, outer_scales, kind, *, fitted_kind, flash_aux=None):
    """Retain all bytes from the final source/fitted bank in N128/K64 tile order."""
    if packed.ndim != 3:
        raise ValueError("Expected packed expert weights [E,N,K/2]")
    experts, n, bytes_k = packed.shape
    k = bytes_k * 2
    if (packed.dtype != torch.uint8 or raw_scales.dtype != torch.uint8
            or raw_scales.shape != (experts, n, k // 16) or not experts or not n
            or outer_scales.shape != (experts,) or outer_scales.dtype != torch.float32
            or kind.shape != (experts,) or kind.dtype != torch.int32
            or k not in (1152, 2304) or n % 128 or fitted_kind not in (1, 2)):
        raise ValueError("Expected NVFP4 and one fitted format in an N128/K64 expert bank")
    row_scales = table = None
    operands = (packed, raw_scales, outer_scales, kind)
    if fitted_kind == 2:
        if flash_aux is None:
            raise ValueError("C1 experts require FP16 row scales and affine tables")
        row_scales, table = flash_aux
        if (row_scales.shape != (experts, n) or table.shape != (experts, 256, 2)
                or row_scales.dtype != torch.float16 or table.dtype != torch.float16):
            raise ValueError("Expected FP16 C1 row scales [E,N] and affine tables [E,256,2]")
        if table.data_ptr() % 4:
            raise ValueError("C1 affine table pairs must be aligned to four bytes")
        operands += (row_scales, table)
    if any(t.device != packed.device or not t.is_contiguous() for t in operands):
        raise ValueError("All mixed expert bank tensors must be contiguous on one device")
    source = kind == 0
    if not bool(torch.all(source | (kind == fitted_kind))):
        raise ValueError("The expert kinds must match source NVFP4 or the declared fitted format")
    raw_source = raw_scales[source]
    if raw_source.numel():
        lo, hi = torch.aminmax(raw_source)
        if not 8 <= int(lo) <= int(hi) <= 126:
            raise ValueError("Source experts require normal E4M3 scale bytes")
        max_outer = torch.finfo(torch.float32).max / (6 * 448)
        if not bool(torch.all(outer_scales[source].abs() <= max_outer)):
            raise ValueError("Expected finite reconstructed FP32 source weights")
    weights, raw = _pack_wmma_packets(packed, raw_scales)
    return MixedWmmaBank(weights, raw, outer_scales, kind, row_scales, table, fitted_kind)


def mixed_wmma_projection(x, bank: MixedWmmaBank, plan, *, fused_activation=None):
    """Project through the sorted routes with one WMMA launch per projection."""
    if (plan.block_m, plan.block_n, plan.block_k) != (16, 128, 64):
        raise ValueError("The mixed WMMA projection requires M16/N128/K64")
    experts, n_tiles, k_tiles, _, _ = bank.packed.shape
    n, k = n_tiles * 128, k_tiles * 64
    route_shape = (plan.num_tokens, plan.top_k)
    if (experts != plan.num_experts or x.dtype != torch.bfloat16
            or x.shape not in ((plan.num_tokens, k), (*route_shape, k))):
        raise ValueError("Expected BF16 mixed expert activations matching the route plan")
    gamma, gate_cap, up_cap = None, 0.0, 0.0
    routing = (plan.sorted_ids, plan.expert_ids, plan.padded_tokens)
    operands = (x, bank.packed, *routing)
    if fused_activation is not None:
        gamma, gate_cap, up_cap = fused_activation
        if (x.ndim != 2 or gamma.shape != route_shape or gamma.dtype != torch.float32
                or gate_cap <= 0 or up_cap <= 0):
            raise ValueError("Expected W13 input, FP32 route weights, and positive softcaps")
        operands += (gamma,)
    if any(t.device != x.device or not t.is_contiguous() for t in operands):
        raise ValueError("All mixed projection tensors must be contiguous on one device")
    out = x.new_empty((*route_shape, n // 2 if gamma is not None else n))
    _mixed_wmma_pretiled[(plan.expert_ids.numel(), n_tiles)](
        x, *bank[:-1], *routing, out, gamma,
        N=n, K=k, ROUTES=plan.num_tokens * plan.top_k,
        TOKENS=plan.top_k if x.ndim == 2 else 1, FITTED_KIND=bank.fitted_kind,
        FUSED=gamma is not None, GATE=gate_cap, UP=up_cap,
        num_warps=4, num_stages=1, enable_fp_fusion=False,
    )
    return out
