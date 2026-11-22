# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Modified or added by Ciru for Inkling 2 Mini on AMD gfx1151.
"""Grouped Inkling prefill projections with in-register weight decoding."""

from dataclasses import dataclass
import os

import torch
from triton.language.extra import libdevice

from vllm.triton_utils import tl, triton

from .ops.expert_e2m1_g16 import _e2m1_doubled_pair
from .ops.expert_nvfp4 import _positive_e4m3fn


BLOCK_M, BLOCK_N, BLOCK_K = 16, 32, 64
WIDE_BLOCK_M, WIDE_BLOCK_N = 32, 64


@dataclass(frozen=True)
class PrefillRoutes:
    sorted_ids: torch.Tensor
    expert_ids: torch.Tensor
    padded_tokens: torch.Tensor
    num_tokens: int
    top_k: int
    num_experts: int
    block_m: int = BLOCK_M
    block_n: int = BLOCK_N
    block_k: int = BLOCK_K


def align_prefill_routes(
    route_ids: torch.Tensor,
    num_experts: int,
    *,
    packed: tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None = None,
) -> PrefillRoutes:
    """Create the expert-block index once for both projections."""
    if route_ids.ndim != 2 or not route_ids.numel() or not route_ids.is_contiguous():
        raise ValueError("expected contiguous nonempty [tokens, top_k] routes")
    if route_ids.dtype not in (torch.int32, torch.int64) or num_experts <= 0:
        raise ValueError("expected integer route IDs and a positive expert count")
    # Increase route reuse only when the average expert has at least one
    # baseline tile of routes. For short chunks, widen output columns only.
    wide = (os.getenv("INKLING_AMD_PREFILL_WIDE_TILE") == "1"
            and route_ids.numel() >= num_experts * BLOCK_M)
    short_min_rows = 1 if os.getenv("INKLING_AMD_PREFILL_WIDE_SMALL_BATCH") == "1" else 64
    short = short_min_rows <= route_ids.shape[0] < 256
    short_n64 = short and os.getenv("INKLING_AMD_PREFILL_SHORT_N64") == "1"
    turbo_prefill = os.getenv("INKLING_AMD_TURBO_PREFILL_MOE", "0") not in ("", "0")
    block_m = WIDE_BLOCK_M if wide else BLOCK_M
    block_n = WIDE_BLOCK_N if wide or short_n64 else BLOCK_N
    block_k = BLOCK_K
    if turbo_prefill and packed is None:
        if route_ids.shape[0] >= 256:
            block_m, block_n, block_k = 64, 128, 64
        elif route_ids.shape[0] >= 2:
            block_m, block_n, block_k = 16, 128, 64
    if short:
        short_n = os.getenv("INKLING_AMD_PREFILL_SHORT_N")
        if short_n:
            block_n = int(short_n)
            if block_n not in (32, 64, 128, 256):
                raise ValueError("Inkling short prefill N tile must be 32, 64, 128 or 256")
        elif os.getenv("INKLING_AMD_PREFILL_SHORT_N128") == "1":
            block_n = 128
        short_k = os.getenv("INKLING_AMD_PREFILL_SHORT_K")
        if short_k:
            block_k = int(short_k)
            if block_k not in (32, 64, 128):
                raise ValueError("Inkling short prefill K tile must be 32, 64 or 128")
        elif os.getenv("INKLING_AMD_PREFILL_SHORT_K128") == "1":
            block_k = 128
    if packed is not None and block_m == 16:
        sorted_ids, expert_ids, padded = packed
    else:
        from vllm.model_executor.layers.fused_moe.moe_align_block_size import (
            moe_align_block_size,
        )

        sorted_ids, expert_ids, padded = moe_align_block_size(
            route_ids, block_m, num_experts, pad_sorted_ids=True,
        )
    return PrefillRoutes(
        sorted_ids, expert_ids, padded, *route_ids.shape, num_experts, block_m, block_n, block_k,
    )


@triton.jit
def _weight_tile(
    Packed, Raw, RowScale, Table, bank_rows, out_mask, cols, packed_cols, outer,
    expert, K: tl.constexpr, KIND: tl.constexpr, NORMAL_SCALES: tl.constexpr,
):
    valid = out_mask[:, None] & (cols[None, :] < K)
    if KIND == 3:
        data = tl.load(Packed + bank_rows[:, None] * K + cols[None, :], valid, 0)
        data = data.to(tl.int32)
        magnitude = _positive_e4m3fn(data & 127)
        values = tl.where((data & 128) != 0, -magnitude, magnitude)
        raw = tl.load(
            Raw + bank_rows[:, None] * (K // 32) + cols[None, :] // 32, valid, 0,
        ).to(tl.uint32)
        # The fp32 exponent field shares the E8M0 bias. Code zero is 2^-127.
        bits = tl.where(raw == 0, 0x00400000, raw << 23).to(tl.uint32)
        weight = values * bits.to(tl.float32, bitcast=True)
    else:
        packed = tl.load(
            Packed + bank_rows[:, None] * (K // 2) + packed_cols[None, :],
            out_mask[:, None] & (packed_cols[None, :] < K // 2), 0,
        ).to(tl.int32)
        if KIND == 0:
            low, high = _e2m1_doubled_pair(packed.to(tl.uint32))
            values = tl.interleave(low, high) * 0.5
            raw = tl.load(
                Raw + bank_rows[:, None] * (K // 16) + cols[None, :] // 16, valid, 0,
            ).to(tl.int32)
            if NORMAL_SCALES:
                bits = ((raw << 4) + (120 << 7)).to(tl.uint16)
                scale = bits.to(tl.bfloat16, bitcast=True).to(tl.float32)
            else:
                scale = _positive_e4m3fn(raw)
            weight = (values * scale) * outer
        elif KIND == 1:
            codes = tl.interleave(packed & 15, packed >> 4)
            values = ((codes ^ 8) - 8).to(tl.float32)
            scales = Raw.to(tl.pointer_type(tl.bfloat16))
            scale = tl.load(
                scales + bank_rows[:, None] * (K // 32) + cols[None, :] // 32, valid, 0,
            ).to(tl.float32)
            weight = values * scale
        else:
            codes = tl.interleave(packed & 15, packed >> 4).to(tl.float32)
            tags = tl.load(
                Raw + bank_rows[:, None] * (K // 16) + cols[None, :] // 16, valid, 0,
            ).to(tl.int32)
            entries = Table + expert * 512 + 2 * tags
            row = tl.load(RowScale + bank_rows, out_mask, 0).to(tl.float32)
            scale = tl.load(entries, valid, 0).to(tl.float32) * row[:, None]
            offset = tl.load(entries + 1, valid, 0).to(tl.float32) * row[:, None]
            weight = codes * scale + offset
    return weight


@triton.jit
def _project_tile(
    X, Packed, Raw, Outer, RowScale, Table, expert, rows, row_mask, out_cols,
    N: tl.constexpr, K: tl.constexpr, M_TILE: tl.constexpr, N_TILE: tl.constexpr,
    K_TILE: tl.constexpr, KIND: tl.constexpr, NORMAL_SCALES: tl.constexpr,
):
    bank_rows = expert.to(tl.int64) * N + out_cols
    out_mask = out_cols < N
    col = tl.arange(0, K_TILE)
    byte = tl.arange(0, K_TILE // 2)
    acc = tl.zeros((M_TILE, N_TILE), tl.float32)
    outer = 1.0
    if KIND == 0:
        outer = tl.load(Outer + expert)
    for group in range(tl.cdiv(K, K_TILE)):
        col_ids = group * K_TILE + col
        packed_ids = group * (K_TILE // 2) + byte
        activations = tl.load(
            X + rows[:, None] * K + col_ids[None, :],
            row_mask[:, None] & (col_ids[None, :] < K), 0,
        )
        weight = _weight_tile(
            Packed, Raw, RowScale, Table, bank_rows, out_mask, col_ids,
            packed_ids, outer, expert, K, KIND, NORMAL_SCALES,
        ).to(tl.bfloat16, fp_downcast_rounding="rtne")
        acc = tl.dot(activations, tl.trans(weight), acc)
    return acc


@triton.jit
def _packed_prefill_kernel(
    X, Packed, Raw, Outer, Kind, RowScale, Table, Sorted, Expert, Padded, Y, Gamma,
    N: tl.constexpr, K: tl.constexpr, ROUTE_COUNT, TOKENS_PER_INPUT: tl.constexpr,
    M_TILE: tl.constexpr, N_TILE: tl.constexpr, K_TILE: tl.constexpr,
    MIXED: tl.constexpr, HAS_FLASH: tl.constexpr, NORMAL_SCALES: tl.constexpr,
    MXFP8: tl.constexpr, FUSED_ACTIVATION: tl.constexpr,
    GATE_CAP: tl.constexpr, UP_CAP: tl.constexpr,
):
    m = tl.program_id(0)
    n = tl.program_id(1)
    if m * M_TILE >= tl.load(Padded):
        return
    route = tl.load(Sorted + m * M_TILE + tl.arange(0, M_TILE)).to(tl.int64)
    row_mask = (route >= 0) & (route < ROUTE_COUNT)
    rows = route // TOKENS_PER_INPUT
    out_cols = n * N_TILE + tl.arange(0, N_TILE)
    expert = tl.load(Expert + m)
    if expert < 0:
        return

    if MXFP8 or not MIXED:
        acc = _project_tile(
            X, Packed, Raw, Outer, RowScale, Table, expert, rows, row_mask,
            out_cols, N, K, M_TILE, N_TILE, K_TILE, 3 if MXFP8 else 0, NORMAL_SCALES,
        )
    else:
        kind = tl.load(Kind + expert)
        if kind == 0:
            acc = _project_tile(
                X, Packed, Raw, Outer, RowScale, Table, expert, rows, row_mask,
                out_cols, N, K, M_TILE, N_TILE, K_TILE, 0, NORMAL_SCALES,
            )
        elif kind == 1:
            acc = _project_tile(
                X, Packed, Raw, Outer, RowScale, Table, expert, rows, row_mask,
                out_cols, N, K, M_TILE, N_TILE, K_TILE, 1, NORMAL_SCALES,
            )
        else:
            acc = tl.full((M_TILE, N_TILE), float("nan"), tl.float32)
            if HAS_FLASH:
                if kind == 2:
                    acc = _project_tile(
                        X, Packed, Raw, Outer, RowScale, Table, expert, rows, row_mask,
                        out_cols, N, K, M_TILE, N_TILE, K_TILE, 2, NORMAL_SCALES,
                    )
    projected = acc.to(tl.bfloat16, fp_downcast_rounding="rtne")
    if FUSED_ACTIVATION:
        # W13 is interleaved. Preserve the BF16 projection boundary and
        # the FP32 multiply order of Inkling2MoE.activate.
        gate, up = tl.split(tl.reshape(projected, (M_TILE, N_TILE // 2, 2)))
        gate = gate.to(tl.float32)
        up = up.to(tl.float32)
        gamma = tl.load(Gamma + route, row_mask, other=0)
        sigmoid = tl.div_rn(1.0, 1.0 + libdevice.exp(-gate))
        mid = GATE_CAP * libdevice.tanh(gate * (1.0 / GATE_CAP))
        mid = ((mid * sigmoid) * UP_CAP) * libdevice.tanh(up * (1.0 / UP_CAP))
        mid = (mid * gamma[:, None]).to(tl.bfloat16, fp_downcast_rounding="rtne")
        cols = n * (N_TILE // 2) + tl.arange(0, N_TILE // 2)
        tl.store(
            Y + route[:, None] * (N // 2) + cols[None, :], mid,
            row_mask[:, None] & (cols[None, :] < N // 2),
        )
    else:
        tl.store(
            Y + route[:, None] * N + out_cols[None, :], projected,
            row_mask[:, None] & (out_cols[None, :] < N),
        )


def packed_prefill_projection(
    x, packed, raw_scales, outer_scales, plan: PrefillRoutes, *, kind=None,
    flash_aux=None, normal_scales=True, weight_format="nvfp4", fused_activation=None,
) -> torch.Tensor:
    """Project BF16 `[T,K]` or `[T,top_k,K]` into original route-slot order.

    The NVFP4 bytes and mixed format metadata match the M1 decoder. MXFP8 uses
    E4M3FN `[E,N,K]` and biased U8 `[E,N,K/32]` scales. Weight tiles round to
    BF16 before multiplication, as in the eager expert fallback.
    For paired W13 rows, `fused_activation=(gamma, gate_cap, up_cap)`
    returns BF16 W2 inputs `[T,top_k,N/2]` instead of projected gate/up pairs.
    """
    if weight_format not in ("nvfp4", "mxfp8") or packed.ndim != 3:
        raise ValueError("expected an NVFP4 or MXFP8 expert bank")
    mxfp8 = weight_format == "mxfp8"
    experts, rows, stored_width = packed.shape
    width = stored_width if mxfp8 else 2 * stored_width
    token_shape = (plan.num_tokens, plan.top_k)
    expected_dtype = torch.float8_e4m3fn if mxfp8 else torch.uint8
    if packed.dtype != expected_dtype or experts != plan.num_experts or not rows:
        raise ValueError("packed weights do not match the route plan or dtype")
    if x.shape not in ((plan.num_tokens, width), (*token_shape, width)):
        raise ValueError("activation must be [T,K] or [T,top_k,K]")
    if x.dtype != torch.bfloat16 or not width or width % 32:
        raise ValueError("expected BF16 activation and K divisible by 32")
    groups = width // (32 if mxfp8 else 16)
    if raw_scales.shape != (experts, rows, groups) or raw_scales.dtype != torch.uint8:
        raise ValueError("raw scales do not match the packed weights")
    operands = [x, packed, raw_scales, plan.sorted_ids, plan.expert_ids, plan.padded_tokens]
    row_scales = table = None
    if not mxfp8:
        if (outer_scales is None or outer_scales.shape != (experts,)
                or outer_scales.dtype != torch.float32):
            raise ValueError("expected NVFP4 FP32 outer scales [E]")
        operands.append(outer_scales)
    if kind is not None:
        if mxfp8 or kind.shape != (experts,) or kind.dtype != torch.int32:
            raise ValueError("expected NVFP4 format kinds int32 [E]")
        operands.append(kind)
    if flash_aux is not None:
        row_scales, table = flash_aux
        if kind is None or row_scales.shape != (experts, rows) or table.shape != (experts, 256, 2):
            raise ValueError("expected mixed FP16 row scales and affine tables")
        if row_scales.dtype != torch.float16 or table.dtype != torch.float16:
            raise ValueError("expected FP16 Flash C1 metadata")
        operands.extend((row_scales, table))
    gamma, gate_cap, up_cap = None, 0.0, 0.0
    output_rows = rows
    if fused_activation is not None:
        gamma, gate_cap, up_cap = fused_activation
        if (x.ndim != 2 or rows % 2 or gamma.shape != token_shape
                or gamma.dtype != torch.float32 or gate_cap <= 0 or up_cap <= 0):
            raise ValueError("expected paired W13 rows, FP32 per-route weights and positive softcaps")
        operands.append(gamma)
        output_rows = rows // 2
    if any(not t.is_contiguous() or t.device != x.device for t in operands):
        raise ValueError("all prefill operands must be contiguous on one device")
    if kind is not None and raw_scales.data_ptr() % 2:
        raise ValueError("mixed raw scales must be aligned for a BF16 view")

    out = x.new_empty((*token_shape, output_rows))
    _packed_prefill_kernel[(plan.expert_ids.numel(), triton.cdiv(rows, plan.block_n))](
        x, packed.view(torch.uint8), raw_scales, outer_scales, kind, row_scales, table,
        plan.sorted_ids, plan.expert_ids, plan.padded_tokens, out, gamma,
        N=rows, K=width, ROUTE_COUNT=plan.num_tokens * plan.top_k,
        TOKENS_PER_INPUT=plan.top_k if x.ndim == 2 else 1,
        M_TILE=plan.block_m, N_TILE=plan.block_n, K_TILE=plan.block_k, MIXED=kind is not None,
        HAS_FLASH=flash_aux is not None, NORMAL_SCALES=normal_scales, MXFP8=mxfp8,
        FUSED_ACTIVATION=fused_activation is not None, GATE_CAP=gate_cap, UP_CAP=up_cap,
        num_warps=4, num_stages=1, enable_fp_fusion=False,
    )
    return out
