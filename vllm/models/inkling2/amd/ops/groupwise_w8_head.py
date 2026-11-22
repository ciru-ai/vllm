# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Modified or added by Ciru for Inkling 2 Mini on AMD gfx1151.
"""Inkling GEMVs with signed I8 weights and K128 FP16 scales."""

import torch
from triton.experimental import gluon
from triton.experimental.gluon import language as gl

from vllm.triton_utils import triton


_WIDTHS = (2304, 3072)
_GROUP_SIZE = 128
_MARKOV_WIDTH = 2 * _GROUP_SIZE
_ROWS = 4


@torch.no_grad()
def pack_groupwise_w8(weight: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Pack BF16 weights once, in row chunks to bound the FP32 workspace."""
    if (
        weight.ndim != 2
        or weight.shape[1] not in (*_WIDTHS, _MARKOV_WIDTH)
        or weight.dtype != torch.bfloat16
    ):
        raise ValueError("Expected BF16 weights with width 256, 2304 or 3072")

    n, width = weight.shape
    group_count = width // _GROUP_SIZE
    codes = torch.empty((n, width), dtype=torch.int8, device=weight.device)
    scales = torch.empty((n, group_count), dtype=torch.float16, device=weight.device)
    chunk_rows = 2048
    min_scale = torch.finfo(torch.float16).tiny * torch.finfo(torch.float16).eps
    for start in range(0, n, chunk_rows):
        stop = min(start + chunk_rows, n)
        values = weight[start:stop].float().reshape(-1, group_count, _GROUP_SIZE)
        group_scales = (values.abs().amax(dim=-1) / 127).clamp_min_(min_scale).half()
        values.div_(group_scales.float().unsqueeze(-1)).round_().clamp_(-127, 127)
        codes[start:stop].copy_(values.reshape(-1, width))
        scales[start:stop].copy_(group_scales)
    return codes, scales


@gluon.jit
def _groupwise_w8_tile(
    X, Q, S, Y, row_start, N: gl.constexpr, GROUPS: gl.constexpr, GROUP_SIZE: gl.constexpr,
    ROWS: gl.constexpr,
):
    # Each wave32 owns one output row and each lane loads four adjacent codes.
    # Replicate the activation layout across waves to keep them independent.
    tile: gl.constexpr = gl.BlockedLayout([1, 4], [1, 32], [4, 1], [1, 0])
    rows = row_start + gl.arange(0, ROWS, gl.SliceLayout(1, tile))
    cols = gl.arange(0, GROUP_SIZE, gl.SliceLayout(0, tile))
    valid = rows < N if N % ROWS else gl.full((ROWS,), True, gl.int1, gl.SliceLayout(1, tile))
    partial = gl.zeros((ROWS, GROUP_SIZE), gl.float32, tile)
    for group_start in range(0, GROUPS, 3):
        for offset in gl.static_range(3):
            group = group_start + offset
            k = group * GROUP_SIZE + cols
            x = gl.load(X + k).to(gl.float32)
            q = gl.load(
                Q + rows[:, None] * (GROUPS * GROUP_SIZE) + k[None, :],
                valid[:, None], other=0,
            ).to(gl.float32)
            scale = gl.load(S + rows * GROUPS + group, valid, other=0).to(gl.float32)
            partial += (q * x[None, :]) * scale[:, None]
    logits = gl.sum(partial, axis=1)
    if Y.dtype.element_ty == gl.bfloat16:
        logits = logits.to(gl.bfloat16, fp_downcast_rounding="rtne")
    gl.store(Y + rows, logits, valid)


@gluon.jit
def _groupwise_w8_head_gemv(
    X, Q, S, Y, N: gl.constexpr, GROUPS: gl.constexpr, GROUP_SIZE: gl.constexpr,
    ROWS: gl.constexpr,
):
    _groupwise_w8_tile(X, Q, S, Y, gl.program_id(0) * ROWS, N, GROUPS, GROUP_SIZE, ROWS)


@gluon.jit
def _groupwise_w8_qkv_gemv(
    XQ, XK, XV, Q, S, Y, Q_END: gl.constexpr, K_END: gl.constexpr, N: gl.constexpr,
    GROUPS: gl.constexpr, GROUP_SIZE: gl.constexpr, ROWS: gl.constexpr,
):
    row_start = gl.program_id(0) * ROWS
    # Q and K boundaries are workgroup-aligned, so the activation is uniform.
    x = gl.where(row_start < Q_END, XQ, gl.where(row_start < K_END, XK, XV))
    _groupwise_w8_tile(x, Q, S, Y, row_start, N, GROUPS, GROUP_SIZE, ROWS)


def groupwise_w8_gemv(
    x: torch.Tensor,
    codes: torch.Tensor,
    scales: torch.Tensor,
    *,
    out_dtype: torch.dtype = torch.bfloat16,
) -> torch.Tensor:
    """Project one BF16 token using packed weights with the same width."""
    if x.ndim != 2 or x.shape[0] != 1 or x.shape[1] not in _WIDTHS or x.dtype != torch.bfloat16:
        raise ValueError("Expected one BF16 activation row of width 2304 or 3072")
    if out_dtype not in (torch.bfloat16, torch.float32):
        raise ValueError("Projection output dtype must be BF16 or FP32")
    width = x.shape[1]
    group_count = width // _GROUP_SIZE
    if (
        codes.ndim != 2 or codes.shape[1] != width or codes.dtype != torch.int8
        or scales.shape != (codes.shape[0], group_count) or scales.dtype != torch.float16
        or codes.device != x.device or scales.device != x.device
        or not codes.is_contiguous() or not scales.is_contiguous()
    ):
        raise ValueError("Expected contiguous packed I8 codes and FP16 group scales")

    n = codes.shape[0]
    if not x.is_cuda:
        groups = codes.reshape(n, group_count, _GROUP_SIZE).float()
        partial = (groups * x.float().reshape(group_count, _GROUP_SIZE)).sum(dim=-1)
        return (partial * scales.float()).sum(dim=-1, keepdim=True).T.to(out_dtype)

    out = x.new_empty((1, n), dtype=out_dtype)
    if n:
        _groupwise_w8_head_gemv[(triton.cdiv(n, _ROWS),)](
            x.contiguous(), codes, scales, out, n, group_count, _GROUP_SIZE, _ROWS,
            num_warps=4,
        )
    return out


def groupwise_w8_qkv_gemv(
    x_q: torch.Tensor, x_k: torch.Tensor, x_v: torch.Tensor,
    codes: torch.Tensor, scales: torch.Tensor, ends: tuple[int, int, int],
) -> torch.Tensor:
    """Project three M1 inputs over contiguous row ranges in a shared W8 bank."""
    q_end, k_end, v_end = ends
    if codes.ndim != 2 or codes.shape[1] not in _WIDTHS or codes.dtype != torch.int8:
        raise ValueError("Expected packed W8 weights of width 2304 or 3072")
    if not (0 < q_end < k_end < v_end <= codes.shape[0]) or any(end % _ROWS for end in ends):
        raise ValueError("Q/K/V row ranges must end on four-row boundaries")
    width = codes.shape[1]
    group_count = width // _GROUP_SIZE
    if (
        scales.shape != (codes.shape[0], group_count) or scales.dtype != torch.float16
        or scales.device != codes.device or not scales.is_contiguous() or not codes.is_contiguous()
    ):
        raise ValueError("Expected contiguous packed W8 weights and FP16 scales")
    activations = (x_q, x_k, x_v)
    if any(
        x.shape != (1, width) or x.dtype != torch.bfloat16
        or x.device != codes.device or not x.is_contiguous()
        for x in activations
    ):
        raise ValueError("Expected three contiguous BF16 activation rows on the packed weight device")

    if not x_q.is_cuda:
        return torch.cat([
            groupwise_w8_gemv(x, codes[start:end], scales[start:end])
            for x, start, end in zip(activations, (0, q_end, k_end), ends)
        ], dim=-1)

    out = x_q.new_empty((1, v_end))
    _groupwise_w8_qkv_gemv[(v_end // _ROWS,)](
        x_q, x_k, x_v, codes, scales, out, q_end, k_end, v_end,
        group_count, _GROUP_SIZE, _ROWS, num_warps=4,
    )
    return out


@gluon.jit
def _groupwise_w8_markov_gemv(
    X, Q, S, Bias, Y, N: gl.constexpr, GROUP_SIZE: gl.constexpr, ROWS: gl.constexpr,
):
    layout: gl.constexpr = gl.BlockedLayout([1, 4], [1, 32], [4, 1], [1, 0])
    rows = gl.program_id(0) * ROWS + gl.arange(0, ROWS, gl.SliceLayout(1, layout))
    cols = gl.arange(0, GROUP_SIZE, gl.SliceLayout(0, layout))
    valid = rows < N
    partial = gl.zeros((ROWS, GROUP_SIZE), gl.float32, layout)
    for group in gl.static_range(2):
        k = group * GROUP_SIZE + cols
        x = gl.load(X + k).to(gl.float32)
        q = gl.load(
            Q + rows[:, None] * (2 * GROUP_SIZE) + k[None, :],
            valid[:, None],
            other=0,
        ).to(gl.float32)
        scale = gl.load(S + rows * 2 + group, valid, other=0).to(gl.float32)
        partial += (q * x[None, :]) * scale[:, None]
    bias = gl.load(Bias + rows, valid, other=0).to(gl.float32)
    out = gl.sum(partial, axis=1) + bias
    gl.store(Y + rows, out.to(gl.bfloat16, fp_downcast_rounding="rtne"), valid)


def groupwise_w8_markov_addmm(
    x: torch.Tensor, codes: torch.Tensor, scales: torch.Tensor, bias: torch.Tensor,
) -> torch.Tensor:
    """Project one rank-256 latent and add the folded bias before BF16 rounding."""
    if x.shape != (1, _MARKOV_WIDTH) or x.dtype != torch.bfloat16:
        raise ValueError("Expected one rank-256 BF16 Markov latent")
    n = bias.numel()
    if (
        bias.shape != (n,)
        or bias.dtype != torch.bfloat16
        or codes.shape != (n, _MARKOV_WIDTH)
        or codes.dtype != torch.int8
        or scales.shape != (n, 2)
        or scales.dtype != torch.float16
        or any(t.device != x.device or not t.is_contiguous() for t in (codes, scales, bias))
    ):
        raise ValueError("Expected contiguous folded W8 weights, FP16 scales and BF16 bias")

    if not x.is_cuda:
        groups = codes.reshape(n, 2, _GROUP_SIZE).float()
        dots = (groups * x.float().reshape(2, _GROUP_SIZE)).sum(dim=-1)
        return ((dots * scales.float()).sum(dim=1) + bias.float()).to(x.dtype)[None, :]

    out = x.new_empty((1, n))
    if n:
        _groupwise_w8_markov_gemv[(triton.cdiv(n, _ROWS),)](
            x.contiguous(), codes, scales, bias, out, n, _GROUP_SIZE, _ROWS, num_warps=4,
        )
    return out


pack_groupwise_w8_head = pack_groupwise_w8
groupwise_w8_head_linear = groupwise_w8_gemv
