# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Row-major signed INT8 QKVR with BF16 activations and row scales."""

import torch

from vllm.triton_utils import tl, triton


@torch.no_grad()
def pack_qkvr_w8(weight: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    values = weight.float()
    scales = (values.abs().amax(dim=1) / 127).clamp_min_(
        torch.finfo(torch.bfloat16).tiny
    ).bfloat16()
    values.div_(scales.float()[:, None]).round_().clamp_(-127, 127)
    return values.to(torch.int8), scales


@triton.jit
def _qkvr_w8_gemv(
    X, Q, S, Y, N: tl.constexpr, K: tl.constexpr,
    BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    rows = tl.program_id(0) * BLOCK_N + tl.arange(0, BLOCK_N)
    cols = tl.arange(0, BLOCK_K)
    x = tl.load(X + cols, cols < K, 0).to(tl.float32)
    q = tl.load(
        Q + rows[:, None] * K + cols[None, :],
        (rows[:, None] < N) & (cols[None, :] < K), 0,
    ).to(tl.float32)
    scales = tl.load(S + rows, rows < N, 0).to(tl.float32)
    y = tl.sum(q * x[None, :], axis=1) * scales
    tl.store(Y + rows, y.to(tl.bfloat16, fp_downcast_rounding="rtne"), rows < N)


@triton.jit
def _qkvr_w8_gemm(
    X, Q, S, Y, M, N: tl.constexpr, K: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    tokens = tl.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M)
    rows = tl.program_id(1) * BLOCK_N + tl.arange(0, BLOCK_N)
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for tile in range(tl.cdiv(K, BLOCK_K)):
        cols = tile * BLOCK_K + tl.arange(0, BLOCK_K)
        x = tl.load(
            X + tokens[:, None] * K + cols[None, :],
            (tokens[:, None] < M) & (cols[None, :] < K), 0,
        )
        q = tl.load(
            Q + rows[:, None] * K + cols[None, :],
            (rows[:, None] < N) & (cols[None, :] < K), 0,
        ).to(tl.bfloat16)
        # Int8 codes convert exactly to BF16. Apply the row scale after the dot.
        acc = tl.dot(x, tl.trans(q), acc)
    scales = tl.load(S + rows, rows < N, 0).to(tl.float32)
    tl.store(
        Y + tokens[:, None] * N + rows[None, :],
        (acc * scales[None, :]).to(tl.bfloat16, fp_downcast_rounding="rtne"),
        (tokens[:, None] < M) & (rows[None, :] < N),
    )


def qkvr_w8_linear(x: torch.Tensor, q: torch.Tensor, scales: torch.Tensor) -> torch.Tensor:
    if not x.is_cuda or x.dtype != torch.bfloat16:
        return ((x.float() @ q.float().T) * scales.float()).to(x.dtype)

    m = x.shape[0]
    n, k = q.shape
    out = x.new_empty((m, n))
    if not m:
        return out
    x = x.contiguous()
    if m == 1:
        block_n = 2
        _qkvr_w8_gemv[(triton.cdiv(n, block_n),)](
            x, q, scales, out, n, k, block_n, triton.next_power_of_2(k), num_warps=4,
        )
    else:
        block_m = 16 if m <= 32 else 32
        block_n = 64
        _qkvr_w8_gemm[(triton.cdiv(m, block_m), triton.cdiv(n, block_n))](
            x, q, scales, out, m, n, k, block_m, block_n, 64, num_warps=4, num_stages=1,
        )
    return out
