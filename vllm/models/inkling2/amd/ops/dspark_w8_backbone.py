# SPDX-License-Identifier: Apache-2.0
# Modified or added by Ciru for Inkling 2 Mini on AMD gfx1151.
"""gfx1151 M8 WMMA for W8-G128 Inkling DSpark draft projections."""

from typing import NamedTuple

import torch
import triton
import triton.language as tl
from triton.experimental import gluon
from triton.experimental.gluon import language as gl
from triton.experimental.gluon.language.amd import AMDWMMALayout
from triton.experimental.gluon.language.amd.rdna3 import wmma


_GROUP_SIZE = 128
_OUT_TILE = 64
_IN_TILE = 64
_SHAPES = ((18432, 2304), (2304, 9216), (5120, 2304), (2304, 3072))


class DraftW8Weight(NamedTuple):
    codes: torch.Tensor
    scales: torch.Tensor
    out_features: int
    in_features: int


@torch.no_grad()
def pack_draft_w8(weight: torch.Tensor) -> DraftW8Weight:
    """Quantize BF16 rows to signed W8-G128 and pack N64/K64 tiles."""
    if weight.shape not in _SHAPES or weight.dtype != torch.bfloat16:
        raise ValueError("Expected a TP1 BF16 Inkling draft projection weight")
    n, k = weight.shape
    groups = k // _GROUP_SIZE
    tiles = (n // _OUT_TILE, k // _IN_TILE, _OUT_TILE, _IN_TILE)
    codes = weight.new_empty(tiles, dtype=torch.int8)
    scales = weight.new_empty((tiles[0], groups, _OUT_TILE), dtype=torch.float16)
    fp16 = torch.finfo(torch.float16)
    for start in range(0, n, 1024):
        values = weight[start : start + 1024].float().reshape(-1, groups, _GROUP_SIZE)
        scale = (values.abs().amax(dim=-1) / 127).clamp_min_(fp16.tiny * fp16.eps)
        scale = scale.half()
        quantized = values.div_(scale.float().unsqueeze(-1)).round_().clamp_(-127, 127)
        nt = values.shape[0] // _OUT_TILE
        region = slice(start // _OUT_TILE, start // _OUT_TILE + nt)
        codes[region].copy_(quantized.reshape(nt, tiles[2], tiles[1], tiles[3]).transpose(1, 2))
        scales[region].copy_(scale.reshape(nt, _OUT_TILE, groups).transpose(1, 2))
    return DraftW8Weight(codes, scales, n, k)


@gluon.jit
def _dspark_w8_projection(X, Q, Scales, Y, N: gl.constexpr, K: gl.constexpr,
                          SPLITS: gl.constexpr):
    tile_n = gl.program_id(0)
    split = gl.program_id(1)
    groups: gl.constexpr = K // 128
    groups_per_split: gl.constexpr = groups // SPLITS
    mma: gl.constexpr = AMDWMMALayout(1, True, [[0, 1], [0, 2]])
    a_load: gl.constexpr = gl.BlockedLayout([1, 64], [16, 2], [1, 4], [0, 1])
    b_load: gl.constexpr = gl.DistributedLinearLayout(
        [[0, 1], [0, 2], [0, 4], [0, 8], [0, 16]],
        [[1, 0], [2, 0], [4, 0], [8, 0], [0, 32]],
        [[16, 0], [32, 0]], [], [64, 64],
    )
    x_row = gl.arange(0, 16, gl.SliceLayout(1, a_load))
    x_col = gl.arange(0, 64, gl.SliceLayout(0, a_load))
    q_row = gl.arange(0, 64, gl.SliceLayout(1, b_load))
    q_col = gl.arange(0, 64, gl.SliceLayout(0, b_load))
    scale_col = gl.arange(0, 64, gl.SliceLayout(0, mma))
    acc = gl.zeros((16, 64), gl.float32, mma)
    for step in range(groups_per_split):
        group = split * groups_per_split + step
        block = gl.zeros((16, 64), gl.float32, mma)
        for half in gl.static_range(2):
            k_tile = group * 2 + half
            x = gl.load(
                X + x_row[:, None] * K + k_tile * 64 + x_col[None, :],
                x_row[:, None] < 8, 0,
            )
            packet = (tile_n * (K // 64) + k_tile) * 4096
            q = gl.load(Q + packet + q_row[:, None] * 64 + q_col[None, :])
            a = gl.convert_layout(x, gl.DotOperandLayout(0, mma, 16), assert_trivial=True)
            b = gl.convert_layout(
                gl.permute(q.to(gl.bfloat16), (1, 0)), gl.DotOperandLayout(1, mma, 16),
            )
            block = wmma(a, b, block)
        scale = gl.load(Scales + (tile_n * groups + group) * 64 + scale_col)
        acc += block * scale[None, :].to(gl.float32)

    out_layout: gl.constexpr = gl.DistributedLinearLayout(
        [[0, 1], [0, 4], [0, 8]],
        [[1, 0], [2, 0], [4, 0], [8, 0], [0, 2]],
        [[0, 16], [0, 32]], [], [16, 64],
    )
    out_row = gl.arange(0, 16, gl.SliceLayout(1, out_layout))
    out_col = gl.arange(0, 64, gl.SliceLayout(0, out_layout))
    values = gl.convert_layout(acc, out_layout)
    if SPLITS == 1:
        values = values.to(gl.bfloat16, fp_downcast_rounding="rtne")
    gl.store(
        Y + (split * 8 + out_row[:, None]) * N + tile_n * 64 + out_col[None, :],
        values, out_row[:, None] < 8,
    )


@triton.jit
def _reduce_draft_w8(Partial, Y, SIZE: tl.constexpr, SPLITS: tl.constexpr,
                     BLOCK: tl.constexpr):
    col = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    split = tl.arange(0, SPLITS)
    part = tl.load(Partial + split[:, None] * SIZE + col[None, :], col[None, :] < SIZE, 0)
    tl.store(Y + col, tl.sum(part, axis=0).to(tl.bfloat16), col < SIZE)


def draft_w8_linear(
    x: torch.Tensor, weight: DraftW8Weight, *, splits: int | None = None,
) -> torch.Tensor:
    """Run M8 FP32 WMMA with W8 weights and return one BF16 projection."""
    n, k = weight.out_features, weight.in_features
    if x.shape != (8, k) or x.dtype != torch.bfloat16 or not x.is_contiguous():
        raise ValueError("Expected contiguous BF16 draft activations with eight rows")
    if not x.is_cuda or x.device != weight.codes.device or x.device != weight.scales.device:
        raise ValueError("Expected the packed weights on the activation's ROCm device")
    if splits is None:
        splits = 4 if k == 9216 else 1
    if splits not in (1, 2, 4, 8) or k % (_GROUP_SIZE * splits):
        raise ValueError("Split K must divide all G128 groups")
    out = torch.empty((8, n), device=x.device, dtype=torch.bfloat16)
    partial = out if splits == 1 else torch.empty(
        (splits, 8, n), device=x.device, dtype=torch.float32,
    )
    _dspark_w8_projection[(n // _OUT_TILE, splits)](
        x, weight.codes, weight.scales, partial, n, k, splits,
        num_warps=4, num_stages=1,
    )
    if splits != 1:
        _reduce_draft_w8[(triton.cdiv(8 * n, 256),)](
            partial, out, 8 * n, splits, 256, num_warps=4,
        )
    return out
