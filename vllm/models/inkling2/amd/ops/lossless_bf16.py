# SPDX-License-Identifier: Apache-2.0
# Modified or added by Ciru for Inkling 2 Mini on AMD gfx1151.
"""Bit-exact BF16 storage and a packed single-row gfx1151 projection."""

from dataclasses import dataclass

import torch

from vllm.triton_utils import tl, triton


TILE_WIDTH = 256


@dataclass(frozen=True)
class LosslessBF16Weights:
    sign_fraction: torch.Tensor
    exponent_pairs: torch.Tensor
    raw_bits: torch.Tensor
    descriptors: torch.Tensor

    @property
    def storage_bytes(self) -> int:
        return sum(
            tensor.nbytes
            for tensor in (
                self.sign_fraction, self.exponent_pairs, self.raw_bits, self.descriptors
            )
        )

    def to(self, device: torch.device | str) -> "LosslessBF16Weights":
        return LosslessBF16Weights(
            self.sign_fraction.to(device),
            self.exponent_pairs.to(device),
            self.raw_bits.to(device),
            self.descriptors.to(device),
        )


def pack_lossless_bf16(weight: torch.Tensor) -> LosslessBF16Weights:
    """Pack each tile whose exponent span fits in four bits.

    A nonnegative descriptor holds the compact tile index above its eight-bit
    base exponent. A negative descriptor is minus the raw tile index minus one.
    Both paths preserve the original BF16 sign, exponent and fraction bits.
    """
    if weight.device.type != "cpu" or weight.dtype != torch.bfloat16 or weight.ndim != 2:
        raise ValueError("expected a CPU BF16 weight matrix")
    rows, width = weight.shape
    if not rows or not width or width % TILE_WIDTH:
        raise ValueError("weight width must be a positive multiple of 256")
    if weight.numel() // TILE_WIDTH > (1 << 23):
        raise ValueError("weight matrix exceeds the descriptor index capacity")

    tiles = weight.contiguous().view(torch.int16).view(-1, TILE_WIDTH)
    sign_fractions, exponent_pairs, raw_blocks, descriptors = [], [], [], []
    packed_count = raw_count = 0
    for tile in tiles.split(8192):
        exponent = (tile >> 7) & 255
        low, high = torch.aminmax(exponent, dim=1)
        fits = high - low <= 15
        packed_tile = tile[fits]
        raw_tile = tile[~fits]
        base = low[fits]
        offsets = exponent[fits] - base[:, None]

        sign_fractions.append(
            ((packed_tile & 127) | ((packed_tile >> 8) & 128)).to(torch.uint8)
        )
        exponent_pairs.append((offsets[:, ::2] | (offsets[:, 1::2] << 4)).to(torch.uint8))
        raw_blocks.append(raw_tile)
        n_packed, n_raw = len(packed_tile), len(raw_tile)
        descriptor = torch.empty(len(tile), dtype=torch.int32)
        descriptor[fits] = (
            torch.arange(packed_count, packed_count + n_packed, dtype=torch.int32) << 8
        ) | base.to(torch.int32)
        descriptor[~fits] = -torch.arange(
            raw_count + 1, raw_count + n_raw + 1, dtype=torch.int32
        )
        descriptors.append(descriptor)
        packed_count += n_packed
        raw_count += n_raw

    return LosslessBF16Weights(
        torch.cat(sign_fractions),
        torch.cat(exponent_pairs),
        torch.cat(raw_blocks),
        torch.cat(descriptors).view(rows, width // TILE_WIDTH),
    )


def unpack_lossless_bf16(packed: LosslessBF16Weights) -> torch.Tensor:
    """Reconstruct BF16 on CPU for an exact bit comparison of the format."""
    if packed.descriptors.device.type != "cpu":
        raise ValueError("round-trip validation uses a CPU packed matrix")
    descriptors = packed.descriptors.flatten()
    fits = descriptors >= 0
    encoded = descriptors[fits]
    indices = (encoded >> 8).long()
    sign_fraction = packed.sign_fraction[indices].to(torch.int32)
    pairs = packed.exponent_pairs[indices].to(torch.int32)
    offsets = torch.stack((pairs & 15, pairs >> 4), dim=-1).view(-1, TILE_WIDTH)
    exponent = (encoded & 255)[:, None] + offsets
    decoded = ((sign_fraction & 127) | ((sign_fraction & 128) << 8) | (exponent << 7))

    tiles = torch.empty((descriptors.numel(), TILE_WIDTH), dtype=torch.int16)
    tiles[fits] = decoded.to(torch.int16)
    tiles[~fits] = packed.raw_bits[(-descriptors[~fits] - 1).long()]
    return tiles.view(packed.descriptors.shape[0], -1).view(torch.bfloat16)


@triton.jit
def _packed_bf16_gemv(
    X, SignFraction, ExponentPairs, RawBits, Descriptors, Y,
    N: tl.constexpr, K: tl.constexpr,
    WIDTH: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    rows = tl.program_id(0) * BLOCK_N + tl.arange(0, BLOCK_N)
    columns = tl.arange(0, BLOCK_K)
    acc = tl.zeros((BLOCK_N, BLOCK_K), tl.float32)

    for block in range(tl.cdiv(K, BLOCK_K)):
        cols = block * BLOCK_K + columns
        valid = (rows[:, None] < N) & (cols[None, :] < K)
        descriptor = tl.load(
            Descriptors + rows[:, None] * (K // WIDTH) + cols[None, :] // WIDTH,
            valid,
            0,
        )
        offset = cols[None, :] % WIDTH
        compressed = descriptor >= 0
        packed_tile = descriptor >> 8
        base = descriptor & 255
        low = tl.load(SignFraction + packed_tile * WIDTH + offset, valid & compressed, 0)
        pair = tl.load(
            ExponentPairs + packed_tile * (WIDTH // 2) + offset // 2,
            valid & compressed,
            0,
        )
        exponent = base + ((pair.to(tl.int32) >> ((offset & 1) * 4)) & 15)
        low = low.to(tl.int32)
        reconstructed = ((low & 127) | ((low & 128) << 8) | (exponent << 7)).to(tl.uint16)
        raw = tl.load(
            RawBits + (-descriptor - 1) * WIDTH + offset,
            valid & ~compressed,
            0,
        ).to(tl.uint16)
        bits = tl.where(compressed, reconstructed, raw)
        values = bits.to(tl.bfloat16, bitcast=True).to(tl.float32)
        x = tl.load(X + cols, cols < K, 0).to(tl.float32)
        acc = tl.fma(values, x[None, :], acc)

    total = tl.sum(acc, axis=1)
    if Y.dtype.element_ty == tl.bfloat16:
        total = total.to(tl.bfloat16, fp_downcast_rounding="rtne")
    tl.store(Y + rows, total, rows < N)


@triton.jit
def _packed_bf16_gemv_paired(
    X, SignFraction, ExponentPairs, RawBits, Descriptors, Y,
    N: tl.constexpr, K: tl.constexpr,
    WIDTH: tl.constexpr, BLOCK_N: tl.constexpr,
):
    rows = tl.program_id(0) * BLOCK_N + tl.arange(0, BLOCK_N)
    pairs = tl.arange(0, WIDTH // 2)
    acc = tl.zeros((BLOCK_N, WIDTH // 2), tl.float32)
    sign_pairs = SignFraction.to(tl.pointer_type(tl.uint16))
    raw_pairs = RawBits.to(tl.pointer_type(tl.uint32))
    input_pairs = X.to(tl.pointer_type(tl.uint32))

    for tile in range(K // WIDTH):
        descriptor = tl.load(Descriptors + rows * (K // WIDTH) + tile, rows < N, 0)
        compressed = descriptor >= 0
        packed_offset = (descriptor >> 8)[:, None] * (WIDTH // 2) + pairs[None, :]
        packed_mask = ((rows < N) & compressed)[:, None]
        sign_fraction = tl.load(sign_pairs + packed_offset, packed_mask, 0).to(tl.uint32)
        exponent_pair = tl.load(ExponentPairs + packed_offset, packed_mask, 0).to(tl.uint32)
        exponent_base = (descriptor & 255).to(tl.uint32)[:, None]
        even_exp = exponent_base + (exponent_pair & 15)
        odd_exp = exponent_base + (exponent_pair >> 4)
        even_sf = sign_fraction & 255
        odd_sf = sign_fraction >> 8
        even_bits = (even_sf & 127) | ((even_sf & 128) << 8) | (even_exp << 7)
        odd_bits = (odd_sf & 127) | ((odd_sf & 128) << 8) | (odd_exp << 7)

        raw_offset = (-descriptor - 1)[:, None] * (WIDTH // 2) + pairs[None, :]
        raw = tl.load(raw_pairs + raw_offset, ((rows < N) & ~compressed)[:, None], 0)
        even_bits = tl.where(compressed[:, None], even_bits, raw & 65535).to(tl.uint16)
        odd_bits = tl.where(compressed[:, None], odd_bits, raw >> 16).to(tl.uint16)
        even_weight = even_bits.to(tl.bfloat16, bitcast=True).to(tl.float32)
        odd_weight = odd_bits.to(tl.bfloat16, bitcast=True).to(tl.float32)

        x_pair = tl.load(input_pairs + tile * (WIDTH // 2) + pairs)
        even_x = x_pair.to(tl.uint16).to(tl.bfloat16, bitcast=True).to(tl.float32)
        odd_x = (x_pair >> 16).to(tl.uint16).to(tl.bfloat16, bitcast=True).to(tl.float32)
        acc = tl.fma(even_weight, even_x[None, :], acc)
        acc = tl.fma(odd_weight, odd_x[None, :], acc)

    total = tl.sum(acc, axis=1)
    if Y.dtype.element_ty == tl.bfloat16:
        total = total.to(tl.bfloat16, fp_downcast_rounding="rtne")
    tl.store(Y + rows, total, rows < N)


def lossless_bf16_gemv(
    x: torch.Tensor,
    packed: LosslessBF16Weights,
    *,
    variant: str = "v1",
    out_dtype: torch.dtype = torch.bfloat16,
) -> torch.Tensor:
    """Project one BF16 row; weight reconstruction stays on chip."""
    rows, tiles_per_row = packed.descriptors.shape
    width = tiles_per_row * TILE_WIDTH
    if (
        not x.is_cuda
        or x.device != packed.descriptors.device
        or x.dtype != torch.bfloat16
        or x.shape != (1, width)
        or not x.is_contiguous()
    ):
        raise ValueError("expected one contiguous GPU BF16 input row")
    if out_dtype not in (torch.bfloat16, torch.float32):
        raise ValueError("expected BF16 or FP32 projection output")
    out = x.new_empty((1, rows), dtype=out_dtype)
    block_rows = 4
    grid = (triton.cdiv(rows, block_rows),)
    if variant == "paired":
        if x.data_ptr() % 4:
            raise ValueError("the paired kernel requires a four-byte aligned input")
        _packed_bf16_gemv_paired[grid](
            x, packed.sign_fraction, packed.exponent_pairs, packed.raw_bits,
            packed.descriptors, out, rows, width, TILE_WIDTH, block_rows, num_warps=4,
        )
    elif variant == "v1":
        _packed_bf16_gemv[grid](
            x, packed.sign_fraction, packed.exponent_pairs, packed.raw_bits,
            packed.descriptors, out, rows, width,
            TILE_WIDTH, block_rows, 1024, num_warps=4,
        )
    else:
        raise ValueError(f"unknown lossless BF16 variant: {variant}")
    return out
