"""Exact Inkling E2M1 G16 weights with BF16 scales for AMD wave32."""

import torch
import triton
import triton.language as tl


@triton.jit
def _e2m1_doubled_pair(packed):
    # One byte-permute selects both magnitudes from {0, 1, 2, 3, 4, 6, 8, 12}.
    selector = (packed & 7) | ((packed << 4) & 0x0700)
    magnitudes = tl.inline_asm_elementwise(
        "v_perm_b32 $0, $1, $2, $3",
        constraints="=v,v,v,v",
        args=[0x0C080604, 0x03020100, selector],
        dtype=tl.uint32,
        is_pure=True,
        pack=1,
    )
    low = (magnitudes & 255).to(tl.float32).to(tl.uint32, bitcast=True)
    high = ((magnitudes >> 8) & 255).to(tl.float32).to(tl.uint32, bitcast=True)
    low = (low | ((packed & 8) << 28)).to(tl.float32, bitcast=True)
    high = (high | ((packed & 128) << 24)).to(tl.float32, bitcast=True)
    return low, high


@triton.jit
def _packed_e2m1_g16_kernel(
    X,
    Packed,
    Scales,
    Out,
    outer_scale,
    N: tl.constexpr,
    GROUPS: tl.constexpr,
    BLOCK_ROWS: tl.constexpr,
    BLOCK_GROUPS: tl.constexpr,
):
    group_bytes: tl.constexpr = 8
    rows = tl.program_id(0) * BLOCK_ROWS + tl.arange(0, BLOCK_ROWS)
    offsets = tl.arange(0, BLOCK_GROUPS * group_bytes)
    row_mask = rows < N
    byte_mask = offsets < GROUPS * group_bytes

    x_even = tl.load(X + 2 * offsets, byte_mask, other=0).to(tl.float32)
    x_odd = tl.load(X + 2 * offsets + 1, byte_mask, other=0).to(tl.float32)
    packed = tl.load(
        Packed + rows[:, None] * (GROUPS * group_bytes) + offsets[None, :],
        row_mask[:, None] & byte_mask[None, :],
        other=0,
    ).to(tl.uint32)
    low, high = _e2m1_doubled_pair(packed)
    pair_dot = low * x_even[None, :] + high * x_odd[None, :]
    group_dot = tl.sum(tl.reshape(pair_dot, (BLOCK_ROWS, BLOCK_GROUPS, group_bytes)), 2)

    groups = tl.arange(0, BLOCK_GROUPS)
    scales = tl.load(
        Scales + rows[:, None] * GROUPS + groups[None, :],
        row_mask[:, None] & (groups[None, :] < GROUPS),
        other=0,
    ).to(tl.float32)
    out = tl.sum(group_dot * scales, 1) * (0.5 * outer_scale)
    tl.store(Out + rows, out, row_mask)


def packed_e2m1_g16_matvec(
    x: torch.Tensor, packed: torch.Tensor, scales: torch.Tensor, outer_scale: float
) -> torch.Tensor:
    """Apply unchanged E2M1 nibbles to BF16 `[K]` and return FP32 `[N]`.

    Each byte stores the even column in its low nibble. Convert the source
    E4M3FN scale array to BF16 before calling. `outer_scale` remains the
    selected expert's FP32 amax divided by its projection's denominator.
    """
    if x.ndim != 1 or packed.ndim != 2 or scales.ndim != 2:
        raise ValueError("expected x[K], packed[N, K / 2], and scales[N, K / 16]")
    rows, packed_width = packed.shape
    width = x.shape[0]
    if (
        not width
        or width % 16
        or packed_width != width // 2
        or scales.shape != (rows, width // 16)
    ):
        raise ValueError("packed weights and G16 scales must match x")
    if x.dtype != torch.bfloat16 or packed.dtype != torch.uint8 or scales.dtype != torch.bfloat16:
        raise ValueError("x and scales must be bfloat16 and packed must be uint8")
    if not x.is_contiguous() or not packed.is_contiguous() or not scales.is_contiguous():
        raise ValueError("x, packed, and scales must be contiguous")
    if packed.device != x.device or scales.device != x.device:
        raise ValueError("x, packed, and scales must be on the same device")

    out = x.new_empty((rows,), dtype=torch.float32)
    if rows:
        block_rows = 4
        groups = width // 16
        _packed_e2m1_g16_kernel[(triton.cdiv(rows, block_rows),)](
            x,
            packed,
            scales,
            out,
            outer_scale,
            N=rows,
            GROUPS=groups,
            BLOCK_ROWS=block_rows,
            BLOCK_GROUPS=triton.next_power_of_2(groups),
            num_warps=4,
        )
    return out
