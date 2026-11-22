"""Direct packed Inkling 2 Mini NVFP4 weight matvec with BF16 activations."""

import torch
import triton
import triton.language as tl


@triton.jit
def _e2m1_doubled(code):
    bits = code & 7
    # Twice the E2M1 magnitudes: 0, 1, 2, 3, 4, 6, 8, 12.
    magnitude = bits + tl.maximum(bits - 4, 0) + tl.where(bits == 7, 2, 0)
    return tl.where((code & 8) != 0, -magnitude, magnitude).to(tl.float32)


@triton.jit
def _positive_e4m3fn(raw):
    exponent = raw >> 3
    mantissa = (raw & 7) + tl.where(exponent == 0, 0, 8)
    # FP32 encoding of 2^(max(exponent, 1) - E4M3 bias - mantissa bits).
    power = ((tl.maximum(exponent, 1) + 117) << 23).to(tl.float32, bitcast=True)
    return tl.where(raw == 0x7F, float("nan"), mantissa.to(tl.float32) * power)


@triton.jit
def _packed_nvfp4_kernel(
    X,
    Packed,
    ScaleBytes,
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
    ).to(tl.int32)
    low = _e2m1_doubled(packed & 15)
    high = _e2m1_doubled(packed >> 4)
    pair_dot = low * x_even[None, :] + high * x_odd[None, :]
    group_dot = tl.sum(tl.reshape(pair_dot, (BLOCK_ROWS, BLOCK_GROUPS, group_bytes)), 2)

    groups = tl.arange(0, BLOCK_GROUPS)
    scale_bytes = tl.load(
        ScaleBytes + rows[:, None] * GROUPS + groups[None, :],
        row_mask[:, None] & (groups[None, :] < GROUPS),
        other=0,
    ).to(tl.int32)
    out = tl.sum(group_dot * _positive_e4m3fn(scale_bytes), 1) * (0.5 * outer_scale)
    tl.store(Out + rows, out, row_mask)


def packed_nvfp4_matvec(
    x: torch.Tensor, packed: torch.Tensor, scale_bytes: torch.Tensor, outer_scale: float
) -> torch.Tensor:
    """Apply `[N, K / 2]` packed E2M1 weights to one `[K]` BF16 activation.

    The low nibble stores the even column. `scale_bytes` holds raw positive
    E4M3FN values with shape `[N, K / 16]`. `outer_scale` is the selected
    expert's amax divided by the projection's outer scale denominator.
    The fused unpack and FP32 reduction return `[N]` without expanding the
    weight matrix. This control uses BF16 activations.
    """
    if x.ndim != 1 or packed.ndim != 2 or scale_bytes.ndim != 2:
        raise ValueError("expected x[K], packed[N, K / 2], and scale_bytes[N, K / 16]")
    rows, packed_width = packed.shape
    width = x.shape[0]
    groups = scale_bytes.shape[1]
    if width != 2 * packed_width or scale_bytes.shape[0] != rows or not groups:
        raise ValueError("packed and scale_bytes shapes must agree with x and each other")
    if width != 16 * groups:
        raise ValueError("scale_bytes must define groups of 16 columns")
    if x.dtype != torch.bfloat16 or packed.dtype != torch.uint8 or scale_bytes.dtype != torch.uint8:
        raise ValueError("x must be bfloat16 and packed and scale_bytes must be uint8")
    if not x.is_contiguous() or not packed.is_contiguous() or not scale_bytes.is_contiguous():
        raise ValueError("x, packed, and scale_bytes must be contiguous")
    if packed.device != x.device or scale_bytes.device != x.device:
        raise ValueError("x, packed, and scale_bytes must be on the same device")

    out = x.new_empty((rows,), dtype=torch.float32)
    if rows:
        block_rows = 4
        _packed_nvfp4_kernel[(triton.cdiv(rows, block_rows),)](
            x,
            packed,
            scale_bytes,
            out,
            outer_scale,
            N=rows,
            GROUPS=groups,
            BLOCK_ROWS=block_rows,
            BLOCK_GROUPS=triton.next_power_of_2(groups),
            num_warps=4,
        )
    return out
