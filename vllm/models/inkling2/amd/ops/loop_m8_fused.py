# SPDX-License-Identifier: Apache-2.0
# Modified or added by Ciru for Inkling 2 Mini on AMD gfx1151.
"""Ciru M8 controller gain/forget and blend using the original BF16 banks."""

import torch
import triton
import triton.language as tl
from triton.experimental import gluon
from triton.experimental.gluon import language as gl
from triton.experimental.gluon.language.extra import libdevice
from triton.language.extra.hip import libdevice as tl_libdevice

from .loop import _packed_rows


@gluon.jit
def _controller_m8_blend_kernel(
    Hidden, Weight, Dest, Message, Blended, stride_dest_s, stride_dest_t,
    stride_message_s, stride_message_t, D: gl.constexpr, B: gl.constexpr,
    LANES_K: gl.constexpr, WAVES: gl.constexpr,
):
    # Keep each rank sum inside one wave. Across the eight tokens, a thread
    # holds the same weight values and has independent dot-product outputs.
    rows_per_wave: gl.constexpr = 32 // LANES_K
    tile: gl.constexpr = gl.BlockedLayout(
        [1, 1, 64 // LANES_K], [1, rows_per_wave, LANES_K], [1, WAVES, 1], [2, 1, 0],
    )
    activation: gl.constexpr = gl.SliceLayout(1, tile)
    bank: gl.constexpr = gl.SliceLayout(0, tile)
    output: gl.constexpr = gl.SliceLayout(2, tile)
    stream = gl.program_id(1)
    first_channel = gl.program_id(0) * B
    tokens = gl.arange(0, 8, gl.SliceLayout(1, activation))
    cols = gl.arange(0, 64, gl.SliceLayout(0, activation))
    channel = first_channel + gl.arange(0, B, gl.SliceLayout(1, bank))
    bank_cols = gl.arange(0, 64, gl.SliceLayout(0, bank))
    h = gl.load(Hidden + (stream * 8 + tokens[:, None]) * 64 + cols[None, :]).to(gl.float32)
    weight_ptr = Weight + (stream * (2 * D) + channel[:, None]) * 64 + bank_cols[None, :]
    w_gain = gl.load(weight_ptr).to(gl.float32)
    gain = gl.sum(h[:, None, :] * w_gain[None, :, :], axis=2)
    w_forget = gl.load(weight_ptr + D * 64).to(gl.float32)
    forget = gl.sum(h[:, None, :] * w_forget[None, :, :], axis=2)

    token = gl.arange(0, 8, gl.SliceLayout(1, output))
    channel = first_channel + gl.arange(0, B, gl.SliceLayout(0, output))
    dest = gl.load(Dest + stream * stride_dest_s + token[:, None] * stride_dest_t + channel[None, :])
    message = gl.load(
        Message + stream * stride_message_s + token[:, None] * stride_message_t + channel[None, :]
    )
    blended = (1.0 - libdevice.tanh(forget)) * dest + libdevice.tanh(gain) * message
    gl.store(Blended + (stream * 8 + token[:, None]) * D + channel[None, :], blended)


@triton.jit
def _controller_m8_rms_kernel(
    Blended, NormWeight, Out, stride_weight_s, norm_eps,
    D: tl.constexpr, BLOCK: tl.constexpr,
):
    token = tl.program_id(0).to(tl.int64)
    stream = tl.program_id(1).to(tl.int64)
    row = stream * 8 + token
    cols = tl.arange(0, BLOCK)
    mask = cols < D
    blended = tl.load(Blended + row * D + cols, mask, other=0)
    rstd = tl_libdevice.rsqrt(tl.div_rn(tl.sum(blended * blended), D) + norm_eps)
    weight = tl.load(NormWeight + stream * stride_weight_s + cols, mask, other=0)
    normalized = blended * rstd * weight.to(tl.float32)
    tl.store(Out + row * D + cols, normalized.to(tl.bfloat16), mask)


def fused_loop_m8_blend(
    hidden: torch.Tensor, weight: torch.Tensor, dest: torch.Tensor,
    message: torch.Tensor, norm_weight: torch.Tensor, norm_eps: float,
) -> torch.Tensor | None:
    if (
        not _packed_rows(dest, message)
        or message.shape[1] != 8
        or torch.version.hip is None
        or torch.cuda.get_device_properties(dest.device).gcnArchName.split(":", 1)[0] != "gfx1151"
    ):
        return None
    streams, _, width = message.shape
    if (
        hidden.shape != (streams, 8, 64)
        or weight.shape != (streams, 2 * width, 64)
        or any(
            x.dtype != torch.bfloat16 or x.device != dest.device or not x.is_contiguous()
            for x in (hidden, weight)
        )
        or norm_weight.shape != (streams, width)
        or norm_weight.dtype not in (torch.bfloat16, torch.float32)
        or norm_weight.device != dest.device or norm_weight.stride(-1) != 1
    ):
        return None

    blended = dest.new_empty(dest.shape)
    out = dest.new_empty(dest.shape, dtype=torch.bfloat16)
    _controller_m8_blend_kernel[(width // 8, streams)](
        hidden, weight, dest, message, blended, *dest.stride()[:2], *message.stride()[:2],
        D=width, B=8, LANES_K=16, WAVES=4, num_warps=4, enable_fp_fusion=False,
    )
    _controller_m8_rms_kernel[(8, streams)](
        blended, norm_weight, out, norm_weight.stride(0), norm_eps,
        D=width, BLOCK=4096, num_warps=4, enable_fp_fusion=False,
    )
    return out
