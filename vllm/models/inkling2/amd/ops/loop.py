# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Modified or added by Ciru for Inkling 2 Mini on AMD gfx1151.
"""Portable reference math for Inkling 2 loop messages and block residuals."""

import os

import torch
import triton
import triton.language as tl
from triton.language.extra.hip import libdevice

from .attn_res import fused_attn_res


_PACKED_ATTN_RES = os.getenv("INKLING_AMD_PACKED_ATTN_RES") == "1"


@triton.jit
def _pair_rms_norm_kernel(
    Dest,
    Message,
    Out,
    stride_dest_s,
    stride_dest_t,
    stride_message_s,
    stride_message_t,
    tokens,
    D: tl.constexpr,
    BLOCK: tl.constexpr,
):
    token = tl.program_id(0).to(tl.int64)
    stream = tl.program_id(1).to(tl.int64)
    cols = tl.arange(0, BLOCK)
    mask = cols < D
    dest = tl.load(
        Dest + stream * stride_dest_s + token * stride_dest_t + cols, mask, other=0
    )
    message = tl.load(
        Message + stream * stride_message_s + token * stride_message_t + cols,
        mask,
        other=0,
    )
    dest_rstd = libdevice.rsqrt(tl.div_rn(tl.sum(dest * dest), D) + 1e-6)
    message_rstd = libdevice.rsqrt(
        tl.div_rn(tl.sum(message * message), D) + 1e-6
    )
    out = Out + (stream * tokens + token) * (2 * D) + cols
    tl.store(out, (dest * dest_rstd).to(tl.bfloat16), mask)
    tl.store(out + D, (message * message_rstd).to(tl.bfloat16), mask)


@triton.jit
def _blend_rms_norm_kernel(
    Dest,
    Message,
    GainForget,
    NormWeight,
    Out,
    stride_dest_s,
    stride_dest_t,
    stride_message_s,
    stride_message_t,
    stride_weight_s,
    tokens,
    norm_eps,
    D: tl.constexpr,
    BLOCK: tl.constexpr,
):
    token = tl.program_id(0).to(tl.int64)
    stream = tl.program_id(1).to(tl.int64)
    row = stream * tokens + token
    cols = tl.arange(0, BLOCK)
    mask = cols < D
    dest = tl.load(
        Dest + stream * stride_dest_s + token * stride_dest_t + cols, mask, other=0
    )
    message = tl.load(
        Message + stream * stride_message_s + token * stride_message_t + cols,
        mask,
        other=0,
    )
    gains = GainForget + row * (2 * D) + cols
    gain = libdevice.tanh(tl.load(gains, mask, other=0))
    forget = libdevice.tanh(tl.load(gains + D, mask, other=0))
    blended = (1.0 - forget) * dest + gain * message
    rstd = libdevice.rsqrt(tl.div_rn(tl.sum(blended * blended), D) + norm_eps)
    weight = tl.load(NormWeight + stream * stride_weight_s + cols, mask, other=0)
    normalized = blended * rstd * weight.to(tl.float32)
    tl.store(Out + row * D + cols, normalized.to(tl.bfloat16), mask)


def _packed_rows(dest: torch.Tensor, message: torch.Tensor) -> bool:
    # AttnRes einsum returns token-major views for multi-token batches.
    # Only channels must be contiguous, and destination streams may broadcast.
    return (
        dest.is_cuda
        and dest.device == message.device
        and dest.shape == message.shape
        and message.ndim == 3
        and message.shape[0] in (1, 3)
        and message.shape[2] == 2304
        and dest.dtype == message.dtype == torch.float32
        and dest.stride(-1) == message.stride(-1) == 1
    )


def _unit_rms(x: torch.Tensor, eps: float) -> torch.Tensor:
    return x * torch.rsqrt(x.square().mean(dim=-1, keepdim=True) + eps)


def pair_rms_norm(
    dest: torch.Tensor, message: torch.Tensor, *, packed: bool = False
) -> torch.Tensor:
    if packed and _packed_rows(dest, message):
        from .hip_runtime import hip_pair_rmsnorm

        hip_out = hip_pair_rmsnorm(dest, message)
        if hip_out is not None:
            return hip_out
        streams, tokens, width = message.shape
        out = message.new_empty((streams, tokens, 2 * width), dtype=torch.bfloat16)
        if tokens:
            _pair_rms_norm_kernel[(tokens, streams)](
                dest, message, out, *dest.stride()[:2], *message.stride()[:2],
                tokens, D=width, BLOCK=4096, num_warps=4, enable_fp_fusion=False,
            )
        return out
    return torch.cat(
        (_unit_rms(dest, 1e-6), _unit_rms(message, 1e-6)), dim=-1
    ).bfloat16()


def blend_rms_norm(
    dest: torch.Tensor,
    message: torch.Tensor,
    gain_forget: torch.Tensor,
    norm_weight: torch.Tensor,
    norm_eps: float,
    *,
    packed: bool = False,
) -> torch.Tensor:
    if packed and _packed_rows(dest, message):
        streams, tokens, width = message.shape
        if (
            gain_forget.shape == (streams, tokens, 2 * width)
            and gain_forget.dtype == torch.float32
            and gain_forget.is_contiguous()
            and gain_forget.device == dest.device
            and norm_weight.shape == (streams, width)
            and norm_weight.dtype in (torch.float32, torch.bfloat16)
            and norm_weight.stride(-1) == 1
            and norm_weight.device == dest.device
        ):
            from .hip_runtime import hip_blend_rmsnorm

            hip_out = hip_blend_rmsnorm(dest, message, gain_forget, norm_weight, norm_eps)
            if hip_out is not None:
                return hip_out
            out = message.new_empty(message.shape, dtype=torch.bfloat16)
            if tokens:
                _blend_rms_norm_kernel[(tokens, streams)](
                    dest, message, gain_forget, norm_weight, out,
                    *dest.stride()[:2], *message.stride()[:2], norm_weight.stride(0),
                    tokens, norm_eps, D=width, BLOCK=4096, num_warps=4,
                    enable_fp_fusion=False,
                )
            return out
    gain, forget = gain_forget.tanh().chunk(2, dim=-1)
    blended = (1.0 - forget) * dest + gain * message
    return (_unit_rms(blended, norm_eps) * norm_weight[:, None]).bfloat16()


def attn_res(
    stack: torch.Tensor,
    last: torch.Tensor,
    addend: torch.Tensor | None,
    queries: torch.Tensor,
    norm_weight: torch.Tensor | None,
    norm_eps: float,
    normalize_values: bool,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Return FP32 mixes, or normalize and round at the BF16 boundary."""
    if _PACKED_ATTN_RES and last.is_cuda:
        fused = fused_attn_res(
            stack, last, addend, queries, norm_weight, norm_eps, normalize_values, eps
        )
        if fused is not None:
            return fused
    tail = last.float()
    if addend is not None:
        tail = tail + addend.float()
    sources = torch.cat((stack.float(), tail[None]), dim=0)
    normalized = _unit_rms(sources, eps)
    logits = torch.einsum("qd,std->qts", queries, normalized)
    probs = logits.softmax(dim=-1)
    values = normalized if normalize_values else sources
    mixed = torch.einsum("qts,std->qtd", probs, values)
    if norm_weight is None:
        return mixed
    return (_unit_rms(mixed, norm_eps) * norm_weight[:, None]).to(last.dtype)
