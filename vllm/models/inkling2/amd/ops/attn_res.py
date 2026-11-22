# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Fused portable row kernel for Inkling 2 block attention residuals."""

import torch
import triton
import triton.language as tl


@triton.jit
def _load_source(Stack, Last, Addend, stride_s, offsets, mask,
                 block, N_STACK: tl.constexpr, HAS_ADDEND: tl.constexpr):
    if block < N_STACK:
        value = tl.load(Stack + block * stride_s + offsets, mask, 0).to(tl.float32)
    else:
        value = tl.load(Last + offsets, mask, 0).to(tl.float32)
        if HAS_ADDEND:
            value += tl.load(Addend + offsets, mask, 0).to(tl.float32)
    return value


@triton.jit
def _attn_res_row_kernel(
    Stack, Last, Addend, Queries, NormWeight, Out,
    stride_stack, stride_stack_row, stride_last, stride_add,
    stride_query, stride_norm, tokens, eps, norm_eps,
    D: tl.constexpr, N_STACK: tl.constexpr, HAS_ADDEND: tl.constexpr,
    NORMALIZE_VALUES: tl.constexpr, APPLY_NORM: tl.constexpr,
    BLOCK_N: tl.constexpr, BLOCK_A: tl.constexpr, BLOCK_B: tl.constexpr,
):
    # Like the donor, keep a power-of-two head and a short tail in registers.
    token = tl.program_id(0).to(tl.int64)
    query = tl.program_id(1).to(tl.int64)
    Stack += token * stride_stack_row
    Last += token * stride_last
    Addend += token * stride_add
    cols_a = tl.arange(0, BLOCK_A)
    cols_b = BLOCK_A + tl.arange(0, BLOCK_B)
    mask_a = cols_a < D
    mask_b = cols_b < D
    blocks = tl.arange(0, BLOCK_N)
    qa = tl.load(Queries + query * stride_query + cols_a, mask_a, 0)
    qb = tl.load(Queries + query * stride_query + cols_b, mask_b, 0)
    source = (Stack, Last, Addend, stride_stack)

    logits = tl.full((BLOCK_N,), float("-inf"), tl.float32)
    inv_rms = tl.zeros((BLOCK_N,), tl.float32)
    for block in tl.static_range(N_STACK + 1):
        a = _load_source(*source, cols_a, mask_a, block, N_STACK, HAS_ADDEND)
        b = _load_source(*source, cols_b, mask_b, block, N_STACK, HAS_ADDEND)
        rms = tl.rsqrt(tl.div_rn(tl.sum(a * a) + tl.sum(b * b), D) + eps)
        dot = tl.sum(qa * a) + tl.sum(qb * b)
        inv_rms = tl.where(blocks == block, rms, inv_rms)
        logits = tl.where(blocks == block, dot * rms, logits)

    probabilities = tl.exp(logits - tl.max(logits))
    inv_total = tl.div_rn(1.0, tl.sum(probabilities))
    if NORMALIZE_VALUES:
        probabilities = probabilities * inv_rms

    mix_a = tl.zeros((BLOCK_A,), tl.float32)
    mix_b = tl.zeros((BLOCK_B,), tl.float32)
    for block in tl.static_range(N_STACK + 1):
        p = tl.sum(tl.where(blocks == block, probabilities, 0.0))
        a = _load_source(*source, cols_a, mask_a, block, N_STACK, HAS_ADDEND)
        b = _load_source(*source, cols_b, mask_b, block, N_STACK, HAS_ADDEND)
        mix_a = tl.fma(p, a, mix_a)
        mix_b = tl.fma(p, b, mix_b)
    mix_a *= inv_total
    mix_b *= inv_total
    if APPLY_NORM:
        rms = tl.rsqrt(tl.div_rn(tl.sum(mix_a * mix_a) + tl.sum(mix_b * mix_b), D) + norm_eps)
        w = NormWeight + query * stride_norm
        wa = tl.load(w + cols_a, mask_a, 0).to(tl.float32)
        wb = tl.load(w + cols_b, mask_b, 0).to(tl.float32)
        mix_a = mix_a * rms * wa
        mix_b = mix_b * rms * wb

    out = Out + (query * tokens + token) * D
    if Out.dtype.element_ty == tl.bfloat16:
        mix_a = mix_a.to(tl.bfloat16, fp_downcast_rounding="rtne")
        mix_b = mix_b.to(tl.bfloat16, fp_downcast_rounding="rtne")
    tl.store(out + cols_a, mix_a, mask_a)
    tl.store(out + cols_b, mix_b, mask_b)



@triton.jit
def _attn_res_q3_kernel(
    Stack, Last, Addend, Queries, NormWeight, Out,
    stride_stack, stride_stack_row, stride_last, stride_add,
    stride_query, stride_norm, tokens, eps, norm_eps,
    D: tl.constexpr, N_STACK: tl.constexpr, HAS_ADDEND: tl.constexpr,
    NORMALIZE_VALUES: tl.constexpr, APPLY_NORM: tl.constexpr,
    BLOCK_N: tl.constexpr, BLOCK_A: tl.constexpr, BLOCK_B: tl.constexpr,
):
    token = tl.program_id(0).to(tl.int64)
    Stack += token * stride_stack_row
    Last += token * stride_last
    Addend += token * stride_add
    cols_a = tl.arange(0, BLOCK_A)
    cols_b = BLOCK_A + tl.arange(0, BLOCK_B)
    mask_a = cols_a < D
    mask_b = cols_b < D
    blocks = tl.arange(0, BLOCK_N)
    q0a = tl.load(Queries + 0 * stride_query + cols_a, mask_a, 0)
    q0b = tl.load(Queries + 0 * stride_query + cols_b, mask_b, 0)
    q1a = tl.load(Queries + 1 * stride_query + cols_a, mask_a, 0)
    q1b = tl.load(Queries + 1 * stride_query + cols_b, mask_b, 0)
    q2a = tl.load(Queries + 2 * stride_query + cols_a, mask_a, 0)
    q2b = tl.load(Queries + 2 * stride_query + cols_b, mask_b, 0)
    source = (Stack, Last, Addend, stride_stack)

    l0 = tl.full((BLOCK_N,), float("-inf"), tl.float32)
    l1 = tl.full((BLOCK_N,), float("-inf"), tl.float32)
    l2 = tl.full((BLOCK_N,), float("-inf"), tl.float32)
    inv_rms = tl.zeros((BLOCK_N,), tl.float32)
    for block in tl.static_range(N_STACK + 1):
        a = _load_source(*source, cols_a, mask_a, block, N_STACK, HAS_ADDEND)
        b = _load_source(*source, cols_b, mask_b, block, N_STACK, HAS_ADDEND)
        rms = tl.rsqrt(tl.div_rn(tl.sum(a * a) + tl.sum(b * b), D) + eps)
        inv_rms = tl.where(blocks == block, rms, inv_rms)
        l0 = tl.where(blocks == block, (tl.sum(q0a * a) + tl.sum(q0b * b)) * rms, l0)
        l1 = tl.where(blocks == block, (tl.sum(q1a * a) + tl.sum(q1b * b)) * rms, l1)
        l2 = tl.where(blocks == block, (tl.sum(q2a * a) + tl.sum(q2b * b)) * rms, l2)

    p0 = tl.exp(l0 - tl.max(l0)); inv0 = tl.div_rn(1.0, tl.sum(p0))
    p1 = tl.exp(l1 - tl.max(l1)); inv1 = tl.div_rn(1.0, tl.sum(p1))
    p2 = tl.exp(l2 - tl.max(l2)); inv2 = tl.div_rn(1.0, tl.sum(p2))
    if NORMALIZE_VALUES:
        p0 *= inv_rms; p1 *= inv_rms; p2 *= inv_rms
    p0 *= inv0; p1 *= inv1; p2 *= inv2

    m0a = tl.zeros((BLOCK_A,), tl.float32); m0b = tl.zeros((BLOCK_B,), tl.float32)
    m1a = tl.zeros((BLOCK_A,), tl.float32); m1b = tl.zeros((BLOCK_B,), tl.float32)
    m2a = tl.zeros((BLOCK_A,), tl.float32); m2b = tl.zeros((BLOCK_B,), tl.float32)
    for block in tl.static_range(N_STACK + 1):
        a = _load_source(*source, cols_a, mask_a, block, N_STACK, HAS_ADDEND)
        b = _load_source(*source, cols_b, mask_b, block, N_STACK, HAS_ADDEND)
        w0 = tl.sum(tl.where(blocks == block, p0, 0.0))
        w1 = tl.sum(tl.where(blocks == block, p1, 0.0))
        w2 = tl.sum(tl.where(blocks == block, p2, 0.0))
        m0a = tl.fma(w0, a, m0a); m0b = tl.fma(w0, b, m0b)
        m1a = tl.fma(w1, a, m1a); m1b = tl.fma(w1, b, m1b)
        m2a = tl.fma(w2, a, m2a); m2b = tl.fma(w2, b, m2b)

    if APPLY_NORM:
        r0 = tl.rsqrt(tl.div_rn(tl.sum(m0a * m0a) + tl.sum(m0b * m0b), D) + norm_eps)
        m0a = m0a * r0 * tl.load(NormWeight + 0 * stride_norm + cols_a, mask_a, 0).to(tl.float32)
        m0b = m0b * r0 * tl.load(NormWeight + 0 * stride_norm + cols_b, mask_b, 0).to(tl.float32)
        r1 = tl.rsqrt(tl.div_rn(tl.sum(m1a * m1a) + tl.sum(m1b * m1b), D) + norm_eps)
        m1a = m1a * r1 * tl.load(NormWeight + 1 * stride_norm + cols_a, mask_a, 0).to(tl.float32)
        m1b = m1b * r1 * tl.load(NormWeight + 1 * stride_norm + cols_b, mask_b, 0).to(tl.float32)
        r2 = tl.rsqrt(tl.div_rn(tl.sum(m2a * m2a) + tl.sum(m2b * m2b), D) + norm_eps)
        m2a = m2a * r2 * tl.load(NormWeight + 2 * stride_norm + cols_a, mask_a, 0).to(tl.float32)
        m2b = m2b * r2 * tl.load(NormWeight + 2 * stride_norm + cols_b, mask_b, 0).to(tl.float32)
    if Out.dtype.element_ty == tl.bfloat16:
        m0a = m0a.to(tl.bfloat16, fp_downcast_rounding="rtne"); m0b = m0b.to(tl.bfloat16, fp_downcast_rounding="rtne")
        m1a = m1a.to(tl.bfloat16, fp_downcast_rounding="rtne"); m1b = m1b.to(tl.bfloat16, fp_downcast_rounding="rtne")
        m2a = m2a.to(tl.bfloat16, fp_downcast_rounding="rtne"); m2b = m2b.to(tl.bfloat16, fp_downcast_rounding="rtne")
    tl.store(Out + (0 * tokens + token) * D + cols_a, m0a, mask_a)
    tl.store(Out + (0 * tokens + token) * D + cols_b, m0b, mask_b)
    tl.store(Out + (1 * tokens + token) * D + cols_a, m1a, mask_a)
    tl.store(Out + (1 * tokens + token) * D + cols_b, m1b, mask_b)
    tl.store(Out + (2 * tokens + token) * D + cols_a, m2a, mask_a)
    tl.store(Out + (2 * tokens + token) * D + cols_b, m2b, mask_b)



@triton.jit
def _attn_res_q4_n1_kernel(
    Stack, Last, Addend, Queries, NormWeight, Out,
    stride_stack, stride_stack_row, stride_last, stride_add,
    stride_query, stride_norm, tokens, eps, norm_eps,
    D: tl.constexpr, HAS_ADDEND: tl.constexpr,
    NORMALIZE_VALUES: tl.constexpr, APPLY_NORM: tl.constexpr,
    BLOCK_A: tl.constexpr, BLOCK_B: tl.constexpr,
):
    token = tl.program_id(0).to(tl.int64)
    Stack += token * stride_stack_row
    Last += token * stride_last
    Addend += token * stride_add
    cols_a = tl.arange(0, BLOCK_A)
    cols_b = BLOCK_A + tl.arange(0, BLOCK_B)
    mask_a = cols_a < D
    mask_b = cols_b < D
    source = (Stack, Last, Addend, stride_stack)

    a0 = _load_source(*source, cols_a, mask_a, 0, 1, HAS_ADDEND)
    b0 = _load_source(*source, cols_b, mask_b, 0, 1, HAS_ADDEND)
    r0 = tl.rsqrt(tl.div_rn(tl.sum(a0 * a0) + tl.sum(b0 * b0), D) + eps)
    a1 = _load_source(*source, cols_a, mask_a, 1, 1, HAS_ADDEND)
    b1 = _load_source(*source, cols_b, mask_b, 1, 1, HAS_ADDEND)
    r1 = tl.rsqrt(tl.div_rn(tl.sum(a1 * a1) + tl.sum(b1 * b1), D) + eps)

    if NORMALIZE_VALUES:
        a0 = a0 * r0
        b0 = b0 * r0
        a1 = a1 * r1
        b1 = b1 * r1

    for qi in tl.static_range(4):
        qa = tl.load(Queries + qi * stride_query + cols_a, mask_a, 0)
        qb = tl.load(Queries + qi * stride_query + cols_b, mask_b, 0)
        if NORMALIZE_VALUES:
            l0 = tl.sum(qa * a0) + tl.sum(qb * b0)
            l1 = tl.sum(qa * a1) + tl.sum(qb * b1)
        else:
            l0 = (tl.sum(qa * a0) + tl.sum(qb * b0)) * r0
            l1 = (tl.sum(qa * a1) + tl.sum(qb * b1)) * r1
        m = tl.maximum(l0, l1)
        e0 = tl.exp(l0 - m)
        e1 = tl.exp(l1 - m)
        inv = tl.div_rn(1.0, e0 + e1)
        w0 = e0 * inv
        w1 = e1 * inv
        ma = tl.fma(w0, a0, w1 * a1)
        mb = tl.fma(w0, b0, w1 * b1)
        if APPLY_NORM:
            rn = tl.rsqrt(tl.div_rn(tl.sum(ma * ma) + tl.sum(mb * mb), D) + norm_eps)
            ma = ma * rn * tl.load(NormWeight + qi * stride_norm + cols_a, mask_a, 0).to(tl.float32)
            mb = mb * rn * tl.load(NormWeight + qi * stride_norm + cols_b, mask_b, 0).to(tl.float32)
        if Out.dtype.element_ty == tl.bfloat16:
            ma = ma.to(tl.bfloat16, fp_downcast_rounding="rtne")
            mb = mb.to(tl.bfloat16, fp_downcast_rounding="rtne")
        tl.store(Out + (qi * tokens + token) * D + cols_a, ma, mask_a)
        tl.store(Out + (qi * tokens + token) * D + cols_b, mb, mask_b)


def fused_attn_res(
    stack: torch.Tensor,
    last: torch.Tensor,
    addend: torch.Tensor | None,
    queries: torch.Tensor,
    norm_weight: torch.Tensor | None,
    norm_eps: float,
    normalize_values: bool,
    eps: float = 1e-6,
) -> torch.Tensor | None:
    """Mix residuals with packed rows. Return None for unsupported geometry."""
    tokens, width = last.shape
    if width != 2304 or stack.shape[1:] != last.shape or queries.shape[-1] != width:
        return None
    if last.dtype not in (torch.bfloat16, torch.float32) or queries.dtype != torch.float32:
        return None
    if addend is not None and addend.shape != last.shape:
        return None
    if norm_weight is not None and norm_weight.shape != queries.shape:
        return None
    tail = last if addend is None else addend
    weight = queries if norm_weight is None else norm_weight
    if any(x.device != last.device or x.stride(-1) != 1
           for x in (stack, last, tail, queries, weight)):
        return None

    dtype = last.dtype if norm_weight is not None else torch.float32
    out = last.new_empty((queries.shape[0], tokens, width), dtype=dtype)
    if tokens:
        if queries.shape[0] == 4 and stack.shape[0] == 1 and tokens >= 64:
            _attn_res_q4_n1_kernel[(tokens,)](
                stack, last, tail, queries, weight, out,
                stack.stride(0), stack.stride(1), last.stride(0), tail.stride(0),
                queries.stride(0), weight.stride(0), tokens, eps, norm_eps,
                D=width, HAS_ADDEND=addend is not None,
                NORMALIZE_VALUES=normalize_values, APPLY_NORM=norm_weight is not None,
                BLOCK_A=2048, BLOCK_B=256, num_warps=4, enable_fp_fusion=False,
            )
        elif queries.shape[0] == 3 and tokens >= 64:
            _attn_res_q3_kernel[(tokens,)](
                stack, last, tail, queries, weight, out,
                stack.stride(0), stack.stride(1), last.stride(0), tail.stride(0),
                queries.stride(0), weight.stride(0), tokens, eps, norm_eps,
                D=width, N_STACK=stack.shape[0], HAS_ADDEND=addend is not None,
                NORMALIZE_VALUES=normalize_values, APPLY_NORM=norm_weight is not None,
                BLOCK_N=triton.next_power_of_2(stack.shape[0] + 1),
                BLOCK_A=2048, BLOCK_B=256, num_warps=8, enable_fp_fusion=False,
            )
        else:
            _attn_res_row_kernel[(tokens, queries.shape[0])](
                stack, last, tail, queries, weight, out,
                stack.stride(0), stack.stride(1), last.stride(0), tail.stride(0),
                queries.stride(0), weight.stride(0), tokens, eps, norm_eps,
                D=width, N_STACK=stack.shape[0], HAS_ADDEND=addend is not None,
                NORMALIZE_VALUES=normalize_values, APPLY_NORM=norm_weight is not None,
                BLOCK_N=triton.next_power_of_2(stack.shape[0] + 1),
                BLOCK_A=2048, BLOCK_B=256, num_warps=8, enable_fp_fusion=False,
            )
    return out
