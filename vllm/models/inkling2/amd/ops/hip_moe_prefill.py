# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 CIRU AI.
"""Backward-compatible re-export of vllm.models.inkling2.amd.ops.hip_runtime."""

from .hip_runtime import (
    EXPECTED_HIP_ABI_VERSION,
    _get_hip_lib,
    _get_scratch,
    can_run_hip_kv_conv_1w,
    can_run_hip_moe_pair,
    compute_hip_src_hash,
    get_hip_lib,
    hip_attn_res_q4_coeffs,
    hip_kv_conv_1w,
    hip_kv_conv_prefill,
    hip_moe_prefill_available,
    hip_moe_prefill_projection,
    hip_moe_reduce,
    hip_unit_enabled,
)

__all__ = [
    "EXPECTED_HIP_ABI_VERSION",
    "can_run_hip_kv_conv_1w",
    "can_run_hip_moe_pair",
    "compute_hip_src_hash",
    "get_hip_lib",
    "hip_attn_res_q4_coeffs",
    "hip_kv_conv_1w",
    "hip_kv_conv_prefill",
    "hip_moe_prefill_available",
    "hip_moe_prefill_projection",
    "hip_moe_reduce",
    "hip_unit_enabled",
]
