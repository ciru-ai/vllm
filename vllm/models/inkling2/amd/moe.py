# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Inkling 2 Mini TP1 ROCm experts, preserving the publisher's packed weights."""

from pathlib import Path
from typing import Literal
import os

import torch
from torch import nn

from vllm.forward_context import get_forward_context
from vllm.model_executor.model_loader.weight_utils import default_weight_loader
from vllm.model_executor.utils import set_weight_attrs

QuantFormat = Literal["nvfp4", "mxfp8"]
PROJECTIONS = ("w13", "w2")
POLICIES = ("source", "signed", "c1-down", "full-c1")
CODA_C1_LAYER = 22
CODA_C1_LAYERS = range(20, 24)
CODA_C1_FIELDS = {
    "packed": "weight",
    "tags": "weight_scale",
    "scales": "c1_row",
    "table": "c1_table",
}


def coda_c1_paths() -> dict[int, Path]:
    legacy = os.getenv("INKLING_AMD_CODA_C1")
    directory = os.getenv("INKLING_AMD_CODA_C1_DIR")
    if legacy and directory:
        raise ValueError("Set only one of INKLING_AMD_CODA_C1 and INKLING_AMD_CODA_C1_DIR")
    if legacy:
        return {CODA_C1_LAYER: Path(legacy)}
    if not directory:
        return {}
    keep_final = os.getenv("INKLING_AMD_CODA_C1_KEEP_FINAL_MXFP8") == "1"
    layers = CODA_C1_LAYERS[:-1] if keep_final else CODA_C1_LAYERS
    paths = {
        layer: Path(directory) / f"layer{layer}-c1.safetensors"
        for layer in layers
    }
    missing = [str(path) for path in paths.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing Inkling coda C1 packets: {', '.join(missing)}")
    return paths


class Inkling2Gate(nn.Module):
    def __init__(self, config):
        super().__init__()
        if not config.use_gate_bias:
            raise ValueError("Inkling 2 Mini requires its router bias")
        self.weight = self._param(config.n_routed_experts, config.hidden_size, dtype=torch.bfloat16)
        self.bias = self._param(config.n_routed_experts, dtype=torch.float32)
        self.topk = config.num_experts_per_tok
        self.route_scale = config.route_scale
        self.packed_m1_router = os.getenv("INKLING_AMD_PACKED_M1_ROUTER", "0") == "1"
        self.packed_m1_scores = os.getenv("INKLING_AMD_PACKED_M1_ROUTER_SCORES", "0") == "1"
        self.packed_m1_selection = os.getenv("INKLING_AMD_PACKED_M1_ROUTER_SELECTION", "0") == "1"
        self.register_buffer("fp32_weight", None, persistent=False)

    @staticmethod
    def _param(*shape, dtype):
        parameter = nn.Parameter(torch.empty(shape, dtype=dtype), requires_grad=False)
        set_weight_attrs(parameter, {"weight_loader": default_weight_loader})
        return parameter

    def fp32_logits(self, x):
        if (
            os.getenv("INKLING_AMD_TURBO_ROUTER_M8") == "1"
            and x.is_cuda
            and x.dtype == torch.bfloat16
            and x.shape == (8, 2304)
            and self.weight.shape == (128, 2304)
        ):
            from .ops.turbo_kernels import router_gemm_m8_fp32

            return router_gemm_m8_fp32(x, self.weight)
        weight = self.fp32_weight if self.fp32_weight is not None else self.weight.float()
        return x.float() @ weight.T

    def forward(self, x):
        scores = None
        if self.packed_m1_scores and x.shape[0] == 1:
            from .ops.router import router_scores_m1

            scores = router_scores_m1(x, self.weight, self.bias)
        elif (
            x.shape[0] >= 64
            and os.getenv("INKLING_AMD_TURBO_PREFILL_MOE", "0") not in ("", "0")
        ):
            from .ops.turbo_kernels import router_scores_prefill

            scores = router_scores_prefill(x, self.weight, self.bias)
        if scores is None:
            score = self.fp32_logits(x).sigmoid()
            corrected = score + self.bias
        else:
            score, corrected = scores
        m1 = x.is_cuda and score.shape == (1, 128) and self.topk == 8
        if self.packed_m1_selection and m1:
            from .ops.router import select_top8_routes_m1

            return select_top8_routes_m1(score, corrected, self.route_scale)
        ids = torch.topk(corrected, self.topk, dim=-1, sorted=False).indices
        if self.packed_m1_router and m1:
            from .ops.router import normalize_top8_routes

            return ids, normalize_top8_routes(score, ids, self.route_scale)
        selected = score.gather(-1, ids)
        return ids, selected / (selected.sum(dim=-1, keepdim=True) + 1e-20) * self.route_scale


class Inkling2Experts(nn.Module):
    def __init__(self, config, quant_format: QuantFormat, coda_c1_path: Path | None = None):
        super().__init__()
        if quant_format not in ("nvfp4", "mxfp8"):
            raise ValueError(f"unsupported Inkling expert format {quant_format}")
        self.nvfp4 = quant_format == "nvfp4"
        self.source_m1_short_k = os.getenv("INKLING_AMD_SOURCE_M1_SHORT_K", "0") == "1"
        self.source_m1_bf16_dot2 = os.getenv("INKLING_AMD_SOURCE_M1_BF16_DOT2", "0") == "1"
        self.source_m1_wave32 = os.getenv("INKLING_AMD_SOURCE_M1_WAVE32", "0") == "1"
        self.source_wmma_banks = {}
        self.verifier_wmma_banks = {}
        turbo_prefill_moe = os.getenv("INKLING_AMD_TURBO_PREFILL_MOE", "0") not in ("", "0")
        target_pretile = (
            (os.getenv("INKLING_AMD_DSPARK_TARGET_GRAPH") == "1"
             and os.getenv("INKLING_AMD_SOURCE_WMMA_PRETILE") == "1")
            or turbo_prefill_moe
        )
        self.coda_wmma = (
            target_pretile and os.getenv("INKLING_AMD_CODA_WMMA_PRETILE") == "1"
        )
        self.mixed_wmma = (
            target_pretile and os.getenv("INKLING_AMD_MIXED_WMMA_PRETILE") == "1"
        )
        self.coda_c1_path = coda_c1_path
        self.num_experts = config.n_routed_experts
        self.topk = config.num_experts_per_tok
        self.shapes = {
            "w13": (2 * config.intermediate_size, config.hidden_size),
            "w2": (config.hidden_size, config.intermediate_size),
        }
        self.register_buffer(
            "e2m1", torch.tensor([0., .5, 1., 1.5, 2., 3., 4., 6., -0., -.5, -1., -1.5, -2., -3., -4., -6.]),
            persistent=False,
        )
        if self.coda_c1_path is not None:
            if self.nvfp4 or self.num_experts != 128:
                raise ValueError("Inkling coda C1 requires the 128-expert publisher MXFP8 layer")
            for name, (n, k) in self.shapes.items():
                shape = (self.num_experts, n)
                for suffix, tensor in (
                    ("weight", torch.empty((*shape, k // 2), dtype=torch.uint8)),
                    ("weight_scale", torch.empty((*shape, k // 16), dtype=torch.uint8)),
                    ("c1_row", torch.empty(shape, dtype=torch.float16)),
                    ("c1_table", torch.empty((self.num_experts, 256, 2), dtype=torch.float16)),
                ):
                    self.register_buffer(f"{name}_{suffix}", tensor, persistent=False)
            self.register_buffer("coda_kind", torch.full((self.num_experts,), 2, dtype=torch.int32), persistent=False)
            self.register_buffer("coda_outer", torch.ones(self.num_experts, dtype=torch.float32), persistent=False)
            return
        for name, (n, k) in self.shapes.items():
            pack, group = (2, 16) if self.nvfp4 else (1, 32)
            self._add(f"{name}_weight", (self.num_experts, n, k // pack),
                      torch.uint8 if self.nvfp4 else torch.float8_e4m3fn)
            self._add(f"{name}_weight_scale", (self.num_experts, n, k // group),
                      torch.float8_e4m3fn if self.nvfp4 else torch.uint8)
            if self.nvfp4:
                self._add(f"{name}_weight_amax", (self.num_experts,), torch.float32)
                self._add(f"{name}_weight_outer_scale_denominator", (1,), torch.float32)
                for suffix in ("outer", "kind", "c1_row", "c1_table"):
                    self.register_buffer(f"{name}_{suffix}", None, persistent=False)
        self._kinds: dict[str, dict[int, int]] = {name: {} for name in PROJECTIONS}
        self.normal_scales = False

    def _add(self, name, shape, dtype):
        parameter = nn.Parameter(torch.empty(shape, dtype=dtype), requires_grad=False)
        set_weight_attrs(parameter, {"weight_loader": default_weight_loader})
        self.register_parameter(name, parameter)

    def packed(self, name):
        return getattr(self, f"{name}_weight"), getattr(self, f"{name}_weight_scale")

    def setup(self, layer_idx, directory, policy):
        if self.coda_c1_path is not None:
            self._load_coda_c1(layer_idx)
            if self.coda_wmma:
                from .ops.coda_wmma_pretile import pack_coda_wmma

                for name in PROJECTIONS:
                    weights, tags = self.packed(name)
                    self.verifier_wmma_banks[name] = pack_coda_wmma(
                        weights, tags, *self._c1_aux(name)
                    )
                print(f"Inkling ROCm layer {layer_idx}: pre-tiled coda C1 W13/W2", flush=True)
            return
        if not self.nvfp4:
            return
        regular = True
        for name in PROJECTIONS:
            _, scale = self.packed(name)
            amin, amax = torch.aminmax(scale.view(torch.uint8))
            regular &= int(amin) >= 8 and int(amax) <= 126
            outer = getattr(self, f"{name}_weight_amax") / getattr(self, f"{name}_weight_outer_scale_denominator")
            setattr(self, f"{name}_outer", outer.detach())
        self.normal_scales = regular
        if (
            regular and (directory is None or policy == "source" or layer_idx not in (6, 12, 17))
            and (
                (os.getenv("INKLING_AMD_DSPARK_TARGET_GRAPH") == "1"
                 and os.getenv("INKLING_AMD_SOURCE_WMMA_PRETILE") == "1")
                or os.getenv("INKLING_AMD_TURBO_PREFILL_MOE", "0") not in ("", "0")
            )
        ):
            from .ops.source_wmma_pretile import pack_source_wmma

            for name in PROJECTIONS:
                weights, scales = self.packed(name)
                outer = getattr(self, f"{name}_outer")
                self.source_wmma_banks[name] = pack_source_wmma(
                    weights, scales.view(torch.uint8), outer,
                )

        if directory is None or layer_idx not in (6, 12, 17):
            return
        path = Path(directory) / f"layer{layer_idx}.pt"
        bank = torch.load(path, map_location="cpu", weights_only=True)
        if bank["layer"] != layer_idx:
            raise ValueError(f"{path} contains the wrong Inkling layer")
        for name, (n, _) in self.shapes.items():
            weights, scales = self.packed(name)
            kinds = torch.zeros(self.num_experts, dtype=torch.int32, device=weights.device)
            row_scales = torch.zeros(self.num_experts, n, dtype=torch.float16, device=weights.device)
            table = torch.zeros(self.num_experts, 256, 2, dtype=torch.float16, device=weights.device)
            # All candidate arms use the mixed decoder at these three layers.
            if policy != "source":
                c1 = policy == "full-c1" or (policy == "c1-down" and name == "w2")
                for expert, expert_bank in bank["fitted"].items():
                    rep = expert_bank["c1" if c1 else "signed"][name]
                    weights[expert].copy_(rep["packed"])
                    scales[expert].view(torch.uint8).copy_(rep["tags"] if c1 else rep["scales"].view(torch.uint8))
                    kinds[expert] = 2 if c1 else 1
                    self._kinds[name][expert] = 2 if c1 else 1
                    if c1:
                        row_scales[expert].copy_(rep["scales"])
                        table[expert].copy_(rep["table"])
            setattr(self, f"{name}_kind", kinds)
            setattr(self, f"{name}_c1_row", row_scales)
            setattr(self, f"{name}_c1_table", table)
            fitted_kinds = set(self._kinds[name].values())
            if self.mixed_wmma and self.normal_scales and len(fitted_kinds) == 1:
                from .ops.mixed_wmma_pretile import pack_mixed_wmma

                self.verifier_wmma_banks[name] = pack_mixed_wmma(
                    weights, scales.view(torch.uint8), getattr(self, f"{name}_outer"), kinds,
                    fitted_kind=fitted_kinds.pop(), flash_aux=(row_scales, table),
                )
        print(f"Inkling ROCm layer {layer_idx}: {policy} expert bank, {len(bank['fitted'])} candidates", flush=True)
        if self.verifier_wmma_banks:
            projections = "/".join(self.verifier_wmma_banks)
            print(f"Inkling ROCm layer {layer_idx}: pre-tiled mixed {projections}", flush=True)

    def _load_coda_c1(self, layer_idx):
        from safetensors import safe_open

        expected_metadata = {
            "format": "inkling-c1-g16-v1",
            "layer": str(layer_idx),
            "experts": str(self.num_experts),
            "w13_layout": "gate-up-interleaved",
            "checkpoint_revision": "6e7c8248f75265bdd4fbab5273304243b5d8c2cb",
        }
        tensors = {
            f"{projection}.{field}": getattr(self, f"{projection}_{suffix}")
            for projection in PROJECTIONS
            for field, suffix in CODA_C1_FIELDS.items()
        }
        with safe_open(self.coda_c1_path, framework="pt", device="cpu") as bank:
            metadata = bank.metadata() or {}
            if any(metadata.get(key) != value for key, value in expected_metadata.items()):
                raise ValueError(f"{self.coda_c1_path}: unexpected Inkling coda C1 metadata")
            if set(bank.keys()) != tensors.keys():
                raise ValueError(f"{self.coda_c1_path}: unexpected Inkling coda C1 tensor names")
            for key, target in tensors.items():
                weight = bank.get_tensor(key)
                if weight.shape != target.shape or weight.dtype != target.dtype:
                    raise ValueError(
                        f"{self.coda_c1_path}: {key} expected {tuple(target.shape)} {target.dtype}, "
                        f"got {tuple(weight.shape)} {weight.dtype}"
                    )
                target.copy_(weight)

    def _c1_aux(self, name):
        return getattr(self, f"{name}_c1_row"), getattr(self, f"{name}_c1_table")

    def dequant(self, name, expert):
        weights, scales = self.packed(name)
        q, s = weights[expert], scales[expert]
        if not self.nvfp4 and self.coda_c1_path is None:
            return (q.float() * torch.exp2(s.float() - 127).repeat_interleave(32, dim=-1)).bfloat16()
        packed = q.int()
        codes = torch.stack((packed & 15, packed >> 4), dim=-1).flatten(1)
        kind = 2 if self.coda_c1_path is not None else self._kinds[name].get(expert, 0)
        if kind == 0:
            outer = getattr(self, f"{name}_outer")[expert]
            return (self.e2m1[codes] * s.float().repeat_interleave(16, dim=-1) * outer).bfloat16()
        if kind == 1:
            scale = s.view(torch.uint8).view(torch.bfloat16).float()
            return (((codes ^ 8) - 8).float() * scale.repeat_interleave(32, dim=-1)).bfloat16()
        row, table = self._c1_aux(name)
        values = table[expert, s.view(torch.uint8).long()].float() * row[expert].float()[:, None, None]
        scale, offset = values.unbind(-1)
        return (codes.float() * scale.repeat_interleave(16, dim=-1)
                + offset.repeat_interleave(16, dim=-1)).bfloat16()

    def project_m1(self, name, x, ids):
        weights, scales = self.packed(name)
        if self.coda_c1_path is not None:
            from .ops.grouped_decode import grouped_projection

            return grouped_projection(x, weights, scales, ids, format="flash-c1", flash_aux=self._c1_aux(name))
        if not self.nvfp4:
            from .moe_mxfp8_decode import mxfp8_decode

            return mxfp8_decode(x, weights, scales, ids)
        from .ops.grouped_decode import grouped_projection

        outer = getattr(self, f"{name}_outer")
        kind = getattr(self, f"{name}_kind")
        if kind is None:
            if (
                self.source_m1_wave32 and self.normal_scales and ids.ndim == 1
                and x.shape[-1] in (1152, 2304) and x.data_ptr() % 16 == 0
                and weights.shape[1] % 4 == 0 and weights.data_ptr() % 4 == 0
            ):
                from .ops.source_wave_decode import source_wave_projection

                return source_wave_projection(x, weights, scales.view(torch.uint8), ids, outer)
            return grouped_projection(
                x, weights, scales.view(torch.uint8), ids, format="source-e2m1",
                outer_scales=outer, normal_scales=self.normal_scales,
                tile_rows=2 if name == "w13" else 4,
                short_k=self.source_m1_short_k,
                bf16_dot2=self.source_m1_bf16_dot2,
            )
        from .ops.mixed_grouped_decode import mixed_grouped_projection

        return mixed_grouped_projection(
            x, weights, scales.view(torch.uint8), ids, kind, outer,
            flash_aux=self._c1_aux(name),
            normal_scales=self.normal_scales,
        )

    def project_prefill(self, name, x, plan, *, fused_activation=None):
        from .moe_prefill import packed_prefill_projection

        turbo_prefill_mode = os.getenv("INKLING_AMD_TURBO_PREFILL_MOE", "0")
        turbo_prefill_on = turbo_prefill_mode not in ("", "0")
        if (
            turbo_prefill_on
            and (self.source_wmma_banks or self.verifier_wmma_banks)
            and (plan.block_m, plan.block_n, plan.block_k) == (64, 128, 64)
        ):
            bank = self.verifier_wmma_banks.get(name)
            if turbo_prefill_mode != "bf16":
                from .ops.hip_runtime import can_run_hip_moe_pair, hip_moe_prefill_projection

                allow_fp16 = (name == "w13") and (fused_activation is not None) and can_run_hip_moe_pair(self, x, plan)
                if bank is not None:
                    bkind = "coda" if self.coda_c1_path is not None else "mixed"
                    hip_out = hip_moe_prefill_projection(
                        x, bank, plan, bank_kind=bkind,
                        fused_activation=fused_activation,
                        out_fp16=allow_fp16,
                    )
                    if hip_out is not None:
                        return hip_out
                elif self.source_wmma_banks and name in self.source_wmma_banks:
                    hip_out = hip_moe_prefill_projection(
                        x, self.source_wmma_banks[name], plan, bank_kind="source",
                        fused_activation=fused_activation,
                        out_fp16=allow_fp16,
                    )
                    if hip_out is not None:
                        return hip_out
            if x.dtype != torch.bfloat16:
                x = x.to(torch.bfloat16)
            from .ops.turbo_kernels import (
                coda_wmma_prefill_m64,
                mixed_wmma_prefill_m64,
                source_wmma_turbo_projection,
            )

            if bank is not None:
                if self.coda_c1_path is not None:
                    if turbo_prefill_mode != "bf16":
                        from .ops.turbo_kernels import coda_wmma_turbo_projection

                        return coda_wmma_turbo_projection(x, bank, plan, fused_activation=fused_activation)
                    return coda_wmma_prefill_m64(x, bank, plan, fused_activation=fused_activation)
                return mixed_wmma_prefill_m64(x, bank, plan, fused_activation=fused_activation)
            if self.source_wmma_banks:
                return source_wmma_turbo_projection(
                    x, self.source_wmma_banks[name], plan,
                    fused_activation=fused_activation,
                    use_f16=(turbo_prefill_mode != "bf16"),
                )
        if (
            (self.source_wmma_banks or self.verifier_wmma_banks)
            and (plan.block_m, plan.block_n, plan.block_k) == (16, 128, 64)
            and (
                (2 <= plan.num_tokens <= 8
                 and get_forward_context().additional_kwargs.get("inkling_source_wmma_pretile"))
                or (turbo_prefill_on and 2 <= plan.num_tokens < 256)
            )
        ):
            bank = self.verifier_wmma_banks.get(name)
            use_hip_decode = (
                os.getenv("INKLING_AMD_TURBO_MOE_F16") == "1"
                or (turbo_prefill_on and turbo_prefill_mode != "bf16")
            )
            if use_hip_decode:
                from .ops.hip_runtime import can_run_hip_moe_decode_pair, hip_moe_decode_projection

                allow_fp16 = (
                    (name == "w13")
                    and (fused_activation is not None)
                    and can_run_hip_moe_decode_pair(self, x, plan)
                )
                if bank is not None:
                    bkind = "coda" if self.coda_c1_path is not None else "mixed"
                    hip_out = hip_moe_decode_projection(
                        x, bank, plan, bank_kind=bkind,
                        fused_activation=fused_activation,
                        out_fp16=allow_fp16,
                    )
                    if hip_out is not None:
                        return hip_out
                elif self.source_wmma_banks and name in self.source_wmma_banks:
                    hip_out = hip_moe_decode_projection(
                        x, self.source_wmma_banks[name], plan, bank_kind="source",
                        fused_activation=fused_activation,
                        out_fp16=allow_fp16,
                    )
                    if hip_out is not None:
                        return hip_out
            if x.dtype != torch.bfloat16:
                x = x.to(torch.bfloat16)
            if bank is not None:
                if self.coda_c1_path is not None:
                    if os.getenv("INKLING_AMD_TURBO_MOE_F16") == "1":
                        from .ops.turbo_kernels import coda_wmma_turbo_projection

                        return coda_wmma_turbo_projection(x, bank, plan, fused_activation=fused_activation)
                    from .ops.coda_wmma_pretile import coda_wmma_projection

                    return coda_wmma_projection(x, bank, plan, fused_activation=fused_activation)
                if os.getenv("INKLING_AMD_TURBO_MOE_F16") == "1":
                    from .ops.turbo_kernels import mixed_wmma_turbo_projection

                    return mixed_wmma_turbo_projection(x, bank, plan, fused_activation=fused_activation)
                from .ops.mixed_wmma_pretile import mixed_wmma_projection

                return mixed_wmma_projection(x, bank, plan, fused_activation=fused_activation)
            if self.source_wmma_banks:
                if os.getenv("INKLING_AMD_TURBO_MOE_F16") == "1" or (
                    turbo_prefill_on and plan.num_tokens != 8 and turbo_prefill_mode != "bf16"
                ):
                    from .ops.turbo_kernels import source_wmma_turbo_projection

                    return source_wmma_turbo_projection(
                        x, self.source_wmma_banks[name], plan,
                        fused_activation=fused_activation, use_f16=True,
                    )
                from .ops.source_wmma_pretile import source_wmma_projection

                return source_wmma_projection(
                    x, self.source_wmma_banks[name], plan, fused_activation=fused_activation,
                )
        weights, scales = self.packed(name)
        if self.coda_c1_path is not None:
            return packed_prefill_projection(
                x, weights, scales, self.coda_outer, plan,
                kind=self.coda_kind, flash_aux=self._c1_aux(name), normal_scales=False,
                fused_activation=fused_activation,
            )
        if not self.nvfp4:
            return packed_prefill_projection(
                x, weights, scales, None, plan, weight_format="mxfp8",
                fused_activation=fused_activation,
            )
        kind = getattr(self, f"{name}_kind")
        flash_aux = None if kind is None else self._c1_aux(name)
        return packed_prefill_projection(
            x, weights, scales.view(torch.uint8), getattr(self, f"{name}_outer"), plan,
            kind=kind, flash_aux=flash_aux, normal_scales=self.normal_scales,
            fused_activation=fused_activation,
        )


class Inkling2MoE(nn.Module):
    def __init__(
        self, config, quant_format: QuantFormat, visit_ids=(), *, layer_idx=None,
        coda_c1_path: Path | None = None,
    ):
        super().__init__()
        self.gate = Inkling2Gate(config)
        self.experts = Inkling2Experts(config, quant_format, coda_c1_path)
        self.gate_cap = config.mlp_gate_softcap
        self.up_cap = config.mlp_up_softcap
        self.layer_idx = layer_idx
        self.visit_ids = visit_ids
        self.packed_mxfp8 = os.getenv("INKLING_AMD_PACKED_MXFP8", "1") == "1"
        self.packed_small_m = os.getenv("INKLING_AMD_PACKED_SMALL_M", "0") == "1"
        self.probe_grouped_only = os.getenv("INKLING_PROBE_GROUPED_ONLY") == "1"
        self.packed_prefill = os.getenv("INKLING_AMD_PACKED_PREFILL", "0") == "1"
        self.packed_mxfp8_prefill = os.getenv("INKLING_AMD_PACKED_MXFP8_PREFILL", "0") == "1"
        self.prefill_fused_activation = os.getenv("INKLING_AMD_PREFILL_FUSED_ACTIVATION", "0") == "1"
        self.prefill_fused_reduce = os.getenv("INKLING_AMD_PREFILL_FUSED_REDUCE", "0") == "1"
        self.fused_m8_router = os.getenv("INKLING_AMD_FUSED_M8_ROUTER", "off")
        if self.fused_m8_router not in ("off", "tail", "align", "select"):
            raise ValueError("INKLING_AMD_FUSED_M8_ROUTER must be off, tail, align, or select")
        self._prepared = False

    def process_weights_after_loading(self):
        if self._prepared:
            return
        directory = os.getenv("INKLING_AMD_EXPERT_BANK")
        policy = os.getenv("INKLING_AMD_EXPERT_POLICY", "source")
        if policy not in POLICIES or (policy != "source" and directory is None):
            raise ValueError(f"Invalid Inkling AMD expert policy or missing bank: {policy}")
        self.gate.fp32_weight = self.gate.weight.float().detach()
        self.experts.setup(self.layer_idx, directory, policy)
        self._prepared = True

    def forward(self, hidden_states, *, pass_idx=0):
        if not self._prepared:
            self.process_weights_after_loading()
        tokens = hidden_states.shape[0]
        direct = tokens == 1 or (self.packed_small_m and 2 <= tokens <= 8)
        direct = direct and not self.probe_grouped_only
        packed_prefill = self.packed_prefill if self.experts.nvfp4 else self.packed_mxfp8_prefill
        fused_plan = None
        if (
            self.fused_m8_router != "off"
            and hidden_states.is_cuda
            and torch.version.hip
            and 2 <= hidden_states.shape[0] <= 8 and hidden_states.shape[1] == 2304
            and self.gate.topk == 8
            and self.experts.num_experts == 128
            and packed_prefill
            and not direct
            and get_forward_context().additional_kwargs.get("inkling_source_wmma_pretile")
        ):
            logits = self.gate.fp32_logits(hidden_states)
            if os.getenv("INKLING_AMD_TURBO_ROUTER_M8") == "1" and self.fused_m8_router == "align":
                from .ops.turbo_kernels import fast_route_m8

                ids, gamma, fused_plan = fast_route_m8(
                    logits, self.gate.bias, self.gate.route_scale,
                )
            else:
                from .ops.router_m8 import route_m8

                ids, gamma, fused_plan = route_m8(
                    logits, self.gate.bias, self.gate.route_scale, self.fused_m8_router,
                )
        else:
            ids, gamma = self.gate(hidden_states)
        if direct and hidden_states.is_cuda and (self.experts.nvfp4 or self.packed_mxfp8):
            from .ops.grouped_decode import inkling_reduce, inkling_softcap

            # Tiny prefill chunks also take this route since this layer receives token rows.
            x = hidden_states
            if tokens == 1:
                x, ids, gamma = x[0], ids.flatten(), gamma.flatten()
            gate_up = self.experts.project_m1("w13", x, ids)
            activated = inkling_softcap(gate_up, gamma, gate_cap=self.gate_cap, up_cap=self.up_cap)
            return inkling_reduce(self.experts.project_m1("w2", activated, ids)).reshape_as(hidden_states)

        if packed_prefill and hidden_states.is_cuda:
            from .moe_prefill import align_prefill_routes

            plan = align_prefill_routes(ids, self.experts.num_experts, packed=fused_plan)
            if self.prefill_fused_activation:
                mid = self.experts.project_prefill(
                    "w13", hidden_states, plan, fused_activation=(gamma, self.gate_cap, self.up_cap),
                )
            else:
                tile = self.experts.project_prefill("w13", hidden_states, plan)
                gate, up = tile[..., ::2].float(), tile[..., 1::2].float()
                mid = self.activate(gate, up, gamma)
            down = self.experts.project_prefill("w2", mid, plan)
            if self.prefill_fused_reduce:
                from .ops.hip_runtime import hip_moe_reduce

                reduced = hip_moe_reduce(down)
                if reduced is not None:
                    return reduced
                from .ops.grouped_decode import inkling_reduce

                return inkling_reduce(down)
            return down.float().sum(dim=1).to(hidden_states.dtype)

        # Only dequantize experts used in this batch.
        down = hidden_states.new_empty((*ids.shape, hidden_states.shape[-1]))
        for expert in ids.unique().tolist():
            row, slot = torch.where(ids == expert)
            tile = hidden_states[row] @ self.experts.dequant("w13", expert).T
            gate, up = tile[:, ::2].float(), tile[:, 1::2].float()
            mid = self.activate(gate, up, gamma[row, slot])
            down[row, slot] = mid @ self.experts.dequant("w2", expert).T
        return down.float().sum(dim=1).to(hidden_states.dtype)

    def activate(self, gate, up, gamma):
        return (self.gate_cap * torch.tanh(gate / self.gate_cap) * gate.sigmoid()
                * self.up_cap * torch.tanh(up / self.up_cap) * gamma[..., None]).bfloat16()
