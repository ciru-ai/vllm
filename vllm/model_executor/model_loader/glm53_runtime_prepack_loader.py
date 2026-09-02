# SPDX-License-Identifier: Apache-2.0
"""Versioned per-rank loader for final GLM-5.3 runtime tensors."""

from __future__ import annotations

import json
import os
import time
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import torch
from torch import nn

from vllm.config import ModelConfig
from vllm.config.load import LoadConfig
from vllm.distributed import (
    get_pp_group,
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
)
from vllm.logger import init_logger
from vllm.model_executor.model_loader.base_loader import BaseModelLoader

logger = init_logger(__name__)

_SCHEMA = "glm53_runtime_prepack_v1"
_LAYOUT = "glm53_iu4_n32_kword_g128_v1"
_MANIFEST = "glm53-runtime-prepack.json"
_PATTERN = "model-rank-{rank}-part-{part}.safetensors"
_F1_IMMUTABLE_BUFFERS = frozenset(
    {
        "f1_weight_s4_dot_i32",
        "f1_scale8_codes_dot",
        "f1_scale8_lut",
    }
)


def _module_and_leaf(name: str) -> tuple[str, str]:
    module_name, separator, leaf = name.rpartition(".")
    if not separator:
        raise ValueError(f"Runtime-prepack tensor has no module prefix: {name!r}")
    return module_name, leaf


class Glm53RuntimePrepackLoader(BaseModelLoader):
    """Load final per-rank tensors without repeating runtime conversion."""

    def __init__(self, load_config: LoadConfig):
        super().__init__(load_config)
        extra = dict(load_config.model_loader_extra_config or {})
        self.manifest_name = extra.pop("manifest", _MANIFEST)
        if extra:
            raise ValueError(
                "Unexpected glm53_runtime_prepack config keys: "
                f"{sorted(extra)}"
            )

    @staticmethod
    def _weights_root(model_config: ModelConfig) -> Path:
        value = model_config.model_weights or model_config.model
        return Path(value).expanduser().resolve()

    def download_model(self, model_config: ModelConfig) -> None:
        root = self._weights_root(model_config)
        if not root.is_dir():
            raise FileNotFoundError(f"GLM53 runtime prepack directory not found: {root}")

    def _load_manifest(self, root: Path) -> dict[str, Any]:
        path = root / self.manifest_name
        with path.open(encoding="utf-8") as handle:
            manifest = json.load(handle)
        if manifest.get("schema") != _SCHEMA:
            raise ValueError(
                f"Unsupported GLM53 runtime prepack schema "
                f"{manifest.get('schema')!r}; expected {_SCHEMA!r}"
            )
        if manifest.get("layout") != _LAYOUT:
            raise ValueError(
                f"Unsupported GLM53 runtime layout {manifest.get('layout')!r}; "
                f"expected {_LAYOUT!r}"
            )
        rank = get_tensor_model_parallel_rank()
        tp_size = get_tensor_model_parallel_world_size()
        if manifest.get("tp_rank") != rank or manifest.get("tp_size") != tp_size:
            raise ValueError(
                "GLM53 runtime prepack rank/topology mismatch: "
                f"artifact rank {manifest.get('tp_rank')}/{manifest.get('tp_size')}, "
                f"runtime rank {rank}/{tp_size}"
            )
        files = manifest.get("files")
        if not isinstance(files, list) or not files:
            raise ValueError("GLM53 runtime prepack manifest requires a files list")
        return manifest

    def load_weights(self, model: nn.Module, model_config: ModelConfig) -> None:
        from safetensors.torch import safe_open

        root = self._weights_root(model_config)
        manifest = self._load_manifest(root)
        modules = dict(model.named_modules())
        pending = dict(model.named_parameters(remove_duplicate=True))
        loaded = 0
        ignored_derived = 0
        started = time.perf_counter()

        for filename in manifest["files"]:
            path = root / filename
            if not path.is_file():
                raise FileNotFoundError(f"GLM53 runtime prepack part not found: {path}")
            with safe_open(path, framework="pt") as handle:
                for name in handle.keys():
                    tensor = handle.get_tensor(name)
                    module_name, leaf = _module_and_leaf(name)
                    module = modules.get(module_name)
                    if module is None:
                        raise KeyError(
                            f"GLM53 runtime prepack has unknown module {module_name!r}"
                        )

                    parameter = pending.pop(name, None)
                    if parameter is not None:
                        parameter.data = tensor.to(device=parameter.device)
                    elif leaf in _F1_IMMUTABLE_BUFFERS:
                        value = tensor.to(device=next(module.parameters()).device)
                        if leaf in module._buffers:
                            module._buffers[leaf] = value
                        else:
                            module.register_buffer(leaf, value, persistent=False)
                    else:
                        # Deferred-attention and model-level post-load hooks may
                        # register derived tensors (for example MLA W_UK_T).
                        # They are recreated by the normal hooks below, so do
                        # not inject or mark their modules as prepacked.
                        ignored_derived += 1
                        continue
                    module._vllm_runtime_prepacked = True
                    loaded += 1

        if pending:
            names = tuple(pending)[:20]
            raise ValueError(
                f"GLM53 runtime prepack is missing {len(pending)} parameters; "
                f"first keys: {names}"
            )
        logger.info_once(
            "Loading GLM53 runtime prepack took %.2f seconds "
            "(%d tensors, %d derived tensors deferred)",
            time.perf_counter() - started,
            loaded,
            ignored_derived,
        )

    @staticmethod
    def _iter_runtime_tensors(model: nn.Module) -> Iterable[tuple[str, torch.Tensor]]:
        yield from model.named_parameters(remove_duplicate=True)
        for module_name, module in model.named_modules():
            for leaf in _F1_IMMUTABLE_BUFFERS:
                value = getattr(module, leaf, None)
                if isinstance(value, torch.Tensor):
                    yield f"{module_name}.{leaf}", value

    @staticmethod
    def save_model(
        model: nn.Module,
        model_config: ModelConfig,
        path: str,
        pattern: str | None = None,
        max_size: int | None = None,
    ) -> None:
        from safetensors.torch import save_file

        root = Path(path).expanduser().resolve()
        root.mkdir(parents=True, exist_ok=True)
        rank = get_tensor_model_parallel_rank()
        pattern = pattern or _PATTERN
        max_size = max_size or 8 * 1024**3
        part_index = 0
        part_bytes = 0
        total_bytes = 0
        part: dict[str, torch.Tensor] = {}
        files: list[str] = []

        def flush() -> None:
            nonlocal part_index, part_bytes, part
            if not part:
                return
            filename = pattern.format(rank=rank, part=part_index)
            save_file(part, root / filename)
            files.append(filename)
            part_index += 1
            part_bytes = 0
            part = {}

        for name, tensor in Glm53RuntimePrepackLoader._iter_runtime_tensors(model):
            size = tensor.numel() * tensor.element_size()
            if part and part_bytes + size > max_size:
                flush()
            part[name] = tensor.detach().to(device="cpu").contiguous()
            part_bytes += size
            total_bytes += size
        flush()

        manifest = {
            "schema": _SCHEMA,
            "layout": _LAYOUT,
            "source_model": model_config.model,
            "architecture": getattr(model_config.hf_config, "architectures", []),
            "quantization": model_config.quantization,
            "dtype": str(model_config.dtype),
            "device_arch": "gfx1151",
            "tp_rank": rank,
            "tp_size": get_tensor_model_parallel_world_size(),
            "pp_size": get_pp_group().world_size,
            "files": files,
            "total_bytes": total_bytes,
            "f1_buffers": sorted(_F1_IMMUTABLE_BUFFERS),
            "scratch_policy": "allocate_on_load",
        }
        with (root / _MANIFEST).open("w", encoding="utf-8") as handle:
            json.dump(manifest, handle, indent=2, sort_keys=True)
            handle.write("\n")
        logger.info(
            "Saved GLM53 runtime prepack rank %d: %d parts, %.2f GiB",
            rank,
            len(files),
            total_bytes / 1024**3,
        )

    def _prepare_weights(self, model_name_or_path: str, revision: str | None):
        del revision
        root = Path(model_name_or_path).expanduser().resolve()
        if not root.is_dir():
            raise FileNotFoundError(f"GLM53 runtime prepack directory not found: {root}")
        return str(root)
