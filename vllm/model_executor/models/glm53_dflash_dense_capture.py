"""One-shot exact-M8 calibration capture for the five DFlash MLPs."""

from __future__ import annotations

import atexit
import os
import threading
from pathlib import Path

import torch
from safetensors.torch import save_file


M = 8
K = 4096
LAYERS = range(5)
_LOCK = threading.Lock()
_BATCHES: dict[int, list[torch.Tensor]] = {layer: [] for layer in LAYERS}
_FINALIZED = False


def _max_batches() -> int:
    return int(os.environ.get("VLLM_GLM53_DFLASH_DENSE_CAPTURE_BATCHES", "32"))


def _output_path() -> Path:
    value = os.environ.get("VLLM_GLM53_DFLASH_DENSE_CAPTURE_OUTPUT", "")
    if not value:
        raise ValueError("DFlash dense capture requires an output path")
    return Path(value)


def capture_input(layer_index: int, value: torch.Tensor) -> None:
    """Capture a bounded set of real pre-gate/up M8 activations."""

    if (
        layer_index not in _BATCHES
        or value.dtype != torch.bfloat16
        or value.ndim != 2
        or tuple(value.shape) != (M, K)
    ):
        return

    limit = _max_batches()
    with _LOCK:
        if _FINALIZED or len(_BATCHES[layer_index]) >= limit:
            return
    host = value.detach().to(device="cpu").contiguous()
    should_finalize = False
    with _LOCK:
        rows = _BATCHES[layer_index]
        if len(rows) < limit:
            rows.append(host)
        should_finalize = all(len(_BATCHES[layer]) >= limit for layer in LAYERS)
    if should_finalize:
        finalize_capture()


def finalize_capture() -> None:
    global _FINALIZED
    with _LOCK:
        if _FINALIZED:
            return
        captured = {layer: list(rows) for layer, rows in _BATCHES.items() if rows}
        if not captured:
            return
        _FINALIZED = True

    tensors = {
        f"layer{layer}.input_bf16": torch.stack(rows).contiguous()
        for layer, rows in captured.items()
    }
    output = _output_path()
    output.parent.mkdir(parents=True, exist_ok=True)
    save_file(
        tensors,
        output,
        metadata={
            "schema": "glm53-dflash-dense-m8-capture-v1",
            "shape": "[batches,8,4096]",
        },
    )
    print(
        "GLM53_DFLASH_DENSE_CAPTURE="
        + str({"output": str(output), "batches": {k: len(v) for k, v in captured.items()}}),
        flush=True,
    )


atexit.register(finalize_capture)
