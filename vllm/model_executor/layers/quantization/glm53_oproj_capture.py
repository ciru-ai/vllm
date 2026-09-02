"""Opt-in exact-M8 pre-OProj capture for the GLM KDA layers.

The model worker imports this helper only through the isolated integration
copy.  Capture starts when the configured window file appears and finalizes
when the campaign removes that file.  Atexit is only a fallback for an
interrupted campaign.
"""

from __future__ import annotations

import atexit
import os
import threading
import time
from pathlib import Path

import torch
from safetensors.torch import save_file


KDA_LAYERS = (
    0, 1, 2, 4, 5, 6, 8, 9, 10, 12, 13, 14, 16, 17, 18, 20, 21, 22,
    24, 25, 26, 28, 29, 30, 32, 33, 34, 36, 37, 38, 40, 41, 42, 44,
)
K = 4096
M = 8

_LOCK = threading.Lock()
_BATCHES: dict[int, list[torch.Tensor]] = {layer: [] for layer in KDA_LAYERS}
_WATCHER_STARTED = False
_FINALIZED = False


def capture_enabled() -> bool:
    return os.getenv("VLLM_GLM53_OPROJ_CAPTURE_ENABLED", "").lower() in {
        "1", "true", "yes", "on"
    }


def _window_path() -> Path:
    value = os.environ.get("VLLM_GLM53_OPROJ_CAPTURE_WINDOW", "")
    if not value:
        raise ValueError("OProj capture requires ..._CAPTURE_WINDOW")
    return Path(value)


def _output_path() -> Path:
    value = os.environ.get("VLLM_GLM53_OPROJ_CAPTURE_OUTPUT", "")
    if not value:
        raise ValueError("OProj capture requires ..._CAPTURE_OUTPUT")
    return Path(value)


def _max_batches() -> int:
    return int(os.environ.get("VLLM_GLM53_OPROJ_CAPTURE_MAX_BATCHES", "128"))


def _watch_window() -> None:
    window = _window_path()
    while window.exists():
        time.sleep(0.2)
    finalize_capture()


def capture_input(layer_index: int, value: torch.Tensor) -> None:
    """Retain one rank-local exact-M8 BF16 input batch on CPU."""

    global _WATCHER_STARTED
    if (
        not capture_enabled()
        or layer_index not in _BATCHES
        or value.dtype != torch.bfloat16
        or value.ndim != 2
        or tuple(value.shape) != (M, K)
        or not _window_path().exists()
    ):
        return

    # This run measures calibration, not speed.  The synchronous 64-KiB copy
    # keeps GPU memory flat and makes finalization independent of CUDA teardown.
    host_batch = value.detach().to(device="cpu").contiguous()
    with _LOCK:
        rows = _BATCHES[layer_index]
        if len(rows) < _max_batches():
            rows.append(host_batch)
        if not _WATCHER_STARTED:
            _WATCHER_STARTED = True
            threading.Thread(
                target=_watch_window,
                name="glm53-oproj-capture-finalizer",
                daemon=True,
            ).start()


def finalize_capture() -> None:
    """Write one safetensors file with even-fit and odd-replay rows."""

    global _FINALIZED
    with _LOCK:
        if _FINALIZED:
            return
        _FINALIZED = True
        captured = {layer: batches for layer, batches in _BATCHES.items() if batches}
        for layer in captured:
            _BATCHES[layer] = []

    if not captured:
        return

    tensors: dict[str, torch.Tensor] = {}
    batch_counts: list[int] = []
    layer_ids: list[int] = []
    for layer in KDA_LAYERS:
        batches = captured.get(layer)
        if not batches:
            continue
        stacked = torch.stack(batches, dim=0)
        tensors[f"layer{layer:02d}.fit_even_bf16"] = (
            stacked[:, 0::2, :].reshape(-1, K).contiguous()
        )
        tensors[f"layer{layer:02d}.replay_odd_bf16"] = (
            stacked[:, 1::2, :].reshape(-1, K).contiguous()
        )
        layer_ids.append(layer)
        batch_counts.append(stacked.shape[0])

    tensors["layer_ids_i16"] = torch.tensor(layer_ids, dtype=torch.int16)
    tensors["batch_counts_i32"] = torch.tensor(batch_counts, dtype=torch.int32)
    output = _output_path()
    output.parent.mkdir(parents=True, exist_ok=True)
    save_file(
        tensors,
        output,
        metadata={
            "schema": "glm53-kda-oproj-m8-capture-v1",
            "fit_rows": "batch positions 0,2,4,6",
            "replay_rows": "batch positions 1,3,5,7",
            "dtype": "BF16",
        },
    )
    print(
        "GLM53_OPROJ_CAPTURE="
        + str({"output": str(output), "layers": len(layer_ids),
               "batches_min": min(batch_counts), "batches_max": max(batch_counts)}),
        flush=True,
    )


atexit.register(finalize_capture)
