"""Load CIRU's corrected ROCr before any Python package can initialize ROCm."""

from __future__ import annotations

import os
import re
import sys
from pathlib import Path


def _ensure_early_rocr_preload() -> None:
    if sys.platform != "linux" or os.environ.get(
            "CIRU_DISABLE_ROCR_IDLE_FIX") == "1":
        return

    runtime = (Path(__file__).resolve().parent / "_rocm_sdk_core" / "lib" /
               "libhsa-runtime64.so.1")
    if not runtime.is_file():
        raise RuntimeError(f"CIRU corrected ROCr runtime is missing: {runtime}")

    preload = os.environ.get("LD_PRELOAD", "")
    entries = [entry for entry in re.split(r"[\s:]+", preload) if entry]
    runtime_text = str(runtime)
    if runtime_text in entries or any(
            Path(entry).name.startswith("libhsa-runtime64.so")
            for entry in entries):
        return

    command_line = Path("/proc/self/cmdline").read_bytes().rstrip(b"\0").split(
        b"\0")
    if not command_line or not command_line[0]:
        raise RuntimeError("cannot reconstruct Python command line for ROCr preload")

    environment = os.environb.copy()
    runtime_bytes = os.fsencode(runtime_text)
    previous = environment.get(b"LD_PRELOAD", b"")
    environment[b"LD_PRELOAD"] = runtime_bytes + (b":" + previous
                                                     if previous else b"")
    os.execve(os.fsencode(sys.executable), command_line, environment)


_ensure_early_rocr_preload()
