#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shlex
import shutil

import torch
from torch.utils.cpp_extension import load


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    tool_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser()
    parser.add_argument("--build-dir", type=Path, default=tool_dir / "build")
    args = parser.parse_args()

    build_dir = args.build_dir.resolve()
    build_dir.mkdir(parents=True, exist_ok=True)
    source_files = [
        tool_dir / "ling_s4s4_op.cu",
        tool_dir / "ling_decode_iu4_op.cu",
    ]
    sources = []
    for source in source_files:
        staged = build_dir / source.name
        shutil.copy2(source, staged)
        sources.append(staged)
    library = load(
        name="ling_decode_iu4_ext",
        sources=[str(path) for path in sources],
        extra_cflags=["-O3", "-std=c++17"],
        extra_cuda_cflags=[
            "-O3",
            "-std=c++17",
            "--offload-arch=gfx1151",
            "-Rpass-analysis=kernel-resource-usage",
        ] + shlex.split(os.environ.get("LING_HIP_FLAGS", "")),
        extra_ldflags=shlex.split(os.environ.get("LING_LINK_FLAGS", "")),
        build_directory=str(build_dir),
        with_cuda=True,
        is_python_module=False,
        verbose=True,
    )
    del library
    output = build_dir / "ling_decode_iu4_ext.so"
    manifest = {
        "library": str(output),
        "library_sha256": sha256(output),
        "sources": {path.name: sha256(path) for path in source_files},
        "torch_version": torch.__version__,
        "torch_hip": torch.version.hip,
    }
    (build_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, sort_keys=True))


if __name__ == "__main__":
    main()
