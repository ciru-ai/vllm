# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 CIRU AI.
"""Loader & runtime wrappers for native RDNA 3.5 prefill kernels (MoE, KV-conv, AttnRes)."""

from __future__ import annotations

import ctypes
import glob
import hashlib
import logging
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

import torch

logger = logging.getLogger(__name__)

_HIP_LIB: ctypes.CDLL | None = None
_HIP_INIT_DONE: bool = False
_SCRATCH: dict[torch.device, tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = {}
_LAST_PLAN: dict[torch.device, object] = {}


def _preload_amdhip() -> None:
    """Ensure libamdhip64.so.7 from the active ROCm Python env is globally visible."""
    patterns = [
        os.path.join(sys.prefix, "lib", "python*", "site-packages", "_rocm_sdk_core", "lib", "libamdhip64.so.7"),
        os.path.join(sys.prefix, "lib", "python*", "site-packages", "_rocm_sdk_core", "lib", "libamdhip64.so"),
        "/opt/rocm/lib/libamdhip64.so.7",
        "/opt/rocm/lib/libamdhip64.so",
    ]
    rocm_path = os.getenv("ROCM_PATH") or os.getenv("ROCM_HOME")
    if rocm_path:
        patterns.insert(0, os.path.join(rocm_path, "lib", "libamdhip64.so.7"))
    for pat in patterns:
        for cand in sorted(glob.glob(pat)):
            if os.path.isfile(cand):
                try:
                    ctypes.CDLL(cand, mode=ctypes.RTLD_GLOBAL)
                    return
                except OSError:
                    pass


def _find_hipcc() -> tuple[str | None, str | None]:
    """Locate a ROCm hipcc binary that has device libraries alongside it."""
    explicit = os.getenv("INKLING_AMD_HIPCC")
    if explicit and os.path.isfile(explicit) and os.access(explicit, os.X_OK):
        return explicit, str(Path(explicit).resolve().parent.parent)

    candidates: list[str] = []
    for pattern in (
        os.path.join(sys.prefix, "lib", "python*", "site-packages", "_rocm_sdk_devel", "bin", "hipcc"),
        "/opt/rocm/bin/hipcc",
    ):
        candidates.extend(sorted(glob.glob(pattern)))
    rocm_path = os.getenv("ROCM_PATH") or os.getenv("ROCM_HOME")
    if rocm_path:
        candidates.insert(0, os.path.join(rocm_path, "bin", "hipcc"))
    in_path = shutil.which("hipcc")
    if in_path:
        candidates.append(in_path)

    for cand in candidates:
        if not (os.path.isfile(cand) and os.access(cand, os.X_OK)):
            continue
        rocm_root = str(Path(cand).resolve().parent.parent)
        has_devlib = (
            os.path.isdir(os.path.join(rocm_root, "amdgcn", "bitcode"))
            or os.path.isdir(os.path.join(rocm_root, "lib", "llvm", "amdgcn", "bitcode"))
            or os.path.isdir(os.path.join(rocm_root, "llvm", "amdgcn", "bitcode"))
        )
        if has_devlib:
            return cand, rocm_root
    return None, None


def _detect_nix_cxx_flags() -> tuple[list[str], dict[str, str]]:
    """Detect NixOS GCC/glibc include & library flags if clang cannot find <cmath>."""
    extra_flags: list[str] = []
    extra_env: dict[str, str] = {}
    gxx = shutil.which("g++")
    if not gxx:
        return extra_flags, extra_env
    resolved = os.path.realpath(gxx)
    if "/nix/store/" not in resolved:
        return extra_flags, extra_env

    gcc_root = str(Path(resolved).parent.parent)
    inc_dirs: list[str] = []
    try:
        proc = subprocess.run(
            [resolved, "-E", "-x", "c++", "-", "-v"],
            input="", text=True, capture_output=True, timeout=5,
        )
        in_block = False
        for line in proc.stderr.splitlines():
            if "#include <...> search starts here:" in line:
                in_block = True
                continue
            if "End of search list." in line:
                break
            if in_block:
                p = line.strip()
                if p and os.path.isdir(p):
                    inc_dirs.append(p)
                    if "/include/c++/" in p and "-gcc-" in p:
                        gcc_root = p.split("/include/c++/")[0]
    except Exception:
        pass
    if inc_dirs:
        cur = os.environ.get("CPLUS_INCLUDE_PATH", "")
        extra_env["CPLUS_INCLUDE_PATH"] = ":".join(inc_dirs + ([cur] if cur else []))
    extra_flags.append(f"--gcc-toolchain={gcc_root}")
    for lib_name in ("libstdc++.so", "crti.o"):
        try:
            out = subprocess.check_output(
                [resolved, f"-print-file-name={lib_name}"], text=True, timeout=5,
            ).strip()
            if out and os.path.isfile(out):
                ldir = os.path.dirname(out)
                if lib_name == "crti.o":
                    extra_flags.append(f"-B{ldir}")
                extra_flags.append(f"-L{ldir}")
                if lib_name == "libstdc++.so":
                    extra_flags.append(f"-Wl,-rpath,{ldir}")
        except Exception:
            pass
    return extra_flags, extra_env


def _get_hip_lib() -> ctypes.CDLL | None:
    global _HIP_LIB, _HIP_INIT_DONE
    if _HIP_INIT_DONE:
        return _HIP_LIB
    _HIP_INIT_DONE = True

    if os.getenv("INKLING_AMD_HIP_PREFILL_MOE", "1") in ("0", "off", "false"):
        return None
    if not torch.cuda.is_available() or not torch.version.hip:
        return None
    arch = getattr(torch.cuda.get_device_properties(0), "gcnArchName", "").split(":")[0]
    if arch != "gfx1151":
        return None

    src_path = Path(__file__).with_name("hip_moe_prefill.hip")
    src_hash = hashlib.sha256(src_path.read_bytes()).hexdigest()[:12] if src_path.is_file() else None
    so_name = f"hip_moe_prefill_{arch}_{src_hash}.so" if src_hash else f"hip_moe_prefill_{arch}.so"

    _preload_amdhip()

    # Tier 1: Prebuilt portable .so via env var or bundled alongside source (no hipcc needed)
    prebuilt_candidates: list[Path] = []
    explicit_so = os.getenv("INKLING_AMD_HIP_PREFILL_SO")
    if explicit_so:
        prebuilt_candidates.append(Path(explicit_so))
    if src_hash:
        prebuilt_candidates.append(Path(__file__).with_name(so_name))
    for cand in prebuilt_candidates:
        if cand.is_file():
            try:
                _HIP_LIB = ctypes.CDLL(str(cand))
                logger.info("Loaded Inkling2 prebuilt native HIP prefill kernels: %s", cand.name)
                return _HIP_LIB
            except Exception as exc:
                logger.warning("Failed to load prebuilt HIP prefill .so (%s): %s", cand, exc)

    if not src_path.is_file() or not src_hash:
        return None

    # Tier 2: Cached or auto-JIT compiled .so via local hipcc
    cache_dir = Path(os.getenv("XDG_CACHE_HOME", os.path.expanduser("~/.cache"))) / "vllm" / "inkling2_hip"
    cache_dir.mkdir(parents=True, exist_ok=True)
    so_path = cache_dir / so_name

    if not so_path.is_file():
        hipcc, rocm_root = _find_hipcc()
        if not hipcc:
            logger.info("Inkling2 HIP prefill: hipcc with device libs not found; using Gluon/Triton fallback.")
            return None
        extra_flags, extra_env = _detect_nix_cxx_flags()
        env = os.environ.copy()
        env.update(extra_env)
        if rocm_root:
            env["ROCM_PATH"] = rocm_root
            env["ROCM_HOME"] = rocm_root
            env["HIP_PATH"] = rocm_root
            extra_flags.extend([f"-L{rocm_root}/lib", f"-Wl,-rpath,{rocm_root}/lib"])
        tmp_fd, tmp_so = tempfile.mkstemp(prefix=".hip_moe_", suffix=".so", dir=str(cache_dir))
        os.close(tmp_fd)
        cmd = [
            hipcc, "-shared", "-fPIC", "-O3", f"--offload-arch={arch}",
            "-Wl,--as-needed", str(src_path), "-o", tmp_so, *extra_flags,
        ]
        try:
            res = subprocess.run(cmd, env=env, capture_output=True, text=True, timeout=120)
            if res.returncode != 0:
                logger.warning("Inkling2 HIP prefill compile failed (using fallback): %s", res.stderr[-600:])
                try:
                    os.unlink(tmp_so)
                except OSError:
                    pass
                return None
            os.replace(tmp_so, str(so_path))
        except Exception as exc:
            logger.warning("Inkling2 HIP prefill build error (using fallback): %s", exc)
            try:
                os.unlink(tmp_so)
            except OSError:
                pass
            return None

    try:
        _HIP_LIB = ctypes.CDLL(str(so_path))
        logger.info("Loaded Inkling2 native HIP prefill kernels: %s", so_path.name)
    except Exception as exc:
        logger.warning("Failed to load Inkling2 HIP prefill .so (using fallback): %s", exc)
        _HIP_LIB = None
    return _HIP_LIB


def _get_scratch(device: torch.device, max_m64: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    entry = _SCRATCH.get(device)
    need_pairs = max_m64 // 2 + 2
    need_tails = max_m64 + 2
    if entry is None or entry[0].numel() < need_pairs or entry[1].numel() < need_tails:
        pair_idx = torch.empty(max(need_pairs, 2048), device=device, dtype=torch.int32)
        tail_idx = torch.empty(max(need_tails, 4096), device=device, dtype=torch.int32)
        counts = torch.empty(2, device=device, dtype=torch.int32)
        entry = (pair_idx, tail_idx, counts)
        _SCRATCH[device] = entry
        _LAST_PLAN.pop(device, None)
    return entry


def hip_moe_prefill_available() -> bool:
    return _get_hip_lib() is not None


def hip_moe_prefill_projection(
    x: torch.Tensor,
    bank,
    plan,
    *,
    bank_kind: str,  # "source", "mixed", "coda"
    fused_activation=None,
    out_fp16: bool = False,
) -> torch.Tensor | None:
    """Run paired-M128 + M64-tail native HIP WMMA MoE prefill projection."""
    lib = _get_hip_lib()
    if lib is None or (plan.block_m, plan.block_n, plan.block_k) != (64, 128, 64):
        return None
    if plan.top_k != 8 or plan.num_experts != 128:
        return None

    experts, n_tiles, k_tiles, _, _ = bank.packed.shape
    n, k = n_tiles * 128, k_tiles * 64
    route_shape = (plan.num_tokens, plan.top_k)
    gamma, gate_cap, up_cap = None, 0.0, 0.0
    fused = fused_activation is not None
    if fused:
        gamma, gate_cap, up_cap = fused_activation

    # Direct BF16 X-load on w13 avoids x.to(float16); w2 consumes FP16 directly from w13.
    use_xbf16 = fused and x.dtype == torch.bfloat16
    if use_xbf16:
        x_in = x if x.is_contiguous() else x.contiguous()
    else:
        x_in = x if x.dtype == torch.float16 else x.to(torch.float16)
        if not x_in.is_contiguous():
            x_in = x_in.contiguous()

    out_dtype = torch.float16 if (fused and out_fp16) else torch.bfloat16
    out = x.new_empty((*route_shape, n // 2 if fused else n), dtype=out_dtype)

    max_m64 = plan.expert_ids.numel()
    pair_idx, tail_idx, counts = _get_scratch(x.device, max_m64)
    stream = torch.cuda.current_stream(x.device).cuda_stream

    # Reuse pair/tail scan when w2 runs on the exact same immutable PrefillRoutes plan as w13.
    if _LAST_PLAN.get(x.device) is not plan or torch.cuda.is_current_stream_capturing():
        lib.launch_pair_m64(
            ctypes.c_void_p(plan.expert_ids.data_ptr()),
            ctypes.c_void_p(plan.padded_tokens.data_ptr()),
            ctypes.c_void_p(pair_idx.data_ptr()),
            ctypes.c_void_p(tail_idx.data_ptr()),
            ctypes.c_void_p(counts.data_ptr()),
            ctypes.c_int(max_m64),
            ctypes.c_void_p(stream),
        )
        if not torch.cuda.is_current_stream_capturing():
            _LAST_PLAN[x.device] = plan

    suf = "_xbf16" if use_xbf16 else ""
    if bank_kind == "source":
        raw_ptr = bank.raw_scales.data_ptr()
        outer_ptr = bank.outer_scales.data_ptr()
        rs_ptr = tab_ptr = kind_ptr = 0
        fn_name = (f"launch_source_w13_{'f16' if out_fp16 else 'bf16'}{suf}") if fused else "launch_source_w2"
    elif bank_kind == "mixed":
        raw_ptr = bank.raw_scales.data_ptr()
        outer_ptr = bank.outer_scales.data_ptr()
        kind_ptr = bank.kind.data_ptr()
        fk = bank.fitted_kind
        rs_ptr = bank.row_scales.data_ptr() if bank.row_scales is not None else 0
        tab_ptr = bank.table.data_ptr() if bank.table is not None else 0
        prefix = f"launch_mixed{fk}"
        fn_name = (f"{prefix}_w13_{'f16' if out_fp16 else 'bf16'}{suf}") if fused else f"{prefix}_w2"
    elif bank_kind == "coda":
        raw_ptr = bank.tags.data_ptr()
        outer_ptr = 0
        kind_ptr = 0
        rs_ptr = bank.row_scales.data_ptr()
        tab_ptr = bank.table.data_ptr()
        fn_name = (f"launch_coda_w13_{'f16' if out_fp16 else 'bf16'}{suf}") if fused else "launch_coda_w2"
    else:
        return None

    fn = getattr(lib, fn_name)
    fn(
        ctypes.c_void_p(x_in.data_ptr()),
        ctypes.c_void_p(bank.packed.data_ptr()),
        ctypes.c_void_p(raw_ptr),
        ctypes.c_void_p(outer_ptr),
        ctypes.c_void_p(rs_ptr),
        ctypes.c_void_p(tab_ptr),
        ctypes.c_void_p(kind_ptr),
        ctypes.c_void_p(plan.sorted_ids.data_ptr()),
        ctypes.c_void_p(plan.expert_ids.data_ptr()),
        ctypes.c_void_p(pair_idx.data_ptr()),
        ctypes.c_void_p(tail_idx.data_ptr()),
        ctypes.c_void_p(counts.data_ptr()),
        ctypes.c_void_p(out.data_ptr()),
        ctypes.c_void_p(gamma.data_ptr() if gamma is not None else 0),
        ctypes.c_int(n),
        ctypes.c_int(k),
        ctypes.c_int(plan.num_tokens * plan.top_k),
        ctypes.c_int(max_m64),
        ctypes.c_float(gate_cap),
        ctypes.c_float(up_cap),
        ctypes.c_void_p(stream),
    )
    return out


def hip_kv_conv_prefill(
    k: torch.Tensor,
    v: torch.Tensor,
    k_weight: torch.Tensor,
    v_weight: torch.Tensor,
    k_dynamic: torch.Tensor | None,
    v_dynamic: torch.Tensor | None,
    cache: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    kv_slot_mapping: torch.Tensor,
    positions: torch.Tensor,
    seq_idx: torch.Tensor,
    slot_mapping: torch.Tensor,
    block_table: torch.Tensor,
    query_start: torch.Tensor,
    *,
    num_tokens: int,
    num_heads: int,
    head_dim: int,
    width: int,
    scale: float,
    clip_threshold: float,
    clip_eps: float,
    norm_eps: float,
) -> bool:
    """Run native 1-warp (zero-barrier, zero-LDS, uint2) fused K+V short-conv prefill."""
    lib = _get_hip_lib()
    if lib is None or num_tokens < 256 or head_dim != 128 or width != 4:
        return False
    dynamic = k_dynamic is not None
    if (
        k.dtype != torch.bfloat16
        or v.dtype != torch.bfloat16
        or k_weight.dtype != torch.bfloat16
        or v_weight.dtype != torch.bfloat16
        or cache.dtype != torch.bfloat16
        or key_cache.dtype != torch.bfloat16
        or value_cache.dtype != torch.bfloat16
        or k.stride(1) != 1
        or v.stride(1) != 1
        or cache.stride(3) != 1
        or key_cache.stride(3) != 1
        or value_cache.stride(3) != 1
        or (k.data_ptr() & 7) != 0
        or (v.data_ptr() & 7) != 0
        or (cache.data_ptr() & 7) != 0
        or (key_cache.data_ptr() & 7) != 0
        or (value_cache.data_ptr() & 7) != 0
        or (k.stride(0) & 3) != 0
        or (v.stride(0) & 3) != 0
        or (cache.stride(0) & 3) != 0
        or (cache.stride(1) & 3) != 0
        or (cache.stride(2) & 3) != 0
        or (key_cache.stride(0) & 3) != 0
        or (key_cache.stride(1) & 3) != 0
        or (key_cache.stride(2) & 3) != 0
    ):
        return False
    if dynamic:
        if (
            v_dynamic is None
            or k_dynamic.dtype != torch.bfloat16
            or v_dynamic.dtype != torch.bfloat16
            or k_dynamic.stride(1) != 1
            or v_dynamic.stride(1) != 1
            or (k_dynamic.data_ptr() & 7) != 0
            or (v_dynamic.data_ptr() & 7) != 0
            or (k_dynamic.stride(0) & 3) != 0
            or (v_dynamic.stride(0) & 3) != 0
            or k_weight.stride(1) != 1
            or v_weight.stride(1) != 1
            or (k_weight.data_ptr() & 7) != 0
            or (v_weight.data_ptr() & 7) != 0
            or (k_weight.stride(0) & 3) != 0
            or (v_weight.stride(0) & 3) != 0
        ):
            return False
        dk_ptr = k_dynamic.data_ptr()
        dv_ptr = v_dynamic.data_ptr()
        sdk = k_dynamic.stride(0)
        sdv = v_dynamic.stride(0)
    else:
        dk_ptr = k_weight.data_ptr()
        dv_ptr = v_weight.data_ptr()
        sdk = 0
        sdv = 0

    stream = torch.cuda.current_stream(k.device).cuda_stream
    lib.launch_hip_kv_conv_1w(
        ctypes.c_void_p(k.data_ptr()),
        ctypes.c_void_p(v.data_ptr()),
        ctypes.c_void_p(k_weight.data_ptr()),
        ctypes.c_void_p(v_weight.data_ptr()),
        ctypes.c_void_p(dk_ptr),
        ctypes.c_void_p(dv_ptr),
        ctypes.c_void_p(cache.data_ptr()),
        ctypes.c_void_p(key_cache.data_ptr()),
        ctypes.c_void_p(value_cache.data_ptr()),
        ctypes.c_void_p(kv_slot_mapping.data_ptr()),
        ctypes.c_void_p(positions.data_ptr()),
        ctypes.c_void_p(seq_idx.data_ptr()),
        ctypes.c_void_p(slot_mapping.data_ptr()),
        ctypes.c_void_p(block_table.data_ptr()),
        ctypes.c_void_p(query_start.data_ptr()),
        ctypes.c_int(num_tokens),
        ctypes.c_int(num_heads),
        ctypes.c_int64(k.stride(0)),
        ctypes.c_int64(v.stride(0)),
        ctypes.c_int64(k_weight.stride(0)),
        ctypes.c_int64(k_weight.stride(1)),
        ctypes.c_int64(sdk),
        ctypes.c_int64(sdv),
        ctypes.c_int64(cache.stride(0)),
        ctypes.c_int64(cache.stride(1)),
        ctypes.c_int64(cache.stride(2)),
        ctypes.c_int64(key_cache.stride(0)),
        ctypes.c_int64(key_cache.stride(1)),
        ctypes.c_int64(key_cache.stride(2)),
        ctypes.c_int(key_cache.shape[1]),
        ctypes.c_int64(block_table.stride(0)),
        ctypes.c_int(block_table.shape[1]),
        ctypes.c_int(cache.shape[2]),
        ctypes.c_float(scale),
        ctypes.c_float(clip_threshold),
        ctypes.c_float(clip_eps),
        ctypes.c_float(norm_eps),
        ctypes.c_int(1 if dynamic else 0),
        ctypes.c_void_p(stream),
    )
    return True


def hip_attn_res_q4_coeffs(
    bank0: torch.Tensor,
    bank1: torch.Tensor,
    queries: torch.Tensor,
    eps: float = 1e-6,
) -> torch.Tensor | None:
    """Compute [4, T, 2] FP32 linear combination coeffs for normalized 2-entry bank."""
    lib = _get_hip_lib()
    if lib is None:
        return None
    if (
        bank0.ndim != 2
        or bank1.shape != bank0.shape
        or bank0.shape[0] < 64
        or bank0.shape[1] != 2304
        or bank0.dtype != torch.bfloat16
        or bank1.dtype != torch.bfloat16
        or queries.shape != (4, 2304)
        or queries.dtype != torch.float32
        or bank0.stride(1) != 1
        or bank1.stride(1) != 1
        or queries.stride(1) != 1
        or (bank0.data_ptr() & 15) != 0
        or (bank1.data_ptr() & 15) != 0
        or (queries.data_ptr() & 15) != 0
        or (bank0.stride(0) & 7) != 0
        or (bank1.stride(0) & 7) != 0
        or (queries.stride(0) & 3) != 0
    ):
        return None
    t_len = bank0.shape[0]
    coeffs = queries.new_empty((4, t_len, 2), dtype=torch.float32)
    stream = torch.cuda.current_stream(bank0.device).cuda_stream
    lib.launch_q4_coeffs(
        ctypes.c_void_p(bank0.data_ptr()),
        ctypes.c_void_p(bank1.data_ptr()),
        ctypes.c_void_p(queries.data_ptr()),
        ctypes.c_void_p(coeffs.data_ptr()),
        ctypes.c_int64(bank0.stride(0)),
        ctypes.c_int64(bank1.stride(0)),
        ctypes.c_int64(queries.stride(0)),
        ctypes.c_int(t_len),
        ctypes.c_float(eps),
        ctypes.c_void_p(stream),
    )
    return coeffs
