# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 CIRU AI.
"""Unified ABI-verified loader & runtime wrappers for Inkling 2 Mini gfx1151 HIP kernels."""

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

EXPECTED_HIP_ABI_VERSION = 3
_HIP_SOURCE_FILES = (
    "hip_common.cuh",
    "hip_moe_prefill.hip",
    "hip_attn_prefill.hip",
    "hip_glue_prefill.hip",
    "hip_attn_decode.hip",
    "hip_quant_decode.hip",
    "hip_moe_decode.hip",
    "hip_glue_decode.hip",
)
_HIP_TU_FILES = (
    "hip_moe_prefill.hip",
    "hip_attn_prefill.hip",
    "hip_glue_prefill.hip",
    "hip_attn_decode.hip",
    "hip_quant_decode.hip",
    "hip_moe_decode.hip",
    "hip_glue_decode.hip",
)
_CORE_REQUIRED_SYMBOLS = (
    "inkling_hip_abi_version",
    "inkling_hip_src_hash",
    "launch_pair_m64",
    "launch_source_w13_f16_xbf16",
    "launch_source_w2",
    "launch_hip_moe_reduce",
    "launch_hip_kv_conv_1w",
    "launch_hip_qr_clip",
    "launch_hip_gather_paged_kv_hnd",
    "launch_hip_flash_rel_attn_prefill",
    "launch_hip_flash_rel_attn_near_merge",
    "launch_q4_coeffs",
    "launch_hip_sconv_prefill",
    "launch_hip_loop_zeromat_prefill",
    "launch_hip_attn_res_prefill",
    "launch_hip_router_prefill",
    "launch_hip_kv_conv_decode",
    "launch_hip_flash_rel_decode_split",
    "launch_hip_decode_combine",
    "launch_hip_w6_wmma",
    "launch_hip_reduce_splits",
    "launch_hip_w6_head_gemv_m1",
    "launch_hip_w4_wmma",
    "launch_hip_w4_markov_argmax",
    "launch_hip_moe_decode_plan",
    "launch_hip_moe_decode_unified",
    "launch_hip_sconv_decode",
    "launch_hip_loop_decode_down",
    "launch_hip_loop_decode_blend_rms",
    "launch_hip_pair_rmsnorm",
    "launch_hip_blend_rmsnorm",
    "launch_hip_router_gemm_m8",
    "launch_hip_route_m8",
    "launch_hip_mix_rmsnorm",
    "launch_hip_rms_norm",
    "launch_hip_qk_norm_rope",
    "launch_hip_dspark_w8",
    "launch_hip_silu_and_mul",
    "launch_hip_add_softcap",
    "launch_hip_markov_gate",
)

_HIP_LIB: ctypes.CDLL | None = None
_HIP_INIT_DONE: bool = False
_HIP_LOADED_PATH: str | None = None
_SCRATCH: dict[torch.device, tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = {}
_DECODE_SCRATCH: dict[tuple[torch.device, str], torch.Tensor] = {}
_LAST_PLAN: dict[torch.device, object] = {}
_ALL_HIP_UNITS = (
    "moe",
    "attn",
    "glue",
    "decode_attn",
    "decode_quant",
    "decode_moe",
    "decode_glue",
)


def _is_disabled(val: str | None) -> bool:
    return val is not None and val.strip().lower() in ("0", "off", "false", "no")


def hip_unit_enabled(unit: str) -> bool:
    """Return True if the specified native HIP unit is enabled via env toggles."""
    if unit == "moe":
        primary = os.getenv("INKLING_AMD_HIP_MOE_PREFILL")
        if primary is not None:
            return not _is_disabled(primary)
        legacy = os.getenv("INKLING_AMD_HIP_PREFILL_MOE")
        if legacy is not None:
            return not _is_disabled(legacy)
        return True
    if unit == "attn":
        return not _is_disabled(os.getenv("INKLING_AMD_HIP_ATTN_PREFILL"))
    if unit == "glue":
        return not _is_disabled(os.getenv("INKLING_AMD_HIP_GLUE_PREFILL"))
    if unit == "decode_attn":
        return not _is_disabled(os.getenv("INKLING_AMD_HIP_DECODE_ATTN"))
    if unit == "decode_quant":
        return not _is_disabled(os.getenv("INKLING_AMD_HIP_DECODE_QUANT"))
    if unit == "decode_moe":
        return not _is_disabled(os.getenv("INKLING_AMD_HIP_DECODE_MOE"))
    if unit == "decode_glue":
        return not _is_disabled(os.getenv("INKLING_AMD_HIP_DECODE_GLUE"))
    raise ValueError(f"Unknown HIP unit: {unit!r}")


def _get_decode_scratch(
    device: torch.device,
    key: str,
    shape: tuple[int, ...],
    dtype: torch.dtype,
) -> torch.Tensor | None:
    """Return a graph-capture-safe scratch tensor backed by a per-(device, key) pool.

    Never allocates lazily when a HIP graph capture is active; returns None if the
    requested byte size exceeds the preallocated capacity during capture.
    """
    numel = 1
    for dim in shape:
        numel *= int(dim)
    elem_size = torch.tensor([], dtype=dtype).element_size()
    nbytes = max(numel * elem_size, 1)
    aligned = ((nbytes + 255) // 256) * 256
    pool_key = (device, key)
    buf = _DECODE_SCRATCH.get(pool_key)
    if buf is None or buf.numel() < aligned:
        if torch.cuda.is_current_stream_capturing():
            return None
        buf = torch.empty((aligned,), dtype=torch.uint8, device=device)
        _DECODE_SCRATCH[pool_key] = buf
    return buf[: numel * elem_size].view(dtype).view(shape)


def compute_hip_src_hash() -> str | None:
    """Compute the 12-char SHA-256 digest across all native HIP sources."""
    base = Path(__file__).parent
    h = hashlib.sha256()
    for name in _HIP_SOURCE_FILES:
        p = base / name
        if not p.is_file():
            return None
        h.update(name.encode("utf-8"))
        h.update(b"\0")
        h.update(p.read_bytes())
        h.update(b"\0")
    return h.hexdigest()[:12]


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
    def _has_devlib(root: str) -> bool:
        return (
            os.path.isdir(os.path.join(root, "amdgcn", "bitcode"))
            or os.path.isdir(os.path.join(root, "lib", "llvm", "amdgcn", "bitcode"))
            or os.path.isdir(os.path.join(root, "llvm", "amdgcn", "bitcode"))
        )

    explicit = os.getenv("INKLING_AMD_HIPCC")
    if explicit is not None:
        if os.path.isfile(explicit) and os.access(explicit, os.X_OK):
            root = str(Path(explicit).resolve().parent.parent)
            return explicit, (root if _has_devlib(root) else None)
        return None, None

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
    """Detect NixOS GCC/glibc include & library flags without injecting RPATH."""
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
        except Exception:
            pass
    return extra_flags, extra_env


def _validate_loaded_lib(
    lib: ctypes.CDLL,
    expected_hash: str | None,
    cand_name: str,
) -> bool:
    """Validate ABI version, compiled source hash, and core launch symbols."""
    for sym in _CORE_REQUIRED_SYMBOLS:
        if not hasattr(lib, sym):
            logger.warning("Rejecting HIP library %s: missing required symbol %s", cand_name, sym)
            return False
    lib.inkling_hip_abi_version.restype = ctypes.c_int
    abi = int(lib.inkling_hip_abi_version())
    if abi != EXPECTED_HIP_ABI_VERSION:
        logger.warning(
            "Rejecting HIP library %s: ABI version %d != expected %d",
            cand_name, abi, EXPECTED_HIP_ABI_VERSION,
        )
        return False
    lib.inkling_hip_src_hash.restype = ctypes.c_char_p
    raw_hash = lib.inkling_hip_src_hash()
    compiled_hash = raw_hash.decode("ascii", errors="replace") if raw_hash else ""
    if expected_hash is not None and compiled_hash != expected_hash:
        logger.warning(
            "Rejecting HIP library %s: compiled hash %s != expected %s",
            cand_name, compiled_hash, expected_hash,
        )
        return False
    return True


def get_hip_lib(required_symbols: tuple[str, ...] = ()) -> ctypes.CDLL | None:
    """Load, validate, and cache the portable Inkling2 gfx1151 native HIP library."""
    global _HIP_LIB, _HIP_INIT_DONE, _HIP_LOADED_PATH
    if not _HIP_INIT_DONE:
        _HIP_INIT_DONE = True
        if not any(hip_unit_enabled(u) for u in _ALL_HIP_UNITS):
            return None
        if not torch.cuda.is_available() or not torch.version.hip:
            return None
        arch = getattr(torch.cuda.get_device_properties(0), "gcnArchName", "").split(":")[0]
        if arch != "gfx1151":
            return None

        src_hash = compute_hip_src_hash()
        so_name = f"hip_inkling_{arch}_{src_hash}.so" if src_hash else f"hip_inkling_{arch}.so"
        _preload_amdhip()

        prebuilt_candidates: list[Path] = []
        explicit_so = os.getenv("INKLING_AMD_HIP_SO") or os.getenv("INKLING_AMD_HIP_PREFILL_SO")
        if explicit_so:
            prebuilt_candidates.append(Path(explicit_so))
        if src_hash:
            prebuilt_candidates.append(Path(__file__).with_name(so_name))

        for cand in prebuilt_candidates:
            if cand.is_file():
                try:
                    lib = ctypes.CDLL(str(cand))
                    if _validate_loaded_lib(lib, src_hash, cand.name):
                        _HIP_LIB = lib
                        _HIP_LOADED_PATH = str(cand)
                        logger.info("Loaded Inkling2 prebuilt native HIP kernels: %s (ABI v%d)", cand.name, EXPECTED_HIP_ABI_VERSION)
                        break
                except Exception as exc:
                    logger.warning("Failed to load prebuilt HIP library (%s): %s", cand, exc)

        if _HIP_LIB is None and src_hash is not None:
            cache_dir = Path(os.getenv("XDG_CACHE_HOME", os.path.expanduser("~/.cache"))) / "vllm" / "inkling2_hip"
            try:
                cache_dir.mkdir(parents=True, exist_ok=True)
            except OSError as exc:
                logger.warning("Inkling2 HIP cache dir %s unavailable (%s); using fallback.", cache_dir, exc)
                cache_dir = None
            so_path = (cache_dir / so_name) if cache_dir is not None else None

            if so_path is not None and so_path.is_file():
                try:
                    lib = ctypes.CDLL(str(so_path))
                    if _validate_loaded_lib(lib, src_hash, so_path.name):
                        _HIP_LIB = lib
                        _HIP_LOADED_PATH = str(so_path)
                        logger.info("Loaded cached Inkling2 native HIP kernels: %s (ABI v%d)", so_path.name, EXPECTED_HIP_ABI_VERSION)
                except Exception:
                    pass

            if _HIP_LIB is None and so_path is not None:
                hipcc, rocm_root = _find_hipcc()
                if not hipcc:
                    logger.info("Inkling2 HIP runtime: hipcc with device libs not found; using Triton/Gluon fallback.")
                else:
                    extra_flags, extra_env = _detect_nix_cxx_flags()
                    env = os.environ.copy()
                    env.update(extra_env)
                    if rocm_root:
                        env["ROCM_PATH"] = rocm_root
                        env["ROCM_HOME"] = rocm_root
                        env["HIP_PATH"] = rocm_root
                        extra_flags.append(f"-L{rocm_root}/lib")
                    base = Path(__file__).parent
                    tu_paths = [str(base / name) for name in _HIP_TU_FILES]
                    tmp_fd, tmp_so = tempfile.mkstemp(prefix=".hip_inkling_", suffix=".so", dir=str(cache_dir))
                    os.close(tmp_fd)
                    cmd = [
                        hipcc, "-shared", "-fPIC", "-O3", f"--offload-arch={arch}",
                        "-Wl,--as-needed",
                        f"-DINKLING_HIP_ABI_VERSION={EXPECTED_HIP_ABI_VERSION}",
                        f"-DINKLING_HIP_SRC_HASH_RAW={src_hash}",
                        f"-I{base}",
                        *tu_paths, "-o", tmp_so, *extra_flags,
                    ]
                    try:
                        res = subprocess.run(cmd, env=env, capture_output=True, text=True, timeout=180)
                        if res.returncode != 0:
                            logger.warning("Inkling2 HIP compile failed (using fallback): %s", res.stderr[:1200])
                            try:
                                os.unlink(tmp_so)
                            except OSError:
                                pass
                        else:
                            os.replace(tmp_so, so_path)
                            lib = ctypes.CDLL(str(so_path))
                            if _validate_loaded_lib(lib, src_hash, so_path.name):
                                _HIP_LIB = lib
                                _HIP_LOADED_PATH = str(so_path)
                                logger.info("Compiled and loaded Inkling2 native HIP kernels: %s (ABI v%d)", so_path.name, EXPECTED_HIP_ABI_VERSION)
                    except Exception as exc:
                        logger.warning("Inkling2 HIP compile error (using fallback): %s", exc)
                        try:
                            os.unlink(tmp_so)
                        except OSError:
                            pass

    if _HIP_LIB is None:
        return None
    for sym in required_symbols:
        if not hasattr(_HIP_LIB, sym):
            return None
    return _HIP_LIB


def _get_hip_lib() -> ctypes.CDLL | None:
    return get_hip_lib()


def _get_scratch(device: torch.device, max_m64: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    cur = _SCRATCH.get(device)
    need_pairs = max_m64 // 2 + 2
    need_tails = max_m64 + 2
    if cur is None or cur[0].numel() < need_pairs or cur[1].numel() < need_tails:
        pair_idx = torch.empty((max(need_pairs, 1024),), dtype=torch.int32, device=device)
        tail_idx = torch.empty((max(need_tails, 2048),), dtype=torch.int32, device=device)
        counts = torch.empty((4,), dtype=torch.int32, device=device)
        cur = (pair_idx, tail_idx, counts)
        _SCRATCH[device] = cur
        _LAST_PLAN.pop(device, None)
    return cur


def hip_moe_prefill_available() -> bool:
    return hip_unit_enabled("moe") and get_hip_lib() is not None


def hip_moe_prefill_projection(
    x: torch.Tensor,
    bank,
    plan,
    *,
    bank_kind: str,
    fused_activation=None,
    out_fp16: bool = False,
) -> torch.Tensor | None:
    """Run paired-M128 + M64-tail native HIP WMMA MoE prefill projection."""
    if not hip_unit_enabled("moe"):
        return None
    lib = get_hip_lib()
    if (
        lib is None
        or (plan.block_m, plan.block_n, plan.block_k) != (64, 128, 64)
        or plan.top_k != 8
        or getattr(plan, "num_experts", None) != 128
    ):
        return None
    if bank_kind not in ("source", "mixed", "coda"):
        return None

    experts, n_tiles, k_tiles, _, _ = bank.packed.shape
    n, k = n_tiles * 128, k_tiles * 64
    if n <= 0 or k <= 0:
        return None
    route_shape = (plan.num_tokens, plan.top_k)
    fused = fused_activation is not None
    if fused:
        if x.ndim != 2 or x.shape != (plan.num_tokens, k):
            return None
    else:
        if x.ndim != 3 or x.shape != (*route_shape, k):
            return None
    x_bf16 = fused and x.dtype == torch.bfloat16 and x.is_contiguous()
    x_in = x if x_bf16 else (x if x.dtype == torch.float16 and x.is_contiguous() else x.to(torch.float16).contiguous())
    out_dtype = torch.float16 if (fused and out_fp16) else torch.bfloat16
    out = x.new_empty((*route_shape, n // 2 if fused else n), dtype=out_dtype)

    max_m64 = plan.expert_ids.numel()
    pair_idx, tail_idx, counts = _get_scratch(x.device, max_m64)
    stream = torch.cuda.current_stream(x.device).cuda_stream

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

    if bank_kind == "source":
        prefix = "source"
        raw_ptr = bank.raw_scales.data_ptr()
        outer_ptr = bank.outer_scales.data_ptr()
        rs_ptr = tab_ptr = kind_ptr = 0
    elif bank_kind == "coda":
        prefix = "coda"
        raw_ptr = bank.tags.data_ptr()
        outer_ptr = 0
        rs_ptr = bank.row_scales.data_ptr()
        tab_ptr = bank.table.data_ptr()
        kind_ptr = 0
    else:
        prefix = f"mixed{bank.fitted_kind}"
        raw_ptr = bank.raw_scales.data_ptr()
        outer_ptr = bank.outer_scales.data_ptr()
        rs_ptr = bank.row_scales.data_ptr() if bank.row_scales is not None else 0
        tab_ptr = bank.table.data_ptr() if bank.table is not None else 0
        kind_ptr = bank.kind.data_ptr()

    if fused:
        out_tag = "f16" if out_fp16 else "bf16"
        x_tag = "_xbf16" if x_bf16 else ""
        suffix = f"w13_{out_tag}{x_tag}"
        gamma, gate_cap, up_cap = fused_activation
        gamma = gamma.float().contiguous()
    else:
        suffix = "w2"
        gamma, gate_cap, up_cap = None, 0.0, 0.0

    fn = getattr(lib, f"launch_{prefix}_{suffix}")
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
        ctypes.c_float(float(gate_cap)),
        ctypes.c_float(float(up_cap)),
        ctypes.c_void_p(stream),
    )
    return out


def can_run_hip_kv_conv_1w(
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
    *,
    num_tokens: int | None = None,
    num_heads: int | None = None,
    head_dim: int = 128,
    width: int = 4,
) -> bool:
    """Validate all invariants required by launch_hip_kv_conv_1w (min tokens: 64)."""
    t_len = k.shape[0] if num_tokens is None else num_tokens
    if not hip_unit_enabled("attn") or get_hip_lib(("launch_hip_kv_conv_1w",)) is None:
        return False
    if t_len < 64 or head_dim != 128 or width != 4:
        return False
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
        or value_cache.shape != key_cache.shape
        or value_cache.stride() != key_cache.stride()
        or v_weight.shape != k_weight.shape
        or v_weight.stride() != k_weight.stride()
        or (k_dynamic is None) != (v_dynamic is None)
    ):
        return False
    if k_dynamic is not None:
        dk_contig = (
            (k_dynamic.ndim == 2 and k_dynamic.stride(1) == 1)
            or (k_dynamic.ndim == 3 and k_dynamic.stride(2) == 1 and k_dynamic.stride(1) == width)
        )
        dv_contig = (
            v_dynamic is not None
            and (
                (v_dynamic.ndim == 2 and v_dynamic.stride(1) == 1)
                or (v_dynamic.ndim == 3 and v_dynamic.stride(2) == 1 and v_dynamic.stride(1) == width)
            )
        )
        if (
            v_dynamic is None
            or k_dynamic.dtype != torch.bfloat16
            or v_dynamic.dtype != torch.bfloat16
            or not dk_contig
            or not dv_contig
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
    return True


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
    if not can_run_hip_kv_conv_1w(
        k, v, k_weight, v_weight, k_dynamic, v_dynamic,
        cache, key_cache, value_cache, kv_slot_mapping,
        num_tokens=num_tokens, head_dim=head_dim, width=width,
    ):
        return False
    if positions.dtype not in (torch.int32, torch.int64) or not positions.is_contiguous():
        return False
    lib = get_hip_lib(("launch_hip_kv_conv_1w",))
    if lib is None:
        return False
    dynamic = k_dynamic is not None
    if dynamic:
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
        ctypes.c_int(1 if positions.dtype == torch.int64 else 0),
        ctypes.c_void_p(stream),
    )
    return True


hip_kv_conv_1w = hip_kv_conv_prefill


def hip_attn_res_q4_coeffs(
    bank0: torch.Tensor,
    bank1: torch.Tensor,
    queries: torch.Tensor,
    eps: float = 1e-6,
) -> torch.Tensor | None:
    """Compute [4, T, 2] FP32 linear combination coeffs for normalized 2-entry bank."""
    if not hip_unit_enabled("glue"):
        return None
    lib = get_hip_lib(("launch_q4_coeffs",))
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


def can_run_hip_moe_pair(experts, x: torch.Tensor, plan) -> bool:
    """Verify both w13 and w2 native HIP MoE banks exist before w13 emits FP16."""
    if not hip_unit_enabled("moe") or get_hip_lib() is None:
        return False
    if (
        (plan.block_m, plan.block_n, plan.block_k) != (64, 128, 64)
        or plan.top_k != 8
        or getattr(plan, "num_experts", None) != 128
    ):
        return False
    if x.ndim != 2 or not x.is_cuda:
        return False
    vb = getattr(experts, "verifier_wmma_banks", None) or {}
    sb = getattr(experts, "source_wmma_banks", None) or {}
    has_w13 = ("w13" in vb) or ("w13" in sb)
    has_w2 = ("w2" in vb) or ("w2" in sb)
    return has_w13 and has_w2


def hip_moe_reduce(down: torch.Tensor) -> torch.Tensor | None:
    """Bit-exact deterministic slot-order (0..7) FP32 sum -> RTNE BF16 route reduction."""
    if down.ndim != 3 or down.shape[0] <= 0:
        return None
    if down.shape[0] >= 64:
        if not hip_unit_enabled("moe"):
            return None
    else:
        if not hip_unit_enabled("decode_moe"):
            return None
    if (
        down.shape[1] != 8
        or down.shape[2] <= 0
        or (down.shape[2] & 7) != 0
        or down.dtype != torch.bfloat16
        or not down.is_cuda
        or not down.is_contiguous()
        or (down.data_ptr() & 15) != 0
    ):
        return None
    lib = get_hip_lib(("launch_hip_moe_reduce",))
    if lib is None:
        return None
    m, _, n = down.shape
    out = down.new_empty((m, n))
    stream = torch.cuda.current_stream(down.device).cuda_stream
    lib.launch_hip_moe_reduce(
        ctypes.c_void_p(down.data_ptr()),
        ctypes.c_void_p(out.data_ptr()),
        ctypes.c_int(m),
        ctypes.c_int(n),
        ctypes.c_void_p(stream),
    )
    return out


# ============================================================================
# Unit B: Exact Q/R Clip & Zero-Materialization BF16-WMMA Flash-Rel-Attention
# ============================================================================

_LOG2E = 1.4426950408889634


def can_run_hip_prefill_attn(
    q: torch.Tensor,
    r: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    max_query_len: int,
    num_splits: int,
    kv_writer_ok: bool,
    *,
    rel_proj_t: torch.Tensor | None = None,
    q_scale: torch.Tensor | None = None,
    r_scale: torch.Tensor | None = None,
    gate: torch.Tensor | None = None,
    sinks: torch.Tensor | None = None,
    output: torch.Tensor | None = None,
    q_out: torch.Tensor | None = None,
) -> bool:
    """Unified B0 writer+producer+consumer predicate for zero-mat HIP prefill attention."""
    if not kv_writer_ok or num_splits != 0 or max_query_len < 16:
        return False
    if not hip_unit_enabled("attn"):
        return False
    lib = get_hip_lib((
        "launch_hip_qr_clip",
        "launch_hip_flash_rel_attn_prefill",
        "launch_hip_gather_paged_kv_hnd",
        "launch_hip_flash_rel_attn_near_merge",
    ))
    if lib is None:
        return False
    if q.ndim != 3 or r.ndim != 3:
        return False
    t_len, h, d = q.shape
    if t_len < 64 or h != 24 or d != 128:
        return False
    if r.shape != (t_len, h, 16):
        return False
    if (
        q.dtype != torch.bfloat16
        or r.dtype != torch.bfloat16
        or key_cache.dtype != torch.bfloat16
        or value_cache.dtype != torch.bfloat16
        or not q.is_cuda
    ):
        return False
    page, kvh, kd = key_cache.shape[1:4]
    if kvh != 8 or kd != 128 or page < 16 or (page % 16) != 0:
        return False
    if value_cache.shape != key_cache.shape:
        return False
    if (
        q.stride(2) != 1
        or r.stride(2) != 1
        or key_cache.stride(3) != 1
        or value_cache.stride(3) != 1
        or (key_cache.data_ptr() & 15) != 0
        or (value_cache.data_ptr() & 15) != 0
        or (key_cache.stride(0) & 7) != 0
        or (key_cache.stride(1) & 7) != 0
        or (key_cache.stride(2) & 7) != 0
        or (value_cache.stride(0) & 7) != 0
        or (value_cache.stride(1) & 7) != 0
        or (value_cache.stride(2) & 7) != 0
    ):
        return False
    if rel_proj_t is not None:
        if (
            rel_proj_t.ndim != 2
            or rel_proj_t.shape[1] != 16
            or rel_proj_t.dtype != torch.bfloat16
            or not rel_proj_t.is_contiguous()
            or (rel_proj_t.data_ptr() & 15) != 0
        ):
            return False
    if q_scale is not None:
        if q_scale.shape != (h, d) or q_scale.dtype != torch.bfloat16 or not q_scale.is_contiguous():
            return False
    if r_scale is not None:
        if r_scale.shape != (h, 16) or r_scale.dtype != torch.bfloat16 or not r_scale.is_contiguous():
            return False
    if gate is not None and (gate.dtype != torch.bfloat16 or gate.stride(-1) != 1):
        return False
    if sinks is not None and (sinks.dtype != torch.bfloat16 or sinks.stride(-1) != 1):
        return False
    if output is not None and (output.dtype != torch.bfloat16 or output.stride(-1) != 1):
        return False
    if q_out is not None:
        if (
            q_out.shape != (t_len, h, d)
            or q_out.dtype != torch.bfloat16
            or not q_out.is_contiguous()
            or (q_out.data_ptr() & 15) != 0
        ):
            return False
    return True


def hip_qr_clip(
    q: torch.Tensor,
    r: torch.Tensor,
    q_scale: torch.Tensor,
    r_scale: torch.Tensor,
    tau: torch.Tensor | None,
    *,
    q_threshold: float = 22.627417,
    r_threshold: float = 8.0,
    eps: float = 1e-6,
    r_fp16: bool = False,
    q_out: torch.Tensor | None = None,
    unit: str = "attn",
) -> tuple[torch.Tensor, torch.Tensor] | None:
    """Exact per-head per-channel learned-scale L2 clip for Q and R (clip before tau)."""
    if not hip_unit_enabled(unit):
        return None
    lib = get_hip_lib(("launch_hip_qr_clip",))
    if lib is None:
        return None
    if q.ndim != 3 or r.ndim != 3 or q.dtype != torch.bfloat16 or r.dtype != torch.bfloat16:
        return None
    t_len, h, d = q.shape
    if d != 128 or r.shape != (t_len, h, 16):
        return None
    if (
        q_scale.shape != (h, d)
        or r_scale.shape != (h, 16)
        or q_scale.dtype != torch.bfloat16
        or r_scale.dtype != torch.bfloat16
        or not q_scale.is_contiguous()
        or not r_scale.is_contiguous()
    ):
        return None
    if q.stride(2) != 1 or r.stride(2) != 1:
        return None
    if q_out is None:
        q_out = torch.empty((t_len, h, d), device=q.device, dtype=torch.bfloat16)
    elif (
        q_out.shape != (t_len, h, d)
        or q_out.dtype != torch.bfloat16
        or not q_out.is_contiguous()
        or (q_out.data_ptr() & 15) != 0
    ):
        return None
    r_out = torch.empty((t_len, h, 16), device=q.device, dtype=torch.float16 if r_fp16 else torch.bfloat16)
    tau_f32 = None
    if tau is not None:
        tau_f32 = tau if (tau.dtype == torch.float32 and tau.is_contiguous()) else tau.to(torch.float32).contiguous()
    stream = torch.cuda.current_stream(q.device).cuda_stream
    lib.launch_hip_qr_clip(
        ctypes.c_void_p(q.data_ptr()),
        ctypes.c_void_p(r.data_ptr()),
        ctypes.c_void_p(q_scale.data_ptr()),
        ctypes.c_void_p(r_scale.data_ptr()),
        ctypes.c_void_p(tau_f32.data_ptr() if tau_f32 is not None else 0),
        ctypes.c_void_p(q_out.data_ptr()),
        ctypes.c_void_p(r_out.data_ptr()),
        ctypes.c_int(t_len),
        ctypes.c_int(h),
        ctypes.c_int64(q.stride(0)),
        ctypes.c_int64(q.stride(1)),
        ctypes.c_int64(r.stride(0)),
        ctypes.c_int64(r.stride(1)),
        ctypes.c_float(float(q_threshold)),
        ctypes.c_float(float(r_threshold)),
        ctypes.c_float(float(eps)),
        ctypes.c_int(1 if tau_f32 is not None else 0),
        ctypes.c_int(1 if r_fp16 else 0),
        ctypes.c_void_p(stream),
    )
    return q_out, r_out


def hip_flash_rel_attn_prefill(
    q: torch.Tensor,
    r: torch.Tensor,
    rel_proj_t: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    gate: torch.Tensor,
    sinks: torch.Tensor,
    result: torch.Tensor,
    *,
    block_table: torch.Tensor,
    cache_seqlens: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    max_seqlen_q: int,
    num_actual_tokens: int,
    softmax_scale: float,
    window: int,
    r_fp16: bool = False,
    round_bias_bf16: bool = True,
) -> bool:
    """Zero-materialization BF16-WMMA Flash-Rel-Attention prefill (+ hybrid long-context merge)."""
    if not hip_unit_enabled("attn"):
        return False
    lib = get_hip_lib((
        "launch_hip_flash_rel_attn_prefill",
        "launch_hip_gather_paged_kv_hnd",
        "launch_hip_flash_rel_attn_near_merge",
    ))
    if lib is None:
        return False
    num_tokens, num_heads, head_dim = q.shape
    page, num_kv_heads, kd = key_cache.shape[1:4]
    extent, d_rel = rel_proj_t.shape
    if (
        head_dim != 128
        or kd != 128
        or d_rel != 16
        or (num_heads % num_kv_heads) != 0
        or page < 16
        or q.dtype != torch.bfloat16
        or key_cache.dtype != torch.bfloat16
        or value_cache.dtype != torch.bfloat16
        or not rel_proj_t.is_contiguous()
        or q.stride(2) != 1
        or r.stride(2) != 1
        or (q.data_ptr() & 15) != 0
        or (q.stride(0) & 7) != 0
        or (q.stride(1) & 7) != 0
        or (r.data_ptr() & 15) != 0
        or (r.stride(0) & 7) != 0
        or (r.stride(1) & 7) != 0
        or (key_cache.data_ptr() & 15) != 0
        or (value_cache.data_ptr() & 15) != 0
        or key_cache.stride(3) != 1
        or value_cache.stride(3) != 1
        or (key_cache.stride(0) & 7) != 0
        or (key_cache.stride(1) & 7) != 0
        or (key_cache.stride(2) & 7) != 0
        or (value_cache.stride(0) & 7) != 0
        or (value_cache.stride(1) & 7) != 0
        or (value_cache.stride(2) & 7) != 0
        or (rel_proj_t.data_ptr() & 15) != 0
        or gate.dtype != torch.bfloat16
        or sinks.dtype != torch.bfloat16
        or result.dtype != torch.bfloat16
        or gate.stride(-1) != 1
        or sinks.stride(-1) != 1
        or result.stride(-1) != 1
    ):
        return False

    expected_r_dtype = torch.float16 if r_fp16 else torch.bfloat16
    if r.dtype != expected_r_dtype or rel_proj_t.dtype != expected_r_dtype:
        return False

    nt = num_actual_tokens
    if nt < num_tokens:
        result[nt:].zero_()
    if nt == 0:
        return True

    num_reqs = cache_seqlens.shape[0]
    group = num_heads // num_kv_heads
    scale_log2 = float(softmax_scale) * _LOG2E
    stream = torch.cuda.current_stream(q.device).cuda_stream

    # Hybrid AOTriton-Far + HIP-Near-Merge for long-context global prefill (>=2K far keys)
    if (
        window < 0
        and max_seqlen_q >= 64
        and page in (16, 32, 64)
        and not torch.cuda.is_current_stream_capturing()
    ):
        seq_cpu = cache_seqlens.tolist()
        cu_cpu = cu_seqlens_q.tolist()
        align = max(32, page)
        hybrid_reqs = []
        short_reqs = []
        for req_i in range(num_reqs):
            ql = cu_cpu[req_i + 1] - cu_cpu[req_i]
            past = seq_cpu[req_i] - ql
            far_len = (max(past - extent, 0) // align) * align
            if ql >= 64 and far_len >= 2048:
                hybrid_reqs.append((req_i, cu_cpu[req_i], ql, far_len))
            elif ql > 0:
                short_reqs.append((req_i, cu_cpu[req_i], ql))
        if hybrid_reqs:
            for req_i, qs, ql in short_reqs:
                qe = qs + ql
                cu1 = cu_seqlens_q.new_tensor([0, ql])
                q_sl = q[qs:qe]
                r_sl = r[qs:qe]
                g_sl = gate[qs:qe]
                s_sl = sinks[qs:qe]
                res_sl = result[qs:qe]
                bt_sl = block_table[req_i : req_i + 1]
                sl_sl = cache_seqlens[req_i : req_i + 1]
                lib.launch_hip_flash_rel_attn_prefill(
                    ctypes.c_void_p(q_sl.data_ptr()),
                    ctypes.c_void_p(r_sl.data_ptr()),
                    ctypes.c_void_p(rel_proj_t.data_ptr()),
                    ctypes.c_void_p(key_cache.data_ptr()),
                    ctypes.c_void_p(value_cache.data_ptr()),
                    ctypes.c_void_p(g_sl.data_ptr()),
                    ctypes.c_void_p(s_sl.data_ptr()),
                    ctypes.c_void_p(res_sl.data_ptr()),
                    ctypes.c_void_p(bt_sl.data_ptr()),
                    ctypes.c_void_p(sl_sl.data_ptr()),
                    ctypes.c_void_p(cu1.data_ptr()),
                    ctypes.c_float(scale_log2),
                    ctypes.c_int64(q_sl.stride(0)), ctypes.c_int64(q_sl.stride(1)),
                    ctypes.c_int64(r_sl.stride(0)), ctypes.c_int64(r_sl.stride(1)),
                    ctypes.c_int64(key_cache.stride(0)), ctypes.c_int64(key_cache.stride(1)), ctypes.c_int64(key_cache.stride(2)),
                    ctypes.c_int64(value_cache.stride(0)), ctypes.c_int64(value_cache.stride(1)), ctypes.c_int64(value_cache.stride(2)),
                    ctypes.c_int64(g_sl.stride(0)), ctypes.c_int64(s_sl.stride(0)),
                    ctypes.c_int64(res_sl.stride(0)), ctypes.c_int64(bt_sl.stride(0)),
                    ctypes.c_int(num_heads), ctypes.c_int(group), ctypes.c_int(page),
                    ctypes.c_int(extent), ctypes.c_int(window),
                    ctypes.c_int(ql), ctypes.c_int(1),
                    ctypes.c_int(1 if r_fp16 else 0),
                    ctypes.c_int(1 if round_bias_bf16 else 0),
                    ctypes.c_void_p(stream),
                )
            for req_i, qs, ql, far_len in hybrid_reqs:
                n_pages = far_len // page
                k_far = q.new_empty((1, num_kv_heads, far_len, head_dim))
                v_far = q.new_empty((1, num_kv_heads, far_len, head_dim))
                bt_row = block_table[req_i]
                lib.launch_hip_gather_paged_kv_hnd(
                    ctypes.c_void_p(key_cache.data_ptr()),
                    ctypes.c_void_p(value_cache.data_ptr()),
                    ctypes.c_void_p(bt_row.data_ptr()),
                    ctypes.c_void_p(k_far.data_ptr()),
                    ctypes.c_void_p(v_far.data_ptr()),
                    ctypes.c_int(n_pages),
                    ctypes.c_int(num_kv_heads),
                    ctypes.c_int(page),
                    ctypes.c_int64(key_cache.stride(0)), ctypes.c_int64(key_cache.stride(1)), ctypes.c_int64(key_cache.stride(2)),
                    ctypes.c_int64(value_cache.stride(0)), ctypes.c_int64(value_cache.stride(1)), ctypes.c_int64(value_cache.stride(2)),
                    ctypes.c_int64(k_far.stride(1)), ctypes.c_int64(k_far.stride(2)),
                    ctypes.c_int64(v_far.stride(1)), ctypes.c_int64(v_far.stride(2)),
                    ctypes.c_void_p(stream),
                )
                q_hnd = q[qs : qs + ql].transpose(0, 1).unsqueeze(0).contiguous()
                out_far, lse_far = torch.ops.aten._scaled_dot_product_flash_attention(
                    q_hnd, k_far, v_far, 0.0, False, False, scale=softmax_scale
                )[:2]
                out_far_3d = out_far[0]
                lse_far_2d = lse_far[0]
                lib.launch_hip_flash_rel_attn_near_merge(
                    ctypes.c_void_p(q.data_ptr()),
                    ctypes.c_void_p(r.data_ptr()),
                    ctypes.c_void_p(rel_proj_t.data_ptr()),
                    ctypes.c_void_p(key_cache.data_ptr()),
                    ctypes.c_void_p(value_cache.data_ptr()),
                    ctypes.c_void_p(gate.data_ptr()),
                    ctypes.c_void_p(sinks.data_ptr()),
                    ctypes.c_void_p(out_far_3d.data_ptr()),
                    ctypes.c_void_p(lse_far_2d.data_ptr()),
                    ctypes.c_void_p(result.data_ptr()),
                    ctypes.c_void_p(block_table.data_ptr()),
                    ctypes.c_void_p(cache_seqlens.data_ptr()),
                    ctypes.c_void_p(cu_seqlens_q.data_ptr()),
                    ctypes.c_float(scale_log2),
                    ctypes.c_int(far_len),
                    ctypes.c_int(req_i),
                    ctypes.c_int64(q.stride(0)), ctypes.c_int64(q.stride(1)),
                    ctypes.c_int64(r.stride(0)), ctypes.c_int64(r.stride(1)),
                    ctypes.c_int64(key_cache.stride(0)), ctypes.c_int64(key_cache.stride(1)), ctypes.c_int64(key_cache.stride(2)),
                    ctypes.c_int64(value_cache.stride(0)), ctypes.c_int64(value_cache.stride(1)), ctypes.c_int64(value_cache.stride(2)),
                    ctypes.c_int64(gate.stride(0)), ctypes.c_int64(sinks.stride(0)),
                    ctypes.c_int64(out_far_3d.stride(0)), ctypes.c_int64(out_far_3d.stride(1)),
                    ctypes.c_int64(lse_far_2d.stride(0)), ctypes.c_int64(lse_far_2d.stride(1)),
                    ctypes.c_int64(result.stride(0)), ctypes.c_int64(block_table.stride(0)),
                    ctypes.c_int(num_heads), ctypes.c_int(group), ctypes.c_int(page),
                    ctypes.c_int(extent), ctypes.c_int(ql),
                    ctypes.c_int(1 if r_fp16 else 0),
                    ctypes.c_int(1 if round_bias_bf16 else 0),
                    ctypes.c_void_p(stream),
                )
            return True

    lib.launch_hip_flash_rel_attn_prefill(
        ctypes.c_void_p(q.data_ptr()),
        ctypes.c_void_p(r.data_ptr()),
        ctypes.c_void_p(rel_proj_t.data_ptr()),
        ctypes.c_void_p(key_cache.data_ptr()),
        ctypes.c_void_p(value_cache.data_ptr()),
        ctypes.c_void_p(gate.data_ptr()),
        ctypes.c_void_p(sinks.data_ptr()),
        ctypes.c_void_p(result.data_ptr()),
        ctypes.c_void_p(block_table.data_ptr()),
        ctypes.c_void_p(cache_seqlens.data_ptr()),
        ctypes.c_void_p(cu_seqlens_q.data_ptr()),
        ctypes.c_float(scale_log2),
        ctypes.c_int64(q.stride(0)), ctypes.c_int64(q.stride(1)),
        ctypes.c_int64(r.stride(0)), ctypes.c_int64(r.stride(1)),
        ctypes.c_int64(key_cache.stride(0)), ctypes.c_int64(key_cache.stride(1)), ctypes.c_int64(key_cache.stride(2)),
        ctypes.c_int64(value_cache.stride(0)), ctypes.c_int64(value_cache.stride(1)), ctypes.c_int64(value_cache.stride(2)),
        ctypes.c_int64(gate.stride(0)), ctypes.c_int64(sinks.stride(0)),
        ctypes.c_int64(result.stride(0)), ctypes.c_int64(block_table.stride(0)),
        ctypes.c_int(num_heads), ctypes.c_int(group), ctypes.c_int(page),
        ctypes.c_int(extent), ctypes.c_int(window),
        ctypes.c_int(max_seqlen_q), ctypes.c_int(num_reqs),
        ctypes.c_int(1 if r_fp16 else 0),
        ctypes.c_int(1 if round_bias_bf16 else 0),
        ctypes.c_void_p(stream),
    )
    return True


# ============================================================================
# Unit C: Native HIP Prefill Glue (sconv, loop_zeromat, attn_res, router)
# ============================================================================

def hip_sconv_prefill(
    x: torch.Tensor,
    weight: torch.Tensor,
    cache: torch.Tensor,
    positions: torch.Tensor,
    block_table: torch.Tensor,
    seq_idx: torch.Tensor,
    slot_mapping: torch.Tensor,
    query_start: torch.Tensor,
    stream_offset: int,
    stream_width: int,
    addend: torch.Tensor | None = None,
    residual: torch.Tensor | None = None,
    out: torch.Tensor | None = None,
) -> torch.Tensor | None:
    """Request- & tile-boundary-safe static W=4 short convolution prefill."""
    if not hip_unit_enabled("glue"):
        return None
    if not x.is_cuda or x.ndim != 2 or x.dtype != torch.bfloat16:
        return None
    num_tokens, channels = x.shape
    if (
        num_tokens < 64
        or weight.ndim != 2
        or weight.shape != (channels, 4)
        or channels != cache.shape[1] * stream_width
        or block_table.shape[1] <= 0
        or cache.shape[2] <= 0
    ):
        return None
    if (
        weight.dtype != torch.bfloat16
        or cache.dtype != torch.bfloat16
        or x.stride(1) != 1
        or cache.stride(3) != 1
    ):
        return None
    if residual is not None and addend is None:
        return None
    if addend is not None and (
        addend.dtype not in (torch.bfloat16, torch.float32)
        or addend.shape != (num_tokens, channels)
        or not addend.is_contiguous()
    ):
        return None
    if residual is not None and (
        residual.dtype not in (torch.bfloat16, torch.float32)
        or residual.shape != (num_tokens, channels)
        or not residual.is_contiguous()
    ):
        return None
    if out is not None and (
        out.dtype not in (torch.bfloat16, torch.float32)
        or out.shape != (num_tokens, channels)
        or not out.is_contiguous()
    ):
        return None
    if positions.dtype not in (torch.int32, torch.int64) or not positions.is_contiguous():
        return None
    lib = get_hip_lib(("launch_hip_sconv_prefill",))
    if lib is None:
        return None
    if out is None:
        out = torch.empty((num_tokens, channels), dtype=cache.dtype, device=x.device)
    scale = (4 + 1) ** -0.5
    stream = torch.cuda.current_stream(x.device).cuda_stream
    lib.launch_hip_sconv_prefill(
        ctypes.c_void_p(x.data_ptr()),
        ctypes.c_void_p(weight.data_ptr()),
        ctypes.c_void_p(cache.data_ptr()),
        ctypes.c_void_p(positions.data_ptr()),
        ctypes.c_void_p(block_table.data_ptr()),
        ctypes.c_void_p(seq_idx.data_ptr()),
        ctypes.c_void_p(slot_mapping.data_ptr()),
        ctypes.c_void_p(query_start.data_ptr()),
        ctypes.c_void_p(addend.data_ptr() if addend is not None else 0),
        ctypes.c_void_p(residual.data_ptr() if residual is not None else 0),
        ctypes.c_void_p(out.data_ptr()),
        ctypes.c_int(num_tokens),
        ctypes.c_int(channels),
        ctypes.c_int(cache.shape[1]),
        ctypes.c_int(stream_offset),
        ctypes.c_int(stream_width),
        ctypes.c_int64(x.stride(0)),
        ctypes.c_int64(weight.stride(0)),
        ctypes.c_int64(weight.stride(1)),
        ctypes.c_int64(cache.stride(0)),
        ctypes.c_int64(cache.stride(1)),
        ctypes.c_int64(cache.stride(2)),
        ctypes.c_int64(block_table.stride(0)),
        ctypes.c_int(block_table.shape[1]),
        ctypes.c_int(cache.shape[2]),
        ctypes.c_float(scale),
        ctypes.c_int(1 if addend is not None else 0),
        ctypes.c_int(1 if residual is not None else 0),
        ctypes.c_int(1 if (addend is not None and addend.dtype == torch.float32) else 0),
        ctypes.c_int(1 if (residual is not None and residual.dtype == torch.float32) else 0),
        ctypes.c_int(1 if out.dtype == torch.float32 else 0),
        ctypes.c_int(1 if positions.dtype == torch.int64 else 0),
        ctypes.c_void_p(stream),
    )
    return out


def hip_loop_zeromat_prefill(
    dest: torch.Tensor,
    bank0: torch.Tensor,
    bank1: torch.Tensor,
    coeffs: torch.Tensor,
    down_weight_t: torch.Tensor,
    up_weight: torch.Tensor,
    norm_weight: torch.Tensor,
    norm_eps: float,
) -> torch.Tensor | None:
    """Native HIP BF16-WMMA loop zero-mat controller + blend + RMSNorm prefill."""
    if not hip_unit_enabled("glue"):
        return None
    if (
        not dest.is_cuda
        or dest.ndim != 3
        or dest.dtype != torch.float32
        or dest.stride(-1) != 1
        or bank0.dtype != torch.bfloat16
        or bank1.dtype != torch.bfloat16
        or coeffs.dtype != torch.float32
        or not coeffs.is_contiguous()
        or norm_weight.dtype != torch.bfloat16
    ):
        return None
    streams, t_len, width = dest.shape
    if (
        t_len < 64
        or width != 2304
        or streams not in (1, 3)
        or bank0.shape != (t_len, width)
        or bank1.shape != (t_len, width)
        or bank0.stride(-1) != 1
        or bank1.stride(-1) != 1
        or coeffs.shape != (streams, t_len, 2)
        or down_weight_t.shape != (streams, 2 * width, 64)
        or up_weight.shape != (streams, 2 * width, 64)
        or down_weight_t.dtype != torch.bfloat16
        or up_weight.dtype != torch.bfloat16
        or not down_weight_t.is_contiguous()
        or not up_weight.is_contiguous()
        or (down_weight_t.data_ptr() & 15) != 0
        or (up_weight.data_ptr() & 15) != 0
        or norm_weight.shape != (streams, width)
        or norm_weight.stride(-1) != 1
        or (dest.data_ptr() & 15) != 0
        or (dest.stride(0) & 3) != 0
        or (dest.stride(1) & 3) != 0
        or (bank0.data_ptr() & 15) != 0
        or (bank1.data_ptr() & 15) != 0
        or (bank0.stride(0) & 7) != 0
        or (bank1.stride(0) & 7) != 0
        or (norm_weight.data_ptr() & 15) != 0
        or (norm_weight.stride(0) & 7) != 0
    ):
        return None
    lib = get_hip_lib(("launch_hip_loop_zeromat_prefill",))
    if lib is None:
        return None
    out = bank0.new_empty((streams, t_len, width), dtype=torch.bfloat16)
    stream = torch.cuda.current_stream(dest.device).cuda_stream
    lib.launch_hip_loop_zeromat_prefill(
        ctypes.c_void_p(dest.data_ptr()),
        ctypes.c_void_p(bank0.data_ptr()),
        ctypes.c_void_p(bank1.data_ptr()),
        ctypes.c_void_p(coeffs.data_ptr()),
        ctypes.c_void_p(down_weight_t.data_ptr()),
        ctypes.c_void_p(up_weight.data_ptr()),
        ctypes.c_void_p(norm_weight.data_ptr()),
        ctypes.c_void_p(out.data_ptr()),
        ctypes.c_int64(dest.stride(0)),
        ctypes.c_int64(dest.stride(1)),
        ctypes.c_int64(bank0.stride(0)),
        ctypes.c_int64(bank1.stride(0)),
        ctypes.c_int64(coeffs.stride(0)),
        ctypes.c_int64(norm_weight.stride(0)),
        ctypes.c_int(t_len),
        ctypes.c_int(streams),
        ctypes.c_float(float(norm_eps)),
        ctypes.c_void_p(stream),
    )
    return out


def hip_attn_res_prefill(
    stack: torch.Tensor,
    last: torch.Tensor,
    addend: torch.Tensor | None,
    queries: torch.Tensor,
    norm_weight: torch.Tensor | None,
    norm_eps: float,
    normalize_values: bool,
    eps: float = 1e-6,
) -> torch.Tensor | None:
    """Native HIP block attention residuals (prefill & decode, D=2304, Q in {1, 3, 4})."""
    if not last.is_cuda or last.ndim != 2:
        return None
    tokens, width = last.shape
    if tokens <= 0:
        return None
    if tokens >= 64:
        if not hip_unit_enabled("glue"):
            return None
    else:
        if not hip_unit_enabled("decode_glue"):
            return None
    n_stack = stack.shape[0]
    q_num = queries.shape[0]
    if (
        tokens <= 0
        or width != 2304
        or q_num not in (1, 3, 4)
        or n_stack < 1
        or n_stack > 17
        or stack.shape[1:] != last.shape
        or queries.shape != (q_num, width)
        or queries.dtype != torch.float32
        or last.dtype not in (torch.bfloat16, torch.float32)
        or stack.dtype != last.dtype
    ):
        return None
    if addend is not None and (addend.shape != last.shape or addend.dtype != last.dtype):
        return None
    if norm_weight is not None and (norm_weight.shape != queries.shape or norm_weight.dtype != torch.bfloat16):
        return None
    tail = last if addend is None else addend
    weight = queries if norm_weight is None else norm_weight
    if any(t.device != last.device or t.stride(-1) != 1 or (t.data_ptr() & 15) != 0
           for t in (stack, last, tail, queries, weight)):
        return None
    elem_align = 3 if last.dtype == torch.float32 else 7
    if (
        (stack.stride(0) & elem_align) != 0
        or (stack.stride(1) & elem_align) != 0
        or (last.stride(0) & elem_align) != 0
        or (tail.stride(0) & elem_align) != 0
        or (queries.stride(0) & 3) != 0
        or (norm_weight is not None and (norm_weight.stride(0) & 7) != 0)
    ):
        return None
    lib = get_hip_lib(("launch_hip_attn_res_prefill",))
    if lib is None:
        return None
    out_dtype = last.dtype if norm_weight is not None else torch.float32
    out = last.new_empty((q_num, tokens, width), dtype=out_dtype)
    stream = torch.cuda.current_stream(last.device).cuda_stream
    lib.launch_hip_attn_res_prefill(
        ctypes.c_void_p(stack.data_ptr()),
        ctypes.c_void_p(last.data_ptr()),
        ctypes.c_void_p(tail.data_ptr()),
        ctypes.c_void_p(queries.data_ptr()),
        ctypes.c_void_p(weight.data_ptr()),
        ctypes.c_void_p(out.data_ptr()),
        ctypes.c_int64(stack.stride(0)),
        ctypes.c_int64(stack.stride(1)),
        ctypes.c_int64(last.stride(0)),
        ctypes.c_int64(tail.stride(0)),
        ctypes.c_int64(queries.stride(0)),
        ctypes.c_int64(weight.stride(0)),
        ctypes.c_int(tokens),
        ctypes.c_int(q_num),
        ctypes.c_int(n_stack),
        ctypes.c_int(1 if addend is not None else 0),
        ctypes.c_int(1 if normalize_values else 0),
        ctypes.c_int(1 if norm_weight is not None else 0),
        ctypes.c_int(1 if last.dtype == torch.float32 else 0),
        ctypes.c_int(1 if out_dtype == torch.float32 else 0),
        ctypes.c_int(0),
        ctypes.c_float(float(eps)),
        ctypes.c_float(float(norm_eps)),
        ctypes.c_void_p(stream),
    )
    return out


def hip_router_prefill(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor] | None:
    """Two-output BF16-WMMA prefill router emitting FP32 (scores, biased)."""
    if not hip_unit_enabled("glue"):
        return None
    if (
        not x.is_cuda
        or x.ndim != 2
        or x.shape[0] < 64
        or x.shape[1] != 2304
        or weight.shape != (128, 2304)
        or bias.shape != (128,)
        or x.dtype != torch.bfloat16
        or weight.dtype != torch.bfloat16
        or bias.dtype != torch.float32
        or x.stride(1) != 1
        or (x.data_ptr() & 15) != 0
        or (x.stride(0) & 7) != 0
        or not weight.is_contiguous()
        or (weight.data_ptr() & 15) != 0
        or not bias.is_contiguous()
    ):
        return None
    lib = get_hip_lib(("launch_hip_router_prefill",))
    if lib is None:
        return None
    t_len = x.shape[0]
    scores = x.new_empty((t_len, 128), dtype=torch.float32)
    biased = x.new_empty((t_len, 128), dtype=torch.float32)
    stream = torch.cuda.current_stream(x.device).cuda_stream
    lib.launch_hip_router_prefill(
        ctypes.c_void_p(x.data_ptr()),
        ctypes.c_void_p(weight.data_ptr()),
        ctypes.c_void_p(bias.data_ptr()),
        ctypes.c_void_p(scores.data_ptr()),
        ctypes.c_void_p(biased.data_ptr()),
        ctypes.c_int(t_len),
        ctypes.c_int64(x.stride(0)),
        ctypes.c_void_p(stream),
    )
    return scores, biased


# ============================================================================
# SP2 UNIT D1: ZERO-MAT 1M-READY NATIVE HIP DECODE ATTENTION
# ============================================================================

def _hip_decode_num_splits(is_local: bool, num_reqs: int) -> int:
    if is_local:
        if num_reqs == 1:
            return 10  # 10 x 8 = 80 CTAs (exact 2.0x 40-CU waves!)
        if num_reqs == 2:
            return 3   # 3 x 16 = 48 CTAs
        if num_reqs <= 4:
            return 2
        return 1
    if num_reqs == 1:
        return 15      # 15 x 8 = 120 CTAs (exact 3.0x 40-CU waves!)
    if num_reqs == 2:
        return 10      # 10 x 16 = 160 CTAs (exact 4.0x 40-CU waves!)
    return 5           # 5 x 8 x B (exact multiple of 40 CUs!)


def can_run_hip_kv_conv_decode(
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
    *,
    num_tokens: int | None = None,
    head_dim: int = 128,
    width: int = 4,
) -> bool:
    """Validate all invariants required by launch_hip_kv_conv_decode (1 <= T <= 63)."""
    t_len = k.shape[0] if num_tokens is None else num_tokens
    if not hip_unit_enabled("decode_attn") or get_hip_lib(("launch_hip_kv_conv_decode",)) is None:
        return False
    if t_len < 1 or t_len > 64 or head_dim != 128 or width != 4:
        return False
    if (
        not k.is_cuda
        or v.device != k.device
        or k_weight.device != k.device
        or v_weight.device != k.device
        or cache.device != k.device
        or key_cache.device != k.device
        or value_cache.device != k.device
        or kv_slot_mapping.device != k.device
        or kv_slot_mapping.dtype != torch.int64
        or not kv_slot_mapping.is_contiguous()
        or k.dtype != torch.bfloat16
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
        or value_cache.shape != key_cache.shape
        or value_cache.stride() != key_cache.stride()
        or v_weight.shape != k_weight.shape
        or v_weight.stride() != k_weight.stride()
        or (k_dynamic is None) != (v_dynamic is None)
    ):
        return False
    if k_dynamic is not None:
        if v_dynamic is None or k_dynamic.device != k.device or v_dynamic.device != k.device:
            return False
        dk_contig = (
            (k_dynamic.ndim == 2 and k_dynamic.stride(1) == 1)
            or (k_dynamic.ndim == 3 and k_dynamic.stride(2) == 1 and k_dynamic.stride(1) == width)
        )
        dv_contig = (
            v_dynamic is not None
            and (
                (v_dynamic.ndim == 2 and v_dynamic.stride(1) == 1)
                or (v_dynamic.ndim == 3 and v_dynamic.stride(2) == 1 and v_dynamic.stride(1) == width)
            )
        )
        if (
            v_dynamic is None
            or k_dynamic.dtype != torch.bfloat16
            or v_dynamic.dtype != torch.bfloat16
            or not dk_contig
            or not dv_contig
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
    return True


def can_run_hip_decode_attn(
    q: torch.Tensor,
    r: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    *,
    max_query_len: int,
    num_splits: int,
    kv_writer_ok: bool,
    rel_proj_t: torch.Tensor | None = None,
    q_scale: torch.Tensor | None = None,
    r_scale: torch.Tensor | None = None,
    gate: torch.Tensor | None = None,
    sinks: torch.Tensor | None = None,
    output: torch.Tensor | None = None,
    q_out: torch.Tensor | None = None,
) -> bool:
    """Upfront atomicity guard for D1 decode attention (checked BEFORE conv_state mutation)."""
    if not hip_unit_enabled("decode_attn") or not kv_writer_ok:
        return False
    lib = get_hip_lib((
        "launch_hip_qr_clip",
        "launch_hip_kv_conv_decode",
        "launch_hip_flash_rel_decode_split",
        "launch_hip_decode_combine",
    ))
    if lib is None:
        return False
    if num_splits <= 0 or max_query_len < 1 or max_query_len > 8:
        return False
    if (
        not q.is_cuda
        or r.device != q.device
        or key_cache.device != q.device
        or value_cache.device != q.device
        or q.ndim != 3
        or r.ndim != 3
        or q.dtype != torch.bfloat16
        or r.dtype != torch.bfloat16
    ):
        return False
    t_len, h, d = q.shape
    if t_len < 1 or t_len > 64 or h != 24 or d != 128 or r.shape != (t_len, h, 16):
        return False
    if key_cache.ndim != 4 or value_cache.shape != key_cache.shape:
        return False
    page, kvh, kd = key_cache.shape[1], key_cache.shape[2], key_cache.shape[3]
    if kvh != 8 or kd != 128 or page not in (16, 64):
        return False
    if (
        key_cache.dtype != torch.bfloat16
        or value_cache.dtype != torch.bfloat16
        or key_cache.stride(3) != 1
        or value_cache.stride(3) != 1
        or (key_cache.data_ptr() & 15) != 0
        or (value_cache.data_ptr() & 15) != 0
        or (key_cache.stride(0) & 7) != 0
        or (key_cache.stride(1) & 7) != 0
        or (key_cache.stride(2) & 7) != 0
        or (value_cache.stride(0) & 7) != 0
        or (value_cache.stride(1) & 7) != 0
        or (value_cache.stride(2) & 7) != 0
        or q.stride(2) != 1
        or r.stride(2) != 1
    ):
        return False
    if rel_proj_t is None or rel_proj_t.ndim != 2 or rel_proj_t.shape[1] != 16:
        return False
    if rel_proj_t.device != q.device or rel_proj_t.dtype != torch.bfloat16 or not rel_proj_t.is_contiguous() or (rel_proj_t.data_ptr() & 15) != 0:
        return False
    if q_scale is not None and (q_scale.device != q.device or q_scale.shape != (h, d) or q_scale.dtype != torch.bfloat16 or not q_scale.is_contiguous()):
        return False
    if r_scale is not None and (r_scale.device != q.device or r_scale.shape != (h, 16) or r_scale.dtype != torch.bfloat16 or not r_scale.is_contiguous()):
        return False
    if q_out is not None and (q_out.device != q.device or q_out.shape != (t_len, h, d) or q_out.dtype != torch.bfloat16 or not q_out.is_contiguous() or (q_out.data_ptr() & 15) != 0):
        return False
    if gate is not None and (gate.device != q.device or gate.shape != (t_len, h * d) or gate.dtype != torch.bfloat16 or gate.stride(1) != 1 or (gate.data_ptr() & 3) != 0 or (gate.stride(0) & 1) != 0):
        return False
    if sinks is not None and (sinks.device != q.device or sinks.shape != (t_len, h) or sinks.dtype != torch.bfloat16 or sinks.stride(1) != 1):
        return False
    if output is not None and (output.device != q.device or output.shape != (t_len, h * d) or output.dtype != torch.bfloat16 or output.stride(1) != 1 or (output.data_ptr() & 3) != 0 or (output.stride(0) & 1) != 0):
        return False
    return True


def hip_kv_conv_decode(
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
    cu_seqlens_q: torch.Tensor,
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
    """Run native 1-warp-per-(kv_head, req) decode K+V short-conv + paged KV cache write."""
    if not can_run_hip_kv_conv_decode(
        k, v, k_weight, v_weight, k_dynamic, v_dynamic,
        cache, key_cache, value_cache, kv_slot_mapping,
        num_tokens=num_tokens, head_dim=head_dim, width=width,
    ):
        return False
    if (
        positions.dtype not in (torch.int32, torch.int64)
        or not positions.is_contiguous()
        or positions.device != k.device
        or seq_idx.dtype != torch.int32
        or not seq_idx.is_contiguous()
        or seq_idx.device != k.device
        or slot_mapping.dtype != torch.int64
        or not slot_mapping.is_contiguous()
        or slot_mapping.device != k.device
        or block_table.dtype != torch.int32
        or block_table.stride(-1) != 1
        or block_table.device != k.device
        or query_start.dtype != torch.int32
        or not query_start.is_contiguous()
        or query_start.device != k.device
        or cu_seqlens_q.dtype != torch.int32
        or not cu_seqlens_q.is_contiguous()
        or cu_seqlens_q.device != k.device
    ):
        return False
    lib = get_hip_lib(("launch_hip_kv_conv_decode",))
    if lib is None:
        return False
    dynamic = k_dynamic is not None
    dk_ptr = k_dynamic.data_ptr() if dynamic else k_weight.data_ptr()
    dv_ptr = v_dynamic.data_ptr() if dynamic else v_weight.data_ptr()
    sdk = k_dynamic.stride(0) if dynamic else 0
    sdv = v_dynamic.stride(0) if dynamic else 0
    num_reqs = cu_seqlens_q.shape[0] - 1
    stream = torch.cuda.current_stream(k.device).cuda_stream
    rc = lib.launch_hip_kv_conv_decode(
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
        ctypes.c_void_p(cu_seqlens_q.data_ptr()),
        ctypes.c_int(num_reqs),
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
        ctypes.c_int(1 if positions.dtype == torch.int64 else 0),
        ctypes.c_void_p(stream),
    )
    return rc == 0


def hip_flash_rel_decode_split(
    q: torch.Tensor,
    r: torch.Tensor | None,
    rel_proj_t: torch.Tensor | None,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    *,
    block_table: torch.Tensor,
    cache_seqlens: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    max_query_len: int,
    softmax_scale: float,
    is_local: bool,
    has_rel: bool = True,
    window: int = -1,
    extent: int = 1024,
    num_splits: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor, int] | None:
    """Launch 1M-ready native HIP Flash-Rel-Decode split kernel (Target or Draft)."""
    if not hip_unit_enabled("decode_attn"):
        return None
    lib = get_hip_lib(("launch_hip_flash_rel_decode_split",))
    if lib is None:
        return None
    if (
        not q.is_cuda
        or key_cache.device != q.device
        or value_cache.device != q.device
        or block_table.device != q.device
        or cache_seqlens.device != q.device
        or cu_seqlens_q.device != q.device
        or block_table.dtype != torch.int32
        or block_table.stride(-1) != 1
        or cache_seqlens.dtype != torch.int32
        or not cache_seqlens.is_contiguous()
        or cu_seqlens_q.dtype != torch.int32
        or not cu_seqlens_q.is_contiguous()
        or q.ndim != 3
        or q.dtype != torch.bfloat16
        or q.stride(2) != 1
        or (q.data_ptr() & 15) != 0
        or (q.stride(0) & 7) != 0
        or (q.stride(1) & 7) != 0
    ):
        return None
    num_tokens, num_heads, head_dim = q.shape
    if num_tokens < 1 or num_tokens > 64 or num_heads != 24 or head_dim != 128:
        return None
    if max_query_len < 1 or max_query_len > 8:
        return None
    page, num_kv_heads = key_cache.shape[1], key_cache.shape[2]
    if num_kv_heads != 8 or key_cache.shape[3] != 128 or page not in (16, 64):
        return None
    if (
        key_cache.dtype != torch.bfloat16
        or value_cache.dtype != torch.bfloat16
        or key_cache.stride(3) != 1
        or value_cache.stride(3) != 1
        or (key_cache.data_ptr() & 15) != 0
        or (value_cache.data_ptr() & 15) != 0
        or (key_cache.stride(0) & 7) != 0
        or (key_cache.stride(1) & 7) != 0
        or (key_cache.stride(2) & 7) != 0
        or (value_cache.stride(0) & 7) != 0
        or (value_cache.stride(1) & 7) != 0
        or (value_cache.stride(2) & 7) != 0
    ):
        return None
    if has_rel:
        if (
            r is None
            or rel_proj_t is None
            or r.device != q.device
            or rel_proj_t.device != q.device
            or r.shape != (num_tokens, num_heads, 16)
            or r.dtype != torch.bfloat16
            or r.stride(2) != 1
            or (r.data_ptr() & 15) != 0
            or (r.stride(0) & 7) != 0
            or (r.stride(1) & 7) != 0
            or rel_proj_t.ndim != 2
            or rel_proj_t.shape[1] != 16
            or rel_proj_t.dtype != torch.bfloat16
            or not rel_proj_t.is_contiguous()
            or (rel_proj_t.data_ptr() & 15) != 0
        ):
            return None
        r_ptr, srt, srh = r.data_ptr(), r.stride(0), r.stride(1)
        rel_ptr = rel_proj_t.data_ptr()
    else:
        r_ptr, srt, srh = q.data_ptr(), 0, 0
        rel_ptr = q.data_ptr()

    num_reqs = cache_seqlens.shape[0]
    splits = _hip_decode_num_splits(is_local, num_reqs) if num_splits is None else int(num_splits)
    if splits <= 0:
        return None

    o_part = q.new_empty((splits * num_tokens, num_heads, head_dim), dtype=torch.float32)
    lse_part = q.new_empty((num_heads, splits * num_tokens), dtype=torch.float32)
    scale_log2 = float(softmax_scale) * 1.4426950408889634
    stream = torch.cuda.current_stream(q.device).cuda_stream
    rc = lib.launch_hip_flash_rel_decode_split(
        ctypes.c_void_p(q.data_ptr()),
        ctypes.c_void_p(r_ptr),
        ctypes.c_void_p(rel_ptr),
        ctypes.c_void_p(key_cache.data_ptr()),
        ctypes.c_void_p(value_cache.data_ptr()),
        ctypes.c_void_p(o_part.data_ptr()),
        ctypes.c_void_p(lse_part.data_ptr()),
        ctypes.c_void_p(block_table.data_ptr()),
        ctypes.c_void_p(cache_seqlens.data_ptr()),
        ctypes.c_void_p(cu_seqlens_q.data_ptr()),
        ctypes.c_int(num_reqs),
        ctypes.c_int(num_tokens),
        ctypes.c_int(splits),
        ctypes.c_int(max_query_len),
        ctypes.c_int(num_heads),
        ctypes.c_int(num_kv_heads),
        ctypes.c_int(page),
        ctypes.c_int(1 if is_local else 0),
        ctypes.c_int(1 if has_rel else 0),
        ctypes.c_int(window),
        ctypes.c_int(extent),
        ctypes.c_float(scale_log2),
        ctypes.c_int64(q.stride(0)),
        ctypes.c_int64(q.stride(1)),
        ctypes.c_int64(srt),
        ctypes.c_int64(srh),
        ctypes.c_int64(key_cache.stride(0)),
        ctypes.c_int64(key_cache.stride(1)),
        ctypes.c_int64(key_cache.stride(2)),
        ctypes.c_int64(value_cache.stride(0)),
        ctypes.c_int64(value_cache.stride(1)),
        ctypes.c_int64(value_cache.stride(2)),
        ctypes.c_int64(o_part.stride(0)),
        ctypes.c_int64(o_part.stride(1)),
        ctypes.c_int64(lse_part.stride(0)),
        ctypes.c_int64(block_table.stride(0)),
        ctypes.c_void_p(stream),
    )
    if rc != 0:
        return None
    return o_part, lse_part, splits


def hip_decode_combine(
    o_part: torch.Tensor,
    lse_part: torch.Tensor,
    gate: torch.Tensor | None,
    sinks: torch.Tensor | None,
    out: torch.Tensor,
    *,
    cu_seqlens_q: torch.Tensor,
    num_splits: int,
    has_gate: bool = True,
) -> bool:
    """Reduce split-KV partials + optional fused gate/sink epilogue into BF16 out."""
    if not hip_unit_enabled("decode_attn"):
        return False
    lib = get_hip_lib(("launch_hip_decode_combine",))
    if lib is None:
        return False
    num_heads, head_dim = o_part.shape[1], o_part.shape[2]
    num_tokens = o_part.shape[0] // num_splits
    if head_dim != 128 or num_tokens <= 0:
        return False
    if (
        not o_part.is_cuda
        or lse_part.device != o_part.device
        or out.device != o_part.device
        or cu_seqlens_q.device != o_part.device
        or cu_seqlens_q.dtype != torch.int32
        or not cu_seqlens_q.is_contiguous()
        or out.dtype != torch.bfloat16
        or (out.data_ptr() & 3) != 0
    ):
        return False
    out_3d = out.view(num_tokens, num_heads, head_dim)
    if out_3d.stride(2) != 1 or (out_3d.stride(0) & 1) != 0 or (out_3d.stride(1) & 1) != 0:
        return False
    if has_gate:
        if (
            gate is None
            or sinks is None
            or gate.device != o_part.device
            or sinks.device != o_part.device
            or gate.dtype != torch.bfloat16
            or sinks.dtype != torch.bfloat16
        ):
            return False
        g_3d = gate.view(num_tokens, num_heads, head_dim)
        if g_3d.stride(2) != 1 or (g_3d.data_ptr() & 3) != 0 or (g_3d.stride(0) & 1) != 0 or (g_3d.stride(1) & 1) != 0:
            return False
        g_ptr, sgt, sgh = g_3d.data_ptr(), g_3d.stride(0), g_3d.stride(1)
        sk_ptr, ssk_t = sinks.data_ptr(), sinks.stride(0)
    else:
        g_ptr, sgt, sgh = out.data_ptr(), 0, 0
        sk_ptr, ssk_t = out.data_ptr(), 0
    num_reqs = cu_seqlens_q.shape[0] - 1
    stream = torch.cuda.current_stream(o_part.device).cuda_stream
    rc = lib.launch_hip_decode_combine(
        ctypes.c_void_p(o_part.data_ptr()),
        ctypes.c_void_p(lse_part.data_ptr()),
        ctypes.c_void_p(g_ptr),
        ctypes.c_void_p(sk_ptr),
        ctypes.c_void_p(out_3d.data_ptr()),
        ctypes.c_void_p(cu_seqlens_q.data_ptr()),
        ctypes.c_int(num_reqs),
        ctypes.c_int(num_tokens),
        ctypes.c_int(num_heads),
        ctypes.c_int(num_splits),
        ctypes.c_int(1 if has_gate else 0),
        ctypes.c_int64(o_part.stride(0)),
        ctypes.c_int64(o_part.stride(1)),
        ctypes.c_int64(lse_part.stride(0)),
        ctypes.c_int64(sgt),
        ctypes.c_int64(sgh),
        ctypes.c_int64(ssk_t),
        ctypes.c_int64(out_3d.stride(0)),
        ctypes.c_int64(out_3d.stride(1)),
        ctypes.c_void_p(stream),
    )
    return rc == 0


def hip_block_decode_attention(
    q: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    out: torch.Tensor,
    *,
    block_table: torch.Tensor,
    cache_seqlens: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    block: int,
    window: int,
    num_splits: int,
) -> bool:
    """Native HIP DSpark draft block decode attention (HAS_REL=False, HAS_GATE=False)."""
    if not hip_unit_enabled("decode_attn"):
        return False
    is_local = window > 0
    res = hip_flash_rel_decode_split(
        q, None, None, key_cache, value_cache,
        block_table=block_table, cache_seqlens=cache_seqlens, cu_seqlens_q=cu_seqlens_q,
        max_query_len=block, softmax_scale=q.shape[-1] ** -0.5,
        is_local=is_local, has_rel=False, window=(window - 1 if is_local else -1),
        extent=0, num_splits=num_splits,
    )
    if res is None:
        return False
    o_part, lse_part, splits = res
    return hip_decode_combine(
        o_part, lse_part, None, None, out,
        cu_seqlens_q=cu_seqlens_q, num_splits=splits, has_gate=False,
    )


# ============================================================================
# SP2 UNIT D2: SINGLE-PASS NATIVE HIP W6 & W4 QUANTIZED DECODE LINEAR
# ============================================================================

def hip_reduce_splits(
    partial: torch.Tensor,
    out: torch.Tensor,
    *,
    splits: int,
    soft_cap: float = 0.0,
) -> bool:
    """Reduce [SPLITS, ...] FP32 partial tensor to BF16 or FP32 out (+ optional softcap)."""
    if not (hip_unit_enabled("decode_quant") or hip_unit_enabled("decode_glue")):
        return False
    lib = get_hip_lib(("launch_hip_reduce_splits",))
    if (
        lib is None
        or splits <= 1
        or not partial.is_cuda
        or out.device != partial.device
        or partial.dtype != torch.float32
        or not partial.is_contiguous()
        or not out.is_contiguous()
    ):
        return False
    if out.dtype not in (torch.bfloat16, torch.float32):
        return False
    total_elems = out.numel()
    if partial.numel() < splits * total_elems:
        return False
    if (total_elems & 3) == 0:
        out_align = 15 if out.dtype == torch.float32 else 7
        if (partial.data_ptr() & 15) != 0 or (out.data_ptr() & out_align) != 0:
            return False
    stream = torch.cuda.current_stream(partial.device).cuda_stream
    rc = lib.launch_hip_reduce_splits(
        ctypes.c_void_p(partial.data_ptr()),
        ctypes.c_void_p(out.data_ptr()),
        ctypes.c_int64(total_elems),
        ctypes.c_int(splits),
        ctypes.c_int(1 if out.dtype == torch.float32 else 0),
        ctypes.c_float(float(soft_cap)),
        ctypes.c_void_p(stream),
    )
    return rc == 0


def hip_w6_wmma_linear(
    x: torch.Tensor,
    weight,
    *,
    splits: int = 1,
    out_dtype: torch.dtype = torch.bfloat16,
    soft_cap: float = 0.0,
) -> torch.Tensor | None:
    """Single-pass native HIP W6 WMMA linear (M=1..64, N_TILE=64, M_TILE=16/32/48/64)."""
    if not hip_unit_enabled("decode_quant"):
        return None
    lib = get_hip_lib(("launch_hip_w6_wmma", "launch_hip_reduce_splits"))
    if lib is None:
        return None
    if x.ndim != 2 or not x.is_cuda or x.dtype != torch.bfloat16:
        return None
    m, k = x.shape
    n_pad = weight.padded_out
    if m < 1 or m > 64 or k != weight.in_features or (k & 63) != 0 or (n_pad & 127) != 0:
        return None
    if out_dtype not in (torch.bfloat16, torch.float32):
        return None
    if splits not in (1, 2, 4) or ((k // 64) % splits) != 0:
        return None
    xc = x.contiguous()
    if (
        not (weight.p4.is_contiguous() and weight.p2.is_contiguous() and weight.scales.is_contiguous())
        or weight.p4.device != xc.device
        or weight.p2.device != xc.device
        or weight.scales.device != xc.device
        or (xc.data_ptr() & 15) != 0
        or (weight.p4.data_ptr() & 15) != 0
        or (weight.p2.data_ptr() & 7) != 0
    ):
        return None
    out = xc.new_empty((m, n_pad), dtype=out_dtype)
    partial = xc.new_empty((splits, m, n_pad), dtype=torch.float32) if splits > 1 else out
    stream = torch.cuda.current_stream(xc.device).cuda_stream
    rc = lib.launch_hip_w6_wmma(
        ctypes.c_void_p(xc.data_ptr()),
        ctypes.c_void_p(xc.data_ptr()),
        ctypes.c_void_p(xc.data_ptr()),
        ctypes.c_void_p(weight.p4.data_ptr()),
        ctypes.c_void_p(weight.p2.data_ptr()),
        ctypes.c_void_p(weight.scales.data_ptr()),
        ctypes.c_void_p(out.data_ptr()),
        ctypes.c_void_p(partial.data_ptr()),
        ctypes.c_int(m),
        ctypes.c_int(n_pad),
        ctypes.c_int(k),
        ctypes.c_int(0),
        ctypes.c_int(0),
        ctypes.c_int(0),
        ctypes.c_int(splits),
        ctypes.c_int(1 if out_dtype == torch.float32 else 0),
        ctypes.c_float(float(soft_cap)),
        ctypes.c_void_p(stream),
    )
    if rc != 0:
        return None
    return out[:, : weight.out_features]


def hip_w6_wmma_qkv_linear(
    x_q: torch.Tensor,
    x_k: torch.Tensor,
    x_v: torch.Tensor,
    weight,
    q_pad: int,
    k_end: int,
) -> torch.Tensor | None:
    """Single-pass native HIP W6 WMMA QKV linear (M=1..64, single weight read at M=40!)."""
    if not hip_unit_enabled("decode_quant"):
        return None
    lib = get_hip_lib(("launch_hip_w6_wmma",))
    if lib is None:
        return None
    if (
        x_q.ndim != 2 or x_k.shape != x_q.shape or x_v.shape != x_q.shape
        or not x_q.is_cuda or x_k.device != x_q.device or x_v.device != x_q.device
        or x_q.dtype != torch.bfloat16 or x_k.dtype != torch.bfloat16 or x_v.dtype != torch.bfloat16
    ):
        return None
    m, k = x_q.shape
    n_pad = weight.padded_out
    if m < 1 or m > 64 or k != weight.in_features or (k & 63) != 0 or (n_pad & 127) != 0:
        return None
    if (q_pad & 127) != 0 or (k_end & 127) != 0:
        return None
    xqc = x_q.contiguous()
    xkc = xqc if x_k is x_q else x_k.contiguous()
    xvc = xqc if x_v is x_q else x_v.contiguous()
    if (
        not (weight.p4.is_contiguous() and weight.p2.is_contiguous() and weight.scales.is_contiguous())
        or weight.p4.device != xqc.device
        or weight.p2.device != xqc.device
        or weight.scales.device != xqc.device
        or (xqc.data_ptr() & 15) != 0
        or (xkc.data_ptr() & 15) != 0
        or (xvc.data_ptr() & 15) != 0
        or (weight.p4.data_ptr() & 15) != 0
        or (weight.p2.data_ptr() & 7) != 0
    ):
        return None
    splits = 1
    out = xqc.new_empty((m, n_pad), dtype=torch.bfloat16)
    partial = xqc.new_empty((splits, m, n_pad), dtype=torch.float32) if splits > 1 else out
    stream = torch.cuda.current_stream(xqc.device).cuda_stream
    rc = lib.launch_hip_w6_wmma(
        ctypes.c_void_p(xqc.data_ptr()),
        ctypes.c_void_p(xkc.data_ptr()),
        ctypes.c_void_p(xvc.data_ptr()),
        ctypes.c_void_p(weight.p4.data_ptr()),
        ctypes.c_void_p(weight.p2.data_ptr()),
        ctypes.c_void_p(weight.scales.data_ptr()),
        ctypes.c_void_p(out.data_ptr()),
        ctypes.c_void_p(partial.data_ptr()),
        ctypes.c_int(m),
        ctypes.c_int(n_pad),
        ctypes.c_int(k),
        ctypes.c_int(q_pad),
        ctypes.c_int(k_end),
        ctypes.c_int(1),
        ctypes.c_int(splits),
        ctypes.c_int(0),
        ctypes.c_float(0.0),
        ctypes.c_void_p(stream),
    )
    if rc != 0:
        return None
    return out


def hip_w6_head_gemv_m1(
    x: torch.Tensor,
    pair: torch.Tensor,
    third: torch.Tensor,
    scales: torch.Tensor,
    *,
    out_dtype: torch.dtype = torch.float32,
    soft_cap: float = 0.0,
) -> torch.Tensor | None:
    """Native HIP groupwise W6 M=1 GEMV consuming exact pack_groupwise_w6_head tensors."""
    if not hip_unit_enabled("decode_quant"):
        return None
    lib = get_hip_lib(("launch_hip_w6_head_gemv_m1",))
    if lib is None:
        return None
    if x.ndim != 2 or x.shape[0] != 1 or not x.is_cuda or x.dtype != torch.bfloat16:
        return None
    k = x.shape[1]
    n = pair.shape[0]
    if n <= 0 or k not in (2304, 3072) or out_dtype not in (torch.bfloat16, torch.float32):
        return None
    if (
        pair.device != x.device
        or third.device != x.device
        or scales.device != x.device
        or pair.dtype != torch.int64
        or third.dtype != torch.int32
        or scales.dtype != torch.float16
        or pair.shape != (n, k // 16)
        or third.shape != pair.shape
        or not (pair.is_contiguous() and third.is_contiguous() and scales.is_contiguous())
        or (pair.data_ptr() & 7) != 0
        or (third.data_ptr() & 3) != 0
    ):
        return None
    if scales.ndim != 2 or scales.shape[0] != n or scales.shape[1] == 0 or (k % scales.shape[1]) != 0:
        return None
    group_size = k // scales.shape[1]
    if group_size not in (32, 64):
        return None
    xc = x.contiguous()
    if (xc.data_ptr() & 15) != 0:
        return None
    out = xc.new_empty((1, n), dtype=out_dtype)
    stream = torch.cuda.current_stream(xc.device).cuda_stream
    rc = lib.launch_hip_w6_head_gemv_m1(
        ctypes.c_void_p(xc.data_ptr()),
        ctypes.c_void_p(pair.data_ptr()),
        ctypes.c_void_p(third.data_ptr()),
        ctypes.c_void_p(scales.data_ptr()),
        ctypes.c_void_p(out.data_ptr()),
        ctypes.c_int(n),
        ctypes.c_int(k),
        ctypes.c_int(group_size),
        ctypes.c_int(1 if out_dtype == torch.float32 else 0),
        ctypes.c_float(float(soft_cap)),
        ctypes.c_void_p(stream),
    )
    if rc != 0:
        return None
    return out


def hip_w4_wmma_linear(
    x: torch.Tensor,
    weight,
    *,
    bias: torch.Tensor | None = None,
    splits: int = 1,
) -> torch.Tensor | None:
    """Single-pass native HIP W4 WMMA linear (+ optional BF16 bias, M=1..64)."""
    if not hip_unit_enabled("decode_quant"):
        return None
    lib = get_hip_lib(("launch_hip_w4_wmma", "launch_hip_reduce_splits"))
    if lib is None:
        return None
    if x.ndim != 2 or not x.is_cuda or x.dtype != torch.bfloat16:
        return None
    m, k = x.shape
    n_out = weight.out_features
    n_pad = weight.padded_out
    if m < 1 or m > 64 or k != weight.in_features or (k & 63) != 0 or (n_pad & 127) != 0:
        return None
    eff_splits = 1 if bias is not None else splits
    if eff_splits not in (1, 2, 4) or ((k // 64) % eff_splits) != 0:
        return None
    xc = x.contiguous()
    if (
        not (weight.packed.is_contiguous() and weight.scales.is_contiguous())
        or weight.packed.device != xc.device
        or weight.scales.device != xc.device
        or (xc.data_ptr() & 15) != 0
        or (weight.packed.data_ptr() & 15) != 0
    ):
        return None
    if bias is not None:
        if bias.device != xc.device or bias.dtype != torch.bfloat16 or not bias.is_contiguous() or bias.numel() < n_out:
            return None
        bias_ptr = bias.data_ptr()
    else:
        bias_ptr = xc.data_ptr()
    out = xc.new_empty((m, n_pad), dtype=torch.bfloat16)
    partial = xc.new_empty((eff_splits, m, n_pad), dtype=torch.float32) if eff_splits > 1 else out
    stream = torch.cuda.current_stream(xc.device).cuda_stream
    rc = lib.launch_hip_w4_wmma(
        ctypes.c_void_p(xc.data_ptr()),
        ctypes.c_void_p(weight.packed.data_ptr()),
        ctypes.c_void_p(weight.scales.data_ptr()),
        ctypes.c_void_p(bias_ptr),
        ctypes.c_void_p(out.data_ptr()),
        ctypes.c_void_p(partial.data_ptr()),
        ctypes.c_int(m),
        ctypes.c_int(n_out),
        ctypes.c_int(n_pad),
        ctypes.c_int(k),
        ctypes.c_int(eff_splits),
        ctypes.c_int(1 if bias is not None else 0),
        ctypes.c_int(0),
        ctypes.c_void_p(stream),
    )
    if rc != 0:
        return None
    return out[:, :n_out]


def hip_w4_markov_argmax(
    latent: torch.Tensor,
    packed: torch.Tensor,
    scales: torch.Tensor,
    bias: torch.Tensor,
    base: torch.Tensor,
    softcap: float,
    valid: int,
) -> torch.Tensor | None:
    """Native HIP W4 Markov fused argmax (exact BF16 RTNE + lower-index tie-break)."""
    if not hip_unit_enabled("decode_quant"):
        return None
    lib = get_hip_lib(("launch_hip_w4_markov_argmax",))
    if lib is None:
        return None
    if (
        valid <= 0 or softcap <= 0.0
        or not latent.is_cuda
        or packed.device != latent.device or scales.device != latent.device
        or bias.device != latent.device or base.device != latent.device
        or latent.dtype != torch.bfloat16 or latent.numel() != 256
        or packed.dtype != torch.uint8 or packed.ndim != 2 or packed.shape[0] < valid or packed.shape[1] != 128
        or scales.dtype != torch.float16 or scales.ndim != 2 or scales.shape[0] < valid or scales.shape[1] != 4
        or bias.dtype != torch.bfloat16 or bias.numel() < valid
        or base.dtype != torch.bfloat16 or base.numel() < valid
        or not (latent.is_contiguous() and packed.is_contiguous() and scales.is_contiguous()
                and bias.is_contiguous() and base.is_contiguous())
        or (packed.data_ptr() & 15) != 0 or (latent.data_ptr() & 3) != 0
    ):
        return None
    splits = (valid + 15) // 16
    part_vals = base.new_empty((splits,), dtype=torch.float32)
    part_ids = base.new_empty((splits,), dtype=torch.int32)
    out = base.new_empty((1,), dtype=torch.int64)
    stream = torch.cuda.current_stream(base.device).cuda_stream
    rc = lib.launch_hip_w4_markov_argmax(
        ctypes.c_void_p(latent.data_ptr()),
        ctypes.c_void_p(packed.data_ptr()),
        ctypes.c_void_p(scales.data_ptr()),
        ctypes.c_void_p(bias.data_ptr()),
        ctypes.c_void_p(base.data_ptr()),
        ctypes.c_void_p(part_vals.data_ptr()),
        ctypes.c_void_p(part_ids.data_ptr()),
        ctypes.c_void_p(out.data_ptr()),
        ctypes.c_int(valid),
        ctypes.c_float(float(softcap)),
        ctypes.c_void_p(stream),
    )
    if rc != 0:
        return None
    return out


# ============================================================================
# SP2 UNIT D3: SINGLE-PASS M16/M32 NATIVE HIP DECODE MoE WRAPPERS
# ============================================================================

_LAST_DECODE_PLAN: dict[torch.device, object] = {}
_CAPTURE_PLANNED: dict[torch.device, object] = {}


def can_run_hip_moe_decode_pair(experts_mod, x: torch.Tensor, plan) -> bool:
    """Return True iff both w13 and w2 decode projections will run via Unit D3 HIP."""
    if (
        not hip_unit_enabled("decode_moe")
        or not x.is_cuda
        or (plan.block_m, plan.block_n, plan.block_k) != (16, 128, 64)
        or plan.top_k != 8
        or plan.num_tokens <= 0
        or get_hip_lib(("launch_hip_moe_decode_plan", "launch_hip_moe_decode_unified")) is None
    ):
        return False
    has_v13 = "w13" in getattr(experts_mod, "verifier_wmma_banks", {})
    has_v2 = "w2" in getattr(experts_mod, "verifier_wmma_banks", {})
    has_s13 = "w13" in getattr(experts_mod, "source_wmma_banks", {})
    has_s2 = "w2" in getattr(experts_mod, "source_wmma_banks", {})
    if not ((has_v13 or has_s13) and (has_v2 or has_s2)):
        return False
    return True


def hip_moe_decode_projection(
    x: torch.Tensor,
    bank,
    plan,
    *,
    bank_kind: str,
    fused_activation: tuple[torch.Tensor, float, float] | None = None,
    out_fp16: bool = False,
) -> torch.Tensor | None:
    """Single-pass M16/M32 native HIP decode MoE projection across source/mixed1/mixed2/coda."""
    if (
        not hip_unit_enabled("decode_moe")
        or not x.is_cuda
        or (plan.block_m, plan.block_n, plan.block_k) != (16, 128, 64)
        or plan.top_k != 8
        or plan.num_tokens <= 0
    ):
        return None
    lib = get_hip_lib(("launch_hip_moe_decode_plan", "launch_hip_moe_decode_unified"))
    if lib is None:
        return None
    _, n_tiles, k_tiles, _, _ = bank.packed.shape
    n, k = n_tiles * 128, k_tiles * 64
    route_shape = (plan.num_tokens, plan.top_k)
    fused = fused_activation is not None
    if fused:
        if x.ndim != 2 or x.shape != (plan.num_tokens, k):
            return None
    else:
        if x.ndim != 3 or x.shape != (*route_shape, k):
            return None
    if x.dtype not in (torch.bfloat16, torch.float16):
        return None
    if (
        not bank.packed.is_contiguous()
        or bank.packed.device != x.device
        or (bank.packed.data_ptr() & 15) != 0
        or plan.sorted_ids.device != x.device
        or plan.expert_ids.device != x.device
        or plan.padded_tokens.device != x.device
        or plan.sorted_ids.dtype != torch.int32
        or plan.expert_ids.dtype != torch.int32
        or plan.padded_tokens.dtype != torch.int32
        or not (plan.sorted_ids.is_contiguous() and plan.expert_ids.is_contiguous() and plan.padded_tokens.is_contiguous())
    ):
        return None
    x_in = x.contiguous()
    if (x_in.data_ptr() & 15) != 0:
        return None
    use_f16_x = 1 if x_in.dtype == torch.float16 else 0
    out_dtype = torch.float16 if (fused and out_fp16) else torch.bfloat16
    out = x.new_empty((*route_shape, n // 2 if fused else n), dtype=out_dtype)

    max_m16 = int(plan.expert_ids.numel())
    routes = int(plan.num_tokens * plan.top_k)
    max_uni = min(max_m16, min(routes, int(plan.num_experts) + (routes + 31) // 32))
    max_uni = max(max_uni, 1)

    uni_desc = _get_decode_scratch(x.device, "moe_dec_unidesc", (max(max_m16, 256),), torch.int32)
    counts = _get_decode_scratch(x.device, "moe_dec_counts", (4,), torch.int32)
    if uni_desc is None or counts is None:
        return None
    stream = torch.cuda.current_stream(x.device).cuda_stream

    capturing = torch.cuda.is_current_stream_capturing()
    need_plan = True
    if capturing:
        if not fused and _CAPTURE_PLANNED.pop(x.device, None) is plan:
            need_plan = False
        else:
            _CAPTURE_PLANNED.pop(x.device, None)
    else:
        _CAPTURE_PLANNED.pop(x.device, None)
        if _LAST_DECODE_PLAN.get(x.device) is plan:
            need_plan = False
    if need_plan:
        rc = lib.launch_hip_moe_decode_plan(
            ctypes.c_void_p(plan.expert_ids.data_ptr()),
            ctypes.c_void_p(plan.padded_tokens.data_ptr()),
            ctypes.c_void_p(uni_desc.data_ptr()),
            ctypes.c_void_p(counts.data_ptr()),
            ctypes.c_int(max_m16),
            ctypes.c_void_p(stream),
        )
        if rc != 0:
            return None
        if capturing:
            if fused:
                _CAPTURE_PLANNED[x.device] = plan
        else:
            _LAST_DECODE_PLAN[x.device] = plan

    if bank_kind == "source":
        if (
            not (bank.raw_scales.is_contiguous() and bank.outer_scales.is_contiguous())
            or bank.raw_scales.device != x.device
            or bank.outer_scales.device != x.device
            or (bank.raw_scales.data_ptr() & 3) != 0
        ):
            return None
        bank_mode = 0
        raw_ptr = bank.raw_scales.data_ptr()
        outer_ptr = bank.outer_scales.data_ptr()
        rs_ptr = tab_ptr = kind_ptr = 0
    elif bank_kind == "coda":
        if (
            not (bank.tags.is_contiguous() and bank.row_scales.is_contiguous() and bank.table.is_contiguous())
            or bank.tags.device != x.device
            or bank.row_scales.device != x.device
            or bank.table.device != x.device
            or (bank.tags.data_ptr() & 3) != 0
        ):
            return None
        bank_mode = 2
        raw_ptr = bank.tags.data_ptr()
        outer_ptr = 0
        rs_ptr = bank.row_scales.data_ptr()
        tab_ptr = bank.table.data_ptr()
        kind_ptr = 0
    else:
        if (
            not (bank.raw_scales.is_contiguous() and bank.outer_scales.is_contiguous() and bank.kind.is_contiguous())
            or bank.raw_scales.device != x.device
            or bank.outer_scales.device != x.device
            or bank.kind.device != x.device
            or (bank.raw_scales.data_ptr() & 3) != 0
            or (bank.row_scales is not None and (not bank.row_scales.is_contiguous() or bank.row_scales.device != x.device))
            or (bank.table is not None and (not bank.table.is_contiguous() or bank.table.device != x.device))
        ):
            return None
        bank_mode = 10 if bank.fitted_kind == 1 else 20
        raw_ptr = bank.raw_scales.data_ptr()
        outer_ptr = bank.outer_scales.data_ptr()
        rs_ptr = bank.row_scales.data_ptr() if bank.row_scales is not None else 0
        tab_ptr = bank.table.data_ptr() if bank.table is not None else 0
        kind_ptr = bank.kind.data_ptr()

    if fused:
        gamma, gate_cap, up_cap = fused_activation
        if gamma.device != x.device:
            return None
        gamma = gamma.float().contiguous()
        gamma_ptr = gamma.data_ptr()
        tokens_div = 8
    else:
        gamma_ptr, gate_cap, up_cap = 0, 0.0, 0.0
        tokens_div = 1

    rc = lib.launch_hip_moe_decode_unified(
        ctypes.c_int(bank_mode),
        ctypes.c_int(1 if fused else 0),
        ctypes.c_int(tokens_div),
        ctypes.c_int(use_f16_x),
        ctypes.c_int(1 if (fused and out_fp16) else 0),
        ctypes.c_void_p(x_in.data_ptr()),
        ctypes.c_void_p(bank.packed.data_ptr()),
        ctypes.c_void_p(raw_ptr),
        ctypes.c_void_p(outer_ptr),
        ctypes.c_void_p(kind_ptr),
        ctypes.c_void_p(rs_ptr),
        ctypes.c_void_p(tab_ptr),
        ctypes.c_void_p(plan.sorted_ids.data_ptr()),
        ctypes.c_void_p(plan.expert_ids.data_ptr()),
        ctypes.c_void_p(uni_desc.data_ptr()),
        ctypes.c_void_p(counts.data_ptr()),
        ctypes.c_void_p(gamma_ptr),
        ctypes.c_void_p(out.data_ptr()),
        ctypes.c_int(routes),
        ctypes.c_int(n),
        ctypes.c_int(k),
        ctypes.c_int(max_uni),
        ctypes.c_float(float(gate_cap)),
        ctypes.c_float(float(up_cap)),
        ctypes.c_void_p(stream),
    )
    return out if rc == 0 else None


# ============================================================================
# SP2 UNIT D4: NATIVE HIP DECODE GLUE & DSPARK WRAPPERS
# ============================================================================


def hip_sconv_decode(
    x: torch.Tensor,
    weight: torch.Tensor,
    cache: torch.Tensor,
    positions: torch.Tensor,
    block_table: torch.Tensor,
    seq_idx: torch.Tensor,
    slot_mapping: torch.Tensor,
    query_start: torch.Tensor,
    stream_offset: int,
    stream_width: int,
    addend: torch.Tensor | None = None,
    residual: torch.Tensor | None = None,
    out: torch.Tensor | None = None,
) -> torch.Tensor | None:
    """Small-M (1..63) request- & aliasing-safe static W=4 short convolution."""
    if not hip_unit_enabled("decode_glue"):
        return None
    if not x.is_cuda or x.ndim != 2 or x.dtype != torch.bfloat16:
        return None
    num_tokens, channels = x.shape
    if (
        num_tokens <= 0
        or num_tokens >= 64
        or weight.device != x.device
        or cache.device != x.device
        or positions.device != x.device
        or block_table.device != x.device
        or seq_idx.device != x.device
        or slot_mapping.device != x.device
        or query_start.device != x.device
        or weight.ndim != 2
        or weight.shape != (channels, 4)
        or channels != cache.shape[1] * stream_width
        or block_table.shape[1] <= 0
        or cache.shape[2] <= 0
        or weight.dtype != torch.bfloat16
        or cache.dtype != torch.bfloat16
        or x.stride(1) != 1
        or cache.stride(3) != 1
    ):
        return None
    if residual is not None and addend is None:
        return None
    if addend is not None and (
        addend.device != x.device
        or addend.dtype not in (torch.bfloat16, torch.float32)
        or addend.shape != (num_tokens, channels)
        or not addend.is_contiguous()
    ):
        return None
    if residual is not None and (
        residual.device != x.device
        or residual.dtype not in (torch.bfloat16, torch.float32)
        or residual.shape != (num_tokens, channels)
        or not residual.is_contiguous()
    ):
        return None
    if out is None:
        out = torch.empty((num_tokens, channels), dtype=cache.dtype, device=x.device)
    if out.device != x.device or out.dtype not in (torch.bfloat16, torch.float32) or out.shape != (num_tokens, channels) or not out.is_contiguous():
        return None
    if (
        positions.dtype not in (torch.int32, torch.int64)
        or block_table.dtype != torch.int32
        or seq_idx.dtype != torch.int32
        or slot_mapping.dtype != torch.int64
        or query_start.dtype != torch.int32
        or not positions.is_contiguous()
        or not seq_idx.is_contiguous()
        or not slot_mapping.is_contiguous()
        or not query_start.is_contiguous()
        or block_table.stride(1) != 1
    ):
        return None
    lib = get_hip_lib(("launch_hip_sconv_decode",))
    if lib is None:
        return None
    stream = torch.cuda.current_stream(x.device).cuda_stream
    rc = lib.launch_hip_sconv_decode(
        ctypes.c_void_p(x.data_ptr()),
        ctypes.c_void_p(weight.data_ptr()),
        ctypes.c_void_p(cache.data_ptr()),
        ctypes.c_void_p(positions.data_ptr()),
        ctypes.c_void_p(block_table.data_ptr()),
        ctypes.c_void_p(seq_idx.data_ptr()),
        ctypes.c_void_p(slot_mapping.data_ptr()),
        ctypes.c_void_p(query_start.data_ptr()),
        ctypes.c_void_p(addend.data_ptr() if addend is not None else 0),
        ctypes.c_void_p(residual.data_ptr() if residual is not None else 0),
        ctypes.c_void_p(out.data_ptr()),
        ctypes.c_int(num_tokens),
        ctypes.c_int(channels),
        ctypes.c_int(cache.shape[1]),
        ctypes.c_int(stream_offset),
        ctypes.c_int(stream_width),
        ctypes.c_int64(x.stride(0)),
        ctypes.c_int64(weight.stride(0)),
        ctypes.c_int64(weight.stride(1)),
        ctypes.c_int64(cache.stride(0)),
        ctypes.c_int64(cache.stride(1)),
        ctypes.c_int64(cache.stride(2)),
        ctypes.c_int64(block_table.stride(0)),
        ctypes.c_int(block_table.shape[1]),
        ctypes.c_int(cache.shape[2]),
        ctypes.c_float(5.0 ** -0.5),
        ctypes.c_int(1 if addend is not None else 0),
        ctypes.c_int(1 if residual is not None else 0),
        ctypes.c_int(1 if (addend is not None and addend.dtype == torch.float32) else 0),
        ctypes.c_int(1 if (residual is not None and residual.dtype == torch.float32) else 0),
        ctypes.c_int(1 if out.dtype == torch.float32 else 0),
        ctypes.c_int(1 if positions.dtype == torch.int64 else 0),
        ctypes.c_void_p(stream),
    )
    return out if rc == 0 else None


def hip_loop_decode_down(pair: torch.Tensor, down_weight: torch.Tensor) -> torch.Tensor | None:
    """Native HIP decode loop controller down projection [S, M, 2*D] @ [S, 64, 2*D]^T -> [S, M, 64]."""
    if not hip_unit_enabled("decode_glue") or not pair.is_cuda or pair.ndim != 3:
        return None
    streams, m, k = pair.shape
    if (
        m <= 0
        or m > 64
        or (k & 1) != 0
        or pair.dtype != torch.bfloat16
        or down_weight.dtype != torch.bfloat16
        or down_weight.shape != (streams, 64, k)
        or down_weight.device != pair.device
    ):
        return None
    lib = get_hip_lib(("launch_hip_loop_decode_down",))
    if lib is None:
        return None
    pair_c = pair.contiguous()
    w_c = down_weight.contiguous()
    if (pair_c.data_ptr() & 3) != 0 or (w_c.data_ptr() & 3) != 0:
        return None
    out = pair.new_empty((streams, m, 64), dtype=torch.bfloat16)
    stream = torch.cuda.current_stream(pair.device).cuda_stream
    rc = lib.launch_hip_loop_decode_down(
        ctypes.c_void_p(pair_c.data_ptr()),
        ctypes.c_void_p(w_c.data_ptr()),
        ctypes.c_void_p(out.data_ptr()),
        ctypes.c_int(streams),
        ctypes.c_int(m),
        ctypes.c_int(k),
        ctypes.c_void_p(stream),
    )
    return out if rc == 0 else None


def hip_loop_decode_blend_rms(
    hidden: torch.Tensor,
    weight: torch.Tensor,
    dest: torch.Tensor,
    message: torch.Tensor,
    norm_weight: torch.Tensor,
    norm_eps: float,
) -> torch.Tensor | None:
    """Single-kernel fused gain/forget dot-64 + tanh blend + RMSNorm (zero intermediate DRAM)."""
    if not hip_unit_enabled("decode_glue") or not dest.is_cuda or dest.ndim != 3:
        return None
    streams, m, width = message.shape
    if (
        streams <= 0
        or m <= 0
        or width != 2304
        or message.device != dest.device
        or hidden.device != dest.device
        or weight.device != dest.device
        or norm_weight.device != dest.device
        or dest.shape != message.shape
        or dest.dtype != torch.float32
        or message.dtype != torch.float32
        or dest.stride(-1) != 1
        or message.stride(-1) != 1
        or hidden.shape != (streams, m, 64)
        or weight.shape != (streams, 2 * width, 64)
        or hidden.dtype != torch.bfloat16
        or weight.dtype != torch.bfloat16
        or not hidden.is_contiguous()
        or not weight.is_contiguous()
        or norm_weight.shape != (streams, width)
        or norm_weight.dtype not in (torch.bfloat16, torch.float32)
        or norm_weight.stride(-1) != 1
        or (hidden.data_ptr() & 3) != 0
        or (dest.data_ptr() & 15) != 0
        or (message.data_ptr() & 15) != 0
        or (weight.data_ptr() & 15) != 0
        or (norm_weight.data_ptr() & 15) != 0
        or (dest.stride(0) & 3) != 0
        or (dest.stride(1) & 3) != 0
        or (message.stride(0) & 3) != 0
        or (message.stride(1) & 3) != 0
    ):
        return None
    lib = get_hip_lib(("launch_hip_loop_decode_blend_rms",))
    if lib is None:
        return None
    out = dest.new_empty(dest.shape, dtype=torch.bfloat16)
    stream = torch.cuda.current_stream(dest.device).cuda_stream
    if 2 <= m <= 64 and streams <= 4 and hasattr(lib, "launch_hip_loop_decode_blend_rms_2pass"):
        blended_tmp = _get_decode_scratch(dest.device, "loop_blend_tmp", (4, 64, 2304), torch.float32)
        part_sq = _get_decode_scratch(dest.device, "loop_blend_psq", (4, 64, 72), torch.float32)
        if blended_tmp is not None and part_sq is not None:
            rc = lib.launch_hip_loop_decode_blend_rms_2pass(
                ctypes.c_void_p(hidden.data_ptr()),
                ctypes.c_void_p(weight.data_ptr()),
                ctypes.c_void_p(dest.data_ptr()),
                ctypes.c_void_p(message.data_ptr()),
                ctypes.c_void_p(norm_weight.data_ptr()),
                ctypes.c_void_p(out.data_ptr()),
                ctypes.c_void_p(blended_tmp.data_ptr()),
                ctypes.c_void_p(part_sq.data_ptr()),
                ctypes.c_int64(dest.stride(0)),
                ctypes.c_int64(dest.stride(1)),
                ctypes.c_int64(message.stride(0)),
                ctypes.c_int64(message.stride(1)),
                ctypes.c_int64(norm_weight.stride(0)),
                ctypes.c_int(streams),
                ctypes.c_int(m),
                ctypes.c_int(1 if norm_weight.dtype == torch.float32 else 0),
                ctypes.c_float(float(norm_eps)),
                ctypes.c_void_p(stream),
            )
            return out if rc == 0 else None
    rc = lib.launch_hip_loop_decode_blend_rms(
        ctypes.c_void_p(hidden.data_ptr()),
        ctypes.c_void_p(weight.data_ptr()),
        ctypes.c_void_p(dest.data_ptr()),
        ctypes.c_void_p(message.data_ptr()),
        ctypes.c_void_p(norm_weight.data_ptr()),
        ctypes.c_void_p(out.data_ptr()),
        ctypes.c_int64(dest.stride(0)),
        ctypes.c_int64(dest.stride(1)),
        ctypes.c_int64(message.stride(0)),
        ctypes.c_int64(message.stride(1)),
        ctypes.c_int64(norm_weight.stride(0)),
        ctypes.c_int(streams),
        ctypes.c_int(m),
        ctypes.c_int(1 if norm_weight.dtype == torch.float32 else 0),
        ctypes.c_float(float(norm_eps)),
        ctypes.c_void_p(stream),
    )
    return out if rc == 0 else None


def hip_pair_rmsnorm(dest: torch.Tensor, message: torch.Tensor) -> torch.Tensor | None:
    """Native HIP paired unit-RMSNorm [S, M, 2304] x 2 -> [S, M, 4608] BF16."""
    if not hip_unit_enabled("decode_glue") or not dest.is_cuda or dest.ndim != 3:
        return None
    streams, m, width = message.shape
    if (
        streams <= 0
        or m <= 0
        or width != 2304
        or message.device != dest.device
        or dest.shape != message.shape
        or dest.dtype != torch.float32
        or message.dtype != torch.float32
        or dest.stride(-1) != 1
        or message.stride(-1) != 1
        or (dest.data_ptr() & 15) != 0
        or (message.data_ptr() & 15) != 0
        or (dest.stride(0) & 3) != 0
        or (dest.stride(1) & 3) != 0
        or (message.stride(0) & 3) != 0
        or (message.stride(1) & 3) != 0
    ):
        return None
    lib = get_hip_lib(("launch_hip_pair_rmsnorm",))
    if lib is None:
        return None
    out = message.new_empty((streams, m, 2 * width), dtype=torch.bfloat16)
    stream = torch.cuda.current_stream(dest.device).cuda_stream
    rc = lib.launch_hip_pair_rmsnorm(
        ctypes.c_void_p(dest.data_ptr()),
        ctypes.c_void_p(message.data_ptr()),
        ctypes.c_void_p(out.data_ptr()),
        ctypes.c_int64(dest.stride(0)),
        ctypes.c_int64(dest.stride(1)),
        ctypes.c_int64(message.stride(0)),
        ctypes.c_int64(message.stride(1)),
        ctypes.c_int(streams),
        ctypes.c_int(m),
        ctypes.c_void_p(stream),
    )
    return out if rc == 0 else None


def hip_blend_rmsnorm(
    dest: torch.Tensor,
    message: torch.Tensor,
    gain_forget: torch.Tensor,
    norm_weight: torch.Tensor,
    norm_eps: float,
) -> torch.Tensor | None:
    """Native HIP tanh blend + RMSNorm [S, M, 2304] -> BF16."""
    if not hip_unit_enabled("decode_glue") or not dest.is_cuda or dest.ndim != 3:
        return None
    streams, m, width = message.shape
    if (
        streams <= 0
        or m <= 0
        or width != 2304
        or message.device != dest.device
        or gain_forget.device != dest.device
        or norm_weight.device != dest.device
        or dest.shape != message.shape
        or dest.dtype != torch.float32
        or message.dtype != torch.float32
        or gain_forget.shape != (streams, m, 2 * width)
        or gain_forget.dtype != torch.float32
        or not gain_forget.is_contiguous()
        or norm_weight.shape != (streams, width)
        or norm_weight.dtype not in (torch.bfloat16, torch.float32)
        or dest.stride(-1) != 1
        or message.stride(-1) != 1
        or norm_weight.stride(-1) != 1
        or (dest.data_ptr() & 15) != 0
        or (message.data_ptr() & 15) != 0
        or (gain_forget.data_ptr() & 15) != 0
        or (norm_weight.data_ptr() & 15) != 0
        or (dest.stride(0) & 3) != 0
        or (dest.stride(1) & 3) != 0
        or (message.stride(0) & 3) != 0
        or (message.stride(1) & 3) != 0
    ):
        return None
    lib = get_hip_lib(("launch_hip_blend_rmsnorm",))
    if lib is None:
        return None
    out = message.new_empty(message.shape, dtype=torch.bfloat16)
    stream = torch.cuda.current_stream(dest.device).cuda_stream
    rc = lib.launch_hip_blend_rmsnorm(
        ctypes.c_void_p(dest.data_ptr()),
        ctypes.c_void_p(message.data_ptr()),
        ctypes.c_void_p(gain_forget.data_ptr()),
        ctypes.c_void_p(norm_weight.data_ptr()),
        ctypes.c_void_p(out.data_ptr()),
        ctypes.c_int64(dest.stride(0)),
        ctypes.c_int64(dest.stride(1)),
        ctypes.c_int64(message.stride(0)),
        ctypes.c_int64(message.stride(1)),
        ctypes.c_int64(norm_weight.stride(0)),
        ctypes.c_int(streams),
        ctypes.c_int(m),
        ctypes.c_int(1 if norm_weight.dtype == torch.float32 else 0),
        ctypes.c_float(float(norm_eps)),
        ctypes.c_void_p(stream),
    )
    return out if rc == 0 else None


def hip_router_gemm_m8(
    x: torch.Tensor,
    w_bf16: torch.Tensor,
    bias: torch.Tensor | None = None,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor] | None:
    """Native HIP FP32 router GEMM for M <= 8 (optionally fusing sigmoid + bias for M=1)."""
    if (
        not hip_unit_enabled("decode_glue")
        or not x.is_cuda
        or x.ndim != 2
        or x.shape[0] <= 0
        or x.shape[0] > 8
        or x.shape[1] != 2304
        or w_bf16.device != x.device
        or w_bf16.shape != (128, 2304)
        or x.dtype != torch.bfloat16
        or w_bf16.dtype != torch.bfloat16
        or not x.is_contiguous()
        or not w_bf16.is_contiguous()
        or (x.data_ptr() & 3) != 0
        or (w_bf16.data_ptr() & 3) != 0
    ):
        return None
    if bias is not None and (bias.device != x.device or bias.shape != (128,) or bias.dtype != torch.float32 or not bias.is_contiguous()):
        return None
    lib = get_hip_lib(("launch_hip_router_gemm_m8",))
    if lib is None:
        return None
    m = x.shape[0]
    out = torch.empty((m, 128), dtype=torch.float32, device=x.device)
    corr = torch.empty_like(out) if bias is not None else None
    stream = torch.cuda.current_stream(x.device).cuda_stream
    rc = lib.launch_hip_router_gemm_m8(
        ctypes.c_void_p(x.data_ptr()),
        ctypes.c_void_p(w_bf16.data_ptr()),
        ctypes.c_void_p(bias.data_ptr() if bias is not None else 0),
        ctypes.c_void_p(out.data_ptr()),
        ctypes.c_void_p(corr.data_ptr() if corr is not None else 0),
        ctypes.c_int(m),
        ctypes.c_int(1 if bias is not None else 0),
        ctypes.c_void_p(stream),
    )
    if rc != 0:
        return None
    return (out, corr) if bias is not None else out


def hip_route_m8(
    logits: torch.Tensor,
    bias: torch.Tensor,
    scale: float,
    mode: str = "align",
    *,
    already_sigmoid: bool = False,
    bias_is_corrected: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None] | None:
    """Bit-exact IEEE-754 stable top-8 router + optional M16 plan packer for M <= 8."""
    if (
        not hip_unit_enabled("decode_glue")
        or mode not in ("tail", "align", "select")
        or not logits.is_cuda
        or bias.device != logits.device
        or logits.ndim != 2
        or logits.shape[0] <= 0
        or logits.shape[0] > 8
        or logits.shape[1] != 128
        or bias.shape != (128,)
        or logits.dtype != torch.float32
        or bias.dtype != torch.float32
        or not logits.is_contiguous()
        or not bias.is_contiguous()
    ):
        return None
    if bias_is_corrected and (logits.shape[0] != 1 or mode != "tail" or not already_sigmoid):
        return None
    lib = get_hip_lib(("launch_hip_route_m8",))
    if lib is None:
        return None
    m = logits.shape[0]
    align = mode != "tail"
    do_sigmoid = (mode != "select") and (not already_sigmoid)
    src_in = logits.sigmoid() if (mode == "select" and not already_sigmoid) else logits
    ids = torch.empty((m, 8), dtype=torch.int64, device=logits.device)
    gamma = torch.empty((m, 8), dtype=torch.float32, device=logits.device)
    plan = None
    if align:
        sorted_ids = torch.empty((m * 128,), dtype=torch.int32, device=logits.device)
        experts = torch.empty((m * 8,), dtype=torch.int32, device=logits.device)
        padded = torch.empty((1,), dtype=torch.int32, device=logits.device)
        plan = (sorted_ids, experts, padded)
    stream = torch.cuda.current_stream(logits.device).cuda_stream
    rc = lib.launch_hip_route_m8(
        ctypes.c_void_p(src_in.data_ptr()),
        ctypes.c_void_p(bias.data_ptr()),
        ctypes.c_void_p(ids.data_ptr()),
        ctypes.c_void_p(gamma.data_ptr()),
        ctypes.c_void_p(plan[0].data_ptr() if plan is not None else 0),
        ctypes.c_void_p(plan[1].data_ptr() if plan is not None else 0),
        ctypes.c_void_p(plan[2].data_ptr() if plan is not None else 0),
        ctypes.c_int(m),
        ctypes.c_int(1 if align else 0),
        ctypes.c_int(1 if do_sigmoid else 0),
        ctypes.c_int(1 if bias_is_corrected else 0),
        ctypes.c_float(float(scale)),
        ctypes.c_void_p(stream),
    )
    return (ids, gamma, plan) if rc == 0 else None


def hip_mix_rmsnorm(
    targets: torch.Tensor, alpha: torch.Tensor, weight: torch.Tensor, eps: float
) -> torch.Tensor | None:
    """Native HIP DSpark alpha-weighted tap mix + RMSNorm."""
    if (
        not hip_unit_enabled("decode_glue")
        or not targets.is_cuda
        or alpha.device != targets.device
        or weight.device != targets.device
        or targets.ndim != 3
        or targets.dtype != torch.bfloat16
        or weight.dtype != torch.bfloat16
        or not weight.is_contiguous()
        or not alpha.is_contiguous()
    ):
        return None
    rows, taps, hidden = targets.shape
    if rows <= 0 or alpha.shape != (taps,) or weight.shape != (hidden,):
        return None
    lib = get_hip_lib(("launch_hip_mix_rmsnorm",))
    if lib is None:
        return None
    alpha_f32 = alpha if alpha.dtype == torch.float32 else alpha.float()
    out = targets.new_empty((rows, hidden))
    stream = torch.cuda.current_stream(targets.device).cuda_stream
    rc = lib.launch_hip_mix_rmsnorm(
        ctypes.c_void_p(targets.data_ptr()),
        ctypes.c_void_p(alpha_f32.data_ptr()),
        ctypes.c_void_p(weight.data_ptr()),
        ctypes.c_void_p(out.data_ptr()),
        ctypes.c_int64(targets.stride(0)),
        ctypes.c_int64(targets.stride(1)),
        ctypes.c_int64(targets.stride(2)),
        ctypes.c_int(rows),
        ctypes.c_int(taps),
        ctypes.c_int(hidden),
        ctypes.c_float(float(eps)),
        ctypes.c_void_p(stream),
    )
    return out if rc == 0 else None


def hip_rms_norm(
    x: torch.Tensor, weight: torch.Tensor, eps: float, out_dtype: torch.dtype | None = None
) -> torch.Tensor | None:
    """Native HIP RMSNorm (BF16/FP32 input & output)."""
    if (
        not hip_unit_enabled("decode_glue")
        or not x.is_cuda
        or weight.device != x.device
        or x.dtype not in (torch.bfloat16, torch.float32)
        or weight.dtype != torch.bfloat16
        or not weight.is_contiguous()
        or x.stride(-1) != 1
    ):
        return None
    x2d = x.reshape(-1, x.shape[-1])
    rows, hidden = x2d.shape
    if rows <= 0 or weight.shape != (hidden,):
        return None
    odt = out_dtype or x.dtype
    if odt not in (torch.bfloat16, torch.float32):
        return None
    lib = get_hip_lib(("launch_hip_rms_norm",))
    if lib is None:
        return None
    out = torch.empty(x.shape, dtype=odt, device=x.device)
    stream = torch.cuda.current_stream(x.device).cuda_stream
    rc = lib.launch_hip_rms_norm(
        ctypes.c_void_p(x2d.data_ptr()),
        ctypes.c_void_p(weight.data_ptr()),
        ctypes.c_void_p(out.data_ptr()),
        ctypes.c_int64(x2d.stride(0)),
        ctypes.c_int(rows),
        ctypes.c_int(hidden),
        ctypes.c_int(1 if x.dtype == torch.float32 else 0),
        ctypes.c_int(1 if odt == torch.float32 else 0),
        ctypes.c_float(float(eps)),
        ctypes.c_void_p(stream),
    )
    return out if rc == 0 else None


def hip_qk_norm_rope_(
    qkv: torch.Tensor,
    positions: torch.Tensor,
    q_weight: torch.Tensor | None,
    k_weight: torch.Tensor | None,
    cos_sin_cache: torch.Tensor,
    num_q_heads: int,
    num_k_heads: int,
    head_dim: int,
    eps: float,
    *,
    norm: bool = True,
) -> bool:
    """In-place native HIP QK-RMSNorm + neox RoPE."""
    if (
        not hip_unit_enabled("decode_glue")
        or not qkv.is_cuda
        or qkv.ndim != 2
        or qkv.dtype != torch.bfloat16
        or qkv.stride(-1) != 1
        or num_q_heads < 0
        or num_k_heads < 0
        or (num_q_heads + num_k_heads) <= 0
        or head_dim <= 0
        or head_dim > 128
        or (head_dim & 1) != 0
        or qkv.shape[1] < (num_q_heads + num_k_heads) * head_dim
        or cos_sin_cache.device != qkv.device
        or cos_sin_cache.ndim != 2
        or cos_sin_cache.shape[1] != head_dim
        or cos_sin_cache.dtype != torch.float32
        or not cos_sin_cache.is_contiguous()
        or positions.device != qkv.device
        or positions.shape != (qkv.shape[0],)
        or positions.dtype != torch.int64
        or not positions.is_contiguous()
    ):
        return False
    if norm and (
        q_weight is None
        or k_weight is None
        or q_weight.device != qkv.device
        or k_weight.device != qkv.device
        or q_weight.shape != (head_dim,)
        or k_weight.shape != (head_dim,)
        or q_weight.dtype != torch.bfloat16
        or k_weight.dtype != torch.bfloat16
        or not q_weight.is_contiguous()
        or not k_weight.is_contiguous()
    ):
        return False
    lib = get_hip_lib(("launch_hip_qk_norm_rope",))
    if lib is None:
        return False
    t_len = qkv.shape[0]
    if t_len <= 0:
        return True
    stream = torch.cuda.current_stream(qkv.device).cuda_stream
    rc = lib.launch_hip_qk_norm_rope(
        ctypes.c_void_p(qkv.data_ptr()),
        ctypes.c_void_p(q_weight.data_ptr() if q_weight is not None else 0),
        ctypes.c_void_p(k_weight.data_ptr() if k_weight is not None else 0),
        ctypes.c_void_p(cos_sin_cache.data_ptr()),
        ctypes.c_void_p(positions.data_ptr()),
        ctypes.c_int64(qkv.stride(0)),
        ctypes.c_int64(num_q_heads * head_dim),
        ctypes.c_int(t_len),
        ctypes.c_int(num_q_heads),
        ctypes.c_int(num_k_heads),
        ctypes.c_int(head_dim),
        ctypes.c_int(1 if norm else 0),
        ctypes.c_float(float(eps)),
        ctypes.c_void_p(stream),
    )
    return rc == 0


def hip_dspark_w8_linear(
    x: torch.Tensor, weight, *, splits: int | None = None
) -> torch.Tensor | None:
    """Native HIP W8-G128 BF16 WMMA linear for DSpark draft projections (M <= 16)."""
    if (
        not hip_unit_enabled("decode_glue")
        or not x.is_cuda
        or x.ndim != 2
        or x.shape[0] <= 0
        or x.shape[0] > 16
        or x.dtype != torch.bfloat16
        or not x.is_contiguous()
        or (x.data_ptr() & 15) != 0
        or x.shape[1] != weight.in_features
        or weight.codes.device != x.device
        or weight.scales.device != x.device
        or not weight.codes.is_contiguous()
        or not weight.scales.is_contiguous()
        or (weight.codes.data_ptr() & 15) != 0
    ):
        return None
    n, k = weight.out_features, weight.in_features
    if splits is None:
        splits = 4 if k == 9216 else 1
    if splits not in (1, 2, 4, 8) or (k % (128 * splits)) != 0 or (n & 63) != 0:
        return None
    lib = get_hip_lib(("launch_hip_dspark_w8", "launch_hip_reduce_splits"))
    if lib is None:
        return None
    m = x.shape[0]
    out = torch.empty((m, n), device=x.device, dtype=torch.bfloat16)
    partial = None
    if splits > 1:
        partial = _get_decode_scratch(x.device, "dspark_w8_partial", (splits, m, n), torch.float32)
        if partial is None:
            return None
    stream = torch.cuda.current_stream(x.device).cuda_stream
    rc = lib.launch_hip_dspark_w8(
        ctypes.c_void_p(x.data_ptr()),
        ctypes.c_void_p(weight.codes.data_ptr()),
        ctypes.c_void_p(weight.scales.data_ptr()),
        ctypes.c_void_p(out.data_ptr()),
        ctypes.c_void_p(partial.data_ptr() if partial is not None else 0),
        ctypes.c_int(m),
        ctypes.c_int(n),
        ctypes.c_int(k),
        ctypes.c_int(splits),
        ctypes.c_void_p(stream),
    )
    return out if rc == 0 else None


def hip_silu_and_mul(x: torch.Tensor) -> torch.Tensor | None:
    """Native HIP silu(gate) * up in FP32 -> BF16."""
    if not hip_unit_enabled("decode_glue") or not x.is_cuda or x.ndim != 2 or x.dtype != torch.bfloat16 or x.stride(1) != 1:
        return None
    rows, width = x.shape
    if rows <= 0 or (width & 1) != 0:
        return None
    lib = get_hip_lib(("launch_hip_silu_and_mul",))
    if lib is None:
        return None
    n = width >> 1
    out = x.new_empty((rows, n))
    stream = torch.cuda.current_stream(x.device).cuda_stream
    rc = lib.launch_hip_silu_and_mul(
        ctypes.c_void_p(x.data_ptr()),
        ctypes.c_void_p(out.data_ptr()),
        ctypes.c_int64(x.stride(0)),
        ctypes.c_int(rows),
        ctypes.c_int(n),
        ctypes.c_void_p(stream),
    )
    return out if rc == 0 else None


def hip_add_softcap(a: torch.Tensor, b: torch.Tensor, softcap: float) -> torch.Tensor | None:
    """Native HIP bf16 softcap * tanh(bf16(a + b) / softcap)."""
    if (
        not hip_unit_enabled("decode_glue")
        or not a.is_cuda
        or b.device != a.device
        or a.ndim != 2
        or a.shape != b.shape
        or a.dtype != torch.bfloat16
        or b.dtype != torch.bfloat16
        or a.stride(1) != 1
        or b.stride(1) != 1
    ):
        return None
    rows, n = a.shape
    if rows <= 0 or n <= 0:
        return None
    lib = get_hip_lib(("launch_hip_add_softcap",))
    if lib is None:
        return None
    out = a.new_empty((rows, n))
    stream = torch.cuda.current_stream(a.device).cuda_stream
    rc = lib.launch_hip_add_softcap(
        ctypes.c_void_p(a.data_ptr()),
        ctypes.c_void_p(b.data_ptr()),
        ctypes.c_void_p(out.data_ptr()),
        ctypes.c_int64(a.stride(0)),
        ctypes.c_int64(b.stride(0)),
        ctypes.c_int(rows),
        ctypes.c_int(n),
        ctypes.c_float(float(softcap)),
        ctypes.c_void_p(stream),
    )
    return out if rc == 0 else None


def hip_markov_gate(raw: torch.Tensor, state: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor] | None:
    """Native HIP DSpark Markov recurrent gate (exact BF16 intermediate rounding)."""
    if (
        not hip_unit_enabled("decode_glue")
        or not raw.is_cuda
        or state.device != raw.device
        or raw.dtype != torch.float32
        or state.dtype != torch.bfloat16
        or not raw.is_contiguous()
        or not state.is_contiguous()
    ):
        return None
    rows, rank = state.shape
    if rows <= 0 or rank <= 0 or rank > 256 or raw.shape != (rows, 3 * rank):
        return None
    lib = get_hip_lib(("launch_hip_markov_gate",))
    if lib is None:
        return None
    out = torch.empty_like(state)
    latent = torch.empty_like(state)
    stream = torch.cuda.current_stream(raw.device).cuda_stream
    rc = lib.launch_hip_markov_gate(
        ctypes.c_void_p(raw.data_ptr()),
        ctypes.c_void_p(state.data_ptr()),
        ctypes.c_void_p(out.data_ptr()),
        ctypes.c_void_p(latent.data_ptr()),
        ctypes.c_int(rows),
        ctypes.c_int(rank),
        ctypes.c_void_p(stream),
    )
    return (out, latent) if rc == 0 else None
