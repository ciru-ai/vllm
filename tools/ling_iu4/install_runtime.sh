#!/usr/bin/env bash
set -Eeuo pipefail

repo_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
venv="${VENV:-$repo_root/.venv}"
uv_cache="${UV_CACHE_DIR:-$repo_root/.cache/uv}"
rocm_index=https://rocm.nightlies.amd.com/whl-multi-arch/
flash_commit=0e60e39473e8df549a20fb5353760f7a65b30e2d
flash_source="$repo_root/.deps/flash-attention"

[[ "$(uname -s)" == Linux ]] || {
  echo "This vLLM/ROCm build requires Linux. Use WSL2/ROCDXG on Windows." >&2
  exit 2
}
command -v git >/dev/null || { echo "Missing git." >&2; exit 2; }
command -v uv >/dev/null || {
  echo "Missing uv. Install it from https://docs.astral.sh/uv/." >&2
  exit 2
}

if grep -qi microsoft /proc/sys/kernel/osrelease 2>/dev/null; then
  [[ -e /dev/dxg ]] || {
    echo "WSL2 detected but /dev/dxg is missing. Install AMD ROCDXG first." >&2
    exit 2
  }
else
  [[ -e /dev/kfd && -d /dev/dri ]] || {
    echo "The AMDGPU KFD/render devices are unavailable (/dev/kfd, /dev/dri)." >&2
    exit 2
  }
fi

mkdir -p "$uv_cache" "$repo_root/.deps"
uv python install 3.12
[[ -x "$venv/bin/python" ]] || uv venv --python 3.12 "$venv"

UV_CACHE_DIR="$uv_cache" uv pip install --python "$venv/bin/python" \
  --index-url "$rocm_index" \
  'torch[device-gfx1151]==2.13.0+rocm7.15.0a20260728' \
  'torchvision[device-gfx1151]==0.28.0+rocm7.15.0a20260728' \
  'torchaudio==2.11.0.2+rocm7.15.0a20260728' \
  'triton==3.8.0+git4cff872c.rocm7.15.0a20260728' \
  'rocm[libraries,devel,device-gfx1151]==7.15.0a20260728'

"$venv/bin/rocm-sdk" init
"$venv/bin/rocm-sdk" test

site="$("$venv/bin/python" - <<'PY'
import site
print(site.getsitepackages()[0])
PY
)"
sdk="$site/_rocm_sdk_devel"
core="$site/_rocm_sdk_core"
[[ -x "$sdk/bin/hipcc" ]]

# Match vLLM's ROCm SDK-wheel environment.
"$venv/bin/python" - "$site/rocm_sdk/__init__.py" <<'PY'
from pathlib import Path
import sys
p = Path(sys.argv[1])
s = p.read_text()
p.write_text(s.replace("rtld_global: bool = True", "rtld_global: bool = False"))
PY

hsakmt_targets="$sdk/lib/cmake/hsakmt/hsakmtTargets.cmake"
if [[ -f "$hsakmt_targets" ]]; then
  sed -i 's|-L/__w/[^;"]*|-L${_IMPORT_PREFIX}/lib/rocm_sysdeps/lib|g' "$hsakmt_targets"
  sed -i 's|/usr/lib64/libc.so;||g' "$hsakmt_targets"
fi

export PATH="$sdk/bin:$sdk/llvm/bin:$venv/bin:$PATH"
export LD_LIBRARY_PATH="$sdk/lib:$core/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export PYTHONPATH="$repo_root:$core/share/amd_smi${PYTHONPATH:+:$PYTHONPATH}"
export HIP_DEVICE_LIB_PATH="$core/lib/llvm/amdgcn/bitcode"
export VLLM_TARGET_DEVICE=rocm
export PYTORCH_ROCM_ARCH=gfx1151
export ROCM_PATH="$sdk" ROCM_HOME="$sdk" ROCM_SOURCE_DIR="$sdk" HIP_PATH="$sdk"
export MAX_JOBS="${MAX_JOBS:-16}"
export CMAKE_BUILD_PARALLEL_LEVEL="$MAX_JOBS"
export CMAKE_PREFIX_PATH="$sdk/lib/cmake:$site/torch/share/cmake"
export CMAKE_ARGS="-DROCM_PATH=$sdk -DCMAKE_HIP_ARCHITECTURES=gfx1151 ${CMAKE_ARGS:-}"

UV_CACHE_DIR="$uv_cache" uv pip install --python "$venv/bin/python" \
  'cmake>=3.26.1,<4' ninja 'packaging>=24.2' \
  'setuptools>=77.0.3,<80' 'setuptools-scm>=8' setuptools-rust wheel 'jinja2>=3.1.6'
UV_CACHE_DIR="$uv_cache" uv pip install --python "$venv/bin/python" "$core/share/amd_smi"
UV_CACHE_DIR="$uv_cache" uv pip install --python "$venv/bin/python" -r "$repo_root/requirements/rocm.txt"
UV_CACHE_DIR="$uv_cache" uv pip install --python "$venv/bin/python" \
  --no-build-isolation --no-deps -e "$repo_root"

if [[ ! -d "$flash_source/.git" ]]; then
  git clone https://github.com/Dao-AILab/flash-attention.git "$flash_source"
fi
git -C "$flash_source" fetch origin "$flash_commit"
git -C "$flash_source" checkout --detach "$flash_commit"
FLASH_ATTENTION_TRITON_AMD_ENABLE=TRUE UV_CACHE_DIR="$uv_cache" \
  uv pip install --python "$venv/bin/python" --no-build-isolation --no-deps "$flash_source"

"$venv/bin/python" "$repo_root/tools/ling_iu4/build_iu4_extension.py"
bash "$repo_root/tools/ling_iu4/install_rocr_idle_fix.sh" "$repo_root"
rocr_override_dir="$repo_root/runtime-overrides/rocr-7.15-gcc-idle"
rocr_idle_runtime="$rocr_override_dir/libhsa-runtime64.so.1.21.0"
(cd "$rocr_override_dir" && sha256sum --check --status SHA256SUMS)
LD_PRELOAD="$rocr_idle_runtime${LD_PRELOAD:+:$LD_PRELOAD}" \
  "$venv/bin/python" - <<'PY'
import importlib
import torch
import triton
assert torch.__version__ == "2.13.0+rocm7.15.0a20260728"
assert torch.version.hip == "7.15.0"
assert triton.__version__ == "3.8.0"
for name in ("vllm._C", "vllm._rocm_C"):
    importlib.import_module(name)
print("CIRU Ling gfx1151 IU4 runtime: PASS")
PY
