#!/usr/bin/env bash
set -Eeuo pipefail

model_path="${MODEL_PATH:-${1:-}}"
if [[ -z "$model_path" ]]; then
  echo "Set MODEL_PATH or pass the downloaded model directory as argument 1." >&2
  exit 2
fi

repo_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
python_bin="${PYTHON_BIN:-python}"
iu4_library="${LING_IU4_LIBRARY:-$repo_root/tools/ling_iu4/build/ling_decode_iu4_ext.so}"
rocr_override_dir="${ROCR_IDLE_RUNTIME_DIR:-$repo_root/runtime-overrides/rocr-7.15-gcc-idle}"
rocr_idle_runtime="$rocr_override_dir/libhsa-runtime64.so.1.21.0"

[[ "$(uname -s)" == Linux ]] || {
  echo "The ROCm IU4 runtime requires Linux. Use WSL2/ROCDXG on Windows." >&2
  exit 2
}
[[ -d "$model_path" ]] || { echo "Model directory not found: $model_path" >&2; exit 2; }
[[ -r "$iu4_library" ]] || {
  echo "IU4 extension not found: $iu4_library" >&2
  echo "Build it first with: $python_bin tools/ling_iu4/build_iu4_extension.py" >&2
  exit 2
}
case "$rocr_override_dir" in
  *[[:space:]:]*)
    echo "ROCR_IDLE_RUNTIME_DIR cannot contain whitespace or ':': $rocr_override_dir" >&2
    exit 2
    ;;
esac
if [[ "${CIRU_DISABLE_ROCR_IDLE_FIX:-0}" == 1 ]]; then
  echo "WARNING: CIRU ROCr idle-runtime override is disabled." >&2
else
  if [[ ! -r "$rocr_idle_runtime" || ! -r "$rocr_override_dir/SHA256SUMS" ]]; then
    echo "Verified ROCr idle-runtime override is missing from: $rocr_override_dir" >&2
    echo "Install it with: bash tools/ling_iu4/install_rocr_idle_fix.sh '$repo_root'" >&2
    exit 2
  fi
  if ! (cd "$rocr_override_dir" && sha256sum --check --status SHA256SUMS); then
    echo "ROCr idle-runtime checksum verification failed: $rocr_idle_runtime" >&2
    exit 2
  fi
fi

export VLLM_TARGET_DEVICE=rocm
export VLLM_ROCM_USE_AITER=0
export VLLM_USE_AITER=0
export VLLM_ROCM_USE_SKINNY_GEMM=0
export VLLM_ROCM_SAFE_MERGE_ATTN_STATES=1
export FLASH_ATTENTION_TRITON_AMD_ENABLE=TRUE
export FLASH_ATTENTION_TRITON_AMD_AUTOTUNE=FALSE
export HIP_FORCE_DEV_KERNARG=1
export HSA_NO_SCRATCH_RECLAIM=1

# CIRU's production allocation: native IU4 for occupied expert work, with the
# canonical packed weights retained for the optimized sparse W4 path.
export VLLM_LING_IU4_DECODE=grouped_tiled
export VLLM_LING_IU4_DIRECT_OUT=0
export VLLM_LING_IU4_TILED_WEIGHTS=1
export VLLM_LING_IU4_MIN_TOKENS=1
export VLLM_LING_IU4_LIBRARY="$iu4_library"
export LING_IU4_PREFILL_QUANT=residual2

args=(
  serve "$model_path"
  --host "${HOST:-127.0.0.1}"
  --port "${PORT:-8000}"
  --served-model-name Ling-3.0-Flash-CIRU-IU4
  --trust-remote-code
  --attention-backend TRITON_MLA
  --moe-backend triton
  --enable-chunked-prefill
  --enable-prefix-caching
  --mamba-cache-mode align
  --enable-auto-tool-choice
  --tool-call-parser ling3
  --reasoning-parser ling3
  --max-model-len "${MAX_MODEL_LEN:-262144}"
  --max-num-seqs "${MAX_NUM_SEQS:-6}"
  --max-num-batched-tokens "${MAX_NUM_BATCHED_TOKENS:-16384}"
  --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION:-0.72}"
  --compilation-config '{"mode":3,"cudagraph_mode":"NONE","compile_sizes":[1,2]}'
  --speculative-config '{"method":"mtp","num_speculative_tokens":1}'
)

if [[ "${DRY_RUN:-0}" == 1 ]]; then
  printf '%q ' "$python_bin" -m vllm.entrypoints.cli.main "${args[@]}"
  printf '\n'
  exit 0
fi

if [[ "${CIRU_DISABLE_ROCR_IDLE_FIX:-0}" != 1 ]]; then
  export LD_PRELOAD="$rocr_idle_runtime${LD_PRELOAD:+:$LD_PRELOAD}"
fi

exec "$python_bin" -m vllm.entrypoints.cli.main "${args[@]}"
