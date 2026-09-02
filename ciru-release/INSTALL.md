# Ciru GLM5.3 Strix runtime installation draft

This source freeze is the runtime for **GLM5.3 Flash CIRU STRIX IU4**. It is
not stock vLLM and must not be replaced with a PyPI vLLM wheel.

## Frozen identities

- vLLM base: `dccf404cd4ae5e9c06e940ad9bc49ee8f762a275`
- Python: `3.14.3`
- ROCm SDK: `10.0.0`, gfx1151 device package
- PyTorch: `2.13.0+rocm10.0.0`
- torchvision: `0.28.0+rocm10.0.0`
- torchaudio: `2.11.0.2+rocm10.0.0`
- Triton: `3.8.0+git4cff872c.rocm10.0.0`
- AITER base: `cdcfa833bdf554ca75594c90dde4316ea9b50199`
- Ciru AITER release commit: `2fe3fff6adb4ecbdc0aa2b9c2ce9618b5e6b3046`
- AITER composable-kernel submodule:
  `fdf4bb7fcc984811cef48ce817d89aac064b984a`

The release commit adds the curated files in `SOURCE_FILES.txt`. Generated
vLLM extensions, AITER JIT objects, and the seven model-specific gfx1151
libraries are deliberately not committed because the qualified development
binaries contain absolute Nix-store and `/srv/llm` runpaths. Build them on the
target until relocatable release wheels are qualified.

## Ubuntu or Debian host

Install a current kernel with gfx1151 support and the build prerequisites:

```bash
sudo apt-get update
sudo apt-get install -y \
  build-essential cargo git libdrm-dev libelf-dev libnuma-dev libssl-dev \
  pkg-config rustc xxd
```

Create the Python 3.14.3 environment. The AMD ROCm 10 wheel index and package
pins below match the qualified stack:

```bash
export VLLM_VENV=/srv/llm/venvs/vllm-rocm10-gfx1151
python3.14 -m venv "$VLLM_VENV"
"$VLLM_VENV/bin/python" -m pip install --upgrade pip
"$VLLM_VENV/bin/python" -m pip install \
  --index-url https://stable.repo.amd.com/rocm/whl-next/ \
  'rocm[libraries,devel,device-gfx1151]==10.0.0' \
  'torch[device-gfx1151]==2.13.0+rocm10.0.0' \
  'torchvision[device-gfx1151]==0.28.0+rocm10.0.0' \
  'torchaudio==2.11.0.2+rocm10.0.0'
"$VLLM_VENV/bin/python" -m pip install \
  --extra-index-url https://stable.repo.amd.com/rocm/whl-next/ \
  -r ciru-release/requirements-runtime.lock
```

Install the SDK-bundled AMD SMI Python package without embedding a host path
in the lock:

```bash
site_packages="$($VLLM_VENV/bin/python - <<'PY'
import site
print(site.getsitepackages()[0])
PY
)"
"$VLLM_VENV/bin/python" -m pip install \
  "$site_packages/_rocm_sdk_core/share/amd_smi"
```

## AITER

Use the separately frozen Ciru AITER checkout rather than an unpinned package.
It contains the two gfx1151 paged-MQA fixes and the PyTorch 2.13 namespace fix.

```bash
export AITER_SOURCE=/srv/llm/releases-src/aiter-glm53-strix-iu4-v0.1.0-rc1
git -C "$AITER_SOURCE" submodule update --init --recursive
BUILD_TARGET=rocm ENABLE_CK=1 \
  "$VLLM_VENV/bin/python" -m pip install --no-build-isolation --no-deps \
  -e "$AITER_SOURCE"
```

The namespace fix changes `Library("aiter")` registration to use unqualified
operator names. Without it, PyTorch 2.13 records `aiter::aiter::<op>` and emits
`ValueError: too many values to unpack` from `torch.library` at shutdown.

## Build this vLLM source

Set the ROCm wheel roots before invoking the normal vLLM build:

```bash
export VLLM_SOURCE=/srv/llm/releases-src/vllm-glm53-strix-iu4-v0.1.0-rc1
export PYTORCH_ROCM_ARCH=gfx1151
export VLLM_TARGET_DEVICE=rocm
export ROCM_PATH="$site_packages/_rocm_sdk_devel"
export ROCM_HOME="$ROCM_PATH"
export HIP_PATH="$ROCM_PATH"
export HIP_DEVICE_LIB_PATH="$site_packages/_rocm_sdk_core/lib/llvm/amdgcn/bitcode"
export CMAKE_PREFIX_PATH="$ROCM_PATH/lib/cmake:$site_packages/torch/share/cmake"
export CMAKE_ARGS="-DROCM_PATH=$ROCM_PATH -DCMAKE_HIP_ARCHITECTURES=gfx1151"
"$VLLM_VENV/bin/python" -m pip install --no-build-isolation --no-deps \
  -e "$VLLM_SOURCE"
```

Build the seven model-specific libraries from their frozen source package and
place them at the names expected by the model repository. A release is not
portable merely because an existing `.so` copies successfully; validate that
its dynamic section has no developer `/srv/llm` or literal Nix-store runpath.

## Model-package contract

The qualified `glm53_runtime_prepack` shards already contain every F1
dense/KDA tensor (102 F1 tensor keys per rank). The runtime-prepacked loader
therefore returns before consulting `VLLM_GLM53_HYBRID_MANIFEST`. Do not ship
the old two loader manifests or 69 per-rank loader packets, and do not export
`VLLM_GLM53_HYBRID_MANIFEST` from the release launcher. The model package must
still include the dense-KDA shared library and its configured maximum-M value.

## NixOS host

Use a pinned flake/module rather than copying store hashes from either
development host. The wrapper must derive its runtime loader path, for example:

```nix
CIRU_HOST_LIBRARY_PATH = lib.makeLibraryPath [
  pkgs.stdenv.cc.cc
  pkgs.numactl
];
```

Build inside a development shell containing `gcc`, `pkg-config`, `libdrm`,
`elfutils`, `numactl`, `openssl`, `rustc`, `cargo`, `gnumake`, `git`, and
`xxd`. Add these two build-only settings for the rootless ROCm SDK:

```bash
export CMAKE_ARGS="${CMAKE_ARGS:-} \
  -DHIP_HIPCC_CMAKE_LINKER_HELPER=$VLLM_SOURCE/ciru-release/nixos/hipcc_cmake_linker_helper \
  -DCMAKE_PROJECT_TOP_LEVEL_INCLUDES=$VLLM_SOURCE/ciru-release/nixos/rocm10_numa_target.cmake"
```

The NixOS service module must also configure the `render` group and the scoped
NHI helper/capability when direct NHI is selected. Portable RCCL-over-USB4 does
not require `CAP_SYS_RAWIO`.

## Source-only runtime environment

Copy `ciru-release/runtime-env.sh.example` outside the checkout, set
`VLLM_SOURCE`, `VLLM_VENV`, and `AITER_SOURCE`, then source it from the model
launcher. On NixOS, set `CIRU_HOST_LIBRARY_PATH` from the wrapper-generated
library path. The file intentionally does not execute a command.

## Verification before a model load

```bash
source /path/to/runtime-env.sh
python -c 'import torch, vllm, aiter; print(torch.__version__, vllm.__version__)'
```

The import must exit without a `torch.library` shutdown traceback. Then run the
actual coordinated two-rank packaged-model smoke. Release qualification adds
the chat, tool-call, prefix-cache, and context-profile gates.
