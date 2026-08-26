# Ling IU4 Runtime v1.1.1

This patch hardens the ROCr idle-CPU correction shipped in v1.0 and retained
in v1.1.0. The corrected library itself is unchanged; the activation path is
now enforced inside the private runtime environment as well as by the CIRU
launcher.

## Why this patch exists

The ROCm 7.15 wheel-bundled ROCr library can leave one EngineCore CPU thread at
100% while the server has no requests. The supported launcher already used a
checksum-verified `LD_PRELOAD`, but a direct `vllm serve` command could bypass
that environment and silently load the wheel copy.

v1.1.1 closes that gap:

- `install_rocr_idle_fix.sh` activates the corrected runtime inside the selected
  venv, preserves the original wheel file for recovery, and installs a Python
  startup bootstrap that enforces `LD_PRELOAD` before ROCm packages import.
- `run_ling_ciru_iu4.sh` still verifies and preloads the corrected runtime and
  now prints an explicit activation banner.
- `verify_rocr_idle_fix.sh` checks the library actually mapped by EngineCore and
  samples instantaneous thread CPU after startup.
- Existing non-default environments are supported with `VENV=/path/to/venv`.

## Upgrade

Stop the running server, update the checkout, activate the fix in the same venv,
and restart:

```bash
git fetch --tags
git checkout ling-iu4-v1.1.1
VENV="${VIRTUAL_ENV:-$PWD/.venv}" \
  bash tools/ling_iu4/install_rocr_idle_fix.sh "$PWD"
MODEL_PATH=/path/to/Ling-3.0-Flash-CIRU-IU4 \
  bash tools/ling_iu4/run_ling_ciru_iu4.sh
```

With the server ready and idle:

```bash
bash tools/ling_iu4/verify_rocr_idle_fix.sh
```

The verifier reports `PASS` only when EngineCore maps a library whose checksum
matches the corrected local build. It also reports aggregate and hottest-thread
CPU over a fresh sample, avoiding the lifetime-average `%CPU` value shown by
some `ps` commands after a compilation-heavy startup.

## Compatibility

Model weights, dynamic scheduler behavior, IU4/W4 kernels, MTP K1, context
support, and serving defaults are unchanged from v1.1.0.
