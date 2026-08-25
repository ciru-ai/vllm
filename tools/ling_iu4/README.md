# Ling 3.0 Flash CIRU IU4 runtime

**Runtime release: v1.1.0** · [`CHANGELOG.md`](CHANGELOG.md) ·
[`RELEASE_NOTES_v1.1.0.md`](RELEASE_NOTES_v1.1.0.md)

This directory contains the `gfx1151` extension and production launcher used by
`jcbtc/Ling-3.0-Flash-CIRU-IU4`. Stock vLLM can load the checkpoint through its
W4A16 path, but it does not provide the native IU4 prefill results reported on
the model card.

The validated source branch is `ciru/ling3-gfx1151-iu4` in
`https://github.com/ciru-ai/vllm`.

## Ubuntu 24.04 / Debian-family setup

This is the copyable build path. CIRU's published measurements were produced on
native Linux with the exact AMD wheel versions below; Ubuntu has not yet been
claimed as a separate clean-host validation.

```bash
sudo apt update
sudo apt install -y build-essential git libnuma-dev libdrm-dev libelf-dev \
  ninja-build pkg-config python3.12 python3.12-dev python3.12-venv xxd
curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"

git clone --branch ling-iu4-v1.1.0 \
  https://github.com/ciru-ai/vllm.git "$HOME/vllm-ling-ciru-iu4"
cd "$HOME/vllm-ling-ciru-iu4"
bash tools/ling_iu4/install_runtime.sh
source .venv/bin/activate
```

The installer also rebuilds the wheel-pinned ROCr userspace library with the
host compiler. The distributed ROCm 7.15 wheel can otherwise leave a CPU core
busy while `gfx1151` is idle. The launcher verifies this local runtime by
checksum and preloads it automatically.

Download and launch:

```bash
hf download jcbtc/Ling-3.0-Flash-CIRU-IU4 \
  --local-dir "$HOME/models/Ling-3.0-Flash-CIRU-IU4"

MODEL_PATH="$HOME/models/Ling-3.0-Flash-CIRU-IU4" \
  bash tools/ling_iu4/run_ling_ciru_iu4.sh
```

The default server listens at `http://127.0.0.1:8000/v1`, enables Ling's native
MTP K1 verifier, uses six scheduler slots, a 16K upper batched-token budget,
and the native 262,144-token context. Ling's scheduler uses the faster 8K
prefill shape for one runnable request or a latched long uncached prompt, then
uses 16K when two or more short/medium prefills are runnable. Decode priority
is unchanged. The validated target is a 128 GB Strix Halo APU.

In the matched warmed C1/32K release comparison, v1.1 improved prefill by
31.17% and reduced wall time by 22.15% versus the previous static-16K launcher.
The matched C6/8K wall result remained within 0.58% of the static path.
The release notes include every retained dynamic-build row: the C1 2K–128K
context sweep, warmed C1–C10 4K concurrency ladder, matched scheduler
boundaries, and the full HermesAgent-20 C6 validation.

To opt out and use a conventional static budget, set both controls explicitly:

```bash
VLLM_LING_DYNAMIC_BATCHED_TOKENS=0 MAX_NUM_BATCHED_TOKENS=8192 \
  MODEL_PATH="$HOME/models/Ling-3.0-Flash-CIRU-IU4" \
  bash tools/ling_iu4/run_ling_ciru_iu4.sh
```

## Idle-CPU hotfix for an existing checkout

Update the public branch, build the corrected ROCr runtime once, and restart
the server:

```bash
git pull --ff-only
bash tools/ling_iu4/install_rocr_idle_fix.sh "$PWD"
```

The hotfix preserves the model, vLLM environment, and IU4 extension. For an
emergency diagnostic rollback only, launch once with
`CIRU_DISABLE_ROCR_IDLE_FIX=1`; this restores the wheel runtime and may restore
the idle CPU spin.

Before building, native Linux must expose `/dev/kfd` and `/dev/dri`; WSL2 users
must install AMD ROCDXG and expose `/dev/dxg`. Native Windows vLLM is not
supported.
