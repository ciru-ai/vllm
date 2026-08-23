# Ling 3.0 Flash CIRU IU4 runtime

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
sudo apt install -y build-essential git libnuma-dev ninja-build python3.12 python3.12-dev python3.12-venv
curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"

git clone --branch ciru/ling3-gfx1151-iu4 \
  https://github.com/ciru-ai/vllm.git "$HOME/vllm-ling-ciru-iu4"
cd "$HOME/vllm-ling-ciru-iu4"
bash tools/ling_iu4/install_runtime.sh
source .venv/bin/activate
```

Download and launch:

```bash
hf download jcbtc/Ling-3.0-Flash-CIRU-IU4 \
  --local-dir "$HOME/models/Ling-3.0-Flash-CIRU-IU4"

MODEL_PATH="$HOME/models/Ling-3.0-Flash-CIRU-IU4" \
  bash tools/ling_iu4/run_ling_ciru_iu4.sh
```

The default server listens at `http://127.0.0.1:8000/v1`, enables Ling's native
MTP K1 verifier, uses six scheduler slots, a 16K batched-token budget, and the
native 262,144-token context. The validated target is a 128 GB Strix Halo APU.

Before building, native Linux must expose `/dev/kfd` and `/dev/dri`; WSL2 users
must install AMD ROCDXG and expose `/dev/dxg`. Native Windows vLLM is not
supported.
