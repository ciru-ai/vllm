# GLM5.3 Flash CIRU STRIX IU4 runtime freeze

This directory records the source-only runtime release over vLLM base
`dccf404cd4ae5e9c06e940ad9bc49ee8f762a275`.

- `SOURCE_FILES.txt` is the exact 34-file production overlay.
- `SOURCE_MANIFEST.sha256` is optional release provenance for the resulting
  source files; it is not an install or startup step.
- `requirements-runtime.lock` normalizes the qualified Python environment.
- `runtime-env.sh.example` is designed to be sourced by a launcher; it never
  executes the launcher command itself.
- `INSTALL.md` describes the Ubuntu/Debian and NixOS build contract.
- `nixos/` contains the two build-only helpers used with the rootless ROCm SDK.

Backup snapshots, experiment evidence, generated extensions, model libraries,
and host-specific runtime wrappers are intentionally excluded. The separately
frozen AITER checkout is pinned by exact commit in `INSTALL.md`.
