# CIRU Ling IU4 runtime changelog

## 1.1.0 — 2026-08-24

- Added the Ling-only dynamic batched-token scheduler.
- Uses the faster 8K prefill shape for one runnable request and for requests that begin above 8K uncached tokens.
- Raises the prefill ceiling to 16K when at least two short or medium requests are waiting, then returns to 8K as concurrency drains.
- Preserves decode priority and provides `VLLM_LING_DYNAMIC_BATCHED_TOKENS=0` as a static-budget opt-out.
- Improved warmed C1/32K prefill by 31.17% and reduced wall time by 22.15% against the 1.0 static-16K launcher while retaining C6 behavior within practical parity.
- Published the complete dynamic-build benchmark record: matched C1/32K and C6/8K boundaries, C1 2K–128K context sweep, warmed C1–C10 4K concurrency ladder, and HermesAgent-20 C6 validation.
- Retained the corrected ROCr idle runtime, residual2 IU4 prefill, W4 sparse decode, MTP K1, prefix caching, and 262K context support.

## 1.0.0 — 2026-08-23

- Initial public CIRU Ling 3.0 Flash `gfx1151` IU4 runtime.
- Added Ling-specific residual2 S4×S4 IU4 prefill, optimized W4 decode, MTP K1, build/install tooling, and the production launcher.
- Added the checksum-verified ROCr idle-CPU correction.
