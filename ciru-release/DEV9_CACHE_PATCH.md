# dev9 Python cache corrections

Runtime version: `0.1.0rc2.dev9+g9255fd9fb9.rocm100`.

This package inherits the complete dev8 source and native wheel payload at
base commit `9255fd9fb9fedf4b29d574a8d8bb21d93892cc98`, including its existing
store-boundary remap and load-allocation projection. The dev9 changes are
additional Python patches, not a new commit at that base revision.

Updated execution paths:

- `vllm/v1/worker/gpu/model_states/mamba_hybrid.py`: cached Mamba state handling.
- `vllm/distributed/kv_transfer/kv_connector/v1/offloading/scheduler.py`:
  capacity-bounded, rank-common external-prefix admission.
- `vllm/v1/core/sched/scheduler.py`: paired admission and GPU-local cache
  lookup at the common hybrid checkpoint.

Native extensions and separately packaged gfx1151 libraries are unchanged.
The three candidate files were read when the dev9 artifacts were generated.
This packaging record makes no independent inference or qualification claim.
See the model repository's gate evidence for the executed configuration.

For a source rebuild, set `VLLM_VERSION_OVERRIDE=0.1.0rc2.dev9+g9255fd9fb9.rocm100` before the
normal vLLM build. The older `PYTHON_PATCH.md` records the inherited dev8 patch;
the original overlay manifest is historical and was not regenerated.
