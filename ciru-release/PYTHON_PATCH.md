# dev8 transfer-group correction

This source package applies Python-only corrections over the dev6 Ciru
runtime at base commit `9255fd9fb9fedf4b29d574a8d8bb21d93892cc98`.
The base commit is not a claim that this additional patch was committed there.

In `vllm/distributed/kv_transfer/kv_connector/v1/offloading/scheduler.py`,
boundary-state handoffs are remapped from physical KV-group IDs to the
filtered transfer-group indices before aligned or partial stores. GLM's
excluded KpoolTail group otherwise shifts the recurrent-state keys. The load
path also projects physical GPU allocations through `source_group_ids` before
pairing them with the filtered transfer-group configuration. Without this,
cache admission applies the wrong group sizes and fails its allocation assertion.

The packaged launcher caps prefill chunks at 2,304 tokens so DFlash's
conservative Mamba alignment rule cannot skip the first recurrent checkpoint.
The matching wheel is version `0.1.0rc2.dev8+g9255fd9fb9.rocm100`; native
extensions, kernel libraries, weights, and sampling defaults are unchanged.
The original 34-file overlay manifest describes the earlier curated overlay;
this file records the additional Python patch.
