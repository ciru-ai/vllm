# Ling IU4 Runtime v1.1.0

This release adds a Ling-specific dynamic prefill budget to the public CIRU `gfx1151` runtime. Model weights and the proven IU4/W4/MTP execution paths are unchanged.

## What changed

- One runnable request uses the faster 8K prefill shape.
- A request that begins above 8K uncached tokens remains latched to 8K until prefill completes.
- Two or more waiting short/medium prefills can use the 16K ceiling.
- The scheduler returns to 8K immediately when concurrency drains to one.
- Existing decode priority is preserved.
- Static serving remains available with `VLLM_LING_DYNAMIC_BATCHED_TOKENS=0`.

## Measured impact on one 128 GB Strix Halo APU

Against the v1.0 static-16K launcher, the order-balanced warmed C1/32K comparison measured:

- Prefill: **+31.17%**
- TTFT: **−23.76%**
- Wall time: **−22.15%**
- Decode: **−0.17%** (parity)

At C6/8K, the dynamic policy retained the static-16K serving path:

- Prefill: −0.59%
- Median per-request decode: −0.25%
- Wall time: +0.58%

The four order-balanced matched server loads were:

| Position | C1 32K PP | C1 TG | C1 TTFT | C1 wall | C6 8K PP | C6 median TG | C6 E2E output/s | C6 wall | C6 accept |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| A1 static 16K | 424.369 | 22.727 | 77.216 | 82.805 | 612.935 | 13.232 | 8.585 | 89.460 | 96.401% |
| C1 dynamic | 558.645 | 22.707 | 58.656 | 64.250 | 608.033 | 12.828 | 8.488 | 90.486 | 95.652% |
| C2 dynamic | 555.673 | 22.731 | 58.970 | 64.558 | 611.529 | 13.255 | 8.565 | 89.670 | 95.652% |
| A2 static 16K | 425.172 | 22.786 | 77.070 | 82.645 | 613.897 | 12.916 | 8.566 | 89.653 | 95.652% |

An earlier dedicated-profile oracle check is retained for completeness. It
asked whether one dynamic profile could exactly match separately tuned static
8K C1 and static 16K C6 profiles; it is not the deployed-default comparison:

| Boundary | Profile | PP | Median TG | E2E output/s | TTFT s | Wall s | Accept |
|---|---|---:|---:|---:|---:|---:|---:|
| C1 16K | Static 8K oracle | 645.290 | 22.430 | 4.120 | 25.390 | 31.052 | 96.92% |
| C1 16K | Dynamic | 632.705 | 22.610 | 4.060 | 25.895 | 31.513 | 96.92% |
| C1 32K | Static 8K oracle | 567.140 | 22.760 | 2.020 | 57.780 | 63.359 | 100.00% |
| C1 32K | Dynamic | 555.525 | 22.797 | 1.983 | 58.986 | 64.557 | 100.00% |
| C6 4K | Static 16K oracle | 638.240 | 14.880 | 16.290 | 38.510 | 47.157 | 96.91% |
| C6 4K | Dynamic | 630.710 | 14.260 | 15.980 | 38.970 | 48.062 | 97.16% |
| C6 8K | Static 16K oracle | 601.760 | 11.310 | 8.460 | 79.050 | 90.832 | 96.40% |
| C6 8K | Dynamic | 599.920 | 12.920 | 8.410 | 81.360 | 91.273 | 96.65% |

The dedicated static-8K oracle was 1.95% and 2.05% faster in C1 wall time at
16K and 32K. Dynamic C6 wall was +1.92% at 4K and +0.49% at 8K versus the
dedicated static-16K oracle. Those small gaps do not justify another serving
profile; the order-balanced comparison above is the causal release comparison.

### Dynamic C1 context sweep

One excluded 512-token warm request was followed by one uncached measured pass
at every requested context, with exactly 128 forced output tokens:

| Input | PP tok/s | TG tok/s | Wall s | MTP acceptance |
|---:|---:|---:|---:|---:|
| 2K | 720.688 | 24.688 | 7.990 | 98.44% |
| 4K | 438.732 | 23.828 | 14.704 | 96.92% |
| 8K | 675.071 | 24.360 | 17.356 | 95.38% |
| 16K | 519.984 | 22.725 | 37.136 | 98.44% |
| 32K | 559.334 | 21.819 | 64.449 | 95.38% |
| 64K | 437.604 | 18.568 | 156.650 | 92.42% |
| 128K | 342.860 | 15.963 | 390.310 | 96.92% |

These are intentionally one-pass boundary measurements, not best-of results.
The isolated 4K row did not reproduce after a same-concurrency warm batch: the
warmed C1/4K row below measured 700.276 PP tok/s. Both values are retained so
the anomalous one-pass result is visible rather than silently discarded.

### Dynamic 4K concurrency ladder

Each row used one excluded same-concurrency warm batch followed by one measured
batch of unique uncached 4096-token prompts and exactly 128 output tokens per
request:

| C | Aggregate PP | Median/request TG | Aggregate TG | E2E output/s | Wall s | MTP acceptance |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 700.276 | 23.578 | 23.578 | 11.392 | 11.236 | 96.92% |
| 2 | 704.722 | 19.851 | 39.458 | 14.173 | 18.062 | 97.67% |
| 3 | 688.311 | 17.704 | 53.107 | 15.343 | 25.027 | 96.91% |
| 4 | 671.804 | 16.416 | 65.186 | 15.910 | 32.181 | 97.29% |
| 5 | 661.500 | 15.088 | 74.887 | 16.228 | 39.439 | 96.60% |
| 6 | 651.611 | 14.202 | 84.525 | 16.435 | 46.730 | 96.15% |
| 7 | 641.725 | 13.345 | 92.634 | 16.508 | 54.276 | 96.48% |
| 8 | 642.692 | 12.743 | **100.015** | 16.804 | 60.937 | 95.03% |
| 9 | 645.142 | 9.140 | 80.513 | 17.025 | 67.665 | 95.24% |
| 10 | 647.727 | 7.060 | 68.972 | **17.168** | 74.558 | 95.40% |

C8 is peak aggregate decode. C9 crosses from two to three 16K prefill waves.
The launcher remains at six default slots because C8 improves end-to-end
output by only 2.25% over C6 while increasing batch wall time by 30.4%.

### Full-agent dynamic-build validation

The thinking-enabled HermesAgent-20 C6 run measured:

| Metric | Result |
|---|---:|
| Score | **90** |
| Outcomes | 16 pass / 1 partial / 3 fail |
| Wall time | 1,387.878 s (23:07.9) |
| Prefill | 279.977 tok/s |
| Median/request decode | 4.817 tok/s |
| Active aggregate decode | 18.413 tok/s |
| Suite output / wall | 17.337 tok/s |
| End-to-end throughput | 52.904 tok/s |
| Mean TTFT | 11.259 s |
| MTP acceptance | 90.989% |
| Prompt / computed / cached tokens | 484,645 / 355,774 / 126,720 |
| Generated tokens / requests | 24,062 / 121 |

This is a standalone dynamic-build validation, not a matched comparison to an
older run with different timeout or sampling conditions.

## Validation

- HumanEval/0–5: 24/24 passing natural completions across four independent launches.
- Full HermesAgent-20 thinking-on C6: score 90, 16 pass / 1 partial / 3 fail.
- All 62 requests in the context and concurrency sweeps had exact completion
  counts, zero preemptions, zero prefix reuse, and no runtime faults.
- Zero preemptions and zero GPU/runtime faults in the retained matched scheduler
  and full-agent runs; HermesAgent-20 had no scenario timeout.
- The ROCr idle-CPU correction from the previous public build remains included.

## Upgrade

```bash
git fetch ciru ciru/ling3-gfx1151-iu4 --tags
git checkout ling-iu4-v1.1.0
```

Restart the server with `tools/ling_iu4/run_ling_ciru_iu4.sh`. Existing model files and the built IU4 extension do not need to be replaced.
