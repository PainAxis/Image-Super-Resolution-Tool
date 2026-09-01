# Performance Baseline

Performance depends on CPU, NumPy build, image content, scale, and FXAA. The
numbers below are a reproducible comparison, not a universal latency promise.

## Reference run

- Date: 2026-08-24
- CPU: Intel Xeon Platinum 8573C (container allocation: 9 cores)
- OS: Linux x86-64, glibc 2.39
- Python: 3.12.13
- NumPy: 2.3.5
- Input: deterministic 512×512 float32 RGB (`seed=20260824`)
- Output: 1024×1024 (1.049 megapixels)

| Pipeline | Audited commit `a44bc0e` | Remediated branch | Speed-up |
|---|---:|---:|---:|
| EASU + RCAS | 2.084 s | 0.606 s | 3.44× |
| EASU + RCAS + FXAA | 4.503 s | 1.253 s | 3.59× |

Each value is the median of five timed runs after one untimed warm-up. The two
variants used the same input and environment; shared-runner timing can still
vary, which is why this is a reproducible reference rather than a CI gate.

Reproduce the current side with:

```bash
python -m benchmarks.benchmark_pipeline --width 512 --height 512 --scale 2
python -m benchmarks.benchmark_pipeline --width 512 --height 512 --scale 2 --fxaa
```

## Memory contract

Algorithms allocate one full output because that is their return value. Their
additional working arrays are tiled. The preflight estimate reserves input RGB,
simultaneous stage buffers, one FXAA luma plane, tile working data, and allocator
headroom. Transparent jobs reserve additional input and output planes for alpha
premultiplication, resizing, and unpremultiplication. It intentionally
overestimates small jobs; the opaque 512×512 at 2× case is reported as 291 MiB.

Default limits admit an opaque 3840×2160 input at 4× only when sufficient RAM
is currently available. The more conservative transparent-job estimate rejects
that combination by default, and the output-pixel limit rejects 8K at 4× before
allocation. See the README for explicit environment overrides.

## Regression policy

Correctness, block invariance, cancellation, and range tests are hard CI gates.
Timing is kept as a documented manual baseline because shared CI runners are too
variable for a trustworthy wall-time gate. A performance-changing PR should run
the commands above before and after on the same machine and explain regressions
greater than 20%.
