# Serving Stress & Sustained Soak Report (`CuTileForge v3.1.0`)

> Live OpenAI HTTP (`127.0.0.1`) validation artifact: `benchmarks/validation_artifacts/v3.1.0/phase14f_serving_stress/manifest.json`

## Concurrency Sweep

| Concurrency | Sent / OK | Success (%) | Agg Throughput (`tok/s`) | Per-Req p50 (`tok/s`) | TTFT p50/p95/p99 (ms) | ITL p50/p95/p99 (ms) | Server VRAM Alloc (MB) | Server RSS (MB) |
|---:|---|---:|---:|---:|---|---|---:|---:|
| 1 | 1 / 1 | 100.0% | 87.93 | 87.96 | 82.22/82.22/82.22 | 6.63/6.71/6.78 | 9080.97 | 5437.3 |
| 4 | 4 / 4 | 100.0% | 113.41 | 63.32 | 36.35/116.16/125.72 | 6.61/6.83/6.88 | 9247.08 | 5437.3 |
| 16 | 16 / 16 | 100.0% | 189.42 | 17.24 | 116.03/887.29/951.16 | 6.62/6.81/6.93 | 9760.66 | 5437.3 |
| 64 | 64 / 64 | 100.0% | 189.01 | 5.01 | 464.99/4371.0/4715.95 | 6.61/6.77/6.83 | 10446.85 | 5437.3 |
| 128 | 128 / 128 | 100.0% | 198.86 | 2.69 | 883.18/8402.1/9124.34 | 6.62/6.76/6.93 | 10451.46 | 5437.3 |

## Sustained Soak Summary

- **Duration:** `900.1 s` | **Succeeded / Failed:** `10092 / 0`
- **Injected Malformed Requests Rejected Safely:** `505`
- **User Output Isolation Verified:** `True`
- **TTFT p50/p95/p99:** `22.56 / 145.07 / 145.96 ms`
- **ITL p50/p95/p99:** `6.62 / 22.13 / 22.4 ms`
- **Post-Warmup Host RSS Slope:** `-0.0 MB/min`
