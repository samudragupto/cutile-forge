# Phase 24 RC2 (`v4.0.0` Candidate) Before/After Performance & Correctness Summary

- **Hardware:** `NVIDIA GeForce RTX 5090 (sm_120)` | **RC2 Commit:** `406660fd46ac` | **Comparison Point:** `6f945b8`
- **Frozen Targets Passed:** **15 / 15**
- **Full 8-Model Regression (`regress --tier full`):** **3 PASS / 5 FAIL** (zero regression vs `v3.9.0`)
- **Qwen2.5-0.5B Runtime Qualification:** `NativeFP16DecoderRuntime` (`use_cuda_graph=True`, `graph_fallback_count=0`, buckets=`[512]`, `sync_interval=32`); strict `regress --tier full` Qwen row executes `UnifiedModelExecutor`.
- **Serving SSE Streaming `C=64` (3-run median):** **2996.54 tok/s** (`90.2%` of `3321.03 tok/s` in-proc)

| Metric | `v3.9.0` Baseline | `6f945b8` Point | Frozen Target | RC2 Final | Delta / Speedup | Target Status |
|---|---:|---:|---:|---:|---|---|
| Mistral-7B Prefill S=512 (ms) | 72.94 | 53.758 | <= 55.0 ms | **53.823** | 1.36x | `PASS` |
| Mistral-7B C=1 TTFT (ms) | 16.7 | 6.21 | <= 10.5 ms | **6.09** | 2.74x | `PASS` |
| Mistral-7B C=1 Decode (tok/s) | 167.03 | 257.26 | >= 220.0 tok/s | **257.27** | 1.54x | `PASS` |
| Mistral-7B C=1 ITL p50 (ms) | 5.82 | 3.89 | <= 4.5 ms | **3.89** | 1.50x | `PASS` |
| Mistral-7B C=4 Decode (tok/s) | 621.53 | 999.21 | >= 800.0 tok/s | **998.26** | 1.61x | `PASS` |
| Mistral-7B C=16 Decode (tok/s) | 2308.02 | 3716.81 | >= 2850.0 tok/s | **3701.66** | 1.60x | `PASS` |
| Qwen2.5-0.5B Prefill S=512 (ms) | 10.74 | 4.169 | <= 8.0 ms | **4.165** | 2.58x | `PASS` |
| Qwen2.5-0.5B C=1 Cached Decode (tok/s) | 139.91 | 888.88 | >= 500.0 tok/s | **892.75** | 6.38x | `PASS` |
| Mixtral-8x7B C=1 Decode (tok/s) | 43.9 | 139.87 | >= 62.0 tok/s | **140.28** | 3.20x | `PASS` |
| Mixtral-8x7B Routing Host Syncs / Step | 256 | 0 | <= 0 | **0** | 256 -> 0 | `PASS` |
| HTTP Non-Stream C=64 (tok/s) | 2519.59 | 2797.32 | >= 2610.0 tok/s | **2947.99** | 1.17x (+17.0%) | `PASS` |
| HTTP Non-Stream C=64 (% of in-proc C=64) | 84.99 | 84.29 | >= 88.0% | **88.77** | 2948.0 / 3321.0 tok/s (in-proc +12.0%) | `PASS` |
| 32L 32K FP8 KV Attention (ms/step) | 7.6 | 3.3038 | <= 3.86 ms | **3.3042** | 2.30x | `PASS` |
| PagedKVCache FP8 Bytes / Token | 2056.0 | 2056.0 | <= 2056.0 B | **2056.0** | unchanged (2056 B) | `PASS` |
| Mistral-7B Peak Allocated VRAM (MB) | 8679.0 | 10524.1 | <= 8800.0 MB | **8592.0** | -87.0 MB | `PASS` |
