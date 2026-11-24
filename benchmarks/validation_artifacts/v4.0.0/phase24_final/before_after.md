# Phase 24 (`v4.0.0` Candidate) Before/After Performance & Correctness Summary

- **Hardware:** `NVIDIA GeForce RTX 5090 (sm_120)` | **Commit:** `f8d92dc22dc3`
- **Full 8-Model Regression (`regress --tier full`):** **3 PASS / 5 FAIL** (zero regression vs `v3.9.0`)
- **Serving SSE Streaming `C=64` (3-run median):** **2946.45 tok/s** (`88.8%` of `3318.87 tok/s` in-proc)

| Metric | `v3.9.0` Baseline | Frozen Target | Phase 24 Final | Delta / Speedup | Target Status |
|---|---:|---:|---:|---|---|
| Mistral-7B Prefill S=512 (ms) | 72.94 | <= 55.0 ms | **53.758** | 1.36x | `PASS` |
| Mistral-7B C=1 TTFT (ms) | 16.7 | <= 10.5 ms | **6.21** | 2.69x | `PASS` |
| Mistral-7B C=1 Decode (tok/s) | 167.03 | >= 220.0 tok/s | **257.26** | 1.54x | `PASS` |
| Mistral-7B C=1 ITL p50 (ms) | 5.82 | <= 4.5 ms | **3.89** | 1.50x | `PASS` |
| Mistral-7B C=4 Decode (tok/s) | 621.53 | >= 800.0 tok/s | **999.21** | 1.61x | `PASS` |
| Mistral-7B C=16 Decode (tok/s) | 2308.02 | >= 2850.0 tok/s | **3716.81** | 1.61x | `PASS` |
| Qwen2.5-0.5B Prefill S=512 (ms) | 10.74 | <= 8.0 ms | **4.169** | 2.58x | `PASS` |
| Qwen2.5-0.5B C=1 Cached Decode (tok/s) | 139.91 | >= 500.0 tok/s | **888.88** | 6.35x | `PASS` |
| Mixtral-8x7B C=1 Decode (tok/s) | 43.9 | >= 62.0 tok/s | **139.87** | 3.19x | `PASS` |
| Mixtral-8x7B Routing Host Syncs / Step | 256 | <= 0 | **0** | 256 -> 0 | `PASS` |
| HTTP Non-Stream C=64 (tok/s) | 2519.59 | >= 2610.0 tok/s | **2797.32** | 1.11x (+11.0%) | `PASS` |
| HTTP Non-Stream C=64 (% of in-proc C=64) | 84.99 | >= 88.0% | **84.29** | 2797.3 / 3318.9 tok/s (in-proc +11.9%) | `FAIL` |
| 32L 32K FP8 KV Attention (ms/step) | 7.6 | <= 3.86 ms | **3.3038** | 2.30x | `PASS` |
| PagedKVCache FP8 Bytes / Token | 2056.0 | <= 2056.0 B | **2056.0** | unchanged (2056 B) | `PASS` |
| Mistral-7B Peak Allocated VRAM (MB) | 8679.0 | <= 8800.0 MB | **10524.1** | +1845.1 MB | `FAIL` |
