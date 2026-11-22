# External Engine Comparison (`CuTileForge v3.6.0`)

> Measured on `2026-11-22` in isolated environments on the same `NVIDIA GeForce RTX 5090 (sm_120, CUDA 13.2)` host (`benchmarks/validation_artifacts/v3.6.0/phase20_external_engines/manifest.json`, `5` measured iterations after `2` warmup runs, `OUT_TOKENS=64`, `temperature=0.0`). HuggingFace `transformers` is recorded only as an eager reference and is **NOT** treated as a substitute for `vLLM`, `SGLang`, or `TensorRT-LLM`.

| Engine / Version | Model / Format | Conc | TTFT (ms) | ITL p50/p99 (ms) | Median [Min, Max] (`tok/s`) | Peak VRAM (MB) | Status |
|---|---|---:|---:|---|---:|---:|---|
| CuTileForge Native (`v3.6.0` runtime; logged as `v3.1.0` in probe) | Mistral-7B-v0.1-GPTQ (`desc_act=True`) | 1 | 16.70 | 5.82 / 5.82 | 167.03 [167.01, 167.10] | 8,570.0 | `MEASURED` |
| CuTileForge Native | Mistral-7B-v0.1-GPTQ (`desc_act=True`) | 4 | 0.00 | 6.436 / 6.439 | 621.53 [621.18, 622.55] | 8,592.5 | `MEASURED` |
| CuTileForge Native | Mistral-7B-v0.1-GPTQ (`desc_act=True`) | 16 | 0.00 | 6.932 / 6.937 | 2,308.02 [2,306.29, 2,309.26] | 8,678.9 | `MEASURED` |
| **SGLang 0.5.23** | Mistral-7B-v0.1-GPTQ (`desc_act=True`) | 1 | 9.30 | 3.686 / 3.686 | **265.01** [261.56, 265.39] | 18,512.0 | `MEASURED` |
| **SGLang 0.5.23** | Mistral-7B-v0.1-GPTQ (`desc_act=True`) | 4 | 4.13 | 4.126 / 4.126 | **969.40** [967.64, 971.17] | 18,532.0 | `MEASURED` |
| **SGLang 0.5.23** | Mistral-7B-v0.1-GPTQ (`desc_act=True`) | 16 | 4.91 | 4.905 / 4.905 | **3,261.84** [3,261.52, 3,265.89] | 18,532.0 | `MEASURED` |
| vLLM 0.34.0 | Mistral-7B-v0.1-GPTQ (`desc_act=True`) | 1 | — | — | — | — | `UNSUPPORTED` |
| HF Transformers 5.23.0 (Eager) | Qwen/Qwen2.5-0.5B (`FP16`) | 1 | 7.50 | 6.004 / 6.040 | 166.56 [165.55, 166.81] | 984.1 | `MEASURED` |
| CuTileForge `UnifiedModelExecutor` | Qwen/Qwen2.5-0.5B (`FP16`, no KV-cache/CUDA-graph) | 1 | 6.17 | 5.873 / 7.319 | 170.27 [136.64, 170.58] | 1,957.3 | `DEGRADED` |
| **vLLM 0.34.0** | Qwen/Qwen2.5-0.5B (`FP16`) | 1 | 3.91 | 1.401 / 1.401 | **694.12** [693.39, 698.15] | 16,914.0 | `MEASURED` |
| **vLLM 0.34.0** | Qwen/Qwen2.5-0.5B (`FP16`) | 4 | 1.58 | 1.577 / 1.577 | **2,535.79** [2,532.04, 2,540.56] | 16,914.0 | `MEASURED` |
| **vLLM 0.34.0** | Qwen/Qwen2.5-0.5B (`FP16`) | 16 | 1.69 | 1.685 / 1.685 | **9,494.11** [9,378.40, 9,579.41] | 16,914.0 | `MEASURED` |
| **SGLang 0.5.23** | Qwen/Qwen2.5-0.5B (`FP16`) | 1 | 4.60 | 1.192 / 1.192 | **803.05** [801.67, 803.51] | 18,418.0 | `MEASURED` |
| **SGLang 0.5.23** | Qwen/Qwen2.5-0.5B (`FP16`) | 4 | 1.39 | 1.392 / 1.392 | **2,873.88** [2,848.33, 2,877.07] | 18,438.0 | `MEASURED` |
| **SGLang 0.5.23** | Qwen/Qwen2.5-0.5B (`FP16`) | 16 | 1.48 | 1.478 / 1.478 | **10,825.80** [10,801.08, 11,044.44] | 18,438.0 | `MEASURED` |
| TensorRT-LLM 1.3.0 | Mistral-7B-v0.1-GPTQ / Qwen2.5-0.5B (`FP16 / GPTQ`) | 1 | — | — | — | — | `NOT_RUN` |

## Analysis (Where CuTileForge Wins vs Where It Loses)

- **Where CuTileForge loses (published prominently):**
  - On `Mistral-7B-v0.1-GPTQ`, `SGLang 0.5.23` (`265.01` @ `C=1`, `969.40` @ `C=4`, `3,261.84 tok/s` @ `C=16`) is **`1.59x` faster at `C=1` and `1.41x` faster at `C=16`** than CuTileForge Native (`167.03`, `621.53`, `2,308.02 tok/s`), with lower ITL (`3.686–4.905 ms` vs `5.82–6.932 ms`).
  - On unquantized `Qwen/Qwen2.5-0.5B` (`FP16`), `CuTileForge UnifiedModelExecutor` lacks a persistent Paged KV-cache and CUDA Graph decode loop (`DEGRADED`: `170.27 tok/s`, though Phase 17/18 kernel tuning improved it from `92.7 tok/s` to parity with eager HF `166.56 tok/s`). Both **`SGLang 0.5.23` (`803.05` @ `C=1`, `10,825.80` @ `C=16`)** and **`vLLM 0.34.0` (`694.12` @ `C=1`, `9,494.11` @ `C=16`)** are **`4.1x–4.7x` faster at `C=1`** and scale to ~10K tok/s at `C=16`.
- **Where CuTileForge holds an advantage:**
  - `Mistral-7B-v0.1-GPTQ` (`desc_act=True`): `vLLM 0.34.0` refuses to load this real checkpoint (`ValidationError`), whereas CuTileForge executes it natively (`167.03 tok/s`).
  - Lower memory reservation than SGLang's static KV pool (`8,570 MB` vs `18,512 MB` reported by `nvidia-smi` for `mem_fraction_static=0.5`).
- **Missing Engine (`NOT_RUN`):**
  - `TensorRT-LLM 1.3.0` (in `/var/tmp/arrhat_models/venvs/trtllm_env`) still fails on `import tensorrt_llm` with:
    `RuntimeError: Vendored triton_kernels module not found at /var/tmp/arrhat_models/venvs/trtllm_env/lib/python3.12/site-packages/triton_kernels`
    and is strictly recorded as `NOT_RUN`.
