# External Engine Comparison (`CuTileForge v3.1.0`)

> Measured in isolated environments on the same RTX 5090 (`sm_120`, CUDA 13.2) host. HuggingFace `transformers` is recorded only as an eager reference and is NOT treated as a substitute for `vLLM`, `SGLang`, or `TensorRT-LLM`.

| Engine/version | Model/format | Concurrency | Hardware | TTFT | ITL p50/p99 | Throughput | Peak VRAM | Status |
|---|---|---:|---|---:|---|---:|---:|---|
| CuTileForge v3.1.0 (Native) | Mistral-7B-v0.1-GPTQ (`GPTQ 4-bit (desc_act=True)`) | 1 | RTX 5090 (`sm_120`) | 16.74 ms | 5.8/5.81 ms | 167.48 tok/s | 8570.0 MB | `MEASURED` |
| CuTileForge v3.1.0 (Native) | Mistral-7B-v0.1-GPTQ (`GPTQ 4-bit (desc_act=True)`) | 4 | RTX 5090 (`sm_120`) | 0.0 ms | 6.461/6.463 ms | 619.12 tok/s | 8592.5 MB | `MEASURED` |
| CuTileForge v3.1.0 (Native) | Mistral-7B-v0.1-GPTQ (`GPTQ 4-bit (desc_act=True)`) | 16 | RTX 5090 (`sm_120`) | 0.0 ms | 6.925/6.928 ms | 2310.52 tok/s | 8678.9 MB | `MEASURED` |
| HF Transformers 5.23.0 (Eager) | Qwen/Qwen2.5-0.5B (`FP16`) | 1 | RTX 5090 (`sm_120`) | 7.74 ms | 6.284/6.325 ms | 159.13 tok/s | 984.1 MB | `MEASURED` |
| CuTileForge UnifiedModelExecutor | Qwen/Qwen2.5-0.5B (`FP16 (no KV-cache/CUDA-graph)`) | 1 | RTX 5090 (`sm_120`) | 11.05 ms | 10.788/11.251 ms | 92.7 tok/s | 1963.6 MB | `DEGRADED` |
| vLLM 0.34.0 | Mistral-7B-v0.1-GPTQ (`GPTQ 4-bit (desc_act=True)`) | 1 | RTX 5090 (`sm_120`) | — | —/— | — | — | `UNSUPPORTED` |
| vLLM 0.34.0 | Qwen/Qwen2.5-0.5B (`FP16`) | 1 | RTX 5090 (`sm_120`) | 3.5 ms | 1.397/1.397 ms | 699.55 tok/s | 16918.0 MB | `MEASURED` |
| vLLM 0.34.0 | Qwen/Qwen2.5-0.5B (`FP16`) | 4 | RTX 5090 (`sm_120`) | 1.57 ms | 1.57/1.57 ms | 2547.65 tok/s | 16918.0 MB | `MEASURED` |
| vLLM 0.34.0 | Qwen/Qwen2.5-0.5B (`FP16`) | 16 | RTX 5090 (`sm_120`) | 1.65 ms | 1.651/1.651 ms | 9693.69 tok/s | 16918.0 MB | `MEASURED` |
| SGLang 0.5.23 | Qwen/Qwen2.5-0.5B (`FP16`) | 1 | RTX 5090 (`sm_120`) | 4.79 ms | 1.187/1.187 ms | 804.45 tok/s | 18422.0 MB | `MEASURED` |
| SGLang 0.5.23 | Qwen/Qwen2.5-0.5B (`FP16`) | 4 | RTX 5090 (`sm_120`) | 1.4 ms | 1.403/1.403 ms | 2851.46 tok/s | 18442.0 MB | `MEASURED` |
| SGLang 0.5.23 | Qwen/Qwen2.5-0.5B (`FP16`) | 16 | RTX 5090 (`sm_120`) | 1.49 ms | 1.494/1.494 ms | 10712.17 tok/s | 18442.0 MB | `MEASURED` |
| SGLang 0.5.23 | Mistral-7B-v0.1-GPTQ (`GPTQ 4-bit (desc_act=True)`) | 1 | RTX 5090 (`sm_120`) | 9.28 ms | 3.681/3.681 ms | 265.36 tok/s | 18516.0 MB | `MEASURED` |
| SGLang 0.5.23 | Mistral-7B-v0.1-GPTQ (`GPTQ 4-bit (desc_act=True)`) | 4 | RTX 5090 (`sm_120`) | 4.12 ms | 4.123/4.123 ms | 970.25 tok/s | 18536.0 MB | `MEASURED` |
| SGLang 0.5.23 | Mistral-7B-v0.1-GPTQ (`GPTQ 4-bit (desc_act=True)`) | 16 | RTX 5090 (`sm_120`) | 4.91 ms | 4.907/4.907 ms | 3260.69 tok/s | 18536.0 MB | `MEASURED` |
| TensorRT-LLM 1.3.0 | Mistral-7B-v0.1-GPTQ / Qwen2.5-0.5B (`FP16 / GPTQ`) | 1 | RTX 5090 (`sm_120`) | — | —/— | — | — | `NOT_RUN` |

## Analysis (Strict Parity Assessment)

- **Where CuTileForge is faster / uniquely functional:**
  - `Mistral-7B-v0.1-GPTQ` (`desc_act=True`, on-disk real checkpoint): `vLLM==0.34.0` refuses to load the checkpoint with a `VllmConfig` `ValidationError` (`desc_act=True` unsupported), whereas CuTileForge executes natively end-to-end.
  - Lower peak VRAM than SGLang's static pool (`8.57 GB` allocated vs `18.52 GB` reported by `nvidia-smi` for SGLang `mem_fraction_static=0.5`).
- **Where CuTileForge is slower:**
  - On `Mistral-7B-v0.1-GPTQ`, `SGLang==0.5.23` (`265.36 tok/s` @ `C=1`, `970.25 tok/s` @ `C=4`, `3,260.69 tok/s` @ `C=16`) outperforms CuTileForge `v3.1.0` (`167.48`, `619.12`, and `2,310.52 tok/s`) in raw throughput and ITL.
  - On unquantized `Qwen/Qwen2.5-0.5B` (FP16), `CuTileForge UnifiedModelExecutor` lacks a Qwen2 Paged KV-cache + CUDA Graph decode loop and achieves `92.7 tok/s`, which is much slower than `SGLang 0.5.23` (`804.45 tok/s`), `vLLM 0.34.0` (`699.55 tok/s`), and eager `HF Transformers` (`159.13 tok/s`).
- **Memory use & missing engines (`NOT_RUN`):**
  - `TensorRT-LLM 1.3.0` installed under `/var/tmp/arrhat_models/venvs/trtllm_env` but failed on `import tensorrt_llm` with `RuntimeError: Vendored triton_kernels module not found`, and is strictly recorded as `NOT_RUN` (never plotted as zero).

Exact error (`vLLM 0.34.0` on `Mistral-7B-v0.1-GPTQ`):
```text
ValidationError: 1 validation error for VllmConfig
  Value error, GPTQ group activation ordering (desc_act=True) is no longer supported. Use a checkpoint with static activation ordering or desc_act=False. [type=value_error, input_value=ArgsKwargs((), {'model_co... 'shutdown_timeout': 0}), input_type=ArgsKwargs]
    For further information visit https://errors.pydantic.dev/2.14/v/value_error
```

Exact error (`TensorRT-LLM 1.3.0` on `Mistral-7B-v0.1-GPTQ / Qwen2.5-0.5B`):
```text
File "/var/tmp/arrhat_models/venvs/trtllm_env/lib/python3.12/site-packages/tensorrt_llm/__init__.py", line 28, in <module>
    _prepare_environment()
  File "/var/tmp/arrhat_models/venvs/trtllm_env/lib/python3.12/site-packages/tensorrt_llm/_bootstrap.py", line 125, in _prepare_environment
    _setup_vendored_triton_kernels()
  File "/var/tmp/arrhat_models/venvs/trtllm_env/lib/python3.12/site-packages/tensorrt_llm/_bootstrap.py", line 108, in _setup_vendored_triton_kernels
    raise RuntimeError(f"Vendored triton_kernels module not found at {vendored}")
RuntimeError: Vendored triton_kernels module not found at /var/tmp/arrhat_models/venvs/trtllm_env/lib/python3.12/site-packages/triton_kernels
```
