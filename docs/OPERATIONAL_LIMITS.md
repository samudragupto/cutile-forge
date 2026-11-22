# Operational Limits, Known Failure Modes & Unsupported Configurations (`CuTileForge v3.6.0`)

All limits below trace directly to verified artifacts under [`benchmarks/validation_artifacts/`](../benchmarks/validation_artifacts/). Metrics are reported strictly with provenance labels; unvalidated or hardware-blocked targets are explicitly excluded from verified claims.

---

## 1. Strict Numerical & 256-Token Greedy Parity Limits

Even when all transformer/encoder operators execute natively on CuTile FP16 (`0` PyTorch compute fallbacks and cosine similarity $\ge 0.999979$), FP16 accumulation ordering across 24–80 layers causes strict tolerance misses relative to upstream PyTorch/HF/Diffusers FP16/FP32 references:

| Model / Architecture | Artifact | Measured Difference | Strict Gate Miss |
|---|---|---|---|
| `Mistral-7B-v0.1-GPTQ` (32L) | `v3.5.0/phase18_regression/regress_full.json` | `cos=1.000000`, `ppl=5.1475 vs 5.1473`, `max_diff=0.0244` | `greedy=4/5` prompts (`99.84%` tokens over 256 tok) |
| `Qwen/Qwen2.5-0.5B` (24L) | `v3.5.0/phase18_regression/regress_full.json` | `cos=0.999997`, `ppl=12.9381 vs 12.9413`, `max_diff=0.0366` | `greedy=4/5` prompts (`99.92%` tokens over 256 tok) |
| `Mixtral-8x7B-GPTQ` (32L) | `v3.5.0/phase18_regression/regress_full.json` | `cos=0.999994`, `ppl=4.3437 vs 4.3447`, `max_diff=0.0547` | `greedy=0/5` prompts (`99.22%` tokens; top-k router ties) |
| `Llama-2-70B-GPTQ` (80L PCIe) | `v3.5.0/phase18_regression/regress_full.json` | `cos=0.999998`, `ppl=4.8154 vs 4.8143`, `max_diff=0.0254` | `greedy=4/5` prompts (`99.92%` tokens over 256 tok) |
| `SigLIP ViT-B/16` (12L) | `v3.5.0/phase18_regression/regress_full.json` | `emb_cos_sim=0.999979` | `max_abs_diff=0.1250` (gate `< 0.05`) |
| `Whisper-Tiny` Encoder (4L) | `v3.5.0/phase18_regression/regress_full.json` | `emb_cos_sim=0.999991` | `max_abs_diff=0.52344` (gate `< 0.1`) |
| `DiT-XL/2-256` (28L Diffusion) | `v3.5.0/phase18_regression/regress_full.json` | `output_cos_sim=0.997016` | `cos < 0.9999`, `max_abs_diff=0.05908 > 0.05` |

Additionally, in multi-layer FP16 MoE models (`deepseek_moe`), near-tied router logits (`~3.7e-5` difference, smaller than FP16 machine epsilon `9.77e-4`) can flip top-k expert selection on individual tokens between PyTorch FP32 reference and native CuTile FP16 execution.

---

## 2. Context-Length & Sliding-Window Attention Limits

Source: [`benchmarks/validation_artifacts/v3.1.0/phase14c_long_context/manifest.json`](../benchmarks/validation_artifacts/v3.1.0/phase14c_long_context/manifest.json).

- **64K context (`65,536` tokens) on `Mistral-7B-v0.1`:** **`UNSUPPORTED`**. Exceeds `max_position_embeddings=32768` and precomputed 32K RoPE tables; YaRN/NTK extension is not integrated into `NativeModelRuntime`.
- **Context `8,192`–`32,768` on `Mistral-7B-v0.1-GPTQ` (`sliding_window=4096`):** Memory, KV paging (`0` evictions), and decode complete cleanly up to `32,768` tokens (`46.97 tok/s` FP16 KV / `37.46 tok/s` FP8 KV), **but** `NativeModelRuntime` / `fused_attention_gqa` runs unclipped global causal attention or diverges from reference beyond `4,096` tokens, yielding severe perplexity degradation above `8K` (`tail_ppl=602.0` at 8K FP16, `9,572` at 16K, `12,157` at 32K vs `1.0118` at 4K).
- **80-layer `Llama-2-70B-GPTQ` on 32 GB RTX 5090:** Requires Stage -1 PCIe streaming (`55` layers HBM + `25` layers pinned DDR5), limiting single-stream decode to `3.88–3.89 tok/s` (`v3.5.0/phase18_regression/regress_full.json`).

---

## 3. Quantization & Format Support Limits

Source: [`benchmarks/validation_artifacts/v3.2.0/phase15_quant_real/manifest.json`](../benchmarks/validation_artifacts/v3.2.0/phase15_quant_real/manifest.json).

- **`GGUF Q5_0` tensors (`UNSUPPORTED`):** Loading mixed-quantization GGUF files containing `Q5_0` tensors (such as `Qwen/Qwen2.5-0.5B-Instruct-GGUF (q4_k_m)` on tensor `blk.0.attn_q.weight`) fails with:
  `GGML tensor type 'Q5_0' has no native CuTile kernel`.
  Verified GGUF types: `Q4_0`, `Q8_0`, `Q6_K`, and pure `Q4_K_M` (`TheBloke/Mistral-7B-v0.1-GGUF`).
- **Large MoE checkpoints (`qwen2_moe`, `deepseek_moe`) (`SYNTHETIC-ONLY`):** No ungated `< 16 GB` FP16 checkpoint fits side-by-side with the PyTorch reference inside 32 GB VRAM for all-layer numerical comparison (`v3.5.0/phase18_coverage/compatibility_matrix.json`).

---

## 4. Runtime Execution Path & Throughput Deficits

Source: [`docs/EXTERNAL_ENGINE_COMPARISON.md`](EXTERNAL_ENGINE_COMPARISON.md).

- **`UnifiedModelExecutor` has no persistent KV-cache or CUDA Graph decode loop:** For unquantized FP16 small decoders (`Qwen/Qwen2.5-0.5B`), `UnifiedModelExecutor` recomputes full sequences on each decode token (`92.7 tok/s`), losing substantially to `SGLang`, `vLLM`, and even eager `HF Transformers`.
- **`Mistral-7B-v0.1-GPTQ` vs `SGLang`:** On single and batched decode (`C=1/4/16`), `SGLang` achieves higher throughput and lower ITL than `NativeModelRuntime` (`v3.6.0/phase20_external_engines/manifest.json`). CuTileForge's advantage on this checkpoint is native `desc_act=True` GPTQ support (where `vLLM==0.34.0` raises a `VllmConfig` `ValidationError`) and lower static VRAM footprint (`~8.6 GB` allocated vs `~18.5 GB` static allocation).
- **Phase 16 `http_nonstream` `C=64` Throughput Gate (`FAIL`):** Under strict Rule 1 (`>= 85%` of in-process batched throughput), the 3-run median for non-streaming HTTP at `C=64` (`2,519.59 tok/s`, `84.99%` of in-process, `v3.3.0/phase16_optimized/manifest.json`) is **`FAIL`**; the single run-0 measurement (`85.02%`) does not override the 3-run median. Streaming SSE `C=64` passes at `90.98%` (`2,697.14 tok/s`).

---

## 5. Hardware, Distribution, Security & Container Limits

- **RTX 4060 Laptop (`sm_89`, 8 GB):** **`UNVALIDATED`**. One-command runner [`scripts/run_laptop_4060.sh`](../scripts/run_laptop_4060.sh) and `cutile-forge import-hardware-report` exist, but real 4060 hardware execution was deferred.
- **Physical Multi-GPU, NCCL & Distributed Paged KV:** **`BLOCKED: hardware`**. This workstation has 1 physical GPU (`nvidia-smi -L` -> `GPU 0: NVIDIA GeForce RTX 5090`). Loopback `LoopbackCommTransport` tests establish mathematical sharding equivalence **only** (`SIMULATED`) and cannot measure inter-GPU bandwidth or scaling.
- **Independent Reproduction:** **`PENDING_INDEPENDENT_RUN`** (`benchmarks/repro_kit/expected_ranges.json` has not yet been executed by an operator other than the agent).
- **Local Container Daemon Check:** **`BLOCKED`** (`no sudo` / user not in `docker` group: `permission denied while trying to connect to the docker API at unix:///var/run/docker.sock`). GHCR image builds run via [`.github/workflows/docker.yml`](../.github/workflows/docker.yml) on GitHub runners instead.
- **HTTP Serving Boundaries:** Designed for trusted / localhost (`127.0.0.1`) or Bearer-key-protected single-node deployment. Excessive prompts / malformed payloads return `HTTP 400/422`; excess concurrency (`> max_concurrent_requests`) returns `HTTP 429` (`v3.1.0/phase14g_reliability/manifest.json`).
