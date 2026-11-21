# Model Compatibility & Long-Context Matrix (`CuTileForge v3.1.0`)

## Real-Checkpoint End-to-End Long-Context (`phase14c_long_context`)

| Model (`weight_source`) | Context | KV Dtype | Status | Prefill / TTFT (ms) | Decode (`tok/s`) | ITL p50/p99 (ms) | Peak VRAM (MB) | KV Bytes | Tail PPL |
|---|---:|---|---|---:|---:|---:|---:|---:|---:|
| Mistral-7B-v0.1-GPTQ (32L) (`real-checkpoint`) | 4096 | fp16 | `MEASURED` | 608.33 | 118.64 | 8.429/107.436 | 5058.8 | 536870912 | 1.0118 |
| Mistral-7B-v0.1-GPTQ (32L) (`real-checkpoint`) | 4096 | fp8_e4m3fn | `MEASURED` | 747.17 | 106.66 | 9.375/728.629 | 5314.8 | 805306368 | 1.0114 |
| Mistral-7B-v0.1-GPTQ (32L) (`real-checkpoint`) | 8192 | fp16 | `MEASURED` | 1256.94 | 103.85 | 9.63/19.573 | 5572.9 | 1073741824 | 602.0305 |
| Mistral-7B-v0.1-GPTQ (32L) (`real-checkpoint`) | 8192 | fp8_e4m3fn | `MEASURED` | 1233.26 | 89.83 | 11.133/22.356 | 6082.9 | 1610612736 | 1.0155 |
| Mistral-7B-v0.1-GPTQ (32L) (`real-checkpoint`) | 16384 | fp16 | `MEASURED` | 2694.02 | 73.84 | 13.542/26.892 | 6606.9 | 2147483648 | 9572.2274 |
| Mistral-7B-v0.1-GPTQ (32L) (`real-checkpoint`) | 16384 | fp8_e4m3fn | `MEASURED` | 2670.13 | 61.17 | 16.348/31.069 | 7618.9 | 3221225472 | 9858.2693 |
| Mistral-7B-v0.1-GPTQ (32L) (`real-checkpoint`) | 32768 | fp16 | `MEASURED` | 6345.75 | 46.97 | 21.288/43.046 | 8719.1 | 4294967296 | 12157.2229 |
| Mistral-7B-v0.1-GPTQ (32L) (`real-checkpoint`) | 32768 | fp8_e4m3fn | `MEASURED` | 6254.63 | 37.46 | 26.693/49.025 | 10691.1 | 6442450944 | 16605.7533 |
| Mistral-7B-v0.1-GPTQ (32L) (`real-checkpoint`) | 65536 | fp16 | `UNSUPPORTED` | — | — | —/— | — | — | — |
| Mixtral-8x7B-Instruct-v0.1-GPTQ (32L) (`real-checkpoint`) | 512 | fp16 | `MEASURED` | 639.7 | 33.03 | 30.27/— | 24224.6 | — | — |
| Mixtral-8x7B-Instruct-v0.1-GPTQ (32L) (`real-checkpoint`) | 2048 | fp16 | `MEASURED` | 1049.34 | 31.39 | 31.85/— | 24866.4 | — | — |
| Mixtral-8x7B-Instruct-v0.1-GPTQ (32L) (`real-checkpoint`) | 4096 | fp16 | `MEASURED` | 1842.25 | 14.65 | 68.26/— | 25887.7 | — | — |
| Mixtral-8x7B-Instruct-v0.1-GPTQ (32L) (`real-checkpoint`) | 8192 | fp16 | `MEASURED` | 3713.01 | 14.74 | 67.84/— | 27934.4 | — | — |

> **Honest Long-Context Finding (`Mistral-7B-v0.1-GPTQ`):** Coherence and perplexity remain tight at `ctx=4096` (`tail_ppl = 1.0118` FP16 / `1.0114` FP8, `32/32` FP8 vs FP16 greedy match). Above `4096` (`8K`, `16K`, `32K`), `Mistral-7B-v0.1` (`sliding_window=4096`) exhibits severe quality/coherence degradation because `fused_attention_gqa` runs unclipped global causal attention rather than enforcing a sliding-window mask, and `65536` (`64K`) is `UNSUPPORTED` (`max_position_embeddings=32768`, RoPE table=`32768`).

## Separate Synthetic Kernel-Only Attention (`phase14c_synthetic_attention`)

> Synthetic 32-layer GQA FlashDecoding microbenchmarks (`weight_source=synthetic-weights`); NOT end-to-end model validation.

| Context | KV Dtype | Median Step (ms) | Min/Max (ms) |
|---:|---|---:|---|
| 4096 | fp16 | 1.736 | 1.734 / 1.754 |
| 4096 | fp8_e4m3fn | 2.383 | 2.37 / 2.387 |
| 8192 | fp16 | 1.329 | 1.328 / 1.339 |
| 8192 | fp8_e4m3fn | 1.862 | 1.856 / 1.865 |
| 16384 | fp16 | 2.142 | 2.131 / 2.147 |
| 16384 | fp8_e4m3fn | 3.79 | 3.783 / 3.794 |
| 32768 | fp16 | 3.837 | 3.825 / 3.856 |
| 32768 | fp8_e4m3fn | 7.58 | 7.572 / 7.739 |

## Registered Architecture Matrix (`benchmarks/results_5090/phase11_architecture_parity.csv`)

| Architecture | Modality | Compatibility Status | Weight Source | Layers | Cosine Sim | Latency (ms) |
|---|---|---|---|---:|---:|---:|
| Mistral-7B-v0.1 | text | `real-checkpoint parity` | `real-checkpoint` | 4 | 0.999999 | 2.181 |
| Llama-2-70B | text | `real-checkpoint parity` | `real-checkpoint` | 2 | 1.000000 | 3.659 |
| Mixtral-8x7B-MoE | text | `real-checkpoint parity` | `real-checkpoint` | 2 | 1.000000 | 3.489 |
| Qwen2.5-0.5B | text | `real-checkpoint parity` | `real-checkpoint` | 24 | 0.999992 | 12.407 |
| SigLIP-ViT-B/16 | vision | `real-checkpoint parity` | `real-checkpoint` | 12 | 0.999986 | 2.083 |
| Whisper-Tiny-Encoder | audio | `real-checkpoint parity` | `real-checkpoint` | 4 | 0.999963 | 1.574 |
| Llama-3.1-ScaledRoPE | text | `synthetic-only` | `synthetic-weights` | 4 | 0.999999 | 3.471 |
| DeepSeek-MoE-Shared+EP4 | text | `synthetic-only` | `synthetic-weights` | 4 | 0.999999 | 12.280 |
| Qwen2-MoE-Shared+Bias | text | `synthetic-only` | `synthetic-weights` | 4 | 0.999999 | 7.807 |
| ViT-B/16 | vision | `synthetic-only` | `synthetic-weights` | 4 | 0.999999 | 0.728 |
| CLIP-ViT-L/14 | vision | `synthetic-only` | `synthetic-weights` | 4 | 1.000000 | 0.874 |
| LLaVA-1.5-Multimodal | multimodal | `synthetic-only` | `synthetic-weights` | 4 | 0.999999 | 2.188 |
| DiT-S/2-Diffusion | diffusion | `synthetic-only` | `synthetic-weights` | 6 | 0.999999 | 1.508 |
