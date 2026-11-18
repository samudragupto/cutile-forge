# CuTileForge Benchmark Results

**Hardware:** Lenovo LOQ 15APH8 | RTX 4060 Laptop GPU (AD107, sm_89, 8 GB GDDR6) | AMD Ryzen 7 7840HS | 16 GB DDR5-5600 | PCIe 4.0 x8
**Model:** Mistral-7B-v0.1 (4-bit GPTQ) | **Engine:** CuTileForge 0.1.0

## GEMM (FP16, 8 Shapes)

| shape | M | N | K | cutile_ms | cuda_ms | cublas_ms | pytorch_ms | cutile_tflops | cuda_tflops | cublas_tflops | pytorch_tflops |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1x12288x4096 | 1 | 12288 | 4096 | 0.0406 | 0.0652 | 0.0468 | 0.0693 | 2.48 | 1.54 | 2.15 | 1.45 |
| 256x12288x4096 | 256 | 12288 | 4096 | 0.2043 | 0.1962 | 0.0917 | 0.1368 | 126.15 | 131.33 | 280.89 | 188.33 |
| 1x14336x4096 | 1 | 14336 | 4096 | 0.0754 | 0.0795 | 0.0797 | 0.0764 | 1.56 | 1.48 | 1.47 | 1.54 |
| 256x14336x4096 | 256 | 14336 | 4096 | 0.2024 | 0.2445 | 0.102 | 0.1532 | 148.54 | 122.95 | 294.89 | 196.25 |
| 1x4096x14336 | 1 | 4096 | 14336 | 0.0734 | 0.2006 | 0.0767 | 0.0773 | 1.6 | 0.59 | 1.53 | 1.52 |
| 256x4096x14336 | 256 | 4096 | 14336 | 0.3724 | 0.2235 | 0.0999 | 0.1676 | 80.74 | 134.54 | 300.84 | 179.4 |
| 1x4096x4096 | 1 | 4096 | 4096 | 0.016 | 0.059 | 0.014 | 0.0365 | 2.09 | 0.57 | 2.4 | 0.92 |
| 512x4096x4096 | 512 | 4096 | 4096 | 0.1099 | 0.1256 | 0.0549 | 0.0959 | 156.39 | 136.83 | 312.86 | 179.08 |

## GEMM + INT4 Dequant (Fused vs Separate)

| shape | fused_ms | separate_ms | fused_tflops | separate_tflops | speedup |
| --- | --- | --- | --- | --- | --- |
| 1x12288x4096 | 0.0344 | 0.1594 | 2.92 | 0.63 | 4.629 |
| 256x12288x4096 | 0.2188 | 0.3028 | 117.8 | 85.11 | 1.384 |
| 1x14336x4096 | 0.0363 | 0.1856 | 3.23 | 0.63 | 5.11 |
| 256x14336x4096 | 0.2188 | 0.3661 | 137.44 | 82.11 | 1.674 |

## Fused FFN (1 Kernel vs 3 Kernels)

| seq_len | fused_ms | separate_ms | speedup |
| --- | --- | --- | --- |
| 1 | 0.059 | 0.494 | 8.375 |
| 64 | 0.1778 | 0.5148 | 2.896 |
| 256 | 0.4534 | 0.7012 | 1.547 |
| 512 | 0.9006 | 3.4877 | 3.873 |

## Grouped-Query FlashAttention-2 (32:8)

| seq_len | cutile_ms | cuda_ms | pytorch_ms | cutile_tflops | cuda_tflops | pytorch_tflops |
| --- | --- | --- | --- | --- | --- | --- |
| 64 | 0.0078 | 0.0257 | 0.0114 | 4.28 | 1.31 | 2.94 |
| 256 | 0.0138 | 0.1022 | 0.0234 | 38.79 | 5.25 | 22.95 |
| 512 | 0.0324 | 0.3514 | 0.0396 | 66.25 | 6.11 | 54.27 |
| 1024 | 0.0754 | 3.5792 | 0.0977 | 113.89 | 2.4 | 87.93 |

## LayerNorm (D=4096)

| seq_len | cutile_ms | cuda_ms | pytorch_ms | cutile_bw_gbs | cuda_bw_gbs |
| --- | --- | --- | --- | --- | --- |
| 1 | 0.0037 | 0.0078 | 0.0063 | 4.38 | 2.09 |
| 64 | 0.0037 | 0.0078 | 0.0063 | 280.07 | 133.75 |
| 256 | 0.0058 | 0.0079 | 0.0078 | 724.15 | 531.73 |
| 1024 | 0.0099 | 0.0131 | 0.0099 | 1693.98 | 1280.31 |

## SiLU Fusion (Fused-in-GEMM vs Separate)

| seq_len | fused_ms | separate_ms | speedup |
| --- | --- | --- | --- |
| 1 | 0.0365 | 0.0377 | 1.032 |
| 64 | 0.0693 | 0.0692 | 1.0 |
| 256 | 0.2311 | 0.2269 | 0.982 |
| 512 | 0.3539 | 0.358 | 1.012 |

## Full Transformer Layer (3-Kernel TFG vs 8-Kernel CUDA)

| seq_len | tfg_kernels | unfused_kernels | tfg_ms | unfused_8k_ms | pytorch_ms | speedup_vs_8k |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | 3 | 8 | 0.1225 | 0.7198 | 0.2567 | 5.877 |
| 64 | 3 | 8 | 0.3234 | 0.7633 | 0.2443 | 2.36 |
| 256 | 3 | 8 | 0.7676 | 3.419 | 0.485 | 4.454 |
| 512 | 3 | 8 | 1.2962 | 4.5301 | 0.8894 | 3.495 |

## Stage -1 PCIe Weight Streaming Overlap

| tile_mb | serial_ms | overlapped_ms | compute_ms | dma_ms | overlap_pct | speedup | pcie_gb_s | pcie_util_pct |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 32 | 1.1373 | 1.3497 | 0.4489 | 0.6884 | 0.0 | 0.843 | 49.46 | 100.0 |
| 64 | 1.8097 | 1.8121 | 0.4492 | 1.3606 | 0.0 | 0.999 | 49.46 | 100.0 |
| 115 | 2.9325 | 2.6027 | 0.4492 | 2.4833 | 73.42 | 1.127 | 49.46 | 100.0 |

## End-to-End Prefill

| prompt_tokens | prefill_ms | tokens_per_sec | peak_vram_gb |
| --- | --- | --- | --- |
| 128 | 94.51 | 1354.37 | 4.286 |
| 512 | 131.91 | 3881.37 | 4.354 |
| 2048 | 369.74 | 5539.02 | 4.662 |
| 4096 | 749.59 | 5464.35 | 5.073 |

## End-to-End Decode (256 Tokens)

| engine | prompt_tokens | generated_tokens | prefill_ms | decode_ms | tokens_per_sec | ms_per_token | peak_vram_gb | pcie_gb_s | pcie_util_pct |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| CuTileForge | 6 | 256 | 24.78 | 5317.18 | 47.96 | 20.85 | 4.289 | 46.4 | 100.0 |

## Long Context (4K-32K KV-Cache Paging)

| context_len | decode_step_ms | kv_evictions | kv_host_fetches | peak_vram_gb |
| --- | --- | --- | --- | --- |
| 4096 | 26.24 | 0 | 0 | 4.826 |
| 8192 | 33.0 | 0 | 0 | 5.363 |
| 16384 | 76.29 | 2056 | 2056 | 6.0 |
| 32768 | 214.67 | 10260 | 10260 | 6.134 |

## Batched Inference (Batch 1-16)

| batch_size | seq_len | latency_ms | tokens_per_sec | peak_vram_gb |
| --- | --- | --- | --- | --- |
| 1 | 32 | 50.83 | 629.54 | 4.285 |
| 2 | 32 | 39.77 | 1609.34 | 4.285 |
| 4 | 32 | 91.06 | 1405.62 | 4.286 |
| 8 | 32 | 102.91 | 2487.62 | 4.302 |
| 16 | 32 | 123.3 | 4152.64 | 4.354 |

## Speculative Decoding (gamma=4)

| gamma | tokens_generated | acceptance_rate | baseline_tok_s | speculative_tok_s | speedup |
| --- | --- | --- | --- | --- | --- |
| 4 | 64 | 0.206 | 40.14 | 55.43 | 1.381 |

## Sustained Thermal Run Summary

| Metric | Value |
| --- | --- |
| Samples | 361 |
| Avg tok/s | 94.69 |
| Max GPU temp | 0.0 C |
| Avg power | 0.0 W |
| Avg clock throttle | 0.00% |

## Generated Plots

- `benchmarks/results_5090/gemm_tflops.png`
- `benchmarks/results_5090/layer_latency.png`
- `benchmarks/results_5090/ffn_speedup.png`
- `benchmarks/results_5090/attention_tflops.png`
- `benchmarks/results_5090/thermal_sustained.png`
