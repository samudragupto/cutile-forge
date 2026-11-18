# CuTileForge Benchmark Results

**Hardware:** Lenovo IdeaPad 15APH8 | RTX 4060 Laptop GPU (AD107, sm_89, 8 GB GDDR6) | AMD Ryzen 7 7735HS | 32 GB DDR5 | PCIe 4.0
**Model:** Mistral-7B-v0.1 (4-bit GPTQ) | **Engine:** CuTileForge 0.1.0

## GEMM (FP16, 8 Shapes)

| shape | M | N | K | cutile_ms | cuda_ms | cublas_ms | pytorch_ms | cutile_tflops | cuda_tflops | cublas_tflops | pytorch_tflops |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1x12288x4096 | 1 | 12288 | 4096 | 0.4127 | 0.4239 | 0.4147 | 0.4086 | 0.24 | 0.24 | 0.24 | 0.25 |
| 256x12288x4096 | 256 | 12288 | 4096 | 1.0783 | 1.7418 | 0.7076 | 1.0066 | 23.9 | 14.79 | 36.42 | 25.6 |
| 1x14336x4096 | 1 | 14336 | 4096 | 0.4813 | 0.4915 | 0.4803 | 0.4762 | 0.24 | 0.24 | 0.24 | 0.25 |
| 256x14336x4096 | 256 | 14336 | 4096 | 1.067 | 2.0173 | 0.6579 | 1.1802 | 28.18 | 14.9 | 45.7 | 25.48 |
| 1x4096x14336 | 1 | 4096 | 14336 | 0.4792 | 0.6195 | 0.4895 | 0.4844 | 0.25 | 0.19 | 0.24 | 0.24 |
| 256x4096x14336 | 256 | 4096 | 14336 | 1.4213 | 2.0726 | 0.5939 | 1.0516 | 21.15 | 14.51 | 50.62 | 28.59 |
| 1x4096x4096 | 1 | 4096 | 4096 | 0.0635 | 0.169 | 0.0707 | 0.0778 | 0.53 | 0.2 | 0.47 | 0.43 |
| 512x4096x4096 | 512 | 4096 | 4096 | 0.6759 | 0.8161 | 0.3523 | 0.6451 | 25.42 | 21.05 | 48.77 | 26.63 |

## GEMM + INT4 Dequant (Fused vs Separate)

| shape | fused_ms | separate_ms | fused_tflops | separate_tflops | speedup |
| --- | --- | --- | --- | --- | --- |
| 1x12288x4096 | 0.1485 | 1.0578 | 0.68 | 0.1 | 7.124 |
| 256x12288x4096 | 1.1028 | 2.3741 | 23.37 | 10.85 | 2.153 |
| 1x14336x4096 | 0.1956 | 1.239 | 0.6 | 0.09 | 6.335 |
| 256x14336x4096 | 1.3588 | 2.772 | 22.13 | 10.85 | 2.04 |

## Fused FFN (1 Kernel vs 3 Kernels)

| seq_len | fused_ms | separate_ms | speedup |
| --- | --- | --- | --- |
| 1 | 0.471 | 2.5825 | 5.483 |
| 64 | 1.3138 | 2.6327 | 2.004 |
| 256 | 3.8881 | 5.6581 | 1.455 |
| 512 | 5.4195 | 9.6282 | 1.777 |

## Grouped-Query FlashAttention-2 (32:8)

| seq_len | cutile_ms | cuda_ms | pytorch_ms | cutile_tflops | cuda_tflops | pytorch_tflops |
| --- | --- | --- | --- | --- | --- | --- |
| 64 | 0.042 | 0.044 | 0.1075 | 0.8 | 0.76 | 0.31 |
| 256 | 0.0451 | 0.4961 | 0.107 | 11.92 | 1.08 | 5.02 |
| 512 | 0.1208 | 1.9139 | 0.2068 | 17.77 | 1.12 | 10.38 |
| 1024 | 0.3948 | 7.4793 | 0.6492 | 21.76 | 1.15 | 13.23 |

## LayerNorm (D=4096)

| seq_len | cutile_ms | cuda_ms | pytorch_ms | cutile_bw_gbs | cuda_bw_gbs |
| --- | --- | --- | --- | --- | --- |
| 1 | 0.038 | 0.0262 | 0.0247 | 0.43 | 0.63 |
| 64 | 0.0451 | 0.0195 | 0.0257 | 23.26 | 53.89 |
| 256 | 0.0379 | 0.0215 | 0.0256 | 110.7 | 195.05 |
| 1024 | 0.043 | 0.0492 | 0.0338 | 390.1 | 341.33 |

## SiLU Fusion (Fused-in-GEMM vs Separate)

| seq_len | fused_ms | separate_ms | speedup |
| --- | --- | --- | --- |
| 1 | 0.1864 | 0.1884 | 1.011 |
| 64 | 0.3717 | 0.4127 | 1.11 |
| 256 | 1.365 | 1.3747 | 1.007 |
| 512 | 2.5866 | 2.6307 | 1.017 |

## Full Transformer Layer (3-Kernel TFG vs 8-Kernel CUDA)

| seq_len | tfg_kernels | unfused_kernels | tfg_ms | unfused_8k_ms | pytorch_ms | speedup_vs_8k |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | 3 | 8 | 0.638 | 3.4898 | 1.3578 | 5.47 |
| 64 | 3 | 8 | 1.6323 | 3.6168 | 1.4602 | 2.216 |
| 256 | 3 | 8 | 4.9342 | 8.1014 | 2.9379 | 1.642 |
| 512 | 3 | 8 | 7.5843 | 14.8828 | 5.8819 | 1.962 |

## Stage -1 PCIe Weight Streaming Overlap

| tile_mb | serial_ms | overlapped_ms | compute_ms | dma_ms | overlap_pct | speedup | pcie_gb_s | pcie_util_pct |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 32 | 3.695 | 2.5907 | 1.0941 | 2.6009 | 100.0 | 1.426 | 13.3 | 84.43 |
| 64 | 6.1862 | 5.0436 | 1.1009 | 5.0853 | 100.0 | 1.227 | 13.3 | 84.43 |
| 115 | 10.175 | 9.145 | 1.0982 | 9.0768 | 93.79 | 1.113 | 13.3 | 84.43 |

## End-to-End Prefill

| prompt_tokens | prefill_ms | tokens_per_sec | peak_vram_gb |
| --- | --- | --- | --- |
| 128 | 146.72 | 872.44 | 4.285 |
| 512 | 357.41 | 1432.52 | 4.362 |
| 2048 | 1366.22 | 1499.02 | 4.671 |
| 4096 | 2775.7 | 1475.66 | 5.082 |

## End-to-End Decode (256 Tokens)

| engine | prompt_tokens | generated_tokens | prefill_ms | decode_ms | tokens_per_sec | ms_per_token | peak_vram_gb | pcie_gb_s | pcie_util_pct |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| CuTileForge | 6 | 256 | 37.58 | 7862.88 | 32.43 | 30.83 | 4.296 | 13.41 | 85.12 |

## vLLM Baseline (256 Tokens)

| engine | tok_per_sec | ms_per_token | peak_vram_gb | tokens | notes |
| --- | --- | --- | --- | --- | --- |
| vllm-0.34.0 | N/A | N/A | N/A | 0 | ValidationError: GPTQ desc_act=True unsupported in vLLM 0.34.0 (CuTileForge supports desc_act=True natively) |

## Long Context (4K-32K KV-Cache Paging)

| context_len | decode_step_ms | kv_evictions | kv_host_fetches | peak_vram_gb |
| --- | --- | --- | --- | --- |
| 4096 | 73.67 | 0 | 0 | 4.817 |
| 8192 | 234.01 | 0 | 0 | 5.371 |
| 16384 | 1862.31 | 10240 | 8192 | 5.942 |
| 32768 | 3874.85 | 26624 | 16384 | 6.009 |

## Batched Inference (Batch 1-16)

| batch_size | seq_len | latency_ms | tokens_per_sec | peak_vram_gb |
| --- | --- | --- | --- | --- |
| 1 | 32 | 67.05 | 477.27 | 4.266 |
| 2 | 32 | 64.45 | 992.98 | 4.273 |
| 4 | 32 | 119.24 | 1073.48 | 4.287 |
| 8 | 32 | 196.13 | 1305.28 | 4.311 |
| 16 | 32 | 355.63 | 1439.69 | 4.362 |

## Speculative Decoding (gamma=4)

| gamma | tokens_generated | acceptance_rate | baseline_tok_s | speculative_tok_s | speedup |
| --- | --- | --- | --- | --- | --- |
| 4 | 64 | 0.302 | 33.13 | 21.77 | 0.657 |

## Sustained Thermal Run Summary

| Metric | Value |
| --- | --- |
| Samples | 361 |
| Avg tok/s | 31.16 |
| Max GPU temp | 83.0 C |
| Avg power | 85.6 W |
| Avg clock throttle | 4.62% |

## Generated Plots

- `benchmarks/results/gemm_tflops.png`
- `benchmarks/results/layer_latency.png`
- `benchmarks/results/ffn_speedup.png`
- `benchmarks/results/attention_tflops.png`
- `benchmarks/results/thermal_sustained.png`
