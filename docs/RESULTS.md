# CuTileForge Benchmark Results

**Hardware:** Lenovo LOQ 15APH8 | RTX 4060 Laptop GPU (AD107, sm_89, 8 GB GDDR6) | AMD Ryzen 7 7840HS | 16 GB DDR5-5600 | PCIe 4.0 x8
**Model:** Mistral-7B-v0.1 (4-bit GPTQ) | **Engine:** CuTileForge 0.1.0

## GEMM (FP16, 8 Shapes)

| shape | M | N | K | cutile_ms | cuda_ms | cublas_ms | pytorch_ms | cutile_tflops | cuda_tflops | cublas_tflops | pytorch_tflops |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1x12288x4096 | 1 | 12288 | 4096 | 0.0427 | 0.0734 | 0.0509 | 0.0693 | 2.36 | 1.37 | 1.98 | 1.45 |
| 256x12288x4096 | 256 | 12288 | 4096 | 0.2388 | 0.2134 | 0.0957 | 0.1451 | 107.91 | 120.77 | 269.2 | 177.65 |
| 1x14336x4096 | 1 | 14336 | 4096 | 0.0749 | 0.0824 | 0.0816 | 0.0767 | 1.57 | 1.43 | 1.44 | 1.53 |
| 256x14336x4096 | 256 | 14336 | 4096 | 0.2208 | 0.246 | 0.1041 | 0.1615 | 136.14 | 122.24 | 288.91 | 186.12 |
| 1x4096x14336 | 1 | 4096 | 14336 | 0.0734 | 0.2208 | 0.0773 | 0.0775 | 1.6 | 0.53 | 1.52 | 1.52 |
| 256x4096x14336 | 256 | 4096 | 14336 | 0.4113 | 0.2372 | 0.104 | 0.1839 | 73.1 | 126.74 | 289.08 | 163.47 |
| 1x4096x4096 | 1 | 4096 | 4096 | 0.0173 | 0.0651 | 0.014 | 0.0402 | 1.94 | 0.52 | 2.4 | 0.83 |
| 512x4096x4096 | 512 | 4096 | 4096 | 0.1143 | 0.1312 | 0.0549 | 0.1016 | 150.26 | 130.9 | 312.68 | 169.01 |

## GEMM + INT4 Dequant (Fused vs Separate)

| shape | fused_ms | separate_ms | fused_tflops | separate_tflops | speedup |
| --- | --- | --- | --- | --- | --- |
| 1x12288x4096 | 0.0345 | 0.1616 | 2.92 | 0.62 | 4.689 |
| 256x12288x4096 | 0.229 | 0.3109 | 112.52 | 82.88 | 1.358 |
| 1x14336x4096 | 0.0364 | 0.1876 | 3.22 | 0.63 | 5.147 |
| 256x14336x4096 | 0.2268 | 0.3662 | 132.58 | 82.09 | 1.615 |

## Fused FFN (1 Kernel vs 3 Kernels)

| seq_len | fused_ms | separate_ms | speedup |
| --- | --- | --- | --- |
| 1 | 0.0644 | 0.518 | 8.042 |
| 64 | 0.2044 | 0.5526 | 2.703 |
| 256 | 0.8699 | 0.7446 | 0.856 |
| 512 | 0.9027 | 3.5558 | 3.939 |

## Grouped-Query FlashAttention-2 (32:8)

| seq_len | cutile_ms | cuda_ms | pytorch_ms | cutile_tflops | cuda_tflops | pytorch_tflops |
| --- | --- | --- | --- | --- | --- | --- |
| 64 | 0.0087 | 0.0263 | 0.0437 | 3.85 | 1.28 | 0.77 |
| 256 | 0.0273 | 0.1123 | 0.0254 | 19.63 | 4.78 | 21.12 |
| 512 | 0.0364 | 0.3744 | 0.0423 | 59.05 | 5.74 | 50.82 |
| 1024 | 0.0845 | 3.6502 | 0.1138 | 101.62 | 2.35 | 75.51 |

## LayerNorm (D=4096)

| seq_len | cutile_ms | cuda_ms | pytorch_ms | cutile_bw_gbs | cuda_bw_gbs |
| --- | --- | --- | --- | --- | --- |
| 1 | 0.0038 | 0.0078 | 0.0078 | 4.27 | 2.09 |
| 64 | 0.0038 | 0.0082 | 0.0078 | 278.88 | 127.25 |
| 256 | 0.0058 | 0.0081 | 0.0078 | 724.15 | 518.07 |
| 1024 | 0.0118 | 0.014 | 0.0099 | 1426.63 | 1199.74 |

## SiLU Fusion (Fused-in-GEMM vs Separate)

| seq_len | fused_ms | separate_ms | speedup |
| --- | --- | --- | --- |
| 1 | 0.0376 | 0.0385 | 1.025 |
| 64 | 0.0795 | 0.0787 | 0.989 |
| 256 | 0.268 | 0.2609 | 0.973 |
| 512 | 0.4063 | 0.4008 | 0.987 |

## Full Transformer Layer (3-Kernel TFG vs 8-Kernel CUDA)

| seq_len | tfg_kernels | unfused_kernels | tfg_ms | unfused_8k_ms | pytorch_ms | speedup_vs_8k |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | 3 | 8 | 0.1334 | 0.7449 | 0.2619 | 5.584 |
| 64 | 3 | 8 | 0.336 | 0.7774 | 0.2432 | 2.313 |
| 256 | 3 | 8 | 3.4181 | 3.4068 | 0.4846 | 0.997 |
| 512 | 3 | 8 | 3.5432 | 4.4665 | 0.8859 | 1.261 |

## Stage -1 PCIe Weight Streaming Overlap

| tile_mb | serial_ms | overlapped_ms | compute_ms | dma_ms | overlap_pct | speedup | pcie_gb_s | pcie_util_pct |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 32 | 1.1583 | 1.3453 | 0.4477 | 0.7107 | 0.0 | 0.861 | 49.09 | 100.0 |
| 64 | 1.8141 | 1.8023 | 0.4475 | 1.3666 | 2.65 | 1.007 | 49.09 | 100.0 |
| 115 | 2.7956 | 2.5947 | 0.2155 | 2.5801 | 93.2 | 1.077 | 49.09 | 100.0 |

## End-to-End Prefill

| prompt_tokens | prefill_ms | tokens_per_sec | peak_vram_gb |
| --- | --- | --- | --- |
| 128 | 99.65 | 1284.45 | 4.286 |
| 512 | 142.89 | 3583.18 | 4.354 |
| 2048 | 380.86 | 5377.32 | 4.662 |
| 4096 | 726.18 | 5640.46 | 5.073 |

## End-to-End Decode (256 Tokens)

| engine | prompt_tokens | generated_tokens | prefill_ms | decode_ms | tokens_per_sec | ms_per_token | peak_vram_gb | pcie_gb_s | pcie_util_pct |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| CuTileForge | 6 | 256 | 24.85 | 5118.23 | 49.82 | 20.07 | 4.32 | 46.48 | 100.0 |

## Long Context (4K-32K KV-Cache Paging)

| context_len | decode_step_ms | kv_evictions | kv_host_fetches | peak_vram_gb |
| --- | --- | --- | --- | --- |
| 4096 | 17.96 | 0 | 0 | 4.826 |
| 8192 | 33.85 | 0 | 0 | 5.363 |
| 16384 | 107854.66 | 10240 | 8192 | 5.933 |
| 32768 | 1315.34 | 26624 | 16384 | 6.0 |

## Batched Inference (Batch 1-16)

| batch_size | seq_len | latency_ms | tokens_per_sec | peak_vram_gb |
| --- | --- | --- | --- | --- |
| 1 | 32 | 24.19 | 1322.86 | 4.285 |
| 2 | 32 | 24.49 | 2613.54 | 4.285 |
| 4 | 32 | 48.23 | 2653.81 | 4.286 |
| 8 | 32 | 51.46 | 4974.75 | 4.302 |
| 16 | 32 | 60.54 | 8457.14 | 4.354 |

## Speculative Decoding (gamma=4)

| gamma | tokens_generated | acceptance_rate | baseline_tok_s | speculative_tok_s | speedup |
| --- | --- | --- | --- | --- | --- |
| 4 | 64 | 0.302 | 95.33 | 68.74 | 0.721 |

## Generated Plots

- `benchmarks/results_5090/gemm_tflops.png`
- `benchmarks/results_5090/layer_latency.png`
- `benchmarks/results_5090/ffn_speedup.png`
- `benchmarks/results_5090/attention_tflops.png`
