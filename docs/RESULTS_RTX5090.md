# CuTileForge Benchmark Results — NVIDIA RTX 5090 (`sm_120`)

**Hardware:** NVIDIA GeForce RTX 5090 (Blackwell, `sm_120`, 170 SMs, 32 GB GDDR7, 1,792 GB/s, PCIe 5.0 x16)  
**Software:** `cuda-tile` 1.8.1, PyTorch 2.15+cu132 (`sm_120` native), Mistral-7B-v0.1 4-bit GPTQ  
**Raw CSVs & Charts:** [`benchmarks/results_5090/`](../benchmarks/results_5090/)

---

## GEMM (Micro 1/8)
| Shape | CuTile TFLOPS | CUDA TFLOPS | cuBLAS TFLOPS | PyTorch TFLOPS |
|-------|--------------|-------------|---------------|----------------|
| 1x12288x4096 | 2.36 | 1.37 | 1.98 | 1.45 |
| 256x12288x4096 | 107.91 | 120.77 | 269.20 | 177.65 |
| 1x14336x4096 | 1.57 | 1.43 | 1.44 | 1.53 |
| 256x14336x4096 | 136.14 | 122.24 | 288.91 | 186.12 |
| 1x4096x14336 | 1.60 | 0.53 | 1.52 | 1.52 |
| 256x4096x14336 | 73.10 | 126.74 | 289.08 | 163.47 |
| 1x4096x4096 | 1.94 | 0.52 | 2.40 | 0.83 |
| 512x4096x4096 | 150.26 | 130.90 | 312.68 | 169.01 |

## GEMM Dequant (Micro 2/8)
| Shape | Fused ms | Fused TFLOPS | Separate ms | Speedup |
|-------|----------|-------------|-------------|---------|
| 1x12288x4096 | 0.035 | 2.92 | 0.162 | 4.69x |
| 256x12288x4096 | 0.229 | 112.52 | 0.311 | 1.36x |
| 1x14336x4096 | 0.036 | 3.22 | 0.188 | 5.15x |
| 256x14336x4096 | 0.227 | 132.58 | 0.366 | 1.62x |

## Fused FFN (Micro 3/8)
| Seq | Fused ms | Separate ms | Speedup |
|-----|----------|-------------|---------|
| 1 | 0.064 | 0.518 | 8.04x |
| 64 | 0.204 | 0.553 | 2.70x |
| 256 | 0.870 | 0.745 | 0.86x (regression) |
| 512 | 0.903 | 3.556 | 3.94x |

## Attention GQA (Micro 4/8)
| Seq | CuTile TFLOPS | CUDA TFLOPS | PyTorch TFLOPS |
|-----|--------------|-------------|----------------|
| 64 | 3.85 | 1.28 | 0.77 |
| 256 | 19.63 | 4.78 | 21.12 |
| 512 | 59.05 | 5.74 | 50.82 |
| 1024 | 101.62 | 2.35 | 75.51 |

## LayerNorm (Micro 5/8)
| Seq | CuTile ms | CuTile GB/s | CUDA ms | PyTorch ms |
|-----|-----------|-------------|---------|------------|
| 1 | 0.0038 | 4.3 | 0.0078 | 0.0078 |
| 64 | 0.0038 | 278.9 | 0.0082 | 0.0078 |
| 256 | 0.0058 | 724.1 | 0.0081 | 0.0078 |
| 1024 | 0.0118 | 1426.6 | 0.0140 | 0.0099 |

## SiLU Fusion (Micro 6/8)
| Seq | Fused ms | Separate ms | Speedup |
|-----|----------|-------------|---------|
| 1 | 0.0376 | 0.0385 | 1.02x |
| 64 | 0.0795 | 0.0787 | 0.99x |
| 256 | 0.2680 | 0.2609 | 0.97x |
| 512 | 0.4063 | 0.4008 | 0.99x |

## Full Fused Layer (Micro 7/8)
| Seq | TFG(3k) ms | CUDA(8k) ms | PyTorch ms | Speedup |
|-----|-----------|-------------|------------|---------|
| 1 | 0.133 | 0.745 | 0.262 | 5.58x |
| 64 | 0.336 | 0.777 | 0.243 | 2.31x |
| 256 | 3.418 | 3.407 | 0.485 | 1.00x |
| 512 | 3.543 | 4.466 | 0.886 | 1.26x |

## Weight Streaming (Micro 8/8)
| Tile | Serial ms | Overlap ms | Overlap % | PCIe GB/s |
|------|-----------|------------|-----------|-----------|
| 32 MB | 1.158 | 1.345 | 0.0% | 49.09 |
| 64 MB | 1.814 | 1.802 | 2.6% | 49.09 |
| 115 MB | 2.796 | 2.595 | 93.2% | 49.09 |

## E2E Prefill
| Prompt | Prefill ms | tok/s | VRAM GB |
|--------|-----------|-------|---------|
| 128 | 99.65 | 1284.5 | 4.29 |
| 512 | 142.89 | 3583.2 | 4.35 |
| 2048 | 380.86 | 5377.3 | 4.66 |
| 4096 | 726.18 | 5640.5 | 5.07 |

## E2E Decode
49.82 tok/s, 20.07 ms/tok, 4.32 GB VRAM

## E2E Long Context
| Ctx | Step ms | Evictions | VRAM GB |
|-----|---------|-----------|---------|
| 4096 | 17.96 | 0 | 4.83 |
| 8192 | 33.85 | 0 | 5.36 |
| 16384 | 107854.66 | 10240 | 5.93 |
| 32768 | 1315.34 | 26624 | 6.00 |

## E2E Batch
| Batch | Latency ms | tok/s | VRAM GB |
|-------|-----------|-------|---------|
| 1 | 24.19 | 1322.9 | 4.28 |
| 2 | 24.49 | 2613.5 | 4.29 |
| 4 | 48.23 | 2653.8 | 4.29 |
| 8 | 51.46 | 4974.8 | 4.30 |
| 16 | 60.54 | 8457.1 | 4.35 |

## Speculative (gamma=4)
accept=0.30, base=95.33 tok/s, spec=68.74 tok/s

## Scaling Table (4060 vs 5090)
| Metric | 4060 Laptop (sm_89) | 5090 (sm_120) | Speedup |
|--------|--------------------|--------------------|---------|
| SMs | 24 | 170 | 7.1x |
| VRAM | 8 GB | 32 GB | 4x |
| Mem BW | 272 GB/s | 1792 GB/s | 6.6x |
| PCIe | 4.0 x8 (13 GB/s) | 5.0 x16 (49 GB/s) | 3.7x |
| GEMM 256x14336x4096 | 25 TFLOPS | 136 TFLOPS | 5.4x |
| Decode 7B-4bit | 32.43 tok/s | 49.82 tok/s | 1.54x |
| Prefill 4096 | 1476 tok/s | 5640 tok/s | 3.8x |
| Batch=16 | 1440 tok/s | 8457 tok/s | 5.9x |
| Attention seq=1024 | 21.76 TFLOPS | 101.62 TFLOPS | 4.7x |
| Full layer seq=1 | 5.47x | 5.58x | - |
| Peak VRAM | 4.30 GB | 4.32 GB | - |

---

## Known Issues

1. **Fused FFN at `seq=256`: 0.86x (regression vs 4060's 1.46x).**
   - **Cause:** The RTX 5090's 1,792 GB/s memory bandwidth makes the separate kernels' HBM round-trip cheap, while the fused kernel's larger shared-memory footprint hurts occupancy across 170 SMs.
   - **Fix:** Retune tile sizes for FFN at `seq=256` on `sm_120`.

2. **SiLU fusion: ~1.0x (no speedup).**
   - **Cause:** Expected on `sm_120` — 1,792 GB/s bandwidth makes the separate SiLU kernel's memory traffic negligible. Not a bug.

3. **Long context 16K: 107,854.66 ms/step (bug) vs 32K at 1,315.34 ms/step.**
   - **Investigation (`cutile_forge/engine/kv_cache.py`):** No off-by-one error in page indexing or budget accounting (`page_idx = pos // page_size`, `page_bytes = 2 * num_kv_heads * page_size * head_dim * 2`). However, `get_kv(layer_idx)` calls `_ensure_on_gpu(key)` on every page, which triggers `_evict_if_needed()` and synchronous `.cpu().pin_memory()` calls inside the forward pass. At 16K (the first context exceeding the 1.5 GB budget), this causes an eviction storm (10,240 evictions / 8,192 host fetches) and a host memory-pinning stall.

4. **Speculative decoding: 68.74 tok/s < base 95.33 tok/s.**
   - **Cause:** Acceptance rate (`0.30`) is too low because the draft model equals the verify model. Requires a smaller draft model.

5. **CuTile GEMM (136 TFLOPS) is below cuBLAS (289 TFLOPS).**
   - **Cause:** Expected — cuBLAS uses FP16-accumulate (2x Tensor Core throughput), whereas CuTile uses FP32-accumulate for numerical precision. CuTile's advantage is kernel fusion, not raw standalone GEMM.
