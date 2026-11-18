# Social Media Drafts (Not Posted)

## 1. Hacker News (`Show HN`)

**Title:** Show HN: CuTileForge – Tile-Fusion Graph Compiler in NVIDIA CuTile DSL (32.4 tok/s on 8 GB RTX 4060)

**Body:**
Hi HN,

I built CuTileForge (`v0.1.0`), an open-source Tile Fusion Graph (TFG) compiler and inference engine for 8 GB-VRAM GPUs written on top of NVIDIA's CuTile Python DSL (`cuda.tile`).

Instead of launching 8 separate kernels per transformer layer (QKV GEMM, GQA Attention, O GEMM, LayerNorm, FFN-up, SiLU, FFN-down, LayerNorm+Residual), CuTileForge applies 6 graph fusion passes to lower each Mistral-7B layer into 3 fused CuTile kernels with 3-stage software pipelining (`latency=3`) and Stage -1 async PCIe DMA weight prefetching from pinned DDR5.

Measured on a Lenovo LOQ 15APH8 (Ryzen 7 7840HS, 16 GB DDR5-5600, RTX 4060 Laptop 8 GB, PCIe 4.0 x8) running `Mistral-7B-v0.1` 4-bit GPTQ (`desc_act=True`):
- **Decode (256 tokens):** 32.43 tok/s (30.83 ms/tok), 4.30 GB peak VRAM
- **Full Layer (3k TFG vs 8k CUDA):** 5.47x faster at `seq=1`, 2.22x at `seq=64`, 1.64x at `seq=256`, 1.96x at `seq=512`
- **INT4 Dequant+GEMM:** 23.37 TFLOPS at `256x12288x4096` (2.15x vs separate CUDA dequant+GEMM), 7.12x at `seq=1`
- **GQA FlashAttention-2 (`seq=1024`):** 21.76 TFLOPS vs PyTorch SDPA 13.23 TFLOPS
- **PCIe 4.0 x8 Overlap:** 93.8–100% compute-DMA overlap at 13.30 GB/s (84.4% of the x8 link ceiling)
- **30-min thermal soak:** 31.16 tok/s avg, 83.0°C max (78.7°C avg), 4.62% avg clock throttle across 54,976 tokens
- **GPTQ `desc_act=True`:** Supported natively via load-time `g_idx` permutation (whereas `vllm==0.34.0` rejects `desc_act=True` checkpoints with a `VllmConfig` validation error)

Repo & benchmarks: https://github.com/samudragupto/cutile-forge (MIT, 76/76 tests passing)

---

## 2. r/LocalLLaMA

**Title:** [Release] CuTileForge v0.1.0: Fusing 8 CUDA kernels into 3 Python CuTile kernels for 8 GB laptop GPUs (32.43 tok/s Mistral-7B 4-bit, 32K context in 6.01 GB)

**Body:**
Sharing **CuTileForge v0.1.0**, a Tile Fusion Graph compiler and inference engine built with NVIDIA's `cuda.tile` Python DSL, tuned specifically for 8 GB VRAM laptops.

**Hardware tested:** Lenovo LOQ 15APH8 (82XT) — Ryzen 7 7840HS, 16 GB DDR5-5600, RTX 4060 Laptop 8 GB (`sm_89`), PCIe 4.0 x8.
**Model:** `TheBloke/Mistral-7B-v0.1-GPTQ` (4-bit, `desc_act=True`, 4.16 GB).

**Highlights:**
- **32.43 tok/s** decode (256 tokens) in **4.30 GB** peak VRAM
- **32K context** in **6.01 GB** VRAM via `PagedKVCache` HBM <-> pinned DDR5 LRU eviction
- **Native `desc_act=True` support**: Tested `vllm==0.34.0` on the same checkpoint and hit `ValidationError: GPTQ group activation ordering (desc_act=True) is no longer supported`; CuTileForge permutes rows by `g_idx` at load time so the Tensor Core K-loop stays branch-free
- **8 -> 3 kernel fusion**: 5.47x layer speedup at `seq=1` vs 8 separate hand-written CUDA C++ kernels (`1.64x–2.22x` at `seq=64..512`)
- **Stage -1 PCIe streaming**: 93.8–100% overlap at 13.30 GB/s (84.4% of PCIe 4.0 x8)
- **30-min sustained run**: 54,976 tokens, 31.16 tok/s avg, 83.0°C max, 4.62% avg throttle

GitHub (MIT): https://github.com/samudragupto/cutile-forge

---

## 3. r/MachineLearning

**Title:** [P] CuTileForge: Tile-Fusion Graph Compiler Lowering Transformer Layers to NVIDIA CuTile Python DSL (8->3 Kernels, 32.43 tok/s on 8 GB RTX 4060)

**Body:**
We open-sourced **CuTileForge** (`v0.1.0`), a compiler and runtime exploring whether NVIDIA's CuTile Python DSL (`cuda.tile`) can serve as a practical compiler target for fused LLM inference on memory-constrained 8 GB GPUs.

Key technical pieces:
1. **Tile Fusion Graph (TFG) IR & 6 Fusion Rules**: Collapses an 8-op transformer layer into 3 fused kernels (`FUSED_QKV_ATTN`, `FUSED_O_LAYERNORM`, `FUSED_FFN_NORM`) with 3-stage software pipelining (`latency=3`) and a 12-config `sm_89` tile autotuner.
2. **Stage -1 PCIe DMA Prefetcher**: Double-buffered async H2D streaming from pinned DDR5-5600 over PCIe 4.0 x8, hiding 93.8–100% of transfer latency behind Tensor Core MMA.
3. **Baselines & Verification**: Hand-written CUDA C++ (`cp.async` + WMMA 3-stage GEMM, FlashAttention-2, LayerNorm, SiLU) and `cublasHgemm` pybind11 baselines, plus 76/76 passing unit and GPU correctness tests (token-exact greedy decode vs PyTorch and WikiText-2 PPL within 5%).

Repo + arXiv-style writeup + benchmark CSVs/plots: https://github.com/samudragupto/cutile-forge
