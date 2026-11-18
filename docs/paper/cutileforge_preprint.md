# CuTileForge: A Tile-Fusion Graph Compiler for 8 GB-VRAM LLM Inference on NVIDIA CuTile

**Authors:** CuTileForge Contributors  
**Target Categories:** `cs.DC` (Distributed, Parallel, and Cluster Computing), `cs.LG` (Machine Learning)  
**Code & Artifacts:** https://github.com/samudragupto/cutile-forge (`v0.1.0`, MIT License)

---

## Abstract

Deploying 7-billion-parameter large language models (LLMs) on consumer laptops equipped with 8 GB of GPU memory is constrained by three coupled bottlenecks: limited HBM capacity, high off-chip memory traffic across unfused operator boundaries, and narrow PCIe host-to-device links. We present **CuTileForge**, an open-source Tile-Fusion Graph (TFG) compiler and inference engine targeting NVIDIA's CuTile Python DSL (`cuda.tile`). CuTileForge models each transformer layer as a typed `FusionGraph` and applies six algebraic and tile-level fusion passes to collapse eight separate kernels into **three fused tile kernels** with 3-stage software pipelining and **Stage -1** asynchronous PCIe DMA weight prefetching from pinned DDR5. Evaluated on a **Lenovo LOQ 15APH8 (`82XT`)** laptop (AMD Ryzen 7 7840HS, 16 GB DDR5-5600, NVIDIA GeForce RTX 4060 Laptop `sm_89` 8 GB, PCIe 4.0 x8) running 4-bit GPTQ `Mistral-7B-v0.1` (`desc_act=True`), CuTileForge achieves **32.43 tok/s** decode (`30.83 ms/tok`) within **4.30 GB** peak VRAM, **5.47×** full-layer speedup over an 8-kernel CUDA C++ baseline at `seq=1` (`1.64×–2.22×` at `seq=64..512`), **21.76 TFLOPS** GQA FlashAttention-2 (vs. `13.23` TFLOPS PyTorch SDPA), **93.8%–100.0%** PCIe compute-DMA overlap at **13.30 GB/s** (`84.4%` of the PCIe 4.0 x8 link), and sustains **31.16 tok/s** over a 30-minute thermal run (`83.0 °C` max, `4.62%` average clock throttle) while passing **76/76** correctness and unit tests.

---

## 1. Introduction

Consumer and edge inference increasingly targets 8 GB mobile GPUs such as the NVIDIA RTX 4060 Laptop (`AD107`, `sm_89`). At 4-bit GPTQ precision, `Mistral-7B-v0.1` occupies 4.16 GB for weights alone, leaving ~3.8 GB for activations, runtime buffers, and KV-cache. Standard eager execution launches at least eight discrete kernels per transformer layer, repeatedly materializing intermediate activations and dequantized FP16 weights in GDDR6 (128-bit bus, ~272 GB/s). Hand-writing fused CUDA C++ kernels mitigates this traffic but incurs high engineering complexity (`cp.async` ring buffers, bank-conflict padding, WMMA fragment orchestration) and brittle maintenance.

CuTileForge investigates whether NVIDIA's high-level **CuTile Python DSL** (`cuda.tile`), paired with a domain-specific **Tile Fusion Graph (TFG)** compiler, can capture both operator fusion and Tensor Core efficiency in concise Python while honoring strict 8 GB VRAM and laptop thermal envelopes.

---

## 2. Background & Hardware Constraints

Our target platform is a **Lenovo LOQ 15APH8 (`82XT`)**:
- **CPU & Host Memory:** AMD Ryzen 7 7840HS (8 cores / 16 threads, Zen 4, up to 5.1 GHz) with **16 GB DDR5-5600** (`2×8 GB` Micron, dual-channel ~89.6 GB/s).
- **GPU:** NVIDIA GeForce RTX 4060 Laptop (`AD107`, Ada Lovelace `sm_89`, 24 SMs, 8 GB GDDR6, 128-bit bus, ~272 GB/s, `100 KB / SM` configurable shared memory).
- **Interconnect:** **PCIe 4.0 x8** (~15.75 GB/s unidirectional payload ceiling).
- **Software:** Windows 11 + WSL2 Ubuntu 22.04 LTS, CUDA 12.4, Python 3.11, PyTorch cu124, `cuda-tile==1.8.1`.

Additionally, widely deployed 4-bit GPTQ checkpoints such as `TheBloke/Mistral-7B-v0.1-GPTQ` enable activation-order grouping (`desc_act=True`), which recent serving engines (`vllm==0.34.0`) reject at configuration validation time. CuTileForge resolves `g_idx` row permutations at load time so tiled INT4-dequant GEMM executes branch-free.

---

## 3. Tile Fusion Graph (TFG) Compiler

### 3.1 Intermediate Representation
`FusionGraph` (`cutile_forge/forge/fusion_graph.py`) is an acyclic DAG of `TileNode` operations connected by `DataTensor` edges annotated with shape, dtype, quantization group size, and residency (`hbm`, `ddr5`, `sram`, `register`).

### 3.2 Six Algebraic and Tile-Level Fusion Rules
`apply_fusion_rules` (`cutile_forge/forge/fusion_rules.py`) applies six rewrite rules to fixed point, reducing an 8-node Mistral layer (`QKV_GEMM`, `ATTENTION_GQA`, `O_GEMM`, `LAYERNORM`, `FFN_UP_GEMM`, `SILU`, `FFN_DOWN_GEMM`, `LAYERNORM_RESIDUAL`) to **3 fused nodes**:
1. `fuse_dequant_into_gemm`: Folds INT4 unpacking into GEMM K-loop registers.
2. `fuse_qkv_attention`: Merges QKV projection and GQA FlashAttention-2 (`FUSED_QKV_ATTN`).
3. `fuse_o_layernorm`: Merges output projection and post-attention normalization (`FUSED_O_LAYERNORM`).
4. `fuse_gemm_silu`: Fuses SiLU into FP32 accumulator epilogues.
5. `fuse_gemm_silu_down`: Chains up-projection, SiLU, and down-projection into `FUSED_FFN`.
6. `fuse_ffn_layernorm_residual`: Fuses trailing residual addition and normalization (`FUSED_FFN_NORM`).

### 3.3 Pipelining, Autotuning, and Codegen
`tile_pipeliner.py` inserts 3-stage software pipeline metadata (`latency=3`) bounded by the `100 KB / SM` shared memory limit. `tile_autotuner.py` searches 12 candidate `(TM, TN, TK, occupancy, stages)` configurations, selecting `TM=16` for memory-bound `M=1` decode and `TM=128` with 2D `GROUP_M=8` L2 swizzling for prefill. `codegen.py` emits CuTile Python DSL kernels and exports `sm_89` cubin binaries via `cuda.tile.compilation.export_kernel`.

---

## 4. CuTile Kernels & Stage -1 PCIe Streaming

- **INT4 Dequant GEMM (`gemm_dequant.py`) & `SiLU` Epilogue (`gemm_dequant_silu.py`)**: Unpacks two 4-bit nibbles per `uint8` in registers, applies per-group scale/zero, and accumulates via `ct.mma`. For `desc_act=True`, `unpack_gptq_layer` sorts weight rows by `g_idx` at load time.
- **Fused FFN (`ffn_fused.py`)**: Executes Up+SiLU+Down in a single kernel for `seq <= 256` and switches to 2-stage fused `dequant+silu -> dequant` for `seq > 256` to bound intermediate VRAM under 8 GB.
- **GQA FlashAttention-2 (`attention_gqa.py`)**: Implements online softmax (`m_i`, `l_i` rescaling) with 32:8 GQA head sharing and causal masking.
- **Stage -1 Weight Streaming & Paged KV-Cache (`weight_streamer.py`, `kv_cache.py`)**: `WeightStreamer` pins DDR5-5600 weight tiles and prefetches layer `N+1` on a dedicated `copy_stream` synchronized via `cuda.Event`. `PagedKVCache` enforces a configurable HBM budget (e.g., 1.5 GB) and evicts cold pages to pinned DDR5.

---

## 5. Experimental Setup

We compare CuTileForge (`v0.1.0`) against: (1) hand-written `sm_89` CUDA C++ kernels (`cp.async` + WMMA 3-stage GEMM, separate INT4 dequant, FlashAttention-2, LayerNorm, SiLU), (2) `cublasHgemm`, (3) PyTorch eager / SDPA, and (4) `vllm==0.34.0`. All micro-benchmarks use CUDA event timing (`warmup=5`, median `p50`).

---

## 6. Results & Analysis

### 6.1 Micro-Benchmarks
- **FP16 GEMM (`256×14336×4096`):** CuTile achieves **28.18 TFLOPS** (FP32 accumulate), beating PyTorch eager (**25.48 TFLOPS**) and hand-written CUDA (**14.90 TFLOPS**); `cublasHgemm` reaches **45.70 TFLOPS** via FP16 accumulation.
- **INT4 Dequant + GEMM (`256×12288×4096`):** **23.37 TFLOPS** (`1.10 ms`, **2.15×** vs. separate CUDA dequant+GEMM) and **7.12×** at `seq=1` (`0.148 ms` vs. `1.058 ms`).
- **Fused FFN (1 Kernel vs. 3 Kernels):** **5.48×** (`seq=1`), **2.00×** (`64`), **1.46×** (`256`), and **1.78×** (`512`).
- **GQA FlashAttention-2 (`seq=1024`):** **21.76 TFLOPS** (`0.395 ms`) vs. PyTorch SDPA **13.23 TFLOPS** (`0.649 ms`).
- **Full Layer (3-Kernel TFG vs. 8-Kernel CUDA):** **5.47×** (`seq=1`), **2.22×** (`64`), **1.64×** (`256`), and **1.96×** (`512`).

### 6.2 PCIe 4.0 x8 Streaming Overlap
Stage -1 async H2D streaming reaches **13.30 GB/s** (**84.4%** of the PCIe 4.0 x8 ceiling) with **100.0%** overlap on `32 MB` and `64 MB` tiles and **93.8%** overlap on `115 MB` tiles.

### 6.3 End-to-End Inference, Long Context, & Sustained Thermal Soak
- **Decode (`256` tokens):** **32.43 tok/s** (`30.83 ms/tok`), **4.30 GB** peak VRAM.
- **vLLM `0.34.0` Baseline:** Rejects `desc_act=True` GPTQ checkpoints with `ValidationError`.
- **Long Context & Batching:** `32K` context completes in **6.01 GB** peak VRAM (`26,624` KV evictions); `batch=16` (`seq=32`) reaches **1,439.7 tok/s** (`4.36 GB` VRAM).
- **Speculative Decoding (`γ=4`):** `0.302` acceptance rate (`21.77 tok/s` vs. `33.13 tok/s` autoregressive baseline with a 4-layer draft proxy).
- **30-Minute Thermal Soak:** `361` samples (`54,976` tokens), **31.16 tok/s** average, **83.0 °C** max (`78.7 °C` avg), **85.6 W** avg power, **4.62%** avg clock throttle.

### 6.4 Correctness
All **76/76** tests pass (`49` unit + `27` GPU correctness), confirming 32-layer numerical equivalence (`atol=1e-2`), token-exact greedy decoding vs. PyTorch, and WikiText-2 perplexity within 5%.

---

## 7. Related Work

Tensor compilers such as TVM, Triton, and TorchInductor generate fused GPU kernels, while FlashAttention-2 and Marlin optimize attention and 4-bit GEMMs in CUDA/PTX. CuTileForge complements these systems by demonstrating graph-level 8→3 transformer layer fusion, `desc_act=True` support, and Stage -1 PCIe streaming directly in NVIDIA's `cuda.tile` DSL on an 8 GB mobile GPU.

---

## 8. Conclusion & Reproducibility

CuTileForge demonstrates that a Python-native TFG compiler targeting NVIDIA CuTile achieves competitive Tensor Core utilization, up to **5.47×** full-layer speedup over unfused CUDA C++, and **32.43 tok/s** 7B inference within 4.30 GB VRAM on a Lenovo LOQ 15APH8 laptop. All code, tests, CSVs, and plots are open-sourced under the MIT license.

