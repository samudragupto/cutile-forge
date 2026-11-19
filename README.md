# CuTileForge (`v1.0.0`)

### *Tile-Fusion Graph Compiler, Hybrid PCIe-Streaming Runtime & `torch.compile` Backend on NVIDIA CuTile (`sm_89` Ada & `sm_120` Blackwell)*

[![CI](https://github.com/samudragupto/cutile-forge/actions/workflows/ci.yml/badge.svg)](https://github.com/samudragupto/cutile-forge/actions/workflows/ci.yml)
[![Release](https://img.shields.io/github/v/release/samudragupto/cutile-forge?color=1b365d)](https://github.com/samudragupto/cutile-forge/releases/tag/v1.0.0)
[![License: MIT](https://img.shields.io/badge/License-MIT-6c757d.svg)](LICENSE)
[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue)](pyproject.toml)
[![CUDA Targets](https://img.shields.io/badge/CUDA-sm__89%20%7C%20sm__120-76B900)](https://developer.nvidia.com/cuda-toolkit)
[![GHCR](https://img.shields.io/badge/ghcr-cutile--forge-blue)](https://github.com/samudragupto/cutile-forge/pkgs/container/cutile-forge)
[![Tests](https://img.shields.io/badge/tests-102%2F102%20passed%20%7C%200%20skipped-16a34a)](tests/)

**Authors:** [Arrhat Nag](https://github.com/samudragupto) (`@samudragupto`), Rupesh Nasre

<p align="center">
  <a href="#abstract">Abstract</a> ·
  <a href="#pillar-i--systems-architecture--compiler-internals">Pillar I: Architecture</a> ·
  <a href="#pillar-ii--research-evaluation--empirical-benchmarks">Pillar II: Research &amp; Benchmarks</a> ·
  <a href="#pillar-iii--developer-guide-api-reference--reproducibility">Pillar III: Developer Guide</a> ·
  <a href="#35-citation--license">Citation</a>
</p>

---

## Abstract

Deploying 7B–70B large language models across heterogeneous consumer GPUs—from memory-constrained **8 GB laptop GPUs (`sm_89`)** to **32 GB Blackwell workstations (`sm_120`)**—exposes a fundamental trade-off between kernel engineering productivity and hardware efficiency. Hand-written CUDA C++/PTX kernels (`cp.async`, WMMA, custom quantized dequantization) require hundreds of lines per operator and break across architectures, whereas eager PyTorch dispatch suffers severe HBM traffic amplification and CPU launch overhead. **CuTileForge** introduces a **Tile-Fusion Graph (TFG)** compiler and inference runtime lowered directly to NVIDIA's **CuTile Python DSL (`cuda.tile`)**. By applying six algebraic and tile-level fusion passes, CuTileForge collapses 8 separate transformer kernels into 3 pipelined tile kernels, combines an **Occupancy-Aware Cost Model** with **Stage -1 PCIe DMA Weight Streaming**, and exposes a zero-code-change **`torch.compile(backend="cutile_forge")`** FX/Dynamo backend. Across real 4-bit GPTQ checkpoints, CuTileForge achieves **32.43 tok/s** on an 8 GB RTX 4060 Laptop, **96.57–117.31 tok/s** on an RTX 5090 (`99.73 tok/s` 30-min sustained), **41.83 tok/s** on resident **Mixtral-8x7B**, and **4.07 tok/s** on **Llama-2-70B** inside 32 GB VRAM (**80.8% of the physical PCIe 5.0 bus ceiling**).

<p align="center">
  <img src="docs/images_5090/fig1_executive_overview.png" alt="Figure 1: CuTileForge v1.0.0 Executive Overview" width="100%"/>
</p>

### Key Findings

1. **Portable 8→3 Kernel Fusion (`sm_89` ↔ `sm_120`):** The identical TFG compiler and 6 fusion rules achieve **5.47×** full-layer speedup on RTX 4060 Laptop (`seq=1`) and **5.88×** (`seq=1`) / **4.45×** (`seq=256`) on RTX 5090 over 8-kernel CUDA C++.
2. **Real 7B, 47B MoE, and 70B Execution on a Single 32 GB GPU:** Evaluates real trained TheBloke GPTQ checkpoints (`Mistral-7B` at **96.57 tok/s** in `4.32 GB`; `Mixtral-8x7B-Instruct` 100% resident at **41.83 tok/s**, `PPL=7.48`, `25.26 GB`; `Llama-2-70B-Chat` 80-layer hybrid `55 HBM + 25 PCIe 5.0` at **4.07 tok/s** in `27.44 GB` peak VRAM, zero OOM).
3. **1,822× Long-Context Paging Speedup (`107,854 ms → 59.20 ms` at 16K):** Pre-allocated async pinned CPU page pools (`swap_stream`) combined with 16-split **FlashDecoding** eliminate synchronous `cudaHostAlloc` stalls and saturate all 170 Blackwell SMs.
4. **3.20× Concurrent Tail-Latency Reduction (`363.1 ms → 113.6 ms` p99):** Async `DynamicBatcher` (`max_batch_size=16`, `8 ms` window) with `BatchedKVCache` and `RadixPrefixCache` serves OpenAI-compatible `/v1/chat/completions` + SSE streaming.
5. **Zero-Code-Change `torch.compile(backend="cutile_forge")`:** Registered Dynamo/FX backend lowers `linear+silu+linear`, `add+layer_norm`, and `scaled_dot_product_attention` into fused CuTile kernels (`atol < 1e-2`), verified across **102 / 102 passing tests (0 skipped)**.

---

## Pillar I — Systems Architecture & Compiler Internals

### 1.1 End-to-End TFG Compiler & Runtime Pipeline

```mermaid
flowchart TD
    W["4-bit GPTQ / AWQ / GGUF Checkpoints (Multi-Shard mmap)"] --> L["WeightLoader (desc_act g_idx & group_size=-1)"]
    L --> DDR5["HybridLayerBudgeter: Resident HBM + Pinned Host DDR5"]
    DDR5 -->|"Stage -1: Async H2D Copy Stream"| WS["WeightStreamer (Double-Buffered PCIe 4.0/5.0 DMA)"]

    G["PyTorch FX / TFG IR (8 Unfused Nodes)"] --> R{"Apply 6 Fusion Rules"}
    R --> F["3 Fused CuTile Nodes (FUSED_QKV_ATTN, FUSED_O_LAYERNORM, FUSED_FFN_NORM)"]
    F --> P["Occupancy Cost Model + Tile Pipeliner (3/4-Stage)"]
    P --> C["CuTile Python DSL (cuda.tile -> SASS sm_89 / sm_120)"]

    WS --> EXEC["GPU Execution (CUDA Graph / Persistent Grouped MoE / Split-KV)"]
    C --> EXEC
    KV["PagedKVCache (FP16/FP8 HBM <-> Async Pinned Pool) + RadixPrefixCache"] <--> EXEC
    EXEC --> OUT["OpenAI Server (DynamicBatcher + SSE) & Speculative (gamma=4)"]
```

### 1.2 Algebraic & Tile-Level Fusion Passes (`8 → 3` Kernels)

```mermaid
flowchart TB
    subgraph Unfused["Before Fusion: 8 Separate Kernels"]
        direction TB
        U1["QKV GEMM"] --> U2["GQA Attention"]
        U2 --> U3["O GEMM"]
        U3 --> U4["LayerNorm"]
        U4 --> U5["FFN-up GEMM"]
        U5 --> U6["SiLU"]
        U6 --> U7["FFN-down GEMM"]
        U7 --> U8["LayerNorm + Residual"]
    end

    subgraph Fused["After TFG Passes: 3 Fused CuTile Kernels"]
        direction TB
        F1["1. FUSED_QKV_ATTN (QKV + RoPE + FlashAttn-2)"]
        F2["2. FUSED_O_LAYERNORM (O-proj + Norm + Residual)"]
        F3["3. FUSED_FFN_NORM (Up + SiLU + Down + Norm)"]
        F1 --> F2 --> F3
    end

    Unfused -->|"6 TFG Fusion Passes"| Fused
```

CuTileForge's `cutile_forge/forge/fusion_rules.py` applies six deterministic graph rewrites over the typed `FusionGraph` IR:

| Rule | Pattern | Fused Output | Eliminated HBM Round-Trip |
|---|---|---|---|
| **R1: `QKVPackRule`** | `Q_proj + K_proj + V_proj` | `FUSED_QKV_PROJ` | 2 redundant input reads |
| **R2: `RoPEAttnRule`** | `QKV + RoPE + GQA_Attn` | `FUSED_QKV_ATTN` | Full `Q/K/V` + RoPE materialization |
| **R3: `EpilogueNormRule`** | `O_proj + Residual + RMSNorm` | `FUSED_O_LAYERNORM` | `D_model` post-attention write/read |
| **R4: `SwiGLURule`** | `Gate_GEMM + Up_GEMM + SiLU` | `FUSED_GEMM_SILU` | `D_ffn` intermediate activation (`14336`) |
| **R5: `FFNCollapseRule`** | `FUSED_GEMM_SILU + Down_GEMM` | `FUSED_FFN_3OP` *(or 2-Stage if `M>=128`)* | `D_ffn` intermediate buffer (when launch-bound) |
| **R6: `PostFFNNormRule`** | `FFN + Residual + RMSNorm` | `FUSED_FFN_NORM` | `D_model` post-FFN write/read |

### 1.3 Stage -1 PCIe DMA Weight Streamer & Hybrid HBM/DDR5 Budgeter

When a model's quantized weights exceed GPU VRAM (e.g., `Llama-2-70B-Chat-GPTQ` at `35.3 GB` on a `32 GB` RTX 5090), `HybridLayerBudgeter` pins the maximum safe prefix of transformer layers in HBM (`55` layers = `24.13 GB`) and stages the remaining tail (`25` layers = `9.74 GB`) in page-locked (`cudaHostAlloc`) host DDR5 ring buffers. During execution, **Stage -1** overlaps PCIe H2D DMA on a dedicated `copy_stream` with CuTile compute on `compute_stream` via double-buffered slots (`Slot A` / `Slot B`) and fine-grained `torch.cuda.Event` barriers:

```mermaid
sequenceDiagram
    participant CPU as "Pinned Host DDR5 (25 Streamed Layers)"
    participant DMA as "Copy Stream (PCIe 5.0 x16 - 49.09 GB/s)"
    participant GPU as "Compute Stream (170 SMs - CuTile Kernels)"

    Note over GPU: "Layers 0..54 execute 100% HBM-resident"
    CPU->>DMA: "Prefetch Layer 55 into Slot A (non_blocking=True)"
    GPU->>GPU: "Compute Layer 54 (HBM)"
    DMA-->>GPU: "Event sync (Slot A ready)"
    par Compute Layer 55 & Prefetch Layer 56
        GPU->>GPU: "Compute Layer 55 (Slot A)"
        CPU->>DMA: "Prefetch Layer 56 into Slot B"
    end
    DMA-->>GPU: "Event sync (Slot B ready)"
```

### 1.4 Formal Cost Models & Analytical Bounds

**Equation 1 — HBM Traffic Elimination via Epilogue-to-Prologue Fusion:**
Collapsing the 8 unfused operators of a transformer block into 3 CuTile kernels eliminates intermediate activation materialization across QKV projections, attention output, FFN gate/up activations, and post-attention residuals:

$$\Delta B_{\text{HBM}} = 2 \cdot M \cdot \left(3 D_{\text{model}} + D_{\text{ffn}} + 2 D_{\text{model}}\right) \cdot \text{sizeof}(\text{FP16})$$

**Equation 2 — Occupancy-Guided FFN Split Criterion (`select_ffn_execution_strategy`):**
Let $M$ denote active tokens, $N_{\text{SM}}$ the multiprocessor count, and $BW_{\text{HBM}}$ the HBM bandwidth:

$$\text{Strategy}(M, N_{\text{SM}}, BW_{\text{HBM}}) = \begin{cases} \text{Fused-3Op-SingleKernel } (128\times 128\times 32), & M \le 64 \text{ (Launch-Bound)} \\ \text{Pipelined-2Stage-HighOcc } (64\times 64\times 32), & M \ge 128 \land N_{\text{SM}} \ge 100 \text{ (Occupancy-Bound)} \end{cases}$$

When $M \le 64$, kernel launch latency dominates and the single-kernel 3-op FFN delivers **8.04×–8.38×** speedup (`seq=1`). When $M \ge 128$ on high-SM Blackwell (`N_SM = 170`, `1,792 GB/s`), the 3-op kernel's register footprint throttles active CTAs per SM (`0.86×` regression at `seq=256`); splitting into `FUSED_GEMM_UP_SILU` → `GEMM_DOWN` with `(64, 64, 32)` tiles restores SM occupancy to `75%` and lifts `seq=256` from **`0.86×` → `1.55×`** (and full-layer `seq=256` from **`1.00×` → `4.45×`**).

**Equation 3 — Hybrid HBM/DDR5 Layer Partitioning & PCIe 5.0 Theoretical Bound:**
Given total layers $L$, per-layer quantized footprint $B_{\text{layer}}$, HBM capacity $V_{\text{HBM}}$, safety factor $\alpha = 0.85$, KV/activation reserves $V_{\text{KV}} + V_{\text{act}}$, and measured PCIe H2D bandwidth $BW_{\text{PCIe}}$:

$$N_{\text{resident}} = \left\lfloor \frac{\alpha \cdot V_{\text{HBM}} - V_{\text{KV}} - V_{\text{act}}}{B_{\text{layer}}} \right\rfloor, \qquad T_{\text{step}} \ge \frac{(L - N_{\text{resident}}) \cdot B_{\text{layer}}}{BW_{\text{PCIe}}} \cdot (1 - \eta_{\text{overlap}}) + T_{\text{compute}}$$

Plugging in **Real `Llama-2-70B-Chat-GPTQ`** on the 32 GB RTX 5090 ($L = 80$, $N_{\text{resident}} = 55$ layers = `24.13 GB` HBM, $N_{\text{streamed}} = 25$ layers = `9.74 GB` pinned DDR5): streaming `9.74 GB` across PCIe 5.0 x16 (`49.09 GB/s`) imposes a **physical PCIe 5.0 bus floor of `198.4 ms/tok` (`5.04 tok/s`)**. CuTileForge achieves **`245.51 ms/tok` (`4.07 tok/s`)** with **`88.1%`** compute-DMA overlap—operating at **80.8% of the physical PCIe 5.0 hardware limit** with zero OOM (`27.44 GB` peak VRAM).

### 1.5 Side-by-Side Kernel Ergonomics (`cuda.tile` Python DSL vs. CUDA C++/WMMA)

CuTileForge captures Tensor Core efficiency and fused epilogues in concise Python rather than brittle 150+ line CUDA C++ templates:

<table>
<tr>
<th>CuTileForge Python DSL (<code>cuda.tile</code>, ~18 lines)</th>
<th>Hand-Written CUDA C++ / WMMA (150+ lines)</th>
</tr>
<tr>
<td>

```python
@ct.kernel
def fused_gemm_silu_kernel(A, B, C, tm: ConstInt, tn: ConstInt, tk: ConstInt):
    pid_m, pid_n = ct.bid(0), ct.bid(1)
    acc = ct.zeros((tm, tn), dtype=ct.float32)
    for k in range(0, A.shape[1], tk):
        a_tile = ct.load(A, index=(pid_m, k // tk), shape=(tm, tk))
        b_tile = ct.load(B, index=(k // tk, pid_n), shape=(tk, tn))
        acc = ct.mma(a_tile, b_tile, acc)
    # Fused SiLU epilogue in registers (zero HBM round-trip)
    x = acc.astype(ct.float16)
    out = x * (1.0 / (1.0 + ct.exp(-x)))
    ct.store(C, index=(pid_m, pid_n), tile=out)
```

</td>
<td>

```cpp
// Requires explicit cp.async ring buffers, smem bank-conflict swizzling,
// nvcuda::wmma::fragment<matrix_a/b/accumulator>, manual barrier parity,
// and separate epilogue shared-memory staging to apply SiLU before st.global:
template <int BM, int BN, int BK, int STAGES>
__global__ void wmma_gemm_silu_cp_async(...) {
    extern __shared__ half smem[];
    // ~40 lines: cp.async.cg.shared.global 3-stage prologue
    // ~50 lines: wmma::load_matrix_sync + mma_sync mainloop
    // ~45 lines: smem epilogue store + __syncthreads() + SiLU + global store
    // Breaks when porting sm_89 -> sm_120 or changing tile shapes!
}
```

</td>
</tr>
</table>

---

## Pillar II — Research Evaluation & Empirical Benchmarks

### 2.1 Dual-Hardware Experimental Testbeds

All timings use CUDA events (`torch.cuda.Event(enable_timing=True)`) with warmup (`5–10` iterations) and median (`p50`) / `p99` reporting over `50–200` iterations; raw CSVs live in [`benchmarks/results/`](benchmarks/results/) and [`benchmarks/results_5090/`](benchmarks/results_5090/).

| Specification | Testbed A: Edge Laptop (`sm_89`) | Testbed B: Workstation (`sm_120`) | Ratio |
|---|---|---|---|
| **Platform** | Lenovo LOQ 15APH8 (`82XT`) | Linux Desktop Workstation | — |
| **GPU / Die** | RTX 4060 Laptop (`AD107`, Ada Lovelace) | RTX 5090 (`GB202`, Blackwell) | — |
| **SMs / Tensor Cores** | 24 SMs / 96 (4th Gen) | 170 SMs / 680 (5th Gen) | **7.08×** |
| **VRAM & Bandwidth** | 8 GB GDDR6 · `272 GB/s` | 32 GB GDDR7 · `1,792 GB/s` | **4.0× / 6.59×** |
| **Host CPU & System RAM** | AMD Ryzen 7 7840HS (8C/16T) · 16 GB DDR5-5600 | Intel Core Ultra 9 285K (24C/24T) · 62 GB DDR5 | — |
| **PCIe Link (Measured H2D)** | PCIe 4.0 x8 (`13.30 GB/s`, `84.4%` bus util) | PCIe 5.0 x16 (`49.09 GB/s`) | **3.69×** |
| **Software Stack** | Windows 11 · CUDA 12.x · `cuda-tile` | Ubuntu Linux · CUDA 13.2 (595.58.03) · PyTorch 2.15+cu132 · `cuda-tile 1.8.1` | — |

### 2.2 Real Large-Model Checkpoint Evaluation (`Mistral-7B`, `Mixtral-8x7B`, `Llama-2-70B`)

<p align="center">
  <img src="docs/images_5090/real_models_v100.png" alt="Real 4-Bit GPTQ Checkpoint Evaluation on RTX 5090" width="100%"/>
</p>

Evaluated real trained 4-bit GPTQ checkpoints (`TheBloke/Mistral-7B-v0.1-GPTQ`, `TheBloke/Mixtral-8x7B-Instruct-v0.1-GPTQ`, `TheBloke/Llama-2-70B-Chat-GPTQ`) alongside synthetic architecture-matched capacity runs ([`real_large_models.csv`](benchmarks/results_5090/real_large_models.csv), [`hybrid_70b.csv`](benchmarks/results_5090/hybrid_70b.csv)):

| Model / Mode | Placement (HBM / PCIe) | Prefill (`tok/s`) | Decode (`tok/s`) | Latency (`ms/tok`) | Peak VRAM | Host RSS | PCIe Overlap | Perplexity |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| **Mistral-7B-v0.1-GPTQ (Real, `sm_89` 4060)** | 32 / 0 | 1,476.0 | **32.43** | 30.83 | **4.30 GB** | — | 0% | WikiText-2 ≤5% |
| **Mistral-7B-v0.1-GPTQ (Real, `sm_120` Graph)** | 32 / 0 | **5,640.5** | **96.57** (`117.31` spec) | **10.36** | **4.29 GB** | — | 0% | Exact match |
| **Mixtral-8x7B (Synthetic `v0.7.0`)** | 32 / 0 | — | — (`42.13 TF` @ M=256) | `0.218 ms`/L | — | — | 0% | N/A |
| **Mixtral-8x7B-Instruct-GPTQ (Real, 23.8 GB)** | **32 / 0** | **2,786.7** | **41.83** | **23.90** | **25.26 GB** | 1.93 GB | 0% (100% HBM) | **7.48** *(ctx=44)* |
| **LLAMA3_70B_4BIT (Synthetic `v0.8.0`)** | 57 / 23 | — | 4.66 | 214.40 | 27.43 GB | — | 99.5% | N/A |
| **Llama-2-70B-Chat-GPTQ (Real, Prompt 1)** | **55 / 25** | **859.2** | **3.90** | **256.23** | **27.44 GB** | 9.07 GB | **84.4%** | — |
| **Llama-2-70B-Chat-GPTQ (Real, Prompt 2)** | **55 / 25** | 15.2 (`1,053 ms`) | **4.07** | **245.51** | **27.44 GB** | 9.08 GB | **88.1%** | — |

*Footnote: Synthetic rows = architecture-matched random weights (capacity/streaming validation); Real rows = TheBloke GPTQ checkpoints, greedy decode (`temperature=0.0`), coherent English verified. Mixtral perplexity (`7.48`) is teacher-forced cross-entropy over a 44-token GPU shared-memory technical passage (`benchmarks/e2e/bench_real_large_models.py`).*

> **Verbatim Mixtral-8x7B-Instruct-v0.1-GPTQ (`[INST] Explain how GPU shared memory bank conflicts occur and how tile swizzling fixes them. [/INST]`):**  
> *"A GPU (Graphics Processing Unit) consists of many small processing cores called CUDA cores, which can execute instructions in parallel. Each CUDA core has its own set of registers and a small amount of fast memory called shared memory. Shared memory is a user-managed cache that is physically close"*
>
> **Verbatim Llama-2-70B-Chat-GPTQ (`The capital of France is`, 80-layer Hybrid 55 HBM + 25 PCIe 5.0):**  
> *"Paris"*

### 2.3 Empirical Roofline Analysis (`sm_89` Ada vs. `sm_120` Blackwell)

<p align="center">
  <img src="docs/images_5090/roofline_ada_vs_blackwell.png" alt="Figure 2: Empirical Ada vs Blackwell Log-Log Roofline" width="95%"/>
</p>

| Kernel | Arch | Arithmetic Intensity (`FLOPs/B`) | Achieved (`TFLOPS`) | `% of CuTile FP32 Ceiling` | Bound Regime |
|---|---|---:|---:|---:|---|
| Decode INT4 GEMV (`1×14336×4096`) | `sm_120` | 2.0 | **3.22** | 90% of `1,792 GB/s` slope | Memory-Bound |
| Persistent 3D-Grid Grouped MoE (`M=256`) | `sm_120` | 37.5 | **42.13** (`68.4×` vs v0.4.0) | 63% of BW slope | Memory-Bound |
| GQA FlashAttention-2 (`seq=1024`) | `sm_120` / `sm_89` | 204.8 | **101.62** / **21.76** | 67.6% / 77.2% | Compute-Bound |
| NVFP4 (`e2m1`) Register-LUT GEMM | `sm_120` | 410.0 | **88.56** | 58.9% (LUT ALU-limited) | Instruction-Bound |
| Fused INT4 Dequant + GEMM (`256×14336`) | `sm_120` / `sm_89` | 410.0 | **132.58** / **23.37** | **88.2%** / **82.9%** | Compute-Bound |
| FP16 Prefill GEMM (`512×4096×4096`) | `sm_120` | 341.0 | **150.26** (`156.39` v0.3.0) | **100.0%** | Compute-Bound |

### 2.4 Kernel Micro-Benchmarks & Precision Trade-offs

<p align="center">
  <img src="docs/images_5090/fig3_microbenchmark_suite.png" alt="Figure 3: Micro-Benchmark Suite on RTX 5090" width="100%"/>
</p>

- **Precision Trade-off (`INT4` vs. `NVFP4 e2m1`):** `fused_gemm_nvfp4_dequant` (`88.56 TFLOPS`) trails `fused_gemm_dequant` (`132.58 TFLOPS`) because decoding 4-bit `e2m1` values in CuTile registers via `exp2`/branchless LUT arithmetic executes more ALU instructions per tile than integer bit-shift unpacking (`(qw >> 4) & 0x0F`). This gap is expected to close once `cuda.tile` exposes native Blackwell `tcgen05` FP4 MMA instructions.
- **FP32 vs. FP16 Accumulate:** CuTile GEMM (`150.26 TFLOPS`) uses FP32 accumulation for numerical stability across 32–80 layers, whereas cuBLAS (`312.68 TFLOPS`) uses FP16 accumulation (2× Blackwell Tensor Core rate). CuTile's advantage is kernel fusion (**5.15×** INT4 decode, **5.88×** full layer, **1.35×** over PyTorch SDPA at `seq=1024`).

### 2.5 Systems Ablation Studies & 30-Minute Sustained Telemetry

<p align="center">
  <img src="docs/images_5090/fig4_systems_telemetry.png" alt="Figure 4: 30-Min Sustained Telemetry and Stage -1 PCIe Overlap" width="100%"/>
</p>

#### Master Release Evolution (`v0.1.0` → `v1.0.0`)

| Release | Architectural Milestone | Key Verified Metric (`sm_120` / `sm_89`) | Tests |
|---|---|---|---:|
| **`v0.1.0`** | TFG Compiler (8→3), 3-Stage Pipeliner, Stage -1 DMA | `32.43 tok/s` Mistral-7B in `4.30 GB` VRAM (RTX 4060 Laptop, `5.47×` layer fusion) | `76/76` |
| **`v0.2.0`** | Blackwell `sm_120` Port + FP8 GEMM | `49.82 tok/s` eager decode, `136.14 TF` GEMM, `49.09 GB/s` PCIe 5.0 | `49` CI |
| **`v0.3.0`** | Async Pinned KV Pool, Occupancy FFN, CUDA Graph, N-Gram Spec | `96.57 tok/s` Graph decode, `117.31 tok/s` spec (`1.21×`), `16K` ctx `107,854 → 76.29 ms`, FFN `seq=256` `0.86× → 1.55×` | `53` CI |
| **`v0.4.0`** | FP8 KV-Cache, NVFP4 (`e2m1`), MoE 8×7B, AWQ/GGUF | `50%` KV VRAM reduction, `88.56 TF` NVFP4, AWQ & GGUF `Q4_K_M` loaders | `57` CI |
| **`v0.5.0`** | OpenAI FastAPI Server + SSE, Roofline, Wheel, Docker | `33` SSE chunks (`859.7 ms`), `99.73 tok/s` 30-min sustained (`0.25%` drift) | `87/87` |
| **`v0.6.0`** | `DynamicBatcher` + `RadixPrefixCache` | 4-way concurrent p99 **`363.1 → 113.6 ms`** (**3.20×** reduction) | `90/90` |
| **`v0.7.0`** | Persistent 3D-Grid Grouped MoE + FlashDecoding Split-KV | Grouped MoE **`42.13 TF`** @ `M=256` (**68.4×**); `16K` ctx **`59.20 ms`** (**1,822×** vs v0.2.0) | `92/92` |
| **`v0.8.0`** | `LLAMA3_70B_4BIT` + `HybridLayerBudgeter` | 80-layer 70B synthetic pass: `27.43 GB` VRAM, `99.5%` overlap, `4.66 tok/s` | `95/95` |
| **`v1.0.0`** | `torch.compile(backend="cutile_forge")` + Real 47B/70B | Real **Mixtral-8x7B** (`41.83 tok/s`, PPL `7.48`) & **Llama-2-70B** (`4.07 tok/s`, `27.44 GB`, zero OOM) | **`102/102`** |

#### Known Issues & Honest Notes

- **MoE `M=1` Decode Headroom:** Real Mixtral-8x7B decode (`41.83 tok/s`) uses per-expert gather; the persistent 3D-grid grouped kernel (`42.13 TFLOPS` at `M=256`) is tuned for `M >= 64`, making fused `M=1` top-2 dispatch the primary next optimization target.
- **Speculative Denominators:** `117.31 tok/s` represents **1.71×** vs. `v0.2.0` speculative (`68.74 tok/s`) and **1.21×** vs. `v0.3.0` CUDA-graph non-speculative decode (`96.57 tok/s`).
- **Container NVML Telemetry:** On the Linux workstation, NVML sensors returned `0.0 °C / 0.0 W` inside the container; 30-minute sustained throughput stability (`361` samples, `179,264` tokens) showed `0.25%` drift (`100.06 → 99.81 tok/s`).
- **Ephemeral Model Storage:** Real 47B/70B checkpoints in `/var/tmp/arrhat_models` are ephemeral; use persistent storage (e.g., `/mnt/shared_storage` when writable) via `STORAGE_ROOT` or `--mixtral` / `--llama70b`.

---

## Pillar III — Developer Guide, API Reference & Reproducibility

### 3.1 60-Second Installation & Quickstart

```sh
# Option A: Pull prebuilt CUDA 13.2 container from GitHub Container Registry
docker pull ghcr.io/samudragupto/cutile-forge:v1.0.0-cu132
docker run --rm --gpus all ghcr.io/samudragupto/cutile-forge:v1.0.0-cu132 \
    cutile-forge --help

# Option B: Editable source install (dash/bash compatible)
git clone https://github.com/samudragupto/cutile-forge.git
cd cutile-forge
python3 -m venv .venv
. .venv/bin/activate
export CUTILE_ARCH=sm_120   # or sm_89 for Ada (RTX 4060)
pip install -e ".[cuda,dev,serve]" --no-build-isolation
```

### 3.2 Python SDK & `torch.compile(backend="cutile_forge")` Usage

#### 1. Zero-Code-Change `torch.compile` Backend

```python
import torch
import cutile_forge  # registers backend="cutile_forge" with torch._dynamo

compiled_block = torch.compile(my_transformer_block, backend="cutile_forge")
out = compiled_block(hidden_states, residual)
```

#### 2. High-Level `InferenceEngine` & `HybridLayerBudgeter` (7B / 47B / 70B)

```python
from cutile_forge.engine import InferenceEngine, HybridLayerBudgeter

engine = InferenceEngine.from_pretrained(
    "~/models/mistral_7b_4bit",
    target_arch="sm_120",
    use_cuda_graph=True,
)
print(engine.generate("The capital of France is", max_new_tokens=32))
```

#### 3. Building & Compiling a Custom `FusionGraph`

```python
from cutile_forge.forge.fusion_graph import build_mistral_layer_tfg
from cutile_forge.forge.codegen import compile_tfg

tfg = build_mistral_layer_tfg(batch_size=1, seq_len=128)
artifact = compile_tfg(tfg, target_arch="sm_120")
print(f"Fused kernels: {len(artifact.kernels)}")  # 3 fused CuTile kernels
```

### 3.3 Production OpenAI Server (`cutile-forge serve`)

```sh
# Launch OpenAI-compatible FastAPI server with DynamicBatcher (max_batch=16, 8ms window) & RadixPrefixCache
cutile-forge serve --model ~/models/mistral_7b_4bit --host 127.0.0.1 --port 8000 --target sm_120

# Non-streaming request
curl http://127.0.0.1:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"mistral_7b_4bit","messages":[{"role":"user","content":"Explain tile fusion."}],"max_tokens":32}'

# SSE token streaming
curl -N http://127.0.0.1:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"mistral_7b_4bit","messages":[{"role":"user","content":"Count from 1 to 5"}],"max_tokens":32,"stream":true}'
```

### 3.4 Repository Architecture & Artifact Reproduction

```text
cutile-forge/
├── cutile_forge/
│   ├── forge/          # TFG IR, 6 fusion rules, occupancy cost model, Stage -1 DMA, FX backend
│   ├── kernels/        # Native cuda.tile kernels (GEMM, INT4/NVFP4, GQA FlashAttn-2, Split-KV, MoE)
│   ├── engine/         # InferenceEngine, PagedKVCache, RadixPrefixCache, HybridLayerBudgeter
│   ├── server/         # OpenAI FastAPI server, DynamicBatcher, SSE streaming
│   └── baselines/      # CUDA C++ (cp.async + WMMA) & cuBLAS pybind11 reference baselines
├── tests/              # Unit & integration tests (CI + GPU)
├── benchmarks/         # Micro-benchmarks, E2E real-model eval, Roofline, CSVs
├── scripts/            # Publication figure generator (300-DPI)
└── docs/               # RESULTS_RTX5090.md, RESULTS.md, ARCHITECTURE.md, preprint, blog
```

```sh
. .venv/bin/activate
export CUTILE_ARCH=sm_120

# 1. Run full quality & correctness gate (102 passed, 0 skipped)
ruff check cutile_forge/ benchmarks/ tests/ examples/ scripts/
black --check cutile_forge/ benchmarks/ tests/ examples/ scripts/
mypy cutile_forge/ --ignore-missing-imports
python3 -m pytest tests/ benchmarks/correctness/ -v --tb=short

# 2. Reproduce benchmarks & regenerate all 300-DPI publication figures
python3 benchmarks/run_all.py --model ~/models/mistral_7b_4bit/ --target sm_120 --gpu rtx5090 --output benchmarks/results_5090/
python3 benchmarks/e2e/bench_real_large_models.py --output benchmarks/results_5090/real_large_models.csv
python3 scripts/generate_publication_figures.py
```

#### Further Documentation
- [docs/RESULTS_RTX5090.md](docs/RESULTS_RTX5090.md) — Complete RTX 5090 (`sm_120`) tables, figures, real 47B/70B logs, and serving telemetry
- [docs/RESULTS.md](docs/RESULTS.md) — Lenovo LOQ 15APH8 RTX 4060 Laptop (`sm_89`, 8 GB VRAM) benchmark suite
- [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) & [docs/TFG_COMPILER.md](docs/TFG_COMPILER.md) — TFG IR, 6 fusion passes, and Stage -1 DMA internals
- [docs/paper/cutileforge_preprint.md](docs/paper/cutileforge_preprint.md) & [docs/blog/introducing-cutileforge.md](docs/blog/introducing-cutileforge.md) — Technical preprint & engineering writeup

### 3.5 Citation & License

```bibtex
@software{cutileforge2026,
  title   = {CuTileForge: A Tile-Fusion Graph Compiler and Hybrid PCIe-Streaming Inference Engine on NVIDIA CuTile},
  author  = {Nag, Arrhat and Nasre, Rupesh},
  year    = {2026},
  version = {1.0.0},
  url     = {https://github.com/samudragupto/cutile-forge},
  license = {MIT}
}
```

MIT — see [LICENSE](LICENSE).
