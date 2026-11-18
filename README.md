# CuTileForge

[![CI](https://github.com/samudragupto/cutile-forge/actions/workflows/ci.yml/badge.svg)](https://github.com/samudragupto/cutile-forge/actions/workflows/ci.yml)
[![Release](https://img.shields.io/github/v/release/samudragupto/cutile-forge?color=2563eb)](https://github.com/samudragupto/cutile-forge/releases/tag/v0.1.0)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-3776ab.svg)](https://www.python.org/downloads/)
[![CUDA sm_89](https://img.shields.io/badge/CUDA-sm__89%20%7C%20CuTile%201.8-76B900.svg)](https://developer.nvidia.com/cuda-toolkit)

**Tile-Fusion Graph (TFG) Compiler & CuTile LLM Inference Engine for 8 GB-VRAM GPUs.**

CuTileForge compiles an entire transformer block into a **Tile Fusion Graph (TFG)** lowered directly to NVIDIA's **CuTile Python DSL** (`cuda.tile`). By applying six algebraic and tile-level fusion passes, it collapses **8 separate kernel launches** (`QKV GEMM`, `GQA Attention`, `O GEMM`, `LayerNorm`, `FFN-up GEMM`, `SiLU`, `FFN-down GEMM`, `LayerNorm+Residual`) into **3 fused tile kernels** with 3-stage software pipelining (`latency=3`) and **Stage -1** pinned-DDR5 PCIe DMA weight streaming.

---

## Why CuTileForge?

- **Fits 7B Inference Comfortably in 8 GB VRAM**: Runs `Mistral-7B-v0.1` 4-bit GPTQ at **32.43 tok/s** (`30.83 ms/tok`) using **4.30 GB** peak VRAM on an RTX 4060 Laptop GPU, and scales to **32K context** in **6.01 GB** via paged HBM/DDR5 KV-cache eviction.
- **Native `desc_act=True` GPTQ Support**: Automatically resolves GPTQ activation-order `g_idx` row permutations at load time with contiguous group tiling—whereas `vllm==0.34.0` rejects `desc_act=True` GPTQ checkpoints with a `VllmConfig` validation error.
- **Python-First Kernel Productivity Without Sacrificing Throughput**: ~20 lines of `cuda.tile` Python DSL matches or beats hand-written 150+ line `cp.async` + WMMA CUDA C++ baselines (`28.18 TFLOPS` vs `14.90 TFLOPS` on `256×14336×4096`) and outperforms PyTorch SDPA across all tested sequence lengths (`21.76 TFLOPS` vs `13.23 TFLOPS` at `seq=1024`).

---

## Architecture & TFG Compilation

```mermaid
graph LR
    subgraph Unfused["Before Fusion: 8 Separate Kernels"]
        direction TB
        U1[QKV GEMM] --> U2[GQA Attention] --> U3[O GEMM] --> U4[LayerNorm] --> U5[FFN-up GEMM] --> U6[SiLU] --> U7[FFN-down GEMM] --> U8[LayerNorm+Residual]
    end
    subgraph Fused["After TFG Passes: 3 CuTile Kernels"]
        direction TB
        T1["1. FUSED_QKV_ATTN<br/>(QKV + RoPE + FlashAttn-2)"] --> T2["2. FUSED_O_LAYERNORM<br/>(O-proj + Norm + Residual)"] --> T3["3. FUSED_FFN_NORM<br/>(Up + SiLU + Down + Norm)"]
    end
```

<p align="center">
  <img src="docs/tfg_layer0.svg" alt="Compiled Mistral Layer 0 TFG SVG" width="540"/>
</p>

See [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) for the full compiler pass pipeline, Stage -1 PCIe DMA sequence diagram, and SM-aware scheduling details, and [docs/CUDA_COMPARISON.md](docs/CUDA_COMPARISON.md) for side-by-side CuTile vs CUDA C++ code.

---

## Benchmarks & Telemetry

**Test Rig:** Lenovo LOQ / IdeaPad 15APH8 · NVIDIA GeForce RTX 4060 Laptop (`AD107`, `sm_89`, 24 SMs, 8 GB GDDR6) · AMD Ryzen 7 7735HS · 32 GB DDR5-4800 · PCIe 4.0 · Model: `TheBloke/Mistral-7B-v0.1-GPTQ` (`D:\models\mistral_7b_4bit\`, 4.16 GB).

### Terminal Run Snapshot

<p align="center">
  <img src="docs/images/terminal_demo.png" alt="CuTileForge CLI Benchmark & Correctness Output" width="880"/>
</p>

### Visual Performance Charts

| Full Transformer Layer (3k TFG vs 8k CUDA) | Fused FFN Speedup vs 3 Separate Kernels |
| :---: | :---: |
| ![Layer Latency](docs/images/layer_latency.png) | ![FFN Speedup](docs/images/ffn_speedup.png) |

| Prefill FP16 GEMM Throughput (`sm_89`) | Grouped-Query FlashAttention-2 (`32:8`) |
| :---: | :---: |
| ![GEMM TFLOPS](docs/images/gemm_tflops.png) | ![Attention TFLOPS](docs/images/attention_tflops.png) |

### 30-Minute Sustained Thermal & Clock Stability

<p align="center">
  <img src="docs/images/thermal_sustained.png" alt="30-Minute Sustained Thermal and Throughput Telemetry" width="820"/>
</p>

### Summary Tables

#### End-to-End Inference (`Mistral-7B-v0.1` 4-bit GPTQ)

| Workload | Configuration | Throughput / Latency | Peak VRAM / Notes |
| --- | --- | --- | --- |
| **Decode (256 tokens)** | `CuTileForge` (batch=1) | **32.43 tok/s** (`30.83 ms/tok`) | **4.30 GB** (`13.41 GB/s` PCIe H2D) |
| **vLLM Baseline (`0.34.0`)** | `quantization="gptq"` / `"gptq_marlin"` | `N/A` (unsupported) | Rejects `desc_act=True` checkpoints |
| **Prefill (`2048` / `4096` tok)** | `CuTileForge` | **1,499.0 tok/s** / **1,475.7 tok/s** | **4.67 GB** / **5.08 GB** |
| **Batched Forward (`batch=16`)** | `seq=32` | **1,439.7 tok/s** (`355.63 ms`) | **4.36 GB** |
| **Long Context (`16K` / `32K`)** | `PagedKVCache` (1.5 GB GPU budget) | `1,862 ms` / `3,875 ms` step | **5.94 GB** / **6.01 GB** (`26,624` evictions) |
| **30-Min Sustained Decode** | `361` samples, `54,976` tokens | **31.16 tok/s** avg | **83.0 °C** max (`78.7 °C` avg), **85.6 W**, `4.62%` avg throttle |

#### Micro-Benchmarks (CuTile vs Hand-Written CUDA C++ vs cuBLAS vs PyTorch)

| Benchmark | Workload / Shape | CuTileForge | Baseline Comparison |
| --- | --- | --- | --- |
| **Full Layer (3k vs 8k)** | `seq = 1 / 64 / 256 / 512` | `0.638 / 1.632 / 4.934 / 7.584 ms` | **5.47× / 2.22× / 1.64× / 1.96×** vs 8-kernel CUDA |
| **Fused FFN (1k vs 3k)** | `seq = 1 / 64 / 256 / 512` | `0.471 / 1.314 / 3.888 / 5.420 ms` | **5.48× / 2.00× / 1.46× / 1.78×** vs 3 CUDA kernels |
| **INT4 Dequant + GEMM** | `256×12288×4096` / `1×12288×4096` | **23.37 TFLOPS** (`1.10 ms`) / `0.148 ms` | **2.15×** / **7.12×** vs separate CUDA dequant+GEMM |
| **FP16 GEMM (Compute-Bound)** | `256×14336×4096` | **28.18 TFLOPS** (FP32-acc) | PyTorch **25.48**, CUDA **14.90**, cuBLAS **45.70** (FP16-acc) |
| **GQA Attention (`32:8`)** | `seq = 256 / 512 / 1024` | **11.92 / 17.77 / 21.76 TFLOPS** | PyTorch SDPA **5.02 / 10.38 / 13.23 TFLOPS** |
| **Stage -1 PCIe Overlap** | `32 / 64 / 115 MB` weight tiles | **100.0% / 100.0% / 93.8%** overlap | **13.30 GB/s** (`84.4%` PCIe utilization) |

Full CSV tables and methodology are in [docs/RESULTS.md](docs/RESULTS.md) and [docs/BENCHMARK_METHODOLOGY.md](docs/BENCHMARK_METHODOLOGY.md).

---

## Quickstart

```bash
# 1. Activate environment & install package + CUDA C++ baselines
conda activate llm-kernel
pip install -e ".[cuda,dev]" --no-build-isolation
python -m cutile_forge.baselines.build

# 2. Download Mistral-7B-v0.1 4-bit GPTQ to D:\models\mistral_7b_4bit\
python models/download.py --output D:/models/mistral_7b_4bit/

# 3. Run all 76 unit + GPU correctness tests
python -m pytest tests/ benchmarks/correctness/ -v

# 4. Run inference & inspect TFG compilation
python examples/generate.py --model D:/models/mistral_7b_4bit/ --prompt "The capital of France is" --max-tokens 32
python examples/visualize_tfg.py --layer 0 --output docs/tfg_layer0.svg
python examples/compare_cuda.py --shape 256x12288x4096

# 5. Run benchmark suite & regenerate docs/RESULTS.md + plots
python benchmarks/run_all.py --model D:/models/mistral_7b_4bit/ --output benchmarks/results/
```

---

## Repository Layout

```text
cutile-forge/
├── cutile_forge/
│   ├── forge/        # TFG IR, 6 fusion rules, 3-stage pipeliner, Stage -1 streamer, autotuner, codegen, SVG viz
│   ├── kernels/      # CuTile DSL kernels (INT4 GEMM, FFN, GQA FlashAttn-2, RMSNorm/LN+Res, RoPE, softmax)
│   ├── engine/       # Mistral-7B model, GPTQ loader (desc_act), PagedKVCache, prefill/decode, speculative
│   ├── baselines/    # Hand-written CUDA C++ (cp.async+WMMA, FlashAttn-2, LN, SiLU) & cuBLAS pybind11 module
│   ├── bench/        # CUDA-event harness, NVML/PCIe/thermal/overlap profilers, Markdown+PNG report builder
│   └── utils/        # 4-bit group quantization, GPTQ unpacking, VRAM tracking, structured logging
├── benchmarks/       # Micro (8), E2E (6 incl. 30-min thermal), and GPU correctness tests (27)
├── tests/            # Unit tests (49) runnable on CPU or GPU
├── examples/         # CLI scripts for generation, TFG SVG export, autotuning, CUDA comparison
├── docs/             # Architecture, CuTile kernels, TFG compiler, tuning guide, RESULTS.md, charts & SVGs
└── .github/workflows/# CI (lint + unit tests), GPU correctness, benchmark, nightly thermal, release
```

---

## Testing & CI

- **Local Verification**: **76 / 76 tests passing** (`49` unit tests in `tests/` + `27` GPU correctness tests in `benchmarks/correctness/`, including 32-layer numerical equivalence, token-exact greedy decode vs PyTorch, and WikiText-2 perplexity within 5%).
- **Lint & Static Analysis**: `ruff`, `black --check`, and `mypy` pass with `0` errors.
- **GitHub Actions**: [ci.yml](.github/workflows/ci.yml) runs `lint` and `unit-tests` on `ubuntu-latest` for every push/PR; GPU-dependent correctness, full benchmarks, and 30-minute thermal runs live in dedicated workflows ([gpu-correctness.yml](.github/workflows/gpu-correctness.yml), [benchmark.yml](.github/workflows/benchmark.yml), [nightly.yml](.github/workflows/nightly.yml)) targeting `[self-hosted, gpu]`.

---

## Documentation

- [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) — TFG compiler pipeline, Stage -1 DMA sequence, and runtime design
- [docs/TFG_COMPILER.md](docs/TFG_COMPILER.md) — Detailed specification of the 6 fusion rules and software pipelining
- [docs/CUTILE_KERNELS.md](docs/CUTILE_KERNELS.md) — Kernel inventory and tile layouts
- [docs/CUDA_COMPARISON.md](docs/CUDA_COMPARISON.md) — CuTile Python DSL vs CUDA C++ comparison
- [docs/BENCHMARK_METHODOLOGY.md](docs/BENCHMARK_METHODOLOGY.md) — Timing, NVML, and PCIe measurement protocol
- [docs/TUNING_GUIDE.md](docs/TUNING_GUIDE.md) — Tile autotuning and 8 GB VRAM memory budgeting
- [docs/RESULTS.md](docs/RESULTS.md) — Full auto-generated benchmark tables and telemetry

## License

MIT — see [LICENSE](LICENSE).
