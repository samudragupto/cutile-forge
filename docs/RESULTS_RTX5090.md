# CuTileForge v0.2.0 — NVIDIA GeForce RTX 5090 (`sm_120`) Benchmark Report (Stub)

## 1. Target Hardware & Compiler Configuration

| Parameter | RTX 5090 (`sm_120`) Configuration | Lenovo LOQ 15APH8 (`sm_89`) Baseline |
| --- | --- | --- |
| **Preset** | `cutile_forge.engine.config.RTX5090` | `cutile_forge.engine.config.LOQ_RTX4060` |
| **Architecture** | Blackwell `sm_120` (`sm=120`, `170` SMs) | Ada Lovelace `sm_89` (`sm=89`, `24` SMs) |
| **VRAM Capacity** | **32 GB GDDR7** (`vram_budget_gb=28.8`) | **8 GB GDDR6** (`vram_budget_gb=7.2`) |
| **Host Interconnect** | PCIe 5.0 x16 (`pcie_gen=5`) | PCIe 4.0 x8 (`13.30 GB/s` sustained H2D) |
| **TFG Codegen Defaults** | `target="sm_120"`, `pipeline_depth=4` (`latency=4`) | `target="sm_89"`, `pipeline_depth=3` (`latency=3`) |
| **FP8 Tensor Core Path** | Enabled (`GPUConfig.fp8=True`, `gemm_fp8_kernel` / `cutile.float8`) | Disabled (`GPUConfig.fp8=False`, falls back to FP16/INT4) |
| **Stage -1 Weight Streamer** | **Auto-disabled for 7B** (`4.16 GB <= 32 GB * 0.85 = 27.2 GB`; 100% HBM-resident) | Enabled when model exceeds `85%` VRAM budget (`93.8–100%` overlap) |
| **Large-Model Configs** | `MISTRAL_8x7B_4BIT` (`experts=8, d_model=4096`), `LLAMA_34B_4BIT` (`d_model=8192, n_layers=48`) | `Mistral-7B-v0.1` 4-bit GPTQ (`d_model=4096, n_layers=32`) |

---

## 2. Architectural Highlights (`v0.2.0`)

1. **4-Stage Software Pipelining (`sm_120`)**:
   `compile_tfg()` now defaults to `target="sm_120"` and `pipeline_depth=4`, emitting `latency=4` hints on `ct.load` across all 3 fused TFG kernels (`FUSED_QKV_ATTN`, `FUSED_O_LAYERNORM`, `FUSED_FFN_NORM`).
2. **FP8 Tensor Core Kernel (`gemm_fp8_kernel`)**:
   `cutile_forge/kernels/gemm_dequant.py` adds `gemm_fp8_kernel` (`dtype=cutile.float8` / `ct.float8_e4m3fn`) and `cutile_gemm_fp8()`, guarded by `GPUConfig.fp8`. On `sm_89` configurations where `fp8=False`, `cutile_gemm_fp8()` automatically falls back to `cutile_gemm_fp16()`.
3. **32 GB Resident Execution (Zero PCIe Streaming Overhead for 7B–34B)**:
   `WeightStreamer` automatically sets `self.enabled = (model_gb > gpu_vram_gb * 0.85)`. On the 32 GB RTX 5090 (`threshold = 27.2 GB`), `Mistral-7B-v0.1` (`4.16 GB`), `Mixtral-8x7B` 4-bit (`~23.5 GB`), and `Llama-34B` 4-bit (`~18.5 GB`) remain resident in GDDR7 HBM without PCIe DMA streaming.

---

## 3. Running the RTX 5090 Benchmark Suite

```bash
python benchmarks/run_all.py \
  --target sm_120 \
  --gpu rtx5090 \
  --model ~/models/mistral_7b_4bit/
```

All CSV outputs are written to `benchmarks/results_5090/`:

| Benchmark Suite | Output CSV | Status on RTX 5090 (`sm_120`) | Lenovo LOQ 15APH8 (`sm_89`) Reference |
| --- | --- | --- | --- |
| **FP16 & FP8 GEMM** | `benchmarks/results_5090/gemm.csv` | *Pending benchmark* | `28.18 TFLOPS` (`256×14336×4096`) |
| **INT4 Dequant + GEMM** | `benchmarks/results_5090/gemm_dequant.csv` | *Pending benchmark* | `23.37 TFLOPS` (`2.15×` vs CUDA at `seq=256`, `7.12×` at `seq=1`) |
| **Fused FFN (1k vs 3k)** | `benchmarks/results_5090/ffn.csv` | *Pending benchmark* | `5.48×` (`seq=1`), `1.78×` (`seq=512`) |
| **GQA FlashAttention-2** | `benchmarks/results_5090/attention.csv` | *Pending benchmark* | `21.76 TFLOPS` vs `13.23 TFLOPS` PyTorch SDPA (`seq=1024`) |
| **Full TFG Layer (3k vs 8k)** | `benchmarks/results_5090/layer.csv` | *Pending benchmark* | `5.47×` (`seq=1`), `2.22×` (`64`), `1.64×` (`256`), `1.96×` (`512`) |
| **E2E Decode (256 tok)** | `benchmarks/results_5090/decode.csv` | *Pending benchmark* | `32.43 tok/s` (`30.83 ms/tok`, `4.30 GB` peak VRAM) |
| **30-Min Sustained Thermal** | `benchmarks/results_5090/sustained/` | *Pending benchmark* | `31.16 tok/s` avg, `83.0 °C` max, `4.62%` avg throttle |

