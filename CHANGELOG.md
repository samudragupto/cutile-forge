# Changelog

## v0.5.0 (2026-11-19)

- **OpenAI-Compatible Server (`cutile_forge.server`)**: FastAPI `/health`, `/v1/models`, and `/v1/chat/completions` (JSON + SSE `stream=True`) with `cutile-forge serve` / `cutile-forge bench` CLI and `Dockerfile`.
- **Empirical Roofline Plotter (`cutile_forge.bench.ncu_roofline`)**: Dual-architecture `sm_89` vs `sm_120` log-log Roofline chart (`docs/images_5090/roofline_ada_vs_blackwell.png`).

## v0.4.0 (2026-11-19)

- **Blackwell Precision & Fused MoE**: `PagedKVCache(kv_dtype="fp8_e4m3fn")` (50% KV VRAM savings), `fused_gemm_nvfp4_dequant` (`e2m1` register LUT, `88.56 TFLOPS`), `fused_topk_gating` + `fused_moe_int4_gemm` (`MoEFFNBlock` for Mixtral-8x7B), and `unpack_awq_to_cutile` / `unpack_gguf_q4_k_to_cutile` multi-format loaders.

## v0.3.0 (2026-11-19)

- **Systems Hardening (`sm_120`)**: Pre-allocated pinned CPU ring-buffer pool + async `swap_stream` in `PagedKVCache` (16K context `107,854 ms` → `76.29 ms`), occupancy cost model `select_ffn_execution_strategy` (`seq=256` FFN `0.86×` → `1.55×`), `torch.cuda.CUDAGraph` decode (`96.57 tok/s`), and `NGramPromptLookupDrafter` + KV rollback (`117.31 tok/s`).

## v0.2.0 (2026-11-19)

- **RTX 5090 (`sm_120`) Support**: `cuda_tile_init`, `[K//2, N]` weight layout assertions, `cutile_gemm_fp8`, 32 GB model presets, and full RTX 5090 benchmark suite (`49.82 tok/s` decode, `94.69 tok/s` 30-min sustained, `136.14 TFLOPS` GEMM).

## v0.1.0 (2026-11-18)

- **TFG Compiler (`cutile_forge.forge`)**: `FusionGraph` IR, 6 fusion rules (8 -> 3 kernels per Mistral layer), 3-stage software pipelining, Stage -1 PCIe DMA `WeightStreamer`, 12-config `TileAutotuner`, `sm_89` CuTile source/cubin `codegen`, and Graphviz/SVG visualization.
- **CuTile Kernels (`cutile_forge.kernels`)**: Fused INT4-dequant GEMM, GEMM+SiLU, single-kernel FFN, FlashAttention-2 GQA (32:8), LayerNorm/RMSNorm + residual, RoPE, online softmax, and top-k/top-p sampling.
- **Inference Engine (`cutile_forge.engine`)**: 4-bit GPTQ loader with `desc_act` `g_idx` support, paged HBM/DDR5 `PagedKVCache`, `Mistral7B`, `InferenceEngine`, `SpeculativeDecoder`, and `TokenGenerator`.
- **CUDA C++ & cuBLAS Baselines (`cutile_forge.baselines`)**: Hand-written 3-stage `cp.async`+WMMA GEMM, separate INT4 dequant, FlashAttention-2, LayerNorm/RMSNorm, SiLU, and cuBLAS HGEMM pybind11 module.
- **Benchmarks & CI/CD**: Full micro and E2E benchmark suite, NVML/PCIe/thermal/overlap profilers, automated Markdown + matplotlib reporting, 76 unit + correctness tests, and GitHub Actions workflows.

