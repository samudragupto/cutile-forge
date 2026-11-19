# Changelog

## v1.0.0 (2026-11-19)

- **PyTorch Dynamo / FX Backend (`cutile_forge.forge.fx_backend`)**: Registered `torch.compile(backend="cutile_forge")` on package import, pattern-matching `linear+silu+linear`, `add+layer_norm`, and `scaled_dot_product_attention` through TFG fusion rules to fused CuTile kernels.

## v0.8.0 (2026-11-19)

- **70B Hybrid HBM + Stage -1 PCIe 5.0 Engine**: Added `LLAMA3_70B_4BIT` (`80L, d=8192, ffn=28672, 64:8 GQA`) and `HybridLayerBudgeter`; verified 80-layer 70B hybrid pass on 32 GB RTX 5090 (`57` HBM + `23` PCIe layers, `27.43 GB` peak VRAM, `99.5%` overlap, `4.66 tok/s`).

## v0.7.0 (2026-11-19)

- **Persistent Grouped MoE & FlashDecoding Split-KV**: Single-launch 3D-grid `moe_grouped_gemm_int4_kernel` (`42.13 TFLOPS` at `M=256`, `19.73 TFLOPS` at `M=64`) and `flash_decoding_split_kv` (`num_splits=16`, 16K context `76.29 ms → 59.20 ms`).

## v0.6.0 (2026-11-19)

- **Dynamic Continuous Batching & Radix Prefix Cache**: Added `RadixPrefixCache` with LRU eviction in `InferenceEngine.prefill()`, `BatchedKVCache` + `decode_batch()`, and async `DynamicBatcher` in `/v1/chat/completions` (4-way concurrent p99 `363.1 ms → 113.6 ms`).

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

