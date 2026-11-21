# Changelog

## v2.0.0 (2026-11-21) — General CuTile Compiler, SDK, Training, Vision & Ecosystem

- **Phases 1–7 (Developer SDK, General Ops, CuTileTune, CuTileRuntime, Training, ONNX, Serving)**: Added imperative `cutile_forge.sdk.Graph` + `compile` + `KernelRegistry`, `cutile-forge doctor`, general CuTile kernels (`bmm`, `reduce_sum`, `reduce_max`, `cumsum`, `transpose`, `gather`, `conv2d`, `gelu`, `silu`), `TuningDB` + regression detection, `PinnedHostPool` / `TensorStreamer` / `GraphRuntime`, `CuTileSiLUFn` / `CuTileRMSNormFn` / `cutile_adamw_step`, `import_onnx_model`, and `/metrics` + optional Bearer authentication on the OpenAI server (`176 → 191` tests).
- **Phase 8 M1 — General Kernel Optimization (`sm_120` RTX 5090)**: Optimized 2D multi-row tiling for `reduce_sum` (`0.0672 ms -> 0.0172 ms` at `16384x1024`), `reduce_max` (`0.0570 ms -> 0.0172 ms`, `0.99×` vs PyTorch), L2-swizzled `bmm_kernel` (`1.88×` vs PyTorch on `16x128x2048x2048`, `1.01×` on `4x1024x1024x1024`), `cumsum` (`2.05×` on `4096x4096`), `transpose` (`0.87×` at `16384x1024`), `activations`, `gather`, and `conv2d`.
- **Phase 8 M2 — CuTileTune v2 (`cutile_forge.autotune`, `TileAutotuner.from_db`)**: Added input-shape fingerprinting (`shape_fingerprint`), JSON benchmark history tracking, and specialized autotuners (`tune_bmm`, `tune_reduction`, `tune_transpose`, `tune_activation`).
- **Phase 8 M3 — Generalized TFG Compiler & CuTileVision v2 (`cutile_forge.forge`, `cutile_forge.vision`)**: Added `OpSpec` / `OpRegistry`, `LayoutStridePass`, `ConstantFoldPass`, `ProfitabilityPass`, `MemoryLifetimePass`, HBM lifetime/timeline telemetry, and `cutile-forge inspect` CLI.
- **Phase 8 M4 — Expanded PyTorch FX/Dynamo Backend (`torch.compile(backend="cutile_forge")`)**: Added `FXOpRegistry`, `ShapeGuard`, `lowered_ops` stats, and standalone CuTile lowerings for `linear`, `addmm`, `matmul`, `layer_norm`, `gelu`, `silu`, `sum`, `amax`, `cumsum`, `conv2d`, and `embedding`.
- **Phase 8 M5 — Training Ecosystem (`cutile_forge.kernels`)**: Added CuTile backward kernels for GELU, Softmax, LayerNorm, Linear, and BMM alongside autograd wrappers (`CuTileLinearFn`, `CuTileGELUFn`, `CuTileLayerNormFn`), `cutile_grad_clip_norm`, `cutile_multi_tensor_adamw` (`1.51×` vs PyTorch on 6-tensor MLP), and 10-step MLP training convergence parity (`max |Δloss| = 1.08e-04`, step-10 `|Δloss| = 2.83e-07`).
- **Phase 8 M6 — Runtime Pool Reuse & Vision Primitives (`cutile_forge.runtime`, `cutile_forge.kernels.vision`)**: Added `MemoryLifetimeAllocator`, adaptive `TensorStreamer` slot sizing, `p2p_capable`, `cutile_depthwise_conv2d`, `cutile_conv2d_backward`, `cutile_max_pool2d`, `cutile_avg_pool2d`, and `CuTilePatchEmbed`.
- **Phase 8 M7 — CuTileBench v2 & Distributed Prototype (`cutile_forge.bench`, `cutile_forge.distributed`)**: Added `RichBenchRecord`, `benchmark_with_telemetry`, `check_regression_gate`, and single-node/prototype-tested `detect_topology`, `p2p_copy`, and `ring_allreduce_sum` (`214/214` passing tests).

## v1.0.0 (2026-11-19)

- **PyTorch Dynamo / FX Backend (`cutile_forge.forge.fx_backend`)**: Registered `torch.compile(backend="cutile_forge")` on package import, pattern-matching `linear+silu+linear`, `add+layer_norm`, and `scaled_dot_product_attention` through TFG fusion rules to fused CuTile kernels.
- **Real Large-Model Checkpoints (`Mixtral-8x7B` & `Llama-2-70B`)**: Lazy memory-mapped multi-shard `.safetensors` reader (`_ShardReader`), per-channel GPTQ (`group_size == -1`) support, real `Mixtral-8x7B-Instruct-v0.1-GPTQ` (`41.83 tok/s`, PPL `7.48`, `25.26 GB` VRAM), and real `Llama-2-70B-Chat-GPTQ` 80-layer hybrid (`55 HBM + 25 PCIe 5.0`, `3.90–4.07 tok/s`, `27.44 GB` VRAM, zero OOM).

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

