# Changelog

## v2.5.0 (2026-11-21) — Phase 10: Universal Quantization Engine

- **Native Multi-Format CuTile Kernels (`cutile_forge.kernels.universal_quant`)**: Implemented dedicated CuTile Tensor Core dequantization + GEMM kernels without load-time format conversion for:
  - **GPTQ INT4** (`group_size in {128, 64, 32, -1}`, `desc_act=True/False`)
  - **AWQ INT4** (`gemm_awq_native_kernel` / `cutile_gemm_awq_native` executing directly on AutoAWQ `[K, N//8]` `int32` `[0, 16, 4, 20, 8, 24, 12, 28]` interleaved bit-packed tensors)
  - **GGUF `Q4_0`, `Q8_0`, `Q4_K`, `Q6_K`, `Q4_K_M`** (`gemm_gguf_q4_0_kernel`, `gemm_int8_dequant_kernel`, `gemm_gguf_q4_k_kernel`, `gemm_gguf_q6_k_kernel` with `0.000000e+00` bit-exact reference dequant parity against `gguf.quants`)
  - **INT8** (`gemm_int8_dequant_kernel` / `cutile_gemm_int8_dequant` for `[K, N]` `int8` + group scales)
  - **FP8 (`e4m3fn`)** per-tensor (`cutile_gemm_fp8` with dynamic activation scaling) and 2D `(128, 128)` block-scaled (`gemm_fp8_block_scaled_kernel` / `cutile_gemm_fp8_block_scaled`)
- **Per-Layer & Per-Projection Mixed-Precision (`MixedPrecisionFormatTable`, `UniversalQuantLinear`, `UniversalQuantModel`)**: Enabled arbitrary per-layer/per-projection quantization format mixing (including canonical llama.cpp `GGUF Q4_K_M` mixing `Q6_K` on `v_proj`/`down_proj` and `Q4_K` on all other projections) with `< 0.05%` WikiText-2 perplexity drift vs FP32 reference across all 14 format configurations.
- **Format-Keyed Autotuning (`CostGuidedAutotuner.tune_quantized_gemm`)**: Extended `CuTileTune v2` to key and persist optimal tile configurations on `(quant_format, group_size, M, N, K)`.

## v2.4.0 (2026-11-21) — Phase 9: Native Runtime Performance, Full-Step CUDA Graphs & Scheduling

- **Full-Step Zero-ATen CUDA Graph Capture (`NativeModelRuntime`)**: Captured the entire 32-layer decode step + LM head argmax + device-side token/position increment (`cutile_record_and_advance_step`) into a single `torch.cuda.CUDAGraph` per context bucket (`512/1024/2048/4096`), eliminating all `164` eager kernel launches and `103` per-token allocations (`0` forbidden ATen calls, `0` eager `ct.launch` calls during replay).
- **Kernel & Layout Optimizations**: Added `(16, 32, 128)` bit-exact narrow-N INT4 dequant tiling for `M <= 16`, pre-packed QKV INT4 projections, `apply_rope_qk_kv_write_pos`, `fused_attention_gqa_decode_pos`, `fused_swiglu_dequant`, `fused_gemm_dequant_add`, head-major zero-gather `NativePagedKVCache`, and `flash_decoding_reduce_kernel`.
- **Measured RTX 5090 Results (`Mistral-7B-v0.1-GPTQ`, 256 tokens, 5 runs)**: Decode throughput improved from `103.22 tok/s` (`v2.3.0` 256-tok baseline; `66.85 tok/s` in `v2.2.0` 32-tok) to **`171.60 tok/s`** (`5.83 ms/tok`), prefill latency dropped from `10.14 ms` (`11.10 ms` in `v2.2.0`) to **`6.66 ms`**, batch-4 continuous batching reached **`672.80 tok/s`**, speculative decoding reached **`369.30–752.29 tok/s`**, and 32-layer 32K KV attention dropped from `9.29 ms` to **`4.28 ms/step`**, with **5/5 prompts × 256 tokens** bit-exact greedy parity.

## v2.3.0 (2026-11-21) — Track C: Unified ModelGraph, Multimodal/MoE & Multi-Format Compatibility

- **Declarative Unified Model IR (`cutile_forge.models.model_graph`, `registry`)**: Added `UnifiedModelGraph` with composable stages (`TextEmbeddingStage`, `VisionPatchEmbedStage`, `MultimodalFusionStage`, `DenseTransformerBlock`, `SparseMoEBlock`, `ViTEncoderBlock`, `LMHeadStage`, `VisionProjectionStage`, `ClassifierHeadStage`), TFG lowering (`to_fusion_graph()`, `compile_tfg()`), scaled RoPE (`compute_scaled_rope_freqs` supporting Linear, NTK/YaRN, and Llama-3 scaling), and portable `UnifiedModelExecutor`.
- **Architecture Family Registry (`MODEL_REGISTRY`)**: Registered `mistral` (sliding window), `llama`, `llama3` (500K RoPE scaling, 128K vocab), `qwen2` (QKV biases), `mixtral`, `deepseek_moe` / `qwen2_moe` (shared + routed fine-grained experts), `vit` / `clip_vision` / `siglip_vision`, and `llava` multimodal (`ViT -> 2-layer GELU projector -> unified LLM prefill stream`).
- **Multi-Format Checkpoint Auto-Detection & Unpackers (`cutile_forge.models.checkpoint_loader`, `cutile_forge.utils.quantize`)**: Added `detect_checkpoint_format`, pure-Python GGUF v2/v3 header parser (`parse_gguf_header`), `unpack_gguf_q4_0_to_cutile`, `unpack_gguf_q8_0_to_fp16`, block-scaled `unpack_fp8_e4m3_to_fp16`, and `plan_checkpoint_placement` integrating `HybridLayerBudgeter` whenever checkpoint footprint exceeds `0.85 * vram_gb`.

## v2.2.0 (2026-11-21) — Track B: Native Model Runtime (Zero-ATen Compute Hot Path)

- **Native Tensor & Stream Manager (`cutile_forge.runtime`)**: Added `CuTileTensor` (zero-copy `MemoryLifetimeAllocator` + DLPack exchange) and `NativeStreamManager` (`compute_stream`, `dma_stream`, `kv_swap_stream` + pooled CUDA events).
- **Native Embedding, LM Head & Elementwise (`embedding_lmhead.py`)**: Added `cutile_embedding_lookup`, fused `cutile_lm_head_fused` (`lm_head_tile_argmax_kernel` + `lm_head_reduce_argmax_kernel` with zero `[M, V]` HBM logits allocation for greedy decode; multi-pass CuTile top-k for `top_k > 1`), `cutile_col_permute`, `cutile_ew_mul`, and `cutile_ew_add`.
- **GPU-Only MoE Routing (`moe_fused.py`)**: Added `cutile_moe_router_topk_kernel` (fused router GEMV + softmax + top-k), atomic `cutile_moe_bucket_kernel`, and `cutile_moe_scatter_combine_kernel` (`fused_moe_int4_native`, zero `.item()` host sync or `torch.topk`/`torch.argsort`).
- **Native Paged KV Cache & Engine (`native_kv_cache.py`, `native_engine.py`)**: Added in-place `cutile_kv_cache_write_kernel` / `cutile_kv_cache_gather_kernel` (`fp16` & `fp8_e4m3fn`) and `NativeModelRuntime` (`0` forbidden ATen compute calls per decode vs. `66` in `InferenceEngine`, `66.85 tok/s`, `11.1 ms` prefill, exact 32-token greedy parity on real `Mistral-7B-v0.1-GPTQ`).

## v2.1.0 (2026-11-21) — Track A: Kernel Performance Engineering & Profiling

- **Nsight Profiling Harness (`cutile_forge.bench.profiler`)**: Added `detect_profilers`, `profile_kernel_ncu`, `profile_timeline_nsys`, `analytical_profile_op`, and `save_profile_json` with graceful `ERR_NVGPUCTRPERM` fallback to CUDA-event analytical HBM/occupancy profiling.
- **Vision Kernel Redesign (`cutile_forge.kernels.vision`, `indexing`)**: Replaced unfold-heavy baselines with coalesced 2D spatial halo `depthwise_conv2d_kernel`, zero-loop `max_pool2d_2x2_s2_kernel` / `avg_pool2d_2x2_s2_kernel` + general `ct.gather` poolers, direct `patch_embed_fused_kernel` (with optional `pos_embed` fusion), zero-copy NCHW/NHWC `conv2d_1x1_nchw_kernel`, and `conv2d_implicit_gemm_kernel`.
- **Gather, Reductions & Fused Norm+Activation (`indexing`, `reductions`, `layernorm_residual`, `activations`, `fx_backend`)**: Added zero-copy int64/int32 `tile_d=512` `cutile_gather`, split-K low-row reductions, `cutile_fused_rmsnorm_silu`, `cutile_fused_layernorm_gelu` (+ FX Pass 2b `layer_norm -> gelu(tanh)` fusion), and `cutile_fused_add_rmsnorm_silu`.
- **Shape-Specialized GEMM Dispatch (`gemm_dequant`)**: Added `select_gemm_kernel_path` (`skinny_splitk`, `medium_l2`, `large_compute`) and deterministic `gemm_fp16_splitk_kernel`.

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

