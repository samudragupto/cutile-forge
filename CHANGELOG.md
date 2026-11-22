# Changelog

## [3.3.0] (2026-11-22) — Phase 16: Serving Architecture & Continuous Batching

**Authors:** Arrhat Nag (`@samudragupto`) and Rupesh Nasre

CuTileForge `v3.3.0` unifies streaming SSE and non-streaming HTTP inference atop slot-isolated continuous batching (`ContinuousBatchScheduler`), eliminates mixed-length request cross-contamination, and verifies 3-run reproducibility across `C = 1 / 4 / 16 / 64` on `Mistral-7B-v0.1-GPTQ`:

- **Unified Continuous Batch Scheduler (`cutile_forge/server/scheduler.py`, `cutile_forge/server/app.py`, `cutile_forge/cli.py`)**: Replaced the `stream=True` serialization lock (`batch=1.0`, `7.16%` of in-process at `C=16`, `27,738 ms` TTFT p99 at `C=64`) with an event-driven `ContinuousBatchScheduler` unifying SSE streaming and non-streaming `asyncio.Future` delivery over densely packed KV slots (`--max-batch-size 64`, `--batch-wait-ms 2.0`, `timeout_keep_alive=60`, FastAPI lifespan shutdown, and CUDA Graph bucket pre-capture).
- **Per-Slot Ragged Continuous Decode & Length-Grouped Prefill (`cutile_forge/runtime/native_engine.py`, `tests/test_phase16_serving_repro.py`)**: Added `prefill_into_slots`, `decode_slots_step` (with steady-state GPU-resident `token_id`/`pos_buf` reuse), and length-grouped `generate_batch` (eliminating left-padding contamination so concurrent mixed-length `C=4` (`4/4`) and `C=16` (`16/16`) greedy outputs match standalone `C=1` `100%`).
- **Mid-Stream Disconnect Cancellation & Zero KV Leaks (`cutile_forge/server/app.py`)**: Cooperatively aborts cancelled SSE streams (`cancel_event`), immediately releases KV slots (`active_kv_slots=0`, `requests_inflight=0`, `disconnects_delta=1`), and exposes live slot occupancy on `/metrics`.
- **Measured 3-Run HTTP Serving Validation (`benchmarks/production/bench_phase16_serving.py`, `v3.3.0/phase16_optimized/`)**: Lifted `Mistral-7B-v0.1-GPTQ` SSE streaming from `84.8 / 95.8 / 145.4 tok/s` to **`568.90 / 1,904.93 / 2,697.14 tok/s`** at `C=4/16/64` (`97.0% / 89.3% / 91.0%` of in-process, `CV <= 1.03%`), dropped streaming TTFT p99 to **`21.1 / 74.4 / 189.0 / 314.1 ms`** and ITL p99 to **`6.2 / 6.2 / 6.0 / 21.6 ms`**, and lifted non-streaming `C=1/4/16/64` to **`97.9% / 97.0% / 90.6% / 84.99%` (`164.22 / 568.83 / 1,931.66 / 2,519.59 tok/s`, `CV <= 0.27%`)**.

## [3.2.0] (2026-11-22) — Phase 15: Correctness Closure & 8-Model Regression Collection

**Authors:** Arrhat Nag (`@samudragupto`) and Rupesh Nasre

CuTileForge `v3.2.0` resolves critical long-context Sliding-Window Attention and FP8 W8A16 quantization correctness bugs, verifies end-to-end all-layer quantization across real checkpoints, and adds the permanent 8-model CLI regression suite:

- **Sliding-Window Attention Fix (`cutile_forge/kernels/attention_gqa.py`, `engine/kv_cache.py`, `engine/config.py`, `engine/model.py`, `runtime/native_engine.py`, `tests/test_phase15_correctness_repro.py`)**: Parsed `sliding_window` into `ModelConfig`, implemented exact `k_pos > q_pos - sliding_window` masking with dead prefix tile skipping in prefill, decode, and FlashDecoding kernels, and pruned evicted KV entries outside `[pos - W + 1, pos]` so real `Mistral-7B-v0.1-GPTQ` 8K tail perplexity drops from `602.03` to `3.1108` (`16K` tail perplexity `3.1532` with exact next-token match).
- **FP8 W8A16 & Real-Checkpoint Quantization Validation (`cutile_forge/kernels/universal_quant.py`, `benchmarks/production/bench_phase15_quant_real.py`)**: Replaced lossy E4M3 activation quantization with native W8A16 dequant-on-the-fly (`cutile_gemm_fp8_w8a16`, `cutile_gemm_fp8_channel_w8a16`), added Q/K/V biases and `rotary_style` handling (`half`/`interleaved`), and measured full all-layer `>= 512`-token perplexity across real `Qwen2.5-0.5B-Instruct` (`GPTQ_Int4`, `AWQ_Int4`, `FP8_channel`, `GGUF_Q4_0`, `GGUF_Q8_0`, `GGUF_Q6_K`) and `Mistral-7B-v0.1` (`GPTQ_Int4_desc_act`, `GGUF_Q4_K_M`) checkpoints while logging `Qwen2.5-0.5B GGUF_Q4_K_M` (`Q5_0` tensors) as honestly `UNSUPPORTED`.
- **8-Model Regression Collection & Release CLI (`cutile_forge/regress.py`, `cutile_forge/cli.py`, `cutile_forge/models/model_graph.py`)**: Added `cutile-forge regress --tier {fast,full}` covering Dense, MoE, Vision, Audio, Multimodal, and Diffusion models (`Mistral-7B`, `Qwen2.5-0.5B`, `Mixtral-8x7B`, `Llama-2-70B` NVMe streaming, `SigLIP`, `Whisper-Tiny`, `LLaVA-1.5-7B`, `DiT-XL-2-256`) alongside `cutile-forge bench release`.

## [3.1.0] (2026-11-21) — Phase 14: Production Validation & Hardware Scaling

**Authors:** Arrhat Nag (`@samudragupto`) and Rupesh Nasre

CuTileForge `v3.1.0` adds permanent release-validation infrastructure, honest negative/degraded accounting, hardened FastAPI/SSE production serving, and reproducible evidence bundles (`benchmarks/validation_artifacts/v3.1.0/`):

- **Validation Schema, Runner & Clean-Room Bundle (`benchmarks/production/schema.py`, `manifest.py`, `runner.py`, `scripts/reproduce_validation.sh`, `docs/REPRODUCE.md`)**: Every benchmark suite emits 7 reproducible artifacts (`manifest.json`, `samples.jsonl`, `summary.json`, `environment.json`, `commands.log`, `stdout.log`, `stderr.log`) with secret redaction, SHA-256 verification, and explicit `MEASURED` / `REPRODUCED` / `SIMULATED` / `DEGRADED` / `UNSUPPORTED` / `NOT_RUN` / `PENDING_HARDWARE` status labels.
- **External Engine Benchmarking (`benchmarks/production/probe_external_engines.py`)**: Installed and measured `SGLang 0.5.23` (`265.36 tok/s` `C=1`, `3,260.69 tok/s` `C=16` on Mistral-7B-GPTQ; `804.45 tok/s` on Qwen2.5-0.5B) and `vLLM 0.34.0` (`699.55 tok/s` on Qwen2.5-0.5B; **`UNSUPPORTED`** on `desc_act=True` Mistral-7B-GPTQ) using PyPI `cu13` wheels; recorded `TensorRT-LLM 1.3.0` as **`NOT_RUN`** (`Vendored triton_kernels module not found`). Documented `Qwen2.5-0.5B` CuTileForge `UnifiedModelExecutor` (`92.7 tok/s`) as **`DEGRADED`** (`0.58×` HF, `0.12×` SGLang).
- **Real Long-Context & Quantization Audits (`bench_long_context_real.py`, `bench_quant_e2e_parity.py`)**: Separated real-checkpoint `4K..32K` Mistral-7B (`118.64 -> 46.97 tok/s` FP16; `64K` `UNSUPPORTED`) and `512..8192` Mixtral-8x7B (`33.03 -> 14.74 tok/s`, `27.93 GB` VRAM) from synthetic FlashDecoding (`phase14c_synthetic_attention`). Verified 32L on-disk GPTQ end-to-end (`END_TO_END_VERIFIED`, `2/2` exact greedy match, `0.0211%` PPL diff) while honestly logging 8L repacked `LAYER_VERIFIED_ONLY` divergence in `FP8_e4m3_tensor` (`1.2888%` PPL diff), `GGUF_Q4_K`, `AWQ_gs128`, and `GPTQ_gs128_nodesc`.
- **Hardened OpenAI Server, 15-Min Soak & 10 Fault Scenarios (`cutile_forge/server/app.py`, `cutile_forge/cli.py`, `bench_serving_stress.py`, `bench_reliability_faults.py`)**: Added strict request validation (`extra="forbid"`, `max_prompt_tokens`), loopback bind default (`127.0.0.1`), single-worker executor offload, client disconnect cancellation for SSE streams, eager CUDA-graph fallback on capture failure, 15-minute live HTTP soak (`10,092/10,092` OK, `0.0 MB/min` RSS slope), and `10/10` fault recovery tests.
- **Permanent CI/Release Validation & Regression Gates (`compare_with_baseline.py`, `run_release_validation.py`, `.github/workflows/validation.yml`, `.github/workflows/production-validation.yml`, `CITATION.cff`)**: Added checked-in `v3.0.0` RTX 5090 baseline, 300-DPI validation figures (`docs/figures/validation_v310/`), and 8 dedicated Phase 14 validation reports in `docs/`.

## [3.0.0] (2026-11-21) — Consolidated Platform Release (Phases 9–13)

CuTileForge `v3.0.0` consolidates the full `v2.4.0` → `v2.8.0` roadmap into a unified, publication-grade CuTile compiler, universal quantization engine, multi-architecture model graph, distributed runtime, and production serving platform (`259/259` passing tests, `0` skipped, `92` `mypy`-clean modules):

- **Phase 9 (`v2.4.0`) — Full-Step Zero-ATen CUDA Graph Runtime (`NativeModelRuntime`)**: Captured the full 32-layer decode step, LM head argmax, and device-side `pos_buf`/history updates into a single `torch.cuda.CUDAGraph` per context bucket (`0` forbidden ATen compute calls, `0` eager `ct.launch` calls during replay), lifting real `Mistral-7B-v0.1-GPTQ` decode from `103.22 tok/s` to **`171.60 tok/s`** (`6.66 ms` prefill, **`5/5`** 256-token prompt parity, **`672.80 tok/s`** `B=4` continuous batching, and **`369.30–752.29 tok/s`** native speculative decoding).
- **Phase 10 (`v2.5.0`) — Universal Quantization Engine (`cutile_forge.kernels.universal_quant`)**: Added native CuTile Tensor Core kernels (no load-time conversion) for `GPTQ` (`gs=32/64/128/-1`, `desc_act`), `AWQ`, `GGUF` (`Q4_0`, `Q8_0`, `Q4_K`, `Q6_K`, mixed `Q4_K_M`), `INT8` (**`96.24 TFLOPS`** @ `M=128`), and `FP8` (per-tensor + 2D `128×128` block-scaled, **`91.77 TFLOPS`** @ `M=128`), with `0.000000e+00` bit-exact reference dequantization vs `gguf.quants` and **`0.0017%–0.0469%`** WikiText-2 perplexity drift.
- **Phase 11 (`v2.6.0`) — Architecture Completeness (`cutile_forge.models`)**: Expanded `MODEL_REGISTRY` & `UnifiedModelGraph` to **`13` architecture families** across dense LLMs (`Mistral`, `Llama-2`, `Llama-3.1`, `Qwen2/2.5`), MoE (`Mixtral`, `DeepSeek-MoE`, `Qwen2-MoE` with shared experts & single-GPU EP), Vision (`ViT`, `CLIP`, `SigLIP`), Audio (`Whisper-Tiny` encoder), Multimodal (`LLaVA`), and Diffusion (`DiT` with AdaLN-Zero); verified `6` families with **`real-checkpoint parity`** (`cos >= 0.999963`) and `7` families as `synthetic-only`.
- **Phase 12 (`v2.7.0`) — Distributed GPU Runtime (`cutile_forge.distributed`)**: Implemented `ColumnParallelLinear`, `RowParallelLinear`, `TensorParallelSwiGLUMLP`, `TensorParallelGQAAttention`, `PipelineParallelEngine`, `ExpertParallelMoE` (`all_to_all`), `DistributedPagedKVCache`, `DistributedScheduler`, and `CommTransport` / `LoopbackCommTransport`, validated for exact single-device equivalence (`max_abs_diff == 0.0` for PP/EP/KV-cache; multi-rank timings on this 1-GPU host labeled `simulated`).
- **Phase 13 (`v2.8.0`) — Production Benchmarking & External Engine Audit (`benchmarks/production/`)**: Measured **`172.58 tok/s`** single-stream (`6.11 ms` TTFT, `5.79 / 6.08 / 6.21 ms` p50/p90/p99 ITL), **`2,337.90 tok/s`** @ `C=16` and **`2,377.11 tok/s`** @ `C=64`, **`3.86 ms/step`** 32-layer `32K` FlashDecoding (`synthetic-weights`), `HF-Transformers` (`161.47 tok/s` on `Qwen2.5-0.5B`), and documented exact `NOT RUN` reasons for `vLLM`, `SGLang`, and `TensorRT-LLM` on this CUDA 13.2 / no-`nvcc` / no-`sudo` workstation.

## v2.8.0 (2026-11-21) — Phase 13: Production Benchmarking Suite & Documentation Sync

- **Production Benchmarking Harness (`benchmarks/production/bench_phase13_production.py`, `benchmarks/results_5090/phase13_production.csv`)**: Measured end-to-end TTFT (`6.11 ms` median), per-token ITL distribution (`p50=5.79 ms`, `p90=6.08 ms`, `p99=6.21 ms`), single-stream throughput (**`172.58 tok/s`**), concurrent batching throughput at concurrency `1` (`172.33 tok/s`), `4` (`617.81 tok/s`), `16` (**`2,337.90 tok/s`**), and `64` (**`2,377.11 tok/s`**), peak GPU VRAM (`8,565.7–8,608.9 MB`) & host RSS (`5,394.5 MB`), and 32-layer Paged KV + FlashDecoding latency at `8K` (`1.33 ms/step`), `16K` (`2.13 ms/step`), and `32K` (`3.86 ms/step` FP16; `7.60 ms/step` FP8).
- **External Engine Evaluation**: Benchmarked HuggingFace `transformers` (`torch 2.15+cu132`) on `Qwen/Qwen2.5-0.5B` (`161.47 tok/s`) and probed `vLLM` (`0.34.0`), `SGLang` (`0.5.23`), and `TensorRT-LLM` (`1.3.0`) in isolated `/var/tmp/arrhat_models/venvs`, recording exact `NOT RUN: <reason>` metadata where prebuilt PyPI wheels pin `torch<=2.14.1` (`cu124`, lacking `sm_120` Blackwell binaries) and source builds are blocked by missing `nvcc` / system OpenMPI without `sudo`.
- **Documentation Sync**: Updated `README.md`, `docs/RESULTS_RTX5090.md`, and `CHANGELOG.md` with measured Phases 9–13 (`v2.4.0 → v2.8.0`) data.

## v2.7.0 (2026-11-21) — Phase 12: Distributed GPU Runtime (TP, PP, EP, Distributed KV-Cache & Scheduler)

- **Multi-Node Communication Layer (`cutile_forge.distributed.runtime`)**: Added `CommTransport` protocol and `LoopbackCommTransport` supporting `all_reduce_sum`, `all_gather`, `reduce_scatter_sum`, `all_to_all`, and `send_recv` with byte-transfer telemetry across logical ranks and nodes.
- **Tensor, Pipeline & Expert Parallelism (`ColumnParallelLinear`, `RowParallelLinear`, `TensorParallelSwiGLUMLP`, `TensorParallelGQAAttention`, `PipelineParallelEngine`, `ExpertParallelMoE`)**: Implemented Megatron-style column/row TP, GPipe/1F1B micro-batch PP, and all-to-all token dispatch/combine EP, verified against single-device execution (`max_abs_diff == 0.0` for PP=4 and EP=4; `< 5e-3` for TP=2/4).
- **Distributed KV-Cache & Topology Scheduler (`DistributedPagedKVCache`, `DistributedScheduler`)**: Added 2D `(PP × TP)` KV-head/layer cache partitioning and greedy min-load `(DP × PP × TP)` request scheduling, with all 1-GPU benchmark metrics explicitly labeled `simulated` in `benchmarks/results_5090/phase12_distributed_simulated.csv`.

## v2.6.0 (2026-11-21) — Phase 11: Architecture Completeness (Llama, Qwen2.5, Mistral, MoE, Vision, Audio, LLaVA, Diffusion)

- **Expanded Architecture Registry & Unified IR (`cutile_forge.models`)**: Added `AudioConvEmbedStage` (Whisper 2-layer Conv1d stem + sinusoidal positional embeddings), `DiffusionPatchEmbedStage` (latent patchify + sinusoidal timestep MLP), `DiTBlock` (AdaLN-Zero `shift/scale/gate` modulation), `EncoderOutputStage`, `DiffusionOutputStage` (AdaLN + patch unpatchify), and `ep_partitions` single-GPU expert-parallel partitioning in `SparseMoEBlock`.
- **HuggingFace Checkpoint Weight Extractors (`checkpoint_loader.py`)**: Added `extract_hf_decoder_weights` (supporting Llama, Mistral, Qwen2/2.5 with QKV biases, and Mixtral 3D `MixtralExperts`), `extract_hf_siglip_vision_weights`, and `extract_hf_whisper_encoder_weights`, plus half-split vs interleaved RoPE selection (`rotary_style="half"`).
- **Real-Checkpoint & Synthetic Parity Verification (`benchmarks/results_5090/phase11_architecture_parity.csv`)**: Verified real-checkpoint parity (`cos >= 0.99996`) on `Mistral-7B-v0.1-GPTQ`, `Llama-2-70B-GPTQ`, `Mixtral-8x7B-GPTQ`, `Qwen/Qwen2.5-0.5B` (24L), `google/siglip-base-patch16-224` (12L), and `openai/whisper-tiny` (4L encoder), plus synthetic parity and bit-exact `EP=1 == EP=4` invariance across `Llama-3.1`, `DeepSeek-MoE`, `Qwen2-MoE`, `ViT-B/16`, `CLIP-ViT-L/14`, `LLaVA-1.5`, and `DiT-S/2`.

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

