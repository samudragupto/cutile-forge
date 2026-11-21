# CuTileForge Architecture (`v2.0.0`)

CuTileForge is a **general-purpose Tile-Fusion Graph (TFG) compiler, async GPU runtime, training/vision ecosystem, and 7B–70B LLM inference engine** targeting NVIDIA **CuTile Python DSL (`cuda.tile`)** across Ada (`sm_89`) and Blackwell (`sm_120`).

## End-to-End Compiler, Runtime & Serving Architecture

```mermaid
flowchart TD
    subgraph Frontends["Model & Graph Frontends"]
        direction TB
        F_SDK["SDK Graph Builder<br/>cutile_forge.sdk.Graph"]
        F_ONNX["ONNX Importer<br/>import_onnx_model"]
        F_FX["torch.compile FX Backend<br/>backend=cutile_forge (ShapeGuard + FXOpRegistry)"]
        F_LLM["LLM Engine<br/>Mistral-7B / Mixtral-8x7B / Llama-2-70B"]
    end

    F_SDK --> PM
    F_ONNX --> PM
    F_FX --> PM
    F_LLM --> PM

    subgraph Compiler["TFG Compiler & Autotuning"]
        direction TB
        PM["TFG PassManager<br/>OpRegistry + 6 Fusion Passes"]
        PASS2["LayoutStride / ConstantFold<br/>Profitability / MemoryLifetime"]
        TUNE["CuTileTune v2 + CuTileVision v2<br/>TuningDB + cutile-forge inspect"]
        PM --> PASS2 --> TUNE
    end

    TUNE --> KERNELS

    subgraph Execution["CuTileKernels & CuTileRuntime"]
        direction TB
        KERNELS["CuTile Kernels (cuda.tile 1.8.1)<br/>LLM Fusion / General Ops / Vision / Training+AdamW"]
        RT["CuTileRuntime<br/>PinnedHostPool + TensorStreamer + MemoryLifetimeAllocator"]
        EXEC["GPU Execution (sm_89 3-stage / sm_120 4-stage)<br/>CUDA Graphs / Persistent MoE / FlashDecoding Split-KV"]
        KERNELS --> EXEC
        RT <--> EXEC
    end

    EXEC --> SRV["OpenAI FastAPI Server<br/>DynamicBatcher + RadixPrefixCache + Bearer Auth + /metrics"]
```

## PCIe Weight Streaming (Stage -1 DMA)

```mermaid
sequenceDiagram
    participant CPU as "Host DDR5 (PinnedHostPool)"
    participant DMA as "Copy Stream (PCIe 4.0/5.0 Async DMA)"
    participant GPU as "Compute Stream (sm_89 / sm_120 Tensor Cores)"

    CPU->>DMA: "Stage Layer N+1 (non_blocking H2D into Slot B)"
    par Overlapped Execution
        GPU->>GPU: "Execute CuTile Kernels (Layer N, Slot A)"
    and
        DMA->>GPU: "Stage -1 H2D Prefetch (Layer N+1, Slot B)"
    end
    DMA-->>GPU: "torch.cuda.Event barrier before Layer N+1"
```

## TFG Fusion Before & After (`8 → 3` Transformer Layer Kernels)

```mermaid
flowchart TB
    subgraph Before["Before Fusion: 8 Separate Kernels"]
        direction TB
        K1["QKV GEMM"] --> K2["GQA Attention"]
        K2 --> K3["O GEMM"]
        K3 --> K4["LayerNorm"]
        K4 --> K5["FFN-up GEMM"]
        K5 --> K6["SiLU"]
        K6 --> K7["FFN-down GEMM"]
        K7 --> K8["LayerNorm + Residual"]
    end

    subgraph After["After TFG Passes: 3 Fused CuTile Kernels"]
        direction TB
        F1["1. FUSED_QKV_ATTN<br/>QKV + RoPE + FlashAttn-2"]
        F2["2. FUSED_O_LAYERNORM<br/>O-proj + Norm + Residual"]
        F3["3. FUSED_FFN_NORM<br/>Up + SiLU + Down + Norm"]
        F1 --> F2 --> F3
    end

    Before -->|"6 TFG Fusion Passes"| After
```

## Key Components

- **`cutile_forge/sdk/` & `cutile_forge/onnx_import.py`**: Imperative `Graph` builder, `Device`, `KernelRegistry`, `compile(...)`, ONNX importer (`import_onnx_model`, `import_graph_spec`), and `cutile_forge.doctor` environment diagnostics (`cutile-forge doctor`).
- **`cutile_forge/forge/`**: Typed `FusionGraph` IR, `OpRegistry` (`op_registry.py`), `PassManager` with 6 algebraic/tile fusion passes (`fusion_rules.py`) plus `LayoutStridePass`, `ConstantFoldPass`, `ProfitabilityPass`, and `MemoryLifetimePass` (`pass_manager.py`), occupancy-aware cost model, 3/4-stage tile pipeliner, persistent cubin/FX artifact cache, and the expanded `torch.compile(backend="cutile_forge")` FX lowering backend (`fx_backend.py`) with `FXOpRegistry` and `ShapeGuard`.
- **`cutile_forge/autotune/` & `cutile_forge/vision/`**: CuTileTune v2 (`TuningDB`, `shape_fingerprint`, benchmark regression tracking, `tune_bmm`, `tune_reduction`, `tune_transpose`, `tune_activation`) and CuTileVision v2 (`inspect_compiled_tfg`, HBM lifetime/savings estimation, execution timeline, `cutile-forge inspect` CLI).
- **`cutile_forge/kernels/`**: Native `cuda.tile` kernels spanning LLM inference (INT4/NVFP4/FP8 GEMM, GEMM+SiLU, single-kernel FFN, GQA FlashAttention-2, FlashDecoding split-KV, persistent 3D-grid grouped MoE, RMSNorm/LayerNorm+residual, RoPE, top-k/top-p sampling), general tensor ops (`bmm`, `reduce_sum`, `reduce_max`, `cumsum`, `transpose`, `gather`, `conv2d`), computer vision primitives (`vision.py`: depthwise conv2d, conv2d backward, max/avg pool2d, `CuTilePatchEmbed`), and training primitives (forward/backward SiLU, GELU, Softmax, RMSNorm, LayerNorm, Linear, BMM, `torch.autograd.Function` wrappers, `cutile_grad_clip_norm`, and `cutile_multi_tensor_adamw`).
- **`cutile_forge/runtime/`**: General async runtime (`PinnedHostPool`, double/triple-buffered `TensorStreamer` with optional `adaptive_slots`, `MemoryLifetimeAllocator`, `GraphRuntime`, and `p2p_capable`).
- **`cutile_forge/engine/` & `cutile_forge/server/`**: Multi-shard `.safetensors` mmap loader (`GPTQ`, `AWQ`, `GGUF Q4_K_M`), `HybridLayerBudgeter`, `PagedKVCache` (FP16/FP8 + async swap pool), `RadixPrefixCache`, `NGramPromptLookupDrafter`, and FastAPI OpenAI server (`/health`, `/metrics`, Bearer auth, `DynamicBatcher`, SSE streaming).
- **`cutile_forge/bench/`**: CUDA event benchmark harness (`Stats`, `RichBenchRecord`, `benchmark_with_telemetry`, `check_regression_gate`), NVML/PCIe/thermal/overlap profilers, and empirical Roofline plotter.
- **`cutile_forge/distributed/` *(single-node / prototype tested)***: `detect_topology`, `p2p_copy`, and deterministic `ring_allreduce_sum` — unit-tested on CPU CI and single-GPU RTX 5090; multi-GPU NVLink scaling awaits multi-GPU hardware validation.
- **`cutile_forge/baselines/`**: Hand-written CUDA C++ (`cp.async` + WMMA) and cuBLAS pybind11 reference baselines.
