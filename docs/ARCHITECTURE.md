# CuTileForge Architecture

CuTileForge compiles an entire transformer layer into a **Tile Fusion Graph (TFG)** and lowers it directly to NVIDIA **CuTile Python DSL** (`cuda.tile`) kernels with 3-stage software pipelining (`latency=3`) and Stage -1 PCIe DMA weight prefetching.

## End-to-End Compiler & Runtime Architecture

```mermaid
flowchart TD
    W["4-bit GPTQ Safetensors (D:\\models\\mistral_7b_4bit\\)"] --> L["WeightLoader (desc_act g_idx permutation)"]
    L --> DDR5["Pinned Host DDR5 / Resident 8 GB HBM"]
    DDR5 -->|"Stage -1: Async H2D Copy Stream"| WS["WeightStreamer (Double-Buffered DMA)"]

    G["Python TFG Definition (8 Unfused Nodes)"] --> R{"Apply 6 Fusion Rules"}
    R --> F["3 Fused CuTile Nodes (FUSED_QKV_ATTN, FUSED_O_LAYERNORM, FUSED_FFN_NORM)"]
    F --> P["Tile Pipeliner (3-Stage) + TileAutotuner (12 Configs)"]
    P --> C["CuTile Codegen (cuda.tile -> SASS sm_89)"]

    WS --> EXEC["RTX 4060 Laptop (24 SMs, sm_89)"]
    C --> EXEC
    KV["PagedKVCache (HBM <-> Pinned DDR5 LRU Eviction)"] <--> EXEC
    EXEC --> OUT["Autoregressive & Speculative (gamma=4) Token Generation"]
```

## PCIe Weight Streaming (Stage -1)

```mermaid
sequenceDiagram
    participant CPU as Ryzen 7 7840HS
    participant DDR5 as Pinned DDR5-5600
    participant DMA as PCIe 4.0 Copy Stream
    participant GPU as RTX 4060 Tensor Cores

    CPU->>DDR5: Pin 4-bit weight tiles
    DDR5->>DMA: cudaMemcpyAsync (Layer N+1)
    par Overlapped Execution
        GPU->>GPU: CuTile 3-stage HMMA (Layer N)
    and
        DMA->>GPU: Stage -1 H2D DMA (Layer N+1)
    end
    DMA-->>GPU: cudaEventSynchronize barrier
```

## TFG Fusion Before & After (8 Kernels to 3)

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
        F1["1. FUSED_QKV_ATTN (QKV + RoPE + FlashAttn-2)"]
        F2["2. FUSED_O_LAYERNORM (O-proj + Norm + Residual)"]
        F3["3. FUSED_FFN_NORM (Up + SiLU + Down + Norm)"]
        F1 --> F2 --> F3
    end
```

## Key Components

- **`cutile_forge/forge/`**: TFG IR (`FusionGraph`, `TileNode`, `Edge`, `DataTensor`), 6 algebraic/tile fusion rules (`fusion_rules.py`), 3-stage software pipeliner (`tile_pipeliner.py`), Stage -1 PCIe DMA prefetcher (`weight_streamer.py`), 12-config `TileAutotuner`, SM-aware scheduler (`scheduler.py`), CuTile source & `sm_89` cubin `codegen.py`, and Graphviz/SVG visualizer (`graph_viz.py`).
- **`cutile_forge/kernels/`**: Native `cuda.tile` kernels for INT4 dequantized GEMM, GEMM+SiLU, single-kernel FFN, FlashAttention-2 GQA (32:8), LayerNorm/RMSNorm + residual, RoPE, online softmax, and top-k/top-p sampling.
- **`cutile_forge/engine/`**: Mistral-7B 32-layer execution engine (`model.py`), 4-bit GPTQ loader with `desc_act` `g_idx` support (`weight_loader.py`), `PagedKVCache` with HBM budget eviction (`kv_cache.py`), prefill/decode loop (`inference.py`), and speculative decoder (`speculative.py`).
- **`cutile_forge/baselines/`**: Hand-written CUDA C++ (`cp.async` + WMMA 3-stage GEMM, FlashAttention-2, LayerNorm, SiLU) and cuBLAS HGEMM pybind11 baselines.
