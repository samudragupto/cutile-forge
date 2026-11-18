# CuTileForge Architecture

CuTileForge compiles an entire transformer layer into a **Tile Fusion Graph (TFG)** and lowers it directly to NVIDIA **CuTile Python DSL** (`cuda.tile`) kernels with 3-stage software pipelining (`latency=3`) and Stage -1 PCIe DMA weight prefetching.

## TFG Compilation Flow

```mermaid
graph TD
    A[Python TFG Definition] --> B[FusionGraph]
    B --> C{Apply 6 Fusion Rules}
    C --> D[3 Fused Tile Nodes]
    D --> E[Tile Pipeliner: 3-stage + Stage-1 PCIe]
    E --> F[Codegen: CuTile DSL -> SASS sm_89]
    F --> G[Runtime: WeightStreamer + KV-Cache]
```

## PCIe Weight Streaming (Stage -1)

```mermaid
sequenceDiagram
    Ryzen 7->>DDR5: Pinned Weight Tiles
    Ryzen 7->>RTX 4060: cudaMemcpyAsync (PCIe 4.0)
    par Compute
        RTX 4060->>Tensor Cores: HMMA on Tile N
    end
    RTX 4060->>HBM: LDGSTS Tile N+1
```

## Fusion Before/After

```mermaid
graph LR
    subgraph Before[8 Kernels]
        K1[QKV GEMM]-->K2[Attn]-->K3[O GEMM]-->K4[LN]-->K5[FFNup]-->K6[SiLU]-->K7[FFNdown]-->K8[LN+Res]
    end
    subgraph After[3 Fused Kernels]
        F1[FUSED_QKV_ATTN]-->F2[FUSED_O_LAYERNORM]-->F3[FUSED_FFN_NORM]
    end
```

## Key Components

- **`cutile_forge/forge/`**: TFG IR (`FusionGraph`, `TileNode`, `Edge`, `DataTensor`), 6 algebraic/tile fusion rules (`fusion_rules.py`), 3-stage software pipeliner (`tile_pipeliner.py`), Stage -1 PCIe DMA prefetcher (`weight_streamer.py`), 12-config `TileAutotuner`, SM-aware scheduler (`scheduler.py`), CuTile source & `sm_89` cubin `codegen.py`, and Graphviz/SVG visualizer (`graph_viz.py`).
- **`cutile_forge/kernels/`**: Native `cuda.tile` kernels for INT4 dequantized GEMM, GEMM+SiLU, single-kernel FFN, FlashAttention-2 GQA (32:8), LayerNorm/RMSNorm + residual, RoPE, online softmax, and top-k/top-p sampling.
- **`cutile_forge/engine/`**: Mistral-7B 32-layer execution engine (`model.py`), 4-bit GPTQ loader with `desc_act` `g_idx` support (`weight_loader.py`), `PagedKVCache` with HBM budget eviction (`kv_cache.py`), prefill/decode loop (`inference.py`), and speculative decoder (`speculative.py`).
- **`cutile_forge/baselines/`**: Hand-written CUDA C++ (`cp.async` + WMMA 3-stage GEMM, FlashAttention-2, LayerNorm, SiLU) and cuBLAS HGEMM pybind11 baselines.
