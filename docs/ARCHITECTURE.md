# CuTileForge Architecture

CuTileForge compiles a transformer layer into a **Tile Fusion Graph (TFG)** and lowers it directly to NVIDIA **CuTile** Python DSL kernels (`cuda.tile`) with 3-stage software pipelining and Stage -1 PCIe DMA weight prefetching.

## System Pipeline

```
  HF 4-bit GPTQ Safetensors (D:\models\mistral_7b_4bit\)
        │
        ▼  unpack_gptq_layer() + desc_act g_idx row permutation
  Pinned DDR5 Host Memory / Resident 8 GB GPU HBM
        │
        ▼  Stage -1: WeightStreamer (async H2D copy_stream + cuda.Event)
  ┌───────────────────────────────────────────────────────────────┐
  │                 Tile Fusion Graph (TFG)                       │
  │  8 unfused ops ──(6 fusion rules)──▶ 3 fused CuTile kernels   │
  │   1. Fused QKV + GQA FlashAttention-2                         │
  │   2. Fused O-proj + RMSNorm + Residual                        │
  │   3. Fused FFN (Up + SiLU + Down) + Residual                  │
  └───────────────────────────────────────────────────────────────┘
        │
        ▼  tile_pipeliner (3-stage) + tile_autotuner (12 configs)
  CuTile Compiler (cuda.tile.compilation -> sm_89 cubin / SASS)
        │
        ▼  PagedKVCache (GPU HBM <-> Pinned DDR5 LRU eviction)
  Autoregressive & Speculative (gamma=4) Token Generation
```

## Key Components

- **`cutile_forge/forge/`**: TFG IR (`FusionGraph`, `TileNode`, `Edge`, `DataTensor`), 6 algebraic/tile fusion rules, 3-stage software pipeliner, Stage -1 PCIe DMA prefetcher, 12-config tile autotuner, SM-aware scheduler, CuTile source/cubin codegen, and Graphviz/SVG visualizer.
- **`cutile_forge/kernels/`**: Native `cuda.tile` kernels for INT4 dequantized GEMM, GEMM+SiLU, single-kernel FFN, FlashAttention-2 GQA (32:8), LayerNorm/RMSNorm + residual, RoPE, online softmax, and top-k/top-p sampling.
- **`cutile_forge/engine/`**: Mistral-7B 32-layer execution engine, 4-bit GPTQ loader, PagedKVCache with HBM budget eviction, prefill/decode loop, and speculative decoder.
- **`cutile_forge/baselines/`**: Hand-written CUDA C++ (`cp.async` + WMMA 3-stage GEMM, FlashAttention-2, LayerNorm, SiLU) and cuBLAS HGEMM pybind11 baselines.

