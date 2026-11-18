# Tile Fusion Graph (TFG) Compiler

The TFG compiler transforms an unfused 8-node transformer layer graph into **3 fused CuTile kernels** through iterative rule application, software pipelining, and SM-aware scheduling.

## 6 Fusion Rules (`cutile_forge/forge/fusion_rules.py`)

1. **`fuse_dequant_into_gemm`**: Folds INT4 dequantization directly into the GEMM tile K-loop so 16-bit weights never materialize in HBM.
2. **`fuse_qkv_attention`**: Fuses `QKV_GEMM` with `ATTENTION_GQA` into `FUSED_QKV_ATTN`.
3. **`fuse_o_layernorm`**: Fuses `O_GEMM` with `LAYERNORM` / `RMSNORM` into `FUSED_O_LAYERNORM`.
4. **`fuse_gemm_silu`**: Fuses `FFN_UP_GEMM` (or `GEMM_DEQUANT`) with `SILU` into `GEMM_SILU`.
5. **`fuse_gemm_silu_down`**: Fuses `GEMM_SILU` with `FFN_DOWN_GEMM` into `FUSED_FFN`.
6. **`fuse_ffn_layernorm_residual`**: Fuses `FUSED_FFN` (or `RESIDUAL_ADD`) with trailing `LAYERNORM_RESIDUAL` into `FUSED_FFN_NORM`.

## Pipeline Stages

- **Stage -1 (`weight_streamer.py`)**: Async H2D PCIe DMA copy of layer `L+1` weights from pinned DDR5 into GPU HBM double-buffer slots via `copy_stream`.
- **Stages 0, 1, 2 (`tile_pipeliner.py`)**: Load -> Tensor Core MMA compute -> Epilogue store (`latency=3` in CuTile), bounded by the `100 KB / SM` shared memory budget on `sm_89`.

