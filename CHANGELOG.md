# Changelog

## v0.1.0 (2026-11-18)

- **TFG Compiler (`cutile_forge.forge`)**: `FusionGraph` IR, 6 fusion rules (8 -> 3 kernels per Mistral layer), 3-stage software pipelining, Stage -1 PCIe DMA `WeightStreamer`, 12-config `TileAutotuner`, `sm_89` CuTile source/cubin `codegen`, and Graphviz/SVG visualization.
- **CuTile Kernels (`cutile_forge.kernels`)**: Fused INT4-dequant GEMM, GEMM+SiLU, single-kernel FFN, FlashAttention-2 GQA (32:8), LayerNorm/RMSNorm + residual, RoPE, online softmax, and top-k/top-p sampling.
- **Inference Engine (`cutile_forge.engine`)**: 4-bit GPTQ loader with `desc_act` `g_idx` support, paged HBM/DDR5 `PagedKVCache`, `Mistral7B`, `InferenceEngine`, `SpeculativeDecoder`, and `TokenGenerator`.
- **CUDA C++ & cuBLAS Baselines (`cutile_forge.baselines`)**: Hand-written 3-stage `cp.async`+WMMA GEMM, separate INT4 dequant, FlashAttention-2, LayerNorm/RMSNorm, SiLU, and cuBLAS HGEMM pybind11 module.
- **Benchmarks & CI/CD**: Full micro and E2E benchmark suite, NVML/PCIe/thermal/overlap profilers, automated Markdown + matplotlib reporting, 76 unit + correctness tests, and GitHub Actions workflows.

