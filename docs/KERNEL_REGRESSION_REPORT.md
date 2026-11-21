# Kernel & Operator Regression Audit (`CuTileForge v3.1.0`)

Honest accounting of both accelerated and degraded/slower CuTile kernels relative to PyTorch eager (`cuBLAS` / `cuDNN`) on RTX 5090 (`sm_120`).

## Where CuTileForge Is Faster (`track_a_before_after.csv`, `general_ops.csv`)

| Operation | Shape | CuTile (ms) | PyTorch (ms) | Speedup | Bandwidth (GB/s) | TFLOPS |
|---|---|---:|---:|---:|---:|---:|
| `patch_embed` | `16x3x224x224,p=16,d=768` | 0.0316 | 0.1264 | **4.01×** | 305.33 | 117.247 |
| `conv2d_1x1` | `16x64x56x56->128` | 0.0172 | 0.0254 | **1.48×** | 1121.25 | 47.84 |
| `fused_rmsnorm_silu` | `1024x4096` | 0.0155 | 0.0478 | **3.08×** | 1082.12 | 0.0 |
| `fused_add_rmsnorm_silu` | `1024x4096` | 0.0193 | 0.0518 | **2.69×** | 1306.37 | 0.0 |
| `bmm` (`general_ops`) | `16x128x2048x2048` | 0.1195 | 0.2249 | **1.88×** | — | — |
| `cumsum` (`general_ops`) | `256x2048` | 0.015 | 0.0172 | **1.15×** | — | — |
| `cumsum` (`general_ops`) | `4096x4096` | 0.0488 | 0.1 | **2.05×** | — | — |
| `cumsum` (`general_ops`) | `16384x1024` | 0.0529 | 0.0631 | **1.19×** | — | — |

## Where CuTileForge Is Slower (`< 1.0×` vs. PyTorch cuBLAS / cuDNN)

| Operation | Shape | CuTile (ms) | PyTorch (ms) | Speedup | Diagnosis |
|---|---|---:|---:|---:|---|
| `depthwise_conv2d` | `16x64x56x56,k=3` | 0.0315 | 0.0125 | **0.4×** | cuBLAS/cuDNN tensor-core or pointwise dispatch retains lower overhead |
| `max_pool2d` | `16x64x56x56,k=2` | 0.0143 | 0.01 | **0.7×** | cuBLAS/cuDNN tensor-core or pointwise dispatch retains lower overhead |
| `avg_pool2d` | `16x64x56x56,k=2` | 0.0145 | 0.0111 | **0.76×** | cuBLAS/cuDNN tensor-core or pointwise dispatch retains lower overhead |
| `conv2d_3x3` | `16x64x56x56->128` | 0.1368 | 0.0692 | **0.51×** | cuBLAS/cuDNN tensor-core or pointwise dispatch retains lower overhead |
| `gather` | `1024x4096` | 0.0136 | 0.0096 | **0.71×** | cuBLAS/cuDNN tensor-core or pointwise dispatch retains lower overhead |
| `silu` | `1024x4096` | 0.0142 | 0.0074 | **0.52×** | cuBLAS/cuDNN tensor-core or pointwise dispatch retains lower overhead |
| `gelu` | `1024x4096` | 0.0145 | 0.0073 | **0.5×** | cuBLAS/cuDNN tensor-core or pointwise dispatch retains lower overhead |
| `reduce_sum` | `4096x4096` | 0.0162 | 0.01 | **0.62×** | cuBLAS/cuDNN tensor-core or pointwise dispatch retains lower overhead |
| `reduce_max` | `4096x4096` | 0.0194 | 0.0152 | **0.78×** | cuBLAS/cuDNN tensor-core or pointwise dispatch retains lower overhead |
| `fused_layernorm_gelu` | `1024x4096` | 0.016 | 0.0152 | **0.95×** | cuBLAS/cuDNN tensor-core or pointwise dispatch retains lower overhead |
| `gemm_skinny_m4` | `4x4096x4096` | 0.0219 | 0.017 | **0.77×** | cuBLAS/cuDNN tensor-core or pointwise dispatch retains lower overhead |
| `gemm_medium_m64` | `64x4096x4096` | 0.0674 | 0.0191 | **0.28×** | cuBLAS/cuDNN tensor-core or pointwise dispatch retains lower overhead |
| `gemm_large_m1024` | `1024x4096x4096` | 0.2282 | 0.2208 | **0.97×** | cuBLAS/cuDNN tensor-core or pointwise dispatch retains lower overhead |
