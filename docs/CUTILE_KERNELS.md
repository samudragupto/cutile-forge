# CuTile Kernels Reference

All kernels in `cutile_forge/kernels/` are written in NVIDIA's `cuda.tile` Python DSL and compiled for `sm_89` (RTX 4060 Laptop).

| File | Kernel | Fusion / Optimization |
| --- | --- | --- |
| `gemm_dequant.py` | `gemm_fp16_kernel`, `gemm_dequant_kernel` | 2D L2 swizzle, 3-stage `latency=3`, in-register INT4 nibble unpacking |
| `gemm_dequant_silu.py` | `gemm_fp16_silu_kernel`, `gemm_dequant_silu_kernel` | Fused SiLU epilogue in FP32 accumulators before HBM store |
| `ffn_fused.py` | `ffn_up_silu_down_kernel`, `ffn_swiglu_kernel`, `ffn_dequant_up_silu_down_kernel` | Fuses Up-proj + SiLU + Down-proj with adaptive 8 GB VRAM dispatch |
| `attention_gqa.py` | `attention_gqa_kernel` | FlashAttention-2 online softmax with 32:8 GQA head sharing & causal mask |
| `layernorm.py` | `layernorm_kernel` | Single-pass row-wise LayerNorm in FP32 accumulation |
| `rmsnorm.py` | `rmsnorm_kernel` | Single-pass Mistral/LLaMA RMSNorm |
| `layernorm_residual.py` | `layernorm_residual_kernel`, `rmsnorm_residual_kernel` | Fused `x + residual` add + norm in one tile pass |
| `rope.py` | `rope_kernel` | Half-split Rotary Position Embedding |
| `softmax.py` | `online_softmax_kernel` | Tiled running-max / running-sum softmax for 32K vocab |
| `sampling.py` | `sample_tokens` | Greedy argmax, temperature, top-k, and top-p nucleus sampling |

