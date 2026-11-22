# Regression Collection Report (`tier=full`)

- **Commit:** `7c250021b764d30df62126e84486feac4db39e17` | **Hardware:** `NVIDIA GeForce RTX 5090 (sm_120)` | **Contended:** `False`

| Category | Model | Reference | Layers | Status | Key Parity / PPL Metrics | Notes |
|---|---|---|---:|---|---|---|
| Dense decoder | Mistral-7B | `Mistral7B(backend='pytorch', GPTQ-32L)` | 32 | `FAIL` | cos=1.000000, diff=0.0244, greedy=4/5 (99.8% tok), ppl=5.147 vs 5.147 (512 tok), FAIL: greedy_exact_matches=4.0 (== 5.0) | NativeModelRuntime zero-ATen CUDA Graph decode + SWA=4096 |
| Small decoder | Qwen2.5-0.5B | `transformers==5.23.0 (AutoModelForCausalLM)` | 24 | `FAIL` | cos=0.999997, diff=0.0366, greedy=4/5 (99.9% tok), ppl=12.938 vs 12.941 (492 tok), FAIL: greedy_exact_matches=4.0 (== 5.0) | UnifiedModelExecutor executes native CuTile FlashAttention + RoPE + GEMM + RMSNorm + SiLU |
| MoE | Mixtral-8x7B | `Mistral7B(backend='pytorch', 32L)` | 32 | `FAIL` | cos=0.999988, diff=0.0771, greedy=3/5 (99.8% tok), ppl=4.343 vs 4.343 (512 tok), FAIL: greedy_exact_matches=3.0 (== 5.0) | 32L HBM GPTQ + native CuTile cutile_moe_router_topk |
| Vision | SigLIP ViT-B/16 | `transformers==5.23.0 (SiglipVisionModel)` | 12 | `FAIL` | cos=0.999979, diff=0.1250, FAIL: max_abs_diff=0.125 (< 0.05) | UnifiedModelExecutor executes native CuTile patch GEMM, LayerNorm, bidirectional FlashAttention, GELU |
| Audio | Whisper-Tiny | `transformers==5.23.0 (WhisperModel.encoder)` | 4 | `FAIL` | cos=0.999996, diff=0.3135, FAIL: max_abs_diff=0.3134765625 (< 0.1) | UnifiedModelExecutor executes native CuTile Conv1d(via cutile_conv2d), LayerNorm, bidirectional FlashAttention, GELU |
| Multimodal | LLaVA-1.5 | `transformers==5.23.0 (LlavaForConditionalGeneration)` | 55 | `PASS` | cos=1.000000, diff=0.0156, greedy=1/1 (100.0% tok) | UnifiedModelExecutor executes native CuTile CLIP ViT + QuickGELU + 2L GELU proj + 32L Llama attention/MLP |
| Diffusion | DiT-XL/2 | `diffusers==0.43.0 (DiTTransformer2DModel)` | 28 | `PASS` | cos=0.999999, diff=0.0044 | User-approved DiT-XL/2-256; UnifiedModelExecutor executes native CuTile LayerNorm, patch GEMM, d=72 FlashAttention |
| Large decoder | Llama-2-70B | `Mistral7B(backend='pytorch', 80L)` | 80 | `FAIL` | cos=0.999998, diff=0.0254, greedy=4/5 (99.9% tok), ppl=4.815 vs 4.814 (512 tok), FAIL: greedy_exact_matches=4.0 (== 5.0) | 80L GPTQ Stage -1 PCIe streaming (55 HBM + 25 DDR5) |
