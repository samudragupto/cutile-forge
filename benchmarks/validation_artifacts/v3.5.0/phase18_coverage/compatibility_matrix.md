# Phase 18 Model Coverage & Compatibility Matrix (`v3.5.0`)

- Commit: `f56f5db82c8010956c4936cb0f631b8c4b17a54d` | Contended: `False`

| Family | Modality | Checkpoint | Weight Source | Native CuTile | Status | Notes |
|---|---|---|---|---|---|---|
| `clip_vision` | vision | `openai/clip-vit-base-patch32` | `real-checkpoint` | `True` | **PASS** | 12L CLIP-ViT-B/32 native patch GEMM, pre/post LayerNorm, QuickGELU, FlashAttention |
| `deepseek_moe` | text | `synthetic-weights` | `synthetic-weights` | `True` | **SYNTHETIC-ONLY** | No <16 GB ungated FP16 checkpoint fits side-by-side in 32 GB VRAM for reference comparison |
| `dit` | diffusion | `facebook/DiT-XL-2-256` | `real-checkpoint` | `True` | **FAIL** | User-approved DiT-XL/2-256; UnifiedModelExecutor executes native CuTile LayerNorm, patch GEMM, d=72 FlashAttention |
| `llama` | text | `/var/tmp/arrhat_models/llama_70b_4bit` | `real-checkpoint` | `True` | **FAIL** | 80L GPTQ Stage -1 PCIe streaming (55 HBM + 25 DDR5) |
| `llama3` | text | `unsloth/Llama-3.2-1B-Instruct` | `real-checkpoint` | `True` | **PASS** | 16L scaled Llama-3 RoPE all-layer native CuTile execution |
| `llava` | multimodal | `llava-hf/llava-1.5-7b-hf` | `real-checkpoint` | `True` | **PASS** | UnifiedModelExecutor executes native CuTile CLIP ViT + QuickGELU + 2L GELU proj + 32L Llama attention/MLP |
| `mistral` | text | `/home/arrhat/models/mistral_7b_4bit` | `real-checkpoint` | `True` | **FAIL** | NativeModelRuntime zero-ATen CUDA Graph decode + SWA=4096 |
| `mixtral` | text | `/var/tmp/arrhat_models/mixtral_8x7b_4bit` | `real-checkpoint` | `True` | **FAIL** | 32L HBM GPTQ + native CuTile cutile_moe_router_topk |
| `qwen2` | text | `Qwen/Qwen2.5-0.5B` | `real-checkpoint` | `True` | **FAIL** | UnifiedModelExecutor executes native CuTile FlashAttention + RoPE + GEMM + RMSNorm + SiLU |
| `qwen2_moe` | text | `synthetic-weights` | `synthetic-weights` | `True` | **SYNTHETIC-ONLY** | No <16 GB ungated FP16 checkpoint fits side-by-side in 32 GB VRAM for reference comparison |
| `siglip_vision` | vision | `google/siglip-base-patch16-224` | `real-checkpoint` | `True` | **FAIL** | UnifiedModelExecutor executes native CuTile patch GEMM, LayerNorm, bidirectional FlashAttention, GELU |
| `vit` | vision | `google/vit-base-patch16-224` | `real-checkpoint` | `True` | **PASS** | 12L ViT-B/16 native patch GEMM, LayerNorm, bidirectional FlashAttention, exact erf-GELU |
| `whisper_encoder` | audio | `openai/whisper-tiny` | `real-checkpoint` | `True` | **FAIL** | UnifiedModelExecutor executes native CuTile Conv1d(via cutile_conv2d), LayerNorm, bidirectional FlashAttention, GELU |
