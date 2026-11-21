# Regression Collection Report (`tier=full`)

- **Commit:** `c78b8fa2968f6bf604c44240241b8e7b384c54fc` | **Hardware:** `NVIDIA GeForce RTX 5090 (sm_120)` | **Contended:** `False`

| Category | Model | Reference | Layers | Status | Key Parity / PPL Metrics | Notes |
|---|---|---|---:|---|---|---|
| Dense decoder | Mistral-7B | `Mistral7B(backend='pytorch', GPTQ-32L)` | 32 | `FAIL` | cos=1.000000, diff=0.0244, greedy=4/5 (99.8% tok), ppl=5.147 vs 5.147 (512 tok), FAIL:cos=1.000000,greedy=4/5 | NativeModelRuntime zero-ATen CUDA Graph decode + SWA=4096 |
| Small decoder | Qwen2.5-0.5B | `transformers==5.23.0 (AutoModelForCausalLM)` | 24 | `FAIL` | cos=0.999998, diff=0.0400, greedy=4/5 (99.9% tok), ppl=12.941 vs 12.941 (492 tok), FAIL:cos=True,ppl=True,greedy=False | UnifiedModelExecutor executes PyTorch matmul/softmax in _causal_attention -> NOT NATIVE |
| MoE | Mixtral-8x7B | `Mistral7B(backend='pytorch', 32L)` | 32 | `FAIL` | cos=0.999994, diff=0.0547, greedy=0/5 (99.2% tok), ppl=4.344 vs 4.345 (512 tok), FAIL:cos=0.999994,greedy=0/5 | 32L HBM GPTQ; 8-col MoE router uses torch.matmul -> NOT NATIVE |
| Vision | SigLIP ViT-B/16 | `transformers==5.23.0 (SiglipVisionModel)` | 12 | `FAIL` | cos=0.999978, diff=0.2031, FAIL:emb_cos=0.999978,max_diff=0.2031 | UnifiedModelExecutor runs PyTorch F.conv2d & F.scaled_dot_product_attention -> NOT NATIVE |
| Audio | Whisper-Tiny | `transformers==5.23.0 (WhisperModel.encoder)` | 4 | `FAIL` | cos=0.999991, diff=0.4316, FAIL:emb_cos=0.999991,max_diff=0.4316 | UnifiedModelExecutor runs PyTorch F.conv1d & F.scaled_dot_product_attention -> NOT NATIVE |
| Multimodal | LLaVA-1.5 | `transformers==5.23.0 (LlavaForConditionalGeneration)` | 55 | `NOT NATIVE` | cos=1.000000, diff=0.0156, greedy=1/1 (100.0% tok) | UnifiedModelExecutor (23L CLIP + 2L GELU proj + 32L Llama) runs PyTorch attention -> NOT NATIVE |
| Diffusion | DiT-XL/2 | `diffusers==0.43.0 (DiTTransformer2DModel)` | 28 | `FAIL` | cos=0.997016, diff=0.0593, FAIL:cos=0.997016,max_diff=0.0593 | User-approved DiT-XL/2-256 replacement; runs PyTorch F.layer_norm & SDPA -> NOT NATIVE |
| Large decoder | Llama-2-70B | `Mistral7B(backend='pytorch', 80L)` | 80 | `FAIL` | cos=0.999998, diff=0.0254, greedy=4/5 (99.9% tok), ppl=4.815 vs 4.814 (512 tok), FAIL:cos=0.999998,greedy=4/5 | 80L GPTQ Stage -1 PCIe streaming (55 HBM + 25 DDR5) |
