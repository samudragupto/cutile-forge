# Regression Collection Report (`tier=fast`)

- **Commit:** `c78b8fa2968f6bf604c44240241b8e7b384c54fc` | **Hardware:** `NVIDIA GeForce RTX 5090 (sm_120)` | **Contended:** `False`

| Category | Model | Reference | Layers | Status | Key Parity / PPL Metrics | Notes |
|---|---|---|---:|---|---|---|
| Small decoder | Qwen2.5-0.5B | `transformers==5.23.0 (AutoModelForCausalLM)` | 24 | `NOT NATIVE` | cos=0.999998, diff=0.0400, greedy=1/1 (100.0% tok), ppl=12.941 vs 12.941 (492 tok) | UnifiedModelExecutor executes PyTorch matmul/softmax in _causal_attention -> NOT NATIVE |
| Vision | SigLIP ViT-B/16 | `transformers==5.23.0 (SiglipVisionModel)` | 12 | `FAIL` | cos=0.999978, diff=0.2031, FAIL:emb_cos=0.999978,max_diff=0.2031 | UnifiedModelExecutor runs PyTorch F.conv2d & F.scaled_dot_product_attention -> NOT NATIVE |
| Audio | Whisper-Tiny | `transformers==5.23.0 (WhisperModel.encoder)` | 4 | `FAIL` | cos=0.999991, diff=0.4316, FAIL:emb_cos=0.999991,max_diff=0.4316 | UnifiedModelExecutor runs PyTorch F.conv1d & F.scaled_dot_product_attention -> NOT NATIVE |
