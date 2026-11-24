# Phase 23 Native Architecture Completion & Dispatch Audit (`commit 788a343`)

- **Hardware:** `NVIDIA GeForce RTX 5090` | **Commit:** `788a34382c4d8936d1775e28b250cfcc614ed3bf`

| Model | Path | Total ATen Ops | Forbidden Compute Ops | Status |
|---|---|---:|---:|---|
| qwen2.5-0.5b | `prefill_seq32_paged_kv` | 6658 | 0 | NATIVE_VERIFIED (0 forbidden) |
| qwen2.5-0.5b | `cached_decode_seq1_paged_kv` | 2267 | 0 | NATIVE_VERIFIED (0 forbidden) |
| siglip-vit-b16 | `forward_pass` | 531 | 0 | NATIVE_VERIFIED (0 forbidden) |
| whisper-tiny | `forward_pass` | 210 | 0 | NATIVE_VERIFIED (0 forbidden) |
| dit-xl-2 | `denoiser_step` | 2219 | 0 | NATIVE_VERIFIED (0 forbidden) |
| llava-1.5 | `multimodal_prefill_paged_kv` | 122475 | 0 | NATIVE_VERIFIED (0 forbidden) |
| llava-1.5 | `cached_text_decode_seq1_paged_kv` | 5099 | 0 | NATIVE_VERIFIED (0 forbidden) |
| mixtral-8x7b | `prefill_seq13_paged_kv` | 6920 | 0 | NATIVE_VERIFIED (0 forbidden) |
| mixtral-8x7b | `decode_seq1_paged_kv` | 3115 | 0 | NATIVE_VERIFIED (0 forbidden) |
