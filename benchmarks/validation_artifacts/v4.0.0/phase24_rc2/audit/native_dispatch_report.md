# Phase 24 Native Execution & Dispatch Audit (`v4.0.0` Candidate)

- **Hardware:** `NVIDIA GeForce RTX 5090 (sm_120)` | **Commit:** `406660fd46ac`
- **Scope Note:** Zero **forbidden ATen compute ops** across 11 audited real-checkpoint execution paths (9 baseline paths from v3.9.0 + `NativeFP16DecoderRuntime` prefill & eager decode on Qwen2.5-0.5B). Mixtral-8x7B M=1 decode now runs zero-sync indexed MoE (`0` routing host syncs / `aten._local_scalar_dense.default`).

| Model | Path | Total ATen Ops | Forbidden Compute Ops | Status |
|---|---|---:|---:|---|
| qwen2.5-0.5b | `prefill_seq12_paged_kv` | 3826 | 0 | NATIVE_VERIFIED (0 forbidden compute) |
| qwen2.5-0.5b | `cached_decode_seq1_paged_kv` | 2122 | 0 | NATIVE_VERIFIED (0 forbidden compute) |
| qwen2.5-0.5b | `native_fp16_prefill_seq12` | 1484 | 0 | NATIVE_VERIFIED (0 forbidden compute) |
| qwen2.5-0.5b | `native_fp16_decode_seq1_eager` | 1339 | 0 | NATIVE_VERIFIED (0 forbidden compute) |
| siglip-vit-b16 | `forward_pass` | 531 | 0 | NATIVE_VERIFIED (0 forbidden compute) |
| whisper-tiny | `forward_pass` | 210 | 0 | NATIVE_VERIFIED (0 forbidden compute) |
| dit-xl-2 | `denoiser_step` | 2219 | 0 | NATIVE_VERIFIED (0 forbidden compute) |
| llava-1.5 | `multimodal_prefill_paged_kv` | 122475 | 0 | NATIVE_VERIFIED (0 forbidden compute) |
| llava-1.5 | `cached_text_decode_seq1_paged_kv` | 5098 | 0 | NATIVE_VERIFIED (0 forbidden compute) |
| mixtral-8x7b | `prefill_seq13_paged_kv` | 7522 | 0 | NATIVE_VERIFIED (0 forbidden compute) |
| mixtral-8x7b | `decode_seq1_paged_kv` | 1840 | 0 | NATIVE_VERIFIED (0 forbidden compute) |
