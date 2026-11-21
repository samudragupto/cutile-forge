# Quantization End-to-End Parity & Accuracy Report (`CuTileForge v3.1.0`)

> Source artifact: `benchmarks/validation_artifacts/v3.1.0/phase14e_quantization/manifest.json`

| Format | Source Kind | Disk Format | Kernel Format | Layers | Category | Greedy Match / 1st Div | Logit Cos Sim | PPL Native vs Ref (`diff%`) | Decode (`tok/s`) | Pack/Load (s) | VRAM (MB) |
|---|---|---|---|---:|---|---|---|---|---:|---:|---:|
| `GPTQ_on_disk_32L_real` | `real-on-disk-checkpoint` | `gptq_int4_gs128_desc_act` | `cutile_int4_dequant` | 32 | `END_TO_END_VERIFIED` | `2/2` | 1.0 | 6.7058 vs 6.7072 (`0.0211%`) | 167.96 | 7.84 | 5476.2 |
| `GPTQ_gs128_desc_repack` | `repacked-from-real-mistral-7b` | `repacked_gptq` | `cutile_gptq` | 8 | `LAYER_VERIFIED_ONLY` | `exact` | 0.999999 | 9429.3887 vs 9430.5488 (`0.0123%`) | 383.81 | 0.0 | 2193.6 |
| `GPTQ_gs128_nodesc` | `repacked-from-real-mistral-7b` | `repacked_gptq` | `cutile_gptq` | 8 | `LAYER_VERIFIED_ONLY` | `div@13` | 0.999999 | 9608.7939 vs 9610.7188 (`0.02%`) | 403.31 | 0.06 | 3106.1 |
| `GPTQ_gs64` | `repacked-from-real-mistral-7b` | `repacked_gptq` | `cutile_gptq` | 8 | `LAYER_VERIFIED_ONLY` | `exact` | 0.999999 | 9633.209 vs 9633.8984 (`0.0072%`) | 289.5 | 0.05 | 3173.6 |
| `GPTQ_gs32` | `repacked-from-real-mistral-7b` | `repacked_gptq` | `cutile_gptq` | 8 | `LAYER_VERIFIED_ONLY` | `exact` | 0.999999 | 9430.8096 vs 9430.9443 (`0.0014%`) | 264.62 | 0.05 | 3289.9 |
| `AWQ_gs128` | `repacked-from-real-mistral-7b` | `repacked_awq` | `cutile_awq` | 8 | `LAYER_VERIFIED_ONLY` | `div@13` | 0.999999 | 9608.7939 vs 9610.7188 (`0.02%`) | 238.67 | 0.12 | 3591.3 |
| `GGUF_Q4_0` | `repacked-from-real-mistral-7b` | `repacked_gguf_q4_0` | `cutile_gguf_q4_0` | 8 | `LAYER_VERIFIED_ONLY` | `exact` | 0.999999 | 9829.2139 vs 9826.0557 (`0.0321%`) | 284.35 | 10.39 | 4143.2 |
| `GGUF_Q8_0` | `repacked-from-real-mistral-7b` | `repacked_gguf_q8_0` | `cutile_gguf_q8_0` | 8 | `LAYER_VERIFIED_ONLY` | `exact` | 0.999999 | 9424.04 vs 9423.5996 (`0.0047%`) | 273.76 | 16.88 | 5746.9 |
| `GGUF_Q4_K` | `repacked-from-real-mistral-7b` | `repacked_gguf_q4_k` | `cutile_gguf_q4_k` | 8 | `LAYER_VERIFIED_ONLY` | `div@5` | 0.999998 | 10135.291 vs 10135.0781 (`0.0021%`) | 274.75 | 0.1 | 4440.9 |
| `GGUF_Q6_K` | `repacked-from-real-mistral-7b` | `repacked_gguf_q6_k` | `cutile_gguf_q6_k` | 8 | `LAYER_VERIFIED_ONLY` | `exact` | 0.999999 | 9504.4326 vs 9509.3096 (`0.0513%`) | 218.72 | 0.11 | 5599.9 |
| `GGUF_Q4_K_M_mixed` | `repacked-from-real-mistral-7b` | `repacked_gguf_q4_k_m` | `cutile_gguf_q4_k_m` | 8 | `LAYER_VERIFIED_ONLY` | `exact` | 0.999999 | 9397.8506 vs 9396.6318 (`0.013%`) | 278.8 | 0.1 | 4805.3 |
| `INT8_gs128` | `repacked-from-real-mistral-7b` | `repacked_int8` | `cutile_int8` | 8 | `LAYER_VERIFIED_ONLY` | `exact` | 0.999999 | 9437.2148 vs 9436.999 (`0.0023%`) | 347.18 | 0.03 | 3881.0 |
| `FP8_e4m3_tensor` | `repacked-from-real-mistral-7b` | `repacked_fp8` | `cutile_fp8` | 8 | `LAYER_VERIFIED_ONLY` | `div@7` | 0.998579 | 9349.7021 vs 9471.7754 (`1.2888%`) | 174.15 | 0.08 | 3740.3 |
| `FP8_e4m3_block128` | `repacked-from-real-mistral-7b` | `repacked_fp8_block` | `cutile_fp8_block` | 8 | `LAYER_VERIFIED_ONLY` | `exact` | 0.999999 | 9450.3555 vs 9449.8057 (`0.0058%`) | 290.18 | 0.02 | 3856.6 |
