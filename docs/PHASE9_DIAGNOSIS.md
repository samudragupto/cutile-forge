# Phase 9 Step 9.0 — Native Runtime Baseline & Per-Token Bottleneck Diagnosis

**Checkpoint:** `/home/arrhat/models/mistral_7b_4bit` (`Mistral-7B-v0.1-GPTQ`, 4-bit, `group_size=128`, `desc_act=True`, 32 layers)  
**Hardware:** NVIDIA GeForce RTX 5090 (`sm_120`, 170 SMs, 32 GB GDDR7 @ 1,792 GB/s, CUDA 13.2, `cuda-tile 1.8.1`)  
**Label:** `real-checkpoint`  
**Pre-benchmark GPU contention check (`nvidia-smi --query-compute-apps=pid,used_memory --format=csv`):** `none` (`contended = False`)  
**Benchmark Script & Data:** [`benchmarks/e2e/bench_phase9_baseline.py`](../benchmarks/e2e/bench_phase9_baseline.py), [`benchmarks/results_5090/phase9_baseline.csv`](../benchmarks/results_5090/phase9_baseline.csv), [`benchmarks/results_5090/phase9_diagnosis.json`](../benchmarks/results_5090/phase9_diagnosis.json)

---

## 1. Unified 256-Token Greedy Decode Baseline (5 Runs, Same Session)

Prompt: `[1, 415, 5565, 302, 4843, 349]` (`<s> The capital of France is`), `max_new_tokens=256`, `temperature=0.0`, `kv_dtype="fp16"`, 16-token warmup before 5 timed 256-token runs:

| Mode | Label | Decode `tok/s` Median (Min – Max) | `ms/token` Median (Min – Max) | Prefill `ms` Median (Min – Max) | Peak VRAM (`GB`) | 256-Token Greedy Parity | Contended |
|---|---|---:|---:|---:|---:|---|---|
| (a) `legacy_InferenceEngine_cuda_graph` | `real-checkpoint` | **99.01** (97.40 – 99.02) | **10.10** (10.10 – 10.27) | **11.07** (11.03 – 11.54) | 3.99 | `True` (reference) | `False` |
| (b) `legacy_InferenceEngine_eager` | `real-checkpoint` | **100.05** (100.02 – 100.07) | **10.00** (9.99 – 10.00) | **10.93** (10.61 – 10.98) | 4.06 | `True` (256/256) | `False` |
| (c1) `NativeModelRuntime_v230_default` (partial graph) | `real-checkpoint` | **103.22** (101.58 – 103.24) | **9.69** (9.69 – 9.84) | **10.14** (10.13 – 11.03) | 4.28 | `True` (256/256) | `False` |
| (c2) `NativeModelRuntime_v230_eager` (`use_cuda_graph=False`) | `real-checkpoint` | **103.67** (103.65 – 103.68) | **9.65** (9.65 – 9.65) | **10.11** (10.06 – 10.11) | 4.53 | `True` (256/256) | `False` |

### Why the `v2.2.0` 32-token single-run figures (`87.23` vs `66.85 tok/s`) differed
In `benchmarks/e2e/bench_track_b_native.py`, warmup ran only `max_new_tokens=8` (positions `6..13`, inside the first 16-token KV tile `tile_s=16`), followed by a single 32-token timed run. When sequence length crossed `pos=16` and `pos=32` during the timed window, `cuda-tile` compiled new grid/tile specializations on the fly. With a 16-token warmup and 256-token steady-state decode over 5 runs, `NativeModelRuntime` (`v2.3.0`) runs at **`103.22 tok/s`** (`9.69 ms/tok`) vs **`99.01 tok/s`** (`10.10 ms/tok`) on legacy `InferenceEngine`. These 5-run 256-token measurements replace `87.23` and `66.85` as the Phase 9 reference baseline.

---

## 2. Per-Token Execution & Time Breakdown

Measured over 32 steady-state decode steps after warmup (`phase9_diagnosis.json`):

| Metric | (a) Legacy `InferenceEngine` (CUDA Graph) | (b) Legacy `InferenceEngine` (Eager) | (c1) `NativeModelRuntime` (`v2.3.0` Partial Graph) | (c2) `NativeModelRuntime` (`v2.3.0` Eager) |
|---|---:|---:|---:|---:|
| Eager `ct.launch` calls / token | 32.0 (+ 66 graph replays) | 385.0 | **164.0** (+ 64 graph replays) | **740.0** |
| CUDA tensor allocations / token | 100.0 | 776.0 | **103.0** | **936.0** |
| GPU CUDA-event time (`ms/token`) | 10.081 | 9.835 | **9.617** | **9.517** |
| Wall-clock time (`ms/token`) | 10.082 | 9.836 | **9.617** | **9.517** |
| Host dispatch time (`ms/token`) | 8.852 | 9.439 | **8.576** | **9.208** |
| Python CPU overhead (`cProfile` `ms/token`) | 8.002 | 10.191 | **7.349** | **13.287** |
| Host syncs / token in `generate()` | 1 (`next_tok.item()`) | 1 (`next_tok.item()`) | **1 (`next_tok.item()`)** | **1 (`next_tok.item()`)** |

### Per-Component GPU Time Inside `NativeModelRuntime` (`v2.3.0` Partial Graph)

| Component (32 Transformer Layers + LM Head) | Isolated GPU Event Time (`ms/token`) | Share of Step | What Executes |
|---|---:|---:|---|
| `_qkv_graphs[0..31]` (32 replays) | **3.143 ms** | 26.9% | `32×` (`cutile_rmsnorm` + 3× `cutile_col_permute` + 3× `fused_gemm_dequant` + 3× `.copy_`) |
| `_mlp_graphs[0..31]` (32 replays) | **6.727 ms** | 57.5% | `32×` (O-proj + `fused_rmsnorm_residual` + Gate/Up/Down INT4 GEMMs + `ew_mul`/`ew_add` + `.copy_`) |
| Eager RoPE + KV Write/Gather + GQA Attention (`32×`) | **1.661 ms** | 14.2% | `64×` `apply_rope` + `32×` `slots.fill_` + `32×` `kv_write` + `32×` full-history `kv_gather` + `32×` `attention_gqa` |
| Eager `cutile_lm_head_fused` (`1×`) | **0.172 ms** | 1.5% | `cutile_rmsnorm` + `lm_head_tile_argmax_kernel` + `lm_head_reduce_argmax_kernel` |

---

## 3. Diagnosis of the Two Bottlenecks & Phase 9 Engineering Plan

1. **Bottleneck 1 — Fragmented Partial CUDA Graph (64 Subgraphs + 164 Eager Launches + 103 Allocations + 160 D2D `.copy_()` per Token):**
   - `NativeModelRuntime` in `v2.3.0` did **not** capture a full-step CUDA Graph. Because `apply_rope`, `NativePagedKVCache.append_and_get`, `fused_attention_gqa`, `cutile_embedding_lookup`, and `cutile_lm_head_fused` took Python `int` position arguments (`pos_offset`, `end_pos`) and allocated output buffers dynamically (`torch.empty_like`), it broke the decode step into **64 separate graph replays per token** interleaved with **164 eager kernel launches**, **103 tensor allocations**, and **160 D2D `.copy_()` staging calls**.
   - Furthermore, `NativePagedKVCache.append_and_get` launched `cutile_kv_cache_gather_kernel` on every layer to copy the entire KV cache history (`[0..end_pos]`) into a scratchpad buffer before attention, even when physical blocks are linearly mapped.
   - Because of the 64 graph-launch transitions and 160 `.copy_()` calls, `NativeModelRuntime(use_cuda_graph=True)` (`103.22 tok/s`) was actually slightly **slower** than `NativeModelRuntime(use_cuda_graph=False)` (`103.67 tok/s`)!

2. **Bottleneck 2 — GPU Kernel Compute/Memory Under-Occupancy on `M=1` INT4 Dequant GEMV (`9.62 ms` GPU time vs `2.12 ms` physical GDDR7 roofline):**
   - Notice that `gpu_event_ms_per_token` (`9.617 ms`) equals `wall_ms_per_token` (`9.617 ms`): the GPU execution stream itself takes `9.6 ms/token`, of which **84.4%** (`3.143 + 6.727 ms`) is spent inside the 32 layers' `M=1` INT4 projections (`q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj`).
   - Why? `_select_gemm_tiles(m=1)` uses `(tm=16, tn=64, tk=64)` with `split_k=1`:
     - `k_proj` / `v_proj` (`N=1024, K=4096`): launches only **`16` CTAs** on a **170-SM** RTX 5090 (**90.6% of SMs idle**).
     - `q_proj` / `o_proj` (`N=4096, K=4096`) and `down_proj` (`N=4096, K=14336`): launches only **`64` CTAs** on **170 SMs** (**62.4% of SMs idle**, with each `down_proj` CTA looping sequentially over `224` K-tiles!).
     - `gate_proj` / `up_proj` (`N=14336, K=4096`): launches `224` CTAs (1.32 waves, leaving tail wave 68% idle).

### Action Plan for Step 9.1 & Step 9.2
1. **True Full-Step Zero-ATen CUDA Graph Capture (Step 9.1):**
   - Make `apply_rope`, `NativePagedKVCache` in-place write (`(B, H_kv, MaxSeq, D)` preallocated pool or direct indexed write reading `pos_ptr`), and `attention_gqa_decode_pos_kernel` read the current token position from a **device-side `int32` tensor (`pos_buf`)** and write outputs into **preallocated static buffers** (`out=...`).
   - Capture the **entire 32-layer decode step** (from `cutile_embedding_lookup` through all 32 layers of `RMSNorm -> QKV -> RoPE -> KV Write -> GQA Decode Attention -> O_proj -> RMSNorm+Residual -> Gate+Up+Down` to `cutile_lm_head_fused` writing `_static_next_token` and incrementing `pos_buf` on device) inside **one single `torch.cuda.CUDAGraph`** per context bucket (`0` eager `ct.launch` calls, `0` allocations, `0` forbidden ATen calls during replay!).
2. **Eliminate Redundant KV Gather & Optimize `M=1` INT4 GEMV Occupancy (Step 9.1 / 9.2):**
   - Store the single-sequence/batch KV cache in head-major layout `(B, H_kv, max_seq_len, D)` (or paged table directly indexed by attention) so decode writes 1 slot in-place and attention reads directly up to `pos_buf + 1` with **zero per-layer `kv_gather` copy**!
   - Add Split-K / narrow-N `M=1` INT4 dequant GEMV kernels (`gemm_dequant_splitk_kernel`, `gemm_dequant_silu_splitk_kernel`) so `N=1024`, `N=4096`, and `K=14336` saturate all 170 SMs on the RTX 5090.
3. **Chained Multi-Step Replay, Continuous Batching, Prefill/Decode Scheduling, Speculative Decoding & Long-Context Paged KV (Step 9.2):**
   - Chained `N`-step graph replay on device without per-token `.item()` host sync (async host read every `N` steps).
   - Chunked prefill / decode interleave scheduler, batch-bucketed continuous batching (`B in {1, 2, 4, 8}`), native speculative decoding with KV rollback, and 8K/16K/32K Split-KV FlashDecoding.

---

## 4. Step 9.1 & Step 9.2 Measured Results (`benchmarks/results_5090/phase9_native_runtime.csv`)

### Capture Mechanism & Context-Length Strategy Decision
- **Capture Mechanism:** `torch.cuda.CUDAGraph` stream capture around `_execute_decode_step_in_workspace`. Chosen over raw `cuda.bindings` stream capture because `ct.launch(torch.cuda.current_stream(), ...)` enqueues CuTile kernels directly onto PyTorch's active capture stream while sharing the CUDA caching allocator pool for static workspace buffers, with zero `TorchDispatchMode` ATen compute calls during replay (`0` forbidden ATen calls, `0` eager `ct.launch` calls).
- **Data-Driven Comparison of Context-Length Strategies (256 tokens, 5 runs, `sync_interval=32`):**
  - `graph_strategy="bucketed"` (`512 / 1024 / 2048 / 4096` buckets + device-side `dynamic_loop=True`): **`171.60 tok/s` median (`169.31 – 171.66 tok/s`)**
  - `graph_strategy="device_masked"` (single fixed-ceiling `2048` graph + `dynamic_loop=False` masking): **`148.90 tok/s` median (`146.53 – 149.07 tok/s`)**
  - **Decision:** `graph_strategy="bucketed"` is selected as the default (`+15.2%` faster) because dynamic loop bounds bounded by the active bucket avoid loading unused KV tiles beyond `cur_pos + 1`.
- **Fallback:** Every single kernel in the 32-layer decode step (embedding gather, RMSNorm, column permute, INT4 dequant GEMM, fused RoPE + paged KV write, GQA decode attention, SwiGLU, residual add, fused LM head argmax, and device step advance) is 100% captured in one `torch.cuda.CUDAGraph`. Eager fallback is used only when layers are offloaded to host DDR5 (`not self._all_layers_on_gpu`) or when `use_cuda_graph=False`.

