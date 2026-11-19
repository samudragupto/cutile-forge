# I Fused 8 CUDA Kernels into 3 Python Tiles on an 8 GB Laptop GPU

**Authors:** Arrhat Nag (`@samudragupto`), Rupesh Nasre  

Running a 7-billion-parameter language model locally on an 8 GB laptop GPU usually forces an uncomfortable compromise: either you accept fragmented PyTorch eager execution with dozens of tiny kernel launches per layer, or you drop down to hundreds of lines of hand-tuned CUDA C++ templates (`cp.async`, shared-memory bank padding, warp-matrix fragments, and custom epilogues) that break the moment you want to fuse a new activation or quantization scheme.

I built **[CuTileForge](https://github.com/samudragupto/cutile-forge)** (`v0.1.0`) to test a different path: **can a high-level Tile Fusion Graph (TFG) compiler written in Python lower an entire transformer layer directly into NVIDIA's CuTile Python DSL (`cuda.tile`), match or beat hand-written CUDA C++ baselines, and sustain >30 tok/s on a real 8 GB laptop GPU without thermal collapse?**

On my **Lenovo LOQ 15APH8 (`82XT`)**—equipped with an **AMD Ryzen 7 7840HS** (8C/16T), **16 GB DDR5-5600** (`2×8 GB` Micron, ~89.6 GB/s dual-channel), and an **NVIDIA GeForce RTX 4060 Laptop GPU** (`AD107`, `sm_89`, 24 SMs, 8 GB GDDR6 on a 128-bit bus, ~272 GB/s) wired over **PCIe 4.0 x8**—the answer is **yes**.

---

## 1. The 8 GB Laptop Bottleneck

When you serve `Mistral-7B-v0.1` in 4-bit GPTQ (`TheBloke/Mistral-7B-v0.1-GPTQ`, **4.16 GB** on disk at `D:\models\mistral_7b_4bit\`), half of your 8 GB VRAM is already spoken for by weights before the first token is generated. The remaining ~3.8 GB has to cover:

1. **KV-cache growth** across long contexts (up to 32K tokens).
2. **Intermediate activations**—especially in the SwiGLU FFN, where the hidden dimension expands from `4096` to `14336`.
3. **Kernel launch and HBM round-trip overhead** during memory-bound autoregressive decode (`M = 1`).

In a standard unfused transformer layer, execution bounces back and forth between global HBM and the SMs across **8 separate kernels**:

```
QKV GEMM -> Attention -> O GEMM -> LayerNorm -> FFN-up GEMM -> SiLU -> FFN-down GEMM -> LayerNorm+Residual
```

Every arrow between those 8 kernels is a full tensor write to GDDR6 followed by a full tensor read by the next kernel. On a 128-bit memory bus (~272 GB/s), those intermediate HBM round-trips and 4-bit-to-FP16 dequantization buffers dominate wall-clock time.

And there was a second surprise when benchmarking against the latest **vLLM (`0.34.0`)** on the exact same checkpoint: `vllm==0.34.0` outright rejects `TheBloke/Mistral-7B-v0.1-GPTQ` with a `VllmConfig` `ValidationError` because `desc_act=True` (GPTQ activation-order `g_idx` permutation) is no longer supported in vLLM V1. For users with existing `desc_act=True` GPTQ weights, the model simply won't boot.

---

## 2. Enter the Tile Fusion Graph (TFG) Compiler

Instead of hand-writing monolithic CUDA C++ kernels for every combination of normalization, activation, and quantization, CuTileForge models a transformer layer as a **Tile Fusion Graph (`FusionGraph`)** in `cutile_forge/forge/`.

Each node (`TileNode`) carries tensor residency metadata (`hbm`, `ddr5`, `sram`, `register`), tile shapes `(TM, TN, TK)`, occupancy hints, and software pipeline depth. Six pattern-matching fusion rules (`apply_fusion_rules`) rewrite the 8-node layer graph to a fixed point:

1. **`fuse_dequant_into_gemm`**: Folds INT4 nibble unpacking and `(w_int - zero) * scale` directly into the GEMM K-loop registers—eliminating 16-bit weight materialization in HBM.
2. **`fuse_qkv_attention`**: Merges `QKV_GEMM` and `ATTENTION_GQA` into `FUSED_QKV_ATTN`.
3. **`fuse_o_layernorm`**: Merges `O_GEMM` with post-attention normalization into `FUSED_O_LAYERNORM`.
4. **`fuse_gemm_silu`**: Fuses `SiLU(x) = x / (1 + exp(-x))` into the FP32 accumulator epilogue before casting to FP16 for store.
5. **`fuse_gemm_silu_down`**: Chains `GEMM_SILU` and `FFN_DOWN_GEMM` into `FUSED_FFN`, keeping the `14336`-wide intermediate inside on-chip tile storage for `seq <= 256` and switching to 2-stage fused `dequant+silu -> dequant` above `seq=256` to bound peak VRAM on 8 GB cards.
6. **`fuse_ffn_layernorm_residual`**: Fuses the trailing residual addition and normalization epilogue into `FUSED_FFN_NORM`.

After rule application, **8 kernels collapse to 3 fused CuTile nodes**:

```
[1. FUSED_QKV_ATTN] ---> [2. FUSED_O_LAYERNORM] ---> [3. FUSED_FFN_NORM]
```

Next, two compiler passes prepare the graph for `sm_89`:
- **`tile_pipeliner.py`**: Assigns 3-stage software pipelining (`latency=3` in `cuda.tile.load`) within the `100 KB / SM` shared-memory limit on Ada Lovelace.
- **`tile_autotuner.py`**: Evaluates 12 candidate `(TM, TN, TK, occupancy, pipeline_stages)` configurations—selecting `TM=16` tiles during `M=1` decode to avoid wasted padding across 24 SMs, and `TM=128` with 2D `GROUP_M=8` L2 cache swizzling during prefill.

---

## 3. Writing Tensor Core Kernels in ~20 Lines of Python (`cuda.tile`)

In `cutile_forge/baselines/cuda_gemm.cu`, a 3-stage `cp.async` + `nvcuda::wmma` tiled FP16 GEMM takes ~150 lines of C++ with explicit shared-memory bank-conflict padding (`SA_STRIDE = BK + 8`), inline PTX `cp.async.ca.shared.global` ring buffers, `commit_group`/`wait_group` barriers, and staging buffers for bounds-checked stores.

In `cutile_forge/kernels/gemm_dequant.py`, the equivalent CuTile kernel is ~20 lines of Python:

```python
@ct.kernel(occupancy=2)
def gemm_fp16_kernel(a_ptr, b_ptr, c_ptr, tm: ct.Constant[int], tn: ct.Constant[int], tk: ct.Constant[int]):
    pid_m = ct.bid(0)
    pid_n = ct.bid(1)
    num_k = ct.num_tiles(a_ptr, axis=1, shape=(tm, tk))
    acc = ct.zeros((tm, tn), dtype=ct.float32)
    for k in range(num_k):
        tile_a = ct.load(a_ptr, index=(pid_m, k), shape=(tm, tk), padding_mode=ct.PaddingMode.ZERO, latency=3)
        tile_b = ct.load(b_ptr, index=(k, pid_n), shape=(tk, tn), padding_mode=ct.PaddingMode.ZERO, latency=3)
        acc = ct.mma(tile_a, tile_b, acc)
    ct.store(c_ptr, index=(pid_m, pid_n), tile=ct.astype(acc, ct.float16))
```

For `desc_act=True` GPTQ weights, `cutile_forge/utils/quantize.py` detects non-sequential `g_idx` vectors at load time, sorts the packed 4-bit weight rows by `g_idx` (`perm = torch.argsort(g_idx, stable=True)`), and stores `perm` on `QuantLinearWeight`. That way, every contiguous 128-row group shares one scale and zero point, and the inner K-loop of `gemm_dequant_kernel` runs at full Tensor Core speed with zero branching.

---

## 4. Stage -1: Overlapping PCIe 4.0 x8 Transfers with Compute

On a laptop GPU, the PCIe link is often narrower than on a desktop workstation. Querying `nvidia-smi` on the Lenovo LOQ 15APH8 shows a **PCIe 4.0 x8** link (`~15.75 GB/s` raw unidirectional payload ceiling, vs ~31.5 GB/s on x16).

That makes **Stage -1** (`WeightStreamer` in `cutile_forge/forge/weight_streamer.py`) especially important: when weights or evicted KV-cache pages live in **16 GB DDR5-5600**, we cannot afford to stall the compute stream waiting for synchronous `cudaMemcpy`. `WeightStreamer` pins host buffers in DDR5, schedules asynchronous H2D transfers for layer `N+1` on a dedicated `copy_stream`, and synchronizes with `torch.cuda.Event` barriers while layer `N` executes Tensor Core `mma` operations.

Measured with CUDA events on `32 MB`, `64 MB`, and `115 MB` 4-bit weight tiles, Stage -1 achieves **13.30 GB/s**—**84.4% utilization of the PCIe 4.0 x8 link**—with **100.0% overlap** on 32/64 MB tiles and **93.8% overlap** on full 115 MB layer tiles.

---

## 5. Benchmark Results on the RTX 4060 Laptop

All benchmarks were measured using CUDA events (`warmup=5`, median `p50`) on the Lenovo LOQ 15APH8 (`sm_89`), with NVML polling GPU temperature, power, and SM clocks.

### Micro-Benchmarks

- **Full Transformer Layer (3-Kernel TFG vs 8-Kernel CUDA)**:
  - `seq=1`: `0.638 ms` vs `3.490 ms` (**5.47× speedup**)
  - `seq=64`: `1.632 ms` vs `3.617 ms` (**2.22× speedup**)
  - `seq=256`: `4.934 ms` vs `8.101 ms` (**1.64× speedup**)
  - `seq=512`: `7.584 ms` vs `14.883 ms` (**1.96× speedup**)
- **Fused FFN (1 Kernel vs 3 Separate CUDA Kernels)**: **5.48×** (`seq=1`), **2.00×** (`seq=64`), **1.46×** (`seq=256`), **1.78×** (`seq=512`).
- **INT4 Dequant + GEMM**: **23.37 TFLOPS** (`1.10 ms`) at `256×12288×4096` (**2.15×** vs separate CUDA dequant+GEMM), and **7.12×** at `seq=1` (`0.148 ms` vs `1.058 ms`).
- **FP16 GEMM (`256×14336×4096`)**: CuTile hits **28.18 TFLOPS** (FP32 accumulation), outperforming PyTorch eager (**25.48 TFLOPS**) and hand-written 3-stage CUDA (**14.90 TFLOPS**), while `cublasHgemm` reaches **45.70 TFLOPS** using FP16 accumulation.
- **GQA FlashAttention-2 (`32:8`, `seq=1024`)**: CuTile reaches **21.76 TFLOPS** (`0.395 ms`) vs PyTorch SDPA **13.23 TFLOPS** (`0.649 ms`).

### End-to-End Mistral-7B-v0.1 (4-bit GPTQ) & 30-Min Thermal Soak

- **Decode (`256` tokens)**: **32.43 tok/s** (`30.83 ms/tok`) at **4.30 GB** peak VRAM.
- **Prefill**: **1,499.0 tok/s** at `2048` tokens (`4.67 GB` VRAM) and **1,475.7 tok/s** at `4096` tokens (`5.08 GB` VRAM).
- **Long Context**: `PagedKVCache` with a 1.5 GB GPU budget handles **32,768 tokens** in **6.01 GB** peak VRAM (`26,624` page evictions to pinned DDR5).
- **30-Minute Sustained Thermal Run**: Across **361** samples (`54,976` generated tokens), the GPU averaged **31.16 tok/s** at **78.7 °C** average (**83.0 °C** max, strictly below 85 °C), **85.6 W** average power, and only **4.62%** average clock throttling.
- **Correctness**: **76 / 76 tests pass** (`49` unit + `27` GPU correctness), including 32-layer numerical equivalence, token-exact greedy decoding vs PyTorch, and WikiText-2 perplexity within 5%.

---

## Try It Yourself

CuTileForge is MIT-licensed and available at **[github.com/samudragupto/cutile-forge](https://github.com/samudragupto/cutile-forge)**. If you have an 8 GB Ada or Ampere GPU and want to experiment with Python-native tile fusion without writing raw PTX, clone `v0.1.0`, run `pytest`, and inspect the generated TFG graphs in `docs/`.

