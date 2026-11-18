# CuTile vs Hand-Written CUDA C++ Comparison

## Code Complexity: Tiled FP16 GEMM

### CuTileForge Python DSL (~20 lines)

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

### Hand-Written CUDA C++ (`cutile_forge/baselines/cuda_gemm.cu`, ~150 lines)

Requires manual shared-memory bank-conflict padding (`SA_STRIDE = BK + 8`), inline PTX `cp.async.ca.shared.global` ring-buffer management, `cp.async.commit_group` / `wait_group`, warp-level `nvcuda::wmma::fragment` indexing, `__syncthreads()`, and shared-memory staging for bounds-checked global stores.

## Performance Summary

See [RESULTS.md](RESULTS.md) for measured latencies, TFLOPS, and fusion speedups on the RTX 4060 Laptop GPU (`sm_89`).

