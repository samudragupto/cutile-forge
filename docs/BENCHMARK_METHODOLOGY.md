# Benchmark Methodology

All benchmarks in `benchmarks/` use `cutile_forge.bench.harness.benchmark` with CUDA Event timing (`torch.cuda.Event(enable_timing=True)`), synchronized warmups, and median (`p50`) + tail (`p95`, `p99`) latency reporting.

## Rules Applied

1. **Warmup & Synchronization**: Every measurement performs `warmup` untimed iterations (default 5) and calls `torch.cuda.synchronize()` before and after timing loops.
2. **Median Reporting**: Reported latencies and derived metrics (TFLOPS, GB/s, speedup) use `p50_ms` to avoid OS scheduling outliers.
3. **Fair Baseline Comparison**:
   - CuTile kernels are compared against compiled `sm_89` CUDA C++ kernels (`_cutile_baselines`), cuBLAS `cublasHgemm`, and PyTorch eager ops on the exact same inputs.
   - INT4 fused kernels are compared against separate CUDA INT4 dequantization + CUDA tiled GEMM.
4. **Telemetry**: NVML polls GPU temperature, power draw, SM clocks, and PCIe link throughput (`nvidia-ml-py`).
5. **Reproducibility**: All benchmark scripts accept `--warmup`, `--iters`, and `--output` flags and emit machine-readable CSVs in `benchmarks/results/`.

