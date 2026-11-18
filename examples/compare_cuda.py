"""Side-by-side timing comparison: CuTile vs hand-written CUDA vs cuBLAS vs PyTorch."""

from __future__ import annotations

import argparse

import torch

from cutile_forge.baselines import load_baselines
from cutile_forge.bench.harness import benchmark
from cutile_forge.kernels.gemm_dequant import cutile_gemm_fp16


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare CuTile vs CUDA vs cuBLAS vs PyTorch")
    parser.add_argument("--shape", default="256x12288x4096", help="MxNxK shape")
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iters", type=int, default=50)
    args = parser.parse_args()

    m, n, k = (int(x) for x in args.shape.lower().split("x"))
    a = torch.randn((m, k), device="cuda", dtype=torch.float16) * 0.1
    b = torch.randn((k, n), device="cuda", dtype=torch.float16) * 0.1
    flops = 2.0 * m * n * k
    baselines = load_baselines()

    entries = [
        ("CuTile (Python DSL)", lambda: cutile_gemm_fp16(a, b)),
        ("Hand-written CUDA", lambda: baselines.cuda_gemm_fp16(a, b)),
        ("cuBLAS HGEMM", lambda: baselines.cublas_gemm_fp16(a, b)),
        ("PyTorch eager", lambda: torch.matmul(a, b)),
    ]

    print(f"Shape: {m}x{n}x{k}")
    print(f"{'Engine':<22s} | {'p50 (ms)':>9s} | {'TFLOPS':>8s}")
    print("-" * 46)
    for label, fn in entries:
        st = benchmark(fn, warmup=args.warmup, iters=args.iters, name=label)
        print(f"{label:<22s} | {st.p50_ms:9.4f} | {st.tflops(flops):8.2f}")


if __name__ == "__main__":
    main()
