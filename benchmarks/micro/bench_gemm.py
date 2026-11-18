"""Micro-benchmark: GEMM across 8 Mistral-7B shapes (CuTile vs CUDA vs cuBLAS vs PyTorch)."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, List, Tuple

import torch

from cutile_forge.baselines import load_baselines
from cutile_forge.bench.harness import benchmark, write_csv
from cutile_forge.kernels.gemm_dequant import cutile_gemm_fp16

# 8 canonical Mistral-7B shapes (M, N, K): decode (M=1) and prefill (M=256/512)
GEMM_SHAPES: List[Tuple[int, int, int]] = [
    (1, 12288, 4096),    # Decode QKV+O or fused QKV
    (256, 12288, 4096),  # Prefill fused QKV
    (1, 14336, 4096),    # Decode FFN gate/up
    (256, 14336, 4096),  # Prefill FFN gate/up
    (1, 4096, 14336),    # Decode FFN down
    (256, 4096, 14336),  # Prefill FFN down
    (1, 4096, 4096),     # Decode O-proj
    (512, 4096, 4096),   # Prefill O-proj
]


def run_gemm_benchmark(warmup: int = 5, iters: int = 50) -> List[Dict[str, Any]]:
    """Run GEMM comparison across all 8 shapes and return structured rows."""
    baselines = load_baselines()
    rows: List[Dict[str, Any]] = []

    for m, n, k in GEMM_SHAPES:
        a = torch.randn((m, k), device="cuda", dtype=torch.float16) * 0.1
        b = torch.randn((k, n), device="cuda", dtype=torch.float16) * 0.1
        flops = 2.0 * m * n * k

        st_cutile = benchmark(
            lambda a=a, b=b: cutile_gemm_fp16(a, b),
            warmup=warmup,
            iters=iters,
            name="cutile",
        )
        st_cuda = benchmark(
            lambda a=a, b=b: baselines.cuda_gemm_fp16(a, b),
            warmup=warmup,
            iters=iters,
            name="cuda",
        )
        st_cublas = benchmark(
            lambda a=a, b=b: baselines.cublas_gemm_fp16(a, b),
            warmup=warmup,
            iters=iters,
            name="cublas",
        )
        st_torch = benchmark(
            lambda a=a, b=b: torch.matmul(a, b),
            warmup=warmup,
            iters=iters,
            name="pytorch",
        )

        row: Dict[str, Any] = {
            "shape": f"{m}x{n}x{k}",
            "M": m,
            "N": n,
            "K": k,
            "cutile_ms": round(st_cutile.p50_ms, 4),
            "cuda_ms": round(st_cuda.p50_ms, 4),
            "cublas_ms": round(st_cublas.p50_ms, 4),
            "pytorch_ms": round(st_torch.p50_ms, 4),
            "cutile_tflops": round(st_cutile.tflops(flops), 2),
            "cuda_tflops": round(st_cuda.tflops(flops), 2),
            "cublas_tflops": round(st_cublas.tflops(flops), 2),
            "pytorch_tflops": round(st_torch.tflops(flops), 2),
        }
        rows.append(row)
        print(
            f"{row['shape']:>16s} | "
            f"CuTile {row['cutile_tflops']:6.2f} TFLOPS ({row['cutile_ms']:.3f} ms) | "
            f"CUDA {row['cuda_tflops']:6.2f} | "
            f"cuBLAS {row['cublas_tflops']:6.2f} | "
            f"PyTorch {row['pytorch_tflops']:6.2f}"
        )

    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark CuTile GEMM vs CUDA/cuBLAS/PyTorch")
    parser.add_argument("--output", type=Path, default=Path("benchmarks/results/gemm.csv"))
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iters", type=int, default=50)
    args = parser.parse_args()

    rows = run_gemm_benchmark(warmup=args.warmup, iters=args.iters)
    write_csv(args.output, rows)
    print(f"Saved GEMM benchmark to {args.output}")


if __name__ == "__main__":
    main()
