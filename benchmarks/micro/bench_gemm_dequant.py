"""Micro-benchmark: Fused INT4 dequant+GEMM (CuTile) vs separate dequant+GEMM (CUDA)."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, List, Tuple

import torch

from cutile_forge.baselines import load_baselines
from cutile_forge.bench.harness import benchmark, write_csv
from cutile_forge.kernels.gemm_dequant import fused_gemm_dequant, pack_int4_weights

SHAPES: List[Tuple[int, int, int]] = [
    (1, 12288, 4096),
    (256, 12288, 4096),
    (1, 14336, 4096),
    (256, 14336, 4096),
]


def run_gemm_dequant_benchmark(warmup: int = 5, iters: int = 50) -> List[Dict[str, Any]]:
    """Benchmark fused INT4-dequant GEMM vs separate CUDA dequant + CUDA GEMM."""
    baselines = load_baselines()
    gs = 128
    rows: List[Dict[str, Any]] = []

    for m, n, k in SHAPES:
        a = torch.randn((m, k), device="cuda", dtype=torch.float16) * 0.05
        qw = pack_int4_weights(torch.randint(0, 16, (k, n), device="cuda", dtype=torch.uint8))
        s = torch.rand((k // gs, n), device="cuda", dtype=torch.float16) * 0.05 + 0.01
        z = torch.randint(4, 12, (k // gs, n), device="cuda", dtype=torch.float16)
        flops = 2.0 * m * n * k

        st_fused = benchmark(
            lambda a=a, qw=qw, s=s, z=z: fused_gemm_dequant(a, qw, s, z, gs),
            warmup=warmup,
            iters=iters,
            name="cutile_fused_dequant",
        )
        st_sep = benchmark(
            lambda a=a, qw=qw, s=s, z=z: baselines.cuda_gemm_dequant_separate(a, qw, s, z, gs),
            warmup=warmup,
            iters=iters,
            name="cuda_sep_dequant",
        )

        speedup = st_sep.p50_ms / max(st_fused.p50_ms, 1e-6)
        row: Dict[str, Any] = {
            "shape": f"{m}x{n}x{k}",
            "fused_ms": round(st_fused.p50_ms, 4),
            "separate_ms": round(st_sep.p50_ms, 4),
            "fused_tflops": round(st_fused.tflops(flops), 2),
            "separate_tflops": round(st_sep.tflops(flops), 2),
            "speedup": round(speedup, 3),
        }
        rows.append(row)
        print(
            f"{row['shape']:>16s} | fused={row['fused_ms']:.3f} ms ({row['fused_tflops']:.2f} TFLOPS) | "
            f"separate={row['separate_ms']:.3f} ms | speedup={speedup:.2f}x"
        )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark fused GEMM+dequant vs separate")
    parser.add_argument("--output", type=Path, default=Path("benchmarks/results/gemm_dequant.csv"))
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iters", type=int, default=50)
    args = parser.parse_args()

    rows = run_gemm_dequant_benchmark(warmup=args.warmup, iters=args.iters)
    write_csv(args.output, rows)
    print(f"Saved GEMM-dequant benchmark to {args.output}")


if __name__ == "__main__":
    main()
