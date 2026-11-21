"""Microbenchmark for Sprint 2 general CuTile kernels vs. PyTorch eager."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from cutile_forge.bench.harness import benchmark, write_csv
from cutile_forge.kernels import (
    cutile_bmm,
    cutile_cumsum,
    cutile_reduce_max,
    cutile_reduce_sum,
    cutile_transpose,
)


def run_benchmark(
    output_csv: Path = Path("benchmarks/results_5090/general_ops.csv"),
    warmup: int = 5,
    iters: int = 50,
) -> list[dict[str, object]]:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU required for general ops microbenchmark")

    rows: list[dict[str, object]] = []
    dtype = torch.float16

    # 1. Batched GEMM (B, M, K) @ (B, K, N)
    for b, m, k, n in [(8, 256, 1024, 1024), (16, 128, 2048, 2048)]:
        a = torch.randn((b, m, k), device="cuda", dtype=dtype) * 0.1
        w = torch.randn((b, k, n), device="cuda", dtype=dtype) * 0.1
        st_ct = benchmark(lambda a=a, w=w: cutile_bmm(a, w), warmup=warmup, iters=iters)
        st_pt = benchmark(lambda a=a, w=w: torch.bmm(a, w), warmup=warmup, iters=iters)
        rows.append(
            {
                "op": "bmm",
                "shape": f"{b}x{m}x{n}x{k}",
                "dtype": "float16",
                "cutile_p50_ms": round(st_ct.p50_ms, 4),
                "torch_p50_ms": round(st_pt.p50_ms, 4),
                "speedup": round(st_pt.p50_ms / max(1e-6, st_ct.p50_ms), 2),
            }
        )

    # 2. Reductions, CumSum, and Transpose
    x = torch.randn((256, 2048), device="cuda", dtype=dtype)
    for name, ct_fn, pt_fn in [
        ("reduce_sum", lambda t=x: cutile_reduce_sum(t, -1), lambda t=x: t.sum(dim=-1)),
        ("reduce_max", lambda t=x: cutile_reduce_max(t, -1), lambda t=x: t.max(dim=-1).values),
        ("cumsum", lambda t=x: cutile_cumsum(t, -1), lambda t=x: torch.cumsum(t, dim=-1)),
        (
            "transpose",
            lambda t=x: cutile_transpose(t, -2, -1),
            lambda t=x: t.transpose(-2, -1).contiguous(),
        ),
    ]:
        st_ct = benchmark(ct_fn, warmup=warmup, iters=iters)
        st_pt = benchmark(pt_fn, warmup=warmup, iters=iters)
        rows.append(
            {
                "op": name,
                "shape": "256x2048",
                "dtype": "float16",
                "cutile_p50_ms": round(st_ct.p50_ms, 4),
                "torch_p50_ms": round(st_pt.p50_ms, 4),
                "speedup": round(st_pt.p50_ms / max(1e-6, st_ct.p50_ms), 2),
            }
        )

    write_csv(output_csv, rows)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark general CuTile kernels vs PyTorch")
    parser.add_argument(
        "--output", type=Path, default=Path("benchmarks/results_5090/general_ops.csv")
    )
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iters", type=int, default=50)
    args = parser.parse_args()
    rows = run_benchmark(output_csv=args.output, warmup=args.warmup, iters=args.iters)
    for r in rows:
        print(r)


if __name__ == "__main__":
    main()
