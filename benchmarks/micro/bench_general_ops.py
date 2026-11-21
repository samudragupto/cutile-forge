"""Microbenchmark for Phase 8A general CuTile kernels vs. PyTorch eager across shape regimes."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
import torch.nn.functional as F

from cutile_forge.bench.harness import benchmark, write_csv
from cutile_forge.kernels import (
    cutile_bmm,
    cutile_conv2d,
    cutile_cumsum,
    cutile_gather,
    cutile_gelu,
    cutile_reduce_max,
    cutile_reduce_sum,
    cutile_silu,
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

    def _rec(op: str, shape: str, ct_fn: object, pt_fn: object) -> None:
        st_ct = benchmark(ct_fn, warmup=warmup, iters=iters)  # type: ignore[arg-type]
        st_pt = benchmark(pt_fn, warmup=warmup, iters=iters)  # type: ignore[arg-type]
        rows.append(
            {
                "op": op,
                "shape": shape,
                "dtype": "float16",
                "cutile_p50_ms": round(st_ct.p50_ms, 4),
                "torch_p50_ms": round(st_pt.p50_ms, 4),
                "speedup": round(st_pt.p50_ms / max(1e-6, st_ct.p50_ms), 2),
            }
        )

    for b, m, k, n in [
        (8, 256, 1024, 1024),
        (16, 128, 2048, 2048),
        (32, 512, 512, 512),
        (4, 1024, 1024, 1024),
    ]:
        a = torch.randn((b, m, k), device="cuda", dtype=dtype) * 0.1
        w = torch.randn((b, k, n), device="cuda", dtype=dtype) * 0.1
        _rec(
            "bmm",
            f"{b}x{m}x{n}x{k}",
            lambda a=a, w=w: cutile_bmm(a, w),
            lambda a=a, w=w: torch.bmm(a, w),
        )

    for r_dim, c_dim in [(256, 2048), (4096, 4096), (16384, 1024)]:
        x = torch.randn((r_dim, c_dim), device="cuda", dtype=dtype)
        _rec(
            "reduce_sum",
            f"{r_dim}x{c_dim}",
            lambda t=x: cutile_reduce_sum(t, -1),
            lambda t=x: t.sum(dim=-1),
        )
        _rec(
            "reduce_max",
            f"{r_dim}x{c_dim}",
            lambda t=x: cutile_reduce_max(t, -1),
            lambda t=x: t.max(dim=-1).values,
        )
        _rec(
            "cumsum",
            f"{r_dim}x{c_dim}",
            lambda t=x: cutile_cumsum(t, -1),
            lambda t=x: torch.cumsum(t, dim=-1),
        )
        _rec(
            "transpose",
            f"{r_dim}x{c_dim}",
            lambda t=x: cutile_transpose(t, -2, -1),
            lambda t=x: t.transpose(-2, -1).contiguous(),
        )

    tbl = torch.randn((32000, 4096), device="cuda", dtype=dtype)
    idx = torch.randint(0, 32000, (1024,), device="cuda", dtype=torch.long)
    _rec("gather", "1024x4096", lambda: cutile_gather(tbl, idx), lambda: F.embedding(idx, tbl))

    rx = torch.randn((1024, 4096), device="cuda", dtype=dtype)
    _rec("silu", "1024x4096", lambda: cutile_silu(rx), lambda: F.silu(rx))
    _rec("gelu", "1024x4096", lambda: cutile_gelu(rx), lambda: F.gelu(rx, approximate="tanh"))

    cx3 = torch.randn((16, 64, 56, 56), device="cuda", dtype=dtype) * 0.1
    cw3 = torch.randn((128, 64, 3, 3), device="cuda", dtype=dtype) * 0.1
    _rec(
        "conv2d_3x3",
        "16x64x56x56->128",
        lambda: cutile_conv2d(cx3, cw3, padding=1),
        lambda: F.conv2d(cx3, cw3, padding=1),
    )

    cw1 = torch.randn((128, 64, 1, 1), device="cuda", dtype=dtype) * 0.1
    _rec(
        "conv2d_1x1",
        "16x64x56x56->128",
        lambda: cutile_conv2d(cx3, cw1),
        lambda: F.conv2d(cx3, cw1),
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
