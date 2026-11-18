"""Micro-benchmark: LayerNorm — CuTile vs CUDA C++ vs PyTorch."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

from cutile_forge.baselines import load_baselines
from cutile_forge.bench.harness import benchmark, write_csv
from cutile_forge.kernels.layernorm import cutile_layernorm


def run_layernorm_benchmark(
    seq_lens: list[int] | None = None,
    warmup: int = 5,
    iters: int = 50,
) -> list[dict[str, Any]]:
    """Benchmark LayerNorm (D=4096) across sequence lengths."""
    baselines = load_baselines()
    seqs = seq_lens or [1, 64, 256, 1024]
    d_model = 4096
    w = torch.randn(d_model, device="cuda", dtype=torch.float16)
    b = torch.randn(d_model, device="cuda", dtype=torch.float16)

    rows: list[dict[str, Any]] = []
    for seq in seqs:
        x = torch.randn((seq, d_model), device="cuda", dtype=torch.float16)
        num_bytes = 2.0 * x.numel() * 2.0  # read x + write y (fp16)

        st_cutile = benchmark(
            lambda x=x: cutile_layernorm(x, w, b, 1e-5),
            warmup=warmup,
            iters=iters,
            name="cutile",
        )
        st_cuda = benchmark(
            lambda x=x: baselines.cuda_layernorm(x, w, b, 1e-5),
            warmup=warmup,
            iters=iters,
            name="cuda",
        )
        st_torch = benchmark(
            lambda x=x: F.layer_norm(x, (d_model,), w, b, 1e-5),
            warmup=warmup,
            iters=iters,
            name="pytorch",
        )

        row: dict[str, Any] = {
            "seq_len": seq,
            "cutile_ms": round(st_cutile.p50_ms, 4),
            "cuda_ms": round(st_cuda.p50_ms, 4),
            "pytorch_ms": round(st_torch.p50_ms, 4),
            "cutile_bw_gbs": round(st_cutile.bandwidth_gb_s(num_bytes), 2),
            "cuda_bw_gbs": round(st_cuda.bandwidth_gb_s(num_bytes), 2),
        }
        rows.append(row)
        print(
            f"seq={seq:4d} | CuTile {row['cutile_ms']:.4f} ms ({row['cutile_bw_gbs']:.1f} GB/s) | "
            f"CUDA {row['cuda_ms']:.4f} ms | PyTorch {row['pytorch_ms']:.4f} ms"
        )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark LayerNorm: CuTile vs CUDA vs PyTorch")
    parser.add_argument("--output", type=Path, default=Path("benchmarks/results/ln.csv"))
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iters", type=int, default=50)
    args = parser.parse_args()

    rows = run_layernorm_benchmark(warmup=args.warmup, iters=args.iters)
    write_csv(args.output, rows)
    print(f"Saved LayerNorm benchmark to {args.output}")


if __name__ == "__main__":
    main()
