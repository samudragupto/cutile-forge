"""Micro-benchmark: SiLU fused-in-GEMM vs separate GEMM + SiLU kernel."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, List

import torch

from cutile_forge.baselines import load_baselines
from cutile_forge.bench.harness import benchmark, write_csv
from cutile_forge.kernels.gemm_dequant import fused_gemm_dequant, pack_int4_weights
from cutile_forge.kernels.gemm_dequant_silu import fused_gemm_dequant_silu


def run_silu_fusion_benchmark(
    seq_lens: List[int] | None = None,
    warmup: int = 5,
    iters: int = 50,
) -> List[Dict[str, Any]]:
    """Compare fused GEMM+dequant+SiLU against separate GEMM+dequant + CUDA SiLU."""
    baselines = load_baselines()
    seqs = seq_lens or [1, 64, 256, 512]
    k_dim, n_dim, gs = 4096, 14336, 128

    qw = pack_int4_weights(torch.randint(0, 16, (k_dim, n_dim), device="cuda", dtype=torch.uint8))
    s = torch.rand((k_dim // gs, n_dim), device="cuda", dtype=torch.float16) * 0.05 + 0.01
    z = torch.randint(4, 12, (k_dim // gs, n_dim), device="cuda", dtype=torch.float16)

    rows: List[Dict[str, Any]] = []
    for seq in seqs:
        x = torch.randn((seq, k_dim), device="cuda", dtype=torch.float16) * 0.05

        st_fused = benchmark(
            lambda x=x: fused_gemm_dequant_silu(x, qw, s, z, gs),
            warmup=warmup,
            iters=iters,
            name="fused_silu",
        )
        st_sep = benchmark(
            lambda x=x: baselines.cuda_silu(fused_gemm_dequant(x, qw, s, z, gs)),
            warmup=warmup,
            iters=iters,
            name="sep_silu",
        )

        speedup = st_sep.p50_ms / max(st_fused.p50_ms, 1e-6)
        row: Dict[str, Any] = {
            "seq_len": seq,
            "fused_ms": round(st_fused.p50_ms, 4),
            "separate_ms": round(st_sep.p50_ms, 4),
            "speedup": round(speedup, 3),
        }
        rows.append(row)
        print(
            f"seq={seq:4d} | fused_gemm_silu={row['fused_ms']:.4f} ms | "
            f"separate={row['separate_ms']:.4f} ms | speedup={speedup:.2f}x"
        )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark SiLU fusion")
    parser.add_argument("--output", type=Path, default=Path("benchmarks/results/silu.csv"))
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iters", type=int, default=50)
    args = parser.parse_args()

    rows = run_silu_fusion_benchmark(warmup=args.warmup, iters=args.iters)
    write_csv(args.output, rows)
    print(f"Saved SiLU fusion benchmark to {args.output}")


if __name__ == "__main__":
    main()
