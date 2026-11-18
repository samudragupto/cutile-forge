"""Micro-benchmark: Fused FFN (1 CuTile kernel) vs 3 separate CUDA kernels."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, List

import torch

from cutile_forge.baselines import load_baselines
from cutile_forge.bench.harness import benchmark, write_csv
from cutile_forge.kernels.ffn_fused import fused_ffn_dequant_up_silu_down
from cutile_forge.kernels.gemm_dequant import pack_int4_weights


def run_ffn_benchmark(
    seq_lens: List[int] | None = None,
    warmup: int = 5,
    iters: int = 30,
) -> List[Dict[str, Any]]:
    """Benchmark 4-bit fused FFN (1 kernel) vs 3 separate CUDA kernels."""
    baselines = load_baselines()
    seqs = seq_lens or [1, 64, 256, 512]
    k_dim, h_dim, d_dim = 4096, 14336, 4096
    group_size = 128

    qw_up = pack_int4_weights(
        torch.randint(0, 16, (k_dim, h_dim), device="cuda", dtype=torch.uint8)
    )
    s_up = torch.rand((k_dim // group_size, h_dim), device="cuda", dtype=torch.float16) * 0.05 + 0.01
    z_up = torch.randint(4, 12, (k_dim // group_size, h_dim), device="cuda", dtype=torch.float16)

    qw_down = pack_int4_weights(
        torch.randint(0, 16, (h_dim, d_dim), device="cuda", dtype=torch.uint8)
    )
    s_down = (
        torch.rand((h_dim // group_size, d_dim), device="cuda", dtype=torch.float16) * 0.05 + 0.01
    )
    z_down = torch.randint(4, 12, (h_dim // group_size, d_dim), device="cuda", dtype=torch.float16)

    rows: List[Dict[str, Any]] = []
    for seq in seqs:
        x = torch.randn((seq, k_dim), device="cuda", dtype=torch.float16) * 0.05
        tm = 16 if seq <= 16 else 64
        m_pad = ((seq + tm - 1) // tm) * tm
        nh = (h_dim + 127) // 128
        workspace = torch.empty((nh * m_pad, d_dim), device="cuda", dtype=torch.float16)

        def run_fused(
            x: torch.Tensor = x,
            workspace: torch.Tensor = workspace,
        ) -> torch.Tensor:
            return fused_ffn_dequant_up_silu_down(
                x, qw_up, s_up, z_up, qw_down, s_down, z_down, group_size, workspace=workspace
            )

        def run_separate(x: torch.Tensor = x) -> torch.Tensor:
            h1 = baselines.cuda_gemm_dequant_separate(x, qw_up, s_up, z_up, group_size)
            h2 = baselines.cuda_silu(h1)
            return baselines.cuda_gemm_dequant_separate(h2, qw_down, s_down, z_down, group_size)

        st_fused = benchmark(run_fused, warmup=warmup, iters=iters, name=f"fused_{seq}")
        st_sep = benchmark(run_separate, warmup=warmup, iters=iters, name=f"sep_{seq}")
        speedup = st_sep.p50_ms / max(st_fused.p50_ms, 1e-6)

        row: Dict[str, Any] = {
            "seq_len": seq,
            "fused_ms": round(st_fused.p50_ms, 4),
            "separate_ms": round(st_sep.p50_ms, 4),
            "speedup": round(speedup, 3),
        }
        rows.append(row)
        print(
            f"seq={seq:4d} | fused={row['fused_ms']:.3f} ms | "
            f"separate={row['separate_ms']:.3f} ms | speedup={speedup:.2f}x"
        )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark Fused FFN vs 3 separate kernels")
    parser.add_argument("--output", type=Path, default=Path("benchmarks/results/ffn.csv"))
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iters", type=int, default=30)
    args = parser.parse_args()

    rows = run_ffn_benchmark(warmup=args.warmup, iters=args.iters)
    write_csv(args.output, rows)
    print(f"Saved FFN benchmark to {args.output}")


if __name__ == "__main__":
    main()

