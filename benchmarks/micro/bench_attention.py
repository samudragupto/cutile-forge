"""Micro-benchmark: CuTile FlashAttention-2 GQA vs CUDA FlashAttention-2 vs PyTorch SDPA."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

from cutile_forge.baselines import load_baselines
from cutile_forge.bench.harness import benchmark, write_csv
from cutile_forge.kernels.attention_gqa import fused_attention_gqa


def run_attention_benchmark(
    seq_lens: list[int] | None = None,
    warmup: int = 5,
    iters: int = 30,
) -> list[dict[str, Any]]:
    """Benchmark GQA 32:8 attention across sequence lengths."""
    baselines = load_baselines()
    seqs = seq_lens or [64, 256, 512, 1024]
    bsz, h_q, h_kv, d_head = 1, 32, 8, 128

    rows: list[dict[str, Any]] = []
    for seq in seqs:
        q = torch.randn((bsz, h_q, seq, d_head), device="cuda", dtype=torch.float16) * 0.5
        k = torch.randn((bsz, h_kv, seq, d_head), device="cuda", dtype=torch.float16) * 0.5
        v = torch.randn((bsz, h_kv, seq, d_head), device="cuda", dtype=torch.float16) * 0.5
        # Causal attention FLOPs: ~2 * B * H_q * S^2 * D (halved for causal) * 2 (QK + PV)
        flops = 2.0 * bsz * h_q * (seq * seq) * d_head

        st_cutile = benchmark(
            lambda q=q, k=k, v=v: fused_attention_gqa(q, k, v, causal=True),
            warmup=warmup,
            iters=iters,
            name="cutile",
        )
        st_cuda = benchmark(
            lambda q=q, k=k, v=v: baselines.cuda_attention_gqa(q, k, v, True),
            warmup=warmup,
            iters=iters,
            name="cuda",
        )

        def run_torch(
            q: torch.Tensor = q, k: torch.Tensor = k, v: torch.Tensor = v
        ) -> torch.Tensor:
            k_rep = k.repeat_interleave(h_q // h_kv, dim=1)
            v_rep = v.repeat_interleave(h_q // h_kv, dim=1)
            return F.scaled_dot_product_attention(q, k_rep, v_rep, is_causal=True)

        st_torch = benchmark(run_torch, warmup=warmup, iters=iters, name="pytorch")

        row: dict[str, Any] = {
            "seq_len": seq,
            "cutile_ms": round(st_cutile.p50_ms, 4),
            "cuda_ms": round(st_cuda.p50_ms, 4),
            "pytorch_ms": round(st_torch.p50_ms, 4),
            "cutile_tflops": round(st_cutile.tflops(flops), 2),
            "cuda_tflops": round(st_cuda.tflops(flops), 2),
            "pytorch_tflops": round(st_torch.tflops(flops), 2),
        }
        rows.append(row)
        print(
            f"seq={seq:4d} | CuTile {row['cutile_tflops']:5.2f} TFLOPS ({row['cutile_ms']:.3f} ms) | "
            f"CUDA {row['cuda_tflops']:5.2f} ({row['cuda_ms']:.3f} ms) | "
            f"PyTorch {row['pytorch_tflops']:5.2f} ({row['pytorch_ms']:.3f} ms)"
        )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark Attention: CuTile vs CUDA vs PyTorch")
    parser.add_argument("--output", type=Path, default=Path("benchmarks/results/attention.csv"))
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iters", type=int, default=30)
    args = parser.parse_args()

    rows = run_attention_benchmark(warmup=args.warmup, iters=args.iters)
    write_csv(args.output, rows)
    print(f"Saved attention benchmark to {args.output}")


if __name__ == "__main__":
    main()
