"""Micro-benchmark: NVFP4 (e2m1) GEMM and 8-expert Fused MoE (Mixtral-8x7B) on RTX 5090."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import torch

from cutile_forge.bench.harness import benchmark, write_csv
from cutile_forge.kernels.gemm_dequant import fused_gemm_nvfp4_dequant, pack_int4_weights
from cutile_forge.kernels.moe_fused import fused_moe_int4_gemm, fused_topk_gating


def run_moe_fp4_benchmark(warmup: int = 5, iters: int = 50) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []

    # 1. NVFP4 (e2m1) GEMM benchmark
    for m, n, k in [(1, 14336, 4096), (256, 14336, 4096)]:
        bs = 128
        a = torch.randn((m, k), device="cuda", dtype=torch.float16) * 0.1
        qw = torch.randint(0, 255, (k // 2, n), device="cuda", dtype=torch.uint8)
        sc = torch.rand((k // bs, n), device="cuda", dtype=torch.float16) * 0.1 + 0.05
        st = benchmark(
            lambda a=a, qw=qw, sc=sc: fused_gemm_nvfp4_dequant(a, qw, sc),
            warmup=warmup,
            iters=iters,
            name=f"nvfp4_{m}x{n}x{k}",
        )
        tflops = round(st.tflops(2.0 * m * n * k), 2)
        rows.append(
            {
                "kernel": f"nvfp4_gemm_{m}x{n}x{k}",
                "M": m,
                "N": n,
                "K": k,
                "p50_ms": round(st.p50_ms, 4),
                "tflops": tflops,
            }
        )
        print(f"NVFP4 {m}x{n}x{k} | {st.p50_ms:.3f} ms | {tflops:.2f} TFLOPS")

    # 2. Fused MoE (8 experts, top_k=2)
    e, top_k, k, n, gs = 8, 2, 4096, 4096, 128
    qw_e = torch.stack(
        [
            pack_int4_weights(torch.randint(0, 16, (k, n), device="cuda", dtype=torch.uint8))
            for _ in range(e)
        ]
    )
    sc_e = torch.rand((e, k // gs, n), device="cuda", dtype=torch.float16) * 0.05 + 0.01
    zr_e = torch.randint(4, 12, (e, k // gs, n), device="cuda", dtype=torch.float16)

    for m in [1, 64, 256]:
        x = torch.randn((m, k), device="cuda", dtype=torch.float16) * 0.05
        r_logits = torch.randn((m, e), device="cuda", dtype=torch.float16)

        def run_moe(
            x: torch.Tensor = x,
            r_logits: torch.Tensor = r_logits,
        ) -> torch.Tensor:
            probs, idx = fused_topk_gating(r_logits, top_k=top_k)
            return fused_moe_int4_gemm(x, qw_e, sc_e, zr_e, probs, idx, group_size=gs)

        st = benchmark(run_moe, warmup=warmup, iters=iters, name=f"moe_8x_{m}")
        tflops = round(st.tflops(2.0 * m * top_k * k * n), 2)
        rows.append(
            {
                "kernel": f"fused_moe_8x7b_m{m}",
                "M": m,
                "N": n,
                "K": k,
                "p50_ms": round(st.p50_ms, 4),
                "tflops": tflops,
            }
        )
        print(f"Fused MoE 8x (M={m:3d}, top2) | {st.p50_ms:.3f} ms | {tflops:.2f} TFLOPS")

    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark NVFP4 GEMM and 8-expert Fused MoE")
    parser.add_argument("--output", type=Path, default=Path("benchmarks/results_5090/moe_fp4.csv"))
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iters", type=int, default=50)
    args = parser.parse_args()

    rows = run_moe_fp4_benchmark(warmup=args.warmup, iters=args.iters)
    write_csv(args.output, rows)
    print(f"Saved MoE/FP4 benchmark to {args.output}")


if __name__ == "__main__":
    main()
