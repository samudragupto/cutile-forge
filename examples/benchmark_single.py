"""Run a single micro-benchmark by name."""

from __future__ import annotations

import argparse

from benchmarks.micro.bench_attention import run_attention_benchmark
from benchmarks.micro.bench_ffn_fused import run_ffn_benchmark
from benchmarks.micro.bench_fused_layer import run_fused_layer_benchmark
from benchmarks.micro.bench_gemm import run_gemm_benchmark
from benchmarks.micro.bench_layernorm import run_layernorm_benchmark


def main() -> None:
    parser = argparse.ArgumentParser(description="Run one CuTileForge micro-benchmark")
    parser.add_argument(
        "--name",
        choices=["gemm", "ffn", "attention", "layernorm", "layer"],
        default="gemm",
    )
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iters", type=int, default=30)
    args = parser.parse_args()

    dispatch = {
        "gemm": lambda: run_gemm_benchmark(warmup=args.warmup, iters=args.iters),
        "ffn": lambda: run_ffn_benchmark(warmup=args.warmup, iters=args.iters),
        "attention": lambda: run_attention_benchmark(warmup=args.warmup, iters=args.iters),
        "layernorm": lambda: run_layernorm_benchmark(warmup=args.warmup, iters=args.iters),
        "layer": lambda: run_fused_layer_benchmark(warmup=args.warmup, iters=args.iters),
    }
    dispatch[args.name]()


if __name__ == "__main__":
    main()
