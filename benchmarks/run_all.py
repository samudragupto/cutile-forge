"""Master benchmark runner executing micro and end-to-end suites and generating reports."""

from __future__ import annotations

import argparse
from pathlib import Path

from benchmarks.micro.bench_attention import run_attention_benchmark
from benchmarks.micro.bench_ffn_fused import run_ffn_benchmark
from benchmarks.micro.bench_fused_layer import run_fused_layer_benchmark
from benchmarks.micro.bench_gemm import run_gemm_benchmark
from benchmarks.micro.bench_gemm_dequant import run_gemm_dequant_benchmark
from benchmarks.micro.bench_layernorm import run_layernorm_benchmark
from benchmarks.micro.bench_silu_fusion import run_silu_fusion_benchmark
from benchmarks.micro.bench_weight_stream import run_weight_stream_benchmark
from cutile_forge.bench.harness import write_csv
from cutile_forge.bench.report import generate_report


def run_all(
    model_dir: Path,
    output_dir: Path,
    micro_only: bool = False,
    warmup: int = 5,
    iters: int = 100,
) -> None:
    """Execute all micro-benchmarks and (unless `micro_only`) end-to-end benchmarks."""
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=== [Micro 1/8] GEMM ===")
    write_csv(output_dir / "gemm.csv", run_gemm_benchmark(warmup=warmup, iters=iters))

    print("=== [Micro 2/8] GEMM Dequant ===")
    write_csv(
        output_dir / "gemm_dequant.csv", run_gemm_dequant_benchmark(warmup=warmup, iters=iters)
    )

    print("=== [Micro 3/8] Fused FFN ===")
    write_csv(output_dir / "ffn.csv", run_ffn_benchmark(warmup=warmup, iters=min(iters, 50)))

    print("=== [Micro 4/8] Attention GQA ===")
    write_csv(
        output_dir / "attention.csv", run_attention_benchmark(warmup=warmup, iters=min(iters, 50))
    )

    print("=== [Micro 5/8] LayerNorm ===")
    write_csv(output_dir / "ln.csv", run_layernorm_benchmark(warmup=warmup, iters=iters))

    print("=== [Micro 6/8] SiLU Fusion ===")
    write_csv(output_dir / "silu.csv", run_silu_fusion_benchmark(warmup=warmup, iters=iters))

    print("=== [Micro 7/8] Full Fused Layer (3k vs 8k) ===")
    write_csv(
        output_dir / "layer.csv", run_fused_layer_benchmark(warmup=warmup, iters=min(iters, 50))
    )

    print("=== [Micro 8/8] Weight Streaming Overlap ===")
    write_csv(
        output_dir / "weight_stream.csv",
        run_weight_stream_benchmark(warmup=min(warmup, 3), iters=min(iters, 20)),
    )

    if not micro_only:
        from benchmarks.e2e.bench_batch import run_batch_benchmark
        from benchmarks.e2e.bench_decode import run_decode_benchmark
        from benchmarks.e2e.bench_long_context import run_long_context_benchmark
        from benchmarks.e2e.bench_prefill import run_prefill_benchmark
        from benchmarks.e2e.bench_speculative import run_speculative_benchmark

        print("=== [E2E 1/5] Prefill ===")
        write_csv(output_dir / "prefill.csv", run_prefill_benchmark(model_dir))

        print("=== [E2E 2/5] Decode ===")
        write_csv(output_dir / "decode.csv", run_decode_benchmark(model_dir))

        print("=== [E2E 3/5] Long Context ===")
        write_csv(output_dir / "long_context.csv", run_long_context_benchmark(model_dir))

        print("=== [E2E 4/5] Batch ===")
        write_csv(output_dir / "batch.csv", run_batch_benchmark(model_dir))

        print("=== [E2E 5/5] Speculative ===")
        write_csv(output_dir / "speculative.csv", run_speculative_benchmark(model_dir))

    generate_report(output_dir, Path("docs/RESULTS.md"))


def main() -> None:
    parser = argparse.ArgumentParser(description="Run full CuTileForge benchmark suite")
    parser.add_argument("--model", type=Path, default=Path(r"D:\models\mistral_7b_4bit"))
    parser.add_argument("--output", type=Path, default=Path("benchmarks/results"))
    parser.add_argument("--micro-only", action="store_true")
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iters", type=int, default=50)
    parser.add_argument("--thermal-logging", action="store_true")
    parser.add_argument("--pcie-logging", action="store_true")
    args = parser.parse_args()

    run_all(
        model_dir=args.model,
        output_dir=args.output,
        micro_only=args.micro_only,
        warmup=args.warmup,
        iters=args.iters,
    )


if __name__ == "__main__":
    main()
