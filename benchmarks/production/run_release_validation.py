"""Master CLI orchestrator for permanent CuTileForge release validation."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from benchmarks.production.runner import pack_artifact_bundle  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="CuTileForge Release Validation Runner")
    parser.add_argument("--profile", default="rtx5090")
    parser.add_argument(
        "--model-dir", type=Path, default=Path("/home/arrhat/models/mistral_7b_4bit")
    )
    parser.add_argument(
        "--mixtral-dir", type=Path, default=Path("/var/tmp/arrhat_models/mixtral_8x7b_4bit")
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("benchmarks/validation_artifacts/v3.1.0"),
    )
    parser.add_argument("--bench-set", choices=("quick", "full"), default="full")
    parser.add_argument("--soak-seconds", type=float, default=900.0)
    parser.add_argument("--version", default="3.1.0")
    args = parser.parse_args()

    out_root = args.output_root
    out_root.mkdir(parents=True, exist_ok=True)

    from benchmarks.production.bench_long_context_real import run_long_context_validation
    from benchmarks.production.bench_multi_gpu_validation import run_multi_gpu_validation
    from benchmarks.production.bench_quant_e2e_parity import run_quant_e2e_parity
    from benchmarks.production.bench_reliability_faults import run_reliability_suite
    from benchmarks.production.bench_serving_stress import run_serving_stress
    from benchmarks.production.probe_external_engines import run_external_engine_validation

    print(f"[release-validation] Starting profile={args.profile} bench_set={args.bench_set}")

    if args.bench_set == "full":
        run_external_engine_validation(args.model_dir, out_root / "phase14b_external_engines")
        run_long_context_validation(
            args.model_dir,
            args.mixtral_dir,
            out_root / "phase14c_long_context",
            out_root / "phase14c_synthetic_attention",
        )
        run_multi_gpu_validation(out_root / "phase14d_multi_gpu")
        run_quant_e2e_parity(
            args.model_dir, out_root / "phase14e_quantization", multi_format_layers=8
        )
        run_serving_stress(
            args.model_dir,
            out_root / "phase14f_serving_stress",
            soak_seconds=args.soak_seconds,
        )
        run_reliability_suite(args.model_dir, out_root / "phase14g_reliability")
    else:
        run_multi_gpu_validation(out_root / "phase14d_multi_gpu")
        run_quant_e2e_parity(
            args.model_dir, out_root / "phase14e_quantization", multi_format_layers=2
        )
        run_serving_stress(
            args.model_dir,
            out_root / "phase14f_serving_stress",
            soak_seconds=min(30.0, args.soak_seconds),
            max_concurrent=16,
        )
        run_reliability_suite(args.model_dir, out_root / "phase14g_reliability")

    # Build reports & figures from raw artifacts
    import subprocess

    subprocess.run(
        [
            sys.executable,
            "scripts/build_validation_report.py",
            "--artifacts-root",
            str(out_root),
            "--docs-dir",
            "docs",
            "--version",
            args.version,
        ],
        check=True,
    )
    subprocess.run(
        [
            sys.executable,
            "scripts/generate_validation_figures.py",
            "--artifacts-root",
            str(out_root),
            "--output-dir",
            "docs/figures/validation_v310",
        ],
        check=True,
    )

    bundle_path = Path(f"dist/cutile_forge-{args.version}-validation-bundle.tar.gz")
    pack_artifact_bundle(out_root, bundle_path)
    meta = {"profile": args.profile, "bundle": str(bundle_path), "artifacts_root": str(out_root)}
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
