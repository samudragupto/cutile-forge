"""Run per-GPU tile shape autotuner across Mistral-7B GEMM shapes."""

from __future__ import annotations

import argparse

from cutile_forge.forge.tile_autotuner import TileAutotuner


def main() -> None:
    parser = argparse.ArgumentParser(description="Autotune CuTile shapes for sm_89")
    parser.add_argument("--target", default="sm_89")
    parser.add_argument("--gpu-bench", action="store_true", help="Run live GPU benchmarking")
    args = parser.parse_args()

    tuner = TileAutotuner(target=args.target)
    shapes = [
        (1, 12288, 4096),
        (256, 12288, 4096),
        (1, 14336, 4096),
        (256, 14336, 4096),
        (256, 4096, 14336),
    ]
    for m, n, k in shapes:
        cfg = tuner.autotune(m, n, k, quantized=True, benchmark_on_gpu=args.gpu_bench)
        print(
            f"Shape {m:3d}x{n:5d}x{k:4d} -> tile=({cfg.tm}, {cfg.tn}, {cfg.tk}), "
            f"occupancy={cfg.occupancy}, stages={cfg.pipeline_stages}"
        )


if __name__ == "__main__":
    main()
