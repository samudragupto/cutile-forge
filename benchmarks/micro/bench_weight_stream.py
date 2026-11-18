"""Micro-benchmark: Stage -1 PCIe tile weight streaming overlap measurement."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, List

import torch

from cutile_forge.bench.harness import write_csv
from cutile_forge.bench.overlap_profiler import measure_compute_dma_overlap
from cutile_forge.bench.pcie_monitor import measure_pcie_h2d_bandwidth
from cutile_forge.kernels.gemm_dequant import fused_gemm_dequant, pack_int4_weights


def run_weight_stream_benchmark(warmup: int = 3, iters: int = 20) -> List[Dict[str, Any]]:
    """Measure Compute-DMA overlap across layer weight sizes (4-bit tiles)."""
    pcie = measure_pcie_h2d_bandwidth(nbytes=64 * 1024 * 1024, iters=10)
    rows: List[Dict[str, Any]] = []

    # Simulate streaming next-layer 4-bit weights (32 MB / 64 MB / 115 MB) while computing GEMM
    for mb in [32, 64, 115]:
        nbytes = mb * 1024 * 1024
        host_tile = torch.empty(nbytes // 2, dtype=torch.float16, pin_memory=True)
        dev_tile = torch.empty(nbytes // 2, dtype=torch.float16, device="cuda")

        a = torch.randn((256, 4096), device="cuda", dtype=torch.float16) * 0.05
        qw = pack_int4_weights(
            torch.randint(0, 16, (4096, 12288), device="cuda", dtype=torch.uint8)
        )
        s = torch.rand((32, 12288), device="cuda", dtype=torch.float16) * 0.05 + 0.01
        z = torch.randint(4, 12, (32, 12288), device="cuda", dtype=torch.float16)

        def compute_fn(
            _stream: torch.cuda.Stream,
            a: torch.Tensor = a,
            qw: torch.Tensor = qw,
            s: torch.Tensor = s,
            z: torch.Tensor = z,
        ) -> None:
            fused_gemm_dequant(a, qw, s, z, 128)

        def dma_fn(
            _stream: torch.cuda.Stream,
            dev_tile: torch.Tensor = dev_tile,
            host_tile: torch.Tensor = host_tile,
        ) -> None:
            dev_tile.copy_(host_tile, non_blocking=True)

        ov = measure_compute_dma_overlap(compute_fn, dma_fn, warmup=warmup, iters=iters)
        row: Dict[str, Any] = {
            "tile_mb": mb,
            "serial_ms": ov.serial_ms,
            "overlapped_ms": ov.overlapped_ms,
            "compute_ms": ov.compute_only_ms,
            "dma_ms": ov.dma_only_ms,
            "overlap_pct": ov.overlap_pct,
            "speedup": ov.speedup,
            "pcie_gb_s": pcie.measured_gb_s,
            "pcie_util_pct": pcie.utilization_pct,
        }
        rows.append(row)
        print(
            f"tile={mb:3d} MB | serial={ov.serial_ms:.3f} ms | overlap={ov.overlapped_ms:.3f} ms | "
            f"overlap_pct={ov.overlap_pct:.1f}% | PCIe={pcie.measured_gb_s:.2f} GB/s ({pcie.utilization_pct:.1f}%)"
        )

    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark Stage -1 PCIe weight streaming overlap")
    parser.add_argument("--output", type=Path, default=Path("benchmarks/results/weight_stream.csv"))
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--iters", type=int, default=20)
    args = parser.parse_args()

    rows = run_weight_stream_benchmark(warmup=args.warmup, iters=args.iters)
    write_csv(args.output, rows)
    print(f"Saved weight stream benchmark to {args.output}")


if __name__ == "__main__":
    main()
