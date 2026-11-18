"""CPU and DDR5 host memory bandwidth profiler."""

from __future__ import annotations

import os
import time
from dataclasses import dataclass

import torch


@dataclass
class CPUProfileStats:
    """Host CPU and DDR5 copy bandwidth summary."""

    cpu_count: int
    ddr5_copy_gb_s: float
    pinned_alloc_ms: float


def profile_ddr5_bandwidth(nbytes: int = 64 * 1024 * 1024, iters: int = 10) -> CPUProfileStats:
    """Measure host DDR5 memory copy bandwidth and pinned allocation overhead."""
    numel = nbytes // 2
    t0 = time.perf_counter()
    src = torch.randn(numel, dtype=torch.float16)
    dst = torch.empty(numel, dtype=torch.float16)
    alloc_ms = (time.perf_counter() - t0) * 1000.0

    dst.copy_(src)
    t_start = time.perf_counter()
    for _ in range(iters):
        dst.copy_(src)
    elapsed = (time.perf_counter() - t_start) / max(1, iters)
    # Copy reads `nbytes` + writes `nbytes` = 2 * nbytes
    gb_s = (2.0 * nbytes / 1e9) / max(elapsed, 1e-9)

    return CPUProfileStats(
        cpu_count=os.cpu_count() or 1,
        ddr5_copy_gb_s=round(gb_s, 2),
        pinned_alloc_ms=round(alloc_ms, 3),
    )
