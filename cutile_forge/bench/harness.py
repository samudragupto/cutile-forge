"""CUDA event benchmark harness with warmup, percentile stats, and CSV export."""

from __future__ import annotations

import csv
import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Sequence

import torch


@dataclass
class Stats:
    """Summary statistics (in milliseconds) for a benchmarked callable."""

    name: str
    iters: int
    warmup: int
    mean_ms: float
    std_ms: float
    min_ms: float
    p50_ms: float
    p95_ms: float
    p99_ms: float
    max_ms: float

    @property
    def median_ms(self) -> float:
        return self.p50_ms

    def tflops(self, flops: float) -> float:
        """Compute TFLOPS using median (p50) latency."""
        if self.p50_ms <= 0.0:
            return 0.0
        return flops / (self.p50_ms * 1e-3) / 1e12

    def bandwidth_gb_s(self, num_bytes: float) -> float:
        """Compute effective memory bandwidth in GB/s using median latency."""
        if self.p50_ms <= 0.0:
            return 0.0
        return num_bytes / (self.p50_ms * 1e-3) / 1e9

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def _percentile(sorted_vals: Sequence[float], pct: float) -> float:
    if not sorted_vals:
        return 0.0
    if len(sorted_vals) == 1:
        return float(sorted_vals[0])
    pos = (len(sorted_vals) - 1) * (pct / 100.0)
    lo = int(math.floor(pos))
    hi = min(lo + 1, len(sorted_vals) - 1)
    frac = pos - lo
    return float(sorted_vals[lo] * (1.0 - frac) + sorted_vals[hi] * frac)


def benchmark(
    fn: Callable[[], Any],
    warmup: int = 5,
    iters: int = 100,
    name: str = "kernel",
    use_cuda_events: bool | None = None,
) -> Stats:
    """Measure per-iteration latency of `fn` with warmup and percentiles."""
    if iters <= 0:
        raise ValueError(f"iters must be positive, got {iters}")
    if warmup < 0:
        raise ValueError(f"warmup must be >= 0, got {warmup}")

    cuda_mode = torch.cuda.is_available() if use_cuda_events is None else use_cuda_events

    for _ in range(warmup):
        fn()
    if cuda_mode:
        torch.cuda.synchronize()

    timings_ms: List[float] = []
    if cuda_mode:
        starts = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
        ends = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
        for i in range(iters):
            starts[i].record()
            fn()
            ends[i].record()
        torch.cuda.synchronize()
        timings_ms = [float(starts[i].elapsed_time(ends[i])) for i in range(iters)]
    else:
        for _ in range(iters):
            t0 = time.perf_counter()
            fn()
            t1 = time.perf_counter()
            timings_ms.append((t1 - t0) * 1000.0)

    sorted_t = sorted(timings_ms)
    mean_ms = sum(sorted_t) / len(sorted_t)
    var = sum((x - mean_ms) ** 2 for x in sorted_t) / max(1, len(sorted_t) - 1)
    std_ms = math.sqrt(var)

    return Stats(
        name=name,
        iters=iters,
        warmup=warmup,
        mean_ms=mean_ms,
        std_ms=std_ms,
        min_ms=sorted_t[0],
        p50_ms=_percentile(sorted_t, 50.0),
        p95_ms=_percentile(sorted_t, 95.0),
        p99_ms=_percentile(sorted_t, 99.0),
        max_ms=sorted_t[-1],
    )


def write_csv(path: str | Path, rows: Iterable[Dict[str, Any]]) -> Path:
    """Write benchmark result dictionaries to a CSV file, creating parent dirs."""
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    row_list = list(rows)
    if not row_list:
        raise ValueError("Cannot write empty benchmark CSV")
    fieldnames = list(row_list[0].keys())
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(row_list)
    return out_path
