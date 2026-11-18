"""CPU-GPU / Compute-DMA overlap percentage profiler using CUDA events."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import torch


@dataclass
class OverlapStats:
    """Overlap breakdown between compute stream and H2D DMA copy stream."""

    serial_ms: float
    overlapped_ms: float
    compute_only_ms: float
    dma_only_ms: float
    overlap_pct: float
    speedup: float


def measure_compute_dma_overlap(
    compute_fn: Callable[[torch.cuda.Stream], Any],
    dma_fn: Callable[[torch.cuda.Stream], Any],
    warmup: int = 3,
    iters: int = 20,
) -> OverlapStats:
    """Measure overlap % between `compute_fn` and `dma_fn` running on separate CUDA streams."""
    if not torch.cuda.is_available():
        return OverlapStats(0.0, 0.0, 0.0, 0.0, 0.0, 1.0)

    comp_stream = torch.cuda.Stream()
    dma_stream = torch.cuda.Stream()

    for _ in range(warmup):
        with torch.cuda.stream(comp_stream):
            compute_fn(comp_stream)
        with torch.cuda.stream(dma_stream):
            dma_fn(dma_stream)
    torch.cuda.synchronize()

    def _time_single(fn: Callable[[torch.cuda.Stream], Any], stream: torch.cuda.Stream) -> float:
        s = torch.cuda.Event(enable_timing=True)
        e = torch.cuda.Event(enable_timing=True)
        s.record(stream)
        for _ in range(iters):
            with torch.cuda.stream(stream):
                fn(stream)
        e.record(stream)
        torch.cuda.synchronize()
        return float(s.elapsed_time(e)) / iters

    comp_ms = _time_single(compute_fn, comp_stream)
    dma_ms = _time_single(dma_fn, dma_stream)
    serial_ms = comp_ms + dma_ms

    s_all = torch.cuda.Event(enable_timing=True)
    e_all = torch.cuda.Event(enable_timing=True)
    s_all.record()
    for _ in range(iters):
        comp_stream.wait_event(torch.cuda.current_stream().record_event())
        dma_stream.wait_event(torch.cuda.current_stream().record_event())
        with torch.cuda.stream(comp_stream):
            compute_fn(comp_stream)
        with torch.cuda.stream(dma_stream):
            dma_fn(dma_stream)
        torch.cuda.current_stream().wait_stream(comp_stream)
        torch.cuda.current_stream().wait_stream(dma_stream)
    e_all.record()
    torch.cuda.synchronize()
    overlapped_ms = float(s_all.elapsed_time(e_all)) / iters

    hidden_ms = max(0.0, serial_ms - overlapped_ms)
    denom = min(comp_ms, dma_ms)
    overlap_pct = min(100.0, (hidden_ms / max(denom, 1e-6)) * 100.0)
    speedup = serial_ms / max(overlapped_ms, 1e-6)

    return OverlapStats(
        serial_ms=round(serial_ms, 4),
        overlapped_ms=round(overlapped_ms, 4),
        compute_only_ms=round(comp_ms, 4),
        dma_only_ms=round(dma_ms, 4),
        overlap_pct=round(overlap_pct, 2),
        speedup=round(speedup, 3),
    )
