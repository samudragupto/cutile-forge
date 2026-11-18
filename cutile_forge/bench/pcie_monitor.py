"""PCIe bandwidth and link utilization measurement."""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class PCIeStats:
    """Measured PCIe H2D transfer bandwidth and utilization against theoretical link cap."""

    link_gen: int
    link_width: int
    theoretical_gb_s: float
    measured_gb_s: float
    utilization_pct: float


def measure_pcie_h2d_bandwidth(nbytes: int = 128 * 1024 * 1024, iters: int = 20) -> PCIeStats:
    """Measure pinned DDR5 -> GPU HBM PCIe H2D bandwidth using CUDA events."""
    link_gen, link_width = 4, 16
    try:
        import pynvml

        pynvml.nvmlInit()
        h = pynvml.nvmlDeviceGetHandleByIndex(0)
        link_gen = int(pynvml.nvmlDeviceGetCurrPcieLinkGeneration(h))
        link_width = int(pynvml.nvmlDeviceGetCurrPcieLinkWidth(h))
        pynvml.nvmlShutdown()
    except Exception:
        pass

    # Practical unidirectional PCIe payload rate (~1.969 GB/s/lane for Gen4, ~0.985 for Gen3)
    per_lane = {3: 0.985, 4: 1.969, 5: 3.938}.get(link_gen, 1.969)
    theoretical = per_lane * link_width

    if not torch.cuda.is_available():
        return PCIeStats(link_gen, link_width, theoretical, 0.0, 0.0)

    numel = nbytes // 2
    host_buf = torch.empty(numel, dtype=torch.float16, pin_memory=True)
    dev_buf = torch.empty(numel, dtype=torch.float16, device="cuda")

    for _ in range(3):
        dev_buf.copy_(host_buf, non_blocking=True)
    torch.cuda.synchronize()

    s = torch.cuda.Event(enable_timing=True)
    e = torch.cuda.Event(enable_timing=True)
    s.record()
    for _ in range(iters):
        dev_buf.copy_(host_buf, non_blocking=True)
    e.record()
    torch.cuda.synchronize()

    sec = (s.elapsed_time(e) / iters) / 1000.0
    gb_s = (nbytes / 1e9) / max(sec, 1e-9)
    util = min(100.0, (gb_s / max(theoretical, 1e-6)) * 100.0)
    return PCIeStats(
        link_gen=link_gen,
        link_width=link_width,
        theoretical_gb_s=round(theoretical, 2),
        measured_gb_s=round(gb_s, 2),
        utilization_pct=round(util, 2),
    )


def estimate_pcie_utilization(
    bytes_transferred: int, elapsed_s: float, theoretical_gb_s: float = 25.0
) -> float:
    """Compute effective PCIe utilization percentage from streamed bytes and elapsed seconds."""
    if elapsed_s <= 0.0 or theoretical_gb_s <= 0.0:
        return 0.0
    gb_s = (bytes_transferred / 1e9) / elapsed_s
    return min(100.0, (gb_s / theoretical_gb_s) * 100.0)
