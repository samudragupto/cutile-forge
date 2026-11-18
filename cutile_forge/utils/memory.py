"""GPU VRAM tracking and allocation helpers."""

from __future__ import annotations

import torch


def get_vram_usage_gb(device: int = 0) -> float:
    """Return currently allocated PyTorch CUDA VRAM in gigabytes."""
    if not torch.cuda.is_available():
        return 0.0
    return float(torch.cuda.memory_allocated(device)) / 1e9


def get_peak_vram_gb(device: int = 0) -> float:
    """Return peak allocated PyTorch CUDA VRAM in gigabytes."""
    if not torch.cuda.is_available():
        return 0.0
    return float(torch.cuda.max_memory_allocated(device)) / 1e9


def reset_peak_vram(device: int = 0) -> None:
    """Reset PyTorch peak CUDA memory statistics."""
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats(device)


def get_vram_summary(device: int = 0) -> dict[str, float]:
    """Return allocated, reserved, peak, and total VRAM in GB."""
    if not torch.cuda.is_available():
        return {"allocated_gb": 0.0, "reserved_gb": 0.0, "peak_gb": 0.0, "total_gb": 0.0}
    props = torch.cuda.get_device_properties(device)
    return {
        "allocated_gb": round(torch.cuda.memory_allocated(device) / 1e9, 3),
        "reserved_gb": round(torch.cuda.memory_reserved(device) / 1e9, 3),
        "peak_gb": round(torch.cuda.max_memory_allocated(device) / 1e9, 3),
        "total_gb": round(props.total_memory / 1e9, 3),
    }
