"""Multi-GPU CUDA topology discovery and peer-to-peer capability inspection."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from cutile_forge.runtime.dma import p2p_capable


@dataclass(frozen=True)
class DeviceInfo:
    """Metadata for a single visible CUDA device."""

    index: int
    name: str
    total_memory_gb: float
    compute_capability: tuple[int, int]


@dataclass
class DeviceTopology:
    """Discovered CUDA device topology and peer-to-peer connectivity matrix."""

    num_devices: int
    devices: list[DeviceInfo]
    p2p_matrix: list[list[bool]]

    @property
    def has_multi_gpu(self) -> bool:
        return self.num_devices > 1


def detect_topology() -> DeviceTopology:
    """Inspect visible CUDA devices and pairwise P2P DMA capability (CPU-safe)."""
    if not torch.cuda.is_available():
        return DeviceTopology(num_devices=0, devices=[], p2p_matrix=[])

    num_devices = torch.cuda.device_count()
    devices: list[DeviceInfo] = []
    for idx in range(num_devices):
        props = torch.cuda.get_device_properties(idx)
        devices.append(
            DeviceInfo(
                index=idx,
                name=props.name,
                total_memory_gb=float(props.total_memory / (1024**3)),
                compute_capability=(int(props.major), int(props.minor)),
            )
        )

    p2p_matrix = [
        [True if i == j else p2p_capable(i, j) for j in range(num_devices)]
        for i in range(num_devices)
    ]
    return DeviceTopology(num_devices=num_devices, devices=devices, p2p_matrix=p2p_matrix)
