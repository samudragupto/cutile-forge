"""Stage -1: PCIe DMA weight tile prefetcher for CPU DDR5 -> GPU HBM double buffering."""

from __future__ import annotations

from dataclasses import dataclass, field

import torch

from cutile_forge.forge.fusion_graph import FusionGraph


@dataclass
class PrefetchSlot:
    """Tracks an asynchronous H2D transfer for a layer or tensor."""

    layer_idx: int
    gpu_tensors: dict[str, torch.Tensor]
    ready_event: torch.cuda.Event | None
    bytes_transferred: int
    host_refs: list[torch.Tensor] = field(default_factory=list)


class HybridLayerBudgeter:
    """Partitions model layers into HBM-resident vs. Stage -1 PCIe-streamed tiers.

    When a model exceeds 85% of available VRAM (e.g., 70B-4bit ~38.5 GB on 32 GB RTX 5090),
    pins the first `n_resident` layers permanently in HBM and streams the remaining
    `n_streamed` layers through double-buffered PCIe DMA overlapped with compute.
    """

    def __init__(
        self,
        total_layers: int,
        layer_bytes: int,
        vram_budget_gb: float = 32.0,
        reserve_slots: int = 2,
        overhead_gb: float = 2.0,
    ) -> None:
        if total_layers <= 0 or layer_bytes <= 0:
            raise ValueError("total_layers and layer_bytes must be positive")
        self.total_layers = total_layers
        self.layer_bytes = layer_bytes
        self.vram_budget_gb = vram_budget_gb
        total_model_gb = (total_layers * layer_bytes) / (1024.0**3)

        if total_model_gb <= vram_budget_gb * 0.85:
            self.n_resident = total_layers
        else:
            usable_bytes = max(
                0.0,
                vram_budget_gb * 0.85 * (1024.0**3)
                - overhead_gb * (1024.0**3)
                - reserve_slots * layer_bytes,
            )
            self.n_resident = min(total_layers, max(0, int(usable_bytes // layer_bytes)))
        self.n_streamed = total_layers - self.n_resident

    def is_resident(self, layer_idx: int) -> bool:
        return 0 <= layer_idx < self.n_resident

    @property
    def resident_gb(self) -> float:
        return (self.n_resident * self.layer_bytes) / (1024.0**3)

    @property
    def streamed_gb(self) -> float:
        return (self.n_streamed * self.layer_bytes) / (1024.0**3)

    def summary(self) -> dict[str, float | int]:
        return {
            "total_layers": self.total_layers,
            "n_resident": self.n_resident,
            "n_streamed": self.n_streamed,
            "resident_gb": round(self.resident_gb, 2),
            "streamed_gb": round(self.streamed_gb, 2),
        }


class WeightStreamer:
    """Stage -1 asynchronous PCIe DMA streamer using pinned host memory and double buffering."""

    def __init__(
        self,
        num_slots: int = 2,
        device: str | torch.device = "cuda",
        model_gb: float = 4.16,
        gpu_vram_gb: float = 8.0,
    ) -> None:
        self.num_slots = max(1, num_slots)
        self.model_gb = model_gb
        self.gpu_vram_gb = gpu_vram_gb
        self.enabled: bool = model_gb > gpu_vram_gb * 0.85
        self.device = torch.device(device)
        self.use_cuda = self.device.type == "cuda" and torch.cuda.is_available()
        self.copy_stream = torch.cuda.Stream(device=self.device) if self.use_cuda else None
        self._slots: dict[int, PrefetchSlot] = {}
        self.total_bytes_streamed = 0

    @staticmethod
    def pin_tensor(tensor: torch.Tensor) -> torch.Tensor:
        """Ensure a CPU tensor resides in page-locked (pinned) DDR5 memory when CUDA is available."""
        cpu_t = tensor.detach().cpu().contiguous()
        if torch.cuda.is_available() and not cpu_t.is_pinned():
            try:
                return cpu_t.pin_memory()
            except RuntimeError:
                return cpu_t
        return cpu_t

    def prefetch_layer(
        self,
        layer_idx: int,
        host_weights: dict[str, torch.Tensor],
    ) -> PrefetchSlot:
        """Schedule asynchronous H2D DMA copy for `layer_idx` on the dedicated copy stream."""
        if layer_idx in self._slots:
            return self._slots[layer_idx]

        # Evict oldest slot if double-buffer capacity exceeded
        if len(self._slots) >= self.num_slots:
            oldest_key = next(iter(self._slots))
            del self._slots[oldest_key]

        gpu_tensors: dict[str, torch.Tensor] = {}
        host_refs: list[torch.Tensor] = []
        total_bytes = 0
        event: torch.cuda.Event | None = None

        if self.use_cuda and self.copy_stream is not None:
            with torch.cuda.stream(self.copy_stream):
                for name, t in host_weights.items():
                    pinned = self.pin_tensor(t)
                    gpu_t = torch.empty_like(pinned, device=self.device)
                    gpu_t.copy_(pinned, non_blocking=True)
                    gpu_tensors[name] = gpu_t
                    host_refs.append(pinned)
                    total_bytes += t.numel() * t.element_size()
            event = torch.cuda.Event()
            event.record(self.copy_stream)
        else:
            for name, t in host_weights.items():
                gpu_tensors[name] = t.clone()
                total_bytes += t.numel() * t.element_size()

        self.total_bytes_streamed += total_bytes
        slot = PrefetchSlot(
            layer_idx=layer_idx,
            gpu_tensors=gpu_tensors,
            ready_event=event,
            bytes_transferred=total_bytes,
            host_refs=host_refs,
        )
        self._slots[layer_idx] = slot
        return slot

    def acquire_layer(
        self,
        layer_idx: int,
        host_weights: dict[str, torch.Tensor] | None = None,
        compute_stream: torch.cuda.Stream | None = None,
    ) -> dict[str, torch.Tensor]:
        """Wait on DMA barrier for `layer_idx` and return resident GPU weight tensors."""
        if layer_idx not in self._slots:
            if host_weights is None:
                raise KeyError(f"Layer {layer_idx} was not prefetched and no weights provided")
            self.prefetch_layer(layer_idx, host_weights)

        slot = self._slots[layer_idx]
        if self.use_cuda and slot.ready_event is not None:
            stream = compute_stream or torch.cuda.current_stream(self.device)
            stream.wait_event(slot.ready_event)
            for gpu_t in slot.gpu_tensors.values():
                gpu_t.record_stream(stream)
        return slot.gpu_tensors

    def release_layer(self, layer_idx: int) -> None:
        """Release buffered GPU weights for `layer_idx`."""
        self._slots.pop(layer_idx, None)

    def clear(self) -> None:
        self._slots.clear()


def attach_weight_streaming(graph: FusionGraph, prefetch_distance: int = 1) -> FusionGraph:
    """Annotate TFG nodes with Stage -1 PCIe DMA prefetch metadata."""
    for idx, node in enumerate(graph.topological_order()):
        node.prefetch_stage = -1
        node.attrs["prefetch_distance"] = prefetch_distance
        node.attrs["prefetch_order"] = idx
    return graph
