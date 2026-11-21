"""General-purpose asynchronous tensor DMA engine with double/triple buffering & backpressure."""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass

import torch

from cutile_forge.runtime.memory import PinnedHostPool


@dataclass
class StreamedBuffer:
    """Tracks a streamed GPU tensor and its DMA completion event."""

    key: str
    gpu_tensor: torch.Tensor
    ready_event: torch.cuda.Event | None
    nbytes: int
    staged_host: torch.Tensor | None = None


class TensorStreamer:
    """Arbitrary-tensor async PCIe DMA streamer supporting double/triple buffering and LRU eviction."""

    def __init__(
        self,
        num_slots: int = 3,
        max_vram_bytes: int | None = None,
        device: str | torch.device = "cuda",
        host_pool: PinnedHostPool | None = None,
        adaptive_slots: bool = False,
    ) -> None:
        if num_slots < 1:
            raise ValueError(f"num_slots must be >= 1, got {num_slots}")
        self.num_slots = num_slots
        self._max_slots = max(2, num_slots)
        self.adaptive_slots = adaptive_slots
        self.max_vram_bytes = max_vram_bytes
        self.device = torch.device(
            device if (str(device) != "cuda" or torch.cuda.is_available()) else "cpu"
        )
        self.use_cuda = self.device.type == "cuda" and torch.cuda.is_available()
        self.copy_stream = torch.cuda.Stream(device=self.device) if self.use_cuda else None
        self.host_pool = host_pool or PinnedHostPool()

        self._slots: OrderedDict[str, StreamedBuffer] = OrderedDict()
        self.vram_used_bytes: int = 0
        self.evictions: int = 0
        self.total_bytes_streamed: int = 0

    def _release_staged_host(self, buf: StreamedBuffer) -> None:
        if buf.staged_host is None:
            return
        if buf.ready_event is not None:
            buf.ready_event.synchronize()
        self.host_pool.release(buf.staged_host)
        buf.staged_host = None

    def evict_lru(self) -> str | None:
        """Evict the least-recently-used GPU slot and return its key."""
        if not self._slots:
            return None
        key, buf = self._slots.popitem(last=False)
        self._release_staged_host(buf)
        self.vram_used_bytes = max(0, self.vram_used_bytes - buf.nbytes)
        self.evictions += 1
        return key

    def _enforce_backpressure(self, incoming_bytes: int) -> None:
        if self.adaptive_slots and self.max_vram_bytes is not None and incoming_bytes > 0:
            self.num_slots = max(2, min(self._max_slots, self.max_vram_bytes // incoming_bytes))
        while len(self._slots) >= self.num_slots:
            self.evict_lru()
        if self.max_vram_bytes is not None:
            while self._slots and (self.vram_used_bytes + incoming_bytes > self.max_vram_bytes):
                self.evict_lru()

    def prefetch(self, key: str, cpu_tensor: torch.Tensor) -> StreamedBuffer:
        """Schedule asynchronous H2D DMA copy for `key` on `copy_stream`."""
        if key in self._slots:
            self._slots.move_to_end(key)
            return self._slots[key]

        nbytes = int(cpu_tensor.numel() * cpu_tensor.element_size())
        self._enforce_backpressure(nbytes)

        event: torch.cuda.Event | None = None
        staged_host: torch.Tensor | None = None
        if self.use_cuda and self.copy_stream is not None:
            pinned = self.host_pool.stage_pinned(cpu_tensor)
            with torch.cuda.stream(self.copy_stream):
                gpu_t = torch.empty_like(pinned, device=self.device)
                gpu_t.copy_(pinned, non_blocking=True)
            event = torch.cuda.Event()
            event.record(self.copy_stream)
            if not cpu_tensor.is_pinned():
                staged_host = pinned
        else:
            gpu_t = cpu_tensor.detach().clone().to(self.device)

        self.vram_used_bytes += nbytes
        self.total_bytes_streamed += nbytes
        buf = StreamedBuffer(
            key=key,
            gpu_tensor=gpu_t,
            ready_event=event,
            nbytes=nbytes,
            staged_host=staged_host,
        )
        self._slots[key] = buf
        return buf

    def acquire(
        self,
        key: str,
        cpu_tensor: torch.Tensor | None = None,
        compute_stream: torch.cuda.Stream | None = None,
    ) -> torch.Tensor:
        """Wait for DMA completion on `compute_stream` and return the GPU tensor."""
        if key not in self._slots:
            if cpu_tensor is None:
                raise KeyError(f"Tensor '{key}' not prefetched and no cpu_tensor provided")
            self.prefetch(key, cpu_tensor)

        self._slots.move_to_end(key)
        buf = self._slots[key]
        if self.use_cuda and buf.ready_event is not None:
            stream = compute_stream or torch.cuda.current_stream(self.device)
            stream.wait_event(buf.ready_event)
        return buf.gpu_tensor

    def release(self, key: str) -> None:
        buf = self._slots.pop(key, None)
        if buf is not None:
            self._release_staged_host(buf)
            self.vram_used_bytes = max(0, self.vram_used_bytes - buf.nbytes)

    def clear(self) -> None:
        for buf in self._slots.values():
            self._release_staged_host(buf)
        self._slots.clear()
        self.vram_used_bytes = 0


def p2p_capable(src_device: int = 0, dst_device: int = 1) -> bool:
    """Return True if CUDA peer-to-peer DMA access is supported between `src` and `dst`."""
    if not torch.cuda.is_available() or torch.cuda.device_count() <= max(src_device, dst_device):
        return False
    try:
        return bool(torch.cuda.can_device_access_peer(src_device, dst_device))
    except Exception:
        return False
