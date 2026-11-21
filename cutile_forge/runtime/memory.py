"""Size-and-dtype-bucketed reusable pinned host memory pool for async DMA staging."""

from __future__ import annotations

import math

import torch


class PinnedHostPool:
    """Reusable pool of page-locked CPU tensors keyed by `(shape, dtype)`."""

    def __init__(self, max_cached_per_bucket: int = 8) -> None:
        self.max_cached_per_bucket = max(1, max_cached_per_bucket)
        self._buckets: dict[tuple[tuple[int, ...], torch.dtype], list[torch.Tensor]] = {}
        self.hits: int = 0
        self.misses: int = 0
        self.allocations: int = 0

    def acquire(self, shape: tuple[int, ...], dtype: torch.dtype) -> torch.Tensor:
        """Obtain a contiguous host buffer (pinned when CUDA is available)."""
        key = (tuple(shape), dtype)
        bucket = self._buckets.get(key)
        if bucket:
            self.hits += 1
            return bucket.pop()

        self.misses += 1
        self.allocations += 1
        pin = torch.cuda.is_available()
        try:
            return torch.empty(shape, dtype=dtype, pin_memory=pin)
        except RuntimeError:
            return torch.empty(shape, dtype=dtype)

    def release(self, tensor: torch.Tensor) -> None:
        """Return a host buffer to the pool for future DMA reuse."""
        if tensor.device.type != "cpu":
            return
        key = (tuple(tensor.shape), tensor.dtype)
        bucket = self._buckets.setdefault(key, [])
        if len(bucket) < self.max_cached_per_bucket:
            bucket.append(tensor)

    def stage_pinned(self, tensor: torch.Tensor) -> torch.Tensor:
        """Return a pinned contiguous copy of `tensor` (reusing pool buffers when possible)."""
        cpu_t = tensor.detach().cpu().contiguous()
        if cpu_t.is_pinned():
            return cpu_t
        buf = self.acquire(tuple(cpu_t.shape), cpu_t.dtype)
        buf.copy_(cpu_t)
        return buf

    def clear(self) -> None:
        self._buckets.clear()
        self.hits = 0
        self.misses = 0
        self.allocations = 0


class MemoryLifetimeAllocator:
    """Slot-based buffer allocator reusing non-overlapping lifetime pools across TFG ops."""

    def __init__(self) -> None:
        self._slots: dict[int, torch.Tensor] = {}
        self.peak_bytes: int = 0
        self.total_requested_bytes: int = 0
        self.reuse_hits: int = 0

    def allocate_slot(
        self,
        slot_id: int,
        shape: tuple[int, ...],
        dtype: torch.dtype,
        device: str | torch.device = "cpu",
    ) -> torch.Tensor:
        """Return a tensor view over `slot_id`'s backing buffer, growing on demand."""
        req_dev = torch.device(
            device if (str(device) != "cuda" or torch.cuda.is_available()) else "cpu"
        )
        if req_dev.type == "cuda" and req_dev.index is None:
            req_dev = torch.device("cuda", torch.cuda.current_device())
        numel = math.prod(shape)
        elem_size = torch.tensor([], dtype=dtype).element_size()
        req_bytes = numel * elem_size
        self.total_requested_bytes += req_bytes

        buf = self._slots.get(slot_id)
        if buf is not None and buf.device == req_dev and buf.numel() >= req_bytes:
            self.reuse_hits += 1
        else:
            new_buf = torch.empty((max(1, req_bytes),), dtype=torch.uint8, device=req_dev)
            self._slots[slot_id] = new_buf
            buf = new_buf
            live_bytes = sum(t.numel() for t in self._slots.values())
            self.peak_bytes = max(self.peak_bytes, live_bytes)

        return buf[:req_bytes].view(dtype).view(shape)

    def reuse_ratio(self) -> float:
        """Fraction of total requested memory avoided via slot buffer reuse."""
        if self.total_requested_bytes <= 0:
            return 0.0
        return max(0.0, 1.0 - (self.peak_bytes / float(self.total_requested_bytes)))

    def reset(self) -> None:
        self._slots.clear()
        self.peak_bytes = 0
        self.total_requested_bytes = 0
        self.reuse_hits = 0
