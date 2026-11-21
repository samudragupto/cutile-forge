"""Size-and-dtype-bucketed reusable pinned host memory pool for async DMA staging."""

from __future__ import annotations

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
