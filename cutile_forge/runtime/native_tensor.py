"""Zero-copy raw buffer tensor wrapper (`CuTileTensor`) backed by DLPack and MemoryLifetimeAllocator."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import torch

from cutile_forge.runtime.memory import MemoryLifetimeAllocator

_TORCH_TO_DLPACK_DTYPE: dict[torch.dtype, str] = {
    torch.float16: "float16",
    torch.bfloat16: "bfloat16",
    torch.float32: "float32",
    torch.int32: "int32",
    torch.int64: "int64",
    torch.uint8: "uint8",
}


@dataclass
class CuTileTensor:
    """Lightweight CuTile tensor view with DLPack exchange and slot-pooled storage."""

    _storage: torch.Tensor
    shape: tuple[int, ...]
    dtype: torch.dtype

    @classmethod
    def from_torch(cls, tensor: torch.Tensor) -> CuTileTensor:
        """Wrap an existing contiguous tensor without copying."""
        c = tensor if tensor.is_contiguous() else tensor.contiguous()
        return cls(_storage=c, shape=tuple(c.shape), dtype=c.dtype)

    @classmethod
    def allocate(
        cls,
        shape: tuple[int, ...],
        dtype: torch.dtype = torch.float16,
        device: str | torch.device = "cuda",
        allocator: MemoryLifetimeAllocator | None = None,
        slot_id: int = 0,
    ) -> CuTileTensor:
        """Allocate or reuse a contiguous buffer from `MemoryLifetimeAllocator`."""
        req_dev = torch.device(
            device if (str(device) != "cuda" or torch.cuda.is_available()) else "cpu"
        )
        if req_dev.type == "cuda" and req_dev.index is None:
            req_dev = torch.device("cuda", torch.cuda.current_device())
        if allocator is not None:
            buf = allocator.allocate_slot(slot_id, shape, dtype, device=req_dev)
        else:
            buf = torch.empty(shape, dtype=dtype, device=req_dev)
        return cls(_storage=buf, shape=tuple(shape), dtype=dtype)

    @property
    def device(self) -> torch.device:
        return self._storage.device

    @property
    def strides(self) -> tuple[int, ...]:
        return tuple(self._storage.stride())

    @property
    def numel(self) -> int:
        return math.prod(self.shape)

    @property
    def nbytes(self) -> int:
        return self.numel * self._storage.element_size()

    def data_ptr(self) -> int:
        return int(self._storage.data_ptr())

    def as_torch_view(self) -> torch.Tensor:
        """Return underlying zero-copy `torch.Tensor` descriptor for `ct.launch`."""
        return self._storage

    def __dlpack__(self, stream: int | None = None) -> Any:
        return self._storage.__dlpack__(stream=stream)

    def __dlpack_device__(self) -> tuple[int, int]:
        return self._storage.__dlpack_device__()

    @classmethod
    def from_dlpack(cls, external: Any) -> CuTileTensor:
        """Zero-copy import from any DLPack producer (`torch.Tensor`, CuPy, etc.)."""
        t = torch.from_dlpack(external)
        return cls.from_torch(t)
