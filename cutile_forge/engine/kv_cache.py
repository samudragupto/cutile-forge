"""Paged KV-cache spanning GPU HBM and pinned CPU DDR5 with LRU page eviction."""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass

import torch


@dataclass
class KVPage:
    """Single fixed-size page storing `(1, H_kv, page_size, D)` K and V tensors."""

    layer_idx: int
    page_idx: int
    k: torch.Tensor
    v: torch.Tensor
    valid_tokens: int
    on_gpu: bool

    @property
    def nbytes(self) -> int:
        return self.k.numel() * self.k.element_size() + self.v.numel() * self.v.element_size()


class PagedKVCache:
    """Paged Key-Value cache with bounded GPU HBM budget and automatic DDR5 offloading."""

    def __init__(
        self,
        num_layers: int,
        num_kv_heads: int,
        head_dim: int,
        page_size: int = 16,
        gpu_budget_bytes: int = 2 * 1024 * 1024 * 1024,
        device: str | torch.device = "cuda",
        dtype: torch.dtype = torch.float16,
    ) -> None:
        if page_size <= 0:
            raise ValueError(f"page_size must be positive, got {page_size}")
        self.num_layers = num_layers
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.page_size = page_size
        self.gpu_budget_bytes = gpu_budget_bytes
        self.device = torch.device(
            device if (str(device) != "cuda" or torch.cuda.is_available()) else "cpu"
        )
        self.dtype = dtype

        self.pages: dict[tuple[int, int], KVPage] = {}
        self.gpu_lru: OrderedDict[tuple[int, int], int] = OrderedDict()
        self.gpu_used_bytes: int = 0
        self.seq_len: int = 0
        self.evictions: int = 0
        self.host_fetches: int = 0

    def _evict_if_needed(self, required_bytes: int) -> None:
        while self.gpu_used_bytes + required_bytes > self.gpu_budget_bytes and self.gpu_lru:
            evict_key, _ = self.gpu_lru.popitem(last=False)
            page = self.pages[evict_key]
            if page.on_gpu:
                k_cpu = page.k.detach().cpu()
                v_cpu = page.v.detach().cpu()
                if torch.cuda.is_available():
                    try:
                        k_cpu = k_cpu.pin_memory()
                        v_cpu = v_cpu.pin_memory()
                    except RuntimeError:
                        pass
                self.gpu_used_bytes -= page.nbytes
                page.k = k_cpu
                page.v = v_cpu
                page.on_gpu = False
                self.evictions += 1

    def _ensure_on_gpu(self, key: tuple[int, int]) -> KVPage:
        page = self.pages[key]
        if not page.on_gpu and self.device.type == "cuda":
            self._evict_if_needed(page.nbytes)
            page.k = page.k.to(self.device, non_blocking=True)
            page.v = page.v.to(self.device, non_blocking=True)
            page.on_gpu = True
            self.gpu_used_bytes += page.nbytes
            self.host_fetches += 1
        if key in self.gpu_lru:
            self.gpu_lru.move_to_end(key)
        elif page.on_gpu:
            self.gpu_lru[key] = page.nbytes
        return page

    def update(self, layer_idx: int, k_new: torch.Tensor, v_new: torch.Tensor) -> int:
        """Append `(B=1, H_kv, S_new, D)` key/value tensors for `layer_idx`.

        Returns the new total sequence length stored for `layer_idx`.
        """
        if k_new.ndim != 4 or k_new.shape[0] != 1:
            raise ValueError(f"Expected (1, H_kv, S_new, D), got {k_new.shape}")
        s_new = k_new.shape[2]
        start_pos = self._layer_seq_len(layer_idx)

        for offset in range(s_new):
            pos = start_pos + offset
            page_idx = pos // self.page_size
            slot = pos % self.page_size
            key = (layer_idx, page_idx)

            if key not in self.pages:
                on_gpu = self.device.type == "cuda"
                page_bytes = 2 * self.num_kv_heads * self.page_size * self.head_dim * 2
                if on_gpu:
                    self._evict_if_needed(page_bytes)
                    if self.gpu_used_bytes + page_bytes > self.gpu_budget_bytes:
                        on_gpu = False
                alloc_dev = self.device if on_gpu else torch.device("cpu")
                k_buf = torch.zeros(
                    (1, self.num_kv_heads, self.page_size, self.head_dim),
                    device=alloc_dev,
                    dtype=self.dtype,
                )
                v_buf = torch.zeros_like(k_buf)
                page = KVPage(
                    layer_idx=layer_idx,
                    page_idx=page_idx,
                    k=k_buf,
                    v=v_buf,
                    valid_tokens=0,
                    on_gpu=on_gpu,
                )
                self.pages[key] = page
                if on_gpu:
                    self.gpu_used_bytes += page.nbytes
                    self.gpu_lru[key] = page.nbytes

            page = self._ensure_on_gpu(key)
            page.k[:, :, slot : slot + 1, :] = k_new[:, :, offset : offset + 1, :].to(page.k.device)
            page.v[:, :, slot : slot + 1, :] = v_new[:, :, offset : offset + 1, :].to(page.v.device)
            page.valid_tokens = max(page.valid_tokens, slot + 1)

        new_len = start_pos + s_new
        if layer_idx == 0:
            self.seq_len = new_len
        return new_len

    def _layer_seq_len(self, layer_idx: int) -> int:
        layer_pages = [p for (l_idx, _), p in self.pages.items() if l_idx == layer_idx]
        if not layer_pages:
            return 0
        max_page = max(layer_pages, key=lambda p: p.page_idx)
        return max_page.page_idx * self.page_size + max_page.valid_tokens

    def get_kv(self, layer_idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Gather full `(1, H_kv, S_kv, D)` K and V tensors on `self.device`."""
        layer_keys = sorted(k for k in self.pages if k[0] == layer_idx)
        if not layer_keys:
            empty = torch.empty(
                (1, self.num_kv_heads, 0, self.head_dim),
                device=self.device,
                dtype=self.dtype,
            )
            return empty, empty

        k_chunks: list[torch.Tensor] = []
        v_chunks: list[torch.Tensor] = []
        for key in layer_keys:
            page = self._ensure_on_gpu(key)
            n = page.valid_tokens
            k_chunks.append(page.k[:, :, :n, :].to(self.device))
            v_chunks.append(page.v[:, :, :n, :].to(self.device))
        return torch.cat(k_chunks, dim=2), torch.cat(v_chunks, dim=2)

    def reset(self) -> None:
        self.pages.clear()
        self.gpu_lru.clear()
        self.gpu_used_bytes = 0
        self.seq_len = 0
        self.evictions = 0
        self.host_fetches = 0
