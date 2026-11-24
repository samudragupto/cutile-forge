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
    swap_event: torch.cuda.Event | None = None
    k_scale: float = 1.0
    v_scale: float = 1.0
    k_scales: torch.Tensor | None = None
    v_scales: torch.Tensor | None = None

    @property
    def nbytes(self) -> int:
        total = self.k.numel() * self.k.element_size() + self.v.numel() * self.v.element_size()
        if self.k_scales is not None:
            total += self.k_scales.numel() * self.k_scales.element_size()
        if self.v_scales is not None:
            total += self.v_scales.numel() * self.v_scales.element_size()
        return total


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
        cpu_pool_pages: int = 64,
        kv_dtype: str = "fp16",
        sliding_window: int | None = None,
    ) -> None:
        if page_size <= 0:
            raise ValueError(f"page_size must be positive, got {page_size}")
        if kv_dtype not in ("fp16", "fp8_e4m3fn"):
            raise ValueError(f"Unsupported kv_dtype: {kv_dtype}")
        self.num_layers = num_layers
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.page_size = page_size
        self.sliding_window = (
            sliding_window if (sliding_window is not None and sliding_window > 0) else None
        )
        self.gpu_budget_bytes = gpu_budget_bytes
        self.device = torch.device(
            device if (str(device) != "cuda" or torch.cuda.is_available()) else "cpu"
        )
        self.kv_dtype = kv_dtype
        self.storage_dtype = torch.float8_e4m3fn if kv_dtype == "fp8_e4m3fn" else dtype
        self.dtype = dtype

        self.pages: dict[tuple[int, int], KVPage] = {}
        self.gpu_lru: OrderedDict[tuple[int, int], int] = OrderedDict()
        self._layer_lens: dict[int, int] = {}
        self.gpu_used_bytes: int = 0
        self.seq_len: int = 0
        self.evictions: int = 0
        self.host_fetches: int = 0
        self.runtime_pin_allocs: int = 0

        self.swap_stream: torch.cuda.Stream | None = (
            torch.cuda.Stream()
            if self.device.type == "cuda" and torch.cuda.is_available()
            else None
        )
        self._cpu_pool: list[tuple[torch.Tensor, torch.Tensor]] = []
        self._prefetched: dict[
            tuple[int, int], tuple[torch.Tensor, torch.Tensor, torch.cuda.Event]
        ] = {}
        if self.device.type == "cuda" and torch.cuda.is_available() and cpu_pool_pages > 0:
            for _ in range(cpu_pool_pages):
                self._cpu_pool.append(self._alloc_pinned_pair())

    def _alloc_pinned_pair(self) -> tuple[torch.Tensor, torch.Tensor]:
        shape = (1, self.num_kv_heads, self.page_size, self.head_dim)
        pin = torch.cuda.is_available()
        try:
            k_cpu = torch.empty(shape, dtype=self.storage_dtype, pin_memory=pin)
            v_cpu = torch.empty(shape, dtype=self.storage_dtype, pin_memory=pin)
        except RuntimeError:
            k_cpu = torch.empty(shape, dtype=self.storage_dtype)
            v_cpu = torch.empty(shape, dtype=self.storage_dtype)
        return k_cpu, v_cpu

    def _acquire_cpu_pair(self) -> tuple[torch.Tensor, torch.Tensor]:
        if self._cpu_pool:
            return self._cpu_pool.pop()
        self.runtime_pin_allocs += 1
        return self._alloc_pinned_pair()

    def _release_cpu_pair(self, k_cpu: torch.Tensor, v_cpu: torch.Tensor) -> None:
        if k_cpu.device.type == "cpu":
            self._cpu_pool.append((k_cpu, v_cpu))

    def _evict_if_needed(self, required_bytes: int) -> None:
        if self.gpu_used_bytes + required_bytes <= self.gpu_budget_bytes or not self.gpu_lru:
            return
        if self.swap_stream is not None:
            self.swap_stream.wait_stream(torch.cuda.current_stream())
        while self.gpu_used_bytes + required_bytes > self.gpu_budget_bytes and self.gpu_lru:
            evict_key, _ = self.gpu_lru.popitem(last=False)
            page = self.pages[evict_key]
            if page.on_gpu:
                k_cpu, v_cpu = self._acquire_cpu_pair()
                if self.swap_stream is not None:
                    page.k.record_stream(self.swap_stream)
                    page.v.record_stream(self.swap_stream)
                    with torch.cuda.stream(self.swap_stream):
                        k_cpu.copy_(page.k, non_blocking=True)
                        v_cpu.copy_(page.v, non_blocking=True)
                    page.swap_event = self.swap_stream.record_event()
                else:
                    k_cpu.copy_(page.k)
                    v_cpu.copy_(page.v)
                    page.swap_event = None
                self.gpu_used_bytes -= page.nbytes
                page.k = k_cpu
                page.v = v_cpu
                page.on_gpu = False
                self.evictions += 1

    def prefetch_page_async(self, page_id: tuple[int, int]) -> None:
        """Asynchronously prefetch an offloaded CPU page onto GPU on `swap_stream`."""
        page = self.pages.get(page_id)
        if page is None or page.on_gpu or self.device.type != "cuda" or page_id in self._prefetched:
            return
        if page.swap_event is not None and self.swap_stream is not None:
            self.swap_stream.wait_event(page.swap_event)
        if self.swap_stream is not None:
            with torch.cuda.stream(self.swap_stream):
                k_gpu = page.k.to(self.device, non_blocking=True)
                v_gpu = page.v.to(self.device, non_blocking=True)
            ev = self.swap_stream.record_event()
            self._prefetched[page_id] = (k_gpu, v_gpu, ev)

    def _ensure_on_gpu(self, key: tuple[int, int]) -> KVPage:
        page = self.pages[key]
        if not page.on_gpu and self.device.type == "cuda":
            self._evict_if_needed(page.nbytes)
            old_k, old_v = page.k, page.v
            if key in self._prefetched:
                k_gpu, v_gpu, ev = self._prefetched.pop(key)
                torch.cuda.current_stream().wait_event(ev)
            else:
                if page.swap_event is not None:
                    torch.cuda.current_stream().wait_event(page.swap_event)
                if self.swap_stream is not None:
                    with torch.cuda.stream(self.swap_stream):
                        k_gpu = old_k.to(self.device, non_blocking=True)
                        v_gpu = old_v.to(self.device, non_blocking=True)
                    ev = self.swap_stream.record_event()
                    torch.cuda.current_stream().wait_event(ev)
                else:
                    k_gpu = old_k.to(self.device, non_blocking=True)
                    v_gpu = old_v.to(self.device, non_blocking=True)
            self._release_cpu_pair(old_k, old_v)
            page.k = k_gpu
            page.v = v_gpu
            page.swap_event = None
            page.on_gpu = True
            self.gpu_used_bytes += page.nbytes
            self.host_fetches += 1
        if key in self.gpu_lru:
            self.gpu_lru.move_to_end(key)
        elif page.on_gpu:
            self.gpu_lru[key] = page.nbytes
        return page

    def update(
        self,
        layer_idx: int,
        k_new: torch.Tensor,
        v_new: torch.Tensor,
        defer_prune: bool = False,
    ) -> int:
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
                elem_bytes = 1 if self.kv_dtype == "fp8_e4m3fn" else 2
                scale_bytes = 2 * self.page_size * 4 if self.kv_dtype == "fp8_e4m3fn" else 0
                page_bytes = (
                    2 * self.num_kv_heads * self.page_size * self.head_dim * elem_bytes
                    + scale_bytes
                )
                if on_gpu:
                    self._evict_if_needed(page_bytes)
                    if self.gpu_used_bytes + page_bytes > self.gpu_budget_bytes:
                        on_gpu = False
                alloc_dev = self.device if on_gpu else torch.device("cpu")
                k_buf = torch.zeros(
                    (1, self.num_kv_heads, self.page_size, self.head_dim),
                    device=alloc_dev,
                    dtype=self.storage_dtype,
                )
                v_buf = torch.zeros_like(k_buf)
                k_scales = (
                    torch.ones((1, 1, self.page_size, 1), device=alloc_dev, dtype=torch.float32)
                    if self.kv_dtype == "fp8_e4m3fn"
                    else None
                )
                v_scales = (
                    torch.ones((1, 1, self.page_size, 1), device=alloc_dev, dtype=torch.float32)
                    if self.kv_dtype == "fp8_e4m3fn"
                    else None
                )
                page = KVPage(
                    layer_idx=layer_idx,
                    page_idx=page_idx,
                    k=k_buf,
                    v=v_buf,
                    valid_tokens=0,
                    on_gpu=on_gpu,
                    k_scales=k_scales,
                    v_scales=v_scales,
                )
                self.pages[key] = page
                if on_gpu:
                    self.gpu_used_bytes += page.nbytes
                    self.gpu_lru[key] = page.nbytes

            page = self._ensure_on_gpu(key)
            k_slice = k_new[:, :, offset : offset + 1, :].to(page.k.device)
            v_slice = v_new[:, :, offset : offset + 1, :].to(page.v.device)
            if self.kv_dtype == "fp8_e4m3fn":
                k_max = float(k_slice.abs().max().item())
                v_max = float(v_slice.abs().max().item())
                req_k_scale = max(k_max / 448.0, 1e-6)
                req_v_scale = max(v_max / 448.0, 1e-6)
                if page.valid_tokens == 0:
                    page.k_scale = req_k_scale
                    page.v_scale = req_v_scale
                else:
                    page.k_scale = max(page.k_scale, req_k_scale)
                    page.v_scale = max(page.v_scale, req_v_scale)
                if page.k_scales is not None and page.v_scales is not None:
                    page.k_scales[:, :, slot : slot + 1, :].fill_(req_k_scale)
                    page.v_scales[:, :, slot : slot + 1, :].fill_(req_v_scale)
                page.k[:, :, slot : slot + 1, :] = (
                    (k_slice.float() / req_k_scale).clamp(-448.0, 448.0).to(torch.float8_e4m3fn)
                )
                page.v[:, :, slot : slot + 1, :] = (
                    (v_slice.float() / req_v_scale).clamp(-448.0, 448.0).to(torch.float8_e4m3fn)
                )
            else:
                page.k[:, :, slot : slot + 1, :] = k_slice
                page.v[:, :, slot : slot + 1, :] = v_slice
            page.valid_tokens = max(page.valid_tokens, slot + 1)

        new_len = start_pos + s_new
        self._layer_lens[layer_idx] = new_len
        if layer_idx == 0:
            self.seq_len = new_len
        if self.sliding_window is not None and not defer_prune:
            self.prune_outside_window(layer_idx)
        return new_len

    def prune_outside_window(self, layer_idx: int) -> None:
        """Drop pages strictly older than `layer_len - sliding_window`."""
        if self.sliding_window is None:
            return
        cur_len = self._layer_seq_len(layer_idx)
        min_needed_pos = cur_len - self.sliding_window
        if min_needed_pos > 0:
            self.prune_before_pos(layer_idx, min_needed_pos)

    def prune_before_pos(self, layer_idx: int, min_pos: int) -> None:
        """Drop pages whose token positions all precede `min_pos`."""
        min_keep_page = min_pos // self.page_size
        stale = [k for k in self.pages if k[0] == layer_idx and k[1] < min_keep_page]
        for k in stale:
            page = self.pages.pop(k)
            self.gpu_lru.pop(k, None)
            if page.on_gpu:
                self.gpu_used_bytes -= page.nbytes
            else:
                self._release_cpu_pair(page.k, page.v)

    def seq_lens_valid(self, layer_idx: int = 0) -> int:
        """Return number of valid tokens currently resident in pages for `layer_idx`."""
        return sum(p.valid_tokens for (l_idx, _), p in self.pages.items() if l_idx == layer_idx)

    def _layer_seq_len(self, layer_idx: int) -> int:
        if layer_idx in self._layer_lens:
            return self._layer_lens[layer_idx]
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
            page = self.pages[key]
            n = page.valid_tokens
            if page.on_gpu:
                self.gpu_lru.move_to_end(key)
                k_t = page.k[:, :, :n, :]
                v_t = page.v[:, :, :n, :]
            else:
                if page.swap_event is not None:
                    torch.cuda.current_stream().wait_event(page.swap_event)
                k_t = page.k[:, :, :n, :].to(self.device, non_blocking=True)
                v_t = page.v[:, :, :n, :].to(self.device, non_blocking=True)
                self.host_fetches += 1
            if self.kv_dtype == "fp8_e4m3fn":
                ks = (
                    page.k_scales[:, :, :n, :].to(device=k_t.device, dtype=torch.float32)
                    if page.k_scales is not None
                    else page.k_scale
                )
                vs = (
                    page.v_scales[:, :, :n, :].to(device=v_t.device, dtype=torch.float32)
                    if page.v_scales is not None
                    else page.v_scale
                )
                k_t = (k_t.float() * ks).to(self.dtype)
                v_t = (v_t.float() * vs).to(self.dtype)
            k_chunks.append(k_t)
            v_chunks.append(v_t)
        return torch.cat(k_chunks, dim=2), torch.cat(v_chunks, dim=2)

    def truncate(self, new_len: int) -> None:
        """Truncate all layers back to `new_len` tokens (used by speculative rollback)."""
        if new_len < 0:
            new_len = 0
        max_page_idx = (new_len - 1) // self.page_size if new_len > 0 else -1
        last_valid = ((new_len - 1) % self.page_size) + 1 if new_len > 0 else 0
        to_del = [k for k in self.pages if k[1] > max_page_idx]
        for k in to_del:
            page = self.pages.pop(k)
            self.gpu_lru.pop(k, None)
            if page.on_gpu:
                self.gpu_used_bytes -= page.nbytes
            else:
                self._release_cpu_pair(page.k, page.v)
        for k, page in self.pages.items():
            if k[1] == max_page_idx:
                page.valid_tokens = min(page.valid_tokens, last_valid)
        for l_idx in list(self._layer_lens):
            self._layer_lens[l_idx] = min(self._layer_lens[l_idx], new_len)
        self.seq_len = new_len

    def reset(self) -> None:
        for page in self.pages.values():
            if not page.on_gpu:
                self._release_cpu_pair(page.k, page.v)
        self._prefetched.clear()
        self.pages.clear()
        self.gpu_lru.clear()
        self._layer_lens.clear()
        self.gpu_used_bytes = 0
        self.seq_len = 0
        self.evictions = 0
        self.host_fetches = 0
