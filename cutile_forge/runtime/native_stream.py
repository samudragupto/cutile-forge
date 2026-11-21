"""Dedicated CUDA compute, DMA prefetch, and KV-swap stream + event pool manager."""

from __future__ import annotations

import torch


class NativeStreamManager:
    """Manages non-blocking CUDA streams and reusable `torch.cuda.Event` barriers."""

    def __init__(self, device: str | torch.device = "cuda") -> None:
        self.device = torch.device(
            device if (str(device) != "cuda" or torch.cuda.is_available()) else "cpu"
        )
        if self.device.type == "cuda" and self.device.index is None:
            self.device = torch.device("cuda", torch.cuda.current_device())
        self.use_cuda = self.device.type == "cuda" and torch.cuda.is_available()

        if self.use_cuda:
            self.compute_stream: torch.cuda.Stream | None = torch.cuda.current_stream(self.device)
            self.dma_stream: torch.cuda.Stream | None = torch.cuda.Stream(device=self.device)
            self.kv_swap_stream: torch.cuda.Stream | None = torch.cuda.Stream(device=self.device)
        else:
            self.compute_stream = None
            self.dma_stream = None
            self.kv_swap_stream = None

        self._event_pool: list[torch.cuda.Event] = []

    def acquire_event(self) -> torch.cuda.Event | None:
        if not self.use_cuda:
            return None
        if self._event_pool:
            return self._event_pool.pop()
        return torch.cuda.Event(enable_timing=False)

    def release_event(self, ev: torch.cuda.Event | None) -> None:
        if ev is not None and len(self._event_pool) < 64:
            self._event_pool.append(ev)

    def record_dma_ready(self) -> torch.cuda.Event | None:
        ev = self.acquire_event()
        if ev is not None and self.dma_stream is not None:
            ev.record(self.dma_stream)
        return ev

    def wait_dma_on_compute(self, ev: torch.cuda.Event | None) -> None:
        if ev is not None and self.compute_stream is not None:
            self.compute_stream.wait_event(ev)
            self.release_event(ev)

    def record_kv_ready(self) -> torch.cuda.Event | None:
        ev = self.acquire_event()
        if ev is not None and self.kv_swap_stream is not None:
            ev.record(self.kv_swap_stream)
        return ev

    def wait_kv_on_compute(self, ev: torch.cuda.Event | None) -> None:
        if ev is not None and self.compute_stream is not None:
            self.compute_stream.wait_event(ev)
            self.release_event(ev)

    def synchronize_all(self) -> None:
        if self.use_cuda:
            if self.dma_stream is not None:
                self.dma_stream.synchronize()
            if self.kv_swap_stream is not None:
                self.kv_swap_stream.synchronize()
            if self.compute_stream is not None:
                self.compute_stream.synchronize()
