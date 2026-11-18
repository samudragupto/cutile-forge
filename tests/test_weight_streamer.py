"""Unit tests for Stage -1 PCIe DMA WeightStreamer."""

from __future__ import annotations

import torch

from cutile_forge.forge.fusion_graph import build_mistral_layer_tfg
from cutile_forge.forge.weight_streamer import WeightStreamer, attach_weight_streaming


def test_prefetch_and_acquire_small_tensor() -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    streamer = WeightStreamer(num_slots=2, device=device)
    w = {"w": torch.randn((32, 64), dtype=torch.float16)}
    slot = streamer.prefetch_layer(0, w)
    assert slot.bytes_transferred == 32 * 64 * 2
    gpu_w = streamer.acquire_layer(0)
    assert torch.allclose(gpu_w["w"].cpu(), w["w"])
    streamer.release_layer(0)


def test_double_buffer_eviction() -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    streamer = WeightStreamer(num_slots=2, device=device)
    for idx in range(3):
        streamer.prefetch_layer(idx, {"w": torch.ones((8, 8), dtype=torch.float16) * idx})
    assert len(streamer._slots) == 2
    assert 0 not in streamer._slots
    assert 1 in streamer._slots and 2 in streamer._slots


def test_acquire_on_demand_fallback() -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    streamer = WeightStreamer(num_slots=2, device=device)
    w = {"x": torch.arange(16, dtype=torch.float16)}
    out = streamer.acquire_layer(5, host_weights=w)
    assert torch.equal(out["x"].cpu(), w["x"])


def test_attach_weight_streaming_metadata() -> None:
    g = build_mistral_layer_tfg(0)
    attach_weight_streaming(g, prefetch_distance=1)
    assert all(n.prefetch_stage == -1 for n in g.nodes.values())
