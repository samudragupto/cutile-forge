"""Unit tests for PagedKVCache (allocation, eviction, reset, OOM budget)."""

from __future__ import annotations

import torch

from cutile_forge.engine.kv_cache import PagedKVCache


def test_kv_cache_update_and_gather() -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    kv = PagedKVCache(num_layers=2, num_kv_heads=8, head_dim=128, page_size=16, device=device)
    k0 = torch.randn((1, 8, 10, 128), device=device, dtype=torch.float16)
    v0 = torch.randn((1, 8, 10, 128), device=device, dtype=torch.float16)
    n = kv.update(0, k0, v0)
    assert n == 10
    k_out, v_out = kv.get_kv(0)
    assert k_out.shape == (1, 8, 10, 128)
    assert torch.allclose(k_out, k0)
    assert torch.allclose(v_out, v0)


def test_kv_cache_multi_page_append() -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    kv = PagedKVCache(num_layers=1, num_kv_heads=8, head_dim=64, page_size=8, device=device)
    k1 = torch.randn((1, 8, 6, 64), device=device, dtype=torch.float16)
    v1 = torch.randn((1, 8, 6, 64), device=device, dtype=torch.float16)
    k2 = torch.randn((1, 8, 6, 64), device=device, dtype=torch.float16)
    v2 = torch.randn((1, 8, 6, 64), device=device, dtype=torch.float16)
    kv.update(0, k1, v1)
    n = kv.update(0, k2, v2)
    assert n == 12
    k_all, v_all = kv.get_kv(0)
    assert torch.allclose(k_all, torch.cat([k1, k2], dim=2))
    assert torch.allclose(v_all, torch.cat([v1, v2], dim=2))


def test_kv_cache_eviction_on_tight_vram_budget() -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # Budget allows only 1 page (page_size=4, 8 heads, head_dim=64 -> 2*8*4*64*2 = 8192 bytes)
    page_bytes = 2 * 8 * 4 * 64 * 2
    kv = PagedKVCache(
        num_layers=2,
        num_kv_heads=8,
        head_dim=64,
        page_size=4,
        gpu_budget_bytes=page_bytes,
        device=device,
    )
    k = torch.randn((1, 8, 12, 64), device=device, dtype=torch.float16)
    v = torch.randn((1, 8, 12, 64), device=device, dtype=torch.float16)
    kv.update(0, k, v)
    if torch.cuda.is_available():
        assert kv.evictions >= 1
        assert kv.gpu_used_bytes <= page_bytes
    k_got, v_got = kv.get_kv(0)
    assert torch.allclose(k_got, k)
    assert torch.allclose(v_got, v)


def test_kv_cache_empty_layer() -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    kv = PagedKVCache(num_layers=2, num_kv_heads=8, head_dim=64, device=device)
    k, v = kv.get_kv(1)
    assert k.shape == (1, 8, 0, 64)
    assert v.shape == (1, 8, 0, 64)


def test_kv_cache_reset() -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    kv = PagedKVCache(num_layers=1, num_kv_heads=4, head_dim=32, device=device)
    kv.update(
        0,
        torch.ones((1, 4, 3, 32), device=device, dtype=torch.float16),
        torch.ones((1, 4, 3, 32), device=device, dtype=torch.float16),
    )
    kv.reset()
    assert kv.seq_len == 0
    assert kv.gpu_used_bytes == 0
    assert len(kv.pages) == 0


def test_kv_cache_invalid_page_size() -> None:
    import pytest

    with pytest.raises(ValueError, match="page_size"):
        PagedKVCache(num_layers=1, num_kv_heads=4, head_dim=32, page_size=0)
