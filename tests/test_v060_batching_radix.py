"""Unit and GPU tests for CuTileForge v0.6.0 (RadixPrefixCache & DynamicBatcher)."""

from __future__ import annotations

import asyncio

import torch

from cutile_forge.engine.config import ModelConfig
from cutile_forge.engine.inference import InferenceEngine
from cutile_forge.engine.model import Mistral7B
from cutile_forge.engine.radix_cache import RadixPrefixCache
from cutile_forge.server.app import DynamicBatcher


def test_radix_prefix_cache_hit_and_eviction() -> None:
    cache = RadixPrefixCache(max_entries=2)
    kv = {0: (torch.ones(1, 2, 4, 8), torch.ones(1, 2, 4, 8))}
    logits = torch.ones(1, 1, 16)

    matched, _, _ = cache.match_prefix([1, 2, 3, 4])
    assert matched == 0
    cache.insert([1, 2, 3, 4], kv, logits)

    # Full hit
    matched, got_kv, got_logits = cache.match_prefix([1, 2, 3, 4])
    assert matched == 4
    assert got_kv is not None and got_logits is not None

    # Partial prefix hit
    matched_p, _, _ = cache.match_prefix([1, 2, 3, 4, 5, 6])
    assert matched_p == 4
    assert cache.prefix_cache_hit_rate > 0.5

    # LRU eviction
    cache.insert([10, 11], kv, logits)
    cache.insert([20, 21], kv, logits)
    assert len(cache._entries) == 2


def test_inference_engine_radix_prefill_reuse() -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    cfg = ModelConfig(
        vocab_size=128,
        hidden_size=128,
        intermediate_size=256,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=32,
    )
    model = Mistral7B.from_random(cfg, device=device, seed=11)
    engine = InferenceEngine(model, use_cuda_graph=False, enable_radix_cache=True)
    prompt = torch.tensor([[1, 2, 3, 4, 5, 6]], dtype=torch.long)

    out1 = engine.generate_ids(prompt, max_new_tokens=6, temperature=0.0)
    out2 = engine.generate_ids(prompt, max_new_tokens=6, temperature=0.0)
    assert out1 == out2
    assert engine.radix_cache.hits >= 1
    assert engine.radix_cache.prefix_cache_hit_rate >= 0.5


def test_dynamic_batcher_coalesces_concurrent_requests() -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    cfg = ModelConfig(
        vocab_size=128,
        hidden_size=128,
        intermediate_size=256,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=32,
    )
    model = Mistral7B.from_random(cfg, device=device, seed=17)
    engine = InferenceEngine(model, use_cuda_graph=False)
    batcher = DynamicBatcher(engine, max_batch_size=16, batch_wait_ms=10.0)

    async def _run() -> list[list[int]]:
        tasks = [
            batcher.submit(torch.tensor([[1, 2, 3 + i]], dtype=torch.long), 4, 0.0, 50, 0.9)
            for i in range(4)
        ]
        return await asyncio.gather(*tasks)

    results = asyncio.run(_run())
    assert len(results) == 4
    assert all(len(r) == 4 for r in results)
