"""Unit tests for `ArtifactCache` and cached `compile_tfg`."""

from __future__ import annotations

from cutile_forge.forge.artifact_cache import (
    GLOBAL_ARTIFACT_CACHE,
    ArtifactCache,
    compute_graph_hash,
)
from cutile_forge.forge.codegen import compile_tfg
from cutile_forge.forge.fusion_graph import build_mistral_layer_tfg


def test_artifact_cache_lru_and_hits() -> None:
    cache = ArtifactCache(max_entries=2)
    assert cache.get("missing") is None
    assert cache.misses == 1

    cache.put("k1", "v1")
    cache.put("k2", "v2")
    assert cache.get("k1") == "v1"  # k1 moves to MRU, k2 becomes LRU
    cache.put("k3", "v3")  # evicts k2
    assert cache.get("k2") is None
    assert cache.get("k3") == "v3"
    assert cache.hits == 2


def test_compile_tfg_uses_cache_when_enabled() -> None:
    GLOBAL_ARTIFACT_CACHE.clear()
    g1 = build_mistral_layer_tfg(0, seq_len=128)
    g2 = build_mistral_layer_tfg(0, seq_len=128)
    assert compute_graph_hash(g1, "sm_120", 4) == compute_graph_hash(g2, "sm_120", 4)

    first = compile_tfg(g1, target="sm_120", pipeline_depth=4, run_passes=True, use_cache=True)
    second = compile_tfg(g2, target="sm_120", pipeline_depth=4, run_passes=True, use_cache=True)
    assert first.cache_hit is False
    assert second.cache_hit is True
    assert len(second.kernels) == 3
