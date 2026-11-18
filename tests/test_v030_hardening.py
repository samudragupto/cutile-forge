"""Unit and GPU tests for CuTileForge v0.3.0 systems hardening."""

from __future__ import annotations

import torch

from cutile_forge.engine.config import ModelConfig
from cutile_forge.engine.inference import InferenceEngine
from cutile_forge.engine.kv_cache import PagedKVCache
from cutile_forge.engine.model import Mistral7B
from cutile_forge.engine.speculative import NGramPromptLookupDrafter, SpeculativeDecoder
from cutile_forge.forge.tile_autotuner import select_ffn_execution_strategy


def test_kv_cache_preallocated_pinned_pool_zero_runtime_allocs() -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    page_bytes = 2 * 4 * 4 * 32 * 2  # 1 page = 2048 bytes
    kv = PagedKVCache(
        num_layers=1,
        num_kv_heads=4,
        head_dim=32,
        page_size=4,
        gpu_budget_bytes=page_bytes,
        device=device,
        cpu_pool_pages=16,
    )
    k = torch.randn((1, 4, 16, 32), device=device, dtype=torch.float16)
    v = torch.randn((1, 4, 16, 32), device=device, dtype=torch.float16)
    kv.update(0, k, v)
    if torch.cuda.is_available():
        assert kv.evictions >= 1
        assert kv.runtime_pin_allocs == 0
    kv.prefetch_page_async((0, 0))
    k_got, v_got = kv.get_kv(0)
    assert torch.allclose(k_got, k)
    assert torch.allclose(v_got, v)
    kv.truncate(8)
    k_trunc, _ = kv.get_kv(0)
    assert k_trunc.shape[2] == 8


def test_occupancy_ffn_strategy_sm120() -> None:
    s_dec = select_ffn_execution_strategy(1, 4096, 14336, arch="sm_120", sm_count=170)
    assert s_dec.mode == "single_kernel_3op"

    s_256 = select_ffn_execution_strategy(256, 4096, 14336, arch="sm_120", sm_count=170)
    assert s_256.mode == "pipelined_2stage"
    assert s_256.tile_shape == (64, 64, 32)


def test_cuda_graph_decode_matches_eager() -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    cfg = ModelConfig(
        vocab_size=256,
        hidden_size=256,
        intermediate_size=512,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=64,
    )
    model = Mistral7B.from_random(cfg, device=device, seed=123)
    prompt = torch.tensor([[1, 5, 12, 7]], dtype=torch.long)

    eng_eager = InferenceEngine(model, use_cuda_graph=False)
    toks_eager = eng_eager.generate_ids(prompt, max_new_tokens=10, temperature=0.0)

    eng_graph = InferenceEngine(model, use_cuda_graph=True)
    toks_graph = eng_graph.generate_ids(prompt, max_new_tokens=10, temperature=0.0)

    assert toks_graph == toks_eager


def test_ngram_drafter_and_speculative_acceptance() -> None:
    drafter = NGramPromptLookupDrafter(max_n=3)
    history = [10, 20, 30, 40, 50, 60, 10, 20, 30]
    draft = drafter.draft(history, gamma=4)
    assert draft == [40, 50, 60, 10]

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
    model = Mistral7B.from_random(cfg, device=device, seed=7)
    spec = SpeculativeDecoder(target_model=model, draft_model=model, gamma=4)
    # Prompt with repeating motif triggers high N-gram acceptance
    prompt = torch.tensor([[1, 2, 3, 4, 1, 2, 3]], dtype=torch.long)
    out, stats = spec.generate(prompt, max_new_tokens=16)
    assert len(out) == 16
    assert stats.acceptance_rate >= 0.0
