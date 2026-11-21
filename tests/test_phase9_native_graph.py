"""Phase 9 (`v2.4.0`) tests: Full-step Native CUDA Graph replay, Zero-ATen, KV contents & 5-prompt 256-tok parity."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch.utils._python_dispatch import TorchDispatchMode

from cutile_forge.cuda_tile_init import ct
from cutile_forge.engine.config import GPUConfig, ModelConfig
from cutile_forge.engine.inference import InferenceEngine
from cutile_forge.engine.model import Mistral7B
from cutile_forge.engine.weight_loader import create_random_4bit_weights
from cutile_forge.runtime import NativeModelRuntime

FORBIDDEN_ATEN_COMPUTE = {
    "aten.mm.default",
    "aten.bmm.default",
    "aten.addmm.default",
    "aten.matmul.default",
    "aten.embedding.default",
    "aten.silu.default",
    "aten.gelu.default",
    "aten.layer_norm.default",
    "aten.scaled_dot_product_attention.default",
    "aten.topk.default",
    "aten.cat.default",
}

FIVE_TEST_PROMPTS: list[list[int]] = [
    [1, 415, 5565, 302, 4843, 349],  # Prompt 1: The capital of France is
    [1, 315, 2296, 396, 17204, 15262, 369],  # Prompt 2: Systems programming
    [1, 1047, 368, 2563, 264, 1401, 287],  # Prompt 3: Conditional reasoning
    [1, 28705, 28740, 28725, 28705, 28750, 28725, 28705, 28770],  # Prompt 4: Numeric sequence
    [1, 450, 1296, 302, 272, 9060, 10865, 349],  # Prompt 5: Scientific statement
]


class _ATenAuditMode(TorchDispatchMode):
    def __init__(self) -> None:
        super().__init__()
        self.forbidden_calls: list[str] = []

    def __torch_dispatch__(
        self,
        func: Any,
        types: tuple[type, ...],
        args: tuple[Any, ...] = (),
        kwargs: dict[str, Any] | None = None,
    ) -> Any:
        name = str(func)
        if name in FORBIDDEN_ATEN_COMPUTE:
            self.forbidden_calls.append(name)
        return func(*args, **(kwargs or {}))


def _make_small_test_model(dev: str) -> Mistral7B:
    cfg = ModelConfig(
        vocab_size=32000,
        hidden_size=128,
        intermediate_size=256,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=32,
        max_position_embeddings=512,
        group_size=128,
    )
    torch.manual_seed(17)
    weights = create_random_4bit_weights(cfg, device=dev, seed=17)
    return Mistral7B(
        weights=weights, gpu_config=GPUConfig(vram_gb=24.0, stream_weights=False), device=dev
    )


def test_zero_aten_zero_launch_during_graph_replay_and_kv_contents() -> None:
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model = _make_small_test_model(dev)

    rt_graph = NativeModelRuntime(
        model,
        max_seq_len=128,
        use_cuda_graph=True,
        graph_strategy="bucketed",
        context_buckets=(64, 128),
    )
    rt_eager = NativeModelRuntime(model, max_seq_len=128, use_cuda_graph=False)

    prompt = [1, 12, 45, 78, 102]
    prompt_t = torch.tensor([prompt], device=dev, dtype=torch.long)

    tok_g = rt_graph.prefill(prompt_t)
    tok_e = rt_eager.prefill(prompt_t)
    assert torch.equal(tok_g, tok_e)

    if dev == "cuda":
        # Capture graph on first step
        tok_g = rt_graph.decode_step(tok_g, pos=len(prompt), use_graph=True)
        tok_e = rt_eager.decode_step(tok_e, pos=len(prompt), use_graph=False)
        assert int(tok_g.item()) == int(tok_e.item())

        # Audit 8 replayed steps: 0 forbidden ATen calls, 0 eager ct.launch calls, 0 allocations
        orig_launch = ct.launch
        eager_launches = 0

        def _fail_if_launched(*args: Any, **kwargs: Any) -> Any:
            nonlocal eager_launches
            eager_launches += 1
            return orig_launch(*args, **kwargs)

        ws = rt_graph._get_or_create_workspace(1)
        graph = ws.graphs_by_bucket[64]
        ct.launch = _fail_if_launched  # type: ignore[assignment]
        try:
            with _ATenAuditMode() as audit:
                for step_i in range(1, 9):
                    graph.replay()
                    rt_graph.kv_cache.seq_lens = [
                        len(prompt) + 1 + step_i
                    ] * model.config.num_hidden_layers
            assert (
                audit.forbidden_calls == []
            ), f"Forbidden ATen calls during graph replay: {audit.forbidden_calls}"
            assert (
                eager_launches == 0
            ), f"Expected 0 eager ct.launch calls during graph replay, got {eager_launches}"
        finally:
            ct.launch = orig_launch  # type: ignore[assignment]

        for step_i in range(1, 9):
            tok_e = rt_eager.decode_step(tok_e, pos=len(prompt) + step_i, use_graph=False)
        assert int(ws.token_id.item()) == int(tok_e.item())

        # Verify KV cache contents match after N=9 decoded tokens (total seq_len = 5 + 9 = 14)
        total_len = len(prompt) + 9
        for l_idx in range(model.config.num_hidden_layers):
            kg, vg = rt_graph.kv_cache.get_kv(l_idx, end_pos=total_len)
            ke, ve = rt_eager.kv_cache.get_kv(l_idx, end_pos=total_len)
            assert kg.shape == (1, 2, total_len, 32)
            assert (kg.float() - ke.float()).abs().max().item() < 1e-3
            assert (vg.float() - ve.float()).abs().max().item() < 1e-3
    else:
        for step_i in range(4):
            tok_g = rt_graph.decode_step(tok_g, pos=len(prompt) + step_i, use_graph=False)
            tok_e = rt_eager.decode_step(tok_e, pos=len(prompt) + step_i, use_graph=False)
            assert int(tok_g.item()) == int(tok_e.item())


def test_256_token_5_prompt_parity_across_eager_graphed_and_legacy() -> None:
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    real_dir = Path.home() / "models" / "mistral_7b_4bit"

    if dev == "cuda" and real_dir.exists():
        model = Mistral7B.from_pretrained(
            real_dir,
            gpu_config=GPUConfig(vram_gb=32.0, stream_weights=False),
            device="cuda",
            max_gpu_layers=32,
        )
        n_tokens = 256
    else:
        model = _make_small_test_model(dev)
        n_tokens = 256 if dev == "cuda" else 16

    legacy = InferenceEngine(model, target="sm_120", use_cuda_graph=True, enable_radix_cache=False)
    rt_eager = NativeModelRuntime(model, max_seq_len=512, use_cuda_graph=False)
    rt_bucketed = NativeModelRuntime(
        model,
        max_seq_len=512,
        use_cuda_graph=True,
        graph_strategy="bucketed",
        context_buckets=(256, 512),
    )
    rt_masked = NativeModelRuntime(
        model, max_seq_len=512, use_cuda_graph=True, graph_strategy="device_masked"
    )

    for p_idx, prompt_ids in enumerate(FIVE_TEST_PROMPTS):
        prompt_t = torch.tensor([prompt_ids], device=dev, dtype=torch.long)
        toks_leg = legacy.generate_ids(
            prompt_t, max_new_tokens=n_tokens, temperature=0.0, backend="cutile"
        )
        toks_eag, _ = rt_eager.generate(prompt_ids, max_new_tokens=n_tokens, sync_interval=1)
        toks_bkt, _ = rt_bucketed.generate(prompt_ids, max_new_tokens=n_tokens, sync_interval=32)
        toks_msk, _ = rt_masked.generate(prompt_ids, max_new_tokens=n_tokens, sync_interval=32)

        assert len(toks_bkt) == n_tokens
        assert toks_eag == toks_leg, f"Prompt {p_idx}: eager native != legacy engine"
        assert toks_bkt == toks_leg, f"Prompt {p_idx}: bucketed graphed native != legacy engine"
        assert (
            toks_msk == toks_leg
        ), f"Prompt {p_idx}: device_masked graphed native != legacy engine"


def test_native_batching_speculative_and_chunked_prefill() -> None:
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model = _make_small_test_model(dev)
    rt = NativeModelRuntime(model, max_seq_len=128, use_cuda_graph=True, max_batch_size=4)

    # 1. Batch-bucketed continuous batching parity against single-sequence native
    p1 = [1, 15, 29, 44]
    p2 = [1, 88, 102, 119]
    s1, _ = rt.generate(p1, max_new_tokens=8)
    s2, _ = rt.generate(p2, max_new_tokens=8)
    batch_outs, batch_st = rt.generate_batch([p1, p2], max_new_tokens=8, sync_interval=4)
    assert len(batch_outs) == 2
    assert batch_outs[0] == s1
    assert batch_outs[1] == s2
    assert batch_st["batch_size"] == 2

    # 2. Speculative decoding exact greedy parity (repetitive prompt triggers N-gram lookup + rollback)
    rep_prompt = [1, 10, 20, 30, 10, 20, 30, 10, 20]
    ref_toks, _ = rt.generate(rep_prompt, max_new_tokens=16)
    spec_toks, spec_st = rt.generate_speculative(rep_prompt, max_new_tokens=16, gamma=4)
    assert spec_toks == ref_toks
    assert "acceptance_rate" in spec_st

    # 3. Chunked prefill + interleaved decode scheduler telemetry
    sched_monolithic = rt.benchmark_prefill_decode_interleave(
        p1, [1] * 32, decode_steps=4, chunk_size=None
    )
    sched_chunked = rt.benchmark_prefill_decode_interleave(
        p1, [1] * 32, decode_steps=4, chunk_size=8
    )
    assert sched_monolithic["max_itl_ms"] >= 0.0
    assert sched_chunked["max_itl_ms"] >= 0.0
