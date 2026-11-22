"""Phase 16 failing reproduction tests for mixed-length isolation, SSE batching & disconnects."""

from __future__ import annotations

import asyncio
import json

import httpx
import torch

from cutile_forge.cli import _NativeServingAdapter
from cutile_forge.engine.config import ModelConfig
from cutile_forge.engine.model import Mistral7B
from cutile_forge.engine.weight_loader import create_random_4bit_weights
from cutile_forge.runtime.native_engine import NativeModelRuntime
from cutile_forge.server import create_app


def _build_small_native_runtime(max_batch_size: int = 8) -> NativeModelRuntime:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    cfg = ModelConfig(
        vocab_size=256,
        hidden_size=128,
        intermediate_size=256,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=32,
        max_position_embeddings=256,
        group_size=64,
    )
    weights = create_random_4bit_weights(cfg, device=device, seed=19)
    model = Mistral7B(weights=weights, device=device)
    return NativeModelRuntime(
        model,
        max_seq_len=256,
        kv_dtype="fp16",
        use_cuda_graph=(device == "cuda"),
        max_batch_size=max_batch_size,
    )


def test_mixed_length_batched_greedy_isolation() -> None:
    """Concurrent mixed-length prompts in generate_batch must match standalone generate."""
    rt = _build_small_native_runtime(max_batch_size=4)
    prompts = [
        [1, 5],
        [1, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29, 31],
        [1, 12, 34, 56],
        [1, 99, 88, 77, 66, 55, 44],
    ]
    solo: list[list[int]] = []
    for p in prompts:
        toks, _ = rt.generate(p, max_new_tokens=12)
        solo.append(toks)

    batched, _ = rt.generate_batch(prompts, max_new_tokens=12)
    for idx, (s, b) in enumerate(zip(solo, batched, strict=True)):
        assert b == s, f"Prompt {idx} contaminated in mixed-length batch: {b} != {s}"


def test_http_streaming_coalesces_concurrent_requests() -> None:
    """Concurrent stream=True HTTP requests must coalesce into multi-request GPU batches."""
    rt = _build_small_native_runtime(max_batch_size=8)
    adapter = _NativeServingAdapter(rt)
    app = create_app(adapter, max_batch_size=8, batch_wait_ms=10.0)

    async def _run() -> dict[str, object]:
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:

            async def _one(prompt_txt: str) -> str:
                chunks: list[str] = []
                async with client.stream(
                    "POST",
                    "/v1/chat/completions",
                    json={
                        "model": "tiny",
                        "messages": [{"role": "user", "content": prompt_txt}],
                        "max_tokens": 8,
                        "temperature": 0.0,
                        "stream": True,
                    },
                ) as resp:
                    assert resp.status_code == 200
                    async for line in resp.aiter_lines():
                        if line.startswith("data: ") and line[6:].strip() != "[DONE]":
                            obj = json.loads(line[6:].strip())
                            chunks.append(obj["choices"][0]["delta"].get("content", ""))
                return "".join(chunks)

            outs = await asyncio.gather(
                _one("alpha"),
                _one("beta gamma"),
                _one("delta epsilon zeta"),
                _one("eta"),
            )
            assert all(len(o) > 0 for o in outs)
            met = (await client.get("/metrics")).json()
            return met

    metrics = asyncio.run(_run())
    assert (
        int(metrics.get("batch_size_max", 0)) >= 2
    ), f"Streaming requests executed unbatched: metrics={metrics}"


def test_http_stream_disconnect_releases_kv_slots_and_counts_disconnect() -> None:
    """Abandoning an SSE stream mid-generation must increment disconnect count and free KV slots."""
    rt = _build_small_native_runtime(max_batch_size=8)
    adapter = _NativeServingAdapter(rt)
    app = create_app(adapter, max_batch_size=8, batch_wait_ms=4.0)

    async def _run() -> dict[str, object]:
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            async with client.stream(
                "POST",
                "/v1/chat/completions",
                json={
                    "model": "tiny",
                    "messages": [{"role": "user", "content": "long stream prompt"}],
                    "max_tokens": 32,
                    "temperature": 0.0,
                    "stream": True,
                },
            ) as resp:
                assert resp.status_code == 200
                seen = 0
                async for line in resp.aiter_lines():
                    if line.startswith("data: "):
                        seen += 1
                        if seen >= 2:
                            break
            await asyncio.sleep(0.2)
            return (await client.get("/metrics")).json()

    metrics = asyncio.run(_run())
    assert (
        int(metrics.get("client_disconnects_total", 0)) >= 1
    ), f"Disconnect not recorded: {metrics}"
    assert int(metrics.get("requests_inflight", -1)) == 0
    assert int(metrics.get("active_kv_slots", -1)) == 0
