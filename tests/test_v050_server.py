"""Unit tests for CuTileForge v0.5.0 OpenAI-compatible server and Roofline plotter."""

from __future__ import annotations

from pathlib import Path

import torch
from fastapi.testclient import TestClient

from cutile_forge.bench.ncu_roofline import generate_roofline_plot
from cutile_forge.engine.config import ModelConfig
from cutile_forge.engine.inference import InferenceEngine
from cutile_forge.engine.model import Mistral7B
from cutile_forge.server.app import create_app


def _make_test_client() -> TestClient:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    cfg = ModelConfig(
        vocab_size=256,
        hidden_size=128,
        intermediate_size=256,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=32,
    )
    model = Mistral7B.from_random(cfg, device=device, seed=42)
    engine = InferenceEngine(model, use_cuda_graph=False, target="sm_120")
    app = create_app(engine, tokenizer=None, model_id="mistral-7b-4bit-test")
    return TestClient(app)


def test_server_health_and_models() -> None:
    client = _make_test_client()
    r_health = client.get("/health")
    assert r_health.status_code == 200
    body = r_health.json()
    assert body["status"] == "ok"
    assert body["target"] == "sm_120"

    r_models = client.get("/v1/models")
    assert r_models.status_code == 200
    assert r_models.json()["data"][0]["id"] == "mistral-7b-4bit-test"


def test_server_chat_completions_sync_and_stream() -> None:
    client = _make_test_client()
    r_sync = client.post(
        "/v1/chat/completions",
        json={
            "model": "mistral-7b-4bit-test",
            "messages": [{"role": "user", "content": "Hello"}],
            "max_tokens": 4,
            "stream": False,
        },
    )
    assert r_sync.status_code == 200
    data = r_sync.json()
    assert data["object"] == "chat.completion"
    assert len(data["choices"]) == 1

    r_stream = client.post(
        "/v1/chat/completions",
        json={
            "model": "mistral-7b-4bit-test",
            "messages": [{"role": "user", "content": "Hello"}],
            "max_tokens": 4,
            "stream": True,
        },
    )
    assert r_stream.status_code == 200
    text = r_stream.text
    assert "data: [DONE]" in text
    assert "chat.completion.chunk" in text


def test_roofline_plot_generation(tmp_path: Path) -> None:
    out = tmp_path / "roofline.png"
    p = generate_roofline_plot(output_path=out)
    assert p.exists()
    assert p.stat().st_size > 10_000
