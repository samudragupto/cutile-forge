"""CPU-safe unit tests for Phase 14 server hardening & controlled fault recovery."""

from __future__ import annotations

import torch
from fastapi.testclient import TestClient

from cutile_forge.engine.config import ModelConfig
from cutile_forge.engine.inference import InferenceEngine
from cutile_forge.engine.model import Mistral7B
from cutile_forge.server.app import create_app


def _make_tiny_engine(use_cuda_graph: bool = False) -> InferenceEngine:
    cfg = ModelConfig(
        vocab_size=256,
        hidden_size=128,
        intermediate_size=256,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=32,
    )
    model = Mistral7B.from_random(cfg, device="cpu", seed=77)
    return InferenceEngine(model, use_cuda_graph=use_cuda_graph, target="sm_120")


def test_auth_invalid_fields_and_oversized_prompt_isolation() -> None:
    eng = _make_tiny_engine()
    app = create_app(
        eng,
        model_id="test-mistral",
        api_key="secret-phase14",
        max_concurrent_requests=2,
        max_prompt_tokens=32,
    )
    client = TestClient(app)

    # 1. Missing / wrong API key -> 401
    r_no_auth = client.post(
        "/v1/chat/completions",
        json={"model": "test-mistral", "messages": [{"role": "user", "content": "hi"}]},
    )
    assert r_no_auth.status_code == 401

    r_bad_auth = client.get("/v1/models", headers={"Authorization": "Bearer wrong"})
    assert r_bad_auth.status_code == 401

    # 2. Unsupported API field -> 422
    headers = {"Authorization": "Bearer secret-phase14"}
    r_extra = client.post(
        "/v1/chat/completions",
        headers=headers,
        json={
            "model": "test-mistral",
            "messages": [{"role": "user", "content": "hi"}],
            "unsupported_knob": 123,
        },
    )
    assert r_extra.status_code == 422

    # 3. Excessive max_tokens -> 422
    r_max_tok = client.post(
        "/v1/chat/completions",
        headers=headers,
        json={
            "model": "test-mistral",
            "messages": [{"role": "user", "content": "hi"}],
            "max_tokens": 99999,
        },
    )
    assert r_max_tok.status_code == 422

    # 4. Oversized prompt -> 400
    r_long = client.post(
        "/v1/chat/completions",
        headers=headers,
        json={
            "model": "test-mistral",
            "messages": [{"role": "user", "content": "x" * 200}],
            "max_tokens": 4,
        },
    )
    assert r_long.status_code == 400

    # 5. Valid authorized request still succeeds cleanly after prior client faults
    r_ok = client.post(
        "/v1/chat/completions",
        headers=headers,
        json={
            "model": "test-mistral",
            "messages": [{"role": "user", "content": "ok"}],
            "max_tokens": 4,
        },
    )
    assert r_ok.status_code == 200
    assert r_ok.json()["usage"]["completion_tokens"] == 4

    r_metrics = client.get("/metrics").json()
    assert r_metrics["requests_inflight"] == 0
    assert r_metrics["errors_total"] >= 2
    assert r_metrics["requests_succeeded"] == 1


def test_inference_engine_graph_capture_failure_falls_back_to_eager(monkeypatch: object) -> None:
    eng = _make_tiny_engine(use_cuda_graph=True)
    # Simulate CUDA Graph eligibility returning True, while capture raises RuntimeError
    monkeypatch.setattr(eng, "_can_use_cuda_graph", lambda _b: True)  # type: ignore[attr-defined]

    def _boom() -> None:
        raise RuntimeError("Simulated CUDA Graph capture failure")

    monkeypatch.setattr(eng, "_capture_decode_graph", _boom)  # type: ignore[attr-defined]

    prompt = torch.tensor([[1, 2, 3]], dtype=torch.long)
    _ = eng.prefill(prompt)
    logits = eng.decode_step(torch.tensor([[4]], dtype=torch.long), pos=3)
    assert logits.shape[-1] == 256
    assert eng.graph_fallback_count == 1
    assert eng.use_cuda_graph is False
