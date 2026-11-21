"""Tests for Phase 7: gather/conv2d kernels, ONNX graph importer, and server hardening."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from fastapi.testclient import TestClient

from cutile_forge.forge.fusion_graph import OpType
from cutile_forge.kernels import cutile_conv2d, cutile_gather
from cutile_forge.onnx_import import import_graph_spec
from cutile_forge.sdk import compile
from cutile_forge.server import create_app


class _DummyConfig:
    vocab_size = 128


class _DummyEngine:
    target = "sm_120"
    config = _DummyConfig()

    def generate_ids(self, prompt_ids: object, max_new_tokens: int = 8, **_: object) -> list[int]:
        _ = prompt_ids
        return [1, 2, 3][:max_new_tokens]

    def stream_ids(self, prompt_ids: object, max_new_tokens: int = 4, **_: object) -> list[int]:
        _ = prompt_ids
        return [4, 5, 6][:max_new_tokens]


def test_cutile_gather_and_conv2d() -> None:
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    for dt in (torch.float16, torch.bfloat16, torch.float32):
        tol = 1e-4 if dt == torch.float32 else 2e-2
        table = torch.randn((100, 77), device=dev, dtype=dt)
        idx = torch.randint(0, 100, (3, 19), device=dev, dtype=torch.long)
        got_g = cutile_gather(table, idx)
        ref_g = F.embedding(idx, table)
        assert torch.equal(got_g, ref_g)

        x = torch.randn((2, 3, 12, 12), device=dev, dtype=dt) * 0.1
        w = torch.randn((8, 3, 3, 3), device=dev, dtype=dt) * 0.1
        b = torch.randn((8,), device=dev, dtype=dt) * 0.1
        got_c = cutile_conv2d(x, w, bias=b, stride=2, padding=1)
        ref_c = F.conv2d(x.float(), w.float(), bias=b.float(), stride=2, padding=1).to(dt)
        assert (got_c - ref_c).abs().max().item() < tol


def test_onnx_graph_spec_importer_lowers_and_fuses() -> None:
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    spec = {
        "name": "mlp_onnx",
        "inputs": [
            {"name": "x", "shape": [16, 64], "dtype": "float16"},
            {"name": "w1", "shape": [64, 128], "dtype": "float16"},
            {"name": "w2", "shape": [128, 64], "dtype": "float16"},
        ],
        "nodes": [
            {"op": "MatMul", "inputs": ["x", "w1"], "output": "h"},
            {"op": "Silu", "inputs": ["h"], "output": "a"},
            {"op": "MatMul", "inputs": ["a", "w2"], "output": "y"},
        ],
        "outputs": ["y"],
    }
    sdk_graph = import_graph_spec(spec)
    compiled = compile(sdk_graph, target="auto", use_cache=False)
    non_in = [k.op_type for k in compiled.kernels if k.op_type != OpType.INPUT]
    assert non_in == [OpType.FUSED_FFN]

    tx = torch.randn((16, 64), device=dev, dtype=torch.float16) * 0.1
    tw1 = torch.randn((64, 128), device=dev, dtype=torch.float16) * 0.1
    tw2 = torch.randn((128, 64), device=dev, dtype=torch.float16) * 0.1
    got = compiled.run(x=tx, w1=tw1, w2=tw2)
    assert isinstance(got, torch.Tensor)
    ref = torch.matmul(F.silu(torch.matmul(tx, tw1)), tw2)
    assert (got - ref).abs().max().item() < 1e-2


def test_server_auth_metrics_and_backpressure() -> None:
    app = create_app(_DummyEngine(), api_key="secret-token", max_concurrent_requests=10)
    client = TestClient(app)

    # /health stays open
    assert client.get("/health").status_code == 200

    # Unauthorized request blocked
    payload = {
        "model": "mistral-7b-4bit",
        "messages": [{"role": "user", "content": "hi"}],
        "max_tokens": 2,
    }
    assert client.post("/v1/chat/completions", json=payload).status_code == 401
    assert client.get("/v1/models").status_code == 401

    # Authorized request succeeds & increments /metrics
    headers = {"Authorization": "Bearer secret-token"}
    resp = client.post("/v1/chat/completions", json=payload, headers=headers)
    assert resp.status_code == 200
    m = client.get("/metrics").json()
    assert m["requests_total"] == 1
    assert m["requests_inflight"] == 0
    assert m["tokens_generated_total"] == 2
    assert m["errors_total"] == 2  # Two 401 responses

    # Max concurrent backpressure check (max_concurrent_requests=0 -> 429)
    app_full = create_app(_DummyEngine(), max_concurrent_requests=0)
    c_full = TestClient(app_full)
    assert c_full.post("/v1/chat/completions", json=payload).status_code == 429
