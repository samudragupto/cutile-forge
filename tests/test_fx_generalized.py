"""Tests for generalized PyTorch FX/Dynamo lowering backend (Phase 8E)."""

from __future__ import annotations

import torch
import torch.nn.functional as F

import cutile_forge  # noqa: F401
import cutile_forge.forge.fx_backend as fx_mod
from cutile_forge.forge.fx_backend import DEFAULT_FX_OP_REGISTRY, ShapeGuard


def test_fx_op_registry_and_shape_guard() -> None:
    ops = DEFAULT_FX_OP_REGISTRY.registered_ops()
    for expected in (
        "addmm",
        "amax",
        "conv2d",
        "cumsum",
        "embedding",
        "gelu",
        "layer_norm",
        "linear",
        "matmul",
        "silu",
        "sum",
    ):
        assert expected in ops

    a = torch.zeros((4, 16), dtype=torch.float32)
    b = torch.ones((4, 16), dtype=torch.float32)
    c = torch.zeros((8, 16), dtype=torch.float32)
    guard = ShapeGuard.from_inputs([a])
    assert guard.matches([b]) is True
    assert guard.matches([c]) is False


def test_fx_generalized_ops_parity_and_stats() -> None:
    fx_mod.clear_fx_cache()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if torch.cuda.is_available() else torch.float32

    w_lin = torch.randn((64, 32), device=dev, dtype=dtype) * 0.05
    b_lin = torch.randn((64,), device=dev, dtype=dtype) * 0.01
    ln_w = torch.ones((64,), device=dev, dtype=dtype)
    ln_b = torch.zeros((64,), device=dev, dtype=dtype)
    bias_addmm = torch.randn((64,), device=dev, dtype=dtype) * 0.01
    m2 = torch.randn((32, 64), device=dev, dtype=dtype) * 0.05

    x = torch.randn((8, 32), device=dev, dtype=dtype) * 0.1

    def fn(inp: torch.Tensor) -> torch.Tensor:
        h1 = F.linear(inp, w_lin, b_lin)
        h2 = torch.addmm(bias_addmm, inp, m2)
        h3 = F.layer_norm(h1, (64,), ln_w, ln_b, 1e-5)
        h4 = F.gelu(h3, approximate="tanh")
        h5 = F.silu(h4)
        s = torch.sum(h5, dim=-1) + torch.sum(h2, dim=-1) * 0.1 + torch.sum(h3, dim=-1) * 0.05
        mx = torch.amax(h5, dim=-1)
        cs = torch.cumsum(h5, dim=-1)[:, -1]
        return torch.cos(s + mx + cs)  # cos stays an intentional fallback op

    torch._dynamo.reset()
    compiled = torch.compile(fn, backend="cutile_forge")
    got = compiled(x)
    ref = fn(x)
    assert (got - ref).abs().max().item() < 2e-2

    stats = fx_mod.LAST_FX_STATS
    assert stats is not None
    assert "linear" in stats.lowered_ops
    assert "addmm" in stats.lowered_ops
    assert "layer_norm" in stats.lowered_ops
    assert "gelu" in stats.lowered_ops
    assert "silu" in stats.lowered_ops
    assert "sum" in stats.lowered_ops
    assert "amax" in stats.lowered_ops
    assert "cumsum" in stats.lowered_ops
    assert "cos" in stats.fallback_ops


def test_fx_generalized_matmul_conv_embedding_and_dynamic_shape() -> None:
    fx_mod.clear_fx_cache()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if torch.cuda.is_available() else torch.float32

    emb_w = torch.randn((32, 16), device=dev, dtype=dtype) * 0.1
    conv_w = torch.randn((8, 4, 3, 3), device=dev, dtype=dtype) * 0.05
    proj = torch.randn((16, 8), device=dev, dtype=dtype) * 0.1

    def fn(idx: torch.Tensor, img: torch.Tensor) -> torch.Tensor:
        e = F.embedding(idx, emb_w)  # (B, 8, 16)
        c = F.conv2d(img, conv_w, padding=1)  # (B, 8, 8, 8)
        pooled = torch.amax(c.flatten(2), dim=-1)  # (B, 8)
        m = torch.matmul(e, proj)  # (B, 8, 8)
        return m.mean(dim=-1) + pooled

    idx1 = torch.randint(0, 32, (2, 8), device=dev)
    img1 = torch.randn((2, 4, 8, 8), device=dev, dtype=dtype) * 0.2

    torch._dynamo.reset()
    compiled = torch.compile(fn, backend="cutile_forge")
    out1 = compiled(idx1, img1)
    ref1 = fn(idx1, img1)
    assert (out1 - ref1).abs().max().item() < 2e-2

    stats = fx_mod.LAST_FX_STATS
    assert stats is not None
    assert "embedding" in stats.lowered_ops
    assert "conv2d" in stats.lowered_ops
    assert "matmul" in stats.lowered_ops

    # Dynamic batch shape triggers recompile / cache miss cleanly
    idx2 = torch.randint(0, 32, (4, 8), device=dev)
    img2 = torch.randn((4, 4, 8, 8), device=dev, dtype=dtype) * 0.2
    out2 = compiled(idx2, img2)
    ref2 = fn(idx2, img2)
    assert out2.shape == ref2.shape
    assert (out2 - ref2).abs().max().item() < 2e-2
