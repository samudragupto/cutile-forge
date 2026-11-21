"""Tests for expanded PyTorch FX/Dynamo backend (Phase 5)."""

from __future__ import annotations

import torch
import torch.nn.functional as F

import cutile_forge  # noqa: F401
import cutile_forge.forge.fx_backend as fx_mod


def test_fx_standalone_linear_silu_bmm_softmax_and_fallback() -> None:
    fx_mod.clear_fx_cache()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if torch.cuda.is_available() else torch.float32

    w = torch.randn((128, 64), device=dev, dtype=dtype) * 0.05
    x = torch.randn((4, 16, 64), device=dev, dtype=dtype) * 0.1
    b_mat = torch.randn((4, 128, 32), device=dev, dtype=dtype) * 0.1

    def fn(inp: torch.Tensor, rhs: torch.Tensor) -> torch.Tensor:
        h = F.silu(F.linear(inp, w))
        m = torch.bmm(h, rhs)
        p = F.softmax(m, dim=-1)
        return torch.sin(p)  # Intentional unsupported op for fallback reporting

    torch._dynamo.reset()
    compiled = torch.compile(fn, backend="cutile_forge")
    got = compiled(x, b_mat)
    ref = fn(x, b_mat)
    assert (got - ref).abs().max().item() < 1e-2

    stats = fx_mod.LAST_FX_STATS
    assert stats is not None
    assert stats.gemm_silu_fusions == 1
    assert stats.bmm_lowerings == 1
    assert stats.softmax_lowerings == 1
    assert stats.cache_hit is False
    assert "sin" in stats.fallback_ops


def test_fx_compile_cache_hit() -> None:
    fx_mod.clear_fx_cache()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if torch.cuda.is_available() else torch.float32

    w = torch.randn((64, 64), device=dev, dtype=dtype) * 0.05
    x = torch.randn((8, 64), device=dev, dtype=dtype) * 0.1

    def fn(inp: torch.Tensor) -> torch.Tensor:
        return F.silu(F.linear(inp, w))

    torch._dynamo.reset()
    c1 = torch.compile(fn, backend="cutile_forge")
    out1 = c1(x)
    assert fx_mod.LAST_FX_STATS is not None and fx_mod.LAST_FX_STATS.cache_hit is False

    torch._dynamo.reset()
    c2 = torch.compile(fn, backend="cutile_forge")
    out2 = c2(x)
    assert fx_mod.LAST_FX_STATS is not None and fx_mod.LAST_FX_STATS.cache_hit is True
    assert torch.equal(out1, out2)
