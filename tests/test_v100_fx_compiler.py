"""Unit and GPU tests for CuTileForge `torch.compile(backend="cutile_forge")`."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

import cutile_forge
import cutile_forge.forge.fx_backend as fx_mod


class _MiniTransformerBlock(nn.Module):
    def __init__(self, d_model: int = 128, ffn_dim: int = 256, n_heads: int = 2) -> None:
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.ln_w = nn.Parameter(torch.ones(d_model))
        self.ln_b = nn.Parameter(torch.zeros(d_model))
        self.w_up = nn.Parameter(torch.randn(ffn_dim, d_model) * 0.05)
        self.w_down = nn.Parameter(torch.randn(d_model, ffn_dim) * 0.05)

    def forward(self, x: torch.Tensor, residual: torch.Tensor) -> torch.Tensor:
        h = F.layer_norm(x + residual, (self.d_model,), self.ln_w, self.ln_b, 1e-5)
        bsz, seq, _ = h.shape
        qkv = h.view(bsz, seq, self.n_heads, self.head_dim).transpose(1, 2)
        attn = F.scaled_dot_product_attention(qkv, qkv, qkv, is_causal=True)
        attn_out = attn.transpose(1, 2).contiguous().view(bsz, seq, self.d_model)
        ffn = F.linear(F.silu(F.linear(attn_out, self.w_up)), self.w_down)
        return attn_out + ffn


def test_cutile_forge_backend_registered() -> None:
    import torch._dynamo as dynamo

    assert cutile_forge.__version__ == "3.1.0"
    assert "cutile_forge" in dynamo.list_backends()


def test_fx_backend_fuses_ffn_ln_and_sdpa() -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    block = _MiniTransformerBlock(d_model=128, ffn_dim=256, n_heads=2).to(
        device=device, dtype=dtype
    )
    x = torch.randn((1, 16, 128), device=device, dtype=dtype) * 0.1
    res = torch.randn((1, 16, 128), device=device, dtype=dtype) * 0.1

    with torch.no_grad():
        ref = block(x, res)
        torch._dynamo.reset()
        compiled = torch.compile(block, backend="cutile_forge")
        out = compiled(x, res)

    assert (out - ref).abs().max().item() < 1e-2
    assert fx_mod.LAST_FX_STATS is not None
    assert fx_mod.LAST_FX_STATS.ffn_fusions == 1
    assert fx_mod.LAST_FX_STATS.ln_res_fusions == 1
    assert fx_mod.LAST_FX_STATS.attn_fusions == 1
    assert fx_mod.LAST_FX_STATS.fused_nodes < fx_mod.LAST_FX_STATS.original_nodes


def test_fx_backend_ffn_only_pattern() -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    w_up = torch.randn((128, 64), device=device, dtype=dtype) * 0.05
    w_down = torch.randn((64, 128), device=device, dtype=dtype) * 0.05
    x = torch.randn((16, 64), device=device, dtype=dtype) * 0.1

    def fn(inp: torch.Tensor) -> torch.Tensor:
        return F.linear(F.silu(F.linear(inp, w_up)), w_down)

    torch._dynamo.reset()
    compiled = torch.compile(fn, backend="cutile_forge")
    assert (compiled(x) - fn(x)).abs().max().item() < 1e-2
    assert fx_mod.LAST_FX_STATS is not None
    assert fx_mod.LAST_FX_STATS.tfg_nodes_before == 3
    assert fx_mod.LAST_FX_STATS.tfg_nodes_after == 1


def test_fx_backend_ln_residual_pattern() -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    w = torch.ones(128, device=device, dtype=dtype)
    b = torch.zeros(128, device=device, dtype=dtype)
    x = torch.randn((8, 128), device=device, dtype=dtype)
    r = torch.randn((8, 128), device=device, dtype=dtype)

    def fn(a: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        return F.layer_norm(a + c, (128,), w, b, 1e-5)

    torch._dynamo.reset()
    compiled = torch.compile(fn, backend="cutile_forge")
    assert (compiled(x, r) - fn(x, r)).abs().max().item() < 1e-2
    assert fx_mod.LAST_FX_STATS is not None
    assert fx_mod.LAST_FX_STATS.ln_res_fusions == 1


def test_fx_backend_sdpa_pattern() -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    q = torch.randn((1, 4, 32, 64), device=device, dtype=dtype) * 0.2
    k = torch.randn((1, 2, 32, 64), device=device, dtype=dtype) * 0.2
    v = torch.randn((1, 2, 32, 64), device=device, dtype=dtype) * 0.2

    def fn(qi: torch.Tensor, ki: torch.Tensor, vi: torch.Tensor) -> torch.Tensor:
        k_rep = ki.repeat_interleave(2, dim=1)
        v_rep = vi.repeat_interleave(2, dim=1)
        return F.scaled_dot_product_attention(qi, k_rep, v_rep, is_causal=True)

    torch._dynamo.reset()
    compiled = torch.compile(fn, backend="cutile_forge")
    assert (compiled(q, k, v) - fn(q, k, v)).abs().max().item() < 1e-2
    assert fx_mod.LAST_FX_STATS is not None
    assert fx_mod.LAST_FX_STATS.attn_fusions == 1


def test_fx_backend_preserves_unmatched_ops() -> None:
    def fn(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        return torch.relu(a * 2.0 + b)

    x = torch.randn(4, 16)
    y = torch.randn(4, 16)
    torch._dynamo.reset()
    compiled = torch.compile(fn, backend="cutile_forge")
    assert torch.allclose(compiled(x, y), fn(x, y))
