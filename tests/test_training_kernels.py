"""Tests for CuTile activation, backward, autograd, and fused AdamW optimizer kernels."""

from __future__ import annotations

import torch
import torch.nn.functional as F

from cutile_forge.kernels import (
    cutile_adamw_step,
    cutile_gelu,
    cutile_rmsnorm_autograd,
    cutile_rmsnorm_backward,
    cutile_silu,
    cutile_silu_autograd,
    cutile_silu_backward,
)


def test_activations_forward_and_backward() -> None:
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    for dt in (torch.float16, torch.bfloat16, torch.float32):
        tol = 1e-4 if dt == torch.float32 else 2e-2
        x = torch.randn((3, 1025), device=dev, dtype=dt)
        grad = torch.randn_like(x)

        got_silu = cutile_silu(x)
        ref_silu = F.silu(x.float()).to(dt)
        assert (got_silu - ref_silu).abs().max().item() < tol

        got_gelu = cutile_gelu(x)
        ref_gelu = F.gelu(x.float(), approximate="tanh").to(dt)
        assert (got_gelu - ref_gelu).abs().max().item() < tol

        x_ref = x.detach().float().requires_grad_(True)
        F.silu(x_ref).backward(grad.float())
        got_dx = cutile_silu_backward(grad, x)
        assert x_ref.grad is not None
        assert (got_dx - x_ref.grad.to(dt)).abs().max().item() < tol


def test_rmsnorm_backward_and_autograd_functions() -> None:
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    tol = 2e-2 if dtype == torch.float16 else 1e-4

    x = (torch.randn((8, 128), device=dev, dtype=dtype) * 0.5).requires_grad_(True)
    w = torch.ones((128,), device=dev, dtype=dtype, requires_grad=True)

    y = cutile_silu_autograd(cutile_rmsnorm_autograd(x, w, eps=1e-5))
    loss = y.pow(2).sum()
    loss.backward()

    # Exact PyTorch autograd reference
    xr = x.detach().float().requires_grad_(True)
    wr = w.detach().float().requires_grad_(True)
    inv_rms = torch.rsqrt(xr.pow(2).mean(dim=-1, keepdim=True) + 1e-5)
    yr = F.silu(xr * inv_rms * wr)
    yr.pow(2).sum().backward()

    assert x.grad is not None and xr.grad is not None
    assert w.grad is not None and wr.grad is not None
    assert (x.grad - xr.grad.to(dtype)).abs().max().item() < tol
    assert (w.grad - wr.grad.to(dtype)).abs().max().item() < tol

    # Standalone backward helper check
    g_out = torch.randn((8, 128), device=dev, dtype=dtype)
    dx_direct, dw_direct = cutile_rmsnorm_backward(g_out, x.detach(), w.detach(), eps=1e-5)
    assert dx_direct.shape == x.shape and dw_direct.shape == w.shape


def test_fused_adamw_matches_pytorch_optimizer() -> None:
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(99)
    shape = (2050,)  # Non-tile-aligned count across 3 tiles (tile_n=1024)

    p_cu = torch.randn(shape, device=dev, dtype=torch.float32)
    p_ref = torch.nn.Parameter(p_cu.clone())
    m_cu = torch.zeros_like(p_cu)
    v_cu = torch.zeros_like(p_cu)
    opt = torch.optim.AdamW([p_ref], lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.01)

    for step in range(1, 6):
        g = torch.randn(shape, device=dev, dtype=torch.float32)
        opt.zero_grad()
        p_ref.grad = g.clone()
        opt.step()

        cutile_adamw_step(
            p_cu,
            g,
            m_cu,
            v_cu,
            step=step,
            lr=1e-3,
            beta1=0.9,
            beta2=0.999,
            eps=1e-8,
            weight_decay=0.01,
        )
        assert (p_cu - p_ref.detach()).abs().max().item() < 1e-5
