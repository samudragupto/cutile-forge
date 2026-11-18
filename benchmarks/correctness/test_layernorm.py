"""Correctness tests: CuTile LayerNorm, RMSNorm, and fused residual norms vs PyTorch."""

from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

from cutile_forge.kernels.layernorm import cutile_layernorm
from cutile_forge.kernels.layernorm_residual import (
    fused_layernorm_residual,
    fused_rmsnorm_residual,
)
from cutile_forge.kernels.rmsnorm import cutile_rmsnorm


@pytest.mark.gpu
def test_layernorm() -> None:
    if not torch.cuda.is_available():
        pytest.skip("CUDA GPU required")
    torch.manual_seed(2)
    x = torch.randn((16, 4096), device="cuda", dtype=torch.float16) * 0.2
    w = torch.randn(4096, device="cuda", dtype=torch.float16) * 0.2
    b = torch.randn(4096, device="cuda", dtype=torch.float16) * 0.2
    out = cutile_layernorm(x, w, b, eps=1e-5)
    ref = F.layer_norm(x.float(), (4096,), w.float(), b.float(), 1e-5).half()
    assert (out - ref).abs().max().item() < 1e-3


@pytest.mark.gpu
def test_rmsnorm() -> None:
    if not torch.cuda.is_available():
        pytest.skip("CUDA GPU required")
    torch.manual_seed(3)
    x = torch.randn((16, 4096), device="cuda", dtype=torch.float16) * 0.2
    w = torch.randn(4096, device="cuda", dtype=torch.float16) * 0.2
    out = cutile_rmsnorm(x, w, eps=1e-5)
    ref = (
        x.float() * torch.rsqrt(x.float().pow(2).mean(dim=-1, keepdim=True) + 1e-5) * w.float()
    ).half()
    assert (out - ref).abs().max().item() < 1e-3


@pytest.mark.gpu
def test_layernorm_and_rmsnorm_residual() -> None:
    if not torch.cuda.is_available():
        pytest.skip("CUDA GPU required")
    torch.manual_seed(4)
    x = torch.randn((16, 4096), device="cuda", dtype=torch.float16) * 0.1
    res = torch.randn((16, 4096), device="cuda", dtype=torch.float16) * 0.1
    w = torch.randn(4096, device="cuda", dtype=torch.float16) * 0.2
    b = torch.randn(4096, device="cuda", dtype=torch.float16) * 0.2

    comb = (x.float() + res.float()).half()
    n_ln, r_ln = fused_layernorm_residual(x, res, w, b, 1e-5)
    assert (r_ln - comb).abs().max().item() < 1e-3
    assert (
        n_ln - F.layer_norm(comb.float(), (4096,), w.float(), b.float(), 1e-5).half()
    ).abs().max().item() < 1e-3

    n_rms, r_rms = fused_rmsnorm_residual(x, res, w, 1e-5)
    ref_rms = (
        comb.float()
        * torch.rsqrt(comb.float().pow(2).mean(dim=-1, keepdim=True) + 1e-5)
        * w.float()
    ).half()
    assert (r_rms - comb).abs().max().item() < 1e-3
    assert (n_rms - ref_rms).abs().max().item() < 1e-3
