"""End-to-end training & backward kernel parity tests (Phase 8F)."""

from __future__ import annotations

import torch
import torch.nn.functional as F

from cutile_forge.kernels import (
    cutile_bmm_backward,
    cutile_gelu_autograd,
    cutile_gelu_backward,
    cutile_grad_clip_norm,
    cutile_layernorm_autograd,
    cutile_layernorm_backward,
    cutile_linear_autograd,
    cutile_linear_backward,
    cutile_multi_tensor_adamw,
    cutile_softmax_backward,
)


def test_activation_and_norm_backwards_match_pytorch() -> None:
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(0)

    # GELU backward
    x = torch.randn((16, 64), device=dev, dtype=torch.float32, requires_grad=True)
    dy = torch.randn_like(x)
    ref_y = F.gelu(x, approximate="tanh")
    ref_y.backward(dy)
    got_dx = cutile_gelu_backward(dy, x.detach())
    assert (got_dx - x.grad).abs().max().item() < 1e-4

    # Autograd GELU
    x2 = x.detach().clone().requires_grad_(True)
    cutile_gelu_autograd(x2).backward(dy)
    assert x2.grad is not None
    assert (x2.grad - x.grad).abs().max().item() < 1e-4

    # Softmax backward
    logits = torch.randn((8, 128), device=dev, dtype=torch.float32, requires_grad=True)
    dy_s = torch.randn_like(logits)
    probs = F.softmax(logits, dim=-1)
    probs.backward(dy_s)
    got_dx_s = cutile_softmax_backward(dy_s, probs.detach(), dim=-1)
    assert (got_dx_s - logits.grad).abs().max().item() < 1e-4

    # LayerNorm backward + autograd
    xl = torch.randn((8, 64), device=dev, dtype=torch.float32, requires_grad=True)
    w = torch.randn((64,), device=dev, dtype=torch.float32, requires_grad=True)
    b = torch.randn((64,), device=dev, dtype=torch.float32, requires_grad=True)
    dy_l = torch.randn_like(xl)
    ref_l = F.layer_norm(xl, (64,), w, b, 1e-5)
    ref_l.backward(dy_l)

    dx_c, dw_c, db_c = cutile_layernorm_backward(dy_l, xl.detach(), w.detach(), eps=1e-5)
    assert (dx_c - xl.grad).abs().max().item() < 1e-4
    assert (dw_c - w.grad).abs().max().item() < 1e-4
    assert (db_c - b.grad).abs().max().item() < 1e-4

    xl2 = xl.detach().clone().requires_grad_(True)
    w2 = w.detach().clone().requires_grad_(True)
    b2 = b.detach().clone().requires_grad_(True)
    cutile_layernorm_autograd(xl2, w2, b2, eps=1e-5).backward(dy_l)
    assert xl2.grad is not None and w2.grad is not None and b2.grad is not None
    assert (xl2.grad - xl.grad).abs().max().item() < 1e-4


def test_linear_and_bmm_backwards_match_pytorch() -> None:
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    tol = 2e-2 if dtype == torch.float16 else 1e-4
    torch.manual_seed(1)

    x = torch.randn((8, 32), device=dev, dtype=dtype, requires_grad=True) * 0.2
    x = x.detach().requires_grad_(True)
    w = (torch.randn((48, 32), device=dev, dtype=dtype) * 0.1).requires_grad_(True)
    b = (torch.randn((48,), device=dev, dtype=dtype) * 0.05).requires_grad_(True)
    dy = torch.randn((8, 48), device=dev, dtype=dtype) * 0.2

    ref = F.linear(x, w, b)
    ref.backward(dy)

    dx_c, dw_c, db_c = cutile_linear_backward(dy, x.detach(), w.detach(), need_bias=True)
    assert db_c is not None
    assert (dx_c - x.grad).abs().max().item() < tol
    assert (dw_c - w.grad).abs().max().item() < tol
    assert (db_c - b.grad).abs().max().item() < tol

    x2 = x.detach().clone().requires_grad_(True)
    w2 = w.detach().clone().requires_grad_(True)
    b2 = b.detach().clone().requires_grad_(True)
    cutile_linear_autograd(x2, w2, b2).backward(dy)
    assert x2.grad is not None and w2.grad is not None and b2.grad is not None
    assert (x2.grad - x.grad).abs().max().item() < tol

    # BMM backward
    ba = (torch.randn((4, 16, 32), device=dev, dtype=dtype) * 0.1).requires_grad_(True)
    bb = (torch.randn((4, 32, 24), device=dev, dtype=dtype) * 0.1).requires_grad_(True)
    dy_b = torch.randn((4, 16, 24), device=dev, dtype=dtype) * 0.1
    torch.bmm(ba, bb).backward(dy_b)
    da_c, db_c2 = cutile_bmm_backward(dy_b, ba.detach(), bb.detach())
    assert (da_c - ba.grad).abs().max().item() < tol
    assert (db_c2 - bb.grad).abs().max().item() < tol


def test_mlp_training_convergence_matches_pytorch() -> None:
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(42)

    batch, d_in, d_hid, d_out = 16, 32, 64, 16
    x = torch.randn((batch, d_in), device=dev, dtype=torch.float32)
    target = torch.randn((batch, d_out), device=dev, dtype=torch.float32)

    # Identical initial weights
    init_w1 = torch.randn((d_hid, d_in), device=dev, dtype=torch.float32) * 0.1
    init_b1 = torch.zeros((d_hid,), device=dev, dtype=torch.float32)
    init_ln_w = torch.ones((d_hid,), device=dev, dtype=torch.float32)
    init_ln_b = torch.zeros((d_hid,), device=dev, dtype=torch.float32)
    init_w2 = torch.randn((d_out, d_hid), device=dev, dtype=torch.float32) * 0.1
    init_b2 = torch.zeros((d_out,), device=dev, dtype=torch.float32)

    cu_params = [
        t.clone().detach().requires_grad_(True)
        for t in (init_w1, init_b1, init_ln_w, init_ln_b, init_w2, init_b2)
    ]
    pt_params = [
        t.clone().detach().requires_grad_(True)
        for t in (init_w1, init_b1, init_ln_w, init_ln_b, init_w2, init_b2)
    ]

    m_state = [torch.zeros_like(p) for p in cu_params]
    v_state = [torch.zeros_like(p) for p in cu_params]
    pt_opt = torch.optim.AdamW(pt_params, lr=5e-2, weight_decay=0.01)

    cu_losses: list[float] = []
    pt_losses: list[float] = []

    for step in range(1, 11):
        # CuTile training step
        for p in cu_params:
            p.grad = None
        h = cutile_linear_autograd(x, cu_params[0], cu_params[1])
        h = cutile_layernorm_autograd(h, cu_params[2], cu_params[3], 1e-5)
        h = cutile_gelu_autograd(h)
        pred = cutile_linear_autograd(h, cu_params[4], cu_params[5])
        loss_cu = F.mse_loss(pred, target)
        loss_cu.backward()
        cutile_grad_clip_norm(cu_params, max_norm=1.0)
        grads = [p.grad if p.grad is not None else torch.zeros_like(p) for p in cu_params]
        cutile_multi_tensor_adamw(
            cu_params, grads, m_state, v_state, step=step, lr=5e-2, weight_decay=0.01
        )
        cu_losses.append(float(loss_cu.item()))

        # PyTorch reference step
        pt_opt.zero_grad()
        hp = F.linear(x, pt_params[0], pt_params[1])
        hp = F.layer_norm(hp, (d_hid,), pt_params[2], pt_params[3], 1e-5)
        hp = F.gelu(hp, approximate="tanh")
        pred_p = F.linear(hp, pt_params[4], pt_params[5])
        loss_pt = F.mse_loss(pred_p, target)
        loss_pt.backward()
        torch.nn.utils.clip_grad_norm_(pt_params, max_norm=1.0)
        pt_opt.step()
        pt_losses.append(float(loss_pt.item()))

    assert cu_losses[-1] < cu_losses[0] * 0.5
    assert abs(cu_losses[-1] - pt_losses[-1]) < 1e-4
    for p_cu, p_pt in zip(cu_params, pt_params, strict=True):
        assert (p_cu - p_pt).abs().max().item() < 1e-3
