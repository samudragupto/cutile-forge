"""Tests for Track A (v2.1.0) optimized kernels, fused norm+act, Split-K GEMM, and profiler."""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn.functional as F

import cutile_forge  # noqa: F401
import cutile_forge.forge.fx_backend as fx_mod
from cutile_forge.bench.profiler import (
    analytical_profile_op,
    detect_profilers,
    profile_kernel_ncu,
    save_profile_json,
)
from cutile_forge.kernels import (
    CuTilePatchEmbed,
    cutile_avg_pool2d,
    cutile_conv2d,
    cutile_depthwise_conv2d,
    cutile_fused_add_rmsnorm_silu,
    cutile_fused_layernorm_gelu,
    cutile_fused_rmsnorm_silu,
    cutile_gather,
    cutile_gelu_autograd,
    cutile_gemm_fp16,
    cutile_grad_clip_norm,
    cutile_layernorm_autograd,
    cutile_linear_autograd,
    cutile_max_pool2d,
    cutile_multi_tensor_adamw,
    cutile_reduce_max,
    cutile_reduce_sum,
    select_gemm_kernel_path,
)


def test_profiler_harness_and_fallback(tmp_path: Path) -> None:
    profs = detect_profilers()
    assert "ncu" in profs and "nsys" in profs

    x = torch.randn((16, 64), dtype=torch.float32)

    def _op() -> torch.Tensor:
        return x * 2.0

    rec = analytical_profile_op(
        "vec_scale",
        _op,
        bytes_transferred=2 * x.numel() * 4,
        flops=x.numel(),
        warmup=1,
        iters=3,
        ncu_diagnostic="ERR_NVGPUCTRPERM synthetic test",
    )
    assert rec.p50_ms > 0.0
    assert rec.achieved_gbps > 0.0
    out_json = save_profile_json([rec], tmp_path / "profile.json")
    assert out_json.exists() and "vec_scale" in out_json.read_text(encoding="utf-8")

    # Ensure profile_kernel_ncu returns structured dict without raising even if unprivileged
    probe = profile_kernel_ncu(["nonexistent_binary_for_fallback_test"], timeout_s=5)
    assert probe["fallback"] is True and probe["diagnostic"] is not None


def test_vision_and_conv_kernels_aligned_and_non_aligned() -> None:
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    tol = 1e-2 if dtype == torch.float16 else 1e-5
    torch.manual_seed(10)

    # Depthwise conv2d: aligned + odd spatial + bias + stride=2
    for n, c, h, w, kh, stride, pad, use_bias in [
        (4, 16, 28, 28, 3, 1, 1, True),
        (2, 12, 19, 23, 3, 2, 1, False),
    ]:
        x = torch.randn((n, c, h, w), device=dev, dtype=dtype) * 0.2
        wt = torch.randn((c, 1, kh, kh), device=dev, dtype=dtype) * 0.1
        b = torch.randn((c,), device=dev, dtype=dtype) * 0.05 if use_bias else None
        got = cutile_depthwise_conv2d(x, wt, bias=b, stride=stride, padding=pad)
        ref = F.conv2d(x, wt, bias=b, stride=stride, padding=pad, groups=c)
        assert got.shape == ref.shape
        assert (got - ref).abs().max().item() < tol

    # Pooling: 2x2 s=2 (even + odd) and 3x3 s=2
    xp = torch.randn((3, 8, 15, 19), device=dev, dtype=dtype) * 0.3
    for ks, st in [(2, 2), (3, 2)]:
        assert (cutile_max_pool2d(xp, ks, st) - F.max_pool2d(xp, ks, st)).abs().max().item() < tol
        assert (cutile_avg_pool2d(xp, ks, st) - F.avg_pool2d(xp, ks, st)).abs().max().item() < tol

    # Conv2D 1x1 (NCHW + channels_last NHWC + bias) and 3x3 implicit GEMM (non-aligned channels)
    x1 = torch.randn((2, 64, 14, 14), device=dev, dtype=dtype) * 0.2
    w1 = torch.randn((96, 64, 1, 1), device=dev, dtype=dtype) * 0.1
    b1 = torch.randn((96,), device=dev, dtype=dtype) * 0.05
    assert (cutile_conv2d(x1, w1, bias=b1) - F.conv2d(x1, w1, bias=b1)).abs().max().item() < tol
    if dev == "cuda":
        x1_nhwc = x1.contiguous(memory_format=torch.channels_last)
        assert (
            cutile_conv2d(x1_nhwc, w1, bias=b1) - F.conv2d(x1_nhwc, w1, bias=b1)
        ).abs().max().item() < tol

    x3 = torch.randn((2, 40, 17, 15), device=dev, dtype=dtype) * 0.2
    w3 = torch.randn((50, 40, 3, 3), device=dev, dtype=dtype) * 0.1
    b3 = torch.randn((50,), device=dev, dtype=dtype) * 0.05
    assert (
        cutile_conv2d(x3, w3, bias=b3, padding=1) - F.conv2d(x3, w3, bias=b3, padding=1)
    ).abs().max().item() < tol

    # PatchEmbed with and without pos_embed
    pe = CuTilePatchEmbed(img_size=32, patch_size=8, in_chans=3, embed_dim=96).to(
        device=dev, dtype=dtype
    )
    img = torch.randn((3, 3, 32, 32), device=dev, dtype=dtype) * 0.2
    pos = torch.randn((1, 16, 96), device=dev, dtype=dtype) * 0.05
    ref_pe = (
        F.conv2d(img, pe.weight, bias=pe.bias, stride=8).flatten(2).transpose(1, 2).contiguous()
        + pos
    )
    assert (pe(img, pos_embed=pos) - ref_pe).abs().max().item() < tol


def test_gather_splitk_reductions_fused_norm_act_and_gemm_dispatch() -> None:
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    tol = 1e-2 if dtype == torch.float16 else 1e-5
    torch.manual_seed(11)

    # Coalesced gather (int64 indices, wide d=512 + odd d=100)
    for d_dim in (512, 100):
        tbl = torch.randn((128, d_dim), device=dev, dtype=dtype)
        idx = torch.randint(0, 128, (3, 7), device=dev, dtype=torch.long)
        assert (cutile_gather(tbl, idx) - F.embedding(idx, tbl)).abs().max().item() < tol

    # Split-K low-row wide reductions
    xr = torch.randn((16, 4096), device=dev, dtype=dtype) * 0.2
    assert (
        cutile_reduce_sum(xr, dim=-1) - xr.float().sum(dim=-1).to(dtype)
    ).abs().max().item() < tol
    assert (cutile_reduce_max(xr, dim=-1) - xr.max(dim=-1).values).abs().max().item() < tol

    # Fused Norm + Activation kernels
    nx = torch.randn((8, 192), device=dev, dtype=dtype) * 0.2
    nr = torch.randn((8, 192), device=dev, dtype=dtype) * 0.2
    nw = torch.randn((192,), device=dev, dtype=dtype) * 0.5 + 1.0
    nb = torch.randn((192,), device=dev, dtype=dtype) * 0.05

    ref_rs = F.silu(
        (nx.float() * torch.rsqrt(nx.float().pow(2).mean(-1, keepdim=True) + 1e-5)).to(dtype) * nw
    )
    assert (cutile_fused_rmsnorm_silu(nx, nw) - ref_rs).abs().max().item() < tol

    ref_lg = F.gelu(F.layer_norm(nx, (192,), nw, nb, 1e-5), approximate="tanh")
    assert (cutile_fused_layernorm_gelu(nx, nw, nb) - ref_lg).abs().max().item() < tol

    comb = nx + nr
    ref_ars = F.silu(
        (comb.float() * torch.rsqrt(comb.float().pow(2).mean(-1, keepdim=True) + 1e-5)).to(dtype)
        * nw
    )
    got_ars, got_res = cutile_fused_add_rmsnorm_silu(nx, nr, nw)
    assert (got_ars - ref_ars).abs().max().item() < tol
    assert (got_res - comb).abs().max().item() < tol

    # FX layer_norm -> gelu(tanh) fusion pattern
    fx_mod.clear_fx_cache()

    def _ln_gelu_fn(t: torch.Tensor) -> torch.Tensor:
        return F.gelu(F.layer_norm(t, (192,), nw, nb, 1e-5), approximate="tanh")

    torch._dynamo.reset()
    compiled = torch.compile(_ln_gelu_fn, backend="cutile_forge")
    assert (compiled(nx) - _ln_gelu_fn(nx)).abs().max().item() < tol
    assert fx_mod.LAST_FX_STATS is not None and "fused_ln_gelu" in fx_mod.LAST_FX_STATS.lowered_ops

    # Shape-specialized GEMM dispatch + correctness across M regimes
    assert select_gemm_kernel_path(4, 4096, 4096)["regime"] == "skinny_splitk"
    assert select_gemm_kernel_path(64, 4096, 4096)["regime"] == "medium_l2"
    assert select_gemm_kernel_path(512, 4096, 4096)["regime"] == "large_compute"
    if dev == "cuda":
        for m_dim in (4, 64, 384):
            ga = torch.randn((m_dim, 512), device="cuda", dtype=torch.float16) * 0.1
            gb = torch.randn((512, 512), device="cuda", dtype=torch.float16) * 0.1
            assert (cutile_gemm_fp16(ga, gb) - torch.matmul(ga, gb)).abs().max().item() < 1e-2


def test_preserve_mlp_training_parity_under_1e6() -> None:
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(42)
    x = torch.randn((64, 32), device=dev, dtype=torch.float32)
    target = torch.randn((64, 16), device=dev, dtype=torch.float32)
    inits = [
        torch.randn((256, 32), device=dev, dtype=torch.float32) * 0.1,
        torch.zeros((256,), device=dev, dtype=torch.float32),
        torch.ones((256,), device=dev, dtype=torch.float32),
        torch.zeros((256,), device=dev, dtype=torch.float32),
        torch.randn((16, 256), device=dev, dtype=torch.float32) * 0.1,
        torch.zeros((16,), device=dev, dtype=torch.float32),
    ]
    p_cu = [t.clone().detach().requires_grad_(True) for t in inits]
    p_pt = [t.clone().detach().requires_grad_(True) for t in inits]
    m_cu = [torch.zeros_like(t) for t in p_cu]
    v_cu = [torch.zeros_like(t) for t in p_cu]
    opt = torch.optim.AdamW(p_pt, lr=5e-2, weight_decay=0.01)

    l_c, l_p = 0.0, 0.0
    for step in range(1, 11):
        for p in p_cu:
            p.grad = None
        h = cutile_gelu_autograd(
            cutile_layernorm_autograd(
                cutile_linear_autograd(x, p_cu[0], p_cu[1]), p_cu[2], p_cu[3], 1e-5
            )
        )
        loss_cu = F.mse_loss(cutile_linear_autograd(h, p_cu[4], p_cu[5]), target)
        loss_cu.backward()
        cutile_grad_clip_norm(p_cu, max_norm=1.0)
        cutile_multi_tensor_adamw(
            p_cu,
            [p.grad if p.grad is not None else torch.zeros_like(p) for p in p_cu],
            m_cu,
            v_cu,
            step=step,
            lr=5e-2,
            weight_decay=0.01,
        )
        l_c = float(loss_cu.item())

        opt.zero_grad()
        hp = F.gelu(
            F.layer_norm(F.linear(x, p_pt[0], p_pt[1]), (256,), p_pt[2], p_pt[3], 1e-5),
            approximate="tanh",
        )
        loss_pt = F.mse_loss(F.linear(hp, p_pt[4], p_pt[5]), target)
        loss_pt.backward()
        torch.nn.utils.clip_grad_norm_(p_pt, max_norm=1.0)
        opt.step()
        l_p = float(loss_pt.item())

    assert abs(l_c - l_p) < 1e-6
