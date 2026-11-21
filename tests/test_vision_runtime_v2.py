"""Tests for Runtime pool reuse, adaptive streaming, and Vision kernels (Phase 8G + 8H)."""

from __future__ import annotations

import torch
import torch.nn.functional as F

from cutile_forge.kernels import (
    CuTilePatchEmbed,
    cutile_avg_pool2d,
    cutile_conv2d,
    cutile_conv2d_backward,
    cutile_depthwise_conv2d,
    cutile_gelu,
    cutile_gelu_backward,
    cutile_max_pool2d,
)
from cutile_forge.runtime import MemoryLifetimeAllocator, TensorStreamer, p2p_capable


def test_memory_lifetime_allocator_and_adaptive_dma() -> None:
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    alloc = MemoryLifetimeAllocator()

    # Reuse slot 0 across multiple tensors -> peak_bytes < total_requested_bytes
    t1 = alloc.allocate_slot(0, (32, 64), torch.float32, device=dev)
    t1.fill_(1.0)
    t2 = alloc.allocate_slot(1, (16, 64), torch.float32, device=dev)
    t2.fill_(2.0)
    t3 = alloc.allocate_slot(0, (32, 64), torch.float16, device=dev)
    t3.zero_()
    t4 = alloc.allocate_slot(0, (32, 64), torch.float32, device=dev)
    t4.fill_(3.0)

    assert alloc.reuse_hits >= 2
    assert alloc.peak_bytes < alloc.total_requested_bytes
    assert alloc.reuse_ratio() > 0.25

    alloc.reset()
    assert alloc.peak_bytes == 0 and alloc.reuse_hits == 0

    # Adaptive slot sizing
    streamer = TensorStreamer(num_slots=6, max_vram_bytes=4096, device=dev, adaptive_slots=True)
    big_t = torch.ones((512,), dtype=torch.float32)  # 2048 bytes -> adaptive num_slots = 2
    streamer.prefetch("k0", big_t)
    streamer.prefetch("k1", big_t)
    streamer.prefetch("k2", big_t)
    assert streamer.num_slots == 2
    assert streamer.evictions >= 1
    streamer.clear()

    assert isinstance(p2p_capable(0, 1), bool)


def test_vision_ops_and_convnet_parity() -> None:
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    tol = 2e-2 if dtype == torch.float16 else 1e-4
    torch.manual_seed(7)

    # Depthwise conv2d
    x_dw = torch.randn((2, 8, 12, 12), device=dev, dtype=dtype) * 0.2
    w_dw = torch.randn((8, 1, 3, 3), device=dev, dtype=dtype) * 0.1
    b_dw = torch.randn((8,), device=dev, dtype=dtype) * 0.05
    got_dw = cutile_depthwise_conv2d(x_dw, w_dw, bias=b_dw, stride=1, padding=1)
    ref_dw = F.conv2d(x_dw, w_dw, bias=b_dw, stride=1, padding=1, groups=8)
    assert (got_dw - ref_dw).abs().max().item() < tol

    # MaxPool2d & AvgPool2d
    got_mp = cutile_max_pool2d(x_dw, kernel_size=2, stride=2, padding=0)
    ref_mp = F.max_pool2d(x_dw, kernel_size=2, stride=2, padding=0)
    assert (got_mp - ref_mp).abs().max().item() < tol

    got_ap = cutile_avg_pool2d(x_dw, kernel_size=2, stride=2, padding=0)
    ref_ap = F.avg_pool2d(x_dw, kernel_size=2, stride=2, padding=0)
    assert (got_ap - ref_ap).abs().max().item() < tol

    # CuTilePatchEmbed vs PyTorch Conv2d flatten+transpose
    patch_embed = CuTilePatchEmbed(img_size=16, patch_size=4, in_chans=3, embed_dim=16).to(
        device=dev, dtype=dtype
    )
    img = torch.randn((2, 3, 16, 16), device=dev, dtype=dtype) * 0.2
    got_pe = patch_embed(img)
    ref_pe = (
        F.conv2d(img, patch_embed.weight, bias=patch_embed.bias, stride=4)
        .flatten(2)
        .transpose(1, 2)
        .contiguous()
    )
    assert got_pe.shape == (2, 16, 16)
    assert (got_pe - ref_pe).abs().max().item() < tol

    # 2-layer ConvNet (Conv2d -> GELU -> AvgPool2d -> DepthwiseConv2d) + Conv2d backward
    x_in = torch.randn((2, 4, 8, 8), device=dev, dtype=dtype, requires_grad=True)
    x_ref = x_in.detach().clone().requires_grad_(True)
    w1 = (torch.randn((8, 4, 3, 3), device=dev, dtype=dtype) * 0.1).detach().requires_grad_(True)
    b1 = (torch.randn((8,), device=dev, dtype=dtype) * 0.02).detach().requires_grad_(True)
    w2 = torch.randn((8, 1, 3, 3), device=dev, dtype=dtype) * 0.1

    # Forward
    c1 = cutile_conv2d(x_in.detach(), w1.detach(), bias=b1.detach(), stride=1, padding=1)
    g1 = cutile_gelu(c1)
    p1 = cutile_avg_pool2d(g1, kernel_size=2, stride=2)
    out_cu = cutile_depthwise_conv2d(p1, w2, stride=1, padding=1)

    c1_pt = F.conv2d(x_ref, w1, bias=b1, stride=1, padding=1)
    g1_pt = F.gelu(c1_pt, approximate="tanh")
    p1_pt = F.avg_pool2d(g1_pt, kernel_size=2, stride=2)
    out_pt = F.conv2d(p1_pt, w2, stride=1, padding=1, groups=8)
    assert (out_cu - out_pt).abs().max().item() < tol

    # Backward parity on Conv2d + GELU chain
    dy_g1 = torch.randn_like(g1_pt)
    g1_pt.backward(dy_g1)
    dy_c1 = cutile_gelu_backward(dy_g1, c1.detach())
    dx_cu, dw_cu, db_cu = cutile_conv2d_backward(
        dy_c1, x_in.detach(), w1.detach(), stride=1, padding=1, need_bias=True
    )
    assert db_cu is not None
    assert (dx_cu - x_ref.grad).abs().max().item() < tol
    assert (dw_cu - w1.grad).abs().max().item() < tol
    assert (db_cu - b1.grad).abs().max().item() < tol
