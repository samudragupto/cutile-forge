"""Unit and GPU tests for CuTileForge v0.4.0 (FP8 KV-cache, NVFP4, Fused MoE, AWQ/GGUF)."""

from __future__ import annotations

import torch

from cutile_forge.engine.kv_cache import PagedKVCache
from cutile_forge.kernels.gemm_dequant import (
    fused_gemm_nvfp4_dequant,
    pack_int4_weights,
    unpack_int4_weights_torch,
)
from cutile_forge.kernels.moe_fused import fused_moe_int4_gemm, fused_topk_gating
from cutile_forge.utils.quantize import unpack_awq_to_cutile, unpack_gguf_q4_k_to_cutile


def test_fp8_kv_cache_memory_and_accuracy() -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    kv16 = PagedKVCache(
        num_layers=1, num_kv_heads=4, head_dim=32, page_size=8, device=device, kv_dtype="fp16"
    )
    kv8 = PagedKVCache(
        num_layers=1, num_kv_heads=4, head_dim=32, page_size=8, device=device, kv_dtype="fp8_e4m3fn"
    )

    k = torch.randn((1, 4, 12, 32), device=device, dtype=torch.float16) * 0.2
    v = torch.randn((1, 4, 12, 32), device=device, dtype=torch.float16) * 0.2
    kv16.update(0, k, v)
    kv8.update(0, k, v)

    if torch.cuda.is_available():
        scale_bytes = len(kv8.pages) * 2 * kv8.page_size * 4
        assert kv8.gpu_used_bytes == kv16.gpu_used_bytes // 2 + scale_bytes

    k_got, v_got = kv8.get_kv(0)
    assert (k_got - k).abs().mean().item() < 1e-2
    assert (v_got - v).abs().mean().item() < 1e-2


def test_nvfp4_dequant_gemm_accuracy() -> None:
    if not torch.cuda.is_available():
        return
    m, n, k, bs = 64, 128, 256, 128
    a = torch.randn((m, k), device="cuda", dtype=torch.float16) * 0.1
    qw = torch.randint(0, 255, (k // 2, n), device="cuda", dtype=torch.uint8)
    scales = torch.rand((k // bs, n), device="cuda", dtype=torch.float16) * 0.5 + 0.1

    out = fused_gemm_nvfp4_dequant(a, qw, scales)
    lut = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0], device="cuda", dtype=torch.float16)
    lo = qw & 0x0F
    hi = (qw >> 4) & 0x0F
    lo_v = torch.where((lo >> 3) != 0, -lut[(lo & 7).long()], lut[(lo & 7).long()])
    hi_v = torch.where((hi >> 3) != 0, -lut[(hi & 7).long()], lut[(hi & 7).long()])
    w = torch.stack([lo_v, hi_v], dim=1).reshape(k, n) * scales.repeat_interleave(bs, dim=0)
    ref = a @ w
    assert (out - ref).abs().max().item() < 1e-2


def test_fused_moe_8_experts_topk2() -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    m, k, n, e, top_k, gs = 32, 256, 128, 8, 2, 128
    x = torch.randn((m, k), device=device, dtype=torch.float16) * 0.05
    logits = torch.randn((m, e), device=device, dtype=torch.float16)
    probs, indices = fused_topk_gating(logits, top_k=top_k)

    qw = torch.stack(
        [
            pack_int4_weights(torch.randint(0, 16, (k, n), device=device, dtype=torch.uint8))
            for _ in range(e)
        ]
    )
    sc = torch.rand((e, k // gs, n), device=device, dtype=torch.float16) * 0.05 + 0.01
    zr = torch.randint(4, 12, (e, k // gs, n), device=device, dtype=torch.float16)

    out = fused_moe_int4_gemm(x, qw, sc, zr, probs, indices, group_size=gs)

    ref = torch.zeros((m, n), device=device, dtype=torch.float16)
    for tok in range(m):
        for slot in range(top_k):
            exp_id = int(indices[tok, slot].item())
            w_fp16 = unpack_int4_weights_torch(qw[exp_id], sc[exp_id], zr[exp_id], group_size=gs)
            ref[tok] += probs[tok, slot] * (x[tok : tok + 1] @ w_fp16).squeeze(0)

    assert (out - ref).abs().max().item() < 1e-2


def test_awq_and_gguf_unpack_roundtrips() -> None:
    k, n, gs = 128, 64, 128
    awq_shifts = torch.tensor([0, 16, 4, 20, 8, 24, 12, 28], dtype=torch.int32)
    w_int = torch.randint(0, 16, (k, n), dtype=torch.int32)
    z_int = torch.randint(1, 15, (k // gs, n), dtype=torch.int32)
    qw_awq = torch.sum((w_int.view(k, n // 8, 8) << awq_shifts), dim=-1, dtype=torch.int32)
    qz_awq = torch.sum((z_int.view(k // gs, n // 8, 8) << awq_shifts), dim=-1, dtype=torch.int32)
    scales = torch.rand((k // gs, n), dtype=torch.float16) + 0.1

    qw_ct, s_ct, z_ct = unpack_awq_to_cutile(qw_awq, qz_awq, scales)
    assert qw_ct.shape == (k // 2, n)
    w_rec = unpack_int4_weights_torch(qw_ct, s_ct, z_ct, group_size=gs)
    w_ref = (w_int.to(torch.float16) - z_int.to(torch.float16)) * scales
    assert torch.allclose(w_rec, w_ref, atol=1e-3)

    # GGUF Q4_K_M roundtrip shape & finite values
    num_blocks = (k * n) // 256
    blocks = torch.randint(0, 255, (num_blocks, 144), dtype=torch.uint8)
    # Set valid small fp16 d=0.5, dmin=0.1
    blocks[:, 0:2] = torch.tensor([0.5], dtype=torch.float16).view(torch.uint8)
    blocks[:, 2:4] = torch.tensor([0.1], dtype=torch.float16).view(torch.uint8)
    qw_g, s_g, z_g = unpack_gguf_q4_k_to_cutile(blocks.reshape(-1), k, n)
    assert qw_g.shape == (k // 2, n)
    assert s_g.shape == (k // 128, n)
    assert z_g.shape == (k // 128, n)
    assert torch.isfinite(s_g).all() and torch.isfinite(z_g).all()
