"""Unit and GPU tests for CuTileForge v0.7.0 (Persistent Grouped MoE & FlashDecoding)."""

from __future__ import annotations

import torch
import torch.nn.functional as F

from cutile_forge.kernels.attention_gqa import flash_decoding_split_kv, fused_attention_gqa
from cutile_forge.kernels.gemm_dequant import pack_int4_weights, unpack_int4_weights_torch
from cutile_forge.kernels.moe_fused import fused_moe_int4_gemm, fused_topk_gating


def test_persistent_grouped_moe_matches_reference() -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    m, k, n, e, top_k, gs = 48, 256, 256, 8, 2, 128
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


def test_flash_decoding_split_kv_matches_sdpa() -> None:
    if not torch.cuda.is_available():
        return
    bsz, h_q, h_kv, s_q, s_kv, d = 1, 32, 8, 1, 4096, 128
    q = torch.randn((bsz, h_q, s_q, d), device="cuda", dtype=torch.float16) * 0.2
    k = torch.randn((bsz, h_kv, s_kv, d), device="cuda", dtype=torch.float16) * 0.2
    v = torch.randn((bsz, h_kv, s_kv, d), device="cuda", dtype=torch.float16) * 0.2

    out_fd = flash_decoding_split_kv(q, k, v, num_splits=16, causal=True)
    out_auto = fused_attention_gqa(q, k, v, causal=True)

    k_rep = k.repeat_interleave(h_q // h_kv, dim=1)
    v_rep = v.repeat_interleave(h_q // h_kv, dim=1)
    ref = F.scaled_dot_product_attention(q, k_rep, v_rep, is_causal=False)

    assert (out_fd - ref).abs().max().item() < 5e-3
    assert torch.allclose(out_fd, out_auto, atol=1e-4)
