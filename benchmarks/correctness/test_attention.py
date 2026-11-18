"""Correctness tests: CuTile GQA FlashAttention-2 vs PyTorch SDPA."""

from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

from cutile_forge.kernels.attention_gqa import fused_attention_gqa


def _ref_gqa(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, causal: bool
) -> torch.Tensor:
    hq, hkv = q.shape[1], k.shape[1]
    k_rep = k.repeat_interleave(hq // hkv, dim=1)
    v_rep = v.repeat_interleave(hq // hkv, dim=1)
    sq, skv = q.shape[2], k.shape[2]
    if causal and sq < skv:
        mask = torch.ones((sq, skv), device=q.device, dtype=torch.bool).tril(diagonal=skv - sq)
        return F.scaled_dot_product_attention(q, k_rep, v_rep, attn_mask=mask)
    return F.scaled_dot_product_attention(q, k_rep, v_rep, is_causal=causal)


@pytest.mark.gpu
@pytest.mark.parametrize(
    ("b", "hq", "hkv", "sq", "skv", "causal"),
    [
        (1, 32, 8, 64, 64, True),
        (1, 32, 8, 256, 256, True),
        (1, 32, 8, 1024, 1024, True),
        (1, 32, 8, 1, 128, True),
        (2, 32, 8, 128, 128, True),
        (1, 32, 8, 128, 128, False),
    ],
)
def test_attention_gqa(b: int, hq: int, hkv: int, sq: int, skv: int, causal: bool) -> None:
    if not torch.cuda.is_available():
        pytest.skip("CUDA GPU required")
    torch.manual_seed(1)
    q = torch.randn((b, hq, sq, 128), device="cuda", dtype=torch.float16) * 0.4
    k = torch.randn((b, hkv, skv, 128), device="cuda", dtype=torch.float16) * 0.4
    v = torch.randn((b, hkv, skv, 128), device="cuda", dtype=torch.float16) * 0.4

    out = fused_attention_gqa(q, k, v, causal=causal)
    ref = _ref_gqa(q, k, v, causal=causal)
    assert (out - ref).abs().max().item() < 1e-2

