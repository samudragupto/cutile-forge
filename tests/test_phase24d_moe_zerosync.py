"""Stage 24D tests: zero-sync device-indexed CuTile MoE FFN parity & zero host-sync verification."""

from __future__ import annotations

import pytest
import torch

from cutile_forge.engine.weight_loader import MistralLayerWeights, QuantLinearWeight
from cutile_forge.kernels.gemm_dequant import pack_int4_weights, unpack_int4_weights_torch
from cutile_forge.kernels.moe_fused import cutile_moe_indexed_ffn


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_moe_indexed_ffn_parity_and_zero_host_sync() -> None:
    torch.manual_seed(7)
    num_exp, d_model, ffn_dim, gs = 4, 256, 512, 128

    def _mk_exps(k: int, n: int) -> list[QuantLinearWeight]:
        out: list[QuantLinearWeight] = []
        for _ in range(num_exp):
            qw = pack_int4_weights(torch.randint(0, 16, (k, n), device="cuda", dtype=torch.uint8))
            sc = (
                torch.rand((k // gs, n), device="cuda", dtype=torch.float16) * 0.02 + 0.005
            ).contiguous()
            zr = torch.full((k // gs, n), 8.0, device="cuda", dtype=torch.float16).contiguous()
            out.append(QuantLinearWeight(qw, sc, zr, group_size=gs, perm=None))
        return out

    w1 = _mk_exps(d_model, ffn_dim)
    w2 = _mk_exps(ffn_dim, d_model)
    w3 = _mk_exps(d_model, ffn_dim)
    dummy = _mk_exps(d_model, d_model)[0]
    lw = MistralLayerWeights(
        layer_idx=0,
        input_layernorm=torch.ones((d_model,), device="cuda", dtype=torch.float16),
        post_attention_layernorm=torch.ones((d_model,), device="cuda", dtype=torch.float16),
        q_proj=dummy,
        k_proj=dummy,
        v_proj=dummy,
        o_proj=dummy,
        gate_proj=w1[0],
        up_proj=w3[0],
        down_proj=w2[0],
        moe_gate=torch.randn((num_exp, d_model), device="cuda", dtype=torch.float16) * 0.1,
        moe_w1=w1,
        moe_w2=w2,
        moe_w3=w3,
    )
    assert lw.pack_moe_experts_inplace()
    assert (
        lw.moe_w1_qw is not None
        and lw.moe_w1_sc is not None
        and lw.moe_w1_zr is not None
        and lw.moe_w2_qw is not None
        and lw.moe_w2_sc is not None
        and lw.moe_w2_zr is not None
        and lw.moe_w3_qw is not None
        and lw.moe_w3_sc is not None
        and lw.moe_w3_zr is not None
    )

    x = torch.randn((1, d_model), device="cuda", dtype=torch.float16) * 0.2
    res = torch.randn((1, d_model), device="cuda", dtype=torch.float16) * 0.2
    indices = torch.tensor([[1, 3]], device="cuda", dtype=torch.int32)
    probs = torch.tensor([[0.7, 0.3]], device="cuda", dtype=torch.float32)

    # Warmup compile before strict zero-sync check
    _ = cutile_moe_indexed_ffn(
        x,
        lw.moe_w1_qw,
        lw.moe_w1_sc,
        lw.moe_w1_zr,
        lw.moe_w3_qw,
        lw.moe_w3_sc,
        lw.moe_w3_zr,
        lw.moe_w2_qw,
        lw.moe_w2_sc,
        lw.moe_w2_zr,
        probs,
        indices,
        residual=res,
        group_size=gs,
    )
    torch.cuda.synchronize()

    # Assert strictly ZERO host synchronizations during MoE execution
    prev_mode = torch.cuda.get_sync_debug_mode()
    torch.cuda.set_sync_debug_mode("error")
    try:
        out = cutile_moe_indexed_ffn(
            x,
            lw.moe_w1_qw,
            lw.moe_w1_sc,
            lw.moe_w1_zr,
            lw.moe_w3_qw,
            lw.moe_w3_sc,
            lw.moe_w3_zr,
            lw.moe_w2_qw,
            lw.moe_w2_sc,
            lw.moe_w2_zr,
            probs,
            indices,
            residual=res,
            group_size=gs,
        )
    finally:
        torch.cuda.set_sync_debug_mode(prev_mode)
    torch.cuda.synchronize()

    # Reference computation via stacked views (verifies inplace packing preserved views)
    acc = torch.zeros((1, d_model), device="cuda", dtype=torch.float32)
    for k_slot, e_i in enumerate(indices[0].tolist()):
        W1 = unpack_int4_weights_torch(w1[e_i].qweight, w1[e_i].scales, w1[e_i].zeros, gs)
        W3 = unpack_int4_weights_torch(w3[e_i].qweight, w3[e_i].scales, w3[e_i].zeros, gs)
        W2 = unpack_int4_weights_torch(w2[e_i].qweight, w2[e_i].scales, w2[e_i].zeros, gs)
        h = (torch.nn.functional.silu(x.float() @ W1.float()) * (x.float() @ W3.float())).to(
            torch.float16
        )
        y = (h.float() @ W2.float()).to(torch.float16)
        acc = acc + y.float() * probs[0, k_slot]
    ref = acc.to(torch.float16) + res

    cos = torch.nn.functional.cosine_similarity(
        out.flatten().float(), ref.flatten().float(), dim=0
    ).item()
    assert cos > 0.9999
    assert (out - ref).abs().max().item() < 0.05
