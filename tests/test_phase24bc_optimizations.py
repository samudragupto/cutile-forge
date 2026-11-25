"""Unit tests for Stage 24B (RMSNorm-permute & Split-K INT4) and Stage 24C (direct FP8 KV attention)."""

from __future__ import annotations

import pytest
import torch

from cutile_forge.kernels.attention_gqa import flash_decoding_split_kv
from cutile_forge.kernels.gemm_dequant import (
    fused_gemm_dequant,
    fused_gemm_dequant_add,
    pack_int4_weights,
    unpack_int4_weights_torch,
)
from cutile_forge.kernels.gemm_dequant_silu import fused_swiglu_dequant
from cutile_forge.kernels.layernorm_residual import fused_rmsnorm_residual
from cutile_forge.kernels.rmsnorm import cutile_rmsnorm
from cutile_forge.runtime.native_kv_cache import NativePagedKVCache


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_rmsnorm_and_residual_fused_permute_parity() -> None:
    torch.manual_seed(0)
    d = 4096
    w = torch.randn((d,), device="cuda", dtype=torch.float16) * 0.5 + 1.0
    perm = torch.randperm(d, device="cuda", dtype=torch.int32)
    perm_l = perm.long()

    for m in (1, 16, 128):
        x = torch.randn((m, d), device="cuda", dtype=torch.float16) * 0.5
        res = torch.randn((m, d), device="cuda", dtype=torch.float16) * 0.5

        y_ref = cutile_rmsnorm(x, w, eps=1e-5)[:, perm_l]
        y_fused = cutile_rmsnorm(x, w, eps=1e-5, perm=perm)
        assert (y_fused - y_ref).abs().max().item() < 1e-3

        n_ref, r_ref = fused_rmsnorm_residual(x, res, w, eps=1e-5)
        n_fused, r_fused = fused_rmsnorm_residual(x, res, w, eps=1e-5, perm=perm)
        assert (r_fused - r_ref).abs().max().item() == 0.0
        assert (n_fused - n_ref[:, perm_l]).abs().max().item() < 1e-3


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_splitk_int4_dequant_gemm_and_swiglu_parity() -> None:
    torch.manual_seed(1)

    def _mk_w(k: int, n: int, gs: int = 128):
        raw = torch.randint(0, 16, (k, n), device="cuda", dtype=torch.uint8)
        qw = pack_int4_weights(raw)
        sc = (
            torch.rand((k // gs, n), device="cuda", dtype=torch.float16) * 0.02 + 0.005
        ).contiguous()
        zr = torch.full((k // gs, n), 8.0, device="cuda", dtype=torch.float16).contiguous()
        return qw, sc, zr

    qw_qkv, sc_qkv, zr_qkv = _mk_w(4096, 6144)
    w_qkv = unpack_int4_weights_torch(qw_qkv, sc_qkv, zr_qkv, 128)
    qw_dn, sc_dn, zr_dn = _mk_w(14336, 4096)
    w_dn = unpack_int4_weights_torch(qw_dn, sc_dn, zr_dn, 128)
    qw_g, sc_g, zr_g = _mk_w(4096, 14336)
    w_g = unpack_int4_weights_torch(qw_g, sc_g, zr_g, 128)
    qw_u, sc_u, zr_u = _mk_w(4096, 14336)
    w_u = unpack_int4_weights_torch(qw_u, sc_u, zr_u, 128)

    part1 = torch.empty((8 * 16 * 14336,), device="cuda", dtype=torch.float32)
    part2 = torch.empty((4 * 16 * 14336,), device="cuda", dtype=torch.float32)

    for m in (1, 16):
        a = torch.randn((m, 4096), device="cuda", dtype=torch.float16) * 0.2
        y_qkv = fused_gemm_dequant(a, qw_qkv, sc_qkv, zr_qkv, 128, part_buf=part1)
        ref_qkv = (a.float() @ w_qkv.float()).to(torch.float16)
        assert (y_qkv - ref_qkv).abs().max().item() < 0.05
        assert (
            torch.nn.functional.cosine_similarity(
                y_qkv.flatten().float(), ref_qkv.flatten().float(), dim=0
            )
            > 0.9999
        )

        y_sw = fused_swiglu_dequant(
            a, qw_g, sc_g, zr_g, qw_u, sc_u, zr_u, 128, part_g_buf=part1, part_u_buf=part2
        )
        g_ref = a.float() @ w_g.float()
        u_ref = a.float() @ w_u.float()
        ref_sw = (torch.nn.functional.silu(g_ref) * u_ref).to(torch.float16)
        assert (y_sw - ref_sw).abs().max().item() < 0.05
        assert (
            torch.nn.functional.cosine_similarity(
                y_sw.flatten().float(), ref_sw.flatten().float(), dim=0
            )
            > 0.9999
        )

        af = torch.randn((m, 14336), device="cuda", dtype=torch.float16) * 0.1
        res = torch.randn((m, 4096), device="cuda", dtype=torch.float16) * 0.2
        y_dn = fused_gemm_dequant_add(af, qw_dn, sc_dn, zr_dn, res, 128, part_buf=part1)
        ref_dn = (af.float() @ w_dn.float()).to(torch.float16) + res
        assert (y_dn - ref_dn).abs().max().item() < 0.08
        assert (
            torch.nn.functional.cosine_similarity(
                y_dn.flatten().float(), ref_dn.flatten().float(), dim=0
            )
            > 0.9999
        )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_flash_decoding_direct_fp8_kv_parity_and_lazy_scratchpad() -> None:
    torch.manual_seed(2)
    ctx_len = 2048
    kvc = NativePagedKVCache(
        num_layers=2,
        num_kv_heads=8,
        head_dim=128,
        block_size=16,
        max_seq_len=ctx_len,
        max_batch_size=1,
        kv_dtype="fp8_e4m3fn",
        device="cuda",
    )
    assert kvc._k_work == [] and kvc._v_work == []

    k_in = torch.randn((1, 8, ctx_len, 128), device="cuda", dtype=torch.float16) * 0.2
    v_in = torch.randn((1, 8, ctx_len, 128), device="cuda", dtype=torch.float16) * 0.2
    kvc.append_and_get(0, k_in, v_in, start_pos=0)
    k_fp8, v_fp8 = kvc.get_kv(0, end_pos=ctx_len, allow_native_dtype=True)
    assert k_fp8.dtype == torch.float8_e4m3fn and v_fp8.dtype == torch.float8_e4m3fn

    q = torch.randn((1, 32, 1, 128), device="cuda", dtype=torch.float16) * 0.1
    out_direct = flash_decoding_split_kv(q, k_fp8, v_fp8, num_splits=16, causal=False)
    out_upcast = flash_decoding_split_kv(
        q, k_fp8.to(torch.float16), v_fp8.to(torch.float16), num_splits=16, causal=False
    )
    assert (out_direct - out_upcast).abs().max().item() < 1e-3


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_native_runtime_inplace_gate_up_perm_zero_duplication() -> None:
    """Verify NativeModelRuntime permutes gate/up weights in-place without duplicating weights."""
    from cutile_forge.engine.config import GPUConfig, ModelConfig
    from cutile_forge.engine.model import Mistral7B
    from cutile_forge.engine.weight_loader import create_random_4bit_weights
    from cutile_forge.runtime import NativeModelRuntime

    cfg = ModelConfig(
        vocab_size=256,
        hidden_size=128,
        intermediate_size=256,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=32,
        group_size=128,
    )
    weights = create_random_4bit_weights(cfg, seed=7, device="cuda")
    for idx, lw in enumerate(weights.layers):
        g = torch.Generator(device="cuda").manual_seed(100 + idx)
        lw.down_proj.perm = torch.randperm(
            cfg.intermediate_size, generator=g, device="cuda", dtype=torch.int32
        )
    model = Mistral7B(weights=weights, gpu_config=GPUConfig(vram_gb=32.0), device="cuda")
    inp = torch.tensor([[1, 5, 9, 13]], device="cuda", dtype=torch.long)
    with torch.no_grad():
        logits_before = model.forward(inp, kv_cache=None, pos_offset=0, backend="pytorch").float()

    rt = NativeModelRuntime(model, max_seq_len=128, block_size=16, use_cuda_graph=False)
    for lw, p in zip(weights.layers, rt._packed_layers, strict=True):
        assert p.gate_proj is lw.gate_proj
        assert p.up_proj is lw.up_proj
        assert p.gate_proj.qweight.data_ptr() == lw.gate_proj.qweight.data_ptr()
        assert p.up_proj.qweight.data_ptr() == lw.up_proj.qweight.data_ptr()
        assert lw.down_proj.perm is None
        assert p.down_perm_i32 is None

    with torch.no_grad():
        logits_after = model.forward(inp, kv_cache=None, pos_offset=0, backend="pytorch").float()
    assert torch.allclose(logits_after, logits_before, atol=1e-2, rtol=1e-2)
    assert (
        torch.nn.functional.cosine_similarity(
            logits_after.flatten(), logits_before.flatten(), dim=0
        ).item()
        > 0.99999
    )
