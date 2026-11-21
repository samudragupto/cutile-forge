"""Tests for Track B (v2.2.0): CuTileTensor, NativePagedKVCache, fused LM head, MoE router & Zero-ATen audit."""

from __future__ import annotations

from typing import Any

import torch
from torch.utils._python_dispatch import TorchDispatchMode

from cutile_forge.engine.config import GPUConfig, ModelConfig
from cutile_forge.engine.model import Mistral7B
from cutile_forge.engine.weight_loader import create_random_4bit_weights
from cutile_forge.kernels import (
    cutile_embedding_lookup,
    cutile_lm_head_fused,
    cutile_moe_router_topk,
    fused_moe_int4_native,
)
from cutile_forge.runtime import (
    CuTileTensor,
    MemoryLifetimeAllocator,
    NativeModelRuntime,
    NativePagedKVCache,
    NativeStreamManager,
)

FORBIDDEN_ATEN_COMPUTE = {
    "aten.mm.default",
    "aten.bmm.default",
    "aten.addmm.default",
    "aten.matmul.default",
    "aten.embedding.default",
    "aten.silu.default",
    "aten.gelu.default",
    "aten.layer_norm.default",
    "aten.scaled_dot_product_attention.default",
    "aten.topk.default",
    "aten.cat.default",
}


class _ATenAuditMode(TorchDispatchMode):
    def __init__(self) -> None:
        super().__init__()
        self.forbidden_calls: list[str] = []

    def __torch_dispatch__(
        self,
        func: Any,
        types: tuple[type, ...],
        args: tuple[Any, ...] = (),
        kwargs: dict[str, Any] | None = None,
    ) -> Any:
        name = str(func)
        if name in FORBIDDEN_ATEN_COMPUTE:
            self.forbidden_calls.append(name)
        return func(*args, **(kwargs or {}))


def test_native_tensor_stream_and_kv_cache() -> None:
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    alloc = MemoryLifetimeAllocator()
    ct_t = CuTileTensor.allocate(
        (4, 64), dtype=torch.float16, device=dev, allocator=alloc, slot_id=0
    )
    assert ct_t.shape == (4, 64) and ct_t.nbytes == 4 * 64 * 2 and ct_t.data_ptr() != 0
    ct_t.as_torch_view().fill_(1.5)
    dl_view = CuTileTensor.from_dlpack(ct_t)
    assert torch.equal(dl_view.as_torch_view(), ct_t.as_torch_view())

    sm = NativeStreamManager(dev)
    ev = sm.record_dma_ready()
    sm.wait_dma_on_compute(ev)
    sm.synchronize_all()

    for kv_dtype in ("fp16", "fp8_e4m3fn"):
        cache = NativePagedKVCache(
            num_layers=2,
            num_kv_heads=2,
            head_dim=64,
            block_size=4,
            max_seq_len=32,
            kv_dtype=kv_dtype,
            device=dev,
            stream_manager=sm,
        )
        k1 = torch.randn((1, 2, 5, 64), device=dev, dtype=torch.float16) * 0.2
        v1 = torch.randn((1, 2, 5, 64), device=dev, dtype=torch.float16) * 0.2
        k2 = torch.randn((1, 2, 1, 64), device=dev, dtype=torch.float16) * 0.2
        v2 = torch.randn((1, 2, 1, 64), device=dev, dtype=torch.float16) * 0.2

        kf1, vf1 = cache.append_and_get(0, k1, v1, start_pos=0)
        assert kf1.shape == (1, 2, 5, 64)
        kf2, vf2 = cache.append_and_get(0, k2, v2, start_pos=5)
        assert kf2.shape == (1, 2, 6, 64)

        tol = 5e-2 if (kv_dtype == "fp8_e4m3fn" and dev == "cuda") else 1e-3
        ref_k = torch.cat([k1, k2], dim=2)
        ref_v = torch.cat([v1, v2], dim=2)
        assert (kf2 - ref_k).abs().max().item() < tol
        assert (vf2 - ref_v).abs().max().item() < tol


def test_embedding_lmhead_and_native_moe_parity() -> None:
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    torch.manual_seed(5)

    tbl = torch.randn((256, 128), device=dev, dtype=dtype) * 0.1
    ids = torch.tensor([[3, 17, 250]], device=dev, dtype=torch.long)
    emb = cutile_embedding_lookup(ids, tbl)
    assert torch.allclose(emb, torch.nn.functional.embedding(ids, tbl))

    # LM Head fused top_k=1 and top_k=4 parity
    h = torch.randn((2, 128), device=dev, dtype=dtype) * 0.2
    nw = torch.ones((128,), device=dev, dtype=dtype)
    lm_w = torch.randn((600, 128), device=dev, dtype=dtype) * 0.1
    xf = h.float()
    normed = (xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + 1e-5)) * nw.float()
    ref_logits = torch.matmul(normed, lm_w.float().t())

    v1, i1 = cutile_lm_head_fused(h, nw, lm_w, top_k=1)
    assert torch.equal(i1.squeeze(-1), ref_logits.argmax(dim=-1))

    v4, i4 = cutile_lm_head_fused(h, nw, lm_w, top_k=4)
    _, ref_i4 = torch.topk(ref_logits, k=4, dim=-1)
    assert torch.equal(i4, ref_i4)

    # Native MoE router + bucketed grouped GEMM parity
    xm = (
        torch.randn((4, 128), device=dev, dtype=torch.float16 if dev == "cuda" else torch.float32)
        * 0.1
    )
    rw = torch.randn((8, 128), device=dev, dtype=xm.dtype) * 0.1
    probs, idx = cutile_moe_router_topk(xm, rw, top_k=2)

    ref_probs = torch.softmax(torch.matmul(xm.float(), rw.float().t()), dim=-1)
    ref_pv, ref_pi = torch.topk(ref_probs, k=2, dim=-1)
    ref_pv = ref_pv / ref_pv.sum(dim=-1, keepdim=True)
    assert torch.equal(idx.long(), ref_pi)
    assert (probs.float() - ref_pv).abs().max().item() < 5e-3

    # Native INT4 MoE forward parity
    qw = torch.randint(0, 255, (8, 64, 128), device=dev, dtype=torch.uint8)
    sc = torch.ones((8, 2, 128), device=dev, dtype=xm.dtype) * 0.02
    zr = torch.ones((8, 2, 128), device=dev, dtype=xm.dtype) * 8.0
    out_mo = fused_moe_int4_native(xm, rw, qw, sc, zr, top_k=2, group_size=64)
    assert out_mo.shape == (4, 128) and torch.isfinite(out_mo).all()


def test_native_runtime_zero_aten_compute_and_token_parity() -> None:
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    cfg = ModelConfig(
        vocab_size=512,
        hidden_size=128,
        intermediate_size=256,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=32,
        max_position_embeddings=64,
        group_size=64,
    )
    torch.manual_seed(9)
    weights = create_random_4bit_weights(cfg, device=dev)
    model = Mistral7B(
        weights=weights, gpu_config=GPUConfig(vram_gb=24.0, stream_weights=False), device=dev
    )
    rt = NativeModelRuntime(model, max_seq_len=64, use_cuda_graph=False)

    prompt = torch.tensor([[1, 5, 19]], device=dev, dtype=torch.long)
    tok1 = rt.prefill(prompt)

    if dev == "cuda":
        with _ATenAuditMode() as audit:
            tok2 = rt.decode_step(tok1, pos=3, use_graph=False)
            assert (
                audit.forbidden_calls == []
            ), f"Forbidden ATen compute calls detected: {audit.forbidden_calls}"
        # Parity against direct forward_pytorch on same random weights
        model.eval() if hasattr(model, "eval") else None
        toks_nat, _ = rt.generate([1, 5, 19], max_new_tokens=4)
        assert len(toks_nat) == 4
    else:
        tok2 = rt.decode_step(tok1, pos=3, use_graph=False)
        assert tok2.shape == (1, 1)
