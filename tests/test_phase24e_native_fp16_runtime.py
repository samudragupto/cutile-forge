"""Stage 24E tests: NativeFP16DecoderRuntime (Qwen2.5-0.5B), FP16 SwiGLU & epilogue GEMMs."""

from __future__ import annotations

from typing import Any

import pytest
import torch
from torch.utils._python_dispatch import TorchDispatchMode

from cutile_forge.engine.kv_cache import PagedKVCache
from cutile_forge.kernels.gemm_dequant import cutile_gemm_fp16
from cutile_forge.kernels.gemm_dequant_silu import fused_swiglu_fp16
from cutile_forge.models import (
    DenseTransformerBlock,
    UnifiedModelExecutor,
    build_model_graph,
    extract_hf_decoder_weights,
    init_synthetic_weights,
)
from cutile_forge.runtime import NativeFP16DecoderRuntime

FORBIDDEN_ATEN_COMPUTE = {
    "aten.mm.default",
    "aten.bmm.default",
    "aten.addmm.default",
    "aten.matmul.default",
    "aten.linear.default",
    "aten.embedding.default",
    "aten.silu.default",
    "aten.gelu.default",
    "aten.sigmoid.default",
    "aten.layer_norm.default",
    "aten.native_layer_norm.default",
    "aten.rms_norm.default",
    "aten.scaled_dot_product_attention.default",
    "aten._scaled_dot_product_flash_attention.default",
    "aten._scaled_dot_product_efficient_attention.default",
    "aten.topk.default",
    "aten.softmax.int",
    "aten._softmax.default",
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


def _greedy_unified_cached(
    ex: UnifiedModelExecutor,
    weights: dict[str, Any],
    prompt_ids: torch.Tensor,
    max_new_tokens: int,
    *,
    page_size: int = 16,
) -> list[int]:
    b0 = ex.graph.blocks[0]
    assert isinstance(b0, DenseTransformerBlock)
    kv = PagedKVCache(
        len(ex.graph.blocks),
        b0.num_key_value_heads,
        b0.head_dim,
        page_size=page_size,
        device=str(prompt_ids.device),
        dtype=torch.float16,
    )
    with torch.no_grad():
        logits = ex.forward(weights, input_ids=prompt_ids, kv_cache=kv, pos_offset=0)
        next_tok = int(logits[0, -1].argmax().item())
        out = [next_tok]
        pos = int(prompt_ids.shape[1])
        for _ in range(max_new_tokens - 1):
            inp = torch.tensor([[next_tok]], device=prompt_ids.device, dtype=torch.long)
            step_logits = ex.forward(weights, input_ids=inp, kv_cache=kv, pos_offset=pos)
            next_tok = int(step_logits[0, -1].argmax().item())
            out.append(next_tok)
            pos += 1
    return out


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("m", [1, 512])
def test_fp16_fused_gemm_epilogues_and_swiglu_parity(m: int) -> None:
    """Verify FP16 bias/residual GEMMs and fused SwiGLU match PyTorch FP32-accumulated refs."""
    torch.manual_seed(42 + m)
    d, qkv_n, ffn_n = 896, 1152, 4864
    a = torch.randn((m, d), device="cuda", dtype=torch.float16) * 0.1
    w_qkv = torch.randn((d, qkv_n), device="cuda", dtype=torch.float16) * 0.05
    bias = torch.randn((qkv_n,), device="cuda", dtype=torch.float16) * 0.05
    w_o = torch.randn((d, d), device="cuda", dtype=torch.float16) * 0.05
    residual = torch.randn((m, d), device="cuda", dtype=torch.float16) * 0.1
    w_g = torch.randn((d, ffn_n), device="cuda", dtype=torch.float16) * 0.05
    w_u = torch.randn((d, ffn_n), device="cuda", dtype=torch.float16) * 0.05

    part = torch.empty((4 * m * max(qkv_n, ffn_n),), device="cuda", dtype=torch.float32)
    part2 = torch.empty((4 * m * ffn_n,), device="cuda", dtype=torch.float32)

    out_bias = cutile_gemm_fp16(a, w_qkv, bias=bias, part_buf=part)
    ref_bias = (a.float() @ w_qkv.float() + bias.float().unsqueeze(0)).half()
    assert torch.allclose(out_bias, ref_bias, atol=2e-2, rtol=1e-2)

    out_res = cutile_gemm_fp16(a, w_o, residual=residual, part_buf=part)
    ref_res = (a.float() @ w_o.float() + residual.float()).half()
    assert torch.allclose(out_res, ref_res, atol=2e-2, rtol=1e-2)

    out_swi = fused_swiglu_fp16(a, w_g, w_u, part_g_buf=part, part_u_buf=part2)
    ref_swi = (
        torch.nn.functional.silu(a.float() @ w_g.float()).half() * (a.float() @ w_u.float()).half()
    )
    assert torch.allclose(out_swi, ref_swi, atol=2e-2, rtol=1e-2)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_native_fp16_runtime_synthetic_qwen_parity_and_zero_aten() -> None:
    """Synthetic Qwen2 FP16 graph: exact page-boundary token parity and zero ATen compute ops."""
    graph = build_model_graph(
        "qwen2",
        vocab_size=256,
        hidden_size=256,
        intermediate_size=512,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=64,
    )
    weights = init_synthetic_weights(graph, seed=17, device="cuda", dtype=torch.float16)
    ex = UnifiedModelExecutor(graph)

    torch.manual_seed(17)
    prompt = torch.randint(0, 256, (1, 19), device="cuda", dtype=torch.long)
    ref_tokens = _greedy_unified_cached(ex, weights, prompt, max_new_tokens=32, page_size=16)

    rt = NativeFP16DecoderRuntime(
        graph, weights, max_seq_len=512, block_size=16, use_cuda_graph=True
    )
    cu_tokens, _ = rt.generate(prompt[0].tolist(), max_new_tokens=32, sync_interval=16)
    assert (
        cu_tokens == ref_tokens
    ), f"Token mismatch vs UnifiedModelExecutor: {cu_tokens} != {ref_tokens}"

    rt.reset()
    with _ATenAuditMode() as audit_pre:
        tok0 = rt.prefill(prompt)
    assert (
        audit_pre.forbidden_calls == []
    ), f"Forbidden ATen in prefill: {audit_pre.forbidden_calls}"

    with _ATenAuditMode() as audit_dec:
        tok1 = rt.decode_step(tok0, pos=19, use_graph=False)
    assert audit_dec.forbidden_calls == [], f"Forbidden ATen in decode: {audit_dec.forbidden_calls}"
    assert int(tok0.item()) == ref_tokens[0]
    assert int(tok1.item()) == ref_tokens[1]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_native_fp16_runtime_real_qwen25_05b_parity_and_zero_aten() -> None:
    """Real Qwen/Qwen2.5-0.5B FP16: page-boundary token parity & zero forbidden ATen compute ops."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    repo = "Qwen/Qwen2.5-0.5B"
    tok = AutoTokenizer.from_pretrained(repo, local_files_only=True)
    hf_m: Any = AutoModelForCausalLM.from_pretrained(
        repo, local_files_only=True, torch_dtype=torch.float16
    )
    hf_m = hf_m.cuda().eval()
    graph, weights = extract_hf_decoder_weights(
        hf_m, num_layers=24, device="cuda", dtype=torch.float16
    )
    del hf_m
    torch.cuda.empty_cache()

    ex = UnifiedModelExecutor(graph)
    prompt_text = (
        "Explain the primary technical advantage of paged key-value caching in modern "
        "GPU transformer inference runtimes:"
    )
    prompt_ids = tok.encode(prompt_text, add_special_tokens=False)
    assert len(prompt_ids) >= 18, "Prompt should cross 16-token page boundary during decode/prefill"
    prompt_t = torch.tensor([prompt_ids], device="cuda", dtype=torch.long)

    ref_tokens = _greedy_unified_cached(ex, weights, prompt_t, max_new_tokens=32, page_size=16)

    rt = NativeFP16DecoderRuntime(
        graph, weights, max_seq_len=2048, block_size=16, use_cuda_graph=True
    )
    cu_tokens, stats = rt.generate(prompt_ids, max_new_tokens=32, sync_interval=32)
    assert cu_tokens == ref_tokens, (
        f"Real Qwen2.5-0.5B parity mismatch vs UnifiedModelExecutor cached decode:\n"
        f"  cu ={cu_tokens}\n"
        f"  ref={ref_tokens}"
    )
    assert stats["generated_tokens"] == 32

    rt.reset()
    with _ATenAuditMode() as audit_pre:
        tok0 = rt.prefill(prompt_t)
    assert (
        audit_pre.forbidden_calls == []
    ), f"Forbidden ATen in prefill: {audit_pre.forbidden_calls}"

    with _ATenAuditMode() as audit_dec:
        tok1 = rt.decode_step(tok0, pos=len(prompt_ids), use_graph=False)
    assert audit_dec.forbidden_calls == [], f"Forbidden ATen in decode: {audit_dec.forbidden_calls}"
    assert int(tok0.item()) == ref_tokens[0]
    assert int(tok1.item()) == ref_tokens[1]
