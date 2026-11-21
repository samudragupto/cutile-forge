"""Phase 15 failing reproduction tests for sliding-window attention & FP8 native parity."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F

from cutile_forge.engine.config import ModelConfig
from cutile_forge.engine.kv_cache import PagedKVCache
from cutile_forge.engine.weight_loader import create_random_4bit_weights
from cutile_forge.kernels.attention_gqa import (
    flash_decoding_split_kv,
    fused_attention_gqa,
    fused_attention_gqa_decode_pos,
)
from cutile_forge.kernels.universal_quant import QuantFormat, UniversalQuantLinear
from cutile_forge.runtime.native_engine import NativeModelRuntime


def _sdpa_sliding_window_ref(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    sliding_window: int,
) -> torch.Tensor:
    """Exact HF & Mistral sliding-window causal attention reference."""
    bsz, h_q, s_q, d_head = q.shape
    _, h_kv, s_kv, _ = k.shape
    kv_offset = s_kv - s_q
    q_pos = torch.arange(s_q, device=q.device).view(s_q, 1) + kv_offset
    k_pos = torch.arange(s_kv, device=q.device).view(1, s_kv)
    mask = (k_pos <= q_pos) & (k_pos > q_pos - sliding_window)
    kr = k.repeat_interleave(h_q // h_kv, dim=1)
    vr = v.repeat_interleave(h_q // h_kv, dim=1)
    return F.scaled_dot_product_attention(q, kr, vr, attn_mask=mask)


def test_model_config_parses_sliding_window(tmp_path: Path) -> None:
    cfg_file = tmp_path / "config.json"
    cfg_file.write_text(
        json.dumps(
            {
                "vocab_size": 32000,
                "hidden_size": 256,
                "intermediate_size": 512,
                "num_hidden_layers": 2,
                "num_attention_heads": 8,
                "num_key_value_heads": 2,
                "sliding_window": 64,
            }
        ),
        encoding="utf-8",
    )
    cfg = ModelConfig.from_json(cfg_file)
    assert getattr(cfg, "sliding_window", None) == 64


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_sliding_window_attention_kernels_above_window() -> None:
    torch.manual_seed(7)
    bsz, h_q, h_kv, s_kv, d = 1, 8, 2, 256, 64
    sw = 64

    # 1. Prefill / multi-query causal GQA kernel above sliding window
    q_pre = torch.randn((bsz, h_q, s_kv, d), device="cuda", dtype=torch.float16)
    k = torch.randn((bsz, h_kv, s_kv, d), device="cuda", dtype=torch.float16) * 1.5
    v = torch.randn((bsz, h_kv, s_kv, d), device="cuda", dtype=torch.float16) * 2.0

    ref_pre = _sdpa_sliding_window_ref(q_pre, k, v, sliding_window=sw)
    out_pre = fused_attention_gqa(q_pre, k, v, causal=True, sliding_window=sw)
    assert (out_pre - ref_pre).abs().max().item() < 5e-3

    # 2. FlashDecoding Split-KV kernel above sliding window
    q_dec = torch.randn((bsz, h_q, 1, d), device="cuda", dtype=torch.float16)
    ref_dec = _sdpa_sliding_window_ref(q_dec, k, v, sliding_window=sw)
    out_split = flash_decoding_split_kv(q_dec, k, v, num_splits=4, causal=True, sliding_window=sw)
    assert (out_split - ref_dec).abs().max().item() < 5e-3

    # 3. CUDA-Graph decode_pos kernel above sliding window
    pos_buf = torch.tensor([s_kv - 1], device="cuda", dtype=torch.int32)
    out_pos = torch.empty_like(q_dec)
    fused_attention_gqa_decode_pos(
        q_dec,
        k,
        v,
        pos_ptr=pos_buf,
        max_kv_len=s_kv,
        out=out_pos,
        dynamic_loop=True,
        sliding_window=sw,
    )
    assert (out_pos - ref_dec).abs().max().item() < 5e-3


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_native_runtime_sliding_window_prefill_decode_batch() -> None:
    from cutile_forge.engine.model import Mistral7B

    torch.manual_seed(11)
    cfg = ModelConfig(
        vocab_size=256,
        hidden_size=128,
        intermediate_size=256,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=32,
        max_position_embeddings=256,
        group_size=64,
        sliding_window=32,
    )
    weights = create_random_4bit_weights(cfg, device="cuda")
    model = Mistral7B(weights=weights, device="cuda")

    rt_graph = NativeModelRuntime(
        model, max_seq_len=256, kv_dtype="fp16", use_cuda_graph=True, max_batch_size=2
    )
    rt_eager = NativeModelRuntime(
        model, max_seq_len=256, kv_dtype="fp16", use_cuda_graph=False, max_batch_size=2
    )

    prompt = list(range(1, 80))  # > sliding_window (32)
    # Chunked prefill + CUDA Graph decode must match non-graphed runtime and PyTorch ref
    rt_eager.reset()
    t_eager = rt_eager.prefill(torch.tensor([prompt], device="cuda"))
    seq_eager = [int(t_eager.item())]
    cur_e = t_eager
    for step in range(8):
        cur_e = rt_eager.decode_step(cur_e, pos=len(prompt) + step, use_graph=False)
        seq_eager.append(int(cur_e.item()))

    rt_graph.reset()
    t_chunk = None
    for s in range(0, len(prompt), 32):
        e = min(len(prompt), s + 32)
        t_chunk = rt_graph.prefill_chunk(
            torch.tensor([prompt[s:e]], device="cuda"),
            start_pos=s,
            emit_next_token=(e == len(prompt)),
        )
    assert t_chunk is not None
    seq_graph = [int(t_chunk.item())]
    cur_g = t_chunk
    for step in range(8):
        cur_g = rt_graph.decode_step(cur_g, pos=len(prompt) + step, use_graph=True)
        seq_graph.append(int(cur_g.item()))
    assert seq_graph == seq_eager

    # Also verify batched B=2 matches single-stream output above window
    outs_b, _ = rt_graph.generate_batch([prompt, prompt], max_new_tokens=9, sync_interval=4)
    assert outs_b[0] == seq_eager
    assert outs_b[1] == seq_eager


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_paged_kv_sliding_window_eviction_parity() -> None:
    torch.manual_seed(17)
    bsz, h_q, h_kv, d = 1, 4, 2, 64
    sw = 32
    kv_full = PagedKVCache(num_layers=1, num_kv_heads=h_kv, head_dim=d, page_size=16, device="cuda")
    kv_sw = PagedKVCache(
        num_layers=1,
        num_kv_heads=h_kv,
        head_dim=d,
        page_size=16,
        device="cuda",
        sliding_window=sw,
    )
    k = torch.randn((bsz, h_kv, 96, d), device="cuda", dtype=torch.float16)
    v = torch.randn((bsz, h_kv, 96, d), device="cuda", dtype=torch.float16)
    kv_full.update(0, k, v)
    kv_sw.update(0, k, v)

    k_f, v_f = kv_full.get_kv(0)
    k_w, v_w = kv_sw.get_kv(0)
    assert (
        kv_sw.seq_lens_valid() <= sw + kv_sw.page_size if hasattr(kv_sw, "seq_lens_valid") else True
    )
    q = torch.randn((bsz, h_q, 1, d), device="cuda", dtype=torch.float16)
    ref = _sdpa_sliding_window_ref(q, k_f, v_f, sliding_window=sw)
    out = fused_attention_gqa(q, k_w, v_w, causal=True, sliding_window=sw)
    assert (out - ref).abs().max().item() < 5e-3


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_fp8_tensor_native_vs_reference_parity() -> None:
    torch.manual_seed(23)
    k, n = 256, 256
    w = torch.randn((k, n), device="cuda", dtype=torch.float16) * 0.05
    uql = UniversalQuantLinear.from_fp16(w, fmt=QuantFormat.FP8)

    # Activation tensor with realistic LLM outlier channel (causes severe E4M3 quantization loss if x is quantized per-tensor)
    x = torch.randn((4, k), device="cuda", dtype=torch.float16) * 0.25
    x[0, 0] = 96.0
    x[1:, :] *= 0.05

    ref = uql.forward_reference(x)
    nat = uql.forward_native(x)
    cos = F.cosine_similarity(nat.float().flatten(), ref.float().flatten(), dim=0).item()
    max_diff = (nat.float() - ref.float()).abs().max().item()
    assert cos >= 0.9999, f"FP8 native vs reference cosine degraded: {cos}"
    assert max_diff < 5e-2, f"FP8 native vs reference max_diff too high: {max_diff}"


def test_weight_streamer_multistream_no_corruption() -> None:
    """Reproduce WeightStreamer caching-allocator race across copy_stream and compute_stream."""
    if not torch.cuda.is_available():
        return

    from cutile_forge.forge.weight_streamer import WeightStreamer

    streamer = WeightStreamer(num_slots=2, device="cuda", model_gb=40.0, gpu_vram_gb=32.0)
    dim = 1024
    w_good = WeightStreamer.pin_tensor(torch.eye(dim, dtype=torch.float16))
    w_nan = WeightStreamer.pin_tensor(torch.full((dim, dim), float("nan"), dtype=torch.float16))
    busy = torch.randn((4096, 4096), device="cuda", dtype=torch.float16) * 0.01
    x = torch.ones((64, dim), device="cuda", dtype=torch.float16)

    outputs: list[torch.Tensor] = []
    torch.cuda.synchronize()
    for step in range(4):
        slot = streamer.acquire_layer(step, {"w": w_good})
        w_active = slot["w"]
        del slot
        z = busy
        for _ in range(15):
            z = z @ busy
        y = x @ w_active
        del w_active
        outputs.append(y)
        streamer.release_layer(step)
        _ = streamer.prefetch_layer(step + 10, {"w": w_nan})

    torch.cuda.synchronize()
    for step, y in enumerate(outputs):
        assert torch.isfinite(y).all(), f"Step {step} corrupted by copy_stream overwrite"
        assert torch.allclose(y, x, atol=1e-3, rtol=1e-3)
