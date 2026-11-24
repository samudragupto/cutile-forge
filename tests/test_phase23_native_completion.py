"""Phase 23 repro tests for native Mixtral MoE dispatch and Qwen/LLaVA cached KV decode."""

from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F
from torch.utils._python_dispatch import TorchDispatchMode

from cutile_forge.engine.config import GPUConfig, ModelConfig
from cutile_forge.engine.kv_cache import PagedKVCache
from cutile_forge.engine.model import Mistral7B
from cutile_forge.engine.weight_loader import QuantLinearWeight, create_random_4bit_weights
from cutile_forge.models import UnifiedModelExecutor, build_model_graph, init_synthetic_weights
from cutile_forge.utils.quantize import quantize_4bit_group

FORBIDDEN_COMPUTE_ATEN = {
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
    "aten.conv2d.default",
    "aten.conv1d.default",
    "aten.convolution.default",
    "aten.topk.default",
    "aten.softmax.int",
    "aten._softmax.default",
    "aten.index_add_.default",
    "aten.index_add.default",
}


class _Phase23Audit(TorchDispatchMode):
    def __init__(self) -> None:
        super().__init__()
        self.forbidden: list[str] = []

    def __torch_dispatch__(
        self,
        func: Any,
        types: tuple[type, ...],
        args: tuple[Any, ...] = (),
        kwargs: dict[str, Any] | None = None,
    ) -> Any:
        name = str(func)
        if name in FORBIDDEN_COMPUTE_ATEN:
            self.forbidden.append(name)
        return func(*args, **(kwargs or {}))


def test_mixtral_layer_zero_forbidden_aten_native_dispatch() -> None:
    """Repro: Mixtral PagedKVCache prefill+decode must emit zero forbidden ATen compute ops."""
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    cfg = ModelConfig(
        vocab_size=256,
        hidden_size=256,
        intermediate_size=512,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=64,
        experts=8,
        top_k_experts=2,
        group_size=128,
    )
    weights = create_random_4bit_weights(cfg, device=dev, seed=42)
    gen = torch.Generator(device="cpu").manual_seed(43)
    dt_fp = torch.float16 if dev == "cuda" else torch.float32
    lw0 = weights.layers[0]
    lw0.moe_gate = (
        torch.randn((cfg.experts, cfg.hidden_size), generator=gen, dtype=torch.float32) * 0.02
    ).to(device=dev, dtype=dt_fp)

    def _rq(k_dim: int, n_dim: int) -> QuantLinearWeight:
        wf = (torch.randn((k_dim, n_dim), generator=gen, dtype=torch.float32) * 0.015).to(
            torch.float16
        )
        qw, s, z = quantize_4bit_group(wf, group_size=cfg.group_size)
        return QuantLinearWeight(qw.to(dev), s.to(dev), z.to(dev), group_size=cfg.group_size)

    lw0.moe_w1 = [_rq(cfg.hidden_size, cfg.intermediate_size) for _ in range(cfg.experts)]
    lw0.moe_w2 = [_rq(cfg.intermediate_size, cfg.hidden_size) for _ in range(cfg.experts)]
    lw0.moe_w3 = [_rq(cfg.hidden_size, cfg.intermediate_size) for _ in range(cfg.experts)]
    model = Mistral7B(weights=weights, gpu_config=GPUConfig(vram_gb=8.0), device=dev)

    kv_cu = PagedKVCache(1, cfg.num_key_value_heads, cfg.head_dim, page_size=16, device=dev)
    kv_pt = PagedKVCache(1, cfg.num_key_value_heads, cfg.head_dim, page_size=16, device=dev)
    ids = torch.randint(0, cfg.vocab_size, (1, 9), device=dev, dtype=torch.long)

    with torch.no_grad():
        if dev == "cuda":
            # Warmup compilation before tracing
            kv_warm = PagedKVCache(
                1, cfg.num_key_value_heads, cfg.head_dim, page_size=16, device=dev
            )
            _ = model.forward(ids[:, :8], kv_cache=kv_warm, pos_offset=0, backend="cutile")
            _ = model.forward(ids[:, 8:9], kv_cache=kv_warm, pos_offset=8, backend="cutile")

            with _Phase23Audit() as tr_prefill:
                cu_prefill = model.forward(
                    ids[:, :8], kv_cache=kv_cu, pos_offset=0, backend="cutile"
                )
            with _Phase23Audit() as tr_decode:
                cu_decode = model.forward(
                    ids[:, 8:9], kv_cache=kv_cu, pos_offset=8, backend="cutile"
                )
            assert (
                tr_prefill.forbidden == []
            ), f"Forbidden ATen in Mixtral prefill: {tr_prefill.forbidden}"
            assert (
                tr_decode.forbidden == []
            ), f"Forbidden ATen in Mixtral decode: {tr_decode.forbidden}"
        else:
            cu_prefill = model.forward(ids[:, :8], kv_cache=kv_cu, pos_offset=0, backend="cutile")
            cu_decode = model.forward(ids[:, 8:9], kv_cache=kv_cu, pos_offset=8, backend="cutile")

        pt_prefill = model.forward(ids[:, :8], kv_cache=kv_pt, pos_offset=0, backend="pytorch")
        pt_decode = model.forward(ids[:, 8:9], kv_cache=kv_pt, pos_offset=8, backend="pytorch")

    cos_p = float(
        F.cosine_similarity(cu_prefill[0, -1].float(), pt_prefill[0, -1].float(), dim=0).item()
    )
    cos_d = float(
        F.cosine_similarity(cu_decode[0, -1].float(), pt_decode[0, -1].float(), dim=0).item()
    )
    assert cos_p >= 0.999, f"Mixtral prefill cosine regression: {cos_p}"
    assert cos_d >= 0.999, f"Mixtral decode cosine regression: {cos_d}"


def test_unified_executor_qwen_cached_decode_native_parity() -> None:
    """Repro: UnifiedModelExecutor Qwen2 supports native PagedKVCache prefill + decode."""
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    dt = torch.float16 if dev == "cuda" else torch.float32
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
    weights = init_synthetic_weights(graph, seed=11, device=dev, dtype=dt)
    ex = UnifiedModelExecutor(graph)

    torch.manual_seed(11)
    ids = torch.randint(0, 256, (1, 8), device=dev, dtype=torch.long)
    kv = PagedKVCache(2, 2, 64, page_size=16, device=dev, dtype=dt)

    with torch.no_grad():
        ref_full = ex.forward(weights, input_ids=ids)
        _ = ex.forward(weights, input_ids=ids[:, :6], kv_cache=kv, pos_offset=0)
        _ = ex.forward(weights, input_ids=ids[:, 6:7], kv_cache=kv, pos_offset=6)
        if dev == "cuda":
            with _Phase23Audit() as tr_dec:
                cu_step = ex.forward(weights, input_ids=ids[:, 7:8], kv_cache=kv, pos_offset=7)
            assert (
                tr_dec.forbidden == []
            ), f"Forbidden ATen in Qwen cached decode: {tr_dec.forbidden}"
        else:
            cu_step = ex.forward(weights, input_ids=ids[:, 7:8], kv_cache=kv, pos_offset=7)

    cos = float(F.cosine_similarity(cu_step[0, -1].float(), ref_full[0, -1].float(), dim=0).item())
    diff = float((cu_step[0, -1].float() - ref_full[0, -1].float()).abs().max().item())
    assert cos >= 0.9999 and diff < 5e-2, f"Qwen cached decode mismatch: cos={cos}, diff={diff}"


def test_unified_executor_llava_cached_decode_native_parity() -> None:
    """Repro: UnifiedModelExecutor LLaVA supports cached text decode without re-running vision."""
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    dt = torch.float16 if dev == "cuda" else torch.float32
    graph = build_model_graph(
        "llava",
        vocab_size=256,
        hidden_size=256,
        intermediate_size=512,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=4,
        head_dim=64,
        vision_hidden_size=128,
        vision_intermediate_size=256,
        vision_num_layers=1,
        vision_num_heads=4,
        image_size=28,
        patch_size=14,
    )
    weights = init_synthetic_weights(graph, seed=19, device=dev, dtype=dt)
    ex = UnifiedModelExecutor(graph)

    torch.manual_seed(19)
    ids = torch.randint(0, 256, (1, 6), device=dev, dtype=torch.long)
    pix = torch.randn((1, 3, 28, 28), device=dev, dtype=dt) * 0.5
    kv = PagedKVCache(2, 4, 64, page_size=16, device=dev, dtype=dt)

    with torch.no_grad():
        ref_full = ex.forward(weights, input_ids=ids, pixel_values=pix)
        _ = ex.forward(weights, input_ids=ids[:, :4], pixel_values=pix, kv_cache=kv, pos_offset=0)
        pos_after_prefill = kv.seq_lens_valid(0)
        _ = ex.forward(
            weights,
            input_ids=ids[:, 4:5],
            pixel_values=None,
            kv_cache=kv,
            pos_offset=pos_after_prefill,
        )
        if dev == "cuda":
            with _Phase23Audit() as tr_dec:
                cu_step = ex.forward(
                    weights,
                    input_ids=ids[:, 5:6],
                    pixel_values=None,
                    kv_cache=kv,
                    pos_offset=pos_after_prefill + 1,
                )
            assert (
                tr_dec.forbidden == []
            ), f"Forbidden ATen in LLaVA cached decode: {tr_dec.forbidden}"
        else:
            cu_step = ex.forward(
                weights,
                input_ids=ids[:, 5:6],
                pixel_values=None,
                kv_cache=kv,
                pos_offset=pos_after_prefill + 1,
            )

    cos = float(F.cosine_similarity(cu_step[0, -1].float(), ref_full[0, -1].float(), dim=0).item())
    diff = float((cu_step[0, -1].float() - ref_full[0, -1].float()).abs().max().item())
    assert cos >= 0.9999 and diff < 5e-2, f"LLaVA cached decode mismatch: cos={cos}, diff={diff}"


def test_unified_executor_encoders_and_dit_zero_forbidden_aten() -> None:
    """Baseline guard: SigLIP, Whisper, and DiT denoiser execute with zero forbidden ATen ops."""
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    dt = torch.float16 if dev == "cuda" else torch.float32
    torch.manual_seed(23)

    specs: list[tuple[str, dict[str, Any], dict[str, Any]]] = [
        (
            "siglip_vision",
            {
                "image_size": 28,
                "patch_size": 14,
                "hidden_size": 128,
                "intermediate_size": 256,
                "num_hidden_layers": 1,
                "num_attention_heads": 4,
                "num_labels": 128,
            },
            {"pixel_values": torch.randn((1, 3, 28, 28), device=dev, dtype=dt) * 0.25},
        ),
        (
            "whisper_encoder",
            {
                "num_mel_bins": 32,
                "max_source_positions": 16,
                "hidden_size": 128,
                "intermediate_size": 256,
                "num_hidden_layers": 1,
                "num_attention_heads": 4,
            },
            {"mel_features": torch.randn((1, 32, 32), device=dev, dtype=dt) * 0.25},
        ),
        (
            "dit",
            {
                "image_size": 8,
                "patch_size": 2,
                "in_channels": 4,
                "out_channels": 4,
                "hidden_size": 128,
                "intermediate_size": 256,
                "num_hidden_layers": 1,
                "num_attention_heads": 4,
            },
            {
                "pixel_values": torch.randn((1, 4, 8, 8), device=dev, dtype=dt) * 0.25,
                "timesteps": torch.tensor([500], device=dev, dtype=torch.long),
                "class_labels": torch.tensor([1], device=dev, dtype=torch.long),
            },
        ),
    ]

    with torch.no_grad():
        for key, overrides, fwd_kwargs in specs:
            g = build_model_graph(key, **overrides)
            w = init_synthetic_weights(g, seed=23, device=dev, dtype=dt)
            ex = UnifiedModelExecutor(g)
            _ = ex.forward(w, **fwd_kwargs)
            if dev == "cuda":
                with _Phase23Audit() as tr:
                    out = ex.forward(w, **fwd_kwargs)
                assert tr.forbidden == [], f"Forbidden ATen in {key}: {tr.forbidden}"
            else:
                out = ex.forward(w, **fwd_kwargs)
            assert torch.isfinite(out).all()
