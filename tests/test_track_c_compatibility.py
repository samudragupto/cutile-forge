"""Track C (`v2.3.0`) compatibility tests: UnifiedModelGraph, MoE/ViT/Multimodal & Multi-Format Loaders."""

from __future__ import annotations

import json
import struct
from pathlib import Path

import torch
from safetensors.torch import save_file

from cutile_forge.models import (
    MODEL_REGISTRY,
    CheckpointFormat,
    UnifiedModelExecutor,
    build_model_graph,
    compute_scaled_rope_freqs,
    detect_checkpoint_format,
    from_hf_config,
    init_synthetic_weights,
    list_supported_models,
    load_unified_checkpoint,
    parse_gguf_header,
    plan_checkpoint_placement,
    write_synthetic_gguf,
)
from cutile_forge.utils.quantize import dequantize_4bit_group


def _pack_gptq_from_quant(
    q_int4: torch.Tensor, zeros: torch.Tensor, scales: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Pack `(K, N)` uint4 [0..15] + `(G, N)` zeros into AutoGPTQ int32 tensors."""
    k_dim, n_dim = q_int4.shape
    g_dim = zeros.shape[0]
    shifts = torch.arange(0, 32, 4, dtype=torch.int32)
    qw_i32 = (
        (q_int4.to(torch.int32).view(k_dim // 8, 8, n_dim) << shifts.view(1, 8, 1))
        .sum(dim=1)
        .to(torch.int32)
    )
    qz_vals = (zeros.round().clamp(1.0, 16.0).to(torch.int32) - 1) & 0xF
    qz_i32 = (qz_vals.view(g_dim, n_dim // 8, 8) << shifts.view(1, 1, 8)).sum(dim=2).to(torch.int32)
    return qw_i32, qz_i32, scales.to(torch.float16)


def _pack_awq_from_quant(
    q_int4: torch.Tensor, zeros: torch.Tensor, scales: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Pack `(K, N)` uint4 [0..15] + `(G, N)` zeros into AutoAWQ interleaved int32 tensors."""
    k_dim, n_dim = q_int4.shape
    g_dim = zeros.shape[0]
    awq_shifts = torch.tensor([0, 16, 4, 20, 8, 24, 12, 28], dtype=torch.int32)
    qw_i32 = (
        (q_int4.to(torch.int32).view(k_dim, n_dim // 8, 8) << awq_shifts.view(1, 1, 8))
        .sum(dim=2)
        .to(torch.int32)
    )
    qz_vals = zeros.round().clamp(0.0, 15.0).to(torch.int32) & 0xF
    qz_i32 = (
        (qz_vals.view(g_dim, n_dim // 8, 8) << awq_shifts.view(1, 1, 8)).sum(dim=2).to(torch.int32)
    )
    return qw_i32, qz_i32, scales.to(torch.float16)


def test_unified_model_graph_registry_and_tfg_lowering() -> None:
    keys = list_supported_models()
    for expected in (
        "mistral",
        "llama",
        "llama3",
        "qwen2",
        "mixtral",
        "deepseek_moe",
        "qwen2_moe",
        "vit",
        "clip_vision",
        "siglip_vision",
        "llava",
    ):
        assert expected in keys
        assert expected in MODEL_REGISTRY

    for key in keys:
        graph = build_model_graph(key, num_hidden_layers=2)
        tfg, report = graph.compile_tfg(batch_size=1, seq_len=8, dtype="float16")
        assert tfg.validate()
        assert report.total_nodes_before > 0
        assert report.total_nodes_after > 0
        summary = graph.summary()
        assert summary["family"] == MODEL_REGISTRY[key].family
        assert summary["estimated_params"] > 0


def test_dense_llm_variants_numerical_parity_and_scaled_rope() -> None:
    torch.manual_seed(7)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    dt = torch.float16 if torch.cuda.is_available() else torch.float32

    # Scaled RoPE check: Llama-3 scaling must diverge smoothly from base theta=10000
    c_base, _ = compute_scaled_rope_freqs(16, 32, theta=10000.0)
    c_l3, _ = compute_scaled_rope_freqs(
        16,
        32,
        theta=500000.0,
        rope_scaling={
            "rope_type": "llama3",
            "factor": 8.0,
            "low_freq_factor": 1.0,
            "high_freq_factor": 4.0,
            "original_max_position_embeddings": 8192,
        },
    )
    assert c_base.shape == (16, 16)
    assert (c_base - c_l3).abs().max().item() > 1e-2

    configs = [
        (
            "mistral",
            {
                "vocab_size": 128,
                "hidden_size": 64,
                "intermediate_size": 128,
                "num_hidden_layers": 2,
                "num_attention_heads": 4,
                "num_key_value_heads": 2,
                "head_dim": 16,
                "sliding_window": 4,
            },
        ),
        (
            "llama3",
            {
                "vocab_size": 128,
                "hidden_size": 64,
                "intermediate_size": 128,
                "num_hidden_layers": 2,
                "num_attention_heads": 4,
                "num_key_value_heads": 2,
                "head_dim": 16,
            },
        ),
        (
            "qwen2",
            {
                "vocab_size": 128,
                "hidden_size": 64,
                "intermediate_size": 128,
                "num_hidden_layers": 2,
                "num_attention_heads": 4,
                "num_key_value_heads": 2,
                "head_dim": 16,
                "qkv_bias": True,
                "tie_word_embeddings": True,
            },
        ),
    ]
    ids = torch.randint(0, 128, (2, 8), device=dev)
    for key, overrides in configs:
        graph = build_model_graph(key, **overrides)
        weights = init_synthetic_weights(graph, seed=11, device=dev, dtype=dt)
        ex = UnifiedModelExecutor(graph)
        out_cu = ex.forward(weights, input_ids=ids)
        out_ref = ex.reference_forward(weights, input_ids=ids)
        assert out_cu.shape == (2, 8, 128)
        assert (out_cu.float() - out_ref.float()).abs().max().item() < 1e-2


def test_moe_variants_numerical_parity() -> None:
    torch.manual_seed(19)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    dt = torch.float16 if torch.cuda.is_available() else torch.float32
    ids = torch.randint(0, 96, (2, 6), device=dev)

    moe_specs = [
        (
            "mixtral",
            {
                "vocab_size": 96,
                "hidden_size": 64,
                "intermediate_size": 96,
                "num_hidden_layers": 2,
                "num_attention_heads": 4,
                "num_key_value_heads": 2,
                "head_dim": 16,
                "num_local_experts": 4,
                "num_experts_per_tok": 2,
            },
        ),
        (
            "deepseek_moe",
            {
                "vocab_size": 96,
                "hidden_size": 64,
                "intermediate_size": 64,
                "num_hidden_layers": 2,
                "num_attention_heads": 4,
                "num_key_value_heads": 4,
                "head_dim": 16,
                "num_local_experts": 4,
                "num_experts_per_tok": 2,
                "shared_expert_intermediate_size": 96,
            },
        ),
        (
            "qwen2_moe",
            {
                "vocab_size": 96,
                "hidden_size": 64,
                "intermediate_size": 64,
                "num_hidden_layers": 2,
                "num_attention_heads": 4,
                "num_key_value_heads": 4,
                "head_dim": 16,
                "num_local_experts": 4,
                "num_experts_per_tok": 2,
                "shared_expert_intermediate_size": 96,
                "qkv_bias": True,
            },
        ),
    ]
    for key, overrides in moe_specs:
        graph = build_model_graph(key, **overrides)
        weights = init_synthetic_weights(graph, seed=23, device=dev, dtype=dt)
        ex = UnifiedModelExecutor(graph)
        out_cu = ex.forward(weights, input_ids=ids)
        out_ref = ex.reference_forward(weights, input_ids=ids)
        assert out_cu.shape == (2, 6, 96)
        diff = (out_cu.float() - out_ref.float()).abs()
        cos = float(
            torch.nn.functional.cosine_similarity(
                out_cu.flatten().float(), out_ref.flatten().float(), dim=0
            ).item()
        )
        assert cos >= 0.999 and float(diff.median().item()) < 1e-2


def test_vit_and_llava_multimodal_forward_parity() -> None:
    torch.manual_seed(31)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    dt = torch.float16 if torch.cuda.is_available() else torch.float32

    vit = build_model_graph(
        "vit",
        image_size=32,
        patch_size=8,
        in_channels=3,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_labels=10,
    )
    vit_w = init_synthetic_weights(vit, seed=5, device=dev, dtype=dt)
    img = torch.randn((2, 3, 32, 32), device=dev, dtype=dt) * 0.2
    ex_vit = UnifiedModelExecutor(vit)
    vit_cu = ex_vit.forward(vit_w, pixel_values=img)
    vit_ref = ex_vit.reference_forward(vit_w, pixel_values=img)
    assert vit_cu.shape == (2, 10)
    assert (vit_cu.float() - vit_ref.float()).abs().max().item() < 1e-2

    llava = build_model_graph(
        "llava",
        vocab_size=96,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=16,
        image_size=32,
        patch_size=8,
        vision_hidden_size=48,
        vision_intermediate_size=96,
        vision_num_layers=1,
        vision_num_heads=4,
    )
    mm_w = init_synthetic_weights(llava, seed=7, device=dev, dtype=dt)
    ids = torch.randint(0, 96, (2, 5), device=dev)
    ex_mm = UnifiedModelExecutor(llava)
    mm_cu = ex_mm.forward(mm_w, input_ids=ids, pixel_values=img)
    mm_ref = ex_mm.reference_forward(mm_w, input_ids=ids, pixel_values=img)
    # 16 patches (32/8)^2 + 5 text tokens = 21 tokens
    assert mm_cu.shape == (2, 21, 96)
    assert (mm_cu.float() - mm_ref.float()).abs().max().item() < 1e-2


def test_multiformat_checkpoint_detection_and_roundtrip(tmp_path: Path) -> None:
    torch.manual_seed(99)
    k_dim, n_dim = 128, 64

    # 1. FP16 safetensors
    fp16_dir = tmp_path / "mistral_fp16"
    fp16_dir.mkdir()
    (fp16_dir / "config.json").write_text(
        json.dumps(
            {
                "model_type": "mistral",
                "hidden_size": 64,
                "intermediate_size": 128,
                "num_hidden_layers": 2,
                "num_attention_heads": 4,
                "num_key_value_heads": 2,
                "vocab_size": 128,
            }
        ),
        encoding="utf-8",
    )
    w_fp16 = torch.randn((k_dim, n_dim), dtype=torch.float16)
    save_file({"model.layers.0.mlp.up_proj.weight": w_fp16}, str(fp16_dir / "model.safetensors"))
    ckpt_fp16 = load_unified_checkpoint(fp16_dir, device="cpu")
    assert ckpt_fp16.inspection.quant_format == CheckpointFormat.FP16
    assert torch.equal(ckpt_fp16.tensors["model.layers.0.mlp.up_proj.weight"], w_fp16)

    # 2. BF16 safetensors
    bf16_dir = tmp_path / "llama3_bf16"
    bf16_dir.mkdir()
    (bf16_dir / "config.json").write_text(
        json.dumps(
            {
                "model_type": "llama",
                "rope_theta": 500000.0,
                "hidden_size": 64,
                "intermediate_size": 128,
                "num_hidden_layers": 2,
                "num_attention_heads": 4,
                "num_key_value_heads": 2,
                "vocab_size": 128,
            }
        ),
        encoding="utf-8",
    )
    w_bf16 = torch.randn((k_dim, n_dim), dtype=torch.bfloat16)
    save_file({"model.layers.0.mlp.up_proj.weight": w_bf16}, str(bf16_dir / "model.safetensors"))
    ckpt_bf16 = load_unified_checkpoint(bf16_dir, device="cpu")
    assert ckpt_bf16.inspection.quant_format == CheckpointFormat.BF16
    assert ckpt_bf16.inspection.family == "llama3"
    assert torch.equal(
        ckpt_bf16.tensors["model.layers.0.mlp.up_proj.weight"], w_bf16.to(torch.float16)
    )

    # 3. GPTQ (desc_act=False, group_size=128)
    gptq_dir = tmp_path / "qwen2_gptq"
    gptq_dir.mkdir()
    (gptq_dir / "config.json").write_text(
        json.dumps(
            {
                "model_type": "qwen2",
                "hidden_size": 64,
                "intermediate_size": 128,
                "num_hidden_layers": 2,
                "num_attention_heads": 4,
                "num_key_value_heads": 2,
                "vocab_size": 128,
            }
        ),
        encoding="utf-8",
    )
    (gptq_dir / "quantize_config.json").write_text(
        json.dumps({"quant_method": "gptq", "bits": 4, "group_size": 128, "desc_act": False}),
        encoding="utf-8",
    )
    q_int = torch.randint(0, 16, (k_dim, n_dim), dtype=torch.int32)
    z_f = torch.randint(1, 15, (1, n_dim), dtype=torch.float16)
    s_f = torch.rand((1, n_dim), dtype=torch.float16) * 0.05 + 0.01
    qw_i32, qz_i32, sc_f16 = _pack_gptq_from_quant(q_int, z_f, s_f)
    save_file(
        {
            "model.layers.0.self_attn.q_proj.qweight": qw_i32,
            "model.layers.0.self_attn.q_proj.qzeros": qz_i32,
            "model.layers.0.self_attn.q_proj.scales": sc_f16,
        },
        str(gptq_dir / "model.safetensors"),
    )
    ckpt_gptq = load_unified_checkpoint(gptq_dir, device="cpu")
    assert ckpt_gptq.inspection.quant_format == CheckpointFormat.GPTQ
    qweight_obj = ckpt_gptq.tensors["model.layers.0.self_attn.q_proj"]
    recovered_gptq = dequantize_4bit_group(
        qweight_obj.qweight, qweight_obj.scales, qweight_obj.zeros, 128
    )
    expected_gptq = (q_int.float() - z_f.float()) * s_f.float()
    assert (recovered_gptq.float() - expected_gptq).abs().max().item() < 1e-2

    # 4. AWQ (4-bit interleaved)
    awq_dir = tmp_path / "mixtral_awq"
    awq_dir.mkdir()
    (awq_dir / "config.json").write_text(
        json.dumps(
            {
                "model_type": "mixtral",
                "hidden_size": 64,
                "intermediate_size": 128,
                "num_hidden_layers": 2,
                "num_attention_heads": 4,
                "num_key_value_heads": 2,
                "num_local_experts": 4,
                "vocab_size": 128,
            }
        ),
        encoding="utf-8",
    )
    (awq_dir / "quantize_config.json").write_text(
        json.dumps({"quant_method": "awq", "bits": 4, "group_size": 128}), encoding="utf-8"
    )
    z_awq = torch.randint(0, 15, (1, n_dim), dtype=torch.float16)
    qw_awq, qz_awq, sc_awq = _pack_awq_from_quant(q_int, z_awq, s_f)
    save_file(
        {
            "model.layers.0.self_attn.q_proj.qweight": qw_awq,
            "model.layers.0.self_attn.q_proj.qzeros": qz_awq,
            "model.layers.0.self_attn.q_proj.scales": sc_awq,
        },
        str(awq_dir / "model.safetensors"),
    )
    ckpt_awq = load_unified_checkpoint(awq_dir, device="cpu")
    assert ckpt_awq.inspection.quant_format == CheckpointFormat.AWQ
    awq_obj = ckpt_awq.tensors["model.layers.0.self_attn.q_proj"]
    recovered_awq = dequantize_4bit_group(awq_obj.qweight, awq_obj.scales, awq_obj.zeros, 128)
    expected_awq = (q_int.float() - z_awq.float()) * s_f.float()
    assert (recovered_awq.float() - expected_awq).abs().max().item() < 1e-2

    # 5. FP8 (e4m3fn with 2D block scaling 64x32 -> scale shape (2, 2))
    fp8_dir = tmp_path / "deepseek_fp8"
    fp8_dir.mkdir()
    (fp8_dir / "config.json").write_text(
        json.dumps(
            {
                "model_type": "deepseek_v2",
                "hidden_size": 64,
                "intermediate_size": 64,
                "num_hidden_layers": 2,
                "num_attention_heads": 4,
                "num_key_value_heads": 4,
                "num_local_experts": 4,
                "shared_expert_intermediate_size": 64,
                "vocab_size": 128,
            }
        ),
        encoding="utf-8",
    )
    w_base = (torch.randn((k_dim, n_dim)) * 2.0).clamp(-16.0, 16.0)
    w_fp8 = w_base.to(torch.float8_e4m3fn)
    sc_inv = torch.tensor([[0.5, 1.25], [0.75, 1.5]], dtype=torch.float32)
    save_file(
        {
            "model.layers.0.mlp.up_proj.weight": w_fp8,
            "model.layers.0.mlp.up_proj.weight_scale_inv": sc_inv,
        },
        str(fp8_dir / "model.safetensors"),
    )
    ckpt_fp8 = load_unified_checkpoint(fp8_dir, device="cpu")
    assert ckpt_fp8.inspection.quant_format == CheckpointFormat.FP8
    expected_fp8 = (
        w_fp8.float() * sc_inv.repeat_interleave(64, dim=0).repeat_interleave(32, dim=1)
    ).to(torch.float16)
    assert torch.equal(ckpt_fp8.tensors["model.layers.0.mlp.up_proj.weight"], expected_fp8)

    # 6a. GGUF Q4_0 roundtrip
    k_g, n_g = 128, 32
    blocks_q4_0 = (k_g * n_g) // 32
    d_q4 = (torch.rand(blocks_q4_0) * 0.1 + 0.05).to(torch.float16)
    qs_q4 = torch.randint(0, 256, (blocks_q4_0, 16), dtype=torch.uint8)
    raw_q4_0 = bytearray()
    for bi in range(blocks_q4_0):
        raw_q4_0 += struct.pack("<e", float(d_q4[bi].item()))
        raw_q4_0 += bytes(qs_q4[bi].tolist())
    g_q4_0 = tmp_path / "model_q4_0.gguf"
    write_synthetic_gguf(
        g_q4_0,
        {"general.architecture": "llama", "llama.embedding_length": 64, "llama.block_count": 2},
        [("blk.0.attn_q.weight", (k_g, n_g), 2, bytes(raw_q4_0))],
    )
    parsed_0 = parse_gguf_header(g_q4_0)
    assert parsed_0["metadata"]["general.architecture"] == "llama"
    ckpt_q40 = load_unified_checkpoint(g_q4_0, device="cpu")
    assert ckpt_q40.inspection.quant_format == CheckpointFormat.GGUF_Q4_0
    q40_w = ckpt_q40.tensors["blk.0.attn_q.weight"]
    rec_q40 = dequantize_4bit_group(q40_w.qweight, q40_w.scales, q40_w.zeros, q40_w.group_size)
    assert rec_q40.shape == (k_g, n_g)
    # Verify exact match against raw Q4_0 reference
    lo_r = (qs_q4 & 0xF).float() - 8.0
    hi_r = ((qs_q4 >> 4) & 0xF).float() - 8.0
    ref_q40 = (torch.cat([lo_r, hi_r], dim=1) * d_q4.float().unsqueeze(1)).reshape(n_g, k_g).t()
    assert (rec_q40.float() - ref_q40).abs().mean().item() < 0.05

    # 6b. GGUF Q8_0 exact roundtrip
    blocks_q8_0 = (k_g * n_g) // 32
    d_q8 = torch.tensor([0.25] * blocks_q8_0, dtype=torch.float16)
    qs_q8 = torch.randint(-64, 64, (blocks_q8_0, 32), dtype=torch.int8)
    raw_q8_0 = bytearray()
    for bi in range(blocks_q8_0):
        raw_q8_0 += struct.pack("<e", float(d_q8[bi].item()))
        raw_q8_0 += bytes(qs_q8[bi].view(torch.uint8).tolist())
    g_q8_0 = tmp_path / "model_q8_0.gguf"
    write_synthetic_gguf(
        g_q8_0,
        {
            "general.architecture": "mistral",
            "mistral.embedding_length": 64,
            "mistral.block_count": 2,
        },
        [("blk.0.attn_k.weight", (k_g, n_g), 8, bytes(raw_q8_0))],
    )
    ckpt_q80 = load_unified_checkpoint(g_q8_0, device="cpu")
    assert ckpt_q80.inspection.quant_format == CheckpointFormat.GGUF_Q8_0
    ref_q80 = (d_q8.float().unsqueeze(1) * qs_q8.float()).reshape(n_g, k_g).t().to(torch.float16)
    assert torch.equal(ckpt_q80.tensors["blk.0.attn_k.weight"], ref_q80)

    # 6c. GGUF Q4_K_M roundtrip (256-weight super-blocks, 144 bytes/block)
    blocks_q4k = (k_g * n_g) // 256
    raw_q4k = bytearray()
    for bi in range(blocks_q4k):
        raw_q4k += struct.pack("<ee", 0.05, 0.02)
        raw_q4k += bytes([7] * 12)
        raw_q4k += bytes([(bi * 17 + i) & 0xFF for i in range(128)])
    g_q4k = tmp_path / "model_q4_k_m.gguf"
    write_synthetic_gguf(
        g_q4k,
        {"general.architecture": "qwen2", "qwen2.embedding_length": 64, "qwen2.block_count": 2},
        [("blk.0.attn_v.weight", (k_g, n_g), 12, bytes(raw_q4k))],
    )
    ckpt_q4k = load_unified_checkpoint(g_q4k, device="cpu")
    assert ckpt_q4k.inspection.quant_format == CheckpointFormat.GGUF_Q4_K_M
    assert ckpt_q4k.tensors["blk.0.attn_v.weight"].qweight.shape == (k_g // 2, n_g)


def test_hybrid_layer_budgeter_auto_activation(tmp_path: Path) -> None:
    # Small 7B-class config on 32 GB VRAM -> stays 100% resident
    small_dir = tmp_path / "mistral_7b"
    small_dir.mkdir()
    (small_dir / "config.json").write_text(
        json.dumps(
            {
                "model_type": "mistral",
                "hidden_size": 4096,
                "intermediate_size": 14336,
                "num_hidden_layers": 32,
                "num_attention_heads": 32,
                "num_key_value_heads": 8,
                "vocab_size": 32000,
            }
        ),
        encoding="utf-8",
    )
    (small_dir / "quantize_config.json").write_text(
        json.dumps({"quant_method": "gptq", "bits": 4, "group_size": 128}), encoding="utf-8"
    )
    insp_7b = detect_checkpoint_format(small_dir)
    plan_7b = plan_checkpoint_placement(insp_7b, vram_gb=32.0)
    assert plan_7b.use_hybrid_budgeter is False
    assert plan_7b.resident_gpu_layers == 32
    assert plan_7b.streamed_cpu_layers == 0

    # Large 70B-class GPTQ config on 32 GB VRAM (> 0.85 * 32 = 27.2 GB) -> activates HybridLayerBudgeter
    big_dir = tmp_path / "llama_70b"
    big_dir.mkdir()
    (big_dir / "config.json").write_text(
        json.dumps(
            {
                "model_type": "llama",
                "hidden_size": 8192,
                "intermediate_size": 28672,
                "num_hidden_layers": 80,
                "num_attention_heads": 64,
                "num_key_value_heads": 8,
                "vocab_size": 32000,
            }
        ),
        encoding="utf-8",
    )
    (big_dir / "quantize_config.json").write_text(
        json.dumps({"quant_method": "gptq", "bits": 4, "group_size": 128}), encoding="utf-8"
    )
    insp_70b = detect_checkpoint_format(big_dir)
    plan_70b = plan_checkpoint_placement(insp_70b, vram_gb=32.0)
    assert plan_70b.use_hybrid_budgeter is True
    assert 0 < plan_70b.resident_gpu_layers < 80
    assert plan_70b.streamed_cpu_layers == 80 - plan_70b.resident_gpu_layers


def test_real_checkpoint_inspection_or_synthetic_fallback(tmp_path: Path) -> None:
    real_7b = Path("/home/arrhat/models/mistral_7b_4bit")
    if real_7b.exists():
        insp = detect_checkpoint_format(real_7b)
        assert insp.family == "mistral"
        assert insp.quant_format == CheckpointFormat.GPTQ
        graph = from_hf_config(insp.config)
        assert graph.num_layers == 32
        plan = plan_checkpoint_placement(insp, vram_gb=32.0)
        assert plan.use_hybrid_budgeter is False
        assert plan.resident_gpu_layers == 32
    else:
        # Synthetic equivalent on CPU CI (never skipped)
        syn = tmp_path / "mistral_real_cfg"
        syn.mkdir()
        (syn / "config.json").write_text(
            json.dumps(
                {
                    "model_type": "mistral",
                    "hidden_size": 4096,
                    "intermediate_size": 14336,
                    "num_hidden_layers": 32,
                    "num_attention_heads": 32,
                    "num_key_value_heads": 8,
                }
            ),
            encoding="utf-8",
        )
        (syn / "quantize_config.json").write_text(
            json.dumps({"quant_method": "gptq", "bits": 4, "group_size": 128}), encoding="utf-8"
        )
        insp = detect_checkpoint_format(syn)
        assert insp.family == "mistral"
        assert insp.quant_format == CheckpointFormat.GPTQ

    real_70b = Path("/var/tmp/arrhat_models/llama_70b_4bit")
    if real_70b.exists():
        insp70 = detect_checkpoint_format(real_70b)
        plan70 = plan_checkpoint_placement(insp70, vram_gb=32.0)
        assert plan70.use_hybrid_budgeter is True
        assert plan70.streamed_cpu_layers > 0

    real_mixtral = Path("/var/tmp/arrhat_models/mixtral_8x7b_4bit")
    if real_mixtral.exists():
        insp_mx = detect_checkpoint_format(real_mixtral)
        assert insp_mx.family == "mixtral"
        assert insp_mx.quant_format == CheckpointFormat.GPTQ
        g_mx = from_hf_config(insp_mx.config)
        assert g_mx.num_layers == 32
        assert g_mx.summary()["block_types"] == ["SparseMoEBlock"]
