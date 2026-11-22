"""Phase 11 (`v2.6.0`) tests: Architecture Completeness (Llama, Qwen2.5, Mistral, MoE, Vision, Audio, LLaVA, Diffusion).

Zero `pytest.skip` policy: every test runs end-to-end on both CUDA (`sm_120`) and CPU CI.
"""

from __future__ import annotations

import torch
from transformers import (
    LlamaConfig,
    LlamaForCausalLM,
    MistralConfig,
    MistralForCausalLM,
    MixtralConfig,
    MixtralForCausalLM,
    Qwen2Config,
    Qwen2ForCausalLM,
    SiglipVisionConfig,
    SiglipVisionModel,
    WhisperConfig,
    WhisperModel,
)

from cutile_forge.models import (
    UnifiedModelExecutor,
    build_model_graph,
    extract_hf_decoder_weights,
    extract_hf_siglip_vision_weights,
    extract_hf_whisper_encoder_weights,
    init_synthetic_weights,
    list_supported_models,
)


def _device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def test_registry_contains_all_phase11_architectures() -> None:
    models = set(list_supported_models())
    expected = {
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
        "whisper_encoder",
        "dit",
    }
    assert expected.issubset(models)


def test_hf_transformers_parity_llama_qwen2_mistral_and_mixtral() -> None:
    """Verify CuTile `UnifiedModelExecutor` matches HuggingFace `transformers` models across Llama, Qwen2, Mistral, Mixtral."""
    torch.manual_seed(1101)
    dev = _device()
    dt = torch.float16 if dev.type == "cuda" else torch.float32
    input_ids = torch.tensor([[1, 15, 42, 88, 103, 7, 29, 64]], device=dev, dtype=torch.long)

    # 1. Llama (GQA, RoPE)
    llama_cfg = LlamaConfig(
        vocab_size=256,
        hidden_size=128,
        intermediate_size=256,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=128,
        rms_norm_eps=1e-5,
    )
    hf_llama = LlamaForCausalLM(llama_cfg).to(device=dev, dtype=torch.float32).eval()
    g_llama, w_llama = extract_hf_decoder_weights(hf_llama, device=dev, dtype=dt)
    with torch.no_grad():
        hf_out = hf_llama(input_ids).logits.float()
        ct_out = UnifiedModelExecutor(g_llama).forward(w_llama, input_ids=input_ids).float()
    assert float((ct_out - hf_out).abs().max().item()) < 2e-2

    # 2. Qwen2 / Qwen2.5 (QKV bias + high-theta RoPE)
    qwen_cfg = Qwen2Config(
        vocab_size=256,
        hidden_size=128,
        intermediate_size=256,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=128,
        rope_theta=1000000.0,
        rms_norm_eps=1e-6,
    )
    hf_qwen = Qwen2ForCausalLM(qwen_cfg).to(device=dev, dtype=torch.float32).eval()
    g_qwen, w_qwen = extract_hf_decoder_weights(hf_qwen, device=dev, dtype=dt)
    with torch.no_grad():
        hf_q_out = hf_qwen(input_ids).logits.float()
        ct_q_out = UnifiedModelExecutor(g_qwen).forward(w_qwen, input_ids=input_ids).float()
    assert float((ct_q_out - hf_q_out).abs().max().item()) < 2e-2

    # 3. Mistral (Sliding-window GQA)
    mistral_cfg = MistralConfig(
        vocab_size=256,
        hidden_size=128,
        intermediate_size=256,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        sliding_window=4,
        max_position_embeddings=128,
        rms_norm_eps=1e-5,
    )
    hf_mistral = MistralForCausalLM(mistral_cfg).to(device=dev, dtype=torch.float32).eval()
    g_mistral, w_mistral = extract_hf_decoder_weights(hf_mistral, device=dev, dtype=dt)
    with torch.no_grad():
        hf_m_out = hf_mistral(input_ids).logits.float()
        ct_m_out = UnifiedModelExecutor(g_mistral).forward(w_mistral, input_ids=input_ids).float()
    assert float((ct_m_out - hf_m_out).abs().max().item()) < 2e-2

    # 4. Mixtral Sparse Top-2 MoE
    mixtral_cfg = MixtralConfig(
        vocab_size=256,
        hidden_size=128,
        intermediate_size=256,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        num_local_experts=4,
        num_experts_per_tok=2,
        max_position_embeddings=128,
        rms_norm_eps=1e-5,
    )
    hf_mixtral = MixtralForCausalLM(mixtral_cfg).to(device=dev, dtype=torch.float32).eval()
    g_mixtral, w_mixtral = extract_hf_decoder_weights(hf_mixtral, device=dev, dtype=dt)
    with torch.no_grad():
        hf_mx_out = hf_mixtral(input_ids).logits.float()
        ct_mx_out = UnifiedModelExecutor(g_mixtral).forward(w_mixtral, input_ids=input_ids).float()
    mx_diff = (ct_mx_out - hf_mx_out).abs()
    mx_cos = float(
        torch.nn.functional.cosine_similarity(
            ct_mx_out.flatten(), hf_mx_out.flatten(), dim=0
        ).item()
    )
    assert mx_cos >= 0.999 and float(mx_diff.median().item()) < 2e-2


def test_advanced_moe_shared_experts_and_single_gpu_ep_partition_invariance() -> None:
    """Verify DeepSeek/Qwen2-MoE shared + fine-grained routed experts and single-GPU EP partition invariance."""
    dev = _device()
    input_ids = torch.tensor([[2, 19, 44, 71, 108, 5, 33, 91]], device=dev, dtype=torch.long)

    g_ep1 = build_model_graph(
        "deepseek_moe",
        vocab_size=256,
        hidden_size=128,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=4,
        head_dim=32,
        num_local_experts=8,
        num_experts_per_tok=3,
        shared_expert_intermediate_size=256,
        ep_partitions=1,
    )
    g_ep4 = build_model_graph(
        "deepseek_moe",
        vocab_size=256,
        hidden_size=128,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=4,
        head_dim=32,
        num_local_experts=8,
        num_experts_per_tok=3,
        shared_expert_intermediate_size=256,
        ep_partitions=4,
    )
    weights = init_synthetic_weights(g_ep1, seed=77, device=dev)

    out_ep1 = UnifiedModelExecutor(g_ep1).forward(weights, input_ids=input_ids)
    out_ep4 = UnifiedModelExecutor(g_ep4).forward(weights, input_ids=input_ids)
    out_ref = UnifiedModelExecutor(g_ep1).reference_forward(weights, input_ids=input_ids)

    # Single-GPU 4-way expert parallelism must be bit-identical to unpartitioned EP=1
    assert float((out_ep1.float() - out_ep4.float()).abs().max().item()) == 0.0
    assert float((out_ep1.float() - out_ref.float()).abs().max().item()) < 2e-2


def test_hf_siglip_vision_and_whisper_audio_encoder_parity() -> None:
    """Verify CuTile SigLIP vision encoder and Whisper audio encoder match HuggingFace `transformers`."""
    torch.manual_seed(1102)
    dev = _device()
    dt = torch.float16 if dev.type == "cuda" else torch.float32

    # 1. SigLIP Vision Encoder
    sig_cfg = SiglipVisionConfig(
        hidden_size=128,
        intermediate_size=256,
        num_hidden_layers=2,
        num_attention_heads=4,
        image_size=64,
        patch_size=16,
    )
    hf_sig = SiglipVisionModel(sig_cfg).to(device=dev, dtype=torch.float32).eval()
    g_sig, w_sig = extract_hf_siglip_vision_weights(hf_sig, device=dev, dtype=dt)
    pixels = torch.randn((2, 3, 64, 64), device=dev, dtype=dt) * 0.2
    with torch.no_grad():
        hf_sig_out = hf_sig(pixels.float()).last_hidden_state.float()
        ct_sig_out = UnifiedModelExecutor(g_sig).forward(w_sig, pixel_values=pixels).float()
    assert ct_sig_out.shape == hf_sig_out.shape
    assert float((ct_sig_out - hf_sig_out).abs().max().item()) < 2e-2

    # 2. Whisper Audio Encoder
    wh_cfg = WhisperConfig(
        num_mel_bins=80,
        d_model=128,
        encoder_attention_heads=4,
        decoder_attention_heads=4,
        encoder_ffn_dim=256,
        decoder_ffn_dim=256,
        encoder_layers=2,
        decoder_layers=2,
        max_source_positions=64,
    )
    hf_wh = WhisperModel(wh_cfg).to(device=dev, dtype=torch.float32).eval()
    g_wh, w_wh = extract_hf_whisper_encoder_weights(hf_wh, device=dev, dtype=dt)
    mel = torch.randn((2, 80, 128), device=dev, dtype=dt) * 0.2
    with torch.no_grad():
        hf_wh_out = hf_wh.encoder(mel.float()).last_hidden_state.float()
        ct_wh_out = UnifiedModelExecutor(g_wh).forward(w_wh, mel_features=mel).float()
    assert ct_wh_out.shape == hf_wh_out.shape
    assert float((ct_wh_out - hf_wh_out).abs().max().item()) < 2e-2


def test_llava_multimodal_and_dit_diffusion_execution_and_tfg_lowering() -> None:
    """Verify LLaVA multimodal pipeline and DiT diffusion transformer execution + TFG compilation."""
    dev = _device()
    dt = torch.float16 if dev.type == "cuda" else torch.float32

    # 1. LLaVA Multimodal
    g_llava = build_model_graph(
        "llava",
        vocab_size=256,
        hidden_size=128,
        intermediate_size=256,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=32,
        image_size=64,
        patch_size=16,
        vision_hidden_size=128,
        vision_intermediate_size=256,
        vision_num_layers=2,
        vision_num_heads=4,
    )
    w_llava = init_synthetic_weights(g_llava, seed=88, device=dev, dtype=dt)
    ids = torch.tensor([[1, 12, 34, 56, 78, 90]], device=dev, dtype=torch.long)
    img = torch.randn((1, 3, 64, 64), device=dev, dtype=dt) * 0.2
    ex_llava = UnifiedModelExecutor(g_llava)
    out_llava_ct = ex_llava.forward(w_llava, input_ids=ids, pixel_values=img)
    out_llava_ref = ex_llava.reference_forward(w_llava, input_ids=ids, pixel_values=img)
    assert out_llava_ct.shape == (1, 16 + 6, 256)
    assert float((out_llava_ct.float() - out_llava_ref.float()).abs().max().item()) < 2e-2

    # 2. Diffusion Transformer (DiT)
    g_dit = build_model_graph(
        "dit",
        image_size=16,
        patch_size=2,
        in_channels=4,
        out_channels=4,
        hidden_size=128,
        intermediate_size=256,
        num_hidden_layers=2,
        num_attention_heads=4,
    )
    w_dit = init_synthetic_weights(g_dit, seed=99, device=dev, dtype=dt)
    latents = torch.randn((2, 4, 16, 16), device=dev, dtype=dt) * 0.2
    timesteps = torch.tensor([250, 750], device=dev, dtype=torch.long)
    ex_dit = UnifiedModelExecutor(g_dit)
    out_dit_ct = ex_dit.forward(w_dit, pixel_values=latents, timesteps=timesteps)
    out_dit_ref = ex_dit.reference_forward(w_dit, pixel_values=latents, timesteps=timesteps)
    assert out_dit_ct.shape == (2, 4, 16, 16)
    assert float((out_dit_ct.float() - out_dit_ref.float()).abs().max().item()) < 2e-2

    tfg, report = g_dit.compile_tfg(batch_size=2, seq_len=64)
    assert len(tfg.nodes) > 0
    assert report is not None
