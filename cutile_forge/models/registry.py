"""Model family registry mapping HuggingFace configs to `UnifiedModelGraph`."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

from cutile_forge.models.model_graph import (
    AudioConvEmbedStage,
    BlockStage,
    ClassifierHeadStage,
    DenseTransformerBlock,
    DiffusionOutputStage,
    DiffusionPatchEmbedStage,
    DiTBlock,
    EncoderOutputStage,
    LMHeadStage,
    Modality,
    MultimodalFusionStage,
    SparseMoEBlock,
    TextEmbeddingStage,
    UnifiedModelGraph,
    VisionPatchEmbedStage,
    ViTEncoderBlock,
)


@dataclass(frozen=True)
class ModelSpec:
    """Registered architecture descriptor inside `MODEL_REGISTRY`."""

    key: str
    display_name: str
    family: str
    modality: Modality
    hf_model_types: tuple[str, ...]
    default_config: dict[str, Any]


MODEL_REGISTRY: dict[str, ModelSpec] = {
    "mistral": ModelSpec(
        key="mistral",
        display_name="Mistral-7B (Sliding-Window GQA)",
        family="mistral",
        modality="text",
        hf_model_types=("mistral",),
        default_config={
            "vocab_size": 32000,
            "hidden_size": 4096,
            "intermediate_size": 14336,
            "num_hidden_layers": 32,
            "num_attention_heads": 32,
            "num_key_value_heads": 8,
            "head_dim": 128,
            "rope_theta": 10000.0,
            "sliding_window": 4096,
            "rms_norm_eps": 1e-5,
            "qkv_bias": False,
        },
    ),
    "llama": ModelSpec(
        key="llama",
        display_name="Llama-2 (Dense GQA/MHA)",
        family="llama",
        modality="text",
        hf_model_types=("llama",),
        default_config={
            "vocab_size": 32000,
            "hidden_size": 4096,
            "intermediate_size": 11008,
            "num_hidden_layers": 32,
            "num_attention_heads": 32,
            "num_key_value_heads": 32,
            "head_dim": 128,
            "rope_theta": 10000.0,
            "rms_norm_eps": 1e-5,
            "qkv_bias": False,
        },
    ),
    "llama3": ModelSpec(
        key="llama3",
        display_name="Llama-3 / 3.1 (Scaled RoPE, 128K Vocab)",
        family="llama3",
        modality="text",
        hf_model_types=("llama3",),
        default_config={
            "vocab_size": 128256,
            "hidden_size": 4096,
            "intermediate_size": 14336,
            "num_hidden_layers": 32,
            "num_attention_heads": 32,
            "num_key_value_heads": 8,
            "head_dim": 128,
            "rope_theta": 500000.0,
            "rope_scaling": {
                "rope_type": "llama3",
                "factor": 8.0,
                "low_freq_factor": 1.0,
                "high_freq_factor": 4.0,
                "original_max_position_embeddings": 8192,
            },
            "rms_norm_eps": 1e-5,
            "qkv_bias": False,
        },
    ),
    "qwen2": ModelSpec(
        key="qwen2",
        display_name="Qwen2 / Qwen2.5 (QKV Bias + High-Theta RoPE)",
        family="qwen2",
        modality="text",
        hf_model_types=("qwen2", "qwen2_5"),
        default_config={
            "vocab_size": 151936,
            "hidden_size": 3584,
            "intermediate_size": 18944,
            "num_hidden_layers": 28,
            "num_attention_heads": 28,
            "num_key_value_heads": 4,
            "head_dim": 128,
            "rope_theta": 1000000.0,
            "rms_norm_eps": 1e-6,
            "qkv_bias": True,
            "tie_word_embeddings": False,
        },
    ),
    "mixtral": ModelSpec(
        key="mixtral",
        display_name="Mixtral-8x7B / 8x22B (Sparse Top-2 MoE)",
        family="mixtral",
        modality="text",
        hf_model_types=("mixtral",),
        default_config={
            "vocab_size": 32000,
            "hidden_size": 4096,
            "intermediate_size": 14336,
            "num_hidden_layers": 32,
            "num_attention_heads": 32,
            "num_key_value_heads": 8,
            "head_dim": 128,
            "num_local_experts": 8,
            "num_experts_per_tok": 2,
            "rope_theta": 1000000.0,
            "rms_norm_eps": 1e-5,
        },
    ),
    "deepseek_moe": ModelSpec(
        key="deepseek_moe",
        display_name="DeepSeek-MoE / DeepSeek-V2 (Shared + Fine-Grained Experts)",
        family="deepseek_moe",
        modality="text",
        hf_model_types=("deepseek", "deepseek_v2", "deepseek_v3"),
        default_config={
            "vocab_size": 102400,
            "hidden_size": 2048,
            "intermediate_size": 1408,
            "num_hidden_layers": 28,
            "num_attention_heads": 16,
            "num_key_value_heads": 16,
            "head_dim": 128,
            "num_local_experts": 64,
            "num_experts_per_tok": 6,
            "shared_expert_intermediate_size": 2816,
            "rope_theta": 10000.0,
            "rms_norm_eps": 1e-6,
        },
    ),
    "qwen2_moe": ModelSpec(
        key="qwen2_moe",
        display_name="Qwen2-MoE (QKV Bias + Shared & Routed Experts)",
        family="qwen2_moe",
        modality="text",
        hf_model_types=("qwen2_moe",),
        default_config={
            "vocab_size": 151936,
            "hidden_size": 2048,
            "intermediate_size": 1408,
            "num_hidden_layers": 24,
            "num_attention_heads": 16,
            "num_key_value_heads": 16,
            "head_dim": 128,
            "num_local_experts": 60,
            "num_experts_per_tok": 4,
            "shared_expert_intermediate_size": 5632,
            "rope_theta": 1000000.0,
            "rms_norm_eps": 1e-6,
            "qkv_bias": True,
        },
    ),
    "vit": ModelSpec(
        key="vit",
        display_name="Vision Transformer (ViT-B/16, ViT-L/14)",
        family="vit",
        modality="vision",
        hf_model_types=("vit", "deit"),
        default_config={
            "image_size": 224,
            "patch_size": 16,
            "in_channels": 3,
            "hidden_size": 768,
            "intermediate_size": 3072,
            "num_hidden_layers": 12,
            "num_attention_heads": 12,
            "num_labels": 1000,
            "layer_norm_eps": 1e-6,
            "use_cls_token": True,
        },
    ),
    "clip_vision": ModelSpec(
        key="clip_vision",
        display_name="CLIP Vision Encoder",
        family="clip_vision",
        modality="vision",
        hf_model_types=("clip_vision_model",),
        default_config={
            "image_size": 224,
            "patch_size": 14,
            "in_channels": 3,
            "hidden_size": 1024,
            "intermediate_size": 4096,
            "num_hidden_layers": 24,
            "num_attention_heads": 16,
            "num_labels": 768,
            "layer_norm_eps": 1e-5,
            "use_cls_token": True,
        },
    ),
    "siglip_vision": ModelSpec(
        key="siglip_vision",
        display_name="SigLIP Vision Encoder (Mean-Pool, Patch Tokens)",
        family="siglip_vision",
        modality="vision",
        hf_model_types=("siglip_vision_model",),
        default_config={
            "image_size": 224,
            "patch_size": 14,
            "in_channels": 3,
            "hidden_size": 1152,
            "intermediate_size": 4304,
            "num_hidden_layers": 27,
            "num_attention_heads": 16,
            "num_labels": 1152,
            "layer_norm_eps": 1e-6,
            "use_cls_token": False,
        },
    ),
    "llava": ModelSpec(
        key="llava",
        display_name="LLaVA Multimodal (ViT + 2-Layer GELU Projector + LLM)",
        family="llava",
        modality="multimodal",
        hf_model_types=("llava", "llava_next"),
        default_config={
            "vocab_size": 32000,
            "hidden_size": 4096,
            "intermediate_size": 11008,
            "num_hidden_layers": 32,
            "num_attention_heads": 32,
            "num_key_value_heads": 32,
            "head_dim": 128,
            "rope_theta": 10000.0,
            "rms_norm_eps": 1e-5,
            "image_size": 224,
            "patch_size": 14,
            "vision_hidden_size": 1024,
            "vision_intermediate_size": 4096,
            "vision_num_layers": 2,
            "vision_num_heads": 16,
        },
    ),
    "whisper_encoder": ModelSpec(
        key="whisper_encoder",
        display_name="Whisper Audio Encoder (2x Conv1d Stem + Bidirectional Transformer)",
        family="whisper_encoder",
        modality="audio",
        hf_model_types=("whisper",),
        default_config={
            "num_mel_bins": 80,
            "max_source_positions": 1500,
            "hidden_size": 384,
            "intermediate_size": 1536,
            "num_hidden_layers": 4,
            "num_attention_heads": 6,
            "layer_norm_eps": 1e-5,
        },
    ),
    "dit": ModelSpec(
        key="dit",
        display_name="Diffusion Transformer (DiT with AdaLN-Zero Conditioning)",
        family="dit",
        modality="diffusion",
        hf_model_types=("dit", "transformer2d"),
        default_config={
            "image_size": 32,
            "patch_size": 2,
            "in_channels": 4,
            "out_channels": 4,
            "hidden_size": 384,
            "intermediate_size": 1536,
            "num_hidden_layers": 12,
            "num_attention_heads": 6,
            "layer_norm_eps": 1e-6,
        },
    ),
}


def list_supported_models() -> list[str]:
    """Return sorted list of registered model family keys."""
    return sorted(MODEL_REGISTRY.keys())


def get_model_spec(name: str) -> ModelSpec:
    """Resolve a registered `ModelSpec` by key or HuggingFace `model_type`."""
    norm = name.strip().lower().replace("-", "_")
    if norm in MODEL_REGISTRY:
        return MODEL_REGISTRY[norm]
    for spec in MODEL_REGISTRY.values():
        if norm in spec.hf_model_types:
            return spec
    raise KeyError(
        f"Unsupported model architecture: '{name}'. Available: {list_supported_models()}"
    )


def _normalize_hf_rope_config(config: dict[str, Any]) -> dict[str, Any]:
    out = dict(config)
    rp = out.get("rope_parameters")
    if isinstance(rp, dict):
        if out.get("rope_theta") is None and rp.get("rope_theta") is not None:
            out["rope_theta"] = rp["rope_theta"]
        if out.get("rope_scaling") is None and rp.get("rope_type") not in (None, "default"):
            out["rope_scaling"] = dict(rp)
    return out


def _resolve_registry_key(config: dict[str, Any], name: str | None = None) -> str:
    if name is not None:
        return get_model_spec(name).key
    norm_cfg = _normalize_hf_rope_config(config)
    model_type = str(norm_cfg.get("model_type", "mistral")).strip().lower().replace("-", "_")
    if model_type == "llama":
        if norm_cfg.get("rope_scaling") or float(norm_cfg.get("rope_theta") or 10000.0) >= 100000.0:
            return "llama3"
        return "llama"
    if model_type in MODEL_REGISTRY:
        return model_type
    for spec in MODEL_REGISTRY.values():
        if model_type in spec.hf_model_types:
            return spec.key
    return "mistral"


def from_hf_config(config: dict[str, Any], name: str | None = None) -> UnifiedModelGraph:
    """Construct a `UnifiedModelGraph` from a HuggingFace-style config dict."""
    norm_cfg = _normalize_hf_rope_config(config)
    key = _resolve_registry_key(norm_cfg, name=name)
    spec = MODEL_REGISTRY[key]
    merged: dict[str, Any] = {**spec.default_config}
    for k_c, v_c in norm_cfg.items():
        if v_c is not None:
            merged[k_c] = v_c

    if spec.modality == "vision":
        d = int(merged["hidden_size"])
        n_heads = int(merged["num_attention_heads"])
        head_dim = int(merged.get("head_dim", d // n_heads))
        mlp_dim = int(merged.get("intermediate_size", merged.get("mlp_dim", 4 * d)))
        n_layers = int(merged["num_hidden_layers"])
        eps = float(merged.get("layer_norm_eps", 1e-6))
        use_cls = bool(merged.get("use_cls_token", key != "siglip_vision"))
        pre_ln = bool(merged.get("pre_layer_norm", key == "clip_vision"))
        input_stage = VisionPatchEmbedStage(
            hidden_size=d,
            image_size=int(merged.get("image_size", 224)),
            patch_size=int(merged.get("patch_size", 16)),
            in_channels=int(merged.get("in_channels", 3)),
            use_cls_token=use_cls,
            pre_layer_norm=pre_ln,
            layer_norm_eps=eps,
        )
        raw_act = str(merged.get("hidden_act", "gelu")).lower()
        vis_act: Literal["gelu", "silu", "quick_gelu", "gelu_erf"] = (
            "quick_gelu"
            if raw_act == "quick_gelu"
            else ("gelu_erf" if (key == "vit" and raw_act == "gelu") else "gelu")
        )
        blocks: list[BlockStage] = [
            ViTEncoderBlock(
                hidden_size=d,
                mlp_dim=mlp_dim,
                num_attention_heads=n_heads,
                head_dim=head_dim,
                layer_norm_eps=eps,
                activation=vis_act,
            )
            for _ in range(n_layers)
        ]
        out_stage = ClassifierHeadStage(
            hidden_size=d,
            num_classes=int(merged.get("num_labels", 1000)),
            layer_norm_eps=eps,
            pool_type="cls" if use_cls else "mean",
        )
        return UnifiedModelGraph(
            name=str(merged.get("_name_or_path", key)),
            family=spec.family,
            modality="vision",
            input_stage=input_stage,
            blocks=blocks,
            output_stage=out_stage,
            metadata={"hf_config": config},
        )

    if spec.modality == "audio":
        d = int(merged.get("d_model", merged["hidden_size"]))
        n_heads = int(merged.get("encoder_attention_heads", merged["num_attention_heads"]))
        head_dim = int(merged.get("head_dim", d // n_heads))
        mlp_dim = int(merged.get("encoder_ffn_dim", merged.get("intermediate_size", 4 * d)))
        n_layers = int(merged.get("encoder_layers", merged["num_hidden_layers"]))
        eps = float(merged.get("layer_norm_eps", 1e-5))
        aud_in = AudioConvEmbedStage(
            hidden_size=d,
            num_mel_bins=int(merged.get("num_mel_bins", 80)),
            max_source_positions=int(merged.get("max_source_positions", 1500)),
        )
        act_fn = str(merged.get("activation_function", "gelu")).lower()
        aud_act: Literal["gelu", "silu", "quick_gelu", "gelu_erf"] = (
            "gelu_erf" if act_fn == "gelu" else "gelu"
        )
        aud_blocks: list[BlockStage] = [
            ViTEncoderBlock(
                hidden_size=d,
                mlp_dim=mlp_dim,
                num_attention_heads=n_heads,
                head_dim=head_dim,
                layer_norm_eps=eps,
                activation=aud_act,
            )
            for _ in range(n_layers)
        ]
        return UnifiedModelGraph(
            name=str(merged.get("_name_or_path", key)),
            family=spec.family,
            modality="audio",
            input_stage=aud_in,
            blocks=aud_blocks,
            output_stage=EncoderOutputStage(hidden_size=d, layer_norm_eps=eps),
            metadata={"hf_config": config},
        )

    if spec.modality == "diffusion":
        d = int(merged["hidden_size"])
        n_heads = int(merged["num_attention_heads"])
        head_dim = int(merged.get("head_dim", d // n_heads))
        mlp_dim = int(merged.get("intermediate_size", 4 * d))
        n_layers = int(merged["num_hidden_layers"])
        eps = float(merged.get("layer_norm_eps", 1e-6))
        img_sz = int(merged.get("image_size", 32))
        patch_sz = int(merged.get("patch_size", 2))
        in_ch = int(merged.get("in_channels", 4))
        out_ch = int(merged.get("out_channels", in_ch))
        dif_in = DiffusionPatchEmbedStage(
            hidden_size=d,
            image_size=img_sz,
            patch_size=patch_sz,
            in_channels=in_ch,
        )
        dif_blocks: list[BlockStage] = [
            DiTBlock(
                hidden_size=d,
                mlp_dim=mlp_dim,
                num_attention_heads=n_heads,
                head_dim=head_dim,
                layer_norm_eps=eps,
            )
            for _ in range(n_layers)
        ]
        return UnifiedModelGraph(
            name=str(merged.get("_name_or_path", key)),
            family=spec.family,
            modality="diffusion",
            input_stage=dif_in,
            blocks=dif_blocks,
            output_stage=DiffusionOutputStage(
                hidden_size=d,
                image_size=img_sz,
                patch_size=patch_sz,
                out_channels=out_ch,
                layer_norm_eps=eps,
            ),
            metadata={"hf_config": config},
        )

    if spec.modality == "multimodal":
        d = int(merged["hidden_size"])
        n_heads = int(merged["num_attention_heads"])
        n_kv = int(merged.get("num_key_value_heads", n_heads))
        head_dim = int(merged.get("head_dim", d // n_heads))
        v_d = int(merged.get("vision_hidden_size", 1024))
        v_heads = int(merged.get("vision_num_heads", 16))
        v_mlp = int(merged.get("vision_intermediate_size", 4 * v_d))
        v_layers = int(merged.get("vision_num_layers", 2))
        vs = VisionPatchEmbedStage(
            hidden_size=v_d,
            image_size=int(merged.get("image_size", 224)),
            patch_size=int(merged.get("patch_size", 14)),
            in_channels=int(merged.get("in_channels", 3)),
            use_cls_token=bool(merged.get("use_cls_token", True)),
        )
        v_blocks = [
            ViTEncoderBlock(
                hidden_size=v_d,
                mlp_dim=v_mlp,
                num_attention_heads=v_heads,
                head_dim=v_d // v_heads,
            )
            for _ in range(v_layers)
        ]
        mm_input = MultimodalFusionStage(
            hidden_size=d,
            vocab_size=int(merged["vocab_size"]),
            vision_stage=vs,
            vision_blocks=v_blocks,
            projector_hidden_size=int(merged.get("projector_hidden_size", d)),
        )
        llm_blocks: list[BlockStage] = [
            DenseTransformerBlock(
                hidden_size=d,
                intermediate_size=int(merged["intermediate_size"]),
                num_attention_heads=n_heads,
                num_key_value_heads=n_kv,
                head_dim=head_dim,
                rms_norm_eps=float(merged.get("rms_norm_eps", 1e-5)),
                rope_theta=float(merged.get("rope_theta", 10000.0)),
                rope_scaling=merged.get("rope_scaling"),
                qkv_bias=bool(merged.get("qkv_bias", False)),
            )
            for _ in range(int(merged["num_hidden_layers"]))
        ]
        mm_out = LMHeadStage(
            hidden_size=d,
            vocab_size=int(merged["vocab_size"]),
            rms_norm_eps=float(merged.get("rms_norm_eps", 1e-5)),
            tie_word_embeddings=bool(merged.get("tie_word_embeddings", False)),
        )
        return UnifiedModelGraph(
            name=str(merged.get("_name_or_path", key)),
            family=spec.family,
            modality="multimodal",
            input_stage=mm_input,
            blocks=llm_blocks,
            output_stage=mm_out,
            metadata={"hf_config": config},
        )

    d = int(merged["hidden_size"])
    n_heads = int(merged["num_attention_heads"])
    n_kv = int(merged.get("num_key_value_heads", n_heads))
    head_dim = int(config.get("head_dim") or (d // n_heads))
    vocab_size = int(merged["vocab_size"])
    eps = float(merged.get("rms_norm_eps", 1e-5))
    n_layers = int(merged["num_hidden_layers"])
    num_experts = int(merged.get("num_local_experts", merged.get("num_experts", 1)))
    is_moe = num_experts > 1 or key in ("mixtral", "deepseek_moe", "qwen2_moe")

    text_blocks: list[BlockStage] = []
    for _ in range(n_layers):
        if is_moe:
            text_blocks.append(
                SparseMoEBlock(
                    hidden_size=d,
                    intermediate_size=int(merged["intermediate_size"]),
                    num_attention_heads=n_heads,
                    num_key_value_heads=n_kv,
                    head_dim=head_dim,
                    num_experts=max(2, num_experts),
                    top_k_experts=int(merged.get("num_experts_per_tok", 2)),
                    shared_expert_intermediate_size=int(
                        merged.get("shared_expert_intermediate_size", 0)
                    ),
                    ep_partitions=int(merged.get("ep_partitions", 1)),
                    rms_norm_eps=eps,
                    rope_theta=float(merged.get("rope_theta", 10000.0)),
                    rope_scaling=merged.get("rope_scaling"),
                    qkv_bias=bool(merged.get("qkv_bias", False)),
                    sliding_window=merged.get("sliding_window"),
                )
            )
        else:
            text_blocks.append(
                DenseTransformerBlock(
                    hidden_size=d,
                    intermediate_size=int(merged["intermediate_size"]),
                    num_attention_heads=n_heads,
                    num_key_value_heads=n_kv,
                    head_dim=head_dim,
                    rms_norm_eps=eps,
                    norm_type=str(merged.get("norm_type", "rmsnorm")),  # type: ignore[arg-type]
                    activation=str(merged.get("hidden_act", "silu")),  # type: ignore[arg-type]
                    rope_theta=float(merged.get("rope_theta", 10000.0)),
                    rope_scaling=merged.get("rope_scaling"),
                    qkv_bias=bool(merged.get("qkv_bias", False)),
                    sliding_window=merged.get("sliding_window"),
                )
            )

    return UnifiedModelGraph(
        name=str(merged.get("_name_or_path", key)),
        family=spec.family,
        modality="text",
        input_stage=TextEmbeddingStage(hidden_size=d, vocab_size=vocab_size),
        blocks=text_blocks,
        output_stage=LMHeadStage(
            hidden_size=d,
            vocab_size=vocab_size,
            rms_norm_eps=eps,
            tie_word_embeddings=bool(merged.get("tie_word_embeddings", False)),
        ),
        metadata={"hf_config": config},
    )


def build_model_graph(name: str, **overrides: Any) -> UnifiedModelGraph:
    """Build a `UnifiedModelGraph` directly from `MODEL_REGISTRY` with optional config overrides."""
    spec = get_model_spec(name)
    cfg = {**spec.default_config, **overrides}
    return from_hf_config(cfg, name=spec.key)
