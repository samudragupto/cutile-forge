"""Declarative UnifiedModelGraph IR, TFG lowering, and portable executor.

Supports dense LLMs (Mistral, Llama-2/3, Qwen2), Sparse MoE models (Mixtral, DeepSeek-MoE,
Qwen2-MoE with shared + routed experts & single-GPU expert parallelism), Vision Transformers
(ViT, CLIP, SigLIP), Audio Encoders (Whisper), Multimodal pipelines (LLaVA: ViT -> 2-layer
GELU projector -> unified LLM prefill stream), and Diffusion models (DiT / UNet).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Literal

import torch
import torch.nn.functional as F

from cutile_forge.forge.fusion_graph import DataTensor, FusionGraph, OpType, TileNode
from cutile_forge.forge.pass_manager import PassReport, default_pass_manager
from cutile_forge.kernels import (
    cutile_gelu,
    cutile_gemm_fp16,
    cutile_layernorm,
    cutile_rmsnorm,
    cutile_silu,
)

Modality = Literal["text", "vision", "audio", "multimodal", "diffusion"]


def compute_scaled_rope_freqs(
    max_seq_len: int,
    head_dim: int,
    theta: float = 10000.0,
    rope_scaling: dict[str, Any] | None = None,
    device: torch.device | str = "cpu",
    dtype: torch.dtype = torch.float32,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute `(cos, sin)` tables supporting standard, Linear, NTK/YaRN, and Llama-3 RoPE."""
    if head_dim % 2 != 0:
        raise ValueError(f"head_dim must be even for RoPE, got {head_dim}")

    eff_theta = float(theta)
    scaling = rope_scaling or {}
    scale_type = str(scaling.get("rope_type", scaling.get("type", "none"))).lower()
    factor = float(scaling.get("factor", 1.0))

    if scale_type in ("ntk", "yarn") and factor > 1.0:
        eff_theta = eff_theta * (factor ** (head_dim / max(1.0, float(head_dim - 2))))

    inv_freq = 1.0 / (
        eff_theta
        ** (torch.arange(0, head_dim, 2, device=device, dtype=torch.float32) / float(head_dim))
    )

    if scale_type == "llama3" and factor > 1.0:
        low_factor = float(scaling.get("low_freq_factor", 1.0))
        high_factor = float(scaling.get("high_freq_factor", 4.0))
        old_ctx = float(scaling.get("original_max_position_embeddings", 8192))
        low_wavelen = old_ctx / low_factor
        high_wavelen = old_ctx / high_factor
        wavelens = (2.0 * math.pi) / inv_freq
        scaled_parts: list[torch.Tensor] = []
        for idx in range(inv_freq.shape[0]):
            wl = float(wavelens[idx].item())
            freq_val = inv_freq[idx]
            if wl > low_wavelen:
                scaled_parts.append(freq_val / factor)
            elif wl < high_wavelen:
                scaled_parts.append(freq_val)
            else:
                smooth = (old_ctx / wl - low_factor) / max(1e-6, high_factor - low_factor)
                scaled_parts.append((1.0 - smooth) * (freq_val / factor) + smooth * freq_val)
        inv_freq = torch.stack(scaled_parts)
    elif scale_type == "linear" and factor > 1.0:
        inv_freq = inv_freq / factor

    t = torch.arange(max_seq_len, device=device, dtype=torch.float32)
    freqs = torch.outer(t, inv_freq)
    return freqs.cos().to(dtype), freqs.sin().to(dtype)


# ---------------------------------------------------------------------------
# Declarative Graph Stages
# ---------------------------------------------------------------------------


@dataclass
class InputStage:
    """Base input stage for UnifiedModelGraph."""

    hidden_size: int


@dataclass
class TextEmbeddingStage(InputStage):
    """Token embedding table stage for decoder LLMs."""

    vocab_size: int = 32000


@dataclass
class VisionPatchEmbedStage(InputStage):
    """2D convolution patch embedding + class token / positional embedding stage."""

    image_size: int = 224
    patch_size: int = 16
    in_channels: int = 3
    use_cls_token: bool = True
    pre_layer_norm: bool = False
    layer_norm_eps: float = 1e-5

    @property
    def num_patches(self) -> int:
        return (self.image_size // self.patch_size) ** 2

    @property
    def num_tokens(self) -> int:
        return self.num_patches + (1 if self.use_cls_token else 0)


@dataclass
class AudioConvEmbedStage(InputStage):
    """Whisper-style 2-layer Conv1d stem (`conv1 k=3 s=1 -> GELU -> conv2 k=3 s=2 -> GELU + pos_embed`)."""

    num_mel_bins: int = 80
    max_source_positions: int = 1500


@dataclass
class DiffusionPatchEmbedStage(InputStage):
    """Diffusion Transformer (DiT) latent patch embedding + timestep conditioning MLP stage."""

    image_size: int = 32
    patch_size: int = 2
    in_channels: int = 4

    @property
    def num_patches(self) -> int:
        return (self.image_size // self.patch_size) ** 2


@dataclass
class BlockStage:
    """Base transformer block stage."""

    hidden_size: int


@dataclass
class ViTEncoderBlock(BlockStage):
    """Bidirectional encoder block (Pre-LN + MHA + GELU MLP)."""

    mlp_dim: int = 3072
    num_attention_heads: int = 12
    head_dim: int = 64
    layer_norm_eps: float = 1e-5
    activation: Literal["gelu", "silu"] = "gelu"
    qkv_bias: bool = True


@dataclass
class MultimodalFusionStage(InputStage):
    """Combined vision encoder + projector + text embedding stage (LLaVA-style)."""

    vocab_size: int = 32000
    vision_stage: VisionPatchEmbedStage = field(
        default_factory=lambda: VisionPatchEmbedStage(hidden_size=1024)
    )
    vision_blocks: list[ViTEncoderBlock] = field(default_factory=list)
    projector_hidden_size: int | None = None
    select_feature: str = "patch"


@dataclass
class DenseTransformerBlock(BlockStage):
    """Decoder transformer block with GQA/MHA, RoPE/scaled RoPE, SwiGLU/GELU, and optional bias."""

    intermediate_size: int = 14336
    num_attention_heads: int = 32
    num_key_value_heads: int = 8
    head_dim: int = 128
    rms_norm_eps: float = 1e-5
    norm_type: Literal["rmsnorm", "layernorm"] = "rmsnorm"
    activation: Literal["silu", "gelu"] = "silu"
    rope_theta: float = 10000.0
    rope_scaling: dict[str, Any] | None = None
    qkv_bias: bool = False
    sliding_window: int | None = None


@dataclass
class SparseMoEBlock(BlockStage):
    """Mixture-of-Experts decoder block supporting routed + optional shared experts & EP partitions."""

    intermediate_size: int = 14336
    num_attention_heads: int = 32
    num_key_value_heads: int = 8
    head_dim: int = 128
    num_experts: int = 8
    top_k_experts: int = 2
    shared_expert_intermediate_size: int = 0
    ep_partitions: int = 1
    rms_norm_eps: float = 1e-5
    rope_theta: float = 10000.0
    rope_scaling: dict[str, Any] | None = None
    qkv_bias: bool = False
    sliding_window: int | None = None


@dataclass
class DiTBlock(BlockStage):
    """Diffusion Transformer (DiT) block with AdaLN-Zero (`shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp`)."""

    mlp_dim: int = 1536
    num_attention_heads: int = 6
    head_dim: int = 64
    layer_norm_eps: float = 1e-6


@dataclass
class OutputStage:
    """Base model output head stage."""

    hidden_size: int


@dataclass
class LMHeadStage(OutputStage):
    """Final normalization + vocabulary LM projection (optionally tied with input embeddings)."""

    vocab_size: int = 32000
    rms_norm_eps: float = 1e-5
    norm_type: Literal["rmsnorm", "layernorm"] = "rmsnorm"
    tie_word_embeddings: bool = False


@dataclass
class VisionProjectionStage(OutputStage):
    """2-layer GELU MLP multimodal projector (LLaVA-style)."""

    out_hidden_size: int = 4096
    layer_norm_eps: float = 1e-5


@dataclass
class ClassifierHeadStage(OutputStage):
    """Vision classification head (`LN -> pool -> linear`)."""

    num_classes: int = 1000
    layer_norm_eps: float = 1e-5
    pool_type: Literal["cls", "mean", "none"] = "cls"


@dataclass
class EncoderOutputStage(OutputStage):
    """Final LayerNorm sequence output stage for audio/vision encoders (`(B, S, D)`)."""

    layer_norm_eps: float = 1e-5


@dataclass
class DiffusionOutputStage(OutputStage):
    """DiT final AdaLN + linear patch unpatchify stage (`(B, C_out, H, W)`)."""

    image_size: int = 32
    patch_size: int = 2
    out_channels: int = 4
    layer_norm_eps: float = 1e-6


# ---------------------------------------------------------------------------
# UnifiedModelGraph + TFG Lowering
# ---------------------------------------------------------------------------


@dataclass
class UnifiedModelGraph:
    """Architecture-agnostic declarative IR for LLMs, MoE, Vision, and Multimodal models."""

    name: str
    family: str
    modality: Modality
    input_stage: InputStage
    blocks: list[BlockStage]
    output_stage: OutputStage
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def num_layers(self) -> int:
        return len(self.blocks)

    def estimate_param_count(self) -> int:
        """Estimate total parameter count across input, block stack, and output stages."""
        total = 0
        if isinstance(self.input_stage, TextEmbeddingStage):
            total += self.input_stage.vocab_size * self.input_stage.hidden_size
        elif isinstance(self.input_stage, VisionPatchEmbedStage):
            p = self.input_stage.patch_size
            total += (
                self.input_stage.hidden_size * self.input_stage.in_channels * p * p
                + self.input_stage.num_tokens * self.input_stage.hidden_size
            )
        elif isinstance(self.input_stage, AudioConvEmbedStage):
            d = self.input_stage.hidden_size
            total += (
                d * self.input_stage.num_mel_bins * 3
                + d * d * 3
                + self.input_stage.max_source_positions * d
            )
        elif isinstance(self.input_stage, DiffusionPatchEmbedStage):
            d = self.input_stage.hidden_size
            p = self.input_stage.patch_size
            total += (
                d * self.input_stage.in_channels * p * p
                + self.input_stage.num_patches * d
                + 2 * d * d
            )
        elif isinstance(self.input_stage, MultimodalFusionStage):
            total += self.input_stage.vocab_size * self.input_stage.hidden_size
            vs = self.input_stage.vision_stage
            total += (
                vs.hidden_size * vs.in_channels * vs.patch_size * vs.patch_size
                + vs.num_tokens * vs.hidden_size
            )
            for vb in self.input_stage.vision_blocks:
                total += 4 * vb.hidden_size * vb.hidden_size + 2 * vb.hidden_size * vb.mlp_dim
            proj_h = self.input_stage.projector_hidden_size or self.input_stage.hidden_size
            total += vs.hidden_size * proj_h + proj_h * self.input_stage.hidden_size

        for blk in self.blocks:
            d = blk.hidden_size
            if isinstance(blk, DenseTransformerBlock):
                kv = blk.num_key_value_heads * blk.head_dim
                q_dim = blk.num_attention_heads * blk.head_dim
                total += d * q_dim + 2 * d * kv + q_dim * d + 3 * d * blk.intermediate_size
            elif isinstance(blk, SparseMoEBlock):
                kv = blk.num_key_value_heads * blk.head_dim
                q_dim = blk.num_attention_heads * blk.head_dim
                total += d * q_dim + 2 * d * kv + q_dim * d
                total += d * blk.num_experts
                total += blk.num_experts * 3 * d * blk.intermediate_size
                if blk.shared_expert_intermediate_size > 0:
                    total += 3 * d * blk.shared_expert_intermediate_size
            elif isinstance(blk, ViTEncoderBlock):
                total += 4 * d * d + 2 * d * blk.mlp_dim
            elif isinstance(blk, DiTBlock):
                total += 10 * d * d + 2 * d * blk.mlp_dim

        if isinstance(self.output_stage, LMHeadStage):
            if not self.output_stage.tie_word_embeddings:
                total += self.output_stage.hidden_size * self.output_stage.vocab_size
        elif isinstance(self.output_stage, VisionProjectionStage):
            total += self.output_stage.hidden_size * self.output_stage.out_hidden_size * 2
        elif isinstance(self.output_stage, ClassifierHeadStage):
            total += self.output_stage.hidden_size * self.output_stage.num_classes
        elif isinstance(self.output_stage, EncoderOutputStage):
            total += 2 * self.output_stage.hidden_size
        elif isinstance(self.output_stage, DiffusionOutputStage):
            d = self.output_stage.hidden_size
            p = self.output_stage.patch_size
            total += 2 * d * d + d * (p * p * self.output_stage.out_channels)
        return total

    def to_fusion_graph(
        self,
        batch_size: int = 1,
        seq_len: int = 16,
        dtype: str = "float16",
        max_lowered_blocks: int = 4,
    ) -> FusionGraph:
        """Lower this declarative model graph into a validated Tile-Fusion Graph (`FusionGraph`)."""
        tokens = max(1, batch_size * seq_len)
        g = FusionGraph(name=f"umg_{self.name}")

        def _connect(src_id: str, dst_id: str, tensor: DataTensor) -> None:
            g.add_edge(src_id, dst_id, tensor)

        prev_id: str
        prev_tensor: DataTensor

        if isinstance(self.input_stage, MultimodalFusionStage):
            vs = self.input_stage.vision_stage
            img_t = DataTensor(
                "image_in", (batch_size, vs.in_channels, vs.image_size, vs.image_size), dtype, "hbm"
            )
            vis_tok = max(1, batch_size * vs.num_tokens)
            patch_out = DataTensor("patch_embeds", (vis_tok, vs.hidden_size), dtype, "hbm")
            g.add_node(
                TileNode(
                    node_id="input_vision_patch",
                    op_type=OpType.INPUT,
                    name="VisionPatchEmbed",
                    inputs=[img_t],
                    outputs=[patch_out],
                    attrs={"patch_size": vs.patch_size, "image_size": vs.image_size},
                )
            )
            proj_out = DataTensor(
                "vision_projected", (vis_tok, self.input_stage.hidden_size), dtype, "hbm"
            )
            g.add_node(
                TileNode(
                    node_id="vision_projector",
                    op_type=OpType.GEMM,
                    name="VisionProjectorMLP",
                    inputs=[patch_out],
                    outputs=[proj_out],
                    attrs={"M": vis_tok, "N": self.input_stage.hidden_size, "K": vs.hidden_size},
                )
            )
            _connect("input_vision_patch", "vision_projector", patch_out)
            merged_tokens = vis_tok + tokens
            prev_tensor = DataTensor(
                "mm_tokens", (merged_tokens, self.input_stage.hidden_size), dtype, "hbm"
            )
            g.add_node(
                TileNode(
                    node_id="mm_concat",
                    op_type=OpType.CUSTOM,
                    name="MultimodalConcat",
                    inputs=[proj_out],
                    outputs=[prev_tensor],
                    attrs={"vision_tokens": vis_tok, "text_tokens": tokens},
                )
            )
            _connect("vision_projector", "mm_concat", proj_out)
            prev_id = "mm_concat"
            eff_tokens = merged_tokens
        elif isinstance(self.input_stage, VisionPatchEmbedStage):
            vis_tok = max(1, batch_size * self.input_stage.num_tokens)
            prev_tensor = DataTensor(
                "patch_embeds", (vis_tok, self.input_stage.hidden_size), dtype, "hbm"
            )
            g.add_node(
                TileNode(
                    node_id="input_patch_embed",
                    op_type=OpType.INPUT,
                    name="VisionPatchEmbed",
                    inputs=[],
                    outputs=[prev_tensor],
                    attrs={"patch_size": self.input_stage.patch_size},
                )
            )
            prev_id = "input_patch_embed"
            eff_tokens = vis_tok
        elif isinstance(self.input_stage, AudioConvEmbedStage):
            aud_tok = max(1, batch_size * seq_len)
            prev_tensor = DataTensor(
                "audio_embeds", (aud_tok, self.input_stage.hidden_size), dtype, "hbm"
            )
            g.add_node(
                TileNode(
                    node_id="input_audio_conv",
                    op_type=OpType.INPUT,
                    name="AudioConvEmbed",
                    inputs=[],
                    outputs=[prev_tensor],
                    attrs={"num_mel_bins": self.input_stage.num_mel_bins},
                )
            )
            prev_id = "input_audio_conv"
            eff_tokens = aud_tok
        elif isinstance(self.input_stage, DiffusionPatchEmbedStage):
            dif_tok = max(1, batch_size * self.input_stage.num_patches)
            prev_tensor = DataTensor(
                "diffusion_patches", (dif_tok, self.input_stage.hidden_size), dtype, "hbm"
            )
            g.add_node(
                TileNode(
                    node_id="input_diffusion_patch",
                    op_type=OpType.INPUT,
                    name="DiffusionPatchEmbed",
                    inputs=[],
                    outputs=[prev_tensor],
                    attrs={"patch_size": self.input_stage.patch_size},
                )
            )
            prev_id = "input_diffusion_patch"
            eff_tokens = dif_tok
        else:
            prev_tensor = DataTensor(
                "hidden_in", (tokens, self.input_stage.hidden_size), dtype, "hbm"
            )
            g.add_node(
                TileNode(
                    node_id="input_embed",
                    op_type=OpType.INPUT,
                    name="TextEmbedding",
                    inputs=[],
                    outputs=[prev_tensor],
                    attrs={"vocab_size": getattr(self.input_stage, "vocab_size", 32000)},
                )
            )
            prev_id = "input_embed"
            eff_tokens = tokens

        lowered_blocks = self.blocks[:max_lowered_blocks]
        for idx, blk in enumerate(lowered_blocks):
            d = blk.hidden_size
            prefix = f"B{idx}"

            if isinstance(blk, (DenseTransformerBlock, SparseMoEBlock)):
                qkv_dim = (blk.num_attention_heads + 2 * blk.num_key_value_heads) * blk.head_dim
                t_qkv = DataTensor(f"{prefix}_qkv", (eff_tokens, qkv_dim), dtype, "hbm")
                t_attn = DataTensor(f"{prefix}_attn", (eff_tokens, d), dtype, "hbm")
                t_o = DataTensor(f"{prefix}_o", (eff_tokens, d), dtype, "hbm")
                t_ln1 = DataTensor(f"{prefix}_ln1", (eff_tokens, d), dtype, "hbm")
                t_up = DataTensor(
                    f"{prefix}_ffn_up", (eff_tokens, blk.intermediate_size), dtype, "hbm"
                )
                t_act = DataTensor(
                    f"{prefix}_silu", (eff_tokens, blk.intermediate_size), dtype, "hbm"
                )
                t_down = DataTensor(f"{prefix}_ffn_down", (eff_tokens, d), dtype, "hbm")
                t_out = DataTensor(f"{prefix}_out", (eff_tokens, d), dtype, "hbm")

                n_qkv = f"{prefix}_qkv"
                n_attn = f"{prefix}_attn"
                n_o = f"{prefix}_o"
                n_ln1 = f"{prefix}_ln1"
                n_up = f"{prefix}_ffn_up"
                n_act = f"{prefix}_silu"
                n_down = f"{prefix}_ffn_down"
                n_out = f"{prefix}_ln_res"

                g.add_node(
                    TileNode(
                        node_id=n_qkv,
                        op_type=OpType.QKV_GEMM,
                        name="QKV GEMM",
                        inputs=[prev_tensor],
                        outputs=[t_qkv],
                        attrs={"M": eff_tokens, "N": qkv_dim, "K": d, "qkv_bias": blk.qkv_bias},
                    )
                )
                _connect(prev_id, n_qkv, prev_tensor)

                g.add_node(
                    TileNode(
                        node_id=n_attn,
                        op_type=OpType.ATTENTION_GQA,
                        name="GQA Attention",
                        inputs=[t_qkv],
                        outputs=[t_attn],
                        attrs={
                            "n_heads": blk.num_attention_heads,
                            "n_kv_heads": blk.num_key_value_heads,
                            "head_dim": blk.head_dim,
                            "rope_theta": blk.rope_theta,
                            "sliding_window": blk.sliding_window,
                        },
                    )
                )
                _connect(n_qkv, n_attn, t_qkv)

                g.add_node(
                    TileNode(
                        node_id=n_o,
                        op_type=OpType.O_GEMM,
                        name="O GEMM",
                        inputs=[t_attn],
                        outputs=[t_o],
                        attrs={
                            "M": eff_tokens,
                            "N": d,
                            "K": blk.num_attention_heads * blk.head_dim,
                        },
                    )
                )
                _connect(n_attn, n_o, t_attn)

                g.add_node(
                    TileNode(
                        node_id=n_ln1,
                        op_type=OpType.LAYERNORM,
                        name="Norm",
                        inputs=[t_o],
                        outputs=[t_ln1],
                        attrs={"D": d},
                    )
                )
                _connect(n_o, n_ln1, t_o)

                ffn_attrs: dict[str, Any] = {
                    "M": eff_tokens,
                    "N": blk.intermediate_size,
                    "K": d,
                }
                if isinstance(blk, SparseMoEBlock):
                    ffn_attrs["num_experts"] = blk.num_experts
                    ffn_attrs["top_k_experts"] = blk.top_k_experts
                    ffn_attrs["shared_expert_intermediate_size"] = (
                        blk.shared_expert_intermediate_size
                    )

                g.add_node(
                    TileNode(
                        node_id=n_up,
                        op_type=OpType.FFN_UP_GEMM,
                        name="MoE Gate+Up" if isinstance(blk, SparseMoEBlock) else "FFN-up GEMM",
                        inputs=[t_ln1],
                        outputs=[t_up],
                        attrs=ffn_attrs,
                    )
                )
                _connect(n_ln1, n_up, t_ln1)

                act_op = (
                    OpType.GELU
                    if isinstance(blk, DenseTransformerBlock) and blk.activation == "gelu"
                    else OpType.SILU
                )
                g.add_node(
                    TileNode(
                        node_id=n_act,
                        op_type=act_op,
                        name="Act",
                        inputs=[t_up],
                        outputs=[t_act],
                        attrs={"D": blk.intermediate_size},
                    )
                )
                _connect(n_up, n_act, t_up)

                g.add_node(
                    TileNode(
                        node_id=n_down,
                        op_type=OpType.FFN_DOWN_GEMM,
                        name="MoE Down" if isinstance(blk, SparseMoEBlock) else "FFN-down GEMM",
                        inputs=[t_act],
                        outputs=[t_down],
                        attrs={"M": eff_tokens, "N": d, "K": blk.intermediate_size},
                    )
                )
                _connect(n_act, n_down, t_act)

                g.add_node(
                    TileNode(
                        node_id=n_out,
                        op_type=OpType.LAYERNORM_RESIDUAL,
                        name="PostNorm+Residual",
                        inputs=[t_down, t_o],
                        outputs=[t_out],
                        attrs={"D": d},
                    )
                )
                _connect(n_down, n_out, t_down)
                _connect(n_o, n_out, t_o)
                prev_id, prev_tensor = n_out, t_out

            elif isinstance(blk, (ViTEncoderBlock, DiTBlock)):
                t_ln1 = DataTensor(f"{prefix}_vit_ln1", (eff_tokens, d), dtype, "hbm")
                t_qkv = DataTensor(f"{prefix}_vit_qkv", (eff_tokens, 3 * d), dtype, "hbm")
                t_attn = DataTensor(f"{prefix}_vit_attn", (eff_tokens, d), dtype, "hbm")
                t_up = DataTensor(f"{prefix}_vit_fc1", (eff_tokens, blk.mlp_dim), dtype, "hbm")
                t_gelu = DataTensor(f"{prefix}_vit_gelu", (eff_tokens, blk.mlp_dim), dtype, "hbm")
                t_out = DataTensor(f"{prefix}_vit_out", (eff_tokens, d), dtype, "hbm")

                g.add_node(
                    TileNode(f"{prefix}_ln1", OpType.LAYERNORM, "ViT LN1", [prev_tensor], [t_ln1])
                )
                _connect(prev_id, f"{prefix}_ln1", prev_tensor)
                g.add_node(
                    TileNode(
                        f"{prefix}_qkv",
                        OpType.QKV_GEMM,
                        "ViT QKV",
                        [t_ln1],
                        [t_qkv],
                        attrs={"M": eff_tokens, "N": 3 * d, "K": d},
                    )
                )
                _connect(f"{prefix}_ln1", f"{prefix}_qkv", t_ln1)
                g.add_node(
                    TileNode(
                        f"{prefix}_attn",
                        OpType.ATTENTION_GQA,
                        "ViT MHA",
                        [t_qkv],
                        [t_attn],
                        attrs={"bidirectional": True, "n_heads": blk.num_attention_heads},
                    )
                )
                _connect(f"{prefix}_qkv", f"{prefix}_attn", t_qkv)
                g.add_node(
                    TileNode(
                        f"{prefix}_fc1",
                        OpType.GEMM,
                        "ViT MLP FC1",
                        [t_attn],
                        [t_up],
                        attrs={"M": eff_tokens, "N": blk.mlp_dim, "K": d},
                    )
                )
                _connect(f"{prefix}_attn", f"{prefix}_fc1", t_attn)
                g.add_node(TileNode(f"{prefix}_gelu", OpType.GELU, "ViT GELU", [t_up], [t_gelu]))
                _connect(f"{prefix}_fc1", f"{prefix}_gelu", t_up)
                g.add_node(
                    TileNode(
                        f"{prefix}_fc2",
                        OpType.GEMM,
                        "ViT MLP FC2",
                        [t_gelu],
                        [t_out],
                        attrs={"M": eff_tokens, "N": d, "K": blk.mlp_dim},
                    )
                )
                _connect(f"{prefix}_gelu", f"{prefix}_fc2", t_gelu)
                prev_id, prev_tensor = f"{prefix}_fc2", t_out

        if isinstance(self.output_stage, LMHeadStage):
            out_t = DataTensor("logits", (eff_tokens, self.output_stage.vocab_size), dtype, "hbm")
            g.add_node(
                TileNode(
                    node_id="output_lm_head",
                    op_type=OpType.GEMM,
                    name="LMHead",
                    inputs=[prev_tensor],
                    outputs=[out_t],
                    attrs={
                        "M": eff_tokens,
                        "N": self.output_stage.vocab_size,
                        "K": self.output_stage.hidden_size,
                    },
                )
            )
            _connect(prev_id, "output_lm_head", prev_tensor)
            g.graph_outputs = ["output_lm_head"]
        elif isinstance(self.output_stage, VisionProjectionStage):
            out_t = DataTensor(
                "mm_projected", (eff_tokens, self.output_stage.out_hidden_size), dtype, "hbm"
            )
            g.add_node(
                TileNode(
                    node_id="output_vision_proj",
                    op_type=OpType.GEMM,
                    name="VisionProjector",
                    inputs=[prev_tensor],
                    outputs=[out_t],
                    attrs={
                        "M": eff_tokens,
                        "N": self.output_stage.out_hidden_size,
                        "K": self.output_stage.hidden_size,
                    },
                )
            )
            _connect(prev_id, "output_vision_proj", prev_tensor)
            g.graph_outputs = ["output_vision_proj"]
        elif isinstance(self.output_stage, ClassifierHeadStage):
            out_t = DataTensor(
                "class_logits", (batch_size, self.output_stage.num_classes), dtype, "hbm"
            )
            g.add_node(
                TileNode(
                    node_id="output_classifier",
                    op_type=OpType.GEMM,
                    name="ClassifierHead",
                    inputs=[prev_tensor],
                    outputs=[out_t],
                    attrs={
                        "M": batch_size,
                        "N": self.output_stage.num_classes,
                        "K": self.output_stage.hidden_size,
                    },
                )
            )
            _connect(prev_id, "output_classifier", prev_tensor)
            g.graph_outputs = ["output_classifier"]
        elif isinstance(self.output_stage, (EncoderOutputStage, DiffusionOutputStage)):
            out_t = DataTensor(
                "encoder_out", (eff_tokens, self.output_stage.hidden_size), dtype, "hbm"
            )
            g.add_node(
                TileNode(
                    node_id="output_encoder_ln",
                    op_type=OpType.LAYERNORM,
                    name="EncoderOutputNorm",
                    inputs=[prev_tensor],
                    outputs=[out_t],
                    attrs={"D": self.output_stage.hidden_size},
                )
            )
            _connect(prev_id, "output_encoder_ln", prev_tensor)
            g.graph_outputs = ["output_encoder_ln"]

        g.validate()
        return g

    def compile_tfg(
        self,
        batch_size: int = 1,
        seq_len: int = 16,
        dtype: str = "float16",
    ) -> tuple[FusionGraph, PassReport]:
        """Lower to TFG and execute the default CuTileForge optimization pipeline."""
        tfg = self.to_fusion_graph(batch_size=batch_size, seq_len=seq_len, dtype=dtype)
        report = default_pass_manager().run(tfg)
        return tfg, report

    def summary(self) -> dict[str, Any]:
        block_types = sorted({type(b).__name__ for b in self.blocks})
        return {
            "name": self.name,
            "family": self.family,
            "modality": self.modality,
            "num_layers": self.num_layers,
            "block_types": block_types,
            "input_stage": type(self.input_stage).__name__,
            "output_stage": type(self.output_stage).__name__,
            "estimated_params": self.estimate_param_count(),
        }


# ---------------------------------------------------------------------------
# Weight Initialization & Portable Executor (CUDA CuTile + Reference PyTorch)
# ---------------------------------------------------------------------------


def _rand_weight(
    shape: tuple[int, ...],
    gen: torch.Generator,
    device: torch.device,
    dtype: torch.dtype,
    scale: float = 0.05,
) -> torch.Tensor:
    w = torch.randn(shape, generator=gen, dtype=torch.float32) * scale
    return w.to(device=device, dtype=dtype)


def _init_block_weights(
    blk: BlockStage,
    gen: torch.Generator,
    device: torch.device,
    dtype: torch.dtype,
) -> dict[str, Any]:
    d = blk.hidden_size
    if isinstance(blk, DenseTransformerBlock):
        q_dim = blk.num_attention_heads * blk.head_dim
        kv_dim = blk.num_key_value_heads * blk.head_dim
        bw: dict[str, Any] = {
            "in_norm_w": torch.ones(d, device=device, dtype=dtype),
            "post_norm_w": torch.ones(d, device=device, dtype=dtype),
            "wq": _rand_weight((d, q_dim), gen, device, dtype),
            "wk": _rand_weight((d, kv_dim), gen, device, dtype),
            "wv": _rand_weight((d, kv_dim), gen, device, dtype),
            "wo": _rand_weight((q_dim, d), gen, device, dtype),
            "w_gate": _rand_weight((d, blk.intermediate_size), gen, device, dtype),
            "w_up": _rand_weight((d, blk.intermediate_size), gen, device, dtype),
            "w_down": _rand_weight((blk.intermediate_size, d), gen, device, dtype),
        }
        if blk.norm_type == "layernorm":
            bw["in_norm_b"] = torch.zeros(d, device=device, dtype=dtype)
            bw["post_norm_b"] = torch.zeros(d, device=device, dtype=dtype)
        if blk.qkv_bias:
            bw["bq"] = _rand_weight((q_dim,), gen, device, dtype, 0.02)
            bw["bk"] = _rand_weight((kv_dim,), gen, device, dtype, 0.02)
            bw["bv"] = _rand_weight((kv_dim,), gen, device, dtype, 0.02)
        return bw

    if isinstance(blk, SparseMoEBlock):
        q_dim = blk.num_attention_heads * blk.head_dim
        kv_dim = blk.num_key_value_heads * blk.head_dim
        experts = [
            {
                "w_gate": _rand_weight((d, blk.intermediate_size), gen, device, dtype),
                "w_up": _rand_weight((d, blk.intermediate_size), gen, device, dtype),
                "w_down": _rand_weight((blk.intermediate_size, d), gen, device, dtype),
            }
            for _ in range(blk.num_experts)
        ]
        bw = {
            "in_norm_w": torch.ones(d, device=device, dtype=dtype),
            "post_norm_w": torch.ones(d, device=device, dtype=dtype),
            "wq": _rand_weight((d, q_dim), gen, device, dtype),
            "wk": _rand_weight((d, kv_dim), gen, device, dtype),
            "wv": _rand_weight((d, kv_dim), gen, device, dtype),
            "wo": _rand_weight((q_dim, d), gen, device, dtype),
            "router": _rand_weight((d, blk.num_experts), gen, device, dtype),
            "experts": experts,
        }
        if blk.qkv_bias:
            bw["bq"] = _rand_weight((q_dim,), gen, device, dtype, 0.02)
            bw["bk"] = _rand_weight((kv_dim,), gen, device, dtype, 0.02)
            bw["bv"] = _rand_weight((kv_dim,), gen, device, dtype, 0.02)
        if blk.shared_expert_intermediate_size > 0:
            bw["shared_expert"] = {
                "w_gate": _rand_weight(
                    (d, blk.shared_expert_intermediate_size), gen, device, dtype
                ),
                "w_up": _rand_weight((d, blk.shared_expert_intermediate_size), gen, device, dtype),
                "w_down": _rand_weight(
                    (blk.shared_expert_intermediate_size, d), gen, device, dtype
                ),
            }
        return bw

    if isinstance(blk, ViTEncoderBlock):
        return {
            "ln1_w": torch.ones(d, device=device, dtype=dtype),
            "ln1_b": torch.zeros(d, device=device, dtype=dtype),
            "ln2_w": torch.ones(d, device=device, dtype=dtype),
            "ln2_b": torch.zeros(d, device=device, dtype=dtype),
            "wqkv": _rand_weight((d, 3 * d), gen, device, dtype),
            "bqkv": torch.zeros(3 * d, device=device, dtype=dtype),
            "wo": _rand_weight((d, d), gen, device, dtype),
            "bo": torch.zeros(d, device=device, dtype=dtype),
            "fc1_w": _rand_weight((d, blk.mlp_dim), gen, device, dtype),
            "fc1_b": torch.zeros(blk.mlp_dim, device=device, dtype=dtype),
            "fc2_w": _rand_weight((blk.mlp_dim, d), gen, device, dtype),
            "fc2_b": torch.zeros(d, device=device, dtype=dtype),
        }

    if isinstance(blk, DiTBlock):
        return {
            "adaln_w": _rand_weight((d, 6 * d), gen, device, dtype, 0.01),
            "adaln_b": torch.zeros(6 * d, device=device, dtype=dtype),
            "wqkv": _rand_weight((d, 3 * d), gen, device, dtype),
            "bqkv": torch.zeros(3 * d, device=device, dtype=dtype),
            "wo": _rand_weight((d, d), gen, device, dtype),
            "bo": torch.zeros(d, device=device, dtype=dtype),
            "fc1_w": _rand_weight((d, blk.mlp_dim), gen, device, dtype),
            "fc1_b": torch.zeros(blk.mlp_dim, device=device, dtype=dtype),
            "fc2_w": _rand_weight((blk.mlp_dim, d), gen, device, dtype),
            "fc2_b": torch.zeros(d, device=device, dtype=dtype),
        }

    raise TypeError(f"Unsupported block type: {type(blk)}")


def init_synthetic_weights(
    graph: UnifiedModelGraph,
    seed: int = 42,
    device: torch.device | str = "cpu",
    dtype: torch.dtype | None = None,
) -> dict[str, Any]:
    """Create deterministic synthetic weights matching `graph` stages."""
    dev = torch.device(device)
    dt = dtype or (torch.float16 if dev.type == "cuda" else torch.float32)
    gen = torch.Generator(device="cpu")
    gen.manual_seed(seed)

    weights: dict[str, Any] = {"blocks": []}

    def _init_vision_stage(vs: VisionPatchEmbedStage) -> dict[str, Any]:
        vw: dict[str, Any] = {
            "patch_w": _rand_weight(
                (vs.hidden_size, vs.in_channels, vs.patch_size, vs.patch_size), gen, dev, dt
            ),
            "patch_b": torch.zeros(vs.hidden_size, device=dev, dtype=dt),
            "pos_embed": _rand_weight((1, vs.num_tokens, vs.hidden_size), gen, dev, dt, 0.02),
        }
        if vs.use_cls_token:
            vw["cls_token"] = _rand_weight((1, 1, vs.hidden_size), gen, dev, dt, 0.02)
        if vs.pre_layer_norm:
            vw["pre_ln_w"] = torch.ones(vs.hidden_size, device=dev, dtype=dt)
            vw["pre_ln_b"] = torch.zeros(vs.hidden_size, device=dev, dtype=dt)
        return vw

    if isinstance(graph.input_stage, TextEmbeddingStage):
        weights["embed_tokens"] = _rand_weight(
            (graph.input_stage.vocab_size, graph.input_stage.hidden_size), gen, dev, dt
        )
    elif isinstance(graph.input_stage, VisionPatchEmbedStage):
        weights["vision_embed"] = _init_vision_stage(graph.input_stage)
    elif isinstance(graph.input_stage, AudioConvEmbedStage):
        ast = graph.input_stage
        d = ast.hidden_size
        weights["audio_embed"] = {
            "conv1_w": _rand_weight((d, ast.num_mel_bins, 3), gen, dev, dt),
            "conv1_b": torch.zeros(d, device=dev, dtype=dt),
            "conv2_w": _rand_weight((d, d, 3), gen, dev, dt),
            "conv2_b": torch.zeros(d, device=dev, dtype=dt),
            "pos_embed": _rand_weight((ast.max_source_positions, d), gen, dev, dt, 0.02),
        }
    elif isinstance(graph.input_stage, DiffusionPatchEmbedStage):
        dst = graph.input_stage
        d = dst.hidden_size
        weights["diffusion_embed"] = {
            "patch_w": _rand_weight(
                (d, dst.in_channels, dst.patch_size, dst.patch_size), gen, dev, dt
            ),
            "patch_b": torch.zeros(d, device=dev, dtype=dt),
            "pos_embed": _rand_weight((1, dst.num_patches, d), gen, dev, dt, 0.02),
            "t_fc1_w": _rand_weight((d, d), gen, dev, dt),
            "t_fc1_b": torch.zeros(d, device=dev, dtype=dt),
            "t_fc2_w": _rand_weight((d, d), gen, dev, dt),
            "t_fc2_b": torch.zeros(d, device=dev, dtype=dt),
        }
    elif isinstance(graph.input_stage, MultimodalFusionStage):
        weights["embed_tokens"] = _rand_weight(
            (graph.input_stage.vocab_size, graph.input_stage.hidden_size), gen, dev, dt
        )
        weights["vision_embed"] = _init_vision_stage(graph.input_stage.vision_stage)
        weights["vision_blocks"] = [
            _init_block_weights(vb, gen, dev, dt) for vb in graph.input_stage.vision_blocks
        ]
        vs_h = graph.input_stage.vision_stage.hidden_size
        proj_h = graph.input_stage.projector_hidden_size or graph.input_stage.hidden_size
        weights["projector"] = {
            "fc1_w": _rand_weight((vs_h, proj_h), gen, dev, dt),
            "fc1_b": torch.zeros(proj_h, device=dev, dtype=dt),
            "fc2_w": _rand_weight((proj_h, graph.input_stage.hidden_size), gen, dev, dt),
            "fc2_b": torch.zeros(graph.input_stage.hidden_size, device=dev, dtype=dt),
        }

    weights["blocks"] = [_init_block_weights(b, gen, dev, dt) for b in graph.blocks]

    if isinstance(graph.output_stage, LMHeadStage):
        weights["norm_w"] = torch.ones(graph.output_stage.hidden_size, device=dev, dtype=dt)
        if graph.output_stage.norm_type == "layernorm":
            weights["norm_b"] = torch.zeros(graph.output_stage.hidden_size, device=dev, dtype=dt)
        if graph.output_stage.tie_word_embeddings and "embed_tokens" in weights:
            weights["lm_head"] = weights["embed_tokens"].t().contiguous()
        else:
            weights["lm_head"] = _rand_weight(
                (graph.output_stage.hidden_size, graph.output_stage.vocab_size), gen, dev, dt
            )
    elif isinstance(graph.output_stage, VisionProjectionStage):
        h = graph.output_stage.hidden_size
        oh = graph.output_stage.out_hidden_size
        weights["norm_w"] = torch.ones(h, device=dev, dtype=dt)
        weights["norm_b"] = torch.zeros(h, device=dev, dtype=dt)
        weights["proj_fc1_w"] = _rand_weight((h, oh), gen, dev, dt)
        weights["proj_fc1_b"] = torch.zeros(oh, device=dev, dtype=dt)
        weights["proj_fc2_w"] = _rand_weight((oh, oh), gen, dev, dt)
        weights["proj_fc2_b"] = torch.zeros(oh, device=dev, dtype=dt)
    elif isinstance(graph.output_stage, ClassifierHeadStage):
        weights["norm_w"] = torch.ones(graph.output_stage.hidden_size, device=dev, dtype=dt)
        weights["norm_b"] = torch.zeros(graph.output_stage.hidden_size, device=dev, dtype=dt)
        weights["cls_w"] = _rand_weight(
            (graph.output_stage.hidden_size, graph.output_stage.num_classes), gen, dev, dt
        )
        weights["cls_b"] = torch.zeros(graph.output_stage.num_classes, device=dev, dtype=dt)
    elif isinstance(graph.output_stage, EncoderOutputStage):
        weights["norm_w"] = torch.ones(graph.output_stage.hidden_size, device=dev, dtype=dt)
        weights["norm_b"] = torch.zeros(graph.output_stage.hidden_size, device=dev, dtype=dt)
    elif isinstance(graph.output_stage, DiffusionOutputStage):
        d = graph.output_stage.hidden_size
        out_dim = (
            graph.output_stage.patch_size
            * graph.output_stage.patch_size
            * graph.output_stage.out_channels
        )
        weights["adaln_out_w"] = _rand_weight((d, 2 * d), gen, dev, dt, 0.01)
        weights["adaln_out_b"] = torch.zeros(2 * d, device=dev, dtype=dt)
        weights["proj_out_w"] = _rand_weight((d, out_dim), gen, dev, dt, 0.01)
        weights["proj_out_b"] = torch.zeros(out_dim, device=dev, dtype=dt)

    return weights


def _apply_rope_torch(
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    rotary_style: str = "interleaved",
) -> torch.Tensor:
    bsz, n_heads, seq_len, d_head = x.shape
    half_d = d_head // 2
    c = cos[:seq_len, :half_d].float().view(1, 1, seq_len, half_d)
    s = sin[:seq_len, :half_d].float().view(1, 1, seq_len, half_d)
    if rotary_style == "half":
        x0 = x[..., :half_d].float()
        x1 = x[..., half_d:].float()
        o0 = x0 * c - x1 * s
        o1 = x1 * c + x0 * s
        return torch.cat([o0, o1], dim=-1).to(x.dtype)
    x_pair = x.float().view(bsz, n_heads, seq_len, 2, half_d)
    x0 = x_pair[:, :, :, 0, :]
    x1 = x_pair[:, :, :, 1, :]
    o0 = x0 * c - x1 * s
    o1 = x1 * c + x0 * s
    return torch.stack([o0, o1], dim=3).reshape(bsz, n_heads, seq_len, d_head).to(x.dtype)


class UnifiedModelExecutor:
    """Portable executor running `UnifiedModelGraph` via CuTile kernels (CUDA) or PyTorch."""

    def __init__(self, graph: UnifiedModelGraph) -> None:
        self.graph = graph

    def _rms_norm(
        self, x: torch.Tensor, w: torch.Tensor, eps: float, use_cutile: bool
    ) -> torch.Tensor:
        if use_cutile and x.is_cuda and x.dtype == torch.float16:
            return cutile_rmsnorm(x, w, eps=eps)
        x_f = x.float()
        var = x_f.pow(2).mean(dim=-1, keepdim=True)
        return (x_f * torch.rsqrt(var + eps)).to(x.dtype) * w

    def _layer_norm(
        self,
        x: torch.Tensor,
        w: torch.Tensor,
        b: torch.Tensor | None,
        eps: float,
        use_cutile: bool,
    ) -> torch.Tensor:
        if use_cutile and x.is_cuda and x.dtype == torch.float16:
            return cutile_layernorm(x, w, bias=b, eps=eps)
        return F.layer_norm(
            x.float(), (x.shape[-1],), w.float(), b.float() if b is not None else None, eps
        ).to(x.dtype)

    def _linear(
        self, x: torch.Tensor, w: torch.Tensor, bias: torch.Tensor | None, use_cutile: bool
    ) -> torch.Tensor:
        shape_prefix = x.shape[:-1]
        x_2d = x.reshape(-1, x.shape[-1])
        if (
            use_cutile
            and x_2d.is_cuda
            and x_2d.dtype == torch.float16
            and w.dtype == torch.float16
            and x_2d.shape[1] >= 16
            and w.shape[1] >= 16
        ):
            out = cutile_gemm_fp16(x_2d, w)
        else:
            out = torch.matmul(x_2d.float(), w.float()).to(x.dtype)
        if bias is not None:
            out = out + bias
        return out.view(*shape_prefix, w.shape[-1])

    def _act(self, x: torch.Tensor, kind: str, use_cutile: bool) -> torch.Tensor:
        if use_cutile and x.is_cuda and x.dtype in (torch.float16, torch.bfloat16, torch.float32):
            return cutile_gelu(x) if kind == "gelu" else cutile_silu(x)
        return (
            F.gelu(x.float(), approximate="tanh").to(x.dtype)
            if kind == "gelu"
            else F.silu(x.float()).to(x.dtype)
        )

    def _run_vision_embed(
        self,
        pixel_values: torch.Tensor,
        vs: VisionPatchEmbedStage,
        vw: dict[str, Any],
        use_cutile: bool = True,
    ) -> torch.Tensor:
        bsz = pixel_values.shape[0]
        patches = F.conv2d(
            pixel_values.float(),
            vw["patch_w"].float(),
            bias=vw["patch_b"].float() if vw.get("patch_b") is not None else None,
            stride=vs.patch_size,
            padding=0,
        )
        tokens = patches.flatten(2).transpose(1, 2).to(pixel_values.dtype)
        if vs.use_cls_token and "cls_token" in vw:
            cls = vw["cls_token"].expand(bsz, -1, -1)
            tokens = torch.cat([cls, tokens], dim=1)
        out = tokens + vw["pos_embed"][:, : tokens.shape[1], :]
        if vs.pre_layer_norm and "pre_ln_w" in vw:
            out = self._layer_norm(
                out, vw["pre_ln_w"], vw.get("pre_ln_b"), vs.layer_norm_eps, use_cutile
            )
        return out

    def _run_audio_embed(
        self,
        mel_features: torch.Tensor,
        ast: AudioConvEmbedStage,
        aw: dict[str, Any],
        use_cutile: bool = True,
    ) -> torch.Tensor:
        c1 = F.conv1d(
            mel_features.float(),
            aw["conv1_w"].float(),
            bias=aw["conv1_b"].float(),
            stride=1,
            padding=1,
        ).to(mel_features.dtype)
        a1 = self._act(c1.transpose(1, 2).contiguous(), "gelu", use_cutile).transpose(1, 2)
        c2 = F.conv1d(
            a1.float(),
            aw["conv2_w"].float(),
            bias=aw["conv2_b"].float(),
            stride=2,
            padding=1,
        ).to(mel_features.dtype)
        a2 = self._act(c2.transpose(1, 2).contiguous(), "gelu", use_cutile)
        seq_len = a2.shape[1]
        return a2 + aw["pos_embed"][:seq_len, :].unsqueeze(0)

    def _timestep_embedding(
        self, timesteps: torch.Tensor, dim: int, dtype: torch.dtype
    ) -> torch.Tensor:
        half = dim // 2
        freqs = torch.exp(
            -math.log(10000.0)
            * torch.arange(0, half, device=timesteps.device, dtype=torch.float32)
            / max(1, half)
        )
        args = timesteps.float().view(-1, 1) * freqs.view(1, -1)
        return torch.cat([torch.cos(args), torch.sin(args)], dim=-1).to(dtype)

    def _run_dit_block(
        self,
        x: torch.Tensor,
        t_cond: torch.Tensor,
        blk: DiTBlock,
        bw: dict[str, Any],
        use_cutile: bool,
    ) -> torch.Tensor:
        bsz, seq_len, d = x.shape
        t_act = self._act(t_cond, "silu", use_cutile)
        mod = self._linear(t_act, bw["adaln_w"], bw["adaln_b"], use_cutile).unsqueeze(1)
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = mod.chunk(6, dim=-1)

        ln1 = F.layer_norm(x.float(), (d,), eps=blk.layer_norm_eps).to(x.dtype)
        h1 = ln1 * (1.0 + scale_msa) + shift_msa
        qkv = self._linear(h1, bw["wqkv"], bw["bqkv"], use_cutile)
        q, k, v = qkv.view(bsz, seq_len, 3, blk.num_attention_heads, blk.head_dim).unbind(dim=2)
        attn = F.scaled_dot_product_attention(
            q.transpose(1, 2).float(),
            k.transpose(1, 2).float(),
            v.transpose(1, 2).float(),
            is_causal=False,
        ).to(x.dtype)
        attn_out = attn.transpose(1, 2).contiguous().view(bsz, seq_len, d)
        x = x + gate_msa * self._linear(attn_out, bw["wo"], bw["bo"], use_cutile)

        ln2 = F.layer_norm(x.float(), (d,), eps=blk.layer_norm_eps).to(x.dtype)
        h2 = ln2 * (1.0 + scale_mlp) + shift_mlp
        mlp_h = self._act(
            self._linear(h2, bw["fc1_w"], bw["fc1_b"], use_cutile), "gelu", use_cutile
        )
        return x + gate_mlp * self._linear(mlp_h, bw["fc2_w"], bw["fc2_b"], use_cutile)

    def _run_vit_block(
        self,
        x: torch.Tensor,
        blk: ViTEncoderBlock,
        bw: dict[str, Any],
        use_cutile: bool,
    ) -> torch.Tensor:
        bsz, seq_len, d = x.shape
        h = self._layer_norm(x, bw["ln1_w"], bw["ln1_b"], blk.layer_norm_eps, use_cutile)
        qkv = self._linear(h, bw["wqkv"], bw["bqkv"], use_cutile)
        q, k, v = qkv.view(bsz, seq_len, 3, blk.num_attention_heads, blk.head_dim).unbind(dim=2)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        attn = F.scaled_dot_product_attention(q.float(), k.float(), v.float(), is_causal=False).to(
            x.dtype
        )
        attn_out = attn.transpose(1, 2).contiguous().view(bsz, seq_len, d)
        x = x + self._linear(attn_out, bw["wo"], bw["bo"], use_cutile)

        h2 = self._layer_norm(x, bw["ln2_w"], bw["ln2_b"], blk.layer_norm_eps, use_cutile)
        mlp_h = self._act(
            self._linear(h2, bw["fc1_w"], bw["fc1_b"], use_cutile), blk.activation, use_cutile
        )
        return x + self._linear(mlp_h, bw["fc2_w"], bw["fc2_b"], use_cutile)

    def _causal_attention(
        self,
        h: torch.Tensor,
        bw: dict[str, Any],
        num_heads: int,
        num_kv_heads: int,
        head_dim: int,
        rope_theta: float,
        rope_scaling: dict[str, Any] | None,
        qkv_bias: bool,
        sliding_window: int | None,
        use_cutile: bool,
    ) -> torch.Tensor:
        bsz, seq_len, _ = h.shape
        q = self._linear(h, bw["wq"], bw.get("bq") if qkv_bias else None, use_cutile)
        k = self._linear(h, bw["wk"], bw.get("bk") if qkv_bias else None, use_cutile)
        v = self._linear(h, bw["wv"], bw.get("bv") if qkv_bias else None, use_cutile)

        q4 = q.view(bsz, seq_len, num_heads, head_dim).transpose(1, 2)
        k4 = k.view(bsz, seq_len, num_kv_heads, head_dim).transpose(1, 2)
        v4 = v.view(bsz, seq_len, num_kv_heads, head_dim).transpose(1, 2)

        cos, sin = compute_scaled_rope_freqs(
            seq_len,
            head_dim,
            theta=rope_theta,
            rope_scaling=rope_scaling,
            device=h.device,
            dtype=h.dtype,
        )
        rotary_style = str(self.graph.metadata.get("rotary_style", "interleaved"))
        q4 = _apply_rope_torch(q4, cos, sin, rotary_style=rotary_style)
        k4 = _apply_rope_torch(k4, cos, sin, rotary_style=rotary_style)

        rep = num_heads // num_kv_heads
        if rep > 1:
            k4 = k4.repeat_interleave(rep, dim=1)
            v4 = v4.repeat_interleave(rep, dim=1)

        scores = torch.matmul(q4.float(), k4.float().transpose(-1, -2)) / math.sqrt(float(head_dim))
        pos = torch.arange(seq_len, device=h.device)
        mask = pos.view(seq_len, 1) >= pos.view(1, seq_len)
        if sliding_window is not None and sliding_window > 0:
            mask = mask & ((pos.view(seq_len, 1) - pos.view(1, seq_len)) < sliding_window)
        scores = scores.masked_fill(~mask.view(1, 1, seq_len, seq_len), float("-inf"))
        probs = torch.softmax(scores, dim=-1)
        attn_out = torch.matmul(probs, v4.float()).to(h.dtype)
        attn_flat = attn_out.transpose(1, 2).contiguous().view(bsz, seq_len, num_heads * head_dim)
        return self._linear(attn_flat, bw["wo"], None, use_cutile)

    def _run_forward(
        self,
        weights: dict[str, Any],
        input_ids: torch.Tensor | None = None,
        pixel_values: torch.Tensor | None = None,
        mel_features: torch.Tensor | None = None,
        timesteps: torch.Tensor | None = None,
        use_cutile: bool = True,
    ) -> torch.Tensor:
        g = self.graph
        t_cond: torch.Tensor | None = None

        if isinstance(g.input_stage, TextEmbeddingStage):
            if input_ids is None:
                raise ValueError("input_ids required for text model")
            hidden = F.embedding(input_ids, weights["embed_tokens"])
        elif isinstance(g.input_stage, VisionPatchEmbedStage):
            if pixel_values is None:
                raise ValueError("pixel_values required for vision model")
            hidden = self._run_vision_embed(
                pixel_values, g.input_stage, weights["vision_embed"], use_cutile=use_cutile
            )
        elif isinstance(g.input_stage, AudioConvEmbedStage):
            inp_mel = mel_features if mel_features is not None else pixel_values
            if inp_mel is None:
                raise ValueError("mel_features required for audio model")
            hidden = self._run_audio_embed(
                inp_mel, g.input_stage, weights["audio_embed"], use_cutile=use_cutile
            )
        elif isinstance(g.input_stage, DiffusionPatchEmbedStage):
            if pixel_values is None:
                raise ValueError("pixel_values (latents) required for diffusion model")
            dw = weights["diffusion_embed"]
            patches = F.conv2d(
                pixel_values.float(),
                dw["patch_w"].float(),
                bias=dw["patch_b"].float(),
                stride=g.input_stage.patch_size,
                padding=0,
            )
            hidden = patches.flatten(2).transpose(1, 2).to(pixel_values.dtype) + dw["pos_embed"]
            ts = (
                timesteps
                if timesteps is not None
                else torch.full(
                    (pixel_values.shape[0],), 500, device=pixel_values.device, dtype=torch.long
                )
            )
            t_emb = self._timestep_embedding(ts, g.input_stage.hidden_size, pixel_values.dtype)
            t_h = self._act(
                self._linear(t_emb, dw["t_fc1_w"], dw["t_fc1_b"], use_cutile), "silu", use_cutile
            )
            t_cond = self._linear(t_h, dw["t_fc2_w"], dw["t_fc2_b"], use_cutile)
        elif isinstance(g.input_stage, MultimodalFusionStage):
            if input_ids is None or pixel_values is None:
                raise ValueError("Both input_ids and pixel_values required for multimodal model")
            vs = g.input_stage.vision_stage
            vis_h = self._run_vision_embed(
                pixel_values, vs, weights["vision_embed"], use_cutile=use_cutile
            )
            for vb, vbw in zip(g.input_stage.vision_blocks, weights["vision_blocks"], strict=True):
                vis_h = self._run_vit_block(vis_h, vb, vbw, use_cutile)
            if vs.use_cls_token and g.input_stage.select_feature == "patch":
                vis_h = vis_h[:, 1:, :]
            proj = weights["projector"]
            vis_proj = self._linear(
                self._act(
                    self._linear(vis_h, proj["fc1_w"], proj["fc1_b"], use_cutile),
                    "gelu",
                    use_cutile,
                ),
                proj["fc2_w"],
                proj["fc2_b"],
                use_cutile,
            )
            txt_h = F.embedding(input_ids, weights["embed_tokens"])
            hidden = torch.cat([vis_proj, txt_h], dim=1)
        else:
            raise TypeError(f"Unsupported input_stage: {type(g.input_stage)}")

        for blk, bw in zip(g.blocks, weights["blocks"], strict=True):
            if isinstance(blk, DenseTransformerBlock):
                if blk.norm_type == "layernorm":
                    h_norm = self._layer_norm(
                        hidden, bw["in_norm_w"], bw.get("in_norm_b"), blk.rms_norm_eps, use_cutile
                    )
                else:
                    h_norm = self._rms_norm(hidden, bw["in_norm_w"], blk.rms_norm_eps, use_cutile)

                hidden = hidden + self._causal_attention(
                    h_norm,
                    bw,
                    blk.num_attention_heads,
                    blk.num_key_value_heads,
                    blk.head_dim,
                    blk.rope_theta,
                    blk.rope_scaling,
                    blk.qkv_bias,
                    blk.sliding_window,
                    use_cutile,
                )

                if blk.norm_type == "layernorm":
                    h_ffn = self._layer_norm(
                        hidden,
                        bw["post_norm_w"],
                        bw.get("post_norm_b"),
                        blk.rms_norm_eps,
                        use_cutile,
                    )
                else:
                    h_ffn = self._rms_norm(hidden, bw["post_norm_w"], blk.rms_norm_eps, use_cutile)

                gate = self._act(
                    self._linear(h_ffn, bw["w_gate"], None, use_cutile), blk.activation, use_cutile
                )
                up = self._linear(h_ffn, bw["w_up"], None, use_cutile)
                hidden = hidden + self._linear(gate * up, bw["w_down"], None, use_cutile)

            elif isinstance(blk, SparseMoEBlock):
                h_norm = self._rms_norm(hidden, bw["in_norm_w"], blk.rms_norm_eps, use_cutile)
                hidden = hidden + self._causal_attention(
                    h_norm,
                    bw,
                    blk.num_attention_heads,
                    blk.num_key_value_heads,
                    blk.head_dim,
                    blk.rope_theta,
                    blk.rope_scaling,
                    blk.qkv_bias,
                    blk.sliding_window,
                    use_cutile,
                )
                h_moe = self._rms_norm(hidden, bw["post_norm_w"], blk.rms_norm_eps, use_cutile)
                bsz_m, seq_m, d_m = h_moe.shape
                h_flat = h_moe.view(-1, d_m)
                router_logits = torch.matmul(h_flat.float(), bw["router"].float())
                topk_vals, topk_idx = torch.topk(router_logits, k=blk.top_k_experts, dim=-1)
                topk_w = torch.softmax(topk_vals, dim=-1).to(h_moe.dtype)

                moe_acc = torch.zeros_like(h_flat, dtype=torch.float32)
                ep_parts = max(1, blk.ep_partitions)
                exps_per_part = max(1, (blk.num_experts + ep_parts - 1) // ep_parts)
                for p_idx in range(ep_parts):
                    e_start = p_idx * exps_per_part
                    e_end = min(blk.num_experts, e_start + exps_per_part)
                    for e_idx in range(e_start, e_end):
                        ew = bw["experts"][e_idx]
                        e_mask = topk_idx == e_idx
                        if not torch.any(e_mask):
                            continue
                        token_w = (topk_w * e_mask.to(topk_w.dtype)).sum(dim=-1, keepdim=True)
                        active_rows = token_w.squeeze(-1) > 0
                        h_sub = h_flat[active_rows]
                        eg = self._act(
                            self._linear(h_sub, ew["w_gate"], None, use_cutile), "silu", use_cutile
                        )
                        eu = self._linear(h_sub, ew["w_up"], None, use_cutile)
                        eo = self._linear(eg * eu, ew["w_down"], None, use_cutile)
                        moe_acc[active_rows] = (
                            moe_acc[active_rows] + (eo * token_w[active_rows]).float()
                        )
                moe_out = moe_acc.to(h_flat.dtype)

                if "shared_expert" in bw:
                    se = bw["shared_expert"]
                    sg = self._act(
                        self._linear(h_flat, se["w_gate"], None, use_cutile), "silu", use_cutile
                    )
                    su = self._linear(h_flat, se["w_up"], None, use_cutile)
                    moe_out = moe_out + self._linear(sg * su, se["w_down"], None, use_cutile)

                hidden = hidden + moe_out.view(bsz_m, seq_m, d_m)

            elif isinstance(blk, ViTEncoderBlock):
                hidden = self._run_vit_block(hidden, blk, bw, use_cutile)
            elif isinstance(blk, DiTBlock):
                if t_cond is None:
                    raise ValueError("DiTBlock requires timestep conditioning")
                hidden = self._run_dit_block(hidden, t_cond, blk, bw, use_cutile)

        if isinstance(g.output_stage, LMHeadStage):
            if g.output_stage.norm_type == "layernorm":
                hidden = self._layer_norm(
                    hidden,
                    weights["norm_w"],
                    weights.get("norm_b"),
                    g.output_stage.rms_norm_eps,
                    use_cutile,
                )
            else:
                hidden = self._rms_norm(
                    hidden, weights["norm_w"], g.output_stage.rms_norm_eps, use_cutile
                )
            return self._linear(hidden, weights["lm_head"], None, use_cutile)

        if isinstance(g.output_stage, VisionProjectionStage):
            hidden = self._layer_norm(
                hidden,
                weights["norm_w"],
                weights.get("norm_b"),
                g.output_stage.layer_norm_eps,
                use_cutile,
            )
            h1 = self._act(
                self._linear(hidden, weights["proj_fc1_w"], weights["proj_fc1_b"], use_cutile),
                "gelu",
                use_cutile,
            )
            return self._linear(h1, weights["proj_fc2_w"], weights["proj_fc2_b"], use_cutile)

        if isinstance(g.output_stage, ClassifierHeadStage):
            hidden = self._layer_norm(
                hidden,
                weights["norm_w"],
                weights.get("norm_b"),
                g.output_stage.layer_norm_eps,
                use_cutile,
            )
            if g.output_stage.pool_type == "cls":
                pooled = hidden[:, 0, :]
            elif g.output_stage.pool_type == "mean":
                pooled = hidden.mean(dim=1)
            else:
                return hidden
            return self._linear(pooled, weights["cls_w"], weights.get("cls_b"), use_cutile)

        if isinstance(g.output_stage, EncoderOutputStage):
            return self._layer_norm(
                hidden,
                weights["norm_w"],
                weights.get("norm_b"),
                g.output_stage.layer_norm_eps,
                use_cutile,
            )

        if isinstance(g.output_stage, DiffusionOutputStage):
            if t_cond is None:
                raise ValueError("DiffusionOutputStage requires timestep conditioning")
            d_out = g.output_stage.hidden_size
            t_act = self._act(t_cond, "silu", use_cutile)
            shift_out, scale_out = (
                self._linear(t_act, weights["adaln_out_w"], weights["adaln_out_b"], use_cutile)
                .unsqueeze(1)
                .chunk(2, dim=-1)
            )
            ln_out = F.layer_norm(hidden.float(), (d_out,), eps=g.output_stage.layer_norm_eps).to(
                hidden.dtype
            )
            h_mod = ln_out * (1.0 + scale_out) + shift_out
            patches_out = self._linear(
                h_mod, weights["proj_out_w"], weights["proj_out_b"], use_cutile
            )
            bsz = patches_out.shape[0]
            p = g.output_stage.patch_size
            grid_h = g.output_stage.image_size // p
            c_out = g.output_stage.out_channels
            return (
                patches_out.view(bsz, grid_h, grid_h, p, p, c_out)
                .permute(0, 5, 1, 3, 2, 4)
                .contiguous()
                .view(bsz, c_out, g.output_stage.image_size, g.output_stage.image_size)
            )

        raise TypeError(f"Unsupported output_stage: {type(g.output_stage)}")

    def forward(
        self,
        weights: dict[str, Any],
        input_ids: torch.Tensor | None = None,
        pixel_values: torch.Tensor | None = None,
        mel_features: torch.Tensor | None = None,
        timesteps: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Execute `UnifiedModelGraph` using CuTile kernels on CUDA (and PyTorch on CPU)."""
        return self._run_forward(
            weights,
            input_ids=input_ids,
            pixel_values=pixel_values,
            mel_features=mel_features,
            timesteps=timesteps,
            use_cutile=True,
        )

    def reference_forward(
        self,
        weights: dict[str, Any],
        input_ids: torch.Tensor | None = None,
        pixel_values: torch.Tensor | None = None,
        mel_features: torch.Tensor | None = None,
        timesteps: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Execute pure-PyTorch reference forward pass for parity verification."""
        return self._run_forward(
            weights,
            input_ids=input_ids,
            pixel_values=pixel_values,
            mel_features=mel_features,
            timesteps=timesteps,
            use_cutile=False,
        )
