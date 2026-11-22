"""Phase 21 failing-then-fixed parity repro tests (zero threshold relaxations)."""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F

from cutile_forge.models.model_graph import (
    AudioConvEmbedStage,
    DiffusionOutputStage,
    DiffusionPatchEmbedStage,
    UnifiedModelExecutor,
    UnifiedModelGraph,
)
from cutile_forge.models.registry import from_hf_config


def test_dit_timestep_embedding_matches_diffusers_downscale() -> None:
    """Repro: DiT-XL/2 uses diffusers downscale_freq_shift=1 (divisor half_dim - 1 = 127)."""
    stage = DiffusionPatchEmbedStage(hidden_size=1152, image_size=32, patch_size=2, in_channels=4)
    graph = UnifiedModelGraph(
        name="DiT-XL-2-256",
        family="dit",
        modality="diffusion",
        input_stage=stage,
        blocks=[],
        output_stage=DiffusionOutputStage(hidden_size=1152),
    )
    ex = UnifiedModelExecutor(graph)

    t = torch.tensor([500], dtype=torch.long)
    cand = ex._timestep_embedding(t, 256, torch.float32)

    # Exact diffusers.models.embeddings.get_timestep_embedding(..., flip_sin_to_cos=True, downscale_freq_shift=1)
    half = 128
    exponent = -math.log(10000.0) * torch.arange(0, half, dtype=torch.float32) / (half - 1.0)
    args = t[:, None].float() * torch.exp(exponent)[None, :]
    ref = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)

    max_diff = float((cand - ref).abs().max().item())
    assert max_diff < 1e-6, f"DiT timestep embedding mismatch: max_diff={max_diff}"


def test_whisper_audio_embed_uses_gelu_erf() -> None:
    """Repro: Whisper encoder uses exact erf-GELU in conv1/conv2 stem, not tanh GELU."""
    graph = from_hf_config(
        {
            "model_type": "whisper",
            "num_mel_bins": 16,
            "max_source_positions": 16,
            "hidden_size": 32,
            "intermediate_size": 64,
            "num_hidden_layers": 1,
            "num_attention_heads": 2,
            "activation_function": "gelu",
        },
        name="whisper_encoder",
    )
    assert isinstance(graph.input_stage, AudioConvEmbedStage)
    ex = UnifiedModelExecutor(graph)

    torch.manual_seed(7)
    mel = torch.randn((1, 16, 16), dtype=torch.float32) * 0.5
    aw = {
        "conv1_w": torch.randn((32, 16, 3), dtype=torch.float32) * 0.2,
        "conv1_b": torch.randn((32,), dtype=torch.float32) * 0.05,
        "conv2_w": torch.randn((32, 32, 3), dtype=torch.float32) * 0.2,
        "conv2_b": torch.randn((32,), dtype=torch.float32) * 0.05,
        "pos_embed": torch.randn((8, 32), dtype=torch.float32) * 0.1,
    }
    cand = ex._run_audio_embed(mel, graph.input_stage, aw, use_cutile=False)

    c1 = F.gelu(
        F.conv1d(mel, aw["conv1_w"], bias=aw["conv1_b"], stride=1, padding=1), approximate="none"
    )
    c2 = F.gelu(
        F.conv1d(c1, aw["conv2_w"], bias=aw["conv2_b"], stride=2, padding=1), approximate="none"
    )
    ref = c2.transpose(1, 2) + aw["pos_embed"].unsqueeze(0)

    max_diff = float((cand - ref).abs().max().item())
    assert max_diff < 1e-5, f"Whisper audio stem GELU mismatch (tanh vs erf): max_diff={max_diff}"
