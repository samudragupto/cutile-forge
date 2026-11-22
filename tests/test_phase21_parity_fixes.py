"""Phase 21 failing-then-fixed parity repro tests (zero threshold relaxations)."""

from __future__ import annotations

import math

import torch

from cutile_forge.models.model_graph import (
    DiffusionOutputStage,
    DiffusionPatchEmbedStage,
    UnifiedModelExecutor,
    UnifiedModelGraph,
)


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
