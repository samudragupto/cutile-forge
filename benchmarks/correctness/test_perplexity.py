"""Correctness test: Perplexity of CuTileForge within 5% of PyTorch reference."""

from __future__ import annotations

import math
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F

from cutile_forge.engine.config import ModelConfig
from cutile_forge.engine.model import Mistral7B


def _compute_ppl(model: Mistral7B, token_ids: torch.Tensor, backend: str) -> float:
    with torch.inference_mode():
        logits = model.forward(token_ids[:, :-1], kv_cache=None, pos_offset=0, backend=backend)
        targets = token_ids[:, 1:].to(logits.device)
        loss = F.cross_entropy(logits.view(-1, logits.shape[-1]).float(), targets.view(-1))
        return float(math.exp(min(20.0, float(loss.item()))))


@pytest.mark.gpu
def test_wikitext2_ppl(model_dir: Path) -> None:
    if not torch.cuda.is_available():
        pytest.skip("CUDA GPU required")

    if (model_dir / "config.json").exists():
        from transformers import AutoTokenizer

        model = Mistral7B.from_pretrained(model_dir, device="cuda")
        tok = AutoTokenizer.from_pretrained(str(model_dir))
        sample_text = (
            "Robert Boulter is an English film, television and theatre actor. "
            "He had a guest-starring role on the television series The Bill in 2000. "
            "This was followed by a starring role in the play Herons written by Simon Stephens, "
            "which was performed in 2001 at the Royal Court Theatre."
        )
        ids = tok.encode(sample_text, return_tensors="pt").to("cuda")
    else:
        cfg = ModelConfig(
            vocab_size=512,
            hidden_size=512,
            intermediate_size=1024,
            num_hidden_layers=4,
            num_attention_heads=8,
            num_key_value_heads=2,
            head_dim=64,
            group_size=128,
        )
        model = Mistral7B.from_random(cfg, device="cuda", seed=77)
        ids = torch.randint(1, 512, (1, 96), device="cuda", dtype=torch.long)

    ppl_ct = _compute_ppl(model, ids, backend="cutile")
    ppl_pt = _compute_ppl(model, ids, backend="pytorch")
    rel_diff = abs(ppl_ct - ppl_pt) / max(1e-6, ppl_pt)
    assert rel_diff < 0.05, f"PPL diff {rel_diff:.4f} >= 0.05 (cutile={ppl_ct:.3f}, pt={ppl_pt:.3f})"

