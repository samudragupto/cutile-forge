"""Correctness tests: Greedy 64-token decode matches PyTorch token-for-token."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from cutile_forge.engine.config import ModelConfig
from cutile_forge.engine.inference import InferenceEngine
from cutile_forge.engine.model import Mistral7B


@pytest.mark.gpu
def test_greedy_64_tokens() -> None:
    if not torch.cuda.is_available():
        pytest.skip("CUDA GPU required")
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
    model = Mistral7B.from_random(cfg, device="cuda", seed=123)
    engine = InferenceEngine(model)
    prompt = torch.tensor([[1, 42, 77, 13, 99, 250]], device="cuda", dtype=torch.long)

    toks_ct = engine.generate_ids(prompt, max_new_tokens=64, temperature=0.0, backend="cutile")
    toks_pt = engine.generate_ids(prompt, max_new_tokens=64, temperature=0.0, backend="pytorch")
    assert len(toks_ct) == 64
    assert toks_ct == toks_pt


@pytest.mark.gpu
def test_greedy_mistral_real_if_available(model_dir: Path) -> None:
    if not torch.cuda.is_available():
        pytest.skip("CUDA GPU required")
    if not (model_dir / "config.json").exists():
        pytest.skip(f"Model not present at {model_dir}")
    model = Mistral7B.from_pretrained(model_dir, device="cuda")
    engine = InferenceEngine(model)
    prompt = torch.tensor([[1, 450, 5565, 302, 4843, 349]], device="cuda", dtype=torch.long)
    toks_ct = engine.generate_ids(prompt, max_new_tokens=16, temperature=0.0, backend="cutile")
    toks_pt = engine.generate_ids(prompt, max_new_tokens=16, temperature=0.0, backend="pytorch")
    assert toks_ct == toks_pt

