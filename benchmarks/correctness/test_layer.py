"""Correctness tests: Single layer and 32-layer Mistral stack CuTile vs PyTorch."""

from __future__ import annotations

import pytest
import torch

from cutile_forge.engine.config import ModelConfig
from cutile_forge.engine.model import Mistral7B, MistralLayer
from cutile_forge.engine.weight_loader import create_random_4bit_weights


@pytest.mark.gpu
def test_single_layer() -> None:
    if not torch.cuda.is_available():
        pytest.skip("CUDA GPU required")
    cfg = ModelConfig(
        vocab_size=512,
        hidden_size=512,
        intermediate_size=1024,
        num_hidden_layers=1,
        num_attention_heads=8,
        num_key_value_heads=2,
        head_dim=64,
        group_size=128,
    )
    w = create_random_4bit_weights(cfg, device="cuda", seed=10)
    model = Mistral7B(w, device="cuda")
    layer: MistralLayer = model.layers[0]
    x = torch.randn((1, 16, 512), device="cuda", dtype=torch.float16) * 0.1

    out_ct = layer.forward_cutile(x, w.layers[0], model.cos, model.sin)
    out_pt = layer.forward_pytorch(x, w.layers[0], model.cos, model.sin)
    err = (out_ct - out_pt).abs().max().item()
    assert err < 1e-2, f"Single layer max_abs_err={err}"


@pytest.mark.gpu
def test_32_layers() -> None:
    if not torch.cuda.is_available():
        pytest.skip("CUDA GPU required")
    cfg = ModelConfig(
        vocab_size=512,
        hidden_size=512,
        intermediate_size=1024,
        num_hidden_layers=32,
        num_attention_heads=8,
        num_key_value_heads=2,
        head_dim=64,
        group_size=128,
    )
    w = create_random_4bit_weights(cfg, device="cuda", seed=11)
    model = Mistral7B(w, device="cuda")
    x = torch.randn((1, 16, 512), device="cuda", dtype=torch.float16) * 0.05

    h_ct = x.clone()
    h_pt = x.clone()
    for idx in range(32):
        h_ct = model.layers[idx].forward_cutile(h_ct, w.layers[idx], model.cos, model.sin)
        h_pt = model.layers[idx].forward_pytorch(h_pt, w.layers[idx], model.cos, model.sin)

    err = (h_ct - h_pt).abs().max().item()
    assert err < 1e-2, f"32-layer max_abs_err={err}"
