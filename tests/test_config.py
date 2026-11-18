"""Unit tests for ModelConfig, GPUConfig, and BenchConfig."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from cutile_forge.engine.config import BenchConfig, GPUConfig, MistralConfig, ModelConfig


def test_model_config_defaults() -> None:
    cfg = MistralConfig()
    assert cfg.hidden_size == 4096
    assert cfg.num_hidden_layers == 32
    assert cfg.gqa_ratio == 4
    assert cfg.qkv_dim == (32 + 16) * 128


def test_model_config_from_json(tmp_path: Path) -> None:
    cfg_file = tmp_path / "config.json"
    cfg_file.write_text(
        json.dumps(
            {
                "vocab_size": 32000,
                "hidden_size": 2048,
                "intermediate_size": 5504,
                "num_hidden_layers": 4,
                "num_attention_heads": 16,
                "num_key_value_heads": 4,
            }
        ),
        encoding="utf-8",
    )
    (tmp_path / "quantize_config.json").write_text(
        json.dumps({"bits": 4, "group_size": 64}), encoding="utf-8"
    )
    cfg = ModelConfig.from_json(cfg_file)
    assert cfg.hidden_size == 2048
    assert cfg.head_dim == 128
    assert cfg.group_size == 64


def test_model_config_validation() -> None:
    with pytest.raises(ValueError):
        ModelConfig(num_attention_heads=32, num_key_value_heads=7)
    with pytest.raises(ValueError):
        ModelConfig(bits=8)


def test_gpu_config_budget_validation() -> None:
    g = GPUConfig()
    assert g.arch == "sm_89"
    with pytest.raises(ValueError):
        GPUConfig(vram_gb=8.0, vram_budget_gb=10.0)


def test_bench_config_validation() -> None:
    b = BenchConfig(warmup=2, iters=10)
    assert b.iters == 10
    with pytest.raises(ValueError):
        BenchConfig(warmup=-1, iters=10)
