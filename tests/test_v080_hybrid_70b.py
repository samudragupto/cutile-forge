"""Unit tests for CuTileForge v0.8.0 (LLAMA3_70B_4BIT preset & HybridLayerBudgeter)."""

from __future__ import annotations

from cutile_forge.engine.config import LLAMA3_70B_4BIT
from cutile_forge.forge.weight_streamer import HybridLayerBudgeter


def test_llama3_70b_preset_config() -> None:
    assert LLAMA3_70B_4BIT.hidden_size == 8192
    assert LLAMA3_70B_4BIT.num_hidden_layers == 80
    assert LLAMA3_70B_4BIT.num_attention_heads == 64
    assert LLAMA3_70B_4BIT.num_key_value_heads == 8
    assert LLAMA3_70B_4BIT.head_dim == 128
    assert LLAMA3_70B_4BIT.intermediate_size == 28672
    assert LLAMA3_70B_4BIT.gqa_ratio == 8


def test_hybrid_layer_budgeter_70b_on_32gb() -> None:
    # ~450 MB per 70B 4-bit layer -> 80 layers ~35.1 GB > 85% of 32 GB
    layer_bytes = 450 * 1024 * 1024
    budgeter = HybridLayerBudgeter(
        total_layers=80,
        layer_bytes=layer_bytes,
        vram_budget_gb=32.0,
        reserve_slots=2,
        overhead_gb=2.0,
    )
    assert 50 <= budgeter.n_resident <= 65
    assert budgeter.n_streamed == 80 - budgeter.n_resident
    assert budgeter.is_resident(0) is True
    assert budgeter.is_resident(budgeter.n_resident - 1) is True
    assert budgeter.is_resident(budgeter.n_resident) is False
    assert budgeter.resident_gb < 28.0


def test_hybrid_layer_budgeter_7b_fits_fully_resident() -> None:
    layer_bytes = 130 * 1024 * 1024  # ~4.16 GB total
    budgeter = HybridLayerBudgeter(total_layers=32, layer_bytes=layer_bytes, vram_budget_gb=32.0)
    assert budgeter.n_resident == 32
    assert budgeter.n_streamed == 0


def test_real_large_checkpoints_config_and_budget_if_present() -> None:
    import os
    from pathlib import Path

    from cutile_forge.engine.config import ModelConfig
    from cutile_forge.engine.weight_loader import _estimate_layer_bytes

    root = Path(os.environ.get("STORAGE_ROOT", "/var/tmp/arrhat_models"))
    mixtral_dir = root / "mixtral_8x7b_4bit"
    if (mixtral_dir / "config.json").exists():
        cfg = ModelConfig.from_json(mixtral_dir / "config.json")
        assert cfg.experts == 8
        assert cfg.num_hidden_layers == 32
        assert cfg.group_size == 128

    llama70_dir = root / "llama_70b_4bit"
    if (llama70_dir / "config.json").exists():
        cfg70 = ModelConfig.from_json(llama70_dir / "config.json")
        assert cfg70.num_hidden_layers == 80
        assert cfg70.hidden_size == 8192
        b = HybridLayerBudgeter(
            total_layers=cfg70.num_hidden_layers,
            layer_bytes=_estimate_layer_bytes(cfg70),
            vram_budget_gb=32.0,
            reserve_slots=2,
            overhead_gb=2.5,
        )
        assert 50 <= b.n_resident <= 60
        assert b.n_streamed == 80 - b.n_resident
