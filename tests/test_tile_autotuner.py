"""Unit tests for TileAutotuner."""

from __future__ import annotations

from cutile_forge.forge.tile_autotuner import CANDIDATE_TILE_CONFIGS, TileAutotuner


def test_12_candidate_configs_present() -> None:
    assert len(CANDIDATE_TILE_CONFIGS) == 12


def test_autotuner_returns_valid_config() -> None:
    tuner = TileAutotuner(target="sm_89")
    cfg_decode = tuner.autotune(1, 12288, 4096, quantized=True)
    cfg_prefill = tuner.autotune(256, 12288, 4096, quantized=True)
    assert cfg_decode in CANDIDATE_TILE_CONFIGS
    assert cfg_prefill in CANDIDATE_TILE_CONFIGS
    assert cfg_decode.tm <= 32
    assert cfg_prefill.tm >= 64


def test_autotuner_caching() -> None:
    tuner = TileAutotuner(target="sm_89")
    c1 = tuner.autotune(256, 4096, 4096)
    c2 = tuner.autotune(256, 4096, 4096)
    assert c1 is c2
