"""Tests for `cutile_forge.autotune` and `cutile_forge.vision`."""

from __future__ import annotations

from pathlib import Path

import torch

from cutile_forge.autotune import CostGuidedAutotuner, TuningDatabase, TuningRecord
from cutile_forge.forge.codegen import compile_tfg
from cutile_forge.forge.fusion_graph import build_mistral_layer_tfg
from cutile_forge.vision import inspect_compilation


def test_tuning_database_persistence_and_regression(tmp_path: Path) -> None:
    db_path = tmp_path / "autotune.json"
    db = TuningDatabase(db_path)
    rec = TuningRecord(
        op="gemm",
        arch="sm_120",
        shape=(256, 4096, 4096),
        dtype="float16",
        tile_shape=(128, 128, 64),
        occupancy=2,
        pipeline_stages=3,
        latency_ms=0.20,
        verified=True,
    )
    db.save(rec)

    reloaded = TuningDatabase(db_path)
    got = reloaded.lookup("gemm", "sm_120", (256, 4096, 4096), "float16")
    assert got == rec

    reg_ok, pct_ok = reloaded.detect_regression(
        "gemm", "sm_120", (256, 4096, 4096), "float16", new_ms=0.21, threshold_pct=15.0
    )
    assert reg_ok is False and pct_ok == 5.0

    reg_bad, pct_bad = reloaded.detect_regression(
        "gemm", "sm_120", (256, 4096, 4096), "float16", new_ms=0.25, threshold_pct=15.0
    )
    assert reg_bad is True and pct_bad == 25.0


def test_cost_guided_autotuner_caches_and_verifies(tmp_path: Path) -> None:
    db = TuningDatabase(tmp_path / "tune.json")
    tuner = CostGuidedAutotuner(target="auto", db=db)

    rec1 = tuner.tune_gemm(
        128,
        256,
        128,
        dtype="float16",
        benchmark_on_gpu=torch.cuda.is_available(),
    )
    assert rec1.verified is True
    assert len(rec1.tile_shape) == 3

    rec2 = tuner.tune_gemm(128, 256, 128, dtype="float16", benchmark_on_gpu=False)
    assert rec2 == rec1


def test_cu_tile_vision_inspect_compilation() -> None:
    orig = build_mistral_layer_tfg(0)
    to_compile = build_mistral_layer_tfg(0)
    compiled = compile_tfg(to_compile, target="sm_120", pipeline_depth=4, run_passes=True)

    summary = inspect_compilation(orig, compiled)
    assert summary.nodes_before == 8
    assert summary.nodes_after == 3
    assert all(d.status == "fused" for d in summary.decisions)
    assert len(summary.kernel_metrics) == 3
    assert all(km["smem_bytes"] > 0 for km in summary.kernel_metrics)

    md = summary.to_markdown()
    assert "CuTileVision Report" in md
    assert "SMEM (B)" in md
    assert summary.to_dict()["nodes_after"] == 3
