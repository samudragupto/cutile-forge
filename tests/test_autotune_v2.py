"""Tests for CuTileTune v2 multi-op autotuning, fingerprinting, and DB seeding."""

from __future__ import annotations

from pathlib import Path

import torch

from cutile_forge.autotune import (
    CostGuidedAutotuner,
    TuningDatabase,
    TuningRecord,
    shape_fingerprint,
)
from cutile_forge.forge.tile_autotuner import TileAutotuner


def test_multi_op_tuning_fingerprints_and_history(tmp_path: Path) -> None:
    db_path = tmp_path / "tune_v2.json"
    db = TuningDatabase(db_path)
    tuner = CostGuidedAutotuner(target="auto", db=db)

    on_gpu = torch.cuda.is_available()
    r_bmm = tuner.tune_bmm(4, 32, 64, 64, benchmark_on_gpu=on_gpu)
    r_red = tuner.tune_reduction(128, 512, op="reduce_max", benchmark_on_gpu=on_gpu)
    r_tr = tuner.tune_transpose(128, 128, benchmark_on_gpu=on_gpu)
    r_act = tuner.tune_activation(16384, op="gelu", benchmark_on_gpu=on_gpu)

    assert all(r.verified for r in (r_bmm, r_red, r_tr, r_act))
    assert r_bmm.fingerprint == shape_fingerprint("bmm", tuner.arch, (4, 32, 64, 64), "float16")
    assert len(db.benchmark_history) == 4

    reloaded = TuningDatabase(db_path)
    assert len(reloaded.all_records()) == 4
    assert len(reloaded.benchmark_history) == 4


def test_tile_autotuner_from_db_preseeds_cache(tmp_path: Path) -> None:
    db = TuningDatabase(tmp_path / "seed.json")
    db.save(
        TuningRecord(
            op="gemm",
            arch="sm_120",
            shape=(64, 512, 512),
            dtype="float16",
            tile_shape=(64, 64, 64),
            occupancy=4,
            pipeline_stages=3,
            latency_ms=0.05,
        )
    )
    seeded = TileAutotuner.from_db(db, target="sm_120")
    picked = seeded.autotune(64, 512, 512, quantized=False)
    assert picked.shape == (64, 64, 64)
