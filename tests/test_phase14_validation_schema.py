"""CPU-safe unit tests for Phase 14 versioned validation schema, manifest, and runner."""

from __future__ import annotations

import json
import tarfile
from pathlib import Path

from benchmarks.production.manifest import build_checkpoint_manifest, capture_allowlisted_env
from benchmarks.production.runner import ValidationRun, pack_artifact_bundle, sha256_text
from benchmarks.production.schema import (
    REQUIRED_RUN_FILES,
    SCHEMA_VERSION,
    CheckpointManifest,
    QuantizationConfig,
    RawSample,
    WorkloadConfig,
    compute_summary_from_samples,
    find_secret_leaks,
    redact_secrets,
    validate_artifact_dir,
)


def test_schema_summary_and_artifact_dir_roundtrip(tmp_path: Path) -> None:
    ckpt_dir = tmp_path / "model"
    ckpt_dir.mkdir()
    (ckpt_dir / "config.json").write_text('{"hidden_size": 128}', encoding="utf-8")
    (ckpt_dir / "weights.bin").write_bytes(b"\x01\x02\x03\x04" * 8)

    ckpt_manifest = build_checkpoint_manifest(
        "test-model", ckpt_dir, cache_path=tmp_path / ".sha_cache.json"
    )
    assert ckpt_manifest.total_bytes > 0
    assert "config.json" in ckpt_manifest.sha256_manifest

    run_dir = tmp_path / "run_001"
    run = ValidationRun(
        run_dir=run_dir,
        run_id="unit_test_run",
        checkpoint=ckpt_manifest,
        quant_config=QuantizationConfig(
            quant_format="gptq",
            group_size=128,
            desc_act=True,
            kv_dtype="fp16",
            disk_format="gptq_int4",
            kernel_format="cutile_int4_dequant",
        ),
        workload=WorkloadConfig(
            batch_size=1,
            concurrency=1,
            prompt_len=16,
            output_len=8,
            warmup_iters=1,
            measured_iters=5,
        ),
        prompt_ids_or_checksum=sha256_text("hello world"),
        execution_mode="synthetic-weights",
        reproduction_status="MEASURED",
        notes="Unit test validation artifact.",
        command_override="pytest tests/test_phase14_validation_schema.py",
    )
    vals = [10.0, 20.0, 30.0, 40.0, 50.0]
    for i, v in enumerate(vals):
        run.add_sample("ttft_ms", v, "ms", iteration=i, request_id=f"req-{i}")
    run.log_stdout("run ok")
    out_dir = run.finalize()

    for fname in REQUIRED_RUN_FILES:
        assert (out_dir / fname).exists(), f"Missing {fname}"

    errs = validate_artifact_dir(out_dir)
    assert errs == [], f"Unexpected validation errors: {errs}"

    summary = json.loads((out_dir / "summary.json").read_text(encoding="utf-8"))
    assert abs(summary["ttft_ms"]["median"] - 30.0) < 1e-6
    assert abs(summary["ttft_ms"]["mean"] - 30.0) < 1e-6
    assert abs(summary["ttft_ms"]["min"] - 10.0) < 1e-6
    assert abs(summary["ttft_ms"]["max"] - 50.0) < 1e-6
    assert abs(summary["ttft_ms"]["p90"] - 46.0) < 1e-6

    manifest = json.loads((out_dir / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["schema_version"] == SCHEMA_VERSION

    bundle = pack_artifact_bundle(out_dir, tmp_path / "bundle.tar.gz")
    assert bundle.exists()
    with tarfile.open(bundle, "r:gz") as tf:
        names = tf.getnames()
    assert any(n.endswith("manifest.json") for n in names)


def test_secret_redaction_and_detection(tmp_path: Path, monkeypatch: object) -> None:
    fake_pat = "ghp_" + "A1B2C3D4E5F6G7H8I9J0"
    fake_hf = "hf_" + "z9y8x7w6v5u4t3s2r1q0"
    monkeypatch.setenv("GITHUB_TOKEN", fake_pat)  # type: ignore[attr-defined]
    monkeypatch.setenv("CUTILE_TEST_VAR", "safe_value")  # type: ignore[attr-defined]

    env = capture_allowlisted_env()
    assert "GITHUB_TOKEN" not in env
    assert env["CUTILE_TEST_VAR"] == "safe_value"

    assert "github_pat" in find_secret_leaks(f"token={fake_pat}")
    assert "hf_token" in find_secret_leaks(f"hf={fake_hf}")
    # Path substring `hf_cache` must not trigger a false-positive secret alert
    assert find_secret_leaks("/var/tmp/arrhat_models/hf_cache") == []

    cleaned = redact_secrets(f"Authorization: Bearer abcdef123456 token={fake_pat}")
    assert fake_pat not in cleaned
    assert "abcdef123456" not in cleaned


def test_compute_summary_single_sample_and_missing_checkpoint(tmp_path: Path) -> None:
    missing = build_checkpoint_manifest("missing", tmp_path / "does_not_exist")
    assert missing == CheckpointManifest(
        model_id="missing", local_path=str(tmp_path / "does_not_exist")
    )

    samples = [RawSample(metric="itl_ms", value=5.5, unit="ms", iteration=0, wall_clock_utc="now")]
    res = compute_summary_from_samples(samples)
    assert res["itl_ms"].median == 5.5
    assert res["itl_ms"].std == 0.0


def test_report_generation_and_baseline_regression_gate(tmp_path: Path) -> None:
    import shutil

    from benchmarks.production.compare_with_baseline import compare_with_baseline
    from scripts.build_validation_report import build_validation_reports

    art_root = tmp_path / "artifacts"
    fixture_src = Path("tests/fixtures/validation_sample")
    shutil.copytree(fixture_src, art_root)
    docs_out = tmp_path / "docs"

    summary = build_validation_reports(
        artifacts_root=art_root,
        results_csv_dir=Path("benchmarks/results_5090"),
        docs_dir=docs_out,
        version="3.1.0",
    )
    assert (art_root / "validation_summary.json").exists()
    assert (docs_out / "HARDWARE_SUPPORT_MATRIX.md").exists()
    assert (docs_out / "KERNEL_REGRESSION_REPORT.md").exists()
    assert (docs_out / "QUANTIZATION_ACCURACY_REPORT.md").exists()

    cmp_res = compare_with_baseline(
        summary, Path("benchmarks/baselines/v3.0.0_rtx5090_baseline.json")
    )
    assert cmp_res.passed is True

    # Simulate an unexplained decode throughput drop (> 5%) -> regression gate must fail
    bad_summary = dict(summary)
    bad_summary["headline"] = dict(summary["headline"])
    bad_summary["headline"]["native_single_stream_tok_s"] = 100.0
    cmp_bad = compare_with_baseline(
        bad_summary, Path("benchmarks/baselines/v3.0.0_rtx5090_baseline.json")
    )
    assert cmp_bad.passed is False
    assert any(r["metric"] == "native_single_stream_tok_s" for r in cmp_bad.regressions)
