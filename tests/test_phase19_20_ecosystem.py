"""CPU + integration tests for Phase 19-20 hardware import, repro kit, limits, and final matrix."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

from benchmarks.production.bench_final_compatibility_matrix import (
    ALLOWED_MATRIX_STATUSES,
    build_compatibility_matrix,
    render_matrix_markdown,
)
from cutile_forge.cli import main as cli_main
from cutile_forge.hardware_reports import import_hardware_report, validate_hardware_report


def test_validate_and_import_hardware_report_success_and_matrix_update(tmp_path: Path) -> None:
    report_dir = tmp_path / "rtx4060_run"
    report_dir.mkdir(parents=True)
    manifest = {
        "run_id": "laptop4060_test_001",
        "git_commit": "368ceed51550bda38f100abca7286b4ba467a618",
        "environment": {
            "gpu_name": "NVIDIA GeForce RTX 4060 Laptop GPU",
            "compute_capability": "sm_89",
            "gpu_count": 1,
            "vram_gb": 8.0,
        },
        "reproduction_status": "MEASURED",
        "records": [{"stage": "decode_c1", "agg_tok_s_med": 41.2}],
    }
    (report_dir / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    (report_dir / "summary.json").write_text(json.dumps({"ok": True}), encoding="utf-8")

    matrix_md = tmp_path / "HARDWARE_SUPPORT_MATRIX.md"
    matrix_md.write_text(
        "| Target Hardware | Execution Scope | Max Verified Context | Status | Evidence Artifact |\n"
        "|---|---|---:|---|---|\n"
        "| NVIDIA RTX 4060 Laptop (`sm_89`, 8 GB GDDR6) | Deferred | N/A | `UNVALIDATED` | `scripts/run_laptop_4060.sh` |\n",
        encoding="utf-8",
    )

    dest = tmp_path / "imported"
    res = import_hardware_report(report_dir, dest_dir=dest, matrix_path=matrix_md)
    assert res["status"] == "IMPORTED"
    assert res["run_id"] == "laptop4060_test_001"
    assert res["matrix_updated"] is True
    assert Path(res["imported_manifest"]).exists()
    assert (dest / "laptop4060_test_001_summary.json").exists()
    updated_text = matrix_md.read_text(encoding="utf-8")
    assert "`MEASURED`" in updated_text
    assert "UNVALIDATED" not in updated_text


def test_validate_hardware_report_rejects_empty_and_missing_fields(tmp_path: Path) -> None:
    empty_file = tmp_path / "empty.json"
    empty_file.write_text("", encoding="utf-8")
    with pytest.raises(ValueError, match="empty"):
        validate_hardware_report(empty_file)

    bad_placeholder = tmp_path / "placeholder.json"
    bad_placeholder.write_text(
        json.dumps(
            {
                "run_id": "TBD",
                "git_commit": "1234567",
                "hardware": {"gpu_name": "RTX 4060"},
                "reproduction_status": "MEASURED",
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="run_id"):
        validate_hardware_report(bad_placeholder)

    bad_status = tmp_path / "bad_status.json"
    bad_status.write_text(
        json.dumps(
            {
                "run_id": "run_ok",
                "git_commit": "abcdef123",
                "hardware": {"gpu_name": "RTX 4060"},
                "reproduction_status": "PENDING_HARDWARE",
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="reproduction_status"):
        validate_hardware_report(bad_status)


def test_cli_import_hardware_report(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "run_id": "cli_test_4060",
                "git_commit": "368ceed51550bda38f100abca7286b4ba467a618",
                "hardware": {"gpu_name": "NVIDIA GeForce RTX 4060", "gpu_count": 1},
                "reproduction_status": "MEASURED",
                "records": [{"ok": True}],
            }
        ),
        encoding="utf-8",
    )
    dest = tmp_path / "dest"
    monkeypatch.setattr(
        sys,
        "argv",
        ["cutile-forge", "import-hardware-report", str(manifest_path), "--dest", str(dest)],
    )
    cli_main()
    assert (dest / "cli_test_4060_manifest.json").exists()


def test_repro_kit_operational_limits_and_matrix_artifacts() -> None:
    root = Path(__file__).resolve().parents[1]
    ranges_path = root / "benchmarks/repro_kit/expected_ranges.json"
    assert ranges_path.exists()
    ranges = json.loads(ranges_path.read_text(encoding="utf-8"))
    assert ranges["independent_reproduction_status"] == "PENDING_INDEPENDENT_RUN"
    assert (root / "benchmarks/repro_kit/requirements-lock.txt").exists()
    assert (root / "benchmarks/repro_kit/run_repro.sh").exists()
    assert (root / "scripts/run_laptop_4060.sh").exists()
    assert (root / "scripts/run_multi_gpu_validation.sh").exists()
    assert (root / "docs/MULTI_GPU_RUNBOOK.md").exists()
    assert (root / "docs/OPERATIONAL_LIMITS.md").exists()

    data = build_compatibility_matrix(
        artifacts_root=root / "benchmarks/validation_artifacts",
        run_live_speculative=False,
    )
    assert data["total_cells"] >= 20
    for cell in data["cells"]:
        assert cell["status"] in ALLOWED_MATRIX_STATUSES
    md = render_matrix_markdown(data)
    assert "Final Campaign Compatibility Matrix" in md
    assert "| Architecture | Checkpoint Format | Quantization Type | Execution Path |" in md
