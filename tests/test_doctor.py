"""Tests for `cutile_forge.doctor` diagnostics and CLI entrypoint."""

from __future__ import annotations

import json
import subprocess
import sys

import pytest
import torch

from cutile_forge.doctor import DiagnosticReport, run_diagnostics


def test_run_diagnostics_structure() -> None:
    report = run_diagnostics(require_gpu=False)
    assert isinstance(report, DiagnosticReport)
    names = {c.name for c in report.checks}
    assert {"python", "pytorch", "cuda_device", "cuda_tile", "pinned_memory", "optional_tools"} <= (
        names
    )
    payload = report.to_dict()
    assert payload["ok"] is True
    assert len(payload["checks"]) == len(report.checks)
    assert "CuTileForge Diagnostics" in report.format_text()


def test_require_gpu_fails_when_cuda_absent(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    report = run_diagnostics(require_gpu=True)
    assert report.ok is False
    cuda_chk = next(c for c in report.checks if c.name == "cuda_device")
    assert cuda_chk.status == "fail"


def test_cli_doctor_json() -> None:
    proc = subprocess.run(
        [sys.executable, "-m", "cutile_forge.cli", "doctor", "--json"],
        capture_output=True,
        text=True,
        check=True,
    )
    data = json.loads(proc.stdout)
    assert data["ok"] is True
    assert isinstance(data["checks"], list)
