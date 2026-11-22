"""Validation and import utilities for external hardware reports (e.g. RTX 4060, multi-GPU)."""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

REQUIRED_MANIFEST_FIELDS: tuple[str, ...] = (
    "run_id",
    "git_commit",
    "hardware",
    "reproduction_status",
)

_PLACEHOLDER_TOKENS: frozenset[str] = frozenset(
    {"", "tbd", "todo", "placeholder", "unknown", "none", "null", "n/a"}
)

_ALLOWED_REPORT_STATUSES: frozenset[str] = frozenset(
    {
        "MEASURED",
        "REPRODUCED",
        "INDEPENDENTLY_REPRODUCED",
        "VERIFIED",
        "END_TO_END_VERIFIED",
        "DEGRADED",
        "PASS",
        "FAIL",
    }
)


def _extract_hardware_info(data: dict[str, Any]) -> dict[str, Any]:
    """Normalize hardware metadata from either top-level `hardware` or `environment`."""
    hw = data.get("hardware")
    if isinstance(hw, dict) and hw:
        return dict(hw)
    if isinstance(hw, str) and hw.strip().lower() not in _PLACEHOLDER_TOKENS:
        return {"gpu_name": hw.strip()}

    env = data.get("environment")
    if isinstance(env, dict):
        gpu_name = str(env.get("gpu_name", "")).strip()
        arch = str(env.get("compute_capability", "")).strip()
        if gpu_name.lower() not in _PLACEHOLDER_TOKENS:
            return {
                "gpu_name": gpu_name,
                "compute_capability": arch,
                "gpu_count": env.get("gpu_count", 1),
                "vram_gb": env.get("vram_gb"),
                "nvidia_driver": env.get("nvidia_driver"),
                "cuda_runtime": env.get("cuda_runtime"),
            }
    return {}


def validate_hardware_report(path: Path) -> dict[str, Any]:
    """Validate an external hardware validation manifest JSON or directory.

    Refuses empty files, invalid JSON, placeholder values, or reports missing
    any required field (`run_id`, `git_commit`, `hardware`, `reproduction_status`).
    """
    target = Path(path)
    if target.is_dir():
        manifest_path = target / "manifest.json"
    else:
        manifest_path = target

    if not manifest_path.exists():
        raise FileNotFoundError(f"Hardware report file not found: {manifest_path}")
    raw_text = manifest_path.read_text(encoding="utf-8").strip()
    if not raw_text:
        raise ValueError(f"Hardware report is empty: {manifest_path}")

    data = json.loads(raw_text)
    if not isinstance(data, dict) or not data:
        raise ValueError(f"Hardware report must be a non-empty JSON object: {manifest_path}")

    run_id = str(data.get("run_id", "")).strip()
    if run_id.lower() in _PLACEHOLDER_TOKENS:
        raise ValueError("Hardware report missing valid non-placeholder 'run_id'")

    git_commit = str(data.get("git_commit", "")).strip()
    if git_commit.lower() in _PLACEHOLDER_TOKENS or len(git_commit) < 7:
        raise ValueError("Hardware report missing valid 'git_commit'")

    hardware = _extract_hardware_info(data)
    if not hardware or str(hardware.get("gpu_name", "")).strip().lower() in _PLACEHOLDER_TOKENS:
        raise ValueError(
            "Hardware report missing non-empty 'hardware' (or 'environment.gpu_name') metadata"
        )

    repro_status = str(data.get("reproduction_status", "")).strip()
    if repro_status not in _ALLOWED_REPORT_STATUSES:
        raise ValueError(
            f"Hardware report 'reproduction_status'={repro_status!r} must be one of "
            f"{sorted(_ALLOWED_REPORT_STATUSES)}"
        )

    records = data.get("records")
    if isinstance(records, list) and len(records) == 0:
        raise ValueError("Hardware report contains an empty 'records' list")

    return {
        "source_path": str(manifest_path),
        "run_id": run_id,
        "git_commit": git_commit,
        "hardware": hardware,
        "reproduction_status": repro_status,
        "manifest": data,
    }


def _update_hardware_matrix(
    matrix_path: Path,
    validated: dict[str, Any],
    imported_manifest: Path,
) -> bool:
    """Update matching unvalidated/blocked row in `docs/HARDWARE_SUPPORT_MATRIX.md`."""
    if not matrix_path.exists():
        return False

    hw = validated["hardware"]
    gpu_name = str(hw.get("gpu_name", "")).lower()
    gpu_count = int(hw.get("gpu_count", 1) or 1)
    status = validated["reproduction_status"]
    lines = matrix_path.read_text(encoding="utf-8").splitlines()
    updated = False
    new_lines: list[str] = []

    for line in lines:
        lower = line.lower()
        match_4060 = "4060" in gpu_name and "4060" in lower and line.startswith("|")
        match_mgpu = gpu_count >= 2 and "physical multi-gpu" in lower and line.startswith("|")
        if match_4060 or match_mgpu:
            cells = [c.strip() for c in line.strip().strip("|").split("|")]
            if len(cells) >= 5:
                cells[3] = f"`{status}`"
                cells[4] = f"`{imported_manifest.as_posix()}`"
                line = "| " + " | ".join(cells) + " |"
                updated = True
        new_lines.append(line)

    if updated:
        matrix_path.write_text("\n".join(new_lines) + "\n", encoding="utf-8")
    return updated


def import_hardware_report(
    report_path: Path,
    dest_dir: Path,
    matrix_path: Path | None = None,
) -> dict[str, Any]:
    """Validate and copy an external hardware report into `dest_dir`."""
    validated = validate_hardware_report(report_path)
    dest = Path(dest_dir)
    dest.mkdir(parents=True, exist_ok=True)

    run_id = validated["run_id"]
    src_manifest = Path(validated["source_path"])
    safe_id = "".join(ch if (ch.isalnum() or ch in ("-", "_")) else "_" for ch in run_id)
    dest_manifest = dest / f"{safe_id}_manifest.json"
    shutil.copyfile(src_manifest, dest_manifest)

    copied_files = [str(dest_manifest)]
    if src_manifest.parent != dest:
        for extra in ("summary.json", "environment.json"):
            src_extra = src_manifest.parent / extra
            if src_extra.exists() and src_extra.stat().st_size > 0:
                dst_extra = dest / f"{safe_id}_{extra}"
                shutil.copyfile(src_extra, dst_extra)
                copied_files.append(str(dst_extra))

    matrix_updated = False
    if matrix_path is not None:
        matrix_updated = _update_hardware_matrix(matrix_path, validated, dest_manifest)

    return {
        "status": "IMPORTED",
        "run_id": run_id,
        "git_commit": validated["git_commit"],
        "hardware": validated["hardware"],
        "reproduction_status": validated["reproduction_status"],
        "imported_manifest": str(dest_manifest),
        "copied_files": copied_files,
        "matrix_updated": matrix_updated,
    }
