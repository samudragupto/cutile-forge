"""Installation, hardware, and CUDA Tile compatibility diagnostics (`cutile-forge doctor`)."""

from __future__ import annotations

import importlib
import importlib.metadata
import os
import platform
import shutil
import sys
from dataclasses import dataclass
from typing import Any, Literal

CheckStatus = Literal["ok", "warn", "fail"]


@dataclass(frozen=True)
class CheckResult:
    """Result of a single environment diagnostic check."""

    name: str
    status: CheckStatus
    detail: str


@dataclass
class DiagnosticReport:
    """Aggregated hardware/software diagnostic report."""

    checks: list[CheckResult]
    ok: bool
    summary: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "summary": self.summary,
            "platform": platform.platform(),
            "checks": [
                {"name": c.name, "status": c.status, "detail": c.detail} for c in self.checks
            ],
        }

    def format_text(self) -> str:
        lines = [f"CuTileForge Diagnostics — {self.summary}"]
        for chk in self.checks:
            tag = chk.status.upper().ljust(4)
            lines.append(f"  [{tag}] {chk.name}: {chk.detail}")
        return "\n".join(lines)


def run_diagnostics(require_gpu: bool = False) -> DiagnosticReport:
    """Inspect Python, PyTorch, CUDA device, `cuda.tile`, pinned memory, and tooling."""
    checks: list[CheckResult] = []

    # 1. Python version
    py_ver = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
    py_ok = sys.version_info >= (3, 11)
    checks.append(
        CheckResult(
            name="python",
            status="ok" if py_ok else "fail",
            detail=f"{py_ver} ({'supported >=3.11' if py_ok else 'requires >=3.11'})",
        )
    )

    # 2. PyTorch
    torch_mod: Any | None = None
    try:
        import torch

        torch_mod = torch
        cuda_ver = torch.version.cuda or "cpu-only"
        checks.append(
            CheckResult(
                name="pytorch",
                status="ok",
                detail=f"torch={torch.__version__}, cuda={cuda_ver}",
            )
        )
    except Exception as exc:
        checks.append(CheckResult(name="pytorch", status="fail", detail=str(exc)))

    # 3. CUDA device capabilities
    cuda_available = bool(torch_mod is not None and torch_mod.cuda.is_available())
    if cuda_available and torch_mod is not None:
        try:
            props = torch_mod.cuda.get_device_properties(0)
            arch = f"sm_{props.major}{props.minor}"
            vram_gb = props.total_memory / (1024.0**3)
            smem_kb = getattr(props, "shared_memory_per_multiprocessor", 0) // 1024
            detail = (
                f"{props.name} ({arch}, {props.multi_processor_count} SMs, "
                f"{vram_gb:.1f} GB VRAM, {smem_kb} KB SMEM/SM)"
            )
            checks.append(CheckResult(name="cuda_device", status="ok", detail=detail))
        except Exception as exc:
            checks.append(
                CheckResult(
                    name="cuda_device",
                    status="fail" if require_gpu else "warn",
                    detail=f"CUDA query failed: {exc}",
                )
            )
    else:
        checks.append(
            CheckResult(
                name="cuda_device",
                status="fail" if require_gpu else "warn",
                detail=f"CUDA not available (CUTILE_ARCH={os.environ.get('CUTILE_ARCH', 'unset')})",
            )
        )

    # 4. cuda.tile availability & required symbols
    try:
        ct_mod = importlib.import_module("cuda.tile")
        try:
            ct_ver = importlib.metadata.version("cuda-tile")
        except importlib.metadata.PackageNotFoundError:
            ct_ver = getattr(ct_mod, "__version__", "installed")
        required_syms = ("kernel", "launch", "mma", "compilation")
        missing = [s for s in required_syms if not hasattr(ct_mod, s)]
        if missing:
            checks.append(
                CheckResult(
                    name="cuda_tile",
                    status="fail" if require_gpu else "warn",
                    detail=f"cuda-tile ({ct_ver}) missing attributes: {', '.join(missing)}",
                )
            )
        else:
            checks.append(
                CheckResult(
                    name="cuda_tile",
                    status="ok",
                    detail=f"cuda-tile={ct_ver} (kernel, launch, mma, compilation ready)",
                )
            )
    except Exception as exc:
        checks.append(
            CheckResult(
                name="cuda_tile",
                status="fail" if require_gpu else "warn",
                detail=f"cuda.tile unavailable: {exc}",
            )
        )

    # 5. Pinned host memory allocation check
    if cuda_available and torch_mod is not None:
        try:
            buf = torch_mod.empty((1024,), dtype=torch_mod.float16, pin_memory=True)
            pinned_ok = bool(buf.is_pinned())
            checks.append(
                CheckResult(
                    name="pinned_memory",
                    status="ok" if pinned_ok else "warn",
                    detail="cudaHostAlloc pinned DDR5 ok" if pinned_ok else "not pinned",
                )
            )
        except Exception as exc:
            checks.append(CheckResult(name="pinned_memory", status="warn", detail=str(exc)))
    else:
        checks.append(
            CheckResult(name="pinned_memory", status="warn", detail="skipped without CUDA")
        )

    # 6. Optional compiler/profiler tools
    found = [tool for tool in ("nvcc", "dot", "ncu", "nsys") if shutil.which(tool) is not None]
    checks.append(
        CheckResult(
            name="optional_tools",
            status="ok" if found else "warn",
            detail=f"found=[{', '.join(found)}]" if found else "none of nvcc/dot/ncu/nsys on PATH",
        )
    )

    ok = all(c.status != "fail" for c in checks)
    n_ok = sum(1 for c in checks if c.status == "ok")
    n_warn = sum(1 for c in checks if c.status == "warn")
    n_fail = sum(1 for c in checks if c.status == "fail")
    summary = f"{'PASS' if ok else 'FAIL'} ({n_ok} ok, {n_warn} warn, {n_fail} fail)"
    return DiagnosticReport(checks=checks, ok=ok, summary=summary)
