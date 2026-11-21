"""Nsight Compute (ncu) & Nsight Systems (nsys) profiling harness with analytical fallback."""

from __future__ import annotations

import csv
import io
import json
import shutil
import subprocess
from collections.abc import Callable, Iterable
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import torch

from cutile_forge.bench.harness import benchmark

NCU_DEFAULT_METRICS = (
    "sm__throughput.avg.pct_of_peak_sustained_elapsed",
    "dram__throughput.avg.pct_of_peak_sustained_elapsed",
    "l1tex__t_sector_hit_rate.pct",
    "sm__warps_active.avg.pct_of_peak_sustained_active",
)


@dataclass
class KernelProfileResult:
    """Structured kernel hardware profile with analytical HBM/SM metrics fallback."""

    name: str
    p50_ms: float
    achieved_gbps: float
    hbm_util_pct: float
    achieved_tflops: float
    estimated_occupancy_pct: float
    profiler_backend: str = "analytical_cuda_events"
    ncu_metrics: dict[str, float] = field(default_factory=dict)
    diagnostic: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def detect_profilers() -> dict[str, str | None]:
    """Detect installed NVIDIA `ncu` and `nsys` CLI binaries."""
    return {
        "ncu": shutil.which("ncu"),
        "nsys": shutil.which("nsys"),
    }


def _parse_ncu_csv(csv_text: str) -> dict[str, float]:
    metrics: dict[str, float] = {}
    lines = [ln for ln in csv_text.splitlines() if ln.strip() and not ln.startswith("==")]
    if len(lines) < 2:
        return metrics
    reader = csv.DictReader(io.StringIO("\n".join(lines)))
    for row in reader:
        m_name = row.get("Metric Name") or row.get("metric_name")
        m_val = row.get("Metric Value") or row.get("metric_value")
        if not m_name or not m_val:
            continue
        try:
            metrics[m_name] = float(m_val.replace(",", "").strip())
        except ValueError:
            continue
    return metrics


def profile_kernel_ncu(
    script_cmd: list[str],
    kernel_regex: str = ".*",
    timeout_s: int = 60,
    metrics: tuple[str, ...] = NCU_DEFAULT_METRICS,
) -> dict[str, Any]:
    """Run Nsight Compute (`ncu`) on `script_cmd`, gracefully falling back on ERR_NVGPUCTRPERM."""
    ncu_bin = shutil.which("ncu")
    if not ncu_bin:
        return {
            "available": False,
            "fallback": True,
            "diagnostic": "ncu binary not found in PATH",
            "metrics": {},
        }

    cmd = [
        ncu_bin,
        "--csv",
        "--metrics",
        ",".join(metrics),
        "--kernel-name-base",
        "demangled",
        "-k",
        kernel_regex,
        "--",
        *script_cmd,
    ]
    try:
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout_s,
            check=False,
        )
    except (subprocess.TimeoutExpired, OSError) as exc:
        return {
            "available": False,
            "fallback": True,
            "diagnostic": f"ncu invocation error: {exc}",
            "metrics": {},
        }

    combined = f"{proc.stdout}\n{proc.stderr}"
    if "ERR_NVGPUCTRPERM" in combined:
        return {
            "available": False,
            "fallback": True,
            "diagnostic": (
                "ERR_NVGPUCTRPERM: NVIDIA GPU Performance Counters restricted to root/CAP_SYS_ADMIN; "
                "using CUDA-event + analytical HBM/occupancy fallback"
            ),
            "metrics": {},
        }

    parsed = _parse_ncu_csv(proc.stdout)
    if proc.returncode != 0 and not parsed:
        snippet = (proc.stderr or proc.stdout).strip()[:300]
        return {
            "available": False,
            "fallback": True,
            "diagnostic": f"ncu exited {proc.returncode}: {snippet}",
            "metrics": {},
        }

    return {
        "available": True,
        "fallback": False,
        "diagnostic": None,
        "metrics": parsed,
    }


def profile_timeline_nsys(
    script_cmd: list[str],
    out_path: str | Path,
    timeout_s: int = 60,
) -> dict[str, Any]:
    """Capture Nsight Systems (`nsys`) timeline `.nsys-rep` and CLI summary cleanly."""
    nsys_bin = shutil.which("nsys")
    if not nsys_bin:
        return {
            "available": False,
            "out_path": str(out_path),
            "diagnostic": "nsys binary not found in PATH",
            "summary": "",
        }

    out_p = Path(out_path)
    out_p.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        nsys_bin,
        "profile",
        "--stats=true",
        "--force-overwrite=true",
        "-o",
        str(out_p),
        *script_cmd,
    ]
    try:
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout_s,
            check=False,
        )
    except (subprocess.TimeoutExpired, OSError) as exc:
        return {
            "available": False,
            "out_path": str(out_p),
            "diagnostic": f"nsys invocation error: {exc}",
            "summary": "",
        }

    if proc.returncode != 0:
        return {
            "available": False,
            "out_path": str(out_p),
            "diagnostic": (proc.stderr or proc.stdout).strip()[:300],
            "summary": proc.stdout[-1000:],
        }

    return {
        "available": True,
        "out_path": str(out_p),
        "diagnostic": None,
        "summary": proc.stdout[-2000:],
    }


def analytical_profile_op(
    name: str,
    fn: Callable[[], Any],
    *,
    bytes_transferred: float,
    flops: float = 0.0,
    peak_bw_gbps: float = 1792.0,
    estimated_occupancy_pct: float = 50.0,
    warmup: int = 5,
    iters: int = 50,
    ncu_diagnostic: str | None = None,
) -> KernelProfileResult:
    """Profile callable via CUDA events and derive analytical HBM bandwidth / TFLOPS."""
    st = benchmark(fn, warmup=warmup, iters=iters, name=name)
    achieved_gbps = st.bandwidth_gb_s(bytes_transferred)
    hbm_util_pct = (achieved_gbps / peak_bw_gbps) * 100.0 if peak_bw_gbps > 0 else 0.0
    achieved_tflops = st.tflops(flops) if flops > 0.0 else 0.0
    backend = "cuda_events_analytical_fallback" if torch.cuda.is_available() else "cpu_analytical"
    return KernelProfileResult(
        name=name,
        p50_ms=round(st.p50_ms, 5),
        achieved_gbps=round(achieved_gbps, 2),
        hbm_util_pct=round(min(100.0, hbm_util_pct), 2),
        achieved_tflops=round(achieved_tflops, 3),
        estimated_occupancy_pct=round(estimated_occupancy_pct, 1),
        profiler_backend=backend,
        diagnostic=ncu_diagnostic,
    )


def save_profile_json(
    results: Iterable[KernelProfileResult],
    path: str | Path,
    extra_meta: dict[str, Any] | None = None,
) -> Path:
    """Persist structured profiling results and hardware diagnostics to JSON."""
    out_p = Path(path)
    out_p.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        "profilers": detect_profilers(),
        "meta": dict(extra_meta or {}),
        "kernels": [r.to_dict() for r in results],
    }
    out_p.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return out_p
