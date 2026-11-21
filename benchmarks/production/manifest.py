"""Environment, git, contention, and cryptographic checkpoint collectors."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
import platform
import socket
import subprocess
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from benchmarks.production.schema import (
    CheckpointManifest,
    ContentionSnapshot,
    EnvironmentInfo,
    redact_secrets,
)

DEFAULT_SHA256_CACHE = Path("/var/tmp/arrhat_models/.sha256_cache.json")

_ALLOWLISTED_ENV_PREFIXES: tuple[str, ...] = (
    "CUTILE_",
    "CUDA_",
    "TORCH_",
    "PYTORCH_",
)
_ALLOWLISTED_ENV_KEYS: frozenset[str] = frozenset(
    {
        "HF_HOME",
        "HF_HUB_OFFLINE",
        "TRANSFORMERS_OFFLINE",
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
    }
)
_BLOCKED_KEY_FRAGMENTS: tuple[str, ...] = ("TOKEN", "KEY", "SECRET", "PASS", "AUTH", "CRED")


def utc_now_iso() -> str:
    """Return current UTC ISO-8601 timestamp."""
    return datetime.now(UTC).isoformat()


def _run_cmd(cmd: list[str], cwd: Path | None = None) -> tuple[int, str]:
    try:
        proc = subprocess.run(
            cmd,
            cwd=str(cwd) if cwd else None,
            capture_output=True,
            text=True,
            timeout=15,
            check=False,
        )
        return proc.returncode, proc.stdout.strip()
    except Exception as exc:
        return 1, str(exc)


def collect_git_state(repo_root: Path | None = None) -> tuple[str, bool, str | None]:
    """Return `(git_commit, dirty_worktree, dirty_diff_sha256)` for the repo."""
    root = repo_root or Path.cwd()
    rc, commit = _run_cmd(["git", "rev-parse", "HEAD"], cwd=root)
    if rc != 0:
        return "unknown", False, None

    rc_st, status_out = _run_cmd(["git", "status", "--porcelain"], cwd=root)
    dirty = rc_st == 0 and bool(status_out.strip())
    if not dirty:
        return commit, False, None

    _, diff_out = _run_cmd(["git", "diff", "HEAD"], cwd=root)
    payload = (status_out + "\n" + diff_out).encode("utf-8", errors="replace")
    return commit, True, hashlib.sha256(payload).hexdigest()


def capture_allowlisted_env() -> dict[str, str]:
    """Capture only safe CUDA/CuTile/Torch/HF config env vars (never tokens or secrets)."""
    out: dict[str, str] = {}
    for k, v in sorted(os.environ.items()):
        ku = k.upper()
        if any(frag in ku for frag in _BLOCKED_KEY_FRAGMENTS):
            continue
        if k in _ALLOWLISTED_ENV_KEYS or any(ku.startswith(p) for p in _ALLOWLISTED_ENV_PREFIXES):
            out[k] = redact_secrets(v)
    return out


def query_gpu_contention() -> ContentionSnapshot:
    """Record active compute processes via `nvidia-smi`."""
    ts = utc_now_iso()
    try:
        proc = subprocess.run(
            ["nvidia-smi", "--query-compute-apps=pid,used_memory", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
        if proc.returncode != 0:
            return ContentionSnapshot(
                timestamp_utc=ts,
                contended=False,
                compute_apps=[],
                query_error=proc.stderr.strip() or f"exit:{proc.returncode}",
            )
        apps: list[dict[str, str]] = []
        for raw_line in proc.stdout.splitlines():
            line = raw_line.strip()
            if not line:
                continue
            parts = [p.strip() for p in line.split(",")]
            apps.append(
                {
                    "pid": parts[0] if parts else line,
                    "used_memory": parts[1] if len(parts) > 1 else "unknown",
                }
            )
        own_pid = str(os.getpid())
        other_apps = [a for a in apps if a.get("pid") != own_pid]
        return ContentionSnapshot(
            timestamp_utc=ts,
            contended=len(other_apps) > 0,
            compute_apps=apps,
            query_error=None,
        )
    except Exception as exc:
        return ContentionSnapshot(
            timestamp_utc=ts,
            contended=False,
            compute_apps=[],
            query_error=str(exc),
        )


def _cpu_model() -> str:
    cpuinfo = Path("/proc/cpuinfo")
    if cpuinfo.exists():
        for line in cpuinfo.read_text(encoding="utf-8", errors="replace").splitlines():
            if line.lower().startswith("model name"):
                _, val = line.split(":", 1)
                return val.strip()
    return platform.processor() or "unknown"


def _ram_gb() -> float:
    meminfo = Path("/proc/meminfo")
    if meminfo.exists():
        for line in meminfo.read_text(encoding="utf-8", errors="replace").splitlines():
            if line.startswith("MemTotal:"):
                kb = int(line.split()[1])
                return round(kb / (1024.0 * 1024.0), 2)
    return 0.0


def _pkg_version(name: str) -> str:
    try:
        return importlib.metadata.version(name)
    except Exception:
        return "not-installed"


def _query_nvml_telemetry() -> tuple[bool, float | None, float | None]:
    """Query temperature and power via NVML; return None instead of zero if unavailable."""
    try:
        import pynvml  # type: ignore[import-untyped]

        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        temp_raw = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
        power_mw = pynvml.nvmlDeviceGetPowerUsage(handle)
        pynvml.nvmlShutdown()
        temp_c = float(temp_raw) if temp_raw and float(temp_raw) > 0 else None
        power_w = round(float(power_mw) / 1000.0, 2) if power_mw and float(power_mw) > 0 else None
        return True, temp_c, power_w
    except Exception:
        return False, None, None


def collect_environment_info() -> EnvironmentInfo:
    """Collect full system, GPU, driver, and package metadata cleanly."""
    import torch

    gpu_count = int(torch.cuda.device_count()) if torch.cuda.is_available() else 0
    if gpu_count > 0:
        props = torch.cuda.get_device_properties(0)
        gpu_name = str(props.name)
        cc = f"sm_{props.major}{props.minor}"
        vram_gb = round(float(props.total_memory) / (1024.0**3), 2)
    else:
        gpu_name = "cpu-only"
        cc = "none"
        vram_gb = 0.0

    rc_drv, drv = _run_cmd(["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"])
    driver_ver = drv.splitlines()[0].strip() if rc_drv == 0 and drv else "unavailable"

    cutile_ver = _pkg_version("cuda-tile")
    if cutile_ver == "not-installed":
        try:
            import cuda.tile as ct  # type: ignore[import-not-found]

            cutile_ver = str(getattr(ct, "__version__", "unknown"))
        except Exception:
            cutile_ver = "unavailable"

    nvml_ok, temp_c, power_w = _query_nvml_telemetry()

    deps = {
        pkg: _pkg_version(pkg)
        for pkg in (
            "numpy",
            "transformers",
            "tokenizers",
            "safetensors",
            "huggingface-hub",
            "gguf",
            "fastapi",
            "uvicorn",
            "httpx",
            "nvidia-ml-py",
        )
    }

    return EnvironmentInfo(
        hostname=socket.gethostname(),
        os=f"{platform.system()} {platform.release()}",
        kernel=platform.version(),
        cpu_model=_cpu_model(),
        ram_gb=_ram_gb(),
        gpu_name=gpu_name,
        gpu_count=gpu_count,
        compute_capability=cc,
        vram_gb=vram_gb,
        nvidia_driver=driver_ver,
        cuda_runtime=str(torch.version.cuda or "none"),
        torch_version=str(torch.__version__),
        cutile_version=cutile_ver,
        python_version=platform.python_version(),
        dependencies=deps,
        nvml_available=nvml_ok,
        gpu_temp_c=temp_c,
        gpu_power_w=power_w,
    )


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            chunk = f.read(4 * 1024 * 1024)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def build_checkpoint_manifest(
    model_id: str,
    local_path: Path | str,
    revision: str | None = None,
    cache_path: Path | None = DEFAULT_SHA256_CACHE,
) -> CheckpointManifest:
    """Inventory files, byte sizes, and exact SHA-256 checksums for a local checkpoint."""
    p = Path(local_path)
    if not p.exists():
        return CheckpointManifest(
            model_id=model_id,
            local_path=str(p),
            revision=revision,
            files=[],
            total_bytes=0,
            sha256_manifest={},
        )

    cache: dict[str, dict[str, Any]] = {}
    if cache_path is not None and cache_path.exists():
        try:
            cache = json.loads(cache_path.read_text(encoding="utf-8"))
        except Exception:
            cache = {}

    file_records: list[dict[str, Any]] = []
    sha_map: dict[str, str] = {}
    total_bytes = 0
    cache_dirty = False

    candidates = [p] if p.is_file() else sorted(f for f in p.rglob("*") if f.is_file())
    for fp in candidates:
        if "/.cache/" in str(fp) or "__pycache__" in str(fp):
            continue
        rel = fp.name if p.is_file() else str(fp.relative_to(p))
        st = fp.stat()
        size = int(st.st_size)
        mtime_ns = int(st.st_mtime_ns)
        abs_key = str(fp.resolve())

        cached = cache.get(abs_key)
        if cached and cached.get("size") == size and cached.get("mtime_ns") == mtime_ns:
            digest = str(cached["sha256"])
        else:
            digest = _sha256_file(fp)
            cache[abs_key] = {"size": size, "mtime_ns": mtime_ns, "sha256": digest}
            cache_dirty = True

        total_bytes += size
        sha_map[rel] = digest
        file_records.append({"path": rel, "bytes": size, "sha256": digest})

    if cache_dirty and cache_path is not None:
        try:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            cache_path.write_text(json.dumps(cache, indent=2), encoding="utf-8")
        except Exception:
            pass

    return CheckpointManifest(
        model_id=model_id,
        local_path=str(p),
        revision=revision,
        files=file_records,
        total_bytes=total_bytes,
        sha256_manifest=sha_map,
    )


def prompt_checksum(prompt_ids: list[int] | str) -> str:
    """Compute stable SHA-256 identifier for prompt token IDs or dataset slice."""
    if isinstance(prompt_ids, str):
        payload = prompt_ids.encode("utf-8")
    else:
        payload = json.dumps(prompt_ids, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()
