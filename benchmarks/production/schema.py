"""Versioned validation schema and statistical summary helpers for CuTileForge."""

from __future__ import annotations

import json
import math
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

SCHEMA_VERSION = "1.0.0"

REQUIRED_RUN_FILES: tuple[str, ...] = (
    "manifest.json",
    "samples.jsonl",
    "summary.json",
    "environment.json",
    "commands.log",
    "stdout.log",
    "stderr.log",
)

ALLOWED_STATUSES: frozenset[str] = frozenset(
    {
        "MEASURED",
        "REPRODUCED",
        "INDEPENDENTLY_REPRODUCED",
        "PENDING_INDEPENDENT_REPRODUCTION",
        "SIMULATED",
        "DEGRADED",
        "UNSUPPORTED",
        "NOT_RUN",
        "PENDING_HARDWARE",
        "VERIFIED",
        "END_TO_END_VERIFIED",
        "LAYER_VERIFIED_ONLY",
    }
)

_SECRET_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("github_pat", re.compile(r"ghp_[A-Za-z0-9]{16,}")),
    ("github_fine_pat", re.compile(r"github_pat_[A-Za-z0-9_]{20,}")),
    ("hf_token", re.compile(r"hf_[A-Za-z0-9]{16,}")),
    ("openai_key", re.compile(r"sk-[A-Za-z0-9\-_]{16,}")),
    ("bearer_header", re.compile(r"Bearer\s+(?!REDACTED)[A-Za-z0-9._\-]{8,}")),
)


def find_secret_leaks(text: str) -> list[str]:
    """Return names of any credential patterns detected in `text`."""
    hits: list[str] = []
    for name, pat in _SECRET_PATTERNS:
        if pat.search(text):
            hits.append(name)
    return hits


def redact_secrets(text: str) -> str:
    """Redact common API/VCS credential tokens before persisting logs or JSON."""
    out = text
    out = re.sub(r"ghp_[A-Za-z0-9]{16,}", "ghp_REDACTED", out)
    out = re.sub(r"github_pat_[A-Za-z0-9_]{20,}", "github_pat_REDACTED", out)
    out = re.sub(r"hf_[A-Za-z0-9]{16,}", "hf_REDACTED", out)
    out = re.sub(r"sk-[A-Za-z0-9\-_]{16,}", "sk-REDACTED", out)
    out = re.sub(r"Bearer\s+[A-Za-z0-9._\-]{8,}", "Bearer REDACTED", out)
    return out


@dataclass
class CheckpointManifest:
    """Exact model checkpoint metadata and cryptographic file manifest."""

    model_id: str
    local_path: str
    revision: str | None = None
    files: list[dict[str, Any]] = field(default_factory=list)
    total_bytes: int = 0
    sha256_manifest: dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> CheckpointManifest:
        return cls(**data)


@dataclass
class QuantizationConfig:
    """Quantization format on disk vs kernel execution format."""

    quant_format: str
    group_size: int | None = None
    desc_act: bool | None = None
    kv_dtype: str = "fp16"
    mixed_precision: dict[str, Any] = field(default_factory=dict)
    disk_format: str | None = None
    kernel_format: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> QuantizationConfig:
        return cls(**data)


@dataclass
class WorkloadConfig:
    """Exact workload, determinism, and synchronization parameters."""

    batch_size: int = 1
    concurrency: int = 1
    prompt_len: int = 0
    output_len: int = 0
    temperature: float = 0.0
    seed: int = 42
    warmup_iters: int = 0
    measured_iters: int = 1
    sync_method: str = "torch.cuda.synchronize"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> WorkloadConfig:
        return cls(**data)


@dataclass
class ContentionSnapshot:
    """Active GPU compute processes observed before/after a validation run."""

    timestamp_utc: str
    contended: bool
    compute_apps: list[dict[str, str]] = field(default_factory=list)
    query_error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ContentionSnapshot:
        return cls(**data)


@dataclass
class EnvironmentInfo:
    """Complete host hardware, CUDA stack, and Python dependency snapshot."""

    hostname: str
    os: str
    kernel: str
    cpu_model: str
    ram_gb: float
    gpu_name: str
    gpu_count: int
    compute_capability: str
    vram_gb: float
    nvidia_driver: str
    cuda_runtime: str
    torch_version: str
    cutile_version: str
    python_version: str
    dependencies: dict[str, str] = field(default_factory=dict)
    nvml_available: bool = False
    gpu_temp_c: float | None = None
    gpu_power_w: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> EnvironmentInfo:
        return cls(**data)


@dataclass
class ValidationManifest:
    """Top-level manifest binding a validation run to code, hardware, data, and status."""

    schema_version: str
    run_id: str
    cutile_forge_version: str
    git_commit: str
    dirty_worktree: bool
    dirty_diff_sha256: str | None
    utc_start: str
    utc_end: str
    environment: EnvironmentInfo
    env_vars: dict[str, str]
    command: str
    cwd: str
    checkpoint: CheckpointManifest
    quant_config: QuantizationConfig
    prompt_ids_or_checksum: str
    workload: WorkloadConfig
    contention_before: ContentionSnapshot
    contention_after: ContentionSnapshot
    execution_mode: str
    reproduction_status: str
    notes: str = ""
    records: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ValidationManifest:
        return cls(
            schema_version=data["schema_version"],
            run_id=data["run_id"],
            cutile_forge_version=data["cutile_forge_version"],
            git_commit=data["git_commit"],
            dirty_worktree=bool(data["dirty_worktree"]),
            dirty_diff_sha256=data.get("dirty_diff_sha256"),
            utc_start=data["utc_start"],
            utc_end=data["utc_end"],
            environment=EnvironmentInfo.from_dict(data["environment"]),
            env_vars=dict(data.get("env_vars", {})),
            command=data["command"],
            cwd=data["cwd"],
            checkpoint=CheckpointManifest.from_dict(data["checkpoint"]),
            quant_config=QuantizationConfig.from_dict(data["quant_config"]),
            prompt_ids_or_checksum=str(data["prompt_ids_or_checksum"]),
            workload=WorkloadConfig.from_dict(data["workload"]),
            contention_before=ContentionSnapshot.from_dict(data["contention_before"]),
            contention_after=ContentionSnapshot.from_dict(data["contention_after"]),
            execution_mode=data["execution_mode"],
            reproduction_status=data["reproduction_status"],
            notes=data.get("notes", ""),
            records=list(data.get("records", [])),
        )


@dataclass
class RawSample:
    """Single timestamped measurement sample saved to `samples.jsonl`."""

    metric: str
    value: float
    unit: str
    iteration: int
    wall_clock_utc: str
    request_id: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> RawSample:
        return cls(
            metric=str(data["metric"]),
            value=float(data["value"]),
            unit=str(data["unit"]),
            iteration=int(data["iteration"]),
            wall_clock_utc=str(data["wall_clock_utc"]),
            request_id=data.get("request_id"),
            metadata=dict(data.get("metadata", {})),
        )


@dataclass
class MetricSummary:
    """Summary statistics strictly derived from raw `RawSample` measurements."""

    metric: str
    unit: str
    count: int
    median: float
    mean: float
    std: float
    min: float
    max: float
    p50: float
    p90: float
    p95: float
    p99: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _percentile(sorted_vals: list[float], q: float) -> float:
    if not sorted_vals:
        return 0.0
    if len(sorted_vals) == 1:
        return sorted_vals[0]
    pos = (len(sorted_vals) - 1) * (q / 100.0)
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return sorted_vals[lo]
    frac = pos - lo
    return sorted_vals[lo] * (1.0 - frac) + sorted_vals[hi] * frac


def compute_summary_from_samples(samples: list[RawSample]) -> dict[str, MetricSummary]:
    """Compute per-metric distribution statistics strictly from `RawSample` records."""
    grouped: dict[str, list[RawSample]] = {}
    for s in samples:
        if not math.isfinite(s.value):
            raise ValueError(f"Non-finite sample value for metric {s.metric!r}: {s.value}")
        grouped.setdefault(s.metric, []).append(s)

    summaries: dict[str, MetricSummary] = {}
    for metric, items in sorted(grouped.items()):
        unit = items[0].unit
        vals = sorted(float(i.value) for i in items)
        n = len(vals)
        mean_v = sum(vals) / n
        var_v = sum((v - mean_v) ** 2 for v in vals) / (n - 1) if n > 1 else 0.0
        summaries[metric] = MetricSummary(
            metric=metric,
            unit=unit,
            count=n,
            median=_percentile(vals, 50.0),
            mean=mean_v,
            std=math.sqrt(var_v),
            min=vals[0],
            max=vals[-1],
            p50=_percentile(vals, 50.0),
            p90=_percentile(vals, 90.0),
            p95=_percentile(vals, 95.0),
            p99=_percentile(vals, 99.0),
        )
    return summaries


def load_samples_jsonl(path: Path) -> list[RawSample]:
    """Load `RawSample` records from a `samples.jsonl` file."""
    samples: list[RawSample] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line_s = line.strip()
        if not line_s:
            continue
        samples.append(RawSample.from_dict(json.loads(line_s)))
    return samples


def validate_manifest_dict(data: dict[str, Any]) -> list[str]:
    """Validate required fields, statuses, and secret hygiene on a manifest dictionary."""
    errors: list[str] = []
    required_top = (
        "schema_version",
        "run_id",
        "cutile_forge_version",
        "git_commit",
        "dirty_worktree",
        "utc_start",
        "utc_end",
        "environment",
        "command",
        "cwd",
        "checkpoint",
        "quant_config",
        "prompt_ids_or_checksum",
        "workload",
        "contention_before",
        "contention_after",
        "execution_mode",
        "reproduction_status",
    )
    for k in required_top:
        if k not in data:
            errors.append(f"Missing required manifest field: {k}")

    if data.get("schema_version") != SCHEMA_VERSION:
        errors.append(
            f"Unsupported schema_version: {data.get('schema_version')!r} != {SCHEMA_VERSION!r}"
        )

    repro = str(data.get("reproduction_status", ""))
    if repro and repro not in ALLOWED_STATUSES:
        errors.append(f"Invalid reproduction_status: {repro!r}")

    serialized = json.dumps(data)
    leaks = find_secret_leaks(serialized)
    if leaks:
        errors.append(f"Secret leak(s) detected in manifest: {leaks}")
    return errors


def validate_artifact_dir(run_dir: Path) -> list[str]:
    """Validate that a run directory has all 7 artifacts, valid manifest, matching summary, and no secrets."""
    errors: list[str] = []
    for fname in REQUIRED_RUN_FILES:
        p = run_dir / fname
        if not p.exists():
            errors.append(f"Missing required artifact file: {p}")

    manifest_path = run_dir / "manifest.json"
    if manifest_path.exists():
        try:
            mdata = json.loads(manifest_path.read_text(encoding="utf-8"))
            errors.extend(validate_manifest_dict(mdata))
        except Exception as exc:
            errors.append(f"Invalid manifest.json: {exc}")

    env_path = run_dir / "environment.json"
    if env_path.exists():
        try:
            edata = json.loads(env_path.read_text(encoding="utf-8"))
            for key in ("hostname", "gpu_name", "torch_version", "python_version"):
                if key not in edata:
                    errors.append(f"environment.json missing {key}")
        except Exception as exc:
            errors.append(f"Invalid environment.json: {exc}")

    samples_path = run_dir / "samples.jsonl"
    summary_path = run_dir / "summary.json"
    if samples_path.exists() and summary_path.exists():
        try:
            samples = load_samples_jsonl(samples_path)
            expected = {k: v.to_dict() for k, v in compute_summary_from_samples(samples).items()}
            recorded = json.loads(summary_path.read_text(encoding="utf-8"))
            for metric, exp in expected.items():
                if metric not in recorded:
                    errors.append(f"summary.json missing metric computed from samples: {metric}")
                    continue
                rec = recorded[metric]
                if int(rec.get("count", -1)) != int(exp["count"]):
                    errors.append(f"summary count mismatch for {metric}")
                for stat in ("median", "mean", "min", "max"):
                    if abs(float(rec.get(stat, 0.0)) - float(exp[stat])) > 1e-6:
                        errors.append(f"summary.{stat} mismatch for {metric}")
        except Exception as exc:
            errors.append(f"Failed validating samples vs summary: {exc}")

    for fname in REQUIRED_RUN_FILES:
        p = run_dir / fname
        if p.exists():
            leaks = find_secret_leaks(p.read_text(encoding="utf-8", errors="replace"))
            if leaks:
                errors.append(f"Secret leak(s) detected in {fname}: {leaks}")

    return errors
