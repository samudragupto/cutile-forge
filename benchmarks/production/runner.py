"""Validation run harness emitting all 7 reproducible artifacts and compressed bundles."""

from __future__ import annotations

import json
import resource
import sys
import tarfile
import time
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cutile_forge
from benchmarks.production.manifest import (
    capture_allowlisted_env,
    collect_environment_info,
    collect_git_state,
    prompt_checksum,
    query_gpu_contention,
    utc_now_iso,
)
from benchmarks.production.schema import (
    SCHEMA_VERSION,
    CheckpointManifest,
    QuantizationConfig,
    RawSample,
    ValidationManifest,
    WorkloadConfig,
    compute_summary_from_samples,
    redact_secrets,
)


def host_rss_mb() -> float:
    """Return process high-water host RSS in MiB (Linux `ru_maxrss` is KiB)."""
    return float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) / 1024.0


def peak_vram_mb() -> tuple[float, float]:
    """Return `(max_allocated_mb, max_reserved_mb)` on active CUDA device, or `(0.0, 0.0)` on CPU."""
    import torch

    if not torch.cuda.is_available():
        return 0.0, 0.0
    alloc = float(torch.cuda.max_memory_allocated()) / (1024.0 * 1024.0)
    resv = float(torch.cuda.max_memory_reserved()) / (1024.0 * 1024.0)
    return alloc, resv


def sha256_text(text: str) -> str:
    """Convenience SHA-256 digest of prompt/dataset text."""
    return prompt_checksum(text)


@dataclass
class TimedDeviceOp:
    """CUDA event timer with explicit synchronization point documentation."""

    elapsed_ms: float = 0.0
    sync_point: str = "torch.cuda.synchronize() after end_event.record()"

    @classmethod
    @contextmanager
    def measure(cls, sync: bool = True):  # type: ignore[no-untyped-def]
        import torch

        res = cls()
        if not torch.cuda.is_available():
            t0 = time.perf_counter()
            try:
                yield res
            finally:
                res.elapsed_ms = (time.perf_counter() - t0) * 1000.0
                res.sync_point = "wall_clock_cpu"
            return

        if sync:
            torch.cuda.synchronize()
        start_ev = torch.cuda.Event(enable_timing=True)
        end_ev = torch.cuda.Event(enable_timing=True)
        start_ev.record()
        try:
            yield res
        finally:
            end_ev.record()
            if sync:
                torch.cuda.synchronize()
            else:
                end_ev.synchronize()
            res.elapsed_ms = float(start_ev.elapsed_time(end_ev))


@dataclass
class TimedServingRequest:
    """Wall-clock timing breakdown for end-to-end serving requests."""

    request_id: str
    queue_ms: float = 0.0
    tokenize_ms: float = 0.0
    prefill_ms: float = 0.0
    first_token_overhead_ms: float = 0.0
    ttft_ms: float = 0.0
    itl_ms_list: list[float] | None = None
    e2e_ms: float = 0.0
    tpot_ms: float = 0.0
    req_tok_s: float = 0.0
    completed_tokens: int = 0
    error_count: int = 0
    timeout_count: int = 0


class ValidationRun:
    """Managed run directory producing manifest, raw samples, summary, env, and logs."""

    def __init__(
        self,
        run_dir: Path,
        run_id: str,
        checkpoint: CheckpointManifest,
        quant_config: QuantizationConfig,
        workload: WorkloadConfig,
        prompt_ids_or_checksum: str,
        execution_mode: str = "real-checkpoint",
        reproduction_status: str = "MEASURED",
        notes: str = "",
        command_override: str | None = None,
    ) -> None:
        self.run_dir = Path(run_dir)
        self.run_id = run_id
        self.checkpoint = checkpoint
        self.quant_config = quant_config
        self.workload = workload
        self.prompt_ids_or_checksum = prompt_ids_or_checksum
        self.execution_mode = execution_mode
        self.reproduction_status = reproduction_status
        self.notes = notes
        self.command = command_override or " ".join(sys.argv)
        self.samples: list[RawSample] = []
        self.records: list[dict[str, Any]] = []
        self._stdout_lines: list[str] = []
        self._stderr_lines: list[str] = []
        self._commands: list[str] = [self.command]
        self._utc_start = utc_now_iso()
        self._env = collect_environment_info()
        self._contention_before = query_gpu_contention()
        self._git_commit, self._dirty, self._diff_sha = collect_git_state()

    def log_command(self, cmd: str) -> None:
        self._commands.append(redact_secrets(cmd))

    def log_stdout(self, msg: str) -> None:
        clean = redact_secrets(msg)
        self._stdout_lines.append(clean)
        print(clean)

    def log_stderr(self, msg: str) -> None:
        clean = redact_secrets(msg)
        self._stderr_lines.append(clean)
        print(clean, file=sys.stderr)

    def add_sample(
        self,
        metric: str,
        value: float,
        unit: str,
        iteration: int = 0,
        request_id: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        self.samples.append(
            RawSample(
                metric=metric,
                value=float(value),
                unit=unit,
                iteration=iteration,
                wall_clock_utc=utc_now_iso(),
                request_id=request_id,
                metadata=metadata or {},
            )
        )

    def add_record(self, record: dict[str, Any]) -> None:
        self.records.append(record)

    def finalize(self) -> Path:
        self.run_dir.mkdir(parents=True, exist_ok=True)
        utc_end = utc_now_iso()
        contention_after = query_gpu_contention()

        manifest = ValidationManifest(
            schema_version=SCHEMA_VERSION,
            run_id=self.run_id,
            cutile_forge_version=cutile_forge.__version__,
            git_commit=self._git_commit,
            dirty_worktree=self._dirty,
            dirty_diff_sha256=self._diff_sha,
            utc_start=self._utc_start,
            utc_end=utc_end,
            environment=self._env,
            env_vars=capture_allowlisted_env(),
            command=redact_secrets(self.command),
            cwd=str(Path.cwd()),
            checkpoint=self.checkpoint,
            quant_config=self.quant_config,
            prompt_ids_or_checksum=self.prompt_ids_or_checksum,
            workload=self.workload,
            contention_before=self._contention_before,
            contention_after=contention_after,
            execution_mode=self.execution_mode,
            reproduction_status=self.reproduction_status,
            notes=redact_secrets(self.notes),
            records=self.records,
        )

        summary = {k: v.to_dict() for k, v in compute_summary_from_samples(self.samples).items()}

        (self.run_dir / "manifest.json").write_text(
            json.dumps(manifest.to_dict(), indent=2), encoding="utf-8"
        )
        (self.run_dir / "environment.json").write_text(
            json.dumps(self._env.to_dict(), indent=2), encoding="utf-8"
        )
        (self.run_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        with (self.run_dir / "samples.jsonl").open("w", encoding="utf-8") as f:
            for s in self.samples:
                f.write(json.dumps(s.to_dict()) + "\n")

        (self.run_dir / "commands.log").write_text(
            "\n".join(self._commands) + "\n", encoding="utf-8"
        )
        (self.run_dir / "stdout.log").write_text(
            "\n".join(self._stdout_lines) + ("\n" if self._stdout_lines else ""),
            encoding="utf-8",
        )
        (self.run_dir / "stderr.log").write_text(
            "\n".join(self._stderr_lines) + ("\n" if self._stderr_lines else ""),
            encoding="utf-8",
        )
        return self.run_dir


def pack_artifact_bundle(source_dir: Path, tar_gz_path: Path) -> Path:
    """Create a reproducible compressed `.tar.gz` validation bundle."""
    src = Path(source_dir)
    dst = Path(tar_gz_path)
    dst.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(dst, "w:gz") as tar:
        tar.add(src, arcname=src.name)
    return dst
