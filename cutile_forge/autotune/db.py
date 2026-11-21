"""Persistent JSON tuning database, shape fingerprinting & regression detector for CuTileTune v2."""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


def shape_fingerprint(op: str, arch: str, shape: tuple[int, ...], dtype: str) -> str:
    """Return a deterministic short fingerprint for a `(op, arch, shape, dtype)` workload."""
    raw = f"{op}:{arch}:{'x'.join(str(d) for d in shape)}:{dtype}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]


@dataclass(frozen=True)
class TuningRecord:
    """Persisted autotuning result for one `(op, arch, shape, dtype)` tuple."""

    op: str
    arch: str
    shape: tuple[int, ...]
    dtype: str
    tile_shape: tuple[int, int, int]
    occupancy: int
    pipeline_stages: int
    latency_ms: float
    verified: bool = True
    fingerprint: str = ""

    def __post_init__(self) -> None:
        if not self.fingerprint:
            object.__setattr__(
                self, "fingerprint", shape_fingerprint(self.op, self.arch, self.shape, self.dtype)
            )


def _record_key(op: str, arch: str, shape: tuple[int, ...], dtype: str) -> str:
    shape_str = "x".join(str(d) for d in shape)
    return f"{op}:{arch}:{shape_str}:{dtype}"


class TuningDatabase:
    """JSON-backed persistent database for tuned CuTile kernel configurations & history."""

    def __init__(self, path: Path | str | None = None) -> None:
        self.path = (
            Path(path)
            if path is not None
            else Path.home() / ".cache" / "cutile_forge" / "autotune.json"
        )
        self._records: dict[str, TuningRecord] = {}
        self.benchmark_history: list[dict[str, Any]] = []
        self._load()

    def _load(self) -> None:
        if not self.path.exists():
            return
        try:
            raw: dict[str, Any] = json.loads(self.path.read_text(encoding="utf-8"))
        except Exception:
            return

        if "records" in raw and isinstance(raw["records"], dict):
            rec_map: dict[str, Any] = raw["records"]
            hist = raw.get("history", [])
            if isinstance(hist, list):
                self.benchmark_history = list(hist)
        else:
            rec_map = raw

        for key, item in rec_map.items():
            self._records[key] = TuningRecord(
                op=str(item["op"]),
                arch=str(item["arch"]),
                shape=tuple(int(x) for x in item["shape"]),
                dtype=str(item["dtype"]),
                tile_shape=(
                    int(item["tile_shape"][0]),
                    int(item["tile_shape"][1]),
                    int(item["tile_shape"][2]),
                ),
                occupancy=int(item["occupancy"]),
                pipeline_stages=int(item["pipeline_stages"]),
                latency_ms=float(item["latency_ms"]),
                verified=bool(item.get("verified", True)),
                fingerprint=str(item.get("fingerprint", "")),
            )

    def _flush(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "records": {k: asdict(v) for k, v in self._records.items()},
            "history": self.benchmark_history,
        }
        self.path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    def lookup(
        self,
        op: str,
        arch: str,
        shape: tuple[int, ...],
        dtype: str = "float16",
    ) -> TuningRecord | None:
        """Retrieve a cached `TuningRecord` if present."""
        return self._records.get(_record_key(op, arch, shape, dtype))

    def save(self, record: TuningRecord) -> None:
        """Persist `record` to memory and disk, appending an entry to `benchmark_history`."""
        key = _record_key(record.op, record.arch, record.shape, record.dtype)
        self._records[key] = record
        self.benchmark_history.append(
            {
                "timestamp": round(time.time(), 3),
                "fingerprint": record.fingerprint,
                "key": key,
                "tile_shape": list(record.tile_shape),
                "latency_ms": record.latency_ms,
                "verified": record.verified,
            }
        )
        self._flush()

    def detect_regression(
        self,
        op: str,
        arch: str,
        shape: tuple[int, ...],
        dtype: str,
        new_ms: float,
        threshold_pct: float = 15.0,
    ) -> tuple[bool, float]:
        """Check whether `new_ms` regresses beyond `threshold_pct` over recorded baseline."""
        rec = self.lookup(op, arch, shape, dtype)
        if rec is None or rec.latency_ms <= 0.0:
            return False, 0.0
        delta_pct = ((new_ms - rec.latency_ms) / rec.latency_ms) * 100.0
        return delta_pct > threshold_pct, round(delta_pct, 2)

    def all_records(self) -> list[TuningRecord]:
        return list(self._records.values())

    def clear(self) -> None:
        self._records.clear()
        self.benchmark_history.clear()
        if self.path.exists():
            self.path.unlink()
