"""Persistent JSON tuning database and regression detector for CuTileTune."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


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


def _record_key(op: str, arch: str, shape: tuple[int, ...], dtype: str) -> str:
    shape_str = "x".join(str(d) for d in shape)
    return f"{op}:{arch}:{shape_str}:{dtype}"


class TuningDatabase:
    """JSON-backed persistent database for tuned CuTile kernel configurations."""

    def __init__(self, path: Path | str | None = None) -> None:
        self.path = (
            Path(path)
            if path is not None
            else Path.home() / ".cache" / "cutile_forge" / "autotune.json"
        )
        self._records: dict[str, TuningRecord] = {}
        self._load()

    def _load(self) -> None:
        if not self.path.exists():
            return
        try:
            raw: dict[str, Any] = json.loads(self.path.read_text(encoding="utf-8"))
        except Exception:
            return
        for key, item in raw.items():
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
            )

    def _flush(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = {k: asdict(v) for k, v in self._records.items()}
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
        """Persist `record` to memory and disk."""
        key = _record_key(record.op, record.arch, record.shape, record.dtype)
        self._records[key] = record
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
        if self.path.exists():
            self.path.unlink()
