"""Thermal, power, and clock logging for sustained runs."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any, Dict, List

from cutile_forge.bench.nvml_logger import NVMLLogger


class ThermalLogger:
    """Records periodic thermal/power/clock + throughput rows to CSV."""

    def __init__(self, output_csv: str | Path, device_index: int = 0) -> None:
        self.output_csv = Path(output_csv)
        self.output_csv.parent.mkdir(parents=True, exist_ok=True)
        self.nvml = NVMLLogger(device_index=device_index)
        self.rows: List[Dict[str, Any]] = []
        self._base_clock: int = 0

    def record(self, elapsed_s: float, tok_s: float) -> Dict[str, Any]:
        sample = self.nvml.sample_once()
        if self._base_clock == 0 and sample.sm_clock_mhz > 0:
            self._base_clock = sample.sm_clock_mhz
        throttle_pct = 0.0
        if self._base_clock > 0 and sample.sm_clock_mhz < self._base_clock:
            throttle_pct = ((self._base_clock - sample.sm_clock_mhz) / float(self._base_clock)) * 100.0

        row: Dict[str, Any] = {
            "timestamp_s": round(elapsed_s, 2),
            "temp_c": round(sample.temp_c, 1),
            "power_w": round(sample.power_w, 2),
            "sm_clock_mhz": sample.sm_clock_mhz,
            "mem_clock_mhz": sample.mem_clock_mhz,
            "gpu_util_pct": sample.gpu_util_pct,
            "vram_used_gb": round(sample.vram_used_gb, 3),
            "tok_s": round(tok_s, 2),
            "throttle_pct": round(throttle_pct, 2),
        }
        self.rows.append(row)
        return row

    def save(self) -> Path:
        if not self.rows:
            raise ValueError("No thermal rows recorded")
        with self.output_csv.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(self.rows[0].keys()))
            writer.writeheader()
            writer.writerows(self.rows)
        return self.output_csv
