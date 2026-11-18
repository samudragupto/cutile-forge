"""Background NVML telemetry poller for GPU temperature, power, clocks, and VRAM."""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from typing import List, Optional


@dataclass
class NVMLSample:
    """Single GPU telemetry sample."""

    timestamp_s: float
    temp_c: float
    power_w: float
    sm_clock_mhz: int
    mem_clock_mhz: int
    gpu_util_pct: int
    vram_used_gb: float
    pcie_tx_mb_s: float
    pcie_rx_mb_s: float


class NVMLLogger:
    """Polls NVML in a daemon thread at `interval_s` (default 0.5 s)."""

    def __init__(self, device_index: int = 0, interval_s: float = 0.5) -> None:
        self.device_index = device_index
        self.interval_s = interval_s
        self.samples: List[NVMLSample] = []
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._start_time = 0.0
        self._nvml = None
        self._handle = None

    def _init_nvml(self) -> bool:
        try:
            import pynvml

            pynvml.nvmlInit()
            self._nvml = pynvml
            self._handle = pynvml.nvmlDeviceGetHandleByIndex(self.device_index)
            return True
        except Exception:
            self._nvml = None
            self._handle = None
            return False

    def sample_once(self) -> NVMLSample:
        """Collect one NVML sample synchronously."""
        now = time.perf_counter() - (self._start_time or time.perf_counter())
        if self._nvml is None or self._handle is None:
            self._init_nvml()

        if self._nvml is not None and self._handle is not None:
            nvml = self._nvml
            h = self._handle
            temp = float(nvml.nvmlDeviceGetTemperature(h, nvml.NVML_TEMPERATURE_GPU))
            power = float(nvml.nvmlDeviceGetPowerUsage(h)) / 1000.0
            sm_clk = int(nvml.nvmlDeviceGetClockInfo(h, nvml.NVML_CLOCK_SM))
            mem_clk = int(nvml.nvmlDeviceGetClockInfo(h, nvml.NVML_CLOCK_MEM))
            util = nvml.nvmlDeviceGetUtilizationRates(h)
            mem_info = nvml.nvmlDeviceGetMemoryInfo(h)
            try:
                tx_kb = nvml.nvmlDeviceGetPcieThroughput(h, nvml.NVML_PCIE_UTIL_TX_BYTES)
                rx_kb = nvml.nvmlDeviceGetPcieThroughput(h, nvml.NVML_PCIE_UTIL_RX_BYTES)
            except Exception:
                tx_kb, rx_kb = 0, 0
            return NVMLSample(
                timestamp_s=now,
                temp_c=temp,
                power_w=power,
                sm_clock_mhz=sm_clk,
                mem_clock_mhz=mem_clk,
                gpu_util_pct=int(util.gpu),
                vram_used_gb=float(mem_info.used) / 1e9,
                pcie_tx_mb_s=float(tx_kb) / 1024.0,
                pcie_rx_mb_s=float(rx_kb) / 1024.0,
            )

        return NVMLSample(
            timestamp_s=now,
            temp_c=0.0,
            power_w=0.0,
            sm_clock_mhz=0,
            mem_clock_mhz=0,
            gpu_util_pct=0,
            vram_used_gb=0.0,
            pcie_tx_mb_s=0.0,
            pcie_rx_mb_s=0.0,
        )

    def _loop(self) -> None:
        while not self._stop.is_set():
            self.samples.append(self.sample_once())
            self._stop.wait(self.interval_s)

    def start(self) -> None:
        self.samples.clear()
        self._start_time = time.perf_counter()
        self._stop.clear()
        self._init_nvml()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self) -> List[NVMLSample]:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None
        if self._nvml is not None:
            try:
                self._nvml.nvmlShutdown()
            except Exception:
                pass
            self._nvml = None
        return list(self.samples)
