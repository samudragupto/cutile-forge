"""Benchmarking, profiling, and reporting harness."""

from cutile_forge.bench.cpu_profiler import CPUProfileStats, profile_ddr5_bandwidth
from cutile_forge.bench.harness import (
    RichBenchRecord,
    Stats,
    benchmark,
    benchmark_with_telemetry,
    check_regression_gate,
    write_csv,
)
from cutile_forge.bench.nvml_logger import NVMLLogger, NVMLSample
from cutile_forge.bench.overlap_profiler import OverlapStats, measure_compute_dma_overlap
from cutile_forge.bench.pcie_monitor import (
    PCIeStats,
    estimate_pcie_utilization,
    measure_pcie_h2d_bandwidth,
)
from cutile_forge.bench.profiler import (
    KernelProfileResult,
    analytical_profile_op,
    detect_profilers,
    profile_kernel_ncu,
    profile_timeline_nsys,
    save_profile_json,
)
from cutile_forge.bench.thermal_logger import ThermalLogger

__all__ = [
    "CPUProfileStats",
    "KernelProfileResult",
    "NVMLLogger",
    "NVMLSample",
    "OverlapStats",
    "PCIeStats",
    "RichBenchRecord",
    "Stats",
    "ThermalLogger",
    "analytical_profile_op",
    "benchmark",
    "benchmark_with_telemetry",
    "check_regression_gate",
    "detect_profilers",
    "estimate_pcie_utilization",
    "measure_compute_dma_overlap",
    "measure_pcie_h2d_bandwidth",
    "profile_ddr5_bandwidth",
    "profile_kernel_ncu",
    "profile_timeline_nsys",
    "save_profile_json",
    "write_csv",
]
