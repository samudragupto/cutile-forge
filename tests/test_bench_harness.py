"""Unit tests for the benchmark harness."""

from __future__ import annotations

from pathlib import Path

from cutile_forge.bench.harness import benchmark, write_csv


def test_benchmark_stats_cpu() -> None:
    acc = [0]

    def work() -> None:
        acc[0] += sum(range(200))

    st = benchmark(work, warmup=2, iters=20, name="cpu_test", use_cuda_events=False)
    assert st.iters == 20
    assert st.min_ms <= st.p50_ms <= st.p95_ms <= st.p99_ms <= st.max_ms
    assert st.mean_ms > 0.0


def test_stats_tflops_and_bandwidth() -> None:
    st = benchmark(lambda: sum(range(50)), warmup=1, iters=5, use_cuda_events=False)
    assert st.tflops(1e9) > 0.0
    assert st.bandwidth_gb_s(1e6) > 0.0


def test_write_csv(tmp_path: Path) -> None:
    out = write_csv(tmp_path / "sub" / "res.csv", [{"a": 1, "b": 2.5}, {"a": 2, "b": 3.5}])
    assert out.exists()
    text = out.read_text(encoding="utf-8")
    assert "a,b" in text
    assert "1,2.5" in text
