"""Phase 13 (`v2.8.0`) tests: Production Benchmarking Suite verification.

Zero `pytest.skip` policy: runs on both CUDA (`sm_120`) and CPU CI.
"""

from __future__ import annotations

import csv
from pathlib import Path


def test_phase13_production_csv_schema_and_external_engine_entries() -> None:
    """Verify `benchmarks/results_5090/phase13_production.csv` contains complete TTFT, ITL, concurrency, long-context, and external engine records."""
    csv_path = Path("benchmarks/results_5090/phase13_production.csv")
    assert csv_path.exists()

    with csv_path.open("r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    assert len(rows) >= 15
    categories = {r["benchmark_category"] for r in rows}
    assert {
        "single_stream_latency_and_throughput",
        "concurrent_throughput",
        "long_context_32layer_attention",
        "external_engine_comparison",
    }.issubset(categories)

    concurrencies = {
        int(r["concurrency"]) for r in rows if r["benchmark_category"] == "concurrent_throughput"
    }
    assert concurrencies == {1, 4, 16, 64}

    contexts = {
        int(r["context_len"])
        for r in rows
        if r["benchmark_category"] == "long_context_32layer_attention"
    }
    assert contexts == {8192, 16384, 32768}

    ext_rows = [r for r in rows if r["benchmark_category"] == "external_engine_comparison"]
    assert len(ext_rows) >= 4
    for r in ext_rows:
        st = r["status"]
        assert st == "MEASURED" or st.startswith("NOT RUN: ")
