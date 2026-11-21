"""Regression gate comparing validation summary against an immutable baseline."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


@dataclass
class ComparisonResult:
    passed: bool
    regressions: list[dict[str, Any]]
    missing_coverage: list[dict[str, Any]]
    checks: list[dict[str, Any]]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def compare_with_baseline(
    summary_data: dict[str, Any],
    baseline_path: Path,
) -> ComparisonResult:
    """Evaluate `validation_summary.json` against immutable baseline & predeclared tolerances.

    Missing external-engine or extra-hardware coverage is logged as `MISSING_COVERAGE`
    rather than failing the release gate. Unexplained regressions beyond tolerance or
    correctness/corruption failures cause `passed=False`.
    """
    baseline = json.loads(baseline_path.read_text(encoding="utf-8"))
    tol = baseline["tolerances"]
    b_metrics = baseline["metrics"]

    regressions: list[dict[str, Any]] = []
    missing_coverage: list[dict[str, Any]] = []
    checks: list[dict[str, Any]] = []

    def _check_higher_better(
        name: str, cur: float | None, base: float, max_drop_pct: float
    ) -> None:
        if cur is None:
            missing_coverage.append({"metric": name, "reason": "not measured in candidate run"})
            return
        pct_change = 100.0 * (cur - base) / max(base, 1e-9)
        ok = pct_change >= -max_drop_pct
        item = {
            "metric": name,
            "baseline": base,
            "current": round(cur, 3),
            "pct_change": round(pct_change, 2),
            "tolerance": f">={-max_drop_pct:.1f}%",
            "status": "PASS" if ok else "REGRESSION",
        }
        checks.append(item)
        if not ok:
            regressions.append(item)

    def _check_lower_better(name: str, cur: float | None, base: float, max_inc_pct: float) -> None:
        if cur is None:
            missing_coverage.append({"metric": name, "reason": "not measured in candidate run"})
            return
        pct_change = 100.0 * (cur - base) / max(base, 1e-9)
        ok = pct_change <= max_inc_pct
        item = {
            "metric": name,
            "baseline": base,
            "current": round(cur, 3),
            "pct_change": round(pct_change, 2),
            "tolerance": f"<={max_inc_pct:.1f}%",
            "status": "PASS" if ok else "REGRESSION",
        }
        checks.append(item)
        if not ok:
            regressions.append(item)

    measured = summary_data.get("headline", {})
    _check_higher_better(
        "native_single_stream_tok_s",
        measured.get("native_single_stream_tok_s"),
        b_metrics["native_single_stream_tok_s"],
        tol["decode_throughput_max_drop_pct"],
    )
    _check_lower_better(
        "native_ttft_p50_ms",
        measured.get("native_ttft_p50_ms"),
        b_metrics["native_ttft_p50_ms"],
        tol["ttft_itl_max_regression_pct"],
    )
    _check_lower_better(
        "native_itl_p50_ms",
        measured.get("native_itl_p50_ms"),
        b_metrics["native_itl_p50_ms"],
        tol["ttft_itl_max_regression_pct"],
    )
    _check_lower_better(
        "native_peak_vram_mb",
        measured.get("native_peak_vram_mb"),
        b_metrics["native_peak_vram_mb"],
        tol["memory_max_growth_pct"],
    )

    # Check correctness / server error gates
    corr_errors = int(summary_data.get("correctness_errors", 0))
    if corr_errors > 0:
        regressions.append(
            {"metric": "correctness_errors", "current": corr_errors, "status": "REGRESSION"}
        )

    err_rate_pct = float(summary_data.get("server_5xx_error_rate_pct", 0.0))
    ok_5xx = err_rate_pct <= float(tol["max_server_5xx_error_rate_pct"])
    checks.append(
        {
            "metric": "server_5xx_error_rate_pct",
            "baseline": 0.0,
            "current": err_rate_pct,
            "tolerance": f"<={tol['max_server_5xx_error_rate_pct']}%",
            "status": "PASS" if ok_5xx else "REGRESSION",
        }
    )
    if not ok_5xx:
        regressions.append(checks[-1])

    # Quantization PPL check (only for formats expected to match reference closely)
    max_ppl_diff = float(summary_data.get("gptq_on_disk_ppl_rel_diff_pct", 0.0))
    ok_ppl = max_ppl_diff <= float(tol["ppl_rel_diff_max_pct"])
    checks.append(
        {
            "metric": "gptq_on_disk_ppl_rel_diff_pct",
            "baseline": 0.0099,
            "current": max_ppl_diff,
            "tolerance": f"<={tol['ppl_rel_diff_max_pct']}%",
            "status": "PASS" if ok_ppl else "REGRESSION",
        }
    )
    if not ok_ppl:
        regressions.append(checks[-1])

    for cov in summary_data.get("missing_coverage", []):
        missing_coverage.append(cov)

    return ComparisonResult(
        passed=len(regressions) == 0,
        regressions=regressions,
        missing_coverage=missing_coverage,
        checks=checks,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare validation summary with immutable baseline"
    )
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument(
        "--baseline",
        type=Path,
        default=Path("benchmarks/baselines/v3.0.0_rtx5090_baseline.json"),
    )
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    sdata = json.loads(args.summary.read_text(encoding="utf-8"))
    res = compare_with_baseline(sdata, args.baseline)
    out_json = json.dumps(res.to_dict(), indent=2)
    if args.output is not None:
        args.output.write_text(out_json, encoding="utf-8")
    print(out_json)
    if not res.passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
