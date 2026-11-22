"""Unit tests for Phase 21 Step 0 explicit GateResult formatting in regress.py."""

from __future__ import annotations

import json

from cutile_forge.regress import (
    RegressionReport,
    RegressionResult,
    evaluate_gate,
    summarize_failed_gates,
)


def test_evaluate_gate_strict_unrounded_boundaries() -> None:
    # Strict threshold boundaries — never round up
    g_pct = evaluate_gate("vs_inproc_pct", 84.99, ">=", 85.0)
    assert g_pct.passed is False
    assert g_pct.raw_value == 84.99
    assert g_pct.comparator == ">="
    assert g_pct.threshold == 85.0

    g_cos = evaluate_gate("logit_cos_sim", 0.99995, ">=", 0.9999)
    assert g_cos.passed is True

    g_lt = evaluate_gate("max_abs_diff", 0.05, "<", 0.05)
    assert g_lt.passed is False

    g_eq = evaluate_gate("greedy_exact_matches", 4.0, "==", 5.0)
    assert g_eq.passed is False


def test_regression_result_json_and_markdown_list_only_failed_gates() -> None:
    raw_cos = 0.9999997615814209  # prints as 1.000000 at 6 decimals but passes >= 0.9999
    gates = [
        evaluate_gate("logit_cos_sim", raw_cos, ">=", 0.9999),
        evaluate_gate("ppl_rel_diff", 0.000038, "<=", 0.01),
        evaluate_gate("greedy_exact_matches", 4.0, "==", 5.0),
    ]
    failing = summarize_failed_gates(gates)
    res = RegressionResult(
        category="Dense decoder",
        model="Mistral-7B",
        checkpoint="/home/arrhat/models/mistral_7b_4bit",
        reference_impl="Mistral7B(backend='pytorch', GPTQ-32L)",
        layers_evaluated=32,
        status="FAIL",
        weight_source="real-checkpoint",
        native_execution=True,
        failing_check=failing,
        gates=gates,
        metrics={
            "logit_cos_sim": round(raw_cos, 6),
            "greedy_exact_matches": 4,
            "greedy_total_prompts": 5,
            "greedy_tok_match_pct": 99.84,
        },
    )
    payload = json.loads(json.dumps(res.to_dict()))
    assert len(payload["gates"]) == 3
    assert payload["gates"][0]["raw_value"] == raw_cos
    assert payload["gates"][0]["comparator"] == ">="
    assert payload["gates"][0]["threshold"] == 0.9999
    assert payload["gates"][0]["passed"] is True
    assert payload["gates"][2]["passed"] is False

    rep = RegressionReport(
        tier="full",
        commit="1b9c776",
        hardware_id="RTX 5090",
        contended=False,
        timestamp_utc="2026-11-22T00:00:00+00:00",
        results=[res],
    )
    md = rep.format_markdown()
    assert "FAIL: greedy_exact_matches=4.0 (== 5.0)" in md
    # Passing gate names/booleans must NOT leak into the FAIL clause
    fail_segment = md.split("FAIL: ")[1]
    assert "logit_cos_sim" not in fail_segment
    assert "ppl_rel_diff" not in fail_segment
    assert "True" not in fail_segment
