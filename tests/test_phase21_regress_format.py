"""Unit tests for Phase 21 Step 0 explicit GateResult formatting in regress.py."""

from __future__ import annotations

import json

import torch

from cutile_forge.regress import (
    RegressionReport,
    RegressionResult,
    classify_greedy_mismatch,
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


def test_classify_greedy_mismatch_ulp_categories_and_fail_invariance() -> None:
    vocab = 16
    # At |logit| ~ 16.0 (in [16, 32)), fp16 ULP = 2^-10 * 16 = 0.015625.
    ulp_16 = float(torch.finfo(torch.float16).eps) * 16.0

    # 1. EXACT_LOGIT_TIE: 0.0 margin in ref
    ref_tie = torch.zeros(vocab, dtype=torch.float32)
    ref_tie[4], ref_tie[9] = 16.0, 16.0
    cu_tie = torch.zeros(vocab, dtype=torch.float32)
    cu_tie[4], cu_tie[9] = 16.0 - ulp_16, 16.0
    d_exact = classify_greedy_mismatch(0, 50, cu_tie, ref_tie, dtype=torch.float16)
    assert d_exact.classification == "EXACT_LOGIT_TIE"
    assert d_exact.ref_margin == 0.0
    assert d_exact.ulp_at_logit == ulp_16

    # 2. NUMERIC_NEAR_TIE: margin <= 2 ULPs (1 ULP here)
    ref_near = torch.zeros(vocab, dtype=torch.float32)
    ref_near[1], ref_near[2] = 20.3125, 20.3125 - ulp_16
    cu_near = torch.zeros(vocab, dtype=torch.float32)
    cu_near[1], cu_near[2] = 20.3125 - ulp_16, 20.3125
    d_near = classify_greedy_mismatch(2, 124, cu_near, ref_near, dtype=torch.float16)
    assert d_near.classification == "NUMERIC_NEAR_TIE"
    assert d_near.ref_margin_ulps is not None and d_near.ref_margin_ulps <= 2.0

    # 3. SUBSTANTIVE_DIVERGENCE: margin > 2 ULPs (10 ULPs in both runtimes)
    ref_sub = torch.zeros(vocab, dtype=torch.float32)
    ref_sub[3], ref_sub[7] = 18.0, 18.0 - 10.0 * ulp_16
    cu_sub = torch.zeros(vocab, dtype=torch.float32)
    cu_sub[3], cu_sub[7] = 18.0 - 10.0 * ulp_16, 18.0
    d_sub = classify_greedy_mismatch(3, 12, cu_sub, ref_sub, dtype=torch.float16)
    assert d_sub.classification == "SUBSTANTIVE_DIVERGENCE"

    # 4. UNCLASSIFIED on missing tensors
    d_unc = classify_greedy_mismatch(1, 5, None, ref_sub, dtype=torch.float16)
    assert d_unc.classification == "UNCLASSIFIED"

    # Strict gate invariance: even EXACT_LOGIT_TIE diagnostics never change FAIL to PASS
    gates = [evaluate_gate("greedy_exact_matches", 4.0, "==", 5.0)]
    failing = summarize_failed_gates(gates)
    res = RegressionResult(
        category="Dense decoder",
        model="Mistral-7B",
        checkpoint="ckpt",
        reference_impl="ref",
        layers_evaluated=32,
        status="PASS" if failing is None else "FAIL",
        weight_source="real-checkpoint",
        native_execution=True,
        failing_check=failing,
        gates=gates,
        diagnostics={"greedy_mismatches": [d_exact.to_dict()]},
    )
    assert res.status == "FAIL"
    assert (
        res.to_dict()["diagnostics"]["greedy_mismatches"][0]["classification"] == "EXACT_LOGIT_TIE"
    )
