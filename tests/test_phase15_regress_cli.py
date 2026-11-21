"""CPU-safe tests for `cutile-forge regress` and `cutile-forge bench release` CLI wiring."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

from cutile_forge.regress import RegressionReport, RegressionResult


def test_regression_report_serialization_and_markdown(tmp_path: Path) -> None:
    rep = RegressionReport(
        tier="full",
        commit="deadbeef",
        hardware_id="RTX 5090 (sm_120)",
        contended=False,
        timestamp_utc="2026-11-22T00:00:00+00:00",
        results=[
            RegressionResult(
                category="Dense decoder",
                model="Mistral-7B",
                checkpoint="/home/arrhat/models/mistral_7b_4bit",
                reference_impl="Mistral7B(backend='pytorch')",
                layers_evaluated=32,
                status="PASS",
                weight_source="real-checkpoint",
                native_execution=True,
                metrics={
                    "logit_cos_sim": 1.0,
                    "ppl_cutile": 6.25,
                    "ppl_ref": 6.25,
                    "ppl_tokens": 512,
                },
            ),
            RegressionResult(
                category="Vision",
                model="SigLIP ViT-B/16",
                checkpoint="google/siglip-base-patch16-224",
                reference_impl="transformers==5.23.0",
                layers_evaluated=12,
                status="NOT NATIVE",
                weight_source="real-checkpoint",
                native_execution=False,
                metrics={"emb_cos_sim": 0.99999},
            ),
        ],
    )
    assert rep.all_non_native_or_pass is True
    md = rep.format_markdown()
    assert "`PASS`" in md
    assert "`NOT NATIVE`" in md
    assert "Mistral-7B" in md
    out_json = tmp_path / "r.json"
    out_json.write_text(json.dumps(rep.to_dict()), encoding="utf-8")
    loaded = json.loads(out_json.read_text(encoding="utf-8"))
    assert len(loaded["results"]) == 2


def test_cli_regress_parser_and_dispatch(tmp_path: Path) -> None:
    from cutile_forge import cli

    fake_rep = RegressionReport(
        tier="fast",
        commit="abc",
        hardware_id="CPU",
        contended=False,
        timestamp_utc="2026-11-22T00:00:00+00:00",
        results=[],
    )
    with (
        patch("cutile_forge.regress.run_regression_suite", return_value=fake_rep) as mock_run,
        patch("sys.argv", ["cutile-forge", "regress", "--tier", "fast", "--output", str(tmp_path)]),
    ):
        try:
            cli.main()
        except SystemExit as exc:
            assert exc.code == 0
        mock_run.assert_called_once()
