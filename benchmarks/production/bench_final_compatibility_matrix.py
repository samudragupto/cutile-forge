"""Final Campaign: Architecture x Checkpoint Format x Quantization x Execution Path Matrix Generator."""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

ALLOWED_MATRIX_STATUSES: frozenset[str] = frozenset({"PASS", "FAIL", "UNSUPPORTED", "NOT RUN"})

MATRIX_AXES: dict[str, list[str]] = {
    "architectures": [
        "mistral",
        "llama",
        "llama3",
        "qwen2",
        "mixtral",
        "qwen2_moe",
        "deepseek_moe",
        "vit",
        "clip_vision",
        "siglip_vision",
        "whisper_encoder",
        "llava",
        "dit",
    ],
    "checkpoint_formats": [
        "safetensors",
        "gguf",
        "hf_transformers",
        "diffusers",
        "synthetic_weights",
    ],
    "quantization_types": [
        "fp16",
        "gptq_int4_desc_act",
        "awq_int4",
        "fp8_channel",
        "gguf_q4_0",
        "gguf_q8_0",
        "gguf_q6_k",
        "gguf_q4_k_m",
        "mixed_int4_5_q5_0",
        "kv_fp8_e4m3fn",
    ],
    "execution_paths": [
        "eager",
        "cuda_graph",
        "batched",
        "speculative",
        "hybrid_streamed",
        "http",
    ],
}


def _load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return dict(json.loads(path.read_text(encoding="utf-8")))


def _find_record(records: list[dict[str, Any]], **match: Any) -> dict[str, Any] | None:
    for rec in records:
        if all(rec.get(k) == v for k, v in match.items()):
            return rec
    return None


def _probe_live_speculative(model_dir: Path) -> dict[str, Any]:
    """Run live speculative decoding greedy-parity check on Mistral-7B-GPTQ."""
    import torch

    from cutile_forge.engine import Mistral7B
    from cutile_forge.engine.config import GPUConfig
    from cutile_forge.engine.weight_loader import load_gptq_weights
    from cutile_forge.runtime import NativeModelRuntime

    if not model_dir.exists():
        return {"status": "NOT RUN", "metric": f"model_dir missing: {model_dir}"}

    weights = load_gptq_weights(model_dir, device="cuda", pin_ddr5=False)
    model = Mistral7B(
        weights=weights,
        gpu_config=GPUConfig(vram_gb=32.0, stream_weights=False),
        device="cuda",
    )
    rt = NativeModelRuntime(model, max_seq_len=512, use_cuda_graph=True)
    # Repetitive prompt triggers N-gram prompt lookup drafting + KV truncation rollback
    prompt = [1, 315, 272, 100, 200, 300, 400, 100, 200, 300, 400, 100, 200]
    base_toks, _ = rt.generate(prompt, max_new_tokens=32, sync_interval=16)
    spec_toks, spec_st = rt.generate_speculative(prompt, max_new_tokens=32, gamma=4)
    exact_match = list(base_toks) == list(spec_toks)
    del rt, model, weights
    torch.cuda.empty_cache()
    return {
        "status": "PASS" if exact_match and len(spec_toks) == 32 else "FAIL",
        "exact_greedy_match": exact_match,
        "tokens": len(spec_toks),
        "total_drafted": int(spec_st.get("total_drafted", 0)),
        "accepted_tokens": int(spec_st.get("accepted_tokens", 0)),
        "acceptance_rate": round(float(spec_st.get("acceptance_rate", 0.0)), 4),
        "decode_tok_s": round(float(spec_st.get("decode_tok_s", 0.0)), 2),
    }


def _status_from_bool(passed: bool) -> str:
    return "PASS" if passed else "FAIL"


def _normalize_regress_status(raw: str | None) -> str:
    if raw == "PASS":
        return "PASS"
    if raw in ("SYNTHETIC-ONLY", "NOT_RUN", "NOT RUN"):
        return "NOT RUN"
    if raw in ("UNSUPPORTED",):
        return "UNSUPPORTED"
    return "FAIL"


def build_compatibility_matrix(
    artifacts_root: Path,
    model_dir: Path = Path("/home/arrhat/models/mistral_7b_4bit"),
    run_live_speculative: bool = True,
) -> dict[str, Any]:
    """Assemble and verify the 4-axis compatibility matrix directly from JSON artifacts."""
    reg_v350 = _load_json(artifacts_root / "v3.5.0/phase18_regression/regress_full.json")
    cov_v350 = _load_json(artifacts_root / "v3.5.0/phase18_coverage/compatibility_matrix.json")
    quant_v320 = _load_json(artifacts_root / "v3.2.0/phase15_quant_real/manifest.json")
    serving_v330 = _load_json(artifacts_root / "v3.3.0/phase16_optimized/manifest.json")
    longctx_v310 = _load_json(artifacts_root / "v3.1.0/phase14c_long_context/manifest.json")
    stress_v360 = _load_json(artifacts_root / "v3.6.0/phase20_serving_stress/manifest.json")

    reg_by_model = {r["model"]: r for r in reg_v350.get("results", [])}
    cov_by_fam = {r["family"]: r for r in cov_v350.get("matrix", [])}
    quant_recs = quant_v320.get("records", [])
    serv_recs = serving_v330.get("records", [])
    long_recs = longctx_v310.get("records", [])

    spec_res = (
        _probe_live_speculative(model_dir)
        if run_live_speculative
        else {"status": "NOT RUN", "metric": "live speculative probe skipped"}
    )

    cells: list[dict[str, Any]] = []

    def add_cell(
        arch: str,
        ckpt_fmt: str,
        quant: str,
        exec_path: str,
        status: str,
        evidence: str,
        summary_metric: str,
        justification: str,
    ) -> None:
        if status not in ALLOWED_MATRIX_STATUSES:
            raise ValueError(
                f"Invalid matrix status {status!r}; must be in {ALLOWED_MATRIX_STATUSES}"
            )
        cells.append(
            {
                "architecture": arch,
                "checkpoint_format": ckpt_fmt,
                "quantization_type": quant,
                "execution_path": exec_path,
                "status": status,
                "evidence_artifact": evidence,
                "measured_metric": summary_metric,
                "justification": justification,
            }
        )

    # 1. Mistral GPTQ execution path sweep (core production target across paths)
    r_mistral = reg_by_model.get("Mistral-7B", {})
    m_m = r_mistral.get("metrics", {})
    add_cell(
        "mistral",
        "safetensors",
        "gptq_int4_desc_act",
        "cuda_graph",
        _normalize_regress_status(r_mistral.get("status")),
        "benchmarks/validation_artifacts/v3.5.0/phase18_regression/regress_full.json",
        (
            f"decode={m_m.get('decode_med_tok_s')} tok/s, cos={m_m.get('logit_cos_sim')}, "
            f"ppl={m_m.get('ppl_cutile')} vs {m_m.get('ppl_ref')}, greedy={m_m.get('greedy_exact_matches')}/{m_m.get('greedy_total_prompts')}"
        ),
        "Primary 7B CUDA-Graph decode path; fails strict 5/5 256-tok greedy match due to 1-prompt tail flip.",
    )

    r_batch = _find_record(serv_recs, stage="inproc_c64")
    add_cell(
        "mistral",
        "safetensors",
        "gptq_int4_desc_act",
        "batched",
        "PASS" if r_batch and r_batch.get("agg_tok_s_med", 0) > 0 else "FAIL",
        "benchmarks/validation_artifacts/v3.3.0/phase16_optimized/manifest.json",
        f"c64 agg_tok_s={r_batch.get('agg_tok_s_med') if r_batch else 'N/A'} [{r_batch.get('agg_tok_s_min') if r_batch else ''}, {r_batch.get('agg_tok_s_max') if r_batch else ''}]",
        "Representative continuous-batching throughput & slot-isolated KV cell at C=64.",
    )

    stress_recs = stress_v360.get("records", []) if stress_v360 else serv_recs
    r_soak = _find_record(stress_recs, stage="sustained_soak") or _find_record(
        serv_recs, stage="run0_http_stream_c64"
    )
    http_ok = bool(
        r_soak
        and (
            (r_soak.get("soak_fail", 0) == 0 and r_soak.get("soak_ok", 0) > 0)
            or (r_soak.get("requests_ok", 0) == r_soak.get("requests_sent", -1))
        )
    )
    http_art = (
        "benchmarks/validation_artifacts/v3.6.0/phase20_serving_stress/manifest.json"
        if stress_v360
        else "benchmarks/validation_artifacts/v3.3.0/phase16_optimized/manifest.json"
    )
    add_cell(
        "mistral",
        "safetensors",
        "gptq_int4_desc_act",
        "http",
        _status_from_bool(http_ok),
        http_art,
        (
            f"soak_ok={r_soak.get('soak_ok')}, soak_fail={r_soak.get('soak_fail')}, itl_p50={r_soak.get('itl_p50_ms')}ms"
            if r_soak and "soak_ok" in r_soak
            else f"http_stream_c64={r_soak.get('agg_tok_s_med') if r_soak else 'N/A'} tok/s"
        ),
        "Representative localhost 127.0.0.1 OpenAI HTTP/SSE continuous-batching soak & serving cell.",
    )

    repro_sum = _find_record(serv_recs, stage="reproducibility_summary_3runs") or {}
    c64_nonstream = repro_sum.get("metrics", {}).get("http_nonstream_c64", {})
    c64_pct = float(c64_nonstream.get("vs_inproc_pct_med", 0.0))
    c64_tok = c64_nonstream.get("agg_tok_s_med", "N/A")
    add_cell(
        "mistral",
        "safetensors",
        "gptq_int4_desc_act",
        "http",
        "PASS" if c64_pct >= 85.0 else "FAIL",
        "benchmarks/validation_artifacts/v3.3.0/phase16_optimized/manifest.json",
        f"http_nonstream_c64={c64_tok} tok/s ({c64_pct}% of in-process vs >=85%)",
        "Phase 16 3-run HTTP non-streaming C=64 throughput gate; 84.99% < 85% is FAIL (single run0 85.02% does not override 3-run median).",
    )

    add_cell(
        "mistral",
        "safetensors",
        "gptq_int4_desc_act",
        "speculative",
        str(spec_res["status"]),
        "benchmarks/validation_artifacts/v3.6.0/final_compatibility_matrix/compatibility_matrix.json",
        json.dumps({k: v for k, v in spec_res.items() if k != "status"}),
        "Representative N-gram prompt lookup + NativePagedKVCache.truncate() speculative rollback cell.",
    )

    r_mis_gguf = _find_record(quant_recs, model="Mistral-7B-v0.1", format="GGUF_Q4_K_M")
    add_cell(
        "mistral",
        "gguf",
        "gguf_q4_k_m",
        "eager",
        str(r_mis_gguf.get("status", "NOT RUN")) if r_mis_gguf else "NOT RUN",
        "benchmarks/validation_artifacts/v3.2.0/phase15_quant_real/manifest.json",
        (
            f"ppl={r_mis_gguf.get('ppl_native')} vs {r_mis_gguf.get('ppl_format_ref')} ({r_mis_gguf.get('ppl_vs_format_ref_diff_pct')}%)"
            if r_mis_gguf
            else "N/A"
        ),
        "Cross-format GGUF Q4_K_M 32L real checkpoint parity cell.",
    )

    r_fp8_kv = _find_record(
        long_recs, model="Mistral-7B-v0.1-GPTQ (32L)", context_len=4096, kv_dtype="fp8_e4m3fn"
    )
    add_cell(
        "mistral",
        "safetensors",
        "kv_fp8_e4m3fn",
        "cuda_graph",
        "PASS" if r_fp8_kv and r_fp8_kv.get("fp8_vs_fp16_greedy_match") == "32/32" else "FAIL",
        "benchmarks/validation_artifacts/v3.1.0/phase14c_long_context/manifest.json",
        (
            f"ctx=4096 decode={r_fp8_kv.get('decode_tok_s')} tok/s, greedy={r_fp8_kv.get('fp8_vs_fp16_greedy_match')}, tail_ppl={r_fp8_kv.get('tail_ppl')}"
            if r_fp8_kv
            else "N/A"
        ),
        "Representative FP8 E4M3 KV-cache compression cell at sliding-window boundary (4K).",
    )

    # 2. Qwen2 across quantization formats & execution paths
    r_qwen = reg_by_model.get("Qwen2.5-0.5B", {})
    mq = r_qwen.get("metrics", {})
    add_cell(
        "qwen2",
        "hf_transformers",
        "fp16",
        "eager",
        _normalize_regress_status(r_qwen.get("status")),
        "benchmarks/validation_artifacts/v3.5.0/phase18_regression/regress_full.json",
        f"cos={mq.get('logit_cos_sim')}, ppl={mq.get('ppl_cutile')} vs {mq.get('ppl_ref')}, greedy={mq.get('greedy_exact_matches')}/{mq.get('greedy_total_prompts')}",
        "UnifiedModelExecutor FP16 native eager path; fails strict 5/5 256-tok greedy gate at 4/5.",
    )
    add_cell(
        "qwen2",
        "hf_transformers",
        "fp16",
        "cuda_graph",
        "UNSUPPORTED",
        "docs/OPERATIONAL_LIMITS.md",
        "No persistent KV-cache/CUDA-graph loop in UnifiedModelExecutor",
        "Explicitly marks missing CUDA Graph KV-cache loop for HF UnifiedModelExecutor decoder path.",
    )

    for qfmt_key, qfmt_name, ckpt_fmt in (
        ("GPTQ_Int4", "gptq_int4_desc_act", "hf_transformers"),
        ("AWQ_Int4", "awq_int4", "hf_transformers"),
        ("FP8_channel", "fp8_channel", "hf_transformers"),
        ("GGUF_Q4_0", "gguf_q4_0", "gguf"),
        ("GGUF_Q8_0", "gguf_q8_0", "gguf"),
        ("GGUF_Q6_K", "gguf_q6_k", "gguf"),
        ("GGUF_Q4_K_M", "mixed_int4_5_q5_0", "gguf"),
    ):
        rec = _find_record(quant_recs, model="Qwen2.5-0.5B-Instruct", format=qfmt_key)
        st = str(rec.get("status", "NOT RUN")) if rec else "NOT RUN"
        metric_str = (
            str(rec.get("error", "UNSUPPORTED"))
            if st == "UNSUPPORTED"
            else (
                f"ppl={rec.get('ppl_native')} vs {rec.get('ppl_format_ref')} ({rec.get('ppl_vs_format_ref_diff_pct')}%)"
                if rec
                else "N/A"
            )
        )
        add_cell(
            "qwen2",
            ckpt_fmt,
            qfmt_name,
            "eager",
            st if st in ALLOWED_MATRIX_STATUSES else "FAIL",
            "benchmarks/validation_artifacts/v3.2.0/phase15_quant_real/manifest.json",
            metric_str,
            f"Representative quantization parity cell for {qfmt_key} on Qwen2.5-0.5B-Instruct.",
        )

    # 3. Large Decoder (70B hybrid streamed vs unquantized)
    r_llama70 = reg_by_model.get("Llama-2-70B", {})
    ml = r_llama70.get("metrics", {})
    add_cell(
        "llama",
        "safetensors",
        "gptq_int4_desc_act",
        "hybrid_streamed",
        _normalize_regress_status(r_llama70.get("status")),
        "benchmarks/validation_artifacts/v3.5.0/phase18_regression/regress_full.json",
        f"decode={ml.get('decode_med_tok_s')} tok/s, cos={ml.get('logit_cos_sim')}, ppl={ml.get('ppl_cutile')} vs {ml.get('ppl_ref')}, greedy={ml.get('greedy_exact_matches')}/{ml.get('greedy_total_prompts')}",
        "80L Stage -1 PCIe streaming cell (55 HBM + 25 DDR5); misses strict 5/5 256-tok greedy at 4/5.",
    )
    add_cell(
        "llama",
        "safetensors",
        "fp16",
        "eager",
        "UNSUPPORTED",
        "docs/OPERATIONAL_LIMITS.md",
        "~140 GB FP16 weights exceed 32 GB VRAM + 62.3 GB host RAM",
        "Explicit capacity boundary for 70B unquantized FP16 on a single RTX 5090 host.",
    )

    # 4. Scaled RoPE Llama-3 & MoE architectures
    r_l3 = cov_by_fam.get("llama3", {})
    ml3 = r_l3.get("key_metrics", {})
    add_cell(
        "llama3",
        "hf_transformers",
        "fp16",
        "eager",
        str(r_l3.get("status", "NOT RUN")),
        "benchmarks/validation_artifacts/v3.5.0/phase18_coverage/compatibility_matrix.json",
        f"cos={ml3.get('min_logit_cos_sim')}, max_diff={ml3.get('max_abs_diff')}, greedy={ml3.get('greedy_exact_prompts')}",
        "All-layer scaled Llama-3 RoPE real-checkpoint cell (Llama-3.2-1B-Instruct).",
    )

    r_mix = reg_by_model.get("Mixtral-8x7B", {})
    mmix = r_mix.get("metrics", {})
    add_cell(
        "mixtral",
        "safetensors",
        "gptq_int4_desc_act",
        "cuda_graph",
        _normalize_regress_status(r_mix.get("status")),
        "benchmarks/validation_artifacts/v3.5.0/phase18_regression/regress_full.json",
        f"decode={mmix.get('decode_med_tok_s')} tok/s, cos={mmix.get('logit_cos_sim')}, ppl={mmix.get('ppl_cutile')} vs {mmix.get('ppl_ref')}, greedy={mmix.get('greedy_exact_matches')}/{mmix.get('greedy_total_prompts')}",
        "Sparse MoE 32L GPTQ + native cutile_moe_router_topk; misses strict 256-tok greedy (0/5) due to router boundary ties.",
    )

    for moe_fam in ("qwen2_moe", "deepseek_moe"):
        rm = cov_by_fam.get(moe_fam, {})
        add_cell(
            moe_fam,
            "synthetic_weights",
            "fp16",
            "eager",
            "NOT RUN",
            "benchmarks/validation_artifacts/v3.5.0/phase18_coverage/compatibility_matrix.json",
            f"synthetic_native={rm.get('native_execution')}, forbidden_aten={rm.get('key_metrics', {}).get('forbidden_aten_calls')}",
            "SYNTHETIC-ONLY: no ungated <16 GB FP16 checkpoint fits side-by-side with reference in 32 GB VRAM.",
        )

    # 5. Vision, Audio, Multimodal & Diffusion
    for fam, ckpt_fmt, reg_key, just in (
        (
            "vit",
            "hf_transformers",
            None,
            "12L ViT-B/16 native patch GEMM + erf-GELU real-checkpoint cell.",
        ),
        (
            "clip_vision",
            "hf_transformers",
            None,
            "12L CLIP-ViT-B/32 native pooler + QuickGELU real-checkpoint cell.",
        ),
        (
            "siglip_vision",
            "hf_transformers",
            "SigLIP ViT-B/16",
            "12L SigLIP vision encoder; max_diff=0.125 exceeds strict 0.05 gate.",
        ),
        (
            "whisper_encoder",
            "hf_transformers",
            "Whisper-Tiny",
            "4L Whisper audio encoder via cutile_conv2d; max_diff=0.52344 exceeds strict 0.1 gate.",
        ),
        (
            "llava",
            "hf_transformers",
            "LLaVA-1.5",
            "55L end-to-end CLIP + projector + Llama multimodal native cell.",
        ),
        (
            "dit",
            "diffusers",
            "DiT-XL/2",
            "28L DiT-XL/2-256 diffusion denoiser with d_head=72 FlashAttention; cos=0.997016 misses strict 0.9999 gate.",
        ),
    ):
        rec = reg_by_model.get(reg_key) if reg_key else cov_by_fam.get(fam)
        st = _normalize_regress_status(rec.get("status") if rec else "NOT RUN")
        metrics_dict = (rec.get("metrics") if reg_key else rec.get("key_metrics")) if rec else {}
        add_cell(
            fam,
            ckpt_fmt,
            "fp16",
            "eager",
            st,
            (
                "benchmarks/validation_artifacts/v3.5.0/phase18_regression/regress_full.json"
                if reg_key
                else "benchmarks/validation_artifacts/v3.5.0/phase18_coverage/compatibility_matrix.json"
            ),
            json.dumps(metrics_dict),
            just,
        )

    add_cell(
        "vit",
        "hf_transformers",
        "fp16",
        "speculative",
        "UNSUPPORTED",
        "docs/OPERATIONAL_LIMITS.md",
        "Non-autoregressive bidirectional encoder",
        "Speculative token drafting does not apply to vision encoders.",
    )

    # 6. Hardware-deferred / Hardware-blocked representative cells
    add_cell(
        "mistral",
        "safetensors",
        "gptq_int4_desc_act",
        "hybrid_streamed",
        "NOT RUN",
        "scripts/run_laptop_4060.sh",
        "RTX 4060 (sm_89, 8 GB) deferred by user; UNVALIDATED",
        "Laptop 4060 8 GB validation was deferred; one-command runner & importer prepared.",
    )

    counts = {s: sum(1 for c in cells if c["status"] == s) for s in sorted(ALLOWED_MATRIX_STATUSES)}

    return {
        "schema_version": "1.0.0",
        "release_version": "3.6.0",
        "generated_utc": datetime.now(UTC).isoformat(),
        "axes": MATRIX_AXES,
        "status_counts": counts,
        "total_cells": len(cells),
        "cells": cells,
    }


def render_matrix_markdown(matrix_data: dict[str, Any]) -> str:
    """Render deterministic Markdown directly from the compatibility matrix JSON."""
    counts = matrix_data["status_counts"]
    axes = matrix_data["axes"]
    lines = [
        f"# Final Campaign Compatibility Matrix (`CuTileForge v{matrix_data['release_version']}`)",
        "",
        "> Auto-generated directly from `compatibility_matrix.json` via `benchmarks/production/bench_final_compatibility_matrix.py`. Do not hand-edit.",
        "",
        "## Matrix Definition (`architecture x checkpoint_format x quantization_type x execution_path`)",
        "",
        f"- **Architectures ({len(axes['architectures'])}):** "
        + ", ".join(f"`{a}`" for a in axes["architectures"]),
        f"- **Checkpoint Formats ({len(axes['checkpoint_formats'])}):** "
        + ", ".join(f"`{c}`" for c in axes["checkpoint_formats"]),
        f"- **Quantization Types ({len(axes['quantization_types'])}):** "
        + ", ".join(f"`{q}`" for q in axes["quantization_types"]),
        f"- **Execution Paths ({len(axes['execution_paths'])}):** "
        + ", ".join(f"`{e}`" for e in axes["execution_paths"]),
        "",
        f"**Selected Representative Cells:** `{matrix_data['total_cells']}` | "
        f"`PASS`: **{counts.get('PASS', 0)}** | "
        f"`FAIL`: **{counts.get('FAIL', 0)}** | "
        f"`UNSUPPORTED`: **{counts.get('UNSUPPORTED', 0)}** | "
        f"`NOT RUN`: **{counts.get('NOT RUN', 0)}**",
        "",
        "| Architecture | Checkpoint Format | Quantization Type | Execution Path | Status | Measured Metric / Reason | Evidence Artifact | Justification |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for c in matrix_data["cells"]:
        lines.append(
            "| `{arch}` | `{ckpt}` | `{quant}` | `{path}` | **{status}** | {metric} | `{art}` | {just} |".format(
                arch=c["architecture"],
                ckpt=c["checkpoint_format"],
                quant=c["quantization_type"],
                path=c["execution_path"],
                status=c["status"],
                metric=c["measured_metric"],
                art=c["evidence_artifact"],
                just=c["justification"],
            )
        )
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate Final Campaign Compatibility Matrix")
    parser.add_argument(
        "--artifacts-root",
        type=Path,
        default=Path("benchmarks/validation_artifacts"),
    )
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=Path("/home/arrhat/models/mistral_7b_4bit"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("benchmarks/validation_artifacts/v3.6.0/final_compatibility_matrix"),
    )
    parser.add_argument(
        "--markdown-out",
        type=Path,
        default=Path("docs/FINAL_COMPATIBILITY_MATRIX.md"),
    )
    parser.add_argument(
        "--skip-live-speculative",
        action="store_true",
    )
    args = parser.parse_args()

    data = build_compatibility_matrix(
        artifacts_root=args.artifacts_root,
        model_dir=args.model_dir,
        run_live_speculative=not args.skip_live_speculative,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    json_path = args.output_dir / "compatibility_matrix.json"
    json_path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")

    md = render_matrix_markdown(data)
    md_out_copy = args.output_dir / "compatibility_matrix.md"
    md_out_copy.write_text(md, encoding="utf-8")
    args.markdown_out.write_text(md, encoding="utf-8")
    print(
        f"Wrote {json_path} and {args.markdown_out} ({data['total_cells']} cells: {data['status_counts']})"
    )


if __name__ == "__main__":
    main()
