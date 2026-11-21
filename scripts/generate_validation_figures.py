"""Generate 300-DPI Phase 14 validation figures strictly from raw JSON/JSONL/CSV artifacts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update(
    {
        "font.size": 10,
        "axes.titlesize": 11,
        "axes.labelsize": 10,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 8.5,
        "figure.dpi": 300,
    }
)

COLORS = ["#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00", "#56B4E9"]


def _load_json(path: Path) -> dict[str, Any]:
    return dict(json.loads(path.read_text(encoding="utf-8"))) if path.exists() else {}


def _load_samples(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(ln) for ln in path.read_text(encoding="utf-8").splitlines() if ln.strip()]


def generate_validation_figures(artifacts_root: Path, output_dir: Path) -> list[Path]:
    """Generate all Phase 14 300-DPI validation figures from raw manifests & JSONL samples."""
    output_dir.mkdir(parents=True, exist_ok=True)
    saved: list[Path] = []

    m_14b = _load_json(artifacts_root / "phase14b_external_engines" / "manifest.json")
    m_14c = _load_json(artifacts_root / "phase14c_long_context" / "manifest.json")
    m_14c_syn = _load_json(artifacts_root / "phase14c_synthetic_attention" / "manifest.json")
    m_14e = _load_json(artifacts_root / "phase14e_quantization" / "manifest.json")
    m_14f = _load_json(artifacts_root / "phase14f_serving_stress" / "manifest.json")
    samples_14f = _load_samples(artifacts_root / "phase14f_serving_stress" / "samples.jsonl")

    # 1. TTFT & ITL Distributions (from live HTTP stress concurrency sweep)
    conc_recs = [
        r for r in m_14f.get("records", []) if str(r.get("stage", "")).startswith("concurrency_")
    ]
    if conc_recs:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.5, 4.4))
        concs = [str(r["concurrency"]) for r in conc_recs]
        x = np.arange(len(concs))
        w = 0.25

        ax1.bar(x - w, [r["ttft_p50_ms"] for r in conc_recs], width=w, color=COLORS[0], label="p50")
        ax1.bar(x, [r["ttft_p95_ms"] for r in conc_recs], width=w, color=COLORS[4], label="p95")
        ax1.bar(x + w, [r["ttft_p99_ms"] for r in conc_recs], width=w, color=COLORS[1], label="p99")
        ax1.set_xticks(x)
        ax1.set_xticklabels([f"C={c}" for c in concs])
        ax1.set_ylabel("TTFT (ms)")
        ax1.set_title("(a) Live HTTP TTFT Percentiles")
        ax1.grid(axis="y", linestyle="--", alpha=0.35)
        ax1.legend(loc="upper left")

        ax2.bar(x - w, [r["itl_p50_ms"] for r in conc_recs], width=w, color=COLORS[0], label="p50")
        ax2.bar(x, [r["itl_p95_ms"] for r in conc_recs], width=w, color=COLORS[4], label="p95")
        ax2.bar(x + w, [r["itl_p99_ms"] for r in conc_recs], width=w, color=COLORS[1], label="p99")
        ax2.set_xticks(x)
        ax2.set_xticklabels([f"C={c}" for c in concs])
        ax2.set_ylabel("ITL / TPOT (ms)")
        ax2.set_ylim(0, max(float(r["itl_p99_ms"]) for r in conc_recs) * 1.2)
        ax2.set_title("(b) Live HTTP Inter-Token Latency Percentiles")
        ax2.grid(axis="y", linestyle="--", alpha=0.35)
        ax2.legend(loc="upper right")

        fig.suptitle("CuTileForge HTTP Serving Latency Distributions (127.0.0.1, Mistral-7B-GPTQ)")
        fig.tight_layout()
        p1 = output_dir / "fig_v310_1_ttft_itl_dist.png"
        fig.savefig(p1, dpi=300)
        plt.close(fig)
        saved.append(p1)

        # 2. Aggregate vs Per-Request Throughput under Concurrency
        fig, ax1 = plt.subplots(figsize=(7.8, 4.5))
        c_vals = [int(r["concurrency"]) for r in conc_recs]
        agg_y = [float(r["agg_tok_s"]) for r in conc_recs]
        req_y = [float(r["per_req_tok_s_p50"]) for r in conc_recs]

        l1 = ax1.plot(
            c_vals,
            agg_y,
            marker="o",
            linewidth=2.2,
            color=COLORS[0],
            label="Aggregate tok/s (left)",
        )
        ax1.set_xscale("log", base=2)
        ax1.set_xticks(c_vals)
        ax1.set_xticklabels([str(c) for c in c_vals])
        ax1.set_xlabel("Concurrent HTTP Clients")
        ax1.set_ylabel("Aggregate Throughput (tok/s)", color=COLORS[0])
        ax1.set_ylim(min(agg_y) * 0.85, max(agg_y) * 1.15)
        ax1.grid(True, linestyle="--", alpha=0.35)

        ax2 = ax1.twinx()
        l2 = ax2.plot(
            c_vals,
            req_y,
            marker="s",
            linestyle="--",
            linewidth=2.0,
            color=COLORS[1],
            label="Per-Request p50 tok/s (right)",
        )
        ax2.set_ylabel("Per-Request p50 Throughput (tok/s)", color=COLORS[1])
        ax2.set_ylim(-5, max(req_y) * 1.15)

        for cx, ay in zip(c_vals, agg_y, strict=True):
            ax1.annotate(
                f"{ay:.0f}",
                (cx, ay),
                textcoords="offset points",
                xytext=(0, 8),
                ha="center",
                fontsize=8.5,
                color=COLORS[0],
            )

        lines = l1 + l2
        labels = [str(ln.get_label()) for ln in lines]
        ax1.legend(lines, labels, loc="center left", frameon=True, framealpha=0.92)
        ax1.set_title("OpenAI HTTP Serving Throughput vs. Concurrency (Mistral-7B-GPTQ)")
        fig.tight_layout()
        p2 = output_dir / "fig_v310_2_concurrency_throughput.png"
        fig.savefig(p2, dpi=300)
        plt.close(fig)
        saved.append(p2)

    # 3. Memory Scaling vs Context Length & Sustained Soak RSS
    mistral_14c = [
        r
        for r in m_14c.get("records", [])
        if "Mistral-7B" in str(r.get("model", "")) and r.get("status") == "MEASURED"
    ]
    rss_samples = [s for s in samples_14f if s.get("metric") == "soak_server_rss_mb"]
    if mistral_14c or rss_samples:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.5, 4.4))
        for idx, kv_dt in enumerate(("fp16", "fp8_e4m3fn")):
            sub = sorted(
                [r for r in mistral_14c if r.get("kv_dtype") == kv_dt],
                key=lambda d: int(d["context_len"]),
            )
            if sub:
                ax1.plot(
                    [int(r["context_len"]) for r in sub],
                    [float(r["peak_vram_allocated_mb"]) / 1024.0 for r in sub],
                    marker="o",
                    linewidth=2.0,
                    color=COLORS[idx],
                    label=f"Peak Alloc VRAM ({kv_dt})",
                )
        ax1.set_xlabel("Context Length (tokens)")
        ax1.set_ylabel("Peak Allocated VRAM (GiB)")
        ax1.set_title("(a) Real Mistral-7B VRAM vs. Context")
        ax1.grid(True, linestyle="--", alpha=0.35)
        ax1.legend()

        if rss_samples:
            ax2.plot(
                [int(s["iteration"]) for s in rss_samples],
                [float(s["value"]) / 1024.0 for s in rss_samples],
                linewidth=1.8,
                color=COLORS[2],
                label="Server Host RSS (GiB)",
            )
            ax2.set_xlabel("Soak Check Sample Index")
            ax2.set_ylabel("Host RSS (GiB)")
            ax2.set_title("(b) Sustained Soak Memory Stability")
            ax2.grid(True, linestyle="--", alpha=0.35)
            ax2.legend()
        fig.tight_layout()
        p3 = output_dir / "fig_v310_3_memory_scaling.png"
        fig.savefig(p3, dpi=300)
        plt.close(fig)
        saved.append(p3)

    # 4. Quantization Accuracy vs Throughput (horizontal dual-panel to avoid any overlap)
    q_recs = [r for r in m_14e.get("records", []) if r.get("logit_cosine_sim") is not None]
    if q_recs:
        q_sorted = sorted(
            q_recs, key=lambda r: float(r.get("wikitext2_ppl_rel_diff_pct", 0.0)), reverse=False
        )
        labels_q = [
            f"{r['format_label']} ({'32L' if r.get('verification_category') == 'END_TO_END_VERIFIED' else '8L'})"
            for r in q_sorted
        ]
        ppls = [float(r.get("wikitext2_ppl_rel_diff_pct", 0.0)) for r in q_sorted]
        spds = [
            float(r.get("decode_tok_s") or r.get("decode_slice_tok_s") or 0.0) for r in q_sorted
        ]
        bar_clrs = [
            COLORS[0] if r.get("verification_category") == "END_TO_END_VERIFIED" else COLORS[4]
            for r in q_sorted
        ]
        y_pos = np.arange(len(q_sorted))

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.2, 5.2), sharey=True)
        ax1.barh(y_pos, ppls, color=bar_clrs)
        ax1.set_yticks(y_pos)
        ax1.set_yticklabels(labels_q, fontsize=8.5)
        ax1.set_xlabel("WikiText-2 PPL Relative Diff vs. FP32 Ref (%)")
        ax1.set_xlim(0, max(ppls) * 1.22)
        ax1.set_title("(a) Quantization PPL Drift (Lower is Better)")
        ax1.grid(axis="x", linestyle="--", alpha=0.35)
        for y, v in zip(y_pos, ppls, strict=True):
            ax1.text(v + max(ppls) * 0.015, y, f"{v:.4f}%", va="center", fontsize=8)

        ax2.barh(y_pos, spds, color=bar_clrs)
        ax2.set_xlabel("Measured Step Rate (tok/s or 8L slice-steps/s)")
        ax2.set_xlim(0, max(spds) * 1.2)
        ax2.set_title("(b) Step Rate (Blue=32L E2E, Orange=8L Slice)")
        ax2.grid(axis="x", linestyle="--", alpha=0.35)
        for y, v in zip(y_pos, spds, strict=True):
            ax2.text(v + max(spds) * 0.015, y, f"{v:.1f}", va="center", fontsize=8)

        fig.tight_layout()
        p4 = output_dir / "fig_v310_4_quant_accuracy_vs_throughput.png"
        fig.savefig(p4, dpi=300)
        plt.close(fig)
        saved.append(p4)

    # 5. Real vs Synthetic Long-Context Scaling
    syn_recs = m_14c_syn.get("records", [])
    if mistral_14c:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.5, 4.4))
        for idx, kv_dt in enumerate(("fp16", "fp8_e4m3fn")):
            sub = sorted(
                [r for r in mistral_14c if r.get("kv_dtype") == kv_dt],
                key=lambda d: int(d["context_len"]),
            )
            if sub:
                ax1.plot(
                    [int(r["context_len"]) for r in sub],
                    [float(r["itl_p50_ms"]) for r in sub],
                    marker="o",
                    color=COLORS[idx],
                    label=f"Real 32L Decode ITL p50 ({kv_dt})",
                )
        ax1.set_xlabel("Context Length (tokens)")
        ax1.set_ylabel("Decode ITL p50 (ms)")
        ax1.set_title("(a) Real Mistral-7B-GPTQ Full-Model Decode")
        ax1.grid(True, linestyle="--", alpha=0.35)
        ax1.legend()

        for idx, kv_dt in enumerate(("fp16", "fp8_e4m3fn")):
            sub_s = sorted(
                [r for r in syn_recs if r.get("kv_dtype") == kv_dt],
                key=lambda d: int(d["context_len"]),
            )
            if sub_s:
                ax2.plot(
                    [int(r["context_len"]) for r in sub_s],
                    [float(r["median_step_ms"]) for r in sub_s],
                    marker="s",
                    linestyle="--",
                    color=COLORS[idx + 2],
                    label=f"Synthetic 32L Attn Only ({kv_dt})",
                )
        ax2.set_xlabel("Context Length (tokens)")
        ax2.set_ylabel("Attention Kernel Step (ms)")
        ax2.set_title("(b) Synthetic FlashDecoding Kernel Only (Separate)")
        ax2.grid(True, linestyle="--", alpha=0.35)
        ax2.legend()
        fig.tight_layout()
        p5 = output_dir / "fig_v310_5_long_context_scaling.png"
        fig.savefig(p5, dpi=300)
        plt.close(fig)
        saved.append(p5)

    # 6. Measured External Engine Comparison ONLY (never plot NOT_RUN as zero)
    meas_ext = [
        r
        for r in m_14b.get("records", [])
        if r.get("status") in ("MEASURED", "DEGRADED")
        and float(r.get("throughput_tok_s") or 0.0) > 0.0
    ]
    if meas_ext:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.2, 4.7))
        m_sub = [r for r in meas_ext if "Mistral" in str(r.get("model", ""))]
        concs_m = sorted({int(r["concurrency"]) for r in m_sub})
        engs_m = sorted({str(r["engine"]) for r in m_sub})
        for idx, eng in enumerate(engs_m):
            vals = []
            for c in concs_m:
                match = next(
                    (
                        float(r["throughput_tok_s"])
                        for r in m_sub
                        if r["engine"] == eng and int(r["concurrency"]) == c
                    ),
                    None,
                )
                vals.append(match or 0.0)
            xs = np.arange(len(concs_m)) + (idx - 0.5) * 0.35
            bars1 = ax1.bar(xs, vals, width=0.35, color=COLORS[idx], label=eng)
            for b, v in zip(bars1, vals, strict=True):
                ax1.text(
                    b.get_x() + b.get_width() / 2.0,
                    v + 45.0,
                    f"{v:.0f}",
                    ha="center",
                    va="bottom",
                    fontsize=8.2,
                )
        ax1.set_xticks(range(len(concs_m)))
        ax1.set_xticklabels([f"C={c}" for c in concs_m])
        ax1.set_ylim(0, max(float(r["throughput_tok_s"]) for r in m_sub) * 1.15)
        ax1.set_ylabel("Throughput (tok/s)")
        ax1.set_title("(a) Mistral-7B-v0.1-GPTQ (vLLM rejected desc_act=True)")
        ax1.grid(axis="y", linestyle="--", alpha=0.35)
        ax1.legend()

        q_sub = [
            r
            for r in meas_ext
            if "Qwen" in str(r.get("model", "")) and int(r.get("concurrency", 1)) == 1
        ]
        if q_sub:
            label_map = {
                "HF Transformers 5.23.0 (Eager)": "HF\nTransformers\n5.23 (Eager)",
                "CuTileForge UnifiedModelExecutor": "CuTileForge\nUnified\n(no KV-cache)",
                "vLLM 0.34.0": "vLLM\n0.34.0",
                "SGLang 0.5.23": "SGLang\n0.5.23",
            }
            names = [label_map.get(str(r["engine"]), str(r["engine"])) for r in q_sub]
            vals_q = [float(r["throughput_tok_s"]) for r in q_sub]
            bars2 = ax2.bar(
                range(len(names)),
                vals_q,
                color=[COLORS[i % len(COLORS)] for i in range(len(names))],
            )
            for b, v in zip(bars2, vals_q, strict=True):
                ax2.text(
                    b.get_x() + b.get_width() / 2.0,
                    v + 12.0,
                    f"{v:.1f}",
                    ha="center",
                    va="bottom",
                    fontsize=8.5,
                )
            ax2.set_xticks(range(len(names)))
            ax2.set_xticklabels(names, fontsize=8.5)
            ax2.set_ylim(0, max(vals_q) * 1.15)
            ax2.set_ylabel("Throughput (tok/s)")
            ax2.set_title("(b) Qwen/Qwen2.5-0.5B FP16 (C=1, Honest Comparison)")
            ax2.grid(axis="y", linestyle="--", alpha=0.35)

        fig.tight_layout()
        p6 = output_dir / "fig_v310_6_external_engines.png"
        fig.savefig(p6, dpi=300)
        plt.close(fig)
        saved.append(p6)

    return saved


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate Phase 14 300-DPI validation figures")
    parser.add_argument(
        "--artifacts-root", type=Path, default=Path("benchmarks/validation_artifacts/v3.1.0")
    )
    parser.add_argument("--output-dir", type=Path, default=Path("docs/figures/validation_v310"))
    args = parser.parse_args()
    paths = generate_validation_figures(args.artifacts_root, args.output_dir)
    for p in paths:
        print(f"Saved figure: {p}")


if __name__ == "__main__":
    main()
