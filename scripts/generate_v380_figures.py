"""Generate 300-DPI validation figures for CuTileForge v3.8.0 from verified JSON artifacts."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def _plot_parity_and_kv(out_dir: Path) -> None:
    reg = json.loads(
        Path(
            "benchmarks/validation_artifacts/v3.8.0/phase22_regression/regress_full.json"
        ).read_text()
    )
    perf = json.loads(
        Path("benchmarks/validation_artifacts/v3.8.0/phase22_perf_mem.json").read_text()
    )

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.4), dpi=300)
    fig.patch.set_facecolor("#ffffff")

    # Panel A: 8-Model Strict Parity Status (Cosine Similarity & Gate Outcome)
    results = reg["results"]
    models = [r["model"] for r in results]
    cos_vals = []
    statuses = [r["status"] for r in results]
    for r in results:
        m = r["metrics"]
        cos_vals.append(
            float(m.get("logit_cos_sim") or m.get("emb_cos_sim") or m.get("output_cos_sim") or 0.0)
        )

    y_pos = np.arange(len(models))
    colors = ["#1b7f4b" if s == "PASS" else "#c0392b" for s in statuses]
    ax1.barh(y_pos, cos_vals, color=colors, height=0.62, edgecolor="#222222", linewidth=0.6)
    ax1.set_xlim(0.999968, 1.000018)
    ax1.set_xticks([0.99997, 0.99998, 0.99999, 1.00000])
    ax1.set_xticklabels(["0.99997", "0.99998", "0.99999", "1.00000"], fontsize=8.5)
    ax1.set_yticks(y_pos)
    ax1.set_yticklabels(models, fontsize=9, fontweight="bold")
    ax1.invert_yaxis()
    ax1.set_xlabel(
        "Cosine Similarity vs Reference (all >= 0.9999 gate)", fontsize=9.5, fontweight="bold"
    )
    ax1.set_title(
        "A. 8-Model Regression (v3.8.0): 2 PASS / 6 FAIL\n(All 8 execute natively; strict gates unchanged)",
        fontsize=10.5,
        fontweight="bold",
        pad=10,
    )
    ax1.grid(axis="x", linestyle=":", alpha=0.5)

    for i, r in enumerate(results):
        s = r["status"]
        m = r["metrics"]
        if r["model"] == "LLaVA-1.5":
            tag = "PASS (1/1, 16 tok)"
        elif s == "PASS":
            tag = f"PASS (diff={m['max_abs_diff']:.5f})"
        elif "greedy_exact_matches" in m:
            tag = f"FAIL ({m['greedy_exact_matches']}/{m['greedy_total_prompts']} greedy)"
        else:
            tag = f"FAIL (diff={m['max_abs_diff']:.4f})"
        ax1.text(
            min(cos_vals[i] + 0.0000015, 1.000002),
            i,
            tag,
            va="center",
            ha="left",
            fontsize=7.8,
            color="#111111",
            fontweight="bold" if s == "PASS" else "normal",
        )

    # Panel B: Phase 22 Before vs After Kernel Latencies
    k_before = perf["before"]["kernels"]
    k_after = perf["after"]["kernels"]
    k_delta = perf["delta"]["kernels"]
    labels = [
        "GQA\nskv=128\nsw=0",
        "GQA\nskv=1024\nsw=0",
        "GQA\nskv=2047\nsw=2040",
        "FlashDec\nskv=2048\nsw=2040",
        "FlashDec\nskv=4096\nsw=4096",
    ]
    keys = [
        "fused_attn_gqa_skv128_sw0",
        "fused_attn_gqa_skv1024_sw0",
        "fused_attn_gqa_skv2047_sw2040",
        "flash_decoding_skv2048_sw2040",
        "flash_decoding_skv4096_sw4096",
    ]
    b_vals = [k_before[k]["median_us"] for k in keys]
    a_vals = [k_after[k]["median_us"] for k in keys]
    d_pcts = [k_delta[k]["delta_pct"] for k in keys]

    x = np.arange(len(keys))
    w = 0.36
    rects1 = ax2.bar(
        x - w / 2,
        b_vals,
        w,
        label="Before Fix (v3.7.1)",
        color="#7f8c8d",
        edgecolor="#222222",
        linewidth=0.6,
    )
    rects2 = ax2.bar(
        x + w / 2,
        a_vals,
        w,
        label="After Fix (v3.8.0)",
        color="#1f6feb",
        edgecolor="#222222",
        linewidth=0.6,
    )
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels, fontsize=8.5)
    ax2.set_ylabel("Median Latency (us, lower is better)", fontsize=9.5, fontweight="bold")
    ax2.set_ylim(0, max(b_vals) * 1.24)
    ax2.set_title(
        "B. Phase 22 Attention & Sliding-Window Fix\n(RTX 5090, synthetic-weights, 200 CUDA events)",
        fontsize=10.5,
        fontweight="bold",
        pad=10,
    )
    ax2.grid(axis="y", linestyle=":", alpha=0.5)
    ax2.legend(loc="upper left", fontsize=8.5)

    for r1, r2, pct in zip(rects1, rects2, d_pcts, strict=True):
        ax2.text(
            r1.get_x() + r1.get_width() / 2,
            r1.get_height() + 0.8,
            f"{r1.get_height():.1f}",
            ha="center",
            va="bottom",
            fontsize=7.2,
            color="#444444",
        )
        ax2.text(
            r2.get_x() + r2.get_width() / 2,
            r2.get_height() + 0.8,
            f"{r2.get_height():.1f}\n({pct:+.1f}%)",
            ha="center",
            va="bottom",
            fontsize=7.2,
            fontweight="bold",
            color="#1b7f4b" if pct < 0 else "#333333",
        )

    plt.tight_layout()
    fig.savefig(out_dir / "fig_v380_parity_and_kv.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def _plot_external_engines(out_dir: Path) -> None:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.5, 4.8), dpi=300)
    fig.patch.set_facecolor("#ffffff")

    # Panel A: Mistral-7B-v0.1-GPTQ (desc_act=True)
    concs = ["C=1", "C=4", "C=16"]
    sglang_m = [265.01, 969.40, 3261.84]
    cutile_m = [167.03, 621.53, 2308.02]
    x = np.arange(len(concs))
    w = 0.35
    b1 = ax1.bar(
        x - w / 2,
        sglang_m,
        w,
        label="SGLang 0.5.23",
        color="#e67e22",
        edgecolor="#222222",
        linewidth=0.6,
    )
    b2 = ax1.bar(
        x + w / 2,
        cutile_m,
        w,
        label="CuTileForge Native",
        color="#1f6feb",
        edgecolor="#222222",
        linewidth=0.6,
    )
    ax1.set_xticks(x)
    ax1.set_xticklabels(concs, fontsize=9.5, fontweight="bold")
    ax1.set_ylabel("Throughput (tok/s, higher is better)", fontsize=9.5, fontweight="bold")
    ax1.set_ylim(0, 3800)
    ax1.set_title(
        "A. Mistral-7B-v0.1-GPTQ (desc_act=True)\nSGLang leads; vLLM 0.34.0 UNSUPPORTED",
        fontsize=10.5,
        fontweight="bold",
        pad=10,
    )
    ax1.grid(axis="y", linestyle=":", alpha=0.5)
    ax1.legend(loc="upper left", fontsize=8.5)
    for rect in list(b1) + list(b2):
        ax1.text(
            rect.get_x() + rect.get_width() / 2,
            rect.get_height() + 55,
            f"{rect.get_height():,.0f}",
            ha="center",
            va="bottom",
            fontsize=8,
            fontweight="bold",
        )

    # Panel B: Qwen/Qwen2.5-0.5B (FP16, C=1)
    engines = ["SGLang\n0.5.23", "vLLM\n0.34.0", "CuTileForge\n(DEGRADED)", "HF Eager\n5.23.0"]
    tok_s = [803.05, 694.12, 170.27, 166.56]
    colors = ["#e67e22", "#8e44ad", "#c0392b", "#7f8c8d"]
    x2 = np.arange(len(engines))
    bars = ax2.bar(x2, tok_s, width=0.55, color=colors, edgecolor="#222222", linewidth=0.6)
    ax2.set_xticks(x2)
    ax2.set_xticklabels(engines, fontsize=9, fontweight="bold")
    ax2.set_ylabel("Single-Stream C=1 (tok/s)", fontsize=9.5, fontweight="bold")
    ax2.set_ylim(0, 940)
    ax2.set_title(
        "B. Qwen/Qwen2.5-0.5B (FP16, C=1)\nUnifiedModelExecutor lacks paged KV + CUDA Graph",
        fontsize=10.5,
        fontweight="bold",
        pad=10,
    )
    ax2.grid(axis="y", linestyle=":", alpha=0.5)
    for rect in bars:
        ax2.text(
            rect.get_x() + rect.get_width() / 2,
            rect.get_height() + 18,
            f"{rect.get_height():.1f}",
            ha="center",
            va="bottom",
            fontsize=8.5,
            fontweight="bold",
        )

    plt.tight_layout()
    fig.savefig(out_dir / "fig_v380_external_engines.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    out_dir = Path("docs/figures/validation_v380")
    out_dir.mkdir(parents=True, exist_ok=True)
    _plot_parity_and_kv(out_dir)
    _plot_external_engines(out_dir)
    print(f"Generated v3.8.0 figures in {out_dir}")


if __name__ == "__main__":
    main()
