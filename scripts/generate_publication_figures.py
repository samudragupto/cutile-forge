"""Generate 300-DPI publication-grade multi-panel figures for CuTileForge v3.0.0."""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

NAVY = "#1b365d"
CORAL = "#d95f02"
TEAL = "#008080"
SLATE = "#6c757d"
GOLD = "#e6ab02"
CRIMSON = "#b91c1c"

# Okabe-Ito colorblind-safe palette (Phase 8 / v2.0.0 figures)
OI_BLUE = "#0072B2"
OI_ORANGE = "#D55E00"
OI_GREEN = "#009E73"
OI_SKY = "#56B4E9"
OI_YELLOW = "#E69F00"
OI_PURPLE = "#CC79A7"
OI_GRAY = "#64748B"


def _apply_pub_style() -> None:
    plt.rcParams.update(
        {
            "figure.dpi": 300,
            "savefig.dpi": 300,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "font.size": 10,
            "axes.titlesize": 11.5,
            "axes.titleweight": "bold",
            "axes.labelsize": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.linestyle": "--",
            "grid.alpha": 0.3,
            "legend.frameon": True,
            "legend.facecolor": "white",
            "legend.edgecolor": "#e2e8f0",
        }
    )


def generate_fig1_executive_overview(out_dir: Path) -> Path:
    """Figure 1: 2x2 Executive Overview (Decode Progression, Real Checkpoints, Long-Ctx, Serving)."""
    fig, axes = plt.subplots(2, 2, figsize=(16, 11), constrained_layout=True)

    # (a) End-to-End Decode & Speculative Throughput Progression
    ax = axes[0, 0]
    labels = [
        "v0.1.0\nAda Laptop\n(sm_89)",
        "v0.2.0\nBlackwell\nEager TFG",
        "v0.3.0\nBlackwell\nCUDA Graph",
        "v0.5.0\n30-Min\nSustained",
        "v0.3.0+\nN-Gram\nSpec (γ=4)",
    ]
    vals = [32.43, 49.82, 96.57, 99.73, 117.31]
    mults = ["1.00×", "1.54×", "2.98×", "3.08×", "3.62×"]
    colors = [SLATE, GOLD, TEAL, NAVY, CORAL]
    x = np.arange(len(labels))
    bars = ax.bar(x, vals, width=0.58, color=colors, edgecolor="none")
    for bar, v, m in zip(bars, vals, mults, strict=True):
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            v + 2.8,
            f"{v:.2f}\n({m})",
            ha="center",
            va="bottom",
            fontsize=8.5,
            fontweight="bold",
            color="#0f172a",
        )
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8.5)
    ax.set_ylabel("Throughput (tok/s)")
    ax.set_ylim(0, 152)
    ax.set_title("(a) End-to-End Decode Progression (Mistral-7B 4-bit GPTQ)")

    # (b) Real Checkpoint Scale-Up & Memory Partitioning on 32 GB RTX 5090
    ax = axes[0, 1]
    models = [
        "Mistral-7B-v0.1\n(4.16 GB disk)\n96.57 tok/s",
        "Mixtral-8x7B\n(23.8 GB, 8 exp)\n41.83 tok/s · PPL 7.48",
        "Llama-2-70B-Chat\n(35.3 GB, 80L)\n4.07 tok/s · 88.1% ovlp",
    ]
    hbm_gb = [4.32, 25.26, 27.44]
    ddr5_gb = [0.0, 1.93, 9.08]
    x = np.arange(len(models))
    w = 0.35
    b1 = ax.bar(x - w / 2, hbm_gb, w, label="GPU HBM VRAM (GB)", color=NAVY)
    b2 = ax.bar(x + w / 2, ddr5_gb, w, label="Pinned Host DDR5 RSS (GB)", color=TEAL)
    ax.axhline(32.0, color=CRIMSON, linestyle="--", linewidth=1.6, label="32.0 GB HBM Ceiling")
    for bar, v in zip(b1, hbm_gb, strict=True):
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            v + 0.7,
            f"{v:.2f} GB",
            ha="center",
            va="bottom",
            fontsize=8.5,
            fontweight="bold",
            color=NAVY,
        )
    for bar, v in zip(b2, ddr5_gb, strict=True):
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            v + 0.7,
            f"{v:.2f} GB",
            ha="center",
            va="bottom",
            fontsize=8.5,
            fontweight="bold",
            color=TEAL,
        )
    ax.set_xticks(x)
    ax.set_xticklabels(models, fontsize=8.5)
    ax.set_ylabel("Memory Footprint (GB)")
    ax.set_ylim(0, 42)
    ax.set_title("(b) Real Checkpoint Scale-Up & Hybrid Memory Partitioning (sm_120)")
    ax.legend(loc="upper left", fontsize=8.5)

    # (c) 16K & 32K Long-Context Paging Latency Ablation (Log Scale)
    ax = axes[1, 0]
    stages = [
        "v0.2.0\nSync pin_memory\nFault",
        "v0.3.0\nAsync Pinned Pool\n(swap_stream)",
        "v0.7.0\n+ FlashDecoding\n(Split-KV=16)",
    ]
    ctx16 = [107854.66, 76.29, 59.20]
    ctx32 = [1315.34, 214.67, 219.57]
    x = np.arange(len(stages))
    w = 0.35
    b16 = ax.bar(x - w / 2, ctx16, w, label="16K Context (ms/step)", color=CORAL)
    b32 = ax.bar(x + w / 2, ctx32, w, label="32K Context (ms/step)", color=NAVY)
    ax.set_yscale("log")
    for bar, v in zip(b16, ctx16, strict=True):
        lbl = f"{v:,.0f}" if v > 1000 else f"{v:.1f}"
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            v * 1.22,
            lbl,
            ha="center",
            va="bottom",
            fontsize=8.5,
            fontweight="bold",
            color=CORAL,
        )
    for bar, v in zip(b32, ctx32, strict=True):
        lbl = f"{v:,.0f}" if v > 1000 else f"{v:.1f}"
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            v * 1.22,
            lbl,
            ha="center",
            va="bottom",
            fontsize=8.5,
            fontweight="bold",
            color=NAVY,
        )
    ax.annotate(
        "1,822× faster\n@ 16K ctx",
        xy=(2 - w / 2, 130.0),
        xytext=(1.25, 14000),
        arrowprops={"arrowstyle": "->", "color": TEAL, "lw": 1.6},
        fontsize=9,
        fontweight="bold",
        color=TEAL,
        ha="center",
    )
    ax.set_xticks(x)
    ax.set_xticklabels(stages, fontsize=8.5)
    ax.set_ylabel("Decode Step Latency (ms, log scale)")
    ax.set_ylim(10, 600000)
    ax.set_title("(c) 16K & 32K Long-Context Paging Latency Ablation")
    ax.legend(loc="upper right", fontsize=8.5)

    # (d) Concurrent Serving Latency Staircase
    ax = axes[1, 1]
    reqs = np.array([1, 2, 3, 4])
    serial_ms = [101.1, 189.4, 277.1, 363.1]
    batched_ms = [113.6, 111.7, 111.1, 108.7]
    ax.plot(
        reqs,
        serial_ms,
        marker="o",
        linewidth=2.2,
        color=CORAL,
        label="v0.5.0 Serial Lock (p99 = 363.1 ms)",
    )
    ax.plot(
        reqs,
        batched_ms,
        marker="s",
        linewidth=2.2,
        color=TEAL,
        label="v0.6.0 DynamicBatcher (p99 = 113.6 ms)",
    )
    for r, v in zip(reqs, serial_ms, strict=True):
        dy = -22 if r == 1 else 14
        ax.text(r, v + dy, f"{v:.1f} ms", ha="center", fontsize=8.5, fontweight="bold", color=CORAL)
    for r, v in zip(reqs, batched_ms, strict=True):
        dy = 15 if r == 1 else -22
        ax.text(r, v + dy, f"{v:.1f} ms", ha="center", fontsize=8.5, fontweight="bold", color=TEAL)
    ax.annotate(
        "3.20× lower\np99 tail latency",
        xy=(4, 116.0),
        xytext=(2.85, 225),
        arrowprops={"arrowstyle": "->", "color": NAVY, "lw": 1.5},
        fontsize=9,
        fontweight="bold",
        color=NAVY,
        ha="center",
    )
    ax.set_xticks(reqs)
    ax.set_xticklabels([f"Req {r}" for r in reqs])
    ax.set_ylabel("End-to-End Request Latency (ms)")
    ax.set_ylim(40, 440)
    ax.set_title("(d) 4-Way Concurrent Serving Latency (max_tokens=8)")
    ax.legend(loc="upper left", fontsize=8.5)

    out_path = out_dir / "fig1_executive_overview.png"
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()
    return out_path


def generate_fig2_roofline(out_dir: Path) -> Path:
    """Figure 2: Dual-Architecture Empirical Log-Log Roofline (Ada sm_89 vs Blackwell sm_120)."""
    fig, ax = plt.subplots(figsize=(13, 8), constrained_layout=True)
    ai = np.logspace(np.log10(0.5), np.log10(1600.0), 500)

    # Ceilings
    bw_5090_cublas = np.minimum(1.792 * ai, 289.0)
    bw_5090_cutile = np.minimum(1.792 * ai, 150.3)
    bw_4060_cublas = np.minimum(0.272 * ai, 45.7)
    bw_4060_cutile = np.minimum(0.272 * ai, 28.2)

    ax.loglog(
        ai,
        bw_5090_cublas,
        "-",
        color=NAVY,
        lw=2.2,
        label="RTX 5090 cuBLAS FP16 (1,792 GB/s, 289 TF)",
    )
    ax.loglog(
        ai, bw_5090_cutile, "--", color=TEAL, lw=2.0, label="RTX 5090 CuTile FP32-acc (150.3 TF)"
    )
    ax.loglog(
        ai,
        bw_4060_cublas,
        "-",
        color=CORAL,
        lw=1.8,
        label="RTX 4060 Laptop cuBLAS FP16 (272 GB/s, 45.7 TF)",
    )
    ax.loglog(
        ai,
        bw_4060_cutile,
        "--",
        color=GOLD,
        lw=1.8,
        label="RTX 4060 Laptop CuTile FP32-acc (28.2 TF)",
    )

    # Ridge points
    ax.scatter([150.3 / 1.792], [150.3], color=TEAL, marker="D", s=50, zorder=6)
    ax.scatter([28.2 / 0.272], [28.2], color=GOLD, marker="D", s=50, zorder=6)

    arrow_kw = {"arrowstyle": "->", "lw": 0.85, "color": "#475569", "shrinkA": 2, "shrinkB": 4}

    # Explicit non-overlapping offsets (dx, dy) + horizontal alignment
    sm120_pts = [
        (2.0, 3.22, "Decode GEMV (M=1)\n3.22 TF", (14, -18), "left"),
        (37.5, 42.13, "Grouped MoE 8x7B (M=256)\n42.13 TF", (-36, 26), "right"),
        (37.5, 4.91, "v0.4.0 Sparse MoE\n4.91 TF", (14, -14), "left"),
        (204.8, 101.62, "GQA FlashAttn-2 (1024)\n101.62 TF", (-20, -28), "right"),
        (240.0, 136.14, "FP16 GEMM (256×14336)\n136.14 TF", (-52, 54), "right"),
        (341.0, 150.26, "FP16 GEMM (512×4096)\n150.26 TF", (0, 25), "center"),
        (410.0, 132.58, "INT4 Dequant+GEMM\n132.58 TF", (22, -4), "left"),
        (410.0, 88.56, "NVFP4 (e2m1) LUT\n88.56 TF", (22, -16), "left"),
    ]
    sm89_pts = [
        (204.8, 21.76, "Ada GQA Attn\n21.76 TF", (-18, -26), "right"),
        (240.0, 28.18, "Ada FP16 GEMM\n28.18 TF", (-12, 24), "right"),
        (410.0, 23.37, "Ada INT4 GEMM\n23.37 TF", (20, -14), "left"),
    ]

    for x_v, y_v, txt, (dx, dy), ha_val in sm120_pts:
        ax.scatter([x_v], [y_v], color=NAVY, marker="o", s=72, zorder=7, edgecolors="white", lw=0.8)
        ax.annotate(
            txt,
            (x_v, y_v),
            xytext=(dx, dy),
            textcoords="offset points",
            fontsize=8.2,
            fontweight="bold",
            color=NAVY,
            ha=ha_val,
            va="center",
            arrowprops=arrow_kw,
        )

    for x_v, y_v, txt, (dx, dy), ha_val in sm89_pts:
        ax.scatter(
            [x_v], [y_v], color=CORAL, marker="^", s=72, zorder=7, edgecolors="white", lw=0.8
        )
        ax.annotate(
            txt,
            (x_v, y_v),
            xytext=(dx, dy),
            textcoords="offset points",
            fontsize=8.2,
            fontweight="bold",
            color=CORAL,
            ha=ha_val,
            va="center",
            arrowprops=arrow_kw,
        )

    # Upward arrow for MoE optimization jump
    ax.annotate(
        "",
        xy=(37.5, 39.0),
        xytext=(37.5, 5.6),
        arrowprops={"arrowstyle": "->", "color": CRIMSON, "lw": 2.0},
    )
    ax.text(
        29.0,
        14.0,
        "8.6× @ M=64\n68.4× @ M=256",
        fontsize=8.5,
        fontweight="bold",
        color=CRIMSON,
        ha="right",
    )

    ax.set_xlabel("Arithmetic Intensity (FLOPs / HBM Byte)")
    ax.set_ylabel("Achieved Throughput (TFLOPS)")
    ax.set_xlim(0.5, 1600)
    ax.set_ylim(0.5, 520)
    ax.set_title("Empirical Log-Log Roofline: Ada Lovelace (sm_89) vs. Blackwell (sm_120)")
    ax.legend(loc="upper left", fontsize=8.5)

    out_path = out_dir / "roofline_ada_vs_blackwell.png"
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()
    return out_path


def generate_fig3_microbenchmarks(out_dir: Path) -> Path:
    """Figure 3: 2x2 Micro-Benchmark Suite (GEMM, INT4/NVFP4, FFN Cost Model, GQA Attention)."""
    fig, axes = plt.subplots(2, 2, figsize=(16, 11), constrained_layout=True)

    # (a) Prefill GEMM Throughput (sm_120)
    ax = axes[0, 0]
    shapes = ["256×12288×4096", "256×14336×4096", "512×4096×4096"]
    ct_tf = [107.91, 136.14, 150.26]
    cu_tf = [120.77, 122.24, 130.90]
    pt_tf = [177.65, 186.12, 169.01]
    cb_tf = [269.20, 288.91, 312.68]
    x = np.arange(len(shapes))
    w = 0.2
    for offset, vals, lbl, col in [
        (-1.5 * w, ct_tf, "CuTile (FP32-acc)", TEAL),
        (-0.5 * w, cu_tf, "CUDA WMMA", GOLD),
        (0.5 * w, pt_tf, "PyTorch", CORAL),
        (1.5 * w, cb_tf, "cuBLAS (FP16-acc)", NAVY),
    ]:
        bars = ax.bar(x + offset, vals, w, label=lbl, color=col)
        for b, v in zip(bars, vals, strict=True):
            ax.text(
                b.get_x() + b.get_width() / 2.0,
                v + 5,
                f"{v:.0f}",
                ha="center",
                fontsize=8,
                fontweight="bold",
            )
    ax.set_xticks(x)
    ax.set_xticklabels(shapes)
    ax.set_ylabel("Throughput (TFLOPS)")
    ax.set_ylim(0, 385)
    ax.set_title("(a) Prefill FP16 GEMM Throughput (RTX 5090, sm_120)")
    ax.legend(loc="upper left", fontsize=8.5)

    # (b) Fused INT4 Dequant+GEMM Speedup vs Separate
    ax = axes[0, 1]
    dq_shapes = [
        "1×12288×4096\n(Decode)",
        "256×12288×4096\n(Prefill)",
        "1×14336×4096\n(Decode)",
        "256×14336×4096\n(Prefill)",
    ]
    speedups = [4.69, 1.36, 5.15, 1.62]
    tflops_lbl = ["2.92 TF", "112.5 TF", "3.22 TF", "132.6 TF"]
    cols = [NAVY, TEAL, NAVY, TEAL]
    x = np.arange(len(dq_shapes))
    bars = ax.bar(x, speedups, width=0.55, color=cols)
    ax.axhline(1.0, color=SLATE, linestyle="--", lw=1.2)
    for b, sp, tf in zip(bars, speedups, tflops_lbl, strict=True):
        ax.text(
            b.get_x() + b.get_width() / 2.0,
            sp + 0.14,
            f"{sp:.2f}×\n({tf})",
            ha="center",
            fontsize=8.5,
            fontweight="bold",
        )
    ax.set_xticks(x)
    ax.set_xticklabels(dq_shapes, fontsize=8.5)
    ax.set_ylabel("Speedup vs. Separate Dequant + GEMM (×)")
    ax.set_ylim(0, 6.6)
    ax.set_title("(b) Fused INT4 Dequant+GEMM Speedup (sm_120)")

    # (c) Occupancy-Aware FFN Cost Model Ablation
    ax = axes[1, 0]
    seqs = ["seq=1", "seq=64", "seq=256", "seq=512"]
    v020 = [8.04, 2.70, 0.86, 3.94]
    v030 = [8.38, 2.90, 1.55, 3.87]
    x = np.arange(len(seqs))
    w = 0.35
    b1 = ax.bar(x - w / 2, v020, w, label="v0.2.0 (Static 3-Op)", color=CORAL)
    b2 = ax.bar(x + w / 2, v030, w, label="v0.3.0+ (Occupancy Cost Model)", color=TEAL)
    ax.axhline(1.0, color=CRIMSON, linestyle="--", lw=1.3, label="1.0× Parity")
    for b, v in zip(b1, v020, strict=True):
        # Put label inside the bar when below the 1.0x parity line so it never touches the dashed line
        if v < 1.0:
            ax.text(
                b.get_x() + b.get_width() / 2.0,
                v * 0.5,
                f"{v:.2f}×",
                ha="center",
                va="center",
                fontsize=8,
                fontweight="bold",
                color="white",
            )
        else:
            ax.text(
                b.get_x() + b.get_width() / 2.0,
                v + 0.18,
                f"{v:.2f}×",
                ha="center",
                fontsize=8,
                fontweight="bold",
                color=CORAL,
            )
    for b, v in zip(b2, v030, strict=True):
        ax.text(
            b.get_x() + b.get_width() / 2.0,
            v + 0.18,
            f"{v:.2f}×",
            ha="center",
            fontsize=8,
            fontweight="bold",
            color=TEAL,
        )
    ax.set_xticks(x)
    ax.set_xticklabels(seqs)
    ax.set_ylabel("Speedup vs. 3 Separate CUDA Kernels (×)")
    ax.set_ylim(0, 10.5)
    ax.set_title("(c) Occupancy-Aware FFN Cost Model Ablation (sm_120)")
    ax.legend(loc="upper right", fontsize=8.5)

    # (d) GQA FlashAttention-2 (32:8) Scaling
    ax = axes[1, 1]
    a_seqs = ["seq=64", "seq=256", "seq=512", "seq=1024"]
    ct_a = [3.85, 19.63, 59.05, 101.62]
    pt_a = [0.77, 21.12, 50.82, 75.51]
    cu_a = [1.28, 4.78, 5.74, 2.35]
    x = np.arange(len(a_seqs))
    w = 0.25
    b_ct = ax.bar(x - w, ct_a, w, label="CuTile FlashAttn-2", color=NAVY)
    b_pt = ax.bar(x, pt_a, w, label="PyTorch SDPA", color=TEAL)
    ax.bar(x + w, cu_a, w, label="CUDA C++ Baseline", color=GOLD)
    for i, (b, v) in enumerate(zip(b_ct, ct_a, strict=True)):
        x_off = -0.04 if i == 1 else 0.0
        ax.text(
            b.get_x() + b.get_width() / 2.0 + x_off,
            v + 2.2,
            f"{v:.1f}",
            ha="center",
            fontsize=8,
            fontweight="bold",
            color=NAVY,
        )
    for i, (b, v) in enumerate(zip(b_pt, pt_a, strict=True)):
        x_off = 0.04 if i == 1 else 0.0
        ax.text(
            b.get_x() + b.get_width() / 2.0 + x_off,
            v + 2.2,
            f"{v:.1f}",
            ha="center",
            fontsize=8,
            fontweight="bold",
            color=TEAL,
        )
    ax.set_xticks(x)
    ax.set_xticklabels(a_seqs)
    ax.set_ylabel("Throughput (TFLOPS)")
    ax.set_ylim(0, 128)
    ax.set_title("(d) GQA FlashAttention-2 (32:8 Heads) Scaling (sm_120)")
    ax.legend(loc="upper left", fontsize=8.5)

    out_path = out_dir / "fig3_microbenchmark_suite.png"
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()
    return out_path


def generate_fig4_systems_telemetry(out_dir: Path) -> Path:
    """Figure 4: 1x2 Systems Telemetry (30-Min Sustained Stability & Stage -1 PCIe DMA Overlap)."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6), constrained_layout=True)

    # (a) 30-Minute Sustained Throughput Stability
    csv_5090 = Path("benchmarks/results_5090/sustained/thermal.csv")
    ts_5090, tps_5090 = [], []
    if csv_5090.exists():
        with csv_5090.open("r", encoding="utf-8") as f:
            for r in csv.DictReader(f):
                ts_5090.append(float(r["timestamp_s"]) / 60.0)
                tps_5090.append(float(r["tok_s"]))

    if ts_5090:
        avg_5090 = float(np.mean(tps_5090))
        ax1.plot(
            ts_5090,
            tps_5090,
            color=NAVY,
            lw=1.8,
            label=f"RTX 5090 v0.5.0+ (361 samples, {avg_5090:.2f} tok/s avg, 0.25% drift)",
        )
        ax1.axhline(avg_5090, color=TEAL, linestyle="--", lw=1.2)

    csv_4060 = Path("benchmarks/results/sustained/thermal.csv")
    if csv_4060.exists():
        ts_4060, tps_4060 = [], []
        with csv_4060.open("r", encoding="utf-8") as f:
            for r in csv.DictReader(f):
                ts_4060.append(float(r["timestamp_s"]) / 60.0)
                tps_4060.append(float(r["tok_s"]))
        ax1.plot(
            ts_4060,
            tps_4060,
            color=CORAL,
            lw=1.6,
            label="RTX 4060 Laptop (31.16 tok/s avg, 83.0 °C max)",
        )
    else:
        ax1.axhline(
            31.16,
            color=CORAL,
            linestyle="-.",
            lw=1.8,
            label="RTX 4060 Laptop Avg: 31.16 tok/s (83.0 °C max, 4.62% throttle)",
        )

    ax1.set_xlabel("Elapsed Time (minutes)")
    ax1.set_ylabel("Decode Throughput (tok/s)")
    ax1.set_xlim(0, 30.2)
    ax1.set_ylim(0, 120)
    ax1.set_title("(a) 30-Minute Sustained Throughput Stability (1,800s)")
    ax1.legend(loc="center right", fontsize=8.5)

    # (b) Stage -1 PCIe DMA Overlap & Bus Saturation
    labels = [
        "Ada PCIe 4.0 x8\nBus Util (%)\n(13.30 GB/s)",
        "Ada Stage -1\nMicro Overlap\n(115 MB tile)",
        "Blackwell PCIe 5.0\nMicro Overlap\n(49.09 GB/s)",
        "Blackwell Real 70B\n25-Layer Overlap\n(9.74 GB/tok)",
    ]
    pcts = [84.4, 93.8, 93.2, 88.1]
    cols = [GOLD, CORAL, TEAL, NAVY]
    x = np.arange(len(labels))
    bars = ax2.bar(x, pcts, width=0.55, color=cols)
    for b, p in zip(bars, pcts, strict=True):
        ax2.text(
            b.get_x() + b.get_width() / 2.0,
            p + 2.0,
            f"{p:.1f}%",
            ha="center",
            fontsize=9.5,
            fontweight="bold",
        )
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels, fontsize=8.5)
    ax2.set_ylabel("Overlap / Bus Utilization (%)")
    ax2.set_ylim(0, 118)
    ax2.set_title("(b) Stage -1 PCIe DMA Overlap & Bus Saturation")

    out_path = out_dir / "fig4_systems_telemetry.png"
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()
    return out_path


def generate_real_models_v100(out_dir: Path) -> Path:
    """Real Checkpoint Evaluation (Mistral-7B, Mixtral-8x7B MoE, Llama-2-70B Hybrid)."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6), constrained_layout=True)

    models = [
        "Mistral-7B-v0.1\n(4.16 GB · 32 HBM / 0 PCIe)",
        "Mixtral-8x7B-v0.1\n(23.8 GB · 32 HBM / 0 PCIe)",
        "Llama-2-70B-Chat\n(35.3 GB · 55 HBM / 25 PCIe)",
    ]
    decode_tps = [96.57, 41.83, 4.07]
    prefill_tps = [4120.0, 2786.68, 859.19]

    # (a) Decode vs Prefill Throughput (Log Scale)
    x = np.arange(len(models))
    w = 0.35
    b1 = ax1.bar(x - w / 2, prefill_tps, w, label="Prefill Throughput (tok/s)", color=NAVY)
    b2 = ax1.bar(x + w / 2, decode_tps, w, label="Decode Throughput (tok/s)", color=TEAL)
    ax1.set_yscale("log")
    for b, v in zip(b1, prefill_tps, strict=True):
        ax1.text(
            b.get_x() + b.get_width() / 2.0,
            v * 1.18,
            f"{v:,.1f}" if v >= 100 else f"{v:.2f}",
            ha="center",
            va="bottom",
            fontsize=8.5,
            fontweight="bold",
            color=NAVY,
        )
    for b, v in zip(b2, decode_tps, strict=True):
        ax1.text(
            b.get_x() + b.get_width() / 2.0,
            v * 1.18,
            f"{v:.2f}",
            ha="center",
            va="bottom",
            fontsize=8.5,
            fontweight="bold",
            color=TEAL,
        )
    ax1.set_xticks(x)
    ax1.set_xticklabels(models, fontsize=8.5)
    ax1.set_ylabel("Throughput (tok/s, log scale)")
    ax1.set_ylim(1, 18000)
    ax1.set_title("(a) Real 4-Bit GPTQ Checkpoint Throughput (RTX 5090)")
    ax1.legend(loc="upper right", fontsize=8.5)

    # (b) HBM vs Pinned Host DDR5 Memory Partitioning
    hbm_gb = [4.32, 25.26, 27.44]
    ddr5_gb = [0.0, 1.93, 9.08]
    b3 = ax2.bar(x - w / 2, hbm_gb, w, label="GPU HBM VRAM (GB)", color=CORAL)
    b4 = ax2.bar(x + w / 2, ddr5_gb, w, label="Pinned Host DDR5 RSS (GB)", color=GOLD)
    ax2.axhline(32.0, color=CRIMSON, linestyle="--", lw=1.6, label="32.0 GB RTX 5090 HBM Ceiling")
    for b, v in zip(b3, hbm_gb, strict=True):
        ax2.text(
            b.get_x() + b.get_width() / 2.0,
            v + 0.7,
            f"{v:.2f} GB",
            ha="center",
            fontsize=8.5,
            fontweight="bold",
            color=CORAL,
        )
    for b, v in zip(b4, ddr5_gb, strict=True):
        ax2.text(
            b.get_x() + b.get_width() / 2.0,
            v + 0.7,
            f"{v:.2f} GB",
            ha="center",
            fontsize=8.5,
            fontweight="bold",
            color="#9a6b00",
        )
    ax2.set_xticks(x)
    ax2.set_xticklabels(models, fontsize=8.5)
    ax2.set_ylabel("Memory Footprint (GB)")
    ax2.set_ylim(0, 42)
    ax2.set_title("(b) Zero-OOM Hybrid HBM + DDR5 Residency")
    ax2.legend(loc="upper left", fontsize=8.5)

    out_path = out_dir / "real_models_v100.png"
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()
    return out_path


def _load_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def generate_fig12_general_ops_phase8(out_dir: Path) -> Path:
    """Figure 12: Phase 8 General GPU Operators vs. PyTorch Eager on RTX 5090 (Okabe-Ito)."""
    rows = _load_csv_rows(Path("benchmarks/results_5090/general_ops.csv"))
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16.5, 7.2), constrained_layout=True)

    # Panel (a): Compute & Scan Ops (BMM + Cumsum across shapes)
    comp_rows = [r for r in rows if r["op"] in ("bmm", "cumsum")]
    labels_a = [f"{r['op']}\n({r['shape']})" for r in comp_rows]
    speedups_a = [float(r["speedup"]) for r in comp_rows]
    lats_a = [float(r["cutile_p50_ms"]) for r in comp_rows]
    colors_a = [OI_BLUE if r["op"] == "bmm" else OI_GREEN for r in comp_rows]

    y_a = np.arange(len(comp_rows))
    bars_a = ax1.barh(y_a, speedups_a, height=0.58, color=colors_a)
    ax1.axvline(1.0, color=OI_ORANGE, linestyle="--", lw=1.6, label="PyTorch Eager Parity (1.00×)")
    for bar, sp, ms in zip(bars_a, speedups_a, lats_a, strict=True):
        ax1.text(
            sp + 0.04,
            bar.get_y() + bar.get_height() / 2.0,
            f"{sp:.2f}× ({ms:.4f} ms)",
            va="center",
            ha="left",
            fontsize=8.5,
            fontweight="bold",
            color="#0f172a",
            bbox={"fc": "white", "ec": "none", "alpha": 0.85, "pad": 1.2},
        )
    ax1.set_yticks(y_a)
    ax1.set_yticklabels(labels_a, fontsize=8.5)
    ax1.invert_yaxis()
    ax1.set_xlabel("Speedup vs. PyTorch Eager (×, higher is better)")
    ax1.set_xlim(0, max(speedups_a) * 1.38)
    ax1.set_title("(a) Batched MatMul (L2-Swizzled) & Parallel Prefix Scan (sm_120)")
    ax1.legend(loc="lower right", fontsize=8.5)

    # Panel (b): Memory-Bound, Elementwise, Indexing & Conv2D Ops
    wanted_b = [
        ("reduce_max", "16384x1024"),
        ("transpose", "16384x1024"),
        ("reduce_max", "4096x4096"),
        ("transpose", "4096x4096"),
        ("reduce_sum", "16384x1024"),
        ("silu", "1024x4096"),
        ("gelu", "1024x4096"),
        ("gather", "1024x4096"),
        ("conv2d_3x3", "16x64x56x56->128"),
        ("conv2d_1x1", "16x64x56x56->128"),
    ]
    lookup = {(r["op"], r["shape"]): r for r in rows}
    sel_b = [lookup[k] for k in wanted_b if k in lookup]
    labels_b = [f"{r['op']} ({r['shape']})" for r in sel_b]
    speedups_b = [float(r["speedup"]) for r in sel_b]
    lats_b = [float(r["cutile_p50_ms"]) for r in sel_b]
    colors_b = [
        OI_GREEN if sp >= 0.95 else (OI_SKY if sp >= 0.70 else OI_YELLOW) for sp in speedups_b
    ]

    y_b = np.arange(len(sel_b))
    bars_b = ax2.barh(y_b, speedups_b, height=0.58, color=colors_b)
    ax2.axvline(
        1.0, color=OI_ORANGE, linestyle="--", lw=1.6, label="PyTorch / cuDNN Parity (1.00×)"
    )
    for bar, sp, ms in zip(bars_b, speedups_b, lats_b, strict=True):
        ax2.text(
            sp + 0.03,
            bar.get_y() + bar.get_height() / 2.0,
            f"{sp:.2f}× ({ms:.4f} ms)",
            va="center",
            ha="left",
            fontsize=8.5,
            fontweight="bold",
            color="#0f172a",
            bbox={"fc": "white", "ec": "none", "alpha": 0.85, "pad": 1.2},
        )
    ax2.set_yticks(y_b)
    ax2.set_yticklabels(labels_b, fontsize=8.5)
    ax2.invert_yaxis()
    ax2.set_xlabel("Speedup vs. PyTorch / cuDNN Eager (×)")
    ax2.set_xlim(0, 1.48)
    ax2.set_title("(b) Reductions, Transpose, Elementwise & Conv2D vs. PyTorch/cuDNN")
    ax2.legend(loc="lower right", fontsize=8.5)

    out_path = out_dir / "fig12_general_ops_phase8.png"
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()
    return out_path


def generate_fig13_training_vision_parity(out_dir: Path) -> Path:
    """Figure 13: Phase 8 Training Convergence Parity & Vision/Optimizer Latency (Okabe-Ito)."""
    conv_rows = _load_csv_rows(Path("benchmarks/results_5090/phase8_mlp_convergence.csv"))
    perf_rows = _load_csv_rows(Path("benchmarks/results_5090/phase8_training_vision.csv"))
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16.5, 6.8), constrained_layout=True)

    # Panel (a): 10-step 2-layer MLP Training Convergence
    steps = [int(r["step"]) for r in conv_rows]
    cu_loss = [float(r["cutile_mse_loss"]) for r in conv_rows]
    pt_loss = [float(r["torch_mse_loss"]) for r in conv_rows]
    diffs = [float(r["abs_diff"]) for r in conv_rows]
    max_diff = max(diffs)
    step10_diff = diffs[-1]

    ax1.plot(
        steps,
        cu_loss,
        "-o",
        color=OI_BLUE,
        lw=2.3,
        markersize=7,
        label="CuTile Autograd + multi-tensor AdamW",
    )
    ax1.plot(
        steps,
        pt_loss,
        "--x",
        color=OI_ORANGE,
        lw=2.0,
        markersize=8,
        label="PyTorch Autograd + torch.optim.AdamW",
    )
    for st, val in [(steps[0], cu_loss[0]), (steps[1], cu_loss[1]), (steps[-1], cu_loss[-1])]:
        dy = 0.35 if st != steps[-1] else 0.30
        ax1.text(
            st,
            val + dy,
            f"{val:.3f}",
            ha="center",
            fontsize=8.5,
            fontweight="bold",
            color=OI_BLUE,
        )
    ax1.annotate(
        f"Exact Step Parity\nmax |Δloss| = {max_diff:.2e}\nStep 10 diff = {step10_diff:.2e}",
        xy=(6, cu_loss[5]),
        xytext=(5.2, max(cu_loss) * 0.62),
        arrowprops={"arrowstyle": "->", "color": OI_GREEN, "lw": 1.5},
        fontsize=8.8,
        fontweight="bold",
        color="#0f172a",
        bbox={"boxstyle": "round,pad=0.35", "fc": "#f8fafc", "ec": OI_GREEN, "lw": 1.2},
    )
    ax1.set_xticks(steps)
    ax1.set_xlabel("Optimizer Step (batch=64, 32->256->16 MLP, lr=0.05)")
    ax1.set_ylabel("Training MSE Loss")
    ax1.set_ylim(0, max(cu_loss) * 1.18)
    ax1.set_title("(a) 10-Step MLP Training Loss Parity (CuTile vs. PyTorch)")
    ax1.legend(loc="upper right", fontsize=8.5)

    # Panel (b): Vision & Optimizer Latency (log y-axis for readability)
    show_ops = [r for r in perf_rows if r["op"] != "mlp_10step_train"]
    op_labels = [r["op"].replace("_", "\n") for r in show_ops]
    ct_ms = [float(r["cutile_p50_ms"]) for r in show_ops]
    pt_ms = [float(r["torch_p50_ms"]) for r in show_ops]
    speedups = [float(r["speedup"]) for r in show_ops]

    x = np.arange(len(show_ops))
    w = 0.36
    b_ct = ax2.bar(x - w / 2, ct_ms, w, color=OI_BLUE, label="CuTile p50 (ms)")
    b_pt = ax2.bar(x + w / 2, pt_ms, w, color=OI_ORANGE, label="PyTorch / cuDNN p50 (ms)")
    ax2.set_yscale("log")
    for bar, v in zip(b_ct, ct_ms, strict=True):
        ax2.text(
            bar.get_x() + bar.get_width() / 2.0,
            v * 1.18,
            f"{v:.3f}",
            ha="center",
            va="bottom",
            fontsize=7.8,
            fontweight="bold",
            color=OI_BLUE,
        )
    for bar, v in zip(b_pt, pt_ms, strict=True):
        ax2.text(
            bar.get_x() + bar.get_width() / 2.0,
            v * 1.18,
            f"{v:.3f}",
            ha="center",
            va="bottom",
            fontsize=7.8,
            fontweight="bold",
            color=OI_ORANGE,
        )
    for xi, c_v, p_v, sp in zip(x, ct_ms, pt_ms, speedups, strict=True):
        top_v = max(c_v, p_v)
        col = OI_GREEN if sp >= 1.0 else OI_GRAY
        ax2.text(
            xi,
            top_v * 2.15,
            f"{sp:.2f}×",
            ha="center",
            va="bottom",
            fontsize=8.2,
            fontweight="bold",
            color=col,
        )

    ax2.set_xticks(x)
    ax2.set_xticklabels(op_labels, fontsize=8.5)
    ax2.set_ylabel("Median Kernel Latency (ms, log scale)")
    ax2.set_ylim(0.005, max(ct_ms) * 6.5)
    ax2.set_title("(b) Vision Ops & Multi-Tensor AdamW Latency (RTX 5090)")
    ax2.legend(loc="upper left", fontsize=8.5)

    out_path = out_dir / "fig13_training_vision_parity.png"
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()
    return out_path


def generate_fig6_tracks_abc_results(out_dir: Path) -> Path:
    """Figure 6: Track A Kernel Optimization, Track B Zero-ATen Runtime, and Track C Matrix."""
    from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec

    track_a_rows = _load_csv_rows(Path("benchmarks/results_5090/track_a_before_after.csv"))
    track_b_rows = _load_csv_rows(Path("benchmarks/results_5090/track_b_native_runtime.csv"))

    fig = plt.figure(figsize=(17.2, 11.8), dpi=300)
    gs = GridSpec(2, 2, figure=fig, height_ratios=[1.15, 1.0], hspace=0.40, wspace=0.28)

    # ------------------------------------------------------------------
    # Panel (a): Track A Before (v2.0.0) vs. After (v2.1.0) on RTX 5090
    # ------------------------------------------------------------------
    ax_a = fig.add_subplot(gs[0, 0])
    target_ops = [
        ("depthwise_conv2d", "depthwise_conv2d\n(16×64×56×56)"),
        ("max_pool2d", "max_pool2d\n(16×64×56×56)"),
        ("avg_pool2d", "avg_pool2d\n(16×64×56×56)"),
        ("patch_embed", "patch_embed\n(16×3×224×224)"),
        ("conv2d_1x1", "conv2d_1x1\n(64→128ch)"),
        ("conv2d_3x3", "conv2d_3x3\n(64→128ch)"),
        ("gather", "gather\n(1024×4096)"),
        ("silu", "silu\n(1024×4096)"),
    ]
    lookup_a = {r["op"]: r for r in track_a_rows}
    labels_a = [lbl for op, lbl in target_ops]
    before_sp = [float(lookup_a[op]["v200_speedup"]) for op, _ in target_ops]
    after_sp = [float(lookup_a[op]["v210_speedup"]) for op, _ in target_ops]

    y = np.arange(len(target_ops))
    h = 0.36
    ax_a.barh(y - h / 2, before_sp, height=h, color=OI_GRAY, label="v2.0.0 Before (Speedup)")
    ax_a.barh(y + h / 2, after_sp, height=h, color=OI_BLUE, label="v2.1.0 After (Speedup)")
    ax_a.axvline(1.0, color=OI_ORANGE, linestyle="--", lw=1.6, label="1.00× PyTorch Parity")

    for yi, b_v, a_v in zip(y, before_sp, after_sp, strict=True):
        x_txt = max(b_v, a_v) + 0.08
        col = OI_GREEN if a_v >= 1.0 else "#0f172a"
        ax_a.text(
            x_txt,
            yi,
            f"{b_v:.2f}× → {a_v:.2f}×",
            va="center",
            ha="left",
            fontsize=8.5,
            fontweight="bold",
            color=col,
        )

    ax_a.set_yticks(y)
    ax_a.set_yticklabels(labels_a, fontsize=8.5)
    ax_a.invert_yaxis()
    ax_a.set_xlim(0, 5.15)
    ax_a.set_xlabel("Relative Speedup vs. PyTorch / cuDNN Eager (×)")
    ax_a.set_title("(a) Track A Kernel Engineering: Before (v2.0.0) vs. After (v2.1.0)")
    ax_a.legend(loc="lower right", fontsize=8.3)

    # ------------------------------------------------------------------
    # Panel (b): Track B Native Zero-ATen Runtime (Mistral-7B-v0.1-GPTQ)
    # ------------------------------------------------------------------
    sub_b = GridSpecFromSubplotSpec(1, 2, subplot_spec=gs[0, 1], wspace=0.40)
    ax_b1 = fig.add_subplot(sub_b[0, 0])
    ax_b2 = fig.add_subplot(sub_b[0, 1])

    row_v20 = next(r for r in track_b_rows if "v2.0.0" in r["runtime"])
    row_v22 = next(r for r in track_b_rows if "v2.2.0" in r["runtime"])

    aten_vals = [
        int(row_v20["aten_compute_calls_per_decode"]),
        int(row_v22["aten_compute_calls_per_decode"]),
    ]
    rt_labels = ["v2.0.0\nInference\nEngine", "v2.2.0\nNative\nRuntime"]
    b1_bars = ax_b1.bar([0, 1], aten_vals, width=0.52, color=[OI_ORANGE, OI_GREEN])
    ax_b1.text(
        0,
        aten_vals[0] + 2.0,
        f"{aten_vals[0]} calls",
        ha="center",
        fontweight="bold",
        fontsize=9,
        color=OI_ORANGE,
    )
    ax_b1.text(
        1,
        3.5,
        "0 calls\n(Zero-ATen\nVerified)",
        ha="center",
        va="bottom",
        fontweight="bold",
        fontsize=8.8,
        color=OI_GREEN,
        bbox={"boxstyle": "round,pad=0.3", "fc": "#ecfdf5", "ec": OI_GREEN, "lw": 1.1},
    )
    _ = b1_bars
    ax_b1.set_xticks([0, 1])
    ax_b1.set_xticklabels(rt_labels, fontsize=8.5)
    ax_b1.set_ylim(0, 82)
    ax_b1.set_ylabel("Forbidden ATen Compute Calls / Decode")
    ax_b1.set_title("(b1) TorchDispatchMode Audit")

    tps_vals = [float(row_v20["decode_tok_s"]), float(row_v22["decode_tok_s"])]
    pf_vals = [float(row_v20["prefill_ms"]), float(row_v22["prefill_ms"])]
    b2_bars = ax_b2.bar([0, 1], tps_vals, width=0.52, color=[OI_GRAY, OI_BLUE])
    ax_b2.text(
        0,
        tps_vals[0] + 2.5,
        f"{tps_vals[0]:.2f} tok/s\n(Graph: {pf_vals[0]:.2f}ms PF)",
        ha="center",
        fontsize=8.0,
        fontweight="bold",
        color="#334155",
    )
    ax_b2.text(
        1,
        tps_vals[1] + 2.5,
        f"{tps_vals[1]:.2f} tok/s\n(Eager: {pf_vals[1]:.2f}ms PF)",
        ha="center",
        fontsize=8.0,
        fontweight="bold",
        color=OI_BLUE,
    )
    _ = b2_bars
    ax_b2.text(
        0.5,
        122.0,
        "Exact 32/32 Greedy Token Parity\n(Eager Native: 0.77× CUDA Graph)",
        ha="center",
        va="top",
        fontsize=7.9,
        fontweight="bold",
        color="#0f172a",
        bbox={"boxstyle": "round,pad=0.32", "fc": "#f8fafc", "ec": OI_SKY, "lw": 1.0},
    )
    ax_b2.set_xticks([0, 1])
    ax_b2.set_xticklabels(rt_labels, fontsize=8.5)
    ax_b2.set_ylim(0, 128)
    ax_b2.set_ylabel("Mistral-7B-GPTQ Decode (tok/s)")
    ax_b2.set_title("(b2) Mistral-7B Throughput & Parity")

    # ------------------------------------------------------------------
    # Panel (c): Track C Compatibility Matrix (Models × Checkpoint Formats)
    # ------------------------------------------------------------------
    ax_c = fig.add_subplot(gs[1, :])
    ax_c.axis("off")
    ax_c.set_title(
        "(c) Track C Unified ModelGraph & Multi-Format Checkpoint Compatibility Matrix (v2.3.0)",
        fontsize=11.5,
        fontweight="bold",
        pad=10,
    )

    col_headers = [
        "Architecture Family",
        "TFG + Exec\nParity",
        "FP16 / BF16\nSafetensors",
        "GPTQ 4-bit\n(g_idx / -1)",
        "AWQ 4-bit\nInterleaved",
        "GGUF\nQ4_K / Q4_0",
        "GGUF\nQ8_0",
        "FP8 e4m3 &\nHybrid 70B",
    ]
    matrix_rows = [
        [
            "Mistral-7B (Sliding-Window)",
            "Parity <1e-2",
            "Exact",
            "Real 7B Verified",
            "Exact Unpack",
            "Unpack+4b",
            "Exact",
            "Hybrid Auto",
        ],
        [
            "Llama-3 / 3.1 (Scaled RoPE)",
            "Parity <1e-2",
            "BF16 Auto",
            "Unpack + Exec",
            "Exact Unpack",
            "Unpack+4b",
            "Exact",
            "70B >85% VRAM",
        ],
        [
            "Qwen2 / 2.5 (QKV Bias)",
            "Parity <1e-2",
            "Exact",
            "Unpack + Exec",
            "Exact Unpack",
            "Unpack+4b",
            "Exact",
            "Hybrid Auto",
        ],
        [
            "Mixtral-8x7B (Top-2 MoE)",
            "Parity <1e-2",
            "Exact",
            "Real MoE Ready",
            "Exact Unpack",
            "Unpack+4b",
            "Exact",
            "Resident/Auto",
        ],
        [
            "DeepSeek / Qwen2-MoE (Shared)",
            "Parity <1e-2",
            "Exact",
            "Unpack + Exec",
            "Exact Unpack",
            "Unpack+4b",
            "Exact",
            "FP8 Block-Scale",
        ],
        [
            "ViT-B/16 / CLIP / SigLIP",
            "Parity <1e-2",
            "Exact",
            "N/A (Dense ViT)",
            "N/A",
            "N/A",
            "N/A",
            "Native PatchEmbed",
        ],
        [
            "LLaVA Multimodal (ViT+LLM)",
            "Parity <1e-2",
            "Exact",
            "LLM Backbone",
            "LLM Backbone",
            "LLM Backbone",
            "Exact",
            "Multi-Stage TFG",
        ],
    ]

    table = ax_c.table(
        cellText=matrix_rows,
        colLabels=col_headers,
        loc="center",
        cellLoc="center",
        bbox=[0.0, 0.0, 1.0, 0.94],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(8.6)
    col_widths = [0.22, 0.11, 0.11, 0.12, 0.11, 0.11, 0.09, 0.13]
    for (r_idx, c_idx), cell in table.get_celld().items():
        cell.set_width(col_widths[c_idx])
        cell.set_edgecolor("#cbd5e1")
        cell.set_linewidth(0.8)
        if r_idx == 0:
            cell.set_facecolor("#1b365d")
            cell.set_text_props(color="white", fontweight="bold", fontsize=8.8)
        else:
            txt = cell.get_text().get_text()
            if c_idx == 0:
                cell.set_facecolor("#f1f5f9")
                cell.set_text_props(color="#0f172a", fontweight="bold", ha="left")
            elif txt == "N/A" or txt.startswith("N/A"):
                cell.set_facecolor("#f8fafc")
                cell.set_text_props(color="#64748b")
            elif "Real" in txt or "FP8" in txt or "70B" in txt or "Verified" in txt:
                cell.set_facecolor("#dbeafe")
                cell.set_text_props(color="#1e3a8a", fontweight="bold")
            else:
                cell.set_facecolor("#dcfce7")
                cell.set_text_props(color="#14532d", fontweight="bold")

    out_path = out_dir / "fig6_tracks_abc_results.png"
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()
    return out_path


def generate_fig9_v300_platform(out_dir: Path) -> Path:
    """Figure 9: CuTileForge v3.0.0 Consolidated Platform Evaluation (300-DPI, Okabe-Ito)."""
    tb_rows = _load_csv_rows(Path("benchmarks/results_5090/track_b_native_runtime.csv"))
    p9_rows = _load_csv_rows(Path("benchmarks/results_5090/phase9_native_runtime.csv"))
    p10_rows = _load_csv_rows(Path("benchmarks/results_5090/phase10_universal_quant.csv"))
    p13_rows = _load_csv_rows(Path("benchmarks/results_5090/phase13_production.csv"))

    fig, axes = plt.subplots(2, 2, figsize=(16.5, 11.2), constrained_layout=True)

    # ------------------------------------------------------------------
    # Panel (a): Native Runtime Evolution (v2.2.0 -> v3.0.0)
    # ------------------------------------------------------------------
    ax = axes[0, 0]
    v22_tps = float(
        next(r["decode_tok_s"] for r in tb_rows if "v2.2.0_NativeModelRuntime" in r["runtime"])
    )
    p9_graph_row = next(
        r for r in p9_rows if r["item"] == "native_full_graph_bucketed_chained_sync32_256tok"
    )
    v23_tps = float(p9_graph_row["before_v230"])
    v24_tps = float(p9_graph_row["after_median"])
    v30_tps = float(
        next(
            r["median_value"]
            for r in p13_rows
            if r["benchmark_category"] == "single_stream_latency_and_throughput"
        )
    )

    stages_a = [
        "v2.2.0\nNative Eager\n(0 ATen)",
        "v2.3.0\nStep 9.0\nEager Baseline",
        "v2.4.0\nFull-Step\nCUDA Graph",
        "v3.0.0\nProduction\nGraph (TTFT 6.1ms)",
    ]
    vals_a = [v22_tps, v23_tps, v24_tps, v30_tps]
    cols_a = [OI_GRAY, OI_SKY, OI_BLUE, OI_GREEN]
    x_a = np.arange(len(stages_a))
    bars_a = ax.bar(x_a, vals_a, width=0.55, color=cols_a)
    for bar, v in zip(bars_a, vals_a, strict=True):
        mult = v / v22_tps
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            v + 4.0,
            f"{v:.2f} tok/s\n({mult:.2f}×)",
            ha="center",
            va="bottom",
            fontsize=8.8,
            fontweight="bold",
            color="#0f172a",
        )
    ax.set_xticks(x_a)
    ax.set_xticklabels(stages_a, fontsize=8.8)
    ax.set_ylabel("Decode Throughput (tok/s)")
    ax.set_ylim(0, 218)
    ax.set_title("(a) Native Runtime Evolution: Mistral-7B-v0.1-GPTQ (sm_120)")

    # ------------------------------------------------------------------
    # Panel (b): Concurrent Production Throughput (Mistral-7B-GPTQ)
    # ------------------------------------------------------------------
    ax = axes[0, 1]
    conc_rows = [r for r in p13_rows if r["benchmark_category"] == "concurrent_throughput"]
    conc_rows.sort(key=lambda r: int(r["concurrency"]))
    concs = [int(r["concurrency"]) for r in conc_rows]
    conc_tps = [float(r["median_value"]) for r in conc_rows]
    x_b = np.arange(len(concs))
    ax.plot(
        x_b,
        conc_tps,
        marker="o",
        markersize=8,
        linewidth=2.4,
        color=OI_BLUE,
        label="CuTileForge Native Batching (32 tok/req)",
    )
    ax.set_yscale("log")
    offsets_b = [(8, 16), (0, 16), (-18, 18), (0, -24)]
    for xi, c_val, tps, (dx, dy) in zip(x_b, concs, conc_tps, offsets_b, strict=True):
        ax.annotate(
            f"C={c_val}: {tps:,.1f} tok/s",
            (xi, tps),
            xytext=(dx, dy),
            textcoords="offset points",
            ha="center",
            va="center",
            fontsize=8.6,
            fontweight="bold",
            color="#0f172a",
            bbox={"boxstyle": "round,pad=0.22", "fc": "white", "ec": OI_SKY, "alpha": 0.94},
        )
    ax.set_xticks(x_b)
    ax.set_xticklabels([f"Concurrency {c}" for c in concs], fontsize=8.8)
    ax.set_ylabel("Aggregate Throughput (tok/s, log scale)")
    ax.set_xlim(-0.4, 3.4)
    ax.set_ylim(95, 6500)
    ax.set_title("(b) Concurrent Production Throughput on RTX 5090 (real-checkpoint)")
    ax.legend(loc="upper left", fontsize=8.6)

    # ------------------------------------------------------------------
    # Panel (c): Universal Quantization Prefill Throughput (M=128)
    # ------------------------------------------------------------------
    ax = axes[1, 0]
    p10_map = {r["format_label"]: r for r in p10_rows}
    wanted_q = [
        ("INT8_gs128", "INT8 (gs=128)"),
        ("FP8_e4m3_block128", "FP8 Block (128×128)"),
        ("AWQ_gs128", "AWQ INT4 (Native)"),
        ("GGUF_Q4_0", "GGUF Q4_0 (18B)"),
        ("GGUF_Q6_K", "GGUF Q6_K (210B)"),
        ("GPTQ_gs128_nodesc", "GPTQ INT4 (gs=128)"),
    ]
    q_labels = [lbl for _, lbl in wanted_q]
    q_tf = [float(p10_map[k]["m128_tflops"]) for k, _ in wanted_q]
    q_ppl = [float(p10_map[k]["wikitext2_ppl_rel_diff_pct"]) for k, _ in wanted_q]
    q_cols = [OI_GREEN, OI_BLUE, OI_SKY, OI_YELLOW, OI_PURPLE, OI_GRAY]

    y_c = np.arange(len(wanted_q))
    bars_c = ax.barh(y_c, q_tf, height=0.58, color=q_cols)
    for bar, tf, ppl in zip(bars_c, q_tf, q_ppl, strict=True):
        ax.text(
            tf + 1.8,
            bar.get_y() + bar.get_height() / 2.0,
            f"{tf:.2f} TF  (ΔPPL {ppl:.3f}%)",
            va="center",
            ha="left",
            fontsize=8.6,
            fontweight="bold",
            color="#0f172a",
        )
    ax.set_yticks(y_c)
    ax.set_yticklabels(q_labels, fontsize=8.8)
    ax.invert_yaxis()
    ax.set_xlabel("Prefill Throughput @ M=128, 4096×4096 (TFLOPS)")
    ax.set_xlim(0, 145)
    ax.set_title("(c) Universal Quantization Prefill Throughput & WikiText-2 Parity")

    # ------------------------------------------------------------------
    # Panel (d): Long-Context 32-Layer FlashDecoding Scaling (8K–32K)
    # ------------------------------------------------------------------
    ax = axes[1, 1]
    lc_rows = [r for r in p13_rows if r["benchmark_category"] == "long_context_32layer_attention"]
    ctxs = [8192, 16384, 32768]
    fp16_ms = [
        float(
            next(
                r["median_value"]
                for r in lc_rows
                if int(r["context_len"]) == c and r["kv_dtype"] == "fp16"
            )
        )
        for c in ctxs
    ]
    fp8_ms = [
        float(
            next(
                r["median_value"]
                for r in lc_rows
                if int(r["context_len"]) == c and r["kv_dtype"] == "fp8_e4m3fn"
            )
        )
        for c in ctxs
    ]
    x_d = np.arange(len(ctxs))
    ax.plot(
        x_d,
        fp16_ms,
        marker="o",
        markersize=8,
        linewidth=2.4,
        color=OI_BLUE,
        label="FP16 KV Cache (Head-Major FlashDecoding)",
    )
    ax.plot(
        x_d,
        fp8_ms,
        marker="s",
        markersize=8,
        linewidth=2.4,
        linestyle="--",
        color=OI_ORANGE,
        label="FP8 e4m3fn KV Cache (50% KV VRAM)",
    )
    for xi, v16, v8 in zip(x_d, fp16_ms, fp8_ms, strict=True):
        fp8_offset = (-18, 12) if xi == 1 else (0, 12)
        ax.annotate(
            f"{v8:.2f} ms",
            (xi, v8),
            xytext=fp8_offset,
            textcoords="offset points",
            ha="center",
            fontsize=8.8,
            fontweight="bold",
            color=OI_ORANGE,
        )
        ax.annotate(
            f"{v16:.2f} ms",
            (xi, v16),
            xytext=(0, -17),
            textcoords="offset points",
            ha="center",
            fontsize=8.8,
            fontweight="bold",
            color=OI_BLUE,
        )
    ax.set_xticks(x_d)
    ax.set_xticklabels(["8K (8,192)", "16K (16,384)", "32K (32,768)"], fontsize=8.8)
    ax.set_ylabel("32-Layer Decode Step Latency (ms/step)")
    ax.set_xlim(-0.3, 2.3)
    ax.set_ylim(0, 9.2)
    ax.set_title("(d) Long-Context 32-Layer Attention Scaling (synthetic-weights)")
    ax.legend(loc="upper left", fontsize=8.5)

    out_path = out_dir / "fig9_v300_platform.png"
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()
    return out_path


def main() -> None:
    _apply_pub_style()
    out_dir = Path("docs/images_5090")
    out_dir.mkdir(parents=True, exist_ok=True)
    for fn in (
        generate_fig1_executive_overview,
        generate_fig2_roofline,
        generate_fig3_microbenchmarks,
        generate_fig4_systems_telemetry,
        generate_real_models_v100,
        generate_fig6_tracks_abc_results,
        generate_fig9_v300_platform,
    ):
        p = fn(out_dir)
        print(f"Generated {p} ({p.stat().st_size // 1024} KB)")

    fig_dir = Path("docs/figures")
    fig_dir.mkdir(parents=True, exist_ok=True)
    for fn in (
        generate_fig12_general_ops_phase8,
        generate_fig13_training_vision_parity,
    ):
        p = fn(fig_dir)
        print(f"Generated {p} ({p.stat().st_size // 1024} KB)")


if __name__ == "__main__":
    main()
