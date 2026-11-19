"""Generate 300-DPI publication-grade multi-panel figures for CuTileForge v1.0.0."""

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
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

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
    bars = ax.bar(x, vals, width=0.6, color=colors, edgecolor="none")
    for bar, v, m in zip(bars, vals, mults, strict=True):
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            v + 2.5,
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
    ax.set_ylim(0, 142)
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
    w = 0.36
    b1 = ax.bar(x - w / 2, hbm_gb, w, label="GPU HBM VRAM (GB)", color=NAVY)
    b2 = ax.bar(x + w / 2, ddr5_gb, w, label="Pinned Host DDR5 RSS (GB)", color=TEAL)
    ax.axhline(32.0, color=CRIMSON, linestyle="--", linewidth=1.6, label="32.0 GB HBM Ceiling")
    for bar, v in zip(b1, hbm_gb, strict=True):
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            v + 0.6,
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
            v + 0.6,
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
    ax.set_ylim(0, 38)
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
            v * 1.18,
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
            v * 1.18,
            lbl,
            ha="center",
            va="bottom",
            fontsize=8.5,
            fontweight="bold",
            color=NAVY,
        )
    ax.annotate(
        "1,822× faster\n@ 16K ctx",
        xy=(2 - w / 2, 59.20),
        xytext=(1.15, 6000),
        arrowprops={"arrowstyle": "->", "color": TEAL, "lw": 1.6},
        fontsize=9,
        fontweight="bold",
        color=TEAL,
        ha="center",
    )
    ax.set_xticks(x)
    ax.set_xticklabels(stages, fontsize=8.5)
    ax.set_ylabel("Decode Step Latency (ms, log scale)")
    ax.set_ylim(10, 400000)
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
        ax.text(r, v + 12, f"{v:.1f} ms", ha="center", fontsize=8.5, fontweight="bold", color=CORAL)
    for r, v in zip(reqs, batched_ms, strict=True):
        ax.text(r, v - 22, f"{v:.1f} ms", ha="center", fontsize=8.5, fontweight="bold", color=TEAL)
    ax.annotate(
        "3.20× lower\np99 tail latency",
        xy=(4, 113.6),
        xytext=(2.85, 220),
        arrowprops={"arrowstyle": "->", "color": NAVY, "lw": 1.5},
        fontsize=9,
        fontweight="bold",
        color=NAVY,
        ha="center",
    )
    ax.set_xticks(reqs)
    ax.set_xticklabels([f"Req {r}" for r in reqs])
    ax.set_ylabel("End-to-End Request Latency (ms)")
    ax.set_ylim(40, 420)
    ax.set_title("(d) 4-Way Concurrent Serving Latency (max_tokens=8)")
    ax.legend(loc="upper left", fontsize=8.5)

    plt.tight_layout()
    out_path = out_dir / "fig1_executive_overview.png"
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()
    return out_path


def generate_fig2_roofline(out_dir: Path) -> Path:
    """Figure 2: Dual-Architecture Empirical Log-Log Roofline (Ada sm_89 vs Blackwell sm_120)."""
    fig, ax = plt.subplots(figsize=(11, 7))
    ai = np.logspace(np.log10(0.5), 3.0, 500)

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
    ax.scatter([150.3 / 1.792], [150.3], color=TEAL, marker="D", s=45, zorder=6)
    ax.scatter([28.2 / 0.272], [28.2], color=GOLD, marker="D", s=45, zorder=6)

    # Empirical measured points (AI in FLOPs/Byte, TFLOPS, label, offset)
    sm120_pts = [
        (2.0, 3.22, "Decode GEMV (M=1)\n3.22 TF", (8, -16)),
        (37.5, 42.13, "Grouped MoE 8x7B (M=256)\n42.13 TF", (8, -8)),
        (37.5, 4.91, "v0.4.0 Sparse MoE\n4.91 TF", (8, -10)),
        (204.8, 101.62, "GQA FlashAttn-2 (1024)\n101.62 TF", (-115, -22)),
        (410.0, 88.56, "NVFP4 (e2m1) LUT\n88.56 TF", (8, -12)),
        (410.0, 132.58, "INT4 Dequant+GEMM\n132.58 TF", (8, -4)),
        (240.0, 136.14, "FP16 GEMM (256x14336)\n136.14 TF", (-130, 10)),
        (341.0, 150.26, "FP16 GEMM (512x4096)\n150.26 TF", (-95, 16)),
    ]
    sm89_pts = [
        (204.8, 21.76, "Ada GQA Attn\n21.76 TF", (-90, -24)),
        (410.0, 23.37, "Ada INT4 GEMM\n23.37 TF", (8, -18)),
        (240.0, 28.18, "Ada FP16 GEMM\n28.18 TF", (-45, 12)),
    ]

    for x_v, y_v, txt, (dx, dy) in sm120_pts:
        ax.scatter([x_v], [y_v], color=NAVY, marker="o", s=70, zorder=7, edgecolors="white", lw=0.8)
        ax.annotate(
            txt,
            (x_v, y_v),
            xytext=(dx, dy),
            textcoords="offset points",
            fontsize=8,
            fontweight="bold",
            color=NAVY,
        )

    for x_v, y_v, txt, (dx, dy) in sm89_pts:
        ax.scatter(
            [x_v], [y_v], color=CORAL, marker="^", s=70, zorder=7, edgecolors="white", lw=0.8
        )
        ax.annotate(
            txt,
            (x_v, y_v),
            xytext=(dx, dy),
            textcoords="offset points",
            fontsize=8,
            fontweight="bold",
            color=CORAL,
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
        fontsize=8,
        fontweight="bold",
        color=CRIMSON,
        ha="right",
    )

    ax.set_xlabel("Arithmetic Intensity (FLOPs / HBM Byte)")
    ax.set_ylabel("Achieved Throughput (TFLOPS)")
    ax.set_xlim(0.5, 1000)
    ax.set_ylim(0.5, 420)
    ax.set_title("Empirical Log-Log Roofline: Ada Lovelace (sm_89) vs. Blackwell (sm_120)")
    ax.legend(loc="upper left", fontsize=8.5)

    plt.tight_layout()
    out_path = out_dir / "roofline_ada_vs_blackwell.png"
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()
    return out_path


def generate_fig3_microbenchmarks(out_dir: Path) -> Path:
    """Figure 3: 2x2 Micro-Benchmark Suite (GEMM, INT4/NVFP4, FFN Cost Model, GQA Attention)."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

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
                v + 4,
                f"{v:.0f}",
                ha="center",
                fontsize=7.5,
                fontweight="bold",
            )
    ax.set_xticks(x)
    ax.set_xticklabels(shapes)
    ax.set_ylabel("Throughput (TFLOPS)")
    ax.set_ylim(0, 355)
    ax.set_title("(a) Prefill FP16 GEMM Throughput (RTX 5090, sm_120)")
    ax.legend(loc="upper left", fontsize=8)

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
            sp + 0.12,
            f"{sp:.2f}×\n({tf})",
            ha="center",
            fontsize=8.5,
            fontweight="bold",
        )
    ax.set_xticks(x)
    ax.set_xticklabels(dq_shapes, fontsize=8.5)
    ax.set_ylabel("Speedup vs. Separate Dequant + GEMM (×)")
    ax.set_ylim(0, 6.2)
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
        ax.text(
            b.get_x() + b.get_width() / 2.0,
            v + 0.15,
            f"{v:.2f}×",
            ha="center",
            fontsize=8,
            fontweight="bold",
            color=CORAL,
        )
    for b, v in zip(b2, v030, strict=True):
        ax.text(
            b.get_x() + b.get_width() / 2.0,
            v + 0.15,
            f"{v:.2f}×",
            ha="center",
            fontsize=8,
            fontweight="bold",
            color=TEAL,
        )
    ax.set_xticks(x)
    ax.set_xticklabels(seqs)
    ax.set_ylabel("Speedup vs. 3 Separate CUDA Kernels (×)")
    ax.set_ylim(0, 9.8)
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
    for b, v in zip(b_ct, ct_a, strict=True):
        ax.text(
            b.get_x() + b.get_width() / 2.0,
            v + 2.0,
            f"{v:.1f}",
            ha="center",
            fontsize=8,
            fontweight="bold",
            color=NAVY,
        )
    for b, v in zip(b_pt, pt_a, strict=True):
        ax.text(
            b.get_x() + b.get_width() / 2.0,
            v + 2.0,
            f"{v:.1f}",
            ha="center",
            fontsize=8,
            fontweight="bold",
            color=TEAL,
        )
    ax.set_xticks(x)
    ax.set_xticklabels(a_seqs)
    ax.set_ylabel("Throughput (TFLOPS)")
    ax.set_ylim(0, 120)
    ax.set_title("(d) GQA FlashAttention-2 (32:8 Heads) Scaling (sm_120)")
    ax.legend(loc="upper left", fontsize=8.5)

    plt.tight_layout()
    out_path = out_dir / "fig3_microbenchmark_suite.png"
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()
    return out_path


def generate_fig4_systems_telemetry(out_dir: Path) -> Path:
    """Figure 4: 1x2 Systems Telemetry (30-Min Sustained Stability & Stage -1 PCIe DMA Overlap)."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))

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
    ax1.set_ylim(0, 118)
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
            p + 1.8,
            f"{p:.1f}%",
            ha="center",
            fontsize=9.5,
            fontweight="bold",
        )
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels, fontsize=8.5)
    ax2.set_ylabel("Overlap / Bus Utilization (%)")
    ax2.set_ylim(0, 112)
    ax2.set_title("(b) Stage -1 PCIe DMA Overlap & Bus Saturation")

    plt.tight_layout()
    out_path = out_dir / "fig4_systems_telemetry.png"
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
    ):
        p = fn(out_dir)
        print(f"Generated {p} ({p.stat().st_size // 1024} KB)")


if __name__ == "__main__":
    main()
