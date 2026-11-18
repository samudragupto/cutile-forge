"""Generate Markdown report (docs/RESULTS.md) and matplotlib plots from benchmark CSVs."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def _md_table(rows: list[dict[str, str]]) -> str:
    if not rows:
        return "_No data recorded._\n"
    cols = list(rows[0].keys())
    header = "| " + " | ".join(cols) + " |"
    sep = "| " + " | ".join("---" for _ in cols) + " |"
    body = ["| " + " | ".join(str(r.get(c, "")) for c in cols) + " |" for r in rows]
    return "\n".join([header, sep] + body) + "\n"


def generate_plots(input_dir: Path) -> list[Path]:
    """Generate PNG charts from available CSV files in `input_dir`."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plots: list[Path] = []

    gemm_rows = _read_csv(input_dir / "gemm.csv")
    if gemm_rows:
        prefill = [r for r in gemm_rows if int(r["M"]) > 1]
        if prefill:
            fig, ax = plt.subplots(figsize=(8, 4.5))
            shapes = [r["shape"] for r in prefill]
            x = list(range(len(shapes)))
            w = 0.2
            ax.bar(
                [i - 1.5 * w for i in x],
                [float(r["cutile_tflops"]) for r in prefill],
                w,
                label="CuTile",
            )
            ax.bar(
                [i - 0.5 * w for i in x],
                [float(r["cuda_tflops"]) for r in prefill],
                w,
                label="CUDA",
            )
            ax.bar(
                [i + 0.5 * w for i in x],
                [float(r["cublas_tflops"]) for r in prefill],
                w,
                label="cuBLAS",
            )
            ax.bar(
                [i + 1.5 * w for i in x],
                [float(r["pytorch_tflops"]) for r in prefill],
                w,
                label="PyTorch",
            )
            ax.set_xticks(x)
            ax.set_xticklabels(shapes, rotation=15)
            ax.set_ylabel("TFLOPS (median)")
            ax.set_title("Prefill GEMM Throughput (RTX 4060 Laptop, sm_89)")
            ax.legend()
            fig.tight_layout()
            out = input_dir / "gemm_tflops.png"
            fig.savefig(out, dpi=150)
            plt.close(fig)
            plots.append(out)

    layer_rows = _read_csv(input_dir / "layer.csv")
    if layer_rows:
        fig, ax = plt.subplots(figsize=(7, 4))
        seqs = [r["seq_len"] for r in layer_rows]
        x = list(range(len(seqs)))
        w = 0.25
        ax.bar(
            [i - w for i in x],
            [float(r["tfg_ms"]) for r in layer_rows],
            w,
            label="CuTileForge TFG (3 kernels)",
        )
        ax.bar(
            x, [float(r["unfused_8k_ms"]) for r in layer_rows], w, label="CUDA Unfused (8 kernels)"
        )
        ax.bar(
            [i + w for i in x],
            [float(r["pytorch_ms"]) for r in layer_rows],
            w,
            label="PyTorch Eager",
        )
        ax.set_xticks(x)
        ax.set_xticklabels([f"seq={s}" for s in seqs])
        ax.set_ylabel("Latency (ms / layer)")
        ax.set_title("Full Transformer Layer Latency: 3-Kernel TFG vs 8-Kernel CUDA")
        ax.legend()
        fig.tight_layout()
        out = input_dir / "layer_latency.png"
        fig.savefig(out, dpi=150)
        plt.close(fig)
        plots.append(out)

    ffn_rows = _read_csv(input_dir / "ffn.csv")
    if ffn_rows:
        fig, ax = plt.subplots(figsize=(7, 4))
        seqs = [r["seq_len"] for r in ffn_rows]
        ax.bar(seqs, [float(r["speedup"]) for r in ffn_rows], color="#2563eb")
        ax.axhline(1.15, color="#dc2626", linestyle="--", label="1.15x target")
        ax.set_ylabel("Speedup (x)")
        ax.set_xlabel("Sequence Length")
        ax.set_title("Fused FFN Speedup vs 3 Separate Kernels")
        ax.legend()
        fig.tight_layout()
        out = input_dir / "ffn_speedup.png"
        fig.savefig(out, dpi=150)
        plt.close(fig)
        plots.append(out)

    attn_rows = _read_csv(input_dir / "attention.csv")
    if attn_rows:
        fig, ax = plt.subplots(figsize=(7.5, 4.2))
        seqs = [r["seq_len"] for r in attn_rows]
        x = list(range(len(seqs)))
        w = 0.25
        ax.bar(
            [i - w for i in x],
            [float(r["cutile_tflops"]) for r in attn_rows],
            w,
            label="CuTile FlashAttn-2 (GQA 32:8)",
        )
        ax.bar(
            x,
            [float(r["pytorch_tflops"]) for r in attn_rows],
            w,
            label="PyTorch SDPA",
        )
        ax.bar(
            [i + w for i in x],
            [float(r["cuda_tflops"]) for r in attn_rows],
            w,
            label="Hand-written CUDA",
        )
        ax.set_xticks(x)
        ax.set_xticklabels([f"seq={s}" for s in seqs])
        ax.set_ylabel("TFLOPS (median)")
        ax.set_title("Grouped-Query Attention (32:8) Throughput")
        ax.legend()
        fig.tight_layout()
        out = input_dir / "attention_tflops.png"
        fig.savefig(out, dpi=150)
        plt.close(fig)
        plots.append(out)

    thermal_path = input_dir / "sustained" / "thermal.csv"
    if not thermal_path.exists():
        thermal_path = input_dir / "thermal.csv"
    thermal_rows = _read_csv(thermal_path)
    if thermal_rows:
        fig, ax1 = plt.subplots(figsize=(8, 4))
        ts = [float(r["timestamp_s"]) / 60.0 for r in thermal_rows]
        temps = [float(r["temp_c"]) for r in thermal_rows]
        tps = [float(r["tok_s"]) for r in thermal_rows]
        ax1.plot(ts, temps, color="#dc2626", label="GPU Temp (C)")
        ax1.set_xlabel("Elapsed Time (min)")
        ax1.set_ylabel("Temperature (C)", color="#dc2626")
        ax2 = ax1.twinx()
        ax2.plot(ts, tps, color="#2563eb", alpha=0.75, label="Throughput (tok/s)")
        ax2.set_ylabel("Throughput (tok/s)", color="#2563eb")
        ax1.set_title("Sustained Thermal & Throughput Telemetry")
        fig.tight_layout()
        out = input_dir / "thermal_sustained.png"
        fig.savefig(out, dpi=150)
        plt.close(fig)
        plots.append(out)

    return plots


def generate_report(input_dir: str | Path, output_md: str | Path) -> Path:
    """Compile all benchmark CSVs in `input_dir` into a Markdown report at `output_md`."""
    in_dir = Path(input_dir)
    out_path = Path(output_md)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    plots = generate_plots(in_dir)

    sections = [
        ("GEMM (FP16, 8 Shapes)", "gemm.csv"),
        ("GEMM + INT4 Dequant (Fused vs Separate)", "gemm_dequant.csv"),
        ("Fused FFN (1 Kernel vs 3 Kernels)", "ffn.csv"),
        ("Grouped-Query FlashAttention-2 (32:8)", "attention.csv"),
        ("LayerNorm (D=4096)", "ln.csv"),
        ("SiLU Fusion (Fused-in-GEMM vs Separate)", "silu.csv"),
        ("Full Transformer Layer (3-Kernel TFG vs 8-Kernel CUDA)", "layer.csv"),
        ("Stage -1 PCIe Weight Streaming Overlap", "weight_stream.csv"),
        ("End-to-End Prefill", "prefill.csv"),
        ("End-to-End Decode (256 Tokens)", "decode.csv"),
        ("vLLM Baseline (256 Tokens)", "vllm_baseline.csv"),
        ("Long Context (4K-32K KV-Cache Paging)", "long_context.csv"),
        ("Batched Inference (Batch 1-16)", "batch.csv"),
        ("Speculative Decoding (gamma=4)", "speculative.csv"),
    ]

    lines: list[str] = [
        "# CuTileForge Benchmark Results",
        "",
        "**Hardware:** Lenovo IdeaPad 15APH8 | RTX 4060 Laptop GPU (AD107, sm_89, 8 GB GDDR6) | "
        "AMD Ryzen 7 7735HS | 32 GB DDR5 | PCIe 4.0",
        "**Model:** Mistral-7B-v0.1 (4-bit GPTQ) | **Engine:** CuTileForge 0.1.0",
        "",
    ]

    for title, fname in sections:
        rows = _read_csv(in_dir / fname)
        if rows:
            lines.append(f"## {title}")
            lines.append("")
            lines.append(_md_table(rows))

    thermal_path = in_dir / "sustained" / "thermal.csv"
    if not thermal_path.exists():
        thermal_path = in_dir / "thermal.csv"
    t_rows = _read_csv(thermal_path)
    if t_rows:
        max_temp = max(float(r["temp_c"]) for r in t_rows)
        avg_power = sum(float(r["power_w"]) for r in t_rows) / len(t_rows)
        avg_tps = sum(float(r["tok_s"]) for r in t_rows) / len(t_rows)
        avg_throttle = sum(float(r["throttle_pct"]) for r in t_rows) / len(t_rows)
        lines.append("## Sustained Thermal Run Summary")
        lines.append("")
        lines.append("| Metric | Value |")
        lines.append("| --- | --- |")
        lines.append(f"| Samples | {len(t_rows)} |")
        lines.append(f"| Avg tok/s | {avg_tps:.2f} |")
        lines.append(f"| Max GPU temp | {max_temp:.1f} C |")
        lines.append(f"| Avg power | {avg_power:.1f} W |")
        lines.append(f"| Avg clock throttle | {avg_throttle:.2f}% |")
        lines.append("")

    if plots:
        lines.append("## Generated Plots")
        lines.append("")
        for p in plots:
            lines.append(f"- `{p.as_posix()}`")
        lines.append("")

    out_path.write_text("\n".join(lines), encoding="utf-8")
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate Markdown + plots from benchmark CSVs")
    parser.add_argument("--input", type=Path, default=Path("benchmarks/results"))
    parser.add_argument("--output", type=Path, default=Path("docs/RESULTS.md"))
    args = parser.parse_args()
    out = generate_report(args.input, args.output)
    print(f"Generated report at {out}")


if __name__ == "__main__":
    main()
