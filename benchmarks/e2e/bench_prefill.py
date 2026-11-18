"""End-to-end prefill benchmark: 128 / 512 / 2048 / 4096 token prompts."""

from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Any

import torch

from cutile_forge.bench.harness import write_csv
from cutile_forge.engine.config import ModelConfig
from cutile_forge.engine.model import Mistral7B
from cutile_forge.utils.memory import get_peak_vram_gb, reset_peak_vram


def _load_model(model_dir: Path) -> Mistral7B:
    if (model_dir / "config.json").exists():
        return Mistral7B.from_pretrained(model_dir, device="cuda")
    return Mistral7B.from_random(ModelConfig(num_hidden_layers=8), device="cuda")


def run_prefill_benchmark(
    model_dir: Path,
    prompt_lens: list[int] | None = None,
) -> list[dict[str, Any]]:
    """Measure prefill latency, tokens/s, and peak VRAM across prompt lengths."""
    model = _load_model(model_dir)
    lens = prompt_lens or [128, 512, 2048, 4096]
    rows: list[dict[str, Any]] = []

    for seq_len in lens:
        prompt = torch.randint(
            1, model.config.vocab_size, (1, seq_len), device="cuda", dtype=torch.long
        )
        with torch.inference_mode():
            # Warmup once on smaller slice
            _ = model.forward(prompt[:, : min(32, seq_len)], last_token_only=True)
            torch.cuda.synchronize()
            reset_peak_vram()
            t0 = time.perf_counter()
            _ = model.forward(prompt, last_token_only=True)
            torch.cuda.synchronize()
            ms = (time.perf_counter() - t0) * 1000.0

        tps = (seq_len * 1000.0) / max(ms, 1e-6)
        vram = get_peak_vram_gb()
        row: dict[str, Any] = {
            "prompt_tokens": seq_len,
            "prefill_ms": round(ms, 2),
            "tokens_per_sec": round(tps, 2),
            "peak_vram_gb": round(vram, 3),
        }
        rows.append(row)
        print(
            f"prompt={seq_len:4d} | prefill={row['prefill_ms']:7.2f} ms | "
            f"{row['tokens_per_sec']:8.1f} tok/s | VRAM={row['peak_vram_gb']:.2f} GB"
        )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="E2E prefill benchmark")
    parser.add_argument("--model", type=Path, default=Path(r"D:\models\mistral_7b_4bit"))
    parser.add_argument("--output", type=Path, default=Path("benchmarks/results/prefill.csv"))
    args = parser.parse_args()

    rows = run_prefill_benchmark(args.model)
    write_csv(args.output, rows)
    print(f"Saved prefill benchmark to {args.output}")


if __name__ == "__main__":
    main()
