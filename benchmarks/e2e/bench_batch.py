"""End-to-end batched throughput benchmark across batch sizes 1, 2, 4, 8, 16."""

from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Any, Dict, List

import torch

from cutile_forge.bench.harness import write_csv
from cutile_forge.engine.config import ModelConfig
from cutile_forge.engine.model import Mistral7B
from cutile_forge.utils.memory import get_peak_vram_gb, reset_peak_vram


def run_batch_benchmark(
    model_dir: Path,
    batches: List[int] | None = None,
    seq_len: int = 32,
) -> List[Dict[str, Any]]:
    """Measure aggregate throughput (tokens/sec) across batch sizes 1..16."""
    if (model_dir / "config.json").exists():
        model = Mistral7B.from_pretrained(model_dir, device="cuda")
    else:
        model = Mistral7B.from_random(ModelConfig(num_hidden_layers=8), device="cuda")

    batch_sizes = batches or [1, 2, 4, 8, 16]
    rows: List[Dict[str, Any]] = []

    for bsz in batch_sizes:
        inp = torch.randint(
            1, model.config.vocab_size, (bsz, seq_len), device="cuda", dtype=torch.long
        )
        with torch.inference_mode():
            _ = model.forward(inp, last_token_only=True)
            torch.cuda.synchronize()
            reset_peak_vram()
            t0 = time.perf_counter()
            iters = 5
            for _ in range(iters):
                _ = model.forward(inp, last_token_only=True)
            torch.cuda.synchronize()
            ms = ((time.perf_counter() - t0) * 1000.0) / iters

        tps = (bsz * seq_len * 1000.0) / max(ms, 1e-6)
        vram = get_peak_vram_gb()
        row: Dict[str, Any] = {
            "batch_size": bsz,
            "seq_len": seq_len,
            "latency_ms": round(ms, 2),
            "tokens_per_sec": round(tps, 2),
            "peak_vram_gb": round(vram, 3),
        }
        rows.append(row)
        print(
            f"batch={bsz:2d} | latency={row['latency_ms']:7.2f} ms | "
            f"{row['tokens_per_sec']:7.1f} tok/s | VRAM={vram:.2f} GB"
        )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="E2E batch size benchmark (1/2/4/8/16)")
    parser.add_argument("--model", type=Path, default=Path(r"D:\models\mistral_7b_4bit"))
    parser.add_argument("--output", type=Path, default=Path("benchmarks/results/batch.csv"))
    args = parser.parse_args()

    rows = run_batch_benchmark(args.model)
    write_csv(args.output, rows)
    print(f"Saved batch benchmark to {args.output}")


if __name__ == "__main__":
    main()
