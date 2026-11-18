"""End-to-end decode benchmark: 256-token autoregressive generation."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, List

import torch

from cutile_forge.bench.harness import write_csv
from cutile_forge.bench.pcie_monitor import measure_pcie_h2d_bandwidth
from cutile_forge.engine.config import ModelConfig
from cutile_forge.engine.inference import InferenceEngine
from cutile_forge.engine.model import Mistral7B


def run_decode_benchmark(
    model_dir: Path,
    prompt_text: str = "The capital of France is",
    max_tokens: int = 256,
) -> List[Dict[str, Any]]:
    """Run 256-token decode and record tok/s, ms/tok, VRAM, and PCIe metrics."""
    if (model_dir / "config.json").exists():
        from transformers import AutoTokenizer

        model = Mistral7B.from_pretrained(model_dir, device="cuda")
        tok = AutoTokenizer.from_pretrained(str(model_dir))
        prompt_ids = tok.encode(prompt_text, return_tensors="pt")
    else:
        model = Mistral7B.from_random(ModelConfig(num_hidden_layers=8), device="cuda")
        prompt_ids = torch.tensor([[1, 450, 5565, 302, 4843, 349]], dtype=torch.long)

    engine = InferenceEngine(model)
    # Warmup short generation
    _ = engine.generate_ids(prompt_ids, max_new_tokens=4, temperature=0.0)
    # Actual timed generation
    out_ids = engine.generate_ids(prompt_ids, max_new_tokens=max_tokens, temperature=0.0)
    stats = engine.last_stats
    assert stats is not None

    pcie = measure_pcie_h2d_bandwidth(nbytes=32 * 1024 * 1024, iters=5)
    row: Dict[str, Any] = {
        "engine": "CuTileForge",
        "prompt_tokens": stats.prompt_tokens,
        "generated_tokens": len(out_ids),
        "prefill_ms": round(stats.prefill_ms, 2),
        "decode_ms": round(stats.decode_ms, 2),
        "tokens_per_sec": round(stats.tokens_per_sec, 2),
        "ms_per_token": round(stats.ms_per_token, 2),
        "peak_vram_gb": round(stats.peak_vram_gb, 3),
        "pcie_gb_s": pcie.measured_gb_s,
        "pcie_util_pct": pcie.utilization_pct,
    }
    print(
        f"CuTileForge Decode | {row['tokens_per_sec']:.2f} tok/s | "
        f"{row['ms_per_token']:.2f} ms/tok | VRAM={row['peak_vram_gb']:.2f} GB"
    )
    return [row]


def main() -> None:
    parser = argparse.ArgumentParser(description="E2E decode benchmark (256 tokens)")
    parser.add_argument("--model", type=Path, default=Path(r"D:\models\mistral_7b_4bit"))
    parser.add_argument("--prompt", default="The capital of France is")
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--output", type=Path, default=Path("benchmarks/results/decode.csv"))
    args = parser.parse_args()

    rows = run_decode_benchmark(args.model, prompt_text=args.prompt, max_tokens=args.max_tokens)
    write_csv(args.output, rows)
    print(f"Saved decode benchmark to {args.output}")


if __name__ == "__main__":
    main()
