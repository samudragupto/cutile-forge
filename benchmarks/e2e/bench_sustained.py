"""30-minute sustained thermal, power, clock, and throughput benchmark."""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import torch

from cutile_forge.bench.thermal_logger import ThermalLogger
from cutile_forge.engine.config import ModelConfig
from cutile_forge.engine.inference import InferenceEngine
from cutile_forge.engine.model import Mistral7B


def run_sustained_benchmark(
    model_dir: Path,
    duration_minutes: float = 30.0,
    log_interval_s: float = 5.0,
    output_dir: Path = Path("benchmarks/results/sustained"),
) -> Path:
    """Continuously run Mistral-7B decode for `duration_minutes`, logging thermal.csv."""
    output_dir.mkdir(parents=True, exist_ok=True)
    if (model_dir / "config.json").exists():
        from transformers import AutoTokenizer

        model = Mistral7B.from_pretrained(model_dir, device="cuda")
        tok = AutoTokenizer.from_pretrained(str(model_dir))
        prompt = tok.encode("The capital of France is", return_tensors="pt")
    else:
        model = Mistral7B.from_random(ModelConfig(num_hidden_layers=8), device="cuda")
        prompt = torch.tensor([[1, 450, 5565, 302, 4843, 349]], dtype=torch.long)

    engine = InferenceEngine(model)
    logger = ThermalLogger(output_dir / "thermal.csv")

    duration_s = duration_minutes * 60.0
    t_start = time.perf_counter()
    next_log = 0.0
    total_tokens = 0

    print(
        f"Starting sustained run for {duration_minutes:.2f} min "
        f"(log_interval={log_interval_s:.1f}s) -> {output_dir / 'thermal.csv'}"
    )
    while True:
        elapsed = time.perf_counter() - t_start
        if elapsed >= duration_s and len(logger.rows) > 0:
            break

        out_ids = engine.generate_ids(prompt, max_new_tokens=64, temperature=0.0)
        total_tokens += len(out_ids)
        tps = engine.last_stats.tokens_per_sec if engine.last_stats else 0.0

        now_elapsed = time.perf_counter() - t_start
        if now_elapsed >= next_log:
            row = logger.record(now_elapsed, tps)
            print(
                f"  t={row['timestamp_s']:6.1f}s | temp={row['temp_c']:4.1f}C | "
                f"power={row['power_w']:5.1f}W | clk={row['sm_clock_mhz']} MHz | "
                f"tok/s={row['tok_s']:.2f}"
            )
            while next_log <= now_elapsed:
                next_log += log_interval_s

        if now_elapsed >= duration_s:
            break

    csv_path = logger.save()
    print(
        f"Sustained run complete: {total_tokens} tokens, {len(logger.rows)} samples -> {csv_path}"
    )
    return csv_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Sustained thermal/power/clock benchmark")
    parser.add_argument("--model", type=Path, default=Path(r"D:\models\mistral_7b_4bit"))
    parser.add_argument("--duration-minutes", type=float, default=30.0)
    parser.add_argument("--log-interval", type=float, default=5.0)
    parser.add_argument("--output", type=Path, default=Path("benchmarks/results/sustained"))
    args = parser.parse_args()

    run_sustained_benchmark(
        model_dir=args.model,
        duration_minutes=args.duration_minutes,
        log_interval_s=args.log_interval,
        output_dir=args.output,
    )


if __name__ == "__main__":
    main()
