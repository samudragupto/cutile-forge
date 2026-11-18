"""End-to-end long-context benchmark: 4K / 8K / 16K / 32K KV-cache paging & eviction."""

from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Any

import torch

from cutile_forge.bench.harness import write_csv
from cutile_forge.engine.config import ModelConfig
from cutile_forge.engine.kv_cache import PagedKVCache
from cutile_forge.engine.model import Mistral7B
from cutile_forge.utils.memory import get_peak_vram_gb, reset_peak_vram


def run_long_context_benchmark(
    model_dir: Path,
    contexts: list[int] | None = None,
) -> list[dict[str, Any]]:
    """Measure decode latency and KV-cache eviction across 4K/8K/16K/32K contexts."""
    if (model_dir / "config.json").exists():
        model = Mistral7B.from_pretrained(model_dir, device="cuda")
    else:
        model = Mistral7B.from_random(ModelConfig(num_hidden_layers=8), device="cuda")

    ctx_lens = contexts or [4096, 8192, 16384, 32768]
    rows: list[dict[str, Any]] = []

    for ctx_len in ctx_lens:
        # 1.5 GB GPU KV budget triggers CPU DDR5 paging for long contexts on 8 GB VRAM
        kv = PagedKVCache(
            num_layers=model.config.num_hidden_layers,
            num_kv_heads=model.config.num_key_value_heads,
            head_dim=model.config.head_dim,
            page_size=64,
            gpu_budget_bytes=int(1.5 * 1024**3),
            device="cuda",
        )
        reset_peak_vram()
        # Populate KV cache efficiently across layers in 1024-token chunks
        chunk = 1024
        for layer_idx in range(model.config.num_hidden_layers):
            rem = ctx_len - 1
            while rem > 0:
                cur = min(chunk, rem)
                k_buf = torch.randn(
                    (1, model.config.num_key_value_heads, cur, model.config.head_dim),
                    device="cuda",
                    dtype=torch.float16,
                )
                v_buf = torch.randn_like(k_buf)
                kv.update(layer_idx, k_buf, v_buf)
                rem -= cur

        tok = torch.tensor([[42]], device="cuda", dtype=torch.long)
        with torch.inference_mode():
            t0 = time.perf_counter()
            _ = model.forward(tok, kv_cache=kv, pos_offset=ctx_len - 1, last_token_only=True)
            torch.cuda.synchronize()
            step_ms = (time.perf_counter() - t0) * 1000.0

        vram = get_peak_vram_gb()
        row: dict[str, Any] = {
            "context_len": ctx_len,
            "decode_step_ms": round(step_ms, 2),
            "kv_evictions": kv.evictions,
            "kv_host_fetches": kv.host_fetches,
            "peak_vram_gb": round(vram, 3),
        }
        rows.append(row)
        print(
            f"ctx={ctx_len:5d} | step={row['decode_step_ms']:6.2f} ms | "
            f"evictions={kv.evictions} | VRAM={vram:.2f} GB"
        )
        kv.reset()
        torch.cuda.empty_cache()

    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="E2E long-context KV-cache paging benchmark")
    parser.add_argument("--model", type=Path, default=Path(r"D:\models\mistral_7b_4bit"))
    parser.add_argument("--output", type=Path, default=Path("benchmarks/results/long_context.csv"))
    args = parser.parse_args()

    rows = run_long_context_benchmark(args.model)
    write_csv(args.output, rows)
    print(f"Saved long-context benchmark to {args.output}")


if __name__ == "__main__":
    main()
