"""Generate text with CuTileForge Mistral-7B 4-bit GPTQ."""

from __future__ import annotations

import argparse
import os
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Run CuTileForge LLM text generation")
    parser.add_argument("--model", type=Path, default=Path(r"D:\models\mistral_7b_4bit"))
    parser.add_argument("--prompt", default="The capital of France is")
    parser.add_argument("--max-tokens", type=int, default=32)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--target", default="sm_120")
    args = parser.parse_args()

    os.environ["CUTILE_ARCH"] = args.target

    from cutile_forge.engine import InferenceEngine, Mistral7B, ModelConfig, TokenGenerator

    if (args.model / "config.json").exists():
        model = Mistral7B.from_pretrained(args.model, device="cuda")
        tok_path: Path | None = args.model
    else:
        print(f"Model not found at {args.model}; running synthetic 4-bit demo model.")
        model = Mistral7B.from_random(ModelConfig(num_hidden_layers=4), device="cuda")
        tok_path = None

    engine = InferenceEngine(model, target=args.target)
    if tok_path is not None:
        gen = TokenGenerator(engine, tokenizer_path=tok_path)
        text = gen.generate(
            args.prompt,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p,
        )
        print(f"Prompt: {args.prompt}")
        print(f"Output: {text}")
        if gen.last_stats:
            s = gen.last_stats
            print(
                f"Stats: {s.tokens_per_sec:.2f} tok/s ({s.ms_per_token:.2f} ms/tok), "
                f"prefill={s.prefill_ms:.1f} ms, peak VRAM={s.peak_vram_gb:.2f} GB"
            )
    else:
        import torch

        ids = engine.generate_ids(
            torch.tensor([[1, 42, 77, 99]]),
            max_new_tokens=args.max_tokens,
            temperature=args.temperature,
        )
        print("Generated token IDs:", ids)


if __name__ == "__main__":
    main()
