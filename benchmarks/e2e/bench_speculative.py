"""End-to-end speculative decoding benchmark (gamma=4) vs autoregressive baseline."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, List

import torch

from cutile_forge.bench.harness import write_csv
from cutile_forge.engine.config import ModelConfig
from cutile_forge.engine.inference import InferenceEngine
from cutile_forge.engine.model import Mistral7B
from cutile_forge.engine.speculative import SpeculativeDecoder


def run_speculative_benchmark(
    model_dir: Path,
    gamma: int = 4,
    max_tokens: int = 64,
) -> List[Dict[str, Any]]:
    """Compare speculative decoding (`gamma=4`) against standard decode."""
    if (model_dir / "config.json").exists():
        from transformers import AutoTokenizer

        model = Mistral7B.from_pretrained(model_dir, device="cuda")
        tok = AutoTokenizer.from_pretrained(str(model_dir))
        prompt = tok.encode("The capital of France is", return_tensors="pt")
    else:
        model = Mistral7B.from_random(ModelConfig(num_hidden_layers=8), device="cuda")
        prompt = torch.tensor([[1, 450, 5565, 302, 4843, 349]], dtype=torch.long)

    # Lightweight 4-layer draft proxy sharing first 4 layers of target
    draft = Mistral7B.from_random(
        ModelConfig(
            vocab_size=model.config.vocab_size,
            hidden_size=model.config.hidden_size,
            intermediate_size=model.config.intermediate_size,
            num_hidden_layers=4,
            num_attention_heads=model.config.num_attention_heads,
            num_key_value_heads=model.config.num_key_value_heads,
            head_dim=model.config.head_dim,
            group_size=model.config.group_size,
        ),
        device="cuda",
    )
    draft.weights.embed_tokens = model.weights.embed_tokens
    draft.weights.norm = model.weights.norm
    draft.weights.lm_head = model.weights.lm_head
    draft.weights.layers = model.weights.layers[:4]

    engine = InferenceEngine(model)
    _ = engine.generate_ids(prompt, max_new_tokens=max_tokens, temperature=0.0)
    base_tps = engine.last_stats.tokens_per_sec if engine.last_stats else 0.0

    spec = SpeculativeDecoder(target_model=model, draft_model=draft, gamma=gamma)
    _, st = spec.generate(prompt, max_new_tokens=max_tokens)

    row: Dict[str, Any] = {
        "gamma": gamma,
        "tokens_generated": st.tokens_generated,
        "acceptance_rate": round(st.acceptance_rate, 3),
        "baseline_tok_s": round(base_tps, 2),
        "speculative_tok_s": round(st.tokens_per_sec, 2),
        "speedup": round(st.tokens_per_sec / max(base_tps, 1e-6), 3),
    }
    print(
        f"Speculative (gamma={gamma}) | accept={row['acceptance_rate']:.2f} | "
        f"base={row['baseline_tok_s']:.2f} tok/s | spec={row['speculative_tok_s']:.2f} tok/s"
    )
    return [row]


def main() -> None:
    parser = argparse.ArgumentParser(description="E2E speculative decoding benchmark")
    parser.add_argument("--model", type=Path, default=Path(r"D:\models\mistral_7b_4bit"))
    parser.add_argument("--gamma", type=int, default=4)
    parser.add_argument("--max-tokens", type=int, default=64)
    parser.add_argument("--output", type=Path, default=Path("benchmarks/results/speculative.csv"))
    args = parser.parse_args()

    rows = run_speculative_benchmark(args.model, gamma=args.gamma, max_tokens=args.max_tokens)
    write_csv(args.output, rows)
    print(f"Saved speculative benchmark to {args.output}")


if __name__ == "__main__":
    main()
