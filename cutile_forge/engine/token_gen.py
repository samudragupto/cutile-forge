"""Token generation wrapper with HuggingFace tokenizer, stop tokens, and streaming."""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from pathlib import Path

import torch

from cutile_forge.engine.inference import GenerationStats, InferenceEngine


class TokenGenerator:
    """Text-level token generator wrapping `InferenceEngine` and a tokenizer."""

    def __init__(
        self,
        engine: InferenceEngine,
        tokenizer_path: str | Path | None = None,
    ) -> None:
        self.engine = engine
        self.tokenizer = None
        if tokenizer_path is not None:
            from transformers import AutoTokenizer

            self.tokenizer = AutoTokenizer.from_pretrained(str(tokenizer_path))

    def encode(self, prompt: str) -> torch.Tensor:
        if self.tokenizer is None:
            raise RuntimeError("No tokenizer loaded; pass tokenizer_path or use generate_ids")
        ids = self.tokenizer.encode(prompt, return_tensors="pt")
        return ids

    def decode(self, token_ids: Sequence[int]) -> str:
        if self.tokenizer is None:
            raise RuntimeError("No tokenizer loaded")
        return str(self.tokenizer.decode(list(token_ids), skip_special_tokens=True))

    def generate(
        self,
        prompt: str,
        max_tokens: int = 64,
        temperature: float = 0.0,
        top_k: int = 50,
        top_p: float = 0.9,
        stop_tokens: Sequence[int] | None = None,
    ) -> str:
        """Generate completion text for `prompt`."""
        prompt_ids = self.encode(prompt)
        eos_id = (
            stop_tokens[0]
            if stop_tokens
            else getattr(self.tokenizer, "eos_token_id", self.engine.config.eos_token_id)
        )
        out_ids = self.engine.generate_ids(
            prompt_ids,
            max_new_tokens=max_tokens,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            eos_token_id=eos_id,
        )
        if stop_tokens:
            trimmed: list[int] = []
            stop_set = set(stop_tokens)
            for tid in out_ids:
                if tid in stop_set:
                    break
                trimmed.append(tid)
            out_ids = trimmed
        return self.decode(out_ids)

    def stream(
        self,
        prompt: str,
        max_tokens: int = 64,
        temperature: float = 0.0,
        top_k: int = 50,
        top_p: float = 0.9,
    ) -> Iterator[str]:
        """Yield incremental decoded text tokens."""
        prompt_ids = self.encode(prompt)
        eos_id = getattr(self.tokenizer, "eos_token_id", self.engine.config.eos_token_id)
        for tid in self.engine.stream_ids(
            prompt_ids,
            max_new_tokens=max_tokens,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            eos_token_id=eos_id,
        ):
            yield self.decode([tid])

    @property
    def last_stats(self) -> GenerationStats | None:
        return self.engine.last_stats
