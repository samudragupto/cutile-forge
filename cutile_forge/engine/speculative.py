"""Speculative decoding (gamma=4) with N-gram prompt lookup and self-draft logit caching."""

from __future__ import annotations

import time
from dataclasses import dataclass

import torch

from cutile_forge.engine.kv_cache import PagedKVCache
from cutile_forge.engine.model import Mistral7B


@dataclass
class SpeculativeStats:
    """Telemetry for a speculative decoding run."""

    tokens_generated: int
    draft_steps: int
    accepted_tokens: int
    acceptance_rate: float
    elapsed_ms: float
    tokens_per_sec: float


class NGramPromptLookupDrafter:
    """Zero-VRAM N-gram prompt-lookup drafter (3-gram / 2-gram / 1-gram fallback)."""

    def __init__(self, max_n: int = 3) -> None:
        self.max_n = max(1, max_n)

    def draft(self, history: list[int], gamma: int) -> list[int]:
        """Draft up to `gamma` tokens by matching the longest suffix in `history`."""
        n_hist = len(history)
        if n_hist < 2:
            return []
        for n in range(min(self.max_n, n_hist - 1), 0, -1):
            suffix = history[-n:]
            # Search backwards for the most recent match before the final suffix
            for idx in range(n_hist - n - 1, -1, -1):
                if history[idx : idx + n] == suffix:
                    start = idx + n
                    end = min(start + gamma, n_hist)
                    cand = history[start:end]
                    if cand:
                        if len(cand) < gamma and len(cand) > 0:
                            # Pad remaining slots cyclically from matched continuation
                            while len(cand) < gamma:
                                cand.append(cand[len(cand) % max(1, end - start)])
                        return cand[:gamma]
        return []


class SpeculativeDecoder:
    """Speculative decoder pairing N-gram prompt lookup + cached self-draft with batched verification."""

    def __init__(
        self,
        target_model: Mistral7B,
        draft_model: Mistral7B | None = None,
        gamma: int = 4,
        max_ngram: int = 3,
    ) -> None:
        self.target = target_model
        self.draft = draft_model or target_model
        self.gamma = max(1, gamma)
        self.ngram_drafter = NGramPromptLookupDrafter(max_n=max_ngram)
        self._logit_cache: dict[tuple[int, ...], list[int]] = {}

    @torch.inference_mode()
    def generate(
        self,
        prompt_ids: torch.Tensor,
        max_new_tokens: int = 64,
    ) -> tuple[list[int], SpeculativeStats]:
        """Run greedy speculative decoding with `gamma` draft tokens verified in parallel."""
        if prompt_ids.ndim == 1:
            prompt_ids = prompt_ids.unsqueeze(0)
        prompt_ids = prompt_ids.to(self.target.device)

        kv = PagedKVCache(
            num_layers=self.target.config.num_hidden_layers,
            num_kv_heads=self.target.config.num_key_value_heads,
            head_dim=self.target.config.head_dim,
            device=self.target.device,
        )

        t0 = time.perf_counter()
        logits = self.target.forward(prompt_ids, kv_cache=kv, pos_offset=0, last_token_only=True)
        cur_tok = int(logits[:, -1, :].argmax(dim=-1).item())
        history: list[int] = prompt_ids[0].tolist() + [cur_tok]
        generated: list[int] = [cur_tok]
        pos = int(prompt_ids.shape[1])

        draft_steps = 0
        accepted_total = 0
        total_drafted = 0

        while len(generated) < max_new_tokens:
            draft_steps += 1
            draft_toks = self.ngram_drafter.draft(history, self.gamma)

            if not draft_toks:
                cache_key = tuple(history[-3:])
                if cache_key in self._logit_cache:
                    draft_toks = list(self._logit_cache[cache_key][: self.gamma])
                else:
                    # Lightweight 1-step fallback draft when no N-gram or cache hit exists
                    d_logits = self.draft.forward(
                        torch.tensor([[cur_tok]], device=self.target.device, dtype=torch.long),
                        kv_cache=None,
                        pos_offset=pos,
                    )
                    nxt = int(d_logits[:, -1, :].argmax(dim=-1).item())
                    draft_toks = [nxt]

            k_draft = len(draft_toks)
            total_drafted += k_draft
            verify_in = torch.tensor(
                [[cur_tok] + draft_toks], device=self.target.device, dtype=torch.long
            )
            t_logits = self.target.forward(verify_in, kv_cache=kv, pos_offset=pos)
            target_preds = t_logits[0].argmax(dim=-1).tolist()

            n_accept = 0
            for j, d_tok in enumerate(draft_toks):
                if target_preds[j] == d_tok:
                    n_accept += 1
                    generated.append(d_tok)
                    history.append(d_tok)
                    if len(generated) >= max_new_tokens:
                        break
                else:
                    tok = int(target_preds[j])
                    generated.append(tok)
                    history.append(tok)
                    break
            else:
                if len(generated) < max_new_tokens:
                    tok = int(target_preds[k_draft])
                    generated.append(tok)
                    history.append(tok)

            # Cache verified continuation for future suffix hits
            if len(history) >= 4:
                self._logit_cache[
                    tuple(history[-min(len(history), 3 + self.gamma) : -self.gamma])
                ] = history[-self.gamma :]

            accepted_total += n_accept
            pos += n_accept + 1
            if n_accept < k_draft:
                kv.truncate(pos)
            cur_tok = generated[-1]

        if self.target.device.type == "cuda":
            torch.cuda.synchronize()
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        total_drafted = max(1, total_drafted)
        stats = SpeculativeStats(
            tokens_generated=len(generated),
            draft_steps=draft_steps,
            accepted_tokens=accepted_total,
            acceptance_rate=accepted_total / float(total_drafted),
            elapsed_ms=elapsed_ms,
            tokens_per_sec=(len(generated) * 1000.0 / elapsed_ms) if elapsed_ms > 0 else 0.0,
        )
        return generated[:max_new_tokens], stats
