"""Speculative decoding (gamma=4) with draft+verify acceptance tracking."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

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


class SpeculativeDecoder:
    """Speculative decoder pairing a lightweight draft pass with full target verification."""

    def __init__(
        self,
        target_model: Mistral7B,
        draft_model: Optional[Mistral7B] = None,
        gamma: int = 4,
    ) -> None:
        self.target = target_model
        self.draft = draft_model or target_model
        self.gamma = max(1, gamma)

    @torch.inference_mode()
    def generate(
        self,
        prompt_ids: torch.Tensor,
        max_new_tokens: int = 64,
    ) -> tuple[List[int], SpeculativeStats]:
        """Run greedy speculative decoding with `gamma` draft tokens verified in parallel."""
        import time

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
        logits = self.target.forward(prompt_ids, kv_cache=kv, pos_offset=0)
        cur_tok = int(logits[:, -1, :].argmax(dim=-1).item())
        generated: List[int] = [cur_tok]
        pos = int(prompt_ids.shape[1])

        draft_steps = 0
        accepted_total = 0

        while len(generated) < max_new_tokens:
            draft_steps += 1
            # Draft `gamma` tokens using first few layers / draft model
            draft_toks: List[int] = []
            ctx = torch.tensor([[cur_tok]], device=self.target.device, dtype=torch.long)
            for _ in range(self.gamma):
                d_logits = self.draft.forward(ctx[:, -1:], kv_cache=None, pos_offset=pos + len(draft_toks))
                nxt = int(d_logits[:, -1, :].argmax(dim=-1).item())
                draft_toks.append(nxt)
                ctx = torch.cat([ctx, torch.tensor([[nxt]], device=self.target.device)], dim=1)

            # Verify [cur_tok, draft_0, ..., draft_{gamma-1}] in one batched target forward
            verify_in = torch.tensor([[cur_tok] + draft_toks], device=self.target.device, dtype=torch.long)
            t_logits = self.target.forward(verify_in, kv_cache=kv, pos_offset=pos)
            target_preds = t_logits[0].argmax(dim=-1).tolist()

            n_accept = 0
            for j, d_tok in enumerate(draft_toks):
                if target_preds[j] == d_tok:
                    n_accept += 1
                    generated.append(d_tok)
                    if len(generated) >= max_new_tokens:
                        break
                else:
                    generated.append(int(target_preds[j]))
                    break
            else:
                if len(generated) < max_new_tokens:
                    generated.append(int(target_preds[self.gamma]))

            accepted_total += n_accept
            cur_tok = generated[-1]
            pos += self.gamma + 1

        if self.target.device.type == "cuda":
            torch.cuda.synchronize()
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        total_drafted = max(1, draft_steps * self.gamma)
        stats = SpeculativeStats(
            tokens_generated=len(generated),
            draft_steps=draft_steps,
            accepted_tokens=accepted_total,
            acceptance_rate=accepted_total / float(total_drafted),
            elapsed_ms=elapsed_ms,
            tokens_per_sec=(len(generated) * 1000.0 / elapsed_ms) if elapsed_ms > 0 else 0.0,
        )
        return generated[:max_new_tokens], stats
