"""Prefill and decode inference loop with TFG dispatch and paged KV-cache."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Iterator, List, Optional

import torch

from cutile_forge.engine.kv_cache import PagedKVCache
from cutile_forge.engine.model import Mistral7B
from cutile_forge.kernels.sampling import sample_tokens
from cutile_forge.utils.memory import get_peak_vram_gb, reset_peak_vram


@dataclass
class GenerationStats:
    """Throughput, latency, and VRAM telemetry for a single generation run."""

    prompt_tokens: int
    generated_tokens: int
    prefill_ms: float
    decode_ms: float
    tokens_per_sec: float
    ms_per_token: float
    peak_vram_gb: float


class InferenceEngine:
    """High-level prefill + decode engine over `Mistral7B` and `PagedKVCache`."""

    def __init__(
        self,
        model: Mistral7B,
        page_size: int = 16,
        kv_gpu_budget_bytes: int = 2 * 1024 * 1024 * 1024,
    ) -> None:
        self.model = model
        self.config = model.config
        self.kv_cache = PagedKVCache(
            num_layers=self.config.num_hidden_layers,
            num_kv_heads=self.config.num_key_value_heads,
            head_dim=self.config.head_dim,
            page_size=page_size,
            gpu_budget_bytes=kv_gpu_budget_bytes,
            device=model.device,
        )
        self.last_stats: Optional[GenerationStats] = None

    @torch.inference_mode()
    def generate_ids(
        self,
        prompt_ids: torch.Tensor,
        max_new_tokens: int = 64,
        temperature: float = 0.0,
        top_k: int = 50,
        top_p: float = 0.9,
        eos_token_id: Optional[int] = None,
        backend: str = "cutile",
    ) -> List[int]:
        """Generate token IDs autoregressively from `prompt_ids` of shape `(1, S)`."""
        if prompt_ids.ndim == 1:
            prompt_ids = prompt_ids.unsqueeze(0)
        prompt_ids = prompt_ids.to(self.model.device)
        self.kv_cache.reset()
        reset_peak_vram()

        t_pre0 = time.perf_counter()
        logits = self.model.forward(
            prompt_ids,
            kv_cache=self.kv_cache,
            pos_offset=0,
            backend=backend,
            last_token_only=True,
        )
        if self.model.device.type == "cuda":
            torch.cuda.synchronize()
        prefill_ms = (time.perf_counter() - t_pre0) * 1000.0

        next_tok = sample_tokens(
            logits[:, -1, :], temperature=temperature, top_k=top_k, top_p=top_p
        )
        generated: List[int] = [int(next_tok.item())]
        pos = int(prompt_ids.shape[1])

        t_dec0 = time.perf_counter()
        for _ in range(max(0, max_new_tokens - 1)):
            if eos_token_id is not None and generated[-1] == eos_token_id:
                break
            cur_in = next_tok.view(1, 1)
            logits = self.model.forward(
                cur_in, kv_cache=self.kv_cache, pos_offset=pos, backend=backend
            )
            next_tok = sample_tokens(
                logits[:, -1, :], temperature=temperature, top_k=top_k, top_p=top_p
            )
            generated.append(int(next_tok.item()))
            pos += 1

        if self.model.device.type == "cuda":
            torch.cuda.synchronize()
        decode_ms = (time.perf_counter() - t_dec0) * 1000.0

        n_dec = max(1, len(generated) - 1) if len(generated) > 1 else 1
        ms_per_tok = decode_ms / n_dec if decode_ms > 0 else prefill_ms
        tps = 1000.0 / ms_per_tok if ms_per_tok > 0 else 0.0
        self.last_stats = GenerationStats(
            prompt_tokens=int(prompt_ids.shape[1]),
            generated_tokens=len(generated),
            prefill_ms=prefill_ms,
            decode_ms=decode_ms,
            tokens_per_sec=tps,
            ms_per_token=ms_per_tok,
            peak_vram_gb=get_peak_vram_gb(),
        )
        return generated

    @torch.inference_mode()
    def stream_ids(
        self,
        prompt_ids: torch.Tensor,
        max_new_tokens: int = 64,
        temperature: float = 0.0,
        top_k: int = 50,
        top_p: float = 0.9,
        eos_token_id: Optional[int] = None,
        backend: str = "cutile",
    ) -> Iterator[int]:
        """Yield generated token IDs one by one."""
        if prompt_ids.ndim == 1:
            prompt_ids = prompt_ids.unsqueeze(0)
        prompt_ids = prompt_ids.to(self.model.device)
        self.kv_cache.reset()

        logits = self.model.forward(
            prompt_ids, kv_cache=self.kv_cache, pos_offset=0, backend=backend
        )
        next_tok = sample_tokens(
            logits[:, -1, :], temperature=temperature, top_k=top_k, top_p=top_p
        )
        tok_id = int(next_tok.item())
        yield tok_id
        if eos_token_id is not None and tok_id == eos_token_id:
            return

        pos = int(prompt_ids.shape[1])
        for _ in range(max(0, max_new_tokens - 1)):
            logits = self.model.forward(
                next_tok.view(1, 1), kv_cache=self.kv_cache, pos_offset=pos, backend=backend
            )
            next_tok = sample_tokens(
                logits[:, -1, :], temperature=temperature, top_k=top_k, top_p=top_p
            )
            tok_id = int(next_tok.item())
            yield tok_id
            pos += 1
            if eos_token_id is not None and tok_id == eos_token_id:
                break

