"""Token sampling helpers (greedy, temperature, top-k, top-p)."""

from __future__ import annotations

import torch

from cutile_forge.kernels.softmax import online_softmax


def sample_tokens(
    logits: torch.Tensor,
    temperature: float = 0.0,
    top_k: int = 50,
    top_p: float = 0.9,
) -> torch.Tensor:
    """Sample next token IDs from logits of shape (B, vocab_size).

    Uses exact greedy argmax when `temperature <= 0.0`, otherwise applies
    temperature scaling, top-k filtering, top-p nucleus filtering, and
    CuTile online softmax before multinomial sampling.
    """
    if logits.ndim == 1:
        logits = logits.unsqueeze(0)

    if temperature <= 0.0:
        return torch.argmax(logits, dim=-1)

    scaled = (logits.float() / float(temperature)).to(torch.float16)

    if top_k > 0 and top_k < scaled.shape[-1]:
        top_vals, _ = torch.topk(scaled, top_k, dim=-1)
        kth_val = top_vals[:, -1].unsqueeze(-1)
        scaled = scaled.masked_fill(scaled < kth_val, float("-inf"))

    probs = (
        online_softmax(scaled).float() if scaled.is_cuda else torch.softmax(scaled.float(), dim=-1)
    )

    if 0.0 < top_p < 1.0:
        sorted_probs, sorted_indices = torch.sort(probs, descending=True, dim=-1)
        cum_probs = torch.cumsum(sorted_probs, dim=-1)
        remove_mask = cum_probs > top_p
        remove_mask[..., 1:] = remove_mask[..., :-1].clone()
        remove_mask[..., 0] = False
        sorted_probs = sorted_probs.masked_fill(remove_mask, 0.0)
        sorted_probs = sorted_probs / sorted_probs.sum(dim=-1, keepdim=True).clamp(min=1e-9)
        sampled_sorted = torch.multinomial(sorted_probs, num_samples=1)
        return sorted_indices.gather(-1, sampled_sorted).squeeze(-1)

    return torch.multinomial(probs, num_samples=1).squeeze(-1)
