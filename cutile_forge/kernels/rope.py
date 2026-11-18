"""Rotary Position Embedding (RoPE) kernel in CuTile."""

from __future__ import annotations

import cuda.tile as ct
import torch


@ct.kernel(occupancy=2)
def rope_kernel(
    x_ptr,
    cos_ptr,
    sin_ptr,
    out_ptr,
    n_heads: int,
    seq_len: int,
    half_d: ct.Constant[int],
) -> None:
    """Apply rotary position embeddings to (B * H * S, 2, half_d) -> out."""
    row = ct.bid(0)
    seq_idx = row % seq_len

    x_pair = ct.astype(
        ct.load(x_ptr, index=(row, 0, 0), shape=(1, 2, half_d), padding_mode=ct.PaddingMode.ZERO),
        ct.float32,
    )
    x0 = ct.reshape(ct.extract(x_pair, index=(0, 0, 0), shape=(1, 1, half_d)), (1, half_d))
    x1 = ct.reshape(ct.extract(x_pair, index=(0, 1, 0), shape=(1, 1, half_d)), (1, half_d))

    c = ct.astype(
        ct.load(cos_ptr, index=(seq_idx, 0), shape=(1, half_d), padding_mode=ct.PaddingMode.ZERO),
        ct.float32,
    )
    s = ct.astype(
        ct.load(sin_ptr, index=(seq_idx, 0), shape=(1, half_d), padding_mode=ct.PaddingMode.ZERO),
        ct.float32,
    )

    o0 = x0 * c - x1 * s
    o1 = x1 * c + x0 * s
    out_pair = ct.cat((ct.reshape(o0, (1, 1, half_d)), ct.reshape(o1, (1, 1, half_d))), axis=1)
    ct.store(out_ptr, index=(row, 0, 0), tile=ct.astype(out_pair, ct.float16))


def precompute_rope_freqs(
    max_seq_len: int,
    head_dim: int,
    theta: float = 10000.0,
    device: torch.device | str = "cuda",
) -> tuple[torch.Tensor, torch.Tensor]:
    """Precompute (cos, sin) tables of shape (max_seq_len, head_dim // 2) in float16."""
    inv_freq = 1.0 / (
        theta ** (torch.arange(0, head_dim, 2, device=device, dtype=torch.float32) / head_dim)
    )
    t = torch.arange(max_seq_len, device=device, dtype=torch.float32)
    freqs = torch.outer(t, inv_freq)
    return freqs.cos().to(torch.float16), freqs.sin().to(torch.float16)


def apply_rope(
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    position_offset: int = 0,
) -> torch.Tensor:
    """Apply RoPE to `x` of shape (B, H, S, D) using CuTile."""
    if x.ndim != 4:
        raise ValueError(f"Expected 4D tensor (B, H, S, D), got {x.shape}")
    bsz, n_heads, seq_len, d_head = x.shape
    half_d = d_head // 2

    cos_slice = cos[position_offset : position_offset + seq_len, :half_d].contiguous()
    sin_slice = sin[position_offset : position_offset + seq_len, :half_d].contiguous()

    x_3d = x.contiguous().view(bsz * n_heads * seq_len, 2, half_d)
    out_3d = torch.empty_like(x_3d)
    rows = x_3d.shape[0]

    ct.launch(
        torch.cuda.current_stream(),
        (rows, 1, 1),
        rope_kernel,
        (x_3d, cos_slice, sin_slice, out_3d, int(n_heads), int(seq_len), half_d),
    )
    return out_3d.view(bsz, n_heads, seq_len, d_head)
