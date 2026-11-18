"""FlashAttention-2 style Grouped-Query Attention (GQA 32:8) in CuTile."""

from __future__ import annotations

import math

import cuda.tile as ct
import torch


@ct.kernel(occupancy=2)
def attention_gqa_kernel(
    q_ptr,
    k_ptr,
    v_ptr,
    o_ptr,
    sm_scale: float,
    s_q: int,
    s_kv: int,
    kv_offset: int,
    gqa_ratio: ct.Constant[int],
    bq: ct.Constant[int],
    bk: ct.Constant[int],
    d_head: ct.Constant[int],
    is_causal: ct.Constant[int],
) -> None:
    """FlashAttention-2 tiled GQA kernel with online softmax.

    Shapes:
        q_ptr: (B * H_q, S_q, D) float16
        k_ptr: (B * H_kv, S_kv, D) float16
        v_ptr: (B * H_kv, S_kv, D) float16
        o_ptr: (B * H_q, S_q, D) float16
    """
    q_tile_idx = ct.bid(0)
    bh_q = ct.bid(1)
    bh_kv = bh_q // gqa_ratio

    q_3d = ct.load(
        q_ptr,
        index=(bh_q, q_tile_idx, 0),
        shape=(1, bq, d_head),
        padding_mode=ct.PaddingMode.ZERO,
        latency=3,
    )
    q = ct.reshape(q_3d, (bq, d_head))

    m_i = ct.full((bq, 1), -1.0e9, dtype=ct.float32)
    l_i = ct.zeros((bq, 1), dtype=ct.float32)
    acc = ct.zeros((bq, d_head), dtype=ct.float32)

    q_pos = ct.reshape(q_tile_idx * bq + ct.arange(bq, dtype=ct.int32) + kv_offset, (bq, 1))
    k_base = ct.arange(bk, dtype=ct.int32)

    max_k = (q_tile_idx + 1) * bq + kv_offset if is_causal else s_kv
    if max_k > s_kv:
        max_k = s_kv
    num_kv_tiles = (max_k + bk - 1) // bk

    for kv_tile in range(num_kv_tiles):
        k_3d = ct.load(
            k_ptr,
            index=(bh_kv, kv_tile, 0),
            shape=(1, bk, d_head),
            padding_mode=ct.PaddingMode.ZERO,
            latency=3,
        )
        v_3d = ct.load(
            v_ptr,
            index=(bh_kv, kv_tile, 0),
            shape=(1, bk, d_head),
            padding_mode=ct.PaddingMode.ZERO,
            latency=3,
        )
        k_tile = ct.reshape(k_3d, (bk, d_head))
        v_tile = ct.reshape(v_3d, (bk, d_head))

        qk = ct.mma(q, ct.transpose(k_tile), ct.zeros((bq, bk), dtype=ct.float32))
        scores = qk * sm_scale

        k_pos = ct.reshape(kv_tile * bk + k_base, (1, bk))
        valid = k_pos < s_kv
        if is_causal:
            valid = valid & (k_pos <= q_pos)
        scores = ct.where(valid, scores, -1.0e9)

        m_tile = ct.max(scores, axis=1, keepdims=True)
        m_new = ct.maximum(m_i, m_tile)
        alpha = ct.exp(m_i - m_new)
        p = ct.exp(scores - m_new)
        l_new = l_i * alpha + ct.sum(p, axis=1, keepdims=True)

        acc = acc * alpha
        acc = ct.mma(ct.astype(p, ct.float16), v_tile, acc)
        m_i = m_new
        l_i = l_new

    out = acc / ct.maximum(l_i, 1.0e-6)
    ct.store(
        o_ptr,
        index=(bh_q, q_tile_idx, 0),
        tile=ct.reshape(ct.astype(out, ct.float16), (1, bq, d_head)),
    )


def fused_attention_gqa(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    causal: bool = True,
    sm_scale: float | None = None,
) -> torch.Tensor:
    """Compute Grouped-Query FlashAttention-2 in CuTile.

    Args:
        q: (B, H_q, S_q, D) float16 CUDA tensor
        k: (B, H_kv, S_kv, D) float16 CUDA tensor
        v: (B, H_kv, S_kv, D) float16 CUDA tensor
        causal: Whether to apply causal masking (aligned to bottom-right for S_q <= S_kv).
        sm_scale: Optional softmax scale (defaults to 1 / sqrt(D)).
    """
    if q.ndim != 4 or k.ndim != 4 or v.ndim != 4:
        raise ValueError(f"Expected 4D tensors (B, H, S, D), got {q.shape}, {k.shape}, {v.shape}")

    bsz, h_q, s_q, d_head = q.shape
    _, h_kv, s_kv, _ = k.shape
    if h_q % h_kv != 0:
        raise ValueError(f"H_q ({h_q}) must be divisible by H_kv ({h_kv})")

    gqa_ratio = h_q // h_kv
    scale = sm_scale if sm_scale is not None else 1.0 / math.sqrt(d_head)
    kv_offset = s_kv - s_q

    bq = 16 if s_q <= 16 else 64
    bk = 64

    q_flat = q.contiguous().view(bsz * h_q, s_q, d_head)
    k_flat = k.contiguous().view(bsz * h_kv, s_kv, d_head)
    v_flat = v.contiguous().view(bsz * h_kv, s_kv, d_head)
    o_flat = torch.empty_like(q_flat)

    grid = (ct.cdiv(s_q, bq), bsz * h_q, 1)
    ct.launch(
        torch.cuda.current_stream(),
        grid,
        attention_gqa_kernel,
        (
            q_flat,
            k_flat,
            v_flat,
            o_flat,
            float(scale),
            int(s_q),
            int(s_kv),
            int(kv_offset),
            gqa_ratio,
            bq,
            bk,
            d_head,
            1 if causal else 0,
        ),
    )
    return o_flat.view(bsz, h_q, s_q, d_head)
