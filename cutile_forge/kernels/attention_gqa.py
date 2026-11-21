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


@ct.kernel(occupancy=2)
def flash_decoding_split_kv_kernel(
    q_ptr,  # (B * H_q, S_q, D)
    k_ptr,  # (B * H_kv, S_kv, D)
    v_ptr,  # (B * H_kv, S_kv, D)
    o_part_ptr,  # (B * H_q, num_splits, bq, D) fp32
    lse_part_ptr,  # (B * H_q, num_splits, bq, 1) fp32
    scale: float,
    s_q: int,
    s_kv: int,
    kv_offset: int,
    tiles_per_split: int,
    gqa_ratio: ct.Constant[int],
    bq: ct.Constant[int],
    bk: ct.Constant[int],
    d_head: ct.Constant[int],
    is_causal: ct.Constant[int],
) -> None:
    """FlashDecoding Split-KV kernel: parallelizes long-context decode across 170 SMs."""
    split_idx = ct.bid(0)
    bh_q = ct.bid(1)
    bh_kv = bh_q // gqa_ratio

    q_tile = ct.reshape(
        ct.load(
            q_ptr,
            index=(bh_q, 0, 0),
            shape=(1, bq, d_head),
            padding_mode=ct.PaddingMode.ZERO,
        ),
        (bq, d_head),
    )

    m_i = ct.full((bq, 1), -1.0e9, dtype=ct.float32)
    l_i = ct.zeros((bq, 1), dtype=ct.float32)
    acc = ct.zeros((bq, d_head), dtype=ct.float32)

    q_pos = ct.reshape(ct.arange(bq, dtype=ct.int32) + kv_offset, (bq, 1))
    k_base = ct.arange(bk, dtype=ct.int32)
    total_kv_tiles = ct.cdiv(s_kv, bk)
    start_tile = split_idx * tiles_per_split
    end_tile = min(start_tile + tiles_per_split, total_kv_tiles)

    for kv_tile in range(start_tile, end_tile):
        k_tile = ct.reshape(
            ct.load(
                k_ptr,
                index=(bh_kv, kv_tile, 0),
                shape=(1, bk, d_head),
                padding_mode=ct.PaddingMode.ZERO,
                latency=3,
            ),
            (bk, d_head),
        )
        v_tile = ct.reshape(
            ct.load(
                v_ptr,
                index=(bh_kv, kv_tile, 0),
                shape=(1, bk, d_head),
                padding_mode=ct.PaddingMode.ZERO,
                latency=3,
            ),
            (bk, d_head),
        )
        scores = ct.mma(q_tile, ct.transpose(k_tile), ct.zeros((bq, bk), dtype=ct.float32))
        scores = scores * scale

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
    lse = m_i + ct.log(ct.maximum(l_i, 1.0e-6))
    ct.store(
        o_part_ptr,
        index=(bh_q, split_idx, 0, 0),
        tile=ct.reshape(out, (1, 1, bq, d_head)),
    )
    ct.store(
        lse_part_ptr,
        index=(bh_q, split_idx, 0, 0),
        tile=ct.reshape(lse, (1, 1, bq, 1)),
    )


@ct.kernel(occupancy=4)
def flash_decoding_reduce_kernel(
    o_part_ptr,  # (BH, splits, 1, D) fp32
    lse_part_ptr,  # (BH, splits) fp32
    out_ptr,  # (BH, 1, D) fp16
    num_splits: int,
    pad_splits: ct.Constant[int],
    d_head: ct.Constant[int],
) -> None:
    """Fuse FlashDecoding Split-KV log-sum-exp reduction in CuTile (zero ATen ops)."""
    bh = ct.bid(0)
    lse = ct.load(
        lse_part_ptr, index=(bh, 0), shape=(1, pad_splits), padding_mode=ct.PaddingMode.NEG_INF
    )
    s_ids = ct.reshape(ct.arange(pad_splits, dtype=ct.int32), (1, pad_splits))
    lse_valid = ct.where(s_ids < num_splits, lse, -1.0e9)
    m_max = ct.max(lse_valid, axis=1, keepdims=True)
    w = ct.where(s_ids < num_splits, ct.exp(lse_valid - m_max), 0.0)
    w_sum = ct.maximum(ct.sum(w, axis=1, keepdims=True), 1.0e-6)

    o_part = ct.reshape(
        ct.load(
            o_part_ptr,
            index=(bh, 0, 0, 0),
            shape=(1, pad_splits, 1, d_head),
            padding_mode=ct.PaddingMode.ZERO,
        ),
        (pad_splits, d_head),
    )
    w_col = ct.reshape(w / w_sum, (pad_splits, 1))
    out = ct.sum(o_part * w_col, axis=0, keepdims=True)
    ct.store(
        out_ptr, index=(bh, 0, 0), tile=ct.reshape(ct.astype(out, out_ptr.dtype), (1, 1, d_head))
    )


def _next_pow2_int(n: int) -> int:
    p = 1
    while p < n:
        p *= 2
    return p


def flash_decoding_split_kv(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    num_splits: int = 16,
    causal: bool = True,
    sm_scale: float | None = None,
    o_part_buf: torch.Tensor | None = None,
    lse_part_buf: torch.Tensor | None = None,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    """FlashDecoding Split-KV attention partitioning `seq_kv` across `num_splits` parallel CTAs."""
    bsz, h_q, s_q, d_head = q.shape
    _, h_kv, s_kv, _ = k_cache.shape
    gqa_ratio = h_q // h_kv
    scale = sm_scale if sm_scale is not None else 1.0 / math.sqrt(d_head)
    kv_offset = s_kv - s_q
    bq = 16
    bk = 64

    total_kv_tiles = (s_kv + bk - 1) // bk
    eff_splits = max(1, min(num_splits, total_kv_tiles))
    tiles_per_split = (total_kv_tiles + eff_splits - 1) // eff_splits

    q_flat = (
        q.view(bsz * h_q, s_q, d_head)
        if q.is_contiguous()
        else q.contiguous().view(bsz * h_q, s_q, d_head)
    )
    k_flat = (
        k_cache.view(bsz * h_kv, s_kv, d_head)
        if k_cache.is_contiguous()
        else k_cache.contiguous().view(bsz * h_kv, s_kv, d_head)
    )
    v_flat = (
        v_cache.view(bsz * h_kv, s_kv, d_head)
        if v_cache.is_contiguous()
        else v_cache.contiguous().view(bsz * h_kv, s_kv, d_head)
    )

    o_part = (
        o_part_buf[: bsz * h_q, :eff_splits]
        if o_part_buf is not None
        else torch.empty((bsz * h_q, eff_splits, bq, d_head), device=q.device, dtype=torch.float32)
    )
    lse_part = (
        lse_part_buf[: bsz * h_q, :eff_splits]
        if lse_part_buf is not None
        else torch.empty((bsz * h_q, eff_splits, bq, 1), device=q.device, dtype=torch.float32)
    )

    grid = (eff_splits, bsz * h_q, 1)
    ct.launch(
        torch.cuda.current_stream(),
        grid,
        flash_decoding_split_kv_kernel,
        (
            q_flat,
            k_flat,
            v_flat,
            o_part,
            lse_part,
            float(scale),
            int(s_q),
            int(s_kv),
            int(kv_offset),
            int(tiles_per_split),
            gqa_ratio,
            bq,
            bk,
            d_head,
            1 if causal else 0,
        ),
    )

    if s_q == 1:
        out_flat = (
            out.view(bsz * h_q, 1, d_head)
            if out is not None
            else torch.empty((bsz * h_q, 1, d_head), device=q.device, dtype=q.dtype)
        )
        o_s1 = o_part[:, :, :1, :].contiguous()
        lse_s1 = lse_part[:, :, 0, 0].contiguous()
        pad_splits = _next_pow2_int(eff_splits)
        ct.launch(
            torch.cuda.current_stream(),
            (bsz * h_q, 1, 1),
            flash_decoding_reduce_kernel,
            (o_s1, lse_s1, out_flat, eff_splits, pad_splits, d_head),
        )
        return out_flat.view(bsz, h_q, 1, d_head)

    lse_s = lse_part[:, :, :s_q, 0]  # (BH, splits, S_q)
    m_max = lse_s.max(dim=1, keepdim=True).values
    w = torch.exp(lse_s - m_max)
    w_sum = w.sum(dim=1, keepdim=True).clamp(min=1e-6)
    res = (o_part[:, :, :s_q, :] * w.unsqueeze(-1)).sum(dim=1) / w_sum.squeeze(1).unsqueeze(-1)
    return res.to(q.dtype).view(bsz, h_q, s_q, d_head)


@ct.kernel(occupancy=2)
def attention_gqa_decode_pos_kernel(
    q_ptr,  # (B * H_q, 1, D) float16
    k_ptr,  # (B * H_kv, S_pool, D) float16 / fp8
    v_ptr,  # (B * H_kv, S_pool, D) float16 / fp8
    pos_ptr,  # (B,) int32
    o_ptr,  # (B * H_q, 1, D) float16
    sm_scale: float,
    max_kv_tiles: int,
    h_q: ct.Constant[int],
    gqa_ratio: ct.Constant[int],
    bq: ct.Constant[int],
    bk: ct.Constant[int],
    d_head: ct.Constant[int],
    dynamic_loop: ct.Constant[int],
) -> None:
    """Single-token GQA decode reading `s_kv = pos_ptr[b] + 1` on GPU (CUDA Graph capturable)."""
    bh_q = ct.bid(1)
    b_idx = bh_q // h_q
    bh_kv = bh_q // gqa_ratio

    cur_pos = ct.sum(
        ct.astype(
            ct.load(pos_ptr, index=(b_idx,), shape=(1,), padding_mode=ct.PaddingMode.ZERO),
            ct.int32,
        )
    )
    s_kv = cur_pos + 1

    q_3d = ct.load(
        q_ptr,
        index=(bh_q, 0, 0),
        shape=(1, bq, d_head),
        padding_mode=ct.PaddingMode.ZERO,
        latency=3,
    )
    q = ct.reshape(ct.astype(q_3d, ct.float16), (bq, d_head))

    m_i = ct.full((bq, 1), -1.0e9, dtype=ct.float32)
    l_i = ct.zeros((bq, 1), dtype=ct.float32)
    acc = ct.zeros((bq, d_head), dtype=ct.float32)
    k_base = ct.arange(bk, dtype=ct.int32)

    loop_tiles = min(max_kv_tiles, (s_kv + bk - 1) // bk) if dynamic_loop == 1 else max_kv_tiles

    for kv_tile in range(loop_tiles):
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
        k_tile = ct.reshape(ct.astype(k_3d, ct.float16), (bk, d_head))
        v_tile = ct.reshape(ct.astype(v_3d, ct.float16), (bk, d_head))

        k_pos = ct.reshape(kv_tile * bk + k_base, (1, bk))
        valid = k_pos < s_kv
        valid_col = ct.reshape(valid, (bk, 1))
        k_tile = ct.where(valid_col, k_tile, ct.astype(0.0, ct.float16))
        v_tile = ct.where(valid_col, v_tile, ct.astype(0.0, ct.float16))

        qk = ct.mma(q, ct.transpose(k_tile), ct.zeros((bq, bk), dtype=ct.float32))
        scores = qk * sm_scale
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
        index=(bh_q, 0, 0),
        tile=ct.reshape(ct.astype(out, ct.float16), (1, bq, d_head)),
    )


def fused_attention_gqa_decode_pos(
    q: torch.Tensor,
    k_pool: torch.Tensor,
    v_pool: torch.Tensor,
    pos_ptr: torch.Tensor,
    max_kv_len: int,
    out: torch.Tensor,
    sm_scale: float | None = None,
    dynamic_loop: bool = True,
) -> torch.Tensor:
    """Launch CUDA-Graph-capturable GQA decode attention reading `pos_ptr` on GPU."""
    bsz, h_q, _, d_head = q.shape
    h_kv = k_pool.shape[1]
    total_slots = k_pool.shape[2]
    gqa_ratio = h_q // h_kv
    scale = sm_scale if sm_scale is not None else 1.0 / math.sqrt(d_head)
    bq = 16
    bk = 64
    max_kv_tiles = (max_kv_len + bk - 1) // bk

    q_flat = q.view(bsz * h_q, 1, d_head)
    k_flat = k_pool.view(bsz * h_kv, total_slots, d_head)
    v_flat = v_pool.view(bsz * h_kv, total_slots, d_head)
    o_flat = out.view(bsz * h_q, 1, d_head)

    ct.launch(
        torch.cuda.current_stream(),
        (1, bsz * h_q, 1),
        attention_gqa_decode_pos_kernel,
        (
            q_flat,
            k_flat,
            v_flat,
            pos_ptr.view(bsz),
            o_flat,
            float(scale),
            int(max_kv_tiles),
            h_q,
            gqa_ratio,
            bq,
            bk,
            d_head,
            1 if dynamic_loop else 0,
        ),
    )
    return out


def fused_attention_gqa(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    causal: bool = True,
    sm_scale: float | None = None,
) -> torch.Tensor:
    """Compute Grouped-Query FlashAttention-2 in CuTile (auto-dispatches FlashDecoding for long-KV decode).

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

    if q.is_cuda and s_q <= 4 and s_kv >= 2048:
        return flash_decoding_split_kv(q, k, v, num_splits=16, causal=causal, sm_scale=sm_scale)

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
