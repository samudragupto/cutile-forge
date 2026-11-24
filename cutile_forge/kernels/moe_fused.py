"""CuTile kernels for fused MoE top-k gating and persistent grouped INT4 expert GEMM."""

from __future__ import annotations

from typing import Any

import torch

from cutile_forge.cuda_tile_init import ct


@ct.kernel
def moe_softmax_kernel(
    logits_ptr,
    probs_ptr,
    num_experts: ct.Constant[int],
) -> None:
    """Compute per-token softmax over router logits in CuTile."""
    row = ct.bid(0)
    tile = ct.astype(
        ct.load(
            logits_ptr,
            index=(row, 0),
            shape=(1, num_experts),
            padding_mode=ct.PaddingMode.ZERO,
        ),
        ct.float32,
    )
    row_max = ct.max(tile, axis=1)
    ex = ct.exp(tile - ct.reshape(row_max, (1, 1)))
    denom = ct.sum(ex, axis=1)
    probs = ex / ct.reshape(denom, (1, 1))
    ct.store(probs_ptr, index=(row, 0), tile=ct.astype(probs, probs_ptr.dtype))


@ct.kernel(occupancy=2)
def moe_grouped_gemm_int4_kernel(
    x_ptr,  # (E, M_pad, K) fp16
    qw_ptr,  # (E, K // 2, N) uint8
    scales_ptr,  # (E, K // gs, N) fp16
    zeros_ptr,  # (E, K // gs, N) fp16
    y_ptr,  # (E, M_pad, N) fp16
    tm: ct.Constant[int],
    tn: ct.Constant[int],
    tk: ct.Constant[int],
    gs: ct.Constant[int],
) -> None:
    """Persistent grouped INT4-dequant + FP16 MMA across all E experts in one launch."""
    e_id = ct.bid(2)
    pid_m = ct.bid(1)
    pid_n = ct.bid(0)

    num_k = ct.num_tiles(x_ptr, axis=2, shape=(1, tm, tk))
    half_tk: ct.Constant[int] = tk // 2
    acc = ct.zeros((tm, tn), dtype=ct.float32)

    for k in range(num_k):
        tile_a = ct.reshape(
            ct.load(
                x_ptr,
                index=(e_id, pid_m, k),
                shape=(1, tm, tk),
                padding_mode=ct.PaddingMode.ZERO,
                latency=3,
            ),
            (tm, tk),
        )
        qw = ct.reshape(
            ct.load(
                qw_ptr,
                index=(e_id, k, pid_n),
                shape=(1, half_tk, tn),
                padding_mode=ct.PaddingMode.ZERO,
                latency=3,
            ),
            (half_tk, tn),
        )
        lo = ct.astype(ct.bitwise_and(qw, 0x0F), ct.float16)
        hi = ct.astype(ct.bitwise_and(ct.bitwise_rshift(qw, 4), 0x0F), ct.float16)
        lo_e = ct.reshape(lo, (half_tk, 1, tn))
        hi_e = ct.reshape(hi, (half_tk, 1, tn))
        w_int = ct.reshape(ct.cat((lo_e, hi_e), axis=1), (tk, tn))

        g_idx = (k * tk) // gs
        scale = ct.reshape(
            ct.load(
                scales_ptr,
                index=(e_id, g_idx, pid_n),
                shape=(1, 1, tn),
                padding_mode=ct.PaddingMode.ZERO,
                latency=3,
            ),
            (1, tn),
        )
        zero = ct.reshape(
            ct.load(
                zeros_ptr,
                index=(e_id, g_idx, pid_n),
                shape=(1, 1, tn),
                padding_mode=ct.PaddingMode.ZERO,
                latency=3,
            ),
            (1, tn),
        )
        w_fp16 = (w_int - zero) * scale
        acc = ct.mma(tile_a, w_fp16, acc)

    ct.store(
        y_ptr, index=(e_id, pid_m, pid_n), tile=ct.reshape(ct.astype(acc, ct.float16), (1, tm, tn))
    )


def fused_topk_gating(
    logits: torch.Tensor,
    top_k: int = 2,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute normalized top-k router probabilities and indices from `(M, E)` logits."""
    if logits.ndim != 2:
        raise ValueError(f"Expected 2D router logits (M, E), got {tuple(logits.shape)}")
    m, num_experts = logits.shape
    if top_k <= 0 or top_k > num_experts:
        raise ValueError(f"top_k={top_k} invalid for num_experts={num_experts}")

    if logits.is_cuda and (num_experts & (num_experts - 1)) == 0:
        probs = torch.empty((m, num_experts), device=logits.device, dtype=torch.float32)
        ct.launch(
            torch.cuda.current_stream(),
            (m, 1, 1),
            moe_softmax_kernel,
            (logits.to(torch.float32).contiguous(), probs, num_experts),
        )
    else:
        probs = torch.softmax(logits.float(), dim=-1)

    topk_vals, topk_idx = torch.topk(probs, k=top_k, dim=-1)
    topk_probs = (topk_vals / topk_vals.sum(dim=-1, keepdim=True).clamp(min=1e-6)).to(torch.float16)
    return topk_probs, topk_idx


def fused_moe_int4_gemm(
    x: torch.Tensor,
    expert_qweights: torch.Tensor,
    expert_scales: torch.Tensor,
    expert_zeros: torch.Tensor,
    router_probs: torch.Tensor,
    topk_indices: torch.Tensor,
    group_size: int = 128,
) -> torch.Tensor:
    """Persistent grouped INT4 expert GEMM executing all active experts in parallel.

    Sorts/compacts routed tokens by expert ID without CPU-GPU syncs and dispatches a
    single 3D-grid CuTile grouped-GEMM kernel across all experts simultaneously.
    """
    m, k = x.shape
    num_experts, _, n = expert_qweights.shape
    top_k = topk_indices.shape[1]

    if not x.is_cuda:
        from cutile_forge.kernels.gemm_dequant import unpack_int4_weights_torch

        out = torch.zeros((m, n), device=x.device, dtype=torch.float16)
        for e in range(num_experts):
            token_mask = topk_indices == e
            if not bool(token_mask.any()):
                continue
            tok_idx, k_slot = torch.where(token_mask)
            w_fp16 = unpack_int4_weights_torch(
                expert_qweights[e], expert_scales[e], expert_zeros[e], group_size=group_size
            )
            y_e = torch.matmul(x.index_select(0, tok_idx), w_fp16)
            out.index_add_(
                0, tok_idx, (y_e * router_probs[tok_idx, k_slot].unsqueeze(1)).to(out.dtype)
            )
        return out

    # Sort routed assignments by expert ID on GPU
    flat_exp = topk_indices.reshape(-1)  # (M * top_k,)
    flat_tok = torch.arange(m, device=x.device).repeat_interleave(top_k)
    flat_prob = router_probs.reshape(-1)

    order = torch.argsort(flat_exp, stable=True)
    sorted_exp = flat_exp.index_select(0, order)
    sorted_tok = flat_tok.index_select(0, order)
    sorted_prob = flat_prob.index_select(0, order)

    counts = torch.bincount(sorted_exp, minlength=num_experts)
    max_per_exp = max(1, int(counts.max().item()))
    tm = 16 if max_per_exp <= 16 else (32 if max_per_exp <= 32 else 64)
    tn = 128
    tk = 64 if group_size >= 64 else 32
    m_pad = ((max_per_exp + tm - 1) // tm) * tm

    # Compute within-expert slot indices completely on GPU
    offsets = torch.cumsum(counts, dim=0) - counts
    slot_in_exp = torch.arange(sorted_exp.numel(), device=x.device) - offsets.index_select(
        0, sorted_exp
    )

    x_grouped = torch.zeros((num_experts, m_pad, k), device=x.device, dtype=torch.float16)
    x_grouped[sorted_exp, slot_in_exp] = x.index_select(0, sorted_tok)
    y_grouped = torch.empty((num_experts, m_pad, n), device=x.device, dtype=torch.float16)

    grid = (ct.cdiv(n, tn), ct.cdiv(m_pad, tm), num_experts)
    ct.launch(
        torch.cuda.current_stream(),
        grid,
        moe_grouped_gemm_int4_kernel,
        (
            x_grouped,
            expert_qweights.contiguous(),
            expert_scales.contiguous(),
            expert_zeros.contiguous(),
            y_grouped,
            tm,
            tn,
            tk,
            group_size,
        ),
    )

    y_routed = y_grouped[sorted_exp, slot_in_exp] * sorted_prob.unsqueeze(1)
    out = torch.zeros((m, n), device=x.device, dtype=torch.float16)
    out.index_add_(0, sorted_tok, y_routed.to(torch.float16))
    return out


def _next_pow2_moe(n: int) -> int:
    p = 1
    while p < n:
        p *= 2
    return p


@ct.kernel(occupancy=2)
def cutile_moe_router_topk_kernel(
    x_ptr,
    router_w_ptr,
    probs_out_ptr,
    idx_out_ptr,
    k_dim: int,
    num_experts: int,
    e_pad: ct.Constant[int],
    top_k: ct.Constant[int],
    top_k_pad: ct.Constant[int],
    tm: ct.Constant[int],
    tk: ct.Constant[int],
) -> None:
    """Fused MoE router projection + softmax + top-k normalization in CuTile (no torch.topk)."""
    pid_m = ct.bid(0)
    num_kt = (k_dim + tk - 1) // tk
    acc = ct.zeros((tm, e_pad), dtype=ct.float32)

    for kt in range(num_kt):
        tx = ct.load(
            x_ptr, index=(pid_m, kt), shape=(tm, tk), padding_mode=ct.PaddingMode.ZERO, latency=3
        )
        tw = ct.load(
            router_w_ptr,
            index=(0, kt),
            shape=(e_pad, tk),
            padding_mode=ct.PaddingMode.ZERO,
            latency=3,
        )
        acc = ct.mma(tx, ct.transpose(tw), acc)

    e_ids = ct.reshape(ct.arange(e_pad, dtype=ct.int32), (1, e_pad))
    valid_e = e_ids < num_experts
    logits = ct.where(valid_e, acc, -1.0e9)
    mx = ct.max(logits, axis=1, keepdims=True)
    ex = ct.where(valid_e, ct.exp(logits - mx), 0.0)
    probs = ex / ct.maximum(ct.sum(ex, axis=1, keepdims=True), 1.0e-6)

    scores = ct.where(valid_e, probs, -1.0)
    sel_vals = ct.zeros((tm, top_k_pad), dtype=ct.float32)
    sel_idx = ct.zeros((tm, top_k_pad), dtype=ct.int32)
    k_ar = ct.reshape(ct.arange(top_k_pad, dtype=ct.int32), (1, top_k_pad))

    for ki in range(top_k):
        best_val = ct.max(scores, axis=1, keepdims=True)
        best_i = ct.reshape(ct.astype(ct.argmax(scores, axis=1), ct.int32), (tm, 1))
        scores = ct.where(e_ids == best_i, -1.0, scores)
        k_mask = k_ar == ki
        sel_vals = ct.where(k_mask, ct.broadcast_to(best_val, (tm, top_k_pad)), sel_vals)
        sel_idx = ct.where(k_mask, ct.broadcast_to(best_i, (tm, top_k_pad)), sel_idx)

    w_sum = ct.maximum(ct.sum(sel_vals, axis=1, keepdims=True), 1.0e-6)
    sel_vals = sel_vals / w_sum

    ct.store(probs_out_ptr, index=(pid_m, 0), tile=ct.astype(sel_vals, probs_out_ptr.dtype))
    ct.store(idx_out_ptr, index=(pid_m, 0), tile=sel_idx)


@ct.kernel(occupancy=4)
def cutile_moe_bucket_kernel(
    x_ptr,  # (M, K)
    idx_ptr,  # (M, top_k) int32
    counts_ptr,  # (E,) int32
    x_grouped_ptr,  # (E, max_per_exp, K)
    slot_map_ptr,  # (M, top_k) int32
    max_per_exp: int,
    k_dim: int,
    tile_k: ct.Constant[int],
) -> None:
    """Assign routed tokens into expert buckets atomically on GPU (zero host sync / argsort)."""
    tok = ct.bid(0)
    slot = ct.bid(1)
    e_tile = ct.load(idx_ptr, index=(tok, slot), shape=(1, 1), padding_mode=ct.PaddingMode.ZERO)
    e_1d = ct.reshape(ct.astype(e_tile, ct.int32), (1,))
    pos_tile = ct.atomic_add(counts_ptr, (e_1d,), 1)
    ct.store(
        slot_map_ptr, index=(tok, slot), tile=ct.reshape(ct.astype(pos_tile, ct.int32), (1, 1))
    )

    row_x = ct.reshape(ct.full((1,), tok, dtype=ct.int32), (1, 1))
    e_idx = ct.reshape(e_1d, (1, 1))
    p_idx = ct.reshape(ct.astype(pos_tile, ct.int32), (1, 1))
    num_kt = (k_dim + tile_k - 1) // tile_k
    for kt in range(num_kt):
        cols = ct.reshape(kt * tile_k + ct.arange(tile_k, dtype=ct.int32), (1, tile_k))
        xv = ct.gather(x_ptr, (row_x, cols), padding_value=0, latency=3)
        ct.scatter(x_grouped_ptr, (e_idx, p_idx, cols), xv)


@ct.kernel(occupancy=4)
def cutile_moe_scatter_combine_kernel(
    y_grouped_ptr,  # (E, max_per_exp, N)
    idx_ptr,  # (M, top_k)
    slot_map_ptr,  # (M, top_k)
    probs_ptr,  # (M, top_k)
    out_ptr,  # (M, N)
    top_k: ct.Constant[int],
    tile_n: ct.Constant[int],
) -> None:
    """Weighted combine of expert outputs back into `(M, N)` without atomic contention."""
    tok = ct.bid(0)
    pid_n = ct.bid(1)
    cols = ct.reshape(pid_n * tile_n + ct.arange(tile_n, dtype=ct.int32), (1, tile_n))
    acc = ct.zeros((1, tile_n), dtype=ct.float32)
    t_idx = ct.reshape(ct.full((1,), tok, dtype=ct.int32), (1, 1))

    for s in range(top_k):
        s_idx = ct.reshape(ct.full((1,), s, dtype=ct.int32), (1, 1))
        e_id = ct.astype(ct.gather(idx_ptr, (t_idx, s_idx), padding_value=0), ct.int32)
        p_pos = ct.astype(ct.gather(slot_map_ptr, (t_idx, s_idx), padding_value=0), ct.int32)
        wt = ct.astype(ct.gather(probs_ptr, (t_idx, s_idx), padding_value=0), ct.float32)
        yv = ct.astype(
            ct.gather(y_grouped_ptr, (e_id, p_pos, cols), padding_value=0, latency=3), ct.float32
        )
        acc = acc + yv * wt

    ct.store(out_ptr, index=(tok, pid_n), tile=ct.astype(acc, out_ptr.dtype))


_ROUTER_SCRATCH_CACHE: dict[tuple[Any, ...], dict[str, torch.Tensor]] = {}


@ct.kernel(occupancy=4)
def moe_router_splitk_kernel(
    x_ptr,  # (M, K) fp16
    rw_ptr,  # (E, K) fp16
    part_ptr,  # (split_k, M, E_pad) fp32
    k_dim: int,
    split_k: ct.Constant[int],
    e_pad: ct.Constant[int],
    tm: ct.Constant[int],
    tk: ct.Constant[int],
) -> None:
    """Split-K Stage 1 for small-batch MoE router projection."""
    pid_m = ct.bid(0)
    sk_id = ct.bid(1)
    num_kt = (k_dim + tk - 1) // tk
    k_per_split = (num_kt + split_k - 1) // split_k
    k_start = sk_id * k_per_split
    k_end = min(num_kt, k_start + k_per_split)
    acc = ct.zeros((tm, e_pad), dtype=ct.float32)
    for kt in range(k_start, k_end):
        tx = ct.load(
            x_ptr, index=(pid_m, kt), shape=(tm, tk), padding_mode=ct.PaddingMode.ZERO, latency=3
        )
        tw = ct.load(
            rw_ptr, index=(0, kt), shape=(e_pad, tk), padding_mode=ct.PaddingMode.ZERO, latency=3
        )
        acc = ct.mma(tx, ct.transpose(tw), acc)
    ct.store(part_ptr, index=(sk_id, pid_m, 0), tile=ct.reshape(acc, (1, tm, e_pad)))


@ct.kernel(occupancy=4)
def moe_router_reduce_topk_kernel(
    part_ptr,  # (split_k, M, E_pad) fp32
    probs_out_ptr,  # (M, top_k_pad)
    idx_out_ptr,  # (M, top_k_pad) int32
    num_experts: int,
    split_k: ct.Constant[int],
    e_pad: ct.Constant[int],
    top_k: ct.Constant[int],
    top_k_pad: ct.Constant[int],
    tm: ct.Constant[int],
) -> None:
    """Split-K Stage 2 reduction + softmax + top-k normalization for MoE router."""
    pid_m = ct.bid(0)
    acc = ct.zeros((tm, e_pad), dtype=ct.float32)
    for sk in range(split_k):
        acc = acc + ct.reshape(
            ct.load(
                part_ptr,
                index=(sk, pid_m, 0),
                shape=(1, tm, e_pad),
                padding_mode=ct.PaddingMode.ZERO,
                latency=3,
            ),
            (tm, e_pad),
        )
    e_ids = ct.reshape(ct.arange(e_pad, dtype=ct.int32), (1, e_pad))
    valid_e = e_ids < num_experts
    logits = ct.where(valid_e, acc, -1.0e9)
    mx = ct.max(logits, axis=1, keepdims=True)
    ex = ct.where(valid_e, ct.exp(logits - mx), 0.0)
    probs = ex / ct.maximum(ct.sum(ex, axis=1, keepdims=True), 1.0e-6)

    scores = ct.where(valid_e, probs, -1.0)
    sel_vals = ct.zeros((tm, top_k_pad), dtype=ct.float32)
    sel_idx = ct.zeros((tm, top_k_pad), dtype=ct.int32)
    k_ar = ct.reshape(ct.arange(top_k_pad, dtype=ct.int32), (1, top_k_pad))
    for ki in range(top_k):
        best_val = ct.max(scores, axis=1, keepdims=True)
        best_i = ct.reshape(ct.astype(ct.argmax(scores, axis=1), ct.int32), (tm, 1))
        scores = ct.where(e_ids == best_i, -1.0, scores)
        k_mask = k_ar == ki
        sel_vals = ct.where(k_mask, ct.broadcast_to(best_val, (tm, top_k_pad)), sel_vals)
        sel_idx = ct.where(k_mask, ct.broadcast_to(best_i, (tm, top_k_pad)), sel_idx)
    sel_vals = sel_vals / ct.maximum(ct.sum(sel_vals, axis=1, keepdims=True), 1.0e-6)
    ct.store(probs_out_ptr, index=(pid_m, 0), tile=ct.astype(sel_vals, probs_out_ptr.dtype))
    ct.store(idx_out_ptr, index=(pid_m, 0), tile=sel_idx)


def cutile_moe_router_topk(
    x: torch.Tensor,
    router_weight: torch.Tensor,
    top_k: int = 2,
    out_dtype: torch.dtype | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute MoE router logits, softmax, and normalized top-k gating in one CuTile kernel."""
    m, k_dim = x.shape
    num_experts = router_weight.shape[0]
    if top_k <= 0 or top_k > num_experts:
        raise ValueError(f"top_k={top_k} invalid for num_experts={num_experts}")
    target_dtype = out_dtype or x.dtype

    if not x.is_cuda or x.dtype != torch.float16:
        logits = torch.matmul(x.float(), router_weight.float().t())
        probs = torch.softmax(logits, dim=-1)
        vals, idx = torch.topk(probs, k=top_k, dim=-1)
        vals = vals / vals.sum(dim=-1, keepdim=True).clamp(min=1e-6)
        return vals.to(target_dtype), idx.to(torch.int32)

    e_pad = max(16, _next_pow2_moe(num_experts))
    top_k_pad = _next_pow2_moe(top_k)
    tm, tk = 16, 64
    x_c = x if x.is_contiguous() else x.contiguous()
    rw_c = router_weight if router_weight.is_contiguous() else router_weight.contiguous()
    probs_out = torch.empty((m, top_k_pad), device=x.device, dtype=target_dtype)
    idx_out = torch.empty((m, top_k_pad), device=x.device, dtype=torch.int32)

    if m <= 16 and k_dim >= 1024 and k_dim % tk == 0:
        split_k = 16 if k_dim >= 2048 else 8
        r_key = (str(x.device), split_k, m, e_pad)
        r_sc = _ROUTER_SCRATCH_CACHE.get(r_key)
        if r_sc is None:
            r_sc = {"part": torch.empty((split_k, m, e_pad), device=x.device, dtype=torch.float32)}
            _ROUTER_SCRATCH_CACHE[r_key] = r_sc
        ct.launch(
            torch.cuda.current_stream(),
            (ct.cdiv(m, tm), split_k, 1),
            moe_router_splitk_kernel,
            (x_c, rw_c, r_sc["part"], k_dim, split_k, e_pad, tm, tk),
        )
        ct.launch(
            torch.cuda.current_stream(),
            (ct.cdiv(m, tm), 1, 1),
            moe_router_reduce_topk_kernel,
            (r_sc["part"], probs_out, idx_out, num_experts, split_k, e_pad, top_k, top_k_pad, tm),
        )
        return probs_out[:, :top_k], idx_out[:, :top_k]

    ct.launch(
        torch.cuda.current_stream(),
        (ct.cdiv(m, tm), 1, 1),
        cutile_moe_router_topk_kernel,
        (
            x_c,
            rw_c,
            probs_out,
            idx_out,
            k_dim,
            num_experts,
            e_pad,
            top_k,
            top_k_pad,
            tm,
            tk,
        ),
    )
    return probs_out[:, :top_k], idx_out[:, :top_k]


@ct.kernel(occupancy=4)
def cutile_moe_weighted_add_kernel(
    out_ptr,  # (M, N) float32/float16
    tok_idx_ptr,  # (R,) int32
    y_ptr,  # (R, N) float16/float32
    w_ptr,  # (R,) float32/float16
    tile_n: ct.Constant[int],
) -> None:
    """Accumulate `out[tok_idx[r], :] += y[r, :] * w[r]` in FP32 without ATen index_add_."""
    r = ct.bid(0)
    pid_n = ct.bid(1)
    cols = ct.reshape(pid_n * tile_n + ct.arange(tile_n, dtype=ct.int32), (1, tile_n))
    dst_row = ct.astype(
        ct.load(tok_idx_ptr, index=(r,), shape=(1,), padding_mode=ct.PaddingMode.ZERO),
        ct.int32,
    )
    dst_2d = ct.reshape(dst_row, (1, 1))
    wt = ct.astype(
        ct.load(w_ptr, index=(r,), shape=(1,), padding_mode=ct.PaddingMode.ZERO),
        ct.float32,
    )
    wt_2d = ct.reshape(wt, (1, 1))
    yv = ct.astype(
        ct.load(y_ptr, index=(r, pid_n), shape=(1, tile_n), padding_mode=ct.PaddingMode.ZERO),
        ct.float32,
    )
    cur = ct.astype(ct.gather(out_ptr, (dst_2d, cols), padding_value=0), ct.float32)
    ct.scatter(out_ptr, (dst_2d, cols), ct.astype(cur + yv * wt_2d, out_ptr.dtype))


def cutile_moe_weighted_add(
    out: torch.Tensor,
    tok_idx: torch.Tensor,
    expert_out: torch.Tensor,
    row_weights: torch.Tensor,
) -> torch.Tensor:
    """Accumulate weighted expert outputs into `out` indexed by distinct `tok_idx` rows."""
    r_count, n_cols = expert_out.shape
    if r_count == 0:
        return out
    if not out.is_cuda or out.dtype not in (torch.float16, torch.float32):
        out.index_add_(
            0,
            tok_idx.to(device=out.device, dtype=torch.long),
            (expert_out.float() * row_weights.float().unsqueeze(1)).to(out.dtype),
        )
        return out

    out_c = out if out.is_contiguous() else out.contiguous()
    idx_c = tok_idx.to(device=out.device, dtype=torch.int32).contiguous()
    y_c = expert_out if expert_out.is_contiguous() else expert_out.contiguous()
    w_c = row_weights.to(device=out.device, dtype=torch.float32).contiguous()
    tile_n = 256 if n_cols >= 256 else 128
    ct.launch(
        torch.cuda.current_stream(),
        (r_count, ct.cdiv(n_cols, tile_n), 1),
        cutile_moe_weighted_add_kernel,
        (out_c, idx_c, y_c, w_c, tile_n),
    )
    if out_c is not out:
        out.copy_(out_c)
    return out


def fused_moe_int4_native(
    x: torch.Tensor,
    router_weight: torch.Tensor,
    expert_qweights: torch.Tensor,
    expert_scales: torch.Tensor,
    expert_zeros: torch.Tensor,
    top_k: int = 2,
    group_size: int = 128,
) -> torch.Tensor:
    """Zero-host-sync, zero-ATen-compute native CuTile MoE block."""
    m, k = x.shape
    num_experts, _, n = expert_qweights.shape
    probs, topk_idx = cutile_moe_router_topk(x, router_weight, top_k=top_k)

    if not x.is_cuda or x.dtype != torch.float16:
        from cutile_forge.kernels.gemm_dequant import unpack_int4_weights_torch

        out = torch.zeros((m, n), device=x.device, dtype=x.dtype)
        for e in range(num_experts):
            mask = topk_idx == e
            if not bool(mask.any()):
                continue
            tok_i, slot_i = torch.where(mask)
            w_e = unpack_int4_weights_torch(
                expert_qweights[e], expert_scales[e], expert_zeros[e], group_size=group_size
            ).to(x.dtype)
            y_e = torch.matmul(x.index_select(0, tok_i).float(), w_e.float()).to(x.dtype)
            out.index_add_(0, tok_i, (y_e * probs[tok_i, slot_i].unsqueeze(1)).to(x.dtype))
        return out

    tm = 16 if m <= 16 else (32 if m <= 32 else 64)
    tn = 128
    tk = 64 if group_size >= 64 else 32
    m_pad_cap = ((m + tm - 1) // tm) * tm

    counts = torch.zeros((num_experts,), device=x.device, dtype=torch.int32)
    x_grouped = torch.zeros((num_experts, m_pad_cap, k), device=x.device, dtype=torch.float16)
    slot_map = torch.empty((m, top_k), device=x.device, dtype=torch.int32)
    probs_c = probs if probs.is_contiguous() else probs.contiguous()
    idx_c = topk_idx if topk_idx.is_contiguous() else topk_idx.contiguous()

    ct.launch(
        torch.cuda.current_stream(),
        (m, top_k, 1),
        cutile_moe_bucket_kernel,
        (x.contiguous(), idx_c, counts, x_grouped, slot_map, m_pad_cap, k, 256),
    )

    y_grouped = torch.empty((num_experts, m_pad_cap, n), device=x.device, dtype=torch.float16)
    ct.launch(
        torch.cuda.current_stream(),
        (ct.cdiv(n, tn), ct.cdiv(m_pad_cap, tm), num_experts),
        moe_grouped_gemm_int4_kernel,
        (
            x_grouped,
            expert_qweights.contiguous(),
            expert_scales.contiguous(),
            expert_zeros.contiguous(),
            y_grouped,
            tm,
            tn,
            tk,
            group_size,
        ),
    )

    out = torch.empty((m, n), device=x.device, dtype=torch.float16)
    tile_n = 256 if n >= 256 else 128
    ct.launch(
        torch.cuda.current_stream(),
        (m, ct.cdiv(n, tile_n), 1),
        cutile_moe_scatter_combine_kernel,
        (y_grouped, idx_c, slot_map, probs_c, out, top_k, tile_n),
    )
    return out


@ct.kernel(occupancy=4)
def moe_indexed_swiglu_splitk_kernel(
    x_ptr,  # (1, K) fp16
    idx_ptr,  # (top_k,) int32
    w1_qw_ptr,  # (E, K // 2, N) uint8
    w1_sc_ptr,  # (E, G, N) fp16
    w1_zr_ptr,  # (E, G, N) fp16
    w3_qw_ptr,  # (E, K // 2, N) uint8
    w3_sc_ptr,  # (E, G, N) fp16
    w3_zr_ptr,  # (E, G, N) fp16
    part_g_ptr,  # (top_k, split_k, tm, N) fp32
    part_u_ptr,  # (top_k, split_k, tm, N) fp32
    num_groups: int,
    split_k: ct.Constant[int],
    tm: ct.Constant[int],
    tn: ct.Constant[int],
    tk: ct.Constant[int],
    gs: ct.Constant[int],
) -> None:
    """Zero-sync M=1 Split-K Gate+Up INT4-dequant GEMM reading routed expert IDs on GPU."""
    pid_n = ct.bid(0)
    k_slot = ct.bid(1)
    sk_id = ct.bid(2)

    e_idx = ct.sum(
        ct.astype(
            ct.load(idx_ptr, index=(k_slot,), shape=(1,), padding_mode=ct.PaddingMode.ZERO),
            ct.int32,
        )
    )
    num_k = ct.num_tiles(x_ptr, axis=1, shape=(tm, tk))
    k_per_split = (num_k + split_k - 1) // split_k
    k_start = sk_id * k_per_split
    k_end = min(num_k, k_start + k_per_split)
    half_tk: ct.Constant[int] = tk // 2

    acc_g = ct.zeros((tm, tn), dtype=ct.float32)
    acc_u = ct.zeros((tm, tn), dtype=ct.float32)

    for k in range(k_start, k_end):
        ta = ct.load(
            x_ptr, index=(0, k), shape=(tm, tk), padding_mode=ct.PaddingMode.ZERO, latency=3
        )
        g_idx = 0 if num_groups == 1 else (k * tk) // gs

        gw = ct.reshape(
            ct.load(
                w1_qw_ptr,
                index=(e_idx, k, pid_n),
                shape=(1, half_tk, tn),
                padding_mode=ct.PaddingMode.ZERO,
                latency=3,
            ),
            (half_tk, tn),
        )
        g_lo = ct.astype(ct.bitwise_and(gw, 0x0F), ct.float16)
        g_hi = ct.astype(ct.bitwise_and(ct.bitwise_rshift(gw, 4), 0x0F), ct.float16)
        g_int = ct.reshape(
            ct.cat(
                (ct.reshape(g_lo, (half_tk, 1, tn)), ct.reshape(g_hi, (half_tk, 1, tn))), axis=1
            ),
            (tk, tn),
        )
        g_sc = ct.reshape(
            ct.load(
                w1_sc_ptr,
                index=(e_idx, g_idx, pid_n),
                shape=(1, 1, tn),
                padding_mode=ct.PaddingMode.ZERO,
                latency=3,
            ),
            (1, tn),
        )
        g_zr = ct.reshape(
            ct.load(
                w1_zr_ptr,
                index=(e_idx, g_idx, pid_n),
                shape=(1, 1, tn),
                padding_mode=ct.PaddingMode.ZERO,
                latency=3,
            ),
            (1, tn),
        )
        acc_g = ct.mma(ta, (g_int - g_zr) * g_sc, acc_g)

        uw = ct.reshape(
            ct.load(
                w3_qw_ptr,
                index=(e_idx, k, pid_n),
                shape=(1, half_tk, tn),
                padding_mode=ct.PaddingMode.ZERO,
                latency=3,
            ),
            (half_tk, tn),
        )
        u_lo = ct.astype(ct.bitwise_and(uw, 0x0F), ct.float16)
        u_hi = ct.astype(ct.bitwise_and(ct.bitwise_rshift(uw, 4), 0x0F), ct.float16)
        u_int = ct.reshape(
            ct.cat(
                (ct.reshape(u_lo, (half_tk, 1, tn)), ct.reshape(u_hi, (half_tk, 1, tn))), axis=1
            ),
            (tk, tn),
        )
        u_sc = ct.reshape(
            ct.load(
                w3_sc_ptr,
                index=(e_idx, g_idx, pid_n),
                shape=(1, 1, tn),
                padding_mode=ct.PaddingMode.ZERO,
                latency=3,
            ),
            (1, tn),
        )
        u_zr = ct.reshape(
            ct.load(
                w3_zr_ptr,
                index=(e_idx, g_idx, pid_n),
                shape=(1, 1, tn),
                padding_mode=ct.PaddingMode.ZERO,
                latency=3,
            ),
            (1, tn),
        )
        acc_u = ct.mma(ta, (u_int - u_zr) * u_sc, acc_u)

    ct.store(part_g_ptr, index=(k_slot, sk_id, 0, pid_n), tile=ct.reshape(acc_g, (1, 1, tm, tn)))
    ct.store(part_u_ptr, index=(k_slot, sk_id, 0, pid_n), tile=ct.reshape(acc_u, (1, 1, tm, tn)))


@ct.kernel(occupancy=4)
def moe_splitk_reduce_swiglu_kernel(
    part_g_ptr,  # (top_k, split_k, tm, N) fp32
    part_u_ptr,  # (top_k, split_k, tm, N) fp32
    inter_ptr,  # (top_k, tm, N) fp16
    split_k: ct.Constant[int],
    tm: ct.Constant[int],
    tn: ct.Constant[int],
) -> None:
    """Reduce Split-K Gate/Up partials across `split_k` and compute SiLU(g) * u in registers."""
    pid_n = ct.bid(0)
    k_slot = ct.bid(1)
    ag = ct.zeros((tm, tn), dtype=ct.float32)
    au = ct.zeros((tm, tn), dtype=ct.float32)
    for sk in range(split_k):
        ag = ag + ct.reshape(
            ct.load(
                part_g_ptr,
                index=(k_slot, sk, 0, pid_n),
                shape=(1, 1, tm, tn),
                padding_mode=ct.PaddingMode.ZERO,
                latency=3,
            ),
            (tm, tn),
        )
        au = au + ct.reshape(
            ct.load(
                part_u_ptr,
                index=(k_slot, sk, 0, pid_n),
                shape=(1, 1, tm, tn),
                padding_mode=ct.PaddingMode.ZERO,
                latency=3,
            ),
            (tm, tn),
        )
    sig = 1.0 / (1.0 + ct.exp(-ag))
    out = ct.astype(ag * sig, ct.float16) * ct.astype(au, ct.float16)
    ct.store(inter_ptr, index=(k_slot, 0, pid_n), tile=ct.reshape(out, (1, tm, tn)))


@ct.kernel(occupancy=4)
def moe_indexed_down_splitk_kernel(
    inter_ptr,  # (top_k, tm, FFN) fp16
    idx_ptr,  # (top_k,) int32
    w2_qw_ptr,  # (E, FFN // 2, D) uint8
    w2_sc_ptr,  # (E, G, D) fp16
    w2_zr_ptr,  # (E, G, D) fp16
    part_ptr,  # (top_k, split_k, tm, D) fp32
    num_groups: int,
    split_k: ct.Constant[int],
    tm: ct.Constant[int],
    tn: ct.Constant[int],
    tk: ct.Constant[int],
    gs: ct.Constant[int],
) -> None:
    """Zero-sync M=1 Split-K Down-proj INT4-dequant GEMM reading routed expert IDs on GPU."""
    pid_n = ct.bid(0)
    k_slot = ct.bid(1)
    sk_id = ct.bid(2)
    e_idx = ct.sum(
        ct.astype(
            ct.load(idx_ptr, index=(k_slot,), shape=(1,), padding_mode=ct.PaddingMode.ZERO),
            ct.int32,
        )
    )
    num_k = ct.num_tiles(inter_ptr, axis=2, shape=(1, tm, tk))
    k_per_split = (num_k + split_k - 1) // split_k
    k_start = sk_id * k_per_split
    k_end = min(num_k, k_start + k_per_split)
    half_tk: ct.Constant[int] = tk // 2
    acc = ct.zeros((tm, tn), dtype=ct.float32)

    for k in range(k_start, k_end):
        ta = ct.reshape(
            ct.load(
                inter_ptr,
                index=(k_slot, 0, k),
                shape=(1, tm, tk),
                padding_mode=ct.PaddingMode.ZERO,
                latency=3,
            ),
            (tm, tk),
        )
        g_idx = 0 if num_groups == 1 else (k * tk) // gs
        qw = ct.reshape(
            ct.load(
                w2_qw_ptr,
                index=(e_idx, k, pid_n),
                shape=(1, half_tk, tn),
                padding_mode=ct.PaddingMode.ZERO,
                latency=3,
            ),
            (half_tk, tn),
        )
        lo = ct.astype(ct.bitwise_and(qw, 0x0F), ct.float16)
        hi = ct.astype(ct.bitwise_and(ct.bitwise_rshift(qw, 4), 0x0F), ct.float16)
        w_int = ct.reshape(
            ct.cat((ct.reshape(lo, (half_tk, 1, tn)), ct.reshape(hi, (half_tk, 1, tn))), axis=1),
            (tk, tn),
        )
        sc = ct.reshape(
            ct.load(
                w2_sc_ptr,
                index=(e_idx, g_idx, pid_n),
                shape=(1, 1, tn),
                padding_mode=ct.PaddingMode.ZERO,
                latency=3,
            ),
            (1, tn),
        )
        zr = ct.reshape(
            ct.load(
                w2_zr_ptr,
                index=(e_idx, g_idx, pid_n),
                shape=(1, 1, tn),
                padding_mode=ct.PaddingMode.ZERO,
                latency=3,
            ),
            (1, tn),
        )
        acc = ct.mma(ta, (w_int - zr) * sc, acc)

    ct.store(part_ptr, index=(k_slot, sk_id, 0, pid_n), tile=ct.reshape(acc, (1, 1, tm, tn)))


@ct.kernel(occupancy=4)
def moe_splitk_reduce_weighted_add_kernel(
    part_ptr,  # (top_k, split_k, tm, D) fp32
    probs_ptr,  # (top_k,) fp32
    res_ptr,  # (1, D) fp16
    out_ptr,  # (1, D) fp16
    top_k: ct.Constant[int],
    split_k: ct.Constant[int],
    tm: ct.Constant[int],
    tn: ct.Constant[int],
    has_residual: ct.Constant[int],
) -> None:
    """Reduce Split-K expert outputs weighted by router `probs` and optionally add `residual`."""
    pid_n = ct.bid(0)
    out_acc = ct.zeros((tm, tn), dtype=ct.float32)
    for k_slot in range(top_k):
        prob = ct.sum(
            ct.astype(
                ct.load(probs_ptr, index=(k_slot,), shape=(1,), padding_mode=ct.PaddingMode.ZERO),
                ct.float32,
            )
        )
        sk_acc = ct.zeros((tm, tn), dtype=ct.float32)
        for sk in range(split_k):
            sk_acc = sk_acc + ct.reshape(
                ct.load(
                    part_ptr,
                    index=(k_slot, sk, 0, pid_n),
                    shape=(1, 1, tm, tn),
                    padding_mode=ct.PaddingMode.ZERO,
                    latency=3,
                ),
                (tm, tn),
            )
        out_acc = out_acc + ct.astype(ct.astype(sk_acc, ct.float16), ct.float32) * prob

    res_fp16 = ct.astype(out_acc, ct.float16)
    if has_residual != 0:
        res_tile = ct.load(
            res_ptr, index=(0, pid_n), shape=(tm, tn), padding_mode=ct.PaddingMode.ZERO, latency=2
        )
        res_fp16 = res_fp16 + res_tile
    ct.store(out_ptr, index=(0, pid_n), tile=res_fp16)


_MOE_SCRATCH_CACHE: dict[tuple[Any, ...], dict[str, torch.Tensor]] = {}


def cutile_moe_indexed_ffn(
    x: torch.Tensor,
    w1_qw_3d: torch.Tensor,
    w1_sc_3d: torch.Tensor,
    w1_zr_3d: torch.Tensor,
    w3_qw_3d: torch.Tensor,
    w3_sc_3d: torch.Tensor,
    w3_zr_3d: torch.Tensor,
    w2_qw_3d: torch.Tensor,
    w2_sc_3d: torch.Tensor,
    w2_zr_3d: torch.Tensor,
    probs: torch.Tensor,
    indices: torch.Tensor,
    residual: torch.Tensor | None = None,
    group_size: int = 128,
) -> torch.Tensor:
    """Execute zero-host-sync M=1 Split-K SwiGLU + weighted Down MoE over 3D packed experts."""
    d_model = x.shape[1]
    ffn_dim = w1_qw_3d.shape[2]
    top_k = int(indices.shape[1])
    tm = 16
    tn_gu = 64 if ffn_dim % 64 == 0 else 32
    tn_dn = 64 if d_model % 64 == 0 else 32
    tk = (
        128
        if (group_size % 128 == 0 and d_model % 128 == 0 and ffn_dim % 128 == 0)
        else (64 if (group_size % 64 == 0 and d_model % 64 == 0 and ffn_dim % 64 == 0) else 32)
    )
    sk_gu = 2 if (d_model // tk >= 2 and d_model >= 2048) else 1
    sk_dn = (
        8
        if (ffn_dim // tk >= 8 and ffn_dim >= 8192)
        else (4 if (ffn_dim // tk >= 4 and ffn_dim >= 2048) else 1)
    )

    key = (str(x.device), top_k, sk_gu, sk_dn, d_model, ffn_dim, tm)
    sc = _MOE_SCRATCH_CACHE.get(key)
    if sc is None:
        sc = {
            "part_g": torch.empty(
                (top_k, sk_gu, tm, ffn_dim), device=x.device, dtype=torch.float32
            ),
            "part_u": torch.empty(
                (top_k, sk_gu, tm, ffn_dim), device=x.device, dtype=torch.float32
            ),
            "inter": torch.zeros((top_k, tm, ffn_dim), device=x.device, dtype=torch.float16),
            "part_dn": torch.empty(
                (top_k, sk_dn, tm, d_model), device=x.device, dtype=torch.float32
            ),
        }
        _MOE_SCRATCH_CACHE[key] = sc

    x_c = x if x.is_contiguous() else x.contiguous()
    idx_c = indices.view(top_k).to(torch.int32).contiguous()
    probs_c = probs.view(top_k).to(torch.float32).contiguous()
    out = torch.empty((1, d_model), device=x.device, dtype=torch.float16)

    ct.launch(
        torch.cuda.current_stream(),
        (ct.cdiv(ffn_dim, tn_gu), top_k, sk_gu),
        moe_indexed_swiglu_splitk_kernel,
        (
            x_c,
            idx_c,
            w1_qw_3d,
            w1_sc_3d,
            w1_zr_3d,
            w3_qw_3d,
            w3_sc_3d,
            w3_zr_3d,
            sc["part_g"],
            sc["part_u"],
            int(w1_sc_3d.shape[1]),
            sk_gu,
            tm,
            tn_gu,
            tk,
            group_size,
        ),
    )
    ct.launch(
        torch.cuda.current_stream(),
        (ct.cdiv(ffn_dim, tn_gu), top_k, 1),
        moe_splitk_reduce_swiglu_kernel,
        (sc["part_g"], sc["part_u"], sc["inter"], sk_gu, tm, tn_gu),
    )
    ct.launch(
        torch.cuda.current_stream(),
        (ct.cdiv(d_model, tn_dn), top_k, sk_dn),
        moe_indexed_down_splitk_kernel,
        (
            sc["inter"],
            idx_c,
            w2_qw_3d,
            w2_sc_3d,
            w2_zr_3d,
            sc["part_dn"],
            int(w2_sc_3d.shape[1]),
            sk_dn,
            tm,
            tn_dn,
            tk,
            group_size,
        ),
    )
    has_res = 1 if residual is not None else 0
    res_c = (
        (residual if residual.is_contiguous() else residual.contiguous())
        if residual is not None
        else x_c
    )
    ct.launch(
        torch.cuda.current_stream(),
        (ct.cdiv(d_model, tn_dn), 1, 1),
        moe_splitk_reduce_weighted_add_kernel,
        (sc["part_dn"], probs_c, res_c, out, top_k, sk_dn, tm, tn_dn, has_res),
    )
    return out
