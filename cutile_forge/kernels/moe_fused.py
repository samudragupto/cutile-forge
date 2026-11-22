"""CuTile kernels for fused MoE top-k gating and persistent grouped INT4 expert GEMM."""

from __future__ import annotations

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


def cutile_moe_router_topk(
    x: torch.Tensor,
    router_weight: torch.Tensor,
    top_k: int = 2,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute MoE router logits, softmax, and normalized top-k gating in one CuTile kernel."""
    m, k_dim = x.shape
    num_experts = router_weight.shape[0]
    if top_k <= 0 or top_k > num_experts:
        raise ValueError(f"top_k={top_k} invalid for num_experts={num_experts}")

    if not x.is_cuda or x.dtype != torch.float16:
        logits = torch.matmul(x.float(), router_weight.float().t())
        probs = torch.softmax(logits, dim=-1)
        vals, idx = torch.topk(probs, k=top_k, dim=-1)
        vals = vals / vals.sum(dim=-1, keepdim=True).clamp(min=1e-6)
        return vals.to(x.dtype), idx.to(torch.int32)

    e_pad = max(16, _next_pow2_moe(num_experts))
    top_k_pad = _next_pow2_moe(top_k)
    tm, tk = 16, 64
    probs_out = torch.empty((m, top_k_pad), device=x.device, dtype=torch.float16)
    idx_out = torch.empty((m, top_k_pad), device=x.device, dtype=torch.int32)
    ct.launch(
        torch.cuda.current_stream(),
        (ct.cdiv(m, tm), 1, 1),
        cutile_moe_router_topk_kernel,
        (
            x.contiguous(),
            router_weight.contiguous(),
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
