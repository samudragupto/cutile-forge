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
    ct.store(probs_ptr, index=(row, 0), tile=ct.astype(probs, ct.float16))


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
        probs = torch.empty((m, num_experts), device=logits.device, dtype=torch.float16)
        ct.launch(
            torch.cuda.current_stream(),
            (m, 1, 1),
            moe_softmax_kernel,
            (logits.to(torch.float16).contiguous(), probs, num_experts),
        )
    else:
        probs = torch.softmax(logits.float(), dim=-1).to(torch.float16)

    topk_vals, topk_idx = torch.topk(probs.float(), k=top_k, dim=-1)
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
