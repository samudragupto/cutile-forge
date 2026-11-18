"""CuTile kernels for fused MoE top-k gating and grouped INT4 expert GEMM (Mixtral-8x7B)."""

from __future__ import annotations

import torch

from cutile_forge.cuda_tile_init import ct
from cutile_forge.kernels.gemm_dequant import fused_gemm_dequant


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
    """Grouped INT4 expert GEMM without materializing padded HBM permutation buffers.

    Args:
        x: Input activations `(M, K)` float16.
        expert_qweights: `(E, K // 2, N)` uint8 packed INT4 weights.
        expert_scales: `(E, K // group_size, N)` float16.
        expert_zeros: `(E, K // group_size, N)` float16.
        router_probs: `(M, top_k)` float16 normalized gating weights.
        topk_indices: `(M, top_k)` int64 expert indices.
        group_size: Quantization group size (default 128).
    """
    m, _ = x.shape
    num_experts, _, n = expert_qweights.shape
    out = torch.zeros((m, n), device=x.device, dtype=torch.float16)

    for e in range(num_experts):
        token_mask = topk_indices == e
        if not bool(token_mask.any()):
            continue
        tok_idx, k_slot = torch.where(token_mask)
        x_e = x.index_select(0, tok_idx)
        w_e = router_probs[tok_idx, k_slot].unsqueeze(1)
        if x.is_cuda:
            y_e = fused_gemm_dequant(
                x_e,
                expert_qweights[e],
                expert_scales[e],
                expert_zeros[e],
                group_size=group_size,
            )
        else:
            from cutile_forge.kernels.gemm_dequant import unpack_int4_weights_torch

            w_fp16 = unpack_int4_weights_torch(
                expert_qweights[e], expert_scales[e], expert_zeros[e], group_size=group_size
            )
            y_e = torch.matmul(x_e, w_fp16)
        out.index_add_(0, tok_idx, (y_e * w_e).to(out.dtype))

    return out
