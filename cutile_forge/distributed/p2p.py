"""Peer-to-peer GPU tensor copy and ring all-reduce prototype primitives."""

from __future__ import annotations

import torch


def p2p_copy(
    src: torch.Tensor,
    dst_device: str | torch.device,
    non_blocking: bool = True,
) -> torch.Tensor:
    """Copy `src` to `dst_device`, falling back to CPU when CUDA is unavailable."""
    target_str = str(dst_device)
    if target_str.startswith("cuda") and not torch.cuda.is_available():
        target = torch.device("cpu")
    else:
        target = torch.device(dst_device)
    if src.device == target:
        return src.clone()
    return src.to(device=target, non_blocking=non_blocking)


def ring_allreduce_sum(tensors: list[torch.Tensor]) -> list[torch.Tensor]:
    """Deterministic ring reduce-scatter + allgather sum across a list of rank tensors."""
    if not tensors:
        return []
    n_ranks = len(tensors)
    if n_ranks == 1:
        return [tensors[0].clone()]

    ref_shape = tensors[0].shape
    for t in tensors[1:]:
        if t.shape != ref_shape:
            raise ValueError(f"All rank tensors must match shape {ref_shape}, got {t.shape}")

    numel = tensors[0].numel()
    if numel == 0:
        return [t.clone() for t in tensors]

    pad = (n_ranks - (numel % n_ranks)) % n_ranks
    chunk_size = (numel + pad) // n_ranks

    bufs: list[list[torch.Tensor]] = []
    for t in tensors:
        flat = t.detach().float().reshape(-1)
        if pad > 0:
            flat = torch.cat([flat, torch.zeros((pad,), dtype=torch.float32, device=flat.device)])
        bufs.append([flat[i * chunk_size : (i + 1) * chunk_size].clone() for i in range(n_ranks)])

    # Ring reduce-scatter: after n_ranks - 1 hops, rank r holds full sum for chunk (r + 1) % n
    for step in range(n_ranks - 1):
        staged_hops = []
        for rank in range(n_ranks):
            send_idx = (rank - step) % n_ranks
            nxt = (rank + 1) % n_ranks
            staged_hops.append((nxt, send_idx, p2p_copy(bufs[rank][send_idx], tensors[nxt].device)))
        for nxt, recv_idx, chunk in staged_hops:
            bufs[nxt][recv_idx] = bufs[nxt][recv_idx] + chunk

    # Ring allgather: propagate each rank's fully reduced chunk around the ring
    for step in range(n_ranks - 1):
        staged_hops = []
        for rank in range(n_ranks):
            send_idx = (rank - step + 1) % n_ranks
            nxt = (rank + 1) % n_ranks
            staged_hops.append((nxt, send_idx, p2p_copy(bufs[rank][send_idx], tensors[nxt].device)))
        for nxt, recv_idx, chunk in staged_hops:
            bufs[nxt][recv_idx] = chunk

    out: list[torch.Tensor] = []
    for rank, orig in enumerate(tensors):
        joined = torch.cat(bufs[rank], dim=0)[:numel]
        out.append(joined.view(ref_shape).to(orig.dtype))
    return out
