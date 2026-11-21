"""Native CuTile embedding lookup, fused LM head + argmax/top-k, and elementwise kernels."""

from __future__ import annotations

import torch

from cutile_forge.cuda_tile_init import ct
from cutile_forge.kernels.indexing import gather_rows_kernel
from cutile_forge.kernels.rmsnorm import cutile_rmsnorm


def _next_pow2(n: int) -> int:
    p = 1
    while p < n:
        p *= 2
    return p


@ct.kernel(occupancy=4)
def ew_mul_kernel(a_ptr, b_ptr, c_ptr, tile_n: ct.Constant[int]) -> None:
    """Elementwise multiply `c = a * b`."""
    pid = ct.bid(0)
    a = ct.load(a_ptr, index=(pid,), shape=(tile_n,), padding_mode=ct.PaddingMode.ZERO, latency=4)
    b = ct.load(b_ptr, index=(pid,), shape=(tile_n,), padding_mode=ct.PaddingMode.ZERO, latency=4)
    ct.store(c_ptr, index=(pid,), tile=ct.astype(a * b, c_ptr.dtype))


@ct.kernel(occupancy=4)
def ew_add_kernel(a_ptr, b_ptr, c_ptr, tile_n: ct.Constant[int]) -> None:
    """Elementwise add `c = a + b`."""
    pid = ct.bid(0)
    a = ct.load(a_ptr, index=(pid,), shape=(tile_n,), padding_mode=ct.PaddingMode.ZERO, latency=4)
    b = ct.load(b_ptr, index=(pid,), shape=(tile_n,), padding_mode=ct.PaddingMode.ZERO, latency=4)
    ct.store(c_ptr, index=(pid,), tile=ct.astype(a + b, c_ptr.dtype))


@ct.kernel(occupancy=4)
def col_permute_kernel(
    x_ptr, perm_ptr, y_ptr, tile_m: ct.Constant[int], tile_d: ct.Constant[int]
) -> None:
    """Gather columns `y[m, d] = x[m, perm[d]]` in CuTile for GPTQ desc_act."""
    pid_m = ct.bid(0)
    pid_d = ct.bid(1)
    row_ids = ct.reshape(pid_m * tile_m + ct.arange(tile_m, dtype=ct.int32), (tile_m, 1))
    col_ids = ct.reshape(
        ct.astype(
            ct.load(perm_ptr, index=(pid_d,), shape=(tile_d,), padding_mode=ct.PaddingMode.ZERO),
            ct.int32,
        ),
        (1, tile_d),
    )
    out = ct.gather(x_ptr, (row_ids, col_ids), padding_value=0, latency=3)
    ct.store(y_ptr, index=(pid_m, pid_d), tile=out)


@ct.kernel(occupancy=2)
def lm_head_tile_argmax_kernel(
    h_ptr,
    w_ptr,
    part_val_ptr,
    part_idx_ptr,
    v_dim: int,
    tm: ct.Constant[int],
    tv: ct.Constant[int],
    tk: ct.Constant[int],
) -> None:
    """Tiled LM head `h @ w.T` reducing local argmax in registers (zero [M, V] HBM write)."""
    pid_v = ct.bid(0)
    pid_m = ct.bid(1)
    num_k = ct.num_tiles(h_ptr, axis=1, shape=(tm, tk))
    acc = ct.zeros((tm, tv), dtype=ct.float32)

    for k in range(num_k):
        th = ct.load(
            h_ptr, index=(pid_m, k), shape=(tm, tk), padding_mode=ct.PaddingMode.ZERO, latency=3
        )
        tw = ct.load(
            w_ptr, index=(pid_v, k), shape=(tv, tk), padding_mode=ct.PaddingMode.ZERO, latency=3
        )
        acc = ct.mma(th, ct.transpose(tw), acc)

    v_ids = ct.reshape(pid_v * tv + ct.arange(tv, dtype=ct.int32), (1, tv))
    masked = ct.where(v_ids < v_dim, acc, -1.0e9)
    best_val = ct.max(masked, axis=1)
    best_local = ct.astype(ct.argmax(masked, axis=1), ct.int32)
    best_global = pid_v * tv + best_local

    ct.store(part_val_ptr, index=(pid_m, pid_v), tile=ct.reshape(best_val, (tm, 1)))
    ct.store(part_idx_ptr, index=(pid_m, pid_v), tile=ct.reshape(best_global, (tm, 1)))


@ct.kernel(occupancy=4)
def lm_head_reduce_argmax_kernel(
    part_val_ptr,
    part_idx_ptr,
    out_val_ptr,
    out_idx_ptr,
    num_vt: int,
    pad_vt: ct.Constant[int],
) -> None:
    """Second-stage reduction over per-vocab-tile winners -> exact global argmax token."""
    row = ct.bid(0)
    v_raw = ct.load(
        part_val_ptr, index=(row, 0), shape=(1, pad_vt), padding_mode=ct.PaddingMode.NEG_INF
    )
    i_raw = ct.astype(
        ct.load(part_idx_ptr, index=(row, 0), shape=(1, pad_vt), padding_mode=ct.PaddingMode.ZERO),
        ct.int32,
    )
    t_ids = ct.reshape(ct.arange(pad_vt, dtype=ct.int32), (1, pad_vt))
    vals = ct.where(t_ids < num_vt, v_raw, -1.0e9)
    best_tile = ct.reshape(ct.astype(ct.argmax(vals, axis=1), ct.int32), (1, 1))
    best_val = ct.max(vals, axis=1)
    mask = t_ids == best_tile
    picked_idx = ct.sum(ct.where(mask, i_raw, 0), axis=1)

    ct.store(out_val_ptr, index=(row, 0), tile=ct.reshape(best_val, (1, 1)))
    ct.store(out_idx_ptr, index=(row, 0), tile=ct.reshape(picked_idx, (1, 1)))


@ct.kernel(occupancy=2)
def lm_head_tile_logits_kernel(
    h_ptr,
    w_ptr,
    logits_ptr,
    tm: ct.Constant[int],
    tv: ct.Constant[int],
    tk: ct.Constant[int],
) -> None:
    """Materialize `(M, V)` FP32 logits tile-by-tile when full top-k > 1 is requested."""
    pid_v = ct.bid(0)
    pid_m = ct.bid(1)
    num_k = ct.num_tiles(h_ptr, axis=1, shape=(tm, tk))
    acc = ct.zeros((tm, tv), dtype=ct.float32)
    for k in range(num_k):
        th = ct.load(
            h_ptr, index=(pid_m, k), shape=(tm, tk), padding_mode=ct.PaddingMode.ZERO, latency=3
        )
        tw = ct.load(
            w_ptr, index=(pid_v, k), shape=(tv, tk), padding_mode=ct.PaddingMode.ZERO, latency=3
        )
        acc = ct.mma(th, ct.transpose(tw), acc)
    ct.store(logits_ptr, index=(pid_m, pid_v), tile=acc)


@ct.kernel(occupancy=2)
def lm_head_row_topk_kernel(
    logits_ptr,
    out_val_ptr,
    out_idx_ptr,
    v_dim: int,
    top_k: ct.Constant[int],
    tv: ct.Constant[int],
) -> None:
    """Tiled multi-pass top-k selection per row in CuTile (no `aten.topk`)."""
    row = ct.bid(0)
    num_vt = (v_dim + tv - 1) // tv
    col_base = ct.arange(tv, dtype=ct.int32)
    picked_val = ct.full((top_k,), -1.0e9, dtype=ct.float32)
    picked_idx = ct.zeros((top_k,), dtype=ct.int32)
    k_ar = ct.arange(top_k, dtype=ct.int32)

    # First pass tracks running max across tiles for k-th winner iteratively
    for ki in range(top_k):
        best_v = ct.full((1,), -1.0e9, dtype=ct.float32)
        best_i = ct.full((1,), 0, dtype=ct.int32)
        for vt in range(num_vt):
            chunk = ct.load(
                logits_ptr, index=(row, vt), shape=(1, tv), padding_mode=ct.PaddingMode.NEG_INF
            )
            cids = ct.reshape(vt * tv + col_base, (1, tv))
            valid = cids < v_dim
            # Mask out any already-picked indices from earlier passes
            for prev in range(ki):
                p_prev = ct.sum(ct.where(k_ar == prev, picked_idx, 0))
                valid = valid & (cids != p_prev)
            masked = ct.where(valid, chunk, -1.0e9)
            cand_v = ct.max(masked, axis=1)
            cand_i = vt * tv + ct.astype(ct.argmax(masked, axis=1), ct.int32)
            take = cand_v > best_v
            best_v = ct.where(take, cand_v, best_v)
            best_i = ct.where(take, cand_i, best_i)
        picked_val = ct.where(k_ar == ki, best_v, picked_val)
        picked_idx = ct.where(k_ar == ki, best_i, picked_idx)

    ct.store(out_val_ptr, index=(row, 0), tile=ct.reshape(picked_val, (1, top_k)))
    ct.store(out_idx_ptr, index=(row, 0), tile=ct.reshape(picked_idx, (1, top_k)))


def cutile_ew_mul(
    a: torch.Tensor, b: torch.Tensor, out: torch.Tensor | None = None
) -> torch.Tensor:
    """Elementwise product `a * b` executed natively in CuTile."""
    res = out if out is not None else torch.empty_like(a)
    if not a.is_cuda or a.dtype not in (torch.float16, torch.bfloat16, torch.float32):
        res.copy_(a * b)
        return res
    a_1d = a.view(-1) if a.is_contiguous() else a.contiguous().view(-1)
    b_1d = b.view(-1) if b.is_contiguous() else b.contiguous().view(-1)
    c_1d = res.view(-1)
    tile_n = 4096 if a_1d.numel() >= 16384 else 1024
    ct.launch(
        torch.cuda.current_stream(),
        (ct.cdiv(a_1d.numel(), tile_n), 1, 1),
        ew_mul_kernel,
        (a_1d, b_1d, c_1d, tile_n),
    )
    return res


def cutile_ew_add(
    a: torch.Tensor, b: torch.Tensor, out: torch.Tensor | None = None
) -> torch.Tensor:
    """Elementwise sum `a + b` executed natively in CuTile."""
    res = out if out is not None else torch.empty_like(a)
    if not a.is_cuda or a.dtype not in (torch.float16, torch.bfloat16, torch.float32):
        res.copy_(a + b)
        return res
    a_1d = a.view(-1) if a.is_contiguous() else a.contiguous().view(-1)
    b_1d = b.view(-1) if b.is_contiguous() else b.contiguous().view(-1)
    c_1d = res.view(-1)
    tile_n = 4096 if a_1d.numel() >= 16384 else 1024
    ct.launch(
        torch.cuda.current_stream(),
        (ct.cdiv(a_1d.numel(), tile_n), 1, 1),
        ew_add_kernel,
        (a_1d, b_1d, c_1d, tile_n),
    )
    return res


def cutile_col_permute(
    x: torch.Tensor, perm: torch.Tensor, out: torch.Tensor | None = None
) -> torch.Tensor:
    """Permute columns `out[m, d] = x[m, perm[d]]` natively in CuTile."""
    m, d = x.shape
    res = out if out is not None else torch.empty_like(x)
    if not x.is_cuda:
        res.copy_(x[:, perm])
        return res
    tile_m = 4 if m >= 4 else 1
    tile_d = 256 if d >= 256 else 128
    ct.launch(
        torch.cuda.current_stream(),
        (ct.cdiv(m, tile_m), ct.cdiv(d, tile_d), 1),
        col_permute_kernel,
        (
            x.contiguous(),
            perm.to(dtype=torch.int32).contiguous(),
            res,
            tile_m,
            tile_d,
        ),
    )
    return res


def cutile_embedding_lookup(
    token_ids: torch.Tensor,
    embed_table: torch.Tensor,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    """Native CuTile embedding gather (`zero F.embedding`), writing to `out` when provided."""
    flat_idx = token_ids.reshape(-1)
    d_dim = embed_table.shape[1]
    out_shape = (*token_ids.shape, d_dim)
    res = (
        out.view(-1, d_dim)
        if out is not None
        else torch.empty(
            (flat_idx.numel(), d_dim), device=embed_table.device, dtype=embed_table.dtype
        )
    )

    if not embed_table.is_cuda or embed_table.dtype not in (
        torch.float16,
        torch.bfloat16,
        torch.float32,
    ):
        res.copy_(
            embed_table.index_select(0, flat_idx.to(device=embed_table.device, dtype=torch.long))
        )
        return res.view(out_shape)

    idx_c = flat_idx if flat_idx.is_contiguous() else flat_idx.contiguous()
    table_c = embed_table if embed_table.is_contiguous() else embed_table.contiguous()
    if d_dim >= 512:
        tile_m = 4 if idx_c.numel() >= 128 else 1
        tile_d = 512
    elif d_dim >= 128:
        tile_m, tile_d = (4 if idx_c.numel() < 16 else 16), 128
    else:
        tile_m, tile_d = (4 if idx_c.numel() < 16 else 32), 64

    grid = (ct.cdiv(idx_c.numel(), tile_m), ct.cdiv(d_dim, tile_d), 1)
    ct.launch(
        torch.cuda.current_stream(),
        grid,
        gather_rows_kernel,
        (table_c, idx_c, res, tile_m, tile_d),
    )
    return res.view(out_shape)


def cutile_lm_head_fused(
    hidden_states: torch.Tensor,
    norm_weight: torch.Tensor,
    lm_head_weight: torch.Tensor,
    eps: float = 1e-5,
    top_k: int = 1,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Fuse final RMSNorm -> LM head matmul -> in-register argmax / top-k (no aten.mm/topk)."""
    orig_prefix = hidden_states.shape[:-1]
    d_model = hidden_states.shape[-1]
    v_dim = lm_head_weight.shape[0]
    m = max(1, hidden_states.numel() // d_model)

    if not hidden_states.is_cuda or hidden_states.dtype != torch.float16:
        xf = hidden_states.float().view(m, d_model)
        normed = (xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + eps)) * norm_weight.float()
        logits = torch.matmul(normed, lm_head_weight.float().t())
        vals, idx = torch.topk(logits, k=top_k, dim=-1)
        return vals.to(hidden_states.dtype).view(*orig_prefix, top_k), idx.view(*orig_prefix, top_k)

    h_norm = cutile_rmsnorm(hidden_states.view(m, d_model), norm_weight, eps=eps)
    w_c = lm_head_weight if lm_head_weight.is_contiguous() else lm_head_weight.contiguous()

    if top_k == 1:
        tm, tv, tk = (16 if m > 1 else 1), 256, 64
        num_vt = ct.cdiv(v_dim, tv)
        pad_vt = _next_pow2(num_vt)
        part_val = torch.empty((m, num_vt), device=hidden_states.device, dtype=torch.float32)
        part_idx = torch.empty((m, num_vt), device=hidden_states.device, dtype=torch.int32)
        ct.launch(
            torch.cuda.current_stream(),
            (num_vt, ct.cdiv(m, tm), 1),
            lm_head_tile_argmax_kernel,
            (h_norm, w_c, part_val, part_idx, v_dim, tm, tv, tk),
        )
        out_val = torch.empty((m, 1), device=hidden_states.device, dtype=torch.float32)
        out_idx = torch.empty((m, 1), device=hidden_states.device, dtype=torch.int32)
        ct.launch(
            torch.cuda.current_stream(),
            (m, 1, 1),
            lm_head_reduce_argmax_kernel,
            (part_val, part_idx, out_val, out_idx, num_vt, pad_vt),
        )
        return (
            out_val.to(hidden_states.dtype).view(*orig_prefix, 1),
            out_idx.to(torch.long).view(*orig_prefix, 1),
        )

    # General top_k > 1: tile logits -> CuTile multi-pass top_k kernel (zero aten.topk)
    tm_l, tv_l, tk_l = (16 if m > 1 else 1), 256, 64
    logits = torch.empty((m, v_dim), device=hidden_states.device, dtype=torch.float32)
    ct.launch(
        torch.cuda.current_stream(),
        (ct.cdiv(v_dim, tv_l), ct.cdiv(m, tm_l), 1),
        lm_head_tile_logits_kernel,
        (h_norm, w_c, logits, tm_l, tv_l, tk_l),
    )
    k_pad = _next_pow2(top_k)
    out_val_k = torch.empty((m, k_pad), device=hidden_states.device, dtype=torch.float32)
    out_idx_k = torch.empty((m, k_pad), device=hidden_states.device, dtype=torch.int32)
    ct.launch(
        torch.cuda.current_stream(),
        (m, 1, 1),
        lm_head_row_topk_kernel,
        (logits, out_val_k, out_idx_k, v_dim, k_pad, 1024),
    )
    return (
        out_val_k[:, :top_k].to(hidden_states.dtype).view(*orig_prefix, top_k),
        out_idx_k[:, :top_k].to(torch.long).view(*orig_prefix, top_k),
    )
