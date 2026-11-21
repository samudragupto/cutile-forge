"""General axis reduction kernels (`cutile_reduce_sum`, `cutile_reduce_max`) in CuTile."""

from __future__ import annotations

import torch

from cutile_forge.cuda_tile_init import ct


@ct.kernel(occupancy=4)
def reduce_sum_kernel(
    x_ptr,
    y_ptr,
    n_cols: int,
    tile_m: ct.Constant[int],
    tile_n: ct.Constant[int],
) -> None:
    """Multi-row 2D tiled sum reduction with FP32 accumulation."""
    row_block = ct.bid(0)
    num_tiles = (n_cols + tile_n - 1) // tile_n
    col_base = ct.arange(tile_n, dtype=ct.int32)
    acc = ct.zeros((tile_m, 1), dtype=ct.float32)

    for t in range(num_tiles):
        tile_x = ct.astype(
            ct.load(
                x_ptr,
                index=(row_block, t),
                shape=(tile_m, tile_n),
                padding_mode=ct.PaddingMode.ZERO,
                latency=3,
            ),
            ct.float32,
        )
        col_idx = ct.broadcast_to(ct.reshape(t * tile_n + col_base, (1, tile_n)), (tile_m, tile_n))
        tile_valid = ct.where(col_idx < n_cols, tile_x, 0.0)
        acc = acc + ct.sum(tile_valid, axis=1, keepdims=True)

    ct.store(y_ptr, index=(row_block, 0), tile=ct.astype(acc, y_ptr.dtype))


@ct.kernel(occupancy=4)
def reduce_sum_splitk_kernel(
    x_ptr,
    part_ptr,
    n_cols: int,
    split_k: ct.Constant[int],
    tile_m: ct.Constant[int],
    tile_n: ct.Constant[int],
) -> None:
    """Split-K coalesced sum reduction for low-row, wide-column tensors."""
    row_block = ct.bid(0)
    sk = ct.bid(1)
    num_tiles = (n_cols + tile_n - 1) // tile_n
    tiles_per_split = (num_tiles + split_k - 1) // split_k
    t_start = sk * tiles_per_split
    t_end = min(num_tiles, t_start + tiles_per_split)
    col_base = ct.arange(tile_n, dtype=ct.int32)
    acc = ct.zeros((tile_m, 1), dtype=ct.float32)

    for t in range(t_start, t_end):
        tile_x = ct.astype(
            ct.load(
                x_ptr,
                index=(row_block, t),
                shape=(tile_m, tile_n),
                padding_mode=ct.PaddingMode.ZERO,
                latency=4,
            ),
            ct.float32,
        )
        col_idx = ct.broadcast_to(ct.reshape(t * tile_n + col_base, (1, tile_n)), (tile_m, tile_n))
        acc = acc + ct.sum(ct.where(col_idx < n_cols, tile_x, 0.0), axis=1, keepdims=True)

    ct.store(part_ptr, index=(row_block, sk), tile=acc)


@ct.kernel(occupancy=4)
def reduce_max_splitk_kernel(
    x_ptr,
    part_ptr,
    n_cols: int,
    split_k: ct.Constant[int],
    tile_m: ct.Constant[int],
    tile_n: ct.Constant[int],
) -> None:
    """Split-K coalesced max reduction for low-row, wide-column tensors."""
    row_block = ct.bid(0)
    sk = ct.bid(1)
    num_tiles = (n_cols + tile_n - 1) // tile_n
    tiles_per_split = (num_tiles + split_k - 1) // split_k
    t_start = sk * tiles_per_split
    t_end = min(num_tiles, t_start + tiles_per_split)
    col_base = ct.arange(tile_n, dtype=ct.int32)
    acc = ct.full((tile_m, 1), -3.4e38, dtype=ct.float32)

    for t in range(t_start, t_end):
        tile_x = ct.astype(
            ct.load(
                x_ptr,
                index=(row_block, t),
                shape=(tile_m, tile_n),
                padding_mode=ct.PaddingMode.NEG_INF,
                latency=4,
            ),
            ct.float32,
        )
        col_idx = ct.broadcast_to(ct.reshape(t * tile_n + col_base, (1, tile_n)), (tile_m, tile_n))
        acc = ct.maximum(
            acc, ct.max(ct.where(col_idx < n_cols, tile_x, -3.4e38), axis=1, keepdims=True)
        )

    ct.store(part_ptr, index=(row_block, sk), tile=acc)


@ct.kernel(occupancy=4)
def reduce_max_kernel(
    x_ptr,
    y_ptr,
    n_cols: int,
    tile_m: ct.Constant[int],
    tile_n: ct.Constant[int],
) -> None:
    """Multi-row 2D tiled max reduction with FP32 accumulation."""
    row_block = ct.bid(0)
    num_tiles = (n_cols + tile_n - 1) // tile_n
    col_base = ct.arange(tile_n, dtype=ct.int32)
    acc = ct.full((tile_m, 1), -3.4e38, dtype=ct.float32)

    for t in range(num_tiles):
        tile_x = ct.astype(
            ct.load(
                x_ptr,
                index=(row_block, t),
                shape=(tile_m, tile_n),
                padding_mode=ct.PaddingMode.NEG_INF,
                latency=3,
            ),
            ct.float32,
        )
        col_idx = ct.broadcast_to(ct.reshape(t * tile_n + col_base, (1, tile_n)), (tile_m, tile_n))
        tile_valid = ct.where(col_idx < n_cols, tile_x, -3.4e38)
        acc = ct.maximum(acc, ct.max(tile_valid, axis=1, keepdims=True))

    ct.store(y_ptr, index=(row_block, 0), tile=ct.astype(acc, y_ptr.dtype))


def _normalize_dim(ndim: int, dim: int) -> int:
    if ndim == 0:
        raise ValueError("Reduction requires tensor with ndim >= 1")
    d = dim if dim >= 0 else ndim + dim
    if d < 0 or d >= ndim:
        raise ValueError(f"Dimension out of range (expected [-{ndim}, {ndim - 1}], got {dim})")
    return d


def _select_reduction_tiles(rows: int, n_cols: int) -> tuple[int, int]:
    """Select 2D `(tile_m, tile_n)` keeping ~16–32 KB per CTA and >= 170 CTAs when possible."""
    tile_n = 1024 if n_cols >= 512 else 256
    if tile_n == 1024:
        tile_m = 16 if rows >= 4096 else (8 if rows >= 256 else 1)
    else:
        tile_m = 32 if rows >= 4096 else (16 if rows >= 256 else (8 if rows >= 32 else 1))
    return tile_m, tile_n


def _launch_row_reduction(
    x: torch.Tensor,
    dim: int,
    keepdim: bool,
    kernel_fn: object,
) -> torch.Tensor:
    d = _normalize_dim(x.ndim, dim)
    moved = x.movedim(d, -1).contiguous()
    lead_shape = moved.shape[:-1]
    n_cols = moved.shape[-1]
    x_2d = moved.view(-1, n_cols)
    rows = x_2d.shape[0]
    y_2d = torch.empty((rows, 1), device=x.device, dtype=x.dtype)

    if rows < 170 and n_cols >= 2048:
        split_k = 8
        tile_m = 4 if rows >= 16 else 1
        tile_n = 512
        part = torch.empty((rows, split_k), device=x.device, dtype=torch.float32)
        split_fn = (
            reduce_sum_splitk_kernel if kernel_fn is reduce_sum_kernel else reduce_max_splitk_kernel
        )
        ct.launch(
            torch.cuda.current_stream(),
            (ct.cdiv(rows, tile_m), split_k, 1),
            split_fn,
            (x_2d, part, n_cols, split_k, tile_m, tile_n),
        )
        reduced = (
            part.sum(dim=1, keepdim=True)
            if kernel_fn is reduce_sum_kernel
            else part.max(dim=1, keepdim=True).values
        )
        y_2d = reduced.to(x.dtype)
    else:
        tile_m, tile_n = _select_reduction_tiles(rows, n_cols)
        grid = (ct.cdiv(rows, tile_m), 1, 1)
        ct.launch(
            torch.cuda.current_stream(),
            grid,
            kernel_fn,
            (x_2d, y_2d, n_cols, tile_m, tile_n),
        )

    if keepdim:
        out_shape = list(x.shape)
        out_shape[d] = 1
        out_moved_shape = (*lead_shape, 1)
        return y_2d.view(out_moved_shape).movedim(-1, d).reshape(out_shape)
    return y_2d.view(lead_shape)


def cutile_reduce_sum(
    x: torch.Tensor,
    dim: int = -1,
    keepdim: bool = False,
) -> torch.Tensor:
    """Compute sum reduction along `dim` using CuTile (FP16, BF16, FP32)."""
    d = _normalize_dim(x.ndim, dim)
    if not x.is_cuda or x.dtype not in (torch.float16, torch.bfloat16, torch.float32):
        return x.float().sum(dim=d, keepdim=keepdim).to(x.dtype)
    return _launch_row_reduction(x, d, keepdim, reduce_sum_kernel)


def cutile_reduce_max(
    x: torch.Tensor,
    dim: int = -1,
    keepdim: bool = False,
) -> torch.Tensor:
    """Compute max reduction along `dim` using CuTile (FP16, BF16, FP32)."""
    d = _normalize_dim(x.ndim, dim)
    if not x.is_cuda or x.dtype not in (torch.float16, torch.bfloat16, torch.float32):
        return x.max(dim=d, keepdim=keepdim).values
    return _launch_row_reduction(x, d, keepdim, reduce_max_kernel)
