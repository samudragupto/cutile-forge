"""General axis reduction kernels (`cutile_reduce_sum`, `cutile_reduce_max`) in CuTile."""

from __future__ import annotations

import torch

from cutile_forge.cuda_tile_init import ct


@ct.kernel(occupancy=2)
def reduce_sum_kernel(
    x_ptr,
    y_ptr,
    n_cols: int,
    tile_n: ct.Constant[int],
) -> None:
    """Row-wise tiled sum reduction with FP32 accumulation."""
    row = ct.bid(0)
    num_tiles = (n_cols + tile_n - 1) // tile_n
    col_base = ct.arange(tile_n, dtype=ct.int32)
    acc = ct.zeros((1, 1), dtype=ct.float32)

    for t in range(num_tiles):
        tile_x = ct.astype(
            ct.load(x_ptr, index=(row, t), shape=(1, tile_n), padding_mode=ct.PaddingMode.ZERO),
            ct.float32,
        )
        col_idx = ct.reshape(t * tile_n + col_base, (1, tile_n))
        tile_valid = ct.where(col_idx < n_cols, tile_x, 0.0)
        acc = acc + ct.sum(tile_valid, axis=1, keepdims=True)

    ct.store(y_ptr, index=(row, 0), tile=ct.astype(acc, y_ptr.dtype))


@ct.kernel(occupancy=2)
def reduce_max_kernel(
    x_ptr,
    y_ptr,
    n_cols: int,
    tile_n: ct.Constant[int],
) -> None:
    """Row-wise tiled max reduction with FP32 accumulation."""
    row = ct.bid(0)
    num_tiles = (n_cols + tile_n - 1) // tile_n
    col_base = ct.arange(tile_n, dtype=ct.int32)
    acc = ct.full((1, 1), -3.4e38, dtype=ct.float32)

    for t in range(num_tiles):
        tile_x = ct.astype(
            ct.load(x_ptr, index=(row, t), shape=(1, tile_n), padding_mode=ct.PaddingMode.NEG_INF),
            ct.float32,
        )
        col_idx = ct.reshape(t * tile_n + col_base, (1, tile_n))
        tile_valid = ct.where(col_idx < n_cols, tile_x, -3.4e38)
        acc = ct.maximum(acc, ct.max(tile_valid, axis=1, keepdims=True))

    ct.store(y_ptr, index=(row, 0), tile=ct.astype(acc, y_ptr.dtype))


def _normalize_dim(ndim: int, dim: int) -> int:
    if ndim == 0:
        raise ValueError("Reduction requires tensor with ndim >= 1")
    d = dim if dim >= 0 else ndim + dim
    if d < 0 or d >= ndim:
        raise ValueError(f"Dimension out of range (expected [-{ndim}, {ndim - 1}], got {dim})")
    return d


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

    tile_n = 1024 if n_cols > 512 else 256
    ct.launch(torch.cuda.current_stream(), (rows, 1, 1), kernel_fn, (x_2d, y_2d, n_cols, tile_n))

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
