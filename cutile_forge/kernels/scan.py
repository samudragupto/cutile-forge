"""Prefix sum / inclusive scan (`cutile_cumsum`) kernel in CuTile."""

from __future__ import annotations

import torch

from cutile_forge.cuda_tile_init import ct


@ct.kernel(occupancy=2)
def cumsum_kernel(
    x_ptr,
    y_ptr,
    n_cols: int,
    tile_n: ct.Constant[int],
) -> None:
    """Row-wise multi-tile inclusive prefix sum with FP32 carry accumulation."""
    row = ct.bid(0)
    num_tiles = (n_cols + tile_n - 1) // tile_n
    col_base = ct.arange(tile_n, dtype=ct.int32)
    prefix = ct.zeros((1, 1), dtype=ct.float32)

    for t in range(num_tiles):
        tile_x = ct.astype(
            ct.load(x_ptr, index=(row, t), shape=(1, tile_n), padding_mode=ct.PaddingMode.ZERO),
            ct.float32,
        )
        col_idx = ct.reshape(t * tile_n + col_base, (1, tile_n))
        tile_valid = ct.where(col_idx < n_cols, tile_x, 0.0)
        tile_scan = ct.cumsum(tile_valid, axis=1) + prefix
        ct.store(y_ptr, index=(row, t), tile=ct.astype(tile_scan, y_ptr.dtype))
        prefix = prefix + ct.sum(tile_valid, axis=1, keepdims=True)


def cutile_cumsum(x: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """Compute inclusive cumulative sum along `dim` using CuTile (FP16, BF16, FP32)."""
    if x.ndim == 0:
        raise ValueError("cumsum requires tensor with ndim >= 1")
    d = dim if dim >= 0 else x.ndim + dim
    if d < 0 or d >= x.ndim:
        raise ValueError(f"Dimension out of range (expected [-{x.ndim}, {x.ndim - 1}], got {dim})")

    if not x.is_cuda or x.dtype not in (torch.float16, torch.bfloat16, torch.float32):
        return torch.cumsum(x.float(), dim=d).to(x.dtype)

    moved = x.movedim(d, -1).contiguous()
    moved_shape = moved.shape
    n_cols = moved_shape[-1]
    x_2d = moved.view(-1, n_cols)
    rows = x_2d.shape[0]
    y_2d = torch.empty_like(x_2d)

    tile_n = 1024 if n_cols > 512 else 256
    ct.launch(
        torch.cuda.current_stream(), (rows, 1, 1), cumsum_kernel, (x_2d, y_2d, n_cols, tile_n)
    )
    return y_2d.view(moved_shape).movedim(-1, d).contiguous()
