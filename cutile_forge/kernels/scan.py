"""Prefix sum / inclusive scan (`cutile_cumsum`) kernel in CuTile."""

from __future__ import annotations

import torch

from cutile_forge.cuda_tile_init import ct


@ct.kernel(occupancy=4)
def cumsum_kernel(
    x_ptr,
    y_ptr,
    n_cols: int,
    tile_m: ct.Constant[int],
    tile_n: ct.Constant[int],
) -> None:
    """Multi-row 2D tiled inclusive prefix sum with FP32 carry accumulation."""
    row_block = ct.bid(0)
    num_tiles = (n_cols + tile_n - 1) // tile_n
    col_base = ct.arange(tile_n, dtype=ct.int32)
    prefix = ct.zeros((tile_m, 1), dtype=ct.float32)

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
        tile_scan = ct.cumsum(tile_valid, axis=1) + prefix
        ct.store(y_ptr, index=(row_block, t), tile=ct.astype(tile_scan, y_ptr.dtype))
        prefix = prefix + ct.sum(tile_valid, axis=1, keepdims=True)


def _select_scan_tiles(rows: int, n_cols: int) -> tuple[int, int]:
    tile_n = 1024 if n_cols >= 512 else 256
    if tile_n == 1024:
        tile_m = 8 if rows >= 1024 else (4 if rows >= 128 else 1)
    else:
        tile_m = 16 if rows >= 1024 else (8 if rows >= 128 else 1)
    return tile_m, tile_n


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

    tile_m, tile_n = _select_scan_tiles(rows, n_cols)
    grid = (ct.cdiv(rows, tile_m), 1, 1)
    ct.launch(
        torch.cuda.current_stream(), grid, cumsum_kernel, (x_2d, y_2d, n_cols, tile_m, tile_n)
    )
    return y_2d.view(moved_shape).movedim(-1, d).contiguous()
