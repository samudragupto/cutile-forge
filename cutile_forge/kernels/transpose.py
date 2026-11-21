"""Tiled tensor transpose (`cutile_transpose`) kernel in CuTile."""

from __future__ import annotations

import torch

from cutile_forge.cuda_tile_init import ct


@ct.kernel(occupancy=4)
def transpose_2d_kernel(
    x_ptr,
    y_ptr,
    tm: ct.Constant[int],
    tn: ct.Constant[int],
) -> None:
    """Tiled batched 2D transpose `(B, M, N) -> (B, N, M)`."""
    pid_n = ct.bid(0)
    pid_m = ct.bid(1)
    b_id = ct.bid(2)

    tile = ct.reshape(
        ct.load(
            x_ptr,
            index=(b_id, pid_m, pid_n),
            shape=(1, tm, tn),
            padding_mode=ct.PaddingMode.ZERO,
            latency=3,
        ),
        (tm, tn),
    )
    tr = ct.reshape(ct.transpose(tile), (1, tn, tm))
    ct.store(y_ptr, index=(b_id, pid_n, pid_m), tile=ct.astype(tr, y_ptr.dtype))


def cutile_transpose(
    x: torch.Tensor,
    dim0: int = -2,
    dim1: int = -1,
) -> torch.Tensor:
    """Transpose `dim0` and `dim1` of `x` via CuTile tiled kernel when swapping last two axes."""
    if x.ndim < 2:
        raise ValueError(f"transpose requires tensor with ndim >= 2, got {tuple(x.shape)}")
    d0 = dim0 if dim0 >= 0 else x.ndim + dim0
    d1 = dim1 if dim1 >= 0 else x.ndim + dim1
    if not (0 <= d0 < x.ndim and 0 <= d1 < x.ndim):
        raise ValueError(f"Invalid dimensions ({dim0}, {dim1}) for shape {tuple(x.shape)}")
    if d0 == d1:
        return x.contiguous()

    lo, hi = (d0, d1) if d0 < d1 else (d1, d0)
    if (
        not x.is_cuda
        or (lo, hi) != (x.ndim - 2, x.ndim - 1)
        or x.dtype not in (torch.float16, torch.bfloat16, torch.float32)
    ):
        return x.transpose(d0, d1).contiguous()

    batch_shape = x.shape[:-2]
    m, n = x.shape[-2], x.shape[-1]
    batch = 1
    for dim in batch_shape:
        batch *= dim

    x_3d = x.contiguous().view(batch, m, n)
    y_3d = torch.empty((batch, n, m), device=x.device, dtype=x.dtype)
    if m >= 64 and n >= 64:
        tm, tn = 64, 64
    else:
        tm, tn = 32, 32
    grid = (ct.cdiv(n, tn), ct.cdiv(m, tm), batch)
    ct.launch(torch.cuda.current_stream(), grid, transpose_2d_kernel, (x_3d, y_3d, tm, tn))
    return y_3d.view(*batch_shape, n, m)
