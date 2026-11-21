"""L2-swizzled, shape-aware Batched Matrix Multiplication (`cutile_bmm`) kernel in CuTile."""

from __future__ import annotations

import torch

from cutile_forge.cuda_tile_init import ct


@ct.kernel(occupancy=2)
def bmm_kernel(
    a_ptr,
    b_ptr,
    c_ptr,
    tm: ct.Constant[int],
    tn: ct.Constant[int],
    tk: ct.Constant[int],
) -> None:
    """Compute `(B, M, K) @ (B, K, N) -> (B, M, N)` with 2D L2 swizzling & FP32 accumulate."""
    group_m: ct.Constant[int] = 8
    pid = ct.bid(0)
    b_id = ct.bid(1)

    num_m = ct.num_tiles(a_ptr, axis=1, shape=(1, tm, tk))
    num_n = ct.num_tiles(b_ptr, axis=2, shape=(1, tk, tn))
    num_pid_in_group = group_m * num_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * group_m
    group_size_m = min(num_m - first_pid_m, group_m)
    pid_m = first_pid_m + (pid % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    num_k = ct.num_tiles(a_ptr, axis=2, shape=(1, tm, tk))
    acc = ct.zeros((tm, tn), dtype=ct.float32)

    for k in range(num_k):
        tile_a = ct.reshape(
            ct.load(
                a_ptr,
                index=(b_id, pid_m, k),
                shape=(1, tm, tk),
                padding_mode=ct.PaddingMode.ZERO,
                latency=3,
            ),
            (tm, tk),
        )
        tile_b = ct.reshape(
            ct.load(
                b_ptr,
                index=(b_id, k, pid_n),
                shape=(1, tk, tn),
                padding_mode=ct.PaddingMode.ZERO,
                latency=3,
            ),
            (tk, tn),
        )
        acc = ct.mma(tile_a, tile_b, acc)

    ct.store(
        c_ptr,
        index=(b_id, pid_m, pid_n),
        tile=ct.reshape(ct.astype(acc, c_ptr.dtype), (1, tm, tn)),
    )


def _select_bmm_tiles(
    batch: int,
    m: int,
    n: int,
    dtype: torch.dtype,
) -> tuple[int, int, int]:
    """Select L2-swizzled tile shape balancing CTA count across SMs vs arithmetic intensity."""
    tk = 32 if dtype == torch.float32 else 64
    if m <= 16:
        return (16, 64, tk)
    if m <= 64 or (m <= 512 and n <= 512 and batch >= 8):
        return (64, 64, tk)
    if m <= 256 and n <= 1024 and batch >= 8:
        return (64, 64, tk)
    return (128, 128, tk)


def cutile_bmm(
    a: torch.Tensor,
    b: torch.Tensor,
    tile_shape: tuple[int, int, int] | None = None,
) -> torch.Tensor:
    """Batched matrix multiplication `(B, M, K) @ (B, K, N) -> (B, M, N)`."""
    if a.ndim != 3 or b.ndim != 3:
        raise ValueError(f"Expected 3D tensors, got {tuple(a.shape)} and {tuple(b.shape)}")
    if a.shape[0] != b.shape[0] or a.shape[2] != b.shape[1]:
        raise ValueError(f"Shape mismatch for bmm: {tuple(a.shape)} vs {tuple(b.shape)}")
    if a.dtype != b.dtype:
        raise ValueError(f"Dtype mismatch for bmm: {a.dtype} vs {b.dtype}")

    if not a.is_cuda:
        return torch.bmm(a.float(), b.float()).to(a.dtype)

    if a.dtype not in (torch.float16, torch.bfloat16, torch.float32):
        return torch.bmm(a, b)

    batch, m, _ = a.shape
    _, _, n = b.shape
    tm, tn, tk = tile_shape or _select_bmm_tiles(batch, m, n, a.dtype)

    a_c = a.contiguous()
    b_c = b.contiguous()
    c = torch.empty((batch, m, n), device=a.device, dtype=a.dtype)
    grid = (ct.cdiv(m, tm) * ct.cdiv(n, tn), batch, 1)
    ct.launch(torch.cuda.current_stream(), grid, bmm_kernel, (a_c, b_c, c, tm, tn, tk))
    return c
