"""General Batched Matrix Multiplication (`cutile_bmm`) kernel in CuTile."""

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
    """Compute `(B, M, K) @ (B, K, N) -> (B, M, N)` using FP32 accumulation."""
    pid_n = ct.bid(0)
    pid_m = ct.bid(1)
    b_id = ct.bid(2)

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


def _select_bmm_tiles(m: int, dtype: torch.dtype) -> tuple[int, int, int]:
    tk = 32 if dtype == torch.float32 else 64
    if m <= 16:
        return (16, 64, tk)
    if m <= 64:
        return (64, 64, tk)
    return (128, 128, 32)


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
    tm, tn, tk = tile_shape or _select_bmm_tiles(m, a.dtype)

    a_c = a.contiguous()
    b_c = b.contiguous()
    c = torch.empty((batch, m, n), device=a.device, dtype=a.dtype)
    grid = (ct.cdiv(n, tn), ct.cdiv(m, tm), batch)
    ct.launch(torch.cuda.current_stream(), grid, bmm_kernel, (a_c, b_c, c, tm, tn, tk))
    return c
