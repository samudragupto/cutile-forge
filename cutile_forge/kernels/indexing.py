"""Indexing (`ct.gather` embedding lookup) and vision (`im2col` + `cutile_bmm` Conv2D) kernels."""

from __future__ import annotations

import torch
import torch.nn.functional as F

from cutile_forge.cuda_tile_init import ct
from cutile_forge.kernels.bmm import cutile_bmm


@ct.kernel(occupancy=2)
def gather_rows_kernel(
    table_ptr,
    idx_ptr,
    out_ptr,
    tile_m: ct.Constant[int],
    tile_d: ct.Constant[int],
) -> None:
    """2D row gather `table[indices, :]` via `ct.gather`."""
    pid_n = ct.bid(0)
    pid_d = ct.bid(1)

    row_ids = ct.reshape(
        ct.astype(
            ct.load(idx_ptr, index=(pid_n,), shape=(tile_m,), padding_mode=ct.PaddingMode.ZERO),
            ct.int32,
        ),
        (tile_m, 1),
    )
    col_ids = ct.reshape(pid_d * tile_d + ct.arange(tile_d, dtype=ct.int32), (1, tile_d))
    gathered = ct.gather(table_ptr, (row_ids, col_ids), padding_value=0)
    ct.store(out_ptr, index=(pid_n, pid_d), tile=gathered)


def cutile_gather(table: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
    """Gather rows from a 2D `(V, D)` table for arbitrary-shape integer `indices`."""
    if table.ndim != 2:
        raise ValueError(f"Expected 2D table (V, D), got {tuple(table.shape)}")
    flat_idx = indices.reshape(-1)
    d_dim = table.shape[1]
    out_shape = (*indices.shape, d_dim)

    if not table.is_cuda or table.dtype not in (torch.float16, torch.bfloat16, torch.float32):
        return table.index_select(0, flat_idx.to(device=table.device, dtype=torch.long)).view(
            out_shape
        )

    idx_i32 = flat_idx.to(device=table.device, dtype=torch.int32).contiguous()
    table_c = table.contiguous()
    out = torch.empty((idx_i32.numel(), d_dim), device=table.device, dtype=table.dtype)
    tile_m, tile_d = 32, 64
    grid = (ct.cdiv(idx_i32.numel(), tile_m), ct.cdiv(d_dim, tile_d), 1)
    ct.launch(
        torch.cuda.current_stream(),
        grid,
        gather_rows_kernel,
        (table_c, idx_i32, out, tile_m, tile_d),
    )
    return out.view(out_shape)


def cutile_conv2d(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None = None,
    stride: int | tuple[int, int] = 1,
    padding: int | tuple[int, int] = 0,
) -> torch.Tensor:
    """2D convolution lowered via im2col (`F.unfold`) + `cutile_bmm`.

    `cuda.tile` 1.8.1 does not expose a native hardware convolution op, so spatial
    patches are unfolded to `(N, K_patch, L)` and multiplied via `cutile_bmm`.
    """
    if x.ndim != 4 or weight.ndim != 4:
        raise ValueError(
            f"Expected 4D NCHW tensors, got x={tuple(x.shape)} weight={tuple(weight.shape)}"
        )

    if not x.is_cuda or x.dtype not in (torch.float16, torch.bfloat16, torch.float32):
        return F.conv2d(x, weight, bias=bias, stride=stride, padding=padding)

    n_batch, _, h_in, w_in = x.shape
    c_out, _, k_h, k_w = weight.shape
    s_h, s_w = (stride, stride) if isinstance(stride, int) else stride
    p_h, p_w = (padding, padding) if isinstance(padding, int) else padding

    h_out = (h_in + 2 * p_h - k_h) // s_h + 1
    w_out = (w_in + 2 * p_w - k_w) // s_w + 1

    # F.unfold supports FP16/BF16/FP32 -> (N, C_in * kH * kW, L)
    cols = F.unfold(x, kernel_size=(k_h, k_w), padding=(p_h, p_w), stride=(s_h, s_w))
    w_flat = weight.view(c_out, -1).unsqueeze(0).expand(n_batch, c_out, -1)
    out = cutile_bmm(w_flat, cols)  # (N, C_out, H_out * W_out)
    out_4d = out.view(n_batch, c_out, h_out, w_out)
    if bias is not None:
        out_4d = out_4d + bias.view(1, c_out, 1, 1)
    return out_4d
