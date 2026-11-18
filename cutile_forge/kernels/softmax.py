"""Online softmax (running max and running sum) kernel in CuTile."""

from __future__ import annotations

import cuda.tile as ct
import torch


@ct.kernel(occupancy=2)
def online_softmax_kernel(
    x_ptr,
    y_ptr,
    n_cols: int,
    tile_n: ct.Constant[int],
) -> None:
    """Row-wise tiled online softmax supporting arbitrary column counts."""
    row = ct.bid(0)
    num_tiles = (n_cols + tile_n - 1) // tile_n
    col_base = ct.arange(tile_n, dtype=ct.int32)

    m_i = ct.full((1, 1), -1.0e9, dtype=ct.float32)
    l_i = ct.zeros((1, 1), dtype=ct.float32)

    for t in range(num_tiles):
        tile_x = ct.astype(
            ct.load(x_ptr, index=(row, t), shape=(1, tile_n), padding_mode=ct.PaddingMode.NEG_INF),
            ct.float32,
        )
        col_idx = ct.reshape(t * tile_n + col_base, (1, tile_n))
        tile_x = ct.where(col_idx < n_cols, tile_x, -1.0e9)
        m_tile = ct.max(tile_x, axis=1, keepdims=True)
        m_new = ct.maximum(m_i, m_tile)
        l_i = l_i * ct.exp(m_i - m_new) + ct.sum(ct.exp(tile_x - m_new), axis=1, keepdims=True)
        m_i = m_new

    inv_l = 1.0 / ct.maximum(l_i, 1.0e-9)
    for t in range(num_tiles):
        tile_x = ct.astype(
            ct.load(x_ptr, index=(row, t), shape=(1, tile_n), padding_mode=ct.PaddingMode.NEG_INF),
            ct.float32,
        )
        col_idx = ct.reshape(t * tile_n + col_base, (1, tile_n))
        tile_x = ct.where(col_idx < n_cols, tile_x, -1.0e9)
        probs = ct.exp(tile_x - m_i) * inv_l
        ct.store(y_ptr, index=(row, t), tile=ct.astype(probs, ct.float16))


def online_softmax(x: torch.Tensor) -> torch.Tensor:
    """Compute softmax over the last dimension using the CuTile online softmax kernel."""
    orig_shape = x.shape
    n_cols = orig_shape[-1]
    x_2d = x.contiguous().view(-1, n_cols)
    rows = x_2d.shape[0]
    y_2d = torch.empty_like(x_2d)

    tile_n = 1024 if n_cols > 512 else 256
    ct.launch(
        torch.cuda.current_stream(),
        (rows, 1, 1),
        online_softmax_kernel,
        (x_2d, y_2d, int(n_cols), tile_n),
    )
    return y_2d.view(orig_shape)
