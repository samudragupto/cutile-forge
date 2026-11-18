"""Standalone LayerNorm kernel in CuTile."""

from __future__ import annotations

import cuda.tile as ct
import torch


@ct.kernel(occupancy=2)
def layernorm_kernel(
    x_ptr,
    w_ptr,
    b_ptr,
    y_ptr,
    eps: float,
    inv_d: float,
    d_model: int,
    d_pad: ct.Constant[int],
) -> None:
    """Row-wise LayerNorm in a single CuTile tile pass."""
    row = ct.bid(0)
    x = ct.astype(
        ct.load(x_ptr, index=(row, 0), shape=(1, d_pad), padding_mode=ct.PaddingMode.ZERO),
        ct.float32,
    )
    w = ct.astype(
        ct.load(w_ptr, index=(0,), shape=(d_pad,), padding_mode=ct.PaddingMode.ZERO),
        ct.float32,
    )
    b = ct.astype(
        ct.load(b_ptr, index=(0,), shape=(d_pad,), padding_mode=ct.PaddingMode.ZERO),
        ct.float32,
    )

    col_idx = ct.reshape(ct.arange(d_pad, dtype=ct.int32), (1, d_pad))
    valid = col_idx < d_model
    x_valid = ct.where(valid, x, 0.0)

    mean = ct.sum(x_valid, axis=1, keepdims=True) * inv_d
    diff = ct.where(valid, x_valid - mean, 0.0)
    var = ct.sum(diff * diff, axis=1, keepdims=True) * inv_d
    inv_std = ct.rsqrt(var + eps)
    normed = diff * inv_std

    out = normed * ct.reshape(w, (1, d_pad)) + ct.reshape(b, (1, d_pad))
    ct.store(y_ptr, index=(row, 0), tile=ct.astype(out, ct.float16))


def _next_pow2(n: int) -> int:
    p = 1
    while p < n:
        p *= 2
    return p


def cutile_layernorm(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None = None,
    eps: float = 1e-5,
) -> torch.Tensor:
    """Compute LayerNorm over the last dimension using CuTile."""
    orig_shape = x.shape
    d_model = orig_shape[-1]
    x_2d = x.contiguous().view(-1, d_model)
    rows = x_2d.shape[0]
    d_pad = _next_pow2(d_model)

    b_tensor = (
        bias.contiguous()
        if bias is not None
        else torch.zeros(d_model, device=x.device, dtype=x.dtype)
    )
    y_2d = torch.empty_like(x_2d)

    ct.launch(
        torch.cuda.current_stream(),
        (rows, 1, 1),
        layernorm_kernel,
        (x_2d, weight.contiguous(), b_tensor, y_2d, float(eps), 1.0 / d_model, int(d_model), d_pad),
    )
    return y_2d.view(orig_shape)
