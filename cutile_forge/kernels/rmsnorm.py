"""Standalone RMSNorm kernel in CuTile (Mistral / LLaMA normalization)."""

from __future__ import annotations

import cuda.tile as ct
import torch


@ct.kernel(occupancy=2)
def rmsnorm_kernel(
    x_ptr,
    w_ptr,
    y_ptr,
    eps: float,
    inv_d: float,
    d_model: int,
    d_pad: ct.Constant[int],
) -> None:
    """Row-wise RMSNorm in a single CuTile tile pass."""
    row = ct.bid(0)
    x = ct.astype(
        ct.load(x_ptr, index=(row, 0), shape=(1, d_pad), padding_mode=ct.PaddingMode.ZERO),
        ct.float32,
    )
    w = ct.astype(
        ct.load(w_ptr, index=(0,), shape=(d_pad,), padding_mode=ct.PaddingMode.ZERO),
        ct.float32,
    )
    col_idx = ct.reshape(ct.arange(d_pad, dtype=ct.int32), (1, d_pad))
    x_valid = ct.where(col_idx < d_model, x, 0.0)
    mean_sq = ct.sum(x_valid * x_valid, axis=1, keepdims=True) * inv_d
    inv_rms = ct.rsqrt(mean_sq + eps)
    out = (x_valid * inv_rms) * ct.reshape(w, (1, d_pad))
    ct.store(y_ptr, index=(row, 0), tile=ct.astype(out, ct.float16))


@ct.kernel(occupancy=4)
def rmsnorm_permute_kernel(
    x_ptr,
    w_ptr,
    perm_ptr,
    y_ptr,
    eps: float,
    inv_d: float,
    d_model: int,
    d_pad: ct.Constant[int],
) -> None:
    """Row-wise RMSNorm followed by column permutation `y[:, d] = rmsnorm(x)[:, perm[d]]`."""
    row = ct.bid(0)
    x = ct.astype(
        ct.load(x_ptr, index=(row, 0), shape=(1, d_pad), padding_mode=ct.PaddingMode.ZERO),
        ct.float32,
    )
    col_idx = ct.reshape(ct.arange(d_pad, dtype=ct.int32), (1, d_pad))
    x_valid = ct.where(col_idx < d_model, x, 0.0)
    mean_sq = ct.sum(x_valid * x_valid, axis=1, keepdims=True) * inv_d
    inv_rms = ct.rsqrt(mean_sq + eps)

    p = ct.reshape(
        ct.astype(
            ct.load(perm_ptr, index=(0,), shape=(d_pad,), padding_mode=ct.PaddingMode.ZERO),
            ct.int32,
        ),
        (1, d_pad),
    )
    row_2d = ct.reshape(ct.full((1,), row, dtype=ct.int32), (1, 1))
    xp = ct.astype(ct.gather(x_ptr, (row_2d, p), padding_value=0), ct.float32)
    wp = ct.astype(ct.gather(w_ptr, (ct.reshape(p, (d_pad,)),), padding_value=0), ct.float32)
    out = (xp * inv_rms) * ct.reshape(wp, (1, d_pad))
    ct.store(y_ptr, index=(row, 0), tile=ct.astype(out, ct.float16))


def _next_pow2(n: int) -> int:
    p = 1
    while p < n:
        p *= 2
    return p


def cutile_rmsnorm(
    x: torch.Tensor,
    weight: torch.Tensor,
    eps: float = 1e-5,
    out: torch.Tensor | None = None,
    perm: torch.Tensor | None = None,
) -> torch.Tensor:
    """Compute RMSNorm over the last dimension using CuTile (zero-alloc when `out` is given)."""
    orig_shape = x.shape
    d_model = orig_shape[-1]
    x_2d = x.view(-1, d_model) if x.is_contiguous() else x.contiguous().view(-1, d_model)
    rows = x_2d.shape[0]
    d_pad = _next_pow2(d_model)
    y_2d = out.view(rows, d_model) if out is not None else torch.empty_like(x_2d)
    w_c = weight if weight.is_contiguous() else weight.contiguous()

    if perm is not None:
        p_c = (
            perm
            if (perm.is_contiguous() and perm.dtype == torch.int32)
            else perm.to(torch.int32).contiguous()
        )
        ct.launch(
            torch.cuda.current_stream(),
            (rows, 1, 1),
            rmsnorm_permute_kernel,
            (x_2d, w_c, p_c, y_2d, float(eps), 1.0 / d_model, int(d_model), d_pad),
        )
    else:
        ct.launch(
            torch.cuda.current_stream(),
            (rows, 1, 1),
            rmsnorm_kernel,
            (x_2d, w_c, y_2d, float(eps), 1.0 / d_model, int(d_model), d_pad),
        )
    return y_2d.view(orig_shape)
