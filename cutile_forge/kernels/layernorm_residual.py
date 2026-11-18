"""Fused LayerNorm + Residual and RMSNorm + Residual CuTile kernels."""

from __future__ import annotations

import cuda.tile as ct
import torch


@ct.kernel(occupancy=2)
def layernorm_residual_kernel(
    x_ptr,
    res_ptr,
    w_ptr,
    b_ptr,
    norm_out_ptr,
    res_out_ptr,
    eps: float,
    inv_d: float,
    d_model: int,
    d_pad: ct.Constant[int],
) -> None:
    """Fused residual_sum = x + res; norm_out = LayerNorm(residual_sum)."""
    row = ct.bid(0)
    x = ct.astype(
        ct.load(x_ptr, index=(row, 0), shape=(1, d_pad), padding_mode=ct.PaddingMode.ZERO),
        ct.float32,
    )
    res = ct.astype(
        ct.load(res_ptr, index=(row, 0), shape=(1, d_pad), padding_mode=ct.PaddingMode.ZERO),
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

    combined = x + res
    ct.store(res_out_ptr, index=(row, 0), tile=ct.astype(combined, ct.float16))

    col_idx = ct.reshape(ct.arange(d_pad, dtype=ct.int32), (1, d_pad))
    valid = col_idx < d_model
    c_valid = ct.where(valid, combined, 0.0)

    mean = ct.sum(c_valid, axis=1, keepdims=True) * inv_d
    diff = ct.where(valid, c_valid - mean, 0.0)
    var = ct.sum(diff * diff, axis=1, keepdims=True) * inv_d
    normed = diff * ct.rsqrt(var + eps)
    out = normed * ct.reshape(w, (1, d_pad)) + ct.reshape(b, (1, d_pad))
    ct.store(norm_out_ptr, index=(row, 0), tile=ct.astype(out, ct.float16))


@ct.kernel(occupancy=2)
def rmsnorm_residual_kernel(
    x_ptr,
    res_ptr,
    w_ptr,
    norm_out_ptr,
    res_out_ptr,
    eps: float,
    inv_d: float,
    d_model: int,
    d_pad: ct.Constant[int],
) -> None:
    """Fused residual_sum = x + res; norm_out = RMSNorm(residual_sum)."""
    row = ct.bid(0)
    x = ct.astype(
        ct.load(x_ptr, index=(row, 0), shape=(1, d_pad), padding_mode=ct.PaddingMode.ZERO),
        ct.float32,
    )
    res = ct.astype(
        ct.load(res_ptr, index=(row, 0), shape=(1, d_pad), padding_mode=ct.PaddingMode.ZERO),
        ct.float32,
    )
    w = ct.astype(
        ct.load(w_ptr, index=(0,), shape=(d_pad,), padding_mode=ct.PaddingMode.ZERO),
        ct.float32,
    )

    combined = x + res
    ct.store(res_out_ptr, index=(row, 0), tile=ct.astype(combined, ct.float16))

    col_idx = ct.reshape(ct.arange(d_pad, dtype=ct.int32), (1, d_pad))
    c_valid = ct.where(col_idx < d_model, combined, 0.0)
    mean_sq = ct.sum(c_valid * c_valid, axis=1, keepdims=True) * inv_d
    out = (c_valid * ct.rsqrt(mean_sq + eps)) * ct.reshape(w, (1, d_pad))
    ct.store(norm_out_ptr, index=(row, 0), tile=ct.astype(out, ct.float16))


def _next_pow2(n: int) -> int:
    p = 1
    while p < n:
        p *= 2
    return p


def fused_layernorm_residual(
    x: torch.Tensor,
    residual: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None = None,
    eps: float = 1e-5,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute (LayerNorm(x + residual), x + residual) in a single CuTile kernel."""
    orig_shape = x.shape
    d_model = orig_shape[-1]
    x_2d = x.contiguous().view(-1, d_model)
    res_2d = residual.contiguous().view(-1, d_model)
    rows = x_2d.shape[0]
    d_pad = _next_pow2(d_model)

    b_tensor = (
        bias.contiguous()
        if bias is not None
        else torch.zeros(d_model, device=x.device, dtype=x.dtype)
    )
    norm_out = torch.empty_like(x_2d)
    res_out = torch.empty_like(x_2d)

    ct.launch(
        torch.cuda.current_stream(),
        (rows, 1, 1),
        layernorm_residual_kernel,
        (
            x_2d,
            res_2d,
            weight.contiguous(),
            b_tensor,
            norm_out,
            res_out,
            float(eps),
            1.0 / d_model,
            d_model,
            d_pad,
        ),
    )
    return norm_out.view(orig_shape), res_out.view(orig_shape)


def fused_rmsnorm_residual(
    x: torch.Tensor,
    residual: torch.Tensor,
    weight: torch.Tensor,
    eps: float = 1e-5,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute (RMSNorm(x + residual), x + residual) in a single CuTile kernel."""
    orig_shape = x.shape
    d_model = orig_shape[-1]
    x_2d = x.contiguous().view(-1, d_model)
    res_2d = residual.contiguous().view(-1, d_model)
    rows = x_2d.shape[0]
    d_pad = _next_pow2(d_model)

    norm_out = torch.empty_like(x_2d)
    res_out = torch.empty_like(x_2d)

    ct.launch(
        torch.cuda.current_stream(),
        (rows, 1, 1),
        rmsnorm_residual_kernel,
        (
            x_2d,
            res_2d,
            weight.contiguous(),
            norm_out,
            res_out,
            float(eps),
            1.0 / d_model,
            d_model,
            d_pad,
        ),
    )
    return norm_out.view(orig_shape), res_out.view(orig_shape)
