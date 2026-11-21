"""Fused LayerNorm + Residual and RMSNorm + Residual CuTile kernels."""

from __future__ import annotations

import math

import cuda.tile as ct
import torch
import torch.nn.functional as F


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
    out_norm: torch.Tensor | None = None,
    out_res: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute (RMSNorm(x + residual), x + residual) in a single CuTile kernel."""
    orig_shape = x.shape
    d_model = orig_shape[-1]
    x_2d = x.view(-1, d_model) if x.is_contiguous() else x.contiguous().view(-1, d_model)
    res_2d = (
        residual.view(-1, d_model)
        if residual.is_contiguous()
        else residual.contiguous().view(-1, d_model)
    )
    rows = x_2d.shape[0]
    d_pad = _next_pow2(d_model)

    norm_out = out_norm.view(rows, d_model) if out_norm is not None else torch.empty_like(x_2d)
    res_out = out_res.view(rows, d_model) if out_res is not None else torch.empty_like(x_2d)
    w_c = weight if weight.is_contiguous() else weight.contiguous()

    ct.launch(
        torch.cuda.current_stream(),
        (rows, 1, 1),
        rmsnorm_residual_kernel,
        (
            x_2d,
            res_2d,
            w_c,
            norm_out,
            res_out,
            float(eps),
            1.0 / d_model,
            d_model,
            d_pad,
        ),
    )
    return norm_out.view(orig_shape), res_out.view(orig_shape)


@ct.kernel(occupancy=2)
def fused_rmsnorm_silu_kernel(
    x_ptr,
    w_ptr,
    out_ptr,
    eps: float,
    inv_d: float,
    d_model: int,
    d_pad: ct.Constant[int],
) -> None:
    """Single-pass SiLU(RMSNorm(x)) in CuTile without intermediate HBM round-trip."""
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
    valid = col_idx < d_model
    x_v = ct.where(valid, x, 0.0)
    mean_sq = ct.sum(x_v * x_v, axis=1, keepdims=True) * inv_d
    normed = (x_v * ct.rsqrt(mean_sq + eps)) * ct.reshape(w, (1, d_pad))
    out = normed / (1.0 + ct.exp(-normed))
    ct.store(out_ptr, index=(row, 0), tile=ct.astype(out, out_ptr.dtype))


@ct.kernel(occupancy=2)
def fused_layernorm_gelu_kernel(
    x_ptr,
    w_ptr,
    b_ptr,
    out_ptr,
    eps: float,
    inv_d: float,
    sqrt_2_over_pi: float,
    d_model: int,
    d_pad: ct.Constant[int],
) -> None:
    """Single-pass GELU_tanh(LayerNorm(x)) in CuTile without intermediate HBM write."""
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
    x_v = ct.where(valid, x, 0.0)
    mean = ct.sum(x_v, axis=1, keepdims=True) * inv_d
    diff = ct.where(valid, x_v - mean, 0.0)
    var = ct.sum(diff * diff, axis=1, keepdims=True) * inv_d
    normed = (diff * ct.rsqrt(var + eps)) * ct.reshape(w, (1, d_pad)) + ct.reshape(b, (1, d_pad))
    inner = sqrt_2_over_pi * (normed + 0.044715 * normed * normed * normed)
    out = 0.5 * normed * (1.0 + ct.tanh(inner))
    ct.store(out_ptr, index=(row, 0), tile=ct.astype(out, out_ptr.dtype))


@ct.kernel(occupancy=2)
def fused_add_rmsnorm_silu_kernel(
    x_ptr,
    res_ptr,
    w_ptr,
    act_out_ptr,
    res_out_ptr,
    eps: float,
    inv_d: float,
    d_model: int,
    d_pad: ct.Constant[int],
) -> None:
    """Single-pass residual = x + res; act_out = SiLU(RMSNorm(residual)) in CuTile."""
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
    ct.store(res_out_ptr, index=(row, 0), tile=ct.astype(combined, res_out_ptr.dtype))

    col_idx = ct.reshape(ct.arange(d_pad, dtype=ct.int32), (1, d_pad))
    valid = col_idx < d_model
    c_valid = ct.where(valid, combined, 0.0)
    mean_sq = ct.sum(c_valid * c_valid, axis=1, keepdims=True) * inv_d
    normed = (c_valid * ct.rsqrt(mean_sq + eps)) * ct.reshape(w, (1, d_pad))
    out = normed / (1.0 + ct.exp(-normed))
    ct.store(act_out_ptr, index=(row, 0), tile=ct.astype(out, act_out_ptr.dtype))


def cutile_fused_rmsnorm_silu(
    x: torch.Tensor,
    weight: torch.Tensor,
    eps: float = 1e-5,
) -> torch.Tensor:
    """Compute `SiLU(RMSNorm(x))` in a single fused CuTile kernel pass."""
    orig_shape = x.shape
    d_model = orig_shape[-1]
    if not x.is_cuda or x.dtype not in (torch.float16, torch.bfloat16, torch.float32):
        xf = x.float()
        y = (xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + eps)).to(x.dtype) * weight
        return F.silu(y)

    x_2d = x.contiguous().view(-1, d_model)
    out = torch.empty_like(x_2d)
    ct.launch(
        torch.cuda.current_stream(),
        (x_2d.shape[0], 1, 1),
        fused_rmsnorm_silu_kernel,
        (x_2d, weight.contiguous(), out, float(eps), 1.0 / d_model, d_model, _next_pow2(d_model)),
    )
    return out.view(orig_shape)


def cutile_fused_layernorm_gelu(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None = None,
    eps: float = 1e-5,
) -> torch.Tensor:
    """Compute `GELU_tanh(LayerNorm(x))` in a single fused CuTile kernel pass."""
    orig_shape = x.shape
    d_model = orig_shape[-1]
    if not x.is_cuda or x.dtype not in (torch.float16, torch.bfloat16, torch.float32):
        return F.gelu(
            F.layer_norm(x, (d_model,), weight=weight, bias=bias, eps=eps), approximate="tanh"
        )

    x_2d = x.contiguous().view(-1, d_model)
    b_tensor = (
        bias.contiguous()
        if bias is not None
        else torch.zeros(d_model, device=x.device, dtype=x.dtype)
    )
    out = torch.empty_like(x_2d)
    ct.launch(
        torch.cuda.current_stream(),
        (x_2d.shape[0], 1, 1),
        fused_layernorm_gelu_kernel,
        (
            x_2d,
            weight.contiguous(),
            b_tensor,
            out,
            float(eps),
            1.0 / d_model,
            math.sqrt(2.0 / math.pi),
            d_model,
            _next_pow2(d_model),
        ),
    )
    return out.view(orig_shape)


def cutile_fused_add_rmsnorm_silu(
    x: torch.Tensor,
    residual: torch.Tensor,
    weight: torch.Tensor,
    eps: float = 1e-5,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute `(SiLU(RMSNorm(x + residual)), x + residual)` in a single fused CuTile kernel."""
    orig_shape = x.shape
    d_model = orig_shape[-1]
    if not x.is_cuda or x.dtype not in (torch.float16, torch.bfloat16, torch.float32):
        comb = x + residual
        cf = comb.float()
        y = (cf * torch.rsqrt(cf.pow(2).mean(-1, keepdim=True) + eps)).to(x.dtype) * weight
        return F.silu(y), comb

    x_2d = x.contiguous().view(-1, d_model)
    res_2d = residual.contiguous().view(-1, d_model)
    act_out = torch.empty_like(x_2d)
    res_out = torch.empty_like(x_2d)
    ct.launch(
        torch.cuda.current_stream(),
        (x_2d.shape[0], 1, 1),
        fused_add_rmsnorm_silu_kernel,
        (
            x_2d,
            res_2d,
            weight.contiguous(),
            act_out,
            res_out,
            float(eps),
            1.0 / d_model,
            d_model,
            _next_pow2(d_model),
        ),
    )
    return act_out.view(orig_shape), res_out.view(orig_shape)
