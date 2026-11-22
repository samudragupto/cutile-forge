"""Activation forward/backward and RMSNorm backward kernels in CuTile."""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F

from cutile_forge.cuda_tile_init import ct
from cutile_forge.kernels.reductions import cutile_reduce_sum


@ct.kernel(occupancy=4)
def silu_forward_kernel(
    x_ptr,
    y_ptr,
    tile_n: ct.Constant[int],
) -> None:
    """Elementwise SiLU forward: x * sigmoid(x) with FP32 math."""
    pid = ct.bid(0)
    x = ct.astype(
        ct.load(
            x_ptr,
            index=(pid,),
            shape=(tile_n,),
            padding_mode=ct.PaddingMode.ZERO,
            latency=4,
        ),
        ct.float32,
    )
    sig = 1.0 / (1.0 + ct.exp(-x))
    out = x * sig
    ct.store(y_ptr, index=(pid,), tile=ct.astype(out, y_ptr.dtype))


@ct.kernel(occupancy=4)
def silu_backward_kernel(
    dy_ptr,
    x_ptr,
    dx_ptr,
    tile_n: ct.Constant[int],
) -> None:
    """Elementwise SiLU backward: dy * (sig + x * sig * (1 - sig))."""
    pid = ct.bid(0)
    dy = ct.astype(
        ct.load(
            dy_ptr,
            index=(pid,),
            shape=(tile_n,),
            padding_mode=ct.PaddingMode.ZERO,
            latency=3,
        ),
        ct.float32,
    )
    x = ct.astype(
        ct.load(
            x_ptr,
            index=(pid,),
            shape=(tile_n,),
            padding_mode=ct.PaddingMode.ZERO,
            latency=3,
        ),
        ct.float32,
    )
    sig = 1.0 / (1.0 + ct.exp(-x))
    dx = dy * (sig * (1.0 + x * (1.0 - sig)))
    ct.store(dx_ptr, index=(pid,), tile=ct.astype(dx, dx_ptr.dtype))


@ct.kernel(occupancy=8)
def gelu_forward_kernel(
    x_ptr,
    y_ptr,
    sqrt_2_over_pi: float,
    tile_n: ct.Constant[int],
) -> None:
    """Elementwise approximate GELU forward in CuTile."""
    pid = ct.bid(0)
    x = ct.astype(
        ct.load(
            x_ptr,
            index=(pid,),
            shape=(tile_n,),
            padding_mode=ct.PaddingMode.ZERO,
            latency=4,
        ),
        ct.float32,
    )
    inner = sqrt_2_over_pi * (x + 0.044715 * x * x * x)
    out = 0.5 * x * (1.0 + ct.tanh(inner))
    ct.store(y_ptr, index=(pid,), tile=ct.astype(out, y_ptr.dtype))


@ct.kernel(occupancy=8)
def quick_gelu_forward_kernel(
    x_ptr,
    y_ptr,
    tile_n: ct.Constant[int],
) -> None:
    """Elementwise QuickGELU (`x * sigmoid(1.702 * x)`) forward in CuTile."""
    pid = ct.bid(0)
    x = ct.astype(
        ct.load(
            x_ptr,
            index=(pid,),
            shape=(tile_n,),
            padding_mode=ct.PaddingMode.ZERO,
            latency=4,
        ),
        ct.float32,
    )
    out = x / (1.0 + ct.exp(-1.702 * x))
    ct.store(y_ptr, index=(pid,), tile=ct.astype(out, y_ptr.dtype))


@ct.kernel(occupancy=8)
def gelu_erf_forward_kernel(
    x_ptr,
    y_ptr,
    tile_n: ct.Constant[int],
) -> None:
    """Exact erf-based GELU (`0.5 * x * (1 + erf(x / sqrt(2)))`) forward in CuTile."""
    pid = ct.bid(0)
    x = ct.astype(
        ct.load(
            x_ptr,
            index=(pid,),
            shape=(tile_n,),
            padding_mode=ct.PaddingMode.ZERO,
            latency=4,
        ),
        ct.float32,
    )
    z = ct.where(x >= 0.0, x, -x) * 0.7071067811865476
    t = 1.0 / (1.0 + 0.3275911 * z)
    poly = (
        (((1.061405429 * t - 1.453152027) * t + 1.421413741) * t - 0.284496736) * t + 0.254829592
    ) * t
    erf_abs = 1.0 - poly * ct.exp(-z * z)
    erf_x = ct.where(x >= 0.0, erf_abs, -erf_abs)
    out = 0.5 * x * (1.0 + erf_x)
    ct.store(y_ptr, index=(pid,), tile=ct.astype(out, y_ptr.dtype))


@ct.kernel(occupancy=4)
def gelu_backward_kernel(
    dy_ptr,
    x_ptr,
    dx_ptr,
    sqrt_2_over_pi: float,
    tile_n: ct.Constant[int],
) -> None:
    """Elementwise approximate GELU backward in CuTile."""
    pid = ct.bid(0)
    dy = ct.astype(
        ct.load(dy_ptr, index=(pid,), shape=(tile_n,), padding_mode=ct.PaddingMode.ZERO, latency=3),
        ct.float32,
    )
    x = ct.astype(
        ct.load(x_ptr, index=(pid,), shape=(tile_n,), padding_mode=ct.PaddingMode.ZERO, latency=3),
        ct.float32,
    )
    x2 = x * x
    inner = sqrt_2_over_pi * (x + 0.044715 * x2 * x)
    t = ct.tanh(inner)
    sech2 = 1.0 - t * t
    d_inner = sqrt_2_over_pi * (1.0 + 3.0 * 0.044715 * x2)
    dx = dy * (0.5 * (1.0 + t) + 0.5 * x * sech2 * d_inner)
    ct.store(dx_ptr, index=(pid,), tile=ct.astype(dx, dx_ptr.dtype))


@ct.kernel(occupancy=2)
def layernorm_backward_dx_kernel(
    dy_ptr,
    x_ptr,
    w_ptr,
    dx_ptr,
    eps: float,
    inv_d: float,
    d_model: int,
    d_pad: ct.Constant[int],
) -> None:
    """Row-wise LayerNorm input gradient kernel in CuTile."""
    row = ct.bid(0)
    dy = ct.astype(
        ct.load(dy_ptr, index=(row, 0), shape=(1, d_pad), padding_mode=ct.PaddingMode.ZERO),
        ct.float32,
    )
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
    dy_v = ct.where(valid, dy, 0.0)

    mean = ct.sum(x_v, axis=1, keepdims=True) * inv_d
    diff = ct.where(valid, x_v - mean, 0.0)
    var = ct.sum(diff * diff, axis=1, keepdims=True) * inv_d
    inv_std = ct.rsqrt(var + eps)
    x_hat = diff * inv_std

    dy_w = dy_v * ct.reshape(w, (1, d_pad))
    mean_dy_w = ct.sum(dy_w, axis=1, keepdims=True) * inv_d
    mean_dy_w_xhat = ct.sum(dy_w * x_hat, axis=1, keepdims=True) * inv_d
    dx = inv_std * (dy_w - mean_dy_w - x_hat * mean_dy_w_xhat)
    ct.store(dx_ptr, index=(row, 0), tile=ct.astype(dx, dx_ptr.dtype))


@ct.kernel(occupancy=2)
def rmsnorm_backward_dx_kernel(
    dy_ptr,
    x_ptr,
    w_ptr,
    dx_ptr,
    eps: float,
    inv_d: float,
    d_model: int,
    d_pad: ct.Constant[int],
) -> None:
    """Row-wise RMSNorm input gradient kernel in CuTile."""
    row = ct.bid(0)
    dy = ct.astype(
        ct.load(dy_ptr, index=(row, 0), shape=(1, d_pad), padding_mode=ct.PaddingMode.ZERO),
        ct.float32,
    )
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
    dy_v = ct.where(valid, dy, 0.0)
    dy_w = dy_v * ct.reshape(w, (1, d_pad))

    mean_sq = ct.sum(x_v * x_v, axis=1, keepdims=True) * inv_d
    inv_rms = ct.rsqrt(mean_sq + eps)
    dot = ct.sum(dy_w * x_v, axis=1, keepdims=True) * inv_d
    dx = inv_rms * dy_w - x_v * (inv_rms * inv_rms * inv_rms * dot)
    ct.store(dx_ptr, index=(row, 0), tile=ct.astype(dx, dx_ptr.dtype))


def _next_pow2(n: int) -> int:
    p = 1
    while p < n:
        p *= 2
    return p


def _pick_tile_1d(numel: int) -> int:
    if numel >= 16384:
        return 2048
    return 1024


def cutile_silu(x: torch.Tensor) -> torch.Tensor:
    """Compute SiLU activation via CuTile (FP16, BF16, FP32)."""
    if not x.is_cuda or x.dtype not in (torch.float16, torch.bfloat16, torch.float32):
        return F.silu(x)
    x_1d = x.view(-1) if x.is_contiguous() else x.contiguous().view(-1)
    y_1d = torch.empty_like(x_1d)
    tile_n = _pick_tile_1d(x_1d.numel())
    grid = (ct.cdiv(x_1d.numel(), tile_n), 1, 1)
    ct.launch(torch.cuda.current_stream(), grid, silu_forward_kernel, (x_1d, y_1d, tile_n))
    return y_1d.view(x.shape)


def cutile_silu_backward(grad_out: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """Compute dL/dx for SiLU via CuTile."""
    if not x.is_cuda or x.dtype not in (torch.float16, torch.bfloat16, torch.float32):
        sig = torch.sigmoid(x.float())
        return (grad_out.float() * sig * (1.0 + x.float() * (1.0 - sig))).to(x.dtype)
    dy_1d = grad_out.view(-1) if grad_out.is_contiguous() else grad_out.contiguous().view(-1)
    x_1d = x.view(-1) if x.is_contiguous() else x.contiguous().view(-1)
    dx_1d = torch.empty_like(x_1d)
    tile_n = _pick_tile_1d(x_1d.numel())
    grid = (ct.cdiv(x_1d.numel(), tile_n), 1, 1)
    ct.launch(torch.cuda.current_stream(), grid, silu_backward_kernel, (dy_1d, x_1d, dx_1d, tile_n))
    return dx_1d.view(x.shape)


def cutile_gelu(x: torch.Tensor) -> torch.Tensor:
    """Compute tanh-approximate GELU activation via CuTile (FP16, BF16, FP32)."""
    if not x.is_cuda or x.dtype not in (torch.float16, torch.bfloat16, torch.float32):
        return F.gelu(x, approximate="tanh")
    x_1d = x.view(-1) if x.is_contiguous() else x.contiguous().view(-1)
    y_1d = torch.empty_like(x_1d)
    tile_n = _pick_tile_1d(x_1d.numel())
    grid = (ct.cdiv(x_1d.numel(), tile_n), 1, 1)
    ct.launch(
        torch.cuda.current_stream(),
        grid,
        gelu_forward_kernel,
        (x_1d, y_1d, math.sqrt(2.0 / math.pi), tile_n),
    )
    return y_1d.view(x.shape)


def cutile_quick_gelu(x: torch.Tensor) -> torch.Tensor:
    """Compute QuickGELU activation (`x * sigmoid(1.702 * x)`) via CuTile."""
    if not x.is_cuda or x.dtype not in (torch.float16, torch.bfloat16, torch.float32):
        xf = x.float()
        return (xf * torch.sigmoid(1.702 * xf)).to(x.dtype)
    x_1d = x.view(-1) if x.is_contiguous() else x.contiguous().view(-1)
    y_1d = torch.empty_like(x_1d)
    tile_n = _pick_tile_1d(x_1d.numel())
    grid = (ct.cdiv(x_1d.numel(), tile_n), 1, 1)
    ct.launch(torch.cuda.current_stream(), grid, quick_gelu_forward_kernel, (x_1d, y_1d, tile_n))
    return y_1d.view(x.shape)


def cutile_gelu_erf(x: torch.Tensor) -> torch.Tensor:
    """Compute exact erf-based GELU (`approximate='none'`) via CuTile."""
    if not x.is_cuda or x.dtype not in (torch.float16, torch.bfloat16, torch.float32):
        return F.gelu(x.float(), approximate="none").to(x.dtype)
    x_1d = x.view(-1) if x.is_contiguous() else x.contiguous().view(-1)
    y_1d = torch.empty_like(x_1d)
    tile_n = _pick_tile_1d(x_1d.numel())
    grid = (ct.cdiv(x_1d.numel(), tile_n), 1, 1)
    ct.launch(torch.cuda.current_stream(), grid, gelu_erf_forward_kernel, (x_1d, y_1d, tile_n))
    return y_1d.view(x.shape)


def cutile_rmsnorm_backward(
    grad_out: torch.Tensor,
    x: torch.Tensor,
    weight: torch.Tensor,
    eps: float = 1e-5,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute `(grad_x, grad_weight)` for RMSNorm over the last dimension."""
    orig_shape = x.shape
    d_model = orig_shape[-1]
    x_2d = x.contiguous().view(-1, d_model)
    dy_2d = grad_out.contiguous().view(-1, d_model)
    rows = x_2d.shape[0]

    # Compute grad_weight via FP32 reduction
    x_f = x_2d.float()
    inv_rms = torch.rsqrt(x_f.pow(2).mean(dim=-1, keepdim=True) + eps)
    grad_w = (dy_2d.float() * x_f * inv_rms).sum(dim=0).to(weight.dtype)

    if not x.is_cuda or x.dtype not in (torch.float16, torch.bfloat16, torch.float32):
        dy_w = dy_2d.float() * weight.float()
        dot = (dy_w * x_f).sum(dim=-1, keepdim=True) / float(d_model)
        dx_2d = (inv_rms * dy_w - x_f * (inv_rms**3) * dot).to(x.dtype)
        return dx_2d.view(orig_shape), grad_w

    d_pad = _next_pow2(d_model)
    dx_2d = torch.empty_like(x_2d)
    ct.launch(
        torch.cuda.current_stream(),
        (rows, 1, 1),
        rmsnorm_backward_dx_kernel,
        (
            dy_2d,
            x_2d,
            weight.contiguous(),
            dx_2d,
            float(eps),
            1.0 / float(d_model),
            int(d_model),
            d_pad,
        ),
    )
    return dx_2d.view(orig_shape), grad_w


def cutile_gelu_backward(grad_out: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """Compute dL/dx for tanh-approximate GELU via CuTile."""
    sqrt_2_over_pi = math.sqrt(2.0 / math.pi)
    if not x.is_cuda or x.dtype not in (torch.float16, torch.bfloat16, torch.float32):
        xf = x.float()
        x2 = xf * xf
        inner = sqrt_2_over_pi * (xf + 0.044715 * x2 * xf)
        t = torch.tanh(inner)
        sech2 = 1.0 - t * t
        d_inner = sqrt_2_over_pi * (1.0 + 3.0 * 0.044715 * x2)
        dx = grad_out.float() * (0.5 * (1.0 + t) + 0.5 * xf * sech2 * d_inner)
        return dx.to(x.dtype)

    dy_1d = grad_out.contiguous().view(-1)
    x_1d = x.contiguous().view(-1)
    dx_1d = torch.empty_like(x_1d)
    tile_n = 4096 if x_1d.numel() >= 16384 else 1024
    grid = (ct.cdiv(x_1d.numel(), tile_n), 1, 1)
    ct.launch(
        torch.cuda.current_stream(),
        grid,
        gelu_backward_kernel,
        (dy_1d, x_1d, dx_1d, sqrt_2_over_pi, tile_n),
    )
    return dx_1d.view(x.shape)


def cutile_softmax_backward(
    grad_out: torch.Tensor,
    y: torch.Tensor,
    dim: int = -1,
) -> torch.Tensor:
    """Compute dL/dx for Softmax output `y` via CuTile reduction."""
    prod = grad_out.float() * y.float()
    if y.is_cuda and y.dtype in (torch.float16, torch.bfloat16, torch.float32):
        dot = cutile_reduce_sum(prod, dim=dim, keepdim=True)
    else:
        dot = prod.sum(dim=dim, keepdim=True)
    return (y.float() * (grad_out.float() - dot)).to(y.dtype)


def cutile_layernorm_backward(
    grad_out: torch.Tensor,
    x: torch.Tensor,
    weight: torch.Tensor,
    eps: float = 1e-5,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute `(grad_x, grad_weight, grad_bias)` for LayerNorm over the last dimension."""
    orig_shape = x.shape
    d_model = orig_shape[-1]
    x_2d = x.contiguous().view(-1, d_model)
    dy_2d = grad_out.contiguous().view(-1, d_model)
    rows = x_2d.shape[0]

    x_f = x_2d.float()
    dy_f = dy_2d.float()
    mean = x_f.mean(dim=-1, keepdim=True)
    diff = x_f - mean
    inv_std = torch.rsqrt((diff * diff).mean(dim=-1, keepdim=True) + eps)
    x_hat = diff * inv_std

    grad_w = (dy_f * x_hat).sum(dim=0).to(weight.dtype)
    grad_b = dy_f.sum(dim=0).to(weight.dtype)

    if not x.is_cuda or x.dtype not in (torch.float16, torch.bfloat16, torch.float32):
        dy_w = dy_f * weight.float()
        mean_dy_w = dy_w.mean(dim=-1, keepdim=True)
        mean_dy_w_xhat = (dy_w * x_hat).mean(dim=-1, keepdim=True)
        dx_2d = (inv_std * (dy_w - mean_dy_w - x_hat * mean_dy_w_xhat)).to(x.dtype)
        return dx_2d.view(orig_shape), grad_w, grad_b

    d_pad = _next_pow2(d_model)
    dx_2d = torch.empty_like(x_2d)
    ct.launch(
        torch.cuda.current_stream(),
        (rows, 1, 1),
        layernorm_backward_dx_kernel,
        (
            dy_2d,
            x_2d,
            weight.contiguous(),
            dx_2d,
            float(eps),
            1.0 / float(d_model),
            int(d_model),
            d_pad,
        ),
    )
    return dx_2d.view(orig_shape), grad_w, grad_b
