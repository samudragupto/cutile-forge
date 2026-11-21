"""Training support kernels: PyTorch autograd Functions and fused AdamW optimizer step."""

from __future__ import annotations

from typing import Any

import torch

from cutile_forge.cuda_tile_init import ct
from cutile_forge.kernels.activations import (
    cutile_rmsnorm_backward,
    cutile_silu,
    cutile_silu_backward,
)
from cutile_forge.kernels.rmsnorm import cutile_rmsnorm


@ct.kernel(occupancy=2)
def adamw_step_kernel(
    param_ptr,
    grad_ptr,
    exp_avg_ptr,
    exp_avg_sq_ptr,
    lr: float,
    beta1: float,
    beta2: float,
    inv_bc1: float,
    inv_bc2: float,
    eps: float,
    weight_decay: float,
    tile_n: ct.Constant[int],
) -> None:
    """Single-launch fused AdamW parameter & moment update in CuTile."""
    pid = ct.bid(0)
    p = ct.astype(
        ct.load(param_ptr, index=(pid,), shape=(tile_n,), padding_mode=ct.PaddingMode.ZERO),
        ct.float32,
    )
    g = ct.astype(
        ct.load(grad_ptr, index=(pid,), shape=(tile_n,), padding_mode=ct.PaddingMode.ZERO),
        ct.float32,
    )
    m = ct.astype(
        ct.load(exp_avg_ptr, index=(pid,), shape=(tile_n,), padding_mode=ct.PaddingMode.ZERO),
        ct.float32,
    )
    v = ct.astype(
        ct.load(exp_avg_sq_ptr, index=(pid,), shape=(tile_n,), padding_mode=ct.PaddingMode.ZERO),
        ct.float32,
    )

    p = p * (1.0 - lr * weight_decay)
    m_new = beta1 * m + (1.0 - beta1) * g
    v_new = beta2 * v + (1.0 - beta2) * (g * g)

    m_hat = m_new * inv_bc1
    v_hat = v_new * inv_bc2
    p_new = p - lr * (m_hat / (ct.sqrt(v_hat) + eps))

    ct.store(param_ptr, index=(pid,), tile=ct.astype(p_new, param_ptr.dtype))
    ct.store(exp_avg_ptr, index=(pid,), tile=ct.astype(m_new, exp_avg_ptr.dtype))
    ct.store(exp_avg_sq_ptr, index=(pid,), tile=ct.astype(v_new, exp_avg_sq_ptr.dtype))


def cutile_adamw_step(
    param: torch.Tensor,
    grad: torch.Tensor,
    exp_avg: torch.Tensor,
    exp_avg_sq: torch.Tensor,
    step: int,
    lr: float = 1e-3,
    beta1: float = 0.9,
    beta2: float = 0.999,
    eps: float = 1e-8,
    weight_decay: float = 0.01,
) -> None:
    """Perform one in-place decoupled AdamW optimization step."""
    if step < 1:
        raise ValueError(f"step must be >= 1, got {step}")

    inv_bc1 = 1.0 / (1.0 - (beta1**step))
    inv_bc2 = 1.0 / (1.0 - (beta2**step))

    if not param.is_cuda or param.dtype not in (torch.float16, torch.bfloat16, torch.float32):
        p_f = param.detach().float()
        g_f = grad.detach().float()
        m_f = exp_avg.detach().float()
        v_f = exp_avg_sq.detach().float()
        p_f.mul_(1.0 - lr * weight_decay)
        m_f.mul_(beta1).add_(g_f, alpha=1.0 - beta1)
        v_f.mul_(beta2).addcmul_(g_f, g_f, value=1.0 - beta2)
        denom = (v_f * inv_bc2).sqrt().add_(eps)
        p_f.addcdiv_(m_f * inv_bc1, denom, value=-lr)
        param.detach().copy_(p_f.to(param.dtype))
        exp_avg.detach().copy_(m_f.to(exp_avg.dtype))
        exp_avg_sq.detach().copy_(v_f.to(exp_avg_sq.dtype))
        return

    p_1d = param.detach().contiguous().view(-1)
    g_1d = grad.detach().contiguous().view(-1)
    m_1d = exp_avg.detach().contiguous().view(-1)
    v_1d = exp_avg_sq.detach().contiguous().view(-1)

    tile_n = 1024
    grid = (ct.cdiv(p_1d.numel(), tile_n), 1, 1)
    ct.launch(
        torch.cuda.current_stream(),
        grid,
        adamw_step_kernel,
        (
            p_1d,
            g_1d,
            m_1d,
            v_1d,
            float(lr),
            float(beta1),
            float(beta2),
            float(inv_bc1),
            float(inv_bc2),
            float(eps),
            float(weight_decay),
            tile_n,
        ),
    )
    if not param.is_contiguous():
        param.detach().copy_(p_1d.view(param.shape))
        exp_avg.detach().copy_(m_1d.view(exp_avg.shape))
        exp_avg_sq.detach().copy_(v_1d.view(exp_avg_sq.shape))


class CuTileSiLUFn(torch.autograd.Function):
    """PyTorch autograd Function backed by CuTile forward and backward SiLU kernels."""

    @staticmethod
    def forward(ctx: Any, x: torch.Tensor) -> torch.Tensor:
        ctx.save_for_backward(x)
        return cutile_silu(x)

    @staticmethod
    def backward(ctx: Any, grad_out: torch.Tensor) -> torch.Tensor:
        (x,) = ctx.saved_tensors
        return cutile_silu_backward(grad_out, x)


class CuTileRMSNormFn(torch.autograd.Function):
    """PyTorch autograd Function backed by CuTile forward and backward RMSNorm kernels."""

    @staticmethod
    def forward(ctx: Any, x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
        ctx.save_for_backward(x, weight)
        ctx.eps = eps
        if x.is_cuda and x.dtype == torch.float16:
            return cutile_rmsnorm(x, weight, eps=eps)
        x_f = x.float()
        inv_rms = torch.rsqrt(x_f.pow(2).mean(dim=-1, keepdim=True) + eps)
        return (x_f * inv_rms * weight.float()).to(x.dtype)

    @staticmethod
    def backward(ctx: Any, grad_out: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, None]:
        x, weight = ctx.saved_tensors
        dx, dw = cutile_rmsnorm_backward(grad_out, x, weight, eps=ctx.eps)
        return dx, dw, None


def cutile_silu_autograd(x: torch.Tensor) -> torch.Tensor:
    """Autograd-compatible SiLU backed by CuTile."""
    return CuTileSiLUFn.apply(x)


def cutile_rmsnorm_autograd(
    x: torch.Tensor,
    weight: torch.Tensor,
    eps: float = 1e-5,
) -> torch.Tensor:
    """Autograd-compatible RMSNorm backed by CuTile."""
    return CuTileRMSNormFn.apply(x, weight, eps)
