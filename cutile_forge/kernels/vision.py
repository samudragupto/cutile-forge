"""Computer vision primitives (Depthwise Conv2D, Conv2D Backward, Pooling, PatchEmbed)."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from cutile_forge.kernels.bmm import cutile_bmm
from cutile_forge.kernels.indexing import cutile_conv2d
from cutile_forge.kernels.reductions import cutile_reduce_max, cutile_reduce_sum


def _pair(val: int | tuple[int, int]) -> tuple[int, int]:
    return (val, val) if isinstance(val, int) else val


def cutile_depthwise_conv2d(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None = None,
    stride: int | tuple[int, int] = 1,
    padding: int | tuple[int, int] = 0,
) -> torch.Tensor:
    """Compute depthwise 2D convolution (`groups == C_in`) backed by CuTile BMM."""
    c_in = x.shape[1]
    if (
        not x.is_cuda
        or x.dtype not in (torch.float16, torch.bfloat16, torch.float32)
        or weight.shape[0] != c_in
        or weight.shape[1] != 1
    ):
        return F.conv2d(x, weight, bias=bias, stride=stride, padding=padding, groups=c_in)

    n_batch, _, h_in, w_in = x.shape
    _, _, k_h, k_w = weight.shape
    s_h, s_w = _pair(stride)
    p_h, p_w = _pair(padding)

    h_out = (h_in + 2 * p_h - k_h) // s_h + 1
    w_out = (w_in + 2 * p_w - k_w) // s_w + 1
    l_out = h_out * w_out

    cols = F.unfold(x, kernel_size=(k_h, k_w), padding=(p_h, p_w), stride=(s_h, s_w))
    cols_chan = (
        cols.view(n_batch, c_in, k_h * k_w, l_out)
        .permute(1, 2, 0, 3)
        .contiguous()
        .view(c_in, k_h * k_w, n_batch * l_out)
    )
    w_chan = weight.view(c_in, 1, k_h * k_w)
    out = cutile_bmm(w_chan, cols_chan)
    out_4d = out.view(c_in, n_batch, h_out, w_out).transpose(0, 1).contiguous()
    if bias is not None:
        out_4d = out_4d + bias.view(1, c_in, 1, 1)
    return out_4d


def cutile_conv2d_backward(
    grad_out: torch.Tensor,
    x: torch.Tensor,
    weight: torch.Tensor,
    stride: int | tuple[int, int] = 1,
    padding: int | tuple[int, int] = 0,
    need_bias: bool = True,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
    """Compute `(dx, dweight, dbias)` for `conv2d` (`groups=1`) via im2col/col2im + CuTile BMM."""
    n_batch, _, h_in, w_in = x.shape
    c_out, _, k_h, k_w = weight.shape
    s_h, s_w = _pair(stride)
    p_h, p_w = _pair(padding)
    l_out = grad_out.shape[2] * grad_out.shape[3]

    dy_mat = grad_out.contiguous().view(n_batch, c_out, l_out)
    x_cols = F.unfold(x, kernel_size=(k_h, k_w), padding=(p_h, p_w), stride=(s_h, s_w))

    dw_per_sample = cutile_bmm(dy_mat, x_cols.transpose(1, 2).contiguous())
    dw = dw_per_sample.float().sum(dim=0).view(weight.shape).to(weight.dtype)

    w_flat = weight.view(c_out, -1).t().contiguous().unsqueeze(0).expand(n_batch, -1, c_out)
    dx_cols = cutile_bmm(w_flat, dy_mat)
    dx = F.fold(
        dx_cols.float(),
        output_size=(h_in, w_in),
        kernel_size=(k_h, k_w),
        padding=(p_h, p_w),
        stride=(s_h, s_w),
    ).to(x.dtype)

    db = grad_out.float().sum(dim=(0, 2, 3)).to(weight.dtype) if need_bias else None
    return dx, dw, db


def cutile_max_pool2d(
    x: torch.Tensor,
    kernel_size: int | tuple[int, int],
    stride: int | tuple[int, int] | None = None,
    padding: int | tuple[int, int] = 0,
) -> torch.Tensor:
    """Compute 2D max pooling via unfold + CuTile max reduction when eligible."""
    stride_val = kernel_size if stride is None else stride
    k_h, k_w = _pair(kernel_size)
    s_h, s_w = _pair(stride_val)
    p_h, p_w = _pair(padding)

    if (
        not x.is_cuda
        or x.dtype not in (torch.float16, torch.bfloat16, torch.float32)
        or p_h != 0
        or p_w != 0
    ):
        return F.max_pool2d(x, kernel_size=(k_h, k_w), stride=(s_h, s_w), padding=(p_h, p_w))

    n_batch, c_in, h_in, w_in = x.shape
    h_out = (h_in - k_h) // s_h + 1
    w_out = (w_in - k_w) // s_w + 1
    cols = F.unfold(x, kernel_size=(k_h, k_w), stride=(s_h, s_w))
    patches = cols.view(n_batch, c_in, k_h * k_w, h_out * w_out)
    return cutile_reduce_max(patches, dim=2).view(n_batch, c_in, h_out, w_out)


def cutile_avg_pool2d(
    x: torch.Tensor,
    kernel_size: int | tuple[int, int],
    stride: int | tuple[int, int] | None = None,
    padding: int | tuple[int, int] = 0,
) -> torch.Tensor:
    """Compute 2D average pooling via unfold + CuTile sum reduction when eligible."""
    stride_val = kernel_size if stride is None else stride
    k_h, k_w = _pair(kernel_size)
    s_h, s_w = _pair(stride_val)
    p_h, p_w = _pair(padding)

    if not x.is_cuda or x.dtype not in (torch.float16, torch.bfloat16, torch.float32):
        return F.avg_pool2d(x, kernel_size=(k_h, k_w), stride=(s_h, s_w), padding=(p_h, p_w))

    n_batch, c_in, h_in, w_in = x.shape
    h_out = (h_in + 2 * p_h - k_h) // s_h + 1
    w_out = (w_in + 2 * p_w - k_w) // s_w + 1
    cols = F.unfold(x, kernel_size=(k_h, k_w), padding=(p_h, p_w), stride=(s_h, s_w))
    patches = cols.view(n_batch, c_in, k_h * k_w, h_out * w_out)
    summed = cutile_reduce_sum(patches, dim=2)
    return (summed / float(k_h * k_w)).view(n_batch, c_in, h_out, w_out)


class CuTilePatchEmbed(nn.Module):
    """ViT-style patch embedding backed by `cutile_conv2d`."""

    def __init__(
        self,
        img_size: int = 224,
        patch_size: int = 16,
        in_chans: int = 3,
        embed_dim: int = 768,
    ) -> None:
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.in_chans = in_chans
        self.embed_dim = embed_dim
        self.num_patches = (img_size // patch_size) ** 2
        self.weight = nn.Parameter(torch.randn(embed_dim, in_chans, patch_size, patch_size) * 0.02)
        self.bias = nn.Parameter(torch.zeros(embed_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        proj = cutile_conv2d(x, self.weight, bias=self.bias, stride=self.patch_size, padding=0)
        return proj.flatten(2).transpose(1, 2).contiguous()
