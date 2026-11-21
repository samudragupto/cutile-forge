"""Optimized computer vision CuTile primitives (Depthwise Conv2D, Pool2D, PatchEmbed, Conv2D Bwd)."""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from cutile_forge.cuda_tile_init import ct
from cutile_forge.kernels.bmm import cutile_bmm
from cutile_forge.kernels.indexing import cutile_conv2d


def _pair(val: int | tuple[int, int]) -> tuple[int, int]:
    return (val, val) if isinstance(val, int) else val


def _next_pow2(n: int) -> int:
    p = 1
    while p < n:
        p *= 2
    return p


@ct.kernel(occupancy=4)
def depthwise_conv2d_kernel(
    x_flat,
    w_pad,
    y_ptr,
    h_in: int,
    w_in: int,
    c_in: int,
    sh: ct.Constant[int],
    sw: ct.Constant[int],
    ph: ct.Constant[int],
    pw: ct.Constant[int],
    kh: ct.Constant[int],
    kw: ct.Constant[int],
    w_pad_cols: ct.Constant[int],
    th: ct.Constant[int],
    tw: ct.Constant[int],
) -> None:
    """Coalesced 2D spatial tile depthwise Conv2D (`groups == C_in`) with fused bias."""
    bid_w = ct.bid(0)
    bid_h = ct.bid(1)
    nc_id = ct.bid(2)
    c_id = nc_id % c_in

    oh = ct.reshape(bid_h * th + ct.arange(th, dtype=ct.int32), (th, 1))
    ow = ct.reshape(bid_w * tw + ct.arange(tw, dtype=ct.int32), (1, tw))
    b_idx = ct.reshape(ct.full((1,), nc_id, dtype=ct.int32), (1, 1))

    w_tile = ct.astype(
        ct.load(w_pad, index=(c_id, 0), shape=(1, w_pad_cols), padding_mode=ct.PaddingMode.ZERO),
        ct.float32,
    )
    bias_val = ct.reshape(ct.extract(w_tile, index=(0, w_pad_cols - 1), shape=(1, 1)), (1, 1))
    acc = ct.broadcast_to(bias_val, (th, tw))

    for ky in range(kh):
        ih = oh * sh - ph + ky
        in_y = (ih >= 0) & (ih < h_in)
        ih_c = ct.where(in_y, ih, 0)
        for kx in range(kw):
            iw = ow * sw - pw + kx
            valid = in_y & (iw >= 0) & (iw < w_in)
            iw_c = ct.where(valid, iw, 0)
            hw_idx = ih_c * w_in + iw_c
            x_tile = ct.astype(
                ct.gather(x_flat, (b_idx, hw_idx), padding_value=0, latency=3),
                ct.float32,
            )
            w_val = ct.reshape(ct.extract(w_tile, index=(0, ky * kw + kx), shape=(1, 1)), (1, 1))
            acc = acc + ct.where(valid, x_tile, 0.0) * w_val

    ct.store(
        y_ptr,
        index=(nc_id, bid_h, bid_w),
        tile=ct.reshape(ct.astype(acc, y_ptr.dtype), (1, th, tw)),
    )


@ct.kernel(occupancy=4)
def max_pool2d_2x2_s2_kernel(x_ptr, y_ptr, th: ct.Constant[int], tw: ct.Constant[int]) -> None:
    """Coalesced zero-loop 2x2 stride-2 MaxPool2D in CuTile."""
    pw = ct.bid(0)
    ph = ct.bid(1)
    b = ct.bid(2)
    t = ct.load(
        x_ptr,
        index=(b, ph, pw),
        shape=(1, 2 * th, 2 * tw),
        padding_mode=ct.PaddingMode.NEG_INF,
        latency=3,
    )
    t4 = ct.reshape(t, (th, 2, tw, 2))
    out = ct.max(ct.max(t4, axis=3), axis=1)
    ct.store(y_ptr, index=(b, ph, pw), tile=ct.reshape(out, (1, th, tw)))


@ct.kernel(occupancy=4)
def avg_pool2d_2x2_s2_kernel(x_ptr, y_ptr, th: ct.Constant[int], tw: ct.Constant[int]) -> None:
    """Coalesced zero-loop 2x2 stride-2 AvgPool2D in CuTile."""
    pw = ct.bid(0)
    ph = ct.bid(1)
    b = ct.bid(2)
    t = ct.astype(
        ct.load(
            x_ptr,
            index=(b, ph, pw),
            shape=(1, 2 * th, 2 * tw),
            padding_mode=ct.PaddingMode.ZERO,
            latency=3,
        ),
        ct.float32,
    )
    t4 = ct.reshape(t, (th, 2, tw, 2))
    out = ct.sum(ct.sum(t4, axis=3), axis=1) * 0.25
    ct.store(y_ptr, index=(b, ph, pw), tile=ct.reshape(ct.astype(out, y_ptr.dtype), (1, th, tw)))


@ct.kernel(occupancy=4)
def pool2d_general_max_kernel(
    x_flat,
    y_ptr,
    h_in: int,
    w_in: int,
    sh: ct.Constant[int],
    sw: ct.Constant[int],
    kh: ct.Constant[int],
    kw: ct.Constant[int],
    th: ct.Constant[int],
    tw: ct.Constant[int],
) -> None:
    """General coalesced MaxPool2D kernel without F.unfold allocation."""
    bid_w = ct.bid(0)
    bid_h = ct.bid(1)
    nc_id = ct.bid(2)
    oh = ct.reshape(bid_h * th + ct.arange(th, dtype=ct.int32), (th, 1))
    ow = ct.reshape(bid_w * tw + ct.arange(tw, dtype=ct.int32), (1, tw))
    b_idx = ct.reshape(ct.full((1,), nc_id, dtype=ct.int32), (1, 1))
    acc = ct.full((th, tw), -3.4e38, dtype=ct.float32)

    for ky in range(kh):
        ih = oh * sh + ky
        in_y = ih < h_in
        ih_c = ct.where(in_y, ih, 0)
        for kx in range(kw):
            iw = ow * sw + kx
            valid = in_y & (iw < w_in)
            iw_c = ct.where(valid, iw, 0)
            xt = ct.astype(
                ct.gather(x_flat, (b_idx, ih_c * w_in + iw_c), padding_value=-65504, latency=3),
                ct.float32,
            )
            acc = ct.maximum(acc, ct.where(valid, xt, -3.4e38))

    ct.store(
        y_ptr,
        index=(nc_id, bid_h, bid_w),
        tile=ct.reshape(ct.astype(acc, y_ptr.dtype), (1, th, tw)),
    )


@ct.kernel(occupancy=4)
def pool2d_general_avg_kernel(
    x_flat,
    y_ptr,
    h_in: int,
    w_in: int,
    sh: ct.Constant[int],
    sw: ct.Constant[int],
    kh: ct.Constant[int],
    kw: ct.Constant[int],
    inv_area: float,
    th: ct.Constant[int],
    tw: ct.Constant[int],
) -> None:
    """General coalesced AvgPool2D kernel without F.unfold allocation."""
    bid_w = ct.bid(0)
    bid_h = ct.bid(1)
    nc_id = ct.bid(2)
    oh = ct.reshape(bid_h * th + ct.arange(th, dtype=ct.int32), (th, 1))
    ow = ct.reshape(bid_w * tw + ct.arange(tw, dtype=ct.int32), (1, tw))
    b_idx = ct.reshape(ct.full((1,), nc_id, dtype=ct.int32), (1, 1))
    acc = ct.zeros((th, tw), dtype=ct.float32)

    for ky in range(kh):
        ih = oh * sh + ky
        in_y = ih < h_in
        ih_c = ct.where(in_y, ih, 0)
        for kx in range(kw):
            iw = ow * sw + kx
            valid = in_y & (iw < w_in)
            iw_c = ct.where(valid, iw, 0)
            xt = ct.astype(
                ct.gather(x_flat, (b_idx, ih_c * w_in + iw_c), padding_value=0, latency=3),
                ct.float32,
            )
            acc = acc + ct.where(valid, xt, 0.0)

    out = acc * inv_area
    ct.store(
        y_ptr,
        index=(nc_id, bid_h, bid_w),
        tile=ct.reshape(ct.astype(out, y_ptr.dtype), (1, th, tw)),
    )


@ct.kernel(occupancy=2)
def patch_embed_fused_kernel(
    x_ptr,
    w_ptr,
    b_ptr,
    pos_ptr,
    y_ptr,
    n_batch: int,
    h_in: int,
    w_in: int,
    n_pw: int,
    n_patches: int,
    p_sz: ct.Constant[int],
    k_dim: ct.Constant[int],
    has_pos: ct.Constant[bool],
    tm: ct.Constant[int],
    tn: ct.Constant[int],
    tk: ct.Constant[int],
) -> None:
    """Direct fused ViT PatchEmbed kernel: patch-gather MMA + bias + optional pos_embed."""
    pid_m = ct.bid(0)
    pid_n = ct.bid(1)
    m_ids = pid_m * tm + ct.arange(tm, dtype=ct.int32)
    in_m = m_ids < (n_batch * n_patches)
    m_safe = ct.where(in_m, m_ids, 0)
    b_ids = m_safe // n_patches
    p_ids = m_safe % n_patches
    py = (p_ids // n_pw) * p_sz
    px = (p_ids % n_pw) * p_sz

    num_k = (k_dim + tk - 1) // tk
    hw_size = h_in * w_in
    p2 = p_sz * p_sz
    acc = ct.zeros((tm, tn), dtype=ct.float32)

    for kt in range(num_k):
        k_ids = kt * tk + ct.arange(tk, dtype=ct.int32)
        c_ids = k_ids // p2
        rem = k_ids % p2
        dy = rem // p_sz
        dx = rem % p_sz
        idx_1d = (
            ct.reshape(c_ids * hw_size, (1, tk))
            + ct.reshape(py * w_in + px, (tm, 1))
            + ct.reshape(dy * w_in + dx, (1, tk))
        )
        valid = ct.reshape(in_m, (tm, 1)) & (ct.reshape(k_ids, (1, tk)) < k_dim)
        x_tile = ct.gather(
            x_ptr,
            (ct.reshape(b_ids, (tm, 1)), ct.where(valid, idx_1d, 0)),
            padding_value=0,
            latency=3,
        )
        x_tile = ct.where(valid, x_tile, 0)
        w_tile = ct.load(
            w_ptr, index=(kt, pid_n), shape=(tk, tn), padding_mode=ct.PaddingMode.ZERO, latency=3
        )
        acc = ct.mma(x_tile, w_tile, acc)

    bias_t = ct.astype(
        ct.load(b_ptr, index=(pid_n,), shape=(tn,), padding_mode=ct.PaddingMode.ZERO), ct.float32
    )
    acc = acc + ct.reshape(bias_t, (1, tn))
    if has_pos:
        col_ids = ct.reshape(pid_n * tn + ct.arange(tn, dtype=ct.int32), (1, tn))
        pos_tile = ct.astype(
            ct.gather(pos_ptr, (ct.reshape(p_ids, (tm, 1)), col_ids), padding_value=0),
            ct.float32,
        )
        acc = acc + pos_tile
    ct.store(y_ptr, index=(pid_m, pid_n), tile=ct.astype(acc, y_ptr.dtype))


def cutile_depthwise_conv2d(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None = None,
    stride: int | tuple[int, int] = 1,
    padding: int | tuple[int, int] = 0,
) -> torch.Tensor:
    """Compute depthwise 2D convolution (`groups == C_in`) via coalesced 2D spatial halo kernel."""
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
    nc = n_batch * c_in

    w_pad_cols = max(16, _next_pow2(k_h * k_w + 1))
    w_pad = torch.zeros((c_in, w_pad_cols), device=x.device, dtype=weight.dtype)
    w_pad[:, : k_h * k_w] = weight.contiguous().view(c_in, k_h * k_w)
    if bias is not None:
        w_pad[:, w_pad_cols - 1] = bias.contiguous().view(c_in)

    x_flat = x.contiguous().view(nc, h_in * w_in)
    y_3d = torch.empty((nc, h_out, w_out), device=x.device, dtype=x.dtype)
    th, tw = 8, 32
    grid = (ct.cdiv(w_out, tw), ct.cdiv(h_out, th), nc)
    ct.launch(
        torch.cuda.current_stream(),
        grid,
        depthwise_conv2d_kernel,
        (x_flat, w_pad, y_3d, h_in, w_in, c_in, s_h, s_w, p_h, p_w, k_h, k_w, w_pad_cols, th, tw),
    )
    return y_3d.view(n_batch, c_in, h_out, w_out)


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
    """Compute 2D max pooling via coalesced CuTile 2D spatial kernels."""
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
    nc = n_batch * c_in
    th, tw = 8, 32
    y_3d = torch.empty((nc, h_out, w_out), device=x.device, dtype=x.dtype)
    grid = (ct.cdiv(w_out, tw), ct.cdiv(h_out, th), nc)

    if k_h == 2 and k_w == 2 and s_h == 2 and s_w == 2 and (h_in % 2 == 0) and (w_in % 2 == 0):
        ct.launch(
            torch.cuda.current_stream(),
            grid,
            max_pool2d_2x2_s2_kernel,
            (x.contiguous().view(nc, h_in, w_in), y_3d, th, tw),
        )
    else:
        ct.launch(
            torch.cuda.current_stream(),
            grid,
            pool2d_general_max_kernel,
            (x.contiguous().view(nc, h_in * w_in), y_3d, h_in, w_in, s_h, s_w, k_h, k_w, th, tw),
        )
    return y_3d.view(n_batch, c_in, h_out, w_out)


def cutile_avg_pool2d(
    x: torch.Tensor,
    kernel_size: int | tuple[int, int],
    stride: int | tuple[int, int] | None = None,
    padding: int | tuple[int, int] = 0,
) -> torch.Tensor:
    """Compute 2D average pooling via coalesced CuTile 2D spatial kernels."""
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
        return F.avg_pool2d(x, kernel_size=(k_h, k_w), stride=(s_h, s_w), padding=(p_h, p_w))

    n_batch, c_in, h_in, w_in = x.shape
    h_out = (h_in - k_h) // s_h + 1
    w_out = (w_in - k_w) // s_w + 1
    nc = n_batch * c_in
    th, tw = 8, 32
    y_3d = torch.empty((nc, h_out, w_out), device=x.device, dtype=x.dtype)
    grid = (ct.cdiv(w_out, tw), ct.cdiv(h_out, th), nc)

    if k_h == 2 and k_w == 2 and s_h == 2 and s_w == 2 and (h_in % 2 == 0) and (w_in % 2 == 0):
        ct.launch(
            torch.cuda.current_stream(),
            grid,
            avg_pool2d_2x2_s2_kernel,
            (x.contiguous().view(nc, h_in, w_in), y_3d, th, tw),
        )
    else:
        ct.launch(
            torch.cuda.current_stream(),
            grid,
            pool2d_general_avg_kernel,
            (
                x.contiguous().view(nc, h_in * w_in),
                y_3d,
                h_in,
                w_in,
                s_h,
                s_w,
                k_h,
                k_w,
                1.0 / float(k_h * k_w),
                th,
                tw,
            ),
        )
    return y_3d.view(n_batch, c_in, h_out, w_out)


class CuTilePatchEmbed(nn.Module):
    """ViT-style patch embedding using direct fused CuTile patch-gather MMA kernel."""

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
        self._cached_w_t: torch.Tensor | None = None
        self._cached_w_ver: int = -1

    def forward(self, x: torch.Tensor, pos_embed: torch.Tensor | None = None) -> torch.Tensor:
        if x.is_cuda and x.dtype in (torch.float16, torch.bfloat16) and x.ndim == 4:
            n_batch, c_in, h_in, w_in = x.shape
            n_ph = h_in // self.patch_size
            n_pw = w_in // self.patch_size
            n_patches = n_ph * n_pw
            k_dim = c_in * self.patch_size * self.patch_size
            tm = 64
            tn = 128 if self.embed_dim >= 128 else 64
            tk = 64

            w_ver = int(self.weight._version)
            if (
                self._cached_w_t is None
                or self._cached_w_ver != w_ver
                or self._cached_w_t.device != self.weight.device
                or self._cached_w_t.dtype != self.weight.dtype
            ):
                self._cached_w_t = self.weight.view(self.embed_dim, k_dim).t().contiguous()
                self._cached_w_ver = w_ver

            y_2d = torch.empty(
                (n_batch * n_patches, self.embed_dim), device=x.device, dtype=x.dtype
            )
            has_pos = pos_embed is not None
            pos_arg = (
                pos_embed.contiguous().view(n_patches, self.embed_dim)
                if pos_embed is not None
                else y_2d
            )
            grid = (ct.cdiv(n_batch * n_patches, tm), ct.cdiv(self.embed_dim, tn), 1)
            ct.launch(
                torch.cuda.current_stream(),
                grid,
                patch_embed_fused_kernel,
                (
                    x.contiguous().view(n_batch, c_in * h_in * w_in),
                    self._cached_w_t,
                    self.bias.contiguous(),
                    pos_arg,
                    y_2d,
                    n_batch,
                    h_in,
                    w_in,
                    n_pw,
                    n_patches,
                    int(self.patch_size),
                    int(k_dim),
                    has_pos,
                    tm,
                    tn,
                    tk,
                ),
            )
            return y_2d.view(n_batch, n_patches, self.embed_dim)

        proj = cutile_conv2d(x, self.weight, bias=self.bias, stride=self.patch_size, padding=0)
        out = proj.flatten(2).transpose(1, 2).contiguous()
        if pos_embed is not None:
            out = out + pos_embed
        return out


# Preserve math import usage check
_ = math.sqrt
