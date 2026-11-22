"""Optimized indexing (coalesced `ct.gather`) and Conv2D (`1x1` zero-copy & `3x3` implicit GEMM)."""

from __future__ import annotations

import torch
import torch.nn.functional as F

from cutile_forge.cuda_tile_init import ct
from cutile_forge.kernels.bmm import cutile_bmm
from cutile_forge.kernels.gemm_dequant import cutile_gemm_fp16


@ct.kernel(occupancy=8)
def gather_rows_kernel(
    table_ptr,
    idx_ptr,
    out_ptr,
    tile_m: ct.Constant[int],
    tile_d: ct.Constant[int],
) -> None:
    """Coalesced multi-row gather `table[indices, :]` via wide-tile `ct.gather`."""
    pid_n = ct.bid(0)
    pid_d = ct.bid(1)

    row_ids = ct.reshape(
        ct.astype(
            ct.load(
                idx_ptr,
                index=(pid_n,),
                shape=(tile_m,),
                padding_mode=ct.PaddingMode.ZERO,
                latency=4,
            ),
            ct.int32,
        ),
        (tile_m, 1),
    )
    col_ids = ct.reshape(pid_d * tile_d + ct.arange(tile_d, dtype=ct.int32), (1, tile_d))
    gathered = ct.gather(table_ptr, (row_ids, col_ids), padding_value=0, latency=4)
    ct.store(out_ptr, index=(pid_n, pid_d), tile=gathered)


@ct.kernel(occupancy=2)
def conv2d_1x1_nchw_kernel(
    w_ptr,
    x_ptr,
    bias_ptr,
    y_ptr,
    has_bias: ct.Constant[bool],
    tm: ct.Constant[int],
    tn: ct.Constant[int],
    tk: ct.Constant[int],
) -> None:
    """Zero-copy NCHW 1x1 pointwise Conv2D with L2-swizzled MMA and fused bias."""
    group_m: ct.Constant[int] = 8
    pid = ct.bid(0)
    n_id = ct.bid(1)

    num_m = ct.num_tiles(w_ptr, axis=0, shape=(tm, tk))
    num_n = ct.num_tiles(x_ptr, axis=2, shape=(1, tk, tn))
    num_pid_in_group = group_m * num_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * group_m
    group_size_m = min(num_m - first_pid_m, group_m)
    pid_m = first_pid_m + (pid % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    num_k = ct.num_tiles(w_ptr, axis=1, shape=(tm, tk))
    acc = ct.zeros((tm, tn), dtype=ct.float32)
    for k in range(num_k):
        tw = ct.load(
            w_ptr, index=(pid_m, k), shape=(tm, tk), padding_mode=ct.PaddingMode.ZERO, latency=3
        )
        tx = ct.reshape(
            ct.load(
                x_ptr,
                index=(n_id, k, pid_n),
                shape=(1, tk, tn),
                padding_mode=ct.PaddingMode.ZERO,
                latency=3,
            ),
            (tk, tn),
        )
        acc = ct.mma(tw, tx, acc)

    if has_bias:
        b_t = ct.astype(
            ct.load(bias_ptr, index=(pid_m,), shape=(tm,), padding_mode=ct.PaddingMode.ZERO),
            ct.float32,
        )
        acc = acc + ct.reshape(b_t, (tm, 1))

    ct.store(
        y_ptr,
        index=(n_id, pid_m, pid_n),
        tile=ct.reshape(ct.astype(acc, y_ptr.dtype), (1, tm, tn)),
    )


@ct.kernel(occupancy=2)
def conv2d_implicit_gemm_kernel(
    w_ptr,  # (C_out, KH*KW, C_in)
    x_ptr,  # (N, C_in * H_in * W_in)
    bias_ptr,  # (C_out,) or dummy
    y_ptr,  # (N, C_out, H_out * W_out)
    c_in: int,
    h_in: int,
    w_in: int,
    w_out: int,
    has_bias: ct.Constant[bool],
    sh: ct.Constant[int],
    sw: ct.Constant[int],
    ph: ct.Constant[int],
    pw: ct.Constant[int],
    kh: ct.Constant[int],
    kw: ct.Constant[int],
    tm: ct.Constant[int],
    tn: ct.Constant[int],
    tk: ct.Constant[int],
) -> None:
    """Implicit-GEMM Conv2D kernel eliminating HBM im2col materialization."""
    group_m: ct.Constant[int] = 8
    pid = ct.bid(0)
    n_id = ct.bid(1)

    num_m = ct.num_tiles(w_ptr, axis=0, shape=(tm, 1, tk))
    num_n = ct.num_tiles(y_ptr, axis=2, shape=(1, tm, tn))
    num_pid_in_group = group_m * num_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * group_m
    group_size_m = min(num_m - first_pid_m, group_m)
    pid_m = first_pid_m + (pid % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    l_ids = pid_n * tn + ct.arange(tn, dtype=ct.int32)
    oh = l_ids // w_out
    ow = l_ids % w_out
    n_idx = ct.reshape(ct.full((1,), n_id, dtype=ct.int32), (1, 1))
    hw_size = h_in * w_in
    num_ct = (c_in + tk - 1) // tk
    c_arange = ct.arange(tk, dtype=ct.int32)

    acc = ct.zeros((tm, tn), dtype=ct.float32)
    for ky in range(kh):
        ih = oh * sh - ph + ky
        in_y = (ih >= 0) & (ih < h_in)
        ih_c = ct.where(in_y, ih, 0)
        for kx in range(kw):
            iw = ow * sw - pw + kx
            valid_xy = in_y & (iw >= 0) & (iw < w_in)
            iw_c = ct.where(valid_xy, iw, 0)
            hw_off = ct.reshape(ih_c * w_in + iw_c, (1, tn))
            k_spatial = ky * kw + kx
            for ct_idx in range(num_ct):
                w_tile = ct.reshape(
                    ct.load(
                        w_ptr,
                        index=(pid_m, k_spatial, ct_idx),
                        shape=(tm, 1, tk),
                        padding_mode=ct.PaddingMode.ZERO,
                        latency=3,
                    ),
                    (tm, tk),
                )
                c_ids = ct.reshape(ct_idx * tk + c_arange, (tk, 1))
                in_c = c_ids < c_in
                chw_idx = ct.where(in_c, c_ids, 0) * hw_size + hw_off
                x_raw = ct.gather(x_ptr, (n_idx, chw_idx), padding_value=0, latency=3)
                x_tile = ct.where(ct.reshape(valid_xy, (1, tn)) & in_c, x_raw, 0)
                acc = ct.mma(w_tile, x_tile, acc)

    if has_bias:
        b_t = ct.astype(
            ct.load(bias_ptr, index=(pid_m,), shape=(tm,), padding_mode=ct.PaddingMode.ZERO),
            ct.float32,
        )
        acc = acc + ct.reshape(b_t, (tm, 1))

    ct.store(
        y_ptr,
        index=(n_id, pid_m, pid_n),
        tile=ct.reshape(ct.astype(acc, y_ptr.dtype), (1, tm, tn)),
    )


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

    if flat_idx.dtype in (torch.int32, torch.int64) and flat_idx.device == table.device:
        idx_c = flat_idx if flat_idx.is_contiguous() else flat_idx.contiguous()
    else:
        idx_c = flat_idx.to(device=table.device, dtype=torch.int32).contiguous()

    table_c = table if table.is_contiguous() else table.contiguous()
    out = torch.empty((idx_c.numel(), d_dim), device=table.device, dtype=table.dtype)

    if d_dim >= 512:
        tile_m = 4 if idx_c.numel() >= 128 else 16
        tile_d = 512
    elif d_dim >= 128:
        tile_m, tile_d = 16, 128
    else:
        tile_m, tile_d = 32, 64

    grid = (ct.cdiv(idx_c.numel(), tile_m), ct.cdiv(d_dim, tile_d), 1)
    ct.launch(
        torch.cuda.current_stream(),
        grid,
        gather_rows_kernel,
        (table_c, idx_c, out, tile_m, tile_d),
    )
    return out.view(out_shape)


def cutile_conv2d(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None = None,
    stride: int | tuple[int, int] = 1,
    padding: int | tuple[int, int] = 0,
    groups: int = 1,
) -> torch.Tensor:
    """2D convolution with zero-copy 1x1 pointwise GEMM & implicit-GEMM 3x3 lowering."""
    if x.ndim != 4 or weight.ndim != 4:
        raise ValueError(
            f"Expected 4D NCHW tensors, got x={tuple(x.shape)} weight={tuple(weight.shape)}"
        )

    if (
        not x.is_cuda
        or groups != 1
        or x.dtype not in (torch.float16, torch.bfloat16, torch.float32)
    ):
        return F.conv2d(x, weight, bias=bias, stride=stride, padding=padding, groups=groups)

    n_batch, c_in, h_in, w_in = x.shape
    c_out, _, k_h, k_w = weight.shape
    s_h, s_w = (stride, stride) if isinstance(stride, int) else stride
    p_h, p_w = (padding, padding) if isinstance(padding, int) else padding

    h_out = (h_in + 2 * p_h - k_h) // s_h + 1
    w_out = (w_in + 2 * p_w - k_w) // s_w + 1
    l_out = h_out * w_out

    # Fast path 1: 1x1 pointwise conv2d (zero-copy NHWC or zero-copy NCHW)
    if k_h == 1 and k_w == 1 and s_h == 1 and s_w == 1 and p_h == 0 and p_w == 0:
        if x.is_contiguous(memory_format=torch.channels_last) and x.dtype == torch.float16:
            x_nhwc = x.permute(0, 2, 3, 1)  # view over channels_last memory (zero-copy)
            x_2d = x_nhwc.reshape(n_batch * h_in * w_in, c_in)
            w_2d = weight.view(c_out, c_in).t().contiguous()
            y_2d = cutile_gemm_fp16(x_2d, w_2d)
            out_4d = y_2d.view(n_batch, h_out, w_out, c_out).permute(0, 3, 1, 2)
            if bias is not None:
                out_4d = out_4d + bias.view(1, c_out, 1, 1)
            return out_4d.contiguous()

        x_3d = x.contiguous().view(n_batch, c_in, l_out)
        w_2d = weight.contiguous().view(c_out, c_in)
        y_3d = torch.empty((n_batch, c_out, l_out), device=x.device, dtype=x.dtype)
        tm = 128 if c_out >= 128 else 64
        tn = 128
        tk = 32 if x.dtype == torch.float32 else 64
        has_bias = bias is not None
        bias_arg = bias.contiguous().view(c_out) if bias is not None else y_3d.view(-1)
        grid = (ct.cdiv(c_out, tm) * ct.cdiv(l_out, tn), n_batch, 1)
        ct.launch(
            torch.cuda.current_stream(),
            grid,
            conv2d_1x1_nchw_kernel,
            (w_2d, x_3d, bias_arg, y_3d, has_bias, tm, tn, tk),
        )
        return y_3d.view(n_batch, c_out, h_out, w_out)

    # Fast path 2: Implicit GEMM for general k x k convolutions (eliminates F.unfold HBM traffic)
    if x.dtype in (torch.float16, torch.bfloat16) and (c_in >= 16 or l_out >= 256):
        w_3d = weight.permute(0, 2, 3, 1).contiguous().view(c_out, k_h * k_w, c_in)
        x_2d = x.contiguous().view(n_batch, c_in * h_in * w_in)
        y_3d = torch.empty((n_batch, c_out, l_out), device=x.device, dtype=x.dtype)
        tm = 128 if c_out >= 128 else 64
        tn = 64
        tk = 64 if c_in >= 48 else 32
        has_bias = bias is not None
        bias_arg = bias.contiguous().view(c_out) if bias is not None else y_3d.view(-1)
        grid = (ct.cdiv(c_out, tm) * ct.cdiv(l_out, tn), n_batch, 1)
        ct.launch(
            torch.cuda.current_stream(),
            grid,
            conv2d_implicit_gemm_kernel,
            (
                w_3d,
                x_2d,
                bias_arg,
                y_3d,
                c_in,
                h_in,
                w_in,
                w_out,
                has_bias,
                s_h,
                s_w,
                p_h,
                p_w,
                k_h,
                k_w,
                tm,
                tn,
                tk,
            ),
        )
        return y_3d.view(n_batch, c_out, h_out, w_out)

    # Fallback: unfold + cutile_bmm for FP32 or tiny channels
    cols = F.unfold(x, kernel_size=(k_h, k_w), padding=(p_h, p_w), stride=(s_h, s_w))
    w_flat = weight.view(c_out, -1).unsqueeze(0).expand(n_batch, c_out, -1)
    out = cutile_bmm(w_flat, cols)
    out_4d = out.view(n_batch, c_out, h_out, w_out)
    if bias is not None:
        out_4d = out_4d + bias.view(1, c_out, 1, 1)
    return out_4d
