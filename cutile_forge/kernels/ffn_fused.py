"""Fused FFN kernels in CuTile: GEMM + SiLU + GEMM (and 4-bit / SwiGLU variants)."""

from __future__ import annotations

import cuda.tile as ct
import torch


@ct.kernel(occupancy=2)
def ffn_up_silu_down_kernel(
    x_ptr,
    wu_ptr,
    wd_ptr,
    y_part_ptr,
    m_pad: ct.Constant[int],
    tm: ct.Constant[int],
    th: ct.Constant[int],
    tk: ct.Constant[int],
    td: ct.Constant[int],
) -> None:
    """Single-kernel FFN: Y = SiLU(X @ W_up) @ W_down without writing intermediate H to HBM."""
    pid_m = ct.bid(0)
    pid_h = ct.bid(1)

    num_k = ct.num_tiles(x_ptr, axis=1, shape=(tm, tk))
    num_d = ct.num_tiles(wd_ptr, axis=1, shape=(th, td))

    acc_up = ct.zeros((tm, th), dtype=ct.float32)
    for k in range(num_k):
        tile_x = ct.load(
            x_ptr, index=(pid_m, k), shape=(tm, tk), padding_mode=ct.PaddingMode.ZERO, latency=3
        )
        tile_wu = ct.load(
            wu_ptr,
            index=(k, pid_h),
            shape=(tk, th),
            padding_mode=ct.PaddingMode.ZERO,
            latency=3,
        )
        acc_up = ct.mma(tile_x, tile_wu, acc_up)

    sig = 1.0 / (1.0 + ct.exp(-acc_up))
    act = ct.astype(acc_up * sig, ct.float16)

    row_tile_offset = pid_h * (m_pad // tm) + pid_m
    for d_tile in range(num_d):
        tile_wd = ct.load(
            wd_ptr,
            index=(pid_h, d_tile),
            shape=(th, td),
            padding_mode=ct.PaddingMode.ZERO,
            latency=3,
        )
        out = ct.mma(act, tile_wd, ct.zeros((tm, td), dtype=ct.float32))
        ct.store(y_part_ptr, index=(row_tile_offset, d_tile), tile=ct.astype(out, ct.float16))


@ct.kernel(occupancy=2)
def ffn_swiglu_kernel(
    x_ptr,
    wg_ptr,
    wu_ptr,
    wd_ptr,
    y_part_ptr,
    m_pad: ct.Constant[int],
    tm: ct.Constant[int],
    th: ct.Constant[int],
    tk: ct.Constant[int],
    td: ct.Constant[int],
) -> None:
    """Single-kernel SwiGLU FFN: Y = (SiLU(X @ W_gate) * (X @ W_up)) @ W_down."""
    pid_m = ct.bid(0)
    pid_h = ct.bid(1)

    num_k = ct.num_tiles(x_ptr, axis=1, shape=(tm, tk))
    num_d = ct.num_tiles(wd_ptr, axis=1, shape=(th, td))

    gate_acc = ct.zeros((tm, th), dtype=ct.float32)
    up_acc = ct.zeros((tm, th), dtype=ct.float32)
    for k in range(num_k):
        tile_x = ct.load(
            x_ptr, index=(pid_m, k), shape=(tm, tk), padding_mode=ct.PaddingMode.ZERO, latency=3
        )
        tile_wg = ct.load(
            wg_ptr,
            index=(k, pid_h),
            shape=(tk, th),
            padding_mode=ct.PaddingMode.ZERO,
            latency=3,
        )
        tile_wu = ct.load(
            wu_ptr,
            index=(k, pid_h),
            shape=(tk, th),
            padding_mode=ct.PaddingMode.ZERO,
            latency=3,
        )
        gate_acc = ct.mma(tile_x, tile_wg, gate_acc)
        up_acc = ct.mma(tile_x, tile_wu, up_acc)

    sig = 1.0 / (1.0 + ct.exp(-gate_acc))
    act = ct.astype((gate_acc * sig) * up_acc, ct.float16)

    row_tile_offset = pid_h * (m_pad // tm) + pid_m
    for d_tile in range(num_d):
        tile_wd = ct.load(
            wd_ptr,
            index=(pid_h, d_tile),
            shape=(th, td),
            padding_mode=ct.PaddingMode.ZERO,
            latency=3,
        )
        out = ct.mma(act, tile_wd, ct.zeros((tm, td), dtype=ct.float32))
        ct.store(y_part_ptr, index=(row_tile_offset, d_tile), tile=ct.astype(out, ct.float16))


@ct.kernel(occupancy=2)
def ffn_dequant_up_silu_down_kernel(
    x_ptr,
    qwu_ptr,
    su_ptr,
    zu_ptr,
    qwd_ptr,
    sd_ptr,
    zd_ptr,
    y_part_ptr,
    m_pad: ct.Constant[int],
    tm: ct.Constant[int],
    th: ct.Constant[int],
    tk: ct.Constant[int],
    td: ct.Constant[int],
    gs: ct.Constant[int],
) -> None:
    """Single-kernel 4-bit dequantized FFN: Y = SiLU(X @ dequant(W_up)) @ dequant(W_down)."""
    pid_m = ct.bid(0)
    pid_h = ct.bid(1)

    num_k = ct.num_tiles(x_ptr, axis=1, shape=(tm, tk))
    num_d = ct.num_tiles(qwd_ptr, axis=1, shape=(th // 2, td))
    half_tk: ct.Constant[int] = tk // 2
    half_th: ct.Constant[int] = th // 2

    acc_up = ct.zeros((tm, th), dtype=ct.float32)
    for k in range(num_k):
        tile_x = ct.load(
            x_ptr, index=(pid_m, k), shape=(tm, tk), padding_mode=ct.PaddingMode.ZERO, latency=3
        )
        qwu = ct.load(
            qwu_ptr,
            index=(k, pid_h),
            shape=(half_tk, th),
            padding_mode=ct.PaddingMode.ZERO,
            latency=3,
        )
        lo = ct.astype(ct.bitwise_and(qwu, 0x0F), ct.float16)
        hi = ct.astype(ct.bitwise_and(ct.bitwise_rshift(qwu, 4), 0x0F), ct.float16)
        w_int = ct.reshape(
            ct.cat((ct.reshape(lo, (half_tk, 1, th)), ct.reshape(hi, (half_tk, 1, th))), axis=1),
            (tk, th),
        )
        g_idx = (k * tk) // gs
        su = ct.load(
            su_ptr, index=(g_idx, pid_h), shape=(1, th), padding_mode=ct.PaddingMode.ZERO, latency=3
        )
        zu = ct.load(
            zu_ptr, index=(g_idx, pid_h), shape=(1, th), padding_mode=ct.PaddingMode.ZERO, latency=3
        )
        wu_fp16 = (w_int - zu) * su
        acc_up = ct.mma(tile_x, wu_fp16, acc_up)

    sig = 1.0 / (1.0 + ct.exp(-acc_up))
    act = ct.astype(acc_up * sig, ct.float16)

    g_h = (pid_h * th) // gs
    row_tile_offset = pid_h * (m_pad // tm) + pid_m
    for d_tile in range(num_d):
        qwd = ct.load(
            qwd_ptr,
            index=(pid_h, d_tile),
            shape=(half_th, td),
            padding_mode=ct.PaddingMode.ZERO,
            latency=3,
        )
        lo_d = ct.astype(ct.bitwise_and(qwd, 0x0F), ct.float16)
        hi_d = ct.astype(ct.bitwise_and(ct.bitwise_rshift(qwd, 4), 0x0F), ct.float16)
        wd_int = ct.reshape(
            ct.cat(
                (ct.reshape(lo_d, (half_th, 1, td)), ct.reshape(hi_d, (half_th, 1, td))), axis=1
            ),
            (th, td),
        )
        sd = ct.load(
            sd_ptr, index=(g_h, d_tile), shape=(1, td), padding_mode=ct.PaddingMode.ZERO, latency=3
        )
        zd = ct.load(
            zd_ptr, index=(g_h, d_tile), shape=(1, td), padding_mode=ct.PaddingMode.ZERO, latency=3
        )
        wd_fp16 = (wd_int - zd) * sd
        out = ct.mma(act, wd_fp16, ct.zeros((tm, td), dtype=ct.float32))
        ct.store(y_part_ptr, index=(row_tile_offset, d_tile), tile=ct.astype(out, ct.float16))


def fused_ffn_up_silu_down(
    x: torch.Tensor,
    w_up: torch.Tensor,
    w_down: torch.Tensor,
    workspace: torch.Tensor | None = None,
) -> torch.Tensor:
    """Compute SiLU(x @ w_up) @ w_down in a single CuTile kernel + partial reduction."""
    m, _ = x.shape
    h, d = w_down.shape
    tm = 16 if m <= 16 else 64
    th, tk, td = 128, 64, 128
    nh = (h + th - 1) // th
    m_pad = ((m + tm - 1) // tm) * tm

    if workspace is None or workspace.shape[0] < nh * m_pad or workspace.shape[1] != d:
        y_part = torch.empty((nh * m_pad, d), device=x.device, dtype=torch.float16)
    else:
        y_part = workspace[: nh * m_pad, :]

    grid = (m_pad // tm, nh, 1)
    ct.launch(
        torch.cuda.current_stream(),
        grid,
        ffn_up_silu_down_kernel,
        (x.contiguous(), w_up.contiguous(), w_down.contiguous(), y_part, m_pad, tm, th, tk, td),
    )
    return y_part.view(nh, m_pad, d)[:, :m, :].sum(dim=0)


def fused_ffn_swiglu(
    x: torch.Tensor,
    w_gate: torch.Tensor,
    w_up: torch.Tensor,
    w_down: torch.Tensor,
    workspace: torch.Tensor | None = None,
) -> torch.Tensor:
    """Compute (SiLU(x @ w_gate) * (x @ w_up)) @ w_down in a single CuTile kernel."""
    m, _ = x.shape
    h, d = w_down.shape
    tm = 16 if m <= 16 else 64
    th, tk, td = 128, 64, 128
    nh = (h + th - 1) // th
    m_pad = ((m + tm - 1) // tm) * tm

    if workspace is None or workspace.shape[0] < nh * m_pad or workspace.shape[1] != d:
        y_part = torch.empty((nh * m_pad, d), device=x.device, dtype=torch.float16)
    else:
        y_part = workspace[: nh * m_pad, :]

    grid = (m_pad // tm, nh, 1)
    ct.launch(
        torch.cuda.current_stream(),
        grid,
        ffn_swiglu_kernel,
        (
            x.contiguous(),
            w_gate.contiguous(),
            w_up.contiguous(),
            w_down.contiguous(),
            y_part,
            m_pad,
            tm,
            th,
            tk,
            td,
        ),
    )
    return y_part.view(nh, m_pad, d)[:, :m, :].sum(dim=0)


def fused_ffn_dequant_up_silu_down(
    x: torch.Tensor,
    qw_up: torch.Tensor,
    s_up: torch.Tensor,
    z_up: torch.Tensor,
    qw_down: torch.Tensor,
    s_down: torch.Tensor,
    z_down: torch.Tensor,
    group_size: int = 128,
    workspace: torch.Tensor | None = None,
) -> torch.Tensor:
    """4-bit quantized FFN: SiLU(x @ dequant(qw_up)) @ dequant(qw_down).

    Uses single-kernel tile fusion for small/medium sequences (m <= 256) to
    eliminate intermediate HBM roundtrips, and switches to 2-stage fused
    dequant+SiLU -> dequant for larger batches to bound 8 GB VRAM footprint.
    """
    m, _ = x.shape
    if m > 256:
        from cutile_forge.kernels.gemm_dequant import fused_gemm_dequant
        from cutile_forge.kernels.gemm_dequant_silu import fused_gemm_dequant_silu

        h_act = fused_gemm_dequant_silu(x, qw_up, s_up, z_up, group_size=group_size)
        return fused_gemm_dequant(h_act, qw_down, s_down, z_down, group_size=group_size)

    h = qw_down.shape[0] * 2
    d = qw_down.shape[1]
    tm = 16 if m <= 16 else (64 if m <= 64 else 128)
    th = 128 if group_size >= 128 else 64
    tk = 64
    td = 128
    nh = (h + th - 1) // th
    m_pad = ((m + tm - 1) // tm) * tm

    if workspace is None or workspace.shape[0] < nh * m_pad or workspace.shape[1] != d:
        y_part = torch.empty((nh * m_pad, d), device=x.device, dtype=torch.float16)
    else:
        y_part = workspace[: nh * m_pad, :]

    grid = (m_pad // tm, nh, 1)
    ct.launch(
        torch.cuda.current_stream(),
        grid,
        ffn_dequant_up_silu_down_kernel,
        (
            x.contiguous(),
            qw_up.contiguous(),
            s_up.contiguous(),
            z_up.contiguous(),
            qw_down.contiguous(),
            s_down.contiguous(),
            z_down.contiguous(),
            y_part,
            m_pad,
            tm,
            th,
            tk,
            td,
            group_size,
        ),
    )
    return y_part.view(nh, m_pad, d)[:, :m, :].sum(dim=0)

