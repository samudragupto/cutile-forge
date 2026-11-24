"""Fused GEMM + INT4 dequant + SiLU activation CuTile kernel."""

from __future__ import annotations

import cuda.tile as ct
import torch

from cutile_forge.kernels.gemm_dequant import _select_gemm_tiles, pack_int4_weights


@ct.kernel(occupancy=2)
def gemm_fp16_silu_kernel(
    a_ptr,
    b_ptr,
    c_ptr,
    tm: ct.Constant[int],
    tn: ct.Constant[int],
    tk: ct.Constant[int],
) -> None:
    """Fused FP16 GEMM + SiLU epilogue in a single CuTile kernel."""
    group_m: ct.Constant[int] = 8
    pid = ct.bid(0)
    num_m = ct.num_tiles(a_ptr, axis=0, shape=(tm, tk))
    num_n = ct.num_tiles(b_ptr, axis=1, shape=(tk, tn))
    num_pid_in_group = group_m * num_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * group_m
    group_size_m = min(num_m - first_pid_m, group_m)
    pid_m = first_pid_m + (pid % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    num_k = ct.num_tiles(a_ptr, axis=1, shape=(tm, tk))
    acc = ct.zeros((tm, tn), dtype=ct.float32)
    for k in range(num_k):
        tile_a = ct.load(
            a_ptr, index=(pid_m, k), shape=(tm, tk), padding_mode=ct.PaddingMode.ZERO, latency=3
        )
        tile_b = ct.load(
            b_ptr, index=(k, pid_n), shape=(tk, tn), padding_mode=ct.PaddingMode.ZERO, latency=3
        )
        acc = ct.mma(tile_a, tile_b, acc)

    sig = 1.0 / (1.0 + ct.exp(-acc))
    out = acc * sig
    ct.store(c_ptr, index=(pid_m, pid_n), tile=ct.astype(out, ct.float16))


@ct.kernel(occupancy=2)
def gemm_dequant_silu_kernel(
    a_ptr,
    qw_ptr,
    scales_ptr,
    zeros_ptr,
    c_ptr,
    tm: ct.Constant[int],
    tn: ct.Constant[int],
    tk: ct.Constant[int],
    gs: ct.Constant[int],
) -> None:
    """Fused INT4-dequant GEMM + SiLU activation in registers before HBM store."""
    group_m: ct.Constant[int] = 8
    pid = ct.bid(0)
    num_m = ct.num_tiles(a_ptr, axis=0, shape=(tm, tk))
    num_n = ct.num_tiles(qw_ptr, axis=1, shape=(tk // 2, tn))
    num_pid_in_group = group_m * num_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * group_m
    group_size_m = min(num_m - first_pid_m, group_m)
    pid_m = first_pid_m + (pid % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    num_k = ct.num_tiles(a_ptr, axis=1, shape=(tm, tk))
    half_tk: ct.Constant[int] = tk // 2
    acc = ct.zeros((tm, tn), dtype=ct.float32)

    for k in range(num_k):
        tile_a = ct.load(
            a_ptr, index=(pid_m, k), shape=(tm, tk), padding_mode=ct.PaddingMode.ZERO, latency=3
        )
        qw = ct.load(
            qw_ptr,
            index=(k, pid_n),
            shape=(half_tk, tn),
            padding_mode=ct.PaddingMode.ZERO,
            latency=3,
        )
        lo = ct.astype(ct.bitwise_and(qw, 0x0F), ct.float16)
        hi = ct.astype(ct.bitwise_and(ct.bitwise_rshift(qw, 4), 0x0F), ct.float16)
        w_int = ct.reshape(
            ct.cat((ct.reshape(lo, (half_tk, 1, tn)), ct.reshape(hi, (half_tk, 1, tn))), axis=1),
            (tk, tn),
        )
        g_idx = (k * tk) // gs
        scale = ct.load(
            scales_ptr,
            index=(g_idx, pid_n),
            shape=(1, tn),
            padding_mode=ct.PaddingMode.ZERO,
            latency=3,
        )
        zero = ct.load(
            zeros_ptr,
            index=(g_idx, pid_n),
            shape=(1, tn),
            padding_mode=ct.PaddingMode.ZERO,
            latency=3,
        )
        w_fp16 = (w_int - zero) * scale
        acc = ct.mma(tile_a, w_fp16, acc)

    sig = 1.0 / (1.0 + ct.exp(-acc))
    out = acc * sig
    ct.store(c_ptr, index=(pid_m, pid_n), tile=ct.astype(out, ct.float16))


def fused_gemm_fp16_silu(
    a: torch.Tensor,
    b: torch.Tensor,
    tile_shape: tuple[int, int, int] | None = None,
) -> torch.Tensor:
    """Compute SiLU(a @ b) in one fused CuTile kernel."""
    m, k = a.shape
    _, n = b.shape
    tm, tn, tk = tile_shape or _select_gemm_tiles(m)
    c = torch.empty((m, n), device=a.device, dtype=torch.float16)
    grid = (ct.cdiv(m, tm) * ct.cdiv(n, tn), 1, 1)
    ct.launch(
        torch.cuda.current_stream(),
        grid,
        gemm_fp16_silu_kernel,
        (a.contiguous(), b.contiguous(), c, tm, tn, tk),
    )
    return c


def fused_gemm_dequant_silu(
    a: torch.Tensor,
    qweight: torch.Tensor,
    scales: torch.Tensor,
    zeros: torch.Tensor,
    group_size: int = 128,
    tile_shape: tuple[int, int, int] | None = None,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    """Compute SiLU(a @ dequant(qweight)) in one fused CuTile kernel."""
    m, k = a.shape
    if qweight.dtype == torch.int32:
        k_div8, n = qweight.shape
        shifts = torch.arange(0, 32, 4, device=qweight.device, dtype=torch.int32).view(1, 8, 1)
        nibbles = ((qweight.unsqueeze(1) >> shifts) & 0xF).to(torch.uint8).reshape(k_div8 * 8, n)
        qw_u8 = pack_int4_weights(nibbles)
    else:
        qw_u8 = qweight if qweight.is_contiguous() else qweight.contiguous()

    n = qw_u8.shape[1]
    tm, tn, tk = tile_shape or _select_gemm_tiles(m, group_size=group_size, n=n)
    if group_size % tk != 0 or k % tk != 0:
        tk = 64 if (group_size % 64 == 0 and k % 64 == 0) else 32
    c = out if out is not None else torch.empty((m, n), device=a.device, dtype=torch.float16)
    a_c = a if a.is_contiguous() else a.contiguous()
    s_c = scales if scales.is_contiguous() else scales.contiguous()
    z_c = zeros if zeros.is_contiguous() else zeros.contiguous()
    grid = (ct.cdiv(m, tm) * ct.cdiv(n, tn), 1, 1)
    ct.launch(
        torch.cuda.current_stream(),
        grid,
        gemm_dequant_silu_kernel,
        (
            a_c,
            qw_u8,
            s_c,
            z_c,
            c,
            tm,
            tn,
            tk,
            group_size,
        ),
    )
    return c


@ct.kernel(occupancy=2)
def gemm_dequant_swiglu_kernel(
    a_ptr,
    gw_ptr,
    gs_ptr,
    gz_ptr,
    uw_ptr,
    us_ptr,
    uz_ptr,
    c_ptr,
    tm: ct.Constant[int],
    tn: ct.Constant[int],
    tk: ct.Constant[int],
    gs: ct.Constant[int],
) -> None:
    """Fused INT4-dequant Gate + Up + SwiGLU (`SiLU(a @ W_gate) * (a @ W_up)`) in one kernel."""
    group_m: ct.Constant[int] = 8
    pid = ct.bid(0)
    num_m = ct.num_tiles(a_ptr, axis=0, shape=(tm, tk))
    num_n = ct.num_tiles(gw_ptr, axis=1, shape=(tk // 2, tn))
    num_pid_in_group = group_m * num_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * group_m
    group_size_m = min(num_m - first_pid_m, group_m)
    pid_m = first_pid_m + (pid % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    num_k = ct.num_tiles(a_ptr, axis=1, shape=(tm, tk))
    half_tk: ct.Constant[int] = tk // 2
    acc_g = ct.zeros((tm, tn), dtype=ct.float32)
    acc_u = ct.zeros((tm, tn), dtype=ct.float32)

    for k in range(num_k):
        tile_a = ct.load(
            a_ptr, index=(pid_m, k), shape=(tm, tk), padding_mode=ct.PaddingMode.ZERO, latency=3
        )
        g_idx = (k * tk) // gs

        gw = ct.load(
            gw_ptr,
            index=(k, pid_n),
            shape=(half_tk, tn),
            padding_mode=ct.PaddingMode.ZERO,
            latency=3,
        )
        g_lo = ct.astype(ct.bitwise_and(gw, 0x0F), ct.float16)
        g_hi = ct.astype(ct.bitwise_and(ct.bitwise_rshift(gw, 4), 0x0F), ct.float16)
        g_int = ct.reshape(
            ct.cat(
                (ct.reshape(g_lo, (half_tk, 1, tn)), ct.reshape(g_hi, (half_tk, 1, tn))), axis=1
            ),
            (tk, tn),
        )
        g_sc = ct.load(
            gs_ptr, index=(g_idx, pid_n), shape=(1, tn), padding_mode=ct.PaddingMode.ZERO, latency=3
        )
        g_zr = ct.load(
            gz_ptr, index=(g_idx, pid_n), shape=(1, tn), padding_mode=ct.PaddingMode.ZERO, latency=3
        )
        acc_g = ct.mma(tile_a, (g_int - g_zr) * g_sc, acc_g)

        uw = ct.load(
            uw_ptr,
            index=(k, pid_n),
            shape=(half_tk, tn),
            padding_mode=ct.PaddingMode.ZERO,
            latency=3,
        )
        u_lo = ct.astype(ct.bitwise_and(uw, 0x0F), ct.float16)
        u_hi = ct.astype(ct.bitwise_and(ct.bitwise_rshift(uw, 4), 0x0F), ct.float16)
        u_int = ct.reshape(
            ct.cat(
                (ct.reshape(u_lo, (half_tk, 1, tn)), ct.reshape(u_hi, (half_tk, 1, tn))), axis=1
            ),
            (tk, tn),
        )
        u_sc = ct.load(
            us_ptr, index=(g_idx, pid_n), shape=(1, tn), padding_mode=ct.PaddingMode.ZERO, latency=3
        )
        u_zr = ct.load(
            uz_ptr, index=(g_idx, pid_n), shape=(1, tn), padding_mode=ct.PaddingMode.ZERO, latency=3
        )
        acc_u = ct.mma(tile_a, (u_int - u_zr) * u_sc, acc_u)

    sig = 1.0 / (1.0 + ct.exp(-acc_g))
    g_fp16 = ct.astype(acc_g * sig, ct.float16)
    u_fp16 = ct.astype(acc_u, ct.float16)
    ct.store(c_ptr, index=(pid_m, pid_n), tile=g_fp16 * u_fp16)


@ct.kernel(occupancy=4)
def gemm_dequant_swiglu_splitk_kernel(
    a_ptr,
    gw_ptr,
    gs_ptr,
    gz_ptr,
    uw_ptr,
    us_ptr,
    uz_ptr,
    part_g_ptr,
    part_u_ptr,
    split_k: ct.Constant[int],
    tm: ct.Constant[int],
    tn: ct.Constant[int],
    tk: ct.Constant[int],
    gs: ct.Constant[int],
) -> None:
    """Split-K INT4-dequant Gate + Up GEMM accumulating FP32 partials `(split_k, M, N)`."""
    pid_m = ct.bid(0)
    pid_n = ct.bid(1)
    sk_id = ct.bid(2)
    num_k = ct.num_tiles(a_ptr, axis=1, shape=(tm, tk))
    k_per_split = (num_k + split_k - 1) // split_k
    k_start = sk_id * k_per_split
    k_end = min(num_k, k_start + k_per_split)
    half_tk: ct.Constant[int] = tk // 2
    acc_g = ct.zeros((tm, tn), dtype=ct.float32)
    acc_u = ct.zeros((tm, tn), dtype=ct.float32)

    for k in range(k_start, k_end):
        tile_a = ct.load(
            a_ptr, index=(pid_m, k), shape=(tm, tk), padding_mode=ct.PaddingMode.ZERO, latency=3
        )
        g_idx = (k * tk) // gs

        gw = ct.load(
            gw_ptr,
            index=(k, pid_n),
            shape=(half_tk, tn),
            padding_mode=ct.PaddingMode.ZERO,
            latency=3,
        )
        g_lo = ct.astype(ct.bitwise_and(gw, 0x0F), ct.float16)
        g_hi = ct.astype(ct.bitwise_and(ct.bitwise_rshift(gw, 4), 0x0F), ct.float16)
        g_int = ct.reshape(
            ct.cat(
                (ct.reshape(g_lo, (half_tk, 1, tn)), ct.reshape(g_hi, (half_tk, 1, tn))), axis=1
            ),
            (tk, tn),
        )
        g_sc = ct.load(
            gs_ptr, index=(g_idx, pid_n), shape=(1, tn), padding_mode=ct.PaddingMode.ZERO, latency=3
        )
        g_zr = ct.load(
            gz_ptr, index=(g_idx, pid_n), shape=(1, tn), padding_mode=ct.PaddingMode.ZERO, latency=3
        )
        acc_g = ct.mma(tile_a, (g_int - g_zr) * g_sc, acc_g)

        uw = ct.load(
            uw_ptr,
            index=(k, pid_n),
            shape=(half_tk, tn),
            padding_mode=ct.PaddingMode.ZERO,
            latency=3,
        )
        u_lo = ct.astype(ct.bitwise_and(uw, 0x0F), ct.float16)
        u_hi = ct.astype(ct.bitwise_and(ct.bitwise_rshift(uw, 4), 0x0F), ct.float16)
        u_int = ct.reshape(
            ct.cat(
                (ct.reshape(u_lo, (half_tk, 1, tn)), ct.reshape(u_hi, (half_tk, 1, tn))), axis=1
            ),
            (tk, tn),
        )
        u_sc = ct.load(
            us_ptr, index=(g_idx, pid_n), shape=(1, tn), padding_mode=ct.PaddingMode.ZERO, latency=3
        )
        u_zr = ct.load(
            uz_ptr, index=(g_idx, pid_n), shape=(1, tn), padding_mode=ct.PaddingMode.ZERO, latency=3
        )
        acc_u = ct.mma(tile_a, (u_int - u_zr) * u_sc, acc_u)

    ct.store(part_g_ptr, index=(sk_id, pid_m, pid_n), tile=ct.reshape(acc_g, (1, tm, tn)))
    ct.store(part_u_ptr, index=(sk_id, pid_m, pid_n), tile=ct.reshape(acc_u, (1, tm, tn)))


@ct.kernel(occupancy=4)
def splitk_reduce_swiglu_fp16_kernel(
    part_g_ptr,
    part_u_ptr,
    out_ptr,
    split_k: ct.Constant[int],
    tm: ct.Constant[int],
    tn: ct.Constant[int],
) -> None:
    """Reduce Split-K Gate & Up FP32 partials and compute SiLU(g) * u in registers."""
    pid_m = ct.bid(0)
    pid_n = ct.bid(1)
    acc_g = ct.zeros((tm, tn), dtype=ct.float32)
    acc_u = ct.zeros((tm, tn), dtype=ct.float32)
    for sk in range(split_k):
        acc_g = acc_g + ct.reshape(
            ct.load(
                part_g_ptr,
                index=(sk, pid_m, pid_n),
                shape=(1, tm, tn),
                padding_mode=ct.PaddingMode.ZERO,
                latency=3,
            ),
            (tm, tn),
        )
        acc_u = acc_u + ct.reshape(
            ct.load(
                part_u_ptr,
                index=(sk, pid_m, pid_n),
                shape=(1, tm, tn),
                padding_mode=ct.PaddingMode.ZERO,
                latency=3,
            ),
            (tm, tn),
        )
    sig = 1.0 / (1.0 + ct.exp(-acc_g))
    out = ct.astype(acc_g * sig, ct.float16) * ct.astype(acc_u, ct.float16)
    ct.store(out_ptr, index=(pid_m, pid_n), tile=out)


def _select_swiglu_tiles(m: int, group_size: int) -> tuple[int, int, int]:
    tk = 64 if group_size >= 64 else 32
    if m <= 16:
        tk_dec = 128 if group_size >= 128 else tk
        return (16, 64, tk_dec)
    if m <= 64:
        return (64, 64, tk)
    return (128, 64, tk)


def fused_swiglu_dequant(
    a: torch.Tensor,
    gate_qweight: torch.Tensor,
    gate_scales: torch.Tensor,
    gate_zeros: torch.Tensor,
    up_qweight: torch.Tensor,
    up_scales: torch.Tensor,
    up_zeros: torch.Tensor,
    group_size: int = 128,
    tile_shape: tuple[int, int, int] | None = None,
    out: torch.Tensor | None = None,
    part_g_buf: torch.Tensor | None = None,
    part_u_buf: torch.Tensor | None = None,
) -> torch.Tensor:
    """Compute `SiLU(a @ dequant(W_gate)) * (a @ dequant(W_up))` in a single CuTile kernel."""
    m, k = a.shape
    gw = gate_qweight if gate_qweight.is_contiguous() else gate_qweight.contiguous()
    uw = up_qweight if up_qweight.is_contiguous() else up_qweight.contiguous()
    n = gw.shape[1]
    tm, tn, tk = tile_shape or _select_swiglu_tiles(m, group_size)
    if group_size % tk != 0 or k % tk != 0:
        tk = 64 if (group_size % 64 == 0 and k % 64 == 0) else 32
    c = out if out is not None else torch.empty((m, n), device=a.device, dtype=torch.float16)
    a_c = a if a.is_contiguous() else a.contiguous()
    gs_c = gate_scales if gate_scales.is_contiguous() else gate_scales.contiguous()
    gz_c = gate_zeros if gate_zeros.is_contiguous() else gate_zeros.contiguous()
    us_c = up_scales if up_scales.is_contiguous() else up_scales.contiguous()
    uz_c = up_zeros if up_zeros.is_contiguous() else up_zeros.contiguous()

    split_k = (
        2
        if (
            tile_shape is None
            and m <= 16
            and k >= 2048
            and tk == 128
            and n >= 4096
            and n % tn == 0
            and not (part_g_buf is None and (k == 8192 or k >= 24576))
        )
        else 1
    )
    if split_k > 1:
        needed = split_k * m * n
        pg = (
            part_g_buf[:needed].view(split_k, m, n)
            if (part_g_buf is not None and part_g_buf.numel() >= needed)
            else torch.empty((split_k, m, n), device=a.device, dtype=torch.float32)
        )
        pu = (
            part_u_buf[:needed].view(split_k, m, n)
            if (part_u_buf is not None and part_u_buf.numel() >= needed)
            else torch.empty((split_k, m, n), device=a.device, dtype=torch.float32)
        )
        ct.launch(
            torch.cuda.current_stream(),
            (ct.cdiv(m, tm), ct.cdiv(n, tn), split_k),
            gemm_dequant_swiglu_splitk_kernel,
            (a_c, gw, gs_c, gz_c, uw, us_c, uz_c, pg, pu, split_k, tm, tn, tk, group_size),
        )
        ct.launch(
            torch.cuda.current_stream(),
            (ct.cdiv(m, tm), ct.cdiv(n, tn), 1),
            splitk_reduce_swiglu_fp16_kernel,
            (pg, pu, c, split_k, tm, tn),
        )
        return c

    grid = (ct.cdiv(m, tm) * ct.cdiv(n, tn), 1, 1)
    ct.launch(
        torch.cuda.current_stream(),
        grid,
        gemm_dequant_swiglu_kernel,
        (a_c, gw, gs_c, gz_c, uw, us_c, uz_c, c, tm, tn, tk, group_size),
    )
    return c


@ct.kernel(occupancy=2)
def gemm_fp16_swiglu_kernel(
    a_ptr,
    wg_ptr,
    wu_ptr,
    c_ptr,
    tm: ct.Constant[int],
    tn: ct.Constant[int],
    tk: ct.Constant[int],
) -> None:
    """Fused FP16 Gate + Up + SwiGLU (`SiLU(a @ W_gate) * (a @ W_up)`) in one kernel."""
    group_m: ct.Constant[int] = 8
    pid = ct.bid(0)
    num_m = ct.num_tiles(a_ptr, axis=0, shape=(tm, tk))
    num_n = ct.num_tiles(wg_ptr, axis=1, shape=(tk, tn))
    num_pid_in_group = group_m * num_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * group_m
    group_size_m = min(num_m - first_pid_m, group_m)
    pid_m = first_pid_m + (pid % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    num_k = ct.num_tiles(a_ptr, axis=1, shape=(tm, tk))
    acc_g = ct.zeros((tm, tn), dtype=ct.float32)
    acc_u = ct.zeros((tm, tn), dtype=ct.float32)
    for k in range(num_k):
        ta = ct.load(
            a_ptr, index=(pid_m, k), shape=(tm, tk), padding_mode=ct.PaddingMode.ZERO, latency=3
        )
        twg = ct.load(
            wg_ptr, index=(k, pid_n), shape=(tk, tn), padding_mode=ct.PaddingMode.ZERO, latency=3
        )
        twu = ct.load(
            wu_ptr, index=(k, pid_n), shape=(tk, tn), padding_mode=ct.PaddingMode.ZERO, latency=3
        )
        acc_g = ct.mma(ta, twg, acc_g)
        acc_u = ct.mma(ta, twu, acc_u)

    sig = 1.0 / (1.0 + ct.exp(-acc_g))
    out = ct.astype(acc_g * sig, ct.float16) * ct.astype(acc_u, ct.float16)
    ct.store(c_ptr, index=(pid_m, pid_n), tile=out)


@ct.kernel(occupancy=4)
def gemm_fp16_swiglu_splitk_kernel(
    a_ptr,
    wg_ptr,
    wu_ptr,
    part_g_ptr,
    part_u_ptr,
    split_k: ct.Constant[int],
    tm: ct.Constant[int],
    tn: ct.Constant[int],
    tk: ct.Constant[int],
) -> None:
    """Split-K FP16 Gate + Up GEMM accumulating FP32 partials `(split_k, M, N)`."""
    pid_m = ct.bid(0)
    pid_n = ct.bid(1)
    sk_id = ct.bid(2)
    num_k = ct.num_tiles(a_ptr, axis=1, shape=(tm, tk))
    k_per_split = (num_k + split_k - 1) // split_k
    k_start = sk_id * k_per_split
    k_end = min(num_k, k_start + k_per_split)
    acc_g = ct.zeros((tm, tn), dtype=ct.float32)
    acc_u = ct.zeros((tm, tn), dtype=ct.float32)
    for k in range(k_start, k_end):
        ta = ct.load(
            a_ptr, index=(pid_m, k), shape=(tm, tk), padding_mode=ct.PaddingMode.ZERO, latency=3
        )
        twg = ct.load(
            wg_ptr, index=(k, pid_n), shape=(tk, tn), padding_mode=ct.PaddingMode.ZERO, latency=3
        )
        twu = ct.load(
            wu_ptr, index=(k, pid_n), shape=(tk, tn), padding_mode=ct.PaddingMode.ZERO, latency=3
        )
        acc_g = ct.mma(ta, twg, acc_g)
        acc_u = ct.mma(ta, twu, acc_u)
    ct.store(part_g_ptr, index=(sk_id, pid_m, pid_n), tile=ct.reshape(acc_g, (1, tm, tn)))
    ct.store(part_u_ptr, index=(sk_id, pid_m, pid_n), tile=ct.reshape(acc_u, (1, tm, tn)))


def fused_swiglu_fp16(
    a: torch.Tensor,
    w_gate: torch.Tensor,
    w_up: torch.Tensor,
    tile_shape: tuple[int, int, int] | None = None,
    out: torch.Tensor | None = None,
    part_g_buf: torch.Tensor | None = None,
    part_u_buf: torch.Tensor | None = None,
) -> torch.Tensor:
    """Compute `SiLU(a @ w_gate) * (a @ w_up)` for FP16 weights in one CuTile pass."""
    m, k = a.shape
    _, n = w_gate.shape
    a_c = a if a.is_contiguous() else a.contiguous()
    wg_c = w_gate if w_gate.is_contiguous() else w_gate.contiguous()
    wu_c = w_up if w_up.is_contiguous() else w_up.contiguous()
    c = out if out is not None else torch.empty((m, n), device=a.device, dtype=torch.float16)

    if tile_shape is None and m <= 16 and k >= 512 and n % 64 == 0:
        tm, tn, tk = 16, 64, 64
        split_k = min(4 if k >= 768 else 2, max(1, k // tk))
        if split_k > 1:
            needed = split_k * m * n
            pg = (
                part_g_buf[:needed].view(split_k, m, n)
                if (part_g_buf is not None and part_g_buf.numel() >= needed)
                else torch.empty((split_k, m, n), device=a.device, dtype=torch.float32)
            )
            pu = (
                part_u_buf[:needed].view(split_k, m, n)
                if (part_u_buf is not None and part_u_buf.numel() >= needed)
                else torch.empty((split_k, m, n), device=a.device, dtype=torch.float32)
            )
            ct.launch(
                torch.cuda.current_stream(),
                (ct.cdiv(m, tm), ct.cdiv(n, tn), split_k),
                gemm_fp16_swiglu_splitk_kernel,
                (a_c, wg_c, wu_c, pg, pu, split_k, tm, tn, tk),
            )
            ct.launch(
                torch.cuda.current_stream(),
                (ct.cdiv(m, tm), ct.cdiv(n, tn), 1),
                splitk_reduce_swiglu_fp16_kernel,
                (pg, pu, c, split_k, tm, tn),
            )
            return c

    tm, tn, tk = tile_shape or ((16, 64, 64) if m <= 16 else (64, 64, 64))
    grid = (ct.cdiv(m, tm) * ct.cdiv(n, tn), 1, 1)
    ct.launch(
        torch.cuda.current_stream(),
        grid,
        gemm_fp16_swiglu_kernel,
        (a_c, wg_c, wu_c, c, tm, tn, tk),
    )
    return c
