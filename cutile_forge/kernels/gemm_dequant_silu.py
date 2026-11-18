"""Fused GEMM + INT4 dequant + SiLU activation CuTile kernel."""

from __future__ import annotations

from typing import Tuple

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
    tile_shape: Tuple[int, int, int] | None = None,
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
    tile_shape: Tuple[int, int, int] | None = None,
) -> torch.Tensor:
    """Compute SiLU(a @ dequant(qweight)) in one fused CuTile kernel."""
    m, k = a.shape
    if qweight.dtype == torch.int32:
        k_div8, n = qweight.shape
        shifts = torch.arange(0, 32, 4, device=qweight.device, dtype=torch.int32).view(1, 8, 1)
        nibbles = ((qweight.unsqueeze(1) >> shifts) & 0xF).to(torch.uint8).reshape(k_div8 * 8, n)
        qw_u8 = pack_int4_weights(nibbles)
    else:
        qw_u8 = qweight.contiguous()

    n = qw_u8.shape[1]
    tm, tn, tk = tile_shape or _select_gemm_tiles(m, group_size=group_size)
    if group_size % tk != 0:
        tk = 64 if group_size % 64 == 0 else 32
    c = torch.empty((m, n), device=a.device, dtype=torch.float16)
    grid = (ct.cdiv(m, tm) * ct.cdiv(n, tn), 1, 1)
    ct.launch(
        torch.cuda.current_stream(),
        grid,
        gemm_dequant_silu_kernel,
        (
            a.contiguous(),
            qw_u8,
            scales.contiguous(),
            zeros.contiguous(),
            c,
            tm,
            tn,
            tk,
            group_size,
        ),
    )
    return c
