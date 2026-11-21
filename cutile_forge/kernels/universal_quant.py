"""Phase 10 Universal Quantization Engine: native CuTile kernels for GPTQ, AWQ, GGUF (Q4_0, Q8_0, Q4_K, Q6_K), INT8, FP8, and Block-Scaled FP8 without load-time format conversion."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Any

import numpy as np
import torch

from cutile_forge.cuda_tile_init import ct
from cutile_forge.kernels.embedding_lmhead import cutile_col_permute
from cutile_forge.kernels.gemm_dequant import (
    fused_gemm_dequant,
    unpack_int4_weights_torch,
)

GGUF_QK_K = 256


class QuantFormat(StrEnum):
    """Canonical quantization format identifiers for native execution & autotuning."""

    FP16 = "fp16"
    GPTQ = "gptq"
    AWQ = "awq"
    GGUF_Q4_0 = "gguf_q4_0"
    GGUF_Q8_0 = "gguf_q8_0"
    GGUF_Q4_K = "gguf_q4_k"
    GGUF_Q6_K = "gguf_q6_k"
    GGUF_Q4_K_M = "gguf_q4_k_m"
    INT8 = "int8"
    FP8 = "fp8"
    FP8_CHANNEL = "fp8_channel"
    FP8_BLOCK = "fp8_block"


# ---------------------------------------------------------------------------
# 1. Native AutoAWQ Kernel (Direct execution on [K, N // 8] int32 interleaved)
# ---------------------------------------------------------------------------


@ct.kernel(occupancy=2)
def gemm_awq_native_kernel(
    a_ptr,  # (M, K) fp16
    qw_i32_ptr,  # (K, N // 8) int32
    qz_i32_ptr,  # (K // gs, N // 8) int32
    scales_ptr,  # (K // gs, N) fp16
    c_ptr,  # (M, N) fp16
    tm: ct.Constant[int],
    tn: ct.Constant[int],
    tk: ct.Constant[int],
    gs: ct.Constant[int],
) -> None:
    """Native AWQ GEMM unpacking `[0, 16, 4, 20, 8, 24, 12, 28]` int32 nibbles in registers."""
    pid_m = ct.bid(0)
    pid_n = ct.bid(1)
    num_k = ct.num_tiles(a_ptr, axis=1, shape=(tm, tk))
    tn_div8: ct.Constant[int] = tn // 8
    acc = ct.zeros((tm, tn), dtype=ct.float32)

    for k in range(num_k):
        tile_a = ct.load(
            a_ptr, index=(pid_m, k), shape=(tm, tk), padding_mode=ct.PaddingMode.ZERO, latency=3
        )
        qw = ct.load(
            qw_i32_ptr,
            index=(k, pid_n),
            shape=(tk, tn_div8),
            padding_mode=ct.PaddingMode.ZERO,
            latency=3,
        )
        # AWQ bit shifts: [0, 16, 4, 20, 8, 24, 12, 28]
        w0 = ct.reshape(ct.astype(ct.bitwise_and(qw, 0x0F), ct.float16), (tk, tn_div8, 1))
        w1 = ct.reshape(
            ct.astype(ct.bitwise_and(ct.bitwise_rshift(qw, 16), 0x0F), ct.float16),
            (tk, tn_div8, 1),
        )
        w2 = ct.reshape(
            ct.astype(ct.bitwise_and(ct.bitwise_rshift(qw, 4), 0x0F), ct.float16),
            (tk, tn_div8, 1),
        )
        w3 = ct.reshape(
            ct.astype(ct.bitwise_and(ct.bitwise_rshift(qw, 20), 0x0F), ct.float16),
            (tk, tn_div8, 1),
        )
        w4 = ct.reshape(
            ct.astype(ct.bitwise_and(ct.bitwise_rshift(qw, 8), 0x0F), ct.float16),
            (tk, tn_div8, 1),
        )
        w5 = ct.reshape(
            ct.astype(ct.bitwise_and(ct.bitwise_rshift(qw, 24), 0x0F), ct.float16),
            (tk, tn_div8, 1),
        )
        w6 = ct.reshape(
            ct.astype(ct.bitwise_and(ct.bitwise_rshift(qw, 12), 0x0F), ct.float16),
            (tk, tn_div8, 1),
        )
        w7 = ct.reshape(
            ct.astype(ct.bitwise_and(ct.bitwise_rshift(qw, 28), 0x0F), ct.float16),
            (tk, tn_div8, 1),
        )
        w_01 = ct.cat((w0, w1), axis=2)
        w_23 = ct.cat((w2, w3), axis=2)
        w_45 = ct.cat((w4, w5), axis=2)
        w_67 = ct.cat((w6, w7), axis=2)
        w_int = ct.reshape(
            ct.cat((ct.cat((w_01, w_23), axis=2), ct.cat((w_45, w_67), axis=2)), axis=2),
            (tk, tn),
        )

        g_idx = (k * tk) // gs
        qz = ct.load(
            qz_i32_ptr,
            index=(g_idx, pid_n),
            shape=(1, tn_div8),
            padding_mode=ct.PaddingMode.ZERO,
            latency=3,
        )
        z0 = ct.reshape(ct.astype(ct.bitwise_and(qz, 0x0F), ct.float16), (1, tn_div8, 1))
        z1 = ct.reshape(
            ct.astype(ct.bitwise_and(ct.bitwise_rshift(qz, 16), 0x0F), ct.float16),
            (1, tn_div8, 1),
        )
        z2 = ct.reshape(
            ct.astype(ct.bitwise_and(ct.bitwise_rshift(qz, 4), 0x0F), ct.float16),
            (1, tn_div8, 1),
        )
        z3 = ct.reshape(
            ct.astype(ct.bitwise_and(ct.bitwise_rshift(qz, 20), 0x0F), ct.float16),
            (1, tn_div8, 1),
        )
        z4 = ct.reshape(
            ct.astype(ct.bitwise_and(ct.bitwise_rshift(qz, 8), 0x0F), ct.float16),
            (1, tn_div8, 1),
        )
        z5 = ct.reshape(
            ct.astype(ct.bitwise_and(ct.bitwise_rshift(qz, 24), 0x0F), ct.float16),
            (1, tn_div8, 1),
        )
        z6 = ct.reshape(
            ct.astype(ct.bitwise_and(ct.bitwise_rshift(qz, 12), 0x0F), ct.float16),
            (1, tn_div8, 1),
        )
        z7 = ct.reshape(
            ct.astype(ct.bitwise_and(ct.bitwise_rshift(qz, 28), 0x0F), ct.float16),
            (1, tn_div8, 1),
        )
        z_01 = ct.cat((z0, z1), axis=2)
        z_23 = ct.cat((z2, z3), axis=2)
        z_45 = ct.cat((z4, z5), axis=2)
        z_67 = ct.cat((z6, z7), axis=2)
        zero = ct.reshape(
            ct.cat((ct.cat((z_01, z_23), axis=2), ct.cat((z_45, z_67), axis=2)), axis=2),
            (1, tn),
        )
        scale = ct.load(
            scales_ptr,
            index=(g_idx, pid_n),
            shape=(1, tn),
            padding_mode=ct.PaddingMode.ZERO,
            latency=3,
        )
        w_fp16 = (w_int - zero) * scale
        acc = ct.mma(tile_a, w_fp16, acc)

    ct.store(c_ptr, index=(pid_m, pid_n), tile=ct.astype(acc, ct.float16))


def cutile_gemm_awq_native(
    a: torch.Tensor,
    qweight_awq: torch.Tensor,
    qzeros_awq: torch.Tensor,
    scales: torch.Tensor,
    group_size: int = 128,
    tile_shape: tuple[int, int, int] | None = None,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    """Execute AWQ `[K, N//8]` int32 weights directly in CuTile without format conversion."""
    m, k = a.shape
    n = qweight_awq.shape[1] * 8
    eff_gs = k if group_size <= 0 else group_size
    tm, tn, tk = tile_shape or ((16, 64, 64) if m <= 16 else (64, 64, 64))
    if eff_gs % tk != 0:
        tk = 32 if eff_gs % 32 == 0 else eff_gs
    c = out if out is not None else torch.empty((m, n), device=a.device, dtype=torch.float16)

    if not a.is_cuda:
        ref_w = dequant_awq_reference(qweight_awq, qzeros_awq, scales, group_size=eff_gs)
        c.copy_(torch.matmul(a.float(), ref_w.float()).to(torch.float16))
        return c

    grid = (ct.cdiv(m, tm), ct.cdiv(n, tn), 1)
    ct.launch(
        torch.cuda.current_stream(),
        grid,
        gemm_awq_native_kernel,
        (
            a.contiguous(),
            qweight_awq.contiguous(),
            qzeros_awq.contiguous(),
            scales.contiguous(),
            c,
            tm,
            tn,
            tk,
            eff_gs,
        ),
    )
    return c


def dequant_awq_reference(
    qweight_awq: torch.Tensor,
    qzeros_awq: torch.Tensor,
    scales: torch.Tensor,
    group_size: int = 128,
) -> torch.Tensor:
    """Exact AutoAWQ reference dequantization (`[0, 16, 4, 20, 8, 24, 12, 28]` bit shifts)."""
    k_dim, n_div8 = qweight_awq.shape
    n_dim = n_div8 * 8
    eff_gs = k_dim if group_size <= 0 else group_size
    shifts = torch.tensor(
        [0, 16, 4, 20, 8, 24, 12, 28], device=qweight_awq.device, dtype=torch.int32
    )
    qw = (
        ((qweight_awq.unsqueeze(-1) >> shifts.view(1, 1, 8)) & 0x0F)
        .reshape(k_dim, n_dim)
        .to(torch.float16)
    )
    qz = (
        ((qzeros_awq.unsqueeze(-1) >> shifts.view(1, 1, 8)) & 0x0F)
        .reshape(-1, n_dim)
        .to(torch.float16)
    )
    z_exp = qz.repeat_interleave(eff_gs, dim=0)[:k_dim]
    s_exp = scales.to(torch.float16).repeat_interleave(eff_gs, dim=0)[:k_dim]
    return ((qw - z_exp) * s_exp).contiguous()


def dequant_gptq_reference(
    qweight_i32: torch.Tensor,
    qzeros_i32: torch.Tensor,
    scales: torch.Tensor,
    group_size: int = 128,
    g_idx: torch.Tensor | None = None,
) -> torch.Tensor:
    """Exact AutoGPTQ reference dequantization supporting all group sizes (`32/64/128/-1`) & `g_idx`."""
    k_div8, n_dim = qweight_i32.shape
    k_dim = k_div8 * 8
    eff_gs = k_dim if group_size <= 0 else group_size
    shifts = torch.arange(0, 32, 4, device=qweight_i32.device, dtype=torch.int32)
    qw = (
        ((qweight_i32.unsqueeze(1) >> shifts.view(1, 8, 1)) & 0x0F)
        .reshape(k_dim, n_dim)
        .to(torch.float16)
    )
    g_dim, n_div8 = qzeros_i32.shape
    qz = (
        (((qzeros_i32.unsqueeze(2) >> shifts.view(1, 1, 8)) & 0x0F) + 1)
        .reshape(g_dim, n_div8 * 8)
        .to(torch.float16)
    )
    sc = scales.to(torch.float16)

    if g_dim == 1:
        return ((qw - qz.expand(k_dim, n_dim)) * sc.expand(k_dim, n_dim)).contiguous()
    if g_idx is not None:
        g = g_idx.to(device=qweight_i32.device, dtype=torch.long)
        return ((qw - qz[g]) * sc[g]).contiguous()
    z_exp = qz.repeat_interleave(eff_gs, dim=0)[:k_dim]
    s_exp = sc.repeat_interleave(eff_gs, dim=0)[:k_dim]
    return ((qw - z_exp) * s_exp).contiguous()


# ---------------------------------------------------------------------------
# 2. Native INT8 & GGUF Q8_0 Kernel (`[K, N]` int8 + `[K // gs, N]` fp16 scales)
# ---------------------------------------------------------------------------


@ct.kernel(occupancy=2)
def gemm_int8_dequant_kernel(
    a_ptr,  # (M, K) fp16
    qw_i8_ptr,  # (K, N) int8
    scales_ptr,  # (K // gs, N) fp16
    c_ptr,  # (M, N) fp16
    tm: ct.Constant[int],
    tn: ct.Constant[int],
    tk: ct.Constant[int],
    gs: ct.Constant[int],
) -> None:
    """Native INT8 / GGUF Q8_0 dequant + FP16 Tensor Core GEMM in CuTile."""
    pid_m = ct.bid(0)
    pid_n = ct.bid(1)
    num_k = ct.num_tiles(a_ptr, axis=1, shape=(tm, tk))
    acc = ct.zeros((tm, tn), dtype=ct.float32)

    for k in range(num_k):
        tile_a = ct.load(
            a_ptr, index=(pid_m, k), shape=(tm, tk), padding_mode=ct.PaddingMode.ZERO, latency=3
        )
        qw = ct.load(
            qw_i8_ptr,
            index=(k, pid_n),
            shape=(tk, tn),
            padding_mode=ct.PaddingMode.ZERO,
            latency=3,
        )
        g_idx = (k * tk) // gs
        scale = ct.load(
            scales_ptr,
            index=(g_idx, pid_n),
            shape=(1, tn),
            padding_mode=ct.PaddingMode.ZERO,
            latency=3,
        )
        w_fp16 = ct.astype(qw, ct.float16) * scale
        acc = ct.mma(tile_a, w_fp16, acc)

    ct.store(c_ptr, index=(pid_m, pid_n), tile=ct.astype(acc, ct.float16))


def quantize_int8_group(
    weight: torch.Tensor,
    group_size: int = 128,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Symmetric group-wise INT8 quantization of `(K, N)` weight matrix -> `(qw_i8, scales_fp16)`."""
    k_dim, n_dim = weight.shape
    eff_gs = k_dim if group_size <= 0 else group_size
    w_f = weight.float().view(k_dim // eff_gs, eff_gs, n_dim)
    amax = w_f.abs().max(dim=1, keepdim=True).values
    scales = (amax / 127.0).clamp(min=1e-7)
    qw_i8 = (w_f / scales).round().clamp(-127.0, 127.0).to(torch.int8).view(k_dim, n_dim)
    return qw_i8.contiguous(), scales.squeeze(1).to(torch.float16).contiguous()


def dequantize_int8_group(
    qw_i8: torch.Tensor,
    scales: torch.Tensor,
    group_size: int = 128,
) -> torch.Tensor:
    """Reference INT8 dequantization `(qw_i8.float16 * scale)`."""
    k_dim = qw_i8.shape[0]
    eff_gs = k_dim if group_size <= 0 else group_size
    s_exp = scales.to(torch.float16).repeat_interleave(eff_gs, dim=0)[:k_dim]
    return (qw_i8.to(torch.float16) * s_exp).contiguous()


def cutile_gemm_int8_dequant(
    a: torch.Tensor,
    qw_i8: torch.Tensor,
    scales: torch.Tensor,
    group_size: int = 128,
    tile_shape: tuple[int, int, int] | None = None,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    """Execute INT8 / GGUF Q8_0 dequantization + GEMM natively in CuTile."""
    m, k = a.shape
    n = qw_i8.shape[1]
    eff_gs = k if group_size <= 0 else group_size
    tm, tn, tk = tile_shape or ((16, 64, 32) if eff_gs == 32 else (16 if m <= 16 else 64, 64, 64))
    if eff_gs % tk != 0:
        tk = 32 if eff_gs % 32 == 0 else eff_gs
    c = out if out is not None else torch.empty((m, n), device=a.device, dtype=torch.float16)

    if not a.is_cuda:
        w_ref = dequantize_int8_group(qw_i8, scales, group_size=eff_gs)
        c.copy_(torch.matmul(a.float(), w_ref.float()).to(torch.float16))
        return c

    grid = (ct.cdiv(m, tm), ct.cdiv(n, tn), 1)
    ct.launch(
        torch.cuda.current_stream(),
        grid,
        gemm_int8_dequant_kernel,
        (a.contiguous(), qw_i8.contiguous(), scales.contiguous(), c, tm, tn, tk, eff_gs),
    )
    return c


# ---------------------------------------------------------------------------
# 3. Native GGUF Kernels & Bit-Exact Reference Unpackers (Q4_0, Q8_0, Q4_K, Q6_K)
# ---------------------------------------------------------------------------


@ct.kernel(occupancy=2)
def gemm_gguf_q4_0_kernel(
    a_ptr,  # (M, K) fp16
    qs_ptr,  # (K // 2, N) uint8 (16 low nibbles = 0..15, 16 high nibbles = 16..31 per 32 block)
    d_ptr,  # (K // 32, N) fp16
    c_ptr,  # (M, N) fp16
    tm: ct.Constant[int],
    tn: ct.Constant[int],
) -> None:
    """Native GGUF Q4_0 GEMM kernel (`tk=32`, low nibbles [0..15] + high nibbles [16..31])."""
    pid_m = ct.bid(0)
    pid_n = ct.bid(1)
    tk: ct.Constant[int] = 32
    half_tk: ct.Constant[int] = 16
    num_k = ct.num_tiles(a_ptr, axis=1, shape=(tm, tk))
    acc = ct.zeros((tm, tn), dtype=ct.float32)

    for k in range(num_k):
        tile_a = ct.load(
            a_ptr, index=(pid_m, k), shape=(tm, tk), padding_mode=ct.PaddingMode.ZERO, latency=3
        )
        qs = ct.load(
            qs_ptr,
            index=(k, pid_n),
            shape=(half_tk, tn),
            padding_mode=ct.PaddingMode.ZERO,
            latency=3,
        )
        d = ct.load(
            d_ptr, index=(k, pid_n), shape=(1, tn), padding_mode=ct.PaddingMode.ZERO, latency=3
        )
        lo = ct.astype(ct.bitwise_and(qs, 0x0F), ct.float16) - ct.astype(8.0, ct.float16)
        hi = ct.astype(ct.bitwise_and(ct.bitwise_rshift(qs, 4), 0x0F), ct.float16) - ct.astype(
            8.0, ct.float16
        )
        w_fp16 = ct.cat((lo, hi), axis=0) * d
        acc = ct.mma(tile_a, w_fp16, acc)

    ct.store(c_ptr, index=(pid_m, pid_n), tile=ct.astype(acc, ct.float16))


@ct.kernel(occupancy=2)
def gemm_gguf_q4_k_kernel(
    a_ptr,  # (M, K) fp16
    qs_ptr,  # (K // 2, N) uint8 (32-byte chunks: low=subblock 2k, high=subblock 2k+1)
    d_sub_ptr,  # (K // 32, N) fp16
    dm_sub_ptr,  # (K // 32, N) fp16
    c_ptr,  # (M, N) fp16
    tm: ct.Constant[int],
    tn: ct.Constant[int],
) -> None:
    """Native GGUF Q4_K GEMM kernel (`tk=64`, 2 sub-blocks of 32 weights per chunk)."""
    pid_m = ct.bid(0)
    pid_n = ct.bid(1)
    tk: ct.Constant[int] = 64
    half_tk: ct.Constant[int] = 32
    num_k = ct.num_tiles(a_ptr, axis=1, shape=(tm, tk))
    acc = ct.zeros((tm, tn), dtype=ct.float32)

    for k in range(num_k):
        tile_a = ct.load(
            a_ptr, index=(pid_m, k), shape=(tm, tk), padding_mode=ct.PaddingMode.ZERO, latency=3
        )
        qs = ct.load(
            qs_ptr,
            index=(k, pid_n),
            shape=(half_tk, tn),
            padding_mode=ct.PaddingMode.ZERO,
            latency=3,
        )
        d0 = ct.load(
            d_sub_ptr,
            index=(2 * k, pid_n),
            shape=(1, tn),
            padding_mode=ct.PaddingMode.ZERO,
            latency=3,
        )
        dm0 = ct.load(
            dm_sub_ptr,
            index=(2 * k, pid_n),
            shape=(1, tn),
            padding_mode=ct.PaddingMode.ZERO,
            latency=3,
        )
        d1 = ct.load(
            d_sub_ptr,
            index=(2 * k + 1, pid_n),
            shape=(1, tn),
            padding_mode=ct.PaddingMode.ZERO,
            latency=3,
        )
        dm1 = ct.load(
            dm_sub_ptr,
            index=(2 * k + 1, pid_n),
            shape=(1, tn),
            padding_mode=ct.PaddingMode.ZERO,
            latency=3,
        )
        lo = ct.astype(ct.bitwise_and(qs, 0x0F), ct.float16) * d0 - dm0
        hi = ct.astype(ct.bitwise_and(ct.bitwise_rshift(qs, 4), 0x0F), ct.float16) * d1 - dm1
        w_fp16 = ct.cat((lo, hi), axis=0)
        acc = ct.mma(tile_a, w_fp16, acc)

    ct.store(c_ptr, index=(pid_m, pid_n), tile=ct.astype(acc, ct.float16))


@ct.kernel(occupancy=2)
def gemm_gguf_q6_k_kernel(
    a_ptr,  # (M, K) fp16
    q_i8_ptr,  # (K, N) int8 signed 6-bit quants in [-32, 31]
    d_sub_ptr,  # (K // 16, N) fp16 sub-block scales (d * sc_i8)
    c_ptr,  # (M, N) fp16
    tm: ct.Constant[int],
    tn: ct.Constant[int],
) -> None:
    """Native GGUF Q6_K GEMM kernel (`tk=32`, 2 sub-blocks of 16 weights per K-tile)."""
    pid_m = ct.bid(0)
    pid_n = ct.bid(1)
    tk: ct.Constant[int] = 32
    sub_k: ct.Constant[int] = 16
    num_k = ct.num_tiles(a_ptr, axis=1, shape=(tm, tk))
    acc = ct.zeros((tm, tn), dtype=ct.float32)

    for k in range(num_k):
        tile_a = ct.load(
            a_ptr, index=(pid_m, k), shape=(tm, tk), padding_mode=ct.PaddingMode.ZERO, latency=3
        )
        q0 = ct.load(
            q_i8_ptr,
            index=(2 * k, pid_n),
            shape=(sub_k, tn),
            padding_mode=ct.PaddingMode.ZERO,
            latency=3,
        )
        q1 = ct.load(
            q_i8_ptr,
            index=(2 * k + 1, pid_n),
            shape=(sub_k, tn),
            padding_mode=ct.PaddingMode.ZERO,
            latency=3,
        )
        d0 = ct.load(
            d_sub_ptr,
            index=(2 * k, pid_n),
            shape=(1, tn),
            padding_mode=ct.PaddingMode.ZERO,
            latency=3,
        )
        d1 = ct.load(
            d_sub_ptr,
            index=(2 * k + 1, pid_n),
            shape=(1, tn),
            padding_mode=ct.PaddingMode.ZERO,
            latency=3,
        )
        w0 = ct.astype(q0, ct.float16) * d0
        w1 = ct.astype(q1, ct.float16) * d1
        w_fp16 = ct.cat((w0, w1), axis=0)
        acc = ct.mma(tile_a, w_fp16, acc)

    ct.store(c_ptr, index=(pid_m, pid_n), tile=ct.astype(acc, ct.float16))


def parse_gguf_q4_0_native(
    blocks_u8: torch.Tensor, k: int, n: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """Extract native `(qs_k2_n, d_k32_n)` from raw `(N * K // 32, 18)` GGUF Q4_0 blocks (zero re-quant)."""
    num_blocks = (k * n) // 32
    raw = blocks_u8.contiguous().view(n, k // 32, 18)
    d = raw[:, :, 0:2].contiguous().view(torch.float16).squeeze(-1).t().contiguous()  # (K//32, N)
    qs = raw[:, :, 2:18].reshape(n, k // 2).t().contiguous()  # (K//2, N)
    _ = num_blocks
    return qs, d


def dequant_gguf_q4_0_reference(blocks_u8: torch.Tensor, k: int, n: int) -> torch.Tensor:
    """Bit-exact reference dequantization of GGUF Q4_0 matching `gguf.quants.Q4_0.dequantize_blocks`."""
    num_blocks = (k * n) // 32
    raw = blocks_u8.contiguous().view(num_blocks, 18)
    d = raw[:, 0:2].contiguous().view(torch.float16).to(torch.float32)  # (B, 1)
    qs = raw[:, 2:18]
    lo = (qs & 0x0F).to(torch.int8) - 8
    hi = ((qs >> 4) & 0x0F).to(torch.int8) - 8
    q = torch.cat([lo, hi], dim=1).to(torch.float32)
    return (d * q).reshape(n, k).t().contiguous()


def parse_gguf_q8_0_native(
    blocks_u8: torch.Tensor, k: int, n: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """Extract native `(qs_i8_kn, d_k32_n)` from raw `(N * K // 32, 34)` GGUF Q8_0 blocks."""
    raw = blocks_u8.contiguous().view(n, k // 32, 34)
    d = raw[:, :, 0:2].contiguous().view(torch.float16).squeeze(-1).t().contiguous()  # (K//32, N)
    qs = raw[:, :, 2:34].contiguous().view(torch.int8).reshape(n, k).t().contiguous()  # (K, N)
    return qs, d


def dequant_gguf_q8_0_reference(blocks_u8: torch.Tensor, k: int, n: int) -> torch.Tensor:
    """Bit-exact reference dequantization of GGUF Q8_0 matching `gguf.quants.Q8_0.dequantize_blocks`."""
    num_blocks = (k * n) // 32
    raw = blocks_u8.contiguous().view(num_blocks, 34)
    d = raw[:, 0:2].contiguous().view(torch.float16).to(torch.float32)
    x = raw[:, 2:34].contiguous().view(torch.int8).to(torch.float32)
    return (x * d).reshape(n, k).t().contiguous()


def parse_gguf_q4_k_native(
    blocks_u8: torch.Tensor, k: int, n: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Extract native `(qs_k2_n, d_sub_k32_n, dm_sub_k32_n)` from GGUF Q4_K blocks (zero re-quant)."""
    num_blocks = (k * n) // 256
    raw = blocks_u8.contiguous().view(num_blocks, 144)
    d = raw[:, 0:2].contiguous().view(torch.float16).to(torch.float32)  # (B, 1)
    dmin = raw[:, 2:4].contiguous().view(torch.float16).to(torch.float32)  # (B, 1)
    sc_b = raw[:, 4:16].view(num_blocks, 3, 4)
    d_low = sc_b[:, 0, :] & 0x3F
    m_low = sc_b[:, 1, :] & 0x3F
    d_hi = (sc_b[:, 2, :] & 0x0F) | ((sc_b[:, 0, :] >> 2) & 0x30)
    m_hi = (sc_b[:, 2, :] >> 4) | ((sc_b[:, 1, :] >> 2) & 0x30)
    sc = torch.cat([d_low, d_hi], dim=-1).to(torch.float32)  # (B, 8)
    mn = torch.cat([m_low, m_hi], dim=-1).to(torch.float32)  # (B, 8)

    d_sub = (d * sc).to(torch.float16).view(n, k // 32).t().contiguous()  # (K//32, N)
    dm_sub = (dmin * mn).to(torch.float16).view(n, k // 32).t().contiguous()  # (K//32, N)
    qs = raw[:, 16:144].reshape(n, k // 2).t().contiguous()  # (K//2, N)
    return qs, d_sub, dm_sub


def dequant_gguf_q4_k_reference(blocks_u8: torch.Tensor, k: int, n: int) -> torch.Tensor:
    """Bit-exact reference dequantization of GGUF Q4_K matching `gguf.quants.Q4_K.dequantize_blocks`."""
    num_blocks = (k * n) // 256
    raw = blocks_u8.contiguous().view(num_blocks, 144)
    d = raw[:, 0:2].contiguous().view(torch.float16).to(torch.float32)
    dmin = raw[:, 2:4].contiguous().view(torch.float16).to(torch.float32)
    sc_b = raw[:, 4:16].view(num_blocks, 3, 4)
    sc = torch.cat(
        [sc_b[:, 0, :] & 0x3F, (sc_b[:, 2, :] & 0x0F) | ((sc_b[:, 0, :] >> 2) & 0x30)], dim=-1
    ).to(torch.float32)
    mn = torch.cat(
        [sc_b[:, 1, :] & 0x3F, (sc_b[:, 2, :] >> 4) | ((sc_b[:, 1, :] >> 2) & 0x30)], dim=-1
    ).to(torch.float32)

    d_eff = (d * sc).unsqueeze(-1)  # (B, 8, 1)
    dm_eff = (dmin * mn).unsqueeze(-1)  # (B, 8, 1)
    qs = raw[:, 16:144].view(num_blocks, 4, 1, 32)
    lo = (qs & 0x0F).to(torch.float32)
    hi = ((qs >> 4) & 0x0F).to(torch.float32)
    q = torch.cat([lo, hi], dim=2).reshape(num_blocks, 8, 32)
    out = (d_eff * q - dm_eff).reshape(n, k).t().contiguous()
    return out


def parse_gguf_q6_k_native(
    blocks_u8: torch.Tensor, k: int, n: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """Extract native `(q_i8_kn, d_sub_k16_n)` from 210-byte GGUF Q6_K super-blocks (zero re-quant)."""
    num_blocks = (k * n) // 256
    raw = blocks_u8.contiguous().view(num_blocks, 210)
    ql = raw[:, 0:128].view(num_blocks, 2, 1, 64)
    qh = raw[:, 128:192].view(num_blocks, 2, 1, 32)
    scales = raw[:, 192:208].contiguous().view(torch.int8).to(torch.float32)  # (B, 16)
    d = raw[:, 208:210].contiguous().view(torch.float16).to(torch.float32)  # (B, 1)

    ql_u = torch.cat([ql & 0x0F, (ql >> 4) & 0x0F], dim=2).reshape(num_blocks, 8, 32)
    qh_u = torch.cat(
        [qh & 0x03, (qh >> 2) & 0x03, (qh >> 4) & 0x03, (qh >> 6) & 0x03], dim=2
    ).reshape(num_blocks, 8, 32)
    q_i8 = ((ql_u | (qh_u << 4)).to(torch.int8) - 32).reshape(n, k).t().contiguous()  # (K, N)
    d_sub = (d * scales).to(torch.float16).reshape(n, k // 16).t().contiguous()  # (K//16, N)
    return q_i8, d_sub


def dequant_gguf_q6_k_reference(blocks_u8: torch.Tensor, k: int, n: int) -> torch.Tensor:
    """Bit-exact reference dequantization of GGUF Q6_K matching `gguf.quants.Q6_K.dequantize_blocks`."""
    num_blocks = (k * n) // 256
    raw = blocks_u8.contiguous().view(num_blocks, 210)
    ql = raw[:, 0:128].view(num_blocks, 2, 1, 64)
    qh = raw[:, 128:192].view(num_blocks, 2, 1, 32)
    scales = raw[:, 192:208].contiguous().view(torch.int8).to(torch.float32)
    d = raw[:, 208:210].contiguous().view(torch.float16).to(torch.float32)
    d_eff = (d * scales).view(num_blocks, 16, 1)

    ql_u = torch.cat([ql & 0x0F, (ql >> 4) & 0x0F], dim=2).reshape(num_blocks, 8, 32)
    qh_u = torch.cat(
        [qh & 0x03, (qh >> 2) & 0x03, (qh >> 4) & 0x03, (qh >> 6) & 0x03], dim=2
    ).reshape(num_blocks, 8, 32)
    q = ((ql_u | (qh_u << 4)).to(torch.int8) - 32).reshape(num_blocks, 16, 16).to(torch.float32)
    return (d_eff * q).reshape(n, k).t().contiguous()


def encode_fp32_to_gguf_q4_0(weight_kn: torch.Tensor) -> torch.Tensor:
    """Encode `(K, N)` weight matrix to raw GGUF Q4_0 `(N * K // 32, 18)` uint8 blocks."""
    try:
        from gguf import quants

        w_nk = weight_kn.detach().cpu().float().t().contiguous().numpy()
        raw = quants.Q4_0.quantize_blocks(w_nk.reshape(-1, 32))
        return torch.from_numpy(np.ascontiguousarray(raw)).to(weight_kn.device)
    except ImportError:
        k, n = weight_kn.shape
        num_blocks = (k * n) // 32
        wb = weight_kn.detach().float().t().contiguous().view(num_blocks, 32)
        idx = wb.abs().argmax(dim=-1, keepdim=True)
        max_val = torch.gather(wb, 1, idx)
        d = (max_val / -8.0).to(torch.float16)
        id_val = torch.where(d.float() != 0, 1.0 / d.float(), torch.zeros_like(d.float()))
        q = ((wb * id_val) + 8.5).to(torch.int32).clamp(0, 15).to(torch.uint8)
        qs = q[:, :16] | (q[:, 16:] << 4)
        d_u8 = d.view(torch.uint8).reshape(num_blocks, 2)
        return torch.cat([d_u8, qs], dim=1).contiguous()


def encode_fp32_to_gguf_q8_0(weight_kn: torch.Tensor) -> torch.Tensor:
    """Encode `(K, N)` weight matrix to raw GGUF Q8_0 `(N * K // 32, 34)` uint8 blocks."""
    try:
        from gguf import quants

        w_nk = weight_kn.detach().cpu().float().t().contiguous().numpy()
        raw = quants.Q8_0.quantize_blocks(w_nk.reshape(-1, 32))
        return torch.from_numpy(np.ascontiguousarray(raw)).to(weight_kn.device)
    except ImportError:
        k, n = weight_kn.shape
        num_blocks = (k * n) // 32
        wb = weight_kn.detach().float().t().contiguous().view(num_blocks, 32)
        amax = wb.abs().max(dim=-1, keepdim=True).values
        d = (amax / 127.0).to(torch.float16)
        id_val = torch.where(d.float() != 0, 1.0 / d.float(), torch.zeros_like(d.float()))
        qs = (wb * id_val).round().clamp(-128, 127).to(torch.int8).view(torch.uint8)
        d_u8 = d.view(torch.uint8).reshape(num_blocks, 2)
        return torch.cat([d_u8, qs], dim=1).contiguous()


def encode_fp32_to_gguf_q4_k(weight_kn: torch.Tensor) -> torch.Tensor:
    """Encode `(K, N)` weight matrix (`K % 256 == 0`) to raw GGUF Q4_K `(N * K // 256, 144)` blocks."""
    k, n = weight_kn.shape
    num_blocks = (k * n) // 256
    w_blocks = weight_kn.detach().float().t().contiguous().view(num_blocks, 8, 32)

    w_min = (-w_blocks.min(dim=-1).values).clamp(min=0.0)  # (B, 8)
    w_max = w_blocks.max(dim=-1).values  # (B, 8)
    sub_scale = ((w_max + w_min) / 15.0).clamp(min=1e-7)

    d = (sub_scale.max(dim=-1, keepdim=True).values / 63.0).clamp(min=1e-7).to(torch.float16)
    dmin = (w_min.max(dim=-1, keepdim=True).values / 63.0).clamp(min=1e-7).to(torch.float16)

    sc_6 = (sub_scale / d.float()).round().clamp(0, 63).to(torch.uint8)  # (B, 8)
    mn_6 = (w_min / dmin.float()).round().clamp(0, 63).to(torch.uint8)  # (B, 8)

    d_eff = (d.float() * sc_6.float()).unsqueeze(-1).clamp(min=1e-7)
    dm_eff = (dmin.float() * mn_6.float()).unsqueeze(-1)
    q_nib = ((w_blocks + dm_eff) / d_eff).round().clamp(0, 15).to(torch.uint8)  # (B, 8, 32)

    q_pairs = q_nib.view(num_blocks, 4, 2, 32)
    qs = (q_pairs[:, :, 0, :] | (q_pairs[:, :, 1, :] << 4)).reshape(num_blocks, 128)

    sc_bytes = torch.zeros((num_blocks, 12), device=weight_kn.device, dtype=torch.uint8)
    sc_bytes[:, 0:4] = (sc_6[:, 0:4] & 0x3F) | ((sc_6[:, 4:8] >> 4) << 6)
    sc_bytes[:, 4:8] = (mn_6[:, 0:4] & 0x3F) | ((mn_6[:, 4:8] >> 4) << 6)
    sc_bytes[:, 8:12] = (sc_6[:, 4:8] & 0x0F) | ((mn_6[:, 4:8] & 0x0F) << 4)

    d_u8 = d.view(torch.uint8).reshape(num_blocks, 2)
    dmin_u8 = dmin.view(torch.uint8).reshape(num_blocks, 2)
    return torch.cat([d_u8, dmin_u8, sc_bytes, qs], dim=1).contiguous()


def encode_fp32_to_gguf_q6_k(weight_kn: torch.Tensor) -> torch.Tensor:
    """Encode `(K, N)` weight matrix (`K % 256 == 0`) to raw GGUF Q6_K `(N * K // 256, 210)` blocks."""
    k, n = weight_kn.shape
    num_blocks = (k * n) // 256
    w_blocks = weight_kn.detach().float().t().contiguous().view(num_blocks, 16, 16)

    sub_amax = w_blocks.abs().max(dim=-1).values  # (B, 16)
    sub_scale = (sub_amax / 31.0).clamp(min=1e-7)
    d = (sub_scale.max(dim=-1, keepdim=True).values / 127.0).clamp(min=1e-7).to(torch.float16)
    sc_i8 = (sub_scale / d.float()).round().clamp(-127, 127).to(torch.int8)  # (B, 16)

    d_eff = (d.float() * sc_i8.float()).unsqueeze(-1)
    safe_d = torch.where(d_eff.abs() < 1e-8, torch.ones_like(d_eff), d_eff)
    q_u6 = ((w_blocks / safe_d).round().clamp(-32, 31).to(torch.int16) + 32).to(torch.uint8)
    q_8x32 = q_u6.view(num_blocks, 8, 32)

    ql_u = (q_8x32 & 0x0F).view(num_blocks, 2, 2, 64)
    ql = (ql_u[:, :, 0, :] | (ql_u[:, :, 1, :] << 4)).reshape(num_blocks, 128)

    qh_u = ((q_8x32 >> 4) & 0x03).view(num_blocks, 2, 4, 32)
    qh = (
        qh_u[:, :, 0, :]
        | (qh_u[:, :, 1, :] << 2)
        | (qh_u[:, :, 2, :] << 4)
        | (qh_u[:, :, 3, :] << 6)
    ).reshape(num_blocks, 64)

    sc_u8 = sc_i8.view(torch.uint8).reshape(num_blocks, 16)
    d_u8 = d.view(torch.uint8).reshape(num_blocks, 2)
    return torch.cat([ql, qh, sc_u8, d_u8], dim=1).contiguous()


def cutile_gemm_gguf_q4_0(
    a: torch.Tensor,
    qs_k2_n: torch.Tensor,
    d_k32_n: torch.Tensor,
    tile_shape: tuple[int, int, int] | None = None,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    """Execute native GGUF Q4_0 GEMM in CuTile (`tk=32`)."""
    m, k = a.shape
    n = qs_k2_n.shape[1]
    tm, tn, _ = tile_shape or ((16, 64, 32) if m <= 16 else (64, 64, 32))
    c = out if out is not None else torch.empty((m, n), device=a.device, dtype=torch.float16)

    if not a.is_cuda:
        lo = (qs_k2_n & 0x0F).to(torch.float32) - 8.0
        hi = ((qs_k2_n >> 4) & 0x0F).to(torch.float32) - 8.0
        w = (
            torch.cat([lo.view(k // 32, 16, n), hi.view(k // 32, 16, n)], dim=1)
            * d_k32_n.float().unsqueeze(1)
        ).reshape(k, n)
        c.copy_(torch.matmul(a.float(), w).to(torch.float16))
        return c

    grid = (ct.cdiv(m, tm), ct.cdiv(n, tn), 1)
    ct.launch(
        torch.cuda.current_stream(),
        grid,
        gemm_gguf_q4_0_kernel,
        (a.contiguous(), qs_k2_n.contiguous(), d_k32_n.contiguous(), c, tm, tn),
    )
    return c


def cutile_gemm_gguf_q4_k(
    a: torch.Tensor,
    qs_k2_n: torch.Tensor,
    d_sub_k32_n: torch.Tensor,
    dm_sub_k32_n: torch.Tensor,
    tile_shape: tuple[int, int, int] | None = None,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    """Execute native GGUF Q4_K GEMM in CuTile (`tk=64`)."""
    m, k = a.shape
    n = qs_k2_n.shape[1]
    tm, tn, _ = tile_shape or ((16, 64, 64) if m <= 16 else (64, 64, 64))
    c = out if out is not None else torch.empty((m, n), device=a.device, dtype=torch.float16)

    if not a.is_cuda:
        qs_c = qs_k2_n.view(k // 64, 32, n)
        d_p = d_sub_k32_n.float().view(k // 64, 2, 1, n)
        dm_p = dm_sub_k32_n.float().view(k // 64, 2, 1, n)
        w0 = (qs_c & 0x0F).float() * d_p[:, 0] - dm_p[:, 0]
        w1 = ((qs_c >> 4) & 0x0F).float() * d_p[:, 1] - dm_p[:, 1]
        w = torch.cat([w0, w1], dim=1).reshape(k, n)
        c.copy_(torch.matmul(a.float(), w).to(torch.float16))
        return c

    grid = (ct.cdiv(m, tm), ct.cdiv(n, tn), 1)
    ct.launch(
        torch.cuda.current_stream(),
        grid,
        gemm_gguf_q4_k_kernel,
        (
            a.contiguous(),
            qs_k2_n.contiguous(),
            d_sub_k32_n.contiguous(),
            dm_sub_k32_n.contiguous(),
            c,
            tm,
            tn,
        ),
    )
    return c


def cutile_gemm_gguf_q6_k(
    a: torch.Tensor,
    q_i8_kn: torch.Tensor,
    d_sub_k16_n: torch.Tensor,
    tile_shape: tuple[int, int, int] | None = None,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    """Execute native GGUF Q6_K GEMM in CuTile (`tk=32`)."""
    m, k = a.shape
    n = q_i8_kn.shape[1]
    tm, tn, _ = tile_shape or ((16, 64, 32) if m <= 16 else (64, 64, 32))
    c = out if out is not None else torch.empty((m, n), device=a.device, dtype=torch.float16)

    if not a.is_cuda:
        w = (q_i8_kn.float().view(k // 16, 16, n) * d_sub_k16_n.float().unsqueeze(1)).reshape(k, n)
        c.copy_(torch.matmul(a.float(), w).to(torch.float16))
        return c

    grid = (ct.cdiv(m, tm), ct.cdiv(n, tn), 1)
    ct.launch(
        torch.cuda.current_stream(),
        grid,
        gemm_gguf_q6_k_kernel,
        (a.contiguous(), q_i8_kn.contiguous(), d_sub_k16_n.contiguous(), c, tm, tn),
    )
    return c


# ---------------------------------------------------------------------------
# 4. Native Block-Scaled FP8 Kernel (`[K, N]` float8_e4m3fn + `[K//bk, N//bn]` scale)
# ---------------------------------------------------------------------------


@ct.kernel(occupancy=2)
def gemm_fp8_block_scaled_kernel(
    a_ptr,  # (M, K) fp16
    w_fp8_ptr,  # (K, N) float8_e4m3fn
    scale_ptr,  # (K // bk, N // bn) fp16
    c_ptr,  # (M, N) fp16
    tm: ct.Constant[int],
    tn: ct.Constant[int],
    tk: ct.Constant[int],
    bk: ct.Constant[int],
    bn: ct.Constant[int],
) -> None:
    """Native 2D block-scaled FP8 (`e4m3fn`) dequant + Tensor Core GEMM in CuTile."""
    pid_m = ct.bid(0)
    pid_n = ct.bid(1)
    num_k = ct.num_tiles(a_ptr, axis=1, shape=(tm, tk))
    acc = ct.zeros((tm, tn), dtype=ct.float32)

    for k in range(num_k):
        tile_a = ct.load(
            a_ptr, index=(pid_m, k), shape=(tm, tk), padding_mode=ct.PaddingMode.ZERO, latency=3
        )
        w_f8 = ct.load(
            w_fp8_ptr,
            index=(k, pid_n),
            shape=(tk, tn),
            padding_mode=ct.PaddingMode.ZERO,
            latency=3,
        )
        kb = (k * tk) // bk
        nb = (pid_n * tn) // bn
        s_blk = ct.load(
            scale_ptr,
            index=(kb, nb),
            shape=(1, 1),
            padding_mode=ct.PaddingMode.ZERO,
            latency=3,
        )
        w_fp16 = ct.astype(w_f8, ct.float16) * s_blk
        acc = ct.mma(tile_a, w_fp16, acc)

    ct.store(c_ptr, index=(pid_m, pid_n), tile=ct.astype(acc, ct.float16))


@ct.kernel(occupancy=2)
def gemm_fp8_w8a16_kernel(
    a_ptr,  # (M, K) fp16
    w_fp8_ptr,  # (K, N) float8_e4m3fn
    c_ptr,  # (M, N) fp16
    scale_b: float,
    tm: ct.Constant[int],
    tn: ct.Constant[int],
    tk: ct.Constant[int],
) -> None:
    """Native per-tensor W8A16 FP8 (`float8_e4m3fn` weight x `fp16` activation) GEMM."""
    pid_m = ct.bid(0)
    pid_n = ct.bid(1)
    num_k = ct.num_tiles(a_ptr, axis=1, shape=(tm, tk))
    acc = ct.zeros((tm, tn), dtype=ct.float32)

    for k in range(num_k):
        tile_a = ct.load(
            a_ptr, index=(pid_m, k), shape=(tm, tk), padding_mode=ct.PaddingMode.ZERO, latency=3
        )
        w_f8 = ct.load(
            w_fp8_ptr,
            index=(k, pid_n),
            shape=(tk, tn),
            padding_mode=ct.PaddingMode.ZERO,
            latency=3,
        )
        w_fp16 = ct.astype(w_f8, ct.float16)
        acc = ct.mma(tile_a, w_fp16, acc)

    ct.store(c_ptr, index=(pid_m, pid_n), tile=ct.astype(acc * scale_b, ct.float16))


def cutile_gemm_fp8_w8a16(
    a: torch.Tensor,
    weight_fp8: torch.Tensor,
    scale_b: float = 1.0,
    tile_shape: tuple[int, int, int] | None = None,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    """Execute per-tensor FP8 weight x FP16 activation GEMM without activation quantization loss."""
    m, _ = a.shape
    n = weight_fp8.shape[1]
    tm, tn, tk = tile_shape or ((16, 64, 64) if m <= 16 else (64, 64, 64))
    c = out if out is not None else torch.empty((m, n), device=a.device, dtype=torch.float16)

    if not a.is_cuda:
        w_ref = weight_fp8.float() * float(scale_b)
        c.copy_(torch.matmul(a.float(), w_ref).to(torch.float16))
        return c

    grid = (ct.cdiv(m, tm), ct.cdiv(n, tn), 1)
    ct.launch(
        torch.cuda.current_stream(),
        grid,
        gemm_fp8_w8a16_kernel,
        (a.contiguous(), weight_fp8.contiguous(), c, float(scale_b), tm, tn, tk),
    )
    return c


@ct.kernel(occupancy=2)
def gemm_fp8_channel_w8a16_kernel(
    a_ptr,  # (M, K) fp16
    w_fp8_ptr,  # (K, N) float8_e4m3fn
    scale_n_ptr,  # (N,) float32
    c_ptr,  # (M, N) fp16
    tm: ct.Constant[int],
    tn: ct.Constant[int],
    tk: ct.Constant[int],
) -> None:
    """Native per-output-channel W8A16 FP8 (`float8_e4m3fn` weight x `fp16` activation) GEMM."""
    pid_m = ct.bid(0)
    pid_n = ct.bid(1)
    num_k = ct.num_tiles(a_ptr, axis=1, shape=(tm, tk))
    acc = ct.zeros((tm, tn), dtype=ct.float32)

    for k in range(num_k):
        tile_a = ct.load(
            a_ptr, index=(pid_m, k), shape=(tm, tk), padding_mode=ct.PaddingMode.ZERO, latency=3
        )
        w_f8 = ct.load(
            w_fp8_ptr,
            index=(k, pid_n),
            shape=(tk, tn),
            padding_mode=ct.PaddingMode.ZERO,
            latency=3,
        )
        acc = ct.mma(tile_a, ct.astype(w_f8, ct.float16), acc)

    s_n = ct.load(scale_n_ptr, index=(pid_n,), shape=(tn,), padding_mode=ct.PaddingMode.ZERO)
    s_row = ct.reshape(s_n, (1, tn))
    ct.store(c_ptr, index=(pid_m, pid_n), tile=ct.astype(acc * s_row, ct.float16))


def cutile_gemm_fp8_channel_w8a16(
    a: torch.Tensor,
    weight_fp8: torch.Tensor,
    scale_n: torch.Tensor,
    tile_shape: tuple[int, int, int] | None = None,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    """Execute per-channel FP8 weight x FP16 activation GEMM natively in CuTile."""
    m, _ = a.shape
    n = weight_fp8.shape[1]
    tm, tn, tk = tile_shape or ((16, 64, 64) if m <= 16 else (64, 64, 64))
    c = out if out is not None else torch.empty((m, n), device=a.device, dtype=torch.float16)

    sc_f32 = scale_n.reshape(-1).to(device=a.device, dtype=torch.float32).contiguous()
    if not a.is_cuda:
        w_ref = weight_fp8.float() * sc_f32.view(1, n)
        c.copy_(torch.matmul(a.float(), w_ref).to(torch.float16))
        return c

    grid = (ct.cdiv(m, tm), ct.cdiv(n, tn), 1)
    ct.launch(
        torch.cuda.current_stream(),
        grid,
        gemm_fp8_channel_w8a16_kernel,
        (a.contiguous(), weight_fp8.contiguous(), sc_f32, c, tm, tn, tk),
    )
    return c


def quantize_fp8_block_scaled(
    weight: torch.Tensor,
    block_size: tuple[int, int] = (128, 128),
) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantize `(K, N)` weight matrix to `float8_e4m3fn` with 2D `(K//bk, N//bn)` block scales."""
    k_dim, n_dim = weight.shape
    bk, bn = block_size
    w_4d = weight.float().view(k_dim // bk, bk, n_dim // bn, bn)
    amax = w_4d.abs().amax(dim=(1, 3), keepdim=True)
    scale = (amax / 240.0).clamp(min=1e-6)
    w_q = (w_4d / scale).clamp(-448.0, 448.0).view(k_dim, n_dim).to(torch.float8_e4m3fn)
    return w_q.contiguous(), scale.squeeze(3).squeeze(1).to(torch.float16).contiguous()


def cutile_gemm_fp8_block_scaled(
    a: torch.Tensor,
    weight_fp8: torch.Tensor,
    scale_inv: torch.Tensor,
    block_size: tuple[int, int] = (128, 128),
    tile_shape: tuple[int, int, int] | None = None,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    """Execute 2D block-scaled FP8 GEMM natively in CuTile without HBM FP16 expansion."""
    m, k = a.shape
    n = weight_fp8.shape[1]
    bk, bn = block_size
    tm, tn, tk = tile_shape or ((16, 64, 64) if m <= 16 else (64, 64, 64))
    c = out if out is not None else torch.empty((m, n), device=a.device, dtype=torch.float16)

    if not a.is_cuda:
        s_exp = scale_inv.float().repeat_interleave(bk, dim=0).repeat_interleave(bn, dim=1)[:k, :n]
        w_ref = weight_fp8.float() * s_exp
        c.copy_(torch.matmul(a.float(), w_ref).to(torch.float16))
        return c

    grid = (ct.cdiv(m, tm), ct.cdiv(n, tn), 1)
    ct.launch(
        torch.cuda.current_stream(),
        grid,
        gemm_fp8_block_scaled_kernel,
        (
            a.contiguous(),
            weight_fp8.contiguous(),
            scale_inv.to(torch.float16).contiguous(),
            c,
            tm,
            tn,
            tk,
            bk,
            bn,
        ),
    )
    return c


# ---------------------------------------------------------------------------
# 5. Universal Quantized Linear & Mixed-Precision Per-Layer Format Table
# ---------------------------------------------------------------------------


@dataclass
class UniversalQuantLinear:
    """Polymorphic native quantized linear projection supporting per-layer mixed precision."""

    fmt: QuantFormat
    k_dim: int
    n_dim: int
    tensors: dict[str, torch.Tensor]
    group_size: int = 128
    block_size: tuple[int, int] = (128, 128)
    tile_shape: tuple[int, int, int] | None = None

    @classmethod
    def from_quant_linear_weight(
        cls,
        qw: Any,
        fmt: QuantFormat | str,
        group_size: int | None = None,
        desc_act: bool | None = None,
        block_size: tuple[int, int] = (128, 128),
    ) -> UniversalQuantLinear:
        """Build a `UniversalQuantLinear` directly from a `QuantLinearWeight` checkpoint tensor."""
        qfmt = QuantFormat(str(fmt).lower())
        k = int(qw.qweight.shape[0] * 2)
        n = int(qw.qweight.shape[1])
        src_gs = int(qw.group_size)
        eff_gs = src_gs if group_size is None else (k if group_size <= 0 else group_size)
        use_desc = (qw.perm is not None) if desc_act is None else desc_act

        if qfmt == QuantFormat.GPTQ and eff_gs == src_gs and use_desc == (qw.perm is not None):
            t_dict: dict[str, torch.Tensor] = {
                "qweight": qw.qweight.contiguous(),
                "scales": qw.scales.contiguous(),
                "zeros": qw.zeros.contiguous(),
            }
            if qw.perm is not None and use_desc:
                t_dict["perm"] = qw.perm.to(dtype=torch.int32).contiguous()
            return cls(qfmt, k, n, t_dict, group_size=eff_gs)

        w_fp16 = unpack_int4_weights_torch(qw.qweight, qw.scales, qw.zeros, group_size=src_gs).to(
            torch.float16
        )
        if qw.perm is not None:
            inv = torch.argsort(qw.perm.long())
            w_fp16 = w_fp16[inv]
        return cls.from_fp16(
            w_fp16,
            fmt=qfmt,
            group_size=group_size if group_size is not None else 128,
            desc_act=bool(use_desc),
            block_size=block_size,
        )

    @classmethod
    def from_fp16(
        cls,
        weight_kn: torch.Tensor,
        fmt: QuantFormat | str,
        group_size: int = 128,
        desc_act: bool = False,
        block_size: tuple[int, int] = (128, 128),
    ) -> UniversalQuantLinear:
        """Encode an existing `(K, N)` FP16/FP32 weight matrix into target native format `fmt`."""
        qfmt = QuantFormat(str(fmt).lower())
        k, n = weight_kn.shape
        dev = weight_kn.device
        eff_gs = k if group_size <= 0 else group_size

        if qfmt == QuantFormat.FP16:
            return cls(qfmt, k, n, {"weight": weight_kn.to(torch.float16).contiguous()})

        if qfmt == QuantFormat.GPTQ:
            from cutile_forge.utils.quantize import quantize_4bit_group

            perm: torch.Tensor | None = None
            w_src = weight_kn
            if desc_act:
                col_norm = weight_kn.float().norm(dim=1)
                perm = torch.argsort(col_norm, descending=True).to(torch.int32)
                w_src = weight_kn[perm.long()]
            qw, sc, zr = quantize_4bit_group(w_src, group_size=eff_gs)
            t_dict: dict[str, torch.Tensor] = {"qweight": qw, "scales": sc, "zeros": zr}
            if perm is not None:
                t_dict["perm"] = perm
            return cls(qfmt, k, n, t_dict, group_size=eff_gs)

        if qfmt == QuantFormat.AWQ:
            w_f = weight_kn.float().view(k // eff_gs, eff_gs, n)
            w_min = w_f.min(dim=1, keepdim=True).values
            w_max = w_f.max(dim=1, keepdim=True).values
            sc = ((w_max - w_min) / 15.0).clamp(min=1e-6)
            zr = (-w_min / sc).round().clamp(0.0, 15.0).to(torch.int32)
            q_i = (w_f / sc + zr.float()).round().clamp(0.0, 15.0).to(torch.int32).view(k, n)
            shifts = torch.tensor([0, 16, 4, 20, 8, 24, 12, 28], device=dev, dtype=torch.int32)
            qw_awq = (q_i.view(k, n // 8, 8) << shifts.view(1, 1, 8)).sum(dim=2).to(torch.int32)
            qz_awq = (
                (zr.squeeze(1).view(k // eff_gs, n // 8, 8) << shifts.view(1, 1, 8))
                .sum(dim=2)
                .to(torch.int32)
            )
            return cls(
                qfmt,
                k,
                n,
                {
                    "qweight_awq": qw_awq.contiguous(),
                    "qzeros_awq": qz_awq.contiguous(),
                    "scales": sc.squeeze(1).to(torch.float16).contiguous(),
                },
                group_size=eff_gs,
            )

        if qfmt == QuantFormat.GGUF_Q4_0:
            raw = encode_fp32_to_gguf_q4_0(weight_kn)
            qs, d = parse_gguf_q4_0_native(raw, k, n)
            return cls(qfmt, k, n, {"qs": qs, "d": d, "raw_blocks": raw}, group_size=32)

        if qfmt == QuantFormat.GGUF_Q8_0:
            raw = encode_fp32_to_gguf_q8_0(weight_kn)
            qs_i8, d = parse_gguf_q8_0_native(raw, k, n)
            return cls(qfmt, k, n, {"qw_i8": qs_i8, "scales": d, "raw_blocks": raw}, group_size=32)

        if qfmt in (QuantFormat.GGUF_Q4_K, QuantFormat.GGUF_Q4_K_M):
            raw = encode_fp32_to_gguf_q4_k(weight_kn)
            qs, d_sub, dm_sub = parse_gguf_q4_k_native(raw, k, n)
            return cls(
                QuantFormat.GGUF_Q4_K,
                k,
                n,
                {"qs": qs, "d_sub": d_sub, "dm_sub": dm_sub, "raw_blocks": raw},
                group_size=32,
            )

        if qfmt == QuantFormat.GGUF_Q6_K:
            raw = encode_fp32_to_gguf_q6_k(weight_kn)
            q_i8, d_sub = parse_gguf_q6_k_native(raw, k, n)
            return cls(
                qfmt,
                k,
                n,
                {"q_i8": q_i8, "d_sub": d_sub, "raw_blocks": raw},
                group_size=16,
            )

        if qfmt == QuantFormat.INT8:
            qw_i8, sc = quantize_int8_group(weight_kn, group_size=eff_gs)
            return cls(qfmt, k, n, {"qw_i8": qw_i8, "scales": sc}, group_size=eff_gs)

        if qfmt == QuantFormat.FP8:
            amax = weight_kn.float().abs().max().clamp(min=1e-6)
            scale_b = float((amax / 240.0).item())
            w_fp8 = (weight_kn.float() / scale_b).clamp(-448.0, 448.0).to(torch.float8_e4m3fn)
            return cls(
                qfmt,
                k,
                n,
                {
                    "weight_fp8": w_fp8.contiguous(),
                    "scale": torch.tensor(scale_b, device=dev, dtype=torch.float32),
                },
            )

        if qfmt == QuantFormat.FP8_BLOCK:
            w_fp8, sc_blk = quantize_fp8_block_scaled(weight_kn, block_size=block_size)
            return cls(
                qfmt,
                k,
                n,
                {"weight_fp8": w_fp8, "scale_inv": sc_blk},
                block_size=block_size,
            )

        raise ValueError(f"Unsupported QuantFormat: {fmt}")

    def forward_native(self, x: torch.Tensor, out: torch.Tensor | None = None) -> torch.Tensor:
        """Execute this projection natively on GPU using the format's dedicated CuTile kernel."""
        if not x.is_cuda:
            res_cpu = self.forward_reference(x)
            if out is not None:
                out.copy_(res_cpu.view(-1, self.n_dim))
                return out.view(*x.shape[:-1], self.n_dim)
            return res_cpu
        orig_prefix = x.shape[:-1]
        x_2d = x.reshape(-1, self.k_dim)

        if self.fmt == QuantFormat.GPTQ:
            perm = self.tensors.get("perm")
            x_in = cutile_col_permute(x_2d, perm) if perm is not None else x_2d
            res = fused_gemm_dequant(
                x_in,
                self.tensors["qweight"],
                self.tensors["scales"],
                self.tensors["zeros"],
                group_size=self.group_size,
                tile_shape=self.tile_shape,
                out=out,
            )
        elif self.fmt == QuantFormat.AWQ:
            res = cutile_gemm_awq_native(
                x_2d,
                self.tensors["qweight_awq"],
                self.tensors["qzeros_awq"],
                self.tensors["scales"],
                group_size=self.group_size,
                tile_shape=self.tile_shape,
                out=out,
            )
        elif self.fmt == QuantFormat.GGUF_Q4_0:
            res = cutile_gemm_gguf_q4_0(
                x_2d, self.tensors["qs"], self.tensors["d"], tile_shape=self.tile_shape, out=out
            )
        elif self.fmt in (QuantFormat.GGUF_Q8_0, QuantFormat.INT8):
            res = cutile_gemm_int8_dequant(
                x_2d,
                self.tensors["qw_i8"],
                self.tensors["scales"],
                group_size=self.group_size,
                tile_shape=self.tile_shape,
                out=out,
            )
        elif self.fmt in (QuantFormat.GGUF_Q4_K, QuantFormat.GGUF_Q4_K_M):
            res = cutile_gemm_gguf_q4_k(
                x_2d,
                self.tensors["qs"],
                self.tensors["d_sub"],
                self.tensors["dm_sub"],
                tile_shape=self.tile_shape,
                out=out,
            )
        elif self.fmt == QuantFormat.GGUF_Q6_K:
            res = cutile_gemm_gguf_q6_k(
                x_2d,
                self.tensors["q_i8"],
                self.tensors["d_sub"],
                tile_shape=self.tile_shape,
                out=out,
            )
        elif self.fmt == QuantFormat.FP8:
            sc_b = float(self.tensors["scale"].item())
            res = cutile_gemm_fp8_w8a16(
                x_2d,
                self.tensors["weight_fp8"],
                scale_b=sc_b,
                tile_shape=self.tile_shape,
                out=out,
            )
        elif self.fmt == QuantFormat.FP8_CHANNEL:
            res = cutile_gemm_fp8_channel_w8a16(
                x_2d,
                self.tensors["weight_fp8"],
                self.tensors["scale_n"],
                tile_shape=self.tile_shape,
                out=out,
            )
        elif self.fmt == QuantFormat.FP8_BLOCK:
            res = cutile_gemm_fp8_block_scaled(
                x_2d,
                self.tensors["weight_fp8"],
                self.tensors["scale_inv"],
                block_size=self.block_size,
                tile_shape=self.tile_shape,
                out=out,
            )
        else:
            from cutile_forge.kernels.gemm_dequant import cutile_gemm_fp16

            res = (
                cutile_gemm_fp16(x_2d, self.tensors["weight"], tile_shape=self.tile_shape)
                if x_2d.is_cuda
                else torch.matmul(x_2d.float(), self.tensors["weight"].float()).to(torch.float16)
            )
        return res.view(*orig_prefix, self.n_dim)

    def dequantize_reference(self) -> torch.Tensor:
        """Dequantize weights using the exact format reference implementation (`(K, N)` FP16)."""
        if self.fmt == QuantFormat.GPTQ:
            w = unpack_int4_weights_torch(
                self.tensors["qweight"],
                self.tensors["scales"],
                self.tensors["zeros"],
                group_size=self.group_size,
            )
            perm = self.tensors.get("perm")
            if perm is not None:
                inv = torch.argsort(perm.long())
                w = w[inv]
            return w.to(torch.float16)
        if self.fmt == QuantFormat.AWQ:
            return dequant_awq_reference(
                self.tensors["qweight_awq"],
                self.tensors["qzeros_awq"],
                self.tensors["scales"],
                group_size=self.group_size,
            )
        if self.fmt == QuantFormat.GGUF_Q4_0:
            return dequant_gguf_q4_0_reference(
                self.tensors["raw_blocks"], self.k_dim, self.n_dim
            ).to(torch.float16)
        if self.fmt == QuantFormat.GGUF_Q8_0:
            return dequant_gguf_q8_0_reference(
                self.tensors["raw_blocks"], self.k_dim, self.n_dim
            ).to(torch.float16)
        if self.fmt in (QuantFormat.GGUF_Q4_K, QuantFormat.GGUF_Q4_K_M):
            return dequant_gguf_q4_k_reference(
                self.tensors["raw_blocks"], self.k_dim, self.n_dim
            ).to(torch.float16)
        if self.fmt == QuantFormat.GGUF_Q6_K:
            return dequant_gguf_q6_k_reference(
                self.tensors["raw_blocks"], self.k_dim, self.n_dim
            ).to(torch.float16)
        if self.fmt == QuantFormat.INT8:
            return dequantize_int8_group(
                self.tensors["qw_i8"], self.tensors["scales"], group_size=self.group_size
            )
        if self.fmt == QuantFormat.FP8:
            return (self.tensors["weight_fp8"].float() * float(self.tensors["scale"].item())).to(
                torch.float16
            )
        if self.fmt == QuantFormat.FP8_CHANNEL:
            sc_n = self.tensors["scale_n"].float().view(1, self.n_dim)
            return (self.tensors["weight_fp8"].float() * sc_n).to(torch.float16)
        if self.fmt == QuantFormat.FP8_BLOCK:
            bk, bn = self.block_size
            s_exp = (
                self.tensors["scale_inv"]
                .float()
                .repeat_interleave(bk, dim=0)
                .repeat_interleave(bn, dim=1)[: self.k_dim, : self.n_dim]
            )
            return (self.tensors["weight_fp8"].float() * s_exp).to(torch.float16)
        return self.tensors["weight"].to(torch.float16)

    def forward_reference(self, x: torch.Tensor) -> torch.Tensor:
        """Reference FP32 matmul against exact reference-dequantized weights."""
        orig_prefix = x.shape[:-1]
        x_2d = x.reshape(-1, self.k_dim).float()
        w_ref = self.dequantize_reference().float()
        return torch.matmul(x_2d, w_ref).to(x.dtype).view(*orig_prefix, self.n_dim)


@dataclass
class MixedPrecisionFormatTable:
    """Per-layer & per-projection format assignment table (e.g., GGUF Q4_K_M mixing Q4_K and Q6_K)."""

    default_format: QuantFormat = QuantFormat.GPTQ
    layer_overrides: dict[int, QuantFormat] | None = None
    projection_overrides: dict[tuple[int, str], QuantFormat] | None = None

    @classmethod
    def gguf_q4_k_m_table(cls, num_layers: int) -> MixedPrecisionFormatTable:
        """Standard llama.cpp `Q4_K_M` policy: `v_proj` & `down_proj` use `Q6_K`, others use `Q4_K`."""
        proj_map: dict[tuple[int, str], QuantFormat] = {}
        for l_idx in range(num_layers):
            proj_map[(l_idx, "v_proj")] = QuantFormat.GGUF_Q6_K
            proj_map[(l_idx, "down_proj")] = QuantFormat.GGUF_Q6_K
        return cls(
            default_format=QuantFormat.GGUF_Q4_K,
            projection_overrides=proj_map,
        )

    def resolve(self, layer_idx: int, proj_name: str) -> QuantFormat:
        if self.projection_overrides and (layer_idx, proj_name) in self.projection_overrides:
            return self.projection_overrides[(layer_idx, proj_name)]
        if self.layer_overrides and layer_idx in self.layer_overrides:
            return self.layer_overrides[layer_idx]
        return self.default_format

    def summary(self, num_layers: int) -> dict[str, int]:
        counts: dict[str, int] = {}
        projs = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")
        for l_idx in range(num_layers):
            for p in projs:
                fmt = self.resolve(l_idx, p).value
                counts[fmt] = counts.get(fmt, 0) + 1
        return counts


@dataclass
class UniversalQuantLayer:
    """Single Transformer layer with 7 projections governed by `MixedPrecisionFormatTable`."""

    layer_idx: int
    input_layernorm: torch.Tensor
    post_attention_layernorm: torch.Tensor
    q_proj: UniversalQuantLinear
    k_proj: UniversalQuantLinear
    v_proj: UniversalQuantLinear
    o_proj: UniversalQuantLinear
    gate_proj: UniversalQuantLinear
    up_proj: UniversalQuantLinear
    down_proj: UniversalQuantLinear
    q_bias: torch.Tensor | None = None
    k_bias: torch.Tensor | None = None
    v_bias: torch.Tensor | None = None


class UniversalQuantModel:
    """End-to-end mixed-precision / multi-format Transformer execution & perplexity evaluator."""

    def __init__(
        self,
        config: Any,
        embed_tokens: torch.Tensor,
        norm: torch.Tensor,
        lm_head: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        layers: list[UniversalQuantLayer],
        format_table: MixedPrecisionFormatTable,
        rotary_style: str = "half",
    ) -> None:
        self.config = config
        self.device = embed_tokens.device
        self.embed_tokens = embed_tokens.contiguous()
        self.norm = norm.contiguous()
        self.lm_head = lm_head.contiguous()
        self.cos = cos.contiguous()
        self.sin = sin.contiguous()
        self.layers = layers
        self.format_table = format_table
        self.rotary_style = rotary_style

    @classmethod
    def from_mistral_model(
        cls,
        model: Any,
        format_table: MixedPrecisionFormatTable | QuantFormat | str,
        group_size: int | None = None,
        desc_act: bool | None = None,
        block_size: tuple[int, int] = (128, 128),
        num_layers: int | None = None,
    ) -> UniversalQuantModel:
        """Convert a loaded `Mistral7B` (or subset of layers) into a multi-format `UniversalQuantModel`."""
        table = (
            format_table
            if isinstance(format_table, MixedPrecisionFormatTable)
            else (
                MixedPrecisionFormatTable.gguf_q4_k_m_table(model.config.num_hidden_layers)
                if QuantFormat(str(format_table).lower()) == QuantFormat.GGUF_Q4_K_M
                else MixedPrecisionFormatTable(
                    default_format=QuantFormat(str(format_table).lower())
                )
            )
        )
        dev = model.device
        limit = num_layers if num_layers is not None else len(model.weights.layers)
        u_layers: list[UniversalQuantLayer] = []
        for idx in range(limit):
            lw = model.weights.layers[idx]

            def _build_proj(proj_name: str, qw: Any, l_idx: int = idx) -> UniversalQuantLinear:
                p_fmt = table.resolve(l_idx, proj_name)
                return UniversalQuantLinear.from_quant_linear_weight(
                    qw.to(dev),
                    fmt=p_fmt,
                    group_size=group_size,
                    desc_act=desc_act,
                    block_size=block_size,
                )

            u_layers.append(
                UniversalQuantLayer(
                    layer_idx=idx,
                    input_layernorm=lw.input_layernorm.to(dev).contiguous(),
                    post_attention_layernorm=lw.post_attention_layernorm.to(dev).contiguous(),
                    q_proj=_build_proj("q_proj", lw.q_proj),
                    k_proj=_build_proj("k_proj", lw.k_proj),
                    v_proj=_build_proj("v_proj", lw.v_proj),
                    o_proj=_build_proj("o_proj", lw.o_proj),
                    gate_proj=_build_proj("gate_proj", lw.gate_proj),
                    up_proj=_build_proj("up_proj", lw.up_proj),
                    down_proj=_build_proj("down_proj", lw.down_proj),
                )
            )

        return cls(
            config=model.config,
            embed_tokens=model.weights.embed_tokens.to(dev),
            norm=model.weights.norm.to(dev),
            lm_head=model.weights.lm_head.to(dev),
            cos=model.cos.to(dev),
            sin=model.sin.to(dev),
            layers=u_layers,
            format_table=table,
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        mode: str = "native",
    ) -> torch.Tensor:
        """Run full forward pass (`mode='native'` uses CuTile kernels; `mode='reference'` uses FP32 ref)."""
        from cutile_forge.kernels.activations import cutile_silu
        from cutile_forge.kernels.attention_gqa import fused_attention_gqa
        from cutile_forge.kernels.embedding_lmhead import (
            cutile_embedding_lookup,
            cutile_ew_add,
            cutile_ew_mul,
        )
        from cutile_forge.kernels.rmsnorm import cutile_rmsnorm
        from cutile_forge.kernels.rope import apply_rope

        use_native = mode == "native" and self.device.type == "cuda"
        b, s = input_ids.shape
        cfg = self.config
        h = (
            cutile_embedding_lookup(input_ids, self.embed_tokens)
            if use_native
            else torch.nn.functional.embedding(input_ids, self.embed_tokens)
        )

        def _rmsnorm(inp: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
            if use_native:
                return cutile_rmsnorm(inp, w, eps=cfg.rms_norm_eps)
            var = inp.float().pow(2).mean(dim=-1, keepdim=True)
            return (inp.float() * torch.rsqrt(var + cfg.rms_norm_eps) * w.float()).to(inp.dtype)

        for lyr in self.layers:
            norm_in = _rmsnorm(h, lyr.input_layernorm)
            q = (
                lyr.q_proj.forward_native(norm_in)
                if mode == "native"
                else lyr.q_proj.forward_reference(norm_in)
            )
            k = (
                lyr.k_proj.forward_native(norm_in)
                if mode == "native"
                else lyr.k_proj.forward_reference(norm_in)
            )
            v = (
                lyr.v_proj.forward_native(norm_in)
                if mode == "native"
                else lyr.v_proj.forward_reference(norm_in)
            )
            if lyr.q_bias is not None:
                q = q + lyr.q_bias.to(q.dtype)
            if lyr.k_bias is not None:
                k = k + lyr.k_bias.to(k.dtype)
            if lyr.v_bias is not None:
                v = v + lyr.v_bias.to(v.dtype)

            q = q.view(b, s, cfg.num_attention_heads, cfg.head_dim).transpose(1, 2).contiguous()
            k = k.view(b, s, cfg.num_key_value_heads, cfg.head_dim).transpose(1, 2).contiguous()
            v = v.view(b, s, cfg.num_key_value_heads, cfg.head_dim).transpose(1, 2).contiguous()
            if self.rotary_style == "interleaved":
                q = torch.cat([q[..., 0::2], q[..., 1::2]], dim=-1).contiguous()
                k = torch.cat([k[..., 0::2], k[..., 1::2]], dim=-1).contiguous()

            sw = int(getattr(cfg, "sliding_window", 0) or 0)
            if use_native:
                q = apply_rope(q, self.cos, self.sin, 0)
                k = apply_rope(k, self.cos, self.sin, 0)
                attn_out = fused_attention_gqa(q, k, v, causal=True, sliding_window=sw)
            else:
                half_d = cfg.head_dim // 2
                c = self.cos[:s, :half_d].view(1, 1, s, half_d).float()
                sn = self.sin[:s, :half_d].view(1, 1, s, half_d).float()
                q0, q1 = q[..., :half_d].float(), q[..., half_d:].float()
                k0, k1 = k[..., :half_d].float(), k[..., half_d:].float()
                q = torch.cat([q0 * c - q1 * sn, q1 * c + q0 * sn], dim=-1).to(h.dtype)
                k = torch.cat([k0 * c - k1 * sn, k1 * c + k0 * sn], dim=-1).to(h.dtype)
                gqa_groups = cfg.num_attention_heads // cfg.num_key_value_heads
                k_exp = k.repeat_interleave(gqa_groups, dim=1)
                v_exp = v.repeat_interleave(gqa_groups, dim=1)
                if sw > 0:
                    q_p = torch.arange(s, device=q.device).view(s, 1)
                    k_p = torch.arange(s, device=q.device).view(1, s)
                    mask = (k_p <= q_p) & (k_p > q_p - sw)
                    attn_out = torch.nn.functional.scaled_dot_product_attention(
                        q.float(), k_exp.float(), v_exp.float(), attn_mask=mask
                    ).to(h.dtype)
                else:
                    attn_out = torch.nn.functional.scaled_dot_product_attention(
                        q.float(), k_exp.float(), v_exp.float(), is_causal=True
                    ).to(h.dtype)

            attn_flat = attn_out.transpose(1, 2).contiguous().view(b, s, cfg.hidden_size)
            o_out = (
                lyr.o_proj.forward_native(attn_flat)
                if mode == "native"
                else lyr.o_proj.forward_reference(attn_flat)
            )
            h = cutile_ew_add(h, o_out) if use_native else (h + o_out)

            norm_post = _rmsnorm(h, lyr.post_attention_layernorm)
            gate = (
                lyr.gate_proj.forward_native(norm_post)
                if mode == "native"
                else lyr.gate_proj.forward_reference(norm_post)
            )
            up = (
                lyr.up_proj.forward_native(norm_post)
                if mode == "native"
                else lyr.up_proj.forward_reference(norm_post)
            )
            act = (
                cutile_ew_mul(cutile_silu(gate), up)
                if use_native
                else (torch.nn.functional.silu(gate.float()) * up.float()).to(h.dtype)
            )
            down = (
                lyr.down_proj.forward_native(act)
                if mode == "native"
                else lyr.down_proj.forward_reference(act)
            )
            h = cutile_ew_add(h, down) if use_native else (h + down)

        h_norm = _rmsnorm(h, self.norm)
        logits = torch.matmul(h_norm.float(), self.lm_head.float().t())
        return logits

    def compute_perplexity(
        self,
        input_ids: torch.Tensor,
        mode: str = "native",
    ) -> float:
        """Compute token-level perplexity (`exp(cross_entropy)`) over `input_ids` (`(1, S)`)."""
        with torch.no_grad():
            logits = self.forward(input_ids, mode=mode)
            shift_logits = logits[:, :-1, :].contiguous().float()
            shift_labels = input_ids[:, 1:].contiguous().long()
            loss = torch.nn.functional.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
            )
            return float(torch.exp(loss).item())
