"""CuTile kernels for FP16 GEMM and fused INT4-dequant GEMM."""

from __future__ import annotations

import cuda.tile as ct
import torch


@ct.kernel(occupancy=2)
def gemm_fp16_kernel(
    a_ptr,
    b_ptr,
    c_ptr,
    tm: ct.Constant[int],
    tn: ct.Constant[int],
    tk: ct.Constant[int],
) -> None:
    """Tiled FP16 GEMM with 2D L2 cache swizzling and 3-stage software pipelining."""
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
            a_ptr,
            index=(pid_m, k),
            shape=(tm, tk),
            padding_mode=ct.PaddingMode.ZERO,
            latency=3,
        )
        tile_b = ct.load(
            b_ptr,
            index=(k, pid_n),
            shape=(tk, tn),
            padding_mode=ct.PaddingMode.ZERO,
            latency=3,
        )
        acc = ct.mma(tile_a, tile_b, acc)
    ct.store(c_ptr, index=(pid_m, pid_n), tile=ct.astype(acc, ct.float16))


@ct.kernel(occupancy=2)
def gemm_dequant_kernel(
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
    """Fused INT4-dequant + FP16 GEMM kernel (3-stage pipeline, L2 swizzle).

    `qw_ptr` has shape (K // 2, N) uint8 where each byte packs two 4-bit weights
    along the K dimension (even K in low nibble, odd K in high nibble).
    `scales_ptr` and `zeros_ptr` have shape (K // gs, N) float16.
    """
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
            a_ptr,
            index=(pid_m, k),
            shape=(tm, tk),
            padding_mode=ct.PaddingMode.ZERO,
            latency=3,
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
        lo_e = ct.reshape(lo, (half_tk, 1, tn))
        hi_e = ct.reshape(hi, (half_tk, 1, tn))
        w_int = ct.reshape(ct.cat((lo_e, hi_e), axis=1), (tk, tn))

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

    ct.store(c_ptr, index=(pid_m, pid_n), tile=ct.astype(acc, ct.float16))


def _select_gemm_tiles(m: int, group_size: int = 128) -> tuple[int, int, int]:
    """Select optimal tile shape (TM, TN, TK) based on batch/sequence length M."""
    tk = 64 if group_size >= 64 else 32
    if m <= 16:
        return (16, 64, tk)
    if m <= 64:
        return (64, 128, tk)
    return (128, 128, 32 if group_size >= 32 else tk)


def pack_int4_weights(w_uint8: torch.Tensor) -> torch.Tensor:
    """Pack uint8 values in [0, 15] of shape (K, N) into (K // 2, N) uint8."""
    if w_uint8.shape[0] % 2 != 0:
        raise ValueError(f"K dimension must be even, got {w_uint8.shape[0]}")
    lo = w_uint8[0::2, :] & 0x0F
    hi = (w_uint8[1::2, :] & 0x0F) << 4
    return (lo | hi).contiguous()


def unpack_int4_weights_torch(
    qweight: torch.Tensor,
    scales: torch.Tensor,
    zeros: torch.Tensor,
    group_size: int = 128,
) -> torch.Tensor:
    """Reference PyTorch dequantization for (K // 2, N) uint8 weights."""
    qw = qweight.to(torch.uint8)
    lo = (qw & 0x0F).to(torch.float16)
    hi = ((qw >> 4) & 0x0F).to(torch.float16)
    k_half, n = qw.shape
    w_int = torch.stack((lo, hi), dim=1).reshape(k_half * 2, n)
    scales_exp = scales.repeat_interleave(group_size, dim=0)[: k_half * 2]
    zeros_exp = zeros.repeat_interleave(group_size, dim=0)[: k_half * 2]
    return (w_int - zeros_exp) * scales_exp


def cutile_gemm_fp16(
    a: torch.Tensor,
    b: torch.Tensor,
    tile_shape: tuple[int, int, int] | None = None,
) -> torch.Tensor:
    """Run tiled FP16 GEMM: (M, K) @ (K, N) -> (M, N) using CuTile."""
    if a.ndim != 2 or b.ndim != 2:
        raise ValueError(f"Expected 2D tensors, got {a.shape} and {b.shape}")
    m, k = a.shape
    k_b, n = b.shape
    if k != k_b:
        raise ValueError(f"Shape mismatch: {a.shape} vs {b.shape}")

    a_c = a.contiguous()
    b_c = b.contiguous()
    c = torch.empty((m, n), device=a.device, dtype=torch.float16)

    tm, tn, tk = tile_shape or _select_gemm_tiles(m)
    grid = (ct.cdiv(m, tm) * ct.cdiv(n, tn), 1, 1)
    ct.launch(torch.cuda.current_stream(), grid, gemm_fp16_kernel, (a_c, b_c, c, tm, tn, tk))
    return c


def fused_gemm_dequant(
    a: torch.Tensor,
    qweight: torch.Tensor,
    scales: torch.Tensor,
    zeros: torch.Tensor,
    group_size: int = 128,
    tile_shape: tuple[int, int, int] | None = None,
) -> torch.Tensor:
    """Run fused INT4 dequantization + GEMM in a single CuTile kernel.

    Args:
        a: Input activations of shape (M, K), float16, CUDA.
        qweight: Packed 4-bit weights of shape (K // 2, N), uint8, CUDA.
        scales: Per-group scales of shape (K // group_size, N), float16, CUDA.
        zeros: Per-group zero points of shape (K // group_size, N), float16, CUDA.
        group_size: Quantization group size along K (default 128).
        tile_shape: Optional override (TM, TN, TK).

    Returns:
        Output tensor of shape (M, N), float16, CUDA.
    """
    m, k = a.shape
    if qweight.dtype == torch.int32:
        # Convert GPTQ int32 (K // 8, N) to (K // 2, N) uint8
        k_div8, n = qweight.shape
        shifts = torch.arange(0, 32, 4, device=qweight.device, dtype=torch.int32).view(1, 8, 1)
        nibbles = ((qweight.unsqueeze(1) >> shifts) & 0xF).to(torch.uint8).reshape(k_div8 * 8, n)
        qw_u8 = pack_int4_weights(nibbles)
    else:
        qw_u8 = qweight.contiguous()

    n = qw_u8.shape[1]
    if qw_u8.shape[0] * 2 != k:
        raise ValueError(f"Weight K mismatch: {qw_u8.shape[0] * 2} vs {k}")

    tm, tn, tk = tile_shape or _select_gemm_tiles(m, group_size=group_size)
    if group_size % tk != 0:
        tk = 64 if group_size % 64 == 0 else (32 if group_size % 32 == 0 else group_size)

    a_c = a.contiguous()
    s_c = scales.contiguous()
    z_c = zeros.contiguous()
    c = torch.empty((m, n), device=a.device, dtype=torch.float16)

    grid = (ct.cdiv(m, tm) * ct.cdiv(n, tn), 1, 1)
    ct.launch(
        torch.cuda.current_stream(),
        grid,
        gemm_dequant_kernel,
        (a_c, qw_u8, s_c, z_c, c, tm, tn, tk, group_size),
    )
    return c
