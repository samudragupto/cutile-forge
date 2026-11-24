"""CuTile kernels for FP16 GEMM, FP8 Tensor Core GEMM (sm_120), and fused INT4-dequant GEMM."""

from __future__ import annotations

from typing import Any

import cuda.tile as ct
import torch

ct.float8 = getattr(ct, "float8", ct.float8_e4m3fn)
cutile = ct


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
def gemm_fp16_bias_kernel(
    a_ptr,
    b_ptr,
    bias_ptr,
    c_ptr,
    tm: ct.Constant[int],
    tn: ct.Constant[int],
    tk: ct.Constant[int],
) -> None:
    """Tiled FP16 GEMM + 1D bias epilogue `(1, N)` in registers."""
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
    bias_t = ct.load(
        bias_ptr, index=(0, pid_n), shape=(1, tn), padding_mode=ct.PaddingMode.ZERO, latency=2
    )
    ct.store(c_ptr, index=(pid_m, pid_n), tile=ct.astype(acc, ct.float16) + bias_t)


@ct.kernel(occupancy=2)
def gemm_fp16_add_kernel(
    a_ptr,
    b_ptr,
    res_ptr,
    c_ptr,
    tm: ct.Constant[int],
    tn: ct.Constant[int],
    tk: ct.Constant[int],
) -> None:
    """Tiled FP16 GEMM + `(M, N)` residual add epilogue in registers."""
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
    res_t = ct.load(
        res_ptr, index=(pid_m, pid_n), shape=(tm, tn), padding_mode=ct.PaddingMode.ZERO, latency=2
    )
    ct.store(c_ptr, index=(pid_m, pid_n), tile=ct.astype(acc, ct.float16) + res_t)


@ct.kernel(occupancy=2)
def gemm_fp8_kernel(
    a_ptr,
    b_ptr,
    scale_a: float,
    scale_b: float,
    c_ptr,
    tm: ct.Constant[int],
    tn: ct.Constant[int],
    tk: ct.Constant[int],
) -> None:
    """Tiled FP8 (dtype=cutile.float8) Tensor Core GEMM with 4-stage software pipelining (sm_120)."""
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
        tile_a = ct.astype(
            ct.load(
                a_ptr,
                index=(pid_m, k),
                shape=(tm, tk),
                padding_mode=ct.PaddingMode.ZERO,
                latency=4,
            ),
            cutile.float8,
        )
        tile_b = ct.astype(
            ct.load(
                b_ptr,
                index=(k, pid_n),
                shape=(tk, tn),
                padding_mode=ct.PaddingMode.ZERO,
                latency=4,
            ),
            cutile.float8,
        )
        acc = ct.mma(tile_a, tile_b, acc)
    out = acc * (scale_a * scale_b)
    ct.store(c_ptr, index=(pid_m, pid_n), tile=ct.astype(out, ct.float16))


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


@ct.kernel(occupancy=2)
def gemm_dequant_add_kernel(
    a_ptr,
    qw_ptr,
    scales_ptr,
    zeros_ptr,
    res_ptr,
    c_ptr,
    tm: ct.Constant[int],
    tn: ct.Constant[int],
    tk: ct.Constant[int],
    gs: ct.Constant[int],
) -> None:
    """Fused INT4-dequant GEMM + residual epilogue `c = ct.astype(acc, fp16) + res`."""
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

    res_tile = ct.load(
        res_ptr,
        index=(pid_m, pid_n),
        shape=(tm, tn),
        padding_mode=ct.PaddingMode.ZERO,
        latency=2,
    )
    out = ct.astype(acc, ct.float16) + res_tile
    ct.store(c_ptr, index=(pid_m, pid_n), tile=out)


def _select_gemm_tiles(m: int, group_size: int = 128, n: int = 4096) -> tuple[int, int, int]:
    """Select optimal tile shape (TM, TN, TK) based on batch/sequence length M and N."""
    if m <= 16:
        if group_size >= 128 and n % 64 == 0:
            return (16, 64, 128)
        if group_size >= 128 and n <= 6144 and (n % 32 == 0):
            return (16, 32, 128)
        tk = 64 if group_size >= 64 else 32
        return (16, 64, tk)
    tk = 64 if group_size >= 64 else 32
    return (64, 128, tk)


@ct.kernel(occupancy=4)
def gemm_dequant_splitk_kernel(
    a_ptr,
    qw_ptr,
    scales_ptr,
    zeros_ptr,
    part_ptr,
    split_k: ct.Constant[int],
    tm: ct.Constant[int],
    tn: ct.Constant[int],
    tk: ct.Constant[int],
    gs: ct.Constant[int],
) -> None:
    """Split-K INT4-dequant + FP16 GEMM writing FP32 partials `(split_k, M, N)`."""
    pid_m = ct.bid(0)
    pid_n = ct.bid(1)
    sk_id = ct.bid(2)
    num_k = ct.num_tiles(a_ptr, axis=1, shape=(tm, tk))
    k_per_split = (num_k + split_k - 1) // split_k
    k_start = sk_id * k_per_split
    k_end = min(num_k, k_start + k_per_split)
    half_tk: ct.Constant[int] = tk // 2
    acc = ct.zeros((tm, tn), dtype=ct.float32)

    for k in range(k_start, k_end):
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
        acc = ct.mma(tile_a, (w_int - zero) * scale, acc)

    ct.store(part_ptr, index=(sk_id, pid_m, pid_n), tile=ct.reshape(acc, (1, tm, tn)))


@ct.kernel(occupancy=4)
def splitk_reduce_add_fp16_kernel(
    part_ptr,
    res_ptr,
    out_ptr,
    split_k: ct.Constant[int],
    tm: ct.Constant[int],
    tn: ct.Constant[int],
) -> None:
    """Fused Split-K FP32->FP16 reduction with residual add in epilogue."""
    pid_m = ct.bid(0)
    pid_n = ct.bid(1)
    acc = ct.zeros((tm, tn), dtype=ct.float32)
    for sk in range(split_k):
        partial = ct.reshape(
            ct.load(
                part_ptr,
                index=(sk, pid_m, pid_n),
                shape=(1, tm, tn),
                padding_mode=ct.PaddingMode.ZERO,
                latency=3,
            ),
            (tm, tn),
        )
        acc = acc + partial
    res_tile = ct.load(
        res_ptr, index=(pid_m, pid_n), shape=(tm, tn), padding_mode=ct.PaddingMode.ZERO, latency=2
    )
    ct.store(out_ptr, index=(pid_m, pid_n), tile=ct.astype(acc, ct.float16) + res_tile)


def _select_dequant_decode_splitk(m: int, n: int, k: int, group_size: int) -> int:
    """Select Split-K factor for small-batch INT4 dequant decode GEMMs."""
    if m > 16 or k < 2048 or group_size < 128 or k % 128 != 0 or n < 2048 or n % 64 != 0:
        return 1
    if k >= 8192:
        return 8
    return 4


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


@ct.kernel(occupancy=4)
def splitk_reduce_fp16_kernel(
    part_ptr,
    out_ptr,
    split_k: ct.Constant[int],
    tm: ct.Constant[int],
    tn: ct.Constant[int],
) -> None:
    """Fused Split-K FP32->FP16 reduction eliminating extra PyTorch sum+cast launches."""
    pid_m = ct.bid(0)
    pid_n = ct.bid(1)
    acc = ct.zeros((tm, tn), dtype=ct.float32)
    for sk in range(split_k):
        partial = ct.reshape(
            ct.load(
                part_ptr,
                index=(sk, pid_m, pid_n),
                shape=(1, tm, tn),
                padding_mode=ct.PaddingMode.ZERO,
                latency=3,
            ),
            (tm, tn),
        )
        acc = acc + partial
    ct.store(out_ptr, index=(pid_m, pid_n), tile=ct.astype(acc, ct.float16))


@ct.kernel(occupancy=4)
def splitk_reduce_bias_fp16_kernel(
    part_ptr,
    bias_ptr,
    out_ptr,
    split_k: ct.Constant[int],
    tm: ct.Constant[int],
    tn: ct.Constant[int],
) -> None:
    """Fused Split-K FP32->FP16 reduction with 1D bias `(1, N)` epilogue."""
    pid_m = ct.bid(0)
    pid_n = ct.bid(1)
    acc = ct.zeros((tm, tn), dtype=ct.float32)
    for sk in range(split_k):
        partial = ct.reshape(
            ct.load(
                part_ptr,
                index=(sk, pid_m, pid_n),
                shape=(1, tm, tn),
                padding_mode=ct.PaddingMode.ZERO,
                latency=3,
            ),
            (tm, tn),
        )
        acc = acc + partial
    bias_tile = ct.load(
        bias_ptr, index=(0, pid_n), shape=(1, tn), padding_mode=ct.PaddingMode.ZERO, latency=2
    )
    ct.store(out_ptr, index=(pid_m, pid_n), tile=ct.astype(acc, ct.float16) + bias_tile)


@ct.kernel(occupancy=2)
def gemm_fp16_splitk_kernel(
    a_ptr,
    b_ptr,
    part_ptr,
    split_k: ct.Constant[int],
    tm: ct.Constant[int],
    tn: ct.Constant[int],
    tk: ct.Constant[int],
) -> None:
    """Split-K FP16 GEMM saturating SMs on skinny/medium shapes."""
    pid_m = ct.bid(0)
    pid_n = ct.bid(1)
    sk_id = ct.bid(2)
    num_k = ct.num_tiles(a_ptr, axis=1, shape=(tm, tk))
    k_per_split = (num_k + split_k - 1) // split_k
    k_start = sk_id * k_per_split
    k_end = min(num_k, k_start + k_per_split)
    acc = ct.zeros((tm, tn), dtype=ct.float32)
    for k in range(k_start, k_end):
        tile_a = ct.load(
            a_ptr, index=(pid_m, k), shape=(tm, tk), padding_mode=ct.PaddingMode.ZERO, latency=3
        )
        tile_b = ct.load(
            b_ptr, index=(k, pid_n), shape=(tk, tn), padding_mode=ct.PaddingMode.ZERO, latency=3
        )
        acc = ct.mma(tile_a, tile_b, acc)
    ct.store(part_ptr, index=(sk_id, pid_m, pid_n), tile=ct.reshape(acc, (1, tm, tn)))


def select_gemm_kernel_path(
    m: int,
    n: int,
    k: int,
    arch: str = "sm_120",
) -> dict[str, Any]:
    """Select specialized GEMM execution path across skinny, medium, and compute-bound regimes."""
    _ = arch
    if m <= 16 and k >= 512:
        split_k = 4 if k >= 768 else 2
        return {
            "regime": "skinny_splitk",
            "tile_shape": (16, 128, 64) if n >= 2048 else (16, 64, 64),
            "split_k": split_k,
            "red_tile": (16, 128) if n >= 2048 else (16, 64),
            "latency": 3,
        }
    if m <= 256:
        # Saturate 170 SMs on low-M medium matrices via Split-K or 64x64 tiles
        if m <= 64 and k >= 1024 and (ct.cdiv(m, 64) * ct.cdiv(n, 64)) < 128:
            return {
                "regime": "medium_l2",
                "tile_shape": (64, 64, 64),
                "split_k": 4,
                "red_tile": (64, 64),
                "latency": 3,
            }
        use_64x64 = n <= 4096 and (ct.cdiv(m, 64) * ct.cdiv(n, 128)) < 170
        return {
            "regime": "medium_l2",
            "tile_shape": (64, 64, 64) if use_64x64 else (64, 128, 64),
            "split_k": 1,
            "red_tile": (64, 64),
            "latency": 3,
        }
    if m <= 512 and (n <= 4096 or (ct.cdiv(m, 64) * ct.cdiv(n, 64)) >= 256):
        return {
            "regime": "large_compute",
            "tile_shape": (64, 64, 64),
            "split_k": 1,
            "red_tile": (64, 64),
            "latency": 3,
        }
    return {
        "regime": "large_compute",
        "tile_shape": (128, 128, 64),
        "split_k": 1,
        "red_tile": (64, 64),
        "latency": 4,
    }


def cutile_gemm_fp16(
    a: torch.Tensor,
    b: torch.Tensor,
    tile_shape: tuple[int, int, int] | None = None,
    out: torch.Tensor | None = None,
    part_buf: torch.Tensor | None = None,
    bias: torch.Tensor | None = None,
    residual: torch.Tensor | None = None,
) -> torch.Tensor:
    """Run tiled FP16 GEMM: (M, K) @ (K, N) -> (M, N) with optional zero-alloc & fused epilogues."""
    if a.ndim != 2 or b.ndim != 2:
        raise ValueError(f"Expected 2D tensors, got {a.shape} and {b.shape}")
    if bias is not None and residual is not None:
        raise ValueError("Specify at most one of bias or residual in cutile_gemm_fp16")
    m, k = a.shape
    k_b, n = b.shape
    if k != k_b:
        raise ValueError(f"Shape mismatch: {a.shape} vs {b.shape}")

    a_c = a if a.is_contiguous() else a.contiguous()
    b_c = b if b.is_contiguous() else b.contiguous()
    c = out if out is not None else torch.empty((m, n), device=a.device, dtype=torch.float16)
    bias_2d = (
        (
            bias.view(1, n)
            if (bias is not None and bias.is_contiguous())
            else bias.contiguous().view(1, n)
        )
        if bias is not None
        else None
    )
    res_2d = (
        (residual if (residual is not None and residual.is_contiguous()) else residual.contiguous())
        if residual is not None
        else None
    )

    if tile_shape is None:
        spec = select_gemm_kernel_path(m, n, k)
        tm, tn, tk = spec["tile_shape"]
        split_k = min(int(spec["split_k"]), max(1, k // tk))
        if split_k > 1:
            rtm, rtn = spec["red_tile"]
            needed = split_k * m * n
            part = (
                part_buf[:needed].view(split_k, m, n)
                if (part_buf is not None and part_buf.numel() >= needed)
                else torch.empty((split_k, m, n), device=a.device, dtype=torch.float32)
            )
            grid_sk = (ct.cdiv(m, tm), ct.cdiv(n, tn), split_k)
            ct.launch(
                torch.cuda.current_stream(),
                grid_sk,
                gemm_fp16_splitk_kernel,
                (a_c, b_c, part, split_k, tm, tn, tk),
            )
            grid_red = (ct.cdiv(m, rtm), ct.cdiv(n, rtn), 1)
            if bias_2d is not None:
                ct.launch(
                    torch.cuda.current_stream(),
                    grid_red,
                    splitk_reduce_bias_fp16_kernel,
                    (part, bias_2d, c, split_k, rtm, rtn),
                )
            elif res_2d is not None:
                ct.launch(
                    torch.cuda.current_stream(),
                    grid_red,
                    splitk_reduce_add_fp16_kernel,
                    (part, res_2d, c, split_k, rtm, rtn),
                )
            else:
                ct.launch(
                    torch.cuda.current_stream(),
                    grid_red,
                    splitk_reduce_fp16_kernel,
                    (part, c, split_k, rtm, rtn),
                )
            return c
    else:
        tm, tn, tk = tile_shape

    grid = (ct.cdiv(m, tm) * ct.cdiv(n, tn), 1, 1)
    if bias_2d is not None:
        ct.launch(
            torch.cuda.current_stream(),
            grid,
            gemm_fp16_bias_kernel,
            (a_c, b_c, bias_2d, c, tm, tn, tk),
        )
    elif res_2d is not None:
        ct.launch(
            torch.cuda.current_stream(),
            grid,
            gemm_fp16_add_kernel,
            (a_c, b_c, res_2d, c, tm, tn, tk),
        )
    else:
        ct.launch(torch.cuda.current_stream(), grid, gemm_fp16_kernel, (a_c, b_c, c, tm, tn, tk))
    return c


def cutile_gemm_fp8(
    a: torch.Tensor,
    b: torch.Tensor,
    scale_a: float = 1.0,
    scale_b: float = 1.0,
    gpu_config: Any | None = None,
    tile_shape: tuple[int, int, int] | None = None,
) -> torch.Tensor:
    """Run tiled FP8 Tensor Core GEMM guarded by `GPUConfig.fp8` (falls back to FP16 if fp8=False)."""
    if a.ndim != 2 or b.ndim != 2:
        raise ValueError(f"Expected 2D tensors, got {a.shape} and {b.shape}")
    m, k = a.shape
    k_b, n = b.shape
    if k != k_b:
        raise ValueError(f"Shape mismatch: {a.shape} vs {b.shape}")

    if gpu_config is not None and not getattr(gpu_config, "fp8", False):
        return cutile_gemm_fp16(
            a.to(torch.float16) * scale_a,
            b.to(torch.float16) * scale_b,
            tile_shape=tile_shape,
        )

    a_fp8 = a.to(torch.float8_e4m3fn).contiguous()
    b_fp8 = b.to(torch.float8_e4m3fn).contiguous()
    c = torch.empty((m, n), device=a.device, dtype=torch.float16)

    tm, tn, tk = tile_shape or _select_gemm_tiles(m)
    grid = (ct.cdiv(m, tm) * ct.cdiv(n, tn), 1, 1)
    ct.launch(
        torch.cuda.current_stream(),
        grid,
        gemm_fp8_kernel,
        (a_fp8, b_fp8, float(scale_a), float(scale_b), c, tm, tn, tk),
    )
    return c


def fused_gemm_dequant(
    a: torch.Tensor,
    qweight: torch.Tensor,
    scales: torch.Tensor,
    zeros: torch.Tensor,
    group_size: int = 128,
    tile_shape: tuple[int, int, int] | None = None,
    out: torch.Tensor | None = None,
    part_buf: torch.Tensor | None = None,
) -> torch.Tensor:
    """Run fused INT4 dequantization + GEMM in a single CuTile kernel."""
    m, k = a.shape
    if qweight.dtype == torch.int32:
        k_div8, n = qweight.shape
        shifts = torch.arange(0, 32, 4, device=qweight.device, dtype=torch.int32).view(1, 8, 1)
        nibbles = ((qweight.unsqueeze(1) >> shifts) & 0xF).to(torch.uint8).reshape(k_div8 * 8, n)
        qw_u8 = pack_int4_weights(nibbles)
    else:
        qw_u8 = qweight if qweight.is_contiguous() else qweight.contiguous()

    n = qw_u8.shape[1]
    assert qw_u8.shape == (k // 2, n), f"Weight B must be [K//2, N], got {tuple(qw_u8.shape)}"
    assert scales.shape == (
        k // group_size,
        n,
    ), f"Scale S must be [K//{group_size}, N], got {tuple(scales.shape)}"
    if qw_u8.shape[0] * 2 != k:
        raise ValueError(f"Weight K mismatch: {qw_u8.shape[0] * 2} vs {k}")

    tm, tn, tk = tile_shape or _select_gemm_tiles(m, group_size=group_size, n=n)
    if group_size % tk != 0 or k % tk != 0:
        tk = (
            64
            if (group_size % 64 == 0 and k % 64 == 0)
            else (32 if group_size % 32 == 0 else group_size)
        )

    a_c = a if a.is_contiguous() else a.contiguous()
    s_c = scales if scales.is_contiguous() else scales.contiguous()
    z_c = zeros if zeros.is_contiguous() else zeros.contiguous()
    c = out if out is not None else torch.empty((m, n), device=a.device, dtype=torch.float16)

    split_k = 1 if tile_shape is not None else _select_dequant_decode_splitk(m, n, k, group_size)
    if split_k > 1 and tk == 128:
        needed = split_k * m * n
        part = (
            part_buf[:needed].view(split_k, m, n)
            if (part_buf is not None and part_buf.numel() >= needed)
            else torch.empty((split_k, m, n), device=a.device, dtype=torch.float32)
        )
        grid_sk = (ct.cdiv(m, tm), ct.cdiv(n, tn), split_k)
        ct.launch(
            torch.cuda.current_stream(),
            grid_sk,
            gemm_dequant_splitk_kernel,
            (a_c, qw_u8, s_c, z_c, part, split_k, tm, tn, tk, group_size),
        )
        ct.launch(
            torch.cuda.current_stream(),
            (ct.cdiv(m, tm), ct.cdiv(n, tn), 1),
            splitk_reduce_fp16_kernel,
            (part, c, split_k, tm, tn),
        )
        return c

    grid = (ct.cdiv(m, tm) * ct.cdiv(n, tn), 1, 1)
    ct.launch(
        torch.cuda.current_stream(),
        grid,
        gemm_dequant_kernel,
        (a_c, qw_u8, s_c, z_c, c, tm, tn, tk, group_size),
    )
    return c


def fused_gemm_dequant_add(
    a: torch.Tensor,
    qweight: torch.Tensor,
    scales: torch.Tensor,
    zeros: torch.Tensor,
    residual: torch.Tensor,
    group_size: int = 128,
    tile_shape: tuple[int, int, int] | None = None,
    out: torch.Tensor | None = None,
    part_buf: torch.Tensor | None = None,
) -> torch.Tensor:
    """Run fused INT4 dequantization + GEMM + residual add (`a @ W + residual`) in one kernel."""
    m, k = a.shape
    qw_u8 = qweight if qweight.is_contiguous() else qweight.contiguous()
    n = qw_u8.shape[1]
    tm, tn, tk = tile_shape or _select_gemm_tiles(m, group_size=group_size, n=n)
    if group_size % tk != 0 or k % tk != 0:
        tk = (
            64
            if (group_size % 64 == 0 and k % 64 == 0)
            else (32 if group_size % 32 == 0 else group_size)
        )

    a_c = a if a.is_contiguous() else a.contiguous()
    s_c = scales if scales.is_contiguous() else scales.contiguous()
    z_c = zeros if zeros.is_contiguous() else zeros.contiguous()
    r_c = residual if residual.is_contiguous() else residual.contiguous()
    c = out if out is not None else torch.empty((m, n), device=a.device, dtype=torch.float16)

    split_k = 1 if tile_shape is not None else _select_dequant_decode_splitk(m, n, k, group_size)
    if split_k > 1 and tk == 128:
        needed = split_k * m * n
        part = (
            part_buf[:needed].view(split_k, m, n)
            if (part_buf is not None and part_buf.numel() >= needed)
            else torch.empty((split_k, m, n), device=a.device, dtype=torch.float32)
        )
        grid_sk = (ct.cdiv(m, tm), ct.cdiv(n, tn), split_k)
        ct.launch(
            torch.cuda.current_stream(),
            grid_sk,
            gemm_dequant_splitk_kernel,
            (a_c, qw_u8, s_c, z_c, part, split_k, tm, tn, tk, group_size),
        )
        ct.launch(
            torch.cuda.current_stream(),
            (ct.cdiv(m, tm), ct.cdiv(n, tn), 1),
            splitk_reduce_add_fp16_kernel,
            (part, r_c, c, split_k, tm, tn),
        )
        return c

    grid = (ct.cdiv(m, tm) * ct.cdiv(n, tn), 1, 1)
    ct.launch(
        torch.cuda.current_stream(),
        grid,
        gemm_dequant_add_kernel,
        (a_c, qw_u8, s_c, z_c, r_c, c, tm, tn, tk, group_size),
    )
    return c


@ct.kernel(occupancy=2)
def gemm_nvfp4_dequant_kernel(
    a_ptr,
    qw_ptr,
    scales_ptr,
    c_ptr,
    tm: ct.Constant[int],
    tn: ct.Constant[int],
    tk: ct.Constant[int],
    bs: ct.Constant[int],
) -> None:
    """Fused NVFP4 (e2m1) dequant + FP16 Tensor Core GEMM on sm_120."""
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
        lo_n = ct.bitwise_and(qw, 0x0F)
        hi_n = ct.bitwise_and(ct.bitwise_rshift(qw, 4), 0x0F)

        # Decode e2m1 nibbles in registers: [0, 0.5, 1, 1.5, 2, 3, 4, 6] with sign bit
        lo_sign = ct.bitwise_rshift(lo_n, 3)
        lo_mag = ct.bitwise_and(lo_n, 0x07)
        lo_exp = ct.bitwise_rshift(lo_mag, 1)
        lo_man = ct.astype(ct.bitwise_and(lo_mag, 1), ct.float16)
        lo_val = ct.where(
            lo_exp == 0,
            ct.astype(0.5, ct.float16) * lo_man,
            (ct.astype(1.0, ct.float16) + ct.astype(0.5, ct.float16) * lo_man)
            * ct.astype(ct.exp2(ct.astype(lo_exp, ct.float32) - 1.0), ct.float16),
        )
        lo_f = ct.where(lo_sign != 0, -lo_val, lo_val)

        hi_sign = ct.bitwise_rshift(hi_n, 3)
        hi_mag = ct.bitwise_and(hi_n, 0x07)
        hi_exp = ct.bitwise_rshift(hi_mag, 1)
        hi_man = ct.astype(ct.bitwise_and(hi_mag, 1), ct.float16)
        hi_val = ct.where(
            hi_exp == 0,
            ct.astype(0.5, ct.float16) * hi_man,
            (ct.astype(1.0, ct.float16) + ct.astype(0.5, ct.float16) * hi_man)
            * ct.astype(ct.exp2(ct.astype(hi_exp, ct.float32) - 1.0), ct.float16),
        )
        hi_f = ct.where(hi_sign != 0, -hi_val, hi_val)

        lo_e = ct.reshape(lo_f, (half_tk, 1, tn))
        hi_e = ct.reshape(hi_f, (half_tk, 1, tn))
        w_e2m1 = ct.reshape(ct.cat((lo_e, hi_e), axis=1), (tk, tn))

        g_idx = (k * tk) // bs
        scale = ct.load(
            scales_ptr,
            index=(g_idx, pid_n),
            shape=(1, tn),
            padding_mode=ct.PaddingMode.ZERO,
            latency=3,
        )
        w_fp16 = w_e2m1 * scale
        acc = ct.mma(tile_a, w_fp16, acc)

    ct.store(c_ptr, index=(pid_m, pid_n), tile=ct.astype(acc, ct.float16))


def fused_gemm_nvfp4_dequant(
    a: torch.Tensor,
    qweight_fp4: torch.Tensor,
    block_scales: torch.Tensor,
    bias: torch.Tensor | None = None,
    tile_shape: tuple[int, int, int] | None = None,
) -> torch.Tensor:
    """Run fused NVFP4 (e2m1) dequantization + GEMM in a single CuTile kernel."""
    m, k = a.shape
    qw_u8 = qweight_fp4.contiguous()
    n = qw_u8.shape[1]
    if qw_u8.shape != (k // 2, n):
        raise ValueError(f"Weight qweight_fp4 must be [K//2, N], got {tuple(qw_u8.shape)}")
    block_size = k // max(1, block_scales.shape[0])
    if block_scales.shape[1] != n or k % block_size != 0:
        raise ValueError(f"Invalid block_scales shape {tuple(block_scales.shape)} for K={k}, N={n}")

    tm, tn, tk = tile_shape or _select_gemm_tiles(m, group_size=block_size)
    if block_size % tk != 0:
        tk = 64 if block_size % 64 == 0 else (32 if block_size % 32 == 0 else block_size)

    a_c = a.contiguous()
    s_c = block_scales.to(torch.float16).contiguous()
    c = torch.empty((m, n), device=a.device, dtype=torch.float16)

    grid = (ct.cdiv(m, tm) * ct.cdiv(n, tn), 1, 1)
    ct.launch(
        torch.cuda.current_stream(),
        grid,
        gemm_nvfp4_dequant_kernel,
        (a_c, qw_u8, s_c, c, tm, tn, tk, block_size),
    )
    if bias is not None:
        c = c + bias.to(c.dtype).view(1, n)
    return c
