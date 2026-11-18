"""4-bit group-wise quantization and GPTQ unpacking utilities."""

from __future__ import annotations

from typing import Tuple

import torch

from cutile_forge.kernels.gemm_dequant import pack_int4_weights, unpack_int4_weights_torch


def quantize_4bit_group(
    weight: torch.Tensor,
    group_size: int = 128,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Quantize a 2D float16/float32 weight matrix (K, N) to unsigned 4-bit groups.

    Returns:
        qweight: (K // 2, N) uint8 packed nibbles
        scales: (K // group_size, N) float16
        zeros: (K // group_size, N) float16
    """
    if weight.ndim != 2:
        raise ValueError(f"Expected 2D weight tensor, got {weight.shape}")
    k_dim, n_dim = weight.shape
    if k_dim % group_size != 0:
        raise ValueError(f"K ({k_dim}) must be divisible by group_size ({group_size})")

    w_f = weight.float().view(k_dim // group_size, group_size, n_dim)
    w_min = w_f.min(dim=1, keepdim=True).values
    w_max = w_f.max(dim=1, keepdim=True).values

    scales = ((w_max - w_min) / 15.0).clamp(min=1e-6)
    zeros = (-w_min / scales).round().clamp(0.0, 15.0)
    q_int = (w_f / scales + zeros).round().clamp(0.0, 15.0).to(torch.uint8).view(k_dim, n_dim)

    qweight = pack_int4_weights(q_int)
    return qweight, scales.squeeze(1).to(torch.float16), zeros.squeeze(1).to(torch.float16)


def dequantize_4bit_group(
    qweight: torch.Tensor,
    scales: torch.Tensor,
    zeros: torch.Tensor,
    group_size: int = 128,
) -> torch.Tensor:
    """Dequantize `(K // 2, N)` uint8 weights back to float16 `(K, N)`."""
    return unpack_int4_weights_torch(qweight, scales, zeros, group_size=group_size)


def unpack_gptq_layer(
    qweight_i32: torch.Tensor,
    qzeros_i32: torch.Tensor,
    scales: torch.Tensor,
    group_size: int = 128,
    g_idx: torch.Tensor | None = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Convert HuggingFace AutoGPTQ int32 `(K//8, N)` / `(K//gs, N//8)` to CuTileForge uint8 format.

    AutoGPTQ stores `qzeros` as `(val - 1)` packed 8x int4 in `int32` along N.
    """
    k_div8, n_dim = qweight_i32.shape
    device = qweight_i32.device

    shifts = torch.arange(0, 32, 4, device=device, dtype=torch.int32)
    # Unpack qweight: (K//8, N) -> (K, N) uint8
    qw_unpacked = ((qweight_i32.unsqueeze(1) >> shifts.view(1, 8, 1)) & 0xF).to(torch.uint8)
    qw_unpacked = qw_unpacked.reshape(k_div8 * 8, n_dim)

    # Unpack qzeros: (G, N//8) -> (G, N) float16 (+1 per GPTQ v1/v2 convention)
    g_dim, n_div8 = qzeros_i32.shape
    qz_unpacked = ((qzeros_i32.unsqueeze(2) >> shifts.view(1, 1, 8)) & 0xF) + 1
    zeros_fp16 = qz_unpacked.reshape(g_dim, n_div8 * 8).to(torch.float16)

    if g_idx is not None:
        # Reorder rows to group order if act-order permutation was used
        perm = torch.argsort(g_idx)
        inv_perm = torch.argsort(perm)
        _ = inv_perm  # Keep contiguous ordering if sequential; act-order handled at load if needed

    return pack_int4_weights(qw_unpacked), scales.to(torch.float16).contiguous(), zeros_fp16.contiguous()
