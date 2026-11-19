"""4-bit group-wise quantization and GPTQ unpacking utilities."""

from __future__ import annotations

import torch

from cutile_forge.kernels.gemm_dequant import pack_int4_weights, unpack_int4_weights_torch


def quantize_4bit_group(
    weight: torch.Tensor,
    group_size: int = 128,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
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
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor | None]:
    """Convert HuggingFace AutoGPTQ int32 `(K//8, N)` / `(K//gs, N//8)` to CuTileForge uint8 format.

    AutoGPTQ stores `qzeros` as `(val - 1)` packed 8x int4 in `int32` along N.
    When `desc_act=True` (`g_idx` is non-sequential), rows are sorted by `g_idx` so
    each contiguous `group_size` block shares one scale/zero, and `perm` is returned
    so callers reorder input columns `x[:, perm]` before matmul.
    """
    k_div8, n_dim = qweight_i32.shape
    device = qweight_i32.device

    shifts = torch.arange(0, 32, 4, device=device, dtype=torch.int32)
    qw_unpacked = ((qweight_i32.unsqueeze(1) >> shifts.view(1, 8, 1)) & 0xF).to(torch.uint8)
    qw_unpacked = qw_unpacked.reshape(k_div8 * 8, n_dim)

    g_dim, n_div8 = qzeros_i32.shape
    qz_unpacked = ((qzeros_i32.unsqueeze(2) >> shifts.view(1, 1, 8)) & 0xF) + 1
    zeros_fp16 = qz_unpacked.reshape(g_dim, n_div8 * 8).to(torch.float16)
    scales_fp16 = scales.to(torch.float16)

    k_dim = k_div8 * 8
    num_groups = k_dim // group_size
    if g_dim == 1 and num_groups > 1:
        # Per-channel GPTQ (group_size == -1): broadcast single scale/zero across K groups
        scales_fp16 = scales_fp16.expand(num_groups, n_dim).contiguous()
        zeros_fp16 = zeros_fp16.expand(num_groups, n_dim).contiguous()
        g_idx = None

    perm: torch.Tensor | None = None
    if g_idx is not None:
        expected = torch.arange(qw_unpacked.shape[0], device=device) // group_size
        if not torch.equal(g_idx.to(device=device, dtype=expected.dtype), expected):
            perm = torch.argsort(g_idx.to(device), stable=True)
            qw_unpacked = qw_unpacked[perm]

    return (
        pack_int4_weights(qw_unpacked),
        scales_fp16.contiguous(),
        zeros_fp16.contiguous(),
        perm,
    )


def unpack_awq_to_cutile(
    qweight_awq: torch.Tensor,
    qzeros_awq: torch.Tensor,
    scales: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Convert AWQ int32 `(K, N // 8)` interleaved `[0, 16, 4, 20, 8, 24, 12, 28]` to CuTile `[K // 2, N]`."""
    k_dim, n_div8 = qweight_awq.shape
    device = qweight_awq.device
    awq_shifts = torch.tensor([0, 16, 4, 20, 8, 24, 12, 28], device=device, dtype=torch.int32)

    qw_u = ((qweight_awq.unsqueeze(-1) >> awq_shifts.view(1, 1, 8)) & 0xF).to(torch.uint8)
    qw_u = qw_u.reshape(k_dim, n_div8 * 8)

    g_dim, _ = qzeros_awq.shape
    qz_u = ((qzeros_awq.unsqueeze(-1) >> awq_shifts.view(1, 1, 8)) & 0xF).to(torch.float16)
    zeros_fp16 = qz_u.reshape(g_dim, n_div8 * 8).contiguous()

    return (
        pack_int4_weights(qw_u),
        scales.to(torch.float16).contiguous(),
        zeros_fp16,
    )


def unpack_gguf_q4_k_to_cutile(
    blocks_u8: torch.Tensor,
    k: int,
    n: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Unpack GGUF Q4_K_M 256-weight super-blocks (144 bytes/block) to CuTile `[K//2, N]` + `[K//128, N]`."""
    if (k * n) % 256 != 0 or k % 128 != 0:
        raise ValueError(f"Q4_K_M requires (K*N)%256==0 and K%128==0, got K={k}, N={n}")
    num_blocks = (k * n) // 256
    raw = blocks_u8.contiguous().view(num_blocks, 144)

    d = raw[:, 0:2].contiguous().view(torch.float16).float().squeeze(-1)  # (B,)
    dmin = raw[:, 2:4].contiguous().view(torch.float16).float().squeeze(-1)  # (B,)
    sc_bytes = raw[:, 4:16].to(torch.int32)  # (B, 12)
    qs = raw[:, 16:144]  # (B, 128)

    # Decode 8 sub-block 6-bit scales & mins
    sc = torch.zeros((num_blocks, 8), device=raw.device, dtype=torch.float32)
    mn = torch.zeros((num_blocks, 8), device=raw.device, dtype=torch.float32)
    sc[:, 0:4] = (sc_bytes[:, 0:4] & 0x3F).float()
    mn[:, 0:4] = (sc_bytes[:, 4:8] & 0x3F).float()
    sc[:, 4:8] = ((sc_bytes[:, 8:12] & 0x0F) | ((sc_bytes[:, 0:4] >> 6) << 4)).float()
    mn[:, 4:8] = (((sc_bytes[:, 8:12] >> 4) & 0x0F) | ((sc_bytes[:, 4:8] >> 6) << 4)).float()

    # Unpack 256 4-bit nibbles per super-block (4 chunks of 64 weights: 32 low + 32 high)
    w_flat = torch.zeros((num_blocks, 256), device=raw.device, dtype=torch.float32)
    for j in range(4):
        q_chunk = qs[:, j * 32 : (j + 1) * 32]
        lo = (q_chunk & 0x0F).float()
        hi = ((q_chunk >> 4) & 0x0F).float()
        s0 = (d * sc[:, 2 * j]).unsqueeze(1)
        m0 = (dmin * mn[:, 2 * j]).unsqueeze(1)
        s1 = (d * sc[:, 2 * j + 1]).unsqueeze(1)
        m1 = (dmin * mn[:, 2 * j + 1]).unsqueeze(1)
        w_flat[:, j * 64 : j * 64 + 32] = s0 * lo - m0
        w_flat[:, j * 64 + 32 : (j + 1) * 64] = s1 * hi - m1

    w_kn = w_flat.reshape(n, k).t().contiguous()
    return quantize_4bit_group(w_kn, group_size=128)
