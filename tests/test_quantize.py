"""Unit tests for 4-bit quantization and GPTQ unpacking."""

from __future__ import annotations

import pytest
import torch

from cutile_forge.utils.quantize import dequantize_4bit_group, quantize_4bit_group, unpack_gptq_layer


def test_4bit_roundtrip_accuracy() -> None:
    w = torch.randn((256, 128), dtype=torch.float16) * 0.1
    qw, scales, zeros = quantize_4bit_group(w, group_size=128)
    assert qw.shape == (128, 128)
    assert qw.dtype == torch.uint8
    w_rec = dequantize_4bit_group(qw, scales, zeros, group_size=128)
    max_err = (w - w_rec).abs().max().item()
    assert max_err < 0.06


def test_4bit_exact_integer_grid() -> None:
    # Values exactly on integer grid [0..15] scaled by 0.25 with zero=0
    grid = torch.arange(0, 16, dtype=torch.float16).unsqueeze(1).repeat(16, 64) * 0.25
    qw, scales, zeros = quantize_4bit_group(grid, group_size=128)
    rec = dequantize_4bit_group(qw, scales, zeros, group_size=128)
    assert (grid - rec).abs().max().item() < 1e-2


def test_unpack_gptq_layer() -> None:
    k, n, gs = 128, 32, 128
    qw_i32 = torch.randint(-2**31, 2**31 - 1, (k // 8, n), dtype=torch.int32)
    qz_i32 = torch.zeros((k // gs, n // 8), dtype=torch.int32)
    scales = torch.ones((k // gs, n), dtype=torch.float16) * 0.1
    qw_u8, s_out, z_out, perm = unpack_gptq_layer(qw_i32, qz_i32, scales, group_size=gs)
    assert qw_u8.shape == (k // 2, n)
    assert qw_u8.dtype == torch.uint8
    assert s_out.shape == (1, n)
    assert torch.all(z_out == 1.0)
    assert perm is None


def test_quantize_invalid_k_raises() -> None:
    with pytest.raises(ValueError, match="divisible"):
        quantize_4bit_group(torch.randn((100, 64), dtype=torch.float16), group_size=128)
