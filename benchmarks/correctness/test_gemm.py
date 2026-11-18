"""Correctness tests: CuTile GEMM (FP16 + INT4 dequant) vs PyTorch across 8 shapes."""

from __future__ import annotations

import pytest
import torch

from cutile_forge.kernels.gemm_dequant import (
    cutile_gemm_fp16,
    fused_gemm_dequant,
    pack_int4_weights,
    unpack_int4_weights_torch,
)

SHAPES = [
    (1, 12288, 4096),
    (256, 12288, 4096),
    (1, 14336, 4096),
    (256, 14336, 4096),
    (1, 4096, 14336),
    (256, 4096, 14336),
    (1, 4096, 4096),
    (512, 4096, 4096),
]


@pytest.mark.gpu
@pytest.mark.parametrize(("m", "n", "k"), SHAPES)
def test_gemm_shape(m: int, n: int, k: int) -> None:
    if not torch.cuda.is_available():
        pytest.skip("CUDA GPU required")
    torch.manual_seed(0)
    a = torch.randn((m, k), device="cuda", dtype=torch.float16) * 0.05
    b = torch.randn((k, n), device="cuda", dtype=torch.float16) * 0.05

    c = cutile_gemm_fp16(a, b)
    ref = torch.matmul(a, b)
    assert (c - ref).abs().max().item() < 1e-2

    w_int = torch.randint(0, 16, (k, n), device="cuda", dtype=torch.uint8)
    qw = pack_int4_weights(w_int)
    scales = torch.rand((k // 128, n), device="cuda", dtype=torch.float16) * 0.02 + 0.005
    zeros = torch.randint(5, 11, (k // 128, n), device="cuda", dtype=torch.float16)

    c_dq = fused_gemm_dequant(a, qw, scales, zeros, group_size=128)
    ref_dq = torch.matmul(a, unpack_int4_weights_torch(qw, scales, zeros, 128))
    assert (c_dq - ref_dq).abs().max().item() < 1e-2

