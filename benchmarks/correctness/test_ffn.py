"""Correctness tests: CuTile fused FFN variants vs PyTorch."""

from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

from cutile_forge.kernels.ffn_fused import (
    fused_ffn_dequant_up_silu_down,
    fused_ffn_swiglu,
    fused_ffn_up_silu_down,
)
from cutile_forge.kernels.gemm_dequant import pack_int4_weights, unpack_int4_weights_torch
from cutile_forge.kernels.gemm_dequant_silu import fused_gemm_dequant_silu


@pytest.mark.gpu
def test_ffn_fused_fp16() -> None:
    if not torch.cuda.is_available():
        pytest.skip("CUDA GPU required")
    torch.manual_seed(5)
    x = torch.randn((32, 512), device="cuda", dtype=torch.float16) * 0.05
    wu = torch.randn((512, 1024), device="cuda", dtype=torch.float16) * 0.05
    wd = torch.randn((1024, 512), device="cuda", dtype=torch.float16) * 0.05
    out = fused_ffn_up_silu_down(x, wu, wd)
    ref = F.silu(x @ wu) @ wd
    assert (out - ref).abs().max().item() < 1e-2


@pytest.mark.gpu
def test_ffn_swiglu() -> None:
    if not torch.cuda.is_available():
        pytest.skip("CUDA GPU required")
    torch.manual_seed(6)
    x = torch.randn((32, 512), device="cuda", dtype=torch.float16) * 0.05
    wg = torch.randn((512, 1024), device="cuda", dtype=torch.float16) * 0.05
    wu = torch.randn((512, 1024), device="cuda", dtype=torch.float16) * 0.05
    wd = torch.randn((1024, 512), device="cuda", dtype=torch.float16) * 0.05
    out = fused_ffn_swiglu(x, wg, wu, wd)
    ref = (F.silu(x @ wg) * (x @ wu)) @ wd
    assert (out - ref).abs().max().item() < 1e-2


@pytest.mark.gpu
def test_gemm_dequant_silu() -> None:
    if not torch.cuda.is_available():
        pytest.skip("CUDA GPU required")
    torch.manual_seed(7)
    x = torch.randn((64, 512), device="cuda", dtype=torch.float16) * 0.05
    qw = pack_int4_weights(torch.randint(0, 16, (512, 1024), device="cuda", dtype=torch.uint8))
    s = torch.rand((4, 1024), device="cuda", dtype=torch.float16) * 0.02 + 0.005
    z = torch.randint(5, 11, (4, 1024), device="cuda", dtype=torch.float16)
    out = fused_gemm_dequant_silu(x, qw, s, z, 128)
    ref = F.silu(x @ unpack_int4_weights_torch(qw, s, z, 128))
    assert (out - ref).abs().max().item() < 1e-2


@pytest.mark.gpu
@pytest.mark.parametrize("seq_len", [16, 300])
def test_ffn_dequant_fused(seq_len: int) -> None:
    if not torch.cuda.is_available():
        pytest.skip("CUDA GPU required")
    torch.manual_seed(8)
    x = torch.randn((seq_len, 512), device="cuda", dtype=torch.float16) * 0.05
    qwu = pack_int4_weights(torch.randint(0, 16, (512, 1024), device="cuda", dtype=torch.uint8))
    su = torch.rand((4, 1024), device="cuda", dtype=torch.float16) * 0.02 + 0.005
    zu = torch.randint(5, 11, (4, 1024), device="cuda", dtype=torch.float16)
    qwd = pack_int4_weights(torch.randint(0, 16, (1024, 512), device="cuda", dtype=torch.uint8))
    sd = torch.rand((8, 512), device="cuda", dtype=torch.float16) * 0.02 + 0.005
    zd = torch.randint(5, 11, (8, 512), device="cuda", dtype=torch.float16)

    out = fused_ffn_dequant_up_silu_down(x, qwu, su, zu, qwd, sd, zd, 128)
    wu = unpack_int4_weights_torch(qwu, su, zu, 128)
    wd = unpack_int4_weights_torch(qwd, sd, zd, 128)
    ref = F.silu(x @ wu) @ wd
    assert (out - ref).abs().max().item() < 1e-2
