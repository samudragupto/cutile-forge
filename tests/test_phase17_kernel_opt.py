"""Regression and parity tests for Phase 17 kernel optimizations (`v3.4.0`)."""

from __future__ import annotations

import torch
import torch.nn.functional as F

from cutile_forge.kernels import (
    cutile_avg_pool2d,
    cutile_gelu,
    cutile_gemm_fp16,
    cutile_max_pool2d,
    cutile_reduce_max,
    cutile_reduce_sum,
    cutile_silu,
    cutile_transpose,
)
from cutile_forge.kernels.gemm_dequant import select_gemm_kernel_path


def test_phase17_gemm_dispatch_and_splitk_reduce_parity() -> None:
    """Verify Split-K dispatch + splitk_reduce_fp16_kernel on aligned & unaligned shapes."""
    torch.manual_seed(7)

    s_skinny = select_gemm_kernel_path(4, 4096, 4096)
    s_med64 = select_gemm_kernel_path(64, 4096, 4096)
    s_med256_dn = select_gemm_kernel_path(256, 4096, 14336)
    s_med256_gt = select_gemm_kernel_path(256, 14336, 4096)
    s_large512 = select_gemm_kernel_path(512, 4096, 4096)

    assert s_skinny["regime"] == "skinny_splitk" and s_skinny["split_k"] > 1
    assert s_med64["regime"] == "medium_l2" and s_med64["split_k"] == 4
    assert s_med256_dn["regime"] == "medium_l2" and s_med256_dn["tile_shape"] == (64, 64, 64)
    assert s_med256_gt["regime"] == "medium_l2" and s_med256_gt["tile_shape"] == (64, 128, 64)
    assert s_large512["regime"] == "large_compute" and s_large512["tile_shape"] == (64, 64, 64)

    if not torch.cuda.is_available():
        return

    for m, n, k in [
        (4, 2048, 2048),
        (7, 333, 1055),
        (64, 2048, 2048),
        (37, 129, 1025),
        (256, 1024, 1024),
        (512, 1024, 512),
    ]:
        a = torch.randn((m, k), device="cuda", dtype=torch.float16) * 0.05
        b = torch.randn((k, n), device="cuda", dtype=torch.float16) * 0.05
        ref = torch.matmul(a, b)
        got = cutile_gemm_fp16(a, b)
        assert got.shape == (m, n)
        assert (got - ref).abs().max().item() < 5e-2, f"GEMM mismatch at {(m, n, k)}"


def test_phase17_pool2d_aligned_and_boundary_parity() -> None:
    """Verify high-occupancy 2x2 Pool2D kernels match PyTorch across sizes and dtypes."""
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(11)
    dtypes = (torch.float16, torch.bfloat16) if dev == "cuda" else (torch.float32,)
    shapes = (
        [(4, 16, 56, 56), (2, 8, 19, 25), (3, 12, 65, 71)] if dev == "cuda" else [(2, 4, 16, 16)]
    )
    for dtype in dtypes:
        for shape in shapes:
            x = torch.randn(shape, device=dev, dtype=dtype)
            got_max = cutile_max_pool2d(x, 2, 2)
            ref_max = F.max_pool2d(x, 2, 2)
            assert (got_max.float() - ref_max.float()).abs().max().item() < 1e-4

            got_avg = cutile_avg_pool2d(x, 2, 2)
            ref_avg = F.avg_pool2d(x, 2, 2)
            assert (got_avg.float() - ref_avg.float()).abs().max().item() < 2e-2


def test_phase17_fast_unmasked_and_masked_reductions() -> None:
    """Verify unmasked fast reductions, non-multiple fallback, and keepdim last-dim views."""
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(19)
    dtypes = (torch.float16, torch.float32) if dev == "cuda" else (torch.float32,)
    shapes = (
        [(256, 2048), (1024, 1024), (300, 1024), (137, 1003), (8, 128, 1024)]
        if dev == "cuda"
        else [(16, 64), (7, 19)]
    )
    for dtype in dtypes:
        for shape in shapes:
            x = torch.randn(shape, device=dev, dtype=dtype) * 0.05
            for keepdim in (False, True):
                got_sum = cutile_reduce_sum(x, dim=-1, keepdim=keepdim)
                ref_sum = x.float().sum(dim=-1, keepdim=keepdim).to(dtype)
                tol = 5e-2 if dtype == torch.float16 else 1e-4
                assert (got_sum - ref_sum).abs().max().item() < tol

                got_max = cutile_reduce_max(x, dim=-1, keepdim=keepdim)
                ref_max = x.max(dim=-1, keepdim=keepdim).values
                assert (got_max - ref_max).abs().max().item() < 1e-4

        # Non-last dim reduction still exact
        x3 = torch.randn((12, 512, 64), device=dev, dtype=dtype) * 0.05
        assert (
            cutile_reduce_sum(x3, dim=1, keepdim=True)
            - x3.float().sum(dim=1, keepdim=True).to(dtype)
        ).abs().max().item() < 5e-2
        assert (
            cutile_reduce_max(x3, dim=1, keepdim=False) - x3.max(dim=1, keepdim=False).values
        ).abs().max().item() < 1e-4


def test_phase17_activations_and_transpose_parity() -> None:
    """Verify updated SiLU/GELU occupancy & tile dispatch and transpose output parity."""
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(29)
    dtype = torch.float16 if dev == "cuda" else torch.float32
    shapes = [(1024, 4096), (137, 1003)] if dev == "cuda" else [(16, 64)]
    for shape in shapes:
        x = torch.randn(shape, device=dev, dtype=dtype)
        assert (cutile_silu(x) - F.silu(x)).abs().max().item() < 1e-2
        assert (cutile_gelu(x) - F.gelu(x, approximate="tanh")).abs().max().item() < 1e-2

    tshapes = [(512, 512), (3, 33, 65)] if dev == "cuda" else [(16, 32), (2, 8, 12)]
    for tshape in tshapes:
        tx = torch.randn(tshape, device=dev, dtype=dtype)
        got_t = cutile_transpose(tx, -2, -1)
        ref_t = tx.transpose(-2, -1).contiguous()
        assert torch.equal(got_t, ref_t)
