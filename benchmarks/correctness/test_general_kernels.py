"""GPU numerical correctness stress tests for Sprint 2 general CuTile kernels."""

from __future__ import annotations

import pytest
import torch

from cutile_forge.kernels import (
    cutile_bmm,
    cutile_cumsum,
    cutile_reduce_max,
    cutile_reduce_sum,
    cutile_transpose,
)

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


@pytest.mark.gpu
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
@pytest.mark.parametrize(
    ("b", "m", "n", "k"),
    [(1, 1, 17, 17), (3, 17, 63, 33), (4, 64, 129, 64), (2, 130, 128, 96)],
)
def test_cutile_bmm_accuracy(dtype: torch.dtype, b: int, m: int, n: int, k: int) -> None:
    torch.manual_seed(42)
    a = torch.randn((b, m, k), device="cuda", dtype=dtype) * 0.1
    w = torch.randn((b, k, n), device="cuda", dtype=dtype) * 0.1
    got = cutile_bmm(a, w)
    ref = torch.bmm(a.float(), w.float()).to(dtype)
    tol = 5e-4 if dtype == torch.float32 else 2e-2
    assert (got - ref).abs().max().item() < tol


@pytest.mark.gpu
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
@pytest.mark.parametrize("n_cols", [1, 17, 63, 256, 513, 1024, 1500])
@pytest.mark.parametrize("dim", [-1, 1])
def test_cutile_reductions_and_cumsum(dtype: torch.dtype, n_cols: int, dim: int) -> None:
    torch.manual_seed(7)
    shape = (4, n_cols, 5) if dim == 1 else (4, 5, n_cols)
    # Use negative offset so reduce_max also tests all-negative rows
    x = torch.randn(shape, device="cuda", dtype=dtype) * 0.05 - 0.5

    got_sum = cutile_reduce_sum(x, dim=dim, keepdim=False)
    ref_sum = x.float().sum(dim=dim).to(dtype)
    if dtype == torch.float32:
        torch.testing.assert_close(got_sum, ref_sum, rtol=1e-4, atol=1e-4)
    else:
        torch.testing.assert_close(got_sum, ref_sum, rtol=1e-2, atol=0.15)

    got_max = cutile_reduce_max(x, dim=dim, keepdim=True)
    ref_max = x.max(dim=dim, keepdim=True).values
    assert (got_max - ref_max).abs().max().item() < 1e-5

    got_cs = cutile_cumsum(x, dim=dim)
    ref_cs = torch.cumsum(x.float(), dim=dim).to(dtype)
    if dtype == torch.float32:
        torch.testing.assert_close(got_cs, ref_cs, rtol=1e-4, atol=1e-4)
    else:
        torch.testing.assert_close(got_cs, ref_cs, rtol=1e-2, atol=0.15)


@pytest.mark.gpu
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
def test_cutile_transpose_and_non_contiguous_inputs(dtype: torch.dtype) -> None:
    torch.manual_seed(11)
    base = torch.randn((2, 70, 90), device="cuda", dtype=dtype)
    sliced = base[:, 3:68:2, 5:80:2]  # Non-contiguous strided slice
    assert not sliced.is_contiguous()

    got_tr = cutile_transpose(sliced, -2, -1)
    ref_tr = sliced.transpose(-2, -1).contiguous()
    assert torch.equal(got_tr, ref_tr)

    got_sum = cutile_reduce_sum(sliced, dim=-1)
    ref_sum = sliced.float().sum(dim=-1).to(dtype)
    tol = 1e-4 if dtype == torch.float32 else 2e-2
    assert (got_sum - ref_sum).abs().max().item() < tol
