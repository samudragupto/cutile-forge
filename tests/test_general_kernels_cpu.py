"""CPU-safe and GPU-compatible tests for general CuTile kernels and SDK graph lowering."""

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
from cutile_forge.sdk import DType, Graph, compile


def test_general_kernel_input_validation() -> None:
    a = torch.randn((2, 4, 8), dtype=torch.float16)
    b_bad = torch.randn((3, 8, 4), dtype=torch.float16)
    with pytest.raises(ValueError, match="Shape mismatch"):
        cutile_bmm(a, b_bad)

    with pytest.raises(ValueError, match="Dimension out of range"):
        cutile_reduce_sum(a, dim=5)

    with pytest.raises(ValueError, match="Dimension out of range"):
        cutile_cumsum(a, dim=-4)

    with pytest.raises(ValueError, match="ndim >= 2"):
        cutile_transpose(torch.randn(8, dtype=torch.float16))


def test_sdk_executes_bmm_reductions_cumsum_and_transpose() -> None:
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    g = Graph("general_ops_chain")
    a = g.input("a", (4, 16, 32), DType.FP16)
    b = g.input("b", (4, 32, 24), DType.FP16)
    p = g.bmm(a, b)
    pt = g.transpose(p, dim0=-2, dim1=-1)
    cs = g.cumsum(pt, dim=-1)
    rs = g.reduce_sum(cs, dim=-1, keepdim=False)
    rm = g.reduce_max(pt, dim=-1, keepdim=True)
    g.output(rs, rm)

    compiled = compile(g, target="auto", use_cache=False)

    ta = torch.randn((4, 16, 32), device=dev, dtype=torch.float16) * 0.1
    tb = torch.randn((4, 32, 24), device=dev, dtype=torch.float16) * 0.1
    outs = compiled.run(a=ta, b=tb)
    assert isinstance(outs, dict)

    ref_p = torch.bmm(ta.float(), tb.float()).to(torch.float16)
    ref_pt = ref_p.transpose(-2, -1).contiguous()
    ref_cs = torch.cumsum(ref_pt.float(), dim=-1).to(torch.float16)
    ref_rs = ref_cs.float().sum(dim=-1).to(torch.float16)
    ref_rm = ref_pt.max(dim=-1, keepdim=True).values

    out_rs = outs[rs.name]
    out_rm = outs[rm.name]
    assert (out_rs - ref_rs).abs().max().item() < 5e-2
    assert (out_rm - ref_rm).abs().max().item() < 1e-2


def test_direct_cutile_general_kernels_match_torch() -> None:
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    for dt in (torch.float16, torch.bfloat16, torch.float32):
        tol = 1e-4 if dt == torch.float32 else 2e-2
        a = torch.randn((3, 17, 33), device=dev, dtype=dt) * 0.1
        b = torch.randn((3, 33, 19), device=dev, dtype=dt) * 0.1

        out_bmm = cutile_bmm(a, b)
        ref_bmm = torch.bmm(a.float(), b.float()).to(dt)
        assert (out_bmm - ref_bmm).abs().max().item() < tol

        out_sum = cutile_reduce_sum(a, dim=-1, keepdim=True)
        assert (out_sum - a.float().sum(dim=-1, keepdim=True).to(dt)).abs().max().item() < tol

        out_max = cutile_reduce_max(a, dim=-1, keepdim=False)
        assert (out_max - a.max(dim=-1).values).abs().max().item() < tol

        out_cs = cutile_cumsum(a, dim=-1)
        assert (out_cs - torch.cumsum(a.float(), dim=-1).to(dt)).abs().max().item() < tol

        out_tr = cutile_transpose(a, -2, -1)
        assert (out_tr - a.transpose(-2, -1).contiguous()).abs().max().item() < tol
