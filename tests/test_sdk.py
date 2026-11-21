"""Unit and GPU correctness tests for the standalone `cutile_forge.sdk` API."""

from __future__ import annotations

import subprocess
import sys

import torch
import torch.nn.functional as F

from cutile_forge.forge.artifact_cache import GLOBAL_ARTIFACT_CACHE
from cutile_forge.forge.fusion_graph import OpType
from cutile_forge.sdk import DType, Graph, KernelRegistry, compile


def test_sdk_does_not_import_llm_engine_or_server() -> None:
    code = (
        "import sys; import cutile_forge.sdk; "
        "assert 'cutile_forge.engine' not in sys.modules; "
        "assert 'cutile_forge.server' not in sys.modules; "
        "print('ISOLATED_OK')"
    )
    proc = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        check=True,
    )
    assert "ISOLATED_OK" in proc.stdout


def test_sdk_fuses_and_runs_matmul_silu() -> None:
    GLOBAL_ARTIFACT_CACHE.clear()
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    g = Graph("matmul_silu")
    x = g.input("x", (32, 64), DType.FP16)
    w = g.input("w", (64, 128), DType.FP16)
    out = g.silu(g.matmul(x, w))
    g.output(out)

    compiled = compile(g, target="auto", use_cache=True)
    assert compiled.cache_hit is False
    op_types = [k.op_type for k in compiled.kernels if k.op_type != OpType.INPUT]
    assert op_types == [OpType.GEMM_SILU]

    compiled2 = compile(g, target="auto", use_cache=True)
    assert compiled2.cache_hit is True

    tx = torch.randn((32, 64), device=dev, dtype=torch.float16) * 0.1
    tw = torch.randn((64, 128), device=dev, dtype=torch.float16) * 0.1
    got = compiled.run(x=tx, w=tw)
    assert isinstance(got, torch.Tensor)
    ref = F.silu(torch.matmul(tx, tw))
    assert (got - ref).abs().max().item() < 1e-2


def test_sdk_fuses_and_runs_full_ffn_and_rmsnorm() -> None:
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    g = Graph("ffn_chain")
    x = g.input("x", (32, 64), DType.FP16)
    wu = g.input("wu", (64, 128), DType.FP16)
    wd = g.input("wd", (128, 64), DType.FP16)
    nw = g.input("nw", (64,), DType.FP16)

    y = g.matmul(g.silu(g.matmul(x, wu)), wd)
    out = g.rmsnorm(y, nw, eps=1e-5)
    g.output(out)

    compiled = compile(g, target="auto")
    non_input_ops = [k.op_type for k in compiled.kernels if k.op_type != OpType.INPUT]
    assert non_input_ops == [OpType.FUSED_FFN_NORM]

    tx = torch.randn((32, 64), device=dev, dtype=torch.float16) * 0.1
    twu = torch.randn((64, 128), device=dev, dtype=torch.float16) * 0.1
    twd = torch.randn((128, 64), device=dev, dtype=torch.float16) * 0.1
    tnw = torch.ones((64,), device=dev, dtype=torch.float16)

    got = compiled.run(x=tx, wu=twu, wd=twd, nw=tnw)
    assert isinstance(got, torch.Tensor)
    ref_y = torch.matmul(F.silu(torch.matmul(tx, twu)), twd).float()
    ref = (ref_y * torch.rsqrt(ref_y.pow(2).mean(dim=-1, keepdim=True) + 1e-5) * tnw.float()).to(
        torch.float16
    )
    assert (got - ref).abs().max().item() < 1e-2


def test_sdk_softmax_and_custom_kernel_registry() -> None:
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    reg = KernelRegistry()

    @reg.register("scale_twice")
    def scale_twice(t: torch.Tensor, factor: float = 2.0) -> torch.Tensor:
        return t * factor

    g = Graph("custom_softmax")
    x = g.input("x", (8, 64), DType.FP16)
    s = g.softmax(x)
    scaled = g.custom("scale_twice", s, out_shape=(8, 64), factor=2.0)
    g.output(scaled)

    compiled = compile(g, target="auto", registry=reg, use_cache=False)
    tx = torch.randn((8, 64), device=dev, dtype=torch.float16)
    got = compiled.run(x=tx)
    assert isinstance(got, torch.Tensor)
    ref = torch.softmax(tx.float(), dim=-1).to(torch.float16) * 2.0
    assert (got - ref).abs().max().item() < 1e-2
