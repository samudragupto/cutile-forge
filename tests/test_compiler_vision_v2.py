"""Tests for Phase 8C/8D: OpRegistry, compiler analysis passes, and CuTileVision v2 CLI."""

from __future__ import annotations

import json
import subprocess
import sys

import pytest
import torch

from cutile_forge.forge.codegen import compile_tfg
from cutile_forge.forge.fusion_graph import (
    DataTensor,
    FusionGraph,
    OpType,
    TileNode,
    build_mistral_layer_tfg,
)
from cutile_forge.forge.op_registry import DEFAULT_OP_REGISTRY
from cutile_forge.forge.pass_manager import (
    ConstantFoldPass,
    LayoutStridePass,
    MemoryLifetimePass,
    ProfitabilityPass,
    default_pass_manager,
)
from cutile_forge.sdk import DType, Graph, compile
from cutile_forge.vision import inspect_compilation


def test_op_registry_traits() -> None:
    assert DEFAULT_OP_REGISTRY.get(OpType.GEMM).bound_regime == "compute"
    assert DEFAULT_OP_REGISTRY.get(OpType.SILU).fusable_epilogue is True
    assert DEFAULT_OP_REGISTRY.get(OpType.CUSTOM).is_pure is False


def test_layout_stride_and_broadcast_validation() -> None:
    g = FusionGraph("bcast")
    a = DataTensor("a", (8, 64), "float16")
    b_ok = DataTensor("b", (1, 64), "float16")
    c = DataTensor("c", (8, 64), "float16")
    g.add_node(TileNode("add", OpType.ADD, "add", inputs=[a, b_ok], outputs=[c]))
    res = LayoutStridePass().run(g)
    assert res.changed is True
    assert a.strides == (64, 1)

    g_bad = FusionGraph("bcast_bad")
    b_bad = DataTensor("b_bad", (8, 32), "float16")
    g_bad.add_node(TileNode("add", OpType.ADD, "add", inputs=[a, b_bad], outputs=[c]))
    with pytest.raises(ValueError, match="Incompatible broadcast"):
        LayoutStridePass().run(g_bad)


def test_constant_fold_double_transpose_and_memory_lifetime() -> None:
    g = Graph("double_tr")
    x = g.input("x", (16, 32), DType.FP16)
    t1 = g.transpose(x, -2, -1)
    t2 = g.transpose(t1, -2, -1)
    y = g.silu(t2)
    g.output(y)

    tfg = g.to_fusion_graph()
    fold = ConstantFoldPass().run(tfg)
    assert fold.changed is True
    assert len(tfg.nodes) == 2  # input + silu only

    compiled = compile(g, target="auto", use_cache=False)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    tx = torch.randn((16, 32), device=dev, dtype=torch.float16)
    got = compiled.run(x=tx)
    assert isinstance(got, torch.Tensor)
    assert (got - torch.nn.functional.silu(tx)).abs().max().item() < 1e-2

    # MemoryLifetimePass & ProfitabilityPass check
    m = build_mistral_layer_tfg(0)
    default_pass_manager().run(m)
    assert all("buffer_slot" in n.attrs for n in m.nodes.values())
    assert all("hbm_bytes_saved" in n.attrs for n in m.nodes.values())
    assert MemoryLifetimePass().run(m).changed is True
    assert ProfitabilityPass().run(m).nodes_after == 3


def test_cu_tile_vision_v2_and_inspect_cli() -> None:
    orig = build_mistral_layer_tfg(0)
    comp = compile_tfg(build_mistral_layer_tfg(0), target="sm_120", run_passes=True)
    summary = inspect_compilation(orig, comp)
    assert summary.total_hbm_bytes > 0
    assert summary.total_hbm_bytes_saved > 0
    assert summary.buffer_slots >= 1
    assert len(summary.timeline_events) == 3

    proc = subprocess.run(
        [sys.executable, "-m", "cutile_forge.cli", "inspect", "--seq-len", "128", "--json"],
        capture_output=True,
        text=True,
        check=True,
    )
    data = json.loads(proc.stdout)
    assert data["nodes_before"] == 8 and data["nodes_after"] == 3
    assert data["buffer_slots"] >= 1
