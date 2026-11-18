"""Unit tests for FusionGraph construction, edge validation, and cycle detection."""

from __future__ import annotations

import pytest

from cutile_forge.forge.fusion_graph import (
    DataTensor,
    FusionGraph,
    OpType,
    TileNode,
    build_mistral_layer_tfg,
)


def test_add_nodes_and_edges() -> None:
    g = FusionGraph("simple")
    t = DataTensor("x", (16, 64))
    g.add_node(TileNode("n1", OpType.GEMM, "gemm1", inputs=[t], outputs=[t]))
    g.add_node(TileNode("n2", OpType.SILU, "silu", inputs=[t], outputs=[t]))
    g.add_edge("n1", "n2", t)
    assert g.validate() is True
    assert [n.node_id for n in g.topological_order()] == ["n1", "n2"]


def test_duplicate_node_raises() -> None:
    g = FusionGraph()
    g.add_node(TileNode("n1", OpType.GEMM, "g1"))
    with pytest.raises(ValueError, match="Duplicate"):
        g.add_node(TileNode("n1", OpType.GEMM, "g2"))


def test_missing_edge_endpoint_raises() -> None:
    g = FusionGraph()
    g.add_node(TileNode("n1", OpType.GEMM, "g1"))
    with pytest.raises(KeyError):
        g.add_edge("n1", "missing", DataTensor("t", (1, 1)))


def test_self_loop_rejected() -> None:
    g = FusionGraph()
    g.add_node(TileNode("n1", OpType.GEMM, "g1"))
    with pytest.raises(ValueError, match="Self-loop"):
        g.add_edge("n1", "n1", DataTensor("t", (1, 1)))


def test_cycle_detection() -> None:
    g = FusionGraph()
    t = DataTensor("t", (16, 16))
    g.add_node(TileNode("a", OpType.GEMM, "A"))
    g.add_node(TileNode("b", OpType.SILU, "B"))
    g.add_edge("a", "b", t)
    g.add_edge("b", "a", t)
    with pytest.raises(ValueError, match="Cycle"):
        g.validate()


def test_remove_node_cleans_edges() -> None:
    g = FusionGraph()
    t = DataTensor("t", (16, 16))
    g.add_node(TileNode("a", OpType.GEMM, "A"))
    g.add_node(TileNode("b", OpType.SILU, "B"))
    g.add_edge("a", "b", t)
    g.remove_node("a")
    assert len(g.nodes) == 1
    assert len(g.edges) == 0


def test_data_tensor_nbytes() -> None:
    assert DataTensor("fp16", (4, 8), dtype="float16").nbytes == 64
    assert DataTensor("fp32", (4, 8), dtype="float32").nbytes == 128
    assert DataTensor("int4", (4, 8), dtype="int4", quantized=True).nbytes == 16


def test_mistral_layer_tfg_has_8_unfused_nodes() -> None:
    g = build_mistral_layer_tfg(0)
    assert len(g.nodes) == 8
    assert g.validate() is True
    assert len(g.topological_order()) == 8
