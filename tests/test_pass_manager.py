"""Unit tests for PassManager, DCE, CSE, shape inference, and target SM resolution."""

from __future__ import annotations

import pytest

from cutile_forge.forge.fusion_graph import (
    DataTensor,
    FusionGraph,
    OpType,
    TileNode,
    build_mistral_layer_tfg,
)
from cutile_forge.forge.pass_manager import (
    CSEPass,
    DCEPass,
    ShapeDtypeInferPass,
    default_pass_manager,
)
from cutile_forge.forge.scheduler import schedule_tfg
from cutile_forge.forge.tile_autotuner import TileAutotuner


def test_shape_infer_validates_and_populates_mnk() -> None:
    g = FusionGraph("mm_ok")
    a = DataTensor("a", (32, 64), "float16")
    b = DataTensor("b", (64, 128), "float16")
    c = DataTensor("c", (32, 128), "float16")
    node = TileNode("mm", OpType.MATMUL, "MM", inputs=[a, b], outputs=[c])
    g.add_node(node)
    res = ShapeDtypeInferPass().run(g)
    assert res.changed is True
    assert (node.attrs["M"], node.attrs["N"], node.attrs["K"]) == (32, 128, 64)

    g_bad = FusionGraph("mm_bad")
    b_bad = DataTensor("b_bad", (32, 128), "float16")
    g_bad.add_node(TileNode("mm", OpType.MATMUL, "MM", inputs=[a, b_bad], outputs=[c]))
    with pytest.raises(ValueError, match="shape mismatch"):
        ShapeDtypeInferPass().run(g_bad)


def test_cse_merges_duplicate_pure_nodes() -> None:
    g = FusionGraph("cse_demo")
    t_in = DataTensor("x", (16, 64), "float16")
    t1 = DataTensor("s1", (16, 64), "float16")
    t2 = DataTensor("s2", (16, 64), "float16")
    t_out = DataTensor("out", (16, 64), "float16")

    g.add_node(TileNode("in_x", OpType.INPUT, "x", outputs=[t_in]))
    g.add_node(TileNode("s1", OpType.SILU, "silu1", inputs=[t_in], outputs=[t1]))
    g.add_node(TileNode("s2", OpType.SILU, "silu2", inputs=[t_in], outputs=[t2]))
    g.add_node(TileNode("add", OpType.ADD, "add", inputs=[t1, t2], outputs=[t_out]))
    g.add_edge("in_x", "s1", t_in)
    g.add_edge("in_x", "s2", t_in)
    g.add_edge("s1", "add", t1)
    g.add_edge("s2", "add", t2)
    g.graph_outputs = ["add"]

    res = CSEPass().run(g)
    assert res.changed is True
    assert res.nodes_before == 4
    assert res.nodes_after == 3
    assert "s2" not in g.nodes


def test_dce_removes_unreachable_nodes_only_with_outputs() -> None:
    g = FusionGraph("dce_demo")
    t = DataTensor("x", (16, 64), "float16")
    t_live = DataTensor("live", (16, 64), "float16")
    t_dead = DataTensor("dead", (16, 64), "float16")
    g.add_node(TileNode("in_x", OpType.INPUT, "x", outputs=[t]))
    g.add_node(TileNode("live_op", OpType.SILU, "live", inputs=[t], outputs=[t_live]))
    g.add_node(TileNode("dead_op", OpType.GELU, "dead", inputs=[t], outputs=[t_dead]))
    g.add_edge("in_x", "live_op", t)
    g.add_edge("in_x", "dead_op", t)

    # Without explicit graph_outputs, DCE is a safe no-op
    noop = DCEPass().run(g)
    assert noop.changed is False
    assert len(g.nodes) == 3

    # With explicit graph_outputs, unreachable node is pruned
    g.graph_outputs = ["live_op"]
    pruned = DCEPass().run(g)
    assert pruned.changed is True
    assert "dead_op" not in g.nodes
    assert len(g.nodes) == 2


def test_default_pass_manager_preserves_mistral_8_to_3_fusion() -> None:
    g = build_mistral_layer_tfg(0)
    report = default_pass_manager().run(g)
    assert report.total_nodes_before == 8
    assert report.total_nodes_after == 3


def test_sm_count_auto_resolution_in_scheduler_and_autotuner() -> None:
    g = build_mistral_layer_tfg(0)
    sched_89 = schedule_tfg(g, target="sm_89")
    sched_120 = schedule_tfg(g, target="sm_120")
    assert sched_120[0].estimated_waves <= sched_89[0].estimated_waves

    tuner_120 = TileAutotuner(target="sm_120")
    tuner_89 = TileAutotuner(target="sm_89")
    assert tuner_120.num_sms == 170
    assert tuner_89.num_sms == 24
    assert tuner_120.smem_limit_bytes > tuner_89.smem_limit_bytes
