"""Unit tests for all 6 TFG fusion rules."""

from __future__ import annotations

from cutile_forge.forge.fusion_graph import (
    DataTensor,
    FusionGraph,
    OpType,
    TileNode,
    build_mistral_layer_tfg,
)
from cutile_forge.forge.fusion_rules import apply_fusion_rules


def test_rule_dequant_into_gemm() -> None:
    g = FusionGraph()
    t = DataTensor("w", (64, 64))
    g.add_node(TileNode("dq", OpType.DEQUANT, "Dequant"))
    g.add_node(TileNode("mm", OpType.GEMM, "GEMM"))
    g.add_edge("dq", "mm", t)
    fired = apply_fusion_rules(g)
    assert "fuse_dequant_into_gemm" in fired
    assert len(g.nodes) == 1
    assert next(iter(g.nodes.values())).attrs.get("dequant_fused") is True


def test_rule_qkv_attention() -> None:
    g = FusionGraph()
    t = DataTensor("qkv", (64, 6144))
    g.add_node(TileNode("qkv", OpType.QKV_GEMM, "QKV"))
    g.add_node(TileNode("attn", OpType.ATTENTION_GQA, "Attn"))
    g.add_edge("qkv", "attn", t)
    fired = apply_fusion_rules(g)
    assert "fuse_qkv_attention" in fired
    assert len(g.nodes) == 1
    assert next(iter(g.nodes.values())).op_type == OpType.FUSED_QKV_ATTN


def test_rule_o_layernorm() -> None:
    g = FusionGraph()
    t = DataTensor("o", (64, 4096))
    g.add_node(TileNode("o", OpType.O_GEMM, "O"))
    g.add_node(TileNode("ln", OpType.LAYERNORM, "LN"))
    g.add_edge("o", "ln", t)
    fired = apply_fusion_rules(g)
    assert "fuse_o_layernorm" in fired
    assert len(g.nodes) == 1
    assert next(iter(g.nodes.values())).op_type == OpType.FUSED_O_LAYERNORM


def test_rule_gemm_silu() -> None:
    g = FusionGraph()
    t = DataTensor("h", (64, 14336))
    g.add_node(TileNode("up", OpType.FFN_UP_GEMM, "Up"))
    g.add_node(TileNode("silu", OpType.SILU, "SiLU"))
    g.add_edge("up", "silu", t)
    fired = apply_fusion_rules(g)
    assert "fuse_gemm_silu" in fired
    assert len(g.nodes) == 1
    assert next(iter(g.nodes.values())).op_type == OpType.GEMM_SILU


def test_rule_gemm_silu_down() -> None:
    g = FusionGraph()
    t = DataTensor("h", (64, 14336))
    g.add_node(TileNode("up", OpType.FFN_UP_GEMM, "Up"))
    g.add_node(TileNode("silu", OpType.SILU, "SiLU"))
    g.add_node(TileNode("down", OpType.FFN_DOWN_GEMM, "Down"))
    g.add_edge("up", "silu", t)
    g.add_edge("silu", "down", t)
    fired = apply_fusion_rules(g)
    assert "fuse_gemm_silu" in fired
    assert "fuse_gemm_silu_down" in fired
    assert len(g.nodes) == 1
    assert next(iter(g.nodes.values())).op_type == OpType.FUSED_FFN


def test_rule_ffn_layernorm_residual() -> None:
    g = FusionGraph()
    t = DataTensor("h", (64, 4096))
    g.add_node(TileNode("up", OpType.FFN_UP_GEMM, "Up"))
    g.add_node(TileNode("silu", OpType.SILU, "SiLU"))
    g.add_node(TileNode("down", OpType.FFN_DOWN_GEMM, "Down"))
    g.add_node(TileNode("lnr", OpType.LAYERNORM_RESIDUAL, "LN+Res"))
    g.add_edge("up", "silu", t)
    g.add_edge("silu", "down", t)
    g.add_edge("down", "lnr", t)
    fired = apply_fusion_rules(g)
    assert "fuse_ffn_layernorm_residual" in fired
    assert len(g.nodes) == 1
    assert next(iter(g.nodes.values())).op_type == OpType.FUSED_FFN_NORM


def test_mistral_layer_fuses_8_to_3() -> None:
    g = build_mistral_layer_tfg(0)
    assert len(g.nodes) == 8
    fired = apply_fusion_rules(g)
    assert len(fired) >= 5
    assert len(g.nodes) == 3
    ops = [n.op_type for n in g.topological_order()]
    assert ops == [OpType.FUSED_QKV_ATTN, OpType.FUSED_O_LAYERNORM, OpType.FUSED_FFN_NORM]


def test_non_matching_pattern_does_not_fuse() -> None:
    g = FusionGraph()
    t = DataTensor("t", (16, 16))
    g.add_node(TileNode("ln", OpType.LAYERNORM, "LN"))
    g.add_node(TileNode("attn", OpType.ATTENTION_GQA, "Attn"))
    g.add_edge("ln", "attn", t)
    fired = apply_fusion_rules(g)
    assert fired == []
    assert len(g.nodes) == 2


def test_dequant_into_qkv_gemm_then_attention() -> None:
    g = FusionGraph()
    t = DataTensor("t", (16, 4096))
    g.add_node(TileNode("dq", OpType.DEQUANT, "Dequant"))
    g.add_node(TileNode("qkv", OpType.QKV_GEMM, "QKV"))
    g.add_node(TileNode("attn", OpType.ATTENTION_GQA, "Attn"))
    g.add_edge("dq", "qkv", t)
    g.add_edge("qkv", "attn", t)
    fired = apply_fusion_rules(g)
    assert "fuse_dequant_into_gemm" in fired
    assert "fuse_qkv_attention" in fired
    assert len(g.nodes) == 1


def test_dequant_into_ffn_up_chains_full_ffn() -> None:
    g = FusionGraph()
    t = DataTensor("t", (16, 4096))
    g.add_node(TileNode("dq", OpType.DEQUANT, "Dequant"))
    g.add_node(TileNode("up", OpType.FFN_UP_GEMM, "Up"))
    g.add_node(TileNode("silu", OpType.SILU, "SiLU"))
    g.add_node(TileNode("down", OpType.FFN_DOWN_GEMM, "Down"))
    g.add_edge("dq", "up", t)
    g.add_edge("up", "silu", t)
    g.add_edge("silu", "down", t)
    apply_fusion_rules(g)
    assert len(g.nodes) == 1
    assert next(iter(g.nodes.values())).op_type == OpType.FUSED_FFN


def test_residual_add_fuses_with_layernorm() -> None:
    g = FusionGraph()
    t = DataTensor("t", (16, 4096))
    g.add_node(TileNode("res", OpType.RESIDUAL_ADD, "Res"))
    g.add_node(TileNode("ln", OpType.RMSNORM, "RMSNorm"))
    g.add_edge("res", "ln", t)
    fired = apply_fusion_rules(g)
    assert "fuse_ffn_layernorm_residual" in fired
    assert len(g.nodes) == 1


def test_idempotent_second_pass() -> None:
    g = build_mistral_layer_tfg(0)
    apply_fusion_rules(g)
    fired_again = apply_fusion_rules(g)
    assert fired_again == []
    assert len(g.nodes) == 3
