"""Six algebraic and tile-level fusion rules for the TFG compiler."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass

from cutile_forge.forge.fusion_graph import FusionGraph, OpType, TileNode


@dataclass
class FusionRule:
    """Pattern-matching condition and graph-rewriting action."""

    name: str
    description: str
    condition: Callable[[FusionGraph, TileNode], bool]
    action: Callable[[FusionGraph, TileNode], bool]


def _single_successor(graph: FusionGraph, node: TileNode, expected_op: OpType) -> TileNode | None:
    succs = graph.successors(node.node_id)
    if len(succs) != 1:
        return None
    dst = graph.nodes[succs[0]]
    return dst if dst.op_type == expected_op else None


def _cond_dequant_gemm(graph: FusionGraph, node: TileNode) -> bool:
    if node.op_type != OpType.DEQUANT:
        return False
    succs = graph.successors(node.node_id)
    if len(succs) != 1:
        return False
    return graph.nodes[succs[0]].op_type in (
        OpType.GEMM,
        OpType.QKV_GEMM,
        OpType.O_GEMM,
        OpType.FFN_UP_GEMM,
        OpType.FFN_DOWN_GEMM,
    )


def _act_dequant_gemm(graph: FusionGraph, node: TileNode) -> bool:
    dst = graph.nodes[graph.successors(node.node_id)[0]]
    graph.fuse_pair(
        node.node_id,
        dst.node_id,
        new_op=dst.op_type,
        new_name=f"{dst.name} (dequant-fused)",
        attrs={"quantized": True, "dequant_fused": True},
    )
    return True


def _cond_qkv_attn(graph: FusionGraph, node: TileNode) -> bool:
    return (
        node.op_type == OpType.QKV_GEMM
        and _single_successor(graph, node, OpType.ATTENTION_GQA) is not None
    )


def _act_qkv_attn(graph: FusionGraph, node: TileNode) -> bool:
    dst = graph.nodes[graph.successors(node.node_id)[0]]
    graph.fuse_pair(
        node.node_id,
        dst.node_id,
        new_op=OpType.FUSED_QKV_ATTN,
        new_name="Fused QKV+Attention",
    )
    return True


def _cond_o_layernorm(graph: FusionGraph, node: TileNode) -> bool:
    if node.op_type != OpType.O_GEMM:
        return False
    return any(
        graph.nodes[s].op_type in (OpType.LAYERNORM, OpType.RMSNORM)
        for s in graph.successors(node.node_id)
    )


def _act_o_layernorm(graph: FusionGraph, node: TileNode) -> bool:
    for s in graph.successors(node.node_id):
        dst = graph.nodes[s]
        if dst.op_type in (OpType.LAYERNORM, OpType.RMSNORM):
            # Preserve skip edge from O_GEMM to LayerNorm+Residual (if present) by duplicating source on fused node
            graph.fuse_pair(
                node.node_id,
                dst.node_id,
                new_op=OpType.FUSED_O_LAYERNORM,
                new_name="Fused O-GEMM+LayerNorm",
            )
            return True
    return False


def _cond_gemm_silu(graph: FusionGraph, node: TileNode) -> bool:
    return node.op_type in (
        OpType.FFN_UP_GEMM,
        OpType.GEMM,
        OpType.GEMM_DEQUANT,
        OpType.MATMUL,
    ) and (_single_successor(graph, node, OpType.SILU) is not None)


def _act_gemm_silu(graph: FusionGraph, node: TileNode) -> bool:
    dst = graph.nodes[graph.successors(node.node_id)[0]]
    graph.fuse_pair(
        node.node_id,
        dst.node_id,
        new_op=OpType.GEMM_SILU,
        new_name="Fused GEMM+SiLU",
    )
    return True


def _cond_gemm_silu_down(graph: FusionGraph, node: TileNode) -> bool:
    if node.op_type != OpType.GEMM_SILU:
        return False
    return (
        _single_successor(graph, node, OpType.FFN_DOWN_GEMM) is not None
        or _single_successor(graph, node, OpType.MATMUL) is not None
        or _single_successor(graph, node, OpType.GEMM) is not None
    )


def _act_gemm_silu_down(graph: FusionGraph, node: TileNode) -> bool:
    dst = graph.nodes[graph.successors(node.node_id)[0]]
    graph.fuse_pair(
        node.node_id,
        dst.node_id,
        new_op=OpType.FUSED_FFN,
        new_name="Fused FFN (Up+SiLU+Down)",
    )
    return True


def _cond_ffn_ln_res(graph: FusionGraph, node: TileNode) -> bool:
    if node.op_type not in (OpType.FUSED_FFN, OpType.RESIDUAL_ADD):
        return False
    succs = graph.successors(node.node_id)
    if len(succs) != 1:
        return False
    return graph.nodes[succs[0]].op_type in (
        OpType.LAYERNORM_RESIDUAL,
        OpType.LAYERNORM,
        OpType.RMSNORM,
    )


def _act_ffn_ln_res(graph: FusionGraph, node: TileNode) -> bool:
    dst = graph.nodes[graph.successors(node.node_id)[0]]
    graph.fuse_pair(
        node.node_id,
        dst.node_id,
        new_op=OpType.FUSED_FFN_NORM,
        new_name="Fused FFN+LayerNorm+Residual",
        attrs={"fused_tail_op": dst.op_type.value},
    )
    return True


DEFAULT_FUSION_RULES: list[FusionRule] = [
    FusionRule(
        name="fuse_dequant_into_gemm",
        description="Fold INT4 dequantization into GEMM tile register load",
        condition=_cond_dequant_gemm,
        action=_act_dequant_gemm,
    ),
    FusionRule(
        name="fuse_qkv_attention",
        description="Fuse QKV projection with GQA FlashAttention-2",
        condition=_cond_qkv_attn,
        action=_act_qkv_attn,
    ),
    FusionRule(
        name="fuse_o_layernorm",
        description="Fuse output projection GEMM with post-attention LayerNorm",
        condition=_cond_o_layernorm,
        action=_act_o_layernorm,
    ),
    FusionRule(
        name="fuse_gemm_silu",
        description="Fuse FFN-up GEMM with SiLU epilogue in registers",
        condition=_cond_gemm_silu,
        action=_act_gemm_silu,
    ),
    FusionRule(
        name="fuse_gemm_silu_down",
        description="Fuse GEMM+SiLU with FFN-down GEMM into one FFN tile kernel",
        condition=_cond_gemm_silu_down,
        action=_act_gemm_silu_down,
    ),
    FusionRule(
        name="fuse_ffn_layernorm_residual",
        description="Fuse FFN output with trailing LayerNorm+Residual epilogue",
        condition=_cond_ffn_ln_res,
        action=_act_ffn_ln_res,
    ),
]


def apply_fusion_rules(
    graph: FusionGraph,
    rules: Sequence[FusionRule] | None = None,
) -> list[str]:
    """Iteratively apply fusion rules until fixed point; returns fired rule names."""
    active_rules = list(rules) if rules is not None else DEFAULT_FUSION_RULES
    fired: list[str] = []

    changed = True
    while changed:
        changed = False
        for node in list(graph.topological_order()):
            if node.node_id not in graph.nodes:
                continue
            for rule in active_rules:
                if rule.condition(graph, node):
                    if rule.action(graph, node):
                        fired.append(rule.name)
                        changed = True
                        break
            if changed:
                break

    graph.validate()
    return fired
