"""Tile Fusion Graph (TFG) intermediate representation for CuTileForge."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any


class OpType(StrEnum):
    """Supported primitive and fused tile operations in a TFG."""

    INPUT = "input"
    MATMUL = "matmul"
    BATCHED_MATMUL = "batched_matmul"
    ADD = "add"
    GELU = "gelu"
    SOFTMAX = "softmax"
    REDUCE_SUM = "reduce_sum"
    REDUCE_MAX = "reduce_max"
    CUMSUM = "cumsum"
    TRANSPOSE = "transpose"
    CUSTOM = "custom"
    DEQUANT = "dequant"
    GEMM = "gemm"
    GEMM_DEQUANT = "gemm_dequant"
    QKV_GEMM = "qkv_gemm"
    ATTENTION_GQA = "attention_gqa"
    O_GEMM = "o_gemm"
    LAYERNORM = "layernorm"
    RMSNORM = "rmsnorm"
    RESIDUAL_ADD = "residual_add"
    LAYERNORM_RESIDUAL = "layernorm_residual"
    FFN_UP_GEMM = "ffn_up_gemm"
    SILU = "silu"
    FFN_DOWN_GEMM = "ffn_down_gemm"
    # Fused composite ops produced by TFG passes
    GEMM_SILU = "gemm_silu"
    FUSED_QKV_ATTN = "fused_qkv_attn"
    FUSED_O_LAYERNORM = "fused_o_layernorm"
    FUSED_FFN = "fused_ffn"
    FUSED_FFN_NORM = "fused_ffn_norm"


@dataclass
class DataTensor:
    """Tensor descriptor flowing along TFG edges."""

    name: str
    shape: tuple[int, ...]
    dtype: str = "float16"
    residency: str = "hbm"  # "hbm", "ddr5", "sram", "register"
    quantized: bool = False
    group_size: int = 128
    layout: str = "row_major"
    strides: tuple[int, ...] | None = None
    alignment: int = 16

    @property
    def nbytes(self) -> int:
        elems = 1
        for dim in self.shape:
            elems *= dim
        if self.quantized or self.dtype == "int4":
            return (elems + 1) // 2
        if self.dtype in ("float16", "bfloat16", "int16"):
            return elems * 2
        if self.dtype in ("float32", "int32"):
            return elems * 4
        return elems


@dataclass
class TileNode:
    """A single compute or fused tile kernel node in a FusionGraph."""

    node_id: str
    op_type: OpType
    name: str
    inputs: list[DataTensor] = field(default_factory=list)
    outputs: list[DataTensor] = field(default_factory=list)
    tile_shape: tuple[int, int, int] = (128, 128, 64)
    pipeline_stages: int = 1
    prefetch_stage: int = -1
    occupancy: int = 2
    attrs: dict[str, Any] = field(default_factory=dict)
    fused_ops: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not self.fused_ops:
            self.fused_ops = [self.op_type.value]


@dataclass
class Edge:
    """Directed data dependency edge between two TileNodes."""

    src_id: str
    dst_id: str
    tensor: DataTensor


class FusionGraph:
    """Directed acyclic graph of TileNodes representing a transformer layer or model."""

    def __init__(self, name: str = "tfg_layer") -> None:
        self.name = name
        self.nodes: dict[str, TileNode] = {}
        self.edges: list[Edge] = []
        self.graph_outputs: list[str] = []

    def add_node(self, node: TileNode) -> TileNode:
        if node.node_id in self.nodes:
            raise ValueError(f"Duplicate node_id: {node.node_id}")
        self.nodes[node.node_id] = node
        return node

    def add_edge(self, src_id: str, dst_id: str, tensor: DataTensor) -> Edge:
        if src_id not in self.nodes:
            raise KeyError(f"Unknown source node: {src_id}")
        if dst_id not in self.nodes:
            raise KeyError(f"Unknown destination node: {dst_id}")
        if src_id == dst_id:
            raise ValueError(f"Self-loop not allowed on node {src_id}")
        edge = Edge(src_id=src_id, dst_id=dst_id, tensor=tensor)
        self.edges.append(edge)
        return edge

    def predecessors(self, node_id: str) -> list[str]:
        return [e.src_id for e in self.edges if e.dst_id == node_id]

    def successors(self, node_id: str) -> list[str]:
        return [e.dst_id for e in self.edges if e.src_id == node_id]

    def outgoing_edges(self, node_id: str) -> list[Edge]:
        return [e for e in self.edges if e.src_id == node_id]

    def incoming_edges(self, node_id: str) -> list[Edge]:
        return [e for e in self.edges if e.dst_id == node_id]

    def remove_node(self, node_id: str) -> None:
        if node_id not in self.nodes:
            raise KeyError(f"Unknown node: {node_id}")
        del self.nodes[node_id]
        self.edges = [e for e in self.edges if e.src_id != node_id and e.dst_id != node_id]

    def fuse_pair(
        self,
        src_id: str,
        dst_id: str,
        new_op: OpType,
        new_name: str | None = None,
        attrs: dict[str, Any] | None = None,
    ) -> TileNode:
        """Merge `src_id -> dst_id` into a single fused TileNode preserving external edges."""
        src = self.nodes[src_id]
        dst = self.nodes[dst_id]
        merged_attrs = {**src.attrs, **dst.attrs, **(attrs or {})}
        merged_ops = list(src.fused_ops) + list(dst.fused_ops)
        src_out_names = {t.name for t in src.outputs}
        fused_inputs = list(src.inputs)
        seen_names = {t.name for t in fused_inputs}
        for inp in dst.inputs:
            if inp.name not in src_out_names and inp.name not in seen_names:
                fused_inputs.append(inp)
                seen_names.add(inp.name)
        fused = TileNode(
            node_id=src.node_id,
            op_type=new_op,
            name=new_name or f"{src.name}+{dst.name}",
            inputs=fused_inputs,
            outputs=list(dst.outputs),
            tile_shape=src.tile_shape,
            pipeline_stages=max(src.pipeline_stages, dst.pipeline_stages),
            prefetch_stage=src.prefetch_stage,
            occupancy=max(src.occupancy, dst.occupancy),
            attrs=merged_attrs,
            fused_ops=merged_ops,
        )

        # Redirect incoming edges to dst (excluding src->dst) and outgoing edges from dst
        new_edges: list[Edge] = []
        for edge in self.edges:
            if edge.src_id == src_id and edge.dst_id == dst_id:
                continue
            if edge.dst_id == dst_id:
                new_edges.append(Edge(src_id=edge.src_id, dst_id=fused.node_id, tensor=edge.tensor))
            elif edge.src_id == dst_id:
                new_edges.append(Edge(src_id=fused.node_id, dst_id=edge.dst_id, tensor=edge.tensor))
            else:
                new_edges.append(edge)

        del self.nodes[dst_id]
        self.nodes[fused.node_id] = fused
        self.edges = new_edges
        self.graph_outputs = [fused.node_id if nid == dst_id else nid for nid in self.graph_outputs]
        return fused

    def topological_order(self) -> list[TileNode]:
        """Return nodes in topological order; raises ValueError if a cycle exists."""
        in_deg: dict[str, int] = {nid: 0 for nid in self.nodes}
        for edge in self.edges:
            if edge.src_id not in self.nodes or edge.dst_id not in self.nodes:
                raise ValueError(f"Dangling edge: {edge.src_id} -> {edge.dst_id}")
            in_deg[edge.dst_id] += 1

        queue = deque([nid for nid, deg in in_deg.items() if deg == 0])
        ordered: list[TileNode] = []
        while queue:
            nid = queue.popleft()
            ordered.append(self.nodes[nid])
            for edge in self.edges:
                if edge.src_id == nid:
                    in_deg[edge.dst_id] -= 1
                    if in_deg[edge.dst_id] == 0:
                        queue.append(edge.dst_id)

        if len(ordered) != len(self.nodes):
            raise ValueError("Cycle detected in FusionGraph")
        return ordered

    def validate(self) -> bool:
        """Validate graph structure (no dangling edges, no self-loops, acyclic)."""
        for edge in self.edges:
            if edge.src_id not in self.nodes or edge.dst_id not in self.nodes:
                raise ValueError(f"Invalid edge endpoint in {edge.src_id} -> {edge.dst_id}")
            if edge.src_id == edge.dst_id:
                raise ValueError(f"Self-loop on node {edge.src_id}")
        self.topological_order()
        return True


def build_mistral_layer_tfg(
    layer_idx: int = 0,
    config: Any | None = None,
    seq_len: int = 256,
) -> FusionGraph:
    """Build the 8-node unfused TFG for a Mistral-7B transformer layer."""
    hidden = getattr(config, "hidden_size", 4096) if config is not None else 4096
    inter = getattr(config, "intermediate_size", 14336) if config is not None else 14336
    n_heads = getattr(config, "num_attention_heads", 32) if config is not None else 32
    n_kv_heads = getattr(config, "num_key_value_heads", 8) if config is not None else 8
    head_dim = getattr(config, "head_dim", 128) if config is not None else 128
    qkv_dim = (n_heads + 2 * n_kv_heads) * head_dim

    g = FusionGraph(name=f"mistral_layer_{layer_idx}")

    t_in = DataTensor("hidden_in", (seq_len, hidden), "float16", "hbm")
    t_qkv = DataTensor("qkv_out", (seq_len, qkv_dim), "float16", "hbm")
    t_attn = DataTensor("attn_out", (seq_len, hidden), "float16", "hbm")
    t_o = DataTensor("o_out", (seq_len, hidden), "float16", "hbm")
    t_ln1 = DataTensor("ln1_out", (seq_len, hidden), "float16", "hbm")
    t_ffn_up = DataTensor("ffn_up_out", (seq_len, inter), "float16", "hbm")
    t_silu = DataTensor("silu_out", (seq_len, inter), "float16", "hbm")
    t_ffn_down = DataTensor("ffn_down_out", (seq_len, hidden), "float16", "hbm")
    t_out = DataTensor("hidden_out", (seq_len, hidden), "float16", "hbm")

    n0 = g.add_node(
        TileNode(
            node_id=f"L{layer_idx}_qkv_gemm",
            op_type=OpType.QKV_GEMM,
            name="QKV GEMM",
            inputs=[t_in],
            outputs=[t_qkv],
            attrs={"layer": layer_idx, "M": seq_len, "N": qkv_dim, "K": hidden, "quantized": True},
        )
    )
    n1 = g.add_node(
        TileNode(
            node_id=f"L{layer_idx}_attention",
            op_type=OpType.ATTENTION_GQA,
            name="GQA Attention",
            inputs=[t_qkv],
            outputs=[t_attn],
            attrs={
                "layer": layer_idx,
                "n_heads": n_heads,
                "n_kv_heads": n_kv_heads,
                "head_dim": head_dim,
            },
        )
    )
    n2 = g.add_node(
        TileNode(
            node_id=f"L{layer_idx}_o_gemm",
            op_type=OpType.O_GEMM,
            name="O GEMM",
            inputs=[t_attn],
            outputs=[t_o],
            attrs={"layer": layer_idx, "M": seq_len, "N": hidden, "K": hidden, "quantized": True},
        )
    )
    n3 = g.add_node(
        TileNode(
            node_id=f"L{layer_idx}_layernorm",
            op_type=OpType.LAYERNORM,
            name="LayerNorm",
            inputs=[t_o],
            outputs=[t_ln1],
            attrs={"layer": layer_idx, "D": hidden},
        )
    )
    n4 = g.add_node(
        TileNode(
            node_id=f"L{layer_idx}_ffn_up",
            op_type=OpType.FFN_UP_GEMM,
            name="FFN-up GEMM",
            inputs=[t_ln1],
            outputs=[t_ffn_up],
            attrs={"layer": layer_idx, "M": seq_len, "N": inter, "K": hidden, "quantized": True},
        )
    )
    n5 = g.add_node(
        TileNode(
            node_id=f"L{layer_idx}_silu",
            op_type=OpType.SILU,
            name="SiLU",
            inputs=[t_ffn_up],
            outputs=[t_silu],
            attrs={"layer": layer_idx, "D": inter},
        )
    )
    n6 = g.add_node(
        TileNode(
            node_id=f"L{layer_idx}_ffn_down",
            op_type=OpType.FFN_DOWN_GEMM,
            name="FFN-down GEMM",
            inputs=[t_silu],
            outputs=[t_ffn_down],
            attrs={"layer": layer_idx, "M": seq_len, "N": hidden, "K": inter, "quantized": True},
        )
    )
    n7 = g.add_node(
        TileNode(
            node_id=f"L{layer_idx}_ln_res",
            op_type=OpType.LAYERNORM_RESIDUAL,
            name="LayerNorm+Residual",
            inputs=[t_ffn_down, t_o],
            outputs=[t_out],
            attrs={"layer": layer_idx, "D": hidden},
        )
    )

    g.add_edge(n0.node_id, n1.node_id, t_qkv)
    g.add_edge(n1.node_id, n2.node_id, t_attn)
    g.add_edge(n2.node_id, n3.node_id, t_o)
    g.add_edge(n3.node_id, n4.node_id, t_ln1)
    g.add_edge(n4.node_id, n5.node_id, t_ffn_up)
    g.add_edge(n5.node_id, n6.node_id, t_silu)
    g.add_edge(n6.node_id, n7.node_id, t_ffn_down)
    g.add_edge(n2.node_id, n7.node_id, t_o)

    g.validate()
    return g
