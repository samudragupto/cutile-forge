"""Compiler pass manager and general IR optimization/analysis passes for TFG."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections import deque
from collections.abc import Sequence
from dataclasses import dataclass, field

from cutile_forge.forge.fusion_graph import Edge, FusionGraph, OpType
from cutile_forge.forge.fusion_rules import FusionRule, apply_fusion_rules


@dataclass
class PassResult:
    """Outcome telemetry of a single compiler pass."""

    pass_name: str
    changed: bool
    nodes_before: int
    nodes_after: int
    notes: list[str] = field(default_factory=list)


@dataclass
class PassReport:
    """Aggregated report across a full `PassManager` pipeline execution."""

    results: list[PassResult]
    total_nodes_before: int
    total_nodes_after: int


class CompilerPass(ABC):
    """Abstract base class for a TFG compiler analysis or transformation pass."""

    name: str = "compiler_pass"

    @abstractmethod
    def run(self, graph: FusionGraph) -> PassResult:
        """Transform or annotate `graph` in place and return a `PassResult`."""


class ShapeDtypeInferPass(CompilerPass):
    """Validates inner dimensions on matmul ops and populates `M`/`N`/`K` tile metadata."""

    name = "shape_dtype_infer"

    def run(self, graph: FusionGraph) -> PassResult:
        nodes_before = len(graph.nodes)
        changed = False
        notes: list[str] = []

        for node in graph.topological_order():
            if node.op_type in (
                OpType.MATMUL,
                OpType.BATCHED_MATMUL,
                OpType.GEMM,
                OpType.GEMM_DEQUANT,
                OpType.QKV_GEMM,
                OpType.O_GEMM,
                OpType.FFN_UP_GEMM,
                OpType.FFN_DOWN_GEMM,
            ):
                if len(node.inputs) >= 2:
                    a, b = node.inputs[0], node.inputs[1]
                    if len(a.shape) >= 2 and len(b.shape) >= 2:
                        if a.shape[-1] != b.shape[-2]:
                            raise ValueError(
                                f"Matmul shape mismatch on node '{node.node_id}': "
                                f"{a.shape} vs {b.shape}"
                            )
                        inferred_m = 1
                        for d in a.shape[:-1]:
                            inferred_m *= d
                        inferred_k = a.shape[-1]
                        inferred_n = b.shape[-1]
                        if "M" not in node.attrs:
                            node.attrs["M"] = inferred_m
                            changed = True
                        if "K" not in node.attrs:
                            node.attrs["K"] = inferred_k
                            changed = True
                        if "N" not in node.attrs:
                            node.attrs["N"] = inferred_n
                            changed = True
                        notes.append(f"{node.node_id}: ({inferred_m}, {inferred_n}, {inferred_k})")

        return PassResult(
            pass_name=self.name,
            changed=changed,
            nodes_before=nodes_before,
            nodes_after=len(graph.nodes),
            notes=notes,
        )


class CSEPass(CompilerPass):
    """Common subexpression elimination across pure identical TFG nodes."""

    name = "cse"

    def run(self, graph: FusionGraph) -> PassResult:
        nodes_before = len(graph.nodes)
        seen: dict[tuple[object, ...], str] = {}
        replaced: dict[str, str] = {}
        notes: list[str] = []

        for node in list(graph.topological_order()):
            if node.op_type in (OpType.INPUT, OpType.CUSTOM):
                continue
            preds = tuple(sorted(graph.predecessors(node.node_id)))
            in_sig = tuple((t.name, t.shape, t.dtype) for t in node.inputs)
            attr_sig = tuple(
                sorted(
                    (k, repr(v))
                    for k, v in node.attrs.items()
                    if k not in ("fusion_legal", "fusion_notes")
                )
            )
            key = (node.op_type.value, preds, in_sig, attr_sig)
            if key in seen:
                canon_id = seen[key]
                replaced[node.node_id] = canon_id
                notes.append(f"merged duplicate {node.node_id} -> {canon_id}")
            else:
                seen[key] = node.node_id

        if replaced:
            new_edges: list[Edge] = []
            for edge in graph.edges:
                if edge.dst_id in replaced:
                    continue
                src_id = replaced.get(edge.src_id, edge.src_id)
                dst_id = edge.dst_id
                canon_out_name = (
                    graph.nodes[src_id].outputs[0].name
                    if edge.src_id in replaced and graph.nodes[src_id].outputs
                    else edge.tensor.name
                )
                tensor = edge.tensor
                if canon_out_name != tensor.name:
                    # Also redirect downstream node input reference name if it matches replaced output
                    dup_out_names = {t.name for t in graph.nodes[edge.src_id].outputs}
                    for inp in graph.nodes[dst_id].inputs:
                        if inp.name in dup_out_names:
                            inp.name = canon_out_name
                    tensor = graph.nodes[src_id].outputs[0]
                new_edges.append(Edge(src_id=src_id, dst_id=dst_id, tensor=tensor))
            graph.edges = new_edges
            for dup_id in replaced:
                graph.nodes.pop(dup_id, None)
            graph.graph_outputs = [replaced.get(nid, nid) for nid in graph.graph_outputs]

        return PassResult(
            pass_name=self.name,
            changed=bool(replaced),
            nodes_before=nodes_before,
            nodes_after=len(graph.nodes),
            notes=notes,
        )


class FusionLegalityPass(CompilerPass):
    """Checks dtype match and fanout legality before algebraic/tile fusion."""

    name = "fusion_legality"

    def run(self, graph: FusionGraph) -> PassResult:
        nodes_before = len(graph.nodes)
        changed = False
        notes: list[str] = []

        for node in graph.topological_order():
            succ_ids = graph.successors(node.node_id)
            out_dtypes = {t.dtype for t in node.outputs}
            legal = True
            reason = "ok"

            if len(succ_ids) > 1 and node.op_type != OpType.O_GEMM:
                legal = False
                reason = f"multi-consumer fanout ({len(succ_ids)})"
            elif len(succ_ids) == 1:
                dst = graph.nodes[succ_ids[0]]
                dst_in_dtypes = {t.dtype for t in dst.inputs}
                if out_dtypes and dst_in_dtypes and not (out_dtypes & dst_in_dtypes):
                    legal = False
                    reason = f"dtype mismatch ({out_dtypes} vs {dst_in_dtypes})"

            if node.attrs.get("fusion_legal") != legal:
                changed = True
            node.attrs["fusion_legal"] = legal
            node.attrs["fusion_notes"] = reason
            notes.append(f"{node.node_id}: {reason}")

        return PassResult(
            pass_name=self.name,
            changed=changed,
            nodes_before=nodes_before,
            nodes_after=len(graph.nodes),
            notes=notes,
        )


class TFGFusionPass(CompilerPass):
    """Applies algebraic and tile-level TFG fusion rules to fixed point."""

    name = "tfg_fusion"

    def __init__(self, rules: Sequence[FusionRule] | None = None) -> None:
        self.rules = rules

    def run(self, graph: FusionGraph) -> PassResult:
        nodes_before = len(graph.nodes)
        fired = apply_fusion_rules(graph, rules=self.rules)
        return PassResult(
            pass_name=self.name,
            changed=bool(fired),
            nodes_before=nodes_before,
            nodes_after=len(graph.nodes),
            notes=fired,
        )


class DCEPass(CompilerPass):
    """Dead-code elimination removing nodes unreachable from explicit `graph_outputs`."""

    name = "dce"

    def run(self, graph: FusionGraph) -> PassResult:
        nodes_before = len(graph.nodes)
        if not graph.graph_outputs:
            return PassResult(
                pass_name=self.name,
                changed=False,
                nodes_before=nodes_before,
                nodes_after=nodes_before,
                notes=["no explicit graph_outputs; skipped"],
            )

        live: set[str] = set()
        queue = deque([nid for nid in graph.graph_outputs if nid in graph.nodes])
        while queue:
            nid = queue.popleft()
            if nid in live:
                continue
            live.add(nid)
            for pred_id in graph.predecessors(nid):
                if pred_id not in live:
                    queue.append(pred_id)

        dead = [nid for nid in list(graph.nodes) if nid not in live]
        for nid in dead:
            graph.remove_node(nid)

        return PassResult(
            pass_name=self.name,
            changed=bool(dead),
            nodes_before=nodes_before,
            nodes_after=len(graph.nodes),
            notes=[f"removed {len(dead)} dead node(s)"],
        )


class PassManager:
    """Runs an ordered sequence of `CompilerPass`es and validates graph invariants."""

    def __init__(self, passes: Sequence[CompilerPass]) -> None:
        self.passes = list(passes)

    def run(self, graph: FusionGraph) -> PassReport:
        total_before = len(graph.nodes)
        results: list[PassResult] = []
        for p in self.passes:
            res = p.run(graph)
            graph.validate()
            results.append(res)
        return PassReport(
            results=results,
            total_nodes_before=total_before,
            total_nodes_after=len(graph.nodes),
        )


def default_pass_manager() -> PassManager:
    """Build the default TFG + general optimization pass pipeline."""
    return PassManager(
        [
            ShapeDtypeInferPass(),
            CSEPass(),
            FusionLegalityPass(),
            TFGFusionPass(),
            DCEPass(),
        ]
    )
