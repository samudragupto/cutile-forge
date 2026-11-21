"""CuTileVision — human-readable compiler diagnostics, fusion explanations & resource reports."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from cutile_forge.forge.codegen import CompiledTFG
from cutile_forge.forge.fusion_graph import FusionGraph


@dataclass(frozen=True)
class FusionDecision:
    """Diagnostic explanation of how an original node was compiled or fused."""

    node_id: str
    op_type: str
    status: str  # "fused" | "kept" | "rejected"
    reason: str
    fused_ops: list[str]


@dataclass
class DiagnosticSummary:
    """Complete compiler inspection report comparing original and compiled TFGs."""

    graph_name: str
    target: str
    nodes_before: int
    nodes_after: int
    decisions: list[FusionDecision] = field(default_factory=list)
    kernel_metrics: list[dict[str, Any]] = field(default_factory=list)
    pass_notes: list[str] = field(default_factory=list)
    total_hbm_bytes: int = 0
    total_hbm_bytes_saved: int = 0
    buffer_slots: int = 0
    timeline_events: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "graph_name": self.graph_name,
            "target": self.target,
            "nodes_before": self.nodes_before,
            "nodes_after": self.nodes_after,
            "total_hbm_bytes": self.total_hbm_bytes,
            "total_hbm_bytes_saved": self.total_hbm_bytes_saved,
            "buffer_slots": self.buffer_slots,
            "decisions": [
                {
                    "node_id": d.node_id,
                    "op_type": d.op_type,
                    "status": d.status,
                    "reason": d.reason,
                    "fused_ops": list(d.fused_ops),
                }
                for d in self.decisions
            ],
            "kernel_metrics": list(self.kernel_metrics),
            "timeline_events": list(self.timeline_events),
            "pass_notes": list(self.pass_notes),
        }

    def to_markdown(self) -> str:
        lines = [
            f"# CuTileVision Report: `{self.graph_name}` (`{self.target}`)",
            "",
            f"- **Nodes:** `{self.nodes_before} -> {self.nodes_after}`",
            f"- **Estimated HBM Traffic:** `{self.total_hbm_bytes:,} B` (saved via fusion: `{self.total_hbm_bytes_saved:,} B`)",
            f"- **Reusable Buffer Slots:** `{self.buffer_slots}`",
            "",
            "## Fusion Decisions",
            "| Node | Output Op | Status | Fused Ops | Reason |",
            "|---|---|---|---|---|",
        ]
        for d in self.decisions:
            ops = " -> ".join(d.fused_ops)
            lines.append(f"| `{d.node_id}` | `{d.op_type}` | {d.status} | `{ops}` | {d.reason} |")

        lines.extend(
            [
                "",
                "## Compiled Kernels & Resource Footprint",
                "| Kernel | Tile Shape | Pipeline Stages | SMEM (B) | Grid | Est. Waves |",
                "|---|---|---:|---:|---|---:|",
            ]
        )
        for km in self.kernel_metrics:
            lines.append(
                f"| `{km['kernel_name']}` | `{km['tile_shape']}` | {km['pipeline_depth']} "
                f"| {km['smem_bytes']} | `{km['grid']}` | {km['estimated_waves']} |"
            )
        return "\n".join(lines)


def inspect_compilation(original: FusionGraph, compiled: CompiledTFG) -> DiagnosticSummary:
    """Inspect a `CompiledTFG` against its `original` graph to explain fusion and SMEM usage."""
    final_graph = compiled.graph or original
    sched_by_id = {s.node_id: s for s in compiled.schedule}

    decisions: list[FusionDecision] = []
    kernel_metrics: list[dict[str, Any]] = []
    timeline_events: list[dict[str, Any]] = []
    total_hbm_bytes = 0
    total_hbm_bytes_saved = 0
    buffer_slots = 0

    for node in final_graph.topological_order():
        total_hbm_bytes += sum(t.nbytes for t in node.inputs) + sum(t.nbytes for t in node.outputs)
        total_hbm_bytes_saved += int(node.attrs.get("hbm_bytes_saved", 0))
        buffer_slots = max(buffer_slots, int(node.attrs.get("total_buffer_slots", 0)))

        if len(node.fused_ops) > 1:
            status = "fused"
            reason = f"merged {len(node.fused_ops)} ops into {node.op_type.value}"
        elif node.attrs.get("fusion_legal") is False:
            status = "rejected"
            reason = str(node.attrs.get("fusion_notes", "legality constraint"))
        else:
            status = "kept"
            reason = str(node.attrs.get("fusion_notes", "standalone node"))

        decisions.append(
            FusionDecision(
                node_id=node.node_id,
                op_type=node.op_type.value,
                status=status,
                reason=reason,
                fused_ops=list(node.fused_ops),
            )
        )

    for order_idx, k in enumerate(compiled.kernels):
        node_ref = final_graph.nodes.get(k.node_id)
        sched = sched_by_id.get(k.node_id)
        smem = int(node_ref.attrs.get("smem_bytes", 0)) if node_ref is not None else 0
        slot = node_ref.attrs.get("buffer_slot") if node_ref is not None else None
        kernel_metrics.append(
            {
                "node_id": k.node_id,
                "kernel_name": k.kernel_name,
                "op_type": k.op_type.value,
                "tile_shape": k.tile_shape,
                "pipeline_depth": k.pipeline_depth,
                "smem_bytes": smem,
                "grid": sched.grid if sched is not None else (1, 1, 1),
                "estimated_waves": sched.estimated_waves if sched is not None else 0.0,
                "buffer_slot": slot,
            }
        )
        timeline_events.append(
            {
                "order": order_idx,
                "node_id": k.node_id,
                "stream_id": sched.stream_id if sched is not None else 0,
                "prefetch_stage": -1,
                "buffer_slot": slot,
            }
        )

    pass_notes: list[str] = []
    if compiled.pass_report is not None:
        for res in compiled.pass_report.results:
            pass_notes.append(f"{res.pass_name}: {res.nodes_before}->{res.nodes_after}")

    return DiagnosticSummary(
        graph_name=compiled.graph_name,
        target=compiled.target,
        nodes_before=len(original.nodes),
        nodes_after=len(final_graph.nodes),
        decisions=decisions,
        kernel_metrics=kernel_metrics,
        pass_notes=pass_notes,
        total_hbm_bytes=total_hbm_bytes,
        total_hbm_bytes_saved=total_hbm_bytes_saved,
        buffer_slots=buffer_slots,
        timeline_events=timeline_events,
    )
