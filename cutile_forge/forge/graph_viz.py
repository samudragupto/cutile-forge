"""Graphviz DOT and SVG visualization for Tile Fusion Graphs."""

from __future__ import annotations

import html
import shutil
import subprocess
from pathlib import Path

from cutile_forge.forge.fusion_graph import FusionGraph, OpType

_OP_COLORS: dict[str, str] = {
    OpType.FUSED_QKV_ATTN.value: "#2563eb",
    OpType.FUSED_O_LAYERNORM.value: "#0d9488",
    OpType.FUSED_FFN.value: "#7c3aed",
    OpType.FUSED_FFN_NORM.value: "#7c3aed",
    OpType.GEMM_SILU.value: "#9333ea",
}


def to_dot(graph: FusionGraph) -> str:
    """Serialize a FusionGraph into Graphviz DOT syntax."""
    lines: list[str] = [
        f'digraph "{graph.name}" {{',
        "  rankdir=TB;",
        '  node [shape=box, style="rounded,filled", fontname="Helvetica", fontsize=11];',
        '  edge [fontname="Helvetica", fontsize=9];',
    ]
    for node in graph.topological_order():
        fill = _OP_COLORS.get(node.op_type.value, "#334155")
        ops_str = " -> ".join(node.fused_ops)
        label = (
            f"{node.name}\\n"
            f"[{node.op_type.value}]\\n"
            f"tile={node.tile_shape}, stages={node.pipeline_stages}\\n"
            f"ops: {ops_str}"
        )
        lines.append(
            f'  "{node.node_id}" [label="{label}", fillcolor="{fill}", fontcolor="white"];'
        )
    for edge in graph.edges:
        t_label = f"{edge.tensor.name} {edge.tensor.shape} ({edge.tensor.dtype})"
        lines.append(f'  "{edge.src_id}" -> "{edge.dst_id}" [label="{t_label}"];')
    lines.append("}")
    return "\n".join(lines)


def _fallback_svg(graph: FusionGraph) -> str:
    """Pure-Python SVG renderer used when the Graphviz `dot` binary is absent."""
    nodes = graph.topological_order()
    width = 640
    box_w = 460
    box_h = 68
    gap_y = 38
    margin_top = 48
    height = margin_top + max(1, len(nodes)) * (box_h + gap_y) + 20

    parts: list[str] = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#0f172a"/>',
        f'<text x="{width // 2}" y="28" text-anchor="middle" fill="#f8fafc" '
        f'font-family="sans-serif" font-size="16" font-weight="bold">'
        f"{html.escape(graph.name)} ({len(nodes)} nodes)</text>",
        '<defs><marker id="arrow" viewBox="0 0 10 10" refX="9" refY="5" '
        'markerWidth="6" markerHeight="6" orient="auto-start-reverse">'
        '<path d="M 0 1 L 10 5 L 0 9 z" fill="#94a3b8"/></marker></defs>',
    ]

    pos: dict[str, tuple[int, int]] = {}
    x0 = (width - box_w) // 2
    for idx, node in enumerate(nodes):
        y0 = margin_top + idx * (box_h + gap_y)
        pos[node.node_id] = (x0 + box_w // 2, y0)
        fill = _OP_COLORS.get(node.op_type.value, "#1e293b")
        ops_str = html.escape(" + ".join(node.fused_ops))
        title = html.escape(f"{node.name} [{node.op_type.value}]")
        meta = html.escape(
            f"tile={node.tile_shape} | stages={node.pipeline_stages} | ops: {ops_str}"
        )
        parts.append(
            f'<rect x="{x0}" y="{y0}" width="{box_w}" height="{box_h}" rx="10" '
            f'fill="{fill}" stroke="#cbd5e1" stroke-width="1.5"/>'
        )
        parts.append(
            f'<text x="{width // 2}" y="{y0 + 26}" text-anchor="middle" fill="#ffffff" '
            f'font-family="sans-serif" font-size="13" font-weight="bold">{title}</text>'
        )
        parts.append(
            f'<text x="{width // 2}" y="{y0 + 48}" text-anchor="middle" fill="#e2e8f0" '
            f'font-family="monospace" font-size="11">{meta}</text>'
        )

    for edge in graph.edges:
        if edge.src_id in pos and edge.dst_id in pos:
            sx, sy = pos[edge.src_id]
            dx, dy = pos[edge.dst_id]
            parts.append(
                f'<line x1="{sx}" y1="{sy + box_h}" x2="{dx}" y2="{dy - 2}" '
                f'stroke="#94a3b8" stroke-width="1.5" marker-end="url(#arrow)"/>'
            )

    parts.append("</svg>")
    return "\n".join(parts)


def render_svg(graph: FusionGraph, output_path: str | Path) -> Path:
    """Render a FusionGraph to an SVG file using Graphviz `dot` or pure-Python fallback."""
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    dot_src = to_dot(graph)

    dot_bin = shutil.which("dot")
    if dot_bin is not None:
        try:
            proc = subprocess.run(
                [dot_bin, "-Tsvg"],
                input=dot_src.encode("utf-8"),
                capture_output=True,
                check=True,
            )
            out.write_bytes(proc.stdout)
            return out
        except subprocess.SubprocessError:
            pass

    out.write_text(_fallback_svg(graph), encoding="utf-8")
    return out
