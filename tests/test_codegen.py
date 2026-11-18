"""Unit tests for TFG codegen, pipelining, and SVG visualization."""

from __future__ import annotations

from pathlib import Path

import torch

from cutile_forge.forge.codegen import compile_tfg
from cutile_forge.forge.fusion_graph import build_mistral_layer_tfg
from cutile_forge.forge.fusion_rules import apply_fusion_rules
from cutile_forge.forge.graph_viz import render_svg, to_dot


def test_compile_tfg_produces_3_kernels() -> None:
    g = build_mistral_layer_tfg(0)
    apply_fusion_rules(g)
    compiled = compile_tfg(g, target="sm_89", pipeline_depth=3, compile_cubin=False)
    assert len(compiled.kernels) == 3
    assert len(compiled.schedule) == 3
    for k in compiled.kernels:
        assert "@ct.kernel" in k.source_code
        assert k.pipeline_depth >= 1


def test_compile_tfg_cubin_on_gpu() -> None:
    if not torch.cuda.is_available():
        return
    g = build_mistral_layer_tfg(0)
    apply_fusion_rules(g)
    compiled = compile_tfg(g, target="sm_89", pipeline_depth=3, compile_cubin=True)
    assert len(compiled.kernels) == 3
    assert any(k.cubin_bytes is not None and len(k.cubin_bytes) > 0 for k in compiled.kernels)


def test_dot_and_svg_generation(tmp_path: Path) -> None:
    g = build_mistral_layer_tfg(0)
    apply_fusion_rules(g)
    dot = to_dot(g)
    assert "digraph" in dot
    svg_path = render_svg(g, tmp_path / "tfg.svg")
    assert svg_path.exists()
    content = svg_path.read_text(encoding="utf-8")
    assert "<svg" in content and "</svg>" in content


def test_compiled_summary() -> None:
    g = build_mistral_layer_tfg(1)
    apply_fusion_rules(g)
    compiled = compile_tfg(g, target="sm_89", pipeline_depth=3)
    text = compiled.summary()
    assert "mistral_layer_1" in text
    assert "kernels=3" in text
