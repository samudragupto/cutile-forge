"""Visualize a Mistral layer Tile Fusion Graph before and after fusion as SVG."""

from __future__ import annotations

import argparse
from pathlib import Path

from cutile_forge.engine.config import MistralConfig
from cutile_forge.forge.codegen import compile_tfg
from cutile_forge.forge.fusion_graph import build_mistral_layer_tfg
from cutile_forge.forge.fusion_rules import apply_fusion_rules
from cutile_forge.forge.graph_viz import render_svg


def main() -> None:
    parser = argparse.ArgumentParser(description="Visualize TFG as SVG")
    parser.add_argument("--layer", type=int, default=0)
    parser.add_argument("--target", type=str, default="sm_120")
    parser.add_argument("--pipeline-depth", type=int, default=4)
    parser.add_argument("--output", type=Path, default=Path("docs/tfg_layer0.svg"))
    args = parser.parse_args()

    cfg = MistralConfig()
    g = build_mistral_layer_tfg(args.layer, cfg)
    before = len(g.nodes)
    fired = apply_fusion_rules(g)
    after = len(g.nodes)
    compiled = compile_tfg(g, target=args.target, pipeline_depth=args.pipeline_depth)
    out_svg = render_svg(g, args.output)
    print(f"Layer {args.layer}: {before} -> {after} nodes (fired={fired})")
    print(compiled.summary())
    print(f"Saved SVG to {out_svg}")


if __name__ == "__main__":
    main()
