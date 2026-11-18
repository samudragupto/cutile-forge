"""Tile Fusion Graph (TFG) compiler package."""

from cutile_forge.forge.codegen import CompiledKernel, CompiledTFG, compile_tfg
from cutile_forge.forge.fusion_graph import (
    DataTensor,
    Edge,
    FusionGraph,
    OpType,
    TileNode,
    build_mistral_layer_tfg,
)
from cutile_forge.forge.fusion_rules import DEFAULT_FUSION_RULES, FusionRule, apply_fusion_rules
from cutile_forge.forge.graph_viz import render_svg, to_dot
from cutile_forge.forge.scheduler import ScheduleEntry, schedule_tfg
from cutile_forge.forge.tile_autotuner import CANDIDATE_TILE_CONFIGS, TileAutotuner, TileConfig
from cutile_forge.forge.tile_pipeliner import PipelinePlan, insert_software_pipeline
from cutile_forge.forge.weight_streamer import PrefetchSlot, WeightStreamer, attach_weight_streaming

__all__ = [
    "CANDIDATE_TILE_CONFIGS",
    "CompiledKernel",
    "CompiledTFG",
    "DEFAULT_FUSION_RULES",
    "DataTensor",
    "Edge",
    "FusionGraph",
    "FusionRule",
    "OpType",
    "PipelinePlan",
    "PrefetchSlot",
    "ScheduleEntry",
    "TileAutotuner",
    "TileConfig",
    "TileNode",
    "WeightStreamer",
    "apply_fusion_rules",
    "attach_weight_streaming",
    "build_mistral_layer_tfg",
    "compile_tfg",
    "insert_software_pipeline",
    "render_svg",
    "schedule_tfg",
    "to_dot",
]
