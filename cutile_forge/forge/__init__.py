"""Tile Fusion Graph (TFG) compiler package."""

from cutile_forge.forge.artifact_cache import (
    GLOBAL_ARTIFACT_CACHE,
    ArtifactCache,
    compute_graph_hash,
)
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
from cutile_forge.forge.pass_manager import (
    CompilerPass,
    CSEPass,
    DCEPass,
    FusionLegalityPass,
    PassManager,
    PassReport,
    PassResult,
    ShapeDtypeInferPass,
    TFGFusionPass,
    default_pass_manager,
)
from cutile_forge.forge.scheduler import ScheduleEntry, schedule_tfg
from cutile_forge.forge.tile_autotuner import CANDIDATE_TILE_CONFIGS, TileAutotuner, TileConfig
from cutile_forge.forge.tile_pipeliner import (
    SM89_MAX_SMEM_BYTES,
    SM120_MAX_SMEM_BYTES,
    PipelinePlan,
    insert_software_pipeline,
    resolve_target_specs,
)
from cutile_forge.forge.weight_streamer import PrefetchSlot, WeightStreamer, attach_weight_streaming

__all__ = [
    "ArtifactCache",
    "CANDIDATE_TILE_CONFIGS",
    "CSEPass",
    "CompiledKernel",
    "CompiledTFG",
    "CompilerPass",
    "DCEPass",
    "DEFAULT_FUSION_RULES",
    "DataTensor",
    "Edge",
    "FusionGraph",
    "FusionLegalityPass",
    "FusionRule",
    "GLOBAL_ARTIFACT_CACHE",
    "OpType",
    "PassManager",
    "PassReport",
    "PassResult",
    "PipelinePlan",
    "PrefetchSlot",
    "SM89_MAX_SMEM_BYTES",
    "SM120_MAX_SMEM_BYTES",
    "ScheduleEntry",
    "ShapeDtypeInferPass",
    "TFGFusionPass",
    "TileAutotuner",
    "TileConfig",
    "TileNode",
    "WeightStreamer",
    "apply_fusion_rules",
    "attach_weight_streaming",
    "build_mistral_layer_tfg",
    "compile_tfg",
    "compute_graph_hash",
    "default_pass_manager",
    "insert_software_pipeline",
    "render_svg",
    "resolve_target_specs",
    "schedule_tfg",
    "to_dot",
]
