"""Software pipeline insertion pass (stages 0, 1, 2) for TFG nodes."""

from __future__ import annotations

from dataclasses import dataclass

from cutile_forge.forge.fusion_graph import FusionGraph, OpType

# Target sm_89 (Ada Lovelace RTX 4060) configurable shared memory limit per SM (bytes)
SM89_MAX_SMEM_BYTES = 100 * 1024


@dataclass
class PipelinePlan:
    """Pipeline schedule metadata for a single TileNode."""

    node_id: str
    stages: int
    smem_bytes: int
    stages_schedule: tuple[str, ...]
    latency_hint: int


def estimate_smem_bytes(tile_shape: tuple[int, int, int], stages: int, quantized: bool) -> int:
    """Estimate shared-memory ring buffer footprint for a tiled GEMM/fused kernel."""
    tm, tn, tk = tile_shape
    a_bytes = tm * tk * 2  # float16 activation tile
    b_bytes = (tk * tn) // 2 if quantized else tk * tn * 2
    return (a_bytes + b_bytes) * stages


def insert_software_pipeline(
    graph: FusionGraph,
    pipeline_depth: int = 3,
    max_smem_bytes: int = SM89_MAX_SMEM_BYTES,
) -> dict[str, PipelinePlan]:
    """Insert 3-stage software pipeline metadata (load -> compute -> store) on TFG nodes.

    Reduces `pipeline_stages` automatically if the tile's shared memory footprint
    would exceed `max_smem_bytes` on the target SM.
    """
    if pipeline_depth < 1 or pipeline_depth > 4:
        raise ValueError(f"pipeline_depth must be in [1, 4], got {pipeline_depth}")

    plans: dict[str, PipelinePlan] = {}
    for node in graph.topological_order():
        quantized = bool(node.attrs.get("quantized", False))
        stages = pipeline_depth

        while (
            stages > 1 and estimate_smem_bytes(node.tile_shape, stages, quantized) > max_smem_bytes
        ):
            stages -= 1

        node.pipeline_stages = stages
        node.attrs["pipeline_depth"] = stages
        node.attrs["latency_hint"] = stages

        if node.op_type in (
            OpType.GEMM,
            OpType.GEMM_DEQUANT,
            OpType.QKV_GEMM,
            OpType.O_GEMM,
            OpType.FFN_UP_GEMM,
            OpType.FFN_DOWN_GEMM,
            OpType.GEMM_SILU,
            OpType.FUSED_QKV_ATTN,
            OpType.FUSED_O_LAYERNORM,
            OpType.FUSED_FFN,
            OpType.FUSED_FFN_NORM,
        ):
            schedule: list[str] = ["stage_-1_pcie_dma"] + [
                f"stage_{i}_{'load' if i == 0 else ('mma_compute' if i < stages - 1 else 'epilogue_store')}"
                for i in range(stages)
            ]
        else:
            schedule = ["stage_0_load", "stage_1_compute_store"]

        smem = estimate_smem_bytes(node.tile_shape, stages, quantized)
        node.attrs["smem_bytes"] = smem
        plans[node.node_id] = PipelinePlan(
            node_id=node.node_id,
            stages=stages,
            smem_bytes=smem,
            stages_schedule=tuple(schedule),
            latency_hint=stages,
        )

    return plans
