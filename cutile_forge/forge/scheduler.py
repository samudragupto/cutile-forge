"""SM-aware tile scheduler for Ada (sm_89), Blackwell (sm_120), and L2 cache swizzling."""

from __future__ import annotations

from dataclasses import dataclass

from cutile_forge.forge.fusion_graph import FusionGraph
from cutile_forge.forge.tile_pipeliner import resolve_target_specs


@dataclass
class ScheduleEntry:
    """Execution schedule metadata for one compiled TFG kernel."""

    node_id: str
    order: int
    grid: tuple[int, int, int]
    swizzle_group_m: int
    estimated_waves: float
    stream_id: int


def schedule_tfg(
    graph: FusionGraph,
    num_sms: int | None = None,
    target: str | None = None,
) -> list[ScheduleEntry]:
    """Assign grid dimensions, L2 swizzle group, and wave estimates across SMs."""
    if num_sms is None:
        _, num_sms, _ = resolve_target_specs(target)
    entries: list[ScheduleEntry] = []
    for idx, node in enumerate(graph.topological_order()):
        m = int(node.attrs.get("M", 256))
        n = int(node.attrs.get("N", 4096))
        tm, tn, _ = node.tile_shape
        grid_m = (m + tm - 1) // tm
        grid_n = (n + tn - 1) // tn
        total_ctas = max(1, grid_m * grid_n)
        occupancy = max(1, node.occupancy)
        waves = total_ctas / float(num_sms * occupancy)
        swizzle = 8 if grid_m >= 8 else max(1, grid_m)
        entries.append(
            ScheduleEntry(
                node_id=node.node_id,
                order=idx,
                grid=(total_ctas, 1, 1),
                swizzle_group_m=swizzle,
                estimated_waves=round(waves, 2),
                stream_id=0,
            )
        )
    return entries
