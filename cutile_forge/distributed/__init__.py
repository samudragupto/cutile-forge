"""Multi-GPU topology detection and peer-to-peer collective primitives."""

from cutile_forge.distributed.p2p import p2p_copy, ring_allreduce_sum
from cutile_forge.distributed.runtime import (
    ColumnParallelLinear,
    CommTransport,
    DistributedPagedKVCache,
    DistributedScheduler,
    ExpertParallelMoE,
    LoopbackCommTransport,
    PipelineParallelEngine,
    PipelineStage,
    RowParallelLinear,
    TensorParallelGQAAttention,
    TensorParallelSwiGLUMLP,
)
from cutile_forge.distributed.topology import DeviceInfo, DeviceTopology, detect_topology

__all__ = [
    "ColumnParallelLinear",
    "CommTransport",
    "DeviceInfo",
    "DeviceTopology",
    "DistributedPagedKVCache",
    "DistributedScheduler",
    "ExpertParallelMoE",
    "LoopbackCommTransport",
    "PipelineParallelEngine",
    "PipelineStage",
    "RowParallelLinear",
    "TensorParallelGQAAttention",
    "TensorParallelSwiGLUMLP",
    "detect_topology",
    "p2p_copy",
    "ring_allreduce_sum",
]
