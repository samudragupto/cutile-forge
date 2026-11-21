"""Compilation artifact cache keyed by canonical TFG topology, target, and pipeline."""

from __future__ import annotations

import hashlib
import json
from collections import OrderedDict
from typing import Any

from cutile_forge.forge.fusion_graph import FusionGraph


def compute_graph_hash(graph: FusionGraph, target: str, pipeline_depth: int) -> str:
    """Compute a deterministic SHA-256 digest of a FusionGraph and target configuration."""
    nodes_payload = []
    for node in graph.topological_order():
        nodes_payload.append(
            {
                "id": node.node_id,
                "op": node.op_type.value,
                "inputs": [(t.name, list(t.shape), t.dtype, t.quantized) for t in node.inputs],
                "outputs": [(t.name, list(t.shape), t.dtype, t.quantized) for t in node.outputs],
                "attrs": sorted((k, repr(v)) for k, v in node.attrs.items()),
            }
        )
    edges_payload = [(e.src_id, e.dst_id, e.tensor.name, list(e.tensor.shape)) for e in graph.edges]
    payload = {
        "name": graph.name,
        "target": target,
        "pipeline_depth": pipeline_depth,
        "graph_outputs": list(graph.graph_outputs),
        "nodes": nodes_payload,
        "edges": edges_payload,
    }
    blob = json.dumps(payload, sort_keys=True).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()


class ArtifactCache:
    """In-memory LRU cache storing compiled TFG artifacts."""

    def __init__(self, max_entries: int = 128) -> None:
        self.max_entries = max(1, max_entries)
        self._entries: OrderedDict[str, Any] = OrderedDict()
        self.hits: int = 0
        self.misses: int = 0

    def get(self, key: str) -> Any | None:
        if key not in self._entries:
            self.misses += 1
            return None
        self.hits += 1
        self._entries.move_to_end(key)
        return self._entries[key]

    def put(self, key: str, value: Any) -> None:
        if key in self._entries:
            self._entries.move_to_end(key)
            self._entries[key] = value
            return
        while len(self._entries) >= self.max_entries:
            self._entries.popitem(last=False)
        self._entries[key] = value

    def clear(self) -> None:
        self._entries.clear()
        self.hits = 0
        self.misses = 0

    def __len__(self) -> int:
        return len(self._entries)


GLOBAL_ARTIFACT_CACHE = ArtifactCache()
