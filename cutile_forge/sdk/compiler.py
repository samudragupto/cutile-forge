"""Compilation and execution API for the CuTileForge SDK."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import torch

from cutile_forge.forge.codegen import CompiledTFG, compile_tfg
from cutile_forge.forge.graph_viz import to_dot
from cutile_forge.forge.pass_manager import PassReport
from cutile_forge.sdk.device import Device
from cutile_forge.sdk.graph import Graph


class KernelRegistry:
    """Registry mapping custom op names to Python/CuTile callables."""

    def __init__(self) -> None:
        self._ops: dict[str, Callable[..., torch.Tensor]] = {}

    def register(
        self,
        name: str,
        fn: Callable[..., torch.Tensor] | None = None,
    ) -> Callable[..., Any]:
        """Register a custom kernel function directly or as a decorator."""
        if fn is not None:
            self._ops[name] = fn
            return fn

        def _decorator(target_fn: Callable[..., torch.Tensor]) -> Callable[..., torch.Tensor]:
            self._ops[name] = target_fn
            return target_fn

        return _decorator

    def get(self, name: str) -> Callable[..., torch.Tensor]:
        if name not in self._ops:
            raise KeyError(f"Unregistered custom kernel: '{name}'")
        return self._ops[name]

    def __contains__(self, name: str) -> bool:
        return name in self._ops


GLOBAL_KERNEL_REGISTRY = KernelRegistry()


@dataclass
class CompiledArtifact:
    """Executable artifact compiled from an SDK `Graph`."""

    compiled_tfg: CompiledTFG
    device: Device
    registry: KernelRegistry

    @property
    def cache_hit(self) -> bool:
        return self.compiled_tfg.cache_hit

    @property
    def pass_report(self) -> PassReport | None:
        return self.compiled_tfg.pass_report

    @property
    def kernels(self) -> list[Any]:
        return self.compiled_tfg.kernels

    def summary(self) -> str:
        return self.compiled_tfg.summary()

    def to_dot(self) -> str:
        if self.compiled_tfg.graph is None:
            raise RuntimeError("No FusionGraph attached to CompiledArtifact")
        return to_dot(self.compiled_tfg.graph)

    def run(self, **tensors: torch.Tensor) -> dict[str, torch.Tensor] | torch.Tensor:
        """Execute the compiled graph and return single tensor or `{name: tensor}` map."""
        out = self.compiled_tfg.run(tensors, custom_registry=self.registry)
        if len(out) == 1:
            return next(iter(out.values()))
        return out


def compile(
    graph: Graph,
    target: str = "auto",
    pipeline_depth: int | None = None,
    autotune: bool = True,
    use_cache: bool = True,
    compile_cubin: bool = False,
    registry: KernelRegistry | None = None,
) -> CompiledArtifact:
    """Compile an SDK `Graph` through TFG optimization passes and CuTile lowering."""
    dev = Device.auto(target)
    depth = pipeline_depth if pipeline_depth is not None else (4 if dev.arch == "sm_120" else 3)
    tfg = graph.to_fusion_graph()
    compiled = compile_tfg(
        tfg,
        target=dev.arch,
        pipeline_depth=depth,
        autotune=autotune,
        compile_cubin=compile_cubin,
        run_passes=True,
        use_cache=use_cache,
    )
    return CompiledArtifact(
        compiled_tfg=compiled,
        device=dev,
        registry=registry or GLOBAL_KERNEL_REGISTRY,
    )
