"""General-purpose graph runtime supporting heterogeneous CPU/GPU tensor execution."""

from __future__ import annotations

from typing import Any

import torch

from cutile_forge.forge.codegen import CompiledTFG
from cutile_forge.runtime.dma import TensorStreamer


class GraphRuntime:
    """Executes compiled TFG/SDK graphs with automatic CPU->GPU async DMA staging."""

    def __init__(
        self,
        num_slots: int = 3,
        max_vram_bytes: int | None = None,
        device: str | torch.device = "cuda",
    ) -> None:
        self.streamer = TensorStreamer(
            num_slots=num_slots,
            max_vram_bytes=max_vram_bytes,
            device=device,
        )
        self.target_is_cuda = self.streamer.use_cuda

    def execute(
        self,
        artifact: Any,
        inputs: dict[str, torch.Tensor],
        custom_registry: Any | None = None,
    ) -> dict[str, torch.Tensor]:
        """Stage CPU tensors asynchronously to CUDA when needed and execute `artifact`."""
        compiled_tfg: CompiledTFG = (
            artifact.compiled_tfg if hasattr(artifact, "compiled_tfg") else artifact
        )
        reg = custom_registry or getattr(artifact, "registry", None)

        staged: dict[str, torch.Tensor] = {}
        cpu_keys: list[str] = []
        for name, t in inputs.items():
            if self.target_is_cuda and t.device.type == "cpu":
                self.streamer.prefetch(name, t)
                cpu_keys.append(name)
            else:
                staged[name] = t

        for name in cpu_keys:
            staged[name] = self.streamer.acquire(name)

        try:
            return compiled_tfg.run(staged, custom_registry=reg)
        finally:
            for name in cpu_keys:
                self.streamer.release(name)
