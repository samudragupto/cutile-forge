"""Device capability and precision abstractions for the CuTileForge SDK."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

import torch

from cutile_forge.forge.tile_pipeliner import resolve_target_specs


class DType(StrEnum):
    """Supported tensor element precision identifiers."""

    FP16 = "float16"
    BF16 = "bfloat16"
    FP32 = "float32"
    FP8_E4M3 = "float8_e4m3fn"
    INT8 = "int8"
    INT4 = "int4"
    NVFP4 = "float4_e2m1fn"


@dataclass(frozen=True)
class Device:
    """Describes an NVIDIA GPU compilation/execution target."""

    arch: str
    num_sms: int
    smem_per_sm_bytes: int
    has_cuda: bool
    name: str

    @classmethod
    def auto(cls, target: str = "auto") -> Device:
        """Detect live GPU capabilities or resolve an explicit SM target (`sm_89`/`sm_120`)."""
        arch, num_sms, smem_bytes = resolve_target_specs(target)
        has_cuda = torch.cuda.is_available()
        if has_cuda and target in ("auto", arch):
            props = torch.cuda.get_device_properties(0)
            dev_name = str(props.name)
        else:
            dev_name = f"Target({arch})"
        return cls(
            arch=arch,
            num_sms=num_sms,
            smem_per_sm_bytes=smem_bytes,
            has_cuda=has_cuda,
            name=dev_name,
        )
