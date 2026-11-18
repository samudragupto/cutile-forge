"""Configuration dataclasses for Mistral-7B, RTX 4060 GPU, and benchmarks."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class ModelConfig:
    """Architecture and quantization parameters for Mistral-7B (and test configs)."""

    vocab_size: int = 32000
    hidden_size: int = 4096
    intermediate_size: int = 14336
    num_hidden_layers: int = 32
    num_attention_heads: int = 32
    num_key_value_heads: int = 8
    head_dim: int = 128
    max_position_embeddings: int = 32768
    rms_norm_eps: float = 1e-5
    rope_theta: float = 10000.0
    bits: int = 4
    group_size: int = 128
    bos_token_id: int = 1
    eos_token_id: int = 2

    def __post_init__(self) -> None:
        if self.hidden_size <= 0 or self.num_hidden_layers <= 0:
            raise ValueError("hidden_size and num_hidden_layers must be positive")
        if self.num_attention_heads % self.num_key_value_heads != 0:
            raise ValueError("num_attention_heads must be divisible by num_key_value_heads")
        if self.bits not in (4, 16):
            raise ValueError(f"Unsupported bit-width: {self.bits}")
        if self.group_size <= 0:
            raise ValueError(f"group_size must be positive, got {self.group_size}")

    @property
    def qkv_dim(self) -> int:
        return (self.num_attention_heads + 2 * self.num_key_value_heads) * self.head_dim

    @property
    def gqa_ratio(self) -> int:
        return self.num_attention_heads // self.num_key_value_heads

    @classmethod
    def from_json(cls, path: str | Path) -> ModelConfig:
        """Load ModelConfig from a HuggingFace config.json (plus optional quantize_config.json)."""
        cfg_path = Path(path)
        data: dict[str, Any] = json.loads(cfg_path.read_text(encoding="utf-8"))
        q_path = cfg_path.parent / "quantize_config.json"
        if q_path.exists():
            q_data = json.loads(q_path.read_text(encoding="utf-8"))
            data.setdefault("bits", q_data.get("bits", 4))
            data.setdefault("group_size", q_data.get("group_size", 128))
        head_dim = data.get(
            "head_dim",
            data.get("hidden_size", 4096) // data.get("num_attention_heads", 32),
        )
        return cls(
            vocab_size=int(data.get("vocab_size", 32000)),
            hidden_size=int(data.get("hidden_size", 4096)),
            intermediate_size=int(data.get("intermediate_size", 14336)),
            num_hidden_layers=int(data.get("num_hidden_layers", 32)),
            num_attention_heads=int(data.get("num_attention_heads", 32)),
            num_key_value_heads=int(data.get("num_key_value_heads", 8)),
            head_dim=int(head_dim),
            max_position_embeddings=int(data.get("max_position_embeddings", 32768)),
            rms_norm_eps=float(data.get("rms_norm_eps", 1e-5)),
            rope_theta=float(data.get("rope_theta", 10000.0)),
            bits=int(data.get("bits", 4)),
            group_size=int(data.get("group_size", 128)),
            bos_token_id=int(data.get("bos_token_id", 1)),
            eos_token_id=int(data.get("eos_token_id", 2)),
        )


MistralConfig = ModelConfig


@dataclass
class GPUConfig:
    """Target GPU hardware constraints (RTX 4060 Laptop 8 GB default)."""

    name: str = "RTX 4060 Laptop GPU"
    arch: str = "sm_89"
    num_sms: int = 24
    vram_gb: float = 8.0
    vram_budget_gb: float = 7.2
    smem_per_sm_kb: int = 100
    pcie_gen: int = 4
    pcie_lanes: int = 16
    pipeline_depth: int = 3
    stream_weights: bool = False  # Stream from pinned DDR5 when True, or keep in 8GB VRAM

    def __post_init__(self) -> None:
        if self.vram_budget_gb > self.vram_gb:
            raise ValueError("vram_budget_gb cannot exceed total vram_gb")


@dataclass
class BenchConfig:
    """Benchmark execution settings."""

    warmup: int = 5
    iters: int = 200
    output_dir: Path = Path("benchmarks/results")
    model_dir: Path = Path(r"D:\models\mistral_7b_4bit")
    thermal_logging: bool = False
    pcie_logging: bool = False
    seq_lens: tuple[int, ...] = field(default=(1, 64, 256, 512))

    def __post_init__(self) -> None:
        if self.warmup < 0 or self.iters <= 0:
            raise ValueError("warmup must be >= 0 and iters > 0")
