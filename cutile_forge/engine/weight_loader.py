"""4-bit GPTQ weight loader targeting pinned DDR5 host memory and 8 GB VRAM GPUs."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import torch

from cutile_forge.engine.config import ModelConfig
from cutile_forge.forge.weight_streamer import WeightStreamer
from cutile_forge.utils.quantize import quantize_4bit_group, unpack_gptq_layer


@dataclass
class QuantLinearWeight:
    """Packed 4-bit linear projection weight in CuTileForge `(K//2, N)` uint8 format."""

    qweight: torch.Tensor  # (K // 2, N) uint8
    scales: torch.Tensor   # (K // group_size, N) float16
    zeros: torch.Tensor    # (K // group_size, N) float16
    group_size: int = 128

    def to(self, device: torch.device | str) -> "QuantLinearWeight":
        return QuantLinearWeight(
            qweight=self.qweight.to(device, non_blocking=True),
            scales=self.scales.to(device, non_blocking=True),
            zeros=self.zeros.to(device, non_blocking=True),
            group_size=self.group_size,
        )

    @property
    def nbytes(self) -> int:
        return (
            self.qweight.numel() * self.qweight.element_size()
            + self.scales.numel() * self.scales.element_size()
            + self.zeros.numel() * self.zeros.element_size()
        )


@dataclass
class MistralLayerWeights:
    """All weights for one Mistral-7B transformer layer."""

    layer_idx: int
    input_layernorm: torch.Tensor
    post_attention_layernorm: torch.Tensor
    q_proj: QuantLinearWeight
    k_proj: QuantLinearWeight
    v_proj: QuantLinearWeight
    o_proj: QuantLinearWeight
    gate_proj: QuantLinearWeight
    up_proj: QuantLinearWeight
    down_proj: QuantLinearWeight

    def to_tensor_dict(self) -> Dict[str, torch.Tensor]:
        return {
            "input_layernorm": self.input_layernorm,
            "post_attention_layernorm": self.post_attention_layernorm,
            "q_qw": self.q_proj.qweight,
            "q_s": self.q_proj.scales,
            "q_z": self.q_proj.zeros,
            "k_qw": self.k_proj.qweight,
            "k_s": self.k_proj.scales,
            "k_z": self.k_proj.zeros,
            "v_qw": self.v_proj.qweight,
            "v_s": self.v_proj.scales,
            "v_z": self.v_proj.zeros,
            "o_qw": self.o_proj.qweight,
            "o_s": self.o_proj.scales,
            "o_z": self.o_proj.zeros,
            "gate_qw": self.gate_proj.qweight,
            "gate_s": self.gate_proj.scales,
            "gate_z": self.gate_proj.zeros,
            "up_qw": self.up_proj.qweight,
            "up_s": self.up_proj.scales,
            "up_z": self.up_proj.zeros,
            "down_qw": self.down_proj.qweight,
            "down_s": self.down_proj.scales,
            "down_z": self.down_proj.zeros,
        }

    @classmethod
    def from_tensor_dict(
        cls, layer_idx: int, d: Dict[str, torch.Tensor], group_size: int = 128
    ) -> "MistralLayerWeights":
        return cls(
            layer_idx=layer_idx,
            input_layernorm=d["input_layernorm"],
            post_attention_layernorm=d["post_attention_layernorm"],
            q_proj=QuantLinearWeight(d["q_qw"], d["q_s"], d["q_z"], group_size),
            k_proj=QuantLinearWeight(d["k_qw"], d["k_s"], d["k_z"], group_size),
            v_proj=QuantLinearWeight(d["v_qw"], d["v_s"], d["v_z"], group_size),
            o_proj=QuantLinearWeight(d["o_qw"], d["o_s"], d["o_z"], group_size),
            gate_proj=QuantLinearWeight(d["gate_qw"], d["gate_s"], d["gate_z"], group_size),
            up_proj=QuantLinearWeight(d["up_qw"], d["up_s"], d["up_z"], group_size),
            down_proj=QuantLinearWeight(d["down_qw"], d["down_s"], d["down_z"], group_size),
        )


@dataclass
class LoadedModelWeights:
    """Complete model weights split across resident GPU memory and pinned DDR5."""

    config: ModelConfig
    embed_tokens: torch.Tensor
    norm: torch.Tensor
    lm_head: torch.Tensor
    layers: List[MistralLayerWeights]


def _pin_dict(d: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    return {k: WeightStreamer.pin_tensor(v) for k, v in d.items()}


def load_gptq_weights(
    model_dir: str | Path,
    config: Optional[ModelConfig] = None,
    device: str | torch.device = "cuda",
    pin_ddr5: bool = True,
    max_gpu_layers: Optional[int] = None,
) -> LoadedModelWeights:
    """Load 4-bit GPTQ safetensors from `model_dir` into pinned DDR5 / GPU HBM."""
    from safetensors import safe_open

    base = Path(model_dir)
    if config is None:
        cfg_file = base / "config.json"
        config = ModelConfig.from_json(cfg_file) if cfg_file.exists() else ModelConfig()

    shard_files = sorted(base.glob("*.safetensors"))
    if not shard_files:
        idx_file = base / "model.safetensors.index.json"
        if idx_file.exists():
            weight_map = json.loads(idx_file.read_text(encoding="utf-8")).get("weight_map", {})
            shard_files = sorted({base / fname for fname in weight_map.values()})
    if not shard_files:
        raise FileNotFoundError(f"No .safetensors files found in {base}")

    raw: Dict[str, torch.Tensor] = {}
    for sf in shard_files:
        with safe_open(str(sf), framework="pt", device="cpu") as f:
            for key in f.keys():
                raw[key] = f.get_tensor(key)

    target_dev = torch.device(device if torch.cuda.is_available() else "cpu")
    embed_tokens = raw["model.embed_tokens.weight"].to(dtype=torch.float16, device=target_dev)
    norm = raw["model.norm.weight"].to(dtype=torch.float16, device=target_dev)
    lm_head = (
        raw["lm_head.weight"]
        if "lm_head.weight" in raw
        else raw["model.embed_tokens.weight"]
    ).to(dtype=torch.float16, device=target_dev)

    gpu_layers = config.num_hidden_layers if max_gpu_layers is None else max_gpu_layers
    layers: List[MistralLayerWeights] = []

    def _load_proj(prefix: str, target: torch.device | str) -> QuantLinearWeight:
        qw_i32 = raw[f"{prefix}.qweight"]
        qz_i32 = raw[f"{prefix}.qzeros"]
        sc = raw[f"{prefix}.scales"]
        g_idx = raw.get(f"{prefix}.g_idx")
        qw_u8, scales_fp16, zeros_fp16 = unpack_gptq_layer(
            qw_i32, qz_i32, sc, group_size=config.group_size, g_idx=g_idx
        )
        if str(target) == "cpu" and pin_ddr5:
            qw_u8 = WeightStreamer.pin_tensor(qw_u8)
            scales_fp16 = WeightStreamer.pin_tensor(scales_fp16)
            zeros_fp16 = WeightStreamer.pin_tensor(zeros_fp16)
        else:
            qw_u8 = qw_u8.to(target)
            scales_fp16 = scales_fp16.to(target)
            zeros_fp16 = zeros_fp16.to(target)
        return QuantLinearWeight(qw_u8, scales_fp16, zeros_fp16, group_size=config.group_size)

    for idx in range(config.num_hidden_layers):
        p = f"model.layers.{idx}"
        layer_dev: torch.device | str = target_dev if idx < gpu_layers else "cpu"
        in_ln = raw[f"{p}.input_layernorm.weight"].to(torch.float16)
        post_ln = raw[f"{p}.post_attention_layernorm.weight"].to(torch.float16)
        if str(layer_dev) == "cpu" and pin_ddr5:
            in_ln = WeightStreamer.pin_tensor(in_ln)
            post_ln = WeightStreamer.pin_tensor(post_ln)
        else:
            in_ln = in_ln.to(layer_dev)
            post_ln = post_ln.to(layer_dev)

        layers.append(
            MistralLayerWeights(
                layer_idx=idx,
                input_layernorm=in_ln,
                post_attention_layernorm=post_ln,
                q_proj=_load_proj(f"{p}.self_attn.q_proj", layer_dev),
                k_proj=_load_proj(f"{p}.self_attn.k_proj", layer_dev),
                v_proj=_load_proj(f"{p}.self_attn.v_proj", layer_dev),
                o_proj=_load_proj(f"{p}.self_attn.o_proj", layer_dev),
                gate_proj=_load_proj(f"{p}.mlp.gate_proj", layer_dev),
                up_proj=_load_proj(f"{p}.mlp.up_proj", layer_dev),
                down_proj=_load_proj(f"{p}.mlp.down_proj", layer_dev),
            )
        )

    return LoadedModelWeights(
        config=config,
        embed_tokens=embed_tokens,
        norm=norm,
        lm_head=lm_head,
        layers=layers,
    )


def create_random_4bit_weights(
    config: ModelConfig,
    device: str | torch.device = "cuda",
    seed: int = 42,
) -> LoadedModelWeights:
    """Generate deterministic random 4-bit quantized weights for testing & micro-runs."""
    gen = torch.Generator(device="cpu").manual_seed(seed)
    dev = torch.device(device if (str(device) != "cuda" or torch.cuda.is_available()) else "cpu")

    def _rand_fp16(shape: tuple[int, ...], scale: float = 0.02) -> torch.Tensor:
        return (torch.randn(shape, generator=gen, dtype=torch.float32) * scale).to(
            dtype=torch.float16, device=dev
        )

    def _rand_quant(k_dim: int, n_dim: int) -> QuantLinearWeight:
        w = (torch.randn((k_dim, n_dim), generator=gen, dtype=torch.float32) * 0.015).to(torch.float16)
        qw, s, z = quantize_4bit_group(w, group_size=config.group_size)
        return QuantLinearWeight(qw.to(dev), s.to(dev), z.to(dev), group_size=config.group_size)

    h = config.hidden_size
    kv_dim = config.num_key_value_heads * config.head_dim
    q_dim = config.num_attention_heads * config.head_dim
    inter = config.intermediate_size

    layers: List[MistralLayerWeights] = []
    for idx in range(config.num_hidden_layers):
        layers.append(
            MistralLayerWeights(
                layer_idx=idx,
                input_layernorm=torch.ones(h, device=dev, dtype=torch.float16),
                post_attention_layernorm=torch.ones(h, device=dev, dtype=torch.float16),
                q_proj=_rand_quant(h, q_dim),
                k_proj=_rand_quant(h, kv_dim),
                v_proj=_rand_quant(h, kv_dim),
                o_proj=_rand_quant(q_dim, h),
                gate_proj=_rand_quant(h, inter),
                up_proj=_rand_quant(h, inter),
                down_proj=_rand_quant(inter, h),
            )
        )

    return LoadedModelWeights(
        config=config,
        embed_tokens=_rand_fp16((config.vocab_size, h), scale=0.05),
        norm=torch.ones(h, device=dev, dtype=torch.float16),
        lm_head=_rand_fp16((config.vocab_size, h), scale=0.05),
        layers=layers,
    )

