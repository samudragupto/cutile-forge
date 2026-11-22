"""Unified multi-format checkpoint auto-detection, unpacking, and hybrid VRAM planning.

Auto-detects model family + quantization format across:
- Unquantized `FP16` / `BF16` (single or multi-shard `.safetensors`)
- `GPTQ` (`desc_act=True/False`, `group_size=32/64/128/-1`)
- `AWQ` (4-bit interleaved `[K, N//8]` -> `[K//2, N]`)
- `GGUF` (`Q4_K_M`, `Q4_0`, `Q8_0` binary header parser + dequant/unpackers)
- `FP8` (`float8_e4m3fn` scalar/channel/block-scaled)
and integrates `HybridLayerBudgeter` automatically when footprint > `0.85 * vram_gb`.
"""

from __future__ import annotations

import json
import struct
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import Any

import torch
from safetensors import safe_open

from cutile_forge.engine.weight_loader import QuantLinearWeight, _ShardReader
from cutile_forge.forge.weight_streamer import HybridLayerBudgeter
from cutile_forge.models.model_graph import UnifiedModelGraph
from cutile_forge.models.registry import from_hf_config
from cutile_forge.utils.quantize import (
    unpack_awq_to_cutile,
    unpack_fp8_e4m3_to_fp16,
    unpack_gguf_q4_0_to_cutile,
    unpack_gguf_q4_k_to_cutile,
    unpack_gguf_q8_0_to_fp16,
    unpack_gptq_layer,
)

# GGUF value type codes (GGUF spec v2/v3)
_GGUF_TYPE_UINT8 = 0
_GGUF_TYPE_INT8 = 1
_GGUF_TYPE_UINT16 = 2
_GGUF_TYPE_INT16 = 3
_GGUF_TYPE_UINT32 = 4
_GGUF_TYPE_INT32 = 5
_GGUF_TYPE_FLOAT32 = 6
_GGUF_TYPE_BOOL = 7
_GGUF_TYPE_STRING = 8
_GGUF_TYPE_ARRAY = 9
_GGUF_TYPE_UINT64 = 10
_GGUF_TYPE_INT64 = 11
_GGUF_TYPE_FLOAT64 = 12

# GGML tensor type IDs
GGML_TYPE_F32 = 0
GGML_TYPE_F16 = 1
GGML_TYPE_Q4_0 = 2
GGML_TYPE_Q8_0 = 8
GGML_TYPE_Q4_K = 12
GGML_TYPE_BF16 = 30


class CheckpointFormat(StrEnum):
    """Supported weight storage and quantization formats."""

    FP16 = "fp16"
    BF16 = "bf16"
    GPTQ = "gptq"
    AWQ = "awq"
    GGUF_Q4_K_M = "gguf_q4_k_m"
    GGUF_Q4_0 = "gguf_q4_0"
    GGUF_Q8_0 = "gguf_q8_0"
    FP8 = "fp8"


@dataclass
class HybridPlacementPlan:
    """Automatic GPU/DDR5 layer residency placement computed by `HybridLayerBudgeter`."""

    total_layers: int
    layer_bytes: int
    estimated_checkpoint_gb: float
    vram_gb: float
    safety_threshold_gb: float
    use_hybrid_budgeter: bool
    resident_gpu_layers: int
    streamed_cpu_layers: int


@dataclass
class CheckpointInspection:
    """Metadata and format inspection result for a checkpoint directory or `.gguf` file."""

    path: Path
    family: str
    quant_format: CheckpointFormat
    config: dict[str, Any]
    shard_files: list[Path]
    estimated_bytes: int
    group_size: int = 128
    desc_act: bool = False
    gguf_metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def estimated_gb(self) -> float:
        return self.estimated_bytes / (1024.0**3)


@dataclass
class UnifiedCheckpoint:
    """Loaded checkpoint package containing `UnifiedModelGraph`, placement plan, and unpacked weights."""

    inspection: CheckpointInspection
    graph: UnifiedModelGraph
    placement: HybridPlacementPlan
    tensors: dict[str, torch.Tensor | QuantLinearWeight]


# ---------------------------------------------------------------------------
# GGUF v2/v3 Binary Header Reader & Synthetic Writer (Tests/Roundtrip)
# ---------------------------------------------------------------------------


def _read_gguf_string(f: Any) -> str:
    (length,) = struct.unpack("<Q", f.read(8))
    return str(f.read(length).decode("utf-8"))


def _read_gguf_value(f: Any, vtype: int) -> Any:
    if vtype == _GGUF_TYPE_UINT8:
        return struct.unpack("<B", f.read(1))[0]
    if vtype == _GGUF_TYPE_INT8:
        return struct.unpack("<b", f.read(1))[0]
    if vtype == _GGUF_TYPE_UINT16:
        return struct.unpack("<H", f.read(2))[0]
    if vtype == _GGUF_TYPE_INT16:
        return struct.unpack("<h", f.read(2))[0]
    if vtype == _GGUF_TYPE_UINT32:
        return struct.unpack("<I", f.read(4))[0]
    if vtype == _GGUF_TYPE_INT32:
        return struct.unpack("<i", f.read(4))[0]
    if vtype == _GGUF_TYPE_FLOAT32:
        return struct.unpack("<f", f.read(4))[0]
    if vtype == _GGUF_TYPE_BOOL:
        return bool(struct.unpack("<B", f.read(1))[0])
    if vtype == _GGUF_TYPE_STRING:
        return _read_gguf_string(f)
    if vtype == _GGUF_TYPE_ARRAY:
        elem_type, count = struct.unpack("<IQ", f.read(12))
        return [_read_gguf_value(f, elem_type) for _ in range(count)]
    if vtype == _GGUF_TYPE_UINT64:
        return struct.unpack("<Q", f.read(8))[0]
    if vtype == _GGUF_TYPE_INT64:
        return struct.unpack("<q", f.read(8))[0]
    if vtype == _GGUF_TYPE_FLOAT64:
        return struct.unpack("<d", f.read(8))[0]
    raise ValueError(f"Unsupported GGUF metadata value type: {vtype}")


def parse_gguf_header(path: str | Path) -> dict[str, Any]:
    """Parse a GGUF v2/v3 file header returning metadata KV dict, tensors list, and `data_offset`."""
    gguf_path = Path(path)
    with gguf_path.open("rb") as f:
        magic = f.read(4)
        if magic != b"GGUF":
            raise ValueError(f"Invalid GGUF magic {magic!r} in {gguf_path}")
        version, tensor_count, kv_count = struct.unpack("<IQQ", f.read(20))
        if version not in (2, 3):
            raise ValueError(f"Unsupported GGUF version {version} in {gguf_path}")

        metadata: dict[str, Any] = {}
        for _ in range(kv_count):
            key = _read_gguf_string(f)
            (vtype,) = struct.unpack("<I", f.read(4))
            metadata[key] = _read_gguf_value(f, vtype)

        tensors: list[dict[str, Any]] = []
        for _ in range(tensor_count):
            name = _read_gguf_string(f)
            (n_dims,) = struct.unpack("<I", f.read(4))
            dims = tuple(struct.unpack(f"<{n_dims}Q", f.read(8 * n_dims)))
            ggml_type, offset = struct.unpack("<IQ", f.read(12))
            tensors.append({"name": name, "dims": dims, "ggml_type": ggml_type, "offset": offset})

        alignment = int(metadata.get("general.alignment", 32))
        cur = f.tell()
        rem = cur % alignment
        data_offset = cur + ((alignment - rem) if rem else 0)

    return {
        "version": version,
        "metadata": metadata,
        "tensors": tensors,
        "data_offset": data_offset,
    }


def _pack_gguf_string(val: str) -> bytes:
    enc = val.encode("utf-8")
    return struct.pack("<Q", len(enc)) + enc


def _pack_gguf_kv(key: str, val: Any) -> bytes:
    out = bytearray(_pack_gguf_string(key))
    if isinstance(val, bool):
        out += struct.pack("<IB", _GGUF_TYPE_BOOL, 1 if val else 0)
    elif isinstance(val, int):
        out += struct.pack("<Ii", _GGUF_TYPE_INT32, val)
    elif isinstance(val, float):
        out += struct.pack("<If", _GGUF_TYPE_FLOAT32, val)
    elif isinstance(val, str):
        out += struct.pack("<I", _GGUF_TYPE_STRING)
        out += _pack_gguf_string(val)
    else:
        raise TypeError(f"Unsupported synthetic GGUF metadata type for '{key}': {type(val)}")
    return bytes(out)


def write_synthetic_gguf(
    path: str | Path,
    metadata: dict[str, Any],
    tensors: list[tuple[str, tuple[int, ...], int, bytes]],
    alignment: int = 32,
) -> Path:
    """Write a minimal spec-compliant GGUF v3 binary file for deterministic roundtrip tests."""
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    header = bytearray()
    header += b"GGUF"
    header += struct.pack("<IQQ", 3, len(tensors), len(metadata))
    for k, v in metadata.items():
        header += _pack_gguf_kv(k, v)

    data_chunks = bytearray()
    for name, dims, ggml_type, payload in tensors:
        rem = len(data_chunks) % alignment
        if rem:
            data_chunks += b"\x00" * (alignment - rem)
        offset = len(data_chunks)
        header += _pack_gguf_string(name)
        header += struct.pack(f"<I{len(dims)}QIQ", len(dims), *dims, ggml_type, offset)
        data_chunks += payload

    rem = len(header) % alignment
    if rem:
        header += b"\x00" * (alignment - rem)

    out_path.write_bytes(bytes(header) + bytes(data_chunks))
    return out_path


# ---------------------------------------------------------------------------
# Auto-Detection & Placement Planning
# ---------------------------------------------------------------------------


def _collect_safetensors(base: Path) -> list[Path]:
    shards = sorted(base.glob("*.safetensors"))
    if shards:
        return shards
    idx_file = base / "model.safetensors.index.json"
    if idx_file.exists():
        weight_map = json.loads(idx_file.read_text(encoding="utf-8")).get("weight_map", {})
        return sorted({base / fname for fname in weight_map.values() if (base / fname).exists()})
    return []


def _estimate_config_bytes(cfg: dict[str, Any], quant_format: CheckpointFormat) -> int:
    hidden = int(cfg.get("hidden_size", 4096))
    inter = int(cfg.get("intermediate_size", 14336))
    n_layers = int(cfg.get("num_hidden_layers", 32))
    n_heads = int(cfg.get("num_attention_heads", 32))
    n_kv = int(cfg.get("num_key_value_heads", n_heads))
    head_dim = int(cfg.get("head_dim", hidden // max(1, n_heads)))
    experts = max(1, int(cfg.get("num_local_experts", 1)))
    vocab = int(cfg.get("vocab_size", 32000))

    attn_p = (
        hidden * (n_heads * head_dim)
        + 2 * hidden * (n_kv * head_dim)
        + (n_heads * head_dim) * hidden
    )
    ffn_p = 3 * hidden * inter * experts
    total_layer_p = n_layers * (attn_p + ffn_p)
    embed_p = 2 * vocab * hidden

    if quant_format in (
        CheckpointFormat.GPTQ,
        CheckpointFormat.AWQ,
        CheckpointFormat.GGUF_Q4_K_M,
        CheckpointFormat.GGUF_Q4_0,
    ):
        bytes_per_param = 0.5625
    elif quant_format in (CheckpointFormat.FP8, CheckpointFormat.GGUF_Q8_0):
        bytes_per_param = 1.05
    else:
        bytes_per_param = 2.0
    return int(total_layer_p * bytes_per_param + embed_p * 2)


def detect_checkpoint_format(checkpoint_path: str | Path) -> CheckpointInspection:
    """Inspect a `.gguf` file or HF checkpoint directory to detect family & weight format."""
    target = Path(checkpoint_path)
    if target.is_file() and target.suffix.lower() == ".gguf":
        gguf = parse_gguf_header(target)
        meta = gguf["metadata"]
        arch = str(meta.get("general.architecture", "mistral")).lower()
        t_types = {int(t["ggml_type"]) for t in gguf["tensors"]}
        if GGML_TYPE_Q4_K in t_types:
            qfmt = CheckpointFormat.GGUF_Q4_K_M
        elif GGML_TYPE_Q4_0 in t_types:
            qfmt = CheckpointFormat.GGUF_Q4_0
        elif GGML_TYPE_Q8_0 in t_types:
            qfmt = CheckpointFormat.GGUF_Q8_0
        else:
            qfmt = CheckpointFormat.FP16

        cfg: dict[str, Any] = {
            "model_type": arch,
            "hidden_size": int(meta.get(f"{arch}.embedding_length", 128)),
            "intermediate_size": int(meta.get(f"{arch}.feed_forward_length", 256)),
            "num_hidden_layers": int(meta.get(f"{arch}.block_count", 2)),
            "num_attention_heads": int(meta.get(f"{arch}.attention.head_count", 4)),
            "num_key_value_heads": int(meta.get(f"{arch}.attention.head_count_kv", 2)),
            "vocab_size": int(meta.get(f"{arch}.vocab_size", 512)),
            "rope_theta": float(meta.get(f"{arch}.rope.freq_base", 10000.0)),
        }
        return CheckpointInspection(
            path=target,
            family=arch,
            quant_format=qfmt,
            config=cfg,
            shard_files=[target],
            estimated_bytes=target.stat().st_size,
            gguf_metadata=gguf,
        )

    if target.is_dir():
        gguf_files = sorted(target.glob("*.gguf"))
        if gguf_files:
            return detect_checkpoint_format(gguf_files[0])

    cfg_path = target / "config.json"
    cfg = json.loads(cfg_path.read_text(encoding="utf-8")) if cfg_path.exists() else {}
    qcfg_path = target / "quantize_config.json"
    qcfg: dict[str, Any] = (
        json.loads(qcfg_path.read_text(encoding="utf-8")) if qcfg_path.exists() else {}
    )
    if not qcfg and isinstance(cfg.get("quantization_config"), dict):
        qcfg = cfg["quantization_config"]

    shards = _collect_safetensors(target)
    quant_method = str(qcfg.get("quant_method", "")).lower()
    group_size = int(qcfg.get("group_size", 128))
    desc_act = bool(qcfg.get("desc_act", False))

    det_fmt: CheckpointFormat | None = None
    if quant_method == "gptq":
        det_fmt = CheckpointFormat.GPTQ
    elif quant_method == "awq":
        det_fmt = CheckpointFormat.AWQ
    elif quant_method == "fp8":
        det_fmt = CheckpointFormat.FP8

    fp16_seen = False
    bf16_seen = False
    fp8_seen = False
    gptq_keys = False
    awq_keys = False

    if shards:
        with safe_open(str(shards[0]), framework="pt", device="cpu") as f:
            keys = list(f.keys())
            for k in keys:
                if k.endswith(".qzeros"):
                    gptq_keys = True
                if k.endswith("weight_scale_inv"):
                    fp8_seen = True
                sl = f.get_slice(k)
                dt_str = str(sl.get_dtype()).lower()
                if "bf16" in dt_str or "bfloat16" in dt_str:
                    bf16_seen = True
                elif "f16" in dt_str or "float16" in dt_str:
                    fp16_seen = True
                elif "f8" in dt_str or "float8" in dt_str:
                    fp8_seen = True
                if det_fmt is None and k.endswith(".qweight"):
                    shape = sl.get_shape()
                    if len(shape) == 2 and not gptq_keys:
                        awq_keys = True

    if det_fmt is None:
        if fp8_seen:
            det_fmt = CheckpointFormat.FP8
        elif gptq_keys and quant_method != "awq":
            det_fmt = CheckpointFormat.GPTQ
        elif awq_keys and quant_method == "awq":
            det_fmt = CheckpointFormat.AWQ
        elif bf16_seen and not fp16_seen:
            det_fmt = CheckpointFormat.BF16
        else:
            det_fmt = CheckpointFormat.FP16

    disk_bytes = sum(s.stat().st_size for s in shards) if shards else 0
    estimated_bytes = disk_bytes if disk_bytes > 0 else _estimate_config_bytes(cfg, det_fmt)

    family = str(cfg.get("model_type", "mistral")).lower().replace("-", "_")
    if family == "llama" and (
        cfg.get("rope_scaling") or float(cfg.get("rope_theta", 10000.0)) >= 100000.0
    ):
        family = "llama3"

    return CheckpointInspection(
        path=target,
        family=family,
        quant_format=det_fmt,
        config=cfg,
        shard_files=shards,
        estimated_bytes=estimated_bytes,
        group_size=group_size,
        desc_act=desc_act,
    )


def plan_checkpoint_placement(
    inspection: CheckpointInspection,
    vram_gb: float = 32.0,
    safety_fraction: float = 0.85,
    overhead_gb: float = 2.5,
) -> HybridPlacementPlan:
    """Plan GPU-resident vs. Stage -1 PCIe-streamed layers using `HybridLayerBudgeter`."""
    total_layers = max(1, int(inspection.config.get("num_hidden_layers", 32)))
    est_gb = inspection.estimated_gb
    threshold_gb = vram_gb * safety_fraction
    layer_bytes = max(1, inspection.estimated_bytes // total_layers)

    if est_gb <= threshold_gb:
        return HybridPlacementPlan(
            total_layers=total_layers,
            layer_bytes=layer_bytes,
            estimated_checkpoint_gb=round(est_gb, 3),
            vram_gb=vram_gb,
            safety_threshold_gb=round(threshold_gb, 3),
            use_hybrid_budgeter=False,
            resident_gpu_layers=total_layers,
            streamed_cpu_layers=0,
        )

    budgeter = HybridLayerBudgeter(
        total_layers=total_layers,
        layer_bytes=layer_bytes,
        vram_budget_gb=vram_gb,
        reserve_slots=2,
        overhead_gb=overhead_gb,
    )
    return HybridPlacementPlan(
        total_layers=total_layers,
        layer_bytes=layer_bytes,
        estimated_checkpoint_gb=round(est_gb, 3),
        vram_gb=vram_gb,
        safety_threshold_gb=round(threshold_gb, 3),
        use_hybrid_budgeter=True,
        resident_gpu_layers=budgeter.n_resident,
        streamed_cpu_layers=budgeter.n_streamed,
    )


def _gguf_payload_bytes(dims: tuple[int, ...], ggml_type: int) -> int:
    numel = 1
    for d in dims:
        numel *= int(d)
    if ggml_type == GGML_TYPE_Q4_K:
        return (numel // 256) * 144
    if ggml_type == GGML_TYPE_Q4_0:
        return (numel // 32) * 18
    if ggml_type == GGML_TYPE_Q8_0:
        return (numel // 32) * 34
    if ggml_type in (GGML_TYPE_F16, GGML_TYPE_BF16):
        return numel * 2
    return numel * 4


def load_unified_checkpoint(
    checkpoint_path: str | Path,
    device: str | torch.device = "cpu",
    vram_gb: float | None = None,
    load_tensors: bool = True,
) -> UnifiedCheckpoint:
    """Auto-detect architecture + quantization format, compute hybrid placement, and unpack weights."""
    insp = detect_checkpoint_format(checkpoint_path)
    graph = from_hf_config(insp.config, name=insp.family)

    target_dev = torch.device(
        device if (str(device) != "cuda" or torch.cuda.is_available()) else "cpu"
    )
    if vram_gb is None:
        if target_dev.type == "cuda" and torch.cuda.is_available():
            eff_vram_gb = torch.cuda.get_device_properties(target_dev).total_memory / (1024.0**3)
        else:
            eff_vram_gb = 32.0
    else:
        eff_vram_gb = vram_gb

    placement = plan_checkpoint_placement(insp, vram_gb=eff_vram_gb)
    unpacked: dict[str, torch.Tensor | QuantLinearWeight] = {}

    if not load_tensors:
        return UnifiedCheckpoint(
            inspection=insp, graph=graph, placement=placement, tensors=unpacked
        )

    if insp.quant_format in (
        CheckpointFormat.GGUF_Q4_K_M,
        CheckpointFormat.GGUF_Q4_0,
        CheckpointFormat.GGUF_Q8_0,
    ):
        gguf = insp.gguf_metadata
        data_offset = int(gguf["data_offset"])
        with insp.path.open("rb") as f:
            for tinfo in gguf["tensors"]:
                tname = str(tinfo["name"])
                dims = tuple(int(x) for x in tinfo["dims"])
                gtype = int(tinfo["ggml_type"])
                nbytes = _gguf_payload_bytes(dims, gtype)
                f.seek(data_offset + int(tinfo["offset"]))
                raw_bytes = f.read(nbytes)
                raw_u8 = torch.frombuffer(bytearray(raw_bytes), dtype=torch.uint8).clone()

                if gtype == GGML_TYPE_Q4_K and len(dims) == 2:
                    k_dim, n_dim = dims
                    qw, sc, zr = unpack_gguf_q4_k_to_cutile(raw_u8, k=k_dim, n=n_dim)
                    unpacked[tname] = QuantLinearWeight(
                        qw.to(target_dev), sc.to(target_dev), zr.to(target_dev), group_size=128
                    )
                elif gtype == GGML_TYPE_Q4_0 and len(dims) == 2:
                    k_dim, n_dim = dims
                    gs = 128 if k_dim % 128 == 0 else (32 if k_dim % 32 == 0 else k_dim)
                    qw, sc, zr = unpack_gguf_q4_0_to_cutile(raw_u8, k=k_dim, n=n_dim, group_size=gs)
                    unpacked[tname] = QuantLinearWeight(
                        qw.to(target_dev), sc.to(target_dev), zr.to(target_dev), group_size=gs
                    )
                elif gtype == GGML_TYPE_Q8_0 and len(dims) == 2:
                    k_dim, n_dim = dims
                    unpacked[tname] = unpack_gguf_q8_0_to_fp16(raw_u8, k=k_dim, n=n_dim).to(
                        target_dev
                    )
                elif gtype == GGML_TYPE_F16:
                    unpacked[tname] = (
                        torch.frombuffer(bytearray(raw_bytes), dtype=torch.float16)
                        .clone()
                        .view(*dims)
                        .to(target_dev)
                    )
                else:
                    unpacked[tname] = (
                        torch.frombuffer(bytearray(raw_bytes), dtype=torch.float32)
                        .clone()
                        .view(*dims)
                        .to(dtype=torch.float16, device=target_dev)
                    )
        return UnifiedCheckpoint(
            inspection=insp, graph=graph, placement=placement, tensors=unpacked
        )

    reader = _ShardReader(insp.shard_files)
    all_keys = sorted(reader._key_to_handle.keys())
    handled: set[str] = set()

    for key in all_keys:
        if key in handled:
            continue
        if key.endswith(".qweight"):
            prefix = key[: -len(".qweight")]
            qw_raw = reader[f"{prefix}.qweight"]
            qz_raw = reader[f"{prefix}.qzeros"]
            sc_raw = reader[f"{prefix}.scales"]
            g_idx = reader.get(f"{prefix}.g_idx")
            handled.update(
                k
                for k in (
                    f"{prefix}.qweight",
                    f"{prefix}.qzeros",
                    f"{prefix}.scales",
                    f"{prefix}.g_idx",
                )
                if k in reader
            )
            if insp.quant_format == CheckpointFormat.AWQ:
                qw, sc, zr = unpack_awq_to_cutile(qw_raw, qz_raw, sc_raw)
                perm = None
                eff_gs = max(1, (qw.shape[0] * 2) // max(1, sc.shape[0]))
            else:
                k_dim = qw_raw.shape[0] * 8
                eff_gs = k_dim if insp.group_size <= 0 else insp.group_size
                qw, sc, zr, perm = unpack_gptq_layer(
                    qw_raw, qz_raw, sc_raw, group_size=eff_gs, g_idx=g_idx
                )
            unpacked[prefix] = QuantLinearWeight(
                qweight=qw.to(target_dev),
                scales=sc.to(target_dev),
                zeros=zr.to(target_dev),
                group_size=eff_gs,
                perm=perm.to(target_dev) if perm is not None else None,
            )
        elif key.endswith(".weight_scale_inv"):
            continue
        elif f"{key}_scale_inv" in reader:
            w_fp8 = reader[key]
            sc_inv = reader[f"{key}_scale_inv"]
            handled.add(key)
            handled.add(f"{key}_scale_inv")
            unpacked[key] = unpack_fp8_e4m3_to_fp16(w_fp8, sc_inv).to(target_dev)
        else:
            handled.add(key)
            t = reader[key]
            if t.dtype in (torch.float32, torch.bfloat16, torch.float16):
                unpacked[key] = t.to(dtype=torch.float16, device=target_dev)
            else:
                unpacked[key] = t.to(target_dev)

    return UnifiedCheckpoint(inspection=insp, graph=graph, placement=placement, tensors=unpacked)


def extract_hf_decoder_weights(
    hf_model: Any,
    num_layers: int | None = None,
    device: torch.device | str = "cuda",
    dtype: torch.dtype = torch.float16,
) -> tuple[UnifiedModelGraph, dict[str, Any]]:
    """Convert an instantiated HuggingFace `AutoModelForCausalLM` (Llama/Mistral/Qwen2/Mixtral) into `(graph, weights)`."""
    dev = torch.device(device if (str(device) != "cuda" or torch.cuda.is_available()) else "cpu")
    dt = dtype if dev.type == "cuda" else torch.float32
    cfg_dict = hf_model.config.to_dict()
    rp = cfg_dict.get("rope_parameters")
    if isinstance(rp, dict):
        if cfg_dict.get("rope_theta") is None and rp.get("rope_theta") is not None:
            cfg_dict["rope_theta"] = rp["rope_theta"]
        if cfg_dict.get("rope_scaling") is None and rp.get("rope_type") not in (None, "default"):
            cfg_dict["rope_scaling"] = dict(rp)
    if num_layers is not None:
        cfg_dict["num_hidden_layers"] = num_layers
    graph = from_hf_config(cfg_dict)
    graph.metadata["rotary_style"] = "half"

    base = getattr(hf_model, "model", hf_model)
    weights: dict[str, Any] = {
        "embed_tokens": base.embed_tokens.weight.detach().to(device=dev, dtype=dt).contiguous(),
        "norm_w": base.norm.weight.detach().to(device=dev, dtype=dt).contiguous(),
        "lm_head": hf_model.lm_head.weight.detach().t().to(device=dev, dtype=dt).contiguous(),
        "blocks": [],
    }
    n_l = num_layers if num_layers is not None else len(base.layers)
    for idx in range(n_l):
        lyr = base.layers[idx]
        attn = lyr.self_attn
        bw: dict[str, Any] = {
            "in_norm_w": lyr.input_layernorm.weight.detach().to(device=dev, dtype=dt).contiguous(),
            "post_norm_w": lyr.post_attention_layernorm.weight.detach()
            .to(device=dev, dtype=dt)
            .contiguous(),
            "wq": attn.q_proj.weight.detach().t().to(device=dev, dtype=dt).contiguous(),
            "wk": attn.k_proj.weight.detach().t().to(device=dev, dtype=dt).contiguous(),
            "wv": attn.v_proj.weight.detach().t().to(device=dev, dtype=dt).contiguous(),
            "wo": attn.o_proj.weight.detach().t().to(device=dev, dtype=dt).contiguous(),
        }
        if getattr(attn.q_proj, "bias", None) is not None:
            bw["bq"] = attn.q_proj.bias.detach().to(device=dev, dtype=dt).contiguous()
            bw["bk"] = attn.k_proj.bias.detach().to(device=dev, dtype=dt).contiguous()
            bw["bv"] = attn.v_proj.bias.detach().to(device=dev, dtype=dt).contiguous()

        moe_mod = getattr(lyr, "block_sparse_moe", None)
        if moe_mod is None and hasattr(getattr(lyr, "mlp", None), "experts"):
            moe_mod = lyr.mlp
        if moe_mod is not None:
            bw["router"] = moe_mod.gate.weight.detach().t().to(device=dev, dtype=dt).contiguous()
            exps = moe_mod.experts
            if hasattr(exps, "gate_up_proj") and hasattr(exps, "down_proj"):
                gu_all = exps.gate_up_proj.detach()
                dn_all = exps.down_proj.detach()
                n_exp = gu_all.shape[0]
                inter_sz = gu_all.shape[1] // 2
                bw["experts"] = [
                    {
                        "w_gate": gu_all[e, :inter_sz, :].t().to(device=dev, dtype=dt).contiguous(),
                        "w_up": gu_all[e, inter_sz:, :].t().to(device=dev, dtype=dt).contiguous(),
                        "w_down": dn_all[e].t().to(device=dev, dtype=dt).contiguous(),
                    }
                    for e in range(n_exp)
                ]
            else:
                bw["experts"] = [
                    {
                        "w_gate": (exp.w1 if hasattr(exp, "w1") else exp.gate_proj)
                        .weight.detach()
                        .t()
                        .to(device=dev, dtype=dt)
                        .contiguous(),
                        "w_up": (exp.w3 if hasattr(exp, "w3") else exp.up_proj)
                        .weight.detach()
                        .t()
                        .to(device=dev, dtype=dt)
                        .contiguous(),
                        "w_down": (exp.w2 if hasattr(exp, "w2") else exp.down_proj)
                        .weight.detach()
                        .t()
                        .to(device=dev, dtype=dt)
                        .contiguous(),
                    }
                    for exp in exps
                ]
        else:
            mlp = lyr.mlp
            bw["w_gate"] = mlp.gate_proj.weight.detach().t().to(device=dev, dtype=dt).contiguous()
            bw["w_up"] = mlp.up_proj.weight.detach().t().to(device=dev, dtype=dt).contiguous()
            bw["w_down"] = mlp.down_proj.weight.detach().t().to(device=dev, dtype=dt).contiguous()
        weights["blocks"].append(bw)
    return graph, weights


def extract_hf_siglip_vision_weights(
    hf_vision: Any,
    device: torch.device | str = "cuda",
    dtype: torch.dtype = torch.float16,
) -> tuple[UnifiedModelGraph, dict[str, Any]]:
    """Convert a HuggingFace `SiglipVisionModel` into `(UnifiedModelGraph, weights)`."""
    dev = torch.device(device if (str(device) != "cuda" or torch.cuda.is_available()) else "cpu")
    dt = dtype if dev.type == "cuda" else torch.float32
    vm = getattr(hf_vision, "vision_model", hf_vision)
    cfg = vm.config
    d = int(cfg.hidden_size)
    cfg_dict = {
        "model_type": "siglip_vision_model",
        "image_size": int(cfg.image_size),
        "patch_size": int(cfg.patch_size),
        "in_channels": int(getattr(cfg, "num_channels", 3)),
        "hidden_size": d,
        "intermediate_size": int(cfg.intermediate_size),
        "num_hidden_layers": int(cfg.num_hidden_layers),
        "num_attention_heads": int(cfg.num_attention_heads),
        "num_labels": d,
        "layer_norm_eps": float(cfg.layer_norm_eps),
        "use_cls_token": False,
    }
    graph = from_hf_config(cfg_dict, name="siglip_vision")
    from cutile_forge.models.model_graph import EncoderOutputStage

    graph.output_stage = EncoderOutputStage(hidden_size=d, layer_norm_eps=float(cfg.layer_norm_eps))

    emb = vm.embeddings
    pos_w = emb.position_embedding.weight.detach().to(device=dev, dtype=dt).unsqueeze(0)
    weights: dict[str, Any] = {
        "vision_embed": {
            "patch_w": emb.patch_embedding.weight.detach().to(device=dev, dtype=dt).contiguous(),
            "patch_b": emb.patch_embedding.bias.detach().to(device=dev, dtype=dt).contiguous(),
            "pos_embed": pos_w.contiguous(),
        },
        "blocks": [],
        "norm_w": vm.post_layernorm.weight.detach().to(device=dev, dtype=dt).contiguous(),
        "norm_b": vm.post_layernorm.bias.detach().to(device=dev, dtype=dt).contiguous(),
    }
    for lyr in vm.encoder.layers:
        attn = lyr.self_attn
        wqkv = torch.cat(
            [
                attn.q_proj.weight.detach().t(),
                attn.k_proj.weight.detach().t(),
                attn.v_proj.weight.detach().t(),
            ],
            dim=1,
        )
        bqkv = torch.cat(
            [
                attn.q_proj.bias.detach(),
                attn.k_proj.bias.detach(),
                attn.v_proj.bias.detach(),
            ],
            dim=0,
        )
        weights["blocks"].append(
            {
                "ln1_w": lyr.layer_norm1.weight.detach().to(device=dev, dtype=dt).contiguous(),
                "ln1_b": lyr.layer_norm1.bias.detach().to(device=dev, dtype=dt).contiguous(),
                "ln2_w": lyr.layer_norm2.weight.detach().to(device=dev, dtype=dt).contiguous(),
                "ln2_b": lyr.layer_norm2.bias.detach().to(device=dev, dtype=dt).contiguous(),
                "wqkv": wqkv.to(device=dev, dtype=dt).contiguous(),
                "bqkv": bqkv.to(device=dev, dtype=dt).contiguous(),
                "wo": attn.out_proj.weight.detach().t().to(device=dev, dtype=dt).contiguous(),
                "bo": attn.out_proj.bias.detach().to(device=dev, dtype=dt).contiguous(),
                "fc1_w": lyr.mlp.fc1.weight.detach().t().to(device=dev, dtype=dt).contiguous(),
                "fc1_b": lyr.mlp.fc1.bias.detach().to(device=dev, dtype=dt).contiguous(),
                "fc2_w": lyr.mlp.fc2.weight.detach().t().to(device=dev, dtype=dt).contiguous(),
                "fc2_b": lyr.mlp.fc2.bias.detach().to(device=dev, dtype=dt).contiguous(),
            }
        )
    return graph, weights


def extract_hf_whisper_encoder_weights(
    hf_whisper: Any,
    device: torch.device | str = "cuda",
    dtype: torch.dtype = torch.float16,
) -> tuple[UnifiedModelGraph, dict[str, Any]]:
    """Convert a HuggingFace `WhisperModel` encoder into `(UnifiedModelGraph, weights)`."""
    dev = torch.device(device if (str(device) != "cuda" or torch.cuda.is_available()) else "cpu")
    dt = dtype if dev.type == "cuda" else torch.float32
    enc = getattr(hf_whisper, "encoder", hf_whisper)
    cfg = hf_whisper.config
    d = int(cfg.d_model)
    cfg_dict = {
        "model_type": "whisper",
        "num_mel_bins": int(cfg.num_mel_bins),
        "max_source_positions": int(cfg.max_source_positions),
        "hidden_size": d,
        "intermediate_size": int(cfg.encoder_ffn_dim),
        "num_hidden_layers": int(cfg.encoder_layers),
        "num_attention_heads": int(cfg.encoder_attention_heads),
        "activation_function": str(getattr(cfg, "activation_function", "gelu")),
        "layer_norm_eps": 1e-5,
    }
    graph = from_hf_config(cfg_dict, name="whisper_encoder")
    weights: dict[str, Any] = {
        "audio_embed": {
            "conv1_w": enc.conv1.weight.detach().to(device=dev, dtype=dt).contiguous(),
            "conv1_b": enc.conv1.bias.detach().to(device=dev, dtype=dt).contiguous(),
            "conv2_w": enc.conv2.weight.detach().to(device=dev, dtype=dt).contiguous(),
            "conv2_b": enc.conv2.bias.detach().to(device=dev, dtype=dt).contiguous(),
            "pos_embed": enc.embed_positions.weight.detach().to(device=dev, dtype=dt).contiguous(),
        },
        "blocks": [],
        "norm_w": enc.layer_norm.weight.detach().to(device=dev, dtype=dt).contiguous(),
        "norm_b": enc.layer_norm.bias.detach().to(device=dev, dtype=dt).contiguous(),
    }
    for lyr in enc.layers:
        attn = lyr.self_attn
        k_bias = (
            attn.k_proj.bias.detach()
            if getattr(attn.k_proj, "bias", None) is not None
            else torch.zeros(d, dtype=attn.q_proj.bias.dtype, device=attn.q_proj.bias.device)
        )
        wqkv = torch.cat(
            [
                attn.q_proj.weight.detach().t(),
                attn.k_proj.weight.detach().t(),
                attn.v_proj.weight.detach().t(),
            ],
            dim=1,
        )
        bqkv = torch.cat(
            [
                attn.q_proj.bias.detach(),
                k_bias,
                attn.v_proj.bias.detach(),
            ],
            dim=0,
        )
        weights["blocks"].append(
            {
                "ln1_w": lyr.self_attn_layer_norm.weight.detach()
                .to(device=dev, dtype=dt)
                .contiguous(),
                "ln1_b": lyr.self_attn_layer_norm.bias.detach()
                .to(device=dev, dtype=dt)
                .contiguous(),
                "ln2_w": lyr.final_layer_norm.weight.detach().to(device=dev, dtype=dt).contiguous(),
                "ln2_b": lyr.final_layer_norm.bias.detach().to(device=dev, dtype=dt).contiguous(),
                "wqkv": wqkv.to(device=dev, dtype=dt).contiguous(),
                "bqkv": bqkv.to(device=dev, dtype=dt).contiguous(),
                "wo": attn.out_proj.weight.detach().t().to(device=dev, dtype=dt).contiguous(),
                "bo": attn.out_proj.bias.detach().to(device=dev, dtype=dt).contiguous(),
                "fc1_w": lyr.fc1.weight.detach().t().to(device=dev, dtype=dt).contiguous(),
                "fc1_b": lyr.fc1.bias.detach().to(device=dev, dtype=dt).contiguous(),
                "fc2_w": lyr.fc2.weight.detach().t().to(device=dev, dtype=dt).contiguous(),
                "fc2_b": lyr.fc2.bias.detach().to(device=dev, dtype=dt).contiguous(),
            }
        )
    return graph, weights


def extract_hf_vit_weights(
    hf_vit: Any,
    device: torch.device | str = "cuda",
    dtype: torch.dtype = torch.float16,
) -> tuple[UnifiedModelGraph, dict[str, Any]]:
    """Convert a HuggingFace `ViTModel` or `ViTForImageClassification` into `(graph, weights)`."""
    dev = torch.device(device if (str(device) != "cuda" or torch.cuda.is_available()) else "cpu")
    dt = dtype if dev.type == "cuda" else torch.float32
    vm = getattr(hf_vit, "vit", hf_vit)
    cfg = vm.config
    d = int(cfg.hidden_size)
    has_cls_head = hasattr(hf_vit, "classifier")
    num_labels = int(hf_vit.classifier.out_features) if has_cls_head else d
    cfg_dict = {
        "model_type": "vit",
        "image_size": int(cfg.image_size),
        "patch_size": int(cfg.patch_size),
        "in_channels": int(getattr(cfg, "num_channels", 3)),
        "hidden_size": d,
        "intermediate_size": int(cfg.intermediate_size),
        "num_hidden_layers": int(cfg.num_hidden_layers),
        "num_attention_heads": int(cfg.num_attention_heads),
        "num_labels": num_labels,
        "hidden_act": str(getattr(cfg, "hidden_act", "gelu")),
        "layer_norm_eps": float(cfg.layer_norm_eps),
        "use_cls_token": True,
    }
    graph = from_hf_config(cfg_dict, name="vit")
    from cutile_forge.models.model_graph import ClassifierHeadStage, EncoderOutputStage

    if has_cls_head:
        graph.output_stage = ClassifierHeadStage(
            hidden_size=d,
            num_classes=num_labels,
            layer_norm_eps=float(cfg.layer_norm_eps),
            pool_type="cls",
        )
    else:
        graph.output_stage = EncoderOutputStage(
            hidden_size=d, layer_norm_eps=float(cfg.layer_norm_eps)
        )

    emb = vm.embeddings
    proj = emb.patch_embeddings.projection
    weights: dict[str, Any] = {
        "vision_embed": {
            "patch_w": proj.weight.detach().to(device=dev, dtype=dt).contiguous(),
            "patch_b": (
                proj.bias.detach().to(device=dev, dtype=dt).contiguous()
                if proj.bias is not None
                else None
            ),
            "cls_token": emb.cls_token.detach().to(device=dev, dtype=dt).contiguous(),
            "pos_embed": emb.position_embeddings.detach().to(device=dev, dtype=dt).contiguous(),
        },
        "blocks": [],
        "norm_w": vm.layernorm.weight.detach().to(device=dev, dtype=dt).contiguous(),
        "norm_b": vm.layernorm.bias.detach().to(device=dev, dtype=dt).contiguous(),
    }
    if has_cls_head:
        weights["cls_w"] = (
            hf_vit.classifier.weight.detach().t().to(device=dev, dtype=dt).contiguous()
        )
        weights["cls_b"] = (
            hf_vit.classifier.bias.detach().to(device=dev, dtype=dt).contiguous()
            if hf_vit.classifier.bias is not None
            else None
        )

    layers = vm.layers if hasattr(vm, "layers") else vm.encoder.layer
    for lyr in layers:
        if hasattr(lyr.attention, "q_proj"):
            q_mod, k_mod, v_mod = lyr.attention.q_proj, lyr.attention.k_proj, lyr.attention.v_proj
            o_mod = lyr.attention.o_proj
            fc1_mod, fc2_mod = lyr.mlp.fc1, lyr.mlp.fc2
        else:
            sa = lyr.attention.attention
            q_mod, k_mod, v_mod = sa.query, sa.key, sa.value
            o_mod = lyr.attention.output.dense
            fc1_mod, fc2_mod = lyr.intermediate.dense, lyr.output.dense

        wqkv = torch.cat(
            [q_mod.weight.detach().t(), k_mod.weight.detach().t(), v_mod.weight.detach().t()],
            dim=1,
        )
        bqkv = torch.cat([q_mod.bias.detach(), k_mod.bias.detach(), v_mod.bias.detach()], dim=0)
        weights["blocks"].append(
            {
                "ln1_w": lyr.layernorm_before.weight.detach().to(device=dev, dtype=dt).contiguous(),
                "ln1_b": lyr.layernorm_before.bias.detach().to(device=dev, dtype=dt).contiguous(),
                "ln2_w": lyr.layernorm_after.weight.detach().to(device=dev, dtype=dt).contiguous(),
                "ln2_b": lyr.layernorm_after.bias.detach().to(device=dev, dtype=dt).contiguous(),
                "wqkv": wqkv.to(device=dev, dtype=dt).contiguous(),
                "bqkv": bqkv.to(device=dev, dtype=dt).contiguous(),
                "wo": o_mod.weight.detach().t().to(device=dev, dtype=dt).contiguous(),
                "bo": o_mod.bias.detach().to(device=dev, dtype=dt).contiguous(),
                "fc1_w": fc1_mod.weight.detach().t().to(device=dev, dtype=dt).contiguous(),
                "fc1_b": fc1_mod.bias.detach().to(device=dev, dtype=dt).contiguous(),
                "fc2_w": fc2_mod.weight.detach().t().to(device=dev, dtype=dt).contiguous(),
                "fc2_b": fc2_mod.bias.detach().to(device=dev, dtype=dt).contiguous(),
            }
        )
    return graph, weights


def extract_hf_clip_vision_weights(
    hf_clip: Any,
    device: torch.device | str = "cuda",
    dtype: torch.dtype = torch.float16,
) -> tuple[UnifiedModelGraph, dict[str, Any]]:
    """Convert a HuggingFace `CLIPVisionModel` into `(UnifiedModelGraph, weights)`."""
    dev = torch.device(device if (str(device) != "cuda" or torch.cuda.is_available()) else "cpu")
    dt = dtype if dev.type == "cuda" else torch.float32
    vm = getattr(hf_clip, "vision_model", hf_clip)
    cfg = vm.config
    d = int(cfg.hidden_size)
    cfg_dict = {
        "model_type": "clip_vision_model",
        "image_size": int(cfg.image_size),
        "patch_size": int(cfg.patch_size),
        "in_channels": int(getattr(cfg, "num_channels", 3)),
        "hidden_size": d,
        "intermediate_size": int(cfg.intermediate_size),
        "num_hidden_layers": int(cfg.num_hidden_layers),
        "num_attention_heads": int(cfg.num_attention_heads),
        "num_labels": d,
        "hidden_act": str(getattr(cfg, "hidden_act", "quick_gelu")),
        "layer_norm_eps": float(cfg.layer_norm_eps),
        "use_cls_token": True,
        "pre_layer_norm": True,
    }
    graph = from_hf_config(cfg_dict, name="clip_vision")
    from cutile_forge.models.model_graph import ClassifierHeadStage

    graph.output_stage = ClassifierHeadStage(
        hidden_size=d, num_classes=d, layer_norm_eps=float(cfg.layer_norm_eps), pool_type="cls"
    )

    emb = vm.embeddings
    weights: dict[str, Any] = {
        "vision_embed": {
            "patch_w": emb.patch_embedding.weight.detach().to(device=dev, dtype=dt).contiguous(),
            "patch_b": (
                emb.patch_embedding.bias.detach().to(device=dev, dtype=dt).contiguous()
                if getattr(emb.patch_embedding, "bias", None) is not None
                else None
            ),
            "cls_token": emb.class_embedding.detach()
            .view(1, 1, d)
            .to(device=dev, dtype=dt)
            .contiguous(),
            "pos_embed": emb.position_embedding.weight.detach()
            .unsqueeze(0)
            .to(device=dev, dtype=dt)
            .contiguous(),
            "pre_ln_w": vm.pre_layrnorm.weight.detach().to(device=dev, dtype=dt).contiguous(),
            "pre_ln_b": vm.pre_layrnorm.bias.detach().to(device=dev, dtype=dt).contiguous(),
        },
        "blocks": [],
        "norm_w": vm.post_layernorm.weight.detach().to(device=dev, dtype=dt).contiguous(),
        "norm_b": vm.post_layernorm.bias.detach().to(device=dev, dtype=dt).contiguous(),
    }
    for lyr in vm.encoder.layers:
        attn = lyr.self_attn
        wqkv = torch.cat(
            [
                attn.q_proj.weight.detach().t(),
                attn.k_proj.weight.detach().t(),
                attn.v_proj.weight.detach().t(),
            ],
            dim=1,
        )
        bqkv = torch.cat(
            [attn.q_proj.bias.detach(), attn.k_proj.bias.detach(), attn.v_proj.bias.detach()], dim=0
        )
        weights["blocks"].append(
            {
                "ln1_w": lyr.layer_norm1.weight.detach().to(device=dev, dtype=dt).contiguous(),
                "ln1_b": lyr.layer_norm1.bias.detach().to(device=dev, dtype=dt).contiguous(),
                "ln2_w": lyr.layer_norm2.weight.detach().to(device=dev, dtype=dt).contiguous(),
                "ln2_b": lyr.layer_norm2.bias.detach().to(device=dev, dtype=dt).contiguous(),
                "wqkv": wqkv.to(device=dev, dtype=dt).contiguous(),
                "bqkv": bqkv.to(device=dev, dtype=dt).contiguous(),
                "wo": attn.out_proj.weight.detach().t().to(device=dev, dtype=dt).contiguous(),
                "bo": attn.out_proj.bias.detach().to(device=dev, dtype=dt).contiguous(),
                "fc1_w": lyr.mlp.fc1.weight.detach().t().to(device=dev, dtype=dt).contiguous(),
                "fc1_b": lyr.mlp.fc1.bias.detach().to(device=dev, dtype=dt).contiguous(),
                "fc2_w": lyr.mlp.fc2.weight.detach().t().to(device=dev, dtype=dt).contiguous(),
                "fc2_b": lyr.mlp.fc2.bias.detach().to(device=dev, dtype=dt).contiguous(),
            }
        )
    return graph, weights
