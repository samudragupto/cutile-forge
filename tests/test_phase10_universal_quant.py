"""Phase 10 verification: Universal Quantization Engine (GPTQ, AWQ, GGUF, INT8, FP8, mixed-precision).

Zero `pytest.skip` policy: every test executes end-to-end on both CUDA (`sm_120`) and CPU CI.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest
import torch

from cutile_forge.autotune.db import TuningDatabase
from cutile_forge.autotune.tuner import CostGuidedAutotuner
from cutile_forge.engine.config import GPUConfig, ModelConfig
from cutile_forge.engine.model import Mistral7B
from cutile_forge.engine.weight_loader import (
    create_random_4bit_weights,
    load_gptq_weights,
)
from cutile_forge.kernels.universal_quant import (
    MixedPrecisionFormatTable,
    QuantFormat,
    UniversalQuantLinear,
    UniversalQuantModel,
    dequant_gguf_q4_0_reference,
    dequant_gguf_q4_k_reference,
    dequant_gguf_q6_k_reference,
    dequant_gguf_q8_0_reference,
    encode_fp32_to_gguf_q4_0,
    encode_fp32_to_gguf_q4_k,
    encode_fp32_to_gguf_q6_k,
    encode_fp32_to_gguf_q8_0,
)

MISTRAL_7B_PATH = Path(
    os.environ.get("CUTILE_MISTRAL_7B_PATH", "/home/arrhat/models/mistral_7b_4bit")
)


def _device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _build_synthetic_mistral(device: torch.device) -> Mistral7B:
    cfg = ModelConfig(
        hidden_size=256,
        intermediate_size=512,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=64,
        vocab_size=256,
        max_position_embeddings=256,
        group_size=128,
    )
    weights = create_random_4bit_weights(cfg, device=str(device), seed=42)
    return Mistral7B(
        weights=weights,
        gpu_config=GPUConfig(vram_gb=24.0, stream_weights=False),
        device=str(device),
    )


def test_gguf_reference_dequant_bit_exact_against_gguf_package() -> None:
    """Verify CuTileForge GGUF dequantizers match `gguf.quants` bit-for-bit across Q4_0, Q8_0, Q4_K, Q6_K."""
    from gguf import quants

    torch.manual_seed(101)
    dev = _device()
    k, n = 256, 64
    w = torch.randn((k, n), device=dev, dtype=torch.float32) * 0.25

    # Q4_0
    raw_q4_0 = encode_fp32_to_gguf_q4_0(w)
    ref_q4_0 = dequant_gguf_q4_0_reference(raw_q4_0, k, n).cpu().numpy()
    gguf_q4_0 = (
        quants.Q4_0.dequantize_blocks(raw_q4_0.cpu().numpy()).reshape(n, k).T.astype(np.float32)
    )
    assert np.max(np.abs(ref_q4_0 - gguf_q4_0)) == 0.0

    # Q8_0
    raw_q8_0 = encode_fp32_to_gguf_q8_0(w)
    ref_q8_0 = dequant_gguf_q8_0_reference(raw_q8_0, k, n).cpu().numpy()
    gguf_q8_0 = (
        quants.Q8_0.dequantize_blocks(raw_q8_0.cpu().numpy()).reshape(n, k).T.astype(np.float32)
    )
    assert np.max(np.abs(ref_q8_0 - gguf_q8_0)) == 0.0

    # Q4_K
    raw_q4_k = encode_fp32_to_gguf_q4_k(w)
    ref_q4_k = dequant_gguf_q4_k_reference(raw_q4_k, k, n).cpu().numpy()
    gguf_q4_k = (
        quants.Q4_K.dequantize_blocks(raw_q4_k.cpu().numpy()).reshape(n, k).T.astype(np.float32)
    )
    assert np.max(np.abs(ref_q4_k - gguf_q4_k)) == 0.0

    # Q6_K
    raw_q6_k = encode_fp32_to_gguf_q6_k(w)
    ref_q6_k = dequant_gguf_q6_k_reference(raw_q6_k, k, n).cpu().numpy()
    gguf_q6_k = (
        quants.Q6_K.dequantize_blocks(raw_q6_k.cpu().numpy()).reshape(n, k).T.astype(np.float32)
    )
    assert np.max(np.abs(ref_q6_k - gguf_q6_k)) == 0.0


@pytest.mark.parametrize(
    "fmt,group_size,desc_act",
    [
        (QuantFormat.GPTQ, 128, False),
        (QuantFormat.GPTQ, 64, False),
        (QuantFormat.GPTQ, 32, False),
        (QuantFormat.GPTQ, -1, False),
        (QuantFormat.GPTQ, 128, True),
        (QuantFormat.AWQ, 128, False),
        (QuantFormat.AWQ, 64, False),
        (QuantFormat.GGUF_Q4_0, 32, False),
        (QuantFormat.GGUF_Q8_0, 32, False),
        (QuantFormat.GGUF_Q4_K, 32, False),
        (QuantFormat.GGUF_Q6_K, 16, False),
        (QuantFormat.INT8, 128, False),
        (QuantFormat.INT8, 64, False),
        (QuantFormat.FP8, 128, False),
        (QuantFormat.FP8_BLOCK, 128, False),
    ],
)
def test_universal_quant_linear_native_vs_reference(
    fmt: QuantFormat, group_size: int, desc_act: bool
) -> None:
    """Verify native CuTile kernel output matches reference dequant + FP32 matmul across all formats."""
    torch.manual_seed(2026)
    dev = _device()
    m, k, n = 16, 256, 256
    x = torch.randn((m, k), device=dev, dtype=torch.float16) * 0.2
    w = torch.randn((k, n), device=dev, dtype=torch.float16) * 0.1

    uql = UniversalQuantLinear.from_fp16(
        w, fmt=fmt, group_size=group_size, desc_act=desc_act, block_size=(128, 128)
    )
    out_native = uql.forward_native(x)
    out_ref = uql.forward_reference(x)

    max_diff = float((out_native.float() - out_ref.float()).abs().max().item())
    assert max_diff < 5e-2, f"{fmt.value} (gs={group_size}, desc={desc_act}) max_diff={max_diff}"


def test_mixed_precision_format_table_and_end_to_end_perplexity() -> None:
    """Verify per-layer / per-projection `MixedPrecisionFormatTable` and end-to-end perplexity parity."""
    dev = _device()
    if dev.type == "cuda" and MISTRAL_7B_PATH.exists():
        weights = load_gptq_weights(
            MISTRAL_7B_PATH, device="cuda", pin_ddr5=False, max_gpu_layers=2
        )
        base_model = Mistral7B(
            weights=weights,
            gpu_config=GPUConfig(vram_gb=24.0, stream_weights=False),
            device="cuda",
        )
        num_test_layers = 2
    else:
        base_model = _build_synthetic_mistral(dev)
        num_test_layers = 2

    # 1. Standard GGUF Q4_K_M mixed table (Q6_K for v_proj & down_proj, Q4_K for others)
    q4km_table = MixedPrecisionFormatTable.gguf_q4_k_m_table(num_test_layers)
    summary = q4km_table.summary(num_test_layers)
    assert summary["gguf_q4_k"] == num_test_layers * 5
    assert summary["gguf_q6_k"] == num_test_layers * 2

    u_model_q4km = UniversalQuantModel.from_mistral_model(
        base_model, format_table=q4km_table, num_layers=num_test_layers
    )
    input_ids = torch.tensor(
        [[1, 45, 112, 89, 23, 199, 14, 77, 88, 120, 34, 56, 78, 90, 101, 202]],
        device=dev,
        dtype=torch.long,
    )
    ppl_native = u_model_q4km.compute_perplexity(input_ids, mode="native")
    ppl_ref = u_model_q4km.compute_perplexity(input_ids, mode="reference")
    rel_diff = abs(ppl_native - ppl_ref) / max(ppl_ref, 1e-6)
    assert rel_diff < 0.01, f"Q4_K_M perplexity drift too high: native={ppl_native}, ref={ppl_ref}"

    # 2. Heterogeneous multi-format table mixing GPTQ, AWQ, GGUF Q4_K, Q6_K, INT8, FP8_BLOCK
    hetero_table = MixedPrecisionFormatTable(
        default_format=QuantFormat.GPTQ,
        layer_overrides={1: QuantFormat.AWQ},
        projection_overrides={
            (0, "v_proj"): QuantFormat.GGUF_Q6_K,
            (0, "down_proj"): QuantFormat.FP8_BLOCK,
            (1, "o_proj"): QuantFormat.INT8,
            (1, "up_proj"): QuantFormat.GGUF_Q4_K,
        },
    )
    u_model_hetero = UniversalQuantModel.from_mistral_model(
        base_model, format_table=hetero_table, num_layers=num_test_layers
    )
    ppl_h_native = u_model_hetero.compute_perplexity(input_ids, mode="native")
    ppl_h_ref = u_model_hetero.compute_perplexity(input_ids, mode="reference")
    assert abs(ppl_h_native - ppl_h_ref) / max(ppl_h_ref, 1e-6) < 0.01


def test_autotuner_quantized_gemm_keyed_on_format_and_group_size(tmp_path: Path) -> None:
    """Verify `CostGuidedAutotuner.tune_quantized_gemm` caches distinct records per `(format, group_size, shape)`."""
    db_file = tmp_path / "phase10_quant_tuning.json"
    db = TuningDatabase(path=db_file)
    tuner = CostGuidedAutotuner(db=db)

    rec_gptq = tuner.tune_quantized_gemm(
        16, 256, 256, quant_format="gptq", group_size=128, benchmark_on_gpu=True
    )
    rec_awq = tuner.tune_quantized_gemm(
        16, 256, 256, quant_format="awq", group_size=64, benchmark_on_gpu=True
    )
    rec_q4k = tuner.tune_quantized_gemm(
        16, 256, 256, quant_format="gguf_q4_k", group_size=32, benchmark_on_gpu=True
    )

    assert rec_gptq.op == "quant_gemm"
    assert rec_gptq.dtype == "gptq_g128"
    assert rec_awq.dtype == "awq_g64"
    assert rec_q4k.dtype == "gguf_q4_k_g32"
    assert rec_q4k.tile_shape[2] == 64

    db_reloaded = TuningDatabase(path=db_file)
    assert (
        db_reloaded.lookup(
            "quant_gemm",
            tuner.arch,
            (16, 256, 256),
            "gptq_g128",
        )
        is not None
    )
    assert (
        db_reloaded.lookup(
            "quant_gemm",
            tuner.arch,
            (16, 256, 256),
            "awq_g64",
        )
        is not None
    )
