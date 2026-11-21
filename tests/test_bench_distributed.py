"""Tests for CuTileBench v2 telemetry/regression gates and distributed P2P primitives (Phase 8I)."""

from __future__ import annotations

import torch

from cutile_forge.bench import benchmark_with_telemetry, check_regression_gate
from cutile_forge.distributed import detect_topology, p2p_copy, ring_allreduce_sum


def test_benchmark_telemetry_and_regression_gate() -> None:
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    a = torch.randn((128, 128), device=dev, dtype=torch.float32)
    b = torch.randn((128, 128), device=dev, dtype=torch.float32)

    def _run_add() -> torch.Tensor:
        return a + b

    rec = benchmark_with_telemetry(
        _run_add,
        flops=128 * 128,
        num_bytes=3 * 128 * 128 * 4,
        warmup=2,
        iters=5,
        name="vec_add",
    )
    assert rec.name == "vec_add"
    assert rec.stats.p50_ms > 0.0
    assert rec.tflops >= 0.0
    assert rec.bandwidth_gb_s >= 0.0
    row = rec.to_dict()
    assert "peak_vram_mb" in row and "device_name" in row

    passed, pct = check_regression_gate(10.0, 10.3, max_regression_pct=5.0)
    assert passed is True and abs(pct - 3.0) < 1e-6

    failed, pct_bad = check_regression_gate(10.0, 11.0, max_regression_pct=5.0)
    assert failed is False and abs(pct_bad - 10.0) < 1e-6

    assert check_regression_gate(0.0, 5.0) == (True, 0.0)


def test_topology_p2p_and_ring_allreduce() -> None:
    topo = detect_topology()
    if torch.cuda.is_available():
        assert topo.num_devices == torch.cuda.device_count()
        assert len(topo.devices) == topo.num_devices
        assert len(topo.p2p_matrix) == topo.num_devices
        assert topo.p2p_matrix[0][0] is True
    else:
        assert topo.num_devices == 0
        assert topo.has_multi_gpu is False

    dev = "cuda:0" if torch.cuda.is_available() else "cpu"
    src = torch.arange(16, dtype=torch.float32, device=dev)
    copied = p2p_copy(src, "cpu")
    assert copied.device.type == "cpu" and torch.equal(copied, src.cpu())

    # Exact ring all-reduce parity for 2 and 4 ranks + non-divisible length
    for ranks, length in ((2, 64), (4, 1025)):
        rank_tensors = [
            torch.randn((length,), dtype=torch.float32, device=dev) for _ in range(ranks)
        ]
        expected = sum(rank_tensors)
        reduced = ring_allreduce_sum(rank_tensors)
        assert len(reduced) == ranks
        for r in reduced:
            assert (r - expected).abs().max().item() < 1e-5
