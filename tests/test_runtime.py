"""Tests for `cutile_forge.runtime` (pinned pool, triple-buffered DMA, backpressure, GraphRuntime)."""

from __future__ import annotations

import subprocess
import sys

import torch
import torch.nn.functional as F

from cutile_forge.runtime import GraphRuntime, PinnedHostPool, TensorStreamer
from cutile_forge.sdk import DType, Graph, compile


def test_runtime_module_isolated_from_llm_engine() -> None:
    code = (
        "import sys; import cutile_forge.runtime; "
        "assert 'cutile_forge.engine' not in sys.modules; "
        "assert 'cutile_forge.server' not in sys.modules; "
        "print('RUNTIME_ISOLATED_OK')"
    )
    proc = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
    assert "RUNTIME_ISOLATED_OK" in proc.stdout


def test_pinned_host_pool_reuses_buffers() -> None:
    pool = PinnedHostPool(max_cached_per_bucket=4)
    b1 = pool.acquire((16, 32), torch.float16)
    assert pool.misses == 1 and pool.allocations == 1
    pool.release(b1)
    b2 = pool.acquire((16, 32), torch.float16)
    assert pool.hits == 1
    assert b2 is b1


def test_tensor_streamer_triple_buffering_and_backpressure() -> None:
    # 3 slots, VRAM cap fits at most 2 tensors of size (32, 32) fp16 = 2048 B each -> 4096 B cap
    streamer = TensorStreamer(num_slots=3, max_vram_bytes=4096)
    t1 = torch.ones((32, 32), dtype=torch.float16) * 1.0
    t2 = torch.ones((32, 32), dtype=torch.float16) * 2.0
    t3 = torch.ones((32, 32), dtype=torch.float16) * 3.0

    streamer.prefetch("t1", t1)
    streamer.prefetch("t2", t2)
    assert streamer.evictions == 0
    assert streamer.vram_used_bytes == 4096

    # Prefetching t3 triggers backpressure eviction of LRU (t1)
    streamer.prefetch("t3", t3)
    assert streamer.evictions == 1
    assert streamer.vram_used_bytes == 4096
    assert torch.equal(streamer.acquire("t2").cpu(), t2)
    assert torch.equal(streamer.acquire("t3").cpu(), t3)


def test_graph_runtime_executes_with_cpu_staged_inputs() -> None:
    g = Graph("streamed_ffn")
    x = g.input("x", (32, 64), DType.FP16)
    w = g.input("w", (64, 128), DType.FP16)
    out = g.silu(g.matmul(x, w))
    g.output(out)

    compiled = compile(g, target="auto", use_cache=False)
    rt = GraphRuntime(num_slots=3)

    tx_cpu = torch.randn((32, 64), device="cpu", dtype=torch.float16) * 0.1
    tw_cpu = torch.randn((64, 128), device="cpu", dtype=torch.float16) * 0.1
    res = rt.execute(compiled, {"x": tx_cpu, "w": tw_cpu})
    got = res[out.name]
    ref = F.silu(torch.matmul(tx_cpu.float(), tw_cpu.float()).to(torch.float16))
    assert (got.cpu() - ref).abs().max().item() < 1e-2
    if torch.cuda.is_available():
        assert rt.streamer.total_bytes_streamed > 0
