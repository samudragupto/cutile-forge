"""Phase 12 (`v2.7.0`) tests: Distributed GPU Runtime (TP, PP, EP, Distributed KV-Cache, Scheduler, Transport).

Zero `pytest.skip` policy: every test executes end-to-end on both single-GPU CUDA (`sm_120`) and CPU CI
using logical partitions + `LoopbackCommTransport` and verifies `multi_rank_output == single_device_output`.
"""

from __future__ import annotations

import torch

from cutile_forge.distributed import (
    ColumnParallelLinear,
    DistributedPagedKVCache,
    DistributedScheduler,
    ExpertParallelMoE,
    LoopbackCommTransport,
    PipelineParallelEngine,
    RowParallelLinear,
    TensorParallelGQAAttention,
    TensorParallelSwiGLUMLP,
)
from cutile_forge.runtime.native_kv_cache import NativePagedKVCache


def _device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def test_tensor_parallel_linear_swiglu_and_gqa_attention_parity() -> None:
    """Verify TP=4 ColumnParallel, RowParallel, SwiGLU MLP, and GQA Attention match single-device execution."""
    torch.manual_seed(1201)
    dev = _device()
    dt = torch.float16
    tx = LoopbackCommTransport(world_size=4, num_nodes=2)

    m, k, n = 16, 256, 512
    x = torch.randn((m, k), device=dev, dtype=dt) * 0.15
    w_col = torch.randn((k, n), device=dev, dtype=dt) * 0.1
    b_col = torch.randn((n,), device=dev, dtype=dt) * 0.02

    col = ColumnParallelLinear(w_col, b_col, tx, gather_output=True)
    out_col_tp = col.forward_ranks([x for _ in range(4)])[0]
    out_col_ref = (torch.matmul(x.float(), w_col.float()) + b_col.float()).to(dt)
    assert float((out_col_tp.float() - out_col_ref.float()).abs().max().item()) < 1e-2

    w_row = torch.randn((n, k), device=dev, dtype=dt) * 0.1
    b_row = torch.randn((k,), device=dev, dtype=dt) * 0.02
    x_row = torch.randn((m, n), device=dev, dtype=dt) * 0.15
    row = RowParallelLinear(w_row, b_row, tx, input_is_parallel=False)
    out_row_tp = row.forward_ranks([x_row for _ in range(4)])[0]
    out_row_ref = (torch.matmul(x_row.float(), w_row.float()) + b_row.float()).to(dt)
    assert float((out_row_tp.float() - out_row_ref.float()).abs().max().item()) < 2e-2

    # SwiGLU MLP TP=4 vs single-device
    w_gate = torch.randn((k, n), device=dev, dtype=dt) * 0.1
    w_up = torch.randn((k, n), device=dev, dtype=dt) * 0.1
    w_down = torch.randn((n, k), device=dev, dtype=dt) * 0.1
    tp_mlp = TensorParallelSwiGLUMLP(w_gate, w_up, w_down, tx)
    out_mlp_tp = tp_mlp.forward(x)

    tx_1 = LoopbackCommTransport(world_size=1)
    single_mlp = TensorParallelSwiGLUMLP(w_gate, w_up, w_down, tx_1)
    out_mlp_single = single_mlp.forward(x)
    assert float((out_mlp_tp.float() - out_mlp_single.float()).abs().max().item()) < 2e-2

    # GQA Attention TP=4 (8 query heads, 4 KV heads -> 2 query heads & 1 KV head per rank)
    x_3d = torch.randn((2, 16, 256), device=dev, dtype=dt) * 0.15
    wq = torch.randn((256, 256), device=dev, dtype=dt) * 0.1
    wk = torch.randn((256, 128), device=dev, dtype=dt) * 0.1
    wv = torch.randn((256, 128), device=dev, dtype=dt) * 0.1
    wo = torch.randn((256, 256), device=dev, dtype=dt) * 0.1
    tp_attn = TensorParallelGQAAttention(wq, wk, wv, wo, 8, 4, 32, tx)
    single_attn = TensorParallelGQAAttention(wq, wk, wv, wo, 8, 4, 32, tx_1)
    out_attn_tp = tp_attn.forward(x_3d)
    out_attn_single = single_attn.forward(x_3d)
    assert float((out_attn_tp.float() - out_attn_single.float()).abs().max().item()) < 2e-2
    assert tx.bytes_transferred > 0


def test_pipeline_parallel_microbatch_schedule_bit_exact_parity() -> None:
    """Verify 4-stage PipelineParallelEngine over 4 micro-batches matches single-device execution bit-for-bit."""
    torch.manual_seed(1202)
    dev = _device()
    dt = torch.float16
    d, ffn = 128, 256

    layers = [
        {
            "in_norm_w": torch.ones(d, device=dev, dtype=dt),
            "post_norm_w": torch.ones(d, device=dev, dtype=dt),
            "w_attn": torch.randn((d, d), device=dev, dtype=dt) * 0.08,
            "w_gate": torch.randn((d, ffn), device=dev, dtype=dt) * 0.08,
            "w_up": torch.randn((d, ffn), device=dev, dtype=dt) * 0.08,
            "w_down": torch.randn((ffn, d), device=dev, dtype=dt) * 0.08,
        }
        for _ in range(4)
    ]
    tx = LoopbackCommTransport(world_size=4, num_nodes=2)
    pp_engine = PipelineParallelEngine(layers, tx, num_microbatches=4)

    x = torch.randn((8, 16, d), device=dev, dtype=dt) * 0.15
    out_pp = pp_engine.forward(x)
    out_single = pp_engine.forward_single_device(x)

    assert float((out_pp.float() - out_single.float()).abs().max().item()) == 0.0


def test_expert_parallel_all_to_all_bit_exact_parity() -> None:
    """Verify EP=4 All-to-All token dispatch + combine matches single-device MoE bit-for-bit."""
    torch.manual_seed(1203)
    dev = _device()
    dt = torch.float16
    d, ffn, num_experts = 128, 256, 8

    router_w = torch.randn((d, num_experts), device=dev, dtype=dt) * 0.2
    experts = [
        {
            "w_gate": torch.randn((d, ffn), device=dev, dtype=dt) * 0.1,
            "w_up": torch.randn((d, ffn), device=dev, dtype=dt) * 0.1,
            "w_down": torch.randn((ffn, d), device=dev, dtype=dt) * 0.1,
        }
        for _ in range(num_experts)
    ]
    tx = LoopbackCommTransport(world_size=4, num_nodes=2)
    ep_moe = ExpertParallelMoE(router_w, experts, top_k=2, transport=tx)

    x = torch.randn((4, 16, d), device=dev, dtype=dt) * 0.15
    out_ep = ep_moe.forward_ep(x)
    out_single = ep_moe.forward_single_device(x)

    assert float((out_ep.float() - out_single.float()).abs().max().item()) == 0.0


def test_distributed_paged_kv_cache_and_scheduler() -> None:
    """Verify `DistributedPagedKVCache` `(PP=2, TP=2)` sharding/gathering and `DistributedScheduler`."""
    torch.manual_seed(1204)
    dev = _device()
    dt = torch.float16

    dist_kv = DistributedPagedKVCache(
        num_layers=4,
        num_kv_heads=4,
        head_dim=64,
        tp_size=2,
        pp_size=2,
        block_size=16,
        max_seq_len=128,
        max_batch_size=2,
        kv_dtype="fp16",
        device=dev,
    )
    single_kv = NativePagedKVCache(
        num_layers=4,
        num_kv_heads=4,
        head_dim=64,
        block_size=16,
        max_seq_len=128,
        max_batch_size=2,
        kv_dtype="fp16",
        device=dev,
    )

    k = torch.randn((2, 4, 20, 64), device=dev, dtype=dt)
    v = torch.randn((2, 4, 20, 64), device=dev, dtype=dt)
    for l_idx in range(4):
        dist_kv.append_full(l_idx, k, v, start_pos=0)
        single_kv.append_and_get(l_idx, k, v, start_pos=0)
        kd, vd = dist_kv.gather_full(l_idx, seq_len=20, batch_size=2)
        ks, vs = single_kv.get_kv(l_idx, end_pos=20, batch_size=2)
        assert torch.equal(kd, ks)
        assert torch.equal(vd, vs)

    sched = DistributedScheduler(dp_size=2, tp_size=2, pp_size=2)
    assert sched.total_world_size == 8
    assigned = sched.assign_requests([1024, 512, 512, 256, 256])
    load_0 = sum(length for _, length in assigned[0])
    load_1 = sum(length for _, length in assigned[1])
    assert load_0 == 1280 and load_1 == 1280
