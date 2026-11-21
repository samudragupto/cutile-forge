"""Phase 12 (`v2.7.0`) Benchmark: Distributed GPU Runtime (Logical Multi-Rank on 1x RTX 5090).

Hardware detected (`nvidia-smi -L`):
- `GPU 0: NVIDIA GeForce RTX 5090` (1 physical GPU).

All performance figures in this benchmark execute over logical partitions + `LoopbackCommTransport`
on the single RTX 5090 and are explicitly labeled `execution_mode="simulated"` while verifying
numerical parity against single-device execution (`max_abs_diff_vs_single_device`).
"""

from __future__ import annotations

import argparse
import csv
import subprocess
from pathlib import Path
from typing import Any

import numpy as np
import torch

from cutile_forge.distributed import (
    DistributedPagedKVCache,
    ExpertParallelMoE,
    LoopbackCommTransport,
    PipelineParallelEngine,
    TensorParallelGQAAttention,
    TensorParallelSwiGLUMLP,
    detect_topology,
)
from cutile_forge.runtime.native_kv_cache import NativePagedKVCache


def _check_gpu_contention() -> tuple[bool, str]:
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-compute-apps=pid,used_memory", "--format=csv,noheader"],
            text=True,
        ).strip()
        if not out:
            return False, "none"
        lines = [line.strip() for line in out.splitlines() if line.strip()]
        return len(lines) > 1, "; ".join(lines)
    except Exception as exc:
        return False, f"query_failed:{exc}"


def _time_fn(fn: Any, runs: int = 5, warmup: int = 3) -> tuple[float, float, float]:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    times: list[float] = []
    for _ in range(runs):
        s_ev = torch.cuda.Event(enable_timing=True)
        e_ev = torch.cuda.Event(enable_timing=True)
        s_ev.record()
        fn()
        e_ev.record()
        torch.cuda.synchronize()
        times.append(float(s_ev.elapsed_time(e_ev)))
    return float(np.median(times)), float(np.min(times)), float(np.max(times))


def run_phase12_benchmark(output_csv: Path) -> list[dict[str, Any]]:
    topo = detect_topology()
    contended, gpu_apps = _check_gpu_contention()
    print(
        f"[Phase 12] Topology: {topo.num_devices} physical GPU(s) "
        f"({[d.name for d in topo.devices]}); all multi-rank figures labeled 'simulated'."
    )

    torch.manual_seed(1200)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dt = torch.float16
    rows: list[dict[str, Any]] = []

    def _record(
        workload: str,
        parallelism: str,
        world_size: int,
        num_nodes: int,
        max_diff: float,
        med_ms: float,
        min_ms: float,
        max_ms: float,
        comm_bytes: int,
    ) -> None:
        row = {
            "workload": workload,
            "parallelism_mode": parallelism,
            "logical_world_size": world_size,
            "logical_nodes": num_nodes,
            "physical_gpus_present": topo.num_devices,
            "execution_mode": "simulated",
            "max_abs_diff_vs_single_device": f"{max_diff:.6e}",
            "median_latency_ms": f"{med_ms:.4f}",
            "min_latency_ms": f"{min_ms:.4f}",
            "max_latency_ms": f"{max_ms:.4f}",
            "loopback_comm_bytes": comm_bytes,
            "gpu_contended": contended,
            "gpu_compute_apps": gpu_apps,
        }
        rows.append(row)
        print(
            f"  [{workload:28s} | {parallelism:10s}] mode=simulated | "
            f"max_diff={max_diff:.3e} | med={med_ms:.3f}ms | comm={comm_bytes}B"
        )

    # 1. Tensor Parallel SwiGLU MLP (d=4096, ffn=14336, M=16) for TP=2 and TP=4
    d_m, ffn_m = 4096, 14336
    x_m = torch.randn((16, d_m), device=dev, dtype=dt) * 0.1
    wg = torch.randn((d_m, ffn_m), device=dev, dtype=dt) * 0.05
    wu = torch.randn((d_m, ffn_m), device=dev, dtype=dt) * 0.05
    wd = torch.randn((ffn_m, d_m), device=dev, dtype=dt) * 0.05
    single_mlp = TensorParallelSwiGLUMLP(wg, wu, wd, LoopbackCommTransport(world_size=1))
    ref_mlp = single_mlp.forward(x_m).float()

    for tp_sz in (2, 4):
        tx = LoopbackCommTransport(world_size=tp_sz, num_nodes=1)
        tp_mlp = TensorParallelSwiGLUMLP(wg, wu, wd, tx)
        out_tp = tp_mlp.forward(x_m).float()
        comm_b = tx.bytes_transferred
        diff = float((out_tp - ref_mlp).abs().max().item())
        med, mn, mx = _time_fn(lambda m_ref=tp_mlp: m_ref.forward(x_m))
        _record("Mistral7B_SwiGLU_M16", f"TP={tp_sz}", tp_sz, 1, diff, med, mn, mx, comm_b)

    # 2. Tensor Parallel GQA Attention (32:8 heads, head_dim=128, B=2, S=64) for TP=2 and TP=4
    x_a = torch.randn((2, 64, 4096), device=dev, dtype=dt) * 0.1
    wq = torch.randn((4096, 4096), device=dev, dtype=dt) * 0.05
    wk = torch.randn((4096, 1024), device=dev, dtype=dt) * 0.05
    wv = torch.randn((4096, 1024), device=dev, dtype=dt) * 0.05
    wo = torch.randn((4096, 4096), device=dev, dtype=dt) * 0.05
    single_attn = TensorParallelGQAAttention(
        wq, wk, wv, wo, 32, 8, 128, LoopbackCommTransport(world_size=1)
    )
    ref_attn = single_attn.forward(x_a).float()

    for tp_sz in (2, 4):
        tx = LoopbackCommTransport(world_size=tp_sz, num_nodes=1)
        tp_attn = TensorParallelGQAAttention(wq, wk, wv, wo, 32, 8, 128, tx)
        out_tp = tp_attn.forward(x_a).float()
        comm_b = tx.bytes_transferred
        diff = float((out_tp - ref_attn).abs().max().item())
        med, mn, mx = _time_fn(lambda a_ref=tp_attn: a_ref.forward(x_a))
        _record("Mistral7B_GQA_B2S64", f"TP={tp_sz}", tp_sz, 1, diff, med, mn, mx, comm_b)

    # 3. Pipeline Parallelism (8 layers, d=1024, ffn=4096, 4 microbatches) for PP=2 and PP=4
    d_p, ffn_p = 1024, 4096
    pp_layers = [
        {
            "in_norm_w": torch.ones(d_p, device=dev, dtype=dt),
            "post_norm_w": torch.ones(d_p, device=dev, dtype=dt),
            "w_attn": torch.randn((d_p, d_p), device=dev, dtype=dt) * 0.04,
            "w_gate": torch.randn((d_p, ffn_p), device=dev, dtype=dt) * 0.04,
            "w_up": torch.randn((d_p, ffn_p), device=dev, dtype=dt) * 0.04,
            "w_down": torch.randn((ffn_p, d_p), device=dev, dtype=dt) * 0.04,
        }
        for _ in range(8)
    ]
    x_pp = torch.randn((8, 32, d_p), device=dev, dtype=dt) * 0.1
    for pp_sz, nodes in ((2, 1), (4, 2)):
        tx = LoopbackCommTransport(world_size=pp_sz, num_nodes=nodes)
        pp_eng = PipelineParallelEngine(pp_layers, tx, num_microbatches=4)
        out_pp = pp_eng.forward(x_pp).float()
        comm_b = tx.bytes_transferred
        ref_pp = pp_eng.forward_single_device(x_pp).float()
        diff = float((out_pp - ref_pp).abs().max().item())
        med, mn, mx = _time_fn(lambda p_ref=pp_eng: p_ref.forward(x_pp))
        _record("Transformer8L_Microbatch4", f"PP={pp_sz}", pp_sz, nodes, diff, med, mn, mx, comm_b)

    # 4. Expert Parallelism (8 experts, top-2, d=1024, ffn=4096) for EP=2 and EP=4
    router_w = torch.randn((d_p, 8), device=dev, dtype=dt) * 0.1
    exps = [
        {
            "w_gate": torch.randn((d_p, ffn_p), device=dev, dtype=dt) * 0.04,
            "w_up": torch.randn((d_p, ffn_p), device=dev, dtype=dt) * 0.04,
            "w_down": torch.randn((ffn_p, d_p), device=dev, dtype=dt) * 0.04,
        }
        for _ in range(8)
    ]
    x_ep = torch.randn((4, 32, d_p), device=dev, dtype=dt) * 0.1
    for ep_sz, nodes in ((2, 1), (4, 2)):
        tx = LoopbackCommTransport(world_size=ep_sz, num_nodes=nodes)
        ep_moe = ExpertParallelMoE(router_w, exps, top_k=2, transport=tx)
        out_ep = ep_moe.forward_ep(x_ep).float()
        comm_b = tx.bytes_transferred
        ref_ep = ep_moe.forward_single_device(x_ep).float()
        diff = float((out_ep - ref_ep).abs().max().item())
        med, mn, mx = _time_fn(lambda e_ref=ep_moe: e_ref.forward_ep(x_ep))
        _record("MoE_8xTop2_AllToAll", f"EP={ep_sz}", ep_sz, nodes, diff, med, mn, mx, comm_b)

    # 5. Distributed Paged KV-Cache (PP=2, TP=2, 8 layers, 8 KV heads, seq=128)
    dist_kv = DistributedPagedKVCache(
        num_layers=8,
        num_kv_heads=8,
        head_dim=128,
        tp_size=2,
        pp_size=2,
        max_seq_len=256,
        max_batch_size=2,
        device=dev,
    )
    single_kv = NativePagedKVCache(
        num_layers=8,
        num_kv_heads=8,
        head_dim=128,
        max_seq_len=256,
        max_batch_size=2,
        device=dev,
    )
    k_t = torch.randn((2, 8, 64, 128), device=dev, dtype=dt)
    v_t = torch.randn((2, 8, 64, 128), device=dev, dtype=dt)
    dist_kv.append_full(0, k_t, v_t, start_pos=0)
    single_kv.append_and_get(0, k_t, v_t, start_pos=0)
    kd, vd = dist_kv.gather_full(0, seq_len=64, batch_size=2)
    ks, vs = single_kv.get_kv(0, end_pos=64, batch_size=2)
    kv_diff = float(
        max(
            (kd.float() - ks.float()).abs().max().item(),
            (vd.float() - vs.float()).abs().max().item(),
        )
    )
    med_kv, min_kv, max_kv = _time_fn(
        lambda: (
            dist_kv.append_full(0, k_t, v_t, start_pos=0),
            dist_kv.gather_full(0, seq_len=64, batch_size=2),
        )
    )
    _record(
        "DistributedPagedKVCache_S64",
        "PP=2,TP=2",
        4,
        1,
        kv_diff,
        med_kv,
        min_kv,
        max_kv,
        k_t.numel() * 4,
    )

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"[Phase 12] Saved {len(rows)} rows to {output_csv}")
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 12 Distributed Runtime Benchmark")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("benchmarks/results_5090/phase12_distributed_simulated.csv"),
    )
    args = parser.parse_args()
    run_phase12_benchmark(args.output)


if __name__ == "__main__":
    main()
