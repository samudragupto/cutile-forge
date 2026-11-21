"""Phase 14D: Physical vs Simulated Multi-GPU TP/PP/EP Validation Harness."""

from __future__ import annotations

import argparse
import os
import subprocess
from pathlib import Path
from typing import Any

import torch

from benchmarks.production.manifest import build_checkpoint_manifest, prompt_checksum
from benchmarks.production.runner import TimedDeviceOp, ValidationRun, peak_vram_mb
from benchmarks.production.schema import QuantizationConfig, WorkloadConfig, validate_artifact_dir
from cutile_forge.distributed import (
    DistributedPagedKVCache,
    ExpertParallelMoE,
    LoopbackCommTransport,
    PipelineParallelEngine,
    TensorParallelGQAAttention,
    TensorParallelSwiGLUMLP,
    detect_topology,
)


def probe_physical_gpus() -> dict[str, Any]:
    """Enumerate physical GPUs via both `nvidia-smi -L` and PyTorch CUDA."""
    smi_lines: list[str] = []
    try:
        out = subprocess.check_output(["nvidia-smi", "-L"], text=True, timeout=10)
        smi_lines = [ln.strip() for ln in out.splitlines() if ln.strip()]
    except Exception as exc:
        smi_lines = [f"error:{exc}"]

    cuda_count = int(torch.cuda.device_count()) if torch.cuda.is_available() else 0
    topo_matrix = ""
    if cuda_count >= 2:
        try:
            topo_matrix = subprocess.check_output(
                ["nvidia-smi", "topo", "-m"], text=True, timeout=10
            ).strip()
        except Exception as exc:
            topo_matrix = f"topo_error:{exc}"

    return {
        "nvidia_smi_L": smi_lines,
        "torch_cuda_device_count": cuda_count,
        "usable_multi_gpu": cuda_count >= 2,
        "topo_matrix": topo_matrix,
    }


def run_single_gpu_simulated_parity(run: ValidationRun, dev: torch.device) -> None:
    """Verify mathematical parity of distributed primitives over loopback on 1 GPU (`SIMULATED`)."""
    torch.manual_seed(1400)
    dt = torch.float16

    # 1. Tensor Parallel SwiGLU MLP (TP=2, TP=4)
    d_m, ffn_m = 4096, 14336
    x_m = torch.randn((16, d_m), device=dev, dtype=dt) * 0.1
    wg = torch.randn((d_m, ffn_m), device=dev, dtype=dt) * 0.05
    wu = torch.randn((d_m, ffn_m), device=dev, dtype=dt) * 0.05
    wd = torch.randn((ffn_m, d_m), device=dev, dtype=dt) * 0.05
    ref_mlp = (
        TensorParallelSwiGLUMLP(wg, wu, wd, LoopbackCommTransport(world_size=1))
        .forward(x_m)
        .float()
    )

    for tp_sz in (2, 4):
        tx = LoopbackCommTransport(world_size=tp_sz, num_nodes=1)
        tp_mlp = TensorParallelSwiGLUMLP(wg, wu, wd, tx)
        out_tp = tp_mlp.forward(x_m).float()
        diff = float((out_tp - ref_mlp).abs().max().item())
        with TimedDeviceOp.measure() as t_op:
            _ = tp_mlp.forward(x_m)
        run.add_sample(f"sim_tp{tp_sz}_mlp_ms", t_op.elapsed_ms, "ms", iteration=0)
        run.add_record(
            {
                "component": f"TensorParallelSwiGLUMLP (TP={tp_sz})",
                "parallelism": f"TP={tp_sz}",
                "status": "SIMULATED",
                "physical_gpus": 1,
                "max_abs_diff_vs_single": diff,
                "loopback_comm_bytes": tx.bytes_transferred,
                "simulated_ms": round(t_op.elapsed_ms, 4),
                "notes": "Single-GPU loopback mathematical equivalence only; NOT physical scaling",
            }
        )

    # 2. Tensor Parallel GQA Attention (TP=2, TP=4)
    x_a = torch.randn((2, 64, 4096), device=dev, dtype=dt) * 0.1
    wq = torch.randn((4096, 4096), device=dev, dtype=dt) * 0.05
    wk = torch.randn((4096, 1024), device=dev, dtype=dt) * 0.05
    wv = torch.randn((4096, 1024), device=dev, dtype=dt) * 0.05
    wo = torch.randn((4096, 4096), device=dev, dtype=dt) * 0.05
    ref_attn = (
        TensorParallelGQAAttention(wq, wk, wv, wo, 32, 8, 128, LoopbackCommTransport(world_size=1))
        .forward(x_a)
        .float()
    )

    for tp_sz in (2, 4):
        tx = LoopbackCommTransport(world_size=tp_sz, num_nodes=1)
        tp_attn = TensorParallelGQAAttention(wq, wk, wv, wo, 32, 8, 128, tx)
        out_tp = tp_attn.forward(x_a).float()
        diff = float((out_tp - ref_attn).abs().max().item())
        with TimedDeviceOp.measure() as t_op:
            _ = tp_attn.forward(x_a)
        run.add_sample(f"sim_tp{tp_sz}_gqa_ms", t_op.elapsed_ms, "ms", iteration=0)
        run.add_record(
            {
                "component": f"TensorParallelGQAAttention (TP={tp_sz})",
                "parallelism": f"TP={tp_sz}",
                "status": "SIMULATED",
                "physical_gpus": 1,
                "max_abs_diff_vs_single": diff,
                "loopback_comm_bytes": tx.bytes_transferred,
                "simulated_ms": round(t_op.elapsed_ms, 4),
                "notes": "Single-GPU loopback mathematical equivalence only",
            }
        )

    # 3. Pipeline Parallelism (PP=2)
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
    tx_pp = LoopbackCommTransport(world_size=2, num_nodes=1)
    pp_eng = PipelineParallelEngine(pp_layers, tx_pp, num_microbatches=4)
    out_pp = pp_eng.forward(x_pp).float()
    ref_pp = pp_eng.forward_single_device(x_pp).float()
    diff_pp = float((out_pp - ref_pp).abs().max().item())
    with TimedDeviceOp.measure() as t_pp:
        _ = pp_eng.forward(x_pp)
    run.add_record(
        {
            "component": "PipelineParallelEngine (PP=2)",
            "parallelism": "PP=2",
            "status": "SIMULATED",
            "physical_gpus": 1,
            "max_abs_diff_vs_single": diff_pp,
            "loopback_comm_bytes": tx_pp.bytes_transferred,
            "simulated_ms": round(t_pp.elapsed_ms, 4),
            "notes": "Single-GPU loopback mathematical equivalence only",
        }
    )

    # 4. Expert Parallel MoE (EP=2)
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
    tx_ep = LoopbackCommTransport(world_size=2, num_nodes=1)
    ep_mod = ExpertParallelMoE(router_w, exps, top_k=2, transport=tx_ep)
    out_ep = ep_mod.forward_ep(x_ep).float()
    ref_ep = ep_mod.forward_single_device(x_ep).float()
    diff_ep = float((out_ep - ref_ep).abs().max().item())
    run.add_record(
        {
            "component": "ExpertParallelMoE (EP=2)",
            "parallelism": "EP=2",
            "status": "SIMULATED",
            "physical_gpus": 1,
            "max_abs_diff_vs_single": diff_ep,
            "loopback_comm_bytes": tx_ep.bytes_transferred,
            "notes": "Single-GPU loopback mathematical equivalence only",
        }
    )

    # 5. Distributed Paged KV Cache
    from cutile_forge.runtime.native_kv_cache import NativePagedKVCache

    dkv = DistributedPagedKVCache(
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
        num_layers=8, num_kv_heads=8, head_dim=128, max_seq_len=256, max_batch_size=2, device=dev
    )
    k_t = torch.randn((2, 8, 64, 128), device=dev, dtype=dt)
    v_t = torch.randn((2, 8, 64, 128), device=dev, dtype=dt)
    dkv.append_full(0, k_t, v_t, start_pos=0)
    single_kv.append_and_get(0, k_t, v_t, start_pos=0)
    kd, vd = dkv.gather_full(0, seq_len=64, batch_size=2)
    ks, vs = single_kv.get_kv(0, end_pos=64, batch_size=2)
    kv_diff = float(
        max(
            (kd.float() - ks.float()).abs().max().item(),
            (vd.float() - vs.float()).abs().max().item(),
        )
    )
    run.add_record(
        {
            "component": "DistributedPagedKVCache (PP=2,TP=2)",
            "parallelism": "PP=2,TP=2 KV",
            "status": "SIMULATED",
            "physical_gpus": 1,
            "max_abs_diff_vs_single": kv_diff,
            "notes": "Shard partitioning verified on 1 GPU; physical P2P migration PENDING_HARDWARE",
        }
    )

    # Physical Multi-GPU targets remain strictly PENDING_HARDWARE
    for phys_target in (
        "TP=2",
        "TP=4",
        "PP=2",
        "EP=2",
        "DistributedPagedKVCache Physical Migration",
    ):
        run.add_record(
            {
                "component": f"Physical Multi-GPU {phys_target}",
                "parallelism": phys_target,
                "status": "PENDING_HARDWARE",
                "physical_gpus": 1,
                "notes": (
                    "Workstation has 1 physical GPU (RTX 5090). Loopback transport was NOT used "
                    "for physical timing. Physical multi-GPU scaling remains unvalidated."
                ),
            }
        )


def _run_nccl_worker(rank: int, world_size: int, out_dict: Any) -> None:
    """Real multi-GPU NCCL validation worker (executed only when >= 2 GPUs exist)."""
    import time

    import torch.distributed as dist

    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29514")
    dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)
    dev = torch.device(f"cuda:{rank}")

    t0 = time.perf_counter()
    x = torch.ones((1024, 4096), device=dev, dtype=torch.float16) * (rank + 1)
    torch.cuda.synchronize(dev)
    t_c0 = time.perf_counter()
    dist.all_reduce(x, op=dist.ReduceOp.SUM)
    torch.cuda.synchronize(dev)
    comm_ms = (time.perf_counter() - t_c0) * 1000.0
    vram_alloc, _ = peak_vram_mb()

    out_dict[rank] = {
        "rank": rank,
        "pid": os.getpid(),
        "nccl_version": ".".join(str(v) for v in torch.cuda.nccl.version()),
        "allreduce_bytes": int(x.numel() * x.element_size()),
        "comm_ms": comm_ms,
        "total_ms": (time.perf_counter() - t0) * 1000.0,
        "peak_vram_mb": vram_alloc,
        "expected_sum_err": float((x - sum(range(1, world_size + 1))).abs().max().item()),
    }
    dist.destroy_process_group()


def run_multi_gpu_validation(output_dir: Path) -> Path:
    probe = probe_physical_gpus()
    topo = detect_topology()
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    mode = "physical-multi-gpu" if probe["usable_multi_gpu"] else "simulated"
    repro = "MEASURED" if probe["usable_multi_gpu"] else "SIMULATED"

    run = ValidationRun(
        run_dir=output_dir,
        run_id="phase14d_multi_gpu",
        checkpoint=build_checkpoint_manifest("synthetic-distributed", Path("/dev/null")),
        quant_config=QuantizationConfig(quant_format="fp16", kv_dtype="fp16"),
        workload=WorkloadConfig(batch_size=16, prompt_len=64, output_len=0, measured_iters=1),
        prompt_ids_or_checksum=prompt_checksum("phase14d_distributed"),
        execution_mode=mode,
        reproduction_status=repro,
        notes=(
            f"nvidia-smi -L: {probe['nvidia_smi_L']}; "
            f"torch.cuda.device_count={probe['torch_cuda_device_count']}; "
            f"detected_devices={[d.name for d in topo.devices]}"
        ),
    )

    if not probe["usable_multi_gpu"]:
        run.log_stdout(
            f"[Phase 14D] Detected {probe['torch_cuda_device_count']} physical GPU(s): "
            f"{probe['nvidia_smi_L']}. Running 1-GPU mathematical parity as SIMULATED; "
            f"physical TP/PP/EP scaling marked PENDING_HARDWARE."
        )
        run_single_gpu_simulated_parity(run, dev)
        run.add_record(
            {
                "component": "Hardware Probe",
                "parallelism": "N/A",
                "status": "PENDING_HARDWARE",
                "physical_gpus": probe["torch_cuda_device_count"],
                "nvidia_smi_L": probe["nvidia_smi_L"],
                "notes": "Physical multi-GPU validation requires >= 2 physical GPUs.",
            }
        )
    else:
        import torch.multiprocessing as mp

        mgr = mp.Manager()
        shared = mgr.dict()
        world = min(2, probe["torch_cuda_device_count"])
        mp.spawn(_run_nccl_worker, args=(world, shared), nprocs=world, join=True)
        for r in range(world):
            info = dict(shared[r])
            run.add_sample(f"nccl_rank{r}_comm_ms", info["comm_ms"], "ms", iteration=r)
            run.add_record(
                {
                    "component": f"Physical NCCL TP={world} Rank {r}",
                    "parallelism": f"TP={world}",
                    "status": "MEASURED",
                    "physical_gpus": probe["torch_cuda_device_count"],
                    "topo_matrix": probe["topo_matrix"],
                    **info,
                }
            )

    out = run.finalize()
    errs = validate_artifact_dir(out)
    assert errs == [], f"Phase 14D artifact validation failed: {errs}"
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 14D Physical / Simulated Multi-GPU Harness")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("benchmarks/validation_artifacts/v3.1.0/phase14d_multi_gpu"),
    )
    args = parser.parse_args()
    run_multi_gpu_validation(args.output_dir)


if __name__ == "__main__":
    main()
