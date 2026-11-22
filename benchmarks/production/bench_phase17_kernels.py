"""Phase 17 Kernel Optimization Benchmark & Parity Harness (benchmarks/production/bench_phase17_kernels.py)."""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import subprocess
from collections.abc import Callable
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

from benchmarks.production.manifest import build_checkpoint_manifest, prompt_checksum
from benchmarks.production.runner import ValidationRun
from benchmarks.production.schema import QuantizationConfig, WorkloadConfig, validate_artifact_dir
from cutile_forge.bench.harness import benchmark
from cutile_forge.kernels import (
    cutile_avg_pool2d,
    cutile_bmm,
    cutile_conv2d,
    cutile_depthwise_conv2d,
    cutile_fused_layernorm_gelu,
    cutile_gather,
    cutile_gelu,
    cutile_gemm_fp16,
    cutile_max_pool2d,
    cutile_reduce_max,
    cutile_reduce_sum,
    cutile_silu,
    cutile_transpose,
)


def _query_contention() -> tuple[bool, str]:
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


def _time_multi_runs(
    fn: Callable[[], Any],
    *,
    warmup: int = 10,
    iters: int = 50,
    runs: int = 5,
) -> tuple[float, float, float]:
    """Run `runs` independent benchmark sweeps and return `(median_ms, min_ms, max_ms)`."""
    # Single upfront warmup
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    medians: list[float] = []
    for _ in range(runs):
        st = benchmark(fn, warmup=2, iters=iters)
        medians.append(float(st.p50_ms))
    return (
        round(statistics.median(medians), 5),
        round(min(medians), 5),
        round(max(medians), 5),
    )


def run_micro_ops(runs: int = 5, iters: int = 50, family: str = "all") -> list[dict[str, Any]]:
    """Benchmark target operators (CuTile vs PyTorch) + accuracy check."""
    torch.manual_seed(42)
    dtype = torch.float16
    results: list[dict[str, Any]] = []

    def _bench_op(
        op: str,
        op_family: str,
        shape: str,
        ct_fn: Callable[[], torch.Tensor],
        pt_fn: Callable[[], torch.Tensor],
        atol: float,
    ) -> None:
        if family not in ("all", op_family):
            return
        ref = pt_fn().float()
        got = ct_fn().float()
        max_abs_diff = float((got - ref).abs().max().item())
        ct_med, ct_min, ct_max = _time_multi_runs(ct_fn, runs=runs, iters=iters)
        pt_med, pt_min, pt_max = _time_multi_runs(pt_fn, runs=runs, iters=iters)
        speedup = round(pt_med / max(1e-7, ct_med), 3)
        results.append(
            {
                "op": op,
                "family": op_family,
                "shape": shape,
                "cutile_med_ms": ct_med,
                "cutile_min_ms": ct_min,
                "cutile_max_ms": ct_max,
                "torch_med_ms": pt_med,
                "torch_min_ms": pt_min,
                "torch_max_ms": pt_max,
                "speedup": speedup,
                "gap_ratio": round(ct_med / max(1e-7, pt_med), 3),
                "max_abs_diff": round(max_abs_diff, 6),
                "accuracy_ok": max_abs_diff <= atol,
                "atol": atol,
                "runs": runs,
                "iters_per_run": iters,
                "provenance": "synthetic-weights",
            }
        )

    # 1. FP16 GEMM shapes
    if family in ("all", "gemm"):
        for name, m, n, k in [
            ("gemm_skinny_m4", 4, 4096, 4096),
            ("gemm_medium_m64", 64, 4096, 4096),
            ("gemm_m256_down", 256, 4096, 14336),
            ("gemm_m256_gate", 256, 14336, 4096),
            ("gemm_m512", 512, 4096, 4096),
            ("gemm_large_m1024", 1024, 4096, 4096),
        ]:
            a = torch.randn((m, k), device="cuda", dtype=dtype) * 0.05
            b = torch.randn((k, n), device="cuda", dtype=dtype) * 0.05
            _bench_op(
                name,
                "gemm",
                f"{m}x{n}x{k}",
                lambda a=a, b=b: cutile_gemm_fp16(a, b),
                lambda a=a, b=b: torch.matmul(a, b),
                atol=0.1,
            )

    # 2. Vision / Conv / Pool
    if family in ("all", "vision"):
        cx = torch.randn((16, 64, 56, 56), device="cuda", dtype=dtype) * 0.1
        cw3 = torch.randn((128, 64, 3, 3), device="cuda", dtype=dtype) * 0.1
        cdw = torch.randn((64, 1, 3, 3), device="cuda", dtype=dtype) * 0.1
        _bench_op(
            "depthwise_conv2d",
            "vision",
            "16x64x56x56,k=3,p=1",
            lambda: cutile_depthwise_conv2d(cx, cdw, padding=1),
            lambda: F.conv2d(cx, cdw, padding=1, groups=64),
            atol=0.05,
        )
        _bench_op(
            "conv2d_3x3",
            "vision",
            "16x64x56x56->128,k=3,p=1",
            lambda: cutile_conv2d(cx, cw3, padding=1),
            lambda: F.conv2d(cx, cw3, padding=1),
            atol=0.05,
        )
        _bench_op(
            "max_pool2d",
            "vision",
            "16x64x56x56,k=2,s=2",
            lambda: cutile_max_pool2d(cx, 2, 2),
            lambda: F.max_pool2d(cx, 2, 2),
            atol=1e-4,
        )
        _bench_op(
            "avg_pool2d",
            "vision",
            "16x64x56x56,k=2,s=2",
            lambda: cutile_avg_pool2d(cx, 2, 2),
            lambda: F.avg_pool2d(cx, 2, 2),
            atol=1e-2,
        )

    # 3. Reductions
    if family in ("all", "reductions"):
        for rows, cols in [(256, 2048), (4096, 4096), (16384, 1024)]:
            rx = torch.randn((rows, cols), device="cuda", dtype=dtype) * 0.05
            _bench_op(
                f"reduce_sum_{rows}x{cols}",
                "reductions",
                f"{rows}x{cols}",
                lambda rx=rx: cutile_reduce_sum(rx, -1),
                lambda rx=rx: rx.sum(dim=-1),
                atol=0.1,
            )
        for rows, cols in [(256, 2048), (4096, 4096)]:
            rx = torch.randn((rows, cols), device="cuda", dtype=dtype)
            _bench_op(
                f"reduce_max_{rows}x{cols}",
                "reductions",
                f"{rows}x{cols}",
                lambda rx=rx: cutile_reduce_max(rx, -1),
                lambda rx=rx: rx.max(dim=-1).values,
                atol=1e-3,
            )

    # 4. Activations, Gather, Transpose, BMM, Fused LN+GELU
    if family in ("all", "elementwise"):
        ax = torch.randn((1024, 4096), device="cuda", dtype=dtype)
        _bench_op(
            "gelu_1024x4096",
            "elementwise",
            "1024x4096",
            lambda: cutile_gelu(ax),
            lambda: F.gelu(ax, approximate="tanh"),
            atol=1e-2,
        )
        _bench_op(
            "silu_1024x4096",
            "elementwise",
            "1024x4096",
            lambda: cutile_silu(ax),
            lambda: F.silu(ax),
            atol=1e-2,
        )

        tbl = torch.randn((32000, 4096), device="cuda", dtype=dtype)
        idx = torch.randint(0, 32000, (1024,), device="cuda", dtype=torch.long)
        _bench_op(
            "gather_1024x4096",
            "elementwise",
            "1024x4096(32000)",
            lambda: cutile_gather(tbl, idx),
            lambda: F.embedding(idx, tbl),
            atol=1e-4,
        )

        for tr_m, tr_n in [(4096, 4096), (16384, 1024)]:
            tx = torch.randn((tr_m, tr_n), device="cuda", dtype=dtype)
            _bench_op(
                f"transpose_{tr_m}x{tr_n}",
                "elementwise",
                f"{tr_m}x{tr_n}",
                lambda tx=tx: cutile_transpose(tx, 0, 1),
                lambda tx=tx: tx.transpose(0, 1).contiguous(),
                atol=1e-4,
            )

        b1_a = torch.randn((32, 512, 512), device="cuda", dtype=dtype) * 0.05
        b1_b = torch.randn((32, 512, 512), device="cuda", dtype=dtype) * 0.05
        _bench_op(
            "bmm_32x512",
            "elementwise",
            "32x512x512x512",
            lambda: cutile_bmm(b1_a, b1_b),
            lambda: torch.bmm(b1_a, b1_b),
            atol=0.05,
        )

        b2_a = torch.randn((8, 256, 1024), device="cuda", dtype=dtype) * 0.05
        b2_b = torch.randn((8, 1024, 1024), device="cuda", dtype=dtype) * 0.05
        _bench_op(
            "bmm_8x256x1024",
            "elementwise",
            "8x256x1024x1024",
            lambda: cutile_bmm(b2_a, b2_b),
            lambda: torch.bmm(b2_a, b2_b),
            atol=0.05,
        )

        nx = torch.randn((1024, 4096), device="cuda", dtype=dtype) * 0.2
        nw = torch.ones((4096,), device="cuda", dtype=dtype)
        nb = torch.zeros((4096,), device="cuda", dtype=dtype)
        _bench_op(
            "fused_layernorm_gelu",
            "elementwise",
            "1024x4096",
            lambda: cutile_fused_layernorm_gelu(nx, nw, nb),
            lambda: F.gelu(F.layer_norm(nx, (4096,), nw, nb, 1e-5), approximate="tanh"),
            atol=1e-2,
        )

    return results


def run_mistral_e2e(model_dir: Path, runs: int = 5) -> dict[str, Any]:
    """Measure real-checkpoint Mistral-7B-v0.1-GPTQ 512-tok prefill & 256-tok decode."""
    from cutile_forge.engine.config import GPUConfig
    from cutile_forge.engine.model import Mistral7B
    from cutile_forge.engine.weight_loader import load_gptq_weights
    from cutile_forge.runtime import NativeModelRuntime

    weights = load_gptq_weights(model_dir, device="cuda", pin_ddr5=False)
    model = Mistral7B(
        weights=weights, gpu_config=GPUConfig(vram_gb=32.0, stream_weights=False), device="cuda"
    )
    rt = NativeModelRuntime(
        model,
        max_seq_len=1024,
        kv_dtype="fp16",
        use_cuda_graph=True,
        graph_strategy="bucketed",
        max_batch_size=1,
    )

    base_prompt = [
        1,
        415,
        5565,
        302,
        4843,
        349,
        1280,
        315,
        2296,
        396,
        17204,
        15262,
        369,
        1047,
        368,
        2563,
    ]
    prefill_prompt = (base_prompt * 32)[:512]
    prefill_t = torch.tensor([prefill_prompt], device="cuda", dtype=torch.long)

    # Warmup prefill + 256-token decode graph capture
    rt.generate(base_prompt, max_new_tokens=256, sync_interval=16)
    rt.reset()
    rt.prefill(prefill_t)
    torch.cuda.synchronize()

    prefill_ms_runs: list[float] = []
    for _ in range(runs):
        rt.reset()
        torch.cuda.synchronize()
        ev0 = torch.cuda.Event(enable_timing=True)
        ev1 = torch.cuda.Event(enable_timing=True)
        ev0.record()
        _ = rt.prefill(prefill_t)
        ev1.record()
        torch.cuda.synchronize()
        prefill_ms_runs.append(float(ev0.elapsed_time(ev1)))

    decode_tok_s_runs: list[float] = []
    decode_itl_runs: list[float] = []
    generated_ids: list[int] = []
    for i in range(runs):
        rt.reset()
        torch.cuda.synchronize()
        toks, stats = rt.generate(base_prompt, max_new_tokens=256, sync_interval=16)
        torch.cuda.synchronize()
        decode_tok_s_runs.append(float(stats["decode_tok_s"]))
        decode_itl_runs.append(1000.0 / max(1e-6, float(stats["decode_tok_s"])))
        if i == 0:
            generated_ids = [int(t) for t in toks]

    tok_hash = hashlib.sha256(json.dumps(generated_ids).encode()).hexdigest()[:16]
    return {
        "checkpoint": str(model_dir),
        "prefill_tokens": 512,
        "decode_tokens": 256,
        "runs": runs,
        "prefill_med_ms": round(statistics.median(prefill_ms_runs), 3),
        "prefill_min_ms": round(min(prefill_ms_runs), 3),
        "prefill_max_ms": round(max(prefill_ms_runs), 3),
        "decode_tok_s_med": round(statistics.median(decode_tok_s_runs), 2),
        "decode_tok_s_min": round(min(decode_tok_s_runs), 2),
        "decode_tok_s_max": round(max(decode_tok_s_runs), 2),
        "decode_itl_med_ms": round(statistics.median(decode_itl_runs), 3),
        "greedy_256_sha256_prefix": tok_hash,
        "greedy_256_first16": generated_ids[:16],
        "greedy_256_tokens": generated_ids,
        "provenance": "real-checkpoint",
    }


def _write_slow_ops_md(rows: list[dict[str, Any]], out_path: Path) -> None:
    slow = sorted([r for r in rows if r["speedup"] < 1.0], key=lambda r: r["speedup"])
    lines = [
        "# Phase 17 Slow Operators Sorted by Speedup Gap (CuTile vs PyTorch)",
        "",
        "| Rank | Operator | Shape | CuTile med (ms) | PyTorch med (ms) | Speedup (PT/CT) | Max Abs Diff | Provenance |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for idx, r in enumerate(slow, start=1):
        lines.append(
            f"| {idx} | `{r['op']}` | `{r['shape']}` | {r['cutile_med_ms']:.4f} "
            f"[{r['cutile_min_ms']:.4f}..{r['cutile_max_ms']:.4f}] | "
            f"{r['torch_med_ms']:.4f} [{r['torch_min_ms']:.4f}..{r['torch_max_ms']:.4f}] | "
            f"**{r['speedup']:.2f}x** | {r['max_abs_diff']:.5f} | `{r['provenance']}` |"
        )
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_comparison_md(
    base_data: dict[str, Any],
    opt_data: dict[str, Any],
    out_path: Path,
) -> None:
    base_ops = {r["op"]: r for r in base_data["ops"]}
    lines = [
        "# Phase 17 Kernel Optimization Before vs After Comparison",
        "",
        f"- Baseline commit: `{base_data.get('git_commit', 'unknown')}` | Contended: `{base_data.get('contended', False)}`",
        f"- Optimized commit: `{opt_data.get('git_commit', 'unknown')}` | Contended: `{opt_data.get('contended', False)}`",
        "",
        "| Operator | Shape | Base CT (ms) | Opt CT (ms) | Delta | PT Base (ms) | Opt Speedup (PT/CT) | Acc OK |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for r in opt_data["ops"]:
        b = base_ops.get(r["op"])
        b_ms = b["cutile_med_ms"] if b else float("nan")
        delta_pct = ((b_ms - r["cutile_med_ms"]) / b_ms * 100.0) if b and b_ms > 0 else 0.0
        sign = "+" if delta_pct >= 0 else ""
        lines.append(
            f"| `{r['op']}` | `{r['shape']}` | {b_ms:.4f} | "
            f"{r['cutile_med_ms']:.4f} [{r['cutile_min_ms']:.4f}..{r['cutile_max_ms']:.4f}] | "
            f"{sign}{delta_pct:.1f}% | {r['torch_med_ms']:.4f} | **{r['speedup']:.2f}x** | {r['accuracy_ok']} |"
        )

    if base_data.get("mistral_e2e") and opt_data.get("mistral_e2e"):
        bm = base_data["mistral_e2e"]
        om = opt_data["mistral_e2e"]
        parity_exact = bm.get("greedy_256_tokens") == om.get("greedy_256_tokens")
        lines.extend(
            [
                "",
                "## End-to-End Mistral-7B-v0.1-GPTQ (`real-checkpoint`)",
                "",
                f"- Prefill (512 tok): {bm['prefill_med_ms']:.2f} ms -> **{om['prefill_med_ms']:.2f} ms** [{om['prefill_min_ms']:.2f}..{om['prefill_max_ms']:.2f}]",
                f"- Decode (256 tok): {bm['decode_tok_s_med']:.2f} tok/s -> **{om['decode_tok_s_med']:.2f} tok/s** [{om['decode_tok_s_min']:.2f}..{om['decode_tok_s_max']:.2f}]",
                f"- Greedy 256-tok parity vs Phase 16 baseline: `{parity_exact}` (sha256 `{bm.get('greedy_256_sha256_prefix')}` vs `{om.get('greedy_256_sha256_prefix')}`)",
            ]
        )

    still_slow = sorted(
        [r for r in opt_data["ops"] if r["speedup"] < 1.0], key=lambda r: r["speedup"]
    )
    lines.extend(["", "## Operators Still Slower Than PyTorch Baseline", ""])
    for r in still_slow:
        lines.append(
            f"- `{r['op']}` (`{r['shape']}`): {r['cutile_med_ms']:.4f} ms vs PyTorch {r['torch_med_ms']:.4f} ms (**{r['speedup']:.2f}x**)"
        )
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 17 Kernel Optimization Benchmark")
    parser.add_argument("--runs", type=int, default=5)
    parser.add_argument("--iters", type=int, default=50)
    parser.add_argument(
        "--family",
        type=str,
        default="all",
        choices=["all", "gemm", "vision", "reductions", "elementwise"],
    )
    parser.add_argument("--e2e-mistral", action="store_true")
    parser.add_argument(
        "--model-dir", type=Path, default=Path("/home/arrhat/models/mistral_7b_4bit")
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("benchmarks/validation_artifacts/v3.4.0/phase17_baseline"),
    )
    parser.add_argument("--compare-baseline", type=Path, default=None)
    args = parser.parse_args()

    contended, gpu_apps = _query_contention()
    print(f"[Phase 17] GPU compute apps: {gpu_apps} (contended={contended})")

    model_dir = args.model_dir
    ckpt = (
        build_checkpoint_manifest(
            model_id="TheBloke/Mistral-7B-v0.1-GPTQ",
            local_path=model_dir,
        )
        if model_dir.exists()
        else build_checkpoint_manifest(
            model_id="synthetic-micro",
            local_path="__synthetic__",
        )
    )
    run = ValidationRun(
        run_dir=args.output,
        run_id=f"phase17-{args.output.name}",
        checkpoint=ckpt,
        quant_config=QuantizationConfig(
            quant_format="gptq_int4_and_fp16", group_size=128, kv_dtype="fp16"
        ),
        workload=WorkloadConfig(
            batch_size=1,
            concurrency=1,
            prompt_len=512,
            output_len=256,
            warmup_iters=10,
            measured_iters=args.runs,
        ),
        prompt_ids_or_checksum=prompt_checksum("phase17_micro_and_mistral"),
        execution_mode="real-checkpoint" if args.e2e_mistral else "synthetic-weights",
        reproduction_status="MEASURED",
    )

    ops = run_micro_ops(runs=args.runs, iters=args.iters, family=args.family)
    for r in ops:
        run.add_sample(f"{r['op']}_cutile_ms", r["cutile_med_ms"], "ms")
        run.add_sample(f"{r['op']}_torch_ms", r["torch_med_ms"], "ms")
        run.add_record({"stage": "micro_op", **r})
        print(
            f"  {r['op']:<22s} | CT={r['cutile_med_ms']:.4f}ms [{r['cutile_min_ms']:.4f}..{r['cutile_max_ms']:.4f}] | "
            f"PT={r['torch_med_ms']:.4f}ms | speedup={r['speedup']:.2f}x | max_diff={r['max_abs_diff']:.5f} ok={r['accuracy_ok']}"
        )

    mistral_e2e: dict[str, Any] | None = None
    if args.e2e_mistral and model_dir.exists():
        mistral_e2e = run_mistral_e2e(model_dir, runs=args.runs)
        run.add_sample("mistral_prefill_512_ms", mistral_e2e["prefill_med_ms"], "ms")
        run.add_sample("mistral_decode_256_tok_s", mistral_e2e["decode_tok_s_med"], "tok/s")
        run.add_record({"stage": "mistral_e2e", **mistral_e2e})
        print(
            f"[Phase 17 Mistral E2E] prefill512={mistral_e2e['prefill_med_ms']:.2f}ms "
            f"[{mistral_e2e['prefill_min_ms']:.2f}..{mistral_e2e['prefill_max_ms']:.2f}] | "
            f"decode256={mistral_e2e['decode_tok_s_med']:.2f} tok/s "
            f"[{mistral_e2e['decode_tok_s_min']:.2f}..{mistral_e2e['decode_tok_s_max']:.2f}] | "
            f"sha={mistral_e2e['greedy_256_sha256_prefix']}"
        )

    run.finalize()
    payload = {
        "git_commit": run._git_commit,
        "contended": contended,
        "gpu_apps": gpu_apps,
        "ops": ops,
        "mistral_e2e": mistral_e2e,
    }
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "phase17_ops.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    _write_slow_ops_md(ops, args.output / "slow_ops_sorted.md")

    if args.compare_baseline is not None:
        base_json = args.compare_baseline / "phase17_ops.json"
        if base_json.exists():
            base_data = json.loads(base_json.read_text(encoding="utf-8"))
            _write_comparison_md(base_data, payload, args.output / "comparison.md")

    errs = validate_artifact_dir(args.output)
    if errs:
        raise RuntimeError(f"Validation artifact check failed: {errs}")


if __name__ == "__main__":
    main()
