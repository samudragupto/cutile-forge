"""Phase 9 Step 9.0: 256-token baseline & per-token diagnostic breakdown on real Mistral-7B-4bit."""

from __future__ import annotations

import cProfile
import json
import pstats
import statistics
import subprocess
import time
from pathlib import Path
from typing import Any

import torch

from cutile_forge.bench.harness import write_csv
from cutile_forge.cuda_tile_init import ct
from cutile_forge.engine.config import GPUConfig
from cutile_forge.engine.inference import InferenceEngine
from cutile_forge.engine.model import Mistral7B
from cutile_forge.runtime.native_engine import NativeModelRuntime


def _query_active_gpu_apps() -> tuple[str, bool]:
    """Return active compute apps from nvidia-smi and whether the GPU is contended."""
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-compute-apps=pid,used_memory", "--format=csv,noheader"],
            text=True,
        ).strip()
    except Exception as exc:  # pragma: no cover
        return f"error:{exc}", False
    if not out:
        return "none", False
    lines = [ln.strip() for ln in out.splitlines() if ln.strip()]
    import os

    my_pid = str(os.getpid())
    others = [ln for ln in lines if not ln.startswith(my_pid)]
    return "; ".join(lines), len(others) > 0


def _measure_single_step_breakdown(
    step_fn: Any,
    n_steps: int = 32,
) -> dict[str, Any]:
    """Measure per-token kernel launches, allocations, CUDA event time, wall time, and Python cProfile overhead."""
    # Count ct.launch calls and CUDA tensor allocations by hooking ct.launch & torch.cuda.memory_stats
    orig_launch = ct.launch
    launch_count = 0

    def _counted_launch(*args: Any, **kwargs: Any) -> Any:
        nonlocal launch_count
        launch_count += 1
        return orig_launch(*args, **kwargs)

    # Warmup 2 steps
    for i in range(2):
        step_fn(i)
    torch.cuda.synchronize()

    # 1. Count ct.launch & memory allocator calls over n_steps
    ct.launch = _counted_launch  # type: ignore[assignment]
    mem_before = torch.cuda.memory_stats()
    alloc_before = int(mem_before.get("allocation.all.allocated", 0))
    try:
        for i in range(n_steps):
            step_fn(i)
        torch.cuda.synchronize()
    finally:
        ct.launch = orig_launch  # type: ignore[assignment]
    mem_after = torch.cuda.memory_stats()
    alloc_after = int(mem_after.get("allocation.all.allocated", 0))

    ct_launches_per_tok = launch_count / float(n_steps)
    allocs_per_tok = (alloc_after - alloc_before) / float(n_steps)

    # 2. Measure GPU kernel time (CUDA events) vs wall-clock time without per-step synchronize
    start_ev = torch.cuda.Event(enable_timing=True)
    end_ev = torch.cuda.Event(enable_timing=True)
    torch.cuda.synchronize()
    t_wall0 = time.perf_counter()
    start_ev.record()
    for i in range(n_steps):
        step_fn(i)
    end_ev.record()
    t_dispatch_ms = (time.perf_counter() - t_wall0) * 1000.0
    torch.cuda.synchronize()
    t_wall_ms = (time.perf_counter() - t_wall0) * 1000.0
    t_gpu_ms = float(start_ev.elapsed_time(end_ev))

    # 3. cProfile Python CPU overhead per token
    prof = cProfile.Profile()
    torch.cuda.synchronize()
    prof.enable()
    for i in range(n_steps):
        step_fn(i)
    prof.disable()
    torch.cuda.synchronize()
    ps = pstats.Stats(prof)
    py_cpu_ms_per_tok = (ps.total_tt * 1000.0) / float(n_steps)  # type: ignore[attr-defined]

    return {
        "ct_launches_per_token": round(ct_launches_per_tok, 1),
        "tensor_allocs_per_token": round(allocs_per_tok, 1),
        "gpu_event_ms_per_token": round(t_gpu_ms / n_steps, 3),
        "wall_ms_per_token": round(t_wall_ms / n_steps, 3),
        "host_dispatch_ms_per_token": round(t_dispatch_ms / n_steps, 3),
        "cprofile_py_ms_per_token": round(py_cpu_ms_per_tok, 3),
    }


def run_phase9_baseline(
    model_dir: Path | None = None,
    output_csv: Path = Path("benchmarks/results_5090/phase9_baseline.csv"),
    diag_json: Path = Path("benchmarks/results_5090/phase9_diagnosis.json"),
    max_new_tokens: int = 256,
    num_runs: int = 5,
) -> list[dict[str, object]]:
    if model_dir is None:
        model_dir = Path.home() / "models" / "mistral_7b_4bit"
    if not torch.cuda.is_available() or not model_dir.exists():
        raise RuntimeError(f"CUDA + checkpoint {model_dir} required")

    gpu_apps_pre, contended_pre = _query_active_gpu_apps()
    print(f"[Pre-benchmark nvidia-smi compute apps] {gpu_apps_pre} (contended={contended_pre})")

    model = Mistral7B.from_pretrained(
        model_dir,
        gpu_config=GPUConfig(vram_gb=32.0, stream_weights=False),
        device="cuda",
        max_gpu_layers=32,
    )

    prompt_ids = [1, 415, 5565, 302, 4843, 349]  # <s> The capital of France is
    prompt_t = torch.tensor([prompt_ids], device="cuda", dtype=torch.long)

    rows: list[dict[str, object]] = []
    tokens_by_mode: dict[str, list[int]] = {}

    # ------------------------------------------------------------------
    # (a) Legacy InferenceEngine with CUDA Graph
    # ------------------------------------------------------------------
    eng_graph = InferenceEngine(
        model, target="sm_120", use_cuda_graph=True, enable_radix_cache=False
    )
    _ = eng_graph.generate_ids(prompt_t, max_new_tokens=16, temperature=0.0, backend="cutile")
    tps_a, pf_a, ms_a = [], [], []
    for r in range(num_runs):
        torch.cuda.reset_peak_memory_stats()
        toks = eng_graph.generate_ids(
            prompt_t, max_new_tokens=max_new_tokens, temperature=0.0, backend="cutile"
        )
        st = eng_graph.last_stats
        assert st is not None
        tps_a.append(st.tokens_per_sec)
        pf_a.append(st.prefill_ms)
        ms_a.append(st.ms_per_token)
        if r == 0:
            tokens_by_mode["legacy_cuda_graph"] = list(toks)
    vram_a = round(torch.cuda.max_memory_allocated() / (1024**3), 2)

    # ------------------------------------------------------------------
    # (b) Legacy InferenceEngine Eager
    # ------------------------------------------------------------------
    eng_eager = InferenceEngine(
        model, target="sm_120", use_cuda_graph=False, enable_radix_cache=False
    )
    _ = eng_eager.generate_ids(prompt_t, max_new_tokens=16, temperature=0.0, backend="cutile")
    tps_b, pf_b, ms_b = [], [], []
    for r in range(num_runs):
        torch.cuda.reset_peak_memory_stats()
        toks = eng_eager.generate_ids(
            prompt_t, max_new_tokens=max_new_tokens, temperature=0.0, backend="cutile"
        )
        st = eng_eager.last_stats
        assert st is not None
        tps_b.append(st.tokens_per_sec)
        pf_b.append(st.prefill_ms)
        ms_b.append(st.ms_per_token)
        if r == 0:
            tokens_by_mode["legacy_eager"] = list(toks)
    vram_b = round(torch.cuda.max_memory_allocated() / (1024**3), 2)

    # ------------------------------------------------------------------
    # (c1) NativeModelRuntime as it exists today (use_cuda_graph=True default)
    # ------------------------------------------------------------------
    rt_cur = NativeModelRuntime(model, max_seq_len=1024, kv_dtype="fp16", use_cuda_graph=True)
    _, _ = rt_cur.generate(prompt_ids, max_new_tokens=16)
    tps_c1, pf_c1, ms_c1 = [], [], []
    for r in range(num_runs):
        torch.cuda.reset_peak_memory_stats()
        toks, st_d = rt_cur.generate(prompt_ids, max_new_tokens=max_new_tokens)
        tps_c1.append(float(st_d["decode_tok_s"]))
        pf_c1.append(float(st_d["prefill_ms"]))
        ms_c1.append(float(st_d["ms_per_token"]))
        if r == 0:
            tokens_by_mode["native_v230_partial_graph"] = list(toks)
    vram_c1 = round(torch.cuda.max_memory_allocated() / (1024**3), 2)

    # ------------------------------------------------------------------
    # (c2) NativeModelRuntime pure eager (use_cuda_graph=False)
    # ------------------------------------------------------------------
    rt_eag = NativeModelRuntime(model, max_seq_len=1024, kv_dtype="fp16", use_cuda_graph=False)
    _, _ = rt_eag.generate(prompt_ids, max_new_tokens=16)
    tps_c2, pf_c2, ms_c2 = [], [], []
    for r in range(num_runs):
        torch.cuda.reset_peak_memory_stats()
        toks, st_d = rt_eag.generate(prompt_ids, max_new_tokens=max_new_tokens)
        tps_c2.append(float(st_d["decode_tok_s"]))
        pf_c2.append(float(st_d["prefill_ms"]))
        ms_c2.append(float(st_d["ms_per_token"]))
        if r == 0:
            tokens_by_mode["native_v230_eager"] = list(toks)
    vram_c2 = round(torch.cuda.max_memory_allocated() / (1024**3), 2)

    gpu_apps_post, contended_post = _query_active_gpu_apps()
    contended = contended_pre or contended_post

    for mode_name, tps_list, ms_list, pf_list, vram_val in [
        ("legacy_InferenceEngine_cuda_graph", tps_a, ms_a, pf_a, vram_a),
        ("legacy_InferenceEngine_eager", tps_b, ms_b, pf_b, vram_b),
        ("NativeModelRuntime_v230_default", tps_c1, ms_c1, pf_c1, vram_c1),
        ("NativeModelRuntime_v230_eager", tps_c2, ms_c2, pf_c2, vram_c2),
    ]:
        rows.append(
            {
                "mode": mode_name,
                "label": "real-checkpoint",
                "checkpoint": "Mistral-7B-v0.1-GPTQ",
                "max_new_tokens": max_new_tokens,
                "runs": num_runs,
                "decode_tok_s_median": round(statistics.median(tps_list), 2),
                "decode_tok_s_min": round(min(tps_list), 2),
                "decode_tok_s_max": round(max(tps_list), 2),
                "ms_per_token_median": round(statistics.median(ms_list), 2),
                "ms_per_token_min": round(min(ms_list), 2),
                "ms_per_token_max": round(max(ms_list), 2),
                "prefill_ms_median": round(statistics.median(pf_list), 2),
                "prefill_ms_min": round(min(pf_list), 2),
                "prefill_ms_max": round(max(pf_list), 2),
                "peak_vram_gb": vram_val,
                "parity_256_vs_legacy_graph": (
                    tokens_by_mode.get(
                        mode_name.replace("legacy_InferenceEngine_", "legacy_")
                        .replace("NativeModelRuntime_v230_default", "native_v230_partial_graph")
                        .replace("NativeModelRuntime_v230_eager", "native_v230_eager"),
                        [],
                    )
                    == tokens_by_mode["legacy_cuda_graph"]
                ),
                "contended": contended,
                "gpu_compute_apps": gpu_apps_post,
            }
        )

    write_csv(output_csv, rows)

    # ------------------------------------------------------------------
    # Per-token diagnostic breakdown for (a), (b), (c1), (c2)
    # ------------------------------------------------------------------
    _ = eng_graph.prefill(prompt_t, backend="cutile")
    tok_in = torch.tensor([[295]], device="cuda", dtype=torch.long)
    diag_a = _measure_single_step_breakdown(
        lambda i: eng_graph.decode_step(tok_in, pos=len(prompt_ids) + (i % 8), backend="cutile")
    )
    diag_a["host_syncs_per_token_in_generate"] = 1  # int(next_tok.item()) per token

    _ = eng_eager.prefill(prompt_t, backend="cutile")
    diag_b = _measure_single_step_breakdown(
        lambda i: eng_eager.decode_step(tok_in, pos=len(prompt_ids) + (i % 8), backend="cutile")
    )
    diag_b["host_syncs_per_token_in_generate"] = 1

    _ = rt_cur.prefill(prompt_t)
    diag_c1 = _measure_single_step_breakdown(
        lambda i: rt_cur.decode_step(tok_in, pos=len(prompt_ids) + (i % 8), use_graph=True)
    )
    diag_c1["host_syncs_per_token_in_generate"] = 1  # int(next_tok.item()) in generate loop

    _ = rt_eag.prefill(prompt_t)
    diag_c2 = _measure_single_step_breakdown(
        lambda i: rt_eag.decode_step(tok_in, pos=len(prompt_ids) + (i % 8), use_graph=False)
    )
    diag_c2["host_syncs_per_token_in_generate"] = 1

    # Also measure component latencies inside NativeModelRuntime decode_step (32 layers)
    torch.cuda.synchronize()
    s_ev = torch.cuda.Event(enable_timing=True)
    e_ev = torch.cuda.Event(enable_timing=True)

    def _time_fn(fn: Any, iters: int = 20) -> float:
        for _ in range(3):
            fn()
        torch.cuda.synchronize()
        s_ev.record()
        for _ in range(iters):
            fn()
        e_ev.record()
        torch.cuda.synchronize()
        return round(float(s_ev.elapsed_time(e_ev)) / iters, 3)

    qkv_32_ms = _time_fn(lambda: [rt_cur._qkv_graphs[idx].replay() for idx in range(32)])
    mlp_32_ms = _time_fn(lambda: [rt_cur._mlp_graphs[idx].replay() for idx in range(32)])

    def _run_32_rope_kv_attn() -> None:
        for idx in range(32):
            q = rt_cur._static_q[idx]
            k = rt_cur._static_k[idx]
            v = rt_cur._static_v[idx]
            from cutile_forge.kernels.attention_gqa import fused_attention_gqa
            from cutile_forge.kernels.rope import apply_rope

            q = apply_rope(q, rt_cur._cos, rt_cur._sin, position_offset=16)
            k = apply_rope(k, rt_cur._cos, rt_cur._sin, position_offset=16)
            kf, vf = rt_cur.kv_cache.append_and_get(idx, k, v, start_pos=16)
            _ = fused_attention_gqa(q, kf, vf, causal=True)

    rope_kv_attn_32_ms = _time_fn(_run_32_rope_kv_attn)
    from cutile_forge.kernels.embedding_lmhead import cutile_lm_head_fused

    lm_head_ms = _time_fn(
        lambda: cutile_lm_head_fused(
            rt_cur._static_hidden,
            rt_cur._norm_w,
            rt_cur._lm_head_w,
            eps=rt_cur.config.rms_norm_eps,
            top_k=1,
        )
    )

    diag_payload = {
        "baseline_rows": rows,
        "step_breakdown": {
            "legacy_cuda_graph": diag_a,
            "legacy_eager": diag_b,
            "native_v230_default_partial_graph": diag_c1,
            "native_v230_eager": diag_c2,
        },
        "native_v230_component_ms_per_token": {
            "qkv_graphs_32_layers_ms": qkv_32_ms,
            "mlp_graphs_32_layers_ms": mlp_32_ms,
            "eager_rope_kv_write_gather_attn_32_layers_ms": rope_kv_attn_32_ms,
            "eager_cutile_lm_head_fused_ms": lm_head_ms,
        },
    }
    diag_json.parent.mkdir(parents=True, exist_ok=True)
    diag_json.write_text(json.dumps(diag_payload, indent=2), encoding="utf-8")
    print(json.dumps(diag_payload, indent=2))
    return rows


if __name__ == "__main__":
    run_phase9_baseline()
