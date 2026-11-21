"""Phase 9 comprehensive benchmark: Before vs. After across all Priority 1 Native Runtime items."""

from __future__ import annotations

import json
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
from cutile_forge.engine.speculative import SpeculativeDecoder
from cutile_forge.kernels.attention_gqa import flash_decoding_split_kv, fused_attention_gqa
from cutile_forge.runtime.native_engine import NativeModelRuntime
from cutile_forge.runtime.native_kv_cache import (
    NativePagedKVCache,
    cutile_kv_cache_gather_hm_kernel,
    cutile_kv_cache_write,
)

FIVE_PROMPTS: list[list[int]] = [
    [1, 415, 5565, 302, 4843, 349],
    [1, 315, 2296, 396, 17204, 15262, 369],
    [1, 1047, 368, 2563, 264, 1401, 287],
    [1, 28705, 28740, 28725, 28705, 28750, 28725, 28705, 28770],
    [1, 450, 1296, 302, 272, 9060, 10865, 349],
]


def _query_active_gpu_apps() -> tuple[str, bool]:
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-compute-apps=pid,used_memory", "--format=csv,noheader"],
            text=True,
        ).strip()
    except Exception as exc:  # pragma: no cover
        return f"error:{exc}", False
    if not out:
        return "none", False
    import os

    my_pid = str(os.getpid())
    lines = [ln.strip() for ln in out.splitlines() if ln.strip()]
    others = [ln for ln in lines if not ln.startswith(my_pid)]
    return "; ".join(lines), len(others) > 0


def run_phase9_benchmarks(
    model_dir: Path | None = None,
    output_csv: Path = Path("benchmarks/results_5090/phase9_native_runtime.csv"),
    num_runs: int = 5,
    max_new_tokens: int = 256,
) -> list[dict[str, object]]:
    if model_dir is None:
        model_dir = Path.home() / "models" / "mistral_7b_4bit"
    if not torch.cuda.is_available() or not model_dir.exists():
        raise RuntimeError(f"CUDA + checkpoint {model_dir} required")

    apps_pre, contended_pre = _query_active_gpu_apps()
    print(f"[Pre-benchmark nvidia-smi compute apps] {apps_pre} (contended={contended_pre})")

    baseline_json = Path("benchmarks/results_5090/phase9_diagnosis.json")
    base_data: dict[str, Any] = (
        json.loads(baseline_json.read_text(encoding="utf-8")) if baseline_json.exists() else {}
    )
    base_lookup = {r["mode"]: r for r in base_data.get("baseline_rows", [])}
    v230_def_tps = float(
        base_lookup.get("NativeModelRuntime_v230_default", {}).get("decode_tok_s_median", 103.22)
    )
    v230_eag_tps = float(
        base_lookup.get("NativeModelRuntime_v230_eager", {}).get("decode_tok_s_median", 103.67)
    )
    v230_pf_ms = float(
        base_lookup.get("NativeModelRuntime_v230_default", {}).get("prefill_ms_median", 10.14)
    )

    model = Mistral7B.from_pretrained(
        model_dir,
        gpu_config=GPUConfig(vram_gb=32.0, stream_weights=False),
        device="cuda",
        max_gpu_layers=32,
    )
    prompt_ids = FIVE_PROMPTS[0]
    prompt_t = torch.tensor([prompt_ids], device="cuda", dtype=torch.long)

    rows: list[dict[str, object]] = []

    def _add_row(
        item: str,
        metric_unit: str,
        before_val: float | str,
        vals: list[float],
        notes: str = "",
        parity: str = "True",
    ) -> None:
        apps_now, cont_now = _query_active_gpu_apps()
        rows.append(
            {
                "item": item,
                "label": "real-checkpoint",
                "metric_unit": metric_unit,
                "before_v230": before_val,
                "after_median": round(statistics.median(vals), 2),
                "after_min": round(min(vals), 2),
                "after_max": round(max(vals), 2),
                "runs": len(vals),
                "parity": parity,
                "contended": contended_pre or cont_now,
                "gpu_compute_apps": apps_now,
                "notes": notes,
            }
        )

    # ------------------------------------------------------------------
    # 1. Legacy reference (256 tokens, 5 runs) + 5-prompt 256-token parity check
    # ------------------------------------------------------------------
    eng_graph = InferenceEngine(
        model, target="sm_120", use_cuda_graph=True, enable_radix_cache=False
    )
    _ = eng_graph.generate_ids(prompt_t, max_new_tokens=16, temperature=0.0, backend="cutile")
    leg_tps: list[float] = []
    leg_toks_p0: list[int] = []
    for r in range(num_runs):
        toks = eng_graph.generate_ids(
            prompt_t, max_new_tokens=max_new_tokens, temperature=0.0, backend="cutile"
        )
        assert eng_graph.last_stats is not None
        leg_tps.append(eng_graph.last_stats.tokens_per_sec)
        if r == 0:
            leg_toks_p0 = list(toks)
    _add_row(
        "legacy_InferenceEngine_cuda_graph_256tok",
        "tok/s",
        99.01,
        leg_tps,
        notes="Legacy engine (66 forbidden ATen calls/decode)",
    )

    # ------------------------------------------------------------------
    # 2. Step 9.1 Strategy B: Native Bucketed CUDA Graph + Chained Replay (sync_interval=32 vs 1)
    # ------------------------------------------------------------------
    rt_bkt = NativeModelRuntime(
        model,
        max_seq_len=2048,
        use_cuda_graph=True,
        graph_strategy="bucketed",
        context_buckets=(512, 1024, 2048),
    )
    _, _ = rt_bkt.generate(prompt_ids, max_new_tokens=16, sync_interval=16)

    bkt_s32_tps, bkt_s32_pf, bkt_s32_ms = [], [], []
    bkt_toks_p0: list[int] = []
    for r in range(num_runs):
        toks, st = rt_bkt.generate(prompt_ids, max_new_tokens=max_new_tokens, sync_interval=32)
        bkt_s32_tps.append(float(st["decode_tok_s"]))
        bkt_s32_pf.append(float(st["prefill_ms"]))
        bkt_s32_ms.append(float(st["ms_per_token"]))
        if r == 0:
            bkt_toks_p0 = list(toks)

    bkt_s1_tps: list[float] = []
    for _ in range(num_runs):
        _, st1 = rt_bkt.generate(prompt_ids, max_new_tokens=max_new_tokens, sync_interval=1)
        bkt_s1_tps.append(float(st1["decode_tok_s"]))

    _add_row(
        "native_full_graph_bucketed_chained_sync32_256tok",
        "tok/s",
        v230_def_tps,
        bkt_s32_tps,
        notes="1 full-step CUDAGraph/token, 0 ATen, 0 eager launches, sync_interval=32",
        parity=str(bkt_toks_p0 == leg_toks_p0),
    )
    _add_row(
        "native_full_graph_bucketed_step_sync1_256tok",
        "tok/s",
        v230_def_tps,
        bkt_s1_tps,
        notes="1 full-step CUDAGraph/token, per-step host sync (sync_interval=1)",
        parity=str(bkt_toks_p0 == leg_toks_p0),
    )
    _add_row(
        "native_prefill_latency",
        "ms",
        v230_pf_ms,
        bkt_s32_pf,
        notes="Packed QKV + fused SwiGLU native prefill",
        parity=str(bkt_toks_p0 == leg_toks_p0),
    )

    # ------------------------------------------------------------------
    # 3. Step 9.1 Strategy A: Single Fixed-Ceiling Device-Masked CUDA Graph (max_seq_len=2048)
    # ------------------------------------------------------------------
    rt_msk = NativeModelRuntime(
        model, max_seq_len=2048, use_cuda_graph=True, graph_strategy="device_masked"
    )
    _, _ = rt_msk.generate(prompt_ids, max_new_tokens=16, sync_interval=16)
    msk_tps: list[float] = []
    msk_toks_p0: list[int] = []
    for r in range(num_runs):
        toks, st = rt_msk.generate(prompt_ids, max_new_tokens=max_new_tokens, sync_interval=32)
        msk_tps.append(float(st["decode_tok_s"]))
        if r == 0:
            msk_toks_p0 = list(toks)
    _add_row(
        "native_full_graph_device_masked_2048_sync32_256tok",
        "tok/s",
        v230_def_tps,
        msk_tps,
        notes="Single fixed-ceiling (2048) device-masked graph (Step 9.1 comparison)",
        parity=str(msk_toks_p0 == leg_toks_p0),
    )

    # ------------------------------------------------------------------
    # 4. Native Eager (use_cuda_graph=False) with packed QKV + fused SwiGLU + narrow-N tiles
    # ------------------------------------------------------------------
    rt_eag = NativeModelRuntime(model, max_seq_len=2048, use_cuda_graph=False)
    _, _ = rt_eag.generate(prompt_ids, max_new_tokens=16)
    eag_tps: list[float] = []
    eag_toks_p0: list[int] = []
    for r in range(num_runs):
        toks, st = rt_eag.generate(prompt_ids, max_new_tokens=max_new_tokens)
        eag_tps.append(float(st["decode_tok_s"]))
        if r == 0:
            eag_toks_p0 = list(toks)
    _add_row(
        "native_eager_optimized_256tok",
        "tok/s",
        v230_eag_tps,
        eag_tps,
        notes="Eager NativeModelRuntime with packed QKV, fused SwiGLU, (16,32,128) tiles",
        parity=str(eag_toks_p0 == leg_toks_p0),
    )

    # ------------------------------------------------------------------
    # 5. 5-Prompt × 256-Token Greedy Parity Audit (Legacy vs Native Eager vs Native Graphed)
    # ------------------------------------------------------------------
    matched_prompts = 0
    for p_ids in FIVE_PROMPTS:
        pt = torch.tensor([p_ids], device="cuda", dtype=torch.long)
        t_l = eng_graph.generate_ids(pt, max_new_tokens=256, temperature=0.0, backend="cutile")
        t_g, _ = rt_bkt.generate(p_ids, max_new_tokens=256, sync_interval=32)
        t_e, _ = rt_eag.generate(p_ids, max_new_tokens=256)
        if t_l == t_g and t_l == t_e:
            matched_prompts += 1
    _add_row(
        "greedy_parity_256tok_5_prompts",
        "prompts_matched",
        1.0,
        [float(matched_prompts)] * num_runs,
        notes=f"{matched_prompts}/5 prompts token-exact over 256 tokens (legacy == eager == graphed)",
        parity=str(matched_prompts == 5),
    )

    # ------------------------------------------------------------------
    # 6. Step 9.2: Prefill/Decode Interleave Scheduling (512-tok prefill arriving mid-decode)
    # ------------------------------------------------------------------
    long_pf_ids = [1] + [415, 5565, 302, 4843, 349, 28723, 315, 2296] * 64  # 513 tokens
    mono_max_itl, chunk_max_itl = [], []
    for _ in range(num_runs):
        st_m = rt_bkt.benchmark_prefill_decode_interleave(
            prompt_ids, long_pf_ids[:512], decode_steps=6, chunk_size=None
        )
        st_c = rt_bkt.benchmark_prefill_decode_interleave(
            prompt_ids, long_pf_ids[:512], decode_steps=6, chunk_size=128
        )
        mono_max_itl.append(st_m["max_itl_ms"])
        chunk_max_itl.append(st_c["max_itl_ms"])
    _add_row(
        "prefill_decode_interleave_max_stall_latency_512tok",
        "ms",
        round(statistics.median(mono_max_itl), 2),
        chunk_max_itl,
        notes="Max active-decode inter-token stall during 512-tok prefill: monolithic vs chunk_size=128",
    )

    # ------------------------------------------------------------------
    # 7. Step 9.2: Continuous Batching on Native Runtime (Batch=4, 64 new tokens)
    # ------------------------------------------------------------------
    batch_prompts = FIVE_PROMPTS[:4]
    seq_b4_tps: list[float] = []
    for _ in range(num_runs):
        t0 = time.perf_counter()
        for bp in batch_prompts:
            _, _ = rt_bkt.generate(bp, max_new_tokens=64, sync_interval=32)
        torch.cuda.synchronize()
        el_s = time.perf_counter() - t0
        seq_b4_tps.append((4 * 63) / el_s)

    rt_batch = NativeModelRuntime(
        model,
        max_seq_len=512,
        use_cuda_graph=True,
        graph_strategy="bucketed",
        context_buckets=(256, 512),
        max_batch_size=4,
    )
    _, _ = rt_batch.generate_batch(batch_prompts, max_new_tokens=8, sync_interval=4)
    batched_b4_tps: list[float] = []
    for _ in range(num_runs):
        _, bst = rt_batch.generate_batch(batch_prompts, max_new_tokens=64, sync_interval=32)
        batched_b4_tps.append(float(bst["total_decode_tok_s"]))
    _add_row(
        "continuous_batching_b4_aggregate_throughput",
        "tok/s",
        round(statistics.median(seq_b4_tps), 2),
        batched_b4_tps,
        notes="4 concurrent requests (64 tokens each): sequential B=1x4 vs batch-bucketed B=4 CUDAGraph",
    )

    # ------------------------------------------------------------------
    # 8. Step 9.2: Speculative Decoding on Native Runtime (Standard & Structured/Repetitive Prompts)
    # ------------------------------------------------------------------
    leg_spec = SpeculativeDecoder(model, gamma=4)
    _ = leg_spec.generate(prompt_t, max_new_tokens=16)

    # (a) Standard prompt (Prompt 1, 256 tokens)
    leg_spec_std_tps: list[float] = []
    leg_spec_std_acc = 0.0
    for _ in range(num_runs):
        _, l_st = leg_spec.generate(prompt_t, max_new_tokens=256)
        leg_spec_std_tps.append(l_st.tokens_per_sec)
        leg_spec_std_acc = l_st.acceptance_rate

    nat_spec_std_tps: list[float] = []
    nat_spec_std_acc = 0.0
    nat_spec_std_toks: list[int] = []
    for r in range(num_runs):
        stoks, n_st = rt_bkt.generate_speculative(prompt_ids, max_new_tokens=256, gamma=4)
        nat_spec_std_tps.append(float(n_st["decode_tok_s"]))
        nat_spec_std_acc = float(n_st["acceptance_rate"])
        if r == 0:
            nat_spec_std_toks = list(stoks)
    _add_row(
        "speculative_native_gamma4_standard_prompt_256tok",
        "tok/s",
        round(statistics.median(leg_spec_std_tps), 2),
        nat_spec_std_tps,
        notes=(
            f"Standard prompt: native accept_rate={nat_spec_std_acc:.3f} "
            f"(legacy_spec={statistics.median(leg_spec_std_tps):.2f} tok/s, "
            f"accept={leg_spec_std_acc:.3f}; native_graph={statistics.median(bkt_s32_tps):.2f} tok/s)"
        ),
        parity=str(nat_spec_std_toks == leg_toks_p0),
    )

    # (b) Structured/clause-repeating prompt where N-gram lookup has high acceptance
    rep_prompt_ids = [
        1,
        415,
        5565,
        302,
        4843,
        349,
        5765,
        28723,
        415,
        5565,
        302,
        4843,
        349,
        5765,
        28723,
        415,
        5565,
        302,
        4843,
    ]
    rep_t = torch.tensor([rep_prompt_ids], device="cuda", dtype=torch.long)
    ref_rep_toks, _ = rt_bkt.generate(rep_prompt_ids, max_new_tokens=256, sync_interval=32)

    leg_spec_rep_tps: list[float] = []
    leg_spec_rep_acc = 0.0
    for _ in range(num_runs):
        _, l_st = leg_spec.generate(rep_t, max_new_tokens=256)
        leg_spec_rep_tps.append(l_st.tokens_per_sec)
        leg_spec_rep_acc = l_st.acceptance_rate

    nat_spec_rep_tps: list[float] = []
    nat_spec_rep_acc = 0.0
    nat_spec_rep_toks: list[int] = []
    for r in range(num_runs):
        stoks, n_st = rt_bkt.generate_speculative(rep_prompt_ids, max_new_tokens=256, gamma=4)
        nat_spec_rep_tps.append(float(n_st["decode_tok_s"]))
        nat_spec_rep_acc = float(n_st["acceptance_rate"])
        if r == 0:
            nat_spec_rep_toks = list(stoks)
    _add_row(
        "speculative_native_gamma4_ngram_match_prompt_256tok",
        "tok/s",
        round(statistics.median(leg_spec_rep_tps), 2),
        nat_spec_rep_tps,
        notes=(
            f"N-gram match prompt: native accept_rate={nat_spec_rep_acc:.3f} "
            f"(legacy_spec={statistics.median(leg_spec_rep_tps):.2f} tok/s, "
            f"accept={leg_spec_rep_acc:.3f}; native_graph={statistics.median(bkt_s32_tps):.2f} tok/s)"
        ),
        parity=str(nat_spec_rep_toks == ref_rep_toks),
    )

    # ------------------------------------------------------------------
    # 9. Step 9.2: Paged KV-Cache + FlashDecoding 32-Layer Step Latency at 8K, 16K, 32K
    # ------------------------------------------------------------------
    # Compare Before (v2.3.0: per-layer kv_gather copy + dynamic allocs) vs After (v2.4.0: zero-gather
    # head-major KV + CuTile flash_decoding_reduce_kernel) across 32 layers
    del rt_batch, rt_msk, rt_eag, eng_graph, leg_spec
    torch.cuda.empty_cache()

    h_q, h_kv, d_head, n_layers = 32, 8, 128, 32
    q_step = torch.randn((1, h_q, 1, d_head), device="cuda", dtype=torch.float16) * 0.1
    k_new = torch.randn((1, h_kv, 1, d_head), device="cuda", dtype=torch.float16) * 0.1
    v_new = torch.randn((1, h_kv, 1, d_head), device="cuda", dtype=torch.float16) * 0.1

    def _bench_long_ctx_pair(ctx_len: int) -> tuple[list[float], list[float]]:
        kv_long = NativePagedKVCache(
            num_layers=1,
            num_kv_heads=h_kv,
            head_dim=d_head,
            block_size=16,
            max_seq_len=ctx_len,
            kv_dtype="fp16",
            device="cuda",
        )
        kv_long.k_pools[0].normal_(0, 0.05)
        kv_long.v_pools[0].normal_(0, 0.05)
        kv_long.seq_lens[0] = ctx_len - 1
        k_work = torch.empty((h_kv, ctx_len, d_head), device="cuda", dtype=torch.float16)
        v_work = torch.empty((h_kv, ctx_len, d_head), device="cuda", dtype=torch.float16)
        o_part_buf = torch.empty((h_q, 16, 16, d_head), device="cuda", dtype=torch.float32)
        lse_part_buf = torch.empty((h_q, 16, 16, 1), device="cuda", dtype=torch.float32)
        out_buf = torch.empty((1, h_q, 1, d_head), device="cuda", dtype=torch.float16)

        def _old_32layer_step() -> None:
            for _ in range(n_layers):
                slots = kv_long._linear_slots[ctx_len - 1 : ctx_len]
                cutile_kv_cache_write(k_new, v_new, kv_long.k_pools[0], kv_long.v_pools[0], slots)
                ct.launch(
                    torch.cuda.current_stream(),
                    (ct.cdiv(ctx_len, 16), h_kv, 1),
                    cutile_kv_cache_gather_hm_kernel,
                    (
                        kv_long.k_pools[0].view(h_kv, ctx_len, d_head),
                        kv_long.v_pools[0].view(h_kv, ctx_len, d_head),
                        kv_long.block_tables[0],
                        k_work,
                        v_work,
                        ctx_len,
                        h_kv,
                        16,
                        16,
                        d_head,
                    ),
                )
                _ = fused_attention_gqa(
                    q_step, k_work.unsqueeze(0), v_work.unsqueeze(0), causal=True
                )

        def _new_32layer_step() -> None:
            for _ in range(n_layers):
                slots = kv_long._linear_slots[ctx_len - 1 : ctx_len]
                cutile_kv_cache_write(k_new, v_new, kv_long.k_pools[0], kv_long.v_pools[0], slots)
                kf, vf = kv_long.get_kv(0, end_pos=ctx_len, batch_size=1)
                _ = flash_decoding_split_kv(
                    q_step,
                    kf,
                    vf,
                    num_splits=16,
                    causal=True,
                    o_part_buf=o_part_buf,
                    lse_part_buf=lse_part_buf,
                    out=out_buf,
                )

        _old_32layer_step()
        _new_32layer_step()
        torch.cuda.synchronize()

        old_ms_list, new_ms_list = [], []
        s_ev = torch.cuda.Event(enable_timing=True)
        e_ev = torch.cuda.Event(enable_timing=True)
        for _ in range(num_runs):
            s_ev.record()
            _old_32layer_step()
            e_ev.record()
            torch.cuda.synchronize()
            old_ms_list.append(float(s_ev.elapsed_time(e_ev)))

            s_ev.record()
            _new_32layer_step()
            e_ev.record()
            torch.cuda.synchronize()
            new_ms_list.append(float(s_ev.elapsed_time(e_ev)))
        return old_ms_list, new_ms_list

    for ctx_len in (8192, 16384, 32768):
        old_ms_list, new_ms_list = _bench_long_ctx_pair(ctx_len)
        _add_row(
            f"paged_kv_attention_32layers_{ctx_len // 1024}k_step_latency",
            "ms/step",
            round(statistics.median(old_ms_list), 2),
            new_ms_list,
            notes=f"{ctx_len}-token KV cache (32 layers): full-history kv_gather vs zero-copy head-major pool",
        )
        torch.cuda.empty_cache()

    write_csv(output_csv, rows)
    for r in rows:
        print(r)
    return rows


if __name__ == "__main__":
    run_phase9_benchmarks()
