"""Phase 16 Step 1 & 2: Serving Architecture Baseline & HTTP Gap Attribution."""

from __future__ import annotations

import argparse
import asyncio
import gc
import json
import os
import statistics
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import httpx
import numpy as np
import torch

from benchmarks.production.bench_reliability_faults import (
    _free_port,
    _stop_owned_server,
    _wait_health,
)
from benchmarks.production.manifest import build_checkpoint_manifest, prompt_checksum
from benchmarks.production.runner import ValidationRun
from benchmarks.production.schema import QuantizationConfig, WorkloadConfig, validate_artifact_dir

PROMPTS: tuple[str, ...] = (
    "Explain how GPU Tensor Cores accelerate matrix multiplication in simple steps.",
    "The capital city of France is Paris, and its history includes",
    "In parallel systems programming, memory bandwidth limits arise when",
    "Write a concise summary of virtual memory paging for GPU caches:",
)


def _measure_in_process_baseline(
    model_dir: Path,
    run: ValidationRun,
    concurrencies: tuple[int, ...] = (1, 4, 16, 64),
    max_new_tokens: int = 64,
    runs_per_level: int = 5,
) -> dict[int, dict[str, Any]]:
    from transformers import AutoTokenizer

    from cutile_forge.engine.config import GPUConfig
    from cutile_forge.engine.model import Mistral7B
    from cutile_forge.engine.weight_loader import load_gptq_weights
    from cutile_forge.runtime import NativeModelRuntime

    tok = AutoTokenizer.from_pretrained(str(model_dir), local_files_only=True)
    weights = load_gptq_weights(model_dir, device="cuda", pin_ddr5=False)
    model = Mistral7B(
        weights=weights, gpu_config=GPUConfig(vram_gb=32.0, stream_weights=False), device="cuda"
    )
    rt = NativeModelRuntime(
        model,
        max_seq_len=512,
        kv_dtype="fp16",
        use_cuda_graph=True,
        max_batch_size=max(concurrencies),
    )

    encoded_prompts = [
        [int(x) for x in tok.encode(f"user: {p}\nassistant:", add_special_tokens=True)]
        for p in PROMPTS
    ]

    # Tokenizer attribution (100 iterations)
    tok_enc_ms: list[float] = []
    tok_dec_ms: list[float] = []
    sample_str = f"user: {PROMPTS[0]}\nassistant:"
    sample_ids = list(range(100, 100 + max_new_tokens))
    for _ in range(100):
        t0 = time.perf_counter()
        _ = tok.encode(sample_str, return_tensors="pt")
        tok_enc_ms.append((time.perf_counter() - t0) * 1000.0)
        t1 = time.perf_counter()
        _ = tok.decode(sample_ids, skip_special_tokens=True)
        tok_dec_ms.append((time.perf_counter() - t1) * 1000.0)

    tok_attr = {
        "encode_med_ms": round(statistics.median(tok_enc_ms), 4),
        "encode_min_ms": round(min(tok_enc_ms), 4),
        "encode_max_ms": round(max(tok_enc_ms), 4),
        "decode_64tok_med_ms": round(statistics.median(tok_dec_ms), 4),
        "decode_64tok_min_ms": round(min(tok_dec_ms), 4),
        "decode_64tok_max_ms": round(max(tok_dec_ms), 4),
    }
    run.add_record({"stage": "tokenizer_attribution", "metrics": tok_attr})

    inproc_summary: dict[int, dict[str, Any]] = {}
    for conc in concurrencies:
        batch_prompts = [encoded_prompts[i % len(encoded_prompts)] for i in range(conc)]
        # Warmup (1 run, includes CUDA graph capture for batch=conc)
        _, _ = rt.generate_batch(batch_prompts, max_new_tokens=max_new_tokens)
        torch.cuda.synchronize()

        agg_rates: list[float] = []
        ttft_list: list[float] = []
        itl_list: list[float] = []
        wall_ms_list: list[float] = []

        for it in range(runs_per_level):
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            outs, st = rt.generate_batch(batch_prompts, max_new_tokens=max_new_tokens)
            torch.cuda.synchronize()
            wall_ms = (time.perf_counter() - t0) * 1000.0
            tot_toks = sum(len(o) for o in outs)
            agg_tps = tot_toks / max(1e-6, wall_ms / 1000.0)
            prefill_ms = float(st.get("prefill_ms", 0.0))
            decode_ms = float(st.get("decode_ms", max(0.0, wall_ms - prefill_ms)))
            itl_ms = decode_ms / max(1, max_new_tokens - 1)

            agg_rates.append(agg_tps)
            ttft_list.append(prefill_ms)
            itl_list.append(itl_ms)
            wall_ms_list.append(wall_ms)

            run.add_sample(f"inproc_c{conc}_agg_tok_s", agg_tps, "tok/s", iteration=it)
            run.add_sample(f"inproc_c{conc}_ttft_ms", prefill_ms, "ms", iteration=it)
            run.add_sample(f"inproc_c{conc}_itl_ms", itl_ms, "ms", iteration=it)

        rec = {
            "stage": f"inproc_c{conc}",
            "mode": "in_process",
            "concurrency": conc,
            "true_batch_size": conc,
            "runs": runs_per_level,
            "agg_tok_s_med": round(statistics.median(agg_rates), 2),
            "agg_tok_s_min": round(min(agg_rates), 2),
            "agg_tok_s_max": round(max(agg_rates), 2),
            "ttft_p50_ms": round(float(np.percentile(ttft_list, 50)), 2),
            "ttft_p90_ms": round(float(np.percentile(ttft_list, 90)), 2),
            "ttft_p99_ms": round(float(np.percentile(ttft_list, 99)), 2),
            "itl_p50_ms": round(float(np.percentile(itl_list, 50)), 2),
            "itl_p90_ms": round(float(np.percentile(itl_list, 90)), 2),
            "itl_p99_ms": round(float(np.percentile(itl_list, 99)), 2),
            "wall_med_ms": round(statistics.median(wall_ms_list), 2),
            "provenance": "real-checkpoint",
        }
        inproc_summary[conc] = rec
        run.add_record(rec)
        run.log_stdout(
            f"[Phase 16 Baseline | IN-PROCESS | C={conc:2d}] "
            f"agg={rec['agg_tok_s_med']:.2f} tok/s [{rec['agg_tok_s_min']:.2f}..{rec['agg_tok_s_max']:.2f}] | "
            f"TTFT p50={rec['ttft_p50_ms']:.2f}ms | ITL p50={rec['itl_p50_ms']:.2f}ms | batch={conc}"
        )

    del rt, model, weights
    gc.collect()
    torch.cuda.empty_cache()
    return inproc_summary


async def _send_http_req(
    client: httpx.AsyncClient,
    base_url: str,
    prompt_text: str,
    max_tokens: int,
    stream: bool,
) -> dict[str, Any]:
    payload = {
        "model": "mistral-7b-4bit",
        "messages": [{"role": "user", "content": prompt_text}],
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "stream": stream,
    }
    t0 = time.perf_counter()
    prev_t = t0
    first_chunk_t: float | None = None
    chunk_itls: list[float] = []
    tokens_out = 0
    parts: list[str] = []

    if stream:
        async with client.stream(
            "POST", f"{base_url}/v1/chat/completions", json=payload, timeout=180.0
        ) as resp:
            if resp.status_code != 200:
                _ = await resp.aread()
                return {"ok": False, "status": resp.status_code, "tokens": 0}
            async for line in resp.aiter_lines():
                if not line.startswith("data: "):
                    continue
                body = line[6:].strip()
                if body == "[DONE]":
                    break
                now = time.perf_counter()
                obj = json.loads(body)
                choice = obj["choices"][0]
                delta = choice["delta"].get("content", "")
                if delta:
                    parts.append(delta)
                if choice.get("finish_reason") is None:
                    tokens_out += 1
                    if first_chunk_t is None:
                        first_chunk_t = now
                    else:
                        chunk_itls.append((now - prev_t) * 1000.0)
                    prev_t = now
        t_end = time.perf_counter()
        ttft_ms = ((first_chunk_t or t_end) - t0) * 1000.0
        e2e_ms = (t_end - t0) * 1000.0
        return {
            "ok": True,
            "tokens": tokens_out,
            "ttft_ms": ttft_ms,
            "itl_list_ms": chunk_itls,
            "e2e_ms": e2e_ms,
            "text": "".join(parts),
        }

    resp = await client.post(f"{base_url}/v1/chat/completions", json=payload, timeout=180.0)
    t_end = time.perf_counter()
    e2e_ms = (t_end - t0) * 1000.0
    if resp.status_code != 200:
        return {"ok": False, "status": resp.status_code, "tokens": 0, "e2e_ms": e2e_ms}
    data = resp.json()
    tokens_out = int(data["usage"]["completion_tokens"])
    return {
        "ok": True,
        "tokens": tokens_out,
        "ttft_ms": e2e_ms,
        "itl_list_ms": [],
        "e2e_ms": e2e_ms,
        "text": str(data["choices"][0]["message"]["content"]),
    }


async def _measure_http_baseline(
    model_dir: Path,
    run: ValidationRun,
    inproc_summary: dict[int, dict[str, Any]],
    concurrencies: tuple[int, ...] = (1, 4, 16, 64),
    max_new_tokens: int = 64,
    runs_per_level: int = 3,
    max_batch_size: int = 64,
    batch_wait_ms: float = 2.0,
    repro_idx: int = 0,
) -> dict[str, Any]:
    port = _free_port()
    base_url = f"http://127.0.0.1:{port}"
    cmd = [
        sys.executable,
        "-m",
        "cutile_forge.cli",
        "serve",
        "--model",
        str(model_dir),
        "--host",
        "127.0.0.1",
        "--port",
        str(port),
        "--native-runtime",
        "--max-concurrent",
        "64",
        "--max-batch-size",
        str(max_batch_size),
        "--batch-wait-ms",
        str(batch_wait_ms),
        "--max-prompt-tokens",
        "2048",
    ]
    env = dict(os.environ)
    env["CUTILE_ARCH"] = env.get("CUTILE_ARCH", "sm_120")
    run.run_dir.mkdir(parents=True, exist_ok=True)
    srv_out = (run.run_dir / "server_stdout.log").open("w", encoding="utf-8")
    srv_err = (run.run_dir / "server_stderr.log").open("w", encoding="utf-8")
    proc = subprocess.Popen(cmd, stdout=srv_out, stderr=srv_err, text=True, env=env)
    results_out: dict[str, Any] = {}

    try:
        boot_ms = _wait_health(base_url, proc, timeout_s=120.0)
        run.log_stdout(f"[Phase 16 Baseline] HTTP server booted in {boot_ms:.1f} ms")

        limits = httpx.Limits(max_connections=128, max_keepalive_connections=64)
        async with httpx.AsyncClient(limits=limits) as client:
            # Measure loopback /health roundtrip cost
            health_ms: list[float] = []
            for _ in range(25):
                th = time.perf_counter()
                r_h = await client.get(f"{base_url}/health", timeout=10.0)
                if r_h.status_code == 200:
                    health_ms.append((time.perf_counter() - th) * 1000.0)
            run.add_record(
                {
                    "stage": "http_loopback_rtt",
                    "health_med_ms": round(statistics.median(health_ms), 3),
                    "health_p99_ms": round(float(np.percentile(health_ms, 99)), 3),
                }
            )

            # Warmup both non-stream and stream paths (plus C=4,16,64 CUDA graph capture warmup)
            for c_warm in (1, 4, 16, 64):
                _ = await asyncio.gather(
                    *[
                        _send_http_req(
                            client,
                            base_url,
                            PROMPTS[i % len(PROMPTS)],
                            max_new_tokens,
                            False,
                        )
                        for i in range(c_warm)
                    ]
                )
            _ = await _send_http_req(client, base_url, PROMPTS[0], max_new_tokens, True)

            for stream_flag, mode_name in ((False, "http_nonstream"), (True, "http_stream")):
                for conc in concurrencies:
                    agg_runs: list[float] = []
                    ttfts_all: list[float] = []
                    itls_all: list[float] = []
                    e2e_all: list[float] = []
                    qwait_means: list[float] = []
                    qwait_maxes: list[float] = []
                    batch_means: list[float] = []
                    batch_maxes: list[int] = []
                    ok_total = 0
                    req_total = 0

                    for it in range(runs_per_level):
                        m0 = (await client.get(f"{base_url}/metrics", timeout=10.0)).json()
                        t0 = time.perf_counter()
                        resps = await asyncio.gather(
                            *[
                                _send_http_req(
                                    client,
                                    base_url,
                                    PROMPTS[i % len(PROMPTS)],
                                    max_new_tokens,
                                    stream_flag,
                                )
                                for i in range(conc)
                            ]
                        )
                        wall_s = time.perf_counter() - t0
                        m1 = (await client.get(f"{base_url}/metrics", timeout=10.0)).json()

                        oks = [r for r in resps if r.get("ok")]
                        ok_total += len(oks)
                        req_total += len(resps)
                        tot_toks = sum(int(r.get("tokens", 0)) for r in oks)
                        agg_tps = tot_toks / max(1e-6, wall_s)
                        agg_runs.append(agg_tps)

                        for r in oks:
                            ttfts_all.append(float(r["ttft_ms"]))
                            e2e_all.append(float(r["e2e_ms"]))
                            if r.get("itl_list_ms"):
                                itls_all.extend(float(x) for x in r["itl_list_ms"])
                            else:
                                itls_all.append(
                                    float(r["e2e_ms"]) / max(1, int(r.get("tokens", 1)))
                                )

                        d_batches = int(m1.get("batches_executed", 0)) - int(
                            m0.get("batches_executed", 0)
                        )
                        d_bsum = int(m1.get("batch_size_sum", 0)) - int(m0.get("batch_size_sum", 0))
                        d_qsum = float(m1.get("queue_wait_ms_sum", 0.0)) - float(
                            m0.get("queue_wait_ms_sum", 0.0)
                        )
                        if d_batches > 0:
                            batch_means.append(d_bsum / d_batches)
                            batch_maxes.append(int(m1.get("batch_size_max", 1)))
                            qwait_means.append(d_qsum / max(1, d_bsum))
                            qwait_maxes.append(float(m1.get("queue_wait_ms_max", 0.0)))
                        else:
                            batch_means.append(1.0)
                            batch_maxes.append(1)
                            qwait_means.append(0.0)
                            qwait_maxes.append(0.0)

                        run.add_sample(
                            f"run{repro_idx}_{mode_name}_c{conc}_agg_tok_s",
                            agg_tps,
                            "tok/s",
                            iteration=it,
                        )

                    inproc_med = inproc_summary[conc]["agg_tok_s_med"]
                    med_tps = statistics.median(agg_runs)
                    ratio_pct = 100.0 * med_tps / max(1e-6, inproc_med)

                    rec = {
                        "stage": f"run{repro_idx}_{mode_name}_c{conc}",
                        "repro_run": repro_idx,
                        "mode": mode_name,
                        "concurrency": conc,
                        "requests_ok": ok_total,
                        "requests_sent": req_total,
                        "runs": runs_per_level,
                        "agg_tok_s_med": round(med_tps, 2),
                        "agg_tok_s_min": round(min(agg_runs), 2),
                        "agg_tok_s_max": round(max(agg_runs), 2),
                        "vs_inproc_pct": round(ratio_pct, 2),
                        "ttft_p50_ms": round(float(np.percentile(ttfts_all, 50)), 2),
                        "ttft_p90_ms": round(float(np.percentile(ttfts_all, 90)), 2),
                        "ttft_p99_ms": round(float(np.percentile(ttfts_all, 99)), 2),
                        "itl_p50_ms": round(float(np.percentile(itls_all, 50)), 2),
                        "itl_p90_ms": round(float(np.percentile(itls_all, 90)), 2),
                        "itl_p99_ms": round(float(np.percentile(itls_all, 99)), 2),
                        "e2e_p50_ms": round(float(np.percentile(e2e_all, 50)), 2),
                        "e2e_p99_ms": round(float(np.percentile(e2e_all, 99)), 2),
                        "queue_wait_mean_ms": round(statistics.mean(qwait_means), 2),
                        "true_batch_size_mean": round(statistics.mean(batch_means), 2),
                        "true_batch_size_max": max(batch_maxes),
                        "provenance": "real-checkpoint",
                    }
                    results_out[f"{mode_name}_c{conc}"] = rec
                    run.add_record(rec)
                    run.log_stdout(
                        f"[Phase 16 Run {repro_idx} | {mode_name:14s} | C={conc:2d}] "
                        f"agg={rec['agg_tok_s_med']:.2f} tok/s ({rec['vs_inproc_pct']:.1f}% of in-proc) "
                        f"[{rec['agg_tok_s_min']:.2f}..{rec['agg_tok_s_max']:.2f}] | "
                        f"TTFT p50/p99={rec['ttft_p50_ms']:.1f}/{rec['ttft_p99_ms']:.1f}ms | "
                        f"ITL p50/p99={rec['itl_p50_ms']:.1f}/{rec['itl_p99_ms']:.1f}ms | "
                        f"qwait={rec['queue_wait_mean_ms']:.2f}ms | batch={rec['true_batch_size_mean']:.2f}"
                    )

            # Request Isolation Check: mixed prompt lengths standalone C=1 vs concurrent C=4 and C=16
            mixed_prompts_4 = [
                "Hi.",
                " ".join(["NVIDIA Blackwell CUDA Tile fusion"] * 40),
                "The capital of Japan is",
                "def quicksort(arr):\n    if len(arr) <= 1:\n        return arr\n",
            ]
            mixed_prompts_16 = [
                f"{mixed_prompts_4[i % 4]} (req {i})" if i >= 4 else mixed_prompts_4[i]
                for i in range(16)
            ]
            solo_outs_16 = [
                (await _send_http_req(client, base_url, p, 32, i % 2 == 1))["text"]
                for i, p in enumerate(mixed_prompts_16)
            ]
            conc_resps_4 = await asyncio.gather(
                *[
                    _send_http_req(client, base_url, p, 32, i % 2 == 1)
                    for i, p in enumerate(mixed_prompts_4)
                ]
            )
            iso_matches_4 = [
                s == r["text"] for s, r in zip(solo_outs_16[:4], conc_resps_4, strict=True)
            ]
            conc_resps_16 = await asyncio.gather(
                *[
                    _send_http_req(client, base_url, p, 32, i % 2 == 1)
                    for i, p in enumerate(mixed_prompts_16)
                ]
            )
            iso_matches_16 = [
                s == r["text"] for s, r in zip(solo_outs_16, conc_resps_16, strict=True)
            ]
            run.add_record(
                {
                    "stage": f"run{repro_idx}_request_isolation",
                    "c4_matches": iso_matches_4,
                    "c4_exact_match_count": sum(1 for x in iso_matches_4 if x),
                    "c16_exact_match_count": sum(1 for x in iso_matches_16 if x),
                    "c16_total": len(mixed_prompts_16),
                }
            )
            run.log_stdout(
                f"[Phase 16 Run {repro_idx} | ISOLATION] C=4: {sum(iso_matches_4)}/4 | C=16: {sum(iso_matches_16)}/16"
            )

            # Cancellation / Mid-stream Disconnect Test
            m_pre = (await client.get(f"{base_url}/metrics", timeout=10.0)).json()
            try:
                async with client.stream(
                    "POST",
                    f"{base_url}/v1/chat/completions",
                    json={
                        "model": "mistral-7b-4bit",
                        "messages": [{"role": "user", "content": PROMPTS[0]}],
                        "max_tokens": 64,
                        "temperature": 0.0,
                        "stream": True,
                    },
                    timeout=30.0,
                ) as resp:
                    seen = 0
                    async for line in resp.aiter_lines():
                        if line.startswith("data: "):
                            seen += 1
                            if seen >= 3:
                                break
            except Exception:
                pass
            await asyncio.sleep(0.5)
            m_post = (await client.get(f"{base_url}/metrics", timeout=10.0)).json()
            disc_delta = int(m_post.get("client_disconnects_total", 0)) - int(
                m_pre.get("client_disconnects_total", 0)
            )
            run.add_record(
                {
                    "stage": f"run{repro_idx}_disconnect_check",
                    "inflight_after": m_post.get("requests_inflight"),
                    "active_kv_slots_after": m_post.get("active_kv_slots"),
                    "disconnects_delta": disc_delta,
                    "vram_alloc_mb": m_post.get("peak_vram_allocated_mb"),
                    "rss_mb": m_post.get("host_rss_mb"),
                }
            )
            run.log_stdout(
                f"[Phase 16 Run {repro_idx} | DISCONNECT] inflight_after={m_post.get('requests_inflight')}, "
                f"active_kv_slots={m_post.get('active_kv_slots')}, disconnects_delta={disc_delta}"
            )
    finally:
        _stop_owned_server(proc)
        srv_out.close()
        srv_err.close()

    return results_out


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 16 Serving Baseline & Reproducibility")
    parser.add_argument(
        "--model-dir", type=Path, default=Path("/home/arrhat/models/mistral_7b_4bit")
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("benchmarks/validation_artifacts/v3.3.0/phase16_optimized"),
    )
    parser.add_argument("--max-tokens", type=int, default=64)
    parser.add_argument("--max-batch-size", type=int, default=64)
    parser.add_argument("--batch-wait-ms", type=float, default=2.0)
    parser.add_argument("--repro-runs", type=int, default=3)
    args = parser.parse_args()

    ckpt = build_checkpoint_manifest("TheBloke/Mistral-7B-v0.1-GPTQ", args.model_dir)
    run = ValidationRun(
        run_dir=args.output,
        run_id="phase16_serving_validation",
        checkpoint=ckpt,
        quant_config=QuantizationConfig(quant_format="gptq", group_size=128, desc_act=True),
        workload=WorkloadConfig(
            batch_size=args.max_batch_size,
            concurrency=64,
            prompt_len=16,
            output_len=args.max_tokens,
            temperature=0.0,
            seed=42,
            warmup_iters=1,
            measured_iters=5,
            sync_method="torch.cuda.synchronize + wall_clock_httpx",
        ),
        prompt_ids_or_checksum=prompt_checksum("phase16_serving_validation"),
        execution_mode="real-checkpoint",
        reproduction_status="MEASURED",
        notes=f"Phase 16 continuous-batching validation ({args.repro_runs} independent server runs).",
    )

    inproc = _measure_in_process_baseline(args.model_dir, run, max_new_tokens=args.max_tokens)
    per_run_results: list[dict[str, Any]] = []
    for r_idx in range(args.repro_runs):
        res_r = asyncio.run(
            _measure_http_baseline(
                args.model_dir,
                run,
                inproc,
                max_new_tokens=args.max_tokens,
                max_batch_size=args.max_batch_size,
                batch_wait_ms=args.batch_wait_ms,
                repro_idx=r_idx,
            )
        )
        per_run_results.append(res_r)

    if per_run_results:
        keys = list(per_run_results[0].keys())
        repro_agg: dict[str, Any] = {}
        for k in keys:
            tps_vals = [float(r[k]["agg_tok_s_med"]) for r in per_run_results]
            pct_vals = [float(r[k]["vs_inproc_pct"]) for r in per_run_results]
            ttft99_vals = [float(r[k]["ttft_p99_ms"]) for r in per_run_results]
            itl99_vals = [float(r[k]["itl_p99_ms"]) for r in per_run_results]
            tps_mean = statistics.mean(tps_vals)
            tps_std = statistics.stdev(tps_vals) if len(tps_vals) > 1 else 0.0
            repro_agg[k] = {
                "agg_tok_s_med": round(statistics.median(tps_vals), 2),
                "agg_tok_s_min": round(min(tps_vals), 2),
                "agg_tok_s_max": round(max(tps_vals), 2),
                "agg_tok_s_cv_pct": round(100.0 * tps_std / max(1e-6, tps_mean), 2),
                "vs_inproc_pct_med": round(statistics.median(pct_vals), 2),
                "ttft_p99_med_ms": round(statistics.median(ttft99_vals), 2),
                "ttft_p99_max_ms": round(max(ttft99_vals), 2),
                "itl_p99_med_ms": round(statistics.median(itl99_vals), 2),
                "itl_p99_max_ms": round(max(itl99_vals), 2),
            }
            run.log_stdout(
                f"[REPRO SUMMARY {k:18s}] tok/s={repro_agg[k]['agg_tok_s_med']:.2f} "
                f"[{repro_agg[k]['agg_tok_s_min']:.2f}..{repro_agg[k]['agg_tok_s_max']:.2f}] "
                f"(CV={repro_agg[k]['agg_tok_s_cv_pct']:.2f}%, {repro_agg[k]['vs_inproc_pct_med']:.1f}% in-proc) | "
                f"TTFT p99 med={repro_agg[k]['ttft_p99_med_ms']:.1f}ms | ITL p99 med={repro_agg[k]['itl_p99_med_ms']:.1f}ms"
            )
        run.add_record({"stage": "reproducibility_summary_3runs", "metrics": repro_agg})

    out = run.finalize()
    errs = validate_artifact_dir(out)
    assert errs == [], f"Phase 16 artifact validation failed: {errs}"


if __name__ == "__main__":
    main()
