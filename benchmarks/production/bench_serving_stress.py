"""Phase 14F: Multi-User OpenAI HTTP Serving Stress & Sustained Soak Suite."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import httpx
import numpy as np

from benchmarks.production.bench_reliability_faults import (
    _free_port,
    _stop_owned_server,
    _wait_health,
)
from benchmarks.production.manifest import build_checkpoint_manifest, prompt_checksum
from benchmarks.production.runner import ValidationRun, host_rss_mb
from benchmarks.production.schema import QuantizationConfig, WorkloadConfig, validate_artifact_dir


async def _send_one_client_req(
    client: httpx.AsyncClient,
    base_url: str,
    user_id: str,
    prompt_text: str,
    max_tokens: int,
    stream: bool,
    headers_extra: dict[str, str] | None = None,
) -> dict[str, Any]:
    headers = {"X-User-Id": user_id}
    if headers_extra:
        headers.update(headers_extra)
    payload = {
        "model": "mistral-7b-4bit",
        "messages": [{"role": "user", "content": prompt_text}],
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "stream": stream,
    }
    t0 = time.perf_counter()
    prev_t: float = t0
    first_chunk_t: float | None = None
    chunk_times: list[float] = []
    tokens_out = 0
    content_parts: list[str] = []
    status_code = 0

    try:
        if stream:
            async with client.stream(
                "POST",
                f"{base_url}/v1/chat/completions",
                headers=headers,
                json=payload,
                timeout=120.0,
            ) as resp:
                status_code = resp.status_code
                if status_code != 200:
                    _ = await resp.aread()
                    return {
                        "ok": False,
                        "status": status_code,
                        "e2e_ms": (time.perf_counter() - t0) * 1000.0,
                        "tokens": 0,
                        "stream": True,
                    }
                async for line in resp.aiter_lines():
                    if not line.startswith("data: "):
                        continue
                    body = line[6:].strip()
                    if body == "[DONE]":
                        break
                    now = time.perf_counter()
                    obj = json.loads(body)
                    delta = obj["choices"][0]["delta"].get("content", "")
                    if delta:
                        content_parts.append(delta)
                    if obj["choices"][0].get("finish_reason") is None:
                        tokens_out += 1
                        if first_chunk_t is None:
                            first_chunk_t = now
                        else:
                            chunk_times.append((now - prev_t) * 1000.0)
                        prev_t = now
            t_end = time.perf_counter()
            ttft_ms = ((first_chunk_t or t_end) - t0) * 1000.0
            e2e_ms = (t_end - t0) * 1000.0
        else:
            resp = await client.post(
                f"{base_url}/v1/chat/completions", headers=headers, json=payload, timeout=120.0
            )
            t_end = time.perf_counter()
            status_code = resp.status_code
            e2e_ms = (t_end - t0) * 1000.0
            if status_code != 200:
                return {
                    "ok": False,
                    "status": status_code,
                    "e2e_ms": e2e_ms,
                    "tokens": 0,
                    "stream": False,
                }
            data = resp.json()
            tokens_out = int(data["usage"]["completion_tokens"])
            content_parts.append(str(data["choices"][0]["message"]["content"]))
            ttft_ms = e2e_ms / max(1, tokens_out)

        return {
            "ok": status_code == 200,
            "status": status_code,
            "stream": stream,
            "ttft_ms": ttft_ms,
            "itl_list_ms": chunk_times,
            "e2e_ms": e2e_ms,
            "tpot_ms": e2e_ms / max(1, tokens_out),
            "tokens": tokens_out,
            "text": "".join(content_parts),
            "user_id": user_id,
        }
    except httpx.TimeoutException:
        return {
            "ok": False,
            "status": 408,
            "timeout": True,
            "e2e_ms": (time.perf_counter() - t0) * 1000.0,
            "tokens": 0,
        }
    except Exception:
        return {
            "ok": False,
            "status": 500,
            "e2e_ms": (time.perf_counter() - t0) * 1000.0,
            "tokens": 0,
        }


async def _run_concurrency_stage(
    client: httpx.AsyncClient,
    base_url: str,
    conc: int,
    short_prompt: str,
    long_prompt: str,
    shared_sys: str,
) -> list[dict[str, Any]]:
    tasks = []
    for i in range(conc):
        uid = f"user-{i % max(1, min(conc, 8))}"
        if i % 4 == 0:
            p = f"{shared_sys}\nUser {i}: Summarize GPU paged KV caching."
            mx = 16
            stream = True
        elif i % 4 == 1:
            p = long_prompt
            mx = 32
            stream = False
        elif i % 4 == 2:
            p = f"{shared_sys}\nUser {i}: Name the capital of France."
            mx = 16
            stream = False
        else:
            p = short_prompt
            mx = 24
            stream = True
        tasks.append(_send_one_client_req(client, base_url, uid, p, mx, stream))
    return await asyncio.gather(*tasks)


async def _run_stress_async(
    model_dir: Path,
    run: ValidationRun,
    soak_seconds: float,
    max_concurrent: int,
) -> None:
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
        str(max_concurrent),
        "--max-batch-size",
        "16",
        "--max-prompt-tokens",
        "2048",
    ]
    env = dict(os.environ)
    env["CUTILE_ARCH"] = env.get("CUTILE_ARCH", "sm_120")
    run.run_dir.mkdir(parents=True, exist_ok=True)
    srv_out_f = (run.run_dir / "server_stdout.log").open("w", encoding="utf-8")
    srv_err_f = (run.run_dir / "server_stderr.log").open("w", encoding="utf-8")
    proc = subprocess.Popen(cmd, stdout=srv_out_f, stderr=srv_err_f, text=True, env=env)

    short_p = "The capital of France is"
    long_p = " ".join(["NVIDIA Blackwell Tensor Core tile fusion"] * 60)
    shared_sys = "System: Answer concisely and factually using GPU engineering terminology."

    try:
        boot_ms = _wait_health(base_url, proc, timeout_s=120.0)
        run.log_stdout(f"[Phase 14F] Server pid={proc.pid} ready in {boot_ms:.1f}ms on {base_url}")

        limits = httpx.Limits(max_connections=256, max_keepalive_connections=128)
        async with httpx.AsyncClient(limits=limits) as client:
            # Warmup request
            _ = await _send_one_client_req(client, base_url, "warmup", short_p, 8, False)

            # 1. Concurrency sweep: 1, 4, 16, 64 (and 128 if max_concurrent >= 128)
            levels = [1, 4, 16, 64]
            if max_concurrent >= 128:
                levels.append(128)

            for conc in levels:
                t_c0 = time.perf_counter()
                res_list = await _run_concurrency_stage(
                    client, base_url, conc, short_p, long_p, shared_sys
                )
                wall_s = time.perf_counter() - t_c0
                met = (await client.get(f"{base_url}/metrics", timeout=10.0)).json()

                ok_reqs = [r for r in res_list if r.get("ok")]
                tot_toks = sum(int(r.get("tokens", 0)) for r in ok_reqs)
                ttfts = [float(r["ttft_ms"]) for r in ok_reqs] or [0.0]
                itls = (
                    [x for r in ok_reqs for x in r.get("itl_list_ms", [])]
                    or [float(r["tpot_ms"]) for r in ok_reqs]
                    or [0.0]
                )
                err_codes: dict[str, int] = {}
                for r in res_list:
                    if not r.get("ok"):
                        sc = str(r.get("status", "unknown"))
                        err_codes[sc] = err_codes.get(sc, 0) + 1

                agg_tok_s = tot_toks / max(1e-6, wall_s)
                per_req_tok_s = (
                    float(
                        np.median([r["tokens"] / max(1e-6, r["e2e_ms"] / 1000.0) for r in ok_reqs])
                    )
                    if ok_reqs
                    else 0.0
                )

                for i, v in enumerate(ttfts):
                    run.add_sample(f"c{conc}_ttft_ms", v, "ms", iteration=i)
                for i, v in enumerate(itls):
                    run.add_sample(f"c{conc}_itl_ms", v, "ms", iteration=i)
                run.add_sample(f"c{conc}_agg_tok_s", agg_tok_s, "tok/s", iteration=0)

                run.add_record(
                    {
                        "stage": f"concurrency_{conc}",
                        "concurrency": conc,
                        "requests_sent": len(res_list),
                        "requests_ok": len(ok_reqs),
                        "success_rate_pct": round(100.0 * len(ok_reqs) / max(1, len(res_list)), 2),
                        "error_distribution": err_codes,
                        "agg_tok_s": round(agg_tok_s, 2),
                        "per_req_tok_s_p50": round(per_req_tok_s, 2),
                        "ttft_p50_ms": round(float(np.percentile(ttfts, 50)), 2),
                        "ttft_p95_ms": round(float(np.percentile(ttfts, 95)), 2),
                        "ttft_p99_ms": round(float(np.percentile(ttfts, 99)), 2),
                        "itl_p50_ms": round(float(np.percentile(itls, 50)), 2),
                        "itl_p95_ms": round(float(np.percentile(itls, 95)), 2),
                        "itl_p99_ms": round(float(np.percentile(itls, 99)), 2),
                        "server_vram_alloc_mb": met.get("peak_vram_allocated_mb", 0.0),
                        "server_vram_resv_mb": met.get("peak_vram_reserved_mb", 0.0),
                        "server_rss_mb": met.get("host_rss_mb", 0.0),
                        "batches_executed": met.get("batches_executed", 0),
                    }
                )
                run.log_stdout(
                    f"  [Phase 14F | C={conc:3d}] ok={len(ok_reqs)}/{len(res_list)} | "
                    f"agg={agg_tok_s:.2f} tok/s | TTFT p50={np.percentile(ttfts, 50):.1f}ms | "
                    f"RSS={met.get('host_rss_mb', 0.0)}MB"
                )

            # 2. Verify prompt-output integrity & user isolation across concurrent users
            iso_res = await asyncio.gather(
                _send_one_client_req(
                    client, base_url, "alice", "The capital of France is", 12, False
                ),
                _send_one_client_req(client, base_url, "bob", "def fibonacci(n):", 12, False),
            )
            alice_txt = iso_res[0].get("text", "")
            bob_txt = iso_res[1].get("text", "")
            isolation_ok = (
                iso_res[0].get("ok", False) and iso_res[1].get("ok", False) and alice_txt != bob_txt
            )

            # 3. Sustained multi-user soak (`soak_seconds`, default 900s = 15 min) with injected bad clients
            run.log_stdout(f"[Phase 14F] Starting sustained soak for {soak_seconds:.0f}s...")
            t_soak0 = time.perf_counter()
            soak_end = t_soak0 + soak_seconds
            rss_timeline: list[tuple[float, float, float]] = []
            soak_ok = 0
            soak_fail = 0
            soak_fault_ok = 0
            soak_ttfts: list[float] = []
            soak_itls: list[float] = []
            round_idx = 0

            while time.perf_counter() < soak_end:
                batch_reqs = [
                    _send_one_client_req(
                        client,
                        base_url,
                        f"soak-{j}",
                        short_p if j % 2 == 0 else long_p[:240],
                        16,
                        j % 2 == 1,
                    )
                    for j in range(4)
                ]
                results = await asyncio.gather(*batch_reqs)
                for r in results:
                    if r.get("ok"):
                        soak_ok += 1
                        soak_ttfts.append(float(r["ttft_ms"]))
                        soak_itls.extend(r.get("itl_list_ms", []) or [float(r["tpot_ms"])])
                    else:
                        soak_fail += 1

                # Inject an invalid request & malformed JSON periodically to ensure no server crash
                if round_idx % 5 == 0:
                    try:
                        r_bad = await client.post(
                            f"{base_url}/v1/chat/completions",
                            headers={"Content-Type": "application/json"},
                            content=b"{malformed",
                            timeout=15.0,
                        )
                        if r_bad.status_code == 422:
                            soak_fault_ok += 1
                    except httpx.HTTPError:
                        soak_fail += 1

                if round_idx % 4 == 0:
                    try:
                        met_s = (await client.get(f"{base_url}/metrics", timeout=15.0)).json()
                        elapsed_m = (time.perf_counter() - t_soak0) / 60.0
                        rss_v = float(met_s.get("host_rss_mb", 0.0))
                        vram_v = float(met_s.get("peak_vram_allocated_mb", 0.0))
                        rss_timeline.append((elapsed_m, rss_v, vram_v))
                        run.add_sample("soak_server_rss_mb", rss_v, "MB", iteration=round_idx)
                        run.add_sample(
                            "soak_server_vram_alloc_mb", vram_v, "MB", iteration=round_idx
                        )
                    except httpx.HTTPError:
                        soak_fail += 1
                round_idx += 1

            # Compute post-warmup RSS slope (after first 20% of samples)
            slope_mb_per_min = 0.0
            if len(rss_timeline) >= 4:
                tail = rss_timeline[max(1, len(rss_timeline) // 5) :]
                xs = np.array([p[0] for p in tail], dtype=np.float64)
                ys = np.array([p[1] for p in tail], dtype=np.float64)
                if np.ptp(xs) > 1e-6:
                    slope_mb_per_min = float(np.polyfit(xs, ys, 1)[0])

            try:
                final_met = (await client.get(f"{base_url}/metrics", timeout=30.0)).json()
            except httpx.HTTPError:
                final_met = {"metrics_error": True}
            run.add_record(
                {
                    "stage": "sustained_soak",
                    "duration_s": round(time.perf_counter() - t_soak0, 1),
                    "soak_ok": soak_ok,
                    "soak_fail": soak_fail,
                    "injected_bad_requests_rejected": soak_fault_ok,
                    "user_isolation_verified": isolation_ok,
                    "ttft_p50_ms": round(float(np.percentile(soak_ttfts or [0.0], 50)), 2),
                    "ttft_p95_ms": round(float(np.percentile(soak_ttfts or [0.0], 95)), 2),
                    "ttft_p99_ms": round(float(np.percentile(soak_ttfts or [0.0], 99)), 2),
                    "itl_p50_ms": round(float(np.percentile(soak_itls or [0.0], 50)), 2),
                    "itl_p95_ms": round(float(np.percentile(soak_itls or [0.0], 95)), 2),
                    "itl_p99_ms": round(float(np.percentile(soak_itls or [0.0], 99)), 2),
                    "rss_slope_mb_per_min": round(slope_mb_per_min, 3),
                    "final_metrics": final_met,
                    "client_rss_mb": round(host_rss_mb(), 1),
                }
            )
    finally:
        _stop_owned_server(proc)
        srv_out_f.close()
        srv_err_f.close()


def run_serving_stress(
    model_dir: Path,
    output_dir: Path,
    soak_seconds: float = 900.0,
    max_concurrent: int = 128,
) -> Path:
    ckpt = build_checkpoint_manifest("TheBloke/Mistral-7B-v0.1-GPTQ", model_dir)
    run = ValidationRun(
        run_dir=output_dir,
        run_id="phase14f_serving_stress",
        checkpoint=ckpt,
        quant_config=QuantizationConfig(quant_format="gptq", group_size=128, desc_act=True),
        workload=WorkloadConfig(
            batch_size=16,
            concurrency=max_concurrent,
            prompt_len=32,
            output_len=32,
            measured_iters=1,
            sync_method="wall_clock_httpx",
        ),
        prompt_ids_or_checksum=prompt_checksum("phase14f_serving_stress"),
        execution_mode="real-checkpoint",
        reproduction_status="MEASURED",
        notes=f"Live 127.0.0.1 OpenAI HTTP server concurrency (1..{max_concurrent}) + {soak_seconds:.0f}s soak.",
    )
    asyncio.run(
        _run_stress_async(model_dir, run, soak_seconds=soak_seconds, max_concurrent=max_concurrent)
    )
    out = run.finalize()
    errs = validate_artifact_dir(out)
    assert errs == [], f"Phase 14F artifact validation failed: {errs}"
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 14F Multi-User Serving Stress Suite")
    parser.add_argument(
        "--model-dir", type=Path, default=Path("/home/arrhat/models/mistral_7b_4bit")
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("benchmarks/validation_artifacts/v3.1.0/phase14f_serving_stress"),
    )
    parser.add_argument("--soak-seconds", type=float, default=900.0)
    parser.add_argument("--max-concurrent", type=int, default=128)
    args = parser.parse_args()
    run_serving_stress(
        args.model_dir,
        args.output_dir,
        soak_seconds=args.soak_seconds,
        max_concurrent=args.max_concurrent,
    )


if __name__ == "__main__":
    main()
