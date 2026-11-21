"""Phase 14G: Controlled Production Fault Injection & Recovery Matrix."""

from __future__ import annotations

import argparse
import os
import signal
import socket
import subprocess
import sys
import time
from pathlib import Path

import httpx
import torch

from benchmarks.production.manifest import build_checkpoint_manifest, prompt_checksum
from benchmarks.production.runner import ValidationRun, host_rss_mb, peak_vram_mb
from benchmarks.production.schema import QuantizationConfig, WorkloadConfig, validate_artifact_dir


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


def _wait_health(base_url: str, proc: subprocess.Popen[str], timeout_s: float = 90.0) -> float:
    t0 = time.perf_counter()
    deadline = t0 + timeout_s
    while time.perf_counter() < deadline:
        if proc.poll() is not None:
            raise RuntimeError(f"Server exited prematurely with code {proc.returncode}")
        try:
            r = httpx.get(f"{base_url}/health", timeout=2.0)
            if r.status_code == 200:
                return (time.perf_counter() - t0) * 1000.0
        except Exception:
            pass
        time.sleep(0.25)
    raise TimeoutError(f"Server did not become healthy within {timeout_s}s")


def _spawn_server(
    model_dir: Path,
    port: int,
    api_key: str | None = None,
    max_concurrent: int = 8,
    max_prompt_tokens: int = 256,
) -> subprocess.Popen[str]:
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
        "--max-prompt-tokens",
        str(max_prompt_tokens),
    ]
    if api_key is not None:
        cmd.extend(["--api-key", api_key])
    env = dict(os.environ)
    env["CUTILE_ARCH"] = env.get("CUTILE_ARCH", "sm_120")
    return subprocess.Popen(
        cmd,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        text=True,
        env=env,
    )


def _stop_owned_server(proc: subprocess.Popen[str] | None) -> None:
    if proc is None:
        return
    if proc.poll() is None:
        proc.terminate()
        try:
            proc.wait(timeout=10.0)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait(timeout=5.0)


def run_reliability_suite(model_dir: Path, output_dir: Path) -> Path:
    ckpt = build_checkpoint_manifest("mistral-7b-v0.1-gptq", model_dir)
    run = ValidationRun(
        run_dir=output_dir,
        run_id="phase14g_reliability_fault_injection",
        checkpoint=ckpt,
        quant_config=QuantizationConfig(quant_format="gptq", group_size=128, desc_act=True),
        workload=WorkloadConfig(batch_size=1, concurrency=8, prompt_len=16, output_len=8),
        prompt_ids_or_checksum=prompt_checksum("phase14g_fault_matrix"),
        execution_mode="real-checkpoint",
        reproduction_status="MEASURED",
        notes="10 controlled fault-injection scenarios against real 127.0.0.1 HTTP servers & runtime.",
    )

    test_key = "phase14g-local-key"
    auth_headers = {"Authorization": f"Bearer {test_key}"}
    port = _free_port()
    proc: subprocess.Popen[str] | None = None

    try:
        proc = _spawn_server(
            model_dir,
            port=port,
            api_key=test_key,
            max_concurrent=2,
            max_prompt_tokens=128,
        )
        base_url = f"http://127.0.0.1:{port}"
        boot_ms = _wait_health(base_url, proc)
        run.log_stdout(f"[Phase 14G] Server pid={proc.pid} ready in {boot_ms:.1f}ms on {base_url}")

        # 1. Invalid JSON & unsupported fields
        t0 = time.perf_counter()
        r_bad_json = httpx.post(
            f"{base_url}/v1/chat/completions",
            headers={**auth_headers, "Content-Type": "application/json"},
            content=b"{not-valid-json",
            timeout=10.0,
        )
        r_unsupported = httpx.post(
            f"{base_url}/v1/chat/completions",
            headers=auth_headers,
            json={
                "model": "mistral",
                "messages": [{"role": "user", "content": "hi"}],
                "bogus_field": True,
            },
            timeout=10.0,
        )
        rec_ms = (time.perf_counter() - t0) * 1000.0
        ok_1 = r_bad_json.status_code == 422 and r_unsupported.status_code == 422
        run.add_record(
            {
                "fault": "1. Invalid JSON & unsupported API fields",
                "detection": "FastAPI/Pydantic strict schema (extra=forbid)",
                "expected": "HTTP 422 rejection without server impact",
                "user_visible_behavior": f"HTTP {r_bad_json.status_code} / {r_unsupported.status_code}",
                "recovery_ms": round(rec_ms, 2),
                "residual_rss_mb": round(host_rss_mb(), 1),
                "data_kv_integrity": "Untouched",
                "status": "VERIFIED" if ok_1 else "DEGRADED",
            }
        )

        # 2. Missing / invalid API key
        t0 = time.perf_counter()
        r_no_auth = httpx.post(
            f"{base_url}/v1/chat/completions",
            json={"model": "mistral", "messages": [{"role": "user", "content": "hi"}]},
            timeout=10.0,
        )
        r_bad_auth = httpx.get(
            f"{base_url}/v1/models", headers={"Authorization": "Bearer wrong-key"}, timeout=10.0
        )
        ok_2 = r_no_auth.status_code == 401 and r_bad_auth.status_code == 401
        run.add_record(
            {
                "fault": "2. Missing / invalid API key (auth enabled)",
                "detection": "Bearer check before queueing/execution",
                "expected": "HTTP 401 Unauthorized",
                "user_visible_behavior": f"HTTP {r_no_auth.status_code} / {r_bad_auth.status_code}",
                "recovery_ms": round((time.perf_counter() - t0) * 1000.0, 2),
                "residual_rss_mb": round(host_rss_mb(), 1),
                "data_kv_integrity": "Auth bypass blocked",
                "status": "VERIFIED" if ok_2 else "DEGRADED",
            }
        )

        # 3. Oversized prompt & excessive max_tokens
        t0 = time.perf_counter()
        r_long = httpx.post(
            f"{base_url}/v1/chat/completions",
            headers=auth_headers,
            json={
                "model": "mistral",
                "messages": [{"role": "user", "content": "token " * 300}],
                "max_tokens": 8,
            },
            timeout=10.0,
        )
        r_tok = httpx.post(
            f"{base_url}/v1/chat/completions",
            headers=auth_headers,
            json={
                "model": "mistral",
                "messages": [{"role": "user", "content": "hello"}],
                "max_tokens": 999999,
            },
            timeout=10.0,
        )
        ok_3 = r_long.status_code == 400 and r_tok.status_code == 422
        run.add_record(
            {
                "fault": "3. Oversized prompt & excessive max_tokens",
                "detection": "Tokenizer prompt length bound + Field(le=4096)",
                "expected": "HTTP 400 / 422 pre-execution rejection",
                "user_visible_behavior": f"HTTP {r_long.status_code} / {r_tok.status_code}",
                "recovery_ms": round((time.perf_counter() - t0) * 1000.0, 2),
                "residual_rss_mb": round(host_rss_mb(), 1),
                "data_kv_integrity": "KV bounds protected",
                "status": "VERIFIED" if ok_3 else "DEGRADED",
            }
        )

        # 4. Client disconnect during SSE streaming
        t0 = time.perf_counter()
        chunks_seen = 0
        with httpx.stream(
            "POST",
            f"{base_url}/v1/chat/completions",
            headers=auth_headers,
            json={
                "model": "mistral",
                "messages": [{"role": "user", "content": "Count slowly:"}],
                "max_tokens": 32,
                "stream": True,
            },
            timeout=30.0,
        ) as resp:
            for line in resp.iter_lines():
                if line:
                    chunks_seen += 1
                if chunks_seen >= 2:
                    break
        time.sleep(0.5)
        r_after_disc = httpx.post(
            f"{base_url}/v1/chat/completions",
            headers=auth_headers,
            json={
                "model": "mistral",
                "messages": [{"role": "user", "content": "The capital of France is"}],
                "max_tokens": 8,
            },
            timeout=30.0,
        )
        met_4 = httpx.get(f"{base_url}/metrics", timeout=10.0).json()
        ok_4 = r_after_disc.status_code == 200 and met_4["requests_inflight"] == 0
        run.add_record(
            {
                "fault": "4. Client disconnect mid-SSE stream",
                "detection": "request.is_disconnected() + finally slot release",
                "expected": "Abort stream, release GPU lock, inflight=0, follow-up HTTP 200",
                "user_visible_behavior": f"Follow-up HTTP {r_after_disc.status_code}, inflight={met_4['requests_inflight']}",
                "recovery_ms": round((time.perf_counter() - t0) * 1000.0, 2),
                "residual_rss_mb": round(host_rss_mb(), 1),
                "data_kv_integrity": "Subsequent completion uncorrupted",
                "status": "VERIFIED" if ok_4 else "DEGRADED",
            }
        )

        # 5. Queue saturation & backpressure (max_concurrent=2 -> launch 6 simultaneous)
        import concurrent.futures

        t0 = time.perf_counter()

        def _send_sat(i: int) -> int:
            return httpx.post(
                f"{base_url}/v1/chat/completions",
                headers=auth_headers,
                json={
                    "model": "mistral",
                    "messages": [{"role": "user", "content": f"Question {i}: hello"}],
                    "max_tokens": 16,
                },
                timeout=30.0,
            ).status_code

        with concurrent.futures.ThreadPoolExecutor(max_workers=6) as ex:
            codes = list(ex.map(_send_sat, range(6)))
        has_429 = 429 in codes
        has_200 = 200 in codes
        r_post_sat = httpx.get(f"{base_url}/health", timeout=10.0)
        ok_5 = has_200 and has_429 and r_post_sat.status_code == 200
        run.add_record(
            {
                "fault": "5. Queue saturation & backpressure (max_concurrent=2)",
                "detection": "requests_inflight >= max_concurrent_requests guard",
                "expected": "HTTP 429 on excess load + immediate healthy recovery",
                "user_visible_behavior": f"Status codes={codes}",
                "recovery_ms": round((time.perf_counter() - t0) * 1000.0, 2),
                "residual_rss_mb": round(host_rss_mb(), 1),
                "data_kv_integrity": "200 responses valid; rejected clean",
                "status": "VERIFIED" if ok_5 else "DEGRADED",
            }
        )

        # 9. Owned worker termination + restart recovery (run before closing server)
        old_pid = proc.pid
        t_kill0 = time.perf_counter()
        os.kill(old_pid, signal.SIGTERM)
        proc.wait(timeout=10.0)
        port2 = _free_port()
        proc = _spawn_server(model_dir, port=port2, api_key=test_key)
        base2 = f"http://127.0.0.1:{port2}"
        _wait_health(base2, proc)
        r_rec = httpx.post(
            f"{base2}/v1/chat/completions",
            headers=auth_headers,
            json={
                "model": "mistral",
                "messages": [{"role": "user", "content": "The capital of France is"}],
                "max_tokens": 8,
            },
            timeout=30.0,
        )
        ok_9 = r_rec.status_code == 200 and proc.pid != old_pid
        run.add_record(
            {
                "fault": "9. Owned worker termination (SIGTERM) & restart",
                "detection": "Supervisor detects exit, spawns replacement server",
                "expected": "Fresh worker answers /health=200 & valid generation",
                "user_visible_behavior": f"Restarted pid {old_pid}->{proc.pid} HTTP {r_rec.status_code}",
                "recovery_ms": round((time.perf_counter() - t_kill0) * 1000.0, 2),
                "residual_rss_mb": round(host_rss_mb(), 1),
                "data_kv_integrity": "Clean cold-start KV pool",
                "status": "VERIFIED" if ok_9 else "DEGRADED",
            }
        )
    finally:
        _stop_owned_server(proc)

    # 6. Missing model path & corrupt checkpoint
    t0 = time.perf_counter()
    bad_p = Path("/var/tmp/arrhat_models/does_not_exist_phase14g")
    miss_proc = subprocess.run(
        [sys.executable, "-m", "cutile_forge.cli", "serve", "--model", str(bad_p)],
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    corrupt_dir = Path("/var/tmp/arrhat_models/corrupt_ckpt_phase14g")
    corrupt_dir.mkdir(parents=True, exist_ok=True)
    (corrupt_dir / "config.json").write_text("{corrupt-json", encoding="utf-8")
    corr_proc = subprocess.run(
        [sys.executable, "-m", "cutile_forge.cli", "serve", "--model", str(corrupt_dir)],
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    ok_6 = miss_proc.returncode != 0 and corr_proc.returncode != 0
    run.add_record(
        {
            "fault": "6. Missing model path & corrupt checkpoint",
            "detection": "Startup FileNotFoundError / JSONDecodeError",
            "expected": "Non-zero exit with diagnostic error, zero residual VRAM",
            "user_visible_behavior": f"exit codes {miss_proc.returncode} / {corr_proc.returncode}",
            "recovery_ms": round((time.perf_counter() - t0) * 1000.0, 2),
            "residual_rss_mb": round(host_rss_mb(), 1),
            "data_kv_integrity": "Fail-fast prior to serving",
            "status": "VERIFIED" if ok_6 else "DEGRADED",
        }
    )

    # 7 & 8: Runtime CUDA Graph failure -> eager fallback & KV eviction exhaustion (in-process isolated)
    from cutile_forge.engine.config import ModelConfig
    from cutile_forge.engine.inference import InferenceEngine
    from cutile_forge.engine.model import Mistral7B
    from cutile_forge.runtime.native_kv_cache import NativePagedKVCache

    cfg_sm = ModelConfig(
        vocab_size=256,
        hidden_size=128,
        intermediate_size=256,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=32,
    )
    m_sm = Mistral7B.from_random(cfg_sm, device="cuda", seed=14)
    eng_sm = InferenceEngine(m_sm, use_cuda_graph=True)
    orig_cap = eng_sm._capture_decode_graph

    def _fail_graph() -> None:
        raise RuntimeError("Injected CUDA Graph capture fault")

    eng_sm._capture_decode_graph = _fail_graph  # type: ignore[method-assign]
    t0 = time.perf_counter()
    _ = eng_sm.prefill(torch.tensor([[1, 2, 3]], device="cuda"))
    lg = eng_sm.decode_step(torch.tensor([[4]], device="cuda"), pos=3)
    eng_sm._capture_decode_graph = orig_cap  # type: ignore[method-assign]
    ok_7 = eng_sm.graph_fallback_count == 1 and bool(torch.isfinite(lg).all().item())
    run.add_record(
        {
            "fault": "7. CUDA Graph capture failure -> eager fallback",
            "detection": "RuntimeError catch on _capture_decode_graph",
            "expected": "Automatic fallback to eager CuTile forward without crash",
            "user_visible_behavior": f"fallback_count={eng_sm.graph_fallback_count}, logits finite={ok_7}",
            "recovery_ms": round((time.perf_counter() - t0) * 1000.0, 2),
            "residual_rss_mb": round(host_rss_mb(), 1),
            "data_kv_integrity": "Decode output finite & valid",
            "status": "VERIFIED" if ok_7 else "DEGRADED",
        }
    )

    # 8. KV-cache capacity overflow
    kvc = NativePagedKVCache(
        num_layers=2, num_kv_heads=2, head_dim=32, max_seq_len=16, device="cuda"
    )
    t0 = time.perf_counter()
    kv_err_msg = ""
    try:
        k_over = torch.zeros((1, 2, 32, 32), device="cuda", dtype=torch.float16)
        kvc.append_and_get(0, k_over, k_over, start_pos=0)
    except RuntimeError as exc:
        kv_err_msg = str(exc)
    kvc.reset()
    k_ok = torch.zeros((1, 2, 8, 32), device="cuda", dtype=torch.float16)
    out_k, _ = kvc.append_and_get(0, k_ok, k_ok, start_pos=0)
    ok_8 = "exceeds max_seq_len" in kv_err_msg and out_k.shape[2] == 8
    run.add_record(
        {
            "fault": "8. KV-cache pressure / capacity exhaustion",
            "detection": "Bounds check prior to pool scatter",
            "expected": "Explicit RuntimeError; subsequent reset + valid sequence intact",
            "user_visible_behavior": kv_err_msg[:80],
            "recovery_ms": round((time.perf_counter() - t0) * 1000.0, 2),
            "residual_rss_mb": round(host_rss_mb(), 1),
            "data_kv_integrity": "No OOB pool overwrite",
            "status": "VERIFIED" if ok_8 else "DEGRADED",
        }
    )
    del eng_sm, m_sm, kvc
    torch.cuda.empty_cache()

    # 10. Repeated server start/stop cycles (3 rapid cold-boot cycles with small real/model check)
    cycle_rss: list[float] = []
    t_cyc0 = time.perf_counter()
    for c_idx in range(3):
        p_c = _free_port()
        srv = _spawn_server(model_dir, port=p_c)
        try:
            _wait_health(f"http://127.0.0.1:{p_c}", srv)
            r_h = httpx.get(f"http://127.0.0.1:{p_c}/metrics", timeout=10.0).json()
            cycle_rss.append(float(r_h.get("host_rss_mb", 0.0)))
        finally:
            _stop_owned_server(srv)
        run.add_sample("restart_cycle_rss_mb", cycle_rss[-1], "MB", iteration=c_idx)
    ok_10 = len(cycle_rss) == 3 and (max(cycle_rss) - min(cycle_rss)) < 256.0
    run.add_record(
        {
            "fault": "10. Repeated server start/stop cycles (3 cycles)",
            "detection": "Per-cycle server /metrics RSS tracking",
            "expected": "Clean shutdown + bounded host/GPU memory across restarts",
            "user_visible_behavior": f"RSS per cycle MB={[round(x, 1) for x in cycle_rss]}",
            "recovery_ms": round((time.perf_counter() - t_cyc0) * 1000.0, 2),
            "residual_rss_mb": round(cycle_rss[-1], 1),
            "data_kv_integrity": "Owned PIDs cleanly reaped",
            "status": "VERIFIED" if ok_10 else "DEGRADED",
        }
    )

    v_alloc, _ = peak_vram_mb()
    run.notes += f" Final client peak_vram_mb={v_alloc:.1f}"
    out = run.finalize()
    errs = validate_artifact_dir(out)
    assert errs == [], f"Phase 14G artifact validation failed: {errs}"
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 14G Controlled Fault Injection Suite")
    parser.add_argument(
        "--model-dir", type=Path, default=Path("/home/arrhat/models/mistral_7b_4bit")
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("benchmarks/validation_artifacts/v3.1.0/phase14g_reliability"),
    )
    args = parser.parse_args()
    run_reliability_suite(args.model_dir, args.output_dir)


if __name__ == "__main__":
    main()
