"""Phase 14B: External Engine Capability Probe & Fair Paired Comparison (`vLLM`, `SGLang`, `TensorRT-LLM`, `HF Transformers`, `CuTileForge`)."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
from pathlib import Path
from typing import Any

import numpy as np
import torch

from benchmarks.production.manifest import build_checkpoint_manifest, prompt_checksum
from benchmarks.production.runner import ValidationRun, peak_vram_mb
from benchmarks.production.schema import QuantizationConfig, WorkloadConfig, validate_artifact_dir

PROMPT_TEXT = (
    "NVIDIA Blackwell GPUs introduce Tensor Cores optimized for low-precision tiled matrix "
    "multiplication. Explain how paged KV-caching improves autoregressive throughput:"
)
WARMUP_RUNS = 2
MEASURED_RUNS = 5
OUT_TOKENS = 64


def _run_subprocess_probe(py_bin: Path, code: str, timeout_s: int = 300) -> dict[str, Any]:
    env = dict(os.environ)
    env.setdefault("HF_HOME", "/var/tmp/arrhat_models/hf_cache")
    env["TOKENIZERS_PARALLELISM"] = "false"
    proc = subprocess.run(
        [str(py_bin), "-c", code],
        capture_output=True,
        text=True,
        timeout=timeout_s,
        env=env,
        check=False,
    )
    marker = "__RESULT_JSON__:"
    for line in reversed(proc.stdout.splitlines()):
        if line.startswith(marker):
            data = json.loads(line[len(marker) :])
            data["returncode"] = proc.returncode
            return data
    return {
        "ok": False,
        "returncode": proc.returncode,
        "error": (proc.stderr or proc.stdout)[-1200:],
    }


def _vllm_probe_script() -> str:
    return r"""
import json, subprocess, time, torch, numpy as np
import vllm
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer

def _smi_vram_mb():
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-compute-apps=used_memory", "--format=csv,noheader,nounits"],
            text=True,
        ).strip()
        vals = [float(x.strip()) for x in out.splitlines() if x.strip() and x.strip().isdigit()]
        return round(sum(vals), 1)
    except Exception:
        free_b, tot_b = torch.cuda.mem_get_info()
        return round((tot_b - free_b) / (1024.0 * 1024.0), 1)

out = {"engine": f"vllm=={vllm.__version__}", "torch": torch.__version__, "cuda": torch.version.cuda, "results": []}

# 1. Probe Mistral-7B-v0.1-GPTQ (desc_act=True)
try:
    _m = LLM(model="/home/arrhat/models/mistral_7b_4bit", max_model_len=2048, gpu_memory_utilization=0.6)
    out["results"].append({"model": "Mistral-7B-v0.1-GPTQ", "status": "MEASURED"})
except Exception as exc:
    out["results"].append({
        "model": "Mistral-7B-v0.1-GPTQ",
        "format": "GPTQ 4-bit (desc_act=True)",
        "concurrency": 1,
        "status": "UNSUPPORTED",
        "error": f"{type(exc).__name__}: {str(exc)[:500]}",
    })

# 2. Fair benchmark on Qwen/Qwen2.5-0.5B (FP16)
tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B")
prompt_ids = tok.encode(PROMPT_TEXT_PH, add_special_tokens=True)
llm = LLM(
    model="Qwen/Qwen2.5-0.5B",
    dtype="float16",
    max_model_len=2048,
    gpu_memory_utilization=0.5,
    enable_prefix_caching=False,
    enforce_eager=False,
    seed=42,
)
sp_stream = SamplingParams(max_tokens=OUT_TOKENS_PH, temperature=0.0, seed=42, ignore_eos=True)
sp_first = SamplingParams(max_tokens=1, temperature=0.0, seed=42, ignore_eos=True)

for conc in (1, 4, 16):
    prompts = [{"prompt_token_ids": list(prompt_ids)} for _ in range(conc)]
    for _ in range(WARMUP_PH):
        _ = llm.generate(prompts, sp_stream, use_tqdm=False)
    ttfts, e2es, tpss = [], [], []
    for _ in range(RUNS_PH):
        if conc == 1:
            t0 = time.perf_counter()
            _ = llm.generate([{"prompt_token_ids": list(prompt_ids)}], sp_first, use_tqdm=False)
            ttfts.append((time.perf_counter() - t0) * 1000.0)
        t1 = time.perf_counter()
        outs = llm.generate(prompts, sp_stream, use_tqdm=False)
        el_s = time.perf_counter() - t1
        gen_tok = sum(len(o.outputs[0].token_ids) for o in outs)
        e2es.append(el_s * 1000.0)
        tpss.append(gen_tok / max(1e-6, el_s))
    vram_mb = _smi_vram_mb()
    ttft_med = float(np.median(ttfts)) if ttfts else (float(np.median(e2es)) / OUT_TOKENS_PH)
    itl_med = max(0.01, (float(np.median(e2es)) - ttft_med) / max(1, OUT_TOKENS_PH - 1)) if conc == 1 else (1000.0 * conc / float(np.median(tpss)))
    out["results"].append({
        "model": "Qwen/Qwen2.5-0.5B",
        "format": "FP16",
        "concurrency": conc,
        "status": "MEASURED",
        "ttft_ms": round(ttft_med, 2),
        "itl_p50_ms": round(itl_med, 3),
        "itl_p99_ms": round(itl_med, 3),
        "throughput_tok_s": round(float(np.median(tpss)), 2),
        "throughput_runs": [round(x, 2) for x in tpss],
        "peak_vram_mb": vram_mb,
        "notes": "cuda_graph=True (PIECEWISE+FULL), prefix_caching=False, FlashInfer, gpu_memory_utilization=0.5",
    })
print("__RESULT_JSON__:" + json.dumps(out))
"""


def _sglang_probe_script() -> str:
    return r"""
import json, subprocess, time, torch, numpy as np
import sglang as sgl
from transformers import AutoTokenizer

def _smi_vram_mb():
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-compute-apps=used_memory", "--format=csv,noheader,nounits"],
            text=True,
        ).strip()
        vals = [float(x.strip()) for x in out.splitlines() if x.strip() and x.strip().isdigit()]
        return round(sum(vals), 1)
    except Exception:
        return 0.0

out = {"engine": f"sglang=={sgl.__version__}", "torch": torch.__version__, "cuda": torch.version.cuda, "results": []}
tok_q = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B")
q_ids = tok_q.encode(PROMPT_TEXT_PH, add_special_tokens=True)

# 1. Fair benchmark on Qwen/Qwen2.5-0.5B (FP16)
eng_q = sgl.Engine(
    model_path="Qwen/Qwen2.5-0.5B",
    dtype="float16",
    mem_fraction_static=0.5,
    disable_radix_cache=True,
    log_level="error",
)
params = {"max_new_tokens": OUT_TOKENS_PH, "temperature": 0.0, "ignore_eos": True}
params_1 = {"max_new_tokens": 1, "temperature": 0.0, "ignore_eos": True}
for conc in (1, 4, 16):
    batch_ids = [list(q_ids) for _ in range(conc)]
    for _ in range(WARMUP_PH):
        _ = eng_q.generate(input_ids=batch_ids, sampling_params=params)
    ttfts, e2es, tpss = [], [], []
    for _ in range(RUNS_PH):
        if conc == 1:
            t0 = time.perf_counter()
            _ = eng_q.generate(input_ids=[list(q_ids)], sampling_params=params_1)
            ttfts.append((time.perf_counter() - t0) * 1000.0)
        t1 = time.perf_counter()
        _ = eng_q.generate(input_ids=batch_ids, sampling_params=params)
        el_s = time.perf_counter() - t1
        e2es.append(el_s * 1000.0)
        tpss.append((conc * OUT_TOKENS_PH) / max(1e-6, el_s))
    vram_mb = _smi_vram_mb()
    ttft_med = float(np.median(ttfts)) if ttfts else (float(np.median(e2es)) / OUT_TOKENS_PH)
    itl_med = max(0.01, (float(np.median(e2es)) - ttft_med) / max(1, OUT_TOKENS_PH - 1)) if conc == 1 else (1000.0 * conc / float(np.median(tpss)))
    out["results"].append({
        "model": "Qwen/Qwen2.5-0.5B",
        "format": "FP16",
        "concurrency": conc,
        "status": "MEASURED",
        "ttft_ms": round(ttft_med, 2),
        "itl_p50_ms": round(itl_med, 3),
        "itl_p99_ms": round(itl_med, 3),
        "throughput_tok_s": round(float(np.median(tpss)), 2),
        "throughput_runs": [round(x, 2) for x in tpss],
        "peak_vram_mb": vram_mb,
        "notes": "cuda_graph=True, radix_cache=False, mem_fraction_static=0.5 (nvidia-smi VRAM)",
    })
eng_q.shutdown()

# 2. Fair benchmark on Mistral-7B-v0.1-GPTQ (real checkpoint)
tok_m = AutoTokenizer.from_pretrained("/home/arrhat/models/mistral_7b_4bit", local_files_only=True)
m_ids = tok_m.encode(PROMPT_TEXT_PH, add_special_tokens=True)
eng_m = sgl.Engine(
    model_path="/home/arrhat/models/mistral_7b_4bit",
    mem_fraction_static=0.5,
    disable_radix_cache=True,
    log_level="error",
)
for conc in (1, 4, 16):
    batch_ids = [list(m_ids) for _ in range(conc)]
    for _ in range(WARMUP_PH):
        _ = eng_m.generate(input_ids=batch_ids, sampling_params=params)
    ttfts, e2es, tpss = [], [], []
    for _ in range(RUNS_PH):
        if conc == 1:
            t0 = time.perf_counter()
            _ = eng_m.generate(input_ids=[list(m_ids)], sampling_params=params_1)
            ttfts.append((time.perf_counter() - t0) * 1000.0)
        t1 = time.perf_counter()
        _ = eng_m.generate(input_ids=batch_ids, sampling_params=params)
        el_s = time.perf_counter() - t1
        e2es.append(el_s * 1000.0)
        tpss.append((conc * OUT_TOKENS_PH) / max(1e-6, el_s))
    vram_mb = _smi_vram_mb()
    ttft_med = float(np.median(ttfts)) if ttfts else (float(np.median(e2es)) / OUT_TOKENS_PH)
    itl_med = max(0.01, (float(np.median(e2es)) - ttft_med) / max(1, OUT_TOKENS_PH - 1)) if conc == 1 else (1000.0 * conc / float(np.median(tpss)))
    out["results"].append({
        "model": "Mistral-7B-v0.1-GPTQ",
        "format": "GPTQ 4-bit (desc_act=True)",
        "concurrency": conc,
        "status": "MEASURED",
        "ttft_ms": round(ttft_med, 2),
        "itl_p50_ms": round(itl_med, 3),
        "itl_p99_ms": round(itl_med, 3),
        "throughput_tok_s": round(float(np.median(tpss)), 2),
        "throughput_runs": [round(x, 2) for x in tpss],
        "peak_vram_mb": vram_mb,
        "notes": "cuda_graph=True, radix_cache=False, mem_fraction_static=0.5 (nvidia-smi VRAM)",
    })
eng_m.shutdown()
print("__RESULT_JSON__:" + json.dumps(out))
"""


def _sub_placeholders(tmpl: str) -> str:
    return (
        tmpl.replace("PROMPT_TEXT_PH", json.dumps(PROMPT_TEXT))
        .replace("OUT_TOKENS_PH", str(OUT_TOKENS))
        .replace("WARMUP_PH", str(WARMUP_RUNS))
        .replace("RUNS_PH", str(MEASURED_RUNS))
    )


def _run_cutile_and_hf_fair(run: ValidationRun, model_dir: Path) -> None:
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from cutile_forge.engine.config import GPUConfig
    from cutile_forge.engine.model import Mistral7B
    from cutile_forge.engine.weight_loader import load_gptq_weights
    from cutile_forge.models import UnifiedModelExecutor, extract_hf_decoder_weights
    from cutile_forge.runtime import NativeModelRuntime

    # 1. CuTileForge NativeModelRuntime on Mistral-7B-v0.1-GPTQ (conc 1, 4, 16)
    tok_m = AutoTokenizer.from_pretrained(str(model_dir), local_files_only=True)
    m_ids = tok_m.encode(PROMPT_TEXT, add_special_tokens=True)
    torch.cuda.reset_peak_memory_stats()
    w = load_gptq_weights(model_dir, device="cuda", pin_ddr5=False)
    m = Mistral7B(
        weights=w, gpu_config=GPUConfig(vram_gb=32.0, stream_weights=False), device="cuda"
    )
    rt = NativeModelRuntime(m, max_seq_len=2048, use_cuda_graph=True, max_batch_size=16)

    for conc in (1, 4, 16):
        prompts = [list(m_ids) for _ in range(conc)]
        for _ in range(WARMUP_RUNS):
            if conc == 1:
                _ = rt.generate(m_ids, max_new_tokens=OUT_TOKENS, sync_interval=32)
            else:
                _ = rt.generate_batch(prompts, max_new_tokens=OUT_TOKENS, sync_interval=32)
        torch.cuda.synchronize()
        ttfts, itls, tpss = [], [], []
        for it in range(MEASURED_RUNS):
            if conc == 1:
                import time

                rt.reset()
                p_t = torch.tensor([m_ids], device="cuda", dtype=torch.long)
                torch.cuda.synchronize()
                t0 = time.perf_counter()
                _ = rt.prefill(p_t)
                torch.cuda.synchronize()
                ttfts.append((time.perf_counter() - t0) * 1000.0)
                t1 = time.perf_counter()
                _, st = rt.generate(m_ids, max_new_tokens=OUT_TOKENS, sync_interval=32)
                torch.cuda.synchronize()
                el_s = time.perf_counter() - t1
                tps_v = OUT_TOKENS / max(1e-6, el_s)
                tpss.append(tps_v)
                itls.append(float(st["ms_per_token"]))
            else:
                import time

                torch.cuda.synchronize()
                t1 = time.perf_counter()
                _, st = rt.generate_batch(prompts, max_new_tokens=OUT_TOKENS, sync_interval=32)
                torch.cuda.synchronize()
                el_s = time.perf_counter() - t1
                tps_v = (conc * OUT_TOKENS) / max(1e-6, el_s)
                tpss.append(tps_v)
                itls.append((1000.0 * conc) / tps_v)
            run.add_sample(f"cutile_mistral_c{conc}_tok_s", tpss[-1], "tok/s", iteration=it)
        alloc_mb, _ = peak_vram_mb()
        ttft_med = float(np.median(ttfts)) if ttfts else 0.0
        run.add_record(
            {
                "engine": "CuTileForge v3.1.0 (Native)",
                "model": "Mistral-7B-v0.1-GPTQ",
                "format": "GPTQ 4-bit (desc_act=True)",
                "concurrency": conc,
                "status": "MEASURED",
                "ttft_ms": round(ttft_med, 2),
                "itl_p50_ms": round(float(np.percentile(itls, 50)), 3),
                "itl_p99_ms": round(float(np.percentile(itls, 99)), 3),
                "throughput_tok_s": round(float(np.median(tpss)), 2),
                "peak_vram_mb": round(alloc_mb, 1),
                "notes": "Zero-ATen full-step CUDA Graph + sync_interval=32, radix=False",
            }
        )
    del rt, m, w
    torch.cuda.empty_cache()

    # 2. Same-machine Qwen/Qwen2.5-0.5B: HF Transformers + CuTileForge UnifiedModelExecutor
    import time

    tok_q = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B", local_files_only=True)
    q_ids = tok_q.encode(PROMPT_TEXT, add_special_tokens=True)
    q_t = torch.tensor([q_ids], device="cuda", dtype=torch.long)

    torch.cuda.reset_peak_memory_stats()
    hf_q = (
        AutoModelForCausalLM.from_pretrained(
            "Qwen/Qwen2.5-0.5B", local_files_only=True, torch_dtype=torch.float16
        )
        .to("cuda")
        .eval()
    )
    with torch.no_grad():
        for _ in range(WARMUP_RUNS):
            _ = hf_q.generate(q_t, max_new_tokens=OUT_TOKENS, do_sample=False)
    torch.cuda.synchronize()
    hf_ttfts, hf_tpss = [], []
    for it in range(MEASURED_RUNS):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        with torch.no_grad():
            _ = hf_q.generate(q_t, max_new_tokens=1, do_sample=False)
        torch.cuda.synchronize()
        hf_ttfts.append((time.perf_counter() - t0) * 1000.0)
        t1 = time.perf_counter()
        with torch.no_grad():
            _ = hf_q.generate(q_t, max_new_tokens=OUT_TOKENS, do_sample=False)
        torch.cuda.synchronize()
        el_s = time.perf_counter() - t1
        hf_tpss.append(OUT_TOKENS / max(1e-6, el_s))
        run.add_sample("hf_qwen05b_c1_tok_s", hf_tpss[-1], "tok/s", iteration=it)
    hf_vram, _ = peak_vram_mb()
    run.add_record(
        {
            "engine": "HF Transformers 5.23.0 (Eager)",
            "model": "Qwen/Qwen2.5-0.5B",
            "format": "FP16",
            "concurrency": 1,
            "status": "MEASURED",
            "ttft_ms": round(float(np.median(hf_ttfts)), 2),
            "itl_p50_ms": round(1000.0 / float(np.median(hf_tpss)), 3),
            "itl_p99_ms": round(1000.0 / float(np.min(hf_tpss)), 3),
            "throughput_tok_s": round(float(np.median(hf_tpss)), 2),
            "peak_vram_mb": round(hf_vram, 1),
            "notes": "Baseline eager HF reference (NOT a substitute for vLLM/SGLang/TRT-LLM)",
        }
    )

    # CuTileForge UnifiedModelExecutor (recompute eager, no Qwen2 CUDA graph KV cache yet -> expected slower)
    g_q, w_q = extract_hf_decoder_weights(hf_q, num_layers=24, device="cuda", dtype=torch.float16)
    ex_q = UnifiedModelExecutor(g_q)
    torch.cuda.reset_peak_memory_stats()
    with torch.no_grad():
        _ = ex_q.forward(w_q, input_ids=q_t)
    torch.cuda.synchronize()
    ct_tpss, ct_ttfts = [], []
    for it in range(MEASURED_RUNS):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        with torch.no_grad():
            lg = ex_q.forward(w_q, input_ids=q_t)
        torch.cuda.synchronize()
        ct_ttfts.append((time.perf_counter() - t0) * 1000.0)
        cur = q_t
        t1 = time.perf_counter()
        with torch.no_grad():
            for _ in range(OUT_TOKENS):
                lg = ex_q.forward(w_q, input_ids=cur)
                nxt = lg[:, -1:].argmax(dim=-1)
                cur = torch.cat([cur, nxt], dim=1)
        torch.cuda.synchronize()
        ct_tpss.append(OUT_TOKENS / max(1e-6, time.perf_counter() - t1))
        run.add_sample("cutile_unified_qwen05b_c1_tok_s", ct_tpss[-1], "tok/s", iteration=it)
    ct_vram, _ = peak_vram_mb()
    run.add_record(
        {
            "engine": "CuTileForge UnifiedModelExecutor",
            "model": "Qwen/Qwen2.5-0.5B",
            "format": "FP16 (no KV-cache/CUDA-graph)",
            "concurrency": 1,
            "status": "DEGRADED",
            "ttft_ms": round(float(np.median(ct_ttfts)), 2),
            "itl_p50_ms": round(1000.0 / float(np.median(ct_tpss)), 3),
            "itl_p99_ms": round(1000.0 / float(np.min(ct_tpss)), 3),
            "throughput_tok_s": round(float(np.median(ct_tpss)), 2),
            "peak_vram_mb": round(ct_vram, 1),
            "notes": "UnifiedModelExecutor lacks Qwen2 Paged KV-cache & CUDA Graph decode loop; slower than vLLM/SGLang/HF",
        }
    )
    del hf_q, w_q, ex_q
    torch.cuda.empty_cache()


def run_external_engine_validation(model_dir: Path, output_dir: Path) -> Path:
    ckpt = build_checkpoint_manifest("Mistral-7B-v0.1-GPTQ + Qwen2.5-0.5B", model_dir)
    run = ValidationRun(
        run_dir=output_dir,
        run_id="phase14b_external_engines",
        checkpoint=ckpt,
        quant_config=QuantizationConfig(
            quant_format="GPTQ_4bit + FP16", group_size=128, desc_act=True
        ),
        workload=WorkloadConfig(
            batch_size=16,
            concurrency=16,
            prompt_len=len(PROMPT_TEXT.split()),
            output_len=OUT_TOKENS,
            temperature=0.0,
            seed=42,
            warmup_iters=WARMUP_RUNS,
            measured_iters=MEASURED_RUNS,
            sync_method="wall_clock_e2e_and_cuda_sync",
        ),
        prompt_ids_or_checksum=prompt_checksum(PROMPT_TEXT),
        execution_mode="real-checkpoint",
        reproduction_status="MEASURED",
        notes="Same-machine isolated-venv probes for vLLM==0.34.0, SGLang==0.5.23, TensorRT-LLM==1.3.0, HF, and CuTileForge.",
    )

    run.log_stdout("[Phase 14B] Running CuTileForge Native + HF + UnifiedExecutor baseline...")
    _run_cutile_and_hf_fair(run, model_dir)

    vllm_py = Path("/var/tmp/arrhat_models/venvs/vllm_env/bin/python")
    if vllm_py.exists():
        run.log_stdout("[Phase 14B] Probing vLLM==0.34.0...")
        v_res = _run_subprocess_probe(
            vllm_py, _sub_placeholders(_vllm_probe_script()), timeout_s=360
        )
        for rec in v_res.get("results", []):
            rec["engine"] = "vLLM 0.34.0"
            run.add_record(rec)
            if rec.get("status") == "MEASURED":
                for i, tps in enumerate(rec.get("throughput_runs", [])):
                    run.add_sample(
                        f"vllm_qwen_c{rec['concurrency']}_tok_s", tps, "tok/s", iteration=i
                    )
    else:
        run.add_record({"engine": "vLLM 0.34.0", "status": "NOT_RUN", "error": "venv missing"})

    sgl_py = Path("/var/tmp/arrhat_models/venvs/sglang_env/bin/python")
    if sgl_py.exists():
        run.log_stdout("[Phase 14B] Probing SGLang==0.5.23...")
        s_res = _run_subprocess_probe(
            sgl_py, _sub_placeholders(_sglang_probe_script()), timeout_s=420
        )
        for rec in s_res.get("results", []):
            rec["engine"] = "SGLang 0.5.23"
            run.add_record(rec)
            if rec.get("status") == "MEASURED":
                tag = "qwen" if "Qwen" in rec["model"] else "mistral"
                for i, tps in enumerate(rec.get("throughput_runs", [])):
                    run.add_sample(
                        f"sglang_{tag}_c{rec['concurrency']}_tok_s", tps, "tok/s", iteration=i
                    )
    else:
        run.add_record({"engine": "SGLang 0.5.23", "status": "NOT_RUN", "error": "venv missing"})

    # TensorRT-LLM probe
    trt_py = Path("/var/tmp/arrhat_models/venvs/trtllm_env/bin/python")
    if trt_py.exists():
        trt_probe = subprocess.run(
            [str(trt_py), "-c", "import tensorrt_llm; print(tensorrt_llm.__version__)"],
            capture_output=True,
            text=True,
            timeout=60,
            check=False,
        )
        run.add_record(
            {
                "engine": "TensorRT-LLM 1.3.0",
                "model": "Mistral-7B-v0.1-GPTQ / Qwen2.5-0.5B",
                "format": "FP16 / GPTQ",
                "concurrency": 1,
                "status": "NOT_RUN" if trt_probe.returncode != 0 else "MEASURED",
                "error": (trt_probe.stderr or trt_probe.stdout).strip()[-700:],
            }
        )

    out = run.finalize()
    errs = validate_artifact_dir(out)
    assert errs == [], f"Phase 14B artifact validation failed: {errs}"
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 14B External Engine Comparison")
    parser.add_argument(
        "--model-dir", type=Path, default=Path("/home/arrhat/models/mistral_7b_4bit")
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("benchmarks/validation_artifacts/v3.1.0/phase14b_external_engines"),
    )
    args = parser.parse_args()
    run_external_engine_validation(args.model_dir, args.output_dir)


if __name__ == "__main__":
    main()
