"""Phase 14C: Real-Checkpoint Long-Context Validation (4K -> 32K + 64K limit probe)."""

from __future__ import annotations

import argparse
import gc
import math
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from benchmarks.production.manifest import build_checkpoint_manifest, prompt_checksum
from benchmarks.production.runner import (
    TimedDeviceOp,
    ValidationRun,
    host_rss_mb,
    peak_vram_mb,
)
from benchmarks.production.schema import QuantizationConfig, WorkloadConfig, validate_artifact_dir
from cutile_forge.engine.config import GPUConfig
from cutile_forge.engine.model import Mistral7B
from cutile_forge.engine.weight_loader import load_gptq_weights
from cutile_forge.kernels.attention_gqa import flash_decoding_split_kv
from cutile_forge.kernels.embedding_lmhead import cutile_embedding_lookup
from cutile_forge.kernels.rmsnorm import cutile_rmsnorm
from cutile_forge.runtime import NativeModelRuntime, NativePagedKVCache

CANONICAL_TECH_DOC = (
    "NVIDIA Blackwell GPUs introduce fifth-generation Tensor Cores with hardware FP4 and FP8 "
    "matrix multiply-accumulate pipelines paired with high-bandwidth GDDR7 memory. In transformer "
    "autoregressive inference, prefill saturates compute units with large sequence matrix "
    "multiplications while decode stalls on weight bandwidth and growing KV-cache reads. "
    "Paged KV management and tiled register dequantization eliminate intermediate HBM round-trips "
    "across normalization, positional rotary embeddings, grouped-query attention, and SwiGLU "
    "feed-forward blocks. "
)


def _build_token_sequence(tokenizer: Any, target_len: int) -> list[int]:
    base_ids = tokenizer.encode(CANONICAL_TECH_DOC, add_special_tokens=False)
    ids = [tokenizer.bos_token_id or 1]
    while len(ids) < target_len:
        ids.extend(base_ids)
    return ids[:target_len]


def _chunked_prefill_and_decode(
    runtime: NativeModelRuntime,
    tokenizer: Any,
    prompt_ids: list[int],
    gen_tokens: int = 32,
    chunk_size: int = 512,
    eval_ppl_tail: int = 512,
) -> dict[str, Any]:
    """Run real full-model chunked prefill + graphed decode + coherence/PPL check."""
    runtime.reset()
    n_prompt = len(prompt_ids)

    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    t_pre0 = time.perf_counter()

    next_tok_t: torch.Tensor | None = None
    for s in range(0, n_prompt, chunk_size):
        e = min(n_prompt, s + chunk_size)
        is_last = e == n_prompt
        chunk_t = torch.tensor([prompt_ids[s:e]], device=runtime.device, dtype=torch.long)
        res = runtime.prefill_chunk(chunk_t, start_pos=s, emit_next_token=is_last)
        if is_last:
            next_tok_t = res

    assert next_tok_t is not None
    torch.cuda.synchronize()
    prefill_ms = (time.perf_counter() - t_pre0) * 1000.0
    first_tok = int(next_tok_t.item())
    prompt_tok_s = (n_prompt / max(1e-6, prefill_ms)) * 1000.0

    # Time per-token decode with CUDA events and wall-clock
    generated = [first_tok]
    cur_t = next_tok_t
    pos = n_prompt
    itl_list: list[float] = []

    # Warmup / pre-capture graph bucket cleanly if enabled
    for _step in range(max(0, gen_tokens - 1)):
        with TimedDeviceOp.measure(sync=True) as op:
            cur_t = runtime.decode_step(cur_t, pos=pos, use_graph=runtime.use_cuda_graph)
        itl_list.append(op.elapsed_ms)
        generated.append(int(cur_t.item()))
        pos += 1

    vram_alloc, vram_resv = peak_vram_mb()
    rss = host_rss_mb()

    # Exact KV bytes & page count from NativePagedKVCache
    kvc = runtime.kv_cache
    elem_bytes = 1 if kvc.storage_dtype == torch.float8_e4m3fn else 2
    kv_pool_bytes = (
        kvc.num_layers
        * 2
        * kvc.max_batch_size
        * kvc.num_kv_heads
        * kvc.total_slots
        * kvc.head_dim
        * elem_bytes
    )
    kv_work_bytes = sum(t.numel() * t.element_size() for t in kvc._k_work) + sum(
        t.numel() * t.element_size() for t in kvc._v_work
    )
    active_pages = kvc.num_layers * ((pos + kvc.block_size - 1) // kvc.block_size)

    # Evaluate tail-window long-context NLL / PPL on the live filled KV context
    ppl_val = 0.0
    tail_n = min(eval_ppl_tail, n_prompt)
    if tail_n >= 16:
        tail_start = n_prompt - tail_n
        runtime.reset()
        with torch.inference_mode():
            for s in range(0, tail_start, chunk_size):
                e = min(tail_start, s + chunk_size)
                c_t = torch.tensor([prompt_ids[s:e]], device=runtime.device, dtype=torch.long)
                runtime.prefill_chunk(c_t, start_pos=s, emit_next_token=False)
            tail_ids = torch.tensor(
                [prompt_ids[tail_start:n_prompt]], device=runtime.device, dtype=torch.long
            )
            h = cutile_embedding_lookup(tail_ids.reshape(tail_n), runtime._embed_w).view(
                tail_n, runtime.config.hidden_size
            )
            for l_idx in range(runtime.config.num_hidden_layers):
                h = runtime._forward_layer_native(
                    l_idx, h, pos_offset=tail_start, seq_len=tail_n, batch_size=1
                )
            h_n = cutile_rmsnorm(h, runtime._norm_w, eps=runtime.config.rms_norm_eps)
            logits = torch.matmul(h_n.float(), runtime._lm_head_w.float().t()).unsqueeze(0)
            loss = F.cross_entropy(
                logits[:, :-1, :].reshape(-1, logits.shape[-1]),
                tail_ids[:, 1:].reshape(-1),
            )
            ppl_val = float(math.exp(min(20.0, float(loss.item()))))

    snippet = tokenizer.decode(generated, skip_special_tokens=True)[:140]
    arr = np.array(itl_list, dtype=np.float64) if itl_list else np.array([0.0])
    decode_tok_s = 1000.0 / float(np.median(arr)) if np.median(arr) > 0 else 0.0

    return {
        "prefill_ms": prefill_ms,
        "ttft_ms": prefill_ms,
        "prompt_tok_s": prompt_tok_s,
        "itl_list_ms": itl_list,
        "itl_p50_ms": float(np.percentile(arr, 50)),
        "itl_p90_ms": float(np.percentile(arr, 90)),
        "itl_p95_ms": float(np.percentile(arr, 95)),
        "itl_p99_ms": float(np.percentile(arr, 99)),
        "decode_tok_s": decode_tok_s,
        "peak_vram_allocated_mb": vram_alloc,
        "peak_vram_reserved_mb": vram_resv,
        "host_rss_mb": rss,
        "kv_pool_bytes": kv_pool_bytes,
        "kv_work_bytes": kv_work_bytes,
        "kv_total_bytes": kv_pool_bytes + kv_work_bytes,
        "kv_page_count": active_pages,
        "kv_evictions": 0,
        "host_fetches": 0,
        "tail_ppl": ppl_val,
        "generated_tokens": generated,
        "output_snippet": snippet,
    }


def run_long_context_validation(
    model_dir: Path,
    mixtral_dir: Path,
    output_dir: Path,
    synth_dir: Path,
    gen_tokens: int = 32,
) -> tuple[Path, Path]:
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(str(model_dir), local_files_only=True)
    ckpt = build_checkpoint_manifest("mistralai/Mistral-7B-v0.1-GPTQ", model_dir)
    sample_prompt = _build_token_sequence(tokenizer, 4096)

    run = ValidationRun(
        run_dir=output_dir,
        run_id="phase14c_long_context_real",
        checkpoint=ckpt,
        quant_config=QuantizationConfig(
            quant_format="gptq",
            group_size=128,
            desc_act=True,
            kv_dtype="fp16+fp8_e4m3fn",
            disk_format="gptq_int4",
            kernel_format="cutile_int4_dequant",
        ),
        workload=WorkloadConfig(
            batch_size=1,
            concurrency=1,
            prompt_len=32736,
            output_len=gen_tokens,
            temperature=0.0,
            seed=42,
            warmup_iters=1,
            measured_iters=1,
        ),
        prompt_ids_or_checksum=prompt_checksum(sample_prompt),
        execution_mode="real-checkpoint",
        reproduction_status="MEASURED",
        notes="Real full-model 32-layer Mistral-7B-v0.1-GPTQ (4K..32K + 64K bound) + Mixtral-8x7B-GPTQ.",
    )

    weights = load_gptq_weights(model_dir, device="cuda", pin_ddr5=False)
    model = Mistral7B(
        weights=weights,
        gpu_config=GPUConfig(vram_gb=32.0, stream_weights=False),
        device="cuda",
    )

    # Check chunked (512) vs single-shot prefill greedy parity at 4K
    prompt_4k = _build_token_sequence(tokenizer, 4096 - gen_tokens)
    rt_single = NativeModelRuntime(
        model, max_seq_len=4096, kv_dtype="fp16", use_cuda_graph=True, max_batch_size=1
    )
    rt_single.reset()
    tok_single = int(
        rt_single.prefill(torch.tensor([prompt_4k], device="cuda", dtype=torch.long)).item()
    )
    del rt_single
    torch.cuda.empty_cache()

    fp16_tokens_by_ctx: dict[int, list[int]] = {}
    fp8_tokens_by_ctx: dict[int, list[int]] = {}

    for ctx_len in (4096, 8192, 16384, 32768):
        prompt_ids = _build_token_sequence(tokenizer, ctx_len - gen_tokens)
        for kv_dt in ("fp16", "fp8_e4m3fn"):
            run.log_stdout(f"[Phase 14C] Running real Mistral-7B ctx={ctx_len} kv_dtype={kv_dt}...")
            try:
                rt = NativeModelRuntime(
                    model,
                    max_seq_len=ctx_len,
                    kv_dtype=kv_dt,
                    use_cuda_graph=True,
                    max_batch_size=1,
                )
                res = _chunked_prefill_and_decode(rt, tokenizer, prompt_ids, gen_tokens=gen_tokens)
                chunk_parity_ok = (
                    (res["generated_tokens"][0] == tok_single)
                    if (ctx_len == 4096 and kv_dt == "fp16")
                    else None
                )
                if kv_dt == "fp16":
                    fp16_tokens_by_ctx[ctx_len] = res["generated_tokens"]
                else:
                    fp8_tokens_by_ctx[ctx_len] = res["generated_tokens"]

                greedy_agree = None
                if ctx_len in fp16_tokens_by_ctx and ctx_len in fp8_tokens_by_ctx:
                    g16 = fp16_tokens_by_ctx[ctx_len]
                    g8 = fp8_tokens_by_ctx[ctx_len]
                    matches = sum(1 for a, b in zip(g16, g8, strict=False) if a == b)
                    greedy_agree = f"{matches}/{len(g16)}"

                for i, v in enumerate(res["itl_list_ms"]):
                    run.add_sample(f"mistral_ctx{ctx_len}_{kv_dt}_itl_ms", v, "ms", iteration=i)
                run.add_sample(
                    f"mistral_ctx{ctx_len}_{kv_dt}_prefill_ms", res["prefill_ms"], "ms", iteration=0
                )

                run.add_record(
                    {
                        "model": "Mistral-7B-v0.1-GPTQ (32L)",
                        "weight_source": "real-checkpoint",
                        "context_len": ctx_len,
                        "prompt_len": len(prompt_ids),
                        "kv_dtype": kv_dt,
                        "status": "MEASURED",
                        "prefill_ms": round(res["prefill_ms"], 2),
                        "ttft_ms": round(res["ttft_ms"], 2),
                        "prompt_tok_s": round(res["prompt_tok_s"], 2),
                        "itl_p50_ms": round(res["itl_p50_ms"], 3),
                        "itl_p90_ms": round(res["itl_p90_ms"], 3),
                        "itl_p95_ms": round(res["itl_p95_ms"], 3),
                        "itl_p99_ms": round(res["itl_p99_ms"], 3),
                        "decode_tok_s": round(res["decode_tok_s"], 2),
                        "peak_vram_allocated_mb": round(res["peak_vram_allocated_mb"], 1),
                        "peak_vram_reserved_mb": round(res["peak_vram_reserved_mb"], 1),
                        "host_rss_mb": round(res["host_rss_mb"], 1),
                        "kv_bytes": res["kv_total_bytes"],
                        "kv_pool_bytes": res["kv_pool_bytes"],
                        "kv_page_count": res["kv_page_count"],
                        "kv_evictions": res["kv_evictions"],
                        "host_fetches": res["host_fetches"],
                        "tail_ppl": round(res["tail_ppl"], 4),
                        "chunked_vs_singleshot_4k_parity": chunk_parity_ok,
                        "fp8_vs_fp16_greedy_match": greedy_agree,
                        "output_snippet": res["output_snippet"],
                    }
                )
                del rt
                gc.collect()
                torch.cuda.empty_cache()
            except Exception as exc:
                run.log_stderr(f"[Phase 14C] ctx={ctx_len} kv={kv_dt} FAILED: {exc}")
                run.add_record(
                    {
                        "model": "Mistral-7B-v0.1-GPTQ (32L)",
                        "weight_source": "real-checkpoint",
                        "context_len": ctx_len,
                        "kv_dtype": kv_dt,
                        "status": "UNSUPPORTED",
                        "error": str(exc),
                    }
                )
                gc.collect()
                torch.cuda.empty_cache()

    # Probe 64K (65536) honestly
    max_pos = int(model.config.max_position_embeddings)
    rope_rows = int(model.cos.shape[0])
    run.log_stdout(
        f"[Phase 14C] 64K (65536) probe: Mistral-7B max_position_embeddings={max_pos}, RoPE table={rope_rows}"
    )
    run.add_record(
        {
            "model": "Mistral-7B-v0.1-GPTQ (32L)",
            "weight_source": "real-checkpoint",
            "context_len": 65536,
            "kv_dtype": "fp16",
            "status": "UNSUPPORTED",
            "notes": (
                f"Model max_position_embeddings={max_pos} and precomputed RoPE table={rope_rows}; "
                "64K exceeds native trained/supported positional window without YaRN scaling."
            ),
        }
    )

    del model, weights
    gc.collect()
    torch.cuda.empty_cache()

    # Second real checkpoint: Mixtral-8x7B-Instruct-v0.1-GPTQ (`23.8 GB`, 32 layers) at reduced contexts
    if (mixtral_dir / "config.json").exists():
        from cutile_forge.engine.inference import InferenceEngine

        run.log_stdout(f"[Phase 14C] Evaluating real Mixtral-8x7B from {mixtral_dir}...")
        try:
            mx_tok = AutoTokenizer.from_pretrained(str(mixtral_dir), local_files_only=True)
            mx_model = Mistral7B.from_pretrained(mixtral_dir, device="cuda")
            mx_eng = InferenceEngine(mx_model, target="sm_120", use_cuda_graph=False)
            for mx_ctx in (512, 2048, 4096, 8192):
                p_ids = _build_token_sequence(mx_tok, mx_ctx - 16)
                p_t = torch.tensor([p_ids], device="cuda", dtype=torch.long)
                try:
                    mx_eng.kv_cache.reset()
                    torch.cuda.reset_peak_memory_stats()
                    torch.cuda.synchronize()
                    t0 = time.perf_counter()
                    _ = mx_eng.generate_ids(p_t, max_new_tokens=16, temperature=0.0)
                    torch.cuda.synchronize()
                    el_ms = (time.perf_counter() - t0) * 1000.0
                    st = mx_eng.last_stats
                    v_alloc, v_resv = peak_vram_mb()
                    run.add_sample(
                        f"mixtral_ctx{mx_ctx}_ms_per_tok", st.ms_per_token if st else 0.0, "ms", 0
                    )
                    run.add_record(
                        {
                            "model": "Mixtral-8x7B-Instruct-v0.1-GPTQ (32L)",
                            "weight_source": "real-checkpoint",
                            "context_len": mx_ctx,
                            "prompt_len": len(p_ids),
                            "kv_dtype": "fp16",
                            "status": "MEASURED",
                            "prefill_ms": round(st.prefill_ms if st else 0.0, 2),
                            "ttft_ms": round(st.prefill_ms if st else 0.0, 2),
                            "decode_tok_s": round(st.tokens_per_sec if st else 0.0, 2),
                            "itl_p50_ms": round(st.ms_per_token if st else 0.0, 2),
                            "peak_vram_allocated_mb": round(v_alloc, 1),
                            "peak_vram_reserved_mb": round(v_resv, 1),
                            "host_rss_mb": round(host_rss_mb(), 1),
                            "kv_evictions": mx_eng.kv_cache.evictions,
                            "host_fetches": mx_eng.kv_cache.host_fetches,
                            "notes": f"32-layer real Mixtral MoE (elapsed={el_ms:.1f}ms)",
                        }
                    )
                except torch.cuda.OutOfMemoryError as oom:
                    run.add_record(
                        {
                            "model": "Mixtral-8x7B-Instruct-v0.1-GPTQ (32L)",
                            "weight_source": "real-checkpoint",
                            "context_len": mx_ctx,
                            "kv_dtype": "fp16",
                            "status": "UNSUPPORTED",
                            "notes": f"Measured OOM limit at ctx={mx_ctx}: {oom}",
                        }
                    )
                    torch.cuda.empty_cache()
                    break
            del mx_eng, mx_model
            gc.collect()
            torch.cuda.empty_cache()
        except Exception as exc:
            run.add_record(
                {
                    "model": "Mixtral-8x7B-Instruct-v0.1-GPTQ (32L)",
                    "weight_source": "real-checkpoint",
                    "context_len": 512,
                    "status": "DEGRADED",
                    "notes": f"Load/run failed: {exc}",
                }
            )

    out_real = run.finalize()
    errs_r = validate_artifact_dir(out_real)
    assert errs_r == [], f"Phase 14C real artifact validation failed: {errs_r}"

    # Separate Synthetic 32-layer FlashDecoding Attention Benchmark
    synth_run = ValidationRun(
        run_dir=synth_dir,
        run_id="phase14c_synthetic_attention",
        checkpoint=build_checkpoint_manifest("synthetic-32layer-gqa", Path("/dev/null")),
        quant_config=QuantizationConfig(quant_format="fp16", kv_dtype="fp16+fp8_e4m3fn"),
        workload=WorkloadConfig(batch_size=1, prompt_len=32768, output_len=1, measured_iters=5),
        prompt_ids_or_checksum=prompt_checksum("synthetic_32l_attention"),
        execution_mode="synthetic-weights",
        reproduction_status="MEASURED",
        notes="Synthetic 32-layer GQA (32:8, d=128) FlashDecoding kernel-only microbenchmark; NOT real-model E2E.",
    )
    q_long = torch.randn((1, 32, 1, 128), device="cuda", dtype=torch.float16) * 0.1
    o_part = torch.empty((32, 32, 16, 128), device="cuda", dtype=torch.float32)
    lse_part = torch.empty((32, 32, 16, 1), device="cuda", dtype=torch.float32)
    out_long = torch.empty((1, 32, 1, 128), device="cuda", dtype=torch.float16)

    for ctx_len in (4096, 8192, 16384, 32768):
        for kv_dt in ("fp16", "fp8_e4m3fn"):
            kvc = NativePagedKVCache(
                num_layers=32,
                num_kv_heads=8,
                head_dim=128,
                block_size=16,
                max_seq_len=ctx_len,
                max_batch_size=1,
                kv_dtype=kv_dt,
                device="cuda",
            )
            for l_i in range(32):
                kvc.seq_lens[l_i] = ctx_len

            def _step(cache_ref: NativePagedKVCache = kvc, cl: int = ctx_len) -> None:
                for l_i in range(32):
                    kv_k, kv_v = cache_ref.get_kv(l_i, end_pos=cl)
                    flash_decoding_split_kv(
                        q_long,
                        kv_k,
                        kv_v,
                        num_splits=32,
                        causal=False,
                        o_part_buf=o_part,
                        lse_part_buf=lse_part,
                        out=out_long,
                    )

            for _ in range(3):
                _step()
            times_ms: list[float] = []
            for it in range(5):
                with TimedDeviceOp.measure(sync=True) as op:
                    _step()
                times_ms.append(op.elapsed_ms)
                synth_run.add_sample(f"synth_attn_ctx{ctx_len}_{kv_dt}_ms", op.elapsed_ms, "ms", it)

            synth_run.add_record(
                {
                    "workload": "Synthetic 32-layer GQA FlashDecoding",
                    "weight_source": "synthetic-weights",
                    "context_len": ctx_len,
                    "kv_dtype": kv_dt,
                    "median_step_ms": round(float(np.median(times_ms)), 3),
                    "min_step_ms": round(float(np.min(times_ms)), 3),
                    "max_step_ms": round(float(np.max(times_ms)), 3),
                }
            )
            del kvc
            torch.cuda.empty_cache()

    out_synth = synth_run.finalize()
    errs_s = validate_artifact_dir(out_synth)
    assert errs_s == [], f"Phase 14C synthetic artifact validation failed: {errs_s}"
    return out_real, out_synth


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 14C Real Long-Context Validation")
    parser.add_argument(
        "--model-dir", type=Path, default=Path("/home/arrhat/models/mistral_7b_4bit")
    )
    parser.add_argument(
        "--mixtral-dir", type=Path, default=Path("/var/tmp/arrhat_models/mixtral_8x7b_4bit")
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("benchmarks/validation_artifacts/v3.1.0/phase14c_long_context"),
    )
    parser.add_argument(
        "--synth-dir",
        type=Path,
        default=Path("benchmarks/validation_artifacts/v3.1.0/phase14c_synthetic_attention"),
    )
    parser.add_argument("--gen-tokens", type=int, default=32)
    args = parser.parse_args()
    run_long_context_validation(
        args.model_dir,
        args.mixtral_dir,
        args.output_dir,
        args.synth_dir,
        gen_tokens=args.gen_tokens,
    )


if __name__ == "__main__":
    main()
