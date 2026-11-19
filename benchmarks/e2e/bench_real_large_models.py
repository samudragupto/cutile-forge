"""Evaluate real Mixtral-8x7B-Instruct-v0.1-GPTQ and Llama-2-70B-Chat-GPTQ on RTX 5090."""

from __future__ import annotations

import argparse
import gc
import math
import os
import time
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

from cutile_forge.bench.harness import write_csv
from cutile_forge.engine.inference import InferenceEngine
from cutile_forge.engine.model import Mistral7B
from cutile_forge.utils.memory import get_peak_vram_gb, reset_peak_vram


def _host_rss_gb() -> float:
    try:
        with open("/proc/self/status", encoding="utf-8") as f:
            for line in f:
                if line.startswith("VmRSS:"):
                    kb = int(line.split()[1])
                    return kb / (1024.0 * 1024.0)
    except OSError:
        pass
    return 0.0


def _compute_ppl(model: Mistral7B, tokenizer: Any, text: str) -> float:
    ids = tokenizer.encode(text, return_tensors="pt").to(model.device)
    if ids.shape[1] < 4:
        return 0.0
    with torch.inference_mode():
        logits = model.forward(ids[:, :-1], kv_cache=None, pos_offset=0)
        targets = ids[:, 1:]
        loss = F.cross_entropy(
            logits.float().reshape(-1, logits.shape[-1]),
            targets.reshape(-1),
        )
    return float(math.exp(min(20.0, float(loss.item()))))


def eval_mixtral(model_dir: Path) -> dict[str, Any]:
    from transformers import AutoTokenizer

    print(f"\n=== [1/2] Loading REAL Mixtral-8x7B-Instruct-v0.1-GPTQ from {model_dir} ===")
    torch.cuda.init()
    reset_peak_vram()
    t_load0 = time.perf_counter()
    tok = AutoTokenizer.from_pretrained(str(model_dir))
    model = Mistral7B.from_pretrained(model_dir, device="cuda")
    load_s = time.perf_counter() - t_load0

    n_res = sum(1 for lw in model.weights.layers if lw.input_layernorm.device.type == "cuda")
    n_str = len(model.weights.layers) - n_res
    print(
        f"Loaded Mixtral in {load_s:.1f}s ({n_res} HBM / {n_str} PCIe, VRAM={get_peak_vram_gb():.2f} GB)"
    )

    engine = InferenceEngine(model, target="sm_120", use_cuda_graph=False)
    prompt = "[INST] Explain how GPU shared memory bank conflicts occur and how tile swizzling fixes them. [/INST]"
    prompt_ids = tok.encode(prompt, return_tensors="pt")

    # Warmup 2 tokens
    _ = engine.generate_ids(prompt_ids, max_new_tokens=2, temperature=0.0)
    reset_peak_vram()

    out_ids = engine.generate_ids(prompt_ids, max_new_tokens=64, temperature=0.0)
    stats = engine.last_stats
    assert stats is not None
    gen_text = tok.decode(out_ids, skip_special_tokens=True)

    ppl_sample = (
        "GPU shared memory is divided into 32 banks of 32-bit words. "
        "When multiple threads in a warp access different addresses in the same bank, "
        "bank conflicts serialize the memory transactions and reduce effective throughput."
    )
    ppl = _compute_ppl(model, tok, ppl_sample)

    # Estimate active MoE TFLOPS during decode (2 active experts * 3 * 2 * d * ffn * 32 layers)
    cfg = model.config
    active_flops_per_tok = 32 * (
        2 * cfg.hidden_size * cfg.hidden_size * 2
        + 2 * 2 * cfg.hidden_size * cfg.num_key_value_heads * cfg.head_dim
        + 2 * cfg.top_k_experts * 3 * cfg.hidden_size * cfg.intermediate_size
    )
    active_tflops = (active_flops_per_tok / 1e12) / max(stats.ms_per_token / 1e3, 1e-6)
    prefill_tps = (stats.prompt_tokens * 1000.0) / max(stats.prefill_ms, 1e-6)
    peak_vram = get_peak_vram_gb()
    rss_gb = _host_rss_gb()

    print(
        f"Mixtral-8x7B | prefill={prefill_tps:.1f} tok/s ({stats.prefill_ms:.1f} ms) | "
        f"decode={stats.tokens_per_sec:.2f} tok/s ({stats.ms_per_token:.2f} ms/tok) | "
        f"active={active_tflops:.2f} TFLOPS | PPL={ppl:.2f} | VRAM={peak_vram:.2f} GB"
    )
    print(f"Mixtral Generated (64 tok):\n{gen_text}\n")

    row: dict[str, Any] = {
        "model": "Mixtral-8x7B-Instruct-v0.1-GPTQ",
        "resident_layers": n_res,
        "streamed_layers": n_str,
        "prefill_ms": round(stats.prefill_ms, 2),
        "prefill_tok_s": round(prefill_tps, 2),
        "decode_tok_s": round(stats.tokens_per_sec, 2),
        "ms_per_tok": round(stats.ms_per_token, 2),
        "active_tflops": round(active_tflops, 2),
        "perplexity": round(ppl, 2),
        "pcie_overlap_pct": 0.0,
        "peak_vram_gb": round(peak_vram, 2),
        "host_rss_gb": round(rss_gb, 2),
        "sample_output": gen_text.replace("\n", " "),
    }

    del engine, model
    gc.collect()
    torch.cuda.empty_cache()
    return row


def eval_llama70b(model_dir: Path) -> list[dict[str, Any]]:
    from transformers import AutoTokenizer

    print(f"\n=== [2/2] Loading REAL Llama-2-70B-Chat-GPTQ from {model_dir} ===")
    torch.cuda.init()
    reset_peak_vram()
    t_load0 = time.perf_counter()
    tok = AutoTokenizer.from_pretrained(str(model_dir))
    model = Mistral7B.from_pretrained(model_dir, device="cuda")
    load_s = time.perf_counter() - t_load0

    n_res = sum(1 for lw in model.weights.layers if lw.input_layernorm.device.type == "cuda")
    n_str = len(model.weights.layers) - n_res
    print(
        f"Loaded Llama-2-70B in {load_s:.1f}s ({n_res} HBM / {n_str} PCIe, "
        f"VRAM={get_peak_vram_gb():.2f} GB, Host RSS={_host_rss_gb():.2f} GB)"
    )

    engine = InferenceEngine(model, target="sm_120", use_cuda_graph=False)
    prompts = [
        "The capital of France is",
        "Explain the difference between latency and throughput in GPU compilers:",
    ]

    # Warmup 1 token
    _ = engine.generate_ids(
        tok.encode(prompts[0], return_tensors="pt"), max_new_tokens=2, temperature=0.0
    )
    reset_peak_vram()

    rows: list[dict[str, Any]] = []
    for p_idx, prompt in enumerate(prompts):
        p_ids = tok.encode(prompt, return_tensors="pt")
        out_ids = engine.generate_ids(p_ids, max_new_tokens=32, temperature=0.0)
        stats = engine.last_stats
        assert stats is not None
        gen_text = tok.decode(out_ids, skip_special_tokens=True)
        prefill_tps = (stats.prompt_tokens * 1000.0) / max(stats.prefill_ms, 1e-6)
        peak_vram = get_peak_vram_gb()
        rss_gb = _host_rss_gb()

        # Estimate PCIe overlap from streamed bytes at 49.09 GB/s
        streamed_gb = (n_str * 435.0) / 1024.0
        dma_est_ms = (streamed_gb / 49.09) * 1000.0
        overlap_pct = max(
            0.0,
            min(
                100.0,
                (1.0 - max(0.0, stats.ms_per_token - dma_est_ms) / max(stats.ms_per_token, 1e-6))
                * 100.0,
            ),
        )

        print(
            f"Llama-2-70B [prompt {p_idx + 1}] | prefill={stats.prefill_ms:.1f} ms | "
            f"decode={stats.tokens_per_sec:.2f} tok/s ({stats.ms_per_token:.2f} ms/tok) | "
            f"VRAM={peak_vram:.2f} GB | RSS={rss_gb:.2f} GB"
        )
        print(f"Prompt: {prompt}\nOutput: {gen_text}\n")

        rows.append(
            {
                "model": f"Llama-2-70B-Chat-GPTQ (prompt {p_idx + 1})",
                "resident_layers": n_res,
                "streamed_layers": n_str,
                "prefill_ms": round(stats.prefill_ms, 2),
                "prefill_tok_s": round(prefill_tps, 2),
                "decode_tok_s": round(stats.tokens_per_sec, 2),
                "ms_per_tok": round(stats.ms_per_token, 2),
                "active_tflops": 0.0,
                "perplexity": 0.0,
                "pcie_overlap_pct": round(overlap_pct, 1),
                "peak_vram_gb": round(peak_vram, 2),
                "host_rss_gb": round(rss_gb, 2),
                "sample_output": gen_text.replace("\n", " "),
            }
        )

    del engine, model
    gc.collect()
    torch.cuda.empty_cache()
    return rows


def main() -> None:
    storage_root = Path(os.environ.get("STORAGE_ROOT", "/var/tmp/arrhat_models"))
    parser = argparse.ArgumentParser(description="Evaluate real Mixtral-8x7B and Llama-2-70B GPTQ")
    parser.add_argument("--mixtral", type=Path, default=storage_root / "mixtral_8x7b_4bit")
    parser.add_argument("--llama70b", type=Path, default=storage_root / "llama_70b_4bit")
    parser.add_argument(
        "--output", type=Path, default=Path("benchmarks/results_5090/real_large_models.csv")
    )
    args = parser.parse_args()

    rows: list[dict[str, Any]] = []
    if (args.mixtral / "config.json").exists():
        rows.append(eval_mixtral(args.mixtral))
    if (args.llama70b / "config.json").exists():
        rows.extend(eval_llama70b(args.llama70b))

    if rows:
        write_csv(args.output, rows)
        print(f"Saved real large-model evaluation to {args.output}")


if __name__ == "__main__":
    main()
