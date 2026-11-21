"""Phase 14E: End-to-End Quantization Parity Validation (Real GPTQ + Repacked Multi-Format)."""

from __future__ import annotations

import argparse
import gc
import time
from pathlib import Path

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
from cutile_forge.kernels.universal_quant import (
    MixedPrecisionFormatTable,
    QuantFormat,
    UniversalQuantModel,
)
from cutile_forge.runtime import NativeModelRuntime

WIKITEXT2_EVAL_SLICE = (
    " = Valkyria Chronicles III = \n"
    " Senjō no Valkyria 3 : Unrecorded Chronicles ( Japanese : 戦場のヴァルキュリア3 , "
    "lit . Valkyria of the Battlefield 3 ) , commonly referred to as Valkyria Chronicles III "
    "outside Japan , is a tactical role @-@ playing video game developed by Sega and "
    "Media.Vision for the PlayStation Portable . Released in January 2011 in Japan , it is "
    "the third game in the Valkyria series . Employing the same fusion of tactical and real "
    "@-@ time gameplay as its predecessors , the story runs parallel to the first game and "
    'follows the " Nameless " , a penal military unit serving the nation of Gallia during '
    "the Second Europan War who perform secret black operations and are pitted against the "
    'Imperial unit " Calamity Raven " . '
    "The game began development in 2010 , carrying over a large portion of the work done on "
    "Valkyria Chronicles II . While it retained the standard features of the series , it also "
    "rebuilt most of the core elements to make the gameplay smoother and more responsive . "
)


def _greedy_generate_uqm(
    u_model: UniversalQuantModel,
    prompt_ids: list[int],
    max_new_tokens: int,
    mode: str,
) -> list[int]:
    cur = torch.tensor([prompt_ids], device=u_model.device, dtype=torch.long)
    out_ids: list[int] = []
    for _ in range(max_new_tokens):
        logits = u_model.forward(cur, mode=mode)
        nxt = int(logits[0, -1].argmax().item())
        out_ids.append(nxt)
        cur = torch.cat(
            [cur, torch.tensor([[nxt]], device=u_model.device, dtype=torch.long)], dim=1
        )
    return out_ids


def _first_divergence(a: list[int], b: list[int]) -> int | None:
    for idx, (x, y) in enumerate(zip(a, b, strict=False)):
        if x != y:
            return idx
    return None


def run_quant_e2e_parity(
    model_dir: Path,
    output_dir: Path,
    multi_format_layers: int = 8,
) -> Path:
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(str(model_dir), local_files_only=True)
    ppl_ids = tokenizer(WIKITEXT2_EVAL_SLICE, return_tensors="pt")["input_ids"][:, :256].to("cuda")
    ppl_len = int(ppl_ids.shape[1])

    prompts = [
        tokenizer.encode("The capital of France is", add_special_tokens=True),
        tokenizer.encode("def matrix_multiply(A, B):", add_special_tokens=True),
    ]
    ckpt = build_checkpoint_manifest("TheBloke/Mistral-7B-v0.1-GPTQ", model_dir)

    run = ValidationRun(
        run_dir=output_dir,
        run_id="phase14e_quant_e2e_parity",
        checkpoint=ckpt,
        quant_config=QuantizationConfig(
            quant_format="gptq+universal",
            group_size=128,
            desc_act=True,
            kv_dtype="fp16",
            disk_format="gptq_int4_desc_act",
            kernel_format="cutile_native_multi_format",
        ),
        workload=WorkloadConfig(
            batch_size=1,
            prompt_len=len(prompts[0]),
            output_len=32,
            temperature=0.0,
            seed=42,
            warmup_iters=2,
            measured_iters=5,
        ),
        prompt_ids_or_checksum=prompt_checksum(WIKITEXT2_EVAL_SLICE),
        execution_mode="real-checkpoint",
        reproduction_status="MEASURED",
        notes=(
            f"WikiText-2 slice tokens={ppl_len} (ctx={ppl_len}), SHA-256 recorded; "
            f"native GPTQ runs full 32 layers (`END_TO_END_VERIFIED`), other formats repacked from real Mistral-7B."
        ),
    )

    # ------------------------------------------------------------------
    # Part 1: Full 32-layer real on-disk GPTQ validation (desc_act=True, gs=128)
    # ------------------------------------------------------------------
    torch.cuda.reset_peak_memory_stats()
    t_load0 = time.perf_counter()
    weights_32 = load_gptq_weights(model_dir, device="cuda", pin_ddr5=False)
    model_32 = Mistral7B(
        weights=weights_32,
        gpu_config=GPUConfig(vram_gb=32.0, stream_weights=False),
        device="cuda",
    )
    load_32_s = time.perf_counter() - t_load0

    rt = NativeModelRuntime(model_32, max_seq_len=2048, use_cuda_graph=True, max_batch_size=1)
    # Warmup
    _, _ = rt.generate(prompts[0], max_new_tokens=8, sync_interval=8)

    greedy_matches = 0
    first_divs: list[int | None] = []
    decode_rates: list[float] = []
    prefill_times: list[float] = []

    for p_idx, p_ids in enumerate(prompts):
        nat_toks, st = rt.generate(p_ids, max_new_tokens=16, sync_interval=16)
        decode_rates.append(float(st["decode_tok_s"]))
        prefill_times.append(float(st["prefill_ms"]))
        # Compare token-by-token greedy against Mistral7B PyTorch reference
        ref_toks: list[int] = []
        cur_t = torch.tensor([p_ids], device="cuda", dtype=torch.long)
        with torch.inference_mode():
            for _ in range(16):
                lg = model_32.forward(
                    cur_t, kv_cache=None, pos_offset=0, backend="pytorch", last_token_only=True
                )
                nxt = int(lg[0, -1].argmax().item())
                ref_toks.append(nxt)
                cur_t = torch.cat(
                    [cur_t, torch.tensor([[nxt]], device="cuda", dtype=torch.long)], dim=1
                )
        div = _first_divergence(nat_toks, ref_toks)
        first_divs.append(div)
        if div is None:
            greedy_matches += 1
        run.add_sample(
            f"real_gptq_32l_prompt{p_idx}_tok_s", float(st["decode_tok_s"]), "tok/s", p_idx
        )

    # Next-token logit error & cosine similarity on prompt 0
    p0_t = torch.tensor([prompts[0]], device="cuda", dtype=torch.long)
    with torch.inference_mode():
        lg_cut = model_32.forward(
            p0_t, kv_cache=None, backend="cutile", last_token_only=True
        ).float()
        lg_ref = model_32.forward(
            p0_t, kv_cache=None, backend="pytorch", last_token_only=True
        ).float()
    logit_max_err_32 = float((lg_cut - lg_ref).abs().max().item())
    logit_cos_32 = float(F.cosine_similarity(lg_cut.flatten(), lg_ref.flatten(), dim=0).item())

    # 32-layer WikiText-2 PPL (native CuTile vs PyTorch reference)
    with torch.inference_mode():
        lg_p_nat = model_32.forward(ppl_ids, kv_cache=None, backend="cutile").float()
        lg_p_ref = model_32.forward(ppl_ids, kv_cache=None, backend="pytorch").float()
        loss_nat = torch.nn.functional.cross_entropy(
            lg_p_nat[:, :-1].reshape(-1, lg_p_nat.size(-1)), ppl_ids[:, 1:].reshape(-1)
        )
        loss_ref = torch.nn.functional.cross_entropy(
            lg_p_ref[:, :-1].reshape(-1, lg_p_ref.size(-1)), ppl_ids[:, 1:].reshape(-1)
        )
        ppl_32_nat = float(torch.exp(loss_nat).item())
        ppl_32_ref = float(torch.exp(loss_ref).item())

    v32_alloc, _ = peak_vram_mb()
    run.add_record(
        {
            "format_label": "GPTQ_on_disk_32L_real",
            "source_kind": "real-on-disk-checkpoint",
            "disk_format": "gptq_int4_gs128_desc_act",
            "kernel_format": "cutile_int4_dequant",
            "layers_evaluated": 32,
            "verification_category": "END_TO_END_VERIFIED",
            "greedy_exact_prompts": f"{greedy_matches}/{len(prompts)}",
            "first_divergence_pos": first_divs,
            "logit_max_abs_err": round(logit_max_err_32, 6),
            "logit_cosine_sim": round(logit_cos_32, 6),
            "wikitext2_tokens": ppl_len,
            "wikitext2_ppl_native": round(ppl_32_nat, 4),
            "wikitext2_ppl_reference": round(ppl_32_ref, 4),
            "wikitext2_ppl_rel_diff_pct": round(
                100.0 * abs(ppl_32_nat - ppl_32_ref) / max(ppl_32_ref, 1e-6), 4
            ),
            "prefill_ms": round(prefill_times[0], 2),
            "decode_tok_s": round(sum(decode_rates) / len(decode_rates), 2),
            "load_time_s": round(load_32_s, 2),
            "peak_vram_mb": round(v32_alloc, 1),
            "host_rss_mb": round(host_rss_mb(), 1),
        }
    )
    del rt, model_32, weights_32
    gc.collect()
    torch.cuda.empty_cache()

    # ------------------------------------------------------------------
    # Part 2: Universal formats on repacked real Mistral-7B (`multi_format_layers` layers)
    # ------------------------------------------------------------------
    base_w = load_gptq_weights(
        model_dir, device="cuda", pin_ddr5=False, max_gpu_layers=multi_format_layers
    )
    base_model = Mistral7B(
        weights=base_w,
        gpu_config=GPUConfig(vram_gb=32.0, stream_weights=False),
        device="cuda",
    )

    specs: list[tuple[str, QuantFormat, int, bool, str]] = [
        ("GPTQ_gs128_desc_repack", QuantFormat.GPTQ, 128, True, "repacked-from-real-mistral-7b"),
        ("GPTQ_gs128_nodesc", QuantFormat.GPTQ, 128, False, "repacked-from-real-mistral-7b"),
        ("GPTQ_gs64", QuantFormat.GPTQ, 64, False, "repacked-from-real-mistral-7b"),
        ("GPTQ_gs32", QuantFormat.GPTQ, 32, False, "repacked-from-real-mistral-7b"),
        ("AWQ_gs128", QuantFormat.AWQ, 128, False, "repacked-from-real-mistral-7b"),
        ("GGUF_Q4_0", QuantFormat.GGUF_Q4_0, 32, False, "repacked-from-real-mistral-7b"),
        ("GGUF_Q8_0", QuantFormat.GGUF_Q8_0, 32, False, "repacked-from-real-mistral-7b"),
        ("GGUF_Q4_K", QuantFormat.GGUF_Q4_K, 32, False, "repacked-from-real-mistral-7b"),
        ("GGUF_Q6_K", QuantFormat.GGUF_Q6_K, 16, False, "repacked-from-real-mistral-7b"),
        ("GGUF_Q4_K_M_mixed", QuantFormat.GGUF_Q4_K_M, 32, False, "repacked-from-real-mistral-7b"),
        ("INT8_gs128", QuantFormat.INT8, 128, False, "repacked-from-real-mistral-7b"),
        ("FP8_e4m3_tensor", QuantFormat.FP8, 128, False, "repacked-from-real-mistral-7b"),
        ("FP8_e4m3_block128", QuantFormat.FP8_BLOCK, 128, False, "repacked-from-real-mistral-7b"),
    ]

    p_eval = prompts[0]
    p_eval_t = torch.tensor([p_eval], device="cuda", dtype=torch.long)

    for label, fmt, gs, desc, src_kind in specs:
        table: MixedPrecisionFormatTable | QuantFormat = (
            MixedPrecisionFormatTable.gguf_q4_k_m_table(multi_format_layers)
            if fmt == QuantFormat.GGUF_Q4_K_M
            else fmt
        )
        torch.cuda.reset_peak_memory_stats()
        t_pack0 = time.perf_counter()
        u_m = UniversalQuantModel.from_mistral_model(
            base_model,
            format_table=table,
            group_size=gs,
            desc_act=desc,
            block_size=(128, 128),
            num_layers=multi_format_layers,
        )
        pack_s = time.perf_counter() - t_pack0

        with torch.inference_mode():
            out_n = u_m.forward(p_eval_t, mode="native").float()
            out_r = u_m.forward(p_eval_t, mode="reference").float()
        lg_err = float((out_n[:, -1] - out_r[:, -1]).abs().max().item())
        lg_cos = float(
            torch.nn.functional.cosine_similarity(
                out_n[:, -1].flatten(), out_r[:, -1].flatten(), dim=0
            ).item()
        )

        nat_seq = _greedy_generate_uqm(u_m, p_eval, max_new_tokens=16, mode="native")
        ref_seq = _greedy_generate_uqm(u_m, p_eval, max_new_tokens=16, mode="reference")
        div_idx = _first_divergence(nat_seq, ref_seq)

        ppl_n = u_m.compute_perplexity(ppl_ids, mode="native")
        ppl_r = u_m.compute_perplexity(ppl_ids, mode="reference")
        ppl_diff_pct = 100.0 * abs(ppl_n - ppl_r) / max(ppl_r, 1e-6)

        with TimedDeviceOp.measure(sync=True) as op_pref:
            _ = u_m.forward(p_eval_t, mode="native")
        # Decode throughput over 16 single-token forward steps
        step_times: list[float] = []
        tok_1 = torch.tensor([[1]], device="cuda", dtype=torch.long)
        for _ in range(2):
            _ = u_m.forward(tok_1, mode="native")
        for it in range(5):
            with TimedDeviceOp.measure(sync=True) as op_s:
                _ = u_m.forward(tok_1, mode="native")
            step_times.append(op_s.elapsed_ms)
            run.add_sample(f"{label}_layer_slice_step_ms", op_s.elapsed_ms, "ms", it)

        med_step_ms = float(sorted(step_times)[len(step_times) // 2])
        v_alloc, _ = peak_vram_mb()
        cat = "END_TO_END_VERIFIED" if multi_format_layers >= 32 else "LAYER_VERIFIED_ONLY"

        run.add_record(
            {
                "format_label": label,
                "source_kind": src_kind,
                "disk_format": f"repacked_{fmt.value}",
                "kernel_format": f"cutile_{fmt.value}",
                "layers_evaluated": multi_format_layers,
                "verification_category": cat,
                "greedy_16tok_exact": div_idx is None,
                "first_divergence_pos": div_idx,
                "logit_max_abs_err": round(lg_err, 6),
                "logit_cosine_sim": round(lg_cos, 6),
                "wikitext2_tokens": ppl_len,
                "wikitext2_ppl_native": round(ppl_n, 4),
                "wikitext2_ppl_reference": round(ppl_r, 4),
                "wikitext2_ppl_rel_diff_pct": round(ppl_diff_pct, 4),
                "prefill_ms": round(op_pref.elapsed_ms, 2),
                "decode_slice_tok_s": round(1000.0 / max(1e-6, med_step_ms), 2),
                "pack_load_s": round(pack_s, 2),
                "peak_vram_mb": round(v_alloc, 1),
            }
        )
        run.log_stdout(
            f"  [Phase 14E | {label:22s}] {cat} | cos={lg_cos:.6f} | div={div_idx} | "
            f"PPL={ppl_n:.4f}/{ppl_r:.4f} ({ppl_diff_pct:.4f}%)"
        )
        del u_m
        gc.collect()
        torch.cuda.empty_cache()

    out = run.finalize()
    errs = validate_artifact_dir(out)
    assert errs == [], f"Phase 14E artifact validation failed: {errs}"
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 14E Universal Quantization E2E Parity")
    parser.add_argument(
        "--model-dir", type=Path, default=Path("/home/arrhat/models/mistral_7b_4bit")
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("benchmarks/validation_artifacts/v3.1.0/phase14e_quantization"),
    )
    parser.add_argument("--layers", type=int, default=8)
    args = parser.parse_args()
    run_quant_e2e_parity(args.model_dir, args.output_dir, multi_format_layers=args.layers)


if __name__ == "__main__":
    main()
