"""Phase 18 Model Coverage & Native Compatibility Matrix Builder (`bench_phase18_coverage.py`)."""

from __future__ import annotations

import argparse
import gc
import json
import subprocess
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch.utils._python_dispatch import TorchDispatchMode

from benchmarks.production.manifest import build_checkpoint_manifest, prompt_checksum
from benchmarks.production.runner import ValidationRun
from benchmarks.production.schema import QuantizationConfig, WorkloadConfig, validate_artifact_dir
from cutile_forge.models import (
    MODEL_REGISTRY,
    UnifiedModelExecutor,
    build_model_graph,
    extract_hf_clip_vision_weights,
    extract_hf_decoder_weights,
    extract_hf_vit_weights,
    init_synthetic_weights,
)

FORBIDDEN_NATIVE_COMPUTE_ATEN = {
    "aten.mm.default",
    "aten.bmm.default",
    "aten.addmm.default",
    "aten.matmul.default",
    "aten.embedding.default",
    "aten.silu.default",
    "aten.gelu.default",
    "aten.layer_norm.default",
    "aten.native_layer_norm.default",
    "aten.scaled_dot_product_attention.default",
    "aten._scaled_dot_product_flash_attention.default",
    "aten._scaled_dot_product_efficient_attention.default",
    "aten._scaled_dot_product_cudnn_attention.default",
    "aten.conv2d.default",
    "aten.conv1d.default",
    "aten.convolution.default",
    "aten.topk.default",
    "aten.softmax.int",
    "aten._softmax.default",
}


class _ComputeATenAudit(TorchDispatchMode):
    def __init__(self) -> None:
        super().__init__()
        self.forbidden_calls: list[str] = []

    def __torch_dispatch__(
        self,
        func: Any,
        types: tuple[type, ...],
        args: tuple[Any, ...] = (),
        kwargs: dict[str, Any] | None = None,
    ) -> Any:
        name = str(func)
        if name in FORBIDDEN_NATIVE_COMPUTE_ATEN:
            self.forbidden_calls.append(name)
        return func(*args, **(kwargs or {}))


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


def _cleanup() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _eval_llama3_real() -> dict[str, Any]:
    import transformers
    from transformers import AutoModelForCausalLM, AutoTokenizer

    repo = "unsloth/Llama-3.2-1B-Instruct"
    tok = AutoTokenizer.from_pretrained(repo, local_files_only=True)
    hf_m = AutoModelForCausalLM.from_pretrained(
        repo, dtype=torch.float16, device_map="cuda", local_files_only=True
    ).eval()
    graph, weights = extract_hf_decoder_weights(hf_m, device="cuda", dtype=torch.float16)
    ex = UnifiedModelExecutor(graph)

    prompts = [
        "The capital city of France is",
        "In parallel GPU systems, Tensor Cores accelerate",
        "The theory of general relativity describes",
        "Python decorators allow functions to",
        "Water boils at 100 degrees Celsius under",
    ]
    all_cos: list[float] = []
    all_diff: list[float] = []
    exact_prompts = 0
    tok_matches = 0
    tok_total = 0
    forbidden_total: list[str] = []

    with torch.no_grad():
        for p_idx, p_str in enumerate(prompts):
            ids = tok(p_str, return_tensors="pt")["input_ids"].cuda()
            with _ComputeATenAudit() as audit:
                cu_logits = ex.forward(weights, input_ids=ids)
            if p_idx == 0:
                forbidden_total = list(audit.forbidden_calls)
            ref_logits = hf_m(ids).logits
            cu_last = cu_logits[0, -1].float()
            ref_last = ref_logits[0, -1].float()
            all_cos.append(float(F.cosine_similarity(cu_last, ref_last, dim=0).item()))
            all_diff.append(float((cu_last - ref_last).abs().max().item()))

            # 32-token greedy parity across 5 prompts
            cur = ids.clone()
            cu_gen: list[int] = []
            for _ in range(32):
                nxt = int(ex.forward(weights, input_ids=cur)[0, -1].argmax().item())
                cu_gen.append(nxt)
                cur = torch.cat(
                    [cur, torch.tensor([[nxt]], device="cuda", dtype=torch.long)], dim=1
                )
            ref_seq = hf_m.generate(ids, max_new_tokens=32, do_sample=False)
            ref_gen = [int(t) for t in ref_seq[0, ids.shape[1] :].tolist()]
            m_cnt = sum(1 for a, b in zip(cu_gen, ref_gen, strict=True) if a == b)
            tok_matches += m_cnt
            tok_total += 32
            if cu_gen == ref_gen:
                exact_prompts += 1

    min_cos = min(all_cos)
    max_diff = max(all_diff)
    ok = min_cos >= 0.9999 and len(forbidden_total) == 0 and exact_prompts == len(prompts)
    res = {
        "family": "llama3",
        "checkpoint": repo,
        "reference": f"transformers=={transformers.__version__}",
        "layers": graph.num_layers,
        "status": "PASS" if ok else "FAIL",
        "native_execution": len(forbidden_total) == 0,
        "forbidden_aten_calls": len(forbidden_total),
        "weight_source": "real-checkpoint",
        "metrics": {
            "min_logit_cos_sim": round(min_cos, 6),
            "max_abs_diff": round(max_diff, 5),
            "greedy_exact_prompts": f"{exact_prompts}/{len(prompts)}",
            "greedy_tok_match_pct": round(100.0 * tok_matches / max(1, tok_total), 2),
        },
        "notes": "16L scaled Llama-3 RoPE all-layer native CuTile execution",
    }
    del hf_m, weights, ex
    _cleanup()
    return res


def _eval_vit_real() -> dict[str, Any]:
    import transformers
    from transformers import ViTForImageClassification

    repo = "google/vit-base-patch16-224"
    hf_v = ViTForImageClassification.from_pretrained(
        repo, dtype=torch.float16, device_map="cuda", local_files_only=True
    ).eval()
    graph, weights = extract_hf_vit_weights(hf_v, device="cuda", dtype=torch.float16)
    ex = UnifiedModelExecutor(graph)

    torch.manual_seed(42)
    pix = torch.randn((1, 3, 224, 224), device="cuda", dtype=torch.float16) * 0.5
    with torch.no_grad():
        with _ComputeATenAudit() as audit:
            cu_out = ex.forward(weights, pixel_values=pix).float()
        ref_out = hf_v(pixel_values=pix).logits.float()

    cos = float(F.cosine_similarity(cu_out.flatten(), ref_out.flatten(), dim=0).item())
    diff = float((cu_out - ref_out).abs().max().item())
    ok = cos >= 0.9999 and diff < 5e-2 and len(audit.forbidden_calls) == 0
    res = {
        "family": "vit",
        "checkpoint": repo,
        "reference": f"transformers=={transformers.__version__}",
        "layers": graph.num_layers,
        "status": "PASS" if ok else "FAIL",
        "native_execution": len(audit.forbidden_calls) == 0,
        "forbidden_aten_calls": len(audit.forbidden_calls),
        "weight_source": "real-checkpoint",
        "metrics": {"cls_logit_cos_sim": round(cos, 6), "max_abs_diff": round(diff, 5)},
        "notes": "12L ViT-B/16 native patch GEMM, LayerNorm, bidirectional FlashAttention, exact erf-GELU",
    }
    del hf_v, weights, ex
    _cleanup()
    return res


def _eval_clip_vision_real() -> dict[str, Any]:
    import transformers
    from transformers import CLIPVisionModel

    repo = "openai/clip-vit-base-patch32"
    hf_c = CLIPVisionModel.from_pretrained(
        repo, dtype=torch.float16, device_map="cuda", local_files_only=True
    ).eval()
    graph, weights = extract_hf_clip_vision_weights(hf_c, device="cuda", dtype=torch.float16)
    ex = UnifiedModelExecutor(graph)

    torch.manual_seed(42)
    pix = torch.randn((1, 3, 224, 224), device="cuda", dtype=torch.float16) * 0.5
    with torch.no_grad():
        with _ComputeATenAudit() as audit:
            cu_out = ex.forward(weights, pixel_values=pix).float()
        ref_out = hf_c(pixel_values=pix).pooler_output.float()

    cos = float(F.cosine_similarity(cu_out.flatten(), ref_out.flatten(), dim=0).item())
    diff = float((cu_out - ref_out).abs().max().item())
    ok = cos >= 0.9999 and diff < 5e-2 and len(audit.forbidden_calls) == 0
    res = {
        "family": "clip_vision",
        "checkpoint": repo,
        "reference": f"transformers=={transformers.__version__}",
        "layers": graph.num_layers,
        "status": "PASS" if ok else "FAIL",
        "native_execution": len(audit.forbidden_calls) == 0,
        "forbidden_aten_calls": len(audit.forbidden_calls),
        "weight_source": "real-checkpoint",
        "metrics": {"pooler_cos_sim": round(cos, 6), "max_abs_diff": round(diff, 5)},
        "notes": "12L CLIP-ViT-B/32 native patch GEMM, pre/post LayerNorm, QuickGELU, FlashAttention",
    }
    del hf_c, weights, ex
    _cleanup()
    return res


def _audit_synthetic_zero_aten(family: str) -> int:
    overrides: dict[str, Any] = {
        "vocab_size": 128,
        "hidden_size": 64,
        "intermediate_size": 128,
        "num_hidden_layers": 2,
        "num_attention_heads": 4,
        "num_key_value_heads": 2 if family not in ("deepseek_moe", "qwen2_moe") else 4,
        "head_dim": 16,
    }
    if family in ("mixtral", "deepseek_moe", "qwen2_moe"):
        overrides["num_local_experts"] = 4
        overrides["num_experts_per_tok"] = 2
        if family != "mixtral":
            overrides["shared_expert_intermediate_size"] = 64
    graph = build_model_graph(family, **overrides)
    weights = init_synthetic_weights(graph, seed=42, device="cuda", dtype=torch.float16)
    ex = UnifiedModelExecutor(graph)
    ids = torch.randint(0, 128, (1, 16), device="cuda", dtype=torch.long)
    with torch.no_grad():
        with _ComputeATenAudit() as audit:
            _ = ex.forward(weights, input_ids=ids)
    return len(audit.forbidden_calls)


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 18 Coverage & Compatibility Matrix")
    parser.add_argument(
        "--regress-json",
        type=Path,
        default=Path("benchmarks/validation_artifacts/v3.5.0/phase18_regression/regress_full.json"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("benchmarks/validation_artifacts/v3.5.0/phase18_coverage"),
    )
    args = parser.parse_args()

    contended, gpu_apps = _query_contention()
    print(f"[Phase 18] GPU compute apps: {gpu_apps} (contended={contended})")

    ckpt = build_checkpoint_manifest(
        model_id="phase18-13-family-coverage",
        local_path="unsloth/Llama-3.2-1B-Instruct",
    )
    run = ValidationRun(
        run_dir=args.output,
        run_id="phase18-model-coverage",
        checkpoint=ckpt,
        quant_config=QuantizationConfig(quant_format="fp16_and_gptq_int4", kv_dtype="fp16"),
        workload=WorkloadConfig(batch_size=1, prompt_len=64, output_len=32, measured_iters=1),
        prompt_ids_or_checksum=prompt_checksum("phase18_13_model_coverage"),
        execution_mode="real-checkpoint",
        reproduction_status="MEASURED",
    )

    newly_real = [_eval_llama3_real(), _eval_vit_real(), _eval_clip_vision_real()]
    for r in newly_real:
        run.add_record({"stage": "new_real_checkpoint", **r})
        print(
            f"  [REAL] {r['family']:<12s} | status={r['status']} | native={r['native_execution']} "
            f"(forbidden={r['forbidden_aten_calls']}) | metrics={r['metrics']}"
        )

    regress_by_model: dict[str, dict[str, Any]] = {}
    if args.regress_json.exists():
        reg = json.loads(args.regress_json.read_text(encoding="utf-8"))
        for r in reg.get("results", []):
            regress_by_model[str(r["model"])] = r

    synth_only_families = {}
    for fam in ("qwen2_moe", "deepseek_moe"):
        forb = _audit_synthetic_zero_aten(fam)
        synth_only_families[fam] = {
            "family": fam,
            "checkpoint": "synthetic-weights",
            "reference": "UnifiedModelExecutor(use_cutile=False)",
            "layers": 2,
            "status": "SYNTHETIC-ONLY",
            "native_execution": forb == 0,
            "forbidden_aten_calls": forb,
            "weight_source": "synthetic-weights",
            "metrics": {"forbidden_aten_calls": forb},
            "notes": "No <16 GB ungated FP16 checkpoint fits side-by-side in 32 GB VRAM for reference comparison",
        }
        run.add_record({"stage": "synthetic_only", **synth_only_families[fam]})

    reg_map = {
        "mistral": "Mistral-7B",
        "llama": "Llama-2-70B",
        "qwen2": "Qwen2.5-0.5B",
        "mixtral": "Mixtral-8x7B",
        "siglip_vision": "SigLIP ViT-B/16",
        "whisper_encoder": "Whisper-Tiny",
        "llava": "LLaVA-1.5",
        "dit": "DiT-XL/2",
    }
    real_map = {r["family"]: r for r in newly_real}

    matrix: list[dict[str, Any]] = []
    for fam in sorted(MODEL_REGISTRY.keys()):
        if fam in real_map:
            r = real_map[fam]
            matrix.append(
                {
                    "family": fam,
                    "modality": MODEL_REGISTRY[fam].modality,
                    "checkpoint": r["checkpoint"],
                    "weight_source": "real-checkpoint",
                    "native_execution": r["native_execution"],
                    "status": r["status"],
                    "key_metrics": r["metrics"],
                    "notes": r["notes"],
                }
            )
        elif fam in reg_map and reg_map[fam] in regress_by_model:
            rr = regress_by_model[reg_map[fam]]
            matrix.append(
                {
                    "family": fam,
                    "modality": MODEL_REGISTRY[fam].modality,
                    "checkpoint": rr["checkpoint"],
                    "weight_source": rr["weight_source"],
                    "native_execution": rr["native_execution"],
                    "status": rr["status"],
                    "key_metrics": rr["metrics"],
                    "notes": rr["notes"],
                }
            )
        elif fam in synth_only_families:
            sr = synth_only_families[fam]
            matrix.append(
                {
                    "family": fam,
                    "modality": MODEL_REGISTRY[fam].modality,
                    "checkpoint": sr["checkpoint"],
                    "weight_source": "synthetic-weights",
                    "native_execution": sr["native_execution"],
                    "status": "SYNTHETIC-ONLY",
                    "key_metrics": sr["metrics"],
                    "notes": sr["notes"],
                }
            )
        else:
            matrix.append(
                {
                    "family": fam,
                    "modality": MODEL_REGISTRY[fam].modality,
                    "checkpoint": "unknown",
                    "weight_source": "none",
                    "native_execution": False,
                    "status": "NOT RUN",
                    "key_metrics": {},
                    "notes": "Regression report missing entry",
                }
            )

    run.finalize()
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "compatibility_matrix.json").write_text(
        json.dumps(
            {"git_commit": run._git_commit, "contended": contended, "matrix": matrix}, indent=2
        ),
        encoding="utf-8",
    )

    lines = [
        "# Phase 18 Model Coverage & Compatibility Matrix (`v3.5.0`)",
        "",
        f"- Commit: `{run._git_commit}` | Contended: `{contended}`",
        "",
        "| Family | Modality | Checkpoint | Weight Source | Native CuTile | Status | Notes |",
        "|---|---|---|---|---|---|---|",
    ]
    for m in matrix:
        lines.append(
            f"| `{m['family']}` | {m['modality']} | `{m['checkpoint']}` | `{m['weight_source']}` | "
            f"`{m['native_execution']}` | **{m['status']}** | {m['notes']} |"
        )
    (args.output / "compatibility_matrix.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    errs = validate_artifact_dir(args.output)
    if errs:
        raise RuntimeError(f"Validation artifact check failed: {errs}")


if __name__ == "__main__":
    main()
