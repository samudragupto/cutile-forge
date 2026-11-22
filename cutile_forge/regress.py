"""Phase 15 Production Regression Collection (`cutile-forge regress --tier {fast,full}`)."""

from __future__ import annotations

import gc
import json
import math
import os
import statistics
import subprocess
import time
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

import torch
import torch.nn.functional as F

RegressionStatus = Literal["PASS", "FAIL", "NOT NATIVE", "SYNTHETIC-ONLY", "NOT RUN"]
RegressionTier = Literal["fast", "full"]

REGRESSION_PROMPTS: tuple[str, ...] = (
    "Explain how GPU Tensor Cores accelerate matrix multiplication.",
    "The capital city of France is Paris, which is famous for",
    "In parallel computing, memory bandwidth bottlenecks occur when",
    "def fibonacci(n: int) -> int:\n    if n <= 1:\n        return n\n",
    "Historically, the invention of the printing press in Europe transformed",
)

PPL_PASSAGE = (
    " = Valkyria Chronicles III = \n"
    " Senjō no Valkyria 3 : Unrecorded Chronicles ( Japanese : 戦場のヴァルキュリア3 , lit . "
    "Valkyria of the Battlefield 3 ) , commonly referred to as Valkyria Chronicles III outside Japan , "
    "is a tactical role @-@ playing video game developed by Sega and Media.Vision for the PlayStation Portable . "
    "Released in January 2011 in Japan , it is the third game in the Valkyria series . "
    "Employing the same fusion of tactical and real @-@ time gameplay as its predecessors , the story runs "
    'parallel to the first game and follows the " Nameless " , a penal military unit serving the nation of '
    'Gallia during the Second Europan War who perform secret black operations and are pitted against the Imperial unit " Calamity Raven " . '
    "The game began development in 2010 , carrying over a large portion of the work done on Valkyria Chronicles II . "
    "While it retained the standard features of the series , it also rebuilt most of the core elements to make the gameplay smoother and more responsive . "
    " = Tower Building of the Little Rock Arsenal = \n"
    " The Tower Building of the Little Rock Arsenal , also known as the U.S. Arsenal Building , is a historic museum "
    "building located in MacArthur Park in downtown Little Rock , Arkansas . Built between 1840 and 1841 , it originally "
    "served as the munitions storage building for the Little Rock Arsenal and is the oldest surviving structure in the city . "
    "During the American Civil War , federal troops peacefully evacuated the arsenal in February 1861 ahead of Arkansas state "
    "militia forces prior to the state's secession from the Union . The building later became the birthplace of General Douglas MacArthur "
    "in January 1880 while his father Arthur MacArthur Jr. commanded the garrison stationed at the post . "
    " = Common starling = \n"
    " The common starling ( Sturnus vulgaris ) , also known as the European starling in North America , is a medium @-@ sized passerine "
    "bird in the starling family , Sturnidae . Measuring about 20 cm in length with glossy black plumage speckled with white at certain times "
    "of the year , the species is native to temperate Europe and western Asia and has been introduced successfully to Australia , New Zealand , "
    "North America , and southern Africa . "
)


@dataclass
class RegressionResult:
    category: str
    model: str
    checkpoint: str
    reference_impl: str
    layers_evaluated: int
    status: RegressionStatus
    weight_source: str
    native_execution: bool
    failing_check: str | None = None
    metrics: dict[str, Any] = field(default_factory=dict)
    notes: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "category": self.category,
            "model": self.model,
            "checkpoint": self.checkpoint,
            "reference_impl": self.reference_impl,
            "layers_evaluated": self.layers_evaluated,
            "status": self.status,
            "weight_source": self.weight_source,
            "native_execution": self.native_execution,
            "failing_check": self.failing_check,
            "metrics": self.metrics,
            "notes": self.notes,
        }


@dataclass
class RegressionReport:
    tier: str
    commit: str
    hardware_id: str
    contended: bool
    timestamp_utc: str
    results: list[RegressionResult]

    @property
    def all_non_native_or_pass(self) -> bool:
        return all(r.status in ("PASS", "NOT NATIVE") for r in self.results)

    def to_dict(self) -> dict[str, Any]:
        return {
            "tier": self.tier,
            "commit": self.commit,
            "hardware_id": self.hardware_id,
            "contended": self.contended,
            "timestamp_utc": self.timestamp_utc,
            "results": [r.to_dict() for r in self.results],
        }

    def format_markdown(self) -> str:
        lines = [
            f"# Regression Collection Report (`tier={self.tier}`)",
            "",
            f"- **Commit:** `{self.commit}` | **Hardware:** `{self.hardware_id}` | **Contended:** `{self.contended}`",
            "",
            "| Category | Model | Reference | Layers | Status | Key Parity / PPL Metrics | Notes |",
            "|---|---|---|---:|---|---|---|",
        ]
        for r in self.results:
            m = r.metrics
            parts: list[str] = []
            if "logit_cos_sim" in m:
                parts.append(f"cos={m['logit_cos_sim']:.6f}")
            if "emb_cos_sim" in m:
                parts.append(f"cos={m['emb_cos_sim']:.6f}")
            if "output_cos_sim" in m:
                parts.append(f"cos={m['output_cos_sim']:.6f}")
            if "max_abs_diff" in m:
                parts.append(f"diff={m['max_abs_diff']:.4f}")
            if "greedy_exact_matches" in m:
                parts.append(
                    f"greedy={m['greedy_exact_matches']}/{m['greedy_total_prompts']} "
                    f"({m.get('greedy_tok_match_pct', 0):.1f}% tok)"
                )
            if "ppl_cutile" in m:
                parts.append(
                    f"ppl={m['ppl_cutile']:.3f} vs {m.get('ppl_ref', 0):.3f} ({m.get('ppl_tokens', 0)} tok)"
                )
            if r.failing_check:
                parts.append(f"FAIL:{r.failing_check}")
            lines.append(
                f"| {r.category} | {r.model} | `{r.reference_impl}` | {r.layers_evaluated} | `{r.status}` | {', '.join(parts) or '—'} | {r.notes} |"
            )
        return "\n".join(lines)


def _git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"


def _hw_and_contention() -> tuple[str, bool]:
    if not torch.cuda.is_available():
        return "CPU", False
    name = torch.cuda.get_device_name(0)
    cc = torch.cuda.get_device_capability(0)
    hw = f"{name} (sm_{cc[0]}{cc[1]})"
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"], text=True
        ).strip()
        pids = [x.strip() for x in out.splitlines() if x.strip()]
        return hw, len(pids) > 1
    except Exception:
        return hw, False


def _cleanup_gpu() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _extract_hf_llava_weights(
    hf_llava: Any, device: str = "cuda", dtype: torch.dtype = torch.float16
) -> tuple[Any, dict[str, Any]]:
    from cutile_forge.models.model_graph import (
        BlockStage,
        DenseTransformerBlock,
        LMHeadStage,
        MultimodalFusionStage,
        UnifiedModelGraph,
        VisionPatchEmbedStage,
        ViTEncoderBlock,
    )

    dev = torch.device(device)
    cfg = hf_llava.config
    vcfg = cfg.vision_config
    tcfg = cfg.text_config
    v_layer_limit = int(vcfg.num_hidden_layers) + int(cfg.vision_feature_layer) + 1  # -2 -> 23

    vs = VisionPatchEmbedStage(
        hidden_size=int(vcfg.hidden_size),
        image_size=int(vcfg.image_size),
        patch_size=int(vcfg.patch_size),
        in_channels=int(getattr(vcfg, "num_channels", 3)),
        use_cls_token=True,
        pre_layer_norm=True,
        layer_norm_eps=float(vcfg.layer_norm_eps),
    )
    raw_act = str(getattr(vcfg, "hidden_act", "quick_gelu"))
    v_act: Literal["gelu", "silu", "quick_gelu"] = (
        raw_act if raw_act in ("gelu", "silu", "quick_gelu") else "gelu"  # type: ignore[assignment]
    )
    v_blocks = [
        ViTEncoderBlock(
            hidden_size=int(vcfg.hidden_size),
            mlp_dim=int(vcfg.intermediate_size),
            num_attention_heads=int(vcfg.num_attention_heads),
            head_dim=int(vcfg.hidden_size) // int(vcfg.num_attention_heads),
            layer_norm_eps=float(vcfg.layer_norm_eps),
            activation=v_act,
        )
        for _ in range(v_layer_limit)
    ]
    n_kv = int(getattr(tcfg, "num_key_value_heads", tcfg.num_attention_heads))
    h_dim = int(tcfg.hidden_size) // int(tcfg.num_attention_heads)
    rope_th = float(getattr(tcfg, "rope_theta", 10000.0))
    t_blocks: list[BlockStage] = [
        DenseTransformerBlock(
            hidden_size=int(tcfg.hidden_size),
            intermediate_size=int(tcfg.intermediate_size),
            num_attention_heads=int(tcfg.num_attention_heads),
            num_key_value_heads=n_kv,
            head_dim=h_dim,
            rope_theta=rope_th,
            rms_norm_eps=float(tcfg.rms_norm_eps),
        )
        for _ in range(int(tcfg.num_hidden_layers))
    ]
    graph = UnifiedModelGraph(
        name="llava-1.5-7b-hf",
        family="llava",
        modality="multimodal",
        input_stage=MultimodalFusionStage(
            hidden_size=int(tcfg.hidden_size),
            vocab_size=int(tcfg.vocab_size),
            vision_stage=vs,
            vision_blocks=v_blocks,
            select_feature=(
                "patch" if cfg.vision_feature_select_strategy == "default" else "cls_patch"
            ),
        ),
        blocks=t_blocks,
        output_stage=LMHeadStage(
            hidden_size=int(tcfg.hidden_size),
            vocab_size=int(tcfg.vocab_size),
            rms_norm_eps=float(tcfg.rms_norm_eps),
        ),
        metadata={"rotary_style": "half"},
    )

    vt = getattr(hf_llava.model.vision_tower, "vision_model", hf_llava.model.vision_tower)
    v_emb = vt.embeddings
    weights: dict[str, Any] = {
        "vision_embed": {
            "patch_w": v_emb.patch_embedding.weight.detach()
            .to(device=dev, dtype=dtype)
            .contiguous(),
            "cls_token": v_emb.class_embedding.detach()
            .view(1, 1, -1)
            .to(device=dev, dtype=dtype)
            .contiguous(),
            "pos_embed": v_emb.position_embedding.weight.detach()
            .unsqueeze(0)
            .to(device=dev, dtype=dtype)
            .contiguous(),
            "pre_ln_w": vt.pre_layrnorm.weight.detach().to(device=dev, dtype=dtype).contiguous(),
            "pre_ln_b": vt.pre_layrnorm.bias.detach().to(device=dev, dtype=dtype).contiguous(),
        },
        "vision_blocks": [],
        "projector": {
            "fc1_w": hf_llava.model.multi_modal_projector.linear_1.weight.detach()
            .t()
            .to(device=dev, dtype=dtype)
            .contiguous(),
            "fc1_b": hf_llava.model.multi_modal_projector.linear_1.bias.detach()
            .to(device=dev, dtype=dtype)
            .contiguous(),
            "fc2_w": hf_llava.model.multi_modal_projector.linear_2.weight.detach()
            .t()
            .to(device=dev, dtype=dtype)
            .contiguous(),
            "fc2_b": hf_llava.model.multi_modal_projector.linear_2.bias.detach()
            .to(device=dev, dtype=dtype)
            .contiguous(),
            "image_token_index": int(cfg.image_token_index),
        },
        "embed_tokens": hf_llava.model.language_model.embed_tokens.weight.detach()
        .to(device=dev, dtype=dtype)
        .contiguous(),
        "norm_w": hf_llava.model.language_model.norm.weight.detach()
        .to(device=dev, dtype=dtype)
        .contiguous(),
        "lm_head": hf_llava.lm_head.weight.detach().t().to(device=dev, dtype=dtype).contiguous(),
        "blocks": [],
    }
    for lyr in vt.encoder.layers[:v_layer_limit]:
        attn = lyr.self_attn
        wqkv = torch.cat(
            [
                attn.q_proj.weight.detach().t(),
                attn.k_proj.weight.detach().t(),
                attn.v_proj.weight.detach().t(),
            ],
            dim=1,
        )
        bqkv = torch.cat(
            [attn.q_proj.bias.detach(), attn.k_proj.bias.detach(), attn.v_proj.bias.detach()], dim=0
        )
        weights["vision_blocks"].append(
            {
                "ln1_w": lyr.layer_norm1.weight.detach().to(device=dev, dtype=dtype).contiguous(),
                "ln1_b": lyr.layer_norm1.bias.detach().to(device=dev, dtype=dtype).contiguous(),
                "ln2_w": lyr.layer_norm2.weight.detach().to(device=dev, dtype=dtype).contiguous(),
                "ln2_b": lyr.layer_norm2.bias.detach().to(device=dev, dtype=dtype).contiguous(),
                "wqkv": wqkv.to(device=dev, dtype=dtype).contiguous(),
                "bqkv": bqkv.to(device=dev, dtype=dtype).contiguous(),
                "wo": attn.out_proj.weight.detach().t().to(device=dev, dtype=dtype).contiguous(),
                "bo": attn.out_proj.bias.detach().to(device=dev, dtype=dtype).contiguous(),
                "fc1_w": lyr.mlp.fc1.weight.detach().t().to(device=dev, dtype=dtype).contiguous(),
                "fc1_b": lyr.mlp.fc1.bias.detach().to(device=dev, dtype=dtype).contiguous(),
                "fc2_w": lyr.mlp.fc2.weight.detach().t().to(device=dev, dtype=dtype).contiguous(),
                "fc2_b": lyr.mlp.fc2.bias.detach().to(device=dev, dtype=dtype).contiguous(),
            }
        )
    for lyr in hf_llava.model.language_model.layers:
        attn = lyr.self_attn
        mlp = lyr.mlp
        weights["blocks"].append(
            {
                "in_norm_w": lyr.input_layernorm.weight.detach()
                .to(device=dev, dtype=dtype)
                .contiguous(),
                "post_norm_w": lyr.post_attention_layernorm.weight.detach()
                .to(device=dev, dtype=dtype)
                .contiguous(),
                "wq": attn.q_proj.weight.detach().t().to(device=dev, dtype=dtype).contiguous(),
                "wk": attn.k_proj.weight.detach().t().to(device=dev, dtype=dtype).contiguous(),
                "wv": attn.v_proj.weight.detach().t().to(device=dev, dtype=dtype).contiguous(),
                "wo": attn.o_proj.weight.detach().t().to(device=dev, dtype=dtype).contiguous(),
                "w_gate": mlp.gate_proj.weight.detach()
                .t()
                .to(device=dev, dtype=dtype)
                .contiguous(),
                "w_up": mlp.up_proj.weight.detach().t().to(device=dev, dtype=dtype).contiguous(),
                "w_down": mlp.down_proj.weight.detach()
                .t()
                .to(device=dev, dtype=dtype)
                .contiguous(),
            }
        )
    return graph, weights


def _extract_diffusers_dit_weights(
    dit_model: Any, device: str = "cuda", dtype: torch.dtype = torch.float16
) -> tuple[Any, dict[str, Any]]:
    from cutile_forge.models.model_graph import (
        DiffusionOutputStage,
        DiffusionPatchEmbedStage,
        DiTBlock,
        UnifiedModelGraph,
    )

    dev = torch.device(device)
    cfg = dit_model.config
    n_heads = int(cfg.num_attention_heads)
    h_dim = int(cfg.attention_head_dim)
    d = n_heads * h_dim
    n_layers = int(cfg.num_layers)
    patch = int(cfg.patch_size)
    sample_sz = int(cfg.sample_size)
    in_ch = int(cfg.in_channels)
    out_ch = int(cfg.out_channels)

    graph = UnifiedModelGraph(
        name="DiT-XL-2-256",
        family="dit",
        modality="diffusion",
        input_stage=DiffusionPatchEmbedStage(
            hidden_size=d, image_size=sample_sz, patch_size=patch, in_channels=in_ch
        ),
        blocks=[
            DiTBlock(
                hidden_size=d,
                mlp_dim=d * 4,
                num_attention_heads=n_heads,
                head_dim=h_dim,
                layer_norm_eps=1e-6,
            )
            for _ in range(n_layers)
        ],
        output_stage=DiffusionOutputStage(
            hidden_size=d,
            image_size=sample_sz,
            patch_size=patch,
            out_channels=out_ch,
            layer_norm_eps=1e-6,
        ),
    )

    weights: dict[str, Any] = {
        "diffusion_embed": {
            "patch_w": dit_model.pos_embed.proj.weight.detach()
            .to(device=dev, dtype=dtype)
            .contiguous(),
            "patch_b": dit_model.pos_embed.proj.bias.detach()
            .to(device=dev, dtype=dtype)
            .contiguous(),
            "pos_embed": dit_model.pos_embed.pos_embed.detach()
            .to(device=dev, dtype=dtype)
            .contiguous(),
        },
        "blocks": [],
        "adaln_out_w": dit_model.proj_out_1.weight.detach()
        .t()
        .to(device=dev, dtype=dtype)
        .contiguous(),
        "adaln_out_b": dit_model.proj_out_1.bias.detach().to(device=dev, dtype=dtype).contiguous(),
        "proj_out_w": dit_model.proj_out_2.weight.detach()
        .t()
        .to(device=dev, dtype=dtype)
        .contiguous(),
        "proj_out_b": dit_model.proj_out_2.bias.detach().to(device=dev, dtype=dtype).contiguous(),
    }
    for blk in dit_model.transformer_blocks:
        attn = blk.attn1
        wqkv = torch.cat(
            [
                attn.to_q.weight.detach().t(),
                attn.to_k.weight.detach().t(),
                attn.to_v.weight.detach().t(),
            ],
            dim=1,
        )
        bqkv = torch.cat(
            [attn.to_q.bias.detach(), attn.to_k.bias.detach(), attn.to_v.bias.detach()], dim=0
        )
        weights["blocks"].append(
            {
                "t_fc1_w": blk.norm1.emb.timestep_embedder.linear_1.weight.detach()
                .t()
                .to(device=dev, dtype=dtype)
                .contiguous(),
                "t_fc1_b": blk.norm1.emb.timestep_embedder.linear_1.bias.detach()
                .to(device=dev, dtype=dtype)
                .contiguous(),
                "t_fc2_w": blk.norm1.emb.timestep_embedder.linear_2.weight.detach()
                .t()
                .to(device=dev, dtype=dtype)
                .contiguous(),
                "t_fc2_b": blk.norm1.emb.timestep_embedder.linear_2.bias.detach()
                .to(device=dev, dtype=dtype)
                .contiguous(),
                "class_embed": blk.norm1.emb.class_embedder.embedding_table.weight.detach()
                .to(device=dev, dtype=dtype)
                .contiguous(),
                "adaln_w": blk.norm1.linear.weight.detach()
                .t()
                .to(device=dev, dtype=dtype)
                .contiguous(),
                "adaln_b": blk.norm1.linear.bias.detach().to(device=dev, dtype=dtype).contiguous(),
                "wqkv": wqkv.to(device=dev, dtype=dtype).contiguous(),
                "bqkv": bqkv.to(device=dev, dtype=dtype).contiguous(),
                "wo": attn.to_out[0].weight.detach().t().to(device=dev, dtype=dtype).contiguous(),
                "bo": attn.to_out[0].bias.detach().to(device=dev, dtype=dtype).contiguous(),
                "fc1_w": blk.ff.net[0]
                .proj.weight.detach()
                .t()
                .to(device=dev, dtype=dtype)
                .contiguous(),
                "fc1_b": blk.ff.net[0].proj.bias.detach().to(device=dev, dtype=dtype).contiguous(),
                "fc2_w": blk.ff.net[2].weight.detach().t().to(device=dev, dtype=dtype).contiguous(),
                "fc2_b": blk.ff.net[2].bias.detach().to(device=dev, dtype=dtype).contiguous(),
            }
        )
    return graph, weights


def _eval_qwen25(num_prompts: int, greedy_tokens: int) -> RegressionResult:
    import transformers
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from cutile_forge.models import UnifiedModelExecutor, extract_hf_decoder_weights

    repo = "Qwen/Qwen2.5-0.5B"
    tok = AutoTokenizer.from_pretrained(repo, local_files_only=True)
    hf_m = AutoModelForCausalLM.from_pretrained(
        repo, dtype=torch.float16, device_map="cuda", local_files_only=True
    )
    hf_m.eval()
    graph, weights = extract_hf_decoder_weights(
        hf_m, num_layers=None, device="cuda", dtype=torch.float16
    )
    ex = UnifiedModelExecutor(graph)

    # 1. Final-logit error + 512-token PPL vs HF Transformers
    ppl_ids = tok.encode(PPL_PASSAGE, add_special_tokens=False)[:512]
    x_ppl = torch.tensor([ppl_ids], device="cuda", dtype=torch.long)
    with torch.no_grad():
        lat: list[float] = []
        for rep in range(3):
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            nat_logits = ex.forward(weights, input_ids=x_ppl).float()
            torch.cuda.synchronize()
            if rep > 0:
                lat.append((time.perf_counter() - t0) * 1000.0)
        ref_logits = hf_m(x_ppl).logits.float()

    cos = float(F.cosine_similarity(nat_logits[0, -1], ref_logits[0, -1], dim=0).item())
    max_diff = float((nat_logits[0, -1] - ref_logits[0, -1]).abs().max().item())
    ppl_cu = math.exp(float(F.cross_entropy(nat_logits[0, :-1], x_ppl[0, 1:]).item()))
    ppl_rf = math.exp(float(F.cross_entropy(ref_logits[0, :-1], x_ppl[0, 1:]).item()))

    # 2. Greedy parity across num_prompts x greedy_tokens (teacher-forced HF parity)
    exact_prompts = 0
    tok_matches = 0
    tok_total = 0
    for p_text in REGRESSION_PROMPTS[:num_prompts]:
        p_ids = tok.encode(p_text, add_special_tokens=False)
        cur = list(p_ids)
        with torch.no_grad():
            for _ in range(greedy_tokens):
                out_l = ex.forward(
                    weights, input_ids=torch.tensor([cur], device="cuda", dtype=torch.long)
                )
                cur.append(int(out_l[0, -1].argmax().item()))
            ref_full = hf_m(torch.tensor([cur], device="cuda", dtype=torch.long)).logits[0]
            ref_preds = ref_full[len(p_ids) - 1 : -1].argmax(dim=-1).tolist()
        gen = cur[len(p_ids) :]
        m_cnt = sum(1 for a, b in zip(gen, ref_preds, strict=True) if a == b)
        tok_matches += m_cnt
        tok_total += len(gen)
        if gen == ref_preds:
            exact_prompts += 1

    del hf_m, weights, ex
    _cleanup_gpu()

    cos_ok = cos >= 0.9999
    ppl_ok = abs(ppl_cu - ppl_rf) / max(1e-9, ppl_rf) <= 0.01
    greedy_ok = exact_prompts == num_prompts
    failing = (
        None
        if (cos_ok and ppl_ok and greedy_ok)
        else f"cos={cos_ok},ppl={ppl_ok},greedy={greedy_ok}"
    )
    return RegressionResult(
        category="Small decoder",
        model="Qwen2.5-0.5B",
        checkpoint=repo,
        reference_impl=f"transformers=={transformers.__version__} (AutoModelForCausalLM)",
        layers_evaluated=24,
        status="PASS" if failing is None else "FAIL",
        weight_source="real-checkpoint",
        native_execution=True,
        failing_check=failing,
        metrics={
            "logit_cos_sim": round(cos, 6),
            "max_abs_diff": round(max_diff, 4),
            "ppl_cutile": round(ppl_cu, 4),
            "ppl_ref": round(ppl_rf, 4),
            "ppl_tokens": len(ppl_ids),
            "greedy_exact_matches": exact_prompts,
            "greedy_total_prompts": num_prompts,
            "greedy_tokens_per_prompt": greedy_tokens,
            "greedy_tok_match_pct": round(100.0 * tok_matches / max(1, tok_total), 2),
            "prefill_512_med_ms": round(statistics.median(lat), 2) if lat else 0.0,
        },
        notes="UnifiedModelExecutor executes native CuTile FlashAttention + RoPE + GEMM + RMSNorm + SiLU",
    )


def _eval_siglip() -> RegressionResult:
    import transformers
    from transformers import SiglipVisionModel

    from cutile_forge.models import UnifiedModelExecutor, extract_hf_siglip_vision_weights

    repo = "google/siglip-base-patch16-224"
    hf_v = SiglipVisionModel.from_pretrained(
        repo, dtype=torch.float16, device_map="cuda", local_files_only=True
    )
    hf_v.eval()
    graph, weights = extract_hf_siglip_vision_weights(hf_v, device="cuda", dtype=torch.float16)
    ex = UnifiedModelExecutor(graph)

    torch.manual_seed(42)
    pix = torch.randn((1, 3, 224, 224), device="cuda", dtype=torch.float16) * 0.5
    with torch.no_grad():
        cu_out = ex.forward(weights, pixel_values=pix).float()
        ref_out = hf_v(pixel_values=pix).last_hidden_state.float()
    cos = float(F.cosine_similarity(cu_out.flatten(), ref_out.flatten(), dim=0).item())
    diff = float((cu_out - ref_out).abs().max().item())

    del hf_v, weights, ex
    _cleanup_gpu()
    ok = cos >= 0.9999 and diff < 5e-2
    return RegressionResult(
        category="Vision",
        model="SigLIP ViT-B/16",
        checkpoint=repo,
        reference_impl=f"transformers=={transformers.__version__} (SiglipVisionModel)",
        layers_evaluated=12,
        status="PASS" if ok else "FAIL",
        weight_source="real-checkpoint",
        native_execution=True,
        failing_check=None if ok else f"emb_cos={cos:.6f},max_diff={diff:.4f}",
        metrics={"emb_cos_sim": round(cos, 6), "max_abs_diff": round(diff, 5)},
        notes="UnifiedModelExecutor executes native CuTile patch GEMM, LayerNorm, bidirectional FlashAttention, GELU",
    )


def _eval_whisper() -> RegressionResult:
    import transformers
    from transformers import WhisperModel

    from cutile_forge.models import UnifiedModelExecutor, extract_hf_whisper_encoder_weights

    repo = "openai/whisper-tiny"
    hf_w = WhisperModel.from_pretrained(
        repo, dtype=torch.float16, device_map="cuda", local_files_only=True
    )
    hf_w.eval()
    graph, weights = extract_hf_whisper_encoder_weights(hf_w, device="cuda", dtype=torch.float16)
    ex = UnifiedModelExecutor(graph)

    torch.manual_seed(42)
    mel = torch.randn((1, 80, 3000), device="cuda", dtype=torch.float16) * 0.25
    with torch.no_grad():
        cu_out = ex.forward(weights, mel_features=mel).float()
        ref_out = hf_w.encoder(mel).last_hidden_state.float()
    cos = float(F.cosine_similarity(cu_out.flatten(), ref_out.flatten(), dim=0).item())
    diff = float((cu_out - ref_out).abs().max().item())

    del hf_w, weights, ex
    _cleanup_gpu()
    ok = cos >= 0.9999 and diff < 1e-1
    return RegressionResult(
        category="Audio",
        model="Whisper-Tiny",
        checkpoint=repo,
        reference_impl=f"transformers=={transformers.__version__} (WhisperModel.encoder)",
        layers_evaluated=4,
        status="PASS" if ok else "FAIL",
        weight_source="real-checkpoint",
        native_execution=True,
        failing_check=None if ok else f"emb_cos={cos:.6f},max_diff={diff:.4f}",
        metrics={"emb_cos_sim": round(cos, 6), "max_abs_diff": round(diff, 5)},
        notes="UnifiedModelExecutor executes native CuTile Conv1d(via cutile_conv2d), LayerNorm, bidirectional FlashAttention, GELU",
    )


def _eval_mistral_native(num_prompts: int, greedy_tokens: int) -> RegressionResult:
    from transformers import AutoTokenizer

    from cutile_forge.engine.config import GPUConfig
    from cutile_forge.engine.model import Mistral7B
    from cutile_forge.engine.weight_loader import load_gptq_weights
    from cutile_forge.kernels.embedding_lmhead import cutile_embedding_lookup
    from cutile_forge.kernels.gemm_dequant import cutile_gemm_fp16
    from cutile_forge.kernels.rmsnorm import cutile_rmsnorm
    from cutile_forge.runtime import NativeModelRuntime

    ckpt_dir = Path("/home/arrhat/models/mistral_7b_4bit")
    tok = AutoTokenizer.from_pretrained(str(ckpt_dir), local_files_only=True)
    weights = load_gptq_weights(ckpt_dir, device="cuda", pin_ddr5=False)
    model = Mistral7B(
        weights=weights, gpu_config=GPUConfig(vram_gb=32.0, stream_weights=False), device="cuda"
    )
    rt = NativeModelRuntime(model, max_seq_len=2048, kv_dtype="fp16", use_cuda_graph=True)

    ppl_ids = [1] + tok.encode(PPL_PASSAGE, add_special_tokens=False)[:511]
    x_ppl = torch.tensor([ppl_ids], device="cuda", dtype=torch.long)
    with torch.no_grad():
        rt.reset()
        h = cutile_embedding_lookup(x_ppl.view(-1), model.weights.embed_tokens).view(
            len(ppl_ids), model.config.hidden_size
        )
        for idx in range(model.config.num_hidden_layers):
            h = rt._forward_layer_native(idx, h, pos_offset=0, seq_len=len(ppl_ids), batch_size=1)
        hn = cutile_rmsnorm(h, model.weights.norm, eps=model.config.rms_norm_eps)
        nat_logits = (
            cutile_gemm_fp16(hn, model.weights.lm_head.t().contiguous())
            .view(1, len(ppl_ids), -1)
            .float()
        )
        ref_logits = model.forward(x_ppl, kv_cache=None, pos_offset=0, backend="pytorch").float()

    cos = float(F.cosine_similarity(nat_logits[0, -1], ref_logits[0, -1], dim=0).item())
    max_diff = float((nat_logits[0, -1] - ref_logits[0, -1]).abs().max().item())
    ppl_cu = math.exp(float(F.cross_entropy(nat_logits[0, :-1], x_ppl[0, 1:]).item()))
    ppl_rf = math.exp(float(F.cross_entropy(ref_logits[0, :-1], x_ppl[0, 1:]).item()))

    exact_prompts = 0
    tok_matches = 0
    tok_total = 0
    dec_rates: list[float] = []
    for p_text in REGRESSION_PROMPTS[:num_prompts]:
        p_ids = [1] + tok.encode(p_text, add_special_tokens=False)
        gen, stats = rt.generate(p_ids, max_new_tokens=greedy_tokens, sync_interval=32)
        dec_rates.append(float(stats["decode_tok_s"]))
        with torch.no_grad():
            full_t = torch.tensor([p_ids + gen], device="cuda", dtype=torch.long)
            ref_full = model.forward(full_t, kv_cache=None, pos_offset=0, backend="pytorch")[0]
            ref_preds = ref_full[len(p_ids) - 1 : -1].argmax(dim=-1).tolist()
        m_cnt = sum(1 for a, b in zip(gen, ref_preds, strict=True) if a == b)
        tok_matches += m_cnt
        tok_total += len(gen)
        if gen == ref_preds:
            exact_prompts += 1

    del rt, model, weights
    _cleanup_gpu()
    ok = cos >= 0.9999 and abs(ppl_cu - ppl_rf) / ppl_rf <= 0.01 and exact_prompts == num_prompts
    return RegressionResult(
        category="Dense decoder",
        model="Mistral-7B",
        checkpoint=str(ckpt_dir),
        reference_impl="Mistral7B(backend='pytorch', GPTQ-32L)",
        layers_evaluated=32,
        status="PASS" if ok else "FAIL",
        weight_source="real-checkpoint",
        native_execution=True,
        failing_check=None if ok else f"cos={cos:.6f},greedy={exact_prompts}/{num_prompts}",
        metrics={
            "logit_cos_sim": round(cos, 6),
            "max_abs_diff": round(max_diff, 4),
            "ppl_cutile": round(ppl_cu, 4),
            "ppl_ref": round(ppl_rf, 4),
            "ppl_tokens": len(ppl_ids),
            "greedy_exact_matches": exact_prompts,
            "greedy_total_prompts": num_prompts,
            "greedy_tokens_per_prompt": greedy_tokens,
            "greedy_tok_match_pct": round(100.0 * tok_matches / max(1, tok_total), 2),
            "decode_med_tok_s": round(statistics.median(dec_rates), 2),
            "decode_min_tok_s": round(min(dec_rates), 2),
            "decode_max_tok_s": round(max(dec_rates), 2),
        },
        notes="NativeModelRuntime zero-ATen CUDA Graph decode + SWA=4096",
    )


def _eval_gptq_streaming_model(
    *,
    category: str,
    model_label: str,
    ckpt_dir: Path,
    num_layers: int,
    max_gpu_layers: int,
    num_prompts: int,
    greedy_tokens: int,
    notes: str,
) -> RegressionResult:
    from transformers import AutoTokenizer

    from cutile_forge.engine.config import GPUConfig
    from cutile_forge.engine.kv_cache import PagedKVCache
    from cutile_forge.engine.model import Mistral7B
    from cutile_forge.engine.weight_loader import load_gptq_weights

    tok = AutoTokenizer.from_pretrained(str(ckpt_dir), local_files_only=True)
    weights = load_gptq_weights(
        ckpt_dir, device="cuda", pin_ddr5=True, max_gpu_layers=max_gpu_layers
    )
    model = Mistral7B(
        weights=weights,
        gpu_config=GPUConfig(vram_gb=32.0, stream_weights=(max_gpu_layers < num_layers)),
        device="cuda",
    )

    ppl_ids = [1] + tok.encode(PPL_PASSAGE, add_special_tokens=False)[:511]
    x_ppl = torch.tensor([ppl_ids], device="cuda", dtype=torch.long)
    with torch.no_grad():
        nat_logits = model.forward(x_ppl, kv_cache=None, pos_offset=0, backend="cutile").float()
        ref_logits = model.forward(x_ppl, kv_cache=None, pos_offset=0, backend="pytorch").float()

    cos = float(F.cosine_similarity(nat_logits[0, -1], ref_logits[0, -1], dim=0).item())
    max_diff = float((nat_logits[0, -1] - ref_logits[0, -1]).abs().max().item())
    ppl_cu = math.exp(float(F.cross_entropy(nat_logits[0, :-1], x_ppl[0, 1:]).item()))
    ppl_rf = math.exp(float(F.cross_entropy(ref_logits[0, :-1], x_ppl[0, 1:]).item()))

    kv = PagedKVCache(
        num_layers=num_layers,
        num_kv_heads=model.config.num_key_value_heads,
        head_dim=model.config.head_dim,
        page_size=16,
        gpu_budget_bytes=2 * 1024 * 1024 * 1024,
        device="cuda",
    )

    exact_prompts = 0
    tok_matches = 0
    tok_total = 0
    dec_rates: list[float] = []
    for p_text in REGRESSION_PROMPTS[:num_prompts]:
        p_ids = [1] + tok.encode(p_text, add_special_tokens=False)
        kv.reset()
        with torch.no_grad():
            p_t = torch.tensor([p_ids], device="cuda", dtype=torch.long)
            out0 = model.forward(
                p_t, kv_cache=kv, pos_offset=0, backend="cutile", last_token_only=True
            )
            cur = int(out0[0, -1].argmax().item())
            gen = [cur]
            pos = len(p_ids)
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            for _ in range(greedy_tokens - 1):
                step_t = torch.tensor([[cur]], device="cuda", dtype=torch.long)
                step_logits = model.forward(
                    step_t, kv_cache=kv, pos_offset=pos, backend="cutile", last_token_only=True
                )
                cur = int(step_logits[0, -1].argmax().item())
                gen.append(cur)
                pos += 1
            torch.cuda.synchronize()
            dec_s = time.perf_counter() - t0
            dec_rates.append((greedy_tokens - 1) / max(1e-6, dec_s))

            full_t = torch.tensor([p_ids + gen], device="cuda", dtype=torch.long)
            ref_full = model.forward(full_t, kv_cache=None, pos_offset=0, backend="pytorch")[0]
            ref_preds = ref_full[len(p_ids) - 1 : -1].argmax(dim=-1).tolist()
        m_cnt = sum(1 for a, b in zip(gen, ref_preds, strict=True) if a == b)
        tok_matches += m_cnt
        tok_total += len(gen)
        if gen == ref_preds:
            exact_prompts += 1

    del kv, model, weights
    _cleanup_gpu()
    ok = cos >= 0.9999 and abs(ppl_cu - ppl_rf) / ppl_rf <= 0.01 and exact_prompts == num_prompts
    status: RegressionStatus = "PASS" if ok else "FAIL"
    return RegressionResult(
        category=category,
        model=model_label,
        checkpoint=str(ckpt_dir),
        reference_impl=f"Mistral7B(backend='pytorch', {num_layers}L)",
        layers_evaluated=num_layers,
        status=status,
        weight_source="real-checkpoint",
        native_execution=True,
        failing_check=None if ok else f"cos={cos:.6f},greedy={exact_prompts}/{num_prompts}",
        metrics={
            "logit_cos_sim": round(cos, 6),
            "max_abs_diff": round(max_diff, 4),
            "ppl_cutile": round(ppl_cu, 4),
            "ppl_ref": round(ppl_rf, 4),
            "ppl_tokens": len(ppl_ids),
            "greedy_exact_matches": exact_prompts,
            "greedy_total_prompts": num_prompts,
            "greedy_tokens_per_prompt": greedy_tokens,
            "greedy_tok_match_pct": round(100.0 * tok_matches / max(1, tok_total), 2),
            "decode_med_tok_s": round(statistics.median(dec_rates), 2),
            "decode_min_tok_s": round(min(dec_rates), 2),
            "decode_max_tok_s": round(max(dec_rates), 2),
        },
        notes=notes,
    )


def _eval_llava() -> RegressionResult:
    import transformers
    from PIL import Image
    from transformers import AutoProcessor, LlavaForConditionalGeneration

    from cutile_forge.models import UnifiedModelExecutor

    repo = "llava-hf/llava-1.5-7b-hf"
    proc = AutoProcessor.from_pretrained(repo, local_files_only=True)
    hf_ll: Any = LlavaForConditionalGeneration.from_pretrained(
        repo, dtype=torch.float16, device_map="cuda", local_files_only=True
    )
    hf_ll.eval()
    graph, weights = _extract_hf_llava_weights(hf_ll, device="cuda", dtype=torch.float16)
    ex = UnifiedModelExecutor(graph)

    # Deterministic synthetic RGB image + visual question
    torch.manual_seed(42)
    img_arr = (torch.rand((336, 336, 3)) * 255).to(torch.uint8).numpy()
    img = Image.fromarray(img_arr)
    prompt = "USER: <image>\nDescribe the image colors briefly. ASSISTANT:"
    inputs = proc(text=prompt, images=img, return_tensors="pt").to("cuda")
    input_ids = inputs["input_ids"]
    pixel_values = inputs["pixel_values"].to(torch.float16)

    with torch.no_grad():
        ref_out = hf_ll(input_ids=input_ids, pixel_values=pixel_values)
        ref_last = ref_out.logits[0, -1].float()
        cu_out = ex.forward(weights, input_ids=input_ids, pixel_values=pixel_values)
        cu_last = cu_out[0, -1].float()

        # Generate 16 image+text greedy tokens
        cu_toks: list[int] = [int(cu_last.argmax().item())]
        cur_ids = torch.cat([input_ids, torch.tensor([[cu_toks[0]]], device="cuda")], dim=1)
        for _ in range(15):
            o = ex.forward(weights, input_ids=cur_ids, pixel_values=pixel_values)
            nt = int(o[0, -1].argmax().item())
            cu_toks.append(nt)
            cur_ids = torch.cat([cur_ids, torch.tensor([[nt]], device="cuda")], dim=1)

        ref_gen = hf_ll.generate(
            input_ids=input_ids, pixel_values=pixel_values, max_new_tokens=16, do_sample=False
        )
        ref_toks = ref_gen[0, input_ids.shape[1] :].tolist()

    cos = float(F.cosine_similarity(cu_last, ref_last, dim=0).item())
    diff = float((cu_last - ref_last).abs().max().item())
    m_cnt = sum(1 for a, b in zip(cu_toks, ref_toks, strict=True) if a == b)

    del hf_ll, weights, ex
    _cleanup_gpu()
    ok = cos >= 0.9999 and cu_toks == ref_toks
    return RegressionResult(
        category="Multimodal",
        model="LLaVA-1.5",
        checkpoint=repo,
        reference_impl=f"transformers=={transformers.__version__} (LlavaForConditionalGeneration)",
        layers_evaluated=55,
        status="PASS" if ok else "FAIL",
        weight_source="real-checkpoint",
        native_execution=True,
        failing_check=None if ok else f"cos={cos:.6f},greedy={m_cnt}/16",
        metrics={
            "logit_cos_sim": round(cos, 6),
            "max_abs_diff": round(diff, 4),
            "greedy_exact_matches": 1 if cu_toks == ref_toks else 0,
            "greedy_total_prompts": 1,
            "greedy_tokens_per_prompt": 16,
            "greedy_tok_match_pct": round(100.0 * m_cnt / 16.0, 2),
        },
        notes="UnifiedModelExecutor executes native CuTile CLIP ViT + QuickGELU + 2L GELU proj + 32L Llama attention/MLP",
    )


def _eval_dit_xl2() -> RegressionResult:
    import diffusers
    from diffusers import DiTTransformer2DModel
    from huggingface_hub import snapshot_download

    from cutile_forge.models import UnifiedModelExecutor

    repo = "facebook/DiT-XL-2-256"
    snap = Path(snapshot_download(repo, local_files_only=True))
    dit = DiTTransformer2DModel.from_pretrained(
        str(snap), subfolder="transformer", torch_dtype=torch.float16
    ).cuda()
    dit.eval()
    graph, weights = _extract_diffusers_dit_weights(dit, device="cuda", dtype=torch.float16)
    ex = UnifiedModelExecutor(graph)

    torch.manual_seed(42)
    latents = torch.randn((1, 4, 32, 32), device="cuda", dtype=torch.float16) * 0.5
    ts = torch.tensor([500], device="cuda", dtype=torch.long)
    cls_lbl = torch.tensor([207], device="cuda", dtype=torch.long)

    with torch.no_grad():
        ref_out = dit(latents, timestep=ts, class_labels=cls_lbl).sample.float()
        cu_out = ex.forward(
            weights, pixel_values=latents, timesteps=ts, class_labels=cls_lbl
        ).float()

    cos = float(F.cosine_similarity(cu_out.flatten(), ref_out.flatten(), dim=0).item())
    diff = float((cu_out - ref_out).abs().max().item())

    del dit, weights, ex
    _cleanup_gpu()
    ok = cos >= 0.9999 and diff < 5e-2
    return RegressionResult(
        category="Diffusion",
        model="DiT-XL/2",
        checkpoint=repo,
        reference_impl=f"diffusers=={diffusers.__version__} (DiTTransformer2DModel)",
        layers_evaluated=28,
        status="PASS" if ok else "FAIL",
        weight_source="real-checkpoint",
        native_execution=True,
        failing_check=None if ok else f"cos={cos:.6f},max_diff={diff:.4f}",
        metrics={"output_cos_sim": round(cos, 6), "max_abs_diff": round(diff, 5)},
        notes="User-approved DiT-XL/2-256; UnifiedModelExecutor executes native CuTile LayerNorm, patch GEMM, d=72 FlashAttention",
    )


def run_regression_suite(
    tier: RegressionTier = "fast",
    output_dir: Path | None = None,
    num_prompts: int = 5,
    greedy_tokens: int = 256,
) -> RegressionReport:
    """Execute the Phase 15 regression suite (`fast` or `full`) against real checkpoints."""
    os.environ.setdefault("HF_HOME", "/var/tmp/arrhat_models/hf_cache")
    commit = _git_commit()
    hw, contended = _hw_and_contention()
    ts_utc = datetime.now(UTC).isoformat()

    results: list[RegressionResult] = []
    if tier == "fast":
        results.append(
            _eval_qwen25(num_prompts=min(1, num_prompts), greedy_tokens=min(64, greedy_tokens))
        )
        results.append(_eval_siglip())
        results.append(_eval_whisper())
    else:
        results.append(_eval_mistral_native(num_prompts=num_prompts, greedy_tokens=greedy_tokens))
        results.append(_eval_qwen25(num_prompts=num_prompts, greedy_tokens=greedy_tokens))
        results.append(
            _eval_gptq_streaming_model(
                category="MoE",
                model_label="Mixtral-8x7B",
                ckpt_dir=Path("/var/tmp/arrhat_models/mixtral_8x7b_4bit"),
                num_layers=32,
                max_gpu_layers=32,
                num_prompts=num_prompts,
                greedy_tokens=greedy_tokens,
                notes="32L HBM GPTQ + native CuTile cutile_moe_router_topk",
            )
        )
        results.append(_eval_siglip())
        results.append(_eval_whisper())
        results.append(_eval_llava())
        results.append(_eval_dit_xl2())
        results.append(
            _eval_gptq_streaming_model(
                category="Large decoder",
                model_label="Llama-2-70B",
                ckpt_dir=Path("/var/tmp/arrhat_models/llama_70b_4bit"),
                num_layers=80,
                max_gpu_layers=55,
                num_prompts=num_prompts,
                greedy_tokens=greedy_tokens,
                notes="80L GPTQ Stage -1 PCIe streaming (55 HBM + 25 DDR5)",
            )
        )

    report = RegressionReport(
        tier=tier,
        commit=commit,
        hardware_id=hw,
        contended=contended,
        timestamp_utc=ts_utc,
        results=results,
    )
    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / f"regress_{tier}.json").write_text(
            json.dumps(report.to_dict(), indent=2), encoding="utf-8"
        )
        (output_dir / f"regress_{tier}.md").write_text(
            report.format_markdown() + "\n", encoding="utf-8"
        )
    return report
