"""Phase 15.3: End-to-End Real-Checkpoint Multi-Format Quantization Validation.

Loads ALL layers of real on-disk checkpoints (no synthetic repacking) and runs >= 512
non-repeated WikiText-2 / technical tokens comparing each format's native CuTile
execution against:
  1. The same quantized weights in a pure-PyTorch reference dequantization runtime, and
  2. The FP16 reference checkpoint (`Qwen/Qwen2.5-0.5B-Instruct` for Qwen2.5 formats).
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import statistics
import time
from pathlib import Path
from typing import Any

import gguf
import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

from benchmarks.production.manifest import build_checkpoint_manifest, prompt_checksum
from benchmarks.production.runner import TimedDeviceOp, ValidationRun, host_rss_mb, peak_vram_mb
from benchmarks.production.schema import QuantizationConfig, WorkloadConfig, validate_artifact_dir
from cutile_forge.engine.config import ModelConfig
from cutile_forge.engine.weight_loader import _ShardReader, load_gptq_weights
from cutile_forge.kernels.rope import precompute_rope_freqs
from cutile_forge.kernels.universal_quant import (
    MixedPrecisionFormatTable,
    QuantFormat,
    UniversalQuantLayer,
    UniversalQuantLinear,
    UniversalQuantModel,
    parse_gguf_q4_0_native,
    parse_gguf_q4_k_native,
    parse_gguf_q6_k_native,
    parse_gguf_q8_0_native,
)
from cutile_forge.utils.quantize import unpack_gptq_layer

# >512 non-repeated real prose tokens combining WikiText-2 articles & architecture text
WIKITEXT2_EXTENDED = (
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
    "The game's battle system , the BLiTZ system , is carried over directly from previous titles during missions . "
    "Players guide squad members across three @-@ dimensional maps using an overhead tactical perspective before switching "
    "to an active third @-@ person view when controlling an individual soldier's movement and weapon targeting . "
    " = Tower Building of the Little Rock Arsenal = \n"
    " The Tower Building of the Little Rock Arsenal , also known as the U.S. Arsenal Building , is a historic museum "
    "building located in MacArthur Park in downtown Little Rock , Arkansas . Built between 1840 and 1841 , it originally "
    "served as the munitions storage building for the Little Rock Arsenal and is the oldest surviving structure in the city . "
    "During the American Civil War , federal troops peacefully evacuated the arsenal in February 1861 ahead of Arkansas state "
    "militia forces prior to the state's secession from the Union . The building later became the birthplace of General Douglas MacArthur "
    "in January 1880 while his father Arthur MacArthur Jr. commanded the garrison stationed at the post . "
    "Following decommissioning as a military installation , the surrounding grounds were transferred to the city of Little Rock in "
    "exchange for land used to establish Fort Logan H. Roots , and the structure has housed the MacArthur Museum of Arkansas Military History since 2001 . "
    " = Common starling = \n"
    " The common starling ( Sturnus vulgaris ) , also known as the European starling in North America , is a medium @-@ sized passerine "
    "bird in the starling family , Sturnidae . Measuring about 20 cm in length with glossy black plumage speckled with white at certain times "
    "of the year , the species is native to temperate Europe and western Asia and has been introduced successfully to Australia , New Zealand , "
    "North America , and southern Africa . Starlings are omnivorous and forage largely on insects and other invertebrates alongside fruit and seeds . "
    "Because large winter roosting flocks can contain hundreds of thousands of individuals and perform coordinated aerial murmurations at dusk , "
    "their social vocal mimicry has been studied extensively in ethology and bioacoustics across urban and agricultural habitats worldwide . "
    " = Robert Boulter = \n"
    " Robert Boulter is an English film , television and theatre actor . He had a guest @-@ starring role on the television series The Bill in 2000 . "
    "This was followed by a starring role in the play Herons written by Simon Stephens , which was performed in 2001 at the Royal Court Theatre . "
    "He had a guest role in the television series Doctors in 2004 , followed by appearances in Casualty , Holby City , and the feature film Donkey Punch in 2008 . "
    " = History of cryptography = \n"
    " Before the modern era , cryptography focused exclusively on message confidentiality using transposition ciphers and polyalphabetic substitution schemes "
    "until mechanical rotor machines and public @-@ key number theory transformed digital security in the twentieth century . "
    " = Architecture of early steam locomotives = \n"
    " Nineteenth @-@ century railway engineers refined fire @-@ tube boilers , blastpipes , and Walschaerts valve gear to convert coal combustion "
    "efficiently into tractive effort across steep mountain grades and transcontinental freight corridors . "
)


def _eval_uqm_metrics(
    uqm: UniversalQuantModel,
    ids_512: torch.Tensor,
    warmup: int = 1,
    runs: int = 5,
) -> dict[str, Any]:
    with torch.no_grad():
        logits_ref = uqm.forward(ids_512, mode="reference")[0].float()
        logits_nat = uqm.forward(ids_512, mode="native")[0].float()
        shift_ref = logits_ref[:-1]
        shift_nat = logits_nat[:-1]
        labels = ids_512[0, 1:].long()
        ppl_ref = math.exp(float(F.cross_entropy(shift_ref, labels).item()))
        ppl_nat = math.exp(float(F.cross_entropy(shift_nat, labels).item()))
        cos = float(F.cosine_similarity(logits_nat[-1], logits_ref[-1], dim=0).item())
        max_abs = float((logits_nat[-1] - logits_ref[-1]).abs().max().item())

        for _ in range(warmup):
            _ = uqm.forward(ids_512[:, :16], mode="native")
        times: list[float] = []
        for _ in range(runs):
            with TimedDeviceOp.measure(sync=True) as op:
                _ = uqm.forward(ids_512, mode="native")
            times.append(op.elapsed_ms)

    return {
        "ppl_native": ppl_nat,
        "ppl_format_ref": ppl_ref,
        "ppl_vs_format_ref_diff_pct": abs(ppl_nat - ppl_ref) / max(1e-9, ppl_ref) * 100.0,
        "logit_cos_sim": cos,
        "logit_max_abs_err": max_abs,
        "times_ms": times,
        "med_ms": statistics.median(times),
        "min_ms": min(times),
        "max_ms": max(times),
    }


def _load_qwen_safetensors_uqm(ckpt_dir: Path, fmt_kind: str) -> UniversalQuantModel:
    cfg_raw = json.loads((ckpt_dir / "config.json").read_text(encoding="utf-8"))
    qcfg = cfg_raw.get("quantization_config", {})
    cfg = ModelConfig.from_json(ckpt_dir / "config.json")
    shards = sorted(ckpt_dir.glob("*.safetensors"))
    reader = _ShardReader(shards)

    embed = reader["model.embed_tokens.weight"].to(device="cuda", dtype=torch.float16).contiguous()
    norm = reader["model.norm.weight"].to(device="cuda", dtype=torch.float16).contiguous()
    if "lm_head.weight" in reader:
        lm_head = reader["lm_head.weight"].to(device="cuda", dtype=torch.float16).contiguous()
    else:
        lm_head = embed

    cos, sin = precompute_rope_freqs(
        min(cfg.max_position_embeddings, 32768),
        cfg.head_dim,
        theta=cfg.rope_theta,
        device="cuda",
    )

    u_layers: list[UniversalQuantLayer] = []
    qfmt_enum = {
        "gptq": QuantFormat.GPTQ,
        "awq": QuantFormat.AWQ,
        "fp8_channel": QuantFormat.FP8_CHANNEL,
    }[fmt_kind]

    for idx in range(cfg.num_hidden_layers):
        p = f"model.layers.{idx}"
        in_ln = (
            reader[f"{p}.input_layernorm.weight"]
            .to(device="cuda", dtype=torch.float16)
            .contiguous()
        )
        post_ln = (
            reader[f"{p}.post_attention_layernorm.weight"]
            .to(device="cuda", dtype=torch.float16)
            .contiguous()
        )
        qb = (
            reader[f"{p}.self_attn.q_proj.bias"].to(device="cuda", dtype=torch.float16).contiguous()
            if f"{p}.self_attn.q_proj.bias" in reader
            else None
        )
        kb = (
            reader[f"{p}.self_attn.k_proj.bias"].to(device="cuda", dtype=torch.float16).contiguous()
            if f"{p}.self_attn.k_proj.bias" in reader
            else None
        )
        vb = (
            reader[f"{p}.self_attn.v_proj.bias"].to(device="cuda", dtype=torch.float16).contiguous()
            if f"{p}.self_attn.v_proj.bias" in reader
            else None
        )

        def _make_proj(name: str, p_prefix: str = p) -> UniversalQuantLinear:
            key_base = (
                f"{p_prefix}.self_attn.{name}"
                if name in ("q_proj", "k_proj", "v_proj", "o_proj")
                else f"{p_prefix}.mlp.{name}"
            )
            if fmt_kind == "gptq":
                qw_raw = reader[f"{key_base}.qweight"]
                qz_raw = reader[f"{key_base}.qzeros"]
                sc_raw = reader[f"{key_base}.scales"]
                g_idx = reader.get(f"{key_base}.g_idx")
                gs = int(qcfg.get("group_size", 128))
                qw, sc, zr, perm = unpack_gptq_layer(
                    qw_raw, qz_raw, sc_raw, group_size=gs, g_idx=g_idx
                )
                tdict: dict[str, torch.Tensor] = {
                    "qweight": qw.cuda().contiguous(),
                    "scales": sc.cuda().contiguous(),
                    "zeros": zr.cuda().contiguous(),
                }
                if perm is not None and bool(qcfg.get("desc_act", False)):
                    tdict["perm"] = perm.to(dtype=torch.int32, device="cuda").contiguous()
                return UniversalQuantLinear(
                    QuantFormat.GPTQ, int(qw.shape[0] * 2), int(qw.shape[1]), tdict, group_size=gs
                )
            if fmt_kind == "awq":
                qw_raw = reader[f"{key_base}.qweight"].cuda().contiguous()
                qz_raw = reader[f"{key_base}.qzeros"].cuda().contiguous()
                sc_raw = (
                    reader[f"{key_base}.scales"].to(device="cuda", dtype=torch.float16).contiguous()
                )
                k_d = int(qw_raw.shape[0])
                n_d = int(qw_raw.shape[1] * 8)
                gs = int(qcfg.get("group_size", 128))
                return UniversalQuantLinear(
                    QuantFormat.AWQ,
                    k_d,
                    n_d,
                    {"qweight_awq": qw_raw, "qzeros_awq": qz_raw, "scales": sc_raw},
                    group_size=gs,
                )
            if fmt_kind == "fp8_channel":
                w_f8 = reader[f"{key_base}.weight"].t().cuda().contiguous()
                sc_n = (
                    reader[f"{key_base}.weight_scale"]
                    .reshape(-1)
                    .to(device="cuda", dtype=torch.float32)
                    .contiguous()
                )
                return UniversalQuantLinear(
                    QuantFormat.FP8_CHANNEL,
                    int(w_f8.shape[0]),
                    int(w_f8.shape[1]),
                    {"weight_fp8": w_f8, "scale_n": sc_n},
                )
            raise ValueError(fmt_kind)

        u_layers.append(
            UniversalQuantLayer(
                layer_idx=idx,
                input_layernorm=in_ln,
                post_attention_layernorm=post_ln,
                q_proj=_make_proj("q_proj"),
                k_proj=_make_proj("k_proj"),
                v_proj=_make_proj("v_proj"),
                o_proj=_make_proj("o_proj"),
                gate_proj=_make_proj("gate_proj"),
                up_proj=_make_proj("up_proj"),
                down_proj=_make_proj("down_proj"),
                q_bias=qb,
                k_bias=kb,
                v_bias=vb,
            )
        )

    return UniversalQuantModel(
        config=cfg,
        embed_tokens=embed,
        norm=norm,
        lm_head=lm_head,
        cos=cos,
        sin=sin,
        layers=u_layers,
        format_table=MixedPrecisionFormatTable(default_format=qfmt_enum),
        rotary_style="half",
    )


def _load_gguf_uqm(gguf_path: Path) -> UniversalQuantModel:
    reader = gguf.GGUFReader(str(gguf_path), "r")
    tensors_by_name = {t.name: t for t in reader.tensors}
    arch = str(reader.fields["general.architecture"].contents())

    def _fval(key: str, default: Any = None) -> Any:
        return reader.fields[key].contents() if key in reader.fields else default

    hidden = int(_fval(f"{arch}.embedding_length"))
    inter = int(_fval(f"{arch}.feed_forward_length"))
    n_layers = int(_fval(f"{arch}.block_count"))
    n_heads = int(_fval(f"{arch}.attention.head_count"))
    n_kv = int(_fval(f"{arch}.attention.head_count_kv", n_heads))
    head_dim = hidden // n_heads
    rms_eps = float(_fval(f"{arch}.attention.layer_norm_rms_epsilon", 1e-6))
    rope_theta = float(_fval(f"{arch}.rope.freq_base", 10000.0))
    vocab = int(tensors_by_name["token_embd.weight"].shape[1])

    cfg = ModelConfig(
        vocab_size=vocab,
        hidden_size=hidden,
        intermediate_size=inter,
        num_hidden_layers=n_layers,
        num_attention_heads=n_heads,
        num_key_value_heads=n_kv,
        head_dim=head_dim,
        max_position_embeddings=32768,
        rms_norm_eps=rms_eps,
        rope_theta=rope_theta,
        sliding_window=4096 if "mistral" in gguf_path.name.lower() else None,
    )

    def _dequant_top(name: str) -> torch.Tensor:
        t = tensors_by_name[name]
        arr = np.array(t.data, copy=True)
        deq = gguf.quants.dequantize(arr, t.tensor_type)
        return torch.from_numpy(np.ascontiguousarray(deq)).to(device="cuda", dtype=torch.float16)

    embed = _dequant_top("token_embd.weight")
    norm = _dequant_top("output_norm.weight")
    lm_head = _dequant_top("output.weight") if "output.weight" in tensors_by_name else embed

    cos, sin = precompute_rope_freqs(32768, head_dim, theta=rope_theta, device="cuda")

    def _make_gguf_proj(tname: str) -> UniversalQuantLinear:
        t = tensors_by_name[tname]
        k_d, n_d = int(t.shape[0]), int(t.shape[1])
        raw_u8 = torch.from_numpy(np.ascontiguousarray(t.data).reshape(-1).copy()).to("cuda")
        tname_type = t.tensor_type.name
        if tname_type == "Q4_0":
            qs, d = parse_gguf_q4_0_native(raw_u8, k_d, n_d)
            return UniversalQuantLinear(
                QuantFormat.GGUF_Q4_0,
                k_d,
                n_d,
                {"qs": qs, "d": d, "raw_blocks": raw_u8},
                group_size=32,
            )
        if tname_type == "Q8_0":
            qw_i8, sc = parse_gguf_q8_0_native(raw_u8, k_d, n_d)
            return UniversalQuantLinear(
                QuantFormat.GGUF_Q8_0,
                k_d,
                n_d,
                {"qw_i8": qw_i8, "scales": sc, "raw_blocks": raw_u8},
                group_size=32,
            )
        if tname_type == "Q4_K":
            qs, d_sub, dm_sub = parse_gguf_q4_k_native(raw_u8, k_d, n_d)
            return UniversalQuantLinear(
                QuantFormat.GGUF_Q4_K,
                k_d,
                n_d,
                {"qs": qs, "d_sub": d_sub, "dm_sub": dm_sub, "raw_blocks": raw_u8},
                group_size=32,
            )
        if tname_type == "Q6_K":
            q_i8, d_sub = parse_gguf_q6_k_native(raw_u8, k_d, n_d)
            return UniversalQuantLinear(
                QuantFormat.GGUF_Q6_K,
                k_d,
                n_d,
                {"q_i8": q_i8, "d_sub": d_sub, "raw_blocks": raw_u8},
                group_size=16,
            )
        raise NotImplementedError(
            f"GGML tensor type '{tname_type}' has no native CuTile kernel ({tname})"
        )

    def _opt_f32(name: str) -> torch.Tensor | None:
        if name not in tensors_by_name:
            return None
        t = tensors_by_name[name]
        return torch.from_numpy(np.ascontiguousarray(t.data).copy()).to(
            device="cuda", dtype=torch.float16
        )

    u_layers: list[UniversalQuantLayer] = []
    for idx in range(n_layers):
        b = f"blk.{idx}"
        u_layers.append(
            UniversalQuantLayer(
                layer_idx=idx,
                input_layernorm=_opt_f32(f"{b}.attn_norm.weight"),
                post_attention_layernorm=_opt_f32(f"{b}.ffn_norm.weight"),
                q_proj=_make_gguf_proj(f"{b}.attn_q.weight"),
                k_proj=_make_gguf_proj(f"{b}.attn_k.weight"),
                v_proj=_make_gguf_proj(f"{b}.attn_v.weight"),
                o_proj=_make_gguf_proj(f"{b}.attn_output.weight"),
                gate_proj=_make_gguf_proj(f"{b}.ffn_gate.weight"),
                up_proj=_make_gguf_proj(f"{b}.ffn_up.weight"),
                down_proj=_make_gguf_proj(f"{b}.ffn_down.weight"),
                q_bias=_opt_f32(f"{b}.attn_q.bias"),
                k_bias=_opt_f32(f"{b}.attn_k.bias"),
                v_bias=_opt_f32(f"{b}.attn_v.bias"),
            )
        )

    rotary_style = "half" if arch == "qwen2" else "interleaved"
    return UniversalQuantModel(
        config=cfg,
        embed_tokens=embed,
        norm=norm,
        lm_head=lm_head,
        cos=cos,
        sin=sin,
        layers=u_layers,
        format_table=MixedPrecisionFormatTable(default_format=QuantFormat.GGUF_Q4_0),
        rotary_style=rotary_style,
    )


def run_phase15_quant_validation(output_dir: Path) -> Path:
    qroot = Path("/var/tmp/arrhat_models/quant_eval")
    mistral_gptq_dir = Path("/home/arrhat/models/mistral_7b_4bit")
    tok_q = AutoTokenizer.from_pretrained(str(qroot / "qwen25_05b_fp16"), local_files_only=True)
    tok_m = AutoTokenizer.from_pretrained(str(mistral_gptq_dir), local_files_only=True)

    q_ids = tok_q.encode(WIKITEXT2_EXTENDED, add_special_tokens=False)[:512]
    m_ids = [1] + tok_m.encode(WIKITEXT2_EXTENDED, add_special_tokens=False)[:511]
    assert len(q_ids) >= 512 and len(m_ids) >= 512

    ckpt = build_checkpoint_manifest("Qwen/Qwen2.5-0.5B-Instruct", qroot / "qwen25_05b_fp16")
    run = ValidationRun(
        run_dir=output_dir,
        run_id="phase15_quant_real",
        checkpoint=ckpt,
        quant_config=QuantizationConfig(
            quant_format="multi_format_real_checkpoints",
            group_size=128,
            desc_act=True,
            kv_dtype="fp16",
            disk_format="gptq+awq+gguf+fp8_channel",
            kernel_format="native_cutile_all_layers",
        ),
        workload=WorkloadConfig(
            batch_size=1,
            concurrency=1,
            prompt_len=512,
            output_len=1,
            temperature=0.0,
            seed=42,
            warmup_iters=1,
            measured_iters=5,
        ),
        prompt_ids_or_checksum=prompt_checksum(q_ids),
        execution_mode="real-checkpoint",
        reproduction_status="MEASURED",
        notes="Phase 15.3 real-checkpoint all-layer perplexity validation (>= 512 non-repeated tokens).",
    )

    # 1. FP16 Baseline on Qwen2.5-0.5B-Instruct (24L)
    run.log_stdout(
        "[Phase 15.3] Computing FP16 baseline (Qwen/Qwen2.5-0.5B-Instruct, 512 tokens)..."
    )
    q_t = torch.tensor([q_ids], device="cuda", dtype=torch.long)
    hf_fp16 = AutoModelForCausalLM.from_pretrained(
        str(qroot / "qwen25_05b_fp16"),
        dtype=torch.float16,
        device_map="cuda",
        local_files_only=True,
    )
    with torch.no_grad():
        fp16_logits = hf_fp16(q_t).logits[0].float()
        q_fp16_ppl = math.exp(float(F.cross_entropy(fp16_logits[:-1], q_t[0, 1:]).item()))
    del hf_fp16
    gc.collect()
    torch.cuda.empty_cache()
    run.log_stdout(f"  Qwen2.5-0.5B-Instruct FP16 PPL (512 tok): {q_fp16_ppl:.4f}")
    run.add_record(
        {
            "model": "Qwen2.5-0.5B-Instruct",
            "format": "FP16_reference",
            "checkpoint": "Qwen/Qwen2.5-0.5B-Instruct",
            "layers": 24,
            "eval_tokens": len(q_ids),
            "weight_source": "real-checkpoint",
            "ppl_native": round(q_fp16_ppl, 4),
            "ppl_format_ref": round(q_fp16_ppl, 4),
            "ppl_fp16_ref": round(q_fp16_ppl, 4),
            "ppl_vs_format_ref_diff_pct": 0.0,
            "ppl_vs_fp16_diff_pct": 0.0,
            "logit_cos_sim": 1.0,
            "status": "PASS",
        }
    )

    # 2. Qwen2.5-0.5B-Instruct native quantized formats (24L)
    qwen_specs: list[tuple[str, str, Path, str]] = [
        (
            "GPTQ_Int4",
            "Qwen/Qwen2.5-0.5B-Instruct-GPTQ-Int4",
            qroot / "qwen25_05b_gptq",
            "safetensors:gptq",
        ),
        ("AWQ_Int4", "Qwen/Qwen2.5-0.5B-Instruct-AWQ", qroot / "qwen25_05b_awq", "safetensors:awq"),
        (
            "FP8_channel",
            "RedHatAI/Qwen2.5-0.5B-Instruct-FP8-dynamic",
            qroot / "qwen25_05b_fp8",
            "safetensors:fp8_channel",
        ),
        (
            "GGUF_Q4_0",
            "Qwen/Qwen2.5-0.5B-Instruct-GGUF (q4_0)",
            qroot / "qwen25_05b_gguf/qwen2.5-0.5b-instruct-q4_0.gguf",
            "gguf",
        ),
        (
            "GGUF_Q8_0",
            "Qwen/Qwen2.5-0.5B-Instruct-GGUF (q8_0)",
            qroot / "qwen25_05b_gguf/qwen2.5-0.5b-instruct-q8_0.gguf",
            "gguf",
        ),
        (
            "GGUF_Q6_K",
            "Qwen/Qwen2.5-0.5B-Instruct-GGUF (q6_k)",
            qroot / "qwen25_05b_gguf/qwen2.5-0.5b-instruct-q6_k.gguf",
            "gguf",
        ),
        (
            "GGUF_Q4_K_M",
            "Qwen/Qwen2.5-0.5B-Instruct-GGUF (q4_k_m)",
            qroot / "qwen25_05b_gguf/qwen2.5-0.5b-instruct-q4_k_m.gguf",
            "gguf",
        ),
    ]
    for fmt_name, ckpt_label, ckpt_path, loader_kind in qwen_specs:
        torch.cuda.reset_peak_memory_stats()
        t0 = time.perf_counter()
        try:
            if loader_kind.startswith("safetensors:"):
                uqm = _load_qwen_safetensors_uqm(ckpt_path, loader_kind.split(":")[1])
            else:
                uqm = _load_gguf_uqm(ckpt_path)
            load_s = time.perf_counter() - t0
            m = _eval_uqm_metrics(uqm, q_t, warmup=1, runs=5)
            vram_alloc, _ = peak_vram_mb()
            rss = host_rss_mb()
            plausible = 1.0 < m["ppl_native"] < 50.0 and m["logit_cos_sim"] >= 0.999
            status = "PASS" if (plausible and m["ppl_vs_format_ref_diff_pct"] <= 0.25) else "FAIL"
            fp16_diff_pct = (m["ppl_native"] - q_fp16_ppl) / max(1e-9, q_fp16_ppl) * 100.0
            run.log_stdout(
                f"  [Qwen2.5-0.5B {fmt_name:12s}] status={status} ppl_nat={m['ppl_native']:.4f} "
                f"ppl_fmt_ref={m['ppl_format_ref']:.4f} ({m['ppl_vs_format_ref_diff_pct']:.4f}%) "
                f"vs_fp16={fp16_diff_pct:+.2f}% cos={m['logit_cos_sim']:.6f} "
                f"med={m['med_ms']:.2f}ms [{m['min_ms']:.2f}..{m['max_ms']:.2f}]"
            )
            run.add_record(
                {
                    "model": "Qwen2.5-0.5B-Instruct",
                    "format": fmt_name,
                    "checkpoint": ckpt_label,
                    "layers": uqm.config.num_hidden_layers,
                    "eval_tokens": len(q_ids),
                    "weight_source": "real-checkpoint",
                    "ppl_native": round(m["ppl_native"], 4),
                    "ppl_format_ref": round(m["ppl_format_ref"], 4),
                    "ppl_fp16_ref": round(q_fp16_ppl, 4),
                    "ppl_vs_format_ref_diff_pct": round(m["ppl_vs_format_ref_diff_pct"], 4),
                    "ppl_vs_fp16_diff_pct": round(fp16_diff_pct, 2),
                    "logit_cos_sim": round(m["logit_cos_sim"], 6),
                    "logit_max_abs_err": round(m["logit_max_abs_err"], 6),
                    "med_prefill_512_ms": round(m["med_ms"], 3),
                    "min_prefill_512_ms": round(m["min_ms"], 3),
                    "max_prefill_512_ms": round(m["max_ms"], 3),
                    "load_s": round(load_s, 2),
                    "peak_vram_allocated_mb": round(vram_alloc, 1),
                    "host_rss_mb": round(rss, 1),
                    "status": status,
                }
            )
            del uqm
        except NotImplementedError as exc:
            run.log_stdout(f"  [Qwen2.5-0.5B {fmt_name:12s}] UNSUPPORTED: {exc}")
            run.add_record(
                {
                    "model": "Qwen2.5-0.5B-Instruct",
                    "format": fmt_name,
                    "checkpoint": ckpt_label,
                    "layers": 24,
                    "eval_tokens": len(q_ids),
                    "weight_source": "real-checkpoint",
                    "status": "UNSUPPORTED",
                    "error": str(exc),
                }
            )
        gc.collect()
        torch.cuda.empty_cache()

    # 3. Mistral-7B-v0.1 Real Checkpoints (32L GPTQ on-disk + 32L GGUF Q4_K_M on-disk)
    m_t = torch.tensor([m_ids], device="cuda", dtype=torch.long)

    # 3a. Mistral-7B-v0.1-GPTQ (32L)
    torch.cuda.reset_peak_memory_stats()
    t0 = time.perf_counter()
    from cutile_forge.engine.config import GPUConfig
    from cutile_forge.engine.model import Mistral7B

    w_m = load_gptq_weights(mistral_gptq_dir, device="cuda", pin_ddr5=False)
    m_model = Mistral7B(
        weights=w_m, gpu_config=GPUConfig(vram_gb=32.0, stream_weights=False), device="cuda"
    )
    u_m_gptq = UniversalQuantModel.from_mistral_model(m_model, QuantFormat.GPTQ, num_layers=32)
    load_s = time.perf_counter() - t0
    mm = _eval_uqm_metrics(u_m_gptq, m_t, warmup=1, runs=5)
    vram_alloc, _ = peak_vram_mb()
    plausible_m = 1.0 < mm["ppl_native"] < 15.0 and mm["logit_cos_sim"] >= 0.9999
    status_m = "PASS" if (plausible_m and mm["ppl_vs_format_ref_diff_pct"] <= 0.25) else "FAIL"
    run.log_stdout(
        f"  [Mistral-7B   GPTQ_32L    ] status={status_m} ppl_nat={mm['ppl_native']:.4f} "
        f"ppl_fmt_ref={mm['ppl_format_ref']:.4f} ({mm['ppl_vs_format_ref_diff_pct']:.4f}%) "
        f"cos={mm['logit_cos_sim']:.6f} med={mm['med_ms']:.2f}ms"
    )
    run.add_record(
        {
            "model": "Mistral-7B-v0.1",
            "format": "GPTQ_Int4_desc_act",
            "checkpoint": str(mistral_gptq_dir),
            "layers": 32,
            "eval_tokens": len(m_ids),
            "weight_source": "real-checkpoint",
            "ppl_native": round(mm["ppl_native"], 4),
            "ppl_format_ref": round(mm["ppl_format_ref"], 4),
            "ppl_vs_format_ref_diff_pct": round(mm["ppl_vs_format_ref_diff_pct"], 4),
            "logit_cos_sim": round(mm["logit_cos_sim"], 6),
            "logit_max_abs_err": round(mm["logit_max_abs_err"], 6),
            "med_prefill_512_ms": round(mm["med_ms"], 3),
            "min_prefill_512_ms": round(mm["min_ms"], 3),
            "max_prefill_512_ms": round(mm["max_ms"], 3),
            "load_s": round(load_s, 2),
            "peak_vram_allocated_mb": round(vram_alloc, 1),
            "status": status_m,
        }
    )
    del u_m_gptq, m_model, w_m
    gc.collect()
    torch.cuda.empty_cache()

    # 3b. Mistral-7B-v0.1-GGUF (Q4_K_M, 32L real binary GGUF)
    torch.cuda.reset_peak_memory_stats()
    t0 = time.perf_counter()
    u_m_gguf = _load_gguf_uqm(qroot / "mistral_7b_gguf/mistral-7b-v0.1.Q4_K_M.gguf")
    load_s = time.perf_counter() - t0
    mg = _eval_uqm_metrics(u_m_gguf, m_t, warmup=1, runs=5)
    vram_alloc, _ = peak_vram_mb()
    plausible_mg = 1.0 < mg["ppl_native"] < 15.0 and mg["logit_cos_sim"] >= 0.9999
    status_mg = "PASS" if (plausible_mg and mg["ppl_vs_format_ref_diff_pct"] <= 0.25) else "FAIL"
    run.log_stdout(
        f"  [Mistral-7B   GGUF_Q4_K_M ] status={status_mg} ppl_nat={mg['ppl_native']:.4f} "
        f"ppl_fmt_ref={mg['ppl_format_ref']:.4f} ({mg['ppl_vs_format_ref_diff_pct']:.4f}%) "
        f"cos={mg['logit_cos_sim']:.6f} med={mg['med_ms']:.2f}ms"
    )
    run.add_record(
        {
            "model": "Mistral-7B-v0.1",
            "format": "GGUF_Q4_K_M",
            "checkpoint": "TheBloke/Mistral-7B-v0.1-GGUF (Q4_K_M)",
            "layers": 32,
            "eval_tokens": len(m_ids),
            "weight_source": "real-checkpoint",
            "ppl_native": round(mg["ppl_native"], 4),
            "ppl_format_ref": round(mg["ppl_format_ref"], 4),
            "ppl_vs_format_ref_diff_pct": round(mg["ppl_vs_format_ref_diff_pct"], 4),
            "logit_cos_sim": round(mg["logit_cos_sim"], 6),
            "logit_max_abs_err": round(mg["logit_max_abs_err"], 6),
            "med_prefill_512_ms": round(mg["med_ms"], 3),
            "min_prefill_512_ms": round(mg["min_ms"], 3),
            "max_prefill_512_ms": round(mg["max_ms"], 3),
            "load_s": round(load_s, 2),
            "peak_vram_allocated_mb": round(vram_alloc, 1),
            "status": status_mg,
        }
    )
    del u_m_gguf
    gc.collect()
    torch.cuda.empty_cache()

    out_path = run.finalize()
    errs = validate_artifact_dir(output_dir)
    if errs:
        raise RuntimeError(f"Validation artifact schema errors: {errs}")
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Phase 15.3 Real-Checkpoint Quantization Validation"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("benchmarks/validation_artifacts/v3.2.0/phase15_quant_real"),
    )
    args = parser.parse_args()
    out = run_phase15_quant_validation(args.output)
    print(f"Phase 15.3 real-checkpoint quantization validation complete: {out}")


if __name__ == "__main__":
    main()
