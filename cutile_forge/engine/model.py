"""Mistral-7B model definition with TFG compilation and CuTile kernel execution."""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn.functional as F

from cutile_forge.engine.config import (
    LLAMA3_70B_4BIT,
    LLAMA_34B_4BIT,
    GPUConfig,
    MISTRAL_8x7B_4BIT,
    ModelConfig,
)
from cutile_forge.engine.kv_cache import PagedKVCache
from cutile_forge.engine.weight_loader import (
    LoadedModelWeights,
    MistralLayerWeights,
    QuantLinearWeight,
    create_random_4bit_weights,
    load_gptq_weights,
)
from cutile_forge.forge.codegen import CompiledTFG, compile_tfg
from cutile_forge.forge.fusion_graph import build_mistral_layer_tfg
from cutile_forge.forge.fusion_rules import apply_fusion_rules
from cutile_forge.forge.weight_streamer import WeightStreamer
from cutile_forge.kernels.attention_gqa import fused_attention_gqa
from cutile_forge.kernels.embedding_lmhead import cutile_embedding_lookup
from cutile_forge.kernels.gemm_dequant import (
    cutile_gemm_fp16,
    fused_gemm_dequant,
    unpack_int4_weights_torch,
)
from cutile_forge.kernels.gemm_dequant_silu import fused_gemm_dequant_silu
from cutile_forge.kernels.layernorm_residual import fused_rmsnorm_residual
from cutile_forge.kernels.moe_fused import (
    cutile_moe_indexed_ffn,
    cutile_moe_router_topk,
    cutile_moe_weighted_add,
    fused_moe_int4_gemm,
    fused_topk_gating,
)
from cutile_forge.kernels.rmsnorm import cutile_rmsnorm
from cutile_forge.kernels.rope import apply_rope, precompute_rope_freqs


def _pytorch_rmsnorm(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    x_f = x.float()
    inv_rms = torch.rsqrt(x_f.pow(2).mean(dim=-1, keepdim=True) + eps)
    return (x_f * inv_rms * weight.float()).to(torch.float16)


def _pytorch_rope(
    x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, pos_offset: int
) -> torch.Tensor:
    seq_len = x.shape[2]
    half_d = x.shape[3] // 2
    c = cos[pos_offset : pos_offset + seq_len, :half_d].view(1, 1, seq_len, half_d).float()
    s = sin[pos_offset : pos_offset + seq_len, :half_d].view(1, 1, seq_len, half_d).float()
    x0 = x[..., :half_d].float()
    x1 = x[..., half_d:].float()
    return torch.cat((x0 * c - x1 * s, x1 * c + x0 * s), dim=-1).to(torch.float16)


def _quant_linear_cutile(x: torch.Tensor, qw: QuantLinearWeight) -> torch.Tensor:
    x_in = x[:, qw.perm] if qw.perm is not None else x
    return fused_gemm_dequant(x_in, qw.qweight, qw.scales, qw.zeros, group_size=qw.group_size)


def _quant_linear_pytorch(x: torch.Tensor, qw: QuantLinearWeight) -> torch.Tensor:
    x_in = x[:, qw.perm] if qw.perm is not None else x
    w_fp16 = unpack_int4_weights_torch(qw.qweight, qw.scales, qw.zeros, group_size=qw.group_size)
    return torch.matmul(x_in, w_fp16)


class MoEFFNBlock:
    """Sparse Mixture-of-Experts FFN block (Mixtral-8x7B) using CuTile fused MoE."""

    def __init__(self, num_experts: int = 8, top_k: int = 2, group_size: int = 128) -> None:
        self.num_experts = num_experts
        self.top_k = top_k
        self.group_size = group_size

    def forward(
        self,
        x: torch.Tensor,
        router_weight: torch.Tensor,
        expert_qweights: torch.Tensor,
        expert_scales: torch.Tensor,
        expert_zeros: torch.Tensor,
    ) -> torch.Tensor:
        if x.is_cuda and x.dtype == torch.float16:
            probs, indices = cutile_moe_router_topk(x, router_weight, top_k=self.top_k)
        else:
            logits = torch.matmul(x, router_weight.t())
            probs, indices = fused_topk_gating(logits, top_k=self.top_k)
        return fused_moe_int4_gemm(
            x,
            expert_qweights,
            expert_scales,
            expert_zeros,
            probs,
            indices,
            group_size=self.group_size,
        )


class MistralLayer:
    """Single Mistral transformer layer executing via CuTile kernels or PyTorch reference."""

    def __init__(self, layer_idx: int, config: ModelConfig) -> None:
        self.layer_idx = layer_idx
        self.config = config
        self.moe_block: MoEFFNBlock | None = (
            MoEFFNBlock(
                num_experts=config.experts,
                top_k=getattr(config, "top_k_experts", 2),
                group_size=config.group_size,
            )
            if config.experts > 1
            else None
        )
        self.tfg = build_mistral_layer_tfg(layer_idx, config=config)
        apply_fusion_rules(self.tfg)
        self.compiled_tfg: CompiledTFG = compile_tfg(
            self.tfg, target="sm_89", pipeline_depth=3, autotune=True, compile_cubin=False
        )

    def forward_cutile(
        self,
        hidden_states: torch.Tensor,
        weights: MistralLayerWeights,
        cos: torch.Tensor,
        sin: torch.Tensor,
        kv_cache: PagedKVCache | None = None,
        pos_offset: int = 0,
    ) -> torch.Tensor:
        """Execute one layer using fused CuTile kernels."""
        bsz, seq_len, hidden_dim = hidden_states.shape
        x_2d = hidden_states.view(bsz * seq_len, hidden_dim)

        x_norm = cutile_rmsnorm(x_2d, weights.input_layernorm, eps=self.config.rms_norm_eps)
        q_dim = self.config.num_attention_heads * self.config.head_dim
        kv_dim = self.config.num_key_value_heads * self.config.head_dim
        if weights.qkv_qw is not None and weights.qkv_sc is not None and weights.qkv_zr is not None:
            qkv = fused_gemm_dequant(
                x_norm,
                weights.qkv_qw,
                weights.qkv_sc,
                weights.qkv_zr,
                group_size=weights.q_proj.group_size,
            )
            q = qkv[:, :q_dim]
            k = qkv[:, q_dim : q_dim + kv_dim]
            v = qkv[:, q_dim + kv_dim :]
        else:
            q = _quant_linear_cutile(x_norm, weights.q_proj)
            k = _quant_linear_cutile(x_norm, weights.k_proj)
            v = _quant_linear_cutile(x_norm, weights.v_proj)

        q = q.reshape(
            bsz, seq_len, self.config.num_attention_heads, self.config.head_dim
        ).transpose(1, 2)
        k = k.reshape(
            bsz, seq_len, self.config.num_key_value_heads, self.config.head_dim
        ).transpose(1, 2)
        v = v.reshape(
            bsz, seq_len, self.config.num_key_value_heads, self.config.head_dim
        ).transpose(1, 2)

        q = apply_rope(q, cos, sin, position_offset=pos_offset)
        k = apply_rope(k, cos, sin, position_offset=pos_offset)

        if kv_cache is not None:
            kv_cache.update(self.layer_idx, k, v, defer_prune=(seq_len > 1))
            k_full, v_full = kv_cache.get_kv(self.layer_idx)
        else:
            k_full, v_full = k, v

        attn_out = fused_attention_gqa(
            q,
            k_full,
            v_full,
            causal=True,
            sliding_window=int(self.config.sliding_window or 0),
        )
        if kv_cache is not None and seq_len > 1 and kv_cache.sliding_window is not None:
            kv_cache.prune_outside_window(self.layer_idx)
        attn_2d = (
            attn_out.view(bsz, hidden_dim)
            if seq_len == 1
            else attn_out.transpose(1, 2).contiguous().view(bsz * seq_len, hidden_dim)
        )

        o_proj = _quant_linear_cutile(attn_2d, weights.o_proj)

        post_norm, residual_mid = fused_rmsnorm_residual(
            o_proj, x_2d, weights.post_attention_layernorm, eps=self.config.rms_norm_eps
        )

        if (
            weights.moe_gate is not None
            and weights.moe_w1 is not None
            and weights.moe_w2 is not None
            and weights.moe_w3 is not None
        ):
            gate_w = (
                weights.moe_gate
                if weights.moe_gate.dtype == post_norm.dtype
                else weights.moe_gate.to(post_norm.dtype)
            )
            probs, indices = cutile_moe_router_topk(
                post_norm,
                gate_w,
                top_k=self.config.top_k_experts,
                out_dtype=torch.float32,
            )
            if (
                post_norm.shape[0] == 1
                and post_norm.is_cuda
                and weights.moe_w1_qw is not None
                and weights.moe_w1_sc is not None
                and weights.moe_w1_zr is not None
                and weights.moe_w2_qw is not None
                and weights.moe_w2_sc is not None
                and weights.moe_w2_zr is not None
                and weights.moe_w3_qw is not None
                and weights.moe_w3_sc is not None
                and weights.moe_w3_zr is not None
            ):
                out_h = cutile_moe_indexed_ffn(
                    post_norm,
                    weights.moe_w1_qw,
                    weights.moe_w1_sc,
                    weights.moe_w1_zr,
                    weights.moe_w3_qw,
                    weights.moe_w3_sc,
                    weights.moe_w3_zr,
                    weights.moe_w2_qw,
                    weights.moe_w2_sc,
                    weights.moe_w2_zr,
                    probs,
                    indices,
                    residual=residual_mid,
                    group_size=self.config.group_size,
                )
                return out_h.view(bsz, seq_len, hidden_dim)
            ffn_acc = torch.zeros_like(post_norm, dtype=torch.float32)
            for e in range(len(weights.moe_w1)):
                mask = indices == e
                if not bool(mask.any()):
                    continue
                tok_idx, k_slot = torch.where(mask)
                x_e = post_norm.index_select(0, tok_idx)
                w1 = weights.moe_w1[e]
                w2 = weights.moe_w2[e]
                w3 = weights.moe_w3[e]
                g_in = x_e[:, w1.perm] if w1.perm is not None else x_e
                g_act = fused_gemm_dequant_silu(
                    g_in, w1.qweight, w1.scales, w1.zeros, group_size=w1.group_size
                )
                u_val = _quant_linear_cutile(x_e, w3)
                y_e = _quant_linear_cutile(g_act * u_val, w2)
                cutile_moe_weighted_add(ffn_acc, tok_idx, y_e, probs[tok_idx, k_slot])
            ffn_out = ffn_acc.to(post_norm.dtype)
            return (residual_mid + ffn_out).view(bsz, seq_len, hidden_dim)

        gate_in = (
            post_norm[:, weights.gate_proj.perm]
            if weights.gate_proj.perm is not None
            else post_norm
        )
        gate_act = fused_gemm_dequant_silu(
            gate_in,
            weights.gate_proj.qweight,
            weights.gate_proj.scales,
            weights.gate_proj.zeros,
            group_size=weights.gate_proj.group_size,
        )
        up_val = _quant_linear_cutile(post_norm, weights.up_proj)
        ffn_out = _quant_linear_cutile(gate_act * up_val, weights.down_proj)

        return (residual_mid + ffn_out).view(bsz, seq_len, hidden_dim)

    def forward_pytorch(
        self,
        hidden_states: torch.Tensor,
        weights: MistralLayerWeights,
        cos: torch.Tensor,
        sin: torch.Tensor,
        kv_cache: PagedKVCache | None = None,
        pos_offset: int = 0,
    ) -> torch.Tensor:
        """Reference PyTorch execution using identical dequantized 4-bit weights."""
        bsz, seq_len, hidden_dim = hidden_states.shape
        x_2d = hidden_states.view(bsz * seq_len, hidden_dim)

        x_norm = _pytorch_rmsnorm(x_2d, weights.input_layernorm, eps=self.config.rms_norm_eps)
        q = _quant_linear_pytorch(x_norm, weights.q_proj)
        k = _quant_linear_pytorch(x_norm, weights.k_proj)
        v = _quant_linear_pytorch(x_norm, weights.v_proj)

        q = q.view(bsz, seq_len, self.config.num_attention_heads, self.config.head_dim).transpose(
            1, 2
        )
        k = k.view(bsz, seq_len, self.config.num_key_value_heads, self.config.head_dim).transpose(
            1, 2
        )
        v = v.view(bsz, seq_len, self.config.num_key_value_heads, self.config.head_dim).transpose(
            1, 2
        )

        q = _pytorch_rope(q, cos, sin, pos_offset)
        k = _pytorch_rope(k, cos, sin, pos_offset)

        if kv_cache is not None:
            kv_cache.update(self.layer_idx, k, v, defer_prune=(seq_len > 1))
            k_full, v_full = kv_cache.get_kv(self.layer_idx)
        else:
            k_full, v_full = k, v

        k_rep = k_full.repeat_interleave(self.config.gqa_ratio, dim=1)
        v_rep = v_full.repeat_interleave(self.config.gqa_ratio, dim=1)
        s_kv = k_full.shape[2]
        sw = int(self.config.sliding_window or 0)
        if sw > 0 or (seq_len > 1 and s_kv > seq_len):
            q_p = torch.arange(seq_len, device=q.device).view(seq_len, 1) + (s_kv - seq_len)
            k_p = torch.arange(s_kv, device=q.device).view(1, s_kv)
            mask = k_p <= q_p
            if sw > 0:
                mask = mask & (k_p > q_p - sw)
            attn_out = F.scaled_dot_product_attention(q, k_rep, v_rep, attn_mask=mask)
        else:
            is_causal = seq_len > 1 and s_kv == seq_len
            attn_out = F.scaled_dot_product_attention(q, k_rep, v_rep, is_causal=is_causal)
        if kv_cache is not None and seq_len > 1 and kv_cache.sliding_window is not None:
            kv_cache.prune_outside_window(self.layer_idx)

        attn_2d = attn_out.transpose(1, 2).contiguous().view(bsz * seq_len, hidden_dim)
        o_proj = _quant_linear_pytorch(attn_2d, weights.o_proj)
        residual_mid = (x_2d.float() + o_proj.float()).to(torch.float16)
        post_norm = _pytorch_rmsnorm(
            residual_mid, weights.post_attention_layernorm, eps=self.config.rms_norm_eps
        )

        if (
            weights.moe_gate is not None
            and weights.moe_w1 is not None
            and weights.moe_w2 is not None
            and weights.moe_w3 is not None
        ):
            r_logits = torch.matmul(post_norm.float(), weights.moe_gate.float().t())
            probs, indices = fused_topk_gating(r_logits, top_k=self.config.top_k_experts)
            ffn_out = torch.zeros_like(post_norm)
            for e in range(len(weights.moe_w1)):
                mask = indices == e
                if not bool(mask.any()):
                    continue
                tok_idx, k_slot = torch.where(mask)
                x_e = post_norm.index_select(0, tok_idx)
                g_e = F.silu(_quant_linear_pytorch(x_e, weights.moe_w1[e]).float()).to(
                    torch.float16
                )
                u_e = _quant_linear_pytorch(x_e, weights.moe_w3[e])
                y_e = _quant_linear_pytorch(g_e * u_e, weights.moe_w2[e])
                ffn_out.index_add_(
                    0, tok_idx, (y_e * probs[tok_idx, k_slot].unsqueeze(1)).to(ffn_out.dtype)
                )
            return (residual_mid + ffn_out).view(bsz, seq_len, hidden_dim)

        gate = F.silu(_quant_linear_pytorch(post_norm, weights.gate_proj).float()).to(torch.float16)
        up = _quant_linear_pytorch(post_norm, weights.up_proj)
        ffn_out = _quant_linear_pytorch(gate * up, weights.down_proj)
        return (residual_mid + ffn_out).view(bsz, seq_len, hidden_dim)


class Mistral7B:
    """32-layer Mistral-7B inference model compiled with CuTileForge TFGs."""

    def __init__(
        self,
        weights: LoadedModelWeights,
        gpu_config: GPUConfig | None = None,
        device: str | torch.device = "cuda",
    ) -> None:
        self.weights = weights
        self.config = weights.config
        self.gpu_config = gpu_config or GPUConfig()
        self.device = torch.device(
            device if (str(device) != "cuda" or torch.cuda.is_available()) else "cpu"
        )
        self.layers: list[MistralLayer] = [
            MistralLayer(i, self.config) for i in range(self.config.num_hidden_layers)
        ]
        if self.device.type == "cuda":
            did_pack = False
            for lw in self.weights.layers:
                if lw.pack_moe_experts_inplace():
                    did_pack = True
                if lw.moe_gate is not None and lw.pack_qkv_inplace():
                    did_pack = True
            if did_pack:
                torch.cuda.empty_cache()
        est_model_gb = (
            self.config.num_hidden_layers
            * self.config.hidden_size
            * self.config.intermediate_size
            * 3
            * (self.config.bits / 8.0)
            * max(1, self.config.experts)
        ) / (1024.0**3)
        self.streamer = WeightStreamer(
            num_slots=2,
            device=self.device,
            model_gb=max(4.16, est_model_gb),
            gpu_vram_gb=self.gpu_config.vram_gb,
        )
        max_pos = min(self.config.max_position_embeddings, 32768)
        self.cos, self.sin = precompute_rope_freqs(
            max_pos, self.config.head_dim, theta=self.config.rope_theta, device=self.device
        )

    @classmethod
    def from_pretrained(
        cls,
        model_dir: str | Path,
        gpu_config: GPUConfig | None = None,
        device: str | torch.device = "cuda",
        max_gpu_layers: int | None = None,
    ) -> Mistral7B:
        gcfg = gpu_config or GPUConfig()
        if max_gpu_layers is None and gcfg.stream_weights:
            max_gpu_layers = 0
        weights = load_gptq_weights(
            model_dir, device=device, pin_ddr5=True, max_gpu_layers=max_gpu_layers
        )
        return cls(weights=weights, gpu_config=gcfg, device=device)

    @classmethod
    def from_random(
        cls,
        config: ModelConfig | None = None,
        gpu_config: GPUConfig | None = None,
        device: str | torch.device = "cuda",
        seed: int = 42,
    ) -> Mistral7B:
        cfg = config or ModelConfig()
        weights = create_random_4bit_weights(cfg, device=device, seed=seed)
        return cls(weights=weights, gpu_config=gpu_config, device=device)

    def _get_layer_weights(self, layer_idx: int) -> MistralLayerWeights:
        lw = self.weights.layers[layer_idx]
        if lw.input_layernorm.device.type == self.device.type:
            return lw
        # Layer resides in pinned DDR5 -> acquire via Stage -1 DMA streamer
        d = self.streamer.acquire_layer(layer_idx, lw.to_tensor_dict())
        return MistralLayerWeights.from_tensor_dict(layer_idx, d, group_size=self.config.group_size)

    def _prefetch_next_layer(self, next_idx: int) -> None:
        if next_idx < self.config.num_hidden_layers:
            lw = self.weights.layers[next_idx]
            if lw.input_layernorm.device.type != self.device.type:
                self.streamer.prefetch_layer(next_idx, lw.to_tensor_dict())

    def forward(
        self,
        input_ids: torch.Tensor,
        kv_cache: PagedKVCache | None = None,
        pos_offset: int = 0,
        backend: str = "cutile",
        last_token_only: bool = False,
    ) -> torch.Tensor:
        """Run a forward pass over `input_ids` of shape `(B, S)` -> logits."""
        if backend == "cutile" and self.device.type == "cuda":
            hidden = cutile_embedding_lookup(input_ids.to(self.device), self.weights.embed_tokens)
        else:
            hidden = F.embedding(input_ids.to(self.device), self.weights.embed_tokens)

        for idx, layer in enumerate(self.layers):
            self._prefetch_next_layer(idx + 1)
            lw = self._get_layer_weights(idx)
            if backend == "cutile" and self.device.type == "cuda":
                hidden = layer.forward_cutile(
                    hidden, lw, self.cos, self.sin, kv_cache=kv_cache, pos_offset=pos_offset
                )
            else:
                hidden = layer.forward_pytorch(
                    hidden, lw, self.cos, self.sin, kv_cache=kv_cache, pos_offset=pos_offset
                )
            if lw is not self.weights.layers[idx]:
                self.streamer.release_layer(idx)

        if last_token_only:
            hidden = hidden[:, -1:, :]

        bsz, seq_len, h_dim = hidden.shape
        h_2d = hidden.view(bsz * seq_len, h_dim)
        if backend == "cutile" and self.device.type == "cuda":
            normed = cutile_rmsnorm(h_2d, self.weights.norm, eps=self.config.rms_norm_eps)
            if (
                normed.dtype == torch.float16
                and self.weights.lm_head.dtype == torch.float16
                and h_dim % 64 == 0
                and self.config.vocab_size % 64 == 0
            ):
                lm_w_t = getattr(self, "_lm_head_t_cache", None)
                if lm_w_t is None or lm_w_t.data_ptr() == 0:
                    lm_w_t = self.weights.lm_head.t().contiguous()
                    self._lm_head_t_cache = lm_w_t
                logits = cutile_gemm_fp16(normed, lm_w_t)
            else:
                logits = torch.matmul(normed, self.weights.lm_head.t())
        else:
            normed = _pytorch_rmsnorm(h_2d, self.weights.norm, eps=self.config.rms_norm_eps)
            logits = torch.matmul(normed, self.weights.lm_head.t())
        return logits.view(bsz, seq_len, self.config.vocab_size)


__all__ = [
    "LLAMA3_70B_4BIT",
    "LLAMA_34B_4BIT",
    "MISTRAL_8x7B_4BIT",
    "Mistral7B",
    "MistralLayer",
    "MoEFFNBlock",
]
