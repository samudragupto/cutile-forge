"""Native 100% CuTile prefill & autoregressive decode engine (zero-ATen compute hot path)."""

from __future__ import annotations

import time
from typing import TYPE_CHECKING, Any

import torch

from cutile_forge.kernels.attention_gqa import fused_attention_gqa
from cutile_forge.kernels.embedding_lmhead import (
    cutile_col_permute,
    cutile_embedding_lookup,
    cutile_ew_add,
    cutile_ew_mul,
    cutile_lm_head_fused,
)
from cutile_forge.kernels.gemm_dequant import fused_gemm_dequant
from cutile_forge.kernels.gemm_dequant_silu import fused_gemm_dequant_silu
from cutile_forge.kernels.layernorm_residual import fused_rmsnorm_residual
from cutile_forge.kernels.rmsnorm import cutile_rmsnorm
from cutile_forge.kernels.rope import apply_rope
from cutile_forge.runtime.native_kv_cache import NativePagedKVCache
from cutile_forge.runtime.native_stream import NativeStreamManager

if TYPE_CHECKING:
    from cutile_forge.engine.model import Mistral7B, MistralLayer
    from cutile_forge.engine.weight_loader import MistralLayerWeights, QuantLinearWeight


def _quant_linear_native(x: torch.Tensor, qw: QuantLinearWeight) -> torch.Tensor:
    x_in = cutile_col_permute(x, qw.perm) if qw.perm is not None else x
    return fused_gemm_dequant(x_in, qw.qweight, qw.scales, qw.zeros, group_size=qw.group_size)


class NativeModelRuntime:
    """End-to-end native CuTile inference runtime eliminating PyTorch compute ops from decode."""

    def __init__(
        self,
        model: Mistral7B,
        max_seq_len: int = 2048,
        block_size: int = 16,
        kv_dtype: str = "fp16",
        use_cuda_graph: bool = True,
    ) -> None:
        self.model = model
        self.config = model.config
        self.device = model.device
        self.use_cuda = self.device.type == "cuda" and torch.cuda.is_available()
        self.use_cuda_graph = use_cuda_graph and self.use_cuda

        self.stream_manager = NativeStreamManager(self.device)
        self.kv_cache = NativePagedKVCache(
            num_layers=self.config.num_hidden_layers,
            num_kv_heads=self.config.num_key_value_heads,
            head_dim=self.config.head_dim,
            block_size=block_size,
            max_seq_len=max_seq_len,
            kv_dtype=kv_dtype,
            device=self.device,
            stream_manager=self.stream_manager,
        )

        self._embed_w = model.weights.embed_tokens.to(self.device)
        self._norm_w = model.weights.norm.to(self.device)
        self._lm_head_w = model.weights.lm_head.to(self.device)
        self._cos = model.cos.to(self.device)
        self._sin = model.sin.to(self.device)
        self._cpu_layers: list[MistralLayer] | None = None
        if not self.use_cuda:
            from cutile_forge.engine.model import MistralLayer

            self._cpu_layers = [
                MistralLayer(i, self.config) for i in range(self.config.num_hidden_layers)
            ]
        self._cpu_kv: list[tuple[torch.Tensor, torch.Tensor] | None] = [
            None
        ] * self.config.num_hidden_layers

        # Optional CUDA-graph capture buffers for static decode QKV & MLP stages
        self._qkv_graphs: list[torch.cuda.CUDAGraph] = []
        self._mlp_graphs: list[torch.cuda.CUDAGraph] = []
        self._static_hidden = torch.zeros(
            (1, self.config.hidden_size), device=self.device, dtype=torch.float16
        )
        self._static_attn = [
            torch.zeros((1, self.config.hidden_size), device=self.device, dtype=torch.float16)
            for _ in range(self.config.num_hidden_layers)
        ]
        self._static_q: list[torch.Tensor] = [
            torch.zeros(
                (1, self.config.num_attention_heads, 1, self.config.head_dim),
                device=self.device,
                dtype=torch.float16,
            )
            for _ in range(self.config.num_hidden_layers)
        ]
        self._static_k: list[torch.Tensor] = [
            torch.zeros(
                (1, self.config.num_key_value_heads, 1, self.config.head_dim),
                device=self.device,
                dtype=torch.float16,
            )
            for _ in range(self.config.num_hidden_layers)
        ]
        self._static_v: list[torch.Tensor] = [
            torch.zeros(
                (1, self.config.num_key_value_heads, 1, self.config.head_dim),
                device=self.device,
                dtype=torch.float16,
            )
            for _ in range(self.config.num_hidden_layers)
        ]
        self._graphs_captured = False

    def reset(self) -> None:
        self.kv_cache.reset()
        self._cpu_kv = [None] * self.config.num_hidden_layers

    def _layer_on_device(self, idx: int) -> MistralLayerWeights:
        lw = self.model.weights.layers[idx]
        if self.use_cuda and lw.input_layernorm.device.type != "cuda":
            from cutile_forge.engine.weight_loader import MistralLayerWeights

            return MistralLayerWeights(
                layer_idx=lw.layer_idx,
                input_layernorm=lw.input_layernorm.to(self.device, non_blocking=True),
                post_attention_layernorm=lw.post_attention_layernorm.to(
                    self.device, non_blocking=True
                ),
                q_proj=lw.q_proj.to(self.device),
                k_proj=lw.k_proj.to(self.device),
                v_proj=lw.v_proj.to(self.device),
                o_proj=lw.o_proj.to(self.device),
                gate_proj=lw.gate_proj.to(self.device),
                up_proj=lw.up_proj.to(self.device),
                down_proj=lw.down_proj.to(self.device),
            )
        return lw

    def _capture_decode_graphs(self) -> None:
        if self._graphs_captured or not self.use_cuda_graph:
            return
        if not all(lw.input_layernorm.device.type == "cuda" for lw in self.model.weights.layers):
            return
        cfg = self.config
        n_layers = cfg.num_hidden_layers

        # Warmup once on current stream before CUDA graph capture
        for idx in range(n_layers):
            lw = self.model.weights.layers[idx]
            xn = cutile_rmsnorm(self._static_hidden, lw.input_layernorm, eps=cfg.rms_norm_eps)
            _ = _quant_linear_native(xn, lw.q_proj)
            _ = _quant_linear_native(xn, lw.k_proj)
            _ = _quant_linear_native(xn, lw.v_proj)
            op = _quant_linear_native(self._static_attn[idx], lw.o_proj)
            pn, _ = fused_rmsnorm_residual(
                op, self._static_hidden, lw.post_attention_layernorm, eps=cfg.rms_norm_eps
            )
            gi = cutile_col_permute(pn, lw.gate_proj.perm) if lw.gate_proj.perm is not None else pn
            ga = fused_gemm_dequant_silu(
                gi,
                lw.gate_proj.qweight,
                lw.gate_proj.scales,
                lw.gate_proj.zeros,
                group_size=lw.gate_proj.group_size,
            )
            uv = _quant_linear_native(pn, lw.up_proj)
            _ = cutile_ew_add(
                self._static_hidden, _quant_linear_native(cutile_ew_mul(ga, uv), lw.down_proj)
            )
        torch.cuda.synchronize()

        self._qkv_graphs = []
        self._mlp_graphs = []
        for idx in range(n_layers):
            lw = self.model.weights.layers[idx]
            qg = torch.cuda.CUDAGraph()
            with torch.cuda.graph(qg):
                xn = cutile_rmsnorm(self._static_hidden, lw.input_layernorm, eps=cfg.rms_norm_eps)
                q = _quant_linear_native(xn, lw.q_proj).view(
                    1, cfg.num_attention_heads, 1, cfg.head_dim
                )
                k = _quant_linear_native(xn, lw.k_proj).view(
                    1, cfg.num_key_value_heads, 1, cfg.head_dim
                )
                v = _quant_linear_native(xn, lw.v_proj).view(
                    1, cfg.num_key_value_heads, 1, cfg.head_dim
                )
                self._static_q[idx].copy_(q)
                self._static_k[idx].copy_(k)
                self._static_v[idx].copy_(v)
            self._qkv_graphs.append(qg)

            mg = torch.cuda.CUDAGraph()
            with torch.cuda.graph(mg):
                op = _quant_linear_native(self._static_attn[idx], lw.o_proj)
                pn, rm = fused_rmsnorm_residual(
                    op, self._static_hidden, lw.post_attention_layernorm, eps=cfg.rms_norm_eps
                )
                gi = (
                    cutile_col_permute(pn, lw.gate_proj.perm)
                    if lw.gate_proj.perm is not None
                    else pn
                )
                ga = fused_gemm_dequant_silu(
                    gi,
                    lw.gate_proj.qweight,
                    lw.gate_proj.scales,
                    lw.gate_proj.zeros,
                    group_size=lw.gate_proj.group_size,
                )
                uv = _quant_linear_native(pn, lw.up_proj)
                ffn = _quant_linear_native(cutile_ew_mul(ga, uv), lw.down_proj)
                self._static_hidden.copy_(cutile_ew_add(rm, ffn))
            self._mlp_graphs.append(mg)

        self._graphs_captured = True

    def _forward_layer_native(
        self,
        idx: int,
        x_2d: torch.Tensor,
        pos_offset: int,
        seq_len: int,
    ) -> torch.Tensor:
        cfg = self.config
        lw = self._layer_on_device(idx)

        if not self.use_cuda:
            from cutile_forge.engine.model import _pytorch_rmsnorm, _quant_linear_pytorch

            assert self._cpu_layers is not None
            # CPU reference fallback for non-GPU CI runners
            prev_kv = self._cpu_kv[idx]
            k_prev = prev_kv[0] if prev_kv is not None else None
            v_prev = prev_kv[1] if prev_kv is not None else None
            xn = _pytorch_rmsnorm(x_2d, lw.input_layernorm, eps=cfg.rms_norm_eps)
            q = (
                _quant_linear_pytorch(xn, lw.q_proj)
                .view(1, seq_len, cfg.num_attention_heads, cfg.head_dim)
                .transpose(1, 2)
            )
            k = (
                _quant_linear_pytorch(xn, lw.k_proj)
                .view(1, seq_len, cfg.num_key_value_heads, cfg.head_dim)
                .transpose(1, 2)
            )
            v = (
                _quant_linear_pytorch(xn, lw.v_proj)
                .view(1, seq_len, cfg.num_key_value_heads, cfg.head_dim)
                .transpose(1, 2)
            )
            half_d = cfg.head_dim // 2
            c = (
                self._cos[pos_offset : pos_offset + seq_len, :half_d]
                .view(1, 1, seq_len, half_d)
                .float()
            )
            s = (
                self._sin[pos_offset : pos_offset + seq_len, :half_d]
                .view(1, 1, seq_len, half_d)
                .float()
            )
            q = torch.cat(
                (
                    q[..., :half_d].float() * c - q[..., half_d:].float() * s,
                    q[..., half_d:].float() * c + q[..., :half_d].float() * s,
                ),
                dim=-1,
            ).to(torch.float16)
            k = torch.cat(
                (
                    k[..., :half_d].float() * c - k[..., half_d:].float() * s,
                    k[..., half_d:].float() * c + k[..., :half_d].float() * s,
                ),
                dim=-1,
            ).to(torch.float16)
            kf = k if k_prev is None else torch.cat([k_prev, k], dim=2)
            vf = v if v_prev is None else torch.cat([v_prev, v], dim=2)
            self._cpu_kv[idx] = (kf, vf)
            kr = kf.repeat_interleave(cfg.gqa_ratio, dim=1)
            vr = vf.repeat_interleave(cfg.gqa_ratio, dim=1)
            att = torch.nn.functional.scaled_dot_product_attention(
                q, kr, vr, is_causal=(seq_len > 1)
            )
            att_2d = att.transpose(1, 2).contiguous().view(seq_len, cfg.hidden_size)
            op = _quant_linear_pytorch(att_2d, lw.o_proj)
            rm = (x_2d.float() + op.float()).to(torch.float16)
            pn = _pytorch_rmsnorm(rm, lw.post_attention_layernorm, eps=cfg.rms_norm_eps)
            gate = torch.nn.functional.silu(_quant_linear_pytorch(pn, lw.gate_proj).float()).to(
                torch.float16
            )
            up = _quant_linear_pytorch(pn, lw.up_proj)
            ffn = _quant_linear_pytorch(gate * up, lw.down_proj)
            return (rm.float() + ffn.float()).to(torch.float16)

        x_n = cutile_rmsnorm(x_2d, lw.input_layernorm, eps=cfg.rms_norm_eps)
        q = _quant_linear_native(x_n, lw.q_proj)
        k = _quant_linear_native(x_n, lw.k_proj)
        v = _quant_linear_native(x_n, lw.v_proj)

        q = q.view(1, seq_len, cfg.num_attention_heads, cfg.head_dim).transpose(1, 2)
        k = k.view(1, seq_len, cfg.num_key_value_heads, cfg.head_dim).transpose(1, 2)
        v = v.view(1, seq_len, cfg.num_key_value_heads, cfg.head_dim).transpose(1, 2)

        q = apply_rope(q, self._cos, self._sin, position_offset=pos_offset)
        k = apply_rope(k, self._cos, self._sin, position_offset=pos_offset)

        k_full, v_full = self.kv_cache.append_and_get(idx, k, v, start_pos=pos_offset)
        attn_out = fused_attention_gqa(q, k_full, v_full, causal=True)
        attn_2d = (
            attn_out.view(1, cfg.hidden_size)
            if seq_len == 1
            else attn_out.transpose(1, 2).contiguous().view(seq_len, cfg.hidden_size)
        )

        o_proj = _quant_linear_native(attn_2d, lw.o_proj)
        post_norm, residual_mid = fused_rmsnorm_residual(
            o_proj, x_2d, lw.post_attention_layernorm, eps=cfg.rms_norm_eps
        )
        g_in = (
            cutile_col_permute(post_norm, lw.gate_proj.perm)
            if lw.gate_proj.perm is not None
            else post_norm
        )
        gate_act = fused_gemm_dequant_silu(
            g_in,
            lw.gate_proj.qweight,
            lw.gate_proj.scales,
            lw.gate_proj.zeros,
            group_size=lw.gate_proj.group_size,
        )
        up_val = _quant_linear_native(post_norm, lw.up_proj)
        act_mul = cutile_ew_mul(gate_act, up_val)
        ffn_out = _quant_linear_native(act_mul, lw.down_proj)
        return cutile_ew_add(residual_mid, ffn_out)

    @torch.inference_mode()
    def prefill(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Prefill prompt `(1, S)` using native CuTile kernels and return greedy next token `(1, 1)`."""
        self.reset()
        ids = input_ids.to(self.device)
        seq_len = int(ids.shape[1])
        h = cutile_embedding_lookup(ids.view(seq_len), self._embed_w).view(
            seq_len, self.config.hidden_size
        )
        for idx in range(self.config.num_hidden_layers):
            h = self._forward_layer_native(idx, h, pos_offset=0, seq_len=seq_len)
        _, next_tok = cutile_lm_head_fused(
            h[-1:, :], self._norm_w, self._lm_head_w, eps=self.config.rms_norm_eps, top_k=1
        )
        return next_tok.view(1, 1)

    @torch.inference_mode()
    def decode_step(
        self,
        token_id: torch.Tensor,
        pos: int,
        use_graph: bool | None = None,
    ) -> torch.Tensor:
        """Execute a single autoregressive decode step (zero ATen compute ops on CUDA)."""
        cfg = self.config
        use_g = self.use_cuda_graph if use_graph is None else (use_graph and self.use_cuda)
        if use_g and not self._graphs_captured:
            self._capture_decode_graphs()

        ids = token_id if token_id.device == self.device else token_id.to(self.device)
        if self._graphs_captured and use_g:
            cutile_embedding_lookup(ids.view(1), self._embed_w, out=self._static_hidden)
            for idx in range(cfg.num_hidden_layers):
                self._qkv_graphs[idx].replay()
                q = apply_rope(self._static_q[idx], self._cos, self._sin, position_offset=pos)
                k = apply_rope(self._static_k[idx], self._cos, self._sin, position_offset=pos)
                v = self._static_v[idx]
                k_full, v_full = self.kv_cache.append_and_get(idx, k, v, start_pos=pos)
                attn_out = fused_attention_gqa(q, k_full, v_full, causal=True)
                self._static_attn[idx].copy_(attn_out.view(1, cfg.hidden_size))
                self._mlp_graphs[idx].replay()
            _, next_tok = cutile_lm_head_fused(
                self._static_hidden, self._norm_w, self._lm_head_w, eps=cfg.rms_norm_eps, top_k=1
            )
            return next_tok.view(1, 1)

        h = cutile_embedding_lookup(ids.view(1), self._embed_w).view(1, cfg.hidden_size)
        for idx in range(cfg.num_hidden_layers):
            h = self._forward_layer_native(idx, h, pos_offset=pos, seq_len=1)
        _, next_tok = cutile_lm_head_fused(
            h, self._norm_w, self._lm_head_w, eps=cfg.rms_norm_eps, top_k=1
        )
        return next_tok.view(1, 1)

    @torch.inference_mode()
    def generate(
        self,
        prompt_ids: list[int],
        max_new_tokens: int = 32,
    ) -> tuple[list[int], dict[str, Any]]:
        """Run full prefill + native decode loop returning `(tokens, telemetry)`."""
        t0 = time.perf_counter()
        prompt_t = torch.tensor([prompt_ids], dtype=torch.long, device=self.device)
        next_tok = self.prefill(prompt_t)
        if self.use_cuda:
            torch.cuda.synchronize()
        t_prefill = (time.perf_counter() - t0) * 1000.0

        generated: list[int] = [int(next_tok.item())]
        cur_pos = len(prompt_ids)
        t_dec0 = time.perf_counter()
        for _ in range(max_new_tokens - 1):
            next_tok = self.decode_step(next_tok, pos=cur_pos)
            cur_pos += 1
            generated.append(int(next_tok.item()))
        if self.use_cuda:
            torch.cuda.synchronize()
        t_decode = (time.perf_counter() - t_dec0) * 1000.0
        gen_n = max(1, len(generated) - 1)
        return generated, {
            "prompt_tokens": len(prompt_ids),
            "generated_tokens": len(generated),
            "prefill_ms": round(t_prefill, 2),
            "decode_ms": round(t_decode, 2),
            "decode_tok_s": round((gen_n / max(1e-6, t_decode)) * 1000.0, 2),
            "ms_per_token": round(t_decode / gen_n, 2),
        }
