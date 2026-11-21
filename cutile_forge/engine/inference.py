"""Prefill and decode inference loop with TFG dispatch and paged KV-cache."""

from __future__ import annotations

import os
import time
from collections.abc import Iterator
from dataclasses import dataclass

import torch
import torch.nn.functional as F

from cutile_forge.engine.kv_cache import PagedKVCache
from cutile_forge.engine.model import Mistral7B, _quant_linear_cutile
from cutile_forge.engine.radix_cache import RadixPrefixCache
from cutile_forge.kernels.attention_gqa import fused_attention_gqa
from cutile_forge.kernels.gemm_dequant_silu import fused_gemm_dequant_silu
from cutile_forge.kernels.layernorm_residual import fused_rmsnorm_residual
from cutile_forge.kernels.rmsnorm import cutile_rmsnorm
from cutile_forge.kernels.rope import apply_rope
from cutile_forge.kernels.sampling import sample_tokens
from cutile_forge.utils.memory import get_peak_vram_gb, reset_peak_vram


@dataclass
class GenerationStats:
    """Throughput, latency, and VRAM telemetry for a single generation run."""

    prompt_tokens: int
    generated_tokens: int
    prefill_ms: float
    decode_ms: float
    tokens_per_sec: float
    ms_per_token: float
    peak_vram_gb: float


class BatchedKVCache:
    """Lightweight batched `(B, H_kv, S_kv, D)` KV cache for coalesced GPU decode."""

    def __init__(self, num_layers: int) -> None:
        self.num_layers = num_layers
        self._k: list[torch.Tensor | None] = [None] * num_layers
        self._v: list[torch.Tensor | None] = [None] * num_layers

    def update(self, layer_idx: int, k_new: torch.Tensor, v_new: torch.Tensor) -> int:
        prev_k = self._k[layer_idx]
        prev_v = self._v[layer_idx]
        if prev_k is None or prev_v is None:
            self._k[layer_idx] = k_new
            self._v[layer_idx] = v_new
        else:
            self._k[layer_idx] = torch.cat([prev_k, k_new], dim=2)
            self._v[layer_idx] = torch.cat([prev_v, v_new], dim=2)
        cur = self._k[layer_idx]
        assert cur is not None
        return int(cur.shape[2])

    def get_kv(self, layer_idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        k = self._k[layer_idx]
        v = self._v[layer_idx]
        assert k is not None and v is not None
        return k, v


class InferenceEngine:
    """High-level prefill + decode engine over `Mistral7B` and `PagedKVCache`."""

    def __init__(
        self,
        model: Mistral7B,
        page_size: int = 16,
        kv_gpu_budget_bytes: int = 2 * 1024 * 1024 * 1024,
        target: str = "sm_120",
        use_cuda_graph: bool = True,
        enable_radix_cache: bool = True,
    ) -> None:
        os.environ.setdefault("CUTILE_ARCH", target)
        self.target = target
        self.model = model
        self.config = model.config
        self.use_cuda_graph = use_cuda_graph
        self.graph_fallback_count: int = 0
        self.enable_radix_cache = enable_radix_cache
        self.radix_cache = RadixPrefixCache(max_entries=64)
        self.kv_cache = PagedKVCache(
            num_layers=self.config.num_hidden_layers,
            num_kv_heads=self.config.num_key_value_heads,
            head_dim=self.config.head_dim,
            page_size=page_size,
            gpu_budget_bytes=kv_gpu_budget_bytes,
            device=model.device,
        )
        self.last_stats: GenerationStats | None = None
        self._decode_graph: torch.cuda.CUDAGraph | None = None
        self._qkv_graphs: list[torch.cuda.CUDAGraph] = []
        self._mlp_graphs: list[torch.cuda.CUDAGraph] = []
        self._embed_graph: torch.cuda.CUDAGraph | None = None
        self._head_graph: torch.cuda.CUDAGraph | None = None
        self._static_input_ids: torch.Tensor | None = None
        self._static_cos: torch.Tensor | None = None
        self._static_sin: torch.Tensor | None = None
        self._static_hidden: torch.Tensor | None = None
        self._static_q: list[torch.Tensor] = []
        self._static_k: list[torch.Tensor] = []
        self._static_v: list[torch.Tensor] = []
        self._static_attn: list[torch.Tensor] = []
        self._static_logits: torch.Tensor | None = None

    def _can_use_cuda_graph(self, backend: str) -> bool:
        if not self.use_cuda_graph or backend != "cutile" or self.model.device.type != "cuda":
            return False
        if not torch.cuda.is_available() or self.kv_cache.evictions > 0 or self.config.experts > 1:
            return False
        return all(lw.input_layernorm.device.type == "cuda" for lw in self.model.weights.layers)

    @torch.inference_mode()
    def _capture_decode_graph(self) -> None:
        """Capture static-shape decode subgraphs (embed, per-layer QKV/MLP, LM head)."""
        if self._decode_graph is not None:
            return
        dev = self.model.device
        cfg = self.config
        n_layers = cfg.num_hidden_layers

        self._static_input_ids = torch.zeros((1, 1), device=dev, dtype=torch.long)
        self._static_cos = torch.zeros((1, cfg.head_dim // 2), device=dev, dtype=torch.float16)
        self._static_sin = torch.zeros((1, cfg.head_dim // 2), device=dev, dtype=torch.float16)
        self._static_hidden = torch.zeros((1, cfg.hidden_size), device=dev, dtype=torch.float16)
        self._static_q = [
            torch.zeros(
                (1, cfg.num_attention_heads, 1, cfg.head_dim), device=dev, dtype=torch.float16
            )
            for _ in range(n_layers)
        ]
        self._static_k = [
            torch.zeros(
                (1, cfg.num_key_value_heads, 1, cfg.head_dim), device=dev, dtype=torch.float16
            )
            for _ in range(n_layers)
        ]
        self._static_v = [
            torch.zeros(
                (1, cfg.num_key_value_heads, 1, cfg.head_dim), device=dev, dtype=torch.float16
            )
            for _ in range(n_layers)
        ]
        self._static_attn = [
            torch.zeros((1, cfg.hidden_size), device=dev, dtype=torch.float16)
            for _ in range(n_layers)
        ]
        self._static_logits = torch.zeros((1, 1, cfg.vocab_size), device=dev, dtype=torch.float16)

        # Warmup once on default stream before capture
        h_w = F.embedding(self._static_input_ids, self.model.weights.embed_tokens).view(
            1, cfg.hidden_size
        )
        for idx in range(n_layers):
            lw = self.model.weights.layers[idx]
            x_n = cutile_rmsnorm(h_w, lw.input_layernorm, eps=cfg.rms_norm_eps)
            q_w = (
                _quant_linear_cutile(x_n, lw.q_proj)
                .view(1, 1, cfg.num_attention_heads, cfg.head_dim)
                .transpose(1, 2)
            )
            k_w = (
                _quant_linear_cutile(x_n, lw.k_proj)
                .view(1, 1, cfg.num_key_value_heads, cfg.head_dim)
                .transpose(1, 2)
            )
            _ = _quant_linear_cutile(x_n, lw.v_proj)
            _ = apply_rope(q_w, self._static_cos, self._static_sin, position_offset=0)
            _ = apply_rope(k_w, self._static_cos, self._static_sin, position_offset=0)
            o_w = _quant_linear_cutile(self._static_attn[idx], lw.o_proj)
            p_n, r_m = fused_rmsnorm_residual(
                o_w, h_w, lw.post_attention_layernorm, eps=cfg.rms_norm_eps
            )
            g_in = p_n[:, lw.gate_proj.perm] if lw.gate_proj.perm is not None else p_n
            g_a = fused_gemm_dequant_silu(
                g_in,
                lw.gate_proj.qweight,
                lw.gate_proj.scales,
                lw.gate_proj.zeros,
                group_size=lw.gate_proj.group_size,
            )
            u_v = _quant_linear_cutile(p_n, lw.up_proj)
            _ = r_m + _quant_linear_cutile(g_a * u_v, lw.down_proj)
        _ = torch.matmul(
            cutile_rmsnorm(h_w, self.model.weights.norm, eps=cfg.rms_norm_eps),
            self.model.weights.lm_head.t(),
        )
        torch.cuda.synchronize()

        self._embed_graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self._embed_graph):
            emb = F.embedding(self._static_input_ids, self.model.weights.embed_tokens).view(
                1, cfg.hidden_size
            )
            self._static_hidden.copy_(emb)

        self._qkv_graphs = []
        self._mlp_graphs = []
        for idx in range(n_layers):
            lw = self.model.weights.layers[idx]
            qg = torch.cuda.CUDAGraph()
            with torch.cuda.graph(qg):
                x_n = cutile_rmsnorm(self._static_hidden, lw.input_layernorm, eps=cfg.rms_norm_eps)
                q = (
                    _quant_linear_cutile(x_n, lw.q_proj)
                    .view(1, 1, cfg.num_attention_heads, cfg.head_dim)
                    .transpose(1, 2)
                )
                k = (
                    _quant_linear_cutile(x_n, lw.k_proj)
                    .view(1, 1, cfg.num_key_value_heads, cfg.head_dim)
                    .transpose(1, 2)
                )
                v = (
                    _quant_linear_cutile(x_n, lw.v_proj)
                    .view(1, 1, cfg.num_key_value_heads, cfg.head_dim)
                    .transpose(1, 2)
                )
                q = apply_rope(q, self._static_cos, self._static_sin, position_offset=0)
                k = apply_rope(k, self._static_cos, self._static_sin, position_offset=0)
                self._static_q[idx].copy_(q)
                self._static_k[idx].copy_(k)
                self._static_v[idx].copy_(v)
            self._qkv_graphs.append(qg)

            mg = torch.cuda.CUDAGraph()
            with torch.cuda.graph(mg):
                o_proj = _quant_linear_cutile(self._static_attn[idx], lw.o_proj)
                post_norm, residual_mid = fused_rmsnorm_residual(
                    o_proj, self._static_hidden, lw.post_attention_layernorm, eps=cfg.rms_norm_eps
                )
                gate_in = (
                    post_norm[:, lw.gate_proj.perm] if lw.gate_proj.perm is not None else post_norm
                )
                gate_act = fused_gemm_dequant_silu(
                    gate_in,
                    lw.gate_proj.qweight,
                    lw.gate_proj.scales,
                    lw.gate_proj.zeros,
                    group_size=lw.gate_proj.group_size,
                )
                up_val = _quant_linear_cutile(post_norm, lw.up_proj)
                ffn_out = _quant_linear_cutile(gate_act * up_val, lw.down_proj)
                self._static_hidden.copy_(residual_mid + ffn_out)
            self._mlp_graphs.append(mg)

        self._head_graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self._head_graph):
            normed = cutile_rmsnorm(
                self._static_hidden, self.model.weights.norm, eps=cfg.rms_norm_eps
            )
            logits = torch.matmul(normed, self.model.weights.lm_head.t())
            self._static_logits.copy_(logits.view(1, 1, cfg.vocab_size))

        self._decode_graph = self._head_graph

    @torch.inference_mode()
    def decode_step(
        self,
        token_ids: torch.Tensor,
        pos: int,
        backend: str = "cutile",
    ) -> torch.Tensor:
        """Execute a single `(1, 1)` decode step, using CUDA Graphs when eligible."""
        if (
            token_ids.shape == (1, 1)
            and self._can_use_cuda_graph(backend)
            and pos < self.model.cos.shape[0]
        ):
            if self._decode_graph is None:
                try:
                    self._capture_decode_graph()
                except RuntimeError as exc:
                    self.use_cuda_graph = False
                    self.graph_fallback_count += 1
                    self._decode_graph = None
                    import logging

                    logging.getLogger(__name__).warning(
                        "CUDA Graph capture failed (%s); falling back to eager decode.", exc
                    )
                    return self.model.forward(
                        token_ids, kv_cache=self.kv_cache, pos_offset=pos, backend=backend
                    )
            assert self._static_input_ids is not None
            assert self._static_cos is not None
            assert self._static_sin is not None
            assert self._embed_graph is not None
            assert self._head_graph is not None
            assert self._static_logits is not None

            self._static_input_ids.copy_(token_ids)
            self._static_cos.copy_(self.model.cos[pos : pos + 1])
            self._static_sin.copy_(self.model.sin[pos : pos + 1])
            self._embed_graph.replay()

            for idx in range(self.config.num_hidden_layers):
                self._qkv_graphs[idx].replay()
                self.kv_cache.update(idx, self._static_k[idx], self._static_v[idx])
                k_full, v_full = self.kv_cache.get_kv(idx)
                attn_out = fused_attention_gqa(
                    self._static_q[idx],
                    k_full,
                    v_full,
                    causal=True,
                    sliding_window=int(self.config.sliding_window or 0),
                )
                self._static_attn[idx].copy_(
                    attn_out.transpose(1, 2).contiguous().view(1, self.config.hidden_size)
                )
                self._mlp_graphs[idx].replay()

            self._head_graph.replay()
            return self._static_logits

        return self.model.forward(
            token_ids, kv_cache=self.kv_cache, pos_offset=pos, backend=backend
        )

    @torch.inference_mode()
    def prefill(
        self,
        prompt_ids: torch.Tensor,
        backend: str = "cutile",
    ) -> torch.Tensor:
        """Run prefill for `(1, S)` prompt with RadixPrefixCache reuse."""
        if prompt_ids.ndim == 1:
            prompt_ids = prompt_ids.unsqueeze(0)
        prompt_ids = prompt_ids.to(self.model.device)
        self.kv_cache.reset()

        tok_list: list[int] = [int(t) for t in prompt_ids[0].tolist()]
        if self.enable_radix_cache and len(tok_list) > 0:
            matched_len, cached_kv, cached_logits = self.radix_cache.match_prefix(tok_list)
            if matched_len > 0 and cached_kv is not None:
                for l_idx, (k_c, v_c) in cached_kv.items():
                    self.kv_cache.update(
                        l_idx, k_c.to(self.model.device), v_c.to(self.model.device)
                    )
                if matched_len == len(tok_list) and cached_logits is not None:
                    return cached_logits.to(self.model.device)
                # Run only the uncached suffix tokens
                suffix = prompt_ids[:, matched_len:]
                logits = self.model.forward(
                    suffix,
                    kv_cache=self.kv_cache,
                    pos_offset=matched_len,
                    backend=backend,
                    last_token_only=True,
                )
                kv_snap = {
                    l_idx: self.kv_cache.get_kv(l_idx)
                    for l_idx in range(self.config.num_hidden_layers)
                }
                self.radix_cache.insert(tok_list, kv_snap, logits)
                return logits

        logits = self.model.forward(
            prompt_ids,
            kv_cache=self.kv_cache,
            pos_offset=0,
            backend=backend,
            last_token_only=True,
        )
        if self.enable_radix_cache and len(tok_list) > 0 and self.kv_cache.evictions == 0:
            kv_snap = {
                l_idx: self.kv_cache.get_kv(l_idx) for l_idx in range(self.config.num_hidden_layers)
            }
            self.radix_cache.insert(tok_list, kv_snap, logits)
        return logits

    @torch.inference_mode()
    def decode_batch(
        self,
        token_ids_batch: torch.Tensor,
        batch_kv: BatchedKVCache,
        pos_offset: int = 0,
        backend: str = "cutile",
    ) -> torch.Tensor:
        """Run a single batched `(B, 1)` decode step on GPU with `BatchedKVCache`."""
        return self.model.forward(
            token_ids_batch,
            kv_cache=batch_kv,  # type: ignore[arg-type]
            pos_offset=pos_offset,
            backend=backend,
            last_token_only=True,
        )

    @torch.inference_mode()
    def generate_batch(
        self,
        prompt_ids_list: list[torch.Tensor],
        max_new_tokens: int = 8,
        temperature: float = 0.0,
        top_k: int = 50,
        top_p: float = 0.9,
        backend: str = "cutile",
    ) -> list[list[int]]:
        """Coalesce multiple prompts into a single batched prefill + decode pass."""
        if len(prompt_ids_list) == 1:
            return [
                self.generate_ids(
                    prompt_ids_list[0],
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    top_k=top_k,
                    top_p=top_p,
                    backend=backend,
                )
            ]

        bsz = len(prompt_ids_list)
        seqs = [p.view(-1).to(self.model.device, dtype=torch.long) for p in prompt_ids_list]
        max_len = max(int(s.numel()) for s in seqs)
        pad_id = 1
        batch_in = torch.full((bsz, max_len), pad_id, device=self.model.device, dtype=torch.long)
        for i, s in enumerate(seqs):
            batch_in[i, max_len - int(s.numel()) :] = s

        batch_kv = BatchedKVCache(self.config.num_hidden_layers)
        logits = self.model.forward(
            batch_in,
            kv_cache=batch_kv,  # type: ignore[arg-type]
            pos_offset=0,
            backend=backend,
            last_token_only=True,
        )
        next_toks = sample_tokens(
            logits[:, -1, :], temperature=temperature, top_k=top_k, top_p=top_p
        )
        outputs: list[list[int]] = [[int(next_toks[i].item())] for i in range(bsz)]
        pos = max_len

        for _ in range(max(0, max_new_tokens - 1)):
            cur_in = next_toks.view(bsz, 1)
            logits = self.decode_batch(cur_in, batch_kv=batch_kv, pos_offset=pos, backend=backend)
            next_toks = sample_tokens(
                logits[:, -1, :], temperature=temperature, top_k=top_k, top_p=top_p
            )
            for i in range(bsz):
                outputs[i].append(int(next_toks[i].item()))
            pos += 1

        return outputs

    @torch.inference_mode()
    def generate_ids(
        self,
        prompt_ids: torch.Tensor,
        max_new_tokens: int = 64,
        temperature: float = 0.0,
        top_k: int = 50,
        top_p: float = 0.9,
        eos_token_id: int | None = None,
        backend: str = "cutile",
    ) -> list[int]:
        """Generate token IDs autoregressively from `prompt_ids` of shape `(1, S)`."""
        if prompt_ids.ndim == 1:
            prompt_ids = prompt_ids.unsqueeze(0)
        prompt_ids = prompt_ids.to(self.model.device)
        reset_peak_vram()

        t_pre0 = time.perf_counter()
        logits = self.prefill(prompt_ids, backend=backend)
        if self.model.device.type == "cuda":
            torch.cuda.synchronize()
        prefill_ms = (time.perf_counter() - t_pre0) * 1000.0

        next_tok = sample_tokens(
            logits[:, -1, :], temperature=temperature, top_k=top_k, top_p=top_p
        )
        generated: list[int] = [int(next_tok.item())]
        pos = int(prompt_ids.shape[1])

        t_dec0 = time.perf_counter()
        for _ in range(max(0, max_new_tokens - 1)):
            if eos_token_id is not None and generated[-1] == eos_token_id:
                break
            cur_in = next_tok.view(1, 1)
            logits = self.decode_step(cur_in, pos=pos, backend=backend)
            next_tok = sample_tokens(
                logits[:, -1, :], temperature=temperature, top_k=top_k, top_p=top_p
            )
            generated.append(int(next_tok.item()))
            pos += 1

        if self.model.device.type == "cuda":
            torch.cuda.synchronize()
        decode_ms = (time.perf_counter() - t_dec0) * 1000.0

        n_dec = max(1, len(generated) - 1) if len(generated) > 1 else 1
        ms_per_tok = decode_ms / n_dec if decode_ms > 0 else prefill_ms
        tps = 1000.0 / ms_per_tok if ms_per_tok > 0 else 0.0
        self.last_stats = GenerationStats(
            prompt_tokens=int(prompt_ids.shape[1]),
            generated_tokens=len(generated),
            prefill_ms=prefill_ms,
            decode_ms=decode_ms,
            tokens_per_sec=tps,
            ms_per_token=ms_per_tok,
            peak_vram_gb=get_peak_vram_gb(),
        )
        return generated

    @torch.inference_mode()
    def stream_ids(
        self,
        prompt_ids: torch.Tensor,
        max_new_tokens: int = 64,
        temperature: float = 0.0,
        top_k: int = 50,
        top_p: float = 0.9,
        eos_token_id: int | None = None,
        backend: str = "cutile",
    ) -> Iterator[int]:
        """Yield generated token IDs one by one."""
        if prompt_ids.ndim == 1:
            prompt_ids = prompt_ids.unsqueeze(0)
        prompt_ids = prompt_ids.to(self.model.device)
        self.kv_cache.reset()

        logits = self.model.forward(
            prompt_ids, kv_cache=self.kv_cache, pos_offset=0, backend=backend
        )
        next_tok = sample_tokens(
            logits[:, -1, :], temperature=temperature, top_k=top_k, top_p=top_p
        )
        tok_id = int(next_tok.item())
        yield tok_id
        if eos_token_id is not None and tok_id == eos_token_id:
            return

        pos = int(prompt_ids.shape[1])
        for _ in range(max(0, max_new_tokens - 1)):
            logits = self.decode_step(next_tok.view(1, 1), pos=pos, backend=backend)
            next_tok = sample_tokens(
                logits[:, -1, :], temperature=temperature, top_k=top_k, top_p=top_p
            )
            tok_id = int(next_tok.item())
            yield tok_id
            pos += 1
            if eos_token_id is not None and tok_id == eos_token_id:
                break
