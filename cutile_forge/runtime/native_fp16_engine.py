"""Native 100% CuTile FP16 decoder runtime with full-step CUDA Graph capture & NativePagedKVCache."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import torch

from cutile_forge.cuda_tile_init import ct
from cutile_forge.kernels.attention_gqa import (
    fused_attention_gqa,
    fused_attention_gqa_decode_pos,
)
from cutile_forge.kernels.embedding_lmhead import (
    cutile_embedding_lookup,
    cutile_lm_head_fused,
    cutile_record_and_advance_step,
)
from cutile_forge.kernels.gemm_dequant import cutile_gemm_fp16
from cutile_forge.kernels.gemm_dequant_silu import fused_swiglu_fp16
from cutile_forge.kernels.layernorm_residual import fused_rmsnorm_residual
from cutile_forge.kernels.rmsnorm import cutile_rmsnorm
from cutile_forge.kernels.rope import apply_rope, apply_rope_qk_kv_write_pos
from cutile_forge.runtime.native_engine import DEFAULT_CONTEXT_BUCKETS, GraphStrategy, _next_pow2
from cutile_forge.runtime.native_kv_cache import NativePagedKVCache
from cutile_forge.runtime.native_stream import NativeStreamManager

if TYPE_CHECKING:
    from cutile_forge.models.model_graph import UnifiedModelGraph


@dataclass
class _PackedFP16Layer:
    """Contiguous FP16 layer weights with fused QKV weight & bias concatenation."""

    in_norm_w: torch.Tensor
    post_norm_w: torch.Tensor
    w_qkv: torch.Tensor
    b_qkv: torch.Tensor | None
    w_o: torch.Tensor
    w_gate: torch.Tensor
    w_up: torch.Tensor
    w_down: torch.Tensor


@dataclass
class _DecodeFP16Workspace:
    """Static preallocated buffers for zero-allocation CUDA Graph capture at fixed `batch_size`."""

    batch_size: int
    token_id: torch.Tensor
    pos_buf: torch.Tensor
    step_buf: torch.Tensor
    history_buf: torch.Tensor
    hidden: torch.Tensor
    norm: torch.Tensor
    qkv: torch.Tensor
    q_rot: torch.Tensor
    attn_out: torch.Tensor
    o_out: torch.Tensor
    res_mid: torch.Tensor
    swiglu_out: torch.Tensor
    splitk_part: torch.Tensor
    splitk_part2: torch.Tensor
    lm_scratch: dict[str, torch.Tensor]
    graphs_by_bucket: dict[int, torch.cuda.CUDAGraph]


class NativeFP16DecoderRuntime:
    """End-to-end native CuTile FP16 decoder runtime (Qwen2.5 / Llama / Mistral FP16)."""

    def __init__(
        self,
        graph: UnifiedModelGraph,
        weights: dict[str, Any],
        max_seq_len: int = 2048,
        block_size: int = 16,
        kv_dtype: str = "fp16",
        use_cuda_graph: bool = True,
        graph_strategy: GraphStrategy = "bucketed",
        context_buckets: tuple[int, ...] = DEFAULT_CONTEXT_BUCKETS,
        max_batch_size: int = 1,
    ) -> None:
        from cutile_forge.models.model_graph import (
            DenseTransformerBlock,
            compute_scaled_rope_freqs,
        )

        if not graph.blocks or not all(isinstance(b, DenseTransformerBlock) for b in graph.blocks):
            raise TypeError("NativeFP16DecoderRuntime supports DenseTransformerBlock graphs only")
        b0 = graph.blocks[0]
        assert isinstance(b0, DenseTransformerBlock)
        if b0.norm_type != "rmsnorm" or b0.activation != "silu":
            raise ValueError(f"Unsupported norm/activation: {b0.norm_type}/{b0.activation}")

        self.graph = graph
        self.num_layers = len(graph.blocks)
        self.num_attention_heads = b0.num_attention_heads
        self.num_key_value_heads = b0.num_key_value_heads
        self.head_dim = b0.head_dim
        self.hidden_size = self.num_attention_heads * self.head_dim
        self.rms_norm_eps = b0.rms_norm_eps
        self.sliding_window = int(b0.sliding_window or 0)
        self.max_seq_len = max_seq_len
        self.block_size = block_size
        self.max_batch_size = max(1, max_batch_size)
        self.graph_strategy: GraphStrategy = graph_strategy
        self.graph_fallback_count: int = 0

        self._embed_w = weights["embed_tokens"].contiguous()
        self.device = self._embed_w.device
        self.use_cuda = self.device.type == "cuda" and torch.cuda.is_available()
        self.use_cuda_graph = use_cuda_graph and self.use_cuda

        valid_buckets = tuple(sorted({min(b, max_seq_len) for b in context_buckets if b > 0}))
        if not valid_buckets or valid_buckets[-1] < max_seq_len:
            valid_buckets = (*valid_buckets, max_seq_len)
        self.context_buckets: tuple[int, ...] = valid_buckets

        self._norm_w = weights["norm_w"].contiguous()
        self._lm_head_rows = weights["lm_head"].t().contiguous()
        self.vocab_size = int(self._lm_head_rows.shape[0])

        self._packed_layers: list[_PackedFP16Layer] = []
        for blk, bw in zip(graph.blocks, weights["blocks"], strict=True):
            assert isinstance(blk, DenseTransformerBlock)
            w_qkv = torch.cat([bw["wq"], bw["wk"], bw["wv"]], dim=1).contiguous()
            b_qkv = (
                torch.cat([bw["bq"], bw["bk"], bw["bv"]], dim=0).contiguous()
                if blk.qkv_bias
                else None
            )
            self._packed_layers.append(
                _PackedFP16Layer(
                    in_norm_w=bw["in_norm_w"].contiguous(),
                    post_norm_w=bw["post_norm_w"].contiguous(),
                    w_qkv=w_qkv,
                    b_qkv=b_qkv,
                    w_o=bw["wo"].contiguous(),
                    w_gate=bw["w_gate"].contiguous(),
                    w_up=bw["w_up"].contiguous(),
                    w_down=bw["w_down"].contiguous(),
                )
            )
        self.intermediate_size = int(self._packed_layers[0].w_gate.shape[1])

        self._cos, self._sin = compute_scaled_rope_freqs(
            max_seq_len,
            self.head_dim,
            theta=b0.rope_theta,
            rope_scaling=b0.rope_scaling,
            device=self.device,
            dtype=torch.float16,
        )

        self.stream_manager = NativeStreamManager(self.device)
        self.kv_cache = NativePagedKVCache(
            num_layers=self.num_layers,
            num_kv_heads=self.num_key_value_heads,
            head_dim=self.head_dim,
            block_size=block_size,
            max_seq_len=max_seq_len,
            max_batch_size=self.max_batch_size,
            kv_dtype=kv_dtype,
            device=self.device,
            stream_manager=self.stream_manager,
        )
        self._workspaces: dict[int, _DecodeFP16Workspace] = {}

    def reset(self) -> None:
        self.kv_cache.reset()

    def _select_bucket(self, pos: int) -> int:
        if self.graph_strategy == "device_masked":
            return self.max_seq_len
        needed = pos + 1
        for b in self.context_buckets:
            if needed <= b:
                return b
        return self.max_seq_len

    def _get_or_create_workspace(self, batch_size: int) -> _DecodeFP16Workspace:
        if batch_size in self._workspaces:
            return self._workspaces[batch_size]
        self.kv_cache.ensure_batch_capacity(batch_size)
        qkv_dim = (self.num_attention_heads + 2 * self.num_key_value_heads) * self.head_dim
        num_vt = ct.cdiv(self.vocab_size, 256)

        token_id = torch.zeros((batch_size, 1), device=self.device, dtype=torch.int32)
        pos_buf = torch.zeros((batch_size,), device=self.device, dtype=torch.int32)
        step_buf = torch.zeros((1,), device=self.device, dtype=torch.int32)
        history_buf = torch.zeros(
            (self.max_seq_len, _next_pow2(batch_size)), device=self.device, dtype=torch.int32
        )
        hidden = torch.zeros(
            (batch_size, self.hidden_size), device=self.device, dtype=torch.float16
        )
        norm = torch.zeros((batch_size, self.hidden_size), device=self.device, dtype=torch.float16)
        qkv = torch.zeros((batch_size, qkv_dim), device=self.device, dtype=torch.float16)
        q_rot = torch.zeros(
            (batch_size, self.num_attention_heads, 1, self.head_dim),
            device=self.device,
            dtype=torch.float16,
        )
        attn_out = torch.zeros(
            (batch_size, self.num_attention_heads, 1, self.head_dim),
            device=self.device,
            dtype=torch.float16,
        )
        o_out = torch.zeros((batch_size, self.hidden_size), device=self.device, dtype=torch.float16)
        res_mid = torch.zeros(
            (batch_size, self.hidden_size), device=self.device, dtype=torch.float16
        )
        swiglu_out = torch.zeros(
            (batch_size, self.intermediate_size), device=self.device, dtype=torch.float16
        )
        max_sk_n = max(self.hidden_size, self.intermediate_size, qkv_dim)
        splitk_part = torch.empty(
            (4 * batch_size * max_sk_n,), device=self.device, dtype=torch.float32
        )
        splitk_part2 = torch.empty(
            (4 * batch_size * self.intermediate_size,), device=self.device, dtype=torch.float32
        )
        lm_scratch = {
            "norm_out": torch.zeros(
                (batch_size, self.hidden_size), device=self.device, dtype=torch.float16
            ),
            "part_val": torch.zeros((batch_size, num_vt), device=self.device, dtype=torch.float32),
            "part_idx": torch.zeros((batch_size, num_vt), device=self.device, dtype=torch.int32),
            "out_val": torch.zeros((batch_size, 1), device=self.device, dtype=torch.float32),
            "out_idx": token_id,
        }
        ws = _DecodeFP16Workspace(
            batch_size=batch_size,
            token_id=token_id,
            pos_buf=pos_buf,
            step_buf=step_buf,
            history_buf=history_buf,
            hidden=hidden,
            norm=norm,
            qkv=qkv,
            q_rot=q_rot,
            attn_out=attn_out,
            o_out=o_out,
            res_mid=res_mid,
            swiglu_out=swiglu_out,
            splitk_part=splitk_part,
            splitk_part2=splitk_part2,
            lm_scratch=lm_scratch,
            graphs_by_bucket={},
        )
        self._workspaces[batch_size] = ws
        return ws

    def _execute_decode_step_in_workspace(self, ws: _DecodeFP16Workspace, bucket_len: int) -> None:
        """Execute 1 complete FP16 decode step inside `ws` using only static addresses."""
        bsz = ws.batch_size
        dyn_loop = self.graph_strategy == "bucketed"

        cutile_embedding_lookup(ws.token_id.view(bsz), self._embed_w, out=ws.hidden)

        for idx, pl in enumerate(self._packed_layers):
            cutile_rmsnorm(ws.hidden, pl.in_norm_w, eps=self.rms_norm_eps, out=ws.norm)
            cutile_gemm_fp16(ws.norm, pl.w_qkv, out=ws.qkv, part_buf=ws.splitk_part, bias=pl.b_qkv)
            apply_rope_qk_kv_write_pos(
                ws.qkv,
                self._cos,
                self._sin,
                ws.pos_buf,
                self.kv_cache.block_tables[idx][:bsz],
                ws.q_rot,
                self.kv_cache.k_pools[idx][:bsz],
                self.kv_cache.v_pools[idx][:bsz],
                n_q_heads=self.num_attention_heads,
                n_kv_heads=self.num_key_value_heads,
                head_dim=self.head_dim,
                block_size=self.block_size,
            )
            fused_attention_gqa_decode_pos(
                ws.q_rot,
                self.kv_cache.k_pools[idx][:bsz],
                self.kv_cache.v_pools[idx][:bsz],
                ws.pos_buf,
                max_kv_len=bucket_len,
                out=ws.attn_out,
                dynamic_loop=dyn_loop,
                sliding_window=self.sliding_window,
            )
            cutile_gemm_fp16(
                ws.attn_out.view(bsz, self.hidden_size),
                pl.w_o,
                out=ws.o_out,
                part_buf=ws.splitk_part,
            )
            fused_rmsnorm_residual(
                ws.o_out,
                ws.hidden,
                pl.post_norm_w,
                eps=self.rms_norm_eps,
                out_norm=ws.norm,
                out_res=ws.res_mid,
            )
            fused_swiglu_fp16(
                ws.norm,
                pl.w_gate,
                pl.w_up,
                out=ws.swiglu_out,
                part_g_buf=ws.splitk_part,
                part_u_buf=ws.splitk_part2,
            )
            cutile_gemm_fp16(
                ws.swiglu_out,
                pl.w_down,
                out=ws.hidden,
                part_buf=ws.splitk_part,
                residual=ws.res_mid,
            )

        cutile_lm_head_fused(
            ws.hidden,
            self._norm_w,
            self._lm_head_rows,
            eps=self.rms_norm_eps,
            top_k=1,
            scratch=ws.lm_scratch,
        )
        cutile_record_and_advance_step(
            ws.token_id,
            ws.pos_buf,
            ws.step_buf,
            ws.history_buf,
            bsz=bsz,
        )

    def _capture_full_step_graph(
        self, ws: _DecodeFP16Workspace, bucket_len: int
    ) -> torch.cuda.CUDAGraph:
        if bucket_len in ws.graphs_by_bucket:
            return ws.graphs_by_bucket[bucket_len]

        saved_tok = ws.token_id.clone()
        saved_pos = ws.pos_buf.clone()
        saved_step = ws.step_buf.clone()
        scratch_slot = min(bucket_len - 1, self.max_seq_len - 1)
        saved_k = [kp[: ws.batch_size, :, scratch_slot, :].clone() for kp in self.kv_cache.k_pools]
        saved_v = [vp[: ws.batch_size, :, scratch_slot, :].clone() for vp in self.kv_cache.v_pools]

        ws.pos_buf.fill_(scratch_slot)
        ws.step_buf.fill_(0)
        self._execute_decode_step_in_workspace(ws, bucket_len)
        torch.cuda.synchronize()

        ws.pos_buf.fill_(scratch_slot)
        ws.step_buf.fill_(0)
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            self._execute_decode_step_in_workspace(ws, bucket_len)
        torch.cuda.synchronize()

        for idx in range(self.num_layers):
            self.kv_cache.k_pools[idx][: ws.batch_size, :, scratch_slot, :].copy_(saved_k[idx])
            self.kv_cache.v_pools[idx][: ws.batch_size, :, scratch_slot, :].copy_(saved_v[idx])
        ws.token_id.copy_(saved_tok)
        ws.pos_buf.copy_(saved_pos)
        ws.step_buf.copy_(saved_step)

        ws.graphs_by_bucket[bucket_len] = g
        return g

    def _forward_prefill_layer(
        self, idx: int, x_2d: torch.Tensor, pos_offset: int, seq_len: int, batch_size: int
    ) -> torch.Tensor:
        pl = self._packed_layers[idx]
        q_dim = self.num_attention_heads * self.head_dim
        kv_dim = self.num_key_value_heads * self.head_dim

        x_n = cutile_rmsnorm(x_2d, pl.in_norm_w, eps=self.rms_norm_eps)
        qkv = cutile_gemm_fp16(x_n, pl.w_qkv, bias=pl.b_qkv)
        q = (
            qkv[:, :q_dim]
            .reshape(batch_size, seq_len, self.num_attention_heads, self.head_dim)
            .transpose(1, 2)
        )
        k = (
            qkv[:, q_dim : q_dim + kv_dim]
            .reshape(batch_size, seq_len, self.num_key_value_heads, self.head_dim)
            .transpose(1, 2)
        )
        v = (
            qkv[:, q_dim + kv_dim :]
            .reshape(batch_size, seq_len, self.num_key_value_heads, self.head_dim)
            .transpose(1, 2)
        )

        q = apply_rope(q, self._cos, self._sin, position_offset=pos_offset)
        k = apply_rope(k, self._cos, self._sin, position_offset=pos_offset)
        k_full, v_full = self.kv_cache.append_and_get(idx, k, v, start_pos=pos_offset)
        attn_out = fused_attention_gqa(
            q, k_full, v_full, causal=True, sliding_window=self.sliding_window
        )
        attn_2d = (
            attn_out.view(batch_size, self.hidden_size)
            if seq_len == 1
            else attn_out.transpose(1, 2).contiguous().view(batch_size * seq_len, self.hidden_size)
        )
        o_proj = cutile_gemm_fp16(attn_2d, pl.w_o)
        post_norm, res_mid = fused_rmsnorm_residual(
            o_proj, x_2d, pl.post_norm_w, eps=self.rms_norm_eps
        )
        act_mul = fused_swiglu_fp16(post_norm, pl.w_gate, pl.w_up)
        return cutile_gemm_fp16(act_mul, pl.w_down, residual=res_mid)

    @torch.inference_mode()
    def prefill(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Prefill `(B, S)` prompt using native CuTile kernels and return greedy `(B, 1)` token."""
        self.reset()
        ids = input_ids.to(self.device)
        if ids.ndim == 1:
            ids = ids.unsqueeze(0)
        bsz, seq_len = int(ids.shape[0]), int(ids.shape[1])
        self.kv_cache.ensure_batch_capacity(bsz)
        h = cutile_embedding_lookup(ids.reshape(bsz * seq_len), self._embed_w).view(
            bsz * seq_len, self.hidden_size
        )
        for idx in range(self.num_layers):
            h = self._forward_prefill_layer(idx, h, pos_offset=0, seq_len=seq_len, batch_size=bsz)
        h_last = h.view(bsz, seq_len, self.hidden_size)[:, -1, :].contiguous()
        _, next_tok = cutile_lm_head_fused(
            h_last, self._norm_w, self._lm_head_rows, eps=self.rms_norm_eps, top_k=1
        )
        ws = self._get_or_create_workspace(bsz)
        ws.token_id.copy_(next_tok.view(bsz, 1).to(torch.int32))
        ws.pos_buf.fill_(seq_len)
        ws.step_buf.fill_(0)
        return next_tok.view(bsz, 1)

    @torch.inference_mode()
    def decode_step(
        self,
        token_id: torch.Tensor | None = None,
        pos: int | None = None,
        use_graph: bool | None = None,
        sync_pos: bool = True,
    ) -> torch.Tensor:
        """Execute a single autoregressive FP16 decode step (zero ATen compute ops on CUDA)."""
        use_g = self.use_cuda_graph if use_graph is None else (use_graph and self.use_cuda)
        bsz = 1 if token_id is None else max(1, int(token_id.shape[0]) if token_id.ndim >= 2 else 1)

        if use_g and self.use_cuda:
            ws = self._get_or_create_workspace(bsz)
            cur_pos = self.kv_cache.seq_lens[0] if pos is None else pos
            bucket_len = self._select_bucket(cur_pos)
            try:
                graph = self._capture_full_step_graph(ws, bucket_len)
            except RuntimeError:
                self.use_cuda_graph = False
                self.graph_fallback_count += 1
            else:
                if token_id is not None and token_id.data_ptr() != ws.token_id.data_ptr():
                    ws.token_id.copy_(token_id.view(bsz, 1))
                if sync_pos and pos is not None:
                    ws.pos_buf.fill_(pos)
                graph.replay()
                self.kv_cache.seq_lens = [cur_pos + 1] * self.num_layers
                return ws.token_id.view(bsz, 1)

        assert token_id is not None
        cur_pos = self.kv_cache.seq_lens[0] if pos is None else pos
        h = cutile_embedding_lookup(token_id.to(self.device).view(bsz), self._embed_w).view(
            bsz, self.hidden_size
        )
        for idx in range(self.num_layers):
            h = self._forward_prefill_layer(idx, h, pos_offset=cur_pos, seq_len=1, batch_size=bsz)
        _, next_tok = cutile_lm_head_fused(
            h, self._norm_w, self._lm_head_rows, eps=self.rms_norm_eps, top_k=1
        )
        return next_tok.view(bsz, 1)

    @torch.inference_mode()
    def generate(
        self,
        prompt_ids: list[int] | torch.Tensor,
        max_new_tokens: int = 32,
        eos_token_id: int | None = None,
        sync_interval: int = 32,
    ) -> tuple[list[int], dict[str, Any]]:
        """Generate `max_new_tokens` greedily using native FP16 CUDA Graphs."""
        ids = (
            torch.tensor([prompt_ids], dtype=torch.long, device=self.device)
            if isinstance(prompt_ids, list)
            else prompt_ids.to(self.device).view(1, -1)
        )
        t0 = time.perf_counter()
        next_tok = self.prefill(ids)
        if self.use_cuda:
            torch.cuda.synchronize()
        t_prefill = (time.perf_counter() - t0) * 1000.0

        generated: list[int] = [int(next_tok.item())]
        cur_pos = int(ids.shape[1])
        rem_tokens = max(0, max_new_tokens - 1)

        if rem_tokens == 0:
            return generated, {
                "prompt_tokens": int(ids.shape[1]),
                "generated_tokens": 1,
                "prefill_ms": round(t_prefill, 2),
                "decode_ms": 0.0,
                "decode_tok_s": 0.0,
                "ms_per_token": 0.0,
            }

        stride = max(1, sync_interval)
        if self.use_cuda_graph and self.use_cuda:
            ws = self._get_or_create_workspace(1)
            last_bucket = self._select_bucket(cur_pos)
            _ = self._capture_full_step_graph(ws, last_bucket)
            ws.step_buf.fill_(0)
            t_dec0 = time.perf_counter()
            steps_left = rem_tokens
            while steps_left > 0:
                b_len = self._select_bucket(cur_pos)
                max_in_bucket = max(1, b_len - cur_pos)
                chunk = min(steps_left, stride, max_in_bucket)
                ws.step_buf.fill_(0)
                graph = self._capture_full_step_graph(ws, b_len)
                for _ in range(chunk):
                    graph.replay()
                chunk_ids = [int(t) for t in ws.history_buf[:chunk, 0].cpu().tolist()]
                cur_pos += chunk
                self.kv_cache.seq_lens = [cur_pos] * self.num_layers
                steps_left -= chunk
                for tid in chunk_ids:
                    generated.append(tid)
                    if eos_token_id is not None and tid == eos_token_id:
                        steps_left = 0
                        break
            torch.cuda.synchronize()
            t_decode = (time.perf_counter() - t_dec0) * 1000.0
        else:
            t_dec0 = time.perf_counter()
            for _ in range(rem_tokens):
                next_tok = self.decode_step(next_tok, pos=cur_pos, use_graph=False)
                cur_pos += 1
                tid = int(next_tok.item())
                generated.append(tid)
                if eos_token_id is not None and tid == eos_token_id:
                    break
            if self.use_cuda:
                torch.cuda.synchronize()
            t_decode = (time.perf_counter() - t_dec0) * 1000.0

        gen_n = max(1, len(generated) - 1)
        return generated, {
            "prompt_tokens": int(ids.shape[1]),
            "generated_tokens": len(generated),
            "prefill_ms": round(t_prefill, 2),
            "decode_ms": round(t_decode, 2),
            "decode_tok_s": round((gen_n / max(1e-6, t_decode)) * 1000.0, 2),
            "ms_per_token": round(t_decode / gen_n, 2),
        }
