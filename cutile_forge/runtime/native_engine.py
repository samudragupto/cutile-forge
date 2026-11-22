"""Native 100% CuTile prefill & autoregressive decode engine (zero-ATen compute hot path).

Phase 9 enhancements:
- Full-step single-`torch.cuda.CUDAGraph` decode capture (zero eager `ct.launch`, zero allocations,
  zero forbidden ATen compute calls, and device-side `pos_buf` / `_static_token_id` feedback).
- Both context-bucketed (`512/1024/2048/4096`) and single fixed-ceiling (`device_masked`) graph strategies.
- Chained multi-step CUDA Graph replay (`sync_interval >= 1`) without per-token `.item()` host sync.
- Packed QKV INT4 dequant GEMM, fused `rope_qk_kv_write_pos_kernel`, fused SwiGLU (`fused_swiglu_dequant`),
  and fused `down_proj + residual` (`fused_gemm_dequant_add`).
- Chunked prefill + interleaved decode scheduler (`prefill_chunk`, `schedule_interleaved_prefill_decode`).
- Batch-bucketed continuous batching (`generate_batch`, `decode_batch_step`).
- Native speculative decoding (`generate_speculative`) with zero-copy `NativePagedKVCache.truncate()`.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal

import torch

from cutile_forge.cuda_tile_init import ct
from cutile_forge.kernels.attention_gqa import (
    fused_attention_gqa,
    fused_attention_gqa_decode_pos,
)
from cutile_forge.kernels.embedding_lmhead import (
    cutile_col_permute,
    cutile_embedding_lookup,
    cutile_ew_mul,
    cutile_lm_head_fused,
    cutile_record_and_advance_step,
)
from cutile_forge.kernels.gemm_dequant import fused_gemm_dequant, fused_gemm_dequant_add
from cutile_forge.kernels.gemm_dequant_silu import (
    fused_gemm_dequant_silu,
    fused_swiglu_dequant,
)
from cutile_forge.kernels.layernorm_residual import fused_rmsnorm_residual
from cutile_forge.kernels.rmsnorm import cutile_rmsnorm
from cutile_forge.kernels.rope import apply_rope, apply_rope_qk_kv_write_pos
from cutile_forge.runtime.native_kv_cache import NativePagedKVCache
from cutile_forge.runtime.native_stream import NativeStreamManager

if TYPE_CHECKING:
    from cutile_forge.engine.model import Mistral7B, MistralLayer
    from cutile_forge.engine.weight_loader import MistralLayerWeights, QuantLinearWeight

GraphStrategy = Literal["bucketed", "device_masked"]
DEFAULT_CONTEXT_BUCKETS: tuple[int, ...] = (512, 1024, 2048, 4096)


def _next_pow2(n: int) -> int:
    p = 1
    while p < n:
        p *= 2
    return p


def _prep_perm_i32(perm: torch.Tensor | None, device: torch.device) -> torch.Tensor | None:
    if perm is None:
        return None
    return perm.to(device=device, dtype=torch.int32).contiguous()


def _perms_equal(p1: torch.Tensor | None, p2: torch.Tensor | None) -> bool:
    if p1 is None and p2 is None:
        return True
    if p1 is None or p2 is None:
        return False
    return bool(torch.equal(p1, p2))


@dataclass
class _PackedNativeLayer:
    """Pre-packed per-layer weights with int32 permutations and packed QKV / fused SwiGLU metadata."""

    layer_idx: int
    input_layernorm: torch.Tensor
    post_attention_layernorm: torch.Tensor
    group_size: int
    # Packed QKV (when q, k, v share perm)
    qkv_packed: bool
    qkv_qweight: torch.Tensor | None
    qkv_scales: torch.Tensor | None
    qkv_zeros: torch.Tensor | None
    qkv_perm_i32: torch.Tensor | None
    # Unpacked Q, K, V fallback
    q_proj: QuantLinearWeight
    k_proj: QuantLinearWeight
    v_proj: QuantLinearWeight
    q_perm_i32: torch.Tensor | None
    k_perm_i32: torch.Tensor | None
    v_perm_i32: torch.Tensor | None
    # O projection
    o_proj: QuantLinearWeight
    o_perm_i32: torch.Tensor | None
    # SwiGLU / MLP
    swiglu_fused: bool
    gate_proj: QuantLinearWeight
    up_proj: QuantLinearWeight
    down_proj: QuantLinearWeight
    gate_perm_i32: torch.Tensor | None
    up_perm_i32: torch.Tensor | None
    down_perm_i32: torch.Tensor | None


@dataclass
class _DecodeGraphWorkspace:
    """Static preallocated buffers for zero-allocation CUDA Graph capture at fixed `batch_size`."""

    batch_size: int
    token_id: torch.Tensor  # (B, 1) int32
    pos_buf: torch.Tensor  # (B,) int32
    step_buf: torch.Tensor  # (1,) int32
    history_buf: torch.Tensor  # (max_seq_len, B) int32
    hidden: torch.Tensor  # (B, D) fp16
    norm: torch.Tensor  # (B, D) fp16
    perm_d: torch.Tensor  # (B, D) fp16
    qkv: torch.Tensor  # (B, QKV_dim) fp16
    q_rot: torch.Tensor  # (B, H_q, 1, head_dim) fp16
    attn_out: torch.Tensor  # (B, H_q, 1, head_dim) fp16
    o_out: torch.Tensor  # (B, D) fp16
    res_mid: torch.Tensor  # (B, D) fp16
    swiglu_out: torch.Tensor  # (B, FFN_dim) fp16
    perm_ffn: torch.Tensor  # (B, FFN_dim) fp16
    lm_scratch: dict[str, torch.Tensor]
    graphs_by_bucket: dict[int, torch.cuda.CUDAGraph]


def _quant_linear_native(
    x: torch.Tensor,
    qw: QuantLinearWeight,
    perm_i32: torch.Tensor | None = None,
    perm_buf: torch.Tensor | None = None,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    p = perm_i32 if perm_i32 is not None else qw.perm
    x_in = cutile_col_permute(x, p, out=perm_buf) if p is not None else x
    return fused_gemm_dequant(
        x_in, qw.qweight, qw.scales, qw.zeros, group_size=qw.group_size, out=out
    )


class NativeModelRuntime:
    """End-to-end native CuTile inference runtime eliminating PyTorch compute ops from decode."""

    def __init__(
        self,
        model: Mistral7B,
        max_seq_len: int = 2048,
        block_size: int = 16,
        kv_dtype: str = "fp16",
        use_cuda_graph: bool = True,
        graph_strategy: GraphStrategy = "bucketed",
        context_buckets: tuple[int, ...] = DEFAULT_CONTEXT_BUCKETS,
        max_batch_size: int = 1,
    ) -> None:
        self.model = model
        self.config = model.config
        self.device = model.device
        self.use_cuda = self.device.type == "cuda" and torch.cuda.is_available()
        self.use_cuda_graph = use_cuda_graph and self.use_cuda
        self.graph_fallback_count: int = 0
        self.graph_strategy: GraphStrategy = graph_strategy
        self.max_seq_len = max_seq_len
        self.block_size = block_size

        valid_buckets = tuple(sorted({min(b, max_seq_len) for b in context_buckets if b > 0}))
        if not valid_buckets or valid_buckets[-1] < max_seq_len:
            valid_buckets = (*valid_buckets, max_seq_len)
        self.context_buckets: tuple[int, ...] = valid_buckets

        self.stream_manager = NativeStreamManager(self.device)
        self.kv_cache = NativePagedKVCache(
            num_layers=self.config.num_hidden_layers,
            num_kv_heads=self.config.num_key_value_heads,
            head_dim=self.config.head_dim,
            block_size=block_size,
            max_seq_len=max_seq_len,
            max_batch_size=max_batch_size,
            kv_dtype=kv_dtype,
            device=self.device,
            stream_manager=self.stream_manager,
        )

        self._embed_w = model.weights.embed_tokens.to(self.device).contiguous()
        self._norm_w = model.weights.norm.to(self.device).contiguous()
        self._lm_head_w = model.weights.lm_head.to(self.device).contiguous()
        self._cos = model.cos.to(self.device).contiguous()
        self._sin = model.sin.to(self.device).contiguous()

        self._cpu_layers: list[MistralLayer] | None = None
        if not self.use_cuda:
            from cutile_forge.engine.model import MistralLayer

            self._cpu_layers = [
                MistralLayer(i, self.config) for i in range(self.config.num_hidden_layers)
            ]
        self._cpu_kv: list[tuple[torch.Tensor, torch.Tensor] | None] = [
            None
        ] * self.config.num_hidden_layers

        self._all_layers_on_gpu = self.use_cuda and all(
            lw.input_layernorm.device.type == "cuda" for lw in self.model.weights.layers
        )
        self._packed_layers: list[_PackedNativeLayer] = []
        if self._all_layers_on_gpu:
            self._packed_layers = [self._pack_layer(lw) for lw in self.model.weights.layers]

        self.max_batch_size = max(1, max_batch_size)
        self._slot_cpu_kv: dict[int, list[tuple[torch.Tensor, torch.Tensor] | None]] = {}
        self._workspaces: dict[int, _DecodeGraphWorkspace] = {}
        self._last_slot_bucket: int = -1
        self._last_active_slots: tuple[int, ...] = ()

    def _pack_layer(self, lw: MistralLayerWeights) -> _PackedNativeLayer:
        qp = _prep_perm_i32(lw.q_proj.perm, self.device)
        kp = _prep_perm_i32(lw.k_proj.perm, self.device)
        vp = _prep_perm_i32(lw.v_proj.perm, self.device)
        op = _prep_perm_i32(lw.o_proj.perm, self.device)
        gp = _prep_perm_i32(lw.gate_proj.perm, self.device)
        up = _prep_perm_i32(lw.up_proj.perm, self.device)
        dp = _prep_perm_i32(lw.down_proj.perm, self.device)

        can_pack_qkv = _perms_equal(qp, kp) and _perms_equal(qp, vp)
        if can_pack_qkv:
            qkv_qw = torch.cat(
                [lw.q_proj.qweight, lw.k_proj.qweight, lw.v_proj.qweight], dim=1
            ).contiguous()
            qkv_sc = torch.cat(
                [lw.q_proj.scales, lw.k_proj.scales, lw.v_proj.scales], dim=1
            ).contiguous()
            qkv_zr = torch.cat(
                [lw.q_proj.zeros, lw.k_proj.zeros, lw.v_proj.zeros], dim=1
            ).contiguous()
        else:
            qkv_qw = None
            qkv_sc = None
            qkv_zr = None

        can_fuse_swiglu = _perms_equal(gp, up)

        return _PackedNativeLayer(
            layer_idx=lw.layer_idx,
            input_layernorm=lw.input_layernorm.contiguous(),
            post_attention_layernorm=lw.post_attention_layernorm.contiguous(),
            group_size=lw.q_proj.group_size,
            qkv_packed=can_pack_qkv,
            qkv_qweight=qkv_qw,
            qkv_scales=qkv_sc,
            qkv_zeros=qkv_zr,
            qkv_perm_i32=qp,
            q_proj=lw.q_proj,
            k_proj=lw.k_proj,
            v_proj=lw.v_proj,
            q_perm_i32=qp,
            k_perm_i32=kp,
            v_perm_i32=vp,
            o_proj=lw.o_proj,
            o_perm_i32=op,
            swiglu_fused=can_fuse_swiglu,
            gate_proj=lw.gate_proj,
            up_proj=lw.up_proj,
            down_proj=lw.down_proj,
            gate_perm_i32=gp,
            up_perm_i32=up,
            down_perm_i32=dp,
        )

    def reset(self) -> None:
        self.kv_cache.reset()
        self._cpu_kv = [None] * self.config.num_hidden_layers
        self._slot_cpu_kv.clear()
        self._last_slot_bucket = -1
        self._last_active_slots = ()

    def clear_slot(self, slot_idx: int) -> None:
        """Clear host KV state for a finished/cancelled request slot."""
        self._slot_cpu_kv.pop(slot_idx, None)
        if slot_idx in self._last_active_slots:
            self._last_slot_bucket = -1
            self._last_active_slots = ()

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

    def _get_or_create_workspace(self, batch_size: int) -> _DecodeGraphWorkspace:
        if batch_size in self._workspaces:
            return self._workspaces[batch_size]
        self.kv_cache.ensure_batch_capacity(batch_size)
        cfg = self.config
        qkv_dim = (cfg.num_attention_heads + 2 * cfg.num_key_value_heads) * cfg.head_dim
        v_dim = self._lm_head_w.shape[0]
        num_vt = ct.cdiv(v_dim, 256)

        token_id = torch.zeros((batch_size, 1), device=self.device, dtype=torch.int32)
        pos_buf = torch.zeros((batch_size,), device=self.device, dtype=torch.int32)
        step_buf = torch.zeros((1,), device=self.device, dtype=torch.int32)
        history_buf = torch.zeros(
            (self.max_seq_len, _next_pow2(batch_size)), device=self.device, dtype=torch.int32
        )
        hidden = torch.zeros((batch_size, cfg.hidden_size), device=self.device, dtype=torch.float16)
        norm = torch.zeros((batch_size, cfg.hidden_size), device=self.device, dtype=torch.float16)
        perm_d = torch.zeros((batch_size, cfg.hidden_size), device=self.device, dtype=torch.float16)
        qkv = torch.zeros((batch_size, qkv_dim), device=self.device, dtype=torch.float16)
        q_rot = torch.zeros(
            (batch_size, cfg.num_attention_heads, 1, cfg.head_dim),
            device=self.device,
            dtype=torch.float16,
        )
        attn_out = torch.zeros(
            (batch_size, cfg.num_attention_heads, 1, cfg.head_dim),
            device=self.device,
            dtype=torch.float16,
        )
        o_out = torch.zeros((batch_size, cfg.hidden_size), device=self.device, dtype=torch.float16)
        res_mid = torch.zeros(
            (batch_size, cfg.hidden_size), device=self.device, dtype=torch.float16
        )
        swiglu_out = torch.zeros(
            (batch_size, cfg.intermediate_size), device=self.device, dtype=torch.float16
        )
        perm_ffn = torch.zeros(
            (batch_size, cfg.intermediate_size), device=self.device, dtype=torch.float16
        )
        lm_scratch = {
            "norm_out": torch.zeros(
                (batch_size, cfg.hidden_size), device=self.device, dtype=torch.float16
            ),
            "part_val": torch.zeros((batch_size, num_vt), device=self.device, dtype=torch.float32),
            "part_idx": torch.zeros((batch_size, num_vt), device=self.device, dtype=torch.int32),
            "out_val": torch.zeros((batch_size, 1), device=self.device, dtype=torch.float32),
            "out_idx": token_id,
        }
        ws = _DecodeGraphWorkspace(
            batch_size=batch_size,
            token_id=token_id,
            pos_buf=pos_buf,
            step_buf=step_buf,
            history_buf=history_buf,
            hidden=hidden,
            norm=norm,
            perm_d=perm_d,
            qkv=qkv,
            q_rot=q_rot,
            attn_out=attn_out,
            o_out=o_out,
            res_mid=res_mid,
            swiglu_out=swiglu_out,
            perm_ffn=perm_ffn,
            lm_scratch=lm_scratch,
            graphs_by_bucket={},
        )
        self._workspaces[batch_size] = ws
        return ws

    def _select_bucket(self, pos: int) -> int:
        if self.graph_strategy == "device_masked":
            return self.max_seq_len
        needed = pos + 1
        for b in self.context_buckets:
            if needed <= b:
                return b
        return self.max_seq_len

    def _execute_decode_step_in_workspace(
        self,
        ws: _DecodeGraphWorkspace,
        bucket_len: int,
    ) -> None:
        """Execute 1 complete 32-layer decode step inside `ws` using only static addresses."""
        cfg = self.config
        bsz = ws.batch_size
        dyn_loop = self.graph_strategy == "bucketed"

        cutile_embedding_lookup(ws.token_id.view(bsz), self._embed_w, out=ws.hidden)

        for idx, pl in enumerate(self._packed_layers):
            cutile_rmsnorm(ws.hidden, pl.input_layernorm, eps=cfg.rms_norm_eps, out=ws.norm)

            if (
                pl.qkv_packed
                and pl.qkv_qweight is not None
                and pl.qkv_scales is not None
                and pl.qkv_zeros is not None
            ):
                qkv_in = (
                    cutile_col_permute(ws.norm, pl.qkv_perm_i32, out=ws.perm_d)
                    if pl.qkv_perm_i32 is not None
                    else ws.norm
                )
                fused_gemm_dequant(
                    qkv_in,
                    pl.qkv_qweight,
                    pl.qkv_scales,
                    pl.qkv_zeros,
                    group_size=pl.group_size,
                    out=ws.qkv,
                )
            else:
                q_dim = cfg.num_attention_heads * cfg.head_dim
                kv_dim = cfg.num_key_value_heads * cfg.head_dim
                _quant_linear_native(
                    ws.norm, pl.q_proj, pl.q_perm_i32, ws.perm_d, out=ws.qkv[:, :q_dim]
                )
                _quant_linear_native(
                    ws.norm,
                    pl.k_proj,
                    pl.k_perm_i32,
                    ws.perm_d,
                    out=ws.qkv[:, q_dim : q_dim + kv_dim],
                )
                _quant_linear_native(
                    ws.norm,
                    pl.v_proj,
                    pl.v_perm_i32,
                    ws.perm_d,
                    out=ws.qkv[:, q_dim + kv_dim :],
                )

            apply_rope_qk_kv_write_pos(
                ws.qkv,
                self._cos,
                self._sin,
                ws.pos_buf,
                self.kv_cache.block_tables[idx][:bsz],
                ws.q_rot,
                self.kv_cache.k_pools[idx][:bsz],
                self.kv_cache.v_pools[idx][:bsz],
                n_q_heads=cfg.num_attention_heads,
                n_kv_heads=cfg.num_key_value_heads,
                head_dim=cfg.head_dim,
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
                sliding_window=int(cfg.sliding_window or 0),
            )

            attn_2d = ws.attn_out.view(bsz, cfg.hidden_size)
            _quant_linear_native(attn_2d, pl.o_proj, pl.o_perm_i32, ws.perm_d, out=ws.o_out)

            fused_rmsnorm_residual(
                ws.o_out,
                ws.hidden,
                pl.post_attention_layernorm,
                eps=cfg.rms_norm_eps,
                out_norm=ws.norm,
                out_res=ws.res_mid,
            )

            if pl.swiglu_fused:
                g_in = (
                    cutile_col_permute(ws.norm, pl.gate_perm_i32, out=ws.perm_d)
                    if pl.gate_perm_i32 is not None
                    else ws.norm
                )
                fused_swiglu_dequant(
                    g_in,
                    pl.gate_proj.qweight,
                    pl.gate_proj.scales,
                    pl.gate_proj.zeros,
                    pl.up_proj.qweight,
                    pl.up_proj.scales,
                    pl.up_proj.zeros,
                    group_size=pl.group_size,
                    out=ws.swiglu_out,
                )
            else:
                g_in = (
                    cutile_col_permute(ws.norm, pl.gate_perm_i32, out=ws.perm_d)
                    if pl.gate_perm_i32 is not None
                    else ws.norm
                )
                ga = fused_gemm_dequant_silu(
                    g_in,
                    pl.gate_proj.qweight,
                    pl.gate_proj.scales,
                    pl.gate_proj.zeros,
                    group_size=pl.group_size,
                )
                uv = _quant_linear_native(ws.norm, pl.up_proj, pl.up_perm_i32, ws.perm_d)
                cutile_ew_mul(ga, uv, out=ws.swiglu_out)

            d_in = (
                cutile_col_permute(ws.swiglu_out, pl.down_perm_i32, out=ws.perm_ffn)
                if pl.down_perm_i32 is not None
                else ws.swiglu_out
            )
            fused_gemm_dequant_add(
                d_in,
                pl.down_proj.qweight,
                pl.down_proj.scales,
                pl.down_proj.zeros,
                ws.res_mid,
                group_size=pl.group_size,
                out=ws.hidden,
            )

        cutile_lm_head_fused(
            ws.hidden,
            self._norm_w,
            self._lm_head_w,
            eps=cfg.rms_norm_eps,
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
        self, ws: _DecodeGraphWorkspace, bucket_len: int
    ) -> torch.cuda.CUDAGraph:
        """Capture a single full-step `torch.cuda.CUDAGraph` for `(ws.batch_size, bucket_len)`."""
        if bucket_len in ws.graphs_by_bucket:
            return ws.graphs_by_bucket[bucket_len]

        # Save live state before warmup/capture so KV cache at current position is untouched
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

        # Restore state
        for idx in range(self.config.num_hidden_layers):
            self.kv_cache.k_pools[idx][: ws.batch_size, :, scratch_slot, :].copy_(saved_k[idx])
            self.kv_cache.v_pools[idx][: ws.batch_size, :, scratch_slot, :].copy_(saved_v[idx])
        ws.token_id.copy_(saved_tok)
        ws.pos_buf.copy_(saved_pos)
        ws.step_buf.copy_(saved_step)

        ws.graphs_by_bucket[bucket_len] = g
        return g

    def _forward_layer_native(
        self,
        idx: int,
        x_2d: torch.Tensor,
        pos_offset: int,
        seq_len: int,
        batch_size: int = 1,
        slot_indices: list[int] | None = None,
        slot_tensor: torch.Tensor | None = None,
    ) -> torch.Tensor:
        cfg = self.config
        lw = self._layer_on_device(idx)

        if not self.use_cuda:
            from cutile_forge.engine.model import _pytorch_rmsnorm, _quant_linear_pytorch

            assert self._cpu_layers is not None
            xn = _pytorch_rmsnorm(x_2d, lw.input_layernorm, eps=cfg.rms_norm_eps)
            q = (
                _quant_linear_pytorch(xn, lw.q_proj)
                .view(batch_size, seq_len, cfg.num_attention_heads, cfg.head_dim)
                .transpose(1, 2)
            )
            k = (
                _quant_linear_pytorch(xn, lw.k_proj)
                .view(batch_size, seq_len, cfg.num_key_value_heads, cfg.head_dim)
                .transpose(1, 2)
            )
            v = (
                _quant_linear_pytorch(xn, lw.v_proj)
                .view(batch_size, seq_len, cfg.num_key_value_heads, cfg.head_dim)
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
            sw = int(cfg.sliding_window or 0)
            if slot_indices is not None:
                att_chunks: list[torch.Tensor] = []
                for bi, sid in enumerate(slot_indices):
                    s_list = self._slot_cpu_kv.setdefault(sid, [None] * cfg.num_hidden_layers)
                    prev_s = s_list[idx]
                    k_b = k[bi : bi + 1]
                    v_b = v[bi : bi + 1]
                    if prev_s is not None and pos_offset > 0:
                        kf = torch.cat([prev_s[0][:, :, :pos_offset, :], k_b], dim=2)
                        vf = torch.cat([prev_s[1][:, :, :pos_offset, :], v_b], dim=2)
                    else:
                        kf, vf = k_b, v_b
                    s_list[idx] = (kf, vf)
                    kr = kf.repeat_interleave(cfg.gqa_ratio, dim=1)
                    vr = vf.repeat_interleave(cfg.gqa_ratio, dim=1)
                    s_kv = int(kf.shape[2])
                    if sw > 0 or (seq_len > 1 and s_kv > seq_len):
                        q_p = torch.arange(seq_len, device=q.device).view(seq_len, 1) + (
                            s_kv - seq_len
                        )
                        k_p = torch.arange(s_kv, device=q.device).view(1, s_kv)
                        c_mask = k_p <= q_p
                        if sw > 0:
                            c_mask = c_mask & (k_p > q_p - sw)
                        att_b = torch.nn.functional.scaled_dot_product_attention(
                            q[bi : bi + 1], kr, vr, attn_mask=c_mask
                        )
                    else:
                        att_b = torch.nn.functional.scaled_dot_product_attention(
                            q[bi : bi + 1], kr, vr, is_causal=(seq_len > 1)
                        )
                    att_chunks.append(att_b)
                att = torch.cat(att_chunks, dim=0)
            else:
                prev_kv = self._cpu_kv[idx]
                k_prev = (
                    prev_kv[0][:, :, :pos_offset, :]
                    if (prev_kv is not None and pos_offset > 0)
                    else None
                )
                v_prev = (
                    prev_kv[1][:, :, :pos_offset, :]
                    if (prev_kv is not None and pos_offset > 0)
                    else None
                )
                kf = k if k_prev is None else torch.cat([k_prev, k], dim=2)
                vf = v if v_prev is None else torch.cat([v_prev, v], dim=2)
                self._cpu_kv[idx] = (kf, vf)
                kr = kf.repeat_interleave(cfg.gqa_ratio, dim=1)
                vr = vf.repeat_interleave(cfg.gqa_ratio, dim=1)
                s_kv = int(kf.shape[2])
                if sw > 0 or (seq_len > 1 and s_kv > seq_len):
                    q_p = torch.arange(seq_len, device=q.device).view(seq_len, 1) + (s_kv - seq_len)
                    k_p = torch.arange(s_kv, device=q.device).view(1, s_kv)
                    c_mask = k_p <= q_p
                    if sw > 0:
                        c_mask = c_mask & (k_p > q_p - sw)
                    att = torch.nn.functional.scaled_dot_product_attention(
                        q, kr, vr, attn_mask=c_mask
                    )
                else:
                    att = torch.nn.functional.scaled_dot_product_attention(
                        q, kr, vr, is_causal=(seq_len > 1)
                    )
            att_2d = att.transpose(1, 2).contiguous().view(batch_size * seq_len, cfg.hidden_size)
            op = _quant_linear_pytorch(att_2d, lw.o_proj)
            rm = (x_2d.float() + op.float()).to(torch.float16)
            pn = _pytorch_rmsnorm(rm, lw.post_attention_layernorm, eps=cfg.rms_norm_eps)
            gate = torch.nn.functional.silu(_quant_linear_pytorch(pn, lw.gate_proj).float()).to(
                torch.float16
            )
            up = _quant_linear_pytorch(pn, lw.up_proj)
            ffn = _quant_linear_pytorch(gate * up, lw.down_proj)
            return (rm.float() + ffn.float()).to(torch.float16)

        pl = self._packed_layers[idx] if self._packed_layers else None
        x_n = cutile_rmsnorm(x_2d, lw.input_layernorm, eps=cfg.rms_norm_eps)

        if (
            pl is not None
            and pl.qkv_packed
            and pl.qkv_qweight is not None
            and pl.qkv_scales is not None
            and pl.qkv_zeros is not None
        ):
            qkv_in = (
                cutile_col_permute(x_n, pl.qkv_perm_i32) if pl.qkv_perm_i32 is not None else x_n
            )
            qkv_out = fused_gemm_dequant(
                qkv_in, pl.qkv_qweight, pl.qkv_scales, pl.qkv_zeros, group_size=pl.group_size
            )
            q_dim = cfg.num_attention_heads * cfg.head_dim
            kv_dim = cfg.num_key_value_heads * cfg.head_dim
            q = qkv_out[:, :q_dim].contiguous()
            k = qkv_out[:, q_dim : q_dim + kv_dim].contiguous()
            v = qkv_out[:, q_dim + kv_dim :].contiguous()
        else:
            q = _quant_linear_native(x_n, lw.q_proj, pl.q_perm_i32 if pl else None)
            k = _quant_linear_native(x_n, lw.k_proj, pl.k_perm_i32 if pl else None)
            v = _quant_linear_native(x_n, lw.v_proj, pl.v_perm_i32 if pl else None)

        q = q.view(batch_size, seq_len, cfg.num_attention_heads, cfg.head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_len, cfg.num_key_value_heads, cfg.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, cfg.num_key_value_heads, cfg.head_dim).transpose(1, 2)

        q = apply_rope(q, self._cos, self._sin, position_offset=pos_offset)
        k = apply_rope(k, self._cos, self._sin, position_offset=pos_offset)

        if slot_indices is not None:
            end_p = pos_offset + seq_len
            if end_p > self.kv_cache.max_seq_len:
                raise RuntimeError(
                    f"KV sequence length {end_p} exceeds max_seq_len={self.kv_cache.max_seq_len}"
                )
            slot_t = (
                slot_tensor
                if slot_tensor is not None
                else torch.tensor(slot_indices, device=self.device, dtype=torch.long)
            )
            self.kv_cache.k_pools[idx][slot_t, :, pos_offset:end_p, :] = k.to(
                self.kv_cache.storage_dtype
            )
            self.kv_cache.v_pools[idx][slot_t, :, pos_offset:end_p, :] = v.to(
                self.kv_cache.storage_dtype
            )
            if pos_offset == 0:
                k_full, v_full = k, v
            else:
                k_full = self.kv_cache.k_pools[idx][slot_t, :, :end_p, :].to(torch.float16)
                v_full = self.kv_cache.v_pools[idx][slot_t, :, :end_p, :].to(torch.float16)
        else:
            k_full, v_full = self.kv_cache.append_and_get(idx, k, v, start_pos=pos_offset)
        attn_out = fused_attention_gqa(
            q,
            k_full,
            v_full,
            causal=True,
            sliding_window=int(cfg.sliding_window or 0),
        )
        attn_2d = (
            attn_out.view(batch_size, cfg.hidden_size)
            if seq_len == 1
            else attn_out.transpose(1, 2).contiguous().view(batch_size * seq_len, cfg.hidden_size)
        )

        o_proj = _quant_linear_native(attn_2d, lw.o_proj, pl.o_perm_i32 if pl else None)
        post_norm, residual_mid = fused_rmsnorm_residual(
            o_proj, x_2d, lw.post_attention_layernorm, eps=cfg.rms_norm_eps
        )

        if pl is not None and pl.swiglu_fused:
            g_in = (
                cutile_col_permute(post_norm, pl.gate_perm_i32)
                if pl.gate_perm_i32 is not None
                else post_norm
            )
            act_mul = fused_swiglu_dequant(
                g_in,
                pl.gate_proj.qweight,
                pl.gate_proj.scales,
                pl.gate_proj.zeros,
                pl.up_proj.qweight,
                pl.up_proj.scales,
                pl.up_proj.zeros,
                group_size=pl.group_size,
            )
        else:
            g_perm = pl.gate_perm_i32 if pl else lw.gate_proj.perm
            g_in = cutile_col_permute(post_norm, g_perm) if g_perm is not None else post_norm
            gate_act = fused_gemm_dequant_silu(
                g_in,
                lw.gate_proj.qweight,
                lw.gate_proj.scales,
                lw.gate_proj.zeros,
                group_size=lw.gate_proj.group_size,
            )
            up_val = _quant_linear_native(post_norm, lw.up_proj, pl.up_perm_i32 if pl else None)
            act_mul = cutile_ew_mul(gate_act, up_val)

        d_perm = pl.down_perm_i32 if pl else lw.down_proj.perm
        d_in = cutile_col_permute(act_mul, d_perm) if d_perm is not None else act_mul
        return fused_gemm_dequant_add(
            d_in,
            lw.down_proj.qweight,
            lw.down_proj.scales,
            lw.down_proj.zeros,
            residual_mid,
            group_size=lw.down_proj.group_size,
        )

    @torch.inference_mode()
    def prefill(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Prefill prompt `(B, S)` using native CuTile kernels and return greedy next token `(B, 1)`."""
        self.reset()
        ids = input_ids.to(self.device)
        if ids.ndim == 1:
            ids = ids.unsqueeze(0)
        bsz, seq_len = int(ids.shape[0]), int(ids.shape[1])
        self.kv_cache.ensure_batch_capacity(bsz)
        h = cutile_embedding_lookup(ids.reshape(bsz * seq_len), self._embed_w).view(
            bsz * seq_len, self.config.hidden_size
        )
        for idx in range(self.config.num_hidden_layers):
            h = self._forward_layer_native(idx, h, pos_offset=0, seq_len=seq_len, batch_size=bsz)
        h_last = h.view(bsz, seq_len, self.config.hidden_size)[:, -1, :].contiguous()
        _, next_tok = cutile_lm_head_fused(
            h_last, self._norm_w, self._lm_head_w, eps=self.config.rms_norm_eps, top_k=1
        )
        if self._all_layers_on_gpu:
            ws = self._get_or_create_workspace(bsz)
            ws.token_id.copy_(next_tok.view(bsz, 1).to(torch.int32))
            ws.pos_buf.fill_(seq_len)
            ws.step_buf.fill_(0)
        return next_tok.view(bsz, 1)

    @torch.inference_mode()
    def prefill_chunk(
        self,
        chunk_ids: torch.Tensor,
        start_pos: int,
        emit_next_token: bool = False,
    ) -> torch.Tensor | None:
        """Process one chunk `(1, S_chunk)` of a chunked prefill at `start_pos` without stalling decodes."""
        if start_pos == 0:
            self.reset()
        ids = chunk_ids.to(self.device)
        if ids.ndim == 1:
            ids = ids.unsqueeze(0)
        bsz, s_chunk = int(ids.shape[0]), int(ids.shape[1])
        self.kv_cache.ensure_batch_capacity(bsz)
        h = cutile_embedding_lookup(ids.reshape(bsz * s_chunk), self._embed_w).view(
            bsz * s_chunk, self.config.hidden_size
        )
        for idx in range(self.config.num_hidden_layers):
            h = self._forward_layer_native(
                idx, h, pos_offset=start_pos, seq_len=s_chunk, batch_size=bsz
            )
        if not emit_next_token:
            return None
        h_last = h.view(bsz, s_chunk, self.config.hidden_size)[:, -1, :].contiguous()
        _, next_tok = cutile_lm_head_fused(
            h_last, self._norm_w, self._lm_head_w, eps=self.config.rms_norm_eps, top_k=1
        )
        if self._all_layers_on_gpu:
            ws = self._get_or_create_workspace(bsz)
            ws.token_id.copy_(next_tok.view(bsz, 1).to(torch.int32))
            ws.pos_buf.fill_(start_pos + s_chunk)
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
        """Execute a single autoregressive decode step (zero ATen compute ops on CUDA)."""
        cfg = self.config
        use_g = self.use_cuda_graph if use_graph is None else (use_graph and self.use_cuda)
        bsz = 1 if token_id is None else max(1, int(token_id.shape[0]) if token_id.ndim >= 2 else 1)

        if use_g and self._all_layers_on_gpu:
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
                self.kv_cache.seq_lens = [cur_pos + 1] * cfg.num_hidden_layers
                return ws.token_id.view(bsz, 1)

        assert token_id is not None
        cur_pos = self.kv_cache.seq_lens[0] if pos is None else pos
        ids = token_id if token_id.device == self.device else token_id.to(self.device)
        if self.use_cuda and self._all_layers_on_gpu:
            ws = self._get_or_create_workspace(bsz)
            if ids.data_ptr() != ws.token_id.data_ptr():
                ws.token_id.copy_(ids.view(bsz, 1))
            ws.pos_buf.fill_(cur_pos)
            bucket_len = self._select_bucket(cur_pos)
            self._execute_decode_step_in_workspace(ws, bucket_len)
            self.kv_cache.seq_lens = [cur_pos + 1] * cfg.num_hidden_layers
            return ws.token_id.view(bsz, 1)

        h = cutile_embedding_lookup(ids.view(bsz), self._embed_w).view(bsz, cfg.hidden_size)
        for idx in range(cfg.num_hidden_layers):
            h = self._forward_layer_native(idx, h, pos_offset=cur_pos, seq_len=1, batch_size=bsz)
        _, next_tok = cutile_lm_head_fused(
            h, self._norm_w, self._lm_head_w, eps=cfg.rms_norm_eps, top_k=1
        )
        return next_tok.view(bsz, 1)

    @torch.inference_mode()
    def generate(
        self,
        prompt_ids: list[int],
        max_new_tokens: int = 32,
        sync_interval: int = 32,
        eos_token_id: int | None = None,
    ) -> tuple[list[int], dict[str, Any]]:
        """Run full prefill + native decode loop (supports chained multi-step CUDA Graph replay)."""
        t0 = time.perf_counter()
        prompt_t = torch.tensor([prompt_ids], dtype=torch.long, device=self.device)
        next_tok = self.prefill(prompt_t)
        if self.use_cuda:
            torch.cuda.synchronize()
        t_prefill = (time.perf_counter() - t0) * 1000.0

        first_id = int(next_tok.item())
        generated: list[int] = [first_id]
        if eos_token_id is not None and first_id == eos_token_id:
            return generated, {
                "prompt_tokens": len(prompt_ids),
                "generated_tokens": 1,
                "prefill_ms": round(t_prefill, 2),
                "decode_ms": 0.0,
                "decode_tok_s": 0.0,
                "ms_per_token": 0.0,
            }

        cur_pos = len(prompt_ids)
        rem_tokens = max(0, max_new_tokens - 1)

        if self.use_cuda_graph and self._all_layers_on_gpu and rem_tokens > 0:
            ws = self._get_or_create_workspace(1)
            # Pre-capture any context buckets we will touch before timing decode
            end_pos = cur_pos + rem_tokens
            if self.graph_strategy == "device_masked":
                _ = self._capture_full_step_graph(ws, self.max_seq_len)
            else:
                for b in self.context_buckets:
                    if cur_pos < b:
                        _ = self._capture_full_step_graph(ws, b)
                    if b >= end_pos:
                        break

            ws.token_id.fill_(first_id)
            ws.pos_buf.fill_(cur_pos)
            ws.step_buf.fill_(0)
            if self.use_cuda:
                torch.cuda.synchronize()

            eff_sync = max(1, sync_interval)
            t_dec0 = time.perf_counter()
            steps_done = 0
            while steps_done < rem_tokens:
                bucket_len = self._select_bucket(cur_pos)
                graph = ws.graphs_by_bucket[bucket_len]
                # Chained replay up to bucket boundary or sync_interval without any host sync
                can_run_in_bucket = max(1, bucket_len - cur_pos)
                chunk = min(eff_sync, rem_tokens - steps_done, can_run_in_bucket)
                for _ in range(chunk):
                    graph.replay()
                cur_pos += chunk
                steps_done += chunk
                self.kv_cache.seq_lens = [cur_pos] * self.config.num_hidden_layers

                # Async/batched read of the chunk's generated token IDs from device history buffer
                chunk_toks = ws.history_buf[steps_done - chunk : steps_done, 0].cpu().tolist()
                stop_now = False
                for tid in chunk_toks:
                    generated.append(int(tid))
                    if eos_token_id is not None and int(tid) == eos_token_id:
                        stop_now = True
                        break
                if stop_now:
                    break

            if self.use_cuda:
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
            "prompt_tokens": len(prompt_ids),
            "generated_tokens": len(generated),
            "prefill_ms": round(t_prefill, 2),
            "decode_ms": round(t_decode, 2),
            "decode_tok_s": round((gen_n / max(1e-6, t_decode)) * 1000.0, 2),
            "ms_per_token": round(t_decode / gen_n, 2),
        }

    @staticmethod
    def _batch_bucket_for(needed_slots: int) -> int:
        for b in (1, 2, 4, 8, 16, 32, 64):
            if needed_slots <= b:
                return b
        return _next_pow2(needed_slots)

    @torch.inference_mode()
    def prefill_into_slots(
        self,
        slot_indices: list[int],
        input_ids: torch.Tensor,
    ) -> torch.Tensor:
        """Prefill `(B_group, S)` into specific KV cache slots without resetting active slots."""
        if not slot_indices:
            return torch.empty((0, 1), dtype=torch.long, device=self.device)
        ids = input_ids.to(self.device)
        if ids.ndim == 1:
            ids = ids.unsqueeze(0)
        bsz, seq_len = int(ids.shape[0]), int(ids.shape[1])
        if bsz != len(slot_indices):
            raise ValueError(f"slot_indices length {len(slot_indices)} != batch size {bsz}")
        needed_cap = max(slot_indices) + 1
        if needed_cap > self.kv_cache.max_batch_size:
            self.kv_cache.ensure_batch_capacity(needed_cap)

        slot_t = torch.tensor(slot_indices, device=self.device, dtype=torch.long)
        h = cutile_embedding_lookup(ids.reshape(bsz * seq_len), self._embed_w).view(
            bsz * seq_len, self.config.hidden_size
        )
        for idx in range(self.config.num_hidden_layers):
            h = self._forward_layer_native(
                idx,
                h,
                pos_offset=0,
                seq_len=seq_len,
                batch_size=bsz,
                slot_indices=slot_indices,
                slot_tensor=slot_t,
            )
        h_last = h.view(bsz, seq_len, self.config.hidden_size)[:, -1, :].contiguous()
        _, next_tok = cutile_lm_head_fused(
            h_last, self._norm_w, self._lm_head_w, eps=self.config.rms_norm_eps, top_k=1
        )
        return next_tok.view(bsz, 1)

    @torch.inference_mode()
    def decode_slots_step(
        self,
        active_slots: list[int],
        token_ids: list[int],
        positions: list[int],
        sync_inputs: bool = True,
    ) -> list[int]:
        """Run one decode step across ragged per-slot `(token_id, pos)` pairs."""
        if not active_slots:
            return []
        if not (len(active_slots) == len(token_ids) == len(positions)):
            raise ValueError("active_slots, token_ids, and positions must have matching lengths")

        if self.use_cuda and self._all_layers_on_gpu:
            b_cap = self._batch_bucket_for(max(active_slots) + 1)
            ws = self._get_or_create_workspace(b_cap)
            max_pos = max(positions)
            bucket_len = self._select_bucket(max_pos)

            slots_key = tuple(active_slots)
            need_sync = (
                sync_inputs
                or b_cap != self._last_slot_bucket
                or slots_key != self._last_active_slots
            )
            if need_sync:
                tok_vals = [1] * b_cap
                pos_vals = [0] * b_cap
                for sid, tid, pos in zip(active_slots, token_ids, positions, strict=True):
                    tok_vals[sid] = int(tid)
                    pos_vals[sid] = int(pos)
                ws.token_id[:, 0].copy_(
                    torch.tensor(tok_vals, dtype=torch.int32, device=self.device)
                )
                ws.pos_buf.copy_(torch.tensor(pos_vals, dtype=torch.int32, device=self.device))
            ws.step_buf.fill_(0)

            ran_graph = False
            if self.use_cuda_graph:
                try:
                    graph = self._capture_full_step_graph(ws, bucket_len)
                    graph.replay()
                    ran_graph = True
                except RuntimeError:
                    self.use_cuda_graph = False
                    self.graph_fallback_count += 1
            if not ran_graph:
                self._execute_decode_step_in_workspace(ws, bucket_len)

            self._last_slot_bucket = b_cap
            self._last_active_slots = slots_key
            out_cpu = ws.token_id[:b_cap, 0].cpu().tolist()
            return [int(out_cpu[sid]) for sid in active_slots]

        out_tokens: list[int] = []
        for sid, tid, pos in zip(active_slots, token_ids, positions, strict=True):
            t_in = torch.tensor([tid], dtype=torch.long, device=self.device)
            h = cutile_embedding_lookup(t_in, self._embed_w).view(1, self.config.hidden_size)
            for idx in range(self.config.num_hidden_layers):
                h = self._forward_layer_native(
                    idx,
                    h,
                    pos_offset=int(pos),
                    seq_len=1,
                    batch_size=1,
                    slot_indices=[sid],
                )
            _, nxt = cutile_lm_head_fused(
                h, self._norm_w, self._lm_head_w, eps=self.config.rms_norm_eps, top_k=1
            )
            out_tokens.append(int(nxt.item()))
        return out_tokens

    @torch.inference_mode()
    def generate_batch(
        self,
        prompt_ids_list: list[list[int]],
        max_new_tokens: int = 32,
        sync_interval: int = 32,
    ) -> tuple[list[list[int]], dict[str, Any]]:
        """Length-grouped continuous batching on `NativeModelRuntime` (zero padding contamination)."""
        bsz = len(prompt_ids_list)
        if bsz == 0:
            return [], {"batch_size": 0, "total_decode_tok_s": 0.0}
        if bsz == 1:
            toks, st = self.generate(
                prompt_ids_list[0], max_new_tokens=max_new_tokens, sync_interval=sync_interval
            )
            return [toks], {
                "batch_size": 1,
                "total_decode_tok_s": st["decode_tok_s"],
                "per_req_tok_s": st["decode_tok_s"],
                "decode_ms": st["decode_ms"],
                "prefill_ms": st["prefill_ms"],
            }

        self.reset()
        b_cap = (
            self._batch_bucket_for(bsz)
            if (self.use_cuda and self.use_cuda_graph and self._all_layers_on_gpu)
            else bsz
        )
        self.kv_cache.ensure_batch_capacity(max(bsz, b_cap))

        by_len: dict[int, list[int]] = {}
        for slot_id, prompt in enumerate(prompt_ids_list):
            by_len.setdefault(len(prompt), []).append(slot_id)

        t0 = time.perf_counter()
        first_toks = [0] * bsz
        for _, slot_group in by_len.items():
            grp_t = torch.tensor(
                [prompt_ids_list[s] for s in slot_group], dtype=torch.long, device=self.device
            )
            nxt = self.prefill_into_slots(slot_group, grp_t)
            for sid, tok_val in zip(
                slot_group, nxt.view(len(slot_group)).cpu().tolist(), strict=True
            ):
                first_toks[sid] = int(tok_val)
        if self.use_cuda:
            torch.cuda.synchronize()
        t_prefill = (time.perf_counter() - t0) * 1000.0

        outputs: list[list[int]] = [[first_toks[b]] for b in range(bsz)]
        positions = [len(p) for p in prompt_ids_list]
        rem_tokens = max(0, max_new_tokens - 1)

        if self.use_cuda_graph and self._all_layers_on_gpu and rem_tokens > 0:
            ws = self._get_or_create_workspace(b_cap)
            max_start_pos = max(positions)
            end_pos = max_start_pos + rem_tokens
            for b in self.context_buckets:
                if max_start_pos < b:
                    _ = self._capture_full_step_graph(ws, b)
                if b >= end_pos:
                    break

            tok_init = [first_toks[b] if b < bsz else 1 for b in range(b_cap)]
            pos_init = [positions[b] if b < bsz else 0 for b in range(b_cap)]
            ws.token_id[:, 0].copy_(torch.tensor(tok_init, dtype=torch.int32, device=self.device))
            ws.pos_buf.copy_(torch.tensor(pos_init, dtype=torch.int32, device=self.device))
            ws.step_buf.fill_(0)
            if self.use_cuda:
                torch.cuda.synchronize()

            eff_sync = max(1, sync_interval)
            t_dec0 = time.perf_counter()
            steps_done = 0
            cur_max_pos = max_start_pos
            while steps_done < rem_tokens:
                bucket_len = self._select_bucket(cur_max_pos)
                graph = ws.graphs_by_bucket[bucket_len]
                chunk = min(eff_sync, rem_tokens - steps_done, max(1, bucket_len - cur_max_pos))
                for _ in range(chunk):
                    graph.replay()
                cur_max_pos += chunk
                steps_done += chunk
                self.kv_cache.seq_lens = [cur_max_pos] * self.config.num_hidden_layers
                chunk_mat = ws.history_buf[steps_done - chunk : steps_done, :bsz].cpu().tolist()
                for row in chunk_mat:
                    for b in range(bsz):
                        outputs[b].append(int(row[b]))
            if self.use_cuda:
                torch.cuda.synchronize()
            t_decode = (time.perf_counter() - t_dec0) * 1000.0
        else:
            t_dec0 = time.perf_counter()
            active = list(range(bsz))
            cur_toks = list(first_toks)
            for _ in range(rem_tokens):
                cur_toks = self.decode_slots_step(active, cur_toks, positions)
                for b in range(bsz):
                    outputs[b].append(cur_toks[b])
                    positions[b] += 1
            if self.use_cuda:
                torch.cuda.synchronize()
            t_decode = (time.perf_counter() - t_dec0) * 1000.0

        total_dec_toks = bsz * max(1, rem_tokens)
        return outputs, {
            "batch_size": bsz,
            "prefill_ms": round(t_prefill, 2),
            "decode_ms": round(t_decode, 2),
            "total_decode_tok_s": round((total_dec_toks / max(1e-6, t_decode)) * 1000.0, 2),
            "per_req_tok_s": round(((total_dec_toks / bsz) / max(1e-6, t_decode)) * 1000.0, 2),
        }

    @torch.inference_mode()
    def generate_speculative(
        self,
        prompt_ids: list[int],
        max_new_tokens: int = 256,
        gamma: int = 4,
        max_ngram: int = 3,
    ) -> tuple[list[int], dict[str, Any]]:
        """Native speculative decoding with N-gram prompt lookup + multi-token CuTile verification."""
        from cutile_forge.engine.speculative import NGramPromptLookupDrafter

        drafter = NGramPromptLookupDrafter(max_n=max_ngram)
        logit_cache: dict[tuple[int, ...], list[int]] = {}

        t0 = time.perf_counter()
        prompt_t = torch.tensor([prompt_ids], dtype=torch.long, device=self.device)
        first_tok_t = self.prefill(prompt_t)
        if self.use_cuda:
            torch.cuda.synchronize()
        t_prefill = (time.perf_counter() - t0) * 1000.0

        cur_tok = int(first_tok_t.item())
        history: list[int] = list(prompt_ids) + [cur_tok]
        generated: list[int] = [cur_tok]
        pos = len(prompt_ids)

        # Warmup graphed single-step fallback before timing
        if self.use_cuda_graph and self._all_layers_on_gpu:
            ws = self._get_or_create_workspace(1)
            for b in self.context_buckets:
                if pos < b:
                    _ = self._capture_full_step_graph(ws, b)
                if b >= pos + max_new_tokens:
                    break

        draft_steps = 0
        accepted_total = 0
        total_drafted = 0

        t_dec0 = time.perf_counter()
        while len(generated) < max_new_tokens:
            draft_steps += 1
            draft_toks = drafter.draft(history, gamma)
            if not draft_toks and len(history) >= 3:
                key3 = tuple(history[-3:])
                if key3 in logit_cache:
                    draft_toks = list(logit_cache[key3][:gamma])

            if not draft_toks:
                # Zero-overhead single-step graphed decode when no N-gram candidate exists
                if self.use_cuda_graph and self._all_layers_on_gpu:
                    ws = self._get_or_create_workspace(1)
                    ws.token_id.fill_(cur_tok)
                    ws.pos_buf.fill_(pos)
                    bucket_len = self._select_bucket(pos)
                    ws.graphs_by_bucket[bucket_len].replay()
                    nxt = int(ws.token_id.item())
                else:
                    t_in = torch.tensor([[cur_tok]], dtype=torch.int32, device=self.device)
                    nxt = int(self.decode_step(t_in, pos=pos, use_graph=False).item())
                pos += 1
                self.kv_cache.seq_lens = [pos] * self.config.num_hidden_layers
                generated.append(nxt)
                history.append(nxt)
                cur_tok = nxt
                continue

            k_draft = len(draft_toks)
            total_drafted += k_draft
            verify_seq = [cur_tok] + draft_toks
            s_v = len(verify_seq)
            v_ids = torch.tensor(verify_seq, dtype=torch.int32, device=self.device)

            h = cutile_embedding_lookup(v_ids, self._embed_w).view(s_v, self.config.hidden_size)
            for idx in range(self.config.num_hidden_layers):
                h = self._forward_layer_native(idx, h, pos_offset=pos, seq_len=s_v, batch_size=1)
            _, pred_toks_t = cutile_lm_head_fused(
                h, self._norm_w, self._lm_head_w, eps=self.config.rms_norm_eps, top_k=1
            )
            target_preds = pred_toks_t.view(s_v).cpu().tolist()

            n_accept = 0
            for j, d_tok in enumerate(draft_toks):
                if int(target_preds[j]) == d_tok:
                    n_accept += 1
                    generated.append(d_tok)
                    history.append(d_tok)
                    if len(generated) >= max_new_tokens:
                        break
                else:
                    tok = int(target_preds[j])
                    generated.append(tok)
                    history.append(tok)
                    break
            else:
                if len(generated) < max_new_tokens:
                    tok = int(target_preds[k_draft])
                    generated.append(tok)
                    history.append(tok)

            if len(history) >= 4 + gamma:
                logit_cache[tuple(history[-(3 + gamma) : -gamma])] = list(history[-gamma:])

            accepted_total += n_accept
            pos += n_accept + 1
            self.kv_cache.truncate(pos)
            cur_tok = generated[-1]

        if self.use_cuda:
            torch.cuda.synchronize()
        t_decode = (time.perf_counter() - t_dec0) * 1000.0
        gen_n = max(1, len(generated) - 1)
        return generated[:max_new_tokens], {
            "tokens_generated": min(len(generated), max_new_tokens),
            "draft_steps": draft_steps,
            "total_drafted": total_drafted,
            "accepted_tokens": accepted_total,
            "acceptance_rate": round(accepted_total / float(max(1, total_drafted)), 4),
            "prefill_ms": round(t_prefill, 2),
            "decode_ms": round(t_decode, 2),
            "decode_tok_s": round((gen_n / max(1e-6, t_decode)) * 1000.0, 2),
            "ms_per_token": round(t_decode / gen_n, 2),
        }

    @torch.inference_mode()
    def benchmark_prefill_decode_interleave(
        self,
        decode_prompt_ids: list[int],
        incoming_prefill_ids: list[int],
        decode_steps: int = 8,
        chunk_size: int | None = 128,
    ) -> dict[str, float]:
        """Measure active decode inter-token latency when an incoming prefill arrives mid-decode.

        When `chunk_size is None`, the incoming prefill runs monolithically after decode step 2,
        stalling step 3. When `chunk_size` is set (e.g. 128), prefill slices interleave with decode steps.
        """
        # Create a second independent KV cache for the incoming prefill request so both coexist cleanly
        prefill_kv = NativePagedKVCache(
            num_layers=self.config.num_hidden_layers,
            num_kv_heads=self.config.num_key_value_heads,
            head_dim=self.config.head_dim,
            block_size=self.block_size,
            max_seq_len=self.max_seq_len,
            kv_dtype=self.kv_cache.kv_dtype,
            device=self.device,
            stream_manager=self.stream_manager,
        )

        prompt_t = torch.tensor([decode_prompt_ids], dtype=torch.long, device=self.device)
        cur_tok = self.prefill(prompt_t)
        cur_pos = len(decode_prompt_ids)
        if self.use_cuda_graph and self._all_layers_on_gpu:
            ws = self._get_or_create_workspace(1)
            _ = self._capture_full_step_graph(ws, self._select_bucket(cur_pos + decode_steps))
            ws.token_id.copy_(cur_tok.view(1, 1).to(torch.int32))
            ws.pos_buf.fill_(cur_pos)
        if self.use_cuda:
            torch.cuda.synchronize()

        prefill_t = torch.tensor([incoming_prefill_ids], dtype=torch.int32, device=self.device)
        p_total = int(prefill_t.shape[1])
        p_cursor = 0
        prefill_cpu_kv: list[tuple[torch.Tensor, torch.Tensor] | None] = [
            None
        ] * self.config.num_hidden_layers

        def _run_prefill_slice(s_start: int, s_end: int) -> None:
            nonlocal prefill_cpu_kv
            sl = prefill_t[:, s_start:s_end]
            s_len = int(sl.shape[1])
            orig_kv = self.kv_cache
            orig_cpu_kv = self._cpu_kv
            self.kv_cache = prefill_kv
            self._cpu_kv = prefill_cpu_kv
            try:
                h = cutile_embedding_lookup(sl.reshape(s_len), self._embed_w).view(
                    s_len, self.config.hidden_size
                )
                for idx in range(self.config.num_hidden_layers):
                    h = self._forward_layer_native(
                        idx, h, pos_offset=s_start, seq_len=s_len, batch_size=1
                    )
            finally:
                prefill_cpu_kv = self._cpu_kv
                self.kv_cache = orig_kv
                self._cpu_kv = orig_cpu_kv

        itl_ms: list[float] = []
        for step in range(decode_steps):
            t_s = time.perf_counter()
            if step >= 1 and p_cursor < p_total:
                if chunk_size is None:
                    _run_prefill_slice(0, p_total)
                    p_cursor = p_total
                else:
                    s_end = min(p_total, p_cursor + chunk_size)
                    _run_prefill_slice(p_cursor, s_end)
                    p_cursor = s_end
            cur_tok = self.decode_step(cur_tok, pos=cur_pos, sync_pos=False)
            cur_pos += 1
            if self.use_cuda:
                torch.cuda.synchronize()
            itl_ms.append((time.perf_counter() - t_s) * 1000.0)

        while p_cursor < p_total:
            s_end = min(p_total, p_cursor + (chunk_size or p_total))
            _run_prefill_slice(p_cursor, s_end)
            p_cursor = s_end
        if self.use_cuda:
            torch.cuda.synchronize()

        import statistics

        return {
            "max_itl_ms": round(max(itl_ms), 2),
            "median_itl_ms": round(statistics.median(itl_ms), 2),
            "total_schedule_ms": round(sum(itl_ms), 2),
        }
