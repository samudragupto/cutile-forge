"""Distributed GPU Runtime (Phase 12): TP, PP, EP, Multi-GPU KV-Cache, Scheduler & Transport.

Supports both multi-GPU physical execution and single-GPU logical-partition + loopback validation.
Every performance telemetry record produced on a 1-GPU host is explicitly tagged `mode="simulated"`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

import torch
import torch.nn.functional as F

from cutile_forge.distributed.p2p import p2p_copy, ring_allreduce_sum
from cutile_forge.kernels import cutile_gemm_fp16, cutile_rmsnorm, cutile_silu
from cutile_forge.runtime.native_kv_cache import NativePagedKVCache

# ---------------------------------------------------------------------------
# 1. Multi-Node / Multi-Rank Communication Layer (`CommTransport`)
# ---------------------------------------------------------------------------


class CommTransport(Protocol):
    """Clean communication abstraction for multi-GPU / multi-node collectives & P2P messaging."""

    world_size: int
    num_nodes: int

    def all_reduce_sum(self, rank_tensors: list[torch.Tensor]) -> list[torch.Tensor]: ...

    def all_gather(self, rank_tensors: list[torch.Tensor], dim: int = -1) -> list[torch.Tensor]: ...

    def reduce_scatter_sum(
        self, rank_tensors: list[torch.Tensor], dim: int = -1
    ) -> list[torch.Tensor]: ...

    def all_to_all(self, send_matrix: list[list[torch.Tensor]]) -> list[list[torch.Tensor]]: ...

    def send_recv(self, tensor: torch.Tensor, src_rank: int, dst_rank: int) -> torch.Tensor: ...


@dataclass
class LoopbackCommTransport:
    """In-process loopback transport for logical multi-rank & multi-node correctness validation."""

    world_size: int = 4
    num_nodes: int = 1
    rank_devices: list[torch.device] = field(default_factory=list)
    bytes_transferred: int = 0
    collective_calls: int = 0

    def __post_init__(self) -> None:
        if not self.rank_devices:
            dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.rank_devices = [dev for _ in range(self.world_size)]

    def node_of_rank(self, rank: int) -> int:
        ranks_per_node = max(1, self.world_size // max(1, self.num_nodes))
        return min(self.num_nodes - 1, rank // ranks_per_node)

    def all_reduce_sum(self, rank_tensors: list[torch.Tensor]) -> list[torch.Tensor]:
        self.collective_calls += 1
        if len(rank_tensors) > 1:
            nbytes = rank_tensors[0].numel() * rank_tensors[0].element_size()
            self.bytes_transferred += 2 * (len(rank_tensors) - 1) * nbytes
        return ring_allreduce_sum(rank_tensors)

    def all_gather(self, rank_tensors: list[torch.Tensor], dim: int = -1) -> list[torch.Tensor]:
        self.collective_calls += 1
        if not rank_tensors:
            return []
        n = len(rank_tensors)
        nbytes = rank_tensors[0].numel() * rank_tensors[0].element_size()
        self.bytes_transferred += (n - 1) * n * nbytes
        out: list[torch.Tensor] = []
        for r in range(n):
            dev = rank_tensors[r].device
            parts = [p2p_copy(t, dev) for t in rank_tensors]
            out.append(torch.cat(parts, dim=dim))
        return out

    def reduce_scatter_sum(
        self, rank_tensors: list[torch.Tensor], dim: int = -1
    ) -> list[torch.Tensor]:
        self.collective_calls += 1
        n = len(rank_tensors)
        reduced = ring_allreduce_sum(rank_tensors)
        out: list[torch.Tensor] = []
        for r in range(n):
            chunks = torch.chunk(reduced[r], n, dim=dim)
            out.append(chunks[r].contiguous())
        return out

    def all_to_all(self, send_matrix: list[list[torch.Tensor]]) -> list[list[torch.Tensor]]:
        """Route `send_matrix[src_rank][dst_rank]` -> `recv_matrix[dst_rank][src_rank]`."""
        self.collective_calls += 1
        n = len(send_matrix)
        recv_matrix: list[list[torch.Tensor]] = [
            [torch.empty(0) for _ in range(n)] for _ in range(n)
        ]
        for src in range(n):
            for dst in range(n):
                t = send_matrix[src][dst]
                if src != dst:
                    self.bytes_transferred += t.numel() * t.element_size()
                recv_matrix[dst][src] = p2p_copy(t, self.rank_devices[dst])
        return recv_matrix

    def send_recv(self, tensor: torch.Tensor, src_rank: int, dst_rank: int) -> torch.Tensor:
        self.collective_calls += 1
        if src_rank != dst_rank:
            self.bytes_transferred += tensor.numel() * tensor.element_size()
        return p2p_copy(tensor, self.rank_devices[dst_rank])


def _linear_op(x: torch.Tensor, w: torch.Tensor, bias: torch.Tensor | None = None) -> torch.Tensor:
    orig_prefix = x.shape[:-1]
    x_2d = x.reshape(-1, x.shape[-1])
    if (
        x_2d.is_cuda
        and x_2d.dtype == torch.float16
        and w.dtype == torch.float16
        and x_2d.shape[1] >= 16
        and w.shape[1] >= 16
    ):
        out = cutile_gemm_fp16(x_2d, w)
    else:
        out = torch.matmul(x_2d.float(), w.float()).to(x.dtype)
    if bias is not None:
        out = out + bias
    return out.view(*orig_prefix, w.shape[-1])


# ---------------------------------------------------------------------------
# 2. Tensor Parallelism (`ColumnParallelLinear`, `RowParallelLinear`, `TensorParallelMLP`, `TensorParallelAttention`)
# ---------------------------------------------------------------------------


class ColumnParallelLinear:
    """Megatron-style column-parallel linear sharding `(K, N)` across `tp_size` ranks (`(K, N // tp_size)`)."""

    def __init__(
        self,
        full_weight_kn: torch.Tensor,
        full_bias: torch.Tensor | None,
        transport: CommTransport,
        gather_output: bool = False,
    ) -> None:
        self.transport = transport
        self.tp_size = transport.world_size
        self.gather_output = gather_output
        k_dim, n_dim = full_weight_kn.shape
        if n_dim % self.tp_size != 0:
            raise ValueError(f"Output dim {n_dim} must be divisible by tp_size={self.tp_size}")
        self.weight_shards = [
            chunk.contiguous() for chunk in torch.chunk(full_weight_kn, self.tp_size, dim=1)
        ]
        self.bias_shards: list[torch.Tensor | None] = (
            [chunk.contiguous() for chunk in torch.chunk(full_bias, self.tp_size, dim=0)]
            if full_bias is not None
            else [None for _ in range(self.tp_size)]
        )

    def forward_ranks(self, rank_inputs: list[torch.Tensor]) -> list[torch.Tensor]:
        shards = [
            _linear_op(rank_inputs[r], self.weight_shards[r], self.bias_shards[r])
            for r in range(self.tp_size)
        ]
        if self.gather_output:
            return self.transport.all_gather(shards, dim=-1)
        return shards


class RowParallelLinear:
    """Megatron-style row-parallel linear sharding `(K, N)` across `tp_size` ranks (`(K // tp_size, N)`) + all-reduce."""

    def __init__(
        self,
        full_weight_kn: torch.Tensor,
        full_bias: torch.Tensor | None,
        transport: CommTransport,
        input_is_parallel: bool = True,
    ) -> None:
        self.transport = transport
        self.tp_size = transport.world_size
        self.input_is_parallel = input_is_parallel
        k_dim, _ = full_weight_kn.shape
        if k_dim % self.tp_size != 0:
            raise ValueError(f"Input dim {k_dim} must be divisible by tp_size={self.tp_size}")
        self.weight_shards = [
            chunk.contiguous() for chunk in torch.chunk(full_weight_kn, self.tp_size, dim=0)
        ]
        self.full_bias = full_bias

    def forward_ranks(self, rank_inputs: list[torch.Tensor]) -> list[torch.Tensor]:
        partials: list[torch.Tensor] = []
        for r in range(self.tp_size):
            x_r = (
                rank_inputs[r]
                if self.input_is_parallel
                else torch.chunk(rank_inputs[r], self.tp_size, dim=-1)[r].contiguous()
            )
            partials.append(_linear_op(x_r, self.weight_shards[r], None))
        reduced = self.transport.all_reduce_sum(partials)
        if self.full_bias is not None:
            reduced = [t + self.full_bias for t in reduced]
        return reduced


class TensorParallelSwiGLUMLP:
    """Tensor-parallel SwiGLU FFN (`gate_proj` & `up_proj` column-parallel, `down_proj` row-parallel)."""

    def __init__(
        self,
        w_gate: torch.Tensor,
        w_up: torch.Tensor,
        w_down: torch.Tensor,
        transport: CommTransport,
    ) -> None:
        self.transport = transport
        self.tp_size = transport.world_size
        self.gate_col = ColumnParallelLinear(w_gate, None, transport, gather_output=False)
        self.up_col = ColumnParallelLinear(w_up, None, transport, gather_output=False)
        self.down_row = RowParallelLinear(w_down, None, transport, input_is_parallel=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rank_x = [x for _ in range(self.tp_size)]
        gates = self.gate_col.forward_ranks(rank_x)
        ups = self.up_col.forward_ranks(rank_x)
        acts = [
            (
                cutile_silu(g) * u
                if g.is_cuda and g.dtype == torch.float16
                else (F.silu(g.float()) * u.float()).to(g.dtype)
            )
            for g, u in zip(gates, ups, strict=True)
        ]
        outs = self.down_row.forward_ranks(acts)
        return outs[0]


class TensorParallelGQAAttention:
    """Tensor-parallel GQA attention partitioning query heads (`H_q // tp_size`) and KV heads (`H_kv // tp_size`)."""

    def __init__(
        self,
        wq: torch.Tensor,
        wk: torch.Tensor,
        wv: torch.Tensor,
        wo: torch.Tensor,
        num_heads: int,
        num_kv_heads: int,
        head_dim: int,
        transport: CommTransport,
    ) -> None:
        self.transport = transport
        self.tp_size = transport.world_size
        if num_heads % self.tp_size != 0 or num_kv_heads % self.tp_size != 0:
            raise ValueError("num_heads and num_kv_heads must be divisible by tp_size")
        self.heads_per_rank = num_heads // self.tp_size
        self.kv_heads_per_rank = num_kv_heads // self.tp_size
        self.head_dim = head_dim
        self.q_col = ColumnParallelLinear(wq, None, transport, gather_output=False)
        self.k_col = ColumnParallelLinear(wk, None, transport, gather_output=False)
        self.v_col = ColumnParallelLinear(wv, None, transport, gather_output=False)
        self.o_row = RowParallelLinear(wo, None, transport, input_is_parallel=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, s, _ = x.shape
        rank_x = [x for _ in range(self.tp_size)]
        q_shards = self.q_col.forward_ranks(rank_x)
        k_shards = self.k_col.forward_ranks(rank_x)
        v_shards = self.v_col.forward_ranks(rank_x)

        rep = self.heads_per_rank // self.kv_heads_per_rank
        attn_shards: list[torch.Tensor] = []
        for r in range(self.tp_size):
            qr = q_shards[r].view(b, s, self.heads_per_rank, self.head_dim).transpose(1, 2)
            kr = k_shards[r].view(b, s, self.kv_heads_per_rank, self.head_dim).transpose(1, 2)
            vr = v_shards[r].view(b, s, self.kv_heads_per_rank, self.head_dim).transpose(1, 2)
            if rep > 1:
                kr = kr.repeat_interleave(rep, dim=1)
                vr = vr.repeat_interleave(rep, dim=1)
            ao = F.scaled_dot_product_attention(
                qr.float(), kr.float(), vr.float(), is_causal=True
            ).to(x.dtype)
            attn_shards.append(
                ao.transpose(1, 2).contiguous().view(b, s, self.heads_per_rank * self.head_dim)
            )
        return self.o_row.forward_ranks(attn_shards)[0]


# ---------------------------------------------------------------------------
# 3. Pipeline Parallelism (`PipelineStage`, `PipelineParallelEngine`)
# ---------------------------------------------------------------------------


@dataclass
class PipelineStage:
    """Single stage in a pipeline-parallel execution schedule holding a contiguous slice of layers."""

    stage_id: int
    layer_indices: list[int]
    layer_blocks: list[dict[str, torch.Tensor]]
    eps: float = 1e-5

    def forward_microbatch(self, h: torch.Tensor) -> torch.Tensor:
        cur = h
        for bw in self.layer_blocks:
            if cur.is_cuda and cur.dtype == torch.float16:
                n1 = cutile_rmsnorm(cur, bw["in_norm_w"], eps=self.eps)
            else:
                v1 = cur.float().pow(2).mean(dim=-1, keepdim=True)
                n1 = (cur.float() * torch.rsqrt(v1 + self.eps) * bw["in_norm_w"].float()).to(
                    cur.dtype
                )
            cur = cur + _linear_op(n1, bw["w_attn"])
            if cur.is_cuda and cur.dtype == torch.float16:
                n2 = cutile_rmsnorm(cur, bw["post_norm_w"], eps=self.eps)
            else:
                v2 = cur.float().pow(2).mean(dim=-1, keepdim=True)
                n2 = (cur.float() * torch.rsqrt(v2 + self.eps) * bw["post_norm_w"].float()).to(
                    cur.dtype
                )
            g = _linear_op(n2, bw["w_gate"])
            u = _linear_op(n2, bw["w_up"])
            act = (
                cutile_silu(g) * u
                if g.is_cuda and g.dtype == torch.float16
                else (F.silu(g.float()) * u.float()).to(g.dtype)
            )
            cur = cur + _linear_op(act, bw["w_down"])
        return cur


class PipelineParallelEngine:
    """GPipe / 1F1B micro-batch pipeline-parallel inference engine over `pp_size` logical stages."""

    def __init__(
        self,
        all_layers: list[dict[str, torch.Tensor]],
        transport: CommTransport,
        num_microbatches: int = 4,
    ) -> None:
        self.transport = transport
        self.pp_size = transport.world_size
        self.num_microbatches = num_microbatches
        total_l = len(all_layers)
        if total_l % self.pp_size != 0:
            raise ValueError(f"Total layers {total_l} must be divisible by pp_size={self.pp_size}")
        per_stage = total_l // self.pp_size
        self.stages = [
            PipelineStage(
                stage_id=s,
                layer_indices=list(range(s * per_stage, (s + 1) * per_stage)),
                layer_blocks=all_layers[s * per_stage : (s + 1) * per_stage],
            )
            for s in range(self.pp_size)
        ]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mbs = list(torch.chunk(x, self.num_microbatches, dim=0))
        for s_idx, stage in enumerate(self.stages):
            next_mbs: list[torch.Tensor] = []
            for mb in mbs:
                out_mb = stage.forward_microbatch(mb)
                if s_idx + 1 < self.pp_size:
                    out_mb = self.transport.send_recv(out_mb, s_idx, s_idx + 1)
                next_mbs.append(out_mb)
            mbs = next_mbs
        return torch.cat(mbs, dim=0)

    def forward_single_device(self, x: torch.Tensor) -> torch.Tensor:
        single_stage = PipelineStage(
            stage_id=0,
            layer_indices=list(range(sum(len(s.layer_blocks) for s in self.stages))),
            layer_blocks=[bw for s in self.stages for bw in s.layer_blocks],
        )
        mbs = list(torch.chunk(x, self.num_microbatches, dim=0))
        return torch.cat([single_stage.forward_microbatch(mb) for mb in mbs], dim=0)


# ---------------------------------------------------------------------------
# 4. Expert Parallelism (`ExpertParallelMoE`) with All-to-All Token Dispatch & Combine
# ---------------------------------------------------------------------------


class ExpertParallelMoE:
    """Distributed Expert-Parallel MoE layer partitioning `E` experts across `ep_size` ranks via `all_to_all`."""

    def __init__(
        self,
        router_w: torch.Tensor,
        experts: list[dict[str, torch.Tensor]],
        top_k: int,
        transport: CommTransport,
    ) -> None:
        self.router_w = router_w
        self.experts = experts
        self.num_experts = len(experts)
        self.top_k = top_k
        self.transport = transport
        self.ep_size = transport.world_size
        if self.num_experts % self.ep_size != 0:
            raise ValueError(
                f"num_experts={self.num_experts} must be divisible by ep_size={self.ep_size}"
            )
        self.experts_per_rank = self.num_experts // self.ep_size

    def forward_ep(self, x: torch.Tensor) -> torch.Tensor:
        orig_shape = x.shape
        x_flat = x.reshape(-1, x.shape[-1])
        num_tokens, d = x_flat.shape

        logits = torch.matmul(x_flat.float(), self.router_w.float())
        topk_vals, topk_idx = torch.topk(logits, k=self.top_k, dim=-1)
        topk_w = torch.softmax(topk_vals, dim=-1).to(x.dtype)

        # Partition input tokens across `ep_size` source ranks
        tok_chunks = torch.chunk(x_flat, self.ep_size, dim=0)
        idx_chunks = torch.chunk(topk_idx, self.ep_size, dim=0)
        w_chunks = torch.chunk(topk_w, self.ep_size, dim=0)

        # 1. All-to-all dispatch: send token representations from each src_rank to each dst_rank
        send_tokens: list[list[torch.Tensor]] = [
            [tok_chunks[src] for _ in range(self.ep_size)] for src in range(self.ep_size)
        ]
        recv_tokens = self.transport.all_to_all(send_tokens)

        # 2. Each EP rank evaluates only its local experts `[dst * E_local, (dst + 1) * E_local)`
        reply_matrix: list[list[torch.Tensor]] = [
            [torch.empty(0) for _ in range(self.ep_size)] for _ in range(self.ep_size)
        ]
        for dst in range(self.ep_size):
            e_start = dst * self.experts_per_rank
            e_end = e_start + self.experts_per_rank
            for src in range(self.ep_size):
                t_in = recv_tokens[dst][src]
                t_idx = idx_chunks[src]
                t_w = w_chunks[src]
                acc = torch.zeros_like(t_in, dtype=torch.float32)
                for e_id in range(e_start, e_end):
                    mask = t_idx == e_id
                    if not torch.any(mask):
                        continue
                    ew = self.experts[e_id]
                    weight_col = (t_w * mask.to(t_w.dtype)).sum(dim=-1, keepdim=True)
                    active = weight_col.squeeze(-1) > 0
                    sub = t_in[active]
                    g = _linear_op(sub, ew["w_gate"])
                    u = _linear_op(sub, ew["w_up"])
                    act = (
                        cutile_silu(g) * u
                        if g.is_cuda and g.dtype == torch.float16
                        else (F.silu(g.float()) * u.float()).to(g.dtype)
                    )
                    out_e = _linear_op(act, ew["w_down"])
                    acc[active] = acc[active] + (out_e * weight_col[active]).float()
                reply_matrix[dst][src] = acc.to(x.dtype)

        # 3. All-to-all combine: return partial outputs from `dst` back to `src`
        combined_recv = self.transport.all_to_all(reply_matrix)
        rank_outputs: list[torch.Tensor] = []
        for src in range(self.ep_size):
            total_src = torch.zeros_like(tok_chunks[src], dtype=torch.float32)
            for dst in range(self.ep_size):
                total_src = total_src + combined_recv[src][dst].float()
            rank_outputs.append(total_src.to(x.dtype))

        return torch.cat(rank_outputs, dim=0).view(orig_shape)

    def forward_single_device(self, x: torch.Tensor) -> torch.Tensor:
        orig_shape = x.shape
        x_flat = x.reshape(-1, x.shape[-1])
        logits = torch.matmul(x_flat.float(), self.router_w.float())
        topk_vals, topk_idx = torch.topk(logits, k=self.top_k, dim=-1)
        topk_w = torch.softmax(topk_vals, dim=-1).to(x.dtype)

        tok_chunks = torch.chunk(x_flat, self.ep_size, dim=0)
        idx_chunks = torch.chunk(topk_idx, self.ep_size, dim=0)
        w_chunks = torch.chunk(topk_w, self.ep_size, dim=0)
        rank_outputs: list[torch.Tensor] = []
        for src in range(self.ep_size):
            t_in = tok_chunks[src]
            t_idx = idx_chunks[src]
            t_w = w_chunks[src]
            total_src = torch.zeros_like(t_in, dtype=torch.float32)
            for dst in range(self.ep_size):
                e_start = dst * self.experts_per_rank
                e_end = e_start + self.experts_per_rank
                acc = torch.zeros_like(t_in, dtype=torch.float32)
                for e_id in range(e_start, e_end):
                    mask = t_idx == e_id
                    if not torch.any(mask):
                        continue
                    ew = self.experts[e_id]
                    weight_col = (t_w * mask.to(t_w.dtype)).sum(dim=-1, keepdim=True)
                    active = weight_col.squeeze(-1) > 0
                    sub = t_in[active]
                    g = _linear_op(sub, ew["w_gate"])
                    u = _linear_op(sub, ew["w_up"])
                    act = (
                        cutile_silu(g) * u
                        if g.is_cuda and g.dtype == torch.float16
                        else (F.silu(g.float()) * u.float()).to(g.dtype)
                    )
                    out_e = _linear_op(act, ew["w_down"])
                    acc[active] = acc[active] + (out_e * weight_col[active]).float()
                total_src = total_src + acc.to(x.dtype).float()
            rank_outputs.append(total_src.to(x.dtype))
        return torch.cat(rank_outputs, dim=0).view(orig_shape)


# ---------------------------------------------------------------------------
# 5. Multi-GPU Distributed KV-Cache & Distributed Request Scheduler
# ---------------------------------------------------------------------------


class DistributedPagedKVCache:
    """Multi-rank KV-cache manager partitioning KV heads across TP ranks and layers across PP stages."""

    def __init__(
        self,
        num_layers: int,
        num_kv_heads: int,
        head_dim: int,
        tp_size: int = 2,
        pp_size: int = 1,
        block_size: int = 16,
        max_seq_len: int = 512,
        max_batch_size: int = 4,
        kv_dtype: str = "fp16",
        device: str | torch.device = "cuda",
    ) -> None:
        if num_kv_heads % tp_size != 0:
            raise ValueError(f"num_kv_heads={num_kv_heads} must be divisible by tp_size={tp_size}")
        if num_layers % pp_size != 0:
            raise ValueError(f"num_layers={num_layers} must be divisible by pp_size={pp_size}")
        self.num_layers = num_layers
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.tp_size = tp_size
        self.pp_size = pp_size
        self.layers_per_stage = num_layers // pp_size
        self.kv_heads_per_tp = num_kv_heads // tp_size
        self.dev = torch.device(
            device if (str(device) != "cuda" or torch.cuda.is_available()) else "cpu"
        )

        # Grid of `[pp_stage][tp_rank]` NativePagedKVCache instances
        self.rank_caches: list[list[NativePagedKVCache]] = [
            [
                NativePagedKVCache(
                    num_layers=self.layers_per_stage,
                    num_kv_heads=self.kv_heads_per_tp,
                    head_dim=head_dim,
                    block_size=block_size,
                    max_seq_len=max_seq_len,
                    max_batch_size=max_batch_size,
                    kv_dtype=kv_dtype,
                    device=self.dev,
                )
                for _ in range(tp_size)
            ]
            for _ in range(pp_size)
        ]

    def reset(self) -> None:
        for stage_row in self.rank_caches:
            for c in stage_row:
                c.reset()

    def append_full(self, layer_idx: int, k: torch.Tensor, v: torch.Tensor, start_pos: int) -> None:
        """Shard `(B, H_kv, S, D)` across TP ranks and route to the owning PP stage cache."""
        pp_stage = layer_idx // self.layers_per_stage
        local_layer = layer_idx % self.layers_per_stage
        k_shards = torch.chunk(k, self.tp_size, dim=1)
        v_shards = torch.chunk(v, self.tp_size, dim=1)
        for tp_rank in range(self.tp_size):
            self.rank_caches[pp_stage][tp_rank].append_and_get(
                local_layer,
                k_shards[tp_rank].contiguous(),
                v_shards[tp_rank].contiguous(),
                start_pos=start_pos,
            )

    def gather_full(
        self, layer_idx: int, seq_len: int, batch_size: int = 1
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Reconstruct full `(B, H_kv, seq_len, D)` KV tensors from TP/PP shards."""
        pp_stage = layer_idx // self.layers_per_stage
        local_layer = layer_idx % self.layers_per_stage
        k_parts: list[torch.Tensor] = []
        v_parts: list[torch.Tensor] = []
        for tp_rank in range(self.tp_size):
            kr, vr = self.rank_caches[pp_stage][tp_rank].get_kv(
                local_layer, end_pos=seq_len, batch_size=batch_size
            )
            k_parts.append(kr)
            v_parts.append(vr)
        return torch.cat(k_parts, dim=1), torch.cat(v_parts, dim=1)


@dataclass
class DistributedScheduler:
    """Topology-aware distributed request scheduler across `(DP × PP × TP)` worker groups."""

    dp_size: int = 2
    tp_size: int = 2
    pp_size: int = 1
    max_batch_per_dp: int = 8

    @property
    def total_world_size(self) -> int:
        return self.dp_size * self.tp_size * self.pp_size

    def assign_requests(self, request_lengths: list[int]) -> dict[int, list[tuple[int, int]]]:
        """Load-balance requests `(req_id, prompt_len)` across DP replicas by greedy min-token load."""
        dp_loads = [0 for _ in range(self.dp_size)]
        assignments: dict[int, list[tuple[int, int]]] = {d: [] for d in range(self.dp_size)}
        indexed = sorted(enumerate(request_lengths), key=lambda x: x[1], reverse=True)
        for req_id, length in indexed:
            best_dp = min(range(self.dp_size), key=lambda d: dp_loads[d])
            assignments[best_dp].append((req_id, length))
            dp_loads[best_dp] += length
        return assignments
