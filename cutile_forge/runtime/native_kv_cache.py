"""Native paged KV-cache manager with in-place CuTile block write & gather (zero aten.cat)."""

from __future__ import annotations

import torch

from cutile_forge.cuda_tile_init import ct
from cutile_forge.runtime.memory import PinnedHostPool
from cutile_forge.runtime.native_stream import NativeStreamManager


def _next_pow2_kv(n: int) -> int:
    p = 1
    while p < n:
        p *= 2
    return p


@ct.kernel(occupancy=4)
def cutile_kv_cache_write_kernel(
    k_src,  # (T, H_kv, D)
    v_src,  # (T, H_kv, D)
    k_pool,  # (S_total, H_kv, D)
    v_pool,  # (S_total, H_kv, D)
    slots_ptr,  # (T,) int32
    d_head: ct.Constant[int],
) -> None:
    """Write newly rotated K/V vectors directly into paged physical HBM blocks."""
    tok_id = ct.bid(0)
    h_id = ct.bid(1)

    slot_val = ct.load(slots_ptr, index=(tok_id,), shape=(1,), padding_mode=ct.PaddingMode.ZERO)
    slot_3d = ct.reshape(ct.astype(slot_val, ct.int32), (1, 1, 1))
    h_3d = ct.reshape(ct.full((1,), h_id, dtype=ct.int32), (1, 1, 1))
    d_3d = ct.reshape(ct.arange(d_head, dtype=ct.int32), (1, 1, d_head))

    k_t = ct.load(
        k_src,
        index=(tok_id, h_id, 0),
        shape=(1, 1, d_head),
        padding_mode=ct.PaddingMode.ZERO,
        latency=3,
    )
    v_t = ct.load(
        v_src,
        index=(tok_id, h_id, 0),
        shape=(1, 1, d_head),
        padding_mode=ct.PaddingMode.ZERO,
        latency=3,
    )

    ct.scatter(k_pool, (slot_3d, h_3d, d_3d), ct.astype(k_t, k_pool.dtype))
    ct.scatter(v_pool, (slot_3d, h_3d, d_3d), ct.astype(v_t, v_pool.dtype))


@ct.kernel(occupancy=4)
def cutile_kv_cache_gather_kernel(
    k_pool,  # (S_total, H_kv, D)
    v_pool,  # (S_total, H_kv, D)
    block_table_ptr,  # (max_blocks,) int32
    k_out,  # (H_kv, seq_len, D)
    v_out,  # (H_kv, seq_len, D)
    seq_len: int,
    block_size: ct.Constant[int],
    tile_s: ct.Constant[int],
    d_head: ct.Constant[int],
) -> None:
    """Gather paged K/V blocks into contiguous `(H_kv, seq_len, D)` buffers for attention."""
    ts_id = ct.bid(0)
    h_id = ct.bid(1)

    s_ids = ts_id * tile_s + ct.arange(tile_s, dtype=ct.int32)
    in_seq = s_ids < seq_len
    s_safe = ct.where(in_seq, s_ids, 0)
    blk_idx = s_safe // block_size
    blk_off = s_safe % block_size

    phys_blk = ct.gather(block_table_ptr, (blk_idx,), padding_value=0)
    slots = phys_blk * block_size + blk_off

    slot_3d = ct.reshape(slots, (1, tile_s, 1))
    h_3d = ct.reshape(ct.full((1,), h_id, dtype=ct.int32), (1, 1, 1))
    d_3d = ct.reshape(ct.arange(d_head, dtype=ct.int32), (1, 1, d_head))

    kv = ct.gather(k_pool, (slot_3d, h_3d, d_3d), padding_value=0, latency=3)
    vv = ct.gather(v_pool, (slot_3d, h_3d, d_3d), padding_value=0, latency=3)

    ct.store(k_out, index=(h_id, ts_id, 0), tile=ct.astype(kv, k_out.dtype))
    ct.store(v_out, index=(h_id, ts_id, 0), tile=ct.astype(vv, v_out.dtype))


def cutile_kv_cache_write(
    k_new: torch.Tensor,
    v_new: torch.Tensor,
    k_pool_flat: torch.Tensor,
    v_pool_flat: torch.Tensor,
    slot_mapping: torch.Tensor,
) -> None:
    """Launch in-place CuTile paged KV-cache block write for `(B, H_kv, S_new, D)` tensors."""
    _, h_kv, s_new, d_head = k_new.shape
    k_src = k_new.squeeze(0).transpose(0, 1).contiguous()  # (S_new, H_kv, D)
    v_src = v_new.squeeze(0).transpose(0, 1).contiguous()
    if not k_new.is_cuda:
        for i in range(s_new):
            sl = int(slot_mapping[i].item())
            k_pool_flat[sl].copy_(k_src[i].to(k_pool_flat.dtype))
            v_pool_flat[sl].copy_(v_src[i].to(v_pool_flat.dtype))
        return

    ct.launch(
        torch.cuda.current_stream(),
        (s_new, h_kv, 1),
        cutile_kv_cache_write_kernel,
        (k_src, v_src, k_pool_flat, v_pool_flat, slot_mapping.contiguous(), d_head),
    )


class NativePagedKVCache:
    """Zero-cat paged KV-cache backed by in-place CuTile block write & gather kernels."""

    def __init__(
        self,
        num_layers: int,
        num_kv_heads: int,
        head_dim: int,
        block_size: int = 16,
        max_seq_len: int = 2048,
        kv_dtype: str = "fp16",
        device: str | torch.device = "cuda",
        stream_manager: NativeStreamManager | None = None,
        host_pool: PinnedHostPool | None = None,
    ) -> None:
        self.num_layers = num_layers
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.block_size = block_size
        self.max_seq_len = max_seq_len
        self.kv_dtype = kv_dtype
        self.device = torch.device(
            device if (str(device) != "cuda" or torch.cuda.is_available()) else "cpu"
        )
        self.use_cuda = self.device.type == "cuda" and torch.cuda.is_available()
        self.stream_manager = stream_manager or NativeStreamManager(self.device)
        self.host_pool = host_pool or PinnedHostPool()

        self.storage_dtype = (
            torch.float8_e4m3fn
            if (kv_dtype == "fp8_e4m3fn" and hasattr(torch, "float8_e4m3fn") and self.use_cuda)
            else torch.float16
        )

        self.blocks_per_layer = (max_seq_len + block_size - 1) // block_size
        total_slots = self.blocks_per_layer * block_size

        self.k_pools: list[torch.Tensor] = [
            torch.empty(
                (total_slots, num_kv_heads, head_dim),
                device=self.device,
                dtype=self.storage_dtype,
            )
            for _ in range(num_layers)
        ]
        self.v_pools: list[torch.Tensor] = [
            torch.empty(
                (total_slots, num_kv_heads, head_dim),
                device=self.device,
                dtype=self.storage_dtype,
            )
            for _ in range(num_layers)
        ]
        self.block_tables: list[torch.Tensor] = [
            torch.arange(self.blocks_per_layer, device=self.device, dtype=torch.int32)
            for _ in range(num_layers)
        ]
        self.seq_lens: list[int] = [0] * num_layers

        # Pre-allocated contiguous work scratchpad so gather returns zero-alloc contiguous slices
        work_elems = num_kv_heads * max_seq_len * head_dim
        self._k_work: list[torch.Tensor] = [
            torch.empty((work_elems,), device=self.device, dtype=torch.float16)
            for _ in range(num_layers)
        ]
        self._v_work: list[torch.Tensor] = [
            torch.empty((work_elems,), device=self.device, dtype=torch.float16)
            for _ in range(num_layers)
        ]
        self._slot_scratch = torch.empty((max_seq_len,), device=self.device, dtype=torch.int32)

    def reset(self) -> None:
        self.seq_lens = [0] * self.num_layers

    def append_and_get(
        self,
        layer_idx: int,
        k_new: torch.Tensor,
        v_new: torch.Tensor,
        start_pos: int | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Append `(1, H_kv, S_new, D)` in-place and return `(1, H_kv, S_tot, D)` FP16 views."""
        cur = self.seq_lens[layer_idx] if start_pos is None else start_pos
        s_new = int(k_new.shape[2])
        end_pos = cur + s_new
        if end_pos > self.max_seq_len:
            raise RuntimeError(
                f"KV sequence length {end_pos} exceeds max_seq_len={self.max_seq_len}"
            )

        if s_new == 1:
            slots = self._slot_scratch[cur : cur + 1]
            # Fill single slot on device without host-sync cat
            slots.fill_(cur)
        else:
            slots = torch.arange(cur, end_pos, device=self.device, dtype=torch.int32)
            self._slot_scratch[cur:end_pos].copy_(slots)
            slots = self._slot_scratch[cur:end_pos]

        cutile_kv_cache_write(k_new, v_new, self.k_pools[layer_idx], self.v_pools[layer_idx], slots)
        self.seq_lens[layer_idx] = end_pos

        n_elems = self.num_kv_heads * end_pos * self.head_dim
        k_out_3d = self._k_work[layer_idx][:n_elems].view(self.num_kv_heads, end_pos, self.head_dim)
        v_out_3d = self._v_work[layer_idx][:n_elems].view(self.num_kv_heads, end_pos, self.head_dim)

        if not self.use_cuda:
            bt = self.block_tables[layer_idx]
            for s in range(end_pos):
                sl = int(bt[s // self.block_size].item()) * self.block_size + (s % self.block_size)
                k_out_3d[:, s, :].copy_(self.k_pools[layer_idx][sl].to(torch.float16))
                v_out_3d[:, s, :].copy_(self.v_pools[layer_idx][sl].to(torch.float16))
            return k_out_3d.unsqueeze(0), v_out_3d.unsqueeze(0)

        tile_s = 16
        ct.launch(
            torch.cuda.current_stream(),
            (ct.cdiv(end_pos, tile_s), self.num_kv_heads, 1),
            cutile_kv_cache_gather_kernel,
            (
                self.k_pools[layer_idx],
                self.v_pools[layer_idx],
                self.block_tables[layer_idx],
                k_out_3d,
                v_out_3d,
                end_pos,
                self.block_size,
                tile_s,
                self.head_dim,
            ),
        )
        return k_out_3d.unsqueeze(0), v_out_3d.unsqueeze(0)
