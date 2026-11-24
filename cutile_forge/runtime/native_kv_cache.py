"""Native paged KV-cache manager with head-major physical blocks & device-position in-place write."""

from __future__ import annotations

import torch

from cutile_forge.cuda_tile_init import ct
from cutile_forge.runtime.memory import PinnedHostPool
from cutile_forge.runtime.native_stream import NativeStreamManager


@ct.kernel(occupancy=4)
def cutile_kv_cache_write_hm_kernel(
    k_src,  # (B * H_kv, S_new, D)
    v_src,  # (B * H_kv, S_new, D)
    k_pool,  # (B * H_kv, S_total, D)
    v_pool,  # (B * H_kv, S_total, D)
    slots_ptr,  # (S_new,) int32
    d_head: ct.Constant[int],
) -> None:
    """Write `(B * H_kv, S_new, D)` K/V slices directly into head-major physical HBM blocks."""
    tok_id = ct.bid(0)
    bh_id = ct.bid(1)

    slot_val = ct.load(slots_ptr, index=(tok_id,), shape=(1,), padding_mode=ct.PaddingMode.ZERO)
    slot_3d = ct.reshape(ct.astype(slot_val, ct.int32), (1, 1, 1))
    bh_3d = ct.reshape(ct.full((1,), bh_id, dtype=ct.int32), (1, 1, 1))
    d_3d = ct.reshape(ct.arange(d_head, dtype=ct.int32), (1, 1, d_head))

    k_t = ct.load(
        k_src,
        index=(bh_id, tok_id, 0),
        shape=(1, 1, d_head),
        padding_mode=ct.PaddingMode.ZERO,
        latency=3,
    )
    v_t = ct.load(
        v_src,
        index=(bh_id, tok_id, 0),
        shape=(1, 1, d_head),
        padding_mode=ct.PaddingMode.ZERO,
        latency=3,
    )

    ct.scatter(k_pool, (bh_3d, slot_3d, d_3d), ct.astype(k_t, k_pool.dtype))
    ct.scatter(v_pool, (bh_3d, slot_3d, d_3d), ct.astype(v_t, v_pool.dtype))


@ct.kernel(occupancy=4)
def cutile_kv_cache_gather_hm_kernel(
    k_pool,  # (B * H_kv, S_total, D)
    v_pool,  # (B * H_kv, S_total, D)
    block_table_ptr,  # (B, max_blocks) int32
    k_out,  # (B * H_kv, seq_len, D)
    v_out,  # (B * H_kv, seq_len, D)
    seq_len: int,
    h_kv: ct.Constant[int],
    block_size: ct.Constant[int],
    tile_s: ct.Constant[int],
    d_head: ct.Constant[int],
) -> None:
    """Gather paged K/V blocks into contiguous `(B * H_kv, seq_len, D)` buffers when needed."""
    ts_id = ct.bid(0)
    bh_id = ct.bid(1)
    b_idx = bh_id // h_kv

    s_ids = ts_id * tile_s + ct.arange(tile_s, dtype=ct.int32)
    in_seq = s_ids < seq_len
    s_safe = ct.where(in_seq, s_ids, 0)
    blk_idx = ct.reshape(s_safe // block_size, (1, tile_s))
    blk_off = ct.reshape(s_safe % block_size, (1, tile_s))
    b_row = ct.reshape(ct.full((1,), b_idx, dtype=ct.int32), (1, 1))

    phys_blk = ct.astype(ct.gather(block_table_ptr, (b_row, blk_idx), padding_value=0), ct.int32)
    slots = phys_blk * block_size + blk_off

    slot_3d = ct.reshape(slots, (1, tile_s, 1))
    bh_3d = ct.reshape(ct.full((1,), bh_id, dtype=ct.int32), (1, 1, 1))
    d_3d = ct.reshape(ct.arange(d_head, dtype=ct.int32), (1, 1, d_head))

    kv = ct.gather(k_pool, (bh_3d, slot_3d, d_3d), padding_value=0, latency=3)
    vv = ct.gather(v_pool, (bh_3d, slot_3d, d_3d), padding_value=0, latency=3)

    ct.store(k_out, index=(bh_id, ts_id, 0), tile=ct.astype(kv, k_out.dtype))
    ct.store(v_out, index=(bh_id, ts_id, 0), tile=ct.astype(vv, v_out.dtype))


def cutile_kv_cache_write(
    k_new: torch.Tensor,
    v_new: torch.Tensor,
    k_pool: torch.Tensor,
    v_pool: torch.Tensor,
    slot_mapping: torch.Tensor,
) -> None:
    """Launch in-place CuTile paged KV-cache write for `(B, H_kv, S_new, D)` into head-major pool."""
    bsz, h_kv, s_new, d_head = k_new.shape
    total_slots = k_pool.shape[2]
    k_src = k_new.contiguous().view(bsz * h_kv, s_new, d_head)
    v_src = v_new.contiguous().view(bsz * h_kv, s_new, d_head)
    k_flat = k_pool[:bsz].view(bsz * h_kv, total_slots, d_head)
    v_flat = v_pool[:bsz].view(bsz * h_kv, total_slots, d_head)

    if not k_new.is_cuda:
        for i in range(s_new):
            sl = int(slot_mapping[i].item())
            k_flat[:, sl, :].copy_(k_src[:, i, :].to(k_pool.dtype))
            v_flat[:, sl, :].copy_(v_src[:, i, :].to(v_pool.dtype))
        return

    ct.launch(
        torch.cuda.current_stream(),
        (s_new, bsz * h_kv, 1),
        cutile_kv_cache_write_hm_kernel,
        (k_src, v_src, k_flat, v_flat, slot_mapping.contiguous(), d_head),
    )


class NativePagedKVCache:
    """Zero-cat paged KV-cache with head-major `(B, H_kv, S_total, D)` physical block pools."""

    def __init__(
        self,
        num_layers: int,
        num_kv_heads: int,
        head_dim: int,
        block_size: int = 16,
        max_seq_len: int = 2048,
        max_batch_size: int = 1,
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
        self.max_batch_size = max(1, max_batch_size)
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
        self.total_slots = self.blocks_per_layer * block_size

        self.k_pools: list[torch.Tensor] = [
            torch.zeros(
                (self.max_batch_size, num_kv_heads, self.total_slots, head_dim),
                device=self.device,
                dtype=self.storage_dtype,
            )
            for _ in range(num_layers)
        ]
        self.v_pools: list[torch.Tensor] = [
            torch.zeros(
                (self.max_batch_size, num_kv_heads, self.total_slots, head_dim),
                device=self.device,
                dtype=self.storage_dtype,
            )
            for _ in range(num_layers)
        ]
        base_table = (
            torch.arange(self.blocks_per_layer, device=self.device, dtype=torch.int32)
            .unsqueeze(0)
            .expand(self.max_batch_size, -1)
            .contiguous()
        )
        self.block_tables: list[torch.Tensor] = [base_table.clone() for _ in range(num_layers)]
        self._is_linear_mapping: bool = True
        self.seq_lens: list[int] = [0] * num_layers

        # Preallocated linear slot index table [0, 1, ..., total_slots - 1] (zero runtime fill/arange)
        self._linear_slots = torch.arange(self.total_slots, device=self.device, dtype=torch.int32)

        # Lazy scratchpad only allocated if FP16 dequantization or non-linear gather is requested
        self._k_work: list[torch.Tensor] = []
        self._v_work: list[torch.Tensor] = []

    def reset(self) -> None:
        self.seq_lens = [0] * self.num_layers

    def truncate(self, new_len: int) -> None:
        """Roll back KV sequence length across all layers to `new_len` (O(1) speculative rollback)."""
        clamped = max(0, min(new_len, self.max_seq_len))
        self.seq_lens = [clamped] * self.num_layers

    def ensure_batch_capacity(self, batch_size: int) -> None:
        """Expand physical KV pools if `batch_size > self.max_batch_size`."""
        if batch_size <= self.max_batch_size:
            return
        self.max_batch_size = batch_size
        self.k_pools = [
            torch.zeros(
                (batch_size, self.num_kv_heads, self.total_slots, self.head_dim),
                device=self.device,
                dtype=self.storage_dtype,
            )
            for _ in range(self.num_layers)
        ]
        self.v_pools = [
            torch.zeros(
                (batch_size, self.num_kv_heads, self.total_slots, self.head_dim),
                device=self.device,
                dtype=self.storage_dtype,
            )
            for _ in range(self.num_layers)
        ]
        base_table = (
            torch.arange(self.blocks_per_layer, device=self.device, dtype=torch.int32)
            .unsqueeze(0)
            .expand(batch_size, -1)
            .contiguous()
        )
        self.block_tables = [base_table.clone() for _ in range(self.num_layers)]

    def get_kv(
        self,
        layer_idx: int,
        end_pos: int | None = None,
        batch_size: int = 1,
        allow_native_dtype: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return `(B, H_kv, S_kv, D)` K and V views (FP16 by default, or native dtype if allowed)."""
        seq_len = self.seq_lens[layer_idx] if end_pos is None else end_pos
        if self._is_linear_mapping and (self.storage_dtype == torch.float16 or allow_native_dtype):
            return (
                self.k_pools[layer_idx][:batch_size, :, :seq_len, :],
                self.v_pools[layer_idx][:batch_size, :, :seq_len, :],
            )

        if not self._k_work:
            work_elems = self.max_batch_size * self.num_kv_heads * self.max_seq_len * self.head_dim
            self._k_work = [
                torch.empty((work_elems,), device=self.device, dtype=torch.float16)
                for _ in range(self.num_layers)
            ]
            self._v_work = [
                torch.empty((work_elems,), device=self.device, dtype=torch.float16)
                for _ in range(self.num_layers)
            ]

        n_elems = batch_size * self.num_kv_heads * seq_len * self.head_dim
        k_out = self._k_work[layer_idx][:n_elems].view(
            batch_size * self.num_kv_heads, seq_len, self.head_dim
        )
        v_out = self._v_work[layer_idx][:n_elems].view(
            batch_size * self.num_kv_heads, seq_len, self.head_dim
        )

        if not self.use_cuda:
            bt = self.block_tables[layer_idx]
            for b in range(batch_size):
                for s in range(seq_len):
                    sl = int(bt[b, s // self.block_size].item()) * self.block_size + (
                        s % self.block_size
                    )
                    k_out[b * self.num_kv_heads : (b + 1) * self.num_kv_heads, s, :].copy_(
                        self.k_pools[layer_idx][b, :, sl, :].to(torch.float16)
                    )
                    v_out[b * self.num_kv_heads : (b + 1) * self.num_kv_heads, s, :].copy_(
                        self.v_pools[layer_idx][b, :, sl, :].to(torch.float16)
                    )
            return (
                k_out.view(batch_size, self.num_kv_heads, seq_len, self.head_dim),
                v_out.view(batch_size, self.num_kv_heads, seq_len, self.head_dim),
            )

        tile_s = 16
        ct.launch(
            torch.cuda.current_stream(),
            (ct.cdiv(seq_len, tile_s), batch_size * self.num_kv_heads, 1),
            cutile_kv_cache_gather_hm_kernel,
            (
                self.k_pools[layer_idx][:batch_size].view(
                    batch_size * self.num_kv_heads, self.total_slots, self.head_dim
                ),
                self.v_pools[layer_idx][:batch_size].view(
                    batch_size * self.num_kv_heads, self.total_slots, self.head_dim
                ),
                self.block_tables[layer_idx][:batch_size],
                k_out,
                v_out,
                seq_len,
                self.num_kv_heads,
                self.block_size,
                tile_s,
                self.head_dim,
            ),
        )
        return (
            k_out.view(batch_size, self.num_kv_heads, seq_len, self.head_dim),
            v_out.view(batch_size, self.num_kv_heads, seq_len, self.head_dim),
        )

    def append_and_get(
        self,
        layer_idx: int,
        k_new: torch.Tensor,
        v_new: torch.Tensor,
        start_pos: int | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Append `(B, H_kv, S_new, D)` in-place and return `(B, H_kv, S_tot, D)` FP16 views."""
        bsz = int(k_new.shape[0])
        self.ensure_batch_capacity(bsz)
        cur = self.seq_lens[layer_idx] if start_pos is None else start_pos
        s_new = int(k_new.shape[2])
        end_pos = cur + s_new
        if end_pos > self.max_seq_len:
            raise RuntimeError(
                f"KV sequence length {end_pos} exceeds max_seq_len={self.max_seq_len}"
            )

        slots = self._linear_slots[cur:end_pos]
        cutile_kv_cache_write(k_new, v_new, self.k_pools[layer_idx], self.v_pools[layer_idx], slots)
        self.seq_lens[layer_idx] = end_pos
        return self.get_kv(layer_idx, end_pos=end_pos, batch_size=bsz)
