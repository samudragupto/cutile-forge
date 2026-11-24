"""Phase 22 PagedKVCache & attention masking correctness and boundary tests."""

from __future__ import annotations

import math

import torch

from cutile_forge.engine.kv_cache import PagedKVCache
from cutile_forge.kernels.attention_gqa import (
    flash_decoding_split_kv,
    fused_attention_gqa,
    fused_attention_gqa_decode_pos,
)
from cutile_forge.runtime.native_kv_cache import NativePagedKVCache, cutile_kv_cache_write


def _fp64_gqa_attn(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    sliding_window: int = 0,
    causal: bool = True,
) -> torch.Tensor:
    _, h_q, s_q, d = q.shape
    h_kv, s_kv = k.shape[1], k.shape[2]
    gqa = h_q // h_kv
    kf = k.double().repeat_interleave(gqa, dim=1)
    vf = v.double().repeat_interleave(gqa, dim=1)
    qf = q.double()
    scores = torch.matmul(qf, kf.transpose(-2, -1)) / math.sqrt(d)
    if causal or sliding_window > 0:
        qp = torch.arange(s_q, device=q.device).view(s_q, 1) + (s_kv - s_q)
        kp = torch.arange(s_kv, device=q.device).view(1, s_kv)
        mask = kp <= qp
        if sliding_window > 0:
            mask = mask & (kp > qp - sliding_window)
        scores = scores.masked_fill(~mask.view(1, 1, s_q, s_kv), -1e30)
    w = torch.softmax(scores, dim=-1)
    return torch.matmul(w, vf).to(q.dtype)


def test_attention_gqa_masks_nan_in_expired_sliding_window_slots() -> None:
    """Repro: expired NaN slots in physically live pages must not poison MMA output."""
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    dt = torch.float16 if dev == "cuda" else torch.float32
    torch.manual_seed(42)
    h_q, h_kv, d = 32, 8, 128
    k_long = torch.randn((1, h_kv, 64, d), device=dev, dtype=dt)
    v_long = torch.randn((1, h_kv, 64, d), device=dev, dtype=dt)
    q1 = torch.randn((1, h_q, 1, d), device=dev, dtype=dt)

    for sw in (32, 37):
        for seq_len in (sw - 1, sw, sw + 1, sw + 15, sw + 16, sw + 17):
            pc = PagedKVCache(
                num_layers=1,
                num_kv_heads=h_kv,
                head_dim=d,
                page_size=16,
                sliding_window=sw,
                device=dev,
                dtype=dt,
            )
            pc.update(0, k_long[:, :, :seq_len, :], v_long[:, :, :seq_len, :], defer_prune=False)
            min_pos = max(0, seq_len - sw)
            min_keep_page = min_pos // 16
            exp_in_kept = min_pos % 16
            expected_pages = list(range(min_keep_page, (seq_len + 15) // 16))
            actual_pages = sorted(k[1] for k in pc.pages if k[0] == 0)
            assert actual_pages == expected_pages

            if exp_in_kept > 0:
                pc.pages[(0, min_keep_page)].k[:, :, :exp_in_kept, :] = float("nan")
                pc.pages[(0, min_keep_page)].v[:, :, :exp_in_kept, :] = float("nan")

            if dev == "cuda":
                kg, vg = pc.get_kv(0)
                out_sw = fused_attention_gqa(q1, kg, vg, causal=True, sliding_window=sw)
                ref_sw = _fp64_gqa_attn(
                    q1, k_long[:, :, :seq_len, :], v_long[:, :, :seq_len, :], sliding_window=sw
                )
                assert torch.isfinite(
                    out_sw
                ).all(), f"NaN leaked for sw={sw}, L={seq_len} (exp_in_kept={exp_in_kept})"
                err = float((out_sw.float() - ref_sw.float()).abs().max().item())
                assert err < 1e-2, f"sw={sw}, L={seq_len}: err={err}"

    if dev == "cuda":
        # FlashDecoding path (s_q=1, s_kv=2049, sw=2040 -> 9 expired leading slots)
        k_fd = torch.randn((1, h_kv, 2049, d), device=dev, dtype=torch.float16)
        v_fd = torch.randn((1, h_kv, 2049, d), device=dev, dtype=torch.float16)
        ref_fd = _fp64_gqa_attn(q1, k_fd, v_fd, sliding_window=2040)
        k_fd[:, :, :9, :] = float("nan")
        v_fd[:, :, :9, :] = float("nan")
        out_fd = flash_decoding_split_kv(
            q1, k_fd, v_fd, num_splits=16, causal=True, sliding_window=2040
        )
        assert torch.isfinite(out_fd).all(), "NaN leaked into flash_decoding_split_kv output"
        assert float((out_fd.float() - ref_fd.float()).abs().max().item()) < 1e-2


def test_paged_kv_fp8_per_slot_scale_avoids_requant_drift() -> None:
    """Repro: FP8 KV page update with a later spike must not re-quantize earlier slots."""
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(42)
    h_kv, d = 8, 128
    k_seq = torch.randn((1, h_kv, 24, d), device=dev, dtype=torch.float16)
    v_seq = torch.randn((1, h_kv, 24, d), device=dev, dtype=torch.float16)
    k_seq[:, :, 10, :] *= 20.0
    v_seq[:, :, 10, :] *= 20.0

    pkv = PagedKVCache(
        num_layers=1,
        num_kv_heads=h_kv,
        head_dim=d,
        page_size=16,
        kv_dtype="fp8_e4m3fn",
        device=dev,
    )
    pkv.update(0, k_seq, v_seq)
    kg, vg = pkv.get_kv(0)

    def _ideal_per_token_fp8(t: torch.Tensor) -> torch.Tensor:
        out = []
        for idx in range(t.shape[2]):
            sl = t[:, :, idx : idx + 1, :]
            scale = max(float(sl.abs().max().item()) / 448.0, 1e-6)
            q = (sl.float() / scale).clamp(-448.0, 448.0).to(torch.float8_e4m3fn)
            out.append((q.float() * scale).to(t.dtype))
        return torch.cat(out, dim=2)

    ideal_k = _ideal_per_token_fp8(k_seq)
    ideal_v = _ideal_per_token_fp8(v_seq)

    early_k_err = float((kg[:, :, :10, :].float() - k_seq[:, :, :10, :].float()).abs().max().item())
    early_v_err = float((vg[:, :, :10, :].float() - v_seq[:, :, :10, :].float()).abs().max().item())
    assert early_k_err < 0.25, f"Earlier K tokens suffered FP8 requantization drift: {early_k_err}"
    assert early_v_err < 0.25, f"Earlier V tokens suffered FP8 requantization drift: {early_v_err}"
    assert torch.equal(kg, ideal_k), (
        f"PagedKVCache K drifted from per-token FP8 oracle: "
        f"{float((kg.float() - ideal_k.float()).abs().max().item())}"
    )
    assert torch.equal(vg, ideal_v), (
        f"PagedKVCache V drifted from per-token FP8 oracle: "
        f"{float((vg.float() - ideal_v.float()).abs().max().item())}"
    )


def test_paged_kv_invariants_sentinels_eviction_and_boundaries() -> None:
    """Verify per-position placement, sentinels, LRU eviction, NativePagedKVCache, and boundaries."""
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    dt = torch.float16 if dev == "cuda" else torch.float32
    torch.manual_seed(7)
    h_q, h_kv, d, page_sz = 32, 8, 128, 16

    k_seq = torch.randn((1, h_kv, 128, d), device=dev, dtype=dt)
    v_seq = torch.randn((1, h_kv, 128, d), device=dev, dtype=dt)
    pkv = PagedKVCache(
        num_layers=2, num_kv_heads=h_kv, head_dim=d, page_size=page_sz, device=dev, dtype=dt
    )
    for pos in range(128):
        new_len = pkv.update(0, k_seq[:, :, pos : pos + 1, :], v_seq[:, :, pos : pos + 1, :])
        assert new_len == pos + 1
        assert pkv.seq_len == pos + 1
        assert pkv.seq_lens_valid(0) == pos + 1
        kg, vg = pkv.get_kv(0)
        assert torch.equal(kg, k_seq[:, :, : pos + 1, :])
        assert torch.equal(vg, v_seq[:, :, : pos + 1, :])

    # Trailing unwritten slots + truncate + reset sentinels
    pkv_s = PagedKVCache(
        num_layers=1, num_kv_heads=h_kv, head_dim=d, page_size=page_sz, device=dev, dtype=dt
    )
    pkv_s.update(0, k_seq[:, :, :20, :], v_seq[:, :, :20, :])
    pkv_s.pages[(0, 1)].k[:, :, 4:, :] = float("nan")
    pkv_s.pages[(0, 1)].v[:, :, 4:, :] = 1e4
    kg_t, _ = pkv_s.get_kv(0)
    assert torch.equal(kg_t, k_seq[:, :, :20, :])
    pkv_s.truncate(10)
    assert (0, 1) not in pkv_s.pages
    pkv_s.pages[(0, 0)].k[:, :, 10:, :] = float("nan")
    kg_tr, _ = pkv_s.get_kv(0)
    assert torch.equal(kg_tr, k_seq[:, :, :10, :])
    pkv_s.reset()
    pkv_s.update(0, k_seq[:, :, :7, :], v_seq[:, :, :7, :])
    assert torch.equal(pkv_s.get_kv(0)[0], k_seq[:, :, :7, :])

    # Chunked prefill + decode
    cp = PagedKVCache(
        num_layers=1, num_kv_heads=h_kv, head_dim=d, page_size=page_sz, device=dev, dtype=dt
    )
    for s0, s1 in ((0, 7), (7, 18), (18, 23)):
        cp.update(0, k_seq[:, :, s0:s1, :], v_seq[:, :, s0:s1, :], defer_prune=True)
    for p in range(23, 35):
        cp.update(0, k_seq[:, :, p : p + 1, :], v_seq[:, :, p : p + 1, :])
    assert torch.equal(cp.get_kv(0)[0], k_seq[:, :, :35, :])

    # Non-linear block table gather in NativePagedKVCache (CPU and CUDA)
    npkv_nl = NativePagedKVCache(
        num_layers=1,
        num_kv_heads=h_kv,
        head_dim=d,
        block_size=16,
        max_seq_len=64,
        max_batch_size=1,
        device=dev,
    )
    npkv_nl.k_pools[0].fill_(float("nan"))
    perm = torch.tensor([[3, 1, 0, 2]], device=dev, dtype=torch.int32)
    npkv_nl.block_tables[0].copy_(perm)
    npkv_nl._is_linear_mapping = False
    slots_nl = torch.tensor(
        [int(perm[0, s // 16].item()) * 16 + (s % 16) for s in range(19)],
        device=dev,
        dtype=torch.int32,
    )
    cutile_kv_cache_write(
        k_seq[:, :, :19, :].to(torch.float16),
        v_seq[:, :, :19, :].to(torch.float16),
        npkv_nl.k_pools[0],
        npkv_nl.v_pools[0],
        slots_nl,
    )
    npkv_nl.seq_lens[0] = 19
    kg_nl, vg_nl = npkv_nl.get_kv(0, end_pos=19, batch_size=1)
    assert torch.equal(kg_nl, k_seq[:, :, :19, :].to(torch.float16))
    assert torch.equal(vg_nl, v_seq[:, :, :19, :].to(torch.float16))

    if dev != "cuda":
        return

    # LRU eviction + host fetch on GPU
    elem_bytes = 2 * h_kv * 16 * d * 2
    pkv_lru = PagedKVCache(
        num_layers=1,
        num_kv_heads=h_kv,
        head_dim=d,
        page_size=16,
        gpu_budget_bytes=elem_bytes * 2,
        device="cuda",
    )
    pkv_lru.update(0, k_seq[:, :, :80, :], v_seq[:, :, :80, :])
    kg_lru, vg_lru = pkv_lru.get_kv(0)
    assert pkv_lru.evictions > 0 and pkv_lru.host_fetches > 0
    assert torch.equal(kg_lru, k_seq[:, :, :80, :]) and torch.equal(vg_lru, v_seq[:, :, :80, :])

    # Page, tile, FlashDecoding boundaries + CUDA Graph decode_pos reuse + mixed batch
    k_long = torch.randn((1, h_kv, 2052, d), device="cuda", dtype=torch.float16)
    v_long = torch.randn((1, h_kv, 2052, d), device="cuda", dtype=torch.float16)
    q1 = torch.randn((1, h_q, 1, d), device="cuda", dtype=torch.float16)
    for seq_len in (15, 16, 17, 31, 32, 33, 63, 64, 65, 2047, 2048, 2049):
        c_b = PagedKVCache(num_layers=1, num_kv_heads=h_kv, head_dim=d, page_size=16, device="cuda")
        c_b.update(0, k_long[:, :, : seq_len - 1, :], v_long[:, :, : seq_len - 1, :])
        c_b.update(
            0, k_long[:, :, seq_len - 1 : seq_len, :], v_long[:, :, seq_len - 1 : seq_len, :]
        )
        kg_b, vg_b = c_b.get_kv(0)
        out_b = fused_attention_gqa(q1, kg_b, vg_b, causal=True)
        ref_b = _fp64_gqa_attn(q1, k_long[:, :, :seq_len, :], v_long[:, :, :seq_len, :])
        assert float((out_b.float() - ref_b.float()).abs().max().item()) < 5e-3

    # Mixed-length batch + CUDA graph capture/replay/reset
    lens_mix = [5, 17, 33, 65]
    nc = NativePagedKVCache(
        num_layers=1,
        num_kv_heads=h_kv,
        head_dim=d,
        block_size=16,
        max_seq_len=128,
        max_batch_size=4,
        device="cuda",
    )
    nc.k_pools[0].fill_(float("nan"))
    nc.v_pools[0].fill_(float("nan"))
    k_m4 = torch.randn((4, h_kv, 65, d), device="cuda", dtype=torch.float16)
    v_m4 = torch.randn((4, h_kv, 65, d), device="cuda", dtype=torch.float16)
    for b, lb in enumerate(lens_mix):
        nc.k_pools[0][b, :, :lb, :] = k_m4[b, :, :lb, :]
        nc.v_pools[0][b, :, :lb, :] = v_m4[b, :, :lb, :]
    q_m4 = torch.randn((4, h_q, 1, d), device="cuda", dtype=torch.float16)
    pos_m4 = torch.tensor([lb - 1 for lb in lens_mix], device="cuda", dtype=torch.int32)
    out_m4 = torch.empty_like(q_m4)
    fused_attention_gqa_decode_pos(
        q_m4, nc.k_pools[0], nc.v_pools[0], pos_m4, max_kv_len=128, out=out_m4, dynamic_loop=True
    )
    for b, lb in enumerate(lens_mix):
        ref_r = _fp64_gqa_attn(
            q_m4[b : b + 1], k_m4[b : b + 1, :, :lb, :], v_m4[b : b + 1, :, :lb, :]
        )
        assert torch.isfinite(out_m4[b : b + 1]).all()
        assert float((out_m4[b : b + 1].float() - ref_r.float()).abs().max().item()) < 5e-3
