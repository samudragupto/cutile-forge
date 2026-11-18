"""Micro-benchmark: Full Mistral-7B layer (3-kernel TFG vs 8-kernel CUDA vs PyTorch)."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, List

import torch
import torch.nn.functional as F

from cutile_forge.baselines import load_baselines
from cutile_forge.bench.harness import benchmark, write_csv
from cutile_forge.kernels.attention_gqa import fused_attention_gqa
from cutile_forge.kernels.ffn_fused import fused_ffn_dequant_up_silu_down
from cutile_forge.kernels.gemm_dequant import (
    fused_gemm_dequant,
    pack_int4_weights,
    unpack_int4_weights_torch,
)
from cutile_forge.kernels.layernorm_residual import fused_rmsnorm_residual


def run_fused_layer_benchmark(
    seq_lens: List[int] | None = None,
    warmup: int = 5,
    iters: int = 30,
) -> List[Dict[str, Any]]:
    """Benchmark full Mistral-7B layer: 3-kernel TFG vs 8-kernel unfused CUDA vs PyTorch."""
    baselines = load_baselines()
    seqs = seq_lens or [1, 64, 256, 512]
    d_model = 4096
    qkv_dim = 6144  # 32*128 + 8*128 + 8*128
    inter = 14336
    gs = 128

    def _mk_qw(k: int, n: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        qw = pack_int4_weights(torch.randint(0, 16, (k, n), device="cuda", dtype=torch.uint8))
        s = torch.rand((k // gs, n), device="cuda", dtype=torch.float16) * 0.02 + 0.005
        z = torch.randint(5, 11, (k // gs, n), device="cuda", dtype=torch.float16)
        return qw, s, z

    qw_qkv, s_qkv, z_qkv = _mk_qw(d_model, qkv_dim)
    qw_o, s_o, z_o = _mk_qw(d_model, d_model)
    qw_up, s_up, z_up = _mk_qw(d_model, inter)
    qw_down, s_down, z_down = _mk_qw(inter, d_model)

    ln_w = torch.ones(d_model, device="cuda", dtype=torch.float16)
    ln_b = torch.zeros(d_model, device="cuda", dtype=torch.float16)

    # Pre-dequantized fp16 weights for PyTorch eager reference
    w_qkv_fp16 = unpack_int4_weights_torch(qw_qkv, s_qkv, z_qkv, gs)
    w_o_fp16 = unpack_int4_weights_torch(qw_o, s_o, z_o, gs)
    w_up_fp16 = unpack_int4_weights_torch(qw_up, s_up, z_up, gs)
    w_down_fp16 = unpack_int4_weights_torch(qw_down, s_down, z_down, gs)

    rows: List[Dict[str, Any]] = []
    for seq in seqs:
        x = torch.randn((seq, d_model), device="cuda", dtype=torch.float16) * 0.05
        tm = 16 if seq <= 16 else 64
        m_pad = ((seq + tm - 1) // tm) * tm
        nh = (inter + 127) // 128
        workspace = torch.empty((nh * m_pad, d_model), device="cuda", dtype=torch.float16)

        def run_tfg_layer(x: torch.Tensor = x, seq: int = seq, ws: torch.Tensor = workspace) -> torch.Tensor:
            # Stage 1: QKV + GQA Attention
            qkv = fused_gemm_dequant(x, qw_qkv, s_qkv, z_qkv, gs)
            q = qkv[:, :4096].view(1, seq, 32, 128).transpose(1, 2)
            k = qkv[:, 4096:5120].view(1, seq, 8, 128).transpose(1, 2)
            v = qkv[:, 5120:6144].view(1, seq, 8, 128).transpose(1, 2)
            attn = fused_attention_gqa(q, k, v, causal=True).transpose(1, 2).contiguous().view(seq, d_model)
            # Stage 2: O GEMM + Fused RMSNorm+Residual
            o = fused_gemm_dequant(attn, qw_o, s_o, z_o, gs)
            post_norm, res_mid = fused_rmsnorm_residual(o, x, ln_w, 1e-5)
            # Stage 3: Single-kernel Fused FFN (Up + SiLU + Down)
            ffn = fused_ffn_dequant_up_silu_down(
                post_norm, qw_up, s_up, z_up, qw_down, s_down, z_down, gs, workspace=ws
            )
            return res_mid + ffn

        def run_cuda_8kernel(x: torch.Tensor = x, seq: int = seq) -> torch.Tensor:
            qkv = baselines.cuda_gemm_dequant_separate(x, qw_qkv, s_qkv, z_qkv, gs)
            q = qkv[:, :4096].view(1, seq, 32, 128).transpose(1, 2)
            k = qkv[:, 4096:5120].view(1, seq, 8, 128).transpose(1, 2)
            v = qkv[:, 5120:6144].view(1, seq, 8, 128).transpose(1, 2)
            attn = baselines.cuda_attention_gqa(q, k, v, True).transpose(1, 2).contiguous().view(seq, d_model)
            o = baselines.cuda_gemm_dequant_separate(attn, qw_o, s_o, z_o, gs)
            ln1 = baselines.cuda_layernorm(o + x, ln_w, ln_b, 1e-5)
            h_up = baselines.cuda_gemm_dequant_separate(ln1, qw_up, s_up, z_up, gs)
            h_silu = baselines.cuda_silu(h_up)
            h_down = baselines.cuda_gemm_dequant_separate(h_silu, qw_down, s_down, z_down, gs)
            return baselines.cuda_layernorm(h_down + o, ln_w, ln_b, 1e-5)

        def run_pytorch_eager(x: torch.Tensor = x, seq: int = seq) -> torch.Tensor:
            qkv = torch.matmul(x, w_qkv_fp16)
            q = qkv[:, :4096].view(1, seq, 32, 128).transpose(1, 2)
            k = qkv[:, 4096:5120].view(1, seq, 8, 128).transpose(1, 2).repeat_interleave(4, dim=1)
            v = qkv[:, 5120:6144].view(1, seq, 8, 128).transpose(1, 2).repeat_interleave(4, dim=1)
            attn = F.scaled_dot_product_attention(q, k, v, is_causal=(seq > 1))
            o = torch.matmul(attn.transpose(1, 2).contiguous().view(seq, d_model), w_o_fp16)
            ln1 = F.layer_norm(o + x, (d_model,), ln_w, ln_b, 1e-5)
            ffn = torch.matmul(F.silu(torch.matmul(ln1, w_up_fp16)), w_down_fp16)
            return F.layer_norm(ffn + o, (d_model,), ln_w, ln_b, 1e-5)

        st_tfg = benchmark(run_tfg_layer, warmup=warmup, iters=iters, name=f"tfg_{seq}")
        st_cuda = benchmark(run_cuda_8kernel, warmup=warmup, iters=iters, name=f"cuda8_{seq}")
        st_pt = benchmark(run_pytorch_eager, warmup=warmup, iters=iters, name=f"pytorch_{seq}")

        speedup = st_cuda.p50_ms / max(st_tfg.p50_ms, 1e-6)
        row: Dict[str, Any] = {
            "seq_len": seq,
            "tfg_kernels": 3,
            "unfused_kernels": 8,
            "tfg_ms": round(st_tfg.p50_ms, 4),
            "unfused_8k_ms": round(st_cuda.p50_ms, 4),
            "pytorch_ms": round(st_pt.p50_ms, 4),
            "speedup_vs_8k": round(speedup, 3),
        }
        rows.append(row)
        print(
            f"seq={seq:4d} | TFG(3k)={row['tfg_ms']:.3f} ms | "
            f"CUDA(8k)={row['unfused_8k_ms']:.3f} ms | PyTorch={row['pytorch_ms']:.3f} ms | "
            f"speedup={speedup:.2f}x"
        )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark full Mistral layer: 3-kernel TFG vs 8-kernel")
    parser.add_argument("--output", type=Path, default=Path("benchmarks/results/layer.csv"))
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iters", type=int, default=30)
    args = parser.parse_args()

    rows = run_fused_layer_benchmark(warmup=args.warmup, iters=args.iters)
    write_csv(args.output, rows)
    print(f"Saved fused layer benchmark to {args.output}")


if __name__ == "__main__":
    main()
