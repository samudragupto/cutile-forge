"""Microbenchmark for Phase 8 vision primitives, multi-tensor AdamW, and 10-step MLP training."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
import torch.nn.functional as F

from cutile_forge.bench.harness import benchmark, write_csv
from cutile_forge.kernels import (
    CuTilePatchEmbed,
    cutile_avg_pool2d,
    cutile_conv2d,
    cutile_depthwise_conv2d,
    cutile_gelu_autograd,
    cutile_grad_clip_norm,
    cutile_layernorm_autograd,
    cutile_linear_autograd,
    cutile_max_pool2d,
    cutile_multi_tensor_adamw,
)


def run_benchmark(
    output_csv: Path = Path("benchmarks/results_5090/phase8_training_vision.csv"),
    convergence_csv: Path = Path("benchmarks/results_5090/phase8_mlp_convergence.csv"),
    warmup: int = 5,
    iters: int = 50,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU required for Phase 8 training & vision benchmark")

    rows: list[dict[str, object]] = []
    dtype = torch.float16

    def _rec(op: str, shape: str, dtype_str: str, ct_fn: object, pt_fn: object) -> None:
        st_ct = benchmark(ct_fn, warmup=warmup, iters=iters)  # type: ignore[arg-type]
        st_pt = benchmark(pt_fn, warmup=warmup, iters=iters)  # type: ignore[arg-type]
        rows.append(
            {
                "op": op,
                "shape": shape,
                "dtype": dtype_str,
                "cutile_p50_ms": round(st_ct.p50_ms, 4),
                "torch_p50_ms": round(st_pt.p50_ms, 4),
                "speedup": round(st_pt.p50_ms / max(1e-6, st_ct.p50_ms), 2),
            }
        )

    # Vision ops (FP16)
    cx = torch.randn((16, 64, 56, 56), device="cuda", dtype=dtype) * 0.1
    cw3 = torch.randn((128, 64, 3, 3), device="cuda", dtype=dtype) * 0.1
    cw1 = torch.randn((128, 64, 1, 1), device="cuda", dtype=dtype) * 0.1
    cdw = torch.randn((64, 1, 3, 3), device="cuda", dtype=dtype) * 0.1

    def _ct_c3() -> torch.Tensor:
        return cutile_conv2d(cx, cw3, padding=1)

    def _pt_c3() -> torch.Tensor:
        return F.conv2d(cx, cw3, padding=1)

    def _ct_c1() -> torch.Tensor:
        return cutile_conv2d(cx, cw1)

    def _pt_c1() -> torch.Tensor:
        return F.conv2d(cx, cw1)

    def _ct_dw() -> torch.Tensor:
        return cutile_depthwise_conv2d(cx, cdw, padding=1)

    def _pt_dw() -> torch.Tensor:
        return F.conv2d(cx, cdw, padding=1, groups=64)

    def _ct_mp() -> torch.Tensor:
        return cutile_max_pool2d(cx, kernel_size=2, stride=2)

    def _pt_mp() -> torch.Tensor:
        return F.max_pool2d(cx, kernel_size=2, stride=2)

    def _ct_ap() -> torch.Tensor:
        return cutile_avg_pool2d(cx, kernel_size=2, stride=2)

    def _pt_ap() -> torch.Tensor:
        return F.avg_pool2d(cx, kernel_size=2, stride=2)

    _rec("conv2d_3x3", "16x64x56x56->128", "float16", _ct_c3, _pt_c3)
    _rec("conv2d_1x1", "16x64x56x56->128", "float16", _ct_c1, _pt_c1)
    _rec("depthwise_conv2d", "16x64x56x56,k=3", "float16", _ct_dw, _pt_dw)
    _rec("max_pool2d", "16x64x56x56,k=2", "float16", _ct_mp, _pt_mp)
    _rec("avg_pool2d", "16x64x56x56,k=2", "float16", _ct_ap, _pt_ap)

    img = torch.randn((16, 3, 224, 224), device="cuda", dtype=dtype) * 0.1
    pe = CuTilePatchEmbed(img_size=224, patch_size=16, in_chans=3, embed_dim=768).to(
        device="cuda", dtype=dtype
    )

    def _ct_pe() -> torch.Tensor:
        return pe(img)

    def _pt_pe() -> torch.Tensor:
        return (
            F.conv2d(img, pe.weight, bias=pe.bias, stride=16)
            .flatten(2)
            .transpose(1, 2)
            .contiguous()
        )

    _rec("patch_embed", "16x3x224x224,p=16,d=768", "float16", _ct_pe, _pt_pe)

    # Multi-tensor AdamW step timing (6 MLP tensors, FP32)
    shapes = [(256, 32), (256,), (256,), (256,), (16, 256), (16,)]
    cu_p = [torch.randn(s, device="cuda", dtype=torch.float32) * 0.1 for s in shapes]
    cu_g = [torch.randn(s, device="cuda", dtype=torch.float32) * 0.01 for s in shapes]
    cu_m = [torch.zeros_like(p) for p in cu_p]
    cu_v = [torch.zeros_like(p) for p in cu_p]

    pt_p = [p.clone().detach().requires_grad_(True) for p in cu_p]
    for p, g in zip(pt_p, cu_g, strict=True):
        p.grad = g.clone()
    pt_adamw = torch.optim.AdamW(pt_p, lr=1e-4, weight_decay=0.01)

    def _ct_adamw() -> None:
        cutile_multi_tensor_adamw(cu_p, cu_g, cu_m, cu_v, step=1, lr=1e-4, weight_decay=0.01)

    def _pt_adamw() -> None:
        pt_adamw.step()

    _rec("adamw_multi_tensor", "6_tensors_mlp", "float32", _ct_adamw, _pt_adamw)

    # Deterministic 10-step 2-layer MLP convergence parity (CuTile vs PyTorch)
    torch.manual_seed(42)
    batch, d_in, d_hid, d_out = 64, 32, 256, 16
    x_mlp = torch.randn((batch, d_in), device="cuda", dtype=torch.float32)
    target = torch.randn((batch, d_out), device="cuda", dtype=torch.float32)

    init_tensors = [
        torch.randn((d_hid, d_in), device="cuda", dtype=torch.float32) * 0.1,
        torch.zeros((d_hid,), device="cuda", dtype=torch.float32),
        torch.ones((d_hid,), device="cuda", dtype=torch.float32),
        torch.zeros((d_hid,), device="cuda", dtype=torch.float32),
        torch.randn((d_out, d_hid), device="cuda", dtype=torch.float32) * 0.1,
        torch.zeros((d_out,), device="cuda", dtype=torch.float32),
    ]

    conv_rows: list[dict[str, object]] = []
    p_cu = [t.clone().detach().requires_grad_(True) for t in init_tensors]
    p_pt = [t.clone().detach().requires_grad_(True) for t in init_tensors]
    m_cu = [torch.zeros_like(t) for t in p_cu]
    v_cu = [torch.zeros_like(t) for t in p_cu]
    opt_pt = torch.optim.AdamW(p_pt, lr=5e-2, weight_decay=0.01)

    for step in range(1, 11):
        for p in p_cu:
            p.grad = None
        h = cutile_linear_autograd(x_mlp, p_cu[0], p_cu[1])
        h = cutile_layernorm_autograd(h, p_cu[2], p_cu[3], 1e-5)
        h = cutile_gelu_autograd(h)
        pred = cutile_linear_autograd(h, p_cu[4], p_cu[5])
        loss_cu = F.mse_loss(pred, target)
        loss_cu.backward()
        cutile_grad_clip_norm(p_cu, max_norm=1.0)
        grads = [p.grad if p.grad is not None else torch.zeros_like(p) for p in p_cu]
        cutile_multi_tensor_adamw(p_cu, grads, m_cu, v_cu, step=step, lr=5e-2, weight_decay=0.01)

        opt_pt.zero_grad()
        hp = F.linear(x_mlp, p_pt[0], p_pt[1])
        hp = F.layer_norm(hp, (d_hid,), p_pt[2], p_pt[3], 1e-5)
        hp = F.gelu(hp, approximate="tanh")
        pred_p = F.linear(hp, p_pt[4], p_pt[5])
        loss_pt = F.mse_loss(pred_p, target)
        loss_pt.backward()
        torch.nn.utils.clip_grad_norm_(p_pt, max_norm=1.0)
        opt_pt.step()

        l_c = float(loss_cu.item())
        l_p = float(loss_pt.item())
        conv_rows.append(
            {
                "step": step,
                "cutile_mse_loss": round(l_c, 6),
                "torch_mse_loss": round(l_p, 6),
                "abs_diff": f"{abs(l_c - l_p):.2e}",
            }
        )

    # Full 10-step MLP training loop latency comparison
    def _run_10step_ct() -> float:
        params = [t.clone().detach().requires_grad_(True) for t in init_tensors]
        ms = [torch.zeros_like(t) for t in params]
        vs = [torch.zeros_like(t) for t in params]
        last_loss = torch.tensor(0.0, device="cuda")
        for step in range(1, 11):
            for p in params:
                p.grad = None
            h = cutile_linear_autograd(x_mlp, params[0], params[1])
            h = cutile_layernorm_autograd(h, params[2], params[3], 1e-5)
            h = cutile_gelu_autograd(h)
            pred = cutile_linear_autograd(h, params[4], params[5])
            last_loss = F.mse_loss(pred, target)
            last_loss.backward()
            cutile_grad_clip_norm(params, max_norm=1.0)
            gs = [p.grad if p.grad is not None else torch.zeros_like(p) for p in params]
            cutile_multi_tensor_adamw(params, gs, ms, vs, step=step, lr=5e-2, weight_decay=0.01)
        return float(last_loss.item())

    def _run_10step_pt() -> float:
        params = [t.clone().detach().requires_grad_(True) for t in init_tensors]
        opt = torch.optim.AdamW(params, lr=5e-2, weight_decay=0.01)
        last_loss = torch.tensor(0.0, device="cuda")
        for _ in range(10):
            opt.zero_grad()
            hp = F.linear(x_mlp, params[0], params[1])
            hp = F.layer_norm(hp, (d_hid,), params[2], params[3], 1e-5)
            hp = F.gelu(hp, approximate="tanh")
            pred_p = F.linear(hp, params[4], params[5])
            last_loss = F.mse_loss(pred_p, target)
            last_loss.backward()
            torch.nn.utils.clip_grad_norm_(params, max_norm=1.0)
            opt.step()
        return float(last_loss.item())

    _rec("mlp_10step_train", "b64,32->256->16", "float32", _run_10step_ct, _run_10step_pt)

    write_csv(output_csv, rows)
    write_csv(convergence_csv, conv_rows)
    return rows, conv_rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 8 training & vision benchmark")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("benchmarks/results_5090/phase8_training_vision.csv"),
    )
    parser.add_argument(
        "--convergence-output",
        type=Path,
        default=Path("benchmarks/results_5090/phase8_mlp_convergence.csv"),
    )
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iters", type=int, default=50)
    args = parser.parse_args()
    rows, conv = run_benchmark(
        output_csv=args.output,
        convergence_csv=args.convergence_output,
        warmup=args.warmup,
        iters=args.iters,
    )
    print(
        f"Wrote {len(rows)} rows -> {args.output} and {len(conv)} steps -> {args.convergence_output}"
    )


if __name__ == "__main__":
    main()
