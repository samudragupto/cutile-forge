"""Track A profiling & before/after kernel audit on RTX 5090 (sm_120)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

from cutile_forge.bench.harness import benchmark, write_csv
from cutile_forge.bench.profiler import (
    KernelProfileResult,
    analytical_profile_op,
    profile_kernel_ncu,
    save_profile_json,
)
from cutile_forge.kernels import (
    CuTilePatchEmbed,
    cutile_avg_pool2d,
    cutile_conv2d,
    cutile_depthwise_conv2d,
    cutile_gather,
    cutile_gelu,
    cutile_gemm_fp16,
    cutile_max_pool2d,
    cutile_reduce_max,
    cutile_reduce_sum,
    cutile_silu,
)

V200_BASELINE_SPEEDUPS = {
    "depthwise_conv2d": 0.03,
    "max_pool2d": 0.05,
    "avg_pool2d": 0.05,
    "patch_embed": 0.25,
    "conv2d_1x1": 0.31,
    "conv2d_3x3": 0.41,
    "gather": 0.50,
    "silu": 0.58,
    "gelu": 0.57,
}


def run_audit(
    csv_out: Path = Path("benchmarks/results_5090/track_a_before_after.csv"),
    profile_out: Path = Path("benchmarks/results_5090/track_a_profile.json"),
    warmup: int = 5,
    iters: int = 50,
    baseline_only: bool = False,
) -> tuple[list[dict[str, object]], list[KernelProfileResult]]:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU required for Track A audit")

    dtype = torch.float16
    ncu_probe = profile_kernel_ncu(
        [sys.executable, "-c", "import torch; x=torch.ones(64,64,device='cuda'); _=x+x"],
        kernel_regex=".*",
        timeout_s=15,
    )
    ncu_diag = ncu_probe.get("diagnostic")

    rows: list[dict[str, object]] = []
    profiles: list[KernelProfileResult] = []

    def _measure(
        op: str,
        shape: str,
        ct_fn: object,
        pt_fn: object,
        *,
        nbytes: float,
        flops: float = 0.0,
        occ: float = 50.0,
    ) -> None:
        prof = analytical_profile_op(
            op,
            ct_fn,  # type: ignore[arg-type]
            bytes_transferred=nbytes,
            flops=flops,
            estimated_occupancy_pct=occ,
            warmup=warmup,
            iters=iters,
            ncu_diagnostic=ncu_diag,
        )
        profiles.append(prof)
        st_pt = benchmark(pt_fn, warmup=warmup, iters=iters)  # type: ignore[arg-type]
        sp = round(st_pt.p50_ms / max(1e-6, prof.p50_ms), 2)
        rows.append(
            {
                "op": op,
                "shape": shape,
                "v200_speedup": V200_BASELINE_SPEEDUPS.get(op, "n/a"),
                "v210_cutile_ms": round(prof.p50_ms, 4),
                "torch_p50_ms": round(st_pt.p50_ms, 4),
                "v210_speedup": sp,
                "achieved_gbps": prof.achieved_gbps,
                "achieved_tflops": prof.achieved_tflops,
            }
        )

    cx = torch.randn((16, 64, 56, 56), device="cuda", dtype=dtype) * 0.1
    cw3 = torch.randn((128, 64, 3, 3), device="cuda", dtype=dtype) * 0.1
    cw1 = torch.randn((128, 64, 1, 1), device="cuda", dtype=dtype) * 0.1
    cdw = torch.randn((64, 1, 3, 3), device="cuda", dtype=dtype) * 0.1

    def _ct_dw() -> torch.Tensor:
        return cutile_depthwise_conv2d(cx, cdw, padding=1)

    def _pt_dw() -> torch.Tensor:
        return F.conv2d(cx, cdw, padding=1, groups=64)

    def _ct_mp() -> torch.Tensor:
        return cutile_max_pool2d(cx, 2, 2)

    def _pt_mp() -> torch.Tensor:
        return F.max_pool2d(cx, 2, 2)

    def _ct_ap() -> torch.Tensor:
        return cutile_avg_pool2d(cx, 2, 2)

    def _pt_ap() -> torch.Tensor:
        return F.avg_pool2d(cx, 2, 2)

    def _ct_c1() -> torch.Tensor:
        return cutile_conv2d(cx, cw1)

    def _pt_c1() -> torch.Tensor:
        return F.conv2d(cx, cw1)

    def _ct_c3() -> torch.Tensor:
        return cutile_conv2d(cx, cw3, padding=1)

    def _pt_c3() -> torch.Tensor:
        return F.conv2d(cx, cw3, padding=1)

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

    tbl = torch.randn((32000, 4096), device="cuda", dtype=dtype)
    idx = torch.randint(0, 32000, (1024,), device="cuda", dtype=torch.long)

    def _ct_g() -> torch.Tensor:
        return cutile_gather(tbl, idx)

    def _pt_g() -> torch.Tensor:
        return F.embedding(idx, tbl)

    rx = torch.randn((1024, 4096), device="cuda", dtype=dtype)

    def _ct_silu() -> torch.Tensor:
        return cutile_silu(rx)

    def _pt_silu() -> torch.Tensor:
        return F.silu(rx)

    def _ct_gelu() -> torch.Tensor:
        return cutile_gelu(rx)

    def _pt_gelu() -> torch.Tensor:
        return F.gelu(rx, approximate="tanh")

    n_elems_cx = 16 * 64 * 56 * 56 * 2
    _measure(
        "depthwise_conv2d",
        "16x64x56x56,k=3",
        _ct_dw,
        _pt_dw,
        nbytes=n_elems_cx * 2,
        flops=2 * 16 * 64 * 56 * 56 * 9,
        occ=75.0,
    )
    _measure("max_pool2d", "16x64x56x56,k=2", _ct_mp, _pt_mp, nbytes=n_elems_cx * 1.25, occ=75.0)
    _measure("avg_pool2d", "16x64x56x56,k=2", _ct_ap, _pt_ap, nbytes=n_elems_cx * 1.25, occ=75.0)
    _measure(
        "patch_embed",
        "16x3x224x224,p=16,d=768",
        _ct_pe,
        _pt_pe,
        nbytes=16 * 3 * 224 * 224 * 2 + 16 * 196 * 768 * 2,
        flops=2 * 16 * 196 * 768 * (3 * 16 * 16),
        occ=50.0,
    )
    _measure(
        "conv2d_1x1",
        "16x64x56x56->128",
        _ct_c1,
        _pt_c1,
        nbytes=16 * (64 + 128) * 56 * 56 * 2,
        flops=2 * 16 * 56 * 56 * 64 * 128,
        occ=50.0,
    )
    _measure(
        "conv2d_3x3",
        "16x64x56x56->128",
        _ct_c3,
        _pt_c3,
        nbytes=16 * (64 + 128) * 56 * 56 * 2 + 128 * 64 * 9 * 2,
        flops=2 * 16 * 56 * 56 * 128 * 64 * 9,
        occ=50.0,
    )
    _measure("gather", "1024x4096", _ct_g, _pt_g, nbytes=2 * 1024 * 4096 * 2, occ=75.0)
    _measure("silu", "1024x4096", _ct_silu, _pt_silu, nbytes=2 * 1024 * 4096 * 2, occ=75.0)
    _measure("gelu", "1024x4096", _ct_gelu, _pt_gelu, nbytes=2 * 1024 * 4096 * 2, occ=75.0)

    # Reductions
    r_big = torch.randn((4096, 4096), device="cuda", dtype=dtype)

    def _ct_rsum() -> torch.Tensor:
        return cutile_reduce_sum(r_big, -1)

    def _pt_rsum() -> torch.Tensor:
        return r_big.sum(dim=-1)

    def _ct_rmax() -> torch.Tensor:
        return cutile_reduce_max(r_big, -1)

    def _pt_rmax() -> torch.Tensor:
        return r_big.max(dim=-1).values

    _measure("reduce_sum", "4096x4096", _ct_rsum, _pt_rsum, nbytes=4096 * 4096 * 2, occ=75.0)
    _measure("reduce_max", "4096x4096", _ct_rmax, _pt_rmax, nbytes=4096 * 4096 * 2, occ=75.0)

    if not baseline_only:
        from cutile_forge.kernels import (
            cutile_fused_add_rmsnorm_silu,
            cutile_fused_layernorm_gelu,
            cutile_fused_rmsnorm_silu,
        )

        nx = torch.randn((1024, 4096), device="cuda", dtype=dtype) * 0.2
        nr = torch.randn((1024, 4096), device="cuda", dtype=dtype) * 0.2
        nw = torch.ones((4096,), device="cuda", dtype=dtype)
        nb = torch.zeros((4096,), device="cuda", dtype=dtype)

        def _ct_rs() -> torch.Tensor:
            return cutile_fused_rmsnorm_silu(nx, nw)

        def _pt_rs() -> torch.Tensor:
            xf = nx.float()
            y = (xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + 1e-5)).to(dtype) * nw
            return F.silu(y)

        def _ct_lg() -> torch.Tensor:
            return cutile_fused_layernorm_gelu(nx, nw, nb)

        def _pt_lg() -> torch.Tensor:
            return F.gelu(F.layer_norm(nx, (4096,), nw, nb, 1e-5), approximate="tanh")

        def _ct_ars() -> tuple[torch.Tensor, torch.Tensor]:
            return cutile_fused_add_rmsnorm_silu(nx, nr, nw)

        def _pt_ars() -> torch.Tensor:
            s = nx + nr
            sf = s.float()
            y = (sf * torch.rsqrt(sf.pow(2).mean(-1, keepdim=True) + 1e-5)).to(dtype) * nw
            return F.silu(y)

        _measure(
            "fused_rmsnorm_silu",
            "1024x4096",
            _ct_rs,
            _pt_rs,
            nbytes=2 * 1024 * 4096 * 2,
            occ=75.0,
        )
        _measure(
            "fused_layernorm_gelu",
            "1024x4096",
            _ct_lg,
            _pt_lg,
            nbytes=2 * 1024 * 4096 * 2,
            occ=75.0,
        )
        _measure(
            "fused_add_rmsnorm_silu",
            "1024x4096",
            _ct_ars,
            _pt_ars,
            nbytes=3 * 1024 * 4096 * 2,
            occ=75.0,
        )

        # 3 GEMM regimes
        for name_g, m_g, n_g, k_g in [
            ("gemm_skinny_m4", 4, 4096, 4096),
            ("gemm_medium_m64", 64, 4096, 4096),
            ("gemm_large_m1024", 1024, 4096, 4096),
        ]:
            ga = torch.randn((m_g, k_g), device="cuda", dtype=dtype) * 0.1
            gb = torch.randn((k_g, n_g), device="cuda", dtype=dtype) * 0.1

            def _ct_gemm(a: torch.Tensor = ga, b: torch.Tensor = gb) -> torch.Tensor:
                return cutile_gemm_fp16(a, b)

            def _pt_gemm(a: torch.Tensor = ga, b: torch.Tensor = gb) -> torch.Tensor:
                return torch.matmul(a, b)

            _measure(
                name_g,
                f"{m_g}x{n_g}x{k_g}",
                _ct_gemm,
                _pt_gemm,
                nbytes=(m_g * k_g + k_g * n_g + m_g * n_g) * 2,
                flops=2 * m_g * n_g * k_g,
                occ=50.0,
            )

    if not baseline_only:
        write_csv(csv_out, rows)
    save_profile_json(profiles, profile_out, extra_meta={"ncu_probe": ncu_probe})
    return rows, profiles


def main() -> None:
    parser = argparse.ArgumentParser(description="Track A profiling and audit")
    parser.add_argument("--baseline-only", action="store_true")
    parser.add_argument(
        "--csv-out", type=Path, default=Path("benchmarks/results_5090/track_a_before_after.csv")
    )
    parser.add_argument(
        "--profile-out", type=Path, default=Path("benchmarks/results_5090/track_a_profile.json")
    )
    args = parser.parse_args()
    rows, _ = run_audit(
        csv_out=args.csv_out, profile_out=args.profile_out, baseline_only=args.baseline_only
    )
    for r in rows:
        print(r)


if __name__ == "__main__":
    main()
