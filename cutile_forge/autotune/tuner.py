"""Cost-model-guided and verified autotuner backed by `TuningDatabase`."""

from __future__ import annotations

import torch

from cutile_forge.autotune.db import TuningDatabase, TuningRecord
from cutile_forge.forge.tile_autotuner import TileAutotuner
from cutile_forge.forge.tile_pipeliner import resolve_target_specs


class CostGuidedAutotuner:
    """Independent autotuning subsystem reusing TFG cost models and persistent caching."""

    def __init__(
        self,
        target: str = "auto",
        db: TuningDatabase | None = None,
    ) -> None:
        arch, num_sms, smem_bytes = resolve_target_specs(target)
        self.arch = arch
        self.num_sms = num_sms
        self.smem_bytes = smem_bytes
        self.db = db or TuningDatabase()
        self.tuner = TileAutotuner(
            target=arch,
            smem_limit_bytes=smem_bytes,
            num_sms=num_sms,
        )

    def tune_gemm(
        self,
        m: int,
        n: int,
        k: int,
        dtype: str = "float16",
        benchmark_on_gpu: bool = False,
        verify_atol: float = 2e-2,
        top_k: int = 4,
    ) -> TuningRecord:
        """Select, verify, and cache the best tile config for `(m, n, k)` GEMM."""
        shape = (m, n, k)
        cached = self.db.lookup("gemm", self.arch, shape, dtype)
        if cached is not None:
            return cached

        candidates = self.tuner.valid_configs(quantized=False)
        ranked = sorted(
            candidates,
            key=lambda c: TileAutotuner._heuristic_score(c, m, n, k, num_sms=self.num_sms),
        )
        best_cfg = ranked[0]
        best_ms = 0.0
        verified = True

        if benchmark_on_gpu and torch.cuda.is_available() and dtype == "float16":
            from cutile_forge.bench.harness import benchmark
            from cutile_forge.kernels.gemm_dequant import cutile_gemm_fp16

            a = torch.randn((m, k), device="cuda", dtype=torch.float16) * 0.1
            b = torch.randn((k, n), device="cuda", dtype=torch.float16) * 0.1
            ref = torch.matmul(a.float(), b.float()).to(torch.float16)

            best_ms = float("inf")
            for cfg in ranked[: max(1, top_k)]:
                try:
                    out = cutile_gemm_fp16(a, b, tile_shape=cfg.shape)
                    err = (out - ref).abs().max().item()
                    if err > verify_atol:
                        continue

                    def _run_bench(
                        mat_a: torch.Tensor = a,
                        mat_b: torch.Tensor = b,
                        shape: tuple[int, int, int] = cfg.shape,
                    ) -> torch.Tensor:
                        return cutile_gemm_fp16(mat_a, mat_b, tile_shape=shape)

                    stats = benchmark(
                        _run_bench,
                        warmup=2,
                        iters=10,
                    )
                    if stats.p50_ms < best_ms:
                        best_ms = stats.p50_ms
                        best_cfg = cfg
                        verified = True
                except Exception:
                    continue
            if best_ms == float("inf"):
                best_ms = 0.0
                verified = False

        rec = TuningRecord(
            op="gemm",
            arch=self.arch,
            shape=shape,
            dtype=dtype,
            tile_shape=best_cfg.shape,
            occupancy=best_cfg.occupancy,
            pipeline_stages=best_cfg.pipeline_stages,
            latency_ms=round(best_ms, 4),
            verified=verified,
        )
        self.db.save(rec)
        return rec
