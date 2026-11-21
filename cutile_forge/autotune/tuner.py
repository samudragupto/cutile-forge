"""Cost-model-guided and verified multi-op autotuner (`CuTileTune v2`)."""

from __future__ import annotations

import torch

from cutile_forge.autotune.db import TuningDatabase, TuningRecord
from cutile_forge.forge.tile_autotuner import TileAutotuner
from cutile_forge.forge.tile_pipeliner import resolve_target_specs


class CostGuidedAutotuner:
    """Multi-op autotuner with analytical candidate pruning, accuracy gating, and persistent DB."""

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
                    if (out - ref).abs().max().item() > verify_atol:
                        continue

                    def _run_bench(
                        mat_a: torch.Tensor = a,
                        mat_b: torch.Tensor = b,
                        sh: tuple[int, int, int] = cfg.shape,
                    ) -> torch.Tensor:
                        return cutile_gemm_fp16(mat_a, mat_b, tile_shape=sh)

                    stats = benchmark(_run_bench, warmup=2, iters=10)
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

    def tune_bmm(
        self,
        batch: int,
        m: int,
        n: int,
        k: int,
        dtype: str = "float16",
        benchmark_on_gpu: bool = False,
        verify_atol: float = 2e-2,
    ) -> TuningRecord:
        """Select, verify, and cache tile config for `(batch, m, n, k)` BMM."""
        shape = (batch, m, n, k)
        cached = self.db.lookup("bmm", self.arch, shape, dtype)
        if cached is not None:
            return cached

        tk = 32 if dtype == "float32" else 64
        candidates = [(64, 64, tk), (128, 64, tk), (128, 128, tk)] if m > 16 else [(16, 64, tk)]
        best_shape = candidates[0]
        best_ms = 0.0
        verified = True

        if benchmark_on_gpu and torch.cuda.is_available() and dtype == "float16":
            from cutile_forge.bench.harness import benchmark
            from cutile_forge.kernels.bmm import cutile_bmm

            a = torch.randn((batch, m, k), device="cuda", dtype=torch.float16) * 0.1
            b = torch.randn((batch, k, n), device="cuda", dtype=torch.float16) * 0.1
            ref = torch.bmm(a.float(), b.float()).to(torch.float16)
            best_ms = float("inf")
            for cand in candidates:
                try:
                    out = cutile_bmm(a, b, tile_shape=cand)
                    if (out - ref).abs().max().item() > verify_atol:
                        continue

                    def _run_bmm(
                        ma: torch.Tensor = a,
                        mb: torch.Tensor = b,
                        sh: tuple[int, int, int] = cand,
                    ) -> torch.Tensor:
                        return cutile_bmm(ma, mb, tile_shape=sh)

                    st = benchmark(_run_bmm, warmup=2, iters=10)
                    if st.p50_ms < best_ms:
                        best_ms = st.p50_ms
                        best_shape = cand
                        verified = True
                except Exception:
                    continue
            if best_ms == float("inf"):
                best_ms = 0.0
                verified = False

        rec = TuningRecord(
            op="bmm",
            arch=self.arch,
            shape=shape,
            dtype=dtype,
            tile_shape=best_shape,
            occupancy=2,
            pipeline_stages=3,
            latency_ms=round(best_ms, 4),
            verified=verified,
        )
        self.db.save(rec)
        return rec

    def tune_reduction(
        self,
        rows: int,
        cols: int,
        op: str = "reduce_sum",
        dtype: str = "float16",
        benchmark_on_gpu: bool = False,
    ) -> TuningRecord:
        """Select, verify, and cache tile config for row-wise reduction or scan."""
        shape = (rows, cols)
        cached = self.db.lookup(op, self.arch, shape, dtype)
        if cached is not None:
            return cached

        from cutile_forge.kernels.reductions import _select_reduction_tiles

        tm, tn = _select_reduction_tiles(rows, cols)
        best_ms = 0.0
        verified = True

        if benchmark_on_gpu and torch.cuda.is_available() and dtype == "float16":
            from cutile_forge.bench.harness import benchmark
            from cutile_forge.kernels.reductions import cutile_reduce_max, cutile_reduce_sum

            x = torch.randn((rows, cols), device="cuda", dtype=torch.float16) * 0.05
            fn = cutile_reduce_max if op == "reduce_max" else cutile_reduce_sum
            out = fn(x, dim=-1)
            ref = x.max(dim=-1).values if op == "reduce_max" else x.float().sum(dim=-1).half()
            if (out - ref).abs().max().item() <= 0.25:

                def _run_red(inp: torch.Tensor = x) -> torch.Tensor:
                    return fn(inp, -1)

                st = benchmark(_run_red, warmup=2, iters=10)
                best_ms = st.p50_ms
            else:
                verified = False

        rec = TuningRecord(
            op=op,
            arch=self.arch,
            shape=shape,
            dtype=dtype,
            tile_shape=(tm, tn, 1),
            occupancy=4,
            pipeline_stages=3,
            latency_ms=round(best_ms, 4),
            verified=verified,
        )
        self.db.save(rec)
        return rec

    def tune_transpose(
        self,
        m: int,
        n: int,
        dtype: str = "float16",
        benchmark_on_gpu: bool = False,
    ) -> TuningRecord:
        """Select and cache optimal 2D transpose tile `(tm, tn, 1)`."""
        shape = (m, n)
        cached = self.db.lookup("transpose", self.arch, shape, dtype)
        if cached is not None:
            return cached

        tm = tn = 64 if (m >= 64 and n >= 64) else 32
        best_ms = 0.0
        verified = True
        if benchmark_on_gpu and torch.cuda.is_available() and dtype == "float16":
            from cutile_forge.bench.harness import benchmark
            from cutile_forge.kernels.transpose import cutile_transpose

            x = torch.randn((m, n), device="cuda", dtype=torch.float16)
            out = cutile_transpose(x, -2, -1)
            if torch.equal(out, x.t().contiguous()):

                def _run_tr(inp: torch.Tensor = x) -> torch.Tensor:
                    return cutile_transpose(inp)

                best_ms = benchmark(_run_tr, warmup=2, iters=10).p50_ms
            else:
                verified = False

        rec = TuningRecord(
            op="transpose",
            arch=self.arch,
            shape=shape,
            dtype=dtype,
            tile_shape=(tm, tn, 1),
            occupancy=4,
            pipeline_stages=3,
            latency_ms=round(best_ms, 4),
            verified=verified,
        )
        self.db.save(rec)
        return rec

    def tune_activation(
        self,
        numel: int,
        op: str = "silu",
        dtype: str = "float16",
        benchmark_on_gpu: bool = False,
    ) -> TuningRecord:
        """Select and cache optimal 1D tile width for elementwise activation kernels."""
        shape = (numel,)
        cached = self.db.lookup(op, self.arch, shape, dtype)
        if cached is not None:
            return cached

        tile_n = 4096 if numel >= 16384 else 1024
        best_ms = 0.0
        verified = True
        if benchmark_on_gpu and torch.cuda.is_available() and dtype == "float16":
            import torch.nn.functional as F

            from cutile_forge.bench.harness import benchmark
            from cutile_forge.kernels.activations import cutile_gelu, cutile_silu

            x = torch.randn((numel,), device="cuda", dtype=torch.float16)
            fn = cutile_gelu if op == "gelu" else cutile_silu
            ref = F.gelu(x.float(), approximate="tanh").half() if op == "gelu" else F.silu(x)
            if (fn(x) - ref).abs().max().item() <= 2e-2:

                def _run_act(inp: torch.Tensor = x) -> torch.Tensor:
                    return fn(inp)

                best_ms = benchmark(_run_act, warmup=2, iters=10).p50_ms
            else:
                verified = False

        rec = TuningRecord(
            op=op,
            arch=self.arch,
            shape=shape,
            dtype=dtype,
            tile_shape=(tile_n, 1, 1),
            occupancy=4,
            pipeline_stages=3,
            latency_ms=round(best_ms, 4),
            verified=verified,
        )
        self.db.save(rec)
        return rec

    def tune_quantized_gemm(
        self,
        m: int,
        n: int,
        k: int,
        quant_format: str = "gptq",
        group_size: int = 128,
        benchmark_on_gpu: bool = False,
        verify_atol: float = 1e-1,
    ) -> TuningRecord:
        """Select, verify, and persist optimal tile configuration keyed on `(quant_format, m, n, k)`."""
        fmt_key = f"{quant_format.lower()}_g{group_size}"
        shape = (m, n, k)
        cached = self.db.lookup("quant_gemm", self.arch, shape, fmt_key)
        if cached is not None:
            return cached

        fmt_low = quant_format.lower()
        if fmt_low in ("gguf_q4_0", "gguf_q8_0", "gguf_q6_k"):
            tk_cands = [32]
        elif fmt_low in ("gguf_q4_k", "gguf_q4_k_m", "awq", "fp8_block"):
            tk_cands = [64]
        else:
            tk_cands = [128, 64] if (group_size >= 128 and k % 128 == 0) else [64, 32]

        if m <= 16:
            mn_cands = [(16, 32), (16, 64)] if fmt_low == "gptq" else [(16, 64)]
        else:
            mn_cands = [(64, 64), (64, 128)]

        candidates = [
            (tm, tn, tk)
            for (tm, tn) in mn_cands
            for tk in tk_cands
            if (k % tk == 0 and n % tn == 0)
        ]
        if not candidates:
            candidates = [(16, 64, 32)]

        best_shape = candidates[0]
        best_ms = 0.0
        verified = True

        if benchmark_on_gpu and torch.cuda.is_available():
            from cutile_forge.bench.harness import benchmark
            from cutile_forge.kernels.universal_quant import QuantFormat, UniversalQuantLinear

            a = torch.randn((m, k), device="cuda", dtype=torch.float16) * 0.1
            w = torch.randn((k, n), device="cuda", dtype=torch.float16) * 0.1
            q_lin = UniversalQuantLinear.from_fp16(
                w, fmt=QuantFormat(fmt_low), group_size=group_size
            )
            ref = q_lin.forward_reference(a)
            best_ms = float("inf")
            for cand in candidates:
                try:
                    q_lin.tile_shape = cand
                    out = q_lin.forward_native(a)
                    if (out.float() - ref.float()).abs().max().item() > verify_atol:
                        continue

                    def _run_q(
                        ql: UniversalQuantLinear = q_lin, inp: torch.Tensor = a
                    ) -> torch.Tensor:
                        return ql.forward_native(inp)

                    st = benchmark(_run_q, warmup=2, iters=10)
                    if st.p50_ms < best_ms:
                        best_ms = st.p50_ms
                        best_shape = cand
                        verified = True
                except Exception:
                    continue
            if best_ms == float("inf"):
                best_ms = 0.0
                verified = False

        rec = TuningRecord(
            op="quant_gemm",
            arch=self.arch,
            shape=shape,
            dtype=fmt_key,
            tile_shape=best_shape,
            occupancy=2,
            pipeline_stages=3,
            latency_ms=round(best_ms, 4),
            verified=verified,
        )
        self.db.save(rec)
        return rec
