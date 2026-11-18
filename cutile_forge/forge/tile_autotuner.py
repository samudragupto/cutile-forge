"""Per-GPU tile shape autotuner across 12 candidate CuTile configurations."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from cutile_forge.forge.tile_pipeliner import SM89_MAX_SMEM_BYTES, estimate_smem_bytes


@dataclass(frozen=True)
class TileConfig:
    """Candidate CuTile kernel configuration."""

    tm: int
    tn: int
    tk: int
    occupancy: int
    pipeline_stages: int

    @property
    def shape(self) -> tuple[int, int, int]:
        return (self.tm, self.tn, self.tk)


# 12 tuned candidate configurations targeting Ada (sm_89) and Ampere+ GPUs
CANDIDATE_TILE_CONFIGS: list[TileConfig] = [
    TileConfig(16, 64, 64, occupancy=4, pipeline_stages=3),
    TileConfig(16, 128, 64, occupancy=4, pipeline_stages=3),
    TileConfig(32, 64, 64, occupancy=4, pipeline_stages=3),
    TileConfig(32, 128, 64, occupancy=2, pipeline_stages=3),
    TileConfig(64, 64, 64, occupancy=4, pipeline_stages=3),
    TileConfig(64, 128, 32, occupancy=2, pipeline_stages=3),
    TileConfig(64, 128, 64, occupancy=2, pipeline_stages=3),
    TileConfig(128, 64, 32, occupancy=2, pipeline_stages=3),
    TileConfig(128, 64, 64, occupancy=2, pipeline_stages=3),
    TileConfig(128, 128, 32, occupancy=2, pipeline_stages=3),
    TileConfig(128, 128, 64, occupancy=2, pipeline_stages=2),
    TileConfig(128, 128, 64, occupancy=1, pipeline_stages=3),
]


@dataclass(frozen=True)
class FFNExecutionStrategy:
    """Selected FFN execution strategy and tile shape from the occupancy cost model."""

    mode: str  # "single_kernel_3op" or "pipelined_2stage"
    tile_shape: tuple[int, int, int]
    estimated_occupancy: float


def select_ffn_execution_strategy(
    m: int,
    n: int,
    k: int,
    arch: str = "sm_120",
    sm_count: int | None = None,
) -> FFNExecutionStrategy:
    """Occupancy & roofline cost model selecting 3-op vs 2-stage pipelined FFN."""
    if sm_count is None:
        if arch == "sm_120":
            sm_count = 170
        elif torch.cuda.is_available():
            sm_count = torch.cuda.get_device_properties(0).multi_processor_count
        else:
            sm_count = 24

    if m <= 64:
        tm = 16 if m <= 16 else 64
        return FFNExecutionStrategy(
            mode="single_kernel_3op",
            tile_shape=(tm, 128, 64),
            estimated_occupancy=0.5,
        )

    if m >= 128 and (arch == "sm_120" or sm_count >= 100):
        # High-occupancy 64x64x32 tile keeps SM utilization high across 170 SMs
        return FFNExecutionStrategy(
            mode="pipelined_2stage",
            tile_shape=(64, 64, 32),
            estimated_occupancy=0.75,
        )

    if m > 256:
        return FFNExecutionStrategy(
            mode="pipelined_2stage",
            tile_shape=(128, 128, 32),
            estimated_occupancy=0.5,
        )

    return FFNExecutionStrategy(
        mode="single_kernel_3op",
        tile_shape=(128, 128, 64),
        estimated_occupancy=0.5,
    )


class TileAutotuner:
    """Searches 12 tile configs for a given (M, N, K) shape and target GPU."""

    def __init__(self, target: str = "sm_89", smem_limit_bytes: int = SM89_MAX_SMEM_BYTES) -> None:
        self.target = target
        self.smem_limit_bytes = smem_limit_bytes
        self._cache: dict[tuple[int, int, int, bool], TileConfig] = {}

    def valid_configs(self, quantized: bool = False) -> list[TileConfig]:
        """Return candidate configs that fit within target shared-memory budget."""
        return [
            cfg
            for cfg in CANDIDATE_TILE_CONFIGS
            if estimate_smem_bytes(cfg.shape, cfg.pipeline_stages, quantized)
            <= self.smem_limit_bytes
        ]

    @staticmethod
    def _heuristic_score(cfg: TileConfig, m: int, n: int, k: int) -> float:
        """Score a config analytically: lower is better (waste + wave quantization)."""
        m_tiles = (m + cfg.tm - 1) // cfg.tm
        n_tiles = (n + cfg.tn - 1) // cfg.tn
        padded_m = m_tiles * cfg.tm
        padded_n = n_tiles * cfg.tn
        waste = float(padded_m * padded_n) / max(1.0, float(m * n))
        # Prefer >= 24 CTAs on RTX 4060 (24 SMs) and higher arithmetic intensity
        num_ctas = m_tiles * n_tiles
        sm_underfill = max(1.0, 24.0 / max(1, num_ctas))
        arith_intensity = (2.0 * cfg.tm * cfg.tn) / max(1.0, float(cfg.tm + cfg.tn))
        return (waste * sm_underfill) / (arith_intensity * (1.0 + 0.05 * cfg.pipeline_stages))

    def autotune(
        self,
        m: int,
        n: int,
        k: int,
        quantized: bool = False,
        benchmark_on_gpu: bool = False,
        warmup: int = 2,
        iters: int = 10,
    ) -> TileConfig:
        """Select the best TileConfig for `(m, n, k)` via GPU benchmarking or analytical model."""
        key = (m, n, k, quantized)
        if key in self._cache:
            return self._cache[key]

        candidates = self.valid_configs(quantized=quantized)
        if not candidates:
            candidates = [CANDIDATE_TILE_CONFIGS[0]]

        if benchmark_on_gpu and torch.cuda.is_available():
            from cutile_forge.bench.harness import benchmark
            from cutile_forge.kernels.gemm_dequant import cutile_gemm_fp16

            a = torch.randn((m, k), device="cuda", dtype=torch.float16) * 0.1
            b = torch.randn((k, n), device="cuda", dtype=torch.float16) * 0.1
            best_cfg = candidates[0]
            best_ms = float("inf")
            # Filter candidates that don't waste huge M padding when M is tiny
            filtered = [c for c in candidates if c.tm <= max(16, m * 2)] or candidates[:4]
            for cfg in filtered:
                try:

                    def _run_cfg(
                        a: torch.Tensor = a,
                        b: torch.Tensor = b,
                        shape: tuple[int, int, int] = cfg.shape,
                    ) -> torch.Tensor:
                        return cutile_gemm_fp16(a, b, tile_shape=shape)

                    st = benchmark(
                        _run_cfg,
                        warmup=warmup,
                        iters=iters,
                        name=f"{cfg.tm}x{cfg.tn}x{cfg.tk}",
                    )
                    if st.p50_ms < best_ms:
                        best_ms = st.p50_ms
                        best_cfg = cfg
                except Exception:
                    continue
            self._cache[key] = best_cfg
            return best_cfg

        best_cfg = min(candidates, key=lambda c: self._heuristic_score(c, m, n, k))
        self._cache[key] = best_cfg
        return best_cfg
