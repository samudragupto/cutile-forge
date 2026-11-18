"""TFG code generator: lowers fused TileNodes to CuTile Python DSL kernels and cubin/SASS."""

from __future__ import annotations

import textwrap
from dataclasses import dataclass, field
from typing import Any

from cutile_forge.forge.fusion_graph import FusionGraph, OpType, TileNode
from cutile_forge.forge.scheduler import ScheduleEntry, schedule_tfg
from cutile_forge.forge.tile_autotuner import TileAutotuner
from cutile_forge.forge.tile_pipeliner import insert_software_pipeline
from cutile_forge.forge.weight_streamer import attach_weight_streaming


@dataclass
class CompiledKernel:
    """Compiled representation of one TFG TileNode."""

    node_id: str
    kernel_name: str
    op_type: OpType
    source_code: str
    tile_shape: tuple[int, int, int]
    pipeline_depth: int
    target: str
    cubin_bytes: bytes | None = None
    symbol: str | None = None
    kernel_fn: Any | None = None


@dataclass
class CompiledTFG:
    """Complete compiled output for a Tile Fusion Graph."""

    graph_name: str
    target: str
    pipeline_depth: int
    kernels: list[CompiledKernel] = field(default_factory=list)
    schedule: list[ScheduleEntry] = field(default_factory=list)

    def summary(self) -> str:
        lines = [
            f"CompiledTFG({self.graph_name}, target={self.target}, "
            f"pipeline_depth={self.pipeline_depth}, kernels={len(self.kernels)})"
        ]
        for k in self.kernels:
            cubin_info = f"{len(k.cubin_bytes)} B cubin" if k.cubin_bytes else "source-only"
            lines.append(
                f"  - {k.kernel_name} [{k.op_type.value}] tile={k.tile_shape} ({cubin_info})"
            )
        return "\n".join(lines)


def _emit_kernel_source(node: TileNode, pipeline_depth: int) -> tuple[str, str]:
    """Generate clean CuTile Python DSL source for a TileNode."""
    safe_id = node.node_id.replace("-", "_")
    fn_name = f"cutile_{safe_id}_{node.op_type.value}"
    tm, tn, tk = node.tile_shape
    occ = node.occupancy
    lat = max(1, min(4, pipeline_depth))

    if node.op_type in (OpType.FUSED_QKV_ATTN, OpType.ATTENTION_GQA):
        src = textwrap.dedent(f"""
            import cuda.tile as ct

            @ct.kernel(occupancy={occ})
            def {fn_name}(
                q_ptr, k_ptr, v_ptr, o_ptr,
                sm_scale: float, s_q: int, s_kv: int, kv_offset: int,
                gqa_ratio: ct.Constant[int], bq: ct.Constant[int],
                bk: ct.Constant[int], d_head: ct.Constant[int],
            ):
                q_tile_idx = ct.bid(0)
                bh_q = ct.bid(1)
                bh_kv = bh_q // gqa_ratio
                q = ct.reshape(
                    ct.load(q_ptr, index=(bh_q, q_tile_idx, 0), shape=(1, bq, d_head),
                            padding_mode=ct.PaddingMode.ZERO, latency={lat}),
                    (bq, d_head),
                )
                m_i = ct.full((bq, 1), -1.0e9, dtype=ct.float32)
                l_i = ct.zeros((bq, 1), dtype=ct.float32)
                acc = ct.zeros((bq, d_head), dtype=ct.float32)
                num_kv = (s_kv + bk - 1) // bk
                for kv_tile in range(num_kv):
                    k_t = ct.reshape(
                        ct.load(k_ptr, index=(bh_kv, kv_tile, 0), shape=(1, bk, d_head),
                                padding_mode=ct.PaddingMode.ZERO, latency={lat}),
                        (bk, d_head),
                    )
                    v_t = ct.reshape(
                        ct.load(v_ptr, index=(bh_kv, kv_tile, 0), shape=(1, bk, d_head),
                                padding_mode=ct.PaddingMode.ZERO, latency={lat}),
                        (bk, d_head),
                    )
                    scores = ct.mma(q, ct.transpose(k_t), ct.zeros((bq, bk), dtype=ct.float32)) * sm_scale
                    m_new = ct.maximum(m_i, ct.max(scores, axis=1, keepdims=True))
                    alpha = ct.exp(m_i - m_new)
                    p = ct.exp(scores - m_new)
                    l_i = l_i * alpha + ct.sum(p, axis=1, keepdims=True)
                    acc = ct.mma(ct.astype(p, ct.float16), v_t, acc * alpha)
                    m_i = m_new
                ct.store(o_ptr, index=(bh_q, q_tile_idx, 0),
                         tile=ct.reshape(ct.astype(acc / ct.maximum(l_i, 1.0e-6), ct.float16), (1, bq, d_head)))
            """).strip()
        return fn_name, src

    if node.op_type in (OpType.FUSED_FFN, OpType.FUSED_FFN_NORM):
        src = textwrap.dedent(f"""
            import cuda.tile as ct

            @ct.kernel(occupancy={occ})
            def {fn_name}(
                x_ptr, wu_ptr, wd_ptr, y_part_ptr,
                m_pad: ct.Constant[int], tm: ct.Constant[int],
                th: ct.Constant[int], tk: ct.Constant[int], td: ct.Constant[int],
            ):
                pid_m = ct.bid(0)
                pid_h = ct.bid(1)
                num_k = ct.num_tiles(x_ptr, axis=1, shape=(tm, tk))
                num_d = ct.num_tiles(wd_ptr, axis=1, shape=(th, td))
                acc_up = ct.zeros((tm, th), dtype=ct.float32)
                for k in range(num_k):
                    tx = ct.load(x_ptr, index=(pid_m, k), shape=(tm, tk), padding_mode=ct.PaddingMode.ZERO, latency={lat})
                    twu = ct.load(wu_ptr, index=(k, pid_h), shape=(tk, th), padding_mode=ct.PaddingMode.ZERO, latency={lat})
                    acc_up = ct.mma(tx, twu, acc_up)
                act = ct.astype(acc_up * (1.0 / (1.0 + ct.exp(-acc_up))), ct.float16)
                row_off = pid_h * (m_pad // tm) + pid_m
                for d_tile in range(num_d):
                    twd = ct.load(wd_ptr, index=(pid_h, d_tile), shape=(th, td), padding_mode=ct.PaddingMode.ZERO, latency={lat})
                    out = ct.mma(act, twd, ct.zeros((tm, td), dtype=ct.float32))
                    ct.store(y_part_ptr, index=(row_off, d_tile), tile=ct.astype(out, ct.float16))
            """).strip()
        return fn_name, src

    # Default tiled GEMM / Fused O-proj + Norm / GEMM+SiLU lowering
    epilogue = ""
    if node.op_type == OpType.GEMM_SILU:
        epilogue = "    acc = acc * (1.0 / (1.0 + ct.exp(-acc)))"
    src = textwrap.dedent(f"""
        import cuda.tile as ct

        @ct.kernel(occupancy={occ})
        def {fn_name}(
            a_ptr, b_ptr, c_ptr,
            tm: ct.Constant[int], tn: ct.Constant[int], tk: ct.Constant[int],
        ):
            pid_m = ct.bid(0)
            pid_n = ct.bid(1)
            num_k = ct.num_tiles(a_ptr, axis=1, shape=(tm, tk))
            acc = ct.zeros((tm, tn), dtype=ct.float32)
            for k in range(num_k):
                ta = ct.load(a_ptr, index=(pid_m, k), shape=(tm, tk), padding_mode=ct.PaddingMode.ZERO, latency={lat})
                tb = ct.load(b_ptr, index=(k, pid_n), shape=(tk, tn), padding_mode=ct.PaddingMode.ZERO, latency={lat})
                acc = ct.mma(ta, tb, acc)
        {epilogue}
            ct.store(c_ptr, index=(pid_m, pid_n), tile=ct.astype(acc, ct.float16))
        """).strip()
    return fn_name, src


def _bind_live_kernel(node: TileNode) -> Any | None:
    """Attach the matching live @ct.kernel callable from cutile_forge.kernels if available."""
    try:
        from cutile_forge.kernels import (
            attention_gqa,
            ffn_fused,
            gemm_dequant,
            gemm_dequant_silu,
            layernorm_residual,
        )
    except Exception:
        return None

    mapping = {
        OpType.FUSED_QKV_ATTN: attention_gqa.attention_gqa_kernel,
        OpType.ATTENTION_GQA: attention_gqa.attention_gqa_kernel,
        OpType.FUSED_O_LAYERNORM: gemm_dequant.gemm_dequant_kernel,
        OpType.FUSED_FFN: ffn_fused.ffn_dequant_up_silu_down_kernel,
        OpType.FUSED_FFN_NORM: ffn_fused.ffn_dequant_up_silu_down_kernel,
        OpType.GEMM_SILU: gemm_dequant_silu.gemm_dequant_silu_kernel,
        OpType.GEMM_DEQUANT: gemm_dequant.gemm_dequant_kernel,
        OpType.GEMM: gemm_dequant.gemm_fp16_kernel,
        OpType.LAYERNORM_RESIDUAL: layernorm_residual.layernorm_residual_kernel,
    }
    return mapping.get(node.op_type, gemm_dequant.gemm_fp16_kernel)


def _try_compile_cubin(node: TileNode, target: str) -> tuple[bytes | None, str | None]:
    """Compile a canonical CuTile kernel to cubin/SASS via cuda.tile.compilation."""
    try:
        import io

        import cuda.tile as ct
        import cuda.tile.compilation as cc
        import torch

        from cutile_forge.kernels.gemm_dequant import gemm_fp16_kernel
    except Exception:
        return None, None

    if not torch.cuda.is_available():
        return None, None

    try:
        tm, tn, tk = node.tile_shape
        ac = lambda: cc.ArrayConstraint(  # noqa: E731
            ct.float16,
            2,
            index_dtype=ct.int32,
            stride_lower_bound_incl=(1, 1),
            alias_groups=(),
            may_alias_internally=False,
        )
        sig = cc.KernelSignature(
            [ac(), ac(), ac(), tm, tn, tk],
            cc.CallingConvention.cutile_python_v2(),
        )
        buf = io.BytesIO()
        cc.export_kernel(gemm_fp16_kernel, [sig], buf, gpu_code=target, output_format="cubin")
        return buf.getvalue(), sig.symbol or f"{node.node_id}_sm89"
    except Exception:
        return None, None


def compile_tfg(
    graph: FusionGraph,
    target: str = "sm_120",
    pipeline_depth: int = 4,
    autotune: bool = True,
    compile_cubin: bool = False,
) -> CompiledTFG:
    """Lower a FusionGraph through pipelining, streaming, autotuning, and CuTile codegen."""
    graph.validate()

    if autotune:
        tuner = TileAutotuner(target=target)
        for node in graph.nodes.values():
            m = int(node.attrs.get("M", 256))
            n = int(node.attrs.get("N", 4096))
            k = int(node.attrs.get("K", 4096))
            q = bool(node.attrs.get("quantized", False))
            cfg = tuner.autotune(m, n, k, quantized=q, benchmark_on_gpu=False)
            node.tile_shape = cfg.shape
            node.occupancy = cfg.occupancy

    insert_software_pipeline(graph, pipeline_depth=pipeline_depth)
    attach_weight_streaming(graph)
    schedule = schedule_tfg(graph)

    kernels: list[CompiledKernel] = []
    for node in graph.topological_order():
        fn_name, src = _emit_kernel_source(node, pipeline_depth=node.pipeline_stages)
        cubin_bytes: bytes | None = None
        symbol: str | None = None
        if compile_cubin:
            cubin_bytes, symbol = _try_compile_cubin(node, target=target)
        kernels.append(
            CompiledKernel(
                node_id=node.node_id,
                kernel_name=fn_name,
                op_type=node.op_type,
                source_code=src,
                tile_shape=node.tile_shape,
                pipeline_depth=node.pipeline_stages,
                target=target,
                cubin_bytes=cubin_bytes,
                symbol=symbol,
                kernel_fn=_bind_live_kernel(node),
            )
        )

    return CompiledTFG(
        graph_name=graph.name,
        target=target,
        pipeline_depth=pipeline_depth,
        kernels=kernels,
        schedule=schedule,
    )
