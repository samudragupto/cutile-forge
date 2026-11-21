"""PyTorch Dynamo / FX lowering backend: `torch.compile(backend="cutile_forge")`."""

from __future__ import annotations

import hashlib
import operator
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import torch
import torch.fx as fx
import torch.nn.functional as F

from cutile_forge.forge.fusion_graph import DataTensor, FusionGraph, OpType, TileNode
from cutile_forge.forge.fusion_rules import apply_fusion_rules


@dataclass
class FXCompileStats:
    """Telemetry recorded on each `cutile_forge` FX compilation."""

    original_nodes: int
    fused_nodes: int
    ffn_fusions: int
    ln_res_fusions: int
    attn_fusions: int
    tfg_nodes_before: int
    tfg_nodes_after: int
    gemm_silu_fusions: int = 0
    bmm_lowerings: int = 0
    softmax_lowerings: int = 0
    cache_hit: bool = False
    fallback_ops: list[str] = field(default_factory=list)


LAST_FX_STATS: FXCompileStats | None = None
FX_COMPILE_CACHE: dict[str, tuple[Callable[..., Any], FXCompileStats]] = {}


def clear_fx_cache() -> None:
    """Clear the compiled FX graph cache."""
    FX_COMPILE_CACHE.clear()


def _cu_fused_ffn_linear(
    x: torch.Tensor,
    w_up: torch.Tensor,
    b_up: torch.Tensor | None,
    w_down: torch.Tensor,
    b_down: torch.Tensor | None,
) -> torch.Tensor:
    """Execute F.linear(F.silu(F.linear(x, w_up, b_up)), w_down, b_down) via CuTile when eligible."""
    if (
        x.is_cuda
        and x.dtype == torch.float16
        and b_up is None
        and b_down is None
        and x.ndim in (2, 3)
        and w_up.shape[0] % 64 == 0
        and w_down.shape[0] % 64 == 0
    ):
        from cutile_forge.kernels.ffn_fused import fused_ffn_up_silu_down

        orig_shape = x.shape
        x_2d = x.reshape(-1, orig_shape[-1])
        out_2d = fused_ffn_up_silu_down(x_2d, w_up.t().contiguous(), w_down.t().contiguous())
        return out_2d.view(*orig_shape[:-1], w_down.shape[0])
    h = F.silu(F.linear(x, w_up, b_up))
    return F.linear(h, w_down, b_down)


def _cu_fused_ln_residual(
    x: torch.Tensor,
    residual: torch.Tensor,
    normalized_shape: tuple[int, ...] | list[int],
    weight: torch.Tensor | None,
    bias: torch.Tensor | None,
    eps: float = 1e-5,
) -> torch.Tensor:
    """Execute layer_norm(x + residual) via CuTile fused_layernorm_residual when eligible."""
    if (
        x.is_cuda
        and x.dtype == torch.float16
        and weight is not None
        and bias is not None
        and len(normalized_shape) == 1
    ):
        from cutile_forge.kernels.layernorm_residual import fused_layernorm_residual

        orig_shape = x.shape
        x_2d = x.reshape(-1, orig_shape[-1])
        r_2d = residual.reshape(-1, orig_shape[-1])
        normed, _ = fused_layernorm_residual(x_2d, r_2d, weight, bias, eps=eps)
        return normed.view(orig_shape)
    return F.layer_norm(x + residual, list(normalized_shape), weight, bias, eps)


def _cu_fused_sdpa(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    attn_mask: torch.Tensor | None = None,
    dropout_p: float = 0.0,
    is_causal: bool = False,
    scale: float | None = None,
) -> torch.Tensor:
    """Lower scaled_dot_product_attention to CuTile fused_attention_gqa when eligible."""
    if (
        q.is_cuda
        and q.dtype == torch.float16
        and attn_mask is None
        and dropout_p == 0.0
        and q.ndim == 4
        and q.shape[1] % k.shape[1] == 0
        and q.shape[3] in (64, 128)
    ):
        from cutile_forge.kernels.attention_gqa import fused_attention_gqa

        return fused_attention_gqa(q, k, v, causal=is_causal, sm_scale=scale)
    return F.scaled_dot_product_attention(
        q, k, v, attn_mask=attn_mask, dropout_p=dropout_p, is_causal=is_causal, scale=scale
    )


def _cu_fused_linear_silu(
    x: torch.Tensor,
    w: torch.Tensor,
    bias: torch.Tensor | None = None,
) -> torch.Tensor:
    """Execute F.silu(F.linear(x, w, bias)) via CuTile fused_gemm_fp16_silu when eligible."""
    if (
        x.is_cuda
        and x.dtype == torch.float16
        and bias is None
        and x.ndim in (2, 3)
        and w.shape[0] % 64 == 0
    ):
        from cutile_forge.kernels.gemm_dequant_silu import fused_gemm_fp16_silu

        orig_shape = x.shape
        out_2d = fused_gemm_fp16_silu(x.reshape(-1, orig_shape[-1]), w.t().contiguous())
        return out_2d.view(*orig_shape[:-1], w.shape[0])
    return F.silu(F.linear(x, w, bias))


def _cu_bmm(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Execute torch.bmm via CuTile cutile_bmm when eligible."""
    if (
        a.is_cuda
        and a.dtype in (torch.float16, torch.bfloat16, torch.float32)
        and a.dtype == b.dtype
    ):
        from cutile_forge.kernels.bmm import cutile_bmm

        return cutile_bmm(a, b)
    return torch.bmm(a, b)


def _cu_softmax(
    x: torch.Tensor,
    dim: int | None = None,
    _stacklevel: int = 3,
    dtype: torch.dtype | None = None,
) -> torch.Tensor:
    """Execute F.softmax over the last axis via CuTile online_softmax when eligible."""
    if (
        x.is_cuda
        and x.dtype == torch.float16
        and dtype is None
        and (dim == -1 or dim == x.ndim - 1)
    ):
        from cutile_forge.kernels.softmax import online_softmax

        return online_softmax(x)
    return F.softmax(x, dim=dim, _stacklevel=_stacklevel, dtype=dtype)


_LOWERED_TARGETS = {
    _cu_fused_ffn_linear,
    _cu_fused_ln_residual,
    _cu_fused_sdpa,
    _cu_fused_linear_silu,
    _cu_bmm,
    _cu_softmax,
}


def _fx_cache_key(gm: fx.GraphModule, example_inputs: list[torch.Tensor]) -> str:
    tensor_sig = [
        (tuple(t.shape), str(t.dtype), str(t.device))
        for t in example_inputs
        if isinstance(t, torch.Tensor)
    ]
    param_sig = [
        (name, t.data_ptr(), tuple(t.shape), str(t.dtype))
        for name, t in list(gm.named_parameters()) + list(gm.named_buffers())
    ]
    raw = f"{gm.code}|{tensor_sig}|{param_sig}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _cu_fused_linear_silu(
    x: torch.Tensor,
    w: torch.Tensor,
    bias: torch.Tensor | None = None,
) -> torch.Tensor:
    """Execute F.silu(F.linear(x, w, bias)) via CuTile fused_gemm_fp16_silu when eligible."""
    if (
        x.is_cuda
        and x.dtype == torch.float16
        and bias is None
        and x.ndim in (2, 3)
        and w.shape[0] % 64 == 0
    ):
        from cutile_forge.kernels.gemm_dequant_silu import fused_gemm_fp16_silu

        orig_shape = x.shape
        out_2d = fused_gemm_fp16_silu(x.reshape(-1, orig_shape[-1]), w.t().contiguous())
        return out_2d.view(*orig_shape[:-1], w.shape[0])
    return F.silu(F.linear(x, w, bias))


def _cu_bmm(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Execute torch.bmm via CuTile cutile_bmm when eligible."""
    if a.is_cuda and a.dtype in (torch.float16, torch.bfloat16, torch.float32) and a.dtype == b.dtype:
        from cutile_forge.kernels.bmm import cutile_bmm

        return cutile_bmm(a, b)
    return torch.bmm(a, b)


def _cu_softmax(
    x: torch.Tensor,
    dim: int | None = None,
    _stacklevel: int = 3,
    dtype: torch.dtype | None = None,
) -> torch.Tensor:
    """Execute F.softmax over the last axis via CuTile online_softmax when eligible."""
    if (
        x.is_cuda
        and x.dtype == torch.float16
        and dtype is None
        and (dim == -1 or dim == x.ndim - 1)
    ):
        from cutile_forge.kernels.softmax import online_softmax

        return online_softmax(x)
    return F.softmax(x, dim=dim, _stacklevel=_stacklevel, dtype=dtype)


_LOWERED_TARGETS = {
    _cu_fused_ffn_linear,
    _cu_fused_ln_residual,
    _cu_fused_sdpa,
    _cu_fused_linear_silu,
    _cu_bmm,
    _cu_softmax,
}


def _fx_cache_key(gm: fx.GraphModule, example_inputs: list[torch.Tensor]) -> str:
    tensor_sig = [
        (tuple(t.shape), str(t.dtype), str(t.device))
        for t in example_inputs
        if isinstance(t, torch.Tensor)
    ]
    param_sig = [
        (name, t.data_ptr(), tuple(t.shape), str(t.dtype))
        for name, t in list(gm.named_parameters()) + list(gm.named_buffers())
    ]
    raw = f"{gm.code}|{tensor_sig}|{param_sig}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def cutile_forge_backend(
    gm: fx.GraphModule,
    example_inputs: list[torch.Tensor],
) -> Callable[..., Any]:
    """PyTorch FX/Dynamo backend lowering transformer subgraphs to fused CuTile kernels."""
    global LAST_FX_STATS
    cache_key = _fx_cache_key(gm, example_inputs)
    if cache_key in FX_COMPILE_CACHE:
        cached_fn, cached_stats = FX_COMPILE_CACHE[cache_key]
        LAST_FX_STATS = FXCompileStats(
            original_nodes=cached_stats.original_nodes,
            fused_nodes=cached_stats.fused_nodes,
            ffn_fusions=cached_stats.ffn_fusions,
            ln_res_fusions=cached_stats.ln_res_fusions,
            attn_fusions=cached_stats.attn_fusions,
            tfg_nodes_before=cached_stats.tfg_nodes_before,
            tfg_nodes_after=cached_stats.tfg_nodes_after,
            gemm_silu_fusions=cached_stats.gemm_silu_fusions,
            bmm_lowerings=cached_stats.bmm_lowerings,
            softmax_lowerings=cached_stats.softmax_lowerings,
            cache_hit=True,
            fallback_ops=list(cached_stats.fallback_ops),
        )
        return cached_fn

    graph = gm.graph
    orig_nodes = len([n for n in graph.nodes if n.op in ("call_function", "call_method")])

    fg = FusionGraph(name="fx_subgraph")
    ffn_fusions = 0
    ln_res_fusions = 0
    attn_fusions = 0
    gemm_silu_fusions = 0
    bmm_lowerings = 0
    softmax_lowerings = 0

    # Pass 1: Match `linear -> silu -> linear`
    nodes_snapshot = list(graph.nodes)
    for node in nodes_snapshot:
        if node.op == "call_function" and node.target is F.linear and len(node.args) >= 2:
            silu_node = node.args[0]
            if (
                isinstance(silu_node, fx.Node)
                and silu_node.op == "call_function"
                and silu_node.target is F.silu
                and len(silu_node.users) == 1
            ):
                up_node = silu_node.args[0]
                if (
                    isinstance(up_node, fx.Node)
                    and up_node.op == "call_function"
                    and up_node.target is F.linear
                    and len(up_node.users) == 1
                ):
                    x_arg = up_node.args[0]
                    w_up = up_node.args[1]
                    b_up = up_node.args[2] if len(up_node.args) > 2 else up_node.kwargs.get("bias")
                    w_down = node.args[1]
                    b_down = node.args[2] if len(node.args) > 2 else node.kwargs.get("bias")
                    with graph.inserting_before(node):
                        fused = graph.call_function(
                            _cu_fused_ffn_linear, args=(x_arg, w_up, b_up, w_down, b_down)
                        )
                    node.replace_all_uses_with(fused)
                    graph.erase_node(node)
                    graph.erase_node(silu_node)
                    graph.erase_node(up_node)
                    ffn_fusions += 1

                    # Record in TFG IR for fusion verification
                    t_spec = DataTensor("h", (1, 1), "float16")
                    u_id = f"fx_up_{ffn_fusions}"
                    s_id = f"fx_silu_{ffn_fusions}"
                    d_id = f"fx_down_{ffn_fusions}"
                    fg.add_node(
                        TileNode(
                            u_id, OpType.FFN_UP_GEMM, name=u_id, inputs=[t_spec], outputs=[t_spec]
                        )
                    )
                    fg.add_node(
                        TileNode(s_id, OpType.SILU, name=s_id, inputs=[t_spec], outputs=[t_spec])
                    )
                    fg.add_node(
                        TileNode(
                            d_id, OpType.FFN_DOWN_GEMM, name=d_id, inputs=[t_spec], outputs=[t_spec]
                        )
                    )
                    fg.add_edge(u_id, s_id, t_spec)
                    fg.add_edge(s_id, d_id, t_spec)

    # Pass 2: Match `add -> layer_norm`
    for node in list(graph.nodes):
        if node.op == "call_function" and node.target is F.layer_norm and len(node.args) >= 2:
            add_node = node.args[0]
            if (
                isinstance(add_node, fx.Node)
                and add_node.op == "call_function"
                and add_node.target in (operator.add, torch.add)
                and len(add_node.users) == 1
                and len(add_node.args) == 2
            ):
                x_a, x_b = add_node.args[0], add_node.args[1]
                norm_shape = node.args[1]
                weight = node.args[2] if len(node.args) > 2 else node.kwargs.get("weight")
                bias = node.args[3] if len(node.args) > 3 else node.kwargs.get("bias")
                eps = node.args[4] if len(node.args) > 4 else node.kwargs.get("eps", 1e-5)
                with graph.inserting_before(node):
                    fused = graph.call_function(
                        _cu_fused_ln_residual, args=(x_a, x_b, norm_shape, weight, bias, eps)
                    )
                node.replace_all_uses_with(fused)
                graph.erase_node(node)
                graph.erase_node(add_node)
                ln_res_fusions += 1

    # Pass 3: Lower `scaled_dot_product_attention` -> `_cu_fused_sdpa`
    for node in list(graph.nodes):
        if node.op == "call_function" and node.target is F.scaled_dot_product_attention:
            with graph.inserting_before(node):
                fused = graph.call_function(_cu_fused_sdpa, args=node.args, kwargs=node.kwargs)
            node.replace_all_uses_with(fused)
            graph.erase_node(node)
            attn_fusions += 1

    # Pass 4: Match standalone `linear -> silu` -> `_cu_fused_linear_silu`
    for node in list(graph.nodes):
        if node.op == "call_function" and node.target is F.silu and len(node.args) >= 1:
            lin_node = node.args[0]
            if (
                isinstance(lin_node, fx.Node)
                and lin_node.op == "call_function"
                and lin_node.target is F.linear
                and len(lin_node.users) == 1
            ):
                x_arg = lin_node.args[0]
                w_arg = lin_node.args[1]
                b_arg = lin_node.args[2] if len(lin_node.args) > 2 else lin_node.kwargs.get("bias")
                with graph.inserting_before(node):
                    fused = graph.call_function(_cu_fused_linear_silu, args=(x_arg, w_arg, b_arg))
                node.replace_all_uses_with(fused)
                graph.erase_node(node)
                graph.erase_node(lin_node)
                gemm_silu_fusions += 1

    # Pass 5: Lower `torch.bmm` -> `_cu_bmm`
    for node in list(graph.nodes):
        if node.op == "call_function" and node.target is torch.bmm and len(node.args) >= 2:
            with graph.inserting_before(node):
                fused = graph.call_function(_cu_bmm, args=(node.args[0], node.args[1]))
            node.replace_all_uses_with(fused)
            graph.erase_node(node)
            bmm_lowerings += 1

    # Pass 6: Lower `F.softmax` -> `_cu_softmax`
    for node in list(graph.nodes):
        if node.op == "call_function" and node.target is F.softmax:
            with graph.inserting_before(node):
                fused = graph.call_function(_cu_softmax, args=node.args, kwargs=node.kwargs)
            node.replace_all_uses_with(fused)
            graph.erase_node(node)
            softmax_lowerings += 1

    fallback_ops: list[str] = []
    for node in graph.nodes:
        if node.op in ("call_function", "call_method") and node.target not in _LOWERED_TARGETS:
            fallback_ops.append(str(getattr(node.target, "__name__", node.target)))

    tfg_before = len(fg.nodes)
    if tfg_before > 0:
        apply_fusion_rules(fg)
    tfg_after = len(fg.nodes)

    graph.lint()
    gm.recompile()
    fused_nodes = len([n for n in graph.nodes if n.op in ("call_function", "call_method")])

    stats = FXCompileStats(
        original_nodes=orig_nodes,
        fused_nodes=fused_nodes,
        ffn_fusions=ffn_fusions,
        ln_res_fusions=ln_res_fusions,
        attn_fusions=attn_fusions,
        tfg_nodes_before=tfg_before,
        tfg_nodes_after=tfg_after,
        gemm_silu_fusions=gemm_silu_fusions,
        bmm_lowerings=bmm_lowerings,
        softmax_lowerings=softmax_lowerings,
        cache_hit=False,
        fallback_ops=fallback_ops,
    )
    LAST_FX_STATS = stats
    FX_COMPILE_CACHE[cache_key] = (gm.forward, stats)
    return gm.forward


def register_cutile_forge_backend() -> None:
    """Register `"cutile_forge"` as a `torch.compile` backend if Dynamo is available."""
    try:
        import torch._dynamo as dynamo

        if "cutile_forge" not in dynamo.list_backends():
            dynamo.register_backend(name="cutile_forge", compiler_fn=cutile_forge_backend)
    except Exception:
        pass
