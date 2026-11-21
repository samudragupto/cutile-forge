"""Generic operator registry defining traits and cost-model properties for TFG IR."""

from __future__ import annotations

from dataclasses import dataclass

from cutile_forge.forge.fusion_graph import OpType


@dataclass(frozen=True)
class OpSpec:
    """Compiler metadata describing one `OpType`."""

    op_type: OpType
    category: str  # "input", "compute", "memory", "elementwise", "reduction", "norm", "composite"
    bound_regime: str  # "compute", "memory", "neutral"
    is_pure: bool = True
    fusable_epilogue: bool = False
    fusable_prologue: bool = False


class OpRegistry:
    """Registry mapping TFG `OpType` values to `OpSpec` traits."""

    def __init__(self) -> None:
        self._specs: dict[OpType, OpSpec] = {}

    def register(self, spec: OpSpec) -> None:
        self._specs[spec.op_type] = spec

    def get(self, op_type: OpType) -> OpSpec:
        if op_type in self._specs:
            return self._specs[op_type]
        return OpSpec(
            op_type=op_type,
            category="unknown",
            bound_regime="memory",
            is_pure=op_type != OpType.CUSTOM,
        )

    def all_ops(self) -> list[OpType]:
        return list(self._specs.keys())


def build_default_op_registry() -> OpRegistry:
    reg = OpRegistry()
    for op in (OpType.INPUT,):
        reg.register(OpSpec(op, "input", "neutral", is_pure=True))
    for op in (
        OpType.GEMM,
        OpType.MATMUL,
        OpType.BATCHED_MATMUL,
        OpType.GEMM_DEQUANT,
        OpType.QKV_GEMM,
        OpType.O_GEMM,
        OpType.FFN_UP_GEMM,
        OpType.FFN_DOWN_GEMM,
        OpType.ATTENTION_GQA,
    ):
        reg.register(OpSpec(op, "compute", "compute", is_pure=True, fusable_prologue=True))
    for op in (OpType.SILU, OpType.GELU, OpType.ADD, OpType.RESIDUAL_ADD):
        reg.register(OpSpec(op, "elementwise", "memory", is_pure=True, fusable_epilogue=True))
    for op in (OpType.SOFTMAX, OpType.REDUCE_SUM, OpType.REDUCE_MAX, OpType.CUMSUM):
        reg.register(OpSpec(op, "reduction", "memory", is_pure=True))
    for op in (OpType.LAYERNORM, OpType.RMSNORM, OpType.LAYERNORM_RESIDUAL):
        reg.register(OpSpec(op, "norm", "memory", is_pure=True, fusable_epilogue=True))
    for op in (OpType.DEQUANT, OpType.TRANSPOSE):
        reg.register(OpSpec(op, "memory", "memory", is_pure=True, fusable_prologue=True))
    for op in (
        OpType.GEMM_SILU,
        OpType.FUSED_QKV_ATTN,
        OpType.FUSED_O_LAYERNORM,
        OpType.FUSED_FFN,
        OpType.FUSED_FFN_NORM,
    ):
        reg.register(OpSpec(op, "composite", "compute", is_pure=True))
    reg.register(OpSpec(OpType.CUSTOM, "custom", "neutral", is_pure=False))
    return reg


DEFAULT_OP_REGISTRY = build_default_op_registry()
