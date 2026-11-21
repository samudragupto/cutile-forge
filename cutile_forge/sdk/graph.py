"""High-level computational Graph builder API for CuTileForge SDK."""

from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Any

from cutile_forge.forge.fusion_graph import DataTensor, FusionGraph, OpType, TileNode
from cutile_forge.sdk.device import DType


@dataclass(frozen=True)
class TensorHandle:
    """Symbolic tensor handle produced by `Graph` operations."""

    name: str
    node_id: str
    shape: tuple[int, ...]
    dtype: str
    tensor: DataTensor


class Graph:
    """Imperative computational graph builder lowered directly to TFG IR."""

    def __init__(self, name: str = "sdk_graph") -> None:
        self.name = name
        self._tfg = FusionGraph(name=name)
        self._counter = 0

    def _next_id(self, prefix: str) -> str:
        self._counter += 1
        return f"{prefix}_{self._counter}"

    def input(
        self,
        name: str,
        shape: tuple[int, ...],
        dtype: DType | str = DType.FP16,
    ) -> TensorHandle:
        """Declare a graph input placeholder."""
        dtype_str = str(dtype.value if isinstance(dtype, DType) else dtype)
        dt = DataTensor(name=name, shape=tuple(shape), dtype=dtype_str)
        nid = self._next_id("in")
        self._tfg.add_node(
            TileNode(
                node_id=nid,
                op_type=OpType.INPUT,
                name=f"Input({name})",
                inputs=[],
                outputs=[dt],
            )
        )
        return TensorHandle(name=name, node_id=nid, shape=tuple(shape), dtype=dtype_str, tensor=dt)

    def _add_op(
        self,
        op_type: OpType,
        prefix: str,
        label: str,
        parents: list[TensorHandle],
        out_shape: tuple[int, ...],
        out_dtype: str | None = None,
        attrs: dict[str, Any] | None = None,
    ) -> TensorHandle:
        nid = self._next_id(prefix)
        out_name = f"{nid}_out"
        dtype_val = out_dtype or parents[0].dtype
        out_dt = DataTensor(name=out_name, shape=out_shape, dtype=dtype_val)
        node = TileNode(
            node_id=nid,
            op_type=op_type,
            name=label,
            inputs=[p.tensor for p in parents],
            outputs=[out_dt],
            attrs=dict(attrs or {}),
        )
        self._tfg.add_node(node)
        for p in parents:
            self._tfg.add_edge(p.node_id, nid, p.tensor)
        return TensorHandle(
            name=out_name, node_id=nid, shape=out_shape, dtype=dtype_val, tensor=out_dt
        )

    def matmul(self, a: TensorHandle, b: TensorHandle) -> TensorHandle:
        if len(a.shape) < 2 or len(b.shape) < 2:
            raise ValueError(f"matmul requires >=2D inputs, got {a.shape} and {b.shape}")
        if a.shape[-1] != b.shape[-2]:
            raise ValueError(f"matmul inner dimension mismatch: {a.shape} vs {b.shape}")
        out_shape = (*a.shape[:-1], b.shape[-1])
        m = 1
        for d in a.shape[:-1]:
            m *= d
        return self._add_op(
            OpType.MATMUL,
            "mm",
            "MatMul",
            [a, b],
            out_shape,
            attrs={"M": m, "K": a.shape[-1], "N": b.shape[-1]},
        )

    def silu(self, x: TensorHandle) -> TensorHandle:
        return self._add_op(OpType.SILU, "silu", "SiLU", [x], x.shape)

    def gelu(self, x: TensorHandle) -> TensorHandle:
        return self._add_op(OpType.GELU, "gelu", "GELU", [x], x.shape)

    def add(self, a: TensorHandle, b: TensorHandle) -> TensorHandle:
        if a.shape != b.shape:
            raise ValueError(f"add shape mismatch: {a.shape} vs {b.shape}")
        return self._add_op(OpType.ADD, "add", "Add", [a, b], a.shape)

    def rmsnorm(self, x: TensorHandle, weight: TensorHandle, eps: float = 1e-5) -> TensorHandle:
        return self._add_op(
            OpType.RMSNORM, "rmsnorm", "RMSNorm", [x, weight], x.shape, attrs={"eps": eps}
        )

    def layernorm(
        self,
        x: TensorHandle,
        weight: TensorHandle,
        bias: TensorHandle | None = None,
        eps: float = 1e-5,
    ) -> TensorHandle:
        parents = [x, weight, bias] if bias is not None else [x, weight]
        return self._add_op(
            OpType.LAYERNORM, "layernorm", "LayerNorm", parents, x.shape, attrs={"eps": eps}
        )

    def softmax(self, x: TensorHandle) -> TensorHandle:
        return self._add_op(OpType.SOFTMAX, "softmax", "Softmax", [x], x.shape)

    def transpose(self, x: TensorHandle, dim0: int = -2, dim1: int = -1) -> TensorHandle:
        dims = list(x.shape)
        d0 = dim0 if dim0 >= 0 else len(dims) + dim0
        d1 = dim1 if dim1 >= 0 else len(dims) + dim1
        dims[d0], dims[d1] = dims[d1], dims[d0]
        return self._add_op(
            OpType.TRANSPOSE,
            "transpose",
            "Transpose",
            [x],
            tuple(dims),
            attrs={"dim0": dim0, "dim1": dim1},
        )

    def reduce_sum(self, x: TensorHandle, dim: int = -1, keepdim: bool = False) -> TensorHandle:
        dims = list(x.shape)
        d = dim if dim >= 0 else len(dims) + dim
        if keepdim:
            dims[d] = 1
        else:
            dims.pop(d)
        return self._add_op(
            OpType.REDUCE_SUM,
            "sum",
            "ReduceSum",
            [x],
            tuple(dims),
            attrs={"dim": dim, "keepdim": keepdim},
        )

    def custom(
        self,
        op_name: str,
        *inputs: TensorHandle,
        out_shape: tuple[int, ...],
        out_dtype: DType | str | None = None,
        **kwargs: Any,
    ) -> TensorHandle:
        dtype_val = (
            str(out_dtype.value if isinstance(out_dtype, DType) else out_dtype)
            if out_dtype is not None
            else inputs[0].dtype
        )
        return self._add_op(
            OpType.CUSTOM,
            "custom",
            f"Custom({op_name})",
            list(inputs),
            out_shape,
            out_dtype=dtype_val,
            attrs={"custom_op": op_name, "custom_kwargs": kwargs},
        )

    def output(self, *handles: TensorHandle) -> None:
        """Mark output handles so DCE preserves only reachable nodes."""
        for h in handles:
            if h.node_id not in self._tfg.graph_outputs:
                self._tfg.graph_outputs.append(h.node_id)

    def to_fusion_graph(self) -> FusionGraph:
        """Return an independent copy of the underlying `FusionGraph` IR."""
        self._tfg.validate()
        return copy.deepcopy(self._tfg)
