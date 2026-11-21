"""ONNX and JSON graph-spec importer lowering models to CuTileForge SDK `Graph`."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from cutile_forge.sdk.device import DType
from cutile_forge.sdk.graph import Graph, TensorHandle


def import_graph_spec(spec: dict[str, Any]) -> Graph:
    """Lower a declarative graph dictionary (`inputs`, `nodes`, `outputs`) into an SDK `Graph`."""
    g = Graph(name=str(spec.get("name", "onnx_imported")))
    env: dict[str, TensorHandle] = {}

    for inp in spec.get("inputs", []):
        name = str(inp["name"])
        shape = tuple(int(d) for d in inp["shape"])
        dtype = str(inp.get("dtype", DType.FP16.value))
        env[name] = g.input(name, shape, dtype=dtype)

    for node in spec.get("nodes", []):
        op = str(node["op"])
        in_handles = [env[str(n)] for n in node["inputs"]]
        attrs = dict(node.get("attrs", {}))
        out_name = str(node["output"])

        if op == "MatMul":
            h = g.matmul(in_handles[0], in_handles[1])
        elif op == "BatchedMatMul":
            h = g.bmm(in_handles[0], in_handles[1])
        elif op == "Silu":
            h = g.silu(in_handles[0])
        elif op == "Gelu":
            h = g.gelu(in_handles[0])
        elif op == "Add":
            h = g.add(in_handles[0], in_handles[1])
        elif op == "Softmax":
            h = g.softmax(in_handles[0])
        elif op == "ReduceSum":
            h = g.reduce_sum(
                in_handles[0],
                dim=int(attrs.get("dim", -1)),
                keepdim=bool(attrs.get("keepdim", False)),
            )
        elif op == "ReduceMax":
            h = g.reduce_max(
                in_handles[0],
                dim=int(attrs.get("dim", -1)),
                keepdim=bool(attrs.get("keepdim", False)),
            )
        elif op == "Transpose":
            h = g.transpose(
                in_handles[0],
                dim0=int(attrs.get("dim0", -2)),
                dim1=int(attrs.get("dim1", -1)),
            )
        else:
            raise ValueError(f"Unsupported ONNX/spec operator: '{op}'")

        env[out_name] = h

    out_handles = [env[str(name)] for name in spec.get("outputs", [])]
    if out_handles:
        g.output(*out_handles)
    return g


def import_onnx_model(model_or_path: Any) -> Graph:
    """Load an `onnx.ModelProto` or `.onnx` file into a CuTileForge SDK `Graph`."""
    try:
        import onnx
    except ImportError as exc:
        raise ImportError(
            "Install optional dependency 'onnx' to load .onnx files directly."
        ) from exc

    model = (
        onnx.load(str(model_or_path)) if isinstance(model_or_path, (str, Path)) else model_or_path
    )
    dtype_map = {1: "float32", 10: "float16", 16: "bfloat16"}
    inputs_spec: list[dict[str, Any]] = []
    for vi in model.graph.input:
        tt = vi.type.tensor_type
        dims = [int(d.dim_value or 1) for d in tt.shape.dim]
        inputs_spec.append(
            {
                "name": vi.name,
                "shape": dims,
                "dtype": dtype_map.get(tt.elem_type, "float16"),
            }
        )

    nodes_spec: list[dict[str, Any]] = []
    for n in model.graph.node:
        attrs = {}
        for a in n.attribute:
            if a.name in ("axis", "keepdims"):
                attrs["dim" if a.name == "axis" else "keepdim"] = int(a.i)
        nodes_spec.append(
            {
                "op": n.op_type,
                "inputs": list(n.input),
                "output": n.output[0],
                "attrs": attrs,
            }
        )

    return import_graph_spec(
        {
            "name": model.graph.name or "onnx_graph",
            "inputs": inputs_spec,
            "nodes": nodes_spec,
            "outputs": [o.name for o in model.graph.output],
        }
    )
