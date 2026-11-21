"""Unified CLI entrypoint (`cutile-forge serve`, `bench`, `doctor`, and `inspect`)."""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections.abc import Iterator
from pathlib import Path
from typing import Any


class _NativeServingAdapter:
    """Adapter exposing `InferenceEngine`-compatible serving methods atop `NativeModelRuntime`."""

    def __init__(self, runtime: Any, target: str = "sm_120") -> None:
        self.runtime = runtime
        self.config = runtime.config
        self.target = target
        self.radix_cache = None

    @property
    def graph_fallback_count(self) -> int:
        return int(getattr(self.runtime, "graph_fallback_count", 0))

    def _to_ids(self, prompt_ids: Any) -> list[int]:
        if hasattr(prompt_ids, "tolist"):
            raw = prompt_ids.tolist()
            return [int(x) for x in (raw[0] if raw and isinstance(raw[0], list) else raw)]
        return [int(x) for x in prompt_ids]

    def generate_ids(
        self,
        prompt_ids: Any,
        max_new_tokens: int = 32,
        temperature: float = 0.0,
        top_k: int = 50,
        top_p: float = 0.9,
    ) -> list[int]:
        ids = self._to_ids(prompt_ids)
        out, _ = self.runtime.generate(ids, max_new_tokens=max_new_tokens, sync_interval=16)
        return list(out)

    def generate_batch(
        self,
        prompts: list[Any],
        max_new_tokens: int = 32,
        temperature: float = 0.0,
        top_k: int = 50,
        top_p: float = 0.9,
    ) -> list[list[int]]:
        batch_ids = [self._to_ids(p) for p in prompts]
        outs, _ = self.runtime.generate_batch(batch_ids, max_new_tokens=max_new_tokens)
        return [list(o) for o in outs]

    def stream_ids(
        self,
        prompt_ids: Any,
        max_new_tokens: int = 32,
        temperature: float = 0.0,
        top_k: int = 50,
        top_p: float = 0.9,
    ) -> Iterator[int]:
        import torch

        ids = self._to_ids(prompt_ids)
        p_t = torch.tensor([ids], dtype=torch.long, device=self.runtime.device)
        tok_t = self.runtime.prefill(p_t)
        yield int(tok_t.item())
        pos = len(ids)
        for _ in range(max(0, max_new_tokens - 1)):
            tok_t = self.runtime.decode_step(tok_t, pos=pos)
            pos += 1
            yield int(tok_t.item())


def main() -> None:
    parser = argparse.ArgumentParser(prog="cutile-forge", description="CuTileForge CLI")
    sub = parser.add_subparsers(dest="command", required=True)

    serve_p = sub.add_parser("serve", help="Start OpenAI-compatible inference server")
    serve_p.add_argument("--model", type=Path, required=True)
    serve_p.add_argument("--host", default="127.0.0.1")
    serve_p.add_argument("--port", type=int, default=8000)
    serve_p.add_argument("--target", default="sm_120")
    serve_p.add_argument("--api-key", default=None)
    serve_p.add_argument("--max-concurrent", type=int, default=64)
    serve_p.add_argument("--max-batch-size", type=int, default=16)
    serve_p.add_argument("--max-prompt-tokens", type=int, default=4096)
    serve_p.add_argument("--native-runtime", action="store_true")

    bench_p = sub.add_parser("bench", help="Run CuTileForge benchmark suite")
    bench_p.add_argument("--model", type=Path, default=Path("/home/arrhat/models/mistral_7b_4bit"))
    bench_p.add_argument("--output", type=Path, default=Path("benchmarks/results_5090"))
    bench_p.add_argument("--target", default="sm_120")
    bench_p.add_argument("--gpu", default="rtx5090")
    bench_p.add_argument("--micro-only", action="store_true")

    doc_p = sub.add_parser("doctor", help="Diagnose CUDA, cuda.tile, and system compatibility")
    doc_p.add_argument("--json", action="store_true", help="Emit structured JSON report")
    doc_p.add_argument(
        "--require-gpu",
        action="store_true",
        help="Fail if CUDA GPU or cuda.tile is unavailable",
    )

    insp_p = sub.add_parser("inspect", help="Inspect TFG fusion, HBM traffic & buffer reuse")
    insp_p.add_argument("--seq-len", type=int, default=256)
    insp_p.add_argument("--target", default="sm_120")
    insp_p.add_argument("--json", action="store_true", help="Emit JSON report")

    args = parser.parse_args()
    target = getattr(args, "target", None)
    if target is not None:
        os.environ["CUTILE_ARCH"] = target

    if args.command == "inspect":
        from cutile_forge.forge.codegen import compile_tfg
        from cutile_forge.forge.fusion_graph import build_mistral_layer_tfg
        from cutile_forge.vision import inspect_compilation

        orig = build_mistral_layer_tfg(0, seq_len=args.seq_len)
        graph = build_mistral_layer_tfg(0, seq_len=args.seq_len)
        compiled = compile_tfg(graph, target=args.target, run_passes=True)
        summary = inspect_compilation(orig, compiled)
        print(json.dumps(summary.to_dict(), indent=2) if args.json else summary.to_markdown())
        return

    if args.command == "doctor":
        from cutile_forge.doctor import run_diagnostics

        report = run_diagnostics(require_gpu=args.require_gpu)
        if args.json:
            print(json.dumps(report.to_dict(), indent=2))
        else:
            print(report.format_text())
        sys.exit(0 if report.ok else 1)

    if args.command == "serve":
        import uvicorn

        from cutile_forge.engine import InferenceEngine, Mistral7B, ModelConfig
        from cutile_forge.engine.config import GPUConfig
        from cutile_forge.engine.weight_loader import load_gptq_weights
        from cutile_forge.server import create_app

        tok = None
        if (args.model / "config.json").exists():
            from transformers import AutoTokenizer

            weights = load_gptq_weights(args.model, device="cuda", pin_ddr5=False)
            model = Mistral7B(
                weights=weights,
                gpu_config=GPUConfig(vram_gb=32.0, stream_weights=False),
                device="cuda",
            )
            tok = AutoTokenizer.from_pretrained(str(args.model), local_files_only=True)
        else:
            if not args.model.exists():
                raise FileNotFoundError(f"Model directory not found: {args.model}")
            model = Mistral7B.from_random(ModelConfig(num_hidden_layers=4), device="cuda")

        if args.native_runtime:
            from cutile_forge.runtime import NativeModelRuntime

            rt = NativeModelRuntime(
                model,
                max_seq_len=max(2048, args.max_prompt_tokens + 256),
                use_cuda_graph=True,
                max_batch_size=min(16, args.max_batch_size),
            )
            engine: Any = _NativeServingAdapter(rt, target=args.target)
        else:
            engine = InferenceEngine(model, target=args.target)
        app = create_app(
            engine,
            tokenizer=tok,
            model_id=args.model.name,
            max_batch_size=args.max_batch_size,
            api_key=args.api_key,
            max_concurrent_requests=args.max_concurrent,
            max_prompt_tokens=args.max_prompt_tokens,
        )
        uvicorn.run(app, host=args.host, port=args.port)
    elif args.command == "bench":
        import importlib

        run_all = importlib.import_module("benchmarks.run_all").run_all

        run_all(
            model_dir=args.model,
            output_dir=args.output,
            micro_only=args.micro_only,
            target=args.target,
            gpu=args.gpu,
        )


if __name__ == "__main__":
    main()
