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
    serve_p.add_argument("--max-batch-size", type=int, default=64)
    serve_p.add_argument("--batch-wait-ms", type=float, default=2.0)
    serve_p.add_argument("--max-prompt-tokens", type=int, default=2048)
    serve_p.add_argument("--native-runtime", action="store_true")

    bench_p = sub.add_parser("bench", help="Run CuTileForge benchmark suite")
    bench_p.add_argument("target_suite", nargs="?", default=None, help="Optional suite: release")
    bench_p.add_argument("--model", type=Path, default=Path("/home/arrhat/models/mistral_7b_4bit"))
    bench_p.add_argument("--output", type=Path, default=Path("benchmarks/results_5090"))
    bench_p.add_argument("--target", default="sm_120")
    bench_p.add_argument("--gpu", default="rtx5090")
    bench_p.add_argument("--micro-only", action="store_true")
    bench_p.add_argument("--soak-seconds", type=int, default=60)
    bench_p.add_argument("--bench-set", choices=["quick", "full"], default="quick")

    reg_p = sub.add_parser("regress", help="Run 8-model production correctness regression suite")
    reg_p.add_argument("--tier", choices=["fast", "full"], default="fast")
    reg_p.add_argument(
        "--output",
        type=Path,
        default=Path("benchmarks/validation_artifacts/v3.2.0/phase15_regression"),
    )
    reg_p.add_argument("--max-tokens", type=int, default=256)
    reg_p.add_argument("--num-prompts", type=int, default=5)
    reg_p.add_argument("--target", default="sm_120")

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

    imp_p = sub.add_parser(
        "import-hardware-report",
        help="Validate and import an external hardware report (e.g., RTX 4060, multi-GPU)",
    )
    imp_p.add_argument("path", type=Path, help="Path to manifest.json or artifact directory")
    imp_p.add_argument(
        "--dest",
        type=Path,
        default=Path("benchmarks/validation_artifacts/imported_hardware"),
        help="Destination directory for imported reports",
    )
    imp_p.add_argument(
        "--update-matrix",
        type=Path,
        default=None,
        help="Optional HARDWARE_SUPPORT_MATRIX.md path to update on successful import",
    )

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
            import torch

            from cutile_forge.runtime import NativeModelRuntime

            req_seq = max(512, args.max_prompt_tokens + 256)
            if torch.cuda.is_available():
                free_b, _ = torch.cuda.mem_get_info()
                bytes_per_slot_tok = (
                    model.config.num_hidden_layers
                    * model.config.num_key_value_heads
                    * model.config.head_dim
                    * 4
                )
                max_fit_seq = int(
                    (free_b * 0.78) // max(1, args.max_batch_size * bytes_per_slot_tok)
                )
                if max_fit_seq >= 512:
                    req_seq = min(req_seq, max_fit_seq)
            rt = NativeModelRuntime(
                model,
                max_seq_len=req_seq,
                use_cuda_graph=True,
                max_batch_size=args.max_batch_size,
            )
            # Pre-capture batch buckets up to args.max_batch_size for first context bucket
            if rt.use_cuda_graph and rt._all_layers_on_gpu:
                first_ctx = rt.context_buckets[0]
                for b_cap in (1, 2, 4, 8, 16, 32, 64):
                    if b_cap <= args.max_batch_size:
                        ws = rt._get_or_create_workspace(b_cap)
                        _ = rt._capture_full_step_graph(ws, first_ctx)
            engine: Any = _NativeServingAdapter(rt, target=args.target)
        else:
            engine = InferenceEngine(model, target=args.target)
        app = create_app(
            engine,
            tokenizer=tok,
            model_id=args.model.name,
            max_batch_size=args.max_batch_size,
            batch_wait_ms=args.batch_wait_ms,
            api_key=args.api_key,
            max_concurrent_requests=args.max_concurrent,
            max_prompt_tokens=args.max_prompt_tokens,
        )
        uvicorn.run(app, host=args.host, port=args.port, timeout_keep_alive=60, access_log=False)
    elif args.command == "bench":
        import importlib

        if args.target_suite == "release":
            old_argv = sys.argv
            sys.argv = [
                "run_release_validation.py",
                "--model-dir",
                str(args.model),
                "--soak-seconds",
                str(args.soak_seconds),
                "--bench-set",
                args.bench_set,
            ]
            try:
                importlib.import_module("benchmarks.production.run_release_validation").main()
            finally:
                sys.argv = old_argv
        else:
            run_all = importlib.import_module("benchmarks.run_all").run_all

            run_all(
                model_dir=args.model,
                output_dir=args.output,
                micro_only=args.micro_only,
                target=args.target,
                gpu=args.gpu,
            )
    elif args.command == "regress":
        from cutile_forge.regress import run_regression_suite

        reg_report = run_regression_suite(
            tier=args.tier,
            output_dir=args.output,
            num_prompts=args.num_prompts,
            greedy_tokens=args.max_tokens,
        )
        print(reg_report.format_markdown())
        sys.exit(0 if reg_report.all_non_native_or_pass else 1)
    elif args.command == "import-hardware-report":
        from cutile_forge.hardware_reports import import_hardware_report

        result = import_hardware_report(
            report_path=args.path,
            dest_dir=args.dest,
            matrix_path=args.update_matrix,
        )
        print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
