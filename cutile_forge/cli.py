"""Unified CLI entrypoint (`cutile-forge serve` and `cutile-forge bench`)."""

from __future__ import annotations

import argparse
import os
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(prog="cutile-forge", description="CuTileForge CLI")
    sub = parser.add_subparsers(dest="command", required=True)

    serve_p = sub.add_parser("serve", help="Start OpenAI-compatible inference server")
    serve_p.add_argument("--model", type=Path, required=True)
    serve_p.add_argument("--host", default="0.0.0.0")
    serve_p.add_argument("--port", type=int, default=8000)
    serve_p.add_argument("--target", default="sm_120")

    bench_p = sub.add_parser("bench", help="Run CuTileForge benchmark suite")
    bench_p.add_argument("--model", type=Path, default=Path("/home/arrhat/models/mistral_7b_4bit"))
    bench_p.add_argument("--output", type=Path, default=Path("benchmarks/results_5090"))
    bench_p.add_argument("--target", default="sm_120")
    bench_p.add_argument("--gpu", default="rtx5090")
    bench_p.add_argument("--micro-only", action="store_true")

    args = parser.parse_args()
    os.environ["CUTILE_ARCH"] = args.target

    if args.command == "serve":
        import uvicorn

        from cutile_forge.engine import InferenceEngine, Mistral7B, ModelConfig
        from cutile_forge.server import create_app

        tok = None
        if (args.model / "config.json").exists():
            from transformers import AutoTokenizer

            model = Mistral7B.from_pretrained(args.model, device="cuda")
            tok = AutoTokenizer.from_pretrained(str(args.model))
        else:
            model = Mistral7B.from_random(ModelConfig(num_hidden_layers=4), device="cuda")
        engine = InferenceEngine(model, target=args.target)
        app = create_app(engine, tokenizer=tok, model_id=args.model.name)
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
