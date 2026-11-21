"""Track B benchmark: NativeModelRuntime vs. InferenceEngine on real Mistral-7B-4bit."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import torch
from torch.utils._python_dispatch import TorchDispatchMode

from cutile_forge.bench.harness import write_csv
from cutile_forge.engine.config import GPUConfig
from cutile_forge.engine.inference import InferenceEngine
from cutile_forge.engine.model import Mistral7B
from cutile_forge.runtime import NativeModelRuntime

FORBIDDEN_ATEN_COMPUTE = {
    "aten.mm.default",
    "aten.bmm.default",
    "aten.addmm.default",
    "aten.matmul.default",
    "aten.embedding.default",
    "aten.silu.default",
    "aten.gelu.default",
    "aten.layer_norm.default",
    "aten.scaled_dot_product_attention.default",
    "aten.topk.default",
    "aten.cat.default",
}


class _AuditMode(TorchDispatchMode):
    def __init__(self) -> None:
        super().__init__()
        self.forbidden: list[str] = []

    def __torch_dispatch__(
        self,
        func: Any,
        types: tuple[type, ...],
        args: tuple[Any, ...] = (),
        kwargs: dict[str, Any] | None = None,
    ) -> Any:
        name = str(func)
        if name in FORBIDDEN_ATEN_COMPUTE:
            self.forbidden.append(name)
        return func(*args, **(kwargs or {}))


def run_benchmark(
    model_dir: Path | None = None,
    output_csv: Path = Path("benchmarks/results_5090/track_b_native_runtime.csv"),
) -> list[dict[str, object]]:
    if model_dir is None:
        model_dir = Path.home() / "models" / "mistral_7b_4bit"
    if not torch.cuda.is_available() or not model_dir.exists():
        raise RuntimeError(f"CUDA + checkpoint {model_dir} required for Track B benchmark")

    model = Mistral7B.from_pretrained(
        model_dir,
        gpu_config=GPUConfig(vram_gb=32.0, stream_weights=False),
        device="cuda",
        max_gpu_layers=32,
    )

    prompt_ids = [1, 415, 5565, 302, 4843, 349]  # <s> The capital of France is
    prompt_t = torch.tensor([prompt_ids], device="cuda", dtype=torch.long)

    # 1. Legacy v2.0.0 InferenceEngine (CUDA Graph throughput + eager ATen audit)
    eng = InferenceEngine(model, target="sm_120", use_cuda_graph=True, enable_radix_cache=False)
    _ = eng.generate_ids(prompt_t, max_new_tokens=8, temperature=0.0, backend="cutile")
    torch.cuda.reset_peak_memory_stats()
    eng_tokens = eng.generate_ids(prompt_t, max_new_tokens=32, temperature=0.0, backend="cutile")
    eng_stats = eng.last_stats
    assert eng_stats is not None
    eng_vram = round(torch.cuda.max_memory_allocated() / (1024**3), 2)

    eng_eager = InferenceEngine(
        model, target="sm_120", use_cuda_graph=False, enable_radix_cache=False
    )
    logits_p = eng_eager.prefill(prompt_t, backend="cutile")
    next_id = logits_p[:, -1, :].argmax(dim=-1, keepdim=True)
    with _AuditMode() as eng_audit:
        _ = eng_eager.decode_step(next_id, pos=len(prompt_ids), backend="cutile")

    # 2. Track B NativeModelRuntime (Zero-ATen compute hot path)
    rt = NativeModelRuntime(model, max_seq_len=512, kv_dtype="fp16", use_cuda_graph=True)
    _, _ = rt.generate(prompt_ids, max_new_tokens=8)
    torch.cuda.reset_peak_memory_stats()
    nat_tokens, nat_stats = rt.generate(prompt_ids, max_new_tokens=32)
    nat_vram = round(torch.cuda.max_memory_allocated() / (1024**3), 2)

    ntok = rt.prefill(prompt_t)
    with _AuditMode() as nat_audit:
        _ = rt.decode_step(ntok, pos=len(prompt_ids), use_graph=False)

    exact_match = eng_tokens[:32] == nat_tokens[:32]
    rows: list[dict[str, object]] = [
        {
            "runtime": "v2.0.0_InferenceEngine",
            "aten_compute_calls_per_decode": len(eng_audit.forbidden),
            "decode_tok_s": round(eng_stats.tokens_per_sec, 2),
            "ms_per_token": round(eng_stats.ms_per_token, 2),
            "prefill_ms": round(eng_stats.prefill_ms, 2),
            "peak_vram_gb": eng_vram,
            "token_parity_32": exact_match,
        },
        {
            "runtime": "v2.2.0_NativeModelRuntime",
            "aten_compute_calls_per_decode": len(nat_audit.forbidden),
            "decode_tok_s": nat_stats["decode_tok_s"],
            "ms_per_token": nat_stats["ms_per_token"],
            "prefill_ms": nat_stats["prefill_ms"],
            "peak_vram_gb": nat_vram,
            "token_parity_32": exact_match,
        },
    ]
    write_csv(output_csv, rows)
    print(f"Exact 32-token greedy parity: {exact_match}")
    print(
        f"InferenceEngine decode ATen compute count: {len(eng_audit.forbidden)} ({set(eng_audit.forbidden)})"
    )
    print(f"NativeModelRuntime decode ATen compute count: {len(nat_audit.forbidden)}")
    for r in rows:
        print(r)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Track B NativeModelRuntime benchmark")
    parser.add_argument(
        "--model-dir", type=Path, default=Path.home() / "models" / "mistral_7b_4bit"
    )
    parser.add_argument(
        "--output", type=Path, default=Path("benchmarks/results_5090/track_b_native_runtime.csv")
    )
    args = parser.parse_args()
    run_benchmark(model_dir=args.model_dir, output_csv=args.output)


if __name__ == "__main__":
    main()
