#!/bin/sh
set -eu

# One-command validation runner for RTX 4060 Laptop (sm_89, 8 GB VRAM).
# Until executed on real RTX 4060 hardware and imported, laptop support remains UNVALIDATED.
#
# Usage:
#   sh scripts/run_laptop_4060.sh [/path/to/mistral_7b_4bit] [output_root] [quick|full]

MODEL_DIR="${1:-/mnt/d/models/mistral_7b_4bit}"
OUTPUT_ROOT="${2:-benchmarks/validation_artifacts/laptop4060}"
BENCH_SET="${3:-quick}"

export CUTILE_ARCH="sm_89"
export CUTILE_GPU_VRAM_GB="8.0"
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"

echo "[laptop4060] Target=sm_89 VRAM_budget=8.0GB model=$MODEL_DIR output=$OUTPUT_ROOT bench_set=$BENCH_SET"
nvidia-smi -L || true

sh scripts/reproduce_validation.sh \
    --profile laptop4060 \
    --model-dir "$MODEL_DIR" \
    --output-root "$OUTPUT_ROOT" \
    --bench-set "$BENCH_SET"

echo "[laptop4060] Done. Import your hardware report back into the repository with:"
echo "  cutile-forge import-hardware-report \"$OUTPUT_ROOT/phase14f_serving_stress/manifest.json\" --update-matrix docs/HARDWARE_SUPPORT_MATRIX.md"
