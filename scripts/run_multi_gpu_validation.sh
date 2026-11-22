#!/bin/sh
set -eu

# Physical Multi-GPU Validation Runner (runs unchanged on >= 2 GPU nodes).
# Emits MEASURED NCCL/all-reduce results when >= 2 physical GPUs are attached;
# marks physical multi-GPU scaling BLOCKED: hardware when < 2 GPUs exist.
#
# Usage:
#   sh scripts/run_multi_gpu_validation.sh [output_dir]

OUTPUT_DIR="${1:-benchmarks/validation_artifacts/multi_gpu_physical}"
VENV_DIR="${VENV_DIR:-.venv}"

if [ -f "$VENV_DIR/bin/activate" ]; then
    . "$VENV_DIR/bin/activate"
fi

echo "=== nvidia-smi -L ==="
nvidia-smi -L
GPU_COUNT="$(nvidia-smi -L | wc -l | tr -d ' ')"

if [ "$GPU_COUNT" -lt 2 ]; then
    echo "BLOCKED: hardware — detected $GPU_COUNT physical GPU(s); >= 2 physical GPUs required for NCCL/TP/PP/EP scaling." >&2
    if [ "${ALLOW_SINGLE_GPU_SIMULATION:-0}" != "1" ]; then
        exit 2
    fi
fi

if [ "$GPU_COUNT" -ge 2 ]; then
    echo "=== nvidia-smi topo -m ==="
    nvidia-smi topo -m || true
fi

export CUTILE_ARCH="${CUTILE_ARCH:-sm_120}"
export HF_HOME="${HF_HOME:-/var/tmp/arrhat_models/hf_cache}"
export PYTHONPATH="${PYTHONPATH:-.}"

python3 benchmarks/production/bench_multi_gpu_validation.py --output-dir "$OUTPUT_DIR"

echo "=== Completed Multi-GPU Validation -> $OUTPUT_DIR/manifest.json ==="
