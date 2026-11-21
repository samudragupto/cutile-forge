#!/bin/sh
set -eu

# CuTileForge Phase 14+ Reproducibility Runner (POSIX dash-compatible)
# Usage:
#   sh scripts/reproduce_validation.sh --profile rtx5090 --model-dir /home/arrhat/models/mistral_7b_4bit
#   sh scripts/reproduce_validation.sh --profile laptop4060 --model-dir /mnt/d/models/mistral_7b_4bit
#   sh scripts/reproduce_validation.sh --profile cpu_ci

PROFILE="rtx5090"
MODEL_DIR="/home/arrhat/models/mistral_7b_4bit"
OUTPUT_ROOT="benchmarks/validation_artifacts/repro"
VENV_DIR=".venv"
BENCH_SET="quick"

while [ "$#" -gt 0 ]; do
    case "$1" in
        --profile)
            PROFILE="$2"
            shift 2
            ;;
        --model-dir)
            MODEL_DIR="$2"
            shift 2
            ;;
        --output-root)
            OUTPUT_ROOT="$2"
            shift 2
            ;;
        --venv)
            VENV_DIR="$2"
            shift 2
            ;;
        --bench-set)
            BENCH_SET="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1" >&2
            exit 2
            ;;
    esac
done

if [ -f "$VENV_DIR/bin/activate" ]; then
    . "$VENV_DIR/bin/activate"
fi

export HF_HOME="${HF_HOME:-/var/tmp/arrhat_models/hf_cache}"

case "$PROFILE" in
    rtx5090)
        export CUTILE_ARCH="sm_120"
        ;;
    laptop4060)
        export CUTILE_ARCH="sm_89"
        ;;
    cpu_ci)
        export CUDA_VISIBLE_DEVICES=""
        ;;
    *)
        export CUTILE_ARCH="${CUTILE_ARCH:-sm_120}"
        ;;
esac

echo "[reproduce] profile=$PROFILE output_root=$OUTPUT_ROOT bench_set=$BENCH_SET"
python3 -m pytest tests/test_phase14_validation_schema.py tests/test_phase14_reliability.py -q

if [ "$PROFILE" = "cpu_ci" ]; then
    CUDA_VISIBLE_DEVICES="" python3 -m pytest tests/ -q
    echo "[reproduce] CPU-only CI validation complete."
    exit 0
fi

python3 -c "import torch; assert torch.cuda.is_available(), 'CUDA GPU required for profile ' + '$PROFILE'"
python3 benchmarks/production/run_release_validation.py \
    --profile "$PROFILE" \
    --model-dir "$MODEL_DIR" \
    --output-root "$OUTPUT_ROOT" \
    --bench-set "$BENCH_SET"
