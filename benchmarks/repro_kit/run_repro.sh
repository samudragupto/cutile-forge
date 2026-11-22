#!/bin/sh
set -eu

# CuTileForge Independent Reproduction Kit (v3.6.0)
# Usage:
#   sh benchmarks/repro_kit/run_repro.sh [/path/to/mistral_7b_4bit] [/var/tmp/arrhat_models/repro_kit_venv]

MODEL_DIR="${1:-/home/arrhat/models/mistral_7b_4bit}"
WORK_VENV="${2:-/var/tmp/arrhat_models/repro_kit_venv}"
OUT_DIR="${3:-benchmarks/validation_artifacts/repro_kit_run}"

export HF_HOME="${HF_HOME:-/var/tmp/arrhat_models/hf_cache}"
export CUTILE_ARCH="${CUTILE_ARCH:-sm_120}"

echo "[repro-kit] Creating clean venv at $WORK_VENV ..."
python3 -m venv "$WORK_VENV"
. "$WORK_VENV/bin/activate"

python3 -m pip install --upgrade pip build
python3 -m pip install --no-build-isolation --no-deps -e .
python3 -m pip install -r benchmarks/repro_kit/requirements-lock.txt

echo "[repro-kit] Verifying cutile-forge doctor --require-gpu ..."
cutile-forge doctor --require-gpu --json

echo "[repro-kit] Running fast CPU+GPU tests & 3-model fast regression ..."
python3 -m pytest tests/test_phase14_validation_schema.py tests/test_phase19_20_ecosystem.py -q
cutile-forge regress --tier fast --output "$OUT_DIR/regress_fast"

if [ -d "$MODEL_DIR" ]; then
    python3 benchmarks/production/bench_phase16_serving.py \
        --model-dir "$MODEL_DIR" \
        --output-dir "$OUT_DIR/serving_check" \
        --runs 1 || true
fi

echo "[repro-kit] Finished. Compare against benchmarks/repro_kit/expected_ranges.json."
