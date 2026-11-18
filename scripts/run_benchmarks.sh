#!/usr/bin/env bash
set -euo pipefail
python benchmarks/run_all.py \
  --model "${1:-/mnt/d/models/mistral_7b_4bit/}" \
  --output benchmarks/results/ \
  --warmup 5 --iters 200

