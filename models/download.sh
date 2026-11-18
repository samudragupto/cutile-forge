#!/usr/bin/env bash
set -euo pipefail

OUT_DIR="${1:-/mnt/d/models/mistral_7b_4bit}"
echo "=== Downloading Mistral-7B 4-bit GPTQ to ${OUT_DIR} ==="
python models/download.py --output "${OUT_DIR}"
echo "=== Verifying ==="
ls -lh "${OUT_DIR}"
du -sh "${OUT_DIR}"
echo "=== Done ==="
