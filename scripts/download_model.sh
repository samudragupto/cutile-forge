#!/usr/bin/env bash
set -euo pipefail
python models/download.py --output "${1:-/mnt/d/models/mistral_7b_4bit/}"

