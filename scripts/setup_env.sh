#!/usr/bin/env bash
set -euo pipefail

pip install --upgrade pip
pip install -e ".[cuda,dev]"
echo "Environment setup complete."

