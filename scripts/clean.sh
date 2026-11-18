#!/usr/bin/env bash
set -euo pipefail
rm -rf cutile_forge/baselines/build/ cutile_forge/baselines/*.so cutile_forge/baselines/*.pyd
rm -rf build/ dist/ *.egg-info
find . -name __pycache__ -exec rm -rf {} +

