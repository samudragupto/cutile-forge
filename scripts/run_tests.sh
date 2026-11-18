#!/usr/bin/env bash
set -euo pipefail
pytest tests/ -v --tb=short
pytest benchmarks/correctness/ -v --tb=short

