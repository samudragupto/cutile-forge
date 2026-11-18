"""Fixtures for correctness tests."""

from __future__ import annotations

import os
from pathlib import Path

import pytest


@pytest.fixture(scope="session")
def model_dir() -> Path:
    return Path(os.environ.get("CUTILE_MODEL_DIR", r"D:\models\mistral_7b_4bit"))
