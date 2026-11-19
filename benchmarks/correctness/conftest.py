"""Fixtures for correctness tests."""

from __future__ import annotations

import os
from pathlib import Path

import pytest


@pytest.fixture(scope="session")
def model_dir() -> Path:
    env_dir = os.environ.get("CUTILE_MODEL_DIR")
    if env_dir:
        return Path(env_dir)
    if os.path.isdir("D:/models/mistral_7b_4bit"):
        return Path("D:/models/mistral_7b_4bit")
    return Path.home() / "models" / "mistral_7b_4bit"
