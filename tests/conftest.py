"""Pytest fixtures for CuTileForge unit and correctness tests."""

from __future__ import annotations

import os
from pathlib import Path

import pytest
import torch


def has_cuda() -> bool:
    return torch.cuda.is_available()


@pytest.fixture(scope="session")
def cuda_available() -> bool:
    return torch.cuda.is_available()


@pytest.fixture(scope="session")
def model_dir() -> Path:
    return Path(os.environ.get("CUTILE_MODEL_DIR", r"D:\models\mistral_7b_4bit"))


@pytest.fixture
def small_fp16_tensors() -> tuple[torch.Tensor, torch.Tensor]:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    a = torch.randn((16, 64), device=device, dtype=torch.float16) * 0.1
    b = torch.randn((64, 64), device=device, dtype=torch.float16) * 0.1
    return a, b
