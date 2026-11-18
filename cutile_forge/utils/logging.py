"""Structured logging utilities."""

from __future__ import annotations

import logging
import sys


def get_logger(name: str = "cutile_forge", level: int = logging.INFO) -> logging.Logger:
    """Return a configured stream logger without duplicate handlers."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        fmt = logging.Formatter("[%(asctime)s] %(levelname)s %(name)s: %(message)s", "%H:%M:%S")
        handler.setFormatter(fmt)
        logger.addHandler(handler)
    logger.setLevel(level)
    return logger
