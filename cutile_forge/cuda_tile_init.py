"""Initialize cuda.tile with the configured target architecture before tensor allocation."""

from __future__ import annotations

import os

import cuda.tile as ct

if hasattr(ct, "init"):
    try:
        ct.init(arch=os.environ.get("CUTILE_ARCH", "sm_120"))
    except Exception:
        pass

