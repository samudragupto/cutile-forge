"""Hand-written CUDA C++ and cuBLAS baseline kernels."""

from __future__ import annotations

import importlib.util
from pathlib import Path
from types import ModuleType

_CACHED_MOD: ModuleType | None = None


def load_baselines(build_if_missing: bool = True) -> ModuleType:
    """Load the compiled `_cutile_baselines` pybind11 CUDA extension."""
    global _CACHED_MOD
    if _CACHED_MOD is not None:
        return _CACHED_MOD

    baselines_dir = Path(__file__).resolve().parent
    candidates = list(baselines_dir.glob("_cutile_baselines*.pyd")) + list(
        baselines_dir.glob("_cutile_baselines*.so")
    )
    if not candidates and build_if_missing:
        from cutile_forge.baselines.build import build_baselines

        built = build_baselines(verbose=False)
        candidates = [built]

    if not candidates:
        raise ImportError(
            "CUDA baselines extension not built. Run `python -m cutile_forge.baselines.build`."
        )

    spec = importlib.util.spec_from_file_location("_cutile_baselines", candidates[0])
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to load extension from {candidates[0]}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    _CACHED_MOD = mod
    return mod


__all__ = ["load_baselines"]
