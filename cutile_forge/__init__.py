"""CuTileForge — Tile-Fusion Graph Compiler & CUDA Tile Ecosystem."""

from cutile_forge.doctor import run_diagnostics
from cutile_forge.forge.fx_backend import register_cutile_forge_backend
from cutile_forge.sdk import Device, DType, Graph, compile

__version__ = "3.2.0"
version = __version__

register_cutile_forge_backend()

__all__ = [
    "DType",
    "Device",
    "Graph",
    "__version__",
    "compile",
    "register_cutile_forge_backend",
    "run_diagnostics",
    "version",
]
