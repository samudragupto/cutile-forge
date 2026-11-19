"""CuTileForge — Tile-Fusion Graph Compiler for 8 GB-VRAM & 32 GB-VRAM LLM Inference."""

from cutile_forge.forge.fx_backend import register_cutile_forge_backend

__version__ = "1.0.0"
version = __version__

register_cutile_forge_backend()

__all__ = ["__version__", "register_cutile_forge_backend", "version"]
