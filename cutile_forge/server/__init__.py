"""OpenAI-compatible FastAPI serving layer for CuTileForge."""

from cutile_forge.server.app import create_app

__all__ = ["create_app"]

