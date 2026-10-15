"""Deployment entrypoint (Vercel / any ASGI host): exposes fraud.serve.app."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from fraud.serve import app  # noqa: E402

__all__ = ["app"]
