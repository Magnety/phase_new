"""Backward-compatible solver imports for the reorganized engine package."""

from .engine import PHASESolver, atomic_torch_save, atomic_write_json

__all__ = ["PHASESolver", "atomic_torch_save", "atomic_write_json"]
