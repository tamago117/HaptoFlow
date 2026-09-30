"""Repo-root-relative path helpers, so saved artifacts carry no machine-specific paths."""
from __future__ import annotations

import os

# src/utils/paths.py → repo root is two levels above src/utils.
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))


def to_project_relative(path: str) -> str:
    """Relativize ``path`` against the repo root; paths outside it are returned unchanged."""
    path = str(path)
    if not os.path.isabs(path):
        return path
    rel = os.path.relpath(path, PROJECT_ROOT)
    return path if rel.startswith("..") else rel


def resolve_project_path(path: str) -> str:
    """Inverse of ``to_project_relative``: anchor a relative path at the repo root."""
    path = str(path)
    return path if os.path.isabs(path) else os.path.join(PROJECT_ROOT, path)
