"""Helpers for ignoring OS-generated filesystem artifact files."""

from __future__ import annotations

from pathlib import Path


def is_ignored_fs_artifact(path: str | Path) -> bool:
    """Return whether a path is an OS-generated artifact, not user data."""
    name = Path(path).name
    return name.startswith("._") or name == ".DS_Store"
