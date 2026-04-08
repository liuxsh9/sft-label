"""Shared combo-key helpers for rarity scoring (used by pipeline + scoring)."""

from __future__ import annotations


def _normalize_combo_dim(labels, dim, limit):
    """Normalize one combo-key dimension into a bounded list of string tags."""
    value = labels.get(dim)
    if isinstance(value, str):
        value = [value]
    if not isinstance(value, list):
        return []
    cleaned = sorted({item for item in value if isinstance(item, str) and item})
    if limit > 0:
        return cleaned[:limit]
    return cleaned


def _combo_key_from_labels(labels):
    """Build a richer combo key for multi-turn/code-SFT rarity."""
    parts = []
    for dim in ("intent", "difficulty", "context"):
        value = labels.get(dim)
        if isinstance(value, str) and value:
            parts.append(f"{dim}={value}")

    for dim, limit in (
        ("language", 2),
        ("domain", 2),
        ("task", 2),
        ("concept", 3),
        ("agentic", 2),
        ("constraint", 2),
    ):
        values = _normalize_combo_dim(labels, dim, limit)
        if values:
            parts.append(f"{dim}={','.join(values)}")

    if not parts:
        return None
    return "|".join(parts)
