"""Helpers for label metadata and downstream eligibility."""

from __future__ import annotations


LABEL_META_KEYS = {
    "confidence",
    "inherited",
    "inherited_from",
    "partial",
    "partial_stage",
    "partial_reason",
}


def is_partial_labels(labels):
    """Return True when labels only contain a partial Pass 1 result."""
    return isinstance(labels, dict) and labels.get("partial") is True


def is_usable_labels(labels):
    """Return True when labels can participate in downstream scoring/stats."""
    return isinstance(labels, dict) and not is_partial_labels(labels)
