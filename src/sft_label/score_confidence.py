"""Helpers for confidence-aware score aggregation."""


SCORE_CONFIDENCE_ANCHOR = 5.5


def _coerce_unit_float(value, default):
    """Clamp a possibly-invalid float-like value into [0, 1]."""
    if isinstance(value, bool):
        return default
    if isinstance(value, (int, float)):
        return max(0.0, min(1.0, float(value)))
    return default


def score_confidence(score_like, default=1.0):
    """Extract scorer confidence from a score/value dict."""
    if not isinstance(score_like, dict):
        return default
    return _coerce_unit_float(score_like.get("confidence"), default)


def apply_score_confidence(value, confidence, anchor=SCORE_CONFIDENCE_ANCHOR):
    """Conservatively shrink uncertain high scores toward a neutral anchor."""
    if value is None:
        return None
    if confidence is None:
        return value
    confidence = _coerce_unit_float(confidence, 1.0)
    if value <= anchor:
        return value
    return anchor + confidence * (value - anchor)
