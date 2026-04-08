"""Shared I/O and numeric-summary utilities.

Extracted from pipeline.py / scoring.py to eliminate duplication.
"""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path


# ---------------------------------------------------------------------------
# Atomic file writers
# ---------------------------------------------------------------------------

def _write_json_atomic(path, payload):
    """Write JSON via a temp file and atomic replace.

    Uses ``tempfile.mkstemp`` so the temp file lives on the same filesystem as
    *path* and is cleaned up even when writing fails.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=str(path.parent),
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        os.replace(tmp_path, path)
    finally:
        try:
            os.unlink(tmp_path)
        except FileNotFoundError:
            pass


def _write_jsonl_atomic(path, records):
    """Write JSONL via a temp file and atomic replace.

    Uses ``tempfile.mkstemp`` so the temp file lives on the same filesystem as
    *path* and is cleaned up even when writing fails.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=str(path.parent),
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            for record in records:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
        os.replace(tmp_path, path)
    finally:
        try:
            os.unlink(tmp_path)
        except FileNotFoundError:
            pass


# ---------------------------------------------------------------------------
# Numeric helpers
# ---------------------------------------------------------------------------

def _numeric_summary(values):
    """Compute min / max / mean / sum / count over a list of numbers."""
    cleaned = [float(v) for v in values if isinstance(v, (int, float)) and not isinstance(v, bool)]
    if not cleaned:
        return {}
    total = sum(cleaned)
    return {
        "count": len(cleaned),
        "sum": round(total, 3),
        "mean": round(total / len(cleaned), 3),
        "min": round(min(cleaned), 3),
        "max": round(max(cleaned), 3),
    }


def _message_payload_bytes(messages):
    """Return UTF-8 payload size for the full request body."""
    return len(json.dumps(messages, ensure_ascii=False).encode("utf-8"))
