"""Helpers for inline label migration and copy-forward by data_id."""

from __future__ import annotations

import copy
import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from sft_label.inline_labels import (
    compute_data_id,
    get_data_id,
    get_data_label,
    mark_migrated,
    set_data_id,
    set_data_label,
    set_meta_field,
)
from sft_label.inline_scoring import discover_inline_jsonl_files, infer_inline_scoring_target


@dataclass(frozen=True)
class InlineMigrationMatch:
    """Copied inline annotation payload plus provenance."""

    data_id: str
    data_label: dict
    source_ref: str


def load_inline_migration_index(input_path) -> tuple[dict[str, InlineMigrationMatch], dict]:
    """Build a copy-forward index from an inline mirrored dataset."""
    target = infer_inline_scoring_target(input_path)
    if target is None:
        raise ValueError(
            f"Migration source must be an inline-labeled run root, dataset root, or JSONL file: {input_path}"
        )

    index: dict[str, InlineMigrationMatch] = {}
    total_rows = 0
    labeled_rows = 0
    duplicate_ids = 0

    for source_file in discover_inline_jsonl_files(target):
        rel_path = target.layout.relative_source_path(source_file)
        with open(source_file, "r", encoding="utf-8") as f:
            for row_number, line in enumerate(f, start=1):
                stripped = line.strip()
                if not stripped:
                    continue
                total_rows += 1
                row = json.loads(stripped)
                data_label = get_data_label(row)
                if not isinstance(data_label, dict):
                    continue
                data_id = get_data_id(row) or compute_data_id(row)

                labeled_rows += 1
                source_ref = f"{rel_path}:{row_number}"
                match = InlineMigrationMatch(
                    data_id=data_id,
                    data_label=copy.deepcopy(data_label),
                    source_ref=source_ref,
                )
                if data_id in index:
                    duplicate_ids += 1
                    continue
                index[data_id] = match

    stats = {
        "source_input": str(Path(input_path).resolve()),
        "indexed_rows": labeled_rows,
        "total_rows": total_rows,
        "unique_data_ids": len(index),
        "duplicate_data_ids": duplicate_ids,
    }
    return index, stats


def seed_row_from_migration(row: dict, match: InlineMigrationMatch, *,
                            timestamp: str | None = None) -> dict:
    """Copy a matched inline data_label onto a target row."""
    now = timestamp or datetime.now().isoformat()
    row_copy = copy.deepcopy(row)
    set_data_id(row_copy, match.data_id)
    data_label = set_data_label(row_copy, match.data_label)
    mark_migrated(data_label, migration_source=match.source_ref, timestamp=now)
    set_meta_field(data_label, "mode", "migrate")
    return row_copy
