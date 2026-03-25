"""Row-centered JSONL ingestion helpers for inline-labeled datasets."""

from __future__ import annotations

import copy
import json
from dataclasses import dataclass
from pathlib import Path

from sft_label.fs_artifacts import is_ignored_fs_artifact
from sft_label.inline_labels import build_turn_id, compute_data_id
from sft_label.preprocessing import normalize_and_slice


@dataclass
class RowSampleBundle:
    """Original row plus its transient slice samples."""

    raw_row: dict
    source_path: Path
    row_number: int
    data_id: str
    samples: list[dict]


def build_row_sample_bundle(
    raw_row: dict,
    source_path,
    row_number: int,
    *,
    annotate_planner_metadata: bool = True,
) -> RowSampleBundle:
    """Create a row bundle with transient slice samples."""
    row_copy = copy.deepcopy(raw_row)
    bundle_samples = normalize_and_slice(
        copy.deepcopy(raw_row),
        source_file=source_path,
        source_row=row_number,
        annotate_planner_metadata=annotate_planner_metadata,
    )
    return RowSampleBundle(
        raw_row=row_copy,
        source_path=Path(source_path),
        row_number=row_number,
        data_id=compute_data_id(row_copy),
        samples=bundle_samples,
    )


def iter_row_sample_bundles_from_jsonl(
    input_path,
    limit: int = 0,
    *,
    annotate_planner_metadata: bool = True,
):
    """Yield row bundles from a JSONL file while preserving row order.

    `limit` follows the CLI's sample-oriented contract: it caps the number of
    transient slice samples, while still yielding whole source rows. If the next
    row would exceed the budget, iteration stops unless no row has been yielded
    yet, in which case the oversized first row is still included.
    """
    input_path = Path(input_path)
    if is_ignored_fs_artifact(input_path):
        raise ValueError(
            f"Detected macOS AppleDouble sidecar file ({input_path.name}). "
            "This is not a real JSONL input and should be ignored."
        )
    sample_count = 0
    yielded_rows = 0
    with open(input_path, "r", encoding="utf-8") as f:
        for line_number, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            bundle = build_row_sample_bundle(
                json.loads(line),
                input_path,
                line_number,
                annotate_planner_metadata=annotate_planner_metadata,
            )
            projected = sample_count + len(bundle.samples)
            if limit > 0 and yielded_rows > 0 and projected > limit:
                break
            yield bundle
            yielded_rows += 1
            sample_count = projected
            if limit > 0 and sample_count >= limit:
                break


def iter_row_sample_bundle_chunks_from_jsonl(input_path, chunk_size: int, limit: int = 0):
    """Yield row bundles in fixed-size chunks with sample-budget limiting."""
    chunk = []
    for bundle in iter_row_sample_bundles_from_jsonl(input_path, limit=limit):
        chunk.append(bundle)
        if len(chunk) >= chunk_size:
            yield chunk
            chunk = []
    if chunk:
        yield chunk


def flatten_row_sample_bundles(
    bundles: list[RowSampleBundle],
    *,
    assign_ids: bool = True,
):
    """Flatten row bundles into transient samples plus bundle lookup."""
    samples = []
    sample_to_bundle = []
    for bundle_idx, bundle in enumerate(bundles):
        for sample in bundle.samples:
            samples.append(sample)
            sample_to_bundle.append(bundle_idx)

    if assign_ids:
        assign_stable_sample_ids(bundles, samples, sample_to_bundle)

    return samples, sample_to_bundle


def assign_stable_sample_ids(
    bundles: list[RowSampleBundle],
    samples: list[dict],
    sample_to_bundle: list[int],
) -> None:
    """Assign deterministic turn ids for transient row-derived samples."""
    seen_per_bundle: dict[int, int] = {}
    for sample_idx, sample in enumerate(samples):
        bundle_idx = sample_to_bundle[sample_idx]
        bundle = bundles[bundle_idx]
        seen_count = seen_per_bundle.get(bundle_idx, 0) + 1
        seen_per_bundle[bundle_idx] = seen_count
        turn_index = (sample.get("metadata") or {}).get("turn_index") or seen_count
        sample["id"] = build_turn_id(bundle.data_id, turn_index)
