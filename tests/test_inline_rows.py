"""Tests for row-centered JSONL ingestion helpers."""

from __future__ import annotations

import json

from sft_label.inline_labels import build_turn_id
from sft_label.inline_rows import (
    build_row_sample_bundle,
    flatten_row_sample_bundles,
    iter_row_sample_bundle_chunks_from_jsonl,
    iter_row_sample_bundles_from_jsonl,
)
from sft_label.pipeline import iter_samples_from_file


def _write_jsonl(path, rows):
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def test_build_row_sample_bundle_preserves_row_metadata(tmp_path):
    path = tmp_path / "data.jsonl"
    row = {
        "id": "row-1",
        "meta_prompt": ["You are helpful"],
        "data": [
            {"role": "user", "content": "q"},
            {"role": "assistant", "content": "a"},
        ],
    }
    path.write_text("")

    bundle = build_row_sample_bundle(row, path, 3)

    assert bundle.source_path == path
    assert bundle.row_number == 3
    assert bundle.raw_row == row
    assert bundle.data_id.startswith("sha256:v1:")
    assert len(bundle.samples) == 1


def test_iter_row_sample_bundles_from_jsonl_preserves_order_and_line_numbers(tmp_path):
    path = tmp_path / "rows.jsonl"
    rows = [
        {
            "id": "r1",
            "data": [
                {"role": "user", "content": "q1"},
                {"role": "assistant", "content": "a1"},
            ],
        },
        {
            "id": "r2",
            "data": [
                {"role": "user", "content": "q2"},
                {"role": "assistant", "content": "a2"},
                {"role": "user", "content": "q3"},
                {"role": "assistant", "content": "a3"},
            ],
        },
    ]
    _write_jsonl(path, rows)

    bundles = list(iter_row_sample_bundles_from_jsonl(path))

    assert [bundle.row_number for bundle in bundles] == [1, 2]
    assert [bundle.raw_row["id"] for bundle in bundles] == ["r1", "r2"]
    assert len(bundles[0].samples) == 1
    assert len(bundles[1].samples) == 2


def test_flatten_row_sample_bundles_tracks_bundle_membership(tmp_path):
    path = tmp_path / "rows.jsonl"
    rows = [
        {
            "id": "r1",
            "data": [
                {"role": "user", "content": "q1"},
                {"role": "assistant", "content": "a1"},
            ],
        },
        {
            "id": "r2",
            "data": [
                {"role": "user", "content": "q2"},
                {"role": "assistant", "content": "a2"},
                {"role": "user", "content": "q3"},
                {"role": "assistant", "content": "a3"},
            ],
        },
    ]
    _write_jsonl(path, rows)
    bundles = list(iter_row_sample_bundles_from_jsonl(path))

    samples, sample_to_bundle = flatten_row_sample_bundles(bundles)

    assert len(samples) == 3
    assert sample_to_bundle == [0, 1, 1]
    assert samples[0]["id"] == build_turn_id(bundles[0].data_id, 1)
    assert samples[1]["id"] == build_turn_id(bundles[1].data_id, 1)
    assert samples[2]["id"] == build_turn_id(bundles[1].data_id, 2)


def test_iter_row_sample_bundle_chunks_from_jsonl_batches_rows(tmp_path):
    path = tmp_path / "rows.jsonl"
    rows = []
    for idx in range(5):
        rows.append(
            {
                "id": f"r{idx}",
                "data": [
                    {"role": "user", "content": f"q{idx}"},
                    {"role": "assistant", "content": f"a{idx}"},
                ],
            }
        )
    _write_jsonl(path, rows)

    chunks = list(iter_row_sample_bundle_chunks_from_jsonl(path, chunk_size=2))

    assert [len(chunk) for chunk in chunks] == [2, 2, 1]
    assert chunks[1][0].row_number == 3


def test_iter_samples_from_file_can_return_row_bundles_for_jsonl(tmp_path):
    path = tmp_path / "rows.jsonl"
    rows = [
        {
            "id": "r1",
            "data": [
                {"role": "user", "content": "q1"},
                {"role": "assistant", "content": "a1"},
            ],
        },
        {
            "id": "r2",
            "data": [
                {"role": "user", "content": "q2"},
                {"role": "assistant", "content": "a2"},
                {"role": "user", "content": "q3"},
                {"role": "assistant", "content": "a3"},
            ],
        },
    ]
    _write_jsonl(path, rows)

    samples, n_raw, bundles, sample_to_bundle = iter_samples_from_file(
        path,
        return_row_bundles=True,
    )

    assert n_raw == 2
    assert len(samples) == 3
    assert len(bundles) == 2
    assert sample_to_bundle == [0, 1, 1]


def test_iter_row_sample_bundles_limit_is_sample_based(tmp_path):
    path = tmp_path / "rows.jsonl"
    rows = [
        {
            "id": "r1",
            "data": [
                {"role": "user", "content": "q1"},
                {"role": "assistant", "content": "a1"},
            ],
        },
        {
            "id": "r2",
            "data": [
                {"role": "user", "content": "q2"},
                {"role": "assistant", "content": "a2"},
                {"role": "user", "content": "q3"},
                {"role": "assistant", "content": "a3"},
            ],
        },
    ]
    _write_jsonl(path, rows)

    bundles = list(iter_row_sample_bundles_from_jsonl(path, limit=2))

    assert len(bundles) == 1
    assert bundles[0].raw_row["id"] == "r1"
    assert len(bundles[0].samples) == 1
