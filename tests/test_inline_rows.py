"""Tests for row-centered JSONL ingestion helpers."""

from __future__ import annotations

import json
import pytest

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


def test_iter_row_sample_bundles_supports_conversations_role_content_single_and_multi_turn_rows(tmp_path):
    path = tmp_path / "openai-conversations.jsonl"
    rows = [
        {
            "id": "oa-conv-single",
            "conversations": [
                {"role": "user", "content": "q1"},
                {"role": "assistant", "content": "a1"},
            ],
        },
        {
            "id": "oa-conv-multi",
            "conversations": [
                {"role": "user", "content": "q2"},
                {"role": "assistant", "content": "a2"},
                {"role": "user", "content": "q3"},
                {"role": "assistant", "content": "a3"},
            ],
        },
    ]
    _write_jsonl(path, rows)

    bundles = list(iter_row_sample_bundles_from_jsonl(path))

    assert len(bundles) == 2
    assert len(bundles[0].samples) == 1
    assert bundles[0].samples[0]["metadata"]["original_format"] == "openai_conversations"
    assert bundles[0].samples[0]["conversations"] == [
        {"from": "human", "value": "q1"},
        {"from": "gpt", "value": "a1"},
    ]
    assert len(bundles[1].samples) == 2
    assert [s["metadata"]["turn_index"] for s in bundles[1].samples] == [1, 2]
    assert [s["metadata"]["total_turns"] for s in bundles[1].samples] == [2, 2]


def test_iter_row_sample_bundles_supports_messages_role_content_rows_and_none_content(tmp_path):
    path = tmp_path / "openai-messages.jsonl"
    rows = [
        {
            "id": "oa-msg-none",
            "messages": [
                {"role": "user", "content": None},
                {"role": "assistant", "content": None},
            ],
        },
        {
            "id": "oa-msg-multi",
            "messages": [
                {"role": "user", "content": "q1"},
                {"role": "assistant", "content": "a1"},
                {"role": "user", "content": "q2"},
                {"role": "assistant", "content": "a2"},
            ],
        },
    ]
    _write_jsonl(path, rows)

    bundles = list(iter_row_sample_bundles_from_jsonl(path))

    assert len(bundles) == 2
    assert bundles[0].samples[0]["metadata"]["original_format"] == "openai_messages"
    assert bundles[0].samples[0]["conversations"] == [
        {"from": "human", "value": ""},
        {"from": "gpt", "value": ""},
    ]
    assert len(bundles[1].samples) == 2


def test_iter_row_sample_bundles_handles_long_trajectory_with_system_tool_and_three_assistant_replies(tmp_path):
    path = tmp_path / "openai-long-trajectory.jsonl"
    rows = [
        {
            "id": "oa-long",
            "messages": [
                {"role": "system", "content": "policy"},
                {"role": "user", "content": "q1"},
                {"role": "assistant", "content": "a1"},
                {"role": "tool", "content": "tool-r1"},
                {"role": "assistant", "content": "a2"},
                {"role": "tool", "content": "tool-r2"},
                {"role": "assistant", "content": "a3"},
            ],
        }
    ]
    _write_jsonl(path, rows)

    bundle = list(iter_row_sample_bundles_from_jsonl(path))[0]

    assert len(bundle.samples) == 3
    assert [s["metadata"]["turn_index"] for s in bundle.samples] == [1, 2, 3]
    assert [s["metadata"]["total_turns"] for s in bundle.samples] == [3, 3, 3]
    assert bundle.samples[2]["conversations"][-2] == {"from": "tool", "value": "tool-r2"}
    assert bundle.samples[2]["conversations"][-1] == {"from": "gpt", "value": "a3"}


def test_row_ingestion_mixed_schema_file_reader_and_flatten_keep_ids_and_turn_metadata(tmp_path):
    path = tmp_path / "mixed.jsonl"
    rows = [
        {
            "id": "pangu",
            "data": [
                {"role": "user", "content": "q0"},
                {"role": "assistant", "content": "a0"},
            ],
        },
        {
            "id": "conv-openai",
            "conversations": [
                {"role": "user", "content": "q1"},
                {"role": "assistant", "content": "a1"},
                {"role": "user", "content": "q2"},
                {"role": "assistant", "content": "a2"},
            ],
        },
        {
            "id": "fallback-to-messages",
            "data": [],
            "conversations": [{"role": "user", "content": ["invalid"]}],
            "messages": [
                {"role": "user", "content": "q3"},
                {"role": "assistant", "content": "a3"},
            ],
        },
    ]
    _write_jsonl(path, rows)

    bundles = list(iter_row_sample_bundles_from_jsonl(path))
    samples, sample_to_bundle = flatten_row_sample_bundles(bundles)

    assert len(bundles) == 3
    assert bundles[0].samples[0]["metadata"]["original_format"] == "pangu"
    assert bundles[1].samples[0]["metadata"]["original_format"] == "openai_conversations"
    assert bundles[2].samples[0]["metadata"]["original_format"] == "openai_messages"
    assert bundles[2].samples[0]["conversations"] == [
        {"from": "human", "value": "q3"},
        {"from": "gpt", "value": "a3"},
    ]
    assert sample_to_bundle == [0, 1, 1, 2]
    assert samples[0]["id"] == build_turn_id(bundles[0].data_id, 1)
    assert samples[1]["id"] == build_turn_id(bundles[1].data_id, 1)
    assert samples[2]["id"] == build_turn_id(bundles[1].data_id, 2)
    assert samples[3]["id"] == build_turn_id(bundles[2].data_id, 1)
    assert [samples[1]["metadata"]["turn_index"], samples[2]["metadata"]["turn_index"]] == [1, 2]
    assert [samples[1]["metadata"]["total_turns"], samples[2]["metadata"]["total_turns"]] == [2, 2]

    file_samples, n_raw, file_bundles, file_sample_to_bundle = iter_samples_from_file(
        path,
        return_row_bundles=True,
    )
    assert n_raw == 3
    assert file_sample_to_bundle == sample_to_bundle
    assert [sample["id"] for sample in file_samples] == [sample["id"] for sample in samples]
    assert [bundle.data_id for bundle in file_bundles] == [bundle.data_id for bundle in bundles]


@pytest.mark.parametrize(
    ("bad_content", "type_name"),
    [
        (["bad"], "list"),
        ({"bad": "content"}, "dict"),
    ],
)
def test_row_ingestion_raises_value_error_for_invalid_selected_text_without_fallback(tmp_path, bad_content, type_name):
    path = tmp_path / "invalid.jsonl"
    _write_jsonl(
        path,
        [
            {
                "id": "invalid",
                "messages": [
                    {"role": "user", "content": bad_content},
                    {"role": "assistant", "content": "a"},
                ],
            }
        ],
    )

    with pytest.raises(ValueError, match=type_name):
        list(iter_row_sample_bundles_from_jsonl(path))
    with pytest.raises(ValueError, match=type_name):
        iter_samples_from_file(path, return_row_bundles=True)


def test_zero_assistant_row_keeps_single_sample_contract_through_file_reader(tmp_path):
    path = tmp_path / "zero-assistant.jsonl"
    _write_jsonl(
        path,
        [
            {
                "id": "zero-assistant",
                "messages": [
                    {"role": "system", "content": "policy"},
                    {"role": "user", "content": "q"},
                    {"role": "tool", "content": "tool-output"},
                ],
            }
        ],
    )

    samples, n_raw, bundles, sample_to_bundle = iter_samples_from_file(
        path,
        return_row_bundles=True,
    )

    assert n_raw == 1
    assert len(bundles) == 1
    assert len(bundles[0].samples) == 1
    assert len(samples) == 1
    assert sample_to_bundle == [0]
    assert samples[0]["id"] == build_turn_id(bundles[0].data_id, 1)
    assert samples[0]["metadata"]["original_format"] == "openai_messages"
    assert samples[0]["conversations"] == [
        {"from": "system", "value": "policy"},
        {"from": "human", "value": "q"},
        {"from": "tool", "value": "tool-output"},
    ]


def test_iter_row_sample_bundles_from_jsonl_rejects_macos_sidecar(tmp_path):
    path = tmp_path / "._rows.jsonl"
    path.write_bytes(b"\xb0\x00\x01")

    with pytest.raises(ValueError, match="AppleDouble sidecar file"):
        list(iter_row_sample_bundles_from_jsonl(path))
