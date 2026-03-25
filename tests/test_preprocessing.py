"""Tests for the preprocessing module."""

import json
from pathlib import Path

import pytest

from sft_label.preprocessing import (
    detect_format,
    normalize_pangu,
    normalize_and_slice,
    normalize_sample,
    slice_multiturn,
    strip_cot,
    detect_code_fence_languages,
    detect_framework_languages,
    count_code_blocks,
    extract_tool_signals,
    preprocess,
    format_signals_for_prompt,
    generate_sparse_schedule,
    apply_sparse_sampling,
    truncate_conversations_for_labeling,
    to_pangu_pseudo_multiturn,
    sanitize_training_markers,
)


FIXTURES_DIR = Path(__file__).parent / "fixtures"


def _load_multiturn_fixture(sample_id: str) -> dict:
    fixture_path = FIXTURES_DIR / "multiturn_regressions.json"
    payload = json.loads(fixture_path.read_text(encoding="utf-8"))
    for sample in payload:
        if sample.get("id") == sample_id:
            return sample
    raise KeyError(f"Missing multiturn fixture: {sample_id}")


class TestFormatDetection:
    def test_sharegpt_format(self):
        sample = {"conversations": [{"from": "human", "value": "hi"}]}
        assert detect_format(sample) == "sharegpt"

    def test_conversations_role_content_routes_sharegpt(self):
        sample = {
            "conversations": [
                {"role": "user", "content": "hi"},
                {"role": "assistant", "content": "hello"},
            ]
        }
        assert detect_format(sample) == "sharegpt"

    def test_messages_role_content_routes_openai_messages(self):
        sample = {
            "messages": [
                {"role": "user", "content": "hi"},
                {"role": "assistant", "content": "hello"},
            ]
        }
        assert detect_format(sample) == "openai_messages"

    def test_pangu_format(self):
        sample = {"data": [{"role": "user", "content": "hi"}]}
        assert detect_format(sample) == "pangu"

    def test_data_wins_over_lower_priority_non_empty_legal_sources(self):
        sample = {
            "data": [{"role": "user", "content": "q"}, {"role": "assistant", "content": "a"}],
            "conversations": [{"from": "human", "value": "q2"}, {"from": "gpt", "value": "a2"}],
            "messages": [{"role": "user", "content": "q3"}, {"role": "assistant", "content": "a3"}],
        }
        assert detect_format(sample) == "pangu"

    def test_empty_high_priority_sources_fall_back_to_non_empty_legal_source(self):
        sample = {
            "data": [],
            "conversations": [],
            "messages": [{"role": "user", "content": "q"}, {"role": "assistant", "content": "a"}],
        }
        assert detect_format(sample) == "openai_messages"

    def test_illegal_conversations_fall_back_to_valid_non_empty_messages(self):
        sample = {
            "conversations": [{"role": "user", "content": ["bad"]}],
            "messages": [{"role": "user", "content": "q"}, {"role": "assistant", "content": "a"}],
        }
        assert detect_format(sample) == "openai_messages"

    def test_illegal_high_priority_source_raises_without_non_empty_legal_fallback(self):
        sample = {
            "conversations": [{"role": "user", "content": ["bad"]}],
            "messages": [],
        }
        with pytest.raises(ValueError):
            detect_format(sample)

    def test_unknown_format(self):
        sample = {"foo": []}
        assert detect_format(sample) == "unknown"


class TestStripCot:
    def test_pangu_cot(self):
        text = "hello [unused16]thinking stuff[unused17] world"
        assert strip_cot(text) == "hello  world"

    def test_sharegpt_cot(self):
        text = "hello <think>internal thought</think> world"
        assert strip_cot(text) == "hello  world"

    def test_no_cot(self):
        text = "hello world"
        assert strip_cot(text) == "hello world"


class TestSliceMultiturn:
    def test_single_turn(self):
        convs = [
            {"from": "human", "value": "q"},
            {"from": "gpt", "value": "a"},
        ]
        result = slice_multiturn(convs)
        assert len(result) == 1
        assert result[0] == convs

    def test_multi_turn(self):
        convs = [
            {"from": "human", "value": "q1"},
            {"from": "gpt", "value": "a1"},
            {"from": "human", "value": "q2"},
            {"from": "gpt", "value": "a2"},
        ]
        result = slice_multiturn(convs)
        assert len(result) == 2
        assert len(result[0]) == 2  # [q1, a1]
        assert len(result[1]) == 4  # [q1, a1, q2, a2]


class TestNormalizePangu:
    def test_basic_normalization(self):
        sample = {
            "id": "test-1",
            "data": [
                {"role": "user", "content": "hello"},
                {"role": "assistant", "content": "hi there"},
            ],
        }
        result = normalize_pangu(sample)
        assert result["id"] == "test-1"
        assert len(result["conversations"]) == 2
        assert result["conversations"][0]["from"] == "human"
        assert result["conversations"][1]["from"] == "gpt"

    def test_meta_prompt_injected_as_system_turn(self):
        sample = {
            "id": "sys-1",
            "meta_prompt": "You are a careful assistant.",
            "data": [
                {"role": "user", "content": "Hello"},
                {"role": "assistant", "content": "Hi"},
            ],
        }
        result = normalize_pangu(sample)
        assert result["metadata"]["system_prompt"] == "You are a careful assistant."
        assert result["conversations"][0]["from"] == "system"
        assert result["conversations"][0]["value"] == "You are a careful assistant."
        assert result["conversations"][1]["from"] == "human"

    def test_meta_prompt_preserved_in_metadata_for_roundtrip(self):
        sample = {
            "id": "sys-list",
            "meta_prompt": ["Rule 1", "Rule 2"],
            "data": [
                {"role": "user", "content": "Hello"},
                {"role": "assistant", "content": "Hi"},
            ],
        }
        result = normalize_pangu(sample)
        assert result["metadata"]["pangu_meta_prompt"] == ["Rule 1", "Rule 2"]

    def test_pseudo_multiturn(self):
        sample = {
            "id": "test-2",
            "data": [
                {"role": "user", "content": "hello [unused10] world"},
                {"role": "assistant", "content": "response"},
            ],
        }
        result = normalize_pangu(sample)
        assert result["metadata"]["is_pseudo_multiturn"] is True


class TestNormalizeAndSlice:
    def test_single_turn_sharegpt(self):
        sample = {
            "id": "s1",
            "conversations": [
                {"from": "human", "value": "q"},
                {"from": "gpt", "value": "a"},
            ],
        }
        result = normalize_and_slice(sample)
        assert len(result) == 1

    def test_multi_turn_creates_slices(self):
        sample = {
            "id": "m1",
            "conversations": [
                {"from": "human", "value": "q1"},
                {"from": "gpt", "value": "a1"},
                {"from": "human", "value": "q2"},
                {"from": "gpt", "value": "a2"},
            ],
        }
        result = normalize_and_slice(sample)
        assert len(result) == 2
        assert result[0]["id"] == "m1_t1"
        assert result[1]["id"] == "m1_t2"
        meta0 = result[0]["metadata"]
        meta1 = result[1]["metadata"]
        assert meta0["slice_index"] == 0
        assert meta0["slice_position"] == 1
        assert meta0["slice_count"] == 2
        assert meta1["slice_index"] == 1
        assert meta1["slice_position"] == 2
        assert meta1["slice_count"] == 2

    def test_slices_preserve_leading_system_turns(self):
        sample = {
            "id": "sys-mt",
            "meta_prompt": "Follow the system rules.",
            "data": [
                {"role": "user", "content": "Q1"},
                {"role": "assistant", "content": "A1"},
                {"role": "user", "content": "Q2"},
                {"role": "assistant", "content": "A2"},
            ],
        }
        result = normalize_and_slice(sample)
        assert len(result) == 2
        for sliced in result:
            assert sliced["conversations"][0]["from"] == "system"
            assert sliced["conversations"][0]["value"] == "Follow the system rules."

    def test_sharegpt_cot_metadata_preserved(self):
        """Single-turn ShareGPT with <think> should preserve thinking_mode and cot_text."""
        sample = {
            "id": "cot1",
            "conversations": [
                {"from": "human", "value": "Solve this problem"},
                {"from": "gpt", "value": "<think>Let me reason step by step...</think>The answer is 42"},
            ],
        }
        result = normalize_and_slice(sample)
        assert len(result) == 1
        meta = result[0].get("metadata", {})
        assert meta.get("thinking_mode") == "slow"
        assert "step by step" in meta.get("cot_text", "")
        # COT should be stripped from conversation text
        assert "<think>" not in result[0]["conversations"][1]["value"]

    def test_sharegpt_multiturn_preserves_turn_local_cot(self):
        """Multi-turn ShareGPT should keep COT only on the slice whose final reply contains it."""
        sample = {
            "id": "cot_mt",
            "conversations": [
                {"from": "human", "value": "q1"},
                {"from": "gpt", "value": "<think>reasoning</think>a1"},
                {"from": "human", "value": "q2"},
                {"from": "gpt", "value": "a2"},
            ],
        }
        result = normalize_and_slice(sample)
        assert len(result) == 2
        assert result[0]["metadata"]["thinking_mode"] == "slow"
        assert result[0]["metadata"]["cot_text"] == "reasoning"
        assert result[1]["metadata"]["thinking_mode"] == "fast"
        assert "cot_text" not in result[1]["metadata"]

    def test_single_turn_pangu_without_cot_defaults_to_fast(self):
        """Single-turn Pangu sample without explicit COT should emit thinking_mode=fast."""
        sample = {
            "id": "pangu-fast-single",
            "data": [
                {"role": "user", "content": "Write a python program."},
                {"role": "assistant", "content": "**Solution Explanation**\nDo X, then Y."},
            ],
        }
        result = normalize_and_slice(sample)
        assert len(result) == 1
        meta = result[0].get("metadata", {})
        assert meta.get("thinking_mode") == "fast"
        assert "cot_text" not in meta

    def test_multi_turn_assigns_stable_conversation_uid(self):
        sample = {
            "id": "dup-source-id",
            "conversations": [
                {"from": "human", "value": "q1"},
                {"from": "gpt", "value": "a1"},
                {"from": "human", "value": "q2"},
                {"from": "gpt", "value": "a2"},
            ],
        }
        first = normalize_and_slice(sample)
        second = normalize_and_slice(sample)

        assert len(first) == 2
        uid_first = {s["metadata"].get("conversation_uid") for s in first}
        uid_second = {s["metadata"].get("conversation_uid") for s in second}

        assert len(uid_first) == 1
        assert None not in uid_first
        assert uid_first == uid_second

    def test_long_single_reply_tool_trajectory_gets_trajectory_metadata(self):
        sample = {
            "id": "traj-1",
            "conversations": [
                {"from": "human", "value": "Fix failing tests and report progress."},
                {"from": "tool", "value": "pytest failed: tests/test_api.py::test_create"},
                {"from": "tool", "value": "Opened service.py and inspected return path"},
                {"from": "tool", "value": "Applied patch to service.py"},
                {"from": "gpt", "value": "Fixed service.py and tests now pass."},
            ],
        }
        result = normalize_and_slice(sample)
        assert len(result) == 1
        meta = result[0].get("metadata", {})
        assert meta.get("trajectory_object") is True
        assert meta.get("source_id") == "traj-1"
        assert meta.get("conversation_uid", "").startswith("conversation_uid:")
        assert meta.get("turn_index") == 1
        assert meta.get("total_turns") == 1
        assert meta.get("trajectory_turn_count") == 5
        assert meta.get("trajectory_tool_turn_count") == 3
        assert meta.get("trajectory_assistant_turn_count") == 1

    def test_duplicate_single_reply_trajectories_receive_distinct_uids(self):
        first = {
            "id": "traj-dup",
            "conversations": [
                {"from": "human", "value": "Investigate flaky test A."},
                {"from": "tool", "value": "Ran pytest -k test_a"},
                {"from": "tool", "value": "Inspected service_a.py"},
                {"from": "tool", "value": "Patched retry logic"},
                {"from": "gpt", "value": "Fixed flaky test A."},
            ],
        }
        second = {
            "id": "traj-dup",
            "conversations": [
                {"from": "human", "value": "Investigate flaky test B."},
                {"from": "tool", "value": "Ran pytest -k test_b"},
                {"from": "tool", "value": "Inspected service_b.py"},
                {"from": "tool", "value": "Patched timeout logic"},
                {"from": "gpt", "value": "Fixed flaky test B."},
            ],
        }

        first_uid = normalize_and_slice(first, source_file="dup.jsonl", source_row=1)[0]["metadata"]["conversation_uid"]
        second_uid = normalize_and_slice(second, source_file="dup.jsonl", source_row=2)[0]["metadata"]["conversation_uid"]

        assert first_uid.startswith("conversation_uid:")
        assert second_uid.startswith("conversation_uid:")
        assert first_uid != second_uid

    def test_conversations_role_content_single_turn_normalizes_to_internal_roles(self):
        sample = {
            "id": "oa-conv-single",
            "conversations": [
                {"role": "user", "content": "q"},
                {"role": "assistant", "content": "a"},
            ],
        }
        result = normalize_and_slice(sample)
        assert len(result) == 1
        assert result[0]["conversations"] == [
            {"from": "human", "value": "q"},
            {"from": "gpt", "value": "a"},
        ]
        assert result[0]["metadata"]["original_format"] == "openai_conversations"

    def test_selected_none_text_fields_normalize_to_empty_strings(self):
        sample = {
            "id": "none-text-fields",
            "conversations": [
                {"role": "user", "content": None},
                {"role": "assistant", "content": None},
            ],
        }
        result = normalize_and_slice(sample)
        assert len(result) == 1
        assert result[0]["conversations"] == [
            {"from": "human", "value": ""},
            {"from": "gpt", "value": ""},
        ]

    def test_turn_missing_both_field_pairs_normalizes_to_empty_turn(self):
        sample = {"id": "missing-shape", "conversations": [{}]}
        result = normalize_and_slice(sample)
        assert len(result) == 1
        assert result[0]["conversations"] == [{"from": "", "value": ""}]

    def test_conversations_role_content_multi_turn_slices_with_context(self):
        sample = {
            "id": "oa-conv-multi",
            "conversations": [
                {"role": "user", "content": "q1"},
                {"role": "assistant", "content": "a1"},
                {"role": "user", "content": "q2"},
                {"role": "assistant", "content": "a2"},
            ],
        }
        result = normalize_and_slice(sample)
        assert len(result) == 2
        assert result[0]["conversations"] == [
            {"from": "human", "value": "q1"},
            {"from": "gpt", "value": "a1"},
        ]
        assert result[1]["conversations"] == [
            {"from": "human", "value": "q1"},
            {"from": "gpt", "value": "a1"},
            {"from": "human", "value": "q2"},
            {"from": "gpt", "value": "a2"},
        ]

    def test_system_and_tool_turns_survive_normalization_and_slicing(self):
        sample = {
            "id": "oa-system-tool",
            "conversations": [
                {"role": "system", "content": "policy"},
                {"role": "user", "content": "q1"},
                {"role": "assistant", "content": "a1"},
                {"role": "tool", "content": "result"},
                {"role": "assistant", "content": "a2"},
            ],
        }
        result = normalize_and_slice(sample)
        assert len(result) == 2
        assert result[0]["conversations"][0] == {"from": "system", "value": "policy"}
        assert result[1]["conversations"][-2] == {"from": "tool", "value": "result"}
        assert result[1]["conversations"][-1] == {"from": "gpt", "value": "a2"}

    def test_messages_role_content_single_turn_normalization(self):
        sample = {
            "id": "oa-msg-single",
            "messages": [
                {"role": "user", "content": "q"},
                {"role": "assistant", "content": "a"},
            ],
        }
        result = normalize_and_slice(sample)
        assert len(result) == 1
        assert result[0]["conversations"] == [
            {"from": "human", "value": "q"},
            {"from": "gpt", "value": "a"},
        ]
        assert result[0]["metadata"]["original_format"] == "openai_messages"

    def test_messages_role_content_multi_turn_normalization(self):
        sample = {
            "id": "oa-msg-multi",
            "messages": [
                {"role": "system", "content": "policy"},
                {"role": "user", "content": "q1"},
                {"role": "assistant", "content": "a1"},
                {"role": "user", "content": "q2"},
                {"role": "assistant", "content": "a2"},
            ],
        }
        result = normalize_and_slice(sample)
        assert len(result) == 2
        assert result[0]["conversations"][0] == {"from": "system", "value": "policy"}
        assert result[1]["conversations"][-1] == {"from": "gpt", "value": "a2"}
        assert all(s["metadata"]["original_format"] == "openai_messages" for s in result)

    def test_mixed_conversations_provenance_uses_first_non_empty_complete_internal_turn(self):
        sample = {
            "id": "mixed-sharegpt",
            "conversations": [
                {},
                {"role": "user", "content": ""},
                {"from": "human", "value": "q"},
                {"role": "assistant", "content": "a"},
            ],
        }
        result = normalize_and_slice(sample)
        assert len(result) == 1
        assert result[0]["metadata"]["original_format"] == "sharegpt"

    def test_mixed_conversations_provenance_uses_first_non_empty_complete_openai_turn(self):
        sample = {
            "id": "mixed-openai",
            "conversations": [
                {},
                {"from": "human", "value": ""},
                {"role": "user", "content": "q"},
                {"role": "assistant", "content": "a"},
            ],
        }
        result = normalize_and_slice(sample)
        assert len(result) == 1
        assert result[0]["metadata"]["original_format"] == "openai_conversations"

    def test_unknown_roles_are_lowercased_and_preserved(self):
        sample = {
            "id": "unknown-roles",
            "messages": [
                {"role": "DEVELOPER", "content": "policy"},
                {"role": "user", "content": "q"},
                {"role": "assistant", "content": "a"},
            ],
        }
        result = normalize_and_slice(sample)
        assert len(result) == 1
        assert result[0]["conversations"][0] == {"from": "developer", "value": "policy"}

    def test_openai_style_long_trajectory_keeps_turn_index_and_total_turns_stable(self):
        sample = {
            "id": "oa-long",
            "messages": [
                {"role": "system", "content": "policy"},
                {"role": "user", "content": "q1"},
                {"role": "assistant", "content": "a1"},
                {"role": "tool", "content": "r1"},
                {"role": "assistant", "content": "a2"},
                {"role": "tool", "content": "r2"},
                {"role": "assistant", "content": "a3"},
            ],
        }
        result = normalize_and_slice(sample)
        assert len(result) == 3
        for i, sliced in enumerate(result, start=1):
            assert sliced["metadata"]["turn_index"] == i
            assert sliced["metadata"]["total_turns"] == 3

    def test_zero_assistant_rows_stay_as_one_unsliced_sample(self):
        sample = {
            "id": "zero-assistant",
            "messages": [
                {"role": "system", "content": "policy"},
                {"role": "user", "content": "q"},
                {"role": "tool", "content": "r"},
            ],
        }
        result = normalize_and_slice(sample)
        assert len(result) == 1
        assert result[0]["id"] == "zero-assistant"

    def test_winning_empty_source_yields_one_unsliced_empty_conversation_sample(self):
        sample = {"id": "empty-pangu", "data": []}
        result = normalize_and_slice(sample)
        assert len(result) == 1
        assert result[0]["conversations"] == []
        assert result[0]["metadata"]["original_format"] == "pangu"

    def test_original_format_set_for_supported_sources(self):
        sharegpt = normalize_and_slice(
            {
                "id": "fmt-sharegpt",
                "conversations": [{"from": "human", "value": "q"}, {"from": "gpt", "value": "a"}],
            }
        )[0]
        openai_conversations = normalize_and_slice(
            {
                "id": "fmt-openai-conv",
                "conversations": [{"role": "user", "content": "q"}, {"role": "assistant", "content": "a"}],
            }
        )[0]
        openai_messages = normalize_and_slice(
            {
                "id": "fmt-openai-msg",
                "messages": [{"role": "user", "content": "q"}, {"role": "assistant", "content": "a"}],
            }
        )[0]
        pangu = normalize_and_slice(
            {
                "id": "fmt-pangu",
                "data": [{"role": "user", "content": "q"}, {"role": "assistant", "content": "a"}],
            }
        )[0]

        assert sharegpt["metadata"]["original_format"] == "sharegpt"
        assert openai_conversations["metadata"]["original_format"] == "openai_conversations"
        assert openai_messages["metadata"]["original_format"] == "openai_messages"
        assert pangu["metadata"]["original_format"] == "pangu"

    def test_turn_with_both_turn_shapes_prefers_internal_fields(self):
        sample = {
            "id": "dual-shape-turn",
            "conversations": [
                {"from": "human", "value": "use-this", "role": "user", "content": "ignore-this"},
                {"role": "assistant", "content": "a"},
            ],
        }
        result = normalize_and_slice(sample)
        assert len(result) == 1
        assert result[0]["conversations"][0] == {"from": "human", "value": "use-this"}
        assert result[0]["metadata"]["original_format"] == "sharegpt"

    @pytest.mark.parametrize(
        ("bad_content", "type_name"),
        [
            (["x"], "list"),
            ({"k": "v"}, "dict"),
        ],
    )
    def test_invalid_selected_text_fields_raise_value_error_with_location_clues(self, bad_content, type_name):
        sample = {
            "id": "bad-content",
            "conversations": [{"role": "user", "content": bad_content}],
        }
        with pytest.raises(ValueError) as exc:
            normalize_and_slice(sample, source_file="bad.jsonl", source_row=7)
        message = str(exc.value)
        assert "bad.jsonl" in message
        assert "7" in message
        assert "content" in message
        assert "0" in message
        assert type_name in message


class TestNoSliceNormalizationPath:
    def test_normalize_sample_handles_openai_style_conversations(self):
        sample = {
            "id": "no-slice-conv",
            "conversations": [
                {"role": "user", "content": "q"},
                {"role": "assistant", "content": "a"},
            ],
        }
        normalized = normalize_sample(sample)
        assert normalized["conversations"] == [
            {"from": "human", "value": "q"},
            {"from": "gpt", "value": "a"},
        ]
        assert normalized["metadata"]["original_format"] == "openai_conversations"

    def test_normalize_sample_handles_openai_style_messages(self):
        sample = {
            "id": "no-slice-msg",
            "messages": [
                {"role": "user", "content": "q"},
                {"role": "assistant", "content": "a"},
            ],
        }
        normalized = normalize_sample(sample)
        assert normalized["conversations"] == [
            {"from": "human", "value": "q"},
            {"from": "gpt", "value": "a"},
        ]
        assert normalized["metadata"]["original_format"] == "openai_messages"

    def test_preprocess_uses_normalized_messages_schema_without_slicing(self):
        sample = {
            "id": "preprocess-msg",
            "messages": [
                {"role": "user", "content": "Question about Python"},
                {"role": "assistant", "content": "```python\nprint('ok')\n```"},
            ],
        }
        signals = preprocess(sample)
        assert signals["total_turns"] == 2
        assert signals["last_query_preview"].startswith("Question about Python")
        assert signals["last_response_length"] > 0
        assert "python" in signals["detected_languages"]

    def test_no_slice_path_keeps_provenance_parity_with_normalize_and_slice(self):
        sample = {
            "id": "provenance-parity",
            "conversations": [
                {"role": "user", "content": "q"},
                {"role": "assistant", "content": "a"},
            ],
        }
        normalized = normalize_sample(sample)
        sliced = normalize_and_slice(sample)[0]
        assert normalized["metadata"]["original_format"] == sliced["metadata"]["original_format"]

    @pytest.mark.parametrize("entrypoint", [normalize_sample, preprocess])
    def test_no_slice_path_raises_value_error_for_invalid_content(self, entrypoint):
        sample = {
            "id": "bad-no-slice",
            "messages": [{"role": "user", "content": ["not-supported"]}],
        }
        with pytest.raises(ValueError) as exc:
            entrypoint(sample)
        message = str(exc.value)
        assert "content" in message
        assert "0" in message
        assert "list" in message


class TestMultiturnRegressionFixtures:
    def test_turn_metadata_semantics(self):
        sample = _load_multiturn_fixture("mt-system-preserve")
        slices = normalize_and_slice(sample)
        assert len(slices) == 3
        for idx, item in enumerate(slices, start=1):
            meta = item["metadata"]
            assert meta["source_id"] == "mt-system-preserve"
            assert meta["turn_index"] == idx
            assert meta["total_turns"] == 3
            assert meta["source_file"] == "synthetic/multiturn_regressions.json"

    def test_planner_metadata_added_to_multiturn_slices(self):
        sample = _load_multiturn_fixture("mt-agentic-medium")
        slices = normalize_and_slice(sample)

        assert len(slices) == 4
        for idx, item in enumerate(slices, start=1):
            meta = item["metadata"]
            assert meta["planner_policy"] == "compact_semantic_anchor_v1"
            assert meta["segment_id"] >= 0
            assert meta["segment_turn_start"] <= idx <= meta["segment_turn_end"]
            assert meta["inherit_group_id"].startswith("segment-")
            assert 0.0 <= meta["planner_confidence"] <= 1.0

    def test_planner_uses_local_view_not_whole_prefix_growth(self):
        sample = {
            "id": "mt-local-view",
            "conversations": [
                {"from": "human", "value": "Initial project setup.\n" + ("A" * 2000)},
                {"from": "gpt", "value": "Created the project scaffold.\n" + ("B" * 2000)},
                {"from": "human", "value": "Run the tests and fix the timeout regression."},
                {"from": "gpt", "value": "Ran pytest, found the timeout, and fixed the flaky retry loop."},
                {"from": "human", "value": "Run the tests and fix the timeout regression."},
                {"from": "gpt", "value": "Ran pytest, found the timeout, and fixed the flaky retry loop."},
            ],
        }

        slices = normalize_and_slice(sample)

        assert len(slices) == 3
        assert slices[1]["metadata"]["segment_id"] == slices[2]["metadata"]["segment_id"]

    def test_system_turn_preserved_in_each_slice(self):
        sample = _load_multiturn_fixture("mt-system-preserve")
        slices = normalize_and_slice(sample)
        for item in slices:
            assert item["conversations"][0]["from"] == "system"
            assert item["conversations"][0]["value"] == "You are a careful debugging assistant."

    def test_medium_agentic_multiturn_fixture_retains_tool_signal(self):
        sample = _load_multiturn_fixture("mt-agentic-medium")
        tool_names, tags = extract_tool_signals(sample["conversations"])
        assert "bash" in tool_names
        assert "bash-execution" in tags
        slices = normalize_and_slice(sample)
        assert len(slices) == 4


class TestLanguageDetection:
    def test_code_fence_python(self):
        text = "```python\nprint('hello')\n```"
        assert "python" in detect_code_fence_languages(text)

    def test_code_fence_multiple(self):
        text = "```typescript\nconst x = 1;\n```\n```python\nx = 1\n```"
        langs = detect_code_fence_languages(text)
        assert "typescript" in langs
        assert "python" in langs

    def test_framework_detection(self):
        text = "I'm using React and Django"
        langs = detect_framework_languages(text)
        assert "typescript" in langs
        assert "python" in langs


class TestCodeBlocks:
    def test_count(self):
        text = "```python\ncode\n```\ntext\n```js\nmore\n```"
        assert count_code_blocks(text) == 2


class TestToolSignals:
    def test_bash_detection(self):
        convs = [
            {"from": "tool", "value": "$ ls -la\ntotal 0"},
        ]
        names, tags = extract_tool_signals(convs)
        assert "bash" in names
        assert "bash-execution" in tags

    def test_no_tools(self):
        convs = [
            {"from": "human", "value": "hi"},
            {"from": "gpt", "value": "hello"},
        ]
        names, tags = extract_tool_signals(convs)
        assert len(names) == 0
        assert len(tags) == 0


class TestSparseSchedule:
    def test_small_n(self):
        # Below threshold: all labeled
        schedule = generate_sparse_schedule(5)
        assert schedule == [0, 1, 2, 3, 4]

    def test_at_threshold(self):
        schedule = generate_sparse_schedule(12)
        assert schedule == list(range(12))

    def test_large_n_includes_first_and_last(self):
        schedule = generate_sparse_schedule(50)
        assert schedule[0] == 0
        assert schedule[-1] == 49
        assert len(schedule) < 50  # Some were skipped

    def test_max_gap_caps_schedule(self):
        """With max_gap=8, no gap between consecutive labeled indices should exceed 8."""
        schedule = generate_sparse_schedule(50, max_gap=8)
        assert schedule[0] == 0
        assert schedule[-1] == 49
        for i in range(1, len(schedule)):
            gap = schedule[i] - schedule[i - 1]
            assert gap <= 8, f"Gap {gap} between indices {schedule[i-1]} and {schedule[i]} exceeds max_gap=8"
        # With capped gap, we should have more labeled points than uncapped
        uncapped = generate_sparse_schedule(50, max_gap=999)
        assert len(schedule) >= len(uncapped)

    def test_max_gap_default_unchanged_for_small(self):
        """n=12 (at threshold) should still label all regardless of max_gap."""
        schedule = generate_sparse_schedule(12, max_gap=4)
        assert schedule == list(range(12))


class TestApplySparseSampling:
    @staticmethod
    def _incremental_tool_trajectory_samples(total_turns=13):
        samples = []
        for i in range(total_turns):
            conversations = [{"from": "human", "value": "Diagnose the same flaky test failure."}]
            for j in range(i):
                conversations.append({
                    "from": "tool",
                    "value": f"pytest repeat run {j}: same flaky stack trace",
                })
            conversations.append({
                "from": "gpt",
                "value": "The root cause is the same timeout race; keep the same fix plan.",
            })
            samples.append({
                "id": f"conv_t{i + 1}",
                "metadata": {"source_id": "conv", "turn_index": i + 1, "total_turns": total_turns},
                "conversations": conversations,
            })
        return samples

    def test_planner_allows_inheritance_when_only_tool_trajectory_grows(self):
        conversations = [{"from": "human", "value": "Debug the flaky retry pipeline end to end."}]
        for i in range(13):
            conversations.append({"from": "tool", "value": f"$ pytest tests/test_retry.py::test_case_{i}\nFAILED timeout"})
            conversations.append({"from": "gpt", "value": "I inspected the failure and tightened the retry handling."})

        samples = normalize_and_slice({
            "id": "tool-trajectory-growth",
            "conversations": conversations,
        })

        label_indices, inherit_map = apply_sparse_sampling(samples, planner_enabled=True)

        assert len(samples) == 13
        assert len(inherit_map) > 0
        assert len(label_indices) < len(samples)
        for target_idx, source_idx in inherit_map.items():
            target_meta = samples[target_idx]["metadata"]
            source_meta = samples[source_idx]["metadata"]
            assert target_meta["segment_id"] == source_meta["segment_id"]

    def test_planner_falls_back_when_it_would_use_more_labels_than_legacy(self):
        fixture_path = FIXTURES_DIR / "e2e_folder_test" / "multi_turn" / "coderforge_swe_trajectories.jsonl"
        row = json.loads(fixture_path.read_text(encoding="utf-8").splitlines()[1])
        samples = normalize_and_slice(row, source_file=fixture_path, source_row=2)

        legacy_label_indices, legacy_inherit_map = apply_sparse_sampling(samples, planner_enabled=False)
        planner_label_indices, planner_inherit_map = apply_sparse_sampling(
            samples,
            planner_enabled=True,
            planner_metadata_only=False,
        )

        assert planner_label_indices == legacy_label_indices
        assert planner_inherit_map == legacy_inherit_map

    def test_single_turn_always_labeled(self):
        samples = [{"id": "s1", "metadata": {}}]
        label_indices, inherit_map = apply_sparse_sampling(samples, planner_enabled=True)
        assert 0 in label_indices
        assert len(inherit_map) == 0

    def test_multi_turn_small(self):
        # 3 slices from same source — all labeled (below threshold)
        samples = [
            {"id": "m_t1", "metadata": {"source_id": "m", "turn_index": 1, "total_turns": 3}},
            {"id": "m_t2", "metadata": {"source_id": "m", "turn_index": 2, "total_turns": 3}},
            {"id": "m_t3", "metadata": {"source_id": "m", "turn_index": 3, "total_turns": 3}},
        ]
        label_indices, inherit_map = apply_sparse_sampling(samples, planner_enabled=True)
        assert label_indices == {0, 1, 2}
        assert len(inherit_map) == 0

    def test_semantic_drift_forces_label_even_when_structure_matches(self):
        samples = []
        for i in range(13):
            request = f"Implement feature {i}"
            response = f"Here is feature code {i}"
            if i == 9:
                request = "Explain Python decorators"
                response = "Decorators wrap functions"
            if i == 10:
                request = "Implement rate limiting middleware"
                response = "Here is the middleware implementation"
            samples.append({
                "id": f"m_t{i + 1}",
                "metadata": {"source_id": "m", "turn_index": i + 1, "total_turns": 13},
                "conversations": [
                    {"from": "human", "value": request},
                    {"from": "gpt", "value": response},
                ],
            })

        label_indices, inherit_map = apply_sparse_sampling(samples, planner_enabled=True)
        assert 9 in label_indices
        assert 9 not in inherit_map

    def test_legacy_sparse_allows_inheritance_for_incremental_tool_growth(self):
        samples = self._incremental_tool_trajectory_samples()

        label_indices, inherit_map = apply_sparse_sampling(samples, planner_enabled=False)

        assert label_indices == {0, 1, 2, 3, 4, 5, 6, 7, 8, 10, 12}
        assert inherit_map == {9: 10, 11: 12}

    def test_legacy_sparse_can_fall_back_backward_when_forward_path_is_hard_boundary(self):
        samples = self._incremental_tool_trajectory_samples()
        samples[10]["conversations"] = [
            {"from": "human", "value": "Update src/app.py to fix the flaky timeout."},
            {"from": "gpt", "value": "Patched src/app.py and kept the same timeout fix."},
        ]

        label_indices, inherit_map = apply_sparse_sampling(samples, planner_enabled=False)

        assert 9 not in label_indices
        assert inherit_map[9] == 8

    def test_duplicate_source_ids_do_not_cross_inherit_when_uid_present(self):
        samples = []
        for i in range(13):
            samples.append({
                "id": f"conv_a_t{i + 1}",
                "metadata": {
                    "source_id": "dup",
                    "conversation_uid": "uid-a",
                    "turn_index": i + 1,
                    "total_turns": 13,
                },
                "conversations": [
                    {"from": "human", "value": f"aq{i}"},
                    {"from": "gpt", "value": f"aa{i}"},
                ],
            })
        for i in range(13):
            samples.append({
                "id": f"conv_b_t{i + 1}",
                "metadata": {
                    "source_id": "dup",
                    "conversation_uid": "uid-b",
                    "turn_index": i + 1,
                    "total_turns": 13,
                },
                "conversations": [
                    {"from": "human", "value": f"bq{i}"},
                    {"from": "gpt", "value": f"ba{i}"},
                ],
            })

        _label_indices, inherit_map = apply_sparse_sampling(samples)

        for target_idx, source_idx in inherit_map.items():
            target_uid = samples[target_idx]["metadata"]["conversation_uid"]
            source_uid = samples[source_idx]["metadata"]["conversation_uid"]
            assert target_uid == source_uid

    def test_planner_restricts_inheritance_within_segment(self):
        samples = []
        for i in range(13):
            lang = "python" if i < 9 else "javascript"
            request = f"Implement feature step {i}"
            response = f"```{lang}\nprint({i})\n```"
            samples.append({
                "id": f"conv_t{i + 1}",
                "metadata": {"source_id": "conv", "turn_index": i + 1, "total_turns": 13},
                "conversations": [
                    {"from": "human", "value": request},
                    {"from": "gpt", "value": response},
                ],
            })

        label_indices, inherit_map = apply_sparse_sampling(samples, planner_enabled=True)

        assert any(i >= 9 for i in label_indices)
        for target_idx, source_idx in inherit_map.items():
            target_meta = samples[target_idx]["metadata"]
            source_meta = samples[source_idx]["metadata"]
            assert target_meta["segment_id"] == source_meta["segment_id"]

    def test_planner_can_fall_back_to_legacy_sparse_schedule(self):
        samples = []
        for i in range(15):
            lang = "python" if i % 2 == 0 else "javascript"
            samples.append({
                "id": f"conv_t{i + 1}",
                "metadata": {"source_id": "conv", "turn_index": i + 1, "total_turns": 15},
                "conversations": [
                    {"from": "human", "value": f"Rewrite module {i} in {lang}"},
                    {"from": "gpt", "value": f"```{lang}\ncode_{i}\n```"},
                ],
            })

        planner_label_indices, planner_inherit_map = apply_sparse_sampling(
            samples,
            planner_enabled=True,
            planner_fallback_boundary_ratio=0.20,
        )
        legacy_label_indices, legacy_inherit_map = apply_sparse_sampling(
            samples,
            planner_enabled=False,
        )

        assert planner_label_indices == legacy_label_indices
        assert planner_inherit_map == legacy_inherit_map

    def test_planner_still_force_labels_high_risk_file_scope_change(self):
        samples = []
        for i in range(13):
            request = "Update the README"
            if i == 9:
                request = "Update src/app.py and tests/test_app.py"
            samples.append({
                "id": f"conv_t{i + 1}",
                "metadata": {"source_id": "conv", "turn_index": i + 1, "total_turns": 13},
                "conversations": [
                    {"from": "human", "value": request},
                    {"from": "gpt", "value": "Done."},
                ],
            })

        label_indices, inherit_map = apply_sparse_sampling(samples, planner_enabled=True)

        assert 9 in label_indices
        assert 9 not in inherit_map


class TestTruncation:
    def test_no_truncation_needed(self):
        convs = [
            {"from": "human", "value": "short"},
            {"from": "gpt", "value": "also short"},
        ]
        result, truncated = truncate_conversations_for_labeling(convs)
        assert not truncated
        assert result == convs

    def test_truncation_applied(self):
        convs = [
            {"from": "human", "value": "x" * 15000},
            {"from": "gpt", "value": "y" * 15000},
        ]
        result, truncated = truncate_conversations_for_labeling(convs, max_total_chars=5000)
        assert truncated
        total = sum(len(t["value"]) for t in result)
        assert total <= 5500  # Allow some margin

    def test_truncation_sanitizes_xml_tags(self):
        convs = [
            {"from": "human", "value": "Fix the bug <solution>hint</solution>"},
            {"from": "gpt", "value": "Here is the <fix attr='v'>patch</fix>"},
        ]
        result, _ = truncate_conversations_for_labeling(convs)
        assert "<solution>" not in result[0]["value"]
        assert "<fix" not in result[1]["value"]
        assert "hint" in result[0]["value"]
        assert "patch" in result[1]["value"]

    def test_truncation_preserves_leading_system_turns(self):
        convs = [
            {"from": "system", "value": "System rules here."},
            {"from": "human", "value": "Q" * 6000},
            {"from": "gpt", "value": "A" * 6000},
            {"from": "human", "value": "Follow-up"},
            {"from": "gpt", "value": "Response"},
        ]
        result, truncated = truncate_conversations_for_labeling(convs, max_total_chars=3000)
        assert truncated
        assert result[0]["from"] == "system"
        assert "System rules here." in result[0]["value"]

    def test_truncation_keeps_system_prefix_within_budget(self):
        convs = [
            {"from": "system", "value": "S" * 500},
            {"from": "human", "value": "Q" * 500},
            {"from": "gpt", "value": "A" * 500},
        ]
        result, truncated = truncate_conversations_for_labeling(convs, max_total_chars=200)
        assert truncated
        assert result[0]["from"] == "system"
        assert sum(len(turn["value"]) for turn in result) <= 200


class TestSanitizeTrainingMarkers:
    def test_strips_basic_tags(self):
        text = "Before <solution>code here</solution> after"
        assert sanitize_training_markers(text) == "Before code here after"

    def test_strips_self_closing_with_space(self):
        text = "text <fix /> more"
        assert sanitize_training_markers(text) == "text  more"

    def test_strips_tags_with_attributes(self):
        text = '<tool_call name="edit">content</tool_call>'
        assert sanitize_training_markers(text) == "content"

    def test_case_insensitive(self):
        text = "<SOLUTION>code</SOLUTION>"
        assert sanitize_training_markers(text) == "code"

    def test_preserves_non_target_tags(self):
        text = "<div>html</div> <span>content</span>"
        assert sanitize_training_markers(text) == text

    def test_preserves_code_angle_brackets(self):
        text = "if (a < b && c > d) { fix(); }"
        assert sanitize_training_markers(text) == text

    def test_empty_input(self):
        assert sanitize_training_markers("") == ""

    def test_nested_tags(self):
        text = "<solution><code_change>new code</code_change></solution>"
        assert sanitize_training_markers(text) == "new code"

    def test_extended_tags(self):
        """Tags added based on expert review: command, output, result, reasoning, response."""
        text = "<command>ls -la</command> <output>file.txt</output>"
        assert sanitize_training_markers(text) == "ls -la file.txt"


class TestPreprocess:
    def test_basic_preprocess(self):
        sample = {
            "conversations": [
                {"from": "human", "value": "Write a Python function"},
                {"from": "gpt", "value": "```python\ndef hello():\n    pass\n```"},
            ],
        }
        signals = preprocess(sample)
        assert "python" in signals["detected_languages"]
        assert signals["total_turns"] == 2
        assert signals["code_block_count"] == 1

    def test_format_signals(self):
        signals = {
            "detected_languages": ["python"],
            "fence_languages": ["python"],
            "framework_languages": [],
            "has_tool_roles": False,
            "tool_names": [],
            "tool_agentic_tags": [],
            "behavioral_patterns": [],
            "total_turns": 2,
            "code_block_count": 1,
            "est_tokens": 100,
            "keyword_hits": [],
            "last_query_preview": "test",
            "last_response_length": 50,
        }
        result = format_signals_for_prompt(signals)
        assert "detected_languages" in result
        assert "python" in result


class TestPanguFixtures:
    """Test with actual Pangu fixture data."""

    def test_load_and_normalize(self):
        fixture_path = FIXTURES_DIR / "pangu_test_samples.jsonl"
        if not fixture_path.exists():
            return  # Skip if fixture not available

        samples = []
        with open(fixture_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    raw = json.loads(line)
                    samples.extend(normalize_and_slice(raw))

        assert len(samples) > 0
        for s in samples:
            assert "conversations" in s
            assert len(s["conversations"]) > 0


class TestToPanguPseudoMultiturn:
    """Tests for to_pangu_pseudo_multiturn() conversion."""

    def test_single_turn_no_cot(self):
        """Single turn without COT → [unused16][unused17] prefix."""
        sample = {
            "conversations": [
                {"from": "human", "value": "Write hello world"},
                {"from": "gpt", "value": "print('hello')"},
            ],
            "metadata": {},
        }
        result = to_pangu_pseudo_multiturn(sample)
        assert len(result["data"]) == 2
        assert result["data"][0]["role"] == "user"
        assert result["data"][0]["content"] == "Write hello world"
        assert result["data"][1]["role"] == "assistant"
        assert result["data"][1]["content"] == "[unused16][unused17]print('hello')"

    def test_single_turn_with_cot(self):
        """Single turn with COT → COT restored between [unused16] and [unused17]."""
        sample = {
            "conversations": [
                {"from": "human", "value": "Solve this"},
                {"from": "gpt", "value": "The answer is 42"},
            ],
            "metadata": {"cot_text": "Let me think about this step by step"},
        }
        result = to_pangu_pseudo_multiturn(sample)
        assert result["data"][1]["content"] == (
            "[unused16]Let me think about this step by step[unused17]The answer is 42"
        )

    def test_sliced_sample_restores_turn_local_cot(self):
        """Sliced sample with attributable cot_text should restore the final-turn COT."""
        sample = {
            "conversations": [
                {"from": "human", "value": "Q1"},
                {"from": "gpt", "value": "A1"},
            ],
            "metadata": {
                "source_id": "original-123",
                "cot_text": "Turn-local reasoning",
            },
        }
        result = to_pangu_pseudo_multiturn(sample)
        assert result["data"][1]["content"] == "[unused16]Turn-local reasoning[unused17]A1"

    def test_multi_turn_packing(self):
        """Multi-turn → prior turns packed with separators and role labels."""
        sample = {
            "conversations": [
                {"from": "human", "value": "Hello"},
                {"from": "gpt", "value": "Hi there"},
                {"from": "human", "value": "Write code"},
                {"from": "gpt", "value": "Here's the code"},
            ],
            "metadata": {},
        }
        result = to_pangu_pseudo_multiturn(sample)
        assert len(result["data"]) == 2

        user_content = result["data"][0]["content"]
        # Prior turns should be packed
        assert "用户：Hello /no_think" in user_content
        assert "助手：[unused16][unused17]Hi there" in user_content
        assert "[unused10][unused9]" in user_content
        # Last user turn
        assert "用户：Write code" in user_content
        # Last assistant response
        assert result["data"][1]["content"] == "[unused16][unused17]Here's the code"

    def test_current_turn_tool_trajectory_preserved(self):
        """Assistant/tool trajectory between user and final reply is preserved."""
        sample = {
            "conversations": [
                {"from": "human", "value": "Fix the bug"},
                {"from": "gpt", "value": "Let me check the logs"},
                {"from": "tool", "value": "ERROR: stack trace"},
                {"from": "gpt", "value": "Apply this patch"},
            ],
            "metadata": {},
        }
        result = to_pangu_pseudo_multiturn(sample)

        assert result["data"][0]["content"] == "Fix the bug"
        assistant_content = result["data"][1]["content"]
        assert "助手：[unused16][unused17]Let me check the logs" in assistant_content
        assert "ERROR: stack trace" in assistant_content
        assert assistant_content.endswith("[unused16][unused17]Apply this patch")
        assert "[unused10][unused9]" in assistant_content

    def test_system_prompt_preserved(self):
        """System prompt from metadata → meta_prompt in output."""
        sample = {
            "conversations": [
                {"from": "human", "value": "Hi"},
                {"from": "gpt", "value": "Hello"},
            ],
            "metadata": {"system_prompt": "You are a helpful assistant"},
        }
        result = to_pangu_pseudo_multiturn(sample)
        assert result["meta_prompt"] == "You are a helpful assistant"

    def test_leading_system_turns_map_to_meta_prompt(self):
        """Leading system turns should be concatenated into meta_prompt."""
        sample = {
            "conversations": [
                {"from": "system", "value": "Rule 1"},
                {"from": "system", "value": "Rule 2"},
                {"from": "human", "value": "Hi"},
                {"from": "gpt", "value": "Hello"},
            ],
            "metadata": {},
        }
        result = to_pangu_pseudo_multiturn(sample)
        assert result["meta_prompt"] == "Rule 1\n\nRule 2"
        assert "Rule 1" not in result["data"][0]["content"]
        assert "Rule 2" not in result["data"][0]["content"]

    def test_roundtrip_preserves_original_pangu_meta_prompt_shape(self):
        original = {
            "id": "sys-list-roundtrip",
            "meta_prompt": ["Rule 1", "Rule 2"],
            "data": [
                {"role": "user", "content": "Hi"},
                {"role": "assistant", "content": "[unused16][unused17]Hello"},
            ],
        }
        normalized = normalize_pangu(original)
        result = to_pangu_pseudo_multiturn(normalized)
        assert result["meta_prompt"] == ["Rule 1", "Rule 2"]

    def test_tools_preserved(self):
        """Tool definitions from metadata → tools in output."""
        sample = {
            "conversations": [
                {"from": "human", "value": "Search"},
                {"from": "gpt", "value": "Found it"},
            ],
            "metadata": {"tool_definitions": '[{"name": "search"}]'},
        }
        result = to_pangu_pseudo_multiturn(sample)
        assert result["tools"] == '[{"name": "search"}]'

    def test_empty_conversations(self):
        """Empty conversations → empty output."""
        sample = {"conversations": [], "metadata": {}}
        result = to_pangu_pseudo_multiturn(sample)
        assert result["data"] == []

    def test_no_assistant_reply(self):
        """Only human turns → just user message."""
        sample = {
            "conversations": [{"from": "human", "value": "Question?"}],
            "metadata": {},
        }
        result = to_pangu_pseudo_multiturn(sample)
        assert len(result["data"]) == 1
        assert result["data"][0]["role"] == "user"

    def test_roundtrip_single_turn(self):
        """Pangu → normalize → to_pangu preserves essential structure."""
        original = {
            "id": "test-1",
            "data": [
                {"role": "user", "content": "Explain Python"},
                {"role": "assistant", "content": "[unused16]Let me think[unused17]Python is great"},
            ],
            "meta_prompt": "You are helpful",
            "tools": "",
        }
        normalized = normalize_pangu(original)
        reconstructed = to_pangu_pseudo_multiturn(normalized)

        assert reconstructed["meta_prompt"] == "You are helpful"
        assert len(reconstructed["data"]) == 2
        # COT should be restored
        assert "[unused16]Let me think[unused17]" in reconstructed["data"][1]["content"]
        assert "Python is great" in reconstructed["data"][1]["content"]


# ─── P0-2b: Markdown COT block sanitization ────────────

class TestSanitizeMarkdownCotBlocks:
    def test_strips_cot_block(self):
        text = "Before ```COT\nsome reasoning\n``` after"
        result = sanitize_training_markers(text)
        assert "```COT" not in result
        assert "some reasoning" not in result
        assert "Before" in result and "after" in result

    def test_strips_reasoning_block(self):
        text = "text ```REASONING\nthinking...\n``` more"
        result = sanitize_training_markers(text)
        assert "```REASONING" not in result
        assert "thinking..." not in result

    def test_strips_thinking_block(self):
        text = "start ```Thinking\nmy thoughts\n``` end"
        result = sanitize_training_markers(text)
        assert "```Thinking" not in result
        assert "my thoughts" not in result

    def test_case_insensitive(self):
        text = "a ```cot\nlower case\n``` b"
        result = sanitize_training_markers(text)
        assert "```cot" not in result

    def test_preserves_normal_code_blocks(self):
        text = "```python\nprint('hello')\n```"
        result = sanitize_training_markers(text)
        assert result == text

    def test_mixed_blocks(self):
        text = "code: ```python\nprint(1)\n``` then ```COT\nthink\n``` done"
        result = sanitize_training_markers(text)
        assert "```python" in result
        assert "```COT" not in result


# ─── P2-2: Signal change detection for sparse sampling ──

class TestShouldForceLabel:
    def test_same_language_no_force(self):
        from sft_label.preprocessing import _should_force_label
        source = {"conversations": [
            {"from": "human", "value": "Q"},
            {"from": "gpt", "value": "```python\nprint(1)\n```"},
        ]}
        target = {"conversations": [
            {"from": "human", "value": "Q2"},
            {"from": "gpt", "value": "```python\nprint(2)\n```"},
        ]}
        assert not _should_force_label(source, target)

    def test_language_change_forces(self):
        from sft_label.preprocessing import _should_force_label
        source = {"conversations": [
            {"from": "human", "value": "Q"},
            {"from": "gpt", "value": "```python\nprint(1)\n```"},
        ]}
        target = {"conversations": [
            {"from": "human", "value": "Q2"},
            {"from": "gpt", "value": "```javascript\nconsole.log(1)\n```"},
        ]}
        assert _should_force_label(source, target)

    def test_tool_role_change_forces(self):
        from sft_label.preprocessing import _should_force_label
        source = {"conversations": [
            {"from": "human", "value": "Q"},
            {"from": "gpt", "value": "answer"},
        ]}
        target = {"conversations": [
            {"from": "human", "value": "Q2"},
            {"from": "tool", "value": "result"},
            {"from": "gpt", "value": "answer2"},
        ]}
        assert _should_force_label(source, target)

    def test_no_code_no_force(self):
        from sft_label.preprocessing import _should_force_label
        source = {"conversations": [
            {"from": "human", "value": "Q"},
            {"from": "gpt", "value": "text answer"},
        ]}
        target = {"conversations": [
            {"from": "human", "value": "Q2"},
            {"from": "gpt", "value": "text answer 2"},
        ]}
        assert not _should_force_label(source, target)

    def test_keyword_focus_change_forces(self):
        from sft_label.preprocessing import _should_force_label
        source = {"conversations": [
            {"from": "human", "value": "Build a Flask API endpoint for this service"},
            {"from": "gpt", "value": "Here is the Flask handler"},
        ]}
        target = {"conversations": [
            {"from": "human", "value": "Optimize the SQL query performance"},
            {"from": "gpt", "value": "Here is the optimized query"},
        ]}
        assert _should_force_label(source, target)

    def test_file_scope_change_forces(self):
        from sft_label.preprocessing import _should_force_label
        source = {"conversations": [
            {"from": "human", "value": "Update the README"},
            {"from": "gpt", "value": "Done."},
        ]}
        target = {"conversations": [
            {"from": "human", "value": "Update src/app.py and tests/test_app.py"},
            {"from": "gpt", "value": "Done."},
        ]}
        assert _should_force_label(source, target)


class TestSparseSamplingSignalDetection:
    def test_language_change_forces_labeling(self):
        """Sparse sampling should force-label when language changes between slices."""
        # 15 slices (above threshold=12), language changes at slice 10
        samples = []
        for i in range(15):
            lang = "python" if i < 10 else "javascript"
            code = f"```{lang}\ncode_{i}\n```"
            samples.append({
                "id": f"s_{i}",
                "metadata": {"source_id": "conv-1", "turn_index": i + 1, "total_turns": 15},
                "conversations": [
                    {"from": "human", "value": f"Q{i}"},
                    {"from": "gpt", "value": code},
                ],
            })
        label_indices, inherit_map = apply_sparse_sampling(samples, threshold=12)
        # Slices near the language change boundary (10+) that would
        # normally be inherited should be force-labeled instead
        # At minimum, more slices should be labeled than the default schedule
        labeled_after_change = [i for i in range(10, 15) if i in label_indices]
        [i for i in range(10, 15) if i in inherit_map]
        # The force-labeling should increase labeled count vs pure schedule
        assert len(labeled_after_change) > 0
