"""Tests for conversation-level value aggregation."""

from __future__ import annotations

import json
import pytest

from sft_label.conversation import (
    build_conversation_key,
    group_by_conversation,
    _position_weight,
    _effective_weights,
    _compute_penalty,
    _merge_labels,
    aggregate_conversation,
    compute_conv_selection_scores,
    aggregate_conversations,
    write_conversation_scores,
)


# ── Helpers ──

def _slice(source_id, turn_index, total_turns, value_score=6.0,
           complexity=5, quality=6, reasoning=5, rarity=4.0,
           inherited=False, flags=None, labels=None, thinking_mode=None,
           score_confidence=None,
           source_file=None,
           conversation_uid=None,
           conversations=None):
    """Build a minimal multi-turn slice sample."""
    s = {
        "id": f"{source_id}_t{turn_index}",
        "metadata": {
            "source_id": source_id,
            "turn_index": turn_index,
            "total_turns": total_turns,
        },
        "labels": labels or {
            "intent": "build", "difficulty": "intermediate",
            "language": ["python"],
        },
        "value": {
            "value_score": value_score,
            "complexity": {"overall": complexity},
            "quality": {"overall": quality},
            "reasoning": {"overall": reasoning},
            "rarity": {"score": rarity},
            "flags": flags or [],
        },
    }
    if inherited:
        s["labels"]["inherited"] = True
    if thinking_mode:
        s["value"]["thinking_mode"] = thinking_mode
    if score_confidence is not None:
        s["value"]["confidence"] = score_confidence
    if source_file:
        s["metadata"]["source_file"] = source_file
    if conversation_uid:
        s["metadata"]["conversation_uid"] = conversation_uid
    if conversations is not None:
        s["conversations"] = conversations
    return s


def _single_turn_sample(score=7.0):
    """Build a single-turn sample."""
    return {
        "id": "single-1",
        "metadata": {"total_turns": 1},
        "labels": {"intent": "build"},
        "value": {"value_score": score},
    }


# ── TestGroupByConversation ──

class TestGroupByConversation:
    def test_groups_multi_turn(self):
        samples = [
            _slice("conv1", 1, 3), _slice("conv1", 2, 3), _slice("conv1", 3, 3),
            _slice("conv2", 1, 2), _slice("conv2", 2, 2),
        ]
        groups = group_by_conversation(samples)
        assert len(groups) == 2
        assert len(groups["conv1"]) == 3
        assert len(groups["conv2"]) == 2

    def test_excludes_single_turn(self):
        samples = [_single_turn_sample(), _slice("conv1", 1, 2), _slice("conv1", 2, 2)]
        groups = group_by_conversation(samples)
        assert len(groups) == 1

    def test_sorts_by_turn_index(self):
        samples = [_slice("c", 3, 3), _slice("c", 1, 3), _slice("c", 2, 3)]
        groups = group_by_conversation(samples)
        indices = [s["metadata"]["turn_index"] for s in groups["c"]]
        assert indices == [1, 2, 3]

    def test_empty_input(self):
        assert group_by_conversation([]) == {}

    def test_no_source_id(self):
        sample = {"id": "x", "metadata": {"total_turns": 3}, "value": {}}
        assert group_by_conversation([sample]) == {}

    def test_source_file_separates_duplicate_source_ids(self):
        samples = [
            _slice("dup", 1, 2, source_file="/tmp/a/scored.json"),
            _slice("dup", 2, 2, source_file="/tmp/a/scored.json"),
            _slice("dup", 1, 2, source_file="/tmp/b/scored.json"),
            _slice("dup", 2, 2, source_file="/tmp/b/scored.json"),
        ]
        groups = group_by_conversation(samples)
        assert set(groups) == {
            build_conversation_key("dup", "/tmp/a/scored.json"),
            build_conversation_key("dup", "/tmp/b/scored.json"),
        }

    def test_conversation_uid_separates_duplicate_source_ids_in_same_file(self):
        samples = [
            _slice("dup", 1, 2, source_file="/tmp/a/scored.json", conversation_uid="uid-a"),
            _slice("dup", 2, 2, source_file="/tmp/a/scored.json", conversation_uid="uid-a"),
            _slice("dup", 1, 2, source_file="/tmp/a/scored.json", conversation_uid="uid-b"),
            _slice("dup", 2, 2, source_file="/tmp/a/scored.json", conversation_uid="uid-b"),
        ]
        groups = group_by_conversation(samples)
        assert set(groups) == {"uid-a", "uid-b"}

    def test_includes_single_slice_trajectory_object(self):
        sample = _slice("traj", 1, 1, conversations=[
            {"from": "human", "value": "Investigate and fix the bug."},
            {"from": "tool", "value": "pytest failed"},
            {"from": "tool", "value": "patched service.py"},
            {"from": "gpt", "value": "Issue fixed and tests pass."},
        ])
        sample["metadata"]["trajectory_object"] = True
        sample["metadata"]["trajectory_turn_count"] = 4
        groups = group_by_conversation([sample])
        assert list(groups) == ["traj"]
        assert len(groups["traj"]) == 1


# ── TestPositionWeight ──

class TestPositionWeight:
    def test_single(self):
        assert _position_weight(0, 1) == 1.0

    def test_two(self):
        assert _position_weight(0, 2) == 1.0
        assert _position_weight(1, 2) == 2.0

    def test_three(self):
        assert _position_weight(0, 3) == 1.0
        assert _position_weight(1, 3) == pytest.approx(1.5)
        assert _position_weight(2, 3) == 2.0

    def test_five(self):
        assert _position_weight(0, 5) == 1.0
        assert _position_weight(2, 5) == pytest.approx(1.5)
        assert _position_weight(4, 5) == 2.0


# ── TestEffectiveWeights ──

class TestEffectiveWeights:
    def test_no_inherited(self):
        slices = [_slice("c", 1, 2), _slice("c", 2, 2)]
        weights = _effective_weights(slices)
        assert weights[0] == pytest.approx(1.0)
        assert weights[1] == pytest.approx(2.0)

    def test_with_inherited(self):
        slices = [_slice("c", 1, 2, inherited=True), _slice("c", 2, 2)]
        weights = _effective_weights(slices)
        assert weights[0] == pytest.approx(0.7)
        assert weights[1] == pytest.approx(2.0)

    def test_all_inherited(self):
        slices = [_slice("c", 1, 2, inherited=True),
                  _slice("c", 2, 2, inherited=True)]
        weights = _effective_weights(slices)
        assert weights[0] == pytest.approx(0.7)
        assert weights[1] == pytest.approx(1.4)


# ── TestComputePenalty ──

class TestComputePenalty:
    def test_high_quality_no_penalty(self):
        slices = [_slice("c", 1, 2, quality=8), _slice("c", 2, 2, quality=7)]
        penalty, floor, flags = _compute_penalty(slices)
        assert penalty == pytest.approx(1.0)
        assert floor == 7
        assert flags == []

    def test_floor_below_5(self):
        slices = [_slice("c", 1, 2, quality=4), _slice("c", 2, 2, quality=7)]
        penalty, floor, flags = _compute_penalty(slices)
        assert penalty == pytest.approx(0.8)
        assert floor == 4

    def test_floor_below_3(self):
        slices = [_slice("c", 1, 2, quality=2), _slice("c", 2, 2, quality=7)]
        penalty, floor, flags = _compute_penalty(slices)
        assert penalty == pytest.approx(0.5)
        assert floor == 2

    def test_negative_flags(self):
        slices = [_slice("c", 1, 2, quality=8, flags=["has-bug"]),
                  _slice("c", 2, 2, quality=8, flags=["incomplete"])]
        penalty, floor, flags = _compute_penalty(slices)
        assert len(flags) == 2
        assert penalty == pytest.approx(1.0 * 0.95 ** 2)

    def test_combined_penalty(self):
        slices = [_slice("c", 1, 2, quality=2, flags=["has-bug"])]
        penalty, floor, flags = _compute_penalty(slices)
        assert penalty == pytest.approx(0.5 * 0.95)

    def test_dedup_flags(self):
        slices = [_slice("c", 1, 2, flags=["has-bug"]),
                  _slice("c", 2, 2, flags=["has-bug"])]
        _, _, flags = _compute_penalty(slices)
        assert flags == ["has-bug"]


# ── TestMergeLabels ──

class TestMergeLabels:
    def test_single_select_last_wins(self):
        slices = [
            _slice("c", 1, 2, labels={"intent": "build", "difficulty": "beginner"}),
            _slice("c", 2, 2, labels={"intent": "learn", "difficulty": "advanced"}),
        ]
        merged = _merge_labels(slices)
        assert merged["intent"] == "learn"
        assert merged["difficulty"] == "advanced"

    def test_multi_select_union(self):
        slices = [
            _slice("c", 1, 2, labels={"language": ["python", "sql"]}),
            _slice("c", 2, 2, labels={"language": ["python", "javascript"]}),
        ]
        merged = _merge_labels(slices)
        assert set(merged["language"]) == {"python", "sql", "javascript"}

    def test_dedup_multi_select(self):
        slices = [
            _slice("c", 1, 2, labels={"domain": ["web"]}),
            _slice("c", 2, 2, labels={"domain": ["web"]}),
        ]
        merged = _merge_labels(slices)
        assert merged["domain"] == ["web"]

    def test_skips_meta_keys(self):
        slices = [_slice("c", 1, 2, labels={
            "intent": "build", "confidence": 0.9, "inherited": True,
        })]
        merged = _merge_labels(slices)
        assert "confidence" not in merged
        assert "inherited" not in merged


# ── TestAggregateConversation ──

class TestAggregateConversation:
    def test_basic_3_turn(self):
        slices = [
            _slice("c1", 1, 3, value_score=5.0, complexity=4),
            _slice("c1", 2, 3, value_score=6.0, complexity=6),
            _slice("c1", 3, 3, value_score=8.0, complexity=9),
        ]
        rec = aggregate_conversation("c1", slices)
        assert rec is not None
        assert rec["conversation_id"] == "c1"
        assert rec["turn_count"] == 3
        assert rec["conv_value"] > 0
        assert rec["conv_value_v2"] == rec["conv_value"]
        assert rec["peak_complexity"] == 9
        assert len(rec["slices"]) == 3

    def test_record_includes_conversation_key(self):
        slices = [
            _slice("c1", 1, 2, source_file="/tmp/a/scored.json"),
            _slice("c1", 2, 2, source_file="/tmp/a/scored.json"),
        ]
        rec = aggregate_conversation(build_conversation_key("c1", "/tmp/a/scored.json"), slices)
        assert rec["conversation_key"] == build_conversation_key("c1", "/tmp/a/scored.json")
        assert rec["source_file"] == "/tmp/a/scored.json"

    def test_single_slice_returns_none(self):
        slices = [_slice("c1", 1, 1)]
        assert aggregate_conversation("c1", slices) is None

    def test_single_slice_trajectory_object_returns_record(self):
        sample = _slice("traj", 1, 1, value_score=8.2, quality=8.0, reasoning=7.0, rarity=6.5, conversations=[
            {"from": "human", "value": "Fix the failing build and confirm tests."},
            {"role": "assistant", "content": "<tool_call>{\"name\":\"execute_bash\",\"arguments\":{\"command\":\"pytest -q\"}}</tool_call>"},
            {"role": "tool", "name": "execute_bash", "content": "FAILED tests/test_api.py::test_create"},
            {"role": "assistant", "content": "<tool_call>{\"name\":\"str_replace_editor\",\"arguments\":{\"path\":\"service.py\",\"old_str\":\"0\",\"new_str\":\"1\"}}</tool_call>"},
            {"role": "tool", "name": "str_replace_editor", "content": "Updated service.py"},
            {"from": "gpt", "value": "Patched service.py and tests now pass."},
        ], labels={"intent": "debug", "difficulty": "advanced", "language": ["python"], "agentic": ["tool-use"]})
        sample["metadata"]["trajectory_object"] = True
        sample["metadata"]["trajectory_turn_count"] = 6
        sample["metadata"]["trajectory_tool_turn_count"] = 2
        sample["metadata"]["trajectory_assistant_turn_count"] = 3

        rec = aggregate_conversation("traj", [sample])

        assert rec is not None
        assert rec["turn_count"] == 6
        assert rec["detail"]["trajectory_v2_enabled"] is True

    def test_missing_values_returns_none(self):
        slices = [
            {"id": "a", "metadata": {"source_id": "c", "turn_index": 1, "total_turns": 2}},
            {"id": "b", "metadata": {"source_id": "c", "turn_index": 2, "total_turns": 2}},
        ]
        assert aggregate_conversation("c", slices) is None

    def test_inherited_weighting(self):
        s1 = _slice("c", 1, 2, value_score=4.0, inherited=True)
        s2 = _slice("c", 2, 2, value_score=8.0)
        rec = aggregate_conversation("c", [s1, s2])
        # Inherited first slice has lower weight → result skews toward 8.0
        assert rec["conv_value"] > 6.0

    def test_penalty_effect(self):
        good = [_slice("c", 1, 2, quality=8), _slice("c", 2, 2, quality=8)]
        bad = [_slice("c", 1, 2, quality=2), _slice("c", 2, 2, quality=8)]
        rec_good = aggregate_conversation("c", good)
        rec_bad = aggregate_conversation("c", bad)
        assert rec_good["conv_value"] > rec_bad["conv_value"]

    def test_clamp_1_10(self):
        slices = [_slice("c", 1, 2, value_score=10.0, quality=10),
                  _slice("c", 2, 2, value_score=10.0, quality=10)]
        rec = aggregate_conversation("c", slices)
        assert 1.0 <= rec["conv_value"] <= 10.0

    def test_low_score_confidence_shrinks_conv_value(self):
        high_conf = aggregate_conversation("c", [
            _slice("c", 1, 2, value_score=9.0, quality=9, reasoning=9, score_confidence=0.95),
            _slice("c", 2, 2, value_score=9.0, quality=9, reasoning=9, score_confidence=0.95),
        ])
        low_conf = aggregate_conversation("c", [
            _slice("c", 1, 2, value_score=9.0, quality=9, reasoning=9, score_confidence=0.2),
            _slice("c", 2, 2, value_score=9.0, quality=9, reasoning=9, score_confidence=0.2),
        ])
        assert low_conf["conv_value"] < high_conf["conv_value"]

    def test_conv_rarity_rewards_observed_label_diversity(self):
        uniform = aggregate_conversation("uniform", [
            _slice("uniform", 1, 3, rarity=7.0,
                   labels={"intent": "build", "difficulty": "intermediate", "language": ["python"]}),
            _slice("uniform", 2, 3, rarity=7.0,
                   labels={"intent": "build", "difficulty": "intermediate", "language": ["python"]}),
            _slice("uniform", 3, 3, rarity=7.0,
                   labels={"intent": "build", "difficulty": "intermediate", "language": ["python"]}),
        ])
        diverse = aggregate_conversation("diverse", [
            _slice("diverse", 1, 3, rarity=7.0,
                   labels={"intent": "build", "difficulty": "intermediate", "language": ["python"]}),
            _slice("diverse", 2, 3, rarity=7.0,
                   labels={"intent": "debug", "difficulty": "advanced", "language": ["rust"]}),
            _slice("diverse", 3, 3, rarity=7.0,
                   labels={"intent": "modify", "difficulty": "advanced", "language": ["go"]}),
        ])
        assert diverse["conv_rarity"] > uniform["conv_rarity"]
        assert diverse["detail"]["label_signature_count"] > uniform["detail"]["label_signature_count"]

    def test_conv_rarity_shrinks_when_inheritance_dominates(self):
        observed = aggregate_conversation("observed", [
            _slice("observed", 1, 3, rarity=8.5, score_confidence=0.95),
            _slice("observed", 2, 3, rarity=8.5, score_confidence=0.95),
            _slice("observed", 3, 3, rarity=8.5, score_confidence=0.95),
        ])
        inherited = aggregate_conversation("inherited", [
            _slice("inherited", 1, 3, rarity=8.5, score_confidence=0.95, inherited=True),
            _slice("inherited", 2, 3, rarity=8.5, score_confidence=0.95, inherited=True),
            _slice("inherited", 3, 3, rarity=8.5, score_confidence=0.95, inherited=True),
        ])
        assert inherited["conv_rarity"] < observed["conv_rarity"]
        assert inherited["rarity_confidence"] < observed["rarity_confidence"]
        assert inherited["observed_turn_ratio"] < observed["observed_turn_ratio"]

    def test_conversation_diagnostics_include_top_bottom_std_and_late_gain(self):
        slices = [
            _slice("diag", 1, 4, value_score=3.0, quality=4),
            _slice("diag", 2, 4, value_score=5.0, quality=6),
            _slice("diag", 3, 4, value_score=8.0, quality=7),
            _slice("diag", 4, 4, value_score=9.0, quality=9),
        ]
        rec = aggregate_conversation("diag", slices)

        assert rec["compression_gap"] == pytest.approx(2.75)
        assert rec["late_turn_gain"] == pytest.approx(6.0)
        assert rec["detail"]["turn_value_mean"] == pytest.approx(6.25)
        assert rec["detail"]["turn_value_min"] == pytest.approx(3.0)
        assert rec["detail"]["turn_value_max"] == pytest.approx(9.0)
        assert rec["detail"]["turn_value_median"] == pytest.approx(6.5)
        assert rec["detail"]["top_k_mean"] == pytest.approx(6.25)
        assert rec["detail"]["bottom_k_mean"] == pytest.approx((3.0 + 5.0 + 8.0) / 3, abs=0.01)
        assert rec["detail"]["turn_value_std"] > 0
        assert rec["detail"]["turn_quality_std"] > 0
        assert rec["detail"]["conv_turn_signal_raw"] >= rec["detail"]["bottom_k_mean"]
        assert rec["detail"]["conv_turn_signal"] <= rec["detail"]["conv_turn_signal_raw"]
        assert rec["trajectory_structure_score"] is not None

    def test_conversation_structure_signals_extract_tools_files_and_test_turns(self):
        full_conversation = [
            {"from": "human", "value": "Fix the repo and keep tests green."},
            {"from": "gpt", "value": "I'll inspect the failing test first."},
            {"role": "assistant", "content": "<tool_call>{\"name\":\"execute_bash\",\"arguments\":{\"command\":\"pytest tests/test_api.py -q\"}}</tool_call>"},
            {"role": "tool", "name": "execute_bash", "content": "FAILED tests/test_api.py::test_create\nservice.go: expected 1 got 0"},
            {"role": "assistant", "content": "<tool_call>{\"name\":\"str_replace_editor\",\"arguments\":{\"path\":\"service.go\",\"old_str\":\"return 0\",\"new_str\":\"return int(id)\"}}</tool_call>"},
            {"role": "tool", "name": "str_replace_editor", "content": "Updated service.go"},
            {"role": "assistant", "content": "Patched service.go and tests/test_api.py should pass now."},
        ]
        slices = [
            _slice("signals", 1, 2, conversations=full_conversation[:4]),
            _slice("signals", 2, 2, conversations=full_conversation),
        ]

        rec = aggregate_conversation("signals", slices)

        assert rec["tool_turn_ratio"] == pytest.approx(4 / 7, abs=0.01)
        assert rec["unique_tool_count"] == 2
        assert rec["unique_file_count"] >= 2
        assert set(rec["detail"]["unique_tools"]) == {"execute_bash", "str_replace_editor"}
        assert rec["detail"]["tool_turn_count"] == 4
        assert rec["detail"]["test_related_turn_count"] >= 2
        assert rec["detail"]["edit_related_turn_count"] >= 2
        assert rec["detail"]["bash_execution_turn_count"] >= 2

    def test_long_tool_heavy_trajectory_emits_v2_calibration(self):
        full_conversation = [
            {"from": "human", "value": "Please investigate the failing integration tests and fix the issue."},
            {"from": "gpt", "value": "I'll inspect the failing tests."},
            {"role": "assistant", "content": "<tool_call>{\"name\":\"execute_bash\",\"arguments\":{\"command\":\"pytest tests/test_api.py -q\"}}</tool_call>"},
            {"role": "tool", "name": "execute_bash", "content": "FAILED tests/test_api.py::test_create"},
            {"role": "assistant", "content": "<tool_call>{\"name\":\"str_replace_editor\",\"arguments\":{\"path\":\"service.py\",\"old_str\":\"return 0\",\"new_str\":\"return 1\"}}</tool_call>"},
            {"role": "tool", "name": "str_replace_editor", "content": "Updated service.py"},
            {"role": "assistant", "content": "<tool_call>{\"name\":\"execute_bash\",\"arguments\":{\"command\":\"pytest tests/test_api.py -q\"}}</tool_call>"},
            {"role": "tool", "name": "execute_bash", "content": "PASSED tests/test_api.py::test_create"},
            {"role": "assistant", "content": "Fixed service.py and tests are now green."},
        ]
        slices = [
            _slice("traj", idx + 1, 8, value_score=score, quality=quality, conversations=full_conversation,
                   labels={"intent": "debug", "difficulty": "advanced", "language": ["python"], "agentic": ["tool-use"]})
            for idx, (score, quality) in enumerate([
                (4.2, 4.0),
                (4.8, 5.0),
                (5.5, 5.0),
                (6.2, 6.0),
                (6.8, 7.0),
                (7.4, 8.0),
                (8.2, 8.0),
                (8.5, 9.0),
            ])
        ]

        rec = aggregate_conversation("traj", slices)

        assert rec["detail"]["trajectory_v2_enabled"] is True
        assert rec["conv_value_v2"] >= rec["conv_value"]
        assert rec["trajectory_structure_score"] >= 5.0
        assert rec["detail"]["conv_turn_signal"] >= rec["detail"]["turn_value_mean"]


# ── TestComputeConvSelectionScores ──

class TestComputeConvSelectionScores:
    def test_basic_ranking(self):
        records = []
        for i in range(10):
            rec = aggregate_conversation(
                f"c{i}",
                [_slice(f"c{i}", 1, 2, value_score=float(i + 1),
                        complexity=i + 1, quality=i + 1, reasoning=i + 1,
                        rarity=float(i + 1)),
                 _slice(f"c{i}", 2, 2, value_score=float(i + 1),
                        complexity=i + 1, quality=i + 1, reasoning=i + 1,
                        rarity=float(i + 1))],
            )
            if rec:
                records.append(rec)
        compute_conv_selection_scores(records)
        # Higher quality → higher selection
        selections = [r["conv_selection"] for r in records if r.get("conv_selection")]
        assert len(selections) > 0

    def test_empty_input(self):
        compute_conv_selection_scores([])  # should not raise

    def test_selection_shrinks_when_inheritance_bias_is_high(self):
        observed = [
            aggregate_conversation(
                f"obs{i}",
                [
                    _slice(f"obs{i}", 1, 2, value_score=9.0, quality=9, reasoning=9, rarity=8.5, score_confidence=0.95),
                    _slice(f"obs{i}", 2, 2, value_score=9.0, quality=9, reasoning=9, rarity=8.5, score_confidence=0.95),
                ],
            )
            for i in range(5)
        ]
        inherited = [
            aggregate_conversation(
                f"inh{i}",
                [
                    _slice(f"inh{i}", 1, 2, value_score=9.0, quality=9, reasoning=9, rarity=8.5, score_confidence=0.95, inherited=True),
                    _slice(f"inh{i}", 2, 2, value_score=9.0, quality=9, reasoning=9, rarity=8.5, score_confidence=0.95, inherited=True),
                ],
            )
            for i in range(5)
        ]
        records = observed + inherited
        compute_conv_selection_scores(records)
        assert inherited[0]["conv_selection"] < observed[0]["conv_selection"]
        assert inherited[0]["detail"]["selection_confidence"] < observed[0]["detail"]["selection_confidence"]
        assert inherited[0]["conv_selection_v2"] <= observed[0]["conv_selection_v2"]

    def test_selection_v2_tracks_structure_for_long_trajectories(self):
        records = []
        for idx, values in enumerate((
            [(4.0, 4.0), (5.0, 5.0), (5.5, 5.0), (6.0, 6.0), (6.5, 6.0), (7.0, 7.0), (7.8, 8.0), (8.2, 9.0)],
            [(5.5, 6.0), (5.7, 6.0), (5.9, 6.0), (6.0, 6.0), (6.1, 6.0), (6.2, 6.0), (6.3, 6.0), (6.4, 6.0)],
        )):
            full_conversation = [
                {"from": "human", "value": "Fix the failing build."},
                {"role": "assistant", "content": "<tool_call>{\"name\":\"execute_bash\",\"arguments\":{\"command\":\"pytest -q\"}}</tool_call>"},
                {"role": "tool", "name": "execute_bash", "content": "output"},
                {"role": "assistant", "content": "<tool_call>{\"name\":\"str_replace_editor\",\"arguments\":{\"path\":\"main.py\",\"old_str\":\"0\",\"new_str\":\"1\"}}</tool_call>"},
                {"role": "tool", "name": "str_replace_editor", "content": "updated"},
                {"role": "assistant", "content": "Fixed the issue and tests are green." if idx == 0 else "I explored several ideas."},
            ]
            slices = [
                _slice(
                    f"sel{idx}",
                    turn_idx + 1,
                    8,
                    value_score=score,
                    quality=quality,
                    rarity=6.0,
                    conversations=full_conversation,
                    labels={"intent": "debug", "difficulty": "advanced", "language": ["python"], "agentic": ["tool-use"]},
                )
                for turn_idx, (score, quality) in enumerate(values)
            ]
            records.append(aggregate_conversation(f"sel{idx}", slices))

        compute_conv_selection_scores(records)
        assert records[0]["conv_selection_v2"] >= records[0]["conv_selection"]
        assert records[0]["conv_selection_v2"] > records[1]["conv_selection_v2"]


# ── TestAggregateConversations ──

class TestAggregateConversations:
    def test_full_pipeline(self):
        samples = [
            _slice("c1", 1, 3, value_score=5.0),
            _slice("c1", 2, 3, value_score=6.0),
            _slice("c1", 3, 3, value_score=8.0),
            _slice("c2", 1, 2, value_score=7.0),
            _slice("c2", 2, 2, value_score=9.0),
        ]
        records = aggregate_conversations(samples)
        assert len(records) == 2
        ids = {r["conversation_id"] for r in records}
        assert ids == {"c1", "c2"}

    def test_duplicate_source_ids_across_files_stay_separate(self):
        samples = [
            _slice("dup", 1, 2, source_file="/tmp/a/scored.json"),
            _slice("dup", 2, 2, source_file="/tmp/a/scored.json"),
            _slice("dup", 1, 2, source_file="/tmp/b/scored.json"),
            _slice("dup", 2, 2, source_file="/tmp/b/scored.json"),
        ]
        records = aggregate_conversations(samples)
        assert len(records) == 2
        keys = {r["conversation_key"] for r in records}
        assert keys == {
            build_conversation_key("dup", "/tmp/a/scored.json"),
            build_conversation_key("dup", "/tmp/b/scored.json"),
        }

    def test_duplicate_source_ids_same_file_use_conversation_uid(self):
        samples = [
            _slice("dup", 1, 2, source_file="/tmp/a/scored.json", conversation_uid="uid-a"),
            _slice("dup", 2, 2, source_file="/tmp/a/scored.json", conversation_uid="uid-a"),
            _slice("dup", 1, 2, source_file="/tmp/a/scored.json", conversation_uid="uid-b"),
            _slice("dup", 2, 2, source_file="/tmp/a/scored.json", conversation_uid="uid-b"),
        ]
        records = aggregate_conversations(samples)
        assert len(records) == 2
        keys = {r["conversation_key"] for r in records}
        assert keys == {"uid-a", "uid-b"}

    def test_single_turn_only_returns_empty(self):
        samples = [_single_turn_sample(), _single_turn_sample()]
        assert aggregate_conversations(samples) == []


# ── TestWriteConversationScores ──

class TestWriteConversationScores:
    def test_write_and_read(self, tmp_path):
        records = aggregate_conversations([
            _slice("c1", 1, 2, value_score=6.0),
            _slice("c1", 2, 2, value_score=8.0),
        ])
        path = tmp_path / "conversation_scores.json"
        write_conversation_scores(records, path)
        assert path.exists()
        with open(path) as f:
            loaded = json.load(f)
        assert len(loaded) == 1
        assert loaded[0]["conversation_id"] == "c1"
