"""Tests for conversation-level value aggregation."""

from __future__ import annotations

import json
import pytest
from pathlib import Path

from sft_label.conversation import (
    group_by_conversation,
    _position_weight,
    _effective_weights,
    _weighted_average,
    _compute_penalty,
    _merge_labels,
    _compute_pure_quality_from_slice,
    aggregate_conversation,
    compute_conv_selection_scores,
    aggregate_conversations,
    write_conversation_scores,
)


# ── Helpers ──

def _slice(source_id, turn_index, total_turns, value_score=6.0,
           complexity=5, quality=6, reasoning=5, rarity=4.0,
           inherited=False, flags=None, labels=None, thinking_mode=None):
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
        assert rec["peak_complexity"] == 9
        assert len(rec["slices"]) == 3

    def test_single_slice_returns_none(self):
        slices = [_slice("c1", 1, 1)]
        assert aggregate_conversation("c1", slices) is None

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
