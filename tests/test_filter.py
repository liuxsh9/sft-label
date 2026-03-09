"""Tests for the value filter tool."""

import json
import pytest
from pathlib import Path

from sft_label.tools.filter_value import (
    filter_samples, run_filter, matches_filter, FilterConfig,
    _find_scored_files,
    _load_conversation_scores,
)


# ── Helpers ──

def _scored(score, **extra):
    """Create a sample with a given value_score."""
    sample = {"id": f"s-{score}", "value": {"value_score": score}}
    sample.update(extra)
    return sample


def _full_sample(score, labels=None, metadata=None, value_extra=None):
    """Create a fully populated sample."""
    sample = {
        "id": f"full-{score}",
        "conversations": [
            {"from": "human", "value": "test question"},
            {"from": "gpt", "value": "test answer"},
        ],
        "labels": labels or {},
        "metadata": metadata or {},
        "value": {"value_score": score, **(value_extra or {})},
    }
    return sample


def _unscored():
    """Create a sample without value data."""
    return {"id": "unscored-1", "messages": []}


# ── filter_samples (backward compat) ──

class TestFilterSamples:
    def test_basic_threshold(self):
        samples = [_scored(3.0), _scored(5.0), _scored(7.0), _scored(9.0)]
        retained, dropped = filter_samples(samples, threshold=6.0)
        assert len(retained) == 2
        assert all(s["value"]["value_score"] >= 6.0 for s in retained)
        assert len(dropped) == 2

    def test_exact_threshold(self):
        samples = [_scored(6.0), _scored(5.9)]
        retained, dropped = filter_samples(samples, threshold=6.0)
        assert len(retained) == 1
        assert retained[0]["value"]["value_score"] == 6.0

    def test_unscored_excluded_by_default(self):
        samples = [_scored(8.0), _unscored()]
        retained, dropped = filter_samples(samples, threshold=5.0)
        assert len(retained) == 1
        assert len(dropped) == 1

    def test_unscored_included_when_flag_set(self):
        samples = [_scored(8.0), _unscored()]
        retained, dropped = filter_samples(samples, threshold=5.0, include_unscored=True)
        assert len(retained) == 2
        assert len(dropped) == 0

    def test_empty_input(self):
        retained, dropped = filter_samples([], threshold=5.0)
        assert retained == []
        assert dropped == []

    def test_all_below_threshold(self):
        samples = [_scored(1.0), _scored(2.0), _scored(3.0)]
        retained, dropped = filter_samples(samples, threshold=5.0)
        assert len(retained) == 0
        assert len(dropped) == 3

    def test_all_above_threshold(self):
        samples = [_scored(7.0), _scored(8.0), _scored(9.0)]
        retained, dropped = filter_samples(samples, threshold=5.0)
        assert len(retained) == 3
        assert len(dropped) == 0

    def test_missing_value_key(self):
        """Sample with value dict but no value_score treated as unscored."""
        samples = [{"id": "x", "value": {"complexity": {"overall": 5}}}]
        retained, dropped = filter_samples(samples, threshold=5.0)
        assert len(retained) == 0
        assert len(dropped) == 1


# ── matches_filter ──

class TestMatchesFilter:
    def test_value_min(self):
        config = FilterConfig(value_min=6.0)
        assert matches_filter(_scored(7.0), config) is True
        assert matches_filter(_scored(5.0), config) is False
        assert matches_filter(_scored(6.0), config) is True

    def test_selection_min(self):
        config = FilterConfig(selection_min=5.0)
        sample = _full_sample(7.0, value_extra={"selection_score": 6.0})
        assert matches_filter(sample, config) is True
        sample_low = _full_sample(7.0, value_extra={"selection_score": 3.0})
        assert matches_filter(sample_low, config) is False

    def test_thinking_mode(self):
        config = FilterConfig(thinking_mode="slow")
        sample_slow = _full_sample(7.0, value_extra={"thinking_mode": "slow"})
        sample_fast = _full_sample(7.0, value_extra={"thinking_mode": "fast"})
        assert matches_filter(sample_slow, config) is True
        assert matches_filter(sample_fast, config) is False

    def test_thinking_mode_from_metadata(self):
        config = FilterConfig(thinking_mode="slow")
        sample = _full_sample(7.0, metadata={"thinking_mode": "slow"})
        assert matches_filter(sample, config) is True

    def test_difficulty_filter(self):
        config = FilterConfig(difficulty=["advanced", "expert"])
        sample_adv = _full_sample(7.0, labels={"difficulty": "advanced"})
        sample_beg = _full_sample(7.0, labels={"difficulty": "beginner"})
        assert matches_filter(sample_adv, config) is True
        assert matches_filter(sample_beg, config) is False

    def test_include_tags_or_logic(self):
        config = FilterConfig(include_tags=["domain:web", "domain:ml"])
        sample_web = _full_sample(7.0, labels={"domain": ["web", "backend"]})
        sample_ml = _full_sample(7.0, labels={"domain": ["ml"]})
        sample_db = _full_sample(7.0, labels={"domain": ["database"]})
        assert matches_filter(sample_web, config) is True
        assert matches_filter(sample_ml, config) is True
        assert matches_filter(sample_db, config) is False

    def test_exclude_tags_or_logic(self):
        config = FilterConfig(exclude_tags=["intent:fix-bug"])
        sample_fix = _full_sample(7.0, labels={"intent": "fix-bug"})
        sample_gen = _full_sample(7.0, labels={"intent": "generate"})
        assert matches_filter(sample_fix, config) is False
        assert matches_filter(sample_gen, config) is True

    def test_exclude_inherited(self):
        config = FilterConfig(exclude_inherited=True)
        sample_inherited = _full_sample(7.0, labels={"inherited": True, "difficulty": "advanced"})
        sample_normal = _full_sample(7.0, labels={"difficulty": "advanced"})
        assert matches_filter(sample_inherited, config) is False
        assert matches_filter(sample_normal, config) is True

    def test_combined_criteria(self):
        """AND logic between different criteria."""
        config = FilterConfig(
            value_min=6.0,
            difficulty=["advanced", "expert"],
            thinking_mode="slow",
        )
        # Meets all criteria
        sample_good = _full_sample(7.0,
                                    labels={"difficulty": "advanced"},
                                    value_extra={"thinking_mode": "slow"})
        assert matches_filter(sample_good, config) is True

        # Fails value
        sample_low = _full_sample(5.0,
                                   labels={"difficulty": "advanced"},
                                   value_extra={"thinking_mode": "slow"})
        assert matches_filter(sample_low, config) is False

        # Fails difficulty
        sample_easy = _full_sample(7.0,
                                    labels={"difficulty": "beginner"},
                                    value_extra={"thinking_mode": "slow"})
        assert matches_filter(sample_easy, config) is False

    def test_unscored_handling(self):
        config_no = FilterConfig(value_min=5.0)
        config_yes = FilterConfig(value_min=5.0, include_unscored=True)
        sample = _unscored()
        assert matches_filter(sample, config_no) is False
        assert matches_filter(sample, config_yes) is True

    def test_verify_source(self, tmp_path):
        source_path = tmp_path / "data.json"
        source_path.touch()
        config = FilterConfig(verify_source=str(source_path))
        sample_match = _full_sample(7.0, metadata={"source_file": str(source_path)})
        sample_wrong = _full_sample(7.0, metadata={"source_file": "/other/path.json"})
        sample_none = _full_sample(7.0)
        assert matches_filter(sample_match, config) is True
        assert matches_filter(sample_wrong, config) is False
        # No source_file metadata → not excluded (graceful handling)
        assert matches_filter(sample_none, config) is True

    def test_tag_without_dimension(self):
        """Tag search without explicit dimension scans all dimensions."""
        config = FilterConfig(include_tags=["python"])
        sample = _full_sample(7.0, labels={"language": ["python", "javascript"]})
        assert matches_filter(sample, config) is True

    def test_no_criteria_passes_all(self):
        """Empty config passes everything (no score-based criteria)."""
        config = FilterConfig()
        assert matches_filter(_scored(3.0), config) is True
        assert matches_filter(_scored(9.0), config) is True


# ── FilterConfig with filter_samples ──

class TestFilterConfig:
    def test_config_based_filter(self):
        config = FilterConfig(value_min=6.0, difficulty=["expert"])
        samples = [
            _full_sample(7.0, labels={"difficulty": "expert"}),
            _full_sample(8.0, labels={"difficulty": "beginner"}),
            _full_sample(4.0, labels={"difficulty": "expert"}),
        ]
        retained, dropped = filter_samples(samples, config=config)
        assert len(retained) == 1
        assert retained[0]["value"]["value_score"] == 7.0


# ── run_filter (integration) ──

class TestRunFilter:
    def test_json_file(self, tmp_path):
        samples = [_scored(3.0), _scored(5.5), _scored(7.0), _scored(8.5)]
        input_file = tmp_path / "scored.json"
        input_file.write_text(json.dumps(samples))

        summary = run_filter(str(input_file), threshold=6.0)

        assert summary["total"] == 4
        assert summary["retained"] == 2
        assert summary["dropped"] == 2
        assert summary["retention_rate"] == pytest.approx(0.5)
        assert summary["mean_value_retained"] == pytest.approx(7.75)

        # Check output files exist
        out_json = Path(summary["output_json"])
        out_jsonl = Path(summary["output_jsonl"])
        assert out_json.exists()
        assert out_jsonl.exists()

        # Verify content
        with open(out_json) as f:
            result = json.load(f)
        assert len(result) == 2
        assert all(s["value"]["value_score"] >= 6.0 for s in result)

    def test_jsonl_file(self, tmp_path):
        samples = [_scored(4.0), _scored(9.0)]
        input_file = tmp_path / "scored.jsonl"
        with open(input_file, "w") as f:
            for s in samples:
                f.write(json.dumps(s) + "\n")

        summary = run_filter(str(input_file), threshold=5.0)
        assert summary["retained"] == 1
        assert summary["dropped"] == 1

    def test_directory_mode(self, tmp_path):
        # Create scored files in directory and subdirectory
        (tmp_path / "scored.json").write_text(json.dumps([_scored(7.0)]))
        sub = tmp_path / "subdir"
        sub.mkdir()
        (sub / "scored_extra.json").write_text(json.dumps([_scored(3.0), _scored(8.0)]))

        summary = run_filter(str(tmp_path), threshold=6.0)
        assert summary["total"] == 3
        assert summary["retained"] == 2

    def test_custom_output(self, tmp_path):
        input_file = tmp_path / "scored.json"
        input_file.write_text(json.dumps([_scored(7.0)]))
        out = tmp_path / "out" / "filtered.json"

        summary = run_filter(str(input_file), threshold=5.0, output_path=str(out))
        assert Path(summary["output_json"]) == out
        assert out.exists()

    def test_no_scored_files_raises(self, tmp_path):
        empty_dir = tmp_path / "empty"
        empty_dir.mkdir()
        with pytest.raises(FileNotFoundError):
            run_filter(str(empty_dir), threshold=5.0)

    def test_empty_samples(self, tmp_path):
        input_file = tmp_path / "scored.json"
        input_file.write_text(json.dumps([]))

        summary = run_filter(str(input_file), threshold=5.0)
        assert summary["total"] == 0
        assert summary["retained"] == 0


# ── Directory mode (complex scenarios) ──

class TestDirectoryModeComplex:
    def test_mixed_json_jsonl(self, tmp_path):
        """Directory with both .json and .jsonl scored files."""
        (tmp_path / "scored_a.json").write_text(json.dumps([_scored(7.0), _scored(8.0)]))
        with open(tmp_path / "scored_b.jsonl", "w") as f:
            f.write(json.dumps(_scored(6.0)) + "\n")
            f.write(json.dumps(_scored(9.0)) + "\n")

        summary = run_filter(str(tmp_path), threshold=7.0)
        assert summary["total"] == 4
        assert summary["retained"] == 3  # 7.0, 8.0, 9.0

    def test_nested_subdirectories(self, tmp_path):
        """Scored files one level deep in subdirectories."""
        sub1 = tmp_path / "run1"
        sub1.mkdir()
        (sub1 / "scored.json").write_text(json.dumps([_scored(7.0)]))

        sub2 = tmp_path / "run2"
        sub2.mkdir()
        (sub2 / "scored_extra.jsonl").write_text(
            json.dumps(_scored(5.0)) + "\n" + json.dumps(_scored(8.0)) + "\n"
        )

        summary = run_filter(str(tmp_path), threshold=6.0)
        assert summary["total"] == 3
        assert summary["retained"] == 2  # 7.0, 8.0

    def test_ignores_non_scored_files(self, tmp_path):
        """Only files matching scored* pattern are loaded."""
        (tmp_path / "scored.json").write_text(json.dumps([_scored(7.0)]))
        (tmp_path / "labeled.json").write_text(json.dumps([_scored(1.0)]))
        (tmp_path / "other.json").write_text(json.dumps([_scored(1.0)]))

        summary = run_filter(str(tmp_path), threshold=5.0)
        assert summary["total"] == 1
        assert summary["retained"] == 1

    def test_directory_with_config(self, tmp_path):
        """Multi-condition filtering across directory files."""
        samples = [
            _full_sample(7.0, labels={"difficulty": "expert"}),
            _full_sample(8.0, labels={"difficulty": "beginner"}),
            _full_sample(9.0, labels={"difficulty": "advanced"}),
        ]
        (tmp_path / "scored.json").write_text(json.dumps(samples))

        config = FilterConfig(value_min=6.0, difficulty=["expert", "advanced"])
        summary = run_filter(str(tmp_path), config=config)
        assert summary["total"] == 3
        assert summary["retained"] == 2  # 7.0 expert, 9.0 advanced

    def test_directory_deduplicates_files(self, tmp_path):
        """Same file shouldn't be loaded twice via different glob patterns."""
        (tmp_path / "scored.json").write_text(json.dumps([_scored(7.0)]))
        # scored.json matches both "scored*.json" at root level
        summary = run_filter(str(tmp_path), threshold=5.0)
        assert summary["total"] == 1

    def test_directory_deduplicates_json_jsonl(self, tmp_path):
        """When both scored.json and scored.jsonl exist, only one is loaded."""
        data = [_scored(7.0), _scored(5.0)]
        (tmp_path / "scored.json").write_text(json.dumps(data))
        with open(tmp_path / "scored.jsonl", "w") as f:
            for s in data:
                f.write(json.dumps(s) + "\n")
        # Both files have identical data — should NOT double-count
        summary = run_filter(str(tmp_path), threshold=5.0)
        assert summary["total"] == 2  # not 4

    def test_directory_jsonl_streaming(self, tmp_path):
        """Large JSONL file in directory mode - verify streaming works."""
        n = 100
        with open(tmp_path / "scored_large.jsonl", "w") as f:
            for i in range(n):
                score = float(i % 10 + 1)
                f.write(json.dumps(_scored(score)) + "\n")

        summary = run_filter(str(tmp_path), threshold=8.0)
        assert summary["total"] == n
        # Scores 8, 9, 10 → 3 out of 10, repeated 10 times
        assert summary["retained"] == 30

    def test_nested_two_level_subdirectories(self, tmp_path):
        """Scored files two levels deep (e.g., code/part1/scored.json)."""
        deep = tmp_path / "code" / "part1"
        deep.mkdir(parents=True)
        (deep / "scored.json").write_text(json.dumps([_scored(7.0), _scored(8.0)]))

        deep2 = tmp_path / "multi_turn" / "trajectories"
        deep2.mkdir(parents=True)
        (deep2 / "scored.json").write_text(json.dumps([_scored(9.0)]))

        summary = run_filter(str(tmp_path), threshold=6.0)
        assert summary["total"] == 3
        assert summary["retained"] == 3

    def test_conversation_scores_nested_two_levels(self, tmp_path):
        """conversation_scores.json two levels deep should be discovered."""
        deep = tmp_path / "multi_turn" / "traj"
        deep.mkdir(parents=True)
        conv_data = [{"conversation_id": "c1", "conv_value": 8.0}]
        (deep / "conversation_scores.json").write_text(json.dumps(conv_data))

        lookup = _load_conversation_scores(str(tmp_path))
        assert "c1" in lookup
        assert lookup["c1"]["conv_value"] == 8.0

    def test_conversation_scores_nested_deep_levels(self, tmp_path):
        """conversation_scores.json deeper than 2 levels should be discovered."""
        deep = tmp_path / "a" / "b" / "c" / "d"
        deep.mkdir(parents=True)
        conv_data = [{"conversation_id": "deep-c1", "conv_value": 9.0}]
        (deep / "conversation_scores.json").write_text(json.dumps(conv_data))

        lookup = _load_conversation_scores(str(tmp_path))
        assert "deep-c1" in lookup
        assert lookup["deep-c1"]["conv_value"] == 9.0

    def test_directory_empty_files(self, tmp_path):
        """Directory with empty scored files."""
        (tmp_path / "scored.json").write_text(json.dumps([]))
        (tmp_path / "scored_b.json").write_text(json.dumps([_scored(7.0)]))

        summary = run_filter(str(tmp_path), threshold=5.0)
        assert summary["total"] == 1
        assert summary["retained"] == 1


# ── Training format output ──

class TestTrainingFormat:
    def test_training_format_sharegpt(self, tmp_path):
        """Training format strips labels and scores for ShareGPT samples."""
        samples = [
            _full_sample(7.0, labels={"difficulty": "advanced"}),
        ]
        input_file = tmp_path / "scored.json"
        input_file.write_text(json.dumps(samples))

        config = FilterConfig(value_min=5.0, output_format="training")
        summary = run_filter(str(input_file), config=config)

        jsonl_path = Path(summary["output_jsonl"])
        assert jsonl_path.exists()

        with open(jsonl_path) as f:
            result = json.loads(f.readline())
        assert "id" in result
        assert "conversations" in result
        assert "labels" not in result
        assert "value" not in result

    def test_training_format_pangu(self, tmp_path):
        """Training format converts Pangu samples back to pseudo-multiturn."""
        sample = _full_sample(7.0,
                               labels={"difficulty": "advanced"},
                               metadata={"original_format": "pangu"})
        input_file = tmp_path / "scored.json"
        input_file.write_text(json.dumps([sample]))

        config = FilterConfig(value_min=5.0, output_format="training")
        summary = run_filter(str(input_file), config=config)

        jsonl_path = Path(summary["output_jsonl"])
        with open(jsonl_path) as f:
            result = json.loads(f.readline())
        assert "data" in result
        assert "id" in result

    def test_scored_format_keeps_all_data(self, tmp_path):
        """Default 'scored' format preserves everything."""
        sample = _full_sample(7.0, labels={"difficulty": "advanced"})
        input_file = tmp_path / "scored.json"
        input_file.write_text(json.dumps([sample]))

        summary = run_filter(str(input_file), threshold=5.0)
        out_json = Path(summary["output_json"])
        with open(out_json) as f:
            result = json.load(f)[0]
        assert "labels" in result
        assert "value" in result


# ── find_scored_files ──

class TestFindScoredFiles:
    def test_finds_jsonl_in_directory(self, tmp_path):
        (tmp_path / "scored.jsonl").touch()
        (tmp_path / "scored_extra.json").touch()
        files = _find_scored_files(tmp_path)
        assert len(files) == 2

    def test_single_file(self, tmp_path):
        f = tmp_path / "my_data.json"
        f.touch()
        files = _find_scored_files(f)
        assert files == [f]

    def test_nonexistent_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            _find_scored_files(tmp_path / "nonexistent")

    def test_finds_scored_files_deep_levels(self, tmp_path):
        deep = tmp_path / "x" / "y" / "z" / "k"
        deep.mkdir(parents=True)
        target = deep / "scored.json"
        target.write_text(json.dumps([_scored(7.0)]))
        files = _find_scored_files(tmp_path)
        assert files == [target]


# ── Source verification ──

class TestSourceVerification:
    def test_verify_source_in_summary(self, tmp_path):
        source_path = tmp_path / "original.json"
        source_path.touch()
        samples = [
            _full_sample(7.0, metadata={"source_file": str(source_path)}),
            _full_sample(8.0, metadata={"source_file": "/other/file.json"}),
            _full_sample(9.0),  # no source_file
        ]
        input_file = tmp_path / "scored.json"
        input_file.write_text(json.dumps(samples))

        config = FilterConfig(verify_source=str(source_path))
        summary = run_filter(str(input_file), config=config)

        # Only the matched + no-metadata samples should be retained
        assert summary["retained"] == 2  # matched + no metadata
        assert "verify_source" in summary


# ── Inherited samples ──

class TestInheritedSamples:
    def test_inherited_included_by_default(self, tmp_path):
        samples = [
            _full_sample(7.0, labels={"difficulty": "advanced", "inherited": True}),
            _full_sample(8.0, labels={"difficulty": "advanced"}),
        ]
        input_file = tmp_path / "scored.json"
        input_file.write_text(json.dumps(samples))

        config = FilterConfig(value_min=6.0)
        summary = run_filter(str(input_file), config=config)
        assert summary["retained"] == 2
        assert summary["inherited_retained"] == 1

    def test_inherited_excluded(self, tmp_path):
        samples = [
            _full_sample(7.0, labels={"difficulty": "advanced", "inherited": True}),
            _full_sample(8.0, labels={"difficulty": "advanced"}),
        ]
        input_file = tmp_path / "scored.json"
        input_file.write_text(json.dumps(samples))

        config = FilterConfig(value_min=6.0, exclude_inherited=True)
        summary = run_filter(str(input_file), config=config)
        assert summary["retained"] == 1


# ── Output path generation ──

class TestOutputPathGeneration:
    def test_auto_name_with_value_min(self, tmp_path):
        input_file = tmp_path / "scored.json"
        input_file.write_text(json.dumps([_scored(7.0)]))

        config = FilterConfig(value_min=6.0)
        summary = run_filter(str(input_file), config=config)
        assert "v6" in Path(summary["output_json"]).name

    def test_auto_name_with_difficulty(self, tmp_path):
        sample = _full_sample(7.0, labels={"difficulty": "expert"})
        input_file = tmp_path / "scored.json"
        input_file.write_text(json.dumps([sample]))

        config = FilterConfig(difficulty=["expert"])
        summary = run_filter(str(input_file), config=config)
        assert "expert" in Path(summary["output_json"]).name

    def test_training_format_outputs_jsonl(self, tmp_path):
        sample = _full_sample(7.0)
        input_file = tmp_path / "scored.json"
        input_file.write_text(json.dumps([sample]))

        config = FilterConfig(value_min=5.0, output_format="training")
        summary = run_filter(str(input_file), config=config)
        assert summary["output_jsonl"].endswith(".jsonl")
        assert summary["output_json"] is None


# ── Conversation-level filtering ──

def _mt_sample(source_id, turn_index, total_turns, score=7.0, labels=None):
    """Create a multi-turn slice sample."""
    return {
        "id": f"{source_id}_t{turn_index}",
        "conversations": [{"from": "human", "value": "q"}, {"from": "gpt", "value": "a"}],
        "metadata": {
            "source_id": source_id,
            "turn_index": turn_index,
            "total_turns": total_turns,
        },
        "labels": labels or {"difficulty": "advanced"},
        "value": {"value_score": score},
    }


class TestConversationFilter:
    def _setup_conv_data(self, tmp_path, conv_records, samples):
        """Write scored.json and conversation_scores.json."""
        input_file = tmp_path / "scored.json"
        input_file.write_text(json.dumps(samples))
        conv_path = tmp_path / "conversation_scores.json"
        conv_path.write_text(json.dumps(conv_records))
        return input_file

    def test_conv_value_min(self, tmp_path):
        samples = [
            _mt_sample("c1", 1, 2, score=5.0),
            _mt_sample("c1", 2, 2, score=6.0),
            _mt_sample("c2", 1, 2, score=8.0),
            _mt_sample("c2", 2, 2, score=9.0),
        ]
        conv_records = [
            {"conversation_id": "c1", "conv_value": 5.5, "conv_selection": 5.0, "peak_complexity": 4},
            {"conversation_id": "c2", "conv_value": 8.5, "conv_selection": 8.0, "peak_complexity": 9},
        ]
        input_file = self._setup_conv_data(tmp_path, conv_records, samples)
        config = FilterConfig(conv_value_min=7.0)
        summary = run_filter(str(input_file), config=config)
        # Only c2 slices retained
        assert summary["retained"] == 2

    def test_conv_selection_min(self, tmp_path):
        samples = [_mt_sample("c1", 1, 2), _mt_sample("c1", 2, 2)]
        conv_records = [
            {"conversation_id": "c1", "conv_value": 7.0, "conv_selection": 3.0, "peak_complexity": 5},
        ]
        input_file = self._setup_conv_data(tmp_path, conv_records, samples)
        config = FilterConfig(conv_selection_min=5.0)
        summary = run_filter(str(input_file), config=config)
        assert summary["retained"] == 0

    def test_peak_complexity_min(self, tmp_path):
        samples = [_mt_sample("c1", 1, 2), _mt_sample("c1", 2, 2)]
        conv_records = [
            {"conversation_id": "c1", "conv_value": 7.0, "conv_selection": 7.0, "peak_complexity": 3},
        ]
        input_file = self._setup_conv_data(tmp_path, conv_records, samples)
        config = FilterConfig(peak_complexity_min=5.0)
        summary = run_filter(str(input_file), config=config)
        assert summary["retained"] == 0

    def test_single_turn_unaffected(self, tmp_path):
        """Single-turn samples should not be affected by conv criteria."""
        samples = [
            _scored(8.0),  # single-turn, no source_id
            _mt_sample("c1", 1, 2, score=8.0),
            _mt_sample("c1", 2, 2, score=8.0),
        ]
        conv_records = [
            {"conversation_id": "c1", "conv_value": 3.0, "conv_selection": 3.0, "peak_complexity": 2},
        ]
        input_file = self._setup_conv_data(tmp_path, conv_records, samples)
        config = FilterConfig(conv_value_min=5.0)
        summary = run_filter(str(input_file), config=config)
        # c1 fails conv criteria, single-turn passes through normally
        assert summary["retained"] == 1

    def test_combined_conv_and_shared(self, tmp_path):
        """Conv criteria + shared criteria (difficulty) both apply."""
        samples = [
            _mt_sample("c1", 1, 2, labels={"difficulty": "advanced"}),
            _mt_sample("c1", 2, 2, labels={"difficulty": "beginner"}),
        ]
        conv_records = [
            {"conversation_id": "c1", "conv_value": 8.0, "conv_selection": 8.0, "peak_complexity": 7},
        ]
        input_file = self._setup_conv_data(tmp_path, conv_records, samples)
        config = FilterConfig(conv_value_min=5.0, difficulty=["advanced"])
        summary = run_filter(str(input_file), config=config)
        # Conv passes, but only 1 slice has difficulty=advanced
        assert summary["retained"] == 1

    def test_conv_and_sample_score_criteria_and_logic(self, tmp_path):
        """Conversation gates must be combined with sample score filters using AND."""
        samples = [
            {
                "id": "c1_t1",
                "conversations": [{"from": "human", "value": "q"}, {"from": "gpt", "value": "a"}],
                "metadata": {"source_id": "c1", "turn_index": 1, "total_turns": 3},
                "labels": {"difficulty": "advanced"},
                "value": {"value_score": 9.0, "selection_score": 9.0},
            },
            {
                "id": "c1_t2",
                "conversations": [{"from": "human", "value": "q"}, {"from": "gpt", "value": "a"}],
                "metadata": {"source_id": "c1", "turn_index": 2, "total_turns": 3},
                "labels": {"difficulty": "advanced"},
                "value": {"value_score": 7.0, "selection_score": 9.0},
            },
            {
                "id": "c1_t3",
                "conversations": [{"from": "human", "value": "q"}, {"from": "gpt", "value": "a"}],
                "metadata": {"source_id": "c1", "turn_index": 3, "total_turns": 3},
                "labels": {"difficulty": "advanced"},
                "value": {"value_score": 9.0, "selection_score": 7.0},
            },
        ]
        conv_records = [
            {"conversation_id": "c1", "conv_value": 8.5, "conv_selection": 8.5, "peak_complexity": 7},
        ]
        input_file = self._setup_conv_data(tmp_path, conv_records, samples)
        config = FilterConfig(
            conv_value_min=8.0,
            value_min=8.0,
            selection_min=8.0,
        )
        summary = run_filter(str(input_file), config=config)
        assert summary["retained"] == 1

    def test_missing_conversation_scores(self, tmp_path):
        """When conversation_scores.json is missing, multi-turn slices are dropped."""
        samples = [_mt_sample("c1", 1, 2), _mt_sample("c1", 2, 2)]
        input_file = tmp_path / "scored.json"
        input_file.write_text(json.dumps(samples))
        # No conversation_scores.json written
        config = FilterConfig(conv_value_min=5.0)
        summary = run_filter(str(input_file), config=config)
        assert summary["retained"] == 0

    def test_turn_count_min_filters_short_conversations(self, tmp_path):
        """turn_count_min should filter out conversations with fewer turns."""
        samples = [
            _mt_sample("short", 1, 2, score=8.0),
            _mt_sample("short", 2, 2, score=8.0),
            _mt_sample("long", 1, 5, score=8.0),
            _mt_sample("long", 2, 5, score=8.0),
            _mt_sample("long", 3, 5, score=8.0),
            _mt_sample("long", 4, 5, score=8.0),
            _mt_sample("long", 5, 5, score=8.0),
        ]
        conv_records = [
            {"conversation_id": "short", "conv_value": 8.0, "conv_selection": 8.0,
             "peak_complexity": 7, "turn_count": 2},
            {"conversation_id": "long", "conv_value": 8.0, "conv_selection": 8.0,
             "peak_complexity": 7, "turn_count": 5},
        ]
        input_file = self._setup_conv_data(tmp_path, conv_records, samples)
        config = FilterConfig(turn_count_min=3)
        summary = run_filter(str(input_file), config=config)
        # Only "long" conversation (5 slices) should be retained
        assert summary["retained"] == 5

    def test_turn_count_max_filters_long_conversations(self, tmp_path):
        """turn_count_max should filter out conversations with too many turns."""
        samples = [
            _mt_sample("short", 1, 2, score=8.0),
            _mt_sample("short", 2, 2, score=8.0),
            _mt_sample("long", 1, 10, score=8.0),
            _mt_sample("long", 2, 10, score=8.0),
        ]
        conv_records = [
            {"conversation_id": "short", "conv_value": 8.0, "conv_selection": 8.0,
             "peak_complexity": 7, "turn_count": 2},
            {"conversation_id": "long", "conv_value": 8.0, "conv_selection": 8.0,
             "peak_complexity": 7, "turn_count": 10},
        ]
        input_file = self._setup_conv_data(tmp_path, conv_records, samples)
        config = FilterConfig(turn_count_max=5)
        summary = run_filter(str(input_file), config=config)
        # Only "short" conversation (2 slices) should be retained
        assert summary["retained"] == 2


# ── Turn-level pruning ──

def _mt_sample_scored(source_id, turn_index, total_turns, score=7.0,
                      quality_overall=7.0, labels=None):
    """Create a multi-turn slice with value_score and quality.overall."""
    return {
        "id": f"{source_id}_t{turn_index}",
        "conversations": [{"from": "human", "value": "q"}, {"from": "gpt", "value": "a"}],
        "metadata": {
            "source_id": source_id,
            "turn_index": turn_index,
            "total_turns": total_turns,
        },
        "labels": labels or {"difficulty": "advanced"},
        "value": {
            "value_score": score,
            "quality": {"overall": quality_overall},
        },
    }


class TestTurnLevelPruning:
    def _setup(self, tmp_path, samples, conv_records=None):
        input_file = tmp_path / "scored.json"
        input_file.write_text(json.dumps(samples))
        if conv_records:
            conv_path = tmp_path / "conversation_scores.json"
            conv_path.write_text(json.dumps(conv_records))
        return input_file

    def test_turn_value_min_prunes_low_slices(self, tmp_path):
        """Low-value slices should be pruned from conversations."""
        samples = [
            _mt_sample_scored("c1", 1, 4, score=8.0),
            _mt_sample_scored("c1", 2, 4, score=3.0),  # low
            _mt_sample_scored("c1", 3, 4, score=2.0),  # low
            _mt_sample_scored("c1", 4, 4, score=9.0),
        ]
        conv_records = [
            {"conversation_id": "c1", "conv_value": 8.0, "conv_selection": 8.0,
             "peak_complexity": 7, "turn_count": 4},
        ]
        input_file = self._setup(tmp_path, samples, conv_records)
        config = FilterConfig(conv_value_min=5.0, turn_value_min=5.0)
        summary = run_filter(str(input_file), config=config)
        # Slices 2 and 3 are below turn_value_min=5, but max_pruned_ratio=0.5
        # means we can prune at most 2 of 4 = 2 slices. Both are low, both pruned.
        assert summary["retained"] == 2
        assert summary["turn_pruned"] == 2

    def test_keep_first_last_protects_endpoints(self, tmp_path):
        """First and last slices should be kept even if below threshold."""
        samples = [
            _mt_sample_scored("c1", 1, 3, score=2.0),  # first, protected
            _mt_sample_scored("c1", 2, 3, score=2.0),  # middle, prunable
            _mt_sample_scored("c1", 3, 3, score=2.0),  # last, protected
        ]
        conv_records = [
            {"conversation_id": "c1", "conv_value": 8.0, "conv_selection": 8.0,
             "peak_complexity": 7, "turn_count": 3},
        ]
        input_file = self._setup(tmp_path, samples, conv_records)
        config = FilterConfig(conv_value_min=5.0, turn_value_min=5.0)
        summary = run_filter(str(input_file), config=config)
        # Only middle slice pruned; first and last protected
        assert summary["retained"] == 2
        assert summary["turn_pruned"] == 1

    def test_max_pruned_ratio_limits_pruning(self, tmp_path):
        """Should not prune more than max_pruned_ratio of turns."""
        samples = [
            _mt_sample_scored("c1", 1, 5, score=9.0),   # first, protected
            _mt_sample_scored("c1", 2, 5, score=1.0),   # low
            _mt_sample_scored("c1", 3, 5, score=1.0),   # low
            _mt_sample_scored("c1", 4, 5, score=1.0),   # low
            _mt_sample_scored("c1", 5, 5, score=9.0),   # last, protected
        ]
        conv_records = [
            {"conversation_id": "c1", "conv_value": 8.0, "conv_selection": 8.0,
             "peak_complexity": 7, "turn_count": 5},
        ]
        input_file = self._setup(tmp_path, samples, conv_records)
        # max_pruned_ratio=0.3 → max 1 prune out of 5
        config = FilterConfig(conv_value_min=5.0, turn_value_min=5.0, max_pruned_ratio=0.3)
        summary = run_filter(str(input_file), config=config)
        # 3 candidates for pruning but only 1 allowed
        assert summary["retained"] == 4
        assert summary["turn_pruned"] == 1

    def test_rescued_slices_are_highest_scoring(self, tmp_path):
        """When max_pruned_ratio limits pruning, highest-scoring candidates are rescued."""
        samples = [
            _mt_sample_scored("c1", 1, 5, score=9.0),   # first, protected
            _mt_sample_scored("c1", 2, 5, score=4.0),   # low, but highest among low
            _mt_sample_scored("c1", 3, 5, score=2.0),   # low
            _mt_sample_scored("c1", 4, 5, score=1.0),   # lowest
            _mt_sample_scored("c1", 5, 5, score=9.0),   # last, protected
        ]
        conv_records = [
            {"conversation_id": "c1", "conv_value": 8.0, "conv_selection": 8.0,
             "peak_complexity": 7, "turn_count": 5},
        ]
        input_file = self._setup(tmp_path, samples, conv_records)
        # max_pruned_ratio=0.3 → max 1 prune. 3 candidates, rescue highest 2.
        config = FilterConfig(conv_value_min=5.0, turn_value_min=5.0, max_pruned_ratio=0.3)
        summary = run_filter(str(input_file), config=config)
        # Should prune the lowest-scoring (score=1.0, turn 4)
        assert summary["turn_pruned"] == 1
        retained_ids = {s["id"] for s in _load_retained(tmp_path)}
        assert "c1_t4" not in retained_ids  # lowest pruned
        assert "c1_t2" in retained_ids  # highest low-scorer rescued

    def test_turn_pruning_without_conv_criteria(self, tmp_path):
        """Turn criteria without conv criteria should apply to all samples."""
        samples = [
            _mt_sample_scored("c1", 1, 3, score=8.0),
            _mt_sample_scored("c1", 2, 3, score=2.0),  # low
            _mt_sample_scored("c1", 3, 3, score=8.0),
            _scored(3.0),  # single-turn, below turn threshold
            _scored(7.0),  # single-turn, above turn threshold
        ]
        input_file = self._setup(tmp_path, samples)
        config = FilterConfig(turn_value_min=5.0)
        summary = run_filter(str(input_file), config=config)
        # Single-turn: score=3.0 pruned, score=7.0 kept
        # Multi-turn: turn 2 (score=2.0) pruned, turns 1 and 3 kept
        assert summary["retained"] == 3

    def test_single_turn_unaffected_by_keep_first_last(self, tmp_path):
        """Single-turn samples should not be affected by turn criteria's keep_first_last."""
        samples = [
            _scored(3.0),  # below, no turn protection
            _scored(8.0),  # above
        ]
        input_file = self._setup(tmp_path, samples)
        config = FilterConfig(turn_value_min=5.0, keep_first_last=True)
        summary = run_filter(str(input_file), config=config)
        assert summary["retained"] == 1


class TestPreserveStructure:
    def test_preserve_structure_scored_format(self, tmp_path):
        root_samples = [_full_sample(5.0), _full_sample(8.0)]
        sub_samples = [_full_sample(7.0), _full_sample(3.0)]

        (tmp_path / "scored.json").write_text(json.dumps(root_samples))
        sub = tmp_path / "nested" / "leaf"
        sub.mkdir(parents=True)
        with open(sub / "scored_extra.jsonl", "w", encoding="utf-8") as f:
            for sample in sub_samples:
                f.write(json.dumps(sample) + "\n")

        out_root = tmp_path / "filtered_struct"
        config = FilterConfig(value_min=6.0, preserve_structure=True, output_format="scored")
        summary = run_filter(str(tmp_path), output_path=str(out_root), config=config)

        assert summary["files_written"] == 2
        assert summary["output_root"] == str(out_root)

        out_root_json = out_root / "scored.json"
        out_nested_jsonl = out_root / "nested" / "leaf" / "scored_extra.jsonl"
        assert out_root_json.exists()
        assert out_nested_jsonl.exists()

        with open(out_root_json, encoding="utf-8") as f:
            root_retained = json.load(f)
        assert len(root_retained) == 1

        with open(out_nested_jsonl, encoding="utf-8") as f:
            nested_retained = [json.loads(line) for line in f if line.strip()]
        assert len(nested_retained) == 1

    def test_preserve_structure_training_format(self, tmp_path):
        root_samples = [_full_sample(5.0), _full_sample(9.0)]
        sub_samples = [_full_sample(7.0), _full_sample(4.0)]

        (tmp_path / "scored.json").write_text(json.dumps(root_samples))
        sub = tmp_path / "nested"
        sub.mkdir(parents=True)
        (sub / "scored_more.json").write_text(json.dumps(sub_samples))

        out_root = tmp_path / "filtered_training_struct"
        config = FilterConfig(value_min=6.0, preserve_structure=True, output_format="training")
        summary = run_filter(str(tmp_path), output_path=str(out_root), config=config)

        assert summary["files_written"] == 2

        out_root_jsonl = out_root / "scored.jsonl"
        out_sub_jsonl = out_root / "nested" / "scored_more.jsonl"
        assert out_root_jsonl.exists()
        assert out_sub_jsonl.exists()

        with open(out_root_jsonl, encoding="utf-8") as f:
            record = json.loads(f.readline())
        assert "id" in record
        assert "conversations" in record
        assert "labels" not in record


def _load_retained(tmp_path):
    """Load retained samples from the filter output."""
    for p in tmp_path.iterdir():
        if p.name.startswith("filtered-") and p.suffix == ".json":
            with open(p) as f:
                return json.load(f)
    return []
