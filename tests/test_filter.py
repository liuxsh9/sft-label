"""Tests for the value filter tool."""

import json
import pytest
from pathlib import Path

from sft_label.tools.filter_value import filter_samples, run_filter


# ── Helpers ──

def _scored(score):
    """Create a sample with a given value_score."""
    return {"id": f"s-{score}", "value": {"value_score": score}}


def _unscored():
    """Create a sample without value data."""
    return {"id": "unscored-1", "messages": []}


# ── filter_samples ──

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
