"""Tests for offline stats recomputation and dashboard regeneration.

Covers:
  - Loading samples from JSON/JSONL
  - Discovering labeled/scored files in directory trees
  - Recomputing Pass 1 stats from labeled output
  - Recomputing Pass 2 stats from scored output
  - Conversation aggregation recomputation
  - Directory-mode recomputation with merging
  - Dashboard regeneration
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from sft_label.artifacts import PASS1_CONVERSATION_STATS_FILE, PASS1_STATS_FILE, DASHBOARDS_DIRNAME
from sft_label.inline_pass1 import merge_pass1_results
from sft_label.inline_rows import build_row_sample_bundle, flatten_row_sample_bundles
from sft_label.tools.recompute import (
    load_samples,
    discover_labeled_files,
    discover_scored_files,
    recompute_stats_from_labeled,
    recompute_value_stats_from_scored,
    run_recompute,
    run_refresh_rarity,
    run_regenerate_dashboard,
    _build_inherit_map,
    _synthesize_monitor,
    _dedup_json_jsonl,
    _regenerate_for_dir,
)


# ─── Fixtures ────────────────────────────────────────────

def _make_labeled_sample(sample_id, intent="build", difficulty="intermediate",
                          languages=None, inherited=False):
    """Create a minimal labeled sample for testing."""
    labels = {
        "intent": intent,
        "difficulty": difficulty,
        "language": languages or ["python"],
        "domain": ["web-backend"],
        "concept": ["api-design"],
        "task": ["feature-implementation"],
        "agentic": [],
        "constraint": [],
        "context": "greenfield",
        "confidence": {
            "intent": 0.95, "language": 0.99, "domain": 0.85,
            "task": 0.90, "difficulty": 0.80, "concept": 0.75,
            "agentic": 0.90, "constraint": 0.90, "context": 0.70,
        },
        "unmapped": [],
    }
    if inherited:
        labels["inherited"] = True
        labels["inherited_from"] = f"{sample_id}-source"

    return {
        "id": sample_id,
        "conversations": [
            {"from": "human", "value": "Hello"},
            {"from": "gpt", "value": "World"},
        ],
        "labels": labels,
        "labeling_monitor": {
            "llm_calls": 2,
            "arbitrated": False,
            "validation_issues": [],
            "consistency_warnings": [],
        },
    }


def _make_scored_sample(sample_id, value_score=7.0, quality=7, complexity=6,
                         reasoning=7, intent="build", difficulty="intermediate"):
    """Create a minimal scored sample for testing."""
    sample = _make_labeled_sample(sample_id, intent=intent, difficulty=difficulty)
    sample["value"] = {
        "value_score": value_score,
        "complexity": {"overall": complexity},
        "quality": {"overall": quality},
        "reasoning": {"overall": reasoning},
        "rarity": {"score": 5.0},
        "thinking_mode": "slow",
        "flags": [],
        "confidence": 0.85,
    }
    return sample


def _make_multiturn_scored_samples(source_id, n_turns=3):
    """Create a set of multi-turn slices for conversation testing."""
    samples = []
    for i in range(n_turns):
        s = _make_scored_sample(f"{source_id}-turn{i}", value_score=5.0 + i)
        s["metadata"] = {
            "source_id": source_id,
            "turn_index": i + 1,
            "total_turns": n_turns,
        }
        samples.append(s)
    return samples


def _inline_full_labels(intent: str = "build") -> dict:
    return {
        "intent": intent,
        "language": ["python"],
        "domain": ["web-backend"],
        "task": ["feature-implementation"],
        "difficulty": "intermediate",
        "concept": ["algorithms"],
        "agentic": [],
        "constraint": [],
        "context": "snippet",
        "confidence": {"intent": 0.9, "language": 0.9, "difficulty": 0.9},
        "unmapped": [],
    }


def _inline_monitor() -> dict:
    return {
        "sample_id": "sample-1",
        "status": "success",
        "llm_calls": 2,
        "total_prompt_tokens": 12,
        "total_completion_tokens": 8,
        "validation_issues": [],
        "consistency_warnings": [],
        "low_confidence_dims": [],
        "arbitrated": False,
        "sample_attempt": 0,
        "elapsed_seconds": 0.5,
    }


def _inline_value(sample_id: str, rarity_score: float = 5.0) -> dict:
    return {
        "complexity": {"instruction": 6, "analytical_depth": 6, "implementation": 6, "overall": 6},
        "quality": {"correctness": 7, "code_quality": 7, "explanation": 7, "completeness": 7, "overall": 7},
        "reasoning": {"clarity": 6, "consistency": 6, "self_correction": False, "overall": 6},
        "rarity": {"score": rarity_score},
        "flags": [],
        "thinking_mode": "fast",
        "value_score": 6.5 if sample_id.endswith("2") else 6.0,
        "selection_score": 6.2 if sample_id.endswith("2") else 5.8,
        "intra_class_rank": 6.0,
        "confidence": 0.85,
    }


def _inline_rows_for_file(
    source_file: Path,
    rows: list[dict],
    labels_per_row: list[list[dict]],
    values_per_row: list[list[dict]] | None = None,
) -> list[dict]:
    merged_rows = []
    for row_number, (row, row_labels) in enumerate(zip(rows, labels_per_row), start=1):
        bundle = build_row_sample_bundle(row, source_file, row_number)
        samples, sample_to_bundle = flatten_row_sample_bundles([bundle])
        monitors = [_inline_monitor() if labels else None for labels in row_labels]
        merged = merge_pass1_results(
            [bundle],
            samples,
            row_labels,
            monitors,
            sample_to_bundle,
            source_file=source_file,
        )
        merged_rows.extend(merged.rows)

    if values_per_row:
        for row, row_values in zip(merged_rows, values_per_row):
            turns = row["extra_info"]["unique_info"]["data_label"]["turns"]
            for turn, value in zip(turns, row_values):
                turn["value"] = value

    return merged_rows


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


# ─── Test load_samples ───────────────────────────────────

class TestLoadSamples:
    def test_load_json(self, tmp_path):
        data = [{"id": "a"}, {"id": "b"}]
        path = tmp_path / "test.json"
        path.write_text(json.dumps(data))
        assert load_samples(path) == data

    def test_load_jsonl(self, tmp_path):
        path = tmp_path / "test.jsonl"
        path.write_text('{"id": "a"}\n{"id": "b"}\n')
        result = load_samples(path)
        assert len(result) == 2
        assert result[0]["id"] == "a"

    def test_load_jsonl_skips_blanks(self, tmp_path):
        path = tmp_path / "test.jsonl"
        path.write_text('{"id": "a"}\n\n{"id": "b"}\n\n')
        assert len(load_samples(path)) == 2

    def test_load_jsonl_error_reports_path_and_line(self, tmp_path):
        path = tmp_path / "broken.jsonl"
        path.write_text('{"id": "a"}\n{"id": }\n')
        with pytest.raises(ValueError, match=r"broken\.jsonl:2:"):
            load_samples(path)

    def test_load_non_list_json_raises(self, tmp_path):
        path = tmp_path / "test.json"
        path.write_text('{"key": "value"}')
        with pytest.raises(ValueError, match="Expected list"):
            load_samples(path)


# ─── Test file discovery ─────────────────────────────────

class TestFileDiscovery:
    def test_discover_labeled_flat(self, tmp_path):
        (tmp_path / "labeled.json").write_text("[]")
        (tmp_path / "labeled.jsonl").write_text("")
        # Should dedup to just .jsonl
        files = discover_labeled_files(tmp_path)
        assert len(files) == 1
        assert files[0].suffix == ".jsonl"

    def test_discover_labeled_nested(self, tmp_path):
        sub = tmp_path / "code" / "file1"
        sub.mkdir(parents=True)
        (sub / "labeled.json").write_text("[]")
        files = discover_labeled_files(tmp_path)
        assert len(files) == 1
        assert "file1" in str(files[0].parent)

    def test_discover_scored_dedup(self, tmp_path):
        (tmp_path / "scored.json").write_text("[]")
        (tmp_path / "scored.jsonl").write_text("")
        files = discover_scored_files(tmp_path)
        assert len(files) == 1
        assert files[0].suffix == ".jsonl"

    def test_discover_empty_dir(self, tmp_path):
        assert discover_labeled_files(tmp_path) == []
        assert discover_scored_files(tmp_path) == []

    def test_dedup_prefers_jsonl(self):
        files = [
            Path("/a/scored.jsonl"),
            Path("/a/scored.json"),
        ]
        result = _dedup_json_jsonl(files)
        assert len(result) == 1
        assert result[0].suffix == ".jsonl"


# ─── Test synthesize_monitor ─────────────────────────────

class TestSynthesizeMonitor:
    def test_success_sample(self):
        s = _make_labeled_sample("test")
        m = _synthesize_monitor(s)
        assert m["status"] == "success"
        assert m["llm_calls"] == 2
        assert m["arbitrated"] is False

    def test_failed_sample(self):
        s = {"id": "test", "labels": None}
        m = _synthesize_monitor(s)
        assert m["status"] == "failed"

    def test_low_confidence_extracted(self):
        s = _make_labeled_sample("test")
        # Set one confidence below threshold
        s["labels"]["confidence"]["context"] = 0.3
        m = _synthesize_monitor(s)
        dims = [d["dim"] for d in m["low_confidence_dims"]]
        assert "context" in dims


# ─── Test inherit_map reconstruction ─────────────────────

class TestBuildInheritMap:
    def test_no_inherited(self):
        samples = [_make_labeled_sample("a"), _make_labeled_sample("b")]
        assert _build_inherit_map(samples) == {}

    def test_inherited_detected(self):
        samples = [
            _make_labeled_sample("a"),
            _make_labeled_sample("b", inherited=True),
            _make_labeled_sample("c"),
        ]
        inherit_map = _build_inherit_map(samples)
        assert 1 in inherit_map
        assert 0 not in inherit_map
        assert 2 not in inherit_map


# ─── Test recompute_stats_from_labeled ───────────────────

class TestRecomputePass1Stats:
    def test_basic_stats(self):
        samples = [
            _make_labeled_sample("a", intent="build", difficulty="intermediate"),
            _make_labeled_sample("b", intent="debug", difficulty="advanced"),
            _make_labeled_sample("c", intent="build", difficulty="beginner"),
        ]
        stats = recompute_stats_from_labeled(samples)

        assert stats["total_samples"] == 3
        assert stats["success"] == 3
        assert stats["failed"] == 0
        assert stats["recomputed"] is True
        assert "recomputed_at" in stats

        # Check tag distributions
        assert stats["tag_distributions"]["intent"]["build"] == 2
        assert stats["tag_distributions"]["intent"]["debug"] == 1
        assert stats["combo_distributions"]

    def test_inherited_excluded_from_distributions(self):
        samples = [
            _make_labeled_sample("a", intent="build"),
            _make_labeled_sample("b", intent="build", inherited=True),
        ]
        stats = recompute_stats_from_labeled(samples)

        assert stats["total_samples"] == 2
        # Only the non-inherited sample should count in distributions
        assert stats["tag_distributions"]["intent"]["build"] == 1
        combo_key = next(iter(stats["combo_distributions"]))
        assert stats["combo_distributions"][combo_key] == 1
        assert stats["sparse_inherited"] == 1
        assert stats["sparse_labeled"] == 1

    def test_token_usage_zeroed(self):
        samples = [_make_labeled_sample("a")]
        stats = recompute_stats_from_labeled(samples)
        assert stats["total_prompt_tokens"] == 0
        assert stats["total_completion_tokens"] == 0
        assert stats["total_tokens"] == 0

    def test_failed_sample(self):
        samples = [
            _make_labeled_sample("a"),
            {"id": "b", "conversations": [], "labels": None},
        ]
        stats = recompute_stats_from_labeled(samples)
        assert stats["total_samples"] == 2
        assert stats["success"] == 1
        assert stats["failed"] == 1


# ─── Test recompute_value_stats_from_scored ──────────────

class TestRecomputePass2Stats:
    def test_basic_value_stats(self):
        samples = [
            _make_scored_sample("a", value_score=7.0, quality=7),
            _make_scored_sample("b", value_score=5.0, quality=5),
            _make_scored_sample("c", value_score=9.0, quality=9),
        ]
        stats = recompute_value_stats_from_scored(samples)

        assert stats["total_scored"] == 3
        assert stats["recomputed"] is True
        assert "score_distributions" in stats
        assert "value_score" in stats["score_distributions"]

    def test_token_usage_zeroed(self):
        samples = [_make_scored_sample("a")]
        stats = recompute_value_stats_from_scored(samples)
        assert stats["total_llm_calls"] == 0
        assert stats["total_prompt_tokens"] == 0

    def test_selection_scores_recomputed(self):
        # Create enough samples for selection score computation
        samples = [_make_scored_sample(f"s{i}", value_score=4.0 + i * 0.1)
                    for i in range(35)]
        stats = recompute_value_stats_from_scored(samples)
        assert stats["total_scored"] == 35


# ─── Test run_recompute (integration) ────────────────────

class TestRunRecompute:
    def test_single_labeled_file(self, tmp_path):
        samples = [_make_labeled_sample(f"s{i}") for i in range(5)]
        path = tmp_path / "labeled.json"
        path.write_text(json.dumps(samples))

        written = run_recompute(str(path), pass_num="1")
        assert "stats" in written
        # Verify stats file content
        stats = json.loads(Path(written["stats"]).read_text())
        assert stats["total_samples"] == 5
        assert stats["recomputed"] is True

    def test_single_labeled_file_writes_conversation_stats_artifact(self, tmp_path):
        samples = []
        for turn_index, intent in ((1, "build"), (2, "debug")):
            sample = _make_labeled_sample(f"conv-1-turn{turn_index}", intent=intent)
            sample["metadata"] = {
                "source_id": "conv-1",
                "source_file": "dataset/train.jsonl",
                "turn_index": turn_index,
                "total_turns": 2,
            }
            samples.append(sample)

        path = tmp_path / "labeled.json"
        path.write_text(json.dumps(samples))

        run_recompute(str(path), pass_num="1")

        conversation_stats = json.loads((tmp_path / PASS1_CONVERSATION_STATS_FILE).read_text())
        assert conversation_stats["mode_id"] == "conversation"
        assert conversation_stats["total"] == 1
        assert conversation_stats["distributions"]["intent"] == {"debug": 1}

    def test_single_scored_file(self, tmp_path):
        samples = [_make_scored_sample(f"s{i}") for i in range(5)]
        path = tmp_path / "scored.json"
        path.write_text(json.dumps(samples))

        written = run_recompute(str(path), pass_num="2")
        assert "stats_value" in written
        stats = json.loads(Path(written["stats_value"]).read_text())
        assert stats["total_scored"] == 5

    def test_both_passes_from_scored(self, tmp_path):
        samples = [_make_scored_sample(f"s{i}") for i in range(5)]
        path = tmp_path / "scored.json"
        path.write_text(json.dumps(samples))

        written = run_recompute(str(path), pass_num="both")
        assert "stats" in written
        assert "stats_value" in written

    def test_directory_mode(self, tmp_path):
        # Create two subdirectories with labeled files
        for sub in ("code/file1", "code/file2"):
            sub_path = tmp_path / sub
            sub_path.mkdir(parents=True)
            samples = [_make_labeled_sample(f"{sub}-s{i}") for i in range(3)]
            (sub_path / "labeled.json").write_text(json.dumps(samples))

        written = run_recompute(str(tmp_path), pass_num="1")
        assert "summary_stats" in written
        summary = json.loads(Path(written["summary_stats"]).read_text())
        assert summary.get("recomputed") is True

    def test_directory_pass1_recompute_works_for_scored_only_trees(self, tmp_path):
        sub_path = tmp_path / "code" / "part1"
        sub_path.mkdir(parents=True)
        samples = [_make_scored_sample(f"s{i}") for i in range(2)]
        (sub_path / "scored.json").write_text(json.dumps(samples))

        written = run_recompute(str(tmp_path), pass_num="1")

        assert "summary_stats" in written
        assert (sub_path / PASS1_STATS_FILE).exists()
        assert (sub_path / PASS1_CONVERSATION_STATS_FILE).exists()

    def test_directory_scored_summary_uses_relative_paths(self, tmp_path):
        for sub in ("code/part1", "multi_turn/part1"):
            sub_path = tmp_path / sub
            sub_path.mkdir(parents=True)
            samples = [_make_scored_sample(f"{sub}-s{i}") for i in range(2)]
            (sub_path / "scored.json").write_text(json.dumps(samples))

        written = run_recompute(str(tmp_path), pass_num="2")
        summary = json.loads(Path(written["summary_stats_value"]).read_text())
        files = {row["file"] for row in summary.get("per_file_summary", [])}
        assert files == {"code/part1/scored.json", "multi_turn/part1/scored.json"}

    def test_directory_scored_with_conversations(self, tmp_path):
        # Multi-turn samples for conversation recomputation
        samples = _make_multiturn_scored_samples("conv-1", n_turns=3)
        samples.extend(_make_multiturn_scored_samples("conv-2", n_turns=4))
        (tmp_path / "scored.json").write_text(json.dumps(samples))

        written = run_recompute(str(tmp_path), pass_num="2")
        assert "summary_stats_value" in written
        assert "conversation_scores" in written

    def test_directory_scored_writes_per_file_conversation_scores(self, tmp_path):
        sub_path = tmp_path / "multi_turn" / "part1"
        sub_path.mkdir(parents=True)
        samples = _make_multiturn_scored_samples("conv-1", n_turns=3)
        (sub_path / "scored.json").write_text(json.dumps(samples))

        written = run_recompute(str(tmp_path), pass_num="2")

        assert "conversation_scores" in written
        assert (sub_path / "conversation_scores.json").exists()

    def test_directory_scored_parallel_workers(self, tmp_path):
        for i in range(4):
            sub_path = tmp_path / "code" / f"part{i:02d}"
            sub_path.mkdir(parents=True)
            samples = [_make_scored_sample(f"s{i}-0"), _make_scored_sample(f"s{i}-1")]
            (sub_path / "scored.json").write_text(json.dumps(samples))

        written = run_recompute(str(tmp_path), pass_num="2", workers=4)
        assert "summary_stats_value" in written
        summary = json.loads(Path(written["summary_stats_value"]).read_text())
        assert len(summary.get("per_file_summary", [])) == 4

    def test_directory_scored_compact_progress_output(self, tmp_path, capsys):
        for i in range(35):
            sub = tmp_path / "code" / f"part{i:02d}"
            sub.mkdir(parents=True)
            samples = [_make_scored_sample(f"s{i}-0"), _make_scored_sample(f"s{i}-1")]
            (sub / "scored.json").write_text(json.dumps(samples))

        run_recompute(str(tmp_path), pass_num="2")
        out = capsys.readouterr().out
        assert "Pass 2 compact mode: log every" in out
        assert "Pass 2 progress" in out
        # Compact mode should avoid per-file verbose lines.
        assert "  Processing:" not in out

    def test_custom_output_dir(self, tmp_path):
        samples = [_make_labeled_sample("s0")]
        input_path = tmp_path / "labeled.json"
        input_path.write_text(json.dumps(samples))

        out_dir = tmp_path / "output"
        written = run_recompute(str(input_path), pass_num="1",
                                output_dir=str(out_dir))
        assert "stats" in written
        assert "output" in written["stats"]

    def test_nonexistent_path_raises(self):
        with pytest.raises(FileNotFoundError):
            run_recompute("/nonexistent/path")

    def test_inline_single_file_writes_meta_artifacts(self, tmp_path):
        run_root = tmp_path / "train_labeled_20260311_120000"
        source_file = run_root / "train" / "train.jsonl"
        rows = _inline_rows_for_file(
            source_file,
            [{
                "id": "conv-inline",
                "conversations": [
                    {"from": "human", "value": "q1"},
                    {"from": "gpt", "value": "a1"},
                    {"from": "human", "value": "q2"},
                    {"from": "gpt", "value": "a2"},
                ],
            }],
            [[_inline_full_labels("build"), _inline_full_labels("modify")]],
            [[_inline_value("conv-inline_t1"), _inline_value("conv-inline_t2")]],
        )
        _write_jsonl(source_file, rows)

        written = run_recompute(str(source_file), pass_num="both")
        artifact_dir = run_root / "meta_label_data" / "files" / "train"

        assert Path(written["stats"]).is_file()
        assert Path(written["stats_value"]).is_file()
        assert (artifact_dir / "conversation_scores.json").exists()
        assert json.loads(Path(written["stats_value"]).read_text())["total_scored"] == 2

    def test_inline_directory_summary_uses_mirrored_file_paths(self, tmp_path):
        run_root = tmp_path / "dataset_labeled_20260311_120000"
        file_a = run_root / "dataset" / "code" / "a.jsonl"
        file_b = run_root / "dataset" / "multi" / "b.jsonl"

        rows_a = _inline_rows_for_file(
            file_a,
            [{
                "meta_prompt": ["system"],
                "data": [
                    {"role": "user", "content": "qa"},
                    {"role": "assistant", "content": "aa"},
                ],
            }],
            [[_inline_full_labels("build")]],
            [[_inline_value("row-a_t1")]],
        )
        rows_b = _inline_rows_for_file(
            file_b,
            [{
                "id": "conv-b",
                "conversations": [
                    {"from": "human", "value": "q1"},
                    {"from": "gpt", "value": "a1"},
                    {"from": "human", "value": "q2"},
                    {"from": "gpt", "value": "a2"},
                ],
            }],
            [[_inline_full_labels("debug"), _inline_full_labels("modify")]],
            [[_inline_value("conv-b_t1"), _inline_value("conv-b_t2")]],
        )
        _write_jsonl(file_a, rows_a)
        _write_jsonl(file_b, rows_b)

        written = run_recompute(str(run_root), pass_num="2")
        summary = json.loads(Path(written["summary_stats_value"]).read_text())
        files = {row["file"] for row in summary.get("per_file_summary", [])}
        assert files == {"code/a.jsonl", "multi/b.jsonl"}

    def test_inline_directory_summary_preserves_multi_suffix_filenames(self, tmp_path):
        run_root = tmp_path / "dataset_labeled_20260311_120000"
        source_file = run_root / "dataset" / "code" / "a.b.c.jsonl"
        rows = _inline_rows_for_file(
            source_file,
            [{
                "meta_prompt": ["system"],
                "data": [
                    {"role": "user", "content": "qa"},
                    {"role": "assistant", "content": "aa"},
                ],
            }],
            [[_inline_full_labels("build")]],
            [[_inline_value("row-a_t1")]],
        )
        _write_jsonl(source_file, rows)

        written = run_recompute(str(run_root), pass_num="2")
        summary = json.loads(Path(written["summary_stats_value"]).read_text())
        files = {row["file"] for row in summary.get("per_file_summary", [])}
        assert files == {"code/a.b.c.jsonl"}

    def test_inline_recompute_rejects_custom_output_dir(self, tmp_path):
        run_root = tmp_path / "dataset_labeled_20260311_120000"
        source_file = run_root / "dataset" / "code" / "a.jsonl"
        rows = _inline_rows_for_file(
            source_file,
            [{
                "meta_prompt": ["system"],
                "data": [
                    {"role": "user", "content": "qa"},
                    {"role": "assistant", "content": "aa"},
                ],
            }],
            [[_inline_full_labels("build")]],
            [[_inline_value("row-a_t1")]],
        )
        _write_jsonl(source_file, rows)

        with pytest.raises(ValueError, match="Inline recompute writes in-place"):
            run_recompute(str(run_root), pass_num="2", output_dir=str(tmp_path / "elsewhere"))


class TestRefreshRarity:
    def test_refresh_logs_progress_for_large_file(self, tmp_path, capsys):
        from sft_label.config import PipelineConfig

        samples = [_make_scored_sample(f"s{i}") for i in range(120)]
        path = tmp_path / "scored.json"
        path.write_text(json.dumps(samples))

        run_refresh_rarity(
            str(path),
            config=PipelineConfig(rarity_score_mode="absolute"),
        )
        out = capsys.readouterr().out
        assert "Starting rarity refresh" in out
        assert "loading scored file" in out
        assert "loaded 120 sample(s)" in out
        assert "computing rarity for 120 sample(s)" in out
        assert "rarity 120/120 (100.0%)" in out
        assert "writing scored outputs" in out
        assert "Completed rarity refresh: refreshed 120 sample(s)" in out

    def test_refresh_logs_context_on_bad_input(self, tmp_path, capsys):
        path = tmp_path / "scored.json"
        path.write_text("[")

        with pytest.raises(ValueError, match=r"Invalid JSON in .*scored\.json:1:2:"):
            run_refresh_rarity(str(path))
        out = capsys.readouterr().out
        assert "failed while loading baseline source" in out
        assert "scored.json" in out

    def test_single_file_refresh_local_baseline(self, tmp_path):
        from sft_label.config import PipelineConfig

        samples = [
            _make_scored_sample("common-1", intent="build", difficulty="beginner"),
            _make_scored_sample("common-2", intent="build", difficulty="beginner"),
            _make_scored_sample("rare-1", intent="debug", difficulty="expert"),
        ]
        path = tmp_path / "scored.json"
        path.write_text(json.dumps(samples))

        written = run_refresh_rarity(
            str(path),
            config=PipelineConfig(rarity_score_mode="absolute"),
        )
        assert "scored_json" in written
        assert "stats_value" in written
        assert int(written.get("rarity_refreshed_samples", "0")) == 3

        refreshed = json.loads((tmp_path / "scored.json").read_text())
        rarity_by_id = {
            s["id"]: (s.get("value") or {}).get("rarity", {}).get("score")
            for s in refreshed
        }
        assert rarity_by_id["rare-1"] > rarity_by_id["common-1"]
        assert rarity_by_id["common-1"] == rarity_by_id["common-2"]

    def test_directory_refresh_with_external_stats(self, tmp_path):
        from sft_label.config import PipelineConfig

        for sub in ("code/part1", "multi_turn/part1"):
            sub_path = tmp_path / sub
            sub_path.mkdir(parents=True)
            samples = [
                _make_scored_sample(f"{sub}-s1", intent="build", difficulty="beginner"),
                _make_scored_sample(f"{sub}-s2", intent="debug", difficulty="expert"),
            ]
            (sub_path / "scored.json").write_text(json.dumps(samples))

        stats = {
            "total_samples": 1000,
            "tag_distributions": {
                "intent": {"build": 600, "debug": 40},
                "difficulty": {"beginner": 700, "expert": 20},
                "concept": {"api-design": 300},
            },
        }
        stats_path = tmp_path / "global_stats.json"
        stats_path.write_text(json.dumps(stats))

        written = run_refresh_rarity(
            str(tmp_path),
            tag_stats_path=str(stats_path),
            config=PipelineConfig(rarity_score_mode="absolute"),
        )
        assert "summary_stats_value" in written
        assert int(written.get("files_refreshed", "0")) == 2

        summary = json.loads(Path(written["summary_stats_value"]).read_text())
        files = {row["file"] for row in summary.get("per_file_summary", [])}
        assert files == {"code/part1/scored.json", "multi_turn/part1/scored.json"}

        for sub in ("code/part1", "multi_turn/part1"):
            refreshed = json.loads((tmp_path / sub / "scored.json").read_text())
            for sample in refreshed:
                rarity = (sample.get("value") or {}).get("rarity") or {}
                stats_ref = rarity.get("stats_ref") or {}
                assert stats_ref.get("source") == str(stats_path)

    def test_directory_refresh_parallel_workers(self, tmp_path):
        from sft_label.config import PipelineConfig

        for sub in ("code/part1", "multi_turn/part1"):
            sub_path = tmp_path / sub
            sub_path.mkdir(parents=True)
            samples = [
                _make_scored_sample(f"{sub}-s1", intent="build", difficulty="beginner"),
                _make_scored_sample(f"{sub}-s2", intent="debug", difficulty="expert"),
            ]
            (sub_path / "scored.json").write_text(json.dumps(samples))

        written = run_refresh_rarity(
            str(tmp_path),
            config=PipelineConfig(rarity_score_mode="absolute"),
            workers=4,
        )
        assert "summary_stats_value" in written
        assert int(written.get("files_refreshed", "0")) == 2

    def test_external_stats_without_combo_uses_local_combo_fallback(self, tmp_path):
        from sft_label.config import PipelineConfig

        samples = [
            _make_scored_sample("s1", intent="build", difficulty="beginner"),
            _make_scored_sample("s2", intent="debug", difficulty="expert"),
        ]
        path = tmp_path / "scored.json"
        path.write_text(json.dumps(samples), encoding="utf-8")

        stats = {
            "total_samples": 1000,
            "distribution_total_samples": 800,
            "tag_distributions": {
                "intent": {"build": 600, "debug": 50},
                "difficulty": {"beginner": 700, "expert": 30},
                "concept": {"api-design": 300},
            },
            # combo_distributions intentionally omitted
        }
        stats_path = tmp_path / "global_stats.json"
        stats_path.write_text(json.dumps(stats), encoding="utf-8")

        written = run_refresh_rarity(
            str(path),
            tag_stats_path=str(stats_path),
            config=PipelineConfig(rarity_score_mode="absolute"),
        )
        assert "stats_value" in written

        refreshed = json.loads((tmp_path / "scored.json").read_text(encoding="utf-8"))
        for sample in refreshed:
            rarity = (sample.get("value") or {}).get("rarity") or {}
            assert rarity.get("combo_rarity", 0.0) > 0.0

        stats_out = json.loads(Path(written["stats_value"]).read_text(encoding="utf-8"))
        assert stats_out["rarity_config"]["combo_mode"] == "hybrid"

    def test_inline_directory_refresh_syncs_back_to_rows(self, tmp_path):
        from sft_label.config import PipelineConfig

        run_root = tmp_path / "dataset_labeled_20260311_120000"
        source_file = run_root / "dataset" / "multi" / "b.jsonl"
        rows = _inline_rows_for_file(
            source_file,
            [{
                "id": "conv-b",
                "conversations": [
                    {"from": "human", "value": "q1"},
                    {"from": "gpt", "value": "a1"},
                    {"from": "human", "value": "q2"},
                    {"from": "gpt", "value": "a2"},
                ],
            }],
            [[_inline_full_labels("build"), _inline_full_labels("debug")]],
            [[_inline_value("conv-b_t1", rarity_score=1.0), _inline_value("conv-b_t2", rarity_score=1.0)]],
        )
        _write_jsonl(source_file, rows)

        written = run_refresh_rarity(
            str(run_root),
            config=PipelineConfig(rarity_score_mode="absolute"),
        )

        assert "summary_stats_value" in written
        updated_rows = [json.loads(line) for line in source_file.read_text(encoding="utf-8").splitlines()]
        turns = updated_rows[0]["extra_info"]["unique_info"]["data_label"]["turns"]
        assert turns[0]["value"]["rarity"]["stats_ref"]["source"]
        assert turns[1]["value"]["selection_score"] is not None
        assert (run_root / "meta_label_data" / "summary_stats_scoring.json").exists()

    def test_inline_refresh_rejects_custom_output_dir(self, tmp_path):
        from sft_label.config import PipelineConfig

        run_root = tmp_path / "dataset_labeled_20260311_120000"
        source_file = run_root / "dataset" / "code" / "a.jsonl"
        rows = _inline_rows_for_file(
            source_file,
            [{
                "meta_prompt": ["system"],
                "data": [
                    {"role": "user", "content": "qa"},
                    {"role": "assistant", "content": "aa"},
                ],
            }],
            [[_inline_full_labels("build")]],
            [[_inline_value("row-a_t1")]],
        )
        _write_jsonl(source_file, rows)

        with pytest.raises(ValueError, match="Inline rarity refresh writes in-place"):
            run_refresh_rarity(
                str(run_root),
                output_dir=str(tmp_path / "elsewhere"),
                config=PipelineConfig(rarity_score_mode="absolute"),
            )


# ─── Test run_regenerate_dashboard ───────────────────────

class TestRegenerateDashboard:
    def _setup_stats_dir(self, tmp_path):
        """Create a directory with stats files for dashboard generation."""
        samples = [_make_labeled_sample(f"s{i}") for i in range(5)]
        stats = recompute_stats_from_labeled(samples)
        (tmp_path / PASS1_STATS_FILE).write_text(json.dumps(stats))
        (tmp_path / "labeled.json").write_text(json.dumps(samples))
        return tmp_path

    def test_pass1_dashboard(self, tmp_path):
        self._setup_stats_dir(tmp_path)
        generated = run_regenerate_dashboard(str(tmp_path), pass_num="1")
        assert len(generated) >= 1
        assert any("dashboard" in str(p) for p in generated)

    def test_pass1_dashboard_logs_progress(self, tmp_path, capsys):
        self._setup_stats_dir(tmp_path)
        generated = run_regenerate_dashboard(str(tmp_path), pass_num="1")
        out = capsys.readouterr().out
        assert len(generated) >= 1
        assert "Starting dashboard regeneration" in out
        assert "Detected single-directory mode" in out
        assert f"generating Pass 1 dashboard from {PASS1_STATS_FILE}" in out
        assert "wrote Pass 1 dashboard" in out
        assert "Completed dashboard regeneration" in out

    def test_no_stats_no_dashboard(self, tmp_path):
        generated = run_regenerate_dashboard(str(tmp_path), pass_num="1")
        assert len(generated) == 0

    def test_non_directory_raises(self, tmp_path):
        f = tmp_path / "file.json"
        f.write_text("[]")
        with pytest.raises(ValueError, match="Expected a directory"):
            run_regenerate_dashboard(str(f))

    def test_batch_mode(self, tmp_path):
        """Test with subdirectories (batch mode)."""
        sub = tmp_path / "code" / "file1"
        sub.mkdir(parents=True)
        samples = [_make_labeled_sample(f"s{i}") for i in range(3)]
        stats = recompute_stats_from_labeled(samples)
        (sub / PASS1_STATS_FILE).write_text(json.dumps(stats))
        (sub / "labeled.json").write_text(json.dumps(samples))

        generated = run_regenerate_dashboard(str(tmp_path), pass_num="1")
        assert len(generated) >= 1

    def test_batch_mode_parallel_workers(self, tmp_path):
        for sub in ("code/file1", "code/file2"):
            sub_path = tmp_path / sub
            sub_path.mkdir(parents=True)
            samples = [_make_labeled_sample(f"{sub}-s{i}") for i in range(3)]
            stats = recompute_stats_from_labeled(samples)
            (sub_path / PASS1_STATS_FILE).write_text(json.dumps(stats))
            (sub_path / "labeled.json").write_text(json.dumps(samples))

        generated = run_regenerate_dashboard(str(tmp_path), pass_num="1", workers=4)
        assert len(generated) >= 2

    def test_batch_mode_discovers_deeper_output_directories(self, tmp_path):
        sub = tmp_path / "nested" / "code" / "file1"
        sub.mkdir(parents=True)
        samples = [_make_labeled_sample(f"s{i}") for i in range(3)]
        stats = recompute_stats_from_labeled(samples)
        (sub / PASS1_STATS_FILE).write_text(json.dumps(stats))
        (sub / "labeled.json").write_text(json.dumps(samples))

        generated = run_regenerate_dashboard(str(tmp_path), pass_num="1")

        assert len(generated) >= 1
        assert any("file1" in str(path) for path in generated)

    def test_regenerate_for_dir_logs_failure_context(self, tmp_path, capsys):
        self._setup_stats_dir(tmp_path)

        def _boom(*args, **kwargs):
            raise RuntimeError("boom")

        generated = _regenerate_for_dir(tmp_path, "1", _boom, None)
        out = capsys.readouterr().out
        assert generated == []
        assert "Warning: Pass 1 dashboard failed" in out
        assert "RuntimeError: boom" in out
        assert str(tmp_path) in out

    def test_inline_dashboards_land_under_meta_dashboard_dir(self, tmp_path):
        run_root = tmp_path / "dataset_labeled_20260311_120000"
        source_file = run_root / "dataset" / "multi" / "b.jsonl"
        rows = _inline_rows_for_file(
            source_file,
            [{
                "id": "conv-b",
                "conversations": [
                    {"from": "human", "value": "q1"},
                    {"from": "gpt", "value": "a1"},
                    {"from": "human", "value": "q2"},
                    {"from": "gpt", "value": "a2"},
                ],
            }],
            [[_inline_full_labels("build"), _inline_full_labels("modify")]],
            [[_inline_value("conv-b_t1"), _inline_value("conv-b_t2")]],
        )
        _write_jsonl(source_file, rows)

        run_recompute(str(run_root), pass_num="both")
        generated = run_regenerate_dashboard(str(run_root), pass_num="both")

        assert len(generated) >= 2
        expected_dir = run_root / "meta_label_data" / DASHBOARDS_DIRNAME
        assert expected_dir.is_dir()
        assert all(Path(path).parent == expected_dir for path in map(Path, generated))
        assert len(list(expected_dir.glob("dashboard_scoring*.html"))) >= 1
