import json
from pathlib import Path
from unittest.mock import patch

from sft_label.artifacts import PASS2_STATS_FILE
from sft_label.config import PipelineConfig
from sft_label.scoring import (
    _rewrite_directory_global_selection,
    compute_value_stats,
    estimate_scoring_directory_workload,
)


def _write_labeled_json(path: Path, count: int):
    samples = [{"id": f"s-{i}", "conversations": [], "labels": {}} for i in range(count)]
    with open(path, "w", encoding="utf-8") as f:
        json.dump(samples, f, ensure_ascii=False)


def _write_labeled_jsonl(path: Path, count: int):
    with open(path, "w", encoding="utf-8") as f:
        for i in range(count):
            sample = {"id": f"j-{i}", "conversations": [], "labels": {}}
            f.write(json.dumps(sample, ensure_ascii=False) + "\n")


def test_estimate_scoring_directory_workload_with_limit(tmp_path):
    f1 = tmp_path / "labeled_a.json"
    f2 = tmp_path / "labeled_b.jsonl"
    _write_labeled_json(f1, 3)
    _write_labeled_jsonl(f2, 4)

    est = estimate_scoring_directory_workload(
        [f1, f2],
        limit=2,
        config=PipelineConfig(sample_max_retries=1),
    )

    assert est.files_planned == 2
    assert est.total_samples == 4
    assert est.baseline_total_llm_calls == 4
    assert est.initial_estimated_llm_calls == 4


def test_estimate_scoring_directory_workload_retry_factor(tmp_path):
    f1 = tmp_path / "labeled_only.json"
    _write_labeled_json(f1, 10)

    est = estimate_scoring_directory_workload(
        [f1],
        config=PipelineConfig(sample_max_retries=5),
    )

    assert est.files_planned == 1
    assert est.total_samples == 10
    assert est.baseline_total_llm_calls == 10
    assert est.initial_estimated_llm_calls == 12


def test_compute_value_stats_can_skip_raw_score_arrays():
    scored = [{
        "labels": {"intent": "build"},
        "value": {
            "complexity": {"overall": 5},
            "quality": {"overall": 7},
            "reasoning": {"overall": 6},
            "rarity": {"score": 4.5},
            "flags": [],
            "thinking_mode": "fast",
            "value_score": 6.2,
            "selection_score": 6.0,
            "intra_class_rank": 5.5,
            "confidence": 0.8,
        },
    }]
    monitors = [{
        "status": "success",
        "llm_calls": 1,
        "prompt_tokens": 12,
        "completion_tokens": 8,
    }]

    stats = compute_value_stats(scored, monitors, include_raw_scores=False)

    assert "_raw_scores" not in stats
    assert stats["total_scored"] == 1


def test_directory_global_rewrite_execution_helper_keeps_completed_runs_global():
    from sft_label.scoring_large_run import should_run_directory_global_selection_rewrite

    assert should_run_directory_global_selection_rewrite({"defer": True}) is True
    assert should_run_directory_global_selection_rewrite({"defer": False}) is True


def test_directory_global_rewrite_streams_jsonl_even_when_scored_json_exists(tmp_path):
    input_dir = tmp_path / "inputs"
    output_dir = tmp_path / "run"
    batch_input = input_dir / "batch" / "labeled.jsonl"
    batch_output = output_dir / "batch"
    batch_input.parent.mkdir(parents=True)
    batch_output.mkdir(parents=True)
    batch_input.write_text("", encoding="utf-8")

    def _sample(sample_id: str, score: float) -> dict:
        return {
            "id": sample_id,
            "conversations": [
                {"from": "human", "value": "Q"},
                {"from": "gpt", "value": "A"},
            ],
            "labels": {"intent": "build", "difficulty": "intermediate", "language": ["python"]},
            "metadata": {"total_turns": 2},
            "value": {
                "complexity": {"overall": score},
                "quality": {"overall": score},
                "reasoning": {"overall": score},
                "rarity": {"score": 5.0},
                "value_score": score,
                "confidence": 0.9,
            },
        }

    samples = [_sample("row-1", 6.0), _sample("row-2", 8.0)]
    (batch_output / "scored.jsonl").write_text(
        "".join(json.dumps(sample, ensure_ascii=False) + "\n" for sample in samples),
        encoding="utf-8",
    )
    (batch_output / "scored.json").write_text(
        json.dumps(samples, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (batch_output / "monitor_value.jsonl").write_text(
        "".join(
            json.dumps({"sample_id": sample["id"], "status": "success", "llm_calls": 1}, ensure_ascii=False) + "\n"
            for sample in samples
        ),
        encoding="utf-8",
    )
    (batch_output / PASS2_STATS_FILE).write_text(
        json.dumps({"input_file": str(batch_input)}, ensure_ascii=False),
        encoding="utf-8",
    )

    with patch(
        "sft_label.scoring._load_scored_samples",
        side_effect=AssertionError("jsonl rewrite should not fall back to full-file load"),
    ):
        _rewrite_directory_global_selection(
            output_dir=output_dir,
            input_dir=input_dir,
            config=PipelineConfig(),
            pprint=lambda *_args, **_kwargs: None,
            generate_dashboard=False,
        )

    rewritten_first = json.loads((batch_output / "scored.jsonl").read_text(encoding="utf-8").splitlines()[0])
    assert rewritten_first["value"]["selection_score"] is not None
