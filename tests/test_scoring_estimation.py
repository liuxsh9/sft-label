import json
from pathlib import Path

from sft_label.config import PipelineConfig
from sft_label.scoring import estimate_scoring_directory_workload


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
    assert est.initial_estimated_llm_calls == 5


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
    assert est.initial_estimated_llm_calls == 13
