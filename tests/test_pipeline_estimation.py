import json
import random
from pathlib import Path

import pytest

from sft_label.config import PipelineConfig
from sft_label.pipeline import (
    RuntimeEtaEstimator,
    _write_checkpoint,
    discover_input_files,
    estimate_directory_workload,
)


def _write_samples(path: Path, count: int):
    samples = []
    for i in range(count):
        samples.append(
            {
                "id": f"sample-{i}",
                "conversations": [
                    {"from": "human", "value": "Write a tiny Python function."},
                    {"from": "gpt", "value": "```python\ndef f():\n    return 1\n```"},
                ],
            }
        )
    with open(path, "w", encoding="utf-8") as f:
        json.dump(samples, f, ensure_ascii=False)


def test_estimate_directory_workload_basics(tmp_path):
    _write_samples(tmp_path / "a.json", 2)
    _write_samples(tmp_path / "b.json", 3)

    files = [(a, r) for a, r in discover_input_files(tmp_path) if r is not None]
    est = estimate_directory_workload(
        files,
        config=PipelineConfig(),
        enable_arbitration=False,
    )

    assert est.files_planned == 2
    assert est.total_raw_conversations == 5
    assert est.total_samples == 5
    assert est.total_labeled_samples == 5
    assert est.total_inherited_samples == 0
    assert est.baseline_total_llm_calls == 10
    assert est.initial_estimated_llm_calls == 10


def test_runtime_eta_estimator_adapts_upward():
    tracker = RuntimeEtaEstimator(total_labeled_samples=50, initial_estimated_calls=100)

    for _ in range(12):
        tracker.update(4)

    assert tracker.calls_done == 48
    assert tracker.samples_done == 12
    assert tracker.avg_calls_per_sample > 3.9
    assert tracker.estimated_total_calls > 100
    assert tracker.eta_seconds() is not None


def test_estimate_directory_workload_preserves_rng_state(tmp_path):
    _write_samples(tmp_path / "c.json", 10)
    files = [(a, r) for a, r in discover_input_files(tmp_path) if r is not None]

    state = random.getstate()
    expected_next = random.random()
    random.setstate(state)

    estimate_directory_workload(
        files,
        config=PipelineConfig(),
        shuffle=True,
    )
    actual_next = random.random()

    assert actual_next == expected_next


def test_discover_input_files_ignores_macos_sidecars(tmp_path):
    _write_samples(tmp_path / "a.json", 1)
    (tmp_path / "._a.jsonl").write_bytes(b"\x00\x01")
    (tmp_path / ".DS_Store").write_text("ignored", encoding="utf-8")

    files = [(a, r) for a, r in discover_input_files(tmp_path) if r is not None]

    assert [path.name for path, _ in files] == ["a.json"]


def test_write_checkpoint_keeps_existing_json_when_dump_fails(tmp_path, monkeypatch):
    checkpoint_path = tmp_path / "checkpoint.json"
    original = {
        "status": "in_progress",
        "completed": ["a.jsonl"],
        "failed": {},
        "total_files": 2,
    }
    checkpoint_path.write_text(json.dumps(original, ensure_ascii=False, indent=2), encoding="utf-8")

    def _broken_dump(payload, fp, *, ensure_ascii=False, indent=2):
        fp.write("{")
        raise RuntimeError("simulated write interruption")

    monkeypatch.setattr("sft_label.pipeline.json.dump", _broken_dump)

    with pytest.raises(RuntimeError, match="simulated write interruption"):
        _write_checkpoint(
            checkpoint_path,
            {
                "status": "done",
                "completed": ["a.jsonl", "b.jsonl"],
                "failed": {},
                "total_files": 2,
            },
        )

    assert json.loads(checkpoint_path.read_text(encoding="utf-8")) == original
