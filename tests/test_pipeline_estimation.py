import json
import random
from pathlib import Path

import pytest

from sft_label import pipeline as pipeline_module
from sft_label.config import PipelineConfig
from sft_label.pipeline import (
    RuntimeEtaEstimator,
    StatsAccumulator,
    _build_http_client_limits,
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


def test_runtime_eta_estimator_blends_recent_and_average_rate(monkeypatch):
    now = 1000.0
    monkeypatch.setattr("sft_label.pipeline.time.time", lambda: now)

    tracker = RuntimeEtaEstimator(total_labeled_samples=200, initial_estimated_calls=200)
    tracker.calls_done = 40
    tracker.record_call_delta(40)

    now += 10.0
    tracker.calls_done = 50
    tracker.record_call_delta(10)
    assert tracker.calls_per_sec == pytest.approx(5.0)

    now += 50.0
    tracker.calls_done = 60
    tracker.record_call_delta(10)

    assert tracker.calls_per_sec == pytest.approx(0.8333333333, rel=1e-3)
    info = tracker.info_line()
    assert "eta_rate 0.8/s" in info
    assert "recent 0.3/s" in info
    assert "avg 1.0/s" in info


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


def test_estimate_directory_workload_skips_planner_annotation_when_planner_disabled(tmp_path, monkeypatch):
    path = tmp_path / "multi.jsonl"
    row = {
        "id": "conv-1",
        "conversations": [
            {"from": "human", "value": "first request"},
            {"from": "gpt", "value": "first reply"},
            {"from": "human", "value": "second request"},
            {"from": "gpt", "value": "second reply"},
        ],
    }
    path.write_text(json.dumps(row, ensure_ascii=False) + "\n", encoding="utf-8")
    files = [(a, r) for a, r in discover_input_files(tmp_path) if r is not None]

    def _unexpected_planner_call(_samples, **_kwargs):
        raise AssertionError("planner annotation should be skipped during estimation")

    monkeypatch.setattr(
        "sft_label.preprocessing.annotate_multiturn_planner_metadata",
        _unexpected_planner_call,
    )

    est = estimate_directory_workload(
        files,
        config=PipelineConfig(),
        enable_arbitration=False,
    )

    assert est.total_raw_conversations == 1
    assert est.total_samples == 2


def test_resolve_workload_estimation_workers_targets_80_percent_cpu(monkeypatch):
    monkeypatch.setattr("sft_label.pipeline.os.cpu_count", lambda: 10)

    assert pipeline_module._resolve_workload_estimation_workers(20) == 8
    assert pipeline_module._resolve_workload_estimation_workers(3) == 3


def test_estimate_directory_workload_parallelizes_multiple_files(monkeypatch):
    files = [
        (Path("a.json"), Path("a.json")),
        (Path("b.json"), Path("b.json")),
    ]
    seen = {"workers": [], "items": []}

    monkeypatch.setattr("sft_label.pipeline._resolve_workload_estimation_workers", lambda file_count: 2)

    def fake_estimate_file(item):
        seen["items"].append(item["rel_path"])
        return {
            "total_raw_conversations": 1,
            "total_samples": 2,
            "total_labeled_samples": 2,
            "total_inherited_samples": 0,
        }

    class FakeProcessPoolExecutor:
        def __init__(self, max_workers):
            seen["workers"].append(max_workers)

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def map(self, fn, items):
            return [fn(item) for item in items]

    monkeypatch.setattr("sft_label.pipeline._estimate_directory_workload_for_file", fake_estimate_file)
    monkeypatch.setattr("sft_label.pipeline.ProcessPoolExecutor", FakeProcessPoolExecutor)

    est = estimate_directory_workload(
        files,
        config=PipelineConfig(),
        enable_arbitration=False,
        parallelize=True,
    )

    assert seen["workers"] == [2]
    assert seen["items"] == ["a.json", "b.json"]
    assert est.files_planned == 2
    assert est.total_raw_conversations == 2
    assert est.total_samples == 4
    assert est.total_labeled_samples == 4


def test_estimate_directory_workload_skips_process_pool_from_stdin(monkeypatch):
    files = [
        (Path("a.json"), Path("a.json")),
        (Path("b.json"), Path("b.json")),
    ]

    monkeypatch.setattr(
        "sft_label.pipeline._estimate_directory_workload_for_file",
        lambda item: {
            "total_raw_conversations": 1,
            "total_samples": 1,
            "total_labeled_samples": 1,
            "total_inherited_samples": 0,
        },
    )
    monkeypatch.setattr("sft_label.pipeline.os.cpu_count", lambda: 10)
    monkeypatch.setattr(pipeline_module.sys.modules["__main__"], "__file__", "<stdin>", raising=False)

    class UnexpectedProcessPoolExecutor:
        def __init__(self, *args, **kwargs):
            raise AssertionError("stdin execution should fall back to serial estimation")

    monkeypatch.setattr("sft_label.pipeline.ProcessPoolExecutor", UnexpectedProcessPoolExecutor)

    est = estimate_directory_workload(
        files,
        config=PipelineConfig(),
        enable_arbitration=False,
        parallelize=True,
    )

    assert est.files_planned == 2
    assert est.total_samples == 2


def test_estimate_directory_workload_defaults_to_serial_for_library_calls(monkeypatch):
    files = [
        (Path("a.json"), Path("a.json")),
        (Path("b.json"), Path("b.json")),
    ]

    monkeypatch.setattr(
        "sft_label.pipeline._estimate_directory_workload_for_file",
        lambda item: {
            "total_raw_conversations": 1,
            "total_samples": 1,
            "total_labeled_samples": 1,
            "total_inherited_samples": 0,
        },
    )

    class UnexpectedProcessPoolExecutor:
        def __init__(self, *args, **kwargs):
            raise AssertionError("library callers should stay serial unless they opt in")

    monkeypatch.setattr("sft_label.pipeline.ProcessPoolExecutor", UnexpectedProcessPoolExecutor)

    est = estimate_directory_workload(
        files,
        config=PipelineConfig(),
        enable_arbitration=False,
    )

    assert est.files_planned == 2
    assert est.total_samples == 2


def test_estimate_directory_workload_keeps_migrate_index_serial(monkeypatch):
    files = [
        (Path("a.json"), Path("a.json")),
        (Path("b.json"), Path("b.json")),
    ]

    monkeypatch.setattr("sft_label.pipeline.os.cpu_count", lambda: 10)
    monkeypatch.setattr(
        "sft_label.pipeline._estimate_directory_workload_for_file",
        lambda item: {
            "total_raw_conversations": 1,
            "total_samples": 1,
            "total_labeled_samples": 1,
            "total_inherited_samples": 0,
        },
    )

    class UnexpectedProcessPoolExecutor:
        def __init__(self, *args, **kwargs):
            raise AssertionError("migrate estimation should stay serial to avoid index fan-out")

    monkeypatch.setattr("sft_label.pipeline.ProcessPoolExecutor", UnexpectedProcessPoolExecutor)

    est = estimate_directory_workload(
        files,
        config=PipelineConfig(),
        enable_arbitration=False,
        parallelize=True,
        migration_index={"row-1": {"match": True}},
    )

    assert est.files_planned == 2
    assert est.total_samples == 2


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


def _stats_labels() -> dict:
    return {
        "intent": "build",
        "difficulty": "intermediate",
        "context": "snippet",
        "language": ["python"],
        "domain": ["web-backend"],
        "task": ["feature-implementation"],
        "concept": ["algorithms"],
        "agentic": [],
        "constraint": [],
        "confidence": {"intent": 0.8},
        "unmapped": [],
    }


def _stats_monitor(call1_bytes: int, call2_bytes: int) -> dict:
    return {
        "sample_id": "sample-1",
        "status": "success",
        "llm_calls": 2,
        "total_prompt_tokens": 10,
        "total_completion_tokens": 6,
        "validation_issues": [],
        "consistency_warnings": [],
        "low_confidence_dims": [],
        "arbitrated": False,
        "sample_attempt": 0,
        "elapsed_seconds": 0.1,
        "assembled_prompt_bytes": call1_bytes,
        "assembled_prompt_bytes_call2": call2_bytes,
    }


def test_stats_accumulator_uses_online_numeric_summaries_for_chunked_metrics():
    acc = StatsAccumulator(compact_labeling_request_bytes=120)

    for idx, (call1_bytes, call2_bytes) in enumerate(((80, 90), (130, 100), (110, 150)), start=1):
        acc.update(
            [_stats_labels()],
            [_stats_monitor(call1_bytes, call2_bytes)],
            samples=[{
                "metadata": {
                    "source_id": "conv-1",
                    "total_turns": 3,
                    "planner_policy": "compact_semantic_anchor_v1",
                    "segment_id": 1 if idx < 3 else 2,
                    "boundary_score": float(idx),
                    "anchor_priority": "high",
                    "anchor_reason": "semantic",
                }
            }],
        )

    assert not isinstance(acc.prompt_bytes_call1, list)
    assert not isinstance(acc.prompt_bytes_call2, list)
    assert not isinstance(acc.planner_segment_counts, list)
    assert not isinstance(acc.planner_boundary_scores, list)

    stats = acc.finalize()
    compact = stats["compact_prompt_bytes"]
    assert compact["call1"]["count"] == 3
    assert compact["call1"]["hard_cap_hits"] == 1
    assert compact["call2"]["hard_cap_hits"] == 1
    assert stats["planner_stats"]["segments_per_conversation"]["count"] >= 1
    assert stats["planner_stats"]["boundary_score"]["count"] == 3


def test_build_http_client_limits_includes_chunked_output_fd_budget(monkeypatch):
    captured = {}

    def fake_resolve_httpx_connection_limits(*, requested_concurrency, extra_connections):
        captured["requested_concurrency"] = requested_concurrency
        captured["extra_connections"] = extra_connections
        return 64, 63, True

    monkeypatch.setattr(
        "sft_label.pipeline.resolve_httpx_connection_limits",
        fake_resolve_httpx_connection_limits,
    )

    _build_http_client_limits(50, chunked_output_files=3)

    assert captured["requested_concurrency"] == 50
    assert captured["extra_connections"] > 10
