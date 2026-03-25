import asyncio
from pathlib import Path
from types import SimpleNamespace

import pytest

from sft_label.config import PipelineConfig
from sft_label.pipeline import (
    _decorate_run_info_with_runtime,
    _effective_chunk_parallelism,
    _effective_chunk_row_count,
    run_directory_pipeline,
    run_one_file,
)


class _FakeProgress:
    def __init__(self):
        self.updates = []
        self.tasks = {}

    def update(self, task_id, **kwargs):
        if "total" in kwargs:
            self.tasks.setdefault(task_id, {})["total"] = kwargs["total"]
        self.updates.append((task_id, kwargs))

    @property
    def console(self):
        return self

    def print(self, *args, **kwargs):
        return None


@pytest.mark.asyncio
async def test_run_one_file_limits_inflight_pass1_tasks(monkeypatch, tmp_path):
    input_path = tmp_path / "input.json"
    input_path.write_text("[]", encoding="utf-8")
    output_dir = tmp_path / "out"

    samples = [
        {"id": f"s-{idx}", "conversations": []}
        for idx in range(6)
    ]
    monkeypatch.setattr(
        "sft_label.pipeline.iter_samples_from_file",
        lambda *args, **kwargs: (list(samples), len(samples)),
    )

    release = asyncio.Event()
    started = asyncio.Event()
    active = 0
    max_active = 0

    async def fake_label_one(*args, **kwargs):
        nonlocal active, max_active
        sample = args[1]
        sample_idx = args[3]
        active += 1
        max_active = max(max_active, active)
        started.set()
        await release.wait()
        active -= 1
        return sample_idx, {"intent": "coding"}, {
            "sample_id": sample["id"],
            "index": sample_idx,
            "llm_calls": 1,
            "total_prompt_tokens": 0,
            "total_completion_tokens": 0,
            "validation_issues": [],
            "consistency_warnings": [],
            "low_confidence_dims": [],
            "arbitrated": False,
            "sample_attempt": 0,
            "status": "success",
            "elapsed_seconds": 0.01,
        }

    monkeypatch.setattr("sft_label.pipeline.label_one", fake_label_one)

    config = PipelineConfig(
        concurrency=2,
        dir_pipeline_watermark=1.0,
        enable_adaptive_runtime=False,
    )

    task = asyncio.create_task(
        run_one_file(
            input_path=input_path,
            output_dir=output_dir,
            http_client=None,
            sem=asyncio.Semaphore(2),
            model="fake-model",
            config=config,
        )
    )

    await started.wait()
    await asyncio.sleep(0.05)
    assert max_active == 2

    release.set()
    stats = await task

    assert stats["total_samples"] == 6
    assert stats["success"] == 6


def test_effective_chunk_row_count_caps_giant_chunks_for_incremental_flush():
    assert _effective_chunk_row_count(5000, 600) == 1200
    assert _effective_chunk_row_count(64, 600) == 64


def test_directory_delegate_keeps_some_chunk_overlap_without_unbounded_fanout():
    assert _effective_chunk_parallelism(3, directory_delegate=True) == 2
    assert _effective_chunk_parallelism(1, directory_delegate=True) == 1
    assert _effective_chunk_parallelism(3, directory_delegate=False) == 3


def test_decorate_run_info_with_runtime_snapshot_appends_effective_limits():
    config = PipelineConfig(enable_adaptive_runtime=True)
    from sft_label.llm_runtime import AdaptiveLLMRuntime

    runtime = AdaptiveLLMRuntime(base_concurrency=12, base_rps=20.0)
    runtime._set_effective_limits(8, 12.0)
    runtime.state = "degraded"
    runtime.gate._in_flight = 5
    setattr(config, "_adaptive_runtime", runtime)

    info = _decorate_run_info_with_runtime("run 10/100 (10.0%) eta 10:00 rate 4.1/s", config)
    assert info.endswith("runtime degraded c=8 rps=12.0 in_flight=5")


@pytest.mark.asyncio
async def test_directory_chunked_delegate_reports_live_progress(monkeypatch, tmp_path):
    input_file = tmp_path / "sample.jsonl"
    input_file.write_text('{"id":"x"}\n', encoding="utf-8")

    async def fake_run_one_file_chunked(*args, **kwargs):
        progress_event_cb = kwargs["progress_event_cb"]
        progress_event_cb({"kind": "llm_delta", "delta_calls": 1, "run_info": "run 1/10 eta --:-- rate warming"})
        progress_event_cb({"kind": "sample_complete", "sample_idx": 0, "chunk_idx": 0})
        progress_event_cb({"kind": "llm_delta", "delta_calls": 1, "run_info": "run 2/10 eta --:-- rate warming"})
        progress_event_cb({"kind": "sample_complete", "sample_idx": 1, "chunk_idx": 0})
        return {
            "total_llm_calls": 2,
            "sparse_labeled": 2,
            "sparse_inherited": 0,
            "total_samples": 2,
            "success": 2,
        }

    monkeypatch.setattr("sft_label.pipeline._should_delegate_directory_jsonl_to_chunked", lambda *args, **kwargs: True)
    monkeypatch.setattr("sft_label.pipeline._run_one_file_chunked", fake_run_one_file_chunked)
    monkeypatch.setattr("sft_label.pipeline.update_checkpoint", lambda *args, **kwargs: None)

    progress = _FakeProgress()
    progress.tasks[1] = {"total": 2}
    progress.tasks[2] = {"total": 10}
    progress.tasks[3] = {"total": 1}

    workload_estimate = SimpleNamespace(
        total_labeled_samples=2,
        initial_estimated_llm_calls=10,
    )

    stats = await run_directory_pipeline(
        dir_files=[(input_file, Path("sample.jsonl"))],
        run_dir=tmp_path / "run",
        model="fake-model",
        concurrency=2,
        checkpoint_path=tmp_path / "checkpoint.json",
        progress=progress,
        file_task=3,
        sample_task=1,
        http_client=None,
        sem=asyncio.Semaphore(2),
        config=PipelineConfig(enable_adaptive_runtime=False),
        llm_task=2,
        workload_estimate=workload_estimate,
        llm_progress_cb=lambda delta, stage: f"run {delta and 2 or 2}/10 eta --:-- rate warming",
    )

    sample_advances = [kwargs for task_id, kwargs in progress.updates if task_id == 1 and kwargs.get("advance") == 1]
    llm_updates = [kwargs for task_id, kwargs in progress.updates if task_id == 2 and kwargs.get("completed") == 1]

    assert len(sample_advances) >= 2
    assert llm_updates
    assert stats[0]["total_llm_calls"] == 2
