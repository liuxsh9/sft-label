import asyncio
from pathlib import Path
from types import SimpleNamespace

import pytest

from sft_label.config import PipelineConfig
from sft_label.pipeline import (
    ChunkCollector,
    _chunk_watchdog_actions,
    _run_one_file_chunked,
    _chunk_has_execution_work,
    _decorate_run_info_with_runtime,
    _effective_chunk_parallelism,
    _effective_chunk_row_count,
    _flushable_chunk_indices,
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


def test_directory_delegate_chunk_row_count_uses_safe_cap():
    assert _effective_chunk_row_count(
        5000,
        1300,
        directory_delegate=True,
        directory_delegate_cap=512,
    ) == 512


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


def test_decorate_run_info_with_runtime_open_cooldown(monkeypatch):
    config = PipelineConfig(enable_adaptive_runtime=True)
    from sft_label.llm_runtime import AdaptiveLLMRuntime

    runtime = AdaptiveLLMRuntime(base_concurrency=12, base_rps=20.0)
    runtime.state = "open"
    runtime._set_effective_limits(1, 0.5)
    runtime._cooldown_until = 150.0
    runtime.gate._in_flight = 0
    setattr(config, "_adaptive_runtime", runtime)
    monkeypatch.setattr("sft_label.llm.progress.time.monotonic", lambda: 120.2)

    info = _decorate_run_info_with_runtime("run 10/100 (10.0%) eta 10:00 rate 4.1/s", config)
    assert info.endswith("runtime open cooldown=30s c=1 rps=0.5 in_flight=0")


def test_finished_unflushed_chunk_does_not_block_new_chunk_admission():
    chunk = ChunkCollector(
        chunk_idx=0,
        samples=[{"id": "s-1"}],
        label_count=1,
    )
    chunk.submit_order = [0]
    chunk.submit_cursor = 1
    chunk.done = 1

    assert _chunk_has_execution_work(chunk) is False


def test_preserve_order_flush_blocks_later_completed_chunks():
    chunk0 = ChunkCollector(chunk_idx=0, samples=[{"id": "s-0"}], label_count=1)
    chunk0.submit_order = [0]
    chunk0.submit_cursor = 1
    chunk0.done = 0

    chunk1 = ChunkCollector(chunk_idx=1, samples=[{"id": "s-1"}], label_count=1)
    chunk1.submit_order = [0]
    chunk1.submit_cursor = 1
    chunk1.done = 1

    collectors = {
        0: chunk0,
        1: chunk1,
    }

    assert _flushable_chunk_indices(
        collectors,
        next_chunk_to_flush=0,
        preserve_order=True,
    ) == []


def test_directory_delegate_can_flush_completed_chunks_out_of_order():
    chunk0 = ChunkCollector(chunk_idx=0, samples=[{"id": "s-0"}], label_count=1)
    chunk0.submit_order = [0]
    chunk0.submit_cursor = 1
    chunk0.done = 0

    chunk1 = ChunkCollector(chunk_idx=1, samples=[{"id": "s-1"}], label_count=1)
    chunk1.submit_order = [0]
    chunk1.submit_cursor = 1
    chunk1.done = 1

    collectors = {
        0: chunk0,
        1: chunk1,
    }

    assert _flushable_chunk_indices(
        collectors,
        next_chunk_to_flush=0,
        preserve_order=False,
    ) == [1]


@pytest.mark.asyncio
async def test_directory_delegate_continues_loading_after_out_of_order_flush(monkeypatch, tmp_path):
    input_file = tmp_path / "input.jsonl"
    input_file.write_text('{"id":"seed"}\n', encoding="utf-8")

    chunk_ids = [0, 1, 2, 3]

    def fake_iter_row_sample_bundle_chunks_from_jsonl(*args, **kwargs):
        for chunk_id in chunk_ids:
            yield [{"chunk_id": chunk_id}]

    def fake_prepare_inline_pass1_batch(row_bundles, **kwargs):
        chunk_id = row_bundles[0]["chunk_id"]
        sample = {"id": f"chunk-{chunk_id}", "conversations": []}
        return SimpleNamespace(
            samples=[sample],
            sample_to_bundle=[0],
            label_indices=[0],
            inherit_map={},
            bundles=row_bundles,
            updated_sample_indices=set(),
            bundle_plans=[],
            stats={},
            labels=[None],
        )

    started_ids = []
    chunk0_release = asyncio.Event()
    chunk3_started = asyncio.Event()
    flushed_ids = []

    async def fake_label_one(*args, **kwargs):
        sample = args[1]
        chunk_id = int(sample["id"].split("-")[-1])
        started_ids.append(chunk_id)
        if chunk_id == 3:
            chunk3_started.set()
        if chunk_id == 0:
            await chunk0_release.wait()
        return 0, {"intent": "coding"}, {"sample_id": sample["id"], "status": "success"}

    def fake_flush_chunk(chunk, *args, **kwargs):
        flushed_ids.append(chunk.chunk_idx)
        chunk.completed = True

    monkeypatch.setattr(
        "sft_label.pipeline.iter_row_sample_bundle_chunks_from_jsonl",
        fake_iter_row_sample_bundle_chunks_from_jsonl,
    )
    monkeypatch.setattr(
        "sft_label.pipeline.prepare_inline_pass1_batch",
        fake_prepare_inline_pass1_batch,
    )
    monkeypatch.setattr("sft_label.pipeline.label_one", fake_label_one)
    monkeypatch.setattr("sft_label.pipeline._flush_chunk", fake_flush_chunk)

    task = asyncio.create_task(
        _run_one_file_chunked(
            input_path=input_file,
            output_dir=tmp_path / "out",
            http_client=None,
            sem=asyncio.Semaphore(8),
            model="fake-model",
            config=PipelineConfig(
                concurrency=8,
                chunk_size=1,
                max_active_chunks=2,
                dir_pipeline_watermark=1.0,
                enable_adaptive_runtime=False,
            ),
            directory_delegate=True,
        )
    )

    await asyncio.wait_for(chunk3_started.wait(), timeout=1.0)
    assert 0 in started_ids
    assert 1 in flushed_ids
    assert 2 in flushed_ids

    chunk0_release.set()
    await task


@pytest.mark.asyncio
async def test_directory_delegate_zero_work_chunks_flush_all_with_bounded_preload(monkeypatch, tmp_path):
    input_file = tmp_path / "input.jsonl"
    input_file.write_text('{"id":"seed"}\n', encoding="utf-8")

    chunk_ids = [0, 1, 2, 3]
    loaded_ids = []
    flushed_ids = []
    first_flush_loaded_snapshot = []

    def fake_iter_row_sample_bundle_chunks_from_jsonl(*args, **kwargs):
        for chunk_id in chunk_ids:
            yield [{"chunk_id": chunk_id}]

    def fake_prepare_inline_pass1_batch(row_bundles, **kwargs):
        chunk_id = row_bundles[0]["chunk_id"]
        loaded_ids.append(chunk_id)
        sample = {"id": f"chunk-{chunk_id}", "conversations": []}
        return SimpleNamespace(
            samples=[sample],
            sample_to_bundle=[0],
            label_indices=[],
            inherit_map={0: 0},
            bundles=row_bundles,
            updated_sample_indices=set(),
            bundle_plans=[],
            stats={},
            labels=[{"intent": "coding"}],
        )

    def fake_flush_chunk(chunk, *args, **kwargs):
        nonlocal first_flush_loaded_snapshot
        if not first_flush_loaded_snapshot:
            first_flush_loaded_snapshot = list(loaded_ids)
        flushed_ids.append(chunk.chunk_idx)
        chunk.completed = True

    monkeypatch.setattr(
        "sft_label.pipeline.iter_row_sample_bundle_chunks_from_jsonl",
        fake_iter_row_sample_bundle_chunks_from_jsonl,
    )
    monkeypatch.setattr(
        "sft_label.pipeline.prepare_inline_pass1_batch",
        fake_prepare_inline_pass1_batch,
    )
    monkeypatch.setattr("sft_label.pipeline._flush_chunk", fake_flush_chunk)

    await _run_one_file_chunked(
        input_path=input_file,
        output_dir=tmp_path / "out-zero-work",
        http_client=None,
        sem=asyncio.Semaphore(8),
        model="fake-model",
        config=PipelineConfig(
            concurrency=8,
            chunk_size=1,
            max_active_chunks=2,
            dir_pipeline_watermark=1.0,
            enable_adaptive_runtime=False,
        ),
        directory_delegate=True,
    )

    assert first_flush_loaded_snapshot == [0, 1]
    assert flushed_ids == [0, 1, 2, 3]


@pytest.mark.asyncio
async def test_chunk_watchdog_cancels_stalled_work_and_allows_flush(monkeypatch, tmp_path):
    input_file = tmp_path / "input-watchdog.jsonl"
    input_file.write_text('{"id":"seed"}\n', encoding="utf-8")

    def fake_iter_row_sample_bundle_chunks_from_jsonl(*args, **kwargs):
        yield [{"chunk_id": 0}]

    def fake_prepare_inline_pass1_batch(row_bundles, **kwargs):
        return SimpleNamespace(
            samples=[{"id": "chunk-0-sample", "conversations": []}],
            sample_to_bundle=[0],
            label_indices=[0],
            inherit_map={},
            bundles=row_bundles,
            updated_sample_indices=set(),
            bundle_plans=[],
            stats={},
            labels=[None],
        )

    async def fake_label_one(*args, **kwargs):
        await asyncio.sleep(3600)
        raise AssertionError("watchdog should cancel this task before completion")

    recovery_called = False
    flushed = []

    async def fake_recovery_sweep(*, all_labels, all_monitors, candidate_indices, **kwargs):
        nonlocal recovery_called
        recovery_called = True
        assert candidate_indices == [0]
        all_labels[0] = {"intent": "coding"}
        all_monitors[0] = {
            "sample_id": "chunk-0-sample",
            "status": "success",
            "retryable_infra": False,
        }
        return 1

    def fake_flush_chunk(chunk, *args, **kwargs):
        flushed.append(
            (
                chunk.chunk_idx,
                chunk.ok,
                chunk.fail,
                chunk.labels[0],
                chunk.monitors[0]["status"],
            )
        )
        chunk.completed = True

    monkeypatch.setattr(
        "sft_label.pipeline.iter_row_sample_bundle_chunks_from_jsonl",
        fake_iter_row_sample_bundle_chunks_from_jsonl,
    )
    monkeypatch.setattr(
        "sft_label.pipeline.prepare_inline_pass1_batch",
        fake_prepare_inline_pass1_batch,
    )
    monkeypatch.setattr("sft_label.pipeline.label_one", fake_label_one)
    monkeypatch.setattr("sft_label.pipeline._run_pass1_recovery_sweep", fake_recovery_sweep)
    monkeypatch.setattr("sft_label.pipeline._flush_chunk", fake_flush_chunk)

    await asyncio.wait_for(
        _run_one_file_chunked(
            input_path=input_file,
            output_dir=tmp_path / "out-watchdog",
            http_client=None,
            sem=asyncio.Semaphore(4),
            model="fake-model",
            config=PipelineConfig(
                concurrency=4,
                chunk_size=1,
                max_active_chunks=2,
                dir_pipeline_watermark=1.0,
                enable_adaptive_runtime=False,
                enable_stage_recovery_sweep=True,
                chunk_stall_timeout=0.05,
            ),
            directory_delegate=True,
        ),
        timeout=1.0,
    )

    assert recovery_called is True
    assert flushed == [(0, 1, 0, {"intent": "coding"}, "success")]


@pytest.mark.asyncio
async def test_chunk_watchdog_does_not_fail_completed_future_waiting_to_be_harvested(monkeypatch, tmp_path):
    input_file = tmp_path / "input-watchdog-race.jsonl"
    input_file.write_text('{"id":"seed"}\n', encoding="utf-8")

    def fake_iter_row_sample_bundle_chunks_from_jsonl(*args, **kwargs):
        yield [{"chunk_id": 0}]

    def fake_prepare_inline_pass1_batch(row_bundles, **kwargs):
        return SimpleNamespace(
            samples=[{"id": "chunk-0-sample", "conversations": []}],
            sample_to_bundle=[0],
            label_indices=[0],
            inherit_map={},
            bundles=row_bundles,
            updated_sample_indices=set(),
            bundle_plans=[],
            stats={},
            labels=[None],
        )

    async def fake_label_one(*args, **kwargs):
        return 0, {"intent": "coding"}, {"sample_id": "chunk-0-sample", "status": "success"}

    real_wait = asyncio.wait
    wait_calls = 0
    flushed = []

    async def fake_wait(fs, **kwargs):
        nonlocal wait_calls
        wait_calls += 1
        if wait_calls == 1:
            await asyncio.sleep(0.02)
            return set(), set(fs)
        return await real_wait(fs, **kwargs)

    def fake_flush_chunk(chunk, *args, **kwargs):
        flushed.append((chunk.ok, chunk.fail, chunk.labels[0], chunk.monitors[0]["status"]))
        chunk.completed = True

    monkeypatch.setattr(
        "sft_label.pipeline.iter_row_sample_bundle_chunks_from_jsonl",
        fake_iter_row_sample_bundle_chunks_from_jsonl,
    )
    monkeypatch.setattr(
        "sft_label.pipeline.prepare_inline_pass1_batch",
        fake_prepare_inline_pass1_batch,
    )
    monkeypatch.setattr("sft_label.pipeline.label_one", fake_label_one)
    monkeypatch.setattr("sft_label.pipeline.asyncio.wait", fake_wait)
    monkeypatch.setattr("sft_label.pipeline._flush_chunk", fake_flush_chunk)

    await asyncio.wait_for(
        _run_one_file_chunked(
            input_path=input_file,
            output_dir=tmp_path / "out-watchdog-race",
            http_client=None,
            sem=asyncio.Semaphore(4),
            model="fake-model",
            config=PipelineConfig(
                concurrency=4,
                chunk_size=1,
                max_active_chunks=2,
                dir_pipeline_watermark=1.0,
                enable_adaptive_runtime=False,
                enable_stage_recovery_sweep=False,
                chunk_stall_timeout=0.01,
            ),
            directory_delegate=True,
        ),
        timeout=1.0,
    )

    assert flushed == [(1, 0, {"intent": "coding"}, "success")]


def test_chunk_watchdog_actions_limits_cancellations_and_hides_zero_cancel_noise():
    actions = _chunk_watchdog_actions(
        entries=[
            {"sample_idx": 0, "submitted_at": 0.0, "done": False},
            {"sample_idx": 1, "submitted_at": 1.0, "done": False},
            {"sample_idx": 2, "submitted_at": 2.0, "done": False},
            {"sample_idx": 3, "submitted_at": 199.5, "done": False},
            {"sample_idx": 4, "submitted_at": 50.0, "done": True},
        ],
        now=200.0,
        chunk_last_progress_at=0.0,
        chunk_stall_timeout=100.0,
        cancel_limit=2,
    )

    assert actions["harvest_sample_indices"] == [4]
    assert actions["cancel_sample_indices"] == [0, 1]
    assert actions["skipped_still_running"] == 2
    assert actions["should_log"] is True
    assert "canceled 2 stalled sample(s)" in actions["log_line"]
    assert "1 completed sample(s) already done" in actions["log_line"]
    assert "2 still running" in actions["log_line"]


def test_chunk_watchdog_actions_skips_zero_cancel_noise_when_only_harvesting_done():
    actions = _chunk_watchdog_actions(
        entries=[
            {"sample_idx": 0, "submitted_at": 0.0, "done": True},
            {"sample_idx": 1, "submitted_at": 0.0, "done": True},
        ],
        now=200.0,
        chunk_last_progress_at=0.0,
        chunk_stall_timeout=100.0,
        cancel_limit=2,
    )

    assert actions["harvest_sample_indices"] == [0, 1]
    assert actions["cancel_sample_indices"] == []
    assert actions["should_log"] is True
    assert "harvested 2 completed sample(s)" in actions["log_line"]
    assert "canceled 0" not in actions["log_line"]


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


@pytest.mark.asyncio
async def test_directory_chunked_delegate_cleans_prior_file_before_directory_completion(monkeypatch, tmp_path):
    first = tmp_path / "a.jsonl"
    second = tmp_path / "b.jsonl"
    third = tmp_path / "c.jsonl"
    first.write_text('{"id":"a"}\n', encoding="utf-8")
    second.write_text('{"id":"b"}\n', encoding="utf-8")
    third.write_text('{"id":"c"}\n', encoding="utf-8")

    starts: list[str] = []
    postprocess_starts: list[str] = []
    release_second = asyncio.Event()

    async def fake_run_one_file_chunked(input_path, *args, **kwargs):
        name = Path(input_path).name
        starts.append(name)
        output_dir = Path(args[0])
        labeled_path = output_dir / "labeled.jsonl"
        output_dir.mkdir(parents=True, exist_ok=True)
        labeled_path.write_text(f"{name}\n", encoding="utf-8")
        if name == "b.jsonl":
            await release_second.wait()
        return {
            "total_llm_calls": 1,
            "sparse_labeled": 1,
            "sparse_inherited": 0,
            "total_samples": 1,
            "success": 1,
                "_pass1_postprocess_job": {
                    "input_name": name,
                    "labeled_path": str(labeled_path),
                    "cleanup_labeled_path": True,
                },
            }

    def fake_run_pass1_postprocess_job(job, pprint=print):
        labeled_path = Path(job["labeled_path"])
        postprocess_starts.append(job["input_name"])
        if labeled_path.exists():
            labeled_path.unlink()

    monkeypatch.setattr("sft_label.pipeline._should_delegate_directory_jsonl_to_chunked", lambda *args, **kwargs: True)
    monkeypatch.setattr("sft_label.pipeline._run_one_file_chunked", fake_run_one_file_chunked)
    monkeypatch.setattr("sft_label.pipeline._run_pass1_postprocess_job", fake_run_pass1_postprocess_job)
    monkeypatch.setattr("sft_label.pipeline.update_checkpoint", lambda *args, **kwargs: None)

    task = asyncio.create_task(
        run_directory_pipeline(
            dir_files=[(first, Path("a.jsonl")), (second, Path("b.jsonl")), (third, Path("c.jsonl"))],
            run_dir=tmp_path / "run",
            model="fake-model",
            concurrency=2,
            checkpoint_path=tmp_path / "checkpoint.json",
            progress=None,
            http_client=None,
            sem=asyncio.Semaphore(2),
            config=PipelineConfig(enable_adaptive_runtime=False),
        )
    )

    await asyncio.sleep(0.05)

    assert starts[:2] == ["a.jsonl", "b.jsonl"]
    assert postprocess_starts == ["a.jsonl"]
    assert not (tmp_path / "run" / "a" / "labeled.jsonl").exists()
    assert (tmp_path / "run" / "b" / "labeled.jsonl").exists()

    release_second.set()
    stats = await task

    assert [item["success"] for item in stats] == [1, 1, 1]
    assert starts == ["a.jsonl", "b.jsonl", "c.jsonl"]
    assert postprocess_starts == ["a.jsonl", "b.jsonl", "c.jsonl"]
