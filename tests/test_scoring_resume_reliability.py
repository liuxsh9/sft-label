import asyncio
import json
import os
from pathlib import Path
from unittest.mock import patch

from sft_label.config import PipelineConfig
from sft_label.scoring import (
    _rebuild_chunked_summaries_and_monitors,
    _finalize_pass2_working_files,
    _pass2_checkpoint_path,
    _pass2_working_path,
    _resolve_pass2_resume_path,
    _selection_summary_from_sample,
    _run_scoring_file_chunked,
)


def _sample(sample_id: str) -> dict:
    return {
        "id": sample_id,
        "conversations": [
            {"from": "human", "value": f"Q:{sample_id}"},
            {"from": "gpt", "value": f"A:{sample_id}"},
        ],
        "labels": {"intent": "build", "difficulty": "intermediate"},
    }


def _value_payload(*, score: float, rarity_score: float, confidence: float) -> dict:
    return {
        "complexity": {
            "instruction": 5,
            "analytical_depth": 5,
            "implementation": 5,
            "overall": 5,
        },
        "quality": {
            "correctness": 7,
            "code_quality": 7,
            "explanation": 7,
            "completeness": 7,
            "overall": 7,
        },
        "reasoning": {
            "clarity": 6,
            "consistency": 6,
            "self_correction": False,
            "overall": 6,
        },
        "rarity": {"score": rarity_score},
        "flags": [],
        "thinking_mode": "fast",
        "value_score": score,
        "confidence": confidence,
    }


def _monitor(sample_id: str, *, status: str = "success", retryable_infra: bool | None = None) -> dict:
    payload = {
        "sample_id": sample_id,
        "status": status,
        "llm_calls": 1,
        "prompt_tokens": 10,
        "completion_tokens": 5,
        "attempts": 1,
        "validation_issues": [],
    }
    if retryable_infra is not None:
        payload["retryable_infra"] = retryable_infra
    return payload


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n", encoding="utf-8")


def _read_jsonl(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _quiet_chunked_config(**overrides) -> PipelineConfig:
    base = dict(
        scoring_concurrency=1,
        chunk_size=1,
        max_active_chunks=2,
        enable_adaptive_runtime=False,
        pass2_heavy_postprocess_mode="defer",
    )
    base.update(overrides)
    return PipelineConfig(**base)


def test_chunked_resume_fastpath_materializes_checkpoint_only_artifacts(tmp_path):
    labeled_path = tmp_path / "labeled.jsonl"
    samples = [_sample(f"sample-{idx}") for idx in range(3)]
    _write_jsonl(labeled_path, samples)

    checkpoint_scored = _pass2_checkpoint_path(tmp_path, "scored.jsonl")
    checkpoint_monitor = _pass2_checkpoint_path(tmp_path, "monitor_value.jsonl")
    checkpoint_scored_rows = [
        {
            **sample,
            "value": _value_payload(score=6.0 + idx, rarity_score=4.0 + idx, confidence=0.8),
        }
        for idx, sample in enumerate(samples)
    ]
    checkpoint_monitor_rows = [_monitor(sample["id"], status="checkpoint-only") for sample in samples]
    _write_jsonl(
        checkpoint_scored,
        checkpoint_scored_rows,
    )
    _write_jsonl(checkpoint_monitor, checkpoint_monitor_rows)
    _write_jsonl(tmp_path / "failed_value.jsonl", [dict(samples[0], value=None)])
    _write_jsonl(
        tmp_path / "score_failures.jsonl",
        [{"sample_id": samples[0]["id"], "status": "stale-final-failure"}],
    )

    async def fail_score_one(*_args, **_kwargs):
        raise AssertionError("score_one must not run during checkpoint-only resume finalize")

    with patch("sft_label.scoring.score_one", side_effect=fail_score_one):
        asyncio.run(
            _run_scoring_file_chunked(
                labeled_path,
                tmp_path,
                None,
                0,
                _quiet_chunked_config(enable_stage_recovery_sweep=False),
                resume=True,
                generate_dashboard=False,
                show_progress=False,
                print_summary=False,
                quiet=True,
            )
        )

    assert (tmp_path / "scored.jsonl").exists()
    assert (tmp_path / "monitor_value.jsonl").exists()
    final_scored_rows = _read_jsonl(tmp_path / "scored.jsonl")
    final_monitor_rows = _read_jsonl(tmp_path / "monitor_value.jsonl")
    assert [row["value"]["value_score"] for row in final_scored_rows] == [6.0, 7.0, 8.0]
    assert {row["status"] for row in final_monitor_rows} == {"checkpoint-only"}
    assert not (tmp_path / "failed_value.jsonl").exists()
    assert not (tmp_path / "score_failures.jsonl").exists()


def test_resolve_resume_path_ignores_empty_or_truncated_checkpoint_when_final_is_valid(tmp_path):
    final_scored_rows = [
        {
            **_sample("sample-final"),
            "value": _value_payload(score=7.3, rarity_score=4.9, confidence=0.92),
        }
    ]
    _write_jsonl(tmp_path / "scored.jsonl", final_scored_rows)

    checkpoint = _pass2_checkpoint_path(tmp_path, "scored.jsonl")
    checkpoint.write_text("", encoding="utf-8")
    assert _resolve_pass2_resume_path(tmp_path, "scored.jsonl") == tmp_path / "scored.jsonl"

    checkpoint.write_text('{"id":"sample-final","value":', encoding="utf-8")
    assert _resolve_pass2_resume_path(tmp_path, "scored.jsonl") == tmp_path / "scored.jsonl"


def test_selection_summary_forwards_config_to_selection_view():
    sample = {
        **_sample("sample-0"),
        "value": _value_payload(score=6.5, rarity_score=4.2, confidence=0.9),
    }
    config = PipelineConfig(value_truncation_budget=321)
    mocked_view = {
        "current_request": "request",
        "trajectory": "trajectory",
        "response": "response",
        "trajectory_turn_count": 1,
        "trajectory_tool_turns": 0,
    }
    with patch("sft_label.scoring._selection_view_from_sample", return_value=mocked_view) as patched_view:
        _selection_summary_from_sample(sample, config=config)

    patched_view.assert_called_once_with(sample, config=config)


def test_rebuild_chunked_summaries_passes_config_to_selection_summary(tmp_path):
    scored_path = tmp_path / "scored.jsonl"
    monitor_path = tmp_path / "monitor_value.jsonl"
    _write_jsonl(
        scored_path,
        [{**_sample("sample-0"), "value": _value_payload(score=7.0, rarity_score=4.1, confidence=0.8)}],
    )
    _write_jsonl(monitor_path, [_monitor("sample-0", status="success")])
    config = PipelineConfig(value_truncation_budget=222)

    with patch(
        "sft_label.scoring._selection_summary_from_sample",
        return_value={"labels": {}, "value_score": 7.0},
    ) as patched_summary:
        _rebuild_chunked_summaries_and_monitors(scored_path, monitor_path, config=config)

    assert patched_summary.call_count == 1
    assert patched_summary.call_args.kwargs["config"] is config


def test_resume_fastpath_ignores_single_corrupt_working_artifact(tmp_path):
    labeled_path = tmp_path / "labeled.jsonl"
    samples = [_sample("sample-0")]
    _write_jsonl(labeled_path, samples)

    checkpoint_scored_rows = [
        {
            **samples[0],
            "value": _value_payload(score=9.4, rarity_score=5.1, confidence=0.93),
        }
    ]
    checkpoint_monitor_rows = [_monitor(samples[0]["id"], status="checkpoint-source")]
    _write_jsonl(_pass2_checkpoint_path(tmp_path, "scored.jsonl"), checkpoint_scored_rows)
    _write_jsonl(_pass2_checkpoint_path(tmp_path, "monitor_value.jsonl"), checkpoint_monitor_rows)
    _pass2_working_path(tmp_path, "scored.jsonl").write_text('{"id":"sample-0","value":', encoding="utf-8")

    async def fail_score_one(*_args, **_kwargs):
        raise AssertionError("score_one must not run during resume fast-path finalize")

    with patch("sft_label.scoring.score_one", side_effect=fail_score_one):
        asyncio.run(
            _run_scoring_file_chunked(
                labeled_path,
                tmp_path,
                None,
                0,
                _quiet_chunked_config(enable_stage_recovery_sweep=False),
                resume=True,
                generate_dashboard=False,
                show_progress=False,
                print_summary=False,
                quiet=True,
            )
        )

    final_scored_rows = _read_jsonl(tmp_path / "scored.jsonl")
    assert [row["value"]["value_score"] for row in final_scored_rows] == [9.4]


def test_resume_setup_does_not_copy_corrupt_nonempty_working_over_checkpoint(tmp_path):
    labeled_path = tmp_path / "labeled.jsonl"
    samples = [_sample("sample-0"), _sample("sample-1")]
    _write_jsonl(labeled_path, samples)

    checkpoint_scored_rows = [
        {
            **samples[0],
            "value": _value_payload(score=8.8, rarity_score=5.2, confidence=0.94),
        }
    ]
    _write_jsonl(_pass2_checkpoint_path(tmp_path, "scored.jsonl"), checkpoint_scored_rows)
    _pass2_working_path(tmp_path, "scored.jsonl").write_text('{"id":"sample-0","value":', encoding="utf-8")

    with patch("sft_label.scoring._reset_pass2_working_files", side_effect=RuntimeError("stop-after-resume-setup")):
        with patch("sft_label.scoring.score_one", side_effect=AssertionError("score_one should not run")):
            try:
                asyncio.run(
                    _run_scoring_file_chunked(
                        labeled_path,
                        tmp_path,
                        None,
                        0,
                        _quiet_chunked_config(enable_stage_recovery_sweep=False),
                        resume=True,
                        generate_dashboard=False,
                        show_progress=False,
                        print_summary=False,
                        quiet=True,
                    )
                )
            except RuntimeError as exc:
                assert str(exc) == "stop-after-resume-setup"
            else:
                raise AssertionError("Expected resume setup interruption")

    assert _read_jsonl(_pass2_checkpoint_path(tmp_path, "scored.jsonl")) == checkpoint_scored_rows


def test_resume_fastpath_rejects_trailing_corrupt_working_and_uses_checkpoint(tmp_path):
    labeled_path = tmp_path / "labeled.jsonl"
    samples = [_sample("sample-0")]
    _write_jsonl(labeled_path, samples)

    checkpoint_scored_rows = [
        {
            **samples[0],
            "value": _value_payload(score=8.2, rarity_score=5.0, confidence=0.92),
        }
    ]
    checkpoint_monitor_rows = [_monitor(samples[0]["id"], status="checkpoint-source")]
    _write_jsonl(_pass2_checkpoint_path(tmp_path, "scored.jsonl"), checkpoint_scored_rows)
    _write_jsonl(_pass2_checkpoint_path(tmp_path, "monitor_value.jsonl"), checkpoint_monitor_rows)

    working_scored = _pass2_working_path(tmp_path, "scored.jsonl")
    working_scored.write_text(
        json.dumps(checkpoint_scored_rows[0], ensure_ascii=False) + "\n" + '{"id":"sample-0","value":' + "\n",
        encoding="utf-8",
    )
    _write_jsonl(_pass2_working_path(tmp_path, "monitor_value.jsonl"), checkpoint_monitor_rows)

    async def fail_score_one(*_args, **_kwargs):
        raise AssertionError("score_one must not run during resume fast-path finalize")

    with patch("sft_label.scoring.score_one", side_effect=fail_score_one):
        asyncio.run(
            _run_scoring_file_chunked(
                labeled_path,
                tmp_path,
                None,
                0,
                _quiet_chunked_config(enable_stage_recovery_sweep=False),
                resume=True,
                generate_dashboard=False,
                show_progress=False,
                print_summary=False,
                quiet=True,
            )
        )

    final_scored_rows = _read_jsonl(tmp_path / "scored.jsonl")
    assert [row["value"]["value_score"] for row in final_scored_rows] == [8.2]


def test_finalize_ignores_corrupt_working_failure_artifact_and_uses_checkpoint(tmp_path):
    checkpoint_failed = [dict(_sample("sample-checkpoint"), value=None)]
    checkpoint_failures = [{"sample_id": "sample-checkpoint", "status": "checkpoint-failure"}]
    final_failed = [dict(_sample("sample-final"), value=None)]
    final_failures = [{"sample_id": "sample-final", "status": "final-failure"}]

    _write_jsonl(_pass2_checkpoint_path(tmp_path, "failed_value.jsonl"), checkpoint_failed)
    _write_jsonl(_pass2_checkpoint_path(tmp_path, "score_failures.jsonl"), checkpoint_failures)
    _write_jsonl(tmp_path / "failed_value.jsonl", final_failed)
    _write_jsonl(tmp_path / "score_failures.jsonl", final_failures)

    _pass2_working_path(tmp_path, "failed_value.jsonl").write_text('{"id":"broken-failure"', encoding="utf-8")
    _pass2_working_path(tmp_path, "score_failures.jsonl").write_text('{"sample_id":"broken-failure"', encoding="utf-8")

    _finalize_pass2_working_files(tmp_path)

    assert _read_jsonl(tmp_path / "failed_value.jsonl") == checkpoint_failed
    assert _read_jsonl(tmp_path / "score_failures.jsonl") == checkpoint_failures


def test_finalize_empty_working_failure_marker_does_not_erase_valid_checkpoint(tmp_path):
    checkpoint_failed = [dict(_sample("sample-checkpoint"), value=None)]
    checkpoint_failures = [{"sample_id": "sample-checkpoint", "status": "checkpoint-failure"}]
    _write_jsonl(_pass2_checkpoint_path(tmp_path, "failed_value.jsonl"), checkpoint_failed)
    _write_jsonl(_pass2_checkpoint_path(tmp_path, "score_failures.jsonl"), checkpoint_failures)
    _pass2_working_path(tmp_path, "failed_value.jsonl").write_text("", encoding="utf-8")
    _pass2_working_path(tmp_path, "score_failures.jsonl").write_text("", encoding="utf-8")

    _finalize_pass2_working_files(tmp_path)

    assert _read_jsonl(tmp_path / "failed_value.jsonl") == checkpoint_failed
    assert _read_jsonl(tmp_path / "score_failures.jsonl") == checkpoint_failures


def test_resume_setup_empty_higher_tier_failure_marker_keeps_lower_tier_failures(tmp_path):
    labeled_path = tmp_path / "labeled.jsonl"
    samples = [_sample("sample-0"), _sample("sample-1")]
    _write_jsonl(labeled_path, samples)

    checkpoint_failed_rows = [dict(samples[0], value=None)]
    checkpoint_failure_rows = [{"sample_id": samples[0]["id"], "status": "checkpoint-failure"}]
    final_failed_rows = [dict(samples[1], value=None)]
    final_failure_rows = [{"sample_id": samples[1]["id"], "status": "final-failure"}]

    # empty working should not beat non-empty checkpoint
    _pass2_working_path(tmp_path, "score_failures.jsonl").write_text("", encoding="utf-8")
    _write_jsonl(_pass2_checkpoint_path(tmp_path, "score_failures.jsonl"), checkpoint_failure_rows)

    # empty checkpoint should not beat non-empty final
    _pass2_checkpoint_path(tmp_path, "failed_value.jsonl").write_text("", encoding="utf-8")
    _write_jsonl(tmp_path / "failed_value.jsonl", final_failed_rows)

    # Provide minimal scored/monitor artifacts to reach resume setup path.
    _write_jsonl(_pass2_checkpoint_path(tmp_path, "scored.jsonl"), [{**samples[0], "value": _value_payload(score=8.0, rarity_score=5.0, confidence=0.9)}])
    _write_jsonl(_pass2_checkpoint_path(tmp_path, "monitor_value.jsonl"), [_monitor(samples[0]["id"])])

    with patch("sft_label.scoring._reset_pass2_working_files", side_effect=RuntimeError("stop-after-resume-setup")):
        try:
            asyncio.run(
                _run_scoring_file_chunked(
                    labeled_path,
                    tmp_path,
                    None,
                    0,
                    _quiet_chunked_config(enable_stage_recovery_sweep=False),
                    resume=True,
                    generate_dashboard=False,
                    show_progress=False,
                    print_summary=False,
                    quiet=True,
                )
            )
        except RuntimeError as exc:
            assert str(exc) == "stop-after-resume-setup"
        else:
            raise AssertionError("Expected resume setup interruption")

    assert _read_jsonl(_pass2_checkpoint_path(tmp_path, "score_failures.jsonl")) == checkpoint_failure_rows
    assert _read_jsonl(_pass2_checkpoint_path(tmp_path, "failed_value.jsonl")) == final_failed_rows


def test_resume_setup_preserves_checkpoint_over_final_when_no_working_exists(tmp_path):
    labeled_path = tmp_path / "labeled.jsonl"
    samples = [_sample(f"sample-{idx}") for idx in range(2)]
    _write_jsonl(labeled_path, samples)

    checkpoint_scored_rows = [
        {
            **samples[0],
            "value": _value_payload(score=9.1, rarity_score=6.2, confidence=0.91),
        }
    ]
    final_scored_rows = [
        {
            **samples[0],
            "value": _value_payload(score=1.2, rarity_score=1.2, confidence=0.1),
        }
    ]
    checkpoint_monitor_rows = [_monitor(samples[0]["id"], status="checkpoint-source")]
    final_monitor_rows = [_monitor(samples[0]["id"], status="final-source")]

    _write_jsonl(_pass2_checkpoint_path(tmp_path, "scored.jsonl"), checkpoint_scored_rows)
    _write_jsonl(_pass2_checkpoint_path(tmp_path, "monitor_value.jsonl"), checkpoint_monitor_rows)
    _write_jsonl(tmp_path / "scored.jsonl", final_scored_rows)
    _write_jsonl(tmp_path / "monitor_value.jsonl", final_monitor_rows)

    checkpoint_failed_rows = [dict(samples[0], value=None)]
    final_failed_rows = [dict(samples[1], value=None)]
    _write_jsonl(_pass2_checkpoint_path(tmp_path, "failed_value.jsonl"), checkpoint_failed_rows)
    _write_jsonl(tmp_path / "failed_value.jsonl", final_failed_rows)
    _write_jsonl(
        _pass2_checkpoint_path(tmp_path, "score_failures.jsonl"),
        [{"sample_id": samples[0]["id"], "status": "checkpoint-failure"}],
    )
    _write_jsonl(
        tmp_path / "score_failures.jsonl",
        [{"sample_id": samples[1]["id"], "status": "final-failure"}],
    )

    with patch("sft_label.scoring._reset_pass2_working_files", side_effect=RuntimeError("stop-after-resume-setup")):
        with patch("sft_label.scoring.score_one", side_effect=AssertionError("score_one should not run")):
            try:
                asyncio.run(
                    _run_scoring_file_chunked(
                        labeled_path,
                        tmp_path,
                        None,
                        0,
                        _quiet_chunked_config(enable_stage_recovery_sweep=False),
                        resume=True,
                        generate_dashboard=False,
                        show_progress=False,
                        print_summary=False,
                        quiet=True,
                    )
                )
            except RuntimeError as exc:
                assert str(exc) == "stop-after-resume-setup"
            else:
                raise AssertionError("Expected resume setup interruption")

    assert _read_jsonl(_pass2_checkpoint_path(tmp_path, "scored.jsonl")) == checkpoint_scored_rows
    assert _read_jsonl(_pass2_checkpoint_path(tmp_path, "monitor_value.jsonl")) == checkpoint_monitor_rows
    assert _read_jsonl(_pass2_checkpoint_path(tmp_path, "failed_value.jsonl")) == checkpoint_failed_rows
    assert _read_jsonl(_pass2_checkpoint_path(tmp_path, "score_failures.jsonl")) == [
        {"sample_id": samples[0]["id"], "status": "checkpoint-failure"}
    ]


def test_resume_setup_uses_final_when_checkpoint_is_empty_or_truncated(tmp_path):
    labeled_path = tmp_path / "labeled.jsonl"
    samples = [_sample("sample-0"), _sample("sample-1")]
    _write_jsonl(labeled_path, samples)

    final_scored_rows = [
        {
            **samples[0],
            "value": _value_payload(score=8.4, rarity_score=5.5, confidence=0.95),
        }
    ]
    final_monitor_rows = [_monitor(samples[0]["id"], status="final-source")]
    _write_jsonl(tmp_path / "scored.jsonl", final_scored_rows)
    _write_jsonl(tmp_path / "monitor_value.jsonl", final_monitor_rows)

    _pass2_checkpoint_path(tmp_path, "scored.jsonl").write_text('{"id":"sample-0","value":', encoding="utf-8")
    _pass2_checkpoint_path(tmp_path, "monitor_value.jsonl").write_text("", encoding="utf-8")

    with patch("sft_label.scoring._reset_pass2_working_files", side_effect=RuntimeError("stop-after-resume-setup")):
        with patch("sft_label.scoring.score_one", side_effect=AssertionError("score_one should not run")):
            try:
                asyncio.run(
                    _run_scoring_file_chunked(
                        labeled_path,
                        tmp_path,
                        None,
                        0,
                        _quiet_chunked_config(enable_stage_recovery_sweep=False),
                        resume=True,
                        generate_dashboard=False,
                        show_progress=False,
                        print_summary=False,
                        quiet=True,
                    )
                )
            except RuntimeError as exc:
                assert str(exc) == "stop-after-resume-setup"
            else:
                raise AssertionError("Expected resume setup interruption")

    assert _read_jsonl(_pass2_checkpoint_path(tmp_path, "scored.jsonl")) == final_scored_rows
    assert _read_jsonl(_pass2_checkpoint_path(tmp_path, "monitor_value.jsonl")) == final_monitor_rows


def test_chunked_recovery_sweep_keeps_failure_outputs_consistent_after_recovery(tmp_path):
    labeled_path = tmp_path / "labeled.jsonl"
    samples = [_sample("sample-recovered"), _sample("sample-failed")]
    _write_jsonl(labeled_path, samples)

    calls: dict[str, int] = {}

    async def mock_score_one(http_client, sample, model, rarity_result,
                             sample_idx, total, sem, config=None, rate_limiter=None):
        sid = sample["id"]
        calls[sid] = calls.get(sid, 0) + 1

        if sid == "sample-recovered" and calls[sid] == 1:
            return None, {
                **_monitor(sid, status="failed", retryable_infra=True),
                "error": "timeout",
                "error_class": "timeout",
                "prompt_tokens": 0,
                "completion_tokens": 0,
            }

        if sid == "sample-failed":
            return None, {
                **_monitor(sid, status="failed", retryable_infra=False),
                "error": "invalid_request",
                "error_class": "input_error",
                "prompt_tokens": 0,
                "completion_tokens": 0,
            }

        return _value_payload(score=7.2, rarity_score=5.0, confidence=0.9), _monitor(sid)

    with patch("sft_label.scoring.score_one", side_effect=mock_score_one):
        asyncio.run(
            _run_scoring_file_chunked(
                labeled_path,
                tmp_path,
                None,
                0,
                _quiet_chunked_config(
                    enable_stage_recovery_sweep=True,
                    recovery_sweep_max_passes=1,
                ),
                resume=False,
                generate_dashboard=False,
                show_progress=False,
                print_summary=False,
                quiet=True,
            )
        )

    failed_ids = {row["id"] for row in _read_jsonl(tmp_path / "failed_value.jsonl")}
    failure_log_ids = {row["sample_id"] for row in _read_jsonl(tmp_path / "score_failures.jsonl")}

    assert failed_ids == {"sample-failed"}
    assert failure_log_ids == {"sample-failed"}


def test_resume_fastpath_runs_recovery_sweep_before_finalize(tmp_path):
    labeled_path = tmp_path / "labeled.jsonl"
    sample = _sample("sample-retryable")
    _write_jsonl(labeled_path, [sample])

    _write_jsonl(_pass2_checkpoint_path(tmp_path, "scored.jsonl"), [dict(sample, value=None)])
    _write_jsonl(
        _pass2_checkpoint_path(tmp_path, "monitor_value.jsonl"),
        [
            {
                **_monitor(sample["id"], status="failed", retryable_infra=True),
                "error": "timeout",
                "error_class": "timeout",
            }
        ],
    )
    _write_jsonl(tmp_path / "failed_value.jsonl", [dict(sample, value=None)])
    _write_jsonl(tmp_path / "score_failures.jsonl", [{"sample_id": sample["id"], "status": "stale"}])

    calls = 0

    async def mock_score_one(http_client, sample, model, rarity_result,
                             sample_idx, total, sem, config=None, rate_limiter=None):
        nonlocal calls
        calls += 1
        return (
            _value_payload(score=8.7, rarity_score=5.2, confidence=0.94),
            _monitor(sample["id"], status="success", retryable_infra=False),
        )

    with patch("sft_label.scoring.score_one", side_effect=mock_score_one):
        stats = asyncio.run(
            _run_scoring_file_chunked(
                labeled_path,
                tmp_path,
                None,
                0,
                _quiet_chunked_config(enable_stage_recovery_sweep=True, recovery_sweep_max_passes=1),
                resume=True,
                generate_dashboard=False,
                show_progress=False,
                print_summary=False,
                quiet=True,
            )
        )

    assert calls == 1
    assert _read_jsonl(tmp_path / "scored.jsonl")[0]["value"]["value_score"] == 8.7
    assert _read_jsonl(tmp_path / "monitor_value.jsonl")[0]["status"] == "success"
    assert not (tmp_path / "failed_value.jsonl").exists()
    assert not (tmp_path / "score_failures.jsonl").exists()
    assert stats["recovery_sweep"]["attempted"] == 1
    assert stats["recovery_sweep"]["recovered"] == 1


def test_resume_fastpath_rebuilds_failure_outputs_from_scored_monitor_after_interruption(tmp_path):
    labeled_path = tmp_path / "labeled.jsonl"
    samples = [_sample("sample-recovered"), _sample("sample-failed")]
    _write_jsonl(labeled_path, samples)

    _write_jsonl(
        _pass2_checkpoint_path(tmp_path, "scored.jsonl"),
        [
            {
                **samples[0],
                "value": _value_payload(score=7.9, rarity_score=5.3, confidence=0.9),
            },
            dict(samples[1], value=None),
        ],
    )
    _write_jsonl(
        _pass2_checkpoint_path(tmp_path, "monitor_value.jsonl"),
        [
            _monitor(samples[0]["id"], status="success"),
            {
                **_monitor(samples[1]["id"], status="failed", retryable_infra=False),
                "error": "invalid_request",
                "error_class": "input_error",
            },
        ],
    )

    # Simulate stale contradiction left by an interrupted prior sweep.
    _write_jsonl(
        tmp_path / "failed_value.jsonl",
        [dict(samples[0], value=None), dict(samples[1], value=None)],
    )
    _write_jsonl(
        tmp_path / "score_failures.jsonl",
        [
            {"sample_id": samples[0]["id"], "status": "stale-failure"},
            {"sample_id": samples[1]["id"], "status": "final-failure"},
        ],
    )

    async def fail_score_one(*_args, **_kwargs):
        raise AssertionError("score_one must not run during resume fast-path finalize")

    with patch("sft_label.scoring.score_one", side_effect=fail_score_one):
        asyncio.run(
            _run_scoring_file_chunked(
                labeled_path,
                tmp_path,
                None,
                0,
                _quiet_chunked_config(enable_stage_recovery_sweep=True, recovery_sweep_max_passes=1),
                resume=True,
                generate_dashboard=False,
                show_progress=False,
                print_summary=False,
                quiet=True,
            )
        )

    failed_rows = _read_jsonl(tmp_path / "failed_value.jsonl")
    failure_log_rows = _read_jsonl(tmp_path / "score_failures.jsonl")
    assert [row["id"] for row in failed_rows] == [samples[1]["id"]]
    assert [row["sample_id"] for row in failure_log_rows] == [samples[1]["id"]]


def test_chunked_resume_fastpath_honors_working_checkpoint_final_precedence(tmp_path):
    labeled_path = tmp_path / "labeled.jsonl"
    samples = [_sample(f"sample-{idx}") for idx in range(2)]
    _write_jsonl(labeled_path, samples)

    final_scored = [
        {
            **sample,
            "value": _value_payload(score=1.5, rarity_score=1.0, confidence=0.1),
        }
        for sample in samples
    ]
    final_monitors = [_monitor(sample["id"], status="final-source") for sample in samples]
    _write_jsonl(tmp_path / "scored.jsonl", final_scored)
    _write_jsonl(tmp_path / "monitor_value.jsonl", final_monitors)

    checkpoint_scored = [
        {
            **sample,
            "value": _value_payload(score=8.5, rarity_score=6.0, confidence=0.99),
        }
        for sample in samples
    ]
    checkpoint_monitors = [_monitor(sample["id"], status="checkpoint-source") for sample in samples]
    _write_jsonl(_pass2_checkpoint_path(tmp_path, "scored.jsonl"), checkpoint_scored)
    _write_jsonl(_pass2_checkpoint_path(tmp_path, "monitor_value.jsonl"), checkpoint_monitors)

    working_scored = [
        {
            **sample,
            "value": _value_payload(score=9.6, rarity_score=7.0, confidence=0.97),
        }
        for sample in samples
    ]
    working_monitors = [_monitor(sample["id"], status="working-source") for sample in samples]
    _write_jsonl(tmp_path / ".scored.jsonl.next", working_scored)
    _write_jsonl(tmp_path / ".monitor_value.jsonl.next", working_monitors)

    # Force freshness ordering opposite of precedence to ensure finalize order is authoritative.
    os.utime(tmp_path / ".scored.jsonl.next", (100, 100))
    os.utime(tmp_path / ".monitor_value.jsonl.next", (100, 100))
    os.utime(_pass2_checkpoint_path(tmp_path, "scored.jsonl"), (200, 200))
    os.utime(_pass2_checkpoint_path(tmp_path, "monitor_value.jsonl"), (200, 200))
    os.utime(tmp_path / "scored.jsonl", (300, 300))
    os.utime(tmp_path / "monitor_value.jsonl", (300, 300))

    async def fail_score_one(*_args, **_kwargs):
        raise AssertionError("score_one must not run during resume fast-path finalize")

    with patch("sft_label.scoring.score_one", side_effect=fail_score_one):
        asyncio.run(
            _run_scoring_file_chunked(
                labeled_path,
                tmp_path,
                None,
                0,
                _quiet_chunked_config(enable_stage_recovery_sweep=False),
                resume=True,
                generate_dashboard=False,
                show_progress=False,
                print_summary=False,
                quiet=True,
            )
        )

    final_scored_rows = _read_jsonl(tmp_path / "scored.jsonl")
    final_monitor_rows = _read_jsonl(tmp_path / "monitor_value.jsonl")
    assert {row["value"]["value_score"] for row in final_scored_rows} == {9.6}
    assert {row["status"] for row in final_monitor_rows} == {"working-source"}
