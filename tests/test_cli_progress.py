import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from sft_label.cli import (
    _CombinedLLMProgressTracker,
    _SemanticProgressPrinter,
    _estimate_end_to_end_llm_calls,
    build_parser,
    cmd_run,
)
from sft_label.config import PipelineConfig


def _write_input(path: Path, n: int):
    data = []
    for i in range(n):
        data.append(
            {
                "id": f"c-{i}",
                "conversations": [
                    {"from": "human", "value": "Write python code."},
                    {"from": "gpt", "value": "```python\nprint(1)\n```"},
                ],
            }
        )
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False)


def test_estimate_end_to_end_llm_calls_single_file(tmp_path):
    input_file = tmp_path / "input.json"
    _write_input(input_file, 3)

    args = SimpleNamespace(
        input=str(input_file),
        resume=None,
        limit=0,
        shuffle=False,
        no_arbitration=False,
    )
    config = PipelineConfig(sample_max_retries=1)
    plan = _estimate_end_to_end_llm_calls(args, config)

    assert plan is not None
    assert plan["pass1_labeled_samples"] == 3
    assert plan["pass2_samples"] == 3
    assert plan["pass1_est_calls"] == 7
    assert plan["pass2_est_calls"] == 3
    assert plan["total_est_calls"] == 10


def test_combined_llm_progress_tracker_updates():
    tracker = _CombinedLLMProgressTracker(100)
    info = tracker.update(12, "pass1")
    assert "run 12/100" in info
    summary = tracker.summary_line()
    assert "p1=12" in summary
    assert "p2=0" in summary


def test_cmd_run_semantic_cluster_failure_exits_cleanly(monkeypatch, capsys):
    async def _fake_run(*args, **kwargs):
        return {"run_dir": "/tmp/fake-run"}

    fake_pipeline = SimpleNamespace(run=_fake_run)

    def _fake_semantic(*args, **kwargs):
        raise ValueError("semantic failed")

    fake_semantic = SimpleNamespace(
        run_semantic_clustering=_fake_semantic,
        format_semantic_summary=lambda stats: "unused",
    )
    monkeypatch.setitem(sys.modules, "sft_label.pipeline", fake_pipeline)
    monkeypatch.setitem(sys.modules, "sft_label.semantic_clustering", fake_semantic)

    parser = build_parser()
    args = parser.parse_args(["run", "--input", "input.json", "--semantic-cluster"])
    with pytest.raises(SystemExit) as exc:
        cmd_run(args)
    assert exc.value.code == 1
    out = capsys.readouterr().out
    assert "Error: semantic failed" in out


def test_semantic_progress_printer_shows_progress_without_spam(capsys):
    printer = _SemanticProgressPrinter()
    printer("start", "Semantic clustering started", None, None)
    printer("embed", "Embedding windows", 1, 10)
    printer("embed", "Embedding windows", 1, 10)  # duplicate percent, suppressed
    printer("embed", "Embedding windows", 5, 10)

    lines = [line for line in capsys.readouterr().out.splitlines() if line.strip()]
    assert any("[semantic:start]" in line for line in lines)
    embed_lines = [line for line in lines if "[semantic:embed]" in line]
    assert len(embed_lines) == 2
