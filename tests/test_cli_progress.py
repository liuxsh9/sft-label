import json
from pathlib import Path
from types import SimpleNamespace

from sft_label.cli import _CombinedLLMProgressTracker, _estimate_end_to_end_llm_calls
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
    assert "global=12/100" in info
    assert "p1=12" in info
    assert "p2=0" in info
