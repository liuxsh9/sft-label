"""Tests for the OpenAI-turn compatibility probe script."""

import sys
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path


SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "probe_openai_turn_compat.py"
SPEC = spec_from_file_location("probe_openai_turn_compat", SCRIPT_PATH)
probe_module = module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = probe_module
SPEC.loader.exec_module(probe_module)


def test_categorize_row_uses_pipeline_resolved_turn_source():
    row = {
        "conversations": [{}],
        "messages": [
            {"role": "user", "content": "q"},
            {"role": "assistant", "content": "a"},
        ],
    }

    categories = probe_module._categorize_row(row)

    assert "single_turn" in categories
    assert "zero_assistant" not in categories


def test_iter_rows_supports_json_arrays(tmp_path):
    path = tmp_path / "sample.json"
    path.write_text(
        '[{"id":"a","messages":[{"role":"user","content":"q"},{"role":"assistant","content":"a"}]},'
        '{"id":"b","messages":[{"role":"user","content":"q2"},{"role":"assistant","content":"a2"}]}]',
        encoding="utf-8",
    )

    rows = list(probe_module._iter_rows(path, max_rows=10))

    assert [row_number for row_number, _ in rows] == [1, 2]
    assert [row["id"] for _, row in rows] == ["a", "b"]


def test_probe_candidate_safely_preserves_errors():
    candidate = probe_module.Candidate(
        category="single_turn",
        source_file=Path("/tmp/bad.jsonl"),
        row_number=7,
        raw_row={"messages": [{"role": "user", "content": ["bad"]}]},
    )

    result = probe_module._probe_candidate_safely(candidate)

    assert result["category"] == "single_turn"
    assert result["row_number"] == 7
    assert "error" in result
    assert "content" in result["error"]
