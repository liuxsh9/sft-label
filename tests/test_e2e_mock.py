"""
End-to-end test with mocked LLM calls.

Tests the full pipeline (Pass 1 → Pass 2 → Filter) using mock data,
verifying terminal output, dashboard generation, and labeling correctness.
"""

from __future__ import annotations

import asyncio
import json
import shutil
from pathlib import Path
from unittest.mock import patch

import pytest

from sft_label.artifacts import (
    PASS1_STATS_FILE,
    PASS2_STATS_FILE,
    PASS2_DASHBOARD_FILE,
    dashboard_relpath,
)
from sft_label.config import PipelineConfig

FIXTURES_DIR = Path(__file__).parent / "fixtures"

# ── Mock LLM responses ──────────────────────────────────────────────

CALL1_RESPONSES = {
    "default": {
        "intent": "build",
        "language": ["python"],
        "domain": ["web-backend"],
        "task": ["feature-implementation"],
        "difficulty": "intermediate",
        "confidence": {
            "intent": 0.95, "language": 0.99, "domain": 0.85,
            "task": 0.90, "difficulty": 0.80,
        },
        "unmapped": [],
    },
    "debug": {
        "intent": "debug",
        "language": ["python"],
        "domain": [],
        "task": ["debugging"],
        "difficulty": "intermediate",
        "confidence": {
            "intent": 0.95, "language": 0.99, "domain": 0.80,
            "task": 0.90, "difficulty": 0.75,
        },
        "unmapped": [],
    },
    "learn": {
        "intent": "learn",
        "language": ["python"],
        "domain": ["security"],
        "task": ["code-explanation"],
        "difficulty": "beginner",
        "confidence": {
            "intent": 0.99, "language": 0.99, "domain": 0.90,
            "task": 0.95, "difficulty": 0.95,
        },
        "unmapped": [],
    },
    "cpp": {
        "intent": "build",
        "language": ["cpp"],
        "domain": [],
        "task": ["feature-implementation"],
        "difficulty": "advanced",
        "confidence": {
            "intent": 0.95, "language": 0.99, "domain": 0.70,
            "task": 0.90, "difficulty": 0.85,
        },
        "unmapped": [],
    },
    "modify": {
        "intent": "modify",
        "language": ["python"],
        "domain": ["web-backend"],
        "task": ["code-refactoring"],
        "difficulty": "intermediate",
        "confidence": {
            "intent": 0.97, "language": 0.99, "domain": 0.85,
            "task": 0.95, "difficulty": 0.80,
        },
        "unmapped": [],
    },
}

CALL2_RESPONSE = {
    "concept": ["algorithms"],
    "agentic": [],
    "constraint": [],
    "context": "snippet",
    "confidence": {
        "concept": 0.90, "agentic": 0.80,
        "constraint": 0.80, "context": 0.95,
    },
    "unmapped": [],
}

SCORE_RESPONSE = {
    "complexity": {
        "instruction": 6, "analytical_depth": 7, "implementation": 6, "overall": 6,
    },
    "quality": {
        "correctness": 8, "code_quality": 7, "explanation": 7,
        "completeness": 7, "overall": 7,
    },
    "reasoning": {
        "clarity": 7, "consistency": 8, "self_correction": False, "overall": 7,
    },
    "flags": ["has-bug"],
    "confidence": 0.85,
}


def _pick_call1(messages):
    """Pick a call1 response based on the conversation content."""
    content = json.dumps(messages, ensure_ascii=False).lower()
    if "oom" in content or "debug" in content or "bug" in content:
        return CALL1_RESPONSES["debug"]
    if "区别" in content or "解释" in content or "sql 注入" in content:
        return CALL1_RESPONSES["learn"]
    if "c++" in content or "cpp" in content or "单例" in content:
        return CALL1_RESPONSES["cpp"]
    if "重构" in content or "refactor" in content:
        return CALL1_RESPONSES["modify"]
    return CALL1_RESPONSES["default"]


_call_counter = 0


async def mock_llm_call(http_client, messages, model,
                        temperature=0.1, max_tokens=1000,
                        max_retries=3, config=None, rate_limiter=None):
    """Mock async_llm_call that returns pre-built responses."""
    global _call_counter
    _call_counter += 1

    # Determine if this is a Call 1 or Call 2 / scoring call by inspecting messages
    msg_text = json.dumps(messages, ensure_ascii=False)
    is_call2 = "Call 1 结果" in msg_text or "call_1_result" in msg_text or "Call 1 result" in msg_text
    is_scoring = "complexity" in msg_text and "quality" in msg_text and "reasoning" in msg_text and "value" in msg_text.lower()

    if is_scoring:
        parsed = SCORE_RESPONSE
    elif is_call2:
        parsed = CALL2_RESPONSE
    else:
        parsed = _pick_call1(messages)

    raw = json.dumps(parsed, ensure_ascii=False)
    usage = {"prompt_tokens": 500, "completion_tokens": 200, "total_tokens": 700}

    # Small delay to simulate network
    await asyncio.sleep(0.001)
    return parsed, raw, usage


# ── Test fixtures ────────────────────────────────────────────────────

@pytest.fixture
def mock_config():
    """Minimal config for fast testing."""
    return PipelineConfig(
        concurrency=5,
        scoring_concurrency=5,
        sample_max_retries=1,
        max_retries=1,
        request_timeout=10,
        litellm_base="http://mock:4000/v1",
        litellm_key="sk-mock",
    )


@pytest.fixture
def single_file_input(tmp_path):
    """Copy smoke_test.json to tmp dir."""
    src = FIXTURES_DIR / "smoke_test.json"
    dst = tmp_path / "input.json"
    shutil.copy(src, dst)
    return dst


@pytest.fixture
def dir_input(tmp_path):
    """Create a directory with multiple small input files."""
    input_dir = tmp_path / "data"
    input_dir.mkdir()

    src = FIXTURES_DIR / "smoke_test.json"
    with open(src, "r", encoding="utf-8") as f:
        all_samples = json.load(f)

    # Split into 2 files
    file1 = all_samples[:3]
    file2 = all_samples[3:]

    with open(input_dir / "batch_a.json", "w", encoding="utf-8") as f:
        json.dump(file1, f, ensure_ascii=False, indent=2)
    with open(input_dir / "batch_b.json", "w", encoding="utf-8") as f:
        json.dump(file2, f, ensure_ascii=False, indent=2)

    return input_dir


@pytest.fixture
def scored_input(tmp_path):
    """Copy smoke_test_value.json to tmp dir for scoring."""
    src = FIXTURES_DIR / "smoke_test_value.json"
    dst = tmp_path / "labeled.json"
    shutil.copy(src, dst)
    return dst


# ── Tests ────────────────────────────────────────────────────────────

class TestE2ESingleFile:
    """End-to-end: single file → Pass 1 → verify outputs."""

    @pytest.mark.asyncio
    async def test_pass1_single_file(self, single_file_input, tmp_path, mock_config, capsys):
        from sft_label.pipeline import run

        output_dir = tmp_path / "output"

        with patch("sft_label.pipeline.async_llm_call", side_effect=mock_llm_call):
            stats = await run(
                input_path=str(single_file_input),
                output=str(output_dir),
                config=mock_config,
            )

        # ── Verify stats ──
        assert stats is not None
        assert stats["total_samples"] >= 5  # multi-turn slicing may produce more
        assert stats["success"] > 0
        assert stats["success_rate"] > 0.5

        # ── Verify output files ──
        run_dir = Path(stats["run_dir"])
        assert (run_dir / "labeled.json").exists()
        assert (run_dir / "labeled.jsonl").exists()
        assert (run_dir / PASS1_STATS_FILE).exists()
        assert (run_dir / "monitor.jsonl").exists()

        # ── Verify labeled data ──
        with open(run_dir / "labeled.json", "r", encoding="utf-8") as f:
            labeled = json.load(f)
        assert len(labeled) >= 5
        for sample in labeled:
            assert "labels" in sample
            labels = sample["labels"]
            assert "intent" in labels
            assert "language" in labels
            assert "difficulty" in labels
            assert "concept" in labels
            assert "context" in labels
            # Validate single-select are strings
            assert isinstance(labels["intent"], str)
            assert isinstance(labels["difficulty"], str)
            assert isinstance(labels["context"], str)
            # Validate multi-select are lists
            assert isinstance(labels["language"], list)
            assert isinstance(labels["concept"], list)

        # ── Verify dashboard ──
        dashboards = list(run_dir.glob("**/dashboard*.html"))
        assert len(dashboards) >= 1, "Dashboard HTML should be generated"

        # ── Verify terminal output is condensed ──
        captured = capsys.readouterr()
        # Header should be condensed single-line (no multi-line === banner before Pipeline)
        assert "Pipeline |" in captured.out
        # Summary should still appear
        assert "success" in captured.out.lower() or "Summary" in captured.out


class TestE2EDirectory:
    """End-to-end: directory → Pass 1 → verify outputs."""

    @pytest.mark.asyncio
    async def test_pass1_directory(self, dir_input, tmp_path, mock_config, capsys):
        from sft_label.pipeline import run

        output_dir = tmp_path / "output"

        with patch("sft_label.pipeline.async_llm_call", side_effect=mock_llm_call):
            stats = await run(
                input_path=str(dir_input),
                output=str(output_dir),
                config=mock_config,
            )

        # ── Verify stats ──
        assert stats is not None
        assert "total_samples" in stats or "per_file_summary" in stats

        # ── Verify output structure ──
        run_dir = Path(stats["run_dir"]) if "run_dir" in stats else output_dir
        assert run_dir.exists()
        dataset_root = run_dir / dir_input.name
        if dataset_root.is_dir():
            assert (run_dir / "meta_label_data").is_dir()
            mirrored_files = sorted(dataset_root.rglob("*.jsonl"))
            assert len(mirrored_files) >= 2, f"Expected mirrored jsonl files under {dataset_root}"
            for mirrored in mirrored_files:
                assert mirrored.read_text(encoding="utf-8").strip(), f"Mirrored file is empty: {mirrored}"
        else:
            subdirs = [d for d in run_dir.iterdir() if d.is_dir() and d.name != "meta_label_data"]
            assert len(subdirs) >= 2, f"Expected >=2 subdirs, got {len(subdirs)}: {subdirs}"
            for sub in subdirs:
                labeled_files = list(sub.glob("labeled*.json"))
                assert len(labeled_files) >= 1, f"No labeled files in {sub}"

        # ── Verify terminal output is clean ──
        captured = capsys.readouterr()
        # No per-file load messages
        assert "[File" not in captured.out
        # No per-file success messages
        assert "✓" not in captured.out or captured.out.count("✓") <= 1
        # No "SKIPPED" messages
        assert "SKIPPED" not in captured.out
        # Banner is condensed (header uses single-line format)
        assert "Pipeline |" in captured.out
        # No verbose conversation → samples messages
        assert "conversations →" not in captured.out


class TestE2EScoring:
    """End-to-end: Pass 2 scoring with mocked LLM."""

    @pytest.mark.asyncio
    async def test_pass2_scoring(self, scored_input, tmp_path, mock_config, capsys):
        from sft_label.scoring import run_scoring

        # Create minimal stats for rarity
        stats_data = {
            "total_samples": 100,
            "tag_distributions": {
                "intent": {"build": 30, "modify": 10, "debug": 20, "learn": 30, "review": 10},
                "language": {"python": 60, "javascript": 20, "cpp": 10, "rust": 5, "go": 5},
                "difficulty": {"beginner": 25, "intermediate": 40, "advanced": 25, "expert": 10},
                "concept": {"algorithms": 30, "data-structures": 20, "data-types": 15,
                            "memory-management": 10, "ownership": 5, "security": 10, "concurrency": 10},
                "domain": {"web-backend": 30, "security": 15, "devops": 10, "data-engineering": 20},
                "task": {"feature-implementation": 40, "debugging": 20, "code-explanation": 25,
                         "code-review": 15},
                "agentic": {"file-operations": 15, "multi-step-reasoning": 10},
                "constraint": {"thread-safe": 10},
                "context": {"snippet": 50, "single-function": 25, "single-file": 15, "multi-file": 10},
            },
        }
        stats_path = tmp_path / "stats.json"
        with open(stats_path, "w", encoding="utf-8") as f:
            json.dump(stats_data, f)

        with patch("sft_label.pipeline.async_llm_call", side_effect=mock_llm_call), \
             patch("sft_label.scoring.async_llm_call", side_effect=mock_llm_call):
            result = await run_scoring(
                input_path=str(scored_input),
                tag_stats_path=str(stats_path),
                config=mock_config,
            )

        # ── Verify result ──
        assert result is not None
        assert result.get("total_scored", 0) > 0

        # ── Verify output files ──
        scored_path = tmp_path / "scored.json"
        assert scored_path.exists()

        with open(scored_path, "r", encoding="utf-8") as f:
            scored = json.load(f)
        assert len(scored) >= 1

        for sample in scored:
            if "value" in sample:
                value = sample["value"]
                assert "value_score" in value
                assert "complexity" in value
                assert "quality" in value
                assert "rarity" in value
                assert 1 <= value["value_score"] <= 10
                # Selection score should be attached
                assert "selection_score" in value

        # ── Verify stats ──
        assert (tmp_path / PASS2_STATS_FILE).exists()

        # ── Verify dashboard ──
        assert (tmp_path / dashboard_relpath(PASS2_DASHBOARD_FILE)).exists()

        # ── Verify terminal output ──
        captured = capsys.readouterr()
        # No verbose setup lines
        assert "Pass A:" not in captured.out
        assert "Pass B:" not in captured.out
        assert "Endpoint:" not in captured.out
        # No "Computing selection scores..." line
        assert "Computing selection scores" not in captured.out
        # Scoring summary should appear
        assert "Scoring" in captured.out or "samples" in captured.out


class TestE2EContinuousMode:
    """End-to-end: Pass 1 + Pass 2 continuous mode."""

    @pytest.mark.asyncio
    async def test_continuous_single_file(self, single_file_input, tmp_path, mock_config, capsys):
        from sft_label.pipeline import run
        from sft_label.scoring import run_scoring

        output_dir = tmp_path / "output"

        # ── Pass 1 ──
        with patch("sft_label.pipeline.async_llm_call", side_effect=mock_llm_call):
            stats = await run(
                input_path=str(single_file_input),
                output=str(output_dir),
                config=mock_config,
            )

        run_dir = Path(stats["run_dir"])
        assert (run_dir / "labeled.json").exists()

        # ── Pass 2 ──
        with patch("sft_label.pipeline.async_llm_call", side_effect=mock_llm_call), \
             patch("sft_label.scoring.async_llm_call", side_effect=mock_llm_call):
            score_result = await run_scoring(
                input_path=str(run_dir / "labeled.json"),
                tag_stats_path=str(run_dir / PASS1_STATS_FILE),
                config=mock_config,
            )

        assert score_result is not None
        assert score_result.get("total_scored", 0) > 0

        # Scored output should exist
        scored_path = run_dir / "scored.json"
        assert scored_path.exists()

        with open(scored_path, "r", encoding="utf-8") as f:
            scored = json.load(f)
        assert len(scored) >= 5

        # ── Filter ──
        from sft_label.tools.filter_value import filter_samples, FilterConfig

        filter_config = FilterConfig(value_min=3.0)
        retained, dropped = filter_samples(scored, config=filter_config)
        assert len(retained) + len(dropped) == len(scored)
        # With mock scores, most should pass a low threshold
        assert len(retained) > 0

        # ── Verify terminal output ──
        captured = capsys.readouterr()
        assert "Pipeline |" in captured.out


class TestE2EDirectoryScoring:
    """End-to-end: directory → Pass 1 → Pass 2 directory scoring."""

    @pytest.mark.asyncio
    async def test_dir_pass1_then_pass2(self, dir_input, tmp_path, mock_config, capsys):
        from sft_label.pipeline import run
        from sft_label.scoring import run_scoring

        output_dir = tmp_path / "output"

        # ── Pass 1 ──
        with patch("sft_label.pipeline.async_llm_call", side_effect=mock_llm_call):
            stats = await run(
                input_path=str(dir_input),
                output=str(output_dir),
                config=mock_config,
            )

        run_dir = Path(stats["run_dir"])
        assert run_dir.exists()

        # ── Pass 2 on the run directory ──
        with patch("sft_label.pipeline.async_llm_call", side_effect=mock_llm_call), \
             patch("sft_label.scoring.async_llm_call", side_effect=mock_llm_call):
            score_result = await run_scoring(
                input_path=str(run_dir),
                config=mock_config,
            )

        assert score_result is not None

        # ── Verify per-file scored outputs ──
        scored_files = sorted(
            {
                *run_dir.glob("**/scored*.json"),
                *run_dir.glob("**/scored*.jsonl"),
            }
        )
        assert len(scored_files) >= 2, f"Expected >=2 scored files, got: {scored_files}"

        # ── Verify dashboards ──
        dashboards = list(run_dir.glob("**/dashboard*.html"))
        assert len(dashboards) >= 2, f"Expected >=2 dashboards, got: {dashboards}"

        # ── Verify terminal is clean ──
        captured = capsys.readouterr()
        # No per-file load/complete noise in directory mode
        assert "conversations →" not in captured.out
        assert "Tag stats loaded" not in captured.out
        assert "Computing selection scores" not in captured.out
