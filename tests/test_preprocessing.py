"""Tests for the preprocessing module."""

import json
from pathlib import Path

from sft_label.preprocessing import (
    detect_format,
    normalize_pangu,
    normalize_and_slice,
    slice_multiturn,
    strip_cot,
    detect_code_fence_languages,
    detect_framework_languages,
    count_code_blocks,
    extract_tool_signals,
    detect_behavioral_patterns,
    preprocess,
    format_signals_for_prompt,
    generate_sparse_schedule,
    apply_sparse_sampling,
    truncate_conversations_for_labeling,
)


FIXTURES_DIR = Path(__file__).parent / "fixtures"


class TestFormatDetection:
    def test_sharegpt_format(self):
        sample = {"conversations": [{"from": "human", "value": "hi"}]}
        assert detect_format(sample) == "sharegpt"

    def test_pangu_format(self):
        sample = {"data": [{"role": "user", "content": "hi"}]}
        assert detect_format(sample) == "pangu"

    def test_unknown_format(self):
        sample = {"messages": []}
        assert detect_format(sample) == "unknown"


class TestStripCot:
    def test_pangu_cot(self):
        text = "hello [unused16]thinking stuff[unused17] world"
        assert strip_cot(text) == "hello  world"

    def test_sharegpt_cot(self):
        text = "hello <think>internal thought</think> world"
        assert strip_cot(text) == "hello  world"

    def test_no_cot(self):
        text = "hello world"
        assert strip_cot(text) == "hello world"


class TestSliceMultiturn:
    def test_single_turn(self):
        convs = [
            {"from": "human", "value": "q"},
            {"from": "gpt", "value": "a"},
        ]
        result = slice_multiturn(convs)
        assert len(result) == 1
        assert result[0] == convs

    def test_multi_turn(self):
        convs = [
            {"from": "human", "value": "q1"},
            {"from": "gpt", "value": "a1"},
            {"from": "human", "value": "q2"},
            {"from": "gpt", "value": "a2"},
        ]
        result = slice_multiturn(convs)
        assert len(result) == 2
        assert len(result[0]) == 2  # [q1, a1]
        assert len(result[1]) == 4  # [q1, a1, q2, a2]


class TestNormalizePangu:
    def test_basic_normalization(self):
        sample = {
            "id": "test-1",
            "data": [
                {"role": "user", "content": "hello"},
                {"role": "assistant", "content": "hi there"},
            ],
        }
        result = normalize_pangu(sample)
        assert result["id"] == "test-1"
        assert len(result["conversations"]) == 2
        assert result["conversations"][0]["from"] == "human"
        assert result["conversations"][1]["from"] == "gpt"

    def test_pseudo_multiturn(self):
        sample = {
            "id": "test-2",
            "data": [
                {"role": "user", "content": "hello [unused10] world"},
                {"role": "assistant", "content": "response"},
            ],
        }
        result = normalize_pangu(sample)
        assert result["metadata"]["is_pseudo_multiturn"] is True


class TestNormalizeAndSlice:
    def test_single_turn_sharegpt(self):
        sample = {
            "id": "s1",
            "conversations": [
                {"from": "human", "value": "q"},
                {"from": "gpt", "value": "a"},
            ],
        }
        result = normalize_and_slice(sample)
        assert len(result) == 1

    def test_multi_turn_creates_slices(self):
        sample = {
            "id": "m1",
            "conversations": [
                {"from": "human", "value": "q1"},
                {"from": "gpt", "value": "a1"},
                {"from": "human", "value": "q2"},
                {"from": "gpt", "value": "a2"},
            ],
        }
        result = normalize_and_slice(sample)
        assert len(result) == 2
        assert result[0]["id"] == "m1_t1"
        assert result[1]["id"] == "m1_t2"


class TestLanguageDetection:
    def test_code_fence_python(self):
        text = "```python\nprint('hello')\n```"
        assert "python" in detect_code_fence_languages(text)

    def test_code_fence_multiple(self):
        text = "```typescript\nconst x = 1;\n```\n```python\nx = 1\n```"
        langs = detect_code_fence_languages(text)
        assert "typescript" in langs
        assert "python" in langs

    def test_framework_detection(self):
        text = "I'm using React and Django"
        langs = detect_framework_languages(text)
        assert "typescript" in langs
        assert "python" in langs


class TestCodeBlocks:
    def test_count(self):
        text = "```python\ncode\n```\ntext\n```js\nmore\n```"
        assert count_code_blocks(text) == 2


class TestToolSignals:
    def test_bash_detection(self):
        convs = [
            {"from": "tool", "value": "$ ls -la\ntotal 0"},
        ]
        names, tags = extract_tool_signals(convs)
        assert "bash" in names
        assert "bash-execution" in tags

    def test_no_tools(self):
        convs = [
            {"from": "human", "value": "hi"},
            {"from": "gpt", "value": "hello"},
        ]
        names, tags = extract_tool_signals(convs)
        assert len(names) == 0
        assert len(tags) == 0


class TestSparseSchedule:
    def test_small_n(self):
        # Below threshold: all labeled
        schedule = generate_sparse_schedule(5)
        assert schedule == [0, 1, 2, 3, 4]

    def test_at_threshold(self):
        schedule = generate_sparse_schedule(12)
        assert schedule == list(range(12))

    def test_large_n_includes_first_and_last(self):
        schedule = generate_sparse_schedule(50)
        assert schedule[0] == 0
        assert schedule[-1] == 49
        assert len(schedule) < 50  # Some were skipped


class TestApplySparseSampling:
    def test_single_turn_always_labeled(self):
        samples = [{"id": "s1", "metadata": {}}]
        label_indices, inherit_map = apply_sparse_sampling(samples)
        assert 0 in label_indices
        assert len(inherit_map) == 0

    def test_multi_turn_small(self):
        # 3 slices from same source — all labeled (below threshold)
        samples = [
            {"id": "m_t1", "metadata": {"source_id": "m", "turn_index": 1, "total_turns": 3}},
            {"id": "m_t2", "metadata": {"source_id": "m", "turn_index": 2, "total_turns": 3}},
            {"id": "m_t3", "metadata": {"source_id": "m", "turn_index": 3, "total_turns": 3}},
        ]
        label_indices, inherit_map = apply_sparse_sampling(samples)
        assert label_indices == {0, 1, 2}
        assert len(inherit_map) == 0


class TestTruncation:
    def test_no_truncation_needed(self):
        convs = [
            {"from": "human", "value": "short"},
            {"from": "gpt", "value": "also short"},
        ]
        result, truncated = truncate_conversations_for_labeling(convs)
        assert not truncated
        assert result == convs

    def test_truncation_applied(self):
        convs = [
            {"from": "human", "value": "x" * 15000},
            {"from": "gpt", "value": "y" * 15000},
        ]
        result, truncated = truncate_conversations_for_labeling(convs, max_total_chars=5000)
        assert truncated
        total = sum(len(t["value"]) for t in result)
        assert total <= 5500  # Allow some margin


class TestPreprocess:
    def test_basic_preprocess(self):
        sample = {
            "conversations": [
                {"from": "human", "value": "Write a Python function"},
                {"from": "gpt", "value": "```python\ndef hello():\n    pass\n```"},
            ],
        }
        signals = preprocess(sample)
        assert "python" in signals["detected_languages"]
        assert signals["total_turns"] == 2
        assert signals["code_block_count"] == 1

    def test_format_signals(self):
        signals = {
            "detected_languages": ["python"],
            "fence_languages": ["python"],
            "framework_languages": [],
            "has_tool_roles": False,
            "tool_names": [],
            "tool_agentic_tags": [],
            "behavioral_patterns": [],
            "total_turns": 2,
            "code_block_count": 1,
            "est_tokens": 100,
            "keyword_hits": [],
            "last_query_preview": "test",
            "last_response_length": 50,
        }
        result = format_signals_for_prompt(signals)
        assert "detected_languages" in result
        assert "python" in result


class TestPanguFixtures:
    """Test with actual Pangu fixture data."""

    def test_load_and_normalize(self):
        fixture_path = FIXTURES_DIR / "pangu_test_samples.jsonl"
        if not fixture_path.exists():
            return  # Skip if fixture not available

        samples = []
        with open(fixture_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    raw = json.loads(line)
                    samples.extend(normalize_and_slice(raw))

        assert len(samples) > 0
        for s in samples:
            assert "conversations" in s
            assert len(s["conversations"]) > 0
