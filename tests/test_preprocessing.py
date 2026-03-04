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
    to_pangu_pseudo_multiturn,
    sanitize_training_markers,
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

    def test_truncation_sanitizes_xml_tags(self):
        convs = [
            {"from": "human", "value": "Fix the bug <solution>hint</solution>"},
            {"from": "gpt", "value": "Here is the <fix attr='v'>patch</fix>"},
        ]
        result, _ = truncate_conversations_for_labeling(convs)
        assert "<solution>" not in result[0]["value"]
        assert "<fix" not in result[1]["value"]
        assert "hint" in result[0]["value"]
        assert "patch" in result[1]["value"]


class TestSanitizeTrainingMarkers:
    def test_strips_basic_tags(self):
        text = "Before <solution>code here</solution> after"
        assert sanitize_training_markers(text) == "Before code here after"

    def test_strips_self_closing_with_space(self):
        text = "text <fix /> more"
        assert sanitize_training_markers(text) == "text  more"

    def test_strips_tags_with_attributes(self):
        text = '<tool_call name="edit">content</tool_call>'
        assert sanitize_training_markers(text) == "content"

    def test_case_insensitive(self):
        text = "<SOLUTION>code</SOLUTION>"
        assert sanitize_training_markers(text) == "code"

    def test_preserves_non_target_tags(self):
        text = "<div>html</div> <span>content</span>"
        assert sanitize_training_markers(text) == text

    def test_preserves_code_angle_brackets(self):
        text = "if (a < b && c > d) { fix(); }"
        assert sanitize_training_markers(text) == text

    def test_empty_input(self):
        assert sanitize_training_markers("") == ""

    def test_nested_tags(self):
        text = "<solution><code_change>new code</code_change></solution>"
        assert sanitize_training_markers(text) == "new code"

    def test_extended_tags(self):
        """Tags added based on expert review: command, output, result, reasoning, response."""
        text = "<command>ls -la</command> <output>file.txt</output>"
        assert sanitize_training_markers(text) == "ls -la file.txt"


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


class TestToPanguPseudoMultiturn:
    """Tests for to_pangu_pseudo_multiturn() conversion."""

    def test_single_turn_no_cot(self):
        """Single turn without COT → [unused16][unused17] prefix."""
        sample = {
            "conversations": [
                {"from": "human", "value": "Write hello world"},
                {"from": "gpt", "value": "print('hello')"},
            ],
            "metadata": {},
        }
        result = to_pangu_pseudo_multiturn(sample)
        assert len(result["data"]) == 2
        assert result["data"][0]["role"] == "user"
        assert result["data"][0]["content"] == "Write hello world"
        assert result["data"][1]["role"] == "assistant"
        assert result["data"][1]["content"] == "[unused16][unused17]print('hello')"

    def test_single_turn_with_cot(self):
        """Single turn with COT → COT restored between [unused16] and [unused17]."""
        sample = {
            "conversations": [
                {"from": "human", "value": "Solve this"},
                {"from": "gpt", "value": "The answer is 42"},
            ],
            "metadata": {"cot_text": "Let me think about this step by step"},
        }
        result = to_pangu_pseudo_multiturn(sample)
        assert result["data"][1]["content"] == (
            "[unused16]Let me think about this step by step[unused17]The answer is 42"
        )

    def test_sliced_sample_no_cot_restored(self):
        """Sliced sample (source_id present) → fast thinking, COT not restored."""
        sample = {
            "conversations": [
                {"from": "human", "value": "Q1"},
                {"from": "gpt", "value": "A1"},
            ],
            "metadata": {
                "source_id": "original-123",
                "cot_text": "This COT is from aggregation, not this turn",
            },
        }
        result = to_pangu_pseudo_multiturn(sample)
        assert result["data"][1]["content"] == "[unused16][unused17]A1"

    def test_multi_turn_packing(self):
        """Multi-turn → prior turns packed with separators and role labels."""
        sample = {
            "conversations": [
                {"from": "human", "value": "Hello"},
                {"from": "gpt", "value": "Hi there"},
                {"from": "human", "value": "Write code"},
                {"from": "gpt", "value": "Here's the code"},
            ],
            "metadata": {},
        }
        result = to_pangu_pseudo_multiturn(sample)
        assert len(result["data"]) == 2

        user_content = result["data"][0]["content"]
        # Prior turns should be packed
        assert "用户：Hello /no_think" in user_content
        assert "助手：[unused16][unused17]Hi there" in user_content
        assert "[unused10][unused9]" in user_content
        # Last user turn
        assert "用户：Write code" in user_content
        # Last assistant response
        assert result["data"][1]["content"] == "[unused16][unused17]Here's the code"

    def test_system_prompt_preserved(self):
        """System prompt from metadata → meta_prompt in output."""
        sample = {
            "conversations": [
                {"from": "human", "value": "Hi"},
                {"from": "gpt", "value": "Hello"},
            ],
            "metadata": {"system_prompt": "You are a helpful assistant"},
        }
        result = to_pangu_pseudo_multiturn(sample)
        assert result["meta_prompt"] == "You are a helpful assistant"

    def test_tools_preserved(self):
        """Tool definitions from metadata → tools in output."""
        sample = {
            "conversations": [
                {"from": "human", "value": "Search"},
                {"from": "gpt", "value": "Found it"},
            ],
            "metadata": {"tool_definitions": '[{"name": "search"}]'},
        }
        result = to_pangu_pseudo_multiturn(sample)
        assert result["tools"] == '[{"name": "search"}]'

    def test_empty_conversations(self):
        """Empty conversations → empty output."""
        sample = {"conversations": [], "metadata": {}}
        result = to_pangu_pseudo_multiturn(sample)
        assert result["data"] == []

    def test_no_assistant_reply(self):
        """Only human turns → just user message."""
        sample = {
            "conversations": [{"from": "human", "value": "Question?"}],
            "metadata": {},
        }
        result = to_pangu_pseudo_multiturn(sample)
        assert len(result["data"]) == 1
        assert result["data"][0]["role"] == "user"

    def test_roundtrip_single_turn(self):
        """Pangu → normalize → to_pangu preserves essential structure."""
        original = {
            "id": "test-1",
            "data": [
                {"role": "user", "content": "Explain Python"},
                {"role": "assistant", "content": "[unused16]Let me think[unused17]Python is great"},
            ],
            "meta_prompt": "You are helpful",
            "tools": "",
        }
        normalized = normalize_pangu(original)
        reconstructed = to_pangu_pseudo_multiturn(normalized)

        assert reconstructed["meta_prompt"] == "You are helpful"
        assert len(reconstructed["data"]) == 2
        # COT should be restored
        assert "[unused16]Let me think[unused17]" in reconstructed["data"][1]["content"]
        assert "Python is great" in reconstructed["data"][1]["content"]

