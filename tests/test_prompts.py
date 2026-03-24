from sft_label.prompts import CALL1_FEWSHOT_COMPACT, CALL1_SYSTEM_COMPACT


COMPACT_CALL1_SYSTEM_BASELINE_LEN = 9257
COMPACT_CALL1_FEWSHOT_BASELINE_TOTAL_CHARS = 2784


def test_call1_compact_prompt_keeps_empty_output_guidance_without_growth():
    assert 'never use "none"' in CALL1_SYSTEM_COMPACT
    assert 'Multi-select=[] when none' in CALL1_SYSTEM_COMPACT
    assert 'single-select="" when none' in CALL1_SYSTEM_COMPACT
    assert "process words like search/explore/analyze do not define intent" in CALL1_SYSTEM_COMPACT
    assert 'never output placeholders like "unspecified", "unknown", or "no specific' in CALL1_SYSTEM_COMPACT
    assert len(CALL1_SYSTEM_COMPACT) <= COMPACT_CALL1_SYSTEM_BASELINE_LEN


def test_call1_compact_fewshot_keeps_output_prediction_anchor_without_growth():
    assert len(CALL1_FEWSHOT_COMPACT) == 10
    assert any("输出什么" in msg["content"] for msg in CALL1_FEWSHOT_COMPACT if msg["role"] == "user")
    assert sum(len(msg["content"]) for msg in CALL1_FEWSHOT_COMPACT) <= COMPACT_CALL1_FEWSHOT_BASELINE_TOTAL_CHARS
