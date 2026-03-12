from __future__ import annotations

import json
from pathlib import Path

from sft_label.artifacts import PASS2_STATS_FILE, PASS2_SUMMARY_STATS_FILE
from sft_label.tools.visualize_value import generate_value_dashboard


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n",
        encoding="utf-8",
    )


def _scored_sample(
    sample_id: str,
    source_file: str,
    *,
    value: float,
    quality: float,
    intent: str = "debug",
    source_id: str | None = None,
) -> dict:
    return {
        "id": sample_id,
        "conversations": [
            {"from": "human", "value": f"question for {sample_id}"},
            {"from": "gpt", "value": f"answer for {sample_id}"},
        ],
        "metadata": {
            "source_file": source_file,
            "thinking_mode": "fast",
            "source_id": source_id or sample_id,
        },
        "labels": {
            "intent": intent,
            "language": ["python"],
            "task": ["bug-fixing"],
            "difficulty": "intermediate",
            "context": "single-file",
        },
        "value": {
            "value_score": value,
            "selection_score": round(value + 0.5, 1),
            "thinking_mode": "fast",
            "confidence": 0.74,
            "flags": ["low-correctness"] if quality < 4 else [],
            "quality": {"overall": quality},
            "complexity": {"overall": 6.0},
            "reasoning": {"overall": 5.0},
            "rarity": {"score": 4.5},
        },
    }


def test_generate_value_dashboard_single_scope_writes_explorer_assets(tmp_path: Path) -> None:
    samples = [
        _scored_sample("sample-1", "train.jsonl", value=3.5, quality=2.0, source_id="conv-1"),
        _scored_sample("sample-2", "train.jsonl", value=7.5, quality=8.0, intent="build", source_id="conv-2"),
    ]
    _write_jsonl(tmp_path / "scored.jsonl", samples)
    _write_json(
        tmp_path / "conversation_scores.json",
        [
            {"conversation_id": "conv-1", "conversation_key": "train.jsonl::conv-1", "source_file": "train.jsonl", "turn_count": 4, "conv_value": 6.8, "conv_selection": 7.1, "peak_complexity": 7, "conv_rarity": 6.3, "observed_turn_ratio": 0.5, "inherited_turn_ratio": 0.5, "rarity_confidence": 0.62},
            {"conversation_id": "conv-2", "conversation_key": "train.jsonl::conv-2", "source_file": "train.jsonl", "turn_count": 1, "conv_value": 7.9, "conv_selection": 8.3, "peak_complexity": 6, "conv_rarity": 7.7, "observed_turn_ratio": 1.0, "inherited_turn_ratio": 0.0, "rarity_confidence": 0.95},
        ],
    )
    _write_json(
        tmp_path / PASS2_STATS_FILE,
        {
            "input_file": "train.jsonl",
            "total_scored": 2,
            "total_failed": 0,
            "total_tokens": 24,
            "score_distributions": {
                "value_score": {"mean": 5.5, "min": 3.5, "max": 7.5},
                "quality_overall": {"mean": 5.0, "min": 2.0, "max": 8.0},
                "selection_score": {"mean": 6.0, "min": 4.0, "max": 8.0},
                "confidence": {"mean": 0.74, "min": 0.74, "max": 0.74},
                "rarity_score": {"mean": 4.5, "min": 4.5, "max": 4.5},
                "complexity_overall": {"mean": 6.0, "min": 6.0, "max": 6.0},
            },
            "histograms": {
                "value_score": [0, 0, 1, 0, 0, 0, 1, 0, 0, 0],
            },
            "thinking_mode_stats": {"fast": {"count": 2, "mean_value": 5.5, "mean_quality": 5.0, "mean_reasoning": 5.0}},
            "flag_counts": {"low-correctness": 1},
            "flag_value_impact": {"low-correctness": {"mean_value": 3.5, "count": 1}},
            "selection_thresholds": {"top_25pct": {"threshold": 8.0, "count": 1}},
            "coverage_at_thresholds": {"5.0": {"retained": 1, "pct": 0.5, "coverage": 0.5}},
            "value_by_tag": {},
            "selection_by_tag": {},
            "weights_used": {"quality": 0.4},
        },
    )

    dashboard_path = generate_value_dashboard(tmp_path, scored_file="scored.jsonl")
    html = dashboard_path.read_text(encoding="utf-8")

    assert dashboard_path.exists()
    assert "Sample Explorer" in html
    assets_dir = dashboard_path.with_suffix("").with_name(f"{dashboard_path.stem}.assets")
    assert assets_dir.exists()
    assert any(path.name.startswith("preview_") for path in assets_dir.iterdir())
    assert any(path.name.startswith("detail_") for path in assets_dir.iterdir())
    assert '"sample_count": 2' in html
    assert "Selection Thresholds" in html
    assert "Coverage At Thresholds" in html
    assert "Flag Impact" in html
    assert 'data-explorer-patch=' in html
    assert 'id="explorer-min-value"' in html
    assert 'id="explorer-flag-query"' in html
    assert 'id="explorer-conv-value-min"' in html
    assert 'id="explorer-turn-count-min"' in html
    assert 'id="explorer-conv-selection-max"' in html
    assert 'id="explorer-observed-turn-ratio-min"' in html
    assert 'id="explorer-observed-turn-ratio-max"' in html
    assert 'id="explorer-rarity-confidence-min"' in html
    assert 'id="explorer-rarity-confidence-max"' in html
    assert 'id="explorer-source-path"' in html
    assert 'id="explorer-max-selection"' in html
    assert 'candidate files' in html
    assert "stream large files chunk-by-chunk" in html
    assert "Preview limited to first" in html
    assert "Raw JSON Preview" in html
    assert "Observed Turn Coverage" in html
    assert "Rarity Confidence" in html
    assert "Low Coverage" in html
    assert "Low Rarity Conf" in html
    assert "EXPLORER_PREVIEW_CACHE_LIMIT = 6" in html
    assert "EXPLORER_DETAIL_CACHE_LIMIT = 18" in html
    assert "restoreDashboardState();" in html
    assert 'window.addEventListener("hashchange"' in html
    preview_asset = next(path for path in assets_dir.iterdir() if path.name.startswith("preview_"))
    preview_text = preview_asset.read_text(encoding="utf-8")
    assert '"conv_value": 6.8' in preview_text
    assert '"turn_count": 4' in preview_text
    assert '"observed_turn_ratio": 0.5' in preview_text
    assert '"rarity_confidence": 0.62' in preview_text


def test_generate_value_dashboard_tree_attaches_file_scope_explorer(tmp_path: Path) -> None:
    meta_root = tmp_path / "meta_label_data"
    artifact_dir = meta_root / "files" / "code" / "sample"
    _write_jsonl(
        artifact_dir / "scored.jsonl",
        [_scored_sample("sample-1", "code/sample.jsonl", value=4.0, quality=3.0, source_id="conv-1")],
    )
    _write_json(
        artifact_dir / "conversation_scores.json",
        [
            {"conversation_id": "conv-1", "conversation_key": "code/sample.jsonl::conv-1", "source_file": "code/sample.jsonl", "turn_count": 3, "conv_value": 6.5, "conv_selection": 6.9, "peak_complexity": 7, "conv_rarity": 6.2, "observed_turn_ratio": 0.67, "inherited_turn_ratio": 0.33, "rarity_confidence": 0.71},
        ],
    )
    _write_json(
        artifact_dir / PASS2_STATS_FILE,
        {
            "input_file": "code/sample.jsonl",
            "total_scored": 1,
            "total_failed": 0,
            "total_tokens": 10,
            "score_distributions": {
                "value_score": {"mean": 4.0, "min": 4.0, "max": 4.0},
                "quality_overall": {"mean": 3.0, "min": 3.0, "max": 3.0},
                "selection_score": {"mean": 4.5, "min": 4.5, "max": 4.5},
                "confidence": {"mean": 0.74, "min": 0.74, "max": 0.74},
            },
            "histograms": {},
            "thinking_mode_stats": {},
            "flag_counts": {},
            "value_by_tag": {},
            "selection_by_tag": {},
        },
    )
    _write_json(
        meta_root / PASS2_SUMMARY_STATS_FILE,
        {
            "input_path": str(tmp_path / "dataset"),
            "total_scored": 1,
            "total_failed": 0,
            "score_distributions": {
                "value_score": {"mean": 4.0, "min": 4.0, "max": 4.0},
                "quality_overall": {"mean": 3.0, "min": 3.0, "max": 3.0},
                "selection_score": {"mean": 4.5, "min": 4.5, "max": 4.5},
                "confidence": {"mean": 0.74, "min": 0.74, "max": 0.74},
            },
            "histograms": {},
            "flag_counts": {},
            "per_file_summary": [
                {
                    "file": "code/sample.jsonl",
                    "count": 1,
                    "mean_value": 4.0,
                    "mean_complexity": 6.0,
                    "mean_quality": 3.0,
                    "mean_selection": 4.5,
                }
            ],
        },
    )

    dashboard_path = generate_value_dashboard(meta_root, scored_file=None, stats_file=PASS2_SUMMARY_STATS_FILE)
    html = dashboard_path.read_text(encoding="utf-8")

    assert dashboard_path.exists()
    assert "file:code/sample.jsonl" in html
    assert '"explorer"' in html
    assert "Conv Value" in html
    assert "Observed Turn Coverage" in html
    assert "Source File" in html
    assets_dir = dashboard_path.with_suffix("").with_name(f"{dashboard_path.stem}.assets")
    assert any(path.name.startswith("preview_") for path in assets_dir.iterdir())
