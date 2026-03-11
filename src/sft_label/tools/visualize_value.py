"""Interactive Pass 2 dashboard generation."""

from __future__ import annotations

import json
from pathlib import Path

from sft_label.artifacts import (
    PASS1_STATS_FILE,
    PASS1_SUMMARY_STATS_FILE,
    PASS2_STATS_FILE,
    PASS2_SUMMARY_STATS_FILE,
    resolve_dashboard_output,
)
from sft_label.tools.dashboard_scopes import build_scope_tree
from sft_label.tools.dashboard_template import render_dashboard_html
from sft_label.tools.visualize_labels import _dashboard_subtitle, compute_viz_data, load_run


def load_value_run(run_dir, scored_file="scored.json", stats_file=PASS2_STATS_FILE):
    """Load scored samples and Pass 2 stats from a run directory."""
    run_dir = Path(run_dir)
    samples = []

    if scored_file:
        scored_path = run_dir / scored_file
        if scored_path.exists():
            if scored_path.suffix == ".jsonl":
                with open(scored_path, encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if line:
                            try:
                                samples.append(json.loads(line))
                            except json.JSONDecodeError:
                                continue
            else:
                with open(scored_path, encoding="utf-8") as f:
                    samples = json.load(f)

    stats = {}
    stats_path = run_dir / stats_file
    if stats_path.exists():
        with open(stats_path, encoding="utf-8") as f:
            stats = json.load(f)

    return samples, stats


def compute_value_viz_data(samples, stats):
    """Transform Pass 2 data into dashboard JSON."""
    total_scored = stats.get("total_scored", len([sample for sample in samples if sample.get("value")]))
    total_failed = stats.get("total_failed", 0)
    distributions = stats.get("score_distributions", {})

    viz = {
        "overview": {
            "total_scored": total_scored,
            "total_failed": total_failed,
            "input_file": stats.get("input_file") or stats.get("input_path", ""),
            "mean_value": distributions.get("value_score", {}).get("mean", 0),
            "mean_complexity": distributions.get("complexity_overall", {}).get("mean", 0),
            "mean_quality": distributions.get("quality_overall", {}).get("mean", 0),
            "median_rarity": distributions.get("rarity_score", {}).get("p50", distributions.get("rarity_score", {}).get("mean", 0)),
            "mean_confidence": distributions.get("confidence", {}).get("mean", 0),
            "total_tokens": stats.get("total_tokens", 0),
        },
        "selection_thresholds": stats.get("selection_thresholds", {}),
        "score_distributions": distributions,
        "sub_score_means": stats.get("sub_score_means", {}),
        "value_by_tag": stats.get("value_by_tag", {}),
        "selection_by_tag": stats.get("selection_by_tag", {}),
        "thinking_mode_stats": stats.get("thinking_mode_stats", {}),
        "flag_counts": stats.get("flag_counts", {}),
        "flag_value_impact": stats.get("flag_value_impact", {}),
        "coverage_at_thresholds": stats.get("coverage_at_thresholds", {}),
        "weights_used": stats.get("weights_used", {}),
        "per_file_summary": stats.get("per_file_summary", []),
    }

    score_keys = [
        "value_score",
        "complexity_overall",
        "quality_overall",
        "reasoning_overall",
        "rarity_score",
        "selection_score",
    ]
    histograms = {}
    if samples:
        for key in score_keys:
            bins = [0] * 10
            for sample in samples:
                value = sample.get("value", {})
                if key == "value_score":
                    score = value.get("value_score")
                elif key == "rarity_score":
                    score = (value.get("rarity") or {}).get("score")
                elif key == "selection_score":
                    score = value.get("selection_score")
                else:
                    dim, sub = key.rsplit("_", 1)
                    score = (value.get(dim) or {}).get(sub)
                if isinstance(score, (int, float)) and 1 <= score <= 10:
                    bins[min(int(score) - 1, 9)] += 1
            histograms[key] = bins
    else:
        histograms = stats.get("histograms", {})
    viz["histograms"] = histograms

    confidence_histogram = [0] * 10
    if samples:
        for sample in samples:
            confidence = (sample.get("value") or {}).get("confidence")
            if isinstance(confidence, (int, float)) and 0 <= confidence <= 1:
                confidence_histogram[min(int(confidence * 10), 9)] += 1
    else:
        confidence_histogram = stats.get("confidence_histogram", confidence_histogram)
    viz["confidence_histogram"] = confidence_histogram

    selection_vs_value = {}
    for sample in samples:
        value = sample.get("value", {})
        value_score = value.get("value_score")
        selection_score = value.get("selection_score")
        if isinstance(value_score, (int, float)) and isinstance(selection_score, (int, float)):
            if 1 <= value_score <= 10 and 1 <= selection_score <= 10:
                key = f"{max(1, min(int(value_score), 10))}|{max(1, min(int(selection_score), 10))}"
                selection_vs_value[key] = selection_vs_value.get(key, 0) + 1
    viz["selection_vs_value"] = selection_vs_value

    return viz


def _load_conversation_scores(run_dir):
    path = Path(run_dir) / "conversation_scores.json"
    if not path.exists():
        return []
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError):
        return []


def _compute_conv_viz_data(conv_records):
    if not conv_records:
        return None

    total = len(conv_records)
    values = [row["conv_value"] for row in conv_records if row.get("conv_value") is not None]
    selections = [row["conv_selection"] for row in conv_records if row.get("conv_selection") is not None]
    peaks = [row["peak_complexity"] for row in conv_records if row.get("peak_complexity") is not None]
    turns = [row.get("turn_count", 0) for row in conv_records]

    payload = {
        "total": total,
        "mean_conv_value": sum(values) / len(values) if values else 0,
        "mean_conv_selection": sum(selections) / len(selections) if selections else 0,
        "mean_peak_complexity": sum(peaks) / len(peaks) if peaks else 0,
        "mean_turns": sum(turns) / len(turns) if turns else 0,
    }

    for key, rows in (
        ("conv_value", values),
        ("conv_selection", selections),
        ("peak_complexity", peaks),
    ):
        bins = [0] * 10
        for value in rows:
            if isinstance(value, (int, float)) and 1 <= value <= 10:
                bins[min(int(value) - 1, 9)] += 1
        payload[f"{key}_hist"] = bins

    turn_distribution = {}
    for turns_value in turns:
        turn_distribution[turns_value] = turn_distribution.get(turns_value, 0) + 1
    payload["turn_distribution"] = turn_distribution
    return payload


def _load_pass1_viz(run_dir: Path, is_global: bool):
    try:
        if is_global:
            stats_file = PASS1_SUMMARY_STATS_FILE
            labeled_file = None
        else:
            stats_file = PASS1_STATS_FILE
            labeled_file = "labeled.json"
        stats_path = run_dir / stats_file
        if not stats_path.exists():
            return None
        samples, stats = load_run(run_dir, labeled_file=labeled_file, stats_file=stats_file)
        return compute_viz_data(samples, stats)
    except Exception:
        return None


def _scope_summary(scope: dict) -> dict:
    pass1 = scope.get("pass1")
    pass2 = scope.get("pass2")
    return {
        "pass1_total": (pass1 or {}).get("total", 0),
        "scored_total": ((pass2 or {}).get("overview") or {}).get("total_scored", 0),
        "mean_value": ((pass2 or {}).get("overview") or {}).get("mean_value"),
        "file_count": len(scope.get("descendant_files", [])) if scope.get("kind") != "file" else 1,
    }


def _single_scope_payload(run_dir: Path, pass2_viz: dict, stats: dict) -> dict:
    pass1_viz = _load_pass1_viz(run_dir, is_global=False)
    conversation = _compute_conv_viz_data(_load_conversation_scores(run_dir))
    label = Path(stats.get("input_file") or run_dir.name or "current").name
    scopes = {
        "global": {
            "id": "global",
            "label": label,
            "kind": "file",
            "path": stats.get("input_file") or label,
            "parent_id": None,
            "children": [],
            "descendant_files": [stats.get("input_file") or label],
            "pass1": pass1_viz,
            "pass2": pass2_viz,
            "conversation": conversation,
        }
    }
    scopes["global"]["summary"] = _scope_summary(scopes["global"])
    return {
        "title": "SFT Labeling & Scoring Dashboard",
        "subtitle": _dashboard_subtitle(run_dir, label),
        "root_id": "global",
        "default_scope_id": "global",
        "initially_expanded": ["global"],
        "scopes": scopes,
    }


def _tree_payload(run_dir: Path) -> dict:
    tree = build_scope_tree(
        run_dir,
        pass1_stats_file=PASS1_STATS_FILE,
        pass1_summary_file=PASS1_SUMMARY_STATS_FILE,
        pass2_stats_file=PASS2_STATS_FILE,
        pass2_summary_file=PASS2_SUMMARY_STATS_FILE,
    )
    scopes = {}
    for scope_id, raw_scope in tree["scopes"].items():
        pass1 = compute_viz_data([], raw_scope["raw_pass1"]) if raw_scope.get("raw_pass1") else None
        pass2 = compute_value_viz_data([], raw_scope["raw_pass2"]) if raw_scope.get("raw_pass2") else None
        conversation = _compute_conv_viz_data(raw_scope.get("raw_conversations", []))
        scopes[scope_id] = {
            "id": raw_scope["id"],
            "label": raw_scope["label"],
            "kind": raw_scope["kind"],
            "path": raw_scope["path"],
            "parent_id": raw_scope["parent_id"],
            "children": raw_scope["children"],
            "descendant_files": raw_scope["descendant_files"],
            "pass1": pass1,
            "pass2": pass2,
            "conversation": conversation,
        }
        scopes[scope_id]["summary"] = _scope_summary(scopes[scope_id])

    root_scope = tree["scopes"][tree["root_id"]]
    scopes[tree["root_id"]]["label"] = root_scope.get("label") or scopes[tree["root_id"]]["label"]

    initially_expanded = ["global"]
    initially_expanded.extend(
        scope_id for scope_id, scope in scopes.items()
        if scope["kind"] == "dir" and scope.get("parent_id") == "global"
    )

    return {
        "title": "SFT Labeling & Scoring Dashboard",
        "subtitle": _dashboard_subtitle(run_dir, scopes[tree["root_id"]]["label"]),
        "root_id": tree["root_id"],
        "default_scope_id": tree["root_id"],
        "initially_expanded": initially_expanded,
        "scopes": scopes,
    }


def generate_value_dashboard(run_dir, scored_file="scored.json",
                             stats_file=PASS2_STATS_FILE,
                             output_file="dashboard_scoring.html",
                             quiet=False):
    """Generate an interactive Pass 2 dashboard."""
    run_dir = Path(run_dir)
    output_path = resolve_dashboard_output(run_dir, output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if scored_file is None or stats_file == PASS2_SUMMARY_STATS_FILE:
        payload = _tree_payload(run_dir)
    else:
        samples, stats = load_value_run(run_dir, scored_file=scored_file, stats_file=stats_file)
        payload = _single_scope_payload(run_dir, compute_value_viz_data(samples, stats), stats)

    output_path.write_text(render_dashboard_html(payload), encoding="utf-8")
    if not quiet:
        print(f"  Dashboard: {output_path}")
    return output_path
