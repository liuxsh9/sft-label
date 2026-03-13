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
from sft_label.tools.dashboard_explorer import build_explorer_assets
from sft_label.tools.dashboard_scopes import build_scope_tree
from sft_label.tools.dashboard_template import render_dashboard_html
from sft_label.tools.dashboard_aggregation import build_pass2_viz, build_scope_summary, load_data_file
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


def compute_value_viz_data(samples, stats, conv_records=None):
    """Transform Pass 2 data into dashboard JSON."""
    return build_pass2_viz(samples, stats, conv_records)


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
    observed_ratios = [row["observed_turn_ratio"] for row in conv_records if row.get("observed_turn_ratio") is not None]
    rarity_confidences = [row["rarity_confidence"] for row in conv_records if row.get("rarity_confidence") is not None]
    inherited_ratios = [row["inherited_turn_ratio"] for row in conv_records if row.get("inherited_turn_ratio") is not None]

    def _ratio_bands(rows):
        bands = {
            "0-25%": 0,
            "25-50%": 0,
            "50-75%": 0,
            "75-100%": 0,
        }
        for value in rows:
            if not isinstance(value, (int, float)):
                continue
            if value < 0.25:
                bands["0-25%"] += 1
            elif value < 0.5:
                bands["25-50%"] += 1
            elif value < 0.75:
                bands["50-75%"] += 1
            else:
                bands["75-100%"] += 1
        return bands

    payload = {
        "total": total,
        "mean_conv_value": sum(values) / len(values) if values else 0,
        "mean_conv_selection": sum(selections) / len(selections) if selections else 0,
        "mean_peak_complexity": sum(peaks) / len(peaks) if peaks else 0,
        "mean_turns": sum(turns) / len(turns) if turns else 0,
        "mean_observed_turn_ratio": sum(observed_ratios) / len(observed_ratios) if observed_ratios else 0,
        "mean_inherited_turn_ratio": sum(inherited_ratios) / len(inherited_ratios) if inherited_ratios else 0,
        "mean_rarity_confidence": sum(rarity_confidences) / len(rarity_confidences) if rarity_confidences else 0,
        "low_observed_coverage_count": sum(1 for value in observed_ratios if value < 0.5),
        "low_rarity_confidence_count": sum(1 for value in rarity_confidences if value < 0.6),
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
    payload["observed_turn_ratio_bands"] = _ratio_bands(observed_ratios)
    payload["rarity_confidence_bands"] = _ratio_bands(rarity_confidences)
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
        if not samples:
            data_path = next(
                (candidate for candidate in (run_dir / "labeled.jsonl", run_dir / "labeled.json") if candidate.exists()),
                None,
            )
            if data_path is not None:
                samples = load_data_file(data_path)
        return compute_viz_data(samples, stats)
    except Exception:
        return None


def _scope_summary(scope: dict) -> dict:
    summary_modes = build_scope_summary(scope)
    scope["summary_modes"] = summary_modes
    return summary_modes["sample"]


def _single_scope_payload(run_dir: Path, pass2_viz: dict, stats: dict) -> dict:
    pass1_viz = _load_pass1_viz(run_dir, is_global=False)
    conv_records = _load_conversation_scores(run_dir)
    conversation = _compute_conv_viz_data(conv_records)
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
        "title_key": "dashboard_title_scoring",
        "subtitle": _dashboard_subtitle(run_dir, label),
        "root_id": "global",
        "default_scope_id": "global",
        "initially_expanded": ["global"],
        "scopes": scopes,
    }


def _tree_payload(run_dir: Path) -> tuple[dict, list[dict]]:
    tree = build_scope_tree(
        run_dir,
        pass1_stats_file=PASS1_STATS_FILE,
        pass1_summary_file=PASS1_SUMMARY_STATS_FILE,
        pass2_stats_file=PASS2_STATS_FILE,
        pass2_summary_file=PASS2_SUMMARY_STATS_FILE,
    )
    scopes = {}
    explorer_sources = []
    file_sample_cache = {
        scope_id: load_data_file(raw_scope.get("pass2_data_path"))
        for scope_id, raw_scope in tree["scopes"].items()
        if raw_scope.get("kind") == "file"
    }
    for scope_id, raw_scope in tree["scopes"].items():
        samples = []
        for leaf_path in raw_scope.get("descendant_files", []):
            samples.extend(file_sample_cache.get(f"file:{leaf_path}", []))
        pass1 = compute_viz_data(samples, raw_scope["raw_pass1"]) if raw_scope.get("raw_pass1") else None
        pass2 = compute_value_viz_data(samples, raw_scope["raw_pass2"], raw_scope.get("raw_conversations", [])) if raw_scope.get("raw_pass2") else None
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
        if raw_scope.get("kind") == "file" and raw_scope.get("pass2_data_path"):
            explorer_sources.append(
                {
                    "scope_id": scope_id,
                    "scope_path": raw_scope["path"],
                    "data_path": raw_scope["pass2_data_path"],
                    "conversation_path": raw_scope.get("conversation_data_path"),
                    "has_scores": True,
                }
            )

    root_scope = tree["scopes"][tree["root_id"]]
    scopes[tree["root_id"]]["label"] = root_scope.get("label") or scopes[tree["root_id"]]["label"]

    initially_expanded = ["global"]
    initially_expanded.extend(
        scope_id for scope_id, scope in scopes.items()
        if scope["kind"] == "dir" and scope.get("parent_id") == "global"
    )

    return {
        "title": "SFT Labeling & Scoring Dashboard",
        "title_key": "dashboard_title_scoring",
        "subtitle": _dashboard_subtitle(run_dir, scopes[tree["root_id"]]["label"]),
        "root_id": tree["root_id"],
        "default_scope_id": tree["root_id"],
        "initially_expanded": initially_expanded,
        "scopes": scopes,
    }, explorer_sources


def _attach_explorer_payload(payload: dict, explorer_meta: dict[str, dict]) -> None:
    if not explorer_meta:
        return
    for scope_id, meta in explorer_meta.items():
        if scope_id in payload.get("scopes", {}):
            payload["scopes"][scope_id]["explorer"] = meta
    payload["explorer"] = {
        "enabled": True,
        "result_limit": 200,
        "detail_limit_notice": "Explorer scans preview shards progressively and loads full details on demand.",
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
        payload, explorer_sources = _tree_payload(run_dir)
    else:
        samples, stats = load_value_run(run_dir, scored_file=scored_file, stats_file=stats_file)
        payload = _single_scope_payload(run_dir, compute_value_viz_data(samples, stats, _load_conversation_scores(run_dir)), stats)
        explorer_sources = []
        data_path = run_dir / scored_file if scored_file else None
        conv_path = run_dir / "conversation_scores.json"
        if data_path and data_path.exists():
            explorer_sources.append(
                {
                    "scope_id": "global",
                    "scope_path": payload["scopes"]["global"]["path"],
                    "data_path": str(data_path),
                    "conversation_path": str(conv_path) if conv_path.exists() else None,
                    "has_scores": True,
                }
            )

    _attach_explorer_payload(payload, build_explorer_assets(output_path, explorer_sources))

    output_path.write_text(render_dashboard_html(payload), encoding="utf-8")
    if not quiet:
        print(f"  Dashboard: {output_path}")
    return output_path
