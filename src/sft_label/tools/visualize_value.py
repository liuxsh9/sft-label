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
from sft_label.config import PASS2_HEAVY_POSTPROCESS_SAMPLE_THRESHOLD
from sft_label.tools.dashboard_scopes import build_scope_tree
from sft_label.tools.dashboard_aggregation import (
    DIMENSIONS,
    _build_unmapped_details,
    _coverage_from_distributions,
    _cross_matrix_from_labels,
    build_pass2_viz,
    build_scope_summary,
    infer_scope_turn_kind,
    infer_scope_turn_kind_from_path,
    load_data_file,
    merge_turn_kinds,
)
from sft_label.tools.visualize_labels import (
    _inject_conversation_mode,
    _dashboard_subtitle,
    _write_dashboard_bundle,
    compute_viz_data,
    load_run,
)


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
    payload = build_pass2_viz(samples, stats, conv_records)
    overview_fields = {
        "prompt_mode": stats.get("prompt_mode", "full"),
        "compact_prompt": bool(stats.get("compact_prompt")),
        "value_truncation_budget": stats.get("value_truncation_budget"),
    }
    targets = [payload]
    targets.extend((payload.get("modes") or {}).values())
    for target in targets:
        if not isinstance(target, dict):
            continue
        overview = target.setdefault("overview", {})
        for key, value in overview_fields.items():
            overview[key] = value
    return payload


def _load_conversation_scores(run_dir):
    path = Path(run_dir) / "conversation_scores.json"
    if not path.exists():
        return []
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError):
        return []


def _first_source_file(samples: list[dict] | None, conv_records: list[dict] | None) -> str | None:
    for sample in samples or []:
        source_file = ((sample or {}).get("metadata") or {}).get("source_file")
        if isinstance(source_file, str) and source_file:
            return source_file
    for record in conv_records or []:
        source_file = (record or {}).get("source_file")
        if isinstance(source_file, str) and source_file:
            return source_file
    return None


def _normalized_dashboard_stats(
    stats: dict | None,
    *,
    scope_kind: str,
    scope_path: str,
    samples: list[dict] | None = None,
    conv_records: list[dict] | None = None,
    run_dir: Path | None = None,
) -> dict | None:
    if not isinstance(stats, dict):
        return stats
    payload = dict(stats)

    if scope_kind == "file":
        source_path = scope_path or _first_source_file(samples, conv_records)
        if source_path:
            payload["input_file"] = source_path
        return payload

    if scope_kind == "dir":
        if scope_path:
            payload["input_path"] = scope_path
        return payload

    if scope_kind == "global" and run_dir is not None:
        payload["input_path"] = payload.get("input_path") or str(run_dir)
    return payload


def _postprocess_dashboard_entry(stats: dict | None) -> dict:
    if not isinstance(stats, dict):
        return {}
    postprocess = stats.get("postprocess")
    if not isinstance(postprocess, dict):
        return {}
    dashboard = postprocess.get("dashboard")
    return dashboard if isinstance(dashboard, dict) else {}


def _resolve_explorer_policy(stats: dict | None) -> dict:
    dashboard = _postprocess_dashboard_entry(stats)
    status = str(dashboard.get("status") or "").strip().lower()
    reason = dashboard.get("reason")
    if status in {"deferred", "pending", "failed"}:
        policy = {"enabled": False, "mode": status}
        if reason:
            policy["reason"] = str(reason)
        return policy
    if status == "completed":
        return {"enabled": True, "mode": "completed"}
    if status:
        policy = {"enabled": False, "mode": "missing"}
        policy["reason"] = f"dashboard_status={status}"
        return policy

    total_scored = int((stats or {}).get("total_scored") or 0)
    threshold = max(int(PASS2_HEAVY_POSTPROCESS_SAMPLE_THRESHOLD), 1)
    reason = "dashboard postprocess metadata missing"
    if total_scored >= threshold:
        reason = f"{reason}; samples={total_scored}>={threshold}"
    return {"enabled": False, "mode": "missing", "reason": reason}


def _pass1_conversation_mode_from_conv_records(conv_records, fallback: dict | None = None) -> dict | None:
    if not conv_records:
        return fallback

    total = len(conv_records)
    distributions = {dim: {} for dim in DIMENSIONS}
    label_rows = []
    unmapped_tags = {}
    llm_labeled_units = 0
    inherited_units = 0

    for rec in conv_records:
        labels = rec.get("merged_labels")
        if not isinstance(labels, dict):
            labels = {}
        if labels:
            label_rows.append(labels)

        for dim in DIMENSIONS:
            value = labels.get(dim)
            if isinstance(value, list):
                for item in value:
                    if item:
                        distributions[dim][item] = distributions[dim].get(item, 0) + 1
            elif value:
                distributions[dim][value] = distributions[dim].get(value, 0) + 1

        for item in labels.get("unmapped") or []:
            if isinstance(item, dict):
                key = f"{item.get('dimension', '?')}:{item.get('value', '?')}"
            else:
                key = str(item)
            unmapped_tags[key] = unmapped_tags.get(key, 0) + 1

        detail = rec.get("detail") or {}
        observed_turns = detail.get("observed_turns")
        inherited_turns = detail.get("inherited_turns")
        if isinstance(observed_turns, int):
            if observed_turns > 0:
                llm_labeled_units += 1
        elif labels:
            llm_labeled_units += 1
        if isinstance(inherited_turns, int) and inherited_turns > 0:
            inherited_units += 1

    distributions = {
        dim: dict(sorted(rows.items(), key=lambda item: (-item[1], item[0])))
        for dim, rows in distributions.items()
    }
    distribution_total = len(label_rows)
    overview = dict((fallback or {}).get("overview") or {})
    overview["success_rate"] = round(distribution_total / max(total, 1), 4)
    overview["unmapped_unique"] = len(unmapped_tags)
    overview["llm_labeled_units"] = llm_labeled_units
    overview["inherited_units"] = inherited_units

    mode = dict(fallback or {})
    mode.update(
        {
            "total": total,
            "distribution_total": distribution_total,
            "distributions": distributions,
            "unmapped_details": _build_unmapped_details([], {"unmapped_tags": unmapped_tags}),
            "confidence_stats": {},
            "cross_matrix": _cross_matrix_from_labels(label_rows),
            "coverage": _coverage_from_distributions(distributions),
            "overview": overview,
            "unit_label": "units",
            "mode_id": "conversation",
        }
    )
    return mode


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
    compression_gaps = [row["compression_gap"] for row in conv_records if row.get("compression_gap") is not None]
    late_turn_gains = [row["late_turn_gain"] for row in conv_records if row.get("late_turn_gain") is not None]
    tool_turn_ratios = [row["tool_turn_ratio"] for row in conv_records if row.get("tool_turn_ratio") is not None]
    unique_tool_counts = [row["unique_tool_count"] for row in conv_records if row.get("unique_tool_count") is not None]
    unique_file_counts = [row["unique_file_count"] for row in conv_records if row.get("unique_file_count") is not None]
    turn_value_stds = [
        (row.get("detail") or {}).get("turn_value_std")
        for row in conv_records
        if (row.get("detail") or {}).get("turn_value_std") is not None
    ]
    test_related_turns = [
        (row.get("detail") or {}).get("test_related_turn_count")
        for row in conv_records
        if (row.get("detail") or {}).get("test_related_turn_count") is not None
    ]
    edit_related_turns = [
        (row.get("detail") or {}).get("edit_related_turn_count")
        for row in conv_records
        if (row.get("detail") or {}).get("edit_related_turn_count") is not None
    ]
    clarification_turns = [
        (row.get("detail") or {}).get("clarification_turn_count")
        for row in conv_records
        if (row.get("detail") or {}).get("clarification_turn_count") is not None
    ]
    recovery_success_ratios = [
        (row.get("detail") or {}).get("recovery_success_ratio")
        for row in conv_records
        if (row.get("detail") or {}).get("recovery_success_ratio") is not None
    ]
    tool_result_success_ratios = [
        (row.get("detail") or {}).get("tool_result_success_ratio")
        for row in conv_records
        if (row.get("detail") or {}).get("tool_result_success_ratio") is not None
    ]
    tool_repeat_ratios = [
        (row.get("detail") or {}).get("tool_repeat_ratio")
        for row in conv_records
        if (row.get("detail") or {}).get("tool_repeat_ratio") is not None
    ]
    verification_turns = [
        (row.get("detail") or {}).get("verification_turn_count")
        for row in conv_records
        if (row.get("detail") or {}).get("verification_turn_count") is not None
    ]
    verification_after_edit_turns = [
        (row.get("detail") or {}).get("verification_after_edit_count")
        for row in conv_records
        if (row.get("detail") or {}).get("verification_after_edit_count") is not None
    ]

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
        "mean_turn_value_std": sum(turn_value_stds) / len(turn_value_stds) if turn_value_stds else 0,
        "mean_peak_minus_mean": sum(compression_gaps) / len(compression_gaps) if compression_gaps else 0,
        "mean_late_turn_gain": sum(late_turn_gains) / len(late_turn_gains) if late_turn_gains else 0,
        "mean_tool_turn_ratio": sum(tool_turn_ratios) / len(tool_turn_ratios) if tool_turn_ratios else 0,
        "mean_unique_tool_count": sum(unique_tool_counts) / len(unique_tool_counts) if unique_tool_counts else 0,
        "mean_unique_file_count": sum(unique_file_counts) / len(unique_file_counts) if unique_file_counts else 0,
        "mean_test_related_turns": sum(test_related_turns) / len(test_related_turns) if test_related_turns else 0,
        "mean_edit_related_turns": sum(edit_related_turns) / len(edit_related_turns) if edit_related_turns else 0,
        "mean_clarification_turns": sum(clarification_turns) / len(clarification_turns) if clarification_turns else 0,
        "mean_recovery_success_ratio": sum(recovery_success_ratios) / len(recovery_success_ratios) if recovery_success_ratios else 0,
        "mean_tool_result_success_ratio": sum(tool_result_success_ratios) / len(tool_result_success_ratios) if tool_result_success_ratios else 0,
        "mean_tool_repeat_ratio": sum(tool_repeat_ratios) / len(tool_repeat_ratios) if tool_repeat_ratios else 0,
        "mean_verification_turns": sum(verification_turns) / len(verification_turns) if verification_turns else 0,
        "mean_verification_after_edit_turns": sum(verification_after_edit_turns) / len(verification_after_edit_turns) if verification_after_edit_turns else 0,
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


def _single_scope_payload(run_dir: Path, samples: list[dict], pass2_viz: dict, stats: dict) -> dict:
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
            "turn_kind": infer_scope_turn_kind(samples),
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


def _tree_payload(
    run_dir: Path,
    *,
    include_conversations: bool = True,
) -> tuple[dict, list[dict], dict | None]:
    tree = build_scope_tree(
        run_dir,
        pass1_stats_file=PASS1_STATS_FILE,
        pass1_summary_file=PASS1_SUMMARY_STATS_FILE,
        pass2_stats_file=PASS2_STATS_FILE,
        pass2_summary_file=PASS2_SUMMARY_STATS_FILE,
        include_pass2_conversations=include_conversations,
    )
    scopes = {}
    explorer_sources = []
    file_conversations = {
        scope_id: list(raw_scope.get("raw_conversations") or [])
        for scope_id, raw_scope in tree["scopes"].items()
        if raw_scope.get("kind") == "file"
    }
    file_turn_kind = {
        scope_id: infer_scope_turn_kind_from_path(raw_scope.get("pass2_data_path"))
        for scope_id, raw_scope in tree["scopes"].items()
        if raw_scope.get("kind") == "file"
    }
    for scope_id, raw_scope in tree["scopes"].items():
        is_file = raw_scope.get("kind") == "file"
        if is_file:
            conv_records = list(file_conversations.get(scope_id, []))
        else:
            conv_records = []
            for leaf_path in raw_scope.get("descendant_files", []):
                conv_records.extend(file_conversations.get(f"file:{leaf_path}", []))
        has_conv_records = bool(conv_records)
        normalized_pass1_stats = _normalized_dashboard_stats(
            raw_scope.get("raw_pass1"),
            scope_kind=raw_scope["kind"],
            scope_path=raw_scope["path"],
            conv_records=conv_records,
            run_dir=run_dir,
        )
        normalized_pass2_stats = _normalized_dashboard_stats(
            raw_scope.get("raw_pass2"),
            scope_kind=raw_scope["kind"],
            scope_path=raw_scope["path"],
            conv_records=conv_records,
            run_dir=run_dir,
        )
        pass1 = compute_viz_data([], normalized_pass1_stats) if normalized_pass1_stats else None
        if include_conversations and has_conv_records:
            pass1_conversation_mode = _pass1_conversation_mode_from_conv_records(
                conv_records,
                fallback=((pass1 or {}).get("modes") or {}).get("conversation"),
            )
            if pass1_conversation_mode:
                pass1 = _inject_conversation_mode(pass1, [pass1_conversation_mode]) if pass1 else {
                    "modes": {"conversation": pass1_conversation_mode}
                }
        elif isinstance((pass1 or {}).get("modes"), dict):
            pass1["modes"].pop("conversation", None)
        pass2 = compute_value_viz_data([], normalized_pass2_stats, conv_records) if normalized_pass2_stats else None
        if not include_conversations or not has_conv_records:
            if isinstance((pass2 or {}).get("modes"), dict):
                pass2["modes"].pop("conversation", None)
            conversation = None
        else:
            conversation = _compute_conv_viz_data(conv_records)
        if is_file:
            turn_kind = file_turn_kind.get(scope_id)
        else:
            turn_kind = merge_turn_kinds(
                [file_turn_kind.get(f"file:{leaf_path}") for leaf_path in raw_scope.get("descendant_files", [])]
            )
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
            "turn_kind": turn_kind,
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

    payload = {
        "title": "SFT Labeling & Scoring Dashboard",
        "title_key": "dashboard_title_scoring",
        "subtitle": _dashboard_subtitle(run_dir, scopes[tree["root_id"]]["label"]),
        "root_id": tree["root_id"],
        "default_scope_id": tree["root_id"],
        "initially_expanded": initially_expanded,
        "scopes": scopes,
    }
    root_pass2_stats = tree["scopes"][tree["root_id"]].get("raw_pass2")
    return payload, explorer_sources, root_pass2_stats


def generate_value_dashboard(run_dir, scored_file="scored.json",
                             stats_file=PASS2_STATS_FILE,
                             output_file="dashboard_scoring.html",
                             quiet=False,
                             static_base_url: str | None = None,
                             stats_hint_override: dict | None = None):
    """Generate an interactive Pass 2 dashboard."""
    run_dir = Path(run_dir)
    output_path = resolve_dashboard_output(run_dir, output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if scored_file is None or stats_file == PASS2_SUMMARY_STATS_FILE:
        if not quiet:
            print("  Scoring dashboard: building scope tree")
        stats_hint = dict(stats_hint_override) if isinstance(stats_hint_override, dict) else {}
        if not stats_hint:
            stats_path = run_dir / stats_file
            if stats_path.exists():
                try:
                    with open(stats_path, encoding="utf-8") as f:
                        loaded = json.load(f)
                    if isinstance(loaded, dict):
                        stats_hint = loaded
                except (OSError, json.JSONDecodeError):
                    stats_hint = {}
        explorer_policy = _resolve_explorer_policy(stats_hint)
        payload, explorer_sources, root_pass2_stats = _tree_payload(
            run_dir,
            include_conversations=bool(explorer_policy.get("enabled", True)),
        )
        if not stats_hint and isinstance(root_pass2_stats, dict):
            explorer_policy = _resolve_explorer_policy(root_pass2_stats)
    else:
        samples, stats = load_value_run(run_dir, scored_file=scored_file, stats_file=stats_file)
        explorer_policy = _resolve_explorer_policy(stats)
        conv_records = _load_conversation_scores(run_dir) if explorer_policy.get("enabled", True) else []
        normalized_stats = _normalized_dashboard_stats(
            stats,
            scope_kind="file",
            scope_path=_first_source_file(samples, conv_records) or "",
            samples=samples,
            conv_records=conv_records,
            run_dir=run_dir,
        ) or {}
        payload = _single_scope_payload(
            run_dir,
            samples,
            compute_value_viz_data(samples, normalized_stats, conv_records),
            normalized_stats,
        )
        if not explorer_policy.get("enabled", True):
            global_scope = payload.get("scopes", {}).get("global") or {}
            if isinstance((global_scope.get("pass1") or {}).get("modes"), dict):
                global_scope["pass1"]["modes"].pop("conversation", None)
            if isinstance((global_scope.get("pass2") or {}).get("modes"), dict):
                global_scope["pass2"]["modes"].pop("conversation", None)
            global_scope["conversation"] = None
            global_scope["summary"] = _scope_summary(global_scope)
        explorer_sources = []
        data_path = run_dir / scored_file if scored_file else None
        conv_path = run_dir / "conversation_scores.json"
        if explorer_policy.get("enabled", True) and data_path and data_path.exists():
            explorer_sources.append(
                {
                    "scope_id": "global",
                    "scope_path": payload["scopes"]["global"]["path"],
                    "data_path": str(data_path),
                    "conversation_path": str(conv_path) if conv_path.exists() else None,
                    "has_scores": True,
                }
            )

    if not quiet:
        print("  Scoring dashboard: writing dashboard bundle")
    _write_dashboard_bundle(
        output_path,
        payload,
        explorer_sources,
        dashboard_type="scoring",
        static_base_url=static_base_url,
        explorer_policy=explorer_policy,
    )
    if not quiet:
        print(f"  Dashboard: {output_path}")
    return output_path
