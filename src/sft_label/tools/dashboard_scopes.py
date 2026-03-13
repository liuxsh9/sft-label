"""Scope discovery and aggregation helpers for interactive dashboards."""

from __future__ import annotations

import json
from pathlib import Path

from sft_label.artifacts import (
    DASHBOARDS_DIRNAME,
    PASS1_STATS_FILE,
    PASS1_SUMMARY_STATS_FILE,
    PASS2_STATS_FILE,
    PASS2_SUMMARY_STATS_FILE,
)
from sft_label.run_layout import FILE_ARTIFACTS_DIRNAME


def _load_json(path: Path):
    if not path.exists():
        return None
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError):
        return None


def merge_label_stats(all_file_stats: list[dict]) -> dict | None:
    """Merge Pass 1 stats without importing the runtime pipeline module."""
    if not all_file_stats:
        return None

    merged = {
        "total_samples": 0,
        "success": 0,
        "failed": 0,
        "distribution_total_samples": 0,
        "total_llm_calls": 0,
        "total_prompt_tokens": 0,
        "total_completion_tokens": 0,
        "total_tokens": 0,
        "arbitrated_count": 0,
        "validation_issue_count": 0,
        "consistency_warning_count": 0,
        "unmapped_unique_count": 0,
        "canonicalization_total_count": 0,
        "canonicalization_unique_count": 0,
        "total_elapsed_seconds": 0,
        "sparse_labeled": 0,
        "sparse_inherited": 0,
        "rows_total": 0,
        "rows_skipped": 0,
        "rows_changed": 0,
        "rows_migrated": 0,
        "rows_pass2_invalidated": 0,
        "preserved_samples": 0,
        "tag_distributions": {},
        "combo_distributions": {},
        "confidence_stats": {},
        "cross_matrix": {},
        "unmapped_tags": {},
        "canonicalization_counts": {},
        "files_processed": len(all_file_stats),
    }

    for st in all_file_stats:
        for key in (
            "total_samples",
            "success",
            "failed",
            "total_llm_calls",
            "total_prompt_tokens",
            "total_completion_tokens",
            "total_tokens",
            "arbitrated_count",
            "validation_issue_count",
            "consistency_warning_count",
            "canonicalization_total_count",
            "sparse_labeled",
            "sparse_inherited",
            "distribution_total_samples",
            "rows_total",
            "rows_skipped",
            "rows_changed",
            "rows_migrated",
            "rows_pass2_invalidated",
            "preserved_samples",
        ):
            merged[key] += st.get(key, 0)

        merged["total_elapsed_seconds"] += st.get("total_elapsed_seconds", 0)

        for dim, dist in st.get("tag_distributions", {}).items():
            bucket = merged["tag_distributions"].setdefault(dim, {})
            for tag, count in dist.items():
                bucket[tag] = bucket.get(tag, 0) + count
        for combo_key, count in st.get("combo_distributions", {}).items():
            merged["combo_distributions"][combo_key] = merged["combo_distributions"].get(combo_key, 0) + count

        for tag, count in st.get("unmapped_tags", {}).items():
            merged["unmapped_tags"][tag] = merged["unmapped_tags"].get(tag, 0) + count
        for key, count in st.get("canonicalization_counts", {}).items():
            merged["canonicalization_counts"][key] = merged["canonicalization_counts"].get(key, 0) + count

        for dim, stats in st.get("confidence_stats", {}).items():
            entry = merged["confidence_stats"].setdefault(
                dim,
                {"sum": 0.0, "count": 0, "min": stats.get("min", 1), "max": stats.get("max", 0), "below_threshold": 0},
            )
            count = stats.get("count", 0)
            if count <= 0:
                continue
            entry["sum"] += stats.get("mean", 0) * count
            entry["count"] += count
            entry["min"] = min(entry["min"], stats.get("min", 1))
            entry["max"] = max(entry["max"], stats.get("max", 0))
            entry["below_threshold"] += stats.get("below_threshold", 0)

        for key, count in st.get("cross_matrix", {}).items():
            merged["cross_matrix"][key] = merged["cross_matrix"].get(key, 0) + count

    for dim, dist in merged["tag_distributions"].items():
        merged["tag_distributions"][dim] = dict(sorted(dist.items(), key=lambda item: -item[1]))
    merged["combo_distributions"] = dict(sorted(merged["combo_distributions"].items(), key=lambda item: -item[1]))

    merged["unmapped_tags"] = dict(sorted(merged["unmapped_tags"].items(), key=lambda item: -item[1]))
    merged["unmapped_unique_count"] = len(merged["unmapped_tags"])
    merged["canonicalization_counts"] = dict(sorted(merged["canonicalization_counts"].items(), key=lambda item: (-item[1], item[0])))
    merged["canonicalization_unique_count"] = len(merged["canonicalization_counts"])

    finalized_conf = {}
    for dim, entry in merged["confidence_stats"].items():
        count = entry.get("count", 0)
        if count <= 0:
            continue
        finalized_conf[dim] = {
            "mean": round(entry["sum"] / count, 3),
            "min": round(entry["min"], 3),
            "max": round(entry["max"], 3),
            "below_threshold": entry["below_threshold"],
            "count": count,
        }
    merged["confidence_stats"] = finalized_conf

    total = merged["total_samples"]
    merged["success_rate"] = round(merged["success"] / max(total, 1), 4)
    merged["avg_calls_per_sample"] = round(merged["total_llm_calls"] / max(total, 1), 2)
    merged["arbitrated_rate"] = round(merged["arbitrated_count"] / max(total, 1), 4)
    return merged


def merge_scoring_stats(file_stats_list: list[dict]) -> dict | None:
    """Merge Pass 2 stats without importing the runtime scoring module."""
    if not file_stats_list:
        return None

    merged = {
        "total_scored": sum(stats.get("total_scored", 0) for stats in file_stats_list),
        "total_failed": sum(stats.get("total_failed", 0) for stats in file_stats_list),
        "total_llm_calls": sum(stats.get("total_llm_calls", 0) for stats in file_stats_list),
        "total_prompt_tokens": sum(stats.get("total_prompt_tokens", 0) for stats in file_stats_list),
        "total_completion_tokens": sum(stats.get("total_completion_tokens", 0) for stats in file_stats_list),
        "total_tokens": sum(stats.get("total_tokens", 0) for stats in file_stats_list),
    }

    merged["per_file_summary"] = [
        {
            "file": stats.get("file") or Path(stats.get("input_file", "unknown")).name,
            "count": stats.get("total_scored", 0),
            "mean_value": stats.get("score_distributions", {}).get("value_score", {}).get("mean", 0),
            "mean_complexity": stats.get("score_distributions", {}).get("complexity_overall", {}).get("mean", 0),
            "mean_quality": stats.get("score_distributions", {}).get("quality_overall", {}).get("mean", 0),
            "mean_rarity": stats.get("score_distributions", {}).get("rarity_score", {}).get("mean", 0),
            "mean_selection": stats.get("score_distributions", {}).get("selection_score", {}).get("mean", 0),
        }
        for stats in file_stats_list
    ]

    all_dist_keys = set()
    for stats in file_stats_list:
        for key, value in stats.get("score_distributions", {}).items():
            if isinstance(value, dict):
                all_dist_keys.add(key)

    merged_distributions = {}
    for key in all_dist_keys:
        total_n = 0
        total_sum = 0.0
        global_min = float("inf")
        global_max = float("-inf")
        for stats in file_stats_list:
            dist = stats.get("score_distributions", {}).get(key, {})
            count = stats.get("total_scored", 0)
            if not isinstance(dist, dict) or not dist or count <= 0:
                continue
            total_sum += dist.get("mean", 0) * count
            total_n += count
            if "min" in dist:
                global_min = min(global_min, dist["min"])
            if "max" in dist:
                global_max = max(global_max, dist["max"])
        if total_n > 0:
            merged_distributions[key] = {
                "mean": round(total_sum / total_n, 2),
                "min": global_min if global_min != float("inf") else 0,
                "max": global_max if global_max != float("-inf") else 0,
                "total_n": total_n,
            }
    merged["score_distributions"] = merged_distributions

    merged_histograms = {}
    for stats in file_stats_list:
        for key, bins in stats.get("histograms", {}).items():
            bucket = merged_histograms.setdefault(key, [0] * 10)
            for idx, value in enumerate(bins):
                if idx < len(bucket):
                    bucket[idx] += value
    merged["histograms"] = merged_histograms

    merged_conf = [0] * 10
    for stats in file_stats_list:
        bins = stats.get("confidence_histogram") or []
        for idx, value in enumerate(bins[:10]):
            merged_conf[idx] += value
    if any(merged_conf):
        merged["confidence_histogram"] = merged_conf

    thinking = {
        "slow": {"count": 0, "sum_value": 0.0, "sum_quality": 0.0, "sum_reasoning": 0.0},
        "fast": {"count": 0, "sum_value": 0.0, "sum_quality": 0.0, "sum_reasoning": 0.0},
    }
    for stats in file_stats_list:
        mode_stats = stats.get("thinking_mode_stats", {})
        for mode in ("slow", "fast"):
            item = mode_stats.get(mode, {})
            count = item.get("count", 0)
            thinking[mode]["count"] += count
            thinking[mode]["sum_value"] += item.get("mean_value", 0) * count
            thinking[mode]["sum_quality"] += item.get("mean_quality", 0) * count
            thinking[mode]["sum_reasoning"] += item.get("mean_reasoning", 0) * count
    merged["thinking_mode_stats"] = {
        mode: {
            "count": payload["count"],
            "mean_value": round(payload["sum_value"] / max(payload["count"], 1), 2),
            "mean_quality": round(payload["sum_quality"] / max(payload["count"], 1), 2),
            "mean_reasoning": round(payload["sum_reasoning"] / max(payload["count"], 1), 2),
        }
        for mode, payload in thinking.items()
        if payload["count"] > 0
    }

    flag_counts = {}
    for stats in file_stats_list:
        for flag, count in stats.get("flag_counts", {}).items():
            flag_counts[flag] = flag_counts.get(flag, 0) + count
    merged["flag_counts"] = dict(sorted(flag_counts.items(), key=lambda item: -item[1]))

    for field in ("value_by_tag", "selection_by_tag"):
        dims = set()
        for stats in file_stats_list:
            dims.update(stats.get(field, {}).keys())
        merged_field = {}
        for dim in dims:
            accum = {}
            for stats in file_stats_list:
                dim_data = stats.get(field, {}).get(dim, {})
                for tag, info in dim_data.items():
                    entry = accum.setdefault(tag, {"sum": 0.0, "count": 0})
                    count = info.get("n", 0)
                    entry["sum"] += info.get("mean", 0) * count
                    entry["count"] += count
            merged_field[dim] = {
                tag: {
                    "mean": round(entry["sum"] / max(entry["count"], 1), 2),
                    "n": entry["count"],
                }
                for tag, entry in sorted(
                    accum.items(),
                    key=lambda item: -(item[1]["sum"] / max(item[1]["count"], 1)),
                )
            }
        merged[field] = merged_field

    merged["weights_used"] = next(
        (stats.get("weights_used") for stats in file_stats_list if stats.get("weights_used")),
        {},
    )
    merged["selection_thresholds"] = next(
        (stats.get("selection_thresholds") for stats in file_stats_list if stats.get("selection_thresholds")),
        {},
    )
    return merged


def _leaf_base(root_dir: Path) -> Path:
    inline_base = root_dir / FILE_ARTIFACTS_DIRNAME
    return inline_base if inline_base.is_dir() else root_dir


_ARTIFACT_LEAF_NAMES = {
    "labeled.json",
    "labeled.jsonl",
    "scored.json",
    "scored.jsonl",
    "monitor.json",
    "monitor.jsonl",
    "monitor_labeling.jsonl",
    "monitor_value.jsonl",
}


def _canonical_leaf_key(path: str | Path) -> str:
    candidate = Path(path)
    if candidate.name in _ARTIFACT_LEAF_NAMES:
        candidate = candidate.parent
    return candidate.with_suffix("").as_posix()


def _preferred_leaf_path(path: str | Path) -> str:
    candidate = Path(path)
    if candidate.name in _ARTIFACT_LEAF_NAMES:
        return candidate.parent.as_posix()
    return candidate.as_posix()


def _guess_leaf_path(rel_dir: Path, stats: dict) -> Path:
    candidate_name = None
    for key in ("input_file", "file", "mirrored_file"):
        value = stats.get(key)
        if isinstance(value, str) and value:
            candidate_name = Path(value).name
            break

    if not candidate_name:
        return rel_dir if rel_dir != Path(".") else Path("current")

    normalized_name = candidate_name
    for prefix in ("labeled_", "scored_", "monitor_"):
        if normalized_name.startswith(prefix):
            normalized_name = normalized_name[len(prefix):]
            break

    candidate_path = Path(normalized_name)
    if rel_dir == Path("."):
        return candidate_path
    if rel_dir.name in {candidate_path.name, candidate_path.stem}:
        return rel_dir.parent / candidate_path.name
    return rel_dir / candidate_path.name


def _with_scoring_file_label(stats: dict | None, file_label: str) -> dict | None:
    """Ensure scoring stats carry a stable source file label for aggregation."""
    if not isinstance(stats, dict):
        return None

    payload = dict(stats)
    label = payload.get("file")
    if isinstance(label, str) and label and Path(label).name not in _ARTIFACT_LEAF_NAMES:
        return payload

    payload["file"] = file_label
    return payload


def _find_sibling_artifact(stats_path: Path, candidates: list[str] | None) -> str | None:
    if not candidates:
        return None
    for name in candidates:
        candidate = stats_path.parent / name
        if candidate.exists():
            return str(candidate)
    return None


def _discover_leaf_records(root_dir: Path, stats_filename: str | None, *,
                           include_conversations: bool = False,
                           data_candidates: list[str] | None = None) -> dict[str, dict]:
    if not stats_filename:
        return {}

    root_dir = Path(root_dir)
    base = _leaf_base(root_dir)
    discovered = {}
    exact_paths = list(base.rglob(stats_filename))
    stem = Path(stats_filename).stem
    suffix = Path(stats_filename).suffix
    suffixed_paths = list(base.rglob(f"{stem}_*{suffix}"))
    seen = set()
    for stats_path in sorted([*exact_paths, *suffixed_paths]):
        if stats_path in seen:
            continue
        seen.add(stats_path)
        rel_dir = stats_path.parent.relative_to(base)
        if base == root_dir and rel_dir == Path("."):
            # Root-level stats are handled as single-scope dashboards, not as a tree leaf.
            continue
        stats = _load_json(stats_path)
        if not isinstance(stats, dict):
            continue

        rel_path = _guess_leaf_path(rel_dir, stats).as_posix()
        conv_records = []
        conv_path_str = None
        if include_conversations:
            conv_path = stats_path.parent / "conversation_scores.json"
            payload = _load_json(conv_path)
            if isinstance(payload, list):
                conv_records = payload
                conv_path_str = str(conv_path)

        discovered[rel_path] = {
            "path": rel_path,
            "stats": stats,
            "conv_records": conv_records,
            "conv_path": conv_path_str,
            "data_path": _find_sibling_artifact(stats_path, data_candidates),
        }
    return discovered


def _summary_stats(root_dir: Path, summary_filename: str | None) -> dict | None:
    if not summary_filename:
        return None
    payload = _load_json(Path(root_dir) / summary_filename)
    return payload if isinstance(payload, dict) else None


def _summary_conversations(root_dir: Path) -> list[dict]:
    payload = _load_json(Path(root_dir) / "conversation_scores.json")
    return payload if isinstance(payload, list) else []


def _normalize_leaf_records(*record_sets: dict[str, dict]) -> tuple[dict[str, dict], ...]:
    canonical_to_path: dict[str, str] = {}
    for records in record_sets:
        for leaf_path in sorted(records):
            key = _canonical_leaf_key(leaf_path)
            canonical_to_path.setdefault(key, _preferred_leaf_path(leaf_path))

    normalized_sets = []
    for records in record_sets:
        normalized = {}
        for leaf_path, payload in records.items():
            key = _canonical_leaf_key(leaf_path)
            normalized[canonical_to_path[key]] = payload
        normalized_sets.append(normalized)
    return tuple(normalized_sets)


def _label_from_stats(stats: dict | None, *, disallowed: set[str]) -> str | None:
    if not isinstance(stats, dict):
        return None
    for key in ("input_path", "input_file", "mirrored_file"):
        value = stats.get(key)
        if not isinstance(value, str) or not value:
            continue
        name = Path(value).name
        if name and name not in disallowed:
            return name
    return None


def _infer_root_label(root_dir: Path, root_pass1: dict | None, root_pass2: dict | None) -> str:
    disallowed = {root_dir.name, FILE_ARTIFACTS_DIRNAME, DASHBOARDS_DIRNAME}
    label = _label_from_stats(root_pass1, disallowed=disallowed)
    if label:
        return label
    label = _label_from_stats(root_pass2, disallowed=disallowed)
    if label:
        return label

    if root_dir.name == "meta_label_data":
        try:
            siblings = [
                child.name
                for child in root_dir.parent.iterdir()
                if child.is_dir() and child.name not in disallowed
            ]
        except OSError:
            siblings = []
        if len(siblings) == 1:
            return siblings[0]

    return root_dir.name or "global"


def build_scope_tree(
    root_dir: Path,
    *,
    pass1_stats_file: str | None = PASS1_STATS_FILE,
    pass1_summary_file: str | None = PASS1_SUMMARY_STATS_FILE,
    pass2_stats_file: str | None = PASS2_STATS_FILE,
    pass2_summary_file: str | None = PASS2_SUMMARY_STATS_FILE,
) -> dict:
    """Build a global/dir/file scope tree for interactive dashboards."""
    root_dir = Path(root_dir)
    pass1_leaves = _discover_leaf_records(
        root_dir,
        pass1_stats_file,
        data_candidates=["labeled.jsonl", "labeled.json"],
    )
    pass2_leaves = _discover_leaf_records(
        root_dir,
        pass2_stats_file,
        include_conversations=True,
        data_candidates=["scored.jsonl", "scored.json"],
    )
    pass1_leaves, pass2_leaves = _normalize_leaf_records(pass1_leaves, pass2_leaves)

    leaf_paths = sorted(set(pass1_leaves) | set(pass2_leaves), key=lambda value: Path(value).as_posix())
    scopes: dict[str, dict] = {
        "global": {
            "id": "global",
            "label": root_dir.name or "global",
            "kind": "global",
            "path": "",
            "parent_id": None,
            "children": [],
            "raw_pass1": None,
            "raw_pass2": None,
            "raw_conversations": [],
            "descendant_files": leaf_paths[:],
        }
    }

    dir_to_files: dict[str, list[str]] = {}
    for leaf_path in leaf_paths:
        path = Path(leaf_path)
        for depth in range(1, len(path.parts)):
            dir_path = Path(*path.parts[:depth]).as_posix()
            dir_to_files.setdefault(dir_path, []).append(leaf_path)

        file_scope_id = f"file:{leaf_path}"
        parent_id = "global" if len(path.parts) == 1 else f"dir:{Path(*path.parts[:-1]).as_posix()}"
        scopes[file_scope_id] = {
            "id": file_scope_id,
            "label": path.name,
            "kind": "file",
            "path": leaf_path,
            "parent_id": parent_id,
            "children": [],
            "raw_pass1": (pass1_leaves.get(leaf_path) or {}).get("stats"),
            "raw_pass2": (pass2_leaves.get(leaf_path) or {}).get("stats"),
            "raw_conversations": (pass2_leaves.get(leaf_path) or {}).get("conv_records", []),
            "pass1_data_path": (pass1_leaves.get(leaf_path) or {}).get("data_path"),
            "pass2_data_path": (pass2_leaves.get(leaf_path) or {}).get("data_path"),
            "conversation_data_path": (pass2_leaves.get(leaf_path) or {}).get("conv_path"),
            "descendant_files": [leaf_path],
        }

    for dir_path in sorted(dir_to_files, key=lambda value: (len(Path(value).parts), value)):
        path = Path(dir_path)
        scope_id = f"dir:{dir_path}"
        parent_id = "global" if len(path.parts) == 1 else f"dir:{Path(*path.parts[:-1]).as_posix()}"

        dir_pass2_inputs = [
            _with_scoring_file_label(scopes[f"file:{leaf_path}"]["raw_pass2"], leaf_path)
            for leaf_path in dir_to_files[dir_path]
        ]
        pass1_stats = merge_label_stats(
            [scopes[f"file:{leaf_path}"]["raw_pass1"] for leaf_path in dir_to_files[dir_path]
             if scopes[f"file:{leaf_path}"]["raw_pass1"]]
        )
        pass2_stats = merge_scoring_stats(
            [stats for stats in dir_pass2_inputs if stats]
        )
        conv_records = []
        for leaf_path in dir_to_files[dir_path]:
            conv_records.extend(scopes[f"file:{leaf_path}"]["raw_conversations"])

        scopes[scope_id] = {
            "id": scope_id,
            "label": path.name,
            "kind": "dir",
            "path": dir_path,
            "parent_id": parent_id,
            "children": [],
            "raw_pass1": pass1_stats,
            "raw_pass2": pass2_stats,
            "raw_conversations": conv_records,
            "pass1_data_path": None,
            "pass2_data_path": None,
            "descendant_files": dir_to_files[dir_path][:],
        }

    root_pass1 = _summary_stats(root_dir, pass1_summary_file) or merge_label_stats(
        [payload["raw_pass1"] for payload in scopes.values() if payload.get("kind") == "file" and payload.get("raw_pass1")]
    )
    root_pass2_inputs = [
        _with_scoring_file_label(
            payload.get("raw_pass2"),
            payload["path"],
        )
        for payload in scopes.values()
        if payload.get("kind") == "file"
    ]
    merged_root_pass2 = merge_scoring_stats([stats for stats in root_pass2_inputs if stats])
    root_pass2 = _summary_stats(root_dir, pass2_summary_file) or merged_root_pass2
    if root_pass2 and merged_root_pass2 and merged_root_pass2.get("per_file_summary"):
        root_pass2 = dict(root_pass2)
        root_pass2["per_file_summary"] = merged_root_pass2["per_file_summary"]
    root_conv = _summary_conversations(root_dir)
    if not root_conv:
        for payload in scopes.values():
            if payload.get("kind") == "file":
                root_conv.extend(payload.get("raw_conversations", []))

    scopes["global"]["raw_pass1"] = root_pass1
    scopes["global"]["raw_pass2"] = root_pass2
    scopes["global"]["raw_conversations"] = root_conv
    scopes["global"]["label"] = _infer_root_label(root_dir, root_pass1, root_pass2)

    child_map: dict[str, set[str]] = {scope_id: set() for scope_id in scopes}
    for scope_id, payload in scopes.items():
        parent_id = payload.get("parent_id")
        if parent_id:
            child_map.setdefault(parent_id, set()).add(scope_id)

    for scope_id, children in child_map.items():
        ordered = sorted(
            children,
            key=lambda child_id: (
                0 if scopes[child_id]["kind"] == "dir" else 1,
                scopes[child_id]["path"],
            ),
        )
        scopes[scope_id]["children"] = ordered

    return {
        "root_id": "global",
        "scopes": scopes,
    }
