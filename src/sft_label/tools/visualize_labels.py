"""Interactive Pass 1 dashboard generation."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import webbrowser
from pathlib import Path

from sft_label.artifacts import (
    PASS1_CONVERSATION_STATS_FILE,
    PASS1_STATS_FILE,
    PASS1_SUMMARY_STATS_FILE,
    dashboard_data_dirname,
    resolve_dashboard_output,
    runtime_static_base_url,
)
from sft_label.tools.dashboard_explorer import build_explorer_assets
from sft_label.tools.dashboard_scopes import build_scope_tree
from sft_label.tools.dashboard_template import ensure_dashboard_runtime_assets, render_dashboard_html


from sft_label.tools.dashboard_aggregation import (
    _coverage_from_distributions,
    build_dashboard_manifest,
    build_pass1_viz,
    build_scope_detail_payload,
    build_scope_summary,
    infer_scope_turn_kind,
    infer_scope_turn_kind_from_path,
    iter_data_file,
    load_data_file,
    merge_turn_kinds,
)


def load_run(run_dir: Path, labeled_file="labeled.json", stats_file=PASS1_STATS_FILE):
    """Load samples and stats from a run directory."""
    run_dir = Path(run_dir)
    samples = []
    if labeled_file:
        labeled_path = run_dir / labeled_file
        if labeled_path.exists():
            with open(labeled_path, encoding="utf-8") as f:
                samples = json.load(f)

    stats = {}
    stats_path = run_dir / stats_file
    if stats_path.exists():
        with open(stats_path, encoding="utf-8") as f:
            stats = json.load(f)

    return samples, stats


def compute_viz_data(samples, stats):
    """Compute all visualization data needed for Pass 1 dashboards."""
    payload = build_pass1_viz(samples, stats)
    overview_fields = {
        "prompt_mode": stats.get("prompt_mode", "full"),
        "compact_prompt": bool(stats.get("compact_prompt")),
        "conversation_char_budget": stats.get("conversation_char_budget"),
        "fewshot_variant": stats.get("fewshot_variant"),
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


def _scope_summary(scope: dict) -> dict:
    summary_modes = build_scope_summary(scope)
    scope["summary_modes"] = summary_modes
    return summary_modes["sample"]


def _dashboard_subtitle(run_dir: Path, root_label: str | None = None) -> str:
    if run_dir.name == "meta_label_data" and root_label:
        dataset_path = run_dir.parent / root_label
        if dataset_path.exists():
            return str(dataset_path)
    return str(run_dir)


def _single_scope_payload(
    run_dir: Path,
    samples: list[dict],
    viz_data: dict,
    stats: dict,
    *,
    data_path: str | Path | None = None,
) -> dict:
    label = Path(stats.get("input_file") or run_dir.name or "current").name
    turn_kind = infer_scope_turn_kind(samples)
    if turn_kind == "unknown" and data_path:
        turn_kind = infer_scope_turn_kind_from_path(data_path)
    scopes = {
        "global": {
            "id": "global",
            "label": label,
            "kind": "file",
            "path": stats.get("input_file") or label,
            "parent_id": None,
            "children": [],
            "descendant_files": [stats.get("input_file") or label],
            "pass1": viz_data,
            "turn_kind": turn_kind,
        }
    }
    scopes["global"]["summary"] = _scope_summary(scopes["global"])
    return {
        "title": "SFT Labeling Dashboard",
        "title_key": "dashboard_title_labeling",
        "subtitle": _dashboard_subtitle(run_dir, label),
        "root_id": "global",
        "default_scope_id": "global",
        "initially_expanded": ["global"],
        "scopes": scopes,
    }


def _load_scope_samples(path: str | Path | None) -> list[dict]:
    if not path:
        return []
    return [row for row in iter_data_file(path)]


def _conversation_stats_path(
    *,
    data_path: str | Path | None = None,
    stats_path: str | Path | None = None,
) -> Path | None:
    if stats_path:
        return Path(stats_path).with_name(PASS1_CONVERSATION_STATS_FILE)
    if data_path:
        return Path(data_path).with_name(PASS1_CONVERSATION_STATS_FILE)
    return None


def _load_pass1_conversation_mode(
    *,
    data_path: str | Path | None = None,
    stats_path: str | Path | None = None,
) -> dict | None:
    conversation_path = _conversation_stats_path(data_path=data_path, stats_path=stats_path)
    if not conversation_path or not conversation_path.exists():
        return None
    ref_paths = [Path(path) for path in (data_path, stats_path) if path]
    try:
        conversation_mtime = conversation_path.stat().st_mtime
        if any(path.exists() and path.stat().st_mtime > conversation_mtime for path in ref_paths):
            return None
    except OSError:
        return None
    try:
        with open(conversation_path, encoding="utf-8") as f:
            payload = json.load(f)
    except (OSError, json.JSONDecodeError):
        return None
    return payload if isinstance(payload, dict) else None


def _merge_pass1_conversation_modes(modes: list[dict], fallback: dict | None) -> dict | None:
    valid_modes = [mode for mode in modes if isinstance(mode, dict)]
    if not valid_modes:
        return fallback

    total = sum(int(mode.get("total", 0) or 0) for mode in valid_modes)
    distribution_total = sum(int(mode.get("distribution_total", mode.get("total", 0)) or 0) for mode in valid_modes)

    distributions: dict[str, dict[str, int]] = {}
    for mode in valid_modes:
        for dim, rows in (mode.get("distributions") or {}).items():
            bucket = distributions.setdefault(dim, {})
            for tag, count in (rows or {}).items():
                bucket[tag] = bucket.get(tag, 0) + int(count)
    distributions = {
        dim: dict(sorted(rows.items(), key=lambda item: (-item[1], item[0])))
        for dim, rows in distributions.items()
    }

    confidence_stats = {}
    conf_accumulator = {}
    for mode in valid_modes:
        for dim, stats in (mode.get("confidence_stats") or {}).items():
            count = int((stats or {}).get("count", 0) or 0)
            if count <= 0:
                continue
            entry = conf_accumulator.setdefault(
                dim,
                {"sum": 0.0, "count": 0, "min": None, "max": None, "below_threshold": 0},
            )
            mean = float((stats or {}).get("mean", 0) or 0)
            entry["sum"] += mean * count
            entry["count"] += count
            entry["below_threshold"] += int((stats or {}).get("below_threshold", 0) or 0)
            current_min = (stats or {}).get("min")
            if isinstance(current_min, (int, float)):
                entry["min"] = current_min if entry["min"] is None else min(entry["min"], current_min)
            current_max = (stats or {}).get("max")
            if isinstance(current_max, (int, float)):
                entry["max"] = current_max if entry["max"] is None else max(entry["max"], current_max)
    for dim, entry in conf_accumulator.items():
        count = entry.get("count", 0) or 0
        if count <= 0:
            continue
        confidence_stats[dim] = {
            "mean": round(entry["sum"] / count, 3),
            "min": round(float(entry["min"]), 3) if isinstance(entry["min"], (int, float)) else 0,
            "max": round(float(entry["max"]), 3) if isinstance(entry["max"], (int, float)) else 0,
            "below_threshold": entry["below_threshold"],
            "count": count,
        }

    cross_counts = {}
    for mode in valid_modes:
        for key, count in ((mode.get("cross_matrix") or {}).get("data") or {}).items():
            cross_counts[key] = cross_counts.get(key, 0) + int(count)
    cross_rows = sorted({key.split("|", 1)[0] for key in cross_counts if "|" in key})
    difficulty_order = ["beginner", "intermediate", "upper-intermediate", "advanced", "expert"]
    cross_cols = [
        difficulty
        for difficulty in difficulty_order
        if any(cross_counts.get(f"{row}|{difficulty}", 0) for row in cross_rows)
    ]
    cross_matrix = {
        "rows": cross_rows,
        "cols": cross_cols,
        "data": dict(sorted(cross_counts.items(), key=lambda item: item[0])),
    }

    unmapped = {}
    for mode in valid_modes:
        by_dim = ((mode.get("unmapped_details") or {}).get("by_dimension") or {})
        for dim, rows in by_dim.items():
            bucket = unmapped.setdefault(dim, {})
            for row in rows or []:
                label = row.get("label")
                if not isinstance(label, str):
                    continue
                entry = bucket.setdefault(label, {"label": label, "count": 0, "examples": []})
                entry["count"] += int(row.get("count", 0) or 0)
                for example in row.get("examples") or []:
                    if len(entry["examples"]) >= 2:
                        break
                    entry["examples"].append(example)
    unmapped_by_dimension = {
        dim: sorted(rows.values(), key=lambda item: (-item["count"], item["label"]))
        for dim, rows in unmapped.items()
    }
    total_occurrences = sum(row["count"] for rows in unmapped_by_dimension.values() for row in rows)
    unmapped_details = {
        "total_occurrences": total_occurrences,
        "by_dimension": unmapped_by_dimension,
    }

    overview = dict((fallback or {}).get("overview") or {})
    overview["success_rate"] = round(distribution_total / max(total, 1), 4)
    overview["unmapped_unique"] = sum(len(rows) for rows in unmapped_by_dimension.values())

    mode = dict(fallback or {})
    mode.update(
        {
            "total": total,
            "distribution_total": distribution_total,
            "distributions": distributions,
            "unmapped_details": unmapped_details,
            "confidence_stats": confidence_stats,
            "cross_matrix": cross_matrix,
            "coverage": _coverage_from_distributions(distributions),
            "overview": overview,
            "unit_label": mode.get("unit_label", "units"),
            "mode_id": "conversation",
        }
    )
    return mode


def _inject_conversation_mode(pass1: dict | None, descendant_modes: list[dict]) -> dict | None:
    if not isinstance(pass1, dict):
        return pass1
    modes = dict(pass1.get("modes") or {})
    merged = _merge_pass1_conversation_modes(descendant_modes, modes.get("conversation"))
    if not merged:
        return pass1
    modes["conversation"] = merged
    payload = dict(pass1)
    payload["modes"] = modes
    return payload


def _tree_payload(run_dir: Path) -> tuple[dict, list[dict]]:
    tree = build_scope_tree(
        run_dir,
        pass1_stats_file=PASS1_STATS_FILE,
        pass1_summary_file=PASS1_SUMMARY_STATS_FILE,
        pass2_stats_file=None,
        pass2_summary_file=None,
    )
    scopes = {}
    explorer_sources = []
    file_pass1: dict[str, dict | None] = {}
    file_conversation_modes: dict[str, dict | None] = {}
    for scope_id, raw_scope in tree["scopes"].items():
        if raw_scope.get("kind") != "file":
            continue
        pass1 = None
        if raw_scope.get("raw_pass1"):
            conversation_mode = _load_pass1_conversation_mode(
                data_path=raw_scope.get("pass1_data_path"),
                stats_path=raw_scope.get("pass1_stats_path"),
            )
            if conversation_mode:
                pass1 = _inject_conversation_mode(
                    compute_viz_data([], raw_scope["raw_pass1"]),
                    [conversation_mode],
                )
            else:
                samples = _load_scope_samples(raw_scope.get("pass1_data_path"))
                pass1 = compute_viz_data(samples, raw_scope["raw_pass1"])
        file_pass1[scope_id] = pass1
        file_conversation_modes[scope_id] = ((pass1 or {}).get("modes") or {}).get("conversation")

    file_turn_kind = {
        scope_id: infer_scope_turn_kind_from_path(raw_scope.get("pass1_data_path"))
        for scope_id, raw_scope in tree["scopes"].items()
        if raw_scope.get("kind") == "file"
    }
    for scope_id, raw_scope in tree["scopes"].items():
        pass1 = None
        if raw_scope.get("raw_pass1"):
            if raw_scope.get("kind") == "file":
                pass1 = file_pass1.get(scope_id)
            else:
                base_pass1 = compute_viz_data([], raw_scope["raw_pass1"])
                descendant_modes = [
                    file_conversation_modes.get(f"file:{leaf_path}")
                    for leaf_path in raw_scope.get("descendant_files", [])
                ]
                pass1 = _inject_conversation_mode(base_pass1, descendant_modes)
        if raw_scope.get("kind") == "file":
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
            "turn_kind": turn_kind,
        }
        scopes[scope_id]["summary"] = _scope_summary(scopes[scope_id])
        data_path = raw_scope.get("pass1_data_path")
        if raw_scope.get("kind") == "file" and data_path:
            explorer_sources.append(
                {
                    "scope_id": scope_id,
                    "scope_path": raw_scope["path"],
                    "data_path": data_path,
                }
            )

    root_pass1 = tree["scopes"][tree["root_id"]].get("raw_pass1") or {}
    root_label = Path(root_pass1.get("input_path") or root_pass1.get("input_file") or run_dir.name).name
    scopes[tree["root_id"]]["label"] = root_label

    initially_expanded = ["global"]
    initially_expanded.extend(
        scope_id for scope_id, scope in scopes.items()
        if scope["kind"] == "dir" and scope.get("parent_id") == "global"
    )

    return {
        "title": "SFT Labeling Dashboard",
        "title_key": "dashboard_title_labeling",
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


def _scope_detail_filename(scope_id: str) -> str:
    safe = "".join(ch if ch.isalnum() else "_" for ch in scope_id).strip("_") or "scope"
    return f"{safe}.json"


def _scope_detail_script_filename(scope_id: str) -> str:
    safe = "".join(ch if ch.isalnum() else "_" for ch in scope_id).strip("_") or "scope"
    return f"{safe}.js"


def _write_dashboard_bundle(
    output_path: Path,
    payload: dict,
    explorer_sources: list[dict],
    *,
    dashboard_type: str,
    static_base_url: str | None = None,
) -> None:
    data_dir = output_path.with_name(dashboard_data_dirname(output_path.name))
    scopes_dir = data_dir / "scopes"
    explorer_dir = data_dir / "explorer"
    shutil.rmtree(data_dir, ignore_errors=True)
    scopes_dir.mkdir(parents=True, exist_ok=True)

    _attach_explorer_payload(payload, build_explorer_assets(output_path, explorer_sources, assets_dir=explorer_dir))
    for scope_id, scope in payload.get("scopes", {}).items():
        scope["detail_path"] = f"{data_dir.name}/scopes/{_scope_detail_filename(scope_id)}"
        scope["detail_script_path"] = f"{data_dir.name}/scopes/{_scope_detail_script_filename(scope_id)}"

    manifest = build_dashboard_manifest(
        title=payload.get("title", "Interactive Dashboard"),
        title_key=payload.get("title_key", "dashboard_title_generic"),
        subtitle=payload.get("subtitle", ""),
        root_id=payload["root_id"],
        default_scope_id=payload["default_scope_id"],
        initially_expanded=payload.get("initially_expanded", []),
        scopes=payload.get("scopes", {}),
        explorer=payload.get("explorer"),
    )
    (data_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False), encoding="utf-8")
    (data_dir / "manifest.js").write_text(
        "window.__SFT_DASHBOARD_DATA__ = window.__SFT_DASHBOARD_DATA__ || {manifest:null, scopes:{}};\n"
        f"window.__SFT_DASHBOARD_DATA__.manifest = {json.dumps(manifest, ensure_ascii=False)};\n",
        encoding="utf-8",
    )

    for scope_id, scope in payload.get("scopes", {}).items():
        detail = build_scope_detail_payload(scope)
        (scopes_dir / _scope_detail_filename(scope_id)).write_text(json.dumps(detail, ensure_ascii=False), encoding="utf-8")
        (scopes_dir / _scope_detail_script_filename(scope_id)).write_text(
            "window.__SFT_DASHBOARD_DATA__ = window.__SFT_DASHBOARD_DATA__ || {manifest:null, scopes:{}};\n"
            f"window.__SFT_DASHBOARD_DATA__.scopes[{json.dumps(scope_id, ensure_ascii=False)}] = {json.dumps(detail, ensure_ascii=False)};\n",
            encoding="utf-8",
        )

    static_base = ensure_dashboard_runtime_assets(output_path, static_base_url=static_base_url or runtime_static_base_url())
    bootstrap = {
        "dashboardType": dashboard_type,
        "manifestUrl": f"{data_dir.name}/manifest.json",
        "manifestScriptUrl": f"{data_dir.name}/manifest.js",
        "staticBaseUrl": static_base,
        "defaultScopeId": payload["default_scope_id"],
        "rootId": payload["root_id"],
    }
    output_path.write_text(
        render_dashboard_html({"title": payload.get("title", "Interactive Dashboard"), "bootstrap": bootstrap}),
        encoding="utf-8",
    )


def generate_dashboard(run_dir: Path, labeled_file="labeled.json",
                       stats_file=PASS1_STATS_FILE, output_file="dashboard_labeling.html",
                       static_base_url: str | None = None) -> Path:
    """Generate an interactive Pass 1 dashboard."""
    run_dir = Path(run_dir)
    output_path = resolve_dashboard_output(run_dir, output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if labeled_file is None:
        if stats_file == PASS1_SUMMARY_STATS_FILE:
            payload, explorer_sources = _tree_payload(run_dir)
        else:
            samples = []
            _, stats = load_run(run_dir, labeled_file=None, stats_file=stats_file)
            payload = None
            explorer_sources = []
            data_path = next(
                (candidate for candidate in (run_dir / "labeled.jsonl", run_dir / "labeled.json") if candidate.exists()),
                None,
            )
            conversation_mode = _load_pass1_conversation_mode(
                data_path=data_path,
                stats_path=run_dir / stats_file,
            )
            if data_path is not None and conversation_mode:
                pass1 = _inject_conversation_mode(compute_viz_data([], stats), [conversation_mode])
            else:
                if data_path is not None:
                    samples = load_data_file(data_path)
                pass1 = compute_viz_data(samples, stats)
            payload = _single_scope_payload(run_dir, samples, pass1, stats, data_path=data_path)
            if data_path is not None:
                explorer_sources.append(
                    {
                        "scope_id": "global",
                        "scope_path": payload["scopes"]["global"]["path"],
                        "data_path": str(data_path),
                    }
                )
    else:
        samples, stats = load_run(run_dir, labeled_file=labeled_file, stats_file=stats_file)
        data_path = next(
            (candidate for candidate in (run_dir / "labeled.jsonl", run_dir / labeled_file) if candidate.exists()),
            None,
        )
        payload = _single_scope_payload(run_dir, samples, compute_viz_data(samples, stats), stats, data_path=data_path)
        explorer_sources = []
        if data_path is not None:
            explorer_sources.append(
                {
                    "scope_id": "global",
                    "scope_path": payload["scopes"]["global"]["path"],
                    "data_path": str(data_path),
                }
            )

    _write_dashboard_bundle(
        output_path,
        payload,
        explorer_sources,
        dashboard_type="labeling",
        static_base_url=static_base_url,
    )
    return output_path


def main():
    parser = argparse.ArgumentParser(description="Generate an interactive label statistics dashboard")
    parser.add_argument("run_dir", help="Path to pipeline run directory")
    parser.add_argument("--open", action="store_true", help="Open in browser")
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    if not (run_dir / "labeled.json").exists() and not (run_dir / PASS1_SUMMARY_STATS_FILE).exists():
        print(f"Error: no labeling artifacts found in {run_dir}")
        sys.exit(1)

    out = generate_dashboard(run_dir)
    print(f"Dashboard: {out}")
    if args.open:
        webbrowser.open(f"file://{out.resolve()}")


if __name__ == "__main__":
    main()
