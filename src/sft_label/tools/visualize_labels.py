"""Interactive Pass 1 dashboard generation."""

from __future__ import annotations

import argparse
import json
import sys
import webbrowser
from pathlib import Path

from sft_label.artifacts import (
    PASS1_STATS_FILE,
    PASS1_SUMMARY_STATS_FILE,
    resolve_dashboard_output,
)
from sft_label.tools.dashboard_explorer import build_explorer_assets
from sft_label.tools.dashboard_scopes import build_scope_tree
from sft_label.tools.dashboard_template import render_dashboard_html


from sft_label.tools.dashboard_aggregation import build_pass1_viz, build_scope_summary, load_data_file


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
    return build_pass1_viz(samples, stats)


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


def _single_scope_payload(run_dir: Path, viz_data: dict, stats: dict) -> dict:
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
            "pass1": viz_data,
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
    file_sample_cache = {
        scope_id: load_data_file(raw_scope.get("pass1_data_path"))
        for scope_id, raw_scope in tree["scopes"].items()
        if raw_scope.get("kind") == "file"
    }
    for scope_id, raw_scope in tree["scopes"].items():
        samples = []
        for leaf_path in raw_scope.get("descendant_files", []):
            samples.extend(file_sample_cache.get(f"file:{leaf_path}", []))
        pass1 = compute_viz_data(samples, raw_scope["raw_pass1"]) if raw_scope.get("raw_pass1") else None
        scopes[scope_id] = {
            "id": raw_scope["id"],
            "label": raw_scope["label"],
            "kind": raw_scope["kind"],
            "path": raw_scope["path"],
            "parent_id": raw_scope["parent_id"],
            "children": raw_scope["children"],
            "descendant_files": raw_scope["descendant_files"],
            "pass1": pass1,
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


def generate_dashboard(run_dir: Path, labeled_file="labeled.json",
                       stats_file=PASS1_STATS_FILE, output_file="dashboard_labeling.html") -> Path:
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
            if data_path is not None:
                samples = load_data_file(data_path)
            payload = _single_scope_payload(run_dir, compute_viz_data(samples, stats), stats)
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
        payload = _single_scope_payload(run_dir, compute_viz_data(samples, stats), stats)
        explorer_sources = []
        data_path = next(
            (candidate for candidate in (run_dir / "labeled.jsonl", run_dir / labeled_file) if candidate.exists()),
            None,
        )
        if data_path is not None:
            explorer_sources.append(
                {
                    "scope_id": "global",
                    "scope_path": payload["scopes"]["global"]["path"],
                    "data_path": str(data_path),
                }
            )

    _attach_explorer_payload(payload, build_explorer_assets(output_path, explorer_sources))

    output_path.write_text(render_dashboard_html(payload), encoding="utf-8")
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
