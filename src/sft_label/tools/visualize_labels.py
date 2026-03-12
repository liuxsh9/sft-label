"""Interactive Pass 1 dashboard generation."""

from __future__ import annotations

import argparse
import json
import sys
import webbrowser
from collections import Counter
from pathlib import Path

from sft_label.artifacts import (
    PASS1_STATS_FILE,
    PASS1_SUMMARY_STATS_FILE,
    resolve_dashboard_output,
)
from sft_label.prompts import TAG_POOLS
from sft_label.tools.dashboard_explorer import build_explorer_assets
from sft_label.tools.dashboard_scopes import build_scope_tree
from sft_label.tools.dashboard_template import render_dashboard_html


DIMENSIONS = [
    "intent",
    "difficulty",
    "language",
    "domain",
    "concept",
    "task",
    "agentic",
    "constraint",
    "context",
]


def _group_unmapped_counts(stats: dict) -> dict[str, list[dict]]:
    """Group flattened stats unmapped tags by dimension."""
    grouped: dict[str, list[dict]] = {}
    counts_by_dim: dict[str, Counter] = {}
    for key, count in (stats.get("unmapped_tags") or {}).items():
        dim, value = str(key).split(":", 1) if ":" in str(key) else ("unknown", str(key))
        counts_by_dim.setdefault(dim, Counter())[value] += int(count)

    for dim, counter in counts_by_dim.items():
        grouped[dim] = [
            {"label": value, "count": count, "examples": []}
            for value, count in counter.most_common()
        ]
    return grouped


def _collect_unmapped_examples(
    samples: list[dict],
    *,
    per_tag_limit: int = 2,
) -> dict[tuple[str, str], list[dict]]:
    """Collect a few sample ids/query previews for each unmapped tag."""
    examples: dict[tuple[str, str], list[dict]] = {}
    for sample in samples:
        labels = sample.get("labels") or {}
        unmapped = labels.get("unmapped") or []
        if not unmapped:
            continue

        query = ""
        for msg in sample.get("conversations", []):
            if msg.get("from") == "human":
                query = str(msg.get("value", "")).replace("\n", " ")[:120]
                break

        seen: set[tuple[str, str]] = set()
        for item in unmapped:
            if isinstance(item, dict):
                dim = str(item.get("dimension", "unknown"))
                value = str(item.get("value", "?"))
            else:
                dim = "unknown"
                value = str(item)
            key = (dim, value)
            if key in seen:
                continue
            seen.add(key)
            bucket = examples.setdefault(key, [])
            if len(bucket) < per_tag_limit:
                bucket.append({"id": str(sample.get("id", "?")), "query": query})
    return examples


def _build_unmapped_details(samples: list[dict], stats: dict) -> dict:
    """Build dashboard-friendly unmapped tag detail payload."""
    grouped = _group_unmapped_counts(stats)
    example_map = _collect_unmapped_examples(samples)

    ordered_dims = sorted(
        grouped,
        key=lambda dim: sum(item["count"] for item in grouped.get(dim, [])),
        reverse=True,
    )
    by_dimension = {}
    total_occurrences = 0
    for dim in ordered_dims:
        rows = []
        for item in grouped.get(dim, []):
            count = int(item.get("count", 0))
            total_occurrences += count
            rows.append(
                {
                    "label": item["label"],
                    "count": count,
                    "examples": example_map.get((dim, item["label"]), []),
                }
            )
        by_dimension[dim] = rows

    return {
        "total_occurrences": total_occurrences,
        "by_dimension": by_dimension,
    }


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
    distributions = stats.get("tag_distributions", {})

    conf_matrix = []
    for sample in samples:
        labels = sample.get("labels") or {}
        conf = labels.get("confidence", {})
        conf_matrix.append({
            "id": sample.get("id", "?"),
            "values": {dim: conf.get(dim, 0) for dim in DIMENSIONS},
        })

    coverage = {}
    for dim in DIMENSIONS:
        pool = set(TAG_POOLS.get(dim, []))
        used = set(distributions.get(dim, {}).keys())
        coverage[dim] = {
            "pool_size": len(pool),
            "used": len(used & pool),
            "unused": sorted(pool - used),
            "rate": len(used & pool) / len(pool) if pool else 0,
        }

    cross = Counter()
    for sample in samples:
        labels = sample.get("labels") or {}
        cross[(labels.get("intent", "?"), labels.get("difficulty", "?"))] += 1

    if not cross and stats.get("cross_matrix"):
        for key, count in (stats.get("cross_matrix") or {}).items():
            parts = key.split("|", 1)
            if len(parts) == 2:
                cross[(parts[0], parts[1])] = count

    intents = sorted({key[0] for key in cross})
    difficulty_order = ["beginner", "intermediate", "upper-intermediate", "advanced", "expert"]
    cross_matrix = {
        "rows": intents,
        "cols": [dim for dim in difficulty_order if any(cross.get((intent, dim), 0) for intent in intents)],
        "data": {f"{intent}|{diff}": cross.get((intent, diff), 0) for intent in intents for diff in difficulty_order},
    }

    return {
        "total": stats.get("total_samples", len(samples)) or len(samples),
        "input_file": stats.get("input_file") or stats.get("input_path", ""),
        "distributions": distributions,
        "unmapped_details": _build_unmapped_details(samples, stats),
        "confidence_stats": stats.get("confidence_stats", {}),
        "conf_matrix": conf_matrix,
        "coverage": coverage,
        "cross_matrix": cross_matrix,
        "overview": {
            "success_rate": stats.get("success_rate", 0),
            "total_tokens": stats.get("total_tokens", 0),
            "arbitrated_rate": stats.get("arbitrated_rate", 0),
            "unmapped_unique": stats.get("unmapped_unique_count", 0),
            "sparse_labeled": stats.get("sparse_labeled", 0),
            "sparse_inherited": stats.get("sparse_inherited", 0),
        },
    }


def _scope_summary(scope: dict) -> dict:
    pass1 = scope.get("pass1")
    return {
        "pass1_total": (pass1 or {}).get("total", 0),
        "scored_total": 0,
        "mean_value": None,
        "file_count": len(scope.get("descendant_files", [])) if scope.get("kind") != "file" else 1,
    }


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
    for scope_id, raw_scope in tree["scopes"].items():
        pass1 = compute_viz_data([], raw_scope["raw_pass1"]) if raw_scope.get("raw_pass1") else None
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
            _, stats = load_run(run_dir, labeled_file=None, stats_file=stats_file)
            payload = _single_scope_payload(run_dir, compute_viz_data([], stats), stats)
            explorer_sources = []
            data_path = next(
                (candidate for candidate in (run_dir / "labeled.jsonl", run_dir / "labeled.json") if candidate.exists()),
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
