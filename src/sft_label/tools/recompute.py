"""
Offline recomputation of stats and dashboards from pipeline output.

Provides two features:
  1. recompute-stats: Rebuild stats.json / stats_value.json from labeled/scored
     output without re-running the LLM pipeline.
  2. regenerate-dashboard: Re-generate HTML dashboards from existing stats/data.

LLM token usage stats will be zero in recomputed output (not preserved in
pipeline output). Recomputed files are marked with "recomputed": true.
"""

from __future__ import annotations

import json
from pathlib import Path
from datetime import datetime

from sft_label.config import CONFIDENCE_THRESHOLD


# ─── Sample loading ──────────────────────────────────────

def load_samples(path):
    """Load samples from a .json or .jsonl file.

    Returns list of sample dicts.
    """
    path = Path(path)
    if path.suffix == ".jsonl":
        samples = []
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    samples.append(json.loads(line))
        return samples

    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, list):
        return data
    raise ValueError(f"Expected list in {path}, got {type(data).__name__}")


def discover_labeled_files(input_dir):
    """Find labeled*.json/jsonl files in a run directory tree."""
    input_dir = Path(input_dir)
    files = []
    for pattern in ("labeled*.json", "labeled*.jsonl",
                     "*/labeled*.json", "*/labeled*.jsonl",
                     "*/*/labeled*.json", "*/*/labeled*.jsonl"):
        files.extend(input_dir.glob(pattern))
    return _dedup_json_jsonl(files)


def discover_scored_files(input_dir):
    """Find scored*.json/jsonl files in a run directory tree."""
    input_dir = Path(input_dir)
    files = []
    for pattern in ("scored*.json", "scored*.jsonl",
                     "*/scored*.json", "*/scored*.jsonl",
                     "*/*/scored*.json", "*/*/scored*.jsonl"):
        files.extend(input_dir.glob(pattern))
    return _dedup_json_jsonl(files)


def _dedup_json_jsonl(files):
    """Deduplicate .json vs .jsonl for the same stem in the same directory."""
    seen = {}
    for f in sorted(set(files)):
        key = (f.parent, f.stem)
        if key in seen:
            existing = seen[key]
            if existing.suffix == ".jsonl" and f.suffix == ".json":
                seen[key] = f
        else:
            seen[key] = f
    return sorted(seen.values())


# ─── Pass 1: Recompute stats from labeled output ────────

def _synthesize_monitor(sample):
    """Build a minimal monitor dict from a labeled sample's labeling_monitor."""
    lm = sample.get("labeling_monitor") or {}
    labels = sample.get("labels")
    return {
        "status": "success" if labels is not None else "failed",
        "llm_calls": lm.get("llm_calls", 0),
        "total_prompt_tokens": 0,
        "total_completion_tokens": 0,
        "arbitrated": lm.get("arbitrated", False),
        "validation_issues": lm.get("validation_issues", []),
        "consistency_warnings": lm.get("consistency_warnings", []),
        "low_confidence_dims": _extract_low_confidence_dims(labels),
    }


def _extract_low_confidence_dims(labels):
    """Extract low-confidence dimensions from labels."""
    if not labels or "confidence" not in labels:
        return []
    result = []
    for dim, score in labels.get("confidence", {}).items():
        if isinstance(score, (int, float)) and score < CONFIDENCE_THRESHOLD:
            result.append({"dim": dim, "score": score})
    return result


def _build_inherit_map(samples):
    """Reconstruct inherit_map from labels.inherited flag."""
    inherit_map = {}
    for idx, s in enumerate(samples):
        labels = s.get("labels")
        if labels and labels.get("inherited"):
            inherit_map[idx] = -1  # source index unknown, but presence is enough
    return inherit_map


def recompute_stats_from_labeled(samples):
    """Rebuild stats.json content from labeled output samples.

    Returns a dict matching the stats.json structure.
    """
    from sft_label.pipeline import compute_stats

    all_monitors = [_synthesize_monitor(s) for s in samples]
    all_labels = [s.get("labels") for s in samples]
    inherit_map = _build_inherit_map(samples)

    stats = compute_stats(all_monitors, all_labels, inherit_map=inherit_map)

    # Count sparse stats
    n_inherited = len(inherit_map)
    n_labeled = sum(1 for l in all_labels if l is not None) - n_inherited
    stats["sparse_labeled"] = n_labeled
    stats["sparse_inherited"] = n_inherited

    # Mark as recomputed
    stats["recomputed"] = True
    stats["recomputed_at"] = datetime.now().isoformat()
    stats["total_prompt_tokens"] = 0
    stats["total_completion_tokens"] = 0
    stats["total_tokens"] = 0

    return stats


# ─── Pass 2: Recompute value stats from scored output ────

def _synthesize_value_monitor(sample):
    """Build a minimal scoring monitor from a scored sample."""
    v = sample.get("value")
    return {
        "status": "success" if v else "failed",
        "llm_calls": 1 if v else 0,
        "prompt_tokens": 0,
        "completion_tokens": 0,
    }


def recompute_value_stats_from_scored(samples):
    """Rebuild stats_value.json content from scored output samples.

    Returns a dict matching the stats_value.json structure.
    """
    from sft_label.scoring import compute_value_stats, compute_selection_scores

    # Recompute selection scores in-place (needed for accurate stats)
    scored = [s for s in samples if s.get("value")]
    if scored:
        compute_selection_scores(scored)

    all_monitors = [_synthesize_value_monitor(s) for s in samples]
    stats = compute_value_stats(samples, all_monitors)

    # Mark as recomputed
    stats["recomputed"] = True
    stats["recomputed_at"] = datetime.now().isoformat()
    stats["total_llm_calls"] = 0
    stats["total_prompt_tokens"] = 0
    stats["total_completion_tokens"] = 0
    stats["total_tokens"] = 0

    return stats


# ─── Orchestration ───────────────────────────────────────

def run_recompute(input_path, pass_num="both", output_dir=None):
    """Recompute stats from labeled/scored pipeline output.

    Args:
        input_path: Path to a single file or run directory.
        pass_num: "1" (Pass 1 only), "2" (Pass 2 only), or "both".
        output_dir: Where to write stats. Default: same directory as input.

    Returns:
        dict with paths of written files.
    """
    input_path = Path(input_path)
    written = {}

    if input_path.is_file():
        out_dir = Path(output_dir) if output_dir else input_path.parent
        out_dir.mkdir(parents=True, exist_ok=True)
        written.update(_recompute_single_file(input_path, pass_num, out_dir))
    elif input_path.is_dir():
        out_dir = Path(output_dir) if output_dir else input_path
        out_dir.mkdir(parents=True, exist_ok=True)
        written.update(_recompute_directory(input_path, pass_num, out_dir))
    else:
        raise FileNotFoundError(f"Input path not found: {input_path}")

    return written


def _recompute_single_file(file_path, pass_num, out_dir):
    """Recompute stats for a single labeled/scored file."""
    written = {}
    samples = load_samples(file_path)

    if pass_num in ("1", "both"):
        # Check if samples have labels (Pass 1 output)
        has_labels = any(s.get("labels") for s in samples)
        if has_labels:
            stats = recompute_stats_from_labeled(samples)
            stats_path = out_dir / "stats.json"
            _write_json(stats_path, stats)
            written["stats"] = str(stats_path)
            print(f"  Pass 1 stats: {stats_path} "
                  f"({stats['success']}/{stats['total_samples']} success)")

    if pass_num in ("2", "both"):
        # Check if samples have value scores (Pass 2 output)
        has_scores = any(s.get("value") for s in samples)
        if has_scores:
            stats = recompute_value_stats_from_scored(samples)
            stats_path = out_dir / "stats_value.json"
            _write_json(stats_path, stats)
            written["stats_value"] = str(stats_path)
            print(f"  Pass 2 stats: {stats_path} "
                  f"({stats['total_scored']} scored)")

            # Also recompute conversation aggregation
            written.update(_recompute_conversations(samples, out_dir))

    return written


def _recompute_conversations(samples, out_dir):
    """Recompute conversation_scores.json from scored samples."""
    from sft_label.conversation import (
        aggregate_conversations, write_conversation_scores,
    )
    records = aggregate_conversations(samples)
    if records:
        conv_path = out_dir / "conversation_scores.json"
        write_conversation_scores(records, conv_path)
        print(f"  Conversations: {conv_path} ({len(records)} conversations)")
        return {"conversation_scores": str(conv_path)}
    return {}


def _recompute_directory(input_dir, pass_num, out_dir):
    """Recompute stats for all files in a run directory."""
    from sft_label.pipeline import merge_stats
    from sft_label.scoring import merge_value_stats

    written = {}
    all_pass1_stats = []
    all_pass2_stats = []

    # Pass 1: labeled files
    if pass_num in ("1", "both"):
        labeled_files = discover_labeled_files(input_dir)
        if labeled_files:
            print(f"Found {len(labeled_files)} labeled file(s)")
            for lf in labeled_files:
                print(f"\n  Processing: {lf.name}")
                samples = load_samples(lf)
                stats = recompute_stats_from_labeled(samples)
                stats["file"] = lf.name

                # Write per-file stats alongside the labeled file
                file_out_dir = out_dir / lf.parent.relative_to(input_dir) \
                    if lf.parent != input_dir else out_dir
                file_out_dir.mkdir(parents=True, exist_ok=True)
                stats_path = file_out_dir / "stats.json"
                _write_json(stats_path, stats)
                print(f"    → {stats_path} "
                      f"({stats['success']}/{stats['total_samples']} success)")
                all_pass1_stats.append(stats)

            # Merge into summary
            if len(all_pass1_stats) > 0:
                summary = merge_stats(all_pass1_stats)
                summary["recomputed"] = True
                summary["recomputed_at"] = datetime.now().isoformat()
                summary_path = out_dir / "summary_stats.json"
                _write_json(summary_path, summary)
                written["summary_stats"] = str(summary_path)
                print(f"\n  Summary: {summary_path} "
                      f"({summary.get('success', 0)} total success)")

    # Pass 2: scored files
    if pass_num in ("2", "both"):
        scored_files = discover_scored_files(input_dir)
        if scored_files:
            print(f"\nFound {len(scored_files)} scored file(s)")
            all_scored_samples = []
            for sf in scored_files:
                print(f"\n  Processing: {sf.name}")
                samples = load_samples(sf)
                all_scored_samples.extend(samples)
                stats = recompute_value_stats_from_scored(samples)
                stats["file"] = sf.name

                file_out_dir = out_dir / sf.parent.relative_to(input_dir) \
                    if sf.parent != input_dir else out_dir
                file_out_dir.mkdir(parents=True, exist_ok=True)
                stats_path = file_out_dir / "stats_value.json"
                _write_json(stats_path, stats)
                print(f"    → {stats_path} "
                      f"({stats['total_scored']} scored)")
                all_pass2_stats.append(stats)

            # Merge into summary
            if len(all_pass2_stats) > 0:
                summary = merge_value_stats(all_pass2_stats)
                summary["recomputed"] = True
                summary["recomputed_at"] = datetime.now().isoformat()
                summary_path = out_dir / "summary_stats_value.json"
                _write_json(summary_path, summary)
                written["summary_stats_value"] = str(summary_path)
                print(f"\n  Summary: {summary_path} "
                      f"({summary.get('total_scored', 0)} total scored)")

            # Conversation aggregation from all scored samples
            if all_scored_samples:
                written.update(
                    _recompute_conversations(all_scored_samples, out_dir))

    return written


# ─── Dashboard regeneration ──────────────────────────────

def run_regenerate_dashboard(input_path, pass_num="both", open_browser=False):
    """Regenerate HTML dashboards from existing stats and data files.

    Args:
        input_path: Path to a run directory.
        pass_num: "1" (Pass 1 only), "2" (Pass 2 only), or "both".
        open_browser: Open generated dashboards in default browser.

    Returns:
        list of generated dashboard paths.
    """
    import webbrowser
    from sft_label.tools.visualize_labels import generate_dashboard
    from sft_label.tools.visualize_value import generate_value_dashboard

    input_path = Path(input_path)
    if not input_path.is_dir():
        raise ValueError(f"Expected a directory, got: {input_path}")

    generated = []

    # Discover subdirectories that contain pipeline output
    subdirs = _find_output_subdirs(input_path)
    is_batch = len(subdirs) > 0

    if is_batch:
        # Per-subdir dashboards
        for subdir in subdirs:
            generated.extend(
                _regenerate_for_dir(subdir, pass_num, generate_dashboard,
                                    generate_value_dashboard))

        # Global dashboards at top level
        generated.extend(
            _regenerate_global(input_path, pass_num, generate_dashboard,
                               generate_value_dashboard))
    else:
        # Single-file run: output is directly in input_path
        generated.extend(
            _regenerate_for_dir(input_path, pass_num, generate_dashboard,
                                generate_value_dashboard))

    if open_browser:
        for path in generated:
            webbrowser.open(f"file://{path.resolve()}")

    return generated


def _find_output_subdirs(run_dir):
    """Find subdirectories containing labeled/scored/stats files."""
    subdirs = []
    for child in sorted(run_dir.iterdir()):
        if not child.is_dir():
            continue
        has_output = any(
            (child / name).exists()
            for name in ("labeled.json", "labeled.jsonl",
                         "scored.json", "scored.jsonl",
                         "stats.json", "stats_value.json")
        )
        if has_output:
            subdirs.append(child)
        # Check one level deeper (e.g., run_dir/code/subdir/)
        for grandchild in sorted(child.iterdir()):
            if grandchild.is_dir():
                has_deep = any(
                    (grandchild / name).exists()
                    for name in ("labeled.json", "labeled.jsonl",
                                 "scored.json", "scored.jsonl",
                                 "stats.json", "stats_value.json")
                )
                if has_deep:
                    subdirs.append(grandchild)
    return subdirs


def _regenerate_for_dir(dir_path, pass_num, gen_p1, gen_p2):
    """Generate dashboards for a single output directory."""
    generated = []

    if pass_num in ("1", "both"):
        if (dir_path / "stats.json").exists():
            try:
                out = gen_p1(dir_path)
                generated.append(out)
                print(f"  Pass 1 dashboard: {out}")
            except Exception as e:
                print(f"  Warning: Pass 1 dashboard failed for {dir_path}: {e}")

    if pass_num in ("2", "both"):
        if (dir_path / "stats_value.json").exists():
            try:
                out_path = gen_p2(dir_path, quiet=True)
                if out_path:
                    generated.append(Path(out_path) if not isinstance(out_path, Path) else out_path)
                    print(f"  Pass 2 dashboard: {out_path}")
            except Exception as e:
                print(f"  Warning: Pass 2 dashboard failed for {dir_path}: {e}")

    return generated


def _regenerate_global(run_dir, pass_num, gen_p1, gen_p2):
    """Generate global/summary dashboards at the top-level run directory."""
    generated = []

    if pass_num in ("1", "both"):
        if (run_dir / "summary_stats.json").exists():
            try:
                out = gen_p1(run_dir, labeled_file=None,
                             stats_file="summary_stats.json",
                             output_file=f"dashboard_{run_dir.name}.html")
                generated.append(out)
                print(f"  Global Pass 1 dashboard: {out}")
            except Exception as e:
                print(f"  Warning: global Pass 1 dashboard failed: {e}")

    if pass_num in ("2", "both"):
        if (run_dir / "summary_stats_value.json").exists():
            try:
                out_path = gen_p2(run_dir, scored_file=None,
                                  stats_file="summary_stats_value.json",
                                  output_file=f"dashboard_value_{run_dir.name}.html",
                                  quiet=True)
                if out_path:
                    generated.append(Path(out_path) if not isinstance(out_path, Path) else out_path)
                    print(f"  Global Pass 2 dashboard: {out_path}")
            except Exception as e:
                print(f"  Warning: global Pass 2 dashboard failed: {e}")

    return generated


# ─── Helpers ─────────────────────────────────────────────

def _write_json(path, data):
    """Write dict to JSON file, stripping non-serializable internal keys."""
    # Remove internal keys like _raw_scores
    clean = {k: v for k, v in data.items() if not k.startswith("_")}
    with open(path, "w", encoding="utf-8") as f:
        json.dump(clean, f, ensure_ascii=False, indent=2)
