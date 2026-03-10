"""
Offline recomputation of stats and dashboards from pipeline output.

Provides two features:
  1. recompute-stats: Rebuild stats_labeling.json / stats_scoring.json from labeled/scored
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
from sft_label.artifacts import (
    PASS1_STATS_FILE,
    PASS1_STATS_FILE_LEGACY,
    PASS1_SUMMARY_STATS_FILE,
    PASS1_SUMMARY_STATS_FILE_LEGACY,
    PASS1_DASHBOARD_FILE,
    PASS1_DASHBOARD_FILE_LEGACY,
    PASS2_STATS_FILE,
    PASS2_STATS_FILE_LEGACY,
    PASS2_SUMMARY_STATS_FILE,
    PASS2_SUMMARY_STATS_FILE_LEGACY,
    PASS2_DASHBOARD_FILE,
    PASS2_DASHBOARD_FILE_LEGACY,
    pass1_global_dashboard_filename,
    pass1_global_dashboard_legacy_filename,
    pass2_global_dashboard_filename,
    pass2_global_dashboard_legacy_filename,
    find_first_existing,
    sync_legacy_aliases,
)


# ─── Sample loading ──────────────────────────────────────

def load_samples(path):
    """Load samples from a .json or .jsonl file.

    Returns list of sample dicts.
    """
    path = Path(path)
    if path.suffix == ".jsonl":
        samples = []
        with open(path, encoding="utf-8") as f:
            for line_no, line in enumerate(f, start=1):
                line = line.strip()
                if line:
                    try:
                        samples.append(json.loads(line))
                    except json.JSONDecodeError as e:
                        raise ValueError(
                            f"Invalid JSONL in {path}:{line_no}:{e.colno}: {e.msg}"
                        ) from e
        return samples

    with open(path, encoding="utf-8") as f:
        try:
            data = json.load(f)
        except json.JSONDecodeError as e:
            raise ValueError(
                f"Invalid JSON in {path}:{e.lineno}:{e.colno}: {e.msg}"
            ) from e
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


def _log_refresh_rarity(message):
    """Emit a flushed progress line for refresh-rarity."""
    stamp = datetime.now().strftime("%H:%M:%S")
    print(f"[refresh-rarity {stamp}] {message}", flush=True)


def _log_regenerate_dashboard(message):
    """Emit a flushed progress line for regenerate-dashboard."""
    stamp = datetime.now().strftime("%H:%M:%S")
    print(f"[regenerate-dashboard {stamp}] {message}", flush=True)


def _format_file_size(path):
    """Best-effort human-readable file size string."""
    try:
        size = path.stat().st_size
    except OSError:
        return ""

    units = ("B", "KB", "MB", "GB", "TB")
    value = float(size)
    unit = units[0]
    for unit in units:
        if value < 1024 or unit == units[-1]:
            break
        value /= 1024.0

    if unit == "B":
        return f"{int(value)} {unit}"
    return f"{value:.1f} {unit}"


def _refresh_progress_interval(total):
    """Choose a progress cadence that stays readable on large files."""
    if total < 100:
        return 0
    return min(5000, max(100, total // 10))


def _relative_file_label(path, root):
    """Render a stable file label relative to the run root when possible."""
    path = Path(path)
    root = Path(root)
    try:
        return str(path.relative_to(root))
    except ValueError:
        return path.name


def _recompute_progress_interval(total_files):
    """Choose logging cadence for recompute-stats in large directories."""
    if total_files <= 20:
        return 1
    # Keep output roughly within ~20 progress lines per pass.
    return min(100, max(10, total_files // 20))


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
    """Rebuild Pass 1 stats content from labeled output samples.

    Returns a dict matching Pass 1 stats structure.
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
    """Rebuild Pass 2 stats content from scored output samples.

    Returns a dict matching Pass 2 stats structure.
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


def run_refresh_rarity(input_path, tag_stats_path=None, output_dir=None, config=None):
    """Refresh Pass 2 rarity/value fields offline from existing scored output.

    Recomputes per-sample:
      - value.rarity
      - value.value_score (rarity-aware composite)
      - value.selection_score / value.intra_class_rank
    and rewrites Pass 2 stats + conversation aggregation.

    Args:
        input_path: scored file or run directory.
        tag_stats_path: optional external Pass 1 stats path for cross-dataset rarity.
        output_dir: output root (default: in-place).
        config: PipelineConfig override; only rarity/value settings are used.

    Returns:
        dict with written artifact paths/counts.
    """
    from sft_label.config import PipelineConfig, VALUE_WEIGHTS, RARITY_WEIGHTS
    from sft_label.scoring import (
        load_tag_stats_context, compute_tag_idf, compute_sample_rarity,
        build_combo_counts, normalize_rarity_scores, resolve_rarity_mode,
        build_tag_distributions, compute_value_score, compute_selection_scores,
        merge_value_stats,
    )

    if config is None:
        config = PipelineConfig()

    in_path = Path(input_path)
    if not in_path.exists():
        raise FileNotFoundError(f"Input path not found: {in_path}")

    out_root = Path(output_dir) if output_dir else in_path.parent if in_path.is_file() else in_path
    out_root.mkdir(parents=True, exist_ok=True)

    def _merge_distributions(base, incoming):
        for dim, counts in incoming.items():
            dim_base = base.setdefault(dim, {})
            for tag, count in counts.items():
                dim_base[tag] = dim_base.get(tag, 0) + count

    def _merge_combo_counts(base, incoming):
        for combo_key, count in incoming.items():
            base[combo_key] = base.get(combo_key, 0) + count

    def _resolve_stats_source():
        if tag_stats_path:
            p = Path(tag_stats_path)
            if not p.exists():
                raise FileNotFoundError(f"tag stats not found: {p}")
            return p
        if in_path.is_file():
            return find_first_existing(in_path.parent, [PASS1_STATS_FILE, PASS1_STATS_FILE_LEGACY])
        summary = find_first_existing(
            in_path, [PASS1_SUMMARY_STATS_FILE, PASS1_SUMMARY_STATS_FILE_LEGACY]
        )
        if summary:
            return summary
        return find_first_existing(in_path, [PASS1_STATS_FILE, PASS1_STATS_FILE_LEGACY])

    stats_source_path = _resolve_stats_source()
    rarity_mode = resolve_rarity_mode(config)

    scored_files = [in_path] if in_path.is_file() else discover_scored_files(in_path)
    if not scored_files:
        return {}

    def _display_path(path):
        path = Path(path)
        if in_path.is_dir():
            try:
                return str(path.relative_to(in_path))
            except ValueError:
                return str(path)
        return path.name

    def _file_phase_label(path, index, total):
        return f"[file {index}/{total}] {_display_path(path)}"

    _log_refresh_rarity(
        f"Starting rarity refresh: input={in_path} output={out_root} "
        f"mode={rarity_mode} files={len(scored_files)}"
    )

    distributions = None
    total_stats_samples = 0
    stats_timestamp = None
    stats_source_str = None
    baseline_combo_counts = None
    combo_mode = "disabled"

    if stats_source_path:
        _log_refresh_rarity(f"Loading rarity baseline stats from {stats_source_path}")
        dists, total, ts, meta = load_tag_stats_context(stats_source_path)
        if dists:
            distributions = dists
            total_stats_samples = total
            stats_timestamp = ts
            stats_source_str = str(stats_source_path)
            baseline_combo_counts = meta.get("combo_counts")
            if baseline_combo_counts:
                combo_mode = "external"
            else:
                _log_refresh_rarity(
                    "External stats has no combo_distributions; "
                    "combo rarity disabled for cross-dataset comparability"
                )
            _log_refresh_rarity(
                f"Using external rarity baseline: {stats_source_path} "
                f"({total_stats_samples} samples)"
            )
        else:
            _log_refresh_rarity(
                f"Warning: {stats_source_path} has no tag_distributions; "
                "falling back to local baseline"
            )

    # Local fallback baseline (aggregated across all scored files in scope)
    if not distributions:
        agg_distributions = {}
        agg_total = 0
        agg_combo_counts = {}
        rw = config.rarity_weights or RARITY_WEIGHTS
        _log_refresh_rarity(
            f"Building local rarity baseline from {len(scored_files)} scored file(s)"
        )
        for index, sf in enumerate(scored_files, start=1):
            phase = f"[baseline {index}/{len(scored_files)}] {_display_path(sf)}"
            size_hint = _format_file_size(sf)
            size_msg = f" ({size_hint})" if size_hint else ""
            _log_refresh_rarity(f"{phase}: loading samples{size_msg}")
            try:
                samples = load_samples(sf)
            except Exception as e:
                _log_refresh_rarity(
                    f"{phase}: failed while loading baseline source: "
                    f"{type(e).__name__}: {e}"
                )
                raise
            _log_refresh_rarity(
                f"{phase}: loaded {len(samples)} sample(s); updating tag distributions"
            )
            local_dist, local_total = build_tag_distributions(samples, rarity_weights=rw)
            if local_dist:
                _merge_distributions(agg_distributions, local_dist)
            agg_total += local_total
            _merge_combo_counts(agg_combo_counts, build_combo_counts(samples))
        if agg_distributions and agg_total > 0:
            distributions = agg_distributions
            total_stats_samples = agg_total
            stats_timestamp = datetime.now().isoformat()
            stats_source_str = f"{in_path}#local"
            baseline_combo_counts = agg_combo_counts
            combo_mode = "local"
            _log_refresh_rarity(f"Using local rarity baseline ({agg_total} samples)")
        else:
            raise ValueError("No valid labels found to build rarity baseline")

    idf_map = compute_tag_idf(distributions, total_stats_samples)
    stats_ref_info = {
        "source": stats_source_str,
        "total_samples": total_stats_samples,
        "timestamp": stats_timestamp or datetime.now().isoformat(),
    }

    def _write_scored_outputs(samples, target_dir):
        scored_json = target_dir / "scored.json"
        scored_jsonl = target_dir / "scored.jsonl"
        _write_json(scored_json, samples)
        with open(scored_jsonl, "w", encoding="utf-8") as f:
            for s in samples:
                f.write(json.dumps(s, ensure_ascii=False) + "\n")
        return scored_json, scored_jsonl

    def _refresh_one(samples, phase):
        rarity_results = []
        combo_counts = baseline_combo_counts if idf_map else None
        rw = config.rarity_weights or RARITY_WEIGHTS
        total_samples_in_file = len(samples)
        progress_interval = _refresh_progress_interval(total_samples_in_file)
        _log_refresh_rarity(
            f"{phase}: computing rarity for {total_samples_in_file} sample(s)"
        )
        for index, s in enumerate(samples, start=1):
            labels = s.get("labels") or {}
            rarity = compute_sample_rarity(
                labels,
                idf_map,
                total_stats_samples,
                rarity_weights=rw,
                combo_alpha=config.rarity_combo_alpha,
                combo_counts=combo_counts,
                stats_ref_info=stats_ref_info,
            )
            rarity_results.append(rarity)
            if progress_interval and (
                index % progress_interval == 0 or index == total_samples_in_file
            ):
                pct = index / total_samples_in_file * 100.0
                _log_refresh_rarity(
                    f"{phase}: rarity {index}/{total_samples_in_file} ({pct:.1f}%)"
                )
        _log_refresh_rarity(f"{phase}: normalizing rarity scores (mode={rarity_mode})")
        normalize_rarity_scores(
            rarity_results,
            mode=rarity_mode,
            total_samples=total_stats_samples,
        )

        weights = config.value_weights or VALUE_WEIGHTS
        refreshed = 0
        _log_refresh_rarity(f"{phase}: updating value_score fields")
        for i, s in enumerate(samples):
            value = s.get("value")
            if not isinstance(value, dict):
                continue
            rarity = rarity_results[i]
            value["rarity"] = rarity
            value["value_score"] = compute_value_score(value, rarity, weights=weights)
            refreshed += 1

        _log_refresh_rarity(f"{phase}: recomputing selection scores")
        compute_selection_scores(samples, config=config)
        _log_refresh_rarity(f"{phase}: recomputing pass2 statistics")
        stats = recompute_value_stats_from_scored(samples)
        stats["rarity_config"] = {
            "stats_ref": stats_source_str,
            "total_samples_in_distribution": total_stats_samples,
            "dimension_weights": rw,
            "combo_alpha": config.rarity_combo_alpha,
            "combo_mode": combo_mode,
            "score_mode": rarity_mode,
        }
        _log_refresh_rarity(f"{phase}: refreshed {refreshed} scored sample(s)")
        return refreshed, stats

    written = {}
    all_file_stats = []
    all_samples = []
    refreshed_total = 0

    for index, sf in enumerate(scored_files, start=1):
        phase = _file_phase_label(sf, index, len(scored_files))
        size_hint = _format_file_size(sf)
        size_msg = f" ({size_hint})" if size_hint else ""
        _log_refresh_rarity(f"{phase}: loading scored file{size_msg}")
        try:
            samples = load_samples(sf)
        except Exception as e:
            _log_refresh_rarity(
                f"{phase}: failed while loading samples: {type(e).__name__}: {e}"
            )
            raise
        _log_refresh_rarity(f"{phase}: loaded {len(samples)} sample(s)")
        try:
            refreshed_count, stats = _refresh_one(samples, phase)
        except Exception as e:
            _log_refresh_rarity(
                f"{phase}: failed while refreshing rarity: {type(e).__name__}: {e}"
            )
            raise
        refreshed_total += refreshed_count
        all_samples.extend(samples)

        if in_path.is_file():
            target_dir = out_root
        else:
            rel = sf.parent.relative_to(in_path)
            target_dir = out_root / rel if rel != Path(".") else out_root
        target_dir.mkdir(parents=True, exist_ok=True)

        _log_refresh_rarity(f"{phase}: writing scored outputs to {target_dir}")
        try:
            scored_json, scored_jsonl = _write_scored_outputs(samples, target_dir)
        except Exception as e:
            _log_refresh_rarity(
                f"{phase}: failed while writing scored outputs: {type(e).__name__}: {e}"
            )
            raise
        stats_path = target_dir / PASS2_STATS_FILE
        try:
            _write_json(stats_path, stats)
        except Exception as e:
            _log_refresh_rarity(
                f"{phase}: failed while writing stats: {type(e).__name__}: {e}"
            )
            raise
        sync_legacy_aliases(stats_path, [PASS2_STATS_FILE_LEGACY])
        _log_refresh_rarity(
            f"{phase}: wrote {scored_json.name}, {scored_jsonl.name}, {stats_path.name}"
        )

        if in_path.is_file():
            written["scored_json"] = str(scored_json)
            written["scored_jsonl"] = str(scored_jsonl)
            written["stats_value"] = str(stats_path)
            _log_refresh_rarity(f"{phase}: recomputing conversation aggregation")
            written.update(_recompute_conversations(samples, target_dir))
        else:
            stats["file"] = _relative_file_label(sf, in_path)
            all_file_stats.append(stats)

    if in_path.is_dir() and all_file_stats:
        _log_refresh_rarity(
            f"Writing merged Pass 2 summary for {len(all_file_stats)} file(s)"
        )
        summary = merge_value_stats(all_file_stats)
        summary["recomputed"] = True
        summary["recomputed_at"] = datetime.now().isoformat()
        summary["rarity_config"] = {
            "stats_ref": stats_source_str,
            "total_samples_in_distribution": total_stats_samples,
            "dimension_weights": config.rarity_weights or RARITY_WEIGHTS,
            "combo_alpha": config.rarity_combo_alpha,
            "combo_mode": combo_mode,
            "score_mode": rarity_mode,
        }
        summary_path = out_root / PASS2_SUMMARY_STATS_FILE
        _write_json(summary_path, summary)
        sync_legacy_aliases(summary_path, [PASS2_SUMMARY_STATS_FILE_LEGACY])
        written["summary_stats_value"] = str(summary_path)
        written["files_refreshed"] = str(len(all_file_stats))
        _log_refresh_rarity(f"Wrote summary stats: {summary_path}")
        if all_samples:
            _log_refresh_rarity(
                f"Recomputing global conversation aggregation from {len(all_samples)} sample(s)"
            )
            written.update(_recompute_conversations(all_samples, out_root))

    written["rarity_refreshed_samples"] = str(refreshed_total)
    _log_refresh_rarity(
        f"Completed rarity refresh: refreshed {refreshed_total} sample(s)"
    )
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
            stats_path = out_dir / PASS1_STATS_FILE
            _write_json(stats_path, stats)
            sync_legacy_aliases(stats_path, [PASS1_STATS_FILE_LEGACY])
            written["stats"] = str(stats_path)
            print(f"  Pass 1 stats: {stats_path} "
                  f"({stats['success']}/{stats['total_samples']} success)")

    if pass_num in ("2", "both"):
        # Check if samples have value scores (Pass 2 output)
        has_scores = any(s.get("value") for s in samples)
        if has_scores:
            stats = recompute_value_stats_from_scored(samples)
            stats_path = out_dir / PASS2_STATS_FILE
            _write_json(stats_path, stats)
            sync_legacy_aliases(stats_path, [PASS2_STATS_FILE_LEGACY])
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
            p1_interval = _recompute_progress_interval(len(labeled_files))
            p1_compact = p1_interval > 1
            if p1_compact:
                print(f"  Pass 1 compact mode: log every {p1_interval} files")
            for idx, lf in enumerate(labeled_files, start=1):
                rel_label = _relative_file_label(lf, input_dir)
                if not p1_compact:
                    print(f"\n  Processing: {rel_label}")
                samples = load_samples(lf)
                stats = recompute_stats_from_labeled(samples)
                stats["file"] = _relative_file_label(lf, input_dir)

                # Write per-file stats alongside the labeled file
                file_out_dir = out_dir / lf.parent.relative_to(input_dir) \
                    if lf.parent != input_dir else out_dir
                file_out_dir.mkdir(parents=True, exist_ok=True)
                stats_path = file_out_dir / PASS1_STATS_FILE
                _write_json(stats_path, stats)
                sync_legacy_aliases(stats_path, [PASS1_STATS_FILE_LEGACY])
                if not p1_compact:
                    print(f"    → {stats_path} "
                          f"({stats['success']}/{stats['total_samples']} success)")
                elif idx == 1 or idx == len(labeled_files) or idx % p1_interval == 0:
                    print(
                        f"  Pass 1 progress {idx}/{len(labeled_files)} | "
                        f"{rel_label} | {stats['success']}/{stats['total_samples']} success"
                    )
                all_pass1_stats.append(stats)

            # Merge into summary
            if len(all_pass1_stats) > 0:
                summary = merge_stats(all_pass1_stats)
                summary["recomputed"] = True
                summary["recomputed_at"] = datetime.now().isoformat()
                summary_path = out_dir / PASS1_SUMMARY_STATS_FILE
                _write_json(summary_path, summary)
                sync_legacy_aliases(summary_path, [PASS1_SUMMARY_STATS_FILE_LEGACY])
                written["summary_stats"] = str(summary_path)
                print(f"\n  Summary: {summary_path} "
                      f"({summary.get('success', 0)} total success)")

    # Pass 2: scored files
    if pass_num in ("2", "both"):
        scored_files = discover_scored_files(input_dir)
        if scored_files:
            print(f"\nFound {len(scored_files)} scored file(s)")
            all_scored_samples = []
            p2_interval = _recompute_progress_interval(len(scored_files))
            p2_compact = p2_interval > 1
            if p2_compact:
                print(f"  Pass 2 compact mode: log every {p2_interval} files")
            for idx, sf in enumerate(scored_files, start=1):
                rel_label = _relative_file_label(sf, input_dir)
                if not p2_compact:
                    print(f"\n  Processing: {rel_label}")
                samples = load_samples(sf)
                all_scored_samples.extend(samples)
                stats = recompute_value_stats_from_scored(samples)
                stats["file"] = _relative_file_label(sf, input_dir)

                file_out_dir = out_dir / sf.parent.relative_to(input_dir) \
                    if sf.parent != input_dir else out_dir
                file_out_dir.mkdir(parents=True, exist_ok=True)
                stats_path = file_out_dir / PASS2_STATS_FILE
                _write_json(stats_path, stats)
                sync_legacy_aliases(stats_path, [PASS2_STATS_FILE_LEGACY])
                if not p2_compact:
                    print(f"    → {stats_path} "
                          f"({stats['total_scored']} scored)")
                elif idx == 1 or idx == len(scored_files) or idx % p2_interval == 0:
                    print(
                        f"  Pass 2 progress {idx}/{len(scored_files)} | "
                        f"{rel_label} | {stats['total_scored']} scored"
                    )
                all_pass2_stats.append(stats)

            # Merge into summary
            if len(all_pass2_stats) > 0:
                summary = merge_value_stats(all_pass2_stats)
                summary["recomputed"] = True
                summary["recomputed_at"] = datetime.now().isoformat()
                summary_path = out_dir / PASS2_SUMMARY_STATS_FILE
                _write_json(summary_path, summary)
                sync_legacy_aliases(summary_path, [PASS2_SUMMARY_STATS_FILE_LEGACY])
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
    _log_regenerate_dashboard(
        f"Starting dashboard regeneration: input={input_path} pass={pass_num}"
    )

    # Discover subdirectories that contain pipeline output
    subdirs = _find_output_subdirs(input_path)
    is_batch = len(subdirs) > 0

    if is_batch:
        _log_regenerate_dashboard(
            f"Detected batch mode with {len(subdirs)} output directorie(s)"
        )
        # Per-subdir dashboards
        for index, subdir in enumerate(subdirs, start=1):
            _log_regenerate_dashboard(
                f"[dir {index}/{len(subdirs)}] Regenerating dashboards for {subdir}"
            )
            generated.extend(
                _regenerate_for_dir(subdir, pass_num, generate_dashboard,
                                    generate_value_dashboard))

        # Global dashboards at top level
        _log_regenerate_dashboard(
            f"Regenerating global dashboards for {input_path}"
        )
        generated.extend(
            _regenerate_global(input_path, pass_num, generate_dashboard,
                               generate_value_dashboard))
    else:
        # Single-file run: output is directly in input_path
        _log_regenerate_dashboard(
            f"Detected single-directory mode for {input_path}"
        )
        generated.extend(
            _regenerate_for_dir(input_path, pass_num, generate_dashboard,
                                generate_value_dashboard))

    if open_browser:
        for path in generated:
            resolved = Path(path).resolve()
            _log_regenerate_dashboard(f"Opening dashboard in browser: {resolved}")
            webbrowser.open(f"file://{resolved}")

    _log_regenerate_dashboard(
        f"Completed dashboard regeneration: generated {len(generated)} dashboard(s)"
    )
    return generated


def _find_output_subdirs(run_dir):
    """Find subdirectories containing labeled/scored/stats files."""
    pass1_stats_names = (PASS1_STATS_FILE, PASS1_STATS_FILE_LEGACY)
    pass2_stats_names = (PASS2_STATS_FILE, PASS2_STATS_FILE_LEGACY)
    subdirs = []
    for child in sorted(run_dir.iterdir()):
        if not child.is_dir():
            continue
        has_output = any(
            (child / name).exists()
            for name in ("labeled.json", "labeled.jsonl",
                         "scored.json", "scored.jsonl",
                         *pass1_stats_names, *pass2_stats_names)
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
                                 *pass1_stats_names, *pass2_stats_names)
                )
                if has_deep:
                    subdirs.append(grandchild)
    return subdirs


def _regenerate_for_dir(dir_path, pass_num, gen_p1, gen_p2):
    """Generate dashboards for a single output directory."""
    generated = []

    if pass_num in ("1", "both"):
        pass1_stats_path = find_first_existing(
            dir_path, [PASS1_STATS_FILE, PASS1_STATS_FILE_LEGACY]
        )
        if pass1_stats_path:
            try:
                _log_regenerate_dashboard(
                    f"{dir_path}: generating Pass 1 dashboard from {pass1_stats_path.name}"
                )
                out = gen_p1(
                    dir_path,
                    stats_file=pass1_stats_path.name,
                    output_file=PASS1_DASHBOARD_FILE,
                )
                sync_legacy_aliases(Path(out), [PASS1_DASHBOARD_FILE_LEGACY])
                generated.append(out)
                _log_regenerate_dashboard(f"{dir_path}: wrote Pass 1 dashboard {out}")
            except Exception as e:
                _log_regenerate_dashboard(
                    f"Warning: Pass 1 dashboard failed for {dir_path}: "
                    f"{type(e).__name__}: {e}"
                )
        else:
            _log_regenerate_dashboard(f"{dir_path}: skipped Pass 1 dashboard (no stats file)")

    if pass_num in ("2", "both"):
        pass2_stats_path = find_first_existing(
            dir_path, [PASS2_STATS_FILE, PASS2_STATS_FILE_LEGACY]
        )
        if pass2_stats_path:
            try:
                _log_regenerate_dashboard(
                    f"{dir_path}: generating Pass 2 dashboard from {pass2_stats_path.name}"
                )
                out_path = gen_p2(
                    dir_path,
                    stats_file=pass2_stats_path.name,
                    output_file=PASS2_DASHBOARD_FILE,
                    quiet=True,
                )
                if out_path:
                    sync_legacy_aliases(Path(out_path), [PASS2_DASHBOARD_FILE_LEGACY])
                    generated.append(Path(out_path) if not isinstance(out_path, Path) else out_path)
                    _log_regenerate_dashboard(f"{dir_path}: wrote Pass 2 dashboard {out_path}")
                else:
                    _log_regenerate_dashboard(
                        f"{dir_path}: Pass 2 dashboard generator returned no output"
                    )
            except Exception as e:
                _log_regenerate_dashboard(
                    f"Warning: Pass 2 dashboard failed for {dir_path}: "
                    f"{type(e).__name__}: {e}"
                )
        else:
            _log_regenerate_dashboard(f"{dir_path}: skipped Pass 2 dashboard (no stats file)")

    return generated


def _regenerate_global(run_dir, pass_num, gen_p1, gen_p2):
    """Generate global/summary dashboards at the top-level run directory."""
    generated = []

    if pass_num in ("1", "both"):
        pass1_summary_path = find_first_existing(
            run_dir, [PASS1_SUMMARY_STATS_FILE, PASS1_SUMMARY_STATS_FILE_LEGACY]
        )
        if pass1_summary_path:
            try:
                _log_regenerate_dashboard(
                    f"{run_dir}: generating global Pass 1 dashboard from {pass1_summary_path.name}"
                )
                out = gen_p1(run_dir, labeled_file=None,
                             stats_file=pass1_summary_path.name,
                             output_file=pass1_global_dashboard_filename(run_dir.name))
                sync_legacy_aliases(Path(out), [pass1_global_dashboard_legacy_filename(run_dir.name)])
                generated.append(out)
                _log_regenerate_dashboard(f"{run_dir}: wrote global Pass 1 dashboard {out}")
            except Exception as e:
                _log_regenerate_dashboard(
                    f"Warning: global Pass 1 dashboard failed for {run_dir}: "
                    f"{type(e).__name__}: {e}"
                )
        else:
            _log_regenerate_dashboard(f"{run_dir}: skipped global Pass 1 dashboard (no summary stats)")

    if pass_num in ("2", "both"):
        pass2_summary_path = find_first_existing(
            run_dir, [PASS2_SUMMARY_STATS_FILE, PASS2_SUMMARY_STATS_FILE_LEGACY]
        )
        if pass2_summary_path:
            try:
                _log_regenerate_dashboard(
                    f"{run_dir}: generating global Pass 2 dashboard from {pass2_summary_path.name}"
                )
                out_path = gen_p2(run_dir, scored_file=None,
                                  stats_file=pass2_summary_path.name,
                                  output_file=pass2_global_dashboard_filename(run_dir.name),
                                  quiet=True)
                if out_path:
                    sync_legacy_aliases(Path(out_path), [pass2_global_dashboard_legacy_filename(run_dir.name)])
                    generated.append(Path(out_path) if not isinstance(out_path, Path) else out_path)
                    _log_regenerate_dashboard(f"{run_dir}: wrote global Pass 2 dashboard {out_path}")
                else:
                    _log_regenerate_dashboard(
                        f"{run_dir}: global Pass 2 dashboard generator returned no output"
                    )
            except Exception as e:
                _log_regenerate_dashboard(
                    f"Warning: global Pass 2 dashboard failed for {run_dir}: "
                    f"{type(e).__name__}: {e}"
                )
        else:
            _log_regenerate_dashboard(f"{run_dir}: skipped global Pass 2 dashboard (no summary stats)")

    return generated


# ─── Helpers ─────────────────────────────────────────────

def _write_json(path, data):
    """Write JSON file, stripping internal keys for dict payloads."""
    if isinstance(data, dict):
        clean = {k: v for k, v in data.items() if not k.startswith("_")}
    else:
        clean = data
    with open(path, "w", encoding="utf-8") as f:
        json.dump(clean, f, ensure_ascii=False, indent=2)
