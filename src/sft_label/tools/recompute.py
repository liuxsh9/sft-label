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
from concurrent.futures import ThreadPoolExecutor, as_completed

from sft_label.config import CONFIDENCE_THRESHOLD
from sft_label.artifacts import (
    PASS1_STATS_FILE,
    PASS1_SUMMARY_STATS_FILE,
    PASS1_DASHBOARD_FILE,
    PASS2_STATS_FILE,
    PASS2_SUMMARY_STATS_FILE,
    PASS2_DASHBOARD_FILE,
    pass1_global_dashboard_filename,
    pass2_global_dashboard_filename,
)
from sft_label.inline_scoring import (
    discover_inline_jsonl_files,
    infer_inline_scoring_target,
    write_inline_pass1_cache,
    write_inline_scored_cache,
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
    for pattern in ("**/labeled*.json", "**/labeled*.jsonl"):
        files.extend(input_dir.glob(pattern))
    return _dedup_json_jsonl(files)


def discover_scored_files(input_dir):
    """Find scored*.json/jsonl files in a run directory tree."""
    input_dir = Path(input_dir)
    files = []
    for pattern in ("**/scored*.json", "**/scored*.jsonl"):
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


def _resolve_worker_count(workers, total_items):
    """Normalize worker count for optional parallel execution."""
    if workers is None:
        workers = 1
    try:
        workers = int(workers)
    except (TypeError, ValueError) as e:
        raise ValueError(f"workers must be a positive integer, got {workers!r}") from e
    if workers < 1:
        raise ValueError(f"workers must be >= 1, got {workers}")
    if total_items <= 0:
        return 1
    return min(workers, total_items)


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
    n_labeled = sum(1 for labels in all_labels if labels is not None) - n_inherited
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


# ─── Inline mirrored dataset helpers ────────────────────

def _inline_file_dashboard_name(prefix: str, target, source_file: Path) -> str:
    rel = target.layout.relative_source_path(source_file).with_suffix("")
    stem = "__".join(rel.parts) if rel.parts else source_file.stem
    return f"{prefix}_{stem}.html"


def _rewrite_inline_per_file_summary(summary_path):
    """Rewrite artifact-relative per-file rows to mirrored dataset file paths."""
    summary_path = Path(summary_path)
    if not summary_path.exists():
        return
    with open(summary_path, encoding="utf-8") as f:
        summary = json.load(f)

    updated = False
    for row in summary.get("per_file_summary", []):
        file_label = row.get("file")
        if not isinstance(file_label, str) or not file_label:
            continue
        file_path = Path(file_label)
        parts = list(file_path.parts)
        if parts and parts[0] == "files":
            file_path = Path(*parts[1:])
        if file_path.name in {"scored.json", "scored.jsonl", "labeled.json", "labeled.jsonl"}:
            if file_path.parent != Path("."):
                row["file"] = f"{file_path.parent}.jsonl"
                updated = True

    if updated:
        _write_json(summary_path, summary)


def _materialize_inline_recompute_artifacts(target, pass_num):
    """Rebuild transient labeled/scored caches from inline rows."""
    source_files = discover_inline_jsonl_files(target)
    selected = source_files if not target.target_path.is_file() else [target.target_path.resolve()]
    for source_file in selected:
        artifact_dir = target.layout.file_artifact_dir(source_file)
        if pass_num in ("1", "both"):
            write_inline_pass1_cache(source_file, artifact_dir)
        if pass_num in ("2", "both"):
            write_inline_scored_cache(source_file, artifact_dir)
    return selected


def _run_recompute_inline(target, pass_num="both", workers=1):
    """Run recompute over an inline mirrored dataset via rebuildable caches."""
    _materialize_inline_recompute_artifacts(target, pass_num)

    if target.target_path.is_file():
        artifact_input = target.layout.file_artifact_dir(target.target_path)
    else:
        artifact_input = target.layout.meta_root

    written = _run_recompute_legacy(
        artifact_input,
        pass_num=pass_num,
        output_dir=artifact_input,
        workers=workers,
    )

    if target.target_path.is_file():
        if pass_num in ("1", "both"):
            pass1_stats = artifact_input / PASS1_STATS_FILE
            if pass1_stats.exists():
                written.setdefault("stats", str(pass1_stats))
        if pass_num in ("2", "both"):
            pass2_stats = artifact_input / PASS2_STATS_FILE
            if pass2_stats.exists():
                written.setdefault("stats_value", str(pass2_stats))
    if "summary_stats_value" in written:
        _rewrite_inline_per_file_summary(written["summary_stats_value"])
    return written


# ─── Orchestration ───────────────────────────────────────

def run_recompute(input_path, pass_num="both", output_dir=None, workers=1):
    inline_target = infer_inline_scoring_target(input_path)
    if inline_target is not None:
        expected_output = (
            inline_target.layout.file_artifact_dir(inline_target.target_path)
            if inline_target.target_path.is_file()
            else inline_target.layout.meta_root
        )
        if output_dir is not None and Path(output_dir).resolve() != expected_output.resolve():
            raise ValueError(
                "Inline recompute writes in-place under meta_label_data; "
                f"use --output {expected_output} or omit --output"
            )
        return _run_recompute_inline(inline_target, pass_num=pass_num, workers=workers)
    return _run_recompute_legacy(input_path, pass_num=pass_num, output_dir=output_dir, workers=workers)


def _run_recompute_legacy(input_path, pass_num="both", output_dir=None, workers=1):
    """Recompute stats from labeled/scored pipeline output.

    Args:
        input_path: Path to a single file or run directory.
        pass_num: "1" (Pass 1 only), "2" (Pass 2 only), or "both".
        output_dir: Where to write stats. Default: same directory as input.
        workers: max number of files to process in parallel in directory mode.

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
        written.update(_recompute_directory(input_path, pass_num, out_dir, workers=workers))
    else:
        raise FileNotFoundError(f"Input path not found: {input_path}")

    return written


def _sync_inline_scored_artifacts(target, source_files):
    """Persist refreshed scored caches back into inline mirrored rows."""
    from sft_label.scoring import _sync_inline_scored_cache_to_dataset

    for source_file in source_files:
        _sync_inline_scored_cache_to_dataset(
            source_file,
            target.layout.file_artifact_dir(source_file),
        )


def run_refresh_rarity(input_path, tag_stats_path=None, output_dir=None, config=None, workers=1):
    inline_target = infer_inline_scoring_target(input_path)
    if inline_target is not None:
        selected = _materialize_inline_recompute_artifacts(inline_target, pass_num="2")
        if inline_target.target_path.is_file():
            refresh_input = inline_target.layout.file_artifact_dir(inline_target.target_path)
        else:
            refresh_input = inline_target.layout.meta_root
        if output_dir is not None and Path(output_dir).resolve() != refresh_input.resolve():
            raise ValueError(
                "Inline rarity refresh writes in-place under meta_label_data; "
                f"use --output {refresh_input} or omit --output"
            )
        written = _run_refresh_rarity_legacy(
            refresh_input,
            tag_stats_path=tag_stats_path,
            output_dir=refresh_input,
            config=config,
            workers=workers,
        )
        _sync_inline_scored_artifacts(inline_target, selected)
        if "summary_stats_value" in written:
            _rewrite_inline_per_file_summary(written["summary_stats_value"])
        return written
    return _run_refresh_rarity_legacy(
        input_path,
        tag_stats_path=tag_stats_path,
        output_dir=output_dir,
        config=config,
        workers=workers,
    )


def _run_refresh_rarity_legacy(input_path, tag_stats_path=None, output_dir=None, config=None, workers=1):
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
        workers: max number of files to process in parallel in directory mode.

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

    def _build_local_combo_fallback():
        agg_combo_counts = {}
        _log_refresh_rarity(
            f"Building local combo fallback from {len(scored_files)} scored file(s)"
        )
        for index, sf in enumerate(scored_files, start=1):
            phase = f"[combo {index}/{len(scored_files)}] {_display_path(sf)}"
            size_hint = _format_file_size(sf)
            size_msg = f" ({size_hint})" if size_hint else ""
            _log_refresh_rarity(f"{phase}: loading samples{size_msg}")
            try:
                samples = load_samples(sf)
            except Exception as e:
                _log_refresh_rarity(
                    f"{phase}: failed while loading combo fallback source: "
                    f"{type(e).__name__}: {e}"
                )
                raise
            _merge_combo_counts(agg_combo_counts, build_combo_counts(samples))
        return agg_combo_counts

    def _resolve_stats_source():
        if tag_stats_path:
            p = Path(tag_stats_path)
            if not p.exists():
                raise FileNotFoundError(f"tag stats not found: {p}")
            return p
        if in_path.is_file():
            stats_path = in_path.parent / PASS1_STATS_FILE
            return stats_path if stats_path.exists() else None
        summary = in_path / PASS1_SUMMARY_STATS_FILE
        if summary.exists():
            return summary
        stats_path = in_path / PASS1_STATS_FILE
        return stats_path if stats_path.exists() else None

    stats_source_path = _resolve_stats_source()
    rarity_mode = resolve_rarity_mode(config)

    scored_files = [in_path] if in_path.is_file() else discover_scored_files(in_path)
    if not scored_files:
        return {}
    worker_count = _resolve_worker_count(workers, len(scored_files))

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
    if in_path.is_dir() and worker_count > 1:
        _log_refresh_rarity(
            f"Parallel mode enabled: workers={worker_count}"
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
                baseline_combo_counts = _build_local_combo_fallback()
                if baseline_combo_counts:
                    combo_mode = "hybrid"
                    _log_refresh_rarity(
                        "External stats has no combo_distributions; "
                        "using local combo fallback"
                    )
                else:
                    _log_refresh_rarity(
                        "External stats has no combo_distributions; "
                        "combo rarity disabled"
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

    def _process_scored_file(index, sf):
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
        _log_refresh_rarity(
            f"{phase}: wrote {scored_json.name}, {scored_jsonl.name}, {stats_path.name}"
        )
        return {
            "index": index,
            "file_path": sf,
            "phase": phase,
            "samples": samples,
            "stats": stats,
            "refreshed_count": refreshed_count,
            "target_dir": target_dir,
            "scored_json": scored_json,
            "scored_jsonl": scored_jsonl,
            "stats_path": stats_path,
        }

    written = {}
    all_file_stats = []
    all_samples = []
    refreshed_total = 0

    processed = []
    if worker_count > 1 and len(scored_files) > 1:
        done = 0
        processed_by_index = {}
        with ThreadPoolExecutor(max_workers=worker_count) as pool:
            future_map = {
                pool.submit(_process_scored_file, index, sf): index
                for index, sf in enumerate(scored_files, start=1)
            }
            for fut in as_completed(future_map):
                index = future_map[fut]
                processed_by_index[index] = fut.result()
                done += 1
                _log_refresh_rarity(
                    f"Parallel progress: {done}/{len(scored_files)} file(s) completed"
                )
        processed = [processed_by_index[i] for i in range(1, len(scored_files) + 1)]
    else:
        for index, sf in enumerate(scored_files, start=1):
            processed.append(_process_scored_file(index, sf))

    for result in processed:
        refreshed_total += result["refreshed_count"]
        all_samples.extend(result["samples"])
        stats = result["stats"]

        if in_path.is_file():
            written["scored_json"] = str(result["scored_json"])
            written["scored_jsonl"] = str(result["scored_jsonl"])
            written["stats_value"] = str(result["stats_path"])
            _log_refresh_rarity(
                f"{result['phase']}: recomputing conversation aggregation"
            )
            written.update(_recompute_conversations(result["samples"], result["target_dir"]))
        else:
            stats["file"] = _relative_file_label(result["file_path"], in_path)
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


def _write_conversation_records(records, out_dir):
    """Persist pre-aggregated conversation records when present."""
    from sft_label.conversation import write_conversation_scores

    if not records:
        return {}

    conv_path = out_dir / "conversation_scores.json"
    write_conversation_scores(records, conv_path)
    return {"conversation_scores": str(conv_path)}


def _recompute_directory(input_dir, pass_num, out_dir, workers=1):
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
            p1_workers = _resolve_worker_count(workers, len(labeled_files))
            p1_interval = _recompute_progress_interval(len(labeled_files))
            p1_compact = p1_interval > 1 and p1_workers == 1
            if p1_workers > 1 and len(labeled_files) > 1:
                print(f"  Pass 1 parallel mode: workers={p1_workers}")
            elif p1_compact:
                print(f"  Pass 1 compact mode: log every {p1_interval} files")

            def _process_labeled_file(idx, lf):
                rel_label = _relative_file_label(lf, input_dir)
                samples = load_samples(lf)
                stats = recompute_stats_from_labeled(samples)
                stats["file"] = rel_label
                file_out_dir = out_dir / lf.parent.relative_to(input_dir) \
                    if lf.parent != input_dir else out_dir
                file_out_dir.mkdir(parents=True, exist_ok=True)
                stats_path = file_out_dir / PASS1_STATS_FILE
                _write_json(stats_path, stats)
                return {
                    "idx": idx,
                    "rel_label": rel_label,
                    "stats_path": stats_path,
                    "stats": stats,
                }

            ordered_results = []
            if p1_workers > 1 and len(labeled_files) > 1:
                done = 0
                results_by_idx = {}
                with ThreadPoolExecutor(max_workers=p1_workers) as pool:
                    futures = {
                        pool.submit(_process_labeled_file, idx, lf): idx
                        for idx, lf in enumerate(labeled_files, start=1)
                    }
                    for fut in as_completed(futures):
                        item = fut.result()
                        results_by_idx[item["idx"]] = item
                        done += 1
                        if done == 1 or done == len(labeled_files) or done % p1_interval == 0:
                            print(
                                f"  Pass 1 progress {done}/{len(labeled_files)} | "
                                f"{item['rel_label']} | "
                                f"{item['stats']['success']}/{item['stats']['total_samples']} success"
                            )
                ordered_results = [results_by_idx[i] for i in range(1, len(labeled_files) + 1)]
            else:
                for idx, lf in enumerate(labeled_files, start=1):
                    rel_label = _relative_file_label(lf, input_dir)
                    if not p1_compact:
                        print(f"\n  Processing: {rel_label}")
                    item = _process_labeled_file(idx, lf)
                    if not p1_compact:
                        print(
                            f"    → {item['stats_path']} "
                            f"({item['stats']['success']}/{item['stats']['total_samples']} success)"
                        )
                    elif idx == 1 or idx == len(labeled_files) or idx % p1_interval == 0:
                        print(
                            f"  Pass 1 progress {idx}/{len(labeled_files)} | "
                            f"{item['rel_label']} | "
                            f"{item['stats']['success']}/{item['stats']['total_samples']} success"
                        )
                    ordered_results.append(item)

            all_pass1_stats.extend(item["stats"] for item in ordered_results)

            # Merge into summary
            if all_pass1_stats:
                summary = merge_stats(all_pass1_stats)
                summary["recomputed"] = True
                summary["recomputed_at"] = datetime.now().isoformat()
                summary_path = out_dir / PASS1_SUMMARY_STATS_FILE
                _write_json(summary_path, summary)
                written["summary_stats"] = str(summary_path)
                print(f"\n  Summary: {summary_path} "
                      f"({summary.get('success', 0)} total success)")

    # Pass 2: scored files
    if pass_num in ("2", "both"):
        scored_files = discover_scored_files(input_dir)
        if scored_files:
            print(f"\nFound {len(scored_files)} scored file(s)")
            all_conv_batches = []
            p2_workers = _resolve_worker_count(workers, len(scored_files))
            p2_interval = _recompute_progress_interval(len(scored_files))
            p2_compact = p2_interval > 1 and p2_workers == 1
            if p2_workers > 1 and len(scored_files) > 1:
                print(f"  Pass 2 parallel mode: workers={p2_workers}")
            elif p2_compact:
                print(f"  Pass 2 compact mode: log every {p2_interval} files")

            def _process_scored_file(idx, sf):
                from sft_label.conversation import aggregate_conversations

                rel_label = _relative_file_label(sf, input_dir)
                samples = load_samples(sf)
                stats = recompute_value_stats_from_scored(samples)
                stats["file"] = rel_label
                file_out_dir = out_dir / sf.parent.relative_to(input_dir) \
                    if sf.parent != input_dir else out_dir
                file_out_dir.mkdir(parents=True, exist_ok=True)
                stats_path = file_out_dir / PASS2_STATS_FILE
                _write_json(stats_path, stats)
                conversation_records = aggregate_conversations(samples)
                _write_conversation_records(conversation_records, file_out_dir)
                return {
                    "idx": idx,
                    "rel_label": rel_label,
                    "conversation_records": conversation_records,
                    "stats_path": stats_path,
                    "stats": stats,
                }

            ordered_results = []
            if p2_workers > 1 and len(scored_files) > 1:
                done = 0
                results_by_idx = {}
                with ThreadPoolExecutor(max_workers=p2_workers) as pool:
                    futures = {
                        pool.submit(_process_scored_file, idx, sf): idx
                        for idx, sf in enumerate(scored_files, start=1)
                    }
                    for fut in as_completed(futures):
                        item = fut.result()
                        results_by_idx[item["idx"]] = item
                        done += 1
                        if done == 1 or done == len(scored_files) or done % p2_interval == 0:
                            print(
                                f"  Pass 2 progress {done}/{len(scored_files)} | "
                                f"{item['rel_label']} | {item['stats']['total_scored']} scored"
                            )
                ordered_results = [results_by_idx[i] for i in range(1, len(scored_files) + 1)]
            else:
                for idx, sf in enumerate(scored_files, start=1):
                    rel_label = _relative_file_label(sf, input_dir)
                    if not p2_compact:
                        print(f"\n  Processing: {rel_label}")
                    item = _process_scored_file(idx, sf)
                    if not p2_compact:
                        print(
                            f"    → {item['stats_path']} "
                            f"({item['stats']['total_scored']} scored)"
                        )
                    elif idx == 1 or idx == len(scored_files) or idx % p2_interval == 0:
                        print(
                            f"  Pass 2 progress {idx}/{len(scored_files)} | "
                            f"{item['rel_label']} | {item['stats']['total_scored']} scored"
                        )
                    ordered_results.append(item)

            for item in ordered_results:
                all_pass2_stats.append(item["stats"])
                if item["conversation_records"]:
                    all_conv_batches.append(item["conversation_records"])

            # Merge into summary
            if all_pass2_stats:
                summary = merge_value_stats(all_pass2_stats)
                summary["recomputed"] = True
                summary["recomputed_at"] = datetime.now().isoformat()
                summary_path = out_dir / PASS2_SUMMARY_STATS_FILE
                _write_json(summary_path, summary)
                written["summary_stats_value"] = str(summary_path)
                print(f"\n  Summary: {summary_path} "
                      f"({summary.get('total_scored', 0)} total scored)")

            # Conversation aggregation from per-file records
            if all_conv_batches:
                from sft_label.conversation import merge_conversation_record_batches

                merged_records = merge_conversation_record_batches(all_conv_batches)
                written.update(_write_conversation_records(merged_records, out_dir))
                if merged_records:
                    print(
                        f"  Conversations: {out_dir / 'conversation_scores.json'} "
                        f"({len(merged_records)} conversations)"
                    )

    return written


# ─── Dashboard regeneration ──────────────────────────────

def _run_regenerate_dashboard_inline(target, pass_num="both", open_browser=False):
    import webbrowser
    from sft_label.tools.visualize_labels import generate_dashboard
    from sft_label.tools.visualize_value import generate_value_dashboard

    generated = []
    selected = discover_inline_jsonl_files(target)
    if target.target_path.is_file():
        selected = [target.target_path.resolve()]

    _log_regenerate_dashboard(
        f"Starting inline dashboard regeneration: input={target.target_path} pass={pass_num}"
    )

    for source_file in selected:
        artifact_dir = target.layout.file_artifact_dir(source_file)
        rel_label = target.layout.relative_source_path(source_file)

        if pass_num in ("1", "both"):
            pass1_stats_path = artifact_dir / PASS1_STATS_FILE
            if pass1_stats_path.exists():
                write_inline_pass1_cache(source_file, artifact_dir)
                _log_regenerate_dashboard(
                    f"{rel_label}: generating inline Pass 1 dashboard from {pass1_stats_path.name}"
                )
                out = generate_dashboard(
                    artifact_dir,
                    labeled_file="labeled.json",
                    stats_file=PASS1_STATS_FILE,
                    output_file=str(target.layout.dashboard_path(
                        _inline_file_dashboard_name("dashboard_labeling", target, source_file)
                    )),
                )
                generated.append(Path(out) if not isinstance(out, Path) else out)

        if pass_num in ("2", "both"):
            pass2_stats_path = artifact_dir / PASS2_STATS_FILE
            if pass2_stats_path.exists():
                write_inline_scored_cache(source_file, artifact_dir)
                _log_regenerate_dashboard(
                    f"{rel_label}: generating inline Pass 2 dashboard from {pass2_stats_path.name}"
                )
                out = generate_value_dashboard(
                    artifact_dir,
                    scored_file="scored.json",
                    stats_file=PASS2_STATS_FILE,
                    output_file=str(target.layout.dashboard_path(
                        _inline_file_dashboard_name("dashboard_scoring", target, source_file)
                    )),
                    quiet=True,
                )
                if out:
                    generated.append(Path(out) if not isinstance(out, Path) else out)

    if not target.target_path.is_file():
        if pass_num in ("1", "both"):
            pass1_summary_path = target.layout.meta_root / PASS1_SUMMARY_STATS_FILE
            if pass1_summary_path.exists():
                out = generate_dashboard(
                    target.layout.meta_root,
                    labeled_file=None,
                    stats_file=PASS1_SUMMARY_STATS_FILE,
                    output_file=str(target.layout.dashboard_path(
                        pass1_global_dashboard_filename(target.layout.dataset_root_name)
                    )),
                )
                generated.append(Path(out) if not isinstance(out, Path) else out)

        if pass_num in ("2", "both"):
            pass2_summary_path = target.layout.meta_root / PASS2_SUMMARY_STATS_FILE
            if pass2_summary_path.exists():
                out = generate_value_dashboard(
                    target.layout.meta_root,
                    scored_file=None,
                    stats_file=PASS2_SUMMARY_STATS_FILE,
                    output_file=str(target.layout.dashboard_path(
                        pass2_global_dashboard_filename(target.layout.dataset_root_name)
                    )),
                    quiet=True,
                )
                if out:
                    generated.append(Path(out) if not isinstance(out, Path) else out)

    if open_browser:
        for path in generated:
            resolved = Path(path).resolve()
            _log_regenerate_dashboard(f"Opening dashboard in browser: {resolved}")
            webbrowser.open(f"file://{resolved}")

    _log_regenerate_dashboard(
        f"Completed inline dashboard regeneration: generated {len(generated)} dashboard(s)"
    )
    return generated


def run_regenerate_dashboard(input_path, pass_num="both", open_browser=False, workers=1):
    inline_target = infer_inline_scoring_target(input_path)
    if inline_target is not None:
        return _run_regenerate_dashboard_inline(
            inline_target,
            pass_num=pass_num,
            open_browser=open_browser,
        )
    return _run_regenerate_dashboard_legacy(
        input_path,
        pass_num=pass_num,
        open_browser=open_browser,
        workers=workers,
    )


def _run_regenerate_dashboard_legacy(input_path, pass_num="both", open_browser=False, workers=1):
    """Regenerate HTML dashboards from existing stats and data files.

    Args:
        input_path: Path to a run directory.
        pass_num: "1" (Pass 1 only), "2" (Pass 2 only), or "both".
        open_browser: Open generated dashboards in default browser.
        workers: max number of directories to process in parallel.

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
        worker_count = _resolve_worker_count(workers, len(subdirs))
        _log_regenerate_dashboard(
            f"Detected batch mode with {len(subdirs)} output directorie(s)"
        )
        if worker_count > 1 and len(subdirs) > 1:
            _log_regenerate_dashboard(
                f"Parallel mode enabled: workers={worker_count}"
            )
        # Per-subdir dashboards
        if worker_count > 1 and len(subdirs) > 1:
            done = 0
            generated_by_index = {}
            with ThreadPoolExecutor(max_workers=worker_count) as pool:
                futures = {
                    pool.submit(
                        _regenerate_for_dir,
                        subdir,
                        pass_num,
                        generate_dashboard,
                        generate_value_dashboard,
                    ): index
                    for index, subdir in enumerate(subdirs, start=1)
                }
                for fut in as_completed(futures):
                    index = futures[fut]
                    subdir = subdirs[index - 1]
                    _log_regenerate_dashboard(
                        f"[dir {index}/{len(subdirs)}] Regenerating dashboards for {subdir}"
                    )
                    generated_by_index[index] = fut.result()
                    done += 1
                    _log_regenerate_dashboard(
                        f"Parallel progress: {done}/{len(subdirs)} directory(ies) completed"
                    )
            for index in range(1, len(subdirs) + 1):
                generated.extend(generated_by_index.get(index, []))
        else:
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
    pass1_stats_names = (PASS1_STATS_FILE,)
    pass2_stats_names = (PASS2_STATS_FILE,)
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
        pass1_stats_path = dir_path / PASS1_STATS_FILE
        if not pass1_stats_path.exists():
            pass1_stats_path = None
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
                generated.append(out)
                _log_regenerate_dashboard(f"{dir_path}: wrote Pass 1 dashboard {out}")
            except Exception as e:
                _log_regenerate_dashboard(
                    f"Warning: Pass 1 dashboard failed for {dir_path}: "
                    f"{type(e).__name__}: {e}"
                )
        else:
            legacy = dir_path / "stats.json"
            if legacy.exists():
                _log_regenerate_dashboard(
                    f"{dir_path}: found legacy stats.json; run "
                    f"'sft-label optimize-layout --input {dir_path} --apply' first"
                )
            else:
                _log_regenerate_dashboard(f"{dir_path}: skipped Pass 1 dashboard (no stats file)")

    if pass_num in ("2", "both"):
        pass2_stats_path = dir_path / PASS2_STATS_FILE
        if not pass2_stats_path.exists():
            pass2_stats_path = None
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
            legacy = dir_path / "stats_value.json"
            if legacy.exists():
                _log_regenerate_dashboard(
                    f"{dir_path}: found legacy stats_value.json; run "
                    f"'sft-label optimize-layout --input {dir_path} --apply' first"
                )
            else:
                _log_regenerate_dashboard(f"{dir_path}: skipped Pass 2 dashboard (no stats file)")

    return generated


def _regenerate_global(run_dir, pass_num, gen_p1, gen_p2):
    """Generate global/summary dashboards at the top-level run directory."""
    generated = []

    if pass_num in ("1", "both"):
        pass1_summary_path = run_dir / PASS1_SUMMARY_STATS_FILE
        if not pass1_summary_path.exists():
            pass1_summary_path = None
        if pass1_summary_path:
            try:
                _log_regenerate_dashboard(
                    f"{run_dir}: generating global Pass 1 dashboard from {pass1_summary_path.name}"
                )
                out = gen_p1(run_dir, labeled_file=None,
                             stats_file=pass1_summary_path.name,
                             output_file=pass1_global_dashboard_filename(run_dir.name))
                generated.append(out)
                _log_regenerate_dashboard(f"{run_dir}: wrote global Pass 1 dashboard {out}")
            except Exception as e:
                _log_regenerate_dashboard(
                    f"Warning: global Pass 1 dashboard failed for {run_dir}: "
                    f"{type(e).__name__}: {e}"
                )
        else:
            legacy = run_dir / "summary_stats.json"
            if legacy.exists():
                _log_regenerate_dashboard(
                    f"{run_dir}: found legacy summary_stats.json; run "
                    f"'sft-label optimize-layout --input {run_dir} --apply' first"
                )
            else:
                _log_regenerate_dashboard(
                    f"{run_dir}: skipped global Pass 1 dashboard (no summary stats)"
                )

    if pass_num in ("2", "both"):
        pass2_summary_path = run_dir / PASS2_SUMMARY_STATS_FILE
        if not pass2_summary_path.exists():
            pass2_summary_path = None
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
            legacy = run_dir / "summary_stats_value.json"
            if legacy.exists():
                _log_regenerate_dashboard(
                    f"{run_dir}: found legacy summary_stats_value.json; run "
                    f"'sft-label optimize-layout --input {run_dir} --apply' first"
                )
            else:
                _log_regenerate_dashboard(
                    f"{run_dir}: skipped global Pass 2 dashboard (no summary stats)"
                )

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
