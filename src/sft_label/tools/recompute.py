"""
Offline recomputation of stats and dashboards from pipeline output.

Provides two features:
  1. recompute-stats: Rebuild stats_labeling.json / stats_scoring.json from labeled/scored
     output without re-running the LLM pipeline.
  2. regenerate-dashboard: Re-generate HTML dashboards from existing stats/data.
  3. complete-postprocess: Materialize deferred Pass 2 conversation aggregation /
     dashboards offline after a large run finishes.

LLM token usage stats will be zero in recomputed output (not preserved in
pipeline output). Recomputed files are marked with "recomputed": true.
"""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed

from sft_label.config import CONFIDENCE_THRESHOLD, PASS2_HEAVY_POSTPROCESS_SAMPLE_THRESHOLD
from sft_label.artifacts import (
    PASS1_CONVERSATION_STATS_FILE,
    PASS1_STATS_FILE,
    PASS1_SUMMARY_STATS_FILE,
    PASS1_DASHBOARD_FILE,
    PASS2_STATS_FILE,
    PASS2_SUMMARY_STATS_FILE,
    PASS2_DASHBOARD_FILE,
    pass1_global_dashboard_filename,
    pass2_global_dashboard_filename,
    prune_dashboard_bundles,
)
from sft_label.inline_scoring import (
    discover_inline_jsonl_files,
    infer_inline_scoring_target,
    inline_source_has_embedded_scores,
    write_inline_pass1_cache,
    write_inline_scored_cache,
)
from sft_label.label_extensions_stats import (
    aggregate_extension_stats as _shared_aggregate_extension_stats,
    merge_extension_stats as _shared_merge_extension_stats,
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


def discover_pass1_source_files(input_dir):
    """Find files that can rebuild Pass 1 stats, preferring labeled over scored."""
    labeled_files = discover_labeled_files(input_dir)
    labeled_dirs = {path.parent.resolve() for path in labeled_files}
    scored_fallbacks = [
        path for path in discover_scored_files(input_dir)
        if path.parent.resolve() not in labeled_dirs
    ]
    return sorted([*labeled_files, *scored_fallbacks])


def _dedup_json_jsonl(files):
    """Deduplicate .json vs .jsonl for the same stem in the same directory."""
    seen = {}
    for f in sorted(set(files)):
        key = (f.parent, f.stem)
        if key in seen:
            existing = seen[key]
            if existing.suffix == ".json" and f.suffix == ".jsonl":
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


def _log_complete_postprocess(message):
    """Emit a flushed progress line for complete-postprocess."""
    stamp = datetime.now().strftime("%H:%M:%S")
    print(f"[complete-postprocess {stamp}] {message}", flush=True)


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


def _extract_extension_payloads(sample: dict) -> dict[str, dict]:
    """Return extension payloads from either inline or legacy storage."""
    if isinstance(sample.get("label_extensions"), dict):
        return sample["label_extensions"]
    labels = sample.get("labels") or {}
    if isinstance(labels.get("label_extensions"), dict):
        return labels["label_extensions"]
    return {}


def _accumulate_confidence_stat(acc: dict, field: str, score: float) -> None:
    stats = acc.setdefault(field, {"sum": 0.0, "min": score, "max": score, "count": 0})
    stats["sum"] += float(score)
    stats["count"] += 1
    stats["min"] = min(stats["min"], float(score))
    stats["max"] = max(stats["max"], float(score))


def _accumulate_confidence_bulk(
    acc: dict,
    field: str,
    *,
    mean: float,
    count: int,
    min_val: float,
    max_val: float,
) -> None:
    if count <= 0:
        return
    stats = acc.setdefault(field, {"sum": 0.0, "min": min_val, "max": max_val, "count": 0})
    stats["sum"] += float(mean) * count
    stats["count"] += count
    stats["min"] = min(stats["min"], float(min_val))
    stats["max"] = max(stats["max"], float(max_val))


def _finalize_confidence_stats(acc: dict) -> dict[str, dict]:
    finalized = {}
    for field, stats in acc.items():
        count = stats.get("count", 0) or 0
        if count <= 0:
            continue
        finalized[field] = {
            "mean": round(stats["sum"] / count, 3),
            "min": round(stats["min"], 3),
            "max": round(stats["max"], 3),
            "count": count,
        }
    return finalized


def _aggregate_extension_stats(samples: list[dict]) -> dict | None:
    return _shared_aggregate_extension_stats(samples)


def _merge_extension_stats(stats_list: list[dict]) -> dict | None:
    return _shared_merge_extension_stats(stats_list)


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

    extension_stats = _aggregate_extension_stats(samples)
    if extension_stats:
        stats["extension_stats"] = extension_stats

    return stats


def _write_pass1_conversation_stats(samples, stats, out_dir):
    """Persist lightweight Pass 1 conversation-mode stats for dashboards."""
    from sft_label.tools.visualize_labels import compute_viz_data

    conversation_mode = ((compute_viz_data(samples, stats) or {}).get("modes") or {}).get("conversation")
    if isinstance(conversation_mode, dict):
        _write_json(Path(out_dir) / PASS1_CONVERSATION_STATS_FILE, conversation_mode)


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
    from sft_label.scoring import compute_value_stats, compute_selection_scores, apply_v2_scores

    # Recompute selection scores in-place (needed for accurate stats)
    scored = [s for s in samples if s.get("value")]
    if scored:
        compute_selection_scores(scored)
        apply_v2_scores(scored)

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
        augment_rarity_result, _refresh_augmented_rarity_after_core_normalization,
        apply_v2_scores, _attach_augmented_rarity_fields, _clear_augmented_rarity_fields,
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
    extension_stats = None
    extension_baseline_source = "external"
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
            extension_stats = meta.get("extension_stats")
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
        agg_extension_stats = []
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
            local_extension_stats = _aggregate_extension_stats(samples)
            if local_extension_stats:
                agg_extension_stats.append({"extension_stats": local_extension_stats})
        if agg_distributions and agg_total > 0:
            distributions = agg_distributions
            total_stats_samples = agg_total
            stats_timestamp = datetime.now().isoformat()
            stats_source_str = f"{in_path}#local"
            baseline_combo_counts = agg_combo_counts
            extension_stats = _merge_extension_stats(agg_extension_stats)
            extension_baseline_source = "local"
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
            core_rarity = compute_sample_rarity(
                labels,
                idf_map,
                total_stats_samples,
                rarity_weights=rw,
                combo_alpha=config.rarity_combo_alpha,
                combo_counts=combo_counts,
                stats_ref_info=stats_ref_info,
            )
            rarity = augment_rarity_result(
                core_rarity,
                s,
                extension_stats=extension_stats,
                config=config,
                baseline_source=extension_baseline_source,
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
        for rarity in rarity_results:
            _refresh_augmented_rarity_after_core_normalization(rarity)

        weights = config.value_weights or VALUE_WEIGHTS
        refreshed = 0
        _log_refresh_rarity(f"{phase}: updating value_score fields")
        for i, s in enumerate(samples):
            value = s.get("value")
            if not isinstance(value, dict):
                continue
            rarity = rarity_results[i]
            extension_mode = getattr(config, "extension_rarity_mode", "off")
            _clear_augmented_rarity_fields(
                value,
                keep_preview=str(extension_mode).strip().lower() == "preview",
            )
            value["rarity"] = rarity
            _attach_augmented_rarity_fields(value, rarity)
            value["value_score"] = compute_value_score(value, rarity, weights=weights)
            refreshed += 1

        _log_refresh_rarity(f"{phase}: recomputing selection scores")
        compute_selection_scores(samples, config=config)
        apply_v2_scores(samples, config=config)
        _log_refresh_rarity(f"{phase}: recomputing pass2 statistics")
        stats = recompute_value_stats_from_scored(samples)
        stats["rarity_config"] = {
            "stats_ref": stats_source_str,
            "total_samples_in_distribution": total_stats_samples,
            "dimension_weights": rw,
            "combo_alpha": config.rarity_combo_alpha,
            "combo_mode": combo_mode,
            "score_mode": rarity_mode,
            "extension_rarity_mode": getattr(config, "extension_rarity_mode", "off"),
            "extension_baseline_source": extension_baseline_source,
        }
        stats["extension_rarity_config"] = {
            "mode": getattr(config, "extension_rarity_mode", "off"),
            "baseline_source": extension_baseline_source,
            "min_extension_baseline_total": getattr(config, "min_extension_baseline_total", None),
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
            "extension_rarity_mode": getattr(config, "extension_rarity_mode", "off"),
            "extension_baseline_source": extension_baseline_source,
        }
        summary["extension_rarity_config"] = {
            "mode": getattr(config, "extension_rarity_mode", "off"),
            "baseline_source": extension_baseline_source,
            "min_extension_baseline_total": getattr(config, "min_extension_baseline_total", None),
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
            _write_pass1_conversation_stats(samples, stats, out_dir)
            written["stats"] = str(stats_path)
            print(f"  Pass 1 stats: {stats_path} "
                  f"({stats['success']}/{stats['total_samples']} success)")

    if pass_num in ("2", "both"):
        # Check if samples have value scores (Pass 2 output)
        has_scores = any(s.get("value") for s in samples)
        if has_scores:
            stats = recompute_value_stats_from_scored(samples)
            stats_path = out_dir / PASS2_STATS_FILE
            _preserve_existing_pass2_postprocess(stats_path, stats)
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
    conv_path = Path(out_dir) / "conversation_scores.json"
    if conv_path.exists():
        conv_path.unlink()
    return {}


def _write_conversation_records(records, out_dir, *, clear_when_empty=False):
    """Persist pre-aggregated conversation records when present."""
    from sft_label.conversation import write_conversation_scores

    conv_path = Path(out_dir) / "conversation_scores.json"
    if not records:
        if clear_when_empty and conv_path.exists():
            conv_path.unlink()
        return {}

    write_conversation_scores(records, conv_path)
    return {"conversation_scores": str(conv_path)}


def _extract_postprocess_payload(stats: dict | None) -> dict | None:
    if not isinstance(stats, dict):
        return None
    postprocess = stats.get("postprocess")
    if isinstance(postprocess, dict) and postprocess:
        normalized: dict[str, dict] = {}
        for key in ("conversation_scores", "dashboard"):
            node = postprocess.get(key)
            status = (
                str((node or {}).get("status") or "").strip().lower()
                if isinstance(node, dict)
                else ""
            )
            normalized[key] = dict(node) if status else {"status": "missing"}
        return normalized
    return None


def _preserve_existing_pass2_postprocess(stats_path: Path, stats: dict) -> None:
    existing = _load_json_payload(stats_path) if Path(stats_path).exists() else {}
    postprocess = _extract_postprocess_payload(existing)
    if postprocess:
        stats["postprocess"] = postprocess


def _merge_summary_postprocess_from_file_stats(all_pass2_stats: list[dict]) -> dict | None:
    """Fail-closed summary postprocess reconstruction from per-file metadata."""
    status_priority = {
        "failed": 7,
        "conflict": 6,
        "missing": 5,
        "pending": 4,
        "deferred": 3,
        "completed": 2,
        "disabled": 1,
    }
    merged: dict[str, dict] = {}

    for field in ("conversation_scores", "dashboard"):
        candidates: list[dict] = []
        for stats in all_pass2_stats:
            postprocess = _extract_postprocess_payload(stats)
            if not postprocess:
                continue
            payload = postprocess.get(field)
            if isinstance(payload, dict):
                candidates.append(dict(payload))
        if not candidates:
            continue
        chosen = max(
            candidates,
            key=lambda payload: status_priority.get(str(payload.get("status") or "").strip().lower(), 0),
        )
        merged[field] = chosen

    return merged or None


def _refresh_inline_pass2_stats_from_cache(stats_path: Path) -> dict:
    existing = _load_json_payload(stats_path) if Path(stats_path).exists() else {}
    refreshed = recompute_value_stats_from_scored([])
    for key in (
        "input_file",
        "input_path",
        "run_dir",
        "mirrored_file",
        "file",
        "prompt_mode",
        "compact_prompt",
        "value_truncation_budget",
    ):
        if key in existing:
            refreshed[key] = existing[key]
    _preserve_existing_pass2_postprocess(stats_path, refreshed)
    _write_json(stats_path, refreshed)
    return refreshed


def _recompute_directory(input_dir, pass_num, out_dir, workers=1):
    """Recompute stats for all files in a run directory."""
    from sft_label.pipeline import merge_stats
    from sft_label.scoring import merge_value_stats

    written = {}
    all_pass1_stats = []
    all_pass2_stats = []

    # Pass 1: labeled files
    if pass_num in ("1", "both"):
        pass1_files = discover_pass1_source_files(input_dir)
        if pass1_files:
            print(f"Found {len(pass1_files)} Pass 1 source file(s)")
            p1_workers = _resolve_worker_count(workers, len(pass1_files))
            p1_interval = _recompute_progress_interval(len(pass1_files))
            p1_compact = p1_interval > 1 and p1_workers == 1
            if p1_workers > 1 and len(pass1_files) > 1:
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
                _write_pass1_conversation_stats(samples, stats, file_out_dir)
                return {
                    "idx": idx,
                    "rel_label": rel_label,
                    "stats_path": stats_path,
                    "stats": stats,
                }

            ordered_results = []
            if p1_workers > 1 and len(pass1_files) > 1:
                done = 0
                results_by_idx = {}
                with ThreadPoolExecutor(max_workers=p1_workers) as pool:
                    futures = {
                        pool.submit(_process_labeled_file, idx, lf): idx
                        for idx, lf in enumerate(pass1_files, start=1)
                    }
                    for fut in as_completed(futures):
                        item = fut.result()
                        results_by_idx[item["idx"]] = item
                        done += 1
                        if done == 1 or done == len(pass1_files) or done % p1_interval == 0:
                            print(
                                f"  Pass 1 progress {done}/{len(pass1_files)} | "
                                f"{item['rel_label']} | "
                                f"{item['stats']['success']}/{item['stats']['total_samples']} success"
                            )
                ordered_results = [results_by_idx[i] for i in range(1, len(pass1_files) + 1)]
            else:
                for idx, lf in enumerate(pass1_files, start=1):
                    rel_label = _relative_file_label(lf, input_dir)
                    if not p1_compact:
                        print(f"\n  Processing: {rel_label}")
                    item = _process_labeled_file(idx, lf)
                    if not p1_compact:
                        print(
                            f"    → {item['stats_path']} "
                            f"({item['stats']['success']}/{item['stats']['total_samples']} success)"
                        )
                    elif idx == 1 or idx == len(pass1_files) or idx % p1_interval == 0:
                        print(
                            f"  Pass 1 progress {idx}/{len(pass1_files)} | "
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
                extension_stats = _merge_extension_stats(all_pass1_stats)
                if extension_stats:
                    summary["extension_stats"] = extension_stats
                summary_path = out_dir / PASS1_SUMMARY_STATS_FILE
                _write_json(summary_path, summary)
                written["summary_stats"] = str(summary_path)
                print(f"\n  Summary: {summary_path} "
                      f"({summary.get('success', 0)} total success)")

    # Pass 2: scored files
    if pass_num in ("2", "both"):
        scored_files = discover_scored_files(input_dir)
        if scored_files:
            existing_summary_postprocess = _extract_postprocess_payload(
                _load_json_payload(out_dir / PASS2_SUMMARY_STATS_FILE)
            )
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
                _preserve_existing_pass2_postprocess(stats_path, stats)
                _write_json(stats_path, stats)
                conversation_records = aggregate_conversations(samples)
                _write_conversation_records(
                    conversation_records,
                    file_out_dir,
                    clear_when_empty=True,
                )
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
                summary_postprocess = (
                    existing_summary_postprocess
                    or _merge_summary_postprocess_from_file_stats(all_pass2_stats)
                )
                if summary_postprocess:
                    summary["postprocess"] = summary_postprocess
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

    def _artifact_cache_has_scored_values(artifact_dir: Path) -> bool:
        for name in ("scored.jsonl", "scored.json"):
            cache_path = artifact_dir / name
            if not cache_path.exists():
                continue
            try:
                if cache_path.suffix == ".jsonl":
                    with open(cache_path, encoding="utf-8") as f:
                        for line in f:
                            line = line.strip()
                            if not line:
                                continue
                            row = json.loads(line)
                            if isinstance((row or {}).get("value"), dict):
                                return True
                else:
                    with open(cache_path, encoding="utf-8") as f:
                        data = json.load(f)
                    if isinstance(data, list):
                        for row in data:
                            if isinstance(row, dict):
                                if isinstance((row or {}).get("value"), dict):
                                    return True
            except (OSError, json.JSONDecodeError):
                continue
        return False

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
                source_has_scores = inline_source_has_embedded_scores(source_file)
                cache_has_scores = _artifact_cache_has_scored_values(artifact_dir)
                if source_has_scores and not cache_has_scores:
                    write_inline_scored_cache(source_file, artifact_dir)
                    cache_has_scores = _artifact_cache_has_scored_values(artifact_dir)
                if not source_has_scores:
                    if cache_has_scores:
                        write_inline_scored_cache(source_file, artifact_dir)
                        cache_has_scores = _artifact_cache_has_scored_values(artifact_dir)
                        _log_regenerate_dashboard(
                            f"{rel_label}: invalidated stale scored cache "
                            "because source no longer embeds Pass 2 values"
                        )
                    _write_conversation_records([], artifact_dir, clear_when_empty=True)
                    _refresh_inline_pass2_stats_from_cache(pass2_stats_path)
                    _log_regenerate_dashboard(
                        f"{rel_label}: refreshed {pass2_stats_path.name} to empty Pass 2 state"
                    )
                if not cache_has_scores:
                    _log_regenerate_dashboard(
                        f"{rel_label}: no embedded Pass 2 values found in source; "
                        "using empty scored cache state"
                    )
                _ensure_conversation_scores_current(artifact_dir)
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


def run_complete_postprocess(input_path, scope="global", open_browser=False, workers=1):
    """Materialize deferred Pass 2 postprocess artifacts offline.

    scope="global" (default) is the large-run-safe mode:
      - recompute per-file conversation_scores.json via streaming
      - merge global conversation_scores.json
      - generate the global Pass 2 dashboard from summary stats only

    scope="all" additionally generates standalone per-file Pass 2 dashboards,
    which may be substantially more memory-intensive on very large files.
    """
    inline_target = infer_inline_scoring_target(input_path)
    if inline_target is not None:
        return _run_complete_postprocess_inline(
            inline_target,
            scope=scope,
            open_browser=open_browser,
            workers=workers,
        )
    return _run_complete_postprocess_legacy(
        input_path,
        scope=scope,
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
        # Per-subdir maintenance only; batch runs retain run-level canonical dashboards.
        if worker_count > 1 and len(subdirs) > 1:
            done = 0
            with ThreadPoolExecutor(max_workers=worker_count) as pool:
                futures = {
                    pool.submit(
                        _prepare_dir_for_global_dashboard,
                        subdir,
                        pass_num,
                    ): index
                    for index, subdir in enumerate(subdirs, start=1)
                }
                for fut in as_completed(futures):
                    index = futures[fut]
                    subdir = subdirs[index - 1]
                    _log_regenerate_dashboard(
                        f"[dir {index}/{len(subdirs)}] Refreshing dashboard inputs for {subdir}"
                    )
                    fut.result()
                    done += 1
                    _log_regenerate_dashboard(
                        f"Parallel progress: {done}/{len(subdirs)} directory(ies) completed"
                    )
        else:
            for index, subdir in enumerate(subdirs, start=1):
                _log_regenerate_dashboard(
                    f"[dir {index}/{len(subdirs)}] Refreshing dashboard inputs for {subdir}"
                )
                _prepare_dir_for_global_dashboard(subdir, pass_num)

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
    patterns = (
        "**/labeled.json",
        "**/labeled.jsonl",
        "**/scored.json",
        "**/scored.jsonl",
        f"**/{PASS1_STATS_FILE}",
        f"**/{PASS2_STATS_FILE}",
    )
    dirs = {
        path.parent
        for pattern in patterns
        for path in run_dir.glob(pattern)
        if path.parent != run_dir
    }
    return sorted(dirs)


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
                _ensure_conversation_scores_current(dir_path)
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


def _prepare_dir_for_global_dashboard(dir_path: Path, pass_num: str) -> None:
    """Refresh per-file inputs needed by the canonical run-level dashboards."""
    if pass_num in ("2", "both"):
        _ensure_conversation_scores_current(dir_path)


def _global_pass2_dashboard_output_name(run_dir: Path, stats: dict | None) -> str:
    total_scored = int((stats or {}).get("total_scored") or 0)
    threshold = max(int(PASS2_HEAVY_POSTPROCESS_SAMPLE_THRESHOLD), 1)
    postprocess = (stats or {}).get("postprocess") or {}
    dashboard = postprocess.get("dashboard") if isinstance(postprocess, dict) else {}
    status = str((dashboard or {}).get("status") or "").strip().lower() if isinstance(dashboard, dict) else ""
    reason = str((dashboard or {}).get("reason") or "").strip() if isinstance(dashboard, dict) else ""
    if total_scored >= threshold or (status == "deferred" and reason.startswith("samples=")):
        return pass2_global_dashboard_filename(run_dir.name)
    return PASS2_DASHBOARD_FILE


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
                             output_file=PASS1_DASHBOARD_FILE)
                generated.append(out)
                prune_dashboard_bundles(
                    run_dir,
                    keep_paths=[out],
                    kind="labeling",
                    recursive=True,
                )
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
                summary_stats = _load_json_payload(pass2_summary_path)
                out_path = gen_p2(run_dir, scored_file=None,
                                  stats_file=pass2_summary_path.name,
                                  output_file=_global_pass2_dashboard_output_name(run_dir, summary_stats),
                                  quiet=True)
                if out_path:
                    out_path = Path(out_path) if not isinstance(out_path, Path) else out_path
                    generated.append(out_path)
                    prune_dashboard_bundles(
                        run_dir,
                        keep_paths=[out_path],
                        kind="scoring",
                        recursive=True,
                    )
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


def _load_json_payload(path):
    path = Path(path)
    if not path.exists():
        return {}
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _conversation_records_need_backfill(records) -> bool:
    if not isinstance(records, list) or not records:
        return False
    for record in records:
        if not isinstance(record, dict):
            continue
        detail = record.get("detail") or {}
        if record.get("conv_value") is not None and (
            detail.get("quality_overall") is None
            or detail.get("reasoning_overall") is None
        ):
            return True
    return False


def _pick_scored_path(dir_path: Path) -> Path | None:
    for name in ("scored.jsonl", "scored.json"):
        path = dir_path / name
        if path.exists():
            return path
    return None


def _ensure_conversation_scores_current(dir_path: Path) -> bool:
    conv_path = Path(dir_path) / "conversation_scores.json"
    records = _load_json_payload(conv_path) if conv_path.exists() else []
    if not _conversation_records_need_backfill(records):
        return False

    scored_path = _pick_scored_path(Path(dir_path))
    if scored_path is None:
        return False

    refreshed_records = _stream_conversation_records(scored_path)
    written = _write_conversation_records(refreshed_records, Path(dir_path))
    if not written:
        return False
    _log_regenerate_dashboard(
        f"{dir_path}: refreshed conversation_scores.json from {scored_path.name} "
        "to backfill quality/reasoning metrics"
    )
    return True


def _completed_postprocess_payload(*, artifact=None, count=None):
    payload = {
        "status": "completed",
        "completed_at": datetime.now().isoformat(),
        "mode": "complete-postprocess",
    }
    if artifact is not None:
        payload["artifact"] = str(artifact)
    if count is not None:
        payload["count"] = int(count)
    return payload


def _pending_dashboard_postprocess_payload(*, reason="pending-dashboard-generation"):
    return {
        "status": "pending",
        "reason": str(reason),
        "completed_at": datetime.now().isoformat(),
        "mode": "complete-postprocess",
    }


def _failed_postprocess_payload(reason):
    return {
        "status": "failed",
        "reason": str(reason),
        "completed_at": datetime.now().isoformat(),
        "mode": "complete-postprocess",
    }


def _update_pass2_postprocess_status(stats_path, *, conversation=None, dashboard=None):
    stats_path = Path(stats_path)
    stats = _load_json_payload(stats_path) if stats_path.exists() else {}
    postprocess = dict(stats.get("postprocess") or {})
    if conversation is not None:
        postprocess["conversation_scores"] = conversation
    if dashboard is not None:
        postprocess["dashboard"] = dashboard
    if postprocess:
        stats["postprocess"] = postprocess
        _write_json(stats_path, stats)
    return stats


def _mark_dashboard_pending(stats_path):
    stats_path = Path(stats_path)
    if not stats_path.exists():
        return
    _update_pass2_postprocess_status(
        stats_path,
        dashboard=_pending_dashboard_postprocess_payload(),
    )


def _dashboard_generation_stats_hint(stats_path, *, artifact=None):
    stats_path = Path(stats_path)
    stats_hint = _load_json_payload(stats_path) if stats_path.exists() else {}
    if not isinstance(stats_hint, dict):
        stats_hint = {}
    postprocess = dict(stats_hint.get("postprocess") or {})
    postprocess["dashboard"] = _completed_postprocess_payload(artifact=artifact)
    stats_hint["postprocess"] = postprocess
    return stats_hint


def _stream_conversation_records(scored_path):
    from sft_label.conversation import aggregate_conversations
    from sft_label.tools.dashboard_aggregation import iter_data_file

    return aggregate_conversations(iter_data_file(scored_path))


def _per_file_pass2_stats_path(scored_path: Path) -> Path:
    return scored_path.parent / PASS2_STATS_FILE


def _complete_postprocess_conversations_for_file(run_dir: Path, scored_path: Path):
    rel_label = _relative_file_label(scored_path, run_dir)
    conv_records = _stream_conversation_records(scored_path)
    written = _write_conversation_records(
        conv_records,
        scored_path.parent,
        clear_when_empty=True,
    )
    conv_count = len(conv_records)
    stats_path = _per_file_pass2_stats_path(scored_path)
    if stats_path.exists():
        _update_pass2_postprocess_status(
            stats_path,
            conversation=_completed_postprocess_payload(
                artifact=written.get("conversation_scores"),
                count=conv_count,
            ),
        )
    return {
        "scored_path": scored_path,
        "rel_label": rel_label,
        "conversation_count": conv_count,
        "conversation_path": written.get("conversation_scores"),
        "stats_path": stats_path if stats_path.exists() else None,
    }


def _complete_postprocess_dashboard_for_file(scored_path: Path):
    from sft_label.tools.visualize_value import generate_value_dashboard

    stats_path = _per_file_pass2_stats_path(scored_path)
    if not stats_path.exists():
        raise FileNotFoundError(f"Missing {PASS2_STATS_FILE} for {scored_path.parent}")

    _mark_dashboard_pending(stats_path)
    output_path = scored_path.parent / PASS2_DASHBOARD_FILE
    out_path = generate_value_dashboard(
        scored_path.parent,
        scored_file=scored_path.name,
        stats_file=stats_path.name,
        output_file=PASS2_DASHBOARD_FILE,
        quiet=True,
        stats_hint_override=_dashboard_generation_stats_hint(
            stats_path,
            artifact=str(output_path),
        ),
    )
    _update_pass2_postprocess_status(
        stats_path,
        dashboard=_completed_postprocess_payload(artifact=out_path),
    )
    return Path(out_path) if not isinstance(out_path, Path) else out_path


def _complete_postprocess_inline_dashboard_for_file(target, source_file: Path):
    from sft_label.tools.visualize_value import generate_value_dashboard

    artifact_dir = target.layout.file_artifact_dir(source_file)
    stats_path = artifact_dir / PASS2_STATS_FILE
    if not stats_path.exists():
        raise FileNotFoundError(f"Missing {PASS2_STATS_FILE} for {artifact_dir}")

    _mark_dashboard_pending(stats_path)
    output_path = target.layout.dashboard_path(
        _inline_file_dashboard_name("dashboard_scoring", target, source_file)
    )
    out_path = generate_value_dashboard(
        artifact_dir,
        scored_file="scored.json",
        stats_file=stats_path.name,
        output_file=str(output_path),
        quiet=True,
        stats_hint_override=_dashboard_generation_stats_hint(
            stats_path,
            artifact=str(output_path),
        ),
    )
    _update_pass2_postprocess_status(
        stats_path,
        dashboard=_completed_postprocess_payload(artifact=out_path),
    )
    return Path(out_path) if not isinstance(out_path, Path) else out_path


def _run_complete_postprocess_inline(target, scope="global", open_browser=False, workers=1):
    import webbrowser
    from sft_label.conversation import merge_conversation_record_batches, write_conversation_scores
    from sft_label.scoring import _sync_inline_scored_cache_to_dataset
    from sft_label.tools.visualize_value import generate_value_dashboard

    scope = str(scope or "global").strip().lower()
    if scope not in {"global", "all"}:
        raise ValueError(f"scope must be 'global' or 'all', got {scope!r}")

    selected = _materialize_inline_recompute_artifacts(target, pass_num="2")
    if target.target_path.is_file():
        selected = [target.target_path.resolve()]

    _log_complete_postprocess(
        f"Starting inline Pass 2 postprocess: input={target.target_path} scope={scope} files={len(selected)}"
    )

    summary_path = target.layout.meta_root / PASS2_SUMMARY_STATS_FILE
    _mark_dashboard_pending(summary_path)

    recompute_input = (
        target.layout.file_artifact_dir(target.target_path)
        if target.target_path.is_file()
        else target.layout.meta_root
    )
    written = _run_recompute_legacy(
        recompute_input,
        pass_num="2",
        output_dir=recompute_input,
        workers=workers,
    )

    generated_dashboards: list[Path] = []

    if target.target_path.is_file():
        artifact_dir = recompute_input
        _scored_samples, conv_records = _sync_inline_scored_cache_to_dataset(
            target.target_path.resolve(),
            artifact_dir,
            strict=False,
        )
        conv_path = artifact_dir / "conversation_scores.json"
        conv_count = len(conv_records)
        stats_path = artifact_dir / PASS2_STATS_FILE
        if stats_path.exists():
            _update_pass2_postprocess_status(
                stats_path,
                conversation=_completed_postprocess_payload(
                    artifact=str(conv_path) if conv_path.exists() else None,
                    count=conv_count,
                ),
            )
        dashboard_path = _complete_postprocess_inline_dashboard_for_file(target, target.target_path.resolve())
        generated_dashboards.append(dashboard_path)
        if open_browser:
            webbrowser.open(dashboard_path.resolve().as_uri())
        _log_complete_postprocess(
            f"Completed inline Pass 2 postprocess: dashboards={len(generated_dashboards)} "
            f"conversations={conv_count}"
        )
        return {
            "command": "complete-postprocess",
            "run_dir": str(target.layout.run_root),
            "scope": scope,
            "files_processed": 1,
            "conversation_scores": str(conv_path) if conv_path.exists() else None,
            "generated_dashboards": [str(path) for path in generated_dashboards],
        }

    if "summary_stats_value" in written:
        _rewrite_inline_per_file_summary(written["summary_stats_value"])

    corrected_conv_batches = []
    for source_file in selected:
        artifact_dir = target.layout.file_artifact_dir(source_file)
        stats_path = artifact_dir / PASS2_STATS_FILE
        scored_samples, conv_records = _sync_inline_scored_cache_to_dataset(
            source_file,
            artifact_dir,
            strict=False,
        )
        conv_path = artifact_dir / "conversation_scores.json"
        conv_count = len(conv_records)
        if stats_path.exists():
            stats = _load_json_payload(stats_path)
            stats["input_file"] = str(source_file)
            stats["mirrored_file"] = str(source_file)
            stats["file"] = target.layout.relative_source_path(source_file).as_posix()
            stats["total_scored"] = len([sample for sample in scored_samples if sample.get("value")])
            if conv_records:
                stats["conversation_records"] = conv_count
            postprocess = dict(stats.get("postprocess") or {})
            postprocess["conversation_scores"] = _completed_postprocess_payload(
                artifact=str(conv_path) if conv_path.exists() else None,
                count=conv_count,
            )
            if scope == "global":
                postprocess["dashboard"] = {
                    "status": "deferred",
                    "reason": "scope=global",
                    "mode": "complete-postprocess",
                    "completed_at": datetime.now().isoformat(),
                }
            stats["postprocess"] = postprocess
            _write_json(stats_path, stats)
        if conv_records:
            corrected_conv_batches.append(conv_records)
        if scope == "all":
            try:
                dashboard_path = _complete_postprocess_inline_dashboard_for_file(target, source_file)
                generated_dashboards.append(dashboard_path)
                _log_complete_postprocess(
                    f"[dash {len(generated_dashboards)}/{len(selected)}] "
                    f"{target.layout.relative_source_path(source_file)} -> {dashboard_path}"
                )
            except Exception as e:
                if stats_path.exists():
                    _update_pass2_postprocess_status(
                        stats_path,
                        dashboard=_failed_postprocess_payload(e),
                    )
                _log_complete_postprocess(
                    f"Warning: inline per-file Pass 2 dashboard failed for "
                    f"{target.layout.relative_source_path(source_file)}: {type(e).__name__}: {e}"
                )

    global_conv_path = target.layout.meta_root / "conversation_scores.json"
    merged_conv_records = merge_conversation_record_batches(corrected_conv_batches)
    if merged_conv_records:
        write_conversation_scores(merged_conv_records, global_conv_path)
    elif global_conv_path.exists():
        global_conv_path.unlink()
    global_conv_count = len(merged_conv_records)
    if summary_path.exists():
        summary = _load_json_payload(summary_path)
        summary["run_dir"] = str(target.layout.run_root)
        postprocess = dict(summary.get("postprocess") or {})
        postprocess["conversation_scores"] = _completed_postprocess_payload(
            artifact=str(global_conv_path) if global_conv_path.exists() else None,
            count=global_conv_count,
        )
        postprocess["dashboard"] = _pending_dashboard_postprocess_payload()
        summary["postprocess"] = postprocess
        _write_json(summary_path, summary)
        try:
            output_path = target.layout.dashboard_path(
                pass2_global_dashboard_filename(target.layout.dataset_root_name)
            )
            dashboard_stats_hint = dict(summary)
            dashboard_postprocess = dict(dashboard_stats_hint.get("postprocess") or {})
            dashboard_postprocess["dashboard"] = _completed_postprocess_payload(artifact=str(output_path))
            dashboard_stats_hint["postprocess"] = dashboard_postprocess
            out_path = generate_value_dashboard(
                target.layout.meta_root,
                scored_file=None,
                stats_file=summary_path.name,
                output_file=str(output_path),
                quiet=True,
                stats_hint_override=dashboard_stats_hint,
            )
            out_path = Path(out_path) if not isinstance(out_path, Path) else out_path
            generated_dashboards.append(out_path)
            postprocess["dashboard"] = _completed_postprocess_payload(artifact=out_path)
            summary["postprocess"] = postprocess
            _write_json(summary_path, summary)
            _log_complete_postprocess(f"Global inline Pass 2 dashboard -> {out_path}")
        except Exception as e:
            postprocess["dashboard"] = _failed_postprocess_payload(e)
            summary["postprocess"] = postprocess
            _write_json(summary_path, summary)
            _log_complete_postprocess(
                f"Warning: global inline Pass 2 dashboard failed: {type(e).__name__}: {e}"
            )

    if open_browser and generated_dashboards:
        latest = Path(generated_dashboards[-1]).resolve()
        _log_complete_postprocess(f"Opening dashboard in browser: {latest}")
        webbrowser.open(latest.as_uri())

    _log_complete_postprocess(
        f"Completed inline Pass 2 postprocess: dashboards={len(generated_dashboards)} "
        f"conversations={global_conv_count}"
    )
    return {
        "command": "complete-postprocess",
        "run_dir": str(target.layout.run_root),
        "scope": scope,
        "files_processed": len(selected),
        "conversation_scores": str(global_conv_path) if global_conv_path.exists() else None,
        "generated_dashboards": [str(path) for path in generated_dashboards],
    }


def _run_complete_postprocess_legacy(input_path, scope="global", open_browser=False, workers=1):
    import webbrowser
    from sft_label.conversation import finalize_conversation_records
    from sft_label.tools.dashboard_aggregation import iter_data_file
    from sft_label.tools.visualize_value import generate_value_dashboard

    scope = str(scope or "global").strip().lower()
    if scope not in {"global", "all"}:
        raise ValueError(f"scope must be 'global' or 'all', got {scope!r}")

    run_dir = Path(input_path)
    if not run_dir.is_dir():
        raise ValueError(f"Expected a directory, got: {input_path}")

    scored_files = discover_scored_files(run_dir)
    if not scored_files:
        raise ValueError(f"No scored files found under {run_dir}")

    _log_complete_postprocess(
        f"Starting deferred Pass 2 postprocess: input={run_dir} scope={scope} files={len(scored_files)}"
    )

    _mark_dashboard_pending(run_dir / PASS2_SUMMARY_STATS_FILE)

    generated_dashboards = []
    conversation_results = []
    conversation_paths = []
    merged_records = []
    conv_workers = _resolve_worker_count(workers, len(scored_files))

    if conv_workers > 1 and len(scored_files) > 1:
        with ThreadPoolExecutor(max_workers=conv_workers) as pool:
            futures = {
                pool.submit(_complete_postprocess_conversations_for_file, run_dir, scored_path): scored_path
                for scored_path in scored_files
            }
            for index, fut in enumerate(as_completed(futures), start=1):
                item = fut.result()
                conv_count = int(item.get("conversation_count") or 0)
                conv_path = item.get("conversation_path")
                if conv_path:
                    conversation_paths.append(Path(conv_path))
                conversation_results.append(item)
                _log_complete_postprocess(
                    f"[conv {index}/{len(scored_files)}] {item['rel_label']} -> "
                    f"{conv_count} conversation(s)"
                )
    else:
        for index, scored_path in enumerate(scored_files, start=1):
            item = _complete_postprocess_conversations_for_file(run_dir, scored_path)
            conv_count = int(item.get("conversation_count") or 0)
            conv_path = item.get("conversation_path")
            if conv_path:
                conversation_paths.append(Path(conv_path))
            conversation_results.append(item)
            _log_complete_postprocess(
                f"[conv {index}/{len(scored_files)}] {item['rel_label']} -> "
                f"{conv_count} conversation(s)"
            )

    conversation_results.sort(key=lambda item: item["rel_label"])
    for conv_path in sorted(conversation_paths):
        if not conv_path.exists():
            continue
        for record in iter_data_file(conv_path):
            if isinstance(record, dict):
                merged_records.append(record)
    merged_records = finalize_conversation_records(merged_records)
    global_conv_written = _write_conversation_records(
        merged_records,
        run_dir,
        clear_when_empty=True,
    )
    _log_complete_postprocess(
        f"Global conversation aggregation -> {len(merged_records)} conversation(s)"
    )

    summary_path = run_dir / PASS2_SUMMARY_STATS_FILE
    if summary_path.exists():
        summary = _load_json_payload(summary_path)
        postprocess = dict(summary.get("postprocess") or {})
        postprocess["conversation_scores"] = _completed_postprocess_payload(
            artifact=global_conv_written.get("conversation_scores"),
            count=len(merged_records),
        )
        postprocess["dashboard"] = _pending_dashboard_postprocess_payload()
        summary["postprocess"] = postprocess
        _write_json(summary_path, summary)
        try:
            output_name = _global_pass2_dashboard_output_name(run_dir, summary)
            output_path = run_dir / "dashboards" / output_name
            out_path = generate_value_dashboard(
                run_dir,
                scored_file=None,
                stats_file=summary_path.name,
                output_file=output_name,
                quiet=True,
                stats_hint_override=_dashboard_generation_stats_hint(
                    summary_path,
                    artifact=str(output_path),
                ),
            )
            out_path = Path(out_path) if not isinstance(out_path, Path) else out_path
            generated_dashboards.append(out_path)
            prune_dashboard_bundles(
                run_dir,
                keep_paths=[out_path],
                kind="scoring",
                recursive=(scope == "global"),
            )
            postprocess["dashboard"] = _completed_postprocess_payload(artifact=out_path)
            _log_complete_postprocess(f"Global Pass 2 dashboard -> {out_path}")
        except Exception as e:
            postprocess["dashboard"] = _failed_postprocess_payload(e)
            _log_complete_postprocess(
                f"Warning: global Pass 2 dashboard failed: {type(e).__name__}: {e}"
            )
        summary["postprocess"] = postprocess
        _write_json(summary_path, summary)

    if scope == "all":
        _log_complete_postprocess("Generating standalone per-file Pass 2 dashboards sequentially")
        for index, item in enumerate(conversation_results, start=1):
            scored_path = item["scored_path"]
            rel_label = item["rel_label"]
            try:
                out_path = _complete_postprocess_dashboard_for_file(scored_path)
                generated_dashboards.append(out_path)
                _log_complete_postprocess(
                    f"[dash {index}/{len(conversation_results)}] {rel_label} -> {out_path}"
                )
            except Exception as e:
                stats_path = item.get("stats_path")
                if stats_path is not None:
                    _update_pass2_postprocess_status(
                        stats_path,
                        dashboard=_failed_postprocess_payload(e),
                    )
                _log_complete_postprocess(
                    f"Warning: per-file Pass 2 dashboard failed for {rel_label}: "
                    f"{type(e).__name__}: {e}"
                )

    if open_browser and generated_dashboards:
        latest = Path(generated_dashboards[-1]).resolve()
        _log_complete_postprocess(f"Opening dashboard in browser: {latest}")
        webbrowser.open(latest.as_uri())

    _log_complete_postprocess(
        f"Completed deferred Pass 2 postprocess: dashboards={len(generated_dashboards)} "
        f"conversations={len(merged_records)}"
    )
    return {
        "command": "complete-postprocess",
        "run_dir": str(run_dir),
        "scope": scope,
        "files_processed": len(scored_files),
        "conversation_scores": global_conv_written.get("conversation_scores"),
        "generated_dashboards": [str(path) for path in generated_dashboards],
    }


# ─── Helpers ─────────────────────────────────────────────

def _write_json(path, data):
    """Write JSON file, stripping internal keys for dict payloads."""
    path = Path(path)
    if isinstance(data, dict):
        clean = {k: v for k, v in data.items() if not k.startswith("_")}
    else:
        clean = data
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=str(path.parent),
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(clean, f, ensure_ascii=False, indent=2)
        os.replace(tmp_path, path)
    finally:
        try:
            os.unlink(tmp_path)
        except FileNotFoundError:
            pass
