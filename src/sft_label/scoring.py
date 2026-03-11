"""
Value Scoring Pipeline (Pass 2)

Evaluates SFT training data value through:
  1. Rarity computation — tag IDF + combo rarity from existing tag distributions
  2. LLM-based scoring — complexity, quality, reasoning assessment
  3. Weighted aggregation — composite value_score

Runs after Pass 1 (tag labeling) and produces scored.json, stats_scoring.json,
dashboard_scoring.html per file.
"""

from __future__ import annotations

import json
import math
import os
import time
import asyncio
import random
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass, field

import httpx
from rich.progress import (
    Progress, SpinnerColumn, BarColumn, TextColumn,
    TimeElapsedColumn, TimeRemainingColumn, MofNCompleteColumn,
)

from sft_label.config import (
    LITELLM_BASE, LITELLM_KEY,
    MAX_RETRIES, SAMPLE_MAX_RETRIES, REQUEST_TIMEOUT,
    VALUE_WEIGHTS, RARITY_WEIGHTS, RARITY_COMBO_ALPHA, RARITY_SCORE_MODE,
    VALUE_TRUNCATION_BUDGET,
    KNOWN_FLAGS, KNOWN_FLAGS_POSITIVE, KNOWN_FLAGS_NEGATIVE,
    CHUNK_SIZE, MAX_ACTIVE_CHUNKS,
    SELECTION_INTRA_WEIGHT, SELECTION_QUALITY_WEIGHT,
    SELECTION_MIN_GROUP_SIZE, SELECTION_SMOOTHING_PRIOR,
    PipelineConfig,
)
from sft_label.preprocessing import (
    detect_thinking_mode, extract_cot_content, truncate_for_scoring,
    count_code_blocks,
)
from sft_label.pipeline import (
    async_llm_call,
    AsyncRateLimiter,
    RuntimeEtaEstimator,
    format_progress_info,
    parse_run_progress,
)
from sft_label.artifacts import (
    PASS1_STATS_FILE,
    PASS1_SUMMARY_STATS_FILE,
    PASS2_STATS_FILE,
    PASS2_SUMMARY_STATS_FILE,
    PASS2_DASHBOARD_FILE,
    pass2_global_dashboard_filename,
)
from sft_label.labels import is_partial_labels, is_usable_labels


# ─────────────────────────────────────────────────────────
# Rarity computation
# ─────────────────────────────────────────────────────────

def _relative_file_label(path, root):
    """Render a stable file label relative to the batch input root."""
    path = Path(path)
    root = Path(root)
    try:
        return str(path.relative_to(root))
    except ValueError:
        return path.name


def _coerce_positive_int(value):
    """Parse a positive integer from JSON-ish input; return 0 when invalid."""
    if isinstance(value, bool):
        return 0
    if isinstance(value, int):
        return value if value > 0 else 0
    if isinstance(value, float):
        iv = int(value)
        return iv if iv > 0 else 0
    return 0


def _normalize_combo_counts(raw_counts):
    """Normalize combo counts map loaded from stats.json.

    Returns None when missing/invalid.
    """
    if not isinstance(raw_counts, dict):
        return None
    cleaned = {}
    for key, value in raw_counts.items():
        if not isinstance(key, str) or not key:
            continue
        count = _coerce_positive_int(value)
        if count > 0:
            cleaned[key] = count
    return cleaned or None


def load_tag_stats_context(stats_path):
    """Load rarity baseline context from a Pass 1 stats file.

    Returns (distributions, distribution_total_samples, timestamp, meta) where
    meta may include:
      - stats_total_samples
      - distribution_total_samples
      - combo_counts
    Returns (None, 0, None, {}) on failure.
    """
    path = Path(stats_path)
    if not path.exists():
        return None, 0, None, {}

    with open(path, "r", encoding="utf-8") as f:
        stats = json.load(f)

    distributions = stats.get("tag_distributions")
    if not distributions:
        return None, 0, None, {}

    stats_total_samples = _coerce_positive_int(stats.get("total_samples", 0))
    distribution_total_samples = _coerce_positive_int(
        stats.get("distribution_total_samples", stats_total_samples)
    )
    if distribution_total_samples <= 0:
        distribution_total_samples = stats_total_samples

    combo_counts = _normalize_combo_counts(stats.get("combo_distributions"))
    timestamp = stats.get("timestamp", None)

    return distributions, distribution_total_samples, timestamp, {
        "stats_total_samples": stats_total_samples,
        "distribution_total_samples": distribution_total_samples,
        "combo_counts": combo_counts,
    }


def load_tag_stats(stats_path):
    """Load tag_distributions from a Pass 1 stats file.

    Uses `distribution_total_samples` when present; falls back to
    `total_samples` for backward compatibility.

    Returns (distributions, total_samples, timestamp) or (None, 0, None) on failure.
    """
    distributions, total_samples, timestamp, _meta = load_tag_stats_context(stats_path)
    return distributions, total_samples, timestamp


_RARITY_DEFAULT_CONFIDENCE = 0.75
_RARITY_INHERITED_CONFIDENCE_PENALTY = 0.65
_RARITY_MULTI_TAG_MAX_BLEND = 0.60
_SELECTION_CONFIDENCE_FLOOR = 0.25


def _coerce_unit_float(value, default):
    """Clamp a possibly-invalid float-like value into [0, 1]."""
    if isinstance(value, bool):
        return default
    if isinstance(value, (int, float)):
        return max(0.0, min(1.0, float(value)))
    return default


def _dimension_label_confidence(labels, dim, default=_RARITY_DEFAULT_CONFIDENCE):
    """Return the effective confidence for one label dimension."""
    if not isinstance(labels, dict):
        return default
    confidence = default
    confidence_map = labels.get("confidence")
    if isinstance(confidence_map, dict):
        confidence = _coerce_unit_float(confidence_map.get(dim), default)
    if labels.get("inherited"):
        confidence *= _RARITY_INHERITED_CONFIDENCE_PENALTY
    return max(0.0, min(1.0, confidence))


def _sample_label_confidence(labels, dims=None, default=_RARITY_DEFAULT_CONFIDENCE):
    """Return the mean confidence across populated label dimensions."""
    if not isinstance(labels, dict):
        return default
    if dims is None:
        dims = [dim for dim in RARITY_WEIGHTS.keys() if labels.get(dim) is not None]
    confidences = []
    for dim in dims:
        tags = labels.get(dim)
        if tags is None:
            continue
        if isinstance(tags, list) and not tags:
            continue
        if isinstance(tags, str) and not tags:
            continue
        confidences.append(_dimension_label_confidence(labels, dim, default=default))
    if not confidences:
        return default * (_RARITY_INHERITED_CONFIDENCE_PENALTY if labels.get("inherited") else 1.0)
    return sum(confidences) / len(confidences)


def _dimension_rarity_prior(dim_idfs):
    """Use the dimension mean as a conservative prior for missing/uncertain tags."""
    if not dim_idfs:
        return 0.0
    values = [v for v in dim_idfs.values() if isinstance(v, (int, float))]
    if not values:
        return 0.0
    return sum(values) / len(values)


def _aggregate_multi_tag_rarity(values):
    """Blend max and mean to reduce single-tag max-IDF amplification."""
    if not values:
        return 0.0
    max_value = max(values)
    mean_value = sum(values) / len(values)
    return (_RARITY_MULTI_TAG_MAX_BLEND * max_value
            + (1.0 - _RARITY_MULTI_TAG_MAX_BLEND) * mean_value)


def compute_tag_idf(distributions, total_samples):
    """Compute IDF for every tag across all dimensions.

    idf(tag) = log2(N / (count + 1))

    Returns {dim: {tag: idf_value}}.
    """
    if not distributions or total_samples <= 0:
        return {}

    idf_map = {}
    for dim, tag_counts in distributions.items():
        idf_map[dim] = {}
        for tag, count in tag_counts.items():
            # Multi-select dimensions can have count >= total_samples; clamp at 0
            # because "negative rarity" is not meaningful for downstream scoring.
            idf_map[dim][tag] = max(0.0, math.log2(total_samples / (count + 1)))

    return idf_map


def compute_sample_rarity(sample_labels, idf_map, total_samples,
                          rarity_weights=None, combo_alpha=None,
                          combo_counts=None, stats_ref_info=None):
    """Compute rarity score for a single sample.

    Returns dict with score, tag_rarity, combo_rarity, stats_ref.
    Score is raw (not normalized to 1-10 yet).
    """
    if rarity_weights is None:
        rarity_weights = RARITY_WEIGHTS
    if combo_alpha is None:
        combo_alpha = RARITY_COMBO_ALPHA
    if not isinstance(sample_labels, dict):
        sample_labels = {}

    if not idf_map:
        return {
            "score": None,
            "tag_rarity": None,
            "combo_rarity": None,
            "stats_ref": stats_ref_info,
        }

    # Weighted tag IDF
    weighted_sum = 0.0
    weight_total = 0.0

    for dim, weight in rarity_weights.items():
        tags = sample_labels.get(dim)
        if tags is None:
            continue

        # Missing dimension in baseline: skip instead of treating everything as
        # maximally rare. This protects external baseline scenarios with sparse
        # or incomplete dimensions.
        dim_idfs = idf_map.get(dim)
        if not dim_idfs:
            continue

        dim_prior = _dimension_rarity_prior(dim_idfs)
        dim_confidence = _dimension_label_confidence(sample_labels, dim)

        if isinstance(tags, list):
            if not tags:
                continue
            tag_idf_values = [
                dim_idfs.get(t, dim_prior)
                for t in tags
                if isinstance(t, str) and t
            ]
            if not tag_idf_values:
                continue
            observed_rarity = _aggregate_multi_tag_rarity(tag_idf_values)
        else:
            # Single-select dimension
            observed_rarity = dim_idfs.get(tags, dim_prior)

        dim_rarity = dim_confidence * observed_rarity + (1 - dim_confidence) * dim_prior

        weighted_sum += weight * dim_rarity
        weight_total += weight

    tag_rarity = weighted_sum / weight_total if weight_total > 0 else 0.0

    # Combo rarity
    combo_rarity = 0.0
    if combo_counts is not None:
        combo_key = _combo_key_from_labels(sample_labels)
        if combo_key:
            combo_count = combo_counts.get(combo_key, 0)
            raw_combo_rarity = math.log2(total_samples / (combo_count + 1))
            combo_confidence = _sample_label_confidence(sample_labels, dims=list(rarity_weights.keys()))
            combo_rarity = (
                combo_confidence * raw_combo_rarity
                + (1 - combo_confidence) * tag_rarity
            )

    raw_score = combo_alpha * tag_rarity + (1 - combo_alpha) * combo_rarity

    return {
        "score": raw_score,
        "tag_rarity": round(tag_rarity, 3),
        "combo_rarity": round(combo_rarity, 3),
        "effective_confidence": round(_sample_label_confidence(sample_labels), 3),
        "uncertainty": round(1.0 - _sample_label_confidence(sample_labels), 3),
        "stats_ref": stats_ref_info,
    }


def _labels_have_rarity_signal(labels, dims=None):
    """Return True when labels contain at least one non-empty rarity tag."""
    if not is_usable_labels(labels):
        return False
    target_dims = list(dims) if dims is not None else list(RARITY_WEIGHTS.keys())
    for dim in target_dims:
        tags = labels.get(dim)
        if isinstance(tags, list):
            if any(isinstance(t, str) and t for t in tags):
                return True
        elif isinstance(tags, str) and tags:
            return True
    return False


def _normalize_concepts(labels):
    """Normalize concept labels into sorted string tags."""
    concepts = labels.get("concept", [])
    if isinstance(concepts, str):
        concepts = [concepts]
    if not isinstance(concepts, list):
        return []
    return sorted([c for c in concepts if isinstance(c, str) and c])


def _combo_key_from_labels(labels):
    """Build combo key from (intent, difficulty, concept[:3])."""
    intent = labels.get("intent", "")
    difficulty = labels.get("difficulty", "")
    intent = intent if isinstance(intent, str) else ""
    difficulty = difficulty if isinstance(difficulty, str) else ""
    concepts = _normalize_concepts(labels)[:3]
    if not intent and not difficulty and not concepts:
        return None
    return f"{intent}|{difficulty}|{','.join(concepts)}"


def _update_combo_counts_from_labels(combo_counts, labels):
    """In-place combo count update; returns True when one combo is counted."""
    if isinstance(labels, dict) and labels.get("inherited"):
        return False
    if not _labels_have_rarity_signal(labels):
        return False
    combo_key = _combo_key_from_labels(labels)
    if not combo_key:
        return False
    combo_counts[combo_key] = combo_counts.get(combo_key, 0) + 1
    return True


def build_combo_counts(samples):
    """Build combo occurrence counts from labeled samples for combo rarity."""
    counts = {}
    for s in samples:
        labels = s.get("labels") or {}
        _update_combo_counts_from_labels(counts, labels)
    return counts


def build_tag_distributions(samples, rarity_weights=None):
    """Build per-dimension tag distributions from labeled samples."""
    dims = list((rarity_weights or RARITY_WEIGHTS).keys())
    distributions = {dim: {} for dim in dims}
    total = 0

    for s in samples:
        labels = s.get("labels") or {}
        if labels.get("inherited"):
            continue
        if not _labels_have_rarity_signal(labels, dims=dims):
            continue
        total += 1
        for dim in dims:
            tags = labels.get(dim)
            if tags is None:
                continue
            if isinstance(tags, list):
                if not tags:
                    continue
                for tag in tags:
                    if not isinstance(tag, str) or not tag:
                        continue
                    distributions[dim][tag] = distributions[dim].get(tag, 0) + 1
            else:
                if isinstance(tags, str) and tags:
                    distributions[dim][tags] = distributions[dim].get(tags, 0) + 1

    # Remove empty dimensions
    distributions = {dim: counts for dim, counts in distributions.items() if counts}
    return distributions, total


def _update_distributions_from_labels(distributions, labels, dims):
    """In-place update for streaming distribution construction.

    Returns:
        bool: True when at least one tag is counted into distributions.
    """
    if not is_usable_labels(labels):
        return False
    if labels.get("inherited"):
        return False
    touched = False
    for dim in dims:
        tags = labels.get(dim)
        if tags is None:
            continue
        dim_dist = distributions.setdefault(dim, {})
        if isinstance(tags, list):
            for tag in tags:
                if not isinstance(tag, str) or not tag:
                    continue
                dim_dist[tag] = dim_dist.get(tag, 0) + 1
                touched = True
        else:
            if isinstance(tags, str) and tags:
                dim_dist[tags] = dim_dist.get(tags, 0) + 1
                touched = True
    return touched


def normalize_rarity_scores(rarity_results, mode="percentile", total_samples=0):
    """Normalize raw rarity scores to 1-10 scale.

    Modes:
      - percentile: within-batch percentile mapping (legacy behavior)
      - absolute: absolute mapping by log2(total_samples) ceiling

    Modifies rarity dicts in-place, setting score to normalized value.
    Returns normalization metadata.
    """
    raw_scores = [r["score"] for r in rarity_results if r["score"] is not None]
    if not raw_scores:
        return {}

    if mode == "absolute":
        n = max(int(total_samples), 2)
        raw_ceiling = max(math.log2(n), 1e-9)
        for r in rarity_results:
            if r["score"] is None:
                continue
            raw = max(0.0, min(float(r["score"]), raw_ceiling))
            r["score"] = round(1 + (raw / raw_ceiling) * 9, 1)
        return {"mode": "absolute", "raw_ceiling": round(raw_ceiling, 4), "total_samples": n}

    raw_scores_sorted = sorted(raw_scores)
    n = len(raw_scores_sorted)

    for r in rarity_results:
        if r["score"] is None:
            continue
        # Find percentile rank
        rank = 0
        for i, v in enumerate(raw_scores_sorted):
            if v >= r["score"]:
                rank = i
                break
        else:
            rank = n - 1
        percentile = rank / max(n - 1, 1)
        # Map to 1-10
        r["score"] = round(1 + percentile * 9, 1)

    # Return breakpoints for stats
    breakpoints = {}
    for pct in [10, 25, 50, 75, 90]:
        idx = int(n * pct / 100)
        breakpoints[f"p{pct}"] = raw_scores_sorted[min(idx, n - 1)]
    breakpoints["mode"] = "percentile"
    return breakpoints


def resolve_rarity_mode(config):
    """Resolve rarity normalization mode from config with safe fallback."""
    mode = getattr(config, "rarity_score_mode", RARITY_SCORE_MODE) or RARITY_SCORE_MODE
    mode = str(mode).strip().lower()
    if mode not in {"absolute", "percentile"}:
        return RARITY_SCORE_MODE
    return mode


# ─────────────────────────────────────────────────────────
# Score validation
# ─────────────────────────────────────────────────────────

def _clamp_overall_to_subscores(dim_dict, sub_keys, dim_name, issues):
    """Clamp overall to within ±2 of mean(sub_scores).

    When |overall - mean(sub_scores)| > 2.0, sets overall = round(mean).
    Operates in-place on dim_dict; appends to issues list.
    """
    overall = dim_dict.get("overall")
    if overall is None:
        return
    valid_subs = [dim_dict[k] for k in sub_keys if dim_dict.get(k) is not None]
    if not valid_subs:
        return
    mean_sub = sum(valid_subs) / len(valid_subs)
    if abs(overall - mean_sub) > 2.0:
        clamped = max(1, min(10, round(mean_sub)))
        issues.append(f"{dim_name}.overall {overall} clamped to {clamped} "
                      f"(mean sub-scores={mean_sub:.1f})")
        dim_dict["overall"] = clamped


def validate_score_response(parsed):
    """Validate LLM scoring response.

    Returns (cleaned_result, issues) where issues is a list of strings.
    """
    issues = []
    if parsed is None:
        return None, ["null response"]

    result = {}

    # Validate complexity
    complexity = parsed.get("complexity", {})
    if isinstance(complexity, dict):
        for key in ("instruction", "analytical_depth", "implementation", "overall"):
            val = complexity.get(key)
            if isinstance(val, (int, float)) and not isinstance(val, bool) and 1 <= val <= 10:
                complexity[key] = int(val)
            else:
                complexity[key] = None
                issues.append(f"complexity.{key} invalid: {val}")
        result["complexity"] = complexity
        _clamp_overall_to_subscores(
            complexity, ["instruction", "analytical_depth", "implementation"],
            "complexity", issues)
    else:
        result["complexity"] = {"instruction": None, "analytical_depth": None, "implementation": None, "overall": None}
        issues.append("complexity not a dict")

    # Validate quality
    quality = parsed.get("quality", {})
    if isinstance(quality, dict):
        for key in ("correctness", "code_quality", "explanation", "completeness", "overall"):
            val = quality.get(key)
            if isinstance(val, (int, float)) and not isinstance(val, bool) and 1 <= val <= 10:
                quality[key] = int(val)
            else:
                quality[key] = None
                issues.append(f"quality.{key} invalid: {val}")
        result["quality"] = quality
        # Enforce overall <= correctness + 2 (prompt constraint)
        q_correct = quality.get("correctness")
        q_overall = quality.get("overall")
        if (q_correct is not None and q_overall is not None
                and q_overall > q_correct + 2):
            quality["overall"] = q_correct + 2
            issues.append(f"quality.overall {q_overall} clamped to {quality['overall']} (correctness+2 rule)")
        _clamp_overall_to_subscores(
            quality, ["correctness", "code_quality", "explanation", "completeness"],
            "quality", issues)
    else:
        result["quality"] = {"correctness": None, "code_quality": None, "explanation": None, "completeness": None, "overall": None}
        issues.append("quality not a dict")

    # Validate reasoning
    reasoning = parsed.get("reasoning", {})
    if isinstance(reasoning, dict):
        for key in ("clarity", "consistency", "overall"):
            val = reasoning.get(key)
            if isinstance(val, (int, float)) and not isinstance(val, bool) and 1 <= val <= 10:
                reasoning[key] = int(val)
            else:
                reasoning[key] = None
                if val is not None:
                    issues.append(f"reasoning.{key} invalid: {val}")
        # self_correction can be bool
        sc = reasoning.get("self_correction")
        if isinstance(sc, bool):
            reasoning["self_correction"] = sc
        else:
            reasoning["self_correction"] = None
        result["reasoning"] = reasoning
        _clamp_overall_to_subscores(
            reasoning, ["clarity", "consistency"],
            "reasoning", issues)
    else:
        result["reasoning"] = {"clarity": None, "consistency": None, "self_correction": None, "overall": None}
        issues.append("reasoning not a dict")

    # Validate flags
    flags = parsed.get("flags", [])
    if isinstance(flags, list):
        known = []
        unknown = []
        for f in flags:
            if isinstance(f, str):
                if f in KNOWN_FLAGS:
                    known.append(f)
                else:
                    unknown.append(f)
        result["flags"] = known
        result["unknown_flags"] = unknown
        if unknown:
            issues.append(f"unknown flags: {unknown}")
    else:
        result["flags"] = []
        result["unknown_flags"] = []

    # Validate confidence
    conf = parsed.get("confidence", 0.5)
    if isinstance(conf, (int, float)) and 0 <= conf <= 1:
        result["confidence"] = round(float(conf), 2)
    else:
        result["confidence"] = 0.5
        issues.append(f"confidence invalid: {conf}")

    # Extract optional rationale (when ENABLE_RATIONALE is on)
    rationale = parsed.get("rationale")
    if isinstance(rationale, str) and rationale.strip():
        result["rationale"] = rationale.strip()[:500]

    return result, issues


# ─────────────────────────────────────────────────────────
# Value score computation
# ─────────────────────────────────────────────────────────

def compute_value_score(score_result, rarity_result, weights=None):
    """Compute weighted composite value_score.

    Uses default rarity=5.0 when rarity is missing (instead of renormalizing
    weights) to maintain cross-batch comparability.  Applies quality floor
    penalty: quality.overall < 4 → value *= 0.7.

    Returns float 1-10 or None if no valid scores.
    """
    if weights is None:
        weights = VALUE_WEIGHTS

    components = {}

    # Extract overall scores
    if score_result:
        c = score_result.get("complexity", {})
        if c and c.get("overall") is not None:
            components["complexity"] = c["overall"]
        q = score_result.get("quality", {})
        if q and q.get("overall") is not None:
            components["quality"] = q["overall"]
        r = score_result.get("reasoning", {})
        if r and r.get("overall") is not None:
            components["reasoning"] = r["overall"]

    if rarity_result and rarity_result.get("score") is not None:
        components["rarity"] = rarity_result["score"]

    if not components:
        return None

    # Require at least 2 LLM dimensions for a reliable composite score;
    # with only 1, weight renormalization inflates a single dimension
    llm_dims = sum(1 for k in ("complexity", "quality", "reasoning")
                   if k in components)
    if llm_dims < 2:
        return None

    # Apply rarity default only when we have at least one LLM score
    if "rarity" not in components and "rarity" in weights and weights["rarity"] > 0:
        components["rarity"] = 5.0

    # Weighted mean
    total_weight = sum(weights.get(k, 0) for k in components)
    if total_weight <= 0:
        return None

    value = sum(weights.get(k, 0) * v for k, v in components.items()) / total_weight

    # Quality floor penalty: buggy code should not pass threshold
    quality_overall = components.get("quality")
    quality_weight = weights.get("quality", 0)
    if quality_overall is not None and quality_overall < 4 and quality_weight > 0:
        value *= 0.7

    return round(max(1.0, min(10.0, value)), 1)


# ─────────────────────────────────────────────────────────
# Selection score (intra-class quality ranking)
# ─────────────────────────────────────────────────────────

_SELECTION_DIMS = ["intent", "language", "domain", "concept", "task",
                   "agentic", "constraint", "context", "difficulty"]


def _extract_pure_quality(value_dict, quality_weights=None):
    """Compute quality score without rarity from a value dict.

    Uses the same VALUE_WEIGHTS but excludes rarity and renormalizes.
    """
    if quality_weights is None:
        quality_weights = {k: v for k, v in VALUE_WEIGHTS.items() if k != "rarity"}

    components = {}
    c = value_dict.get("complexity", {})
    if isinstance(c, dict) and c.get("overall") is not None:
        components["complexity"] = c["overall"]
    q = value_dict.get("quality", {})
    if isinstance(q, dict) and q.get("overall") is not None:
        components["quality"] = q["overall"]
    r = value_dict.get("reasoning", {})
    if isinstance(r, dict) and r.get("overall") is not None:
        components["reasoning"] = r["overall"]

    if not components:
        return None

    total_w = sum(quality_weights.get(k, 0) for k in components)
    if total_w <= 0:
        return None

    pq = sum(quality_weights.get(k, 0) * v for k, v in components.items()) / total_w

    # Quality floor penalty: consistent with compute_value_score
    quality_overall = components.get("quality")
    if quality_overall is not None and quality_overall < 4:
        pq *= 0.7

    return pq


def _selection_quality_weights(config=None):
    """Resolve quality-only weights from config.value_weights."""
    base_weights = config.value_weights if config and config.value_weights else VALUE_WEIGHTS
    return {k: v for k, v in base_weights.items() if k != "rarity"}


def _selection_dim_confidence(labels, dim):
    """Confidence used for selection shrinkage and fusion."""
    return max(_SELECTION_CONFIDENCE_FLOOR, _dimension_label_confidence(labels, dim))


def _selection_summary_from_sample(sample):
    """Extract the lightweight fields needed for global selection ranking."""
    value = sample.get("value") or {}
    return {
        "labels": sample.get("labels") or {},
        "complexity_overall": (value.get("complexity") or {}).get("overall"),
        "quality_overall": (value.get("quality") or {}).get("overall"),
        "reasoning_overall": (value.get("reasoning") or {}).get("overall"),
        "rarity_score": (value.get("rarity") or {}).get("score"),
        "value_score": value.get("value_score"),
    }


def _compute_percentiles(indexed_values):
    """Assign tie-aware percentiles to (idx, value) pairs."""
    if not indexed_values:
        return {}

    ordered = sorted(indexed_values, key=lambda item: (item[1], item[0]))
    n = len(ordered)
    scale = max(n - 1, 1)
    percentiles = {}

    pos = 0
    while pos < n:
        end = pos
        value = ordered[pos][1]
        while end + 1 < n and ordered[end + 1][1] == value:
            end += 1
        percentile = ((pos + end) / 2) / scale
        for member_pos in range(pos, end + 1):
            percentiles[ordered[member_pos][0]] = percentile
        pos = end + 1

    return percentiles


def compute_selection_scores(samples, rarity_weights=None,
                              intra_weight=None, min_group_size=None,
                              config=None):
    """Compute intra-class quality rankings and selection scores.

    Post-scoring pass (no LLM). For each tag in each dimension, ranks samples
    by pure quality (quality without rarity) within the tag group, then fuses
    across dimensions using rarity_weights.

    Modifies samples in-place: adds selection_score and intra_class_rank to
    each sample's value dict.

    Args:
        samples: list of sample dicts with 'value' and 'labels' fields
        rarity_weights: per-dimension weights for fusion (default: RARITY_WEIGHTS)
        intra_weight: weight for intra-class quality vs rarity (default: 0.75)
        min_group_size: min samples per tag for percentile computation (default: 30)
        config: PipelineConfig override
    """
    if rarity_weights is None:
        rarity_weights = (config.rarity_weights if config and config.rarity_weights
                          else RARITY_WEIGHTS)
    if intra_weight is None:
        intra_weight = (config.selection_intra_weight if config
                        else SELECTION_INTRA_WEIGHT)
    if min_group_size is None:
        min_group_size = (config.selection_min_group_size if config
                          else SELECTION_MIN_GROUP_SIZE)

    quality_weights = _selection_quality_weights(config)
    label_confidences = [
        {dim: _selection_dim_confidence(s.get("labels") or {}, dim) for dim in _SELECTION_DIMS}
        for s in samples
    ]

    # Step 1: Compute pure_quality for each sample
    pure_qualities = []
    for s in samples:
        v = s.get("value")
        if v:
            pure_qualities.append(_extract_pure_quality(v, quality_weights))
        else:
            pure_qualities.append(None)

    # Step 2: Global percentile (needed for Bayesian shrinkage)
    valid_pq = [(i, pq) for i, pq in enumerate(pure_qualities) if pq is not None]
    global_percentiles = _compute_percentiles(valid_pq)

    # Step 3: Per-dimension, per-tag quality percentile with Bayesian shrinkage
    smoothing_prior = (config.selection_smoothing_prior if config
                       else SELECTION_SMOOTHING_PRIOR)
    dim_percentiles = [{} for _ in samples]  # [{dim: (percentile_sum, count)}, ...]

    for dim in _SELECTION_DIMS:
        # Build tag -> list of (sample_idx, pure_quality)
        tag_groups = {}
        for i, s in enumerate(samples):
            if pure_qualities[i] is None:
                continue
            labels = s.get("labels") or {}
            if not is_usable_labels(labels):
                continue
            tags = labels.get(dim)
            if tags is None:
                continue
            if isinstance(tags, list):
                for t in tags:
                    tag_groups.setdefault(t, []).append((i, pure_qualities[i]))
            else:
                tag_groups.setdefault(tags, []).append((i, pure_qualities[i]))

        for tag, members in tag_groups.items():
            if len(members) < min_group_size:
                continue

            effective_n = sum(label_confidences[idx].get(dim, _SELECTION_CONFIDENCE_FLOOR)
                              for idx, _pq in members)
            # Bayesian shrinkage: blend tag-group percentile toward global
            shrinkage = effective_n / (effective_n + smoothing_prior)
            tag_percentiles = _compute_percentiles(members)
            for idx, _pq in members:
                tag_pct = tag_percentiles[idx]
                g_pct = global_percentiles.get(idx, 0.5)
                sample_conf = label_confidences[idx].get(dim, _SELECTION_CONFIDENCE_FLOOR)
                percentile = (
                    sample_conf * (shrinkage * tag_pct + (1 - shrinkage) * g_pct)
                    + (1 - sample_conf) * g_pct
                )
                # Multi-select: average across tags in this dimension
                if dim in dim_percentiles[idx]:
                    prev, cnt = dim_percentiles[idx][dim]
                    dim_percentiles[idx][dim] = (prev + percentile, cnt + 1)
                else:
                    dim_percentiles[idx][dim] = (percentile, 1)

    # Step 4: Weighted fusion across dimensions → intra_class_rank + selection_score
    quality_weight = (config.selection_quality_weight if config
                      else SELECTION_QUALITY_WEIGHT)
    rarity_weight = 1.0 - intra_weight - quality_weight

    for i, s in enumerate(samples):
        v = s.get("value")
        if v is None:
            continue

        percs = dim_percentiles[i]

        if percs:
            weighted_sum = 0.0
            weight_total = 0.0
            for dim, (pct_sum, pct_cnt) in percs.items():
                w = rarity_weights.get(dim, 1.0) * label_confidences[i].get(dim, _SELECTION_CONFIDENCE_FLOOR)
                weighted_sum += w * (pct_sum / pct_cnt)  # mean percentile
                weight_total += w
            if weight_total > 0:
                fused = weighted_sum / weight_total  # 0-1
                intra_class_rank = round(fused * 9 + 1, 2)
            else:
                intra_class_rank = None
        else:
            # Fallback: global percentile
            if i in global_percentiles:
                intra_class_rank = round(global_percentiles[i] * 9 + 1, 2)
            else:
                intra_class_rank = None

        v["intra_class_rank"] = intra_class_rank

        rarity_score = (v.get("rarity") or {}).get("score")
        pq = pure_qualities[i]
        pq_scaled = max(1.0, min(10.0, pq)) if pq is not None else 5.5

        if intra_class_rank is not None:
            if rarity_score is not None:
                v["selection_score"] = round(
                    intra_weight * intra_class_rank +
                    quality_weight * pq_scaled +
                    rarity_weight * rarity_score, 2)
            else:
                # No rarity: redistribute rarity_weight into intra + quality
                v["selection_score"] = round(
                    (intra_weight + rarity_weight * 0.5) * intra_class_rank +
                    (quality_weight + rarity_weight * 0.5) * pq_scaled, 2)
        else:
            # Ultimate fallback: use value_score
            v["selection_score"] = v.get("value_score")


def compute_selection_scores_from_summaries(summaries, rarity_weights=None,
                                             intra_weight=None,
                                             min_group_size=None, config=None):
    """Compute selection scores from lightweight summary dicts (chunked mode).

    Same algorithm as compute_selection_scores but works on summary dicts
    instead of full sample dicts.

    Args:
        summaries: list of dicts with keys:
            complexity_overall, quality_overall, reasoning_overall,
            rarity_score, labels, value_score
    Returns:
        list of dicts: {selection_score, intra_class_rank} per summary
    """
    if rarity_weights is None:
        rarity_weights = (config.rarity_weights if config and config.rarity_weights
                          else RARITY_WEIGHTS)
    if intra_weight is None:
        intra_weight = (config.selection_intra_weight if config
                        else SELECTION_INTRA_WEIGHT)
    if min_group_size is None:
        min_group_size = (config.selection_min_group_size if config
                          else SELECTION_MIN_GROUP_SIZE)

    quality_weights = _selection_quality_weights(config)
    label_confidences = [
        {dim: _selection_dim_confidence(s.get("labels") or {}, dim) for dim in _SELECTION_DIMS}
        for s in summaries
    ]

    # Step 1: pure_quality from summary fields
    pure_qualities = []
    for s in summaries:
        components = {}
        if s.get("complexity_overall") is not None:
            components["complexity"] = s["complexity_overall"]
        if s.get("quality_overall") is not None:
            components["quality"] = s["quality_overall"]
        if s.get("reasoning_overall") is not None:
            components["reasoning"] = s["reasoning_overall"]
        if components:
            tw = sum(quality_weights.get(k, 0) for k in components)
            pq = sum(quality_weights.get(k, 0) * v for k, v in components.items()) / tw if tw > 0 else None
            # Quality floor penalty: consistent with compute_value_score
            if pq is not None and components.get("quality") is not None and components["quality"] < 4:
                pq *= 0.7
        else:
            pq = None
        pure_qualities.append(pq)

    # Step 2: Global percentile (needed for Bayesian shrinkage)
    valid_pq = [(i, pq) for i, pq in enumerate(pure_qualities) if pq is not None]
    global_percentiles = _compute_percentiles(valid_pq)

    # Step 3: Per-tag percentile with Bayesian shrinkage
    smoothing_prior = (config.selection_smoothing_prior if config
                       else SELECTION_SMOOTHING_PRIOR)
    dim_percentiles = [{} for _ in summaries]

    for dim in _SELECTION_DIMS:
        tag_groups = {}
        for i, s in enumerate(summaries):
            if pure_qualities[i] is None:
                continue
            labels = s.get("labels") or {}
            if not is_usable_labels(labels):
                continue
            tags = labels.get(dim)
            if tags is None:
                continue
            if isinstance(tags, list):
                for t in tags:
                    tag_groups.setdefault(t, []).append((i, pure_qualities[i]))
            else:
                tag_groups.setdefault(tags, []).append((i, pure_qualities[i]))

        for tag, members in tag_groups.items():
            if len(members) < min_group_size:
                continue
            effective_n = sum(label_confidences[idx].get(dim, _SELECTION_CONFIDENCE_FLOOR)
                              for idx, _pq in members)
            shrinkage = effective_n / (effective_n + smoothing_prior)
            tag_percentiles = _compute_percentiles(members)
            for idx, _pq in members:
                tag_pct = tag_percentiles[idx]
                g_pct = global_percentiles.get(idx, 0.5)
                sample_conf = label_confidences[idx].get(dim, _SELECTION_CONFIDENCE_FLOOR)
                percentile = (
                    sample_conf * (shrinkage * tag_pct + (1 - shrinkage) * g_pct)
                    + (1 - sample_conf) * g_pct
                )
                # Multi-select: average across tags in this dimension
                if dim in dim_percentiles[idx]:
                    prev, cnt = dim_percentiles[idx][dim]
                    dim_percentiles[idx][dim] = (prev + percentile, cnt + 1)
                else:
                    dim_percentiles[idx][dim] = (percentile, 1)

    # Step 4: Fusion
    quality_weight = (config.selection_quality_weight if config
                      else SELECTION_QUALITY_WEIGHT)
    rarity_weight = 1.0 - intra_weight - quality_weight

    results = []
    for i, s in enumerate(summaries):
        percs = dim_percentiles[i]

        if percs:
            weighted_sum = 0.0
            weight_total = 0.0
            for dim, (pct_sum, pct_cnt) in percs.items():
                w = rarity_weights.get(dim, 1.0) * label_confidences[i].get(dim, _SELECTION_CONFIDENCE_FLOOR)
                weighted_sum += w * (pct_sum / pct_cnt)  # mean percentile
                weight_total += w
            if weight_total > 0:
                fused = weighted_sum / weight_total
                intra_class_rank = round(fused * 9 + 1, 2)
            else:
                intra_class_rank = None
        else:
            if i in global_percentiles:
                intra_class_rank = round(global_percentiles[i] * 9 + 1, 2)
            else:
                intra_class_rank = None

        rarity_score = s.get("rarity_score")
        pq = pure_qualities[i]
        pq_scaled = max(1.0, min(10.0, pq)) if pq is not None else 5.5

        if intra_class_rank is not None:
            if rarity_score is not None:
                selection_score = round(
                    intra_weight * intra_class_rank +
                    quality_weight * pq_scaled +
                    rarity_weight * rarity_score, 2)
            else:
                selection_score = round(
                    (intra_weight + rarity_weight * 0.5) * intra_class_rank +
                    (quality_weight + rarity_weight * 0.5) * pq_scaled, 2)
        else:
            selection_score = s.get("value_score")

        results.append({
            "selection_score": selection_score,
            "intra_class_rank": intra_class_rank,
        })

    return results


# ─────────────────────────────────────────────────────────
# Per-sample scoring
# ─────────────────────────────────────────────────────────

async def score_one(http_client, sample, model, rarity_result,
                    sample_idx, total, sem, config=None, rate_limiter=None):
    """Score a single sample: truncate, call LLM, validate, compute value_score.

    Retry loop is OUTSIDE the semaphore so failed/retrying samples don't block
    other tasks from acquiring a slot (matching Pass 1's label_one pattern).

    Returns (value_dict, monitor_dict).
    """
    from sft_label.prompts_value import build_scoring_messages

    _max_retries = config.sample_max_retries if config else SAMPLE_MAX_RETRIES
    _weights = config.value_weights if config and config.value_weights else VALUE_WEIGHTS

    sample_id = sample.get("id", f"sample-{sample_idx}")
    conversations = sample.get("conversations", [])
    labels = sample.get("labels") or {}
    metadata = sample.get("metadata") or {}

    monitor = {
        "sample_id": sample_id,
        "llm_calls": 0,
        "status": "pending",
        "attempts": 0,
        "error": None,
        "validation_issues": [],
        "prompt_tokens": 0,
        "completion_tokens": 0,
    }

    if is_partial_labels(labels):
        monitor["status"] = "skipped_partial_labels"
        monitor["error"] = labels.get("partial_reason", "partial labels")
        return None, monitor

    # Detect thinking mode: prefer metadata (Pangu COT stripped during Pass 1),
    # fallback to scanning conversations (ShareGPT retains COT markers)
    thinking_mode = metadata.get("thinking_mode") or detect_thinking_mode(conversations)

    # Extract COT: prefer metadata.cot_text (preserved from Pangu preprocessing),
    # fallback to extracting from conversations (ShareGPT)
    saved_cot = metadata.get("cot_text", "")
    if saved_cot:
        cot_text = saved_cot
    else:
        cot_text, _, _ = extract_cot_content(conversations)

    # Truncate for scoring
    _compact = config.prompt_mode == "compact" if config else False
    _budget = config.value_truncation_budget if config else None
    # Compact mode: reduce budget to fit within firewall size limits
    if _compact and config and _budget == VALUE_TRUNCATION_BUDGET:
        from sft_label.config import COMPACT_VALUE_TRUNCATION_BUDGET
        _budget = COMPACT_VALUE_TRUNCATION_BUDGET
    truncated = truncate_for_scoring(
        conversations, thinking_mode, cot_text=cot_text,
        budget=_budget,
    )

    # Build messages
    code_block_count = count_code_blocks(
        " ".join(t.get("value", "") for t in conversations)
    )
    total_turns = len(conversations)

    # Multi-turn slice position (for incomplete flag calibration)
    turn_index = metadata.get("turn_index")
    total_turns_meta = metadata.get("total_turns")

    messages = build_scoring_messages(
        truncated=truncated,
        thinking_mode=thinking_mode,
        labels=labels,
        total_turns=total_turns,
        code_block_count=code_block_count,
        enable_rationale=config.enable_rationale if config else False,
        turn_index=turn_index,
        total_turns_meta=total_turns_meta,
        compact=_compact,
    )

    # LLM call with retry — retry OUTSIDE semaphore
    for attempt in range(_max_retries):
        if attempt > 0:
            # Backoff outside semaphore so slot is free for others
            base_wait = 2 ** attempt * 2
            await asyncio.sleep(base_wait + random.uniform(0, base_wait))

        async with sem:
            monitor["attempts"] = attempt + 1
            parsed, raw, usage = await async_llm_call(
                http_client, messages, model,
                temperature=0.1, max_tokens=800, config=config,
                rate_limiter=rate_limiter,
            )
            monitor["llm_calls"] += 1
            monitor["prompt_tokens"] += usage.get("prompt_tokens", 0)
            monitor["completion_tokens"] += usage.get("completion_tokens", 0)

        if parsed is None:
            monitor["error"] = raw[:300] if raw else "null response"
            monitor["error_response"] = usage.get("error_response") or (raw[:500] if raw else "")
            # Detect COT mimicry — LLM echoed COT format instead of JSON
            if raw and raw.lstrip().startswith(("```COT", "```cot", "```Cot",
                                                "[COT]", "COT\n", "«cot»")):
                monitor["error"] = f"cot_mimicry: {raw[:200]}"
            if usage.get("non_retryable"):
                break
            continue

        score_result, issues = validate_score_response(parsed)
        monitor["validation_issues"] = issues

        if score_result is None:
            monitor["error"] = "validation failed"
            continue

        # Success
        value_score = compute_value_score(score_result, rarity_result, _weights)

        value = {
            "complexity": score_result.get("complexity"),
            "quality": score_result.get("quality"),
            "reasoning": score_result.get("reasoning"),
            "rarity": rarity_result,
            "flags": score_result.get("flags", []),
            "thinking_mode": thinking_mode,
            "value_score": value_score,
            "confidence": score_result.get("confidence", 0.5),
        }
        monitor["status"] = "success"
        return value, monitor

    # All retries exhausted
    monitor["status"] = "failed"
    return None, monitor


# ─────────────────────────────────────────────────────────
# Aggregate statistics
# ─────────────────────────────────────────────────────────

def _percentiles(values, pcts=(10, 25, 50, 75, 90)):
    """Compute percentiles from a list of values."""
    if not values:
        return {}
    s = sorted(values)
    n = len(s)
    result = {"mean": round(sum(s) / n, 2), "std": 0.0, "min": s[0], "max": s[-1]}
    if n > 1:
        mean = result["mean"]
        result["std"] = round((sum((x - mean) ** 2 for x in s) / (n - 1)) ** 0.5, 2)
    for p in pcts:
        idx = min(int(n * p / 100), n - 1)
        result[f"p{p}"] = round(s[idx], 2)
    return result


def _histogram_bins(values):
    """Compute histogram bins (1-10) from a list of numeric values."""
    bins = [0] * 10
    for v in values:
        if isinstance(v, (int, float)) and 1 <= v <= 10:
            idx = min(int(v) - 1, 9)
            bins[idx] += 1
    return bins


def compute_value_stats(scored_samples, all_monitors):
    """Compute aggregate statistics for value scoring.

    Returns dict matching Pass 2 stats structure.
    """
    values = [s.get("value", {}) for s in scored_samples if s.get("value")]

    total_scored = len(values)
    total_failed = sum(1 for m in all_monitors if m.get("status") == "failed")

    # Score distributions
    def extract_scores(key_path):
        result = []
        for v in values:
            obj = v
            for k in key_path.split("."):
                obj = obj.get(k, {}) if isinstance(obj, dict) else None
                if obj is None:
                    break
            if isinstance(obj, (int, float)):
                result.append(obj)
        return result

    score_distributions = {
        "value_score": _percentiles(extract_scores("value_score")),
        "selection_score": _percentiles(extract_scores("selection_score")),
        "intra_class_rank": _percentiles(extract_scores("intra_class_rank")),
        "complexity_overall": _percentiles(extract_scores("complexity.overall")),
        "quality_overall": _percentiles(extract_scores("quality.overall")),
        "reasoning_overall": _percentiles(extract_scores("reasoning.overall")),
        "rarity_score": _percentiles(extract_scores("rarity.score")),
    }

    # Distribution bias detection (quality.overall)
    distribution_warnings = []
    quality_scores = extract_scores("quality.overall")
    if len(quality_scores) >= 20:
        n_q = len(quality_scores)
        buckets = {
            "1-3": sum(1 for s in quality_scores if 1 <= s <= 3) / n_q * 100,
            "4-6": sum(1 for s in quality_scores if 4 <= s <= 6) / n_q * 100,
            "7-8": sum(1 for s in quality_scores if 7 <= s <= 8) / n_q * 100,
            "9-10": sum(1 for s in quality_scores if 9 <= s <= 10) / n_q * 100,
        }
        expected = {"1-3": 15, "4-6": 50, "7-8": 30, "9-10": 5}
        for bucket, actual_pct in buckets.items():
            exp_pct = expected[bucket]
            deviation = actual_pct - exp_pct
            if abs(deviation) > 20:
                direction = "over" if deviation > 0 else "under"
                distribution_warnings.append(
                    f"quality.overall bucket {bucket}: {actual_pct:.1f}% "
                    f"({direction}-represented vs expected {exp_pct}%, "
                    f"deviation={deviation:+.1f}pp)"
                )
    score_distributions["distribution_warnings"] = distribution_warnings

    # Confidence distribution
    confidence_scores = extract_scores("confidence")
    score_distributions["confidence"] = _percentiles(confidence_scores)

    # Attach raw score arrays for cross-file merging (not serialized to JSON)
    _raw_scores = {
        "value_score": extract_scores("value_score"),
        "selection_score": extract_scores("selection_score"),
        "intra_class_rank": extract_scores("intra_class_rank"),
        "complexity_overall": extract_scores("complexity.overall"),
        "quality_overall": extract_scores("quality.overall"),
        "reasoning_overall": extract_scores("reasoning.overall"),
        "rarity_score": extract_scores("rarity.score"),
    }

    # Histogram bins (1-10) for dashboard charts
    histograms = {k: _histogram_bins(v) for k, v in _raw_scores.items()}

    # Sub-score means
    sub_score_means = {
        "complexity": {},
        "quality": {},
        "reasoning": {},
    }
    for dim in ("complexity", "quality", "reasoning"):
        obj_template = values[0].get(dim, {}) if values else {}
        for key in obj_template:
            scores = extract_scores(f"{dim}.{key}")
            scores = [s for s in scores if isinstance(s, (int, float))]
            if scores:
                sub_score_means[dim][key] = round(sum(scores) / len(scores), 2)

    # Value by tag (mean value per tag per dimension)
    tag_dims = ["intent", "difficulty", "domain", "concept", "task", "agentic", "constraint", "context"]
    value_by_tag = {}
    selection_by_tag = {}
    for dim in tag_dims:
        tag_values = {}  # {tag: [value_scores]}
        tag_selections = {}  # {tag: [selection_scores]}
        for s in scored_samples:
            v = s.get("value", {})
            vs = v.get("value_score")
            ss = v.get("selection_score")
            if vs is None:
                continue
            labels = s.get("labels") or {}
            tags = labels.get(dim)
            if tags is None:
                continue
            if isinstance(tags, list):
                for t in tags:
                    tag_values.setdefault(t, []).append(vs)
                    if ss is not None:
                        tag_selections.setdefault(t, []).append(ss)
            else:
                tag_values.setdefault(tags, []).append(vs)
                if ss is not None:
                    tag_selections.setdefault(tags, []).append(ss)
        value_by_tag[dim] = {
            t: {"mean": round(sum(vs) / len(vs), 2), "n": len(vs)}
            for t, vs in sorted(tag_values.items(), key=lambda x: -sum(x[1]) / len(x[1]))
        }
        selection_by_tag[dim] = {
            t: {"mean": round(sum(vs) / len(vs), 2), "n": len(vs)}
            for t, vs in sorted(tag_selections.items(), key=lambda x: -sum(x[1]) / len(x[1]))
        }

    # Thinking mode stats
    slow_values = [v for v in values if v.get("thinking_mode") == "slow"]
    fast_values = [v for v in values if v.get("thinking_mode") == "fast"]
    thinking_mode_stats = {
        "slow": {
            "count": len(slow_values),
            "mean_value": round(sum(v.get("value_score", 0) or 0 for v in slow_values) / max(len(slow_values), 1), 2),
            "mean_quality": round(sum((v.get("quality", {}).get("overall") or 0) for v in slow_values) / max(len(slow_values), 1), 2),
            "mean_reasoning": round(sum((v.get("reasoning", {}).get("overall") or 0) for v in slow_values) / max(len(slow_values), 1), 2),
        },
        "fast": {
            "count": len(fast_values),
            "mean_value": round(sum(v.get("value_score", 0) or 0 for v in fast_values) / max(len(fast_values), 1), 2),
            "mean_quality": round(sum((v.get("quality", {}).get("overall") or 0) for v in fast_values) / max(len(fast_values), 1), 2),
            "mean_reasoning": round(sum((v.get("reasoning", {}).get("overall") or 0) for v in fast_values) / max(len(fast_values), 1), 2),
        },
    }

    # Flag counts and impact
    flag_counts = {}
    flag_value_sums = {}
    for v in values:
        for f in v.get("flags", []):
            flag_counts[f] = flag_counts.get(f, 0) + 1
            flag_value_sums.setdefault(f, []).append(v.get("value_score", 0) or 0)

    flag_value_impact = {
        f: {"mean_value": round(sum(vs) / len(vs), 2), "count": len(vs)}
        for f, vs in flag_value_sums.items()
    }

    # Selection thresholds
    all_value_scores = sorted([v.get("value_score") for v in values if v.get("value_score") is not None])
    n_vs = len(all_value_scores)
    selection_thresholds = {}
    for pct_label, pct in [("top_10pct", 90), ("top_25pct", 75), ("top_50pct", 50)]:
        if n_vs > 0:
            idx = min(int(n_vs * pct / 100), n_vs - 1)
            threshold = all_value_scores[idx]
            count = sum(1 for v in all_value_scores if v >= threshold)
            selection_thresholds[pct_label] = {"threshold": round(threshold, 1), "count": count}

    # Coverage at thresholds
    coverage_at_thresholds = {}
    for threshold_val in [5.0, 6.0, 7.0, 8.0]:
        retained = [s for s in scored_samples
                    if s.get("value", {}).get("value_score") is not None
                    and s["value"]["value_score"] >= threshold_val]
        all_tags = set()
        retained_tags = set()
        for s in scored_samples:
            labels = s.get("labels") or {}
            for dim in tag_dims:
                t = labels.get(dim)
                if isinstance(t, list):
                    all_tags.update(f"{dim}:{x}" for x in t)
                elif t:
                    all_tags.add(f"{dim}:{t}")
        for s in retained:
            labels = s.get("labels") or {}
            for dim in tag_dims:
                t = labels.get(dim)
                if isinstance(t, list):
                    retained_tags.update(f"{dim}:{x}" for x in t)
                elif t:
                    retained_tags.add(f"{dim}:{t}")

        lost = sorted(all_tags - retained_tags)
        coverage_at_thresholds[str(threshold_val)] = {
            "retained": len(retained),
            "pct": round(len(retained) / max(total_scored, 1), 2),
            "tags_lost": lost[:20],
            "coverage": round(len(retained_tags) / max(len(all_tags), 1), 3),
        }

    # LLM usage
    total_llm_calls = sum(m.get("llm_calls", 0) for m in all_monitors)
    total_prompt_tokens = sum(m.get("prompt_tokens", 0) for m in all_monitors)
    total_completion_tokens = sum(m.get("completion_tokens", 0) for m in all_monitors)

    return {
        "total_scored": total_scored,
        "total_failed": total_failed,
        "total_llm_calls": total_llm_calls,
        "total_prompt_tokens": total_prompt_tokens,
        "total_completion_tokens": total_completion_tokens,
        "total_tokens": total_prompt_tokens + total_completion_tokens,
        "score_distributions": score_distributions,
        "_raw_scores": _raw_scores,
        "histograms": histograms,
        "sub_score_means": sub_score_means,
        "value_by_tag": value_by_tag,
        "selection_by_tag": selection_by_tag,
        "thinking_mode_stats": thinking_mode_stats,
        "flag_counts": dict(sorted(flag_counts.items(), key=lambda x: -x[1])),
        "flag_value_impact": flag_value_impact,
        "selection_thresholds": selection_thresholds,
        "coverage_at_thresholds": coverage_at_thresholds,
    }


# ─────────────────────────────────────────────────────────
# Pipeline entry points
# ─────────────────────────────────────────────────────────

def _create_progress():
    """Create a Rich progress bar for scoring."""
    return Progress(
        SpinnerColumn(),
        TextColumn("{task.description}"),
        BarColumn(bar_width=30),
        MofNCompleteColumn(),
        TextColumn("•", style="dim"),
        TimeElapsedColumn(),
        TextColumn("•", style="dim"),
        TimeRemainingColumn(),
        TextColumn("{task.fields[info]}", style="cyan"),
    )


async def run_scoring(input_path, output_dir=None, tag_stats_path=None,
                      limit=0, config=None, resume=False, llm_progress_cb=None):
    """Run value scoring (Pass 2) on pre-labeled data.

    Args:
        input_path: Path to labeled.json or directory of labeled files
        output_dir: Where to write outputs (default: same directory as input)
        tag_stats_path: Path to Pass 1 stats for rarity computation
        limit: Max samples to score (0 = all)
        config: PipelineConfig override
        resume: If True, skip samples that already have scores in scored.jsonl
    """
    if config is None:
        config = PipelineConfig()

    input_path = Path(input_path)

    # Determine if directory or single file
    if input_path.is_dir():
        return await _run_scoring_directory(
            input_path, output_dir, tag_stats_path, limit, config,
            llm_progress_cb=llm_progress_cb,
        )
    else:
        return await _run_scoring_file(
            input_path, output_dir, tag_stats_path, limit, config,
            resume=resume, llm_progress_cb=llm_progress_cb,
        )


async def _run_scoring_file_chunked(input_path, output_dir, tag_stats_path,
                                     limit, config, llm_progress_cb=None):
    """Score a JSONL labeled file in chunks to bound memory.

    Two-pass approach:
      Pass A (lightweight, no LLM): Stream through JSONL to compute rarity.
      Pass B (chunked, watermark LLM): Read chunks, attach pre-computed rarity,
             run LLM scoring, write scored JSONL incrementally.
    """
    from sft_label.pipeline import iter_chunks_from_jsonl

    input_path = Path(input_path)
    if output_dir is None:
        output_dir = input_path.parent
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    chunk_size = config.chunk_size if config else CHUNK_SIZE
    max_active = config.max_active_chunks if config else MAX_ACTIVE_CHUNKS

    # ── Load tag stats for rarity ──
    rarity_mode = resolve_rarity_mode(config)
    stats_ref_info = None
    idf_map = {}
    total_stats_samples = 0
    combo_counts = None
    combo_mode = "disabled"

    if tag_stats_path is None:
        candidate = input_path.parent / PASS1_STATS_FILE
        if candidate.exists():
            tag_stats_path = str(candidate)

    stats_source = tag_stats_path
    if stats_source:
        distributions, total_stats_samples, ts, meta = load_tag_stats_context(stats_source)
        if distributions:
            idf_map = compute_tag_idf(distributions, total_stats_samples)
            combo_counts = meta.get("combo_counts")
            stats_ref_info = {
                "source": str(stats_source),
                "total_samples": total_stats_samples,
                "timestamp": ts or datetime.now().isoformat(),
            }
            missing_dims = [
                d for d in (config.rarity_weights or RARITY_WEIGHTS).keys()
                if d not in idf_map
            ]
            if missing_dims:
                print("  Warning: rarity baseline missing dimensions; ignored in IDF: "
                      + ", ".join(missing_dims))
            if combo_counts:
                combo_mode = "external"
            else:
                combo_mode = "disabled"
                print("  External stats has no combo_distributions; "
                      "combo rarity disabled for cross-dataset comparability")
        else:
            print(f"  Warning: {stats_source} has no tag_distributions, fallback to local baseline")
    else:
        print("  No external tag stats found, fallback to local rarity baseline")

    # ── Pass A: Compute rarity (stream, no LLM) ──
    raw_rarities = []
    local_combo_counts = {}
    processed_count = 0
    usable_count = 0
    local_dims = list((config.rarity_weights or RARITY_WEIGHTS).keys())
    local_distributions = {dim: {} for dim in local_dims}
    with open(input_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            sample = json.loads(line)
            labels = sample.get("labels") or {}
            if _update_distributions_from_labels(local_distributions, labels, local_dims):
                usable_count += 1
            _update_combo_counts_from_labels(local_combo_counts, labels)
            processed_count += 1
            del sample
            if limit > 0 and processed_count >= limit:
                break

    if not idf_map and usable_count > 0:
        # Fallback: build IDF from current labeled file so duplicates do not
        # collapse into pseudo-uniform percentile-only rarity.
        local_distributions = {d: c for d, c in local_distributions.items() if c}
        if local_distributions:
            total_stats_samples = usable_count
            idf_map = compute_tag_idf(local_distributions, total_stats_samples)
            stats_source = f"{input_path}#local"
            stats_ref_info = {
                "source": stats_source,
                "total_samples": total_stats_samples,
                "timestamp": datetime.now().isoformat(),
            }
            combo_counts = local_combo_counts
            combo_mode = "local"
            print(f"  Using local rarity baseline ({total_stats_samples} samples)")
    elif idf_map and combo_counts:
        combo_mode = "external"

    if not idf_map:
        print("  Warning: rarity unavailable (no valid tag distributions)")

    # Second lightweight pass to compute per-sample rarity (needs combo_counts)
    sample_idx = 0
    with open(input_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            sample = json.loads(line)
            labels = sample.get("labels") or {}
            rarity = compute_sample_rarity(
                labels, idf_map, total_stats_samples,
                rarity_weights=config.rarity_weights or RARITY_WEIGHTS,
                combo_alpha=config.rarity_combo_alpha,
                combo_counts=combo_counts,
                stats_ref_info=stats_ref_info,
            )
            raw_rarities.append(rarity)
            del sample
            sample_idx += 1
            if limit > 0 and sample_idx >= limit:
                break

    normalize_rarity_scores(
        raw_rarities,
        mode=rarity_mode,
        total_samples=total_stats_samples,
    )
    total = len(raw_rarities)

    # ── Pass B: Chunked LLM scoring ──
    _rps = f"rps={config.rps_limit}(warmup={config.rps_warmup}s)" if config.rps_limit > 0 else "rps=unlimited"
    print(f"  Scoring {total} samples | model={config.scoring_model} concurrency={config.scoring_concurrency} {_rps}")

    rate_limiter = AsyncRateLimiter(config.rps_limit, warmup=config.rps_warmup) if config.rps_limit > 0 else None
    sem = asyncio.Semaphore(config.scoring_concurrency)
    scored_count = 0
    failed_count = 0
    first_error_logged = False
    start_time = time.time()

    # Lightweight per-sample summaries for final stats
    score_summaries = []  # list of dicts: {value_score, complexity, quality, reasoning, rarity, flags, thinking_mode, labels}
    all_monitors = []

    out_scored = open(output_dir / "scored.jsonl", "w", encoding="utf-8")
    out_monitor = open(output_dir / "monitor_value.jsonl", "w", encoding="utf-8")
    out_failed = open(output_dir / "failed_value.jsonl", "w", encoding="utf-8")
    out_failures_log = open(output_dir / "score_failures.jsonl", "w", encoding="utf-8")

    try:
        async with httpx.AsyncClient(
            proxy=None,
            timeout=config.request_timeout,
            limits=httpx.Limits(
                max_connections=config.scoring_concurrency + 10,
                max_keepalive_connections=config.scoring_concurrency,
            ),
        ) as client:

            # Chunk state tracking
            global_offset = 0
            active_chunks = {}  # chunk_idx -> {samples, pending_count, done_samples, offset}
            pending_futures = set()
            chunk_gen = iter_chunks_from_jsonl(input_path, chunk_size, limit=limit)
            chunks_loaded = 0
            gen_exhausted = False

            async def _tagged_score(idx, chunk_idx, sample, rarity_result):
                nonlocal scored_count, failed_count, first_error_logged
                value, monitor = await score_one(
                    client, sample, config.scoring_model, rarity_result,
                    idx, total, sem, config=config, rate_limiter=rate_limiter,
                )
                return chunk_idx, idx, value, monitor, sample

            def load_next_scoring_chunk():
                nonlocal chunks_loaded, global_offset, gen_exhausted
                try:
                    raw_records = next(chunk_gen)
                except StopIteration:
                    gen_exhausted = True
                    return None

                # These are already-labeled samples from the JSONL
                samples = []
                for raw in raw_records:
                    samples.append(raw)

                chunk_idx = chunks_loaded
                offset = global_offset
                global_offset += len(samples)
                chunks_loaded += 1

                active_chunks[chunk_idx] = {
                    "samples": samples,
                    "offset": offset,
                    "total": len(samples),
                    "done": 0,
                    "values": [None] * len(samples),
                    "monitors": [None] * len(samples),
                }

                # Submit scoring tasks
                for i, sample in enumerate(samples):
                    abs_idx = offset + i
                    rarity_result = raw_rarities[abs_idx] if abs_idx < len(raw_rarities) else {"score": None, "tag_rarity": None, "combo_rarity": None, "stats_ref": stats_ref_info}
                    fut = asyncio.ensure_future(
                        _tagged_score(abs_idx, chunk_idx, sample, rarity_result)
                    )
                    pending_futures.add(fut)

                return chunk_idx

            def maybe_load_more_scoring():
                active_count = sum(1 for c in active_chunks.values()
                                   if c["done"] < c["total"])
                while (not gen_exhausted
                       and len(pending_futures) < int(config.scoring_concurrency * config.dir_pipeline_watermark)
                       and active_count < max_active):
                    if load_next_scoring_chunk() is None:
                        break
                    active_count += 1

            def flush_scoring_chunk(chunk_idx):
                nonlocal scored_count, failed_count
                chunk_data = active_chunks[chunk_idx]
                for i, sample in enumerate(chunk_data["samples"]):
                    value = chunk_data["values"][i]
                    monitor = chunk_data["monitors"][i]

                    if value:
                        sample["value"] = value
                        scored_count += 1
                        # Summary for final stats
                        score_summaries.append({
                            "value_score": value.get("value_score"),
                            "complexity_overall": (value.get("complexity") or {}).get("overall"),
                            "quality_overall": (value.get("quality") or {}).get("overall"),
                            "reasoning_overall": (value.get("reasoning") or {}).get("overall"),
                            "rarity_score": (value.get("rarity") or {}).get("score"),
                            "flags": value.get("flags", []),
                            "thinking_mode": value.get("thinking_mode"),
                            "labels": sample.get("labels"),
                        })
                    else:
                        failed_count += 1
                        out_failed.write(json.dumps(sample, ensure_ascii=False) + "\n")
                        # Failure log record
                        record = {
                            "sample_id": sample.get("id", f"sample-{chunk_data['offset'] + i}"),
                            "status": monitor["status"] if monitor else "no_result",
                            "error": (monitor.get("error", "") if monitor else "no monitor record"),
                            "error_response": (monitor.get("error_response", "")[:1000] if monitor else ""),
                            "attempts": (monitor.get("attempts", 0) if monitor else 0),
                        }
                        out_failures_log.write(json.dumps(record, ensure_ascii=False) + "\n")

                    out_scored.write(json.dumps(sample, ensure_ascii=False) + "\n")

                    if monitor:
                        all_monitors.append(monitor)
                        out_monitor.write(json.dumps(monitor, ensure_ascii=False) + "\n")

                # Release
                chunk_data["samples"] = None
                chunk_data["values"] = None
                chunk_data["monitors"] = None

            with _create_progress() as progress:
                task = progress.add_task("Pass 2", total=total, info="")

                # Initial load
                maybe_load_more_scoring()

                while pending_futures:
                    done, pending_futures = await asyncio.wait(
                        pending_futures, return_when=asyncio.FIRST_COMPLETED)

                    for fut in done:
                        chunk_idx, abs_idx, value, monitor, _ = fut.result()
                        chunk_data = active_chunks[chunk_idx]
                        local_idx = abs_idx - chunk_data["offset"]

                        chunk_data["values"][local_idx] = value
                        chunk_data["monitors"][local_idx] = monitor
                        chunk_data["done"] += 1

                        if not value and not first_error_logged and monitor:
                            err = monitor.get("error", "unknown")
                            err_resp = monitor.get("error_response", "")
                            print(f"  [!] First failure: sample={monitor.get('sample_id')} "
                                  f"attempts={monitor.get('attempts')} err={err[:200]}")
                            if err_resp and err_resp != err:
                                print(f"      response={err_resp[:200]}")
                            first_error_logged = True

                        _info = format_progress_info(
                            ok_count=scored_count + (1 if value else 0),
                            fail_count=failed_count + (0 if value else 1),
                            request_stats=rate_limiter.stats if rate_limiter else None,
                        )
                        if llm_progress_cb and monitor:
                            run_info = llm_progress_cb(monitor.get("llm_calls", 0), "pass2")
                            if run_info:
                                _info = f"{_info} • {run_info}"
                        progress.update(task, advance=1, info=_info)

                        # Check if chunk is done
                        if chunk_data["done"] >= chunk_data["total"]:
                            flush_scoring_chunk(chunk_idx)

                    maybe_load_more_scoring()

    finally:
        out_scored.close()
        out_monitor.close()
        out_failed.close()
        out_failures_log.close()

    elapsed = time.time() - start_time

    # Remove empty failed file
    failed_path = output_dir / "failed_value.jsonl"
    if failed_path.exists() and failed_path.stat().st_size == 0:
        failed_path.unlink()

    # Remove empty failure log
    failures_log_path = output_dir / "score_failures.jsonl"
    if failures_log_path.exists() and failures_log_path.stat().st_size == 0:
        failures_log_path.unlink()

    # ── Compute selection scores from summaries ──
    selection_results = compute_selection_scores_from_summaries(
        score_summaries, config=config)

    # Attach selection scores back to summaries (for stats computation)
    for summary, sel in zip(score_summaries, selection_results):
        summary["selection_score"] = sel["selection_score"]
        summary["intra_class_rank"] = sel["intra_class_rank"]

    # Re-stream scored.jsonl to attach selection_score and intra_class_rank
    scored_path = output_dir / "scored.jsonl"
    scored_tmp = output_dir / "scored.jsonl.tmp"
    scored_idx = 0  # index into selection_results (only for samples with value)
    with open(scored_path, "r", encoding="utf-8") as fin, \
         open(scored_tmp, "w", encoding="utf-8") as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            sample = json.loads(line)
            if sample.get("value"):
                if scored_idx < len(selection_results):
                    sample["value"]["selection_score"] = selection_results[scored_idx]["selection_score"]
                    sample["value"]["intra_class_rank"] = selection_results[scored_idx]["intra_class_rank"]
                    scored_idx += 1
            fout.write(json.dumps(sample, ensure_ascii=False) + "\n")
    scored_tmp.rename(scored_path)

    # ── Compute stats from summaries ──
    stats = _compute_value_stats_from_summaries(score_summaries, all_monitors, total)
    stats["elapsed_seconds"] = round(elapsed, 1)
    stats["model"] = config.scoring_model
    stats["input_file"] = str(input_path)
    if rate_limiter:
        stats["http_request_stats"] = rate_limiter.stats.to_dict()
    stats["weights_used"] = config.value_weights or VALUE_WEIGHTS
    stats["rarity_config"] = {
        "stats_ref": str(stats_source) if stats_source else None,
        "total_samples_in_distribution": total_stats_samples,
        "dimension_weights": config.rarity_weights or RARITY_WEIGHTS,
        "combo_alpha": config.rarity_combo_alpha,
        "combo_mode": combo_mode,
        "score_mode": rarity_mode,
    }
    stats["chunked"] = True

    stats_path = output_dir / PASS2_STATS_FILE
    stats_to_write = {k: v for k, v in stats.items() if k != "_raw_scores"}
    tmp_stats = stats_path.with_suffix(".tmp.json")
    with open(tmp_stats, "w", encoding="utf-8") as f:
        json.dump(stats_to_write, f, ensure_ascii=False, indent=2)
    os.replace(tmp_stats, stats_path)

    # Conversation-level aggregation (re-read scored.jsonl for chunked mode)
    try:
        from sft_label.conversation import aggregate_conversations, write_conversation_scores
        scored_path = output_dir / "scored.jsonl"
        if scored_path.exists():
            conv_samples = []
            with open(scored_path, encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        conv_samples.append(json.loads(line))
            conv_records = aggregate_conversations(conv_samples)
            if conv_records:
                write_conversation_scores(conv_records, output_dir / "conversation_scores.json")
            del conv_samples
    except Exception as e:
        print(f"  Warning: conversation aggregation failed: {e}")

    # Dashboard
    try:
        from sft_label.tools.visualize_value import generate_value_dashboard
        generate_value_dashboard(output_dir, scored_file="scored.jsonl",
                                 stats_file=PASS2_STATS_FILE,
                                 output_file=PASS2_DASHBOARD_FILE)
    except Exception as e:
        print(f"  Warning: dashboard generation failed: {e}")

    print_scoring_summary(stats, output_dir)

    return stats


def _compute_value_stats_from_summaries(summaries, all_monitors, total_input):
    """Compute value stats from lightweight per-sample summaries.

    Produces the same structure as compute_value_stats but without needing
    all scored samples in memory.
    """
    total_scored = len(summaries)
    total_failed = sum(1 for m in all_monitors if m.get("status") == "failed")

    # Score distributions
    def _gather(key):
        return [s[key] for s in summaries if s.get(key) is not None]

    score_distributions = {
        "value_score": _percentiles(_gather("value_score")),
        "selection_score": _percentiles(_gather("selection_score")),
        "intra_class_rank": _percentiles(_gather("intra_class_rank")),
        "complexity_overall": _percentiles(_gather("complexity_overall")),
        "quality_overall": _percentiles(_gather("quality_overall")),
        "reasoning_overall": _percentiles(_gather("reasoning_overall")),
        "rarity_score": _percentiles(_gather("rarity_score")),
    }

    # Histogram bins (1-10) for dashboard charts
    _hist_keys = ["value_score", "selection_score", "intra_class_rank",
                  "complexity_overall", "quality_overall", "reasoning_overall",
                  "rarity_score"]
    histograms = {k: _histogram_bins(_gather(k)) for k in _hist_keys}

    # Thinking mode stats
    slow = [s for s in summaries if s.get("thinking_mode") == "slow"]
    fast = [s for s in summaries if s.get("thinking_mode") == "fast"]
    thinking_mode_stats = {
        "slow": {
            "count": len(slow),
            "mean_value": round(sum(s.get("value_score", 0) or 0 for s in slow) / max(len(slow), 1), 2),
            "mean_quality": round(sum(s.get("quality_overall", 0) or 0 for s in slow) / max(len(slow), 1), 2),
            "mean_reasoning": round(sum(s.get("reasoning_overall", 0) or 0 for s in slow) / max(len(slow), 1), 2),
        },
        "fast": {
            "count": len(fast),
            "mean_value": round(sum(s.get("value_score", 0) or 0 for s in fast) / max(len(fast), 1), 2),
            "mean_quality": round(sum(s.get("quality_overall", 0) or 0 for s in fast) / max(len(fast), 1), 2),
            "mean_reasoning": round(sum(s.get("reasoning_overall", 0) or 0 for s in fast) / max(len(fast), 1), 2),
        },
    }

    # Flag counts
    flag_counts = {}
    flag_value_sums = {}
    for s in summaries:
        for f in s.get("flags", []):
            flag_counts[f] = flag_counts.get(f, 0) + 1
            flag_value_sums.setdefault(f, []).append(s.get("value_score", 0) or 0)

    flag_value_impact = {
        f: {"mean_value": round(sum(vs) / len(vs), 2), "count": len(vs)}
        for f, vs in flag_value_sums.items()
    }

    # Selection thresholds
    all_value_scores = sorted([s["value_score"] for s in summaries
                                if s.get("value_score") is not None])
    n_vs = len(all_value_scores)
    selection_thresholds = {}
    for pct_label, pct in [("top_10pct", 90), ("top_25pct", 75), ("top_50pct", 50)]:
        if n_vs > 0:
            idx = min(int(n_vs * pct / 100), n_vs - 1)
            threshold = all_value_scores[idx]
            count = sum(1 for v in all_value_scores if v >= threshold)
            selection_thresholds[pct_label] = {"threshold": round(threshold, 1), "count": count}

    # Value and selection by tag
    tag_dims = ["intent", "difficulty", "domain", "concept", "task",
                "agentic", "constraint", "context"]
    value_by_tag = {}
    selection_by_tag = {}
    for dim in tag_dims:
        tag_values = {}
        tag_selections = {}
        for s in summaries:
            vs = s.get("value_score")
            ss = s.get("selection_score")
            if vs is None:
                continue
            labels = s.get("labels") or {}
            tags = labels.get(dim)
            if tags is None:
                continue
            if isinstance(tags, list):
                for t in tags:
                    tag_values.setdefault(t, []).append(vs)
                    if ss is not None:
                        tag_selections.setdefault(t, []).append(ss)
            else:
                tag_values.setdefault(tags, []).append(vs)
                if ss is not None:
                    tag_selections.setdefault(tags, []).append(ss)
        value_by_tag[dim] = {
            t: {"mean": round(sum(vs) / len(vs), 2), "n": len(vs)}
            for t, vs in sorted(tag_values.items(), key=lambda x: -sum(x[1]) / len(x[1]))
        }
        selection_by_tag[dim] = {
            t: {"mean": round(sum(vs) / len(vs), 2), "n": len(vs)}
            for t, vs in sorted(tag_selections.items(), key=lambda x: -sum(x[1]) / len(x[1]))
        }

    # LLM usage
    total_llm_calls = sum(m.get("llm_calls", 0) for m in all_monitors)
    total_prompt_tokens = sum(m.get("prompt_tokens", 0) for m in all_monitors)
    total_completion_tokens = sum(m.get("completion_tokens", 0) for m in all_monitors)

    return {
        "total_scored": total_scored,
        "total_failed": total_failed,
        "total_llm_calls": total_llm_calls,
        "total_prompt_tokens": total_prompt_tokens,
        "total_completion_tokens": total_completion_tokens,
        "total_tokens": total_prompt_tokens + total_completion_tokens,
        "score_distributions": score_distributions,
        "histograms": histograms,
        "value_by_tag": value_by_tag,
        "selection_by_tag": selection_by_tag,
        "thinking_mode_stats": thinking_mode_stats,
        "flag_counts": dict(sorted(flag_counts.items(), key=lambda x: -x[1])),
        "flag_value_impact": flag_value_impact,
        "selection_thresholds": selection_thresholds,
    }


def print_scoring_summary(stats, run_dir, is_batch=False):
    """Print final scoring summary to stdout (mirrors Pass 1's print_summary)."""
    print(f"\n{'='*80}")
    label = "BATCH SCORING COMPLETE" if is_batch else "SCORING COMPLETE"
    elapsed = stats.get('elapsed_seconds', 0)
    print(f"{label} in {elapsed:.1f}s ({elapsed/60:.1f} min)")
    print(f"{'='*80}")

    total_scored = stats.get('total_scored', 0)
    total_failed = stats.get('total_failed', 0)
    total = total_scored + total_failed
    success_rate = total_scored / total * 100 if total > 0 else 0
    if is_batch:
        print(f"Files:       {stats.get('files_processed', '?')}")
    print(f"Scored:      {total_scored}/{total} ({success_rate:.1f}%)")
    print(f"LLM calls:   {stats.get('total_llm_calls', 0)}")
    total_tokens = stats.get('total_tokens', 0)
    print(f"Tokens:      {total_tokens:,}")
    if elapsed > 0 and total_scored > 0:
        print(f"Throughput:  {total_scored / elapsed:.1f} samples/sec")

    # HTTP request stats
    http = stats.get("http_request_stats")
    if http and http.get("total_http_requests", 0) > 0:
        t = http["total_http_requests"]
        ok = http["success"]
        rate = http["success_rate"]
        errs = http.get("errors", {})
        timeouts = http.get("timeouts", 0)
        err_parts = [f"{code}×{n}" for code, n in errs.items()]
        if timeouts:
            err_parts.append(f"timeout×{timeouts}")
        err_str = f" — errors: {', '.join(err_parts)}" if err_parts else ""
        print(f"HTTP:        {ok}/{t} ({rate}%){err_str}")

    # Score distributions
    dists = stats.get("score_distributions", {})
    dims = [
        ("value_score", "Value"),
        ("complexity_overall", "Complexity"),
        ("quality_overall", "Quality"),
        ("reasoning_overall", "Reasoning"),
        ("rarity_score", "Rarity"),
        ("selection_score", "Selection"),
    ]
    print(f"\nScore distributions (mean ± std):")
    for key, label in dims:
        d = dists.get(key, {})
        mean = d.get("mean", 0)
        std = d.get("std", 0)
        p50 = d.get("p50", 0)
        bar = "█" * int(mean)
        print(f"  {label:12s} {mean:5.2f} ±{std:4.2f}  p50={p50:.1f}  {bar}")

    # Thinking mode breakdown
    ts = stats.get("thinking_mode_stats", {})
    slow = ts.get("slow", {})
    fast = ts.get("fast", {})
    if slow.get("count", 0) > 0 or fast.get("count", 0) > 0:
        print(f"\nThinking modes:")
        if slow.get("count", 0) > 0:
            print(f"  slow:  {slow['count']:4d}  val={slow.get('mean_value', 0):.2f}  "
                  f"qual={slow.get('mean_quality', 0):.2f}  "
                  f"reas={slow.get('mean_reasoning', 0):.2f}")
        if fast.get("count", 0) > 0:
            print(f"  fast:  {fast['count']:4d}  val={fast.get('mean_value', 0):.2f}  "
                  f"qual={fast.get('mean_quality', 0):.2f}  "
                  f"reas={fast.get('mean_reasoning', 0):.2f}")

    # Top flags
    flag_counts = stats.get("flag_counts", {})
    if flag_counts:
        top_flags = list(flag_counts.items())[:8]
        flags_str = ", ".join(f"{f}({n})" for f, n in top_flags)
        print(f"\nFlags: {flags_str}")

    # Selection thresholds
    thresholds = stats.get("selection_thresholds", {})
    if thresholds:
        parts = []
        for k in ["top_10pct", "top_25pct", "top_50pct"]:
            t = thresholds.get(k, {})
            if t:
                parts.append(f"{k.replace('_', ' ')}≥{t.get('threshold', 0):.1f}({t.get('count', 0)})")
        if parts:
            print(f"Thresholds: {', '.join(parts)}")

    # Per-file summary in batch mode
    per_file = stats.get("per_file_summary", [])
    if is_batch and per_file:
        print(f"\nPer-file scores:")
        for pf in per_file:
            print(f"  {pf.get('file', '?'):30s}  n={pf.get('count', 0):4d}  "
                  f"val={pf.get('mean_value', 0):.2f}  "
                  f"qual={pf.get('mean_quality', 0):.2f}  "
                  f"sel={pf.get('mean_selection', 0):.2f}")

    print(f"\nRun dir: {run_dir}")


async def _run_scoring_file(input_path, output_dir, tag_stats_path, limit, config, resume=False,
                            llm_progress_cb=None):
    """Score a single labeled file."""
    input_path = Path(input_path)

    # Dispatch to chunked pipeline for JSONL files (memory-bounded)
    if input_path.suffix == ".jsonl":
        return await _run_scoring_file_chunked(
            input_path, output_dir, tag_stats_path, limit, config,
            llm_progress_cb=llm_progress_cb,
        )

    if output_dir is None:
        output_dir = input_path.parent
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load labeled data (only .json reaches here; .jsonl dispatched above)
    print(f"Loading labeled data from {input_path}...")
    with open(input_path, "r", encoding="utf-8") as f:
        samples = json.load(f)

    if limit > 0:
        samples = samples[:limit]
    total = len(samples)

    # Resume: load previously scored samples from scored.jsonl
    resumed_values = {}
    if resume:
        scored_jsonl_path = output_dir / "scored.jsonl"
        if scored_jsonl_path.exists():
            with open(scored_jsonl_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    scored_sample = json.loads(line)
                    sid = scored_sample.get("id", "")
                    v = scored_sample.get("value")
                    if sid and v:
                        resumed_values[sid] = v
            if resumed_values:
                print(f"  Resume: loaded {len(resumed_values)} pre-scored samples from scored.jsonl")

    # Load tag stats for rarity
    rarity_mode = resolve_rarity_mode(config)
    stats_ref_info = None
    idf_map = {}
    total_stats_samples = 0
    combo_counts = None
    combo_mode = "disabled"

    if tag_stats_path:
        stats_source = tag_stats_path
    else:
        # Auto-discover Pass 1 stats in same directory
        candidate = input_path.parent / PASS1_STATS_FILE
        stats_source = str(candidate) if candidate.exists() else None

    if stats_source:
        distributions, total_stats_samples, ts, meta = load_tag_stats_context(stats_source)
        if distributions:
            idf_map = compute_tag_idf(distributions, total_stats_samples)
            combo_counts = meta.get("combo_counts")
            stats_ref_info = {
                "source": str(stats_source),
                "total_samples": total_stats_samples,
                "timestamp": ts or datetime.now().isoformat(),
            }
            missing_dims = [
                d for d in (config.rarity_weights or RARITY_WEIGHTS).keys()
                if d not in idf_map
            ]
            if missing_dims:
                print("  Warning: rarity baseline missing dimensions; ignored in IDF: "
                      + ", ".join(missing_dims))
            if combo_counts:
                combo_mode = "external"
            else:
                combo_mode = "disabled"
                print("  External stats has no combo_distributions; "
                      "combo rarity disabled for cross-dataset comparability")
        else:
            print(f"  Warning: {stats_source} has no tag_distributions, fallback to local baseline")
    else:
        print("  No external tag stats found, fallback to local rarity baseline")

    if not idf_map and samples:
        local_distributions, local_total = build_tag_distributions(
            samples, rarity_weights=config.rarity_weights or RARITY_WEIGHTS
        )
        if local_distributions and local_total > 0:
            total_stats_samples = local_total
            idf_map = compute_tag_idf(local_distributions, total_stats_samples)
            stats_source = f"{input_path}#local"
            stats_ref_info = {
                "source": stats_source,
                "total_samples": total_stats_samples,
                "timestamp": datetime.now().isoformat(),
            }
            combo_counts = build_combo_counts(samples)
            combo_mode = "local"
            print(f"  Using local rarity baseline ({total_stats_samples} samples)")

    if not idf_map:
        print("  Warning: rarity unavailable (no valid tag distributions)")

    # Compute rarity for all samples
    rarity_results = []
    for s in samples:
        labels = s.get("labels") or {}
        rarity = compute_sample_rarity(
            labels, idf_map, total_stats_samples,
            rarity_weights=config.rarity_weights or RARITY_WEIGHTS,
            combo_alpha=config.rarity_combo_alpha,
            combo_counts=combo_counts,
            stats_ref_info=stats_ref_info,
        )
        rarity_results.append(rarity)

    # Normalize rarity to 1-10
    normalize_rarity_scores(
        rarity_results,
        mode=rarity_mode,
        total_samples=total_stats_samples,
    )

    # Run LLM scoring
    rate_limiter = AsyncRateLimiter(config.rps_limit, warmup=config.rps_warmup) if config.rps_limit > 0 else None
    sem = asyncio.Semaphore(config.scoring_concurrency)
    all_monitors = [None] * total
    all_values = [None] * total

    # Pre-fill resumed values
    skipped_count = 0
    if resumed_values:
        for idx, s in enumerate(samples):
            sid = s.get("id", "")
            if sid in resumed_values:
                all_values[idx] = resumed_values[sid]
                all_monitors[idx] = {"sample_id": sid, "status": "resumed",
                                     "llm_calls": 0, "prompt_tokens": 0,
                                     "completion_tokens": 0, "attempts": 0}
                skipped_count += 1

    scored_count = skipped_count
    failed_count = 0
    first_error_logged = False
    start_time = time.time()

    to_score = [i for i in range(total) if all_values[i] is None]
    _rps = f"rps={config.rps_limit}(warmup={config.rps_warmup}s)" if config.rps_limit > 0 else "rps=unlimited"
    print(f"  Scoring {len(to_score)} samples (skipped {skipped_count} resumed) "
          f"| model={config.scoring_model} concurrency={config.scoring_concurrency} {_rps}")

    async with httpx.AsyncClient(
        proxy=None,
        timeout=config.request_timeout,
        limits=httpx.Limits(
            max_connections=config.scoring_concurrency + 10,
            max_keepalive_connections=config.scoring_concurrency,
        ),
    ) as client:
        async def score_task(idx):
            nonlocal scored_count, failed_count, first_error_logged
            value, monitor = await score_one(
                client, samples[idx], config.scoring_model, rarity_results[idx],
                idx, total, sem, config=config, rate_limiter=rate_limiter,
            )
            all_values[idx] = value
            all_monitors[idx] = monitor
            if value:
                scored_count += 1
            else:
                failed_count += 1
                if not first_error_logged and monitor:
                    err = monitor.get("error", "unknown")
                    err_resp = monitor.get("error_response", "")
                    print(f"  [!] First failure: sample={monitor.get('sample_id')} "
                          f"attempts={monitor.get('attempts')} err={err[:200]}")
                    if err_resp and err_resp != err:
                        print(f"      response={err_resp[:200]}")
                    first_error_logged = True
            return idx

        with _create_progress() as progress:
            task = progress.add_task("Pass 2", total=len(to_score), info="")

            tasks = [asyncio.create_task(score_task(i)) for i in to_score]
            for coro in asyncio.as_completed(tasks):
                idx = await coro
                _info = format_progress_info(
                    ok_count=scored_count,
                    fail_count=failed_count,
                    request_stats=rate_limiter.stats if rate_limiter else None,
                )
                monitor = all_monitors[idx]
                if llm_progress_cb and monitor:
                    run_info = llm_progress_cb(monitor.get("llm_calls", 0), "pass2")
                    if run_info:
                        _info = f"{_info} • {run_info}"
                progress.update(task, advance=1, info=_info)

    elapsed = time.time() - start_time

    # Attach value to samples
    for i, s in enumerate(samples):
        if all_values[i]:
            s["value"] = all_values[i]

    # Compute selection scores (intra-class quality ranking, no LLM)
    compute_selection_scores(samples, config=config)

    # Write outputs (atomic: write to .tmp then rename)
    scored_path = output_dir / "scored.json"
    tmp_scored = scored_path.with_suffix(".tmp.json")
    with open(tmp_scored, "w", encoding="utf-8") as f:
        json.dump(samples, f, ensure_ascii=False, indent=2)
    os.replace(tmp_scored, scored_path)

    scored_jsonl_path = output_dir / "scored.jsonl"
    tmp_jsonl = scored_jsonl_path.with_suffix(".tmp.jsonl")
    with open(tmp_jsonl, "w", encoding="utf-8") as f:
        for s in samples:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")
    os.replace(tmp_jsonl, scored_jsonl_path)

    # Monitor
    monitor_path = output_dir / "monitor_value.jsonl"
    with open(monitor_path, "w", encoding="utf-8") as f:
        for m in all_monitors:
            if m:
                f.write(json.dumps(m, ensure_ascii=False) + "\n")

    # Failed samples
    failed_samples = [s for i, s in enumerate(samples) if all_values[i] is None]
    if failed_samples:
        failed_path = output_dir / "failed_value.jsonl"
        with open(failed_path, "w", encoding="utf-8") as f:
            for s in failed_samples:
                f.write(json.dumps(s, ensure_ascii=False) + "\n")

    # Failure log (aligned with Pass 1 format)
    failed_indices = [i for i, v in enumerate(all_values) if v is None]
    if failed_indices:
        with open(output_dir / "score_failures.jsonl", "w", encoding="utf-8") as f:
            for i in failed_indices:
                m = all_monitors[i]
                record = {
                    "sample_id": samples[i].get("id", f"sample-{i}"),
                    "status": m["status"] if m else "no_result",
                    "error": (m.get("error", "") if m else "no monitor record"),
                    "error_response": (m.get("error_response", "")[:1000] if m else ""),
                    "attempts": (m.get("attempts", 0) if m else 0),
                }
                f.write(json.dumps(record, ensure_ascii=False) + "\n")

    # Stats
    stats = compute_value_stats(samples, [m for m in all_monitors if m])
    stats["elapsed_seconds"] = round(elapsed, 1)
    stats["model"] = config.scoring_model
    stats["input_file"] = str(input_path)
    if rate_limiter:
        stats["http_request_stats"] = rate_limiter.stats.to_dict()
    stats["weights_used"] = config.value_weights or VALUE_WEIGHTS
    stats["rarity_config"] = {
        "stats_ref": str(stats_source) if stats_source else None,
        "total_samples_in_distribution": total_stats_samples,
        "dimension_weights": config.rarity_weights or RARITY_WEIGHTS,
        "combo_alpha": config.rarity_combo_alpha,
        "combo_mode": combo_mode,
        "score_mode": rarity_mode,
    }

    stats_path = output_dir / PASS2_STATS_FILE
    stats_to_write = {k: v for k, v in stats.items() if k != "_raw_scores"}
    tmp_stats = stats_path.with_suffix(".tmp.json")
    with open(tmp_stats, "w", encoding="utf-8") as f:
        json.dump(stats_to_write, f, ensure_ascii=False, indent=2)
    os.replace(tmp_stats, stats_path)

    # Conversation-level aggregation
    try:
        from sft_label.conversation import aggregate_conversations, write_conversation_scores
        conv_records = aggregate_conversations(samples)
        if conv_records:
            write_conversation_scores(conv_records, output_dir / "conversation_scores.json")
    except Exception as e:
        print(f"  Warning: conversation aggregation failed: {e}")

    # Dashboard
    try:
        from sft_label.tools.visualize_value import generate_value_dashboard
        generate_value_dashboard(output_dir, scored_file="scored.json",
                                 stats_file=PASS2_STATS_FILE,
                                 output_file=PASS2_DASHBOARD_FILE)
    except Exception as e:
        print(f"  Warning: dashboard generation failed: {e}")

    stats["elapsed_seconds"] = round(elapsed, 1)
    if rate_limiter:
        stats["http_request_stats"] = rate_limiter.stats.to_dict()
    print_scoring_summary(stats, output_dir)

    return stats


@dataclass
class _ScoringFileCollector:
    """Track per-file scoring results for cross-file scoring pipeline."""
    file_idx: int
    labeled_path: Path
    output_dir: Path
    samples: list
    rarity_results: list
    total: int
    stats_source: str = None
    idf_map: dict = field(default_factory=dict)
    total_stats_samples: int = 0
    stats_ref_info: dict = None
    combo_mode: str = "disabled"
    done: int = 0
    ok: int = 0
    fail: int = 0
    values: list = field(default_factory=list)
    monitors: list = field(default_factory=list)
    completed: bool = False

    def __post_init__(self):
        self.values = [None] * self.total
        self.monitors = [None] * self.total


@dataclass
class ScoringDirectoryWorkloadEstimate:
    """Pre-run workload estimate for directory scoring."""
    files_planned: int
    total_samples: int
    baseline_total_llm_calls: int
    initial_estimated_llm_calls: int
    scan_elapsed_seconds: float


def _count_scoring_samples_in_file(labeled_path, limit=0):
    """Count samples in a labeled file with per-file limit applied."""
    if labeled_path.suffix == ".jsonl":
        count = 0
        with open(labeled_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    count += 1
                    if limit > 0 and count >= limit:
                        break
        return count

    with open(labeled_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, list):
        total = len(data)
    else:
        total = 1 if data else 0
    if limit > 0:
        total = min(total, limit)
    return total


def estimate_scoring_directory_workload(labeled_files, *, limit=0, config=None):
    """Estimate scoring workload before directory-mode execution."""
    start = time.time()
    total_samples = 0
    for labeled_path in labeled_files:
        total_samples += _count_scoring_samples_in_file(labeled_path, limit=limit)

    baseline_calls = total_samples
    # Pass 2 has 1 guaranteed scoring call per sample; retries may increase calls.
    sample_retries = max(getattr(config, "sample_max_retries", SAMPLE_MAX_RETRIES), 1)
    retry_factor = 1.0 + min(0.2, 0.05 * (sample_retries - 1))
    initial_est_calls = int(round(total_samples * retry_factor))
    initial_est_calls = max(initial_est_calls, baseline_calls)

    return ScoringDirectoryWorkloadEstimate(
        files_planned=len(labeled_files),
        total_samples=total_samples,
        baseline_total_llm_calls=baseline_calls,
        initial_estimated_llm_calls=initial_est_calls,
        scan_elapsed_seconds=round(time.time() - start, 2),
    )


def _discover_scored_output_files(root_dir):
    """Find logical scored outputs in a run tree, preferring JSON over JSONL."""
    root_dir = Path(root_dir)
    candidates = []
    for pattern in ("scored*.json", "scored*.jsonl", "**/scored*.json", "**/scored*.jsonl"):
        candidates.extend(root_dir.glob(pattern))

    preferred = {}
    for path in sorted(set(candidates)):
        key = (path.parent, path.stem)
        existing = preferred.get(key)
        if existing is None:
            preferred[key] = path
            continue
        if existing.suffix == ".jsonl" and path.suffix == ".json":
            preferred[key] = path

    return sorted(preferred.values())


def _load_scored_samples(path):
    """Load scored samples from .json or .jsonl."""
    path = Path(path)
    if path.suffix == ".jsonl":
        samples = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    samples.append(json.loads(line))
        return samples

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data if isinstance(data, list) else []


def _write_scored_samples(primary_path, samples):
    """Write scored outputs back to disk, keeping JSON/JSONL siblings in sync."""
    primary_path = Path(primary_path)
    dir_path = primary_path.parent
    json_path = dir_path / "scored.json"
    jsonl_path = dir_path / "scored.jsonl"

    if json_path.exists() or primary_path.suffix == ".json":
        tmp_json = json_path.with_suffix(".tmp.json")
        with open(tmp_json, "w", encoding="utf-8") as f:
            json.dump(samples, f, ensure_ascii=False, indent=2)
        os.replace(tmp_json, json_path)

    if jsonl_path.exists() or primary_path.suffix == ".jsonl":
        tmp_jsonl = jsonl_path.with_suffix(".tmp.jsonl")
        with open(tmp_jsonl, "w", encoding="utf-8") as f:
            for sample in samples:
                f.write(json.dumps(sample, ensure_ascii=False) + "\n")
        os.replace(tmp_jsonl, jsonl_path)


def _load_monitor_records(dir_path):
    """Load scoring monitors for per-file stats recomputation."""
    monitor_path = Path(dir_path) / "monitor_value.jsonl"
    if not monitor_path.exists():
        return []

    monitors = []
    with open(monitor_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                monitors.append(json.loads(line))
    return monitors


def _load_existing_pass2_stats(dir_path):
    """Load existing per-file Pass 2 stats if present."""
    stats_path = Path(dir_path) / PASS2_STATS_FILE
    if not stats_path.exists():
        return {}
    with open(stats_path, "r", encoding="utf-8") as f:
        return json.load(f)


def _rewrite_directory_global_selection(output_dir, input_dir, config, pprint=print):
    """Recompute selection globally across directory outputs and rewrite files."""
    scored_files = _discover_scored_output_files(output_dir)
    if not scored_files:
        return []

    pprint(f"  Recomputing global selection across {len(scored_files)} scored file(s)")

    summaries = []
    file_entries = []
    for scored_path in scored_files:
        samples = _load_scored_samples(scored_path)
        scored_count = 0
        for sample in samples:
            if sample.get("value"):
                summaries.append(_selection_summary_from_sample(sample))
                scored_count += 1
        file_entries.append({
            "path": scored_path,
            "scored_count": scored_count,
        })

    selection_results = compute_selection_scores_from_summaries(summaries, config=config)

    updated_stats = []
    cursor = 0
    for entry in file_entries:
        scored_path = entry["path"]
        samples = _load_scored_samples(scored_path)
        for sample in samples:
            value = sample.get("value")
            if not value:
                continue
            if cursor >= len(selection_results):
                break
            selection = selection_results[cursor]
            value["selection_score"] = selection["selection_score"]
            value["intra_class_rank"] = selection["intra_class_rank"]
            cursor += 1

        _write_scored_samples(scored_path, samples)

        monitors = _load_monitor_records(scored_path.parent)
        stats = compute_value_stats(samples, monitors)
        existing_stats = _load_existing_pass2_stats(scored_path.parent)
        for key in (
            "elapsed_seconds",
            "model",
            "input_file",
            "http_request_stats",
            "weights_used",
            "rarity_config",
            "chunked",
        ):
            if key in existing_stats:
                stats[key] = existing_stats[key]

        stats_path = scored_path.parent / PASS2_STATS_FILE
        tmp_stats = stats_path.with_suffix(".tmp.json")
        with open(tmp_stats, "w", encoding="utf-8") as f:
            json.dump({k: v for k, v in stats.items() if k != "_raw_scores"},
                      f, ensure_ascii=False, indent=2)
        os.replace(tmp_stats, stats_path)

        try:
            from sft_label.tools.visualize_value import generate_value_dashboard
            generate_value_dashboard(
                scored_path.parent,
                scored_file="scored.json" if (scored_path.parent / "scored.json").exists() else "scored.jsonl",
                stats_file=PASS2_STATS_FILE,
                output_file=PASS2_DASHBOARD_FILE,
            )
        except Exception:
            pass

        input_file = existing_stats.get("input_file")
        file_label_source = Path(input_file) if input_file else scored_path
        stats["file"] = _relative_file_label(file_label_source, input_dir)
        updated_stats.append(stats)

    return updated_stats


def _flush_scoring_file(collector, config, pprint=print):
    """Write all scoring outputs for a completed file and release memory.

    Returns the stats dict.
    """
    samples = collector.samples
    all_values = collector.values
    all_monitors = collector.monitors
    output_dir = collector.output_dir
    total = collector.total

    # Attach value to samples
    for i, s in enumerate(samples):
        if all_values[i]:
            s["value"] = all_values[i]

    # Compute selection scores (intra-class quality ranking, no LLM)
    compute_selection_scores(samples, config=config)

    # Write outputs (atomic: write to .tmp then rename)
    output_dir.mkdir(parents=True, exist_ok=True)

    scored_path = output_dir / "scored.json"
    tmp_scored = scored_path.with_suffix(".tmp.json")
    with open(tmp_scored, "w", encoding="utf-8") as f:
        json.dump(samples, f, ensure_ascii=False, indent=2)
    os.replace(tmp_scored, scored_path)

    scored_jsonl_path = output_dir / "scored.jsonl"
    tmp_jsonl = scored_jsonl_path.with_suffix(".tmp.jsonl")
    with open(tmp_jsonl, "w", encoding="utf-8") as f:
        for s in samples:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")
    os.replace(tmp_jsonl, scored_jsonl_path)

    monitor_path = output_dir / "monitor_value.jsonl"
    with open(monitor_path, "w", encoding="utf-8") as f:
        for m in all_monitors:
            if m:
                f.write(json.dumps(m, ensure_ascii=False) + "\n")

    failed_samples = [s for i, s in enumerate(samples) if all_values[i] is None]
    if failed_samples:
        failed_path = output_dir / "failed_value.jsonl"
        with open(failed_path, "w", encoding="utf-8") as f:
            for s in failed_samples:
                f.write(json.dumps(s, ensure_ascii=False) + "\n")

    # Failure log (aligned with Pass 1 format)
    failed_indices = [i for i, v in enumerate(all_values) if v is None]
    if failed_indices:
        with open(output_dir / "score_failures.jsonl", "w", encoding="utf-8") as f:
            for i in failed_indices:
                m = all_monitors[i]
                record = {
                    "sample_id": samples[i].get("id", f"sample-{i}"),
                    "source_file": str(collector.labeled_path),
                    "status": m["status"] if m else "no_result",
                    "error": (m.get("error", "") if m else "no monitor record"),
                    "error_response": (m.get("error_response", "")[:1000] if m else ""),
                    "attempts": (m.get("attempts", 0) if m else 0),
                }
                f.write(json.dumps(record, ensure_ascii=False) + "\n")

    valid_monitors = [m for m in all_monitors if m]
    stats = compute_value_stats(samples, valid_monitors)
    stats["model"] = config.scoring_model
    stats["input_file"] = str(collector.labeled_path)
    stats["weights_used"] = config.value_weights or VALUE_WEIGHTS
    stats["rarity_config"] = {
        "stats_ref": collector.stats_source,
        "total_samples_in_distribution": collector.total_stats_samples,
        "dimension_weights": config.rarity_weights or RARITY_WEIGHTS,
        "combo_alpha": config.rarity_combo_alpha,
        "combo_mode": collector.combo_mode,
        "score_mode": resolve_rarity_mode(config),
    }

    stats_path = output_dir / PASS2_STATS_FILE
    stats_to_write = {k: v for k, v in stats.items() if k != "_raw_scores"}
    tmp_stats = stats_path.with_suffix(".tmp.json")
    with open(tmp_stats, "w", encoding="utf-8") as f:
        json.dump(stats_to_write, f, ensure_ascii=False, indent=2)
    os.replace(tmp_stats, stats_path)

    # Conversation-level aggregation
    try:
        from sft_label.conversation import aggregate_conversations, write_conversation_scores
        conv_records = aggregate_conversations(samples)
        if conv_records:
            write_conversation_scores(conv_records, output_dir / "conversation_scores.json")
    except Exception:
        pass

    try:
        from sft_label.tools.visualize_value import generate_value_dashboard
        generate_value_dashboard(output_dir, scored_file="scored.json",
                                 stats_file=PASS2_STATS_FILE,
                                 output_file=PASS2_DASHBOARD_FILE,
                                 quiet=True)
    except Exception:
        pass

    # Release memory
    collector.samples = None
    collector.values = None
    collector.monitors = None
    collector.rarity_results = None
    collector.completed = True

    return stats


async def _run_scoring_directory(input_dir, output_dir, tag_stats_path, limit, config,
                                 llm_progress_cb=None):
    """Score all labeled files in a directory with cross-file parallelism.

    Uses watermark-based file loading (same pattern as Pass 1's
    run_directory_pipeline) to keep the semaphore saturated across files.
    """
    input_dir = Path(input_dir)
    if output_dir is None:
        output_dir = input_dir
    output_dir = Path(output_dir)

    # Find labeled files (flat, one-level subdirs, or deeply nested)
    # Try progressively deeper searches; include both .json and .jsonl
    labeled_files = sorted(input_dir.glob("labeled*.json")) + sorted(input_dir.glob("labeled*.jsonl"))
    if not labeled_files:
        labeled_files = sorted(input_dir.glob("*/labeled*.json")) + sorted(input_dir.glob("*/labeled*.jsonl"))
    if not labeled_files:
        # Deep nesting: recursive glob
        labeled_files = sorted(input_dir.glob("**/labeled*.json")) + sorted(input_dir.glob("**/labeled*.jsonl"))
    # Separate .json and .jsonl: prefer .json if both exist in same dir
    json_files = [f for f in labeled_files if not f.name.endswith(".jsonl")]
    jsonl_files = [f for f in labeled_files if f.name.endswith(".jsonl")]
    # For dirs that only have .jsonl (chunked pipeline output), include them
    json_dirs = {f.parent for f in json_files}
    for jl in jsonl_files:
        if jl.parent not in json_dirs:
            json_files.append(jl)
    labeled_files = sorted(json_files)

    if not labeled_files:
        print(f"No labeled*.json/jsonl files found in {input_dir}")
        return {}

    rarity_mode = resolve_rarity_mode(config)

    print(f"Found {len(labeled_files)} labeled files in {input_dir}")
    workload_estimate = estimate_scoring_directory_workload(
        labeled_files,
        limit=limit,
        config=config,
    )
    print(
        "  Plan | "
        f"{workload_estimate.files_planned} files, "
        f"{workload_estimate.total_samples} samples, "
        f"llm~{workload_estimate.initial_estimated_llm_calls}, "
        f"scan {workload_estimate.scan_elapsed_seconds:.1f}s"
    )

    # Use Pass 1 summary stats for rarity if no explicit stats provided
    if tag_stats_path is None:
        candidate = input_dir / PASS1_SUMMARY_STATS_FILE
        if candidate.exists():
            tag_stats_path = str(candidate)
        else:
            # Try per-file stats (flat, one-level, or recursive)
            stats_files = sorted(input_dir.glob("stats_labeling*.json"))
            if not stats_files:
                stats_files = sorted(input_dir.glob("*/stats_labeling*.json"))
            if not stats_files:
                stats_files = sorted(input_dir.glob("**/stats_labeling*.json"))
            if stats_files:
                tag_stats_path = str(stats_files[0])

    # Load global tag stats once (shared across all files)
    global_idf_map = {}
    global_total_stats_samples = 0
    global_stats_ref_info = None
    global_stats_source = str(tag_stats_path) if tag_stats_path else None
    global_combo_counts = None
    global_combo_mode = "disabled"

    def _scan_local_distributions():
        dims = list((config.rarity_weights or RARITY_WEIGHTS).keys())
        distributions = {dim: {} for dim in dims}
        total = 0
        combo_counts = {}
        for labeled_path in labeled_files:
            if labeled_path.suffix == ".jsonl":
                with open(labeled_path, "r", encoding="utf-8") as f:
                    for i, line in enumerate(f):
                        if limit > 0 and i >= limit:
                            break
                        line = line.strip()
                        if not line:
                            continue
                        sample = json.loads(line)
                        labels = sample.get("labels") or {}
                        if _update_distributions_from_labels(distributions, labels, dims):
                            total += 1
                        _update_combo_counts_from_labels(combo_counts, labels)
            else:
                with open(labeled_path, "r", encoding="utf-8") as f:
                    samples = json.load(f)
                if limit > 0:
                    samples = samples[:limit]
                for sample in samples:
                    labels = sample.get("labels") or {}
                    if _update_distributions_from_labels(distributions, labels, dims):
                        total += 1
                    _update_combo_counts_from_labels(combo_counts, labels)
        distributions = {d: c for d, c in distributions.items() if c}
        return distributions, total, combo_counts

    local_distributions = None

    if tag_stats_path:
        distributions, global_total_stats_samples, ts, meta = load_tag_stats_context(tag_stats_path)
        if distributions:
            global_idf_map = compute_tag_idf(distributions, global_total_stats_samples)
            global_combo_counts = meta.get("combo_counts")
            global_stats_ref_info = {
                "source": str(tag_stats_path),
                "total_samples": global_total_stats_samples,
                "timestamp": ts or datetime.now().isoformat(),
            }
            missing_dims = [
                d for d in (config.rarity_weights or RARITY_WEIGHTS).keys()
                if d not in global_idf_map
            ]
            if missing_dims:
                print("  Warning: rarity baseline missing dimensions; ignored in IDF: "
                      + ", ".join(missing_dims))
            if global_combo_counts:
                global_combo_mode = "external"
            else:
                global_combo_mode = "disabled"
                print("  External stats has no combo_distributions; "
                      "combo rarity disabled for cross-dataset comparability")
        else:
            print(f"  Warning: {tag_stats_path} has no tag_distributions, fallback to local baseline")

    if not global_idf_map:
        local_distributions, local_total, local_combo_counts = _scan_local_distributions()
        if local_distributions and local_total > 0:
            global_total_stats_samples = local_total
            global_idf_map = compute_tag_idf(local_distributions, local_total)
            global_combo_counts = local_combo_counts
            global_combo_mode = "local"
            global_stats_source = f"{input_dir}#local"
            global_stats_ref_info = {
                "source": global_stats_source,
                "total_samples": global_total_stats_samples,
                "timestamp": datetime.now().isoformat(),
            }
            print(f"  Using local rarity baseline ({local_total} samples)")
        else:
            print("  Warning: rarity unavailable (no valid tag distributions)")

    shared_tag_stats_path = str(tag_stats_path) if tag_stats_path else None
    temp_tag_stats_path = None
    if shared_tag_stats_path is None and local_distributions and global_total_stats_samples > 0:
        temp_tag_stats_path = output_dir / ".directory_rarity_stats.json"
        with open(temp_tag_stats_path, "w", encoding="utf-8") as f:
            json.dump({
                "tag_distributions": local_distributions,
                "combo_distributions": global_combo_counts or {},
                "total_samples": global_total_stats_samples,
                "distribution_total_samples": global_total_stats_samples,
                "timestamp": datetime.now().isoformat(),
            }, f, ensure_ascii=False, indent=2)
        shared_tag_stats_path = str(temp_tag_stats_path)

    streaming_files = [p for p in labeled_files if p.suffix == ".jsonl"]
    resident_files = [p for p in labeled_files if p.suffix != ".jsonl"]

    concurrency = config.scoring_concurrency
    watermark = int(concurrency * config.dir_pipeline_watermark)
    max_active = config.dir_pipeline_max_files
    rate_limiter = AsyncRateLimiter(config.rps_limit, warmup=config.rps_warmup) if config.rps_limit > 0 else None
    sem = asyncio.Semaphore(concurrency)
    pprint = print

    _rps = f"rps={config.rps_limit}(warmup={config.rps_warmup}s)" if config.rps_limit > 0 else "rps=unlimited"
    print(f"  Scoring: model={config.scoring_model} concurrency={concurrency} {_rps}")

    batch_start = time.time()

    # --- Helper: wrap score_one with file/sample tracking ---
    async def _tagged_score(client, sample, rarity_result, file_idx, sample_idx,
                            total_in_file):
        value, monitor = await score_one(
            client, sample, config.scoring_model, rarity_result,
            sample_idx, total_in_file, sem, config=config, rate_limiter=rate_limiter,
        )
        return file_idx, sample_idx, value, monitor

    # --- Load a file and submit its tasks ---
    def load_and_submit(file_idx, labeled_path, client, pending_futures):
        # Load samples
        with open(labeled_path, "r", encoding="utf-8") as f:
            if labeled_path.suffix == ".jsonl":
                samples = [json.loads(line) for line in f if line.strip()]
            else:
                samples = json.load(f)
        if limit > 0:
            samples = samples[:limit]
        total = len(samples)

        # Per-file rarity: always use one shared combo baseline for consistency
        combo_counts = global_combo_counts if global_idf_map else None
        rarity_results = []
        for s in samples:
            labels = s.get("labels") or {}
            rarity = compute_sample_rarity(
                labels, global_idf_map, global_total_stats_samples,
                rarity_weights=config.rarity_weights or RARITY_WEIGHTS,
                combo_alpha=config.rarity_combo_alpha,
                combo_counts=combo_counts,
                stats_ref_info=global_stats_ref_info,
            )
            rarity_results.append(rarity)
        normalize_rarity_scores(
            rarity_results,
            mode=rarity_mode,
            total_samples=global_total_stats_samples,
        )

        file_output_dir = labeled_path.parent
        collector = _ScoringFileCollector(
            file_idx=file_idx,
            labeled_path=labeled_path,
            output_dir=file_output_dir,
            samples=samples,
            rarity_results=rarity_results,
            total=total,
            stats_source=global_stats_source,
            idf_map=global_idf_map,
            total_stats_samples=global_total_stats_samples,
            stats_ref_info=global_stats_ref_info,
            combo_mode=global_combo_mode,
        )

        # Submit scoring tasks
        for i in range(total):
            fut = asyncio.ensure_future(
                _tagged_score(client, samples[i], rarity_results[i],
                              file_idx, i, total)
            )
            pending_futures.add(fut)

        return collector

    # --- Try to load more files if below watermark ---
    def maybe_load_more(pending_futures, collectors, next_to_load, client):
        active_count = sum(1 for c in collectors.values() if not c.completed)
        while (next_to_load < len(resident_files)
               and len(pending_futures) < watermark
               and active_count < max_active):
            labeled_path = resident_files[next_to_load]
            c = load_and_submit(next_to_load, labeled_path, client, pending_futures)
            collectors[c.file_idx] = c
            next_to_load += 1
            active_count += 1
        return next_to_load

    # --- Main watermark-driven loop ---
    collectors = {}
    pending_futures = set()
    all_file_stats = []
    next_to_load = 0
    first_error_logged = False
    eta_tracker = RuntimeEtaEstimator(
        total_labeled_samples=workload_estimate.total_samples,
        initial_estimated_calls=workload_estimate.initial_estimated_llm_calls,
    )

    try:
        for labeled_path in streaming_files:
            print(f"  Streaming JSONL scoring for {labeled_path.relative_to(input_dir)}")
            stats = await _run_scoring_file_chunked(
                labeled_path,
                labeled_path.parent,
                shared_tag_stats_path,
                limit,
                config,
                llm_progress_cb=llm_progress_cb,
            )
            stats["file"] = _relative_file_label(labeled_path, input_dir)
            all_file_stats.append(stats)

        if resident_files:
            async with httpx.AsyncClient(
                proxy=None,
                timeout=config.request_timeout,
                limits=httpx.Limits(
                    max_connections=concurrency + 10,
                    max_keepalive_connections=concurrency,
                ),
            ) as client:
                with _create_progress() as progress:
                    global_llm_info = None
                    file_task = progress.add_task("Files", total=len(resident_files), info="")
                    sample_task = progress.add_task(
                        "Pass 2",
                        total=sum(_count_scoring_samples_in_file(p, limit=limit) for p in resident_files),
                        visible=bool(resident_files),
                        info="starting...",
                    )
                    llm_task = progress.add_task(
                        "LLM",
                        total=max(eta_tracker.estimated_total_calls, 1),
                        visible=bool(resident_files),
                        info=eta_tracker.info_line(),
                    )
                    pprint = progress.console.print

                    # Initial load
                    next_to_load = maybe_load_more(pending_futures, collectors, next_to_load, client)

                    while pending_futures:
                        done, pending_futures = await asyncio.wait(
                            pending_futures, return_when=asyncio.FIRST_COMPLETED)

                        for fut in done:
                            file_idx, sample_idx, value, monitor = fut.result()
                            c = collectors[file_idx]

                            if 0 <= sample_idx < c.total:
                                c.values[sample_idx] = value
                                c.monitors[sample_idx] = monitor

                            c.done += 1
                            if value:
                                c.ok += 1
                            else:
                                c.fail += 1
                                if not first_error_logged and monitor:
                                    err = monitor.get("error", "unknown")
                                    err_resp = monitor.get("error_response", "")
                                    pprint(f"  [!] First failure: {monitor.get('sample_id')} err={err[:200]}")
                                    if err_resp and err_resp != err:
                                        pprint(f"      response={err_resp[:200]}")
                                    first_error_logged = True

                            total_ok = sum(cc.ok for cc in collectors.values())
                            total_fail = sum(cc.fail for cc in collectors.values())
                            _info = format_progress_info(
                                ok_count=total_ok,
                                fail_count=total_fail,
                                label=c.labeled_path.name,
                                request_stats=rate_limiter.stats if rate_limiter else None,
                            )
                            if llm_progress_cb and monitor:
                                global_llm_info = llm_progress_cb(monitor.get("llm_calls", 0), "pass2")
                            progress.update(sample_task, advance=1, info=_info)
                            eta_tracker.update(monitor.get("llm_calls", 0) if monitor else 0)
                            global_counts = parse_run_progress(global_llm_info) if global_llm_info else None
                            if global_counts:
                                g_done, g_total = global_counts
                                progress.update(
                                    llm_task,
                                    total=max(g_total, 1),
                                    completed=min(g_done, g_total),
                                    info=global_llm_info,
                                )
                            else:
                                progress.update(
                                    llm_task,
                                    total=max(eta_tracker.estimated_total_calls, eta_tracker.calls_done, 1),
                                    completed=eta_tracker.calls_done,
                                    info=eta_tracker.info_line(),
                                )

                            # Check if file is fully done
                            if c.done >= c.total and not c.completed:
                                stats = _flush_scoring_file(c, config, pprint=pprint)
                                stats["file"] = _relative_file_label(c.labeled_path, input_dir)
                                all_file_stats.append(stats)
                                progress.update(file_task, advance=1)

                        # After processing batch, check if we should load more files
                        next_to_load = maybe_load_more(pending_futures, collectors, next_to_load, client)

                        active_names = [c.labeled_path.name for c in collectors.values()
                                        if not c.completed]
                        progress.update(file_task, info=", ".join(active_names)[:60] if active_names else "done")
    finally:
        if temp_tag_stats_path and temp_tag_stats_path.exists():
            temp_tag_stats_path.unlink()

    elapsed = time.time() - batch_start

    if all_file_stats:
        rewritten_file_stats = _rewrite_directory_global_selection(
            output_dir=output_dir,
            input_dir=input_dir,
            config=config,
            pprint=pprint,
        )
        if rewritten_file_stats:
            all_file_stats = rewritten_file_stats

    # Write global summary
    if all_file_stats:
        summary = _merge_value_stats(all_file_stats)
        summary["elapsed_seconds"] = round(elapsed, 1)
        summary["model"] = config.scoring_model
        summary["input_path"] = str(input_dir)
        summary["files_processed"] = len(all_file_stats)
        summary["planned_files"] = workload_estimate.files_planned
        summary["planned_samples"] = workload_estimate.total_samples
        summary["planned_baseline_llm_calls"] = workload_estimate.baseline_total_llm_calls
        summary["planned_initial_llm_calls"] = workload_estimate.initial_estimated_llm_calls
        summary["planning_elapsed_seconds"] = workload_estimate.scan_elapsed_seconds
        summary["rarity_config"] = {
            "stats_ref": global_stats_source,
            "total_samples_in_distribution": global_total_stats_samples,
            "dimension_weights": config.rarity_weights or RARITY_WEIGHTS,
            "combo_alpha": config.rarity_combo_alpha,
            "combo_mode": global_combo_mode,
            "score_mode": rarity_mode,
        }
        if rate_limiter:
            summary["http_request_stats"] = rate_limiter.stats.to_dict()

        summary_path = output_dir / PASS2_SUMMARY_STATS_FILE
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)

        # Global conversation-level aggregation across all files
        try:
            from sft_label.conversation import aggregate_conversations, write_conversation_scores
            all_conv_samples = []
            for sub in sorted(output_dir.iterdir()):
                if not sub.is_dir():
                    continue
                for pattern in ("scored.json", "scored.jsonl"):
                    p = sub / pattern
                    if p.exists():
                        if p.suffix == ".jsonl":
                            with open(p, encoding="utf-8") as f:
                                for line in f:
                                    line = line.strip()
                                    if line:
                                        all_conv_samples.append(json.loads(line))
                        else:
                            with open(p, encoding="utf-8") as f:
                                data = json.load(f)
                            if isinstance(data, list):
                                all_conv_samples.extend(data)
                        break  # prefer .json, skip .jsonl if .json found
            conv_records = aggregate_conversations(all_conv_samples)
            if conv_records:
                write_conversation_scores(conv_records, output_dir / "conversation_scores.json")
            del all_conv_samples
        except Exception as e:
            print(f"  Warning: global conversation aggregation failed: {e}")

        # Global dashboard
        try:
            from sft_label.tools.visualize_value import generate_value_dashboard
            dir_name = input_dir.name
            generate_value_dashboard(output_dir, scored_file=None,
                                     stats_file=PASS2_SUMMARY_STATS_FILE,
                                     output_file=pass2_global_dashboard_filename(dir_name),
                                     quiet=True)
        except Exception as e:
            print(f"  Warning: global dashboard generation failed: {e}")

        print_scoring_summary(summary, output_dir, is_batch=True)

    return summary if all_file_stats else {}


def _merge_value_stats(file_stats_list):
    """Merge per-file value stats into a global summary."""
    merged = {
        "total_scored": sum(s.get("total_scored", 0) for s in file_stats_list),
        "total_failed": sum(s.get("total_failed", 0) for s in file_stats_list),
        "total_llm_calls": sum(s.get("total_llm_calls", 0) for s in file_stats_list),
        "total_prompt_tokens": sum(s.get("total_prompt_tokens", 0) for s in file_stats_list),
        "total_completion_tokens": sum(s.get("total_completion_tokens", 0) for s in file_stats_list),
        "total_tokens": sum(s.get("total_tokens", 0) for s in file_stats_list),
    }

    # Per-file summary
    merged["per_file_summary"] = [
        {
            "file": s.get("file", "unknown"),
            "count": s.get("total_scored", 0),
            "mean_value": s.get("score_distributions", {}).get("value_score", {}).get("mean", 0),
            "mean_complexity": s.get("score_distributions", {}).get("complexity_overall", {}).get("mean", 0),
            "mean_quality": s.get("score_distributions", {}).get("quality_overall", {}).get("mean", 0),
            "mean_rarity": s.get("score_distributions", {}).get("rarity_score", {}).get("mean", 0),
            "mean_selection": s.get("score_distributions", {}).get("selection_score", {}).get("mean", 0),
        }
        for s in file_stats_list
    ]

    # Merge score distributions: use raw scores when available for exact percentiles
    all_dist_keys = set()
    for s in file_stats_list:
        all_dist_keys.update(s.get("score_distributions", {}).keys())
    merged_distributions = {}
    for key in all_dist_keys:
        # Collect raw values from all files (attached by compute_value_stats)
        all_values = []
        for s in file_stats_list:
            raw = s.get("_raw_scores", {}).get(key)
            if raw:
                all_values.extend(raw)

        if all_values:
            # Exact percentiles from raw data
            merged_distributions[key] = _percentiles(all_values)
        else:
            # Fallback: weighted mean + global min/max (no raw data available)
            total_n = 0
            total_sum = 0.0
            global_min = float("inf")
            global_max = float("-inf")
            for s in file_stats_list:
                dist = s.get("score_distributions", {}).get(key, {})
                n = s.get("total_scored", 0)
                if not dist or n == 0 or not isinstance(dist, dict):
                    continue
                mean = dist.get("mean", 0)
                total_sum += mean * n
                total_n += n
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

    # Merge histogram bins: sum across files
    merged_histograms = {}
    for s in file_stats_list:
        for key, bins in s.get("histograms", {}).items():
            if key not in merged_histograms:
                merged_histograms[key] = [0] * 10
            for i, v in enumerate(bins):
                merged_histograms[key][i] += v
    merged["histograms"] = merged_histograms

    # Merge thinking mode stats: sum counts, weighted means
    merged_thinking = {"slow": {"count": 0, "sum_value": 0.0, "sum_quality": 0.0, "sum_reasoning": 0.0},
                       "fast": {"count": 0, "sum_value": 0.0, "sum_quality": 0.0, "sum_reasoning": 0.0}}
    for s in file_stats_list:
        ts = s.get("thinking_mode_stats", {})
        for mode in ("slow", "fast"):
            ms = ts.get(mode, {})
            n = ms.get("count", 0)
            merged_thinking[mode]["count"] += n
            merged_thinking[mode]["sum_value"] += ms.get("mean_value", 0) * n
            merged_thinking[mode]["sum_quality"] += ms.get("mean_quality", 0) * n
            merged_thinking[mode]["sum_reasoning"] += ms.get("mean_reasoning", 0) * n
    merged["thinking_mode_stats"] = {
        mode: {
            "count": d["count"],
            "mean_value": round(d["sum_value"] / max(d["count"], 1), 2),
            "mean_quality": round(d["sum_quality"] / max(d["count"], 1), 2),
            "mean_reasoning": round(d["sum_reasoning"] / max(d["count"], 1), 2),
        }
        for mode, d in merged_thinking.items()
    }

    # Merge flag counts: sum across files
    merged_flags = {}
    for s in file_stats_list:
        for flag, count in s.get("flag_counts", {}).items():
            merged_flags[flag] = merged_flags.get(flag, 0) + count
    merged["flag_counts"] = dict(sorted(merged_flags.items(), key=lambda x: -x[1]))

    # Merge value_by_tag: weighted mean across files
    all_tag_dims = set()
    for s in file_stats_list:
        all_tag_dims.update(s.get("value_by_tag", {}).keys())
    merged_by_tag = {}
    for dim in all_tag_dims:
        tag_accum = {}  # tag -> {sum, count}
        for s in file_stats_list:
            dim_data = s.get("value_by_tag", {}).get(dim, {})
            for tag, info in dim_data.items():
                if tag not in tag_accum:
                    tag_accum[tag] = {"sum": 0.0, "count": 0}
                n = info.get("n", 0)
                tag_accum[tag]["sum"] += info.get("mean", 0) * n
                tag_accum[tag]["count"] += n
        merged_by_tag[dim] = {
            t: {"mean": round(a["sum"] / max(a["count"], 1), 2), "n": a["count"]}
            for t, a in sorted(tag_accum.items(),
                                key=lambda x: -x[1]["sum"] / max(x[1]["count"], 1))
        }
    merged["value_by_tag"] = merged_by_tag

    # Merge selection_by_tag: same pattern as value_by_tag
    all_sel_dims = set()
    for s in file_stats_list:
        all_sel_dims.update(s.get("selection_by_tag", {}).keys())
    merged_sel_by_tag = {}
    for dim in all_sel_dims:
        tag_accum = {}
        for s in file_stats_list:
            dim_data = s.get("selection_by_tag", {}).get(dim, {})
            for tag, info in dim_data.items():
                if tag not in tag_accum:
                    tag_accum[tag] = {"sum": 0.0, "count": 0}
                n = info.get("n", 0)
                tag_accum[tag]["sum"] += info.get("mean", 0) * n
                tag_accum[tag]["count"] += n
        merged_sel_by_tag[dim] = {
            t: {"mean": round(a["sum"] / max(a["count"], 1), 2), "n": a["count"]}
            for t, a in sorted(tag_accum.items(),
                                key=lambda x: -x[1]["sum"] / max(x[1]["count"], 1))
        }
    merged["selection_by_tag"] = merged_sel_by_tag

    return merged


# Public alias for external use (recompute module)
merge_value_stats = _merge_value_stats
