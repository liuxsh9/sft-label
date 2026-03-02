"""
Value Scoring Pipeline (Pass 2)

Evaluates SFT training data value through:
  1. Rarity computation — tag IDF + combo rarity from existing tag distributions
  2. LLM-based scoring — complexity, quality, reasoning assessment
  3. Weighted aggregation — composite value_score

Runs after Pass 1 (tag labeling) and produces scored.json, stats_value.json,
dashboard_value.html per file.
"""

from __future__ import annotations

import json
import math
import time
import asyncio
import random
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass

import httpx
from rich.progress import (
    Progress, SpinnerColumn, BarColumn, TextColumn,
    TimeElapsedColumn, TimeRemainingColumn, MofNCompleteColumn,
)

from sft_label.config import (
    LITELLM_BASE, LITELLM_KEY,
    MAX_RETRIES, SAMPLE_MAX_RETRIES, REQUEST_TIMEOUT,
    VALUE_WEIGHTS, RARITY_WEIGHTS, RARITY_COMBO_ALPHA,
    KNOWN_FLAGS, KNOWN_FLAGS_POSITIVE, KNOWN_FLAGS_NEGATIVE,
    CHUNK_SIZE, MAX_ACTIVE_CHUNKS,
    SELECTION_INTRA_WEIGHT, SELECTION_MIN_GROUP_SIZE,
    PipelineConfig,
)
from sft_label.preprocessing import (
    detect_thinking_mode, extract_cot_content, truncate_for_scoring,
    count_code_blocks,
)
from sft_label.pipeline import async_llm_call


# ─────────────────────────────────────────────────────────
# Rarity computation
# ─────────────────────────────────────────────────────────

def load_tag_stats(stats_path):
    """Load tag_distributions from a stats.json file.

    Returns (distributions, total_samples, timestamp) or (None, 0, None) on failure.
    """
    path = Path(stats_path)
    if not path.exists():
        return None, 0, None

    with open(path, "r", encoding="utf-8") as f:
        stats = json.load(f)

    distributions = stats.get("tag_distributions")
    if not distributions:
        return None, 0, None

    total_samples = stats.get("total_samples", 0)
    timestamp = stats.get("timestamp", None)

    return distributions, total_samples, timestamp


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
            idf_map[dim][tag] = math.log2(total_samples / (count + 1))

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

        dim_idfs = idf_map.get(dim, {})

        if isinstance(tags, list):
            if not tags:
                continue
            tag_idf_values = [dim_idfs.get(t, math.log2(total_samples)) for t in tags]
            dim_rarity = sum(tag_idf_values) / len(tag_idf_values)
        else:
            # Single-select dimension
            dim_rarity = dim_idfs.get(tags, math.log2(total_samples))

        weighted_sum += weight * dim_rarity
        weight_total += weight

    tag_rarity = weighted_sum / weight_total if weight_total > 0 else 0.0

    # Combo rarity
    combo_rarity = 0.0
    if combo_counts is not None:
        intent = sample_labels.get("intent", "")
        difficulty = sample_labels.get("difficulty", "")
        concepts = sample_labels.get("concept", [])
        combo_key = f"{intent}|{difficulty}|{','.join(sorted(concepts[:3]))}"
        combo_count = combo_counts.get(combo_key, 0)
        combo_rarity = math.log2(total_samples / (combo_count + 1))

    raw_score = combo_alpha * tag_rarity + (1 - combo_alpha) * combo_rarity

    return {
        "score": raw_score,
        "tag_rarity": round(tag_rarity, 3),
        "combo_rarity": round(combo_rarity, 3),
        "stats_ref": stats_ref_info,
    }


def build_combo_counts(samples):
    """Build combo occurrence counts from labeled samples for combo rarity."""
    counts = {}
    for s in samples:
        labels = s.get("labels") or {}
        intent = labels.get("intent", "")
        difficulty = labels.get("difficulty", "")
        concepts = labels.get("concept", [])
        combo_key = f"{intent}|{difficulty}|{','.join(sorted(concepts[:3]))}"
        counts[combo_key] = counts.get(combo_key, 0) + 1
    return counts


def normalize_rarity_scores(rarity_results):
    """Normalize raw rarity scores to 1-10 scale via percentile mapping.

    Modifies rarity dicts in-place, setting score to normalized value.
    Returns the percentile breakpoints used.
    """
    raw_scores = [r["score"] for r in rarity_results if r["score"] is not None]
    if not raw_scores:
        return {}

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
    return breakpoints


# ─────────────────────────────────────────────────────────
# Score validation
# ─────────────────────────────────────────────────────────

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
        for key in ("instruction", "reasoning", "implementation", "overall"):
            val = complexity.get(key)
            if isinstance(val, (int, float)) and 1 <= val <= 10:
                complexity[key] = int(val)
            else:
                complexity[key] = None
                issues.append(f"complexity.{key} invalid: {val}")
        result["complexity"] = complexity
    else:
        result["complexity"] = {"instruction": None, "reasoning": None, "implementation": None, "overall": None}
        issues.append("complexity not a dict")

    # Validate quality
    quality = parsed.get("quality", {})
    if isinstance(quality, dict):
        for key in ("correctness", "code_quality", "explanation", "completeness", "overall"):
            val = quality.get(key)
            if isinstance(val, (int, float)) and 1 <= val <= 10:
                quality[key] = int(val)
            else:
                quality[key] = None
                issues.append(f"quality.{key} invalid: {val}")
        result["quality"] = quality
    else:
        result["quality"] = {"correctness": None, "code_quality": None, "explanation": None, "completeness": None, "overall": None}
        issues.append("quality not a dict")

    # Validate reasoning
    reasoning = parsed.get("reasoning", {})
    if isinstance(reasoning, dict):
        for key in ("clarity", "consistency", "overall"):
            val = reasoning.get(key)
            if isinstance(val, (int, float)) and 1 <= val <= 10:
                reasoning[key] = int(val)
            elif isinstance(val, bool):
                reasoning[key] = val  # self_correction is bool
            else:
                reasoning[key] = None
        # self_correction can be bool
        sc = reasoning.get("self_correction")
        if isinstance(sc, bool):
            reasoning["self_correction"] = sc
        else:
            reasoning["self_correction"] = None
        result["reasoning"] = reasoning
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

    return result, issues


# ─────────────────────────────────────────────────────────
# Value score computation
# ─────────────────────────────────────────────────────────

def compute_value_score(score_result, rarity_result, weights=None):
    """Compute weighted composite value_score.

    Handles null rarity by renormalizing weights.
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

    # Weighted mean with renormalization for missing components
    total_weight = sum(weights.get(k, 0) for k in components)
    if total_weight <= 0:
        return None

    value = sum(weights.get(k, 0) * v for k, v in components.items()) / total_weight
    return round(value, 1)


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

    return sum(quality_weights.get(k, 0) * v for k, v in components.items()) / total_w


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

    quality_weights = {k: v for k, v in VALUE_WEIGHTS.items() if k != "rarity"}

    # Step 1: Compute pure_quality for each sample
    pure_qualities = []
    for s in samples:
        v = s.get("value")
        if v:
            pure_qualities.append(_extract_pure_quality(v, quality_weights))
        else:
            pure_qualities.append(None)

    # Step 2: Per-dimension, per-tag quality percentile
    dim_percentiles = [{} for _ in samples]  # [{dim: percentile}, ...]

    for dim in _SELECTION_DIMS:
        # Build tag -> list of (sample_idx, pure_quality)
        tag_groups = {}
        for i, s in enumerate(samples):
            if pure_qualities[i] is None:
                continue
            labels = s.get("labels") or {}
            tags = labels.get(dim)
            if tags is None:
                continue
            if isinstance(tags, list):
                for t in tags:
                    tag_groups.setdefault(t, []).append((i, pure_qualities[i]))
            else:
                tag_groups.setdefault(tags, []).append((i, pure_qualities[i]))

        # For each tag, compute percentiles
        for tag, members in tag_groups.items():
            if len(members) < min_group_size:
                continue

            sorted_members = sorted(members, key=lambda x: x[1])
            n = len(sorted_members)
            for rank, (idx, _pq) in enumerate(sorted_members):
                percentile = rank / max(n - 1, 1)
                # Multi-select: take min (conservative)
                if dim in dim_percentiles[idx]:
                    dim_percentiles[idx][dim] = min(dim_percentiles[idx][dim], percentile)
                else:
                    dim_percentiles[idx][dim] = percentile

    # Step 3: Global percentile fallback for samples not in any large-enough group
    valid_pq = [(i, pq) for i, pq in enumerate(pure_qualities) if pq is not None]
    global_percentiles = {}
    if valid_pq:
        sorted_global = sorted(valid_pq, key=lambda x: x[1])
        n = len(sorted_global)
        for rank, (idx, _pq) in enumerate(sorted_global):
            global_percentiles[idx] = rank / max(n - 1, 1)

    # Step 4: Weighted fusion across dimensions → intra_class_rank + selection_score
    for i, s in enumerate(samples):
        v = s.get("value")
        if v is None:
            continue

        percs = dim_percentiles[i]

        if percs:
            weighted_sum = 0.0
            weight_total = 0.0
            for dim, pct in percs.items():
                w = rarity_weights.get(dim, 1.0)
                weighted_sum += w * pct
                weight_total += w
            if weight_total > 0:
                fused = weighted_sum / weight_total  # 0-1
                intra_class_rank = round(fused * 9 + 1, 1)
            else:
                intra_class_rank = None
        else:
            # Fallback: global percentile
            if i in global_percentiles:
                intra_class_rank = round(global_percentiles[i] * 9 + 1, 1)
            else:
                intra_class_rank = None

        v["intra_class_rank"] = intra_class_rank

        rarity_score = (v.get("rarity") or {}).get("score")
        if intra_class_rank is not None:
            if rarity_score is not None:
                v["selection_score"] = round(
                    intra_weight * intra_class_rank + (1 - intra_weight) * rarity_score, 1)
            else:
                v["selection_score"] = intra_class_rank
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

    quality_weights = {k: v for k, v in VALUE_WEIGHTS.items() if k != "rarity"}
    total_qw = sum(quality_weights.values())

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
        else:
            pq = None
        pure_qualities.append(pq)

    # Step 2: Per-tag percentile
    dim_percentiles = [{} for _ in summaries]

    for dim in _SELECTION_DIMS:
        tag_groups = {}
        for i, s in enumerate(summaries):
            if pure_qualities[i] is None:
                continue
            labels = s.get("labels") or {}
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
            sorted_members = sorted(members, key=lambda x: x[1])
            n = len(sorted_members)
            for rank, (idx, _pq) in enumerate(sorted_members):
                percentile = rank / max(n - 1, 1)
                if dim in dim_percentiles[idx]:
                    dim_percentiles[idx][dim] = min(dim_percentiles[idx][dim], percentile)
                else:
                    dim_percentiles[idx][dim] = percentile

    # Step 3: Global fallback
    valid_pq = [(i, pq) for i, pq in enumerate(pure_qualities) if pq is not None]
    global_percentiles = {}
    if valid_pq:
        sorted_global = sorted(valid_pq, key=lambda x: x[1])
        n = len(sorted_global)
        for rank, (idx, _pq) in enumerate(sorted_global):
            global_percentiles[idx] = rank / max(n - 1, 1)

    # Step 4: Fusion
    results = []
    for i, s in enumerate(summaries):
        percs = dim_percentiles[i]

        if percs:
            weighted_sum = 0.0
            weight_total = 0.0
            for dim, pct in percs.items():
                w = rarity_weights.get(dim, 1.0)
                weighted_sum += w * pct
                weight_total += w
            if weight_total > 0:
                fused = weighted_sum / weight_total
                intra_class_rank = round(fused * 9 + 1, 1)
            else:
                intra_class_rank = None
        else:
            if i in global_percentiles:
                intra_class_rank = round(global_percentiles[i] * 9 + 1, 1)
            else:
                intra_class_rank = None

        rarity_score = s.get("rarity_score")
        if intra_class_rank is not None:
            if rarity_score is not None:
                selection_score = round(
                    intra_weight * intra_class_rank + (1 - intra_weight) * rarity_score, 1)
            else:
                selection_score = intra_class_rank
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
                    sample_idx, total, sem, config=None):
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
    truncated = truncate_for_scoring(
        conversations, thinking_mode, cot_text=cot_text,
        budget=config.value_truncation_budget if config else None,
    )

    # Build messages
    code_block_count = count_code_blocks(
        " ".join(t.get("value", "") for t in conversations)
    )
    total_turns = len(conversations)

    messages = build_scoring_messages(
        truncated=truncated,
        thinking_mode=thinking_mode,
        labels=labels,
        total_turns=total_turns,
        code_block_count=code_block_count,
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
            )
            monitor["llm_calls"] += 1
            monitor["prompt_tokens"] += usage.get("prompt_tokens", 0)
            monitor["completion_tokens"] += usage.get("completion_tokens", 0)

        if parsed is None:
            monitor["error"] = raw[:300] if raw else "null response"
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


def compute_value_stats(scored_samples, all_monitors):
    """Compute aggregate statistics for value scoring.

    Returns dict matching stats_value.json structure.
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
    for dim in tag_dims:
        tag_values = {}  # {tag: [value_scores]}
        for s in scored_samples:
            v = s.get("value", {})
            vs = v.get("value_score")
            if vs is None:
                continue
            labels = s.get("labels") or {}
            tags = labels.get(dim)
            if tags is None:
                continue
            if isinstance(tags, list):
                for t in tags:
                    tag_values.setdefault(t, []).append(vs)
            else:
                tag_values.setdefault(tags, []).append(vs)
        value_by_tag[dim] = {
            t: {"mean": round(sum(vs) / len(vs), 2), "n": len(vs)}
            for t, vs in sorted(tag_values.items(), key=lambda x: -sum(x[1]) / len(x[1]))
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
        "sub_score_means": sub_score_means,
        "value_by_tag": value_by_tag,
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
                      limit=0, config=None):
    """Run value scoring (Pass 2) on pre-labeled data.

    Args:
        input_path: Path to labeled.json or directory of labeled files
        output_dir: Where to write outputs (default: same directory as input)
        tag_stats_path: Path to stats.json for rarity computation
        limit: Max samples to score (0 = all)
        config: PipelineConfig override
    """
    if config is None:
        config = PipelineConfig()

    input_path = Path(input_path)

    # Determine if directory or single file
    if input_path.is_dir():
        return await _run_scoring_directory(input_path, output_dir, tag_stats_path, limit, config)
    else:
        return await _run_scoring_file(input_path, output_dir, tag_stats_path, limit, config)


async def _run_scoring_file_chunked(input_path, output_dir, tag_stats_path,
                                     limit, config):
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
    stats_ref_info = None
    idf_map = {}
    total_stats_samples = 0

    if tag_stats_path is None:
        candidate = input_path.parent / "stats.json"
        if candidate.exists():
            tag_stats_path = str(candidate)

    stats_source = tag_stats_path
    if stats_source:
        distributions, total_stats_samples, ts = load_tag_stats(stats_source)
        if distributions:
            idf_map = compute_tag_idf(distributions, total_stats_samples)
            stats_ref_info = {
                "source": str(stats_source),
                "total_samples": total_stats_samples,
                "timestamp": ts or datetime.now().isoformat(),
            }
            print(f"  Tag stats loaded from {stats_source} (N={total_stats_samples})")
        else:
            print(f"  Warning: {stats_source} has no tag_distributions, rarity will be null")
    else:
        print("  No tag stats found, rarity will be null")

    # ── Pass A: Compute rarity (stream, no LLM) ──
    print(f"  Pass A: Computing rarity from {input_path}...")
    raw_rarities = []
    combo_counts = {}
    sample_count = 0
    with open(input_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            sample = json.loads(line)
            labels = sample.get("labels") or {}
            # Build combo counts on the fly
            intent = labels.get("intent", "")
            difficulty = labels.get("difficulty", "")
            concepts = labels.get("concept", [])
            combo_key = f"{intent}|{difficulty}|{','.join(sorted(concepts[:3]))}"
            combo_counts[combo_key] = combo_counts.get(combo_key, 0) + 1
            sample_count += 1
            del sample
            if limit > 0 and sample_count >= limit:
                break

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

    normalize_rarity_scores(raw_rarities)
    total = len(raw_rarities)
    print(f"  {total} samples, rarity computed")

    # ── Pass B: Chunked LLM scoring ──
    print(f"  Pass B: LLM scoring model={config.scoring_model} concurrency={config.scoring_concurrency}")
    print(f"  Endpoint: {config.litellm_base}")

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

    try:
        async with httpx.AsyncClient(
            timeout=config.request_timeout,
            limits=httpx.Limits(max_connections=config.scoring_concurrency + 10),
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
                    idx, total, sem, config=config,
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
                       and len(pending_futures) < config.scoring_concurrency * 2
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

                    out_scored.write(json.dumps(sample, ensure_ascii=False) + "\n")

                    if monitor:
                        all_monitors.append(monitor)
                        out_monitor.write(json.dumps(monitor, ensure_ascii=False) + "\n")

                # Release
                chunk_data["samples"] = None
                chunk_data["values"] = None
                chunk_data["monitors"] = None

            with _create_progress() as progress:
                task = progress.add_task("Scoring", total=total, info="")

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
                            print(f"  [!] First failure: sample={monitor.get('sample_id')} "
                                  f"attempts={monitor.get('attempts')} err={err[:200]}")
                            first_error_logged = True

                        progress.update(task, advance=1,
                                        info=f"ok={scored_count + (1 if value else 0)} fail={failed_count + (0 if value else 1)}")

                        # Check if chunk is done
                        if chunk_data["done"] >= chunk_data["total"]:
                            flush_scoring_chunk(chunk_idx)

                    maybe_load_more_scoring()

    finally:
        out_scored.close()
        out_monitor.close()
        out_failed.close()

    elapsed = time.time() - start_time

    # Remove empty failed file
    failed_path = output_dir / "failed_value.jsonl"
    if failed_path.exists() and failed_path.stat().st_size == 0:
        failed_path.unlink()

    # ── Compute selection scores from summaries ──
    print("  Computing selection scores (intra-class ranking)...")
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
    stats["weights_used"] = config.value_weights or VALUE_WEIGHTS
    stats["rarity_config"] = {
        "stats_ref": str(stats_source) if stats_source else None,
        "total_samples_in_distribution": total_stats_samples,
        "dimension_weights": config.rarity_weights or RARITY_WEIGHTS,
        "combo_alpha": config.rarity_combo_alpha,
    }
    stats["chunked"] = True

    stats_path = output_dir / "stats_value.json"
    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)

    # Dashboard
    try:
        from sft_label.tools.visualize_value import generate_value_dashboard
        generate_value_dashboard(output_dir, scored_file=None,
                                 stats_file="stats_value.json",
                                 output_file="dashboard_value.html")
    except Exception as e:
        print(f"  Warning: dashboard generation failed: {e}")

    print(f"\nScoring complete: {scored_count}/{total} scored, {failed_count} failed")
    print(f"  Time: {elapsed:.1f}s | Output: {output_dir}")

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

    # Thinking mode stats
    slow = [s for s in summaries if s.get("thinking_mode") == "slow"]
    fast = [s for s in summaries if s.get("thinking_mode") == "fast"]
    thinking_mode_stats = {
        "slow": {
            "count": len(slow),
            "mean_value": round(sum(s.get("value_score", 0) or 0 for s in slow) / max(len(slow), 1), 2),
        },
        "fast": {
            "count": len(fast),
            "mean_value": round(sum(s.get("value_score", 0) or 0 for s in fast) / max(len(fast), 1), 2),
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

    # Value by tag
    tag_dims = ["intent", "difficulty", "domain", "concept", "task",
                "agentic", "constraint", "context"]
    value_by_tag = {}
    for dim in tag_dims:
        tag_values = {}
        for s in summaries:
            vs = s.get("value_score")
            if vs is None:
                continue
            labels = s.get("labels") or {}
            tags = labels.get(dim)
            if tags is None:
                continue
            if isinstance(tags, list):
                for t in tags:
                    tag_values.setdefault(t, []).append(vs)
            else:
                tag_values.setdefault(tags, []).append(vs)
        value_by_tag[dim] = {
            t: {"mean": round(sum(vs) / len(vs), 2), "n": len(vs)}
            for t, vs in sorted(tag_values.items(), key=lambda x: -sum(x[1]) / len(x[1]))
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
        "value_by_tag": value_by_tag,
        "thinking_mode_stats": thinking_mode_stats,
        "flag_counts": dict(sorted(flag_counts.items(), key=lambda x: -x[1])),
        "flag_value_impact": flag_value_impact,
        "selection_thresholds": selection_thresholds,
    }


async def _run_scoring_file(input_path, output_dir, tag_stats_path, limit, config):
    """Score a single labeled file."""
    input_path = Path(input_path)

    # Dispatch to chunked pipeline for JSONL files (memory-bounded)
    if input_path.suffix == ".jsonl":
        return await _run_scoring_file_chunked(
            input_path, output_dir, tag_stats_path, limit, config,
        )

    if output_dir is None:
        output_dir = input_path.parent
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load labeled data
    print(f"Loading labeled data from {input_path}...")
    with open(input_path, "r", encoding="utf-8") as f:
        if input_path.suffix == ".jsonl":
            samples = [json.loads(line) for line in f if line.strip()]
        else:
            samples = json.load(f)

    if limit > 0:
        samples = samples[:limit]
    total = len(samples)
    print(f"  {total} samples loaded")

    # Load tag stats for rarity
    stats_ref_info = None
    idf_map = {}
    total_stats_samples = 0
    combo_counts = None

    if tag_stats_path:
        stats_source = tag_stats_path
    else:
        # Auto-discover stats.json in same directory
        candidate = input_path.parent / "stats.json"
        if candidate.exists():
            stats_source = str(candidate)
        else:
            stats_source = None

    if stats_source:
        distributions, total_stats_samples, ts = load_tag_stats(stats_source)
        if distributions:
            idf_map = compute_tag_idf(distributions, total_stats_samples)
            stats_ref_info = {
                "source": str(stats_source),
                "total_samples": total_stats_samples,
                "timestamp": ts or datetime.now().isoformat(),
            }
            combo_counts = build_combo_counts(samples)
            print(f"  Tag stats loaded from {stats_source} (N={total_stats_samples})")
        else:
            print(f"  Warning: {stats_source} has no tag_distributions, rarity will be null")
    else:
        print("  No tag stats found, rarity will be null")

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
    normalize_rarity_scores(rarity_results)

    # Run LLM scoring
    sem = asyncio.Semaphore(config.scoring_concurrency)
    all_monitors = [None] * total
    all_values = [None] * total

    scored_count = 0
    failed_count = 0
    first_error_logged = False
    start_time = time.time()

    print(f"  Scoring: model={config.scoring_model} concurrency={config.scoring_concurrency} timeout={config.request_timeout}s")
    print(f"  Endpoint: {config.litellm_base}")

    async with httpx.AsyncClient(
        timeout=config.request_timeout,
        limits=httpx.Limits(max_connections=config.scoring_concurrency + 10),
    ) as client:
        async def score_task(idx):
            nonlocal scored_count, failed_count, first_error_logged
            value, monitor = await score_one(
                client, samples[idx], config.scoring_model, rarity_results[idx],
                idx, total, sem, config=config,
            )
            all_values[idx] = value
            all_monitors[idx] = monitor
            if value:
                scored_count += 1
            else:
                failed_count += 1
                if not first_error_logged and monitor:
                    err = monitor.get("error", "unknown")
                    print(f"  [!] First failure: sample={monitor.get('sample_id')} "
                          f"attempts={monitor.get('attempts')} err={err[:200]}")
                    first_error_logged = True
            return idx

        with _create_progress() as progress:
            task = progress.add_task("Scoring", total=total, info="")

            tasks = [asyncio.create_task(score_task(i)) for i in range(total)]
            for coro in asyncio.as_completed(tasks):
                idx = await coro
                progress.update(task, advance=1,
                                info=f"ok={scored_count} fail={failed_count}")

    elapsed = time.time() - start_time

    # Attach value to samples
    for i, s in enumerate(samples):
        if all_values[i]:
            s["value"] = all_values[i]

    # Compute selection scores (intra-class quality ranking, no LLM)
    print("  Computing selection scores (intra-class ranking)...")
    compute_selection_scores(samples, config=config)

    # Write outputs
    scored_path = output_dir / "scored.json"
    with open(scored_path, "w", encoding="utf-8") as f:
        json.dump(samples, f, ensure_ascii=False, indent=2)

    scored_jsonl_path = output_dir / "scored.jsonl"
    with open(scored_jsonl_path, "w", encoding="utf-8") as f:
        for s in samples:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")

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

    # Stats
    stats = compute_value_stats(samples, [m for m in all_monitors if m])
    stats["elapsed_seconds"] = round(elapsed, 1)
    stats["model"] = config.scoring_model
    stats["weights_used"] = config.value_weights or VALUE_WEIGHTS
    stats["rarity_config"] = {
        "stats_ref": str(stats_source) if stats_source else None,
        "total_samples_in_distribution": total_stats_samples,
        "dimension_weights": config.rarity_weights or RARITY_WEIGHTS,
        "combo_alpha": config.rarity_combo_alpha,
    }

    stats_path = output_dir / "stats_value.json"
    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)

    # Dashboard
    try:
        from sft_label.tools.visualize_value import generate_value_dashboard
        generate_value_dashboard(output_dir, scored_file="scored.json",
                                 stats_file="stats_value.json",
                                 output_file="dashboard_value.html")
    except Exception as e:
        print(f"  Warning: dashboard generation failed: {e}")

    print(f"\nScoring complete: {scored_count}/{total} scored, {failed_count} failed")
    print(f"  Time: {elapsed:.1f}s | Output: {output_dir}")

    return stats


async def _run_scoring_directory(input_dir, output_dir, tag_stats_path, limit, config):
    """Score all labeled files in a directory."""
    input_dir = Path(input_dir)
    if output_dir is None:
        output_dir = input_dir
    output_dir = Path(output_dir)

    # Find labeled files (flat, one-level subdirs, or deeply nested)
    # Try progressively deeper searches; also include .jsonl files
    labeled_files = sorted(input_dir.glob("labeled*.json"))
    if not labeled_files:
        labeled_files = sorted(input_dir.glob("*/labeled*.json"))
    if not labeled_files:
        # Deep nesting: recursive glob
        labeled_files = sorted(input_dir.glob("**/labeled*.json"))
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

    print(f"Found {len(labeled_files)} labeled files in {input_dir}")

    # Use summary_stats.json for rarity if no explicit stats provided
    if tag_stats_path is None:
        candidate = input_dir / "summary_stats.json"
        if candidate.exists():
            tag_stats_path = str(candidate)
        else:
            # Try per-file stats (flat, one-level, or recursive)
            stats_files = sorted(input_dir.glob("stats*.json"))
            if not stats_files:
                stats_files = sorted(input_dir.glob("*/stats*.json"))
            if not stats_files:
                stats_files = sorted(input_dir.glob("**/stats*.json"))
            stats_files = [f for f in stats_files if "value" not in f.name and "summary" not in f.name]
            if stats_files:
                tag_stats_path = str(stats_files[0])

    all_file_stats = []
    batch_start = time.time()

    for labeled_file in labeled_files:
        prefix = labeled_file.stem.replace("labeled_", "").replace("labeled", "")
        # Write scored output alongside the labeled file (may be in a subdir)
        file_output_dir = labeled_file.parent
        print(f"\n--- Scoring {labeled_file.relative_to(input_dir)} ---")

        file_stats = await _run_scoring_file(
            labeled_file, file_output_dir, tag_stats_path, limit, config,
        )
        if file_stats:
            file_stats["file"] = labeled_file.name
            all_file_stats.append(file_stats)

    # Write global summary
    if all_file_stats:
        summary = _merge_value_stats(all_file_stats)
        summary["elapsed_seconds"] = round(time.time() - batch_start, 1)
        summary["model"] = config.scoring_model
        summary["files_processed"] = len(all_file_stats)

        summary_path = output_dir / "summary_stats_value.json"
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)

        # Global dashboard
        try:
            from sft_label.tools.visualize_value import generate_value_dashboard
            dir_name = input_dir.name
            generate_value_dashboard(output_dir, scored_file=None,
                                     stats_file="summary_stats_value.json",
                                     output_file=f"dashboard_value_{dir_name}.html")
        except Exception as e:
            print(f"  Warning: global dashboard generation failed: {e}")

        print(f"\nAll files scored. Summary: {summary_path}")

    return summary if all_file_stats else {}


def _merge_value_stats(file_stats_list):
    """Merge per-file value stats into a global summary."""
    merged = {
        "total_scored": sum(s.get("total_scored", 0) for s in file_stats_list),
        "total_failed": sum(s.get("total_failed", 0) for s in file_stats_list),
        "total_llm_calls": sum(s.get("total_llm_calls", 0) for s in file_stats_list),
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
        }
        for s in file_stats_list
    ]

    # Merge score distributions (use weighted means)
    for key in ("score_distributions", "thinking_mode_stats", "flag_counts"):
        if file_stats_list and key in file_stats_list[0]:
            merged[key] = file_stats_list[0][key]  # Simplified: use first file's stats as base

    return merged
