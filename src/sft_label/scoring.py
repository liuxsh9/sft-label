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
        labels = s.get("labels", {})
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
# Per-sample scoring
# ─────────────────────────────────────────────────────────

async def score_one(http_client, sample, model, rarity_result,
                    sample_idx, total, sem, config=None):
    """Score a single sample: truncate, call LLM, validate, compute value_score.

    Returns (value_dict, monitor_dict).
    """
    from sft_label.prompts_value import build_scoring_messages

    _max_retries = config.sample_max_retries if config else SAMPLE_MAX_RETRIES
    _weights = config.value_weights if config and config.value_weights else VALUE_WEIGHTS

    sample_id = sample.get("id", f"sample-{sample_idx}")
    conversations = sample.get("conversations", [])
    labels = sample.get("labels", {})

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

    # Detect thinking mode from raw conversations
    thinking_mode = detect_thinking_mode(conversations)

    # Extract COT content
    cot_text, cot_chars, convs_without_cot = extract_cot_content(conversations)

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

    # LLM call with retry
    async with sem:
        for attempt in range(_max_retries):
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
            labels = s.get("labels", {})
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
            labels = s.get("labels", {})
            for dim in tag_dims:
                t = labels.get(dim)
                if isinstance(t, list):
                    all_tags.update(f"{dim}:{x}" for x in t)
                elif t:
                    all_tags.add(f"{dim}:{t}")
        for s in retained:
            labels = s.get("labels", {})
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


async def _run_scoring_file(input_path, output_dir, tag_stats_path, limit, config):
    """Score a single labeled file."""
    input_path = Path(input_path)
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
        labels = s.get("labels", {})
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
    start_time = time.time()

    async with httpx.AsyncClient(
        proxy=None,
        trust_env=False,
        timeout=config.request_timeout,
        limits=httpx.Limits(max_connections=config.scoring_concurrency + 10),
    ) as client:
        async def score_task(idx):
            nonlocal scored_count, failed_count
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

    # Find labeled files
    labeled_files = sorted(input_dir.glob("labeled*.json"))
    labeled_files = [f for f in labeled_files if not f.name.endswith(".jsonl")]

    if not labeled_files:
        print(f"No labeled*.json files found in {input_dir}")
        return {}

    print(f"Found {len(labeled_files)} labeled files in {input_dir}")

    # Use summary_stats.json for rarity if no explicit stats provided
    if tag_stats_path is None:
        candidate = input_dir / "summary_stats.json"
        if candidate.exists():
            tag_stats_path = str(candidate)
        else:
            # Try first per-file stats
            stats_files = sorted(input_dir.glob("stats*.json"))
            stats_files = [f for f in stats_files if "value" not in f.name and "summary" not in f.name]
            if stats_files:
                tag_stats_path = str(stats_files[0])

    all_file_stats = []
    batch_start = time.time()

    for labeled_file in labeled_files:
        prefix = labeled_file.stem.replace("labeled_", "").replace("labeled", "")
        print(f"\n--- Scoring {labeled_file.name} ---")

        file_stats = await _run_scoring_file(
            labeled_file, output_dir, tag_stats_path, limit, config,
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
