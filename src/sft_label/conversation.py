"""
Conversation-Level Value Aggregation (Post-scoring, no LLM)

Aggregates per-turn slice scores into conversation-level metrics for
multi-turn conversations. Enables conversation-as-a-unit filtering.

Runs after Pass 2 scoring. Pure computation — no LLM calls, no async.
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

from sft_label.config import (
    CONV_CONFIDENCE_INHERITED,
    CONV_QUALITY_PENALTIES,
    CONV_QUALITY_PENALTY_DEFAULT,
    CONV_FLAG_PENALTY_BASE,
    CONV_AGENTIC_QUALITY_PERCENTILE,
    CONV_RARITY_MEAN_WEIGHT,
    CONV_RARITY_PEAK_WEIGHT,
    CONV_RARITY_DIVERSITY_BONUS,
    CONV_COVERAGE_CONFIDENCE_FLOOR,
    VALUE_WEIGHTS,
    RARITY_WEIGHTS,
    SELECTION_INTRA_WEIGHT,
    SELECTION_QUALITY_WEIGHT,
    SELECTION_MIN_GROUP_SIZE,
    SELECTION_SMOOTHING_PRIOR,
    KNOWN_FLAGS_NEGATIVE,
)
from sft_label.labels import LABEL_META_KEYS
from sft_label.score_confidence import apply_score_confidence, score_confidence


def build_conversation_key(source_id, source_file=None):
    """Build a stable conversation key across files."""
    if not source_id:
        return None
    if source_file:
        return f"{source_file}::{source_id}"
    return source_id


def sample_conversation_key(sample):
    """Build the conversation key for a scored sample."""
    meta = sample.get("metadata") or {}
    return build_conversation_key(meta.get("source_id"), meta.get("source_file"))


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


# ─── Grouping ────────────────────────────────────────────

def group_by_conversation(samples):
    """Group samples by conversation, filter to multi-turn only.

    Returns dict {conversation_key: [slices sorted by turn_index]}.
    Only includes conversations where total_turns > 1.
    """
    groups = defaultdict(list)
    for s in samples:
        meta = s.get("metadata") or {}
        source_id = meta.get("source_id")
        total_turns = meta.get("total_turns", 1)
        conv_key = build_conversation_key(source_id, meta.get("source_file"))
        if conv_key and total_turns > 1:
            groups[conv_key].append(s)

    # Sort each group by turn_index
    for conv_key in groups:
        groups[conv_key].sort(
            key=lambda s: (s.get("metadata") or {}).get("turn_index", 0)
        )

    return dict(groups)


# ─── Weighting helpers ───────────────────────────────────

def _position_weight(i, n):
    """Linear position weight: 1.0 for first turn → 2.0 for last turn.

    Args:
        i: 0-based turn index within the conversation
        n: total number of slices in the conversation
    """
    if n <= 1:
        return 1.0
    return 1.0 + 1.0 * i / (n - 1)


def _effective_weights(slices):
    """Compute effective weight per slice: position_weight × confidence.

    Inherited slices get reduced confidence (CONV_CONFIDENCE_INHERITED).
    """
    n = len(slices)
    weights = []
    for i, s in enumerate(slices):
        pw = _position_weight(i, n)
        labels = s.get("labels") or {}
        conf = CONV_CONFIDENCE_INHERITED if labels.get("inherited") else 1.0
        weights.append(pw * conf)
    return weights


def _position_weights(slices):
    """Return raw position weights without inherited confidence discount."""
    n = len(slices)
    return [_position_weight(i, n) for i, _s in enumerate(slices)]


def _weighted_average(slices, weights, extract_fn):
    """Generic weighted average over slices, skipping None values."""
    total_w = 0.0
    total_v = 0.0
    for s, w in zip(slices, weights):
        val = extract_fn(s)
        if val is not None:
            total_w += w
            total_v += w * val
    if total_w <= 0:
        return None
    return total_v / total_w


# ─── Penalty computation ─────────────────────────────────

def _compute_penalty(slices):
    """Compute quality penalty from floor quality and negative flags.

    Returns (penalty, quality_floor, negative_flags_list).
    """
    # Find quality floor (min quality across slices)
    quality_values = []
    for s in slices:
        v = s.get("value") or {}
        q = (v.get("quality") or {}).get("overall")
        if q is not None:
            quality_values.append(q)

    quality_floor = min(quality_values) if quality_values else None

    # For conversations with agentic behaviors, use p10 quality instead of min.
    # Agent trajectories naturally have low-quality "error" turns (tool returns
    # error, exploratory dead ends) that are normal exploration, not quality failures.
    has_agentic = any(
        (s.get("labels") or {}).get("agentic")
        for s in slices
    )
    if has_agentic and len(quality_values) >= 5:
        sorted_q = sorted(quality_values)
        p10_idx = max(0, int(len(sorted_q) * CONV_AGENTIC_QUALITY_PERCENTILE))
        quality_floor = sorted_q[p10_idx]

    # Quality floor penalty
    q_penalty = CONV_QUALITY_PENALTY_DEFAULT
    if quality_floor is not None:
        for threshold in sorted(CONV_QUALITY_PENALTIES.keys()):
            if quality_floor < threshold:
                q_penalty = CONV_QUALITY_PENALTIES[threshold]
                break

    # Collect unique negative flags across all slices
    neg_flags = set()
    for s in slices:
        v = s.get("value") or {}
        flags = v.get("flags") or []
        for f in flags:
            if f in KNOWN_FLAGS_NEGATIVE:
                neg_flags.add(f)

    neg_flags_list = sorted(neg_flags)

    # Flag penalty: 0.95 ^ count
    flag_penalty = CONV_FLAG_PENALTY_BASE ** len(neg_flags_list)

    penalty = q_penalty * flag_penalty
    return penalty, quality_floor, neg_flags_list


# ─── Label merging ───────────────────────────────────────

_SINGLE_SELECT_DIMS = {"intent", "difficulty", "context"}
_MULTI_SELECT_DIMS = {"language", "domain", "task", "concept", "agentic", "constraint"}
_LABEL_META_KEYS = LABEL_META_KEYS


def _merge_labels(slices):
    """Merge labels across slices.

    Single-select dims: last slice wins.
    Multi-select dims: union of all tags.
    """
    merged = {}
    for s in slices:
        labels = s.get("labels") or {}
        for dim, val in labels.items():
            if dim in _LABEL_META_KEYS:
                continue
            if dim in _SINGLE_SELECT_DIMS:
                merged[dim] = val
            elif dim in _MULTI_SELECT_DIMS:
                existing = merged.get(dim, [])
                if not isinstance(existing, list):
                    existing = [existing]
                if isinstance(val, list):
                    for v in val:
                        if v not in existing:
                            existing.append(v)
                elif val not in existing:
                    existing.append(val)
                merged[dim] = existing
            else:
                # Unknown dim: last-wins
                merged[dim] = val
    return merged


def _label_signature(labels):
    """Build a stable label-state signature for trajectory diversity."""
    if not isinstance(labels, dict):
        return ()

    parts = []
    for dim in sorted(_SINGLE_SELECT_DIMS | _MULTI_SELECT_DIMS):
        if dim in _SINGLE_SELECT_DIMS:
            value = labels.get(dim)
            if isinstance(value, str) and value:
                parts.append((dim, value))
            continue

        value = labels.get(dim)
        if isinstance(value, str):
            value = [value]
        if not isinstance(value, list):
            continue
        cleaned = tuple(sorted({item for item in value if isinstance(item, str) and item}))
        if cleaned:
            parts.append((dim, cleaned))

    return tuple(parts)


def _coverage_metrics(slices):
    """Estimate how much of a conversation is directly labeled vs inherited."""
    if not slices:
        return {
            "observed_turns": 0,
            "inherited_turns": 0,
            "observed_turn_ratio": 0.0,
            "inherited_turn_ratio": 0.0,
            "observed_weight_ratio": 0.0,
            "coverage_confidence": CONV_COVERAGE_CONFIDENCE_FLOOR,
        }

    raw_weights = _position_weights(slices)
    total_weight = sum(raw_weights) or 1.0

    observed_turns = 0
    inherited_turns = 0
    observed_weight = 0.0

    for s, raw_weight in zip(slices, raw_weights):
        labels = s.get("labels") or {}
        if labels.get("inherited"):
            inherited_turns += 1
        else:
            observed_turns += 1
            observed_weight += raw_weight

    observed_turn_ratio = observed_turns / len(slices)
    observed_weight_ratio = observed_weight / total_weight
    coverage_confidence = (
        CONV_COVERAGE_CONFIDENCE_FLOOR
        + (1.0 - CONV_COVERAGE_CONFIDENCE_FLOOR) * observed_weight_ratio
    )

    return {
        "observed_turns": observed_turns,
        "inherited_turns": inherited_turns,
        "observed_turn_ratio": observed_turn_ratio,
        "inherited_turn_ratio": inherited_turns / len(slices),
        "observed_weight_ratio": observed_weight_ratio,
        "coverage_confidence": min(1.0, max(0.0, coverage_confidence)),
    }


def _compute_conv_rarity(slices, weights, score_conf):
    """Aggregate conversation rarity with diversity and inheritance shrinkage."""
    rarity_values = []
    for s, weight in zip(slices, weights):
        rarity = ((s.get("value") or {}).get("rarity") or {}).get("score")
        if rarity is None:
            continue
        labels = s.get("labels") or {}
        rarity_values.append((rarity, weight, bool(labels.get("inherited"))))

    if not rarity_values:
        return None, {}

    total_weight = sum(weight for _rarity, weight, _inherited in rarity_values) or 1.0
    rarity_mean = sum(rarity * weight for rarity, weight, _inherited in rarity_values) / total_weight

    observed_rarities = [rarity for rarity, _weight, inherited in rarity_values if not inherited]
    rarity_peak = max(observed_rarities) if observed_rarities else max(rarity for rarity, _weight, _inherited in rarity_values)

    signatures = {
        _label_signature(s.get("labels") or {})
        for s in slices
        if not (s.get("labels") or {}).get("inherited")
    }
    signatures.discard(())
    signature_count = len(signatures)
    transition_scale = max(len(slices) - 1, 1)
    diversity_ratio = min(1.0, max(0.0, (signature_count - 1) / transition_scale))
    diversity_bonus = CONV_RARITY_DIVERSITY_BONUS * diversity_ratio

    coverage = _coverage_metrics(slices)
    rarity_conf = min(
        1.0,
        0.5 * coverage["coverage_confidence"] + 0.5 * max(0.0, min(1.0, score_conf or 1.0)),
    )

    raw_rarity = (
        CONV_RARITY_MEAN_WEIGHT * rarity_mean
        + CONV_RARITY_PEAK_WEIGHT * rarity_peak
        + diversity_bonus
    )
    conv_rarity = apply_score_confidence(min(10.0, raw_rarity), rarity_conf)
    conv_rarity = max(1.0, min(10.0, conv_rarity))

    return conv_rarity, {
        "rarity_mean": rarity_mean,
        "rarity_peak": rarity_peak,
        "rarity_diversity_bonus": diversity_bonus,
        "label_signature_count": signature_count,
        "rarity_confidence": rarity_conf,
        **coverage,
    }


# ─── Pure quality from slice ─────────────────────────────

def _compute_pure_quality_from_slice(s):
    """Compute pure quality (without rarity) from a slice's value dict.

    Uses VALUE_WEIGHTS excluding rarity, renormalized.
    """
    v = s.get("value") or {}
    components = {}

    c = (v.get("complexity") or {}).get("overall")
    if c is not None:
        components["complexity"] = c

    q = (v.get("quality") or {}).get("overall")
    if q is not None:
        components["quality"] = q

    r = (v.get("reasoning") or {}).get("overall")
    if r is not None:
        components["reasoning"] = r

    if not components:
        return None

    quality_weights = {k: w for k, w in VALUE_WEIGHTS.items() if k != "rarity"}
    total_w = sum(quality_weights.get(k, 0) for k in components)
    if total_w <= 0:
        return None

    pq = sum(quality_weights.get(k, 0) * v for k, v in components.items()) / total_w

    # NOTE: No quality floor penalty here — penalty is applied only at the
    # conversation level via _compute_penalty() to avoid double-penalizing.
    # The sample-level _extract_pure_quality (in scoring.py) keeps its own
    # 0.7× penalty because it feeds directly into selection_score.

    pq = apply_score_confidence(pq, score_confidence(v))
    return pq


# ─── Per-conversation aggregation ────────────────────────

def aggregate_conversation(conversation_key, slices):
    """Aggregate scores for a single conversation.

    Returns a record dict, or None if insufficient data (e.g. single slice).
    """
    if len(slices) < 2:
        return None

    first_meta = (slices[0].get("metadata") or {}) if slices else {}
    source_id = first_meta.get("source_id") or conversation_key
    source_file = first_meta.get("source_file")

    weights = _effective_weights(slices)

    # Weighted average of value_score → q_base
    q_base = _weighted_average(
        slices, weights,
        lambda s: (s.get("value") or {}).get("value_score")
    )
    if q_base is None:
        return None

    # Max-pooling blend: prevent a single bad early turn from
    # drowning out genuinely good later turns
    valid_values = [
        (s.get("value") or {}).get("value_score")
        for s in slices
    ]
    valid_values = [v for v in valid_values if v is not None]
    if valid_values:
        q_base = 0.7 * q_base + 0.3 * max(valid_values)

    # Penalty
    penalty, quality_floor, neg_flags = _compute_penalty(slices)

    score_conf = _weighted_average(
        slices, weights,
        lambda s: score_confidence(s.get("value") or {}, default=1.0)
    )

    # conv_value = q_base × penalty, clamped to [1, 10]
    conv_value = apply_score_confidence(q_base * penalty, score_conf)
    conv_value = max(1.0, min(10.0, conv_value))

    # Peak complexity
    complexities = []
    for s in slices:
        v = s.get("value") or {}
        c = (v.get("complexity") or {}).get("overall")
        if c is not None:
            complexities.append(c)
    peak_complexity = max(complexities) if complexities else None

    # Conversation rarity: weighted mean + peak + label-state diversity,
    # then shrink when the conversation is dominated by inherited slices.
    conv_rarity, rarity_detail = _compute_conv_rarity(slices, weights, score_conf)

    # Pure quality (for selection score computation)
    pure_quality = _weighted_average(
        slices, weights,
        _compute_pure_quality_from_slice
    )

    # Thinking mode: use first slice's mode (all slices from same conv share it)
    thinking_mode = None
    for s in slices:
        v = s.get("value") or {}
        meta = s.get("metadata") or {}
        tm = v.get("thinking_mode") or meta.get("thinking_mode")
        if tm:
            thinking_mode = tm
            break

    # Merged labels
    merged_labels = _merge_labels(slices)

    # Turn count
    turn_count = len(slices)

    # Slice details
    slice_details = []
    for i, (s, w) in enumerate(zip(slices, weights)):
        v = s.get("value") or {}
        labels = s.get("labels") or {}
        meta = s.get("metadata") or {}
        slice_details.append({
            "id": s.get("id", ""),
            "turn_index": meta.get("turn_index", i),
            "value_score": v.get("value_score"),
            "confidence": CONV_CONFIDENCE_INHERITED if labels.get("inherited") else 1.0,
            "effective_weight": round(w, 3),
        })

    observed_turn_ratio = rarity_detail.get("observed_turn_ratio")
    inherited_turn_ratio = rarity_detail.get("inherited_turn_ratio")
    rarity_confidence = rarity_detail.get("rarity_confidence")

    return {
        "conversation_id": source_id,
        "conversation_key": build_conversation_key(source_id, source_file),
        "source_file": source_file,
        "turn_count": turn_count,
        "conv_value": round(conv_value, 2),
        "conv_selection": None,  # filled by compute_conv_selection_scores
        "peak_complexity": peak_complexity,
        "conv_rarity": round(conv_rarity, 2) if conv_rarity is not None else None,
        "observed_turn_ratio": round(observed_turn_ratio, 2) if observed_turn_ratio is not None else None,
        "inherited_turn_ratio": round(inherited_turn_ratio, 2) if inherited_turn_ratio is not None else None,
        "rarity_confidence": round(rarity_confidence, 2) if rarity_confidence is not None else None,
        "thinking_mode": thinking_mode,
        "merged_labels": merged_labels,
        "detail": {
            "q_base": round(q_base, 2),
            "penalty": round(penalty, 3),
            "quality_floor": quality_floor,
            "negative_flags": neg_flags,
            "pure_quality": round(pure_quality, 2) if pure_quality is not None else None,
            "score_confidence": round(score_conf, 2) if score_conf is not None else None,
            "observed_turns": rarity_detail.get("observed_turns"),
            "inherited_turns": rarity_detail.get("inherited_turns"),
            "observed_weight_ratio": round(rarity_detail["observed_weight_ratio"], 3) if "observed_weight_ratio" in rarity_detail else None,
            "coverage_confidence": round(rarity_detail["coverage_confidence"], 3) if "coverage_confidence" in rarity_detail else None,
            "rarity_mean": round(rarity_detail["rarity_mean"], 2) if "rarity_mean" in rarity_detail else None,
            "rarity_peak": round(rarity_detail["rarity_peak"], 2) if "rarity_peak" in rarity_detail else None,
            "rarity_diversity_bonus": round(rarity_detail["rarity_diversity_bonus"], 3) if "rarity_diversity_bonus" in rarity_detail else None,
            "label_signature_count": rarity_detail.get("label_signature_count"),
            "conv_intra_class_rank": None,  # filled later
            "selection_confidence": None,  # filled later
        },
        "slices": slice_details,
    }


# ─── Conversation-level selection scores ─────────────────

_CONV_SELECTION_DIMS = ["intent", "language", "domain", "concept", "task",
                        "agentic", "constraint", "context", "difficulty"]


def compute_conv_selection_scores(records):
    """Compute per-tag percentile of pure_quality → conv_selection.

    Similar to sample-level selection scores but operates on conversation
    records. Uses Bayesian shrinkage to handle small tag groups.
    """
    if not records:
        return

    min_group_size = max(1, SELECTION_MIN_GROUP_SIZE)
    smoothing_prior = max(1, SELECTION_SMOOTHING_PRIOR // 5)  # lighter prior for fewer records
    rarity_weights = RARITY_WEIGHTS
    intra_weight = SELECTION_INTRA_WEIGHT

    # Global percentile (needed for Bayesian shrinkage)
    valid_pq = []
    for i, rec in enumerate(records):
        pq = (rec.get("detail") or {}).get("pure_quality")
        if pq is not None:
            valid_pq.append((i, pq))

    global_percentiles = _compute_percentiles(valid_pq)

    # Per-record percentiles across dimensions
    dim_percentiles = [{} for _ in records]

    for dim in _CONV_SELECTION_DIMS:
        tag_groups = defaultdict(list)
        for i, rec in enumerate(records):
            pq = (rec.get("detail") or {}).get("pure_quality")
            if pq is None:
                continue
            labels = rec.get("merged_labels") or {}
            tags = labels.get(dim)
            if tags is None:
                continue
            if isinstance(tags, list):
                for t in tags:
                    tag_groups[t].append((i, pq))
            else:
                tag_groups[tags].append((i, pq))

        for tag, members in tag_groups.items():
            if len(members) < min_group_size:
                continue
            n = len(members)
            shrinkage = n / (n + smoothing_prior)
            tag_percentiles = _compute_percentiles(members)
            for idx, _pq in members:
                tag_pct = tag_percentiles[idx]
                g_pct = global_percentiles.get(idx, 0.5)
                percentile = shrinkage * tag_pct + (1 - shrinkage) * g_pct
                if dim in dim_percentiles[idx]:
                    prev, cnt = dim_percentiles[idx][dim]
                    dim_percentiles[idx][dim] = (prev + percentile, cnt + 1)
                else:
                    dim_percentiles[idx][dim] = (percentile, 1)

    # Fuse into intra_class_rank per record
    quality_weight = SELECTION_QUALITY_WEIGHT
    rarity_fuse_weight = 1.0 - intra_weight - quality_weight

    for i, rec in enumerate(records):
        percs = dim_percentiles[i]
        if percs:
            total_w = 0.0
            total_v = 0.0
            for dim, (pct_sum, pct_cnt) in percs.items():
                w = rarity_weights.get(dim, 1.0)
                total_w += w
                total_v += w * (pct_sum / pct_cnt)  # arithmetic mean percentile
            if total_w <= 0:
                continue
            intra_rank = total_v / total_w
            intra_rank_scaled = 1.0 + 9.0 * intra_rank  # scale to 1-10
        else:
            global_pct = global_percentiles.get(i)
            if global_pct is None:
                continue
            intra_rank_scaled = 1.0 + 9.0 * global_pct

        # Absolute quality component
        pq = (rec.get("detail") or {}).get("pure_quality")
        pq_scaled = max(1.0, min(10.0, pq)) if pq is not None else 5.5

        # Fuse with rarity and quality
        rarity = rec.get("conv_rarity")
        if rarity is not None:
            conv_selection = (intra_weight * intra_rank_scaled +
                              quality_weight * pq_scaled +
                              rarity_fuse_weight * rarity)
        else:
            conv_selection = ((intra_weight + rarity_fuse_weight * 0.5) * intra_rank_scaled +
                              (quality_weight + rarity_fuse_weight * 0.5) * pq_scaled)

        selection_conf = min(
            1.0,
            0.5 * max(0.0, min(1.0, (rec.get("rarity_confidence") or 1.0))) +
            0.5 * max(0.0, min(1.0, ((rec.get("detail") or {}).get("score_confidence") or 1.0))),
        )
        conv_selection = apply_score_confidence(conv_selection, selection_conf)
        conv_selection = max(1.0, min(10.0, conv_selection))
        rec["conv_selection"] = round(conv_selection, 2)
        rec["detail"]["conv_intra_class_rank"] = round(intra_rank_scaled, 2)
        rec["detail"]["selection_confidence"] = round(selection_conf, 2)


# ─── Top-level entry points ─────────────────────────────

def aggregate_conversations(samples):
    """Top-level: group → aggregate each → compute selection → return list.

    Args:
        samples: list of scored sample dicts (from Pass 2 output)

    Returns:
        list of conversation record dicts, or empty list if no multi-turn.
    """
    groups = group_by_conversation(samples)
    if not groups:
        return []

    records = []
    for source_id, slices in groups.items():
        rec = aggregate_conversation(source_id, slices)
        if rec is not None:
            records.append(rec)

    if records:
        compute_conv_selection_scores(records)

    return records


def write_conversation_scores(records, path):
    """Write conversation_scores.json."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=2)
