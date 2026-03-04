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
    VALUE_WEIGHTS,
    RARITY_WEIGHTS,
    SELECTION_INTRA_WEIGHT,
    SELECTION_MIN_GROUP_SIZE,
    SELECTION_SMOOTHING_PRIOR,
    KNOWN_FLAGS_NEGATIVE,
)


# ─── Grouping ────────────────────────────────────────────

def group_by_conversation(samples):
    """Group samples by conversation, filter to multi-turn only.

    Returns dict {source_id: [slices sorted by turn_index]}.
    Only includes conversations where total_turns > 1.
    """
    groups = defaultdict(list)
    for s in samples:
        meta = s.get("metadata") or {}
        source_id = meta.get("source_id")
        total_turns = meta.get("total_turns", 1)
        if source_id and total_turns > 1:
            groups[source_id].append(s)

    # Sort each group by turn_index
    for source_id in groups:
        groups[source_id].sort(
            key=lambda s: (s.get("metadata") or {}).get("turn_index", 0)
        )

    return dict(groups)


# ─── Weighting helpers ───────────────────────────────────

def _position_weight(i, n):
    """Linear position weight: 1.0 for first turn → 3.0 for last turn.

    Args:
        i: 0-based turn index within the conversation
        n: total number of slices in the conversation
    """
    if n <= 1:
        return 1.0
    return 1.0 + 2.0 * i / (n - 1)


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
_LABEL_META_KEYS = {"confidence", "inherited", "inherited_from"}


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

    # Quality floor penalty: consistent with compute_value_score
    quality_overall = components.get("quality")
    if quality_overall is not None and quality_overall < 4:
        pq *= 0.7

    return pq


# ─── Per-conversation aggregation ────────────────────────

def aggregate_conversation(source_id, slices):
    """Aggregate scores for a single conversation.

    Returns a record dict, or None if insufficient data (e.g. single slice).
    """
    if len(slices) < 2:
        return None

    weights = _effective_weights(slices)

    # Weighted average of value_score → q_base
    q_base = _weighted_average(
        slices, weights,
        lambda s: (s.get("value") or {}).get("value_score")
    )
    if q_base is None:
        return None

    # Penalty
    penalty, quality_floor, neg_flags = _compute_penalty(slices)

    # conv_value = q_base × penalty, clamped to [1, 10]
    conv_value = max(1.0, min(10.0, q_base * penalty))

    # Peak complexity
    complexities = []
    for s in slices:
        v = s.get("value") or {}
        c = (v.get("complexity") or {}).get("overall")
        if c is not None:
            complexities.append(c)
    peak_complexity = max(complexities) if complexities else None

    # Weighted rarity
    conv_rarity = _weighted_average(
        slices, weights,
        lambda s: ((s.get("value") or {}).get("rarity") or {}).get("score")
    )

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

    return {
        "conversation_id": source_id,
        "turn_count": turn_count,
        "conv_value": round(conv_value, 2),
        "conv_selection": None,  # filled by compute_conv_selection_scores
        "peak_complexity": peak_complexity,
        "conv_rarity": round(conv_rarity, 2) if conv_rarity is not None else None,
        "thinking_mode": thinking_mode,
        "merged_labels": merged_labels,
        "detail": {
            "q_base": round(q_base, 2),
            "penalty": round(penalty, 3),
            "quality_floor": quality_floor,
            "negative_flags": neg_flags,
            "pure_quality": round(pure_quality, 2) if pure_quality is not None else None,
            "conv_intra_class_rank": None,  # filled later
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

    global_percentiles = {}
    if valid_pq:
        sorted_global = sorted(valid_pq, key=lambda x: x[1])
        n_global = len(sorted_global)
        for rank, (idx, _pq) in enumerate(sorted_global):
            global_percentiles[idx] = rank / max(n_global - 1, 1)

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
            sorted_members = sorted(members, key=lambda x: x[1])
            n = len(sorted_members)
            shrinkage = n / (n + smoothing_prior)
            for rank, (idx, _pq) in enumerate(sorted_members):
                tag_pct = rank / max(n - 1, 1)
                g_pct = global_percentiles.get(idx, 0.5)
                percentile = shrinkage * tag_pct + (1 - shrinkage) * g_pct
                if dim in dim_percentiles[idx]:
                    prev, cnt = dim_percentiles[idx][dim]
                    dim_percentiles[idx][dim] = (prev + percentile, cnt + 1)
                else:
                    dim_percentiles[idx][dim] = (percentile, 1)

    # Fuse into intra_class_rank per record
    for i, rec in enumerate(records):
        percs = dim_percentiles[i]
        if not percs:
            continue

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

        # Fuse with rarity
        rarity = rec.get("conv_rarity")
        if rarity is not None:
            conv_selection = (intra_weight * intra_rank_scaled +
                              (1 - intra_weight) * rarity)
        else:
            conv_selection = intra_rank_scaled

        conv_selection = max(1.0, min(10.0, conv_selection))
        rec["conv_selection"] = round(conv_selection, 2)
        rec["detail"]["conv_intra_class_rank"] = round(intra_rank_scaled, 2)


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
