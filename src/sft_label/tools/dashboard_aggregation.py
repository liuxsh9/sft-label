from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

from sft_label.config import FILE_RANKING_KEEP_RATE_THRESHOLD, FILE_RANKING_KEEP_RATE_THRESHOLDS
from sft_label.conversation import (
    _effective_weights,
    _merge_labels,
    _weighted_average,
    aggregate_conversations,
    build_conversation_key,
    group_by_conversation,
    is_conversation_object,
)
from sft_label.label_extensions_stats import build_extension_dashboard_payload
from sft_label.labels import is_usable_labels
from sft_label.prompts import TAG_POOLS
from sft_label.score_confidence import score_confidence

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
_SINGLE_SELECT_DIMS = {"intent", "difficulty", "context"}
_MULTI_SELECT_DIMS = {"language", "domain", "concept", "task", "agentic", "constraint"}
_PASS2_TAG_DIMS = ["intent", "difficulty", "domain", "concept", "task", "agentic", "constraint", "context"]


def load_data_file(path: str | Path | None) -> list[dict]:
    return list(iter_data_file(path))


def iter_data_file(path: str | Path | None):
    if not path:
        return
    data_path = Path(path)
    if not data_path.exists():
        return
    if data_path.suffix == ".jsonl":
        with open(data_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        row = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if isinstance(row, dict):
                        yield row
        return
    decoder = json.JSONDecoder()
    started = False
    eof = False
    buffer = ""
    with open(data_path, encoding="utf-8") as f:
        while True:
            if not eof:
                chunk = f.read(65536)
                if chunk:
                    buffer += chunk
                else:
                    eof = True

            while True:
                buffer = buffer.lstrip()
                if not buffer:
                    break

                if not started:
                    if buffer[0] != "[":
                        return
                    started = True
                    buffer = buffer[1:]
                    continue

                if buffer[0] == "]":
                    return
                if buffer[0] == ",":
                    buffer = buffer[1:]
                    continue

                try:
                    row, consumed = decoder.raw_decode(buffer)
                except json.JSONDecodeError:
                    if eof:
                        return
                    break

                buffer = buffer[consumed:]
                if isinstance(row, dict):
                    yield row

            if eof:
                return


def _collect_unmapped_examples(
    samples: list[dict],
    *,
    per_tag_limit: int = 2,
) -> dict[tuple[str, str], list[dict]]:
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


def _group_unmapped_counts(stats: dict) -> dict[str, list[dict]]:
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


def _build_unmapped_details(samples: list[dict], stats: dict) -> dict:
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


def _cross_matrix_from_labels(label_rows: list[dict]) -> dict:
    cross = Counter()
    for labels in label_rows:
        cross[(labels.get("intent", "?"), labels.get("difficulty", "?"))] += 1

    intents = sorted({key[0] for key in cross})
    difficulty_order = ["beginner", "intermediate", "upper-intermediate", "advanced", "expert"]
    return {
        "rows": intents,
        "cols": [dim for dim in difficulty_order if any(cross.get((intent, dim), 0) for intent in intents)],
        "data": {f"{intent}|{diff}": cross.get((intent, diff), 0) for intent in intents for diff in difficulty_order},
    }


def _coverage_from_distributions(distributions: dict) -> dict:
    coverage = {}
    for dim in DIMENSIONS:
        pool = set(TAG_POOLS.get(dim, []))
        used = set((distributions.get(dim) or {}).keys())
        coverage[dim] = {
            "pool_size": len(pool),
            "used": len(used & pool),
            "unused": sorted(pool - used),
            "rate": len(used & pool) / len(pool) if pool else 0,
        }
    return coverage


def _empty_label_mode(stats: dict) -> dict:
    distributions = stats.get("tag_distributions", {})
    return {
        "total": stats.get("total_samples", 0),
        "input_file": stats.get("input_file") or stats.get("input_path", ""),
        "distributions": distributions,
        "unmapped_details": _build_unmapped_details([], stats),
        "confidence_stats": stats.get("confidence_stats", {}),
        "conf_matrix": [],
        "coverage": _coverage_from_distributions(distributions),
        "cross_matrix": _cross_matrix_from_labels([]) if not stats.get("cross_matrix") else {
            "rows": sorted({key.split("|", 1)[0] for key in stats.get("cross_matrix", {}) if "|" in key}),
            "cols": [dim for dim in ["beginner", "intermediate", "upper-intermediate", "advanced", "expert"]
                     if any((stats.get("cross_matrix", {}) or {}).get(f"{row}|{dim}", 0)
                            for row in sorted({key.split("|", 1)[0] for key in stats.get("cross_matrix", {}) if "|" in key}))],
            "data": stats.get("cross_matrix", {}),
        },
        "overview": {
            "success_rate": stats.get("success_rate", 0),
            "total_tokens": stats.get("total_tokens", 0),
            "arbitrated_rate": stats.get("arbitrated_rate", 0),
            "unmapped_unique": stats.get("unmapped_unique_count", 0),
            "sparse_labeled": stats.get("sparse_labeled", 0),
            "sparse_inherited": stats.get("sparse_inherited", 0),
        },
        "unit_label": "samples",
        "mode_id": "sample",
    }


def _aggregate_dimension_confidence(slices: list[dict], dim: str):
    if dim in _SINGLE_SELECT_DIMS:
        for sample in reversed(slices):
            labels = sample.get("labels") or {}
            confidence = labels.get("confidence") or {}
            score = confidence.get(dim)
            if isinstance(score, (int, float)):
                return round(float(score), 3)

    weighted = []
    fallback = []
    for sample in slices:
        labels = sample.get("labels") or {}
        confidence = labels.get("confidence") or {}
        score = confidence.get(dim)
        if not isinstance(score, (int, float)):
            continue
        value = labels.get(dim)
        if dim in _MULTI_SELECT_DIMS and not value:
            continue
        item = (float(score), 0.7 if labels.get("inherited") else 1.0)
        if labels.get("inherited"):
            fallback.append(item)
        else:
            weighted.append(item)

    candidates = weighted or fallback
    if not candidates:
        return None
    total_weight = sum(weight for _score, weight in candidates) or 1.0
    return round(sum(score * weight for score, weight in candidates) / total_weight, 3)


def _merge_unmapped(slices: list[dict]) -> list[dict]:
    seen = set()
    merged = []
    for sample in slices:
        labels = sample.get("labels") or {}
        for item in labels.get("unmapped") or []:
            if isinstance(item, dict):
                dim = str(item.get("dimension", "unknown"))
                value = str(item.get("value", "?"))
                key = (dim, value)
                if key in seen:
                    continue
                seen.add(key)
                merged.append({"dimension": dim, "value": value})
            else:
                value = str(item)
                key = ("unknown", value)
                if key in seen:
                    continue
                seen.add(key)
                merged.append({"dimension": "unknown", "value": value})
    return merged


def build_pass1_conversation_units(samples: list[dict]) -> list[dict]:
    if not samples:
        return []

    groups = group_by_conversation(samples)
    set(groups)
    units = []

    for sample in samples:
        meta = sample.get("metadata") or {}
        key = build_conversation_key(meta.get("source_id"), meta.get("source_file"))
        if key and is_conversation_object(meta):
            continue
        units.append(sample)

    for conv_key, slices in groups.items():
        merged_labels = _merge_labels(slices)
        for dim in DIMENSIONS:
            conf = _aggregate_dimension_confidence(slices, dim)
            if conf is not None:
                merged_labels.setdefault("confidence", {})[dim] = conf
        merged_unmapped = _merge_unmapped(slices)
        if merged_unmapped:
            merged_labels["unmapped"] = merged_unmapped
        last = slices[-1]
        meta = dict(last.get("metadata") or {})
        meta["source_id"] = meta.get("source_id") or conv_key
        meta["conversation_key"] = conv_key
        unit = {
            "id": conv_key,
            "metadata": meta,
            "conversations": last.get("conversations") or [],
            "labels": merged_labels,
        }
        units.append(unit)

    return units


def build_pass1_conversation_units_from_iter(samples_iter) -> list[dict]:
    units = []
    grouped: dict[str, dict] = {}
    group_order: list[str] = []

    def _update_group(state: dict, sample: dict) -> None:
        labels = sample.get("labels") or {}
        meta = sample.get("metadata") or {}
        state["last"] = sample

        merged = state["merged_labels"]
        for dim, val in labels.items():
            if dim in {"confidence", "unmapped", "inherited", "inherited_from"}:
                continue
            if dim in _SINGLE_SELECT_DIMS:
                merged[dim] = val
            elif dim in _MULTI_SELECT_DIMS:
                existing = merged.setdefault(dim, [])
                if not isinstance(existing, list):
                    existing = [existing]
                if isinstance(val, list):
                    for item in val:
                        if item not in existing:
                            existing.append(item)
                elif val not in existing:
                    existing.append(val)
                merged[dim] = existing
            else:
                merged[dim] = val

        if labels.get("inherited"):
            merged["inherited"] = True
            if labels.get("inherited_from"):
                merged["inherited_from"] = labels.get("inherited_from")

        confidence = labels.get("confidence") or {}
        for dim in DIMENSIONS:
            score = confidence.get(dim)
            if not isinstance(score, (int, float)):
                continue
            if dim in _SINGLE_SELECT_DIMS:
                state["single_conf"][dim] = round(float(score), 3)
                continue
            value = labels.get(dim)
            if dim in _MULTI_SELECT_DIMS and not value:
                continue
            bucket_name = "fallback" if labels.get("inherited") else "observed"
            bucket = state["multi_conf"][dim][bucket_name]
            weight = 0.7 if labels.get("inherited") else 1.0
            bucket["sum"] += float(score) * weight
            bucket["weight"] += weight

        for item in labels.get("unmapped") or []:
            if isinstance(item, dict):
                dim = str(item.get("dimension", "unknown"))
                value = str(item.get("value", "?"))
                key = (dim, value)
                if key not in state["unmapped_seen"]:
                    state["unmapped_seen"].add(key)
                    state["unmapped"].append({"dimension": dim, "value": value})
            else:
                value = str(item)
                key = ("unknown", value)
                if key not in state["unmapped_seen"]:
                    state["unmapped_seen"].add(key)
                    state["unmapped"].append({"dimension": "unknown", "value": value})

        state["meta"] = meta

    for sample in samples_iter:
        meta = sample.get("metadata") or {}
        conv_key = build_conversation_key(meta.get("source_id"), meta.get("source_file"))
        if conv_key and is_conversation_object(meta):
            if conv_key not in grouped:
                grouped[conv_key] = {
                    "merged_labels": {},
                    "single_conf": {},
                    "multi_conf": {
                        dim: {
                            "observed": {"sum": 0.0, "weight": 0.0},
                            "fallback": {"sum": 0.0, "weight": 0.0},
                        }
                        for dim in DIMENSIONS
                    },
                    "unmapped": [],
                    "unmapped_seen": set(),
                    "last": None,
                    "meta": {},
                }
                group_order.append(conv_key)
            _update_group(grouped[conv_key], sample)
        else:
            units.append(sample)

    for conv_key in group_order:
        state = grouped[conv_key]
        merged_labels = dict(state["merged_labels"])
        confidence_payload = {}
        for dim, value in state["single_conf"].items():
            confidence_payload[dim] = value
        for dim, buckets in state["multi_conf"].items():
            primary = buckets["observed"]
            fallback = buckets["fallback"]
            chosen = primary if primary["weight"] > 0 else fallback
            if chosen["weight"] > 0:
                confidence_payload[dim] = round(chosen["sum"] / chosen["weight"], 3)
        if confidence_payload:
            merged_labels["confidence"] = confidence_payload
        if state["unmapped"]:
            merged_labels["unmapped"] = state["unmapped"]

        last = state["last"] or {}
        meta = dict(last.get("metadata") or {})
        meta["source_id"] = meta.get("source_id") or conv_key
        meta["conversation_key"] = conv_key
        units.append(
            {
                "id": conv_key,
                "metadata": meta,
                "conversations": last.get("conversations") or [],
                "labels": merged_labels,
            }
        )

    return units


def build_pass1_conversation_mode(samples: list[dict], stats: dict) -> dict:
    conversation_units = build_pass1_conversation_units(samples)
    return _compute_label_mode(
        conversation_units,
        conversation_units,
        stats,
        mode_id="conversation",
        unit_label="units",
    )


def build_pass1_conversation_mode_from_iter(samples_iter, stats: dict) -> dict:
    conversation_units = build_pass1_conversation_units_from_iter(samples_iter)
    return _compute_label_mode(
        conversation_units,
        conversation_units,
        stats,
        mode_id="conversation",
        unit_label="units",
    )


def _compute_label_mode(all_units: list[dict], dist_units: list[dict], stats: dict, *, mode_id: str, unit_label: str) -> dict:
    if not all_units and not dist_units:
        empty = _empty_label_mode(stats)
        empty["mode_id"] = mode_id
        empty["unit_label"] = unit_label
        return empty

    total = len(all_units)
    success = sum(1 for sample in all_units if is_usable_labels(sample.get("labels")))

    distributions = {}
    for dim in DIMENSIONS:
        dist = {}
        for sample in dist_units:
            labels = sample.get("labels") or {}
            if not is_usable_labels(labels):
                continue
            value = labels.get(dim, [])
            if isinstance(value, list):
                for item in value:
                    dist[item] = dist.get(item, 0) + 1
            elif value:
                dist[value] = dist.get(value, 0) + 1
        distributions[dim] = dict(sorted(dist.items(), key=lambda item: (-item[1], item[0])))

    all_unmapped = {}
    for sample in dist_units:
        labels = sample.get("labels") or {}
        if not is_usable_labels(labels):
            continue
        for item in labels.get("unmapped", []):
            key = f"{item.get('dimension', '?')}:{item.get('value', '?')}" if isinstance(item, dict) else str(item)
            all_unmapped[key] = all_unmapped.get(key, 0) + 1

    conf_stats = {}
    conf_matrix = []
    for sample in all_units:
        labels = sample.get("labels") or {}
        conf = labels.get("confidence") or {}
        if not is_usable_labels(labels):
            continue
        conf_matrix.append({
            "id": sample.get("id", "?"),
            "values": {dim: conf.get(dim, 0) for dim in DIMENSIONS},
        })
    for dim in DIMENSIONS:
        scores = []
        for sample in dist_units:
            labels = sample.get("labels") or {}
            if not is_usable_labels(labels):
                continue
            score = (labels.get("confidence") or {}).get(dim)
            if isinstance(score, (int, float)):
                scores.append(float(score))
        if scores:
            conf_stats[dim] = {
                "mean": round(sum(scores) / len(scores), 3),
                "min": round(min(scores), 3),
                "max": round(max(scores), 3),
                "below_threshold": sum(1 for score in scores if score < 0.7),
                "count": len(scores),
            }

    label_rows = [sample.get("labels") or {} for sample in dist_units if is_usable_labels(sample.get("labels"))]
    cross_matrix = _cross_matrix_from_labels(label_rows)
    distributions_total = sum(1 for sample in dist_units if is_usable_labels(sample.get("labels")))

    synthetic_stats = {
        "unmapped_tags": dict(sorted(all_unmapped.items(), key=lambda item: (-item[1], item[0]))),
    }
    return {
        "total": total,
        "distribution_total": distributions_total,
        "input_file": stats.get("input_file") or stats.get("input_path", ""),
        "distributions": distributions,
        "unmapped_details": _build_unmapped_details(dist_units, synthetic_stats),
        "confidence_stats": conf_stats,
        "conf_matrix": conf_matrix,
        "coverage": _coverage_from_distributions(distributions),
        "cross_matrix": cross_matrix,
        "overview": {
            "success_rate": round(success / max(total, 1), 4),
            "total_tokens": stats.get("total_tokens", 0),
            "arbitrated_rate": stats.get("arbitrated_rate", 0),
            "unmapped_unique": len(all_unmapped),
            "sparse_labeled": stats.get("sparse_labeled", 0),
            "sparse_inherited": stats.get("sparse_inherited", 0),
        },
        "unit_label": unit_label,
        "mode_id": mode_id,
    }


def build_pass1_viz(samples: list[dict], stats: dict) -> dict:
    sample_all = samples
    sample_dist = [
        sample for sample in samples
        if is_usable_labels(sample.get("labels")) and not (sample.get("labels") or {}).get("inherited")
    ]
    sample_mode = _compute_label_mode(sample_all, sample_dist, stats, mode_id="sample", unit_label="samples")
    conversation_mode = build_pass1_conversation_mode(samples, stats)
    payload = dict(sample_mode)
    payload["modes"] = {
        "sample": sample_mode,
        "conversation": conversation_mode,
    }
    extension_payload = build_extension_dashboard_payload(samples, extension_stats=stats.get("extension_stats"))
    if extension_payload:
        payload["extensions"] = extension_payload
    return payload


def _percentiles(values: list[float]) -> dict:
    if not values:
        return {}
    ordered = sorted(values)
    n = len(ordered)

    def _pick(p: float) -> float:
        idx = min(max(int(round((n - 1) * p)), 0), n - 1)
        return ordered[idx]

    return {
        "mean": round(sum(ordered) / n, 2),
        "min": ordered[0],
        "max": ordered[-1],
        "p50": _pick(0.5),
    }


def _histogram_bins(values: list[float]) -> list[int]:
    bins = [0] * 10
    for value in values:
        if isinstance(value, (int, float)) and 1 <= value <= 10:
            bins[min(int(value) - 1, 9)] += 1
    return bins


_PASS2_EXTENSION_ONLY_KEYS = {
    "extension_rarity_score",
    "rarity_v2_score",
    "value_score_v2",
    "selection_score_v2",
}


def _has_numeric_distribution(distribution: dict | None) -> bool:
    if not isinstance(distribution, dict):
        return False
    return any(isinstance(value, (int, float)) for value in distribution.values())


def _has_nonzero_histogram(histogram: list[int] | None) -> bool:
    if not isinstance(histogram, list):
        return False
    return any(isinstance(value, (int, float)) and value > 0 for value in histogram)


def _prune_empty_pass2_extension_only_fields(payload: dict) -> dict:
    score_distributions = payload.get("score_distributions")
    histograms = payload.get("histograms")

    metric_presence = {
        key: (
            _has_numeric_distribution((score_distributions or {}).get(key))
            or _has_nonzero_histogram((histograms or {}).get(key))
        )
        for key in _PASS2_EXTENSION_ONLY_KEYS
    }

    if isinstance(score_distributions, dict):
        payload["score_distributions"] = {
            key: value
            for key, value in score_distributions.items()
            if key not in _PASS2_EXTENSION_ONLY_KEYS or metric_presence.get(key)
        }

    if isinstance(histograms, dict):
        payload["histograms"] = {
            key: value
            for key, value in histograms.items()
            if key not in _PASS2_EXTENSION_ONLY_KEYS or metric_presence.get(key)
        }

    overview = payload.get("overview")
    if isinstance(overview, dict):
        if not metric_presence.get("extension_rarity_score"):
            overview["mean_extension_rarity"] = None
            overview["extension_rarity_mode"] = None
            overview["extension_baseline_source"] = None
        if not metric_presence.get("rarity_v2_score"):
            overview["mean_rarity_v2"] = None
        if not metric_presence.get("value_score_v2"):
            overview["mean_value_v2"] = None
        if not metric_presence.get("selection_score_v2"):
            overview["mean_selection_v2"] = None

    per_file_summary = payload.get("per_file_summary")
    if isinstance(per_file_summary, list):
        scrubbed = []
        for row in per_file_summary:
            if not isinstance(row, dict):
                scrubbed.append(row)
                continue
            clean = dict(row)
            if not metric_presence.get("extension_rarity_score"):
                clean.pop("mean_extension_rarity", None)
            if not metric_presence.get("rarity_v2_score"):
                clean.pop("mean_rarity_v2", None)
            if not metric_presence.get("selection_score_v2"):
                clean.pop("mean_selection_v2", None)
            scrubbed.append(clean)
        payload["per_file_summary"] = scrubbed

    return payload


def _filter_pass2_extension_only_fields(payload: dict, *, mode_id: str) -> dict:
    if mode_id == "sample":
        return _prune_empty_pass2_extension_only_fields(payload)

    score_distributions = payload.get("score_distributions")
    if isinstance(score_distributions, dict):
        payload["score_distributions"] = {
            key: value for key, value in score_distributions.items() if key not in _PASS2_EXTENSION_ONLY_KEYS
        }

    histograms = payload.get("histograms")
    if isinstance(histograms, dict):
        payload["histograms"] = {
            key: value for key, value in histograms.items() if key not in _PASS2_EXTENSION_ONLY_KEYS
        }

    overview = payload.get("overview")
    if isinstance(overview, dict):
        for key in (
            "mean_value_v2",
            "mean_selection_v2",
            "mean_extension_rarity",
            "mean_rarity_v2",
            "extension_rarity_mode",
            "extension_baseline_source",
        ):
            overview[key] = None

    per_file_summary = payload.get("per_file_summary")
    if isinstance(per_file_summary, list):
        scrubbed = []
        for row in per_file_summary:
            if not isinstance(row, dict):
                scrubbed.append(row)
                continue
            clean = dict(row)
            for key in ("mean_extension_rarity", "mean_rarity_v2", "mean_selection_v2"):
                clean.pop(key, None)
            scrubbed.append(clean)
        payload["per_file_summary"] = scrubbed

    return payload


def _stats_only_pass2_mode(stats: dict, *, mode_id: str, unit_label: str, per_file_summary: list[dict] | None = None) -> dict:
    score_distributions = {
        key: value
        for key, value in (stats.get("score_distributions") or {}).items()
        if isinstance(value, dict)
    }
    if mode_id != "sample":
        for key in _PASS2_EXTENSION_ONLY_KEYS:
            score_distributions.pop(key, None)
    rarity_distribution = score_distributions.get("rarity_score") or {}
    extension_distribution = score_distributions.get("extension_rarity_score") or {}
    rarity_v2_distribution = score_distributions.get("rarity_v2_score") or {}
    extension_enabled = mode_id == "sample"
    payload = {
        "overview": {
            "total_scored": stats.get("total_scored", 0),
            "total_failed": stats.get("total_failed", 0),
            "input_file": stats.get("input_file") or stats.get("input_path", ""),
            "mean_value": (score_distributions.get("value_score") or {}).get("mean", 0),
            "mean_selection": (score_distributions.get("selection_score") or {}).get("mean", 0),
            "mean_value_v2": (score_distributions.get("value_score_v2") or {}).get("mean", 0) if extension_enabled else None,
            "mean_selection_v2": (score_distributions.get("selection_score_v2") or {}).get("mean", 0) if extension_enabled else None,
            "mean_complexity": (score_distributions.get("complexity_overall") or {}).get("mean", 0),
            "mean_quality": (score_distributions.get("quality_overall") or {}).get("mean", 0),
            "median_rarity": rarity_distribution.get("p50", rarity_distribution.get("mean", 0)),
            "mean_extension_rarity": extension_distribution.get("mean", 0) if extension_enabled else None,
            "mean_rarity_v2": rarity_v2_distribution.get("mean", 0) if extension_enabled else None,
            "mean_confidence": (score_distributions.get("confidence") or {}).get("mean", 0),
            "total_tokens": stats.get("total_tokens", 0),
            "extension_rarity_mode": ((stats.get("extension_rarity_config") or {}).get("mode")) if extension_enabled else None,
            "extension_baseline_source": ((stats.get("extension_rarity_config") or {}).get("baseline_source")) if extension_enabled else None,
        },
        "selection_thresholds": stats.get("selection_thresholds", {}),
        "score_distributions": score_distributions,
        "sub_score_means": stats.get("sub_score_means", {}),
        "value_by_tag": stats.get("value_by_tag", {}),
        "selection_by_tag": stats.get("selection_by_tag", {}),
        "thinking_mode_stats": stats.get("thinking_mode_stats", {}),
        "flag_counts": stats.get("flag_counts", {}),
        "flag_value_impact": stats.get("flag_value_impact", {}),
        "coverage_at_thresholds": stats.get("coverage_at_thresholds", {}),
        "weights_used": stats.get("weights_used", {}),
        "per_file_summary": per_file_summary or stats.get("per_file_summary", []),
        "histograms": (
            {key: value for key, value in (stats.get("histograms") or {}).items() if key not in _PASS2_EXTENSION_ONLY_KEYS}
            if mode_id != "sample" else stats.get("histograms", {})
        ),
        "confidence_histogram": stats.get("confidence_histogram", [0] * 10),
        "selection_vs_value": stats.get("selection_vs_value", {}),
        "unit_label": unit_label,
        "mode_id": mode_id,
    }
    return _filter_pass2_extension_only_fields(payload, mode_id=mode_id)


def _conversation_units_missing_subscore_metrics(units: list[dict]) -> bool:
    if not units:
        return False
    return all(
        unit.get("quality_overall") is None and unit.get("reasoning_overall") is None
        for unit in units
    )


def _backfill_conversation_mode_from_stats(mode: dict, stats: dict) -> None:
    score_distributions = stats.get("score_distributions") or {}
    mode_distributions = mode.setdefault("score_distributions", {})
    for key in ("complexity_overall", "quality_overall", "reasoning_overall", "confidence"):
        if not isinstance(mode_distributions.get(key), dict) or "mean" not in mode_distributions.get(key, {}):
            value = score_distributions.get(key)
            if isinstance(value, dict):
                mode_distributions[key] = value

    mode_sub_scores = mode.setdefault("sub_score_means", {})
    if not isinstance(mode_sub_scores.get("complexity"), dict):
        mode_sub_scores["complexity"] = {}
    if not isinstance(mode_sub_scores.get("quality"), dict):
        mode_sub_scores["quality"] = {}
    if not isinstance(mode_sub_scores.get("reasoning"), dict):
        mode_sub_scores["reasoning"] = {}
    if mode_sub_scores["complexity"].get("overall") is None:
        complexity_mean = (mode_distributions.get("complexity_overall") or {}).get("mean")
        if complexity_mean is not None:
            mode_sub_scores["complexity"]["overall"] = complexity_mean
    if mode_sub_scores["quality"].get("overall") is None:
        quality_mean = (mode_distributions.get("quality_overall") or {}).get("mean")
        if quality_mean is not None:
            mode_sub_scores["quality"]["overall"] = quality_mean
    if mode_sub_scores["reasoning"].get("overall") is None:
        reasoning_mean = (mode_distributions.get("reasoning_overall") or {}).get("mean")
        if reasoning_mean is not None:
            mode_sub_scores["reasoning"]["overall"] = reasoning_mean

    overview = mode.setdefault("overview", {})
    for overview_key, distribution_key in (
        ("mean_complexity", "complexity_overall"),
        ("mean_quality", "quality_overall"),
        ("mean_confidence", "confidence"),
    ):
        if overview.get(overview_key) in (None, 0):
            mean_value = (mode_distributions.get(distribution_key) or {}).get("mean")
            if mean_value is not None:
                overview[overview_key] = mean_value

    stats_file_summary = {
        row.get("file"): row
        for row in (stats.get("per_file_summary") or [])
        if isinstance(row, dict) and row.get("file")
    }
    for row in mode.get("per_file_summary") or []:
        if not isinstance(row, dict):
            continue
        stats_row = stats_file_summary.get(row.get("file"))
        if isinstance(stats_row, dict):
            if row.get("mean_quality") in (None, 0):
                stats_mean_quality = stats_row.get("mean_quality")
                if stats_mean_quality is not None:
                    row["mean_quality"] = stats_mean_quality
            if row.get("mean_complexity") in (None, 0):
                stats_mean_complexity = stats_row.get("mean_complexity")
                if stats_mean_complexity is not None:
                    row["mean_complexity"] = stats_mean_complexity
            continue
        if row.get("mean_quality") in (None, 0):
            overview_quality = overview.get("mean_quality")
            if overview_quality is not None:
                row["mean_quality"] = overview_quality
        if row.get("mean_complexity") in (None, 0):
            overview_complexity = overview.get("mean_complexity")
            if overview_complexity is not None:
                row["mean_complexity"] = overview_complexity

    stats_thinking_mode = {
        key: value
        for key, value in (stats.get("thinking_mode_stats") or {}).items()
        if isinstance(value, dict)
    }
    if not mode.get("thinking_mode_stats") and stats_thinking_mode:
        mode["thinking_mode_stats"] = {
            key: dict(value)
            for key, value in stats_thinking_mode.items()
        }
    for mode_key, mode_stats in (mode.get("thinking_mode_stats") or {}).items():
        if not isinstance(mode_stats, dict):
            continue
        stats_mode = stats_thinking_mode.get(mode_key)
        if not isinstance(stats_mode, dict):
            continue
        if mode_stats.get("mean_quality") in (None, 0):
            stats_mean_quality = stats_mode.get("mean_quality")
            if stats_mean_quality is not None:
                mode_stats["mean_quality"] = stats_mean_quality
        if mode_stats.get("mean_reasoning") in (None, 0):
            stats_mean_reasoning = stats_mode.get("mean_reasoning")
            if stats_mean_reasoning is not None:
                mode_stats["mean_reasoning"] = stats_mean_reasoning


def _sample_pass2_unit(sample: dict) -> dict | None:
    value = sample.get("value") or {}
    if not value:
        return None
    metadata = sample.get("metadata") or {}
    total_turns = metadata.get("total_turns")
    trajectory_turn_count = metadata.get("trajectory_turn_count")
    conversations = sample.get("conversations") or []
    turn_count = None
    for candidate in (total_turns, trajectory_turn_count):
        if isinstance(candidate, int) and candidate > 0:
            turn_count = candidate
            break
    if turn_count is None and isinstance(conversations, list) and conversations:
        turn_count = len(conversations)
    if turn_count is None:
        turn_count = 1
    return {
        "id": sample.get("id", ""),
        "labels": sample.get("labels") or {},
        "value_score": value.get("value_score"),
        "selection_score": value.get("selection_score"),
        "value_score_v2": value.get("value_score_v2"),
        "selection_score_v2": value.get("selection_score_v2"),
        "complexity_overall": (value.get("complexity") or {}).get("overall"),
        "quality_overall": (value.get("quality") or {}).get("overall"),
        "reasoning_overall": (value.get("reasoning") or {}).get("overall"),
        "rarity_score": (value.get("rarity") or {}).get("score"),
        "extension_rarity_score": (value.get("rarity_extension") or {}).get("score"),
        "rarity_v2_score": (value.get("rarity_v2") or {}).get("score"),
        "confidence": value.get("confidence"),
        "thinking_mode": value.get("thinking_mode") or (sample.get("metadata") or {}).get("thinking_mode"),
        "flags": value.get("flags") or [],
        "turn_count": turn_count,
        "source_file": metadata.get("source_file"),
    }


def _group_key_for_sample(sample: dict):
    meta = sample.get("metadata") or {}
    return build_conversation_key(meta.get("source_id"), meta.get("source_file"))


def _score_from_slices(slices: list[dict], field: str):
    weights = _effective_weights(slices)
    if field == "confidence":
        return round(_weighted_average(slices, weights, lambda s: score_confidence(s.get("value") or {}, default=1.0)) or 0, 2)
    if field == "complexity_overall":
        return _weighted_average(slices, weights, lambda s: ((s.get("value") or {}).get("complexity") or {}).get("overall"))
    if field == "quality_overall":
        return _weighted_average(slices, weights, lambda s: ((s.get("value") or {}).get("quality") or {}).get("overall"))
    if field == "reasoning_overall":
        return _weighted_average(slices, weights, lambda s: ((s.get("value") or {}).get("reasoning") or {}).get("overall"))
    raise KeyError(field)


def build_pass2_conversation_units(samples: list[dict], conv_records: list[dict] | None = None) -> list[dict]:
    if not samples and not conv_records:
        return []

    if not conv_records and samples:
        conv_records = aggregate_conversations(samples)
    conv_records = conv_records or []

    groups = group_by_conversation(samples)
    units = []
    conv_by_key = {
        item.get("conversation_key") or build_conversation_key(item.get("conversation_id"), item.get("source_file")): item
        for item in conv_records
    }

    for sample in samples:
        meta = sample.get("metadata") or {}
        if is_conversation_object(meta):
            continue
        unit = _sample_pass2_unit(sample)
        if unit:
            units.append(unit)

    for conv_key, slices in groups.items():
        rec = conv_by_key.get(conv_key)
        if rec is None:
            continue
        units.append({
            "id": conv_key,
            "labels": rec.get("merged_labels") or {},
            "value_score": rec.get("conv_value"),
            "selection_score": rec.get("conv_selection"),
            "complexity_overall": _score_from_slices(slices, "complexity_overall"),
            "quality_overall": _score_from_slices(slices, "quality_overall"),
            "reasoning_overall": _score_from_slices(slices, "reasoning_overall"),
            "rarity_score": rec.get("conv_rarity"),
            "confidence": _score_from_slices(slices, "confidence"),
            "thinking_mode": rec.get("thinking_mode"),
            "flags": sorted({flag for sample in slices for flag in ((sample.get("value") or {}).get("flags") or [])}),
            "turn_count": rec.get("turn_count"),
            "source_file": rec.get("source_file"),
        })

    covered = set(groups)
    for conv_key, rec in conv_by_key.items():
        if conv_key in covered:
            continue
        detail = rec.get("detail") or {}
        labels = rec.get("merged_labels")
        if not isinstance(labels, dict):
            labels = {}
        flags = []
        for key in ("negative_flags", "flags"):
            raw_flags = detail.get(key)
            if isinstance(raw_flags, list):
                flags.extend(str(flag) for flag in raw_flags if flag)
        confidence = detail.get("score_confidence")
        if not isinstance(confidence, (int, float)):
            confidence = rec.get("rarity_confidence")
        units.append({
            "id": conv_key,
            "labels": labels,
            "value_score": rec.get("conv_value"),
            "selection_score": rec.get("conv_selection"),
            "complexity_overall": rec.get("peak_complexity"),
            "quality_overall": detail.get("quality_overall"),
            "reasoning_overall": detail.get("reasoning_overall"),
            "rarity_score": rec.get("conv_rarity"),
            "confidence": confidence,
            "thinking_mode": rec.get("thinking_mode"),
            "flags": sorted(set(flags)),
            "turn_count": rec.get("turn_count"),
            "source_file": rec.get("source_file"),
        })

    return units


def _compute_pass2_from_units(units: list[dict], stats: dict, *, mode_id: str, unit_label: str, per_file_summary: list[dict] | None = None) -> dict:
    if not units and not stats:
        return {
            "overview": {},
            "selection_thresholds": {},
            "score_distributions": {},
            "sub_score_means": {},
            "value_by_tag": {},
            "selection_by_tag": {},
            "thinking_mode_stats": {},
            "flag_counts": {},
            "flag_value_impact": {},
            "coverage_at_thresholds": {},
            "weights_used": {},
            "per_file_summary": per_file_summary or [],
            "histograms": {},
            "confidence_histogram": [0] * 10,
            "selection_vs_value": {},
            "unit_label": unit_label,
            "mode_id": mode_id,
        }
    if not units:
        return _stats_only_pass2_mode(
            stats,
            mode_id=mode_id,
            unit_label=unit_label,
            per_file_summary=per_file_summary,
        )

    def extract(field: str) -> list[float]:
        return [unit[field] for unit in units if isinstance(unit.get(field), (int, float))]

    total_scored = len([unit for unit in units if unit.get("value_score") is not None])
    distributions = {
        "value_score": _percentiles(extract("value_score")),
        "selection_score": _percentiles(extract("selection_score")),
        "value_score_v2": _percentiles(extract("value_score_v2")),
        "selection_score_v2": _percentiles(extract("selection_score_v2")),
        "complexity_overall": _percentiles(extract("complexity_overall")),
        "quality_overall": _percentiles(extract("quality_overall")),
        "reasoning_overall": _percentiles(extract("reasoning_overall")),
        "rarity_score": _percentiles(extract("rarity_score")),
        "extension_rarity_score": _percentiles(extract("extension_rarity_score")),
        "rarity_v2_score": _percentiles(extract("rarity_v2_score")),
        "confidence": _percentiles(extract("confidence")),
    }

    histograms = {}
    for key in [
        "value_score",
        "value_score_v2",
        "complexity_overall",
        "quality_overall",
        "reasoning_overall",
        "rarity_score",
        "extension_rarity_score",
        "rarity_v2_score",
        "selection_score",
        "selection_score_v2",
    ]:
        values = extract(key)
        if values:
            histograms[key] = _histogram_bins(values)

    sub_score_means = {
        "complexity": {"overall": round(sum(extract("complexity_overall")) / max(len(extract("complexity_overall")), 1), 2)} if extract("complexity_overall") else {},
        "quality": {"overall": round(sum(extract("quality_overall")) / max(len(extract("quality_overall")), 1), 2)} if extract("quality_overall") else {},
        "reasoning": {"overall": round(sum(extract("reasoning_overall")) / max(len(extract("reasoning_overall")), 1), 2)} if extract("reasoning_overall") else {},
    }

    value_by_tag = {}
    selection_by_tag = {}
    for dim in _PASS2_TAG_DIMS:
        tag_values = {}
        tag_selections = {}
        for unit in units:
            value_score = unit.get("value_score")
            selection_score = unit.get("selection_score")
            if value_score is None:
                continue
            tags = (unit.get("labels") or {}).get(dim)
            if tags is None:
                continue
            if not isinstance(tags, list):
                tags = [tags]
            for tag in tags:
                tag_values.setdefault(tag, []).append(value_score)
                if selection_score is not None:
                    tag_selections.setdefault(tag, []).append(selection_score)
        value_by_tag[dim] = {
            tag: {"mean": round(sum(values) / len(values), 2), "n": len(values)}
            for tag, values in sorted(tag_values.items(), key=lambda item: (-(sum(item[1]) / len(item[1])), item[0]))
        }
        selection_by_tag[dim] = {
            tag: {"mean": round(sum(values) / len(values), 2), "n": len(values)}
            for tag, values in sorted(tag_selections.items(), key=lambda item: (-(sum(item[1]) / len(item[1])), item[0]))
        }

    slow_units = [unit for unit in units if unit.get("thinking_mode") == "slow"]
    fast_units = [unit for unit in units if unit.get("thinking_mode") == "fast"]
    thinking_mode_stats = {
        "slow": {
            "count": len(slow_units),
            "mean_value": round(sum(unit.get("value_score", 0) or 0 for unit in slow_units) / max(len(slow_units), 1), 2),
            "mean_quality": round(sum(unit.get("quality_overall", 0) or 0 for unit in slow_units) / max(len(slow_units), 1), 2),
            "mean_reasoning": round(sum(unit.get("reasoning_overall", 0) or 0 for unit in slow_units) / max(len(slow_units), 1), 2),
        },
        "fast": {
            "count": len(fast_units),
            "mean_value": round(sum(unit.get("value_score", 0) or 0 for unit in fast_units) / max(len(fast_units), 1), 2),
            "mean_quality": round(sum(unit.get("quality_overall", 0) or 0 for unit in fast_units) / max(len(fast_units), 1), 2),
            "mean_reasoning": round(sum(unit.get("reasoning_overall", 0) or 0 for unit in fast_units) / max(len(fast_units), 1), 2),
        },
    }
    thinking_mode_stats = {key: value for key, value in thinking_mode_stats.items() if value["count"] > 0}

    flag_counts = {}
    flag_value_sums = {}
    for unit in units:
        for flag in unit.get("flags") or []:
            flag_counts[flag] = flag_counts.get(flag, 0) + 1
            flag_value_sums.setdefault(flag, []).append(unit.get("value_score", 0) or 0)
    flag_value_impact = {
        flag: {"mean_value": round(sum(values) / len(values), 2), "count": len(values)}
        for flag, values in flag_value_sums.items()
    }

    all_value_scores = sorted(extract("value_score"))
    selection_thresholds = {}
    n_values = len(all_value_scores)
    for pct_label, pct in [("top_10pct", 90), ("top_25pct", 75), ("top_50pct", 50)]:
        if n_values > 0:
            idx = min(int(n_values * pct / 100), n_values - 1)
            threshold = all_value_scores[idx]
            count = sum(1 for value in all_value_scores if value >= threshold)
            selection_thresholds[pct_label] = {"threshold": round(threshold, 1), "count": count}

    coverage_at_thresholds = {}
    for threshold in [5.0, 6.0, 7.0, 8.0]:
        retained = [unit for unit in units if isinstance(unit.get("value_score"), (int, float)) and unit["value_score"] >= threshold]
        all_tags = set()
        retained_tags = set()
        for bucket, target in ((units, all_tags), (retained, retained_tags)):
            for unit in bucket:
                labels = unit.get("labels") or {}
                for dim in _PASS2_TAG_DIMS:
                    value = labels.get(dim)
                    if isinstance(value, list):
                        target.update(f"{dim}:{item}" for item in value)
                    elif value:
                        target.add(f"{dim}:{value}")
        coverage_at_thresholds[str(threshold)] = {
            "retained": len(retained),
            "pct": round(len(retained) / max(total_scored, 1), 2),
            "tags_lost": sorted(all_tags - retained_tags)[:20],
            "coverage": round(len(retained_tags) / max(len(all_tags), 1), 3),
        }

    confidence_histogram = [0] * 10
    for value in extract("confidence"):
        if 0 <= value <= 1:
            confidence_histogram[min(int(value * 10), 9)] += 1

    selection_vs_value = {}
    for unit in units:
        value_score = unit.get("value_score")
        selection_score = unit.get("selection_score")
        if isinstance(value_score, (int, float)) and isinstance(selection_score, (int, float)):
            if 1 <= value_score <= 10 and 1 <= selection_score <= 10:
                key = f"{max(1, min(int(value_score), 10))}|{max(1, min(int(selection_score), 10))}"
                selection_vs_value[key] = selection_vs_value.get(key, 0) + 1

    payload = {
        "overview": {
            "total_scored": total_scored,
            "total_failed": stats.get("total_failed", 0),
            "input_file": stats.get("input_file") or stats.get("input_path", ""),
            "mean_value": distributions.get("value_score", {}).get("mean", 0),
            "mean_selection": distributions.get("selection_score", {}).get("mean", 0),
            "mean_value_v2": distributions.get("value_score_v2", {}).get("mean", 0),
            "mean_selection_v2": distributions.get("selection_score_v2", {}).get("mean", 0),
            "mean_complexity": distributions.get("complexity_overall", {}).get("mean", 0),
            "mean_quality": distributions.get("quality_overall", {}).get("mean", 0),
            "median_rarity": distributions.get("rarity_score", {}).get("p50", distributions.get("rarity_score", {}).get("mean", 0)),
            "mean_extension_rarity": distributions.get("extension_rarity_score", {}).get("mean", 0),
            "mean_rarity_v2": distributions.get("rarity_v2_score", {}).get("mean", 0),
            "mean_confidence": distributions.get("confidence", {}).get("mean", 0),
            "total_tokens": stats.get("total_tokens", 0),
            "extension_rarity_mode": ((stats.get("extension_rarity_config") or {}).get("mode")),
            "extension_baseline_source": ((stats.get("extension_rarity_config") or {}).get("baseline_source")),
        },
        "selection_thresholds": selection_thresholds,
        "score_distributions": distributions,
        "sub_score_means": sub_score_means,
        "value_by_tag": value_by_tag,
        "selection_by_tag": selection_by_tag,
        "thinking_mode_stats": thinking_mode_stats,
        "flag_counts": dict(sorted(flag_counts.items(), key=lambda item: (-item[1], item[0]))),
        "flag_value_impact": flag_value_impact,
        "coverage_at_thresholds": coverage_at_thresholds,
        "weights_used": stats.get("weights_used", {}),
        "per_file_summary": per_file_summary or [],
        "histograms": histograms,
        "confidence_histogram": confidence_histogram,
        "selection_vs_value": selection_vs_value,
        "unit_label": unit_label,
        "mode_id": mode_id,
    }
    return _filter_pass2_extension_only_fields(payload, mode_id=mode_id)


def _group_units_by_source_file(units: list[dict]) -> dict[str, list[dict]]:
    grouped = {}
    for unit in units:
        key = unit.get("source_file") or "current"
        grouped.setdefault(key, []).append(unit)
    return grouped


def _compute_keep_rates_from_units(units: list[dict]) -> dict[str, float]:
    scored = [
        unit.get("selection_score")
        for unit in units
        if isinstance(unit.get("selection_score"), (int, float))
    ]
    total = len(scored)
    return {
        f"{float(threshold):.1f}": (
            round(sum(1 for score in scored if score >= float(threshold)) / total, 4)
            if total else 0
        )
        for threshold in FILE_RANKING_KEEP_RATE_THRESHOLDS
    }


def _build_per_file_summary(units: list[dict]) -> list[dict]:
    summaries = []
    for file_name, bucket in sorted(_group_units_by_source_file(units).items()):
        scored = [unit for unit in bucket if unit.get("value_score") is not None]
        if not scored:
            continue

        def _mean(field: str):
            values = [unit.get(field) for unit in scored if isinstance(unit.get(field), (int, float))]
            return round(sum(values) / len(values), 2) if values else 0

        keep_rates = _compute_keep_rates_from_units(scored)
        summaries.append({
            "file": Path(file_name).name if file_name != "current" else file_name,
            "count": len(scored),
            "mean_value": _mean("value_score"),
            "mean_complexity": _mean("complexity_overall"),
            "mean_quality": _mean("quality_overall"),
            "mean_rarity": _mean("rarity_score"),
            "mean_extension_rarity": _mean("extension_rarity_score"),
            "mean_rarity_v2": _mean("rarity_v2_score"),
            "mean_selection": _mean("selection_score"),
            "mean_selection_v2": _mean("selection_score_v2"),
            "keep_rates": keep_rates,
            "keep_rate_7": keep_rates.get(f"{FILE_RANKING_KEEP_RATE_THRESHOLD:.1f}", 0),
            "mean_turns": _mean("turn_count"),
        })
    return summaries


def build_pass2_viz(samples: list[dict], stats: dict, conv_records: list[dict] | None = None) -> dict:
    sample_units = [unit for sample in samples if (unit := _sample_pass2_unit(sample))]
    conversation_units = build_pass2_conversation_units(samples, conv_records)

    sample_mode = _compute_pass2_from_units(
        sample_units,
        stats,
        mode_id="sample",
        unit_label="samples",
        per_file_summary=stats.get("per_file_summary") or _build_per_file_summary(sample_units),
    )
    conversation_mode = _compute_pass2_from_units(
        conversation_units,
        stats,
        mode_id="conversation",
        unit_label="units",
        per_file_summary=_build_per_file_summary(conversation_units),
    )
    if _conversation_units_missing_subscore_metrics(conversation_units):
        _backfill_conversation_mode_from_stats(conversation_mode, stats)
    payload = dict(sample_mode)
    payload["modes"] = {
        "sample": sample_mode,
        "conversation": conversation_mode,
    }
    return payload


def infer_scope_turn_kind(samples: list[dict]) -> str | None:
    if not samples:
        return None
    has_single = False
    has_multi = False
    for sample in samples:
        source_id = ((sample or {}).get("metadata") or {}).get("source_id")
        if source_id in (None, ""):
            has_single = True
        else:
            has_multi = True
        if has_single and has_multi:
            return "mixed"
    if has_multi:
        return "multi"
    if has_single:
        return "single"
    return None


def infer_scope_turn_kind_from_path(path: str | Path | None) -> str | None:
    has_single = False
    has_multi = False
    for sample in iter_data_file(path):
        source_id = ((sample or {}).get("metadata") or {}).get("source_id")
        if source_id in (None, ""):
            has_single = True
        else:
            has_multi = True
        if has_single and has_multi:
            return "mixed"
    if has_multi:
        return "multi"
    if has_single:
        return "single"
    return None


def merge_turn_kinds(kinds: list[str | None]) -> str | None:
    has_single = False
    has_multi = False
    for kind in kinds:
        if kind == "mixed":
            return "mixed"
        if kind == "single":
            has_single = True
        elif kind == "multi":
            has_multi = True
    if has_single and has_multi:
        return "mixed"
    if has_multi:
        return "multi"
    if has_single:
        return "single"
    return None


def build_scope_summary(scope: dict) -> dict:
    pass1 = scope.get("pass1") or {}
    pass2 = scope.get("pass2") or {}
    sample_pass1 = (pass1.get("modes") or {}).get("sample", pass1)
    sample_pass2 = (pass2.get("modes") or {}).get("sample", pass2)
    conv_pass1 = (pass1.get("modes") or {}).get("conversation", sample_pass1)
    conv_pass2 = (pass2.get("modes") or {}).get("conversation", sample_pass2)
    turn_kind = scope.get("turn_kind")
    return {
        "sample": {
            "pass1_total": (sample_pass1 or {}).get("total", 0),
            "scored_total": ((sample_pass2 or {}).get("overview") or {}).get("total_scored", 0),
            "mean_value": ((sample_pass2 or {}).get("overview") or {}).get("mean_value"),
            "mean_selection": ((sample_pass2 or {}).get("overview") or {}).get("mean_selection"),
            "file_count": len(scope.get("descendant_files", [])) if scope.get("kind") != "file" else 1,
            "turn_kind": turn_kind,
        },
        "conversation": {
            "pass1_total": (conv_pass1 or {}).get("total", 0),
            "scored_total": ((conv_pass2 or {}).get("overview") or {}).get("total_scored", 0),
            "mean_value": ((conv_pass2 or {}).get("overview") or {}).get("mean_value"),
            "mean_selection": ((conv_pass2 or {}).get("overview") or {}).get("mean_selection"),
            "file_count": len(scope.get("descendant_files", [])) if scope.get("kind") != "file" else 1,
            "turn_kind": turn_kind,
        },
    }


def _round_numbers(value, digits: int = 4):
    if isinstance(value, float):
        return round(value, digits)
    if isinstance(value, list):
        return [_round_numbers(item, digits=digits) for item in value]
    if isinstance(value, dict):
        return {key: _round_numbers(item, digits=digits) for key, item in value.items()}
    return value


def _trim_unmapped_examples(mode: dict, example_limit: int = 1) -> dict:
    if not isinstance(mode, dict):
        return mode
    details = (mode.get("unmapped_details") or {}).get("by_dimension") or {}
    for rows in details.values():
        for row in rows or []:
            examples = row.get("examples")
            if isinstance(examples, list):
                row["examples"] = examples[:example_limit]
    return mode


def _trim_extension_examples(bucket: dict, example_limit: int = 1) -> dict:
    if not isinstance(bucket, dict):
        return bucket
    for mode_payload in (bucket.get("modes") or {}).values():
        if not isinstance(mode_payload, dict):
            continue
        for ext_payload in (mode_payload.get("extensions") or {}).values():
            details = (ext_payload.get("unmapped_details") or {}).get("by_dimension") or {}
            for rows in details.values():
                for row in rows or []:
                    examples = row.get("examples")
                    if isinstance(examples, list):
                        row["examples"] = examples[:example_limit]
    return bucket


def _export_dashboard_bucket(bucket: dict | None) -> dict | None:
    if not bucket:
        return None
    exported_modes = {}
    for mode_name, mode_payload in (bucket.get("modes") or {}).items():
        if not isinstance(mode_payload, dict):
            continue
        cleaned = {
            key: value
            for key, value in mode_payload.items()
            if key != "conf_matrix"
        }
        exported_modes[mode_name] = _round_numbers(_trim_unmapped_examples(cleaned))
    extensions = bucket.get("extensions")
    exported_extensions = None
    if isinstance(extensions, dict):
        exported_extensions = _round_numbers(_trim_extension_examples(dict(extensions)))
    if not exported_modes and not exported_extensions:
        return None
    payload = {"modes": exported_modes} if exported_modes else {}
    if exported_extensions:
        payload["extensions"] = exported_extensions
    return payload


def build_scope_detail_payload(scope: dict) -> dict:
    payload = {
        "id": scope.get("id"),
        "label": scope.get("label"),
        "kind": scope.get("kind"),
        "path": scope.get("path"),
        "summary": _round_numbers(scope.get("summary") or {}),
        "summary_modes": _round_numbers(scope.get("summary_modes") or {}),
        "pass1": _export_dashboard_bucket(scope.get("pass1")),
        "pass2": _export_dashboard_bucket(scope.get("pass2")),
        "conversation": _round_numbers(scope.get("conversation")) if scope.get("conversation") else None,
    }
    return {key: value for key, value in payload.items() if value is not None}


def build_dashboard_manifest(
    *,
    title: str,
    title_key: str,
    subtitle: str,
    root_id: str,
    default_scope_id: str,
    initially_expanded: list[str],
    scopes: dict[str, dict],
    explorer: dict | None = None,
) -> dict:
    manifest_scopes = {}
    for scope_id, scope in scopes.items():
        manifest_scopes[scope_id] = {
            "id": scope.get("id"),
            "label": scope.get("label"),
            "kind": scope.get("kind"),
            "path": scope.get("path"),
            "parent_id": scope.get("parent_id"),
            "children": scope.get("children") or [],
            "descendant_files": scope.get("descendant_files") or [],
            "summary": _round_numbers(scope.get("summary") or {}),
            "summary_modes": _round_numbers(scope.get("summary_modes") or {}),
            "detail_path": scope.get("detail_path"),
            "detail_script_path": scope.get("detail_script_path"),
            "has_pass1": bool(scope.get("pass1")),
            "has_pass2": bool(scope.get("pass2")),
            "has_conversation": bool(scope.get("conversation")),
        }
        if scope.get("explorer"):
            manifest_scopes[scope_id]["explorer"] = _round_numbers(scope["explorer"])
    return {
        "title": title,
        "title_key": title_key,
        "subtitle": subtitle,
        "root_id": root_id,
        "default_scope_id": default_scope_id,
        "initially_expanded": initially_expanded,
        "scopes": manifest_scopes,
        "explorer": _round_numbers(explorer) if explorer else None,
    }
