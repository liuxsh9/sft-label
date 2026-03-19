from __future__ import annotations

import json
from collections import Counter, defaultdict
from typing import TYPE_CHECKING

from sft_label.conversation import build_conversation_key

if TYPE_CHECKING:  # pragma: no cover
    from sft_label.label_extensions_schema import ExtensionSpec


def _truncate_text(text: str | None, limit: int = 280) -> str:
    value = str(text or "").strip()
    if len(value) <= limit:
        return value
    return value[: max(limit - 1, 1)].rstrip() + "…"


def _extension_config_from_spec(spec: "ExtensionSpec") -> dict:
    schema = {}
    for field_name, field in spec.schema.items():
        options = [str(item) for item in field.options]
        schema[field_name] = {
            "type": field.field_type,
            "description": field.description,
            "options": options,
            "option_count": len(options),
        }
    return {
        "display_name": spec.display_name,
        "description": spec.description,
        "source": spec.source,
        "prompt": spec.prompt,
        "prompt_preview": _truncate_text(spec.prompt),
        "schema": schema,
        "trigger": {key: list(values) for key, values in (spec.trigger or {}).items()},
    }


def _extension_config_map(extension_specs: list["ExtensionSpec"] | None) -> dict[str, dict]:
    mapping: dict[str, dict] = {}
    for spec in extension_specs or []:
        mapping[spec.id] = _extension_config_from_spec(spec)
    return mapping


def _config_map_from_extension_stats(extension_stats: dict | None) -> dict[str, dict]:
    mapping: dict[str, dict] = {}
    specs = (extension_stats or {}).get("specs") or {}
    for ext_id, spec in specs.items():
        config = (spec or {}).get("config")
        if isinstance(config, dict):
            mapping[str(ext_id)] = config
    return mapping


def _attach_extension_configs(payload: dict | None, config_map: dict[str, dict]) -> dict | None:
    if not payload or not config_map:
        return payload
    for bucket in [payload, *((payload.get("modes") or {}).values())]:
        if not isinstance(bucket, dict):
            continue
        for ext_id, ext_payload in ((bucket.get("extensions") or {}).items()):
            if ext_id in config_map and isinstance(ext_payload, dict):
                ext_payload["config"] = config_map[ext_id]
    return payload


def _extract_extensions(sample: dict) -> dict:
    if not isinstance(sample, dict):
        return {}
    payload = sample.get("label_extensions")
    if isinstance(payload, dict) and payload:
        return payload
    labels = sample.get("labels") or {}
    payload = labels.get("label_extensions")
    if isinstance(payload, dict) and payload:
        return payload
    return {}


def _query_preview(sample: dict, limit: int = 120) -> str:
    for msg in sample.get("conversations", []) or []:
        if msg.get("from") == "human":
            return str(msg.get("value", "")).replace("\n", " ")[:limit]
    return ""


def _merge_unmapped(unmapped: list[dict] | None) -> list[dict]:
    seen = set()
    merged = []
    for item in unmapped or []:
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
        merged.append({"dimension": dim, "value": value})
    return merged


def _canonicalize_config(config: dict | None) -> str | None:
    if not isinstance(config, dict):
        return None
    return json.dumps(config, ensure_ascii=False, sort_keys=True)


def _seed_spec_singleton_state(spec_stats: dict) -> None:
    spec_version = spec_stats.get("spec_version")
    spec_hash = spec_stats.get("spec_hash")
    config = spec_stats.get("config")
    spec_stats["_seen_spec_versions"] = (
        {str(spec_version)} if spec_version not in (None, "") else set()
    )
    spec_stats["_seen_spec_hashes"] = (
        {str(spec_hash)} if spec_hash not in (None, "") else set()
    )
    spec_stats["_seen_configs"] = (
        {_canonicalize_config(config)} if _canonicalize_config(config) else set()
    )


def _update_spec_singletons(spec_stats: dict, *, spec_version=None, spec_hash=None, config=None) -> None:
    if "_seen_spec_versions" not in spec_stats:
        _seed_spec_singleton_state(spec_stats)

    if spec_version not in (None, ""):
        spec_stats["_seen_spec_versions"].add(str(spec_version))
    if spec_hash not in (None, ""):
        spec_stats["_seen_spec_hashes"].add(str(spec_hash))
    config_key = _canonicalize_config(config)
    if config_key:
        spec_stats["_seen_configs"].add(config_key)

    seen_versions = spec_stats["_seen_spec_versions"]
    seen_hashes = spec_stats["_seen_spec_hashes"]
    seen_configs = spec_stats["_seen_configs"]

    spec_stats["spec_version"] = next(iter(seen_versions)) if len(seen_versions) == 1 else None
    spec_stats["spec_hash"] = next(iter(seen_hashes)) if len(seen_hashes) == 1 else None
    if len(seen_hashes) > 1 or len(seen_configs) != 1:
        spec_stats["config"] = None
    elif config_key and isinstance(config, dict):
        spec_stats["config"] = config


def _finalize_spec_singletons(specs: dict[str, dict]) -> None:
    for spec_stats in specs.values():
        spec_stats.pop("_seen_spec_versions", None)
        spec_stats.pop("_seen_spec_hashes", None)
        spec_stats.pop("_seen_configs", None)


def _merge_extension_payloads(slices: list[dict]) -> dict[str, dict]:
    merged: dict[str, dict] = {}
    confidence_accum: dict[str, dict[str, dict[str, float]]] = defaultdict(dict)
    unmapped_accum: dict[str, set[tuple[str, str]]] = defaultdict(set)
    for sample in slices:
        for ext_id, payload in _extract_extensions(sample).items():
            if not isinstance(payload, dict):
                continue
            entry = merged.setdefault(
                ext_id,
                {
                    "status": payload.get("status"),
                    "matched": bool(payload.get("matched")),
                    "spec_version": payload.get("spec_version"),
                    "spec_hash": payload.get("spec_hash"),
                    "labels": {},
                    "confidence": {},
                    "unmapped": [],
                },
            )
            if payload.get("status"):
                entry["status"] = payload.get("status")
            if payload.get("spec_version"):
                entry["spec_version"] = payload.get("spec_version")
            if payload.get("spec_hash"):
                entry["spec_hash"] = payload.get("spec_hash")
            entry["matched"] = bool(entry.get("matched")) or bool(payload.get("matched"))

            labels = payload.get("labels") or {}
            for field, value in labels.items():
                if isinstance(value, list):
                    bucket = entry["labels"].setdefault(field, [])
                    for item in value:
                        if item and item not in bucket:
                            bucket.append(item)
                elif value not in (None, ""):
                    entry["labels"][field] = value

            confidence = payload.get("confidence") or {}
            for field, score in confidence.items():
                if isinstance(score, bool) or not isinstance(score, (int, float)):
                    continue
                acc = confidence_accum[ext_id].setdefault(field, {"sum": 0.0, "count": 0.0})
                acc["sum"] += float(score)
                acc["count"] += 1.0

            for item in payload.get("unmapped") or []:
                if isinstance(item, dict):
                    dim = str(item.get("dimension", "unknown"))
                    value = str(item.get("value", "?"))
                else:
                    dim = "unknown"
                    value = str(item)
                unmapped_accum[ext_id].add((dim, value))

    for ext_id, entry in merged.items():
        for field, stats in confidence_accum.get(ext_id, {}).items():
            if stats["count"] > 0:
                entry.setdefault("confidence", {})[field] = stats["sum"] / stats["count"]
        entry["unmapped"] = [
            {"dimension": dim, "value": value}
            for dim, value in sorted(unmapped_accum.get(ext_id, set()))
        ]
    return merged


def build_extension_conversation_units(samples: list[dict]) -> list[dict]:
    if not samples:
        return []
    groups = defaultdict(list)
    units = []
    for sample in samples:
        meta = sample.get("metadata") or {}
        conv_key = build_conversation_key(meta.get("source_id"), meta.get("source_file"))
        if conv_key:
            groups[conv_key].append(sample)
        elif _extract_extensions(sample):
            units.append(sample)

    for conv_key, slices in groups.items():
        slices = sorted(slices, key=lambda item: (item.get("metadata") or {}).get("turn_index", 0))
        merged_extensions = _merge_extension_payloads(slices)
        last = slices[-1]
        meta = dict(last.get("metadata") or {})
        meta["source_id"] = meta.get("source_id") or conv_key
        meta["conversation_key"] = conv_key
        unit = {
            "id": conv_key,
            "metadata": meta,
            "conversations": last.get("conversations") or [],
            "label_extensions": merged_extensions,
        }
        units.append(unit)
    return units


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


def _new_extension_baseline(spec_version: str | None, spec_hash: str) -> dict:
    return {
        "spec_version": spec_version,
        "spec_hash": spec_hash,
        "total": 0,
        "matched": 0,
        "success": 0,
        "invalid": 0,
        "failed": 0,
        "skipped": 0,
        "baseline_total": 0,
        "field_value_distributions": {},
        "field_presence_counts": {},
        "confidence_stats": {},
    }


def _iter_present_values(value) -> list:
    if isinstance(value, list):
        return [item for item in value if item not in (None, "")]
    if value in (None, ""):
        return []
    return [value]


def _accumulate_field_value_stats(
    value_distributions: dict,
    presence_counts: dict,
    labels: dict,
) -> None:
    if not isinstance(labels, dict):
        return
    for field, value in labels.items():
        present_values = _iter_present_values(value)
        if not present_values:
            continue
        presence_counts[field] = presence_counts.get(field, 0) + 1
        dist = value_distributions.setdefault(field, {})
        for item in present_values:
            dist[item] = dist.get(item, 0) + 1


def aggregate_extension_stats(samples: list[dict], extension_specs: list["ExtensionSpec"] | None = None) -> dict | None:
    specs: dict[str, dict] = {}
    confidence_acc: dict[str, dict] = {}
    baseline_confidence_acc: dict[tuple[str, str], dict] = {}
    config_map = _extension_config_map(extension_specs)

    for sample in samples:
        payloads = _extract_extensions(sample)
        if not payloads:
            continue
        for spec_id, payload in payloads.items():
            if not isinstance(payload, dict):
                continue
            spec_stats = specs.setdefault(
                spec_id,
                {
                    "spec_version": payload.get("spec_version"),
                    "spec_hash": payload.get("spec_hash"),
                    "config": config_map.get(spec_id),
                    "total": 0,
                    "matched": 0,
                    "status_counts": {},
                    "field_distributions": {},
                    "baselines": {},
                    "confidence_stats": {},
                    "unmapped_counts": {},
                },
            )
            _update_spec_singletons(
                spec_stats,
                spec_version=payload.get("spec_version"),
                spec_hash=payload.get("spec_hash"),
                config=config_map.get(spec_id),
            )

            spec_stats["total"] += 1
            if payload.get("matched") is True:
                spec_stats["matched"] += 1

            status = payload.get("status") or "unknown"
            spec_stats["status_counts"][status] = spec_stats["status_counts"].get(status, 0) + 1

            labels = payload.get("labels") or {}
            if isinstance(labels, dict):
                for field, value in labels.items():
                    dist = spec_stats["field_distributions"].setdefault(field, {})
                    if isinstance(value, list):
                        for item in value:
                            if item:
                                dist[item] = dist.get(item, 0) + 1
                    elif value:
                        dist[value] = dist.get(value, 0) + 1

            confidence = payload.get("confidence") or {}
            if isinstance(confidence, dict):
                spec_conf_acc = confidence_acc.setdefault(spec_id, {})
                for field, score in confidence.items():
                    if isinstance(score, (int, float)):
                        _accumulate_confidence_stat(spec_conf_acc, field, float(score))

            spec_hash = payload.get("spec_hash")
            if spec_hash:
                baseline = spec_stats["baselines"].setdefault(
                    str(spec_hash),
                    _new_extension_baseline(payload.get("spec_version"), str(spec_hash)),
                )
                if not baseline.get("spec_version") and payload.get("spec_version"):
                    baseline["spec_version"] = payload.get("spec_version")
                baseline["total"] += 1
                if payload.get("matched") is True:
                    baseline["matched"] += 1
                if status in {"success", "invalid", "failed", "skipped"}:
                    baseline[status] += 1
                if status == "success" and payload.get("matched") is True:
                    baseline["baseline_total"] += 1
                    _accumulate_field_value_stats(
                        baseline["field_value_distributions"],
                        baseline["field_presence_counts"],
                        labels,
                    )
                    if isinstance(confidence, dict):
                        baseline_conf_acc = baseline_confidence_acc.setdefault((str(spec_id), str(spec_hash)), {})
                        for field, score in confidence.items():
                            if isinstance(score, (int, float)):
                                _accumulate_confidence_stat(baseline_conf_acc, field, float(score))

            unmapped = payload.get("unmapped") or []
            if isinstance(unmapped, list):
                for item in unmapped:
                    if isinstance(item, dict):
                        dim = str(item.get("dimension", "unknown"))
                        value = str(item.get("value", "?"))
                        key = f"{dim}:{value}"
                    else:
                        key = str(item)
                    spec_stats["unmapped_counts"][key] = spec_stats["unmapped_counts"].get(key, 0) + 1

    for spec_id, acc in confidence_acc.items():
        specs[spec_id]["confidence_stats"] = _finalize_confidence_stats(acc)
    for (spec_id, spec_hash), acc in baseline_confidence_acc.items():
        specs[spec_id]["baselines"][spec_hash]["confidence_stats"] = _finalize_confidence_stats(acc)

    if not specs:
        return None
    _finalize_spec_singletons(specs)
    return {"specs": specs}


def merge_extension_stats(stats_list: list[dict]) -> dict | None:
    specs: dict[str, dict] = {}
    confidence_acc: dict[str, dict] = {}
    baseline_confidence_acc: dict[tuple[str, str], dict] = {}

    for stats in stats_list:
        extension_stats = stats.get("extension_stats") or {}
        for spec_id, spec in (extension_stats.get("specs") or {}).items():
            if not isinstance(spec, dict):
                continue
            merged = specs.setdefault(
                spec_id,
                {
                    "spec_version": spec.get("spec_version"),
                    "spec_hash": spec.get("spec_hash"),
                    "config": spec.get("config"),
                    "total": 0,
                    "matched": 0,
                    "status_counts": {},
                    "field_distributions": {},
                    "baselines": {},
                    "confidence_stats": {},
                    "unmapped_counts": {},
                },
            )
            _update_spec_singletons(
                merged,
                spec_version=spec.get("spec_version"),
                spec_hash=spec.get("spec_hash"),
                config=spec.get("config"),
            )

            merged["total"] += int(spec.get("total", 0) or 0)
            merged["matched"] += int(spec.get("matched", 0) or 0)

            for status, count in (spec.get("status_counts") or {}).items():
                merged["status_counts"][status] = merged["status_counts"].get(status, 0) + int(count or 0)

            for field, dist in (spec.get("field_distributions") or {}).items():
                merged_dist = merged["field_distributions"].setdefault(field, {})
                for value, count in (dist or {}).items():
                    merged_dist[value] = merged_dist.get(value, 0) + int(count or 0)

            for key, count in (spec.get("unmapped_counts") or {}).items():
                merged["unmapped_counts"][key] = merged["unmapped_counts"].get(key, 0) + int(count or 0)

            for field, conf in (spec.get("confidence_stats") or {}).items():
                if not isinstance(conf, dict):
                    continue
                count = int(conf.get("count", 0) or 0)
                if count <= 0:
                    continue
                mean = float(conf.get("mean", 0.0) or 0.0)
                min_val = float(conf.get("min", mean))
                max_val = float(conf.get("max", mean))
                spec_conf_acc = confidence_acc.setdefault(spec_id, {})
                _accumulate_confidence_bulk(
                    spec_conf_acc,
                    field,
                    mean=mean,
                    count=count,
                    min_val=min_val,
                    max_val=max_val,
                )

            for spec_hash, baseline in (spec.get("baselines") or {}).items():
                if not isinstance(baseline, dict):
                    continue
                merged_baseline = merged["baselines"].setdefault(
                    str(spec_hash),
                    _new_extension_baseline(
                        baseline.get("spec_version"),
                        str(baseline.get("spec_hash") or spec_hash),
                    ),
                )
                if not merged_baseline.get("spec_version") and baseline.get("spec_version"):
                    merged_baseline["spec_version"] = baseline.get("spec_version")
                merged_baseline["total"] += int(baseline.get("total", 0) or 0)
                merged_baseline["matched"] += int(baseline.get("matched", 0) or 0)
                merged_baseline["success"] += int(baseline.get("success", 0) or 0)
                merged_baseline["invalid"] += int(baseline.get("invalid", 0) or 0)
                merged_baseline["failed"] += int(baseline.get("failed", 0) or 0)
                merged_baseline["skipped"] += int(baseline.get("skipped", 0) or 0)
                merged_baseline["baseline_total"] += int(baseline.get("baseline_total", 0) or 0)

                for field, dist in (baseline.get("field_value_distributions") or {}).items():
                    merged_dist = merged_baseline["field_value_distributions"].setdefault(field, {})
                    for value, count in (dist or {}).items():
                        merged_dist[value] = merged_dist.get(value, 0) + int(count or 0)

                for field, count in (baseline.get("field_presence_counts") or {}).items():
                    merged_baseline["field_presence_counts"][field] = (
                        merged_baseline["field_presence_counts"].get(field, 0) + int(count or 0)
                    )

                for field, conf in (baseline.get("confidence_stats") or {}).items():
                    if not isinstance(conf, dict):
                        continue
                    count = int(conf.get("count", 0) or 0)
                    if count <= 0:
                        continue
                    mean = float(conf.get("mean", 0.0) or 0.0)
                    min_val = float(conf.get("min", mean))
                    max_val = float(conf.get("max", mean))
                    baseline_acc = baseline_confidence_acc.setdefault((str(spec_id), str(spec_hash)), {})
                    _accumulate_confidence_bulk(
                        baseline_acc,
                        field,
                        mean=mean,
                        count=count,
                        min_val=min_val,
                        max_val=max_val,
                    )

    for spec_id, acc in confidence_acc.items():
        specs[spec_id]["confidence_stats"] = _finalize_confidence_stats(acc)
    for (spec_id, spec_hash), acc in baseline_confidence_acc.items():
        specs[spec_id]["baselines"][spec_hash]["confidence_stats"] = _finalize_confidence_stats(acc)

    if not specs:
        return None
    _finalize_spec_singletons(specs)
    return {"specs": specs}


def _dashboard_extension_payload_from_stats(extension_stats: dict) -> dict | None:
    specs = (extension_stats or {}).get("specs") or {}
    if not isinstance(specs, dict) or not specs:
        return None

    extensions: dict[str, dict] = {}
    total_units = 0
    for ext_id, spec in sorted(specs.items()):
        if not isinstance(spec, dict):
            continue
        total = int(spec.get("total", 0) or 0)
        total_units = max(total_units, total)
        unmapped_counts = {}
        for key, count in (spec.get("unmapped_counts") or {}).items():
            dim, _, value = str(key).partition(":")
            unmapped_counts[(dim or "unknown", value or "?")] = int(count or 0)
        extensions[ext_id] = {
            "id": ext_id,
            "total": total,
            "matched": int(spec.get("matched", 0) or 0),
            "config": dict(spec.get("config") or {}) if isinstance(spec.get("config"), dict) else None,
            "status_counts": dict(spec.get("status_counts") or {}),
            "spec_versions": ({str(spec["spec_version"]): total} if spec.get("spec_version") else {}),
            "spec_hashes": ({str(spec["spec_hash"]): total} if spec.get("spec_hash") else {}),
            "labels": dict(spec.get("field_distributions") or {}),
            "confidence_stats": dict(spec.get("confidence_stats") or {}),
            "unmapped_details": _build_unmapped_details(unmapped_counts, {}),
        }
        extensions[ext_id]["summary"] = _extension_summary(
            extensions[ext_id]["labels"],
            total=extensions[ext_id]["total"],
            matched=extensions[ext_id]["matched"],
            status_counts=extensions[ext_id]["status_counts"],
        )

    if not extensions:
        return None
    sample_payload = {
        "mode_id": "sample",
        "unit_label": "samples",
        "total_units": total_units,
        "extensions": extensions,
    }
    return {
        "extensions": extensions,
        "mode_id": "sample",
        "unit_label": "samples",
        "total_units": total_units,
        "modes": {"sample": sample_payload},
    }


def _build_unmapped_details(
    counts: dict[tuple[str, str], int],
    examples: dict[tuple[str, str], list[dict]],
) -> dict:
    grouped: dict[str, list[dict]] = {}
    total_occurrences = 0
    counts_by_dim: dict[str, Counter] = {}
    for (dim, value), count in counts.items():
        counts_by_dim.setdefault(dim, Counter())[value] += int(count)

    for dim, counter in counts_by_dim.items():
        rows = []
        for value, count in counter.most_common():
            total_occurrences += count
            rows.append(
                {
                    "label": value,
                    "count": count,
                    "examples": examples.get((dim, value), []),
                }
            )
        grouped[dim] = rows

    return {"total_occurrences": total_occurrences, "by_dimension": grouped}


def _extension_summary(labels: dict, *, total: int, matched: int, status_counts: dict) -> dict:
    field_count = len(labels or {})
    value_count = 0
    for value in (labels or {}).values():
        if isinstance(value, dict):
            value_count += len(value)
    return {
        "field_count": field_count,
        "value_count": value_count,
        "matched_rate": round(matched / max(total, 1), 4) if total else 0.0,
        "status_variant_count": len([key for key, count in (status_counts or {}).items() if count]),
    }


def build_extension_stats(
    units: list[dict],
    *,
    mode_id: str,
    unit_label: str,
    example_limit: int = 2,
    extension_configs: dict[str, dict] | None = None,
) -> dict:
    total_units = len(units)
    ext_stats: dict[str, dict] = {}

    for unit in units:
        extensions = _extract_extensions(unit)
        if not extensions:
            continue
        query = _query_preview(unit)
        unit_id = str(unit.get("id", "?"))
        for ext_id, payload in extensions.items():
            if not isinstance(payload, dict):
                continue
            stats = ext_stats.setdefault(
                ext_id,
                {
                    "id": ext_id,
                    "total": 0,
                    "matched": 0,
                    "status_counts": Counter(),
                    "spec_versions": Counter(),
                    "spec_hashes": Counter(),
                    "labels": defaultdict(Counter),
                    "confidence_values": defaultdict(list),
                    "unmapped_counts": Counter(),
                    "unmapped_examples": defaultdict(list),
                },
            )
            stats["total"] += 1
            if payload.get("matched"):
                stats["matched"] += 1

            status = payload.get("status") or "unknown"
            stats["status_counts"][status] += 1

            spec_version = payload.get("spec_version")
            if spec_version:
                stats["spec_versions"][str(spec_version)] += 1

            spec_hash = payload.get("spec_hash")
            if spec_hash:
                stats["spec_hashes"][str(spec_hash)] += 1

            labels = payload.get("labels") or {}
            for field, value in labels.items():
                bucket = stats["labels"][field]
                if isinstance(value, list):
                    for item in value:
                        if item not in (None, ""):
                            bucket[str(item)] += 1
                elif value not in (None, ""):
                    bucket[str(value)] += 1

            confidence = payload.get("confidence") or {}
            for field, score in confidence.items():
                if isinstance(score, bool) or not isinstance(score, (int, float)):
                    continue
                stats["confidence_values"][field].append(float(score))

            for item in payload.get("unmapped") or []:
                if isinstance(item, dict):
                    dim = str(item.get("dimension", "unknown"))
                    value = str(item.get("value", "?"))
                else:
                    dim = "unknown"
                    value = str(item)
                key = (dim, value)
                stats["unmapped_counts"][key] += 1
                examples = stats["unmapped_examples"][key]
                if len(examples) < example_limit:
                    examples.append({"id": unit_id, "query": query})

    extensions: dict[str, dict] = {}
    for ext_id in sorted(ext_stats):
        stats = ext_stats[ext_id]
        confidence_stats = {}
        for field, values in stats["confidence_values"].items():
            if not values:
                continue
            confidence_stats[field] = {
                "mean": round(sum(values) / len(values), 3),
                "min": round(min(values), 3),
                "max": round(max(values), 3),
                "below_threshold": sum(1 for score in values if score < 0.7),
                "count": len(values),
            }

        extensions[ext_id] = {
            "id": ext_id,
            "total": stats["total"],
            "matched": stats["matched"],
            "config": dict((extension_configs or {}).get(ext_id) or {}) if ext_id in (extension_configs or {}) else None,
            "status_counts": dict(stats["status_counts"]),
            "spec_versions": dict(stats["spec_versions"]),
            "spec_hashes": dict(stats["spec_hashes"]),
            "labels": {
                field: dict(sorted(bucket.items(), key=lambda item: (-item[1], item[0])))
                for field, bucket in stats["labels"].items()
            },
            "confidence_stats": confidence_stats,
            "unmapped_details": _build_unmapped_details(stats["unmapped_counts"], stats["unmapped_examples"]),
        }
        extensions[ext_id]["summary"] = _extension_summary(
            extensions[ext_id]["labels"],
            total=extensions[ext_id]["total"],
            matched=extensions[ext_id]["matched"],
            status_counts=extensions[ext_id]["status_counts"],
        )

    return {
        "mode_id": mode_id,
        "unit_label": unit_label,
        "total_units": total_units,
        "extensions": extensions,
    }


def build_extension_dashboard_payload(samples: list[dict], extension_stats: dict | None = None) -> dict | None:
    config_map = _config_map_from_extension_stats(extension_stats)
    if samples:
        has_extensions = any(_extract_extensions(sample) for sample in samples)
        if has_extensions:
            conversation_units = build_extension_conversation_units(samples)
            return _attach_extension_configs({
                "modes": {
                    "sample": build_extension_stats(
                        samples,
                        mode_id="sample",
                        unit_label="samples",
                        extension_configs=config_map,
                    ),
                    "conversation": build_extension_stats(
                        conversation_units,
                        mode_id="conversation",
                        unit_label="units",
                        extension_configs=config_map,
                    ),
                }
            }, config_map)
    if extension_stats:
        return _attach_extension_configs(_dashboard_extension_payload_from_stats(extension_stats), config_map)
    return None


__all__ = [
    "aggregate_extension_stats",
    "build_extension_conversation_units",
    "build_extension_dashboard_payload",
    "build_extension_stats",
    "merge_extension_stats",
]
