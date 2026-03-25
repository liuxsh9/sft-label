"""Pass 1 helpers for merging transient slice labels back into JSONL rows."""

from __future__ import annotations

import copy
import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from sft_label.inline_labels import (
    DEFAULT_LABEL_VERSION,
    clear_pass2_state,
    compact_data_label,
    compact_labeling_monitor_summary,
    compact_turn_record,
    compact_conversation_record,
    ensure_data_label,
    get_data_label,
    get_data_label_meta,
    mark_stage_timestamp,
    replace_data_label,
    set_conversation_record,
    set_data_id,
    set_meta_field,
    upsert_turn_record,
)
from sft_label.inline_migration import InlineMigrationMatch, seed_row_from_migration
from sft_label.inline_rows import RowSampleBundle
from sft_label.inline_rows import assign_stable_sample_ids
from sft_label.label_extensions_schema import spec_hash
from sft_label.labels import is_usable_labels
from sft_label.preprocessing import apply_sparse_sampling


@dataclass
class InlinePass1MergeResult:
    """Merged inline rows plus sample-level compatibility artifacts."""

    rows: list[dict]
    samples: list[dict]
    monitor_records: list[dict]
    failed_samples: list[dict]


@dataclass(frozen=True)
class InlineBundlePlan:
    """Mode-aware Pass 1 plan for a single source row bundle."""

    local_label_indices: list[int]
    local_inherit_map: dict[int, int]
    preserved_labels: list[dict | None]
    use_existing_data_label: bool
    pass1_changed: bool
    invalidate_pass2: bool
    migrated: bool = False


@dataclass(frozen=True)
class InlinePreparedBatch:
    """Flattened samples plus mode-aware plan metadata."""

    bundles: list[RowSampleBundle]
    samples: list[dict]
    sample_to_bundle: list[int]
    labels: list[dict | None]
    label_indices: list[int]
    inherit_map: dict[int, int]
    updated_sample_indices: set[int]
    bundle_plans: list[InlineBundlePlan]
    stats: dict


def _sample_turn_index(sample: dict, default_turn_index: int) -> int:
    metadata = sample.get("metadata") or {}
    return metadata.get("turn_index") or default_turn_index


def _turn_label_payload_from_record(turn_record: dict | None):
    if not isinstance(turn_record, dict):
        return None
    labels = copy.deepcopy(turn_record.get("labels"))
    if not is_usable_labels(labels):
        return None
    label_extensions = turn_record.get("label_extensions")
    if isinstance(label_extensions, dict) and label_extensions:
        labels["label_extensions"] = copy.deepcopy(label_extensions)
    if turn_record.get("inherited"):
        labels["inherited"] = True
    if turn_record.get("inherited_from"):
        labels["inherited_from"] = turn_record.get("inherited_from")
    return labels


def _existing_turn_records(bundle: RowSampleBundle) -> dict[int, dict]:
    data_label = get_data_label(bundle.raw_row, default={}) or {}
    turns = data_label.get("turns") or []
    return {
        turn.get("turn_index"): copy.deepcopy(turn)
        for turn in turns
        if isinstance(turn, dict) and turn.get("turn_index") is not None
    }


def _preserved_labels_for_bundle(bundle: RowSampleBundle) -> list[dict | None]:
    existing_turns = _existing_turn_records(bundle)
    preserved = []
    for default_idx, sample in enumerate(bundle.samples, start=1):
        turn_index = _sample_turn_index(sample, default_idx)
        preserved.append(_turn_label_payload_from_record(existing_turns.get(turn_index)))
    return preserved


def _needs_full_relabel(bundle: RowSampleBundle, *, label_version: str) -> bool:
    meta = get_data_label_meta(bundle.raw_row, default={}) or {}
    existing = meta.get("label_version")
    return existing != label_version


def _is_bundle_complete(bundle: RowSampleBundle, *, label_version: str) -> bool:
    if _needs_full_relabel(bundle, label_version=label_version):
        return False
    preserved = _preserved_labels_for_bundle(bundle)
    return bool(bundle.samples) and all(is_usable_labels(labels) for labels in preserved)


def _extension_fingerprints(extension_specs) -> dict[str, dict[str, str]]:
    fingerprints: dict[str, dict[str, str]] = {}
    for spec in extension_specs or []:
        if not getattr(spec, "enabled", True):
            continue
        fingerprints[str(spec.id)] = {
            "spec_version": str(spec.spec_version),
            "spec_hash": spec_hash(spec),
        }
    return fingerprints


def _needs_extension_refresh(bundle: RowSampleBundle, *, extension_specs=None, label_version: str) -> bool:
    if not extension_specs or _needs_full_relabel(bundle, label_version=label_version):
        return False

    current = _extension_fingerprints(extension_specs)
    if not current:
        return False

    meta = get_data_label_meta(bundle.raw_row, default={}) or {}
    existing = meta.get("extension_specs")
    if not isinstance(existing, dict) or existing != current:
        return True

    existing_turns = _existing_turn_records(bundle)
    for default_idx, sample in enumerate(bundle.samples, start=1):
        turn_index = _sample_turn_index(sample, default_idx)
        turn = existing_turns.get(turn_index) or {}
        label_extensions = turn.get("label_extensions") or {}
        if not isinstance(label_extensions, dict) or set(label_extensions) != set(current):
            return True

    return False


def _sparse_plan(samples: list[dict], sparse_kwargs: dict) -> tuple[list[int], dict[int, int]]:
    label_indices, inherit_map = apply_sparse_sampling(samples, **(sparse_kwargs or {}))
    return list(label_indices), dict(inherit_map)


def _plan_bundle(bundle: RowSampleBundle, *, mode: str, label_version: str,
                 sparse_kwargs: dict, extension_specs=None) -> InlineBundlePlan:
    preserved_labels = _preserved_labels_for_bundle(bundle)

    if mode == "refresh":
        label_indices, inherit_map = _sparse_plan(bundle.samples, sparse_kwargs)
        return InlineBundlePlan(
            local_label_indices=label_indices,
            local_inherit_map=inherit_map,
            preserved_labels=[None] * len(bundle.samples),
            use_existing_data_label=False,
            pass1_changed=bool(bundle.samples),
            invalidate_pass2=bool(bundle.samples),
        )

    if _needs_full_relabel(bundle, label_version=label_version):
        label_indices, inherit_map = _sparse_plan(bundle.samples, sparse_kwargs)
        return InlineBundlePlan(
            local_label_indices=label_indices,
            local_inherit_map=inherit_map,
            preserved_labels=[None] * len(bundle.samples),
            use_existing_data_label=False,
            pass1_changed=bool(bundle.samples),
            invalidate_pass2=bool(bundle.samples),
        )

    missing_indices = [
        idx
        for idx, labels in enumerate(preserved_labels)
        if not is_usable_labels(labels)
    ]
    if not missing_indices:
        if _needs_extension_refresh(bundle, extension_specs=extension_specs, label_version=label_version):
            return InlineBundlePlan(
                local_label_indices=list(range(len(bundle.samples))),
                local_inherit_map={},
                preserved_labels=preserved_labels,
                use_existing_data_label=True,
                pass1_changed=bool(bundle.samples),
                invalidate_pass2=False,
            )
        return InlineBundlePlan(
            local_label_indices=[],
            local_inherit_map={},
            preserved_labels=preserved_labels,
            use_existing_data_label=True,
            pass1_changed=False,
            invalidate_pass2=False,
        )

    return InlineBundlePlan(
        local_label_indices=missing_indices,
        local_inherit_map={},
        preserved_labels=preserved_labels,
        use_existing_data_label=True,
        pass1_changed=True,
        invalidate_pass2=True,
    )


def prepare_inline_pass1_batch(
    bundles: list[RowSampleBundle],
    *,
    mode: str = "refresh",
    label_version: str = DEFAULT_LABEL_VERSION,
    sparse_kwargs: dict | None = None,
    extension_specs: list | None = None,
    migration_index: dict[str, InlineMigrationMatch] | None = None,
    timestamp: str | None = None,
) -> InlinePreparedBatch:
    """Prepare flattened samples and mode-aware Pass 1 work for a bundle batch."""
    now = timestamp or datetime.now().isoformat()
    prepared_bundles = [copy.deepcopy(bundle) for bundle in bundles]
    migrated_bundle_indices: set[int] = set()

    migrated_rows = 0
    migration_hits = 0
    for idx, bundle in enumerate(prepared_bundles):
        match = (migration_index or {}).get(bundle.data_id)
        if match is None:
            continue
        prepared_bundles[idx] = RowSampleBundle(
            raw_row=seed_row_from_migration(bundle.raw_row, match, timestamp=now),
            source_path=bundle.source_path,
            row_number=bundle.row_number,
            data_id=bundle.data_id,
            samples=copy.deepcopy(bundle.samples),
        )
        migrated_rows += 1
        migration_hits += 1
        migrated_bundle_indices.add(idx)

    samples: list[dict] = []
    sample_to_bundle: list[int] = []
    labels: list[dict | None] = []
    label_indices: list[int] = []
    inherit_map: dict[int, int] = {}
    updated_sample_indices: set[int] = set()
    bundle_plans: list[InlineBundlePlan] = []
    global_idx = 0

    skipped_rows = 0
    changed_rows = 0
    pass2_invalidated_rows = 0
    preserved_samples = 0

    for bundle_idx, bundle in enumerate(prepared_bundles):
        bundle_plan = _plan_bundle(
            bundle,
            mode=mode,
            label_version=label_version,
            sparse_kwargs=sparse_kwargs or {},
            extension_specs=extension_specs,
        )
        if bundle_idx in migrated_bundle_indices:
            bundle_plan = InlineBundlePlan(
                local_label_indices=bundle_plan.local_label_indices,
                local_inherit_map=bundle_plan.local_inherit_map,
                preserved_labels=bundle_plan.preserved_labels,
                use_existing_data_label=bundle_plan.use_existing_data_label,
                pass1_changed=bundle_plan.pass1_changed,
                invalidate_pass2=bundle_plan.invalidate_pass2,
                migrated=True,
            )
        bundle_plans.append(bundle_plan)

        if bundle_plan.pass1_changed:
            changed_rows += 1
        else:
            skipped_rows += 1
        if bundle_plan.invalidate_pass2:
            pass2_invalidated_rows += 1
        preserved_samples += sum(1 for payload in bundle_plan.preserved_labels if is_usable_labels(payload))

        local_to_global: dict[int, int] = {}
        for local_idx, sample in enumerate(bundle.samples):
            samples.append(copy.deepcopy(sample))
            sample_to_bundle.append(bundle_idx)
            labels.append(copy.deepcopy(bundle_plan.preserved_labels[local_idx]))
            local_to_global[local_idx] = global_idx
            global_idx += 1

        for local_idx in bundle_plan.local_label_indices:
            label_indices.append(local_to_global[local_idx])
            updated_sample_indices.add(local_to_global[local_idx])
        for local_idx, source_local_idx in bundle_plan.local_inherit_map.items():
            inherit_map[local_to_global[local_idx]] = local_to_global[source_local_idx]
            updated_sample_indices.add(local_to_global[local_idx])

    stats = {
        "rows_total": len(prepared_bundles),
        "rows_skipped": skipped_rows,
        "rows_changed": changed_rows,
        "rows_migrated": migrated_rows,
        "rows_pass2_invalidated": pass2_invalidated_rows,
        "preserved_samples": preserved_samples,
        "migration_hits": migration_hits,
    }

    assign_stable_sample_ids(prepared_bundles, samples, sample_to_bundle)

    return InlinePreparedBatch(
        bundles=prepared_bundles,
        samples=samples,
        sample_to_bundle=sample_to_bundle,
        labels=labels,
        label_indices=label_indices,
        inherit_map=inherit_map,
        updated_sample_indices=updated_sample_indices,
        bundle_plans=bundle_plans,
        stats=stats,
    )


def _infer_source_format(raw_row: dict, samples: list[dict]) -> str:
    for sample in samples:
        metadata = sample.get("metadata") or {}
        original_format = metadata.get("original_format")
        if original_format:
            return str(original_format)
    if "data" in raw_row:
        return "pangu"
    if "messages" in raw_row:
        return "openai_messages"
    if "conversations" in raw_row:
        return "sharegpt"
    return "unknown"


def apply_inherited_labels(samples: list[dict], labels: list, inherit_map: dict | None = None) -> list:
    """Copy sparse-sampled labels forward into inherited slices."""
    resolved = list(labels)
    for unlabeled_idx, source_idx in (inherit_map or {}).items():
        if source_idx >= len(resolved) or resolved[source_idx] is None:
            continue
        inherited = copy.deepcopy(resolved[source_idx])
        inherited["inherited"] = True
        inherited["inherited_from"] = samples[source_idx].get("id")
        resolved[unlabeled_idx] = inherited
    return resolved


def _sample_monitor_summary(monitor: dict | None) -> dict | None:
    return compact_labeling_monitor_summary(monitor)


def _extract_query_preview(sample: dict, max_len: int = 240) -> str:
    conversations = sample.get("conversations")
    if isinstance(conversations, list):
        for msg in conversations:
            if msg.get("from") == "human":
                text = str(msg.get("value", "")).replace("\n", " ").replace("\r", "").strip()
                return text[:max_len] + ("…" if len(text) > max_len else "")
    data = sample.get("data")
    if isinstance(data, list):
        for msg in data:
            if msg.get("role") == "user":
                text = str(msg.get("content", "")).replace("\n", " ").replace("\r", "").strip()
                return text[:max_len] + ("…" if len(text) > max_len else "")
    return ""


def build_unmapped_event_records(samples: list[dict]) -> list[dict]:
    """Flatten per-sample unmapped labels into lightweight review rows."""
    rows: list[dict] = []
    for sample in samples:
        labels = sample.get("labels") or {}
        unmapped_items = labels.get("unmapped") or []
        if not isinstance(unmapped_items, list) or not unmapped_items:
            continue
        metadata = sample.get("metadata") or {}
        base = {
            "sample_id": sample.get("id"),
            "source_file": metadata.get("source_file"),
            "source_id": metadata.get("source_id"),
            "turn_index": metadata.get("turn_index"),
            "total_turns": metadata.get("total_turns"),
            "query": _extract_query_preview(sample),
            "intent": labels.get("intent", ""),
            "difficulty": labels.get("difficulty", ""),
            "context": labels.get("context", ""),
            "language": copy.deepcopy(labels.get("language", [])),
            "domain": copy.deepcopy(labels.get("domain", [])),
            "task": copy.deepcopy(labels.get("task", [])),
            "concept": copy.deepcopy(labels.get("concept", [])),
            "agentic": copy.deepcopy(labels.get("agentic", [])),
            "constraint": copy.deepcopy(labels.get("constraint", [])),
            "confidence": copy.deepcopy(labels.get("confidence", {})),
            "canonicalized": copy.deepcopy(labels.get("canonicalized", [])),
            "monitor_status": ((sample.get("labeling_monitor") or {}).get("status")),
        }
        if labels.get("inherited"):
            base["inherited"] = True
        if labels.get("inherited_from"):
            base["inherited_from"] = labels.get("inherited_from")
        for item in unmapped_items:
            if isinstance(item, dict):
                dim = str(item.get("dimension", "?"))
                value = str(item.get("value", "")).strip()
            else:
                dim = "?"
                value = str(item).strip()
            if not value:
                continue
            row = copy.deepcopy(base)
            row["dimension"] = dim
            row["value"] = value
            rows.append(row)
    return rows


def build_sample_artifacts(
    samples: list[dict],
    labels: list,
    monitors: list,
    *,
    source_file,
) -> tuple[list[dict], list[dict], list[dict]]:
    """Build sample-level compatibility records from Pass 1 results."""
    rendered_samples: list[dict] = []
    monitor_records: list[dict] = []
    failed_samples: list[dict] = []

    for sample, label_payload, monitor in zip(samples, labels, monitors):
        sample_record = copy.deepcopy(sample)
        sample_record.setdefault("metadata", {})["source_file"] = str(source_file)
        core_labels, label_extensions, _inherited, _inherited_from = _normalize_turn_labels(label_payload)
        sample_record["labels"] = core_labels
        if label_extensions:
            sample_record["label_extensions"] = copy.deepcopy(label_extensions)

        monitor_summary = None
        if monitor:
            monitor_record = copy.deepcopy(monitor)
            monitor_record.setdefault("source_file", str(source_file))
            monitor_records.append(monitor_record)
            monitor_summary = {
                "status": monitor.get("status"),
                "llm_calls": monitor.get("llm_calls", 0),
                "arbitrated": bool(monitor.get("arbitrated")),
                "validation_issues": copy.deepcopy(monitor.get("validation_issues", [])),
                "consistency_warnings": copy.deepcopy(monitor.get("consistency_warnings", [])),
            }

        if monitor_summary:
            sample_record["labeling_monitor"] = monitor_summary

        if label_payload is None or not is_usable_labels(label_payload):
            failed_record = copy.deepcopy(sample_record)
            failed_record.pop("labels", None)
            failed_record.pop("labeling_monitor", None)
            failed_samples.append(failed_record)

        rendered_samples.append(sample_record)

    return rendered_samples, monitor_records, failed_samples


def _normalize_turn_labels(label_payload):
    if label_payload is None:
        return None, None, False, None
    labels = copy.deepcopy(label_payload)
    label_extensions = labels.pop("label_extensions", None)
    inherited = bool(labels.pop("inherited", False))
    inherited_from = labels.pop("inherited_from", None)
    return labels, label_extensions, inherited, inherited_from


def _turn_record(sample: dict, label_payload, monitor: dict | None, *, default_turn_index: int) -> dict:
    turn_index = (sample.get("metadata") or {}).get("turn_index") or default_turn_index
    labels, label_extensions, inherited, inherited_from = _normalize_turn_labels(label_payload)

    record = {
        "turn_index": turn_index,
        "sample_id": sample.get("id"),
        "labels": labels,
    }
    if label_extensions:
        record["label_extensions"] = copy.deepcopy(label_extensions)
    if inherited:
        record["inherited"] = True
    if inherited_from:
        record["inherited_from"] = inherited_from

    monitor_summary = _sample_monitor_summary(monitor)
    if monitor_summary:
        record["labeling_monitor"] = monitor_summary
    elif inherited:
        record["labeling_monitor"] = {"status": "inherited"}
    else:
        record["labeling_monitor"] = {"status": "no_result"}

    return record


def _conversation_record(bundle: RowSampleBundle, turn_records: list[dict]) -> dict:
    first_sample = bundle.samples[0] if bundle.samples else {}
    first_meta = first_sample.get("metadata") or {}
    return {
        "conversation_id": first_meta.get("source_id") or bundle.data_id,
        "turn_count": len(turn_records),
    }


def merge_pass1_results(
    bundles: list[RowSampleBundle],
    samples: list[dict],
    labels: list,
    monitors: list,
    sample_to_bundle: list[int],
    *,
    source_file,
    label_version: str = DEFAULT_LABEL_VERSION,
    mode: str = "refresh",
    timestamp: str | None = None,
    bundle_plans: list[InlineBundlePlan] | None = None,
    updated_sample_indices: set[int] | None = None,
) -> InlinePass1MergeResult:
    """Merge slice-level Pass 1 results back into row-local inline annotations."""
    assign_stable_sample_ids(bundles, samples, sample_to_bundle)
    resolved_labels = apply_inherited_labels(samples, labels)
    rendered_samples, monitor_records, failed_samples = build_sample_artifacts(
        samples,
        resolved_labels,
        monitors,
        source_file=source_file,
    )

    rows = [copy.deepcopy(bundle.raw_row) for bundle in bundles]
    data_labels = []
    now = timestamp or datetime.now().isoformat()
    updated_sample_indices = set(updated_sample_indices or set())
    bundle_plans = bundle_plans or [
        InlineBundlePlan(
            local_label_indices=[],
            local_inherit_map={},
            preserved_labels=[],
            use_existing_data_label=False,
            pass1_changed=True,
            invalidate_pass2=(mode == "refresh"),
        )
        for _ in bundles
    ]

    for row, bundle, bundle_plan in zip(rows, bundles, bundle_plans):
        set_data_id(row, bundle.data_id)
        source_format = _infer_source_format(bundle.raw_row, bundle.samples)
        existing_meta = ((get_data_label(row, default={}) or {}).get("meta") or {})
        meta_updates = {}
        for key in ("migrated_at", "migration_source"):
            if bundle_plan.migrated and existing_meta.get(key):
                meta_updates[key] = existing_meta[key]
        if bundle_plan.use_existing_data_label and isinstance(get_data_label(row), dict):
            data_label = ensure_data_label(
                row,
                label_version=label_version,
                source_format=source_format,
                mode=mode,
                timestamp=now,
            )
        else:
            data_label = replace_data_label(
                row,
                label_version=label_version,
                source_format=source_format,
                mode=mode,
                timestamp=now,
                turns=[],
                conversation={},
                meta_updates=meta_updates or None,
            )
        set_meta_field(data_label, "source_format", source_format)
        data_labels.append(data_label)

    turns_by_bundle: list[dict[int, dict]] = [
        _existing_turn_records(bundle) if plan.use_existing_data_label else {}
        for bundle, plan in zip(bundles, bundle_plans)
    ]
    for sample_idx, sample in enumerate(samples):
        bundle_idx = sample_to_bundle[sample_idx]
        bundle_plan = bundle_plans[bundle_idx]
        default_turn_index = len(turns_by_bundle[bundle_idx]) + 1
        turn_index = _sample_turn_index(sample, default_turn_index)
        existing_turn = turns_by_bundle[bundle_idx].get(turn_index)
        should_update = sample_idx in updated_sample_indices or existing_turn is None or not bundle_plan.use_existing_data_label

        if not should_update:
            continue

        record = _turn_record(
            sample,
            resolved_labels[sample_idx],
            monitors[sample_idx],
            default_turn_index=default_turn_index,
        )
        if (
            isinstance(existing_turn, dict)
            and not bundle_plan.invalidate_pass2
            and (existing_turn.get("labels") or None) == (record.get("labels") or None)
        ):
            if isinstance(existing_turn.get("value"), dict):
                record["value"] = copy.deepcopy(existing_turn["value"])
            if isinstance(existing_turn.get("scoring_monitor"), dict):
                record["scoring_monitor"] = copy.deepcopy(existing_turn["scoring_monitor"])
        turns_by_bundle[bundle_idx][record["turn_index"]] = record
        upsert_turn_record(data_labels[bundle_idx], record)

    for bundle_idx, (bundle, bundle_plan) in enumerate(zip(bundles, bundle_plans)):
        preserved_turns = _existing_turn_records(bundle) if bundle_plan.use_existing_data_label else {}
        core_changed = False
        for turn_index, turn_record in turns_by_bundle[bundle_idx].items():
            previous = preserved_turns.get(turn_index) or {}
            if (previous.get("labels") or None) != (turn_record.get("labels") or None):
                core_changed = True
                break

        if (bundle_plan.invalidate_pass2 or core_changed) and bundle_plan.pass1_changed:
            clear_pass2_state(data_labels[bundle_idx], timestamp=now)

        turn_records = [
            turns_by_bundle[bundle_idx][turn_index]
            for turn_index in sorted(turns_by_bundle[bundle_idx])
        ]
        extension_specs = {}
        for turn_record in turn_records:
            for ext_id, payload in (turn_record.get("label_extensions") or {}).items():
                if not isinstance(payload, dict):
                    continue
                extension_specs[ext_id] = {
                    "spec_version": payload.get("spec_version"),
                    "spec_hash": payload.get("spec_hash"),
                }
        if extension_specs:
            data_labels[bundle_idx].setdefault("meta", {})["extension_specs"] = extension_specs
        compact_turns = []
        for turn_record in turn_records:
            compact_turn = compact_turn_record(turn_record, data_id=bundle.data_id)
            if compact_turn is not None:
                compact_turns.append(compact_turn)
        data_labels[bundle_idx]["turns"] = compact_turns
        conversation_payload = {}
        if bundle_plan.use_existing_data_label and not bundle_plan.invalidate_pass2:
            conversation_payload = copy.deepcopy(data_labels[bundle_idx].get("conversation", {}))
        conversation_payload.update(_conversation_record(bundle, turn_records))
        set_conversation_record(
            data_labels[bundle_idx],
            compact_conversation_record(conversation_payload),
        )
        if bundle_plan.pass1_changed or mode == "refresh":
            set_meta_field(data_labels[bundle_idx], "mode", mode)
            mark_stage_timestamp(data_labels[bundle_idx], "pass1", timestamp=now)
        compact_data_label(data_labels[bundle_idx])

    return InlinePass1MergeResult(
        rows=rows,
        samples=rendered_samples,
        monitor_records=monitor_records,
        failed_samples=failed_samples,
    )


def write_rows_jsonl(path, rows: list[dict]) -> None:
    """Write mirrored inline rows as JSONL."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json_dumps(row) + "\n")


def append_rows_jsonl(handle, rows: list[dict]) -> None:
    """Append inline rows to an already-open JSONL file handle."""
    for row in rows:
        handle.write(json_dumps(row) + "\n")


def json_dumps(value) -> str:
    return json.dumps(value, ensure_ascii=False)
