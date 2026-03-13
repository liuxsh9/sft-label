"""Helpers for inline row annotations stored under extra_info.unique_info."""

from __future__ import annotations

import copy
import hashlib
import json
from datetime import datetime


EXTRA_INFO_FIELD = "extra_info"
UNIQUE_INFO_FIELD = "unique_info"
DATA_ID_FIELD = "data_id"
DATA_LABEL_FIELD = "data_label"

DATA_ID_ALGORITHM = "sha256"
DATA_ID_VERSION = "v1"
DATA_LABEL_SCHEMA_VERSION = "1"
DEFAULT_LABEL_VERSION = "inline-v1"
TURN_ID_ALGORITHM = "sha256"
TURN_ID_VERSION = "v1"
TURN_ID_HEX_LEN = 20

_INLINE_CONVERSATION_KEYS = (
    "conversation_id",
    "conversation_key",
    "source_file",
    "turn_count",
    "conv_value",
    "conv_selection",
    "peak_complexity",
    "conv_rarity",
    "observed_turn_ratio",
    "inherited_turn_ratio",
    "rarity_confidence",
    "compression_gap",
    "late_turn_gain",
    "tool_turn_ratio",
    "unique_tool_count",
    "unique_file_count",
    "thinking_mode",
)

_INLINE_CONVERSATION_DETAIL_KEYS = (
    "q_base",
    "penalty",
    "quality_floor",
    "negative_flags",
    "pure_quality",
    "score_confidence",
    "observed_turns",
    "inherited_turns",
    "observed_weight_ratio",
    "coverage_confidence",
    "rarity_mean",
    "rarity_peak",
    "rarity_diversity_bonus",
    "label_signature_count",
    "conv_intra_class_rank",
    "selection_confidence",
    "turn_value_mean",
    "turn_value_min",
    "turn_value_max",
    "turn_value_std",
    "turn_quality_std",
    "top_k_mean",
    "bottom_k_mean",
    "peak_minus_mean",
    "late_turn_gain",
    "tool_turn_count",
    "unique_tools",
    "test_related_turn_count",
    "edit_related_turn_count",
    "bash_execution_turn_count",
)


def _stable_json_dumps(value) -> str:
    """Serialize nested data deterministically for identity hashing."""
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def build_data_id_payload(row: dict) -> dict:
    """Build the canonical payload used for stable row identity."""
    payload_data = row.get("data")
    if payload_data is None and "conversations" in row:
        payload_data = row.get("conversations")
    return {
        "meta_prompt": row.get("meta_prompt"),
        "data": payload_data,
    }


def compute_data_id(row: dict) -> str:
    """Compute a deterministic identity from training content only."""
    payload = build_data_id_payload(row)
    digest = hashlib.sha256(_stable_json_dumps(payload).encode("utf-8")).hexdigest()
    return f"{DATA_ID_ALGORITHM}:{DATA_ID_VERSION}:{digest}"


def build_turn_id(data_id: str, turn_index: int) -> str:
    """Build a short deterministic turn identifier from row identity + turn index."""
    payload = f"{data_id}#{int(turn_index)}"
    digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()[:TURN_ID_HEX_LEN]
    return f"tid:{TURN_ID_VERSION}:{digest}"


def get_extra_info(row: dict, *, create: bool = False) -> dict | None:
    """Get or create the extra_info container."""
    extra_info = row.get(EXTRA_INFO_FIELD)
    if extra_info is None:
        if not create:
            return None
        extra_info = {}
        row[EXTRA_INFO_FIELD] = extra_info
    if not isinstance(extra_info, dict):
        raise TypeError(f"{EXTRA_INFO_FIELD} must be a dict when present")
    return extra_info


def get_unique_info(row: dict, *, create: bool = False) -> dict | None:
    """Get or create the unique_info container."""
    extra_info = get_extra_info(row, create=create)
    if extra_info is None:
        return None

    unique_info = extra_info.get(UNIQUE_INFO_FIELD)
    if unique_info is None:
        if not create:
            return None
        unique_info = {}
        extra_info[UNIQUE_INFO_FIELD] = unique_info
    if not isinstance(unique_info, dict):
        raise TypeError(f"{EXTRA_INFO_FIELD}.{UNIQUE_INFO_FIELD} must be a dict when present")
    return unique_info


def get_data_id(row: dict, default=None):
    """Return the embedded data_id if present."""
    unique_info = get_unique_info(row, create=False)
    if unique_info is None:
        return default
    return unique_info.get(DATA_ID_FIELD, default)


def set_data_id(row: dict, data_id: str) -> str:
    """Persist a data_id under extra_info.unique_info."""
    unique_info = get_unique_info(row, create=True)
    unique_info[DATA_ID_FIELD] = data_id
    return data_id


def ensure_data_id(row: dict) -> str:
    """Return an existing embedded data_id or create one."""
    existing = get_data_id(row)
    if existing:
        return existing
    return set_data_id(row, compute_data_id(row))


def get_data_label(row: dict, default=None):
    """Return the embedded data_label if present."""
    unique_info = get_unique_info(row, create=False)
    if unique_info is None:
        return default
    return unique_info.get(DATA_LABEL_FIELD, default)


def get_data_label_meta(row: dict, default=None):
    """Return the embedded data_label.meta mapping if present."""
    data_label = get_data_label(row, default=None)
    if not isinstance(data_label, dict):
        return default
    meta = data_label.get("meta")
    if not isinstance(meta, dict):
        return default
    return meta


def build_data_label(
    *,
    label_version: str = DEFAULT_LABEL_VERSION,
    source_format: str = "unknown",
    mode: str = "incremental",
    timestamp: str | None = None,
    turns: list[dict] | None = None,
    conversation: dict | None = None,
    meta_updates: dict | None = None,
) -> dict:
    """Build a fresh embedded data_label payload."""
    now = timestamp or datetime.now().isoformat()
    meta = {
        "schema_version": DATA_LABEL_SCHEMA_VERSION,
        "label_version": label_version,
        "source_format": source_format,
        "mode": mode,
        "created_at": now,
        "updated_at": now,
        "pass1_at": None,
        "pass2_at": None,
        "recomputed_at": None,
        "migrated_at": None,
        "migration_source": None,
    }
    if meta_updates:
        meta.update(copy.deepcopy(meta_updates))
    return {
        "meta": meta,
        "turns": copy.deepcopy(turns or []),
        "conversation": copy.deepcopy(conversation or {}),
    }


def set_data_label(row: dict, data_label: dict) -> dict:
    """Persist a data_label under extra_info.unique_info."""
    unique_info = get_unique_info(row, create=True)
    unique_info[DATA_LABEL_FIELD] = copy.deepcopy(data_label)
    return unique_info[DATA_LABEL_FIELD]


def replace_data_label(
    row: dict,
    *,
    label_version: str = DEFAULT_LABEL_VERSION,
    source_format: str = "unknown",
    mode: str = "refresh",
    timestamp: str | None = None,
    turns: list[dict] | None = None,
    conversation: dict | None = None,
    meta_updates: dict | None = None,
) -> dict:
    """Replace the embedded data_label with a fresh payload."""
    data_label = build_data_label(
        label_version=label_version,
        source_format=source_format,
        mode=mode,
        timestamp=timestamp,
        turns=turns,
        conversation=conversation,
        meta_updates=meta_updates,
    )
    return set_data_label(row, data_label)


def ensure_data_label(
    row: dict,
    *,
    label_version: str = DEFAULT_LABEL_VERSION,
    source_format: str = "unknown",
    mode: str = "incremental",
    timestamp: str | None = None,
) -> dict:
    """Return an existing data_label or create a fresh one."""
    existing = get_data_label(row)
    if isinstance(existing, dict):
        return existing
    return replace_data_label(
        row,
        label_version=label_version,
        source_format=source_format,
        mode=mode,
        timestamp=timestamp,
    )


def set_meta_field(data_label: dict, key: str, value) -> dict:
    """Set a single metadata field on an embedded data_label."""
    data_label.setdefault("meta", {})
    data_label["meta"][key] = value
    return data_label


def compact_data_label_meta(meta: dict | None) -> dict:
    """Drop empty metadata entries from an embedded data_label.meta payload."""
    if not isinstance(meta, dict):
        return {}
    compact = {}
    for key, value in meta.items():
        if value is None:
            continue
        compact[key] = copy.deepcopy(value)
    return compact


def compact_labeling_monitor_summary(monitor: dict | None) -> dict | None:
    """Keep only meaningful inline Pass 1 monitor fields."""
    if not isinstance(monitor, dict):
        return None

    summary = {}
    status = monitor.get("status")
    if status:
        summary["status"] = status

    llm_calls = monitor.get("llm_calls", 0)
    if llm_calls:
        summary["llm_calls"] = llm_calls

    prompt_tokens = monitor.get("total_prompt_tokens", monitor.get("prompt_tokens", 0))
    if prompt_tokens:
        summary["total_prompt_tokens"] = prompt_tokens

    completion_tokens = monitor.get("total_completion_tokens", monitor.get("completion_tokens", 0))
    if completion_tokens:
        summary["total_completion_tokens"] = completion_tokens

    elapsed_seconds = monitor.get("elapsed_seconds")
    if elapsed_seconds is not None:
        summary["elapsed_seconds"] = elapsed_seconds

    if monitor.get("arbitrated"):
        summary["arbitrated"] = True

    validation_issues = monitor.get("validation_issues") or []
    if validation_issues:
        summary["validation_issues"] = copy.deepcopy(validation_issues)

    consistency_warnings = monitor.get("consistency_warnings") or []
    if consistency_warnings:
        summary["consistency_warnings"] = copy.deepcopy(consistency_warnings)

    error = monitor.get("error")
    if error:
        summary["error"] = str(error)[:500]

    error_response = monitor.get("error_response")
    if error_response:
        summary["error_response"] = str(error_response)[:500]

    return summary or None


def compact_scoring_monitor_summary(monitor: dict | None) -> dict | None:
    """Keep only meaningful inline Pass 2 monitor fields."""
    if not isinstance(monitor, dict):
        return None

    summary = {}
    status = monitor.get("status")
    if status:
        summary["status"] = status

    llm_calls = monitor.get("llm_calls", 0)
    if llm_calls:
        summary["llm_calls"] = llm_calls

    prompt_tokens = monitor.get("prompt_tokens", 0)
    if prompt_tokens:
        summary["prompt_tokens"] = prompt_tokens

    completion_tokens = monitor.get("completion_tokens", 0)
    if completion_tokens:
        summary["completion_tokens"] = completion_tokens

    attempts = monitor.get("attempts", 0)
    if attempts:
        summary["attempts"] = attempts

    validation_issues = monitor.get("validation_issues") or []
    if validation_issues:
        summary["validation_issues"] = copy.deepcopy(validation_issues)

    error = monitor.get("error")
    if error:
        summary["error"] = str(error)[:500]

    error_response = monitor.get("error_response")
    if error_response:
        summary["error_response"] = str(error_response)[:500]

    return summary or None


def compact_turn_record(turn_record: dict | None, *, data_id: str | None = None) -> dict | None:
    """Drop redundant inline turn fields while keeping resume-critical state."""
    if not isinstance(turn_record, dict):
        return None

    turn_index = turn_record.get("turn_index")
    if turn_index is None:
        return None

    compact = {"turn_index": turn_index}

    sample_id = build_turn_id(data_id, turn_index) if data_id else turn_record.get("sample_id")
    if sample_id:
        compact["sample_id"] = sample_id

    labels = turn_record.get("labels")
    if labels is not None:
        compact["labels"] = copy.deepcopy(labels)

    if turn_record.get("inherited"):
        compact["inherited"] = True

    inherited_from = turn_record.get("inherited_from")
    if inherited_from:
        compact["inherited_from"] = inherited_from

    labeling_monitor = compact_labeling_monitor_summary(turn_record.get("labeling_monitor"))
    if labeling_monitor:
        compact["labeling_monitor"] = labeling_monitor

    value = turn_record.get("value")
    if isinstance(value, dict):
        compact["value"] = copy.deepcopy(value)

    scoring_monitor = compact_scoring_monitor_summary(turn_record.get("scoring_monitor"))
    if scoring_monitor:
        compact["scoring_monitor"] = scoring_monitor

    return compact


def compact_conversation_record(conversation_record: dict | None) -> dict:
    """Persist only stable conversation-level aggregates on inline rows."""
    if not isinstance(conversation_record, dict):
        return {}

    compact = {}
    for key in _INLINE_CONVERSATION_KEYS:
        value = conversation_record.get(key)
        if value is None:
            continue
        if key in {"conversation_id", "conversation_key", "source_file", "thinking_mode"} and value == "":
            continue
        compact[key] = copy.deepcopy(value)

    detail = conversation_record.get("detail")
    if isinstance(detail, dict):
        compact_detail = {}
        for key in _INLINE_CONVERSATION_DETAIL_KEYS:
            value = detail.get(key)
            if value is None:
                continue
            if key == "unique_tools" and not value:
                continue
            compact_detail[key] = copy.deepcopy(value)
        if compact_detail:
            compact["detail"] = compact_detail
    return compact


def compact_data_label(data_label: dict | None) -> dict:
    """Apply inline schema compaction to metadata, turns, and conversation."""
    if not isinstance(data_label, dict):
        return {}

    turns = []
    for turn in data_label.get("turns", []) or []:
        compact_turn = compact_turn_record(turn)
        if compact_turn is not None:
            turns.append(compact_turn)
    turns.sort(key=lambda item: item.get("turn_index", 0))

    data_label["meta"] = compact_data_label_meta(data_label.get("meta"))
    data_label["turns"] = turns
    data_label["conversation"] = compact_conversation_record(data_label.get("conversation"))
    return data_label


def mark_stage_timestamp(data_label: dict, stage: str, timestamp: str | None = None) -> dict:
    """Update a stage timestamp and the overall updated_at marker."""
    now = timestamp or datetime.now().isoformat()
    set_meta_field(data_label, f"{stage}_at", now)
    set_meta_field(data_label, "updated_at", now)
    return data_label


def mark_migrated(
    data_label: dict,
    *,
    migration_source: str | None = None,
    timestamp: str | None = None,
) -> dict:
    """Record migration provenance on the embedded data_label."""
    now = timestamp or datetime.now().isoformat()
    set_meta_field(data_label, "migrated_at", now)
    set_meta_field(data_label, "updated_at", now)
    set_meta_field(data_label, "migration_source", migration_source)
    return data_label


def clear_pass2_state(data_label: dict, *, timestamp: str | None = None) -> dict:
    """Drop Pass 2-derived fields after Pass 1 changes invalidate them."""
    now = timestamp or datetime.now().isoformat()
    for turn in data_label.get("turns", []) or []:
        if isinstance(turn, dict):
            turn.pop("value", None)
            turn.pop("scoring_monitor", None)

    conversation = data_label.get("conversation")
    if isinstance(conversation, dict):
        for key in (
            "conversation_key",
            "source_file",
            "conv_value",
            "conv_selection",
            "peak_complexity",
            "conv_rarity",
            "observed_turn_ratio",
            "inherited_turn_ratio",
            "rarity_confidence",
            "compression_gap",
            "late_turn_gain",
            "tool_turn_ratio",
            "unique_tool_count",
            "unique_file_count",
            "thinking_mode",
            "merged_labels",
            "detail",
            "slices",
        ):
            conversation.pop(key, None)

    set_meta_field(data_label, "pass2_at", None)
    set_meta_field(data_label, "updated_at", now)
    return data_label


def upsert_turn_record(data_label: dict, turn_record: dict) -> dict:
    """Insert or replace a turn record by turn_index while keeping order."""
    if "turn_index" not in turn_record:
        raise KeyError("turn_record must include turn_index")

    turns = data_label.setdefault("turns", [])
    turn_index = turn_record["turn_index"]
    payload = copy.deepcopy(turn_record)
    for idx, existing in enumerate(turns):
        if existing.get("turn_index") == turn_index:
            turns[idx] = payload
            break
    else:
        turns.append(payload)
        turns.sort(key=lambda item: item.get("turn_index", 0))
    return data_label


def set_conversation_record(data_label: dict, conversation_record: dict) -> dict:
    """Replace the conversation aggregate payload."""
    data_label["conversation"] = copy.deepcopy(conversation_record)
    return data_label
