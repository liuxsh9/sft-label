"""Helpers for rebuilding transient sample caches from inline-labeled JSONL rows."""

from __future__ import annotations

import copy
import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from sft_label.conversation import (
    aggregate_conversation,
    build_conversation_key,
    compute_conv_selection_scores,
)
from sft_label.fs_artifacts import is_ignored_fs_artifact
from sft_label.inline_labels import (
    build_turn_id,
    compact_conversation_record,
    compact_data_label,
    compact_scoring_monitor_summary,
    compact_turn_record,
    get_data_id,
    get_data_label,
    mark_stage_timestamp,
    set_conversation_record,
)
from sft_label.inline_rows import RowSampleBundle, iter_row_sample_bundles_from_jsonl
from sft_label.run_layout import InlineRunLayout, META_LABEL_DATA_DIRNAME
from sft_label.labels import is_usable_labels


@dataclass(frozen=True)
class InlineScoringTarget:
    """Resolved inline scoring target inside a mirrored run layout."""

    layout: InlineRunLayout
    target_path: Path

    @property
    def is_run_root(self) -> bool:
        return self.target_path == self.layout.run_root

    @property
    def is_dataset_root(self) -> bool:
        return self.target_path == self.layout.dataset_root


def _contains_mirrored_jsonl(root: Path) -> bool:
    """Return whether a directory contains at least one mirrored JSONL file."""
    return any(path.is_file() and not is_ignored_fs_artifact(path) for path in root.rglob("*.jsonl"))


def _looks_like_generated_run_dir(path: Path) -> bool:
    """Return True for standard generated directories under a run root."""
    return path.name in {
        META_LABEL_DATA_DIRNAME,
        "manifest",
        "dashboards",
    }


def _single_dataset_root(run_root: Path) -> Path | None:
    """Infer the mirrored dataset root beneath a run root."""
    dataset_dirs = []
    for child in sorted(run_root.iterdir()):
        if not child.is_dir() or _looks_like_generated_run_dir(child):
            continue
        if _contains_mirrored_jsonl(child):
            dataset_dirs.append(child)
    if len(dataset_dirs) == 1:
        return dataset_dirs[0]
    return None


def _looks_like_generated_run_file(input_path: Path) -> bool:
    """Return True for standard run artifacts that should not be treated as mirrored sources."""
    name = input_path.name
    generated_prefixes = ("labeled", "scored", "stats_", "dashboard_", "summary_stats_")
    generated_names = {
        "conversation_scores.json",
        "conversation_stats_labeling.json",
        "monitor_value.jsonl",
        "failed_value.jsonl",
        "score_failures.jsonl",
    }
    return name.startswith(generated_prefixes) or name in generated_names


def infer_inline_scoring_target(input_path) -> InlineScoringTarget | None:
    """Infer an inline mirrored run target from a run root, dataset root, or file path."""
    input_path = Path(input_path).resolve()

    if input_path.is_file():
        for ancestor in [input_path.parent, *input_path.parents]:
            has_meta_root = (ancestor / META_LABEL_DATA_DIRNAME).is_dir()
            if not has_meta_root and "_labeled_" not in ancestor.name:
                continue
            if not has_meta_root and _looks_like_generated_run_file(input_path):
                continue
            try:
                rel = input_path.relative_to(ancestor)
            except ValueError:
                continue
            if len(rel.parts) < 2:
                continue
            dataset_root = ancestor / rel.parts[0]
            if dataset_root.is_dir():
                return InlineScoringTarget(
                    layout=InlineRunLayout.from_paths(dataset_root, ancestor),
                    target_path=input_path,
                )
        return None

    if not input_path.is_dir():
        return None

    if (input_path / META_LABEL_DATA_DIRNAME).is_dir():
        dataset_root = _single_dataset_root(input_path)
        if dataset_root is not None:
            return InlineScoringTarget(
                layout=InlineRunLayout.from_paths(dataset_root, input_path),
                target_path=input_path,
            )

    if (input_path.parent / META_LABEL_DATA_DIRNAME).is_dir():
        return InlineScoringTarget(
            layout=InlineRunLayout.from_paths(input_path, input_path.parent),
            target_path=input_path,
        )

    if "_labeled_" in input_path.name:
        dataset_root = _single_dataset_root(input_path)
        if dataset_root is not None:
            return InlineScoringTarget(
                layout=InlineRunLayout.from_paths(dataset_root, input_path),
                target_path=input_path,
            )

    if "_labeled_" in input_path.parent.name and _contains_mirrored_jsonl(input_path):
        return InlineScoringTarget(
            layout=InlineRunLayout.from_paths(input_path, input_path.parent),
            target_path=input_path,
        )

    return None


def discover_inline_jsonl_files(target: InlineScoringTarget) -> list[Path]:
    """Discover mirrored dataset JSONL files covered by an inline scoring target."""
    if target.target_path.is_file():
        return [target.target_path]
    root = target.target_path if target.target_path != target.layout.run_root else target.layout.dataset_root
    return sorted(
        path.resolve()
        for path in root.rglob("*.jsonl")
        if path.is_file() and not is_ignored_fs_artifact(path)
    )


def _labels_from_turn_record(turn_record: dict) -> dict | None:
    labels = copy.deepcopy(turn_record.get("labels"))
    if not isinstance(labels, dict):
        return None
    if turn_record.get("inherited"):
        labels["inherited"] = True
    if turn_record.get("inherited_from"):
        labels["inherited_from"] = turn_record.get("inherited_from")
    return labels


def _sample_from_turn_record(
    bundle: RowSampleBundle,
    turn_record: dict,
    *,
    default_idx: int,
) -> dict:
    sample_copy = copy.deepcopy(bundle.samples[default_idx - 1])
    metadata = sample_copy.setdefault("metadata", {})
    turn_index = turn_record.get("turn_index") or metadata.get("turn_index") or default_idx
    metadata["turn_index"] = turn_index
    metadata["source_file"] = str(bundle.source_path)
    metadata["data_id"] = get_data_id(bundle.raw_row)

    stable_sample_id = build_turn_id(bundle.data_id, turn_index)
    legacy_sample_id = turn_record.get("sample_id")
    if legacy_sample_id and legacy_sample_id != stable_sample_id:
        metadata["legacy_sample_id"] = legacy_sample_id
    sample_copy["id"] = stable_sample_id

    if "labels" in turn_record:
        sample_copy["labels"] = _labels_from_turn_record(turn_record)
    if isinstance(turn_record.get("label_extensions"), dict):
        sample_copy["label_extensions"] = copy.deepcopy(turn_record["label_extensions"])
    if isinstance(turn_record.get("value"), dict):
        sample_copy["value"] = copy.deepcopy(turn_record["value"])
    if isinstance(turn_record.get("labeling_monitor"), dict):
        sample_copy["labeling_monitor"] = copy.deepcopy(turn_record["labeling_monitor"])
    if isinstance(turn_record.get("scoring_monitor"), dict):
        sample_copy["scoring_monitor"] = copy.deepcopy(turn_record["scoring_monitor"])
    return sample_copy


def embedded_samples_from_bundle(
    bundle: RowSampleBundle,
    *,
    require_labels: bool = False,
    require_usable_labels: bool = False,
    require_value: bool = False,
) -> list[dict]:
    """Extract transient samples from one inline-labeled row bundle."""
    data_label = get_data_label(bundle.raw_row, default={}) or {}
    turns = data_label.get("turns") or []
    turn_by_index = {
        turn.get("turn_index"): turn
        for turn in turns
        if isinstance(turn, dict) and turn.get("turn_index") is not None
    }

    extracted_samples: list[dict] = []
    for default_idx, sample in enumerate(bundle.samples, start=1):
        metadata = sample.get("metadata") or {}
        turn_index = metadata.get("turn_index") or default_idx
        turn_record = turn_by_index.get(turn_index)
        if turn_record is None:
            continue
        sample_copy = _sample_from_turn_record(bundle, turn_record, default_idx=default_idx)
        labels = sample_copy.get("labels")
        if require_labels and labels is None:
            continue
        if require_usable_labels and not is_usable_labels(labels):
            continue
        if require_value and not isinstance(sample_copy.get("value"), dict):
            continue
        extracted_samples.append(sample_copy)

    return extracted_samples


def scoreable_samples_from_bundle(bundle: RowSampleBundle) -> list[dict]:
    """Extract transient scoring samples from one inline-labeled row bundle."""
    return embedded_samples_from_bundle(bundle, require_usable_labels=True)


def load_inline_pass1_file(input_path, limit: int = 0):
    """Load an inline mirrored JSONL file into transient Pass 1 samples."""
    input_path = Path(input_path)
    bundles: list[RowSampleBundle] = []
    samples: list[dict] = []
    sample_to_bundle: list[int] = []
    sample_budget = 0

    for bundle in iter_row_sample_bundles_from_jsonl(input_path, limit=0):
        bundle_samples = embedded_samples_from_bundle(bundle)
        projected = sample_budget + len(bundle_samples)
        if limit > 0 and bundles and projected > limit:
            break

        bundle_idx = len(bundles)
        bundles.append(bundle)
        samples.extend(bundle_samples)
        sample_to_bundle.extend([bundle_idx] * len(bundle_samples))
        sample_budget = projected

        if limit > 0 and sample_budget >= limit:
            break

    return bundles, samples, sample_to_bundle


def load_inline_scoring_file(input_path, limit: int = 0):
    """Load an inline mirrored JSONL file into transient scoring samples."""
    input_path = Path(input_path)
    bundles: list[RowSampleBundle] = []
    samples: list[dict] = []
    sample_to_bundle: list[int] = []
    sample_budget = 0

    for bundle in iter_row_sample_bundles_from_jsonl(input_path, limit=0):
        bundle_samples = scoreable_samples_from_bundle(bundle)
        projected = sample_budget + len(bundle_samples)
        if limit > 0 and bundles and projected > limit:
            break

        bundle_idx = len(bundles)
        bundles.append(bundle)
        samples.extend(bundle_samples)
        sample_to_bundle.extend([bundle_idx] * len(bundle_samples))
        sample_budget = projected

        if limit > 0 and sample_budget >= limit:
            break

    return bundles, samples, sample_to_bundle


def inline_source_has_embedded_scores(input_path, limit: int = 1) -> bool:
    """Return whether an inline mirrored source file contains embedded Pass 2 values."""
    input_path = Path(input_path)
    found = 0
    for bundle in iter_row_sample_bundles_from_jsonl(input_path, limit=0):
        scored_samples = embedded_samples_from_bundle(bundle, require_usable_labels=True, require_value=True)
        if scored_samples:
            return True
        found += 1
        if limit > 0 and found >= limit:
            break
    return False


def write_inline_labeled_cache(source_file, artifact_dir, limit: int = 0):
    """Materialize inline embedded Pass 1 labels as rebuildable labeled caches."""
    bundles, samples, _sample_to_bundle = load_inline_scoring_file(source_file, limit=limit)
    artifact_dir = Path(artifact_dir)
    artifact_dir.mkdir(parents=True, exist_ok=True)

    labeled_json = artifact_dir / "labeled.json"
    labeled_jsonl = artifact_dir / "labeled.jsonl"
    tmp_json = labeled_json.with_name(f".{labeled_json.name}.tmp")
    with open(tmp_json, "w", encoding="utf-8") as f:
        json.dump(samples, f, ensure_ascii=False, indent=2)
    tmp_json.replace(labeled_json)

    write_jsonl_atomic(labeled_jsonl, samples)
    return bundles, samples


def write_inline_pass1_cache(source_file, artifact_dir, limit: int = 0):
    """Materialize inline embedded Pass 1 labels, including partial/missing turns."""
    bundles, samples, _sample_to_bundle = load_inline_pass1_file(source_file, limit=limit)
    artifact_dir = Path(artifact_dir)
    artifact_dir.mkdir(parents=True, exist_ok=True)

    labeled_json = artifact_dir / "labeled.json"
    labeled_jsonl = artifact_dir / "labeled.jsonl"
    write_json_atomic(labeled_json, samples)
    write_jsonl_atomic(labeled_jsonl, samples)
    return bundles, samples


def write_inline_scored_cache(source_file, artifact_dir, limit: int = 0):
    """Materialize inline embedded Pass 2 values as rebuildable scored caches."""
    input_path = Path(source_file)
    bundles: list[RowSampleBundle] = []
    samples: list[dict] = []
    sample_budget = 0

    for bundle in iter_row_sample_bundles_from_jsonl(input_path, limit=0):
        bundle_samples = embedded_samples_from_bundle(
            bundle,
            require_usable_labels=True,
            require_value=True,
        )
        projected = sample_budget + len(bundle_samples)
        if limit > 0 and bundles and projected > limit:
            break

        bundles.append(bundle)
        samples.extend(bundle_samples)
        sample_budget = projected

        if limit > 0 and sample_budget >= limit:
            break

    artifact_dir = Path(artifact_dir)
    artifact_dir.mkdir(parents=True, exist_ok=True)

    scored_json = artifact_dir / "scored.json"
    scored_jsonl = artifact_dir / "scored.jsonl"
    write_json_atomic(scored_json, samples)
    write_jsonl_atomic(scored_jsonl, samples)
    return bundles, samples


def _scoring_monitor_summary(monitor: dict | None) -> dict | None:
    return compact_scoring_monitor_summary(monitor)


def _single_turn_conversation_update(sample: dict) -> dict:
    value = sample.get("value") or {}
    metadata = sample.get("metadata") or {}
    labels = copy.deepcopy(sample.get("labels") or {})
    inherited = bool(labels.get("inherited"))
    flags = [str(flag) for flag in (value.get("flags") or []) if flag]
    source_id = metadata.get("source_id")
    if not source_id:
        data_id = metadata.get("data_id") or sample.get("id") or ""
        source_row = metadata.get("source_row")
        if source_row not in (None, ""):
            source_id = f"{data_id}::row:{source_row}"
        else:
            source_id = data_id
    source_file = metadata.get("source_file")
    conversation_uid = metadata.get("conversation_uid")
    conversation_key = build_conversation_key(source_id, source_file, conversation_uid)
    return {
        "conversation_id": source_id,
        "conversation_uid": conversation_uid or conversation_key,
        "conversation_key": conversation_key,
        "source_file": source_file,
        "turn_count": 1,
        "conv_value": value.get("value_score"),
        "conv_value_v2": value.get("value_score"),
        "conv_selection": value.get("selection_score"),
        "conv_selection_v2": value.get("selection_score"),
        "conv_selection_extension_v2": value.get("selection_score_v2") or value.get("selection_score"),
        "peak_complexity": (value.get("complexity") or {}).get("overall"),
        "conv_rarity": ((value.get("rarity") or {}).get("score")),
        "conv_rarity_extension_v2": ((value.get("rarity_extension") or {}).get("score")),
        "trajectory_structure_score": None,
        "thinking_mode": value.get("thinking_mode") or metadata.get("thinking_mode"),
        "merged_labels": labels,
        "observed_turn_ratio": 0.0 if inherited else 1.0,
        "inherited_turn_ratio": 1.0 if inherited else 0.0,
        "rarity_confidence": value.get("confidence"),
        "compression_gap": 0.0,
        "late_turn_gain": 0.0,
        "tool_turn_ratio": 0.0,
        "unique_tool_count": 0,
        "unique_file_count": 0,
        "detail": {
            "observed_turns": 0 if inherited else 1,
            "inherited_turns": 1 if inherited else 0,
            "score_confidence": value.get("confidence"),
            "quality_overall": (value.get("quality") or {}).get("overall"),
            "reasoning_overall": (value.get("reasoning") or {}).get("overall"),
            "negative_flags": flags,
            "flags": flags,
        },
    }


def _updated_rows_by_number(
    bundles: list[RowSampleBundle],
    samples_by_bundle: list[list[dict]],
    monitor_lookup: dict[str, dict],
) -> tuple[dict[int, dict], list[dict]]:
    """Build updated row payloads keyed by source line number."""
    updated_rows = {}
    conversation_records = []
    for bundle_idx, bundle in enumerate(bundles):
        updated_row, conversation = update_inline_row_with_scored_samples(
            bundle.raw_row,
            samples_by_bundle[bundle_idx],
            monitor_lookup=monitor_lookup,
        )
        updated_rows[bundle.row_number] = updated_row
        if conversation:
            conversation_records.append(conversation)
    return updated_rows, conversation_records


def update_inline_row_with_scored_samples(
    row: dict,
    scored_samples: list[dict],
    *,
    monitor_lookup: dict[str, dict] | None = None,
    timestamp: str | None = None,
) -> tuple[dict, dict | None]:
    """Write scored turn results back into one inline-labeled row."""
    row_copy = copy.deepcopy(row)
    data_label = get_data_label(row_copy, default={}) or {}
    conversation_record: dict | None = None
    turns = data_label.setdefault("turns", [])
    turn_by_index = {
        turn.get("turn_index"): turn
        for turn in turns
        if isinstance(turn, dict) and turn.get("turn_index") is not None
    }

    updated_turn_samples: list[dict] = []
    for default_idx, sample in enumerate(scored_samples, start=1):
        metadata = sample.get("metadata") or {}
        turn_index = metadata.get("turn_index") or default_idx
        turn_record = turn_by_index.get(turn_index)
        if turn_record is None:
            turn_record = {"turn_index": turn_index}
            turns.append(turn_record)
            turns.sort(key=lambda item: item.get("turn_index", 0))
            turn_by_index[turn_index] = turn_record

        value = sample.get("value")
        if isinstance(value, dict):
            turn_record["value"] = copy.deepcopy(value)
        else:
            turn_record.pop("value", None)

        monitor = None
        if monitor_lookup:
            monitor = monitor_lookup.get(sample.get("id"))
        monitor_summary = _scoring_monitor_summary(monitor)
        if monitor_summary:
            turn_record["scoring_monitor"] = monitor_summary
        elif sample.get("value"):
            turn_record["scoring_monitor"] = {"status": "success", "llm_calls": 0}
        else:
            turn_record.pop("scoring_monitor", None)

        updated_turn_samples.append(copy.deepcopy(sample))

    if updated_turn_samples:
        if len(updated_turn_samples) >= 2:
            first_meta = updated_turn_samples[0].get("metadata") or {}
            conv_key = (
                first_meta.get("conversation_uid")
                or first_meta.get("source_id")
                or get_data_id(row_copy)
            )
            conversation_update = aggregate_conversation(conv_key, updated_turn_samples)
            if conversation_update is not None:
                compute_conv_selection_scores([conversation_update])
        else:
            conversation_update = _single_turn_conversation_update(updated_turn_samples[0])

        if conversation_update:
            conversation_record = copy.deepcopy(conversation_update)
            set_conversation_record(
                data_label,
                compact_conversation_record(conversation_update),
            )
        mark_stage_timestamp(data_label, "pass2", timestamp=timestamp or datetime.now().isoformat())

    compact_turns = []
    for turn_record in sorted(turn_by_index.values(), key=lambda item: item.get("turn_index", 0)):
        compact_turn = compact_turn_record(turn_record, data_id=get_data_id(row_copy))
        if compact_turn is not None:
            compact_turns.append(compact_turn)
    data_label["turns"] = compact_turns
    compact_data_label(data_label)

    return row_copy, conversation_record


def stream_inline_rows_with_scoring(input_path, limit: int = 0):
    """Yield (row_bundle, scoreable_samples) pairs in deterministic scoring order."""
    sample_budget = 0
    for bundle in iter_row_sample_bundles_from_jsonl(input_path, limit=0):
        bundle_samples = scoreable_samples_from_bundle(bundle)
        projected = sample_budget + len(bundle_samples)
        if limit > 0 and sample_budget > 0 and projected > limit:
            break
        yield bundle, bundle_samples
        sample_budget = projected
        if limit > 0 and sample_budget >= limit:
            break


def write_jsonl_atomic(path, records) -> None:
    """Write JSONL to a temp file and replace atomically."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f".{path.name}.tmp")
    with open(tmp_path, "w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    tmp_path.replace(path)


def write_json_atomic(path, payload) -> None:
    """Write JSON to a temp file and replace atomically."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f".{path.name}.tmp")
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    tmp_path.replace(path)
