"""Sample explorer asset generation for dashboard drill-down."""

from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path

from sft_label.conversation import build_conversation_key
from sft_label.preprocessing import extract_last_turn

PREVIEW_CHUNK_SIZE = 600
DETAIL_CHUNK_SIZE = 150
PREVIEW_MAX_LEN = 220
RESPONSE_PREVIEW_MAX_LEN = 260


def _js_assign(registry: str, key: str, payload) -> str:
    return (
        f"window.__SFT_DASHBOARD_EXPLORER__ = window.__SFT_DASHBOARD_EXPLORER__ || "
        "{preview:{}, detail:{}};\n"
        f"window.__SFT_DASHBOARD_EXPLORER__.{registry}[{json.dumps(key, ensure_ascii=False)}] = "
        f"{json.dumps(payload, ensure_ascii=False)};\n"
    )


def _truncate(text: str, max_len: int) -> str:
    text = (text or "").replace("\r", "").replace("\n", " ").strip()
    if len(text) <= max_len:
        return text
    return text[: max_len - 1] + "…"


def _iter_json_or_jsonl(path: Path):
    if path.suffix == ".jsonl":
        with open(path, "r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                try:
                    yield json.loads(line)
                except json.JSONDecodeError:
                    continue
        return

    with open(path, "r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if isinstance(payload, list):
        for item in payload:
            if isinstance(item, dict):
                yield item


def _score_value(value: dict, dotted_key: str):
    current = value
    for part in dotted_key.split("."):
        if not isinstance(current, dict):
            return None
        current = current.get(part)
    return current if isinstance(current, (int, float)) else None


def _flatten_tags(labels: dict) -> tuple[list[str], list[str]]:
    if not isinstance(labels, dict):
        return [], []

    flat = []
    dims = []
    for key, value in labels.items():
        if key in {"confidence", "canonicalized", "unmapped", "inherited", "inherited_from", "partial", "partial_stage", "partial_reason", "label_extensions"}:
            continue
        if isinstance(value, list):
            for item in value:
                if item:
                    flat.append(str(item))
                    dims.append(f"{key}:{item}")
        elif value:
            flat.append(str(value))
            dims.append(f"{key}:{value}")
    return sorted(set(flat)), sorted(set(dims))


def _flatten_extension_tags(sample: dict) -> list[str]:
    payload = sample.get("label_extensions")
    if not isinstance(payload, dict):
        return []

    tags: list[str] = []
    for ext_id, ext_payload in payload.items():
        if not isinstance(ext_payload, dict):
            continue
        tags.append(f"extid:{ext_id}")
        labels = ext_payload.get("labels") or {}
        if not isinstance(labels, dict):
            continue
        for field, value in labels.items():
            tags.append(f"extfield:{ext_id}:{field}")
            if isinstance(value, list):
                for item in value:
                    if item not in (None, ""):
                        tags.append(f"ext:{ext_id}:{field}:{item}")
            elif value not in (None, ""):
                tags.append(f"ext:{ext_id}:{field}:{value}")
    return sorted(set(tags))


def _detail_payload(sample: dict) -> dict:
    payload = {
        "id": sample.get("id", ""),
        "conversations": sample.get("conversations") or [],
        "labels": sample.get("labels") or {},
        "value": sample.get("value") or {},
        "metadata": sample.get("metadata") or {},
    }
    if isinstance(sample.get("label_extensions"), dict):
        payload["label_extensions"] = sample["label_extensions"]
    if isinstance(sample.get("labeling_monitor"), dict):
        payload["labeling_monitor"] = sample["labeling_monitor"]
    if isinstance(sample.get("scoring_monitor"), dict):
        payload["scoring_monitor"] = sample["scoring_monitor"]
    return payload


def _load_conversation_lookup(path: Path | None) -> dict[str, dict]:
    if not path or not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    if not isinstance(payload, list):
        return {}

    lookup = {}
    for item in payload:
        if not isinstance(item, dict):
            continue
        key = item.get("conversation_key") or build_conversation_key(
            item.get("conversation_id"),
            item.get("source_file"),
        )
        if key:
            lookup[key] = item
    return lookup


def _preview_payload(sample: dict, *, doc_id: str, detail_chunk: str | None, scope_id: str, scope_path: str) -> dict:
    labels = sample.get("labels") or {}
    value = sample.get("value") or {}
    metadata = sample.get("metadata") or {}
    conversations = sample.get("conversations") or []
    query_text, response_text = extract_last_turn(conversations)
    flat_tags, dim_tags = _flatten_tags(labels)
    extension_tags = _flatten_extension_tags(sample)

    return {
        "doc_id": doc_id,
        "detail_chunk": detail_chunk,
        "scope_id": scope_id,
        "scope_path": scope_path,
        "sample_id": sample.get("id", ""),
        "source_file": metadata.get("source_file", ""),
        "source_id": metadata.get("source_id", ""),
        "turn_index": metadata.get("turn_index"),
        "total_turns": metadata.get("total_turns"),
        "query_preview": _truncate(query_text, PREVIEW_MAX_LEN),
        "response_preview": _truncate(response_text, RESPONSE_PREVIEW_MAX_LEN),
        "intent": labels.get("intent", ""),
        "difficulty": labels.get("difficulty", ""),
        "context": labels.get("context", ""),
        "languages": labels.get("language") or [],
        "tasks": labels.get("task") or [],
        "flat_tags": flat_tags,
        "dim_tags": dim_tags,
        "extension_tags": extension_tags,
        "thinking_mode": value.get("thinking_mode") or metadata.get("thinking_mode", ""),
        "inherited": bool(labels.get("inherited")),
        "flags": value.get("flags") or [],
        "value_score": _score_value(value, "value_score"),
        "quality_overall": _score_value(value, "quality.overall"),
        "complexity_overall": _score_value(value, "complexity.overall"),
        "reasoning_overall": _score_value(value, "reasoning.overall"),
        "selection_score": _score_value(value, "selection_score"),
        "rarity_score": _score_value(value, "rarity.score"),
        "confidence": _score_value(value, "confidence"),
    }


def _scope_slug(scope_id: str) -> str:
    digest = hashlib.sha1(scope_id.encode("utf-8")).hexdigest()[:10]
    readable = "".join(ch if ch.isalnum() else "_" for ch in scope_id).strip("_")
    readable = readable[:48] or "scope"
    return f"{readable}_{digest}"


def _chunk_relpath(scope_slug: str, kind: str, chunk_index: int) -> str:
    return f"{kind}_{scope_slug}_{chunk_index:04d}.js"


def build_explorer_assets(
    output_path: Path,
    scope_sources: list[dict],
    assets_dir: Path | None = None,
    *,
    include_detail_assets: bool = False,
) -> dict[str, dict]:
    """Write sidecar JS assets for the sample explorer.

    Returns explorer metadata keyed by scope id.
    """

    output_path = Path(output_path)
    output_dir = output_path.parent
    assets_dir = Path(assets_dir) if assets_dir is not None else output_path.with_suffix("").with_name(f"{output_path.stem}.assets")
    shutil.rmtree(assets_dir, ignore_errors=True)
    assets_dir.mkdir(parents=True, exist_ok=True)

    scope_metadata: dict[str, dict] = {}
    for scope_source in scope_sources:
        scope_id = scope_source["scope_id"]
        scope_path = scope_source.get("scope_path", "")
        data_path = Path(scope_source["data_path"])
        if not data_path.exists():
            continue
        conversation_lookup = _load_conversation_lookup(
            Path(scope_source["conversation_path"])
            if scope_source.get("conversation_path")
            else None
        )

        scope_slug = _scope_slug(scope_id)
        preview_rows = []
        preview_chunks = []
        detail_chunks = []
        preview_chunk_index = 0
        sample_count = 0
        detail_rows = {} if include_detail_assets else None
        detail_chunk_index = 0

        def flush_detail():
            nonlocal detail_rows, detail_chunk_index
            if not include_detail_assets or not detail_rows:
                return None
            chunk_key = f"{scope_id}:detail:{detail_chunk_index}"
            relpath = _chunk_relpath(scope_slug, "detail", detail_chunk_index)
            (assets_dir / relpath).write_text(
                _js_assign("detail", chunk_key, detail_rows),
                encoding="utf-8",
            )
            detail_chunks.append(
                {
                    "key": chunk_key,
                    "path": relpath,
                    "count": len(detail_rows),
                }
            )
            detail_rows = {}
            detail_chunk_index += 1
            return chunk_key

        current_detail_key = f"{scope_id}:detail:{detail_chunk_index}" if include_detail_assets else None
        for sample_count, sample in enumerate(_iter_json_or_jsonl(data_path), start=1):
            doc_id = f"{scope_id}|{sample_count - 1}"
            meta = sample.get("metadata") or {}
            conversation_key = build_conversation_key(meta.get("source_id"), meta.get("source_file"))
            conversation = conversation_lookup.get(conversation_key) or {}
            if include_detail_assets:
                detail_payload = _detail_payload(sample)
                if conversation:
                    detail_payload["conversation"] = {
                        "conversation_id": conversation.get("conversation_id"),
                        "conversation_key": conversation.get("conversation_key"),
                        "turn_count": conversation.get("turn_count"),
                        "conv_value": conversation.get("conv_value"),
                        "conv_selection": conversation.get("conv_selection"),
                        "peak_complexity": conversation.get("peak_complexity"),
                        "conv_rarity": conversation.get("conv_rarity"),
                        "observed_turn_ratio": conversation.get("observed_turn_ratio"),
                        "inherited_turn_ratio": conversation.get("inherited_turn_ratio"),
                        "rarity_confidence": conversation.get("rarity_confidence"),
                        "compression_gap": conversation.get("compression_gap"),
                        "late_turn_gain": conversation.get("late_turn_gain"),
                        "tool_turn_ratio": conversation.get("tool_turn_ratio"),
                        "unique_tool_count": conversation.get("unique_tool_count"),
                        "unique_file_count": conversation.get("unique_file_count"),
                        "thinking_mode": conversation.get("thinking_mode"),
                        "detail": conversation.get("detail") or {},
                    }
                detail_rows[doc_id] = detail_payload
            preview = _preview_payload(
                sample,
                doc_id=doc_id,
                detail_chunk=current_detail_key,
                scope_id=scope_id,
                scope_path=scope_path,
            )
            if conversation:
                preview.update(
                    {
                        "conversation_id": conversation.get("conversation_id"),
                        "conversation_key": conversation.get("conversation_key"),
                        "conv_value": conversation.get("conv_value"),
                        "conv_selection": conversation.get("conv_selection"),
                        "peak_complexity": conversation.get("peak_complexity"),
                        "turn_count": conversation.get("turn_count"),
                        "conv_rarity": conversation.get("conv_rarity"),
                        "observed_turn_ratio": conversation.get("observed_turn_ratio"),
                        "inherited_turn_ratio": conversation.get("inherited_turn_ratio"),
                        "rarity_confidence": conversation.get("rarity_confidence"),
                        "compression_gap": conversation.get("compression_gap"),
                        "late_turn_gain": conversation.get("late_turn_gain"),
                        "tool_turn_ratio": conversation.get("tool_turn_ratio"),
                        "unique_tool_count": conversation.get("unique_tool_count"),
                        "unique_file_count": conversation.get("unique_file_count"),
                    }
                )
            preview_rows.append(preview)

            if include_detail_assets and len(detail_rows) >= DETAIL_CHUNK_SIZE:
                flush_detail()
                current_detail_key = f"{scope_id}:detail:{detail_chunk_index}"

            if len(preview_rows) >= PREVIEW_CHUNK_SIZE:
                chunk_key = f"{scope_id}:preview:{preview_chunk_index}"
                relpath = _chunk_relpath(scope_slug, "preview", preview_chunk_index)
                (assets_dir / relpath).write_text(
                    _js_assign("preview", chunk_key, preview_rows),
                    encoding="utf-8",
                )
                preview_chunks.append(
                    {
                        "key": chunk_key,
                        "path": relpath,
                        "count": len(preview_rows),
                    }
                )
                preview_rows = []
                preview_chunk_index += 1

        if include_detail_assets and detail_rows:
            flush_detail()

        if preview_rows:
            chunk_key = f"{scope_id}:preview:{preview_chunk_index}"
            relpath = _chunk_relpath(scope_slug, "preview", preview_chunk_index)
            (assets_dir / relpath).write_text(
                _js_assign("preview", chunk_key, preview_rows),
                encoding="utf-8",
            )
            preview_chunks.append(
                {
                    "key": chunk_key,
                    "path": relpath,
                    "count": len(preview_rows),
                }
            )

        if sample_count <= 0:
            continue

        scope_metadata[scope_id] = {
            "sample_count": sample_count,
            "assets_dir": str(assets_dir.relative_to(output_dir)),
            "preview_chunks": preview_chunks,
            "detail_chunks": detail_chunks,
            "detail_enabled": include_detail_assets,
            "has_scores": bool(scope_source.get("has_scores")),
            "default_sort": "quality_asc" if scope_source.get("has_scores") else "sample_id_asc",
        }

    return scope_metadata
