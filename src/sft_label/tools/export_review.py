"""Export labeled samples to a review CSV/TSV for human auditing."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from sft_label.inline_scoring import (
    discover_inline_jsonl_files,
    infer_inline_scoring_target,
    load_inline_pass1_file,
)


REVIEW_FIELDNAMES = [
    "id",
    "source_file",
    "data_id",
    "turn_index",
    "query",
    "intent",
    "difficulty",
    "language",
    "domain",
    "task",
    "concept",
    "agentic",
    "constraint",
    "context",
    "llm_calls",
    "elapsed_s",
    "tokens",
    "confidence_min",
]

EXTENSION_RARITY_PREVIEW_FIELDNAMES = [
    "extension_rarity_preview_score",
    "extension_rarity_preview_confidence",
    "extension_rarity_preview_matched_specs",
    "extension_rarity_preview_baseline_source",
]

EXTENSION_RARITY_V2_FIELDNAMES = [
    "rarity_v2_score",
    "value_score_v2",
    "selection_score_v2",
]


def load_monitor(path):
    """Load monitor JSONL into a dict keyed by sample_id."""
    monitors = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            monitors[rec.get("sample_id", rec.get("index", ""))] = rec
    return monitors


def extract_query(conversations, max_len=100):
    """Extract the first human message, truncated."""
    for msg in conversations:
        if msg.get("from") == "human":
            text = msg.get("value", "").replace("\n", " ").replace("\r", "")
            if len(text) > max_len:
                return text[:max_len] + "…"
            return text
    return ""


def find_min_confidence(labels):
    """Return the dimension with the lowest confidence score."""
    conf = labels.get("confidence", {})
    if not conf:
        return ""
    numeric = {k: v for k, v in conf.items() if isinstance(v, (int, float))}
    if not numeric:
        return ""
    min_dim = min(numeric, key=numeric.get)
    return f"{min_dim}({numeric[min_dim]:.2f})"


def join_tags(val):
    """Join list tags with comma, or return string as-is."""
    if isinstance(val, list):
        return ", ".join(val)
    return str(val) if val else ""


def _extract_extension_payloads(sample: dict) -> dict[str, dict]:
    if isinstance(sample.get("label_extensions"), dict):
        return sample["label_extensions"]
    labels = sample.get("labels") or {}
    if isinstance(labels.get("label_extensions"), dict):
        return labels["label_extensions"]
    return {}


def _collect_extension_specs(samples: list[dict]) -> dict[str, dict]:
    specs: dict[str, dict] = {}
    for sample in samples:
        payloads = _extract_extension_payloads(sample)
        for spec_id, payload in payloads.items():
            if not isinstance(payload, dict):
                continue
            spec = specs.setdefault(spec_id, {"labels": set(), "confidence": set(), "has_spec_version": False, "has_spec_hash": False})
            labels = payload.get("labels") or {}
            if isinstance(labels, dict):
                spec["labels"].update(labels.keys())
            confidence = payload.get("confidence") or {}
            if isinstance(confidence, dict):
                spec["confidence"].update(confidence.keys())
            if payload.get("spec_version") not in (None, ""):
                spec["has_spec_version"] = True
            if payload.get("spec_hash") not in (None, ""):
                spec["has_spec_hash"] = True
    return specs


def _extension_fieldnames(specs: dict[str, dict]) -> list[str]:
    fieldnames: list[str] = []
    for spec_id in sorted(specs):
        fieldnames.append(f"ext_{spec_id}_status")
        fieldnames.append(f"ext_{spec_id}_matched")
        if specs[spec_id].get("has_spec_version"):
            fieldnames.append(f"ext_{spec_id}_spec_version")
        if specs[spec_id].get("has_spec_hash"):
            fieldnames.append(f"ext_{spec_id}_spec_hash")
        for field in sorted(specs[spec_id]["labels"]):
            fieldnames.append(f"ext_{spec_id}_labels_{field}")
        for field in sorted(specs[spec_id]["confidence"]):
            fieldnames.append(f"ext_{spec_id}_confidence_{field}")
        fieldnames.append(f"ext_{spec_id}_unmapped")
    return fieldnames


def _extension_row_payload(sample: dict, specs: dict[str, dict]) -> dict[str, str]:
    payloads = _extract_extension_payloads(sample)
    rows: dict[str, str] = {}
    for spec_id in sorted(specs):
        payload = payloads.get(spec_id) if isinstance(payloads, dict) else None
        rows[f"ext_{spec_id}_status"] = payload.get("status", "") if isinstance(payload, dict) else ""
        matched = payload.get("matched") if isinstance(payload, dict) else None
        rows[f"ext_{spec_id}_matched"] = str(matched) if matched is not None else ""
        if specs[spec_id].get("has_spec_version"):
            rows[f"ext_{spec_id}_spec_version"] = payload.get("spec_version", "") if isinstance(payload, dict) else ""
        if specs[spec_id].get("has_spec_hash"):
            rows[f"ext_{spec_id}_spec_hash"] = payload.get("spec_hash", "") if isinstance(payload, dict) else ""
        labels = payload.get("labels") if isinstance(payload, dict) else None
        confidence = payload.get("confidence") if isinstance(payload, dict) else None
        for field in sorted(specs[spec_id]["labels"]):
            value = labels.get(field) if isinstance(labels, dict) else ""
            rows[f"ext_{spec_id}_labels_{field}"] = join_tags(value)
        for field in sorted(specs[spec_id]["confidence"]):
            score = confidence.get(field) if isinstance(confidence, dict) else None
            rows[f"ext_{spec_id}_confidence_{field}"] = str(score) if isinstance(score, (int, float)) else ""
        unmapped = payload.get("unmapped") if isinstance(payload, dict) else None
        if isinstance(unmapped, list):
            tokens = set()
            for item in unmapped:
                if isinstance(item, dict):
                    dim = str(item.get("dimension", "unknown"))
                    value = str(item.get("value", "?")).strip()
                    tokens.add(f"{dim}:{value}")
                else:
                    tokens.add(str(item))
            rows[f"ext_{spec_id}_unmapped"] = ", ".join(sorted(tokens))
        else:
            rows[f"ext_{spec_id}_unmapped"] = ""
    return rows


def _extension_rarity_row_payload(sample: dict) -> dict[str, str]:
    value = sample.get("value") if isinstance(sample.get("value"), dict) else {}
    rarity_extension = value.get("rarity_extension") if isinstance(value.get("rarity_extension"), dict) else {}
    rarity_v2 = value.get("rarity_v2") if isinstance(value.get("rarity_v2"), dict) else {}

    return {
        "extension_rarity_preview_score": (
            str(rarity_extension.get("score"))
            if isinstance(rarity_extension.get("score"), (int, float)) else ""
        ),
        "extension_rarity_preview_confidence": (
            str(rarity_extension.get("confidence"))
            if isinstance(rarity_extension.get("confidence"), (int, float)) else ""
        ),
        "extension_rarity_preview_matched_specs": (
            str(rarity_extension.get("matched_specs"))
            if isinstance(rarity_extension.get("matched_specs"), (int, float)) else ""
        ),
        "extension_rarity_preview_baseline_source": (
            str(rarity_extension.get("baseline_source", "")) if rarity_extension else ""
        ),
        "rarity_v2_score": (
            str(rarity_v2.get("score"))
            if isinstance(rarity_v2.get("score"), (int, float)) else ""
        ),
        "value_score_v2": (
            str(value.get("value_score_v2"))
            if isinstance(value.get("value_score_v2"), (int, float)) else ""
        ),
        "selection_score_v2": (
            str(value.get("selection_score_v2"))
            if isinstance(value.get("selection_score_v2"), (int, float)) else ""
        ),
    }


def _rows_have_any_values(rows: list[dict], fieldnames: list[str]) -> bool:
    return any(str(row.get(name, "")).strip() for row in rows for name in fieldnames)


def _load_json_or_jsonl(path: Path):
    """Load samples from one JSON or JSONL file."""
    if path.suffix == ".jsonl":
        samples = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    samples.append(json.loads(line))
        return samples

    with open(path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    if isinstance(payload, list):
        return payload
    return [payload]


def _discover_legacy_labeled_files(input_path: Path):
    """Discover labeled JSON/JSONL files in a legacy layout."""
    if input_path.is_file():
        return [input_path]

    files = sorted(set(input_path.rglob("labeled*.json")) | set(input_path.rglob("labeled*.jsonl")))
    dedup = {}
    for file_path in files:
        key = (file_path.parent, file_path.stem)
        if key in dedup and dedup[key].suffix == ".json" and file_path.suffix == ".jsonl":
            dedup[key] = file_path
        elif key not in dedup:
            dedup[key] = file_path
    return sorted(dedup.values())


def _load_legacy_review_samples(input_path: Path):
    """Load legacy sample-root labeled outputs for review export."""
    samples = []
    labeled_files = _discover_legacy_labeled_files(input_path)
    for labeled_path in labeled_files:
        for sample in _load_json_or_jsonl(labeled_path):
            metadata = sample.setdefault("metadata", {})
            metadata.setdefault("source_file", str(labeled_path))
            samples.append(sample)
    return samples


def _load_inline_review_samples(input_path: Path):
    """Load transient review samples rebuilt from inline mirrored JSONL rows."""
    target = infer_inline_scoring_target(input_path)
    if target is None:
        return None

    if target.target_path.is_file():
        source_files = [target.target_path.resolve()]
    else:
        source_files = discover_inline_jsonl_files(target)

    samples = []
    for source_file in source_files:
        _bundles, file_samples, _sample_to_bundle = load_inline_pass1_file(source_file)
        samples.extend(file_samples)
    return samples


def load_review_samples(input_path):
    """Load review-export samples from inline or legacy labeled inputs."""
    input_path = Path(input_path)
    inline_samples = _load_inline_review_samples(input_path)
    if inline_samples is not None:
        return inline_samples
    return _load_legacy_review_samples(input_path)


def build_review_rows(samples, monitors=None, *, include_extensions: bool = False, extension_specs: dict[str, dict] | None = None):
    """Build flat review rows from labeled samples."""
    monitors = monitors or {}
    extension_specs = extension_specs or {}
    rows = []
    for sample in samples:
        sid = sample.get("id", "")
        labels = sample.get("labels")
        metadata = sample.get("metadata") or {}
        source_file = metadata.get("source_file", "")
        data_id = metadata.get("data_id", "")
        turn_index = metadata.get("turn_index", "")
        query = extract_query(sample.get("conversations", []))

        if labels is None:
            rows.append({
                "id": sid,
                "source_file": source_file,
                "data_id": data_id,
                "turn_index": turn_index,
                "query": query,
            })
            continue

        mon = monitors.get(sid, sample.get("labeling_monitor", {}))
        total_tokens = (
            mon.get("total_prompt_tokens", 0) +
            mon.get("total_completion_tokens", 0)
        )

        row = {
            "id": sid,
            "source_file": source_file,
            "data_id": data_id,
            "turn_index": turn_index,
            "query": query,
            "intent": labels.get("intent", ""),
            "difficulty": labels.get("difficulty", ""),
            "language": join_tags(labels.get("language", [])),
            "domain": join_tags(labels.get("domain", [])),
            "task": join_tags(labels.get("task", [])),
            "concept": join_tags(labels.get("concept", [])),
            "agentic": join_tags(labels.get("agentic", [])),
            "constraint": join_tags(labels.get("constraint", [])),
            "context": labels.get("context", ""),
            "llm_calls": mon.get("llm_calls", ""),
            "elapsed_s": mon.get("elapsed_seconds", ""),
            "tokens": total_tokens if total_tokens > 0 else "",
            "confidence_min": find_min_confidence(labels),
        }
        if include_extensions and extension_specs:
            row.update(_extension_row_payload(sample, extension_specs))
            row.update(_extension_rarity_row_payload(sample))
        rows.append(row)
    return rows


def export_review(input_path, output_path, monitor_path="", output_format="csv", *, include_extensions: bool = False):
    """Export review rows from inline or legacy labeled inputs."""
    input_path = Path(input_path)
    output_path = Path(output_path)
    fmt = output_format
    if output_path.suffix == ".tsv":
        fmt = "tsv"

    samples = load_review_samples(input_path)
    monitors = {}
    if monitor_path and Path(monitor_path).exists():
        monitors = load_monitor(monitor_path)

    extension_specs = _collect_extension_specs(samples) if include_extensions else {}
    rows = build_review_rows(
        samples,
        monitors=monitors,
        include_extensions=include_extensions,
        extension_specs=extension_specs,
    )

    delimiter = "\t" if fmt == "tsv" else ","
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = REVIEW_FIELDNAMES
    if include_extensions:
        fieldnames = fieldnames + _extension_fieldnames(extension_specs)
        if _rows_have_any_values(rows, EXTENSION_RARITY_PREVIEW_FIELDNAMES):
            fieldnames = fieldnames + EXTENSION_RARITY_PREVIEW_FIELDNAMES
        if _rows_have_any_values(rows, EXTENSION_RARITY_V2_FIELDNAMES):
            fieldnames = fieldnames + EXTENSION_RARITY_V2_FIELDNAMES
    with open(output_path, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=fieldnames,
            delimiter=delimiter,
            extrasaction="ignore",
        )
        writer.writeheader()
        writer.writerows(rows)

    summary = {
        "rows": len(rows),
        "labeled_rows": sum(1 for sample in samples if sample.get("labels") is not None),
        "format": fmt.upper(),
        "columns": len(fieldnames),
        "output": str(output_path),
    }
    return summary


def main():
    parser = argparse.ArgumentParser(
        description="Export labeled or inline-mirrored review data to CSV/TSV",
    )
    parser.add_argument(
        "--input",
        required=True,
        help="Input labeled JSON/JSONL, mirrored inline JSONL, or run directory",
    )
    parser.add_argument("--monitor", default="", help="Monitor JSONL file (optional)")
    parser.add_argument("--output", required=True, help="Output CSV path")
    parser.add_argument(
        "--format",
        choices=["csv", "tsv"],
        default="csv",
        help="Output format (default: csv, auto-detected from extension)",
    )
    parser.add_argument(
        "--include-extensions",
        action="store_true",
        help="Include extension-label columns (opt-in).",
    )
    args = parser.parse_args()

    summary = export_review(
        args.input,
        args.output,
        monitor_path=args.monitor,
        output_format=args.format,
        include_extensions=args.include_extensions,
    )

    print(f"Exported {summary['rows']} rows to {summary['output']}")
    print(f"Format: {summary['format']}, Columns: {summary['columns']}")
    print(f"Labeled: {summary['labeled_rows']}/{summary['rows']}")


if __name__ == "__main__":
    main()
