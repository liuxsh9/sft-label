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


def build_review_rows(samples, monitors=None):
    """Build flat review rows from labeled samples."""
    monitors = monitors or {}
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

        rows.append({
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
        })
    return rows


def export_review(input_path, output_path, monitor_path="", output_format="csv"):
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

    rows = build_review_rows(samples, monitors=monitors)

    delimiter = "\t" if fmt == "tsv" else ","
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=REVIEW_FIELDNAMES,
            delimiter=delimiter,
            extrasaction="ignore",
        )
        writer.writeheader()
        writer.writerows(rows)

    summary = {
        "rows": len(rows),
        "labeled_rows": sum(1 for sample in samples if sample.get("labels") is not None),
        "format": fmt.upper(),
        "columns": len(REVIEW_FIELDNAMES),
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
    args = parser.parse_args()

    summary = export_review(
        args.input,
        args.output,
        monitor_path=args.monitor,
        output_format=args.format,
    )

    print(f"Exported {summary['rows']} rows to {summary['output']}")
    print(f"Format: {summary['format']}, Columns: {summary['columns']}")
    print(f"Labeled: {summary['labeled_rows']}/{summary['rows']}")


if __name__ == "__main__":
    main()
