"""
Export labeled samples to a review CSV/TSV for human auditing.

Reads labeled JSON (pipeline output) and optional monitor JSONL,
produces a flat table with one row per sample.

Usage:
  python3 labeling/export_review.py \
    --input labeling/data/labeled_e2e_test.json \
    --monitor labeling/data/monitor_e2e_test.jsonl \
    --output labeling/data/review_e2e_test.csv
"""

import argparse
import csv
import json
from pathlib import Path


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


def main():
    parser = argparse.ArgumentParser(description="Export labeled samples to review CSV")
    parser.add_argument("--input", required=True, help="Labeled JSON file")
    parser.add_argument("--monitor", default="", help="Monitor JSONL file (optional)")
    parser.add_argument("--output", required=True, help="Output CSV path")
    parser.add_argument("--format", choices=["csv", "tsv"], default="csv",
                        help="Output format (default: csv, auto-detected from extension)")
    args = parser.parse_args()

    # Auto-detect format from extension
    fmt = args.format
    if args.output.endswith(".tsv"):
        fmt = "tsv"

    # Load data
    with open(args.input, "r", encoding="utf-8") as f:
        samples = json.load(f)

    monitors = {}
    if args.monitor and Path(args.monitor).exists():
        monitors = load_monitor(args.monitor)

    # Build rows
    fieldnames = [
        "id", "query",
        "intent", "difficulty", "language", "domain", "task",
        "concept", "agentic", "constraint", "context",
        "llm_calls", "elapsed_s", "tokens", "confidence_min",
    ]

    rows = []
    for sample in samples:
        sid = sample.get("id", "")
        labels = sample.get("labels")
        if labels is None:
            # Failed sample
            rows.append({"id": sid, "query": extract_query(sample.get("conversations", []))})
            continue

        mon = monitors.get(sid, sample.get("labeling_monitor", {}))

        # Token count from monitor
        total_tokens = (
            mon.get("total_prompt_tokens", 0) +
            mon.get("total_completion_tokens", 0)
        )

        row = {
            "id": sid,
            "query": extract_query(sample.get("conversations", [])),
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
        rows.append(row)

    # Write CSV/TSV
    delimiter = "\t" if fmt == "tsv" else ","
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter=delimiter,
                                extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)

    print(f"Exported {len(rows)} rows to {output_path}")
    print(f"Format: {fmt.upper()}, Columns: {len(fieldnames)}")

    # Quick stats
    has_labels = sum(1 for s in samples if s.get("labels") is not None)
    print(f"Labeled: {has_labels}/{len(samples)}")


if __name__ == "__main__":
    main()
