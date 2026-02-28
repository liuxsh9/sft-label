"""
Unmapped Tag Analyzer

Scans labeled JSON for unmapped tags, aggregates by dimension and value,
and prints an actionable report for taxonomy pool iteration.

Usage:
  python3 labeling/analyze_unmapped.py labeling/data/labeled_deepseek_v4.json
  python3 labeling/analyze_unmapped.py labeling/data/labeled_*.json   # multiple files
"""

import json
import sys
from collections import defaultdict
from pathlib import Path

from sft_label.prompts import TAG_POOLS


def guess_dimension(value):
    """Try to match an unmapped string value to its most likely dimension."""
    for dim, pool in TAG_POOLS.items():
        if value in pool:
            return dim  # actually mapped — shouldn't happen, but handle gracefully
    # Heuristic: check if it looks like a known dimension's style
    return "unknown"


def analyze(paths):
    # dimension -> value -> list of (sample_id, query_preview)
    unmapped = defaultdict(lambda: defaultdict(list))
    total_samples = 0
    total_labeled = 0

    for path in paths:
        with open(path, "r", encoding="utf-8") as f:
            samples = json.load(f)
        for sample in samples:
            total_samples += 1
            labels = sample.get("labels")
            if labels is None:
                continue
            total_labeled += 1
            sid = sample.get("id", "?")
            query = ""
            for msg in sample.get("conversations", []):
                if msg.get("from") == "human":
                    query = msg["value"].replace("\n", " ")[:80]
                    break
            for item in labels.get("unmapped", []):
                if isinstance(item, dict):
                    dim = item.get("dimension", "unknown")
                    val = item.get("value", "?")
                else:
                    val = str(item)
                    dim = guess_dimension(val)
                unmapped[dim][val].append((sid, query))

    return unmapped, total_samples, total_labeled


def print_report(unmapped, total_samples, total_labeled, paths):
    total_occurrences = sum(len(v) for dim in unmapped.values() for v in dim.values())
    unique_tags = sum(len(dim) for dim in unmapped.values())

    print(f"{'='*70}")
    print(f"Unmapped Tag Analysis")
    print(f"{'='*70}")
    print(f"Files:    {', '.join(str(p) for p in paths)}")
    print(f"Samples:  {total_labeled} labeled / {total_samples} total")
    print(f"Unmapped: {total_occurrences} occurrences, {unique_tags} unique tags")

    if not unmapped:
        print(f"\nNo unmapped tags found. Tag pools are fully covering.")
        return

    # Sort dimensions by total occurrences
    dim_totals = {dim: sum(len(samples) for samples in vals.values())
                  for dim, vals in unmapped.items()}

    for dim in sorted(dim_totals, key=dim_totals.get, reverse=True):
        vals = unmapped[dim]
        print(f"\n{'─'*70}")
        print(f"  {dim.upper()}  ({dim_totals[dim]} occurrences, {len(vals)} unique)")
        print(f"{'─'*70}")

        # Sort values by frequency
        for val in sorted(vals, key=lambda v: len(vals[v]), reverse=True):
            samples = vals[val]
            freq = len(samples)
            pct = freq / total_labeled * 100
            action = "ADD" if freq >= 3 else "review" if freq >= 2 else "skip?"
            print(f"  {val:30s}  {freq:3d}x  ({pct:4.1f}%)  → {action}")
            # Show up to 2 example queries
            for sid, query in samples[:2]:
                print(f"    eg. [{sid}] {query}")

    # Summary recommendation
    print(f"\n{'='*70}")
    print(f"Recommendations:")
    print(f"{'='*70}")
    candidates = []
    for dim, vals in unmapped.items():
        for val, samples in vals.items():
            if len(samples) >= 3:
                candidates.append((dim, val, len(samples)))
    if candidates:
        candidates.sort(key=lambda x: -x[2])
        print(f"  Tags to ADD to pool (freq >= 3):")
        for dim, val, freq in candidates:
            print(f"    taxonomy/tags/{dim}.yaml  ←  {val} ({freq}x)")
    else:
        print(f"  No high-frequency unmapped tags. Pool coverage is good.")

    low = [(dim, val, len(s)) for dim, vals in unmapped.items()
           for val, s in vals.items() if len(s) < 3]
    if low:
        print(f"  Low-frequency (review manually): {len(low)} tags")


def main():
    if len(sys.argv) < 2:
        print("Usage: python3 labeling/analyze_unmapped.py <labeled.json> [...]")
        sys.exit(1)

    paths = [Path(p) for p in sys.argv[1:]]
    for p in paths:
        if not p.exists():
            print(f"Error: {p} not found")
            sys.exit(1)

    unmapped, total_samples, total_labeled = analyze(paths)
    print_report(unmapped, total_samples, total_labeled, paths)


if __name__ == "__main__":
    main()
