#!/usr/bin/env python3
"""
Sample and convert downloaded HuggingFace parquet data to sft-label formats.

Works on local parquet files (output of download_hf_dataset.py).

Usage examples:
    # Sample 200 items, output as ShareGPT JSON
    python scripts/sample_to_sft.py \
        /Volumes/MOVESPEED/datasets/open-source-sft/open-r1__Mixture-of-Thoughts/code/ \
        -n 200 -f sharegpt -o data/mot_code_200.json

    # Sample 100 items, output as Pangu JSON (converts <think> → [unused16/17])
    python scripts/sample_to_sft.py \
        /Volumes/MOVESPEED/datasets/open-source-sft/open-r1__Mixture-of-Thoughts/code/ \
        -n 100 -f pangu -o data/mot_code_pangu_100.json

    # No sampling, convert all items
    python scripts/sample_to_sft.py \
        /path/to/parquet_dir/ -f sharegpt -o data/full.json

    # Read a single parquet file
    python scripts/sample_to_sft.py \
        /path/to/file.parquet -n 50 -f sharegpt -o data/sample.json

Requires: pip install pyarrow  (included with `datasets`)
"""

from __future__ import annotations

import argparse
import json
import random
import re
import sys
from pathlib import Path


# ─────────────────────────────────────────────────────────
# Data loading (parquet + JSONL + JSON)
# ─────────────────────────────────────────────────────────

_DATA_SUFFIXES = {".parquet", ".jsonl", ".json"}


def find_data_files(path: Path) -> list[Path]:
    """Find all data files under the given path (parquet, JSONL, JSON).

    Excludes macOS resource forks (._*) and hidden directories.
    """
    if path.is_file() and path.suffix in _DATA_SUFFIXES and not path.name.startswith("._"):
        return [path]
    if path.is_dir():
        files = sorted(
            f for f in path.rglob("*")
            if f.suffix in _DATA_SUFFIXES and not f.name.startswith("._")
        )
        if not files:
            print(f"Warning: no data files found under {path}", file=sys.stderr)
        return files
    print(f"Error: {path} is not a file or directory", file=sys.stderr)
    sys.exit(1)


def _load_jsonl(path: Path) -> list[dict]:
    """Load records from a JSONL file."""
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def _load_json(path: Path) -> list[dict]:
    """Load records from a JSON file (expects array at top level)."""
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, list):
        return data
    return [data]


def load_records(files: list[Path]) -> list[dict]:
    """Load all records from data files (parquet, JSONL, or JSON)."""
    records = []
    for f in files:
        if f.suffix == ".parquet":
            try:
                import pyarrow.parquet as pq
            except ImportError:
                print("Error: pyarrow not installed. Run: uv sync --extra data", file=sys.stderr)
                sys.exit(1)
            table = pq.read_table(f)
            batch = table.to_pylist()
        elif f.suffix == ".jsonl":
            batch = _load_jsonl(f)
        elif f.suffix == ".json":
            batch = _load_json(f)
        else:
            continue
        records.extend(batch)
        print(f"  Loaded {len(batch):,} records from {f.name}")
    print(f"  Total: {len(records):,} records from {len(files)} file(s)")
    return records


# ─────────────────────────────────────────────────────────
# Format conversion
# ─────────────────────────────────────────────────────────

# HF role → ShareGPT role
_ROLE_TO_SHAREGPT = {
    "user": "human",
    "human": "human",
    "assistant": "gpt",
    "gpt": "gpt",
    "system": "system",
    "tool": "tool",
}

# HF role → Pangu role
_ROLE_TO_PANGU = {
    "user": "user",
    "human": "user",
    "assistant": "assistant",
    "gpt": "assistant",
    "system": "system",
    "tool": "tool",
}

# <think>...</think> or <thinking>...</thinking>
_THINK_RE = re.compile(r"<(think|thinking)>(.*?)</\1>", re.DOTALL)


def _extract_messages(record: dict) -> list[dict] | None:
    """Extract messages list from a HF record (handles various field names)."""
    for key in ("messages", "conversations"):
        if key in record and record[key]:
            return record[key]
    return None


def _serialize_tool_calls(tool_calls: list[dict]) -> str:
    """Serialize tool_calls array into text when content is null.

    Some HF datasets (e.g. Nemotron) use OpenAI-style tool_calls with null content.
    """
    parts = []
    for tc in tool_calls:
        func = tc.get("function", {})
        name = func.get("name", "")
        args = func.get("arguments", "{}")
        if isinstance(args, dict):
            args = json.dumps(args, ensure_ascii=False)
        parts.append(f'<tool_call>\n{{"name": "{name}", "arguments": {args}}}\n</tool_call>')
    return "\n".join(parts)


def _get_role_content(msg: dict) -> tuple[str, str]:
    """Extract (role, content) from a message dict."""
    role = msg.get("role") or msg.get("from") or ""
    content = msg.get("content") or msg.get("value") or ""
    # Handle assistant messages with null content + tool_calls array
    if not content and msg.get("tool_calls"):
        content = _serialize_tool_calls(msg["tool_calls"])
    return role, content


def to_sharegpt(record: dict, idx: int) -> dict | None:
    """Convert a HF record to ShareGPT format.

    ShareGPT: {"conversations": [{"from": "human/gpt", "value": "..."}]}
    Keeps <think>...</think> blocks as-is (sft-label knows how to handle them).
    """
    messages = _extract_messages(record)
    if not messages:
        return None

    conversations = []
    for msg in messages:
        role, content = _get_role_content(msg)
        sharegpt_role = _ROLE_TO_SHAREGPT.get(role, role)
        conversations.append({"from": sharegpt_role, "value": content})

    if not conversations:
        return None

    result = {
        "id": record.get("id") or f"hf-{idx}",
        "conversations": conversations,
    }

    # Preserve source metadata
    for key in ("source", "num_tokens", "id", "dataset", "category", "generator"):
        if key in record and record[key] is not None:
            result[key] = record[key]

    return result


def _think_to_pangu_cot(text: str) -> str:
    """Convert <think>...</think> blocks to Pangu [unused16]...[unused17] format."""
    def replacer(m):
        inner = m.group(2).strip()
        if inner:
            return f"[unused16]{inner}[unused17]"
        return ""
    return _THINK_RE.sub(replacer, text)


def to_pangu(record: dict, idx: int) -> dict | None:
    """Convert a HF record to Pangu format.

    Pangu: {"data": [{"role": "user/assistant", "content": "..."}]}
    Converts <think>...</think> → [unused16]...[unused17].
    """
    messages = _extract_messages(record)
    if not messages:
        return None

    data = []
    for msg in messages:
        role, content = _get_role_content(msg)
        pangu_role = _ROLE_TO_PANGU.get(role, role)

        # Convert thinking blocks to Pangu COT format
        if pangu_role == "assistant":
            content = _think_to_pangu_cot(content)

        data.append({"role": pangu_role, "content": content})

    if not data:
        return None

    result = {
        "id": record.get("id") or f"hf-{idx}",
        "data": data,
    }

    # Preserve source metadata
    for key in ("source", "num_tokens", "id", "dataset", "category", "generator"):
        if key in record and record[key] is not None:
            result[key] = record[key]

    return result


CONVERTERS = {
    "sharegpt": to_sharegpt,
    "pangu": to_pangu,
}


# ─────────────────────────────────────────────────────────
# Stats reporting
# ─────────────────────────────────────────────────────────

def print_stats(records: list[dict], converted: list[dict], fmt: str):
    """Print summary statistics about the converted data."""
    # Count turns
    key = "conversations" if fmt == "sharegpt" else "data"
    role_key = "from" if fmt == "sharegpt" else "role"
    assistant_role = "gpt" if fmt == "sharegpt" else "assistant"

    total_turns = sum(len(s.get(key, [])) for s in converted)
    assistant_turns = sum(
        1 for s in converted
        for t in s.get(key, [])
        if t.get(role_key) == assistant_role
    )

    # Check for thinking/COT content
    cot_count = 0
    for s in converted:
        for t in s.get(key, []):
            content = t.get("value") or t.get("content") or ""
            if "<think" in content or "[unused16]" in content:
                cot_count += 1
                break

    print(f"\n  Samples:         {len(converted):,}")
    print(f"  Total turns:     {total_turns:,}")
    print(f"  Assistant turns: {assistant_turns:,}")

    # Multi-turn and tool stats
    multi_turn = sum(1 for s in converted if len(s.get(key, [])) > 2)
    tool_turns = sum(1 for s in converted for t in s.get(key, []) if t.get(role_key) == "tool")
    print(f"  Multi-turn:      {multi_turn:,} ({100*multi_turn/max(len(converted),1):.0f}%)")
    if tool_turns:
        print(f"  Tool turns:      {tool_turns:,}")

    print(f"  With COT/think:  {cot_count:,} ({100*cot_count/max(len(converted),1):.0f}%)")

    # Source distribution (if available)
    sources = {}
    for r in converted:
        src = r.get("source", "unknown")
        sources[src] = sources.get(src, 0) + 1
    if sources and len(sources) > 1:
        print(f"  Sources:")
        for src, count in sorted(sources.items(), key=lambda x: -x[1])[:10]:
            print(f"    {src}: {count:,}")


# ─────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Sample and convert HF parquet data to sft-label formats",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("input", type=Path,
                        help="Data file or directory (parquet, JSONL, or JSON)")
    parser.add_argument("--format", "-f", choices=["sharegpt", "pangu"], default="sharegpt",
                        help="Output format (default: sharegpt)")
    parser.add_argument("--sample", "-n", type=int, default=None,
                        help="Number of samples to randomly select (default: all)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducible sampling (default: 42)")
    parser.add_argument("--output", "-o", type=Path, required=True,
                        help="Output JSON file path")

    args = parser.parse_args()

    # Load
    print(f"Loading data from {args.input} ...")
    files = find_data_files(args.input)
    if not files:
        sys.exit(1)
    records = load_records(files)

    # Sample
    total_loaded = len(records)
    if args.sample and args.sample < len(records):
        random.seed(args.seed)
        records = random.sample(records, args.sample)
        print(f"Sampled {args.sample:,} / {total_loaded:,} records (seed={args.seed})")

    # Convert
    converter = CONVERTERS[args.format]
    converted = []
    skipped = 0
    for i, rec in enumerate(records):
        result = converter(rec, i)
        if result:
            converted.append(result)
        else:
            skipped += 1

    if skipped:
        print(f"Skipped {skipped} records (no conversation data)")

    print_stats(records, converted, args.format)

    # Write
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(converted, f, ensure_ascii=False, indent=None)
        f.write("\n")

    size_mb = args.output.stat().st_size / (1024 * 1024)
    print(f"\nSaved → {args.output} ({size_mb:.1f} MB)")


if __name__ == "__main__":
    main()
