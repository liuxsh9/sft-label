#!/usr/bin/env python3
"""Probe OpenAI-style turn compatibility against a dataset directory.

This walks a directory, discovers JSON/JSONL inputs through the same directory-discovery
path used by the pipeline, samples representative rows, and then replays each sampled row
through the file-reader / `iter_samples_from_file(..., return_row_bundles=True)` path.
"""

from __future__ import annotations

import argparse
import json
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from sft_label.pipeline import discover_input_files, iter_samples_from_file
from sft_label.preprocessing import detect_format, _resolve_turn_source_and_provenance


@dataclass
class Candidate:
    category: str
    source_file: Path
    row_number: int
    raw_row: dict


def _resolved_turns(row: dict) -> list[dict]:
    try:
        resolved = _resolve_turn_source_and_provenance(row)
    except ValueError:
        return []
    return [t for t in resolved.get("normalized_turns", []) if isinstance(t, dict)]


def _raw_role(turn: dict) -> str:
    return str(turn.get("role") or turn.get("from") or "").strip().lower()


def _content_preview(turn: dict) -> str:
    text = turn.get("content")
    if text is None:
        text = turn.get("value", "")
    if text is None:
        text = ""
    if not isinstance(text, str):
        return f"<{type(text).__name__}>"
    return text.replace("\n", " ")[:80]


def _categorize_row(row: dict) -> set[str]:
    turns = _resolved_turns(row)
    roles = [_raw_role(t) for t in turns]
    assistant_count = sum(r in {"assistant", "gpt"} for r in roles)
    has_system = "system" in roles
    has_tool = "tool" in roles
    categories: set[str] = set()
    if assistant_count == 1 and len(turns) == 2 and set(roles) <= {"user", "assistant", "human", "gpt"}:
        categories.add("single_turn")
    if assistant_count >= 2:
        categories.add("multi_turn")
    if assistant_count >= 3 and (has_system or has_tool):
        categories.add("long_trajectory")
    if assistant_count == 0:
        categories.add("zero_assistant")
    if has_system:
        categories.add("has_system")
    if has_tool:
        categories.add("has_tool")
    return categories


def _iter_json_rows(path: Path, max_rows: int) -> Iterable[tuple[int, dict]]:
    with path.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    if isinstance(payload, dict):
        payload = [payload]
    if not isinstance(payload, list):
        return
    for row_number, row in enumerate(payload, start=1):
        if max_rows > 0 and row_number > max_rows:
            break
        if isinstance(row, dict):
            yield row_number, row


def _iter_jsonl_rows(path: Path, max_rows: int) -> Iterable[tuple[int, dict]]:
    with path.open("r", encoding="utf-8") as f:
        seen = 0
        for line_number, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            yield line_number, json.loads(line)
            seen += 1
            if max_rows > 0 and seen >= max_rows:
                break


def _iter_rows(path: Path, max_rows: int) -> Iterable[tuple[int, dict]]:
    if path.suffix.lower() == ".json":
        yield from _iter_json_rows(path, max_rows)
        return
    yield from _iter_jsonl_rows(path, max_rows)


def _probe_candidate(candidate: Candidate) -> dict:
    with tempfile.TemporaryDirectory(prefix="turn-compat-probe-") as tmpdir:
        temp_path = Path(tmpdir) / "sample.jsonl"
        temp_path.write_text(json.dumps(candidate.raw_row, ensure_ascii=False) + "\n", encoding="utf-8")
        samples, n_raw, bundles, sample_to_bundle = iter_samples_from_file(temp_path, return_row_bundles=True)
    normalized_turns = samples[-1].get("conversations", []) if samples else []
    role_order = [str(t.get("from", "")) for t in normalized_turns]
    provenance = samples[0].get("metadata", {}).get("original_format") if samples else None
    assistant_samples = sum(1 for s in samples if (s.get("metadata") or {}).get("turn_index"))
    raw_turns = _resolved_turns(candidate.raw_row)
    return {
        "category": candidate.category,
        "source_file": str(candidate.source_file),
        "row_number": candidate.row_number,
        "detected_format": detect_format(candidate.raw_row),
        "assistant_replies_in_raw": sum(_raw_role(t) in {"assistant", "gpt"} for t in raw_turns),
        "n_raw": n_raw,
        "n_samples": len(samples),
        "assistant_samples_with_turn_index": assistant_samples,
        "sample_to_bundle": list(sample_to_bundle),
        "bundle_count": len(bundles),
        "provenance": provenance,
        "role_order": role_order,
        "raw_roles": [_raw_role(t) for t in raw_turns],
        "first_preview": _content_preview(raw_turns[0]) if raw_turns else "",
        "last_preview": _content_preview(raw_turns[-1]) if raw_turns else "",
    }


def _probe_candidate_safely(candidate: Candidate) -> dict:
    try:
        return _probe_candidate(candidate)
    except Exception as exc:
        return {
            "category": candidate.category,
            "source_file": str(candidate.source_file),
            "row_number": candidate.row_number,
            "error": str(exc),
        }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="Dataset directory to scan")
    ap.add_argument("--max-files", type=int, default=200, help="Max discovered files to scan")
    ap.add_argument("--max-rows-per-file", type=int, default=2000, help="Max rows to inspect per file")
    args = ap.parse_args()

    input_root = Path(args.input).expanduser().resolve()
    discovered = discover_input_files(input_root)
    files = [abs_path for abs_path, rel_path in discovered if rel_path is not None][: args.max_files]

    wanted = ["single_turn", "multi_turn", "long_trajectory", "zero_assistant"]
    chosen: dict[str, Candidate] = {}

    for path in files:
        if path.suffix.lower() not in {".jsonl", ".json"}:
            continue
        for row_number, row in _iter_rows(path, args.max_rows_per_file):
            for category in _categorize_row(row):
                if category in wanted and category not in chosen:
                    chosen[category] = Candidate(category=category, source_file=path, row_number=row_number, raw_row=row)
            if all(category in chosen for category in wanted[:3]):
                # zero_assistant is optional in the target dataset; keep scanning only if missing and cheap.
                if "zero_assistant" in chosen or row_number >= args.max_rows_per_file:
                    break
        if all(category in chosen for category in wanted[:3]) and "zero_assistant" in chosen:
            break

    print(f"Directory: {input_root}")
    print(f"Discovered files: {len(discovered)}")
    print(f"Scanned files: {len(files)}")
    print()

    for category in wanted:
        candidate = chosen.get(category)
        if candidate is None:
            print(f"[{category}] not found")
            continue
        result = _probe_candidate_safely(candidate)
        print(f"[{category}]")
        print(f"  source: {result['source_file']}:{result['row_number']}")
        if "error" in result:
            print(f"  error: {result['error']}")
            print()
            continue
        print(f"  detect_format: {result['detected_format']}")
        print(f"  provenance: {result['provenance']}")
        print(f"  raw_roles: {result['raw_roles']}")
        print(f"  normalized_role_order: {result['role_order']}")
        print(f"  assistant_replies_in_raw: {result['assistant_replies_in_raw']}")
        print(f"  n_raw_rows: {result['n_raw']}  bundle_count: {result['bundle_count']}  n_samples: {result['n_samples']}")
        print(f"  first_preview: {result['first_preview']}")
        print(f"  last_preview: {result['last_preview']}")
        print()


if __name__ == "__main__":
    main()
