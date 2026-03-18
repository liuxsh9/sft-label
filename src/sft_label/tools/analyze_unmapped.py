"""Inspect unmapped tags from labeled/scored outputs or labeling stats."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Iterable

from sft_label.artifacts import (
    PASS1_STATS_FILE,
    PASS1_STATS_FILE_LEGACY,
    find_first_existing,
)
from sft_label.inline_scoring import (
    discover_inline_jsonl_files,
    infer_inline_scoring_target,
    load_inline_pass1_file,
)
from sft_label.prompts import TAG_POOLS


DEFAULT_TOP = 20
DEFAULT_EXAMPLES = 2
_STATS_CANDIDATES = (PASS1_STATS_FILE, PASS1_STATS_FILE_LEGACY)


def guess_dimension(value: str) -> str:
    """Try to match an unmapped string value to its most likely dimension."""
    for dim, pool in TAG_POOLS.items():
        if value in pool:
            return dim
    return "unknown"


def _iter_json_or_jsonl(path: Path):
    """Yield records from a JSON or JSONL file."""
    if path.suffix == ".jsonl":
        with open(path, "r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if line:
                    yield json.loads(line)
        return

    with open(path, "r", encoding="utf-8") as handle:
        payload = json.load(handle)

    if isinstance(payload, list):
        yield from payload
    elif isinstance(payload, dict) and isinstance(payload.get("samples"), list):
        yield from payload["samples"]
    else:
        yield payload


def _discover_legacy_sample_files(input_path: Path) -> list[Path]:
    """Discover labeled/scored files in a legacy run layout."""
    if input_path.is_file():
        return [input_path]

    all_files = sorted(
        set(input_path.rglob("labeled*.json"))
        | set(input_path.rglob("labeled*.jsonl"))
        | set(input_path.rglob("scored*.json"))
        | set(input_path.rglob("scored*.jsonl"))
    )
    files_by_parent: defaultdict[Path, list[Path]] = defaultdict(list)
    for file_path in all_files:
        files_by_parent[file_path.parent].append(file_path)

    dedup: dict[tuple[Path, str], Path] = {}
    for parent, parent_files in files_by_parent.items():
        labeled_files = [path for path in parent_files if path.name.startswith("labeled")]
        selected_files = labeled_files or parent_files
        for file_path in selected_files:
            key = (parent, file_path.stem)
            existing = dedup.get(key)
            if existing is None or (existing.suffix == ".json" and file_path.suffix == ".jsonl"):
                dedup[key] = file_path
    return sorted(dedup.values())


def _load_inline_samples(input_path: Path) -> tuple[list[dict], list[Path]] | None:
    """Load transient samples from an inline mirrored dataset/run."""
    target = infer_inline_scoring_target(input_path)
    if target is None:
        return None

    if target.target_path.is_file():
        source_files = [target.target_path.resolve()]
    else:
        source_files = discover_inline_jsonl_files(target)

    samples: list[dict] = []
    for source_file in source_files:
        _bundles, file_samples, _sample_to_bundle = load_inline_pass1_file(source_file)
        samples.extend(file_samples)
    return samples, source_files


def _load_samples(input_path: Path) -> tuple[list[dict], list[Path], str] | None:
    """Load samples from inline or legacy layouts."""
    inline = _load_inline_samples(input_path)
    if inline is not None:
        samples, source_files = inline
        return samples, source_files, "inline"

    if input_path.is_file():
        if input_path.name in _STATS_CANDIDATES:
            return None
        return list(_iter_json_or_jsonl(input_path)), [input_path], "file"

    if input_path.is_dir():
        files = _discover_legacy_sample_files(input_path)
        if not files:
            return None
        samples: list[dict] = []
        for file_path in files:
            samples.extend(_iter_json_or_jsonl(file_path))
        return samples, files, "directory"

    raise FileNotFoundError(f"Input path not found: {input_path}")


def _resolve_stats_path(input_path: Path) -> Path | None:
    """Resolve a labeling stats file from a path or run directory."""
    if input_path.is_file():
        return input_path if input_path.name in _STATS_CANDIDATES else None
    if input_path.is_dir():
        return find_first_existing(input_path, _STATS_CANDIDATES)
    return None


def analyze_samples(samples: Iterable[dict]) -> tuple[dict[str, dict[str, list[tuple[str, str]]]], int, int]:
    """Aggregate unmapped tags by dimension and value from sample payloads."""
    unmapped: defaultdict[str, defaultdict[str, list[tuple[str, str]]]] = defaultdict(
        lambda: defaultdict(list)
    )
    total_samples = 0
    total_labeled = 0

    for sample in samples:
        total_samples += 1
        labels = sample.get("labels")
        if labels is None:
            continue
        total_labeled += 1
        sample_id = sample.get("id", "?")
        query = ""
        for msg in sample.get("conversations", []):
            if msg.get("from") == "human":
                query = str(msg.get("value", "")).replace("\n", " ")[:80]
                break
        for item in labels.get("unmapped", []):
            if isinstance(item, dict):
                dim = str(item.get("dimension", "unknown"))
                value = str(item.get("value", "?"))
            else:
                value = str(item)
                dim = guess_dimension(value)
            unmapped[dim][value].append((str(sample_id), query))

    return dict(unmapped), total_samples, total_labeled


def _stats_to_unmapped(stats_payload: dict) -> dict[str, dict[str, int]]:
    """Convert flattened stats `unmapped_tags` to dimension/value counters."""
    grouped: defaultdict[str, dict[str, int]] = defaultdict(dict)
    for key, count in (stats_payload.get("unmapped_tags") or {}).items():
        dim, value = str(key).split(":", 1) if ":" in str(key) else ("unknown", str(key))
        grouped[dim][value] = int(count)
    return dict(grouped)


def _filtered_dims(
    unmapped: dict[str, dict[str, object]],
    *,
    dimension: str | None,
) -> list[str]:
    dims = list(unmapped)
    if dimension:
        dims = [dim for dim in dims if dim == dimension]
    return sorted(
        dims,
        key=lambda dim: sum(
            len(v) if isinstance(v, list) else int(v)
            for v in unmapped.get(dim, {}).values()
        ),
        reverse=True,
    )


def print_sample_report(
    unmapped: dict[str, dict[str, list[tuple[str, str]]]],
    *,
    total_samples: int,
    total_labeled: int,
    paths: Iterable[Path],
    top: int,
    examples: int,
    dimension: str | None,
) -> int:
    """Print a detailed unmapped report with example samples."""
    total_occurrences = sum(len(v) for dim in unmapped.values() for v in dim.values())
    unique_tags = sum(len(dim) for dim in unmapped.values())
    selected_dims = _filtered_dims(unmapped, dimension=dimension)

    print("=" * 70)
    print("Unmapped Tag Analysis")
    print("=" * 70)
    print(f"Files:    {', '.join(str(p) for p in paths)}")
    print(f"Samples:  {total_labeled} labeled / {total_samples} total")
    print(f"Unmapped: {total_occurrences} occurrences, {unique_tags} unique tags")

    if dimension and not selected_dims:
        print(f"\nNo unmapped tags found for dimension '{dimension}'.")
        return 0

    if total_occurrences == 0:
        print("\nNo unmapped tags found. Tag pools are fully covering.")
        return 0

    for dim in selected_dims:
        values = unmapped[dim]
        dim_total = sum(len(items) for items in values.values())
        print(f"\n{'-' * 70}")
        print(f"  {dim.upper()}  ({dim_total} occurrences, {len(values)} unique)")
        print(f"{'-' * 70}")
        ranked_values = sorted(values, key=lambda value: len(values[value]), reverse=True)
        for value in ranked_values[:top]:
            sample_refs = values[value]
            freq = len(sample_refs)
            pct = (freq / total_labeled * 100) if total_labeled else 0.0
            action = "ADD" if freq >= 3 else "review" if freq >= 2 else "skip?"
            print(f"  {value:30s}  {freq:3d}x  ({pct:4.1f}%)  -> {action}")
            for sample_id, query in sample_refs[:examples]:
                print(f"    eg. [{sample_id}] {query}")

    candidates = []
    for dim, values in unmapped.items():
        for value, sample_refs in values.items():
            if len(sample_refs) >= 3:
                candidates.append((dim, value, len(sample_refs)))
    candidates.sort(key=lambda item: -item[2])

    print(f"\n{'=' * 70}")
    print("Recommendations")
    print("=" * 70)
    if candidates:
        print("  Tags to add to pool (freq >= 3):")
        for dim, value, freq in candidates[:top]:
            print(f"    taxonomy/tags/{dim}.yaml <- {value} ({freq}x)")
    else:
        print("  No high-frequency unmapped tags. Pool coverage is good.")

    low_freq = [
        (dim, value, len(sample_refs))
        for dim, values in unmapped.items()
        for value, sample_refs in values.items()
        if len(sample_refs) < 3
    ]
    if low_freq:
        print(f"  Low-frequency (review manually): {len(low_freq)} tags")
    return total_occurrences


def print_stats_report(
    unmapped: dict[str, dict[str, int]],
    *,
    stats_path: Path,
    top: int,
    dimension: str | None,
) -> int:
    """Print an unmapped summary from stats-only artifacts."""
    total_occurrences = sum(count for dim in unmapped.values() for count in dim.values())
    unique_tags = sum(len(dim) for dim in unmapped.values())
    selected_dims = _filtered_dims(unmapped, dimension=dimension)

    print("=" * 70)
    print("Unmapped Tag Summary")
    print("=" * 70)
    print(f"Stats:    {stats_path}")
    print(f"Unmapped: {total_occurrences} occurrences, {unique_tags} unique tags")

    if dimension and not selected_dims:
        print(f"\nNo unmapped tags found for dimension '{dimension}'.")
        return 0

    if total_occurrences == 0:
        print("\nNo unmapped tags found in labeling stats.")
        return 0

    for dim in selected_dims:
        values = unmapped[dim]
        dim_total = sum(values.values())
        print(f"\n{'-' * 70}")
        print(f"  {dim.upper()}  ({dim_total} occurrences, {len(values)} unique)")
        print(f"{'-' * 70}")
        for value, count in sorted(values.items(), key=lambda item: -item[1])[:top]:
            action = "ADD" if count >= 3 else "review" if count >= 2 else "skip?"
            print(f"  {value:30s}  {count:3d}x  -> {action}")
    return total_occurrences


def run_unmapped_analysis(
    input_path: str | Path,
    *,
    top: int = DEFAULT_TOP,
    examples: int = DEFAULT_EXAMPLES,
    dimension: str | None = None,
    stats_only: bool = False,
) -> dict[str, int | str]:
    """Inspect unmapped tags from sample outputs or fallback labeling stats."""
    path = Path(input_path).resolve()
    if not path.exists():
        raise FileNotFoundError(f"Input path not found: {path}")
    if top < 1:
        raise ValueError("--top must be >= 1")
    if examples < 0:
        raise ValueError("--examples must be >= 0")

    if not stats_only:
        loaded = _load_samples(path)
        if loaded is not None:
            samples, source_paths, source_kind = loaded
            unmapped, total_samples, total_labeled = analyze_samples(samples)
            total_occurrences = print_sample_report(
                unmapped,
                total_samples=total_samples,
                total_labeled=total_labeled,
                paths=source_paths,
                top=top,
                examples=examples,
                dimension=dimension,
            )
            return {
                "source_kind": source_kind,
                "total_samples": total_samples,
                "total_labeled": total_labeled,
                "total_occurrences": total_occurrences,
                "unique_tags": sum(len(dim) for dim in unmapped.values()),
            }

    stats_path = _resolve_stats_path(path)
    if stats_path is None:
        raise ValueError(
            "Could not find labeled/scored samples or labeling stats under the input path. "
            "Pass a labeled/scored file, inline run root/file, run directory, or stats_labeling.json."
        )

    with open(stats_path, "r", encoding="utf-8") as handle:
        stats_payload = json.load(handle)
    unmapped = _stats_to_unmapped(stats_payload)
    total_occurrences = print_stats_report(
        unmapped,
        stats_path=stats_path,
        top=top,
        dimension=dimension,
    )
    return {
        "source_kind": "stats",
        "total_samples": int(stats_payload.get("total_samples", 0)),
        "total_labeled": int(stats_payload.get("distribution_total_samples", 0)),
        "total_occurrences": total_occurrences,
        "unique_tags": sum(len(dim) for dim in unmapped.values()),
    }


def build_parser() -> argparse.ArgumentParser:
    """Build a standalone parser for direct module execution."""
    parser = argparse.ArgumentParser(
        prog="sft-label analyze-unmapped",
        description="Inspect unmapped tags from labeled/scored outputs or stats_labeling.json.",
    )
    parser.add_argument(
        "--input",
        required=True,
        help="Labeled/scored file, inline run root/file, run directory, or stats_labeling.json",
    )
    parser.add_argument(
        "--dimension",
        default=None,
        help="Only show one dimension (e.g. task, concept, unknown)",
    )
    parser.add_argument(
        "--top",
        type=int,
        default=DEFAULT_TOP,
        help=f"Max unmapped tags to show per dimension (default: {DEFAULT_TOP})",
    )
    parser.add_argument(
        "--examples",
        type=int,
        default=DEFAULT_EXAMPLES,
        help=f"Max sample examples to show per unmapped tag (default: {DEFAULT_EXAMPLES})",
    )
    parser.add_argument(
        "--stats-only",
        action="store_true",
        help="Only read unmapped summary from stats_labeling.json/stats.json",
    )
    return parser


def main(argv: list[str] | None = None):
    parser = build_parser()
    args = parser.parse_args(argv)
    run_unmapped_analysis(
        args.input,
        top=args.top,
        examples=args.examples,
        dimension=args.dimension,
        stats_only=args.stats_only,
    )


if __name__ == "__main__":
    main()
