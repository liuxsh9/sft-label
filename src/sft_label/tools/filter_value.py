"""
Value-based sample filter.

Filters scored SFT data by value_score threshold, outputting high-value
samples as JSON + JSONL.

Usage via CLI:
  sft-label filter --input scored.json --threshold 6.0
  sft-label filter --input run_dir/ --threshold 7.0 --include-unscored
"""

import json
from pathlib import Path


def _find_scored_files(input_path: Path):
    """Find scored JSON files from a file or directory path."""
    if input_path.is_file():
        return [input_path]

    if input_path.is_dir():
        files = []
        for pattern in ("scored*.json", "*/scored*.json"):
            files.extend(input_path.glob(pattern))
        # Deduplicate and sort
        return sorted(set(files))

    raise FileNotFoundError(f"Input path not found: {input_path}")


def _load_samples(path: Path):
    """Load samples from a JSON or JSONL file."""
    if path.suffix == ".jsonl":
        samples = []
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    samples.append(json.loads(line))
        return samples

    with open(path, encoding="utf-8") as f:
        data = json.load(f)

    if isinstance(data, list):
        return data
    raise ValueError(f"Expected list in {path}, got {type(data).__name__}")


def filter_samples(samples, threshold, include_unscored=False):
    """Filter samples by value_score threshold.

    Returns (retained, dropped) tuple of lists.
    """
    retained = []
    dropped = []
    for s in samples:
        value = s.get("value")
        if value is None or value.get("value_score") is None:
            # Unscored sample
            if include_unscored:
                retained.append(s)
            else:
                dropped.append(s)
            continue

        if value["value_score"] >= threshold:
            retained.append(s)
        else:
            dropped.append(s)

    return retained, dropped


def run_filter(input_path, threshold, output_path=None, include_unscored=False):
    """Run the filter pipeline.

    Args:
        input_path: Path to scored JSON/JSONL file or directory.
        threshold: Minimum value_score to retain.
        output_path: Output file path. Auto-generated if None.
        include_unscored: Keep samples without value scores.

    Returns:
        dict with summary stats.
    """
    input_path = Path(input_path)
    scored_files = _find_scored_files(input_path)

    if not scored_files:
        raise FileNotFoundError(f"No scored files found in {input_path}")

    # Load all samples
    all_samples = []
    for f in scored_files:
        all_samples.extend(_load_samples(f))

    # Filter
    retained, dropped = filter_samples(all_samples, threshold, include_unscored)

    # Compute output path
    if output_path is None:
        if input_path.is_file():
            stem = input_path.stem
            parent = input_path.parent
        else:
            stem = "scored"
            parent = input_path
        thresh_str = f"{threshold:g}"
        output_path = parent / f"{stem}-filtered-{thresh_str}.json"
    output_path = Path(output_path)

    # Write JSON
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(retained, f, ensure_ascii=False, indent=2)

    # Write JSONL alongside
    jsonl_path = output_path.with_suffix(".jsonl")
    with open(jsonl_path, "w", encoding="utf-8") as f:
        for s in retained:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")

    # Compute summary
    total = len(all_samples)
    n_retained = len(retained)
    n_dropped = len(dropped)
    scored_retained = [s for s in retained if s.get("value", {}).get("value_score") is not None]
    mean_value = (
        sum(s["value"]["value_score"] for s in scored_retained) / len(scored_retained)
        if scored_retained else 0.0
    )

    summary = {
        "total": total,
        "retained": n_retained,
        "dropped": n_dropped,
        "retention_rate": n_retained / total if total > 0 else 0.0,
        "mean_value_retained": mean_value,
        "threshold": threshold,
        "output_json": str(output_path),
        "output_jsonl": str(jsonl_path),
    }

    # Print summary
    print(f"\n  Filter Summary (threshold >= {threshold:g})")
    print(f"  {'Total samples:':<22} {total}")
    print(f"  {'Retained:':<22} {n_retained}")
    print(f"  {'Dropped:':<22} {n_dropped}")
    print(f"  {'Retention rate:':<22} {n_retained / total * 100:.1f}%" if total else "")
    print(f"  {'Mean value (retained):':<22} {mean_value:.2f}")
    print(f"  Output: {output_path}")
    print(f"  Output: {jsonl_path}")

    return summary
