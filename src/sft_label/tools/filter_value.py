"""
Value-based sample filter with multi-condition support.

Filters scored SFT data by multiple criteria (value, selection, tags,
difficulty, thinking mode) and optionally outputs in training format.

Usage via CLI:
  sft-label filter --input scored.json --threshold 6.0
  sft-label filter --input scored.json --value-min 6 --difficulty advanced,expert
  sft-label filter --input run_dir/ --value-min 7 --format training
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class FilterConfig:
    """Multi-condition filter configuration."""
    value_min: float | None = None          # min value_score
    selection_min: float | None = None      # min selection_score
    include_tags: list[str] = field(default_factory=list)   # "dim:tag", OR logic
    exclude_tags: list[str] = field(default_factory=list)   # "dim:tag", OR logic
    difficulty: list[str] = field(default_factory=list)      # e.g. ["advanced", "expert"]
    thinking_mode: str | None = None        # "slow" or "fast"
    exclude_inherited: bool = False
    include_unscored: bool = False
    output_format: str = "scored"           # "scored" or "training"
    verify_source: str | None = None        # expected source file path
    # Conversation-level criteria (multi-turn only)
    conv_value_min: float | None = None     # min conversation-level value
    conv_selection_min: float | None = None  # min conversation-level selection
    peak_complexity_min: float | None = None # min peak complexity across turns


def _parse_dim_tag(dim_tag):
    """Parse 'dim:tag' string into (dimension, tag) tuple."""
    if ":" in dim_tag:
        dim, tag = dim_tag.split(":", 1)
        return dim.strip(), tag.strip()
    return None, dim_tag.strip()


def _sample_has_tag(labels, dim, tag):
    """Check if sample labels contain a specific dim:tag."""
    if not labels:
        return False
    if dim:
        val = labels.get(dim)
        if isinstance(val, list):
            return tag in val
        return val == tag
    # No dimension specified — search all dimensions
    for key, val in labels.items():
        if key in ("confidence", "inherited", "inherited_from"):
            continue
        if isinstance(val, list) and tag in val:
            return True
        if val == tag:
            return True
    return False


def matches_filter(sample, config):
    """Check if a sample matches all filter criteria.

    Returns True if the sample should be retained.
    All criteria use AND logic between different criteria,
    include_tags/exclude_tags use OR logic within their list.
    """
    labels = sample.get("labels") or {}
    value = sample.get("value") or {}
    metadata = sample.get("metadata") or {}

    # Check inherited
    if config.exclude_inherited and labels.get("inherited"):
        return False

    # Check source verification
    if config.verify_source:
        sample_source = metadata.get("source_file", "")
        if sample_source:
            try:
                if Path(sample_source).resolve() != Path(config.verify_source).resolve():
                    return False
            except (OSError, ValueError):
                return False

    # Check scored/unscored
    has_score = value.get("value_score") is not None
    if not has_score and not config.include_unscored:
        # Only skip if we actually have score-based criteria
        if config.value_min is not None or config.selection_min is not None:
            return False

    # Check value_min
    if config.value_min is not None:
        score = value.get("value_score")
        if score is None:
            if not config.include_unscored:
                return False
        elif score < config.value_min:
            return False

    # Check selection_min
    if config.selection_min is not None:
        score = value.get("selection_score")
        if score is None:
            if not config.include_unscored:
                return False
        elif score < config.selection_min:
            return False

    # Check thinking_mode
    if config.thinking_mode:
        sample_mode = value.get("thinking_mode") or metadata.get("thinking_mode", "")
        if sample_mode and sample_mode != config.thinking_mode:
            return False

    # Check difficulty
    if config.difficulty:
        sample_diff = labels.get("difficulty", "")
        if sample_diff and sample_diff not in config.difficulty:
            return False

    # Check include_tags (OR: sample must have at least one)
    if config.include_tags:
        matched_any = False
        for dt in config.include_tags:
            dim, tag = _parse_dim_tag(dt)
            if _sample_has_tag(labels, dim, tag):
                matched_any = True
                break
        if not matched_any:
            return False

    # Check exclude_tags (OR: sample must not have any)
    if config.exclude_tags:
        for dt in config.exclude_tags:
            dim, tag = _parse_dim_tag(dt)
            if _sample_has_tag(labels, dim, tag):
                return False

    return True


# ── Conversation-level filtering helpers ──

def _load_conversation_scores(input_path):
    """Load conversation_scores.json from input path or its subdirectories.

    Returns dict {conversation_id: record}.
    """
    input_path = Path(input_path)
    lookup = {}

    paths_to_try = []
    if input_path.is_file():
        parent = input_path.parent
        paths_to_try.append(parent / "conversation_scores.json")
    elif input_path.is_dir():
        paths_to_try.append(input_path / "conversation_scores.json")
        for sub in sorted(input_path.iterdir()):
            if sub.is_dir():
                p = sub / "conversation_scores.json"
                if p.exists():
                    paths_to_try.append(p)

    for p in paths_to_try:
        if not p.exists():
            continue
        try:
            with open(p, encoding="utf-8") as f:
                records = json.load(f)
            for rec in records:
                cid = rec.get("conversation_id")
                if cid:
                    lookup[cid] = rec
        except (json.JSONDecodeError, OSError):
            continue

    return lookup


def _matches_conv_criteria(conv_record, config):
    """Check if a conversation record meets conversation-level criteria."""
    if config.conv_value_min is not None:
        cv = conv_record.get("conv_value")
        if cv is None or cv < config.conv_value_min:
            return False
    if config.conv_selection_min is not None:
        cs = conv_record.get("conv_selection")
        if cs is None or cs < config.conv_selection_min:
            return False
    if config.peak_complexity_min is not None:
        pc = conv_record.get("peak_complexity")
        if pc is None or pc < config.peak_complexity_min:
            return False
    return True


def _has_conv_criteria(config):
    """Check if any conversation-level criteria are set."""
    return any([
        config.conv_value_min is not None,
        config.conv_selection_min is not None,
        config.peak_complexity_min is not None,
    ])


def _is_multi_turn(sample):
    """Check if a sample is from a multi-turn conversation."""
    meta = sample.get("metadata") or {}
    return meta.get("source_id") and meta.get("total_turns", 1) > 1


def filter_samples(samples, threshold=None, include_unscored=False, config=None):
    """Filter samples by criteria. Backward-compatible API.

    Returns (retained, dropped) tuple of lists.
    """
    if config is None:
        config = FilterConfig(value_min=threshold, include_unscored=include_unscored)

    retained = []
    dropped = []
    for s in samples:
        if matches_filter(s, config):
            retained.append(s)
        else:
            dropped.append(s)

    return retained, dropped


# ── File discovery and loading ──

def _find_scored_files(input_path: Path):
    """Find scored JSON/JSONL files from a file or directory path."""
    if input_path.is_file():
        return [input_path]

    if input_path.is_dir():
        files = []
        for pattern in ("scored*.json", "scored*.jsonl",
                         "*/scored*.json", "*/scored*.jsonl"):
            files.extend(input_path.glob(pattern))
        # Deduplicate: when both .json and .jsonl exist for the same stem
        # in the same directory, prefer .json (avoids double-counting)
        seen_stems = {}
        for f in sorted(set(files)):
            key = (f.parent, f.stem)
            if key in seen_stems:
                # Keep .json over .jsonl
                existing = seen_stems[key]
                if existing.suffix == ".jsonl" and f.suffix == ".json":
                    seen_stems[key] = f
                # else keep existing (.json already there)
            else:
                seen_stems[key] = f
        return sorted(seen_stems.values())

    raise FileNotFoundError(f"Input path not found: {input_path}")


def _iter_samples_streaming(path: Path):
    """Yield samples line-by-line from a JSONL file (memory-efficient)."""
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    yield json.loads(line)
                except json.JSONDecodeError:
                    continue


def _load_samples(path: Path):
    """Load all samples from a JSON or JSONL file."""
    if path.suffix == ".jsonl":
        return list(_iter_samples_streaming(path))

    with open(path, encoding="utf-8") as f:
        data = json.load(f)

    if isinstance(data, list):
        return data
    raise ValueError(f"Expected list in {path}, got {type(data).__name__}")


def _iter_samples_from_files(paths):
    """Iterate samples from multiple files, auto-detecting format."""
    for path in paths:
        if path.suffix == ".jsonl":
            yield from _iter_samples_streaming(path)
        else:
            try:
                with open(path, encoding="utf-8") as f:
                    data = json.load(f)
                if isinstance(data, list):
                    yield from data
            except (json.JSONDecodeError, OSError):
                continue


# ── Training format conversion ──

def _convert_to_training_format(sample):
    """Convert a scored sample to training format.

    For Pangu-original samples, reconstruct pseudo-multi-turn format.
    For ShareGPT, strip labels/scores and keep id + conversations.
    """
    metadata = sample.get("metadata") or {}

    if metadata.get("original_format") == "pangu":
        from sft_label.preprocessing import to_pangu_pseudo_multiturn
        result = to_pangu_pseudo_multiturn(sample)
        result["id"] = sample.get("id", "")
        return result

    # ShareGPT: strip labels and scores
    return {
        "id": sample.get("id", ""),
        "conversations": sample.get("conversations", []),
    }


# ── Output path generation ──

def _generate_output_name(config, input_path):
    """Generate descriptive output filename from filter params.

    Uses 'filtered-' prefix to avoid being re-discovered by _find_scored_files
    when output lands in the same directory as input.
    """
    parts = []
    if config.value_min is not None:
        parts.append(f"v{config.value_min:g}")
    if config.selection_min is not None:
        parts.append(f"s{config.selection_min:g}")
    if config.difficulty:
        parts.append("-".join(config.difficulty))
    if config.thinking_mode:
        parts.append(config.thinking_mode)
    if config.exclude_inherited:
        parts.append("no-inherited")
    if config.conv_value_min is not None:
        parts.append(f"cv{config.conv_value_min:g}")
    if config.conv_selection_min is not None:
        parts.append(f"cs{config.conv_selection_min:g}")
    if config.peak_complexity_min is not None:
        parts.append(f"pc{config.peak_complexity_min:g}")

    suffix = "-".join(parts) if parts else "all"

    if input_path.is_file():
        stem = input_path.stem
        parent = input_path.parent
    else:
        stem = "scored"
        parent = input_path

    fmt_ext = ".jsonl" if config.output_format == "training" else ".json"
    return parent / f"filtered-{suffix}{fmt_ext}"


# ── Main entry point ──

def run_filter(input_path, threshold=None, output_path=None,
               include_unscored=False, config=None):
    """Run the filter pipeline.

    Args:
        input_path: Path to scored JSON/JSONL file or directory.
        threshold: Minimum value_score to retain (legacy, use config).
        output_path: Output file path. Auto-generated if None.
        include_unscored: Keep samples without value scores (legacy).
        config: FilterConfig for multi-condition filtering.

    Returns:
        dict with summary stats.
    """
    input_path = Path(input_path)

    # Build config from legacy params if not provided
    if config is None:
        config = FilterConfig(
            value_min=threshold,
            include_unscored=include_unscored,
        )

    scored_files = _find_scored_files(input_path)

    if not scored_files:
        raise FileNotFoundError(f"No scored files found in {input_path}")

    # Load conversation scores if conv criteria are set
    use_conv = _has_conv_criteria(config)
    conv_lookup = {}
    if use_conv:
        conv_lookup = _load_conversation_scores(input_path)

    # Filter using streaming iteration
    retained = []
    total = 0

    for sample in _iter_samples_from_files(scored_files):
        total += 1
        if use_conv and _is_multi_turn(sample):
            # Multi-turn: check conv criteria first
            source_id = (sample.get("metadata") or {}).get("source_id")
            conv_rec = conv_lookup.get(source_id) if source_id else None
            if conv_rec is None or not _matches_conv_criteria(conv_rec, config):
                continue
            # Still apply shared slice criteria (tags, difficulty, thinking_mode)
            # but skip slice-level value_min/selection_min
            shared_config = FilterConfig(
                include_tags=config.include_tags,
                exclude_tags=config.exclude_tags,
                difficulty=config.difficulty,
                thinking_mode=config.thinking_mode,
                exclude_inherited=config.exclude_inherited,
                include_unscored=True,
                verify_source=config.verify_source,
            )
            if matches_filter(sample, shared_config):
                retained.append(sample)
        else:
            if matches_filter(sample, config):
                retained.append(sample)

    n_retained = len(retained)
    n_dropped = total - n_retained

    # Compute output path
    if output_path is None:
        output_path = _generate_output_name(config, input_path)
    output_path = Path(output_path)

    # Write output
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if config.output_format == "training":
        # Training format: JSONL only
        with open(output_path, "w", encoding="utf-8") as f:
            for s in retained:
                converted = _convert_to_training_format(s)
                f.write(json.dumps(converted, ensure_ascii=False) + "\n")
        jsonl_path = output_path
        json_path = None
    else:
        # Scored format: JSON + JSONL
        json_path = output_path if output_path.suffix == ".json" else output_path.with_suffix(".json")
        jsonl_path = json_path.with_suffix(".jsonl")

        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(retained, f, ensure_ascii=False, indent=2)
        with open(jsonl_path, "w", encoding="utf-8") as f:
            for s in retained:
                f.write(json.dumps(s, ensure_ascii=False) + "\n")

    # Compute summary
    scored_retained = [s for s in retained if (s.get("value") or {}).get("value_score") is not None]
    mean_value = (
        sum(s["value"]["value_score"] for s in scored_retained) / len(scored_retained)
        if scored_retained else 0.0
    )
    inherited_count = sum(1 for s in retained if (s.get("labels") or {}).get("inherited"))

    # Source verification stats
    verify_stats = {}
    if config.verify_source:
        matched = sum(1 for s in retained
                      if (s.get("metadata") or {}).get("source_file"))
        no_meta = sum(1 for s in retained
                      if not (s.get("metadata") or {}).get("source_file"))
        verify_stats = {"matched": matched, "no_metadata": no_meta}

    # Build filter description
    criteria = []
    if config.value_min is not None:
        criteria.append(f"value >= {config.value_min:g}")
    if config.selection_min is not None:
        criteria.append(f"selection >= {config.selection_min:g}")
    if config.difficulty:
        criteria.append(f"difficulty in [{', '.join(config.difficulty)}]")
    if config.thinking_mode:
        criteria.append(f"thinking = {config.thinking_mode}")
    if config.include_tags:
        criteria.append(f"include_tags = {config.include_tags}")
    if config.exclude_tags:
        criteria.append(f"exclude_tags = {config.exclude_tags}")
    if config.exclude_inherited:
        criteria.append("exclude inherited")
    if config.conv_value_min is not None:
        criteria.append(f"conv_value >= {config.conv_value_min:g}")
    if config.conv_selection_min is not None:
        criteria.append(f"conv_selection >= {config.conv_selection_min:g}")
    if config.peak_complexity_min is not None:
        criteria.append(f"peak_complexity >= {config.peak_complexity_min:g}")
    filter_desc = " AND ".join(criteria) if criteria else "none"

    summary = {
        "total": total,
        "retained": n_retained,
        "dropped": n_dropped,
        "retention_rate": n_retained / total if total > 0 else 0.0,
        "mean_value_retained": mean_value,
        "inherited_retained": inherited_count,
        "threshold": config.value_min,
        "filter_criteria": filter_desc,
        "output_format": config.output_format,
        "output_json": str(json_path) if json_path else None,
        "output_jsonl": str(jsonl_path),
    }
    if verify_stats:
        summary["verify_source"] = verify_stats

    # Print summary
    print(f"\n  Filter Summary ({filter_desc})")
    print(f"  {'Total samples:':<24} {total}")
    print(f"  {'Retained:':<24} {n_retained}")
    print(f"  {'Dropped:':<24} {n_dropped}")
    if total:
        print(f"  {'Retention rate:':<24} {n_retained / total * 100:.1f}%")
    print(f"  {'Mean value (retained):':<24} {mean_value:.2f}")
    if inherited_count:
        print(f"  {'Inherited (retained):':<24} {inherited_count}")
    if verify_stats:
        print(f"  {'Source verified:':<24} {verify_stats['matched']} matched, {verify_stats['no_metadata']} no metadata")
    print(f"  {'Format:':<24} {config.output_format}")
    if json_path:
        print(f"  Output: {json_path}")
    print(f"  Output: {jsonl_path}")

    return summary
