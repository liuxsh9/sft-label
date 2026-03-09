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

from sft_label.conversation import build_conversation_key

VALID_MISSING_GATE_POLICIES = {"fail", "ignore"}


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
    # Turn count criteria (multi-turn only)
    turn_count_min: int | None = None       # min total_turns in conversation
    turn_count_max: int | None = None       # max total_turns in conversation
    # Sub-score hard floors
    correctness_min: float | None = None    # min quality.correctness (hard floor)
    missing_gate_policy: str = "fail"       # "fail" or "ignore" when gate fields are missing
    # Turn-level pruning (within conversations that pass conv-level criteria)
    turn_value_min: float | None = None     # min per-slice value_score
    turn_quality_min: float | None = None   # min per-slice quality.overall
    max_pruned_ratio: float = 0.5           # don't prune more than 50% of turns
    keep_first_last: bool = True            # always keep first and last turns
    preserve_structure: bool = False        # directory mode: mirror input tree


def _validate_missing_gate_policy(policy: str):
    """Validate policy used when required hard-gate fields are missing."""
    if policy not in VALID_MISSING_GATE_POLICIES:
        allowed = ", ".join(sorted(VALID_MISSING_GATE_POLICIES))
        raise ValueError(f"Invalid missing_gate_policy '{policy}'. Expected one of: {allowed}")


def _init_missing_gate_drop_counts():
    """Initialize per-criterion drop counters for missing gate fields."""
    return {
        "correctness_min": 0,
        "thinking_mode": 0,
        "turn_value_min": 0,
        "turn_quality_min": 0,
    }


def _record_missing_drop(missing_gate_drops, criterion):
    """Increment a missing-field drop counter."""
    if missing_gate_drops is not None and criterion in missing_gate_drops:
        missing_gate_drops[criterion] += 1


def _record_missing_drop_reasons(missing_gate_drops, reasons):
    """Increment counters for all missing reasons attached to one dropped sample."""
    for criterion in reasons:
        _record_missing_drop(missing_gate_drops, criterion)


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


def matches_filter(sample, config, missing_gate_drops=None):
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

    # Check correctness_min (hard floor on code correctness)
    if config.correctness_min is not None:
        quality = value.get("quality") or value.get("scores", {}).get("quality") or {}
        correctness = quality.get("correctness") if isinstance(quality, dict) else None
        if correctness is None:
            if config.missing_gate_policy == "fail":
                _record_missing_drop(missing_gate_drops, "correctness_min")
                return False
        elif correctness < config.correctness_min:
            return False

    # Check thinking_mode
    if config.thinking_mode:
        sample_mode = value.get("thinking_mode") or metadata.get("thinking_mode", "")
        if sample_mode:
            if sample_mode != config.thinking_mode:
                return False
        elif config.missing_gate_policy == "fail":
            _record_missing_drop(missing_gate_drops, "thinking_mode")
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
        paths_to_try.append(input_path.parent / "conversation_scores.json")
    elif input_path.is_dir():
        paths_to_try.extend(sorted(input_path.rglob("conversation_scores.json")))

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
    if config.turn_count_min is not None:
        tc = conv_record.get("turn_count")
        if tc is None or tc < config.turn_count_min:
            return False
    if config.turn_count_max is not None:
        tc = conv_record.get("turn_count")
        if tc is None or tc > config.turn_count_max:
            return False
    return True


def _has_conv_criteria(config):
    """Check if any conversation-level criteria are set."""
    return any([
        config.conv_value_min is not None,
        config.conv_selection_min is not None,
        config.peak_complexity_min is not None,
        config.turn_count_min is not None,
        config.turn_count_max is not None,
    ])


def _is_multi_turn(sample):
    """Check if a sample is from a multi-turn conversation."""
    meta = sample.get("metadata") or {}
    return bool(meta.get("source_id")) and meta.get("total_turns", 1) > 1


def _init_conv_lookup_stats():
    """Initialize diagnostics for conversation record lookup."""
    return {
        "canonical_hits": 0,
        "legacy_fallback_hits": 0,
        "missing_identity": 0,
        "missing_record": 0,
    }


def _resolve_conversation_record(sample, conv_lookup, conv_stats):
    """Resolve conversation score record using canonical key + legacy fallback."""
    meta = sample.get("metadata") or {}
    source_id = meta.get("source_id")
    canonical_id = build_conversation_key(meta)

    if not canonical_id:
        if conv_stats is not None:
            conv_stats["missing_identity"] += 1
        return None

    conv_rec = conv_lookup.get(canonical_id)
    if conv_rec is not None:
        if conv_stats is not None:
            conv_stats["canonical_hits"] += 1
        return conv_rec

    if canonical_id != source_id and source_id:
        legacy_rec = conv_lookup.get(source_id)
        if legacy_rec is not None:
            if conv_stats is not None:
                conv_stats["legacy_fallback_hits"] += 1
            return legacy_rec

    if conv_stats is not None:
        conv_stats["missing_record"] += 1
    return None


def _has_turn_criteria(config):
    """Check if any turn-level pruning criteria are set."""
    return config.turn_value_min is not None or config.turn_quality_min is not None


def _passes_turn_criteria(sample, config):
    """Check if a single slice passes turn-level thresholds.

    Returns tuple:
      (passes: bool, missing_reasons: list[str])
    where missing_reasons includes criteria that failed due to missing fields
    under strict missing_gate_policy=fail.
    """
    value = sample.get("value") or {}
    missing_reasons = []
    failed = False
    if config.turn_value_min is not None:
        vs = value.get("value_score")
        if vs is None:
            if config.missing_gate_policy == "fail":
                missing_reasons.append("turn_value_min")
                failed = True
        elif vs < config.turn_value_min:
            failed = True
    if config.turn_quality_min is not None:
        quality = value.get("quality") or value.get("scores", {}).get("quality") or {}
        qo = quality.get("overall") if isinstance(quality, dict) else None
        if qo is None:
            if config.missing_gate_policy == "fail":
                missing_reasons.append("turn_quality_min")
                failed = True
        elif qo < config.turn_quality_min:
            failed = True
    return not failed, missing_reasons


def filter_samples(samples, threshold=None, include_unscored=False, config=None):
    """Filter samples by criteria. Backward-compatible API.

    Returns (retained, dropped) tuple of lists.
    """
    if config is None:
        config = FilterConfig(value_min=threshold, include_unscored=include_unscored)
    _validate_missing_gate_policy(config.missing_gate_policy)

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
        files = sorted(set(input_path.rglob("scored*.json")) | set(input_path.rglob("scored*.jsonl")))
        # Deduplicate: when both .json and .jsonl exist for the same stem
        # in the same directory, prefer .jsonl for streaming scalability.
        seen_stems = {}
        for f in files:
            key = (f.parent, f.stem)
            if key in seen_stems:
                # Keep .jsonl over .json
                existing = seen_stems[key]
                if existing.suffix == ".json" and f.suffix == ".jsonl":
                    seen_stems[key] = f
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


def _iter_json_array_streaming(path: Path, chunk_size: int = 1024 * 1024):
    """Yield items from a JSON array without loading the full file into memory."""
    decoder = json.JSONDecoder()
    buffer = ""
    started = False
    expect_value = False
    done = False

    with open(path, encoding="utf-8") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            buffer += chunk
            idx = 0

            while True:
                # Skip whitespace
                while idx < len(buffer) and buffer[idx].isspace():
                    idx += 1

                if not started:
                    if idx >= len(buffer):
                        break
                    if buffer[idx] != "[":
                        raise ValueError(f"Expected JSON array in {path}")
                    started = True
                    expect_value = True
                    idx += 1
                    continue

                if expect_value:
                    # Could be empty array or end after comma-separated values
                    if idx >= len(buffer):
                        break
                    if buffer[idx] == "]":
                        done = True
                        idx += 1
                        break

                    try:
                        item, next_idx = decoder.raw_decode(buffer, idx)
                    except json.JSONDecodeError:
                        # Need more bytes for current item
                        break
                    yield item
                    idx = next_idx
                    expect_value = False
                    continue

                # Expect comma or closing bracket
                if idx >= len(buffer):
                    break
                if buffer[idx] == ",":
                    idx += 1
                    expect_value = True
                    continue
                if buffer[idx] == "]":
                    done = True
                    idx += 1
                    break
                if buffer[idx].isspace():
                    idx += 1
                    continue
                raise ValueError(f"Invalid JSON array syntax in {path}")

            # Keep unconsumed suffix for next chunk
            buffer = buffer[idx:]
            if done:
                # Ensure trailing bytes are just whitespace
                if buffer.strip():
                    raise ValueError(f"Unexpected trailing content in {path}")
                break

    if not done:
        # Handle tiny files where closing bracket is in remaining buffer
        idx = 0
        while idx < len(buffer) and buffer[idx].isspace():
            idx += 1
        if not started:
            if idx >= len(buffer):
                raise ValueError(f"Empty file: {path}")
            if buffer[idx] != "[":
                raise ValueError(f"Expected JSON array in {path}")
            started = True
            expect_value = True
            idx += 1
        while True:
            while idx < len(buffer) and buffer[idx].isspace():
                idx += 1
            if expect_value:
                if idx < len(buffer) and buffer[idx] == "]":
                    done = True
                    idx += 1
                    break
                if idx >= len(buffer):
                    break
                try:
                    item, next_idx = decoder.raw_decode(buffer, idx)
                except json.JSONDecodeError as e:
                    raise ValueError(f"Malformed JSON array in {path}") from e
                yield item
                idx = next_idx
                expect_value = False
                continue
            if idx >= len(buffer):
                break
            if buffer[idx] == ",":
                idx += 1
                expect_value = True
                continue
            if buffer[idx] == "]":
                done = True
                idx += 1
                break
            if buffer[idx].isspace():
                idx += 1
                continue
            raise ValueError(f"Invalid JSON array syntax in {path}")

        if not done:
            raise ValueError(f"Unterminated JSON array in {path}")
        if buffer[idx:].strip():
            raise ValueError(f"Unexpected trailing content in {path}")


def _load_samples(path: Path):
    """Load all samples from a JSON or JSONL file."""
    if path.suffix == ".jsonl":
        return list(_iter_samples_streaming(path))

    return list(_iter_json_array_streaming(path))


def _iter_samples_from_files(paths):
    """Iterate samples from multiple files, auto-detecting format."""
    for path in paths:
        if path.suffix == ".jsonl":
            yield from _iter_samples_streaming(path)
        else:
            try:
                yield from _iter_json_array_streaming(path)
            except (ValueError, OSError):
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
    suffix = _build_suffix(config)

    if input_path.is_file():
        parent = input_path.parent
    else:
        parent = input_path

    fmt_ext = ".jsonl" if config.output_format == "training" else ".json"
    return parent / f"filtered-{suffix}{fmt_ext}"


def _build_suffix(config):
    """Build filename suffix from active filter criteria."""
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
    if config.turn_count_min is not None:
        parts.append(f"tc{config.turn_count_min}")
    if config.turn_count_max is not None:
        parts.append(f"tc-max{config.turn_count_max}")
    if config.turn_value_min is not None:
        parts.append(f"tv{config.turn_value_min:g}")
    if config.turn_quality_min is not None:
        parts.append(f"tq{config.turn_quality_min:g}")
    return "-".join(parts) if parts else "all"


def _build_filter_desc(config):
    """Human-readable filter description."""
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
    if config.turn_count_min is not None:
        criteria.append(f"turn_count >= {config.turn_count_min}")
    if config.turn_count_max is not None:
        criteria.append(f"turn_count <= {config.turn_count_max}")
    if config.correctness_min is not None:
        criteria.append(f"correctness >= {config.correctness_min:g}")
    if config.turn_value_min is not None:
        criteria.append(f"turn_value >= {config.turn_value_min:g}")
    if config.turn_quality_min is not None:
        criteria.append(f"turn_quality >= {config.turn_quality_min:g}")
    if (
        config.correctness_min is not None
        or config.thinking_mode
        or config.turn_value_min is not None
        or config.turn_quality_min is not None
    ):
        criteria.append(f"missing_gate_policy = {config.missing_gate_policy}")
    return " AND ".join(criteria) if criteria else "none"


def _passes_with_conv(sample, config, use_conv, conv_lookup, conv_stats, missing_gate_drops=None):
    """Apply conversation-level gate (if enabled) + full sample-level filter."""
    total_turns = (sample.get("metadata") or {}).get("total_turns", 1)
    if use_conv and total_turns > 1:
        conv_rec = _resolve_conversation_record(sample, conv_lookup, conv_stats)
        if conv_rec is None or not _matches_conv_criteria(conv_rec, config):
            return False
    return matches_filter(sample, config, missing_gate_drops=missing_gate_drops)


def _prune_turns(retained, config, missing_gate_drops=None):
    """Turn-level pruning: prune low-value slices within conversations."""
    n_turn_pruned = 0
    if not (_has_turn_criteria(config) and retained):
        return retained, n_turn_pruned

    mt_groups = {}  # source_id -> [(index_in_retained, sample)]
    final_retained = []
    for i, sample in enumerate(retained):
        if _is_multi_turn(sample):
            source_id = (sample.get("metadata") or {}).get("source_id")
            mt_groups.setdefault(source_id, []).append((i, sample))
        else:
            passes, missing_reasons = _passes_turn_criteria(sample, config)
            if passes:
                final_retained.append(sample)
            else:
                n_turn_pruned += 1
                _record_missing_drop_reasons(missing_gate_drops, missing_reasons)

    for members in mt_groups.values():
        n = len(members)
        if n <= 1:
            final_retained.extend(s for _, s in members)
            continue

        members.sort(key=lambda x: (x[1].get("metadata") or {}).get("turn_index", 0))

        max_prune = int(n * config.max_pruned_ratio)
        candidates_to_prune = []
        missing_reasons_by_turn = {}
        for j, (_orig_idx, sample) in enumerate(members):
            is_first = (j == 0)
            is_last = (j == n - 1)
            if config.keep_first_last and (is_first or is_last):
                continue
            passes, missing_reasons = _passes_turn_criteria(sample, config)
            if not passes:
                vs = (sample.get("value") or {}).get("value_score", 0.0) or 0.0
                candidates_to_prune.append((j, vs))
                if missing_reasons:
                    missing_reasons_by_turn[j] = missing_reasons

        if max_prune == 0:
            candidates_to_prune = []
        elif len(candidates_to_prune) > max_prune:
            candidates_to_prune.sort(key=lambda x: x[1], reverse=True)
            candidates_to_prune = candidates_to_prune[-max_prune:]

        prune_set = {j for j, _ in candidates_to_prune}
        n_turn_pruned += len(prune_set)
        for j in prune_set:
            _record_missing_drop_reasons(missing_gate_drops, missing_reasons_by_turn.get(j, []))

        for j, (_orig_idx, sample) in enumerate(members):
            if j not in prune_set:
                final_retained.append(sample)

    return final_retained, n_turn_pruned


def _init_stream_stats():
    """Initialize mutable stats used by streaming filter paths."""
    return {
        "total": 0,
        "retained": 0,
        "value_sum": 0.0,
        "value_count": 0,
        "inherited_retained": 0,
        "verify_matched": 0,
        "verify_no_meta": 0,
    }


def _accumulate_retained_stats(sample, config, stats):
    """Update retained-sample stats for one sample."""
    value = sample.get("value") or {}
    score = value.get("value_score")
    if score is not None:
        stats["value_sum"] += score
        stats["value_count"] += 1

    if (sample.get("labels") or {}).get("inherited"):
        stats["inherited_retained"] += 1

    if config.verify_source:
        if (sample.get("metadata") or {}).get("source_file"):
            stats["verify_matched"] += 1
        else:
            stats["verify_no_meta"] += 1


def _iter_filtered_samples(
    scored_files, config, use_conv, conv_lookup, conv_stats, stats, missing_gate_drops=None
):
    """Stream filtered samples and update stats in-place."""
    for sample in _iter_samples_from_files(scored_files):
        stats["total"] += 1
        if not _passes_with_conv(
            sample, config, use_conv, conv_lookup, conv_stats, missing_gate_drops=missing_gate_drops
        ):
            continue
        stats["retained"] += 1
        _accumulate_retained_stats(sample, config, stats)
        yield sample


def _collect_filtered_samples(
    scored_files, config, use_conv, conv_lookup, conv_stats, missing_gate_drops=None
):
    """Collect filtered samples (needed for turn-level pruning path)."""
    retained = []
    total = 0
    for sample in _iter_samples_from_files(scored_files):
        total += 1
        if _passes_with_conv(
            sample, config, use_conv, conv_lookup, conv_stats, missing_gate_drops=missing_gate_drops
        ):
            retained.append(sample)
    retained, n_turn_pruned = _prune_turns(retained, config, missing_gate_drops=missing_gate_drops)
    return total, retained, n_turn_pruned


def _write_json_array(samples, json_path):
    """Write samples as JSON array."""
    with open(json_path, "w", encoding="utf-8") as f:
        f.write("[\n")
        first = True
        for s in samples:
            if not first:
                f.write(",\n")
            f.write(json.dumps(s, ensure_ascii=False))
            first = False
        f.write("\n]\n")


def _write_jsonl(samples, jsonl_path, convert_training=False):
    """Write samples as JSONL."""
    with open(jsonl_path, "w", encoding="utf-8") as f:
        for s in samples:
            item = _convert_to_training_format(s) if convert_training else s
            f.write(json.dumps(item, ensure_ascii=False) + "\n")


def _write_outputs(retained_samples, output_path, config):
    """Write output files for default mode."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if config.output_format == "training":
        _write_jsonl(retained_samples, output_path, convert_training=True)
        return None, output_path

    json_path = output_path if output_path.suffix == ".json" else output_path.with_suffix(".json")
    jsonl_path = json_path.with_suffix(".jsonl")
    _write_json_array(retained_samples, json_path)
    _write_jsonl(retained_samples, jsonl_path, convert_training=False)
    return json_path, jsonl_path


def _write_outputs_streaming(filtered_iter, output_path, config):
    """Write output files in streaming mode (no in-memory retained list)."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if config.output_format == "training":
        with open(output_path, "w", encoding="utf-8") as out_jsonl:
            for s in filtered_iter:
                out_jsonl.write(json.dumps(_convert_to_training_format(s), ensure_ascii=False) + "\n")
        return None, output_path

    json_path = output_path if output_path.suffix == ".json" else output_path.with_suffix(".json")
    jsonl_path = json_path.with_suffix(".jsonl")
    with open(json_path, "w", encoding="utf-8") as out_json, \
         open(jsonl_path, "w", encoding="utf-8") as out_jsonl:
        out_json.write("[\n")
        first = True
        for s in filtered_iter:
            if not first:
                out_json.write(",\n")
            payload = json.dumps(s, ensure_ascii=False)
            out_json.write(payload)
            out_jsonl.write(payload + "\n")
            first = False
        out_json.write("\n]\n")
    return json_path, jsonl_path


def _mirror_output_path(root_output, input_root, scored_file, config):
    """Compute mirrored output path for one scored file."""
    rel = scored_file.relative_to(input_root)
    out = root_output / rel
    if config.output_format == "training":
        out = out.with_suffix(".jsonl")
    return out


def _run_filter_preserve_structure(
    input_path,
    scored_files,
    output_path,
    config,
    use_conv,
    conv_lookup,
    conv_stats,
    missing_gate_drops,
):
    """Directory mode: mirror input folder structure and file count."""
    suffix = _build_suffix(config)
    if output_path is None:
        output_root = input_path / f"filtered-{suffix}"
    else:
        output_root = Path(output_path)
    output_root.mkdir(parents=True, exist_ok=True)

    total = 0
    retained = 0
    value_sum = 0.0
    value_count = 0
    inherited_retained = 0
    turn_pruned = 0
    files_written = 0
    verify_matched = 0
    verify_no_meta = 0
    output_files = []

    for scored_file in scored_files:
        out_file = _mirror_output_path(output_root, input_path, scored_file, config)
        out_file.parent.mkdir(parents=True, exist_ok=True)

        if _has_turn_criteria(config):
            file_total, file_retained, file_turn_pruned = _collect_filtered_samples(
                [scored_file], config, use_conv, conv_lookup, conv_stats,
                missing_gate_drops=missing_gate_drops,
            )
            turn_pruned += file_turn_pruned
            total += file_total
            file_stats = _init_stream_stats()
            retained += len(file_retained)
            for s in file_retained:
                file_stats["retained"] += 1
                _accumulate_retained_stats(s, config, file_stats)
            value_sum += file_stats["value_sum"]
            value_count += file_stats["value_count"]
            inherited_retained += file_stats["inherited_retained"]
            verify_matched += file_stats["verify_matched"]
            verify_no_meta += file_stats["verify_no_meta"]

            if config.output_format == "training" or out_file.suffix == ".jsonl":
                _write_jsonl(file_retained, out_file, convert_training=(config.output_format == "training"))
            else:
                _write_json_array(file_retained, out_file)
        else:
            file_stats = _init_stream_stats()
            stream = _iter_filtered_samples(
                [scored_file], config, use_conv, conv_lookup, conv_stats, file_stats,
                missing_gate_drops=missing_gate_drops,
            )
            if config.output_format == "training" or out_file.suffix == ".jsonl":
                _write_jsonl(stream, out_file, convert_training=(config.output_format == "training"))
            else:
                _write_json_array(stream, out_file)
            total += file_stats["total"]
            retained += file_stats["retained"]
            value_sum += file_stats["value_sum"]
            value_count += file_stats["value_count"]
            inherited_retained += file_stats["inherited_retained"]
            verify_matched += file_stats["verify_matched"]
            verify_no_meta += file_stats["verify_no_meta"]

        files_written += 1
        output_files.append(str(out_file))

    summary = {
        "total": total,
        "retained": retained,
        "dropped": total - retained,
        "retention_rate": retained / total if total > 0 else 0.0,
        "mean_value_retained": (value_sum / value_count) if value_count else 0.0,
        "inherited_retained": inherited_retained,
        "turn_pruned": turn_pruned,
        "threshold": config.value_min,
        "filter_criteria": _build_filter_desc(config),
        "output_format": config.output_format,
        "output_json": None,
        "output_jsonl": None,
        "output_root": str(output_root),
        "output_files": output_files,
        "files_written": files_written,
        "missing_gate_drops": dict(missing_gate_drops),
    }
    if config.verify_source:
        summary["verify_source"] = {"matched": verify_matched, "no_metadata": verify_no_meta}
    if use_conv and conv_stats is not None:
        summary["conversation_lookup"] = dict(conv_stats)
    return summary


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
    _validate_missing_gate_policy(config.missing_gate_policy)
    missing_gate_drops = _init_missing_gate_drop_counts()

    scored_files = _find_scored_files(input_path)

    if not scored_files:
        raise FileNotFoundError(f"No scored files found in {input_path}")

    # Load conversation scores if conv criteria are set
    use_conv = _has_conv_criteria(config)
    conv_lookup = {}
    conv_stats = _init_conv_lookup_stats() if use_conv else None
    if use_conv:
        conv_lookup = _load_conversation_scores(input_path)

    if config.preserve_structure and input_path.is_dir():
        summary = _run_filter_preserve_structure(
            input_path=input_path,
            scored_files=scored_files,
            output_path=output_path,
            config=config,
            use_conv=use_conv,
            conv_lookup=conv_lookup,
            conv_stats=conv_stats,
            missing_gate_drops=missing_gate_drops,
        )
        _print_summary(summary)
        return summary

    # Compute output path
    if output_path is None:
        output_path = _generate_output_name(config, input_path)
    output_path = Path(output_path)

    if _has_turn_criteria(config):
        total, retained, n_turn_pruned = _collect_filtered_samples(
            scored_files, config, use_conv, conv_lookup, conv_stats,
            missing_gate_drops=missing_gate_drops,
        )
        n_retained = len(retained)
        retained_stats = _init_stream_stats()
        for s in retained:
            retained_stats["retained"] += 1
            _accumulate_retained_stats(s, config, retained_stats)
        mean_value = (
            retained_stats["value_sum"] / retained_stats["value_count"]
            if retained_stats["value_count"] else 0.0
        )
        inherited_count = retained_stats["inherited_retained"]
        verify_matched = retained_stats["verify_matched"]
        verify_no_meta = retained_stats["verify_no_meta"]
        json_path, jsonl_path = _write_outputs(retained, output_path, config)
    else:
        stream_stats = _init_stream_stats()
        stream = _iter_filtered_samples(
            scored_files, config, use_conv, conv_lookup, conv_stats, stream_stats,
            missing_gate_drops=missing_gate_drops,
        )
        json_path, jsonl_path = _write_outputs_streaming(stream, output_path, config)
        total = stream_stats["total"]
        n_retained = stream_stats["retained"]
        n_turn_pruned = 0
        mean_value = (
            stream_stats["value_sum"] / stream_stats["value_count"]
            if stream_stats["value_count"] else 0.0
        )
        inherited_count = stream_stats["inherited_retained"]
        verify_matched = stream_stats["verify_matched"]
        verify_no_meta = stream_stats["verify_no_meta"]

    summary = {
        "total": total,
        "retained": n_retained,
        "dropped": total - n_retained,
        "retention_rate": n_retained / total if total > 0 else 0.0,
        "mean_value_retained": mean_value,
        "inherited_retained": inherited_count,
        "turn_pruned": n_turn_pruned,
        "threshold": config.value_min,
        "filter_criteria": _build_filter_desc(config),
        "output_format": config.output_format,
        "output_json": str(json_path) if json_path else None,
        "output_jsonl": str(jsonl_path) if jsonl_path else None,
        "missing_gate_drops": dict(missing_gate_drops),
    }
    if config.verify_source:
        summary["verify_source"] = {"matched": verify_matched, "no_metadata": verify_no_meta}
    if use_conv and conv_stats is not None:
        summary["conversation_lookup"] = dict(conv_stats)

    _print_summary(summary)
    return summary


def _print_summary(summary):
    """Print summary in a stable format."""
    print(f"\n  Filter Summary ({summary['filter_criteria']})")
    print(f"  {'Total samples:':<24} {summary['total']}")
    print(f"  {'Retained:':<24} {summary['retained']}")
    print(f"  {'Dropped:':<24} {summary['dropped']}")
    total = summary["total"]
    if total:
        print(f"  {'Retention rate:':<24} {summary['retention_rate'] * 100:.1f}%")
    print(f"  {'Mean value (retained):':<24} {summary['mean_value_retained']:.2f}")
    if summary.get("inherited_retained"):
        print(f"  {'Inherited (retained):':<24} {summary['inherited_retained']}")
    if summary.get("turn_pruned"):
        print(f"  {'Turn-pruned:':<24} {summary['turn_pruned']}")
    if summary.get("verify_source"):
        verify = summary["verify_source"]
        print(f"  {'Source verified:':<24} {verify['matched']} matched, {verify['no_metadata']} no metadata")
    if summary.get("missing_gate_drops"):
        missing = summary["missing_gate_drops"]
        missing_total = sum(missing.values())
        if missing_total:
            print(
                f"  {'Missing gate drops:':<24} "
                f"correctness={missing['correctness_min']}, "
                f"thinking_mode={missing['thinking_mode']}, "
                f"turn_value={missing['turn_value_min']}, "
                f"turn_quality={missing['turn_quality_min']}"
            )
    if summary.get("conversation_lookup"):
        conv = summary["conversation_lookup"]
        print(
            f"  {'Conv lookup:':<24} "
            f"canonical={conv['canonical_hits']}, "
            f"legacy_fallback={conv['legacy_fallback_hits']}, "
            f"missing_id={conv['missing_identity']}, "
            f"missing_record={conv['missing_record']}"
        )
    print(f"  {'Format:':<24} {summary['output_format']}")
    if summary.get("output_root"):
        print(f"  {'Output root:':<24} {summary['output_root']}")
        print(f"  {'Files written:':<24} {summary.get('files_written', 0)}")
    else:
        if summary.get("output_json"):
            print(f"  Output: {summary['output_json']}")
        if summary.get("output_jsonl"):
            print(f"  Output: {summary['output_jsonl']}")
