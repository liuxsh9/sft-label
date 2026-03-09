"""Interactive launcher for sft-label CLI workflows."""

from __future__ import annotations

import os
import shlex
from dataclasses import dataclass, field
from typing import Callable


DEFAULT_LITELLM_BASE = "http://localhost:4000/v1"


InputFn = Callable[[str], str]
OutputFn = Callable[[str], None]


@dataclass
class LaunchPlan:
    """Executable launch plan built from interactive prompts."""

    argv: list[str]
    env_overrides: dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class Workflow:
    key: str
    label: str
    description: str
    group: str


WORKFLOWS = [
    Workflow(
        key="run-pass1",
        label="Pass 1 labeling",
        description="Run tag labeling only",
        group="Pipeline",
    ),
    Workflow(
        key="run-pass1-pass2",
        label="Pass 1 + Pass 2",
        description="Labeling followed by value scoring",
        group="Pipeline",
    ),
    Workflow(
        key="run-pass1-pass2-semantic",
        label="Pass 1 + Pass 2 + Pass 4",
        description="Labeling, scoring, then semantic clustering",
        group="Pipeline",
    ),
    Workflow(
        key="score",
        label="Pass 2 scoring only",
        description="Score pre-labeled data",
        group="Pipeline",
    ),
    Workflow(
        key="semantic",
        label="Pass 4 semantic clustering",
        description="Run SemHash + ANN clustering",
        group="Pipeline",
    ),
    Workflow(
        key="filter",
        label="Pass 3 filtering",
        description="Filter scored data for training selection",
        group="Data Curation",
    ),
    Workflow(
        key="recompute-stats",
        label="Recompute stats",
        description="Rebuild stats without LLM calls",
        group="Maintenance",
    ),
    Workflow(
        key="regenerate-dashboard",
        label="Regenerate dashboards",
        description="Rebuild HTML dashboards from existing stats",
        group="Maintenance",
    ),
    Workflow(
        key="export-semantic",
        label="Export semantic rows",
        description="Export representative/all semantic windows",
        group="Export",
    ),
    Workflow(
        key="export-review",
        label="Export review CSV/TSV",
        description="Flatten labeled data for manual review",
        group="Export",
    ),
    Workflow(
        key="validate",
        label="Validate taxonomy",
        description="Validate taxonomy files and consistency",
        group="Maintenance",
    ),
]


def format_command(argv: list[str]) -> str:
    """Render argv as a shell-safe command preview."""
    return "sft-label " + " ".join(shlex.quote(part) for part in argv)


def format_env_prefix(env_overrides: dict[str, str]) -> str:
    """Render env overrides for preview output."""
    if not env_overrides:
        return ""
    return " ".join(
        f"{k}={shlex.quote(v)}" for k, v in sorted(env_overrides.items())
    )


def build_launch_plan(
    input_fn: InputFn = input,
    output_fn: OutputFn = print,
) -> LaunchPlan | None:
    """Prompt for workflow/options and return executable launch plan."""
    _say(output_fn, "Interactive task launcher")
    _say(output_fn, "Choose a workflow and fill only required options.")
    _say(output_fn, "")

    wf = _ask_workflow(input_fn, output_fn)
    if wf is None:
        return None

    if wf.key == "run-pass1":
        return _build_run_plan(input_fn, output_fn, chain_score=False, chain_semantic=False)
    if wf.key == "run-pass1-pass2":
        return _build_run_plan(input_fn, output_fn, chain_score=True, chain_semantic=False)
    if wf.key == "run-pass1-pass2-semantic":
        return _build_run_plan(input_fn, output_fn, chain_score=True, chain_semantic=True)
    if wf.key == "score":
        return _build_score_plan(input_fn, output_fn)
    if wf.key == "semantic":
        return _build_semantic_plan(input_fn, output_fn)
    if wf.key == "filter":
        return _build_filter_plan(input_fn, output_fn)
    if wf.key == "recompute-stats":
        return _build_recompute_plan(input_fn, output_fn)
    if wf.key == "regenerate-dashboard":
        return _build_regenerate_dashboard_plan(input_fn, output_fn)
    if wf.key == "export-semantic":
        return _build_export_semantic_plan(input_fn, output_fn)
    if wf.key == "export-review":
        return _build_export_review_plan(input_fn, output_fn)
    if wf.key == "validate":
        return LaunchPlan(argv=["validate"])
    raise ValueError(f"Unknown workflow key: {wf.key}")


def _build_run_plan(
    input_fn: InputFn,
    output_fn: OutputFn,
    *,
    chain_score: bool,
    chain_semantic: bool,
) -> LaunchPlan:
    argv = ["run"]
    env_overrides: dict[str, str] = {}

    mode = _ask_choice(
        input_fn,
        output_fn,
        "Run mode",
        [
            ("new", "Start new run", "Use --input and optional --output"),
            ("resume", "Resume existing run", "Use --resume and continue an interrupted run"),
        ],
        default_index=1,
    )
    if mode == "new":
        argv.extend(["--input", _ask_required_text(input_fn, "Input file/dir path")])
    else:
        argv.extend(["--resume", _ask_required_text(input_fn, "Run directory to resume")])
        if _ask_yes_no(input_fn, "Also set --input path", default=False):
            argv.extend(["--input", _ask_required_text(input_fn, "Input file/dir path")])

    output_path = _ask_optional_text(input_fn, "Output directory (optional)")
    if output_path:
        argv.extend(["--output", output_path])

    limit = _ask_int(input_fn, "Sample limit per file (0 = all)", default=0)
    if limit:
        argv.extend(["--limit", str(limit)])

    if _ask_yes_no(input_fn, "Shuffle samples before processing", default=False):
        argv.append("--shuffle")

    if not _ask_yes_no(input_fn, "Enable arbitration pass", default=True):
        argv.append("--no-arbitration")

    prompt_mode = _ask_choice(
        input_fn,
        output_fn,
        "Prompt mode",
        [
            ("full", "full", "Higher quality, larger payload"),
            ("compact", "compact", "Smaller payload for size-limited endpoints"),
        ],
        default_index=1,
    )
    if prompt_mode != "full":
        argv.extend(["--prompt-mode", prompt_mode])

    model = _ask_optional_text(input_fn, "LLM model override (optional)")
    if model:
        argv.extend(["--model", model])

    if chain_score:
        argv.append("--score")
        tag_stats = _ask_optional_text(input_fn, "Tag stats path for rarity (optional)")
        if tag_stats:
            argv.extend(["--tag-stats", tag_stats])

    if chain_semantic:
        argv.append("--semantic-cluster")

    if _ask_yes_no(input_fn, "Override LITELLM_BASE / LITELLM_KEY for this run", default=False):
        env_overrides = _ask_llm_env_overrides(input_fn)

    argv.extend(_ask_extra_flags(input_fn))
    return LaunchPlan(argv=argv, env_overrides=env_overrides)


def _build_score_plan(input_fn: InputFn, output_fn: OutputFn) -> LaunchPlan:
    argv = ["score", "--input", _ask_required_text(input_fn, "Input labeled file/dir")]
    env_overrides: dict[str, str] = {}

    tag_stats = _ask_optional_text(input_fn, "Tag stats path for rarity (optional)")
    if tag_stats:
        argv.extend(["--tag-stats", tag_stats])

    limit = _ask_int(input_fn, "Sample limit (0 = all)", default=0)
    if limit:
        argv.extend(["--limit", str(limit)])

    if _ask_yes_no(input_fn, "Resume scoring and skip existing samples", default=False):
        argv.append("--resume")

    prompt_mode = _ask_choice(
        input_fn,
        output_fn,
        "Prompt mode",
        [
            ("full", "full", "Higher quality, larger payload"),
            ("compact", "compact", "Smaller payload for size-limited endpoints"),
        ],
        default_index=1,
    )
    if prompt_mode != "full":
        argv.extend(["--prompt-mode", prompt_mode])

    model = _ask_optional_text(input_fn, "LLM model override (optional)")
    if model:
        argv.extend(["--model", model])

    if _ask_yes_no(input_fn, "Override LITELLM_BASE / LITELLM_KEY for this run", default=False):
        env_overrides = _ask_llm_env_overrides(input_fn)

    argv.extend(_ask_extra_flags(input_fn))
    return LaunchPlan(argv=argv, env_overrides=env_overrides)


def _build_semantic_plan(input_fn: InputFn, output_fn: OutputFn) -> LaunchPlan:
    argv = ["semantic-cluster", "--input", _ask_required_text(input_fn, "Input file/dir")]
    env_overrides: dict[str, str] = {}

    output_path = _ask_optional_text(input_fn, "Output directory (optional)")
    if output_path:
        argv.extend(["--output", output_path])

    limit = _ask_int(input_fn, "Sample limit (0 = all)", default=0)
    if limit:
        argv.extend(["--limit", str(limit)])

    if _ask_yes_no(input_fn, "Enable resume compatibility check", default=False):
        argv.append("--resume")

    if not _ask_yes_no(input_fn, "Export representative windows", default=True):
        argv.append("--no-export-representatives")

    provider = "local"
    if _ask_yes_no(input_fn, "Configure semantic advanced parameters", default=False):
        overrides = _ask_semantic_overrides(input_fn, output_fn)
        argv.extend(overrides["argv"])
        provider = overrides["provider"]

    if provider == "api" and _ask_yes_no(
        input_fn,
        "Provider is api. Override LITELLM_BASE / LITELLM_KEY now",
        default=False,
    ):
        env_overrides = _ask_llm_env_overrides(input_fn)

    argv.extend(_ask_extra_flags(input_fn))
    return LaunchPlan(argv=argv, env_overrides=env_overrides)


def _build_filter_plan(input_fn: InputFn, output_fn: OutputFn) -> LaunchPlan:
    argv = ["filter", "--input", _ask_required_text(input_fn, "Input scored file/dir")]
    criteria_count = 0

    output_path = _ask_optional_text(input_fn, "Output file path (optional)")
    if output_path:
        argv.extend(["--output", output_path])

    output_format = _ask_choice(
        input_fn,
        output_fn,
        "Output format",
        [
            ("scored", "scored", "Keep metadata and scoring fields"),
            ("training", "training", "Export training-ready stripped output"),
        ],
        default_index=1,
    )
    if output_format != "scored":
        argv.extend(["--format", output_format])

    include_unscored = _ask_yes_no(input_fn, "Include unscored samples", default=False)
    if include_unscored:
        argv.append("--include-unscored")

    value_min = _ask_optional_float(input_fn, "--value-min (optional)")
    if value_min is not None:
        argv.extend(["--value-min", str(value_min)])
        criteria_count += 1

    selection_min = _ask_optional_float(input_fn, "--selection-min (optional)")
    if selection_min is not None:
        argv.extend(["--selection-min", str(selection_min)])
        criteria_count += 1

    difficulty = _ask_optional_text(input_fn, "Difficulty list (comma-separated, optional)")
    if difficulty:
        argv.extend(["--difficulty", difficulty])
        criteria_count += 1

    thinking_mode = _ask_choice(
        input_fn,
        output_fn,
        "Thinking mode filter",
        [
            ("", "any", "No thinking_mode filter"),
            ("slow", "slow", "Require explicit reasoning traces"),
            ("fast", "fast", "Require concise reasoning mode"),
        ],
        default_index=1,
    )
    if thinking_mode:
        argv.extend(["--thinking-mode", thinking_mode])
        criteria_count += 1

    include_tags = _ask_optional_text(input_fn, "Include tags (dim:tag, comma-separated, optional)")
    if include_tags:
        argv.extend(["--include-tags", include_tags])
        criteria_count += 1

    exclude_tags = _ask_optional_text(input_fn, "Exclude tags (dim:tag, comma-separated, optional)")
    if exclude_tags:
        argv.extend(["--exclude-tags", exclude_tags])
        criteria_count += 1

    if _ask_yes_no(input_fn, "Exclude inherited labels", default=False):
        argv.append("--exclude-inherited")
        criteria_count += 1

    verify_source = _ask_optional_text(input_fn, "Verify source path (optional)")
    if verify_source:
        argv.extend(["--verify-source", verify_source])
        criteria_count += 1

    if _ask_yes_no(input_fn, "Configure conversation-level filters", default=False):
        conv_value_min = _ask_optional_float(input_fn, "--conv-value-min (optional)")
        if conv_value_min is not None:
            argv.extend(["--conv-value-min", str(conv_value_min)])
            criteria_count += 1

        conv_selection_min = _ask_optional_float(input_fn, "--conv-selection-min (optional)")
        if conv_selection_min is not None:
            argv.extend(["--conv-selection-min", str(conv_selection_min)])
            criteria_count += 1

        peak_complexity_min = _ask_optional_float(input_fn, "--peak-complexity-min (optional)")
        if peak_complexity_min is not None:
            argv.extend(["--peak-complexity-min", str(peak_complexity_min)])
            criteria_count += 1

        turn_count_min = _ask_optional_int(input_fn, "--turn-count-min (optional)")
        if turn_count_min is not None:
            argv.extend(["--turn-count-min", str(turn_count_min)])
            criteria_count += 1

        turn_count_max = _ask_optional_int(input_fn, "--turn-count-max (optional)")
        if turn_count_max is not None:
            argv.extend(["--turn-count-max", str(turn_count_max)])
            criteria_count += 1

    if _ask_yes_no(input_fn, "Configure turn-level pruning", default=False):
        turn_value_min = _ask_optional_float(input_fn, "--turn-value-min (optional)")
        if turn_value_min is not None:
            argv.extend(["--turn-value-min", str(turn_value_min)])
            criteria_count += 1

        turn_quality_min = _ask_optional_float(input_fn, "--turn-quality-min (optional)")
        if turn_quality_min is not None:
            argv.extend(["--turn-quality-min", str(turn_quality_min)])
            criteria_count += 1

        correctness_min = _ask_optional_float(input_fn, "--correctness-min (optional)")
        if correctness_min is not None:
            argv.extend(["--correctness-min", str(correctness_min)])
            criteria_count += 1

        max_pruned_ratio = _ask_optional_float(input_fn, "--max-pruned-ratio (optional)")
        if max_pruned_ratio is not None:
            argv.extend(["--max-pruned-ratio", str(max_pruned_ratio)])

        if not _ask_yes_no(input_fn, "Keep first/last turns when pruning", default=True):
            argv.append("--no-keep-first-last")

    if criteria_count == 0 and not include_unscored:
        _say(output_fn, "")
        _say(output_fn, "Filter requires at least one criterion. Setting --value-min now.")
        fallback = _ask_float(input_fn, "Fallback --value-min", default=6.0)
        argv.extend(["--value-min", str(fallback)])

    argv.extend(_ask_extra_flags(input_fn))
    return LaunchPlan(argv=argv)


def _build_recompute_plan(input_fn: InputFn, output_fn: OutputFn) -> LaunchPlan:
    argv = ["recompute-stats", "--input", _ask_required_text(input_fn, "Input file/dir")]
    pass_num = _ask_choice(
        input_fn,
        output_fn,
        "Pass selection",
        [
            ("both", "both", "Recompute pass 1 and pass 2 stats"),
            ("1", "1", "Recompute pass 1 stats"),
            ("2", "2", "Recompute pass 2 stats"),
        ],
        default_index=1,
    )
    if pass_num != "both":
        argv.extend(["--pass", pass_num])

    output_dir = _ask_optional_text(input_fn, "Output directory override (optional)")
    if output_dir:
        argv.extend(["--output", output_dir])

    argv.extend(_ask_extra_flags(input_fn))
    return LaunchPlan(argv=argv)


def _build_regenerate_dashboard_plan(input_fn: InputFn, output_fn: OutputFn) -> LaunchPlan:
    argv = ["regenerate-dashboard", "--input", _ask_required_text(input_fn, "Run directory")]
    pass_num = _ask_choice(
        input_fn,
        output_fn,
        "Pass selection",
        [
            ("both", "both", "Regenerate both dashboards"),
            ("1", "1", "Regenerate pass 1 dashboard"),
            ("2", "2", "Regenerate pass 2 dashboard"),
        ],
        default_index=1,
    )
    if pass_num != "both":
        argv.extend(["--pass", pass_num])

    if _ask_yes_no(input_fn, "Open dashboards in browser", default=False):
        argv.append("--open")

    argv.extend(_ask_extra_flags(input_fn))
    return LaunchPlan(argv=argv)


def _build_export_semantic_plan(input_fn: InputFn, output_fn: OutputFn) -> LaunchPlan:
    del output_fn  # unused
    argv = [
        "export-semantic",
        "--input",
        _ask_required_text(input_fn, "Semantic artifacts directory"),
        "--output",
        _ask_required_text(input_fn, "Output JSONL path"),
    ]
    if _ask_yes_no(input_fn, "Include non-representative windows", default=False):
        argv.append("--include-all")
    argv.extend(_ask_extra_flags(input_fn))
    return LaunchPlan(argv=argv)


def _build_export_review_plan(input_fn: InputFn, output_fn: OutputFn) -> LaunchPlan:
    argv = [
        "export-review",
        "--input",
        _ask_required_text(input_fn, "Input labeled JSON path"),
        "--output",
        _ask_required_text(input_fn, "Output CSV/TSV path"),
    ]

    monitor = _ask_optional_text(input_fn, "Monitor JSONL path (optional)")
    if monitor:
        argv.extend(["--monitor", monitor])

    fmt = _ask_choice(
        input_fn,
        output_fn,
        "Output format override",
        [
            ("", "auto", "Auto detect from --output extension"),
            ("csv", "csv", "Force CSV"),
            ("tsv", "tsv", "Force TSV"),
        ],
        default_index=1,
    )
    if fmt:
        argv.extend(["--format", fmt])

    argv.extend(_ask_extra_flags(input_fn))
    return LaunchPlan(argv=argv)


def _ask_semantic_overrides(input_fn: InputFn, output_fn: OutputFn) -> dict:
    """Collect optional semantic flags."""
    argv: list[str] = []

    long_turn_threshold = _ask_optional_int(input_fn, "--semantic-long-turn-threshold (optional)")
    if long_turn_threshold is not None:
        argv.extend(["--semantic-long-turn-threshold", str(long_turn_threshold)])

    window_size = _ask_optional_int(input_fn, "--semantic-window-size (optional)")
    if window_size is not None:
        argv.extend(["--semantic-window-size", str(window_size)])

    window_stride = _ask_optional_int(input_fn, "--semantic-window-stride (optional)")
    if window_stride is not None:
        argv.extend(["--semantic-window-stride", str(window_stride)])

    prefix_turns = _ask_optional_int(input_fn, "--semantic-pinned-prefix-max-turns (optional)")
    if prefix_turns is not None:
        argv.extend(["--semantic-pinned-prefix-max-turns", str(prefix_turns)])

    provider = _ask_choice(
        input_fn,
        output_fn,
        "Embedding provider",
        [
            ("", "default(local)", "Keep existing default"),
            ("local", "local", "CPU-first local hash embedding"),
            ("api", "api", "OpenAI-compatible /v1/embeddings backend"),
        ],
        default_index=1,
    )
    effective_provider = provider or "local"
    if provider:
        argv.extend(["--semantic-embedding-provider", provider])

    model = _ask_optional_text(input_fn, "--semantic-embedding-model (optional)")
    if model:
        argv.extend(["--semantic-embedding-model", model])

    embedding_dim = _ask_optional_int(input_fn, "--semantic-embedding-dim (optional)")
    if embedding_dim is not None:
        argv.extend(["--semantic-embedding-dim", str(embedding_dim)])

    embedding_batch = _ask_optional_int(input_fn, "--semantic-embedding-batch-size (optional)")
    if embedding_batch is not None:
        argv.extend(["--semantic-embedding-batch-size", str(embedding_batch)])

    embedding_workers = _ask_optional_int(input_fn, "--semantic-embedding-max-workers (optional)")
    if embedding_workers is not None:
        argv.extend(["--semantic-embedding-max-workers", str(embedding_workers)])

    semhash_bits = _ask_choice(
        input_fn,
        output_fn,
        "SemHash bits",
        [
            ("", "default(256)", "Keep existing default"),
            ("64", "64", "Smaller hash, lower memory"),
            ("128", "128", "Balanced hash width"),
            ("256", "256", "Default"),
            ("512", "512", "Wider hash, better separation"),
        ],
        default_index=1,
    )
    if semhash_bits:
        argv.extend(["--semantic-semhash-bits", semhash_bits])

    semhash_seed = _ask_optional_int(input_fn, "--semantic-semhash-seed (optional)")
    if semhash_seed is not None:
        argv.extend(["--semantic-semhash-seed", str(semhash_seed)])

    semhash_bands = _ask_optional_int(input_fn, "--semantic-semhash-bands (optional)")
    if semhash_bands is not None:
        argv.extend(["--semantic-semhash-bands", str(semhash_bands)])

    hamming_radius = _ask_optional_int(input_fn, "--semantic-hamming-radius (optional)")
    if hamming_radius is not None:
        argv.extend(["--semantic-hamming-radius", str(hamming_radius)])

    ann_top_k = _ask_optional_int(input_fn, "--semantic-ann-top-k (optional)")
    if ann_top_k is not None:
        argv.extend(["--semantic-ann-top-k", str(ann_top_k)])

    ann_threshold = _ask_optional_float(input_fn, "--semantic-ann-sim-threshold (optional)")
    if ann_threshold is not None:
        argv.extend(["--semantic-ann-sim-threshold", str(ann_threshold)])

    output_prefix = _ask_optional_text(input_fn, "--semantic-output-prefix (optional)")
    if output_prefix:
        argv.extend(["--semantic-output-prefix", output_prefix])

    return {"argv": argv, "provider": effective_provider}


def _ask_llm_env_overrides(input_fn: InputFn) -> dict[str, str]:
    base_default = os.environ.get("LITELLM_BASE", DEFAULT_LITELLM_BASE)
    key_default = os.environ.get("LITELLM_KEY", "")
    base = _ask_text(input_fn, "LITELLM_BASE", default=base_default)
    if key_default:
        raw = input_fn("LITELLM_KEY [enter=keep current, '-'=clear]: ").strip()
        if raw == "":
            key = key_default
        elif raw == "-":
            key = ""
        else:
            key = raw
    else:
        key = _ask_optional_text(input_fn, "LITELLM_KEY (optional)")
    return {"LITELLM_BASE": base, "LITELLM_KEY": key}


def _ask_workflow(input_fn: InputFn, output_fn: OutputFn) -> Workflow | None:
    group_order = []
    grouped: dict[str, list[Workflow]] = {}
    for wf in WORKFLOWS:
        if wf.group not in grouped:
            grouped[wf.group] = []
            group_order.append(wf.group)
        grouped[wf.group].append(wf)

    index_to_workflow: dict[int, Workflow] = {}
    idx = 1
    for group in group_order:
        _say(output_fn, f"[{group}]")
        for wf in grouped[group]:
            index_to_workflow[idx] = wf
            _say(output_fn, f"  {idx}. {wf.label} - {wf.description}")
            idx += 1
        _say(output_fn, "")
    _say(output_fn, "  0. Cancel")
    _say(output_fn, "")

    max_choice = len(WORKFLOWS)
    while True:
        raw = input_fn(f"Select workflow number [0-{max_choice}, default 1]: ").strip()
        if raw == "":
            raw = "1"
        if not raw.isdigit():
            _say(output_fn, "Enter a number.")
            continue
        num = int(raw)
        if num == 0:
            return None
        if num in index_to_workflow:
            return index_to_workflow[num]
        _say(output_fn, "Selection out of range.")


def _ask_extra_flags(input_fn: InputFn) -> list[str]:
    while True:
        raw = input_fn(
            "Extra raw CLI flags (optional, e.g. --foo 1 --bar x): "
        ).strip()
        if not raw:
            return []
        try:
            return shlex.split(raw)
        except ValueError as e:
            print(f"Invalid shell-style flags: {e}")


def _ask_required_text(input_fn: InputFn, prompt: str) -> str:
    return _ask_text(input_fn, prompt, required=True)


def _ask_optional_text(input_fn: InputFn, prompt: str) -> str:
    return _ask_text(input_fn, prompt, required=False)


def _ask_text(
    input_fn: InputFn,
    prompt: str,
    *,
    default: str | None = None,
    required: bool = False,
) -> str:
    while True:
        suffix = f" [{default}]" if default is not None else ""
        value = input_fn(f"{prompt}{suffix}: ").strip()
        if value:
            return value
        if default is not None:
            return default
        if not required:
            return ""
        print("This field is required.")


def _ask_int(input_fn: InputFn, prompt: str, *, default: int) -> int:
    while True:
        raw = input_fn(f"{prompt} [{default}]: ").strip()
        if raw == "":
            return default
        try:
            return int(raw)
        except ValueError:
            print("Enter an integer.")


def _ask_optional_int(input_fn: InputFn, prompt: str) -> int | None:
    while True:
        raw = input_fn(f"{prompt}: ").strip()
        if raw == "":
            return None
        try:
            return int(raw)
        except ValueError:
            print("Enter an integer or leave blank.")


def _ask_float(input_fn: InputFn, prompt: str, *, default: float) -> float:
    while True:
        raw = input_fn(f"{prompt} [{default}]: ").strip()
        if raw == "":
            return default
        try:
            return float(raw)
        except ValueError:
            print("Enter a number.")


def _ask_optional_float(input_fn: InputFn, prompt: str) -> float | None:
    while True:
        raw = input_fn(f"{prompt}: ").strip()
        if raw == "":
            return None
        try:
            return float(raw)
        except ValueError:
            print("Enter a number or leave blank.")


def _ask_yes_no(input_fn: InputFn, prompt: str, *, default: bool) -> bool:
    default_hint = "Y/n" if default else "y/N"
    while True:
        raw = input_fn(f"{prompt} [{default_hint}]: ").strip().lower()
        if raw == "":
            return default
        if raw in ("y", "yes"):
            return True
        if raw in ("n", "no"):
            return False
        print("Enter y or n.")


def _ask_choice(
    input_fn: InputFn,
    output_fn: OutputFn,
    title: str,
    options: list[tuple[str, str, str]],
    *,
    default_index: int,
) -> str:
    _say(output_fn, "")
    _say(output_fn, title)
    for idx, (_, label, desc) in enumerate(options, start=1):
        _say(output_fn, f"  {idx}. {label} - {desc}")
    _say(output_fn, "")

    while True:
        raw = input_fn(
            f"Select [1-{len(options)}, default {default_index}]: "
        ).strip()
        if raw == "":
            raw = str(default_index)
        if not raw.isdigit():
            _say(output_fn, "Enter a number.")
            continue
        idx = int(raw)
        if 1 <= idx <= len(options):
            return options[idx - 1][0]
        _say(output_fn, "Selection out of range.")


def _say(output_fn: OutputFn, text: str):
    output_fn(text)
