"""
CLI entry point for sft-label.

Subcommands:
  sft-label run                  — Run Pass 1 tag labeling (optionally + Pass 2)
  sft-label score                — Run Pass 2 value scoring on pre-labeled data
  sft-label semantic-cluster     — Run trajectory SemHash + ANN clustering
  sft-label export-semantic      — Export representative windows from clustering artifacts
  sft-label filter               — Filter scored data by value threshold
  sft-label validate             — Validate taxonomy definitions
  sft-label export-review        — Export labeled data to review CSV
  sft-label recompute-stats      — Recompute stats from labeled/scored output (no LLM)
  sft-label regenerate-dashboard — Regenerate HTML dashboards from stats/data

Model and concurrency are configured in config.py (PipelineConfig),
not via CLI flags. Override via library API if needed.
"""

import argparse
import asyncio
import sys


def cmd_run(args):
    """Run the labeling pipeline."""
    if not args.input and not args.resume:
        print("Error: --input is required (unless using --resume)")
        sys.exit(1)

    from sft_label.config import PipelineConfig
    from sft_label.pipeline import run

    config = PipelineConfig(prompt_mode=args.prompt_mode)
    if args.model:
        config.labeling_model = args.model
        config.scoring_model = args.model
    try:
        stats = asyncio.run(run(
            input_path=args.input,
            output=args.output,
            resume=args.resume,
            limit=args.limit,
            shuffle=args.shuffle,
            enable_arbitration=not args.no_arbitration,
            config=config,
        ))
    except (FileNotFoundError, ValueError) as e:
        print(f"Error: {e}")
        sys.exit(1)

    # Continuous mode: run Pass 2 scoring after Pass 1
    if getattr(args, "score", False) and stats:
        run_dir = stats.get("run_dir")
        if run_dir:
            print("\n── Pass 2: Value Scoring ──\n")
            from sft_label.scoring import run_scoring
            asyncio.run(run_scoring(
                input_path=run_dir,
                output_dir=run_dir,
                tag_stats_path=getattr(args, "tag_stats", None),
                config=config,
            ))

    # Optional semantic clustering pass (trajectory-level)
    if getattr(args, "semantic_cluster", False) and stats:
        run_dir = stats.get("run_dir")
        if run_dir:
            print("\n── Trajectory Semantic Clustering ──\n")
            from sft_label.semantic_clustering import (
                run_semantic_clustering,
                format_semantic_summary,
            )
            sc_stats = run_semantic_clustering(
                input_path=run_dir,
                output_dir=run_dir,
                config=config,
                export_representatives=True,
            )
            print(format_semantic_summary(sc_stats))


def cmd_validate(args):
    """Validate taxonomy definitions."""
    from sft_label.validate import run_validation

    report = run_validation()
    report.print_report()

    if report.has_errors():
        print("\n❌ Validation failed with errors")
        sys.exit(1)
    else:
        print("\n✅ Validation passed")
        sys.exit(0)


def cmd_export_review(args):
    """Export labeled data to review CSV."""
    from sft_label.tools.export_review import main as export_main
    # Patch sys.argv for the tool's own argparse
    sys.argv = ["sft-label export-review"] + args.tool_args
    export_main()


def cmd_score(args):
    """Run value scoring (Pass 2) on pre-labeled data."""
    from sft_label.config import PipelineConfig
    from sft_label.scoring import run_scoring

    config = PipelineConfig(prompt_mode=args.prompt_mode)
    if args.model:
        config.scoring_model = args.model
    try:
        asyncio.run(run_scoring(
            input_path=args.input,
            tag_stats_path=getattr(args, "tag_stats", None),
            limit=args.limit,
            config=config,
            resume=getattr(args, "resume", False),
        ))
    except (FileNotFoundError, ValueError) as e:
        print(f"Error: {e}")
        sys.exit(1)


def cmd_filter(args):
    """Filter scored data by value threshold and other criteria."""
    from sft_label.tools.filter_value import run_filter, FilterConfig

    # Build FilterConfig from CLI args
    config = FilterConfig(
        value_min=args.value_min if args.value_min is not None else args.threshold,
        selection_min=args.selection_min,
        include_tags=[t.strip() for t in args.include_tags.split(",") if t.strip()] if args.include_tags else [],
        exclude_tags=[t.strip() for t in args.exclude_tags.split(",") if t.strip()] if args.exclude_tags else [],
        difficulty=[d.strip() for d in args.difficulty.split(",") if d.strip()] if args.difficulty else [],
        thinking_mode=args.thinking_mode,
        exclude_inherited=args.exclude_inherited,
        include_unscored=args.include_unscored,
        output_format=args.output_format,
        verify_source=args.verify_source,
        conv_value_min=args.conv_value_min,
        conv_selection_min=args.conv_selection_min,
        peak_complexity_min=args.peak_complexity_min,
        turn_count_min=args.turn_count_min,
        turn_count_max=args.turn_count_max,
        correctness_min=args.correctness_min,
        turn_value_min=args.turn_value_min,
        turn_quality_min=args.turn_quality_min,
        max_pruned_ratio=args.max_pruned_ratio,
        keep_first_last=not args.no_keep_first_last,
    )

    # Validate: at least one criterion must be set
    has_criterion = any([
        config.value_min is not None,
        config.selection_min is not None,
        config.include_tags,
        config.exclude_tags,
        config.difficulty,
        config.thinking_mode,
        config.exclude_inherited,
        config.verify_source,
        config.conv_value_min is not None,
        config.conv_selection_min is not None,
        config.peak_complexity_min is not None,
        config.turn_count_min is not None,
        config.turn_count_max is not None,
        config.correctness_min is not None,
        config.turn_value_min is not None,
        config.turn_quality_min is not None,
    ])
    if not has_criterion and not config.include_unscored:
        print("Error: At least one filter criterion is required "
              "(e.g. --value-min, --selection-min, --difficulty, etc.)")
        sys.exit(1)

    try:
        run_filter(
            input_path=args.input,
            output_path=args.output,
            config=config,
        )
    except (FileNotFoundError, ValueError) as e:
        print(f"Error: {e}")
        sys.exit(1)


def _apply_semantic_overrides(config, args):
    """Apply CLI semantic options onto PipelineConfig."""
    for key in (
        "semantic_long_turn_threshold",
        "semantic_window_size",
        "semantic_window_stride",
        "semantic_pinned_prefix_max_turns",
        "semantic_embedding_provider",
        "semantic_embedding_model",
        "semantic_embedding_dim",
        "semantic_embedding_batch_size",
        "semantic_embedding_max_workers",
        "semantic_semhash_bits",
        "semantic_semhash_seed",
        "semantic_semhash_bands",
        "semantic_hamming_radius",
        "semantic_ann_top_k",
        "semantic_ann_sim_threshold",
        "semantic_output_prefix",
    ):
        val = getattr(args, key, None)
        if val is not None:
            setattr(config, key, val)


def cmd_semantic_cluster(args):
    """Run trajectory-level semantic clustering."""
    from sft_label.config import PipelineConfig
    from sft_label.semantic_clustering import (
        run_semantic_clustering,
        format_semantic_summary,
        validate_semantic_config,
    )

    config = PipelineConfig()
    _apply_semantic_overrides(config, args)

    try:
        validate_semantic_config(config)
        stats = run_semantic_clustering(
            input_path=args.input,
            output_dir=args.output,
            limit=args.limit,
            config=config,
            resume=args.resume,
            export_representatives=not args.no_export_representatives,
        )
    except (FileNotFoundError, ValueError) as e:
        print(f"Error: {e}")
        sys.exit(1)

    print(format_semantic_summary(stats))


def cmd_recompute_stats(args):
    """Recompute stats from labeled/scored pipeline output."""
    from sft_label.tools.recompute import run_recompute

    try:
        written = run_recompute(
            input_path=args.input,
            pass_num=args.pass_num,
            output_dir=args.output,
        )
    except (FileNotFoundError, ValueError) as e:
        print(f"Error: {e}")
        sys.exit(1)

    if not written:
        print("No stats files generated. Check that input contains "
              "labeled/scored output.")
        sys.exit(1)
    print(f"\nDone. Wrote {len(written)} file(s).")


def cmd_regenerate_dashboard(args):
    """Regenerate HTML dashboards from existing stats and data."""
    from sft_label.tools.recompute import run_regenerate_dashboard

    try:
        generated = run_regenerate_dashboard(
            input_path=args.input,
            pass_num=args.pass_num,
            open_browser=getattr(args, "open", False),
        )
    except (FileNotFoundError, ValueError) as e:
        print(f"Error: {e}")
        sys.exit(1)

    if not generated:
        print("No dashboards generated. Check that input directory "
              "contains stats files.")
        sys.exit(1)
    print(f"\nDone. Generated {len(generated)} dashboard(s).")


def cmd_export_semantic(args):
    """Export representative windows from semantic clustering artifacts."""
    from sft_label.tools.export_semantic_clusters import run_export_semantic_clusters

    try:
        count = run_export_semantic_clusters(
            input_dir=args.input,
            output_path=args.output,
            include_non_representative=args.include_all,
        )
    except (FileNotFoundError, ValueError) as e:
        print(f"Error: {e}")
        sys.exit(1)
    print(f"Exported {count} semantic window rows to {args.output}")


def main():
    parser = argparse.ArgumentParser(
        prog="sft-label",
        description="SFT Capability Taxonomy & Auto-Labeling Pipeline",
    )
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # --- run ---
    run_parser = subparsers.add_parser(
        "run",
        help="Run Pass 1 tag labeling pipeline",
        description="Run the tag labeling pipeline (Pass 1). "
                    "Optionally chain Pass 2 value scoring with --score.",
        epilog="Model and concurrency are configured in config.py "
               "(DEFAULT_LABELING_MODEL, DEFAULT_CONCURRENCY). "
               "Override via PipelineConfig in library mode.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    run_parser.add_argument("--input", type=str, default=None,
                            help="Input file (.json/.jsonl) or directory")
    run_parser.add_argument("--output", type=str, default=None,
                            help="Output directory (default: sibling of input)")
    run_parser.add_argument("--resume", type=str, default=None,
                            help="Resume from an existing run directory")
    run_parser.add_argument("--limit", type=int, default=0,
                            help="Max samples per file, 0 = all (default: 0)")
    run_parser.add_argument("--shuffle", action="store_true",
                            help="Randomly shuffle samples before processing")
    run_parser.add_argument("--no-arbitration", action="store_true",
                            help="Disable low-confidence arbitration pass")
    run_parser.add_argument("--score", action="store_true",
                            help="Chain Pass 2 value scoring after labeling")
    run_parser.add_argument("--semantic-cluster", action="store_true",
                            help="Chain trajectory semantic clustering after labeling/scoring")
    run_parser.add_argument("--tag-stats", type=str, default=None,
                            help="Path to stats.json for Pass 2 rarity (used with --score)")
    run_parser.add_argument("--prompt-mode", type=str, choices=["full", "compact"],
                            default="full",
                            help="Prompt mode: 'full' (all few-shots) or 'compact' (reduced for small payloads)")
    run_parser.add_argument("--model", type=str, default=None,
                            help="LLM model name (default: gpt-4o-mini)")

    # --- validate ---
    subparsers.add_parser("validate", help="Validate taxonomy definitions")

    # --- score ---
    score_parser = subparsers.add_parser(
        "score",
        help="Run Pass 2 value scoring on pre-labeled data",
        description="Run value scoring (Pass 2) on pre-labeled data. "
                    "Computes complexity, quality, reasoning, and rarity scores.",
        epilog="Model and concurrency are configured in config.py "
               "(DEFAULT_SCORING_MODEL, DEFAULT_SCORING_CONCURRENCY). "
               "Override via PipelineConfig in library mode.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    score_parser.add_argument("--input", type=str, required=True,
                               help="Path to labeled.json or directory of labeled files")
    score_parser.add_argument("--tag-stats", type=str, default=None,
                               help="Path to stats.json for rarity (default: auto-discover)")
    score_parser.add_argument("--limit", type=int, default=0,
                               help="Max samples to score, 0 = all (default: 0)")
    score_parser.add_argument("--resume", action="store_true",
                               help="Resume scoring: skip samples already in scored.jsonl")
    score_parser.add_argument("--prompt-mode", type=str, choices=["full", "compact"],
                               default="full",
                               help="Prompt mode: 'full' (all few-shots) or 'compact' (reduced for small payloads)")
    score_parser.add_argument("--model", type=str, default=None,
                               help="LLM model name (default: gpt-4o-mini)")

    # --- semantic-cluster ---
    semantic_parser = subparsers.add_parser(
        "semantic-cluster",
        help="Run trajectory SemHash + ANN clustering",
        description="Segment long trajectories with pinned prefix, compute semantic embeddings, "
                    "build SemHash + ANN clusters, and select representative windows by SNR.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    semantic_parser.add_argument("--input", type=str, required=True,
                                 help="Input file/dir (prefers scored/labeled outputs)")
    semantic_parser.add_argument("--output", type=str, default=None,
                                 help="Output directory (default: input dir)")
    semantic_parser.add_argument("--limit", type=int, default=0,
                                 help="Max samples to process, 0 = all")
    semantic_parser.add_argument("--resume", action="store_true",
                                 help="Check compatibility against existing manifest before run")
    semantic_parser.add_argument("--no-export-representatives", action="store_true",
                                 help="Do not write representative windows output file")
    semantic_parser.add_argument("--semantic-long-turn-threshold", type=int, default=None,
                                 help="Only trajectories above this turn count use sliding windows")
    semantic_parser.add_argument("--semantic-window-size", type=int, default=None,
                                 help="Window body size in turns")
    semantic_parser.add_argument("--semantic-window-stride", type=int, default=None,
                                 help="Sliding stride in turns")
    semantic_parser.add_argument("--semantic-pinned-prefix-max-turns", type=int, default=None,
                                 help="Max turns kept in pinned task-definition prefix")
    semantic_parser.add_argument("--semantic-embedding-provider", choices=["local", "api"], default=None,
                                 help="Embedding backend: local or api")
    semantic_parser.add_argument("--semantic-embedding-model", type=str, default=None,
                                 help="Embedding model id")
    semantic_parser.add_argument("--semantic-embedding-dim", type=int, default=None,
                                 help="Embedding vector dimension (local backend)")
    semantic_parser.add_argument("--semantic-embedding-batch-size", type=int, default=None,
                                 help="Embedding batch size")
    semantic_parser.add_argument("--semantic-embedding-max-workers", type=int, default=None,
                                 help="Embedding worker count for local backend")
    semantic_parser.add_argument("--semantic-semhash-bits", type=int, default=None,
                                 help="SemHash bit width (64/128/256/512)")
    semantic_parser.add_argument("--semantic-semhash-seed", type=int, default=None,
                                 help="SemHash random seed")
    semantic_parser.add_argument("--semantic-semhash-bands", type=int, default=None,
                                 help="SemHash band count")
    semantic_parser.add_argument("--semantic-hamming-radius", type=int, default=None,
                                 help="Hamming radius for coarse candidate filtering")
    semantic_parser.add_argument("--semantic-ann-top-k", type=int, default=None,
                                 help="Max refined neighbors per window")
    semantic_parser.add_argument("--semantic-ann-sim-threshold", type=float, default=None,
                                 help="Cosine similarity threshold for refined links")
    semantic_parser.add_argument("--semantic-output-prefix", type=str, default=None,
                                 help="Artifact filename prefix")

    # --- export-semantic ---
    export_sem_parser = subparsers.add_parser(
        "export-semantic",
        help="Export semantic clustering rows for review/training",
    )
    export_sem_parser.add_argument("--input", type=str, required=True,
                                   help="Directory containing semantic clustering artifacts")
    export_sem_parser.add_argument("--output", type=str, required=True,
                                   help="Output JSONL path")
    export_sem_parser.add_argument("--include-all", action="store_true",
                                   help="Include non-representative rows (default: representatives only)")

    # --- recompute-stats ---
    recompute_parser = subparsers.add_parser(
        "recompute-stats",
        help="Recompute stats from labeled/scored output (no LLM)",
        description="Rebuild stats.json / stats_value.json from existing pipeline output. "
                    "Useful after manually editing labels or merging runs.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    recompute_parser.add_argument("--input", type=str, required=True,
                                   help="Labeled/scored file or run directory")
    recompute_parser.add_argument("--pass", type=str, default="both",
                                   choices=["1", "2", "both"], dest="pass_num",
                                   help="Which pass stats to recompute (default: both)")
    recompute_parser.add_argument("--output", type=str, default=None,
                                   help="Output directory (default: same as input)")

    # --- regenerate-dashboard ---
    regen_parser = subparsers.add_parser(
        "regenerate-dashboard",
        help="Regenerate HTML dashboards from existing stats/data",
        description="Re-generate HTML dashboards from existing stats and data files. "
                    "Requires stats files to exist (run recompute-stats first if needed).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    regen_parser.add_argument("--input", type=str, required=True,
                               help="Run directory containing stats/data files")
    regen_parser.add_argument("--pass", type=str, default="both",
                               choices=["1", "2", "both"], dest="pass_num",
                               help="Which dashboards to regenerate (default: both)")
    regen_parser.add_argument("--open", action="store_true",
                               help="Open dashboards in browser after generation")

    # --- export-review ---
    review_parser = subparsers.add_parser("export-review",
                                          help="Export labeled data to review CSV")
    review_parser.add_argument("tool_args", nargs=argparse.REMAINDER,
                               help="Arguments passed to export_review tool")

    # --- filter ---
    filter_parser = subparsers.add_parser(
        "filter",
        help="Filter scored data by value and other criteria",
        description="Select high-value samples from scored data. "
                    "Supports multi-condition filtering and training format output.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    filter_parser.add_argument("--input", type=str, required=True,
                                help="Input scored file (.json/.jsonl) or directory")
    filter_parser.add_argument("--value-min", type=float, default=None,
                                help="Minimum value_score to retain")
    filter_parser.add_argument("--threshold", type=float, default=None,
                                help="Alias for --value-min (backward compat)")
    filter_parser.add_argument("--selection-min", type=float, default=None,
                                help="Minimum selection_score to retain")
    filter_parser.add_argument("--include-tags", type=str, default=None,
                                help="Require at least one of these tags (comma-separated, 'dim:tag' format)")
    filter_parser.add_argument("--exclude-tags", type=str, default=None,
                                help="Exclude samples with any of these tags (comma-separated, 'dim:tag' format)")
    filter_parser.add_argument("--difficulty", type=str, default=None,
                                help="Allowed difficulty levels (comma-separated, e.g. 'advanced,expert')")
    filter_parser.add_argument("--thinking-mode", type=str, choices=["slow", "fast"], default=None,
                                help="Filter by thinking mode (slow or fast)")
    filter_parser.add_argument("--exclude-inherited", action="store_true",
                                help="Exclude samples with inherited labels")
    filter_parser.add_argument("--output", type=str, default=None,
                                help="Output file path (default: auto-generated alongside input)")
    filter_parser.add_argument("--include-unscored", action="store_true",
                                help="Keep samples without value scores (default: exclude)")
    filter_parser.add_argument("--format", type=str, choices=["scored", "training"],
                                default="scored", dest="output_format",
                                help="Output format: 'scored' (with labels) or 'training' (training-ready)")
    filter_parser.add_argument("--verify-source", type=str, default=None,
                                help="Only retain samples from this source file path")
    filter_parser.add_argument("--conv-value-min", type=float, default=None,
                                help="Min conversation-level value score (multi-turn)")
    filter_parser.add_argument("--conv-selection-min", type=float, default=None,
                                help="Min conversation-level selection score (multi-turn)")
    filter_parser.add_argument("--peak-complexity-min", type=float, default=None,
                                help="Min peak complexity across turns (multi-turn)")
    filter_parser.add_argument("--turn-count-min", type=int, default=None,
                                help="Min total turns in conversation (multi-turn)")
    filter_parser.add_argument("--turn-count-max", type=int, default=None,
                                help="Max total turns in conversation (multi-turn)")
    filter_parser.add_argument("--correctness-min", type=float, default=None,
                                help="Min quality.correctness score (hard floor)")
    filter_parser.add_argument("--turn-value-min", type=float, default=None,
                                help="Min per-turn value_score (turn-level pruning)")
    filter_parser.add_argument("--turn-quality-min", type=float, default=None,
                                help="Min per-turn quality.overall (turn-level pruning)")
    filter_parser.add_argument("--max-pruned-ratio", type=float, default=0.5,
                                help="Max fraction of turns to prune per conversation (default: 0.5)")
    filter_parser.add_argument("--no-keep-first-last", action="store_true",
                                help="Allow pruning first/last turns in conversations")

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        sys.exit(1)

    if args.command == "run":
        cmd_run(args)
    elif args.command == "validate":
        cmd_validate(args)
    elif args.command == "score":
        cmd_score(args)
    elif args.command == "semantic-cluster":
        cmd_semantic_cluster(args)
    elif args.command == "export-semantic":
        cmd_export_semantic(args)
    elif args.command == "filter":
        cmd_filter(args)
    elif args.command == "export-review":
        cmd_export_review(args)
    elif args.command == "recompute-stats":
        cmd_recompute_stats(args)
    elif args.command == "regenerate-dashboard":
        cmd_regenerate_dashboard(args)


if __name__ == "__main__":
    main()
