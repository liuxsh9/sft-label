"""
CLI entry point for sft-label.

Subcommands:
  sft-label run            — Run Pass 1 tag labeling (optionally + Pass 2)
  sft-label score          — Run Pass 2 value scoring on pre-labeled data
  sft-label filter         — Filter scored data by value threshold
  sft-label validate       — Validate taxonomy definitions
  sft-label export-review  — Export labeled data to review CSV

Model and concurrency are configured in config.py (PipelineConfig),
not via CLI flags. Override via library API if needed.
"""

import argparse
import asyncio
import sys


def cmd_run(args):
    """Run the labeling pipeline."""
    from sft_label.config import PipelineConfig
    from sft_label.pipeline import run

    config = PipelineConfig()
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
            print("\n=== Starting Pass 2: Value Scoring ===\n")
            from sft_label.scoring import run_scoring
            asyncio.run(run_scoring(
                input_path=run_dir,
                output_dir=run_dir,
                tag_stats_path=getattr(args, "tag_stats", None),
                config=config,
            ))


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

    config = PipelineConfig()
    try:
        asyncio.run(run_scoring(
            input_path=args.input,
            tag_stats_path=getattr(args, "tag_stats", None),
            limit=args.limit,
            config=config,
        ))
    except (FileNotFoundError, ValueError) as e:
        print(f"Error: {e}")
        sys.exit(1)


def cmd_filter(args):
    """Filter scored data by value threshold."""
    from sft_label.tools.filter_value import run_filter

    try:
        run_filter(
            input_path=args.input,
            threshold=args.threshold,
            output_path=args.output,
            include_unscored=args.include_unscored,
        )
    except (FileNotFoundError, ValueError) as e:
        print(f"Error: {e}")
        sys.exit(1)


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
    run_parser.add_argument("--input", type=str, required=True,
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
    run_parser.add_argument("--tag-stats", type=str, default=None,
                            help="Path to stats.json for Pass 2 rarity (used with --score)")

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

    # --- export-review ---
    review_parser = subparsers.add_parser("export-review",
                                          help="Export labeled data to review CSV")
    review_parser.add_argument("tool_args", nargs=argparse.REMAINDER,
                               help="Arguments passed to export_review tool")

    # --- filter ---
    filter_parser = subparsers.add_parser(
        "filter",
        help="Filter scored data by value threshold",
        description="Select high-value samples from scored data. "
                    "Filters by value_score >= threshold.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    filter_parser.add_argument("--input", type=str, required=True,
                                help="Input scored file (.json/.jsonl) or directory")
    filter_parser.add_argument("--threshold", type=float, required=True,
                                help="Minimum value_score to retain")
    filter_parser.add_argument("--output", type=str, default=None,
                                help="Output file path (default: auto-generated alongside input)")
    filter_parser.add_argument("--include-unscored", action="store_true",
                                help="Keep samples without value scores (default: exclude)")

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
    elif args.command == "filter":
        cmd_filter(args)
    elif args.command == "export-review":
        cmd_export_review(args)


if __name__ == "__main__":
    main()
