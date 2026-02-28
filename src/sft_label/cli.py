"""
CLI entry point for sft-label.

Subcommands:
  sft-label run       — Run the labeling pipeline
  sft-label validate  — Validate taxonomy definitions
  sft-label export-review — Export labeled data to review CSV
"""

import argparse
import asyncio
import sys


def cmd_run(args):
    """Run the labeling pipeline."""
    from sft_label.config import PipelineConfig
    from sft_label.pipeline import run

    config = PipelineConfig(model=args.model, concurrency=args.concurrency)
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
                model=args.model,
                concurrency=args.concurrency,
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

    config = PipelineConfig(model=args.model, concurrency=args.concurrency)
    try:
        asyncio.run(run_scoring(
            input_path=args.input,
            tag_stats_path=getattr(args, "tag_stats", None),
            model=args.model,
            concurrency=args.concurrency,
            limit=args.limit,
            config=config,
        ))
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
    run_parser = subparsers.add_parser("run", help="Run the labeling pipeline")
    run_parser.add_argument("--input", type=str, required=True,
                            help="Input file or directory")
    run_parser.add_argument("--output", type=str, default=None,
                            help="Output directory")
    run_parser.add_argument("--resume", type=str, default=None,
                            help="Resume from an existing run directory")
    run_parser.add_argument("--model", type=str, default="gpt-4o-mini")
    run_parser.add_argument("--concurrency", type=int, default=100)
    run_parser.add_argument("--limit", type=int, default=0,
                            help="Max samples per file (0 = all)")
    run_parser.add_argument("--shuffle", action="store_true")
    run_parser.add_argument("--no-arbitration", action="store_true")
    run_parser.add_argument("--score", action="store_true",
                            help="Run Pass 2 value scoring after labeling")

    # --- validate ---
    subparsers.add_parser("validate", help="Validate taxonomy definitions")

    # --- score ---
    score_parser = subparsers.add_parser("score",
                                          help="Run value scoring (Pass 2) on pre-labeled data")
    score_parser.add_argument("--input", type=str, required=True,
                               help="Path to labeled.json or directory")
    score_parser.add_argument("--tag-stats", type=str, default=None,
                               help="Path to stats.json for rarity computation")
    score_parser.add_argument("--model", type=str, default="gpt-4o-mini")
    score_parser.add_argument("--concurrency", type=int, default=100)
    score_parser.add_argument("--limit", type=int, default=0,
                               help="Max samples to score (0 = all)")

    # --- export-review ---
    review_parser = subparsers.add_parser("export-review",
                                          help="Export labeled data to review CSV")
    review_parser.add_argument("tool_args", nargs=argparse.REMAINDER,
                               help="Arguments passed to export_review tool")

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
    elif args.command == "export-review":
        cmd_export_review(args)


if __name__ == "__main__":
    main()
