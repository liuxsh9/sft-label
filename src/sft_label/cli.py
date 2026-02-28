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
        asyncio.run(run(
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

    # --- validate ---
    subparsers.add_parser("validate", help="Validate taxonomy definitions")

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
    elif args.command == "export-review":
        cmd_export_review(args)


if __name__ == "__main__":
    main()
