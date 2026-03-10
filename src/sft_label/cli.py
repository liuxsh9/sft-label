"""
CLI entry point for sft-label.

Subcommands:
  sft-label start                — Interactive launcher for grouped workflows
  sft-label run                  — Run Pass 1 tag labeling (optionally + Pass 2)
  sft-label score                — Run Pass 2 value scoring on pre-labeled data
  sft-label refresh-rarity       — Refresh Pass 2 rarity/value fields (no LLM)
  sft-label semantic-cluster     — Run trajectory SemHash + ANN clustering
  sft-label export-semantic      — Export representative windows from clustering artifacts
  sft-label optimize-layout      — One-time optimize historical output layout
  sft-label filter               — Filter scored data by value threshold
  sft-label validate             — Validate taxonomy definitions
  sft-label export-review        — Export labeled data to review CSV
  sft-label recompute-stats      — Recompute stats from labeled/scored output (no LLM)
  sft-label regenerate-dashboard — Regenerate HTML dashboards from stats/data

Most defaults come from config.py (PipelineConfig).
Selected runtime knobs (concurrency/rps/timeout/retries) can be overridden via CLI flags.
"""

import argparse
import asyncio
import os
import sys
import time
import random
from pathlib import Path


def _start_msg(lang: str, zh: str, en: str) -> str:
    return en if lang == "en" else zh


def _is_sensitive_env_key(key: str) -> bool:
    upper = key.upper()
    return any(token in upper for token in ("KEY", "TOKEN", "SECRET", "PASSWORD"))


def _mask_secret(value: str) -> str:
    if not value:
        return "(empty)"
    if len(value) <= 6:
        return "*" * len(value)
    return f"{value[:2]}***{value[-2:]}"


def _format_start_env_lines(env_overrides: dict[str, str]) -> list[str]:
    lines: list[str] = []
    for key in sorted(env_overrides):
        value = env_overrides[key]
        shown = _mask_secret(value) if _is_sensitive_env_key(key) else value
        lines.append(f"  {key}={shown}")
    return lines


def _apply_runtime_overrides(config, args):
    """Apply optional runtime overrides from CLI args onto PipelineConfig."""
    shared_concurrency = getattr(args, "concurrency", None)
    legacy_scoring_concurrency = getattr(args, "scoring_concurrency", None)
    if shared_concurrency is None and legacy_scoring_concurrency is not None:
        shared_concurrency = legacy_scoring_concurrency
    if shared_concurrency is not None:
        # One shared LLM concurrency setting for Pass 1 and Pass 2.
        config.concurrency = shared_concurrency
        config.scoring_concurrency = shared_concurrency

    for attr, arg_name in (
        ("rps_limit", "rps_limit"),
        ("rps_warmup", "rps_warmup"),
        ("request_timeout", "request_timeout"),
        ("max_retries", "max_retries"),
    ):
        val = getattr(args, arg_name, None)
        if val is not None:
            setattr(config, attr, val)


def _format_eta(seconds) -> str:
    if seconds is None:
        return "--:--"
    sec = max(int(seconds), 0)
    h, rem = divmod(sec, 3600)
    m, s = divmod(rem, 60)
    if h > 0:
        return f"{h:02d}:{m:02d}:{s:02d}"
    return f"{m:02d}:{s:02d}"


class _CombinedLLMProgressTracker:
    """Track global LLM call progress across Pass 1 and Pass 2."""

    def __init__(self, planned_total_calls: int):
        self.planned_total_calls = max(int(planned_total_calls), 1)
        self.calls_done = 0
        self.pass1_calls = 0
        self.pass2_calls = 0
        self._start_time = time.time()
        self._smoothed_cps = 0.0

    def update(self, delta_calls: int, stage: str) -> str:
        delta = max(int(delta_calls or 0), 0)
        self.calls_done += delta
        if stage == "pass1":
            self.pass1_calls += delta
        elif stage == "pass2":
            self.pass2_calls += delta

        elapsed = max(time.time() - self._start_time, 1e-6)
        instant_cps = self.calls_done / elapsed
        if self._smoothed_cps <= 0:
            self._smoothed_cps = instant_cps
        else:
            self._smoothed_cps = 0.2 * instant_cps + 0.8 * self._smoothed_cps

        return self.progress_line()

    def eta_seconds(self):
        if self._smoothed_cps <= 0:
            return None
        remaining = max(self.planned_total_calls - self.calls_done, 0)
        return remaining / self._smoothed_cps

    def summary_line(self) -> str:
        cps = f"{self._smoothed_cps:.1f}/s" if self._smoothed_cps > 0 else "warming"
        eta = _format_eta(self.eta_seconds())
        return (
            f"{self.progress_line()} eta {eta} rate {cps} "
            f"p1={self.pass1_calls} p2={self.pass2_calls}"
        )

    def progress_line(self) -> str:
        pct = min(self.calls_done / max(self.planned_total_calls, 1) * 100, 100.0)
        return f"run {self.calls_done}/{self.planned_total_calls} ({pct:.1f}%)"


def _estimate_pass2_calls(total_samples: int, sample_max_retries: int) -> int:
    retries = max(int(sample_max_retries), 1)
    retry_factor = 1.0 + min(0.2, 0.05 * (retries - 1))
    est = int(round(total_samples * retry_factor))
    return max(est, total_samples)


def _estimate_end_to_end_llm_calls(args, config):
    """Estimate Pass1+Pass2 total LLM calls for run --score mode."""
    if args.resume or not args.input:
        return None

    input_path = Path(args.input)
    if not input_path.exists():
        return None

    from sft_label.pipeline import (
        discover_input_files,
        estimate_directory_workload,
        iter_samples_from_file,
        apply_sparse_sampling,
    )

    if input_path.is_dir():
        files = discover_input_files(input_path)
        dir_files = [(a, r) for a, r in files if r is not None]
        if not dir_files:
            return None
        est = estimate_directory_workload(
            dir_files,
            limit=args.limit,
            shuffle=args.shuffle,
            config=config,
            enable_arbitration=not args.no_arbitration,
        )
        pass1_calls = est.initial_estimated_llm_calls
        pass1_labeled = est.total_labeled_samples
        pass2_samples = est.total_samples
    else:
        rng_state = random.getstate()
        try:
            samples, _ = iter_samples_from_file(
                input_path,
                limit=args.limit,
                shuffle=args.shuffle,
            )
        finally:
            random.setstate(rng_state)

        sparse_kw = dict(
            full_label_count=config.sparse_full_label_count,
            gap_multiplier=config.sparse_gap_multiplier,
            min_gap=config.sparse_min_gap,
            max_gap=config.sparse_max_gap,
            threshold=config.sparse_threshold,
        )
        label_indices, _ = apply_sparse_sampling(samples, **sparse_kw)
        pass1_labeled = len(label_indices)
        pass2_samples = len(samples)
        baseline = pass1_labeled * 2
        calls_per_sample = 2.2 if not args.no_arbitration else 2.0
        pass1_calls = max(int(round(pass1_labeled * calls_per_sample)), baseline)

    pass2_calls = _estimate_pass2_calls(pass2_samples, config.sample_max_retries)
    return {
        "pass1_labeled_samples": pass1_labeled,
        "pass2_samples": pass2_samples,
        "pass1_est_calls": pass1_calls,
        "pass2_est_calls": pass2_calls,
        "total_est_calls": pass1_calls + pass2_calls,
    }


def cmd_run(args):
    """Run the labeling pipeline."""
    if not args.input and not args.resume:
        print("Error: --input is required (unless using --resume)")
        sys.exit(1)

    from sft_label.config import PipelineConfig
    from sft_label.pipeline import run

    config = PipelineConfig(prompt_mode=args.prompt_mode)
    _apply_runtime_overrides(config, args)
    if args.model:
        config.labeling_model = args.model
        config.scoring_model = args.model

    llm_progress_cb = None
    global_tracker = None
    if getattr(args, "score", False):
        plan = _estimate_end_to_end_llm_calls(args, config)
        if plan:
            global_tracker = _CombinedLLMProgressTracker(plan["total_est_calls"])
            print(
                "Run plan | "
                f"llm~{plan['total_est_calls']} = "
                f"p1~{plan['pass1_est_calls']} ({plan['pass1_labeled_samples']} labeled) + "
                f"p2~{plan['pass2_est_calls']} ({plan['pass2_samples']} samples)"
            )
            llm_progress_cb = global_tracker.update
    try:
        stats = asyncio.run(run(
            input_path=args.input,
            output=args.output,
            resume=args.resume,
            limit=args.limit,
            shuffle=args.shuffle,
            enable_arbitration=not args.no_arbitration,
            config=config,
            llm_progress_cb=llm_progress_cb,
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
                llm_progress_cb=llm_progress_cb,
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
    patched = [
        "sft-label export-review",
        "--input", args.input,
        "--output", args.output,
    ]
    if args.monitor:
        patched.extend(["--monitor", args.monitor])
    if args.output_format:
        patched.extend(["--format", args.output_format])
    sys.argv = patched
    try:
        export_main()
    except (FileNotFoundError, ValueError) as e:
        print(f"Error: {e}")
        sys.exit(1)


def cmd_score(args):
    """Run value scoring (Pass 2) on pre-labeled data."""
    from sft_label.config import PipelineConfig
    from sft_label.scoring import run_scoring

    config = PipelineConfig(prompt_mode=args.prompt_mode)
    _apply_runtime_overrides(config, args)
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
        preserve_structure=args.preserve_structure,
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


def cmd_refresh_rarity(args):
    """Refresh Pass 2 rarity/value fields from existing scored output."""
    from sft_label.config import PipelineConfig
    from sft_label.tools.recompute import run_refresh_rarity

    config = PipelineConfig()
    if getattr(args, "mode", None):
        config.rarity_score_mode = args.mode

    try:
        written = run_refresh_rarity(
            input_path=args.input,
            tag_stats_path=getattr(args, "tag_stats", None),
            output_dir=args.output,
            config=config,
        )
    except (FileNotFoundError, ValueError) as e:
        print(f"Error: {e}")
        sys.exit(1)

    if not written:
        print("No rarity updates generated. Check that input contains scored output.")
        sys.exit(1)
    print(f"\nDone. Refreshed rarity for {written.get('rarity_refreshed_samples', '0')} sample(s).")


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


def cmd_optimize_layout(args):
    """One-time optimize historical output file layout."""
    from sft_label.tools.optimize_layout import (
        run_optimize_layout,
        format_optimize_layout_summary,
    )

    try:
        summary = run_optimize_layout(
            input_path=args.input,
            apply=getattr(args, "apply", False),
            prune_legacy=getattr(args, "prune_legacy", False),
            manifest_path=getattr(args, "manifest", None),
        )
    except (FileNotFoundError, ValueError) as e:
        print(f"Error: {e}")
        sys.exit(1)

    print(format_optimize_layout_summary(summary))
    if not getattr(args, "apply", False):
        print("\nDry-run only. Re-run with --apply to execute changes.")


def cmd_start(args, parser):
    """Interactive launcher for grouped workflows."""
    from sft_label.launcher import (
        build_launch_plan,
        format_command,
        interactive_input,
        is_back_token,
        set_language,
        sanitize_prompt_input,
    )

    lang = getattr(args, "lang", "zh")
    if getattr(args, "en", False):
        lang = "en"
    set_language(lang)

    plan = build_launch_plan(input_fn=interactive_input, language=lang)
    if plan is None:
        print(_start_msg(lang, "已取消启动。", "Launch cancelled."))
        return

    cmd_preview = format_command(plan.argv)
    print(f"\n{_start_msg(lang, '执行确认', 'Launch summary')}")
    print("-" * 56)
    print(_start_msg(lang, "命令：", "Command:"))
    print(f"  {cmd_preview}")
    if plan.env_overrides:
        print(_start_msg(lang, "环境变量覆盖（敏感值已脱敏）：", "Environment overrides (sensitive values masked):"))
        for line in _format_start_env_lines(plan.env_overrides):
            print(line)
    else:
        print(_start_msg(lang, "环境变量覆盖：无", "Environment overrides: none"))
    print("-" * 56)

    if args.dry_run:
        print(f"\n{_start_msg(lang, '仅预览模式，未执行命令。', 'Dry run mode: command not executed.')}")
        return

    confirm_raw = interactive_input(
        _start_msg(lang, "立即执行？ [Y/b/n]: ", "Execute now? [Y/b/n]: ")
    )
    confirm, had_control = sanitize_prompt_input(confirm_raw)
    if had_control:
        print(_start_msg(lang, "检测到方向键/控制字符输入，已忽略。", "Detected arrow/control key input and ignored."))
    if is_back_token(confirm):
        print(_start_msg(lang, "已取消执行。", "Launch aborted."))
        return
    confirm = confirm.lower()
    if confirm in ("n", "no"):
        print(_start_msg(lang, "已取消执行。", "Launch aborted."))
        return

    for key, value in plan.env_overrides.items():
        os.environ[key] = value

    try:
        launched_args = parser.parse_args(plan.argv)
    except SystemExit:
        print(_start_msg(
            lang,
            "生成参数解析失败，请重试 `sft-label start`。",
            "Generated arguments failed to parse. Re-run `sft-label start`.",
        ))
        return

    if launched_args.command == "start":
        print(_start_msg(lang, "交互启动器不能再次启动自身。", "Interactive launcher cannot launch itself."))
        return

    dispatch_command(launched_args, parser=parser)


def build_parser():
    """Build CLI parser for all subcommands."""
    parser = argparse.ArgumentParser(
        prog="sft-label",
        description="SFT Capability Taxonomy & Auto-Labeling Pipeline",
    )
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # --- start ---
    start_parser = subparsers.add_parser(
        "start",
        help="Interactive launcher for grouped workflows",
        description="Interactive task launcher that groups pipeline features "
                    "and builds a runnable command for you.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    start_parser.add_argument("--dry-run", action="store_true",
                              help="Only print generated command; do not execute it")
    start_parser.add_argument("--lang", choices=["zh", "en"], default="zh",
                              help="Language for interactive prompts (default: zh)")
    start_parser.add_argument("--en", action="store_true",
                              help=argparse.SUPPRESS)

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
    run_parser.add_argument("--concurrency", type=int, default=None,
                            help="Override pass1 labeling concurrency")
    run_parser.add_argument("--scoring-concurrency", type=int, default=None,
                            help=argparse.SUPPRESS)
    run_parser.add_argument("--rps-limit", type=float, default=None,
                            help="Override global LLM requests/sec limit")
    run_parser.add_argument("--rps-warmup", type=float, default=None,
                            help="Override RPS warmup seconds")
    run_parser.add_argument("--request-timeout", type=int, default=None,
                            help="Override per-request timeout seconds")
    run_parser.add_argument("--max-retries", type=int, default=None,
                            help="Override max retries per request")

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
    score_parser.add_argument("--concurrency", type=int, default=None,
                               help="Override shared LLM concurrency (used by pass1/pass2)")
    score_parser.add_argument("--scoring-concurrency", type=int, default=None,
                               help=argparse.SUPPRESS)
    score_parser.add_argument("--rps-limit", type=float, default=None,
                               help="Override global LLM requests/sec limit")
    score_parser.add_argument("--rps-warmup", type=float, default=None,
                               help="Override RPS warmup seconds")
    score_parser.add_argument("--request-timeout", type=int, default=None,
                               help="Override per-request timeout seconds")
    score_parser.add_argument("--max-retries", type=int, default=None,
                               help="Override max retries per request")

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

    # --- optimize-layout ---
    optimize_parser = subparsers.add_parser(
        "optimize-layout",
        help="One-time optimize historical output layout",
        description="Normalize legacy output naming/layout in an existing run "
                    "directory. Default is dry-run; use --apply to execute.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    optimize_parser.add_argument("--input", type=str, required=True,
                                 help="Run directory root to optimize")
    optimize_parser.add_argument("--apply", action="store_true",
                                 help="Apply planned operations (default: dry-run)")
    optimize_parser.add_argument("--prune-legacy", action="store_true",
                                 help="Delete legacy alias files when canonical equivalents exist")
    optimize_parser.add_argument("--manifest", type=str, default=None,
                                 help="Manifest output path (default: <input>/layout_optimization_manifest.json)")

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

    # --- refresh-rarity ---
    refresh_parser = subparsers.add_parser(
        "refresh-rarity",
        help="Refresh Pass 2 rarity/value fields from existing scored output (no LLM)",
        description="Recompute value.rarity, value_score, selection_score and Pass 2 stats "
                    "from existing scored output without LLM calls.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    refresh_parser.add_argument("--input", type=str, required=True,
                                help="Scored file (.json/.jsonl) or run directory")
    refresh_parser.add_argument("--tag-stats", type=str, default=None,
                                help="Path to Pass 1 stats for cross-dataset rarity baseline")
    refresh_parser.add_argument("--output", type=str, default=None,
                                help="Output directory (default: in-place)")
    refresh_parser.add_argument("--mode", type=str, default=None,
                                choices=["absolute", "percentile"],
                                help="Rarity score normalization mode override")

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
    review_parser = subparsers.add_parser(
        "export-review",
        help="Export labeled data to review CSV",
    )
    review_parser.add_argument("--input", type=str, required=True,
                               help="Input labeled JSON file")
    review_parser.add_argument("--monitor", type=str, default="",
                               help="Monitor JSONL file (optional)")
    review_parser.add_argument("--output", type=str, required=True,
                               help="Output CSV/TSV path")
    review_parser.add_argument("--format", choices=["csv", "tsv"], default=None,
                               dest="output_format",
                               help="Output format override (default: infer from extension)")

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
    filter_parser.add_argument("--preserve-structure", action="store_true",
                                help="Directory input only: mirror input folder structure and file count")

    return parser


def dispatch_command(args, parser):
    """Dispatch parsed args to command handlers."""
    if args.command == "start":
        cmd_start(args, parser)
    elif args.command == "run":
        cmd_run(args)
    elif args.command == "validate":
        cmd_validate(args)
    elif args.command == "score":
        cmd_score(args)
    elif args.command == "semantic-cluster":
        cmd_semantic_cluster(args)
    elif args.command == "export-semantic":
        cmd_export_semantic(args)
    elif args.command == "optimize-layout":
        cmd_optimize_layout(args)
    elif args.command == "filter":
        cmd_filter(args)
    elif args.command == "export-review":
        cmd_export_review(args)
    elif args.command == "recompute-stats":
        cmd_recompute_stats(args)
    elif args.command == "refresh-rarity":
        cmd_refresh_rarity(args)
    elif args.command == "regenerate-dashboard":
        cmd_regenerate_dashboard(args)


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command is None:
        parser.print_help()
        sys.exit(1)

    dispatch_command(args, parser=parser)


if __name__ == "__main__":
    main()
