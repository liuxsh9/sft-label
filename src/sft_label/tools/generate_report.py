"""
Labeling Report Generator

Reads labeled data and stats to produce a human-readable summary report.

Usage:
  python3 labeling/generate_report.py
"""

import json
from pathlib import Path
from datetime import datetime

DATA_DIR = Path(__file__).parent.parent / "data"
LABELED_FILE = DATA_DIR / "labeled_deepseek_v4.json"
STATS_FILE = DATA_DIR / "stats_deepseek_v4.json"
REPORT_FILE = DATA_DIR / "labeling_report.md"


def load_data():
    with open(LABELED_FILE, "r", encoding="utf-8") as f:
        samples = json.load(f)
    with open(STATS_FILE, "r", encoding="utf-8") as f:
        stats = json.load(f)
    return samples, stats


def format_distribution_table(dist, max_items=20):
    """Format tag distribution as markdown table."""
    if not dist:
        return "_(empty)_\n"
    total = sum(dist.values())
    lines = ["| Tag | Count | % |", "|-----|-------|---|"]
    for tag, count in list(dist.items())[:max_items]:
        pct = count / total * 100
        bar = "█" * int(pct / 2)
        lines.append(f"| {tag} | {count} | {pct:.1f}% {bar} |")
    if len(dist) > max_items:
        lines.append(f"| _...{len(dist) - max_items} more_ | | |")
    return "\n".join(lines) + "\n"


def pick_examples(samples, n=5):
    """Pick diverse labeled examples for the report."""
    # Try to pick one from each intent
    by_intent = {}
    for s in samples:
        labels = s.get("labels")
        if not labels:
            continue
        intent = labels.get("intent", "unknown")
        if intent not in by_intent:
            by_intent[intent] = s

    picked = list(by_intent.values())[:n]

    # Fill remaining with diverse samples
    if len(picked) < n:
        for s in samples:
            if s not in picked and s.get("labels"):
                picked.append(s)
            if len(picked) >= n:
                break

    return picked


def format_example(sample, idx):
    """Format one labeled example as markdown."""
    labels = sample.get("labels", {})
    convs = sample.get("conversations", [])
    meta = sample.get("metadata", {})
    monitor = sample.get("labeling_monitor", {})

    # Get first human message as query preview
    query = ""
    for c in convs:
        if c.get("from") == "human":
            query = c.get("value", "")[:200]
            break

    lines = [
        f"### Example {idx+1}: {labels.get('intent', '?')} / {labels.get('difficulty', '?')}",
        "",
        f"> **Query**: {query}{'...' if len(query) >= 200 else ''}",
        "",
        f"| Dimension | Labels |",
        f"|-----------|--------|",
    ]

    for dim in ["intent", "language", "domain", "concept", "task", "constraint", "agentic", "context", "difficulty"]:
        val = labels.get(dim, "")
        if isinstance(val, list):
            val_str = ", ".join(val) if val else "_(empty)_"
        else:
            val_str = val if val else "_(empty)_"
        conf = labels.get("confidence", {}).get(dim, "?")
        if isinstance(conf, float):
            conf = f"{conf:.2f}"
        lines.append(f"| **{dim}** | {val_str} (conf: {conf}) |")

    if monitor.get("consistency_warnings"):
        lines.append("")
        lines.append(f"**Warnings**: {'; '.join(monitor['consistency_warnings'])}")

    lines.append("")
    return "\n".join(lines)


def generate_report(samples, stats):
    lines = []

    # Header
    lines.append("# SFT Auto-Labeling Trial Report")
    lines.append("")
    lines.append(f"> Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"> Model: {stats.get('model', 'unknown')}")
    lines.append(f"> Input: {stats.get('input_file', 'unknown')}")
    lines.append("")

    # Overview
    lines.append("## 1. Overview")
    lines.append("")
    lines.append(f"| Metric | Value |")
    lines.append(f"|--------|-------|")
    lines.append(f"| Total samples | {stats['total_samples']} |")
    lines.append(f"| Success rate | {stats['success_rate']*100:.1f}% ({stats['success']}/{stats['total_samples']}) |")
    lines.append(f"| Avg LLM calls/sample | {stats['avg_calls_per_sample']:.1f} |")
    lines.append(f"| Total LLM calls | {stats['total_llm_calls']} |")
    lines.append(f"| Total tokens | {stats['total_tokens']:,} |")
    lines.append(f"| Arbitration rate | {stats['arbitrated_rate']*100:.1f}% ({stats['arbitrated_count']}) |")
    lines.append(f"| Validation issues | {stats['validation_issue_count']} samples |")
    lines.append(f"| Consistency warnings | {stats['consistency_warning_count']} samples |")
    lines.append(f"| Unmapped tags (unique) | {stats['unmapped_unique_count']} |")
    lines.append("")

    # Confidence
    lines.append("## 2. Confidence by Dimension")
    lines.append("")
    lines.append("| Dimension | Mean | Min | Max | Below Threshold |")
    lines.append("|-----------|------|-----|-----|-----------------|")
    for dim in ["intent", "difficulty", "context", "language", "domain", "task", "concept", "agentic", "constraint"]:
        cs = stats.get("confidence_stats", {}).get(dim, {})
        if cs:
            bar = "█" * int(cs["mean"] * 20)
            lines.append(f"| {dim} | {cs['mean']:.3f} {bar} | {cs['min']:.2f} | {cs['max']:.2f} | {cs['below_threshold']} |")
    lines.append("")

    if stats.get("low_confidence_frequency"):
        lines.append("**Low-confidence frequency by dimension:**")
        for dim, count in stats["low_confidence_frequency"].items():
            lines.append(f"- {dim}: {count} samples")
        lines.append("")

    # Tag distributions
    lines.append("## 3. Tag Distributions")
    lines.append("")
    for dim in ["intent", "difficulty", "language", "domain", "concept", "task", "agentic", "constraint", "context"]:
        dist = stats.get("tag_distributions", {}).get(dim, {})
        lines.append(f"### {dim.capitalize()}")
        lines.append("")
        lines.append(format_distribution_table(dist))
        lines.append("")

    # Unmapped tags
    if stats.get("unmapped_tags"):
        lines.append("## 4. Unmapped Tags")
        lines.append("")
        lines.append("Tags the model wanted to assign but weren't in the tag pool:")
        lines.append("")
        lines.append("| Tag | Count | Action Needed |")
        lines.append("|-----|-------|---------------|")
        for tag, count in stats["unmapped_tags"].items():
            lines.append(f"| {tag} | {count} | Review for inclusion |")
        lines.append("")

    # Example labeled items
    lines.append("## 5. Example Labeled Items")
    lines.append("")
    examples = pick_examples(samples, n=6)
    for idx, ex in enumerate(examples):
        lines.append(format_example(ex, idx))

    # Recommendations
    lines.append("## 6. Observations & Next Steps")
    lines.append("")

    # Auto-generate some observations
    obs = []
    if stats["success_rate"] < 0.95:
        obs.append(f"- **Success rate ({stats['success_rate']*100:.1f}%)** is below 95%. Check failed samples for patterns.")
    if stats["avg_calls_per_sample"] > 3:
        obs.append(f"- **High arbitration rate** ({stats['arbitrated_rate']*100:.1f}%). Consider improving prompts for low-confidence dimensions.")
    if stats["unmapped_unique_count"] > 10:
        obs.append(f"- **{stats['unmapped_unique_count']} unmapped tags** detected. Review for potential taxonomy expansion.")

    # Check for dimension-specific issues
    for dim, cs in stats.get("confidence_stats", {}).items():
        if cs.get("mean", 1) < 0.75:
            obs.append(f"- **{dim}** has low mean confidence ({cs['mean']:.3f}). Prompt may need better examples/instructions for this dimension.")
        if cs.get("below_threshold", 0) > stats["total_samples"] * 0.2:
            obs.append(f"- **{dim}** has {cs['below_threshold']} samples below confidence threshold ({cs['below_threshold']/stats['total_samples']*100:.0f}%). Consider adding more few-shot examples.")

    if not obs:
        obs.append("- All metrics within expected ranges. Ready for larger-scale testing.")

    lines.extend(obs)
    lines.append("")
    lines.append("---")
    lines.append(f"_Report generated by `labeling/generate_report.py` at {datetime.now().isoformat()}_")

    return "\n".join(lines)


def main():
    if not LABELED_FILE.exists():
        print(f"Error: {LABELED_FILE} not found. Run labeling/pipeline.py first.")
        return
    if not STATS_FILE.exists():
        print(f"Error: {STATS_FILE} not found. Run labeling/pipeline.py first.")
        return

    samples, stats = load_data()
    report = generate_report(samples, stats)

    REPORT_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(REPORT_FILE, "w", encoding="utf-8") as f:
        f.write(report)

    print(f"Report generated: {REPORT_FILE}")
    print(f"  {stats['total_samples']} samples, {stats['success']} labeled successfully")


if __name__ == "__main__":
    main()
