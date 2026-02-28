"""
Model Comparison Report Generator

Compares labeling results from multiple models on the same dataset.
Produces a markdown report with:
  - Agreement rates per dimension
  - Confidence comparison
  - Tag distribution divergence
  - Specific disagreement examples
  - Model selection recommendation

Usage:
  python3 labeling/compare_models.py
"""

import json
from pathlib import Path
from datetime import datetime
from collections import Counter

DATA_DIR = Path(__file__).parent.parent / "data"
REPORT_FILE = DATA_DIR / "model_comparison_report.md"


def load_labeled(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_stats(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def compute_agreement(samples_a, samples_b, dim, is_single=False):
    """Compute agreement rate between two model's labels for a dimension."""
    agree = 0
    total = 0
    partial = 0

    for sa, sb in zip(samples_a, samples_b):
        la = sa.get("labels")
        lb = sb.get("labels")
        if not la or not lb:
            continue
        total += 1

        va = la.get(dim, "" if is_single else [])
        vb = lb.get(dim, "" if is_single else [])

        if is_single:
            if va == vb:
                agree += 1
        else:
            set_a = set(va) if isinstance(va, list) else {va}
            set_b = set(vb) if isinstance(vb, list) else {vb}
            if set_a == set_b:
                agree += 1
            elif set_a & set_b:
                partial += 1

    if total == 0:
        return {"total": 0, "exact": 0, "partial": 0, "exact_rate": 0, "partial_rate": 0}

    return {
        "total": total,
        "exact": agree,
        "partial": partial,
        "exact_rate": round(agree / total, 4),
        "partial_rate": round((agree + partial) / total, 4) if not is_single else round(agree / total, 4),
    }


def compute_jaccard(samples_a, samples_b, dim):
    """Compute average Jaccard similarity for multi-select dimensions."""
    scores = []
    for sa, sb in zip(samples_a, samples_b):
        la = sa.get("labels")
        lb = sb.get("labels")
        if not la or not lb:
            continue
        va = set(la.get(dim, []) if isinstance(la.get(dim), list) else [la.get(dim, "")])
        vb = set(lb.get(dim, []) if isinstance(lb.get(dim), list) else [lb.get(dim, "")])
        va.discard("")
        vb.discard("")
        if not va and not vb:
            scores.append(1.0)
        elif not va or not vb:
            scores.append(0.0)
        else:
            scores.append(len(va & vb) / len(va | vb))
    return round(sum(scores) / max(len(scores), 1), 4) if scores else 0.0


def find_disagreements(samples_a, samples_b, dim, is_single=False, max_examples=5):
    """Find specific disagreement examples."""
    examples = []
    for sa, sb in zip(samples_a, samples_b):
        la = sa.get("labels")
        lb = sb.get("labels")
        if not la or not lb:
            continue

        va = la.get(dim, "" if is_single else [])
        vb = lb.get(dim, "" if is_single else [])

        disagree = False
        if is_single:
            disagree = va != vb
        else:
            disagree = set(va if isinstance(va, list) else [va]) != set(vb if isinstance(vb, list) else [vb])

        if disagree:
            query = ""
            for c in sa.get("conversations", []):
                if c.get("from") == "human":
                    query = c.get("value", "")[:120]
                    break
            examples.append({
                "id": sa.get("id", "?"),
                "query": query,
                "model_a": va,
                "model_b": vb,
                "conf_a": la.get("confidence", {}).get(dim, "?"),
                "conf_b": lb.get("confidence", {}).get(dim, "?"),
            })
        if len(examples) >= max_examples:
            break
    return examples


def generate_comparison_report(model_a_name, model_b_name, samples_a, samples_b, stats_a, stats_b):
    lines = []

    lines.append("# Model Comparison Report: Labeling Pipeline Evaluation")
    lines.append("")
    lines.append(f"> Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"> Dataset: {stats_a.get('total_samples', '?')} samples")
    lines.append(f"> Models: **{model_a_name}** vs **{model_b_name}**")
    lines.append("")

    # 1. Overview
    lines.append("## 1. Pipeline Metrics Overview")
    lines.append("")
    lines.append(f"| Metric | {model_a_name} | {model_b_name} |")
    lines.append(f"|--------|{'---'*len(model_a_name)}|{'---'*len(model_b_name)}|")
    lines.append(f"| Success rate | {stats_a.get('success_rate', 0)*100:.1f}% | {stats_b.get('success_rate', 0)*100:.1f}% |")
    lines.append(f"| Avg calls/sample | {stats_a.get('avg_calls_per_sample', 0):.1f} | {stats_b.get('avg_calls_per_sample', 0):.1f} |")
    lines.append(f"| Total tokens | {stats_a.get('total_tokens', 0):,} | {stats_b.get('total_tokens', 0):,} |")
    lines.append(f"| Arbitration rate | {stats_a.get('arbitrated_rate', 0)*100:.1f}% | {stats_b.get('arbitrated_rate', 0)*100:.1f}% |")
    lines.append(f"| Elapsed | {stats_a.get('total_elapsed_seconds', 0):.0f}s | {stats_b.get('total_elapsed_seconds', 0):.0f}s |")
    lines.append(f"| Unmapped tags | {stats_a.get('unmapped_unique_count', 0)} | {stats_b.get('unmapped_unique_count', 0)} |")
    lines.append("")

    # 2. Confidence Comparison
    lines.append("## 2. Confidence Comparison (mean)")
    lines.append("")
    lines.append(f"| Dimension | {model_a_name} | {model_b_name} | Delta |")
    lines.append(f"|-----------|{'---'*len(model_a_name)}|{'---'*len(model_b_name)}|-------|")
    for dim in ["intent", "language", "domain", "task", "difficulty", "concept", "agentic", "constraint", "context"]:
        ca = stats_a.get("confidence_stats", {}).get(dim, {}).get("mean", 0)
        cb = stats_b.get("confidence_stats", {}).get(dim, {}).get("mean", 0)
        delta = cb - ca
        sign = "+" if delta >= 0 else ""
        bar_a = "█" * int(ca * 10)
        bar_b = "█" * int(cb * 10)
        lines.append(f"| {dim} | {ca:.3f} {bar_a} | {cb:.3f} {bar_b} | {sign}{delta:.3f} |")
    lines.append("")

    # 3. Inter-Model Agreement
    lines.append("## 3. Inter-Model Agreement")
    lines.append("")
    lines.append("| Dimension | Select | Exact Match | Partial Match | Jaccard |")
    lines.append("|-----------|--------|-------------|---------------|---------|")

    single_dims = {"intent", "difficulty", "context"}
    for dim in ["intent", "language", "domain", "task", "difficulty", "concept", "agentic", "constraint", "context"]:
        is_single = dim in single_dims
        agr = compute_agreement(samples_a, samples_b, dim, is_single)
        jac = compute_jaccard(samples_a, samples_b, dim)
        select_type = "single" if is_single else "multi"
        exact_bar = "█" * int(agr["exact_rate"] * 10)
        lines.append(f"| {dim} | {select_type} | {agr['exact_rate']*100:.1f}% {exact_bar} | {agr['partial_rate']*100:.1f}% | {jac:.3f} |")
    lines.append("")

    # 4. Tag Distribution Comparison
    lines.append("## 4. Tag Distribution Comparison")
    lines.append("")
    for dim in ["intent", "difficulty", "domain", "concept", "task"]:
        lines.append(f"### {dim.capitalize()}")
        lines.append("")
        dist_a = stats_a.get("tag_distributions", {}).get(dim, {})
        dist_b = stats_b.get("tag_distributions", {}).get(dim, {})
        all_tags = sorted(set(dist_a.keys()) | set(dist_b.keys()))

        if all_tags:
            lines.append(f"| Tag | {model_a_name} | {model_b_name} | Diff |")
            lines.append(f"|-----|{'---'*len(model_a_name)}|{'---'*len(model_b_name)}|------|")
            for tag in all_tags[:15]:
                ca = dist_a.get(tag, 0)
                cb = dist_b.get(tag, 0)
                diff = cb - ca
                sign = "+" if diff > 0 else ""
                lines.append(f"| {tag} | {ca} | {cb} | {sign}{diff} |")
            lines.append("")

    # 5. Disagreement Examples
    lines.append("## 5. Key Disagreements")
    lines.append("")

    for dim in ["intent", "difficulty", "concept", "agentic", "context"]:
        is_single = dim in single_dims
        examples = find_disagreements(samples_a, samples_b, dim, is_single, max_examples=3)
        if examples:
            lines.append(f"### {dim.capitalize()} disagreements")
            lines.append("")
            for ex in examples:
                lines.append(f"**{ex['id']}**: _{ex['query']}{'...' if len(ex['query']) >= 120 else ''}_")
                a_str = ex['model_a'] if isinstance(ex['model_a'], str) else ', '.join(ex['model_a'])
                b_str = ex['model_b'] if isinstance(ex['model_b'], str) else ', '.join(ex['model_b'])
                lines.append(f"- {model_a_name}: `{a_str}` (conf: {ex['conf_a']})")
                lines.append(f"- {model_b_name}: `{b_str}` (conf: {ex['conf_b']})")
                lines.append("")

    # 6. Unmapped Tags Comparison
    lines.append("## 6. Unmapped Tags")
    lines.append("")
    unmapped_a = stats_a.get("unmapped_tags", {})
    unmapped_b = stats_b.get("unmapped_tags", {})
    all_unmapped = sorted(set(unmapped_a.keys()) | set(unmapped_b.keys()))
    if all_unmapped:
        lines.append(f"| Tag | {model_a_name} | {model_b_name} |")
        lines.append(f"|-----|{'---'*len(model_a_name)}|{'---'*len(model_b_name)}|")
        for tag in all_unmapped:
            lines.append(f"| {tag} | {unmapped_a.get(tag, 0)} | {unmapped_b.get(tag, 0)} |")
        lines.append("")
    else:
        lines.append("No unmapped tags from either model.")
        lines.append("")

    # 7. Recommendations
    lines.append("## 7. Model Selection Recommendation")
    lines.append("")

    # Auto-generate recommendations
    token_ratio = stats_b.get("total_tokens", 0) / max(stats_a.get("total_tokens", 1), 1)
    time_ratio = stats_b.get("total_elapsed_seconds", 0) / max(stats_a.get("total_elapsed_seconds", 1), 1)

    # Calculate overall agreement
    total_exact = 0
    total_checks = 0
    for dim in ["intent", "language", "domain", "task", "difficulty", "concept", "agentic", "constraint", "context"]:
        is_single = dim in single_dims
        agr = compute_agreement(samples_a, samples_b, dim, is_single)
        total_exact += agr["exact"]
        total_checks += agr["total"]
    overall_agreement = total_exact / max(total_checks, 1)

    lines.append(f"**Overall inter-model agreement: {overall_agreement*100:.1f}%**")
    lines.append("")
    lines.append(f"- Token usage ratio ({model_b_name}/{model_a_name}): {token_ratio:.2f}x")
    lines.append(f"- Time ratio ({model_b_name}/{model_a_name}): {time_ratio:.2f}x")
    lines.append("")

    if overall_agreement > 0.8:
        lines.append(f"High agreement (>{80}%) suggests both models produce consistent labels. "
                      f"The cheaper model ({model_a_name}) may be sufficient for production use.")
    elif overall_agreement > 0.6:
        lines.append(f"Moderate agreement ({overall_agreement*100:.0f}%). Consider using the stronger model "
                      f"({model_b_name}) for dimensions with low agreement, and the lighter model for high-agreement dimensions.")
    else:
        lines.append(f"Low agreement ({overall_agreement*100:.0f}%). Significant calibration differences. "
                      f"Recommend further analysis of which model is more accurate (human review needed).")

    lines.append("")
    lines.append("---")
    lines.append(f"_Report generated by `labeling/compare_models.py` at {datetime.now().isoformat()}_")
    return "\n".join(lines)


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-a", type=str, default="deepseek-v3.2")
    parser.add_argument("--model-b", type=str, default="claude-sonnet-4-6")
    parser.add_argument("--labeled-a", type=str, default=str(DATA_DIR / "labeled_deepseek_v2.json"))
    parser.add_argument("--labeled-b", type=str, default=str(DATA_DIR / "labeled_sonnet_v2.json"))
    parser.add_argument("--stats-a", type=str, default=str(DATA_DIR / "stats_deepseek_v2.json"))
    parser.add_argument("--stats-b", type=str, default=str(DATA_DIR / "stats_sonnet_v2.json"))
    parser.add_argument("--output", type=str, default=str(REPORT_FILE))
    args = parser.parse_args()

    labeled_a_path = Path(args.labeled_a)
    labeled_b_path = Path(args.labeled_b)
    stats_a_path = Path(args.stats_a)
    stats_b_path = Path(args.stats_b)

    if not labeled_a_path.exists():
        print(f"Error: {labeled_a_path} not found")
        return
    if not labeled_b_path.exists():
        print(f"Error: {labeled_b_path} not found")
        return

    samples_a = load_labeled(labeled_a_path)
    samples_b = load_labeled(labeled_b_path)

    stats_a = load_stats(stats_a_path) if stats_a_path.exists() else {}
    stats_b = load_stats(stats_b_path) if stats_b_path.exists() else {}

    report = generate_comparison_report(
        args.model_a, args.model_b,
        samples_a, samples_b,
        stats_a, stats_b,
    )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(report)

    print(f"Comparison report generated: {output_path}")


if __name__ == "__main__":
    main()
