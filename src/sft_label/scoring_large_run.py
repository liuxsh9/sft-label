from __future__ import annotations

import json
import os
from pathlib import Path


def should_run_directory_global_selection_rewrite(postprocess_policy: dict | None) -> bool:
    # Directory-global selection is core Pass 2 completion semantics, so it must
    # always run even when heavy postprocessing is deferred.
    _ = postprocess_policy
    return True


def _iter_scored_samples(path, *, load_scored_samples):
    """Yield scored samples from .jsonl or .json without forcing JSONL into memory."""
    path = Path(path)
    if path.suffix == ".jsonl":
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    yield json.loads(line)
        return

    for sample in load_scored_samples(path):
        yield sample


def stream_selection_summaries(
    scored_path,
    *,
    config=None,
    selection_summary_from_sample,
    load_scored_samples,
):
    """Collect lightweight selection summaries for a scored file."""
    summaries = []
    scored_count = 0
    for sample in _iter_scored_samples(scored_path, load_scored_samples=load_scored_samples):
        if sample.get("value"):
            summaries.append(selection_summary_from_sample(sample, config=config))
            scored_count += 1
    return summaries, scored_count


def rewrite_scored_jsonl_selection(
    scored_path,
    selection_results,
    cursor,
    *,
    config=None,
    apply_v2_scores,
):
    """Rewrite a scored.jsonl file in a streaming fashion."""
    scored_path = Path(scored_path)
    tmp_path = scored_path.with_name(f".{scored_path.name}.tmp")
    with open(scored_path, "r", encoding="utf-8") as fin, open(tmp_path, "w", encoding="utf-8") as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            sample = json.loads(line)
            if sample.get("value"):
                if cursor >= len(selection_results):
                    raise ValueError(f"selection result underflow while rewriting {scored_path}")
                selection = selection_results[cursor]
                sample["value"]["selection_score"] = selection["selection_score"]
                sample["value"]["intra_class_rank"] = selection["intra_class_rank"]
                apply_v2_scores([sample], config=config)
                cursor += 1
            fout.write(json.dumps(sample, ensure_ascii=False) + "\n")
    os.replace(tmp_path, scored_path)
    return cursor


def rewrite_scored_json_sibling_from_jsonl(scored_jsonl_path, scored_json_path):
    """Keep scored.json in sync with rewritten scored.jsonl without full-file loads."""
    scored_jsonl_path = Path(scored_jsonl_path)
    scored_json_path = Path(scored_json_path)
    if not scored_json_path.exists():
        return

    tmp_json = scored_json_path.with_suffix(".tmp.json")
    with open(scored_jsonl_path, "r", encoding="utf-8") as fin, open(tmp_json, "w", encoding="utf-8") as fout:
        fout.write("[")
        wrote_any = False
        for line in fin:
            line = line.strip()
            if not line:
                continue
            if wrote_any:
                fout.write(",")
            fout.write("\n  ")
            fout.write(line)
            wrote_any = True
        if wrote_any:
            fout.write("\n")
        fout.write("]\n")
    os.replace(tmp_json, scored_json_path)


def rewrite_directory_global_selection(
    *,
    output_dir,
    input_dir,
    config,
    pass2_stats_file,
    pass2_dashboard_file,
    discover_scored_output_files,
    selection_summary_from_sample,
    compute_selection_scores_from_summaries,
    apply_v2_scores,
    load_scored_samples,
    write_scored_samples,
    compute_value_stats_from_summaries,
    load_monitor_totals,
    load_existing_pass2_stats,
    relative_file_label,
    generate_dashboard_fn=None,
    pprint=print,
    generate_dashboard=True,
):
    """Recompute selection globally across directory outputs and rewrite files."""
    scored_files = discover_scored_output_files(output_dir)
    if not scored_files:
        return []

    pprint(f"  Recomputing global selection across {len(scored_files)} scored file(s)")

    summaries = []
    file_entries = []
    for scored_path in scored_files:
        file_summaries, scored_count = stream_selection_summaries(
            scored_path,
            config=config,
            selection_summary_from_sample=selection_summary_from_sample,
            load_scored_samples=load_scored_samples,
        )
        summaries.extend(file_summaries)
        file_entries.append({
            "path": scored_path,
            "scored_count": scored_count,
            "monitor_totals": load_monitor_totals(scored_path.parent),
        })

    try:
        compute_selection_scores_from_summaries(
            summaries,
            config=config,
            return_results=False,
        )
    except TypeError:
        selection_results = compute_selection_scores_from_summaries(summaries, config=config)
        for summary, selection in zip(summaries, selection_results):
            summary["selection_score"] = selection["selection_score"]
            summary["intra_class_rank"] = selection["intra_class_rank"]

    updated_stats = []
    cursor = 0
    for entry in file_entries:
        scored_path = entry["path"]
        file_selection_results = summaries[cursor: cursor + entry["scored_count"]]
        if len(file_selection_results) != entry["scored_count"]:
            raise ValueError(
                f"selection result count mismatch for {scored_path}: "
                f"expected {entry['scored_count']}, got {len(file_selection_results)}"
            )
        file_summaries, reread_scored_count = stream_selection_summaries(
            scored_path,
            config=config,
            selection_summary_from_sample=selection_summary_from_sample,
            load_scored_samples=load_scored_samples,
        )
        if reread_scored_count != entry["scored_count"]:
            raise ValueError(
                f"scored sample count changed while rewriting {scored_path}: "
                f"expected {entry['scored_count']}, got {reread_scored_count}"
            )
        for summary, selection in zip(file_summaries, file_selection_results):
            summary["selection_score"] = selection["selection_score"]
            summary["intra_class_rank"] = selection["intra_class_rank"]
            sample_stub = {"value": summary}
            apply_v2_scores([sample_stub], config=config)
            summary["value_score_v2"] = sample_stub["value"].get("value_score_v2")
            summary["selection_score_v2"] = sample_stub["value"].get("selection_score_v2")

        json_sibling = scored_path.parent / "scored.json"
        if scored_path.suffix == ".jsonl":
            cursor = rewrite_scored_jsonl_selection(
                scored_path,
                summaries,
                cursor,
                config=config,
                apply_v2_scores=apply_v2_scores,
            )
            rewrite_scored_json_sibling_from_jsonl(scored_path, json_sibling)
        else:
            samples = load_scored_samples(scored_path)
            for sample in samples:
                value = sample.get("value")
                if not value:
                    continue
                if cursor >= len(summaries):
                    raise ValueError(f"selection result underflow while rewriting {scored_path}")
                selection = summaries[cursor]
                value["selection_score"] = selection["selection_score"]
                value["intra_class_rank"] = selection["intra_class_rank"]
                apply_v2_scores([sample], config=config)
                cursor += 1
            write_scored_samples(scored_path, samples)

        stats = compute_value_stats_from_summaries(
            file_summaries,
            entry["monitor_totals"],
            entry["scored_count"],
        )
        existing_stats = load_existing_pass2_stats(scored_path.parent)
        for key in (
            "elapsed_seconds",
            "model",
            "input_file",
            "http_request_stats",
            "weights_used",
            "rarity_config",
            "extension_rarity_config",
            "adaptive_runtime",
            "recovery_sweep",
            "postprocess",
            "chunked",
        ):
            if key in existing_stats:
                stats[key] = existing_stats[key]

        input_file = existing_stats.get("input_file")
        file_label_source = Path(input_file) if input_file else scored_path
        stats["file"] = relative_file_label(file_label_source, input_dir)

        stats_path = scored_path.parent / pass2_stats_file
        tmp_stats = stats_path.with_suffix(".tmp.json")
        with open(tmp_stats, "w", encoding="utf-8") as f:
            json.dump({k: v for k, v in stats.items() if k != "_raw_scores"},
                      f, ensure_ascii=False, indent=2)
        os.replace(tmp_stats, stats_path)

        if generate_dashboard and generate_dashboard_fn is not None:
            try:
                generate_dashboard_fn(
                    scored_path.parent,
                    scored_file="scored.json" if (scored_path.parent / "scored.json").exists() else "scored.jsonl",
                    stats_file=pass2_stats_file,
                    output_file=pass2_dashboard_file,
                    quiet=True,
                )
            except Exception:
                pass
        updated_stats.append(stats)

    if cursor != len(summaries):
        raise ValueError(
            f"global selection rewrite consumed {cursor} results, expected {len(summaries)}"
        )

    return updated_stats
