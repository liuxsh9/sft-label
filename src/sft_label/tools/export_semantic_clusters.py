"""Export semantic clustering rows for review/training consumption."""

from __future__ import annotations

import json
from pathlib import Path

from sft_label.semantic_artifacts import (
    manifest_output_prefix,
    resolve_semantic_artifact_dir,
)


def _load_jsonl(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def _discover_file(input_dir: Path, suffix: str) -> Path:
    candidates = sorted(input_dir.glob(f"*{suffix}"))
    if not candidates:
        raise FileNotFoundError(f"No semantic artifact matching '*{suffix}' in {input_dir}")
    return candidates[0]


def run_export_semantic_clusters(
    input_dir: str | Path,
    output_path: str | Path,
    include_non_representative: bool = False,
) -> int:
    """Export rows joined from cluster membership + window content."""
    base = Path(input_dir)
    if not base.exists():
        raise FileNotFoundError(f"Input directory does not exist: {base}")
    base = resolve_semantic_artifact_dir(base)

    prefix = manifest_output_prefix(base)
    if prefix:
        windows_file = base / f"{prefix}_windows.jsonl"
        members_file = base / f"{prefix}_cluster_membership.jsonl"
        if not windows_file.exists() or not members_file.exists():
            raise FileNotFoundError(
                "Semantic manifest prefix is set but required artifacts are missing: "
                f"{windows_file.name}, {members_file.name}"
            )
    else:
        windows_file = _discover_file(base, "_windows.jsonl")
        members_file = _discover_file(base, "_cluster_membership.jsonl")

    windows = {}
    for row in _load_jsonl(windows_file):
        wid = row.get("window_id")
        if wid:
            windows[wid] = row

    out_rows = []
    missing_window_refs = 0
    for member in _load_jsonl(members_file):
        if not include_non_representative and not member.get("representative"):
            continue
        wid = member.get("window_id")
        window = windows.get(wid)
        if not window:
            missing_window_refs += 1
            continue
        out_rows.append({
            "cluster_id": member.get("cluster_id"),
            "representative": bool(member.get("representative")),
            "snr": member.get("snr"),
            "action_tokens": member.get("action_tokens"),
            "observation_tokens": member.get("observation_tokens"),
            "value_score": member.get("value_score"),
            "window": window,
        })

    if missing_window_refs:
        raise ValueError(
            "Cluster membership references missing windows: "
            f"{missing_window_refs} row(s) missing from {windows_file.name}"
        )

    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        for row in out_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    return len(out_rows)
