"""One-time run-directory layout optimizer.

This tool normalizes historical artifact naming/layout without changing
runtime pipeline behavior:
1. Normalize per-file directory artifacts from suffixed names to canonical names.
2. Promote legacy aliases to canonical names when canonical files are missing.
3. Optionally prune legacy aliases when canonical equivalents exist.
4. Write a manifest with planned/executed actions for auditability.
"""

from __future__ import annotations

import json
import shutil
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path

from sft_label.artifacts import (
    PASS1_DASHBOARD_FILE,
    PASS1_DASHBOARD_FILE_LEGACY,
    PASS1_STATS_FILE,
    PASS1_STATS_FILE_LEGACY,
    PASS1_SUMMARY_STATS_FILE,
    PASS1_SUMMARY_STATS_FILE_LEGACY,
    PASS2_DASHBOARD_FILE,
    PASS2_DASHBOARD_FILE_LEGACY,
    PASS2_STATS_FILE,
    PASS2_STATS_FILE_LEGACY,
    PASS2_SUMMARY_STATS_FILE,
    PASS2_SUMMARY_STATS_FILE_LEGACY,
)


MANIFEST_FILENAME = "layout_optimization_manifest.json"
MANIFEST_SCHEMA = "sft-label-layout-opt-v1"


@dataclass
class LayoutAction:
    """A planned or executed file operation."""

    kind: str
    reason: str
    src: str | None = None
    dst: str | None = None
    status: str = "planned"
    detail: str | None = None


def _files_identical(a: Path, b: Path) -> bool:
    """Best-effort byte comparison."""
    if not a.exists() or not b.exists():
        return False
    if a.stat().st_size != b.stat().st_size:
        return False
    with open(a, "rb") as fa, open(b, "rb") as fb:
        while True:
            ca = fa.read(1024 * 1024)
            cb = fb.read(1024 * 1024)
            if ca != cb:
                return False
            if not ca:
                return True


def _is_candidate_dir(path: Path) -> bool:
    """Whether a directory likely contains pipeline outputs."""
    try:
        files = [p.name for p in path.iterdir() if p.is_file()]
    except OSError:
        return False

    if not files:
        return False

    for name in files:
        if name == "checkpoint.json":
            return True
        if name == "conversation_scores.json":
            return True
        if name.endswith(".json") and (
            name.startswith("labeled")
            or name.startswith("scored")
            or name.startswith("stats")
            or name.startswith("summary_stats")
        ):
            return True
        if name.endswith(".jsonl") and (
            name.startswith("labeled")
            or name.startswith("scored")
            or name.startswith("monitor")
            or name.startswith("failed")
            or name.endswith("failures.jsonl")
        ):
            return True
        if name.startswith("dashboard") and name.endswith(".html"):
            return True
    return False


def _iter_candidate_dirs(root: Path) -> list[Path]:
    """Collect root and nested directories that appear to contain outputs."""
    dirs = [root]
    dirs.extend(p for p in root.rglob("*") if p.is_dir())
    return [d for d in sorted(dirs) if _is_candidate_dir(d)]


def _append_action(actions: list[LayoutAction], action: LayoutAction, dedup: set[tuple]):
    key = (action.kind, action.src, action.dst, action.reason)
    if key in dedup:
        return
    dedup.add(key)
    actions.append(action)


def _plan_suffix_normalization(path: Path, actions: list[LayoutAction], dedup: set[tuple]):
    """Normalize directory-mode suffixed file names to unsuffixed canonical names."""
    stem = path.name
    mapping = (
        (f"labeled_{stem}.json", "labeled.json"),
        (f"labeled_{stem}.jsonl", "labeled.jsonl"),
        (f"monitor_{stem}.jsonl", "monitor.jsonl"),
        (f"failed_samples_{stem}.jsonl", "failed_samples.jsonl"),
        (f"stats_labeling_{stem}.json", PASS1_STATS_FILE),
        (f"stats_{stem}.json", PASS1_STATS_FILE_LEGACY),
        (f"dashboard_labeling_{stem}.html", PASS1_DASHBOARD_FILE),
        (f"dashboard_{stem}.html", PASS1_DASHBOARD_FILE_LEGACY),
    )

    for src_name, dst_name in mapping:
        src = path / src_name
        dst = path / dst_name
        if not src.exists():
            continue
        if src == dst:
            continue
        if not dst.exists():
            _append_action(
                actions,
                LayoutAction(
                    kind="move",
                    reason="normalize_suffixed_name",
                    src=str(src),
                    dst=str(dst),
                ),
                dedup,
            )
            continue
        if _files_identical(src, dst):
            _append_action(
                actions,
                LayoutAction(
                    kind="delete",
                    reason="drop_duplicate_suffixed_file",
                    src=str(src),
                ),
                dedup,
            )
        else:
            _append_action(
                actions,
                LayoutAction(
                    kind="conflict",
                    reason="conflicting_suffixed_and_unsuffixed_files",
                    src=str(src),
                    dst=str(dst),
                    status="conflict",
                ),
                dedup,
            )


def _plan_promote_legacy(path: Path, actions: list[LayoutAction], dedup: set[tuple]):
    """If canonical is missing but legacy exists, copy legacy -> canonical."""
    planned_dsts = {a.dst for a in actions if a.dst}
    pairs = (
        (PASS1_STATS_FILE_LEGACY, PASS1_STATS_FILE),
        (PASS2_STATS_FILE_LEGACY, PASS2_STATS_FILE),
        (PASS1_SUMMARY_STATS_FILE_LEGACY, PASS1_SUMMARY_STATS_FILE),
        (PASS2_SUMMARY_STATS_FILE_LEGACY, PASS2_SUMMARY_STATS_FILE),
        (PASS1_DASHBOARD_FILE_LEGACY, PASS1_DASHBOARD_FILE),
        (PASS2_DASHBOARD_FILE_LEGACY, PASS2_DASHBOARD_FILE),
    )
    for legacy_name, canonical_name in pairs:
        legacy = path / legacy_name
        canonical = path / canonical_name
        legacy_will_exist = legacy.exists() or str(legacy) in planned_dsts
        if legacy_will_exist and not canonical.exists():
            _append_action(
                actions,
                LayoutAction(
                    kind="copy",
                    reason="promote_legacy_alias_to_canonical",
                    src=str(legacy),
                    dst=str(canonical),
                ),
                dedup,
            )


def _plan_prune_legacy(path: Path, actions: list[LayoutAction], dedup: set[tuple]):
    """Plan deletion of legacy aliases when canonical equivalents exist."""
    planned_dsts = {a.dst for a in actions if a.dst}

    def _has_planned_copy(src: Path, dst: Path) -> bool:
        return any(
            a.kind == "copy" and a.src == str(src) and a.dst == str(dst)
            for a in actions
        )

    alias_pairs = (
        (PASS1_STATS_FILE_LEGACY, PASS1_STATS_FILE),
        (PASS2_STATS_FILE_LEGACY, PASS2_STATS_FILE),
        (PASS1_SUMMARY_STATS_FILE_LEGACY, PASS1_SUMMARY_STATS_FILE),
        (PASS2_SUMMARY_STATS_FILE_LEGACY, PASS2_SUMMARY_STATS_FILE),
        (PASS1_DASHBOARD_FILE_LEGACY, PASS1_DASHBOARD_FILE),
        (PASS2_DASHBOARD_FILE_LEGACY, PASS2_DASHBOARD_FILE),
    )
    for legacy_name, canonical_name in alias_pairs:
        legacy = path / legacy_name
        canonical = path / canonical_name
        legacy_will_exist = legacy.exists() or str(legacy) in planned_dsts
        if not legacy_will_exist:
            continue
        canonical_will_exist = canonical.exists() or str(canonical) in planned_dsts
        if not canonical_will_exist:
            continue

        if canonical.exists() and _files_identical(legacy, canonical):
            _append_action(
                actions,
                LayoutAction(
                    kind="delete",
                    reason="prune_legacy_alias",
                    src=str(legacy),
                ),
                dedup,
            )
        elif canonical.exists():
            _append_action(
                actions,
                LayoutAction(
                    kind="conflict",
                    reason="legacy_and_canonical_differ",
                    src=str(legacy),
                    dst=str(canonical),
                    status="conflict",
                ),
                dedup,
            )
        elif _has_planned_copy(legacy, canonical):
            _append_action(
                actions,
                LayoutAction(
                    kind="delete",
                    reason="prune_legacy_alias_after_copy",
                    src=str(legacy),
                ),
                dedup,
            )

    temp_stats = path / ".directory_rarity_stats.json"
    if temp_stats.exists():
        _append_action(
            actions,
            LayoutAction(
                kind="delete",
                reason="remove_temporary_rarity_stats",
                src=str(temp_stats),
            ),
            dedup,
        )

    for name in ("failed_samples.jsonl", "failed_value.jsonl", "score_failures.jsonl", "failures.jsonl"):
        fp = path / name
        if fp.exists():
            try:
                if fp.stat().st_size == 0:
                    _append_action(
                        actions,
                        LayoutAction(
                            kind="delete",
                            reason="remove_empty_failure_file",
                            src=str(fp),
                        ),
                        dedup,
                    )
            except OSError:
                continue


def _plan_actions(root: Path, prune_legacy: bool) -> tuple[list[LayoutAction], list[Path]]:
    actions: list[LayoutAction] = []
    dedup: set[tuple] = set()
    dirs = _iter_candidate_dirs(root)

    for path in dirs:
        _plan_suffix_normalization(path, actions, dedup)
        _plan_promote_legacy(path, actions, dedup)
        if prune_legacy:
            _plan_prune_legacy(path, actions, dedup)

    return actions, dirs


def _execute_action(action: LayoutAction):
    """Execute one planned file action in-place."""
    if action.kind == "conflict":
        action.status = "conflict"
        return

    src = Path(action.src) if action.src else None
    dst = Path(action.dst) if action.dst else None

    try:
        if action.kind == "move":
            if src is None or dst is None:
                action.status = "error"
                action.detail = "missing src/dst"
                return
            if not src.exists():
                action.status = "skipped"
                action.detail = "source missing"
                return
            dst.parent.mkdir(parents=True, exist_ok=True)
            if dst.exists():
                if _files_identical(src, dst):
                    src.unlink()
                    action.status = "done"
                    action.detail = "removed duplicate source"
                else:
                    action.status = "conflict"
                    action.detail = "destination exists with different content"
                return
            src.replace(dst)
            action.status = "done"
            return

        if action.kind == "copy":
            if src is None or dst is None:
                action.status = "error"
                action.detail = "missing src/dst"
                return
            if not src.exists():
                action.status = "skipped"
                action.detail = "source missing"
                return
            dst.parent.mkdir(parents=True, exist_ok=True)
            if dst.exists():
                if _files_identical(src, dst):
                    action.status = "skipped"
                    action.detail = "destination already identical"
                else:
                    action.status = "conflict"
                    action.detail = "destination exists with different content"
                return
            shutil.copyfile(src, dst)
            action.status = "done"
            return

        if action.kind == "delete":
            if src is None:
                action.status = "error"
                action.detail = "missing src"
                return
            if not src.exists():
                action.status = "skipped"
                action.detail = "source missing"
                return
            src.unlink()
            action.status = "done"
            return

        action.status = "error"
        action.detail = f"unsupported action kind: {action.kind}"
    except Exception as e:  # pragma: no cover - defensive
        action.status = "error"
        action.detail = f"{type(e).__name__}: {e}"


def _write_manifest(manifest_path: Path, payload: dict):
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def run_optimize_layout(
    input_path: str | Path,
    *,
    apply: bool = False,
    prune_legacy: bool = False,
    manifest_path: str | Path | None = None,
) -> dict:
    """Plan/apply one-time layout optimization for historical outputs."""
    root = Path(input_path)
    if not root.exists():
        raise FileNotFoundError(f"Input path not found: {root}")
    if not root.is_dir():
        raise ValueError(f"Expected directory input, got: {root}")

    actions, scanned_dirs = _plan_actions(root, prune_legacy=prune_legacy)

    if apply:
        for action in actions:
            _execute_action(action)

    summary = {
        "schema": MANIFEST_SCHEMA,
        "generated_at": datetime.now().isoformat(),
        "root": str(root),
        "apply": apply,
        "prune_legacy": prune_legacy,
        "directories_scanned": len(scanned_dirs),
        "directories_with_actions": len({str(Path(a.src).parent) for a in actions if a.src}),
        "planned_actions": len(actions),
        "executed_done": sum(1 for a in actions if a.status == "done"),
        "executed_skipped": sum(1 for a in actions if a.status == "skipped"),
        "conflicts": sum(1 for a in actions if a.status == "conflict"),
        "errors": sum(1 for a in actions if a.status == "error"),
    }

    manifest = {
        "summary": summary,
        "actions": [asdict(a) for a in actions],
    }

    out_manifest = Path(manifest_path) if manifest_path else root / MANIFEST_FILENAME
    _write_manifest(out_manifest, manifest)
    summary["manifest_path"] = str(out_manifest)
    return summary


def format_optimize_layout_summary(summary: dict) -> str:
    """Render a concise human-readable summary."""
    lines = [
        f"Layout optimization | root={summary.get('root')}",
        f"  mode: {'APPLY' if summary.get('apply') else 'DRY-RUN'}",
        f"  prune_legacy: {bool(summary.get('prune_legacy'))}",
        f"  scanned_dirs: {summary.get('directories_scanned', 0)}",
        f"  dirs_with_actions: {summary.get('directories_with_actions', 0)}",
        f"  planned_actions: {summary.get('planned_actions', 0)}",
    ]
    if summary.get("apply"):
        lines.extend([
            f"  done/skipped: {summary.get('executed_done', 0)}/{summary.get('executed_skipped', 0)}",
            f"  conflicts/errors: {summary.get('conflicts', 0)}/{summary.get('errors', 0)}",
        ])
    else:
        lines.append(
            f"  conflicts (planned): {summary.get('conflicts', 0)}"
        )
    lines.append(f"  manifest: {summary.get('manifest_path')}")
    return "\n".join(lines)
