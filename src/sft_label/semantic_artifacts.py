"""Helpers for locating semantic clustering artifact directories."""

from __future__ import annotations

import json
from pathlib import Path

from sft_label.inline_scoring import infer_inline_scoring_target
from sft_label.run_layout import META_LABEL_DATA_DIRNAME

SEMANTIC_DIRNAME = "semantic"
SEMANTIC_MANIFEST_FILE = "semantic_cluster_manifest.json"
SEMANTIC_STATS_FILE = "semantic_cluster_stats.json"

_SEMANTIC_PREFIX_SUFFIXES = (
    "_windows.jsonl",
    "_embeddings.jsonl",
    "_semhash.jsonl",
    "_cluster_membership.jsonl",
    "_clusters.json",
    "_representatives.jsonl",
)


def manifest_output_prefix(input_dir: str | Path) -> str | None:
    """Return the semantic output prefix recorded in the manifest, if present."""
    manifest_path = Path(input_dir) / SEMANTIC_MANIFEST_FILE
    if not manifest_path.exists():
        return None
    try:
        with open(manifest_path, "r", encoding="utf-8") as f:
            manifest = json.load(f)
    except (OSError, ValueError, TypeError):
        return None
    params = manifest.get("parameters") or {}
    prefix = params.get("output_prefix")
    if isinstance(prefix, str) and prefix.strip():
        return prefix.strip()
    return None


def iter_semantic_artifact_paths(input_dir: str | Path) -> list[Path]:
    """List semantic artifact files found directly under one directory."""
    base = Path(input_dir)
    files: list[Path] = []

    for name in (SEMANTIC_MANIFEST_FILE, SEMANTIC_STATS_FILE):
        path = base / name
        if path.is_file():
            files.append(path)

    prefix = manifest_output_prefix(base)
    if prefix:
        for suffix in _SEMANTIC_PREFIX_SUFFIXES:
            path = base / f"{prefix}{suffix}"
            if path.is_file():
                files.append(path)
        return sorted({path.resolve() for path in files})

    for suffix in _SEMANTIC_PREFIX_SUFFIXES:
        files.extend(path.resolve() for path in base.glob(f"*{suffix}") if path.is_file())
    return sorted({path.resolve() for path in files})


def looks_like_semantic_dir(input_dir: str | Path) -> bool:
    """Whether a directory appears to contain semantic clustering artifacts."""
    return bool(iter_semantic_artifact_paths(input_dir))


def resolve_semantic_output_dir(
    input_path: str | Path,
    output_dir: str | Path | None = None,
    *,
    prefer_existing: bool = False,
) -> Path:
    """Resolve the semantic artifact output directory.

    Inline mirrored run roots default to ``meta_label_data/semantic`` so the
    run root stays focused on dataset/output files.
    """
    input_path = Path(input_path).resolve()
    inline_target = infer_inline_scoring_target(input_path)

    if inline_target is not None:
        semantic_dir = inline_target.layout.run_artifact_path(SEMANTIC_DIRNAME)
        if output_dir is not None:
            out_dir = Path(output_dir).resolve()
            if out_dir in {
                inline_target.layout.run_root,
                inline_target.layout.meta_root,
            }:
                return semantic_dir
            return out_dir
        if prefer_existing and not looks_like_semantic_dir(semantic_dir):
            legacy_dir = inline_target.layout.run_root
            if looks_like_semantic_dir(legacy_dir):
                return legacy_dir
        if input_path.is_file():
            return input_path.parent
        return semantic_dir

    if output_dir is not None:
        return Path(output_dir).resolve()

    return input_path if input_path.is_dir() else input_path.parent


def resolve_semantic_artifact_dir(input_dir: str | Path) -> Path:
    """Resolve where semantic artifacts live for export/inspection commands."""
    base = Path(input_dir).resolve()
    if base.is_file():
        base = base.parent

    candidates: list[Path] = []
    if looks_like_semantic_dir(base):
        candidates.append(base)

    semantic_child = base / SEMANTIC_DIRNAME
    if base.name == META_LABEL_DATA_DIRNAME and looks_like_semantic_dir(semantic_child):
        candidates.append(semantic_child)

    inline_semantic = base / META_LABEL_DATA_DIRNAME / SEMANTIC_DIRNAME
    if looks_like_semantic_dir(inline_semantic):
        candidates.append(inline_semantic)

    inline_target = infer_inline_scoring_target(base)
    if inline_target is not None:
        semantic_dir = inline_target.layout.run_artifact_path(SEMANTIC_DIRNAME)
        if looks_like_semantic_dir(semantic_dir):
            candidates.insert(0, semantic_dir.resolve())
        if looks_like_semantic_dir(inline_target.layout.run_root):
            candidates.append(inline_target.layout.run_root.resolve())

    seen: set[Path] = set()
    ordered: list[Path] = []
    for candidate in candidates:
        resolved = candidate.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        ordered.append(resolved)

    if ordered:
        return ordered[0]
    return base
