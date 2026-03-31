"""Artifact naming helpers.

Canonical names are task-explicit (labeling/scoring) and we also keep
legacy aliases for backward compatibility.
"""

from __future__ import annotations

import os
import shutil
from pathlib import Path
from typing import Iterable

from sft_label.fs_artifacts import is_ignored_fs_artifact

# Shared output directory for runtime-generated dashboards
DASHBOARDS_DIRNAME = "dashboards"
DASHBOARD_RUNTIME_DIRNAME = "_dashboard_static"
DASHBOARD_RUNTIME_VERSION = "v1"
DASHBOARD_STATIC_BASE_URL_ENV = "SFT_LABEL_DASHBOARD_STATIC_BASE_URL"

# Pass 1 (labeling)
PASS1_STATS_FILE = "stats_labeling.json"
PASS1_STATS_FILE_LEGACY = "stats.json"
PASS1_SUMMARY_STATS_FILE = "summary_stats_labeling.json"
PASS1_SUMMARY_STATS_FILE_LEGACY = "summary_stats.json"
PASS1_CONVERSATION_STATS_FILE = "conversation_stats_labeling.json"
PASS1_DASHBOARD_FILE = "dashboard_labeling.html"
PASS1_DASHBOARD_FILE_LEGACY = "dashboard.html"

# Pass 2 (value scoring)
PASS2_STATS_FILE = "stats_scoring.json"
PASS2_STATS_FILE_LEGACY = "stats_value.json"
PASS2_SUMMARY_STATS_FILE = "summary_stats_scoring.json"
PASS2_SUMMARY_STATS_FILE_LEGACY = "summary_stats_value.json"
PASS2_DASHBOARD_FILE = "dashboard_scoring.html"
PASS2_DASHBOARD_FILE_LEGACY = "dashboard_value.html"


def pass1_stats_filename(suffix: str = "") -> str:
    return f"stats_labeling{suffix}.json"


def pass1_stats_legacy_filename(suffix: str = "") -> str:
    return f"stats{suffix}.json"


def pass1_dashboard_filename(suffix: str = "") -> str:
    return f"dashboard_labeling{suffix}.html"


def pass1_dashboard_legacy_filename(suffix: str = "") -> str:
    return f"dashboard{suffix}.html"


def pass1_global_dashboard_filename(input_name: str) -> str:
    return f"dashboard_labeling_{input_name}.html"


def pass1_global_dashboard_legacy_filename(input_name: str) -> str:
    return f"dashboard_{input_name}.html"


def pass2_global_dashboard_filename(input_name: str) -> str:
    return f"dashboard_scoring_{input_name}.html"


def pass2_global_dashboard_legacy_filename(input_name: str) -> str:
    return f"dashboard_value_{input_name}.html"


def dashboard_relpath(filename: str) -> Path:
    """Return the relative runtime location for a dashboard HTML file."""
    return Path(DASHBOARDS_DIRNAME) / filename


def dashboard_data_dirname(filename: str | Path) -> str:
    """Return the sidecar data directory name for a dashboard HTML file."""
    name = Path(filename).name
    stem = Path(name).stem
    return f"{stem}.data"


def runtime_static_base_url() -> str | None:
    """Return the configured shared static dashboard asset base URL, if any."""
    value = os.getenv(DASHBOARD_STATIC_BASE_URL_ENV, "").strip()
    return value or None


def resolve_dashboard_output(base_dir: Path | str, output_file: str | Path) -> Path:
    """Resolve a dashboard output path under the runtime dashboards directory.

    Relative filenames are written to ``<base_dir>/dashboards/<filename>``.
    Absolute paths and explicit relative subpaths are preserved.
    """
    base_dir = Path(base_dir)
    output_path = Path(output_file)
    if output_path.is_absolute():
        return output_path
    if len(output_path.parts) > 1:
        return base_dir / output_path
    return base_dir / dashboard_relpath(output_path.name)


def sync_legacy_aliases(src_path: Path, alias_names: Iterable[str]) -> None:
    """Copy src_path to each alias file name in the same directory."""
    if not src_path.exists():
        return
    for alias_name in alias_names:
        if not alias_name:
            continue
        alias_path = src_path.with_name(alias_name)
        if alias_path == src_path:
            continue
        shutil.copyfile(src_path, alias_path)


def find_first_existing(base_dir: Path, candidate_names: Iterable[str]) -> Path | None:
    """Return the first candidate file that exists under base_dir."""
    for name in candidate_names:
        path = base_dir / name
        if path.exists():
            return path
    return None


def classify_dashboard_kind(filename: str | Path) -> str | None:
    """Classify a dashboard filename as labeling/scoring for retention logic."""
    name = Path(filename).name.lower()
    if name.startswith("dashboard_scoring") or name.startswith("dashboard_value"):
        return "scoring"
    if name.startswith("dashboard_labeling") or name == PASS1_DASHBOARD_FILE_LEGACY or (
        name.startswith("dashboard_") and not name.startswith("dashboard_scoring")
    ):
        return "labeling"
    return None


def remove_dashboard_bundle(html_path: Path | str) -> None:
    """Delete one dashboard HTML file and its sidecar data directory if present."""
    def _safe_unlink(path: Path) -> None:
        try:
            path.unlink()
        except FileNotFoundError:
            return

    def _ignore_rmtree_error(_func, path, exc_info) -> None:
        exc = exc_info[1]
        if isinstance(exc, FileNotFoundError):
            return
        if is_ignored_fs_artifact(path):
            return
        raise exc

    html_path = Path(html_path)
    data_dir = html_path.with_name(dashboard_data_dirname(html_path.name))
    html_sidecar = html_path.with_name(f"._{html_path.name}")
    data_dir_sidecar = data_dir.with_name(f"._{data_dir.name}")
    if html_path.exists():
        _safe_unlink(html_path)
    if html_sidecar.exists():
        _safe_unlink(html_sidecar)
    if data_dir.is_dir():
        shutil.rmtree(data_dir, onerror=_ignore_rmtree_error)
    if data_dir_sidecar.exists():
        _safe_unlink(data_dir_sidecar)


def prune_dashboard_bundles(
    root_dir: Path | str,
    *,
    keep_paths: Iterable[Path | str] = (),
    kind: str | None = None,
    recursive: bool = False,
) -> list[Path]:
    """Delete dashboard bundles under ``root_dir`` that are not explicitly kept.

    When ``recursive`` is true, scans every ``dashboards/*.html`` bundle in the run.
    This is used to retain only canonical run-level dashboards for directory runs.
    """
    root_dir = Path(root_dir)
    keep = {Path(path).resolve() for path in keep_paths if path is not None}
    pattern = f"**/{DASHBOARDS_DIRNAME}/*.html" if recursive else f"{DASHBOARDS_DIRNAME}/*.html"
    removed: list[Path] = []
    for html_path in sorted(root_dir.glob(pattern)):
        if html_path.resolve() in keep:
            continue
        if kind is not None and classify_dashboard_kind(html_path.name) != kind:
            continue
        remove_dashboard_bundle(html_path)
        removed.append(html_path)
    return removed
