"""Artifact naming helpers.

Canonical names are task-explicit (labeling/scoring) and we also keep
legacy aliases for backward compatibility.
"""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Iterable

# Shared output directory for runtime-generated dashboards
DASHBOARDS_DIRNAME = "dashboards"

# Pass 1 (labeling)
PASS1_STATS_FILE = "stats_labeling.json"
PASS1_STATS_FILE_LEGACY = "stats.json"
PASS1_SUMMARY_STATS_FILE = "summary_stats_labeling.json"
PASS1_SUMMARY_STATS_FILE_LEGACY = "summary_stats.json"
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
