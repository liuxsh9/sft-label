"""Helpers for mirrored inline run directory layout."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path


META_LABEL_DATA_DIRNAME = "meta_label_data"
FILE_ARTIFACTS_DIRNAME = "files"
CACHE_DIRNAME = "cache"


def resolve_run_root(input_path, output: str | Path | None = None,
                     timestamp: str | None = None,
                     base_dir: str | Path | None = None) -> Path:
    """Resolve the root directory for a labeled inline run."""
    if output is not None:
        return Path(output).resolve()

    input_path = Path(input_path)
    run_ts = timestamp or datetime.now().strftime("%Y%m%d_%H%M%S")
    name = input_path.name if input_path.is_dir() else input_path.stem
    root_parent = Path(base_dir).resolve() if base_dir is not None else input_path.parent
    return root_parent / f"{name}_labeled_{run_ts}"


@dataclass(frozen=True)
class InlineRunLayout:
    """Resolved layout for a mirrored inline run."""

    input_path: Path
    run_root: Path

    @classmethod
    def from_paths(cls, input_path, run_root) -> "InlineRunLayout":
        return cls(Path(input_path).resolve(), Path(run_root).resolve())

    @property
    def dataset_root_name(self) -> str:
        return self.input_path.name if self.input_path.is_dir() else self.input_path.stem

    @property
    def dataset_root(self) -> Path:
        return self.run_root / self.dataset_root_name

    @property
    def meta_root(self) -> Path:
        return self.run_root / META_LABEL_DATA_DIRNAME

    @property
    def file_meta_root(self) -> Path:
        return self.meta_root / FILE_ARTIFACTS_DIRNAME

    @property
    def cache_root(self) -> Path:
        return self.meta_root / CACHE_DIRNAME

    def relative_source_path(self, source_path) -> Path:
        """Return the source path relative to the configured input root."""
        source_path = Path(source_path).resolve()
        if self.input_path.is_dir():
            return source_path.relative_to(self.input_path)
        if source_path != self.input_path:
            raise ValueError(
                f"Expected source path {self.input_path}, got {source_path}"
            )
        return Path(source_path.name)

    def mirrored_dataset_path(self, source_path) -> Path:
        """Return the mirrored output path for a source file."""
        return self.dataset_root / self.relative_source_path(source_path)

    def file_artifact_dir(self, source_path) -> Path:
        """Return the per-source artifact directory under meta_label_data/files."""
        rel = self.relative_source_path(source_path)
        return self.file_meta_root / rel.with_suffix("")

    def file_artifact_path(self, source_path, filename: str) -> Path:
        """Return the path to a per-source artifact."""
        return self.file_artifact_dir(source_path) / filename

    def run_artifact_path(self, *parts: str) -> Path:
        """Return a run-scoped artifact path under meta_label_data."""
        return self.meta_root.joinpath(*parts)

    def cache_path(self, *parts: str) -> Path:
        """Return a cache path under meta_label_data/cache."""
        return self.cache_root.joinpath(*parts)

    def dashboard_path(self, filename: str) -> Path:
        """Return a run-root dashboard path."""
        return self.run_root / filename
