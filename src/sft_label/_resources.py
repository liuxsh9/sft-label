"""
Resource loading utilities for embedded taxonomy data.

Uses importlib.resources to load YAML files from the package,
so taxonomy works correctly when installed as a package.
"""

import sys
from pathlib import Path

import yaml


def _get_taxonomy_dir() -> Path:
    """Get the path to the taxonomy directory within the package."""
    if sys.version_info >= (3, 9):
        from importlib.resources import files
        return Path(str(files("sft_label.taxonomy")))
    else:
        # Fallback for Python 3.9
        return Path(__file__).parent / "taxonomy"


def _get_tags_dir() -> Path:
    """Get the path to the tags directory within the package."""
    if sys.version_info >= (3, 9):
        from importlib.resources import files
        return Path(str(files("sft_label.taxonomy"))) / "tags"
    else:
        return Path(__file__).parent / "taxonomy" / "tags"


def load_taxonomy_yaml() -> dict:
    """Load the taxonomy.yaml schema definition."""
    taxonomy_path = _get_taxonomy_dir() / "taxonomy.yaml"
    with open(taxonomy_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_all_tag_yamls() -> list:
    """Load all tag definition YAML files from taxonomy/tags/."""
    tags_dir = _get_tags_dir()
    all_tags = []
    for yaml_file in sorted(tags_dir.glob("*.yaml")):
        with open(yaml_file, "r", encoding="utf-8") as f:
            tags = yaml.safe_load(f)
            if tags:
                all_tags.extend(tags)
    return all_tags
