"""
Taxonomy Validation

Validates the capability taxonomy for:
- Category orthogonality
- Tag uniqueness
- Schema compliance
- Referential integrity
- Metadata quality
- Distribution balance

Adapted from scripts/validate_taxonomy.py to load taxonomy from package data
via importlib.resources.
"""

import sys
from pathlib import Path
from typing import List, Dict, Any
from collections import defaultdict

import yaml

from sft_label._resources import load_taxonomy_yaml, load_all_tag_yamls


class ValidationReport:
    """Collects validation errors, warnings, and statistics."""

    def __init__(self):
        self.errors = []
        self.warnings = []
        self.stats = {}

    def add_error(self, message: str):
        self.errors.append(message)

    def add_warning(self, message: str):
        self.warnings.append(message)

    def add_stat(self, key: str, value: Any):
        self.stats[key] = value

    def has_errors(self) -> bool:
        return len(self.errors) > 0

    def print_report(self):
        """Print formatted validation report."""
        print("\n" + "=" * 70)
        print("TAXONOMY VALIDATION REPORT")
        print("=" * 70)

        if self.errors:
            print(f"\n❌ ERRORS ({len(self.errors)}):")
            for error in self.errors:
                print(f"  - {error}")
        else:
            print("\n✓ No errors found")

        if self.warnings:
            print(f"\n⚠️  WARNINGS ({len(self.warnings)}):")
            for warning in self.warnings:
                print(f"  - {warning}")
        else:
            print("\n✓ No warnings")

        if self.stats:
            print(f"\n📊 STATISTICS:")
            for key, value in self.stats.items():
                print(f"  {key}: {value}")

        print("\n" + "=" * 70)


def validate_category_orthogonality(tags: List[Dict[str, Any]], report: ValidationReport):
    """Ensure no tag appears in multiple categories."""
    print("Validating category orthogonality...")

    tag_id_to_categories = defaultdict(list)

    for tag in tags:
        tag_id = tag.get("id")
        category = tag.get("category")
        if tag_id and category:
            tag_id_to_categories[tag_id].append(category)

    for tag_id, categories in tag_id_to_categories.items():
        if len(categories) > 1:
            report.add_error(f"Tag '{tag_id}' appears in multiple categories: {categories}")


def validate_tag_uniqueness(tags: List[Dict[str, Any]], report: ValidationReport):
    """Ensure tag IDs, aliases, and names are unique."""
    print("Validating tag uniqueness...")

    # Check ID uniqueness
    id_counts = defaultdict(int)
    for tag in tags:
        tag_id = tag.get("id")
        if tag_id:
            id_counts[tag_id] += 1

    for tag_id, count in id_counts.items():
        if count > 1:
            report.add_error(f"Duplicate tag ID: '{tag_id}' appears {count} times")

    # Check alias conflicts across different tags
    alias_to_tags = defaultdict(set)
    for tag in tags:
        tag_id = tag.get("id")
        for alias in tag.get("aliases", []):
            alias_to_tags[alias].add(tag_id)

    for alias, tag_ids in alias_to_tags.items():
        if len(tag_ids) > 1:
            report.add_warning(f"Alias '{alias}' used by multiple tags: {sorted(tag_ids)}")

    # Check name uniqueness within category
    category_names = defaultdict(list)
    for tag in tags:
        category = tag.get("category")
        name = tag.get("name")
        if category and name:
            category_names[category].append(name)

    for category, names in category_names.items():
        name_counts = defaultdict(int)
        for name in names:
            name_counts[name] += 1

        for name, count in name_counts.items():
            if count > 1:
                report.add_error(f"Duplicate name '{name}' in category '{category}' ({count} times)")


def validate_schema_compliance(tags: List[Dict[str, Any]], taxonomy: Dict[str, Any], report: ValidationReport):
    """Validate that all tags comply with schema requirements."""
    print("Validating schema compliance...")

    required_fields = ["id", "name", "category"]
    # Language metadata controlled vocabularies
    valid_paradigms = [
        "imperative", "functional", "object-oriented", "declarative",
        "logic", "procedural", "concurrent", "event-driven"
    ]
    valid_typings = ["static", "dynamic", "gradual", "duck", "strong-static", "weak-dynamic"]
    valid_runtimes = ["compiled", "interpreted", "jit", "transpiled", "hybrid"]
    valid_use_cases = [
        "web", "systems", "data-science", "mobile", "embedded", "scripting",
        "devops", "scientific", "game-dev", "blockchain", "markup", "config", "build"
    ]

    # Get hierarchical categories
    hierarchical_categories = set()
    for cat in taxonomy.get("categories", []):
        if cat.get("hierarchical"):
            hierarchical_categories.add(cat["name"])

    for tag in tags:
        tag_id = tag.get("id", "unknown")

        # Check required fields
        for field in required_fields:
            if field not in tag:
                report.add_error(f"Tag '{tag_id}' missing required field: {field}")

        # Check Language metadata fields
        category = tag.get("category")
        if category == "Language":
            # Validate paradigm (optional, multi-value)
            if "paradigm" in tag:
                paradigms = tag["paradigm"]
                if not isinstance(paradigms, list):
                    report.add_error(f"Language tag '{tag_id}' paradigm must be a list")
                else:
                    for p in paradigms:
                        if p not in valid_paradigms:
                            report.add_error(f"Language tag '{tag_id}' has invalid paradigm: {p}")

            # Validate typing (optional, single value)
            if "typing" in tag:
                typing = tag["typing"]
                if typing not in valid_typings:
                    report.add_error(f"Language tag '{tag_id}' has invalid typing: {typing}")

            # Validate runtime (optional, single value)
            if "runtime" in tag:
                runtime = tag["runtime"]
                if runtime not in valid_runtimes:
                    report.add_error(f"Language tag '{tag_id}' has invalid runtime: {runtime}")

            # Validate use_cases (optional, multi-value)
            if "use_cases" in tag:
                use_cases = tag["use_cases"]
                if not isinstance(use_cases, list):
                    report.add_error(f"Language tag '{tag_id}' use_cases must be a list")
                else:
                    for uc in use_cases:
                        if uc not in valid_use_cases:
                            report.add_error(f"Language tag '{tag_id}' has invalid use_case: {uc}")

        # Check hierarchical categories have subcategory
        if category in hierarchical_categories:
            if "subcategory" not in tag:
                report.add_error(f"Tag '{tag_id}' in hierarchical category '{category}' missing subcategory")


def validate_referential_integrity(tags: List[Dict[str, Any]], taxonomy: Dict[str, Any], report: ValidationReport):
    """Validate all references between taxonomy elements."""
    print("Validating referential integrity...")

    # Build valid category and subcategory sets
    valid_categories = set()
    valid_subcategories = defaultdict(set)

    for cat in taxonomy.get("categories", []):
        cat_name = cat["name"]
        valid_categories.add(cat_name)

        for subcat in cat.get("subcategories", []):
            valid_subcategories[cat_name].add(subcat["name"])

    # Build set of all tag IDs
    all_tag_ids = {tag.get("id") for tag in tags if tag.get("id")}

    # Build set of all language tag IDs
    language_tag_ids = {tag.get("id") for tag in tags if tag.get("category") == "Language"}

    for tag in tags:
        tag_id = tag.get("id", "unknown")

        # Validate category exists
        category = tag.get("category")
        if category and category not in valid_categories:
            report.add_error(f"Tag '{tag_id}' has invalid category: {category}")

        # Validate subcategory exists
        if "subcategory" in tag:
            subcategory = tag["subcategory"]
            if category and subcategory not in valid_subcategories.get(category, set()):
                report.add_error(f"Tag '{tag_id}' has invalid subcategory '{subcategory}' for category '{category}'")

        # Validate related_tags exist
        for related_id in tag.get("related_tags", []):
            if related_id not in all_tag_ids:
                report.add_error(f"Tag '{tag_id}' references non-existent related tag: {related_id}")

        # Validate language_scope references valid languages
        for lang in tag.get("language_scope", []):
            if lang not in language_tag_ids:
                report.add_warning(f"Tag '{tag_id}' references unknown language in language_scope: {lang}")


def validate_metadata_quality(tags: List[Dict[str, Any]], report: ValidationReport):
    """Check metadata quality standards."""
    print("Validating metadata quality...")

    for tag in tags:
        tag_id = tag.get("id", "unknown")

        # Check description length
        if "description" in tag:
            desc_len = len(tag["description"])
            if desc_len > 200:
                report.add_warning(f"Tag '{tag_id}' has long description ({desc_len} chars)")

        # Check aliases are lowercase
        for alias in tag.get("aliases", []):
            if alias != alias.lower():
                report.add_warning(f"Tag '{tag_id}' has non-lowercase alias: {alias}")


def validate_distribution(tags: List[Dict[str, Any]], report: ValidationReport):
    """Check tag distribution across categories and subcategories."""
    print("Validating distribution...")

    # Count tags per category
    category_counts = defaultdict(int)
    subcategory_counts = defaultdict(lambda: defaultdict(int))

    for tag in tags:
        category = tag.get("category")
        if category:
            category_counts[category] += 1

            subcategory = tag.get("subcategory")
            if subcategory:
                subcategory_counts[category][subcategory] += 1

    # Check for empty categories
    for category, count in category_counts.items():
        if count == 0:
            report.add_error(f"Category '{category}' has no tags")

    # Check for imbalanced subcategories
    for category, subcats in subcategory_counts.items():
        total = sum(subcats.values())
        for subcat, count in subcats.items():
            percentage = (count / total) * 100 if total > 0 else 0
            if percentage > 80:
                report.add_warning(f"Subcategory '{category}/{subcat}' has {percentage:.1f}% of tags (imbalanced)")

    # Add statistics
    report.add_stat("Total tags", len(tags))
    for category, count in sorted(category_counts.items()):
        report.add_stat(f"  {category}", count)


def run_validation(taxonomy_path=None, tags_dir=None):
    """Run all validations and return the report.

    If paths are None, loads from embedded package data.
    """
    if taxonomy_path and tags_dir:
        # External paths mode
        with open(taxonomy_path, 'r') as f:
            taxonomy = yaml.safe_load(f)
        all_tags = []
        for yaml_file in Path(tags_dir).glob("*.yaml"):
            with open(yaml_file, 'r') as f:
                tags = yaml.safe_load(f)
                if tags:
                    all_tags.extend(tags)
    else:
        # Package data mode
        taxonomy = load_taxonomy_yaml()
        all_tags = load_all_tag_yamls()

    print(f"Loaded {len(all_tags)} tags\n")

    report = ValidationReport()

    validate_category_orthogonality(all_tags, report)
    validate_tag_uniqueness(all_tags, report)
    validate_schema_compliance(all_tags, taxonomy, report)
    validate_referential_integrity(all_tags, taxonomy, report)
    validate_metadata_quality(all_tags, report)
    validate_distribution(all_tags, report)

    return report


def main():
    """CLI entry point for taxonomy validation."""
    report = run_validation()
    report.print_report()

    if report.has_errors():
        print("\n❌ Validation failed with errors")
        sys.exit(1)
    else:
        print("\n✅ Validation passed")
        sys.exit(0)


if __name__ == '__main__':
    main()
