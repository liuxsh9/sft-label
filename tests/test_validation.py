"""Tests for taxonomy validation."""

from sft_label.validate import run_validation, ValidationReport
from sft_label._resources import load_taxonomy_yaml, load_all_tag_yamls


class TestResourceLoading:
    def test_load_taxonomy(self):
        taxonomy = load_taxonomy_yaml()
        assert "version" in taxonomy
        assert "categories" in taxonomy
        assert len(taxonomy["categories"]) == 9

    def test_load_tags(self):
        tags = load_all_tag_yamls()
        assert len(tags) > 200  # Should be ~221 tags
        # Check basic structure
        for tag in tags[:5]:
            assert "id" in tag
            assert "name" in tag
            assert "category" in tag


class TestValidation:
    def test_full_validation_passes(self):
        report = run_validation()
        assert not report.has_errors(), f"Validation errors: {report.errors}"
        # Expected: 1 warning about 'make' alias
        assert len(report.warnings) == 1
        assert "make" in report.warnings[0]

    def test_tag_count_matches(self):
        report = run_validation()
        assert report.stats["Total tags"] == 221


class TestValidationReport:
    def test_empty_report(self):
        report = ValidationReport()
        assert not report.has_errors()
        assert len(report.errors) == 0
        assert len(report.warnings) == 0

    def test_error_tracking(self):
        report = ValidationReport()
        report.add_error("test error")
        assert report.has_errors()
        assert len(report.errors) == 1

    def test_warning_tracking(self):
        report = ValidationReport()
        report.add_warning("test warning")
        assert not report.has_errors()
        assert len(report.warnings) == 1
