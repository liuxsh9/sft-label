"""Tests for taxonomy validation."""

from sft_label.validate import run_validation, ValidationReport
from sft_label._resources import load_taxonomy_yaml, load_all_tag_yamls
from sft_label.pipeline import validate_tags, TAG_ALIASES, CROSS_CATEGORY_CORRECTIONS
from sft_label.prompts import TAG_POOLS


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
        assert report.stats["Total tags"] == 223


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


class TestTagPools:
    """Verify new tags are present in TAG_POOLS."""

    def test_debugging_in_concept_pool(self):
        assert "debugging" in TAG_POOLS["concept"]

    def test_code_exploration_in_task_pool(self):
        assert "code-exploration" in TAG_POOLS["task"]

    def test_pool_counts(self):
        assert len(TAG_POOLS["concept"]) == 26
        assert len(TAG_POOLS["task"]) == 22


class TestAliasResolution:
    """Verify alias mapping in validate_tags()."""

    def test_language_alias_cpp(self):
        result = {"intent": "build", "language": ["c++", "python"],
                  "domain": [], "task": ["feature-implementation"],
                  "difficulty": "intermediate", "confidence": {}, "unmapped": []}
        cleaned, issues = validate_tags(result, "call1")
        assert "cpp" in cleaned["language"]
        assert "c++" not in cleaned["language"]
        assert len(issues) == 0

    def test_concept_alias_dynamic_programming(self):
        result = {"concept": ["dynamic-programming", "data-structures"],
                  "agentic": [], "constraint": [], "context": "snippet",
                  "confidence": {}, "unmapped": []}
        cleaned, issues = validate_tags(result, "call2")
        assert "algorithms" in cleaned["concept"]
        assert "dynamic-programming" not in cleaned["concept"]
        assert len(issues) == 0

    def test_agentic_alias_execute_python_code(self):
        result = {"concept": [], "agentic": ["execute_python_code", "planning"],
                  "constraint": [], "context": "single-file",
                  "confidence": {}, "unmapped": []}
        cleaned, issues = validate_tags(result, "call2")
        assert "code-execution" in cleaned["agentic"]
        assert "execute_python_code" not in cleaned["agentic"]
        assert len(issues) == 0

    def test_task_alias_test_creation(self):
        result = {"intent": "build", "language": ["python"],
                  "domain": [], "task": ["test-creation"],
                  "difficulty": "intermediate", "confidence": {}, "unmapped": []}
        cleaned, issues = validate_tags(result, "call1")
        assert "testing-task" in cleaned["task"]
        assert len(issues) == 0

    def test_domain_alias_security_to_cybersecurity(self):
        result = {"intent": "build", "language": ["python"],
                  "domain": ["security"], "task": ["feature-implementation"],
                  "difficulty": "intermediate", "confidence": {}, "unmapped": []}
        cleaned, issues = validate_tags(result, "call1")
        assert "cybersecurity" in cleaned["domain"]
        assert len(issues) == 0

    def test_concept_security_not_aliased(self):
        """concept:security is a valid tag and must NOT be aliased to cybersecurity."""
        result = {"concept": ["security"], "agentic": [], "constraint": [],
                  "context": "single-file", "confidence": {}, "unmapped": []}
        cleaned, issues = validate_tags(result, "call2")
        assert "security" in cleaned["concept"]
        assert "cybersecurity" not in cleaned["concept"]
        assert len(issues) == 0

    def test_dedup_after_alias(self):
        """Multiple aliases resolving to the same tag should deduplicate."""
        result = {"concept": ["dynamic-programming", "dp", "algorithms"],
                  "agentic": [], "constraint": [], "context": "snippet",
                  "confidence": {}, "unmapped": []}
        cleaned, issues = validate_tags(result, "call2")
        assert cleaned["concept"].count("algorithms") == 1
        assert len(issues) == 0


class TestCrossCategoryCorrection:
    """Verify misplaced tags are silently dropped from wrong dimensions."""

    def test_algorithm_dropped_from_domain(self):
        result = {"intent": "build", "language": ["python"],
                  "domain": ["algorithm", "web-backend"], "task": ["feature-implementation"],
                  "difficulty": "intermediate", "confidence": {}, "unmapped": []}
        cleaned, issues = validate_tags(result, "call1")
        assert "algorithm" not in cleaned["domain"]
        assert "web-backend" in cleaned["domain"]
        # Silently dropped, not in unmapped
        unmapped_values = [u["value"] for u in cleaned["unmapped"]]
        assert "algorithm" not in unmapped_values

    def test_documentation_dropped_from_domain(self):
        result = {"intent": "build", "language": ["python"],
                  "domain": ["documentation"], "task": ["documentation"],
                  "difficulty": "beginner", "confidence": {}, "unmapped": []}
        cleaned, issues = validate_tags(result, "call1")
        assert "documentation" not in cleaned["domain"]
        assert "documentation" in cleaned["task"]
