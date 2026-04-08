"""Tests for taxonomy validation."""

from sft_label.validate import run_validation, ValidationReport
from sft_label._resources import load_taxonomy_yaml, load_all_tag_yamls
from sft_label.pipeline import (
    validate_tags,
    check_consistency,
    compute_stats,
)
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
        # Expected warnings: 'make' alias conflict + modify aliases overlapping
        # with existing tags (refactor→code-refactoring, optimize→code-optimization,
        # upgrade→migration) + long description
        assert len(report.warnings) == 5
        warning_text = "\n".join(report.warnings)
        assert "make" in warning_text
        assert "refactor" in warning_text
        assert "optimize" in warning_text

    def test_tag_count_matches(self):
        report = run_validation()
        assert report.stats["Total tags"] == 242


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

    def test_agentic_aliases_tool_names_to_capabilities(self):
        result = {"concept": [], "agentic": ["grep", "read", "mobile_list_apps", "run-script"],
                  "constraint": [], "context": "repository",
                  "confidence": {}, "unmapped": []}
        cleaned, issues = validate_tags(result, "call2")
        assert "static-analysis" in cleaned["agentic"]
        assert "file-operations" in cleaned["agentic"]
        assert "ui-automation" in cleaned["agentic"]
        assert "code-execution" in cleaned["agentic"]
        assert len(issues) == 0

    def test_task_alias_test_creation(self):
        result = {"intent": "build", "language": ["python"],
                  "domain": [], "task": ["test-creation"],
                  "difficulty": "intermediate", "confidence": {}, "unmapped": []}
        cleaned, issues = validate_tags(result, "call1")
        assert "testing-task" in cleaned["task"]
        assert len(issues) == 0

    def test_task_alias_code_testing(self):
        result = {"intent": "build", "language": ["python"],
                  "domain": [], "task": ["code-testing"],
                  "difficulty": "intermediate", "confidence": {}, "unmapped": []}
        cleaned, issues = validate_tags(result, "call1")
        assert cleaned["task"] == ["testing-task"]
        assert cleaned["canonicalized"] == [{
            "source_dimension": "task",
            "source_value": "code-testing",
            "target_dimension": "task",
            "canonical_value": "testing-task",
            "reason": "alias",
        }]
        assert len(issues) == 0

    def test_none_sentinel_in_call1_multiselects_is_treated_as_empty(self):
        result = {
            "intent": "build",
            "language": ["python", "none"],
            "domain": ["none"],
            "task": ["feature-implementation", "none"],
            "difficulty": "intermediate",
            "confidence": {},
            "unmapped": [{"dimension": "domain", "value": "none"}],
        }
        cleaned, issues = validate_tags(result, "call1")
        assert cleaned["language"] == ["python"]
        assert cleaned["domain"] == []
        assert cleaned["task"] == ["feature-implementation"]
        assert cleaned["unmapped"] == []
        assert len(issues) == 0

    def test_placeholder_sentinels_are_treated_as_empty(self):
        result = {
            "intent": "unspecified",
            "language": ["python", "no programming language detected"],
            "domain": ["no specific domain detected"],
            "task": ["no specific task detected"],
            "difficulty": "unknown",
            "confidence": {},
            "unmapped": [
                {"dimension": "intent", "value": "user query not present"},
                {"dimension": "difficulty", "value": "no difficulty detected"},
            ],
        }
        cleaned, issues = validate_tags(result, "call1")
        assert cleaned["intent"] == ""
        assert cleaned["language"] == ["python"]
        assert cleaned["domain"] == []
        assert cleaned["task"] == []
        assert cleaned["difficulty"] == ""
        assert cleaned["unmapped"] == []
        assert len(issues) == 0

    def test_domain_alias_security_to_cybersecurity(self):
        result = {"intent": "build", "language": ["python"],
                  "domain": ["security"], "task": ["feature-implementation"],
                  "difficulty": "intermediate", "confidence": {}, "unmapped": []}
        cleaned, issues = validate_tags(result, "call1")
        assert "cybersecurity" in cleaned["domain"]
        assert len(issues) == 0

    def test_language_alias_bash_to_shell(self):
        result = {"intent": "build", "language": ["bash", "python"],
                  "domain": [], "task": ["feature-implementation"],
                  "difficulty": "intermediate", "confidence": {}, "unmapped": []}
        cleaned, issues = validate_tags(result, "call1")
        assert "shell" in cleaned["language"]
        assert "bash" not in cleaned["language"]
        assert len(issues) == 0

    def test_domain_alias_ml_to_machine_learning(self):
        result = {"intent": "build", "language": ["python"],
                  "domain": ["ml"], "task": ["feature-implementation"],
                  "difficulty": "intermediate", "confidence": {}, "unmapped": []}
        cleaned, issues = validate_tags(result, "call1")
        assert cleaned["domain"] == ["machine-learning"]
        assert len(issues) == 0

    def test_difficulty_aliases_common_llm_levels(self):
        result = {"intent": "build", "language": ["python"],
                  "domain": [], "task": ["feature-implementation"],
                  "difficulty": "medium", "confidence": {}, "unmapped": []}
        cleaned, issues = validate_tags(result, "call1")
        assert cleaned["difficulty"] == "intermediate"
        assert len(issues) == 0

        result["difficulty"] = "hard"
        cleaned, issues = validate_tags(result, "call1")
        assert cleaned["difficulty"] == "advanced"
        assert len(issues) == 0

        result["difficulty"] = "easy"
        cleaned, issues = validate_tags(result, "call1")
        assert cleaned["difficulty"] == "beginner"
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

    def test_single_select_invalid_container_type(self):
        result = {
            "intent": ["build"],
            "language": ["python"],
            "domain": [],
            "task": ["feature-implementation"],
            "difficulty": {"level": "intermediate"},
            "confidence": {},
            "unmapped": [],
        }
        cleaned, issues = validate_tags(result, "call1")
        assert cleaned["intent"] == ""
        assert cleaned["difficulty"] == ""
        assert any("intent: invalid type" in issue for issue in issues)
        assert any("difficulty: invalid type" in issue for issue in issues)

    def test_confidence_invalid_shape_and_values(self):
        result = {
            "concept": ["algorithms"],
            "agentic": [],
            "constraint": [],
            "context": "snippet",
            "confidence": {
                "concept": "0.9",
                "agentic": 1.2,
                "context": 0.8,
                "extra": 0.5,
            },
            "unmapped": [],
        }
        cleaned, issues = validate_tags(result, "call2")
        assert cleaned["confidence"] == {"context": 0.8}
        assert any("confidence.concept: invalid type" in issue for issue in issues)
        assert any("confidence.agentic: out of range" in issue for issue in issues)


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


class TestCrossDimensionRescue:
    """Verify valid tags are rescued instead of counted as unmapped."""

    def test_call1_rescues_language_from_wrong_dimension(self):
        result = {"intent": "build", "language": [],
                  "domain": ["go"], "task": ["feature-implementation"],
                  "difficulty": "intermediate", "confidence": {}, "unmapped": []}
        cleaned, issues = validate_tags(result, "call1")
        assert cleaned["domain"] == []
        assert cleaned["language"] == ["go"]
        assert cleaned["unmapped"] == []
        assert len(issues) == 0

    def test_call2_rescues_agentic_from_concept(self):
        result = {"concept": ["file-operations"], "agentic": [],
                  "constraint": [], "context": "repository",
                  "confidence": {}, "unmapped": []}
        cleaned, issues = validate_tags(result, "call2")
        assert cleaned["concept"] == []
        assert cleaned["agentic"] == ["file-operations"]
        assert cleaned["unmapped"] == []
        assert len(issues) == 0

    def test_call2_rescues_refactoring_into_task(self):
        result = {"concept": ["refactoring"], "agentic": [],
                  "constraint": [], "context": "repository",
                  "confidence": {}, "unmapped": []}
        cleaned, issues = validate_tags(result, "call2")
        assert cleaned["concept"] == []
        assert cleaned["unmapped"] == []
        assert cleaned["canonicalized"] == [{
            "source_dimension": "concept",
            "source_value": "refactoring",
            "target_dimension": "task",
            "canonical_value": "code-refactoring",
            "reason": "cross_dimension_rescue",
        }]
        assert len(issues) == 0

    def test_raw_unmapped_string_rescues_into_current_dimension(self):
        result = {"concept": [], "agentic": [],
                  "constraint": [], "context": "repository",
                  "confidence": {}, "unmapped": ["planning"]}
        cleaned, issues = validate_tags(result, "call2")
        assert cleaned["agentic"] == ["planning"]
        assert cleaned["unmapped"] == []
        assert len(issues) == 0

    def test_raw_unmapped_same_dimension_alias_resolves(self):
        result = {"intent": "debug", "language": ["python"],
                  "domain": [], "task": [],
                  "difficulty": "intermediate", "confidence": {},
                  "unmapped": [{"dimension": "task", "value": "code-fixing"}]}
        cleaned, issues = validate_tags(result, "call1")
        assert cleaned["task"] == ["bug-fixing"]
        assert cleaned["unmapped"] == []
        assert len(issues) == 0


class TestModifyConsistency:
    """Verify consistency rules for the modify intent tag."""

    def test_modify_with_refactoring_passes(self):
        labels = {"intent": "modify", "language": ["python"], "domain": [],
                  "task": ["code-refactoring"], "difficulty": "intermediate",
                  "concept": [], "agentic": [], "constraint": [], "context": "single-file"}
        warnings = check_consistency(labels)
        assert not any("modify" in w.lower() for w in warnings)

    def test_modify_without_relevant_task_warns(self):
        labels = {"intent": "modify", "language": ["python"], "domain": [],
                  "task": ["code-explanation"], "difficulty": "intermediate",
                  "concept": [], "agentic": [], "constraint": [], "context": "single-file"}
        warnings = check_consistency(labels)
        assert any("modify" in w.lower() for w in warnings)

    def test_modify_with_empty_task_no_warning(self):
        labels = {"intent": "modify", "language": ["python"], "domain": [],
                  "task": [], "difficulty": "intermediate",
                  "concept": [], "agentic": [], "constraint": [], "context": "single-file"}
        warnings = check_consistency(labels)
        assert not any("modify" in w.lower() for w in warnings)


class TestCanonicalizationStats:
    def test_compute_stats_counts_canonicalizations(self):
        labels = [{
            "intent": "modify",
            "language": ["python"],
            "domain": [],
            "task": ["testing-task", "code-refactoring"],
            "difficulty": "intermediate",
            "concept": [],
            "agentic": [],
            "constraint": [],
            "context": "repository",
            "confidence": {},
            "unmapped": [],
            "canonicalized": [
                {
                    "source_dimension": "task",
                    "source_value": "code-testing",
                    "target_dimension": "task",
                    "canonical_value": "testing-task",
                    "reason": "alias",
                },
                {
                    "source_dimension": "concept",
                    "source_value": "refactoring",
                    "target_dimension": "task",
                    "canonical_value": "code-refactoring",
                    "reason": "cross_dimension_rescue",
                },
            ],
        }]
        monitors = [{
            "llm_calls": 2,
            "total_prompt_tokens": 10,
            "total_completion_tokens": 5,
            "arbitrated": False,
            "validation_issues": [],
            "consistency_warnings": [],
            "low_confidence_dims": [],
        }]

        stats = compute_stats(monitors, labels)

        assert stats["canonicalization_total_count"] == 2
        assert stats["canonicalization_unique_count"] == 2
        assert stats["canonicalization_counts"] == {
            "concept:refactoring->task:code-refactoring": 1,
            "task:code-testing->task:testing-task": 1,
        }
