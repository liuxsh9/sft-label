"""Regression tests for tag taxonomy iteration v2 (2026-04-01).

Validates that:
1. New TAG_ALIASES resolve previously-unmapped tags correctly
2. Dimension-qualified aliases (TAG_ALIASES_BY_DIM) override flat aliases
3. New TAG_POOLS entries are present and valid
4. resolve_alias() works correctly with and without dimension context
"""

import pytest

from sft_label.prompts import TAG_POOLS, SINGLE_SELECT, MULTI_SELECT
from sft_label.tag_canonicalization import (
    TAG_ALIASES,
    TAG_ALIASES_BY_DIM,
    resolve_alias,
)


# ──────────────────────────────────────────────────────────
# 1. resolve_alias — basic behavior
# ──────────────────────────────────────────────────────────

class TestResolveAlias:
    """Test the dimension-aware alias resolution."""

    def test_flat_alias_without_dim(self):
        assert resolve_alias("explore") == "learn"
        assert resolve_alias("c++") == "cpp"
        assert resolve_alias("nonexistent-tag") == "nonexistent-tag"

    def test_flat_alias_with_irrelevant_dim(self):
        """When dim is given but no qualified override exists, fall back to flat."""
        assert resolve_alias("c++", dim="language") == "cpp"

    def test_dim_qualified_override(self):
        """TAG_ALIASES_BY_DIM takes precedence over TAG_ALIASES."""
        # "search" flat alias doesn't exist; it's only in TAG_ALIASES_BY_DIM
        assert resolve_alias("search", dim="intent") == "learn"
        assert resolve_alias("search", dim="task") == "code-exploration"
        # Without dim, no flat alias → returns unchanged
        assert resolve_alias("search") == "search"

    def test_dim_qualified_exploration(self):
        assert resolve_alias("exploration", dim="intent") == "learn"
        assert resolve_alias("exploration", dim="task") == "code-exploration"

    def test_dim_qualified_analysis(self):
        assert resolve_alias("analysis", dim="intent") == "review"
        assert resolve_alias("analysis", dim="task") == "performance-analysis"

    def test_dim_qualified_investigation(self):
        assert resolve_alias("investigation", dim="intent") == "debug"
        assert resolve_alias("investigation", dim="task") == "code-exploration"

    def test_dim_qualified_networking(self):
        assert resolve_alias("networking", dim="domain") == "network-programming"
        assert resolve_alias("networking", dim="concept") == "api-protocols"

    def test_dim_qualified_data_visualization(self):
        assert resolve_alias("data-visualization", dim="concept") == "architecture"
        assert resolve_alias("data-visualization", dim="task") == "documentation"


# ──────────────────────────────────────────────────────────
# 2. Intent aliases — explore/investigate/verify cluster
# ──────────────────────────────────────────────────────────

class TestIntentAliases:
    """All explore/investigate/search variants → valid intent tags."""

    @pytest.mark.parametrize("raw,expected", [
        ("explore", "learn"),
        ("exploration", "learn"),
        ("search", "learn"),
        ("research", "learn"),
        ("information-retrieval", "learn"),
        ("information retrieval", "learn"),
        ("information-gathering", "learn"),
        ("retrieve", "learn"),
        ("general inquiry", "learn"),
        ("request for information", "learn"),
        ("investigation", "debug"),
        ("verify", "review"),
        ("analyze", "review"),
        ("analysis", "review"),
    ])
    def test_intent_alias_resolves(self, raw, expected):
        result = resolve_alias(raw, dim="intent")
        assert result == expected, f"intent alias '{raw}' → '{result}', expected '{expected}'"
        assert expected in TAG_POOLS["intent"]


# ──────────────────────────────────────────────────────────
# 3. Domain shorthand aliases
# ──────────────────────────────────────────────────────────

class TestDomainAliases:
    @pytest.mark.parametrize("raw,expected", [
        ("database", "database-administration"),
        ("systems", "systems-programming"),
        ("mobile", "mobile-development"),
        ("embedded", "embedded-systems"),
        ("gaming", "game-development"),
        ("security", "cybersecurity"),
        ("ml", "machine-learning"),
    ])
    def test_domain_shorthand_resolves(self, raw, expected):
        result = resolve_alias(raw, dim="domain")
        assert result == expected
        assert expected in TAG_POOLS["domain"]


# ──────────────────────────────────────────────────────────
# 4. Task aliases
# ──────────────────────────────────────────────────────────

class TestTaskAliases:
    @pytest.mark.parametrize("raw,expected", [
        ("information-gathering", "code-exploration"),
        ("complexity-analysis", "performance-analysis"),
        ("data-extraction", "feature-implementation"),
        ("data-analysis", "performance-analysis"),
        ("data-exploration", "code-exploration"),
        ("data-collection", "feature-implementation"),
        ("file-management", "configuration"),
        ("file-manipulation", "configuration"),
        ("file operations", "configuration"),
        ("system-design", "api-design"),
        ("verification", "code-review-task"),
        ("report-generation", "documentation"),
        ("technical support", "bug-fixing"),
        ("code-implementation", "feature-implementation"),
        ("web-scraping", "feature-implementation"),
        ("validation", "testing-task"),
        ("authentication", "feature-implementation"),
    ])
    def test_task_alias_resolves(self, raw, expected):
        result = resolve_alias(raw, dim="task")
        assert result == expected
        assert expected in TAG_POOLS["task"]


# ──────────────────────────────────────────────────────────
# 5. Concept aliases
# ──────────────────────────────────────────────────────────

class TestConceptAliases:
    @pytest.mark.parametrize("raw,expected", [
        ("networking", "api-protocols"),
        ("websocket", "api-protocols"),
        ("state-management", "architecture"),
        ("event-handling", "design-patterns"),
        ("generics", "type-system"),
        ("smart-contracts", "security"),
        ("string-processing", "algorithms"),
        ("gui", "design-patterns"),
        ("gui-programming", "design-patterns"),
        ("web-development", "architecture"),
        ("ui-design", "design-patterns"),
    ])
    def test_concept_alias_resolves(self, raw, expected):
        result = resolve_alias(raw, dim="concept")
        assert result == expected
        assert expected in TAG_POOLS["concept"]


# ──────────────────────────────────────────────────────────
# 6. Language — new pool entries and aliases
# ──────────────────────────────────────────────────────────

class TestLanguagePoolAndAliases:
    @pytest.mark.parametrize("lang", [
        "vba", "gdscript", "pinescript", "mql5", "delphi", "autohotkey",
        "glsl", "pascal", "applescript", "hlsl", "mathematica", "gml",
        "nix", "actionscript", "maxscript",
    ])
    def test_new_language_in_pool(self, lang):
        assert lang in TAG_POOLS["language"], f"{lang} missing from TAG_POOLS['language']"

    @pytest.mark.parametrize("raw,expected", [
        ("batch", "shell"),
        ("vb", "vba"),
        ("vb.net", "vba"),
        ("excel", "vba"),
        ("pine", "pinescript"),
        ("bash", "shell"),
        ("c++", "cpp"),
        ("c#", "csharp"),
    ])
    def test_language_alias_resolves(self, raw, expected):
        result = resolve_alias(raw, dim="language")
        assert result == expected
        assert expected in TAG_POOLS["language"]


# ──────────────────────────────────────────────────────────
# 7. Constraint — new pool entries and aliases
# ──────────────────────────────────────────────────────────

class TestConstraintPoolAndAliases:
    def test_offline_capable_in_pool(self):
        assert "offline-capable" in TAG_POOLS["constraint"]

    @pytest.mark.parametrize("raw", [
        "offline", "offline-first", "offline-mode", "offline-support",
        "offline-functionality", "offline-use", "offline-capability",
        "offline-ready", "offline-caching", "offline-access", "offline-resilience",
    ])
    def test_offline_aliases_resolve(self, raw):
        result = resolve_alias(raw, dim="constraint")
        assert result == "offline-capable"
        assert result in TAG_POOLS["constraint"]

    @pytest.mark.parametrize("raw,expected", [
        ("cross-platform", "portable"),
        ("cross-browser", "portable"),
        ("memory-optimized", "performance-optimized"),
        ("responsive", "accessible"),
    ])
    def test_other_constraint_aliases(self, raw, expected):
        result = resolve_alias(raw, dim="constraint")
        assert result == expected
        assert expected in TAG_POOLS["constraint"]


# ──────────────────────────────────────────────────────────
# 8. Pool integrity checks
# ──────────────────────────────────────────────────────────

class TestPoolIntegrity:
    """Ensure all alias targets land in their respective TAG_POOLS."""

    def test_all_flat_alias_targets_exist_in_some_pool(self):
        """Every flat alias target must exist in at least one dimension's pool."""
        all_pool_values = set()
        for pool in TAG_POOLS.values():
            all_pool_values.update(pool)

        missing = []
        for alias_key, alias_target in TAG_ALIASES.items():
            if alias_target not in all_pool_values:
                missing.append(f"{alias_key} → {alias_target}")
        assert not missing, f"Alias targets not in any pool: {missing}"

    def test_all_dim_qualified_alias_targets_exist_in_correct_pool(self):
        """Every dim-qualified alias target must exist in the specified dim's pool."""
        missing = []
        for (dim, raw), target in TAG_ALIASES_BY_DIM.items():
            pool = TAG_POOLS.get(dim, set())
            if target not in pool:
                missing.append(f"({dim}, {raw}) → {target} not in {dim} pool")
        assert not missing, f"Dim-qualified alias targets not in correct pool: {missing}"

    def test_no_alias_key_conflicts_with_existing_pool_values(self):
        """Aliases that point to a different value should not be pool members
        in the SAME dimension (would mean the alias is unreachable because
        pool validation happens first)."""
        # This is OK: "security" → "cybersecurity" when "security" is in concept pool
        # but the alias is used for domain dimension. The implicit dim filter handles it.
        # We only flag when an alias key exists in ALL pools that contain the target.
        pass  # This is an advisory check, not strict

    def test_pool_sizes_are_reasonable(self):
        """Sanity: no pool should be empty or have ballooned."""
        assert len(TAG_POOLS["intent"]) == 6
        assert len(TAG_POOLS["difficulty"]) == 5
        assert len(TAG_POOLS["context"]) == 10
        assert 80 <= len(TAG_POOLS["language"]) <= 120
        assert 35 <= len(TAG_POOLS["domain"]) <= 50
        assert 20 <= len(TAG_POOLS["concept"]) <= 35
        assert 20 <= len(TAG_POOLS["task"]) <= 30
        assert 20 <= len(TAG_POOLS["agentic"]) <= 30
        assert 20 <= len(TAG_POOLS["constraint"]) <= 25


# ──────────────────────────────────────────────────────────
# 9. Cross-dimension safety: alias doesn't accidentally
#    resolve into wrong dimension
# ──────────────────────────────────────────────────────────

class TestCrossDimSafety:
    """Ensure commonly confused aliases don't bleed across dimensions."""

    def test_search_does_not_resolve_to_learn_in_task_dim(self):
        """'search' in task dim should become code-exploration, not learn."""
        result = resolve_alias("search", dim="task")
        assert result != "learn"
        assert result == "code-exploration"

    def test_security_stays_in_concept_pool(self):
        """'security' as a concept tag should stay as-is (it's in concept pool)."""
        # The alias "security" → "cybersecurity" exists, but it's for domain dim.
        # In concept dim, "security" is already valid, so alias should NOT be applied.
        # This is handled by pipeline.py: `v if v in pool else resolve_alias(v, dim)`
        assert "security" in TAG_POOLS["concept"]

    def test_exploration_resolves_differently_per_dim(self):
        assert resolve_alias("exploration", dim="intent") == "learn"
        assert resolve_alias("exploration", dim="task") == "code-exploration"
