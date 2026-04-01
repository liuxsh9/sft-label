"""Canonical tag remapping helpers for Pass 1 validation."""

from __future__ import annotations


# ---------------------------------------------------------------------------
# Alias → canonical tag mapping.  Applied before pool validation so that
# common LLM outputs (underscore variants, abbreviations, sub-topic names,
# and a few high-value SWE synonyms) are silently resolved instead of
# becoming unmapped.
#
# Two tables cooperate:
#   TAG_ALIASES          – dimension-agnostic (legacy, still used as fallback)
#   TAG_ALIASES_BY_DIM   – dimension-qualified overrides; checked first when
#                          the current dimension is known.
#
# Lookup order (see resolve_alias()):
#   1. TAG_ALIASES_BY_DIM[(dim, value)]  →  if found, use it
#   2. TAG_ALIASES[value]                →  fallback
# ---------------------------------------------------------------------------

TAG_ALIASES = {
    # language
    "c++": "cpp",
    "c#": "csharp",
    "f#": "fsharp",
    "bash": "shell",
    # concept — algorithm sub-topics
    "dynamic-programming": "algorithms",
    "dp": "algorithms",
    "graph-theory": "algorithms",
    "geometry": "algorithms",
    "combinatorics": "algorithms",
    "bit-manipulation": "algorithms",
    "string-manipulation": "algorithms",
    "string-algorithms": "algorithms",
    "mathematics": "algorithms",
    "greedy": "algorithms",
    "backtracking": "algorithms",
    "divide-and-conquer": "algorithms",
    # concept/task crossovers frequently emitted by LLMs on SWE data
    "refactoring": "code-refactoring",
    "performance-optimization": "code-optimization",
    "parsing": "algorithms",
    # agentic — format variants
    "execute_python_code": "code-execution",
    "execute_python": "code-execution",
    "execute_code": "code-execution",
    "run_code": "code-execution",
    "execute_bash": "bash-execution",
    "run_bash": "bash-execution",
    "shell_execution": "bash-execution",
    # task
    "test-creation": "testing-task",
    "test creation": "testing-task",
    "code-testing": "testing-task",
    "write-tests": "testing-task",
    "issue analysis": "code-exploration",
    "fix implementation": "bug-fixing",
    "code-fixing": "bug-fixing",
    "code-correction": "bug-fixing",
    # domain
    "security": "cybersecurity",
    "ml": "machine-learning",
    "database": "database-administration",
    "systems": "systems-programming",
    "mobile": "mobile-development",
    "embedded": "embedded-systems",
    "gaming": "game-development",
    # intent — modify aliases
    "refactor": "modify",
    "optimize": "modify",
    "transform": "modify",
    # intent — explore/investigate cluster → learn (default flat alias)
    # Note: "search", "research", "exploration", "information-gathering" are also in
    # TAG_ALIASES_BY_DIM with dimension-specific targets (see below). The flat entries
    # here serve as the no-dim fallback and for any dim where "learn" is valid.
    "explore": "learn",
    "exploration": "learn",
    "research": "learn",
    "information-retrieval": "learn",
    "information retrieval": "learn",
    # "information-gathering" intentionally omitted here — it's in TAG_ALIASES_BY_DIM
    # with intent→learn and task→code-exploration. The flat alias "code-exploration"
    # (defined in the task cluster below) is the correct no-dim fallback.
    "retrieve": "learn",
    "general inquiry": "learn",
    "request for information": "learn",
    # intent — investigate/verify → debug/review
    "investigation": "debug",
    "verify": "review",
    "analyze": "review",
    "analysis": "review",
    # difficulty
    "easy": "beginner",
    "medium": "intermediate",
    "hard": "advanced",
    # language — common abbreviations
    "objc": "objective-c",
    "objective_c": "objective-c",
    "js": "javascript",
    "ts": "typescript",
    "rb": "ruby",
    "py": "python",
    # language — family consolidation
    "vb": "vba",
    "vb.net": "vba",
    "excel": "vba",
    "pine": "pinescript",
    "batch": "shell",
    "wolfram": "mathematica",
    "wolfram mathematica": "mathematica",
    # concept — additional mappings
    "regex": "algorithms",
    "regular-expressions": "algorithms",
    "async-await": "concurrency",
    "system-programming": "concurrency",
    "networking": "api-protocols",
    "websocket": "api-protocols",
    "state-management": "architecture",
    "event-handling": "design-patterns",
    "generics": "type-system",
    "smart-contracts": "security",
    "string-processing": "algorithms",
    "image-processing": "algorithms",
    "linear-algebra": "algorithms",
    "gui": "design-patterns",
    "gui-programming": "design-patterns",
    "web-development": "architecture",
    "ui-design": "design-patterns",
    "frontend-development": "architecture",
    # agentic — common tool-name emissions
    "grep": "static-analysis",
    "read": "file-operations",
    "run-script": "code-execution",
    "run_script": "code-execution",
    "mobile_list_available_devices": "ui-automation",
    "mobile-use-device": "ui-automation",
    "mobile_use_device": "ui-automation",
    "mobile-list-available-devices": "ui-automation",
    "mobile_list_apps": "ui-automation",
    "mobile_launch_app": "ui-automation",
    "mobile_list_elements_on_screen": "ui-automation",
    # constraint — offline cluster
    "offline": "offline-capable",
    "offline-first": "offline-capable",
    "offline-mode": "offline-capable",
    "offline-support": "offline-capable",
    "offline-functionality": "offline-capable",
    "offline-use": "offline-capable",
    "offline-capability": "offline-capable",
    "offline-ready": "offline-capable",
    "offline-caching": "offline-capable",
    "offline-access": "offline-capable",
    "offline-resilience": "offline-capable",
    # constraint — other consolidation
    "cross-platform": "portable",
    "cross-browser": "portable",
    "memory-optimized": "performance-optimized",
    "responsive": "accessible",
    # task — exploration/search cluster
    "information-gathering": "code-exploration",
    "data-retrieval": "code-exploration",
    "file-search": "code-exploration",
    "file-exploration": "code-exploration",
    "file exploration": "code-exploration",
    "directory-listing": "code-exploration",
    # task — implementation cluster
    "data-extraction": "feature-implementation",
    "data-collection": "feature-implementation",
    "code-implementation": "feature-implementation",
    "content-creation": "feature-implementation",
    # task — performance-analysis cluster
    "complexity-analysis": "performance-analysis",
    "disk-usage-estimation": "performance-analysis",
    # task — configuration cluster
    "file-management": "configuration",
    "file-manipulation": "configuration",
    "file operations": "configuration",
    "file-sorting": "configuration",
    "backup": "configuration",
    # task — documentation cluster
    "report-generation": "documentation",
    "formatting": "documentation",
    # task — code-review-task cluster
    "verification": "code-review-task",
    # task — bug-fixing cluster
    "technical support": "bug-fixing",
    # task — api-design cluster
    "system-design": "api-design",
    # task — other mappings
    "data-analysis": "performance-analysis",
    "data-exploration": "code-exploration",
    "data-visualization": "documentation",
    "web-scraping": "feature-implementation",
    "validation": "testing-task",
    "entity-recognition": "feature-implementation",
    "project-management": "configuration",
    "project-planning": "configuration",
    "currency-conversion": "feature-implementation",
    "file-display": "code-exploration",
    "file-comparison": "code-exploration",
    "calculation": "feature-implementation",
    "data-processing": "feature-implementation",
    "authentication": "feature-implementation",
}

# ---------------------------------------------------------------------------
# Dimension-qualified aliases — override TAG_ALIASES when the current
# dimension is known.  Key = (dimension, raw_value), value = canonical tag.
#
# Use this for values that mean different things in different dimensions,
# e.g. "search" → "learn" (intent) vs "code-exploration" (task).
# ---------------------------------------------------------------------------
TAG_ALIASES_BY_DIM: dict[tuple[str, str], str] = {
    # "search" in intent = learn, in task = code-exploration
    ("intent", "search"): "learn",
    ("task", "search"): "code-exploration",
    ("task", "searching"): "code-exploration",
    # "research" in intent = learn, in task = code-exploration
    ("intent", "research"): "learn",
    ("task", "research"): "code-exploration",
    # "exploration" in intent = learn, in task = code-exploration
    ("intent", "exploration"): "learn",
    ("task", "exploration"): "code-exploration",
    # "analysis" in intent = review, in task = performance-analysis
    ("intent", "analysis"): "review",
    ("task", "analysis"): "performance-analysis",
    ("intent", "analyze"): "review",
    ("task", "analyze"): "performance-analysis",
    # "investigation" in intent = debug, in task = code-exploration
    ("intent", "investigation"): "debug",
    ("task", "investigation"): "code-exploration",
    # "information-gathering" in intent = learn, in task = code-exploration
    ("intent", "information-gathering"): "learn",
    ("task", "information-gathering"): "code-exploration",
    # "information retrieval" variants
    ("intent", "information-retrieval"): "learn",
    ("intent", "information retrieval"): "learn",
    # "verification" in intent = review, in task = code-review-task
    ("intent", "verify"): "review",
    ("task", "verification"): "code-review-task",
    # "data-visualization" in concept → documentation (task) but not concept
    ("concept", "data-visualization"): "architecture",
    ("task", "data-visualization"): "documentation",
    # domain "web-development" shouldn't map in domain (stays unmapped)
    # but in concept it maps to architecture
    ("concept", "web-development"): "architecture",
    ("concept", "ui-design"): "design-patterns",
    ("concept", "web-scraping"): "api-protocols",
    # "networking" in domain = network-programming, in concept = api-protocols
    ("domain", "networking"): "network-programming",
    ("concept", "networking"): "api-protocols",
}


def resolve_alias(value: str, dim: str | None = None) -> str:
    """Look up the canonical form for *value*, dimension-aware.

    1. If *dim* is given, check TAG_ALIASES_BY_DIM[(dim, value)] first.
    2. Fall back to TAG_ALIASES[value].
    3. Return the original value if no alias matches.
    """
    if dim is not None:
        qualified = TAG_ALIASES_BY_DIM.get((dim, value))
        if qualified is not None:
            return qualified
    return TAG_ALIASES.get(value, value)


def make_canonicalization_event(
    source_dimension: str,
    source_value: str,
    target_dimension: str,
    canonical_value: str,
    reason: str,
) -> dict:
    """Build a stable audit record for one canonicalization step."""
    return {
        "source_dimension": source_dimension,
        "source_value": source_value,
        "target_dimension": target_dimension,
        "canonical_value": canonical_value,
        "reason": reason,
    }


def canonicalization_stat_key(event: dict) -> str:
    """Build a stable flattened key for stats aggregation."""
    src_dim = event.get("source_dimension", "?")
    src_val = event.get("source_value", "?")
    dst_dim = event.get("target_dimension", "?")
    dst_val = event.get("canonical_value", "?")
    return f"{src_dim}:{src_val}->{dst_dim}:{dst_val}"
