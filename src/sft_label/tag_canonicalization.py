"""Canonical tag remapping helpers for Pass 1 validation."""

from __future__ import annotations


# Alias → canonical tag mapping. Applied before pool validation so that
# common LLM outputs (underscore variants, abbreviations, sub-topic names,
# and a few high-value SWE synonyms) are silently resolved instead of
# becoming unmapped.
TAG_ALIASES = {
    # language
    "c++": "cpp",
    "c#": "csharp",
    "f#": "fsharp",
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
    # intent — modify aliases
    "refactor": "modify",
    "optimize": "modify",
    "transform": "modify",
    # language — common abbreviations
    "objc": "objective-c",
    "objective_c": "objective-c",
    "js": "javascript",
    "ts": "typescript",
    "rb": "ruby",
    "py": "python",
    # concept — additional mappings
    "regex": "algorithms",
    "regular-expressions": "algorithms",
    "async-await": "concurrency",
    "system-programming": "concurrency",
}


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
