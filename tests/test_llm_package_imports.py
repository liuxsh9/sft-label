"""Verify that all LLM symbols are importable from their new locations
AND that backward-compatible paths still work.

Written BEFORE the move (TDD RED phase), expected to FAIL initially.
After the extraction is complete, all tests should PASS (GREEN phase).
"""
from __future__ import annotations


# ─── llm/runtime.py (migrated from llm_runtime.py) ───────────────


def test_runtime_importable_from_llm_package():
    """New sft_label.llm.runtime path works."""
    from sft_label.llm.runtime import (  # noqa: F401
        AdaptiveLLMRuntime,
        AdaptiveRateLimiter,
        DynamicConcurrencyGate,
        FatalFailureMonitor,
        OutcomeClass,
        PipelineAbortError,
        RequestOutcome,
        classify_exception,
        classify_http_result,
        monitor_to_outcome_class,
    )


def test_runtime_backward_compat_via_llm_runtime():
    """Old sft_label.llm_runtime import path still works (shim)."""
    from sft_label.llm_runtime import (  # noqa: F401
        AdaptiveLLMRuntime,
        FatalFailureMonitor,
        OutcomeClass,
        PipelineAbortError,
        RequestOutcome,
        classify_exception,
        classify_http_result,
        monitor_to_outcome_class,
    )


# ─── llm/transport.py (extracted from pipeline.py) ────────────────


def test_transport_importable_from_llm_package():
    """New sft_label.llm.transport path works."""
    from sft_label.llm.transport import (  # noqa: F401
        AsyncRateLimiter,
        RequestStats,
        async_llm_call,
    )


def test_transport_backward_compat_via_pipeline():
    """Old sft_label.pipeline import path still works (re-export)."""
    from sft_label.pipeline import (  # noqa: F401
        AsyncRateLimiter,
        RequestStats,
        async_llm_call,
    )


# ─── llm/progress.py (extracted from pipeline.py) ─────────────────


def test_progress_importable_from_llm_package():
    """New sft_label.llm.progress path works."""
    from sft_label.llm.progress import (  # noqa: F401
        RuntimeEtaEstimator,
        format_progress_info,
        parse_run_progress,
        _format_runtime_progress,
        _decorate_run_info_with_runtime,
        _update_llm_task_progress,
    )


def test_progress_backward_compat_via_pipeline():
    """Old sft_label.pipeline import path still works (re-export)."""
    from sft_label.pipeline import (  # noqa: F401
        RuntimeEtaEstimator,
        format_progress_info,
        parse_run_progress,
    )


# ─── llm/factory.py (extracted from pipeline.py) ──────────────────


def test_factory_importable_from_llm_package():
    """New sft_label.llm.factory path works."""
    from sft_label.llm.factory import (  # noqa: F401
        _build_adaptive_runtime,
        _config_adaptive_runtime,
        _make_recovery_config,
        _outcome_from_usage,
    )


def test_factory_backward_compat_via_pipeline():
    """Old sft_label.pipeline import path still works (re-export)."""
    from sft_label.pipeline import (  # noqa: F401
        _build_adaptive_runtime,
        _config_adaptive_runtime,
    )


# ─── llm/http.py (extracted from pipeline.py) ─────────────────────


def test_http_importable_from_llm_package():
    """New sft_label.llm.http path works."""
    from sft_label.llm.http import (  # noqa: F401
        _build_http_client_limits,
        _chunked_output_fd_budget,
    )


def test_http_backward_compat_via_pipeline():
    """Old sft_label.pipeline import path still works (re-export)."""
    from sft_label.pipeline import _build_http_client_limits  # noqa: F401


# ─── llm/__init__.py (convenience re-exports) ─────────────────────


def test_llm_package_convenience_imports():
    """sft_label.llm exposes key symbols for convenience."""
    from sft_label.llm import (  # noqa: F401
        AdaptiveLLMRuntime,
        AsyncRateLimiter,
        OutcomeClass,
        PipelineAbortError,
        RequestOutcome,
        RuntimeEtaEstimator,
        async_llm_call,
        format_progress_info,
    )


# ─── Architectural invariant: scoring.py no longer imports from pipeline ──


def test_scoring_does_not_import_llm_symbols_from_pipeline():
    """scoring.py should get LLM symbols from sft_label.llm, not pipeline."""
    import ast
    import inspect
    import sft_label.scoring as scoring

    source = inspect.getsource(scoring)
    tree = ast.parse(source)

    pipeline_imports: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module == "sft_label.pipeline":
            pipeline_imports.extend(alias.name for alias in node.names)

    llm_symbols = {
        "async_llm_call",
        "AsyncRateLimiter",
        "RuntimeEtaEstimator",
        "format_progress_info",
        "parse_run_progress",
    }
    violations = llm_symbols & set(pipeline_imports)
    assert not violations, (
        f"scoring.py still imports LLM symbols from pipeline: {violations}"
    )


# ─── Identity checks: symbols from old and new paths are the same object ──


def test_async_llm_call_identity():
    """The re-exported symbol is the exact same function object."""
    from sft_label.llm.transport import async_llm_call as new
    from sft_label.pipeline import async_llm_call as old

    assert new is old


def test_adaptive_llm_runtime_identity():
    """The class from old and new paths is the same object."""
    from sft_label.llm.runtime import AdaptiveLLMRuntime as new
    from sft_label.llm_runtime import AdaptiveLLMRuntime as old

    assert new is old


def test_runtime_eta_estimator_identity():
    """RuntimeEtaEstimator from old and new paths is the same object."""
    from sft_label.llm.progress import RuntimeEtaEstimator as new
    from sft_label.pipeline import RuntimeEtaEstimator as old

    assert new is old
