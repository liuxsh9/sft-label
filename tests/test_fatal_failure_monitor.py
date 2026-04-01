"""Tests for FatalFailureMonitor and related abort infrastructure."""

import json
import pytest

from sft_label.llm_runtime import (
    FatalFailureMonitor,
    OutcomeClass,
    PipelineAbortError,
    classify_http_result,
    monitor_to_outcome_class,
)


# ── FatalFailureMonitor unit tests ──────────────────────────


class TestFatalStreak:
    def test_streak_triggers_abort(self):
        """5 consecutive AUTH_ERRORs → should_abort=True."""
        m = FatalFailureMonitor(fatal_streak_limit=5)
        for i in range(5):
            m.record(OutcomeClass.AUTH_ERROR, sample_id=f"s{i}", error="401")
        assert m.should_abort
        assert "连续 5" in m.abort_reason

    def test_streak_resets_on_success(self):
        """4 AUTH + 1 SUCCESS + 4 AUTH → should_abort=False."""
        m = FatalFailureMonitor(fatal_streak_limit=5)
        for _ in range(4):
            m.record(OutcomeClass.AUTH_ERROR)
        m.record(OutcomeClass.SUCCESS)
        for _ in range(4):
            m.record(OutcomeClass.AUTH_ERROR)
        assert not m.should_abort

    def test_streak_resets_on_non_fatal_failure(self):
        """4 AUTH + 1 TIMEOUT + 4 AUTH → should_abort=False."""
        m = FatalFailureMonitor(fatal_streak_limit=5)
        for _ in range(4):
            m.record(OutcomeClass.AUTH_ERROR)
        m.record(OutcomeClass.TIMEOUT)
        for _ in range(4):
            m.record(OutcomeClass.AUTH_ERROR)
        assert not m.should_abort

    def test_streak_stays_aborted(self):
        """Once aborted, should_abort stays True even after a success."""
        m = FatalFailureMonitor(fatal_streak_limit=2)
        m.record(OutcomeClass.AUTH_ERROR)
        m.record(OutcomeClass.AUTH_ERROR)
        assert m.should_abort
        m.record(OutcomeClass.SUCCESS)
        assert m.should_abort  # sticky


class TestGlobalRate:
    def test_global_rate_triggers_abort(self):
        """20 samples, 19 failures → 95% → should_abort=True."""
        m = FatalFailureMonitor(
            fatal_streak_limit=100,  # disable streak signal
            global_failure_rate_limit=0.95,
            global_rate_min_observations=20,
        )
        m.record(OutcomeClass.SUCCESS)
        for i in range(19):
            m.record(OutcomeClass.TIMEOUT, sample_id=f"s{i}")
        assert m.should_abort
        assert "全局失败率" in m.abort_reason
        assert "95" in m.abort_reason

    def test_global_rate_respects_min_obs(self):
        """5/5 failures but min_obs=20 → should_abort=False."""
        m = FatalFailureMonitor(
            fatal_streak_limit=100,
            global_failure_rate_limit=0.95,
            global_rate_min_observations=20,
        )
        for i in range(5):
            m.record(OutcomeClass.TIMEOUT, sample_id=f"s{i}")
        assert not m.should_abort

    def test_global_rate_below_threshold(self):
        """20 samples, 18 failures → 90% < 95% → should_abort=False."""
        m = FatalFailureMonitor(
            fatal_streak_limit=100,
            global_failure_rate_limit=0.95,
            global_rate_min_observations=20,
        )
        for _ in range(2):
            m.record(OutcomeClass.SUCCESS)
        for i in range(18):
            m.record(OutcomeClass.TIMEOUT, sample_id=f"s{i}")
        assert not m.should_abort


class TestMonitorBehavior:
    def test_disabled_monitor_never_aborts(self):
        """enabled=False → should_abort always False."""
        m = FatalFailureMonitor(fatal_streak_limit=1, enabled=False)
        m.record(OutcomeClass.AUTH_ERROR)
        m.record(OutcomeClass.AUTH_ERROR)
        m.record(OutcomeClass.AUTH_ERROR)
        assert not m.should_abort

    def test_content_filtered_not_fatal(self):
        """CONTENT_FILTERED resets streak, doesn't trigger abort."""
        m = FatalFailureMonitor(fatal_streak_limit=3)
        m.record(OutcomeClass.AUTH_ERROR)
        m.record(OutcomeClass.AUTH_ERROR)
        m.record(OutcomeClass.CONTENT_FILTERED)  # resets streak
        m.record(OutcomeClass.AUTH_ERROR)
        m.record(OutcomeClass.AUTH_ERROR)
        assert not m.should_abort

    def test_input_error_not_fatal(self):
        """INPUT_ERROR resets streak — single-sample context overflow is not global."""
        m = FatalFailureMonitor(fatal_streak_limit=3)
        m.record(OutcomeClass.AUTH_ERROR)
        m.record(OutcomeClass.AUTH_ERROR)
        m.record(OutcomeClass.INPUT_ERROR)  # resets streak
        m.record(OutcomeClass.AUTH_ERROR)
        m.record(OutcomeClass.AUTH_ERROR)
        assert not m.should_abort

    def test_to_dict_serializable(self):
        """to_dict() returns a JSON-serializable dict."""
        m = FatalFailureMonitor(fatal_streak_limit=5)
        m.record(OutcomeClass.AUTH_ERROR, sample_id="test", error="billing exhausted")
        m.record(OutcomeClass.SUCCESS)
        d = m.to_dict()
        # Should be JSON-serializable
        json_str = json.dumps(d)
        assert json_str
        # Check key fields
        assert d["enabled"] is True
        assert d["aborted"] is False
        assert d["total"] == 2
        assert d["total_failed"] == 1
        assert d["total_success"] == 1
        assert d["fatal_streak"] == 0
        assert d["last_fatal_error"] == "billing exhausted"
        assert d["last_fatal_sample_id"] == "test"

    def test_custom_fatal_classes(self):
        """Custom fatal_classes override the default."""
        m = FatalFailureMonitor(
            fatal_streak_limit=2,
            fatal_classes=frozenset({OutcomeClass.TIMEOUT}),
        )
        m.record(OutcomeClass.AUTH_ERROR)
        m.record(OutcomeClass.AUTH_ERROR)
        assert not m.should_abort  # AUTH_ERROR not in custom fatal_classes
        m.record(OutcomeClass.TIMEOUT)
        m.record(OutcomeClass.TIMEOUT)
        assert m.should_abort


# ── HTTP 402 classification ──────────────────────────────


class TestHttp402Classification:
    def test_402_classified_as_auth_error(self):
        """HTTP 402 should be classified as AUTH_ERROR."""
        outcome = classify_http_result(402, error_text="Payment Required")
        assert outcome.classification is OutcomeClass.AUTH_ERROR
        assert outcome.status_code == 402

    def test_402_not_retryable(self):
        """HTTP 402 should not be retryable."""
        outcome = classify_http_result(402, error_text="Payment Required")
        assert not outcome.is_retryable

    def test_401_still_auth_error(self):
        """HTTP 401 should still be classified as AUTH_ERROR."""
        outcome = classify_http_result(401, error_text="Unauthorized")
        assert outcome.classification is OutcomeClass.AUTH_ERROR


# ── PipelineAbortError ──────────────────────────────────


class TestPipelineAbortError:
    def test_basic_construction(self):
        err = PipelineAbortError("billing exhausted", {"total": 5})
        assert str(err) == "billing exhausted"
        assert err.reason == "billing exhausted"
        assert err.details == {"total": 5}

    def test_default_details(self):
        err = PipelineAbortError("test")
        assert err.details == {}

    def test_is_exception(self):
        with pytest.raises(PipelineAbortError) as exc_info:
            raise PipelineAbortError("test abort")
        assert exc_info.value.reason == "test abort"


# ── monitor_to_outcome_class ────────────────────────────


class TestMonitorToOutcomeClass:
    def test_none_monitor(self):
        assert monitor_to_outcome_class(None) is OutcomeClass.TRANSIENT_ERROR

    def test_success_monitor(self):
        assert monitor_to_outcome_class({"status": "success"}) is OutcomeClass.SUCCESS

    def test_auth_error_monitor(self):
        assert monitor_to_outcome_class({
            "status": "call1_failed",
            "error_class": "auth_error",
        }) is OutcomeClass.AUTH_ERROR

    def test_timeout_monitor(self):
        assert monitor_to_outcome_class({
            "status": "sample_total_timeout",
            "error_class": "timeout",
        }) is OutcomeClass.TIMEOUT

    def test_unknown_error_class_falls_back(self):
        assert monitor_to_outcome_class({
            "status": "failed",
            "error_class": "some_unknown_class",
        }) is OutcomeClass.TRANSIENT_ERROR

    def test_missing_error_class_falls_back(self):
        assert monitor_to_outcome_class({
            "status": "failed",
        }) is OutcomeClass.TRANSIENT_ERROR
