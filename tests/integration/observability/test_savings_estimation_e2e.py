# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

# Copyright (c) 2026 OmniNode Team
"""End-to-end integration test for the savings estimation pipeline.

Validates the full flow:
  1. Ingest mock events (llm-call-completed, hook-context-injected, validator-catch)
  2. Ingest session-outcome to trigger finalization
  3. Verify ServiceSavingsEstimator produces savings-estimated.v1 with correct totals
  4. Verify the output shape matches the expected wire format
  5. Verify heuristic savings from validator catches with severity and diminishing returns

No real Kafka broker is needed -- the test drives ServiceSavingsEstimator directly
via its ``ingest_event()`` / ``finalize_ready_sessions()`` API.

Tickets: OMN-5555, OMN-7494
"""

from __future__ import annotations

import time
from typing import Any, cast
from unittest.mock import patch

import pytest

from omnibase_infra.services.observability.savings_estimation.config import (
    ConfigSavingsEstimation,
)
from omnibase_infra.services.observability.savings_estimation.consumer import (
    EnumCatchSeverity,
    ServiceSavingsEstimator,
    ValidatorCatchSignal,
    _compute_validator_catch_savings,
)

# ---------------------------------------------------------------------------
# Test fixtures
# ---------------------------------------------------------------------------

TOPIC_LLM_CALL = "onex.evt.omniintelligence.llm-call-completed.v1"
TOPIC_SESSION_OUTCOME = "onex.evt.omniclaude.session-outcome.v1"
TOPIC_HOOK_INJECTION = "onex.evt.omniclaude.hook-context-injected.v1"
TOPIC_VALIDATOR_CATCH = "onex.evt.omniclaude.validator-catch.v1"
TOPIC_PATTERN_ENFORCEMENT = "onex.evt.omniclaude.pattern-enforcement.v1"

SESSION_ID = "test-session-001"
CORRELATION_ID = "corr-001"


def _build_config(grace_window: float = 0.0) -> ConfigSavingsEstimation:
    """Build a test config with zero grace window for immediate finalization."""
    return ConfigSavingsEstimation(
        kafka_bootstrap_servers="localhost:19092",
        grace_window_seconds=max(grace_window, 1.0),
        session_timeout_seconds=3600.0,
        max_sessions=100,
        finalized_session_cache_size=1000,
        schema_version="1.0",
    )


def _build_service(
    grace_window: float = 0.0,
) -> ServiceSavingsEstimator:
    return ServiceSavingsEstimator(
        config=_build_config(grace_window),
    )


# ---------------------------------------------------------------------------
# Required output fields per ModelSavingsEstimate wire format
# ---------------------------------------------------------------------------

REQUIRED_OUTPUT_FIELDS = {
    "schema_version",
    "session_id",
    "correlation_id",
    "timestamp_iso",
    "actual_total_tokens",
    "actual_cost_usd",
    "actual_model_id",
    "counterfactual_model_id",
    "direct_savings_usd",
    "heuristic_savings_usd",
    "direct_tokens_saved",
    "estimated_total_savings_usd",
    "estimated_total_tokens_saved",
    "categories",
    "direct_confidence",
    "heuristic_confidence_avg",
    "estimation_method",
    "treatment_group",
    "is_measured",
    "pricing_manifest_version",
    "completeness_status",
    "source_event_id",
}


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.integration
class TestSavingsEstimationE2E:
    """End-to-end integration tests for the full savings estimation pipeline."""

    async def test_full_pipeline_produces_savings_event(self) -> None:
        """Ingest a realistic session's events and verify the output estimate."""
        service = _build_service()

        # Step 1: Produce LLM call events
        service.ingest_event(
            TOPIC_LLM_CALL,
            {
                "session_id": SESSION_ID,
                "model_id": "claude-opus-4-6",
                "prompt_tokens": 5000,
                "completion_tokens": 1000,
            },
        )
        service.ingest_event(
            TOPIC_LLM_CALL,
            {
                "session_id": SESSION_ID,
                "model_id": "claude-opus-4-6",
                "prompt_tokens": 3000,
                "completion_tokens": 500,
            },
        )

        # Step 2: Produce hook-context-injected events (generates effectiveness entries)
        service.ingest_event(
            TOPIC_HOOK_INJECTION,
            {
                "session_id": SESSION_ID,
                "tokens_injected": 200,
                "patterns_count": 3,
            },
        )

        # Step 3: Produce validator-catch events
        service.ingest_event(
            TOPIC_VALIDATOR_CATCH,
            {
                "session_id": SESSION_ID,
                "validator_type": "pre_commit",
                "severity": "error",
            },
        )

        # Step 4: Produce session-outcome to trigger finalization
        service.ingest_event(
            TOPIC_SESSION_OUTCOME,
            {
                "session_id": SESSION_ID,
                "correlation_id": CORRELATION_ID,
                "treatment_group": "treatment",
            },
        )

        assert service.active_session_count == 1

        # Patch time.monotonic to simulate grace window elapsed
        original_monotonic = time.monotonic
        with patch("time.monotonic", side_effect=lambda: original_monotonic() + 60):
            results = await service.finalize_ready_sessions()

        # Should produce exactly one savings estimate
        assert len(results) == 1
        estimate = results[0]

        # Verify all required fields are present
        missing_fields = REQUIRED_OUTPUT_FIELDS - set(estimate.keys())
        assert not missing_fields, f"Missing fields in estimate: {missing_fields}"

        # Verify session identity
        assert estimate["session_id"] == SESSION_ID
        assert estimate["schema_version"] == "1.0"
        assert estimate["estimation_method"] == "tiered_attribution_v1"

        # Verify deterministic source_event_id
        assert estimate["source_event_id"] == "savings-test-session-001-v1.0"

        # Verify actual token totals (5000+1000 + 3000+500 = 9500)
        assert estimate["actual_total_tokens"] == 9500

        # Verify direct savings > 0 (injection signals produce savings)
        assert isinstance(estimate["direct_savings_usd"], float)
        assert estimate["direct_savings_usd"] > 0.0
        assert isinstance(estimate["direct_tokens_saved"], int)
        assert estimate["direct_tokens_saved"] > 0

        # Verify heuristic savings from validator catch (severity=error -> CRITICAL)
        assert isinstance(estimate["heuristic_savings_usd"], float)
        assert estimate["heuristic_savings_usd"] > 0.0

        # Verify total = direct + heuristic
        expected_total = (
            estimate["direct_savings_usd"] + estimate["heuristic_savings_usd"]
        )
        assert abs(estimate["estimated_total_savings_usd"] - expected_total) < 1e-8

        # Verify counterfactual model is set
        assert estimate["counterfactual_model_id"] is not None

        # Verify categories list includes validator_catch
        categories = cast("list[dict[str, Any]]", estimate["categories"])
        assert isinstance(categories, (list, tuple))
        assert len(categories) >= 1
        cat_names = [c["category"] for c in categories]
        assert "validator_catch" in cat_names

        # Session should now be finalized
        assert service.is_finalized(SESSION_ID)
        assert service.active_session_count == 0

    async def test_empty_session_produces_no_output(self) -> None:
        """A session with only session-outcome and no signals produces nothing."""
        service = _build_service()

        service.ingest_event(
            TOPIC_SESSION_OUTCOME,
            {
                "session_id": "empty-session",
                "correlation_id": "corr-empty",
            },
        )

        original_monotonic = time.monotonic
        with patch("time.monotonic", side_effect=lambda: original_monotonic() + 60):
            results = await service.finalize_ready_sessions()

        assert len(results) == 0

    async def test_duplicate_session_is_rejected_after_finalization(self) -> None:
        """Events for a finalized session are silently dropped."""
        service = _build_service()

        # First session lifecycle
        service.ingest_event(
            TOPIC_HOOK_INJECTION,
            {
                "session_id": SESSION_ID,
                "tokens_injected": 300,
                "patterns_count": 5,
            },
        )
        service.ingest_event(
            TOPIC_SESSION_OUTCOME,
            {"session_id": SESSION_ID, "correlation_id": CORRELATION_ID},
        )

        original_monotonic = time.monotonic
        with patch("time.monotonic", side_effect=lambda: original_monotonic() + 60):
            results = await service.finalize_ready_sessions()

        assert len(results) == 1
        assert service.is_finalized(SESSION_ID)

        # Try to ingest again for the same session
        service.ingest_event(
            TOPIC_HOOK_INJECTION,
            {
                "session_id": SESSION_ID,
                "tokens_injected": 999,
                "patterns_count": 10,
            },
        )

        # Session should NOT be re-created
        assert service.active_session_count == 0

    async def test_multiple_sessions_finalize_independently(self) -> None:
        """Two sessions finalize with independent totals."""
        service = _build_service()

        for sid, tokens in [("sess-a", 1000), ("sess-b", 2000)]:
            service.ingest_event(
                TOPIC_LLM_CALL,
                {
                    "session_id": sid,
                    "model_id": "claude-opus-4-6",
                    "prompt_tokens": tokens,
                    "completion_tokens": 100,
                },
            )
            service.ingest_event(
                TOPIC_HOOK_INJECTION,
                {
                    "session_id": sid,
                    "tokens_injected": 150,
                    "patterns_count": 3,
                },
            )
            service.ingest_event(
                TOPIC_SESSION_OUTCOME,
                {"session_id": sid, "correlation_id": f"corr-{sid}"},
            )

        original_monotonic = time.monotonic
        with patch("time.monotonic", side_effect=lambda: original_monotonic() + 60):
            results = await service.finalize_ready_sessions()

        assert len(results) == 2
        by_session = {r["session_id"]: r for r in results}
        assert by_session["sess-a"]["actual_total_tokens"] == 1100
        assert by_session["sess-b"]["actual_total_tokens"] == 2100

    async def test_pattern_injection_produces_savings(self) -> None:
        """Pattern injection generates savings via effectiveness entries."""
        service = _build_service()

        service.ingest_event(
            TOPIC_HOOK_INJECTION,
            {
                "session_id": SESSION_ID,
                "tokens_injected": 500,
                "patterns_count": 5,
            },
        )
        service.ingest_event(
            TOPIC_SESSION_OUTCOME,
            {"session_id": SESSION_ID, "correlation_id": CORRELATION_ID},
        )

        original_monotonic = time.monotonic
        with patch("time.monotonic", side_effect=lambda: original_monotonic() + 60):
            results = await service.finalize_ready_sessions()

        assert len(results) == 1
        estimate = results[0]

        # Injection of 500 tokens should produce savings
        assert estimate["direct_tokens_saved"] > 0
        assert estimate["direct_savings_usd"] > 0.0

    async def test_session_without_outcome_does_not_finalize_early(self) -> None:
        """Sessions without session-outcome are not finalized before timeout."""
        service = _build_service()

        service.ingest_event(
            TOPIC_HOOK_INJECTION,
            {
                "session_id": SESSION_ID,
                "tokens_injected": 200,
                "patterns_count": 3,
            },
        )

        # Without session-outcome, grace window doesn't apply
        original_monotonic = time.monotonic
        with patch("time.monotonic", side_effect=lambda: original_monotonic() + 60):
            results = await service.finalize_ready_sessions()

        # Should not finalize -- session_timeout_seconds is 3600
        assert len(results) == 0
        assert service.active_session_count == 1


@pytest.mark.integration
class TestValidatorCatchSavings:
    """Tests for heuristic avoided-rework savings from validator catches (OMN-7494)."""

    async def test_validator_catch_produces_heuristic_savings(self) -> None:
        """A single CRITICAL validator catch produces heuristic savings."""
        service = _build_service()

        service.ingest_event(
            TOPIC_HOOK_INJECTION,
            {
                "session_id": SESSION_ID,
                "tokens_injected": 100,
                "patterns_count": 2,
            },
        )
        service.ingest_event(
            TOPIC_VALIDATOR_CATCH,
            {
                "session_id": SESSION_ID,
                "validator_type": "pre_commit",
                "severity": "critical",
            },
        )
        service.ingest_event(
            TOPIC_SESSION_OUTCOME,
            {"session_id": SESSION_ID, "correlation_id": CORRELATION_ID},
        )

        original_monotonic = time.monotonic
        with patch("time.monotonic", side_effect=lambda: original_monotonic() + 60):
            results = await service.finalize_ready_sessions()

        assert len(results) == 1
        estimate = results[0]

        # CRITICAL catch should produce $0.50 heuristic savings
        assert estimate["heuristic_savings_usd"] == pytest.approx(0.50, abs=0.01)

        # Categories should include validator_catch
        categories = cast("list[dict[str, Any]]", estimate["categories"])
        validator_cats = [c for c in categories if c["category"] == "validator_catch"]
        assert len(validator_cats) == 1
        assert validator_cats[0]["savings_usd"] > 0
        assert validator_cats[0]["confidence"] == pytest.approx(0.7, abs=0.01)

    async def test_multiple_catches_have_diminishing_returns(self) -> None:
        """Multiple validator catches in one session saturate rather than stack."""
        # Single critical catch
        single_usd, _, _ = _compute_validator_catch_savings(
            [ValidatorCatchSignal(severity=EnumCatchSeverity.CRITICAL)]
        )

        # Five critical catches
        five_catches = [
            ValidatorCatchSignal(severity=EnumCatchSeverity.CRITICAL) for _ in range(5)
        ]
        five_usd, _, _ = _compute_validator_catch_savings(five_catches)

        # Five catches should be less than 5x a single catch (diminishing returns)
        assert five_usd < single_usd * 5
        # But more than a single catch
        assert five_usd > single_usd

    async def test_severity_ordering_affects_savings(self) -> None:
        """CRITICAL catches produce more savings than MINOR catches."""
        critical_usd, critical_tokens, _ = _compute_validator_catch_savings(
            [ValidatorCatchSignal(severity=EnumCatchSeverity.CRITICAL)]
        )
        minor_usd, minor_tokens, _ = _compute_validator_catch_savings(
            [ValidatorCatchSignal(severity=EnumCatchSeverity.MINOR)]
        )

        assert critical_usd > minor_usd
        assert critical_tokens > minor_tokens

    async def test_pattern_enforcement_topic_ingested(self) -> None:
        """Events from pattern-enforcement.v1 topic are ingested as catches."""
        service = _build_service()

        service.ingest_event(
            TOPIC_HOOK_INJECTION,
            {
                "session_id": SESSION_ID,
                "tokens_injected": 100,
                "patterns_count": 2,
            },
        )
        # Use pattern-enforcement topic instead of validator-catch
        service.ingest_event(
            TOPIC_PATTERN_ENFORCEMENT,
            {
                "session_id": SESSION_ID,
                "severity": "major",
                "validator_type": "poly_enforcer",
            },
        )
        service.ingest_event(
            TOPIC_SESSION_OUTCOME,
            {"session_id": SESSION_ID, "correlation_id": CORRELATION_ID},
        )

        original_monotonic = time.monotonic
        with patch("time.monotonic", side_effect=lambda: original_monotonic() + 60):
            results = await service.finalize_ready_sessions()

        assert len(results) == 1
        estimate = results[0]
        assert estimate["heuristic_savings_usd"] > 0

    async def test_no_catches_produces_zero_heuristic(self) -> None:
        """A session without validator catches has zero heuristic savings."""
        service = _build_service()

        service.ingest_event(
            TOPIC_HOOK_INJECTION,
            {
                "session_id": SESSION_ID,
                "tokens_injected": 500,
                "patterns_count": 5,
            },
        )
        service.ingest_event(
            TOPIC_SESSION_OUTCOME,
            {"session_id": SESSION_ID, "correlation_id": CORRELATION_ID},
        )

        original_monotonic = time.monotonic
        with patch("time.monotonic", side_effect=lambda: original_monotonic() + 60):
            results = await service.finalize_ready_sessions()

        assert len(results) == 1
        estimate = results[0]
        assert estimate["heuristic_savings_usd"] == 0.0
        # Total should equal direct (no heuristic)
        assert estimate["estimated_total_savings_usd"] == estimate["direct_savings_usd"]

    async def test_counterfactual_model_for_sonnet(self) -> None:
        """Sonnet model gets opus as counterfactual."""
        service = _build_service()

        service.ingest_event(
            TOPIC_LLM_CALL,
            {
                "session_id": SESSION_ID,
                "model_id": "claude-sonnet-4",
                "prompt_tokens": 5000,
                "completion_tokens": 1000,
            },
        )
        service.ingest_event(
            TOPIC_HOOK_INJECTION,
            {
                "session_id": SESSION_ID,
                "tokens_injected": 200,
                "patterns_count": 3,
            },
        )
        service.ingest_event(
            TOPIC_SESSION_OUTCOME,
            {"session_id": SESSION_ID, "correlation_id": CORRELATION_ID},
        )

        original_monotonic = time.monotonic
        with patch("time.monotonic", side_effect=lambda: original_monotonic() + 60):
            results = await service.finalize_ready_sessions()

        assert len(results) == 1
        estimate = results[0]
        assert estimate["counterfactual_model_id"] == "claude-opus-4-6"

    async def test_only_validator_catches_still_produces_estimate(self) -> None:
        """A session with only validator catches (no injection) still produces output."""
        service = _build_service()

        service.ingest_event(
            TOPIC_VALIDATOR_CATCH,
            {
                "session_id": SESSION_ID,
                "severity": "critical",
                "validator_type": "pre_commit",
            },
        )
        service.ingest_event(
            TOPIC_SESSION_OUTCOME,
            {"session_id": SESSION_ID, "correlation_id": CORRELATION_ID},
        )

        original_monotonic = time.monotonic
        with patch("time.monotonic", side_effect=lambda: original_monotonic() + 60):
            results = await service.finalize_ready_sessions()

        assert len(results) == 1
        estimate = results[0]
        # Direct savings should be zero (no injection)
        assert estimate["direct_savings_usd"] == 0.0
        # Heuristic savings should be nonzero (validator catch)
        assert estimate["heuristic_savings_usd"] > 0.0
        assert estimate["estimated_total_savings_usd"] > 0.0


@pytest.mark.integration
class TestSeverityClassification:
    """Tests for severity string classification."""

    def test_error_maps_to_critical(self) -> None:
        from omnibase_infra.services.observability.savings_estimation.consumer import (
            _classify_severity,
        )

        assert _classify_severity("error") == EnumCatchSeverity.CRITICAL
        assert _classify_severity("CRITICAL") == EnumCatchSeverity.CRITICAL
        assert _classify_severity("fatal") == EnumCatchSeverity.CRITICAL

    def test_warning_maps_to_major(self) -> None:
        from omnibase_infra.services.observability.savings_estimation.consumer import (
            _classify_severity,
        )

        assert _classify_severity("warning") == EnumCatchSeverity.MAJOR
        assert _classify_severity("MAJOR") == EnumCatchSeverity.MAJOR
        assert _classify_severity("warn") == EnumCatchSeverity.MAJOR

    def test_unknown_maps_to_minor(self) -> None:
        from omnibase_infra.services.observability.savings_estimation.consumer import (
            _classify_severity,
        )

        assert _classify_severity("info") == EnumCatchSeverity.MINOR
        assert _classify_severity("minor") == EnumCatchSeverity.MINOR
        assert _classify_severity("unknown") == EnumCatchSeverity.MINOR
