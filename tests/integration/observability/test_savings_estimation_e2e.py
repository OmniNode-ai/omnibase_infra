# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

# Copyright (c) 2026 OmniNode Team
"""End-to-end integration test for the savings estimation pipeline.

Validates the full flow:
  1. Ingest mock events (llm-call-completed, hook-context-injected, validator-catch)
  2. Ingest session-outcome to trigger finalization
  3. Verify ServiceSavingsEstimator produces savings-estimated.v1 with correct totals
  4. Verify the output shape matches the expected wire format

No real Kafka broker is needed — the test drives ServiceSavingsEstimator directly
via its ``ingest_event()`` / ``finalize_ready_sessions()`` API.

Ticket: OMN-5555
"""

from __future__ import annotations

import time
from typing import Any, cast
from unittest.mock import patch

import pytest

from omnibase_infra.models.pricing.model_pricing_entry import ModelPricingEntry
from omnibase_infra.models.pricing.model_pricing_table import ModelPricingTable
from omnibase_infra.services.observability.savings_estimation.config import (
    ConfigSavingsEstimation,
)
from omnibase_infra.services.observability.savings_estimation.consumer import (
    ServiceSavingsEstimator,
)

# ---------------------------------------------------------------------------
# Test fixtures
# ---------------------------------------------------------------------------

TOPIC_LLM_CALL = "onex.evt.omniintelligence.llm-call-completed.v1"
TOPIC_SESSION_OUTCOME = "onex.evt.omniclaude.session-outcome.v1"
TOPIC_HOOK_INJECTION = "onex.evt.omniclaude.hook-context-injected.v1"
TOPIC_VALIDATOR_CATCH = "onex.evt.omniclaude.validator-catch.v1"

SESSION_ID = "test-session-001"
CORRELATION_ID = "corr-001"


def _build_pricing_table() -> ModelPricingTable:
    """Build a minimal pricing table with a local and a paid model."""
    return ModelPricingTable(
        schema_version="1.0",
        models={
            "qwen2.5-coder-14b": ModelPricingEntry(
                input_cost_per_1k=0.0,
                output_cost_per_1k=0.0,
                effective_date="2026-01-01",
                note="Local model - zero cost",
            ),
            "claude-opus-4-6": ModelPricingEntry(
                input_cost_per_1k=0.015,
                output_cost_per_1k=0.075,
                effective_date="2026-01-01",
            ),
        },
    )


def _build_config(grace_window: float = 0.0) -> ConfigSavingsEstimation:
    """Build a test config with zero grace window for immediate finalization."""
    return ConfigSavingsEstimation(
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
        pricing_table=_build_pricing_table(),
    )


# ---------------------------------------------------------------------------
# Required output fields per ContractSavingsEstimate wire format
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
    "direct_tokens_saved",
    "estimated_total_savings_usd",
    "estimated_total_tokens_saved",
    "categories",
    "direct_confidence",
    "heuristic_confidence_avg",
    "estimation_method",
    "treatment_group",
    "is_measured",
    "measurement_basis",
    "baseline_session_id",
    "pricing_manifest_version",
    "completeness_status",
    "extensions",
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

        # Step 1: Produce LLM call events (local model -> savings via routing)
        service.ingest_event(
            TOPIC_LLM_CALL,
            {
                "session_id": SESSION_ID,
                "model_id": "qwen2.5-coder-14b",
                "prompt_tokens": 5000,
                "completion_tokens": 1000,
            },
        )
        service.ingest_event(
            TOPIC_LLM_CALL,
            {
                "session_id": SESSION_ID,
                "model_id": "qwen2.5-coder-14b",
                "prompt_tokens": 3000,
                "completion_tokens": 500,
            },
        )

        # Step 2: Produce hook-context-injected events
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
        assert estimate["correlation_id"] == CORRELATION_ID
        assert estimate["schema_version"] == "1.0"
        assert estimate["treatment_group"] == "treatment"
        assert estimate["estimation_method"] == "tiered_attribution_v1"

        # Verify deterministic source_event_id
        assert estimate["source_event_id"] == "savings-test-session-001-v1.0"

        # Verify actual token totals (5000+1000 + 3000+500 = 9500)
        assert estimate["actual_total_tokens"] == 9500

        # Local model -> actual cost should be 0
        assert estimate["actual_cost_usd"] == 0.0

        # Verify direct savings > 0 (local routing generates savings)
        assert isinstance(estimate["direct_savings_usd"], float)
        assert estimate["direct_savings_usd"] > 0.0
        assert estimate["direct_tokens_saved"] == 9500
        assert estimate["direct_confidence"] == 1.0

        # Verify categories list
        categories = cast("list[dict[str, Any]]", estimate["categories"])
        assert isinstance(categories, list)
        assert len(categories) >= 1

        category_names = [c["category"] for c in categories]
        assert "local_routing" in category_names

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
            TOPIC_LLM_CALL,
            {
                "session_id": SESSION_ID,
                "model_id": "qwen2.5-coder-14b",
                "prompt_tokens": 1000,
                "completion_tokens": 200,
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
            TOPIC_LLM_CALL,
            {
                "session_id": SESSION_ID,
                "model_id": "qwen2.5-coder-14b",
                "prompt_tokens": 9999,
                "completion_tokens": 9999,
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
                    "model_id": "qwen2.5-coder-14b",
                    "prompt_tokens": tokens,
                    "completion_tokens": 100,
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

    async def test_validator_catch_produces_heuristic_savings(self) -> None:
        """Validator catches generate Tier B heuristic savings."""
        service = _build_service()

        service.ingest_event(
            TOPIC_VALIDATOR_CATCH,
            {
                "session_id": SESSION_ID,
                "validator_type": "pre_commit",
                "severity": "error",
            },
        )
        service.ingest_event(
            TOPIC_VALIDATOR_CATCH,
            {
                "session_id": SESSION_ID,
                "validator_type": "ci",
                "severity": "error",
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

        categories = cast("list[dict[str, Any]]", estimate["categories"])
        validator_cat = next(
            (c for c in categories if c["category"] == "validator_catches"), None
        )
        assert validator_cat is not None
        assert validator_cat["tier"] == "heuristic"
        assert validator_cat["tokens_saved"] > 0
        assert validator_cat["confidence"] > 0.0

    async def test_pattern_injection_produces_heuristic_savings(self) -> None:
        """Pattern injection generates Tier B savings with regen multiplier."""
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

        categories = cast("list[dict[str, Any]]", estimate["categories"])
        injection_cat = next(
            (c for c in categories if c["category"] == "pattern_injection"), None
        )
        assert injection_cat is not None
        assert injection_cat["tier"] == "heuristic"
        # 500 tokens * 3.0 regen_multiplier = 1500 tokens saved
        assert injection_cat["tokens_saved"] == 1500
        assert injection_cat["confidence"] == 0.8

    async def test_session_without_outcome_does_not_finalize_early(self) -> None:
        """Sessions without session-outcome are not finalized before timeout."""
        service = _build_service()

        service.ingest_event(
            TOPIC_LLM_CALL,
            {
                "session_id": SESSION_ID,
                "model_id": "qwen2.5-coder-14b",
                "prompt_tokens": 1000,
                "completion_tokens": 200,
            },
        )

        # Without session-outcome, grace window doesn't apply
        original_monotonic = time.monotonic
        with patch("time.monotonic", side_effect=lambda: original_monotonic() + 60):
            results = await service.finalize_ready_sessions()

        # Should not finalize — session_timeout_seconds is 3600
        assert len(results) == 0
        assert service.active_session_count == 1
