# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""End-to-end integration test for the savings estimation pipeline.

Exercises the full flow: event ingestion across all four consumed topics,
session correlation, grace window finalization, and output estimate
validation — without requiring live Kafka or Postgres.

Pipeline under test:
    llm-call-completed.v1  ─┐
    session-outcome.v1     ─┤─► ServiceSavingsEstimator ─► HandlerSavingsEstimator
    hook-context-injected.v1─┤      (correlate)                (compute)
    validator-catch.v1     ─┘          │
                                       ▼
                              savings-estimated.v1 (output event)

Related Tickets:
    - OMN-5555: Integration test e2e savings pipeline
"""

from __future__ import annotations

from unittest.mock import patch

import pytest

from omnibase_infra.models.pricing.model_pricing_table import ModelPricingTable
from omnibase_infra.services.observability.savings_estimation.config import (
    ConfigSavingsEstimation,
)
from omnibase_infra.services.observability.savings_estimation.consumer import (
    ServiceSavingsEstimator,
)

# Pricing manifest with a paid API model and a free local model.
_PRICING_DATA = {
    "schema_version": "1.0.0",
    "models": {
        "claude-opus-4-6": {
            "input_cost_per_1k": 0.015,
            "output_cost_per_1k": 0.075,
            "effective_date": "2026-02-01",
        },
        "qwen3-coder-30b-a3b": {
            "input_cost_per_1k": 0.0,
            "output_cost_per_1k": 0.0,
            "effective_date": "2026-03-19",
        },
    },
}

_T0 = 1000.0


@pytest.fixture
def pricing_table() -> ModelPricingTable:
    return ModelPricingTable.from_dict(_PRICING_DATA)


@pytest.fixture
def config() -> ConfigSavingsEstimation:
    return ConfigSavingsEstimation(
        grace_window_seconds=5.0,
        session_timeout_seconds=60.0,
        max_sessions=100,
        finalized_session_cache_size=1000,
    )


@pytest.fixture
def service(
    config: ConfigSavingsEstimation,
    pricing_table: ModelPricingTable,
) -> ServiceSavingsEstimator:
    return ServiceSavingsEstimator(config, pricing_table)


# -----------------------------------------------------------------------
# Full pipeline: all four event types → single estimate
# -----------------------------------------------------------------------


@pytest.mark.asyncio
async def test_full_pipeline_all_event_types(
    service: ServiceSavingsEstimator,
) -> None:
    """Ingest all four event types for one session and verify the output estimate.

    This is the canonical happy-path: LLM calls on a local model (Tier A savings),
    pattern injection (Tier B), validator catch (Tier B), then session-outcome
    triggers finalization after the grace window.
    """
    session_id = "e2e-full-pipeline"
    topics = service._config.consumed_topics

    with patch("time.monotonic", return_value=_T0):
        # 1. LLM call on a free local model → local routing savings
        service.ingest_event(
            topics[0],  # llm-call-completed
            {
                "session_id": session_id,
                "model_id": "qwen3-coder-30b-a3b",
                "prompt_tokens": 2000,
                "completion_tokens": 800,
            },
        )

        # 2. Second LLM call on a paid API model → no routing savings
        service.ingest_event(
            topics[0],
            {
                "session_id": session_id,
                "model_id": "claude-opus-4-6",
                "prompt_tokens": 500,
                "completion_tokens": 200,
            },
        )

        # 3. Pattern injection signal
        service.ingest_event(
            topics[2],  # hook-context-injected
            {
                "session_id": session_id,
                "tokens_injected": 400,
                "patterns_count": 3,
            },
        )

        # 4. Validator catch signal
        service.ingest_event(
            topics[3],  # validator-catch
            {
                "session_id": session_id,
                "validator_type": "pre_commit",
                "severity": "error",
            },
        )

        # 5. Session outcome finalizes
        service.ingest_event(
            topics[1],  # session-outcome
            {
                "session_id": session_id,
                "correlation_id": "corr-e2e",
                "treatment_group": "treatment",
            },
        )

    assert service.active_session_count == 1

    # Before grace window — nothing finalized
    with patch("time.monotonic", return_value=_T0 + 2.0):
        results = await service.finalize_ready_sessions()
    assert len(results) == 0

    # After grace window — estimate produced
    with patch("time.monotonic", return_value=_T0 + 6.0):
        results = await service.finalize_ready_sessions()

    assert len(results) == 1
    estimate = results[0]

    # --- Structural validation of the savings-estimated.v1 event ---
    assert estimate["session_id"] == session_id
    assert estimate["correlation_id"] == "corr-e2e"
    assert estimate["schema_version"] == "1.0"
    assert estimate["source_event_id"] == f"savings-{session_id}-v1.0"
    assert estimate["estimation_method"] == "tiered_attribution_v1"
    assert estimate["treatment_group"] == "treatment"

    # Actual cost fields
    assert isinstance(estimate["actual_total_tokens"], int)
    assert estimate["actual_total_tokens"] == 2000 + 800 + 500 + 200  # 3500
    assert isinstance(estimate["actual_cost_usd"], float)
    assert estimate["actual_cost_usd"] > 0  # paid model contributes cost

    # Savings fields
    assert isinstance(estimate["direct_savings_usd"], float)
    assert estimate["direct_savings_usd"] > 0  # local routing savings
    assert isinstance(estimate["estimated_total_savings_usd"], float)
    assert estimate["estimated_total_savings_usd"] >= estimate["direct_savings_usd"]
    assert isinstance(estimate["direct_tokens_saved"], int)
    assert estimate["direct_tokens_saved"] > 0
    assert isinstance(estimate["estimated_total_tokens_saved"], int)

    # Confidence
    assert estimate["direct_confidence"] == 1.0  # Tier A is always 1.0

    # Categories — at least local_routing, pattern_injection, validator_catches
    categories = estimate["categories"]
    assert isinstance(categories, list)
    category_names = {c["category"] for c in categories}
    assert "local_routing" in category_names
    assert "pattern_injection" in category_names
    assert "validator_catches" in category_names

    # Each category has required fields
    for cat in categories:
        assert "category" in cat
        assert "tier" in cat
        assert "tokens_saved" in cat
        assert "cost_saved_usd" in cat
        assert "confidence" in cat
        assert "method" in cat
        assert "evidence" in cat

    # Local routing should be Tier A (direct)
    local_routing = next(c for c in categories if c["category"] == "local_routing")
    assert local_routing["tier"] == "direct"
    assert local_routing["confidence"] == 1.0
    assert local_routing["tokens_saved"] == 2800  # 2000 + 800 from local model

    # Pattern injection should be Tier B (heuristic)
    pattern_cat = next(c for c in categories if c["category"] == "pattern_injection")
    assert pattern_cat["tier"] == "heuristic"
    assert pattern_cat["confidence"] == 0.8
    assert pattern_cat["tokens_saved"] > 0

    # Validator catches should be Tier B
    validator_cat = next(c for c in categories if c["category"] == "validator_catches")
    assert validator_cat["tier"] == "heuristic"
    assert validator_cat["tokens_saved"] > 0

    # Session is marked as finalized (idempotent)
    assert service.is_finalized(session_id)
    assert service.active_session_count == 0


# -----------------------------------------------------------------------
# Multi-session pipeline: concurrent sessions finalize independently
# -----------------------------------------------------------------------


@pytest.mark.asyncio
async def test_multi_session_pipeline(
    service: ServiceSavingsEstimator,
) -> None:
    """Two concurrent sessions with different signal mixes finalize independently."""
    topics = service._config.consumed_topics

    with patch("time.monotonic", return_value=_T0):
        # Session A: local model only
        service.ingest_event(
            topics[0],
            {
                "session_id": "sess-a",
                "model_id": "qwen3-coder-30b-a3b",
                "prompt_tokens": 1000,
                "completion_tokens": 500,
            },
        )
        service.ingest_event(
            topics[1],
            {"session_id": "sess-a", "correlation_id": "corr-a"},
        )

        # Session B: paid model + injection
        service.ingest_event(
            topics[0],
            {
                "session_id": "sess-b",
                "model_id": "claude-opus-4-6",
                "prompt_tokens": 800,
                "completion_tokens": 300,
            },
        )
        service.ingest_event(
            topics[2],
            {
                "session_id": "sess-b",
                "tokens_injected": 200,
                "patterns_count": 1,
            },
        )
        service.ingest_event(
            topics[1],
            {"session_id": "sess-b", "correlation_id": "corr-b"},
        )

    assert service.active_session_count == 2

    # Finalize both after grace window
    with patch("time.monotonic", return_value=_T0 + 6.0):
        results = await service.finalize_ready_sessions()

    assert len(results) == 2
    result_ids = {str(r["session_id"]) for r in results}
    assert result_ids == {"sess-a", "sess-b"}

    # Session A has local routing savings (local model)
    sess_a = next(r for r in results if r["session_id"] == "sess-a")
    assert float(str(sess_a["direct_savings_usd"])) > 0

    # Session B has no local routing savings but has pattern injection
    sess_b = next(r for r in results if r["session_id"] == "sess-b")
    categories_b = sess_b["categories"]
    assert isinstance(categories_b, list)
    cat_names_b = {c["category"] for c in categories_b}
    assert "pattern_injection" in cat_names_b

    assert service.active_session_count == 0


# -----------------------------------------------------------------------
# Timeout finalization without session-outcome
# -----------------------------------------------------------------------


@pytest.mark.asyncio
async def test_timeout_finalization_produces_estimate(
    service: ServiceSavingsEstimator,
) -> None:
    """Sessions without outcome still finalize after timeout with valid estimate."""
    topics = service._config.consumed_topics

    with patch("time.monotonic", return_value=_T0):
        service.ingest_event(
            topics[0],
            {
                "session_id": "sess-orphan",
                "model_id": "qwen3-coder-30b-a3b",
                "prompt_tokens": 500,
                "completion_tokens": 200,
            },
        )
        # No session-outcome sent

    # Not yet timed out
    with patch("time.monotonic", return_value=_T0 + 30.0):
        results = await service.finalize_ready_sessions()
    assert len(results) == 0

    # Past timeout (60s)
    with patch("time.monotonic", return_value=_T0 + 61.0):
        results = await service.finalize_ready_sessions()

    assert len(results) == 1
    estimate = results[0]
    assert estimate["session_id"] == "sess-orphan"
    assert estimate["schema_version"] == "1.0"
    assert isinstance(estimate["categories"], list)
    assert float(str(estimate["direct_savings_usd"])) > 0  # local model → savings


# -----------------------------------------------------------------------
# Replay / idempotency: finalized session rejects duplicate events
# -----------------------------------------------------------------------


@pytest.mark.asyncio
async def test_replay_idempotency(
    service: ServiceSavingsEstimator,
) -> None:
    """After finalization, replayed events for the same session are discarded."""
    topics = service._config.consumed_topics

    with patch("time.monotonic", return_value=_T0):
        service.ingest_event(
            topics[0],
            {
                "session_id": "sess-replay",
                "model_id": "qwen3-coder-30b-a3b",
                "prompt_tokens": 100,
                "completion_tokens": 50,
            },
        )
        service.ingest_event(
            topics[1],
            {"session_id": "sess-replay"},
        )

    with patch("time.monotonic", return_value=_T0 + 6.0):
        results = await service.finalize_ready_sessions()
    assert len(results) == 1

    # Replay the same events — should be silently ignored
    service.ingest_event(
        topics[0],
        {
            "session_id": "sess-replay",
            "model_id": "qwen3-coder-30b-a3b",
            "prompt_tokens": 9999,
            "completion_tokens": 9999,
        },
    )
    assert service.active_session_count == 0
    assert service.is_finalized("sess-replay")

    # No new estimates produced
    with patch("time.monotonic", return_value=_T0 + 12.0):
        results = await service.finalize_ready_sessions()
    assert len(results) == 0


# -----------------------------------------------------------------------
# Completeness status reflects signal coverage
# -----------------------------------------------------------------------


@pytest.mark.asyncio
async def test_completeness_status_phase_limited(
    service: ServiceSavingsEstimator,
) -> None:
    """Estimate with no delegation or RAG signals reports phase_limited status."""
    topics = service._config.consumed_topics

    with patch("time.monotonic", return_value=_T0):
        service.ingest_event(
            topics[0],
            {
                "session_id": "sess-phase",
                "model_id": "qwen3-coder-30b-a3b",
                "prompt_tokens": 100,
                "completion_tokens": 50,
            },
        )
        service.ingest_event(
            topics[1],
            {"session_id": "sess-phase"},
        )

    with patch("time.monotonic", return_value=_T0 + 6.0):
        results = await service.finalize_ready_sessions()

    assert len(results) == 1
    assert results[0]["completeness_status"] == "phase_limited"


# -----------------------------------------------------------------------
# Source event ID is deterministic across runs
# -----------------------------------------------------------------------


@pytest.mark.asyncio
async def test_source_event_id_deterministic(
    service: ServiceSavingsEstimator,
) -> None:
    """source_event_id is derived from session_id + schema_version (deterministic)."""
    topics = service._config.consumed_topics

    with patch("time.monotonic", return_value=_T0):
        service.ingest_event(
            topics[0],
            {
                "session_id": "sess-det",
                "model_id": "qwen3-coder-30b-a3b",
                "prompt_tokens": 100,
                "completion_tokens": 50,
            },
        )
        service.ingest_event(
            topics[1],
            {"session_id": "sess-det"},
        )

    with patch("time.monotonic", return_value=_T0 + 6.0):
        results = await service.finalize_ready_sessions()

    assert results[0]["source_event_id"] == "savings-sess-det-v1.0"
