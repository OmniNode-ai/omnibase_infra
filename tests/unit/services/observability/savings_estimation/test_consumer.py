# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

# Copyright (c) 2026 OmniNode Team
"""Tests for ServiceSavingsEstimator consumer.

Covers: correlation, grace window, timeout, idempotency, missing signals.

Tracking:
    - OMN-5550: Create ServiceSavingsEstimator Kafka consumer
"""

from __future__ import annotations

from unittest.mock import patch

import pytest

from omnibase_infra.services.observability.savings_estimation.config import (
    ConfigSavingsEstimation,
)
from omnibase_infra.services.observability.savings_estimation.consumer import (
    ServiceSavingsEstimator,
)

# Base monotonic time for deterministic clock control in tests.
_T0 = 1000.0


@pytest.fixture
def config() -> ConfigSavingsEstimation:
    return ConfigSavingsEstimation(
        kafka_bootstrap_servers="localhost:19092",
        grace_window_seconds=5.0,
        session_timeout_seconds=60.0,
        max_sessions=100,
        finalized_session_cache_size=1000,
    )


@pytest.fixture
def service(config: ConfigSavingsEstimation) -> ServiceSavingsEstimator:
    return ServiceSavingsEstimator(config)


@pytest.mark.unit
@pytest.mark.asyncio
async def test_correlation_by_session_id(service: ServiceSavingsEstimator) -> None:
    """Events with the same session_id are correlated into one session."""
    service.ingest_event(
        "onex.evt.omniintelligence.llm-call-completed.v1",
        {
            "session_id": "sess-1",
            "model_id": "qwen3-coder-30b-a3b",
            "prompt_tokens": 1000,
            "completion_tokens": 500,
        },
    )
    service.ingest_event(
        "onex.evt.omniclaude.validator-catch.v1",
        {
            "session_id": "sess-1",
            "validator_type": "pre_commit",
            "severity": "error",
        },
    )

    assert service.active_session_count == 1


@pytest.mark.unit
@pytest.mark.asyncio
async def test_grace_window_finalization(service: ServiceSavingsEstimator) -> None:
    """Session finalizes after grace window elapses post session-outcome."""
    with patch("time.monotonic", return_value=_T0):
        service.ingest_event(
            "onex.evt.omniclaude.hook-context-injected.v1",
            {
                "session_id": "sess-2",
                "tokens_injected": 300,
                "patterns_count": 5,
            },
        )
        service.ingest_event(
            "onex.evt.omniclaude.session-outcome.v1",
            {"session_id": "sess-2", "correlation_id": "corr-2"},
        )

    # Before grace window (only 1s later, grace is 5s)
    with patch("time.monotonic", return_value=_T0 + 1.0):
        results = await service.finalize_ready_sessions()
    assert len(results) == 0

    # After grace window (6s later, grace is 5s)
    with patch("time.monotonic", return_value=_T0 + 6.0):
        results = await service.finalize_ready_sessions()
    assert len(results) == 1
    assert results[0]["session_id"] == "sess-2"
    assert results[0]["source_event_id"] == "savings-sess-2-v1.0"


@pytest.mark.unit
@pytest.mark.asyncio
async def test_grace_window_includes_late_events(
    service: ServiceSavingsEstimator,
) -> None:
    """Events arriving after outcome but within grace window are included."""
    with patch("time.monotonic", return_value=_T0):
        service.ingest_event(
            "onex.evt.omniclaude.hook-context-injected.v1",
            {
                "session_id": "sess-grace",
                "tokens_injected": 500,
                "patterns_count": 3,
            },
        )
        service.ingest_event(
            "onex.evt.omniclaude.session-outcome.v1",
            {"session_id": "sess-grace"},
        )

    # Late event within grace window (2s after outcome, grace is 5s)
    with patch("time.monotonic", return_value=_T0 + 2.0):
        service.ingest_event(
            "onex.evt.omniclaude.hook-context-injected.v1",
            {
                "session_id": "sess-grace",
                "tokens_injected": 300,
                "patterns_count": 2,
            },
        )

    # Finalize after grace window
    with patch("time.monotonic", return_value=_T0 + 6.0):
        results = await service.finalize_ready_sessions()
    assert len(results) == 1
    # The session should have both injection signals
    assert results[0]["session_id"] == "sess-grace"


@pytest.mark.unit
@pytest.mark.asyncio
async def test_session_timeout(service: ServiceSavingsEstimator) -> None:
    """Sessions without outcome finalize after timeout."""
    service.ingest_event(
        "onex.evt.omniclaude.hook-context-injected.v1",
        {
            "session_id": "sess-3",
            "tokens_injected": 200,
            "patterns_count": 3,
        },
    )
    # Pin created_at so we can control timeout math deterministically
    service._sessions["sess-3"].created_at = _T0

    # Not yet timed out (30s, timeout is 60s)
    with patch("time.monotonic", return_value=_T0 + 30.0):
        results = await service.finalize_ready_sessions()
    assert len(results) == 0

    # Past timeout (61s, timeout is 60s)
    with patch("time.monotonic", return_value=_T0 + 61.0):
        results = await service.finalize_ready_sessions()
    assert len(results) == 1
    assert results[0]["session_id"] == "sess-3"


@pytest.mark.unit
@pytest.mark.asyncio
async def test_idempotency_finalized_sessions(
    service: ServiceSavingsEstimator,
) -> None:
    """Finalized sessions are not re-processed on subsequent events."""
    with patch("time.monotonic", return_value=_T0):
        service.ingest_event(
            "onex.evt.omniclaude.hook-context-injected.v1",
            {
                "session_id": "sess-4",
                "tokens_injected": 500,
                "patterns_count": 5,
            },
        )
        service.ingest_event(
            "onex.evt.omniclaude.session-outcome.v1",
            {"session_id": "sess-4"},
        )

    with patch("time.monotonic", return_value=_T0 + 6.0):
        results = await service.finalize_ready_sessions()
    assert len(results) == 1

    # Ingest another event for same session -- should be ignored
    service.ingest_event(
        "onex.evt.omniclaude.hook-context-injected.v1",
        {
            "session_id": "sess-4",
            "tokens_injected": 200,
            "patterns_count": 2,
        },
    )

    assert service.active_session_count == 0
    assert service.is_finalized("sess-4")


@pytest.mark.unit
@pytest.mark.asyncio
async def test_missing_signals_returns_none(
    service: ServiceSavingsEstimator,
) -> None:
    """Sessions with no signals at all do not produce estimates."""
    with patch("time.monotonic", return_value=_T0):
        service.ingest_event(
            "onex.evt.omniclaude.session-outcome.v1",
            {"session_id": "sess-5"},
        )

    with patch("time.monotonic", return_value=_T0 + 6.0):
        results = await service.finalize_ready_sessions()
    assert len(results) == 0


@pytest.mark.unit
@pytest.mark.asyncio
async def test_partial_signals_produce_estimate(
    service: ServiceSavingsEstimator,
) -> None:
    """Sessions with only injection signals (no LLM calls) still produce estimates."""
    with patch("time.monotonic", return_value=_T0):
        service.ingest_event(
            "onex.evt.omniclaude.hook-context-injected.v1",
            {
                "session_id": "sess-partial",
                "tokens_injected": 500,
                "patterns_count": 3,
            },
        )
        service.ingest_event(
            "onex.evt.omniclaude.session-outcome.v1",
            {"session_id": "sess-partial"},
        )

    with patch("time.monotonic", return_value=_T0 + 6.0):
        results = await service.finalize_ready_sessions()
    assert len(results) == 1
    assert results[0]["session_id"] == "sess-partial"


@pytest.mark.unit
@pytest.mark.asyncio
async def test_deterministic_source_event_id(
    service: ServiceSavingsEstimator,
) -> None:
    """source_event_id is deterministic from session_id and schema_version."""
    with patch("time.monotonic", return_value=_T0):
        service.ingest_event(
            "onex.evt.omniclaude.hook-context-injected.v1",
            {
                "session_id": "sess-6",
                "tokens_injected": 200,
                "patterns_count": 2,
            },
        )
        service.ingest_event(
            "onex.evt.omniclaude.session-outcome.v1",
            {"session_id": "sess-6"},
        )

    with patch("time.monotonic", return_value=_T0 + 6.0):
        results = await service.finalize_ready_sessions()
    assert results[0]["source_event_id"] == "savings-sess-6-v1.0"


@pytest.mark.unit
@pytest.mark.asyncio
async def test_lru_eviction() -> None:
    """Buffer evicts oldest session when max_sessions is reached."""
    # Use model_construct to bypass validation for testing with small max_sessions
    config = ConfigSavingsEstimation.model_construct(
        kafka_bootstrap_servers="localhost:19092",
        kafka_group_id="savings-estimation",
        consumed_topics=[],
        produce_topic="",
        auto_offset_reset="earliest",
        batch_size=100,
        batch_timeout_ms=1000,
        grace_window_seconds=5.0,
        session_timeout_seconds=3600.0,
        max_sessions=2,
        finalized_session_cache_size=1000,
        schema_version="1.0",
    )
    service = ServiceSavingsEstimator(config)

    service.ingest_event(
        "onex.evt.omniclaude.hook-context-injected.v1",
        {
            "session_id": "old",
            "tokens_injected": 100,
            "patterns_count": 1,
        },
    )
    service.ingest_event(
        "onex.evt.omniclaude.hook-context-injected.v1",
        {
            "session_id": "mid",
            "tokens_injected": 100,
            "patterns_count": 1,
        },
    )
    # This should evict "old"
    service.ingest_event(
        "onex.evt.omniclaude.hook-context-injected.v1",
        {
            "session_id": "new",
            "tokens_injected": 100,
            "patterns_count": 1,
        },
    )

    assert service.active_session_count == 2
    assert "old" not in service._sessions


@pytest.mark.unit
@pytest.mark.asyncio
async def test_multiple_sessions_independent(
    service: ServiceSavingsEstimator,
) -> None:
    """Multiple sessions are tracked independently."""
    with patch("time.monotonic", return_value=_T0):
        service.ingest_event(
            "onex.evt.omniclaude.hook-context-injected.v1",
            {
                "session_id": "sess-a",
                "tokens_injected": 100,
                "patterns_count": 2,
            },
        )
        service.ingest_event(
            "onex.evt.omniclaude.hook-context-injected.v1",
            {
                "session_id": "sess-b",
                "tokens_injected": 200,
                "patterns_count": 4,
            },
        )

    assert service.active_session_count == 2

    # Only finalize sess-a
    with patch("time.monotonic", return_value=_T0):
        service.ingest_event(
            "onex.evt.omniclaude.session-outcome.v1",
            {"session_id": "sess-a"},
        )

    with patch("time.monotonic", return_value=_T0 + 6.0):
        results = await service.finalize_ready_sessions()
    assert len(results) == 1
    assert results[0]["session_id"] == "sess-a"
    assert service.active_session_count == 1


@pytest.mark.unit
@pytest.mark.asyncio
async def test_empty_session_id_ignored(service: ServiceSavingsEstimator) -> None:
    """Events without session_id are silently ignored."""
    service.ingest_event(
        "onex.evt.omniintelligence.llm-call-completed.v1",
        {
            "model_id": "qwen3-coder-30b-a3b",
            "prompt_tokens": 100,
            "completion_tokens": 50,
        },
    )
    assert service.active_session_count == 0


@pytest.mark.unit
@pytest.mark.asyncio
async def test_llm_only_session_produces_zero_savings_estimate(
    service: ServiceSavingsEstimator,
) -> None:
    """Sessions with only LLM calls (no injections) produce a zero-savings estimate."""
    with patch("time.monotonic", return_value=_T0):
        service.ingest_event(
            "onex.evt.omniintelligence.llm-call-completed.v1",
            {
                "session_id": "sess-llm-only",
                "model_id": "claude-opus-4-6",
                "prompt_tokens": 1000,
                "completion_tokens": 500,
            },
        )
        service.ingest_event(
            "onex.evt.omniclaude.session-outcome.v1",
            {"session_id": "sess-llm-only"},
        )

    with patch("time.monotonic", return_value=_T0 + 6.0):
        results = await service.finalize_ready_sessions()
    assert len(results) == 1
    assert results[0]["session_id"] == "sess-llm-only"
    assert results[0]["direct_tokens_saved"] == 0
    assert results[0]["direct_savings_usd"] == 0.0
