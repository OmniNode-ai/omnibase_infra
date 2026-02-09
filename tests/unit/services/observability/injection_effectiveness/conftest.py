# SPDX-License-Identifier: MIT
# Copyright (c) 2026 OmniNode Team
"""Shared fixtures for injection effectiveness unit tests.

Provides mock asyncpg.Pool, connection, and test data factories.

Related Tickets:
    - OMN-2078: Golden path: injection metrics + ledger storage
"""

from __future__ import annotations

from datetime import UTC, datetime
from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4

import pytest

from omnibase_infra.services.observability.injection_effectiveness.models import (
    ModelContextUtilizationEvent,
    ModelLatencyBreakdownEvent,
    ModelPatternUtilization,
)


@pytest.fixture
def mock_pool() -> MagicMock:
    """Create a mock asyncpg.Pool with connection context manager."""
    pool = MagicMock()
    conn = AsyncMock()
    conn.fetchrow = AsyncMock(return_value=None)
    conn.fetchval = AsyncMock(return_value=0)
    conn.fetch = AsyncMock(return_value=[])
    conn.execute = AsyncMock()
    conn.executemany = AsyncMock()

    # Support async context manager for conn.transaction()
    tx = AsyncMock()
    tx.__aenter__ = AsyncMock(return_value=tx)
    tx.__aexit__ = AsyncMock(return_value=False)
    conn.transaction = MagicMock(return_value=tx)

    # Support async context manager for pool.acquire()
    ctx = AsyncMock()
    ctx.__aenter__ = AsyncMock(return_value=conn)
    ctx.__aexit__ = AsyncMock(return_value=False)
    pool.acquire = MagicMock(return_value=ctx)

    # Attach connection for direct access in tests
    pool._test_conn = conn
    return pool


@pytest.fixture
def sample_session_id():
    """Generate a sample session ID."""
    return uuid4()


@pytest.fixture
def sample_correlation_id():
    """Generate a sample correlation ID."""
    return uuid4()


@pytest.fixture
def sample_context_utilization_event(sample_session_id, sample_correlation_id):
    """Create a sample context utilization event."""
    return ModelContextUtilizationEvent(
        session_id=sample_session_id,
        correlation_id=sample_correlation_id,
        cohort="treatment",
        cohort_identity_type="session_id",
        total_injected_tokens=1500,
        patterns_injected=3,
        utilization_score=0.85,
        utilization_method="identifier_match",
        injected_identifiers_count=10,
        reused_identifiers_count=7,
        pattern_utilizations=(
            ModelPatternUtilization(
                pattern_id=uuid4(),
                utilization_score=0.9,
                utilization_method="identifier_match",
            ),
            ModelPatternUtilization(
                pattern_id=uuid4(),
                utilization_score=0.3,
                utilization_method="identifier_match",
            ),
        ),
        created_at=datetime.now(UTC),
    )


@pytest.fixture
def sample_latency_event(sample_session_id, sample_correlation_id):
    """Create a sample latency breakdown event."""
    return ModelLatencyBreakdownEvent(
        session_id=sample_session_id,
        correlation_id=sample_correlation_id,
        prompt_id=uuid4(),
        cohort="treatment",
        cache_hit=False,
        routing_latency_ms=15,
        retrieval_latency_ms=120,
        injection_latency_ms=8,
        user_latency_ms=350,
        emitted_at=datetime.now(UTC),
        created_at=datetime.now(UTC),
    )


def make_effectiveness_row(
    session_id=None,
    correlation_id=None,
    utilization_score=0.75,
    cohort="treatment",
):
    """Factory for mock asyncpg Row dicts for injection_effectiveness table."""
    now = datetime.now(UTC)
    return {
        "session_id": session_id or uuid4(),
        "correlation_id": correlation_id or uuid4(),
        "realm": None,
        "runtime_id": None,
        "routing_path": None,
        "cohort": cohort,
        "cohort_identity_type": "session_id",
        "total_injected_tokens": 1500,
        "patterns_injected": 3,
        "utilization_score": utilization_score,
        "utilization_method": "identifier_match",
        "injected_identifiers_count": 10,
        "reused_identifiers_count": 7,
        "agent_match_score": 0.9,
        "expected_agent": "coder",
        "actual_agent": "coder",
        "user_visible_latency_ms": 350,
        "created_at": now,
        "updated_at": now,
    }


def make_latency_row(session_id=None, prompt_id=None):
    """Factory for mock asyncpg Row dicts for latency_breakdowns table."""
    now = datetime.now(UTC)
    return {
        "id": uuid4(),
        "session_id": session_id or uuid4(),
        "prompt_id": prompt_id or uuid4(),
        "cohort": "treatment",
        "cache_hit": False,
        "routing_latency_ms": 15,
        "retrieval_latency_ms": 120,
        "injection_latency_ms": 8,
        "user_latency_ms": 350,
        "emitted_at": now,
        "created_at": now,
    }


def make_pattern_hit_rate_row(pattern_id=None, sample_count=25):
    """Factory for mock asyncpg Row dicts for pattern_hit_rates table."""
    now = datetime.now(UTC)
    return {
        "id": uuid4(),
        "pattern_id": pattern_id or uuid4(),
        "domain_id": None,
        "utilization_method": "identifier_match",
        "utilization_score": 0.72,
        "hit_count": 18,
        "miss_count": 7,
        "sample_count": sample_count,
        "confidence": 0.72 if sample_count >= 20 else None,
        "created_at": now,
        "updated_at": now,
    }
