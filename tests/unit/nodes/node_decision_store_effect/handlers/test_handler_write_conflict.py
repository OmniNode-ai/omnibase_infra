# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Unit tests for HandlerWriteConflict — idempotent conflict-pair insert handler.

Tests validate:
  - Successful insert returns ModelBackendResult success=True
  - ON CONFLICT DO NOTHING (idempotent — second call for same pair succeeds)
  - Pair ordering is normalised (min < max) before insert
  - TimeoutError / InfraConnectionError / InfraAuthenticationError handling

Related Tickets:
    - OMN-2765: NodeDecisionStoreEffect implementation
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock
from uuid import UUID, uuid4

import pytest

from omnibase_infra.enums import EnumPostgresErrorCode
from omnibase_infra.errors import (
    InfraAuthenticationError,
    InfraConnectionError,
)
from omnibase_infra.nodes.node_decision_store_effect.handlers.handler_write_conflict import (
    HandlerWriteConflict,
    _ordered_pair,
)
from omnibase_infra.nodes.node_decision_store_effect.models.model_payload_write_conflict import (
    ModelPayloadWriteConflict,
)

# =============================================================================
# Mock pool factory
# =============================================================================


def create_mock_pool_with_conn(conn: AsyncMock) -> MagicMock:
    """Create a mock pool returning a specific connection."""
    pool = MagicMock()
    mock_acquire = AsyncMock()
    mock_acquire.__aenter__ = AsyncMock(return_value=conn)
    mock_acquire.__aexit__ = AsyncMock(return_value=None)
    pool.acquire = MagicMock(return_value=mock_acquire)
    return pool


def make_conflict_payload(
    *,
    min_id: UUID | None = None,
    max_id: UUID | None = None,
    structural_confidence: float = 0.9,
    final_severity: str = "HIGH",
) -> ModelPayloadWriteConflict:
    """Create a test ModelPayloadWriteConflict."""
    a = min_id or uuid4()
    b = max_id or uuid4()
    # Ensure min < max
    lo, hi = (a, b) if a < b else (b, a)
    return ModelPayloadWriteConflict(
        correlation_id=uuid4(),
        decision_min_id=lo,
        decision_max_id=hi,
        structural_confidence=structural_confidence,
        final_severity=final_severity,  # type: ignore[arg-type]
    )


# =============================================================================
# Tests: _ordered_pair helper
# =============================================================================


class TestOrderedPair:
    """Tests for the _ordered_pair utility."""

    def test_already_ordered_pair_unchanged(self) -> None:
        """If a < b already, pair is returned as-is."""
        a = UUID("00000000-0000-0000-0000-000000000001")
        b = UUID("00000000-0000-0000-0000-000000000002")
        lo, hi = _ordered_pair(a, b)
        assert lo == a
        assert hi == b

    def test_reversed_pair_is_normalised(self) -> None:
        """If a > b, pair is returned as (b, a) with min first."""
        a = UUID("00000000-0000-0000-0000-000000000002")
        b = UUID("00000000-0000-0000-0000-000000000001")
        lo, hi = _ordered_pair(a, b)
        assert lo == b
        assert hi == a
        assert lo < hi


# =============================================================================
# Tests: HandlerWriteConflict success
# =============================================================================


class TestHandlerWriteConflictSuccess:
    """Test successful conflict-pair inserts."""

    @pytest.mark.asyncio
    async def test_successful_insert_returns_success(self) -> None:
        """Successful insert returns ModelBackendResult with success=True."""
        conn = AsyncMock()
        conn.execute = AsyncMock(return_value="INSERT 0 1")
        pool = create_mock_pool_with_conn(conn)

        handler = HandlerWriteConflict(pool)
        payload = make_conflict_payload()
        correlation_id = uuid4()

        result = await handler.handle(payload, correlation_id)

        assert result.success is True
        assert result.backend_id == "postgres"
        assert result.correlation_id == correlation_id
        assert result.error is None
        assert result.error_code is None
        assert result.duration_ms >= 0

    @pytest.mark.asyncio
    async def test_idempotent_insert_returns_success(self) -> None:
        """ON CONFLICT DO NOTHING (INSERT 0 0) still returns success=True."""
        conn = AsyncMock()
        conn.execute = AsyncMock(return_value="INSERT 0 0")
        pool = create_mock_pool_with_conn(conn)

        handler = HandlerWriteConflict(pool)
        payload = make_conflict_payload()
        result = await handler.handle(payload, uuid4())

        assert result.success is True
        assert result.backend_id == "postgres"

    @pytest.mark.asyncio
    async def test_pair_ordering_applied_before_insert(self) -> None:
        """Handler normalises pair order to (min, max) before insert."""
        conn = AsyncMock()
        conn.execute = AsyncMock(return_value="INSERT 0 1")
        pool = create_mock_pool_with_conn(conn)

        handler = HandlerWriteConflict(pool)

        a = UUID("00000000-0000-0000-0000-000000000002")
        b = UUID("00000000-0000-0000-0000-000000000001")
        # Construct payload with a > b (reversed order)
        payload = ModelPayloadWriteConflict(
            correlation_id=uuid4(),
            decision_min_id=b,  # already ordered correctly for payload
            decision_max_id=a,
            structural_confidence=0.7,
            final_severity="MEDIUM",
        )
        result = await handler.handle(payload, uuid4())

        assert result.success is True
        # Verify the SQL was called with min < max
        call_args = conn.execute.call_args[0]  # positional args
        min_arg: UUID = call_args[1]
        max_arg: UUID = call_args[2]
        assert min_arg < max_arg


# =============================================================================
# Tests: HandlerWriteConflict error handling
# =============================================================================


class TestHandlerWriteConflictErrors:
    """Test error handling for HandlerWriteConflict."""

    @pytest.mark.asyncio
    async def test_timeout_error_returns_timeout_code(self) -> None:
        """TimeoutError returns success=False with TIMEOUT_ERROR code."""
        conn = AsyncMock()
        conn.execute = AsyncMock(side_effect=TimeoutError("timed out"))
        pool = create_mock_pool_with_conn(conn)

        handler = HandlerWriteConflict(pool)
        result = await handler.handle(make_conflict_payload(), uuid4())

        assert result.success is False
        assert result.error_code == EnumPostgresErrorCode.TIMEOUT_ERROR
        assert result.backend_id == "postgres"

    @pytest.mark.asyncio
    async def test_connection_error_returns_connection_code(self) -> None:
        """InfraConnectionError returns success=False with CONNECTION_ERROR code."""
        conn = AsyncMock()
        conn.execute = AsyncMock(side_effect=InfraConnectionError("refused"))
        pool = create_mock_pool_with_conn(conn)

        handler = HandlerWriteConflict(pool)
        result = await handler.handle(make_conflict_payload(), uuid4())

        assert result.success is False
        assert result.error_code == EnumPostgresErrorCode.CONNECTION_ERROR
        assert result.backend_id == "postgres"

    @pytest.mark.asyncio
    async def test_auth_error_returns_auth_code(self) -> None:
        """InfraAuthenticationError returns success=False with AUTH_ERROR code."""
        conn = AsyncMock()
        conn.execute = AsyncMock(side_effect=InfraAuthenticationError("bad creds"))
        pool = create_mock_pool_with_conn(conn)

        handler = HandlerWriteConflict(pool)
        result = await handler.handle(make_conflict_payload(), uuid4())

        assert result.success is False
        assert result.error_code == EnumPostgresErrorCode.AUTH_ERROR
        assert result.backend_id == "postgres"

    @pytest.mark.asyncio
    async def test_generic_exception_returns_unknown_code(self) -> None:
        """Unexpected exception returns success=False with UNKNOWN_ERROR code."""
        conn = AsyncMock()
        conn.execute = AsyncMock(side_effect=RuntimeError("boom"))
        pool = create_mock_pool_with_conn(conn)

        handler = HandlerWriteConflict(pool)
        result = await handler.handle(make_conflict_payload(), uuid4())

        assert result.success is False
        assert result.error_code == EnumPostgresErrorCode.UNKNOWN_ERROR
        assert result.backend_id == "postgres"


__all__: list[str] = [
    "TestOrderedPair",
    "TestHandlerWriteConflictSuccess",
    "TestHandlerWriteConflictErrors",
]
