# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Unit tests for HandlerWriteDecision — two-stage decision write handler.

Tests validate:
Stage 1:
  - Successful upsert returns ModelBackendResult success=True
  - scope_services normalisation (sort + lowercase)
  - scope_domain normalisation + validation against ALLOWED_DOMAINS
  - created_at future-timestamp rejection (> 5 min)
  - superseded_by forces status=SUPERSEDED
  - TimeoutError / InfraConnectionError / InfraAuthenticationError handling

Stage 2:
  - structural_confidence() pure function coverage
  - Conflict pairs >= 0.3 written; pairs < 0.3 skipped
  - ACTIVE invariant: demote to PROPOSED when confidence >= 0.9 and no dismissal
  - ACTIVE invariant: no demotion when dismissed conflict exists

Related Tickets:
    - OMN-2765: NodeDecisionStoreEffect implementation
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import UUID, uuid4

import pytest

from omnibase_infra.enums import EnumPostgresErrorCode
from omnibase_infra.errors import (
    InfraAuthenticationError,
    InfraConnectionError,
)
from omnibase_infra.nodes.node_decision_store_effect.handlers.handler_write_decision import (
    ACTIVE_INVARIANT_THRESHOLD,
    ALLOWED_DOMAINS,
    CONFLICT_WRITE_THRESHOLD,
    DecisionScopeKey,
    HandlerWriteDecision,
    structural_confidence,
)
from omnibase_infra.nodes.node_decision_store_effect.models.model_payload_write_decision import (
    ModelPayloadWriteDecision,
)

# Fixed test time for deterministic testing
TEST_NOW = datetime(2026, 2, 25, 12, 0, 0, tzinfo=UTC)


# =============================================================================
# Mock pool factory
# =============================================================================


def create_mock_pool() -> MagicMock:
    """Create a mock asyncpg pool with acquire() context manager support.

    Returns a pool where each acquire() produces a fresh AsyncMock connection
    with fetchval, fetchrow, fetch, fetchrow, execute methods.
    """
    pool = MagicMock()

    def make_conn() -> AsyncMock:
        conn = AsyncMock()
        conn.fetchval = AsyncMock()
        conn.fetchrow = AsyncMock()
        conn.fetch = AsyncMock(return_value=[])
        conn.execute = AsyncMock()

        # Transaction context manager
        txn = AsyncMock()
        txn.__aenter__ = AsyncMock(return_value=None)
        txn.__aexit__ = AsyncMock(return_value=None)
        conn.transaction = MagicMock(return_value=txn)

        return conn

    # Create two distinct connections for Stage 1 and Stage 2
    conn1 = make_conn()
    conn2 = make_conn()
    _conns = [conn1, conn2]
    _idx = [0]

    def acquire_side_effect() -> Any:
        idx = _idx[0] % len(_conns)
        _idx[0] += 1
        acm = AsyncMock()
        acm.__aenter__ = AsyncMock(return_value=_conns[idx])
        acm.__aexit__ = AsyncMock(return_value=None)
        return acm

    pool.acquire = MagicMock(side_effect=acquire_side_effect)
    return pool


def _stage1_conn(pool: MagicMock) -> AsyncMock:
    """Return the Stage 1 connection from the pool."""
    # Reset side_effect to inspect the first created conn
    return (
        pool.acquire.side_effect.__self__
        if hasattr(pool.acquire.side_effect, "__self__")
        else MagicMock()
    )


# =============================================================================
# Payload factory
# =============================================================================


def make_payload(
    *,
    scope_domain: str = "infra",
    scope_services: list[str] | None = None,
    scope_layer: str = "architecture",
    status: str = "ACTIVE",
    superseded_by: UUID | None = None,
    created_at: datetime | None = None,
    decision_id: UUID | None = None,
) -> ModelPayloadWriteDecision:
    """Create a test ModelPayloadWriteDecision."""
    return ModelPayloadWriteDecision(
        correlation_id=uuid4(),
        decision_id=decision_id or uuid4(),
        title="Test decision",
        decision_type="DESIGN_PATTERN",
        status=status,  # type: ignore[arg-type]
        scope_domain=scope_domain,
        scope_services=scope_services if scope_services is not None else [],
        scope_layer=scope_layer,  # type: ignore[arg-type]
        rationale="Testing",
        alternatives=[],
        tags=[],
        source="manual",
        epic_id=None,
        supersedes=[],
        superseded_by=superseded_by,
        created_at=created_at or TEST_NOW,
        created_by="test@example.com",
    )


def _configure_stage1_success(conn: AsyncMock, was_insert: bool = True) -> None:
    """Configure conn to return successful Stage 1 responses."""
    # SQL_NOW response
    conn.fetchval.return_value = TEST_NOW
    # SQL_FETCH_OLD_STATUS — no prior row
    # SQL_UPSERT_DECISION response
    conn.fetchrow.return_value = {
        "was_insert": was_insert,
        "new_status": "ACTIVE",
    }


# =============================================================================
# Tests: structural_confidence pure function
# =============================================================================


class TestStructuralConfidence:
    """Tests for the structural_confidence() pure function."""

    def test_different_domains_returns_zero(self) -> None:
        """Different domains always return 0.0."""
        score = structural_confidence(
            DecisionScopeKey("infra", "architecture", []),
            DecisionScopeKey("auth", "architecture", []),
        )
        assert score == 0.0

    def test_same_domain_different_layer_returns_0_4(self) -> None:
        """Same domain, different layer returns 0.4."""
        score = structural_confidence(
            DecisionScopeKey("infra", "architecture", ["svc-a"]),
            DecisionScopeKey("infra", "design", ["svc-a"]),
        )
        assert score == 0.4

    def test_same_domain_layer_both_empty_services_returns_0_9(self) -> None:
        """Same domain + layer, both service lists empty returns 0.9."""
        score = structural_confidence(
            DecisionScopeKey("infra", "architecture", []),
            DecisionScopeKey("infra", "architecture", []),
        )
        assert score == 0.9

    def test_same_domain_layer_one_empty_returns_0_8(self) -> None:
        """Same domain + layer, one side empty returns 0.8."""
        score = structural_confidence(
            DecisionScopeKey("infra", "architecture", ["svc-a"]),
            DecisionScopeKey("infra", "architecture", []),
        )
        assert score == 0.8

    def test_same_domain_layer_identical_services_returns_1_0(self) -> None:
        """Same domain + layer + identical non-empty service sets returns 1.0."""
        score = structural_confidence(
            DecisionScopeKey("infra", "architecture", ["svc-a", "svc-b"]),
            DecisionScopeKey("infra", "architecture", ["svc-b", "svc-a"]),
        )
        assert score == 1.0

    def test_same_domain_layer_overlapping_services_returns_0_7(self) -> None:
        """Same domain + layer, overlapping non-empty service sets returns 0.7."""
        score = structural_confidence(
            DecisionScopeKey("infra", "architecture", ["svc-a", "svc-b"]),
            DecisionScopeKey("infra", "architecture", ["svc-b", "svc-c"]),
        )
        assert score == 0.7

    def test_same_domain_layer_disjoint_services_returns_0_3(self) -> None:
        """Same domain + layer, disjoint non-empty service sets returns 0.3."""
        score = structural_confidence(
            DecisionScopeKey("infra", "architecture", ["svc-a"]),
            DecisionScopeKey("infra", "architecture", ["svc-b"]),
        )
        assert score == 0.3

    def test_constants_are_correct(self) -> None:
        """Verify exported constants match spec."""
        assert CONFLICT_WRITE_THRESHOLD == 0.3
        assert ACTIVE_INVARIANT_THRESHOLD == 0.9


# =============================================================================
# Tests: HandlerWriteDecision Stage 1
# =============================================================================


class TestHandlerWriteDecisionStage1Success:
    """Test successful Stage 1 upserts."""

    @pytest.mark.asyncio
    async def test_successful_insert_returns_success(self) -> None:
        """Successful new decision insert returns success=True."""
        pool = create_mock_pool()
        # Stage 1 conn is the first acquired
        conn1 = (
            pool.acquire.side_effect().__aenter__.return_value if False else MagicMock()
        )

        # Simpler approach: patch _run_two_stage to verify it's called
        handler = HandlerWriteDecision.__new__(HandlerWriteDecision)
        handler._pool = MagicMock()

        mock_acquire = AsyncMock()
        mock_conn = AsyncMock()
        mock_conn.fetchval = AsyncMock(return_value=TEST_NOW.replace(tzinfo=None))
        mock_conn.fetchrow = AsyncMock(
            side_effect=[
                None,  # SQL_FETCH_OLD_STATUS — no prior row
                {"was_insert": True, "new_status": "ACTIVE"},  # SQL_UPSERT_DECISION
            ]
        )
        mock_conn.fetch = AsyncMock(return_value=[])  # Stage 2 active decisions
        txn = AsyncMock()
        txn.__aenter__ = AsyncMock(return_value=None)
        txn.__aexit__ = AsyncMock(return_value=None)
        mock_conn.transaction = MagicMock(return_value=txn)
        mock_acquire.__aenter__ = AsyncMock(return_value=mock_conn)
        mock_acquire.__aexit__ = AsyncMock(return_value=None)
        handler._pool.acquire = MagicMock(return_value=mock_acquire)

        payload = make_payload()
        correlation_id = uuid4()

        result = await handler.handle(payload, correlation_id)

        assert result.success is True
        assert result.backend_id == "postgres"
        assert result.correlation_id == correlation_id
        assert result.error is None
        assert result.error_code is None

    @pytest.mark.asyncio
    async def test_scope_services_normalized_before_insert(self) -> None:
        """scope_services are sorted and lowercased before DB insert."""
        handler = HandlerWriteDecision.__new__(HandlerWriteDecision)
        handler._pool = MagicMock()

        mock_acquire = AsyncMock()
        mock_conn = AsyncMock()
        mock_conn.fetchval = AsyncMock(return_value=TEST_NOW.replace(tzinfo=None))
        mock_conn.fetchrow = AsyncMock(
            side_effect=[
                None,
                {"was_insert": True, "new_status": "ACTIVE"},
            ]
        )
        mock_conn.fetch = AsyncMock(return_value=[])
        txn = AsyncMock()
        txn.__aenter__ = AsyncMock(return_value=None)
        txn.__aexit__ = AsyncMock(return_value=None)
        mock_conn.transaction = MagicMock(return_value=txn)
        mock_acquire.__aenter__ = AsyncMock(return_value=mock_conn)
        mock_acquire.__aexit__ = AsyncMock(return_value=None)
        handler._pool.acquire = MagicMock(return_value=mock_acquire)

        payload = make_payload(scope_services=["ZService", "aService", "MService"])
        result = await handler.handle(payload, uuid4())

        assert result.success is True
        # Verify the upsert SQL call received sorted lowercase services in JSON
        upsert_call = mock_conn.fetchrow.call_args_list[1]
        # $7 (index 7, 0-based: $1=decision_id,...$7=scope_services) in SQL
        # args are positional — index 6 (0-based after SQL string)
        call_args = upsert_call[0]  # positional args
        # call_args[0] = SQL, [1]=decision_id ... [7]=scope_services (JSON string)
        import json

        scope_services_arg = json.loads(call_args[7])
        assert scope_services_arg == ["aservice", "mservice", "zservice"]

    @pytest.mark.asyncio
    async def test_superseded_by_forces_superseded_status(self) -> None:
        """When superseded_by is set, effective_status is forced to SUPERSEDED."""
        handler = HandlerWriteDecision.__new__(HandlerWriteDecision)
        handler._pool = MagicMock()

        mock_acquire = AsyncMock()
        mock_conn = AsyncMock()
        mock_conn.fetchval = AsyncMock(return_value=TEST_NOW.replace(tzinfo=None))
        mock_conn.fetchrow = AsyncMock(
            side_effect=[
                None,
                {"was_insert": False, "new_status": "SUPERSEDED"},
            ]
        )
        mock_conn.fetch = AsyncMock(return_value=[])
        txn = AsyncMock()
        txn.__aenter__ = AsyncMock(return_value=None)
        txn.__aexit__ = AsyncMock(return_value=None)
        mock_conn.transaction = MagicMock(return_value=txn)
        mock_acquire.__aenter__ = AsyncMock(return_value=mock_conn)
        mock_acquire.__aexit__ = AsyncMock(return_value=None)
        handler._pool.acquire = MagicMock(return_value=mock_acquire)

        superseder_id = uuid4()
        payload = make_payload(status="ACTIVE", superseded_by=superseder_id)
        result = await handler.handle(payload, uuid4())

        assert result.success is True
        # Verify the status arg passed to upsert is SUPERSEDED
        upsert_call = mock_conn.fetchrow.call_args_list[1]
        call_args = upsert_call[0]
        # $5 = status (index 5 in call_args after SQL)
        assert call_args[5] == "SUPERSEDED"

    @pytest.mark.asyncio
    async def test_future_created_at_rejected(self) -> None:
        """created_at more than 5 min in future returns success=False."""
        handler = HandlerWriteDecision.__new__(HandlerWriteDecision)
        handler._pool = MagicMock()

        mock_acquire = AsyncMock()
        mock_conn = AsyncMock()
        db_now = TEST_NOW
        # created_at is 6 minutes in future
        future_ts = TEST_NOW + timedelta(minutes=6)
        mock_conn.fetchval = AsyncMock(return_value=db_now.replace(tzinfo=None))
        txn = AsyncMock()
        txn.__aenter__ = AsyncMock(return_value=None)
        txn.__aexit__ = AsyncMock(return_value=None)
        mock_conn.transaction = MagicMock(return_value=txn)
        mock_acquire.__aenter__ = AsyncMock(return_value=mock_conn)
        mock_acquire.__aexit__ = AsyncMock(return_value=None)
        handler._pool.acquire = MagicMock(return_value=mock_acquire)

        payload = make_payload(created_at=future_ts)
        result = await handler.handle(payload, uuid4())

        assert result.success is False
        # ValueError is caught as UNKNOWN_ERROR by MixinPostgresOpExecutor
        assert result.error_code == EnumPostgresErrorCode.UNKNOWN_ERROR
        assert result.backend_id == "postgres"

    @pytest.mark.asyncio
    async def test_invalid_scope_domain_rejected(self) -> None:
        """Invalid scope_domain returns success=False."""
        handler = HandlerWriteDecision.__new__(HandlerWriteDecision)
        handler._pool = MagicMock()

        mock_acquire = AsyncMock()
        mock_conn = AsyncMock()
        mock_conn.fetchval = AsyncMock(return_value=TEST_NOW.replace(tzinfo=None))
        txn = AsyncMock()
        txn.__aenter__ = AsyncMock(return_value=None)
        txn.__aexit__ = AsyncMock(return_value=None)
        mock_conn.transaction = MagicMock(return_value=txn)
        mock_acquire.__aenter__ = AsyncMock(return_value=mock_conn)
        mock_acquire.__aexit__ = AsyncMock(return_value=None)
        handler._pool.acquire = MagicMock(return_value=mock_acquire)

        payload = make_payload(scope_domain="not-a-valid-domain")
        result = await handler.handle(payload, uuid4())

        assert result.success is False
        assert result.error_code == EnumPostgresErrorCode.UNKNOWN_ERROR

    def test_allowed_domains_constant_matches_spec(self) -> None:
        """ALLOWED_DOMAINS matches the spec exactly."""
        spec_domains = frozenset(
            [
                "transport",
                "data-model",
                "auth",
                "api",
                "infra",
                "testing",
                "code-structure",
                "security",
                "observability",
                "custom",
            ]
        )
        assert spec_domains == ALLOWED_DOMAINS


# =============================================================================
# Tests: HandlerWriteDecision error handling
# =============================================================================


class TestHandlerWriteDecisionErrors:
    """Test error handling for HandlerWriteDecision."""

    def _make_handler_with_error(self, error: Exception) -> HandlerWriteDecision:
        handler = HandlerWriteDecision.__new__(HandlerWriteDecision)
        handler._pool = MagicMock()

        mock_acquire = AsyncMock()
        mock_conn = AsyncMock()
        mock_conn.fetchval = AsyncMock(side_effect=error)
        txn = AsyncMock()
        txn.__aenter__ = AsyncMock(return_value=None)
        txn.__aexit__ = AsyncMock(return_value=None)
        mock_conn.transaction = MagicMock(return_value=txn)
        mock_acquire.__aenter__ = AsyncMock(return_value=mock_conn)
        mock_acquire.__aexit__ = AsyncMock(return_value=None)
        handler._pool.acquire = MagicMock(return_value=mock_acquire)
        return handler

    @pytest.mark.asyncio
    async def test_timeout_error_returns_timeout_code(self) -> None:
        """TimeoutError returns success=False with TIMEOUT_ERROR code."""
        handler = self._make_handler_with_error(TimeoutError("timed out"))
        result = await handler.handle(make_payload(), uuid4())
        assert result.success is False
        assert result.error_code == EnumPostgresErrorCode.TIMEOUT_ERROR

    @pytest.mark.asyncio
    async def test_connection_error_returns_connection_code(self) -> None:
        """InfraConnectionError returns success=False with CONNECTION_ERROR code."""
        handler = self._make_handler_with_error(InfraConnectionError("refused"))
        result = await handler.handle(make_payload(), uuid4())
        assert result.success is False
        assert result.error_code == EnumPostgresErrorCode.CONNECTION_ERROR

    @pytest.mark.asyncio
    async def test_auth_error_returns_auth_code(self) -> None:
        """InfraAuthenticationError returns success=False with AUTH_ERROR code."""
        handler = self._make_handler_with_error(InfraAuthenticationError("bad creds"))
        result = await handler.handle(make_payload(), uuid4())
        assert result.success is False
        assert result.error_code == EnumPostgresErrorCode.AUTH_ERROR

    @pytest.mark.asyncio
    async def test_generic_exception_returns_unknown_code(self) -> None:
        """Unexpected exception returns success=False with UNKNOWN_ERROR code."""
        handler = self._make_handler_with_error(RuntimeError("boom"))
        result = await handler.handle(make_payload(), uuid4())
        assert result.success is False
        assert result.error_code == EnumPostgresErrorCode.UNKNOWN_ERROR


# =============================================================================
# Tests: Stage 2 conflict detection
# =============================================================================


class TestHandlerWriteDecisionStage2:
    """Test Stage 2 conflict detection logic."""

    def _make_handler_stage2(
        self,
        *,
        active_rows: list[dict[str, object]],
        dismissed_exists: bool = False,
        db_now: datetime = TEST_NOW,
    ) -> tuple[HandlerWriteDecision, AsyncMock]:
        """Create a handler wired for Stage 2 testing.

        Returns (handler, stage2_conn) where stage2_conn is the second
        connection used in Stage 2.
        """
        handler = HandlerWriteDecision.__new__(HandlerWriteDecision)
        handler._pool = MagicMock()

        # Stage 1 conn
        conn1 = AsyncMock()
        conn1.fetchval = AsyncMock(return_value=db_now.replace(tzinfo=None))
        conn1.fetchrow = AsyncMock(
            side_effect=[
                None,  # SQL_FETCH_OLD_STATUS
                {"was_insert": True, "new_status": "ACTIVE"},  # SQL_UPSERT
            ]
        )
        txn = AsyncMock()
        txn.__aenter__ = AsyncMock(return_value=None)
        txn.__aexit__ = AsyncMock(return_value=None)
        conn1.transaction = MagicMock(return_value=txn)

        # Stage 2 conn
        conn2 = AsyncMock()
        conn2.fetch = AsyncMock(return_value=active_rows)
        conn2.fetchval = AsyncMock(return_value=1 if dismissed_exists else None)
        conn2.fetchrow = AsyncMock(
            return_value={"conflict_id": uuid4()}
        )  # INSERT RETURNING
        conn2.execute = AsyncMock()

        acm1 = AsyncMock()
        acm1.__aenter__ = AsyncMock(return_value=conn1)
        acm1.__aexit__ = AsyncMock(return_value=None)

        acm2 = AsyncMock()
        acm2.__aenter__ = AsyncMock(return_value=conn2)
        acm2.__aexit__ = AsyncMock(return_value=None)

        call_count = [0]

        def acquire_factory() -> AsyncMock:
            idx = call_count[0]
            call_count[0] += 1
            return acm1 if idx == 0 else acm2

        handler._pool.acquire = MagicMock(side_effect=acquire_factory)
        return handler, conn2

    @pytest.mark.asyncio
    async def test_no_active_decisions_no_conflicts_written(self) -> None:
        """When no other ACTIVE decisions exist, no conflicts are written."""
        handler, conn2 = self._make_handler_stage2(active_rows=[])
        result = await handler.handle(make_payload(), uuid4())

        assert result.success is True
        conn2.fetchrow.assert_not_called()  # INSERT CONFLICT not called

    @pytest.mark.asyncio
    async def test_high_confidence_pair_writes_conflict(self) -> None:
        """A pair with structural_confidence >= 0.9 (both empty services, same scope) writes a conflict."""
        decision_id_new = uuid4()
        decision_id_other = uuid4()

        active_rows = [
            {
                "decision_id": decision_id_other,
                "scope_domain": "infra",
                "scope_layer": "architecture",
                "scope_services": "[]",
                "status": "ACTIVE",
                "superseded_by": None,
            }
        ]

        # dismissed_exists=False → demotion happens
        handler, conn2 = self._make_handler_stage2(
            active_rows=active_rows, dismissed_exists=False
        )

        payload = make_payload(
            decision_id=decision_id_new,
            scope_domain="infra",
            scope_layer="architecture",
            scope_services=[],
        )
        result = await handler.handle(payload, uuid4())

        assert result.success is True
        # Conflict INSERT was called
        assert conn2.fetchrow.called

    @pytest.mark.asyncio
    async def test_active_invariant_demotes_to_proposed_without_dismissal(self) -> None:
        """ACTIVE invariant: high-confidence pair without dismissal demotes new decision."""
        decision_id_new = uuid4()
        decision_id_other = uuid4()

        active_rows = [
            {
                "decision_id": decision_id_other,
                "scope_domain": "infra",
                "scope_layer": "architecture",
                "scope_services": "[]",
                "status": "ACTIVE",
                "superseded_by": None,
            }
        ]

        handler, conn2 = self._make_handler_stage2(
            active_rows=active_rows, dismissed_exists=False
        )

        payload = make_payload(
            decision_id=decision_id_new,
            scope_domain="infra",
            scope_layer="architecture",
            scope_services=[],
        )
        await handler.handle(payload, uuid4())

        # SQL_DEMOTE_TO_PROPOSED should have been called
        assert conn2.execute.called
        execute_calls = [str(call) for call in conn2.execute.call_args_list]
        assert any("PROPOSED" in c for c in execute_calls)

    @pytest.mark.asyncio
    async def test_active_invariant_no_demotion_when_dismissed_conflict_exists(
        self,
    ) -> None:
        """ACTIVE invariant: no demotion when a DISMISSED conflict already exists."""
        decision_id_new = uuid4()
        decision_id_other = uuid4()

        active_rows = [
            {
                "decision_id": decision_id_other,
                "scope_domain": "infra",
                "scope_layer": "architecture",
                "scope_services": "[]",
                "status": "ACTIVE",
                "superseded_by": None,
            }
        ]

        # dismissed_exists=True → no demotion
        handler, conn2 = self._make_handler_stage2(
            active_rows=active_rows, dismissed_exists=True
        )

        payload = make_payload(
            decision_id=decision_id_new,
            scope_domain="infra",
            scope_layer="architecture",
            scope_services=[],
        )
        await handler.handle(payload, uuid4())

        # SQL_DEMOTE_TO_PROPOSED should NOT have been called
        execute_calls = [str(call) for call in conn2.execute.call_args_list]
        assert not any("PROPOSED" in c for c in execute_calls)

    @pytest.mark.asyncio
    async def test_low_confidence_pair_not_written(self) -> None:
        """Pairs with structural_confidence < 0.3 are not written to decision_conflicts."""
        decision_id_new = uuid4()
        decision_id_other = uuid4()

        # Different domain → confidence = 0.0
        active_rows = [
            {
                "decision_id": decision_id_other,
                "scope_domain": "auth",  # different domain
                "scope_layer": "architecture",
                "scope_services": "[]",
                "status": "ACTIVE",
                "superseded_by": None,
            }
        ]

        handler, conn2 = self._make_handler_stage2(active_rows=active_rows)

        payload = make_payload(
            decision_id=decision_id_new,
            scope_domain="infra",
            scope_layer="architecture",
            scope_services=[],
        )
        result = await handler.handle(payload, uuid4())

        assert result.success is True
        # Conflict INSERT NOT called for confidence < 0.3
        conn2.fetchrow.assert_not_called()

    @pytest.mark.asyncio
    async def test_stage2_failure_does_not_rollback_stage1(self) -> None:
        """Stage 2 exception is swallowed; result is still success=True."""
        handler = HandlerWriteDecision.__new__(HandlerWriteDecision)
        handler._pool = MagicMock()

        # Stage 1 conn — succeeds
        conn1 = AsyncMock()
        conn1.fetchval = AsyncMock(return_value=TEST_NOW.replace(tzinfo=None))
        conn1.fetchrow = AsyncMock(
            side_effect=[
                None,
                {"was_insert": True, "new_status": "ACTIVE"},
            ]
        )
        txn = AsyncMock()
        txn.__aenter__ = AsyncMock(return_value=None)
        txn.__aexit__ = AsyncMock(return_value=None)
        conn1.transaction = MagicMock(return_value=txn)

        # Stage 2 conn — raises RuntimeError
        conn2 = AsyncMock()
        conn2.fetch = AsyncMock(side_effect=RuntimeError("Stage 2 exploded"))

        acm1 = AsyncMock()
        acm1.__aenter__ = AsyncMock(return_value=conn1)
        acm1.__aexit__ = AsyncMock(return_value=None)

        acm2 = AsyncMock()
        acm2.__aenter__ = AsyncMock(return_value=conn2)
        acm2.__aexit__ = AsyncMock(return_value=None)

        call_count = [0]

        def acquire_factory() -> AsyncMock:
            idx = call_count[0]
            call_count[0] += 1
            return acm1 if idx == 0 else acm2

        handler._pool.acquire = MagicMock(side_effect=acquire_factory)

        result = await handler.handle(make_payload(), uuid4())

        # Stage 1 committed → overall success=True despite Stage 2 failure
        assert result.success is True
        assert result.error is None


__all__: list[str] = [
    "TestStructuralConfidence",
    "TestHandlerWriteDecisionStage1Success",
    "TestHandlerWriteDecisionErrors",
    "TestHandlerWriteDecisionStage2",
]
