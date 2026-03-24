# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Unit tests for ServiceStaleRegistrationCleanup.

Tests the stale registration cleanup service that resets projections
stuck in AWAITING_ACK or ACCEPTED state past their ack_deadline.

Related Tickets:
    - OMN-5821: Reset stale awaiting_ack registrations
"""

from __future__ import annotations

from datetime import UTC, datetime
from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4

import pytest

from omnibase_infra.enums import EnumRegistrationState
from omnibase_infra.errors import InfraConnectionError, RuntimeHostError
from omnibase_infra.services.service_stale_registration_cleanup import (
    ServiceStaleRegistrationCleanup,
    StaleCleanupReport,
)


@pytest.fixture
def mock_pool() -> MagicMock:
    """Create a mock asyncpg connection pool."""
    pool = MagicMock()
    conn = AsyncMock()
    pool.acquire.return_value.__aenter__ = AsyncMock(return_value=conn)
    pool.acquire.return_value.__aexit__ = AsyncMock(return_value=False)
    return pool


@pytest.fixture
def mock_conn(mock_pool: MagicMock) -> AsyncMock:
    """Get the mock connection from the mock pool."""
    return mock_pool.acquire.return_value.__aenter__.return_value


@pytest.fixture
def service(mock_pool: MagicMock) -> ServiceStaleRegistrationCleanup:
    """Create a cleanup service with mock pool."""
    return ServiceStaleRegistrationCleanup(mock_pool)


class TestStaleCleanupReport:
    """Tests for the StaleCleanupReport dataclass."""

    def test_report_creation(self) -> None:
        """Report can be created with all fields."""
        corr_id = uuid4()
        now = datetime.now(UTC)
        report = StaleCleanupReport(
            reset_count=5,
            scanned_count=10,
            correlation_id=corr_id,
            executed_at=now,
        )
        assert report.reset_count == 5
        assert report.scanned_count == 10
        assert report.correlation_id == corr_id
        assert report.executed_at == now

    def test_report_is_frozen(self) -> None:
        """Report is immutable."""
        report = StaleCleanupReport(
            reset_count=0,
            scanned_count=0,
            correlation_id=uuid4(),
            executed_at=datetime.now(UTC),
        )
        with pytest.raises(AttributeError):
            report.reset_count = 1  # type: ignore[misc]


class TestServiceStaleRegistrationCleanup:
    """Tests for the cleanup service."""

    @pytest.mark.asyncio
    async def test_no_stale_registrations(
        self,
        service: ServiceStaleRegistrationCleanup,
        mock_conn: AsyncMock,
    ) -> None:
        """Returns zero counts when no stale registrations exist."""
        mock_conn.fetchval.return_value = 0

        report = await service.cleanup_stale_registrations()

        assert report.reset_count == 0
        assert report.scanned_count == 0
        # Should not call execute (no update needed)
        mock_conn.execute.assert_not_called()

    @pytest.mark.asyncio
    async def test_resets_stale_registrations(
        self,
        service: ServiceStaleRegistrationCleanup,
        mock_conn: AsyncMock,
    ) -> None:
        """Resets stale registrations and returns correct counts."""
        mock_conn.fetchval.return_value = 5
        mock_conn.execute.return_value = "UPDATE 5"

        report = await service.cleanup_stale_registrations()

        assert report.reset_count == 5
        assert report.scanned_count == 5
        mock_conn.execute.assert_called_once()

        # Verify the update SQL sets state to ACK_TIMED_OUT
        # call_args[0][0] is the SQL string, [0][1] is $1 (the state value)
        call_args = mock_conn.execute.call_args
        assert call_args[0][1] == EnumRegistrationState.ACK_TIMED_OUT.value

    @pytest.mark.asyncio
    async def test_queries_correct_states(
        self,
        service: ServiceStaleRegistrationCleanup,
        mock_conn: AsyncMock,
    ) -> None:
        """Queries for AWAITING_ACK and ACCEPTED states."""
        mock_conn.fetchval.return_value = 0

        await service.cleanup_stale_registrations()

        # fetchval(sql, domain, states, now) -- [0][0]=sql, [0][1]=domain, [0][2]=states
        call_args = mock_conn.fetchval.call_args
        states_arg = call_args[0][2]
        assert EnumRegistrationState.AWAITING_ACK.value in states_arg
        assert EnumRegistrationState.ACCEPTED.value in states_arg

    @pytest.mark.asyncio
    async def test_uses_provided_correlation_id(
        self,
        service: ServiceStaleRegistrationCleanup,
        mock_conn: AsyncMock,
    ) -> None:
        """Uses provided correlation_id in report."""
        mock_conn.fetchval.return_value = 0
        corr_id = uuid4()

        report = await service.cleanup_stale_registrations(correlation_id=corr_id)

        assert report.correlation_id == corr_id

    @pytest.mark.asyncio
    async def test_generates_correlation_id_when_not_provided(
        self,
        service: ServiceStaleRegistrationCleanup,
        mock_conn: AsyncMock,
    ) -> None:
        """Generates a correlation_id when none is provided."""
        mock_conn.fetchval.return_value = 0

        report = await service.cleanup_stale_registrations()

        assert report.correlation_id is not None

    @pytest.mark.asyncio
    async def test_connection_error_raises_infra_connection_error(
        self,
        service: ServiceStaleRegistrationCleanup,
        mock_conn: AsyncMock,
    ) -> None:
        """Raises InfraConnectionError on database connection failure."""
        import asyncpg

        mock_conn.fetchval.side_effect = asyncpg.PostgresConnectionError(
            "connection refused"
        )

        with pytest.raises(InfraConnectionError):
            await service.cleanup_stale_registrations()

    @pytest.mark.asyncio
    async def test_generic_error_raises_runtime_host_error(
        self,
        service: ServiceStaleRegistrationCleanup,
        mock_conn: AsyncMock,
    ) -> None:
        """Raises RuntimeHostError on unexpected failures."""
        mock_conn.fetchval.side_effect = RuntimeError("unexpected")

        with pytest.raises(RuntimeHostError):
            await service.cleanup_stale_registrations()

    @pytest.mark.asyncio
    async def test_partial_update_returns_actual_count(
        self,
        service: ServiceStaleRegistrationCleanup,
        mock_conn: AsyncMock,
    ) -> None:
        """Returns actual update count even if different from scan count."""
        mock_conn.fetchval.return_value = 10
        mock_conn.execute.return_value = (
            "UPDATE 8"  # Race: 2 were updated between count and update
        )

        report = await service.cleanup_stale_registrations()

        assert report.reset_count == 8
        assert report.scanned_count == 10

    @pytest.mark.asyncio
    async def test_custom_domain(
        self,
        service: ServiceStaleRegistrationCleanup,
        mock_conn: AsyncMock,
    ) -> None:
        """Passes custom domain to queries."""
        mock_conn.fetchval.return_value = 0

        await service.cleanup_stale_registrations(domain="custom_domain")

        # fetchval(sql, domain, states, now) -- [0][1]=domain
        call_args = mock_conn.fetchval.call_args
        domain_arg = call_args[0][1]
        assert domain_arg == "custom_domain"


class TestReducerRetriableStatesConsistency:
    """Verify that the cleanup target state is in the reducer's retriable set.

    This is the regression guard: if someone changes _RETRIABLE_STATES in
    RegistrationReducerService to no longer include ACK_TIMED_OUT, this test
    will fail, alerting us that the cleanup service's target state would no
    longer trigger re-registration on the next introspection event.
    """

    def test_ack_timed_out_is_retriable(self) -> None:
        """ACK_TIMED_OUT must remain in _RETRIABLE_STATES for cleanup to work."""
        from omnibase_infra.nodes.node_registration_orchestrator.services.registration_reducer_service import (
            _RETRIABLE_STATES,
        )

        assert EnumRegistrationState.ACK_TIMED_OUT in _RETRIABLE_STATES, (
            "ACK_TIMED_OUT was removed from _RETRIABLE_STATES. "
            "This breaks the stale registration cleanup service "
            "(ServiceStaleRegistrationCleanup) which transitions stale nodes "
            "to ACK_TIMED_OUT so the next introspection re-registers them."
        )

    def test_awaiting_ack_is_retriable(self) -> None:
        """AWAITING_ACK must remain in _RETRIABLE_STATES for self-healing."""
        from omnibase_infra.nodes.node_registration_orchestrator.services.registration_reducer_service import (
            _RETRIABLE_STATES,
        )

        assert EnumRegistrationState.AWAITING_ACK in _RETRIABLE_STATES, (
            "AWAITING_ACK was removed from _RETRIABLE_STATES. "
            "This breaks self-healing for nodes stuck in AWAITING_ACK "
            "from a prior runtime version (pre-OMN-5132)."
        )

    def test_direct_to_active_skips_awaiting_ack(self) -> None:
        """Introspection -> ACTIVE must not produce AWAITING_ACK state.

        This is the core regression guard for OMN-5132. If someone
        reintroduces the AWAITING_ACK intermediate state in
        decide_introspection(), nodes will stall again.
        """
        from omnibase_core.enums import EnumNodeKind
        from omnibase_infra.models.registration.model_node_introspection_event import (
            ModelNodeIntrospectionEvent,
        )
        from omnibase_infra.nodes.node_registration_orchestrator.services.registration_reducer_service import (
            RegistrationReducerService,
        )

        reducer = RegistrationReducerService()
        node_id = uuid4()
        now = datetime.now(UTC)
        corr_id = uuid4()

        # Simulate a new node introspection event
        event = ModelNodeIntrospectionEvent(
            node_id=node_id,
            node_type=EnumNodeKind.EFFECT,
            correlation_id=corr_id,
            timestamp=now,
        )

        decision = reducer.decide_introspection(
            projection=None,
            event=event,
            correlation_id=corr_id,
            now=now,
        )

        assert decision.action == "emit", "New node should trigger registration"
        assert decision.new_state == EnumRegistrationState.ACTIVE, (
            "New node must transition directly to ACTIVE (OMN-5132). "
            "If this fails, the AWAITING_ACK intermediate state was "
            "reintroduced, which will cause nodes to stall."
        )


__all__: list[str] = []
