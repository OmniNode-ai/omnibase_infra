# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""
Comprehensive unit tests for ProjectionReaderRegistration.

This test suite validates:
- Reader instantiation with asyncpg connection pool
- Entity state lookup (get_entity_state)
- Registration status queries (get_registration_status)
- State-based queries (get_by_state)
- Overdue deadline queries (ack and liveness timeouts)
- State counting aggregation (count_by_state)
- Error handling for database failures
- Circuit breaker integration

Test Organization:
    - TestProjectionReaderBasics: Instantiation and configuration
    - TestProjectionReaderEntityState: Entity state lookups
    - TestProjectionReaderByState: State-filtered queries
    - TestProjectionReaderOverdueQueries: Timeout deadline queries
    - TestProjectionReaderAggregation: Count and aggregation queries
    - TestProjectionReaderErrorHandling: Error scenarios
    - TestProjectionReaderCircuitBreaker: Circuit breaker behavior

Coverage Goals:
    - >90% code coverage for reader
    - All query paths tested
    - Error handling validated
    - Circuit breaker integration tested

Related Tickets:
    - OMN-944 (F1): Implement Registration Projection Schema
    - OMN-940 (F0): Define Projector Execution Model
    - OMN-930 (C0): Projection Reader Protocol
    - OMN-932 (C2): Durable Timeout Handling
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from unittest.mock import AsyncMock, MagicMock
from uuid import UUID, uuid4

import asyncpg
import pytest

from omnibase_infra.enums import EnumRegistrationState
from omnibase_infra.errors import (
    InfraConnectionError,
    InfraTimeoutError,
    InfraUnavailableError,
    RuntimeHostError,
)
from omnibase_infra.models.projection import ModelRegistrationProjection
from omnibase_infra.models.registration.model_node_capabilities import (
    ModelNodeCapabilities,
)
from omnibase_infra.projectors.projection_reader_registration import (
    ProjectionReaderRegistration,
)


def create_mock_row(
    entity_id: UUID | None = None,
    state: EnumRegistrationState = EnumRegistrationState.ACTIVE,
    domain: str = "registration",
    ack_deadline: datetime | None = None,
    liveness_deadline: datetime | None = None,
    ack_timeout_emitted_at: datetime | None = None,
    liveness_timeout_emitted_at: datetime | None = None,
    contract_type: str | None = "effect",
    intent_types: list[str] | None = None,
    protocols: list[str] | None = None,
    capability_tags: list[str] | None = None,
    contract_version: str | None = "1.0.0",
) -> dict:
    """Create a mock database row with sensible defaults."""
    now = datetime.now(UTC)
    node_id = entity_id or uuid4()
    capabilities = ModelNodeCapabilities(postgres=True, read=True)

    return {
        "entity_id": node_id,
        "domain": domain,
        "current_state": state.value,
        "node_type": "effect",
        "node_version": "1.0.0",
        "capabilities": capabilities.model_dump_json(),
        # Capability fields (OMN-1134)
        "contract_type": contract_type,
        "intent_types": intent_types or [],
        "protocols": protocols or [],
        "capability_tags": capability_tags or [],
        "contract_version": contract_version,
        # Timeout fields
        "ack_deadline": ack_deadline,
        "liveness_deadline": liveness_deadline,
        "ack_timeout_emitted_at": ack_timeout_emitted_at,
        "liveness_timeout_emitted_at": liveness_timeout_emitted_at,
        "last_heartbeat_at": None,
        "last_applied_event_id": uuid4(),
        "last_applied_offset": 100,
        "last_applied_sequence": None,
        "last_applied_partition": "0",
        "registered_at": now,
        "updated_at": now,
        "correlation_id": uuid4(),
    }


@pytest.fixture
def mock_pool() -> MagicMock:
    """Create a mock asyncpg connection pool."""
    pool = MagicMock(spec=asyncpg.Pool)
    return pool


@pytest.fixture
def mock_connection() -> AsyncMock:
    """Create a mock asyncpg connection."""
    conn = AsyncMock()
    return conn


@pytest.fixture
def reader(mock_pool: MagicMock) -> ProjectionReaderRegistration:
    """Create a ProjectionReaderRegistration instance with mocked pool."""
    return ProjectionReaderRegistration(pool=mock_pool)


@pytest.mark.unit
class TestProjectionReaderBasics:
    """Test basic reader instantiation and configuration."""

    def test_reader_instantiation(self, mock_pool: MagicMock) -> None:
        """Test that reader initializes correctly with connection pool."""
        reader = ProjectionReaderRegistration(pool=mock_pool)

        assert reader._pool is mock_pool
        # Verify circuit breaker is initialized
        assert hasattr(reader, "_circuit_breaker_lock")
        assert reader._circuit_breaker_failures == 0
        assert reader._circuit_breaker_open is False

    def test_reader_circuit_breaker_config(
        self, reader: ProjectionReaderRegistration
    ) -> None:
        """Test that circuit breaker is configured correctly."""
        # Default config: threshold=5, reset_timeout=60.0
        assert reader.circuit_breaker_threshold == 5
        assert reader.circuit_breaker_reset_timeout == 60.0
        assert reader.service_name == "projection_reader.registration"

    def test_row_to_projection_conversion(
        self, reader: ProjectionReaderRegistration
    ) -> None:
        """Test internal row to projection conversion."""
        mock_row = create_mock_row()

        # Access private method for testing conversion logic
        projection = reader._row_to_projection(mock_row)

        assert isinstance(projection, ModelRegistrationProjection)
        assert projection.entity_id == mock_row["entity_id"]
        assert projection.current_state == EnumRegistrationState.ACTIVE
        assert projection.node_type == "effect"
        assert projection.node_version == "1.0.0"
        assert isinstance(projection.capabilities, ModelNodeCapabilities)

    def test_row_to_projection_with_dict_capabilities(
        self, reader: ProjectionReaderRegistration
    ) -> None:
        """Test row conversion with dict capabilities (already parsed)."""
        mock_row = create_mock_row()
        # Simulate already-parsed dict instead of JSON string
        mock_row["capabilities"] = {"postgres": True, "read": True}

        projection = reader._row_to_projection(mock_row)

        assert isinstance(projection.capabilities, ModelNodeCapabilities)
        assert projection.capabilities.postgres is True


@pytest.mark.unit
@pytest.mark.asyncio
class TestProjectionReaderEntityState:
    """Test entity state lookup methods."""

    async def test_get_entity_state_found(
        self,
        reader: ProjectionReaderRegistration,
        mock_pool: MagicMock,
        mock_connection: AsyncMock,
    ) -> None:
        """Test get_entity_state returns projection when found."""
        mock_pool.acquire.return_value.__aenter__.return_value = mock_connection
        mock_pool.acquire.return_value.__aexit__.return_value = None
        mock_row = create_mock_row()
        mock_connection.fetchrow.return_value = mock_row

        entity_id = mock_row["entity_id"]
        result = await reader.get_entity_state(entity_id=entity_id)

        assert result is not None
        assert isinstance(result, ModelRegistrationProjection)
        assert result.entity_id == entity_id

    async def test_get_entity_state_not_found(
        self,
        reader: ProjectionReaderRegistration,
        mock_pool: MagicMock,
        mock_connection: AsyncMock,
    ) -> None:
        """Test get_entity_state returns None when not found."""
        mock_pool.acquire.return_value.__aenter__.return_value = mock_connection
        mock_pool.acquire.return_value.__aexit__.return_value = None
        mock_connection.fetchrow.return_value = None

        entity_id = uuid4()
        result = await reader.get_entity_state(entity_id=entity_id)

        assert result is None

    async def test_get_entity_state_custom_domain(
        self,
        reader: ProjectionReaderRegistration,
        mock_pool: MagicMock,
        mock_connection: AsyncMock,
    ) -> None:
        """Test get_entity_state with custom domain."""
        mock_pool.acquire.return_value.__aenter__.return_value = mock_connection
        mock_pool.acquire.return_value.__aexit__.return_value = None
        mock_row = create_mock_row(domain="custom_domain")
        mock_connection.fetchrow.return_value = mock_row

        entity_id = mock_row["entity_id"]
        result = await reader.get_entity_state(
            entity_id=entity_id,
            domain="custom_domain",
        )

        assert result is not None
        assert result.domain == "custom_domain"
        # Verify domain was passed to query
        call_args = mock_connection.fetchrow.call_args
        assert call_args is not None
        args = call_args[0]
        assert "custom_domain" in args

    async def test_get_entity_state_with_correlation_id(
        self,
        reader: ProjectionReaderRegistration,
        mock_pool: MagicMock,
        mock_connection: AsyncMock,
    ) -> None:
        """Test get_entity_state propagates correlation ID."""
        mock_pool.acquire.return_value.__aenter__.return_value = mock_connection
        mock_pool.acquire.return_value.__aexit__.return_value = None
        mock_connection.fetchrow.return_value = create_mock_row()

        correlation_id = uuid4()
        await reader.get_entity_state(
            entity_id=uuid4(),
            correlation_id=correlation_id,
        )

        # Should not raise - correlation ID used for tracing

    async def test_get_registration_status_found(
        self,
        reader: ProjectionReaderRegistration,
        mock_pool: MagicMock,
        mock_connection: AsyncMock,
    ) -> None:
        """Test get_registration_status returns state when found."""
        mock_pool.acquire.return_value.__aenter__.return_value = mock_connection
        mock_pool.acquire.return_value.__aexit__.return_value = None
        mock_connection.fetchrow.return_value = {
            "current_state": EnumRegistrationState.ACTIVE.value
        }

        entity_id = uuid4()
        result = await reader.get_registration_status(entity_id=entity_id)

        assert result == EnumRegistrationState.ACTIVE

    async def test_get_registration_status_not_found(
        self,
        reader: ProjectionReaderRegistration,
        mock_pool: MagicMock,
        mock_connection: AsyncMock,
    ) -> None:
        """Test get_registration_status returns None when not found."""
        mock_pool.acquire.return_value.__aenter__.return_value = mock_connection
        mock_pool.acquire.return_value.__aexit__.return_value = None
        mock_connection.fetchrow.return_value = None

        entity_id = uuid4()
        result = await reader.get_registration_status(entity_id=entity_id)

        assert result is None

    async def test_get_registration_status_all_states(
        self,
        reader: ProjectionReaderRegistration,
        mock_pool: MagicMock,
        mock_connection: AsyncMock,
    ) -> None:
        """Test get_registration_status works for all states."""
        mock_pool.acquire.return_value.__aenter__.return_value = mock_connection
        mock_pool.acquire.return_value.__aexit__.return_value = None

        for state in EnumRegistrationState:
            mock_connection.fetchrow.return_value = {"current_state": state.value}

            result = await reader.get_registration_status(entity_id=uuid4())

            assert result == state


@pytest.mark.unit
@pytest.mark.asyncio
class TestProjectionReaderByState:
    """Test state-filtered queries."""

    async def test_get_by_state_returns_list(
        self,
        reader: ProjectionReaderRegistration,
        mock_pool: MagicMock,
        mock_connection: AsyncMock,
    ) -> None:
        """Test get_by_state returns list of projections."""
        mock_pool.acquire.return_value.__aenter__.return_value = mock_connection
        mock_pool.acquire.return_value.__aexit__.return_value = None

        # Create mock rows for multiple active nodes
        mock_rows = [
            create_mock_row(state=EnumRegistrationState.ACTIVE),
            create_mock_row(state=EnumRegistrationState.ACTIVE),
            create_mock_row(state=EnumRegistrationState.ACTIVE),
        ]
        mock_connection.fetch.return_value = mock_rows

        result = await reader.get_by_state(state=EnumRegistrationState.ACTIVE)

        assert isinstance(result, list)
        assert len(result) == 3
        for proj in result:
            assert isinstance(proj, ModelRegistrationProjection)
            assert proj.current_state == EnumRegistrationState.ACTIVE

    async def test_get_by_state_empty_result(
        self,
        reader: ProjectionReaderRegistration,
        mock_pool: MagicMock,
        mock_connection: AsyncMock,
    ) -> None:
        """Test get_by_state returns empty list when no matches."""
        mock_pool.acquire.return_value.__aenter__.return_value = mock_connection
        mock_pool.acquire.return_value.__aexit__.return_value = None
        mock_connection.fetch.return_value = []

        result = await reader.get_by_state(state=EnumRegistrationState.REJECTED)

        assert result == []

    async def test_get_by_state_respects_limit(
        self,
        reader: ProjectionReaderRegistration,
        mock_pool: MagicMock,
        mock_connection: AsyncMock,
    ) -> None:
        """Test get_by_state respects limit parameter."""
        mock_pool.acquire.return_value.__aenter__.return_value = mock_connection
        mock_pool.acquire.return_value.__aexit__.return_value = None
        mock_connection.fetch.return_value = [create_mock_row()]

        limit = 50
        await reader.get_by_state(
            state=EnumRegistrationState.ACTIVE,
            limit=limit,
        )

        # Verify limit was passed to query
        call_args = mock_connection.fetch.call_args
        assert call_args is not None
        args = call_args[0]
        assert limit in args

    async def test_get_by_state_custom_domain(
        self,
        reader: ProjectionReaderRegistration,
        mock_pool: MagicMock,
        mock_connection: AsyncMock,
    ) -> None:
        """Test get_by_state with custom domain."""
        mock_pool.acquire.return_value.__aenter__.return_value = mock_connection
        mock_pool.acquire.return_value.__aexit__.return_value = None
        mock_connection.fetch.return_value = []

        custom_domain = "custom_domain"
        await reader.get_by_state(
            state=EnumRegistrationState.ACTIVE,
            domain=custom_domain,
        )

        # Verify domain was passed to query
        call_args = mock_connection.fetch.call_args
        assert call_args is not None
        args = call_args[0]
        assert custom_domain in args


@pytest.mark.unit
@pytest.mark.asyncio
class TestProjectionReaderOverdueQueries:
    """Test overdue deadline queries for timeout handling."""

    async def test_get_overdue_ack_registrations(
        self,
        reader: ProjectionReaderRegistration,
        mock_pool: MagicMock,
        mock_connection: AsyncMock,
    ) -> None:
        """Test get_overdue_ack_registrations returns overdue nodes."""
        mock_pool.acquire.return_value.__aenter__.return_value = mock_connection
        mock_pool.acquire.return_value.__aexit__.return_value = None

        now = datetime.now(UTC)
        past_deadline = now - timedelta(minutes=5)

        # Create mock rows for overdue ack
        mock_rows = [
            create_mock_row(
                state=EnumRegistrationState.AWAITING_ACK,
                ack_deadline=past_deadline,
                ack_timeout_emitted_at=None,
            ),
        ]
        mock_connection.fetch.return_value = mock_rows

        result = await reader.get_overdue_ack_registrations(now=now)

        assert isinstance(result, list)
        assert len(result) == 1
        assert result[0].current_state == EnumRegistrationState.AWAITING_ACK

    async def test_get_overdue_ack_registrations_empty(
        self,
        reader: ProjectionReaderRegistration,
        mock_pool: MagicMock,
        mock_connection: AsyncMock,
    ) -> None:
        """Test get_overdue_ack_registrations returns empty when no overdue."""
        mock_pool.acquire.return_value.__aenter__.return_value = mock_connection
        mock_pool.acquire.return_value.__aexit__.return_value = None
        mock_connection.fetch.return_value = []

        now = datetime.now(UTC)
        result = await reader.get_overdue_ack_registrations(now=now)

        assert result == []

    async def test_get_overdue_ack_respects_limit(
        self,
        reader: ProjectionReaderRegistration,
        mock_pool: MagicMock,
        mock_connection: AsyncMock,
    ) -> None:
        """Test get_overdue_ack_registrations respects limit."""
        mock_pool.acquire.return_value.__aenter__.return_value = mock_connection
        mock_pool.acquire.return_value.__aexit__.return_value = None
        mock_connection.fetch.return_value = []

        now = datetime.now(UTC)
        limit = 25
        await reader.get_overdue_ack_registrations(now=now, limit=limit)

        # Verify limit was passed to query
        call_args = mock_connection.fetch.call_args
        assert call_args is not None
        args = call_args[0]
        assert limit in args

    async def test_get_overdue_liveness_registrations(
        self,
        reader: ProjectionReaderRegistration,
        mock_pool: MagicMock,
        mock_connection: AsyncMock,
    ) -> None:
        """Test get_overdue_liveness_registrations returns overdue active nodes."""
        mock_pool.acquire.return_value.__aenter__.return_value = mock_connection
        mock_pool.acquire.return_value.__aexit__.return_value = None

        now = datetime.now(UTC)
        past_deadline = now - timedelta(minutes=10)

        # Create mock rows for overdue liveness
        mock_rows = [
            create_mock_row(
                state=EnumRegistrationState.ACTIVE,
                liveness_deadline=past_deadline,
                liveness_timeout_emitted_at=None,
            ),
        ]
        mock_connection.fetch.return_value = mock_rows

        result = await reader.get_overdue_liveness_registrations(now=now)

        assert isinstance(result, list)
        assert len(result) == 1
        assert result[0].current_state == EnumRegistrationState.ACTIVE

    async def test_get_overdue_liveness_registrations_empty(
        self,
        reader: ProjectionReaderRegistration,
        mock_pool: MagicMock,
        mock_connection: AsyncMock,
    ) -> None:
        """Test get_overdue_liveness_registrations returns empty when no overdue."""
        mock_pool.acquire.return_value.__aenter__.return_value = mock_connection
        mock_pool.acquire.return_value.__aexit__.return_value = None
        mock_connection.fetch.return_value = []

        now = datetime.now(UTC)
        result = await reader.get_overdue_liveness_registrations(now=now)

        assert result == []

    async def test_get_overdue_liveness_respects_limit(
        self,
        reader: ProjectionReaderRegistration,
        mock_pool: MagicMock,
        mock_connection: AsyncMock,
    ) -> None:
        """Test get_overdue_liveness_registrations respects limit."""
        mock_pool.acquire.return_value.__aenter__.return_value = mock_connection
        mock_pool.acquire.return_value.__aexit__.return_value = None
        mock_connection.fetch.return_value = []

        now = datetime.now(UTC)
        limit = 30
        await reader.get_overdue_liveness_registrations(now=now, limit=limit)

        # Verify limit was passed to query
        call_args = mock_connection.fetch.call_args
        assert call_args is not None
        args = call_args[0]
        assert limit in args


@pytest.mark.unit
@pytest.mark.asyncio
class TestProjectionReaderAggregation:
    """Test count and aggregation queries."""

    async def test_count_by_state_returns_dict(
        self,
        reader: ProjectionReaderRegistration,
        mock_pool: MagicMock,
        mock_connection: AsyncMock,
    ) -> None:
        """Test count_by_state returns state count dictionary."""
        mock_pool.acquire.return_value.__aenter__.return_value = mock_connection
        mock_pool.acquire.return_value.__aexit__.return_value = None

        mock_rows = [
            {"current_state": EnumRegistrationState.ACTIVE.value, "count": 10},
            {
                "current_state": EnumRegistrationState.PENDING_REGISTRATION.value,
                "count": 5,
            },
            {"current_state": EnumRegistrationState.LIVENESS_EXPIRED.value, "count": 2},
        ]
        mock_connection.fetch.return_value = mock_rows

        result = await reader.count_by_state()

        assert isinstance(result, dict)
        assert result[EnumRegistrationState.ACTIVE] == 10
        assert result[EnumRegistrationState.PENDING_REGISTRATION] == 5
        assert result[EnumRegistrationState.LIVENESS_EXPIRED] == 2

    async def test_count_by_state_empty_result(
        self,
        reader: ProjectionReaderRegistration,
        mock_pool: MagicMock,
        mock_connection: AsyncMock,
    ) -> None:
        """Test count_by_state returns empty dict when no projections."""
        mock_pool.acquire.return_value.__aenter__.return_value = mock_connection
        mock_pool.acquire.return_value.__aexit__.return_value = None
        mock_connection.fetch.return_value = []

        result = await reader.count_by_state()

        assert result == {}

    async def test_count_by_state_custom_domain(
        self,
        reader: ProjectionReaderRegistration,
        mock_pool: MagicMock,
        mock_connection: AsyncMock,
    ) -> None:
        """Test count_by_state with custom domain."""
        mock_pool.acquire.return_value.__aenter__.return_value = mock_connection
        mock_pool.acquire.return_value.__aexit__.return_value = None
        mock_connection.fetch.return_value = []

        custom_domain = "custom_domain"
        await reader.count_by_state(domain=custom_domain)

        # Verify domain was passed to query
        call_args = mock_connection.fetch.call_args
        assert call_args is not None
        args = call_args[0]
        assert custom_domain in args

    async def test_count_by_state_all_states(
        self,
        reader: ProjectionReaderRegistration,
        mock_pool: MagicMock,
        mock_connection: AsyncMock,
    ) -> None:
        """Test count_by_state handles all possible states."""
        mock_pool.acquire.return_value.__aenter__.return_value = mock_connection
        mock_pool.acquire.return_value.__aexit__.return_value = None

        # Create mock rows for all states
        mock_rows = [
            {"current_state": state.value, "count": i + 1}
            for i, state in enumerate(EnumRegistrationState)
        ]
        mock_connection.fetch.return_value = mock_rows

        result = await reader.count_by_state()

        assert len(result) == len(EnumRegistrationState)
        for state in EnumRegistrationState:
            assert state in result


@pytest.mark.unit
@pytest.mark.asyncio
class TestProjectionReaderErrorHandling:
    """Test error handling for database failures."""

    async def test_get_entity_state_connection_error(
        self,
        reader: ProjectionReaderRegistration,
        mock_pool: MagicMock,
    ) -> None:
        """Test get_entity_state handles connection errors."""
        mock_pool.acquire.return_value.__aenter__.side_effect = (
            asyncpg.PostgresConnectionError("Connection refused")
        )

        with pytest.raises(InfraConnectionError) as exc_info:
            await reader.get_entity_state(entity_id=uuid4())

        assert "Failed to connect" in str(exc_info.value)

    async def test_get_entity_state_timeout_error(
        self,
        reader: ProjectionReaderRegistration,
        mock_pool: MagicMock,
        mock_connection: AsyncMock,
    ) -> None:
        """Test get_entity_state handles timeout errors."""
        mock_pool.acquire.return_value.__aenter__.return_value = mock_connection
        mock_pool.acquire.return_value.__aexit__.return_value = None
        mock_connection.fetchrow.side_effect = asyncpg.QueryCanceledError("timeout")

        with pytest.raises(InfraTimeoutError) as exc_info:
            await reader.get_entity_state(entity_id=uuid4())

        assert "timed out" in str(exc_info.value)

    async def test_get_entity_state_generic_error(
        self,
        reader: ProjectionReaderRegistration,
        mock_pool: MagicMock,
        mock_connection: AsyncMock,
    ) -> None:
        """Test get_entity_state handles generic errors."""
        mock_pool.acquire.return_value.__aenter__.return_value = mock_connection
        mock_pool.acquire.return_value.__aexit__.return_value = None
        mock_connection.fetchrow.side_effect = Exception("Unknown error")

        with pytest.raises(RuntimeHostError) as exc_info:
            await reader.get_entity_state(entity_id=uuid4())

        assert "Failed to get entity state" in str(exc_info.value)

    async def test_get_registration_status_connection_error(
        self,
        reader: ProjectionReaderRegistration,
        mock_pool: MagicMock,
    ) -> None:
        """Test get_registration_status handles connection errors."""
        mock_pool.acquire.return_value.__aenter__.side_effect = (
            asyncpg.PostgresConnectionError("Connection refused")
        )

        with pytest.raises(InfraConnectionError) as exc_info:
            await reader.get_registration_status(entity_id=uuid4())

        assert "Failed to connect" in str(exc_info.value)

    async def test_get_by_state_connection_error(
        self,
        reader: ProjectionReaderRegistration,
        mock_pool: MagicMock,
    ) -> None:
        """Test get_by_state handles connection errors."""
        mock_pool.acquire.return_value.__aenter__.side_effect = (
            asyncpg.PostgresConnectionError("Connection refused")
        )

        with pytest.raises(InfraConnectionError) as exc_info:
            await reader.get_by_state(state=EnumRegistrationState.ACTIVE)

        assert "Failed to connect" in str(exc_info.value)

    async def test_get_overdue_ack_connection_error(
        self,
        reader: ProjectionReaderRegistration,
        mock_pool: MagicMock,
    ) -> None:
        """Test get_overdue_ack_registrations handles connection errors."""
        mock_pool.acquire.return_value.__aenter__.side_effect = (
            asyncpg.PostgresConnectionError("Connection refused")
        )

        with pytest.raises(InfraConnectionError) as exc_info:
            await reader.get_overdue_ack_registrations(now=datetime.now(UTC))

        assert "Failed to connect" in str(exc_info.value)

    async def test_get_overdue_liveness_connection_error(
        self,
        reader: ProjectionReaderRegistration,
        mock_pool: MagicMock,
    ) -> None:
        """Test get_overdue_liveness_registrations handles connection errors."""
        mock_pool.acquire.return_value.__aenter__.side_effect = (
            asyncpg.PostgresConnectionError("Connection refused")
        )

        with pytest.raises(InfraConnectionError) as exc_info:
            await reader.get_overdue_liveness_registrations(now=datetime.now(UTC))

        assert "Failed to connect" in str(exc_info.value)

    async def test_count_by_state_connection_error(
        self,
        reader: ProjectionReaderRegistration,
        mock_pool: MagicMock,
    ) -> None:
        """Test count_by_state handles connection errors."""
        mock_pool.acquire.return_value.__aenter__.side_effect = (
            asyncpg.PostgresConnectionError("Connection refused")
        )

        with pytest.raises(InfraConnectionError) as exc_info:
            await reader.count_by_state()

        assert "Failed to connect" in str(exc_info.value)


@pytest.mark.unit
@pytest.mark.asyncio
class TestProjectionReaderCircuitBreaker:
    """Test circuit breaker behavior."""

    async def test_circuit_breaker_opens_after_threshold_failures(
        self,
        mock_pool: MagicMock,
    ) -> None:
        """Test circuit breaker opens after threshold failures."""
        reader = ProjectionReaderRegistration(pool=mock_pool)

        # Simulate connection failures to reach threshold
        mock_pool.acquire.return_value.__aenter__.side_effect = (
            asyncpg.PostgresConnectionError("Connection refused")
        )

        # Make 5 failed calls (default threshold)
        for _ in range(5):
            with pytest.raises(InfraConnectionError):
                await reader.get_entity_state(entity_id=uuid4())

        # Circuit should now be open
        assert reader._circuit_breaker_open is True
        assert reader._circuit_breaker_failures >= 5

    async def test_circuit_breaker_blocks_when_open(
        self,
        mock_pool: MagicMock,
    ) -> None:
        """Test circuit breaker blocks operations when open."""
        reader = ProjectionReaderRegistration(pool=mock_pool)

        # Simulate connection failures to open circuit
        mock_pool.acquire.return_value.__aenter__.side_effect = (
            asyncpg.PostgresConnectionError("Connection refused")
        )

        # Exhaust threshold
        for _ in range(5):
            with pytest.raises(InfraConnectionError):
                await reader.get_entity_state(entity_id=uuid4())

        # Next call should be blocked by circuit breaker
        with pytest.raises(InfraUnavailableError) as exc_info:
            await reader.get_entity_state(entity_id=uuid4())

        assert "Circuit breaker is open" in str(exc_info.value)

    async def test_circuit_breaker_resets_on_success(
        self,
        reader: ProjectionReaderRegistration,
        mock_pool: MagicMock,
        mock_connection: AsyncMock,
    ) -> None:
        """Test circuit breaker resets after successful operation."""
        # First, simulate a failure
        mock_pool.acquire.return_value.__aenter__.side_effect = (
            asyncpg.PostgresConnectionError("Connection refused")
        )

        with pytest.raises(InfraConnectionError):
            await reader.get_entity_state(entity_id=uuid4())

        assert reader._circuit_breaker_failures == 1

        # Now simulate success
        mock_pool.acquire.return_value.__aenter__.side_effect = None
        mock_pool.acquire.return_value.__aenter__.return_value = mock_connection
        mock_pool.acquire.return_value.__aexit__.return_value = None
        mock_connection.fetchrow.return_value = create_mock_row()

        await reader.get_entity_state(entity_id=uuid4())

        # Circuit breaker should be reset
        assert reader._circuit_breaker_failures == 0
        assert reader._circuit_breaker_open is False

    async def test_circuit_breaker_blocks_all_operations(
        self,
        mock_pool: MagicMock,
    ) -> None:
        """Test circuit breaker blocks all reader operations when open."""
        reader = ProjectionReaderRegistration(pool=mock_pool)

        # Open circuit breaker
        mock_pool.acquire.return_value.__aenter__.side_effect = (
            asyncpg.PostgresConnectionError("Connection refused")
        )

        for _ in range(5):
            with pytest.raises(InfraConnectionError):
                await reader.get_entity_state(entity_id=uuid4())

        # All operations should be blocked
        with pytest.raises(InfraUnavailableError):
            await reader.get_entity_state(entity_id=uuid4())

        with pytest.raises(InfraUnavailableError):
            await reader.get_registration_status(entity_id=uuid4())

        with pytest.raises(InfraUnavailableError):
            await reader.get_by_state(state=EnumRegistrationState.ACTIVE)

        with pytest.raises(InfraUnavailableError):
            await reader.get_overdue_ack_registrations(now=datetime.now(UTC))

        with pytest.raises(InfraUnavailableError):
            await reader.get_overdue_liveness_registrations(now=datetime.now(UTC))

        with pytest.raises(InfraUnavailableError):
            await reader.count_by_state()


@pytest.mark.unit
@pytest.mark.asyncio
class TestProjectionReaderCapabilityQueries:
    """Test capability-based query methods (OMN-1134)."""

    # ============================================================
    # get_by_capability_tag tests
    # ============================================================

    async def test_get_by_capability_tag_returns_matching(
        self,
        reader: ProjectionReaderRegistration,
        mock_pool: MagicMock,
        mock_connection: AsyncMock,
    ) -> None:
        """Test get_by_capability_tag returns matching registrations."""
        mock_pool.acquire.return_value.__aenter__.return_value = mock_connection
        mock_pool.acquire.return_value.__aexit__.return_value = None

        mock_rows = [
            create_mock_row(capability_tags=["postgres.storage", "kafka.consumer"]),
            create_mock_row(capability_tags=["postgres.storage"]),
        ]
        mock_connection.fetch.return_value = mock_rows

        result = await reader.get_by_capability_tag("postgres.storage")

        assert isinstance(result, list)
        assert len(result) == 2
        for proj in result:
            assert "postgres.storage" in proj.capability_tags

    async def test_get_by_capability_tag_empty_result(
        self,
        reader: ProjectionReaderRegistration,
        mock_pool: MagicMock,
        mock_connection: AsyncMock,
    ) -> None:
        """Test get_by_capability_tag returns empty list when no matches."""
        mock_pool.acquire.return_value.__aenter__.return_value = mock_connection
        mock_pool.acquire.return_value.__aexit__.return_value = None
        mock_connection.fetch.return_value = []

        result = await reader.get_by_capability_tag("nonexistent.tag")

        assert result == []

    async def test_get_by_capability_tag_with_state_filter(
        self,
        reader: ProjectionReaderRegistration,
        mock_pool: MagicMock,
        mock_connection: AsyncMock,
    ) -> None:
        """Test get_by_capability_tag with state filter."""
        mock_pool.acquire.return_value.__aenter__.return_value = mock_connection
        mock_pool.acquire.return_value.__aexit__.return_value = None

        mock_rows = [
            create_mock_row(
                state=EnumRegistrationState.ACTIVE,
                capability_tags=["postgres.storage"],
            ),
        ]
        mock_connection.fetch.return_value = mock_rows

        result = await reader.get_by_capability_tag(
            "postgres.storage",
            state=EnumRegistrationState.ACTIVE,
        )

        assert len(result) == 1
        assert result[0].current_state == EnumRegistrationState.ACTIVE

        # Verify SQL includes state filter
        call_args = mock_connection.fetch.call_args
        sql = call_args[0][0]
        assert "current_state" in sql

    async def test_get_by_capability_tag_without_state_filter(
        self,
        reader: ProjectionReaderRegistration,
        mock_pool: MagicMock,
        mock_connection: AsyncMock,
    ) -> None:
        """Test get_by_capability_tag without state filter queries all states."""
        mock_pool.acquire.return_value.__aenter__.return_value = mock_connection
        mock_pool.acquire.return_value.__aexit__.return_value = None
        mock_connection.fetch.return_value = []

        await reader.get_by_capability_tag("postgres.storage")

        # Verify SQL does NOT include state filter (3 params: domain, tag, limit)
        call_args = mock_connection.fetch.call_args
        args = call_args[0]
        assert len(args) == 4  # sql + 3 params

    # ============================================================
    # get_by_intent_type tests
    # ============================================================

    async def test_get_by_intent_type_returns_matching(
        self,
        reader: ProjectionReaderRegistration,
        mock_pool: MagicMock,
        mock_connection: AsyncMock,
    ) -> None:
        """Test get_by_intent_type returns matching registrations."""
        mock_pool.acquire.return_value.__aenter__.return_value = mock_connection
        mock_pool.acquire.return_value.__aexit__.return_value = None

        mock_rows = [
            create_mock_row(intent_types=["postgres.upsert", "postgres.query"]),
        ]
        mock_connection.fetch.return_value = mock_rows

        result = await reader.get_by_intent_type("postgres.upsert")

        assert len(result) == 1
        assert "postgres.upsert" in result[0].intent_types

    async def test_get_by_intent_type_with_state_filter(
        self,
        reader: ProjectionReaderRegistration,
        mock_pool: MagicMock,
        mock_connection: AsyncMock,
    ) -> None:
        """Test get_by_intent_type with state filter."""
        mock_pool.acquire.return_value.__aenter__.return_value = mock_connection
        mock_pool.acquire.return_value.__aexit__.return_value = None
        mock_connection.fetch.return_value = []

        await reader.get_by_intent_type(
            "postgres.query",
            state=EnumRegistrationState.ACTIVE,
        )

        # Verify SQL includes state filter (4 params: domain, intent, state, limit)
        call_args = mock_connection.fetch.call_args
        args = call_args[0]
        assert len(args) == 5  # sql + 4 params
        assert EnumRegistrationState.ACTIVE.value in args

    # ============================================================
    # get_by_protocol tests
    # ============================================================

    async def test_get_by_protocol_returns_matching(
        self,
        reader: ProjectionReaderRegistration,
        mock_pool: MagicMock,
        mock_connection: AsyncMock,
    ) -> None:
        """Test get_by_protocol returns matching registrations."""
        mock_pool.acquire.return_value.__aenter__.return_value = mock_connection
        mock_pool.acquire.return_value.__aexit__.return_value = None

        mock_rows = [
            create_mock_row(
                protocols=["ProtocolDatabaseAdapter", "ProtocolEventPublisher"]
            ),
        ]
        mock_connection.fetch.return_value = mock_rows

        result = await reader.get_by_protocol("ProtocolDatabaseAdapter")

        assert len(result) == 1
        assert "ProtocolDatabaseAdapter" in result[0].protocols

    async def test_get_by_protocol_with_state_filter(
        self,
        reader: ProjectionReaderRegistration,
        mock_pool: MagicMock,
        mock_connection: AsyncMock,
    ) -> None:
        """Test get_by_protocol with state filter."""
        mock_pool.acquire.return_value.__aenter__.return_value = mock_connection
        mock_pool.acquire.return_value.__aexit__.return_value = None
        mock_connection.fetch.return_value = []

        await reader.get_by_protocol(
            "ProtocolEventPublisher",
            state=EnumRegistrationState.PENDING_REGISTRATION,
        )

        # Verify state filter is in params
        call_args = mock_connection.fetch.call_args
        args = call_args[0]
        assert EnumRegistrationState.PENDING_REGISTRATION.value in args

    # ============================================================
    # get_by_contract_type tests
    # ============================================================

    async def test_get_by_contract_type_returns_matching(
        self,
        reader: ProjectionReaderRegistration,
        mock_pool: MagicMock,
        mock_connection: AsyncMock,
    ) -> None:
        """Test get_by_contract_type returns matching registrations."""
        mock_pool.acquire.return_value.__aenter__.return_value = mock_connection
        mock_pool.acquire.return_value.__aexit__.return_value = None

        mock_rows = [
            create_mock_row(contract_type="effect"),
            create_mock_row(contract_type="effect"),
        ]
        mock_connection.fetch.return_value = mock_rows

        result = await reader.get_by_contract_type("effect")

        assert len(result) == 2
        for proj in result:
            assert proj.contract_type == "effect"

    async def test_get_by_contract_type_with_state_filter(
        self,
        reader: ProjectionReaderRegistration,
        mock_pool: MagicMock,
        mock_connection: AsyncMock,
    ) -> None:
        """Test get_by_contract_type with state filter."""
        mock_pool.acquire.return_value.__aenter__.return_value = mock_connection
        mock_pool.acquire.return_value.__aexit__.return_value = None

        mock_rows = [
            create_mock_row(
                contract_type="reducer",
                state=EnumRegistrationState.ACTIVE,
            ),
        ]
        mock_connection.fetch.return_value = mock_rows

        result = await reader.get_by_contract_type(
            "reducer",
            state=EnumRegistrationState.ACTIVE,
        )

        assert len(result) == 1
        assert result[0].contract_type == "reducer"
        assert result[0].current_state == EnumRegistrationState.ACTIVE

    async def test_get_by_contract_type_all_valid_types(
        self,
        reader: ProjectionReaderRegistration,
        mock_pool: MagicMock,
        mock_connection: AsyncMock,
    ) -> None:
        """Test get_by_contract_type works for all valid contract types."""
        mock_pool.acquire.return_value.__aenter__.return_value = mock_connection
        mock_pool.acquire.return_value.__aexit__.return_value = None
        mock_connection.fetch.return_value = []

        for contract_type in ["effect", "compute", "reducer", "orchestrator"]:
            await reader.get_by_contract_type(contract_type)

            # Verify contract type was passed to query
            call_args = mock_connection.fetch.call_args
            args = call_args[0]
            assert contract_type in args

    # ============================================================
    # get_by_capability_tags_all tests
    # ============================================================

    async def test_get_by_capability_tags_all_returns_matching(
        self,
        reader: ProjectionReaderRegistration,
        mock_pool: MagicMock,
        mock_connection: AsyncMock,
    ) -> None:
        """Test get_by_capability_tags_all returns registrations with ALL tags."""
        mock_pool.acquire.return_value.__aenter__.return_value = mock_connection
        mock_pool.acquire.return_value.__aexit__.return_value = None

        mock_rows = [
            create_mock_row(
                capability_tags=["postgres.storage", "transactions", "async"]
            ),
        ]
        mock_connection.fetch.return_value = mock_rows

        result = await reader.get_by_capability_tags_all(
            ["postgres.storage", "transactions"]
        )

        assert len(result) == 1
        # Both tags must be present
        assert "postgres.storage" in result[0].capability_tags
        assert "transactions" in result[0].capability_tags

    async def test_get_by_capability_tags_all_with_state_filter(
        self,
        reader: ProjectionReaderRegistration,
        mock_pool: MagicMock,
        mock_connection: AsyncMock,
    ) -> None:
        """Test get_by_capability_tags_all with state filter."""
        mock_pool.acquire.return_value.__aenter__.return_value = mock_connection
        mock_pool.acquire.return_value.__aexit__.return_value = None
        mock_connection.fetch.return_value = []

        await reader.get_by_capability_tags_all(
            ["postgres.storage"],
            state=EnumRegistrationState.ACTIVE,
        )

        # Verify SQL includes state filter
        call_args = mock_connection.fetch.call_args
        args = call_args[0]
        assert EnumRegistrationState.ACTIVE.value in args

    # ============================================================
    # get_by_capability_tags_any tests
    # ============================================================

    async def test_get_by_capability_tags_any_returns_matching(
        self,
        reader: ProjectionReaderRegistration,
        mock_pool: MagicMock,
        mock_connection: AsyncMock,
    ) -> None:
        """Test get_by_capability_tags_any returns registrations with ANY tag."""
        mock_pool.acquire.return_value.__aenter__.return_value = mock_connection
        mock_pool.acquire.return_value.__aexit__.return_value = None

        mock_rows = [
            create_mock_row(capability_tags=["postgres.storage"]),
            create_mock_row(capability_tags=["mysql.storage"]),
        ]
        mock_connection.fetch.return_value = mock_rows

        result = await reader.get_by_capability_tags_any(
            ["postgres.storage", "mysql.storage", "sqlite.storage"]
        )

        assert len(result) == 2

    async def test_get_by_capability_tags_any_with_state_filter(
        self,
        reader: ProjectionReaderRegistration,
        mock_pool: MagicMock,
        mock_connection: AsyncMock,
    ) -> None:
        """Test get_by_capability_tags_any with state filter."""
        mock_pool.acquire.return_value.__aenter__.return_value = mock_connection
        mock_pool.acquire.return_value.__aexit__.return_value = None
        mock_connection.fetch.return_value = []

        await reader.get_by_capability_tags_any(
            ["postgres.storage", "mysql.storage"],
            state=EnumRegistrationState.LIVENESS_EXPIRED,
        )

        # Verify SQL includes state filter
        call_args = mock_connection.fetch.call_args
        args = call_args[0]
        assert EnumRegistrationState.LIVENESS_EXPIRED.value in args

    # ============================================================
    # Error handling tests for capability queries
    # ============================================================

    async def test_get_by_capability_tag_connection_error(
        self,
        reader: ProjectionReaderRegistration,
        mock_pool: MagicMock,
    ) -> None:
        """Test get_by_capability_tag handles connection errors."""
        mock_pool.acquire.return_value.__aenter__.side_effect = (
            asyncpg.PostgresConnectionError("Connection refused")
        )

        with pytest.raises(InfraConnectionError) as exc_info:
            await reader.get_by_capability_tag("postgres.storage")

        assert "Failed to connect" in str(exc_info.value)

    async def test_get_by_intent_type_timeout_error(
        self,
        reader: ProjectionReaderRegistration,
        mock_pool: MagicMock,
        mock_connection: AsyncMock,
    ) -> None:
        """Test get_by_intent_type handles timeout errors."""
        mock_pool.acquire.return_value.__aenter__.return_value = mock_connection
        mock_pool.acquire.return_value.__aexit__.return_value = None
        mock_connection.fetch.side_effect = asyncpg.QueryCanceledError("timeout")

        with pytest.raises(InfraTimeoutError) as exc_info:
            await reader.get_by_intent_type("postgres.query")

        assert "timed out" in str(exc_info.value)

    async def test_get_by_protocol_generic_error(
        self,
        reader: ProjectionReaderRegistration,
        mock_pool: MagicMock,
        mock_connection: AsyncMock,
    ) -> None:
        """Test get_by_protocol handles generic errors."""
        mock_pool.acquire.return_value.__aenter__.return_value = mock_connection
        mock_pool.acquire.return_value.__aexit__.return_value = None
        mock_connection.fetch.side_effect = Exception("Unknown error")

        with pytest.raises(RuntimeHostError) as exc_info:
            await reader.get_by_protocol("ProtocolDatabaseAdapter")

        assert "Failed to query by protocol" in str(exc_info.value)

    async def test_circuit_breaker_blocks_capability_queries(
        self,
        mock_pool: MagicMock,
    ) -> None:
        """Test circuit breaker blocks capability queries when open."""
        reader = ProjectionReaderRegistration(pool=mock_pool)

        # Open circuit breaker by triggering failures
        mock_pool.acquire.return_value.__aenter__.side_effect = (
            asyncpg.PostgresConnectionError("Connection refused")
        )

        for _ in range(5):
            with pytest.raises(InfraConnectionError):
                await reader.get_entity_state(entity_id=uuid4())

        # All capability queries should be blocked
        with pytest.raises(InfraUnavailableError):
            await reader.get_by_capability_tag("postgres.storage")

        with pytest.raises(InfraUnavailableError):
            await reader.get_by_intent_type("postgres.query")

        with pytest.raises(InfraUnavailableError):
            await reader.get_by_protocol("ProtocolDatabaseAdapter")

        with pytest.raises(InfraUnavailableError):
            await reader.get_by_contract_type("effect")

        with pytest.raises(InfraUnavailableError):
            await reader.get_by_capability_tags_all(["postgres.storage"])

        with pytest.raises(InfraUnavailableError):
            await reader.get_by_capability_tags_any(["postgres.storage"])

    # ============================================================
    # Empty tags list validation tests
    # ============================================================

    async def test_get_by_capability_tags_all_rejects_empty_list(
        self,
        reader: ProjectionReaderRegistration,
    ) -> None:
        """Empty tags list should raise ValueError for get_by_capability_tags_all."""
        with pytest.raises(ValueError, match="tags list cannot be empty"):
            await reader.get_by_capability_tags_all([])

    async def test_get_by_capability_tags_any_rejects_empty_list(
        self,
        reader: ProjectionReaderRegistration,
    ) -> None:
        """Empty tags list should raise ValueError for get_by_capability_tags_any."""
        with pytest.raises(ValueError, match="tags list cannot be empty"):
            await reader.get_by_capability_tags_any([])
