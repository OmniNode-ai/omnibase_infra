# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""
Comprehensive unit tests for ProjectorRegistration.

This test suite validates:
- Projector instantiation with asyncpg connection pool
- Schema initialization (with mocked database)
- Persist method with offset-based idempotency
- Staleness detection logic
- Error handling for database failures
- Circuit breaker integration

Test Organization:
    - TestProjectorRegistrationBasics: Instantiation and initialization
    - TestProjectorRegistrationPersist: Persist method functionality
    - TestProjectorRegistrationStaleness: Staleness detection
    - TestProjectorRegistrationErrorHandling: Error scenarios
    - TestProjectorRegistrationCircuitBreaker: Circuit breaker behavior

Coverage Goals:
    - >90% code coverage for projector
    - All database operation paths tested
    - Error handling validated
    - Circuit breaker integration tested

Related Tickets:
    - OMN-944 (F1): Implement Registration Projection Schema
    - OMN-940 (F0): Define Projector Execution Model
"""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

import asyncpg
import pytest

from omnibase_infra.enums import EnumRegistrationState
from omnibase_infra.errors import (
    InfraConnectionError,
    InfraTimeoutError,
    InfraUnavailableError,
    RuntimeHostError,
)
from omnibase_infra.models.projection import (
    ModelRegistrationProjection,
    ModelSequenceInfo,
)
from omnibase_infra.models.registration.model_node_capabilities import (
    ModelNodeCapabilities,
)
from omnibase_infra.projectors.projector_registration import ProjectorRegistration


def create_test_projection(
    entity_id: None = None,
    state: EnumRegistrationState = EnumRegistrationState.ACTIVE,
    offset: int = 100,
) -> ModelRegistrationProjection:
    """Create a test projection with sensible defaults."""
    now = datetime.now(UTC)
    return ModelRegistrationProjection(
        entity_id=entity_id or uuid4(),
        domain="registration",
        current_state=state,
        node_type="effect",
        node_version="1.0.0",
        capabilities=ModelNodeCapabilities(postgres=True, read=True),
        ack_deadline=now,
        liveness_deadline=now,
        last_applied_event_id=uuid4(),
        last_applied_offset=offset,
        registered_at=now,
        updated_at=now,
    )


def create_test_sequence_info(
    sequence: int = 100,
    partition: str | None = "0",
    offset: int | None = 100,
) -> ModelSequenceInfo:
    """Create a test sequence info with sensible defaults."""
    return ModelSequenceInfo(
        sequence=sequence,
        partition=partition,
        offset=offset,
    )


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
def projector(mock_pool: MagicMock) -> ProjectorRegistration:
    """Create a ProjectorRegistration instance with mocked pool."""
    return ProjectorRegistration(pool=mock_pool)


@pytest.mark.unit
@pytest.mark.asyncio
class TestProjectorRegistrationBasics:
    """Test basic projector instantiation and initialization."""

    async def test_projector_instantiation(self, mock_pool: MagicMock) -> None:
        """Test that projector initializes correctly with connection pool."""
        projector = ProjectorRegistration(pool=mock_pool)

        assert projector._pool is mock_pool
        # Verify circuit breaker is initialized
        assert hasattr(projector, "_circuit_breaker_lock")
        assert projector._circuit_breaker_failures == 0
        assert projector._circuit_breaker_open is False

    async def test_projector_circuit_breaker_config(
        self, projector: ProjectorRegistration
    ) -> None:
        """Test that circuit breaker is configured correctly."""
        # Default config: threshold=5, reset_timeout=60.0
        assert projector.circuit_breaker_threshold == 5
        assert projector.circuit_breaker_reset_timeout == 60.0
        assert projector.service_name == "projector.registration"

    async def test_initialize_schema_success(
        self,
        projector: ProjectorRegistration,
        mock_pool: MagicMock,
        mock_connection: AsyncMock,
    ) -> None:
        """Test successful schema initialization."""
        # Setup mock context manager for pool.acquire()
        mock_pool.acquire.return_value.__aenter__.return_value = mock_connection
        mock_pool.acquire.return_value.__aexit__.return_value = None

        # Mock the schema file existence
        with patch.object(Path, "exists", return_value=True):
            with patch.object(
                Path, "read_text", return_value="CREATE TABLE IF NOT EXISTS test;"
            ):
                await projector.initialize_schema()

        # Verify execute was called
        mock_connection.execute.assert_called_once()

    async def test_initialize_schema_file_not_found(
        self, projector: ProjectorRegistration
    ) -> None:
        """Test schema initialization when schema file does not exist."""
        with patch.object(Path, "exists", return_value=False):
            with pytest.raises(RuntimeHostError) as exc_info:
                await projector.initialize_schema()

            assert "Schema file not found" in str(exc_info.value)

    async def test_initialize_schema_connection_error(
        self,
        projector: ProjectorRegistration,
        mock_pool: MagicMock,
    ) -> None:
        """Test schema initialization handles connection errors."""
        # Setup mock to raise connection error
        mock_pool.acquire.return_value.__aenter__.side_effect = (
            asyncpg.PostgresConnectionError("Connection refused")
        )

        with patch.object(Path, "exists", return_value=True):
            with patch.object(Path, "read_text", return_value="CREATE TABLE test;"):
                with pytest.raises(InfraConnectionError) as exc_info:
                    await projector.initialize_schema()

                assert "Failed to connect" in str(exc_info.value)

    async def test_initialize_schema_timeout(
        self,
        projector: ProjectorRegistration,
        mock_pool: MagicMock,
        mock_connection: AsyncMock,
    ) -> None:
        """Test schema initialization handles timeout errors."""
        mock_pool.acquire.return_value.__aenter__.return_value = mock_connection
        mock_pool.acquire.return_value.__aexit__.return_value = None
        mock_connection.execute.side_effect = asyncpg.QueryCanceledError("timeout")

        with patch.object(Path, "exists", return_value=True):
            with patch.object(Path, "read_text", return_value="CREATE TABLE test;"):
                with pytest.raises(InfraTimeoutError) as exc_info:
                    await projector.initialize_schema()

                assert "timed out" in str(exc_info.value)


@pytest.mark.unit
@pytest.mark.asyncio
class TestProjectorRegistrationPersist:
    """Test persist method functionality."""

    async def test_persist_success_returns_true(
        self,
        projector: ProjectorRegistration,
        mock_pool: MagicMock,
        mock_connection: AsyncMock,
    ) -> None:
        """Test successful persist returns True."""
        mock_pool.acquire.return_value.__aenter__.return_value = mock_connection
        mock_pool.acquire.return_value.__aexit__.return_value = None
        # fetchrow returns a record when upsert succeeds
        mock_connection.fetchrow.return_value = {"entity_id": uuid4()}

        projection = create_test_projection()
        sequence_info = create_test_sequence_info(sequence=100)

        result = await projector.persist(
            projection=projection,
            entity_id=projection.entity_id,
            domain="registration",
            sequence_info=sequence_info,
        )

        assert result is True
        mock_connection.fetchrow.assert_called_once()

    async def test_persist_stale_returns_false(
        self,
        projector: ProjectorRegistration,
        mock_pool: MagicMock,
        mock_connection: AsyncMock,
    ) -> None:
        """Test persist returns False when update is stale (rejected)."""
        mock_pool.acquire.return_value.__aenter__.return_value = mock_connection
        mock_pool.acquire.return_value.__aexit__.return_value = None
        # fetchrow returns None when WHERE clause rejects (stale)
        mock_connection.fetchrow.return_value = None

        projection = create_test_projection()
        sequence_info = create_test_sequence_info(sequence=50)  # Older sequence

        result = await projector.persist(
            projection=projection,
            entity_id=projection.entity_id,
            domain="registration",
            sequence_info=sequence_info,
        )

        assert result is False

    async def test_persist_with_correlation_id(
        self,
        projector: ProjectorRegistration,
        mock_pool: MagicMock,
        mock_connection: AsyncMock,
    ) -> None:
        """Test persist propagates correlation ID."""
        mock_pool.acquire.return_value.__aenter__.return_value = mock_connection
        mock_pool.acquire.return_value.__aexit__.return_value = None
        mock_connection.fetchrow.return_value = {"entity_id": uuid4()}

        projection = create_test_projection()
        sequence_info = create_test_sequence_info()
        correlation_id = uuid4()

        result = await projector.persist(
            projection=projection,
            entity_id=projection.entity_id,
            domain="registration",
            sequence_info=sequence_info,
            correlation_id=correlation_id,
        )

        assert result is True
        # Verify correlation_id was passed in parameters (last param)
        call_args = mock_connection.fetchrow.call_args
        assert call_args is not None
        # The 17th positional argument (after SQL) is correlation_id
        params = call_args[0][1:]  # Skip SQL
        assert correlation_id in params

    async def test_persist_connection_error(
        self,
        projector: ProjectorRegistration,
        mock_pool: MagicMock,
    ) -> None:
        """Test persist handles connection errors."""
        mock_pool.acquire.return_value.__aenter__.side_effect = (
            asyncpg.PostgresConnectionError("Connection refused")
        )

        projection = create_test_projection()
        sequence_info = create_test_sequence_info()

        with pytest.raises(InfraConnectionError) as exc_info:
            await projector.persist(
                projection=projection,
                entity_id=projection.entity_id,
                domain="registration",
                sequence_info=sequence_info,
            )

        assert "Failed to connect" in str(exc_info.value)

    async def test_persist_timeout_error(
        self,
        projector: ProjectorRegistration,
        mock_pool: MagicMock,
        mock_connection: AsyncMock,
    ) -> None:
        """Test persist handles timeout errors."""
        mock_pool.acquire.return_value.__aenter__.return_value = mock_connection
        mock_pool.acquire.return_value.__aexit__.return_value = None
        mock_connection.fetchrow.side_effect = asyncpg.QueryCanceledError("timeout")

        projection = create_test_projection()
        sequence_info = create_test_sequence_info()

        with pytest.raises(InfraTimeoutError) as exc_info:
            await projector.persist(
                projection=projection,
                entity_id=projection.entity_id,
                domain="registration",
                sequence_info=sequence_info,
            )

        assert "timed out" in str(exc_info.value)

    async def test_persist_generic_error(
        self,
        projector: ProjectorRegistration,
        mock_pool: MagicMock,
        mock_connection: AsyncMock,
    ) -> None:
        """Test persist handles generic database errors."""
        mock_pool.acquire.return_value.__aenter__.return_value = mock_connection
        mock_pool.acquire.return_value.__aexit__.return_value = None
        mock_connection.fetchrow.side_effect = Exception("Unknown error")

        projection = create_test_projection()
        sequence_info = create_test_sequence_info()

        with pytest.raises(RuntimeHostError) as exc_info:
            await projector.persist(
                projection=projection,
                entity_id=projection.entity_id,
                domain="registration",
                sequence_info=sequence_info,
            )

        assert "Failed to persist projection" in str(exc_info.value)

    async def test_persist_uses_offset_as_sequence(
        self,
        projector: ProjectorRegistration,
        mock_pool: MagicMock,
        mock_connection: AsyncMock,
    ) -> None:
        """Test persist uses offset when available for sequencing."""
        mock_pool.acquire.return_value.__aenter__.return_value = mock_connection
        mock_pool.acquire.return_value.__aexit__.return_value = None
        mock_connection.fetchrow.return_value = {"entity_id": uuid4()}

        projection = create_test_projection()
        # Sequence info with offset
        sequence_info = ModelSequenceInfo(sequence=50, partition="0", offset=100)

        await projector.persist(
            projection=projection,
            entity_id=projection.entity_id,
            domain="registration",
            sequence_info=sequence_info,
        )

        # Verify fetchrow was called (SQL params include offset value)
        call_args = mock_connection.fetchrow.call_args
        assert call_args is not None
        params = call_args[0][1:]  # Skip SQL
        # last_applied_offset should be offset (100) when offset is available
        assert 100 in params


@pytest.mark.unit
@pytest.mark.asyncio
class TestProjectorRegistrationStaleness:
    """Test staleness detection functionality."""

    async def test_is_stale_returns_false_when_no_projection_exists(
        self,
        projector: ProjectorRegistration,
        mock_pool: MagicMock,
        mock_connection: AsyncMock,
    ) -> None:
        """Test is_stale returns False when no existing projection."""
        mock_pool.acquire.return_value.__aenter__.return_value = mock_connection
        mock_pool.acquire.return_value.__aexit__.return_value = None
        # No existing row
        mock_connection.fetchrow.return_value = None

        entity_id = uuid4()
        sequence_info = create_test_sequence_info(sequence=100)

        result = await projector.is_stale(
            entity_id=entity_id,
            domain="registration",
            sequence_info=sequence_info,
        )

        assert result is False

    async def test_is_stale_returns_true_when_sequence_is_older(
        self,
        projector: ProjectorRegistration,
        mock_pool: MagicMock,
        mock_connection: AsyncMock,
    ) -> None:
        """Test is_stale returns True when incoming sequence is older."""
        mock_pool.acquire.return_value.__aenter__.return_value = mock_connection
        mock_pool.acquire.return_value.__aexit__.return_value = None
        # Existing row has offset 200
        mock_connection.fetchrow.return_value = {
            "last_applied_offset": 200,
            "last_applied_sequence": None,
        }

        entity_id = uuid4()
        # Incoming sequence is 100 (older than 200)
        sequence_info = create_test_sequence_info(sequence=100)

        result = await projector.is_stale(
            entity_id=entity_id,
            domain="registration",
            sequence_info=sequence_info,
        )

        assert result is True

    async def test_is_stale_returns_false_when_sequence_is_newer(
        self,
        projector: ProjectorRegistration,
        mock_pool: MagicMock,
        mock_connection: AsyncMock,
    ) -> None:
        """Test is_stale returns False when incoming sequence is newer."""
        mock_pool.acquire.return_value.__aenter__.return_value = mock_connection
        mock_pool.acquire.return_value.__aexit__.return_value = None
        # Existing row has offset 100
        mock_connection.fetchrow.return_value = {
            "last_applied_offset": 100,
            "last_applied_sequence": None,
        }

        entity_id = uuid4()
        # Incoming sequence is 200 (newer than 100)
        sequence_info = create_test_sequence_info(sequence=200)

        result = await projector.is_stale(
            entity_id=entity_id,
            domain="registration",
            sequence_info=sequence_info,
        )

        assert result is False

    async def test_is_stale_connection_error(
        self,
        projector: ProjectorRegistration,
        mock_pool: MagicMock,
    ) -> None:
        """Test is_stale handles connection errors."""
        mock_pool.acquire.return_value.__aenter__.side_effect = (
            asyncpg.PostgresConnectionError("Connection refused")
        )

        entity_id = uuid4()
        sequence_info = create_test_sequence_info()

        with pytest.raises(InfraConnectionError) as exc_info:
            await projector.is_stale(
                entity_id=entity_id,
                domain="registration",
                sequence_info=sequence_info,
            )

        assert "Failed to connect" in str(exc_info.value)

    async def test_is_stale_generic_error(
        self,
        projector: ProjectorRegistration,
        mock_pool: MagicMock,
        mock_connection: AsyncMock,
    ) -> None:
        """Test is_stale handles generic database errors."""
        mock_pool.acquire.return_value.__aenter__.return_value = mock_connection
        mock_pool.acquire.return_value.__aexit__.return_value = None
        mock_connection.fetchrow.side_effect = Exception("Unknown error")

        entity_id = uuid4()
        sequence_info = create_test_sequence_info()

        with pytest.raises(RuntimeHostError) as exc_info:
            await projector.is_stale(
                entity_id=entity_id,
                domain="registration",
                sequence_info=sequence_info,
            )

        assert "Failed to check staleness" in str(exc_info.value)


@pytest.mark.unit
@pytest.mark.asyncio
class TestProjectorRegistrationCircuitBreaker:
    """Test circuit breaker behavior."""

    async def test_circuit_breaker_opens_after_threshold_failures(
        self,
        mock_pool: MagicMock,
    ) -> None:
        """Test circuit breaker opens after threshold failures."""
        projector = ProjectorRegistration(pool=mock_pool)

        # Simulate connection failures to reach threshold
        mock_pool.acquire.return_value.__aenter__.side_effect = (
            asyncpg.PostgresConnectionError("Connection refused")
        )

        projection = create_test_projection()
        sequence_info = create_test_sequence_info()

        # Make 5 failed calls (default threshold)
        for _ in range(5):
            with pytest.raises(InfraConnectionError):
                await projector.persist(
                    projection=projection,
                    entity_id=projection.entity_id,
                    domain="registration",
                    sequence_info=sequence_info,
                )

        # Circuit should now be open
        assert projector._circuit_breaker_open is True
        assert projector._circuit_breaker_failures >= 5

    async def test_circuit_breaker_blocks_when_open(
        self,
        mock_pool: MagicMock,
    ) -> None:
        """Test circuit breaker blocks operations when open."""
        projector = ProjectorRegistration(pool=mock_pool)

        # Simulate connection failures to open circuit
        mock_pool.acquire.return_value.__aenter__.side_effect = (
            asyncpg.PostgresConnectionError("Connection refused")
        )

        projection = create_test_projection()
        sequence_info = create_test_sequence_info()

        # Exhaust threshold
        for _ in range(5):
            with pytest.raises(InfraConnectionError):
                await projector.persist(
                    projection=projection,
                    entity_id=projection.entity_id,
                    domain="registration",
                    sequence_info=sequence_info,
                )

        # Next call should be blocked by circuit breaker
        with pytest.raises(InfraUnavailableError) as exc_info:
            await projector.persist(
                projection=projection,
                entity_id=projection.entity_id,
                domain="registration",
                sequence_info=sequence_info,
            )

        assert "Circuit breaker is open" in str(exc_info.value)

    async def test_circuit_breaker_resets_on_success(
        self,
        projector: ProjectorRegistration,
        mock_pool: MagicMock,
        mock_connection: AsyncMock,
    ) -> None:
        """Test circuit breaker resets after successful operation."""
        # First, simulate a failure
        mock_pool.acquire.return_value.__aenter__.side_effect = (
            asyncpg.PostgresConnectionError("Connection refused")
        )

        projection = create_test_projection()
        sequence_info = create_test_sequence_info()

        with pytest.raises(InfraConnectionError):
            await projector.persist(
                projection=projection,
                entity_id=projection.entity_id,
                domain="registration",
                sequence_info=sequence_info,
            )

        assert projector._circuit_breaker_failures == 1

        # Now simulate success
        mock_pool.acquire.return_value.__aenter__.side_effect = None
        mock_pool.acquire.return_value.__aenter__.return_value = mock_connection
        mock_pool.acquire.return_value.__aexit__.return_value = None
        mock_connection.fetchrow.return_value = {"entity_id": uuid4()}

        await projector.persist(
            projection=projection,
            entity_id=projection.entity_id,
            domain="registration",
            sequence_info=sequence_info,
        )

        # Circuit breaker should be reset
        assert projector._circuit_breaker_failures == 0
        assert projector._circuit_breaker_open is False


@pytest.mark.unit
@pytest.mark.asyncio
class TestProjectorRegistrationEdgeCases:
    """Test edge cases and boundary conditions."""

    async def test_persist_with_all_optional_fields_none(
        self,
        projector: ProjectorRegistration,
        mock_pool: MagicMock,
        mock_connection: AsyncMock,
    ) -> None:
        """Test persist with optional fields set to None."""
        mock_pool.acquire.return_value.__aenter__.return_value = mock_connection
        mock_pool.acquire.return_value.__aexit__.return_value = None
        mock_connection.fetchrow.return_value = {"entity_id": uuid4()}

        now = datetime.now(UTC)
        projection = ModelRegistrationProjection(
            entity_id=uuid4(),
            current_state=EnumRegistrationState.PENDING_REGISTRATION,
            node_type="effect",
            last_applied_event_id=uuid4(),
            registered_at=now,
            updated_at=now,
            # Optional fields left as None/default
            ack_deadline=None,
            liveness_deadline=None,
            ack_timeout_emitted_at=None,
            liveness_timeout_emitted_at=None,
        )

        sequence_info = ModelSequenceInfo(sequence=1)

        result = await projector.persist(
            projection=projection,
            entity_id=projection.entity_id,
            domain="registration",
            sequence_info=sequence_info,
        )

        assert result is True

    async def test_persist_with_complex_capabilities(
        self,
        projector: ProjectorRegistration,
        mock_pool: MagicMock,
        mock_connection: AsyncMock,
    ) -> None:
        """Test persist with complex capabilities object."""
        mock_pool.acquire.return_value.__aenter__.return_value = mock_connection
        mock_pool.acquire.return_value.__aexit__.return_value = None
        mock_connection.fetchrow.return_value = {"entity_id": uuid4()}

        now = datetime.now(UTC)
        capabilities = ModelNodeCapabilities(
            postgres=True,
            read=True,
            write=True,
            database=True,
            transactions=True,
            batch_size=100,
            max_batch=1000,
            supported_types=["json", "csv", "xml"],
            config={"timeout": 30, "retry": 3},
        )

        projection = ModelRegistrationProjection(
            entity_id=uuid4(),
            current_state=EnumRegistrationState.ACTIVE,
            node_type="effect",
            capabilities=capabilities,
            last_applied_event_id=uuid4(),
            registered_at=now,
            updated_at=now,
        )

        sequence_info = create_test_sequence_info()

        result = await projector.persist(
            projection=projection,
            entity_id=projection.entity_id,
            domain="registration",
            sequence_info=sequence_info,
        )

        assert result is True

    async def test_is_stale_with_sequence_only_no_offset(
        self,
        projector: ProjectorRegistration,
        mock_pool: MagicMock,
        mock_connection: AsyncMock,
    ) -> None:
        """Test is_stale works with sequence-only (non-Kafka) transports."""
        mock_pool.acquire.return_value.__aenter__.return_value = mock_connection
        mock_pool.acquire.return_value.__aexit__.return_value = None
        # Existing row uses sequence, not offset
        mock_connection.fetchrow.return_value = {
            "last_applied_offset": None,
            "last_applied_sequence": 100,
        }

        entity_id = uuid4()
        # Non-Kafka sequence info
        sequence_info = ModelSequenceInfo(sequence=50)  # Older

        result = await projector.is_stale(
            entity_id=entity_id,
            domain="registration",
            sequence_info=sequence_info,
        )

        assert result is True

    async def test_persist_all_registration_states(
        self,
        projector: ProjectorRegistration,
        mock_pool: MagicMock,
        mock_connection: AsyncMock,
    ) -> None:
        """Test persist works with all registration states."""
        mock_pool.acquire.return_value.__aenter__.return_value = mock_connection
        mock_pool.acquire.return_value.__aexit__.return_value = None
        mock_connection.fetchrow.return_value = {"entity_id": uuid4()}

        for state in EnumRegistrationState:
            projection = create_test_projection(state=state)
            sequence_info = create_test_sequence_info()

            result = await projector.persist(
                projection=projection,
                entity_id=projection.entity_id,
                domain="registration",
                sequence_info=sequence_info,
            )

            assert result is True

    async def test_persist_custom_domain(
        self,
        projector: ProjectorRegistration,
        mock_pool: MagicMock,
        mock_connection: AsyncMock,
    ) -> None:
        """Test persist with custom domain namespace."""
        mock_pool.acquire.return_value.__aenter__.return_value = mock_connection
        mock_pool.acquire.return_value.__aexit__.return_value = None
        mock_connection.fetchrow.return_value = {"entity_id": uuid4()}

        projection = create_test_projection()
        sequence_info = create_test_sequence_info()
        custom_domain = "custom_registration_domain"

        result = await projector.persist(
            projection=projection,
            entity_id=projection.entity_id,
            domain=custom_domain,
            sequence_info=sequence_info,
        )

        assert result is True
        # Verify domain was passed to query
        call_args = mock_connection.fetchrow.call_args
        assert call_args is not None
        params = call_args[0][1:]  # Skip SQL
        assert custom_domain in params
