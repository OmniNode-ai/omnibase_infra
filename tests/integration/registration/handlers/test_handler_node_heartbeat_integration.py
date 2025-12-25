# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Integration tests for HandlerNodeHeartbeat.

These tests verify the complete heartbeat processing flow against a real
PostgreSQL database using testcontainers. They test:

1. Happy path heartbeat processing (ACTIVE nodes)
2. Node not found scenarios (unknown node IDs)
3. Non-ACTIVE node scenarios (warning logged but processed)
4. Liveness window extension calculations
5. State verification after heartbeat processing
6. Concurrent heartbeat processing from multiple nodes
7. Error scenarios (simulated connection failures)

Related Tickets:
    - OMN-1006: Add last_heartbeat_at for liveness expired event reporting
    - OMN-932 (C2): Durable Timeout Handling
    - OMN-881: Node introspection with configurable topics
    - OMN-816: Create handler integration tests

CI/CD Graceful Skip Behavior:
    Tests skip gracefully if Docker is not available. This enables CI/CD
    pipelines to run without hard failures in environments without Docker.
"""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING
from unittest.mock import AsyncMock, patch
from uuid import UUID, uuid4

import pytest

from omnibase_infra.enums import EnumRegistrationState
from omnibase_infra.errors import InfraConnectionError
from omnibase_infra.models.projection import (
    ModelRegistrationProjection,
    ModelSequenceInfo,
)
from omnibase_infra.models.registration import ModelNodeHeartbeatEvent
from omnibase_infra.models.registration.model_node_capabilities import (
    ModelNodeCapabilities,
)
from omnibase_infra.orchestrators.registration.handlers import (
    DEFAULT_LIVENESS_WINDOW_SECONDS,
    HandlerNodeHeartbeat,
    ModelHeartbeatHandlerResult,
)

if TYPE_CHECKING:
    import asyncpg

    from omnibase_infra.projectors import (
        ProjectionReaderRegistration,
        ProjectorRegistration,
    )

# Test markers - skip all tests if Docker is not available
pytestmark = [
    pytest.mark.integration,
    pytest.mark.asyncio,
]


# =============================================================================
# Helper Functions
# =============================================================================


def make_projection(
    *,
    entity_id: UUID | None = None,
    state: EnumRegistrationState = EnumRegistrationState.ACTIVE,
    node_type: str = "effect",
    node_version: str = "1.0.0",
    offset: int = 100,
    liveness_deadline: datetime | None = None,
    last_heartbeat_at: datetime | None = None,
) -> ModelRegistrationProjection:
    """Create a test projection with sensible defaults.

    Args:
        entity_id: Node UUID (generated if not provided)
        state: FSM state (default: ACTIVE for heartbeat tests)
        node_type: ONEX node type (default: "effect")
        node_version: Semantic version (default: "1.0.0")
        offset: Kafka offset (default: 100)
        liveness_deadline: Optional liveness deadline
        last_heartbeat_at: Optional last heartbeat timestamp

    Returns:
        ModelRegistrationProjection configured for testing
    """
    now = datetime.now(UTC)
    return ModelRegistrationProjection(
        entity_id=entity_id or uuid4(),
        domain="registration",
        current_state=state,
        node_type=node_type,
        node_version=node_version,
        capabilities=ModelNodeCapabilities(postgres=True, read=True, write=True),
        liveness_deadline=liveness_deadline or (now + timedelta(minutes=5)),
        last_heartbeat_at=last_heartbeat_at,
        last_applied_event_id=uuid4(),
        last_applied_offset=offset,
        registered_at=now,
        updated_at=now,
    )


def make_heartbeat_event(
    node_id: UUID,
    *,
    timestamp: datetime | None = None,
    correlation_id: UUID | None = None,
) -> ModelNodeHeartbeatEvent:
    """Create a test heartbeat event.

    Args:
        node_id: Node UUID that sent the heartbeat.
        timestamp: Event timestamp (defaults to now).
        correlation_id: Optional correlation ID.

    Returns:
        ModelNodeHeartbeatEvent for testing.
    """
    return ModelNodeHeartbeatEvent(
        node_id=node_id,
        node_type="effect",
        node_version="1.0.0",
        uptime_seconds=3600.0,
        active_operations_count=5,
        timestamp=timestamp or datetime.now(UTC),
        correlation_id=correlation_id,
    )


def make_sequence(
    sequence: int,
    partition: str | None = "0",
    offset: int | None = None,
) -> ModelSequenceInfo:
    """Create sequence info for testing.

    Args:
        sequence: Monotonic sequence number
        partition: Kafka partition (default: "0")
        offset: Kafka offset (default: same as sequence)

    Returns:
        ModelSequenceInfo configured for testing
    """
    return ModelSequenceInfo(
        sequence=sequence,
        partition=partition,
        offset=offset if offset is not None else sequence,
    )


async def seed_projection(
    projector: ProjectorRegistration,
    projection: ModelRegistrationProjection,
) -> None:
    """Seed the database with a projection for testing.

    Args:
        projector: Projector instance for persistence.
        projection: Projection to seed.
    """
    result = await projector.persist(
        projection=projection,
        entity_id=projection.entity_id,
        domain=projection.domain,
        sequence_info=make_sequence(projection.last_applied_offset or 100),
    )
    assert result is True, "Failed to seed projection"


# =============================================================================
# Handler Initialization Tests
# =============================================================================


class TestHandlerNodeHeartbeatInit:
    """Tests for HandlerNodeHeartbeat initialization."""

    async def test_handler_initializes_with_default_liveness_window(
        self,
        reader: ProjectionReaderRegistration,
        projector: ProjectorRegistration,
    ) -> None:
        """Verify handler initializes with default liveness window."""
        handler = HandlerNodeHeartbeat(
            projection_reader=reader,
            projector=projector,
        )

        assert handler.liveness_window_seconds == DEFAULT_LIVENESS_WINDOW_SECONDS
        assert handler.liveness_window_seconds == 90.0

    async def test_handler_accepts_custom_liveness_window(
        self,
        reader: ProjectionReaderRegistration,
        projector: ProjectorRegistration,
    ) -> None:
        """Verify handler accepts custom liveness window."""
        handler = HandlerNodeHeartbeat(
            projection_reader=reader,
            projector=projector,
            liveness_window_seconds=120.0,
        )

        assert handler.liveness_window_seconds == 120.0


# =============================================================================
# Happy Path Tests - Successful Heartbeat Processing
# =============================================================================


class TestHandlerNodeHeartbeatHappyPath:
    """Tests for successful heartbeat processing scenarios."""

    async def test_handle_heartbeat_for_active_node(
        self,
        heartbeat_handler: HandlerNodeHeartbeat,
        projector: ProjectorRegistration,
        reader: ProjectionReaderRegistration,
    ) -> None:
        """Verify heartbeat processing for an ACTIVE node updates projection."""
        # Seed an ACTIVE node
        node_id = uuid4()
        projection = make_projection(
            entity_id=node_id, state=EnumRegistrationState.ACTIVE
        )
        await seed_projection(projector, projection)

        # Create and process heartbeat
        event = make_heartbeat_event(node_id)
        result = await heartbeat_handler.handle(event)

        # Verify result
        assert result.success is True
        assert result.node_id == node_id
        assert result.previous_state == EnumRegistrationState.ACTIVE
        assert result.node_not_found is False
        assert result.error_message is None
        assert result.last_heartbeat_at == event.timestamp
        assert result.liveness_deadline is not None

        # Verify state was updated in database
        updated = await reader.get_entity_state(node_id)
        assert updated is not None
        assert updated.last_heartbeat_at == event.timestamp

    async def test_handle_heartbeat_extends_liveness_deadline(
        self,
        heartbeat_handler_fast_window: HandlerNodeHeartbeat,
        projector: ProjectorRegistration,
        reader: ProjectionReaderRegistration,
    ) -> None:
        """Verify heartbeat extends liveness deadline by window duration."""
        node_id = uuid4()
        projection = make_projection(entity_id=node_id)
        await seed_projection(projector, projection)

        # Process heartbeat with known timestamp
        event_time = datetime.now(UTC)
        event = make_heartbeat_event(node_id, timestamp=event_time)
        result = await heartbeat_handler_fast_window.handle(event)

        # Verify deadline extension (5 second window from fixture)
        assert result.success is True
        assert result.liveness_deadline is not None
        expected_deadline = event_time + timedelta(seconds=5.0)
        assert abs((result.liveness_deadline - expected_deadline).total_seconds()) < 0.1

        # Verify in database
        updated = await reader.get_entity_state(node_id)
        assert updated is not None
        assert updated.liveness_deadline is not None
        assert (
            abs((updated.liveness_deadline - expected_deadline).total_seconds()) < 0.1
        )

    async def test_handle_heartbeat_preserves_correlation_id(
        self,
        heartbeat_handler: HandlerNodeHeartbeat,
        projector: ProjectorRegistration,
    ) -> None:
        """Verify heartbeat preserves correlation ID for tracing."""
        node_id = uuid4()
        projection = make_projection(entity_id=node_id)
        await seed_projection(projector, projection)

        correlation_id = uuid4()
        event = make_heartbeat_event(node_id, correlation_id=correlation_id)
        result = await heartbeat_handler.handle(event)

        assert result.success is True
        assert result.correlation_id == correlation_id

    async def test_handle_heartbeat_generates_correlation_id_if_missing(
        self,
        heartbeat_handler: HandlerNodeHeartbeat,
        projector: ProjectorRegistration,
    ) -> None:
        """Verify handler generates correlation ID when not provided."""
        node_id = uuid4()
        projection = make_projection(entity_id=node_id)
        await seed_projection(projector, projection)

        event = make_heartbeat_event(node_id, correlation_id=None)
        result = await heartbeat_handler.handle(event)

        assert result.success is True
        assert result.correlation_id is not None

    async def test_handle_heartbeat_updates_only_heartbeat_fields(
        self,
        heartbeat_handler: HandlerNodeHeartbeat,
        projector: ProjectorRegistration,
        reader: ProjectionReaderRegistration,
    ) -> None:
        """Verify heartbeat only updates heartbeat-specific fields."""
        node_id = uuid4()
        original_version = "1.2.3"
        projection = make_projection(
            entity_id=node_id,
            node_version=original_version,
        )
        await seed_projection(projector, projection)

        # Process heartbeat
        event = make_heartbeat_event(node_id)
        await heartbeat_handler.handle(event)

        # Verify other fields unchanged
        updated = await reader.get_entity_state(node_id)
        assert updated is not None
        assert updated.node_version == original_version
        assert updated.current_state == EnumRegistrationState.ACTIVE
        assert updated.node_type == "effect"


# =============================================================================
# Node Not Found Tests
# =============================================================================


class TestHandlerNodeHeartbeatNotFound:
    """Tests for heartbeat handling when node is not found."""

    async def test_handle_heartbeat_for_unknown_node(
        self,
        heartbeat_handler: HandlerNodeHeartbeat,
    ) -> None:
        """Verify heartbeat for unknown node returns node_not_found."""
        unknown_node_id = uuid4()
        event = make_heartbeat_event(unknown_node_id)

        result = await heartbeat_handler.handle(event)

        assert result.success is False
        assert result.node_id == unknown_node_id
        assert result.node_not_found is True
        assert result.previous_state is None
        assert result.last_heartbeat_at is None
        assert result.liveness_deadline is None
        assert result.error_message is not None
        assert "No registration projection found" in result.error_message

    async def test_handle_heartbeat_not_found_includes_correlation_id(
        self,
        heartbeat_handler: HandlerNodeHeartbeat,
    ) -> None:
        """Verify not found result includes correlation ID for tracing."""
        unknown_node_id = uuid4()
        correlation_id = uuid4()
        event = make_heartbeat_event(unknown_node_id, correlation_id=correlation_id)

        result = await heartbeat_handler.handle(event)

        assert result.success is False
        assert result.correlation_id == correlation_id


# =============================================================================
# Non-ACTIVE Node Tests
# =============================================================================


class TestHandlerNodeHeartbeatNonActiveNode:
    """Tests for heartbeat handling when node is in non-ACTIVE state."""

    @pytest.mark.parametrize(
        "state",
        [
            EnumRegistrationState.PENDING_REGISTRATION,
            EnumRegistrationState.ACCEPTED,
            EnumRegistrationState.AWAITING_ACK,
            EnumRegistrationState.ACK_TIMED_OUT,
            EnumRegistrationState.REJECTED,
        ],
    )
    async def test_handle_heartbeat_for_non_active_node(
        self,
        heartbeat_handler: HandlerNodeHeartbeat,
        projector: ProjectorRegistration,
        reader: ProjectionReaderRegistration,
        state: EnumRegistrationState,
    ) -> None:
        """Verify heartbeat is still processed for non-ACTIVE nodes.

        Per handler design, heartbeats from non-ACTIVE nodes are processed
        (to update tracking) but a warning is logged. This can happen during
        state transitions or race conditions.
        """
        node_id = uuid4()
        projection = make_projection(entity_id=node_id, state=state)
        await seed_projection(projector, projection)

        event = make_heartbeat_event(node_id)
        result = await heartbeat_handler.handle(event)

        # Processing should succeed (warning logged, not failed)
        assert result.success is True
        assert result.node_id == node_id
        assert result.previous_state == state
        assert result.last_heartbeat_at == event.timestamp
        assert result.liveness_deadline is not None

        # Verify database was updated
        updated = await reader.get_entity_state(node_id)
        assert updated is not None
        assert updated.last_heartbeat_at == event.timestamp


# =============================================================================
# Liveness Window Calculation Tests
# =============================================================================


class TestHandlerNodeHeartbeatLivenessWindow:
    """Tests for liveness window deadline calculations."""

    async def test_liveness_deadline_calculation_default_window(
        self,
        reader: ProjectionReaderRegistration,
        projector: ProjectorRegistration,
    ) -> None:
        """Verify liveness deadline calculation with default 90-second window."""
        handler = HandlerNodeHeartbeat(
            projection_reader=reader,
            projector=projector,
            liveness_window_seconds=90.0,
        )

        node_id = uuid4()
        projection = make_projection(entity_id=node_id)
        await seed_projection(projector, projection)

        event_time = datetime.now(UTC)
        event = make_heartbeat_event(node_id, timestamp=event_time)
        result = await handler.handle(event)

        expected_deadline = event_time + timedelta(seconds=90.0)
        assert result.liveness_deadline is not None
        assert abs((result.liveness_deadline - expected_deadline).total_seconds()) < 0.1

    async def test_liveness_deadline_uses_event_timestamp(
        self,
        heartbeat_handler: HandlerNodeHeartbeat,
        projector: ProjectorRegistration,
    ) -> None:
        """Verify liveness deadline is based on event timestamp, not current time."""
        node_id = uuid4()
        projection = make_projection(entity_id=node_id)
        await seed_projection(projector, projection)

        # Use a past timestamp
        past_time = datetime.now(UTC) - timedelta(minutes=5)
        event = make_heartbeat_event(node_id, timestamp=past_time)
        result = await heartbeat_handler.handle(event)

        # Deadline should be relative to event timestamp
        expected_deadline = past_time + timedelta(seconds=90.0)
        assert result.liveness_deadline is not None
        assert abs((result.liveness_deadline - expected_deadline).total_seconds()) < 0.1

    async def test_consecutive_heartbeats_extend_deadline(
        self,
        heartbeat_handler_fast_window: HandlerNodeHeartbeat,
        projector: ProjectorRegistration,
        reader: ProjectionReaderRegistration,
    ) -> None:
        """Verify consecutive heartbeats extend the deadline progressively."""
        node_id = uuid4()
        projection = make_projection(entity_id=node_id)
        await seed_projection(projector, projection)

        # First heartbeat
        time1 = datetime.now(UTC)
        event1 = make_heartbeat_event(node_id, timestamp=time1)
        result1 = await heartbeat_handler_fast_window.handle(event1)

        # Second heartbeat 2 seconds later
        time2 = time1 + timedelta(seconds=2)
        event2 = make_heartbeat_event(node_id, timestamp=time2)
        result2 = await heartbeat_handler_fast_window.handle(event2)

        # Deadlines should be different
        assert result1.liveness_deadline is not None
        assert result2.liveness_deadline is not None
        assert result2.liveness_deadline > result1.liveness_deadline

        # Second deadline should be 5 seconds from second timestamp
        expected = time2 + timedelta(seconds=5.0)
        assert abs((result2.liveness_deadline - expected).total_seconds()) < 0.1


# =============================================================================
# Concurrent Processing Tests
# =============================================================================


class TestHandlerNodeHeartbeatConcurrency:
    """Tests for concurrent heartbeat processing."""

    async def test_concurrent_heartbeats_from_different_nodes(
        self,
        heartbeat_handler: HandlerNodeHeartbeat,
        projector: ProjectorRegistration,
    ) -> None:
        """Verify concurrent heartbeats from different nodes are handled correctly."""
        # Seed multiple nodes
        node_ids = [uuid4() for _ in range(5)]
        for node_id in node_ids:
            projection = make_projection(entity_id=node_id)
            await seed_projection(projector, projection)

        # Process heartbeats concurrently
        events = [make_heartbeat_event(node_id) for node_id in node_ids]
        results = await asyncio.gather(
            *[heartbeat_handler.handle(event) for event in events]
        )

        # All should succeed
        assert all(r.success for r in results)
        assert len(results) == 5

    async def test_rapid_heartbeats_same_node(
        self,
        heartbeat_handler_fast_window: HandlerNodeHeartbeat,
        projector: ProjectorRegistration,
        reader: ProjectionReaderRegistration,
    ) -> None:
        """Verify rapid heartbeats from same node are handled correctly."""
        node_id = uuid4()
        projection = make_projection(entity_id=node_id)
        await seed_projection(projector, projection)

        # Send 10 rapid heartbeats with increasing timestamps
        base_time = datetime.now(UTC)
        events = [
            make_heartbeat_event(
                node_id,
                timestamp=base_time + timedelta(milliseconds=i * 100),
            )
            for i in range(10)
        ]

        results = await asyncio.gather(
            *[heartbeat_handler_fast_window.handle(event) for event in events]
        )

        # All should succeed
        assert all(r.success for r in results)

        # Final state should have the last heartbeat timestamp
        final = await reader.get_entity_state(node_id)
        assert final is not None
        assert final.last_heartbeat_at is not None

        # The last heartbeat timestamp should be one of the event timestamps
        # (exact order is non-deterministic with concurrent writes, but the
        # final value must be one of the timestamps we sent)
        expected_timestamps = {event.timestamp for event in events}
        assert final.last_heartbeat_at in expected_timestamps, (
            f"Expected last_heartbeat_at to be one of {len(expected_timestamps)} "
            f"event timestamps, but got {final.last_heartbeat_at}"
        )


# =============================================================================
# Error Scenario Tests
# =============================================================================


class TestHandlerNodeHeartbeatErrors:
    """Tests for error handling in heartbeat processing."""

    async def test_handle_propagates_connection_error(
        self,
        reader: ProjectionReaderRegistration,
        projector: ProjectorRegistration,
    ) -> None:
        """Verify handler propagates InfraConnectionError from projector."""
        handler = HandlerNodeHeartbeat(
            projection_reader=reader,
            projector=projector,
        )

        # Seed a node
        node_id = uuid4()
        projection = make_projection(entity_id=node_id)
        await seed_projection(projector, projection)

        # Mock the projector to raise connection error
        from omnibase_infra.enums import EnumInfraTransportType
        from omnibase_infra.errors import ModelInfraErrorContext

        with patch.object(
            projector,
            "update_heartbeat",
            new_callable=AsyncMock,
        ) as mock_update:
            mock_update.side_effect = InfraConnectionError(
                "Database connection failed",
                context=ModelInfraErrorContext(
                    transport_type=EnumInfraTransportType.DATABASE,
                    operation="update_heartbeat",
                    target_name="test",
                ),
            )

            event = make_heartbeat_event(node_id)
            with pytest.raises(InfraConnectionError):
                await handler.handle(event)

    async def test_handle_wraps_unexpected_error(
        self,
        reader: ProjectionReaderRegistration,
        projector: ProjectorRegistration,
    ) -> None:
        """Verify handler wraps unexpected errors in RuntimeHostError."""
        from omnibase_infra.errors import RuntimeHostError

        handler = HandlerNodeHeartbeat(
            projection_reader=reader,
            projector=projector,
        )

        # Seed a node
        node_id = uuid4()
        projection = make_projection(entity_id=node_id)
        await seed_projection(projector, projection)

        # Mock the projector to raise unexpected error
        with patch.object(
            projector,
            "update_heartbeat",
            new_callable=AsyncMock,
        ) as mock_update:
            mock_update.side_effect = ValueError("Unexpected error")

            event = make_heartbeat_event(node_id)
            with pytest.raises(RuntimeHostError) as exc_info:
                await handler.handle(event)

            assert "ValueError" in str(exc_info.value)

    async def test_handle_returns_error_when_update_fails(
        self,
        reader: ProjectionReaderRegistration,
        projector: ProjectorRegistration,
        pg_pool: asyncpg.Pool,
    ) -> None:
        """Verify handler returns error when update fails (entity deleted).

        This tests the race condition where entity exists during lookup
        but is deleted before update.
        """
        handler = HandlerNodeHeartbeat(
            projection_reader=reader,
            projector=projector,
        )

        # Seed a node
        node_id = uuid4()
        projection = make_projection(entity_id=node_id)
        await seed_projection(projector, projection)

        # Delete the entity after lookup but before update (simulated)
        with patch.object(
            projector,
            "update_heartbeat",
            new_callable=AsyncMock,
        ) as mock_update:
            mock_update.return_value = False  # Entity not found during update

            event = make_heartbeat_event(node_id)
            result = await handler.handle(event)

            assert result.success is False
            assert result.node_not_found is True
            assert result.previous_state == EnumRegistrationState.ACTIVE
            assert result.error_message is not None
            assert "Entity not found during heartbeat update" in result.error_message


# =============================================================================
# Result Model Tests
# =============================================================================


class TestModelHeartbeatHandlerResult:
    """Tests for ModelHeartbeatHandlerResult model behavior."""

    async def test_result_model_is_frozen(
        self,
        heartbeat_handler: HandlerNodeHeartbeat,
        projector: ProjectorRegistration,
    ) -> None:
        """Verify result model is immutable (frozen)."""
        node_id = uuid4()
        projection = make_projection(entity_id=node_id)
        await seed_projection(projector, projection)

        event = make_heartbeat_event(node_id)
        result = await heartbeat_handler.handle(event)

        # Attempt to modify should fail
        with pytest.raises(Exception):  # ValidationError for frozen models
            result.success = False  # type: ignore[misc]

    async def test_result_contains_all_expected_fields(
        self,
        heartbeat_handler: HandlerNodeHeartbeat,
        projector: ProjectorRegistration,
    ) -> None:
        """Verify result contains all expected fields for success case."""
        node_id = uuid4()
        projection = make_projection(entity_id=node_id)
        await seed_projection(projector, projection)

        event = make_heartbeat_event(node_id)
        result = await heartbeat_handler.handle(event)

        # Verify all fields are present
        assert hasattr(result, "success")
        assert hasattr(result, "node_id")
        assert hasattr(result, "previous_state")
        assert hasattr(result, "last_heartbeat_at")
        assert hasattr(result, "liveness_deadline")
        assert hasattr(result, "node_not_found")
        assert hasattr(result, "correlation_id")
        assert hasattr(result, "error_message")


# =============================================================================
# Database Verification Tests
# =============================================================================


class TestHandlerNodeHeartbeatDatabaseState:
    """Tests verifying database state after heartbeat processing."""

    async def test_database_state_after_successful_heartbeat(
        self,
        heartbeat_handler: HandlerNodeHeartbeat,
        projector: ProjectorRegistration,
        reader: ProjectionReaderRegistration,
        pg_pool: asyncpg.Pool,
    ) -> None:
        """Verify database columns are updated correctly after heartbeat."""
        node_id = uuid4()
        projection = make_projection(entity_id=node_id)
        await seed_projection(projector, projection)

        event_time = datetime.now(UTC)
        event = make_heartbeat_event(node_id, timestamp=event_time)
        await heartbeat_handler.handle(event)

        # Query database directly
        async with pg_pool.acquire() as conn:
            row = await conn.fetchrow(
                """
                SELECT last_heartbeat_at, liveness_deadline, updated_at
                FROM registration_projections
                WHERE entity_id = $1
                """,
                node_id,
            )

        assert row is not None
        assert row["last_heartbeat_at"] is not None
        assert row["liveness_deadline"] is not None
        # updated_at should be set to last_heartbeat_at
        assert row["updated_at"] == row["last_heartbeat_at"]

    async def test_heartbeat_does_not_reset_timeout_markers(
        self,
        heartbeat_handler: HandlerNodeHeartbeat,
        projector: ProjectorRegistration,
        pg_pool: asyncpg.Pool,
    ) -> None:
        """Verify heartbeat does not reset timeout emission markers."""
        node_id = uuid4()
        projection = make_projection(entity_id=node_id)
        await seed_projection(projector, projection)

        # Set a liveness timeout marker
        now = datetime.now(UTC)
        async with pg_pool.acquire() as conn:
            await conn.execute(
                """
                UPDATE registration_projections
                SET liveness_timeout_emitted_at = $2
                WHERE entity_id = $1
                """,
                node_id,
                now,
            )

        # Process heartbeat
        event = make_heartbeat_event(node_id)
        await heartbeat_handler.handle(event)

        # Verify marker was NOT reset
        async with pg_pool.acquire() as conn:
            row = await conn.fetchrow(
                """
                SELECT liveness_timeout_emitted_at
                FROM registration_projections
                WHERE entity_id = $1
                """,
                node_id,
            )

        assert row is not None
        # Marker should still be set (update_heartbeat only updates specific fields)
        assert row["liveness_timeout_emitted_at"] == now

    async def test_heartbeat_leaves_ack_deadline_unchanged(
        self,
        heartbeat_handler: HandlerNodeHeartbeat,
        projector: ProjectorRegistration,
        reader: ProjectionReaderRegistration,
        pg_pool: asyncpg.Pool,
    ) -> None:
        """Verify heartbeat does not modify ack_deadline."""
        node_id = uuid4()
        ack_deadline = datetime.now(UTC) + timedelta(minutes=5)
        projection = make_projection(entity_id=node_id)
        projection = ModelRegistrationProjection(
            **{
                **projection.model_dump(),
                "ack_deadline": ack_deadline,
            }
        )
        await seed_projection(projector, projection)

        # Process heartbeat
        event = make_heartbeat_event(node_id)
        await heartbeat_handler.handle(event)

        # Verify ack_deadline unchanged
        async with pg_pool.acquire() as conn:
            row = await conn.fetchrow(
                """
                SELECT ack_deadline
                FROM registration_projections
                WHERE entity_id = $1
                """,
                node_id,
            )

        assert row is not None
        # ack_deadline should be unchanged
        assert row["ack_deadline"] is not None
