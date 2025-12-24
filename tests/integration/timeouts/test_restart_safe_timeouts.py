# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Integration tests for restart-safe durable timeout behavior.

These tests verify OMN-932 acceptance criteria:
- Deadlines stored in projections survive restarts
- Orchestrator queries for overdue entities periodically
- Timeout events emitted correctly
- Restart-safe behavior verified
- No in-memory-only deadlines
- Emission markers prevent duplicates

Design Principles:
    1. Use injected `now` from RuntimeTick, not system clock
    2. Simulate restart by recreating service instances
    3. Count emitted events to verify exactly-once
    4. Verify marker updates prevent duplicates

Test Categories:
    - Restart-safe timeout emission (ack and liveness)
    - Deadline persistence across restarts
    - Exactly-once emission semantics
    - Marker-based deduplication
    - Message replay handling

Related Tickets:
    - OMN-932 (C2): Durable Timeout Handling
    - OMN-944 (F1): Implement Registration Projection Schema
    - OMN-940 (F0): Define Projector Execution Model
"""

from __future__ import annotations

from collections.abc import Callable
from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING
from uuid import uuid4

import pytest

from omnibase_infra.enums import EnumRegistrationState
from omnibase_infra.models.projection import (
    ModelRegistrationProjection,
    ModelSequenceInfo,
)
from omnibase_infra.runtime.models.model_runtime_tick import ModelRuntimeTick
from omnibase_infra.services import (
    ServiceTimeoutEmission,
    ServiceTimeoutQuery,
)

if TYPE_CHECKING:
    from tests.integration.timeouts.conftest import (
        InMemoryProjectionStore,
        MockEventBus,
        MockProjector,
    )

# =============================================================================
# Test Constants
# =============================================================================

# Time offsets for deadline scenarios
OVERDUE_DEADLINE_OFFSET = timedelta(minutes=10)
"""Offset for creating past/overdue deadlines in tests."""

FUTURE_DEADLINE_OFFSET = timedelta(hours=1)
"""Offset for creating future (not yet due) deadlines in tests."""

MARKER_TIME_OFFSET = timedelta(minutes=5)
"""Offset for emission marker timestamps in tests."""

# Projection sequence values
DEFAULT_TEST_OFFSET = 100
"""Default offset/sequence value for test projections."""

DEFAULT_TEST_PARTITION = "0"
"""Default partition value for test projections."""

# Test markers
pytestmark = [
    pytest.mark.integration,
    pytest.mark.asyncio,
]


# =============================================================================
# Helper Functions
# =============================================================================


def create_timeout_query_service(
    reader: InMemoryProjectionStore,
    batch_size: int = 100,
) -> ServiceTimeoutQuery:
    """Create ServiceTimeoutQuery with mock reader.

    Args:
        reader: In-memory projection store acting as reader
        batch_size: Maximum entities to return per query

    Returns:
        ServiceTimeoutQuery configured with mock reader
    """
    return ServiceTimeoutQuery(
        projection_reader=reader,  # type: ignore[arg-type]
        batch_size=batch_size,
    )


def create_timeout_emission_service(
    query_service: ServiceTimeoutQuery,
    event_bus: MockEventBus,
    projector: MockProjector,
    environment: str = "test",
    namespace: str = "onex",
) -> ServiceTimeoutEmission:
    """Create ServiceTimeoutEmission with mock dependencies.

    Args:
        query_service: Timeout query service
        event_bus: Mock event bus for capturing events
        projector: Mock projector for marker updates
        environment: Environment identifier
        namespace: Namespace for topic routing

    Returns:
        ServiceTimeoutEmission configured with mocks
    """
    return ServiceTimeoutEmission(
        timeout_query=query_service,
        event_bus=event_bus,  # type: ignore[arg-type]
        projector=projector,  # type: ignore[arg-type]
        environment=environment,
        namespace=namespace,
    )


# =============================================================================
# Restart-Safe Timeout Emission Tests
# =============================================================================


class TestRestartSafeTimeouts:
    """Test restart-safe timeout behavior.

    These tests verify that the timeout system correctly emits events
    exactly once, even after service restarts, by using durable
    projection markers.
    """

    async def test_timeout_event_emitted_exactly_once_after_restart(
        self,
        mock_event_bus: MockEventBus,
        in_memory_store: InMemoryProjectionStore,
        mock_projector: MockProjector,
        runtime_tick_factory: Callable[..., ModelRuntimeTick],
        projection_factory: Callable[..., ModelRegistrationProjection],
    ) -> None:
        """Verify exactly one timeout event emitted even after restart.

        Scenario:
        1. Create projection with overdue ack_deadline
        2. Process first tick -> timeout event emitted, marker set
        3. Simulate restart (recreate services)
        4. Process second tick -> NO new event (marker already set)
        5. Assert exactly one event was emitted total
        """
        # Arrange: Create projection with overdue deadline
        now = datetime.now(UTC)
        past_deadline = now - OVERDUE_DEADLINE_OFFSET

        entity_id = uuid4()
        projection = projection_factory(
            entity_id=entity_id,
            state=EnumRegistrationState.AWAITING_ACK,
            ack_deadline=past_deadline,
            offset=DEFAULT_TEST_OFFSET,
        )

        # Store projection
        await in_memory_store.persist(
            projection=projection,
            entity_id=entity_id,
            domain="registration",
            sequence_info=ModelSequenceInfo(
                sequence=DEFAULT_TEST_OFFSET,
                offset=DEFAULT_TEST_OFFSET,
                partition=DEFAULT_TEST_PARTITION,
            ),
        )

        # Create first service instances (before restart)
        query_service_1 = create_timeout_query_service(in_memory_store)
        emission_service_1 = create_timeout_emission_service(
            query_service=query_service_1,
            event_bus=mock_event_bus,
            projector=mock_projector,
        )

        # First tick - should emit timeout event
        tick_1 = runtime_tick_factory(now=now, sequence_number=1)
        result_1 = await emission_service_1.process_timeouts(
            now=tick_1.now,
            tick_id=tick_1.tick_id,
            correlation_id=tick_1.correlation_id,
        )

        assert result_1.ack_timeouts_emitted == 1
        assert result_1.markers_updated == 1
        assert mock_event_bus.count_events("ack-timed-out") == 1

        # Simulate restart: Recreate services (NEW instances)
        query_service_2 = create_timeout_query_service(in_memory_store)
        emission_service_2 = create_timeout_emission_service(
            query_service=query_service_2,
            event_bus=mock_event_bus,
            projector=mock_projector,
        )

        # Second tick (after restart) - should NOT emit new event
        tick_2 = runtime_tick_factory(now=now, sequence_number=2)
        result_2 = await emission_service_2.process_timeouts(
            now=tick_2.now,
            tick_id=tick_2.tick_id,
            correlation_id=tick_2.correlation_id,
        )

        # Verify no new events emitted
        assert result_2.ack_timeouts_emitted == 0
        assert mock_event_bus.count_events("ack-timed-out") == 1  # Still just 1

    async def test_deadlines_survive_restart(
        self,
        in_memory_store: InMemoryProjectionStore,
        projection_factory: Callable[..., ModelRegistrationProjection],
    ) -> None:
        """Verify deadlines stored in projection survive service restart.

        Scenario:
        1. Create projection with future deadline
        2. Simulate restart
        3. Query projection -> deadline still present
        """
        # Arrange: Create projection with future deadline
        now = datetime.now(UTC)
        future_deadline = now + FUTURE_DEADLINE_OFFSET

        entity_id = uuid4()
        projection = projection_factory(
            entity_id=entity_id,
            state=EnumRegistrationState.AWAITING_ACK,
            ack_deadline=future_deadline,
            offset=DEFAULT_TEST_OFFSET,
        )

        # Store projection
        await in_memory_store.persist(
            projection=projection,
            entity_id=entity_id,
            domain="registration",
            sequence_info=ModelSequenceInfo(
                sequence=DEFAULT_TEST_OFFSET,
                offset=DEFAULT_TEST_OFFSET,
                partition=DEFAULT_TEST_PARTITION,
            ),
        )

        # "Restart" is simulated - in-memory store persists (like PostgreSQL would)

        # Query projection after "restart"
        stored = await in_memory_store.get_entity_state(entity_id)

        # Verify deadline survived
        assert stored is not None
        assert stored.ack_deadline == future_deadline
        assert stored.entity_id == entity_id

    async def test_only_unmarked_entities_get_timeout_events(
        self,
        mock_event_bus: MockEventBus,
        in_memory_store: InMemoryProjectionStore,
        mock_projector: MockProjector,
        runtime_tick_factory: Callable[..., ModelRuntimeTick],
        projection_factory: Callable[..., ModelRegistrationProjection],
    ) -> None:
        """Verify only entities without markers get timeout events.

        Scenario:
        1. Create 3 projections with overdue deadlines
        2. Mark 2 with emission markers
        3. Process tick -> only 1 event emitted (for unmarked entity)
        """
        # Arrange
        now = datetime.now(UTC)
        past_deadline = now - OVERDUE_DEADLINE_OFFSET
        marker_time = now - MARKER_TIME_OFFSET

        # Create 3 projections, all with overdue deadlines
        proj_unmarked = projection_factory(
            state=EnumRegistrationState.AWAITING_ACK,
            ack_deadline=past_deadline,
            ack_timeout_emitted_at=None,  # UNMARKED
            offset=DEFAULT_TEST_OFFSET,
        )
        proj_marked_1 = projection_factory(
            state=EnumRegistrationState.AWAITING_ACK,
            ack_deadline=past_deadline,
            ack_timeout_emitted_at=marker_time,  # MARKED
            offset=DEFAULT_TEST_OFFSET,
        )
        proj_marked_2 = projection_factory(
            state=EnumRegistrationState.AWAITING_ACK,
            ack_deadline=past_deadline,
            ack_timeout_emitted_at=marker_time,  # MARKED
            offset=DEFAULT_TEST_OFFSET,
        )

        # Store all projections
        for proj in [proj_unmarked, proj_marked_1, proj_marked_2]:
            await in_memory_store.persist(
                projection=proj,
                entity_id=proj.entity_id,
                domain="registration",
                sequence_info=ModelSequenceInfo(
                    sequence=DEFAULT_TEST_OFFSET,
                    offset=DEFAULT_TEST_OFFSET,
                    partition=DEFAULT_TEST_PARTITION,
                ),
            )

        # Create services
        query_service = create_timeout_query_service(in_memory_store)
        emission_service = create_timeout_emission_service(
            query_service=query_service,
            event_bus=mock_event_bus,
            projector=mock_projector,
        )

        # Process tick
        tick = runtime_tick_factory(now=now, sequence_number=1)
        result = await emission_service.process_timeouts(
            now=tick.now,
            tick_id=tick.tick_id,
            correlation_id=tick.correlation_id,
        )

        # Only 1 event should be emitted (for unmarked entity)
        assert result.ack_timeouts_emitted == 1
        assert mock_event_bus.count_events("ack-timed-out") == 1

        # Verify the emitted event is for the unmarked entity
        events = mock_event_bus.get_events_for_topic("ack-timed-out")
        assert len(events) == 1
        # Access node_id via getattr to satisfy type checker (event is typed as object)
        assert getattr(events[0], "node_id", None) == proj_unmarked.entity_id

    async def test_marker_prevents_duplicate_on_replay(
        self,
        mock_event_bus: MockEventBus,
        in_memory_store: InMemoryProjectionStore,
        mock_projector: MockProjector,
        runtime_tick_factory: Callable[..., ModelRuntimeTick],
        projection_factory: Callable[..., ModelRegistrationProjection],
    ) -> None:
        """Verify marker prevents duplicate events on message replay.

        Scenario:
        1. Create projection with overdue deadline
        2. Process tick -> event emitted, marker set
        3. Replay same tick -> no new event
        """
        # Arrange
        now = datetime.now(UTC)
        past_deadline = now - OVERDUE_DEADLINE_OFFSET

        entity_id = uuid4()
        projection = projection_factory(
            entity_id=entity_id,
            state=EnumRegistrationState.AWAITING_ACK,
            ack_deadline=past_deadline,
            offset=DEFAULT_TEST_OFFSET,
        )

        await in_memory_store.persist(
            projection=projection,
            entity_id=entity_id,
            domain="registration",
            sequence_info=ModelSequenceInfo(
                sequence=DEFAULT_TEST_OFFSET,
                offset=DEFAULT_TEST_OFFSET,
                partition=DEFAULT_TEST_PARTITION,
            ),
        )

        # Create services
        query_service = create_timeout_query_service(in_memory_store)
        emission_service = create_timeout_emission_service(
            query_service=query_service,
            event_bus=mock_event_bus,
            projector=mock_projector,
        )

        # First processing - should emit
        tick = runtime_tick_factory(now=now, sequence_number=1)
        result_1 = await emission_service.process_timeouts(
            now=tick.now,
            tick_id=tick.tick_id,
            correlation_id=tick.correlation_id,
        )

        assert result_1.ack_timeouts_emitted == 1

        # Replay SAME tick (same tick_id, same now)
        # This simulates message replay after Kafka rebalance
        result_2 = await emission_service.process_timeouts(
            now=tick.now,
            tick_id=tick.tick_id,
            correlation_id=tick.correlation_id,
        )

        # No new event should be emitted
        assert result_2.ack_timeouts_emitted == 0
        assert mock_event_bus.count_events("ack-timed-out") == 1

    async def test_liveness_expiration_exactly_once(
        self,
        mock_event_bus: MockEventBus,
        in_memory_store: InMemoryProjectionStore,
        mock_projector: MockProjector,
        runtime_tick_factory: Callable[..., ModelRuntimeTick],
        projection_factory: Callable[..., ModelRegistrationProjection],
    ) -> None:
        """Verify liveness expiration event emitted exactly once.

        Same pattern as ack timeout but for liveness_deadline.
        """
        # Arrange
        now = datetime.now(UTC)
        past_deadline = now - OVERDUE_DEADLINE_OFFSET

        entity_id = uuid4()
        projection = projection_factory(
            entity_id=entity_id,
            state=EnumRegistrationState.ACTIVE,  # Must be ACTIVE for liveness
            liveness_deadline=past_deadline,
            offset=DEFAULT_TEST_OFFSET,
        )

        await in_memory_store.persist(
            projection=projection,
            entity_id=entity_id,
            domain="registration",
            sequence_info=ModelSequenceInfo(
                sequence=DEFAULT_TEST_OFFSET,
                offset=DEFAULT_TEST_OFFSET,
                partition=DEFAULT_TEST_PARTITION,
            ),
        )

        # First tick
        query_service_1 = create_timeout_query_service(in_memory_store)
        emission_service_1 = create_timeout_emission_service(
            query_service=query_service_1,
            event_bus=mock_event_bus,
            projector=mock_projector,
        )

        tick_1 = runtime_tick_factory(now=now, sequence_number=1)
        result_1 = await emission_service_1.process_timeouts(
            now=tick_1.now,
            tick_id=tick_1.tick_id,
            correlation_id=tick_1.correlation_id,
        )

        assert result_1.liveness_expirations_emitted == 1
        assert mock_event_bus.count_events("liveness-expired") == 1

        # Simulate restart + second tick
        query_service_2 = create_timeout_query_service(in_memory_store)
        emission_service_2 = create_timeout_emission_service(
            query_service=query_service_2,
            event_bus=mock_event_bus,
            projector=mock_projector,
        )

        tick_2 = runtime_tick_factory(now=now, sequence_number=2)
        result_2 = await emission_service_2.process_timeouts(
            now=tick_2.now,
            tick_id=tick_2.tick_id,
            correlation_id=tick_2.correlation_id,
        )

        # No new event after restart
        assert result_2.liveness_expirations_emitted == 0
        assert mock_event_bus.count_events("liveness-expired") == 1


# =============================================================================
# Timeout Query Tests
# =============================================================================


class TestTimeoutQuery:
    """Tests for ServiceTimeoutQuery behavior."""

    async def test_query_returns_only_overdue_entities(
        self,
        in_memory_store: InMemoryProjectionStore,
        runtime_tick_factory: Callable[..., ModelRuntimeTick],
        projection_factory: Callable[..., ModelRegistrationProjection],
    ) -> None:
        """Verify query returns only entities past their deadline."""
        now = datetime.now(UTC)
        past_deadline = now - OVERDUE_DEADLINE_OFFSET
        future_deadline = now + OVERDUE_DEADLINE_OFFSET

        # Create projections with different deadline states
        proj_overdue = projection_factory(
            state=EnumRegistrationState.AWAITING_ACK,
            ack_deadline=past_deadline,
            offset=DEFAULT_TEST_OFFSET,
        )
        proj_not_due = projection_factory(
            state=EnumRegistrationState.AWAITING_ACK,
            ack_deadline=future_deadline,
            offset=DEFAULT_TEST_OFFSET,
        )

        for proj in [proj_overdue, proj_not_due]:
            await in_memory_store.persist(
                projection=proj,
                entity_id=proj.entity_id,
                domain="registration",
                sequence_info=ModelSequenceInfo(
                    sequence=DEFAULT_TEST_OFFSET,
                    offset=DEFAULT_TEST_OFFSET,
                    partition=DEFAULT_TEST_PARTITION,
                ),
            )

        # Query
        query_service = create_timeout_query_service(in_memory_store)
        result = await query_service.find_overdue_entities(now=now)

        # Only overdue entity returned
        assert len(result.ack_timeouts) == 1
        assert result.ack_timeouts[0].entity_id == proj_overdue.entity_id

    async def test_query_respects_injected_time(
        self,
        in_memory_store: InMemoryProjectionStore,
        projection_factory: Callable[..., ModelRegistrationProjection],
    ) -> None:
        """Verify query uses injected time, not system clock."""
        # Create projection with deadline at a specific time
        deadline = datetime(2025, 1, 15, 12, 0, 0, tzinfo=UTC)

        projection = projection_factory(
            state=EnumRegistrationState.AWAITING_ACK,
            ack_deadline=deadline,
            offset=DEFAULT_TEST_OFFSET,
        )

        await in_memory_store.persist(
            projection=projection,
            entity_id=projection.entity_id,
            domain="registration",
            sequence_info=ModelSequenceInfo(
                sequence=DEFAULT_TEST_OFFSET,
                offset=DEFAULT_TEST_OFFSET,
                partition=DEFAULT_TEST_PARTITION,
            ),
        )

        query_service = create_timeout_query_service(in_memory_store)

        # Query with time BEFORE deadline - should find nothing
        before_deadline = datetime(2025, 1, 15, 11, 0, 0, tzinfo=UTC)
        result_before = await query_service.find_overdue_entities(now=before_deadline)
        assert len(result_before.ack_timeouts) == 0

        # Query with time AFTER deadline - should find the entity
        after_deadline = datetime(2025, 1, 15, 13, 0, 0, tzinfo=UTC)
        result_after = await query_service.find_overdue_entities(now=after_deadline)
        assert len(result_after.ack_timeouts) == 1

    async def test_query_filters_by_state(
        self,
        in_memory_store: InMemoryProjectionStore,
        projection_factory: Callable[..., ModelRegistrationProjection],
    ) -> None:
        """Verify ack timeout query only returns AWAITING_ACK/ACCEPTED states."""
        now = datetime.now(UTC)
        past_deadline = now - OVERDUE_DEADLINE_OFFSET

        # Create projections in different states
        proj_awaiting = projection_factory(
            state=EnumRegistrationState.AWAITING_ACK,
            ack_deadline=past_deadline,
            offset=DEFAULT_TEST_OFFSET,
        )
        proj_active = projection_factory(
            state=EnumRegistrationState.ACTIVE,  # Wrong state for ack timeout
            ack_deadline=past_deadline,
            offset=DEFAULT_TEST_OFFSET,
        )
        proj_rejected = projection_factory(
            state=EnumRegistrationState.REJECTED,  # Wrong state
            ack_deadline=past_deadline,
            offset=DEFAULT_TEST_OFFSET,
        )

        for proj in [proj_awaiting, proj_active, proj_rejected]:
            await in_memory_store.persist(
                projection=proj,
                entity_id=proj.entity_id,
                domain="registration",
                sequence_info=ModelSequenceInfo(
                    sequence=DEFAULT_TEST_OFFSET,
                    offset=DEFAULT_TEST_OFFSET,
                    partition=DEFAULT_TEST_PARTITION,
                ),
            )

        query_service = create_timeout_query_service(in_memory_store)
        result = await query_service.find_overdue_entities(now=now)

        # Only AWAITING_ACK should be returned for ack timeout
        assert len(result.ack_timeouts) == 1
        assert result.ack_timeouts[0].entity_id == proj_awaiting.entity_id


# =============================================================================
# Emission Marker Tests
# =============================================================================


class TestEmissionMarkers:
    """Tests for emission marker behavior."""

    async def test_marker_set_after_successful_emit(
        self,
        mock_event_bus: MockEventBus,
        in_memory_store: InMemoryProjectionStore,
        mock_projector: MockProjector,
        runtime_tick_factory: Callable[..., ModelRuntimeTick],
        projection_factory: Callable[..., ModelRegistrationProjection],
    ) -> None:
        """Verify marker is set in projection after successful emit."""
        now = datetime.now(UTC)
        past_deadline = now - OVERDUE_DEADLINE_OFFSET

        entity_id = uuid4()
        projection = projection_factory(
            entity_id=entity_id,
            state=EnumRegistrationState.AWAITING_ACK,
            ack_deadline=past_deadline,
            offset=DEFAULT_TEST_OFFSET,
        )

        await in_memory_store.persist(
            projection=projection,
            entity_id=entity_id,
            domain="registration",
            sequence_info=ModelSequenceInfo(
                sequence=DEFAULT_TEST_OFFSET,
                offset=DEFAULT_TEST_OFFSET,
                partition=DEFAULT_TEST_PARTITION,
            ),
        )

        # Verify marker is initially None
        stored_before = await in_memory_store.get_entity_state(entity_id)
        assert stored_before is not None
        assert stored_before.ack_timeout_emitted_at is None

        # Process timeout
        query_service = create_timeout_query_service(in_memory_store)
        emission_service = create_timeout_emission_service(
            query_service=query_service,
            event_bus=mock_event_bus,
            projector=mock_projector,
        )

        tick = runtime_tick_factory(now=now, sequence_number=1)
        await emission_service.process_timeouts(
            now=tick.now,
            tick_id=tick.tick_id,
            correlation_id=tick.correlation_id,
        )

        # Verify marker is now set
        stored_after = await in_memory_store.get_entity_state(entity_id)
        assert stored_after is not None
        assert stored_after.ack_timeout_emitted_at is not None
        assert stored_after.ack_timeout_emitted_at == now

    async def test_liveness_marker_independent_of_ack_marker(
        self,
        mock_event_bus: MockEventBus,
        in_memory_store: InMemoryProjectionStore,
        mock_projector: MockProjector,
        runtime_tick_factory: Callable[..., ModelRuntimeTick],
        projection_factory: Callable[..., ModelRegistrationProjection],
    ) -> None:
        """Verify ack and liveness markers are independent."""
        now = datetime.now(UTC)
        past_deadline = now - OVERDUE_DEADLINE_OFFSET

        # Create ACTIVE node with both deadlines overdue
        entity_id = uuid4()
        projection = projection_factory(
            entity_id=entity_id,
            state=EnumRegistrationState.ACTIVE,
            ack_deadline=past_deadline,  # Won't trigger (wrong state for ack)
            liveness_deadline=past_deadline,  # Will trigger
            offset=DEFAULT_TEST_OFFSET,
        )

        await in_memory_store.persist(
            projection=projection,
            entity_id=entity_id,
            domain="registration",
            sequence_info=ModelSequenceInfo(
                sequence=DEFAULT_TEST_OFFSET,
                offset=DEFAULT_TEST_OFFSET,
                partition=DEFAULT_TEST_PARTITION,
            ),
        )

        # Process timeout
        query_service = create_timeout_query_service(in_memory_store)
        emission_service = create_timeout_emission_service(
            query_service=query_service,
            event_bus=mock_event_bus,
            projector=mock_projector,
        )

        tick = runtime_tick_factory(now=now, sequence_number=1)
        result = await emission_service.process_timeouts(
            now=tick.now,
            tick_id=tick.tick_id,
            correlation_id=tick.correlation_id,
        )

        # Only liveness should be emitted (ACTIVE state)
        assert result.ack_timeouts_emitted == 0
        assert result.liveness_expirations_emitted == 1

        # Verify only liveness marker is set
        stored = await in_memory_store.get_entity_state(entity_id)
        assert stored is not None
        assert stored.ack_timeout_emitted_at is None  # Not set
        assert stored.liveness_timeout_emitted_at is not None  # Set


# =============================================================================
# Edge Case Tests
# =============================================================================


class TestEdgeCases:
    """Tests for edge cases and boundary conditions."""

    async def test_no_overdue_entities_returns_empty(
        self,
        in_memory_store: InMemoryProjectionStore,
        projection_factory: Callable[..., ModelRegistrationProjection],
    ) -> None:
        """Verify empty result when no entities are overdue."""
        now = datetime.now(UTC)
        future_deadline = now + FUTURE_DEADLINE_OFFSET

        projection = projection_factory(
            state=EnumRegistrationState.AWAITING_ACK,
            ack_deadline=future_deadline,
            offset=DEFAULT_TEST_OFFSET,
        )

        await in_memory_store.persist(
            projection=projection,
            entity_id=projection.entity_id,
            domain="registration",
            sequence_info=ModelSequenceInfo(
                sequence=DEFAULT_TEST_OFFSET,
                offset=DEFAULT_TEST_OFFSET,
                partition=DEFAULT_TEST_PARTITION,
            ),
        )

        query_service = create_timeout_query_service(in_memory_store)
        result = await query_service.find_overdue_entities(now=now)

        assert len(result.ack_timeouts) == 0
        assert len(result.liveness_expirations) == 0
        assert result.total_overdue_count == 0

    async def test_process_timeouts_with_no_overdue(
        self,
        mock_event_bus: MockEventBus,
        in_memory_store: InMemoryProjectionStore,
        mock_projector: MockProjector,
        runtime_tick_factory: Callable[..., ModelRuntimeTick],
    ) -> None:
        """Verify processing with no overdue entities emits nothing."""
        now = datetime.now(UTC)

        query_service = create_timeout_query_service(in_memory_store)
        emission_service = create_timeout_emission_service(
            query_service=query_service,
            event_bus=mock_event_bus,
            projector=mock_projector,
        )

        tick = runtime_tick_factory(now=now, sequence_number=1)
        result = await emission_service.process_timeouts(
            now=tick.now,
            tick_id=tick.tick_id,
            correlation_id=tick.correlation_id,
        )

        assert result.total_emitted == 0
        assert result.markers_updated == 0
        assert not result.has_errors
        assert mock_event_bus.count_events() == 0

    async def test_batch_size_limits_query_results(
        self,
        in_memory_store: InMemoryProjectionStore,
        projection_factory: Callable[..., ModelRegistrationProjection],
    ) -> None:
        """Verify batch_size limits number of returned entities."""
        now = datetime.now(UTC)
        past_deadline = now - OVERDUE_DEADLINE_OFFSET

        # Create 10 overdue projections
        for _ in range(10):
            proj = projection_factory(
                state=EnumRegistrationState.AWAITING_ACK,
                ack_deadline=past_deadline,
                offset=DEFAULT_TEST_OFFSET,
            )
            await in_memory_store.persist(
                projection=proj,
                entity_id=proj.entity_id,
                domain="registration",
                sequence_info=ModelSequenceInfo(
                    sequence=DEFAULT_TEST_OFFSET,
                    offset=DEFAULT_TEST_OFFSET,
                    partition=DEFAULT_TEST_PARTITION,
                ),
            )

        # Query with batch_size=5
        query_service = create_timeout_query_service(in_memory_store, batch_size=5)
        result = await query_service.find_overdue_entities(now=now)

        # Should return at most 5
        assert len(result.ack_timeouts) <= 5

    async def test_multiple_ticks_process_remaining_entities(
        self,
        mock_event_bus: MockEventBus,
        in_memory_store: InMemoryProjectionStore,
        mock_projector: MockProjector,
        runtime_tick_factory: Callable[..., ModelRuntimeTick],
        projection_factory: Callable[..., ModelRegistrationProjection],
    ) -> None:
        """Verify multiple ticks can process all overdue entities."""
        now = datetime.now(UTC)
        past_deadline = now - OVERDUE_DEADLINE_OFFSET

        # Create 5 overdue projections
        for _ in range(5):
            proj = projection_factory(
                state=EnumRegistrationState.AWAITING_ACK,
                ack_deadline=past_deadline,
                offset=DEFAULT_TEST_OFFSET,
            )
            await in_memory_store.persist(
                projection=proj,
                entity_id=proj.entity_id,
                domain="registration",
                sequence_info=ModelSequenceInfo(
                    sequence=DEFAULT_TEST_OFFSET,
                    offset=DEFAULT_TEST_OFFSET,
                    partition=DEFAULT_TEST_PARTITION,
                ),
            )

        # Create service with batch_size=2
        query_service = create_timeout_query_service(in_memory_store, batch_size=2)
        emission_service = create_timeout_emission_service(
            query_service=query_service,
            event_bus=mock_event_bus,
            projector=mock_projector,
        )

        total_emitted = 0

        # Process multiple ticks until all are handled
        for seq in range(1, 10):  # Max 9 ticks
            tick = runtime_tick_factory(now=now, sequence_number=seq)
            result = await emission_service.process_timeouts(
                now=tick.now,
                tick_id=tick.tick_id,
                correlation_id=tick.correlation_id,
            )
            total_emitted += result.ack_timeouts_emitted

            if result.ack_timeouts_emitted == 0:
                break  # All processed

        # All 5 should have been processed
        assert total_emitted == 5
        assert mock_event_bus.count_events("ack-timed-out") == 5
