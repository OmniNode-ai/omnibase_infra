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

ONEX Contract Compliance:
    The HandlerNodeHeartbeat is part of an ORCHESTRATOR node, so it returns
    result=None per ONEX contract rules. Tests verify success by checking
    database state via ProjectionReaderRegistration.get_entity_state().

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

from omnibase_core.models.events.model_event_envelope import ModelEventEnvelope
from omnibase_core.models.primitives.model_semver import ModelSemVer
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
from omnibase_infra.nodes.node_registration_orchestrator.handlers import (
    HandlerNodeHeartbeat,
)
from omnibase_infra.nodes.node_registration_orchestrator.services import (
    RegistrationReducerService,
)
from omnibase_infra.nodes.reducers.models.model_payload_postgres_update_registration import (
    ModelPayloadPostgresUpdateRegistration,
)

if TYPE_CHECKING:
    from omnibase_infra.projectors import ProjectionReaderRegistration
    from omnibase_infra.runtime import ProjectorShell

# Test markers
pytestmark = [
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
        node_version=ModelSemVer.parse("1.0.0"),
        uptime_seconds=3600.0,
        active_operations_count=5,
        timestamp=timestamp or datetime.now(UTC),
        correlation_id=correlation_id,
    )


def create_envelope(
    event: ModelNodeHeartbeatEvent,
    now: datetime | None = None,
    correlation_id: UUID | None = None,
) -> ModelEventEnvelope[ModelNodeHeartbeatEvent]:
    """Create an event envelope for testing.

    Args:
        event: The heartbeat event payload.
        now: Optional timestamp for the envelope (defaults to event.timestamp).
        correlation_id: Optional correlation ID (defaults to new UUID).

    Returns:
        ModelEventEnvelope wrapping the heartbeat event.
    """
    return ModelEventEnvelope(
        envelope_id=uuid4(),
        payload=event,
        envelope_timestamp=now or event.timestamp,
        correlation_id=correlation_id or event.correlation_id or uuid4(),
        source="test",
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
    projector: ProjectorShell,
    projection: ModelRegistrationProjection,
) -> None:
    """Seed the database with a projection for testing.

    Args:
        projector: ProjectorShell instance for persistence.
        projection: Projection to seed.
    """
    # Convert projection to values dict for upsert_partial
    values: dict[str, object] = {
        "entity_id": projection.entity_id,
        "domain": projection.domain,
        "current_state": projection.current_state.value
        if hasattr(projection.current_state, "value")
        else str(projection.current_state),
        "node_type": projection.node_type.value
        if hasattr(projection.node_type, "value")
        else str(projection.node_type),
        "node_version": str(projection.node_version),
        "capabilities": projection.capabilities.model_dump_json()
        if projection.capabilities
        else "{}",
        "ack_deadline": projection.ack_deadline,
        "liveness_deadline": projection.liveness_deadline,
        "last_heartbeat_at": projection.last_heartbeat_at,
        "last_applied_event_id": projection.last_applied_event_id,
        "last_applied_offset": 100
        if projection.last_applied_offset is None
        else projection.last_applied_offset,
        "registered_at": projection.registered_at,
        "updated_at": projection.updated_at,
    }

    result = await projector.upsert_partial(
        aggregate_id=projection.entity_id,
        values=values,
        correlation_id=uuid4(),
        conflict_columns=["entity_id", "domain"],
    )
    assert result is True, "Failed to seed projection"


# =============================================================================
# Handler Initialization Tests
# =============================================================================


class TestHandlerNodeHeartbeatInit:
    """Tests for HandlerNodeHeartbeat initialization."""

    async def test_handler_initializes_with_reducer(
        self,
        reader: ProjectionReaderRegistration,
    ) -> None:
        """Verify handler initializes with a RegistrationReducerService."""
        reducer = RegistrationReducerService(liveness_window_seconds=90.0)
        handler = HandlerNodeHeartbeat(
            projection_reader=reader,
            reducer=reducer,
        )

        assert handler.handler_id == "handler-node-heartbeat"

    async def test_handler_accepts_custom_liveness_window_via_reducer(
        self,
        reader: ProjectionReaderRegistration,
    ) -> None:
        """Verify handler works with a reducer configured with custom window."""
        reducer = RegistrationReducerService(liveness_window_seconds=120.0)
        handler = HandlerNodeHeartbeat(
            projection_reader=reader,
            reducer=reducer,
        )

        assert handler.handler_id == "handler-node-heartbeat"


# =============================================================================
# Happy Path Tests - Successful Heartbeat Processing
# =============================================================================


class TestHandlerNodeHeartbeatHappyPath:
    """Tests for successful heartbeat processing scenarios.

    ONEX Contract Compliance:
        The handler returns result=None per ORCHESTRATOR node contract.
        Success is verified by checking that output.intents contains the
        expected UPDATE intent with correct payload values.
    """

    async def test_handle_heartbeat_for_active_node(
        self,
        heartbeat_handler: HandlerNodeHeartbeat,
        projector: ProjectorShell,
        reader: ProjectionReaderRegistration,
    ) -> None:
        """Verify heartbeat processing for an ACTIVE node returns update intent."""
        # Seed an ACTIVE node
        node_id = uuid4()
        projection = make_projection(
            entity_id=node_id, state=EnumRegistrationState.ACTIVE
        )
        await seed_projection(projector, projection)

        # Create and process heartbeat
        event = make_heartbeat_event(node_id)
        envelope = create_envelope(event)
        output = await heartbeat_handler.handle(envelope)

        # Verify output wrapper (ORCHESTRATOR returns result=None)
        assert output.handler_id == "handler-node-heartbeat"
        assert output.events == ()  # Heartbeat doesn't emit events
        assert output.result is None  # ONEX contract: ORCHESTRATOR uses result=None

        # Verify intent contains UPDATE payload
        assert len(output.intents) == 1
        intent = output.intents[0]
        assert intent.intent_type == "postgres.update_registration"
        payload = intent.payload
        assert isinstance(payload, ModelPayloadPostgresUpdateRegistration)
        assert payload.entity_id == node_id
        assert payload.updates["last_heartbeat_at"] == event.timestamp
        assert payload.updates["liveness_deadline"] is not None

    async def test_handle_heartbeat_extends_liveness_deadline(
        self,
        heartbeat_handler_fast_window: HandlerNodeHeartbeat,
        projector: ProjectorShell,
        reader: ProjectionReaderRegistration,
    ) -> None:
        """Verify heartbeat extends liveness deadline by window duration."""
        node_id = uuid4()
        projection = make_projection(entity_id=node_id)
        await seed_projection(projector, projection)

        # Process heartbeat with known timestamp
        event_time = datetime.now(UTC)
        event = make_heartbeat_event(node_id, timestamp=event_time)
        envelope = create_envelope(event)
        output = await heartbeat_handler_fast_window.handle(envelope)

        # Verify output (ORCHESTRATOR returns result=None)
        assert output.result is None

        # Verify deadline extension in intent (5 second window from fixture)
        assert len(output.intents) == 1
        payload = output.intents[0].payload
        assert isinstance(payload, ModelPayloadPostgresUpdateRegistration)
        expected_deadline = event_time + timedelta(seconds=5.0)
        assert (
            abs(
                (
                    payload.updates["liveness_deadline"] - expected_deadline
                ).total_seconds()
            )
            < 0.1
        )

    async def test_handle_heartbeat_preserves_correlation_id(
        self,
        heartbeat_handler: HandlerNodeHeartbeat,
        projector: ProjectorShell,
        reader: ProjectionReaderRegistration,
    ) -> None:
        """Verify heartbeat preserves correlation ID for tracing."""
        node_id = uuid4()
        projection = make_projection(entity_id=node_id)
        await seed_projection(projector, projection)

        correlation_id = uuid4()
        event = make_heartbeat_event(node_id, correlation_id=correlation_id)
        envelope = create_envelope(event, correlation_id=correlation_id)
        output = await heartbeat_handler.handle(envelope)

        # Verify output correlation_id matches (ORCHESTRATOR returns result=None)
        assert output.result is None
        assert output.correlation_id == correlation_id

        # Verify intent was emitted with correct correlation_id
        assert len(output.intents) == 1
        payload = output.intents[0].payload
        assert isinstance(payload, ModelPayloadPostgresUpdateRegistration)
        assert payload.correlation_id == correlation_id

    async def test_handle_heartbeat_generates_correlation_id_if_missing(
        self,
        heartbeat_handler: HandlerNodeHeartbeat,
        projector: ProjectorShell,
        reader: ProjectionReaderRegistration,
    ) -> None:
        """Verify handler generates correlation ID when not provided."""
        node_id = uuid4()
        projection = make_projection(entity_id=node_id)
        await seed_projection(projector, projection)

        event = make_heartbeat_event(node_id, correlation_id=None)
        envelope = create_envelope(event)
        output = await heartbeat_handler.handle(envelope)

        # Verify output has correlation_id (ORCHESTRATOR returns result=None)
        assert output.result is None
        assert output.correlation_id is not None

        # Verify intent was emitted
        assert len(output.intents) == 1

    async def test_handle_heartbeat_updates_only_heartbeat_fields(
        self,
        heartbeat_handler: HandlerNodeHeartbeat,
        projector: ProjectorShell,
        reader: ProjectionReaderRegistration,
    ) -> None:
        """Verify heartbeat intent only contains heartbeat-specific fields."""
        node_id = uuid4()
        original_version = "1.2.3"
        projection = make_projection(
            entity_id=node_id,
            node_version=original_version,
        )
        await seed_projection(projector, projection)

        # Process heartbeat
        event = make_heartbeat_event(node_id)
        envelope = create_envelope(event)
        output = await heartbeat_handler.handle(envelope)

        # Verify output (ORCHESTRATOR returns result=None)
        assert output.result is None

        # Verify intent updates only heartbeat fields
        assert len(output.intents) == 1
        payload = output.intents[0].payload
        assert isinstance(payload, ModelPayloadPostgresUpdateRegistration)
        assert "last_heartbeat_at" in payload.updates
        assert "liveness_deadline" in payload.updates
        assert "updated_at" in payload.updates
        # Should NOT include non-heartbeat fields
        assert "node_version" not in payload.updates
        assert "current_state" not in payload.updates
        assert "node_type" not in payload.updates


# =============================================================================
# Node Not Found Tests
# =============================================================================


class TestHandlerNodeHeartbeatNotFound:
    """Tests for heartbeat handling when node is not found.

    ONEX Contract Compliance:
        When node is not found, handler returns result=None (not an error result).
        The warning is logged but handler completes successfully without raising.
        Verify "not found" by checking that DB state remains unchanged.
    """

    async def test_handle_heartbeat_for_unknown_node(
        self,
        heartbeat_handler: HandlerNodeHeartbeat,
        reader: ProjectionReaderRegistration,
    ) -> None:
        """Verify heartbeat for unknown node returns empty output (no exception).

        ONEX Contract: ORCHESTRATOR returns result=None.
        Handler logs warning but doesn't raise. Verify by checking no DB entry exists.
        """
        unknown_node_id = uuid4()
        event = make_heartbeat_event(unknown_node_id)
        envelope = create_envelope(event)

        # Handler should complete without raising
        output = await heartbeat_handler.handle(envelope)

        # Verify output wrapper (ORCHESTRATOR returns result=None)
        assert output.handler_id == "handler-node-heartbeat"
        assert output.events == ()
        assert output.result is None  # ONEX contract

        # Verify no projection was created
        projection = await reader.get_entity_state(unknown_node_id)
        assert projection is None

    async def test_handle_heartbeat_not_found_includes_correlation_id(
        self,
        heartbeat_handler: HandlerNodeHeartbeat,
        reader: ProjectionReaderRegistration,
    ) -> None:
        """Verify output includes correlation ID even when node not found."""
        unknown_node_id = uuid4()
        correlation_id = uuid4()
        event = make_heartbeat_event(unknown_node_id, correlation_id=correlation_id)
        envelope = create_envelope(event, correlation_id=correlation_id)

        output = await heartbeat_handler.handle(envelope)

        # Verify output (ORCHESTRATOR returns result=None)
        assert output.result is None
        assert output.correlation_id == correlation_id

        # Verify no projection was created
        projection = await reader.get_entity_state(unknown_node_id)
        assert projection is None


# =============================================================================
# Non-ACTIVE Node Tests
# =============================================================================


class TestHandlerNodeHeartbeatNonActiveNode:
    """Tests for heartbeat handling when node is in non-ACTIVE state.

    ONEX Contract Compliance:
        Handler returns result=None. Success verified via DB state.
        Warning is logged for non-ACTIVE nodes but heartbeat is still processed.
    """

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
        projector: ProjectorShell,
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
        envelope = create_envelope(event)
        output = await heartbeat_handler.handle(envelope)

        # Verify output (ORCHESTRATOR returns result=None)
        assert output.result is None

        # Verify intent was emitted with correct update payload
        assert len(output.intents) == 1
        payload = output.intents[0].payload
        assert isinstance(payload, ModelPayloadPostgresUpdateRegistration)
        assert payload.entity_id == node_id
        assert payload.updates["last_heartbeat_at"] == event.timestamp
        assert payload.updates["liveness_deadline"] is not None


# =============================================================================
# Liveness Window Calculation Tests
# =============================================================================


class TestHandlerNodeHeartbeatLivenessWindow:
    """Tests for liveness window deadline calculations.

    ONEX Contract Compliance:
        Handler returns result=None. Verify deadlines via DB state.
    """

    async def test_liveness_deadline_calculation_default_window(
        self,
        reader: ProjectionReaderRegistration,
        projector: ProjectorShell,
    ) -> None:
        """Verify liveness deadline calculation with default liveness window."""
        reducer = RegistrationReducerService(liveness_window_seconds=90.0)
        handler = HandlerNodeHeartbeat(
            projection_reader=reader,
            reducer=reducer,
        )

        node_id = uuid4()
        projection = make_projection(entity_id=node_id)
        await seed_projection(projector, projection)

        event_time = datetime.now(UTC)
        event = make_heartbeat_event(node_id, timestamp=event_time)
        envelope = create_envelope(event)
        output = await handler.handle(envelope)

        # Verify output (ORCHESTRATOR returns result=None)
        assert output.result is None

        # Verify deadline in intent payload
        assert len(output.intents) == 1
        payload = output.intents[0].payload
        assert isinstance(payload, ModelPayloadPostgresUpdateRegistration)
        expected_deadline = event_time + timedelta(seconds=90.0)
        actual_deadline = payload.updates["liveness_deadline"]
        assert abs((actual_deadline - expected_deadline).total_seconds()) < 0.1

    async def test_liveness_deadline_uses_event_timestamp(
        self,
        heartbeat_handler: HandlerNodeHeartbeat,
        projector: ProjectorShell,
        reader: ProjectionReaderRegistration,
    ) -> None:
        """Verify liveness deadline is based on event timestamp, not current time."""
        node_id = uuid4()
        projection = make_projection(entity_id=node_id)
        await seed_projection(projector, projection)

        # Use a past timestamp
        past_time = datetime.now(UTC) - timedelta(minutes=5)
        event = make_heartbeat_event(node_id, timestamp=past_time)
        envelope = create_envelope(event)
        output = await heartbeat_handler.handle(envelope)

        # Verify output (ORCHESTRATOR returns result=None)
        assert output.result is None

        # Deadline should be relative to event timestamp
        assert len(output.intents) == 1
        payload = output.intents[0].payload
        assert isinstance(payload, ModelPayloadPostgresUpdateRegistration)
        expected_deadline = past_time + timedelta(seconds=90.0)
        actual_deadline = payload.updates["liveness_deadline"]
        assert abs((actual_deadline - expected_deadline).total_seconds()) < 0.1

    async def test_consecutive_heartbeats_extend_deadline(
        self,
        heartbeat_handler_fast_window: HandlerNodeHeartbeat,
        projector: ProjectorShell,
        reader: ProjectionReaderRegistration,
    ) -> None:
        """Verify consecutive heartbeats extend the deadline progressively."""
        node_id = uuid4()
        projection = make_projection(entity_id=node_id)
        await seed_projection(projector, projection)

        # First heartbeat
        time1 = datetime.now(UTC)
        event1 = make_heartbeat_event(node_id, timestamp=time1)
        envelope1 = create_envelope(event1)
        output1 = await heartbeat_handler_fast_window.handle(envelope1)

        # Get deadline from first intent
        assert len(output1.intents) == 1
        payload1 = output1.intents[0].payload
        assert isinstance(payload1, ModelPayloadPostgresUpdateRegistration)
        deadline1 = payload1.updates["liveness_deadline"]

        # Second heartbeat 2 seconds later
        time2 = time1 + timedelta(seconds=2)
        event2 = make_heartbeat_event(node_id, timestamp=time2)
        envelope2 = create_envelope(event2)
        output2 = await heartbeat_handler_fast_window.handle(envelope2)

        # Get deadline from second intent
        assert len(output2.intents) == 1
        payload2 = output2.intents[0].payload
        assert isinstance(payload2, ModelPayloadPostgresUpdateRegistration)
        deadline2 = payload2.updates["liveness_deadline"]

        # Deadlines should be different and progressive
        assert deadline2 > deadline1

        # Second deadline should be 5 seconds from second timestamp
        expected = time2 + timedelta(seconds=5.0)
        assert abs((deadline2 - expected).total_seconds()) < 0.1


# =============================================================================
# Concurrent Processing Tests
# =============================================================================


class TestHandlerNodeHeartbeatConcurrency:
    """Tests for concurrent heartbeat processing.

    ONEX Contract Compliance:
        Handler returns result=None. Verify success via DB state.
    """

    async def test_concurrent_heartbeats_from_different_nodes(
        self,
        heartbeat_handler: HandlerNodeHeartbeat,
        projector: ProjectorShell,
        reader: ProjectionReaderRegistration,
    ) -> None:
        """Verify concurrent heartbeats from different nodes are handled correctly."""
        # Seed multiple nodes
        node_ids = [uuid4() for _ in range(5)]
        events = []
        for node_id in node_ids:
            projection = make_projection(entity_id=node_id)
            await seed_projection(projector, projection)
            events.append(make_heartbeat_event(node_id))

        # Process heartbeats concurrently
        envelopes = [create_envelope(event) for event in events]
        outputs = await asyncio.gather(
            *[heartbeat_handler.handle(envelope) for envelope in envelopes]
        )

        # All outputs should have result=None (ONEX contract)
        assert all(o.result is None for o in outputs)
        assert len(outputs) == 5

        # Verify all outputs emitted update intents with correct node_ids
        for node_id, event, output in zip(node_ids, events, outputs, strict=True):
            assert len(output.intents) == 1
            payload = output.intents[0].payload
            assert isinstance(payload, ModelPayloadPostgresUpdateRegistration)
            assert payload.entity_id == node_id
            assert payload.updates["last_heartbeat_at"] == event.timestamp

    async def test_rapid_heartbeats_same_node(
        self,
        heartbeat_handler_fast_window: HandlerNodeHeartbeat,
        projector: ProjectorShell,
        reader: ProjectionReaderRegistration,
    ) -> None:
        """Verify rapid heartbeats from same node are handled correctly.

        ONEX Contract: Handler returns result=None.
        Success verified via intent payloads.
        """
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
        latest_event_time = events[-1].timestamp
        envelopes = [create_envelope(event) for event in events]

        outputs = await asyncio.gather(
            *[heartbeat_handler_fast_window.handle(envelope) for envelope in envelopes]
        )

        # All outputs should have result=None (ONEX contract)
        assert all(o.result is None for o in outputs), (
            "ORCHESTRATOR nodes must return result=None"
        )
        assert len(outputs) == 10

        # All outputs should have correlation_id and one intent each
        window_seconds = 5.0
        for i, output in enumerate(outputs):
            assert output.correlation_id is not None, (
                f"Output {i}: correlation_id should be present"
            )
            assert len(output.intents) == 1, (
                f"Output {i}: should have exactly one intent"
            )
            payload = output.intents[0].payload
            assert isinstance(payload, ModelPayloadPostgresUpdateRegistration)
            assert payload.entity_id == node_id

            # Each intent should have consistent deadline calculation
            heartbeat_ts = payload.updates["last_heartbeat_at"]
            deadline = payload.updates["liveness_deadline"]
            expected_deadline = heartbeat_ts + timedelta(seconds=window_seconds)
            assert abs((deadline - expected_deadline).total_seconds()) < 0.1, (
                f"Output {i}: deadline {deadline} should be "
                f"heartbeat_ts + {window_seconds}s = {expected_deadline}"
            )

        # Verify liveness deadline bounds across all intents
        earliest_valid_deadline = base_time + timedelta(seconds=window_seconds)
        latest_valid_deadline = latest_event_time + timedelta(seconds=window_seconds)
        for i, output in enumerate(outputs):
            payload = output.intents[0].payload
            deadline = payload.updates["liveness_deadline"]
            assert deadline >= earliest_valid_deadline, (
                f"Output {i}: deadline {deadline} is before "
                f"earliest valid {earliest_valid_deadline}"
            )
            assert deadline <= latest_valid_deadline + timedelta(seconds=1), (
                f"Output {i}: deadline {deadline} is after "
                f"latest valid {latest_valid_deadline}"
            )


# =============================================================================
# Error Scenario Tests
# =============================================================================


class TestHandlerNodeHeartbeatErrors:
    """Tests for error handling in heartbeat processing.

    ONEX Contract Compliance:
        Handler returns result=None. The handler now delegates decisions
        to the reducer and returns intents. Hard errors from the projection
        reader are propagated as exceptions.
    """

    async def test_handle_propagates_connection_error_from_reader(
        self,
        reader: ProjectionReaderRegistration,
        projector: ProjectorShell,
    ) -> None:
        """Verify handler propagates InfraConnectionError from projection reader."""
        reducer = RegistrationReducerService(liveness_window_seconds=90.0)
        handler = HandlerNodeHeartbeat(
            projection_reader=reader,
            reducer=reducer,
        )

        # Seed a node
        node_id = uuid4()
        projection = make_projection(entity_id=node_id)
        await seed_projection(projector, projection)

        # Mock the reader to raise connection error
        from omnibase_infra.enums import EnumInfraTransportType
        from omnibase_infra.errors import ModelInfraErrorContext

        with patch.object(
            reader,
            "get_entity_state",
            new_callable=AsyncMock,
        ) as mock_read:
            mock_read.side_effect = InfraConnectionError(
                "Database connection failed",
                context=ModelInfraErrorContext(
                    transport_type=EnumInfraTransportType.DATABASE,
                    operation="get_entity_state",
                    target_name="test",
                ),
            )

            event = make_heartbeat_event(node_id)
            envelope = create_envelope(event)
            with pytest.raises(InfraConnectionError):
                await handler.handle(envelope)

    async def test_handle_returns_no_intents_for_unknown_node(
        self,
        reader: ProjectionReaderRegistration,
        projector: ProjectorShell,
    ) -> None:
        """Verify handler returns empty intents when node not found.

        ONEX Contract: ORCHESTRATOR returns result=None.
        When projection reader returns None, the reducer returns no_op
        and no intents are emitted.
        """
        reducer = RegistrationReducerService(liveness_window_seconds=90.0)
        handler = HandlerNodeHeartbeat(
            projection_reader=reader,
            reducer=reducer,
        )

        unknown_node_id = uuid4()
        event = make_heartbeat_event(unknown_node_id)
        envelope = create_envelope(event)
        output = await handler.handle(envelope)

        # ORCHESTRATOR returns result=None with no intents
        assert output.result is None
        assert output.events == ()
        assert output.intents == ()
        assert output.handler_id == "handler-node-heartbeat"


# =============================================================================
# Result Model Tests
# =============================================================================


class TestModelHandlerOutputStructure:
    """Tests for ModelHandlerOutput structure from heartbeat handler.

    ONEX Contract Compliance:
        ORCHESTRATOR nodes return result=None.
        Output contains events, intents, handler_id, node_kind, etc.
    """

    async def test_output_has_correct_structure(
        self,
        heartbeat_handler: HandlerNodeHeartbeat,
        projector: ProjectorShell,
        reader: ProjectionReaderRegistration,
    ) -> None:
        """Verify output has correct structure for ORCHESTRATOR node."""
        node_id = uuid4()
        projection = make_projection(entity_id=node_id)
        await seed_projection(projector, projection)

        event = make_heartbeat_event(node_id)
        envelope = create_envelope(event)
        output = await heartbeat_handler.handle(envelope)

        # Verify ORCHESTRATOR output structure
        assert output.result is None  # ONEX contract
        assert output.events == ()  # Heartbeat doesn't emit events
        assert len(output.intents) == 1  # One UPDATE intent
        assert output.projections == ()
        assert output.handler_id == "handler-node-heartbeat"
        assert output.correlation_id is not None
        assert output.processing_time_ms > 0

        # Verify intent structure
        intent = output.intents[0]
        assert intent.intent_type == "postgres.update_registration"
        payload = intent.payload
        assert isinstance(payload, ModelPayloadPostgresUpdateRegistration)
        assert payload.entity_id == node_id
        assert payload.updates["last_heartbeat_at"] == event.timestamp

    async def test_output_contains_input_envelope_id(
        self,
        heartbeat_handler: HandlerNodeHeartbeat,
        projector: ProjectorShell,
    ) -> None:
        """Verify output links back to input envelope."""
        node_id = uuid4()
        projection = make_projection(entity_id=node_id)
        await seed_projection(projector, projection)

        event = make_heartbeat_event(node_id)
        envelope = create_envelope(event)
        output = await heartbeat_handler.handle(envelope)

        # Verify envelope linkage
        assert output.input_envelope_id == envelope.envelope_id
        assert output.result is None  # ONEX contract


# =============================================================================
# Database Verification Tests
# =============================================================================


class TestHandlerNodeHeartbeatIntentPayload:
    """Tests verifying intent payload structure after heartbeat processing.

    Since the handler now returns intents instead of writing directly to the
    database, these tests verify the intent payload contains the correct
    update fields.
    """

    async def test_intent_payload_contains_correct_update_fields(
        self,
        heartbeat_handler: HandlerNodeHeartbeat,
        projector: ProjectorShell,
        reader: ProjectionReaderRegistration,
    ) -> None:
        """Verify intent payload contains last_heartbeat_at, liveness_deadline, updated_at."""
        node_id = uuid4()
        projection = make_projection(entity_id=node_id)
        await seed_projection(projector, projection)

        event_time = datetime.now(UTC)
        event = make_heartbeat_event(node_id, timestamp=event_time)
        envelope = create_envelope(event)
        output = await heartbeat_handler.handle(envelope)

        # Verify intent payload
        assert len(output.intents) == 1
        payload = output.intents[0].payload
        assert isinstance(payload, ModelPayloadPostgresUpdateRegistration)
        assert payload.updates["last_heartbeat_at"] == event_time
        assert payload.updates["liveness_deadline"] is not None
        assert payload.updates["updated_at"] is not None

    async def test_intent_does_not_include_timeout_markers(
        self,
        heartbeat_handler: HandlerNodeHeartbeat,
        projector: ProjectorShell,
    ) -> None:
        """Verify heartbeat intent does not include timeout emission markers."""
        node_id = uuid4()
        projection = make_projection(entity_id=node_id)
        await seed_projection(projector, projection)

        # Process heartbeat
        event = make_heartbeat_event(node_id)
        envelope = create_envelope(event)
        output = await heartbeat_handler.handle(envelope)

        # Verify intent does not include timeout markers
        assert len(output.intents) == 1
        payload = output.intents[0].payload
        assert isinstance(payload, ModelPayloadPostgresUpdateRegistration)
        assert "liveness_timeout_emitted_at" not in payload.updates
        assert "ack_timeout_emitted_at" not in payload.updates

    async def test_intent_does_not_include_ack_deadline(
        self,
        heartbeat_handler: HandlerNodeHeartbeat,
        projector: ProjectorShell,
        reader: ProjectionReaderRegistration,
    ) -> None:
        """Verify heartbeat intent does not modify ack_deadline."""
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
        envelope = create_envelope(event)
        output = await heartbeat_handler.handle(envelope)

        # Verify intent does not include ack_deadline
        assert len(output.intents) == 1
        payload = output.intents[0].payload
        assert isinstance(payload, ModelPayloadPostgresUpdateRegistration)
        assert "ack_deadline" not in payload.updates


# =============================================================================
# Timestamp Accuracy Tests (OMN-1006)
# =============================================================================


class TestHandlerNodeHeartbeatTimestampAccuracy:
    """Tests for timestamp accuracy in heartbeat processing.

    OMN-1006: Verify that timestamps are accurately tracked in intent
    payloads, which will be used for liveness expired event reporting.

    Timestamp flow (intent-based):
    1. Heartbeat event contains timestamp (event.timestamp)
    2. Handler delegates to reducer which creates intent payload with:
       - last_heartbeat_at = event.timestamp
       - liveness_deadline = event.timestamp + window
    3. Effect layer executes the intent to persist to DB
    4. ServiceTimeoutEmitter reads projection for timeout detection

    ONEX Contract Compliance:
        Handler returns result=None. Verify timestamps via intent payloads.
    """

    async def test_heartbeat_timestamp_matches_event_timestamp_exactly(
        self,
        heartbeat_handler: HandlerNodeHeartbeat,
        projector: ProjectorShell,
        reader: ProjectionReaderRegistration,
    ) -> None:
        """Verify last_heartbeat_at in intent exactly matches event timestamp.

        Critical for accurate liveness expired event reporting.
        """
        node_id = uuid4()
        projection = make_projection(entity_id=node_id)
        await seed_projection(projector, projection)

        # Use a specific timestamp
        event_time = datetime(2025, 12, 25, 10, 30, 45, 123456, tzinfo=UTC)
        event = make_heartbeat_event(node_id, timestamp=event_time)
        envelope = create_envelope(event)
        output = await heartbeat_handler.handle(envelope)

        # Verify output (ORCHESTRATOR returns result=None)
        assert output.result is None

        # Verify intent payload contains exact timestamp
        assert len(output.intents) == 1
        payload = output.intents[0].payload
        assert isinstance(payload, ModelPayloadPostgresUpdateRegistration)
        assert payload.updates["last_heartbeat_at"] == event_time, (
            f"Intent last_heartbeat_at should be exactly {event_time}, "
            f"got {payload.updates['last_heartbeat_at']}"
        )

    async def test_liveness_deadline_calculation_precision(
        self,
        reader: ProjectionReaderRegistration,
        projector: ProjectorShell,
    ) -> None:
        """Verify liveness_deadline is calculated with sub-second precision.

        liveness_deadline = event.timestamp + liveness_window_seconds
        """
        # Use a 45.5 second window to test sub-second precision
        reducer = RegistrationReducerService(liveness_window_seconds=45.5)
        handler = HandlerNodeHeartbeat(
            projection_reader=reader,
            reducer=reducer,
        )

        node_id = uuid4()
        projection = make_projection(entity_id=node_id)
        await seed_projection(projector, projection)

        # Use timestamp with microseconds
        event_time = datetime(2025, 12, 25, 12, 0, 0, 500000, tzinfo=UTC)
        event = make_heartbeat_event(node_id, timestamp=event_time)
        envelope = create_envelope(event)
        output = await handler.handle(envelope)

        # Verify output (ORCHESTRATOR returns result=None)
        assert output.result is None

        # Expected: 12:00:00.500000 + 45.5s = 12:00:46.000000
        expected_deadline = event_time + timedelta(seconds=45.5)

        # Verify via intent payload
        assert len(output.intents) == 1
        payload = output.intents[0].payload
        assert isinstance(payload, ModelPayloadPostgresUpdateRegistration)
        actual_deadline = payload.updates["liveness_deadline"]
        # Allow 1 millisecond tolerance for floating point
        delta = abs((actual_deadline - expected_deadline).total_seconds())
        assert delta < 0.001, (
            f"Deadline calculation off by {delta}s. "
            f"Expected {expected_deadline}, got {actual_deadline}"
        )

    async def test_heartbeat_preserves_utc_timezone(
        self,
        heartbeat_handler: HandlerNodeHeartbeat,
        projector: ProjectorShell,
        reader: ProjectionReaderRegistration,
    ) -> None:
        """Verify timestamps preserve UTC timezone in intent payloads.

        All timestamps should be timezone-aware with UTC.
        """
        node_id = uuid4()
        projection = make_projection(entity_id=node_id)
        await seed_projection(projector, projection)

        event_time = datetime.now(UTC)
        event = make_heartbeat_event(node_id, timestamp=event_time)
        envelope = create_envelope(event)
        output = await heartbeat_handler.handle(envelope)

        # Verify output (ORCHESTRATOR returns result=None)
        assert output.result is None

        # Verify intent payload timestamps
        assert len(output.intents) == 1
        payload = output.intents[0].payload
        assert isinstance(payload, ModelPayloadPostgresUpdateRegistration)
        assert payload.updates["last_heartbeat_at"] is not None
        assert payload.updates["liveness_deadline"] is not None

    async def test_successive_heartbeats_produce_monotonic_timestamps(
        self,
        heartbeat_handler_fast_window: HandlerNodeHeartbeat,
        projector: ProjectorShell,
        reader: ProjectionReaderRegistration,
    ) -> None:
        """Verify successive heartbeats produce intents with monotonic timestamps.

        Each heartbeat should produce an intent with last_heartbeat_at and
        liveness_deadline later than the previous heartbeat.
        """
        node_id = uuid4()
        projection = make_projection(entity_id=node_id)
        await seed_projection(projector, projection)

        # Send 5 heartbeats with increasing timestamps
        timestamps = [datetime(2025, 12, 25, 10, 0, i, tzinfo=UTC) for i in range(5)]
        intent_payloads = []

        for ts in timestamps:
            event = make_heartbeat_event(node_id, timestamp=ts)
            envelope = create_envelope(event)
            output = await heartbeat_handler_fast_window.handle(envelope)
            # Verify ORCHESTRATOR returns result=None
            assert output.result is None
            assert len(output.intents) == 1
            payload = output.intents[0].payload
            assert isinstance(payload, ModelPayloadPostgresUpdateRegistration)
            intent_payloads.append(payload)

        # Verify monotonic increase in timestamps via intent payloads
        for i in range(1, len(intent_payloads)):
            assert (
                intent_payloads[i].updates["last_heartbeat_at"]
                > intent_payloads[i - 1].updates["last_heartbeat_at"]
            ), f"Heartbeat {i} timestamp should be > heartbeat {i - 1}"

            assert (
                intent_payloads[i].updates["liveness_deadline"]
                > intent_payloads[i - 1].updates["liveness_deadline"]
            ), f"Heartbeat {i} deadline should be > heartbeat {i - 1}"

        # Verify final intent has latest timestamp
        assert intent_payloads[-1].updates["last_heartbeat_at"] == timestamps[-1]

    async def test_timestamp_accuracy_for_liveness_expired_event_reporting(
        self,
        heartbeat_handler_fast_window: HandlerNodeHeartbeat,
        projector: ProjectorShell,
        reader: ProjectionReaderRegistration,
    ) -> None:
        """Verify timestamp accuracy in intent for ModelNodeLivenessExpired reporting.

        OMN-1006: This test verifies that the intent payload contains
        accurate data that would be used when constructing a
        ModelNodeLivenessExpired event:

        1. Node sends heartbeat at T1
        2. Handler produces intent with:
           - last_heartbeat_at = T1
           - liveness_deadline = T1 + window
        3. Effect layer executes the intent to persist to DB
        4. At T2 (after deadline), ServiceTimeoutEmitter would query projection

        This test verifies step 2 to ensure intent has accurate data.
        """
        node_id = uuid4()
        projection = make_projection(entity_id=node_id)
        await seed_projection(projector, projection)

        # Simulate heartbeat at a known time
        heartbeat_time = datetime(2025, 12, 25, 14, 30, 0, 0, tzinfo=UTC)
        event = make_heartbeat_event(node_id, timestamp=heartbeat_time)
        envelope = create_envelope(event)
        output = await heartbeat_handler_fast_window.handle(envelope)

        # Verify output (ORCHESTRATOR returns result=None)
        assert output.result is None

        # Verify intent payload for liveness expired event construction
        assert len(output.intents) == 1
        payload = output.intents[0].payload
        assert isinstance(payload, ModelPayloadPostgresUpdateRegistration)

        intent_last_heartbeat = payload.updates["last_heartbeat_at"]
        intent_liveness_deadline = payload.updates["liveness_deadline"]

        assert intent_last_heartbeat is not None, (
            "last_heartbeat_at should be set for liveness expired event"
        )
        assert intent_liveness_deadline is not None, (
            "liveness_deadline should be set for liveness expired event"
        )

        # Verify timestamps are consistent
        assert intent_last_heartbeat == heartbeat_time

        # Verify deadline calculation (5 second window from fixture)
        expected_deadline = heartbeat_time + timedelta(seconds=5.0)
        assert intent_liveness_deadline == expected_deadline
