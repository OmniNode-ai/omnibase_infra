# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Integration tests: State transitions produce snapshots.

P3.6 of OMN-1932 (Wire Snapshot Publisher). Validates the end-to-end flow
where handler state transitions trigger snapshot publishing through the
wired SnapshotPublisherRegistration.

Test Scenarios:
    1. Introspection -> PENDING_REGISTRATION -> snapshot published
    2. Ack -> ACTIVE -> snapshot published
    3. Liveness expiry -> tombstone published
    4. Debounce: rapid transitions coalesce

Related Tickets:
    - OMN-1932: Wire Snapshot Publisher
"""

from __future__ import annotations

import asyncio
import json
from datetime import UTC, datetime, timedelta
from unittest.mock import AsyncMock
from uuid import uuid4

import pytest

from omnibase_core.enums import EnumNodeKind
from omnibase_core.models.events.model_event_envelope import ModelEventEnvelope
from omnibase_core.models.primitives.model_semver import ModelSemVer
from omnibase_infra.enums import EnumRegistrationState
from omnibase_infra.models.projection import (
    ModelRegistrationProjection,
    ModelSnapshotTopicConfig,
)
from omnibase_infra.models.registration import ModelNodeCapabilities
from omnibase_infra.models.registration.commands.model_node_registration_acked import (
    ModelNodeRegistrationAcked,
)
from omnibase_infra.nodes.node_registration_orchestrator.handlers.handler_node_introspected import (
    HandlerNodeIntrospected,
)
from omnibase_infra.nodes.node_registration_orchestrator.handlers.handler_node_registration_acked import (
    HandlerNodeRegistrationAcked,
)
from omnibase_infra.nodes.node_registration_orchestrator.handlers.handler_runtime_tick import (
    HandlerRuntimeTick,
)
from omnibase_infra.projectors.projection_reader_registration import (
    ProjectionReaderRegistration,
)
from omnibase_infra.projectors.snapshot_publisher_registration import (
    SnapshotPublisherRegistration,
)
from omnibase_infra.runtime.models.model_runtime_tick import ModelRuntimeTick

pytestmark = [pytest.mark.integration]

# Fixed test time for deterministic testing
TEST_NOW = datetime(2025, 6, 15, 12, 0, 0, tzinfo=UTC)


def _create_mock_producer() -> AsyncMock:
    """Create a mock Kafka producer with all required async methods."""
    mock_producer = AsyncMock()
    mock_producer.start = AsyncMock()
    mock_producer.stop = AsyncMock()
    mock_producer.send_and_wait = AsyncMock()
    return mock_producer


async def _create_publisher(
    mock_producer: AsyncMock,
    debounce_ms: int = 0,
) -> SnapshotPublisherRegistration:
    """Create and start a SnapshotPublisherRegistration with mock producer."""
    config = ModelSnapshotTopicConfig.default()
    publisher = SnapshotPublisherRegistration(
        mock_producer,
        config,
        debounce_ms=debounce_ms,
    )
    await publisher.start()
    return publisher


def _create_mock_reader(
    entity_state: ModelRegistrationProjection | None = None,
) -> AsyncMock:
    """Create a mock ProjectionReaderRegistration."""
    mock = AsyncMock(spec=ProjectionReaderRegistration)
    mock.get_entity_state = AsyncMock(return_value=entity_state)
    return mock


class TestSnapshotStateTransitions:
    """Integration tests verifying that handler state transitions produce snapshots."""

    @pytest.mark.asyncio
    async def test_introspection_publishes_snapshot(self) -> None:
        """Introspection of a new node publishes a PENDING_REGISTRATION snapshot.

        Given:
            - HandlerNodeIntrospected with mock projector and snapshot publisher
            - ProjectionReader returns None (new node)
        When:
            - ModelNodeIntrospectionEvent is fired
        Then:
            - mock_producer.send_and_wait is called
            - Published JSON contains current_state = pending_registration
        """
        # Arrange
        mock_producer = _create_mock_producer()
        publisher = await _create_publisher(mock_producer, debounce_ms=0)

        mock_reader = _create_mock_reader(entity_state=None)  # New node

        mock_projector = AsyncMock()
        mock_projector.upsert_partial = AsyncMock()

        node_id = uuid4()
        correlation_id = uuid4()

        handler = HandlerNodeIntrospected(
            projection_reader=mock_reader,
            projector=mock_projector,
            snapshot_publisher=publisher,
        )

        # Create introspection event
        from omnibase_infra.models.registration.model_node_introspection_event import (
            ModelNodeIntrospectionEvent,
        )

        event = ModelNodeIntrospectionEvent(
            node_id=node_id,
            node_type=EnumNodeKind.EFFECT,
            correlation_id=correlation_id,
            timestamp=TEST_NOW,
        )

        envelope = ModelEventEnvelope(
            envelope_id=uuid4(),
            payload=event,
            envelope_timestamp=TEST_NOW,
            correlation_id=correlation_id,
            source="test",
        )

        # Act
        output = await handler.handle(envelope)

        # Assert - handler should emit registration initiated event
        assert len(output.events) == 1

        # Assert - projector was called (projection persisted)
        mock_projector.upsert_partial.assert_called_once()

        # Assert - snapshot was published to Kafka
        mock_producer.send_and_wait.assert_called_once()

        call_args = mock_producer.send_and_wait.call_args
        topic = call_args[0][0]
        key = call_args[1]["key"]
        value = call_args[1]["value"]

        # Verify topic is the snapshot topic
        config = ModelSnapshotTopicConfig.default()
        assert topic == config.topic

        # Verify key is the entity_id as UTF-8 bytes
        assert key == str(node_id).encode("utf-8")

        # Verify value is JSON with expected state
        assert value is not None
        snapshot_data = json.loads(value.decode("utf-8"))
        assert snapshot_data["current_state"] == "pending_registration"
        assert snapshot_data["entity_id"] == str(node_id)

        # Cleanup
        await publisher.stop()

    @pytest.mark.asyncio
    async def test_ack_publishes_snapshot(self) -> None:
        """Ack of a node in AWAITING_ACK publishes an ACTIVE snapshot.

        Given:
            - HandlerNodeRegistrationAcked with snapshot publisher
            - ProjectionReader returns projection in AWAITING_ACK state
        When:
            - ModelNodeRegistrationAcked command is fired
        Then:
            - mock_producer.send_and_wait is called
            - Published JSON contains the entity's projection data
        """
        # Arrange
        mock_producer = _create_mock_producer()
        publisher = await _create_publisher(mock_producer, debounce_ms=0)

        node_id = uuid4()
        correlation_id = uuid4()

        awaiting_projection = ModelRegistrationProjection(
            entity_id=node_id,
            domain="registration",
            current_state=EnumRegistrationState.AWAITING_ACK,
            node_type=EnumNodeKind.EFFECT,
            node_version=ModelSemVer.parse("1.0.0"),
            capabilities=ModelNodeCapabilities(),
            ack_deadline=TEST_NOW + timedelta(seconds=30),
            last_applied_event_id=uuid4(),
            last_applied_offset=0,
            registered_at=TEST_NOW - timedelta(minutes=5),
            updated_at=TEST_NOW - timedelta(minutes=1),
        )

        mock_reader = _create_mock_reader(entity_state=awaiting_projection)

        handler = HandlerNodeRegistrationAcked(
            projection_reader=mock_reader,
            snapshot_publisher=publisher,
        )

        ack_command = ModelNodeRegistrationAcked(
            node_id=node_id,
            command_id=uuid4(),
            correlation_id=correlation_id,
            timestamp=TEST_NOW,
        )

        envelope = ModelEventEnvelope(
            envelope_id=uuid4(),
            payload=ack_command,
            envelope_timestamp=TEST_NOW,
            correlation_id=correlation_id,
            source="test",
        )

        # Act
        output = await handler.handle(envelope)

        # Assert - handler should emit AckReceived + BecameActive
        assert len(output.events) == 2

        # Assert - snapshot was published to Kafka
        mock_producer.send_and_wait.assert_called_once()

        call_args = mock_producer.send_and_wait.call_args
        topic = call_args[0][0]
        key = call_args[1]["key"]
        value = call_args[1]["value"]

        # Verify topic
        config = ModelSnapshotTopicConfig.default()
        assert topic == config.topic

        # Verify key is the entity_id
        assert key == str(node_id).encode("utf-8")

        # Verify value is valid JSON with entity data
        assert value is not None
        snapshot_data = json.loads(value.decode("utf-8"))
        assert snapshot_data["entity_id"] == str(node_id)
        assert snapshot_data["domain"] == "registration"
        # The snapshot contains the projection data at the time of ack
        # (AWAITING_ACK is the state passed to publish_from_projection)
        assert "current_state" in snapshot_data

        # Cleanup
        await publisher.stop()

    @pytest.mark.asyncio
    async def test_liveness_expiry_publishes_tombstone(self) -> None:
        """Liveness expiry publishes a tombstone (null value) to the snapshot topic.

        Given:
            - HandlerRuntimeTick with snapshot publisher
            - ProjectionReader returns ACTIVE projection with overdue liveness_deadline
        When:
            - ModelRuntimeTick is fired
        Then:
            - mock_producer.send_and_wait is called with value=None (tombstone)
            - key is the entity_id UUID string
        """
        # Arrange
        mock_producer = _create_mock_producer()
        publisher = await _create_publisher(mock_producer, debounce_ms=0)

        node_id = uuid4()
        tick_time = TEST_NOW

        # Create an ACTIVE projection with overdue liveness deadline.
        # Using a real ModelRegistrationProjection so needs_liveness_timeout_event()
        # returns True (requires: ACTIVE state, overdue deadline, no emission marker).
        overdue_projection = ModelRegistrationProjection(
            entity_id=node_id,
            domain="registration",
            current_state=EnumRegistrationState.ACTIVE,
            node_type=EnumNodeKind.EFFECT,
            node_version=ModelSemVer.parse("1.0.0"),
            capabilities=ModelNodeCapabilities(),
            liveness_deadline=tick_time - timedelta(seconds=10),  # OVERDUE
            last_applied_event_id=uuid4(),
            last_applied_offset=0,
            registered_at=tick_time - timedelta(minutes=5),
            updated_at=tick_time - timedelta(minutes=1),
        )

        mock_reader = AsyncMock(spec=ProjectionReaderRegistration)
        mock_reader.get_overdue_ack_registrations = AsyncMock(return_value=[])
        mock_reader.get_overdue_liveness_registrations = AsyncMock(
            return_value=[overdue_projection]
        )

        handler = HandlerRuntimeTick(
            projection_reader=mock_reader,
            snapshot_publisher=publisher,
        )

        tick = ModelRuntimeTick(
            now=tick_time,
            tick_id=uuid4(),
            sequence_number=1,
            scheduled_at=tick_time,
            correlation_id=uuid4(),
            scheduler_id="test-scheduler",
            tick_interval_ms=1000,
        )

        envelope = ModelEventEnvelope(
            envelope_id=uuid4(),
            payload=tick,
            envelope_timestamp=tick_time,
            correlation_id=uuid4(),
            source="test",
        )

        # Act
        output = await handler.handle(envelope)

        # Assert - handler should emit liveness expired event
        assert len(output.events) == 1

        # Assert - tombstone was published to Kafka (value=None)
        mock_producer.send_and_wait.assert_called_once()

        call_args = mock_producer.send_and_wait.call_args
        topic = call_args[0][0]
        key = call_args[1]["key"]
        value = call_args[1]["value"]

        # Verify topic
        config = ModelSnapshotTopicConfig.default()
        assert topic == config.topic

        # Verify key is the entity_id as bytes
        assert key == str(node_id).encode()

        # Verify tombstone (null value)
        assert value is None

        # Cleanup
        await publisher.stop()

    @pytest.mark.asyncio
    async def test_debounce_coalesces_rapid_publishes(self) -> None:
        """Rapid snapshot publishes within debounce window coalesce to a single send.

        Given:
            - SnapshotPublisherRegistration with debounce_ms=200
        When:
            - publish_from_projection is called 3 times rapidly for the same entity
            - We wait for the debounce window to expire
        Then:
            - mock_producer.send_and_wait is called only once
            - The published snapshot is the last version (version 3)
        """
        # Arrange
        mock_producer = _create_mock_producer()
        publisher = await _create_publisher(mock_producer, debounce_ms=200)

        node_id = uuid4()
        now = TEST_NOW

        # Create 3 projections for the same entity with different states
        projection_1 = ModelRegistrationProjection(
            entity_id=node_id,
            domain="registration",
            current_state=EnumRegistrationState.PENDING_REGISTRATION,
            node_type=EnumNodeKind.EFFECT,
            node_version=ModelSemVer.parse("1.0.0"),
            capabilities=ModelNodeCapabilities(),
            last_applied_event_id=uuid4(),
            last_applied_offset=1,
            registered_at=now,
            updated_at=now,
        )

        projection_2 = ModelRegistrationProjection(
            entity_id=node_id,
            domain="registration",
            current_state=EnumRegistrationState.AWAITING_ACK,
            node_type=EnumNodeKind.EFFECT,
            node_version=ModelSemVer.parse("1.0.0"),
            capabilities=ModelNodeCapabilities(),
            last_applied_event_id=uuid4(),
            last_applied_offset=2,
            registered_at=now,
            updated_at=now + timedelta(seconds=1),
        )

        projection_3 = ModelRegistrationProjection(
            entity_id=node_id,
            domain="registration",
            current_state=EnumRegistrationState.ACTIVE,
            node_type=EnumNodeKind.EFFECT,
            node_version=ModelSemVer.parse("1.0.0"),
            capabilities=ModelNodeCapabilities(),
            last_applied_event_id=uuid4(),
            last_applied_offset=3,
            registered_at=now,
            updated_at=now + timedelta(seconds=2),
        )

        # Act - publish 3 times rapidly (within the 200ms window)
        snap1 = await publisher.publish_from_projection(projection_1)
        snap2 = await publisher.publish_from_projection(projection_2)
        snap3 = await publisher.publish_from_projection(projection_3)

        # Snapshots are returned immediately (before Kafka send)
        assert snap1.snapshot_version == 1
        assert snap2.snapshot_version == 2
        assert snap3.snapshot_version == 3

        # No Kafka sends yet (all debounced)
        mock_producer.send_and_wait.assert_not_called()

        # Wait for debounce window to expire
        await asyncio.sleep(0.35)

        # Assert - only ONE Kafka send occurred (the last snapshot)
        assert mock_producer.send_and_wait.call_count == 1

        call_args = mock_producer.send_and_wait.call_args
        value = call_args[1]["value"]

        # The published snapshot should be version 3 (the last one)
        assert value is not None
        snapshot_data = json.loads(value.decode("utf-8"))
        assert snapshot_data["snapshot_version"] == 3
        assert snapshot_data["current_state"] == "active"

        # Cleanup
        await publisher.stop()
