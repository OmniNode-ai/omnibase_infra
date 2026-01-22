# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Comprehensive unit tests for ServiceTimeoutEmitter.

This test suite validates:
- Emitter instantiation with dependencies
- Normal emission flow for ack timeouts
- Normal emission flow for liveness expirations
- Marker update after successful emit
- Error handling when publish fails (marker not updated)
- Error handling when marker update fails
- Restart-safe behavior (only processes unmarked entities)
- Correlation and causation ID propagation
- Topic building with environment and namespace
- Result model properties
- Config model usage

Test Organization:
    - TestServiceTimeoutEmitterBasics: Instantiation and configuration
    - TestServiceTimeoutEmitterProcessTimeouts: Main processing flow
    - TestServiceTimeoutEmitterAckTimeout: Ack-specific tests
    - TestServiceTimeoutEmitterLivenessExpiration: Liveness-specific tests
    - TestServiceTimeoutEmitterErrorHandling: Error scenarios
    - TestServiceTimeoutEmitterExactlyOnce: Exactly-once semantics
    - TestModelTimeoutEmissionResult: Result model tests
    - TestModelTimeoutEmissionConfig: Config model tests

Coverage Goals:
    - >90% code coverage for emitter
    - All emission paths tested
    - Error handling validated
    - Exactly-once semantics verified

Related Tickets:
    - OMN-932 (C2): Durable Timeout Handling
    - OMN-944 (F1): Implement Registration Projection Schema
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from unittest.mock import AsyncMock, MagicMock
from uuid import UUID, uuid4

import pytest
from pydantic import ValidationError

from omnibase_core.container import ModelONEXContainer
from omnibase_core.enums.enum_node_kind import EnumNodeKind
from omnibase_core.models.primitives.model_semver import ModelSemVer
from omnibase_infra.enums import EnumInfraTransportType, EnumRegistrationState
from omnibase_infra.errors import (
    InfraConnectionError,
    InfraTimeoutError,
    InfraUnavailableError,
    ModelTimeoutErrorContext,
    ProtocolConfigurationError,
)
from omnibase_infra.models.projection import ModelRegistrationProjection
from omnibase_infra.models.registration.model_node_capabilities import (
    ModelNodeCapabilities,
)
from omnibase_infra.services import (
    ModelTimeoutEmissionConfig,
    ModelTimeoutEmissionResult,
    ModelTimeoutQueryResult,
    ServiceTimeoutEmitter,
)

# =============================================================================
# Test Constants
# =============================================================================

# Time offsets for deadline scenarios
ACK_TIMEOUT_OFFSET = timedelta(minutes=5)
"""Offset for creating past/overdue ack deadlines in tests."""

LIVENESS_TIMEOUT_OFFSET = timedelta(minutes=10)
"""Offset for creating past/overdue liveness deadlines in tests."""

# Query duration bounds for validation
MAX_REASONABLE_QUERY_DURATION_MS = 10000.0
"""Maximum reasonable query duration in milliseconds (10 seconds)."""

# Default sequence values for test projections
DEFAULT_TEST_OFFSET = 100
"""Default offset/sequence value for test projections."""


def create_mock_projection(
    state: EnumRegistrationState = EnumRegistrationState.ACTIVE,
    ack_deadline: datetime | None = None,
    liveness_deadline: datetime | None = None,
    ack_timeout_emitted_at: datetime | None = None,
    liveness_timeout_emitted_at: datetime | None = None,
    entity_id: UUID | None = None,
    last_heartbeat_at: datetime | None = None,
) -> ModelRegistrationProjection:
    """Create a mock projection with sensible defaults.

    Args:
        state: Registration state (default: ACTIVE)
        ack_deadline: Optional ack deadline for timeout scenarios
        liveness_deadline: Optional liveness deadline for expiry scenarios
        ack_timeout_emitted_at: Marker for ack timeout emission
        liveness_timeout_emitted_at: Marker for liveness expiry emission
        entity_id: Node UUID (generated if not provided)
        last_heartbeat_at: Optional last heartbeat timestamp (OMN-1006)

    Returns:
        ModelRegistrationProjection configured for testing
    """
    now = datetime.now(UTC)
    return ModelRegistrationProjection(
        entity_id=entity_id or uuid4(),
        domain="registration",
        current_state=state,
        node_type=EnumNodeKind.EFFECT,
        node_version=ModelSemVer.parse("1.0.0"),
        capabilities=ModelNodeCapabilities(),
        ack_deadline=ack_deadline,
        liveness_deadline=liveness_deadline,
        ack_timeout_emitted_at=ack_timeout_emitted_at,
        liveness_timeout_emitted_at=liveness_timeout_emitted_at,
        last_heartbeat_at=last_heartbeat_at,
        last_applied_event_id=uuid4(),
        last_applied_offset=100,
        registered_at=now,
        updated_at=now,
    )


@pytest.fixture
def mock_timeout_query() -> AsyncMock:
    """Create a mock timeout query processor."""
    query = AsyncMock()
    query.find_overdue_entities = AsyncMock(
        return_value=ModelTimeoutQueryResult(
            ack_timeouts=[],
            liveness_expirations=[],
            query_time=datetime.now(UTC),
            query_duration_ms=1.0,
        )
    )
    return query


@pytest.fixture
def mock_event_bus() -> AsyncMock:
    """Create a mock event bus."""
    bus = AsyncMock()
    bus.publish_envelope = AsyncMock()
    return bus


@pytest.fixture
def mock_projector() -> AsyncMock:
    """Create a mock ProjectorShell."""
    projector = AsyncMock()
    projector.partial_update = AsyncMock(return_value=True)
    return projector


@pytest.fixture
def mock_container() -> MagicMock:
    """Create a mock ModelONEXContainer."""
    return MagicMock(spec=ModelONEXContainer)


@pytest.fixture
def processor(
    mock_container: MagicMock,
    mock_timeout_query: AsyncMock,
    mock_event_bus: AsyncMock,
    mock_projector: AsyncMock,
) -> ServiceTimeoutEmitter:
    """Create a ServiceTimeoutEmitter instance with mocked dependencies."""
    config = ModelTimeoutEmissionConfig(environment="test", namespace="omnitest")
    return ServiceTimeoutEmitter(
        container=mock_container,
        timeout_query=mock_timeout_query,
        event_bus=mock_event_bus,
        projector=mock_projector,
        config=config,
    )


@pytest.mark.unit
@pytest.mark.asyncio
class TestServiceTimeoutEmitterBasics:
    """Test basic processor instantiation and configuration."""

    async def test_processor_instantiation(
        self,
        mock_container: MagicMock,
        mock_timeout_query: AsyncMock,
        mock_event_bus: AsyncMock,
        mock_projector: AsyncMock,
    ) -> None:
        """Test that processor initializes correctly with dependencies."""
        processor = ServiceTimeoutEmitter(
            container=mock_container,
            timeout_query=mock_timeout_query,
            event_bus=mock_event_bus,
            projector=mock_projector,
        )

        assert processor._container is mock_container
        assert processor._timeout_query is mock_timeout_query
        assert processor._event_bus is mock_event_bus
        assert processor._projector is mock_projector

    async def test_processor_default_environment_and_namespace(
        self,
        mock_container: MagicMock,
        mock_timeout_query: AsyncMock,
        mock_event_bus: AsyncMock,
        mock_projector: AsyncMock,
    ) -> None:
        """Test default environment and namespace values."""
        processor = ServiceTimeoutEmitter(
            container=mock_container,
            timeout_query=mock_timeout_query,
            event_bus=mock_event_bus,
            projector=mock_projector,
        )

        assert processor.environment == "local"
        assert processor.namespace == "onex"

    async def test_processor_custom_environment_and_namespace_via_config(
        self,
        mock_container: MagicMock,
        mock_timeout_query: AsyncMock,
        mock_event_bus: AsyncMock,
        mock_projector: AsyncMock,
    ) -> None:
        """Test custom environment and namespace via config model."""
        config = ModelTimeoutEmissionConfig(environment="prod", namespace="myapp")
        processor = ServiceTimeoutEmitter(
            container=mock_container,
            timeout_query=mock_timeout_query,
            event_bus=mock_event_bus,
            projector=mock_projector,
            config=config,
        )

        assert processor.environment == "prod"
        assert processor.namespace == "myapp"

    async def test_build_topic(self, processor: ServiceTimeoutEmitter) -> None:
        """Test topic building with environment and namespace."""
        topic = processor._build_topic("{env}.{namespace}.test.topic.v1")

        assert topic == "test.omnitest.test.topic.v1"


@pytest.mark.unit
@pytest.mark.asyncio
class TestServiceTimeoutEmitterProcessTimeouts:
    """Test main process_timeouts flow."""

    async def test_process_timeouts_empty_results(
        self,
        processor: ServiceTimeoutEmitter,
        mock_timeout_query: AsyncMock,
        mock_event_bus: AsyncMock,
        mock_projector: AsyncMock,
    ) -> None:
        """Test process_timeouts returns empty result when no overdue."""
        now = datetime.now(UTC)
        tick_id = uuid4()
        correlation_id = uuid4()

        result = await processor.process_timeouts(
            now=now,
            tick_id=tick_id,
            correlation_id=correlation_id,
        )

        assert isinstance(result, ModelTimeoutEmissionResult)
        assert result.ack_timeouts_emitted == 0
        assert result.liveness_expirations_emitted == 0
        assert result.markers_updated == 0
        assert result.errors == ()
        assert result.tick_id == tick_id
        assert result.correlation_id == correlation_id
        assert result.processing_time_ms >= 0.0

        # Verify no publishes occurred
        mock_event_bus.publish_envelope.assert_not_called()
        mock_projector.partial_update.assert_not_called()

    async def test_process_timeouts_with_ack_timeouts(
        self,
        processor: ServiceTimeoutEmitter,
        mock_timeout_query: AsyncMock,
        mock_event_bus: AsyncMock,
        mock_projector: AsyncMock,
    ) -> None:
        """Test process_timeouts emits ack timeout events."""
        now = datetime.now(UTC)
        tick_id = uuid4()
        correlation_id = uuid4()
        past_deadline = now - ACK_TIMEOUT_OFFSET

        ack_projections = [
            create_mock_projection(
                state=EnumRegistrationState.AWAITING_ACK,
                ack_deadline=past_deadline,
            ),
        ]

        mock_timeout_query.find_overdue_entities.return_value = ModelTimeoutQueryResult(
            ack_timeouts=ack_projections,
            liveness_expirations=[],
            query_time=now,
            query_duration_ms=1.0,
        )

        result = await processor.process_timeouts(
            now=now,
            tick_id=tick_id,
            correlation_id=correlation_id,
        )

        assert result.ack_timeouts_emitted == 1
        assert result.liveness_expirations_emitted == 0
        assert result.markers_updated == 1
        assert result.errors == ()

        # Verify publish and marker update via partial_update
        mock_event_bus.publish_envelope.assert_called_once()
        mock_projector.partial_update.assert_called_once()
        # Verify ack_timeout_emitted_at was updated
        call_args = mock_projector.partial_update.call_args
        assert "ack_timeout_emitted_at" in call_args.kwargs["updates"]

    async def test_process_timeouts_with_liveness_expirations(
        self,
        processor: ServiceTimeoutEmitter,
        mock_timeout_query: AsyncMock,
        mock_event_bus: AsyncMock,
        mock_projector: AsyncMock,
    ) -> None:
        """Test process_timeouts emits liveness expiration events."""
        now = datetime.now(UTC)
        tick_id = uuid4()
        correlation_id = uuid4()
        past_deadline = now - LIVENESS_TIMEOUT_OFFSET

        liveness_projections = [
            create_mock_projection(
                state=EnumRegistrationState.ACTIVE,
                liveness_deadline=past_deadline,
            ),
        ]

        mock_timeout_query.find_overdue_entities.return_value = ModelTimeoutQueryResult(
            ack_timeouts=[],
            liveness_expirations=liveness_projections,
            query_time=now,
            query_duration_ms=1.0,
        )

        result = await processor.process_timeouts(
            now=now,
            tick_id=tick_id,
            correlation_id=correlation_id,
        )

        assert result.ack_timeouts_emitted == 0
        assert result.liveness_expirations_emitted == 1
        assert result.markers_updated == 1
        assert result.errors == ()

        # Verify publish and marker update via partial_update
        mock_event_bus.publish_envelope.assert_called_once()
        mock_projector.partial_update.assert_called_once()
        # Verify liveness_timeout_emitted_at was updated
        call_args = mock_projector.partial_update.call_args
        assert "liveness_timeout_emitted_at" in call_args.kwargs["updates"]

    async def test_process_timeouts_with_both_types(
        self,
        processor: ServiceTimeoutEmitter,
        mock_timeout_query: AsyncMock,
        mock_event_bus: AsyncMock,
        mock_projector: AsyncMock,
    ) -> None:
        """Test process_timeouts handles both timeout types."""
        now = datetime.now(UTC)
        tick_id = uuid4()
        correlation_id = uuid4()
        past_deadline = now - ACK_TIMEOUT_OFFSET

        ack_projections = [
            create_mock_projection(
                state=EnumRegistrationState.AWAITING_ACK,
                ack_deadline=past_deadline,
            ),
        ]
        liveness_projections = [
            create_mock_projection(
                state=EnumRegistrationState.ACTIVE,
                liveness_deadline=past_deadline,
            ),
            create_mock_projection(
                state=EnumRegistrationState.ACTIVE,
                liveness_deadline=past_deadline,
            ),
        ]

        mock_timeout_query.find_overdue_entities.return_value = ModelTimeoutQueryResult(
            ack_timeouts=ack_projections,
            liveness_expirations=liveness_projections,
            query_time=now,
            query_duration_ms=1.0,
        )

        result = await processor.process_timeouts(
            now=now,
            tick_id=tick_id,
            correlation_id=correlation_id,
        )

        assert result.ack_timeouts_emitted == 1
        assert result.liveness_expirations_emitted == 2
        assert result.markers_updated == 3
        assert result.total_emitted == 3
        assert result.errors == ()

        # 3 publishes, 3 partial_update calls (1 ack + 2 liveness)
        assert mock_event_bus.publish_envelope.call_count == 3
        assert mock_projector.partial_update.call_count == 3


@pytest.mark.unit
@pytest.mark.asyncio
class TestServiceTimeoutEmitterAckTimeout:
    """Test ack-specific timeout emission."""

    async def test_emit_ack_timeout_publishes_correct_event(
        self,
        processor: ServiceTimeoutEmitter,
        mock_event_bus: AsyncMock,
        mock_projector: AsyncMock,
    ) -> None:
        """Test _emit_ack_timeout publishes correct event."""
        now = datetime.now(UTC)
        tick_id = uuid4()
        correlation_id = uuid4()
        past_deadline = now - ACK_TIMEOUT_OFFSET
        node_id = uuid4()

        projection = create_mock_projection(
            state=EnumRegistrationState.AWAITING_ACK,
            ack_deadline=past_deadline,
            entity_id=node_id,
        )

        await processor._emit_ack_timeout(
            projection=projection,
            detected_at=now,
            tick_id=tick_id,
            correlation_id=correlation_id,
        )

        # Verify publish was called with correct topic
        mock_event_bus.publish_envelope.assert_called_once()
        call_args = mock_event_bus.publish_envelope.call_args

        # Check topic
        assert "node-registration-ack-timed-out" in call_args.kwargs["topic"]

        # Check event content - using canonical field names from consolidated model
        # Events are wrapped in ModelEventEnvelope, access payload for event fields
        envelope = call_args.kwargs["envelope"]
        event = envelope.payload
        assert event.node_id == node_id
        assert event.deadline_at == past_deadline  # Renamed from ack_deadline
        assert event.emitted_at == now  # Renamed from detected_at
        assert event.previous_state == EnumRegistrationState.AWAITING_ACK
        assert event.correlation_id == correlation_id
        assert event.causation_id == tick_id

    async def test_emit_ack_timeout_updates_marker(
        self,
        processor: ServiceTimeoutEmitter,
        mock_event_bus: AsyncMock,
        mock_projector: AsyncMock,
    ) -> None:
        """Test _emit_ack_timeout updates marker via partial_update after publish."""
        now = datetime.now(UTC)
        tick_id = uuid4()
        correlation_id = uuid4()
        past_deadline = now - ACK_TIMEOUT_OFFSET
        node_id = uuid4()

        projection = create_mock_projection(
            state=EnumRegistrationState.AWAITING_ACK,
            ack_deadline=past_deadline,
            entity_id=node_id,
        )

        await processor._emit_ack_timeout(
            projection=projection,
            detected_at=now,
            tick_id=tick_id,
            correlation_id=correlation_id,
        )

        # Verify partial_update was called for marker update
        mock_projector.partial_update.assert_called_once()
        call_args = mock_projector.partial_update.call_args

        # Verify the correct parameters were passed
        assert call_args.kwargs["aggregate_id"] == node_id
        assert call_args.kwargs["correlation_id"] == correlation_id
        assert "ack_timeout_emitted_at" in call_args.kwargs["updates"]
        assert call_args.kwargs["updates"]["ack_timeout_emitted_at"] == now

    async def test_emit_ack_timeout_raises_on_missing_deadline(
        self,
        processor: ServiceTimeoutEmitter,
        mock_event_bus: AsyncMock,
    ) -> None:
        """Test _emit_ack_timeout raises when ack_deadline is None."""
        projection = create_mock_projection(
            state=EnumRegistrationState.AWAITING_ACK,
            ack_deadline=None,  # Missing deadline
        )

        with pytest.raises(ProtocolConfigurationError, match="ack_deadline is None"):
            await processor._emit_ack_timeout(
                projection=projection,
                detected_at=datetime.now(UTC),
                tick_id=uuid4(),
                correlation_id=uuid4(),
            )


@pytest.mark.unit
@pytest.mark.asyncio
class TestServiceTimeoutEmitterLivenessExpiration:
    """Test liveness-specific expiration emission."""

    async def test_emit_liveness_expiration_publishes_correct_event(
        self,
        processor: ServiceTimeoutEmitter,
        mock_event_bus: AsyncMock,
        mock_projector: AsyncMock,
    ) -> None:
        """Test _emit_liveness_expiration publishes correct event."""
        now = datetime.now(UTC)
        tick_id = uuid4()
        correlation_id = uuid4()
        past_deadline = now - LIVENESS_TIMEOUT_OFFSET
        node_id = uuid4()

        projection = create_mock_projection(
            state=EnumRegistrationState.ACTIVE,
            liveness_deadline=past_deadline,
            entity_id=node_id,
        )

        await processor._emit_liveness_expiration(
            projection=projection,
            detected_at=now,
            tick_id=tick_id,
            correlation_id=correlation_id,
        )

        # Verify publish was called with correct topic
        mock_event_bus.publish_envelope.assert_called_once()
        call_args = mock_event_bus.publish_envelope.call_args

        # Check topic
        assert "node-liveness-expired" in call_args.kwargs["topic"]

        # Check event content - events are wrapped in ModelEventEnvelope
        envelope = call_args.kwargs["envelope"]
        event = envelope.payload
        assert event.node_id == node_id
        assert event.liveness_deadline == past_deadline
        assert event.detected_at == now
        assert event.correlation_id == correlation_id
        assert event.causation_id == tick_id

    async def test_emit_liveness_expiration_updates_marker(
        self,
        processor: ServiceTimeoutEmitter,
        mock_event_bus: AsyncMock,
        mock_projector: AsyncMock,
    ) -> None:
        """Test _emit_liveness_expiration updates marker via partial_update after publish."""
        now = datetime.now(UTC)
        tick_id = uuid4()
        correlation_id = uuid4()
        past_deadline = now - LIVENESS_TIMEOUT_OFFSET
        node_id = uuid4()

        projection = create_mock_projection(
            state=EnumRegistrationState.ACTIVE,
            liveness_deadline=past_deadline,
            entity_id=node_id,
        )

        await processor._emit_liveness_expiration(
            projection=projection,
            detected_at=now,
            tick_id=tick_id,
            correlation_id=correlation_id,
        )

        # Verify partial_update was called for marker update
        mock_projector.partial_update.assert_called_once()
        call_args = mock_projector.partial_update.call_args

        # Verify the correct parameters were passed
        assert call_args.kwargs["aggregate_id"] == node_id
        assert call_args.kwargs["correlation_id"] == correlation_id
        assert "liveness_timeout_emitted_at" in call_args.kwargs["updates"]
        assert call_args.kwargs["updates"]["liveness_timeout_emitted_at"] == now

    async def test_emit_liveness_expiration_raises_on_missing_deadline(
        self,
        processor: ServiceTimeoutEmitter,
        mock_event_bus: AsyncMock,
    ) -> None:
        """Test _emit_liveness_expiration raises when liveness_deadline is None."""
        projection = create_mock_projection(
            state=EnumRegistrationState.ACTIVE,
            liveness_deadline=None,  # Missing deadline
        )

        with pytest.raises(
            ProtocolConfigurationError, match="liveness_deadline is None"
        ):
            await processor._emit_liveness_expiration(
                projection=projection,
                detected_at=datetime.now(UTC),
                tick_id=uuid4(),
                correlation_id=uuid4(),
            )

    async def test_emit_liveness_expiration_includes_last_heartbeat_at(
        self,
        processor: ServiceTimeoutEmitter,
        mock_event_bus: AsyncMock,
        mock_projector: AsyncMock,
    ) -> None:
        """Test _emit_liveness_expiration includes accurate last_heartbeat_at.

        OMN-1006: Verify that ModelNodeLivenessExpired events include the
        last_heartbeat_at timestamp from the projection. This is critical
        for debugging liveness issues and understanding when the node
        was last known to be alive.

        Related Tickets:
            - OMN-1006: Add last_heartbeat_at for liveness expired event reporting
        """
        now = datetime.now(UTC)
        tick_id = uuid4()
        correlation_id = uuid4()
        past_deadline = now - LIVENESS_TIMEOUT_OFFSET
        node_id = uuid4()

        # Create projection with explicit last_heartbeat_at
        last_hb = now - timedelta(minutes=15)  # 15 minutes ago
        projection = create_mock_projection(
            state=EnumRegistrationState.ACTIVE,
            liveness_deadline=past_deadline,
            entity_id=node_id,
            last_heartbeat_at=last_hb,
        )

        await processor._emit_liveness_expiration(
            projection=projection,
            detected_at=now,
            tick_id=tick_id,
            correlation_id=correlation_id,
        )

        # Verify publish was called
        mock_event_bus.publish_envelope.assert_called_once()
        call_args = mock_event_bus.publish_envelope.call_args

        # CRITICAL: Verify last_heartbeat_at is correctly passed to event
        # Events are wrapped in ModelEventEnvelope, access payload for event fields
        envelope = call_args.kwargs["envelope"]
        event = envelope.payload
        assert event.last_heartbeat_at == last_hb, (
            f"Expected last_heartbeat_at={last_hb}, got {event.last_heartbeat_at}"
        )
        assert event.liveness_deadline == past_deadline
        assert event.detected_at == now

    async def test_emit_liveness_expiration_handles_none_last_heartbeat_at(
        self,
        processor: ServiceTimeoutEmitter,
        mock_event_bus: AsyncMock,
        mock_projector: AsyncMock,
    ) -> None:
        """Test _emit_liveness_expiration handles None last_heartbeat_at.

        OMN-1006: When a node has never sent a heartbeat (e.g., during initial
        registration), last_heartbeat_at will be None. The emitter should
        handle this gracefully.
        """
        now = datetime.now(UTC)
        tick_id = uuid4()
        correlation_id = uuid4()
        past_deadline = now - LIVENESS_TIMEOUT_OFFSET
        node_id = uuid4()

        # Create projection WITHOUT last_heartbeat_at (node never sent heartbeat)
        projection = create_mock_projection(
            state=EnumRegistrationState.ACTIVE,
            liveness_deadline=past_deadline,
            entity_id=node_id,
            last_heartbeat_at=None,  # No heartbeat received
        )

        await processor._emit_liveness_expiration(
            projection=projection,
            detected_at=now,
            tick_id=tick_id,
            correlation_id=correlation_id,
        )

        # Verify publish was called
        mock_event_bus.publish_envelope.assert_called_once()
        call_args = mock_event_bus.publish_envelope.call_args

        # Verify None is correctly passed (event model allows None)
        # Events are wrapped in ModelEventEnvelope, access payload for event fields
        envelope = call_args.kwargs["envelope"]
        event = envelope.payload
        assert event.last_heartbeat_at is None
        assert event.liveness_deadline == past_deadline
        assert event.detected_at == now


@pytest.mark.unit
@pytest.mark.asyncio
class TestServiceTimeoutEmitterErrorHandling:
    """Test error handling for emission operations."""

    async def test_process_timeouts_captures_publish_errors(
        self,
        processor: ServiceTimeoutEmitter,
        mock_timeout_query: AsyncMock,
        mock_event_bus: AsyncMock,
        mock_projector: AsyncMock,
    ) -> None:
        """Test process_timeouts captures errors but continues processing."""
        now = datetime.now(UTC)
        tick_id = uuid4()
        correlation_id = uuid4()
        past_deadline = now - ACK_TIMEOUT_OFFSET

        # Two projections - first will fail, second should succeed
        node1_id = uuid4()
        node2_id = uuid4()
        ack_projections = [
            create_mock_projection(
                state=EnumRegistrationState.AWAITING_ACK,
                ack_deadline=past_deadline,
                entity_id=node1_id,
            ),
            create_mock_projection(
                state=EnumRegistrationState.AWAITING_ACK,
                ack_deadline=past_deadline,
                entity_id=node2_id,
            ),
        ]

        mock_timeout_query.find_overdue_entities.return_value = ModelTimeoutQueryResult(
            ack_timeouts=ack_projections,
            liveness_expirations=[],
            query_time=now,
            query_duration_ms=1.0,
        )

        # First publish fails, second succeeds
        mock_event_bus.publish_envelope.side_effect = [
            InfraConnectionError("Connection failed"),
            None,  # Success
        ]

        result = await processor.process_timeouts(
            now=now,
            tick_id=tick_id,
            correlation_id=correlation_id,
        )

        # First failed, second succeeded
        assert result.ack_timeouts_emitted == 1
        assert result.errors == (
            f"ack_timeout failed for node {node1_id}: InfraConnectionError",
        )
        assert result.has_errors is True

    async def test_process_timeouts_marker_update_failure_not_counted(
        self,
        processor: ServiceTimeoutEmitter,
        mock_timeout_query: AsyncMock,
        mock_event_bus: AsyncMock,
        mock_projector: AsyncMock,
    ) -> None:
        """Test that marker update failure treats the whole operation as failed.

        This test validates the atomic counting semantics:
        - When marker update fails, counters are NOT incremented
        - Even though the event WAS published to Kafka
        - This is intentional for exactly-once semantics from the system's perspective
        - The entity will be re-processed on next tick (marker still NULL)
        - Downstream consumers should deduplicate by event_id if needed

        See ModelTimeoutEmissionResult docstring for full counter semantics.
        """
        now = datetime.now(UTC)
        tick_id = uuid4()
        correlation_id = uuid4()
        past_deadline = now - ACK_TIMEOUT_OFFSET
        node_id = uuid4()

        ack_projections = [
            create_mock_projection(
                state=EnumRegistrationState.AWAITING_ACK,
                ack_deadline=past_deadline,
                entity_id=node_id,
            ),
        ]

        mock_timeout_query.find_overdue_entities.return_value = ModelTimeoutQueryResult(
            ack_timeouts=ack_projections,
            liveness_expirations=[],
            query_time=now,
            query_duration_ms=1.0,
        )

        # Publish succeeds but partial_update (marker update) fails
        mock_event_bus.publish_envelope.return_value = None
        mock_projector.partial_update.side_effect = InfraConnectionError(
            "Marker update failed"
        )

        result = await processor.process_timeouts(
            now=now,
            tick_id=tick_id,
            correlation_id=correlation_id,
        )

        # Counters NOT incremented even though event was published to Kafka.
        # This is intentional - operation is atomic (publish + marker update).
        # Entity will be re-processed on next tick since marker is still NULL.
        assert result.ack_timeouts_emitted == 0
        assert result.markers_updated == 0
        assert len(result.errors) == 1
        assert "InfraConnectionError" in result.errors[0]

    async def test_query_error_propagates(
        self,
        processor: ServiceTimeoutEmitter,
        mock_timeout_query: AsyncMock,
    ) -> None:
        """Test that query errors propagate (not captured)."""
        mock_timeout_query.find_overdue_entities.side_effect = InfraUnavailableError(
            "Circuit breaker open"
        )

        with pytest.raises(InfraUnavailableError):
            await processor.process_timeouts(
                now=datetime.now(UTC),
                tick_id=uuid4(),
                correlation_id=uuid4(),
            )


@pytest.mark.unit
@pytest.mark.asyncio
class TestServiceTimeoutEmitterExactlyOnce:
    """Test exactly-once semantics for timeout emission."""

    async def test_marker_update_after_publish_success(
        self,
        processor: ServiceTimeoutEmitter,
        mock_event_bus: AsyncMock,
        mock_projector: AsyncMock,
    ) -> None:
        """Test marker is only updated AFTER successful publish."""
        now = datetime.now(UTC)
        past_deadline = now - ACK_TIMEOUT_OFFSET
        node_id = uuid4()

        projection = create_mock_projection(
            state=EnumRegistrationState.AWAITING_ACK,
            ack_deadline=past_deadline,
            entity_id=node_id,
        )

        # Track call order
        call_order: list[str] = []
        mock_event_bus.publish_envelope.side_effect = (
            lambda **kwargs: call_order.append("publish")
        )
        mock_projector.partial_update.side_effect = lambda **kwargs: call_order.append(
            "marker_update"
        )

        await processor._emit_ack_timeout(
            projection=projection,
            detected_at=now,
            tick_id=uuid4(),
            correlation_id=uuid4(),
        )

        # Verify order: publish THEN partial_update marker update
        assert call_order == ["publish", "marker_update"]

    async def test_marker_not_updated_on_publish_failure(
        self,
        processor: ServiceTimeoutEmitter,
        mock_event_bus: AsyncMock,
        mock_projector: AsyncMock,
    ) -> None:
        """Test marker is NOT updated when publish fails."""
        now = datetime.now(UTC)
        past_deadline = now - ACK_TIMEOUT_OFFSET

        projection = create_mock_projection(
            state=EnumRegistrationState.AWAITING_ACK,
            ack_deadline=past_deadline,
        )

        mock_event_bus.publish_envelope.side_effect = InfraTimeoutError(
            "Publish timeout",
            context=ModelTimeoutErrorContext(
                transport_type=EnumInfraTransportType.KAFKA,
                operation="publish_envelope",
            ),
        )

        with pytest.raises(InfraTimeoutError):
            await processor._emit_ack_timeout(
                projection=projection,
                detected_at=now,
                tick_id=uuid4(),
                correlation_id=uuid4(),
            )

        # Marker should NOT have been updated (partial_update not called)
        mock_projector.partial_update.assert_not_called()

    async def test_restart_safe_only_unmarked_processed(
        self,
        processor: ServiceTimeoutEmitter,
        mock_timeout_query: AsyncMock,
        mock_event_bus: AsyncMock,
        mock_projector: AsyncMock,
    ) -> None:
        """Test that only entities without markers are processed.

        This test validates restart-safe behavior by simulating a scenario
        where the query processor returns only unmarked entities (as it should
        based on the SQL WHERE clause filtering).
        """
        now = datetime.now(UTC)
        tick_id = uuid4()
        correlation_id = uuid4()
        past_deadline = now - ACK_TIMEOUT_OFFSET

        # Only one entity returned (already marked ones filtered by query)
        ack_projections = [
            create_mock_projection(
                state=EnumRegistrationState.AWAITING_ACK,
                ack_deadline=past_deadline,
                ack_timeout_emitted_at=None,  # Not yet emitted
            ),
        ]

        mock_timeout_query.find_overdue_entities.return_value = ModelTimeoutQueryResult(
            ack_timeouts=ack_projections,
            liveness_expirations=[],
            query_time=now,
            query_duration_ms=1.0,
        )

        result = await processor.process_timeouts(
            now=now,
            tick_id=tick_id,
            correlation_id=correlation_id,
        )

        # Only the unmarked entity should be processed
        assert result.ack_timeouts_emitted == 1
        assert mock_event_bus.publish_envelope.call_count == 1


@pytest.mark.unit
class TestModelTimeoutEmissionResult:
    """Test ModelTimeoutEmissionResult model."""

    def test_result_model_creation(self) -> None:
        """Test result model can be created with required fields."""
        tick_id = uuid4()
        correlation_id = uuid4()
        result = ModelTimeoutEmissionResult(
            ack_timeouts_emitted=1,
            liveness_expirations_emitted=2,
            markers_updated=3,
            errors=[],
            processing_time_ms=10.5,
            tick_id=tick_id,
            correlation_id=correlation_id,
        )

        assert result.ack_timeouts_emitted == 1
        assert result.liveness_expirations_emitted == 2
        assert result.markers_updated == 3
        assert result.processing_time_ms == 10.5
        assert result.tick_id == tick_id
        assert result.correlation_id == correlation_id

    def test_total_emitted_property(self) -> None:
        """Test total_emitted property calculation."""
        result = ModelTimeoutEmissionResult(
            ack_timeouts_emitted=3,
            liveness_expirations_emitted=5,
            markers_updated=8,
            errors=[],
            processing_time_ms=1.0,
            tick_id=uuid4(),
            correlation_id=uuid4(),
        )

        assert result.total_emitted == 8

    def test_has_errors_true(self) -> None:
        """Test has_errors returns True when errors exist."""
        result = ModelTimeoutEmissionResult(
            ack_timeouts_emitted=0,
            liveness_expirations_emitted=0,
            markers_updated=0,
            errors=["Error 1", "Error 2"],
            processing_time_ms=1.0,
            tick_id=uuid4(),
            correlation_id=uuid4(),
        )

        assert result.has_errors is True

    def test_has_errors_false(self) -> None:
        """Test has_errors returns False when no errors."""
        result = ModelTimeoutEmissionResult(
            ack_timeouts_emitted=1,
            liveness_expirations_emitted=1,
            markers_updated=2,
            errors=[],
            processing_time_ms=1.0,
            tick_id=uuid4(),
            correlation_id=uuid4(),
        )

        assert result.has_errors is False

    def test_result_model_defaults(self) -> None:
        """Test result model defaults."""
        result = ModelTimeoutEmissionResult(
            processing_time_ms=1.0,
            tick_id=uuid4(),
            correlation_id=uuid4(),
        )

        assert result.ack_timeouts_emitted == 0
        assert result.liveness_expirations_emitted == 0
        assert result.markers_updated == 0
        assert result.errors == ()

    def test_result_model_is_frozen(self) -> None:
        """Test result model is immutable."""
        result = ModelTimeoutEmissionResult(
            processing_time_ms=1.0,
            tick_id=uuid4(),
            correlation_id=uuid4(),
        )

        with pytest.raises(ValidationError):
            result.ack_timeouts_emitted = 999  # type: ignore[misc]

    def test_result_model_rejects_negative_values(self) -> None:
        """Test result model rejects negative counts."""
        with pytest.raises(ValidationError):
            ModelTimeoutEmissionResult(
                ack_timeouts_emitted=-1,
                processing_time_ms=1.0,
                tick_id=uuid4(),
                correlation_id=uuid4(),
            )


@pytest.mark.unit
class TestModelTimeoutEmissionConfig:
    """Test ModelTimeoutEmissionConfig model."""

    def test_config_defaults(self) -> None:
        """Test config model defaults."""
        config = ModelTimeoutEmissionConfig()

        assert config.environment == "local"
        assert config.namespace == "onex"

    def test_config_custom_values(self) -> None:
        """Test config model with custom values."""
        config = ModelTimeoutEmissionConfig(
            environment="prod",
            namespace="myapp",
        )

        assert config.environment == "prod"
        assert config.namespace == "myapp"

    def test_config_is_frozen(self) -> None:
        """Test config model is immutable."""
        config = ModelTimeoutEmissionConfig()

        with pytest.raises(ValidationError):
            config.environment = "changed"  # type: ignore[misc]


@pytest.mark.unit
@pytest.mark.asyncio
class TestServiceTimeoutEmitterTopicBuilding:
    """Test topic building functionality."""

    async def test_ack_timeout_topic_format(
        self,
        mock_container: MagicMock,
        mock_timeout_query: AsyncMock,
        mock_event_bus: AsyncMock,
        mock_projector: AsyncMock,
    ) -> None:
        """Test ack timeout topic is correctly formatted."""
        config = ModelTimeoutEmissionConfig(environment="prod", namespace="myservice")
        processor = ServiceTimeoutEmitter(
            container=mock_container,
            timeout_query=mock_timeout_query,
            event_bus=mock_event_bus,
            projector=mock_projector,
            config=config,
        )

        now = datetime.now(UTC)
        past_deadline = now - ACK_TIMEOUT_OFFSET

        projection = create_mock_projection(
            state=EnumRegistrationState.AWAITING_ACK,
            ack_deadline=past_deadline,
        )

        await processor._emit_ack_timeout(
            projection=projection,
            detected_at=now,
            tick_id=uuid4(),
            correlation_id=uuid4(),
        )

        call_args = mock_event_bus.publish_envelope.call_args
        assert (
            call_args.kwargs["topic"]
            == "prod.myservice.onex.evt.node-registration-ack-timed-out.v1"
        )

    async def test_liveness_expired_topic_format(
        self,
        mock_container: MagicMock,
        mock_timeout_query: AsyncMock,
        mock_event_bus: AsyncMock,
        mock_projector: AsyncMock,
    ) -> None:
        """Test liveness expired topic is correctly formatted."""
        config = ModelTimeoutEmissionConfig(environment="staging", namespace="testapp")
        processor = ServiceTimeoutEmitter(
            container=mock_container,
            timeout_query=mock_timeout_query,
            event_bus=mock_event_bus,
            projector=mock_projector,
            config=config,
        )

        now = datetime.now(UTC)
        past_deadline = now - LIVENESS_TIMEOUT_OFFSET

        projection = create_mock_projection(
            state=EnumRegistrationState.ACTIVE,
            liveness_deadline=past_deadline,
        )

        await processor._emit_liveness_expiration(
            projection=projection,
            detected_at=now,
            tick_id=uuid4(),
            correlation_id=uuid4(),
        )

        call_args = mock_event_bus.publish_envelope.call_args
        assert (
            call_args.kwargs["topic"]
            == "staging.testapp.onex.evt.node-liveness-expired.v1"
        )
