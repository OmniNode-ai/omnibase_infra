# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Comprehensive unit tests for ServiceTimeoutEmission.

This test suite validates:
- Service instantiation with dependencies
- Normal emission flow for ack timeouts
- Normal emission flow for liveness expirations
- Marker update after successful emit
- Error handling when publish fails (marker not updated)
- Error handling when marker update fails
- Restart-safe behavior (only processes unmarked entities)
- Correlation and causation ID propagation
- Topic building with environment and namespace
- Result model properties

Test Organization:
    - TestServiceTimeoutEmissionBasics: Instantiation and configuration
    - TestServiceTimeoutEmissionProcessTimeouts: Main processing flow
    - TestServiceTimeoutEmissionAckTimeout: Ack-specific tests
    - TestServiceTimeoutEmissionLivenessExpiration: Liveness-specific tests
    - TestServiceTimeoutEmissionErrorHandling: Error scenarios
    - TestServiceTimeoutEmissionExactlyOnce: Exactly-once semantics
    - TestModelTimeoutEmissionResult: Result model tests

Coverage Goals:
    - >90% code coverage for service
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
from uuid import uuid4

import pytest

from omnibase_infra.enums import EnumRegistrationState
from omnibase_infra.errors import (
    InfraConnectionError,
    InfraTimeoutError,
    InfraUnavailableError,
)
from omnibase_infra.models.projection import ModelRegistrationProjection
from omnibase_infra.models.registration.model_node_capabilities import (
    ModelNodeCapabilities,
)
from omnibase_infra.services import (
    ModelTimeoutEmissionResult,
    ModelTimeoutQueryResult,
    ServiceTimeoutEmission,
)


def create_mock_projection(
    state: EnumRegistrationState = EnumRegistrationState.ACTIVE,
    ack_deadline: datetime | None = None,
    liveness_deadline: datetime | None = None,
    ack_timeout_emitted_at: datetime | None = None,
    liveness_timeout_emitted_at: datetime | None = None,
    entity_id: None = None,
) -> ModelRegistrationProjection:
    """Create a mock projection with sensible defaults."""
    now = datetime.now(UTC)
    return ModelRegistrationProjection(
        entity_id=entity_id or uuid4(),
        domain="registration",
        current_state=state,
        node_type="effect",
        node_version="1.0.0",
        capabilities=ModelNodeCapabilities(),
        ack_deadline=ack_deadline,
        liveness_deadline=liveness_deadline,
        ack_timeout_emitted_at=ack_timeout_emitted_at,
        liveness_timeout_emitted_at=liveness_timeout_emitted_at,
        last_applied_event_id=uuid4(),
        last_applied_offset=100,
        registered_at=now,
        updated_at=now,
    )


@pytest.fixture
def mock_timeout_query() -> AsyncMock:
    """Create a mock timeout query service."""
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
    """Create a mock projector."""
    projector = AsyncMock()
    projector.persist = AsyncMock(return_value=True)
    # Keep these for backwards compatibility with some tests
    projector.update_ack_timeout_marker = AsyncMock(return_value=True)
    projector.update_liveness_timeout_marker = AsyncMock(return_value=True)
    return projector


@pytest.fixture
def service(
    mock_timeout_query: AsyncMock,
    mock_event_bus: AsyncMock,
    mock_projector: AsyncMock,
) -> ServiceTimeoutEmission:
    """Create a ServiceTimeoutEmission instance with mocked dependencies."""
    return ServiceTimeoutEmission(
        timeout_query=mock_timeout_query,
        event_bus=mock_event_bus,
        projector=mock_projector,
        environment="test",
        namespace="omnitest",
    )


@pytest.mark.unit
@pytest.mark.asyncio
class TestServiceTimeoutEmissionBasics:
    """Test basic service instantiation and configuration."""

    async def test_service_instantiation(
        self,
        mock_timeout_query: AsyncMock,
        mock_event_bus: AsyncMock,
        mock_projector: AsyncMock,
    ) -> None:
        """Test that service initializes correctly with dependencies."""
        service = ServiceTimeoutEmission(
            timeout_query=mock_timeout_query,
            event_bus=mock_event_bus,
            projector=mock_projector,
        )

        assert service._timeout_query is mock_timeout_query
        assert service._event_bus is mock_event_bus
        assert service._projector is mock_projector

    async def test_service_default_environment_and_namespace(
        self,
        mock_timeout_query: AsyncMock,
        mock_event_bus: AsyncMock,
        mock_projector: AsyncMock,
    ) -> None:
        """Test default environment and namespace values."""
        service = ServiceTimeoutEmission(
            timeout_query=mock_timeout_query,
            event_bus=mock_event_bus,
            projector=mock_projector,
        )

        assert service.environment == "local"
        assert service.namespace == "onex"

    async def test_service_custom_environment_and_namespace(
        self,
        mock_timeout_query: AsyncMock,
        mock_event_bus: AsyncMock,
        mock_projector: AsyncMock,
    ) -> None:
        """Test custom environment and namespace."""
        service = ServiceTimeoutEmission(
            timeout_query=mock_timeout_query,
            event_bus=mock_event_bus,
            projector=mock_projector,
            environment="prod",
            namespace="myapp",
        )

        assert service.environment == "prod"
        assert service.namespace == "myapp"

    async def test_build_topic(self, service: ServiceTimeoutEmission) -> None:
        """Test topic building with environment and namespace."""
        topic = service._build_topic("{env}.{namespace}.test.topic.v1")

        assert topic == "test.omnitest.test.topic.v1"


@pytest.mark.unit
@pytest.mark.asyncio
class TestServiceTimeoutEmissionProcessTimeouts:
    """Test main process_timeouts flow."""

    async def test_process_timeouts_empty_results(
        self,
        service: ServiceTimeoutEmission,
        mock_timeout_query: AsyncMock,
        mock_event_bus: AsyncMock,
        mock_projector: AsyncMock,
    ) -> None:
        """Test process_timeouts returns empty result when no overdue."""
        now = datetime.now(UTC)
        tick_id = uuid4()
        correlation_id = uuid4()

        result = await service.process_timeouts(
            now=now,
            tick_id=tick_id,
            correlation_id=correlation_id,
        )

        assert isinstance(result, ModelTimeoutEmissionResult)
        assert result.ack_timeouts_emitted == 0
        assert result.liveness_expirations_emitted == 0
        assert result.markers_updated == 0
        assert result.errors == []
        assert result.tick_id == tick_id
        assert result.correlation_id == correlation_id
        assert result.processing_time_ms >= 0.0

        # Verify no publishes occurred
        mock_event_bus.publish_envelope.assert_not_called()
        mock_projector.update_ack_timeout_marker.assert_not_called()
        mock_projector.update_liveness_timeout_marker.assert_not_called()

    async def test_process_timeouts_with_ack_timeouts(
        self,
        service: ServiceTimeoutEmission,
        mock_timeout_query: AsyncMock,
        mock_event_bus: AsyncMock,
        mock_projector: AsyncMock,
    ) -> None:
        """Test process_timeouts emits ack timeout events."""
        now = datetime.now(UTC)
        tick_id = uuid4()
        correlation_id = uuid4()
        past_deadline = now - timedelta(minutes=5)

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

        result = await service.process_timeouts(
            now=now,
            tick_id=tick_id,
            correlation_id=correlation_id,
        )

        assert result.ack_timeouts_emitted == 1
        assert result.liveness_expirations_emitted == 0
        assert result.markers_updated == 1
        assert result.errors == []

        # Verify publish and marker update via persist
        mock_event_bus.publish_envelope.assert_called_once()
        mock_projector.persist.assert_called_once()

    async def test_process_timeouts_with_liveness_expirations(
        self,
        service: ServiceTimeoutEmission,
        mock_timeout_query: AsyncMock,
        mock_event_bus: AsyncMock,
        mock_projector: AsyncMock,
    ) -> None:
        """Test process_timeouts emits liveness expiration events."""
        now = datetime.now(UTC)
        tick_id = uuid4()
        correlation_id = uuid4()
        past_deadline = now - timedelta(minutes=10)

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

        result = await service.process_timeouts(
            now=now,
            tick_id=tick_id,
            correlation_id=correlation_id,
        )

        assert result.ack_timeouts_emitted == 0
        assert result.liveness_expirations_emitted == 1
        assert result.markers_updated == 1
        assert result.errors == []

        # Verify publish and marker update via persist
        mock_event_bus.publish_envelope.assert_called_once()
        mock_projector.persist.assert_called_once()

    async def test_process_timeouts_with_both_types(
        self,
        service: ServiceTimeoutEmission,
        mock_timeout_query: AsyncMock,
        mock_event_bus: AsyncMock,
        mock_projector: AsyncMock,
    ) -> None:
        """Test process_timeouts handles both timeout types."""
        now = datetime.now(UTC)
        tick_id = uuid4()
        correlation_id = uuid4()
        past_deadline = now - timedelta(minutes=5)

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

        result = await service.process_timeouts(
            now=now,
            tick_id=tick_id,
            correlation_id=correlation_id,
        )

        assert result.ack_timeouts_emitted == 1
        assert result.liveness_expirations_emitted == 2
        assert result.markers_updated == 3
        assert result.total_emitted == 3
        assert result.errors == []

        # 3 publishes, 3 persist calls (1 for ack, 2 for liveness)
        assert mock_event_bus.publish_envelope.call_count == 3
        assert mock_projector.persist.call_count == 3


@pytest.mark.unit
@pytest.mark.asyncio
class TestServiceTimeoutEmissionAckTimeout:
    """Test ack-specific timeout emission."""

    async def test_emit_ack_timeout_publishes_correct_event(
        self,
        service: ServiceTimeoutEmission,
        mock_event_bus: AsyncMock,
        mock_projector: AsyncMock,
    ) -> None:
        """Test _emit_ack_timeout publishes correct event."""
        now = datetime.now(UTC)
        tick_id = uuid4()
        correlation_id = uuid4()
        past_deadline = now - timedelta(minutes=5)
        node_id = uuid4()

        projection = create_mock_projection(
            state=EnumRegistrationState.AWAITING_ACK,
            ack_deadline=past_deadline,
            entity_id=node_id,
        )

        await service._emit_ack_timeout(
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

        # Check event content
        event = call_args.kwargs["envelope"]
        assert event.node_id == node_id
        assert event.ack_deadline == past_deadline
        assert event.detected_at == now
        assert event.previous_state == EnumRegistrationState.AWAITING_ACK
        assert event.correlation_id == correlation_id
        assert event.causation_id == tick_id

    async def test_emit_ack_timeout_updates_marker(
        self,
        service: ServiceTimeoutEmission,
        mock_event_bus: AsyncMock,
        mock_projector: AsyncMock,
    ) -> None:
        """Test _emit_ack_timeout updates marker via persist after publish."""
        now = datetime.now(UTC)
        tick_id = uuid4()
        correlation_id = uuid4()
        past_deadline = now - timedelta(minutes=5)
        node_id = uuid4()

        projection = create_mock_projection(
            state=EnumRegistrationState.AWAITING_ACK,
            ack_deadline=past_deadline,
            entity_id=node_id,
        )

        await service._emit_ack_timeout(
            projection=projection,
            detected_at=now,
            tick_id=tick_id,
            correlation_id=correlation_id,
        )

        # Verify persist was called to update the marker
        mock_projector.persist.assert_called_once()
        call_args = mock_projector.persist.call_args

        # Verify the projection has the marker set
        updated_projection = call_args.kwargs["projection"]
        assert updated_projection.ack_timeout_emitted_at == now
        assert updated_projection.entity_id == node_id

    async def test_emit_ack_timeout_raises_on_missing_deadline(
        self,
        service: ServiceTimeoutEmission,
        mock_event_bus: AsyncMock,
    ) -> None:
        """Test _emit_ack_timeout raises when ack_deadline is None."""
        projection = create_mock_projection(
            state=EnumRegistrationState.AWAITING_ACK,
            ack_deadline=None,  # Missing deadline
        )

        with pytest.raises(ValueError, match="ack_deadline is None"):
            await service._emit_ack_timeout(
                projection=projection,
                detected_at=datetime.now(UTC),
                tick_id=uuid4(),
                correlation_id=uuid4(),
            )


@pytest.mark.unit
@pytest.mark.asyncio
class TestServiceTimeoutEmissionLivenessExpiration:
    """Test liveness-specific expiration emission."""

    async def test_emit_liveness_expiration_publishes_correct_event(
        self,
        service: ServiceTimeoutEmission,
        mock_event_bus: AsyncMock,
        mock_projector: AsyncMock,
    ) -> None:
        """Test _emit_liveness_expiration publishes correct event."""
        now = datetime.now(UTC)
        tick_id = uuid4()
        correlation_id = uuid4()
        past_deadline = now - timedelta(minutes=10)
        node_id = uuid4()

        projection = create_mock_projection(
            state=EnumRegistrationState.ACTIVE,
            liveness_deadline=past_deadline,
            entity_id=node_id,
        )

        await service._emit_liveness_expiration(
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

        # Check event content
        event = call_args.kwargs["envelope"]
        assert event.node_id == node_id
        assert event.liveness_deadline == past_deadline
        assert event.detected_at == now
        assert event.correlation_id == correlation_id
        assert event.causation_id == tick_id

    async def test_emit_liveness_expiration_updates_marker(
        self,
        service: ServiceTimeoutEmission,
        mock_event_bus: AsyncMock,
        mock_projector: AsyncMock,
    ) -> None:
        """Test _emit_liveness_expiration updates marker via persist after publish."""
        now = datetime.now(UTC)
        tick_id = uuid4()
        correlation_id = uuid4()
        past_deadline = now - timedelta(minutes=10)
        node_id = uuid4()

        projection = create_mock_projection(
            state=EnumRegistrationState.ACTIVE,
            liveness_deadline=past_deadline,
            entity_id=node_id,
        )

        await service._emit_liveness_expiration(
            projection=projection,
            detected_at=now,
            tick_id=tick_id,
            correlation_id=correlation_id,
        )

        # Verify persist was called to update the marker
        mock_projector.persist.assert_called_once()
        call_args = mock_projector.persist.call_args

        # Verify the projection has the marker set
        updated_projection = call_args.kwargs["projection"]
        assert updated_projection.liveness_timeout_emitted_at == now
        assert updated_projection.entity_id == node_id

    async def test_emit_liveness_expiration_raises_on_missing_deadline(
        self,
        service: ServiceTimeoutEmission,
        mock_event_bus: AsyncMock,
    ) -> None:
        """Test _emit_liveness_expiration raises when liveness_deadline is None."""
        projection = create_mock_projection(
            state=EnumRegistrationState.ACTIVE,
            liveness_deadline=None,  # Missing deadline
        )

        with pytest.raises(ValueError, match="liveness_deadline is None"):
            await service._emit_liveness_expiration(
                projection=projection,
                detected_at=datetime.now(UTC),
                tick_id=uuid4(),
                correlation_id=uuid4(),
            )


@pytest.mark.unit
@pytest.mark.asyncio
class TestServiceTimeoutEmissionErrorHandling:
    """Test error handling for emission operations."""

    async def test_process_timeouts_captures_publish_errors(
        self,
        service: ServiceTimeoutEmission,
        mock_timeout_query: AsyncMock,
        mock_event_bus: AsyncMock,
        mock_projector: AsyncMock,
    ) -> None:
        """Test process_timeouts captures errors but continues processing."""
        now = datetime.now(UTC)
        tick_id = uuid4()
        correlation_id = uuid4()
        past_deadline = now - timedelta(minutes=5)

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

        result = await service.process_timeouts(
            now=now,
            tick_id=tick_id,
            correlation_id=correlation_id,
        )

        # First failed, second succeeded
        assert result.ack_timeouts_emitted == 1
        assert result.errors == [f"ack_timeout failed for node {node1_id}: InfraConnectionError"]
        assert result.has_errors is True

    async def test_process_timeouts_marker_update_failure_not_counted(
        self,
        service: ServiceTimeoutEmission,
        mock_timeout_query: AsyncMock,
        mock_event_bus: AsyncMock,
        mock_projector: AsyncMock,
    ) -> None:
        """Test that persist failure is captured as error."""
        now = datetime.now(UTC)
        tick_id = uuid4()
        correlation_id = uuid4()
        past_deadline = now - timedelta(minutes=5)
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

        # Publish succeeds but persist fails
        mock_event_bus.publish_envelope.return_value = None
        mock_projector.persist.side_effect = InfraConnectionError(
            "Persist failed"
        )

        result = await service.process_timeouts(
            now=now,
            tick_id=tick_id,
            correlation_id=correlation_id,
        )

        # Counted as failure since persist failed
        assert result.ack_timeouts_emitted == 0
        assert result.markers_updated == 0
        assert len(result.errors) == 1
        assert "InfraConnectionError" in result.errors[0]

    async def test_query_error_propagates(
        self,
        service: ServiceTimeoutEmission,
        mock_timeout_query: AsyncMock,
    ) -> None:
        """Test that query errors propagate (not captured)."""
        mock_timeout_query.find_overdue_entities.side_effect = InfraUnavailableError(
            "Circuit breaker open"
        )

        with pytest.raises(InfraUnavailableError):
            await service.process_timeouts(
                now=datetime.now(UTC),
                tick_id=uuid4(),
                correlation_id=uuid4(),
            )


@pytest.mark.unit
@pytest.mark.asyncio
class TestServiceTimeoutEmissionExactlyOnce:
    """Test exactly-once semantics for timeout emission."""

    async def test_marker_update_after_publish_success(
        self,
        service: ServiceTimeoutEmission,
        mock_event_bus: AsyncMock,
        mock_projector: AsyncMock,
    ) -> None:
        """Test marker is only updated AFTER successful publish."""
        now = datetime.now(UTC)
        past_deadline = now - timedelta(minutes=5)
        node_id = uuid4()

        projection = create_mock_projection(
            state=EnumRegistrationState.AWAITING_ACK,
            ack_deadline=past_deadline,
            entity_id=node_id,
        )

        # Track call order
        call_order: list[str] = []
        mock_event_bus.publish_envelope.side_effect = lambda **kwargs: call_order.append("publish")
        mock_projector.persist.side_effect = lambda **kwargs: call_order.append("persist")

        await service._emit_ack_timeout(
            projection=projection,
            detected_at=now,
            tick_id=uuid4(),
            correlation_id=uuid4(),
        )

        # Verify order: publish THEN persist (marker update)
        assert call_order == ["publish", "persist"]

    async def test_marker_not_updated_on_publish_failure(
        self,
        service: ServiceTimeoutEmission,
        mock_event_bus: AsyncMock,
        mock_projector: AsyncMock,
    ) -> None:
        """Test marker is NOT updated when publish fails."""
        now = datetime.now(UTC)
        past_deadline = now - timedelta(minutes=5)

        projection = create_mock_projection(
            state=EnumRegistrationState.AWAITING_ACK,
            ack_deadline=past_deadline,
        )

        mock_event_bus.publish_envelope.side_effect = InfraTimeoutError("Publish timeout")

        with pytest.raises(InfraTimeoutError):
            await service._emit_ack_timeout(
                projection=projection,
                detected_at=now,
                tick_id=uuid4(),
                correlation_id=uuid4(),
            )

        # Marker should NOT have been updated (persist not called)
        mock_projector.persist.assert_not_called()

    async def test_restart_safe_only_unmarked_processed(
        self,
        service: ServiceTimeoutEmission,
        mock_timeout_query: AsyncMock,
        mock_event_bus: AsyncMock,
        mock_projector: AsyncMock,
    ) -> None:
        """Test that only entities without markers are processed.

        This test validates restart-safe behavior by simulating a scenario
        where the query service returns only unmarked entities (as it should
        based on the SQL WHERE clause filtering).
        """
        now = datetime.now(UTC)
        tick_id = uuid4()
        correlation_id = uuid4()
        past_deadline = now - timedelta(minutes=5)

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

        result = await service.process_timeouts(
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
        assert result.errors == []

    def test_result_model_is_frozen(self) -> None:
        """Test result model is immutable."""
        result = ModelTimeoutEmissionResult(
            processing_time_ms=1.0,
            tick_id=uuid4(),
            correlation_id=uuid4(),
        )

        with pytest.raises(Exception):  # Pydantic validation error
            result.ack_timeouts_emitted = 999  # type: ignore[misc]

    def test_result_model_rejects_negative_values(self) -> None:
        """Test result model rejects negative counts."""
        with pytest.raises(Exception):  # Pydantic validation error
            ModelTimeoutEmissionResult(
                ack_timeouts_emitted=-1,
                processing_time_ms=1.0,
                tick_id=uuid4(),
                correlation_id=uuid4(),
            )


@pytest.mark.unit
@pytest.mark.asyncio
class TestServiceTimeoutEmissionTopicBuilding:
    """Test topic building functionality."""

    async def test_ack_timeout_topic_format(
        self,
        mock_timeout_query: AsyncMock,
        mock_event_bus: AsyncMock,
        mock_projector: AsyncMock,
    ) -> None:
        """Test ack timeout topic is correctly formatted."""
        service = ServiceTimeoutEmission(
            timeout_query=mock_timeout_query,
            event_bus=mock_event_bus,
            projector=mock_projector,
            environment="prod",
            namespace="myservice",
        )

        now = datetime.now(UTC)
        past_deadline = now - timedelta(minutes=5)

        projection = create_mock_projection(
            state=EnumRegistrationState.AWAITING_ACK,
            ack_deadline=past_deadline,
        )

        await service._emit_ack_timeout(
            projection=projection,
            detected_at=now,
            tick_id=uuid4(),
            correlation_id=uuid4(),
        )

        call_args = mock_event_bus.publish_envelope.call_args
        assert call_args.kwargs["topic"] == "prod.myservice.onex.evt.node-registration-ack-timed-out.v1"

    async def test_liveness_expired_topic_format(
        self,
        mock_timeout_query: AsyncMock,
        mock_event_bus: AsyncMock,
        mock_projector: AsyncMock,
    ) -> None:
        """Test liveness expired topic is correctly formatted."""
        service = ServiceTimeoutEmission(
            timeout_query=mock_timeout_query,
            event_bus=mock_event_bus,
            projector=mock_projector,
            environment="staging",
            namespace="testapp",
        )

        now = datetime.now(UTC)
        past_deadline = now - timedelta(minutes=10)

        projection = create_mock_projection(
            state=EnumRegistrationState.ACTIVE,
            liveness_deadline=past_deadline,
        )

        await service._emit_liveness_expiration(
            projection=projection,
            detected_at=now,
            tick_id=uuid4(),
            correlation_id=uuid4(),
        )

        call_args = mock_event_bus.publish_envelope.call_args
        assert call_args.kwargs["topic"] == "staging.testapp.onex.evt.node-liveness-expired.v1"
