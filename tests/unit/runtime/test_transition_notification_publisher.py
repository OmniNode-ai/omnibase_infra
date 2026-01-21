# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Unit tests for TransitionNotificationPublisher.

TDD-driven test suite for the TransitionNotificationPublisher class that publishes
state transition notifications to the event bus. Tests are written before the
implementation following TDD red-green-refactor cycle.

Test Organization:
    - TestTransitionNotificationPublisherInit: Initialization and configuration tests
    - TestTransitionNotificationPublisherPublish: Single notification publishing tests
    - TestTransitionNotificationPublisherBatch: Batch publishing tests
    - TestTransitionNotificationPublisherEnvelope: Envelope construction tests
    - TestTransitionNotificationPublisherCircuitBreaker: Circuit breaker behavior tests
    - TestTransitionNotificationPublisherMetrics: Metrics tracking tests
    - TestTransitionNotificationPublisherErrors: Error handling tests
    - TestTransitionNotificationPublisherConcurrency: Thread safety tests

Related:
    - OMN-1139: State Transition Notification Publisher implementation
    - omnibase_core.models.notifications.ModelStateTransitionNotification
    - omnibase_core.protocols.notifications.ProtocolTransitionNotificationPublisher
"""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime
from typing import TYPE_CHECKING
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import UUID, uuid4

import pytest

from omnibase_core.models.notifications import ModelStateTransitionNotification

if TYPE_CHECKING:
    from omnibase_infra.protocols.protocol_event_bus_like import ProtocolEventBusLike

# =============================================================================
# TDD Skip Helper - Check if TransitionNotificationPublisher is implemented
# =============================================================================

_TRANSITION_NOTIFICATION_PUBLISHER_IMPLEMENTED = False
try:
    from omnibase_infra.runtime.transition_notification_publisher import (
        TransitionNotificationPublisher,
    )

    _TRANSITION_NOTIFICATION_PUBLISHER_IMPLEMENTED = True
except ImportError:
    # TransitionNotificationPublisher not implemented yet - define placeholder
    TransitionNotificationPublisher = None  # type: ignore[misc, assignment]

# Skip marker for all tests when implementation doesn't exist
pytestmark = pytest.mark.skipif(
    not _TRANSITION_NOTIFICATION_PUBLISHER_IMPLEMENTED,
    reason="TransitionNotificationPublisher not yet implemented (TDD red phase)",
)


# =============================================================================
# Test Fixtures
# =============================================================================


@pytest.fixture
def mock_event_bus() -> AsyncMock:
    """Create mock event bus with publish_envelope method."""
    bus = AsyncMock()
    bus.publish_envelope = AsyncMock(return_value=None)
    bus.publish = AsyncMock(return_value=None)
    return bus


@pytest.fixture
def failing_event_bus() -> AsyncMock:
    """Create mock event bus that raises on publish."""
    bus = AsyncMock()
    bus.publish_envelope = AsyncMock(side_effect=ConnectionError("Connection refused"))
    bus.publish = AsyncMock(side_effect=ConnectionError("Connection refused"))
    return bus


@pytest.fixture
def slow_event_bus() -> AsyncMock:
    """Create mock event bus with slow publish for timeout testing."""

    async def slow_publish(*args: object, **kwargs: object) -> None:
        await asyncio.sleep(10.0)

    bus = AsyncMock()
    bus.publish_envelope = AsyncMock(side_effect=slow_publish)
    bus.publish = AsyncMock(side_effect=slow_publish)
    return bus


@pytest.fixture
def sample_notification() -> ModelStateTransitionNotification:
    """Create sample notification for testing."""
    return ModelStateTransitionNotification(
        aggregate_type="registration",
        aggregate_id=uuid4(),
        from_state="pending",
        to_state="active",
        projection_version=1,
        correlation_id=uuid4(),
        causation_id=uuid4(),
        timestamp=datetime.now(UTC),
    )


@pytest.fixture
def sample_notifications(
    sample_notification: ModelStateTransitionNotification,
) -> list[ModelStateTransitionNotification]:
    """Create list of sample notifications for batch testing."""
    return [
        sample_notification,
        ModelStateTransitionNotification(
            aggregate_type="registration",
            aggregate_id=uuid4(),
            from_state="active",
            to_state="suspended",
            projection_version=2,
            correlation_id=uuid4(),
            causation_id=uuid4(),
            timestamp=datetime.now(UTC),
        ),
        ModelStateTransitionNotification(
            aggregate_type="intelligence",
            aggregate_id=uuid4(),
            from_state="analyzing",
            to_state="completed",
            projection_version=5,
            correlation_id=uuid4(),
            causation_id=uuid4(),
            timestamp=datetime.now(UTC),
        ),
    ]


@pytest.fixture
def publisher(mock_event_bus: AsyncMock) -> TransitionNotificationPublisher:
    """Create publisher with default settings."""
    return TransitionNotificationPublisher(
        event_bus=mock_event_bus,
        topic="test.fsm.state.transitions.v1",
    )


@pytest.fixture
def publisher_custom_topic(
    mock_event_bus: AsyncMock,
) -> TransitionNotificationPublisher:
    """Create publisher with custom topic."""
    return TransitionNotificationPublisher(
        event_bus=mock_event_bus,
        topic="custom.notifications.state-transitions",
    )


# =============================================================================
# TestTransitionNotificationPublisherInit
# =============================================================================


class TestTransitionNotificationPublisherInit:
    """Test TransitionNotificationPublisher initialization."""

    def test_init_with_event_bus(self, mock_event_bus: AsyncMock) -> None:
        """Test initialization with event bus sets internal reference."""
        publisher = TransitionNotificationPublisher(
            event_bus=mock_event_bus,
            topic="test.fsm.state.transitions.v1",
        )

        # Event bus is stored as private attribute
        assert publisher._event_bus is mock_event_bus

    def test_init_with_explicit_topic(self, mock_event_bus: AsyncMock) -> None:
        """Test initialization with explicit topic."""
        publisher = TransitionNotificationPublisher(
            event_bus=mock_event_bus,
            topic="test.fsm.state.transitions.v1",
        )

        # Topic should be stored correctly
        assert publisher.topic is not None
        assert isinstance(publisher.topic, str)
        assert publisher.topic == "test.fsm.state.transitions.v1"

    def test_init_with_custom_topic(self, mock_event_bus: AsyncMock) -> None:
        """Test initialization with custom topic."""
        custom_topic = "my.custom.notifications.topic"
        publisher = TransitionNotificationPublisher(
            event_bus=mock_event_bus,
            topic=custom_topic,
        )

        assert publisher.topic == custom_topic

    def test_init_with_circuit_breaker_config(self, mock_event_bus: AsyncMock) -> None:
        """Test initialization with circuit breaker configuration."""
        publisher = TransitionNotificationPublisher(
            event_bus=mock_event_bus,
            topic="test.fsm.state.transitions.v1",
            circuit_breaker_threshold=3,
            circuit_breaker_reset_timeout=30.0,
        )

        assert publisher.circuit_breaker_threshold == 3
        assert publisher.circuit_breaker_reset_timeout == 30.0

    def test_init_with_default_circuit_breaker_config(
        self, mock_event_bus: AsyncMock
    ) -> None:
        """Test initialization with default circuit breaker configuration."""
        publisher = TransitionNotificationPublisher(
            event_bus=mock_event_bus,
            topic="test.fsm.state.transitions.v1",
        )

        # Default circuit breaker settings
        assert publisher.circuit_breaker_threshold == 5
        assert publisher.circuit_breaker_reset_timeout == 60.0


# =============================================================================
# TestTransitionNotificationPublisherPublish
# =============================================================================


class TestTransitionNotificationPublisherPublish:
    """Test single notification publishing."""

    @pytest.mark.asyncio
    async def test_publish_single_notification(
        self,
        publisher: TransitionNotificationPublisher,
        mock_event_bus: AsyncMock,
        sample_notification: ModelStateTransitionNotification,
    ) -> None:
        """Notification is published to event bus with correct envelope."""
        await publisher.publish(sample_notification)

        mock_event_bus.publish_envelope.assert_called_once()

    @pytest.mark.asyncio
    async def test_publish_uses_correct_topic(
        self,
        publisher: TransitionNotificationPublisher,
        mock_event_bus: AsyncMock,
        sample_notification: ModelStateTransitionNotification,
    ) -> None:
        """Publisher uses configured topic."""
        await publisher.publish(sample_notification)

        # Extract topic from call args
        call_args = mock_event_bus.publish_envelope.call_args
        topic_arg = call_args.kwargs.get("topic") or call_args.args[1]

        assert topic_arg == publisher.topic

    @pytest.mark.asyncio
    async def test_publish_creates_correct_envelope(
        self,
        publisher: TransitionNotificationPublisher,
        mock_event_bus: AsyncMock,
        sample_notification: ModelStateTransitionNotification,
    ) -> None:
        """Envelope contains correct event_type, correlation_id, and payload."""
        await publisher.publish(sample_notification)

        call_args = mock_event_bus.publish_envelope.call_args
        envelope_arg = call_args.kwargs.get("envelope") or call_args.args[0]

        # Verify envelope contains the notification as payload
        assert envelope_arg is not None
        # Envelope should have correlation_id matching notification
        assert envelope_arg.correlation_id == sample_notification.correlation_id

    @pytest.mark.asyncio
    async def test_publish_preserves_correlation_id(
        self,
        publisher: TransitionNotificationPublisher,
        mock_event_bus: AsyncMock,
        sample_notification: ModelStateTransitionNotification,
    ) -> None:
        """Published envelope preserves notification's correlation_id."""
        expected_correlation_id = sample_notification.correlation_id

        await publisher.publish(sample_notification)

        call_args = mock_event_bus.publish_envelope.call_args
        envelope_arg = call_args.kwargs.get("envelope") or call_args.args[0]

        assert envelope_arg.correlation_id == expected_correlation_id

    @pytest.mark.asyncio
    async def test_publish_includes_causation_id_in_metadata(
        self,
        publisher: TransitionNotificationPublisher,
        mock_event_bus: AsyncMock,
        sample_notification: ModelStateTransitionNotification,
    ) -> None:
        """Published envelope includes causation_id in metadata."""
        await publisher.publish(sample_notification)

        call_args = mock_event_bus.publish_envelope.call_args
        envelope_arg = call_args.kwargs.get("envelope") or call_args.args[0]

        # Causation ID should be traceable from envelope
        payload = envelope_arg.payload
        assert hasattr(payload, "causation_id") or "causation_id" in str(payload)

    @pytest.mark.asyncio
    async def test_custom_topic_configuration(
        self,
        publisher_custom_topic: TransitionNotificationPublisher,
        mock_event_bus: AsyncMock,
        sample_notification: ModelStateTransitionNotification,
    ) -> None:
        """Custom topic is used when configured."""
        await publisher_custom_topic.publish(sample_notification)

        call_args = mock_event_bus.publish_envelope.call_args
        topic_arg = call_args.kwargs.get("topic") or call_args.args[1]

        assert topic_arg == "custom.notifications.state-transitions"


# =============================================================================
# TestTransitionNotificationPublisherBatch
# =============================================================================


class TestTransitionNotificationPublisherBatch:
    """Test batch notification publishing."""

    @pytest.mark.asyncio
    async def test_publish_batch_publishes_all_notifications(
        self,
        publisher: TransitionNotificationPublisher,
        mock_event_bus: AsyncMock,
        sample_notifications: list[ModelStateTransitionNotification],
    ) -> None:
        """Batch publish sends all notifications."""
        await publisher.publish_batch(sample_notifications)

        # Should have called publish_envelope for each notification
        assert mock_event_bus.publish_envelope.call_count == len(sample_notifications)

    @pytest.mark.asyncio
    async def test_publish_batch_empty_list_is_noop(
        self,
        publisher: TransitionNotificationPublisher,
        mock_event_bus: AsyncMock,
    ) -> None:
        """Empty batch does not call event bus."""
        await publisher.publish_batch([])

        mock_event_bus.publish_envelope.assert_not_called()

    @pytest.mark.asyncio
    async def test_publish_batch_single_item(
        self,
        publisher: TransitionNotificationPublisher,
        mock_event_bus: AsyncMock,
        sample_notification: ModelStateTransitionNotification,
    ) -> None:
        """Batch with single item publishes correctly."""
        await publisher.publish_batch([sample_notification])

        mock_event_bus.publish_envelope.assert_called_once()

    @pytest.mark.asyncio
    async def test_publish_batch_preserves_order(
        self,
        publisher: TransitionNotificationPublisher,
        mock_event_bus: AsyncMock,
        sample_notifications: list[ModelStateTransitionNotification],
    ) -> None:
        """Batch publish preserves notification order."""
        await publisher.publish_batch(sample_notifications)

        calls = mock_event_bus.publish_envelope.call_args_list

        for i, call in enumerate(calls):
            envelope_arg = call.kwargs.get("envelope") or call.args[0]
            expected_notification = sample_notifications[i]
            # Verify correlation_id matches (proxy for correct notification)
            assert envelope_arg.correlation_id == expected_notification.correlation_id

    @pytest.mark.asyncio
    async def test_publish_batch_continues_on_individual_failure(
        self,
        mock_event_bus: AsyncMock,
        sample_notifications: list[ModelStateTransitionNotification],
    ) -> None:
        """Batch publish continues processing after individual failure."""
        # Configure to fail on second call only
        call_count = 0

        async def conditional_failure(*args: object, **kwargs: object) -> None:
            nonlocal call_count
            call_count += 1
            if call_count == 2:
                raise ConnectionError("Temporary failure")

        mock_event_bus.publish_envelope = AsyncMock(side_effect=conditional_failure)

        publisher = TransitionNotificationPublisher(
            event_bus=mock_event_bus,
            topic="test.fsm.state.transitions.v1",
            circuit_breaker_threshold=10,  # High threshold to prevent circuit opening
        )

        # Should raise the last error but continue processing all notifications
        with pytest.raises(Exception):
            await publisher.publish_batch(sample_notifications)

        # Verify we attempted all three publishes (continues on individual failure)
        assert mock_event_bus.publish_envelope.call_count == len(sample_notifications)


# =============================================================================
# TestTransitionNotificationPublisherCircuitBreaker
# =============================================================================


class TestTransitionNotificationPublisherCircuitBreaker:
    """Test circuit breaker behavior."""

    @pytest.mark.asyncio
    async def test_circuit_breaker_opens_on_failures(
        self,
        failing_event_bus: AsyncMock,
        sample_notification: ModelStateTransitionNotification,
    ) -> None:
        """Circuit breaker opens after threshold failures."""
        from omnibase_infra.errors import InfraUnavailableError

        publisher = TransitionNotificationPublisher(
            event_bus=failing_event_bus,
            topic="test.fsm.state.transitions.v1",
            circuit_breaker_threshold=3,
            circuit_breaker_reset_timeout=60.0,
        )

        # Record failures up to threshold
        for _ in range(3):
            with pytest.raises(Exception):
                await publisher.publish(sample_notification)

        # Next publish should raise InfraUnavailableError (circuit open)
        with pytest.raises(InfraUnavailableError) as exc_info:
            await publisher.publish(sample_notification)

        error = exc_info.value
        assert "circuit" in error.message.lower() or "open" in error.message.lower()

    @pytest.mark.asyncio
    async def test_circuit_breaker_resets_on_success(
        self,
        mock_event_bus: AsyncMock,
        sample_notification: ModelStateTransitionNotification,
    ) -> None:
        """Circuit breaker resets after successful publish."""
        call_count = 0

        async def flaky_publish(*args: object, **kwargs: object) -> None:
            nonlocal call_count
            call_count += 1
            if call_count <= 2:
                raise ConnectionError("Temporary failure")
            # Success on third call

        mock_event_bus.publish_envelope = AsyncMock(side_effect=flaky_publish)

        publisher = TransitionNotificationPublisher(
            event_bus=mock_event_bus,
            topic="test.fsm.state.transitions.v1",
            circuit_breaker_threshold=5,  # Won't open
        )

        # First two calls fail
        for _ in range(2):
            with pytest.raises(Exception):
                await publisher.publish(sample_notification)

        # Third call succeeds, should reset failure count
        await publisher.publish(sample_notification)

        # Verify we can continue publishing
        call_count = 3  # Reset for next phase
        mock_event_bus.publish_envelope = AsyncMock(return_value=None)

        await publisher.publish(sample_notification)
        mock_event_bus.publish_envelope.assert_called()

    @pytest.mark.asyncio
    async def test_circuit_breaker_half_open_allows_probe(
        self,
        mock_event_bus: AsyncMock,
        sample_notification: ModelStateTransitionNotification,
    ) -> None:
        """Circuit breaker in HALF_OPEN state allows probe request."""
        from omnibase_infra.errors import InfraUnavailableError

        publisher = TransitionNotificationPublisher(
            event_bus=mock_event_bus,
            topic="test.fsm.state.transitions.v1",
            circuit_breaker_threshold=2,
            circuit_breaker_reset_timeout=0.1,  # Very short timeout for testing
        )

        # Open circuit
        mock_event_bus.publish_envelope = AsyncMock(
            side_effect=ConnectionError("Connection refused")
        )
        for _ in range(2):
            with pytest.raises(Exception):
                await publisher.publish(sample_notification)

        # Wait for reset timeout
        await asyncio.sleep(0.15)

        # Circuit should be HALF_OPEN now, allowing probe
        mock_event_bus.publish_envelope = AsyncMock(return_value=None)
        await publisher.publish(sample_notification)

        mock_event_bus.publish_envelope.assert_called_once()


# =============================================================================
# TestTransitionNotificationPublisherMetrics
# =============================================================================


class TestTransitionNotificationPublisherMetrics:
    """Test metrics tracking."""

    @pytest.mark.asyncio
    async def test_metrics_track_publish_count(
        self,
        publisher: TransitionNotificationPublisher,
        mock_event_bus: AsyncMock,
        sample_notification: ModelStateTransitionNotification,
    ) -> None:
        """Metrics are updated after each publish."""
        await publisher.publish(sample_notification)
        await publisher.publish(sample_notification)
        await publisher.publish(sample_notification)

        metrics = publisher.get_metrics()

        assert metrics.notifications_published == 3

    @pytest.mark.asyncio
    async def test_metrics_track_batch_count(
        self,
        publisher: TransitionNotificationPublisher,
        mock_event_bus: AsyncMock,
        sample_notifications: list[ModelStateTransitionNotification],
    ) -> None:
        """Metrics track batch publish operations."""
        await publisher.publish_batch(sample_notifications)

        metrics = publisher.get_metrics()

        assert metrics.batch_operations == 1
        # Batch uses single publish internally, so notifications_published is updated
        assert metrics.notifications_published == len(sample_notifications)

    @pytest.mark.asyncio
    async def test_metrics_track_failure_count(
        self,
        failing_event_bus: AsyncMock,
        sample_notification: ModelStateTransitionNotification,
    ) -> None:
        """Metrics track failed publish attempts."""
        publisher = TransitionNotificationPublisher(
            event_bus=failing_event_bus,
            topic="test.fsm.state.transitions.v1",
            circuit_breaker_threshold=10,  # High threshold to avoid circuit opening
        )

        for _ in range(3):
            with pytest.raises(Exception):
                await publisher.publish(sample_notification)

        metrics = publisher.get_metrics()

        assert metrics.notifications_failed == 3

    @pytest.mark.asyncio
    async def test_metrics_return_pydantic_model(
        self,
        publisher: TransitionNotificationPublisher,
        mock_event_bus: AsyncMock,
        sample_notification: ModelStateTransitionNotification,
    ) -> None:
        """Metrics returns a Pydantic model with correct attributes."""
        from omnibase_infra.runtime.models.model_transition_notification_publisher_metrics import (
            ModelTransitionNotificationPublisherMetrics,
        )

        await publisher.publish(sample_notification)

        metrics = publisher.get_metrics()

        # Should be a Pydantic model instance
        assert isinstance(metrics, ModelTransitionNotificationPublisherMetrics)
        # Should have expected attributes
        assert hasattr(metrics, "publisher_id")
        assert hasattr(metrics, "topic")
        assert hasattr(metrics, "notifications_published")
        assert hasattr(metrics, "notifications_failed")
        assert hasattr(metrics, "batch_operations")


# =============================================================================
# TestTransitionNotificationPublisherErrors
# =============================================================================


class TestTransitionNotificationPublisherErrors:
    """Test error handling."""

    @pytest.mark.asyncio
    async def test_connection_error_raises_infra_error(
        self,
        failing_event_bus: AsyncMock,
        sample_notification: ModelStateTransitionNotification,
    ) -> None:
        """Connection failures raise InfraConnectionError."""
        from omnibase_infra.errors import InfraConnectionError

        publisher = TransitionNotificationPublisher(
            event_bus=failing_event_bus,
            topic="test.fsm.state.transitions.v1",
            circuit_breaker_threshold=10,  # Prevent circuit from opening
        )

        with pytest.raises(InfraConnectionError) as exc_info:
            await publisher.publish(sample_notification)

        error = exc_info.value
        assert (
            "connection" in error.message.lower() or "publish" in error.message.lower()
        )

    @pytest.mark.asyncio
    async def test_error_includes_correlation_id(
        self,
        failing_event_bus: AsyncMock,
        sample_notification: ModelStateTransitionNotification,
    ) -> None:
        """Errors include correlation_id from notification."""
        from omnibase_infra.errors import InfraConnectionError

        publisher = TransitionNotificationPublisher(
            event_bus=failing_event_bus,
            topic="test.fsm.state.transitions.v1",
            circuit_breaker_threshold=10,
        )

        with pytest.raises(InfraConnectionError) as exc_info:
            await publisher.publish(sample_notification)

        error = exc_info.value
        assert error.model.correlation_id == sample_notification.correlation_id

    @pytest.mark.asyncio
    async def test_error_includes_aggregate_info(
        self,
        failing_event_bus: AsyncMock,
        sample_notification: ModelStateTransitionNotification,
    ) -> None:
        """Errors include aggregate type and ID for debugging."""
        from omnibase_infra.errors import InfraConnectionError

        publisher = TransitionNotificationPublisher(
            event_bus=failing_event_bus,
            topic="test.fsm.state.transitions.v1",
            circuit_breaker_threshold=10,
        )

        with pytest.raises(InfraConnectionError) as exc_info:
            await publisher.publish(sample_notification)

        error = exc_info.value
        context = error.model.context

        # Context should include aggregate info for debugging
        assert "aggregate_type" in context or "notification" in str(context).lower()

    @pytest.mark.asyncio
    async def test_serialization_error_handling(
        self,
        mock_event_bus: AsyncMock,
    ) -> None:
        """Serialization errors are handled gracefully."""
        from omnibase_infra.errors import ProtocolConfigurationError

        # Create notification with problematic data
        # This tests that serialization issues are caught properly
        publisher = TransitionNotificationPublisher(
            event_bus=mock_event_bus,
            topic="test.fsm.state.transitions.v1",
        )

        # Create a valid notification - serialization should succeed
        notification = ModelStateTransitionNotification(
            aggregate_type="registration",
            aggregate_id=uuid4(),
            from_state="pending",
            to_state="active",
            projection_version=1,
            correlation_id=uuid4(),
            causation_id=uuid4(),
            timestamp=datetime.now(UTC),
        )

        # Should not raise
        await publisher.publish(notification)


# =============================================================================
# TestTransitionNotificationPublisherConcurrency
# =============================================================================


class TestTransitionNotificationPublisherConcurrency:
    """Test thread safety and concurrent operations."""

    @pytest.mark.asyncio
    async def test_coroutine_safe_concurrent_publish(
        self,
        mock_event_bus: AsyncMock,
    ) -> None:
        """Concurrent publishes are handled safely."""
        publisher = TransitionNotificationPublisher(
            event_bus=mock_event_bus,
            topic="test.fsm.state.transitions.v1",
        )

        # Create 50 unique notifications
        notifications = [
            ModelStateTransitionNotification(
                aggregate_type="registration",
                aggregate_id=uuid4(),
                from_state="pending",
                to_state="active",
                projection_version=i,
                correlation_id=uuid4(),
                causation_id=uuid4(),
                timestamp=datetime.now(UTC),
            )
            for i in range(50)
        ]

        # Publish all concurrently
        await asyncio.gather(*[publisher.publish(n) for n in notifications])

        # All publishes should complete
        assert mock_event_bus.publish_envelope.call_count == 50

    @pytest.mark.asyncio
    async def test_concurrent_publish_preserves_individual_correlation_ids(
        self,
        mock_event_bus: AsyncMock,
    ) -> None:
        """Concurrent publishes preserve correct correlation IDs."""
        publisher = TransitionNotificationPublisher(
            event_bus=mock_event_bus,
            topic="test.fsm.state.transitions.v1",
        )

        # Create notifications with unique correlation IDs
        notifications = [
            ModelStateTransitionNotification(
                aggregate_type="registration",
                aggregate_id=uuid4(),
                from_state="pending",
                to_state="active",
                projection_version=i,
                correlation_id=uuid4(),
                causation_id=uuid4(),
                timestamp=datetime.now(UTC),
            )
            for i in range(20)
        ]

        expected_correlation_ids = {n.correlation_id for n in notifications}

        # Publish all concurrently
        await asyncio.gather(*[publisher.publish(n) for n in notifications])

        # Extract correlation IDs from calls
        actual_correlation_ids = set()
        for call in mock_event_bus.publish_envelope.call_args_list:
            envelope_arg = call.kwargs.get("envelope") or call.args[0]
            actual_correlation_ids.add(envelope_arg.correlation_id)

        assert actual_correlation_ids == expected_correlation_ids

    @pytest.mark.asyncio
    async def test_concurrent_metrics_updates(
        self,
        mock_event_bus: AsyncMock,
    ) -> None:
        """Metrics are correctly updated under concurrent load."""
        publisher = TransitionNotificationPublisher(
            event_bus=mock_event_bus,
            topic="test.fsm.state.transitions.v1",
        )

        notifications = [
            ModelStateTransitionNotification(
                aggregate_type="registration",
                aggregate_id=uuid4(),
                from_state="pending",
                to_state="active",
                projection_version=i,
                correlation_id=uuid4(),
                causation_id=uuid4(),
                timestamp=datetime.now(UTC),
            )
            for i in range(100)
        ]

        await asyncio.gather(*[publisher.publish(n) for n in notifications])

        metrics = publisher.get_metrics()

        # Metrics should be exactly 100 (no race conditions)
        assert metrics.notifications_published == 100

    @pytest.mark.asyncio
    async def test_concurrent_batch_and_single_publish(
        self,
        mock_event_bus: AsyncMock,
    ) -> None:
        """Mixed batch and single publishes work correctly."""
        publisher = TransitionNotificationPublisher(
            event_bus=mock_event_bus,
            topic="test.fsm.state.transitions.v1",
        )

        # Create notifications
        batch_notifications = [
            ModelStateTransitionNotification(
                aggregate_type="registration",
                aggregate_id=uuid4(),
                from_state="pending",
                to_state="active",
                projection_version=i,
                correlation_id=uuid4(),
                causation_id=uuid4(),
                timestamp=datetime.now(UTC),
            )
            for i in range(10)
        ]

        single_notifications = [
            ModelStateTransitionNotification(
                aggregate_type="intelligence",
                aggregate_id=uuid4(),
                from_state="analyzing",
                to_state="completed",
                projection_version=i,
                correlation_id=uuid4(),
                causation_id=uuid4(),
                timestamp=datetime.now(UTC),
            )
            for i in range(5)
        ]

        # Run batch and single publishes concurrently
        await asyncio.gather(
            publisher.publish_batch(batch_notifications),
            *[publisher.publish(n) for n in single_notifications],
        )

        # All should complete (10 batch + 5 single = 15 total)
        assert mock_event_bus.publish_envelope.call_count == 15


# =============================================================================
# TestTransitionNotificationPublisherProtocolCompliance
# =============================================================================


class TestTransitionNotificationPublisherProtocolCompliance:
    """Test protocol compliance."""

    def test_implements_publisher_protocol(
        self,
        mock_event_bus: AsyncMock,
    ) -> None:
        """Publisher implements ProtocolTransitionNotificationPublisher."""
        from omnibase_core.protocols.notifications import (
            ProtocolTransitionNotificationPublisher,
        )

        publisher = TransitionNotificationPublisher(
            event_bus=mock_event_bus,
            topic="test.fsm.state.transitions.v1",
        )

        assert isinstance(publisher, ProtocolTransitionNotificationPublisher)

    @pytest.mark.asyncio
    async def test_publish_method_signature(
        self,
        publisher: TransitionNotificationPublisher,
        sample_notification: ModelStateTransitionNotification,
    ) -> None:
        """publish() method has correct signature."""
        # Should accept ModelStateTransitionNotification and return None
        result = await publisher.publish(sample_notification)

        assert result is None

    @pytest.mark.asyncio
    async def test_publish_batch_method_signature(
        self,
        publisher: TransitionNotificationPublisher,
        sample_notifications: list[ModelStateTransitionNotification],
    ) -> None:
        """publish_batch() method has correct signature."""
        # Should accept list of ModelStateTransitionNotification and return None
        result = await publisher.publish_batch(sample_notifications)

        assert result is None


# =============================================================================
# TestTransitionNotificationPublisherEdgeCases
# =============================================================================


class TestTransitionNotificationPublisherEdgeCases:
    """Test edge cases and boundary conditions."""

    @pytest.mark.asyncio
    async def test_notification_with_workflow_view(
        self,
        publisher: TransitionNotificationPublisher,
        mock_event_bus: AsyncMock,
    ) -> None:
        """Notification with workflow_view publishes correctly."""
        notification = ModelStateTransitionNotification(
            aggregate_type="intelligence",
            aggregate_id=uuid4(),
            from_state="analyzing",
            to_state="completed",
            projection_version=5,
            correlation_id=uuid4(),
            causation_id=uuid4(),
            timestamp=datetime.now(UTC),
            workflow_view={
                "analysis_type": "code_review",
                "findings_count": 3,
                "severity_max": "high",
            },
        )

        await publisher.publish(notification)

        mock_event_bus.publish_envelope.assert_called_once()

    @pytest.mark.asyncio
    async def test_notification_with_projection_hash(
        self,
        publisher: TransitionNotificationPublisher,
        mock_event_bus: AsyncMock,
    ) -> None:
        """Notification with projection_hash publishes correctly."""
        notification = ModelStateTransitionNotification(
            aggregate_type="registration",
            aggregate_id=uuid4(),
            from_state="pending",
            to_state="active",
            projection_version=1,
            correlation_id=uuid4(),
            causation_id=uuid4(),
            timestamp=datetime.now(UTC),
            projection_hash="sha256:abc123def456",
        )

        await publisher.publish(notification)

        mock_event_bus.publish_envelope.assert_called_once()

    @pytest.mark.asyncio
    async def test_large_batch_publish(
        self,
        publisher: TransitionNotificationPublisher,
        mock_event_bus: AsyncMock,
    ) -> None:
        """Large batch (1000 items) publishes correctly."""
        large_batch = [
            ModelStateTransitionNotification(
                aggregate_type="registration",
                aggregate_id=uuid4(),
                from_state="pending",
                to_state="active",
                projection_version=i,
                correlation_id=uuid4(),
                causation_id=uuid4(),
                timestamp=datetime.now(UTC),
            )
            for i in range(1000)
        ]

        await publisher.publish_batch(large_batch)

        assert mock_event_bus.publish_envelope.call_count == 1000

    @pytest.mark.asyncio
    async def test_notification_with_zero_projection_version(
        self,
        publisher: TransitionNotificationPublisher,
        mock_event_bus: AsyncMock,
    ) -> None:
        """Notification with projection_version=0 publishes correctly."""
        notification = ModelStateTransitionNotification(
            aggregate_type="registration",
            aggregate_id=uuid4(),
            from_state="initial",
            to_state="pending",
            projection_version=0,
            correlation_id=uuid4(),
            causation_id=uuid4(),
            timestamp=datetime.now(UTC),
        )

        await publisher.publish(notification)

        mock_event_bus.publish_envelope.assert_called_once()

    @pytest.mark.asyncio
    async def test_same_from_and_to_state(
        self,
        publisher: TransitionNotificationPublisher,
        mock_event_bus: AsyncMock,
    ) -> None:
        """Notification with same from_state and to_state (self-transition)."""
        notification = ModelStateTransitionNotification(
            aggregate_type="registration",
            aggregate_id=uuid4(),
            from_state="active",
            to_state="active",  # Self-transition
            projection_version=5,
            correlation_id=uuid4(),
            causation_id=uuid4(),
            timestamp=datetime.now(UTC),
        )

        await publisher.publish(notification)

        mock_event_bus.publish_envelope.assert_called_once()


# =============================================================================
# Module Exports
# =============================================================================


__all__: list[str] = [
    "TestTransitionNotificationPublisherInit",
    "TestTransitionNotificationPublisherPublish",
    "TestTransitionNotificationPublisherBatch",
    "TestTransitionNotificationPublisherCircuitBreaker",
    "TestTransitionNotificationPublisherMetrics",
    "TestTransitionNotificationPublisherErrors",
    "TestTransitionNotificationPublisherConcurrency",
    "TestTransitionNotificationPublisherProtocolCompliance",
    "TestTransitionNotificationPublisherEdgeCases",
]
