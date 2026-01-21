# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""
Transition Notification Publisher Implementation.

Publishes state transition notifications after projection commits. This enables
orchestrators to reliably detect state transitions via the Observer pattern,
maintaining loose coupling between reducers and workflow coordinators.

Architecture Overview:
    This service implements post-commit notification publishing in the ONEX
    state machine architecture:

    1. Reducers commit state transitions to projections
    2. Post-commit hook creates ModelStateTransitionNotification
    3. TransitionNotificationPublisher publishes to event bus
    4. Orchestrators subscribe and coordinate downstream workflows

    ```
    Reducer -> Projection Commit -> Notification Publisher -> Event Bus
                                            |
                                            v
                                    Orchestrators (subscribers)
    ```

Design Principles:
    - **Loose Coupling**: Reducers don't know about orchestrators
    - **At-Least-Once Delivery**: Consumers handle idempotency via projection_version
    - **Circuit Breaker**: Resilience against event bus failures
    - **Correlation Tracking**: Full distributed tracing support

Concurrency Safety:
    This implementation is coroutine-safe for concurrent async publishing.
    Uses asyncio locks for circuit breaker state management. Note: This is
    coroutine-safe, not thread-safe. For multi-threaded access, additional
    synchronization would be required.

Error Handling:
    All methods raise ONEX error types:
    - InfraConnectionError: Event bus unavailable or connection failed
    - InfraTimeoutError: Publish operation timed out
    - InfraUnavailableError: Circuit breaker open

Example Usage:
    ```python
    from omnibase_infra.runtime import TransitionNotificationPublisher
    from omnibase_core.models.notifications import ModelStateTransitionNotification

    # Initialize publisher with event bus
    publisher = TransitionNotificationPublisher(
        event_bus=kafka_event_bus,
        topic="onex.fsm.state.transitions.v1",
    )

    # Publish single notification
    notification = ModelStateTransitionNotification(
        aggregate_type="registration",
        aggregate_id=entity_id,
        from_state="pending",
        to_state="active",
        projection_version=1,
        correlation_id=correlation_id,
        causation_id=event_id,
        timestamp=datetime.now(UTC),
    )
    await publisher.publish(notification)

    # Batch publish
    await publisher.publish_batch([notification1, notification2])

    # Get metrics
    metrics = publisher.get_metrics()
    print(f"Published {metrics.notifications_published} notifications")
    ```

Related Tickets:
    - OMN-1139: Implement TransitionNotificationPublisher

See Also:
    - ProtocolTransitionNotificationPublisher: Protocol definition (omnibase_core)
    - ModelStateTransitionNotification: Notification model (omnibase_core)
    - ProtocolEventBusLike: Event bus protocol
"""

from __future__ import annotations

import asyncio
import logging
import time
from datetime import UTC, datetime
from typing import TYPE_CHECKING
from uuid import UUID, uuid4

from omnibase_core.models.events.model_event_envelope import ModelEventEnvelope
from omnibase_core.models.notifications import ModelStateTransitionNotification
from omnibase_core.protocols.notifications import (
    ProtocolTransitionNotificationPublisher,
)
from omnibase_infra.enums import EnumInfraTransportType
from omnibase_infra.errors import (
    InfraConnectionError,
    InfraTimeoutError,
    InfraUnavailableError,
    ModelInfraErrorContext,
    ModelTimeoutErrorContext,
)
from omnibase_infra.mixins import MixinAsyncCircuitBreaker
from omnibase_infra.models.resilience import ModelCircuitBreakerConfig
from omnibase_infra.runtime.models.model_transition_notification_publisher_metrics import (
    ModelTransitionNotificationPublisherMetrics,
)
from omnibase_infra.utils.util_error_sanitization import sanitize_error_string

if TYPE_CHECKING:
    from omnibase_infra.protocols import ProtocolEventBusLike

logger = logging.getLogger(__name__)


class TransitionNotificationPublisher(MixinAsyncCircuitBreaker):
    """Publishes transition notifications after projection commits.

    Implements ProtocolTransitionNotificationPublisher from omnibase_core.
    Provides at-least-once delivery semantics for state transition notifications
    to enable orchestrator coordination without tight coupling to reducers.

    Features:
        - Protocol compliant (ProtocolTransitionNotificationPublisher)
        - Circuit breaker resilience (MixinAsyncCircuitBreaker)
        - Metrics tracking for observability
        - Batch publishing for efficiency
        - Correlation ID propagation for distributed tracing

    Circuit Breaker:
        Uses MixinAsyncCircuitBreaker for resilience:
        - Opens after consecutive failures (configurable threshold)
        - Resets after timeout period (configurable)
        - Raises InfraUnavailableError when open

    Thread Safety:
        Coroutine-safe via asyncio.Lock for circuit breaker state.
        Not thread-safe - use only from async context.

    Attributes:
        _event_bus: Event bus for publishing notifications
        _topic: Target topic for notifications
        _lock: Async lock for metrics updates
        _publisher_id: Unique identifier for this publisher instance

    Example:
        >>> publisher = TransitionNotificationPublisher(event_bus, topic="notifications.v1")
        >>> await publisher.publish(notification)
        >>> metrics = publisher.get_metrics()
        >>> print(f"Success rate: {metrics.publish_success_rate():.2%}")
    """

    def __init__(
        self,
        event_bus: ProtocolEventBusLike,
        topic: str = "onex.fsm.state.transitions.v1",
        *,
        publisher_id: str | None = None,
        circuit_breaker_threshold: int = 5,
        circuit_breaker_reset_timeout: float = 60.0,
    ) -> None:
        """Initialize transition notification publisher.

        Args:
            event_bus: Event bus implementing ProtocolEventBusLike for publishing.
                Must support publish_envelope() method.
            topic: Target topic for transition notifications.
                Default: "onex.fsm.state.transitions.v1"
            publisher_id: Optional unique identifier for this publisher instance.
                If not provided, a UUID will be generated.
            circuit_breaker_threshold: Maximum failures before opening circuit.
                Default: 5
            circuit_breaker_reset_timeout: Seconds before automatic reset.
                Default: 60.0

        Example:
            >>> publisher = TransitionNotificationPublisher(
            ...     event_bus=kafka_event_bus,
            ...     topic="onex.fsm.state.transitions.v1",
            ...     circuit_breaker_threshold=3,
            ...     circuit_breaker_reset_timeout=30.0,
            ... )
        """
        self._event_bus = event_bus
        self._topic = topic
        self._publisher_id = publisher_id or f"transition-publisher-{uuid4()!s}"
        self._lock = asyncio.Lock()

        # Metrics tracking
        self._notifications_published = 0
        self._notifications_failed = 0
        self._batch_operations = 0
        self._batch_notifications_total = 0
        self._last_publish_at: datetime | None = None
        self._last_publish_duration_ms: float = 0.0
        self._total_publish_duration_ms: float = 0.0
        self._max_publish_duration_ms: float = 0.0
        self._started_at = datetime.now(UTC)

        # Initialize circuit breaker with configured settings
        # Note: the mixin sets self.circuit_breaker_threshold and
        # self.circuit_breaker_reset_timeout as instance attributes
        cb_config = ModelCircuitBreakerConfig(
            threshold=circuit_breaker_threshold,
            reset_timeout_seconds=circuit_breaker_reset_timeout,
            service_name=f"transition-notification-publisher.{topic}",
            transport_type=EnumInfraTransportType.KAFKA,
        )
        self._init_circuit_breaker_from_config(cb_config)

        logger.info(
            "TransitionNotificationPublisher initialized",
            extra={
                "publisher_id": self._publisher_id,
                "topic": self._topic,
                "circuit_breaker_threshold": circuit_breaker_threshold,
                "circuit_breaker_reset_timeout": circuit_breaker_reset_timeout,
            },
        )

    @property
    def topic(self) -> str:
        """Get the configured topic."""
        return self._topic

    @property
    def publisher_id(self) -> str:
        """Get the publisher identifier."""
        return self._publisher_id

    async def publish(
        self,
        notification: ModelStateTransitionNotification,
    ) -> None:
        """Publish a single state transition notification.

        Wraps the notification in a ModelEventEnvelope and publishes to the
        configured topic via the event bus. Implements at-least-once delivery
        semantics - consumers should handle idempotency via projection_version.

        Args:
            notification: The state transition notification to publish.

        Raises:
            InfraConnectionError: If event bus connection fails.
            InfraTimeoutError: If publish operation times out.
            InfraUnavailableError: If circuit breaker is open.

        Example:
            >>> notification = ModelStateTransitionNotification(
            ...     aggregate_type="registration",
            ...     aggregate_id=uuid4(),
            ...     from_state="pending",
            ...     to_state="active",
            ...     projection_version=1,
            ...     correlation_id=uuid4(),
            ...     causation_id=uuid4(),
            ...     timestamp=datetime.now(UTC),
            ... )
            >>> await publisher.publish(notification)
        """
        correlation_id = notification.correlation_id
        start_time = time.monotonic()

        # Check circuit breaker before operation
        async with self._circuit_breaker_lock:
            await self._check_circuit_breaker("publish", correlation_id)

        ctx = ModelInfraErrorContext.with_correlation(
            correlation_id=correlation_id,
            transport_type=EnumInfraTransportType.KAFKA,
            operation="publish_transition_notification",
            target_name=self._topic,
        )

        try:
            # Create envelope wrapping the notification
            envelope = ModelEventEnvelope(
                payload=notification.model_dump(),
                correlation_id=notification.correlation_id,
                source_tool=self._publisher_id,
            )

            # Publish to event bus
            await self._event_bus.publish_envelope(envelope, self._topic)

            # Calculate duration
            duration_ms = (time.monotonic() - start_time) * 1000

            # Record success
            async with self._circuit_breaker_lock:
                await self._reset_circuit_breaker()

            # Update metrics
            async with self._lock:
                self._notifications_published += 1
                self._last_publish_at = datetime.now(UTC)
                self._last_publish_duration_ms = duration_ms
                self._total_publish_duration_ms += duration_ms
                self._max_publish_duration_ms = max(
                    self._max_publish_duration_ms, duration_ms
                )

            logger.debug(
                "Published transition notification",
                extra={
                    "aggregate_type": notification.aggregate_type,
                    "aggregate_id": str(notification.aggregate_id),
                    "from_state": notification.from_state,
                    "to_state": notification.to_state,
                    "projection_version": notification.projection_version,
                    "correlation_id": str(correlation_id),
                    "duration_ms": duration_ms,
                },
            )

        except TimeoutError as e:
            await self._handle_failure("publish", correlation_id)
            timeout_ctx = ModelTimeoutErrorContext(
                transport_type=EnumInfraTransportType.KAFKA,
                operation="publish_transition_notification",
                target_name=self._topic,
                correlation_id=correlation_id,
            )
            raise InfraTimeoutError(
                f"Timeout publishing transition notification for "
                f"{notification.aggregate_type}:{notification.aggregate_id}",
                context=timeout_ctx,
            ) from e

        except Exception as e:
            await self._handle_failure("publish", correlation_id)
            raise InfraConnectionError(
                f"Failed to publish transition notification for "
                f"{notification.aggregate_type}:{notification.aggregate_id}",
                context=ctx,
            ) from e

    async def publish_batch(
        self,
        notifications: list[ModelStateTransitionNotification],
    ) -> None:
        """Publish multiple state transition notifications.

        Publishes each notification sequentially, continuing on individual
        failures. This method is provided for efficiency when multiple
        transitions occur in a single unit of work.

        Ordering:
            Notifications are published in the order provided. The order is
            preserved when delivery order matters for workflow correctness.

        Error Handling:
            If any notification fails to publish, the error is raised after
            attempting all notifications. Partial success is possible.

        Args:
            notifications: List of notifications to publish.

        Raises:
            InfraConnectionError: If event bus connection fails.
            InfraTimeoutError: If publish operation times out.
            InfraUnavailableError: If circuit breaker is open.

        Example:
            >>> notifications = [notification1, notification2, notification3]
            >>> await publisher.publish_batch(notifications)
        """
        if not notifications:
            return

        correlation_id = notifications[0].correlation_id if notifications else uuid4()
        start_time = time.monotonic()

        # Check circuit breaker before starting batch
        async with self._circuit_breaker_lock:
            await self._check_circuit_breaker("publish_batch", correlation_id)

        success_count = 0
        last_error: Exception | None = None

        for notification in notifications:
            try:
                await self.publish(notification)
                success_count += 1
            except (
                InfraConnectionError,
                InfraTimeoutError,
                InfraUnavailableError,
            ) as e:
                last_error = e
                logger.warning(
                    "Failed to publish notification in batch",
                    extra={
                        "aggregate_type": notification.aggregate_type,
                        "aggregate_id": str(notification.aggregate_id),
                        "error": sanitize_error_string(str(e)),
                        "correlation_id": str(notification.correlation_id),
                    },
                )
                # Continue with remaining notifications

        # Calculate duration
        duration_ms = (time.monotonic() - start_time) * 1000

        # Update batch metrics
        async with self._lock:
            self._batch_operations += 1
            self._batch_notifications_total += success_count

        failure_count = len(notifications) - success_count

        logger.info(
            "Batch publish completed",
            extra={
                "total": len(notifications),
                "success": success_count,
                "failed": failure_count,
                "duration_ms": duration_ms,
                "correlation_id": str(correlation_id),
            },
        )

        # Raise with count information if any failures occurred
        if last_error is not None:
            ctx = ModelInfraErrorContext.with_correlation(
                correlation_id=correlation_id,
                transport_type=EnumInfraTransportType.KAFKA,
                operation="publish_batch",
                target_name=self._topic,
            )
            raise InfraConnectionError(
                f"Batch publish partially failed: {failure_count}/{len(notifications)} "
                f"notifications failed ({success_count} succeeded). "
                f"Last error: {sanitize_error_string(str(last_error))}",
                context=ctx,
            ) from last_error

    async def _handle_failure(
        self,
        operation: str,
        correlation_id: UUID,
    ) -> None:
        """Handle a publish failure by recording circuit breaker failure.

        Args:
            operation: Operation name for logging
            correlation_id: Correlation ID for tracing
        """
        async with self._circuit_breaker_lock:
            await self._record_circuit_failure(operation, correlation_id)

        async with self._lock:
            self._notifications_failed += 1

    def get_metrics(self) -> ModelTransitionNotificationPublisherMetrics:
        """Get current publisher metrics.

        Returns a snapshot of the publisher's operational metrics including
        notification counts, timing information, and circuit breaker state.

        Returns:
            ModelTransitionNotificationPublisherMetrics with current values.

        Example:
            >>> metrics = publisher.get_metrics()
            >>> print(f"Published: {metrics.notifications_published}")
            >>> print(f"Success rate: {metrics.publish_success_rate():.2%}")
            >>> print(f"Healthy: {metrics.is_healthy()}")
        """
        # Get circuit breaker state
        cb_state = self._get_circuit_breaker_state()
        cb_open = cb_state.get("state") == "open"
        failures_value = cb_state.get("failures", 0)
        consecutive_failures = failures_value if isinstance(failures_value, int) else 0

        # Calculate average duration (only from successful publishes since
        # _total_publish_duration_ms is only updated on success)
        average_duration = (
            self._total_publish_duration_ms / self._notifications_published
            if self._notifications_published > 0
            else 0.0
        )

        return ModelTransitionNotificationPublisherMetrics(
            publisher_id=self._publisher_id,
            topic=self._topic,
            notifications_published=self._notifications_published,
            notifications_failed=self._notifications_failed,
            batch_operations=self._batch_operations,
            batch_notifications_total=self._batch_notifications_total,
            last_publish_at=self._last_publish_at,
            last_publish_duration_ms=self._last_publish_duration_ms,
            average_publish_duration_ms=average_duration,
            max_publish_duration_ms=self._max_publish_duration_ms,
            circuit_breaker_open=cb_open,
            consecutive_failures=consecutive_failures,
            started_at=self._started_at,
        )


# Protocol compliance check (runtime_checkable allows isinstance checks)
def _verify_protocol_compliance() -> None:  # pragma: no cover
    """Verify TransitionNotificationPublisher implements the protocol.

    This function is never called at runtime - it exists purely for static
    type checking verification that the implementation is protocol-compliant.
    """
    from typing import cast

    from omnibase_infra.event_bus.event_bus_inmemory import EventBusInmemory

    # Create instance to verify protocol compliance
    bus = cast("ProtocolEventBusLike", EventBusInmemory())
    publisher: ProtocolTransitionNotificationPublisher = (
        TransitionNotificationPublisher(event_bus=bus)
    )
    # Use the variable to silence unused warnings
    _ = publisher


__all__: list[str] = ["TransitionNotificationPublisher"]
