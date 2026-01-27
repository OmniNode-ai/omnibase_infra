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
        topic=SUFFIX_FSM_STATE_TRANSITIONS,
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
from typing import TYPE_CHECKING, NamedTuple
from uuid import UUID

from omnibase_core.models.events.model_event_envelope import ModelEventEnvelope
from omnibase_core.models.notifications import ModelStateTransitionNotification
from omnibase_core.protocols.notifications import (
    ProtocolTransitionNotificationPublisher,
)
from omnibase_core.utils.util_uuid_service import UtilUUID
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
from omnibase_infra.topics import SUFFIX_FSM_STATE_TRANSITIONS
from omnibase_infra.utils.util_error_sanitization import sanitize_error_string

if TYPE_CHECKING:
    from omnibase_infra.protocols import ProtocolEventBusLike

logger = logging.getLogger(__name__)


class FailedNotificationRecord(NamedTuple):
    """Record of a failed notification publish attempt.

    Used to track failures during batch publishing operations with clear
    field semantics for error reporting and debugging.

    Attributes:
        aggregate_type: The type of aggregate that failed (e.g., "registration").
        aggregate_id: The ID of the aggregate (as string for error reporting).
        error_message: Sanitized error message describing the failure.
    """

    aggregate_type: str
    aggregate_id: str
    error_message: str


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

    # Default maximum number of failures to track in memory during batch operations.
    # Prevents unbounded memory growth for very large batches with many failures.
    # Can be overridden via constructor parameter for large batch tuning.
    DEFAULT_MAX_TRACKED_FAILURES: int = 100

    def __init__(
        self,
        event_bus: ProtocolEventBusLike,
        topic: str,
        *,
        publisher_id: str | None = None,
        circuit_breaker_threshold: int = 5,
        circuit_breaker_reset_timeout: float = 60.0,
        max_tracked_failures: int = DEFAULT_MAX_TRACKED_FAILURES,
    ) -> None:
        """Initialize transition notification publisher.

        Args:
            event_bus: Event bus implementing ProtocolEventBusLike for publishing.
                Must support publish_envelope() method.
            topic: Target topic for transition notifications. Required.
                This should be configured in the projector's contract or
                notification config rather than hardcoded. Example topics:
                - "onex.fsm.state.transitions.v1"
                - "registration.state.transitions.v1"
            publisher_id: Optional unique identifier for this publisher instance.
                If not provided, a UUID will be generated.
            circuit_breaker_threshold: Maximum failures before opening circuit.
                Default: 5
            circuit_breaker_reset_timeout: Seconds before automatic reset.
                Default: 60.0
            max_tracked_failures: Maximum number of failures to track in memory
                during batch operations. Prevents unbounded memory growth for
                very large batches with many failures. For large batch operations,
                this can be tuned higher to capture more failure details.
                Default: 100

        Example:
            >>> publisher = TransitionNotificationPublisher(
            ...     event_bus=kafka_event_bus,
            ...     topic="onex.fsm.state.transitions.v1",
            ...     circuit_breaker_threshold=3,
            ...     circuit_breaker_reset_timeout=30.0,
            ...     max_tracked_failures=200,  # Tune for large batches
            ... )
        """
        self._event_bus = event_bus
        self._topic = topic
        self._publisher_id = (
            publisher_id or f"transition-publisher-{UtilUUID.generate()!s}"
        )
        self._lock = asyncio.Lock()
        self._max_tracked_failures = max_tracked_failures

        # Metrics tracking
        self._notifications_published = 0
        self._notifications_failed = 0
        self._batch_operations = 0
        self._batch_notifications_attempted = 0
        self._batch_notifications_total = 0
        self._batch_failures_truncated = 0
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
                "max_tracked_failures": self._max_tracked_failures,
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
            # Create envelope wrapping the notification model directly.
            # ModelEventEnvelope[T] is generic and handles Pydantic models natively,
            # serializing them lazily when needed via to_dict_lazy().
            envelope = ModelEventEnvelope[ModelStateTransitionNotification](
                payload=notification,
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

        except (InfraUnavailableError, InfraTimeoutError):
            # Re-raise infrastructure errors without wrapping - preserve error semantics
            await self._handle_failure("publish", correlation_id)
            raise

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

        Circuit Breaker Behavior:
            The circuit breaker is checked only at the start of the batch
            operation. However, individual publish() calls within the batch
            can trip the circuit breaker if they fail. If the circuit breaker
            opens mid-batch (due to accumulated failures from individual
            publish calls), subsequent notifications in the batch will fail
            with InfraUnavailableError. This is expected "partial success"
            behavior - the batch continues attempting all notifications, but
            failures are recorded and reported at the end.

        Correlation ID Behavior:
            The batch uses the **first notification's correlation_id** for all
            batch-level operations:

            - Circuit breaker checks (at batch start)
            - Batch summary logging ("Batch publish completed")
            - Error context creation (when raising InfraConnectionError)
            - Failure summary logging ("Batch publish failures - details")

            However, **individual notification errors are logged with their own
            correlation_id**. When a specific notification fails within the batch,
            the warning log entry includes that notification's correlation_id,
            not the batch correlation_id.

            This design is intentional:

            1. **Batch-level traceability**: Using a single correlation_id for
               batch operations allows operators to correlate all batch-related
               log entries and metrics under one trace ID.

            2. **Per-notification traceability**: Individual failure logs retain
               their specific correlation_id, enabling operators to trace the
               complete lifecycle of each notification independently.

            Example log correlation::

                # Batch-level log (uses first notification's correlation_id)
                {"message": "Batch publish completed", "correlation_id": "aaa-111"}

                # Individual failure log (uses that notification's correlation_id)
                {"message": "Failed to publish notification in batch",
                 "correlation_id": "bbb-222"}

        Args:
            notifications: List of notifications to publish.

        Raises:
            InfraConnectionError: If event bus connection fails.
            InfraTimeoutError: If publish operation times out.
            InfraUnavailableError: If circuit breaker is open (at batch start
                or if tripped mid-batch by individual publish failures).

        Example:
            >>> notifications = [notification1, notification2, notification3]
            >>> await publisher.publish_batch(notifications)
        """
        if not notifications:
            return

        correlation_id = notifications[0].correlation_id
        start_time = time.monotonic()

        # Batch-level circuit breaker check for fail-fast behavior.
        # NOTE: This check is NOT redundant with the per-notification check in publish().
        # - This check: Fail-fast before starting any work if circuit is already open
        # - Per-notification checks in publish(): Handle circuit opening MID-batch due to
        #   accumulated failures during batch processing (expected partial-success behavior)
        # See docstring "Circuit Breaker Behavior" section for full explanation.
        async with self._circuit_breaker_lock:
            await self._check_circuit_breaker("publish_batch", correlation_id)

        success_count = 0
        last_error: Exception | None = None
        failed_notifications: list[FailedNotificationRecord] = []
        truncation_occurred = False
        # Track error types to determine most severe error for final raise.
        # Severity order: InfraUnavailableError > InfraTimeoutError > InfraConnectionError
        encountered_unavailable = False
        encountered_timeout = False

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
                # Track error types for determining most severe error to raise
                if isinstance(e, InfraUnavailableError):
                    encountered_unavailable = True
                elif isinstance(e, InfraTimeoutError):
                    encountered_timeout = True
                # Only track failures up to the limit to prevent unbounded memory growth
                if len(failed_notifications) < self._max_tracked_failures:
                    failed_notifications.append(
                        FailedNotificationRecord(
                            aggregate_type=notification.aggregate_type,
                            aggregate_id=str(notification.aggregate_id),
                            error_message=sanitize_error_string(str(e)),
                        )
                    )
                else:
                    # Mark that truncation occurred (limit reached)
                    truncation_occurred = True
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
            self._batch_notifications_attempted += len(notifications)
            self._batch_notifications_total += success_count
            if truncation_occurred:
                self._batch_failures_truncated += 1

        failure_count = len(notifications) - success_count

        # Log aggregate failure information when truncation occurs
        if truncation_occurred:
            failure_summary = self._summarize_failure_types(failed_notifications)
            untracked_failures = failure_count - len(failed_notifications)
            logger.warning(
                "Batch publish failure tracking truncated",
                extra={
                    "correlation_id": str(correlation_id),
                    "total_failures": failure_count,
                    "tracked_failures": len(failed_notifications),
                    "untracked_failures": untracked_failures,
                    "max_tracked_failures": self._max_tracked_failures,
                    "failure_type_summary": failure_summary,
                },
            )

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

        # Raise with detailed failure information if any failures occurred
        if last_error is not None:
            ctx = ModelInfraErrorContext.with_correlation(
                correlation_id=correlation_id,
                transport_type=EnumInfraTransportType.KAFKA,
                operation="publish_batch",
                target_name=self._topic,
            )

            # Log failure details for debugging before raising truncated error.
            # Limit logged failures to prevent oversized log entries while
            # preserving full counts for metrics and observability.
            max_logged_failures = 10
            logged_failures = [
                {
                    "aggregate_type": record.aggregate_type,
                    "aggregate_id": record.aggregate_id,
                    "error_message": record.error_message,
                }
                for record in failed_notifications[:max_logged_failures]
            ]
            failures_truncated = len(failed_notifications) > max_logged_failures

            logger.warning(
                "Batch publish failures - details",
                extra={
                    "correlation_id": str(correlation_id),
                    "topic": self._topic,
                    "total_notifications": len(notifications),
                    "success_count": success_count,
                    "failure_count": failure_count,
                    "tracked_failures": len(failed_notifications),
                    "max_tracked_failures": self._max_tracked_failures,
                    "logged_failures": len(logged_failures),
                    "failures_truncated": failures_truncated,
                    "failures": logged_failures,
                },
            )

            # Build detailed error message showing first 3 failures
            failure_details = "; ".join(
                f"{record.aggregate_type}:{record.aggregate_id[:8]}... - "
                f"{record.error_message[:50]}"
                for record in failed_notifications[:3]
            )
            if failure_count > 3:
                failure_details += f" ... and {failure_count - 3} more"

            error_message = (
                f"Batch publish partially failed: {failure_count}/{len(notifications)} "
                f"notifications failed ({success_count} succeeded). "
                f"Failures: [{failure_details}]"
            )

            # Raise the most severe error type encountered during batch processing.
            # Severity order: InfraUnavailableError > InfraTimeoutError > InfraConnectionError
            # This preserves error semantics so callers can handle appropriately
            # (e.g., retry on timeout, skip on unavailable).
            if encountered_unavailable:
                raise InfraUnavailableError(
                    error_message,
                    context=ctx,
                ) from last_error
            if encountered_timeout:
                timeout_ctx = ModelTimeoutErrorContext(
                    transport_type=EnumInfraTransportType.KAFKA,
                    operation="publish_batch",
                    target_name=self._topic,
                    correlation_id=correlation_id,
                )
                raise InfraTimeoutError(
                    error_message,
                    context=timeout_ctx,
                ) from last_error
            raise InfraConnectionError(
                error_message,
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

    def _summarize_failure_types(
        self, failures: list[FailedNotificationRecord]
    ) -> dict[str, int]:
        """Summarize failure types by grouping error messages.

        Groups failures by a simplified error pattern (first 50 characters of
        the error message) to help operators understand what types of errors
        are occurring, even when detailed failure records are truncated.

        Args:
            failures: List of failed notification records to summarize.

        Returns:
            Dictionary mapping error pattern (truncated error message) to
            the count of failures with that pattern.

        Example:
            >>> failures = [
            ...     FailedNotificationRecord("reg", "id1", "Connection refused to broker"),
            ...     FailedNotificationRecord("reg", "id2", "Connection refused to broker"),
            ...     FailedNotificationRecord("reg", "id3", "Timeout waiting for response"),
            ... ]
            >>> summary = publisher._summarize_failure_types(failures)
            >>> # {"Connection refused to broker": 2, "Timeout waiting for response": 1}
        """
        summary: dict[str, int] = {}
        for failure in failures:
            # Use first 50 chars as the pattern key for grouping
            pattern = failure.error_message[:50]
            summary[pattern] = summary.get(pattern, 0) + 1
        return summary

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
            batch_notifications_attempted=self._batch_notifications_attempted,
            batch_notifications_total=self._batch_notifications_total,
            batch_failures_truncated=self._batch_failures_truncated,
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
        TransitionNotificationPublisher(
            event_bus=bus,
            topic=SUFFIX_FSM_STATE_TRANSITIONS,
        )
    )
    # Use the variable to silence unused warnings
    _ = publisher


__all__: list[str] = ["FailedNotificationRecord", "TransitionNotificationPublisher"]
