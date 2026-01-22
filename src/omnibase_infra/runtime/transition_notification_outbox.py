# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Transition Notification Outbox for guaranteed delivery.

This module implements the outbox pattern for state transition notifications.
The outbox stores notifications in the same database transaction as projections,
then processes them asynchronously via a background processor to ensure
at-least-once delivery semantics.

At-Least-Once Delivery Semantics:
    This implementation guarantees that every notification will be delivered
    **at least once**, but **duplicates are possible** during failure scenarios:

    - If the publisher succeeds but the database update fails, the notification
      will be re-published on the next processing cycle.
    - If the processor crashes after publishing but before marking as processed,
      the notification will be re-published when the processor restarts.
    - Network partitions or timeouts can cause similar duplicate delivery.

    **CRITICAL**: Consumers MUST implement idempotent message handling. This
    typically means:

    - Tracking processed notification IDs (using ``notification_id`` field)
    - Using database upserts with conflict detection
    - Designing state transitions to be idempotent (same transition twice = no-op)

Database Schema (must be created before use):
    ```sql
    CREATE TABLE transition_notification_outbox (
        id BIGSERIAL PRIMARY KEY,
        notification_data JSONB NOT NULL,
        created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
        processed_at TIMESTAMPTZ,
        retry_count INT NOT NULL DEFAULT 0,
        last_error TEXT,
        aggregate_type TEXT NOT NULL,
        aggregate_id UUID NOT NULL
    );

    -- Index for efficient pending notification queries
    CREATE INDEX idx_outbox_pending ON transition_notification_outbox (created_at)
        WHERE processed_at IS NULL;

    -- Index for aggregate-specific queries
    CREATE INDEX idx_outbox_aggregate ON transition_notification_outbox
        (aggregate_type, aggregate_id);
    ```

Key Features:
    - Stores notifications in same transaction as projection writes
    - Background processor publishes pending notifications
    - SELECT FOR UPDATE SKIP LOCKED for safe concurrent processing
    - Retry tracking with error recording
    - Configurable batch size and poll interval
    - Graceful shutdown with proper lifecycle management

Concurrency Safety:
    This implementation is coroutine-safe using asyncio primitives:
    - Background loop protected by asyncio.Lock
    - Shutdown signaling via asyncio.Event
    Note: This is coroutine-safe, not thread-safe.

Related Tickets:
    - OMN-1139: TransitionNotificationOutbox implementation (Optional Enhancement)

.. versionadded:: 0.8.0
"""

from __future__ import annotations

import asyncio
import logging
from uuid import UUID

import asyncpg

# Use core model and protocol
from omnibase_core.models.notifications import ModelStateTransitionNotification
from omnibase_core.protocols.notifications import (
    ProtocolTransitionNotificationPublisher,
)
from omnibase_core.utils.util_uuid_service import UtilUUID
from omnibase_infra.enums import EnumInfraTransportType
from omnibase_infra.errors import (
    InfraConnectionError,
    InfraTimeoutError,
    ModelInfraErrorContext,
    ModelTimeoutErrorContext,
    ProtocolConfigurationError,
    RuntimeHostError,
)
from omnibase_infra.models.projectors.util_sql_identifiers import quote_identifier
from omnibase_infra.runtime.models.model_transition_notification_outbox_metrics import (
    ModelTransitionNotificationOutboxMetrics,
)
from omnibase_infra.utils.util_error_sanitization import sanitize_error_string

logger = logging.getLogger(__name__)


class TransitionNotificationOutbox:
    """Outbox pattern for guaranteed notification delivery.

    Stores notifications in the same database transaction as projections,
    ensuring at-least-once semantics. A background processor publishes
    pending notifications asynchronously.

    Warning:
        **Duplicate Delivery**: This implementation provides at-least-once
        delivery, meaning **duplicates are possible** during failures. If the
        publisher succeeds but the subsequent database update fails (marking
        the notification as processed), the notification will be re-published
        on the next processing cycle. Consumers MUST implement idempotent
        message handling to safely handle duplicate notifications.

    The outbox pattern solves the dual-write problem: when you need to
    update a database AND publish an event, either operation could fail
    independently, leading to inconsistent state. By writing the event
    to an outbox table in the same transaction as the data change, we
    guarantee atomicity. A separate process then reads from the outbox
    and publishes events.

    Dead Letter Queue (DLQ) Support:
        When configured with ``max_retries`` and ``dlq_publisher``, notifications
        that exceed the retry threshold are moved to a dead letter queue instead
        of being retried indefinitely. This prevents poison messages from blocking
        the outbox and provides a way to inspect and replay failed notifications.

        DLQ notifications are published with the original notification payload,
        allowing downstream consumers to process or investigate failures.

    Warning:
        **DLQ Unavailability Risk**: If the DLQ itself becomes permanently
        unavailable, notifications that have exceeded ``max_retries`` will
        continue to be retried indefinitely. This occurs because ``retry_count``
        is intentionally NOT incremented when DLQ publish fails (to preserve
        the retry state for when the DLQ recovers).

        **Monitoring Recommendation**: Monitor for notifications matching:
        ``processed_at IS NULL AND retry_count >= max_retries``. Notifications
        in this state indicate DLQ availability issues requiring operator
        intervention.

    Attributes:
        table_name: Name of the outbox table (default: "transition_notification_outbox")
        batch_size: Number of notifications to process per batch (default: 100)
        poll_interval: Seconds between processing polls when idle (default: 1.0)
        shutdown_timeout: Seconds to wait for graceful shutdown during stop() (default: 10.0)
        is_running: Whether the background processor is running
        max_retries: Maximum retry attempts before moving to DLQ (None if DLQ disabled)
        dlq_topic: DLQ topic name for metrics/logging (None if DLQ disabled)

    Concurrency Safety:
        This implementation is coroutine-safe using asyncio primitives:
        - Background loop protected by ``_lock`` (asyncio.Lock)
        - Shutdown signaling via ``_shutdown_event`` (asyncio.Event)
        Note: This is coroutine-safe, not thread-safe.

    Example:
        >>> from asyncpg import create_pool
        >>> from omnibase_infra.runtime import TransitionNotificationOutbox
        >>>
        >>> # Create outbox with publisher
        >>> pool = await create_pool(dsn)
        >>> publisher = KafkaTransitionPublisher()
        >>> outbox = TransitionNotificationOutbox(
        ...     pool=pool,
        ...     publisher=publisher,
        ...     batch_size=50,
        ...     poll_interval_seconds=0.5,
        ... )
        >>>
        >>> # Start background processor
        >>> await outbox.start()
        >>>
        >>> # In projection transaction - store notification
        >>> async with pool.acquire() as conn:
        ...     async with conn.transaction():
        ...         # Update projection...
        ...         await projector.project(event, correlation_id)
        ...         # Store notification in same transaction
        ...         await outbox.store(notification, conn)
        >>>
        >>> # Stop gracefully
        >>> await outbox.stop()

    Example with DLQ:
        >>> # Create outbox with DLQ support
        >>> dlq_publisher = KafkaDLQPublisher(topic="notifications-dlq")
        >>> outbox = TransitionNotificationOutbox(
        ...     pool=pool,
        ...     publisher=publisher,
        ...     max_retries=3,
        ...     dlq_publisher=dlq_publisher,
        ...     dlq_topic="notifications-dlq",
        ... )
        >>> # Notifications failing 3+ times will be moved to DLQ

    Related:
        - OMN-1139: TransitionNotificationOutbox implementation
        - ProtocolTransitionNotificationPublisher: Publisher protocol
        - ModelStateTransitionNotification: Notification model
    """

    # Default configuration values
    DEFAULT_TABLE_NAME: str = "transition_notification_outbox"
    DEFAULT_BATCH_SIZE: int = 100
    DEFAULT_POLL_INTERVAL_SECONDS: float = 1.0
    DEFAULT_QUERY_TIMEOUT_SECONDS: float = 30.0
    DEFAULT_STRICT_TRANSACTION_MODE: bool = True
    DEFAULT_SHUTDOWN_TIMEOUT_SECONDS: float = 10.0
    MAX_ERROR_MESSAGE_LENGTH: int = 1000

    def __init__(
        self,
        pool: asyncpg.Pool,
        publisher: ProtocolTransitionNotificationPublisher,
        table_name: str = DEFAULT_TABLE_NAME,
        batch_size: int = DEFAULT_BATCH_SIZE,
        poll_interval_seconds: float = DEFAULT_POLL_INTERVAL_SECONDS,
        query_timeout_seconds: float = DEFAULT_QUERY_TIMEOUT_SECONDS,
        strict_transaction_mode: bool = DEFAULT_STRICT_TRANSACTION_MODE,
        shutdown_timeout_seconds: float = DEFAULT_SHUTDOWN_TIMEOUT_SECONDS,
        max_retries: int | None = None,
        dlq_publisher: ProtocolTransitionNotificationPublisher | None = None,
        dlq_topic: str | None = None,
    ) -> None:
        """Initialize the TransitionNotificationOutbox.

        Args:
            pool: asyncpg connection pool for database access.
            publisher: Publisher implementation for delivering notifications.
            table_name: Name of the outbox table (default: "transition_notification_outbox").
            batch_size: Maximum notifications to process per batch (default: 100).
            poll_interval_seconds: Seconds between polls when idle (default: 1.0).
            query_timeout_seconds: Timeout for database queries (default: 30.0).
            strict_transaction_mode: If True (default), raises ProtocolConfigurationError
                when store() is called outside a transaction context, providing
                fail-fast behavior to catch misconfiguration early. If False,
                logs a warning but continues execution (atomicity not guaranteed).
            shutdown_timeout_seconds: Timeout in seconds for graceful shutdown
                during stop() (default: 10.0). If the background processor does
                not complete within this timeout, it will be cancelled.
            max_retries: Maximum retry attempts before moving notification to DLQ.
                Must be >= 1 if specified. If None (default), DLQ is disabled.
            dlq_publisher: Publisher for dead letter queue. Required if max_retries
                is specified. If None when max_retries is set, raises
                ProtocolConfigurationError.
            dlq_topic: Topic name for DLQ (for metrics/logging purposes).
                Optional, used for observability.

        Raises:
            ProtocolConfigurationError: If pool or publisher is None, if
                configuration values are invalid, if max_retries < 1, or if
                max_retries is set but dlq_publisher is None.
        """
        context = ModelInfraErrorContext(
            transport_type=EnumInfraTransportType.DATABASE,
            operation="outbox_init",
        )

        if pool is None:
            raise ProtocolConfigurationError(
                "pool cannot be None",
                context=context,
            )
        if publisher is None:
            raise ProtocolConfigurationError(
                "publisher cannot be None",
                context=context,
            )
        if batch_size < 1:
            raise ProtocolConfigurationError(
                f"batch_size must be >= 1, got {batch_size}",
                context=context,
                parameter="batch_size",
                value=batch_size,
            )
        if poll_interval_seconds <= 0:
            raise ProtocolConfigurationError(
                f"poll_interval_seconds must be > 0, got {poll_interval_seconds}",
                context=context,
                parameter="poll_interval_seconds",
                value=poll_interval_seconds,
            )
        if shutdown_timeout_seconds <= 0:
            raise ProtocolConfigurationError(
                f"shutdown_timeout_seconds must be > 0, got {shutdown_timeout_seconds}",
                context=context,
                parameter="shutdown_timeout_seconds",
                value=shutdown_timeout_seconds,
            )

        # DLQ validation
        if max_retries is not None and max_retries < 1:
            raise ProtocolConfigurationError(
                f"max_retries must be >= 1, got {max_retries}",
                context=context,
                parameter="max_retries",
                value=max_retries,
            )

        if max_retries is not None and dlq_publisher is None:
            raise ProtocolConfigurationError(
                "dlq_publisher is required when max_retries is configured",
                context=context,
                parameter="dlq_publisher",
            )

        if dlq_publisher is not None and max_retries is None:
            logger.warning(
                "dlq_publisher configured but max_retries is None - DLQ will never be used",
                extra={
                    "table_name": table_name,
                    "dlq_topic": dlq_topic,
                },
            )

        self._pool = pool
        self._publisher = publisher
        self._table_name = table_name
        self._batch_size = batch_size
        self._poll_interval = poll_interval_seconds
        self._query_timeout = query_timeout_seconds
        self._strict_transaction_mode = strict_transaction_mode
        self._shutdown_timeout = shutdown_timeout_seconds

        # State management
        self._running = False
        self._lock = asyncio.Lock()
        self._shutdown_event = asyncio.Event()
        self._processor_task: asyncio.Task[None] | None = None

        # Metrics tracking
        self._notifications_stored: int = 0
        self._notifications_processed: int = 0
        self._notifications_failed: int = 0
        self._notifications_sent_to_dlq: int = 0
        self._dlq_publish_failures: int = 0

        # DLQ configuration
        self._max_retries = max_retries
        self._dlq_publisher = dlq_publisher
        self._dlq_topic = dlq_topic

        logger.debug(
            "TransitionNotificationOutbox initialized",
            extra={
                "table_name": table_name,
                "batch_size": batch_size,
                "poll_interval_seconds": poll_interval_seconds,
                "strict_transaction_mode": strict_transaction_mode,
                "shutdown_timeout_seconds": shutdown_timeout_seconds,
                "max_retries": max_retries,
                "dlq_enabled": dlq_publisher is not None,
                "dlq_topic": dlq_topic,
            },
        )

    @property
    def table_name(self) -> str:
        """Return the outbox table name."""
        return self._table_name

    @property
    def batch_size(self) -> int:
        """Return the batch size for processing."""
        return self._batch_size

    @property
    def poll_interval(self) -> float:
        """Return the poll interval in seconds."""
        return self._poll_interval

    @property
    def shutdown_timeout(self) -> float:
        """Return the shutdown timeout in seconds."""
        return self._shutdown_timeout

    @property
    def is_running(self) -> bool:
        """Return whether the background processor is running."""
        return self._running

    @property
    def notifications_stored(self) -> int:
        """Return total notifications stored."""
        return self._notifications_stored

    @property
    def notifications_processed(self) -> int:
        """Return total notifications successfully processed."""
        return self._notifications_processed

    @property
    def notifications_failed(self) -> int:
        """Return total notifications that failed processing."""
        return self._notifications_failed

    @property
    def strict_transaction_mode(self) -> bool:
        """Return whether strict transaction mode is enabled.

        When enabled, store() raises ProtocolConfigurationError if called
        outside a transaction context, rather than just logging a warning.
        """
        return self._strict_transaction_mode

    @property
    def max_retries(self) -> int | None:
        """Return the max retries before DLQ, or None if DLQ disabled."""
        return self._max_retries

    @property
    def dlq_topic(self) -> str | None:
        """Return the DLQ topic name for metrics/logging."""
        return self._dlq_topic

    @property
    def notifications_sent_to_dlq(self) -> int:
        """Return total notifications sent to DLQ."""
        return self._notifications_sent_to_dlq

    @property
    def dlq_publish_failures(self) -> int:
        """Return count of failed DLQ publish attempts.

        Non-zero values indicate DLQ availability issues. Monitor this metric
        to detect when the DLQ is unavailable, which can cause infinite retry
        loops for notifications that have exceeded max_retries.
        """
        return self._dlq_publish_failures

    async def store(
        self,
        notification: ModelStateTransitionNotification,
        conn: asyncpg.Connection,
    ) -> None:
        """Store notification in outbox using the same connection/transaction.

        This method MUST be called within the same transaction as the projection
        write to ensure atomicity. The notification will be picked up by the
        background processor and published asynchronously.

        Warning:
            If called outside a transaction (auto-commit mode), behavior depends
            on ``strict_transaction_mode``:

            - **strict_transaction_mode=True** (default): Raises ProtocolConfigurationError
              immediately, providing fail-fast behavior to catch misconfiguration early.
            - **strict_transaction_mode=False**: Logs a WARNING but continues execution.
              The atomicity guarantee with projection writes will be broken in this case.

        Args:
            notification: The state transition notification to store.
            conn: The database connection from the current transaction.
                MUST be the same connection used for the projection write.

        Raises:
            ProtocolConfigurationError: If strict_transaction_mode is True and
                store() is called outside a transaction context.
            InfraConnectionError: If database connection fails.
            InfraTimeoutError: If store operation times out.
            RuntimeHostError: For other database errors.

        Example:
            >>> async with pool.acquire() as conn:
            ...     async with conn.transaction():
            ...         # Update projection in same transaction
            ...         await projector.project(event, correlation_id)
            ...         # Store notification - uses same transaction
            ...         await outbox.store(notification, conn)
        """
        correlation_id = notification.correlation_id
        ctx = ModelInfraErrorContext(
            transport_type=EnumInfraTransportType.DATABASE,
            operation="outbox_store",
            target_name=self._table_name,
            correlation_id=correlation_id,
        )

        # Check transaction context - behavior depends on strict_transaction_mode
        if not conn.is_in_transaction():
            if self._strict_transaction_mode:
                raise ProtocolConfigurationError(
                    "store() called outside transaction context in strict mode - "
                    "atomicity with projection not guaranteed",
                    context=ctx,
                )
            logger.warning(
                "store() called outside transaction context - "
                "atomicity with projection not guaranteed",
                extra={
                    "table_name": self._table_name,
                    "aggregate_type": notification.aggregate_type,
                    "aggregate_id": str(notification.aggregate_id),
                    "correlation_id": str(correlation_id),
                },
            )

        # Build INSERT query - table name from trusted config, quoted for safety
        # S608: Safe - table name from constructor, quoted via quote_identifier()
        table_quoted = quote_identifier(self._table_name)
        query = f"""
            INSERT INTO {table_quoted}
            (notification_data, aggregate_type, aggregate_id)
            VALUES ($1, $2, $3)
        """  # noqa: S608

        try:
            await conn.execute(
                query,
                notification.model_dump_json(),
                notification.aggregate_type,
                notification.aggregate_id,
                timeout=self._query_timeout,
            )

            self._notifications_stored += 1

            logger.debug(
                "Notification stored in outbox",
                extra={
                    "aggregate_type": notification.aggregate_type,
                    "aggregate_id": str(notification.aggregate_id),
                    "correlation_id": str(correlation_id),
                },
            )

        except asyncpg.PostgresConnectionError as e:
            raise InfraConnectionError(
                f"Failed to store notification in outbox: {self._table_name}",
                context=ctx,
            ) from e

        except asyncpg.QueryCanceledError as e:
            timeout_ctx = ModelTimeoutErrorContext(
                transport_type=EnumInfraTransportType.DATABASE,
                operation="outbox_store",
                target_name=self._table_name,
                correlation_id=correlation_id,
            )
            raise InfraTimeoutError(
                f"Timeout storing notification in outbox: {self._table_name}",
                context=timeout_ctx,
            ) from e

        except Exception as e:
            raise RuntimeHostError(
                f"Failed to store notification: {type(e).__name__}",
                context=ctx,
            ) from e

    async def process_pending(self) -> int:
        """Process pending notifications from outbox.

        Fetches pending notifications using SELECT FOR UPDATE SKIP LOCKED
        for safe concurrent processing, publishes them via the publisher,
        and marks them as processed.

        Returns:
            Count of successfully processed notifications.

        Raises:
            InfraConnectionError: If database connection fails.
            InfraTimeoutError: If query times out.
            RuntimeHostError: For other database errors.

        Note:
            Individual notification publish failures are recorded but do not
            cause the method to raise. The failed notification's retry_count
            and last_error are updated in the database.
        """
        correlation_id = UtilUUID.generate_correlation_id()
        ctx = ModelInfraErrorContext(
            transport_type=EnumInfraTransportType.DATABASE,
            operation="outbox_process_pending",
            target_name=self._table_name,
            correlation_id=correlation_id,
        )

        # Build queries - table name from trusted config, quoted for safety
        table_quoted = quote_identifier(self._table_name)

        # SELECT query with FOR UPDATE SKIP LOCKED for concurrent safety
        # S608: Safe - table name from constructor, quoted via quote_identifier()
        select_query = f"""
            SELECT id, notification_data, retry_count
            FROM {table_quoted}
            WHERE processed_at IS NULL
            ORDER BY created_at
            LIMIT $1
            FOR UPDATE SKIP LOCKED
        """  # noqa: S608

        # UPDATE queries
        # S608: Safe - table name from constructor, quoted via quote_identifier()
        update_success_query = f"""
            UPDATE {table_quoted}
            SET processed_at = NOW()
            WHERE id = $1
        """  # noqa: S608

        update_failure_query = f"""
            UPDATE {table_quoted}
            SET retry_count = retry_count + 1, last_error = $2
            WHERE id = $1
        """  # noqa: S608

        # S608: Safe - table name from constructor, quoted via quote_identifier()
        update_dlq_query = f"""
            UPDATE {table_quoted}
            SET processed_at = NOW(), last_error = $2
            WHERE id = $1
        """  # noqa: S608

        try:
            async with self._pool.acquire() as conn:
                # Wrap in transaction to maintain row locks from SELECT FOR UPDATE
                # Without explicit transaction, locks are released immediately after SELECT
                async with conn.transaction():
                    # Fetch pending notifications
                    rows = await conn.fetch(
                        select_query,
                        self._batch_size,
                        timeout=self._query_timeout,
                    )

                    if not rows:
                        return 0

                    processed = 0

                    for row in rows:
                        row_id: int = row["id"]
                        notification_data = row["notification_data"]
                        row_retry_count: int = row["retry_count"]

                        try:
                            # Parse notification - asyncpg returns dict for JSONB columns
                            if isinstance(notification_data, dict):
                                notification = (
                                    ModelStateTransitionNotification.model_validate(
                                        notification_data
                                    )
                                )
                            else:
                                notification = ModelStateTransitionNotification.model_validate_json(
                                    notification_data
                                )

                            # Check if notification should be moved to DLQ
                            if self._should_move_to_dlq(row_retry_count):
                                dlq_success = await self._move_to_dlq(
                                    row_id=row_id,
                                    notification=notification,
                                    retry_count=row_retry_count,
                                    conn=conn,
                                    update_dlq_query=update_dlq_query,
                                    correlation_id=correlation_id,
                                )
                                if dlq_success:
                                    processed += (
                                        1  # Count as processed since it's been handled
                                    )
                                # Skip normal publishing regardless - DLQ failures will retry
                                continue

                            # Publish notification
                            await self._publisher.publish(notification)

                            # Mark as processed
                            await conn.execute(
                                update_success_query,
                                row_id,
                                timeout=self._query_timeout,
                            )

                            processed += 1
                            self._notifications_processed += 1

                            logger.debug(
                                "Notification published from outbox",
                                extra={
                                    "outbox_id": row_id,
                                    "aggregate_type": notification.aggregate_type,
                                    "aggregate_id": str(notification.aggregate_id),
                                    "correlation_id": str(notification.correlation_id),
                                },
                            )

                        except Exception as e:
                            # Record failure but continue processing other notifications
                            self._notifications_failed += 1
                            error_message = sanitize_error_string(str(e))

                            try:
                                await conn.execute(
                                    update_failure_query,
                                    row_id,
                                    error_message[
                                        : self.MAX_ERROR_MESSAGE_LENGTH
                                    ],  # Truncate for DB column
                                    timeout=self._query_timeout,
                                )
                            except (asyncpg.PostgresError, TimeoutError) as update_err:
                                # Log but continue - the outbox row will be retried
                                logger.warning(
                                    "Failed to record outbox failure, row will be retried",
                                    extra={
                                        "outbox_id": row_id,
                                        "original_error": error_message,
                                        "update_error": sanitize_error_string(
                                            str(update_err)
                                        ),
                                        "update_error_type": type(update_err).__name__,
                                        "correlation_id": str(correlation_id),
                                    },
                                )

                            logger.warning(
                                "Failed to publish notification from outbox",
                                extra={
                                    "outbox_id": row_id,
                                    "error": error_message,
                                    "error_type": type(e).__name__,
                                    "correlation_id": str(correlation_id),
                                },
                            )

                    return processed

        except asyncpg.PostgresConnectionError as e:
            raise InfraConnectionError(
                f"Failed to connect for outbox processing: {self._table_name}",
                context=ctx,
            ) from e

        except asyncpg.QueryCanceledError as e:
            timeout_ctx = ModelTimeoutErrorContext(
                transport_type=EnumInfraTransportType.DATABASE,
                operation="outbox_process_pending",
                target_name=self._table_name,
                correlation_id=correlation_id,
            )
            raise InfraTimeoutError(
                f"Timeout processing outbox: {self._table_name}",
                context=timeout_ctx,
            ) from e

        except Exception as e:
            raise RuntimeHostError(
                f"Failed to process outbox: {type(e).__name__}",
                context=ctx,
            ) from e

    async def start(self) -> None:
        """Start the background processor.

        Starts a background task that continuously processes pending
        notifications from the outbox. The processor polls at the configured
        interval when idle.

        Idempotency:
            Calling start() on an already-running processor is a no-op
            with a warning log.

        Example:
            >>> outbox = TransitionNotificationOutbox(pool, publisher)
            >>> await outbox.start()
            >>> # Processor now running in background
        """
        async with self._lock:
            # Check both _running flag and whether task exists and is not done
            # This prevents starting a second loop if stop() is in progress
            if self._running or (
                self._processor_task is not None and not self._processor_task.done()
            ):
                logger.warning(
                    "Outbox processor already running or task still active, ignoring start()",
                    extra={"table_name": self._table_name},
                )
                return

            self._shutdown_event.clear()
            self._running = True
            self._processor_task = asyncio.create_task(self._processor_loop())

        logger.info(
            "Outbox processor started",
            extra={
                "table_name": self._table_name,
                "batch_size": self._batch_size,
                "poll_interval_seconds": self._poll_interval,
            },
        )

    async def stop(self) -> None:
        """Stop the background processor gracefully.

        Signals the processor to stop and waits for any in-flight processing
        to complete. After stop() returns, no more notifications will be
        processed until start() is called again.

        Idempotency:
            Calling stop() on an already-stopped processor is a no-op.

        Thread Safety:
            The shutdown event is set and processor task captured while holding
            the lock to prevent race conditions with concurrent start() calls.
            The task is awaited outside the lock to avoid deadlock.

        Example:
            >>> await outbox.stop()
            >>> # Processor stopped, safe to shutdown
        """
        # Capture task reference while holding lock to prevent race with start()
        async with self._lock:
            if not self._running:
                logger.debug(
                    "Outbox processor already stopped, ignoring stop()",
                    extra={"table_name": self._table_name},
                )
                return

            self._running = False
            # Signal shutdown INSIDE lock to prevent race with start() clearing it
            self._shutdown_event.set()
            # Capture task reference INSIDE lock before releasing
            processor_task = self._processor_task

        # Wait for processor task to complete OUTSIDE lock to avoid deadlock
        if processor_task is not None:
            try:
                await asyncio.wait_for(processor_task, timeout=self._shutdown_timeout)
            except TimeoutError:
                logger.warning(
                    "Outbox processor did not complete within timeout, cancelling",
                    extra={"table_name": self._table_name},
                )
                processor_task.cancel()
                try:
                    await processor_task
                except asyncio.CancelledError:
                    pass
            except asyncio.CancelledError:
                pass

        # Clear task reference safely - only if it's still the same task
        async with self._lock:
            if self._processor_task is processor_task:
                self._processor_task = None

        logger.info(
            "Outbox processor stopped",
            extra={
                "table_name": self._table_name,
                "notifications_stored": self._notifications_stored,
                "notifications_processed": self._notifications_processed,
                "notifications_failed": self._notifications_failed,
            },
        )

    async def _processor_loop(self) -> None:
        """Background loop that processes pending notifications.

        This method runs continuously until stop() is called, processing
        pending notifications in batches. When no notifications are pending,
        it sleeps for the configured poll interval.

        Error Handling:
            Processing errors are logged but do not crash the loop. The
            loop continues processing after errors to maintain availability.
        """
        logger.debug(
            "Outbox processor loop started",
            extra={"table_name": self._table_name},
        )

        try:
            while not self._shutdown_event.is_set():
                try:
                    # Process pending notifications
                    processed = await self.process_pending()

                    # If no notifications processed, wait before polling again
                    if processed == 0:
                        try:
                            await asyncio.wait_for(
                                self._shutdown_event.wait(),
                                timeout=self._poll_interval,
                            )
                            # Shutdown event was set - exit loop
                            break
                        except TimeoutError:
                            # Poll interval elapsed - continue processing
                            pass

                except Exception as e:
                    # Log error but continue processing
                    logger.exception(
                        "Error in outbox processor loop, continuing",
                        extra={
                            "table_name": self._table_name,
                            "error": sanitize_error_string(str(e)),
                            "error_type": type(e).__name__,
                        },
                    )
                    # Wait before retrying after error
                    try:
                        await asyncio.wait_for(
                            self._shutdown_event.wait(),
                            timeout=self._poll_interval,
                        )
                        break
                    except TimeoutError:
                        pass

        except asyncio.CancelledError:
            logger.info(
                "Outbox processor loop cancelled",
                extra={"table_name": self._table_name},
            )
            raise

        finally:
            logger.debug(
                "Outbox processor loop exiting",
                extra={
                    "table_name": self._table_name,
                    "notifications_processed": self._notifications_processed,
                },
            )

    def _should_move_to_dlq(self, retry_count: int) -> bool:
        """Check if notification should be moved to DLQ.

        Args:
            retry_count: Current retry count for the notification.

        Returns:
            True if the notification should be moved to DLQ, False otherwise.
        """
        if self._max_retries is None or self._dlq_publisher is None:
            return False
        return retry_count >= self._max_retries

    async def _move_to_dlq(
        self,
        row_id: int,
        notification: ModelStateTransitionNotification,
        retry_count: int,
        conn: asyncpg.Connection,
        update_dlq_query: str,
        correlation_id: UUID,
    ) -> bool:
        """Move a notification to the dead letter queue.

        Publishes the notification to the DLQ via the dlq_publisher, marks
        the original record as processed with an error message, and updates
        metrics.

        Args:
            row_id: Database row ID of the notification.
            notification: The parsed notification to move to DLQ.
            retry_count: Current retry count for the notification.
            conn: Database connection for updates.
            update_dlq_query: SQL query to mark notification as processed.
            correlation_id: Correlation ID for logging.

        Returns:
            True if the notification was successfully moved to DLQ, False otherwise.

        Note:
            If DLQ publish fails, the notification is NOT marked as processed
            and will be retried on the next processing cycle. This ensures
            no data loss even if the DLQ is temporarily unavailable. The
            retry_count is NOT incremented on DLQ failure since it already
            exceeds max_retries.

        Warning:
            If the DLQ is **permanently** unavailable, this creates an infinite
            retry loop for notifications exceeding max_retries. Monitor for
            ``processed_at IS NULL AND retry_count >= max_retries`` to detect
            this condition.
        """
        if self._dlq_publisher is None:
            # Should not happen due to _should_move_to_dlq check, but defensive
            return False

        dlq_error_message = f"Moved to DLQ after {retry_count} retries"

        try:
            # Publish to DLQ
            await self._dlq_publisher.publish(notification)

            # Mark as processed with DLQ error message
            await conn.execute(
                update_dlq_query,
                row_id,
                dlq_error_message[: self.MAX_ERROR_MESSAGE_LENGTH],
                timeout=self._query_timeout,
            )

            self._notifications_sent_to_dlq += 1
            self._notifications_processed += 1  # DLQ-handled counts as processed

            logger.warning(
                "Notification moved to DLQ after exceeding max retries",
                extra={
                    "outbox_id": row_id,
                    "aggregate_type": notification.aggregate_type,
                    "aggregate_id": str(notification.aggregate_id),
                    "correlation_id": str(notification.correlation_id),
                    "retry_count": retry_count,
                    "max_retries": self._max_retries,
                    "dlq_topic": self._dlq_topic,
                    "batch_correlation_id": str(correlation_id),
                },
            )

            return True

        except Exception as e:
            # DLQ publish failed - do NOT mark as processed
            # Notification will be retried on next cycle without incrementing retry_count
            # WARNING: If DLQ is permanently unavailable, this creates infinite retries.
            # Monitor: processed_at IS NULL AND retry_count >= max_retries
            self._dlq_publish_failures += 1
            error_message = sanitize_error_string(str(e))
            logger.exception(
                "Failed to publish notification to DLQ, will retry",
                extra={
                    "outbox_id": row_id,
                    "aggregate_type": notification.aggregate_type,
                    "aggregate_id": str(notification.aggregate_id),
                    "correlation_id": str(notification.correlation_id),
                    "retry_count": retry_count,
                    "error": error_message,
                    "error_type": type(e).__name__,
                    "dlq_topic": self._dlq_topic,
                    "batch_correlation_id": str(correlation_id),
                },
            )
            return False

    def get_metrics(self) -> ModelTransitionNotificationOutboxMetrics:
        """Return current outbox metrics for observability.

        Returns:
            Typed metrics model containing:
            - table_name: The outbox table name
            - is_running: Whether processor is running
            - notifications_stored: Total notifications stored
            - notifications_processed: Total notifications successfully processed
            - notifications_failed: Total notifications that failed processing
            - notifications_sent_to_dlq: Total notifications moved to DLQ
            - dlq_publish_failures: Count of failed DLQ publish attempts
            - batch_size: Configured batch size
            - poll_interval_seconds: Configured poll interval
            - max_retries: Max retries before DLQ (None if DLQ disabled)
            - dlq_topic: DLQ topic name (None if DLQ disabled)

        Example:
            >>> metrics = outbox.get_metrics()
            >>> print(f"Processed: {metrics.notifications_processed}")
            >>> print(f"Sent to DLQ: {metrics.notifications_sent_to_dlq}")
            >>> if metrics.dlq_publish_failures > 0:
            ...     print(f"WARNING: {metrics.dlq_publish_failures} DLQ failures")
        """
        return ModelTransitionNotificationOutboxMetrics(
            table_name=self._table_name,
            is_running=self._running,
            notifications_stored=self._notifications_stored,
            notifications_processed=self._notifications_processed,
            notifications_failed=self._notifications_failed,
            notifications_sent_to_dlq=self._notifications_sent_to_dlq,
            dlq_publish_failures=self._dlq_publish_failures,
            batch_size=self._batch_size,
            poll_interval_seconds=self._poll_interval,
            max_retries=self._max_retries,
            dlq_topic=self._dlq_topic,
        )

    async def cleanup_processed(
        self,
        retention_days: int = 7,
    ) -> int:
        """Delete old processed notifications from outbox.

        Removes processed notifications older than the specified retention
        period to prevent table bloat. Should be called periodically via
        cron or scheduled task.

        Args:
            retention_days: Number of days to retain processed records.
                Must be >= 0. Default: 7 days.

        Returns:
            Count of deleted records.

        Raises:
            ProtocolConfigurationError: If retention_days is negative.
            InfraConnectionError: If database connection fails.
            InfraTimeoutError: If query times out.
            RuntimeHostError: For other database errors.

        Example:
            >>> # Delete records processed more than 7 days ago
            >>> deleted = await outbox.cleanup_processed(retention_days=7)
            >>> print(f"Cleaned up {deleted} old records")
            >>>
            >>> # Delete all processed records immediately
            >>> deleted = await outbox.cleanup_processed(retention_days=0)
        """
        correlation_id = UtilUUID.generate_correlation_id()
        ctx = ModelInfraErrorContext(
            transport_type=EnumInfraTransportType.DATABASE,
            operation="outbox_cleanup",
            target_name=self._table_name,
            correlation_id=correlation_id,
        )

        if retention_days < 0:
            raise ProtocolConfigurationError(
                f"retention_days must be >= 0, got {retention_days}",
                context=ctx,
                parameter="retention_days",
                value=retention_days,
            )

        table_quoted = quote_identifier(self._table_name)
        # S608: Safe - table name from constructor, quoted via quote_identifier()
        # retention_days passed as $1 parameter via make_interval()
        query = f"""
            DELETE FROM {table_quoted}
            WHERE processed_at IS NOT NULL
            AND processed_at < NOW() - make_interval(days => $1)
        """  # noqa: S608

        try:
            async with self._pool.acquire() as conn:
                result = await conn.execute(
                    query,
                    retention_days,
                    timeout=self._query_timeout,
                )
                # Parse result like "DELETE 42"
                deleted_count = int(result.split()[-1]) if result else 0

                logger.info(
                    "Cleaned up processed outbox records",
                    extra={
                        "table_name": self._table_name,
                        "retention_days": retention_days,
                        "deleted_count": deleted_count,
                        "correlation_id": str(correlation_id),
                    },
                )

                return deleted_count

        except asyncpg.PostgresConnectionError as e:
            raise InfraConnectionError(
                f"Failed to cleanup outbox: {self._table_name}",
                context=ctx,
            ) from e

        except asyncpg.QueryCanceledError as e:
            timeout_ctx = ModelTimeoutErrorContext(
                transport_type=EnumInfraTransportType.DATABASE,
                operation="outbox_cleanup",
                target_name=self._table_name,
                correlation_id=correlation_id,
            )
            raise InfraTimeoutError(
                f"Timeout cleaning up outbox: {self._table_name}",
                context=timeout_ctx,
            ) from e

        except Exception as e:
            raise RuntimeHostError(
                f"Failed to cleanup outbox: {type(e).__name__}",
                context=ctx,
            ) from e


__all__: list[str] = [
    "TransitionNotificationOutbox",
]
