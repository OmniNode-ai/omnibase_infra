# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Transition Notification Outbox for guaranteed delivery.

This module implements the outbox pattern for state transition notifications.
The outbox stores notifications in the same database transaction as projections,
then processes them asynchronously via a background processor to ensure
at-least-once delivery semantics. Consumers must handle idempotency.

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
from uuid import UUID, uuid4

import asyncpg

# Use core model and protocol
from omnibase_core.models.notifications import ModelStateTransitionNotification
from omnibase_core.protocols.notifications import (
    ProtocolTransitionNotificationPublisher,
)
from omnibase_infra.enums import EnumInfraTransportType
from omnibase_infra.errors import (
    InfraConnectionError,
    InfraTimeoutError,
    ModelInfraErrorContext,
    ModelTimeoutErrorContext,
    ProtocolConfigurationError,
    RuntimeHostError,
)
from omnibase_infra.runtime.models.model_transition_notification_outbox_metrics import (
    ModelTransitionNotificationOutboxMetrics,
)
from omnibase_infra.utils.util_error_sanitization import sanitize_error_string

logger = logging.getLogger(__name__)


class TransitionNotificationOutbox:
    """Outbox pattern for guaranteed notification delivery.

    Stores notifications in the same database transaction as projections,
    ensuring at-least-once semantics. A background processor publishes
    pending notifications asynchronously. Consumers must handle idempotency.

    The outbox pattern solves the dual-write problem: when you need to
    update a database AND publish an event, either operation could fail
    independently, leading to inconsistent state. By writing the event
    to an outbox table in the same transaction as the data change, we
    guarantee atomicity. A separate process then reads from the outbox
    and publishes events.

    Attributes:
        table_name: Name of the outbox table (default: "transition_notification_outbox")
        batch_size: Number of notifications to process per batch (default: 100)
        poll_interval: Seconds between processing polls when idle (default: 1.0)
        is_running: Whether the background processor is running

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

        Raises:
            ProtocolConfigurationError: If pool or publisher is None, or if
                configuration values are invalid.
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

        self._pool = pool
        self._publisher = publisher
        self._table_name = table_name
        self._batch_size = batch_size
        self._poll_interval = poll_interval_seconds
        self._query_timeout = query_timeout_seconds
        self._strict_transaction_mode = strict_transaction_mode

        # State management
        self._running = False
        self._lock = asyncio.Lock()
        self._shutdown_event = asyncio.Event()
        self._processor_task: asyncio.Task[None] | None = None

        # Metrics tracking
        self._notifications_stored: int = 0
        self._notifications_processed: int = 0
        self._notifications_failed: int = 0

        logger.debug(
            "TransitionNotificationOutbox initialized",
            extra={
                "table_name": table_name,
                "batch_size": batch_size,
                "poll_interval_seconds": poll_interval_seconds,
                "strict_transaction_mode": strict_transaction_mode,
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

        # Build INSERT query - table name from trusted config
        # S608: Safe - table name from constructor, not user input
        query = f"""
            INSERT INTO {self._table_name}
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
        correlation_id = uuid4()
        ctx = ModelInfraErrorContext(
            transport_type=EnumInfraTransportType.DATABASE,
            operation="outbox_process_pending",
            target_name=self._table_name,
            correlation_id=correlation_id,
        )

        # Build SELECT query with FOR UPDATE SKIP LOCKED for concurrent safety
        # S608: Safe - table name from constructor, not user input
        select_query = f"""
            SELECT id, notification_data
            FROM {self._table_name}
            WHERE processed_at IS NULL
            ORDER BY created_at
            LIMIT $1
            FOR UPDATE SKIP LOCKED
        """  # noqa: S608

        # Build UPDATE queries
        # S608: Safe - table name from constructor, not user input
        update_success_query = f"""
            UPDATE {self._table_name}
            SET processed_at = NOW()
            WHERE id = $1
        """  # noqa: S608

        update_failure_query = f"""
            UPDATE {self._table_name}
            SET retry_count = retry_count + 1, last_error = $2
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
            if self._running:
                logger.warning(
                    "Outbox processor already running, ignoring start()",
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

        Example:
            >>> await outbox.stop()
            >>> # Processor stopped, safe to shutdown
        """
        async with self._lock:
            if not self._running:
                logger.debug(
                    "Outbox processor already stopped, ignoring stop()",
                    extra={"table_name": self._table_name},
                )
                return

            self._running = False

        # Signal shutdown to processor loop
        self._shutdown_event.set()

        # Wait for processor task to complete
        if self._processor_task is not None:
            try:
                await asyncio.wait_for(self._processor_task, timeout=10.0)
            except TimeoutError:
                logger.warning(
                    "Outbox processor did not complete within timeout, cancelling",
                    extra={"table_name": self._table_name},
                )
                self._processor_task.cancel()
                try:
                    await self._processor_task
                except asyncio.CancelledError:
                    pass
            except asyncio.CancelledError:
                pass
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

    def get_metrics(self) -> ModelTransitionNotificationOutboxMetrics:
        """Return current outbox metrics for observability.

        Returns:
            Typed metrics model containing:
            - table_name: The outbox table name
            - is_running: Whether processor is running
            - notifications_stored: Total notifications stored
            - notifications_processed: Total notifications successfully processed
            - notifications_failed: Total notifications that failed processing
            - batch_size: Configured batch size
            - poll_interval_seconds: Configured poll interval

        Example:
            >>> metrics = outbox.get_metrics()
            >>> print(f"Processed: {metrics.notifications_processed}")
        """
        return ModelTransitionNotificationOutboxMetrics(
            table_name=self._table_name,
            is_running=self._running,
            notifications_stored=self._notifications_stored,
            notifications_processed=self._notifications_processed,
            notifications_failed=self._notifications_failed,
            batch_size=self._batch_size,
            poll_interval_seconds=self._poll_interval,
        )


__all__: list[str] = [
    "TransitionNotificationOutbox",
]
