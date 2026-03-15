# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

# Copyright (c) 2025 OmniNode Team
# ruff: noqa: S608
# S608 disabled: SQL table name 'delivery_attempts' is a hardcoded constant,
# not user input. No SQL injection risk.
"""RetryWorker service for subscription notification delivery.

A background async service that polls the delivery_attempts table for failed
notifications and re-invokes delivery with exponential backoff. Uses circuit
breaker resilience for database operations and moves exhausted retries to DLQ.

Design Decisions:
    - Pool injection: asyncpg.Pool is injected, not created/managed
    - Idempotent: SELECT FOR UPDATE SKIP LOCKED prevents duplicate processing
    - Circuit breaker: MixinAsyncCircuitBreaker for database resilience
    - Exponential backoff: Configurable base, multiplier, and max cap
    - DLQ escalation: Moves attempts past max retries to DLQ status
    - Graceful shutdown: asyncio.Event-based shutdown signaling
    - Delivery callback: Pluggable delivery function for notification dispatch

Table Schema (delivery_attempts):
    CREATE TABLE IF NOT EXISTS delivery_attempts (
        id UUID PRIMARY KEY,
        subscription_id UUID NOT NULL,
        notification_payload TEXT NOT NULL,
        status VARCHAR(20) NOT NULL DEFAULT 'pending',
        attempt_count INTEGER NOT NULL DEFAULT 0,
        max_attempts INTEGER NOT NULL DEFAULT 5,
        next_retry_at TIMESTAMP WITH TIME ZONE,
        last_error TEXT DEFAULT '',
        created_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT now(),
        updated_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT now()
    );
    CREATE INDEX IF NOT EXISTS idx_delivery_attempts_retry
        ON delivery_attempts(status, next_retry_at)
        WHERE status = 'failed';

Related Tickets:
    - OMN-1454: Implement RetryWorker for subscription notification delivery
    - OMN-1393: HandlerSubscription (records retry schedules)

Example:
    >>> import asyncpg
    >>> from omnibase_infra.services.retry_worker import (
    ...     ConfigRetryWorker,
    ...     ServiceRetryWorker,
    ... )
    >>>
    >>> async def deliver(payload: str) -> None:
    ...     # Custom delivery logic
    ...     await send_notification(payload)
    >>>
    >>> config = ConfigRetryWorker(
    ...     postgres_dsn="postgresql://postgres:secret@localhost:5432/omnibase_infra",
    ... )
    >>> pool = await asyncpg.create_pool(dsn=config.postgres_dsn)
    >>> worker = ServiceRetryWorker(pool=pool, config=config, deliver_fn=deliver)
    >>>
    >>> # Run single poll cycle
    >>> result = await worker.poll_and_retry()
    >>> print(f"Retried {result.retries_attempted}, succeeded {result.retries_succeeded}")
    >>>
    >>> # Or run continuous loop
    >>> await worker.run()
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import Awaitable, Callable
from datetime import UTC, datetime, timedelta
from uuid import UUID, uuid4

import asyncpg

from omnibase_core.types import JsonType
from omnibase_infra.enums import EnumInfraTransportType
from omnibase_infra.errors import (
    InfraConnectionError,
    InfraTimeoutError,
    ModelInfraErrorContext,
    ModelTimeoutErrorContext,
    RuntimeHostError,
)
from omnibase_infra.mixins import MixinAsyncCircuitBreaker
from omnibase_infra.services.retry_worker.config_retry_worker import ConfigRetryWorker
from omnibase_infra.services.retry_worker.models.model_delivery_attempt import (
    EnumDeliveryStatus,
    ModelDeliveryAttempt,
)
from omnibase_infra.services.retry_worker.models.model_retry_result import (
    ModelRetryResult,
)

logger = logging.getLogger(__name__)

# Table name constant - not user input, safe for SQL interpolation.
TABLE_NAME = "delivery_attempts"


class ServiceRetryWorker(MixinAsyncCircuitBreaker):
    """Async background worker for retrying failed notification deliveries.

    Polls the delivery_attempts table for rows where status='failed' and
    next_retry_at <= NOW(), then re-invokes delivery via a pluggable callback.
    Uses exponential backoff for scheduling retries and moves exhausted attempts
    to DLQ status.

    The worker is idempotent: it uses SELECT FOR UPDATE SKIP LOCKED to prevent
    multiple worker instances from processing the same row concurrently.

    Features:
        - Configurable polling interval and batch size
        - Exponential backoff with configurable base, multiplier, and cap
        - Circuit breaker for database resilience
        - DLQ escalation after max retry attempts
        - Pluggable delivery callback
        - Graceful shutdown via asyncio.Event
        - Per-run metrics via ModelRetryResult

    Attributes:
        _pool: Injected asyncpg connection pool.
        _config: Retry worker configuration.
        _deliver_fn: Async callback for notification delivery.
        _shutdown_event: Event for signaling graceful shutdown.
        _last_result: Most recent retry result for health checks.

    Example:
        >>> pool = await asyncpg.create_pool(dsn="postgresql://...")
        >>> config = ConfigRetryWorker(postgres_dsn="postgresql://...")
        >>> worker = ServiceRetryWorker(pool=pool, config=config, deliver_fn=my_deliver)
        >>>
        >>> # One-shot poll
        >>> result = await worker.poll_and_retry()
        >>>
        >>> # Continuous loop
        >>> await worker.run()  # Runs until stop() is called
    """

    def __init__(
        self,
        pool: asyncpg.Pool,
        config: ConfigRetryWorker,
        deliver_fn: Callable[[str], Awaitable[None]],
    ) -> None:
        """Initialize the retry worker.

        Args:
            pool: asyncpg connection pool (lifecycle managed externally).
            config: Retry worker configuration.
            deliver_fn: Async callback invoked to deliver a notification.
                Receives the notification_payload string. Should raise on failure.
        """
        self._pool = pool
        self._config = config
        self._deliver_fn = deliver_fn
        self._shutdown_event = asyncio.Event()
        self._last_result: ModelRetryResult | None = None

        # Initialize circuit breaker mixin
        self._init_circuit_breaker(
            threshold=config.circuit_breaker_threshold,
            reset_timeout=config.circuit_breaker_reset_timeout,
            service_name="retry-worker",
            transport_type=EnumInfraTransportType.DATABASE,
            half_open_successes=config.circuit_breaker_half_open_successes,
        )

        logger.info(
            "ServiceRetryWorker initialized",
            extra={
                "poll_interval_seconds": config.poll_interval_seconds,
                "batch_size": config.batch_size,
                "max_retry_attempts": config.max_retry_attempts,
                "backoff_base_seconds": config.backoff_base_seconds,
                "backoff_multiplier": config.backoff_multiplier,
                "backoff_max_seconds": config.backoff_max_seconds,
                "circuit_breaker_threshold": config.circuit_breaker_threshold,
            },
        )

    # =========================================================================
    # Properties
    # =========================================================================

    @property
    def last_result(self) -> ModelRetryResult | None:
        """Get the most recent retry result for health checks."""
        return self._last_result

    # =========================================================================
    # Backoff Calculation
    # =========================================================================

    def calculate_next_retry_at(self, attempt_count: int) -> datetime:
        """Calculate the next retry time using exponential backoff.

        Uses the formula: delay = min(base * (multiplier ^ attempt_count), max)

        Args:
            attempt_count: The current attempt number (0-indexed).

        Returns:
            Datetime for the next retry attempt.
        """
        delay = self._config.backoff_base_seconds * (
            self._config.backoff_multiplier**attempt_count
        )
        capped_delay = min(delay, self._config.backoff_max_seconds)
        return datetime.now(UTC) + timedelta(seconds=capped_delay)

    # =========================================================================
    # Core Retry Logic
    # =========================================================================

    async def poll_and_retry(
        self,
        correlation_id: UUID | None = None,
    ) -> ModelRetryResult:
        """Run a single poll-and-retry cycle.

        Fetches pending retries from the database, attempts delivery for each,
        and updates records accordingly. Attempts that exceed max retries are
        moved to DLQ status.

        Args:
            correlation_id: Optional correlation ID for tracing.

        Returns:
            ModelRetryResult with retry metrics for this cycle.

        Raises:
            InfraUnavailableError: If circuit breaker is open.
        """
        op_correlation_id = correlation_id or uuid4()
        started_at = datetime.now(UTC)
        start_time = time.monotonic()

        retries_succeeded = 0
        retries_failed = 0
        moved_to_dlq = 0
        errors: list[tuple[str, str]] = []

        logger.info(
            "Starting retry poll cycle",
            extra={
                "correlation_id": str(op_correlation_id),
                "batch_size": self._config.batch_size,
            },
        )

        # Fetch pending retries
        pending = await self._fetch_pending_retries(op_correlation_id)

        for attempt in pending:
            try:
                # Check if max retries exceeded
                if attempt.attempt_count >= attempt.max_attempts:
                    await self._move_to_dlq(attempt, op_correlation_id)
                    moved_to_dlq += 1
                    logger.warning(
                        "Delivery attempt moved to DLQ",
                        extra={
                            "correlation_id": str(op_correlation_id),
                            "attempt_id": str(attempt.id),
                            "attempt_count": attempt.attempt_count,
                            "max_attempts": attempt.max_attempts,
                        },
                    )
                    continue

                # Attempt delivery
                await self._attempt_delivery(attempt, op_correlation_id)
                retries_succeeded += 1

                logger.info(
                    "Retry delivery succeeded",
                    extra={
                        "correlation_id": str(op_correlation_id),
                        "attempt_id": str(attempt.id),
                        "attempt_count": attempt.attempt_count + 1,
                    },
                )

            except Exception as e:  # noqa: BLE001 — boundary: logs warning and degrades
                retries_failed += 1
                error_msg = f"{type(e).__name__}: {e}"
                errors.append((str(attempt.id), error_msg))

                logger.warning(
                    "Retry delivery failed",
                    extra={
                        "correlation_id": str(op_correlation_id),
                        "attempt_id": str(attempt.id),
                        "attempt_count": attempt.attempt_count + 1,
                        "error": error_msg,
                    },
                )

        completed_at = datetime.now(UTC)
        elapsed_ms = int((time.monotonic() - start_time) * 1000)
        retries_attempted = retries_succeeded + retries_failed + moved_to_dlq

        result = ModelRetryResult(
            correlation_id=op_correlation_id,
            started_at=started_at,
            completed_at=completed_at,
            retries_attempted=retries_attempted,
            retries_succeeded=retries_succeeded,
            retries_failed=retries_failed,
            moved_to_dlq=moved_to_dlq,
            duration_ms=elapsed_ms,
            errors=tuple(errors),
        )

        self._last_result = result

        logger.info(
            "Retry poll cycle completed",
            extra={
                "correlation_id": str(op_correlation_id),
                "retries_attempted": retries_attempted,
                "retries_succeeded": retries_succeeded,
                "retries_failed": retries_failed,
                "moved_to_dlq": moved_to_dlq,
                "duration_ms": elapsed_ms,
            },
        )

        return result

    async def _fetch_pending_retries(
        self,
        correlation_id: UUID,
    ) -> list[ModelDeliveryAttempt]:
        """Fetch pending retry attempts from the database.

        Uses SELECT FOR UPDATE SKIP LOCKED for idempotent multi-worker
        processing. Only fetches rows where status='failed' and
        next_retry_at <= NOW().

        Args:
            correlation_id: Correlation ID for tracing.

        Returns:
            List of delivery attempts eligible for retry.

        Raises:
            InfraConnectionError: If database connection fails.
            InfraTimeoutError: If query times out.
            InfraUnavailableError: If circuit breaker is open.
        """
        # Check circuit breaker
        async with self._circuit_breaker_lock:
            await self._check_circuit_breaker(
                operation="fetch_pending_retries",
                correlation_id=correlation_id,
            )

        context = ModelInfraErrorContext(
            transport_type=EnumInfraTransportType.DATABASE,
            operation="fetch_pending_retries",
            target_name="retry-worker",
            correlation_id=correlation_id,
        )

        sql = f"""
            SELECT id, subscription_id, notification_payload, status,
                   attempt_count, max_attempts, next_retry_at, last_error,
                   created_at, updated_at
            FROM {TABLE_NAME}
            WHERE status = $1 AND next_retry_at <= $2
            ORDER BY next_retry_at ASC
            LIMIT $3
            FOR UPDATE SKIP LOCKED
        """

        try:
            async with self._pool.acquire() as conn:
                rows = await conn.fetch(
                    sql,
                    EnumDeliveryStatus.FAILED.value,
                    datetime.now(UTC),
                    self._config.batch_size,
                    timeout=self._config.query_timeout_seconds,
                )

            # Record circuit breaker success
            async with self._circuit_breaker_lock:
                await self._reset_circuit_breaker()

            return [
                ModelDeliveryAttempt(
                    id=row["id"],
                    subscription_id=row["subscription_id"],
                    notification_payload=row["notification_payload"],
                    status=EnumDeliveryStatus(row["status"]),
                    attempt_count=row["attempt_count"],
                    max_attempts=row["max_attempts"],
                    next_retry_at=row["next_retry_at"],
                    last_error=row["last_error"] or "",
                    created_at=row["created_at"],
                    updated_at=row["updated_at"],
                )
                for row in rows
            ]

        except asyncpg.QueryCanceledError as e:
            async with self._circuit_breaker_lock:
                await self._record_circuit_failure(
                    operation="fetch_pending_retries",
                    correlation_id=correlation_id,
                )
            raise InfraTimeoutError(
                "Fetch pending retries timed out",
                context=ModelTimeoutErrorContext(
                    transport_type=context.transport_type,
                    operation=context.operation,
                    target_name=context.target_name,
                    correlation_id=context.correlation_id,
                    timeout_seconds=self._config.query_timeout_seconds,
                ),
            ) from e
        except asyncpg.PostgresConnectionError as e:
            async with self._circuit_breaker_lock:
                await self._record_circuit_failure(
                    operation="fetch_pending_retries",
                    correlation_id=correlation_id,
                )
            raise InfraConnectionError(
                "Database connection failed during fetch_pending_retries",
                context=context,
            ) from e
        except asyncpg.PostgresError as e:
            async with self._circuit_breaker_lock:
                await self._record_circuit_failure(
                    operation="fetch_pending_retries",
                    correlation_id=correlation_id,
                )
            raise RuntimeHostError(
                f"Database error during fetch_pending_retries: {type(e).__name__}",
                context=context,
            ) from e

    async def _attempt_delivery(
        self,
        attempt: ModelDeliveryAttempt,
        correlation_id: UUID,
    ) -> None:
        """Attempt to deliver a notification and update the record.

        Invokes the delivery callback. On success, marks the attempt as
        succeeded. On failure, increments the attempt count and schedules
        the next retry using exponential backoff.

        Args:
            attempt: The delivery attempt to retry.
            correlation_id: Correlation ID for tracing.

        Raises:
            Exception: Re-raises delivery errors after recording failure.
        """
        new_attempt_count = attempt.attempt_count + 1

        try:
            # Invoke delivery with timeout
            await asyncio.wait_for(
                self._deliver_fn(attempt.notification_payload),
                timeout=self._config.delivery_timeout_seconds,
            )

            # Delivery succeeded - update record
            await self._update_attempt_status(
                attempt_id=attempt.id,
                status=EnumDeliveryStatus.SUCCEEDED,
                attempt_count=new_attempt_count,
                last_error="",
                next_retry_at=None,
                correlation_id=correlation_id,
            )

        except Exception as e:
            # Delivery failed - schedule next retry with backoff
            next_retry = self.calculate_next_retry_at(new_attempt_count)
            error_msg = f"{type(e).__name__}: {e}"

            await self._update_attempt_status(
                attempt_id=attempt.id,
                status=EnumDeliveryStatus.FAILED,
                attempt_count=new_attempt_count,
                last_error=error_msg,
                next_retry_at=next_retry,
                correlation_id=correlation_id,
            )

            raise

    async def _move_to_dlq(
        self,
        attempt: ModelDeliveryAttempt,
        correlation_id: UUID,
    ) -> None:
        """Move a delivery attempt to dead letter queue status.

        Called when max retry attempts have been exceeded. Updates the
        status to DLQ and clears the next_retry_at.

        Args:
            attempt: The delivery attempt to move to DLQ.
            correlation_id: Correlation ID for tracing.
        """
        await self._update_attempt_status(
            attempt_id=attempt.id,
            status=EnumDeliveryStatus.DLQ,
            attempt_count=attempt.attempt_count,
            last_error=attempt.last_error,
            next_retry_at=None,
            correlation_id=correlation_id,
        )

    async def _update_attempt_status(
        self,
        attempt_id: UUID,
        status: EnumDeliveryStatus,
        attempt_count: int,
        last_error: str,
        next_retry_at: datetime | None,
        correlation_id: UUID,
    ) -> None:
        """Update a delivery attempt record in the database.

        Args:
            attempt_id: ID of the delivery attempt to update.
            status: New status value.
            attempt_count: Updated attempt count.
            last_error: Error message from latest attempt (empty on success).
            next_retry_at: Next scheduled retry time (None for terminal states).
            correlation_id: Correlation ID for tracing.

        Raises:
            InfraConnectionError: If database connection fails.
            InfraTimeoutError: If update times out.
        """
        context = ModelInfraErrorContext(
            transport_type=EnumInfraTransportType.DATABASE,
            operation="update_attempt_status",
            target_name="retry-worker",
            correlation_id=correlation_id,
        )

        sql = f"""
            UPDATE {TABLE_NAME}
            SET status = $1,
                attempt_count = $2,
                last_error = $3,
                next_retry_at = $4,
                updated_at = $5
            WHERE id = $6
        """

        try:
            async with self._pool.acquire() as conn:
                await conn.execute(
                    sql,
                    status.value,
                    attempt_count,
                    last_error,
                    next_retry_at,
                    datetime.now(UTC),
                    attempt_id,
                    timeout=self._config.query_timeout_seconds,
                )

            # Record circuit breaker success
            async with self._circuit_breaker_lock:
                await self._reset_circuit_breaker()

        except asyncpg.QueryCanceledError as e:
            async with self._circuit_breaker_lock:
                await self._record_circuit_failure(
                    operation="update_attempt_status",
                    correlation_id=correlation_id,
                )
            raise InfraTimeoutError(
                f"Update attempt status timed out for {attempt_id}",
                context=ModelTimeoutErrorContext(
                    transport_type=context.transport_type,
                    operation=context.operation,
                    target_name=context.target_name,
                    correlation_id=context.correlation_id,
                    timeout_seconds=self._config.query_timeout_seconds,
                ),
            ) from e
        except asyncpg.PostgresConnectionError as e:
            async with self._circuit_breaker_lock:
                await self._record_circuit_failure(
                    operation="update_attempt_status",
                    correlation_id=correlation_id,
                )
            raise InfraConnectionError(
                f"Database connection failed during update for {attempt_id}",
                context=context,
            ) from e
        except asyncpg.PostgresError as e:
            async with self._circuit_breaker_lock:
                await self._record_circuit_failure(
                    operation="update_attempt_status",
                    correlation_id=correlation_id,
                )
            raise RuntimeHostError(
                f"Database error during update for {attempt_id}: {type(e).__name__}",
                context=context,
            ) from e

    # =========================================================================
    # Continuous Loop
    # =========================================================================

    async def run(self) -> None:
        """Run the continuous retry worker loop.

        Executes poll_and_retry() at the configured interval until stop()
        is called. Each cycle is independent - failures in one cycle do not
        prevent the next cycle.

        Example:
            >>> worker = ServiceRetryWorker(pool=pool, config=config, deliver_fn=fn)
            >>> await worker.run()  # Runs until stop() is called
        """
        logger.info(
            "Starting retry worker loop",
            extra={
                "poll_interval_seconds": self._config.poll_interval_seconds,
                "batch_size": self._config.batch_size,
            },
        )

        while not self._shutdown_event.is_set():
            try:
                await self.poll_and_retry()
            except Exception:
                logger.exception(
                    "Retry worker poll cycle failed",
                    extra={
                        "poll_interval_seconds": self._config.poll_interval_seconds,
                    },
                )

            # Wait for the configured interval or shutdown signal
            try:
                await asyncio.wait_for(
                    self._shutdown_event.wait(),
                    timeout=self._config.poll_interval_seconds,
                )
                # If we get here, shutdown was signaled
                break
            except TimeoutError:
                # Normal timeout - continue to next poll
                continue

        logger.info("Retry worker loop stopped")

    def stop(self) -> None:
        """Signal the worker loop to stop.

        Non-blocking. The loop will complete its current sleep/poll
        cycle and then exit.

        Example:
            >>> worker.stop()
            >>> # Loop will exit after current interval
        """
        self._shutdown_event.set()
        logger.info("Retry worker shutdown signaled")

    # =========================================================================
    # Health Check
    # =========================================================================

    def get_health_status(self) -> dict[str, JsonType]:
        """Return health status for monitoring.

        Returns:
            Dictionary with health status including circuit breaker state,
            last retry result, and configuration.
        """
        circuit_state = self._get_circuit_breaker_state()

        last_result_info: dict[str, JsonType] | None = None
        if self._last_result is not None:
            errors_json: dict[str, JsonType] | None = (
                dict(self._last_result.errors) if self._last_result.errors else None
            )
            last_result_info = {
                "correlation_id": str(self._last_result.correlation_id),
                "completed_at": self._last_result.completed_at.isoformat(),
                "retries_attempted": self._last_result.retries_attempted,
                "retries_succeeded": self._last_result.retries_succeeded,
                "retries_failed": self._last_result.retries_failed,
                "moved_to_dlq": self._last_result.moved_to_dlq,
                "duration_ms": self._last_result.duration_ms,
                "errors": errors_json,
            }

        return {
            "service": "retry-worker",
            "circuit_breaker": circuit_state,
            "last_result": last_result_info,
            "config": {
                "poll_interval_seconds": self._config.poll_interval_seconds,
                "batch_size": self._config.batch_size,
                "max_retry_attempts": self._config.max_retry_attempts,
                "backoff_base_seconds": self._config.backoff_base_seconds,
                "backoff_multiplier": self._config.backoff_multiplier,
                "backoff_max_seconds": self._config.backoff_max_seconds,
            },
        }


__all__ = ["ServiceRetryWorker", "TABLE_NAME"]
