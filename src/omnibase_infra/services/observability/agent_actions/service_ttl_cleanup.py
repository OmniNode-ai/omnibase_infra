# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""TTL Cleanup Service for Observability Tables.

This module provides an async service that periodically deletes rows older
than the configured retention period from observability tables. It uses
batched DELETEs to avoid lock contention and circuit breaker resilience
for database operations.

Design Decisions:
    - Pool injection: asyncpg.Pool is injected, not created/managed
    - Batch deletes: Uses DELETE ... WHERE ... LIMIT for efficient batch processing
    - Per-table TTL columns: created_at for most tables, updated_at for execution logs
    - Circuit breaker: MixinAsyncCircuitBreaker for database resilience
    - Metrics: Returns ModelTTLCleanupResult with per-table deletion counts
    - Graceful shutdown: asyncio.Event-based shutdown signaling

Table-to-TTL-column mapping:
    | Table                         | TTL Column   | Reason                          |
    |-------------------------------|--------------|---------------------------------|
    | agent_actions                 | created_at   | Append-only, no updates         |
    | agent_routing_decisions       | created_at   | Append-only, no updates         |
    | agent_transformation_events   | created_at   | Append-only, no updates         |
    | router_performance_metrics    | created_at   | Append-only, no updates         |
    | agent_detection_failures      | created_at   | Append-only, no updates         |
    | agent_execution_logs          | updated_at   | Lifecycle records, avoid mid-flight deletion |
    | agent_status_events           | created_at   | Append-only, no updates         |

Related Tickets:
    - OMN-1759: Implement 30-day TTL cleanup for observability tables
    - OMN-1743: Created the observability tables (Phase 1)

Example:
    >>> import asyncpg
    >>> from omnibase_infra.services.observability.agent_actions.service_ttl_cleanup import (
    ...     ServiceTTLCleanup,
    ... )
    >>> from omnibase_infra.services.observability.agent_actions.config_ttl_cleanup import (
    ...     ConfigTTLCleanup,
    ... )
    >>>
    >>> config = ConfigTTLCleanup(
    ...     postgres_dsn="postgresql://postgres:secret@localhost:5432/omninode_bridge",
    ...     retention_days=30,
    ... )
    >>> pool = await asyncpg.create_pool(dsn=config.postgres_dsn)
    >>> service = ServiceTTLCleanup(pool=pool, config=config)
    >>>
    >>> # Run single cleanup
    >>> result = await service.cleanup_once()
    >>> print(f"Deleted {result.total_rows_deleted} rows")
    >>>
    >>> # Or run continuous loop
    >>> await service.run()
"""

from __future__ import annotations

import asyncio
import logging
import time
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
from omnibase_infra.services.observability.agent_actions.config_ttl_cleanup import (
    ConfigTTLCleanup,
)
from omnibase_infra.services.observability.agent_actions.models.model_ttl_cleanup_result import (
    ModelTTLCleanupResult,
)

logger = logging.getLogger(__name__)

# Allowlist of table names that may be cleaned. Prevents SQL injection
# if table_ttl_columns is populated from external configuration.
ALLOWED_TABLES: frozenset[str] = frozenset(
    {
        "agent_actions",
        "agent_routing_decisions",
        "agent_transformation_events",
        "router_performance_metrics",
        "agent_detection_failures",
        "agent_execution_logs",
        "agent_status_events",
    }
)

# Allowlist of column names for TTL cleanup.
ALLOWED_TTL_COLUMNS: frozenset[str] = frozenset({"created_at", "updated_at"})


class ServiceTTLCleanup(MixinAsyncCircuitBreaker):
    """Async service for TTL cleanup of observability tables.

    Periodically deletes rows older than the configured retention period
    from each observability table. Uses batched DELETEs to avoid lock
    contention and circuit breaker resilience for database operations.

    The service can be run in two modes:
    1. Single cleanup: Call ``cleanup_once()`` for a one-shot cleanup
    2. Continuous loop: Call ``run()`` for periodic cleanup on an interval

    Features:
        - Batched DELETE queries to minimize lock contention
        - Per-table TTL column support (created_at vs updated_at)
        - Circuit breaker for database resilience
        - Detailed per-table metrics via ModelTTLCleanupResult
        - Graceful shutdown via asyncio.Event
        - Table and column name allowlisting for SQL injection prevention

    Attributes:
        _pool: Injected asyncpg connection pool.
        _config: TTL cleanup configuration.
        _shutdown_event: Event for signaling graceful shutdown.
        _last_result: Most recent cleanup result for health checks.

    Example:
        >>> pool = await asyncpg.create_pool(dsn="postgresql://...")
        >>> config = ConfigTTLCleanup(postgres_dsn="postgresql://...", retention_days=30)
        >>> service = ServiceTTLCleanup(pool=pool, config=config)
        >>>
        >>> # One-shot cleanup
        >>> result = await service.cleanup_once()
        >>> print(f"Deleted {result.total_rows_deleted} rows in {result.duration_ms}ms")
        >>>
        >>> # Continuous loop
        >>> await service.run()  # Runs until stop() is called
    """

    def __init__(
        self,
        pool: asyncpg.Pool,
        config: ConfigTTLCleanup,
    ) -> None:
        """Initialize the TTL cleanup service.

        Args:
            pool: asyncpg connection pool (lifecycle managed externally).
            config: TTL cleanup configuration.

        Raises:
            ProtocolConfigurationError: If config contains invalid table names.
        """
        self._pool = pool
        self._config = config
        self._shutdown_event = asyncio.Event()
        self._last_result: ModelTTLCleanupResult | None = None

        # Validate table names against allowlist
        invalid_tables = set(config.table_ttl_columns.keys()) - ALLOWED_TABLES
        if invalid_tables:
            from omnibase_infra.errors import ProtocolConfigurationError

            raise ProtocolConfigurationError(
                f"Invalid table names in TTL cleanup config: {invalid_tables}. "
                f"Allowed tables: {ALLOWED_TABLES}"
            )

        # Initialize circuit breaker mixin
        self._init_circuit_breaker(
            threshold=config.circuit_breaker_threshold,
            reset_timeout=config.circuit_breaker_reset_timeout,
            service_name="ttl-cleanup-service",
            transport_type=EnumInfraTransportType.DATABASE,
            half_open_successes=config.circuit_breaker_half_open_successes,
        )

        logger.info(
            "ServiceTTLCleanup initialized",
            extra={
                "retention_days": config.retention_days,
                "batch_size": config.batch_size,
                "interval_seconds": config.interval_seconds,
                "tables": list(config.table_ttl_columns.keys()),
                "circuit_breaker_threshold": config.circuit_breaker_threshold,
            },
        )

    # =========================================================================
    # Properties
    # =========================================================================

    @property
    def last_result(self) -> ModelTTLCleanupResult | None:
        """Get the most recent cleanup result for health checks.

        Returns:
            Most recent ModelTTLCleanupResult, or None if no cleanup has run.
        """
        return self._last_result

    # =========================================================================
    # Core Cleanup Logic
    # =========================================================================

    async def cleanup_once(
        self,
        correlation_id: UUID | None = None,
    ) -> ModelTTLCleanupResult:
        """Run a single cleanup pass across all configured tables.

        Iterates over each table, deleting rows older than the retention
        period in batches. Collects per-table metrics and returns a
        comprehensive result.

        Args:
            correlation_id: Optional correlation ID for tracing.

        Returns:
            ModelTTLCleanupResult with per-table deletion counts and timing.

        Raises:
            InfraUnavailableError: If circuit breaker is open.
        """
        op_correlation_id = correlation_id or uuid4()
        started_at = datetime.now(UTC)
        start_time = time.monotonic()

        tables_cleaned: dict[str, int] = {}
        errors: dict[str, str] = {}
        cutoff = datetime.now(UTC) - timedelta(days=self._config.retention_days)

        logger.info(
            "Starting TTL cleanup run",
            extra={
                "correlation_id": str(op_correlation_id),
                "cutoff": cutoff.isoformat(),
                "retention_days": self._config.retention_days,
                "tables": list(self._config.table_ttl_columns.keys()),
            },
        )

        for table_name, ttl_column in self._config.table_ttl_columns.items():
            try:
                deleted = await self._cleanup_table(
                    table_name=table_name,
                    ttl_column=ttl_column,
                    cutoff=cutoff,
                    correlation_id=op_correlation_id,
                )
                tables_cleaned[table_name] = deleted

                if deleted > 0:
                    logger.info(
                        "Cleaned table",
                        extra={
                            "correlation_id": str(op_correlation_id),
                            "table": table_name,
                            "rows_deleted": deleted,
                            "cutoff": cutoff.isoformat(),
                        },
                    )

            except Exception as e:
                error_msg = f"{type(e).__name__}: {e}"
                errors[table_name] = error_msg
                tables_cleaned[table_name] = 0

                logger.warning(
                    "Failed to clean table",
                    extra={
                        "correlation_id": str(op_correlation_id),
                        "table": table_name,
                        "error": error_msg,
                    },
                )

        completed_at = datetime.now(UTC)
        elapsed_ms = int((time.monotonic() - start_time) * 1000)
        total_deleted = sum(tables_cleaned.values())

        result = ModelTTLCleanupResult(
            correlation_id=op_correlation_id,
            started_at=started_at,
            completed_at=completed_at,
            tables_cleaned=tables_cleaned,
            total_rows_deleted=total_deleted,
            duration_ms=elapsed_ms,
            errors=errors,
        )

        self._last_result = result

        logger.info(
            "TTL cleanup run completed",
            extra={
                "correlation_id": str(op_correlation_id),
                "total_rows_deleted": total_deleted,
                "duration_ms": elapsed_ms,
                "tables_cleaned": tables_cleaned,
                "errors": errors if errors else None,
            },
        )

        return result

    async def _cleanup_table(
        self,
        table_name: str,
        ttl_column: str,
        cutoff: datetime,
        correlation_id: UUID,
    ) -> int:
        """Delete expired rows from a single table in batches.

        Uses batched DELETE with LIMIT to avoid long-running locks.
        Repeats until fewer rows than batch_size are deleted (table clean).

        Args:
            table_name: Name of the table to clean (must be in ALLOWED_TABLES).
            ttl_column: Column name for TTL comparison (created_at or updated_at).
            cutoff: Timestamp cutoff - rows older than this are deleted.
            correlation_id: Correlation ID for tracing.

        Returns:
            Total number of rows deleted from this table.

        Raises:
            InfraConnectionError: If database connection fails.
            InfraTimeoutError: If operation times out.
            InfraUnavailableError: If circuit breaker is open.
        """
        # Validate table and column names against allowlists (defense in depth)
        if table_name not in ALLOWED_TABLES:
            raise ValueError(f"Table {table_name} not in allowed tables")
        if ttl_column not in ALLOWED_TTL_COLUMNS:
            raise ValueError(f"Column {ttl_column} not in allowed TTL columns")

        # Check circuit breaker
        async with self._circuit_breaker_lock:
            await self._check_circuit_breaker(
                operation=f"cleanup_{table_name}",
                correlation_id=correlation_id,
            )

        context = ModelInfraErrorContext(
            transport_type=EnumInfraTransportType.DATABASE,
            operation=f"ttl_cleanup_{table_name}",
            target_name=table_name,
            correlation_id=correlation_id,
        )

        # Use a CTE-based batched delete: DELETE rows matching a subquery
        # with LIMIT. This avoids holding locks on the entire table.
        # The table and column names are validated against allowlists above,
        # so string interpolation here is safe from SQL injection.
        sql = f"""
            DELETE FROM {table_name}
            WHERE ctid IN (
                SELECT ctid FROM {table_name}
                WHERE {ttl_column} < $1
                LIMIT $2
            )
        """  # noqa: S608 - table/column validated against allowlists

        total_deleted = 0

        try:
            while True:
                async with self._pool.acquire() as conn:
                    result = await conn.execute(
                        sql,
                        cutoff,
                        self._config.batch_size,
                        timeout=self._config.query_timeout_seconds,
                    )

                # Parse "DELETE N" result string
                batch_deleted = int(result.split()[-1])
                total_deleted += batch_deleted

                logger.debug(
                    "Batch delete completed",
                    extra={
                        "correlation_id": str(correlation_id),
                        "table": table_name,
                        "batch_deleted": batch_deleted,
                        "total_deleted": total_deleted,
                    },
                )

                # If we deleted fewer than batch_size, the table is clean
                if batch_deleted < self._config.batch_size:
                    break

            # Record success
            async with self._circuit_breaker_lock:
                await self._reset_circuit_breaker()

            return total_deleted

        except asyncpg.QueryCanceledError as e:
            async with self._circuit_breaker_lock:
                await self._record_circuit_failure(
                    operation=f"cleanup_{table_name}",
                    correlation_id=correlation_id,
                )
            raise InfraTimeoutError(
                f"TTL cleanup of {table_name} timed out",
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
                    operation=f"cleanup_{table_name}",
                    correlation_id=correlation_id,
                )
            raise InfraConnectionError(
                f"Database connection failed during TTL cleanup of {table_name}",
                context=context,
            ) from e
        except asyncpg.PostgresError as e:
            async with self._circuit_breaker_lock:
                await self._record_circuit_failure(
                    operation=f"cleanup_{table_name}",
                    correlation_id=correlation_id,
                )
            raise RuntimeHostError(
                f"Database error during TTL cleanup of {table_name}: {type(e).__name__}",
                context=context,
            ) from e

    # =========================================================================
    # Continuous Loop
    # =========================================================================

    async def run(self) -> None:
        """Run the continuous TTL cleanup loop.

        Executes cleanup_once() at the configured interval until stop()
        is called. Each run is independent - failures in one run do not
        prevent the next run.

        Example:
            >>> service = ServiceTTLCleanup(pool=pool, config=config)
            >>> # Runs until stop() is called
            >>> await service.run()
        """
        logger.info(
            "Starting TTL cleanup loop",
            extra={
                "interval_seconds": self._config.interval_seconds,
                "retention_days": self._config.retention_days,
            },
        )

        while not self._shutdown_event.is_set():
            try:
                await self.cleanup_once()
            except Exception:
                logger.exception(
                    "TTL cleanup run failed",
                    extra={
                        "interval_seconds": self._config.interval_seconds,
                    },
                )

            # Wait for the configured interval or shutdown signal
            try:
                await asyncio.wait_for(
                    self._shutdown_event.wait(),
                    timeout=self._config.interval_seconds,
                )
                # If we get here, shutdown was signaled
                break
            except TimeoutError:
                # Normal timeout - continue to next cleanup
                continue

        logger.info("TTL cleanup loop stopped")

    def stop(self) -> None:
        """Signal the cleanup loop to stop.

        Non-blocking. The loop will complete its current sleep/cleanup
        cycle and then exit.

        Example:
            >>> service.stop()
            >>> # Loop will exit after current interval
        """
        self._shutdown_event.set()
        logger.info("TTL cleanup shutdown signaled")

    # =========================================================================
    # Health Check
    # =========================================================================

    def get_health_status(self) -> dict[str, JsonType]:
        """Return health status for monitoring.

        Returns:
            Dictionary with health status including circuit breaker state,
            last cleanup result, and configuration.
        """
        circuit_state = self._get_circuit_breaker_state()

        last_result_info: dict[str, JsonType] | None = None
        if self._last_result is not None:
            # Widen dict[str, str] to dict[str, JsonType] for type compatibility.
            # dict is invariant in its value type, so explicit construction is needed.
            errors_json: dict[str, JsonType] | None = (
                dict(self._last_result.errors) if self._last_result.errors else None
            )
            last_result_info = {
                "correlation_id": str(self._last_result.correlation_id),
                "completed_at": self._last_result.completed_at.isoformat(),
                "total_rows_deleted": self._last_result.total_rows_deleted,
                "duration_ms": self._last_result.duration_ms,
                "errors": errors_json,
            }

        return {
            "service": "ttl-cleanup",
            "circuit_breaker": circuit_state,
            "last_result": last_result_info,
            "config": {
                "retention_days": self._config.retention_days,
                "batch_size": self._config.batch_size,
                "interval_seconds": self._config.interval_seconds,
                "tables": list(self._config.table_ttl_columns.keys()),
            },
        }


__all__ = [
    "ALLOWED_TABLES",
    "ALLOWED_TTL_COLUMNS",
    "ServiceTTLCleanup",
]
