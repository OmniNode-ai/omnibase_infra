# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""PostgreSQL Writer for Skill Lifecycle Observability (OMN-2934).

Persists skill-started and skill-completed events consumed from Kafka
into the ``skill_executions`` table.

Design Decisions:
    - Pool injection: asyncpg.Pool is injected, not created/managed here.
    - Batch inserts: Uses executemany for efficient batch processing.
    - Idempotency: ON CONFLICT (event_id) DO NOTHING per table contract.
    - Circuit breaker: MixinAsyncCircuitBreaker for resilience.
    - run_id join key: started and completed rows share run_id (partition key).

Idempotency Contract:
    | Table            | Unique Key | Conflict Action |
    |------------------|------------|-----------------|
    | skill_executions | event_id   | DO NOTHING      |

Example:
    >>> import asyncpg
    >>> from omnibase_infra.services.observability.skill_lifecycle.writer_postgres import (
    ...     WriterSkillLifecyclePostgres,
    ... )
    >>>
    >>> pool = await asyncpg.create_pool(dsn="postgresql://...")
    >>> writer = WriterSkillLifecyclePostgres(pool)
    >>>
    >>> count = await writer.write_started(started_events)
    >>> count = await writer.write_completed(completed_events)
"""

from __future__ import annotations

import logging
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

logger = logging.getLogger(__name__)


class WriterSkillLifecyclePostgres(MixinAsyncCircuitBreaker):
    """PostgreSQL writer for skill lifecycle observability events.

    Provides batch write methods for skill-started and skill-completed events
    with idempotency guarantees and circuit breaker resilience.

    The asyncpg.Pool is injected and its lifecycle is managed externally.

    Attributes:
        _pool: Injected asyncpg connection pool.
        DEFAULT_QUERY_TIMEOUT_SECONDS: Default timeout for database queries.

    Example:
        >>> pool = await asyncpg.create_pool(dsn="postgresql://...")
        >>> writer = WriterSkillLifecyclePostgres(pool)
        >>> count = await writer.write_started(started_batch)
    """

    DEFAULT_QUERY_TIMEOUT_SECONDS: float = 30.0

    def __init__(
        self,
        pool: asyncpg.Pool,
        circuit_breaker_threshold: int = 5,
        circuit_breaker_reset_timeout: float = 60.0,
        circuit_breaker_half_open_successes: int = 1,
        query_timeout: float | None = None,
    ) -> None:
        """Initialize the PostgreSQL writer with an injected pool.

        Args:
            pool: asyncpg connection pool (lifecycle managed externally).
            circuit_breaker_threshold: Failures before opening circuit (default: 5).
            circuit_breaker_reset_timeout: Seconds before auto-reset (default: 60.0).
            circuit_breaker_half_open_successes: Successful requests required to close
                circuit from half-open state (default: 1).
            query_timeout: Timeout in seconds for database queries.
        """
        self._pool = pool
        self._query_timeout = query_timeout or self.DEFAULT_QUERY_TIMEOUT_SECONDS

        self._init_circuit_breaker(
            threshold=circuit_breaker_threshold,
            reset_timeout=circuit_breaker_reset_timeout,
            service_name="skill-lifecycle-postgres-writer",
            transport_type=EnumInfraTransportType.DATABASE,
            half_open_successes=circuit_breaker_half_open_successes,
        )

        logger.info(
            "WriterSkillLifecyclePostgres initialized",
            extra={
                "circuit_breaker_threshold": circuit_breaker_threshold,
                "circuit_breaker_reset_timeout": circuit_breaker_reset_timeout,
                "circuit_breaker_half_open_successes": circuit_breaker_half_open_successes,
                "query_timeout": self._query_timeout,
            },
        )

    async def write_started(
        self,
        events: list[dict[str, object]],
        correlation_id: UUID | None = None,
    ) -> int:
        """Write batch of skill-started events to PostgreSQL.

        Uses INSERT ... ON CONFLICT (event_id) DO NOTHING for idempotency.

        Args:
            events: List of parsed skill-started event dicts with keys:
                event_id, run_id, skill_name, skill_id, repo_id,
                correlation_id, args_count, emitted_at, session_id.
            correlation_id: Optional correlation ID for tracing.

        Returns:
            Count of events in the batch.

        Raises:
            InfraConnectionError: If database connection fails.
            InfraTimeoutError: If operation times out.
            InfraUnavailableError: If circuit breaker is open.
        """
        if not events:
            return 0

        op_correlation_id = correlation_id or uuid4()

        async with self._circuit_breaker_lock:
            await self._check_circuit_breaker(
                operation="write_started",
                correlation_id=op_correlation_id,
            )

        context = ModelInfraErrorContext(
            transport_type=EnumInfraTransportType.DATABASE,
            operation="write_started",
            target_name="skill_executions",
            correlation_id=op_correlation_id,
        )

        sql = """
            INSERT INTO skill_executions (
                event_id, run_id, event_type,
                skill_name, skill_id, repo_id,
                correlation_id, args_count, session_id,
                emitted_at
            )
            VALUES ($1, $2, 'started', $3, $4, $5, $6, $7, $8, $9)
            ON CONFLICT (event_id) DO NOTHING
        """

        try:
            async with self._pool.acquire() as conn:
                await conn.executemany(
                    sql,
                    [
                        (
                            e["event_id"],
                            e["run_id"],
                            e["skill_name"],
                            e.get("skill_id"),
                            e["repo_id"],
                            e["correlation_id"],
                            e.get("args_count"),
                            e.get("session_id"),
                            e["emitted_at"],
                        )
                        for e in events
                    ],
                )

            async with self._circuit_breaker_lock:
                await self._reset_circuit_breaker()

            logger.debug(
                "Wrote skill-started batch",
                extra={
                    "count": len(events),
                    "correlation_id": str(op_correlation_id),
                },
            )
            return len(events)

        except asyncpg.QueryCanceledError as e:
            async with self._circuit_breaker_lock:
                await self._record_circuit_failure(
                    operation="write_started",
                    correlation_id=op_correlation_id,
                )
            raise InfraTimeoutError(
                "Write skill-started events timed out",
                context=ModelTimeoutErrorContext(
                    transport_type=context.transport_type,
                    operation=context.operation,
                    target_name=context.target_name,
                    correlation_id=context.correlation_id,
                    timeout_seconds=self._query_timeout,
                ),
            ) from e
        except asyncpg.PostgresConnectionError as e:
            async with self._circuit_breaker_lock:
                await self._record_circuit_failure(
                    operation="write_started",
                    correlation_id=op_correlation_id,
                )
            raise InfraConnectionError(
                "Database connection failed during write_started",
                context=context,
            ) from e
        except asyncpg.PostgresError as e:
            async with self._circuit_breaker_lock:
                await self._record_circuit_failure(
                    operation="write_started",
                    correlation_id=op_correlation_id,
                )
            raise RuntimeHostError(
                f"Database error during write_started: {type(e).__name__}",
                context=context,
            ) from e

    async def write_completed(
        self,
        events: list[dict[str, object]],
        correlation_id: UUID | None = None,
    ) -> int:
        """Write batch of skill-completed events to PostgreSQL.

        Uses INSERT ... ON CONFLICT (event_id) DO NOTHING for idempotency.

        Args:
            events: List of parsed skill-completed event dicts with keys:
                event_id, run_id, skill_name, repo_id, correlation_id,
                status, duration_ms, error_type, started_emit_failed,
                emitted_at, session_id.
            correlation_id: Optional correlation ID for tracing.

        Returns:
            Count of events in the batch.

        Raises:
            InfraConnectionError: If database connection fails.
            InfraTimeoutError: If operation times out.
            InfraUnavailableError: If circuit breaker is open.
        """
        if not events:
            return 0

        op_correlation_id = correlation_id or uuid4()

        async with self._circuit_breaker_lock:
            await self._check_circuit_breaker(
                operation="write_completed",
                correlation_id=op_correlation_id,
            )

        context = ModelInfraErrorContext(
            transport_type=EnumInfraTransportType.DATABASE,
            operation="write_completed",
            target_name="skill_executions",
            correlation_id=op_correlation_id,
        )

        sql = """
            INSERT INTO skill_executions (
                event_id, run_id, event_type,
                skill_name, repo_id,
                correlation_id, status, duration_ms,
                error_type, started_emit_failed,
                session_id, emitted_at
            )
            VALUES ($1, $2, 'completed', $3, $4, $5, $6, $7, $8, $9, $10, $11)
            ON CONFLICT (event_id) DO NOTHING
        """

        try:
            async with self._pool.acquire() as conn:
                await conn.executemany(
                    sql,
                    [
                        (
                            e["event_id"],
                            e["run_id"],
                            e["skill_name"],
                            e["repo_id"],
                            e["correlation_id"],
                            e["status"],
                            e.get("duration_ms"),
                            e.get("error_type"),
                            e.get("started_emit_failed", False),
                            e.get("session_id"),
                            e["emitted_at"],
                        )
                        for e in events
                    ],
                )

            async with self._circuit_breaker_lock:
                await self._reset_circuit_breaker()

            logger.debug(
                "Wrote skill-completed batch",
                extra={
                    "count": len(events),
                    "correlation_id": str(op_correlation_id),
                },
            )
            return len(events)

        except asyncpg.QueryCanceledError as e:
            async with self._circuit_breaker_lock:
                await self._record_circuit_failure(
                    operation="write_completed",
                    correlation_id=op_correlation_id,
                )
            raise InfraTimeoutError(
                "Write skill-completed events timed out",
                context=ModelTimeoutErrorContext(
                    transport_type=context.transport_type,
                    operation=context.operation,
                    target_name=context.target_name,
                    correlation_id=context.correlation_id,
                    timeout_seconds=self._query_timeout,
                ),
            ) from e
        except asyncpg.PostgresConnectionError as e:
            async with self._circuit_breaker_lock:
                await self._record_circuit_failure(
                    operation="write_completed",
                    correlation_id=op_correlation_id,
                )
            raise InfraConnectionError(
                "Database connection failed during write_completed",
                context=context,
            ) from e
        except asyncpg.PostgresError as e:
            async with self._circuit_breaker_lock:
                await self._record_circuit_failure(
                    operation="write_completed",
                    correlation_id=op_correlation_id,
                )
            raise RuntimeHostError(
                f"Database error during write_completed: {type(e).__name__}",
                context=context,
            ) from e

    def get_circuit_breaker_state(self) -> dict[str, JsonType]:
        """Return current circuit breaker state for health checks.

        Returns:
            Dict containing circuit breaker state information.
        """
        return self._get_circuit_breaker_state()


__all__ = ["WriterSkillLifecyclePostgres"]
