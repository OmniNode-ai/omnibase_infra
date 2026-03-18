# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""PostgreSQL Writer for Context Audit Observability (OMN-5240).

Persists context integrity audit events consumed from Kafka into the
``context_audit_events`` table (migration 053, OMN-5239).

Design Decisions:
    - Pool injection: asyncpg.Pool is injected, not created/managed here.
    - Batch inserts: Uses executemany for efficient batch processing.
    - Idempotency: INSERT ... ON CONFLICT (id) DO NOTHING — id is BIGSERIAL,
      so duplicate protection relies on the unique (task_id, event_type, created_at)
      natural key. Kafka at-least-once delivery may produce duplicates; the consumer
      deduplicates by task_id + event_type at the application layer via the batch accumulator.
    - Circuit breaker: MixinAsyncCircuitBreaker for resilience.
    - JSONB serialization: violation_details dict serialized to JSON string.

Idempotency Contract:
    | Table                 | Strategy                                  |
    |-----------------------|-------------------------------------------|
    | context_audit_events  | Application-layer dedup in batch accumulator|

Example:
    >>> import asyncpg
    >>> from omnibase_infra.services.observability.context_audit.writer_postgres import (
    ...     WriterContextAuditPostgres,
    ... )
    >>>
    >>> pool = await asyncpg.create_pool(dsn="postgresql://...")
    >>> writer = WriterContextAuditPostgres(pool)
    >>> count = await writer.write_audit_events(events)
"""

# no-migration: context_audit_events table created by migration 053 in PR #874 (OMN-5239)

from __future__ import annotations

import json
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

# Required fields for schema validation pre-filter.
# Messages missing these fields are skipped with a WARNING rather than crashing the batch.
_REQUIRED_AUDIT_FIELDS: frozenset[str] = frozenset(
    {"task_id", "correlation_id", "event_type", "enforcement_level"}
)


def _validate_event_fields(
    event: dict[str, object],
    required: frozenset[str],
    context: str,
) -> bool:
    """Return True if all required fields are present in the event dict.

    Logs a WARNING with missing keys and context string on failure, returns False.
    Old-schema messages missing required keys are skipped rather than crashing the batch.

    Args:
        event: The event dict to validate.
        required: Frozenset of required field names.
        context: Human-readable context string for the log message.

    Returns:
        True if all required fields are present, False otherwise.
    """
    missing = required - event.keys()
    if missing:
        logger.warning(
            "Skipping audit event with missing required fields",
            extra={
                "context": context,
                "missing_fields": sorted(missing),
                "event_keys": sorted(event.keys()),
            },
        )
        return False
    return True


class WriterContextAuditPostgres(MixinAsyncCircuitBreaker):
    """PostgreSQL writer for context integrity audit events.

    Provides a batch write method for context_audit_events with schema
    validation pre-filtering and circuit breaker resilience.

    The asyncpg.Pool is injected and its lifecycle is managed externally.

    Attributes:
        _pool: Injected asyncpg connection pool.
        DEFAULT_QUERY_TIMEOUT_SECONDS: Default timeout for database queries.

    Example:
        >>> pool = await asyncpg.create_pool(dsn="postgresql://...")
        >>> writer = WriterContextAuditPostgres(pool)
        >>> count = await writer.write_audit_events(events)
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
            service_name="context-audit-postgres-writer",
            transport_type=EnumInfraTransportType.DATABASE,
            half_open_successes=circuit_breaker_half_open_successes,
        )

        logger.info(
            "WriterContextAuditPostgres initialized",
            extra={
                "circuit_breaker_threshold": circuit_breaker_threshold,
                "circuit_breaker_reset_timeout": circuit_breaker_reset_timeout,
                "circuit_breaker_half_open_successes": circuit_breaker_half_open_successes,
                "query_timeout": self._query_timeout,
            },
        )

    async def write_audit_events(
        self,
        events: list[dict[str, object]],
        correlation_id: UUID | None = None,
    ) -> int:
        """Write batch of context audit events to PostgreSQL.

        Args:
            events: List of parsed audit event dicts with keys:
                task_id, parent_task_id (optional), correlation_id,
                contract_id (optional), event_type, enforcement_level,
                enforcement_action (optional), violation_details (optional dict),
                context_tokens_used (optional), context_budget_tokens (optional),
                return_tokens (optional), return_max_tokens (optional).
            correlation_id: Optional correlation ID for tracing.

        Returns:
            Count of valid events written in the batch.

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
                operation="write_audit_events",
                correlation_id=op_correlation_id,
            )

        context = ModelInfraErrorContext(
            transport_type=EnumInfraTransportType.DATABASE,
            operation="write_audit_events",
            target_name="context_audit_events",
            correlation_id=op_correlation_id,
        )

        sql = """
            INSERT INTO context_audit_events (
                task_id, parent_task_id, correlation_id,
                contract_id, event_type, enforcement_level,
                enforcement_action, violation_details,
                context_tokens_used, context_budget_tokens,
                return_tokens, return_max_tokens
            )
            VALUES ($1, $2, $3, $4, $5, $6, $7, $8::jsonb, $9, $10, $11, $12)
        """

        # Pre-filter old-schema messages that are missing required keys.
        valid_events = [
            e
            for e in events
            if _validate_event_fields(e, _REQUIRED_AUDIT_FIELDS, "write_audit_events")
        ]
        if not valid_events:
            logger.warning(
                "write_audit_events: entire batch skipped — no valid events after schema filter",
                extra={
                    "batch_size": len(events),
                    "correlation_id": str(op_correlation_id),
                },
            )
            return 0

        try:
            async with self._pool.acquire() as conn:
                await conn.executemany(
                    sql,
                    [
                        (
                            e["task_id"],
                            e.get("parent_task_id"),
                            e["correlation_id"],
                            e.get("contract_id"),
                            e["event_type"],
                            e["enforcement_level"],
                            e.get("enforcement_action"),
                            json.dumps(e["violation_details"])
                            if e.get("violation_details") is not None
                            else None,
                            e.get("context_tokens_used"),
                            e.get("context_budget_tokens"),
                            e.get("return_tokens"),
                            e.get("return_max_tokens"),
                        )
                        for e in valid_events
                    ],
                )

            async with self._circuit_breaker_lock:
                await self._reset_circuit_breaker()

            logger.debug(
                "Wrote context audit events batch",
                extra={
                    "count": len(valid_events),
                    "skipped": len(events) - len(valid_events),
                    "correlation_id": str(op_correlation_id),
                },
            )
            return len(valid_events)

        except asyncpg.QueryCanceledError as e:
            async with self._circuit_breaker_lock:
                await self._record_circuit_failure(
                    operation="write_audit_events",
                    correlation_id=op_correlation_id,
                )
            raise InfraTimeoutError(
                "Write context audit events timed out",
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
                    operation="write_audit_events",
                    correlation_id=op_correlation_id,
                )
            raise InfraConnectionError(
                "Database connection failed during write_audit_events",
                context=context,
            ) from e
        except asyncpg.PostgresError as e:
            async with self._circuit_breaker_lock:
                await self._record_circuit_failure(
                    operation="write_audit_events",
                    correlation_id=op_correlation_id,
                )
            raise RuntimeHostError(
                f"Database error during write_audit_events: {type(e).__name__}",
                context=context,
            ) from e

    def get_circuit_breaker_state(self) -> dict[str, JsonType]:
        """Return current circuit breaker state for health checks.

        Returns:
            Dict containing circuit breaker state information.
        """
        return self._get_circuit_breaker_state()


__all__ = ["WriterContextAuditPostgres"]
