# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""PostgreSQL Writer for Consumer Health read-model projection (OMN-6757).

Persists consumer health events to the ``consumer_health_events`` table
for omnidash ``/consumer-health`` dashboard queries.

Design Decisions:
    - Pool injection: asyncpg.Pool is injected, not created/managed here.
    - Batch inserts: Uses executemany for efficient batch processing.
    - Idempotency: ON CONFLICT (event_id) DO NOTHING.
    - Circuit breaker: MixinAsyncCircuitBreaker for resilience.

Idempotency Contract:
    | Table                  | Unique Key | Conflict Action |
    |------------------------|------------|-----------------|
    | consumer_health_events | event_id   | DO NOTHING      |
"""
# no-migration: schema created in migration 056 (same PR, prior commit)

from __future__ import annotations

import logging
from datetime import datetime
from uuid import UUID, uuid4

import asyncpg

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

_REQUIRED_FIELDS: frozenset[str] = frozenset(
    {
        "event_id",
        "consumer_identity",
        "consumer_group",
        "topic",
        "event_type",
        "severity",
        "fingerprint",
        "emitted_at",
    }
)

_INSERT_SQL = """\
INSERT INTO consumer_health_events (
    event_id, correlation_id, consumer_identity, consumer_group, topic,
    event_type, severity, fingerprint, rebalance_duration_ms,
    partitions_assigned, partitions_revoked, error_message, error_type,
    hostname, service_label, emitted_at, ingested_at
) VALUES (
    $1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14, $15, $16, NOW()
) ON CONFLICT (event_id) DO NOTHING
"""


def _validate_event(event: dict[str, object], context: str) -> bool:
    """Check that all required fields are present."""
    missing = _REQUIRED_FIELDS - event.keys()
    if missing:
        logger.warning(
            "Skipping %s event: missing fields %s (event_id=%s)",
            context,
            sorted(missing),
            event.get("event_id", "?"),
        )
        return False
    return True


class WriterConsumerHealthPostgres(MixinAsyncCircuitBreaker):
    """Batch writer for consumer health events to PostgreSQL."""

    def __init__(
        self,
        pool: asyncpg.Pool,  # type: ignore[type-arg]
        *,
        circuit_breaker_threshold: int = 5,
        circuit_breaker_reset_timeout: float = 60.0,
        circuit_breaker_half_open_successes: int = 1,
    ) -> None:
        self._pool = pool
        self._init_circuit_breaker(
            threshold=circuit_breaker_threshold,
            reset_timeout=circuit_breaker_reset_timeout,
            service_name="consumer-health-postgres-writer",
            transport_type=EnumInfraTransportType.DATABASE,
            half_open_successes=circuit_breaker_half_open_successes,
        )

    async def write_batch(
        self,
        events: list[dict[str, object]],
        *,
        correlation_id: UUID | None = None,
    ) -> int:
        """Write a batch of consumer health events to PostgreSQL.

        Args:
            events: List of deserialized consumer health event dicts.
            correlation_id: Optional correlation ID for tracing.

        Returns:
            Number of events successfully written.

        Raises:
            InfraConnectionError: If the database connection fails.
            InfraTimeoutError: If the query times out.
        """
        cid = correlation_id or uuid4()
        context = ModelInfraErrorContext.with_correlation(
            correlation_id=cid,
            transport_type=EnumInfraTransportType.DATABASE,
            operation="write_consumer_health_batch",
        )

        async with self._circuit_breaker_lock:
            await self._check_circuit_breaker("write_batch", cid)

        valid = [e for e in events if _validate_event(e, "consumer_health")]
        if not valid:
            return 0

        rows = []
        for e in valid:
            rows.append(
                (
                    UUID(str(e["event_id"])),
                    UUID(str(e.get("correlation_id", uuid4()))),
                    str(e["consumer_identity"]),
                    str(e["consumer_group"]),
                    str(e["topic"]),
                    str(e["event_type"]),
                    str(e["severity"]),
                    str(e["fingerprint"]),
                    e.get("rebalance_duration_ms"),
                    e.get("partitions_assigned"),
                    e.get("partitions_revoked"),
                    str(e.get("error_message", "")),
                    str(e.get("error_type", "")),
                    str(e.get("hostname", "")),
                    str(e.get("service_label", "")),
                    datetime.fromisoformat(str(e["emitted_at"])),
                )
            )

        try:
            async with self._pool.acquire() as conn:
                await conn.executemany(_INSERT_SQL, rows)
            written = len(rows)

            async with self._circuit_breaker_lock:
                await self._reset_circuit_breaker()

            logger.debug("Wrote %d consumer health events", written)
            return written
        except asyncpg.QueryCanceledError as exc:
            async with self._circuit_breaker_lock:
                await self._record_circuit_failure(
                    operation="write_consumer_health_batch",
                    correlation_id=cid,
                )
            raise InfraTimeoutError(
                "Write consumer health events timed out",
                context=ModelTimeoutErrorContext(
                    transport_type=context.transport_type,
                    operation=context.operation,
                    target_name=context.target_name,
                    correlation_id=context.correlation_id,
                    timeout_seconds=30.0,
                ),
            ) from exc
        except asyncpg.PostgresConnectionError as exc:
            async with self._circuit_breaker_lock:
                await self._record_circuit_failure(
                    operation="write_consumer_health_batch",
                    correlation_id=cid,
                )
            raise InfraConnectionError(
                f"Database connection failed during write_consumer_health_batch: {exc}",
                context=context,
            ) from exc
        except asyncpg.PostgresError as exc:
            async with self._circuit_breaker_lock:
                await self._record_circuit_failure(
                    operation="write_consumer_health_batch",
                    correlation_id=cid,
                )
            raise RuntimeHostError(
                f"Database error during write_consumer_health_batch: {type(exc).__name__}",
                context=context,
            ) from exc
