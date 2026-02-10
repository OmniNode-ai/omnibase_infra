# SPDX-License-Identifier: MIT
# Copyright (c) 2026 OmniNode Team
"""PostgreSQL ledger sink for injection effectiveness events.

This module provides a ledger sink that writes injection effectiveness
session events to the event_ledger table for audit trail and replay.
Each session gets a single ledger entry summarising the injection outcome.

Design Decisions:
    - Pool injection: asyncpg.Pool is injected, not created/managed
    - Idempotent writes: Uses ON CONFLICT (topic, partition, kafka_offset) DO NOTHING
    - Event value: JSON-serialized injection effectiveness summary
    - Circuit breaker: MixinAsyncCircuitBreaker for resilience

Related Tickets:
    - OMN-2078: Golden path: injection metrics + ledger storage

Example:
    >>> pool = await asyncpg.create_pool(dsn="postgresql://...")
    >>> sink = LedgerSinkInjectionEffectivenessPostgres(pool)
    >>>
    >>> await sink.append_session_entry(
    ...     session_id=session_id,
    ...     correlation_id=correlation_id,
    ...     event_type="context_utilization",
    ...     event_payload=b'{"session_id": "...", "utilization_score": 0.85}',
    ...     kafka_topic="onex.evt.omniclaude.context-utilization.v1",
    ...     kafka_partition=0,
    ...     kafka_offset=42,
    ... )
"""

from __future__ import annotations

import json
import logging
from datetime import UTC, datetime
from uuid import UUID

import asyncpg

from omnibase_infra.enums import EnumInfraTransportType
from omnibase_infra.mixins import MixinAsyncCircuitBreaker
from omnibase_infra.utils.util_db_error_context import db_operation_error_context

logger = logging.getLogger(__name__)

# Ledger source identifier for injection effectiveness events
LEDGER_SOURCE = "injection-effectiveness-consumer"


class LedgerSinkInjectionEffectivenessPostgres(MixinAsyncCircuitBreaker):
    """PostgreSQL ledger sink for injection effectiveness audit trail.

    Writes injection effectiveness events to the event_ledger table for
    durable audit trail and replay capability. Each event gets a single
    ledger entry with the event type, correlation ID, and serialized payload.

    Attributes:
        _pool: Injected asyncpg connection pool.
        DEFAULT_QUERY_TIMEOUT_SECONDS: Default timeout for writes (30s).
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
        """Initialize the ledger sink with an injected pool.

        Args:
            pool: asyncpg connection pool (lifecycle managed externally).
            circuit_breaker_threshold: Failures before opening circuit (default: 5).
            circuit_breaker_reset_timeout: Seconds before auto-reset (default: 60.0).
            circuit_breaker_half_open_successes: Successes to close from half-open (default: 1).
            query_timeout: Timeout in seconds for writes (default: 30.0).
        """
        self._pool = pool
        self._query_timeout = (
            query_timeout
            if query_timeout is not None
            else self.DEFAULT_QUERY_TIMEOUT_SECONDS
        )

        self._init_circuit_breaker(
            threshold=circuit_breaker_threshold,
            reset_timeout=circuit_breaker_reset_timeout,
            service_name="injection-effectiveness-ledger-sink",
            transport_type=EnumInfraTransportType.DATABASE,
            half_open_successes=circuit_breaker_half_open_successes,
        )

        logger.info(
            "LedgerSinkInjectionEffectivenessPostgres initialized",
            extra={"query_timeout": self._query_timeout},
        )

    async def append_session_entry(
        self,
        *,
        session_id: UUID,
        correlation_id: UUID,
        event_type: str,
        event_payload: bytes,
        kafka_topic: str,
        kafka_partition: int,
        kafka_offset: int,
        event_timestamp: datetime | None = None,
    ) -> UUID | None:
        """Append an injection effectiveness event to the event ledger.

        Uses INSERT ... ON CONFLICT DO NOTHING with the
        (topic, partition, kafka_offset) unique constraint for idempotency.

        Args:
            session_id: Session identifier (used as event_key).
            correlation_id: Correlation ID for distributed tracing.
            event_type: Event type discriminator (e.g., "context_utilization").
            event_payload: JSON-serialized event payload as bytes.
            kafka_topic: Kafka topic the event was consumed from.
            kafka_partition: Kafka partition number.
            kafka_offset: Kafka offset within the partition.
            event_timestamp: Event timestamp from producer (optional).

        Returns:
            UUID of the created ledger entry, or None if duplicate.

        Raises:
            InfraConnectionError: If database connection fails.
            InfraTimeoutError: If operation times out.
            InfraUnavailableError: If circuit breaker is open.
        """
        async with self._circuit_breaker_lock:
            await self._check_circuit_breaker(
                operation="append_session_entry",
                correlation_id=correlation_id,
            )

        sql = """
            INSERT INTO event_ledger (
                topic, partition, kafka_offset,
                event_key, event_value, onex_headers,
                correlation_id, event_type, source,
                event_timestamp, ledger_written_at
            )
            VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, NOW())
            ON CONFLICT (topic, partition, kafka_offset) DO NOTHING
            RETURNING ledger_entry_id
        """

        # Build ONEX headers with session context
        onex_headers = json.dumps(
            {
                "session_id": str(session_id),
                "source": LEDGER_SOURCE,
                "event_type": event_type,
            }
        )

        async with db_operation_error_context(
            operation="append_session_entry",
            target_name="event_ledger",
            correlation_id=correlation_id,
            timeout_seconds=self._query_timeout,
            circuit_breaker=self,
        ):
            async with self._pool.acquire() as conn:
                async with conn.transaction():
                    timeout_ms = int(self._query_timeout * 1000)
                    await conn.execute(
                        "SET LOCAL statement_timeout = $1", str(timeout_ms)
                    )

                    raw_result = await conn.fetchval(
                        sql,
                        kafka_topic,
                        kafka_partition,
                        kafka_offset,
                        str(session_id).encode(),  # event_key as BYTEA
                        event_payload,  # event_value as BYTEA
                        onex_headers,  # onex_headers as JSONB
                        correlation_id,
                        event_type,
                        LEDGER_SOURCE,
                        event_timestamp
                        if event_timestamp is not None
                        else datetime.now(UTC),
                    )
                    result: UUID | None = raw_result

            async with self._circuit_breaker_lock:
                await self._reset_circuit_breaker()

            if result is None:
                logger.debug(
                    "Ledger entry duplicate (idempotent skip)",
                    extra={
                        "kafka_topic": kafka_topic,
                        "kafka_partition": kafka_partition,
                        "kafka_offset": kafka_offset,
                        "correlation_id": str(correlation_id),
                    },
                )
                return None

            logger.debug(
                "Ledger entry appended",
                extra={
                    "ledger_entry_id": str(result),
                    "session_id": str(session_id),
                    "event_type": event_type,
                    "correlation_id": str(correlation_id),
                },
            )
            return result

    async def append_batch(
        self,
        entries: list[dict[str, object]],
        correlation_id: UUID,
    ) -> int:
        """Append a batch of entries to the event ledger.

        Each entry dict must contain: session_id, event_type, event_payload,
        kafka_topic, kafka_partition, kafka_offset. Optional: event_timestamp.

        Uses executemany for batch efficiency. Duplicates are silently skipped.

        Args:
            entries: List of entry dicts with required fields.
            correlation_id: Correlation ID for tracing.

        Returns:
            Number of entries submitted (not necessarily inserted; duplicates
            are silently skipped by ON CONFLICT DO NOTHING and executemany
            does not report per-row affected counts).

        Raises:
            InfraConnectionError: If database connection fails.
            InfraTimeoutError: If operation times out.
            InfraUnavailableError: If circuit breaker is open.
        """
        if not entries:
            return 0

        required_keys = {
            "session_id",
            "event_type",
            "event_payload",
            "kafka_topic",
            "kafka_partition",
            "kafka_offset",
        }
        for i, entry in enumerate(entries):
            missing = required_keys - entry.keys()
            if missing:
                msg = f"Entry {i} missing required keys: {sorted(missing)}"
                raise ValueError(msg)
            if not isinstance(entry["session_id"], UUID):
                msg = f"Entry {i} session_id must be UUID, got {type(entry['session_id']).__name__}"
                raise TypeError(msg)
            if not isinstance(entry["event_type"], str):
                msg = f"Entry {i} event_type must be str, got {type(entry['event_type']).__name__}"
                raise TypeError(msg)
            if not isinstance(entry["event_payload"], bytes):
                msg = f"Entry {i} event_payload must be bytes, got {type(entry['event_payload']).__name__}"
                raise TypeError(msg)
            if not isinstance(entry["kafka_partition"], int):
                msg = f"Entry {i} kafka_partition must be int, got {type(entry['kafka_partition']).__name__}"
                raise TypeError(msg)
            if not isinstance(entry["kafka_offset"], int):
                msg = f"Entry {i} kafka_offset must be int, got {type(entry['kafka_offset']).__name__}"
                raise TypeError(msg)

        async with self._circuit_breaker_lock:
            await self._check_circuit_breaker(
                operation="append_batch",
                correlation_id=correlation_id,
            )

        sql = """
            INSERT INTO event_ledger (
                topic, partition, kafka_offset,
                event_key, event_value, onex_headers,
                correlation_id, event_type, source,
                event_timestamp, ledger_written_at
            )
            VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, NOW())
            ON CONFLICT (topic, partition, kafka_offset) DO NOTHING
        """

        now = datetime.now(UTC)

        async with db_operation_error_context(
            operation="append_batch",
            target_name="event_ledger",
            correlation_id=correlation_id,
            timeout_seconds=self._query_timeout,
            circuit_breaker=self,
        ):
            async with self._pool.acquire() as conn:
                async with conn.transaction():
                    timeout_ms = int(self._query_timeout * 1000)
                    await conn.execute(
                        "SET LOCAL statement_timeout = $1", str(timeout_ms)
                    )

                    await conn.executemany(
                        sql,
                        [
                            (
                                e["kafka_topic"],
                                e["kafka_partition"],
                                e["kafka_offset"],
                                str(e["session_id"]).encode(),
                                e["event_payload"],
                                json.dumps(
                                    {
                                        "session_id": str(e["session_id"]),
                                        "source": LEDGER_SOURCE,
                                        "event_type": e["event_type"],
                                    }
                                ),
                                correlation_id,
                                e["event_type"],
                                LEDGER_SOURCE,
                                e.get("event_timestamp")
                                if e.get("event_timestamp") is not None
                                else now,
                            )
                            for e in entries
                        ],
                    )

            async with self._circuit_breaker_lock:
                await self._reset_circuit_breaker()

            logger.debug(
                "Ledger batch appended",
                extra={
                    "count": len(entries),
                    "correlation_id": str(correlation_id),
                },
            )
            return len(entries)


__all__ = ["LedgerSinkInjectionEffectivenessPostgres"]
