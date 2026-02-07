# SPDX-License-Identifier: MIT
# Copyright (c) 2026 OmniNode Team
"""PostgreSQL implementation of the validation event ledger repository.

This module provides the PostgresValidationLedgerRepository which implements
ProtocolValidationLedgerRepository using asyncpg for direct PostgreSQL access.

Table: validation_event_ledger
    - Stores raw validation event envelopes for cross-repo replay
    - Idempotent writes via (kafka_topic, kafka_partition, kafka_offset) UNIQUE
    - Deterministic replay ordering via kafka_partition, kafka_offset
    - Time-based + min-run retention for cleanup

Idempotency:
    Uses INSERT ... ON CONFLICT (kafka_topic, kafka_partition, kafka_offset) DO NOTHING
    RETURNING id. If RETURNING returns no rows, the event was already in the ledger
    (duplicate). Duplicates are not errors - they enable idempotent replay.

Bytes Encoding:
    envelope_bytes is stored as BYTEA in PostgreSQL. On read, it is encoded to
    base64 via SQL encode(envelope_bytes, 'base64') for safe transport in
    Pydantic models.

Security Note:
    - DSN / pool credentials are managed externally (caller provides pool)
    - All queries use parameterized positional parameters ($1, $2, ...)
    - No dynamic table names or SQL string interpolation
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING
from uuid import UUID

import asyncpg

from omnibase_infra.enums import EnumInfraTransportType
from omnibase_infra.errors import ModelInfraErrorContext
from omnibase_infra.errors.repository import (
    RepositoryExecutionError,
    RepositoryTimeoutError,
)

if TYPE_CHECKING:
    from datetime import datetime

    from omnibase_infra.models.validation_ledger import (
        ModelValidationLedgerAppendResult,
        ModelValidationLedgerEntry,
        ModelValidationLedgerQuery,
        ModelValidationLedgerReplayBatch,
    )

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# SQL Constants
# ---------------------------------------------------------------------------

# Idempotent append with duplicate detection via RETURNING
_SQL_APPEND = """
INSERT INTO validation_event_ledger (
    run_id,
    repo_id,
    event_type,
    event_version,
    occurred_at,
    kafka_topic,
    kafka_partition,
    kafka_offset,
    envelope_bytes,
    envelope_hash
) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10)
ON CONFLICT (kafka_topic, kafka_partition, kafka_offset) DO NOTHING
RETURNING id
"""

# Query by run_id with base64-encoded envelope_bytes for transport
_SQL_QUERY_BY_RUN_ID = """
SELECT
    id,
    run_id,
    repo_id,
    event_type,
    event_version,
    occurred_at,
    kafka_topic,
    kafka_partition,
    kafka_offset,
    encode(envelope_bytes, 'base64') AS envelope_bytes,
    envelope_hash,
    created_at
FROM validation_event_ledger
WHERE run_id = $1
ORDER BY kafka_topic, kafka_partition, kafka_offset
LIMIT $2
OFFSET $3
"""

# Base SELECT used by the dynamic query() method
_SQL_SELECT_BASE = """
SELECT
    id,
    run_id,
    repo_id,
    event_type,
    event_version,
    occurred_at,
    kafka_topic,
    kafka_partition,
    kafka_offset,
    encode(envelope_bytes, 'base64') AS envelope_bytes,
    envelope_hash,
    created_at
FROM validation_event_ledger
"""

# Find protected run_ids: most recent N runs per repo
_SQL_PROTECTED_RUNS = """
SELECT DISTINCT run_id
FROM (
    SELECT run_id, repo_id,
           ROW_NUMBER() OVER (
               PARTITION BY repo_id
               ORDER BY MAX(created_at) DESC
           ) AS rn
    FROM validation_event_ledger
    GROUP BY run_id, repo_id
) ranked
WHERE rn <= $1
"""

# Batched delete for expired entries not in protected runs
_SQL_DELETE_EXPIRED_BATCH = """
DELETE FROM validation_event_ledger
WHERE id IN (
    SELECT id FROM validation_event_ledger
    WHERE created_at < NOW() - INTERVAL '1 day' * $1
      AND run_id != ALL($2::uuid[])
    LIMIT $3
)
"""

# Maximum cleanup iterations to prevent runaway loops
_MAX_CLEANUP_ITERATIONS = 100


class PostgresValidationLedgerRepository:
    """PostgreSQL implementation of ProtocolValidationLedgerRepository.

    Uses asyncpg connection pool for async database access. All operations
    use parameterized queries for safety and idempotent patterns for
    reliability.

    Attributes:
        _pool: asyncpg connection pool provided by caller.

    Example:
        >>> pool = await asyncpg.create_pool(dsn="postgresql://...")
        >>> repo = PostgresValidationLedgerRepository(pool)
        >>> result = await repo.append(
        ...     run_id=run_id,
        ...     repo_id="omnibase_core",
        ...     event_type="NodeRegistered",
        ...     event_version="1.0.0",
        ...     occurred_at=datetime.now(UTC),
        ...     kafka_topic="validation.events",
        ...     kafka_partition=0,
        ...     kafka_offset=42,
        ...     envelope_bytes=b"...",
        ...     envelope_hash="sha256:abc123",
        ... )
    """

    def __init__(self, pool: asyncpg.Pool) -> None:
        """Initialize with an asyncpg connection pool.

        Args:
            pool: An already-initialized asyncpg connection pool.
                The caller is responsible for pool lifecycle management.
        """
        self._pool = pool

    async def append(
        self,
        *,
        run_id: UUID,
        repo_id: str,
        event_type: str,
        event_version: str,
        occurred_at: datetime,
        kafka_topic: str,
        kafka_partition: int,
        kafka_offset: int,
        envelope_bytes: bytes,
        envelope_hash: str,
    ) -> ModelValidationLedgerAppendResult:
        """Append a validation event to the ledger.

        Executes an idempotent INSERT via ON CONFLICT DO NOTHING. Detects
        duplicates by checking whether the RETURNING clause produced a row.

        Args:
            run_id: UUID of the validation run this event belongs to.
            repo_id: Repository identifier (e.g., "omnibase_core").
            event_type: Fully qualified event type name.
            event_version: Semantic version of the event schema.
            occurred_at: Timestamp when the event originally occurred.
            kafka_topic: Kafka topic the event was consumed from.
            kafka_partition: Kafka partition number.
            kafka_offset: Kafka offset within the partition.
            envelope_bytes: Raw envelope bytes stored as BYTEA.
            envelope_hash: SHA-256 hash of the envelope for integrity.

        Returns:
            ModelValidationLedgerAppendResult with success, entry id, and
            duplicate flag.

        Raises:
            RepositoryExecutionError: If the INSERT fails.
            RepositoryTimeoutError: If the operation times out.
        """
        from omnibase_infra.models.validation_ledger import (
            ModelValidationLedgerAppendResult,
        )

        context = ModelInfraErrorContext.with_correlation(
            transport_type=EnumInfraTransportType.DATABASE,
            operation="validation_ledger.append",
        )

        try:
            async with self._pool.acquire() as conn:
                row = await conn.fetchrow(
                    _SQL_APPEND,
                    run_id,
                    repo_id,
                    event_type,
                    event_version,
                    occurred_at,
                    kafka_topic,
                    kafka_partition,
                    kafka_offset,
                    envelope_bytes,
                    envelope_hash,
                )

            if row is not None:
                ledger_entry_id = row["id"]
                duplicate = False
                logger.debug(
                    "Validation event appended to ledger",
                    extra={
                        "ledger_entry_id": str(ledger_entry_id),
                        "kafka_topic": kafka_topic,
                        "kafka_partition": kafka_partition,
                        "kafka_offset": kafka_offset,
                    },
                )
            else:
                ledger_entry_id = None
                duplicate = True
                logger.debug(
                    "Duplicate validation event detected (already in ledger)",
                    extra={
                        "kafka_topic": kafka_topic,
                        "kafka_partition": kafka_partition,
                        "kafka_offset": kafka_offset,
                    },
                )

            return ModelValidationLedgerAppendResult(
                success=True,
                ledger_entry_id=ledger_entry_id,
                duplicate=duplicate,
                kafka_topic=kafka_topic,
                kafka_partition=kafka_partition,
                kafka_offset=kafka_offset,
            )

        except asyncpg.QueryCanceledError as e:
            raise RepositoryTimeoutError(
                "Validation ledger append timed out",
                op_name="append",
                table="validation_event_ledger",
                context=context,
            ) from e
        except asyncpg.PostgresError as e:
            raise RepositoryExecutionError(
                f"Failed to append to validation ledger: {type(e).__name__}",
                op_name="append",
                table="validation_event_ledger",
                sql_fingerprint="INSERT INTO validation_event_ledger ... ON CONFLICT DO NOTHING",
                context=context,
            ) from e

    async def query_by_run_id(
        self,
        run_id: UUID,
        limit: int = 100,
        offset: int = 0,
    ) -> list[ModelValidationLedgerEntry]:
        """Query entries for a specific validation run.

        Returns entries ordered by kafka_topic, kafka_partition, kafka_offset
        for deterministic replay ordering.

        Args:
            run_id: The validation run UUID to query for.
            limit: Maximum number of entries to return (default: 100).
            offset: Number of entries to skip for pagination (default: 0).

        Returns:
            List of ModelValidationLedgerEntry ordered for deterministic replay.

        Raises:
            RepositoryExecutionError: If the query fails.
            RepositoryTimeoutError: If the query times out.
        """
        context = ModelInfraErrorContext.with_correlation(
            transport_type=EnumInfraTransportType.DATABASE,
            operation="validation_ledger.query_by_run_id",
        )

        try:
            async with self._pool.acquire() as conn:
                rows = await conn.fetch(
                    _SQL_QUERY_BY_RUN_ID,
                    run_id,
                    limit,
                    offset,
                )

            return [self._row_to_entry(row) for row in rows]

        except asyncpg.QueryCanceledError as e:
            raise RepositoryTimeoutError(
                "Validation ledger query_by_run_id timed out",
                op_name="query_by_run_id",
                table="validation_event_ledger",
                context=context,
            ) from e
        except asyncpg.PostgresError as e:
            raise RepositoryExecutionError(
                f"Failed to query validation ledger by run_id: {type(e).__name__}",
                op_name="query_by_run_id",
                table="validation_event_ledger",
                sql_fingerprint="SELECT ... FROM validation_event_ledger WHERE run_id = $1",
                context=context,
            ) from e

    async def query(
        self,
        query: ModelValidationLedgerQuery,
    ) -> ModelValidationLedgerReplayBatch:
        """Query with flexible filters, returns paginated results.

        Builds a dynamic WHERE clause from the non-None fields of the query
        model. Executes a COUNT(*) query first for total_count, then the
        main SELECT with LIMIT/OFFSET.

        Args:
            query: Query parameters with optional filters and pagination.

        Returns:
            ModelValidationLedgerReplayBatch with entries, total_count,
            has_more, and the original query.

        Raises:
            RepositoryExecutionError: If the query fails.
            RepositoryTimeoutError: If the query times out.
        """
        from omnibase_infra.models.validation_ledger import (
            ModelValidationLedgerReplayBatch,
        )

        context = ModelInfraErrorContext.with_correlation(
            transport_type=EnumInfraTransportType.DATABASE,
            operation="validation_ledger.query",
        )

        try:
            # Build dynamic WHERE clause
            where_clauses: list[str] = []
            parameters: list[object] = []
            param_index = 1

            if query.run_id is not None:
                where_clauses.append(f"run_id = ${param_index}")
                parameters.append(query.run_id)
                param_index += 1

            if query.repo_id is not None:
                where_clauses.append(f"repo_id = ${param_index}")
                parameters.append(query.repo_id)
                param_index += 1

            if query.event_type is not None:
                where_clauses.append(f"event_type = ${param_index}")
                parameters.append(query.event_type)
                param_index += 1

            if query.start_time is not None:
                where_clauses.append(f"occurred_at >= ${param_index}")
                parameters.append(query.start_time)
                param_index += 1

            if query.end_time is not None:
                where_clauses.append(f"occurred_at < ${param_index}")
                parameters.append(query.end_time)
                param_index += 1

            # Assemble WHERE
            where_sql = ""
            if where_clauses:
                where_sql = "WHERE " + " AND ".join(where_clauses)

            # COUNT query for total
            # S608: where_sql is built from parameterized conditions ($N), not user input
            count_sql = (
                f"SELECT COUNT(*) AS total FROM validation_event_ledger {where_sql}"  # noqa: S608
            )

            # Main SELECT with ordering and pagination
            select_sql = (
                _SQL_SELECT_BASE
                + where_sql
                + " ORDER BY kafka_topic, kafka_partition, kafka_offset"
                + f" LIMIT ${param_index} OFFSET ${param_index + 1}"
            )

            # Add pagination parameters for the SELECT query
            select_parameters = list(parameters) + [query.limit, query.offset]

            async with self._pool.acquire() as conn:
                # Execute count query
                count_row = await conn.fetchrow(count_sql, *parameters)
                total_count = int(count_row["total"]) if count_row else 0

                # Execute main query
                rows = await conn.fetch(select_sql, *select_parameters)

            entries = tuple(self._row_to_entry(row) for row in rows)
            has_more = query.offset + query.limit < total_count

            return ModelValidationLedgerReplayBatch(
                entries=entries,
                total_count=total_count,
                has_more=has_more,
                query=query,
            )

        except asyncpg.QueryCanceledError as e:
            raise RepositoryTimeoutError(
                "Validation ledger query timed out",
                op_name="query",
                table="validation_event_ledger",
                context=context,
            ) from e
        except asyncpg.PostgresError as e:
            raise RepositoryExecutionError(
                f"Failed to query validation ledger: {type(e).__name__}",
                op_name="query",
                table="validation_event_ledger",
                context=context,
            ) from e

    async def cleanup_expired(
        self,
        retention_days: int = 30,
        min_runs_per_repo: int = 25,
        batch_size: int = 1000,
    ) -> int:
        """Delete old entries respecting time-based and min-run retention.

        Two-phase cleanup:
        1. Find protected run_ids (most recent min_runs_per_repo per repo)
        2. Batched DELETE of entries older than retention_days whose run_id
           is NOT in the protected set

        Uses batched deletion (like StoreIdempotencyPostgres.cleanup_expired)
        to avoid long-running locks.

        Args:
            retention_days: Days to retain entries (default: 30).
            min_runs_per_repo: Minimum recent runs to protect per repo
                (default: 25).
            batch_size: Records to delete per batch (default: 1000).

        Returns:
            Total number of entries deleted across all batches.

        Raises:
            RepositoryExecutionError: If database operation fails.
            RepositoryTimeoutError: If cleanup times out.
        """
        context = ModelInfraErrorContext.with_correlation(
            transport_type=EnumInfraTransportType.DATABASE,
            operation="validation_ledger.cleanup_expired",
        )

        try:
            # Phase 1: Identify protected run_ids
            async with self._pool.acquire() as conn:
                protected_rows = await conn.fetch(
                    _SQL_PROTECTED_RUNS,
                    min_runs_per_repo,
                )

            protected_run_ids: list[UUID] = [row["run_id"] for row in protected_rows]

            logger.debug(
                "Protected run_ids for cleanup",
                extra={
                    "protected_count": len(protected_run_ids),
                    "retention_days": retention_days,
                    "min_runs_per_repo": min_runs_per_repo,
                },
            )

            # Phase 2: Batched deletion
            total_deleted = 0
            iteration = 0

            while iteration < _MAX_CLEANUP_ITERATIONS:
                iteration += 1

                async with self._pool.acquire() as conn:
                    result = await conn.execute(
                        _SQL_DELETE_EXPIRED_BATCH,
                        retention_days,
                        protected_run_ids,
                        batch_size,
                    )
                    # asyncpg returns "DELETE N"
                    batch_deleted = int(result.split()[-1]) if result else 0

                total_deleted += batch_deleted

                logger.debug(
                    "Validation ledger cleanup batch completed",
                    extra={
                        "batch_deleted": batch_deleted,
                        "total_deleted": total_deleted,
                        "iteration": iteration,
                        "batch_size": batch_size,
                    },
                )

                # If we deleted fewer than batch_size, no more rows to delete
                if batch_deleted < batch_size:
                    break

            logger.info(
                "Validation ledger cleanup completed",
                extra={
                    "total_deleted": total_deleted,
                    "retention_days": retention_days,
                    "min_runs_per_repo": min_runs_per_repo,
                    "protected_runs": len(protected_run_ids),
                    "iterations": iteration,
                    "batch_size": batch_size,
                },
            )

            return total_deleted

        except asyncpg.QueryCanceledError as e:
            raise RepositoryTimeoutError(
                "Validation ledger cleanup timed out",
                op_name="cleanup_expired",
                table="validation_event_ledger",
                context=context,
            ) from e
        except asyncpg.PostgresError as e:
            raise RepositoryExecutionError(
                f"Failed to cleanup validation ledger: {type(e).__name__}",
                op_name="cleanup_expired",
                table="validation_event_ledger",
                context=context,
            ) from e

    def _row_to_entry(self, row: asyncpg.Record) -> ModelValidationLedgerEntry:
        """Convert a database row to ModelValidationLedgerEntry.

        The row comes from asyncpg which returns asyncpg.Record objects.
        envelope_bytes is already base64-encoded via SQL encode() in the
        SELECT statement.

        Args:
            row: asyncpg.Record from a SELECT query.

        Returns:
            ModelValidationLedgerEntry populated from the row data.
        """
        from omnibase_infra.models.validation_ledger import (
            ModelValidationLedgerEntry,
        )

        return ModelValidationLedgerEntry(
            id=row["id"],
            run_id=row["run_id"],
            repo_id=row["repo_id"],
            event_type=row["event_type"],
            event_version=row["event_version"],
            occurred_at=row["occurred_at"],
            kafka_topic=row["kafka_topic"],
            kafka_partition=row["kafka_partition"],
            kafka_offset=row["kafka_offset"],
            envelope_bytes=row["envelope_bytes"],  # Already base64 from SQL encode()
            envelope_hash=row["envelope_hash"],
            created_at=row["created_at"],
        )


__all__ = ["PostgresValidationLedgerRepository"]
