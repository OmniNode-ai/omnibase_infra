# SPDX-License-Identifier: MIT
# Copyright (c) 2026 OmniNode Team
"""PostgreSQL implementation of ProtocolValidationLedgerRepository.

Provides concrete persistence for cross-repo validation events with:
    - Idempotent writes via ON CONFLICT DO NOTHING
    - Flexible query with dynamic WHERE clause building
    - Replay ordering by (kafka_partition, kafka_offset)
    - Combined retention: time-based + minimum run count per repo

Design Decisions:
    - Uses asyncpg connection pool directly (not HandlerDb) for simpler
      standalone repository usage. HandlerDb composition can be added
      later if circuit breaker / ledger sink integration is needed.
    - Base64 ↔ BYTEA encoding happens at the repository boundary.
    - All queries use parameterized SQL to prevent injection.

Ticket: OMN-1908
"""

from __future__ import annotations

import base64
import logging
from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING
from uuid import UUID

from omnibase_infra.enums import EnumInfraTransportType
from omnibase_infra.errors import ModelInfraErrorContext
from omnibase_infra.errors.repository.errors_repository import (
    RepositoryExecutionError,
)
from omnibase_infra.models.validation_ledger import (
    ModelValidationLedgerAppendResult,
    ModelValidationLedgerEntry,
    ModelValidationLedgerQuery,
    ModelValidationLedgerReplayBatch,
)

if TYPE_CHECKING:
    import asyncpg

logger = logging.getLogger(__name__)

# Default pagination limits
_DEFAULT_LIMIT: int = 100
_MAX_LIMIT: int = 10000

# SQL: Idempotent append with duplicate detection
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
    envelope_hash,
    created_at
) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11)
ON CONFLICT (kafka_topic, kafka_partition, kafka_offset) DO NOTHING
RETURNING id
"""

# SQL: Query by run_id ordered for replay
_SQL_QUERY_BY_RUN_ID = """
SELECT
    id, run_id, repo_id, event_type, event_version, occurred_at,
    kafka_topic, kafka_partition, kafka_offset,
    encode(envelope_bytes, 'base64') as envelope_bytes,
    envelope_hash, created_at
FROM validation_event_ledger
WHERE run_id = $1
ORDER BY kafka_partition, kafka_offset
LIMIT $2
OFFSET $3
"""

# SQL: Count by run_id
_SQL_COUNT_BY_RUN_ID = """
SELECT COUNT(*) as total
FROM validation_event_ledger
WHERE run_id = $1
"""

# SQL: Base query for flexible filtering
_SQL_QUERY_BASE = """
SELECT
    id, run_id, repo_id, event_type, event_version, occurred_at,
    kafka_topic, kafka_partition, kafka_offset,
    encode(envelope_bytes, 'base64') as envelope_bytes,
    envelope_hash, created_at
FROM validation_event_ledger
WHERE 1=1
"""

_SQL_COUNT_BASE = """
SELECT COUNT(*) as total
FROM validation_event_ledger
WHERE 1=1
"""

# SQL: Cleanup expired entries with run-count floor
# Uses a CTE to identify runs that must be preserved (most recent N runs per repo),
# then deletes entries NOT in the preserved set that are older than the cutoff.
_SQL_CLEANUP_EXPIRED = """
WITH preserved_runs AS (
    SELECT DISTINCT ON (repo_id, run_id) repo_id, run_id
    FROM (
        SELECT repo_id, run_id,
               DENSE_RANK() OVER (PARTITION BY repo_id ORDER BY MAX(occurred_at) DESC) AS run_rank
        FROM validation_event_ledger
        GROUP BY repo_id, run_id
    ) ranked
    WHERE run_rank <= $1
),
deletable AS (
    SELECT vel.id
    FROM validation_event_ledger vel
    LEFT JOIN preserved_runs pr ON vel.repo_id = pr.repo_id AND vel.run_id = pr.run_id
    WHERE pr.run_id IS NULL
      AND vel.created_at < $2
    LIMIT $3
)
DELETE FROM validation_event_ledger
WHERE id IN (SELECT id FROM deletable)
"""


class PostgresValidationLedgerRepository:
    """PostgreSQL-backed repository for validation event ledger persistence.

    Implements ProtocolValidationLedgerRepository with:
    - Idempotent writes via ON CONFLICT DO NOTHING
    - Flexible query building from optional filter parameters
    - Replay ordering by (kafka_partition, kafka_offset)
    - Combined retention policy (time-based + min runs per repo)

    Args:
        pool: asyncpg connection pool for database operations.
    """

    def __init__(self, pool: asyncpg.Pool) -> None:
        """Initialize with asyncpg connection pool.

        Args:
            pool: Initialized asyncpg connection pool.
        """
        self._pool = pool

    async def append(
        self,
        entry: ModelValidationLedgerEntry,
    ) -> ModelValidationLedgerAppendResult:
        """Append a validation event to the ledger with idempotent write support.

        Decodes base64 envelope_bytes to BYTEA, executes INSERT with
        ON CONFLICT DO NOTHING, and detects duplicates via RETURNING.

        Args:
            entry: Validation ledger entry to persist.

        Returns:
            ModelValidationLedgerAppendResult with success/duplicate info.

        Raises:
            RepositoryExecutionError: If database operation fails.
        """
        # Decode base64 envelope to bytes for BYTEA storage
        try:
            envelope_bytes_raw = base64.b64decode(entry.envelope_bytes)
        except Exception as e:
            ctx = ModelInfraErrorContext.with_correlation(
                transport_type=EnumInfraTransportType.DATABASE,
                operation="validation_ledger.append",
            )
            raise RepositoryExecutionError(
                f"Failed to decode base64 envelope_bytes: {type(e).__name__}",
                op_name="append",
                table="validation_event_ledger",
                retriable=False,
                context=ctx,
            ) from e

        try:
            row = await self._pool.fetchrow(
                _SQL_APPEND,
                entry.run_id,  # $1
                entry.repo_id,  # $2
                entry.event_type,  # $3
                entry.event_version,  # $4
                entry.occurred_at,  # $5
                entry.kafka_topic,  # $6
                entry.kafka_partition,  # $7
                entry.kafka_offset,  # $8
                envelope_bytes_raw,  # $9 (BYTEA)
                entry.envelope_hash,  # $10
                entry.created_at,  # $11
            )
        except Exception as e:
            ctx = ModelInfraErrorContext.with_correlation(
                transport_type=EnumInfraTransportType.DATABASE,
                operation="validation_ledger.append",
            )
            raise RepositoryExecutionError(
                f"Failed to append validation event: {type(e).__name__}",
                op_name="append",
                table="validation_event_ledger",
                sql_fingerprint="INSERT INTO validation_event_ledger (...) ON CONFLICT DO NOTHING",
                context=ctx,
            ) from e

        if row is not None:
            entry_id = UUID(str(row["id"]))
            duplicate = False
            logger.debug(
                "Validation event appended to ledger",
                extra={
                    "entry_id": str(entry_id),
                    "run_id": str(entry.run_id),
                    "event_type": entry.event_type,
                },
            )
        else:
            entry_id = None
            duplicate = True
            logger.debug(
                "Duplicate validation event detected",
                extra={
                    "kafka_topic": entry.kafka_topic,
                    "kafka_partition": entry.kafka_partition,
                    "kafka_offset": entry.kafka_offset,
                },
            )

        return ModelValidationLedgerAppendResult(
            success=True,
            entry_id=entry_id,
            duplicate=duplicate,
            kafka_topic=entry.kafka_topic,
            kafka_partition=entry.kafka_partition,
            kafka_offset=entry.kafka_offset,
        )

    async def query_by_run_id(
        self,
        run_id: UUID,
        limit: int = _DEFAULT_LIMIT,
        offset: int = 0,
    ) -> list[ModelValidationLedgerEntry]:
        """Query validation ledger entries by run ID.

        Returns entries ordered by (kafka_partition, kafka_offset) for
        correct replay ordering.

        Args:
            run_id: The validation run ID to search for.
            limit: Maximum entries to return.
            offset: Number of entries to skip.

        Returns:
            List of entries ordered by Kafka offset for replay.

        Raises:
            RepositoryExecutionError: If database operation fails.
        """
        limit = self._normalize_limit(limit)
        try:
            rows = await self._pool.fetch(
                _SQL_QUERY_BY_RUN_ID,
                run_id,
                limit,
                offset,
            )
        except Exception as e:
            ctx = ModelInfraErrorContext.with_correlation(
                transport_type=EnumInfraTransportType.DATABASE,
                operation="validation_ledger.query_by_run_id",
            )
            raise RepositoryExecutionError(
                f"Failed to query validation ledger by run_id: {type(e).__name__}",
                op_name="query_by_run_id",
                table="validation_event_ledger",
                context=ctx,
            ) from e

        return [self._row_to_entry(row) for row in rows]

    async def query(
        self,
        query: ModelValidationLedgerQuery,
    ) -> ModelValidationLedgerReplayBatch:
        """Execute a flexible query with optional filters and pagination.

        Builds a dynamic WHERE clause from non-None query fields.

        Args:
            query: Query parameters with optional filters.

        Returns:
            ModelValidationLedgerReplayBatch with entries and pagination metadata.

        Raises:
            RepositoryExecutionError: If database operation fails.
        """
        limit = self._normalize_limit(query.limit)

        # Build dynamic SQL
        data_sql, count_sql, params, count_params = self._build_query(query, limit)

        try:
            rows = await self._pool.fetch(data_sql, *params)
            count_row = await self._pool.fetchrow(count_sql, *count_params)
        except Exception as e:
            ctx = ModelInfraErrorContext.with_correlation(
                transport_type=EnumInfraTransportType.DATABASE,
                operation="validation_ledger.query",
            )
            raise RepositoryExecutionError(
                f"Failed to query validation ledger: {type(e).__name__}",
                op_name="query",
                table="validation_event_ledger",
                context=ctx,
            ) from e

        total_count = int(count_row["total"]) if count_row else 0
        entries = [self._row_to_entry(row) for row in rows]
        has_more = query.offset + len(entries) < total_count

        return ModelValidationLedgerReplayBatch(
            entries=entries,
            total_count=total_count,
            has_more=has_more,
            query=query,
        )

    async def cleanup_expired(
        self,
        retention_days: int = 30,
        min_runs_per_repo: int = 25,
        max_deletions: int = 1000,
    ) -> int:
        """Remove expired validation ledger entries.

        Implements combined retention policy:
        1. Delete entries older than retention_days
        2. BUT preserve at least min_runs_per_repo distinct runs per repo

        Args:
            retention_days: Delete entries older than this many days.
            min_runs_per_repo: Minimum distinct runs to keep per repo.
            max_deletions: Maximum rows to delete per call.

        Returns:
            Number of entries deleted.

        Raises:
            RepositoryExecutionError: If database operation fails.
        """
        cutoff = datetime.now(UTC) - timedelta(days=retention_days)

        try:
            result = await self._pool.execute(
                _SQL_CLEANUP_EXPIRED,
                min_runs_per_repo,  # $1
                cutoff,  # $2
                max_deletions,  # $3
            )
        except Exception as e:
            ctx = ModelInfraErrorContext.with_correlation(
                transport_type=EnumInfraTransportType.DATABASE,
                operation="validation_ledger.cleanup_expired",
            )
            raise RepositoryExecutionError(
                f"Failed to cleanup expired validation ledger entries: {type(e).__name__}",
                op_name="cleanup_expired",
                table="validation_event_ledger",
                context=ctx,
            ) from e

        # asyncpg returns "DELETE N" string
        deleted = int(result.split(" ")[-1]) if result else 0
        logger.info(
            "Cleaned up %d expired validation ledger entries",
            deleted,
            extra={
                "retention_days": retention_days,
                "min_runs_per_repo": min_runs_per_repo,
                "cutoff": cutoff.isoformat(),
            },
        )
        return deleted

    # =========================================================================
    # Private Helpers
    # =========================================================================

    def _normalize_limit(self, limit: int) -> int:
        """Normalize limit to valid range."""
        if limit < 1:
            return _DEFAULT_LIMIT
        if limit > _MAX_LIMIT:
            return _MAX_LIMIT
        return limit

    def _build_query(
        self,
        query: ModelValidationLedgerQuery,
        limit: int,
    ) -> tuple[str, str, list[object], list[object]]:
        """Build dynamic SQL from query parameters.

        Returns:
            Tuple of (data_sql, count_sql, data_params, count_params).
        """
        where_parts: list[str] = []
        params: list[object] = []
        param_idx = 1

        if query.run_id is not None:
            where_parts.append(f"AND run_id = ${param_idx}")
            params.append(query.run_id)
            param_idx += 1

        if query.repo_id is not None:
            where_parts.append(f"AND repo_id = ${param_idx}")
            params.append(query.repo_id)
            param_idx += 1

        if query.event_type is not None:
            where_parts.append(f"AND event_type = ${param_idx}")
            params.append(query.event_type)
            param_idx += 1

        if query.start_time is not None:
            where_parts.append(f"AND occurred_at >= ${param_idx}")
            params.append(query.start_time)
            param_idx += 1

        if query.end_time is not None:
            where_parts.append(f"AND occurred_at < ${param_idx}")
            params.append(query.end_time)
            param_idx += 1

        where_clause = " ".join(where_parts)

        # Count SQL uses same filters but no ordering/pagination
        count_sql = _SQL_COUNT_BASE + where_clause
        count_params = list(params)

        # Data SQL adds ordering and pagination
        data_sql = (
            _SQL_QUERY_BASE
            + where_clause
            + "\nORDER BY kafka_partition, kafka_offset"
            + f"\nLIMIT ${param_idx}"
            + f"\nOFFSET ${param_idx + 1}"
        )
        params.extend([limit, query.offset])

        return data_sql, count_sql, params, count_params

    def _row_to_entry(self, row: asyncpg.Record) -> ModelValidationLedgerEntry:
        """Convert a database row to ModelValidationLedgerEntry.

        envelope_bytes is already base64-encoded via SQL encode().
        """
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
            envelope_bytes=row["envelope_bytes"],
            envelope_hash=row["envelope_hash"],
            created_at=row["created_at"],
        )


__all__ = ["PostgresValidationLedgerRepository"]
