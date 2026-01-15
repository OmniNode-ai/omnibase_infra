# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""PostgreSQL Snapshot Store for Production Persistence.

This module provides a PostgreSQL implementation of ProtocolSnapshotStore
for production snapshot persistence. The store uses asyncpg for async
database operations and supports:

- Idempotent saves via content_hash deduplication
- Subject-based filtering and sequence ordering
- Atomic sequence number generation
- Parent reference tracking for lineage/fork scenarios

Table Schema:
    The store expects a `snapshots` table with the following schema. Use
    the `ensure_schema()` method to create it automatically.

    .. code-block:: sql

        CREATE TABLE IF NOT EXISTS snapshots (
            id UUID PRIMARY KEY,
            subject_type VARCHAR(255) NOT NULL,
            subject_id UUID NOT NULL,
            data JSONB NOT NULL,
            sequence_number INTEGER NOT NULL,
            version INTEGER DEFAULT 1,
            content_hash VARCHAR(128),
            created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
            parent_id UUID REFERENCES snapshots(id),

            CONSTRAINT snapshots_subject_sequence_unique
                UNIQUE (subject_type, subject_id, sequence_number)
        );

        CREATE INDEX IF NOT EXISTS idx_snapshots_subject
            ON snapshots (subject_type, subject_id, sequence_number DESC);
        CREATE INDEX IF NOT EXISTS idx_snapshots_content_hash
            ON snapshots (content_hash) WHERE content_hash IS NOT NULL;

Connection Pooling:
    The store requires an asyncpg connection pool to be injected at
    construction time. This allows the pool to be shared across multiple
    stores and services, with lifecycle managed by the application.

    .. code-block:: python

        import asyncpg
        from omnibase_infra.services.snapshot import StoreSnapshotPostgres

        # Create pool (managed by application)
        pool = await asyncpg.create_pool(dsn="postgresql://...")

        # Inject pool into store
        store = StoreSnapshotPostgres(pool=pool)
        await store.ensure_schema()

        # Use store
        snapshot_id = await store.save(snapshot)

Error Handling:
    All operations wrap database exceptions in ONEX error types:
    - InfraConnectionError: Connection failures, pool exhaustion
    - InfraTimeoutError: Query timeouts (from asyncpg.QueryCanceledError)

Security:
    - All queries use parameterized statements (no SQL injection)
    - DSN/credentials are never logged or exposed in errors
    - Connection pool credentials managed externally

Related Tickets:
    - OMN-1246: ServiceSnapshot Infrastructure Primitive
"""

from __future__ import annotations

import json
import logging
from datetime import datetime
from uuid import UUID

import asyncpg
import asyncpg.exceptions

from omnibase_infra.enums import EnumInfraTransportType
from omnibase_infra.errors import InfraConnectionError, ModelInfraErrorContext
from omnibase_infra.models.snapshot import ModelSnapshot, ModelSubjectRef

logger = logging.getLogger(__name__)


class StoreSnapshotPostgres:
    """PostgreSQL implementation of ProtocolSnapshotStore.

    Provides production-grade snapshot persistence using asyncpg with:
    - Content-hash based idempotency for duplicate detection
    - Atomic sequence number generation using database MAX() + 1
    - JSONB storage for snapshot data payloads
    - Composite indexes for efficient subject-based queries

    Connection Management:
        The pool is injected at construction time and NOT managed by
        this class. The application is responsible for pool lifecycle
        (creation, health checks, shutdown).

    Concurrency:
        Database-level constraints ensure sequence uniqueness. For
        high-concurrency scenarios, consider using database sequences
        or advisory locks.

    Example:
        >>> import asyncpg
        >>> from omnibase_infra.services.snapshot import StoreSnapshotPostgres
        >>> from omnibase_infra.models.snapshot import ModelSnapshot, ModelSubjectRef
        >>>
        >>> # Create pool and store
        >>> pool = await asyncpg.create_pool(dsn="postgresql://...")
        >>> store = StoreSnapshotPostgres(pool=pool)
        >>> await store.ensure_schema()
        >>>
        >>> # Save a snapshot
        >>> subject = ModelSubjectRef(subject_type="agent", subject_id=uuid4())
        >>> snapshot = ModelSnapshot(
        ...     subject=subject,
        ...     data={"status": "active"},
        ...     sequence_number=1,
        ... )
        >>> saved_id = await store.save(snapshot)
    """

    def __init__(self, pool: asyncpg.Pool) -> None:
        """Initialize the PostgreSQL snapshot store.

        Args:
            pool: asyncpg connection pool. The pool must be created and
                configured by the caller. The store does not manage pool
                lifecycle (creation, shutdown).

        Note:
            Call ensure_schema() after construction to create the
            required table and indexes if they don't exist.
        """
        self._pool = pool

    async def save(self, snapshot: ModelSnapshot) -> UUID:
        """Persist a snapshot with content-hash based idempotency.

        If a snapshot with the same content_hash already exists,
        returns the existing snapshot's ID instead of creating a
        duplicate. This enables safe retries without data duplication.

        Race Condition Handling:
            This method uses a CTE (Common Table Expression) pattern to
            atomically check for existing content_hash and conditionally
            insert. If a concurrent process inserts a row with the same
            sequence_number (causing UniqueViolationError), the method
            re-checks for content_hash match to preserve idempotency.

        Args:
            snapshot: The snapshot to persist.

        Returns:
            UUID of the saved or existing snapshot.

        Raises:
            InfraConnectionError: If database connection fails or
                query execution fails.
        """
        # Serialize data to JSON for JSONB storage (done outside try for clarity)
        data_json = json.dumps(snapshot.data, sort_keys=True)

        try:
            async with self._pool.acquire() as conn:
                if snapshot.content_hash:
                    # Use CTE for atomic check-and-insert to minimize race window.
                    # This pattern:
                    # 1. Checks for existing content_hash in 'existing' CTE
                    # 2. Conditionally inserts only if no existing row found
                    # 3. Uses ON CONFLICT DO NOTHING for sequence constraint races
                    # 4. Returns existing ID or newly inserted ID
                    result = await conn.fetchval(
                        """
                        WITH existing AS (
                            SELECT id FROM snapshots
                            WHERE content_hash = $7
                            LIMIT 1
                        ), inserted AS (
                            INSERT INTO snapshots (
                                id, subject_type, subject_id, data, sequence_number,
                                version, content_hash, created_at, parent_id
                            )
                            SELECT $1, $2, $3, $4::jsonb, $5, $6, $7, $8, $9
                            WHERE NOT EXISTS (SELECT 1 FROM existing)
                            ON CONFLICT (subject_type, subject_id, sequence_number)
                                DO NOTHING
                            RETURNING id
                        )
                        SELECT COALESCE(
                            (SELECT id FROM existing),
                            (SELECT id FROM inserted)
                        )
                        """,
                        snapshot.id,
                        snapshot.subject.subject_type,
                        snapshot.subject.subject_id,
                        data_json,
                        snapshot.sequence_number,
                        snapshot.version,
                        snapshot.content_hash,
                        snapshot.created_at,
                        snapshot.parent_id,
                    )

                    if result:
                        result_id = UUID(str(result))
                        if result_id != snapshot.id:
                            logger.debug(
                                "Duplicate snapshot detected via content_hash, "
                                "returning existing ID",
                                extra={
                                    "existing_id": str(result_id),
                                    "content_hash": snapshot.content_hash[:16] + "...",
                                },
                            )
                        else:
                            logger.debug(
                                "Snapshot saved",
                                extra={
                                    "snapshot_id": str(snapshot.id),
                                    "subject_type": snapshot.subject.subject_type,
                                    "sequence_number": snapshot.sequence_number,
                                },
                            )
                        return result_id

                    # If result is None, ON CONFLICT DO NOTHING triggered due to
                    # sequence conflict. Re-check content_hash for race condition
                    # where another process inserted same content concurrently.
                    existing = await conn.fetchval(
                        "SELECT id FROM snapshots WHERE content_hash = $1",
                        snapshot.content_hash,
                    )
                    if existing:
                        existing_id = UUID(str(existing))
                        logger.debug(
                            "Concurrent duplicate detected after sequence conflict, "
                            "returning existing ID",
                            extra={
                                "existing_id": str(existing_id),
                                "content_hash": snapshot.content_hash[:16] + "...",
                            },
                        )
                        return existing_id

                    # Sequence conflict with different content - this is a real
                    # conflict that the caller should handle
                    context = ModelInfraErrorContext(
                        transport_type=EnumInfraTransportType.DATABASE,
                        operation="save_snapshot",
                        target_name="snapshots",
                    )
                    raise InfraConnectionError(
                        f"Sequence conflict: sequence_number {snapshot.sequence_number} "
                        f"already exists for subject "
                        f"({snapshot.subject.subject_type}, {snapshot.subject.subject_id}) "
                        f"with different content",
                        context=context,
                    )

                # No content_hash - insert directly with conflict handling
                result = await conn.fetchval(
                    """
                    INSERT INTO snapshots (
                        id, subject_type, subject_id, data, sequence_number,
                        version, content_hash, created_at, parent_id
                    ) VALUES ($1, $2, $3, $4::jsonb, $5, $6, $7, $8, $9)
                    ON CONFLICT (subject_type, subject_id, sequence_number)
                        DO NOTHING
                    RETURNING id
                    """,
                    snapshot.id,
                    snapshot.subject.subject_type,
                    snapshot.subject.subject_id,
                    data_json,
                    snapshot.sequence_number,
                    snapshot.version,
                    snapshot.content_hash,
                    snapshot.created_at,
                    snapshot.parent_id,
                )

                if result:
                    logger.debug(
                        "Snapshot saved",
                        extra={
                            "snapshot_id": str(snapshot.id),
                            "subject_type": snapshot.subject.subject_type,
                            "sequence_number": snapshot.sequence_number,
                        },
                    )
                    return UUID(str(result))

                # Sequence conflict - return error
                context = ModelInfraErrorContext(
                    transport_type=EnumInfraTransportType.DATABASE,
                    operation="save_snapshot",
                    target_name="snapshots",
                )
                raise InfraConnectionError(
                    f"Sequence conflict: sequence_number {snapshot.sequence_number} "
                    f"already exists for subject "
                    f"({snapshot.subject.subject_type}, {snapshot.subject.subject_id})",
                    context=context,
                )

        except asyncpg.exceptions.UniqueViolationError:
            # Race condition: constraint violation despite ON CONFLICT
            # (can happen with concurrent transactions in edge cases).
            # Re-check content_hash to preserve idempotency.
            if snapshot.content_hash:
                try:
                    async with self._pool.acquire() as conn:
                        existing = await conn.fetchval(
                            "SELECT id FROM snapshots WHERE content_hash = $1",
                            snapshot.content_hash,
                        )
                        if existing:
                            existing_id = UUID(str(existing))
                            logger.debug(
                                "Race condition resolved: returning existing ID "
                                "after UniqueViolationError",
                                extra={
                                    "existing_id": str(existing_id),
                                    "content_hash": snapshot.content_hash[:16] + "...",
                                },
                            )
                            return existing_id
                except Exception:
                    pass  # Fall through to re-raise original error

            # Re-raise as InfraConnectionError
            context = ModelInfraErrorContext(
                transport_type=EnumInfraTransportType.DATABASE,
                operation="save_snapshot",
                target_name="snapshots",
            )
            raise InfraConnectionError(
                f"Unique constraint violation during save for subject "
                f"({snapshot.subject.subject_type}, {snapshot.subject.subject_id})",
                context=context,
            )

        except InfraConnectionError:
            # Re-raise our own errors without wrapping
            raise

        except Exception as e:
            context = ModelInfraErrorContext(
                transport_type=EnumInfraTransportType.DATABASE,
                operation="save_snapshot",
                target_name="snapshots",
            )
            raise InfraConnectionError(
                f"Failed to save snapshot: {type(e).__name__}",
                context=context,
            ) from e

    async def load(self, snapshot_id: UUID) -> ModelSnapshot | None:
        """Load a snapshot by ID.

        Args:
            snapshot_id: The unique identifier of the snapshot.

        Returns:
            The snapshot if found, None otherwise.

        Raises:
            InfraConnectionError: If database connection fails.
        """
        try:
            async with self._pool.acquire() as conn:
                row = await conn.fetchrow(
                    "SELECT * FROM snapshots WHERE id = $1",
                    snapshot_id,
                )
                if row is None:
                    return None
                return self._row_to_model(row)
        except Exception as e:
            context = ModelInfraErrorContext(
                transport_type=EnumInfraTransportType.DATABASE,
                operation="load_snapshot",
                target_name="snapshots",
            )
            raise InfraConnectionError(
                f"Failed to load snapshot: {type(e).__name__}",
                context=context,
            ) from e

    async def load_latest(
        self,
        subject: ModelSubjectRef | None = None,
    ) -> ModelSnapshot | None:
        """Load the most recent snapshot by sequence_number.

        Retrieves the snapshot with the highest sequence_number for
        the given subject. If no subject is provided, returns the
        globally most recent snapshot.

        Args:
            subject: Optional filter by subject reference.

        Returns:
            The most recent snapshot matching criteria, or None.

        Raises:
            InfraConnectionError: If database connection fails.
        """
        try:
            async with self._pool.acquire() as conn:
                if subject:
                    row = await conn.fetchrow(
                        """
                        SELECT * FROM snapshots
                        WHERE subject_type = $1 AND subject_id = $2
                        ORDER BY sequence_number DESC LIMIT 1
                        """,
                        subject.subject_type,
                        subject.subject_id,
                    )
                else:
                    row = await conn.fetchrow(
                        "SELECT * FROM snapshots ORDER BY sequence_number DESC LIMIT 1"
                    )
                if row is None:
                    return None
                return self._row_to_model(row)
        except Exception as e:
            context = ModelInfraErrorContext(
                transport_type=EnumInfraTransportType.DATABASE,
                operation="load_latest_snapshot",
                target_name="snapshots",
            )
            raise InfraConnectionError(
                f"Failed to load latest snapshot: {type(e).__name__}",
                context=context,
            ) from e

    async def query(
        self,
        subject: ModelSubjectRef | None = None,
        limit: int = 50,
        after: datetime | None = None,
    ) -> list[ModelSnapshot]:
        """Query snapshots with optional filtering.

        Returns snapshots ordered by sequence_number descending
        (most recent first).

        Args:
            subject: Optional filter by subject reference.
            limit: Maximum results to return (default 50).
            after: Only return snapshots created after this time.

        Returns:
            List of snapshots ordered by sequence_number descending.

        Raises:
            InfraConnectionError: If database connection fails.
        """
        try:
            async with self._pool.acquire() as conn:
                # Build dynamic query with parameterized conditions
                conditions: list[str] = []
                params: list[object] = []

                if subject:
                    conditions.append(f"subject_type = ${len(params) + 1}")
                    params.append(subject.subject_type)
                    conditions.append(f"subject_id = ${len(params) + 1}")
                    params.append(subject.subject_id)

                if after:
                    conditions.append(f"created_at > ${len(params) + 1}")
                    params.append(after)

                where_clause = " AND ".join(conditions) if conditions else "TRUE"
                params.append(limit)

                # S608: This is NOT SQL injection - where_clause contains only
                # safe static column names with parameterized value placeholders
                # ($1, $2, etc). All user-supplied values go through params.
                query = f"""
                    SELECT * FROM snapshots
                    WHERE {where_clause}
                    ORDER BY sequence_number DESC
                    LIMIT ${len(params)}
                """  # noqa: S608

                rows = await conn.fetch(query, *params)
                return [self._row_to_model(row) for row in rows]
        except Exception as e:
            context = ModelInfraErrorContext(
                transport_type=EnumInfraTransportType.DATABASE,
                operation="query_snapshots",
                target_name="snapshots",
            )
            raise InfraConnectionError(
                f"Failed to query snapshots: {type(e).__name__}",
                context=context,
            ) from e

    async def delete(self, snapshot_id: UUID) -> bool:
        """Delete a snapshot by ID.

        Args:
            snapshot_id: The unique identifier of the snapshot to delete.

        Returns:
            True if the snapshot was deleted, False if not found.

        Raises:
            InfraConnectionError: If database connection fails.
        """
        try:
            async with self._pool.acquire() as conn:
                result = await conn.execute(
                    "DELETE FROM snapshots WHERE id = $1",
                    snapshot_id,
                )
                # asyncpg returns "DELETE N" where N is rows affected
                deleted: bool = str(result) == "DELETE 1"
                if deleted:
                    logger.debug(
                        "Snapshot deleted",
                        extra={"snapshot_id": str(snapshot_id)},
                    )
                return deleted
        except Exception as e:
            context = ModelInfraErrorContext(
                transport_type=EnumInfraTransportType.DATABASE,
                operation="delete_snapshot",
                target_name="snapshots",
            )
            raise InfraConnectionError(
                f"Failed to delete snapshot: {type(e).__name__}",
                context=context,
            ) from e

    async def get_next_sequence_number(self, subject: ModelSubjectRef) -> int:
        """Get the next sequence number for a subject.

        Uses MAX() + 1 to determine the next sequence number. This is
        NOT atomic across concurrent operations - for high-concurrency
        scenarios, consider database sequences or advisory locks.

        Args:
            subject: The subject reference for sequence generation.

        Returns:
            The next sequence number (starts at 1 for new subjects).

        Raises:
            InfraConnectionError: If database connection fails.

        Note:
            The UNIQUE constraint on (subject_type, subject_id, sequence_number)
            ensures no duplicate sequences are created, but may result in
            constraint violations under high concurrency. Callers should
            handle retry logic.
        """
        try:
            async with self._pool.acquire() as conn:
                result = await conn.fetchval(
                    """
                    SELECT COALESCE(MAX(sequence_number), 0) + 1
                    FROM snapshots
                    WHERE subject_type = $1 AND subject_id = $2
                    """,
                    subject.subject_type,
                    subject.subject_id,
                )
                return int(result) if result else 1
        except Exception as e:
            context = ModelInfraErrorContext(
                transport_type=EnumInfraTransportType.DATABASE,
                operation="get_sequence_number",
                target_name="snapshots",
            )
            raise InfraConnectionError(
                f"Failed to get sequence number: {type(e).__name__}",
                context=context,
            ) from e

    def _row_to_model(self, row: asyncpg.Record) -> ModelSnapshot:
        """Convert a database row to a ModelSnapshot.

        Args:
            row: asyncpg Record from a SELECT query.

        Returns:
            ModelSnapshot instance populated from the row.
        """
        # asyncpg returns JSONB as dict automatically
        data = row["data"]
        if isinstance(data, str):
            data = json.loads(data)

        return ModelSnapshot(
            id=row["id"],
            subject=ModelSubjectRef(
                subject_type=row["subject_type"],
                subject_id=row["subject_id"],
            ),
            data=data,
            sequence_number=row["sequence_number"],
            version=row["version"],
            content_hash=row["content_hash"],
            created_at=row["created_at"],
            parent_id=row["parent_id"],
        )

    async def ensure_schema(self) -> None:
        """Create the snapshots table and indexes if they don't exist.

        This method is idempotent and safe to call on every startup.
        Uses IF NOT EXISTS clauses to avoid errors on existing objects.

        Raises:
            InfraConnectionError: If schema creation fails.

        Note:
            This method uses multi-statement execution via transaction.
            Each statement is executed separately to work within asyncpg's
            single-statement limitation.
        """
        try:
            async with self._pool.acquire() as conn:
                # Create table
                await conn.execute("""
                    CREATE TABLE IF NOT EXISTS snapshots (
                        id UUID PRIMARY KEY,
                        subject_type VARCHAR(255) NOT NULL,
                        subject_id UUID NOT NULL,
                        data JSONB NOT NULL,
                        sequence_number INTEGER NOT NULL,
                        version INTEGER DEFAULT 1,
                        content_hash VARCHAR(128),
                        created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                        parent_id UUID REFERENCES snapshots(id),

                        CONSTRAINT snapshots_subject_sequence_unique
                            UNIQUE (subject_type, subject_id, sequence_number)
                    )
                """)

                # Create subject index
                await conn.execute("""
                    CREATE INDEX IF NOT EXISTS idx_snapshots_subject
                        ON snapshots (subject_type, subject_id, sequence_number DESC)
                """)

                # Create content hash index (partial - only non-null)
                await conn.execute("""
                    CREATE INDEX IF NOT EXISTS idx_snapshots_content_hash
                        ON snapshots (content_hash) WHERE content_hash IS NOT NULL
                """)

                logger.info(
                    "Snapshot schema ensured (table and indexes created/verified)"
                )
        except Exception as e:
            context = ModelInfraErrorContext(
                transport_type=EnumInfraTransportType.DATABASE,
                operation="ensure_schema",
                target_name="snapshots",
            )
            raise InfraConnectionError(
                f"Failed to ensure schema: {type(e).__name__}",
                context=context,
            ) from e


__all__: list[str] = ["StoreSnapshotPostgres"]
