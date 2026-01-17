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

        -- UNIQUE partial index enables atomic ON CONFLICT upserts
        CREATE UNIQUE INDEX IF NOT EXISTS idx_snapshots_content_hash
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
from datetime import UTC, datetime, timedelta
from uuid import UUID

import asyncpg
import asyncpg.exceptions

from omnibase_infra.enums import EnumInfraTransportType
from omnibase_infra.errors import (
    InfraConnectionError,
    ModelInfraErrorContext,
    ProtocolConfigurationError,
)
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
            This method uses INSERT ON CONFLICT with a unique partial index
            on content_hash to achieve atomic idempotency. The database-level
            unique constraint eliminates TOCTOU race conditions that would
            occur with separate SELECT-then-INSERT patterns.

            Conflict scenarios:
            - Same content_hash (any sequence): Returns existing ID via ON CONFLICT
            - Same sequence, different content_hash: Raises UniqueViolationError
            - No conflicts: Normal insert

        Args:
            snapshot: The snapshot to persist.

        Returns:
            UUID of the saved or existing snapshot.

        Raises:
            InfraConnectionError: If database connection fails or
                query execution fails.

        Note:
            Requires ensure_schema() to have created the unique partial index
            on content_hash. See ensure_schema() for details.
        """
        # Serialize data to JSON for JSONB storage (done outside try for clarity)
        data_json = json.dumps(snapshot.data, sort_keys=True)

        try:
            async with self._pool.acquire() as conn:
                if snapshot.content_hash:
                    # Atomic upsert using ON CONFLICT on the unique content_hash index.
                    # This eliminates the TOCTOU race condition by letting the database
                    # handle the check-and-insert atomically:
                    # - If content_hash exists: DO UPDATE (no-op) returns existing row
                    # - If content_hash is new: INSERT returns new row
                    # - If sequence conflicts: Raises UniqueViolationError (handled below)
                    #
                    # The DO UPDATE SET id = snapshots.id is a no-op that enables
                    # RETURNING to return the existing row's id.
                    result = await conn.fetchval(
                        """
                        INSERT INTO snapshots (
                            id, subject_type, subject_id, data, sequence_number,
                            version, content_hash, created_at, parent_id
                        ) VALUES ($1, $2, $3, $4::jsonb, $5, $6, $7, $8, $9)
                        ON CONFLICT (content_hash) WHERE content_hash IS NOT NULL
                            DO UPDATE SET id = snapshots.id
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

                    # Result should never be None with DO UPDATE, but handle defensively
                    context = ModelInfraErrorContext(
                        transport_type=EnumInfraTransportType.DATABASE,
                        operation="save_snapshot",
                        target_name="snapshots",
                    )
                    raise InfraConnectionError(
                        "Unexpected NULL result from upsert",
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

        except asyncpg.exceptions.UniqueViolationError as e:
            # UniqueViolationError occurs when:
            # 1. Sequence constraint violated (same subject + sequence, different content)
            # 2. Rare race on content_hash unique index (concurrent identical inserts)
            #
            # For case 2, check if content_hash exists and return it for idempotency.
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

            # Sequence conflict with different content - this is a real conflict
            context = ModelInfraErrorContext(
                transport_type=EnumInfraTransportType.DATABASE,
                operation="save_snapshot",
                target_name="snapshots",
            )
            raise InfraConnectionError(
                f"Unique constraint violation during save for subject "
                f"({snapshot.subject.subject_type}, {snapshot.subject.subject_id}): "
                f"{e.constraint_name or 'unknown constraint'}",
                context=context,
            ) from e

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

    async def load_many(self, snapshot_ids: list[UUID]) -> dict[UUID, ModelSnapshot]:
        """Load multiple snapshots by ID in a single query.

        Uses a batch query with ANY() for efficient multi-row fetch,
        avoiding N+1 query patterns when loading multiple snapshots.

        Args:
            snapshot_ids: List of snapshot UUIDs to load.

        Returns:
            Dictionary mapping snapshot ID to ModelSnapshot for found
            snapshots. Missing IDs are not included in the result.

        Raises:
            InfraConnectionError: If database connection fails.

        Example:
            >>> snapshots = await store.load_many([id1, id2, id3])
            >>> for sid, snap in snapshots.items():
            ...     print(f"{sid}: seq={snap.sequence_number}")
        """
        if not snapshot_ids:
            return {}

        try:
            async with self._pool.acquire() as conn:
                rows = await conn.fetch(
                    "SELECT * FROM snapshots WHERE id = ANY($1::uuid[])",
                    snapshot_ids,
                )
                return {row["id"]: self._row_to_model(row) for row in rows}
        except Exception as e:
            context = ModelInfraErrorContext(
                transport_type=EnumInfraTransportType.DATABASE,
                operation="load_many_snapshots",
                target_name="snapshots",
            )
            raise InfraConnectionError(
                f"Failed to load snapshots: {type(e).__name__}",
                context=context,
            ) from e

    async def load_latest(
        self,
        subject: ModelSubjectRef | None = None,
    ) -> ModelSnapshot | None:
        """Load the most recent snapshot by sequence_number.

        Retrieves the snapshot with the highest sequence_number,
        optionally filtered by subject. "Most recent" is determined
        by sequence_number (not created_at) for consistent ordering.

        Args:
            subject: Optional filter by subject reference.

                - If provided: Returns the latest snapshot for that specific
                  subject (highest sequence_number within that subject).
                - If None: Returns the globally latest snapshot across ALL
                  subjects (highest sequence_number in the entire store).

        Returns:
            The most recent snapshot matching criteria, or None if no
            snapshots exist.

        Raises:
            InfraConnectionError: If database connection fails.

        Note:
            When ``subject=None``, "globally latest" means the snapshot with
            the highest sequence_number across all subjects. Since sequence
            numbers are per-subject (each subject starts at 1), this may NOT
            correspond to the most recently created snapshot by wall-clock
            time. Use ``query(after=timestamp)`` if you need time-based
            ordering across subjects.

        Examples:
            >>> # Get latest for a specific subject
            >>> subject = ModelSubjectRef(
            ...     subject_type="node_registration",
            ...     subject_id=node_uuid,
            ... )
            >>> latest = await store.load_latest(subject=subject)
            >>> # Returns snapshot with highest sequence_number for this subject

            >>> # Get globally latest across ALL subjects
            >>> global_latest = await store.load_latest(subject=None)
            >>> # Returns snapshot with highest sequence_number in entire store
            >>> # Note: This is NOT necessarily the most recent by created_at
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

    async def load_latest_many(
        self,
        subjects: list[ModelSubjectRef],
    ) -> dict[tuple[str, UUID], ModelSnapshot]:
        """Load the latest snapshot for multiple subjects in a single query.

        Uses a window function to efficiently fetch the latest snapshot per
        subject in one database round-trip, avoiding N+1 query patterns.

        Args:
            subjects: List of subject references to load latest snapshots for.

        Returns:
            Dictionary mapping (subject_type, subject_id) tuple to the latest
            ModelSnapshot for that subject. Subjects with no snapshots are
            not included in the result.

        Raises:
            InfraConnectionError: If database connection fails.

        Example:
            >>> subjects = [
            ...     ModelSubjectRef(subject_type="agent", subject_id=agent_id),
            ...     ModelSubjectRef(subject_type="workflow", subject_id=wf_id),
            ... ]
            >>> latest = await store.load_latest_many(subjects)
            >>> for (stype, sid), snap in latest.items():
            ...     print(f"{stype}/{sid}: seq={snap.sequence_number}")
        """
        if not subjects:
            return {}

        try:
            async with self._pool.acquire() as conn:
                # Build arrays for subject_type and subject_id to match
                subject_types = [s.subject_type for s in subjects]
                subject_ids = [s.subject_id for s in subjects]

                # Use window function to get latest per subject in one query
                # The query uses a CTE to rank snapshots per subject, then
                # filters to only keep the top-ranked (latest) per subject.
                rows = await conn.fetch(
                    """
                    WITH ranked AS (
                        SELECT *,
                            ROW_NUMBER() OVER (
                                PARTITION BY subject_type, subject_id
                                ORDER BY sequence_number DESC
                            ) as rn
                        FROM snapshots
                        WHERE (subject_type, subject_id) IN (
                            SELECT * FROM UNNEST($1::text[], $2::uuid[])
                        )
                    )
                    SELECT * FROM ranked WHERE rn = 1
                    """,
                    subject_types,
                    subject_ids,
                )

                return {
                    (row["subject_type"], row["subject_id"]): self._row_to_model(row)
                    for row in rows
                }
        except Exception as e:
            context = ModelInfraErrorContext(
                transport_type=EnumInfraTransportType.DATABASE,
                operation="load_latest_many_snapshots",
                target_name="snapshots",
            )
            raise InfraConnectionError(
                f"Failed to load latest snapshots: {type(e).__name__}",
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
        """Get the next sequence number for a subject with advisory lock.

        Uses PostgreSQL advisory locks to ensure atomic sequence allocation.
        The lock is held only during the MAX() query, allowing concurrent
        access to different subjects while preventing race conditions for
        the same subject.

        Advisory Lock Strategy:
            Uses pg_advisory_xact_lock() with a hash of (subject_type, subject_id)
            to create a subject-specific lock. This ensures:
            - Concurrent calls for the SAME subject serialize
            - Concurrent calls for DIFFERENT subjects proceed in parallel
            - Lock is automatically released at transaction end

        Args:
            subject: The subject reference for sequence generation.

        Returns:
            The next sequence number (starts at 1 for new subjects).

        Raises:
            InfraConnectionError: If database connection fails.

        Note:
            For atomic allocate-and-save operations, prefer save_with_auto_sequence()
            which combines sequence allocation and insert in a single transaction.

        Concurrency Guarantees:
            - No duplicate sequence numbers for the same subject
            - No gaps in sequence numbers (unless deletes occur)
            - Monotonically increasing per subject
        """
        try:
            async with self._pool.acquire() as conn:
                # Use a transaction to hold the advisory lock during the query.
                # The lock key is derived from a hash of subject identifiers.
                # pg_advisory_xact_lock takes a bigint, so we use hashtext()
                # on the concatenated subject identifiers.
                async with conn.transaction():
                    # Acquire advisory lock for this specific subject.
                    # hashtext() returns a stable 32-bit hash; we cast to bigint.
                    # This serializes concurrent calls for the same subject.
                    # Note: subject_id must be converted to str() for text concatenation
                    # in the hashtext() call - asyncpg requires string type for || operator.
                    await conn.execute(
                        """
                        SELECT pg_advisory_xact_lock(
                            hashtext($1 || '::' || $2)::bigint
                        )
                        """,
                        subject.subject_type,
                        str(subject.subject_id),
                    )

                    # Now safely get the next sequence number while holding the lock
                    result = await conn.fetchval(
                        """
                        SELECT COALESCE(MAX(sequence_number), 0) + 1
                        FROM snapshots
                        WHERE subject_type = $1 AND subject_id = $2
                        """,
                        subject.subject_type,
                        subject.subject_id,
                    )
                    # Lock released automatically when transaction ends
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

    async def save_with_auto_sequence(
        self,
        subject: ModelSubjectRef,
        data: dict[str, object],
        *,
        version: int = 1,
        content_hash: str | None = None,
        parent_id: UUID | None = None,
    ) -> tuple[UUID, int]:
        """Atomically allocate sequence number and save snapshot.

        This method combines sequence allocation and insert into a single
        atomic transaction, eliminating the TOCTOU race condition that
        exists when calling get_next_sequence_number() followed by save()
        separately.

        Atomicity Guarantees:
            1. Advisory lock prevents concurrent sequence allocation for same subject
            2. Sequence allocation and INSERT occur in same transaction
            3. If INSERT fails, no sequence number is "consumed"
            4. Content-hash deduplication still applies (returns existing if duplicate)

        Race Condition Handling:
            - Same content_hash: Returns existing snapshot ID and sequence number
            - Concurrent saves for same subject: Serialized via advisory lock
            - Database constraint violations: Wrapped as InfraConnectionError

        Args:
            subject: The subject reference for the snapshot.
            data: The snapshot payload as a JSON-compatible dictionary.
            version: Version number for the snapshot (default 1).
            content_hash: Optional content hash for deduplication. If provided
                and a snapshot with this hash already exists, returns the
                existing snapshot's ID and sequence number.
            parent_id: Optional parent snapshot ID for lineage tracking.

        Returns:
            Tuple of (snapshot_id, sequence_number) for the saved or existing snapshot.

        Raises:
            InfraConnectionError: If database connection fails or
                constraint violation occurs.

        Example:
            >>> subject = ModelSubjectRef(subject_type="agent", subject_id=uuid4())
            >>> snapshot_id, seq_num = await store.save_with_auto_sequence(
            ...     subject=subject,
            ...     data={"status": "active"},
            ...     content_hash="sha256:abc123...",
            ... )
            >>> print(f"Saved snapshot {snapshot_id} with sequence {seq_num}")
        """
        from uuid import uuid4

        snapshot_id = uuid4()
        data_json = json.dumps(data, sort_keys=True)
        created_at = datetime.now(UTC)

        try:
            async with self._pool.acquire() as conn:
                async with conn.transaction():
                    # Step 1: Check for existing content_hash FIRST (before locking).
                    # This avoids unnecessary lock acquisition for duplicates and
                    # eliminates the race condition between check and insert by
                    # performing the check within the same transaction that will
                    # do the insert. The transaction isolation ensures consistency.
                    if content_hash:
                        existing = await conn.fetchrow(
                            """
                            SELECT id, sequence_number FROM snapshots
                            WHERE content_hash = $1
                            """,
                            content_hash,
                        )
                        if existing:
                            existing_id = UUID(str(existing["id"]))
                            existing_seq = int(existing["sequence_number"])
                            logger.debug(
                                "Duplicate snapshot detected via content_hash, "
                                "returning existing ID",
                                extra={
                                    "existing_id": str(existing_id),
                                    "content_hash": content_hash[:16] + "...",
                                },
                            )
                            return existing_id, existing_seq

                    # Step 2: Acquire advisory lock for this subject.
                    # This serializes concurrent saves for the same subject,
                    # preventing race conditions in sequence number allocation.
                    # The lock is held until the transaction commits/rollbacks.
                    # Note: subject_id must be converted to str() for text concatenation
                    # in the hashtext() call - asyncpg requires string type for || operator.
                    await conn.execute(
                        """
                        SELECT pg_advisory_xact_lock(
                            hashtext($1 || '::' || $2)::bigint
                        )
                        """,
                        subject.subject_type,
                        str(subject.subject_id),
                    )

                    # Step 3: Allocate next sequence number while holding lock.
                    # The advisory lock ensures no concurrent transaction can
                    # read the same MAX value for this subject.
                    sequence_number = await conn.fetchval(
                        """
                        SELECT COALESCE(MAX(sequence_number), 0) + 1
                        FROM snapshots
                        WHERE subject_type = $1 AND subject_id = $2
                        """,
                        subject.subject_type,
                        subject.subject_id,
                    )
                    sequence_number = int(sequence_number) if sequence_number else 1

                    # Step 4: Insert with ON CONFLICT for content_hash idempotency.
                    # Even though we checked above, a concurrent transaction might
                    # have committed between our check and lock acquisition (before
                    # the lock was acquired). The ON CONFLICT handles this edge case.
                    if content_hash:
                        result = await conn.fetchrow(
                            """
                            INSERT INTO snapshots (
                                id, subject_type, subject_id, data, sequence_number,
                                version, content_hash, created_at, parent_id
                            ) VALUES ($1, $2, $3, $4::jsonb, $5, $6, $7, $8, $9)
                            ON CONFLICT (content_hash) WHERE content_hash IS NOT NULL
                                DO UPDATE SET id = snapshots.id
                            RETURNING id, sequence_number
                            """,
                            snapshot_id,
                            subject.subject_type,
                            subject.subject_id,
                            data_json,
                            sequence_number,
                            version,
                            content_hash,
                            created_at,
                            parent_id,
                        )
                    else:
                        # No content_hash - insert directly
                        result = await conn.fetchrow(
                            """
                            INSERT INTO snapshots (
                                id, subject_type, subject_id, data, sequence_number,
                                version, content_hash, created_at, parent_id
                            ) VALUES ($1, $2, $3, $4::jsonb, $5, $6, $7, $8, $9)
                            RETURNING id, sequence_number
                            """,
                            snapshot_id,
                            subject.subject_type,
                            subject.subject_id,
                            data_json,
                            sequence_number,
                            version,
                            content_hash,
                            created_at,
                            parent_id,
                        )

                    if result:
                        result_id = UUID(str(result["id"]))
                        result_seq = int(result["sequence_number"])

                        if result_id != snapshot_id:
                            # Existing snapshot returned via ON CONFLICT
                            logger.debug(
                                "Duplicate snapshot detected during insert, "
                                "returning existing ID",
                                extra={
                                    "existing_id": str(result_id),
                                    "sequence_number": result_seq,
                                },
                            )
                        else:
                            logger.debug(
                                "Snapshot saved atomically",
                                extra={
                                    "snapshot_id": str(snapshot_id),
                                    "subject_type": subject.subject_type,
                                    "sequence_number": result_seq,
                                },
                            )
                        return result_id, result_seq

                    # Should never reach here with valid INSERT
                    context = ModelInfraErrorContext(
                        transport_type=EnumInfraTransportType.DATABASE,
                        operation="save_with_auto_sequence",
                        target_name="snapshots",
                    )
                    raise InfraConnectionError(
                        "Unexpected NULL result from atomic insert",
                        context=context,
                    )

        except asyncpg.exceptions.UniqueViolationError as e:
            # This can occur if:
            # 1. Very rare race on content_hash between check and insert
            # 2. Sequence constraint violated (shouldn't happen with advisory lock)
            #
            # For content_hash conflicts, try to return the existing row.
            # Use the same connection pool but a new connection to avoid
            # transaction state issues.
            if content_hash:
                try:
                    async with self._pool.acquire() as recovery_conn:
                        existing = await recovery_conn.fetchrow(
                            "SELECT id, sequence_number FROM snapshots "
                            "WHERE content_hash = $1",
                            content_hash,
                        )
                        if existing:
                            existing_id = UUID(str(existing["id"]))
                            existing_seq = int(existing["sequence_number"])
                            logger.debug(
                                "Race condition resolved: returning existing ID "
                                "after UniqueViolationError",
                                extra={
                                    "existing_id": str(existing_id),
                                    "content_hash": content_hash[:16] + "...",
                                },
                            )
                            return existing_id, existing_seq
                except Exception:
                    pass  # Fall through to re-raise original error

            context = ModelInfraErrorContext(
                transport_type=EnumInfraTransportType.DATABASE,
                operation="save_with_auto_sequence",
                target_name="snapshots",
            )
            raise InfraConnectionError(
                f"Unique constraint violation: {e.constraint_name or 'unknown'}",
                context=context,
            ) from e

        except InfraConnectionError:
            raise

        except Exception as e:
            context = ModelInfraErrorContext(
                transport_type=EnumInfraTransportType.DATABASE,
                operation="save_with_auto_sequence",
                target_name="snapshots",
            )
            raise InfraConnectionError(
                f"Failed to save snapshot atomically: {type(e).__name__}",
                context=context,
            ) from e

    async def cleanup_expired(
        self,
        *,
        max_age_seconds: int | None = None,
        keep_latest_n: int | None = None,
        subject: ModelSubjectRef | None = None,
    ) -> int:
        """Remove expired snapshots based on retention policy.

        Supports multiple retention strategies:
        - Time-based: Delete snapshots older than max_age_seconds
        - Count-based: Keep only the N most recent per subject
        - Subject-scoped: Apply policy only to a specific subject

        When both max_age_seconds and keep_latest_n are provided, snapshots
        must satisfy BOTH conditions to be deleted (i.e., be older than
        max_age AND not in the latest N).

        Args:
            max_age_seconds: Delete snapshots older than this many seconds.
            keep_latest_n: Always retain the N most recent per subject.
            subject: If provided, apply cleanup only to this subject.

        Returns:
            Number of snapshots deleted.

        Raises:
            ProtocolConfigurationError: If keep_latest_n is provided but < 1.
            InfraConnectionError: If database connection fails.

        Note:
            For keep_latest_n, this uses a window function to identify
            snapshots outside the retention window per subject. This is
            efficient for moderate numbers of subjects but may require
            batching for very large datasets.
        """
        if keep_latest_n is not None and keep_latest_n < 1:
            raise ProtocolConfigurationError(
                "keep_latest_n must be >= 1",
                keep_latest_n=keep_latest_n,
            )

        # If neither policy is specified, no-op
        if max_age_seconds is None and keep_latest_n is None:
            return 0

        try:
            async with self._pool.acquire() as conn:
                # Build conditions for deletion
                conditions: list[str] = []
                params: list[object] = []

                # Subject filter (applies to all strategies)
                if subject is not None:
                    conditions.append(f"subject_type = ${len(params) + 1}")
                    params.append(subject.subject_type)
                    conditions.append(f"subject_id = ${len(params) + 1}")
                    params.append(subject.subject_id)

                subject_filter = " AND ".join(conditions) if conditions else "TRUE"

                # Strategy 1: Age-based only (simpler query)
                if max_age_seconds is not None and keep_latest_n is None:
                    cutoff_time = datetime.now(UTC) - timedelta(seconds=max_age_seconds)
                    params.append(cutoff_time)

                    # S608: Safe - subject_filter contains only parameterized placeholders
                    delete_query = f"""
                        DELETE FROM snapshots
                        WHERE {subject_filter}
                        AND created_at < ${len(params)}
                    """  # noqa: S608

                    result = await conn.execute(delete_query, *params)
                    # asyncpg returns "DELETE N"
                    deleted_str = str(result).replace("DELETE ", "")
                    return int(deleted_str) if deleted_str.isdigit() else 0

                # Strategy 2: Keep latest N only
                if keep_latest_n is not None and max_age_seconds is None:
                    params.append(keep_latest_n)

                    # Use window function to rank snapshots per subject
                    # S608: Safe - subject_filter contains only parameterized placeholders
                    delete_query = f"""
                        DELETE FROM snapshots
                        WHERE id IN (
                            SELECT id FROM (
                                SELECT id,
                                    ROW_NUMBER() OVER (
                                        PARTITION BY subject_type, subject_id
                                        ORDER BY sequence_number DESC
                                    ) as rn
                                FROM snapshots
                                WHERE {subject_filter}
                            ) ranked
                            WHERE rn > ${len(params)}
                        )
                    """  # noqa: S608

                    result = await conn.execute(delete_query, *params)
                    deleted_str = str(result).replace("DELETE ", "")
                    return int(deleted_str) if deleted_str.isdigit() else 0

                # Strategy 3: Combined (both age and count)
                # Delete if: older than max_age AND NOT in latest N
                # NOTE: max_age_seconds is validated non-None by strategy check above,
                # but mypy cannot narrow the Optional[float] type through control flow.
                cutoff_time = datetime.now(UTC) - timedelta(
                    seconds=max_age_seconds  # type: ignore[arg-type]  # NOTE: control flow narrowing limitation
                )
                params.append(cutoff_time)
                cutoff_param_idx = len(params)

                params.append(keep_latest_n)
                keep_n_param_idx = len(params)

                # S608: Safe - subject_filter contains only parameterized placeholders
                delete_query = f"""
                    DELETE FROM snapshots
                    WHERE id IN (
                        SELECT id FROM (
                            SELECT id,
                                created_at,
                                ROW_NUMBER() OVER (
                                    PARTITION BY subject_type, subject_id
                                    ORDER BY sequence_number DESC
                                ) as rn
                            FROM snapshots
                            WHERE {subject_filter}
                        ) ranked
                        WHERE rn > ${keep_n_param_idx}
                        AND created_at < ${cutoff_param_idx}
                    )
                """  # noqa: S608

                result = await conn.execute(delete_query, *params)
                deleted_str = str(result).replace("DELETE ", "")
                return int(deleted_str) if deleted_str.isdigit() else 0

        except ProtocolConfigurationError:
            # Re-raise configuration validation errors
            raise
        except Exception as e:
            context = ModelInfraErrorContext(
                transport_type=EnumInfraTransportType.DATABASE,
                operation="cleanup_expired",
                target_name="snapshots",
            )
            raise InfraConnectionError(
                f"Failed to cleanup expired snapshots: {type(e).__name__}",
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

        Schema Design:
            The content_hash column has a UNIQUE partial index to enable
            atomic idempotency checks via INSERT ON CONFLICT. This eliminates
            TOCTOU race conditions that would occur with separate SELECT-then-
            INSERT patterns.

            Constraints:
            - Primary key on id (UUID)
            - Unique constraint on (subject_type, subject_id, sequence_number)
            - Unique partial index on content_hash WHERE content_hash IS NOT NULL

        Raises:
            InfraConnectionError: If schema creation fails.

        Note:
            This method uses multi-statement execution via transaction.
            Each statement is executed separately to work within asyncpg's
            single-statement limitation.

        Migration Note:
            If upgrading from a schema with non-unique idx_snapshots_content_hash,
            the old index will be dropped and replaced with a unique index.
            Ensure no duplicate content_hash values exist before migration.
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

                # Drop old non-unique content_hash index if it exists (for migration).
                # This is safe because CREATE UNIQUE INDEX IF NOT EXISTS will fail
                # if a non-unique index with the same name exists.
                await conn.execute("""
                    DROP INDEX IF EXISTS idx_snapshots_content_hash
                """)

                # Create UNIQUE partial index on content_hash.
                # This enables atomic ON CONFLICT upserts and prevents duplicate
                # content_hash entries, eliminating TOCTOU race conditions.
                await conn.execute("""
                    CREATE UNIQUE INDEX IF NOT EXISTS idx_snapshots_content_hash
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
