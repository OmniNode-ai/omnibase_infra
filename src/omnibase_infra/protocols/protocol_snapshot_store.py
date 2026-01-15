# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Protocol definition for snapshot storage backends.

This module defines the ProtocolSnapshotStore interface for backends that
provide persistent storage for snapshots. Implementations may use PostgreSQL,
in-memory stores, or other storage mechanisms.

Architecture Context:
    In the ONEX snapshot system:
    - ServiceSnapshot handles business logic (idempotency, sequencing)
    - ProtocolSnapshotStore defines the storage contract
    - Concrete implementations (PostgreSQL, in-memory) provide persistence

    This separation enables:
    - Testing with in-memory stores
    - Swapping storage backends without changing service logic
    - Clear boundaries for error handling and retry logic

Ordering Guarantees:
    Snapshots maintain ordering via sequence_number (not created_at):
    - sequence_number is monotonically increasing per subject
    - get_next_sequence_number provides atomic sequence generation
    - load_latest returns highest sequence_number, not most recent time

Idempotency:
    Implementations should handle duplicate saves via content_hash:
    - If a snapshot with matching content_hash exists, return existing ID
    - This enables safe retries without creating duplicates

Example Usage:
    ```python
    from omnibase_infra.protocols import ProtocolSnapshotStore
    from omnibase_infra.models.snapshot import ModelSnapshot, ModelSubjectRef

    class PostgresSnapshotStore:
        '''Concrete implementation using PostgreSQL.'''

        async def save(self, snapshot: ModelSnapshot) -> UUID:
            '''Persist snapshot to PostgreSQL.'''
            # Check for duplicate via content_hash
            existing = await self._find_by_hash(snapshot.content_hash)
            if existing:
                return existing.id

            # Insert with next sequence number
            return await self._insert(snapshot)

        async def load(self, snapshot_id: UUID) -> ModelSnapshot | None:
            '''Load snapshot by ID.'''
            row = await self._db.fetchrow(
                "SELECT * FROM snapshots WHERE id = $1",
                snapshot_id,
            )
            return ModelSnapshot.model_validate(row) if row else None

    # Protocol conformance check via duck typing (per ONEX conventions)
    store = PostgresSnapshotStore()
    assert hasattr(store, 'save') and callable(store.save)
    assert hasattr(store, 'load') and callable(store.load)
    assert hasattr(store, 'load_latest') and callable(store.load_latest)
    assert hasattr(store, 'query') and callable(store.query)
    assert hasattr(store, 'delete') and callable(store.delete)
    assert hasattr(store, 'get_next_sequence_number') and callable(
        store.get_next_sequence_number
    )
    ```

Error Handling:
    All methods should raise OnexError subclasses on failure:
    - InfraConnectionError: Database/storage unavailable
    - InfraTimeoutError: Operation timed out
    - ProtocolConfigurationError: Invalid configuration

See Also:
    - OMN-1246: ServiceSnapshot implementation
    - omnibase_infra.models.snapshot for model definitions
    - omnibase_infra.services.snapshot for repository logic
"""

from __future__ import annotations

from datetime import datetime
from typing import TYPE_CHECKING, Protocol, runtime_checkable
from uuid import UUID

if TYPE_CHECKING:
    from omnibase_infra.models.snapshot import ModelSnapshot, ModelSubjectRef

__all__ = [
    "ProtocolSnapshotStore",
]


@runtime_checkable
class ProtocolSnapshotStore(Protocol):
    """Backend persistence protocol for snapshot storage.

    Implementations provide concrete storage (PostgreSQL, in-memory, etc.)
    while ServiceSnapshot handles business logic.

    Ordering Guarantees:
        - sequence_number is the canonical ordering field
        - load_latest returns highest sequence_number, not created_at
        - get_next_sequence_number provides atomic sequence generation

    Idempotency:
        - save() should deduplicate via content_hash if provided
        - Return existing ID when duplicate detected

    Concurrency Safety:
        Implementations must be coroutine-safe for concurrent async operations.
        Multiple coroutines may invoke methods concurrently. Use asyncio.Lock
        for shared mutable state (coroutine-safe, not thread-safe).

    Error Handling:
        All methods should raise OnexError subclasses on failure:
        - InfraConnectionError: Storage unavailable
        - InfraTimeoutError: Operation timed out
        - ProtocolConfigurationError: Invalid configuration

    Example Implementation:
        ```python
        class InMemorySnapshotStore:
            def __init__(self) -> None:
                self._snapshots: dict[UUID, ModelSnapshot] = {}
                self._sequences: dict[str, int] = {}
                self._lock = asyncio.Lock()

            async def save(self, snapshot: ModelSnapshot) -> UUID:
                async with self._lock:
                    # Check for duplicate via content_hash
                    for existing in self._snapshots.values():
                        if existing.content_hash == snapshot.content_hash:
                            return existing.id
                    self._snapshots[snapshot.id] = snapshot
                    return snapshot.id

            async def load(self, snapshot_id: UUID) -> ModelSnapshot | None:
                return self._snapshots.get(snapshot_id)

            async def get_next_sequence_number(
                self, subject: ModelSubjectRef
            ) -> int:
                async with self._lock:
                    key = f"{subject.subject_type}:{subject.subject_id}"
                    seq = self._sequences.get(key, 0) + 1
                    self._sequences[key] = seq
                    return seq
        ```
    """

    async def save(self, snapshot: ModelSnapshot) -> UUID:
        """Persist a snapshot. Returns snapshot ID.

        Implementations should:
        - Handle idempotency via content_hash if provided
        - Enforce sequence_number ordering per subject
        - Return existing ID if duplicate detected

        Args:
            snapshot: The snapshot to persist. Must have a valid ID.

        Returns:
            The snapshot ID (either newly saved or existing duplicate).

        Raises:
            InfraConnectionError: If storage is unavailable
            InfraTimeoutError: If operation times out
            OnexError: For configuration or serialization errors

        Example:
            ```python
            from uuid import uuid4, UUID

            # subject_id must be a UUID (not str)
            node_id: UUID = uuid4()
            state_dict: dict = {"status": "active", "version": "1.0.0"}

            snapshot = ModelSnapshot(
                id=uuid4(),
                subject=ModelSubjectRef(
                    subject_type="node_registration",
                    subject_id=node_id,  # Must be UUID, not str
                ),
                data=state_dict,
                sequence_number=42,
                # content_hash is computed automatically from data
            )
            saved_id = await store.save(snapshot)
            ```

        Implementation Notes:
            - Check content_hash for duplicates before insert
            - Use database constraints for sequence ordering
            - Include correlation_id in logs for tracing
        """
        ...

    async def load(self, snapshot_id: UUID) -> ModelSnapshot | None:
        """Load a snapshot by ID. Returns None if not found.

        Args:
            snapshot_id: The unique identifier of the snapshot.

        Returns:
            The snapshot if found, None otherwise.

        Raises:
            InfraConnectionError: If storage is unavailable
            InfraTimeoutError: If operation times out

        Example:
            ```python
            snapshot = await store.load(snapshot_id)
            if snapshot is None:
                logger.warning(f"Snapshot {snapshot_id} not found")
                return None
            return snapshot.data
            ```
        """
        ...

    async def load_many(self, snapshot_ids: list[UUID]) -> dict[UUID, ModelSnapshot]:
        """Load multiple snapshots by ID in a single query.

        Uses batch loading for efficient multi-row fetch, avoiding N+1
        query patterns when loading multiple snapshots.

        Args:
            snapshot_ids: List of snapshot UUIDs to load.

        Returns:
            Dictionary mapping snapshot ID to ModelSnapshot for found
            snapshots. Missing IDs are not included in the result.

        Raises:
            InfraConnectionError: If storage is unavailable
            InfraTimeoutError: If operation times out

        Example:
            ```python
            snapshots = await store.load_many([id1, id2, id3])
            for sid, snap in snapshots.items():
                print(f"{sid}: seq={snap.sequence_number}")
            ```

        Implementation Notes:
            - Use batch query (e.g., WHERE id = ANY($1))
            - Avoid N separate queries
            - Return dict for O(1) lookup by ID
        """
        ...

    async def load_latest(
        self,
        subject: ModelSubjectRef | None = None,
    ) -> ModelSnapshot | None:
        """Load most recent snapshot, optionally filtered by subject.

        "Most recent" is determined by sequence_number (not created_at).
        This ensures consistent ordering even with clock skew.

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
            InfraConnectionError: If storage is unavailable
            InfraTimeoutError: If operation times out

        Note:
            When ``subject=None``, "globally latest" means the snapshot with
            the highest sequence_number across all subjects. Since sequence
            numbers are per-subject (each subject starts at 1), this may NOT
            correspond to the most recently created snapshot by wall-clock
            time. Use ``query(after=timestamp)`` if you need time-based
            ordering across subjects.

        Warning:
            The ``subject=None`` behavior returns a snapshot based on
            sequence_number comparison across different subjects. This is
            useful for "get any recent snapshot" scenarios but NOT for
            "get the most recently saved snapshot" which requires time-based
            queries. Example of potentially surprising behavior:

            - Subject A has snapshots with sequence_number 1, 2, 3
            - Subject B has a single snapshot with sequence_number 1
            - Subject B's snapshot was saved 1 minute ago
            - Subject A's latest (seq=3) was saved 1 hour ago
            - ``load_latest(subject=None)`` returns Subject A's snapshot (seq=3)
            - To get Subject B's snapshot (most recent by time), use:
              ``query(after=one_minute_ago, limit=1)``

        Examples:
            ```python
            from uuid import uuid4, UUID
            from datetime import datetime, timedelta, UTC

            # Get latest snapshot for a specific node
            node_id: UUID = uuid4()  # subject_id must be UUID, not str
            subject = ModelSubjectRef(
                subject_type="node_registration",
                subject_id=node_id,  # Must be UUID, not str
            )
            latest = await store.load_latest(subject=subject)
            # Returns snapshot with highest sequence_number for this subject

            # Get globally latest snapshot across ALL subjects
            global_latest = await store.load_latest(subject=None)
            # Returns snapshot with highest sequence_number in entire store
            # Note: This is NOT necessarily the most recent by created_at
            # because sequence_number is per-subject (each subject starts at 1)

            # Alternative: Get most recent by wall-clock time
            one_second_ago = datetime.now(UTC) - timedelta(seconds=1)
            recent = await store.query(after=one_second_ago, limit=1)
            time_based_latest = recent[0] if recent else None
            ```

        Implementation Notes:
            - Order by sequence_number DESC, not created_at
            - Use index on (subject_type, subject_id, sequence_number)
            - LIMIT 1 for efficiency
        """
        ...

    async def load_latest_many(
        self,
        subjects: list[ModelSubjectRef],
    ) -> dict[tuple[str, UUID], ModelSnapshot]:
        """Load the latest snapshot for multiple subjects in a single query.

        Uses batch loading to efficiently fetch the latest snapshot per
        subject in one database round-trip, avoiding N+1 query patterns.

        Args:
            subjects: List of subject references to load latest snapshots for.

        Returns:
            Dictionary mapping (subject_type, subject_id) tuple to the latest
            ModelSnapshot for that subject. Subjects with no snapshots are
            not included in the result.

        Raises:
            InfraConnectionError: If storage is unavailable
            InfraTimeoutError: If operation times out

        Example:
            ```python
            subjects = [
                ModelSubjectRef(subject_type="agent", subject_id=agent_id),
                ModelSubjectRef(subject_type="workflow", subject_id=wf_id),
            ]
            latest = await store.load_latest_many(subjects)
            for (stype, sid), snap in latest.items():
                print(f"{stype}/{sid}: seq={snap.sequence_number}")
            ```

        Implementation Notes:
            - Use window function to get latest per subject in one query
            - Avoid N separate queries
            - Return dict keyed by (subject_type, subject_id) tuple
        """
        ...

    async def query(
        self,
        subject: ModelSubjectRef | None = None,
        limit: int = 50,
        after: datetime | None = None,
    ) -> list[ModelSnapshot]:
        """Query snapshots with optional filtering.

        Returns snapshots ordered by sequence_number descending (most
        recent first).

        Args:
            subject: Optional filter by subject reference.
            limit: Maximum results to return (default 50).
            after: Only return snapshots created after this time.

        Returns:
            List of snapshots ordered by sequence_number descending.
            Empty list if no snapshots match criteria.

        Raises:
            InfraConnectionError: If storage is unavailable
            InfraTimeoutError: If operation times out

        Example:
            ```python
            from uuid import uuid4, UUID
            from datetime import datetime, timedelta, UTC

            # Get last 10 snapshots for a subject
            node_id: UUID = uuid4()  # subject_id must be UUID, not str
            subject = ModelSubjectRef(
                subject_type="node_registration",
                subject_id=node_id,  # Must be UUID, not str
            )
            snapshots = await store.query(subject=subject, limit=10)

            # Get snapshots created in the last hour
            one_hour_ago = datetime.now(UTC) - timedelta(hours=1)
            recent = await store.query(after=one_hour_ago, limit=100)
            ```

        Implementation Notes:
            - Always order by sequence_number DESC
            - Apply subject filter before limit
            - Use pagination for large result sets
        """
        ...

    async def delete(self, snapshot_id: UUID) -> bool:
        """Delete a snapshot by ID. Returns True if deleted, False if not found.

        Use Case: Clean up old snapshots for data retention or when a
        snapshot is superseded and no longer needed.

        Args:
            snapshot_id: The unique identifier of the snapshot to delete.

        Returns:
            True if the snapshot was deleted, False if not found.

        Raises:
            InfraConnectionError: If storage is unavailable
            InfraTimeoutError: If operation times out

        Example:
            ```python
            deleted = await store.delete(old_snapshot_id)
            if deleted:
                logger.info(f"Deleted snapshot {old_snapshot_id}")
            else:
                logger.warning(f"Snapshot {old_snapshot_id} not found")
            ```

        Implementation Notes:
            - Use soft delete if audit trail is required
            - Consider cascade behavior for related data
            - Include correlation_id in logs for tracing
        """
        ...

    async def get_next_sequence_number(self, subject: ModelSubjectRef) -> int:
        """Get the next sequence number for a subject.

        Used to maintain ordering guarantees. Sequence numbers are
        monotonically increasing per subject.

        Args:
            subject: The subject reference for which to get the next
                sequence number.

        Returns:
            The next sequence number (starts at 1 for new subjects).

        Raises:
            InfraConnectionError: If storage is unavailable
            InfraTimeoutError: If operation times out

        Example:
            ```python
            from uuid import uuid4, UUID

            node_id: UUID = uuid4()  # subject_id must be UUID, not str
            subject = ModelSubjectRef(
                subject_type="node_registration",
                subject_id=node_id,  # Must be UUID, not str
            )
            seq = await store.get_next_sequence_number(subject)
            # seq is guaranteed to be greater than any existing
            # sequence_number for this subject
            ```

        Implementation Notes:
            - Must be atomic (no gaps, no duplicates)
            - Use database sequences or SELECT MAX + 1 with locking
            - Consider high-concurrency scenarios
        """
        ...

    async def cleanup_expired(
        self,
        *,
        max_age_seconds: int | None = None,
        keep_latest_n: int | None = None,
        subject: ModelSubjectRef | None = None,
    ) -> int:
        """Remove expired snapshots based on retention policy.

        Supports multiple retention strategies that can be combined:
        - Time-based: Delete snapshots older than max_age_seconds
        - Count-based: Keep only the N most recent per subject
        - Subject-scoped: Apply policy only to a specific subject

        When both max_age_seconds and keep_latest_n are provided, snapshots
        must satisfy BOTH conditions to be deleted (i.e., be older than
        max_age AND not in the latest N).

        Args:
            max_age_seconds: Delete snapshots with created_at older than
                this many seconds ago. If None, no age-based filtering.
            keep_latest_n: Always retain the N most recent snapshots per
                subject (by sequence_number). If None, no count-based
                retention. Must be >= 1 if provided.
            subject: If provided, apply cleanup only to this subject.
                If None, apply cleanup globally across all subjects.

        Returns:
            Number of snapshots deleted.

        Raises:
            InfraConnectionError: If storage is unavailable
            InfraTimeoutError: If operation times out
            ProtocolConfigurationError: If keep_latest_n is provided but < 1

        Example:
            ```python
            # Delete snapshots older than 30 days
            deleted = await store.cleanup_expired(
                max_age_seconds=30 * 24 * 60 * 60,
            )

            # Keep only last 10 snapshots per subject
            deleted = await store.cleanup_expired(
                keep_latest_n=10,
            )

            # Combined: Delete if older than 7 days AND not in latest 5
            deleted = await store.cleanup_expired(
                max_age_seconds=7 * 24 * 60 * 60,
                keep_latest_n=5,
            )

            # Cleanup only for a specific subject
            from uuid import uuid4, UUID
            agent_id: UUID = uuid4()  # subject_id must be UUID, not str
            subject = ModelSubjectRef(
                subject_type="agent",
                subject_id=agent_id,  # Must be UUID, not str
            )
            deleted = await store.cleanup_expired(
                max_age_seconds=60 * 60,  # 1 hour
                subject=subject,
            )
            ```

        Implementation Notes:
            - At least one of max_age_seconds or keep_latest_n should be provided
            - If neither is provided, return 0 (no-op)
            - For keep_latest_n, order by sequence_number (not created_at)
            - Use batch deletes for efficiency in large datasets
            - Consider adding index on created_at for time-based queries
        """
        ...
