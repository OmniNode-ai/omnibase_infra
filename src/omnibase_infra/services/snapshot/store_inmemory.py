# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""In-memory snapshot store for testing.

Provides an in-memory implementation of ProtocolSnapshotStore suitable for
unit tests and development scenarios. This store maintains all data in memory
with no persistence across process restarts.

Features:
    - Content-hash based idempotency on save
    - Sequence number tracking per subject (atomic)
    - asyncio.Lock for coroutine safety
    - Test helpers (clear, count) for easy cleanup

Thread Safety:
    This implementation uses asyncio.Lock which is coroutine-safe but NOT
    thread-safe. For multi-threaded scenarios, use a thread-safe store
    implementation or wrap with appropriate locks.

Related Tickets:
    - OMN-1246: ServiceSnapshot Infrastructure Primitive

Example:
    >>> import asyncio
    >>> from uuid import uuid4
    >>> from omnibase_infra.models.snapshot import ModelSnapshot, ModelSubjectRef
    >>> from omnibase_infra.services.snapshot import StoreSnapshotInMemory
    >>>
    >>> async def demo():
    ...     store = StoreSnapshotInMemory()
    ...     subject = ModelSubjectRef(subject_type="test", subject_id=uuid4())
    ...     seq = await store.get_next_sequence_number(subject)
    ...     snapshot = ModelSnapshot(subject=subject, data={"key": "value"}, sequence_number=seq)
    ...     saved_id = await store.save(snapshot)
    ...     loaded = await store.load(saved_id)
    ...     assert loaded is not None
    ...     assert loaded.data == {"key": "value"}
    >>>
    >>> asyncio.run(demo())
"""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime, timedelta
from uuid import UUID

from omnibase_infra.errors import ProtocolConfigurationError
from omnibase_infra.models.snapshot import ModelSnapshot, ModelSubjectRef


class StoreSnapshotInMemory:
    """In-memory implementation of ProtocolSnapshotStore for testing.

    Provides a fully-functional snapshot store that maintains all data in
    memory. Useful for unit tests where persistence is not required.

    Features:
        - Content-hash based idempotency: Duplicate saves return existing ID
        - Sequence number tracking: Atomic per-subject sequence generation
        - asyncio.Lock: Safe for concurrent coroutine access
        - Test helpers: clear() and count() for test cleanup and assertions

    Attributes:
        _snapshots: Dictionary mapping snapshot IDs to snapshots.
        _sequences: Dictionary mapping subject keys to sequence counters.
        _hash_index: Dictionary mapping content_hash to snapshot ID for O(1) dedup lookup.
        _lock: asyncio.Lock for coroutine-safe operations.

    Example:
        >>> import asyncio
        >>> from uuid import uuid4
        >>> from omnibase_infra.models.snapshot import ModelSnapshot, ModelSubjectRef
        >>>
        >>> async def test_save_load():
        ...     store = StoreSnapshotInMemory()
        ...     subject = ModelSubjectRef(subject_type="agent", subject_id=uuid4())
        ...
        ...     # Get sequence number and create snapshot
        ...     seq = await store.get_next_sequence_number(subject)
        ...     snapshot = ModelSnapshot(subject=subject, data={"status": "active"}, sequence_number=seq)
        ...
        ...     # Save and verify
        ...     saved_id = await store.save(snapshot)
        ...     assert await store.count() == 1
        ...
        ...     # Load and verify
        ...     loaded = await store.load(saved_id)
        ...     assert loaded is not None
        ...     assert loaded.data["status"] == "active"
        ...
        ...     # Cleanup
        ...     store.clear()
        ...     assert store.count() == 0
        >>>
        >>> asyncio.run(test_save_load())
    """

    def __init__(self) -> None:
        """Initialize empty in-memory store."""
        self._snapshots: dict[UUID, ModelSnapshot] = {}
        self._sequences: dict[str, int] = {}  # subject_key -> sequence
        self._hash_index: dict[
            str, UUID
        ] = {}  # content_hash -> snapshot_id (O(1) dedup)
        self._lock = asyncio.Lock()

    async def save(self, snapshot: ModelSnapshot) -> UUID:
        """Save snapshot. Returns existing ID if duplicate content_hash.

        Implements idempotency by checking content_hash before saving.
        If a snapshot with matching content_hash already exists, the
        existing snapshot's ID is returned without creating a duplicate.

        Args:
            snapshot: The snapshot to persist.

        Returns:
            The snapshot ID (either newly saved or existing duplicate).

        Example:
            >>> import asyncio
            >>> from uuid import uuid4
            >>> from omnibase_infra.models.snapshot import ModelSnapshot, ModelSubjectRef
            >>>
            >>> async def test_idempotency():
            ...     store = StoreSnapshotInMemory()
            ...     subject = ModelSubjectRef(subject_type="test", subject_id=uuid4())
            ...
            ...     # Save first snapshot
            ...     snap1 = ModelSnapshot(subject=subject, data={"key": "value"}, sequence_number=1)
            ...     id1 = await store.save(snap1)
            ...
            ...     # Save duplicate (same content_hash)
            ...     snap2 = ModelSnapshot(subject=subject, data={"key": "value"}, sequence_number=2)
            ...     id2 = await store.save(snap2)
            ...
            ...     # Returns existing ID due to content_hash match
            ...     assert id1 == id2
            ...     assert store.count() == 1
            >>>
            >>> asyncio.run(test_idempotency())
        """
        async with self._lock:
            # O(1) duplicate check via hash index
            if snapshot.content_hash and snapshot.content_hash in self._hash_index:
                return self._hash_index[snapshot.content_hash]

            self._snapshots[snapshot.id] = snapshot
            if snapshot.content_hash:
                self._hash_index[snapshot.content_hash] = snapshot.id
            return snapshot.id

    async def load(self, snapshot_id: UUID) -> ModelSnapshot | None:
        """Load snapshot by ID.

        Args:
            snapshot_id: The unique identifier of the snapshot.

        Returns:
            The snapshot if found, None otherwise.

        Example:
            >>> import asyncio
            >>> from uuid import uuid4
            >>> from omnibase_infra.models.snapshot import ModelSnapshot, ModelSubjectRef
            >>>
            >>> async def test_load():
            ...     store = StoreSnapshotInMemory()
            ...     subject = ModelSubjectRef(subject_type="test", subject_id=uuid4())
            ...     snapshot = ModelSnapshot(subject=subject, data={}, sequence_number=1)
            ...
            ...     await store.save(snapshot)
            ...     loaded = await store.load(snapshot.id)
            ...     assert loaded is not None
            ...
            ...     # Non-existent ID returns None
            ...     missing = await store.load(uuid4())
            ...     assert missing is None
            >>>
            >>> asyncio.run(test_load())
        """
        return self._snapshots.get(snapshot_id)

    async def load_many(self, snapshot_ids: list[UUID]) -> dict[UUID, ModelSnapshot]:
        """Load multiple snapshots by ID.

        Uses batch lookup for efficient multi-row fetch.

        Args:
            snapshot_ids: List of snapshot UUIDs to load.

        Returns:
            Dictionary mapping snapshot ID to ModelSnapshot for found
            snapshots. Missing IDs are not included in the result.

        Example:
            >>> import asyncio
            >>> from uuid import uuid4
            >>> from omnibase_infra.models.snapshot import ModelSnapshot, ModelSubjectRef
            >>>
            >>> async def test_load_many():
            ...     store = StoreSnapshotInMemory()
            ...     subject = ModelSubjectRef(subject_type="test", subject_id=uuid4())
            ...     snap1 = ModelSnapshot(subject=subject, data={"v": 1}, sequence_number=1)
            ...     snap2 = ModelSnapshot(subject=subject, data={"v": 2}, sequence_number=2)
            ...     await store.save(snap1)
            ...     await store.save(snap2)
            ...
            ...     results = await store.load_many([snap1.id, snap2.id])
            ...     assert len(results) == 2
            ...     assert results[snap1.id].data["v"] == 1
            >>>
            >>> asyncio.run(test_load_many())
        """
        return {
            sid: self._snapshots[sid] for sid in snapshot_ids if sid in self._snapshots
        }

    async def load_latest(
        self,
        subject: ModelSubjectRef | None = None,
    ) -> ModelSnapshot | None:
        """Get most recent snapshot by sequence_number.

        "Most recent" is determined by sequence_number, not created_at.
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

        Note:
            When ``subject=None``, "globally latest" means the snapshot with
            the highest sequence_number across all subjects. Since sequence
            numbers are per-subject (each subject starts at 1), this may NOT
            correspond to the most recently created snapshot by wall-clock
            time. Use ``query(after=timestamp)`` if you need time-based
            ordering across subjects.

        Example:
            >>> import asyncio
            >>> from uuid import uuid4
            >>> from omnibase_infra.models.snapshot import ModelSnapshot, ModelSubjectRef
            >>>
            >>> async def test_load_latest():
            ...     store = StoreSnapshotInMemory()
            ...     subject = ModelSubjectRef(subject_type="test", subject_id=uuid4())
            ...
            ...     # Save multiple snapshots for the subject
            ...     snap1 = ModelSnapshot(subject=subject, data={"v": 1}, sequence_number=1)
            ...     snap2 = ModelSnapshot(subject=subject, data={"v": 2}, sequence_number=2)
            ...     await store.save(snap1)
            ...     await store.save(snap2)
            ...
            ...     # Get latest for specific subject
            ...     latest = await store.load_latest(subject=subject)
            ...     assert latest is not None
            ...     assert latest.sequence_number == 2
            ...
            ...     # Get globally latest across ALL subjects
            ...     global_latest = await store.load_latest(subject=None)
            ...     # Returns snapshot with highest sequence_number in store
            >>>
            >>> asyncio.run(test_load_latest())
        """
        candidates = list(self._snapshots.values())
        if subject:
            subject_key = subject.to_key()
            candidates = [s for s in candidates if s.subject.to_key() == subject_key]

        if not candidates:
            return None

        return max(candidates, key=lambda s: s.sequence_number)

    async def load_latest_many(
        self,
        subjects: list[ModelSubjectRef],
    ) -> dict[tuple[str, UUID], ModelSnapshot]:
        """Load the latest snapshot for multiple subjects.

        Uses batch lookup for efficient multi-subject query.

        Args:
            subjects: List of subject references to load latest snapshots for.

        Returns:
            Dictionary mapping (subject_type, subject_id) tuple to the latest
            ModelSnapshot for that subject. Subjects with no snapshots are
            not included in the result.

        Example:
            >>> import asyncio
            >>> from uuid import uuid4
            >>> from omnibase_infra.models.snapshot import ModelSnapshot, ModelSubjectRef
            >>>
            >>> async def test_load_latest_many():
            ...     store = StoreSnapshotInMemory()
            ...     s1 = ModelSubjectRef(subject_type="agent", subject_id=uuid4())
            ...     s2 = ModelSubjectRef(subject_type="workflow", subject_id=uuid4())
            ...
            ...     snap1 = ModelSnapshot(subject=s1, data={"v": 1}, sequence_number=1)
            ...     snap2 = ModelSnapshot(subject=s2, data={"v": 2}, sequence_number=1)
            ...     await store.save(snap1)
            ...     await store.save(snap2)
            ...
            ...     results = await store.load_latest_many([s1, s2])
            ...     assert len(results) == 2
            >>>
            >>> asyncio.run(test_load_latest_many())
        """
        if not subjects:
            return {}

        # Group snapshots by subject key
        snapshots_by_subject: dict[str, list[ModelSnapshot]] = {}
        for snapshot in self._snapshots.values():
            key = snapshot.subject.to_key()
            if key not in snapshots_by_subject:
                snapshots_by_subject[key] = []
            snapshots_by_subject[key].append(snapshot)

        # Get latest for each requested subject
        result: dict[tuple[str, UUID], ModelSnapshot] = {}
        for subject in subjects:
            key = subject.to_key()
            candidates = snapshots_by_subject.get(key, [])
            if candidates:
                latest = max(candidates, key=lambda s: s.sequence_number)
                result[(subject.subject_type, subject.subject_id)] = latest

        return result

    async def query(
        self,
        subject: ModelSubjectRef | None = None,
        limit: int = 50,
        after: datetime | None = None,
    ) -> list[ModelSnapshot]:
        """Query with filtering, ordered by sequence_number desc.

        Returns snapshots ordered by sequence_number descending (most
        recent first).

        Args:
            subject: Optional filter by subject reference.
            limit: Maximum results to return (default 50).
            after: Only return snapshots created after this time.

        Returns:
            List of snapshots ordered by sequence_number descending.
            Empty list if no snapshots match criteria.

        Example:
            >>> import asyncio
            >>> from uuid import uuid4
            >>> from omnibase_infra.models.snapshot import ModelSnapshot, ModelSubjectRef
            >>>
            >>> async def test_query():
            ...     store = StoreSnapshotInMemory()
            ...     subject = ModelSubjectRef(subject_type="test", subject_id=uuid4())
            ...
            ...     # Save multiple snapshots
            ...     for i in range(5):
            ...         snap = ModelSnapshot(subject=subject, data={"i": i}, sequence_number=i+1)
            ...         await store.save(snap)
            ...
            ...     # Query with limit
            ...     results = await store.query(subject=subject, limit=3)
            ...     assert len(results) == 3
            ...     # Ordered by sequence_number descending
            ...     assert results[0].sequence_number == 5
            >>>
            >>> asyncio.run(test_query())
        """
        candidates = list(self._snapshots.values())

        if subject:
            subject_key = subject.to_key()
            candidates = [s for s in candidates if s.subject.to_key() == subject_key]

        if after:
            candidates = [s for s in candidates if s.created_at > after]

        # Sort by sequence_number descending
        candidates.sort(key=lambda s: s.sequence_number, reverse=True)

        return candidates[:limit]

    async def delete(self, snapshot_id: UUID) -> bool:
        """Delete snapshot. Returns True if deleted.

        Args:
            snapshot_id: The unique identifier of the snapshot to delete.

        Returns:
            True if the snapshot was deleted, False if not found.

        Example:
            >>> import asyncio
            >>> from uuid import uuid4
            >>> from omnibase_infra.models.snapshot import ModelSnapshot, ModelSubjectRef
            >>>
            >>> async def test_delete():
            ...     store = StoreSnapshotInMemory()
            ...     subject = ModelSubjectRef(subject_type="test", subject_id=uuid4())
            ...     snapshot = ModelSnapshot(subject=subject, data={}, sequence_number=1)
            ...
            ...     await store.save(snapshot)
            ...     assert store.count() == 1
            ...
            ...     deleted = await store.delete(snapshot.id)
            ...     assert deleted is True
            ...     assert store.count() == 0
            ...
            ...     # Deleting non-existent returns False
            ...     deleted_again = await store.delete(snapshot.id)
            ...     assert deleted_again is False
            >>>
            >>> asyncio.run(test_delete())
        """
        async with self._lock:
            if snapshot_id in self._snapshots:
                snapshot = self._snapshots[snapshot_id]
                # Remove from hash index if content_hash exists
                if snapshot.content_hash and snapshot.content_hash in self._hash_index:
                    del self._hash_index[snapshot.content_hash]
                del self._snapshots[snapshot_id]
                return True
            return False

    async def get_next_sequence_number(self, subject: ModelSubjectRef) -> int:
        """Get next sequence number for subject (atomic).

        Sequence numbers are monotonically increasing per subject,
        starting at 1 for new subjects.

        Args:
            subject: The subject reference for which to get the next
                sequence number.

        Returns:
            The next sequence number (starts at 1 for new subjects).

        Example:
            >>> import asyncio
            >>> from uuid import uuid4
            >>> from omnibase_infra.models.snapshot import ModelSubjectRef
            >>>
            >>> async def test_sequence():
            ...     store = StoreSnapshotInMemory()
            ...     subject = ModelSubjectRef(subject_type="test", subject_id=uuid4())
            ...
            ...     seq1 = await store.get_next_sequence_number(subject)
            ...     seq2 = await store.get_next_sequence_number(subject)
            ...     seq3 = await store.get_next_sequence_number(subject)
            ...
            ...     assert seq1 == 1
            ...     assert seq2 == 2
            ...     assert seq3 == 3
            >>>
            >>> asyncio.run(test_sequence())
        """
        async with self._lock:
            key = subject.to_key()
            seq = self._sequences.get(key, 0) + 1
            self._sequences[key] = seq
            return seq

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
        must satisfy BOTH conditions to be deleted.

        Args:
            max_age_seconds: Delete snapshots older than this many seconds.
            keep_latest_n: Always retain the N most recent per subject.
            subject: If provided, apply cleanup only to this subject.

        Returns:
            Number of snapshots deleted.

        Raises:
            ProtocolConfigurationError: If keep_latest_n is provided but < 1.

        Example:
            >>> import asyncio
            >>> from uuid import uuid4
            >>> from datetime import UTC, datetime, timedelta
            >>> from omnibase_infra.models.snapshot import ModelSnapshot, ModelSubjectRef
            >>>
            >>> async def test_cleanup():
            ...     store = StoreSnapshotInMemory()
            ...     subject = ModelSubjectRef(subject_type="test", subject_id=uuid4())
            ...
            ...     # Create old snapshots
            ...     for i in range(10):
            ...         snap = ModelSnapshot(subject=subject, data={"i": i}, sequence_number=i+1)
            ...         await store.save(snap)
            ...
            ...     # Keep only last 5
            ...     deleted = await store.cleanup_expired(keep_latest_n=5)
            ...     assert deleted == 5
            ...     assert store.count() == 5
            >>>
            >>> asyncio.run(test_cleanup())
        """
        if keep_latest_n is not None and keep_latest_n < 1:
            raise ProtocolConfigurationError(
                "keep_latest_n must be >= 1",
                keep_latest_n=keep_latest_n,
            )

        # If neither policy is specified, no-op
        if max_age_seconds is None and keep_latest_n is None:
            return 0

        async with self._lock:
            # Calculate cutoff time for age-based filtering
            cutoff_time: datetime | None = None
            if max_age_seconds is not None:
                cutoff_time = datetime.now(UTC) - timedelta(seconds=max_age_seconds)

            # Group snapshots by subject for keep_latest_n logic
            snapshots_by_subject: dict[str, list[ModelSnapshot]] = {}
            for snapshot in self._snapshots.values():
                subject_key = snapshot.subject.to_key()
                # Filter by subject if specified
                if subject is not None and subject_key != subject.to_key():
                    continue
                if subject_key not in snapshots_by_subject:
                    snapshots_by_subject[subject_key] = []
                snapshots_by_subject[subject_key].append(snapshot)

            # Identify snapshots to delete
            ids_to_delete: set[UUID] = set()

            for subject_key, snapshots in snapshots_by_subject.items():
                # Sort by sequence_number descending (newest first)
                sorted_snapshots = sorted(
                    snapshots,
                    key=lambda s: s.sequence_number,
                    reverse=True,
                )

                for i, snapshot in enumerate(sorted_snapshots):
                    should_delete = False

                    # Check age-based policy
                    if cutoff_time is not None and snapshot.created_at < cutoff_time:
                        should_delete = True

                    # Check count-based policy (skip if in latest N)
                    if keep_latest_n is not None and i < keep_latest_n:
                        # Always keep the latest N, even if old
                        should_delete = False
                    elif keep_latest_n is not None and i >= keep_latest_n:
                        # If using keep_latest_n without max_age, delete excess
                        if max_age_seconds is None:
                            should_delete = True
                        # Otherwise, should_delete is already set by age check

                    if should_delete:
                        ids_to_delete.add(snapshot.id)

            # Perform deletion
            deleted_count = 0
            for snapshot_id in ids_to_delete:
                if snapshot_id in self._snapshots:
                    snapshot = self._snapshots[snapshot_id]
                    # Remove from hash index if content_hash exists
                    if (
                        snapshot.content_hash
                        and snapshot.content_hash in self._hash_index
                    ):
                        del self._hash_index[snapshot.content_hash]
                    del self._snapshots[snapshot_id]
                    deleted_count += 1

            return deleted_count

    # Test helpers

    def clear(self) -> None:
        """Clear all data (for test cleanup).

        Removes all snapshots and resets sequence counters.
        This is a synchronous method for convenient test cleanup.

        Example:
            >>> store = StoreSnapshotInMemory()
            >>> # ... populate store ...
            >>> store.clear()
            >>> assert store.count() == 0
        """
        self._snapshots.clear()
        self._sequences.clear()
        self._hash_index.clear()

    def count(self) -> int:
        """Get total snapshot count.

        Returns the number of snapshots currently stored.
        This is a synchronous method for convenient test assertions.

        Returns:
            Number of snapshots in the store.

        Example:
            >>> store = StoreSnapshotInMemory()
            >>> assert store.count() == 0
        """
        return len(self._snapshots)


__all__: list[str] = ["StoreSnapshotInMemory"]
