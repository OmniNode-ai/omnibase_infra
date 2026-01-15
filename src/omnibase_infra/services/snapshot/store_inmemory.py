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
from datetime import datetime
from uuid import UUID

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
            # Check for duplicate via content_hash
            if snapshot.content_hash:
                for existing in self._snapshots.values():
                    if existing.content_hash == snapshot.content_hash:
                        return existing.id
            self._snapshots[snapshot.id] = snapshot
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

    async def load_latest(
        self,
        subject: ModelSubjectRef | None = None,
    ) -> ModelSnapshot | None:
        """Get most recent snapshot by sequence_number.

        "Most recent" is determined by sequence_number, not created_at.
        This ensures consistent ordering even with clock skew.

        Args:
            subject: Optional filter by subject reference. If None,
                returns the globally most recent snapshot.

        Returns:
            The most recent snapshot matching criteria, or None if no
            snapshots exist.

        Example:
            >>> import asyncio
            >>> from uuid import uuid4
            >>> from omnibase_infra.models.snapshot import ModelSnapshot, ModelSubjectRef
            >>>
            >>> async def test_load_latest():
            ...     store = StoreSnapshotInMemory()
            ...     subject = ModelSubjectRef(subject_type="test", subject_id=uuid4())
            ...
            ...     # Save multiple snapshots
            ...     snap1 = ModelSnapshot(subject=subject, data={"v": 1}, sequence_number=1)
            ...     snap2 = ModelSnapshot(subject=subject, data={"v": 2}, sequence_number=2)
            ...     await store.save(snap1)
            ...     await store.save(snap2)
            ...
            ...     # Get latest
            ...     latest = await store.load_latest(subject=subject)
            ...     assert latest is not None
            ...     assert latest.sequence_number == 2
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
