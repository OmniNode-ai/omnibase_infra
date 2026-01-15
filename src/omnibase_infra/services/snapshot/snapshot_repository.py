# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Generic Snapshot Repository for Point-in-Time State Capture.

This module provides the SnapshotRepository class, a high-level repository for
managing point-in-time snapshots of entity state. The repository handles:

- Snapshot creation with automatic sequence numbering
- Retrieval and querying with subject filtering
- Structural diffing between snapshots
- Fork operations for creating derived snapshots

Architecture Context:
    SnapshotRepository is the business logic layer for snapshots:
    - Uses ProtocolSnapshotStore for persistence (injectable backend)
    - Manages sequence number generation with locking
    - Provides convenience methods (diff, fork) over raw storage

Thread Safety:
    The service uses asyncio.Lock for sequence number generation, ensuring
    coroutine-safe operations. For multi-process deployments, the store
    implementation must provide atomic sequence generation.

Related Tickets:
    - OMN-1246: SnapshotRepository Infrastructure Primitive
"""

from __future__ import annotations

import asyncio
from datetime import datetime
from typing import TYPE_CHECKING
from uuid import UUID

from omnibase_core.types import JsonType

from omnibase_infra.models.snapshot import (
    ModelSnapshot,
    ModelSnapshotDiff,
    ModelSubjectRef,
)
from omnibase_infra.protocols import ProtocolSnapshotStore

if TYPE_CHECKING:
    from omnibase_core.container import ModelONEXContainer


class SnapshotRepository:
    """Generic snapshot repository with injectable persistence backend.

    Provides a high-level interface for snapshot management, including
    creation, retrieval, diffing, and fork operations. Uses a protocol-
    based store for persistence, allowing different backends (PostgreSQL,
    in-memory, etc.) to be injected.

    Concurrency:
        Uses asyncio.Lock for sequence number generation to prevent
        duplicate sequence numbers within a single process. For
        distributed deployments, the store must provide atomic
        sequence generation.

    Attributes:
        _store: The persistence backend implementing ProtocolSnapshotStore.
        _container: Optional ONEX container for dependency injection.
        _lock: Asyncio lock for coroutine-safe sequence generation.

    Example:
        >>> from omnibase_infra.services.snapshot import SnapshotRepository
        >>> from omnibase_infra.models.snapshot import ModelSubjectRef
        >>>
        >>> # Create repository with in-memory store (for testing)
        >>> store = InMemorySnapshotStore()
        >>> repo = SnapshotRepository(store=store)
        >>>
        >>> # Create a snapshot
        >>> subject = ModelSubjectRef(
        ...     subject_type="agent",
        ...     subject_id="agent-001",
        ... )
        >>> snapshot_id = await repo.create(
        ...     subject=subject,
        ...     data={"status": "active", "config": {"timeout": 30}},
        ... )
        >>>
        >>> # Retrieve latest snapshot
        >>> latest = await repo.get_latest(subject=subject)
        >>> latest.data["status"]
        'active'
    """

    def __init__(
        self,
        store: ProtocolSnapshotStore,
        container: ModelONEXContainer | None = None,
    ) -> None:
        """Initialize the snapshot repository.

        Args:
            store: Persistence backend implementing ProtocolSnapshotStore.
            container: Optional ONEX container for dependency injection.
        """
        self._store = store
        self._container = container
        self._lock = asyncio.Lock()

    async def create(
        self,
        subject: ModelSubjectRef,
        data: dict[str, JsonType],
        *,
        parent_id: UUID | None = None,
    ) -> UUID:
        """Create and persist a new snapshot.

        Creates a snapshot with automatic sequence number assignment and
        content hashing. The sequence number is generated atomically
        using the store's get_next_sequence_number method.

        Args:
            subject: Reference identifying the entity being snapshotted.
            data: The snapshot payload as a JSON-compatible dictionary.
            parent_id: Optional parent snapshot ID for lineage tracking.

        Returns:
            The UUID of the created snapshot.

        Raises:
            InfraConnectionError: If the store is unavailable.
            InfraTimeoutError: If the operation times out.

        Example:
            >>> subject = ModelSubjectRef(
            ...     subject_type="workflow",
            ...     subject_id="wf-123",
            ... )
            >>> snapshot_id = await service.create(
            ...     subject=subject,
            ...     data={"state": "running", "step": 5},
            ... )
        """
        async with self._lock:
            sequence_number = await self._store.get_next_sequence_number(subject)

        content_hash = ModelSnapshot.compute_content_hash(data)

        snapshot = ModelSnapshot(
            subject=subject,
            data=data,
            sequence_number=sequence_number,
            content_hash=content_hash,
            parent_id=parent_id,
        )

        return await self._store.save(snapshot)

    async def get(self, snapshot_id: UUID) -> ModelSnapshot | None:
        """Retrieve a snapshot by ID.

        Args:
            snapshot_id: The unique identifier of the snapshot.

        Returns:
            The snapshot if found, None otherwise.

        Raises:
            InfraConnectionError: If the store is unavailable.
            InfraTimeoutError: If the operation times out.

        Example:
            >>> snapshot = await service.get(snapshot_id)
            >>> if snapshot:
            ...     print(f"Found: {snapshot.data}")
        """
        return await self._store.load(snapshot_id)

    async def get_latest(
        self,
        subject: ModelSubjectRef | None = None,
    ) -> ModelSnapshot | None:
        """Get the most recent snapshot by sequence_number.

        Retrieves the snapshot with the highest sequence number,
        optionally filtered by subject. Note: ordering is by
        sequence_number, not created_at timestamp.

        Args:
            subject: Optional filter by subject reference. If None,
                returns the globally most recent snapshot.

        Returns:
            The most recent snapshot matching criteria, or None if
            no snapshots exist.

        Raises:
            InfraConnectionError: If the store is unavailable.
            InfraTimeoutError: If the operation times out.

        Example:
            >>> subject = ModelSubjectRef(
            ...     subject_type="node",
            ...     subject_id="node-xyz",
            ... )
            >>> latest = await service.get_latest(subject=subject)
            >>> if latest:
            ...     print(f"Sequence: {latest.sequence_number}")
        """
        return await self._store.load_latest(subject)

    async def list(
        self,
        subject: ModelSubjectRef | None = None,
        limit: int = 50,
        after: datetime | None = None,
    ) -> list[ModelSnapshot]:
        """List snapshots with optional filtering and pagination.

        Returns snapshots ordered by sequence_number descending
        (most recent first).

        Args:
            subject: Optional filter by subject reference.
            limit: Maximum number of snapshots to return (default 50).
            after: Only return snapshots created after this time.

        Returns:
            List of snapshots ordered by sequence_number descending.
            Empty list if no snapshots match criteria.

        Raises:
            InfraConnectionError: If the store is unavailable.
            InfraTimeoutError: If the operation times out.

        Example:
            >>> from datetime import datetime, timedelta, UTC
            >>> one_hour_ago = datetime.now(UTC) - timedelta(hours=1)
            >>> recent = await service.list(limit=10, after=one_hour_ago)
            >>> for snap in recent:
            ...     print(f"{snap.id}: seq={snap.sequence_number}")
        """
        return await self._store.query(subject=subject, limit=limit, after=after)

    async def diff(
        self,
        base_id: UUID,
        target_id: UUID,
    ) -> ModelSnapshotDiff:
        """Compute structural diff between two snapshots.

        Performs a shallow structural comparison between the base and
        target snapshots, identifying keys that were added, removed,
        or changed.

        Args:
            base_id: UUID of the base (original) snapshot.
            target_id: UUID of the target (new) snapshot.

        Returns:
            A ModelSnapshotDiff describing the structural differences.

        Raises:
            ValueError: If either snapshot is not found.
            InfraConnectionError: If the store is unavailable.
            InfraTimeoutError: If the operation times out.

        Example:
            >>> diff = await service.diff(base_id=snap1_id, target_id=snap2_id)
            >>> print(f"Added: {diff.added}")
            >>> print(f"Removed: {diff.removed}")
            >>> for key, change in diff.changed.items():
            ...     print(f"{key}: {change.from_value} -> {change.to_value}")
        """
        base = await self._store.load(base_id)
        target = await self._store.load(target_id)

        if base is None:
            raise ValueError(f"Base snapshot not found: {base_id}")
        if target is None:
            raise ValueError(f"Target snapshot not found: {target_id}")

        return ModelSnapshotDiff.compute(
            base_data=base.data,
            target_data=target.data,
            base_id=base_id,
            target_id=target_id,
        )

    async def fork(
        self,
        snapshot_id: UUID,
        mutations: dict[str, JsonType] | None = None,
    ) -> ModelSnapshot:
        """Create a new snapshot from an existing one, with mutations.

        Forks the source snapshot by applying mutations to its data
        and creating a new snapshot with:
        - A new UUID
        - A new sequence number
        - parent_id set to the source snapshot ID
        - Merged data (source data + mutations)

        Args:
            snapshot_id: UUID of the source snapshot to fork.
            mutations: Optional dictionary of changes to apply to the
                source data. If None, creates an exact copy.

        Returns:
            The newly created fork snapshot.

        Raises:
            ValueError: If the source snapshot is not found.
            InfraConnectionError: If the store is unavailable.
            InfraTimeoutError: If the operation times out.

        Example:
            >>> forked = await service.fork(
            ...     snapshot_id=original_id,
            ...     mutations={"status": "paused", "reason": "maintenance"},
            ... )
            >>> forked.parent_id == original_id
            True
            >>> forked.data["status"]
            'paused'
        """
        source = await self._store.load(snapshot_id)
        if source is None:
            raise ValueError(f"Source snapshot not found: {snapshot_id}")

        # Get next sequence number for the subject
        async with self._lock:
            sequence_number = await self._store.get_next_sequence_number(source.subject)

        # Create forked snapshot with mutations applied
        forked = source.with_mutations(
            mutations=mutations or {},
            sequence_number=sequence_number,
        )

        await self._store.save(forked)
        return forked

    async def delete(self, snapshot_id: UUID) -> bool:
        """Delete a snapshot by ID.

        Args:
            snapshot_id: The unique identifier of the snapshot to delete.

        Returns:
            True if the snapshot was deleted, False if not found.

        Raises:
            InfraConnectionError: If the store is unavailable.
            InfraTimeoutError: If the operation times out.

        Example:
            >>> deleted = await service.delete(old_snapshot_id)
            >>> if deleted:
            ...     print("Snapshot removed")
            >>> else:
            ...     print("Snapshot not found")
        """
        return await self._store.delete(snapshot_id)


__all__: list[str] = ["SnapshotRepository"]
