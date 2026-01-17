# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Generic Snapshot Service for Point-in-Time State Capture.

This module provides the ServiceSnapshot class, a high-level service for
managing point-in-time snapshots of entity state. The service handles:

- Snapshot creation with automatic sequence numbering
- Retrieval and querying with subject filtering
- Structural diffing between snapshots
- Fork operations for creating derived snapshots

Architecture Context:
    ServiceSnapshot is the business logic layer for snapshots:
    - Uses ProtocolSnapshotStore for persistence (injectable backend)
    - Manages sequence number generation with locking
    - Provides convenience methods (diff, fork) over raw storage

Thread Safety:
    The service uses asyncio.Lock for sequence number generation, ensuring
    coroutine-safe operations. For multi-process deployments, the store
    implementation must provide atomic sequence generation.

Related Tickets:
    - OMN-1246: ServiceSnapshot Infrastructure Primitive
"""

from __future__ import annotations

import asyncio
import builtins
from datetime import datetime
from uuid import UUID

from omnibase_core.container import ModelONEXContainer
from omnibase_core.enums import EnumCoreErrorCode
from omnibase_core.types import JsonType
from omnibase_infra.errors import ProtocolConfigurationError, RuntimeHostError
from omnibase_infra.models.snapshot import (
    ModelSnapshot,
    ModelSnapshotDiff,
    ModelSubjectRef,
)
from omnibase_infra.protocols import ProtocolSnapshotStore


class SnapshotNotFoundError(RuntimeHostError):
    """Raised when a requested snapshot does not exist.

    Used when diff or fork operations reference a snapshot ID that
    cannot be found in the store. This indicates a caller error
    (invalid ID) rather than an infrastructure failure.

    Example:
        >>> raise SnapshotNotFoundError(
        ...     "Base snapshot not found",
        ...     snapshot_id=base_id,
        ... )
    """

    def __init__(
        self,
        message: str,
        snapshot_id: UUID | None = None,
        **extra_context: object,
    ) -> None:
        """Initialize SnapshotNotFoundError.

        Args:
            message: Human-readable error message
            snapshot_id: The UUID of the snapshot that was not found
            **extra_context: Additional context information
        """
        if snapshot_id is not None:
            extra_context["snapshot_id"] = str(snapshot_id)
        super().__init__(
            message=message,
            error_code=EnumCoreErrorCode.RESOURCE_NOT_FOUND,
            context=None,
            **extra_context,
        )


class ServiceSnapshot:
    """Generic snapshot service with injectable persistence backend.

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
        _container: ONEX container for dependency injection.
        _lock: Asyncio lock for coroutine-safe sequence generation.

    Example:
        >>> from omnibase_core.container import ModelONEXContainer
        >>> from omnibase_infra.services.snapshot import ServiceSnapshot
        >>> from omnibase_infra.services.snapshot import StoreSnapshotInMemory
        >>> from omnibase_infra.models.snapshot import ModelSubjectRef
        >>>
        >>> # Create service with in-memory store (for testing)
        >>> store = StoreSnapshotInMemory()
        >>> container = ModelONEXContainer()
        >>> service = ServiceSnapshot(store=store, container=container)
        >>>
        >>> # Create a snapshot
        >>> subject = ModelSubjectRef(
        ...     subject_type="agent",
        ...     subject_id="agent-001",
        ... )
        >>> snapshot_id = await service.create(
        ...     subject=subject,
        ...     data={"status": "active", "config": {"timeout": 30}},
        ... )
        >>>
        >>> # Retrieve latest snapshot
        >>> latest = await service.get_latest(subject=subject)
        >>> latest.data["status"]
        'active'
    """

    def __init__(
        self,
        store: ProtocolSnapshotStore,
        container: ModelONEXContainer,
    ) -> None:
        """Initialize the snapshot service.

        Args:
            store: Persistence backend implementing ProtocolSnapshotStore.
            container: ONEX container for dependency injection.
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
            subject: Optional filter by subject reference.

                - If provided: Returns the latest snapshot for that specific
                  subject (highest sequence_number within that subject).
                - If None: Returns the globally latest snapshot across ALL
                  subjects (highest sequence_number in the entire store).

        Returns:
            The most recent snapshot matching criteria, or None if no
            snapshots exist.

        Raises:
            InfraConnectionError: If the store is unavailable.
            InfraTimeoutError: If the operation times out.

        Note:
            When ``subject=None``, "globally latest" means the snapshot with
            the highest sequence_number across all subjects. Since sequence
            numbers are per-subject (each subject starts at 1), this may NOT
            correspond to the most recently created snapshot by wall-clock
            time. Use ``list(after=timestamp)`` if you need time-based
            ordering across subjects.

        Examples:
            >>> # Get latest for a specific subject
            >>> subject = ModelSubjectRef(
            ...     subject_type="node",
            ...     subject_id="node-xyz",
            ... )
            >>> latest = await service.get_latest(subject=subject)
            >>> if latest:
            ...     print(f"Sequence: {latest.sequence_number}")

            >>> # Get globally latest across ALL subjects
            >>> global_latest = await service.get_latest(subject=None)
            >>> # Returns snapshot with highest sequence_number in store
            >>> # Note: This is NOT necessarily the most recent by created_at

            >>> # Alternative: Get most recent by wall-clock time
            >>> from datetime import datetime, timedelta, UTC
            >>> one_second_ago = datetime.now(UTC) - timedelta(seconds=1)
            >>> recent = await service.list(after=one_second_ago, limit=1)
            >>> time_based_latest = recent[0] if recent else None
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

    async def get_many(
        self,
        snapshot_ids: builtins.list[UUID],
        *,
        skip_missing: bool = False,
    ) -> builtins.list[ModelSnapshot]:
        """Load multiple snapshots in a single batch query.

        Uses batch loading for efficient multi-row fetch, avoiding N+1
        query patterns when loading multiple snapshots.

        Args:
            snapshot_ids: List of snapshot UUIDs to load.
            skip_missing: If True, missing snapshots are silently skipped.
                If False (default), raises SnapshotNotFoundError for any
                missing snapshot.

        Returns:
            List of snapshots in the same order as snapshot_ids (when
            skip_missing=False). When skip_missing=True, the returned
            list may be shorter and only contains found snapshots.

        Raises:
            SnapshotNotFoundError: If any snapshot is not found and
                skip_missing is False.
            InfraConnectionError: If the store is unavailable.
            InfraTimeoutError: If the operation times out.

        Example:
            >>> # Load multiple snapshots at once
            >>> snapshots = await service.get_many([id1, id2, id3])
            >>> assert len(snapshots) == 3
            >>>
            >>> # Skip missing snapshots
            >>> snapshots = await service.get_many(
            ...     [id1, unknown_id, id3],
            ...     skip_missing=True,
            ... )
            >>> # Returns only found snapshots
        """
        if not snapshot_ids:
            return []

        # Use batch load for single database round-trip
        results_dict = await self._store.load_many(snapshot_ids)

        snapshots: builtins.list[ModelSnapshot] = []
        for snapshot_id in snapshot_ids:
            snapshot = results_dict.get(snapshot_id)
            if snapshot is None:
                if not skip_missing:
                    raise SnapshotNotFoundError(
                        f"Snapshot not found: {snapshot_id}",
                        snapshot_id=snapshot_id,
                    )
                # skip_missing=True: silently skip
                continue

            snapshots.append(snapshot)

        return snapshots

    async def get_latest_many(
        self,
        subjects: builtins.list[ModelSubjectRef],
    ) -> builtins.list[ModelSnapshot | None]:
        """Load the latest snapshot for multiple subjects in a single batch query.

        Uses batch loading for efficient multi-row fetch, avoiding N+1
        query patterns when querying multiple subjects.

        Args:
            subjects: List of subject references to load latest snapshots for.

        Returns:
            List of snapshots (or None) in the same order as the input
            subjects. Each entry is the latest snapshot for that subject,
            or None if no snapshots exist for that subject.

        Raises:
            InfraConnectionError: If the store is unavailable.
            InfraTimeoutError: If the operation times out.

        Example:
            >>> subjects = [
            ...     ModelSubjectRef(subject_type="agent", subject_id=agent_id),
            ...     ModelSubjectRef(subject_type="workflow", subject_id=workflow_id),
            ... ]
            >>> snapshots = await service.get_latest_many(subjects)
            >>> for subject, snapshot in zip(subjects, snapshots):
            ...     if snapshot:
            ...         print(f"{subject.subject_id}: seq={snapshot.sequence_number}")
        """
        if not subjects:
            return []

        # Use batch load for single database round-trip
        results_dict = await self._store.load_latest_many(subjects)

        # Build result list maintaining input order
        snapshots: builtins.list[ModelSnapshot | None] = []
        for subject in subjects:
            key = (subject.subject_type, subject.subject_id)
            snapshots.append(results_dict.get(key))

        return snapshots

    async def diff(
        self,
        base_id: UUID,
        target_id: UUID,
    ) -> ModelSnapshotDiff:
        """Compute structural diff between two snapshots.

        Performs a shallow structural comparison between the base and
        target snapshots, identifying keys that were added, removed,
        or changed.

        Both snapshots are loaded in parallel for better performance.

        Args:
            base_id: UUID of the base (original) snapshot.
            target_id: UUID of the target (new) snapshot.

        Returns:
            A ModelSnapshotDiff describing the structural differences.

        Raises:
            SnapshotNotFoundError: If either snapshot is not found.
            InfraConnectionError: If the store is unavailable.
            InfraTimeoutError: If the operation times out.

        Example:
            >>> diff = await service.diff(base_id=snap1_id, target_id=snap2_id)
            >>> print(f"Added: {diff.added}")
            >>> print(f"Removed: {diff.removed}")
            >>> for key, change in diff.changed.items():
            ...     print(f"{key}: {change.from_value} -> {change.to_value}")
        """
        # Load both snapshots in parallel for better performance
        base_task = self._store.load(base_id)
        target_task = self._store.load(target_id)
        base, target = await asyncio.gather(base_task, target_task)

        if base is None:
            raise SnapshotNotFoundError(
                f"Base snapshot not found: {base_id}",
                snapshot_id=base_id,
            )
        if target is None:
            raise SnapshotNotFoundError(
                f"Target snapshot not found: {target_id}",
                snapshot_id=target_id,
            )

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
            SnapshotNotFoundError: If the source snapshot is not found.
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
            raise SnapshotNotFoundError(
                f"Source snapshot not found: {snapshot_id}",
                snapshot_id=snapshot_id,
            )

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

    async def cleanup_expired(
        self,
        *,
        max_age_seconds: int | None = None,
        keep_latest_n: int | None = None,
        subject: ModelSubjectRef | None = None,
    ) -> int:
        """Remove expired snapshots based on retention policy.

        Provides a high-level interface for snapshot cleanup with multiple
        retention strategies that can be combined:

        - **Time-based** (max_age_seconds): Delete snapshots older than
          the specified age. Useful for compliance or storage cost control.
        - **Count-based** (keep_latest_n): Retain only the N most recent
          snapshots per subject. Useful for audit trail limits.
        - **Subject-scoped** (subject): Apply cleanup only to a specific
          subject. Useful for per-entity retention policies.

        When both max_age_seconds and keep_latest_n are provided, snapshots
        must satisfy BOTH conditions to be deleted. This means:
        - Snapshots in the latest N per subject are ALWAYS retained
        - Snapshots outside the latest N are deleted ONLY if also older
          than max_age_seconds

        This dual-policy approach provides a safety net: even if you
        configure aggressive age-based cleanup, you'll always retain
        recent history for each subject.

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
            ProtocolConfigurationError: If keep_latest_n is provided but < 1.
            InfraConnectionError: If the store is unavailable.
            InfraTimeoutError: If the operation times out.

        Example:
            >>> from datetime import timedelta
            >>> from omnibase_infra.models.snapshot import ModelSubjectRef
            >>>
            >>> # Delete snapshots older than 30 days
            >>> deleted = await service.cleanup_expired(
            ...     max_age_seconds=30 * 24 * 60 * 60,
            ... )
            >>> print(f"Deleted {deleted} old snapshots")
            >>>
            >>> # Keep only last 10 snapshots per subject
            >>> deleted = await service.cleanup_expired(
            ...     keep_latest_n=10,
            ... )
            >>> print(f"Trimmed {deleted} excess snapshots")
            >>>
            >>> # Combined: Delete if older than 7 days AND not in latest 5
            >>> deleted = await service.cleanup_expired(
            ...     max_age_seconds=7 * 24 * 60 * 60,
            ...     keep_latest_n=5,
            ... )
            >>>
            >>> # Cleanup only for a specific subject
            >>> subject = ModelSubjectRef(
            ...     subject_type="workflow",
            ...     subject_id=workflow_id,
            ... )
            >>> deleted = await service.cleanup_expired(
            ...     max_age_seconds=60 * 60,  # 1 hour
            ...     subject=subject,
            ... )

        Scheduling Recommendations:
            This method is typically called from a scheduled task or cron job:

            - **Daily cleanup**: Run once per day with conservative settings
              (e.g., max_age_seconds=30 days, keep_latest_n=100)
            - **Aggressive cleanup**: For high-volume systems, run hourly
              with tighter limits (e.g., max_age_seconds=7 days, keep_latest_n=10)
            - **Per-subject cleanup**: Call during subject lifecycle events
              (e.g., when a workflow completes, clean up its old snapshots)

        Performance Notes:
            For very large datasets, consider:
            - Running cleanup during low-traffic periods
            - Using subject-scoped cleanup in batches
            - Adding an index on created_at if using time-based cleanup frequently
        """
        return await self._store.cleanup_expired(
            max_age_seconds=max_age_seconds,
            keep_latest_n=keep_latest_n,
            subject=subject,
        )


__all__: list[str] = ["SnapshotNotFoundError", "ServiceSnapshot"]
