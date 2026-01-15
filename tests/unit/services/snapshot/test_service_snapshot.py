# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Unit tests for ServiceSnapshot.

Tests the ServiceSnapshot using the in-memory store backend.
Covers all service methods: create, get, get_latest, list, diff, fork, delete.

Related Tickets:
    - OMN-1246: ServiceSnapshot Infrastructure Primitive
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from uuid import UUID, uuid4

import pytest
from omnibase_core.container import ModelONEXContainer

from omnibase_infra.models.snapshot import (
    ModelFieldChange,
    ModelSnapshot,
    ModelSnapshotDiff,
    ModelSubjectRef,
)
from omnibase_infra.services.snapshot import (
    ServiceSnapshot,
    SnapshotNotFoundError,
    StoreSnapshotInMemory,
)


@pytest.fixture
def container() -> ModelONEXContainer:
    """Create ONEX container for dependency injection."""
    return ModelONEXContainer()


@pytest.fixture
def store() -> StoreSnapshotInMemory:
    """Create fresh in-memory store for each test."""
    return StoreSnapshotInMemory()


@pytest.fixture
def service(
    store: StoreSnapshotInMemory, container: ModelONEXContainer
) -> ServiceSnapshot:
    """Create service with in-memory backend."""
    return ServiceSnapshot(store=store, container=container)


@pytest.fixture
def subject() -> ModelSubjectRef:
    """Create a test subject reference."""
    return ModelSubjectRef(subject_type="test", subject_id=uuid4())


class TestServiceSnapshotCreate:
    """Tests for create() method."""

    @pytest.mark.asyncio
    async def test_create_returns_uuid(
        self, service: ServiceSnapshot, subject: ModelSubjectRef
    ) -> None:
        """create() returns a UUID."""
        snapshot_id = await service.create(
            subject=subject,
            data={"key": "value"},
        )
        assert isinstance(snapshot_id, UUID)

    @pytest.mark.asyncio
    async def test_create_persists_data(
        self, service: ServiceSnapshot, subject: ModelSubjectRef
    ) -> None:
        """create() persists the snapshot data."""
        data = {"status": "active", "count": 42}
        snapshot_id = await service.create(subject=subject, data=data)

        snapshot = await service.get(snapshot_id)
        assert snapshot is not None
        assert snapshot.data == data

    @pytest.mark.asyncio
    async def test_create_assigns_sequence_number(
        self, service: ServiceSnapshot, subject: ModelSubjectRef
    ) -> None:
        """create() assigns monotonically increasing sequence numbers."""
        id1 = await service.create(subject=subject, data={"n": 1})
        id2 = await service.create(subject=subject, data={"n": 2})

        s1 = await service.get(id1)
        s2 = await service.get(id2)

        assert s1 is not None
        assert s2 is not None
        assert s1.sequence_number < s2.sequence_number

    @pytest.mark.asyncio
    async def test_create_starts_sequence_at_one(
        self, service: ServiceSnapshot, subject: ModelSubjectRef
    ) -> None:
        """create() starts sequence numbers at 1 for new subjects."""
        snapshot_id = await service.create(subject=subject, data={"first": True})

        snapshot = await service.get(snapshot_id)
        assert snapshot is not None
        assert snapshot.sequence_number == 1

    @pytest.mark.asyncio
    async def test_create_with_parent_id(
        self, service: ServiceSnapshot, subject: ModelSubjectRef
    ) -> None:
        """create() supports parent_id for lineage tracking."""
        parent_id = await service.create(subject=subject, data={"parent": True})
        child_id = await service.create(
            subject=subject,
            data={"child": True},
            parent_id=parent_id,
        )

        child = await service.get(child_id)
        assert child is not None
        assert child.parent_id == parent_id

    @pytest.mark.asyncio
    async def test_create_stores_subject_reference(
        self, service: ServiceSnapshot, subject: ModelSubjectRef
    ) -> None:
        """create() stores the subject reference correctly."""
        snapshot_id = await service.create(subject=subject, data={"test": True})

        snapshot = await service.get(snapshot_id)
        assert snapshot is not None
        assert snapshot.subject.subject_type == subject.subject_type
        assert snapshot.subject.subject_id == subject.subject_id

    @pytest.mark.asyncio
    async def test_create_computes_content_hash(
        self, service: ServiceSnapshot, subject: ModelSubjectRef
    ) -> None:
        """create() computes content hash for the snapshot."""
        snapshot_id = await service.create(subject=subject, data={"hash": "test"})

        snapshot = await service.get(snapshot_id)
        assert snapshot is not None
        assert snapshot.content_hash is not None
        assert len(snapshot.content_hash) == 64  # SHA-256 hex length


class TestServiceSnapshotGet:
    """Tests for get() method."""

    @pytest.mark.asyncio
    async def test_get_returns_snapshot(
        self, service: ServiceSnapshot, subject: ModelSubjectRef
    ) -> None:
        """get() returns the snapshot for valid ID."""
        snapshot_id = await service.create(subject=subject, data={"test": True})
        snapshot = await service.get(snapshot_id)
        assert snapshot is not None
        assert snapshot.id == snapshot_id

    @pytest.mark.asyncio
    async def test_get_returns_none_for_unknown_id(
        self, service: ServiceSnapshot
    ) -> None:
        """get() returns None for unknown ID."""
        result = await service.get(uuid4())
        assert result is None

    @pytest.mark.asyncio
    async def test_get_returns_complete_snapshot(
        self, service: ServiceSnapshot, subject: ModelSubjectRef
    ) -> None:
        """get() returns snapshot with all fields populated."""
        data = {"key": "value", "count": 42}
        snapshot_id = await service.create(subject=subject, data=data)

        snapshot = await service.get(snapshot_id)
        assert snapshot is not None
        assert snapshot.id == snapshot_id
        assert snapshot.data == data
        assert snapshot.sequence_number >= 1
        assert snapshot.content_hash is not None
        assert snapshot.created_at is not None


class TestServiceSnapshotGetLatest:
    """Tests for get_latest() method."""

    @pytest.mark.asyncio
    async def test_get_latest_returns_highest_sequence(
        self, service: ServiceSnapshot, subject: ModelSubjectRef
    ) -> None:
        """get_latest() returns snapshot with highest sequence_number."""
        await service.create(subject=subject, data={"n": 1})
        await service.create(subject=subject, data={"n": 2})
        latest_id = await service.create(subject=subject, data={"n": 3})

        latest = await service.get_latest(subject=subject)
        assert latest is not None
        assert latest.id == latest_id
        assert latest.data["n"] == 3

    @pytest.mark.asyncio
    async def test_get_latest_filters_by_subject(
        self, service: ServiceSnapshot
    ) -> None:
        """get_latest() filters by subject when provided."""
        subject1 = ModelSubjectRef(subject_type="type_a", subject_id=uuid4())
        subject2 = ModelSubjectRef(subject_type="type_b", subject_id=uuid4())

        await service.create(subject=subject1, data={"s": 1})
        id2 = await service.create(subject=subject2, data={"s": 2})

        latest = await service.get_latest(subject=subject2)
        assert latest is not None
        assert latest.id == id2
        assert latest.data["s"] == 2

    @pytest.mark.asyncio
    async def test_get_latest_returns_none_when_empty(
        self, service: ServiceSnapshot, subject: ModelSubjectRef
    ) -> None:
        """get_latest() returns None when no snapshots exist."""
        result = await service.get_latest(subject=subject)
        assert result is None

    @pytest.mark.asyncio
    async def test_get_latest_without_subject_returns_global_latest(
        self, service: ServiceSnapshot
    ) -> None:
        """get_latest() without subject returns globally latest snapshot."""
        subject1 = ModelSubjectRef(subject_type="type_a", subject_id=uuid4())
        subject2 = ModelSubjectRef(subject_type="type_b", subject_id=uuid4())

        await service.create(subject=subject1, data={"order": 1})
        await service.create(subject=subject2, data={"order": 2})

        # Global latest should have the highest sequence number
        latest = await service.get_latest()
        assert latest is not None


class TestServiceSnapshotList:
    """Tests for list() method."""

    @pytest.mark.asyncio
    async def test_list_returns_all_matching(
        self, service: ServiceSnapshot, subject: ModelSubjectRef
    ) -> None:
        """list() returns all snapshots for subject."""
        for i in range(5):
            await service.create(subject=subject, data={"n": i})

        results = await service.list(subject=subject)
        assert len(results) == 5

    @pytest.mark.asyncio
    async def test_list_respects_limit(
        self, service: ServiceSnapshot, subject: ModelSubjectRef
    ) -> None:
        """list() respects the limit parameter."""
        for i in range(10):
            await service.create(subject=subject, data={"n": i})

        results = await service.list(subject=subject, limit=3)
        assert len(results) == 3

    @pytest.mark.asyncio
    async def test_list_ordered_by_sequence_desc(
        self, service: ServiceSnapshot, subject: ModelSubjectRef
    ) -> None:
        """list() returns results ordered by sequence_number descending."""
        for i in range(5):
            await service.create(subject=subject, data={"n": i})

        results = await service.list(subject=subject)
        sequences = [s.sequence_number for s in results]
        assert sequences == sorted(sequences, reverse=True)

    @pytest.mark.asyncio
    async def test_list_filters_by_subject(self, service: ServiceSnapshot) -> None:
        """list() filters results by subject when provided."""
        subject1 = ModelSubjectRef(subject_type="type_a", subject_id=uuid4())
        subject2 = ModelSubjectRef(subject_type="type_b", subject_id=uuid4())

        for i in range(3):
            await service.create(subject=subject1, data={"s1": i})
        for i in range(2):
            await service.create(subject=subject2, data={"s2": i})

        results1 = await service.list(subject=subject1)
        results2 = await service.list(subject=subject2)

        assert len(results1) == 3
        assert len(results2) == 2

    @pytest.mark.asyncio
    async def test_list_returns_empty_when_no_matches(
        self, service: ServiceSnapshot, subject: ModelSubjectRef
    ) -> None:
        """list() returns empty list when no snapshots match."""
        results = await service.list(subject=subject)
        assert results == []

    @pytest.mark.asyncio
    async def test_list_with_after_filter(
        self, service: ServiceSnapshot, subject: ModelSubjectRef
    ) -> None:
        """list() filters by created_at when after parameter is provided."""
        # Create a snapshot
        await service.create(subject=subject, data={"old": True})

        # Use a timestamp in the future
        future_time = datetime.now(UTC) + timedelta(hours=1)

        # Should return empty since no snapshots after future_time
        results = await service.list(subject=subject, after=future_time)
        assert results == []


class TestServiceSnapshotDiff:
    """Tests for diff() method."""

    @pytest.mark.asyncio
    async def test_diff_computes_added_keys(
        self, service: ServiceSnapshot, subject: ModelSubjectRef
    ) -> None:
        """diff() identifies added keys."""
        id1 = await service.create(subject=subject, data={"a": 1})
        id2 = await service.create(subject=subject, data={"a": 1, "b": 2})

        diff = await service.diff(base_id=id1, target_id=id2)
        assert "b" in diff.added

    @pytest.mark.asyncio
    async def test_diff_computes_removed_keys(
        self, service: ServiceSnapshot, subject: ModelSubjectRef
    ) -> None:
        """diff() identifies removed keys."""
        id1 = await service.create(subject=subject, data={"a": 1, "b": 2})
        id2 = await service.create(subject=subject, data={"a": 1})

        diff = await service.diff(base_id=id1, target_id=id2)
        assert "b" in diff.removed

    @pytest.mark.asyncio
    async def test_diff_computes_changed_keys(
        self, service: ServiceSnapshot, subject: ModelSubjectRef
    ) -> None:
        """diff() identifies changed values."""
        id1 = await service.create(subject=subject, data={"a": 1})
        id2 = await service.create(subject=subject, data={"a": 2})

        diff = await service.diff(base_id=id1, target_id=id2)
        assert "a" in diff.changed
        assert diff.changed["a"].from_value == 1
        assert diff.changed["a"].to_value == 2

    @pytest.mark.asyncio
    async def test_diff_returns_empty_for_identical_data(
        self, service: ServiceSnapshot, subject: ModelSubjectRef
    ) -> None:
        """diff() returns empty diff for identical data."""
        id1 = await service.create(subject=subject, data={"a": 1, "b": 2})
        id2 = await service.create(subject=subject, data={"a": 1, "b": 2})

        diff = await service.diff(base_id=id1, target_id=id2)
        assert diff.is_empty()
        assert diff.added == []
        assert diff.removed == []
        assert diff.changed == {}

    @pytest.mark.asyncio
    async def test_diff_raises_for_unknown_base(
        self, service: ServiceSnapshot, subject: ModelSubjectRef
    ) -> None:
        """diff() raises SnapshotNotFoundError for unknown base_id."""
        target_id = await service.create(subject=subject, data={"a": 1})

        with pytest.raises(SnapshotNotFoundError, match="Base snapshot not found"):
            await service.diff(base_id=uuid4(), target_id=target_id)

    @pytest.mark.asyncio
    async def test_diff_raises_for_unknown_target(
        self, service: ServiceSnapshot, subject: ModelSubjectRef
    ) -> None:
        """diff() raises SnapshotNotFoundError for unknown target_id."""
        base_id = await service.create(subject=subject, data={"a": 1})

        with pytest.raises(SnapshotNotFoundError, match="Target snapshot not found"):
            await service.diff(base_id=base_id, target_id=uuid4())

    @pytest.mark.asyncio
    async def test_diff_contains_correct_ids(
        self, service: ServiceSnapshot, subject: ModelSubjectRef
    ) -> None:
        """diff() contains the correct base_id and target_id."""
        id1 = await service.create(subject=subject, data={"a": 1})
        id2 = await service.create(subject=subject, data={"a": 2})

        diff = await service.diff(base_id=id1, target_id=id2)
        assert diff.base_id == id1
        assert diff.target_id == id2

    @pytest.mark.asyncio
    async def test_diff_with_complex_nested_changes(
        self, service: ServiceSnapshot, subject: ModelSubjectRef
    ) -> None:
        """diff() handles nested value changes."""
        id1 = await service.create(
            subject=subject, data={"config": {"timeout": 30, "retries": 3}}
        )
        id2 = await service.create(
            subject=subject, data={"config": {"timeout": 60, "retries": 3}}
        )

        diff = await service.diff(base_id=id1, target_id=id2)
        assert "config" in diff.changed
        assert diff.changed["config"].from_value == {"timeout": 30, "retries": 3}
        assert diff.changed["config"].to_value == {"timeout": 60, "retries": 3}


class TestServiceSnapshotFork:
    """Tests for fork() method."""

    @pytest.mark.asyncio
    async def test_fork_creates_new_snapshot(
        self, service: ServiceSnapshot, subject: ModelSubjectRef
    ) -> None:
        """fork() creates a new snapshot from existing."""
        source_id = await service.create(subject=subject, data={"a": 1})
        forked = await service.fork(snapshot_id=source_id)

        assert forked.id != source_id
        assert forked.parent_id == source_id

    @pytest.mark.asyncio
    async def test_fork_applies_mutations(
        self, service: ServiceSnapshot, subject: ModelSubjectRef
    ) -> None:
        """fork() applies mutations to forked data."""
        source_id = await service.create(subject=subject, data={"a": 1, "b": 2})
        forked = await service.fork(
            snapshot_id=source_id,
            mutations={"b": 3, "c": 4},
        )

        assert forked.data["a"] == 1  # Unchanged
        assert forked.data["b"] == 3  # Mutated
        assert forked.data["c"] == 4  # Added

    @pytest.mark.asyncio
    async def test_fork_without_mutations_copies_data(
        self, service: ServiceSnapshot, subject: ModelSubjectRef
    ) -> None:
        """fork() without mutations creates exact copy of data."""
        source_id = await service.create(subject=subject, data={"a": 1, "b": 2})
        forked = await service.fork(snapshot_id=source_id)

        assert forked.data == {"a": 1, "b": 2}

    @pytest.mark.asyncio
    async def test_fork_raises_for_unknown_source(
        self, service: ServiceSnapshot
    ) -> None:
        """fork() raises SnapshotNotFoundError for unknown source."""
        with pytest.raises(SnapshotNotFoundError, match="Source snapshot not found"):
            await service.fork(snapshot_id=uuid4())

    @pytest.mark.asyncio
    async def test_fork_increments_sequence_number(
        self, service: ServiceSnapshot, subject: ModelSubjectRef
    ) -> None:
        """fork() assigns a new sequence number to forked snapshot."""
        source_id = await service.create(subject=subject, data={"a": 1})
        source = await service.get(source_id)
        assert source is not None

        forked = await service.fork(snapshot_id=source_id)
        assert forked.sequence_number > source.sequence_number

    @pytest.mark.asyncio
    async def test_fork_persists_to_store(
        self, service: ServiceSnapshot, subject: ModelSubjectRef
    ) -> None:
        """fork() persists the forked snapshot to the store."""
        source_id = await service.create(subject=subject, data={"a": 1})
        forked = await service.fork(snapshot_id=source_id, mutations={"a": 2})

        # Should be retrievable
        loaded = await service.get(forked.id)
        assert loaded is not None
        assert loaded.data["a"] == 2

    @pytest.mark.asyncio
    async def test_fork_preserves_subject(
        self, service: ServiceSnapshot, subject: ModelSubjectRef
    ) -> None:
        """fork() preserves the subject from the source snapshot."""
        source_id = await service.create(subject=subject, data={"a": 1})
        forked = await service.fork(snapshot_id=source_id)

        assert forked.subject.subject_type == subject.subject_type
        assert forked.subject.subject_id == subject.subject_id


class TestServiceSnapshotDelete:
    """Tests for delete() method."""

    @pytest.mark.asyncio
    async def test_delete_returns_true_when_found(
        self, service: ServiceSnapshot, subject: ModelSubjectRef
    ) -> None:
        """delete() returns True when snapshot deleted."""
        snapshot_id = await service.create(subject=subject, data={"a": 1})
        result = await service.delete(snapshot_id)
        assert result is True

    @pytest.mark.asyncio
    async def test_delete_returns_false_when_not_found(
        self, service: ServiceSnapshot
    ) -> None:
        """delete() returns False when snapshot not found."""
        result = await service.delete(uuid4())
        assert result is False

    @pytest.mark.asyncio
    async def test_delete_removes_snapshot(
        self, service: ServiceSnapshot, subject: ModelSubjectRef
    ) -> None:
        """delete() actually removes the snapshot."""
        snapshot_id = await service.create(subject=subject, data={"a": 1})
        await service.delete(snapshot_id)

        result = await service.get(snapshot_id)
        assert result is None

    @pytest.mark.asyncio
    async def test_delete_does_not_affect_other_snapshots(
        self, service: ServiceSnapshot, subject: ModelSubjectRef
    ) -> None:
        """delete() does not affect other snapshots."""
        id1 = await service.create(subject=subject, data={"n": 1})
        id2 = await service.create(subject=subject, data={"n": 2})

        await service.delete(id1)

        # id2 should still exist
        snapshot2 = await service.get(id2)
        assert snapshot2 is not None
        assert snapshot2.data["n"] == 2


class TestServiceSnapshotEdgeCases:
    """Tests for edge cases and special scenarios."""

    @pytest.mark.asyncio
    async def test_empty_data_snapshot(
        self, service: ServiceSnapshot, subject: ModelSubjectRef
    ) -> None:
        """Service handles snapshots with empty data."""
        snapshot_id = await service.create(subject=subject, data={})

        snapshot = await service.get(snapshot_id)
        assert snapshot is not None
        assert snapshot.data == {}

    @pytest.mark.asyncio
    async def test_complex_nested_data(
        self, service: ServiceSnapshot, subject: ModelSubjectRef
    ) -> None:
        """Service handles complex nested data structures."""
        complex_data = {
            "string": "value",
            "number": 42,
            "float": 3.14,
            "boolean": True,
            "null": None,
            "array": [1, 2, 3],
            "nested": {
                "deep": {
                    "value": "found",
                },
            },
        }
        snapshot_id = await service.create(subject=subject, data=complex_data)

        snapshot = await service.get(snapshot_id)
        assert snapshot is not None
        assert snapshot.data == complex_data

    @pytest.mark.asyncio
    async def test_multiple_subjects_isolation(self, service: ServiceSnapshot) -> None:
        """Snapshots are properly isolated between subjects."""
        subject1 = ModelSubjectRef(subject_type="agent", subject_id=uuid4())
        subject2 = ModelSubjectRef(subject_type="workflow", subject_id=uuid4())

        await service.create(subject=subject1, data={"s1": 1})
        await service.create(subject=subject1, data={"s1": 2})
        await service.create(subject=subject2, data={"s2": 1})

        list1 = await service.list(subject=subject1)
        list2 = await service.list(subject=subject2)

        assert len(list1) == 2
        assert len(list2) == 1

        latest1 = await service.get_latest(subject=subject1)
        latest2 = await service.get_latest(subject=subject2)

        assert latest1 is not None
        assert latest2 is not None
        assert latest1.data == {"s1": 2}
        assert latest2.data == {"s2": 1}

    @pytest.mark.asyncio
    async def test_sequence_numbers_isolated_by_subject(
        self, service: ServiceSnapshot
    ) -> None:
        """Sequence numbers are isolated per subject."""
        subject1 = ModelSubjectRef(subject_type="type_a", subject_id=uuid4())
        subject2 = ModelSubjectRef(subject_type="type_b", subject_id=uuid4())

        # Create snapshots for both subjects
        id1 = await service.create(subject=subject1, data={"a": 1})
        id2 = await service.create(subject=subject2, data={"b": 1})

        snap1 = await service.get(id1)
        snap2 = await service.get(id2)

        assert snap1 is not None
        assert snap2 is not None
        # Both should start at 1 since they're different subjects
        assert snap1.sequence_number == 1
        assert snap2.sequence_number == 1
