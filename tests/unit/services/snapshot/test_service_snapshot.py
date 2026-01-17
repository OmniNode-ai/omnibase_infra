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
        """get_latest() without subject returns snapshot with highest sequence_number.

        Since sequence numbers are per-subject, we create multiple snapshots for
        one subject to ensure it has the highest sequence_number globally.
        """
        subject1 = ModelSubjectRef(subject_type="type_a", subject_id=uuid4())
        subject2 = ModelSubjectRef(subject_type="type_b", subject_id=uuid4())

        # Create 1 snapshot for subject1 (seq=1)
        await service.create(subject=subject1, data={"order": 1})
        # Create 2 snapshots for subject2 (seq=1, then seq=2)
        await service.create(subject=subject2, data={"order": 2})
        id3 = await service.create(subject=subject2, data={"order": 3})

        # Global latest should be the snapshot with highest sequence_number (seq=2)
        latest = await service.get_latest()
        assert latest is not None
        # Verify exact snapshot ID
        assert latest.id == id3
        # Verify sequence_number is the max (2 for subject2)
        assert latest.sequence_number == 2
        # Verify data integrity
        assert latest.data["order"] == 3
        assert latest.subject.subject_type == "type_b"

    @pytest.mark.asyncio
    async def test_get_latest_global_returns_truly_latest_across_subjects(
        self, service: ServiceSnapshot
    ) -> None:
        """get_latest(subject=None) returns the snapshot with highest sequence_number globally.

        This test verifies that when multiple subjects have snapshots,
        the global latest returns the one with the highest sequence_number
        regardless of which subject it belongs to.

        Note: Sequence numbers are per-subject (each subject starts at 1).
        "Global latest" means the snapshot with the highest sequence_number
        across all subjects, NOT the most recently created by wall-clock time.
        """
        subject_a = ModelSubjectRef(subject_type="agent", subject_id=uuid4())
        subject_b = ModelSubjectRef(subject_type="workflow", subject_id=uuid4())
        subject_c = ModelSubjectRef(subject_type="node", subject_id=uuid4())

        # Create snapshots in interleaved order across subjects
        # Sequence numbers are PER-SUBJECT:
        # subject_a: seq 1, 2, 3
        # subject_b: seq 1, 2
        # subject_c: seq 1
        id_a1 = await service.create(subject=subject_a, data={"name": "a1"})  # a: seq=1
        id_b1 = await service.create(subject=subject_b, data={"name": "b1"})  # b: seq=1
        id_a2 = await service.create(subject=subject_a, data={"name": "a2"})  # a: seq=2
        id_c1 = await service.create(subject=subject_c, data={"name": "c1"})  # c: seq=1
        id_b2 = await service.create(subject=subject_b, data={"name": "b2"})  # b: seq=2
        # This should be the global latest (highest seq=3)
        id_a3 = await service.create(subject=subject_a, data={"name": "a3"})  # a: seq=3

        # Get all snapshots to verify sequence numbers
        snap_a1 = await service.get(id_a1)
        snap_b1 = await service.get(id_b1)
        snap_a2 = await service.get(id_a2)
        snap_c1 = await service.get(id_c1)
        snap_b2 = await service.get(id_b2)
        snap_a3 = await service.get(id_a3)

        # Verify all snapshots exist
        assert snap_a1 is not None
        assert snap_b1 is not None
        assert snap_a2 is not None
        assert snap_c1 is not None
        assert snap_b2 is not None
        assert snap_a3 is not None

        # Verify per-subject sequence numbers
        assert snap_a1.sequence_number == 1
        assert snap_a2.sequence_number == 2
        assert snap_a3.sequence_number == 3  # Highest globally
        assert snap_b1.sequence_number == 1
        assert snap_b2.sequence_number == 2
        assert snap_c1.sequence_number == 1

        # Verify global latest
        global_latest = await service.get_latest()
        assert global_latest is not None

        # Verify exact snapshot ID (strongest assertion)
        assert global_latest.id == id_a3
        # Verify sequence_number is the global maximum (3)
        assert global_latest.sequence_number == 3
        assert global_latest.sequence_number == snap_a3.sequence_number
        # Verify it's higher than all other snapshots
        assert global_latest.sequence_number > snap_a1.sequence_number
        assert global_latest.sequence_number > snap_a2.sequence_number
        assert global_latest.sequence_number > snap_b1.sequence_number
        assert global_latest.sequence_number > snap_b2.sequence_number
        assert global_latest.sequence_number > snap_c1.sequence_number
        # Verify data integrity
        assert global_latest.data["name"] == "a3"
        assert global_latest.subject.subject_type == "agent"

    @pytest.mark.asyncio
    async def test_get_latest_global_vs_subject_specific(
        self, service: ServiceSnapshot
    ) -> None:
        """Verify global latest differs from subject-specific latest.

        When different subjects have different latest snapshots, global latest
        should return the one with highest sequence_number across all subjects,
        while subject-specific latest returns the highest for that subject only.

        Note: Sequence numbers are per-subject. To ensure a clear global winner,
        we create more snapshots for one subject (giving it a higher max seq).
        """
        subject_a = ModelSubjectRef(subject_type="type_a", subject_id=uuid4())
        subject_b = ModelSubjectRef(subject_type="type_b", subject_id=uuid4())

        # Create snapshots:
        # subject_a: seq 1, 2 (2 snapshots)
        # subject_b: seq 1, 2, 3 (3 snapshots) - will be global latest
        await service.create(subject=subject_a, data={"v": "a1"})  # a: seq=1
        await service.create(subject=subject_b, data={"v": "b1"})  # b: seq=1
        id_a2 = await service.create(subject=subject_a, data={"v": "a2"})  # a: seq=2
        await service.create(subject=subject_b, data={"v": "b2"})  # b: seq=2
        id_b3 = await service.create(subject=subject_b, data={"v": "b3"})  # b: seq=3

        # Verify sequence numbers
        snap_a2 = await service.get(id_a2)
        snap_b3 = await service.get(id_b3)
        assert snap_a2 is not None
        assert snap_b3 is not None
        assert snap_a2.sequence_number == 2
        assert snap_b3.sequence_number == 3

        # Global latest should be b3 (seq=3 > seq=2)
        global_latest = await service.get_latest()
        assert global_latest is not None
        assert global_latest.id == id_b3
        assert global_latest.sequence_number == 3
        assert global_latest.data["v"] == "b3"

        # Subject-specific latest for subject_a should be a2
        latest_a = await service.get_latest(subject=subject_a)
        assert latest_a is not None
        assert latest_a.id == id_a2
        assert latest_a.sequence_number == 2
        assert latest_a.data["v"] == "a2"

        # Subject-specific latest for subject_b should be b3
        latest_b = await service.get_latest(subject=subject_b)
        assert latest_b is not None
        assert latest_b.id == id_b3
        assert latest_b.sequence_number == 3
        assert latest_b.data["v"] == "b3"

        # Verify global and subject_a latest are different
        assert global_latest.id != latest_a.id
        assert global_latest.sequence_number > latest_a.sequence_number

    @pytest.mark.asyncio
    async def test_get_latest_global_single_snapshot(
        self, service: ServiceSnapshot
    ) -> None:
        """get_latest(subject=None) works correctly with a single snapshot."""
        subject = ModelSubjectRef(subject_type="singleton", subject_id=uuid4())
        only_id = await service.create(subject=subject, data={"only": True})

        global_latest = await service.get_latest()

        assert global_latest is not None
        assert global_latest.id == only_id
        assert global_latest.sequence_number == 1
        assert global_latest.data["only"] is True

    @pytest.mark.asyncio
    async def test_get_latest_global_empty_store(
        self, service: ServiceSnapshot
    ) -> None:
        """get_latest(subject=None) returns None when store is empty."""
        global_latest = await service.get_latest()
        assert global_latest is None

    @pytest.mark.asyncio
    async def test_get_latest_global_tied_sequences_returns_max(
        self, service: ServiceSnapshot
    ) -> None:
        """get_latest(subject=None) with tied max sequences returns one of them.

        When multiple subjects have the same max sequence_number, the returned
        snapshot is implementation-defined (any snapshot with that sequence).
        This test verifies the returned snapshot HAS the max sequence, not which
        specific one is returned.

        Note: Since sequence numbers are per-subject, creating one snapshot per
        subject results in all having sequence_number=1, creating a tie.
        """
        subject_a = ModelSubjectRef(subject_type="type_a", subject_id=uuid4())
        subject_b = ModelSubjectRef(subject_type="type_b", subject_id=uuid4())
        subject_c = ModelSubjectRef(subject_type="type_c", subject_id=uuid4())

        # Create exactly 1 snapshot for each subject (all have seq=1)
        id_a = await service.create(subject=subject_a, data={"name": "a"})
        id_b = await service.create(subject=subject_b, data={"name": "b"})
        id_c = await service.create(subject=subject_c, data={"name": "c"})

        # Get the actual snapshots to verify sequence numbers are tied
        snap_a = await service.get(id_a)
        snap_b = await service.get(id_b)
        snap_c = await service.get(id_c)
        assert snap_a is not None
        assert snap_b is not None
        assert snap_c is not None
        assert snap_a.sequence_number == 1
        assert snap_b.sequence_number == 1
        assert snap_c.sequence_number == 1

        # Global latest should return one of the tied snapshots
        global_latest = await service.get_latest()
        assert global_latest is not None

        # Verify it has the max sequence number (1 in this case)
        assert global_latest.sequence_number == 1
        # Verify it's one of our snapshots (could be any of them)
        assert global_latest.id in {id_a, id_b, id_c}
        # Verify data is from one of the snapshots
        assert global_latest.data["name"] in {"a", "b", "c"}

    @pytest.mark.asyncio
    async def test_get_latest_global_multiple_subjects_different_depths(
        self, service: ServiceSnapshot
    ) -> None:
        """get_latest(subject=None) correctly selects from subjects with different snapshot counts.

        When subjects have different numbers of snapshots, global latest should
        return the snapshot with the highest sequence_number (from the subject
        with the most snapshots).
        """
        subject_shallow = ModelSubjectRef(subject_type="shallow", subject_id=uuid4())
        subject_deep = ModelSubjectRef(subject_type="deep", subject_id=uuid4())

        # Create 1 snapshot for shallow subject (seq=1)
        id_shallow = await service.create(
            subject=subject_shallow, data={"depth": "shallow"}
        )

        # Create 5 snapshots for deep subject (seq=1,2,3,4,5)
        for i in range(4):
            await service.create(subject=subject_deep, data={"depth": "deep", "n": i})
        id_deep_latest = await service.create(
            subject=subject_deep, data={"depth": "deep", "n": 5}
        )

        # Verify sequence numbers
        snap_shallow = await service.get(id_shallow)
        snap_deep = await service.get(id_deep_latest)
        assert snap_shallow is not None
        assert snap_deep is not None
        assert snap_shallow.sequence_number == 1
        assert snap_deep.sequence_number == 5

        # Global latest should be from deep subject (seq=5 > seq=1)
        global_latest = await service.get_latest()
        assert global_latest is not None
        assert global_latest.id == id_deep_latest
        assert global_latest.sequence_number == 5
        assert global_latest.data["depth"] == "deep"
        assert global_latest.data["n"] == 5
        assert global_latest.subject.subject_type == "deep"


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


class TestServiceSnapshotGetMany:
    """Tests for get_many() method - parallel batch loading."""

    @pytest.mark.asyncio
    async def test_get_many_returns_all_snapshots(
        self, service: ServiceSnapshot, subject: ModelSubjectRef
    ) -> None:
        """get_many() returns all requested snapshots."""
        id1 = await service.create(subject=subject, data={"n": 1})
        id2 = await service.create(subject=subject, data={"n": 2})
        id3 = await service.create(subject=subject, data={"n": 3})

        snapshots = await service.get_many([id1, id2, id3])

        assert len(snapshots) == 3
        assert snapshots[0].id == id1
        assert snapshots[1].id == id2
        assert snapshots[2].id == id3

    @pytest.mark.asyncio
    async def test_get_many_preserves_order(
        self, service: ServiceSnapshot, subject: ModelSubjectRef
    ) -> None:
        """get_many() returns snapshots in the same order as requested."""
        id1 = await service.create(subject=subject, data={"n": 1})
        id2 = await service.create(subject=subject, data={"n": 2})
        id3 = await service.create(subject=subject, data={"n": 3})

        # Request in different order
        snapshots = await service.get_many([id3, id1, id2])

        assert snapshots[0].id == id3
        assert snapshots[1].id == id1
        assert snapshots[2].id == id2

    @pytest.mark.asyncio
    async def test_get_many_raises_for_missing_snapshot(
        self, service: ServiceSnapshot, subject: ModelSubjectRef
    ) -> None:
        """get_many() raises SnapshotNotFoundError for missing snapshots."""
        id1 = await service.create(subject=subject, data={"n": 1})
        missing_id = uuid4()

        with pytest.raises(SnapshotNotFoundError) as exc_info:
            await service.get_many([id1, missing_id])

        assert str(missing_id) in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_get_many_skip_missing(
        self, service: ServiceSnapshot, subject: ModelSubjectRef
    ) -> None:
        """get_many() with skip_missing=True skips missing snapshots."""
        id1 = await service.create(subject=subject, data={"n": 1})
        id3 = await service.create(subject=subject, data={"n": 3})
        missing_id = uuid4()

        snapshots = await service.get_many([id1, missing_id, id3], skip_missing=True)

        assert len(snapshots) == 2
        # Only found snapshots are returned
        assert snapshots[0].id == id1
        assert snapshots[1].id == id3

    @pytest.mark.asyncio
    async def test_get_many_empty_list(self, service: ServiceSnapshot) -> None:
        """get_many() returns empty list for empty input."""
        snapshots = await service.get_many([])
        assert snapshots == []

    @pytest.mark.asyncio
    async def test_get_many_single_snapshot(
        self, service: ServiceSnapshot, subject: ModelSubjectRef
    ) -> None:
        """get_many() works with single snapshot."""
        id1 = await service.create(subject=subject, data={"n": 1})

        snapshots = await service.get_many([id1])

        assert len(snapshots) == 1
        assert snapshots[0].id == id1

    @pytest.mark.asyncio
    async def test_get_many_all_missing_raises(self, service: ServiceSnapshot) -> None:
        """get_many() raises for first missing snapshot when all are missing."""
        missing_ids = [uuid4(), uuid4(), uuid4()]

        with pytest.raises(SnapshotNotFoundError):
            await service.get_many(missing_ids)

    @pytest.mark.asyncio
    async def test_get_many_all_missing_skip(self, service: ServiceSnapshot) -> None:
        """get_many() with skip_missing=True returns empty for all missing."""
        missing_ids = [uuid4(), uuid4(), uuid4()]

        snapshots = await service.get_many(missing_ids, skip_missing=True)

        assert snapshots == []


class TestServiceSnapshotGetLatestMany:
    """Tests for get_latest_many() method - parallel latest snapshot loading."""

    @pytest.mark.asyncio
    async def test_get_latest_many_returns_latest_for_each_subject(
        self, service: ServiceSnapshot
    ) -> None:
        """get_latest_many() returns latest snapshot for each subject."""
        subject1 = ModelSubjectRef(subject_type="type_a", subject_id=uuid4())
        subject2 = ModelSubjectRef(subject_type="type_b", subject_id=uuid4())

        # Create multiple snapshots for each subject
        await service.create(subject=subject1, data={"v": 1})
        id1_latest = await service.create(subject=subject1, data={"v": 2})
        await service.create(subject=subject2, data={"v": 10})
        id2_latest = await service.create(subject=subject2, data={"v": 20})

        results = await service.get_latest_many([subject1, subject2])

        assert len(results) == 2
        assert results[0] is not None
        assert results[0].id == id1_latest
        assert results[0].data["v"] == 2
        assert results[1] is not None
        assert results[1].id == id2_latest
        assert results[1].data["v"] == 20

    @pytest.mark.asyncio
    async def test_get_latest_many_preserves_order(
        self, service: ServiceSnapshot
    ) -> None:
        """get_latest_many() returns results in same order as input subjects."""
        subject1 = ModelSubjectRef(subject_type="type_a", subject_id=uuid4())
        subject2 = ModelSubjectRef(subject_type="type_b", subject_id=uuid4())
        subject3 = ModelSubjectRef(subject_type="type_c", subject_id=uuid4())

        await service.create(subject=subject1, data={"s": "a"})
        await service.create(subject=subject2, data={"s": "b"})
        await service.create(subject=subject3, data={"s": "c"})

        # Request in different order
        results = await service.get_latest_many([subject3, subject1, subject2])

        assert len(results) == 3
        assert results[0] is not None
        assert results[0].data["s"] == "c"
        assert results[1] is not None
        assert results[1].data["s"] == "a"
        assert results[2] is not None
        assert results[2].data["s"] == "b"

    @pytest.mark.asyncio
    async def test_get_latest_many_returns_none_for_missing_subjects(
        self, service: ServiceSnapshot
    ) -> None:
        """get_latest_many() returns None for subjects without snapshots."""
        subject1 = ModelSubjectRef(subject_type="type_a", subject_id=uuid4())
        subject2 = ModelSubjectRef(subject_type="type_b", subject_id=uuid4())
        subject3 = ModelSubjectRef(subject_type="type_c", subject_id=uuid4())

        # Only create snapshots for subject1 and subject3
        await service.create(subject=subject1, data={"v": 1})
        await service.create(subject=subject3, data={"v": 3})

        results = await service.get_latest_many([subject1, subject2, subject3])

        assert len(results) == 3
        assert results[0] is not None
        assert results[0].data["v"] == 1
        assert results[1] is None  # subject2 has no snapshots
        assert results[2] is not None
        assert results[2].data["v"] == 3

    @pytest.mark.asyncio
    async def test_get_latest_many_empty_list(self, service: ServiceSnapshot) -> None:
        """get_latest_many() returns empty list for empty input."""
        results = await service.get_latest_many([])
        assert results == []

    @pytest.mark.asyncio
    async def test_get_latest_many_single_subject(
        self, service: ServiceSnapshot, subject: ModelSubjectRef
    ) -> None:
        """get_latest_many() works with single subject."""
        await service.create(subject=subject, data={"n": 1})
        latest_id = await service.create(subject=subject, data={"n": 2})

        results = await service.get_latest_many([subject])

        assert len(results) == 1
        assert results[0] is not None
        assert results[0].id == latest_id

    @pytest.mark.asyncio
    async def test_get_latest_many_all_missing(self, service: ServiceSnapshot) -> None:
        """get_latest_many() returns all None for nonexistent subjects."""
        subjects = [
            ModelSubjectRef(subject_type="missing", subject_id=uuid4()),
            ModelSubjectRef(subject_type="missing", subject_id=uuid4()),
        ]

        results = await service.get_latest_many(subjects)

        assert len(results) == 2
        assert results[0] is None
        assert results[1] is None

    @pytest.mark.asyncio
    async def test_get_latest_many_duplicate_subjects(
        self, service: ServiceSnapshot, subject: ModelSubjectRef
    ) -> None:
        """get_latest_many() handles duplicate subjects correctly."""
        latest_id = await service.create(subject=subject, data={"n": 1})

        # Request same subject twice
        results = await service.get_latest_many([subject, subject])

        assert len(results) == 2
        assert results[0] is not None
        assert results[0].id == latest_id
        assert results[1] is not None
        assert results[1].id == latest_id
