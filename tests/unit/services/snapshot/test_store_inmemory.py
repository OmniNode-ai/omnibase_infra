# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Unit tests for StoreSnapshotInMemory.

Tests the in-memory snapshot store implementation directly, covering
all ProtocolSnapshotStore methods and test helper methods.

Related Tickets:
    - OMN-1246: SnapshotRepository Infrastructure Primitive
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from uuid import UUID, uuid4

import pytest

from omnibase_infra.models.snapshot import ModelSnapshot, ModelSubjectRef
from omnibase_infra.services.snapshot import StoreSnapshotInMemory


@pytest.fixture
def store() -> StoreSnapshotInMemory:
    """Create fresh in-memory store for each test."""
    return StoreSnapshotInMemory()


@pytest.fixture
def subject() -> ModelSubjectRef:
    """Create a test subject reference."""
    return ModelSubjectRef(subject_type="test", subject_id=uuid4())


class TestStoreSnapshotInMemorySave:
    """Tests for save() method."""

    @pytest.mark.asyncio
    async def test_save_returns_snapshot_id(
        self, store: StoreSnapshotInMemory, subject: ModelSubjectRef
    ) -> None:
        """save() returns the snapshot ID."""
        snapshot = ModelSnapshot(
            subject=subject, data={"key": "value"}, sequence_number=1
        )
        saved_id = await store.save(snapshot)
        assert saved_id == snapshot.id

    @pytest.mark.asyncio
    async def test_save_idempotent_on_content_hash(
        self, store: StoreSnapshotInMemory, subject: ModelSubjectRef
    ) -> None:
        """save() returns existing ID when content_hash matches."""
        snap1 = ModelSnapshot(subject=subject, data={"same": "data"}, sequence_number=1)
        snap2 = ModelSnapshot(subject=subject, data={"same": "data"}, sequence_number=2)

        id1 = await store.save(snap1)
        id2 = await store.save(snap2)

        # Same content_hash should return existing ID
        assert id1 == id2
        assert store.count() == 1

    @pytest.mark.asyncio
    async def test_save_different_content_creates_new(
        self, store: StoreSnapshotInMemory, subject: ModelSubjectRef
    ) -> None:
        """save() creates new snapshot when content differs."""
        snap1 = ModelSnapshot(subject=subject, data={"v": 1}, sequence_number=1)
        snap2 = ModelSnapshot(subject=subject, data={"v": 2}, sequence_number=2)

        id1 = await store.save(snap1)
        id2 = await store.save(snap2)

        assert id1 != id2
        assert store.count() == 2


class TestStoreSnapshotInMemoryLoad:
    """Tests for load() method."""

    @pytest.mark.asyncio
    async def test_load_returns_saved_snapshot(
        self, store: StoreSnapshotInMemory, subject: ModelSubjectRef
    ) -> None:
        """load() returns the saved snapshot."""
        snapshot = ModelSnapshot(
            subject=subject, data={"key": "value"}, sequence_number=1
        )
        await store.save(snapshot)

        loaded = await store.load(snapshot.id)
        assert loaded is not None
        assert loaded.id == snapshot.id
        assert loaded.data == {"key": "value"}

    @pytest.mark.asyncio
    async def test_load_returns_none_for_missing(
        self, store: StoreSnapshotInMemory
    ) -> None:
        """load() returns None for non-existent ID."""
        result = await store.load(uuid4())
        assert result is None


class TestStoreSnapshotInMemoryLoadLatest:
    """Tests for load_latest() method."""

    @pytest.mark.asyncio
    async def test_load_latest_returns_highest_sequence(
        self, store: StoreSnapshotInMemory, subject: ModelSubjectRef
    ) -> None:
        """load_latest() returns snapshot with highest sequence_number."""
        for i in range(1, 4):
            snap = ModelSnapshot(subject=subject, data={"n": i}, sequence_number=i)
            await store.save(snap)

        latest = await store.load_latest(subject=subject)
        assert latest is not None
        assert latest.sequence_number == 3
        assert latest.data["n"] == 3

    @pytest.mark.asyncio
    async def test_load_latest_filters_by_subject(
        self, store: StoreSnapshotInMemory
    ) -> None:
        """load_latest() filters by subject."""
        subj1 = ModelSubjectRef(subject_type="a", subject_id=uuid4())
        subj2 = ModelSubjectRef(subject_type="b", subject_id=uuid4())

        snap1 = ModelSnapshot(subject=subj1, data={"s": 1}, sequence_number=1)
        snap2 = ModelSnapshot(subject=subj2, data={"s": 2}, sequence_number=2)
        await store.save(snap1)
        await store.save(snap2)

        latest_subj1 = await store.load_latest(subject=subj1)
        assert latest_subj1 is not None
        assert latest_subj1.data["s"] == 1

    @pytest.mark.asyncio
    async def test_load_latest_returns_none_when_empty(
        self, store: StoreSnapshotInMemory, subject: ModelSubjectRef
    ) -> None:
        """load_latest() returns None when no snapshots exist."""
        result = await store.load_latest(subject=subject)
        assert result is None

    @pytest.mark.asyncio
    async def test_load_latest_without_subject_returns_global(
        self, store: StoreSnapshotInMemory
    ) -> None:
        """load_latest() without subject returns global latest."""
        subj1 = ModelSubjectRef(subject_type="a", subject_id=uuid4())
        subj2 = ModelSubjectRef(subject_type="b", subject_id=uuid4())

        snap1 = ModelSnapshot(subject=subj1, data={"v": 1}, sequence_number=1)
        snap2 = ModelSnapshot(
            subject=subj2, data={"v": 2}, sequence_number=100
        )  # Higher seq
        await store.save(snap1)
        await store.save(snap2)

        latest = await store.load_latest()
        assert latest is not None
        assert latest.sequence_number == 100


class TestStoreSnapshotInMemoryQuery:
    """Tests for query() method."""

    @pytest.mark.asyncio
    async def test_query_returns_all_for_subject(
        self, store: StoreSnapshotInMemory, subject: ModelSubjectRef
    ) -> None:
        """query() returns all snapshots for subject."""
        for i in range(1, 6):
            snap = ModelSnapshot(subject=subject, data={"n": i}, sequence_number=i)
            await store.save(snap)

        results = await store.query(subject=subject)
        assert len(results) == 5

    @pytest.mark.asyncio
    async def test_query_respects_limit(
        self, store: StoreSnapshotInMemory, subject: ModelSubjectRef
    ) -> None:
        """query() respects limit parameter."""
        for i in range(1, 11):
            snap = ModelSnapshot(subject=subject, data={"n": i}, sequence_number=i)
            await store.save(snap)

        results = await store.query(subject=subject, limit=3)
        assert len(results) == 3

    @pytest.mark.asyncio
    async def test_query_ordered_by_sequence_desc(
        self, store: StoreSnapshotInMemory, subject: ModelSubjectRef
    ) -> None:
        """query() orders results by sequence_number descending."""
        for i in range(1, 6):
            snap = ModelSnapshot(subject=subject, data={"n": i}, sequence_number=i)
            await store.save(snap)

        results = await store.query(subject=subject)
        sequences = [s.sequence_number for s in results]
        assert sequences == [5, 4, 3, 2, 1]

    @pytest.mark.asyncio
    async def test_query_filters_by_after(
        self, store: StoreSnapshotInMemory, subject: ModelSubjectRef
    ) -> None:
        """query() filters by created_at > after."""
        # Create snapshot
        snap = ModelSnapshot(subject=subject, data={"test": True}, sequence_number=1)
        await store.save(snap)

        # Query with future timestamp
        future = datetime.now(UTC) + timedelta(hours=1)
        results = await store.query(subject=subject, after=future)
        assert len(results) == 0

    @pytest.mark.asyncio
    async def test_query_returns_empty_when_no_matches(
        self, store: StoreSnapshotInMemory, subject: ModelSubjectRef
    ) -> None:
        """query() returns empty list when no matches."""
        results = await store.query(subject=subject)
        assert results == []


class TestStoreSnapshotInMemoryDelete:
    """Tests for delete() method."""

    @pytest.mark.asyncio
    async def test_delete_returns_true_on_success(
        self, store: StoreSnapshotInMemory, subject: ModelSubjectRef
    ) -> None:
        """delete() returns True when snapshot deleted."""
        snapshot = ModelSnapshot(subject=subject, data={}, sequence_number=1)
        await store.save(snapshot)

        result = await store.delete(snapshot.id)
        assert result is True

    @pytest.mark.asyncio
    async def test_delete_returns_false_for_missing(
        self, store: StoreSnapshotInMemory
    ) -> None:
        """delete() returns False for non-existent ID."""
        result = await store.delete(uuid4())
        assert result is False

    @pytest.mark.asyncio
    async def test_delete_removes_from_store(
        self, store: StoreSnapshotInMemory, subject: ModelSubjectRef
    ) -> None:
        """delete() removes snapshot from store."""
        snapshot = ModelSnapshot(subject=subject, data={}, sequence_number=1)
        await store.save(snapshot)
        assert store.count() == 1

        await store.delete(snapshot.id)
        assert store.count() == 0
        assert await store.load(snapshot.id) is None


class TestStoreSnapshotInMemorySequenceNumber:
    """Tests for get_next_sequence_number() method."""

    @pytest.mark.asyncio
    async def test_sequence_starts_at_one(
        self, store: StoreSnapshotInMemory, subject: ModelSubjectRef
    ) -> None:
        """get_next_sequence_number() starts at 1 for new subjects."""
        seq = await store.get_next_sequence_number(subject)
        assert seq == 1

    @pytest.mark.asyncio
    async def test_sequence_increments(
        self, store: StoreSnapshotInMemory, subject: ModelSubjectRef
    ) -> None:
        """get_next_sequence_number() increments monotonically."""
        seq1 = await store.get_next_sequence_number(subject)
        seq2 = await store.get_next_sequence_number(subject)
        seq3 = await store.get_next_sequence_number(subject)

        assert seq1 == 1
        assert seq2 == 2
        assert seq3 == 3

    @pytest.mark.asyncio
    async def test_sequence_isolated_by_subject(
        self, store: StoreSnapshotInMemory
    ) -> None:
        """get_next_sequence_number() is isolated per subject."""
        subj1 = ModelSubjectRef(subject_type="a", subject_id=uuid4())
        subj2 = ModelSubjectRef(subject_type="b", subject_id=uuid4())

        seq1_a = await store.get_next_sequence_number(subj1)
        seq1_b = await store.get_next_sequence_number(subj2)
        seq2_a = await store.get_next_sequence_number(subj1)

        assert seq1_a == 1
        assert seq1_b == 1
        assert seq2_a == 2


class TestStoreSnapshotInMemoryTestHelpers:
    """Tests for test helper methods."""

    def test_count_returns_zero_initially(self, store: StoreSnapshotInMemory) -> None:
        """count() returns 0 for empty store."""
        assert store.count() == 0

    @pytest.mark.asyncio
    async def test_count_returns_correct_value(
        self, store: StoreSnapshotInMemory, subject: ModelSubjectRef
    ) -> None:
        """count() returns correct snapshot count."""
        for i in range(1, 4):
            snap = ModelSnapshot(subject=subject, data={"n": i}, sequence_number=i)
            await store.save(snap)

        assert store.count() == 3

    @pytest.mark.asyncio
    async def test_clear_removes_all_data(
        self, store: StoreSnapshotInMemory, subject: ModelSubjectRef
    ) -> None:
        """clear() removes all snapshots and sequences."""
        for i in range(1, 4):
            snap = ModelSnapshot(subject=subject, data={"n": i}, sequence_number=i)
            await store.save(snap)

        assert store.count() == 3
        store.clear()
        assert store.count() == 0

        # Sequence should reset
        seq = await store.get_next_sequence_number(subject)
        assert seq == 1
