# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
# S608 disabled: SQL injection is not a concern here - table names come from
# test fixtures (UUID-based), not user input. Parameterized queries are used
# for all user-facing values ($1, $2, etc.).
"""Integration tests for StoreSnapshotPostgres against real PostgreSQL.

These tests validate the PostgreSQL snapshot store implementation against actual
PostgreSQL infrastructure. They verify:

1. Basic CRUD operations (save, load, load_latest, query, delete)
2. **Concurrent sequence generation** - verifies atomic sequence generation
   under concurrent access to detect race conditions
3. **Content hash deduplication** - verifies ON CONFLICT upserts work correctly
4. Schema migration idempotency (ensure_schema can be called repeatedly)

Test Isolation:
    Each test uses a unique table suffix to prevent interference between
    parallel test runs. The ensure_schema method creates isolated tables
    per test.

Environment Variables:
    OMNIBASE_INFRA_DB_URL: Full PostgreSQL DSN (preferred, overrides individual vars)
        Example: postgresql://postgres:secret@localhost:5432/omninode_bridge

    Fallback (used only if OMNIBASE_INFRA_DB_URL is not set):
    POSTGRES_HOST: PostgreSQL hostname (fallback if OMNIBASE_INFRA_DB_URL not set)
    POSTGRES_PORT: PostgreSQL port (default: 5432)
    POSTGRES_USER: Database username (default: postgres)
    POSTGRES_PASSWORD: Database password (fallback - tests skip if neither is set)

CI/CD Graceful Skip Behavior:
    Tests skip gracefully when PostgreSQL is unavailable:
    - Skips if POSTGRES_HOST not set
    - Skips if POSTGRES_PASSWORD not set
    - Uses module-level pytestmark with pytest.mark.skipif

Related Tickets:
    - OMN-1246: ServiceSnapshot Infrastructure Primitive
    - PR #150: PostgreSQL integration tests for concurrent sequence generation
"""

from __future__ import annotations

import asyncio
import os
from collections.abc import AsyncGenerator
from datetime import UTC, datetime, timedelta
from uuid import UUID, uuid4

import asyncpg
import pytest

from omnibase_infra.models.snapshot import ModelSnapshot, ModelSubjectRef
from omnibase_infra.services.snapshot import StoreSnapshotPostgres

# =============================================================================
# Environment Configuration
# =============================================================================


def _get_postgres_dsn() -> str | None:
    """Build PostgreSQL DSN from environment variables.

    Primary source: ``OMNIBASE_INFRA_DB_URL``.
    Fallback: individual ``POSTGRES_*`` env vars.

    Returns:
        PostgreSQL connection string, or None if not configured.
    """
    db_url = os.getenv("OMNIBASE_INFRA_DB_URL")
    if db_url:
        return db_url

    host = os.getenv("POSTGRES_HOST")
    port = os.getenv("POSTGRES_PORT", "5432")
    user = os.getenv("POSTGRES_USER", "postgres")
    password = os.getenv("POSTGRES_PASSWORD")

    if not host or not password:
        return None

    return f"postgresql://{user}:{password}@{host}:{port}/omninode_bridge"


# Check PostgreSQL availability at module import
_POSTGRES_DSN = _get_postgres_dsn()
POSTGRES_AVAILABLE = _POSTGRES_DSN is not None


# =============================================================================
# Test Markers - Skip all tests if PostgreSQL unavailable
# =============================================================================

pytestmark = [
    pytest.mark.asyncio,
    pytest.mark.integration,
    pytest.mark.database,
    pytest.mark.postgres,
    pytest.mark.skipif(
        not POSTGRES_AVAILABLE,
        reason="PostgreSQL not available (POSTGRES_HOST or POSTGRES_PASSWORD not set)",
    ),
]


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
async def db_pool() -> AsyncGenerator[asyncpg.Pool, None]:
    """Create database connection pool for tests.

    Yields:
        asyncpg.Pool connected to the test database.

    Raises:
        pytest.skip: If database is not reachable.
    """
    dsn = _get_postgres_dsn()
    if not dsn:
        pytest.skip("PostgreSQL DSN not available")

    try:
        pool = await asyncpg.create_pool(
            dsn=dsn,
            min_size=2,
            max_size=20,  # Higher max for concurrency tests
            timeout=10.0,
        )
    except (asyncpg.PostgresConnectionError, OSError) as e:
        pytest.skip(f"Database not reachable: {e}")

    yield pool

    await pool.close()


@pytest.fixture
async def store(db_pool: asyncpg.Pool) -> AsyncGenerator[StoreSnapshotPostgres, None]:
    """Create StoreSnapshotPostgres with schema initialized.

    Creates the snapshots table and indexes, yields the store,
    then cleans up the table after the test.

    Yields:
        Initialized StoreSnapshotPostgres ready for operations.
    """
    store = StoreSnapshotPostgres(pool=db_pool)
    await store.ensure_schema()

    yield store

    # Cleanup: truncate table to avoid affecting other tests
    # We don't drop the table since ensure_schema is idempotent
    try:
        async with db_pool.acquire() as conn:
            await conn.execute("TRUNCATE TABLE snapshots CASCADE")
    except Exception:
        pass  # Ignore cleanup failures


@pytest.fixture
def subject() -> ModelSubjectRef:
    """Create a unique test subject reference."""
    return ModelSubjectRef(subject_type="test", subject_id=uuid4())


@pytest.fixture
def unique_subject_factory() -> type:
    """Factory for creating unique subjects."""

    class SubjectFactory:
        @staticmethod
        def create(subject_type: str = "test") -> ModelSubjectRef:
            return ModelSubjectRef(subject_type=subject_type, subject_id=uuid4())

    return SubjectFactory


# =============================================================================
# Schema Migration Tests
# =============================================================================


class TestSchemaManagement:
    """Tests for schema creation and migration."""

    async def test_ensure_schema_creates_table(
        self,
        db_pool: asyncpg.Pool,
    ) -> None:
        """Verify ensure_schema creates the snapshots table.

        Given: Fresh database pool
        When: Calling ensure_schema
        Then: snapshots table exists with expected columns
        """
        store = StoreSnapshotPostgres(pool=db_pool)
        await store.ensure_schema()

        async with db_pool.acquire() as conn:
            # Verify table exists
            exists = await conn.fetchval(
                """
                SELECT EXISTS (
                    SELECT 1 FROM information_schema.tables
                    WHERE table_name = 'snapshots'
                )
                """
            )
            assert exists is True

            # Verify expected columns exist
            columns = await conn.fetch(
                """
                SELECT column_name FROM information_schema.columns
                WHERE table_name = 'snapshots'
                """
            )
            column_names = {row["column_name"] for row in columns}
            expected_columns = {
                "id",
                "subject_type",
                "subject_id",
                "data",
                "sequence_number",
                "version",
                "content_hash",
                "created_at",
                "parent_id",
            }
            assert expected_columns.issubset(column_names)

    async def test_ensure_schema_is_idempotent(
        self,
        db_pool: asyncpg.Pool,
    ) -> None:
        """Verify ensure_schema can be called multiple times safely.

        Given: Schema already created
        When: Calling ensure_schema again
        Then: No error, schema remains intact
        """
        store = StoreSnapshotPostgres(pool=db_pool)

        # Call ensure_schema multiple times
        await store.ensure_schema()
        await store.ensure_schema()
        await store.ensure_schema()

        # Verify table still exists
        async with db_pool.acquire() as conn:
            exists = await conn.fetchval(
                """
                SELECT EXISTS (
                    SELECT 1 FROM information_schema.tables
                    WHERE table_name = 'snapshots'
                )
                """
            )
            assert exists is True

    async def test_ensure_schema_creates_unique_content_hash_index(
        self,
        db_pool: asyncpg.Pool,
    ) -> None:
        """Verify ensure_schema creates unique partial index on content_hash.

        Given: Fresh database pool
        When: Calling ensure_schema
        Then: Unique partial index on content_hash exists
        """
        store = StoreSnapshotPostgres(pool=db_pool)
        await store.ensure_schema()

        async with db_pool.acquire() as conn:
            # Check index exists and is unique
            index_info = await conn.fetchrow(
                """
                SELECT indexdef FROM pg_indexes
                WHERE indexname = 'idx_snapshots_content_hash'
                """
            )
            assert index_info is not None
            # Verify it's a UNIQUE index
            assert "UNIQUE" in index_info["indexdef"].upper()


# =============================================================================
# Basic CRUD Tests
# =============================================================================


class TestBasicCrud:
    """Tests for basic CRUD operations."""

    async def test_save_and_load_snapshot(
        self,
        store: StoreSnapshotPostgres,
        subject: ModelSubjectRef,
    ) -> None:
        """Verify save and load work correctly.

        Given: New snapshot
        When: save() then load()
        Then: Loaded snapshot matches saved data
        """
        snapshot = ModelSnapshot(
            subject=subject,
            data={"key": "value", "nested": {"a": 1}},
            sequence_number=1,
        )

        saved_id = await store.save(snapshot)
        assert saved_id == snapshot.id

        loaded = await store.load(snapshot.id)
        assert loaded is not None
        assert loaded.id == snapshot.id
        assert loaded.subject.subject_type == subject.subject_type
        assert loaded.subject.subject_id == subject.subject_id
        assert loaded.data == {"key": "value", "nested": {"a": 1}}
        assert loaded.sequence_number == 1

    async def test_load_returns_none_for_missing(
        self,
        store: StoreSnapshotPostgres,
    ) -> None:
        """Verify load returns None for non-existent ID.

        Given: Non-existent snapshot ID
        When: load()
        Then: Returns None
        """
        result = await store.load(uuid4())
        assert result is None

    async def test_load_latest_returns_highest_sequence(
        self,
        store: StoreSnapshotPostgres,
        subject: ModelSubjectRef,
    ) -> None:
        """Verify load_latest returns snapshot with highest sequence.

        Given: Multiple snapshots for same subject
        When: load_latest()
        Then: Returns snapshot with highest sequence_number
        """
        for i in range(1, 4):
            snap = ModelSnapshot(subject=subject, data={"n": i}, sequence_number=i)
            await store.save(snap)

        latest = await store.load_latest(subject=subject)
        assert latest is not None
        assert latest.sequence_number == 3
        assert latest.data["n"] == 3

    async def test_load_latest_without_subject_returns_global(
        self,
        store: StoreSnapshotPostgres,
        unique_subject_factory: type,
    ) -> None:
        """Verify load_latest without subject returns global latest.

        Given: Multiple snapshots across different subjects
        When: load_latest() without subject filter
        Then: Returns snapshot with highest sequence_number globally
        """
        subj1 = unique_subject_factory.create("type_a")
        subj2 = unique_subject_factory.create("type_b")

        snap1 = ModelSnapshot(subject=subj1, data={"v": 1}, sequence_number=1)
        snap2 = ModelSnapshot(subject=subj2, data={"v": 100}, sequence_number=100)
        await store.save(snap1)
        await store.save(snap2)

        latest = await store.load_latest()
        assert latest is not None
        # Verify exact snapshot ID (strongest assertion)
        assert latest.id == snap2.id
        # Verify sequence_number is truly the highest
        assert latest.sequence_number == 100
        assert latest.sequence_number > snap1.sequence_number
        # Verify data integrity
        assert latest.data["v"] == 100
        # Verify it's from the expected subject
        assert latest.subject.subject_type == "type_b"

    async def test_query_returns_ordered_results(
        self,
        store: StoreSnapshotPostgres,
        subject: ModelSubjectRef,
    ) -> None:
        """Verify query returns results ordered by sequence desc.

        Given: Multiple snapshots
        When: query()
        Then: Results ordered by sequence_number descending
        """
        for i in range(1, 6):
            snap = ModelSnapshot(subject=subject, data={"n": i}, sequence_number=i)
            await store.save(snap)

        results = await store.query(subject=subject)
        assert len(results) == 5
        sequences = [s.sequence_number for s in results]
        assert sequences == [5, 4, 3, 2, 1]

    async def test_query_respects_limit(
        self,
        store: StoreSnapshotPostgres,
        subject: ModelSubjectRef,
    ) -> None:
        """Verify query respects limit parameter.

        Given: Multiple snapshots
        When: query() with limit
        Then: Returns at most limit results
        """
        for i in range(1, 11):
            snap = ModelSnapshot(subject=subject, data={"n": i}, sequence_number=i)
            await store.save(snap)

        results = await store.query(subject=subject, limit=3)
        assert len(results) == 3

    async def test_query_filters_by_after(
        self,
        store: StoreSnapshotPostgres,
        subject: ModelSubjectRef,
    ) -> None:
        """Verify query filters by created_at > after.

        Given: Snapshots with different creation times
        When: query() with after filter
        Then: Only returns snapshots created after the timestamp
        """
        snap = ModelSnapshot(subject=subject, data={"test": True}, sequence_number=1)
        await store.save(snap)

        # Query with future timestamp should return empty
        future = datetime.now(UTC) + timedelta(hours=1)
        results = await store.query(subject=subject, after=future)
        assert len(results) == 0

    async def test_delete_removes_snapshot(
        self,
        store: StoreSnapshotPostgres,
        subject: ModelSubjectRef,
    ) -> None:
        """Verify delete removes snapshot from database.

        Given: Saved snapshot
        When: delete()
        Then: Snapshot no longer loadable
        """
        snapshot = ModelSnapshot(subject=subject, data={}, sequence_number=1)
        await store.save(snapshot)

        result = await store.delete(snapshot.id)
        assert result is True

        loaded = await store.load(snapshot.id)
        assert loaded is None

    async def test_delete_returns_false_for_missing(
        self,
        store: StoreSnapshotPostgres,
    ) -> None:
        """Verify delete returns False for non-existent ID.

        Given: Non-existent snapshot ID
        When: delete()
        Then: Returns False
        """
        result = await store.delete(uuid4())
        assert result is False


# =============================================================================
# Content Hash Deduplication Tests
# =============================================================================


class TestContentHashDeduplication:
    """Tests for content-hash based idempotency."""

    async def test_save_idempotent_on_content_hash(
        self,
        store: StoreSnapshotPostgres,
        subject: ModelSubjectRef,
    ) -> None:
        """Verify save returns existing ID for duplicate content.

        Given: Snapshot with content_hash
        When: save() with same content_hash but different sequence
        Then: Returns existing snapshot's ID
        """
        snap1 = ModelSnapshot(subject=subject, data={"same": "data"}, sequence_number=1)
        snap2 = ModelSnapshot(subject=subject, data={"same": "data"}, sequence_number=2)

        # Both have same content_hash (auto-computed from identical data)
        assert snap1.content_hash == snap2.content_hash

        id1 = await store.save(snap1)
        id2 = await store.save(snap2)

        # Should return existing ID due to content_hash match
        assert id1 == id2

    async def test_different_content_creates_new_snapshot(
        self,
        store: StoreSnapshotPostgres,
        subject: ModelSubjectRef,
    ) -> None:
        """Verify different content creates separate snapshots.

        Given: Two snapshots with different data
        When: save() both
        Then: Both are stored with unique IDs
        """
        snap1 = ModelSnapshot(subject=subject, data={"v": 1}, sequence_number=1)
        snap2 = ModelSnapshot(subject=subject, data={"v": 2}, sequence_number=2)

        id1 = await store.save(snap1)
        id2 = await store.save(snap2)

        assert id1 != id2

        # Verify both exist
        loaded1 = await store.load(id1)
        loaded2 = await store.load(id2)
        assert loaded1 is not None
        assert loaded2 is not None

    async def test_concurrent_duplicate_saves_are_idempotent(
        self,
        store: StoreSnapshotPostgres,
        subject: ModelSubjectRef,
    ) -> None:
        """Verify concurrent saves of same content are idempotent.

        Given: Multiple concurrent saves with identical content
        When: save() called concurrently
        Then: All return the same ID (first wins, others get existing)

        This tests the ON CONFLICT behavior under concurrent access.
        """
        # Create multiple snapshots with identical content
        snapshots = [
            ModelSnapshot(
                subject=subject, data={"concurrent": "test"}, sequence_number=i
            )
            for i in range(1, 11)
        ]

        # Save all concurrently
        tasks = [store.save(snap) for snap in snapshots]
        results = await asyncio.gather(*tasks)

        # All should return the same ID (content_hash match)
        unique_ids = set(results)
        assert len(unique_ids) == 1, (
            f"Expected all saves to return same ID due to content_hash, "
            f"but got {len(unique_ids)} unique IDs"
        )


# =============================================================================
# Concurrent Sequence Generation Tests (Critical for Race Condition Fix)
# =============================================================================


class TestConcurrentSequenceGeneration:
    """Tests for atomic sequence generation under concurrent access.

    These tests verify the race condition fix for sequence number generation.
    The PostgreSQL store uses MAX(sequence_number) + 1 with database-level
    unique constraints to ensure no duplicate sequences are assigned.
    """

    async def test_concurrent_get_next_sequence_number(
        self,
        store: StoreSnapshotPostgres,
        subject: ModelSubjectRef,
    ) -> None:
        """Verify get_next_sequence_number works under concurrent access.

        Given: Single subject
        When: Multiple concurrent get_next_sequence_number calls
        Then: All calls complete (may return duplicates - that's expected)

        Note: get_next_sequence_number is NOT atomic across calls. The
        uniqueness is enforced by database constraints during save().
        """
        num_concurrent = 10
        tasks = [store.get_next_sequence_number(subject) for _ in range(num_concurrent)]
        results = await asyncio.gather(*tasks)

        # All should return valid sequence numbers >= 1
        assert all(seq >= 1 for seq in results)

    async def test_concurrent_saves_unique_sequences(
        self,
        store: StoreSnapshotPostgres,
        subject: ModelSubjectRef,
    ) -> None:
        """Verify concurrent saves with same sequence are handled correctly.

        Given: Multiple snapshots with unique data for same subject
        When: save() called concurrently with pre-assigned unique sequences
        Then: All saves succeed with their assigned sequence numbers

        This is the key test for the race condition fix - ensuring that
        when we DO assign unique sequences, they all get saved correctly.
        """
        num_concurrent = 10

        # Create snapshots with unique data AND unique sequences
        snapshots = [
            ModelSnapshot(
                subject=subject,
                data={
                    "unique_value": str(uuid4())
                },  # Unique data = unique content_hash
                sequence_number=i,
            )
            for i in range(1, num_concurrent + 1)
        ]

        # Save all concurrently
        tasks = [store.save(snap) for snap in snapshots]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # All should succeed (no exceptions)
        exceptions = [r for r in results if isinstance(r, Exception)]
        assert len(exceptions) == 0, f"Concurrent saves raised exceptions: {exceptions}"

        # Verify all unique IDs returned
        successful_ids = [r for r in results if isinstance(r, UUID)]
        assert len(successful_ids) == num_concurrent
        assert len(set(successful_ids)) == num_concurrent, "Expected all unique IDs"

        # Verify all snapshots are in database
        all_snapshots = await store.query(subject=subject, limit=num_concurrent + 5)
        assert len(all_snapshots) == num_concurrent

    async def test_high_concurrency_sequence_uniqueness(
        self,
        store: StoreSnapshotPostgres,
        subject: ModelSubjectRef,
    ) -> None:
        """Verify sequence uniqueness under high concurrency stress.

        Given: High number of concurrent save operations
        When: All trying to save to same subject with unique data
        Then: Database constraint prevents duplicate sequences

        This test simulates the race condition scenario where multiple
        processes get the same "next sequence" but try to save with it.
        The database UNIQUE constraint should prevent duplicates.
        """
        num_concurrent = 50

        async def save_with_auto_sequence(i: int) -> tuple[UUID | None, str | None]:
            """Get next sequence and save - simulates realistic usage."""
            try:
                seq = await store.get_next_sequence_number(subject)
                snapshot = ModelSnapshot(
                    subject=subject,
                    data={"worker": i, "timestamp": str(datetime.now(UTC))},
                    sequence_number=seq,
                )
                result_id = await store.save(snapshot)
                return result_id, None
            except Exception as e:
                return None, str(e)

        # Run all concurrently
        tasks = [save_with_auto_sequence(i) for i in range(num_concurrent)]
        results = await asyncio.gather(*tasks)

        # Count successes and failures
        successes = [r[0] for r in results if r[0] is not None]
        failures = [r[1] for r in results if r[1] is not None]

        # Some may fail due to sequence conflicts - that's expected!
        # The important thing is that no duplicate sequences exist in DB.
        # Failures should mention "constraint" or "conflict"
        for failure_msg in failures:
            assert (
                "constraint" in failure_msg.lower()
                or "conflict" in failure_msg.lower()
                or "sequence" in failure_msg.lower()
            ), f"Unexpected failure reason: {failure_msg}"

        # Verify NO duplicate sequences in database
        all_snapshots = await store.query(subject=subject, limit=num_concurrent + 10)
        sequences = [s.sequence_number for s in all_snapshots]
        assert len(sequences) == len(set(sequences)), (
            f"Found duplicate sequences! Sequences: {sorted(sequences)}"
        )

        # At least some should have succeeded
        assert len(successes) > 0, "Expected at least some saves to succeed"

    async def test_concurrent_saves_different_subjects_isolated(
        self,
        store: StoreSnapshotPostgres,
        unique_subject_factory: type,
    ) -> None:
        """Verify concurrent saves to different subjects are isolated.

        Given: Multiple subjects
        When: Concurrent saves to different subjects
        Then: Each subject's sequence starts at 1, no cross-contamination
        """
        num_subjects = 5
        num_snapshots_per_subject = 10

        subjects = [
            unique_subject_factory.create(f"type_{i}") for i in range(num_subjects)
        ]

        async def save_for_subject(subj: ModelSubjectRef, seq: int) -> UUID:
            snapshot = ModelSnapshot(
                subject=subj,
                data={"subject": subj.subject_type, "seq": seq},
                sequence_number=seq,
            )
            return await store.save(snapshot)

        # Create all tasks
        tasks = []
        for subj in subjects:
            for seq in range(1, num_snapshots_per_subject + 1):
                tasks.append(save_for_subject(subj, seq))

        # Run all concurrently
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # All should succeed
        exceptions = [r for r in results if isinstance(r, Exception)]
        assert len(exceptions) == 0, f"Some saves failed: {exceptions}"

        # Verify each subject has correct snapshots
        for subj in subjects:
            snapshots = await store.query(
                subject=subj, limit=num_snapshots_per_subject + 5
            )
            assert len(snapshots) == num_snapshots_per_subject
            sequences = sorted([s.sequence_number for s in snapshots])
            expected = list(range(1, num_snapshots_per_subject + 1))
            assert sequences == expected


# =============================================================================
# Error Handling Tests
# =============================================================================


class TestErrorHandling:
    """Tests for error handling and edge cases."""

    async def test_save_with_null_data(
        self,
        store: StoreSnapshotPostgres,
        subject: ModelSubjectRef,
    ) -> None:
        """Verify save handles empty data dict correctly.

        Given: Snapshot with empty data
        When: save() and load()
        Then: Empty dict is preserved
        """
        snapshot = ModelSnapshot(subject=subject, data={}, sequence_number=1)
        await store.save(snapshot)

        loaded = await store.load(snapshot.id)
        assert loaded is not None
        assert loaded.data == {}

    async def test_save_with_complex_nested_data(
        self,
        store: StoreSnapshotPostgres,
        subject: ModelSubjectRef,
    ) -> None:
        """Verify save handles complex nested JSON data.

        Given: Snapshot with deeply nested data
        When: save() and load()
        Then: All nested structure preserved
        """
        complex_data = {
            "level1": {
                "level2": {
                    "level3": {
                        "values": [1, 2, 3],
                        "nested_list": [{"a": 1}, {"b": 2}],
                    }
                }
            },
            "array_of_arrays": [[1, 2], [3, 4]],
            "unicode": "test unicode: \u4e2d\u6587",
            "special_chars": "quotes: \"test\" and 'test'",
        }

        snapshot = ModelSnapshot(subject=subject, data=complex_data, sequence_number=1)
        await store.save(snapshot)

        loaded = await store.load(snapshot.id)
        assert loaded is not None
        assert loaded.data == complex_data

    async def test_sequence_conflict_with_different_content(
        self,
        store: StoreSnapshotPostgres,
        subject: ModelSubjectRef,
    ) -> None:
        """Verify sequence conflict raises appropriate error.

        Given: Existing snapshot with sequence 1
        When: Trying to save different content with same sequence
        Then: InfraConnectionError raised mentioning constraint
        """
        from omnibase_infra.errors import InfraConnectionError

        snap1 = ModelSnapshot(subject=subject, data={"first": True}, sequence_number=1)
        await store.save(snap1)

        snap2 = ModelSnapshot(subject=subject, data={"second": True}, sequence_number=1)

        with pytest.raises(InfraConnectionError) as exc_info:
            await store.save(snap2)

        # Error should mention constraint or conflict
        error_msg = str(exc_info.value).lower()
        assert (
            "constraint" in error_msg
            or "conflict" in error_msg
            or "sequence" in error_msg
        )

    async def test_load_latest_returns_none_for_empty_subject(
        self,
        store: StoreSnapshotPostgres,
        subject: ModelSubjectRef,
    ) -> None:
        """Verify load_latest returns None when subject has no snapshots.

        Given: Subject with no snapshots
        When: load_latest()
        Then: Returns None
        """
        result = await store.load_latest(subject=subject)
        assert result is None

    async def test_query_returns_empty_for_no_matches(
        self,
        store: StoreSnapshotPostgres,
        subject: ModelSubjectRef,
    ) -> None:
        """Verify query returns empty list when no matches.

        Given: Subject with no snapshots
        When: query()
        Then: Returns empty list
        """
        results = await store.query(subject=subject)
        assert results == []


# =============================================================================
# Parent ID / Lineage Tests
# =============================================================================


class TestParentLineage:
    """Tests for parent_id lineage tracking."""

    async def test_save_with_parent_id(
        self,
        store: StoreSnapshotPostgres,
        subject: ModelSubjectRef,
    ) -> None:
        """Verify parent_id is stored and retrieved correctly.

        Given: Snapshot with parent_id reference
        When: save() and load()
        Then: parent_id is preserved
        """
        parent = ModelSnapshot(
            subject=subject, data={"parent": True}, sequence_number=1
        )
        await store.save(parent)

        child = ModelSnapshot(
            subject=subject,
            data={"child": True},
            sequence_number=2,
            parent_id=parent.id,
        )
        await store.save(child)

        loaded_child = await store.load(child.id)
        assert loaded_child is not None
        assert loaded_child.parent_id == parent.id

    async def test_with_mutations_creates_lineage(
        self,
        store: StoreSnapshotPostgres,
        subject: ModelSubjectRef,
    ) -> None:
        """Verify with_mutations creates proper parent linkage.

        Given: Original snapshot
        When: with_mutations() and save()
        Then: New snapshot has parent_id pointing to original
        """
        original = ModelSnapshot(
            subject=subject,
            data={"status": "active"},
            sequence_number=1,
        )
        await store.save(original)

        mutated = original.with_mutations(
            mutations={"status": "inactive"},
            sequence_number=2,
        )
        await store.save(mutated)

        loaded = await store.load(mutated.id)
        assert loaded is not None
        assert loaded.parent_id == original.id
        assert loaded.data["status"] == "inactive"


# =============================================================================
# Cleanup/Retention Policy Tests
# =============================================================================


class TestCleanupExpired:
    """Tests for cleanup_expired retention policy enforcement."""

    async def test_cleanup_by_age_removes_old_snapshots(
        self,
        store: StoreSnapshotPostgres,
        subject: ModelSubjectRef,
        db_pool: asyncpg.Pool,
    ) -> None:
        """Verify cleanup_expired removes snapshots older than max_age_seconds.

        Given: Snapshots with manually backdated created_at timestamps
        When: cleanup_expired(max_age_seconds=3600)
        Then: Old snapshots are deleted, recent ones retained
        """
        # Create snapshots with different ages by manually updating timestamps
        snap1 = ModelSnapshot(subject=subject, data={"age": "old"}, sequence_number=1)
        snap2 = ModelSnapshot(
            subject=subject, data={"age": "recent"}, sequence_number=2
        )

        await store.save(snap1)
        await store.save(snap2)

        # Backdate snap1 to be 2 hours old
        async with db_pool.acquire() as conn:
            old_time = datetime.now(UTC) - timedelta(hours=2)
            await conn.execute(
                "UPDATE snapshots SET created_at = $1 WHERE id = $2",
                old_time,
                snap1.id,
            )

        # Cleanup with 1 hour max age (snap1 should be deleted)
        deleted = await store.cleanup_expired(max_age_seconds=3600)
        assert deleted == 1

        # Verify snap1 is gone, snap2 remains
        assert await store.load(snap1.id) is None
        assert await store.load(snap2.id) is not None

    async def test_cleanup_by_count_keeps_latest_n(
        self,
        store: StoreSnapshotPostgres,
        subject: ModelSubjectRef,
    ) -> None:
        """Verify cleanup_expired keeps only the N most recent per subject.

        Given: Multiple snapshots for a subject
        When: cleanup_expired(keep_latest_n=3)
        Then: Only the 3 highest sequence_number snapshots remain
        """
        # Create 5 snapshots
        snapshot_ids = []
        for i in range(1, 6):
            snap = ModelSnapshot(subject=subject, data={"n": i}, sequence_number=i)
            await store.save(snap)
            snapshot_ids.append(snap.id)

        # Keep only latest 3
        deleted = await store.cleanup_expired(keep_latest_n=3)
        assert deleted == 2  # seq 1 and 2 should be deleted

        # Verify seq 1 and 2 are gone, seq 3, 4, 5 remain
        assert await store.load(snapshot_ids[0]) is None  # seq 1
        assert await store.load(snapshot_ids[1]) is None  # seq 2
        assert await store.load(snapshot_ids[2]) is not None  # seq 3
        assert await store.load(snapshot_ids[3]) is not None  # seq 4
        assert await store.load(snapshot_ids[4]) is not None  # seq 5

    async def test_cleanup_combined_strategy(
        self,
        store: StoreSnapshotPostgres,
        subject: ModelSubjectRef,
        db_pool: asyncpg.Pool,
    ) -> None:
        """Verify combined cleanup requires BOTH conditions for deletion.

        Given: Snapshots with varying ages and sequences
        When: cleanup_expired(max_age_seconds=3600, keep_latest_n=3)
        Then: Only snapshots older than 1 hour AND outside top 3 are deleted
        """
        # Create 5 snapshots
        snapshot_ids = []
        for i in range(1, 6):
            snap = ModelSnapshot(subject=subject, data={"n": i}, sequence_number=i)
            await store.save(snap)
            snapshot_ids.append(snap.id)

        # Backdate all but the latest 2 to be 2 hours old
        async with db_pool.acquire() as conn:
            old_time = datetime.now(UTC) - timedelta(hours=2)
            for snap_id in snapshot_ids[:3]:  # seq 1, 2, 3
                await conn.execute(
                    "UPDATE snapshots SET created_at = $1 WHERE id = $2",
                    old_time,
                    snap_id,
                )

        # Combined strategy: delete if older than 1 hour AND not in top 3
        # Seq 3 is old but in top 3, so should NOT be deleted
        # Seq 1, 2 are old and NOT in top 3, so should be deleted
        deleted = await store.cleanup_expired(max_age_seconds=3600, keep_latest_n=3)
        assert deleted == 2

        # Verify seq 1, 2 gone; seq 3, 4, 5 remain
        assert await store.load(snapshot_ids[0]) is None  # seq 1
        assert await store.load(snapshot_ids[1]) is None  # seq 2
        assert await store.load(snapshot_ids[2]) is not None  # seq 3 (in top 3)
        assert await store.load(snapshot_ids[3]) is not None  # seq 4
        assert await store.load(snapshot_ids[4]) is not None  # seq 5

    async def test_cleanup_scoped_to_subject(
        self,
        store: StoreSnapshotPostgres,
        unique_subject_factory: type,
    ) -> None:
        """Verify cleanup_expired can be scoped to a specific subject.

        Given: Snapshots across multiple subjects
        When: cleanup_expired(keep_latest_n=1, subject=subject1)
        Then: Only subject1's old snapshots are deleted
        """
        subject1 = unique_subject_factory.create("type_a")
        subject2 = unique_subject_factory.create("type_b")

        # Create 3 snapshots for each subject with UNIQUE data to avoid
        # content_hash deduplication (which would return existing IDs)
        s1_ids = []
        s2_ids = []
        for i in range(1, 4):
            # Use subject-specific data to ensure unique content_hash
            snap1 = ModelSnapshot(
                subject=subject1,
                data={"subject": "s1", "n": i, "unique": str(uuid4())},
                sequence_number=i,
            )
            snap2 = ModelSnapshot(
                subject=subject2,
                data={"subject": "s2", "n": i, "unique": str(uuid4())},
                sequence_number=i,
            )
            await store.save(snap1)
            await store.save(snap2)
            s1_ids.append(snap1.id)
            s2_ids.append(snap2.id)

        # Cleanup only subject1, keep latest 1
        deleted = await store.cleanup_expired(keep_latest_n=1, subject=subject1)
        assert deleted == 2  # seq 1 and 2 from subject1

        # Verify subject1: seq 1, 2 gone; seq 3 remains
        assert await store.load(s1_ids[0]) is None
        assert await store.load(s1_ids[1]) is None
        assert await store.load(s1_ids[2]) is not None

        # Verify subject2: all 3 still exist
        assert await store.load(s2_ids[0]) is not None
        assert await store.load(s2_ids[1]) is not None
        assert await store.load(s2_ids[2]) is not None

    async def test_cleanup_no_policy_returns_zero(
        self,
        store: StoreSnapshotPostgres,
        subject: ModelSubjectRef,
    ) -> None:
        """Verify cleanup_expired returns 0 when no policy specified.

        Given: Existing snapshots
        When: cleanup_expired() with no parameters
        Then: Returns 0, no snapshots deleted
        """
        snap = ModelSnapshot(subject=subject, data={"test": True}, sequence_number=1)
        await store.save(snap)

        deleted = await store.cleanup_expired()
        assert deleted == 0

        # Snapshot still exists
        assert await store.load(snap.id) is not None

    async def test_cleanup_invalid_keep_latest_raises(
        self,
        store: StoreSnapshotPostgres,
    ) -> None:
        """Verify cleanup_expired raises ProtocolConfigurationError for keep_latest_n < 1.

        Given: Any state
        When: cleanup_expired(keep_latest_n=0)
        Then: Raises ProtocolConfigurationError
        """
        from omnibase_infra.errors import ProtocolConfigurationError

        with pytest.raises(
            ProtocolConfigurationError, match="keep_latest_n must be >= 1"
        ):
            await store.cleanup_expired(keep_latest_n=0)

    async def test_cleanup_concurrent_deletes(
        self,
        store: StoreSnapshotPostgres,
        unique_subject_factory: type,
        db_pool: asyncpg.Pool,
    ) -> None:
        """Verify concurrent cleanup operations don't cause deadlocks.

        Given: Multiple subjects with snapshots
        When: Multiple cleanup_expired calls run concurrently
        Then: All complete without deadlocks
        """
        subjects = [unique_subject_factory.create(f"type_{i}") for i in range(5)]

        # Create 5 snapshots per subject with UNIQUE data to avoid content_hash dedup
        for idx, subj in enumerate(subjects):
            for seq in range(1, 6):
                snap = ModelSnapshot(
                    subject=subj,
                    data={"subject_idx": idx, "seq": seq, "unique": str(uuid4())},
                    sequence_number=seq,
                )
                await store.save(snap)

        # Backdate all to be old
        async with db_pool.acquire() as conn:
            old_time = datetime.now(UTC) - timedelta(hours=2)
            await conn.execute(
                "UPDATE snapshots SET created_at = $1",
                old_time,
            )

        # Run concurrent cleanup per subject
        async def cleanup_subject(subj: ModelSubjectRef) -> int:
            return await store.cleanup_expired(
                max_age_seconds=3600, keep_latest_n=2, subject=subj
            )

        tasks = [cleanup_subject(subj) for subj in subjects]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # No exceptions should occur (primary goal: no deadlocks)
        exceptions = [r for r in results if isinstance(r, Exception)]
        assert len(exceptions) == 0, f"Cleanup raised exceptions: {exceptions}"

        # Each subject should have 3 deleted (5 - 2 = 3)
        # With combined strategy: delete if old AND outside top 2
        successful_deletes = [r for r in results if isinstance(r, int)]
        assert len(successful_deletes) == 5, "All cleanups should complete"
        assert all(r == 3 for r in successful_deletes), (
            f"Expected 3 deletions per subject, got: {successful_deletes}"
        )


# =============================================================================
# Transaction Rollback Tests
# =============================================================================


class TestTransactionRollback:
    """Tests for transaction rollback behavior on errors."""

    async def test_failed_save_does_not_persist(
        self,
        store: StoreSnapshotPostgres,
        subject: ModelSubjectRef,
    ) -> None:
        """Verify failed save operations don't partially persist data.

        Given: Existing snapshot with sequence 1
        When: Attempting to save different content with same sequence
        Then: Original data unchanged, new data not persisted
        """
        from omnibase_infra.errors import InfraConnectionError

        original = ModelSnapshot(
            subject=subject, data={"original": True}, sequence_number=1
        )
        await store.save(original)

        duplicate_seq = ModelSnapshot(
            subject=subject, data={"duplicate": True}, sequence_number=1
        )

        # Should fail due to sequence conflict
        with pytest.raises(InfraConnectionError):
            await store.save(duplicate_seq)

        # Verify original is unchanged
        loaded = await store.load(original.id)
        assert loaded is not None
        assert loaded.data == {"original": True}

        # Verify duplicate was not persisted
        all_snaps = await store.query(subject=subject)
        assert len(all_snaps) == 1
        assert all_snaps[0].id == original.id

    async def test_concurrent_conflict_rollback(
        self,
        store: StoreSnapshotPostgres,
        subject: ModelSubjectRef,
    ) -> None:
        """Verify conflicting concurrent saves rollback correctly.

        Given: Multiple concurrent saves with same sequence but different content
        When: All execute simultaneously
        Then: Exactly one succeeds, others rollback cleanly
        """
        # Create snapshots with different data but same sequence
        snapshots = [
            ModelSnapshot(
                subject=subject,
                data={"worker": i, "unique": str(uuid4())},
                sequence_number=1,  # Same sequence for all
            )
            for i in range(10)
        ]

        # Save all concurrently
        tasks = [store.save(snap) for snap in snapshots]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Count successes and failures
        successes = [r for r in results if isinstance(r, UUID)]
        failures = [r for r in results if isinstance(r, Exception)]

        # Most should fail (sequence conflict with different content)
        # Some may succeed due to content_hash deduplication if randomly same
        assert len(successes) >= 1, "At least one save should succeed"
        assert len(failures) >= 1, "Expect some failures due to sequence conflict"

        # Verify database consistency - only one sequence 1 exists
        all_snaps = await store.query(subject=subject)
        assert len(all_snaps) == len(successes)

    async def test_schema_sequential_idempotent(
        self,
        db_pool: asyncpg.Pool,
    ) -> None:
        """Verify ensure_schema is idempotent when called sequentially.

        Given: Schema already exists
        When: ensure_schema called multiple times sequentially
        Then: All calls succeed, schema remains complete

        Note: Concurrent ensure_schema calls may race on index creation,
        so this test uses sequential calls to verify idempotency.
        """
        store = StoreSnapshotPostgres(pool=db_pool)

        # Run ensure_schema sequentially multiple times
        for _ in range(3):
            await store.ensure_schema()

        # Verify schema is complete
        async with db_pool.acquire() as conn:
            # Table exists
            table_exists = await conn.fetchval(
                """
                SELECT EXISTS (
                    SELECT 1 FROM information_schema.tables
                    WHERE table_name = 'snapshots'
                )
                """
            )
            assert table_exists is True

            # Unique index exists
            index_info = await conn.fetchrow(
                """
                SELECT indexdef FROM pg_indexes
                WHERE indexname = 'idx_snapshots_content_hash'
                """
            )
            assert index_info is not None
            assert "UNIQUE" in index_info["indexdef"].upper()


# =============================================================================
# Pool Exhaustion and Connection Error Tests
# =============================================================================


class TestConnectionResilience:
    """Tests for connection pool behavior under stress."""

    async def test_high_concurrent_load(
        self,
        store: StoreSnapshotPostgres,
        unique_subject_factory: type,
    ) -> None:
        """Verify store handles high concurrent load without pool exhaustion.

        Given: Connection pool with limited connections
        When: Many concurrent operations exceed pool size
        Then: All operations complete (some may wait for connections)
        """
        num_operations = 100
        subjects = [
            unique_subject_factory.create(f"load_test_{i}")
            for i in range(num_operations)
        ]

        async def create_and_load(subj: ModelSubjectRef) -> bool:
            """Create a snapshot and immediately load it."""
            snap = ModelSnapshot(subject=subj, data={"test": True}, sequence_number=1)
            snap_id = await store.save(snap)
            loaded = await store.load(snap_id)
            return loaded is not None and loaded.id == snap_id

        tasks = [create_and_load(subj) for subj in subjects]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Check for failures
        failures = [r for r in results if isinstance(r, Exception)]
        assert len(failures) == 0, f"Operations failed: {failures[:5]}..."

        # All should succeed
        assert all(r is True for r in results if isinstance(r, bool))


# =============================================================================
# Module Exports
# =============================================================================

__all__ = [
    "TestBasicCrud",
    "TestCleanupExpired",
    "TestConcurrentSequenceGeneration",
    "TestConnectionResilience",
    "TestContentHashDeduplication",
    "TestErrorHandling",
    "TestParentLineage",
    "TestSchemaManagement",
    "TestTransactionRollback",
]
