# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Integration tests for registration projection using testcontainers.

These tests verify the ProjectorRegistration and ProjectionReaderRegistration
against a real PostgreSQL database using testcontainers. They test:

1. Schema initialization from SQL file
2. Projection persistence with database verification
3. Query operations (by entity, state, deadline scans)
4. Ordering and sequencing enforcement
5. Idempotency (re-projecting same event)
6. Stale update rejection

Related Tickets:
    - OMN-944 (F1): Implement Registration Projection Schema
    - OMN-940 (F0): Define Projector Execution Model
    - OMN-932 (C2): Durable Timeout Handling
"""

from __future__ import annotations

import asyncio
import random
from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING
from uuid import UUID, uuid4

import pytest

from omnibase_infra.enums import EnumRegistrationState
from omnibase_infra.models.projection import (
    ModelRegistrationProjection,
    ModelSequenceInfo,
)
from omnibase_infra.models.registration.model_node_capabilities import (
    ModelNodeCapabilities,
)

if TYPE_CHECKING:
    import asyncpg

    from omnibase_infra.projectors import (
        ProjectionReaderRegistration,
        ProjectorRegistration,
    )

# Test markers
pytestmark = [
    pytest.mark.integration,
    pytest.mark.asyncio,
]


# =============================================================================
# Helper Functions
# =============================================================================


def make_projection(
    *,
    entity_id: UUID | None = None,
    state: EnumRegistrationState = EnumRegistrationState.PENDING_REGISTRATION,
    node_type: str = "effect",
    node_version: str = "1.0.0",
    event_id: str | None = None,
    offset: int = 0,
    ack_deadline: datetime | None = None,
    liveness_deadline: datetime | None = None,
) -> ModelRegistrationProjection:
    """Create a test projection with sensible defaults.

    Args:
        entity_id: Node UUID (generated if not provided)
        state: FSM state (default: PENDING_REGISTRATION)
        node_type: ONEX node type (default: "effect")
        node_version: Semantic version (default: "1.0.0")
        event_id: Last applied event ID (generated if not provided)
        offset: Kafka offset (default: 0)
        ack_deadline: Optional ack deadline
        liveness_deadline: Optional liveness deadline

    Returns:
        ModelRegistrationProjection configured for testing
    """
    now = datetime.now(UTC)
    return ModelRegistrationProjection(
        entity_id=uuid4() if entity_id is None else entity_id,
        domain="registration",
        current_state=state,
        node_type=node_type,
        node_version=node_version,
        capabilities=ModelNodeCapabilities(postgres=True, read=True, write=True),
        ack_deadline=ack_deadline,
        liveness_deadline=liveness_deadline,
        last_applied_event_id=uuid4() if event_id is None else event_id,
        last_applied_offset=offset,
        registered_at=now,
        updated_at=now,
    )


def make_sequence(
    sequence: int,
    partition: str | None = "0",
    offset: int | None = None,
) -> ModelSequenceInfo:
    """Create sequence info for testing.

    Args:
        sequence: Monotonic sequence number
        partition: Kafka partition (default: "0")
        offset: Kafka offset (default: same as sequence)

    Returns:
        ModelSequenceInfo configured for testing
    """
    return ModelSequenceInfo(
        sequence=sequence,
        partition=partition,
        offset=offset if offset is not None else sequence,
    )


# =============================================================================
# Schema Initialization Tests
# =============================================================================


class TestSchemaInitialization:
    """Tests for schema initialization from SQL file."""

    async def test_schema_creates_table(
        self,
        pg_pool: asyncpg.Pool,
    ) -> None:
        """Verify schema creates registration_projections table."""
        async with pg_pool.acquire() as conn:
            row = await conn.fetchrow(
                """
                SELECT EXISTS (
                    SELECT 1 FROM information_schema.tables
                    WHERE table_name = 'registration_projections'
                )
                """
            )

        assert row is not None
        assert row[0] is True

    async def test_schema_creates_enum_type(
        self,
        pg_pool: asyncpg.Pool,
    ) -> None:
        """Verify schema creates registration_state enum type."""
        async with pg_pool.acquire() as conn:
            row = await conn.fetchrow(
                """
                SELECT EXISTS (
                    SELECT 1 FROM pg_type WHERE typname = 'registration_state'
                )
                """
            )

        assert row is not None
        assert row[0] is True

    async def test_schema_creates_indexes(
        self,
        pg_pool: asyncpg.Pool,
    ) -> None:
        """Verify schema creates expected indexes."""
        expected_indexes = [
            "idx_registration_ack_deadline",
            "idx_registration_liveness_deadline",
            "idx_registration_current_state",
            "idx_registration_domain_state",
            "idx_registration_last_event_id",
            "idx_registration_capabilities",
            "idx_registration_ack_timeout_scan",
            "idx_registration_liveness_timeout_scan",
        ]

        async with pg_pool.acquire() as conn:
            rows = await conn.fetch(
                """
                SELECT indexname FROM pg_indexes
                WHERE tablename = 'registration_projections'
                """
            )

        index_names = {row["indexname"] for row in rows}

        for expected in expected_indexes:
            assert expected in index_names, f"Missing index: {expected}"

    async def test_schema_is_idempotent(
        self,
        projector: ProjectorRegistration,
    ) -> None:
        """Verify schema can be re-applied without errors (idempotent)."""
        # Schema already applied by fixture; re-apply should succeed
        await projector.initialize_schema()

        # No exception means success


# =============================================================================
# Projector Persistence Tests
# =============================================================================


class TestProjectorPersistence:
    """Tests for ProjectorRegistration.persist()."""

    async def test_persist_inserts_new_projection(
        self,
        projector: ProjectorRegistration,
        pg_pool: asyncpg.Pool,
    ) -> None:
        """Verify persist creates new projection in database."""
        projection = make_projection(offset=100)
        sequence = make_sequence(100)

        result = await projector.persist(
            projection=projection,
            entity_id=projection.entity_id,
            domain=projection.domain,
            sequence_info=sequence,
        )

        assert result is True

        # Verify in database
        async with pg_pool.acquire() as conn:
            row = await conn.fetchrow(
                """
                SELECT entity_id, current_state, node_type, last_applied_offset
                FROM registration_projections
                WHERE entity_id = $1 AND domain = $2
                """,
                projection.entity_id,
                projection.domain,
            )

        assert row is not None
        assert row["entity_id"] == projection.entity_id
        assert row["current_state"] == "pending_registration"
        assert row["node_type"] == "effect"
        assert row["last_applied_offset"] == 100

    async def test_persist_updates_existing_projection(
        self,
        projector: ProjectorRegistration,
    ) -> None:
        """Verify persist updates existing projection with newer sequence."""
        projection = make_projection(offset=100)
        sequence1 = make_sequence(100)

        # Initial insert
        result1 = await projector.persist(
            projection=projection,
            entity_id=projection.entity_id,
            domain=projection.domain,
            sequence_info=sequence1,
        )
        assert result1 is True

        # Update with newer sequence
        projection.current_state = EnumRegistrationState.ACTIVE
        projection.last_applied_offset = 200
        sequence2 = make_sequence(200)

        result2 = await projector.persist(
            projection=projection,
            entity_id=projection.entity_id,
            domain=projection.domain,
            sequence_info=sequence2,
        )
        assert result2 is True

    async def test_persist_rejects_stale_update(
        self,
        projector: ProjectorRegistration,
    ) -> None:
        """Verify persist rejects stale updates (older sequence)."""
        projection = make_projection(offset=200)
        sequence_newer = make_sequence(200)

        # Insert with offset 200
        result1 = await projector.persist(
            projection=projection,
            entity_id=projection.entity_id,
            domain=projection.domain,
            sequence_info=sequence_newer,
        )
        assert result1 is True

        # Attempt update with older offset (100)
        projection.current_state = EnumRegistrationState.REJECTED
        projection.last_applied_offset = 100
        sequence_older = make_sequence(100)

        result2 = await projector.persist(
            projection=projection,
            entity_id=projection.entity_id,
            domain=projection.domain,
            sequence_info=sequence_older,
        )

        # Should be rejected as stale
        assert result2 is False

    async def test_persist_stores_capabilities_as_jsonb(
        self,
        projector: ProjectorRegistration,
        pg_pool: asyncpg.Pool,
    ) -> None:
        """Verify capabilities are stored as JSONB and can be queried."""
        projection = make_projection(offset=100)
        projection.capabilities = ModelNodeCapabilities(
            postgres=True,
            read=True,
            write=True,
            batch_size=100,
        )
        sequence = make_sequence(100)

        await projector.persist(
            projection=projection,
            entity_id=projection.entity_id,
            domain=projection.domain,
            sequence_info=sequence,
        )

        # Query using JSONB operators
        async with pg_pool.acquire() as conn:
            row = await conn.fetchrow(
                """
                SELECT entity_id FROM registration_projections
                WHERE capabilities @> '{"postgres": true}'
                """
            )

        assert row is not None
        assert row["entity_id"] == projection.entity_id

    async def test_persist_stores_deadlines(
        self,
        projector: ProjectorRegistration,
        pg_pool: asyncpg.Pool,
    ) -> None:
        """Verify ack and liveness deadlines are stored correctly."""
        now = datetime.now(UTC)
        ack_deadline = now + timedelta(minutes=5)
        liveness_deadline = now + timedelta(hours=1)

        projection = make_projection(
            offset=100,
            ack_deadline=ack_deadline,
            liveness_deadline=liveness_deadline,
        )
        sequence = make_sequence(100)

        await projector.persist(
            projection=projection,
            entity_id=projection.entity_id,
            domain=projection.domain,
            sequence_info=sequence,
        )

        async with pg_pool.acquire() as conn:
            row = await conn.fetchrow(
                """
                SELECT ack_deadline, liveness_deadline
                FROM registration_projections
                WHERE entity_id = $1
                """,
                projection.entity_id,
            )

        assert row is not None
        assert row["ack_deadline"] is not None
        assert row["liveness_deadline"] is not None


# =============================================================================
# Idempotency Tests
# =============================================================================


class TestIdempotency:
    """Tests for idempotent projection behavior."""

    async def test_same_sequence_is_idempotent(
        self,
        projector: ProjectorRegistration,
        reader: ProjectionReaderRegistration,
    ) -> None:
        """Verify re-projecting same sequence is idempotent (no update)."""
        projection = make_projection(offset=100)
        sequence = make_sequence(100)

        # First persist
        result1 = await projector.persist(
            projection=projection,
            entity_id=projection.entity_id,
            domain=projection.domain,
            sequence_info=sequence,
        )
        assert result1 is True

        # Modify projection but use same sequence
        projection.node_version = "2.0.0"

        # Second persist with same sequence
        result2 = await projector.persist(
            projection=projection,
            entity_id=projection.entity_id,
            domain=projection.domain,
            sequence_info=sequence,
        )

        # Should be rejected (same sequence = stale)
        assert result2 is False

        # Verify original version is preserved
        stored = await reader.get_entity_state(projection.entity_id)
        assert stored is not None
        assert stored.node_version == "1.0.0"

    async def test_is_stale_check(
        self,
        projector: ProjectorRegistration,
    ) -> None:
        """Verify is_stale() correctly identifies stale sequences."""
        projection = make_projection(offset=100)
        sequence = make_sequence(100)

        # Insert projection
        await projector.persist(
            projection=projection,
            entity_id=projection.entity_id,
            domain=projection.domain,
            sequence_info=sequence,
        )

        # Check older sequence is stale
        older_seq = make_sequence(50)
        is_stale = await projector.is_stale(
            entity_id=projection.entity_id,
            domain=projection.domain,
            sequence_info=older_seq,
        )
        assert is_stale is True

        # Check newer sequence is not stale
        newer_seq = make_sequence(150)
        is_stale = await projector.is_stale(
            entity_id=projection.entity_id,
            domain=projection.domain,
            sequence_info=newer_seq,
        )
        assert is_stale is False

    async def test_is_stale_for_nonexistent_entity(
        self,
        projector: ProjectorRegistration,
    ) -> None:
        """Verify is_stale returns False for non-existent entity."""
        non_existent_id = uuid4()
        sequence = make_sequence(100)

        is_stale = await projector.is_stale(
            entity_id=non_existent_id,
            domain="registration",
            sequence_info=sequence,
        )

        # Not stale if entity doesn't exist
        assert is_stale is False


# =============================================================================
# Reader Query Tests
# =============================================================================


class TestReaderQueries:
    """Tests for ProjectionReaderRegistration query methods."""

    async def test_get_entity_state(
        self,
        projector: ProjectorRegistration,
        reader: ProjectionReaderRegistration,
    ) -> None:
        """Verify get_entity_state retrieves full projection."""
        projection = make_projection(
            state=EnumRegistrationState.ACTIVE,
            offset=100,
        )
        sequence = make_sequence(100)

        await projector.persist(
            projection=projection,
            entity_id=projection.entity_id,
            domain=projection.domain,
            sequence_info=sequence,
        )

        result = await reader.get_entity_state(projection.entity_id)

        assert result is not None
        assert result.entity_id == projection.entity_id
        assert result.current_state == EnumRegistrationState.ACTIVE
        assert result.node_type == "effect"
        assert result.capabilities.postgres is True

    async def test_get_entity_state_returns_none_for_missing(
        self,
        reader: ProjectionReaderRegistration,
    ) -> None:
        """Verify get_entity_state returns None for non-existent entity."""
        result = await reader.get_entity_state(uuid4())
        assert result is None

    async def test_get_registration_status(
        self,
        projector: ProjectorRegistration,
        reader: ProjectionReaderRegistration,
    ) -> None:
        """Verify get_registration_status returns just the state."""
        projection = make_projection(
            state=EnumRegistrationState.AWAITING_ACK,
            offset=100,
        )
        sequence = make_sequence(100)

        await projector.persist(
            projection=projection,
            entity_id=projection.entity_id,
            domain=projection.domain,
            sequence_info=sequence,
        )

        result = await reader.get_registration_status(projection.entity_id)

        assert result == EnumRegistrationState.AWAITING_ACK

    async def test_get_by_state(
        self,
        projector: ProjectorRegistration,
        reader: ProjectionReaderRegistration,
    ) -> None:
        """Verify get_by_state filters by FSM state."""
        # Create projections in different states
        active1 = make_projection(state=EnumRegistrationState.ACTIVE, offset=100)
        active2 = make_projection(state=EnumRegistrationState.ACTIVE, offset=100)
        pending = make_projection(
            state=EnumRegistrationState.PENDING_REGISTRATION, offset=100
        )

        for proj in [active1, active2, pending]:
            await projector.persist(
                projection=proj,
                entity_id=proj.entity_id,
                domain=proj.domain,
                sequence_info=make_sequence(100),
            )

        # Query active projections
        results = await reader.get_by_state(EnumRegistrationState.ACTIVE)

        assert len(results) == 2
        assert all(p.current_state == EnumRegistrationState.ACTIVE for p in results)

    async def test_count_by_state(
        self,
        projector: ProjectorRegistration,
        reader: ProjectionReaderRegistration,
    ) -> None:
        """Verify count_by_state returns correct counts per state."""
        # Create projections in different states
        projections = [
            make_projection(state=EnumRegistrationState.ACTIVE, offset=100),
            make_projection(state=EnumRegistrationState.ACTIVE, offset=100),
            make_projection(state=EnumRegistrationState.ACTIVE, offset=100),
            make_projection(
                state=EnumRegistrationState.PENDING_REGISTRATION, offset=100
            ),
            make_projection(state=EnumRegistrationState.REJECTED, offset=100),
        ]

        for proj in projections:
            await projector.persist(
                projection=proj,
                entity_id=proj.entity_id,
                domain=proj.domain,
                sequence_info=make_sequence(100),
            )

        counts = await reader.count_by_state()

        assert counts[EnumRegistrationState.ACTIVE] == 3
        assert counts[EnumRegistrationState.PENDING_REGISTRATION] == 1
        assert counts[EnumRegistrationState.REJECTED] == 1


# =============================================================================
# Deadline Scan Tests (C2 Durable Timeout)
# =============================================================================


class TestDeadlineScans:
    """Tests for deadline scanning queries (C2 durable timeout support)."""

    async def test_get_overdue_ack_registrations(
        self,
        projector: ProjectorRegistration,
        reader: ProjectionReaderRegistration,
    ) -> None:
        """Verify get_overdue_ack_registrations finds overdue ack deadlines."""
        now = datetime.now(UTC)
        past = now - timedelta(minutes=10)
        future = now + timedelta(minutes=10)

        # Overdue ack (should be found)
        overdue = make_projection(
            state=EnumRegistrationState.AWAITING_ACK,
            offset=100,
            ack_deadline=past,
        )

        # Future ack (should not be found)
        not_due = make_projection(
            state=EnumRegistrationState.AWAITING_ACK,
            offset=100,
            ack_deadline=future,
        )

        # Active (wrong state, should not be found)
        wrong_state = make_projection(
            state=EnumRegistrationState.ACTIVE,
            offset=100,
            ack_deadline=past,
        )

        for proj in [overdue, not_due, wrong_state]:
            await projector.persist(
                projection=proj,
                entity_id=proj.entity_id,
                domain=proj.domain,
                sequence_info=make_sequence(100),
            )

        results = await reader.get_overdue_ack_registrations(now)

        assert len(results) == 1
        assert results[0].entity_id == overdue.entity_id

    async def test_get_overdue_ack_excludes_already_emitted(
        self,
        projector: ProjectorRegistration,
        reader: ProjectionReaderRegistration,
        pg_pool: asyncpg.Pool,
    ) -> None:
        """Verify overdue ack scan excludes already-emitted timeouts."""
        now = datetime.now(UTC)
        past = now - timedelta(minutes=10)

        # Overdue but already emitted
        overdue = make_projection(
            state=EnumRegistrationState.AWAITING_ACK,
            offset=100,
            ack_deadline=past,
        )

        await projector.persist(
            projection=overdue,
            entity_id=overdue.entity_id,
            domain=overdue.domain,
            sequence_info=make_sequence(100),
        )

        # Mark as emitted
        async with pg_pool.acquire() as conn:
            await conn.execute(
                """
                UPDATE registration_projections
                SET ack_timeout_emitted_at = $1
                WHERE entity_id = $2
                """,
                now,
                overdue.entity_id,
            )

        results = await reader.get_overdue_ack_registrations(now)

        assert len(results) == 0

    async def test_get_overdue_liveness_registrations(
        self,
        projector: ProjectorRegistration,
        reader: ProjectionReaderRegistration,
    ) -> None:
        """Verify get_overdue_liveness_registrations finds expired liveness."""
        now = datetime.now(UTC)
        past = now - timedelta(minutes=10)
        future = now + timedelta(minutes=10)

        # Overdue liveness (should be found)
        overdue = make_projection(
            state=EnumRegistrationState.ACTIVE,
            offset=100,
            liveness_deadline=past,
        )

        # Future liveness (should not be found)
        not_due = make_projection(
            state=EnumRegistrationState.ACTIVE,
            offset=100,
            liveness_deadline=future,
        )

        # Wrong state (should not be found)
        wrong_state = make_projection(
            state=EnumRegistrationState.AWAITING_ACK,
            offset=100,
            liveness_deadline=past,
        )

        for proj in [overdue, not_due, wrong_state]:
            await projector.persist(
                projection=proj,
                entity_id=proj.entity_id,
                domain=proj.domain,
                sequence_info=make_sequence(100),
            )

        results = await reader.get_overdue_liveness_registrations(now)

        assert len(results) == 1
        assert results[0].entity_id == overdue.entity_id


# =============================================================================
# Ordering and Sequencing Tests
# =============================================================================


class TestOrdering:
    """Tests for ordering enforcement with multiple events."""

    async def test_out_of_order_events_handled_correctly(
        self,
        projector: ProjectorRegistration,
        reader: ProjectionReaderRegistration,
    ) -> None:
        """Verify out-of-order events are handled correctly."""
        entity_id = uuid4()
        event_id_1 = uuid4()
        event_id_2 = uuid4()
        event_id_3 = uuid4()

        now = datetime.now(UTC)

        # Create projections with different offsets
        proj_offset_100 = ModelRegistrationProjection(
            entity_id=entity_id,
            current_state=EnumRegistrationState.PENDING_REGISTRATION,
            node_type="effect",
            last_applied_event_id=event_id_1,
            last_applied_offset=100,
            registered_at=now,
            updated_at=now,
        )

        proj_offset_200 = ModelRegistrationProjection(
            entity_id=entity_id,
            current_state=EnumRegistrationState.ACCEPTED,
            node_type="effect",
            last_applied_event_id=event_id_2,
            last_applied_offset=200,
            registered_at=now,
            updated_at=now,
        )

        proj_offset_150 = ModelRegistrationProjection(
            entity_id=entity_id,
            current_state=EnumRegistrationState.REJECTED,  # Would be wrong state
            node_type="effect",
            last_applied_event_id=event_id_3,
            last_applied_offset=150,
            registered_at=now,
            updated_at=now,
        )

        # Apply events out of order: 100, 200, 150
        result_100 = await projector.persist(
            projection=proj_offset_100,
            entity_id=entity_id,
            domain="registration",
            sequence_info=make_sequence(100),
        )
        assert result_100 is True

        result_200 = await projector.persist(
            projection=proj_offset_200,
            entity_id=entity_id,
            domain="registration",
            sequence_info=make_sequence(200),
        )
        assert result_200 is True

        # Out-of-order event at 150 should be rejected
        result_150 = await projector.persist(
            projection=proj_offset_150,
            entity_id=entity_id,
            domain="registration",
            sequence_info=make_sequence(150),
        )
        assert result_150 is False

        # Final state should be from offset 200
        stored = await reader.get_entity_state(entity_id)
        assert stored is not None
        assert stored.current_state == EnumRegistrationState.ACCEPTED
        assert stored.last_applied_offset == 200

    async def test_sequence_progression_through_states(
        self,
        projector: ProjectorRegistration,
        reader: ProjectionReaderRegistration,
    ) -> None:
        """Verify correct state progression with increasing sequences."""
        entity_id = uuid4()
        now = datetime.now(UTC)

        states_and_offsets = [
            (EnumRegistrationState.PENDING_REGISTRATION, 100),
            (EnumRegistrationState.ACCEPTED, 200),
            (EnumRegistrationState.AWAITING_ACK, 300),
            (EnumRegistrationState.ACK_RECEIVED, 400),
            (EnumRegistrationState.ACTIVE, 500),
        ]

        for state, offset in states_and_offsets:
            proj = ModelRegistrationProjection(
                entity_id=entity_id,
                current_state=state,
                node_type="effect",
                last_applied_event_id=uuid4(),
                last_applied_offset=offset,
                registered_at=now,
                updated_at=now,
            )

            result = await projector.persist(
                projection=proj,
                entity_id=entity_id,
                domain="registration",
                sequence_info=make_sequence(offset),
            )
            assert result is True

        # Final state should be ACTIVE
        stored = await reader.get_entity_state(entity_id)
        assert stored is not None
        assert stored.current_state == EnumRegistrationState.ACTIVE
        assert stored.last_applied_offset == 500


# =============================================================================
# Multi-Domain Tests
# =============================================================================


class TestMultiDomain:
    """Tests for multi-domain projection support."""

    async def test_same_entity_different_domains(
        self,
        projector: ProjectorRegistration,
        reader: ProjectionReaderRegistration,
    ) -> None:
        """Verify same entity can have projections in different domains."""
        entity_id = uuid4()
        now = datetime.now(UTC)

        # Projection in domain "registration"
        proj_reg = ModelRegistrationProjection(
            entity_id=entity_id,
            domain="registration",
            current_state=EnumRegistrationState.ACTIVE,
            node_type="effect",
            last_applied_event_id=uuid4(),
            last_applied_offset=100,
            registered_at=now,
            updated_at=now,
        )

        # Projection in domain "test-domain"
        proj_test = ModelRegistrationProjection(
            entity_id=entity_id,
            domain="test-domain",
            current_state=EnumRegistrationState.PENDING_REGISTRATION,
            node_type="effect",
            last_applied_event_id=uuid4(),
            last_applied_offset=50,
            registered_at=now,
            updated_at=now,
        )

        await projector.persist(
            projection=proj_reg,
            entity_id=entity_id,
            domain="registration",
            sequence_info=make_sequence(100),
        )

        await projector.persist(
            projection=proj_test,
            entity_id=entity_id,
            domain="test-domain",
            sequence_info=make_sequence(50),
        )

        # Query each domain
        reg_result = await reader.get_entity_state(entity_id, domain="registration")
        test_result = await reader.get_entity_state(entity_id, domain="test-domain")

        assert reg_result is not None
        assert reg_result.current_state == EnumRegistrationState.ACTIVE

        assert test_result is not None
        assert test_result.current_state == EnumRegistrationState.PENDING_REGISTRATION


# =============================================================================
# Concurrent Projection Tests
# =============================================================================


class TestConcurrency:
    """Tests for concurrent projection operations."""

    async def test_concurrent_projections_for_different_entities(
        self,
        projector: ProjectorRegistration,
        reader: ProjectionReaderRegistration,
    ) -> None:
        """Verify concurrent projections for different entities succeed."""
        # Create 10 projections concurrently
        projections = [make_projection(offset=100) for _ in range(10)]

        async def persist_projection(proj: ModelRegistrationProjection) -> bool:
            return await projector.persist(
                projection=proj,
                entity_id=proj.entity_id,
                domain=proj.domain,
                sequence_info=make_sequence(100),
            )

        results = await asyncio.gather(*[persist_projection(p) for p in projections])

        # All should succeed
        assert all(results)

        # Verify all stored
        counts = await reader.count_by_state()
        assert counts[EnumRegistrationState.PENDING_REGISTRATION] == 10

    async def test_concurrent_updates_same_entity(
        self,
        projector: ProjectorRegistration,
        reader: ProjectionReaderRegistration,
    ) -> None:
        """Verify concurrent updates to same entity handle ordering correctly."""
        entity_id = uuid4()
        now = datetime.now(UTC)

        # Create projections with different offsets
        projections = []
        for offset in range(100, 1100, 100):  # 100, 200, ..., 1000
            proj = ModelRegistrationProjection(
                entity_id=entity_id,
                current_state=EnumRegistrationState.PENDING_REGISTRATION,
                node_type="effect",
                node_version=f"1.0.{offset}",  # Use version to track which won
                last_applied_event_id=uuid4(),
                last_applied_offset=offset,
                registered_at=now,
                updated_at=now,
            )
            projections.append((proj, offset))

        # Shuffle order to simulate out-of-order arrival
        random.shuffle(projections)

        async def persist_projection(
            proj: ModelRegistrationProjection, offset: int
        ) -> bool:
            return await projector.persist(
                projection=proj,
                entity_id=entity_id,
                domain="registration",
                sequence_info=make_sequence(offset),
            )

        await asyncio.gather(*[persist_projection(p, o) for p, o in projections])

        # Final state should be from highest offset
        stored = await reader.get_entity_state(entity_id)
        assert stored is not None
        assert stored.last_applied_offset == 1000
        assert stored.node_version == "1.0.1000"
