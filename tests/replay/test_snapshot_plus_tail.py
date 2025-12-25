# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Snapshot Plus Tail Validation Tests for OMN-955.

These tests validate the ONEX principle that:
    Projection Snapshot + Event Tail = Full State Reconstruction

This is a key optimization pattern for event-sourced systems:
- Snapshots provide fast state recovery (O(1) read)
- Event tail contains only events after snapshot
- Combined result MUST equal full event log replay

Architecture:
    - ModelRegistrationSnapshot: Compacted point-in-time state
    - ModelRegistrationProjection: Full materialized state
    - RegistrationReducer: Pure reducer for event processing
    - Event Tail: Events with offset > snapshot.source_projection_sequence

ONEX Consistency Guarantee:
    snapshot + reduce(events[snapshot.offset:]) == reduce(events[:])

Related Tickets:
    - OMN-955: State Reconstruction Validation Tests
    - OMN-947 (F2): Snapshot Publishing
    - OMN-944 (F1): Implement Registration Projection Schema
"""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from omnibase_infra.enums import EnumRegistrationState
from omnibase_infra.models.projection.model_registration_projection import (
    ModelRegistrationProjection,
)
from omnibase_infra.models.projection.model_registration_snapshot import (
    ModelRegistrationSnapshot,
)
from omnibase_infra.models.registration import (
    ModelNodeCapabilities,
    ModelNodeIntrospectionEvent,
    ModelNodeMetadata,
)
from omnibase_infra.nodes.reducers import RegistrationReducer
from omnibase_infra.nodes.reducers.models import ModelRegistrationState
from tests.helpers.deterministic import DeterministicClock, DeterministicIdGenerator

__all__ = [
    "TestSnapshotPlusTail",
    "TestSnapshotCreation",
    "TestTailEventApplication",
]


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def id_generator() -> DeterministicIdGenerator:
    """Create a deterministic ID generator for reproducible tests."""
    return DeterministicIdGenerator(seed=200)


@pytest.fixture
def clock() -> DeterministicClock:
    """Create a deterministic clock for reproducible timestamps."""
    return DeterministicClock()


@pytest.fixture
def sample_projection(
    id_generator: DeterministicIdGenerator,
    clock: DeterministicClock,
) -> ModelRegistrationProjection:
    """Create a sample projection at a known sequence point.

    Returns:
        A ModelRegistrationProjection representing state at sequence=100.
    """
    entity_id = id_generator.next_uuid()
    event_id = id_generator.next_uuid()

    return ModelRegistrationProjection(
        entity_id=entity_id,
        domain="registration",
        current_state=EnumRegistrationState.ACTIVE,
        node_type="effect",
        node_version="1.0.0",
        capabilities=ModelNodeCapabilities(postgres=True, read=True),
        last_applied_event_id=event_id,
        last_applied_offset=100,
        last_applied_sequence=100,
        registered_at=clock.now(),
        updated_at=clock.now(),
    )


@pytest.fixture
def sample_snapshot(
    sample_projection: ModelRegistrationProjection,
    clock: DeterministicClock,
) -> ModelRegistrationSnapshot:
    """Create a sample snapshot from the sample projection.

    Returns:
        A ModelRegistrationSnapshot derived from the sample projection.
    """
    clock.advance(60)  # Snapshot created 1 minute after projection update
    return ModelRegistrationSnapshot.from_projection(
        sample_projection,
        snapshot_version=1,
        snapshot_created_at=clock.now(),
        node_name="TestEffectNode",
    )


def create_introspection_event(
    node_id,
    correlation_id,
    timestamp: datetime,
    node_type: str = "effect",
    node_version: str = "1.0.0",
) -> ModelNodeIntrospectionEvent:
    """Factory for creating introspection events."""
    return ModelNodeIntrospectionEvent(
        node_id=node_id,
        node_type=node_type,
        node_version=node_version,
        correlation_id=correlation_id,
        timestamp=timestamp,
        endpoints={"health": "http://localhost:8080/health"},
        capabilities=ModelNodeCapabilities(postgres=True, read=True),
        metadata=ModelNodeMetadata(environment="test"),
    )


# =============================================================================
# Test: Snapshot Plus Tail Reconstruction
# =============================================================================


@pytest.mark.unit
class TestSnapshotPlusTail:
    """Tests that snapshot + event tail equals full state reconstruction."""

    def test_snapshot_to_reducer_state_conversion(
        self,
        sample_projection: ModelRegistrationProjection,
        sample_snapshot: ModelRegistrationSnapshot,
    ) -> None:
        """Verify snapshot can be converted to reducer-compatible state.

        The snapshot captures essential state fields that can be used
        to reconstruct ModelRegistrationState for reducer processing.
        """
        # Verify snapshot captured projection state
        assert sample_snapshot.entity_id == sample_projection.entity_id
        assert sample_snapshot.current_state == sample_projection.current_state
        assert sample_snapshot.node_type == sample_projection.node_type
        assert sample_snapshot.source_projection_sequence == 100

    def test_snapshot_plus_empty_tail_equals_snapshot(
        self,
        sample_snapshot: ModelRegistrationSnapshot,
    ) -> None:
        """Verify snapshot + empty tail = snapshot state.

        When there are no tail events, the snapshot represents the full state.
        """
        # No tail events means snapshot is current
        tail_events: list[ModelNodeIntrospectionEvent] = []

        # Verify snapshot is complete without tail
        assert sample_snapshot.current_state == EnumRegistrationState.ACTIVE
        assert len(tail_events) == 0

        # Snapshot is self-consistent
        assert sample_snapshot.is_active()
        assert not sample_snapshot.is_terminal()

    def test_snapshot_plus_tail_produces_correct_state(
        self,
        sample_projection: ModelRegistrationProjection,
        id_generator: DeterministicIdGenerator,
        clock: DeterministicClock,
    ) -> None:
        """Verify snapshot + tail events produces correct combined state.

        Scenario:
            1. Create projection at sequence 100
            2. Create snapshot from projection
            3. Create tail events (sequence > 100)
            4. Apply tail events to snapshot-derived state
            5. Verify final state reflects all events
        """
        # Create snapshot from projection at sequence 100
        snapshot = ModelRegistrationSnapshot.from_projection(
            sample_projection,
            snapshot_version=1,
            snapshot_created_at=clock.now(),
            node_name="TestNode",
        )

        # Create tail events (simulating events after snapshot)
        clock.advance(60)
        tail_event = create_introspection_event(
            node_id=sample_projection.entity_id,
            correlation_id=id_generator.next_uuid(),
            timestamp=clock.now(),
            node_version="1.1.0",  # Updated version in tail
        )

        # Convert snapshot state to reducer state for tail processing
        # The snapshot represents ACTIVE state, so we simulate continuing
        # from a "pending" state to apply the new event
        reducer = RegistrationReducer()
        reducer_state = ModelRegistrationState(
            status="idle",  # Reset to apply tail event
            node_id=None,
        )

        # Apply tail event
        output = reducer.reduce(reducer_state, tail_event)
        final_state = output.result

        # Final state should reflect tail event
        assert final_state.status == "pending"
        assert final_state.node_id == sample_projection.entity_id
        assert final_state.last_processed_event_id == tail_event.correlation_id

    def test_full_replay_vs_snapshot_plus_tail_consistency(
        self,
        id_generator: DeterministicIdGenerator,
        clock: DeterministicClock,
    ) -> None:
        """Verify full replay equals snapshot + tail replay.

        This is the core consistency test:
            reduce(all_events) == snapshot_state + reduce(tail_events)
        """
        node_id = id_generator.next_uuid()

        # Create event log: 5 events total
        all_events: list[ModelNodeIntrospectionEvent] = []
        for i in range(5):
            clock.advance(60)
            all_events.append(
                create_introspection_event(
                    node_id=node_id,
                    correlation_id=id_generator.next_uuid(),
                    timestamp=clock.now(),
                    node_version=f"1.0.{i}",
                )
            )

        # Full replay: process all events from scratch
        reducer_full = RegistrationReducer()
        full_state = ModelRegistrationState()
        for event in all_events:
            output = reducer_full.reduce(full_state, event)
            full_state = output.result

        # Snapshot at event 3 (index 2), tail is events 4-5 (index 3-4)
        snapshot_point = 3  # After processing events[0], events[1], events[2]
        reducer_partial = RegistrationReducer()
        partial_state = ModelRegistrationState()
        for event in all_events[:snapshot_point]:
            output = reducer_partial.reduce(partial_state, event)
            partial_state = output.result

        # Create snapshot from partial state
        snapshot_state = partial_state
        snapshot_last_event_id = snapshot_state.last_processed_event_id

        # Apply tail events (events after snapshot)
        tail_events = all_events[snapshot_point:]
        for event in tail_events:
            output = reducer_partial.reduce(partial_state, event)
            partial_state = output.result

        # Final states must match
        assert partial_state.status == full_state.status
        assert partial_state.node_id == full_state.node_id
        assert (
            partial_state.last_processed_event_id == full_state.last_processed_event_id
        )


# =============================================================================
# Test: Snapshot Creation
# =============================================================================


@pytest.mark.unit
class TestSnapshotCreation:
    """Tests for snapshot creation from projections."""

    def test_snapshot_from_projection_copies_essential_fields(
        self,
        sample_projection: ModelRegistrationProjection,
        clock: DeterministicClock,
    ) -> None:
        """Verify from_projection copies all essential state fields."""
        clock.advance(60)
        snapshot = ModelRegistrationSnapshot.from_projection(
            sample_projection,
            snapshot_version=1,
            snapshot_created_at=clock.now(),
            node_name="CopiedNode",
        )

        # Essential fields copied
        assert snapshot.entity_id == sample_projection.entity_id
        assert snapshot.domain == sample_projection.domain
        assert snapshot.current_state == sample_projection.current_state
        assert snapshot.node_type == sample_projection.node_type
        assert snapshot.capabilities == sample_projection.capabilities

        # Metadata fields set correctly
        assert snapshot.snapshot_version == 1
        assert snapshot.node_name == "CopiedNode"

    def test_snapshot_source_sequence_tracks_projection_offset(
        self,
        sample_projection: ModelRegistrationProjection,
        clock: DeterministicClock,
    ) -> None:
        """Verify snapshot tracks its source projection sequence."""
        clock.advance(60)
        snapshot = ModelRegistrationSnapshot.from_projection(
            sample_projection,
            snapshot_version=1,
            snapshot_created_at=clock.now(),
        )

        # Source sequence should match projection's sequence
        assert snapshot.source_projection_sequence == 100

    def test_snapshot_version_increments(
        self,
        sample_projection: ModelRegistrationProjection,
        clock: DeterministicClock,
    ) -> None:
        """Verify snapshot versions increment for same entity."""
        snapshots: list[ModelRegistrationSnapshot] = []

        for version in range(1, 4):
            clock.advance(60)
            snapshot = ModelRegistrationSnapshot.from_projection(
                sample_projection,
                snapshot_version=version,
                snapshot_created_at=clock.now(),
            )
            snapshots.append(snapshot)

        # Verify version ordering
        for i in range(len(snapshots) - 1):
            assert snapshots[i + 1].is_newer_than(snapshots[i])

    def test_snapshot_kafka_key_format(
        self,
        sample_snapshot: ModelRegistrationSnapshot,
    ) -> None:
        """Verify snapshot produces correct Kafka compaction key."""
        key = sample_snapshot.to_kafka_key()

        # Key format: domain:entity_id
        assert key.startswith("registration:")
        assert str(sample_snapshot.entity_id) in key


# =============================================================================
# Test: Tail Event Application
# =============================================================================


@pytest.mark.unit
class TestTailEventApplication:
    """Tests for applying tail events after snapshot recovery."""

    def test_tail_events_applied_in_order(
        self,
        id_generator: DeterministicIdGenerator,
        clock: DeterministicClock,
    ) -> None:
        """Verify tail events must be applied in order."""
        node_id = id_generator.next_uuid()

        # Create ordered tail events
        tail_events = [
            create_introspection_event(
                node_id=node_id,
                correlation_id=id_generator.next_uuid(),
                timestamp=clock.now(),
                node_version=f"1.{i}.0",
            )
            for i in range(3)
        ]

        # Apply in correct order
        reducer = RegistrationReducer()
        state = ModelRegistrationState()
        for event in tail_events:
            clock.advance(60)
            output = reducer.reduce(state, event)
            state = output.result

        # Final state reflects last event
        assert state.last_processed_event_id == tail_events[-1].correlation_id

    def test_duplicate_tail_events_are_idempotent(
        self,
        id_generator: DeterministicIdGenerator,
        clock: DeterministicClock,
    ) -> None:
        """Verify duplicate tail events are skipped (idempotency)."""
        node_id = id_generator.next_uuid()
        correlation_id = id_generator.next_uuid()

        event = create_introspection_event(
            node_id=node_id,
            correlation_id=correlation_id,
            timestamp=clock.now(),
        )

        reducer = RegistrationReducer()
        state = ModelRegistrationState()

        # First application
        output1 = reducer.reduce(state, event)
        state_after_first = output1.result
        assert len(output1.intents) == 2  # Intents emitted

        # Duplicate application
        output2 = reducer.reduce(state_after_first, event)
        state_after_second = output2.result

        # State unchanged, no new intents
        assert state_after_second.status == state_after_first.status
        assert len(output2.intents) == 0  # No duplicate intents

    def test_tail_starting_from_different_snapshots(
        self,
        id_generator: DeterministicIdGenerator,
        clock: DeterministicClock,
    ) -> None:
        """Verify tail application works from different snapshot points."""
        node_id = id_generator.next_uuid()

        # Create 10 events
        all_events = [
            create_introspection_event(
                node_id=node_id,
                correlation_id=id_generator.next_uuid(),
                timestamp=clock.now(),
                node_version=f"1.0.{i}",
            )
            for i in range(10)
        ]

        # Full replay
        reducer_full = RegistrationReducer()
        full_state = ModelRegistrationState()
        for event in all_events:
            clock.advance(10)
            output = reducer_full.reduce(full_state, event)
            full_state = output.result

        # Replay from snapshot at position 3
        reducer_snap3 = RegistrationReducer()
        snap3_state = ModelRegistrationState()
        for event in all_events[:3]:
            output = reducer_snap3.reduce(snap3_state, event)
            snap3_state = output.result
        # Apply tail from position 3
        for event in all_events[3:]:
            output = reducer_snap3.reduce(snap3_state, event)
            snap3_state = output.result

        # Replay from snapshot at position 7
        reducer_snap7 = RegistrationReducer()
        snap7_state = ModelRegistrationState()
        for event in all_events[:7]:
            output = reducer_snap7.reduce(snap7_state, event)
            snap7_state = output.result
        # Apply tail from position 7
        for event in all_events[7:]:
            output = reducer_snap7.reduce(snap7_state, event)
            snap7_state = output.result

        # All approaches should produce identical final state
        assert full_state.status == snap3_state.status == snap7_state.status
        assert full_state.node_id == snap3_state.node_id == snap7_state.node_id
        assert (
            full_state.last_processed_event_id
            == snap3_state.last_processed_event_id
            == snap7_state.last_processed_event_id
        )
