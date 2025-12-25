# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Event Sequence Capture Tests for OMN-955.

This module tests the ability to capture event sequences from registration
workflows for later replay verification. Event sequences are the foundation
of replay testing - they record the events and expected outcomes that can
be replayed to verify reducer determinism.

Test Coverage:
    - Capturing events from registration workflows
    - Creating event log fixtures
    - Serializing and deserializing event sequences
    - Validating sequence integrity

Architecture:
    Event sequences capture the following for each event:
    1. The event itself (ModelNodeIntrospectionEvent)
    2. Expected state status after processing
    3. Expected intent count
    4. Sequence number for ordering

    These sequences enable:
    - Determinism testing (replay produces same output)
    - Idempotency testing (duplicate events are skipped)
    - Out-of-order detection (sequence violations are caught)

Related:
    - conftest.py: EventSequenceLog and EventFactory fixtures
    - RegistrationReducer: Pure reducer under test
    - OMN-955: Event Replay Verification ticket
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from typing import TYPE_CHECKING
from uuid import UUID

import pytest

from omnibase_infra.models.registration import (
    ModelNodeCapabilities,
    ModelNodeIntrospectionEvent,
    ModelNodeMetadata,
)
from omnibase_infra.nodes.reducers import RegistrationReducer
from omnibase_infra.nodes.reducers.models import ModelRegistrationState

if TYPE_CHECKING:
    from tests.replay.conftest import EventFactory, EventSequenceLog


# =============================================================================
# Constants
# =============================================================================

EXPECTED_REGISTRATION_INTENTS = 2  # Consul + PostgreSQL


# =============================================================================
# Event Sequence Capture Tests
# =============================================================================


@pytest.mark.unit
class TestEventSequenceCapture:
    """Tests for capturing event sequences from registration workflows."""

    def test_capture_single_event_sequence(
        self,
        reducer: RegistrationReducer,
        initial_state: ModelRegistrationState,
        event_factory: EventFactory,
        event_sequence_log: EventSequenceLog,
    ) -> None:
        """Test capturing a single event in the sequence log.

        Verifies that:
        - Event can be added to the sequence log
        - Expected status and intent count are recorded
        - Sequence number is assigned correctly
        """
        event = event_factory.create_event()
        output = reducer.reduce(initial_state, event)

        event_sequence_log.append(
            event=event,
            expected_status=output.result.status,
            expected_intent_count=len(output.intents),
        )

        assert len(event_sequence_log) == 1
        entry = event_sequence_log.entries[0]
        assert entry.event == event
        assert entry.expected_status == "pending"
        assert entry.expected_intent_count == EXPECTED_REGISTRATION_INTENTS
        assert entry.sequence_number == 1

    def test_capture_multiple_events_sequence(
        self,
        reducer: RegistrationReducer,
        event_factory: EventFactory,
        event_sequence_log: EventSequenceLog,
    ) -> None:
        """Test capturing multiple events in sequence.

        Verifies that:
        - Multiple events can be captured
        - Each event gets correct sequence number
        - Events are stored in order
        """
        events = event_factory.create_event_sequence(count=5)
        state = ModelRegistrationState()

        for event in events:
            output = reducer.reduce(state, event)
            event_sequence_log.append(
                event=event,
                expected_status=output.result.status,
                expected_intent_count=len(output.intents),
            )
            # Use fresh state for each event (separate registrations)
            state = ModelRegistrationState()

        assert len(event_sequence_log) == 5

        for i, entry in enumerate(event_sequence_log.entries):
            assert entry.sequence_number == i + 1
            assert entry.expected_status == "pending"
            assert entry.expected_intent_count == EXPECTED_REGISTRATION_INTENTS

    def test_capture_workflow_with_state_transitions(
        self,
        reducer: RegistrationReducer,
        event_factory: EventFactory,
        event_sequence_log: EventSequenceLog,
    ) -> None:
        """Test capturing a workflow with multiple state transitions.

        Verifies that:
        - State transitions are captured correctly
        - Each step's expected status reflects the transition
        """
        # Create first event
        event1 = event_factory.create_event(node_type="effect")
        state = ModelRegistrationState()
        output1 = reducer.reduce(state, event1)

        event_sequence_log.append(
            event=event1,
            expected_status=output1.result.status,
            expected_intent_count=len(output1.intents),
        )

        # Create second event (different node)
        event2 = event_factory.create_event(
            node_type="compute",
            advance_time_seconds=60,
        )
        state2 = ModelRegistrationState()  # Fresh state for different node
        output2 = reducer.reduce(state2, event2)

        event_sequence_log.append(
            event=event2,
            expected_status=output2.result.status,
            expected_intent_count=len(output2.intents),
        )

        assert len(event_sequence_log) == 2
        assert event_sequence_log.entries[0].expected_status == "pending"
        assert event_sequence_log.entries[1].expected_status == "pending"


@pytest.mark.unit
class TestEventSequenceSerialization:
    """Tests for serializing and deserializing event sequences."""

    def test_serialize_empty_sequence(
        self,
        event_sequence_log: EventSequenceLog,
    ) -> None:
        """Test serializing an empty event sequence.

        Verifies that:
        - Empty sequence serializes correctly
        - Initial state is preserved
        """
        data = event_sequence_log.to_dict()

        assert "initial_state" in data
        assert "entries" in data
        assert len(data["entries"]) == 0
        assert data["initial_state"]["status"] == "idle"

    def test_serialize_sequence_with_events(
        self,
        reducer: RegistrationReducer,
        event_factory: EventFactory,
        event_sequence_log: EventSequenceLog,
    ) -> None:
        """Test serializing a sequence with events.

        Verifies that:
        - Events are serialized correctly
        - Expected status and intent count are preserved
        - Sequence numbers are preserved
        """
        events = event_factory.create_event_sequence(count=3)
        state = ModelRegistrationState()

        for event in events:
            output = reducer.reduce(state, event)
            event_sequence_log.append(
                event=event,
                expected_status=output.result.status,
                expected_intent_count=len(output.intents),
            )
            state = ModelRegistrationState()

        data = event_sequence_log.to_dict()

        assert len(data["entries"]) == 3
        for i, entry_data in enumerate(data["entries"]):
            assert entry_data["sequence_number"] == i + 1
            assert entry_data["expected_status"] == "pending"
            assert entry_data["expected_intent_count"] == EXPECTED_REGISTRATION_INTENTS
            assert "event" in entry_data

    def test_deserialize_sequence(
        self,
        reducer: RegistrationReducer,
        event_factory: EventFactory,
        event_sequence_log: EventSequenceLog,
    ) -> None:
        """Test deserializing a sequence from dictionary.

        Verifies that:
        - Deserialized sequence matches original
        - Events are properly reconstructed
        - Initial state is preserved
        """
        from tests.replay.conftest import EventSequenceLog

        # Create and serialize a sequence
        events = event_factory.create_event_sequence(count=2)
        state = ModelRegistrationState()

        for event in events:
            output = reducer.reduce(state, event)
            event_sequence_log.append(
                event=event,
                expected_status=output.result.status,
                expected_intent_count=len(output.intents),
            )
            state = ModelRegistrationState()

        # Serialize and deserialize
        data = event_sequence_log.to_dict()
        restored_log = EventSequenceLog.from_dict(data)

        # Verify restoration
        assert len(restored_log) == len(event_sequence_log)
        assert restored_log.initial_state == event_sequence_log.initial_state

        for orig, restored in zip(
            event_sequence_log.entries, restored_log.entries, strict=True
        ):
            assert restored.sequence_number == orig.sequence_number
            assert restored.expected_status == orig.expected_status
            assert restored.expected_intent_count == orig.expected_intent_count
            # Events should be equivalent
            assert restored.event.node_id == orig.event.node_id
            assert restored.event.node_type == orig.event.node_type
            assert restored.event.correlation_id == orig.event.correlation_id

    def test_serialize_to_json_and_back(
        self,
        reducer: RegistrationReducer,
        event_factory: EventFactory,
        event_sequence_log: EventSequenceLog,
    ) -> None:
        """Test JSON serialization round-trip.

        Verifies that:
        - Sequence can be serialized to JSON string
        - JSON can be parsed back to dictionary
        - Deserialization produces equivalent log
        """
        from tests.replay.conftest import EventSequenceLog

        # Create sequence
        events = event_factory.create_event_sequence(count=2)
        state = ModelRegistrationState()

        for event in events:
            output = reducer.reduce(state, event)
            event_sequence_log.append(
                event=event,
                expected_status=output.result.status,
                expected_intent_count=len(output.intents),
            )
            state = ModelRegistrationState()

        # Full JSON round-trip
        json_string = json.dumps(event_sequence_log.to_dict(), default=str)
        parsed_data = json.loads(json_string)
        restored_log = EventSequenceLog.from_dict(parsed_data)

        assert len(restored_log) == len(event_sequence_log)


@pytest.mark.unit
class TestEventLogFixtures:
    """Tests for creating event log fixtures."""

    def test_create_registration_workflow_fixture(
        self,
        complete_registration_sequence: list[ModelNodeIntrospectionEvent],
    ) -> None:
        """Test creating a complete registration workflow fixture.

        Verifies that:
        - Fixture creates expected number of events
        - Events have valid structure
        """
        assert len(complete_registration_sequence) == 5

        for event in complete_registration_sequence:
            assert event.node_id is not None
            assert event.node_type in ("effect", "compute", "reducer", "orchestrator")
            assert event.correlation_id is not None
            assert event.timestamp is not None

    def test_create_multi_node_type_fixture(
        self,
        multi_node_type_sequence: list[tuple[str, ModelNodeIntrospectionEvent]],
    ) -> None:
        """Test creating a multi-node-type fixture.

        Verifies that:
        - All four node types are represented
        - Each event is valid
        """
        assert len(multi_node_type_sequence) == 4

        node_types = {entry[0] for entry in multi_node_type_sequence}
        assert node_types == {"effect", "compute", "reducer", "orchestrator"}

    def test_capture_and_replay_fixture(
        self,
        reducer: RegistrationReducer,
        complete_registration_sequence: list[ModelNodeIntrospectionEvent],
    ) -> None:
        """Test capturing and replaying a complete workflow fixture.

        Verifies that:
        - Workflow can be captured
        - Capture produces consistent results
        """
        from tests.replay.conftest import EventSequenceLog

        log = EventSequenceLog()

        for event in complete_registration_sequence:
            state = ModelRegistrationState()
            output = reducer.reduce(state, event)
            log.append(
                event=event,
                expected_status=output.result.status,
                expected_intent_count=len(output.intents),
            )

        # All events should produce pending status and 2 intents
        assert len(log) == 5
        for entry in log.entries:
            assert entry.expected_status == "pending"
            assert entry.expected_intent_count == EXPECTED_REGISTRATION_INTENTS


@pytest.mark.unit
class TestEventSequenceIntegrity:
    """Tests for validating event sequence integrity."""

    def test_sequence_preserves_chronological_order(
        self,
        event_factory: EventFactory,
        event_sequence_log: EventSequenceLog,
    ) -> None:
        """Test that sequence preserves chronological event order.

        Verifies that:
        - Events are stored in append order
        - Timestamps are monotonically increasing
        """
        events = event_factory.create_event_sequence(count=5, time_between_events=60)

        for event in events:
            event_sequence_log.append(
                event=event,
                expected_status="pending",
                expected_intent_count=2,
            )

        # Verify order
        for i in range(len(event_sequence_log) - 1):
            current = event_sequence_log.entries[i]
            next_entry = event_sequence_log.entries[i + 1]
            assert current.event.timestamp < next_entry.event.timestamp

    def test_sequence_tracks_unique_events(
        self,
        event_factory: EventFactory,
        event_sequence_log: EventSequenceLog,
    ) -> None:
        """Test that each event in sequence has unique IDs.

        Verifies that:
        - Each event has a unique node_id
        - Each event has a unique correlation_id
        """
        events = event_factory.create_event_sequence(count=5)

        node_ids: set[UUID] = set()
        correlation_ids: set[UUID] = set()

        for event in events:
            event_sequence_log.append(
                event=event,
                expected_status="pending",
                expected_intent_count=2,
            )
            node_ids.add(event.node_id)
            correlation_ids.add(event.correlation_id)

        # All IDs should be unique
        assert len(node_ids) == 5
        assert len(correlation_ids) == 5

    def test_sequence_log_immutability_of_entries(
        self,
        event_factory: EventFactory,
        event_sequence_log: EventSequenceLog,
    ) -> None:
        """Test that EventSequenceEntry objects are immutable.

        Verifies that:
        - Entry dataclass is frozen
        - Attempting to modify raises an error
        """
        event = event_factory.create_event()
        event_sequence_log.append(
            event=event,
            expected_status="pending",
            expected_intent_count=2,
        )

        entry = event_sequence_log.entries[0]

        # Frozen dataclass should prevent modification
        with pytest.raises(Exception):  # FrozenInstanceError or similar
            entry.expected_status = "complete"  # type: ignore[misc]
