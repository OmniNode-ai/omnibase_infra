# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""State Reconstruction Validation Tests for OMN-955.

These tests validate the core ONEX principle that state can be fully
reconstructed from the event log alone. This is fundamental to event sourcing
and pure reducer patterns.

Key ONEX Principle:
    Given an event log [e1, e2, e3, ...], applying reduce(state, event) for each
    event starting from initial state MUST produce the final state exactly as
    it was when the events were first processed.

Architecture:
    - RegistrationReducer: Pure reducer that processes introspection events
    - ModelRegistrationState: Immutable state (frozen=True) with transition methods
    - reduce(): Pure function (state, event) -> ModelReducerOutput[state]
    - No side effects, no I/O, completely deterministic

Test Categories:
    - TestEventLogReconstruction: Rebuild state from event sequence
    - TestMultipleReconstructionScenarios: Various workflows (new, update, heartbeat)
    - TestEventLogConsistency: Verify replay produces identical state

Related Tickets:
    - OMN-955: State Reconstruction Validation Tests
    - OMN-914: Reducer Purity Enforcement Gates
    - OMN-889: Infrastructure MVP
"""

from __future__ import annotations

import pytest
from omnibase_core.models.primitives.model_semver import ModelSemVer

from omnibase_infra.models.registration import ModelNodeIntrospectionEvent
from omnibase_infra.nodes.reducers import RegistrationReducer
from omnibase_infra.nodes.reducers.models import ModelRegistrationState
from tests.helpers import (
    DeterministicClock,
    DeterministicIdGenerator,
    create_introspection_event,
)

# =============================================================================
# Module-Level Markers
# =============================================================================
# These markers enable selective test execution:
#   pytest -m "replay" - run only replay tests
#   pytest -m "not replay" - skip replay tests

pytestmark = [
    pytest.mark.replay,
]

__all__ = [
    "TestEventLogReconstruction",
    "TestMultipleReconstructionScenarios",
    "TestEventLogConsistency",
]


# =============================================================================
# Fixtures
# =============================================================================
# Note: Core fixtures (reducer, initial_state, id_generator, clock) are provided
# by tests/replay/conftest.py. Do not duplicate them here.


# =============================================================================
# Test: Event Log Reconstruction
# =============================================================================


@pytest.mark.unit
class TestEventLogReconstruction:
    """Tests that state can be rebuilt entirely from event log.

    ONEX Principle:
        The event log is the source of truth. Given the same initial state
        and the same sequence of events, the reducer MUST produce the same
        final state every time.
    """

    def test_state_from_single_event(
        self,
        reducer: RegistrationReducer,
        initial_state: ModelRegistrationState,
        id_generator: DeterministicIdGenerator,
        clock: DeterministicClock,
    ) -> None:
        """Verify state reconstruction from a single event.

        The simplest case: one event produces one state transition.
        """
        node_id = id_generator.next_uuid()
        correlation_id = id_generator.next_uuid()
        event = create_introspection_event(
            node_id=node_id,
            correlation_id=correlation_id,
            timestamp=clock.now(),
        )

        # Apply single event
        output = reducer.reduce(initial_state, event)

        # Verify state transition
        assert output.result.status == "pending"
        assert output.result.node_id == node_id
        assert output.result.last_processed_event_id == correlation_id
        assert len(output.intents) == 2  # Consul + PostgreSQL intents

    def test_state_from_event_sequence(
        self,
        reducer: RegistrationReducer,
        initial_state: ModelRegistrationState,
        id_generator: DeterministicIdGenerator,
        clock: DeterministicClock,
    ) -> None:
        """Verify state reconstruction from a sequence of events.

        Simulates processing an event log where the same reducer processes
        multiple events sequentially, accumulating state changes.
        """
        node_id = id_generator.next_uuid()

        # Create event log (sequence of events)
        events: list[ModelNodeIntrospectionEvent] = []
        for i in range(3):
            correlation_id = id_generator.next_uuid()
            clock.advance(60)  # 1 minute between events
            events.append(
                create_introspection_event(
                    node_id=node_id,
                    correlation_id=correlation_id,
                    timestamp=clock.now(),
                    node_version=f"1.0.{i}",
                )
            )

        # Apply event log sequentially
        state = initial_state
        for event in events:
            output = reducer.reduce(state, event)
            state = output.result

        # Final state should reflect all events processed
        # Since each event is unique, state accumulates
        assert state.status == "pending"
        assert state.node_id == node_id
        # Last processed event should be the final event
        assert state.last_processed_event_id == events[-1].correlation_id

    def test_reconstruction_matches_original_processing(
        self,
        reducer: RegistrationReducer,
        initial_state: ModelRegistrationState,
        id_generator: DeterministicIdGenerator,
        clock: DeterministicClock,
    ) -> None:
        """Verify reconstruction produces identical state to original processing.

        This is the core ONEX guarantee: replaying the event log produces
        the exact same state as the original processing.
        """
        node_id = id_generator.next_uuid()
        correlation_id = id_generator.next_uuid()
        event = create_introspection_event(
            node_id=node_id,
            correlation_id=correlation_id,
            timestamp=clock.now(),
        )

        # First processing (original)
        output_original = reducer.reduce(initial_state, event)
        state_original = output_original.result

        # Reconstruction (replay with fresh state)
        fresh_state = ModelRegistrationState()
        output_replay = reducer.reduce(fresh_state, event)
        state_replay = output_replay.result

        # States must be identical
        assert state_replay.status == state_original.status
        assert state_replay.node_id == state_original.node_id
        assert state_replay.consul_confirmed == state_original.consul_confirmed
        assert state_replay.postgres_confirmed == state_original.postgres_confirmed
        assert (
            state_replay.last_processed_event_id
            == state_original.last_processed_event_id
        )
        assert state_replay.failure_reason == state_original.failure_reason


# =============================================================================
# Test: Multiple Reconstruction Scenarios
# =============================================================================


@pytest.mark.unit
class TestMultipleReconstructionScenarios:
    """Tests for various reconstruction scenarios: new node, update, heartbeat."""

    def test_new_node_registration_scenario(
        self,
        reducer: RegistrationReducer,
        initial_state: ModelRegistrationState,
        id_generator: DeterministicIdGenerator,
        clock: DeterministicClock,
    ) -> None:
        """Test reconstruction of new node registration workflow.

        Scenario: A fresh node sends its first introspection event.
        Expected: State transitions from idle -> pending.
        """
        node_id = id_generator.next_uuid()
        correlation_id = id_generator.next_uuid()

        event = create_introspection_event(
            node_id=node_id,
            correlation_id=correlation_id,
            timestamp=clock.now(),
            node_type="effect",
        )

        # Apply event
        output = reducer.reduce(initial_state, event)

        # Verify new node registration state
        assert output.result.status == "pending"
        assert output.result.node_id == node_id
        assert output.result.consul_confirmed is False
        assert output.result.postgres_confirmed is False
        assert len(output.intents) == 2

        # Verify intents for both backends (extension pattern)
        intent_types = {
            intent.payload.intent_type
            for intent in output.intents
            if intent.intent_type == "extension"
        }
        assert "consul.register" in intent_types
        assert "postgres.upsert_registration" in intent_types

    def test_node_update_scenario(
        self,
        reducer: RegistrationReducer,
        id_generator: DeterministicIdGenerator,
        clock: DeterministicClock,
    ) -> None:
        """Test reconstruction of node update workflow.

        Scenario: An existing node sends an updated introspection event.
        Expected: State reflects the update (different correlation_id).
        """
        node_id = id_generator.next_uuid()

        # First event - initial registration
        event1 = create_introspection_event(
            node_id=node_id,
            correlation_id=id_generator.next_uuid(),
            timestamp=clock.now(),
            node_version=ModelSemVer.parse("1.0.0"),
        )

        # Process first event
        initial_state = ModelRegistrationState()
        output1 = reducer.reduce(initial_state, event1)
        state_after_first = output1.result

        assert state_after_first.status == "pending"

        # Second event - update with new version
        clock.advance(60)
        event2 = create_introspection_event(
            node_id=node_id,
            correlation_id=id_generator.next_uuid(),
            timestamp=clock.now(),
            node_version=ModelSemVer.parse("1.1.0"),  # Version update
        )

        # Process second event
        output2 = reducer.reduce(state_after_first, event2)
        state_after_second = output2.result

        # State should reflect update
        assert state_after_second.status == "pending"
        assert state_after_second.node_id == node_id
        assert state_after_second.last_processed_event_id == event2.correlation_id
        assert state_after_second.last_processed_event_id != event1.correlation_id

    def test_heartbeat_scenario_idempotency(
        self,
        reducer: RegistrationReducer,
        id_generator: DeterministicIdGenerator,
        clock: DeterministicClock,
    ) -> None:
        """Test heartbeat scenario where same event is replayed.

        Scenario: Same event (same correlation_id) is processed twice.
        Expected: Second processing returns unchanged state with no intents.
        """
        node_id = id_generator.next_uuid()
        correlation_id = id_generator.next_uuid()

        event = create_introspection_event(
            node_id=node_id,
            correlation_id=correlation_id,
            timestamp=clock.now(),
        )

        # First processing
        initial_state = ModelRegistrationState()
        output1 = reducer.reduce(initial_state, event)
        state_after_first = output1.result

        assert len(output1.intents) == 2  # Intents emitted

        # Second processing (duplicate/heartbeat)
        output2 = reducer.reduce(state_after_first, event)
        state_after_second = output2.result

        # State unchanged, no new intents (idempotency)
        assert state_after_second.status == state_after_first.status
        assert state_after_second.node_id == state_after_first.node_id
        assert len(output2.intents) == 0  # No duplicate intents

    def test_failed_state_reset_scenario(
        self,
        reducer: RegistrationReducer,
        id_generator: DeterministicIdGenerator,
        clock: DeterministicClock,
    ) -> None:
        """Test reconstruction of failed state with reset workflow.

        Scenario: Registration failed, then reset event received.
        Expected: State transitions from failed -> idle.
        """
        node_id = id_generator.next_uuid()

        # Start with a failed state (simulating prior failure)
        failed_state = ModelRegistrationState(
            status="failed",
            node_id=node_id,
            failure_reason="consul_failed",
            last_processed_event_id=id_generator.next_uuid(),
        )

        # Apply reset
        reset_event_id = id_generator.next_uuid()
        output = reducer.reduce_reset(failed_state, reset_event_id)

        # Verify reset
        assert output.result.status == "idle"
        assert output.result.node_id is None
        assert output.result.failure_reason is None
        assert output.result.last_processed_event_id == reset_event_id

    def test_multiple_node_types(
        self,
        reducer: RegistrationReducer,
        initial_state: ModelRegistrationState,
        id_generator: DeterministicIdGenerator,
        clock: DeterministicClock,
    ) -> None:
        """Test reconstruction works for all ONEX node types.

        Each node type (effect, compute, reducer, orchestrator) should
        be processed correctly with appropriate intents.
        """
        node_types = ["effect", "compute", "reducer", "orchestrator"]

        for node_type in node_types:
            node_id = id_generator.next_uuid()
            correlation_id = id_generator.next_uuid()
            clock.advance(60)

            event = create_introspection_event(
                node_id=node_id,
                correlation_id=correlation_id,
                timestamp=clock.now(),
                node_type=node_type,
            )

            # Fresh state for each node type
            fresh_state = ModelRegistrationState()
            output = reducer.reduce(fresh_state, event)

            # Verify processing
            assert output.result.status == "pending", (
                f"Node type {node_type} failed to transition to pending"
            )
            assert output.result.node_id == node_id
            assert len(output.intents) == 2

            # Verify intent targets include node type
            for intent in output.intents:
                if (
                    intent.intent_type == "extension"
                    and intent.payload.intent_type == "consul.register"
                ):
                    assert node_type in intent.target


# =============================================================================
# Test: Event Log Consistency
# =============================================================================


@pytest.mark.unit
class TestEventLogConsistency:
    """Tests that event log replay is always consistent."""

    def test_multiple_replays_produce_identical_state(
        self,
        initial_state: ModelRegistrationState,
        id_generator: DeterministicIdGenerator,
        clock: DeterministicClock,
    ) -> None:
        """Verify multiple replays of the same event log produce identical state.

        This is the ultimate consistency guarantee: any number of replays
        with the same inputs produces the same outputs.
        """
        node_id = id_generator.next_uuid()
        correlation_id = id_generator.next_uuid()

        event = create_introspection_event(
            node_id=node_id,
            correlation_id=correlation_id,
            timestamp=clock.now(),
        )

        # Perform multiple replays with fresh reducer instances
        results: list[ModelRegistrationState] = []
        for _ in range(5):
            reducer = RegistrationReducer()
            fresh_state = ModelRegistrationState()
            output = reducer.reduce(fresh_state, event)
            results.append(output.result)

        # All results must be identical
        first_result = results[0]
        for i, result in enumerate(results[1:], start=2):
            assert result.status == first_result.status, f"Replay {i} status mismatch"
            assert result.node_id == first_result.node_id, (
                f"Replay {i} node_id mismatch"
            )
            assert result.consul_confirmed == first_result.consul_confirmed, (
                f"Replay {i} consul_confirmed mismatch"
            )
            assert result.postgres_confirmed == first_result.postgres_confirmed, (
                f"Replay {i} postgres_confirmed mismatch"
            )
            assert (
                result.last_processed_event_id == first_result.last_processed_event_id
            ), f"Replay {i} last_processed_event_id mismatch"

    def test_event_log_order_matters(
        self,
        id_generator: DeterministicIdGenerator,
        clock: DeterministicClock,
    ) -> None:
        """Verify that event log order affects final state.

        The event log is an ordered sequence. Different orderings of
        independent events MAY produce different intermediate states
        but MUST be deterministic for the same ordering.
        """
        node_id = id_generator.next_uuid()

        # Create two distinct events for the same node
        event1 = create_introspection_event(
            node_id=node_id,
            correlation_id=id_generator.next_uuid(),
            timestamp=clock.now(),
            node_version=ModelSemVer.parse("1.0.0"),
        )

        clock.advance(60)
        event2 = create_introspection_event(
            node_id=node_id,
            correlation_id=id_generator.next_uuid(),
            timestamp=clock.now(),
            node_version=ModelSemVer.parse("2.0.0"),
        )

        # Process in order: event1, then event2
        reducer = RegistrationReducer()
        state = ModelRegistrationState()
        state = reducer.reduce(state, event1).result
        state_order1 = reducer.reduce(state, event2).result

        # Process in reverse order: event2, then event1
        reducer2 = RegistrationReducer()
        state = ModelRegistrationState()
        state = reducer2.reduce(state, event2).result
        state_order2 = reducer2.reduce(state, event1).result

        # Final state should reflect the LAST event processed
        # (last_processed_event_id differs based on order)
        assert state_order1.last_processed_event_id == event2.correlation_id
        assert state_order2.last_processed_event_id == event1.correlation_id

    def test_state_reconstruction_after_crash(
        self,
        id_generator: DeterministicIdGenerator,
        clock: DeterministicClock,
    ) -> None:
        """Simulate state reconstruction after a crash.

        Scenario: System crashed after processing some events.
        Recovery: Replay the same events from the log.
        Expected: Final state matches pre-crash state.
        """
        node_id = id_generator.next_uuid()

        # Create event log
        events = [
            create_introspection_event(
                node_id=node_id,
                correlation_id=id_generator.next_uuid(),
                timestamp=clock.now(),
                node_version=f"1.0.{i}",
            )
            for i in range(3)
        ]

        # "Original" processing before crash
        reducer = RegistrationReducer()
        state = ModelRegistrationState()
        for event in events:
            clock.advance(60)
            output = reducer.reduce(state, event)
            state = output.result

        pre_crash_state = state

        # Simulate crash: lose all in-memory state
        # Recovery: replay from event log with fresh reducer
        reducer_recovered = RegistrationReducer()
        recovered_state = ModelRegistrationState()
        for event in events:
            output = reducer_recovered.reduce(recovered_state, event)
            recovered_state = output.result

        # Recovered state must match pre-crash state
        assert recovered_state.status == pre_crash_state.status
        assert recovered_state.node_id == pre_crash_state.node_id
        assert (
            recovered_state.last_processed_event_id
            == pre_crash_state.last_processed_event_id
        )
