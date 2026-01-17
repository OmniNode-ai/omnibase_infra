# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Reducer Replay Determinism Tests for OMN-955.

This module tests that replaying events through the RegistrationReducer
produces identical state. This is the core guarantee of pure reducers:
given the same inputs, they MUST produce the same outputs.

Determinism Guarantees:
    1. Same state + same event = same output (pure function property)
    2. Independent of time of execution
    3. Independent of reducer instance
    4. Independent of system state

Test Coverage:
    - Single event determinism
    - Multi-event sequence determinism
    - Cross-instance determinism
    - Replay after serialization
    - All node types produce deterministic results

Architecture:
    The RegistrationReducer is a pure function:
    - reduce(state, event) -> ModelReducerOutput
    - No internal state
    - No I/O operations
    - Deterministic: same inputs always produce same outputs

Related:
    - test_reducer_purity.py: Structural and behavioral purity tests
    - RegistrationReducer: Pure reducer under test
    - OMN-955: Event Replay Verification ticket
"""

from __future__ import annotations

import copy
import json
from datetime import datetime
from typing import TYPE_CHECKING
from uuid import UUID

import pytest

from omnibase_core.models.primitives.model_semver import ModelSemVer
from omnibase_infra.models.registration import (
    ModelNodeCapabilities,
    ModelNodeIntrospectionEvent,
    ModelNodeMetadata,
)
from omnibase_infra.nodes.reducers import RegistrationReducer
from omnibase_infra.nodes.reducers.models import ModelRegistrationState
from tests.helpers.replay_utils import compare_outputs

if TYPE_CHECKING:
    from omnibase_core.nodes import ModelReducerOutput
    from tests.replay.conftest import EventFactory, NodeType


# =============================================================================
# Module-Level Markers
# =============================================================================
# These markers enable selective test execution:
#   pytest -m "replay" - run only replay tests
#   pytest -m "not replay" - skip replay tests

pytestmark = [
    pytest.mark.replay,
]

# =============================================================================
# Constants
# =============================================================================

EXPECTED_REGISTRATION_INTENTS = 2


# =============================================================================
# Single Event Determinism Tests
# =============================================================================


@pytest.mark.unit
class TestSingleEventDeterminism:
    """Tests for single event replay determinism."""

    def test_same_input_produces_same_output(
        self,
        reducer: RegistrationReducer,
        fixed_node_id: UUID,
        fixed_correlation_id: UUID,
        fixed_timestamp: datetime,
    ) -> None:
        """Test that identical inputs produce identical outputs.

        This is the core pure function guarantee.
        """
        state = ModelRegistrationState()
        event = ModelNodeIntrospectionEvent(
            node_id=fixed_node_id,
            node_type="effect",
            node_version=ModelSemVer.parse("1.0.0"),
            correlation_id=fixed_correlation_id,
            timestamp=fixed_timestamp,
            endpoints={"health": "http://localhost:8080/health"},
            declared_capabilities=ModelNodeCapabilities(),
            metadata=ModelNodeMetadata(),
        )

        # Run reducer multiple times with identical inputs
        output1 = reducer.reduce(state, event)

        # Reset state for second run
        state2 = ModelRegistrationState()
        output2 = reducer.reduce(state2, event)

        # Outputs must be identical (excluding operation_id which is always unique)
        are_equal, differences = compare_outputs(output1, output2)
        assert are_equal, f"Outputs differ: {differences}"

    def test_determinism_across_multiple_runs(
        self,
        fixed_node_id: UUID,
        fixed_correlation_id: UUID,
        fixed_timestamp: datetime,
    ) -> None:
        """Test determinism across many consecutive runs.

        Validates that the reducer is consistently deterministic,
        not just in isolated cases.
        """
        event = ModelNodeIntrospectionEvent(
            node_id=fixed_node_id,
            node_type="compute",
            node_version=ModelSemVer.parse("2.0.0"),
            correlation_id=fixed_correlation_id,
            timestamp=fixed_timestamp,
            endpoints={},
            declared_capabilities=ModelNodeCapabilities(),
            metadata=ModelNodeMetadata(),
        )

        # Run 10 times with fresh reducer and state each time
        outputs: list[ModelReducerOutput] = []
        for _ in range(10):
            reducer = RegistrationReducer()
            fresh_state = ModelRegistrationState()
            output = reducer.reduce(fresh_state, event)
            outputs.append(output)

        # All outputs must match the first one
        for i, output in enumerate(outputs[1:], start=2):
            are_equal, differences = compare_outputs(outputs[0], output)
            assert are_equal, f"Run {i} differs from run 1: {differences}"

    def test_determinism_for_all_node_types(
        self,
        reducer: RegistrationReducer,
        fixed_node_id: UUID,
        fixed_correlation_id: UUID,
        fixed_timestamp: datetime,
    ) -> None:
        """Test that all node types produce deterministic results.

        Verifies that the reducer's determinism holds regardless
        of the node type being processed.
        """
        node_types: list[str] = ["effect", "compute", "reducer", "orchestrator"]

        for node_type in node_types:
            event = ModelNodeIntrospectionEvent(
                node_id=fixed_node_id,
                node_type=node_type,  # type: ignore[arg-type]
                node_version=ModelSemVer.parse("1.0.0"),
                correlation_id=fixed_correlation_id,
                timestamp=fixed_timestamp,
                endpoints={"health": "http://localhost:8080/health"},
                declared_capabilities=ModelNodeCapabilities(),
                metadata=ModelNodeMetadata(),
            )

            state1 = ModelRegistrationState()
            state2 = ModelRegistrationState()

            output1 = reducer.reduce(state1, event)
            output2 = reducer.reduce(state2, event)

            are_equal, differences = compare_outputs(output1, output2)
            assert are_equal, (
                f"Node type {node_type} is not deterministic: {differences}"
            )


# =============================================================================
# Multi-Event Sequence Determinism Tests
# =============================================================================


@pytest.mark.unit
class TestMultiEventSequenceDeterminism:
    """Tests for multi-event sequence replay determinism."""

    def test_sequence_replay_produces_identical_state(
        self,
        reducer: RegistrationReducer,
        event_factory: EventFactory,
    ) -> None:
        """Test that replaying a sequence produces identical results.

        This test captures a sequence and replays it, verifying
        that both runs produce identical intermediate and final states.
        """
        events = event_factory.create_event_sequence(count=5)

        # First replay
        states1: list[ModelRegistrationState] = []
        intents1: list[int] = []

        for event in events:
            state = ModelRegistrationState()
            output = reducer.reduce(state, event)
            states1.append(output.result)
            intents1.append(len(output.intents))

        # Second replay (reset factory for identical events)
        event_factory.reset()
        events2 = event_factory.create_event_sequence(count=5)

        states2: list[ModelRegistrationState] = []
        intents2: list[int] = []

        for event in events2:
            state = ModelRegistrationState()
            output = reducer.reduce(state, event)
            states2.append(output.result)
            intents2.append(len(output.intents))

        # Verify all states match
        for i, (s1, s2) in enumerate(zip(states1, states2, strict=True)):
            assert s1.status == s2.status, f"State {i} status mismatch"
            assert s1.node_id == s2.node_id, f"State {i} node_id mismatch"

        # Verify all intent counts match
        assert intents1 == intents2, "Intent counts mismatch"

    def test_sequence_order_independence_for_isolated_events(
        self,
        reducer: RegistrationReducer,
        event_factory: EventFactory,
    ) -> None:
        """Test that isolated events produce same results regardless of prior events.

        Since each event starts from a fresh initial state, the order
        of other events should not affect the result.
        """
        events = event_factory.create_event_sequence(count=3)

        # Process in forward order
        forward_outputs: list[ModelReducerOutput] = []
        for event in events:
            state = ModelRegistrationState()
            output = reducer.reduce(state, event)
            forward_outputs.append(output)

        # Process in reverse order (but each with fresh state)
        reverse_outputs: list[ModelReducerOutput] = []
        for event in reversed(events):
            state = ModelRegistrationState()
            output = reducer.reduce(state, event)
            reverse_outputs.append(output)
        reverse_outputs.reverse()  # Restore original order for comparison

        # Each event should produce identical output regardless of processing order
        for i, (fwd, rev) in enumerate(
            zip(forward_outputs, reverse_outputs, strict=True)
        ):
            are_equal, differences = compare_outputs(fwd, rev)
            assert are_equal, (
                f"Event {i} output differs by processing order: {differences}"
            )

    def test_mixed_node_type_sequence_determinism(
        self,
        reducer: RegistrationReducer,
        multi_node_type_sequence: list[tuple[NodeType, ModelNodeIntrospectionEvent]],
    ) -> None:
        """Test determinism with mixed node types in sequence.

        Verifies that a sequence with different node types is
        deterministic when replayed.
        """
        # First pass
        outputs1: list[tuple[str, str, int]] = []  # (node_type, status, intent_count)
        for node_type, event in multi_node_type_sequence:
            state = ModelRegistrationState()
            output = reducer.reduce(state, event)
            outputs1.append((node_type, output.result.status, len(output.intents)))

        # Second pass with same events
        outputs2: list[tuple[str, str, int]] = []
        for node_type, event in multi_node_type_sequence:
            state = ModelRegistrationState()
            output = reducer.reduce(state, event)
            outputs2.append((node_type, output.result.status, len(output.intents)))

        assert outputs1 == outputs2, "Mixed node type sequence not deterministic"


# =============================================================================
# Cross-Instance Determinism Tests
# =============================================================================


@pytest.mark.unit
class TestCrossInstanceDeterminism:
    """Tests for determinism across different reducer instances."""

    def test_different_instances_same_output(
        self,
        fixed_node_id: UUID,
        fixed_correlation_id: UUID,
        fixed_timestamp: datetime,
    ) -> None:
        """Test that different reducer instances produce same output.

        Verifies that reducers have no hidden instance state.
        """
        event = ModelNodeIntrospectionEvent(
            node_id=fixed_node_id,
            node_type="effect",
            node_version=ModelSemVer.parse("1.0.0"),
            correlation_id=fixed_correlation_id,
            timestamp=fixed_timestamp,
            endpoints={"health": "http://localhost:8080/health"},
            declared_capabilities=ModelNodeCapabilities(),
            metadata=ModelNodeMetadata(),
        )

        # Create 5 separate reducer instances
        reducers = [RegistrationReducer() for _ in range(5)]

        outputs: list[ModelReducerOutput] = []
        for reducer in reducers:
            state = ModelRegistrationState()
            output = reducer.reduce(state, event)
            outputs.append(output)

        # All outputs must be identical
        for i, output in enumerate(outputs[1:], start=2):
            are_equal, differences = compare_outputs(outputs[0], output)
            assert are_equal, f"Instance {i} differs from instance 1: {differences}"

    def test_reducer_reuse_determinism(
        self,
        reducer: RegistrationReducer,
        event_factory: EventFactory,
    ) -> None:
        """Test that reusing a reducer instance is deterministic.

        A single reducer instance should produce the same output
        whether it's the first or hundredth time it processes an event.
        """
        # Create multiple events
        events = event_factory.create_event_sequence(count=10)

        # Process all events once with the same reducer
        first_pass: list[ModelReducerOutput] = []
        for event in events:
            state = ModelRegistrationState()
            output = reducer.reduce(state, event)
            first_pass.append(output)

        # Process all events again with the same reducer
        second_pass: list[ModelReducerOutput] = []
        for event in events:
            state = ModelRegistrationState()
            output = reducer.reduce(state, event)
            second_pass.append(output)

        # All outputs should match
        for i, (out1, out2) in enumerate(zip(first_pass, second_pass, strict=True)):
            are_equal, differences = compare_outputs(out1, out2)
            assert are_equal, f"Event {i} differs on reuse: {differences}"


# =============================================================================
# Replay After Serialization Tests
# =============================================================================


@pytest.mark.unit
class TestReplayAfterSerialization:
    """Tests for determinism after event serialization/deserialization."""

    def test_serialization_round_trip_determinism(
        self,
        reducer: RegistrationReducer,
        fixed_node_id: UUID,
        fixed_correlation_id: UUID,
        fixed_timestamp: datetime,
    ) -> None:
        """Test that serializing and deserializing an event preserves determinism.

        Events may be stored and replayed later. The serialization
        process must not affect the reducer's output.
        """
        original_event = ModelNodeIntrospectionEvent(
            node_id=fixed_node_id,
            node_type="effect",
            node_version=ModelSemVer.parse("1.0.0"),
            correlation_id=fixed_correlation_id,
            timestamp=fixed_timestamp,
            endpoints={"health": "http://localhost:8080/health"},
            declared_capabilities=ModelNodeCapabilities(postgres=True, read=True),
            metadata=ModelNodeMetadata(environment="test"),
        )

        # Process original event
        state1 = ModelRegistrationState()
        output1 = reducer.reduce(state1, original_event)

        # Serialize and deserialize event
        event_dict = original_event.model_dump(mode="json")
        deserialized_event = ModelNodeIntrospectionEvent.model_validate(event_dict)

        # Process deserialized event
        state2 = ModelRegistrationState()
        output2 = reducer.reduce(state2, deserialized_event)

        are_equal, differences = compare_outputs(output1, output2)
        assert are_equal, f"Serialization affected determinism: {differences}"

    def test_json_serialization_determinism(
        self,
        reducer: RegistrationReducer,
        event_factory: EventFactory,
    ) -> None:
        """Test determinism with JSON serialization.

        Validates that JSON round-trip doesn't affect reducer output.
        """
        events = event_factory.create_event_sequence(count=3)

        # Process original events
        original_outputs: list[ModelReducerOutput] = []
        for event in events:
            state = ModelRegistrationState()
            output = reducer.reduce(state, event)
            original_outputs.append(output)

        # Serialize all events to JSON
        json_events = [json.loads(event.model_dump_json()) for event in events]

        # Deserialize and process
        deserialized_outputs: list[ModelReducerOutput] = []
        for event_dict in json_events:
            event = ModelNodeIntrospectionEvent.model_validate(event_dict)
            state = ModelRegistrationState()
            output = reducer.reduce(state, event)
            deserialized_outputs.append(output)

        # Compare all outputs
        for i, (orig, deser) in enumerate(
            zip(original_outputs, deserialized_outputs, strict=True)
        ):
            are_equal, differences = compare_outputs(orig, deser)
            assert are_equal, (
                f"Event {i} JSON round-trip not deterministic: {differences}"
            )


# =============================================================================
# State Immutability in Replay Tests
# =============================================================================


@pytest.mark.unit
class TestStateImmutabilityInReplay:
    """Tests for state immutability during replay scenarios."""

    def test_input_state_unchanged_after_replay(
        self,
        reducer: RegistrationReducer,
        event_factory: EventFactory,
    ) -> None:
        """Test that replaying events doesn't mutate input state.

        The reducer must return new state objects, not mutate inputs.
        """
        events = event_factory.create_event_sequence(count=3)

        for event in events:
            initial_state = ModelRegistrationState()
            state_copy = copy.deepcopy(initial_state)

            _ = reducer.reduce(initial_state, event)

            # Initial state should be unchanged
            assert initial_state.status == state_copy.status
            assert initial_state.node_id == state_copy.node_id
            assert initial_state.consul_confirmed == state_copy.consul_confirmed
            assert initial_state.postgres_confirmed == state_copy.postgres_confirmed

    def test_replay_preserves_state_snapshot(
        self,
        reducer: RegistrationReducer,
        event_factory: EventFactory,
    ) -> None:
        """Test that state snapshots are preserved during replay.

        Each reduction should create an independent state snapshot.
        """
        events = event_factory.create_event_sequence(count=5)

        # Capture all intermediate states
        states: list[ModelRegistrationState] = []
        for event in events:
            state = ModelRegistrationState()
            output = reducer.reduce(state, event)
            states.append(output.result)

        # All states should be independent and unchanged
        for i, state in enumerate(states):
            assert state.status == "pending", f"State {i} was modified"
            assert state.node_id is not None, f"State {i} lost node_id"
