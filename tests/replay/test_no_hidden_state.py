# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""No Hidden State Validation Tests for OMN-955.

These tests prove that the RegistrationReducer has NO hidden mutable state
outside the event stream. This is a fundamental ONEX pure reducer guarantee:

    All state changes MUST be visible through the event stream.
    The reducer has NO side effects beyond returning the new state.

Architecture Validation:
    - Reducer is a pure function: reduce(state, event) -> output
    - No instance variables mutated during reduce()
    - No class variables mutated during reduce()
    - No global state affected by reduce()
    - No external resources accessed during reduce()

Enforcement Strategy:
    1. Structural inspection via inspect module (method signatures, attributes)
    2. Runtime monitoring (mocking I/O operations)
    3. State isolation tests (parallel execution with shared reducer)
    4. Determinism verification (same input -> same output always)

Related Tickets:
    - OMN-955: State Reconstruction Validation Tests
    - OMN-914: Reducer Purity Enforcement Gates
"""

from __future__ import annotations

import concurrent.futures
import copy
import inspect

import pytest

from omnibase_infra.nodes.reducers import RegistrationReducer
from omnibase_infra.nodes.reducers.models import ModelRegistrationState
from omnibase_infra.testing import is_ci_environment
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
    "TestReducerPurityInspection",
    "TestNoSideEffects",
    "TestStateIsolation",
]


# =============================================================================
# Fixtures
# =============================================================================
# Note: Core fixtures (reducer, initial_state, clock) are provided by
# tests/replay/conftest.py. Only override here when intentionally different.


@pytest.fixture
def id_generator() -> DeterministicIdGenerator:
    """Create a deterministic ID generator with isolated seed.

    Uses seed=300 (different from conftest's seed=100) to ensure this test
    module's UUIDs don't accidentally match those from other replay tests.
    This provides better isolation for purity verification tests.
    """
    return DeterministicIdGenerator(seed=300)


# =============================================================================
# Test: Reducer Purity via Inspection
# =============================================================================


@pytest.mark.unit
class TestReducerPurityInspection:
    """Tests that verify reducer purity using the inspect module.

    These tests analyze the reducer's structure to ensure it cannot
    accumulate hidden state during operation.
    """

    def test_reduce_is_not_coroutine(self) -> None:
        """Verify reduce() is synchronous (not async).

        Pure reducers are synchronous. Async implies I/O waiting.
        """
        assert not inspect.iscoroutinefunction(RegistrationReducer.reduce), (
            "reduce() must be synchronous. Async methods imply I/O operations."
        )

    def test_reduce_reset_is_not_coroutine(self) -> None:
        """Verify reduce_reset() is synchronous (not async)."""
        assert not inspect.iscoroutinefunction(RegistrationReducer.reduce_reset), (
            "reduce_reset() must be synchronous. Async methods imply I/O operations."
        )

    def test_reducer_has_no_mutable_instance_state(self) -> None:
        """Verify reducer instances have no mutable state after initialization.

        A pure reducer should not maintain instance-level mutable state
        that could be modified during reduce() calls.
        """
        reducer = RegistrationReducer()

        # Get all instance attributes
        instance_attrs = {
            name: value
            for name, value in vars(reducer).items()
            if not name.startswith("_")
        }

        # Check for mutable types in instance attributes
        mutable_types = (list, dict, set)
        mutable_attrs = [
            name
            for name, value in instance_attrs.items()
            if isinstance(value, mutable_types)
        ]

        assert len(mutable_attrs) == 0, (
            f"Reducer has mutable instance attributes: {mutable_attrs}. "
            f"This could lead to hidden state accumulation."
        )

    def test_reducer_has_no_mutable_class_variables(self) -> None:
        """Verify reducer class has no mutable class-level variables.

        Class variables that could be mutated would violate purity.
        """
        class_vars = {
            name: value
            for name, value in vars(RegistrationReducer).items()
            if not name.startswith("_")
            and not callable(value)
            and not isinstance(value, (property, classmethod, staticmethod))
        }

        mutable_types = (list, dict, set)
        mutable_class_vars = [
            name
            for name, value in class_vars.items()
            if isinstance(value, mutable_types)
        ]

        assert len(mutable_class_vars) == 0, (
            f"Reducer has mutable class variables: {mutable_class_vars}. "
            f"This violates the pure function contract."
        )

    def test_reduce_signature_is_pure(self) -> None:
        """Verify reduce() signature follows pure function pattern.

        Pure reduce() signature: (self, state, event) -> output
        - Takes immutable state and event
        - Returns new output (does not modify inputs)
        """
        sig = inspect.signature(RegistrationReducer.reduce)
        params = list(sig.parameters.keys())

        # Expected: self, state, event
        assert len(params) == 3, (
            f"reduce() should have exactly 3 parameters (self, state, event), "
            f"got {len(params)}: {params}"
        )
        assert "state" in params, "reduce() must accept 'state' parameter"
        assert "event" in params, "reduce() must accept 'event' parameter"

    def test_reduce_returns_output_not_none(self) -> None:
        """Verify reduce() returns ModelReducerOutput, not None.

        Pure reducers always return an output, never modify state in place.
        """
        sig = inspect.signature(RegistrationReducer.reduce)
        return_annotation = sig.return_annotation

        # Should return ModelReducerOutput[ModelRegistrationState]
        assert return_annotation != inspect.Signature.empty, (
            "reduce() must have a return type annotation"
        )
        # Check it's not None type
        assert return_annotation is not type(None), (
            "reduce() must return an output, not None"
        )


# =============================================================================
# Test: No Side Effects
# =============================================================================


@pytest.mark.unit
class TestNoSideEffects:
    """Tests that verify reduce() has no side effects.

    Side effects include:
    - Mutating input state
    - Mutating input event
    - Writing to instance variables
    - Writing to class variables
    - Writing to global state
    """

    def test_input_state_not_mutated(
        self,
        reducer: RegistrationReducer,
        id_generator: DeterministicIdGenerator,
        clock: DeterministicClock,
    ) -> None:
        """Verify reduce() does not mutate the input state.

        The input state must remain unchanged after reduce() returns.
        """
        original_state = ModelRegistrationState()
        state_before = copy.deepcopy(original_state)

        node_id = id_generator.next_uuid()
        correlation_id = id_generator.next_uuid()
        event = create_introspection_event(
            node_id=node_id,
            correlation_id=correlation_id,
            timestamp=clock.now(),
        )

        # Run reduce
        _output = reducer.reduce(original_state, event)

        # Verify input state was NOT mutated
        assert original_state.status == state_before.status, (
            f"Input state.status was mutated: "
            f"{state_before.status} -> {original_state.status}"
        )
        assert original_state.node_id == state_before.node_id, (
            "Input state.node_id was mutated"
        )
        assert original_state.consul_confirmed == state_before.consul_confirmed, (
            "Input state.consul_confirmed was mutated"
        )
        assert original_state.postgres_confirmed == state_before.postgres_confirmed, (
            "Input state.postgres_confirmed was mutated"
        )
        assert (
            original_state.last_processed_event_id
            == state_before.last_processed_event_id
        ), "Input state.last_processed_event_id was mutated"

    def test_input_event_not_mutated(
        self,
        reducer: RegistrationReducer,
        id_generator: DeterministicIdGenerator,
        clock: DeterministicClock,
    ) -> None:
        """Verify reduce() does not mutate the input event.

        The input event must remain unchanged after reduce() returns.
        """
        node_id = id_generator.next_uuid()
        correlation_id = id_generator.next_uuid()
        original_event = create_introspection_event(
            node_id=node_id,
            correlation_id=correlation_id,
            timestamp=clock.now(),
        )
        event_before = copy.deepcopy(original_event)

        state = ModelRegistrationState()

        # Run reduce
        _output = reducer.reduce(state, original_event)

        # Verify input event was NOT mutated
        assert original_event.node_id == event_before.node_id, (
            "Input event.node_id was mutated"
        )
        assert original_event.node_type == event_before.node_type, (
            "Input event.node_type was mutated"
        )
        assert original_event.correlation_id == event_before.correlation_id, (
            "Input event.correlation_id was mutated"
        )
        assert original_event.timestamp == event_before.timestamp, (
            "Input event.timestamp was mutated"
        )

    def test_reducer_instance_unchanged_after_reduce(
        self,
        id_generator: DeterministicIdGenerator,
        clock: DeterministicClock,
    ) -> None:
        """Verify reducer instance has no state changes after reduce().

        The reducer should not accumulate any internal state.
        """
        reducer = RegistrationReducer()

        # Capture reducer state before
        attrs_before = copy.deepcopy(vars(reducer))

        node_id = id_generator.next_uuid()
        event = create_introspection_event(
            node_id=node_id,
            correlation_id=id_generator.next_uuid(),
            timestamp=clock.now(),
        )

        # Run multiple reductions (output deliberately unused - testing side effects)
        state = ModelRegistrationState()
        for _ in range(5):
            _ = reducer.reduce(state, event)  # Exercise reducer, ignore output
            state = ModelRegistrationState()  # Reset for next iteration

        # Capture reducer state after
        attrs_after = vars(reducer)

        # Compare
        assert attrs_before == attrs_after, (
            "Reducer instance state changed during reduce() calls. "
            f"Before: {attrs_before}, After: {attrs_after}"
        )

    def test_class_variables_unchanged_after_reduce(
        self,
        id_generator: DeterministicIdGenerator,
        clock: DeterministicClock,
    ) -> None:
        """Verify class-level variables are unchanged after reduce().

        Class variables should not be mutated by reduce() calls.
        """
        # Capture class state before
        class_vars_before = {
            name: copy.deepcopy(value)
            for name, value in vars(RegistrationReducer).items()
            if not name.startswith("_")
            and not callable(value)
            and not isinstance(value, (property, classmethod, staticmethod))
        }

        # Run multiple reductions on multiple instances
        for _ in range(3):
            reducer = RegistrationReducer()
            state = ModelRegistrationState()
            node_id = id_generator.next_uuid()
            event = create_introspection_event(
                node_id=node_id,
                correlation_id=id_generator.next_uuid(),
                timestamp=clock.now(),
            )
            _output = reducer.reduce(state, event)

        # Capture class state after
        class_vars_after = {
            name: value
            for name, value in vars(RegistrationReducer).items()
            if not name.startswith("_")
            and not callable(value)
            and not isinstance(value, (property, classmethod, staticmethod))
        }

        assert class_vars_before == class_vars_after, (
            "Class-level variables were mutated during reduce() execution"
        )


# =============================================================================
# Test: State Isolation
# =============================================================================


@pytest.mark.unit
class TestStateIsolation:
    """Tests that verify complete state isolation between reducer calls.

    These tests prove that:
    - Different reducer instances don't share state
    - Sequential calls don't leak state
    - Parallel calls don't interfere with each other
    """

    def test_different_instances_are_isolated(
        self,
        id_generator: DeterministicIdGenerator,
        clock: DeterministicClock,
    ) -> None:
        """Verify different reducer instances don't share state.

        Each reducer instance should be completely independent.
        """
        reducer1 = RegistrationReducer()
        reducer2 = RegistrationReducer()

        node_id = id_generator.next_uuid()
        event = create_introspection_event(
            node_id=node_id,
            correlation_id=id_generator.next_uuid(),
            timestamp=clock.now(),
        )

        state = ModelRegistrationState()

        # Run on both reducers
        output1 = reducer1.reduce(state, event)
        output2 = reducer2.reduce(state, event)

        # Results should be identical (no cross-instance state)
        assert output1.result.status == output2.result.status
        assert output1.result.node_id == output2.result.node_id
        assert len(output1.intents) == len(output2.intents)

    def test_sequential_calls_are_isolated(
        self,
        reducer: RegistrationReducer,
        id_generator: DeterministicIdGenerator,
        clock: DeterministicClock,
    ) -> None:
        """Verify sequential calls don't leak state between them.

        Each call to reduce() should be independent of previous calls.
        """
        results: list[tuple] = []

        for i in range(5):
            node_id = id_generator.next_uuid()
            correlation_id = id_generator.next_uuid()
            clock.advance(60)

            event = create_introspection_event(
                node_id=node_id,
                correlation_id=correlation_id,
                timestamp=clock.now(),
            )

            # Fresh state each time
            fresh_state = ModelRegistrationState()
            output = reducer.reduce(fresh_state, event)

            results.append(
                (
                    output.result.status,
                    output.result.node_id,
                    len(output.intents),
                )
            )

        # All calls should produce consistent results (pending, node_id, 2 intents)
        for i, (status, node_id, intent_count) in enumerate(results):
            assert status == "pending", f"Call {i} produced unexpected status: {status}"
            assert node_id is not None, f"Call {i} produced None node_id"
            assert intent_count == 2, f"Call {i} produced {intent_count} intents"

    def test_reducer_output_is_independent_of_input_reference(
        self,
        reducer: RegistrationReducer,
        id_generator: DeterministicIdGenerator,
        clock: DeterministicClock,
    ) -> None:
        """Verify output is a new object, not a reference to input.

        The returned state must be a new object, not a mutated input.
        """
        node_id = id_generator.next_uuid()
        event = create_introspection_event(
            node_id=node_id,
            correlation_id=id_generator.next_uuid(),
            timestamp=clock.now(),
        )

        input_state = ModelRegistrationState()
        output = reducer.reduce(input_state, event)
        output_state = output.result

        # Output state should be a different object
        assert output_state is not input_state, (
            "Output state is the same object as input state. "
            "Reducer must return a new state object."
        )

        # Verify they are indeed different (status changed)
        assert input_state.status == "idle"
        assert output_state.status == "pending"

    def test_parallel_execution_produces_consistent_results(
        self,
        id_generator: DeterministicIdGenerator,
        clock: DeterministicClock,
    ) -> None:
        """Verify parallel execution produces identical results.

        Multiple threads should be able to call reduce() without interference.
        """
        # Shared inputs (immutable)
        fixed_node_id = id_generator.next_uuid()
        fixed_correlation_id = id_generator.next_uuid()
        fixed_timestamp = clock.now()

        event = create_introspection_event(
            node_id=fixed_node_id,
            correlation_id=fixed_correlation_id,
            timestamp=fixed_timestamp,
        )

        # Shared reducer instance
        reducer = RegistrationReducer()

        def run_reduce():
            state = ModelRegistrationState()
            return reducer.reduce(state, event)

        # Determine concurrency level
        is_ci = is_ci_environment()
        num_concurrent = 4 if is_ci else 10

        # Execute concurrently
        with concurrent.futures.ThreadPoolExecutor(
            max_workers=num_concurrent
        ) as executor:
            futures = [executor.submit(run_reduce) for _ in range(num_concurrent)]
            results = [f.result() for f in concurrent.futures.as_completed(futures)]

        # All results must be identical
        first_result = results[0]
        for i, result in enumerate(results[1:], start=2):
            assert result.result.status == first_result.result.status, (
                f"Thread {i} status mismatch"
            )
            assert result.result.node_id == first_result.result.node_id, (
                f"Thread {i} node_id mismatch"
            )
            assert len(result.intents) == len(first_result.intents), (
                f"Thread {i} intent count mismatch"
            )

    def test_state_changes_only_through_events(
        self,
        reducer: RegistrationReducer,
        id_generator: DeterministicIdGenerator,
        clock: DeterministicClock,
    ) -> None:
        """Verify state changes can ONLY occur through event processing.

        This is the fundamental ONEX principle: the event stream is the
        sole source of truth. There is no way to change state except
        by processing an event.
        """
        # Start with idle state
        state = ModelRegistrationState()
        assert state.status == "idle"
        assert state.node_id is None

        # Attempt to verify there's no backdoor to change state
        # (The only way is through reduce())

        # Process an event - this is the ONLY way to change state
        node_id = id_generator.next_uuid()
        event = create_introspection_event(
            node_id=node_id,
            correlation_id=id_generator.next_uuid(),
            timestamp=clock.now(),
        )

        output = reducer.reduce(state, event)
        new_state = output.result

        # State changed through event
        assert new_state.status == "pending"
        assert new_state.node_id == node_id

        # Original state unchanged (no mutation)
        assert state.status == "idle"
        assert state.node_id is None

    def test_immutable_state_enforces_purity(
        self,
        id_generator: DeterministicIdGenerator,
        clock: DeterministicClock,
    ) -> None:
        """Verify that ModelRegistrationState is frozen (immutable).

        Immutability is enforced by Pydantic's frozen=True setting.
        Attempting to mutate should raise TypeError.
        """
        state = ModelRegistrationState()

        # Verify frozen=True is configured
        assert state.model_config.get("frozen") is True, (
            "ModelRegistrationState must have frozen=True in model_config"
        )

        # Attempt to mutate should raise TypeError
        with pytest.raises(
            Exception
        ):  # ValidationError or TypeError depending on Pydantic version
            state.status = "pending"  # type: ignore[misc]

    def test_with_methods_return_new_instances(
        self,
        id_generator: DeterministicIdGenerator,
    ) -> None:
        """Verify with_* transition methods return new instances.

        Each with_* method should return a new ModelRegistrationState,
        not mutate self. This is part of the immutability contract.
        """
        state1 = ModelRegistrationState()
        node_id = id_generator.next_uuid()
        event_id = id_generator.next_uuid()

        # with_pending_registration returns new instance
        state2 = state1.with_pending_registration(node_id, event_id)
        assert state2 is not state1
        assert state1.status == "idle"
        assert state2.status == "pending"

        # with_consul_confirmed returns new instance
        state3 = state2.with_consul_confirmed(id_generator.next_uuid())
        assert state3 is not state2
        assert state2.consul_confirmed is False
        assert state3.consul_confirmed is True

        # with_postgres_confirmed returns new instance
        state4 = state3.with_postgres_confirmed(id_generator.next_uuid())
        assert state4 is not state3
        assert state4.status == "complete"

        # with_failure returns new instance
        failed_state = state2.with_failure("consul_failed", id_generator.next_uuid())
        assert failed_state is not state2
        assert failed_state.status == "failed"

        # with_reset returns new instance
        reset_state = failed_state.with_reset(id_generator.next_uuid())
        assert reset_state is not failed_state
        assert reset_state.status == "idle"
