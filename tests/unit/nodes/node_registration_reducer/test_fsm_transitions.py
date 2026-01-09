# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""FSM State Transition Tests for NodeRegistrationReducer.

This test suite validates the FSM state transitions defined in contract.yaml
for the NodeRegistrationReducer. The FSM manages the registration lifecycle:

    idle -> pending -> partial -> complete
                   \\           \
                    -> failed <-

FSM Transitions (from contract.yaml):
    - idle -> pending (trigger: introspection_received)
    - pending -> partial (trigger: consul_confirmed OR postgres_confirmed)
    - pending -> failed (trigger: error_received)
    - partial -> complete (trigger: consul_confirmed OR postgres_confirmed - second)
    - partial -> failed (trigger: error_received)
    - failed -> idle (trigger: reset)
    - complete -> idle (trigger: reset)
    - idle -> failed (trigger: validation_failed)

The FSM state transitions are implemented via ModelRegistrationState.with_*()
methods, which create new immutable state instances.

Related:
    - contract.yaml: FSM state machine definition
    - ModelRegistrationState: Immutable state model with transition methods
    - OMN-1104: Refactor to declarative reducer
"""

from __future__ import annotations

from uuid import uuid4

import pytest

from omnibase_infra.nodes.node_registration_reducer.models import (
    ModelRegistrationState,
)

# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def idle_state() -> ModelRegistrationState:
    """Create an initial idle state for testing.

    Returns:
        A new ModelRegistrationState in idle status.
    """
    return ModelRegistrationState()


@pytest.fixture
def pending_state() -> ModelRegistrationState:
    """Create a pending state for testing.

    Returns:
        A ModelRegistrationState in pending status with a node_id.
    """
    state = ModelRegistrationState()
    return state.with_pending_registration(node_id=uuid4(), event_id=uuid4())


@pytest.fixture
def partial_state_consul_confirmed() -> ModelRegistrationState:
    """Create a partial state with Consul confirmed.

    Returns:
        A ModelRegistrationState in partial status with consul_confirmed=True.
    """
    state = ModelRegistrationState()
    pending = state.with_pending_registration(node_id=uuid4(), event_id=uuid4())
    return pending.with_consul_confirmed(event_id=uuid4())


@pytest.fixture
def partial_state_postgres_confirmed() -> ModelRegistrationState:
    """Create a partial state with PostgreSQL confirmed.

    Returns:
        A ModelRegistrationState in partial status with postgres_confirmed=True.
    """
    state = ModelRegistrationState()
    pending = state.with_pending_registration(node_id=uuid4(), event_id=uuid4())
    return pending.with_postgres_confirmed(event_id=uuid4())


@pytest.fixture
def failed_state() -> ModelRegistrationState:
    """Create a failed state for testing.

    Returns:
        A ModelRegistrationState in failed status.
    """
    state = ModelRegistrationState()
    pending = state.with_pending_registration(node_id=uuid4(), event_id=uuid4())
    return pending.with_failure(reason="consul_failed", event_id=uuid4())


@pytest.fixture
def complete_state() -> ModelRegistrationState:
    """Create a complete state for testing.

    Returns:
        A ModelRegistrationState in complete status with both confirmations.
    """
    state = ModelRegistrationState()
    pending = state.with_pending_registration(node_id=uuid4(), event_id=uuid4())
    partial = pending.with_consul_confirmed(event_id=uuid4())
    return partial.with_postgres_confirmed(event_id=uuid4())


# =============================================================================
# FSM Transition Tests
# =============================================================================


@pytest.mark.unit
class TestFSMTransitions:
    """Tests for FSM state transitions defined in contract.yaml."""

    def test_idle_to_pending_on_introspection(
        self,
        idle_state: ModelRegistrationState,
    ) -> None:
        """Test FSM transition: idle -> pending (trigger: introspection_received).

        From contract.yaml:
            - from_state: "idle"
              to_state: "pending"
              trigger: "introspection_received"
              description: "Start registration on introspection event"

        The introspection_received trigger initiates the registration workflow
        by transitioning from idle to pending state.
        """
        assert idle_state.status == "idle"

        node_id = uuid4()
        event_id = uuid4()
        new_state = idle_state.with_pending_registration(node_id, event_id)

        assert new_state.status == "pending"
        assert new_state.node_id == node_id
        assert new_state.last_processed_event_id == event_id
        assert new_state.consul_confirmed is False
        assert new_state.postgres_confirmed is False
        assert new_state.failure_reason is None

    def test_pending_to_partial_on_first_consul_confirmation(
        self,
        pending_state: ModelRegistrationState,
    ) -> None:
        """Test FSM transition: pending -> partial (trigger: consul_confirmed).

        From contract.yaml:
            - from_state: "pending"
              to_state: "partial"
              trigger: "consul_confirmed"
              description: "Consul registration confirmed"

        The first confirmation (Consul) transitions from pending to partial.
        """
        assert pending_state.status == "pending"
        assert pending_state.consul_confirmed is False
        assert pending_state.postgres_confirmed is False

        event_id = uuid4()
        new_state = pending_state.with_consul_confirmed(event_id)

        assert new_state.status == "partial"
        assert new_state.consul_confirmed is True
        assert new_state.postgres_confirmed is False
        assert new_state.last_processed_event_id == event_id

    def test_pending_to_partial_on_first_postgres_confirmation(
        self,
        pending_state: ModelRegistrationState,
    ) -> None:
        """Test FSM transition: pending -> partial (trigger: postgres_confirmed).

        From contract.yaml:
            - from_state: "pending"
              to_state: "partial"
              trigger: "postgres_confirmed"
              description: "PostgreSQL registration confirmed"

        The first confirmation (PostgreSQL) transitions from pending to partial.
        """
        assert pending_state.status == "pending"
        assert pending_state.consul_confirmed is False
        assert pending_state.postgres_confirmed is False

        event_id = uuid4()
        new_state = pending_state.with_postgres_confirmed(event_id)

        assert new_state.status == "partial"
        assert new_state.consul_confirmed is False
        assert new_state.postgres_confirmed is True
        assert new_state.last_processed_event_id == event_id

    def test_partial_to_complete_on_second_confirmation_postgres(
        self,
        partial_state_consul_confirmed: ModelRegistrationState,
    ) -> None:
        """Test FSM transition: partial -> complete (trigger: postgres_confirmed).

        From contract.yaml:
            - from_state: "partial"
              to_state: "complete"
              trigger: "postgres_confirmed"
              description: "PostgreSQL confirmed after consul - registration complete"
              conditions:
                - expression: "consul_confirmed is_true"
                  required: true

        When Consul is already confirmed, PostgreSQL confirmation completes.
        """
        assert partial_state_consul_confirmed.status == "partial"
        assert partial_state_consul_confirmed.consul_confirmed is True
        assert partial_state_consul_confirmed.postgres_confirmed is False

        event_id = uuid4()
        new_state = partial_state_consul_confirmed.with_postgres_confirmed(event_id)

        assert new_state.status == "complete"
        assert new_state.consul_confirmed is True
        assert new_state.postgres_confirmed is True
        assert new_state.last_processed_event_id == event_id

    def test_partial_to_complete_on_second_confirmation_consul(
        self,
        partial_state_postgres_confirmed: ModelRegistrationState,
    ) -> None:
        """Test FSM transition: partial -> complete (trigger: consul_confirmed).

        From contract.yaml:
            - from_state: "partial"
              to_state: "complete"
              trigger: "consul_confirmed"
              description: "Consul confirmed after postgres - registration complete"
              conditions:
                - expression: "postgres_confirmed is_true"
                  required: true

        When PostgreSQL is already confirmed, Consul confirmation completes.
        """
        assert partial_state_postgres_confirmed.status == "partial"
        assert partial_state_postgres_confirmed.consul_confirmed is False
        assert partial_state_postgres_confirmed.postgres_confirmed is True

        event_id = uuid4()
        new_state = partial_state_postgres_confirmed.with_consul_confirmed(event_id)

        assert new_state.status == "complete"
        assert new_state.consul_confirmed is True
        assert new_state.postgres_confirmed is True
        assert new_state.last_processed_event_id == event_id

    def test_pending_to_failed_on_error(
        self,
        pending_state: ModelRegistrationState,
    ) -> None:
        """Test FSM transition: pending -> failed (trigger: error_received).

        From contract.yaml:
            - from_state: "pending"
              to_state: "failed"
              trigger: "error_received"
              description: "Registration failed from pending state"

        An error during registration transitions from pending to failed.
        """
        assert pending_state.status == "pending"

        event_id = uuid4()
        new_state = pending_state.with_failure(
            reason="consul_failed",
            event_id=event_id,
        )

        assert new_state.status == "failed"
        assert new_state.failure_reason == "consul_failed"
        assert new_state.last_processed_event_id == event_id
        # Confirmation state is preserved for diagnostics
        assert new_state.consul_confirmed is False
        assert new_state.postgres_confirmed is False

    def test_partial_to_failed_on_error(
        self,
        partial_state_consul_confirmed: ModelRegistrationState,
    ) -> None:
        """Test FSM transition: partial -> failed (trigger: error_received).

        From contract.yaml:
            - from_state: "partial"
              to_state: "failed"
              trigger: "error_received"
              description: "Registration failed from partial state"

        An error during registration transitions from partial to failed.
        The confirmation state is preserved for diagnostic purposes.
        """
        assert partial_state_consul_confirmed.status == "partial"
        assert partial_state_consul_confirmed.consul_confirmed is True

        event_id = uuid4()
        new_state = partial_state_consul_confirmed.with_failure(
            reason="postgres_failed",
            event_id=event_id,
        )

        assert new_state.status == "failed"
        assert new_state.failure_reason == "postgres_failed"
        assert new_state.last_processed_event_id == event_id
        # Confirmation state preserved for diagnostics
        assert new_state.consul_confirmed is True
        assert new_state.postgres_confirmed is False

    def test_failed_to_idle_on_reset(
        self,
        failed_state: ModelRegistrationState,
    ) -> None:
        """Test FSM transition: failed -> idle (trigger: reset).

        From contract.yaml:
            - from_state: "failed"
              to_state: "idle"
              trigger: "reset"
              description: "Reset from failed state for retry"

        A reset event allows recovery from failed state to idle for retry.
        """
        assert failed_state.status == "failed"
        assert failed_state.failure_reason is not None

        event_id = uuid4()
        new_state = failed_state.with_reset(event_id)

        assert new_state.status == "idle"
        assert new_state.node_id is None
        assert new_state.consul_confirmed is False
        assert new_state.postgres_confirmed is False
        assert new_state.failure_reason is None
        assert new_state.last_processed_event_id == event_id

    def test_complete_to_idle_on_reset(
        self,
        complete_state: ModelRegistrationState,
    ) -> None:
        """Test FSM transition: complete -> idle (trigger: reset).

        From contract.yaml:
            - from_state: "complete"
              to_state: "idle"
              trigger: "reset"
              description: "Reset from complete state for re-registration"

        A reset event allows re-registration from complete state.
        """
        assert complete_state.status == "complete"
        assert complete_state.consul_confirmed is True
        assert complete_state.postgres_confirmed is True

        event_id = uuid4()
        new_state = complete_state.with_reset(event_id)

        assert new_state.status == "idle"
        assert new_state.node_id is None
        assert new_state.consul_confirmed is False
        assert new_state.postgres_confirmed is False
        assert new_state.failure_reason is None
        assert new_state.last_processed_event_id == event_id


@pytest.mark.unit
class TestFSMValidationFailure:
    """Tests for validation failure transition.

    From contract.yaml:
        - from_state: "idle"
          to_state: "failed"
          trigger: "validation_failed"
          description: "Event validation failed"
    """

    def test_idle_to_failed_on_validation_failure(
        self,
        idle_state: ModelRegistrationState,
    ) -> None:
        """Test FSM transition: idle -> failed (trigger: validation_failed).

        When an introspection event fails validation (e.g., missing node_id),
        the FSM transitions directly from idle to failed.
        """
        assert idle_state.status == "idle"

        event_id = uuid4()
        new_state = idle_state.with_failure(
            reason="validation_failed",
            event_id=event_id,
        )

        assert new_state.status == "failed"
        assert new_state.failure_reason == "validation_failed"
        assert new_state.last_processed_event_id == event_id


@pytest.mark.unit
class TestFSMStateImmutability:
    """Tests for FSM state immutability guarantees.

    The ModelRegistrationState is frozen (immutable), ensuring that FSM
    transitions create new instances without modifying the original state.
    """

    def test_transition_creates_new_instance(
        self,
        idle_state: ModelRegistrationState,
    ) -> None:
        """Verify that state transitions create new instances.

        Original state must remain unchanged after transition.
        """
        original_status = idle_state.status
        node_id = uuid4()
        event_id = uuid4()

        new_state = idle_state.with_pending_registration(node_id, event_id)

        # Original state unchanged
        assert idle_state.status == original_status
        assert idle_state.node_id is None

        # New state has updated values
        assert new_state.status == "pending"
        assert new_state.node_id == node_id

        # Different instances
        assert idle_state is not new_state

    def test_chained_transitions_preserve_immutability(
        self,
        idle_state: ModelRegistrationState,
    ) -> None:
        """Verify that chained transitions preserve immutability.

        Each transition in a chain must create a new instance.
        """
        # Capture all intermediate states
        states = [idle_state]

        pending = idle_state.with_pending_registration(uuid4(), uuid4())
        states.append(pending)

        partial = pending.with_consul_confirmed(uuid4())
        states.append(partial)

        complete = partial.with_postgres_confirmed(uuid4())
        states.append(complete)

        reset = complete.with_reset(uuid4())
        states.append(reset)

        # All states are different instances
        for i, state in enumerate(states):
            for j, other in enumerate(states):
                if i != j:
                    assert state is not other, (
                        f"State {i} and {j} should be different instances"
                    )

        # Verify expected statuses in order
        expected_statuses = ["idle", "pending", "partial", "complete", "idle"]
        for state, expected in zip(states, expected_statuses, strict=True):
            assert state.status == expected


@pytest.mark.unit
class TestFSMCanReset:
    """Tests for can_reset() method which validates reset preconditions."""

    def test_can_reset_from_failed_state(
        self,
        failed_state: ModelRegistrationState,
    ) -> None:
        """Verify can_reset() returns True for failed state."""
        assert failed_state.can_reset() is True

    def test_can_reset_from_complete_state(
        self,
        complete_state: ModelRegistrationState,
    ) -> None:
        """Verify can_reset() returns True for complete state."""
        assert complete_state.can_reset() is True

    def test_cannot_reset_from_idle_state(
        self,
        idle_state: ModelRegistrationState,
    ) -> None:
        """Verify can_reset() returns False for idle state."""
        assert idle_state.can_reset() is False

    def test_cannot_reset_from_pending_state(
        self,
        pending_state: ModelRegistrationState,
    ) -> None:
        """Verify can_reset() returns False for pending state."""
        assert pending_state.can_reset() is False

    def test_cannot_reset_from_partial_state(
        self,
        partial_state_consul_confirmed: ModelRegistrationState,
    ) -> None:
        """Verify can_reset() returns False for partial state."""
        assert partial_state_consul_confirmed.can_reset() is False


@pytest.mark.unit
class TestFSMIdempotency:
    """Tests for FSM idempotency via event ID tracking.

    From contract.yaml:
        idempotency:
          enabled: true
          strategy: "event_id_tracking"
    """

    def test_is_duplicate_event_returns_true_for_same_id(
        self,
        idle_state: ModelRegistrationState,
    ) -> None:
        """Verify is_duplicate_event() returns True for matching event ID."""
        event_id = uuid4()
        state = idle_state.with_pending_registration(uuid4(), event_id)

        assert state.is_duplicate_event(event_id) is True

    def test_is_duplicate_event_returns_false_for_different_id(
        self,
        idle_state: ModelRegistrationState,
    ) -> None:
        """Verify is_duplicate_event() returns False for different event ID."""
        event_id = uuid4()
        state = idle_state.with_pending_registration(uuid4(), event_id)

        assert state.is_duplicate_event(uuid4()) is False

    def test_is_duplicate_event_returns_false_for_initial_state(
        self,
        idle_state: ModelRegistrationState,
    ) -> None:
        """Verify is_duplicate_event() returns False when no event processed."""
        assert idle_state.last_processed_event_id is None
        assert idle_state.is_duplicate_event(uuid4()) is False


@pytest.mark.unit
class TestFSMFailureReasons:
    """Tests for valid failure reasons defined in the FSM.

    The FSM supports specific failure reasons for diagnostic purposes.
    """

    @pytest.mark.parametrize(
        "failure_reason",
        [
            "validation_failed",
            "consul_failed",
            "postgres_failed",
            "both_failed",
            "invalid_reset_state",
        ],
    )
    def test_all_failure_reasons_are_valid(
        self,
        pending_state: ModelRegistrationState,
        failure_reason: str,
    ) -> None:
        """Verify all defined failure reasons can be set."""
        event_id = uuid4()
        new_state = pending_state.with_failure(
            reason=failure_reason,  # type: ignore[arg-type]
            event_id=event_id,
        )

        assert new_state.status == "failed"
        assert new_state.failure_reason == failure_reason
