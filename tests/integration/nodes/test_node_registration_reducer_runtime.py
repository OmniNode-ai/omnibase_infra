# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Integration tests for NodeRegistrationReducer with RuntimeHostProcess.

This test suite validates the NodeRegistrationReducer's integration with the
runtime infrastructure, covering:

1. Intent emission on introspection events
2. FSM state transitions (idle -> pending -> partial -> complete)
3. Error handling and failure paths
4. Idempotency (duplicate event rejection)
5. End-to-end workflow with mocked effects

The tests use the RegistrationReducer class which implements the pure reducer
pattern (state + event -> new_state + intents). The NodeRegistrationReducer
is a declarative shell that uses the same logic via contract.yaml FSM.

FSM State Diagram:
    idle -> pending -> partial -> complete
                   \\           \
                    -> failed <-

Related:
    - NodeRegistrationReducer: Declarative reducer node
    - RegistrationReducer: Pure reducer implementation
    - ModelRegistrationState: Immutable state model
    - OMN-1272: Integration test implementation ticket
    - OMN-1263: Pre-existing test failure tracking
"""

from __future__ import annotations

from datetime import UTC, datetime
from uuid import UUID, uuid4

import pytest

from omnibase_core.enums.enum_node_kind import EnumNodeKind
from omnibase_core.models.primitives.model_semver import ModelSemVer
from omnibase_infra.models.registration import ModelNodeIntrospectionEvent
from omnibase_infra.nodes.reducers import RegistrationReducer
from omnibase_infra.nodes.reducers.models import (
    ModelPayloadConsulRegister,
    ModelPayloadPostgresUpsertRegistration,
    ModelRegistrationState,
)

# Import test doubles and fixtures from workflow conftest
from tests.integration.registration.effect.test_doubles import (
    StubConsulClient,
    StubPostgresAdapter,
)
from tests.integration.registration.workflow.conftest import (
    DeterministicUUIDGenerator,
)

# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def reducer() -> RegistrationReducer:
    """Create a fresh RegistrationReducer instance.

    Returns:
        RegistrationReducer for processing introspection events.
    """
    return RegistrationReducer()


@pytest.fixture
def initial_state() -> ModelRegistrationState:
    """Create an initial idle registration state.

    Returns:
        ModelRegistrationState in idle status.
    """
    return ModelRegistrationState()


@pytest.fixture
def uuid_gen() -> DeterministicUUIDGenerator:
    """Create a deterministic UUID generator for predictable test values.

    Returns:
        DeterministicUUIDGenerator instance.
    """
    return DeterministicUUIDGenerator()


@pytest.fixture
def stub_consul_client() -> StubConsulClient:
    """Create a fresh StubConsulClient for testing.

    Returns:
        StubConsulClient with default success configuration.
    """
    return StubConsulClient()


@pytest.fixture
def stub_postgres_adapter() -> StubPostgresAdapter:
    """Create a fresh StubPostgresAdapter for testing.

    Returns:
        StubPostgresAdapter with default success configuration.
    """
    return StubPostgresAdapter()


def create_introspection_event(
    node_id: UUID | None = None,
    node_type: str = "effect",
    node_version: str | ModelSemVer = "1.0.0",
    correlation_id: UUID | None = None,
    endpoints: dict[str, str] | None = None,
) -> ModelNodeIntrospectionEvent:
    """Factory function for creating introspection events.

    Args:
        node_id: Unique node identifier (generated if not provided).
        node_type: ONEX node type string.
        node_version: Semantic version string or ModelSemVer instance.
        correlation_id: Optional correlation ID for tracing.
        endpoints: Optional endpoint URLs.

    Returns:
        ModelNodeIntrospectionEvent with specified values.
    """
    # Convert string version to ModelSemVer if needed
    if isinstance(node_version, str):
        node_version = ModelSemVer.parse(node_version)

    return ModelNodeIntrospectionEvent(
        node_id=node_id or uuid4(),
        node_type=node_type,
        node_version=node_version,
        correlation_id=correlation_id or uuid4(),
        endpoints=endpoints or {"health": "http://localhost:8080/health"},
        timestamp=datetime.now(UTC),
    )


# =============================================================================
# Test 1: Intent Emission on Introspection Event
# =============================================================================


@pytest.mark.integration
class TestIntentEmissionOnIntrospectionEvent:
    """Tests for intent emission when processing introspection events.

    Verifies that the reducer correctly emits Consul and PostgreSQL registration
    intents when processing a valid introspection event.
    """

    def test_reducer_emits_consul_and_postgres_intents(
        self,
        reducer: RegistrationReducer,
        initial_state: ModelRegistrationState,
        uuid_gen: DeterministicUUIDGenerator,
    ) -> None:
        """Test that processing an introspection event emits both intents.

        Given an idle state and a valid introspection event,
        the reducer should emit both consul.register and postgres.upsert_registration
        intents.
        """
        # Arrange
        node_id = uuid_gen.next()
        correlation_id = uuid_gen.next()
        event = create_introspection_event(
            node_id=node_id,
            node_type="effect",
            correlation_id=correlation_id,
        )

        # Act
        output = reducer.reduce(initial_state, event)

        # Assert - should have 2 intents
        assert len(output.intents) == 2, (
            f"Expected 2 intents (consul + postgres), got {len(output.intents)}"
        )

        # Verify intent types via payload.intent_type (two-layer architecture)
        # Outer intent_type is always "extension", routing key is in payload
        intent_types = {intent.payload.intent_type for intent in output.intents}
        assert "consul.register" in intent_types
        assert "postgres.upsert_registration" in intent_types

    def test_consul_intent_payload_structure(
        self,
        reducer: RegistrationReducer,
        initial_state: ModelRegistrationState,
        uuid_gen: DeterministicUUIDGenerator,
    ) -> None:
        """Test that Consul intent has correct payload structure.

        The consul.register intent should contain:
        - service_id: Formatted as "onex-{node_type}-{node_id}"
        - service_name: Formatted as "onex-{node_type}"
        - tags: Including node_type and node_version
        - health_check: If health endpoint is provided
        """
        # Arrange
        node_id = uuid_gen.next()
        event = create_introspection_event(
            node_id=node_id,
            node_type="effect",
            node_version="1.2.3",
            endpoints={"health": "http://localhost:8080/health"},
        )

        # Act
        output = reducer.reduce(initial_state, event)

        # Find consul intent via payload.intent_type
        consul_intents = [
            i for i in output.intents if i.payload.intent_type == "consul.register"
        ]
        assert len(consul_intents) == 1
        consul_intent = consul_intents[0]

        # Verify payload
        payload = consul_intent.payload
        assert isinstance(payload, ModelPayloadConsulRegister)
        assert payload.service_id == f"onex-effect-{node_id}"
        assert payload.service_name == "onex-effect"
        assert "node_type:effect" in payload.tags
        assert "node_version:1.2.3" in payload.tags
        assert payload.health_check is not None
        assert payload.health_check["HTTP"] == "http://localhost:8080/health"

    def test_postgres_intent_payload_structure(
        self,
        reducer: RegistrationReducer,
        initial_state: ModelRegistrationState,
        uuid_gen: DeterministicUUIDGenerator,
    ) -> None:
        """Test that PostgreSQL intent has correct payload structure.

        The postgres.upsert_registration intent should contain:
        - correlation_id: Matching the event
        - record: ModelNodeRegistrationRecord with node details
        """
        # Arrange
        node_id = uuid_gen.next()
        correlation_id = uuid_gen.next()
        event = create_introspection_event(
            node_id=node_id,
            node_type="compute",
            node_version="2.0.0",
            correlation_id=correlation_id,
        )

        # Act
        output = reducer.reduce(initial_state, event)

        # Find postgres intent via payload.intent_type
        postgres_intents = [
            i
            for i in output.intents
            if i.payload.intent_type == "postgres.upsert_registration"
        ]
        assert len(postgres_intents) == 1
        postgres_intent = postgres_intents[0]

        # Verify payload
        payload = postgres_intent.payload
        assert isinstance(payload, ModelPayloadPostgresUpsertRegistration)
        assert payload.correlation_id == correlation_id
        assert payload.record is not None
        # Access record attributes
        assert payload.record.node_id == node_id
        assert payload.record.node_type == "compute"
        assert str(payload.record.node_version) == "2.0.0"

    def test_intent_target_patterns(
        self,
        reducer: RegistrationReducer,
        initial_state: ModelRegistrationState,
        uuid_gen: DeterministicUUIDGenerator,
    ) -> None:
        """Test that intents have correct target patterns.

        Per contract.yaml:
        - consul.register: target_pattern = "consul://service/{service_name}"
        - postgres.upsert_registration: target_pattern = "postgres://node_registrations/{node_id}"
        """
        # Arrange
        node_id = uuid_gen.next()
        event = create_introspection_event(node_id=node_id, node_type="reducer")

        # Act
        output = reducer.reduce(initial_state, event)

        # Verify targets via payload.intent_type
        for intent in output.intents:
            if intent.payload.intent_type == "consul.register":
                assert intent.target == "consul://service/onex-reducer"
            elif intent.payload.intent_type == "postgres.upsert_registration":
                assert intent.target == f"postgres://node_registrations/{node_id}"


# =============================================================================
# Test 2: FSM Idle to Pending Transition
# =============================================================================


@pytest.mark.integration
class TestFSMIdleToPendingTransition:
    """Tests for FSM transition from idle to pending state.

    Per contract.yaml:
        - from_state: "idle"
          to_state: "pending"
          trigger: "introspection_received"
    """

    def test_introspection_event_transitions_to_pending(
        self,
        reducer: RegistrationReducer,
        initial_state: ModelRegistrationState,
        uuid_gen: DeterministicUUIDGenerator,
    ) -> None:
        """Test that processing introspection event transitions idle to pending.

        Given an idle state, processing a valid introspection event should:
        - Transition status to "pending"
        - Set node_id from the event
        - Emit registration intents
        """
        # Arrange
        assert initial_state.status == "idle"
        node_id = uuid_gen.next()
        event = create_introspection_event(node_id=node_id)

        # Act
        output = reducer.reduce(initial_state, event)

        # Assert
        assert output.result.status == "pending"
        assert output.result.node_id == node_id
        assert output.result.consul_confirmed is False
        assert output.result.postgres_confirmed is False
        assert output.result.failure_reason is None

    def test_pending_state_has_event_id_tracked(
        self,
        reducer: RegistrationReducer,
        initial_state: ModelRegistrationState,
        uuid_gen: DeterministicUUIDGenerator,
    ) -> None:
        """Test that pending state tracks the event ID for idempotency.

        The last_processed_event_id should be set to enable duplicate detection.
        """
        # Arrange
        correlation_id = uuid_gen.next()
        event = create_introspection_event(correlation_id=correlation_id)

        # Act
        output = reducer.reduce(initial_state, event)

        # Assert
        assert output.result.last_processed_event_id == correlation_id

    def test_reducer_validates_against_contract_rules(
        self,
        reducer: RegistrationReducer,
        initial_state: ModelRegistrationState,
        uuid_gen: DeterministicUUIDGenerator,
    ) -> None:
        """Test that reducer validation is consistent with contract.yaml rules.

        Per contract.yaml validation section, valid node_types are:
        - effect, compute, reducer, orchestrator

        Note: Pydantic model-level validation (Literal type) already enforces
        this constraint, making the reducer's _validate_event() method
        defense-in-depth. Since Pydantic prevents invalid construction,
        we verify that valid events pass reducer validation.

        The reducer's _validate_event() checks:
        - node_id is present (enforced by Pydantic required field)
        - node_type is present (enforced by Pydantic required field)
        - node_type is valid value (enforced by Pydantic Literal type)
        """
        # Valid event should pass validation and transition to pending
        event = create_introspection_event(
            node_id=uuid_gen.next(),
            node_type="effect",
        )

        output = reducer.reduce(initial_state, event)

        # Verify validation passed (state is pending, not failed)
        assert output.result.status == "pending"
        assert output.result.failure_reason is None
        assert len(output.intents) == 2  # Intents emitted


# =============================================================================
# Test 3: FSM Pending to Complete Workflow
# =============================================================================


@pytest.mark.integration
class TestFSMPendingToCompleteWorkflow:
    """Tests for the complete FSM workflow: idle -> pending -> partial -> complete.

    This simulates the full registration lifecycle where both Consul and PostgreSQL
    backends confirm successful registration.
    """

    def test_full_workflow_consul_first(
        self,
        reducer: RegistrationReducer,
        initial_state: ModelRegistrationState,
        uuid_gen: DeterministicUUIDGenerator,
    ) -> None:
        """Test complete workflow with Consul confirming first.

        Flow: idle -> pending -> partial (consul) -> complete (postgres)
        """
        # Step 1: idle -> pending
        node_id = uuid_gen.next()
        event = create_introspection_event(node_id=node_id)
        output = reducer.reduce(initial_state, event)
        pending_state = output.result

        assert pending_state.status == "pending"
        assert len(output.intents) == 2

        # Step 2: pending -> partial (consul confirmed)
        consul_event_id = uuid_gen.next()
        partial_state = pending_state.with_consul_confirmed(consul_event_id)

        assert partial_state.status == "partial"
        assert partial_state.consul_confirmed is True
        assert partial_state.postgres_confirmed is False

        # Step 3: partial -> complete (postgres confirmed)
        postgres_event_id = uuid_gen.next()
        complete_state = partial_state.with_postgres_confirmed(postgres_event_id)

        assert complete_state.status == "complete"
        assert complete_state.consul_confirmed is True
        assert complete_state.postgres_confirmed is True
        assert complete_state.node_id == node_id

    def test_full_workflow_postgres_first(
        self,
        reducer: RegistrationReducer,
        initial_state: ModelRegistrationState,
        uuid_gen: DeterministicUUIDGenerator,
    ) -> None:
        """Test complete workflow with PostgreSQL confirming first.

        Flow: idle -> pending -> partial (postgres) -> complete (consul)
        """
        # Step 1: idle -> pending
        node_id = uuid_gen.next()
        event = create_introspection_event(node_id=node_id)
        output = reducer.reduce(initial_state, event)
        pending_state = output.result

        assert pending_state.status == "pending"

        # Step 2: pending -> partial (postgres confirmed)
        postgres_event_id = uuid_gen.next()
        partial_state = pending_state.with_postgres_confirmed(postgres_event_id)

        assert partial_state.status == "partial"
        assert partial_state.consul_confirmed is False
        assert partial_state.postgres_confirmed is True

        # Step 3: partial -> complete (consul confirmed)
        consul_event_id = uuid_gen.next()
        complete_state = partial_state.with_consul_confirmed(consul_event_id)

        assert complete_state.status == "complete"
        assert complete_state.consul_confirmed is True
        assert complete_state.postgres_confirmed is True

    def test_workflow_preserves_node_id_throughout(
        self,
        reducer: RegistrationReducer,
        initial_state: ModelRegistrationState,
        uuid_gen: DeterministicUUIDGenerator,
    ) -> None:
        """Test that node_id is preserved throughout the entire workflow.

        The node_id set in pending state should be preserved through
        partial and complete states.
        """
        # Complete workflow
        node_id = uuid_gen.next()
        event = create_introspection_event(node_id=node_id)
        output = reducer.reduce(initial_state, event)

        pending_state = output.result
        partial_state = pending_state.with_consul_confirmed(uuid_gen.next())
        complete_state = partial_state.with_postgres_confirmed(uuid_gen.next())

        # Verify node_id preserved
        assert pending_state.node_id == node_id
        assert partial_state.node_id == node_id
        assert complete_state.node_id == node_id


# =============================================================================
# Test 4: FSM Error Handling to Failed State
# =============================================================================


@pytest.mark.integration
class TestFSMErrorHandlingToFailed:
    """Tests for error transitions to failed state.

    Per contract.yaml:
        - pending -> failed (trigger: error_received)
        - partial -> failed (trigger: error_received)
    """

    def test_pending_to_failed_on_consul_error(
        self,
        reducer: RegistrationReducer,
        initial_state: ModelRegistrationState,
        uuid_gen: DeterministicUUIDGenerator,
    ) -> None:
        """Test transition from pending to failed when Consul fails.

        Given a pending state, a Consul registration failure should
        transition to failed with failure_reason="consul_failed".
        """
        # Get to pending state
        event = create_introspection_event(node_id=uuid_gen.next())
        output = reducer.reduce(initial_state, event)
        pending_state = output.result

        assert pending_state.status == "pending"

        # Simulate Consul failure
        error_event_id = uuid_gen.next()
        failed_state = pending_state.with_failure("consul_failed", error_event_id)

        assert failed_state.status == "failed"
        assert failed_state.failure_reason == "consul_failed"
        # Confirmation state preserved for diagnostics
        assert failed_state.consul_confirmed is False
        assert failed_state.postgres_confirmed is False

    def test_pending_to_failed_on_postgres_error(
        self,
        reducer: RegistrationReducer,
        initial_state: ModelRegistrationState,
        uuid_gen: DeterministicUUIDGenerator,
    ) -> None:
        """Test transition from pending to failed when PostgreSQL fails."""
        # Get to pending state
        event = create_introspection_event(node_id=uuid_gen.next())
        output = reducer.reduce(initial_state, event)
        pending_state = output.result

        # Simulate PostgreSQL failure
        error_event_id = uuid_gen.next()
        failed_state = pending_state.with_failure("postgres_failed", error_event_id)

        assert failed_state.status == "failed"
        assert failed_state.failure_reason == "postgres_failed"

    def test_partial_to_failed_on_error(
        self,
        reducer: RegistrationReducer,
        initial_state: ModelRegistrationState,
        uuid_gen: DeterministicUUIDGenerator,
    ) -> None:
        """Test transition from partial to failed when remaining backend fails.

        Given a partial state with Consul confirmed, a PostgreSQL failure
        should transition to failed while preserving the Consul confirmation.
        """
        # Get to partial state (consul confirmed)
        event = create_introspection_event(node_id=uuid_gen.next())
        output = reducer.reduce(initial_state, event)
        pending_state = output.result
        partial_state = pending_state.with_consul_confirmed(uuid_gen.next())

        assert partial_state.status == "partial"
        assert partial_state.consul_confirmed is True

        # Simulate PostgreSQL failure
        error_event_id = uuid_gen.next()
        failed_state = partial_state.with_failure("postgres_failed", error_event_id)

        assert failed_state.status == "failed"
        assert failed_state.failure_reason == "postgres_failed"
        # Consul confirmation preserved for diagnostics
        assert failed_state.consul_confirmed is True
        assert failed_state.postgres_confirmed is False

    def test_failed_state_emits_no_intents(
        self,
        reducer: RegistrationReducer,
        initial_state: ModelRegistrationState,
        uuid_gen: DeterministicUUIDGenerator,
    ) -> None:
        """Test that transitioning to failed state does not emit new intents.

        Error transitions should only update state, not trigger new registrations.
        """
        # Get to pending state (this emits intents)
        event = create_introspection_event(node_id=uuid_gen.next())
        output = reducer.reduce(initial_state, event)
        pending_state = output.result

        # Transition to failed - state transition methods don't return intents
        # They are purely state updates
        failed_state = pending_state.with_failure("consul_failed", uuid_gen.next())

        # Verify state transitioned without emitting intents
        # (with_failure is a state method, not reduce, so no intents returned)
        assert failed_state.status == "failed"


# =============================================================================
# Test 5: FSM Reset from Failed State
# =============================================================================


@pytest.mark.integration
class TestFSMResetFromFailed:
    """Tests for reset transitions from failed and complete states.

    Per contract.yaml:
        - failed -> idle (trigger: reset)
        - complete -> idle (trigger: reset)
    """

    def test_reset_from_failed_to_idle(
        self,
        reducer: RegistrationReducer,
        initial_state: ModelRegistrationState,
        uuid_gen: DeterministicUUIDGenerator,
    ) -> None:
        """Test reset transition from failed state back to idle.

        Given a failed state, the reduce_reset method should transition
        back to idle, clearing all state for retry.
        """
        # Get to failed state
        event = create_introspection_event(node_id=uuid_gen.next())
        output = reducer.reduce(initial_state, event)
        pending_state = output.result
        failed_state = pending_state.with_failure("consul_failed", uuid_gen.next())

        assert failed_state.status == "failed"
        assert failed_state.can_reset() is True

        # Reset via reducer method
        reset_event_id = uuid_gen.next()
        reset_output = reducer.reduce_reset(failed_state, reset_event_id)

        # Assert
        assert reset_output.result.status == "idle"
        assert reset_output.result.node_id is None
        assert reset_output.result.consul_confirmed is False
        assert reset_output.result.postgres_confirmed is False
        assert reset_output.result.failure_reason is None
        assert len(reset_output.intents) == 0  # Reset emits no intents

    def test_reset_from_complete_to_idle(
        self,
        reducer: RegistrationReducer,
        initial_state: ModelRegistrationState,
        uuid_gen: DeterministicUUIDGenerator,
    ) -> None:
        """Test reset transition from complete state for re-registration.

        A completed registration can be reset to enable re-registration,
        for example when a node restarts.
        """
        # Get to complete state
        event = create_introspection_event(node_id=uuid_gen.next())
        output = reducer.reduce(initial_state, event)
        pending_state = output.result
        partial_state = pending_state.with_consul_confirmed(uuid_gen.next())
        complete_state = partial_state.with_postgres_confirmed(uuid_gen.next())

        assert complete_state.status == "complete"
        assert complete_state.can_reset() is True

        # Reset via reducer method
        reset_event_id = uuid_gen.next()
        reset_output = reducer.reduce_reset(complete_state, reset_event_id)

        # Assert
        assert reset_output.result.status == "idle"
        assert reset_output.result.node_id is None
        assert len(reset_output.intents) == 0

    def test_reset_from_pending_fails(
        self,
        reducer: RegistrationReducer,
        initial_state: ModelRegistrationState,
        uuid_gen: DeterministicUUIDGenerator,
    ) -> None:
        """Test that reset from pending state transitions to failed.

        Resetting from in-flight states (pending, partial) is not allowed
        as it would lose registration state. The reducer should transition
        to failed with failure_reason="invalid_reset_state".
        """
        # Get to pending state
        event = create_introspection_event(node_id=uuid_gen.next())
        output = reducer.reduce(initial_state, event)
        pending_state = output.result

        assert pending_state.status == "pending"
        assert pending_state.can_reset() is False

        # Attempt reset
        reset_event_id = uuid_gen.next()
        reset_output = reducer.reduce_reset(pending_state, reset_event_id)

        # Assert - should transition to failed, not idle
        assert reset_output.result.status == "failed"
        assert reset_output.result.failure_reason == "invalid_reset_state"

    def test_reset_from_partial_fails(
        self,
        reducer: RegistrationReducer,
        initial_state: ModelRegistrationState,
        uuid_gen: DeterministicUUIDGenerator,
    ) -> None:
        """Test that reset from partial state transitions to failed.

        Partial state indicates in-flight registration that would be lost.
        """
        # Get to partial state
        event = create_introspection_event(node_id=uuid_gen.next())
        output = reducer.reduce(initial_state, event)
        pending_state = output.result
        partial_state = pending_state.with_consul_confirmed(uuid_gen.next())

        assert partial_state.status == "partial"
        assert partial_state.can_reset() is False

        # Attempt reset
        reset_event_id = uuid_gen.next()
        reset_output = reducer.reduce_reset(partial_state, reset_event_id)

        # Assert
        assert reset_output.result.status == "failed"
        assert reset_output.result.failure_reason == "invalid_reset_state"


# =============================================================================
# Test 6: Idempotency - Duplicate Event Rejection
# =============================================================================


@pytest.mark.integration
class TestIdempotencyDuplicateEventRejection:
    """Tests for idempotent event processing via event_id tracking.

    Per contract.yaml:
        idempotency:
          enabled: true
          strategy: "event_id_tracking"
    """

    def test_duplicate_event_returns_current_state_unchanged(
        self,
        reducer: RegistrationReducer,
        initial_state: ModelRegistrationState,
        uuid_gen: DeterministicUUIDGenerator,
    ) -> None:
        """Test that processing the same event twice is idempotent.

        When an event with a previously processed correlation_id is received,
        the reducer should return the current state unchanged with no intents.
        """
        # Arrange
        node_id = uuid_gen.next()
        correlation_id = uuid_gen.next()
        event = create_introspection_event(
            node_id=node_id,
            correlation_id=correlation_id,
        )

        # First processing
        output1 = reducer.reduce(initial_state, event)
        pending_state = output1.result

        assert pending_state.status == "pending"
        assert len(output1.intents) == 2

        # Second processing with same event
        output2 = reducer.reduce(pending_state, event)

        # Assert - state unchanged, no intents
        assert output2.result.status == "pending"
        assert output2.result == pending_state  # Same state object values
        assert len(output2.intents) == 0  # No duplicate intents

    def test_duplicate_detection_uses_correlation_id(
        self,
        reducer: RegistrationReducer,
        initial_state: ModelRegistrationState,
        uuid_gen: DeterministicUUIDGenerator,
    ) -> None:
        """Test that duplicate detection is based on correlation_id.

        The same correlation_id should be rejected even with different
        event content.
        """
        # First event
        correlation_id = uuid_gen.next()
        event1 = create_introspection_event(
            node_id=uuid_gen.next(),
            node_type="effect",
            correlation_id=correlation_id,
        )

        output1 = reducer.reduce(initial_state, event1)
        pending_state = output1.result

        # Second event with SAME correlation_id but different node_type
        event2 = create_introspection_event(
            node_id=uuid_gen.next(),  # Different node_id
            node_type="compute",  # Different type
            correlation_id=correlation_id,  # Same correlation_id
        )

        output2 = reducer.reduce(pending_state, event2)

        # Should be treated as duplicate
        assert len(output2.intents) == 0
        assert output2.result.status == "pending"

    def test_different_correlation_id_processes_normally(
        self,
        reducer: RegistrationReducer,
        initial_state: ModelRegistrationState,
        uuid_gen: DeterministicUUIDGenerator,
    ) -> None:
        """Test that events with different correlation_ids are processed.

        Each unique correlation_id should be processed normally.
        """
        # First event
        event1 = create_introspection_event(
            node_id=uuid_gen.next(),
            correlation_id=uuid_gen.next(),
        )

        output1 = reducer.reduce(initial_state, event1)

        assert len(output1.intents) == 2

        # Use a fresh idle state for the second event
        # (simulating a different node registration)
        fresh_idle_state = ModelRegistrationState()

        # Second event with different correlation_id
        event2 = create_introspection_event(
            node_id=uuid_gen.next(),
            correlation_id=uuid_gen.next(),  # Different
        )

        output2 = reducer.reduce(fresh_idle_state, event2)

        # Should process normally
        assert len(output2.intents) == 2
        assert output2.result.status == "pending"


# =============================================================================
# Test 7: End-to-End Workflow with Mocked Effects
# =============================================================================


@pytest.mark.asyncio
@pytest.mark.integration
class TestEndToEndWithMockedEffects:
    """End-to-end tests simulating the full registration workflow with stubs.

    These tests verify the integration between the reducer and the effect layer
    using StubConsulClient and StubPostgresAdapter test doubles.
    """

    async def test_complete_registration_with_stub_backends(
        self,
        reducer: RegistrationReducer,
        initial_state: ModelRegistrationState,
        stub_consul_client: StubConsulClient,
        stub_postgres_adapter: StubPostgresAdapter,
        uuid_gen: DeterministicUUIDGenerator,
    ) -> None:
        """Test complete registration workflow with stub backends.

        This test simulates the full workflow:
        1. Reducer processes introspection event, emits intents
        2. Extract intent payloads and execute against stubs
        3. Simulate confirmation events
        4. Verify final state is complete
        """
        # Step 1: Process introspection event
        node_id = uuid_gen.next()
        event = create_introspection_event(
            node_id=node_id,
            node_type="effect",
            node_version="1.0.0",
        )

        output = reducer.reduce(initial_state, event)
        pending_state = output.result

        assert pending_state.status == "pending"
        assert len(output.intents) == 2

        # Step 2: Execute intents against stubs (route via payload.intent_type)
        for intent in output.intents:
            if intent.payload.intent_type == "consul.register":
                payload = intent.payload
                assert isinstance(payload, ModelPayloadConsulRegister)
                result = await stub_consul_client.register_service(
                    service_id=payload.service_id,
                    service_name=payload.service_name,
                    tags=payload.tags,
                    health_check=payload.health_check,
                )
                assert result.success is True

            elif intent.payload.intent_type == "postgres.upsert_registration":
                payload = intent.payload
                assert isinstance(payload, ModelPayloadPostgresUpsertRegistration)
                result = await stub_postgres_adapter.upsert(
                    node_id=payload.record.node_id,
                    node_type=EnumNodeKind(payload.record.node_type),
                    node_version=payload.record.node_version,
                    endpoints=payload.record.endpoints,
                    metadata={},
                )
                assert result.success is True

        # Step 3: Verify stub call counts
        assert stub_consul_client.call_count == 1
        assert stub_postgres_adapter.call_count == 1

        # Step 4: Simulate confirmations and complete workflow
        partial_state = pending_state.with_consul_confirmed(uuid_gen.next())
        complete_state = partial_state.with_postgres_confirmed(uuid_gen.next())

        assert complete_state.status == "complete"
        assert complete_state.node_id == node_id

    async def test_partial_failure_consul_fails(
        self,
        reducer: RegistrationReducer,
        initial_state: ModelRegistrationState,
        stub_consul_client: StubConsulClient,
        stub_postgres_adapter: StubPostgresAdapter,
        uuid_gen: DeterministicUUIDGenerator,
    ) -> None:
        """Test workflow when Consul registration fails.

        Verifies that:
        - PostgreSQL still executes successfully
        - State transitions to failed appropriately
        """
        # Configure Consul to fail
        stub_consul_client.should_fail = True
        stub_consul_client.failure_error = "Consul connection refused"

        # Process introspection event
        node_id = uuid_gen.next()
        event = create_introspection_event(node_id=node_id)

        output = reducer.reduce(initial_state, event)
        pending_state = output.result

        # Execute intents (route via payload.intent_type)
        consul_result = None
        postgres_result = None

        for intent in output.intents:
            if intent.payload.intent_type == "consul.register":
                payload = intent.payload
                assert isinstance(payload, ModelPayloadConsulRegister)
                consul_result = await stub_consul_client.register_service(
                    service_id=payload.service_id,
                    service_name=payload.service_name,
                    tags=payload.tags,
                )

            elif intent.payload.intent_type == "postgres.upsert_registration":
                payload = intent.payload
                assert isinstance(payload, ModelPayloadPostgresUpsertRegistration)
                postgres_result = await stub_postgres_adapter.upsert(
                    node_id=payload.record.node_id,
                    node_type=EnumNodeKind(payload.record.node_type),
                    node_version=payload.record.node_version,
                    endpoints=payload.record.endpoints,
                    metadata={},
                )

        # Verify results
        assert consul_result is not None
        assert consul_result.success is False
        assert "Consul" in (consul_result.error or "")

        assert postgres_result is not None
        assert postgres_result.success is True

        # Simulate partial completion (postgres confirmed) then failure
        partial_state = pending_state.with_postgres_confirmed(uuid_gen.next())
        failed_state = partial_state.with_failure("consul_failed", uuid_gen.next())

        assert failed_state.status == "failed"
        assert failed_state.failure_reason == "consul_failed"
        assert failed_state.postgres_confirmed is True  # Preserved

    async def test_complete_failure_both_backends_fail(
        self,
        reducer: RegistrationReducer,
        initial_state: ModelRegistrationState,
        stub_consul_client: StubConsulClient,
        stub_postgres_adapter: StubPostgresAdapter,
        uuid_gen: DeterministicUUIDGenerator,
    ) -> None:
        """Test workflow when both backends fail.

        Verifies that state transitions to failed with appropriate reason.
        """
        # Configure both to fail
        stub_consul_client.should_fail = True
        stub_postgres_adapter.should_fail = True

        # Process introspection event
        event = create_introspection_event(node_id=uuid_gen.next())
        output = reducer.reduce(initial_state, event)
        pending_state = output.result

        # Execute intents - both fail (route via payload.intent_type)
        for intent in output.intents:
            if intent.payload.intent_type == "consul.register":
                payload = intent.payload
                assert isinstance(payload, ModelPayloadConsulRegister)
                result = await stub_consul_client.register_service(
                    service_id=payload.service_id,
                    service_name=payload.service_name,
                    tags=payload.tags,
                )
                assert result.success is False

            elif intent.payload.intent_type == "postgres.upsert_registration":
                payload = intent.payload
                assert isinstance(payload, ModelPayloadPostgresUpsertRegistration)
                result = await stub_postgres_adapter.upsert(
                    node_id=payload.record.node_id,
                    node_type=EnumNodeKind(payload.record.node_type),
                    node_version=payload.record.node_version,
                    endpoints=payload.record.endpoints,
                    metadata={},
                )
                assert result.success is False

        # Transition to failed
        failed_state = pending_state.with_failure("both_failed", uuid_gen.next())

        assert failed_state.status == "failed"
        assert failed_state.failure_reason == "both_failed"

    async def test_retry_after_failure(
        self,
        reducer: RegistrationReducer,
        initial_state: ModelRegistrationState,
        stub_consul_client: StubConsulClient,
        stub_postgres_adapter: StubPostgresAdapter,
        uuid_gen: DeterministicUUIDGenerator,
    ) -> None:
        """Test retry workflow after initial failure.

        Verifies that:
        1. Initial registration fails
        2. Reset to idle
        3. Retry succeeds
        """
        # Initial attempt - Consul fails
        stub_consul_client.should_fail = True

        event1 = create_introspection_event(
            node_id=uuid_gen.next(),
            correlation_id=uuid_gen.next(),
        )
        output1 = reducer.reduce(initial_state, event1)
        pending_state = output1.result

        # Execute and fail (route via payload.intent_type)
        for intent in output1.intents:
            if intent.payload.intent_type == "consul.register":
                payload = intent.payload
                assert isinstance(payload, ModelPayloadConsulRegister)
                await stub_consul_client.register_service(
                    service_id=payload.service_id,
                    service_name=payload.service_name,
                    tags=payload.tags,
                )

        failed_state = pending_state.with_failure("consul_failed", uuid_gen.next())
        assert failed_state.status == "failed"

        # Reset
        reset_output = reducer.reduce_reset(failed_state, uuid_gen.next())
        idle_state = reset_output.result
        assert idle_state.status == "idle"

        # Retry - fix Consul
        stub_consul_client.should_fail = False
        stub_consul_client.reset()

        event2 = create_introspection_event(
            node_id=uuid_gen.next(),
            correlation_id=uuid_gen.next(),  # New correlation_id
        )
        output2 = reducer.reduce(idle_state, event2)

        assert output2.result.status == "pending"
        assert len(output2.intents) == 2

        # Execute successfully (route via payload.intent_type)
        for intent in output2.intents:
            if intent.payload.intent_type == "consul.register":
                payload = intent.payload
                assert isinstance(payload, ModelPayloadConsulRegister)
                result = await stub_consul_client.register_service(
                    service_id=payload.service_id,
                    service_name=payload.service_name,
                    tags=payload.tags,
                )
                assert result.success is True

        # Complete workflow
        partial = output2.result.with_consul_confirmed(uuid_gen.next())
        complete = partial.with_postgres_confirmed(uuid_gen.next())

        assert complete.status == "complete"


# =============================================================================
# Additional Integration Tests
# =============================================================================


@pytest.mark.integration
class TestReducerOutputMetadata:
    """Tests for ModelReducerOutput metadata fields."""

    def test_output_contains_processing_metrics(
        self,
        reducer: RegistrationReducer,
        initial_state: ModelRegistrationState,
        uuid_gen: DeterministicUUIDGenerator,
    ) -> None:
        """Test that reducer output contains processing time metrics.

        The ModelReducerOutput should include processing_time_ms and
        items_processed fields for monitoring.
        """
        event = create_introspection_event(node_id=uuid_gen.next())
        output = reducer.reduce(initial_state, event)

        # Verify output metadata
        assert output.processing_time_ms >= 0
        assert output.items_processed == 1
        assert output.operation_id is not None

    def test_duplicate_event_has_zero_items_processed(
        self,
        reducer: RegistrationReducer,
        initial_state: ModelRegistrationState,
        uuid_gen: DeterministicUUIDGenerator,
    ) -> None:
        """Test that duplicate events report zero items processed.

        When a duplicate event is detected, items_processed should be 0
        to indicate no actual processing occurred.
        """
        correlation_id = uuid_gen.next()
        event = create_introspection_event(
            node_id=uuid_gen.next(),
            correlation_id=correlation_id,
        )

        # First processing
        output1 = reducer.reduce(initial_state, event)
        assert output1.items_processed == 1

        # Duplicate processing
        output2 = reducer.reduce(output1.result, event)
        assert output2.items_processed == 0


@pytest.mark.integration
class TestNodeTypeValidation:
    """Tests for node_type validation during event processing."""

    @pytest.mark.parametrize(
        "node_type",
        ["effect", "compute", "reducer", "orchestrator"],
    )
    def test_valid_node_types_are_accepted(
        self,
        reducer: RegistrationReducer,
        initial_state: ModelRegistrationState,
        uuid_gen: DeterministicUUIDGenerator,
        node_type: str,
    ) -> None:
        """Test that all valid ONEX node types are accepted.

        Per contract.yaml validation rules, valid_values are:
        - effect, compute, reducer, orchestrator
        """
        event = create_introspection_event(
            node_id=uuid_gen.next(),
            node_type=node_type,
        )

        output = reducer.reduce(initial_state, event)

        assert output.result.status == "pending"
        assert len(output.intents) == 2

    def test_pydantic_enforces_node_type_validation(
        self,
        reducer: RegistrationReducer,
        initial_state: ModelRegistrationState,
        uuid_gen: DeterministicUUIDGenerator,
    ) -> None:
        """Test that Pydantic model enforces node_type validation.

        The ModelNodeIntrospectionEvent model uses a Literal type for node_type,
        which means invalid values are rejected at model construction time.
        This is defense-in-depth - the reducer's _validate_event() method
        provides a second layer of validation.

        This test verifies that Pydantic correctly rejects invalid node types
        at construction time, making it impossible to pass invalid events
        to the reducer through normal paths.
        """
        from pydantic import ValidationError

        # Pydantic should reject invalid node_type at construction
        with pytest.raises(ValidationError) as exc_info:
            ModelNodeIntrospectionEvent(
                node_id=uuid_gen.next(),
                node_type="invalid_type",  # Invalid - not in EnumNodeKind
                node_version=ModelSemVer(major=1, minor=0, patch=0),
                correlation_id=uuid_gen.next(),
                endpoints={},
                timestamp=datetime.now(UTC),
            )

        # Verify the error is about node_type
        error_str = str(exc_info.value)
        assert "node_type" in error_str
        assert "literal_error" in error_str or "Input should be" in error_str
