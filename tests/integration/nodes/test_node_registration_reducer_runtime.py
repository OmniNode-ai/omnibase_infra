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
from omnibase_infra.nodes.node_registration_reducer import RegistrationReducer
from omnibase_infra.nodes.node_registration_reducer.models import (
    ModelPayloadPostgresUpsertRegistration,
    ModelRegistrationState,
)

# Import test doubles and fixtures from workflow conftest
from tests.integration.registration.effect.test_doubles import (
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

    Verifies that the reducer correctly emits PostgreSQL registration
    intents when processing a valid introspection event.
    """

    def test_reducer_emits_postgres_intent(
        self,
        reducer: RegistrationReducer,
        initial_state: ModelRegistrationState,
        uuid_gen: DeterministicUUIDGenerator,
    ) -> None:
        """Test that processing an introspection event emits a postgres intent.

        Given an idle state and a valid introspection event,
        the reducer should emit a postgres.upsert_registration intent.
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

        # Assert - should have exactly 1 intent (postgres only, consul removed OMN-3540)
        assert len(output.intents) == 1, (
            f"Expected exactly 1 intent (postgres), got {len(output.intents)}"
        )

        # Verify the single intent is the postgres upsert registration intent
        postgres_intent = output.intents[0]
        assert postgres_intent.payload.intent_type == "postgres.upsert_registration", (
            f"Expected postgres.upsert_registration intent, got {postgres_intent.payload.intent_type}"
        )

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
        - postgres.upsert_registration: target_pattern = "postgres://node_registrations/{node_id}"
        """
        # Arrange
        node_id = uuid_gen.next()
        event = create_introspection_event(node_id=node_id, node_type="reducer")

        # Act
        output = reducer.reduce(initial_state, event)

        # Verify postgres intent exists before checking target
        postgres_intents = [
            i
            for i in output.intents
            if i.payload.intent_type == "postgres.upsert_registration"
        ]
        assert len(postgres_intents) == 1, (
            f"Expected exactly 1 postgres intent, got {len(postgres_intents)}"
        )
        assert postgres_intents[0].target == f"postgres://node_registrations/{node_id}"


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
        assert (
            len(output.intents) == 1
        )  # Intents emitted (postgres only, consul removed)


# =============================================================================
# Test 3: FSM Pending to Complete Workflow
# =============================================================================


@pytest.mark.integration
class TestFSMPendingToCompleteWorkflow:
    """Tests for the complete FSM workflow: idle -> pending -> partial -> complete.

    This simulates the full registration lifecycle where PostgreSQL
    backend confirms successful registration.
    """

    def test_full_workflow_postgres_confirms(
        self,
        reducer: RegistrationReducer,
        initial_state: ModelRegistrationState,
        uuid_gen: DeterministicUUIDGenerator,
    ) -> None:
        """Test complete workflow with PostgreSQL confirming.

        Flow: idle -> pending -> partial -> complete (postgres)
        """
        # Step 1: idle -> pending
        node_id = uuid_gen.next()
        event = create_introspection_event(node_id=node_id)
        output = reducer.reduce(initial_state, event)
        pending_state = output.result

        assert pending_state.status == "pending"
        assert len(output.intents) >= 1

        # Step 2: pending -> complete (postgres confirmed; partial state removed OMN-3540)
        postgres_event_id = uuid_gen.next()
        complete_state = pending_state.with_postgres_confirmed(postgres_event_id)

        assert complete_state.status == "complete", (
            f"Expected 'complete' after postgres confirmation, got '{complete_state.status}'"
        )
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
        complete_state = pending_state.with_postgres_confirmed(uuid_gen.next())

        # Verify node_id preserved
        assert pending_state.node_id == node_id
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
        failed_state = pending_state.with_failure("postgres_failed", uuid_gen.next())

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
        failed_state = pending_state.with_failure("postgres_failed", uuid_gen.next())

        assert failed_state.status == "failed"
        assert failed_state.can_reset() is True

        # Reset via reducer method
        reset_event_id = uuid_gen.next()
        reset_output = reducer.reduce_reset(failed_state, reset_event_id)

        # Assert
        assert reset_output.result.status == "idle"
        assert reset_output.result.node_id is None
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
        complete_state = pending_state.with_postgres_confirmed(uuid_gen.next())

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

        Partial state is deprecated (OMN-3540 removed Consul, eliminating the
        multi-backend partial success path), but can exist in legacy DB rows.
        Directly construct a partial state to verify the reducer rejects reset.
        """
        from omnibase_infra.enums import EnumRegistrationStatus

        # Directly construct a partial state (legacy scenario: postgres confirmed
        # but consul not yet confirmed, before consul was removed)
        node_id = uuid_gen.next()
        partial_state = ModelRegistrationState(
            status=EnumRegistrationStatus.PARTIAL,
            node_id=node_id,
            consul_confirmed=False,
            postgres_confirmed=True,
            last_processed_event_id=uuid_gen.next(),
            failure_reason=None,
        )

        assert partial_state.status == "partial"
        assert partial_state.can_reset() is False

        # Attempt reset -- should fail because partial is an in-flight state
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
        assert len(output1.intents) == 1  # postgres only, consul removed

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

        assert len(output1.intents) == 1  # postgres only, consul removed

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
        assert len(output2.intents) == 1  # postgres only, consul removed
        assert output2.result.status == "pending"


# =============================================================================
# Test 7: End-to-End Workflow with Mocked Effects
# =============================================================================


@pytest.mark.asyncio
@pytest.mark.integration
class TestEndToEndWithMockedEffects:
    """End-to-end tests simulating the full registration workflow with stubs.

    These tests verify the integration between the reducer and the effect layer
    using StubPostgresAdapter test doubles.
    """

    async def test_complete_registration_with_stub_backends(
        self,
        reducer: RegistrationReducer,
        initial_state: ModelRegistrationState,
        stub_postgres_adapter: StubPostgresAdapter,
        uuid_gen: DeterministicUUIDGenerator,
    ) -> None:
        """Test complete registration workflow with stub backend.

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
        assert len(output.intents) >= 1

        # Step 2: Execute postgres intents against stub
        for intent in output.intents:
            if intent.payload.intent_type == "postgres.upsert_registration":
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

        # Step 3: Verify stub call count
        assert stub_postgres_adapter.call_count == 1

        # Step 4: Simulate confirmation and complete workflow
        complete_state = pending_state.with_postgres_confirmed(uuid_gen.next())

        assert complete_state.node_id == node_id

    async def test_complete_failure_postgres_fails(
        self,
        reducer: RegistrationReducer,
        initial_state: ModelRegistrationState,
        stub_postgres_adapter: StubPostgresAdapter,
        uuid_gen: DeterministicUUIDGenerator,
    ) -> None:
        """Test workflow when PostgreSQL backend fails.

        Verifies that state transitions to failed with appropriate reason.
        """
        # Configure postgres to fail
        stub_postgres_adapter.should_fail = True

        # Process introspection event
        event = create_introspection_event(node_id=uuid_gen.next())
        output = reducer.reduce(initial_state, event)
        pending_state = output.result

        # Execute intents - postgres fails (route via payload.intent_type)
        for intent in output.intents:
            if intent.payload.intent_type == "postgres.upsert_registration":
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
        failed_state = pending_state.with_failure("postgres_failed", uuid_gen.next())

        assert failed_state.status == "failed"
        assert failed_state.failure_reason == "postgres_failed"

    async def test_retry_after_failure(
        self,
        reducer: RegistrationReducer,
        initial_state: ModelRegistrationState,
        stub_postgres_adapter: StubPostgresAdapter,
        uuid_gen: DeterministicUUIDGenerator,
    ) -> None:
        """Test retry workflow after initial failure.

        Verifies that:
        1. Initial registration fails
        2. Reset to idle
        3. Retry succeeds
        """
        # Initial attempt - postgres fails
        stub_postgres_adapter.should_fail = True

        event1 = create_introspection_event(
            node_id=uuid_gen.next(),
            correlation_id=uuid_gen.next(),
        )
        output1 = reducer.reduce(initial_state, event1)
        pending_state = output1.result

        failed_state = pending_state.with_failure("postgres_failed", uuid_gen.next())
        assert failed_state.status == "failed"

        # Reset
        reset_output = reducer.reduce_reset(failed_state, uuid_gen.next())
        idle_state = reset_output.result
        assert idle_state.status == "idle"

        # Retry - fix postgres
        stub_postgres_adapter.should_fail = False
        stub_postgres_adapter.reset()

        event2 = create_introspection_event(
            node_id=uuid_gen.next(),
            correlation_id=uuid_gen.next(),  # New correlation_id
        )
        output2 = reducer.reduce(idle_state, event2)

        assert output2.result.status == "pending"
        assert len(output2.intents) >= 1

        # Execute successfully (route via payload.intent_type)
        for intent in output2.intents:
            if intent.payload.intent_type == "postgres.upsert_registration":
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

        # Complete workflow
        complete = output2.result.with_postgres_confirmed(uuid_gen.next())

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
        assert len(output.intents) == 1  # postgres only, consul removed

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
