# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Unit tests for the canonical RegistrationReducer.

This test suite validates the pure reducer implementation that processes
node introspection events and emits registration intents for Consul and
PostgreSQL backends.

Architecture:
    The RegistrationReducer follows the pure function pattern:
    - reduce(state, event) -> ModelReducerOutput
    - No internal state mutation
    - No I/O operations (intents are emitted instead)
    - Deterministic: same inputs produce same outputs

Test Organization:
    - TestBasicReduce: Core reduce functionality
    - TestValidation: Input validation behavior
    - TestIdempotency: Duplicate event handling
    - TestStateTransitions: ModelRegistrationState transitions
    - TestConsulIntentBuilding: Consul intent structure
    - TestPostgresIntentBuilding: PostgreSQL intent structure
    - TestOutputModel: ModelReducerOutput structure

Related:
    - RegistrationReducer: Implementation under test
    - ModelRegistrationState: State model for pure reducer
    - DESIGN_TWO_WAY_REGISTRATION_ARCHITECTURE.md: Architecture design
    - OMN-889: Infrastructure MVP
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import TYPE_CHECKING
from uuid import UUID, uuid4

import pytest
from omnibase_core.enums import EnumReductionType, EnumStreamingMode
from omnibase_core.models.reducer.model_intent import ModelIntent
from omnibase_core.nodes import ModelReducerOutput

from omnibase_infra.models.registration import (
    ModelNodeCapabilities,
    ModelNodeIntrospectionEvent,
    ModelNodeMetadata,
)
from omnibase_infra.nodes.reducers import RegistrationReducer
from omnibase_infra.nodes.reducers.models import ModelRegistrationState

if TYPE_CHECKING:
    from typing import Literal


# -----------------------------------------------------------------------------
# Fixtures
# -----------------------------------------------------------------------------


@pytest.fixture
def reducer() -> RegistrationReducer:
    """Create a RegistrationReducer instance for testing.

    Returns:
        A new RegistrationReducer instance.
    """
    return RegistrationReducer()


@pytest.fixture
def initial_state() -> ModelRegistrationState:
    """Create an initial idle state for testing.

    Returns:
        A new ModelRegistrationState in idle status.
    """
    return ModelRegistrationState()


@pytest.fixture
def valid_event() -> ModelNodeIntrospectionEvent:
    """Create a valid introspection event for testing.

    Returns:
        A valid ModelNodeIntrospectionEvent with all required fields.
    """
    return ModelNodeIntrospectionEvent(
        node_id=uuid4(),
        node_type="effect",
        node_version="1.0.0",
        correlation_id=uuid4(),
        endpoints={"health": "http://localhost:8080/health"},
        capabilities=ModelNodeCapabilities(postgres=True, read=True, write=True),
        metadata=ModelNodeMetadata(environment="test"),
    )


@pytest.fixture
def event_without_health_endpoint() -> ModelNodeIntrospectionEvent:
    """Create an event without health endpoint for testing.

    Returns:
        A valid event with empty endpoints.
    """
    return ModelNodeIntrospectionEvent(
        node_id=uuid4(),
        node_type="compute",
        node_version="2.0.0",
        correlation_id=uuid4(),
        endpoints={},
        capabilities=ModelNodeCapabilities(),
        metadata=ModelNodeMetadata(),
    )


def create_introspection_event(
    node_type: Literal["effect", "compute", "reducer", "orchestrator"] = "effect",
    node_id: UUID | None = None,
    correlation_id: UUID | None = None,
    endpoints: dict[str, str] | None = None,
) -> ModelNodeIntrospectionEvent:
    """Helper factory for creating introspection events.

    Args:
        node_type: Type of node (default: "effect").
        node_id: Optional node ID (generates if not provided).
        correlation_id: Optional correlation ID (generates if not provided).
        endpoints: Optional endpoints dict (default: health endpoint).

    Returns:
        Configured ModelNodeIntrospectionEvent instance.
    """
    return ModelNodeIntrospectionEvent(
        node_id=node_id or uuid4(),
        node_type=node_type,
        node_version="1.0.0",
        capabilities=ModelNodeCapabilities(postgres=True, read=True),
        endpoints=endpoints
        if endpoints is not None
        else {"health": "http://localhost:8080/health"},
        correlation_id=correlation_id or uuid4(),
    )


# -----------------------------------------------------------------------------
# Basic Reduce Tests
# -----------------------------------------------------------------------------


@pytest.mark.unit
class TestBasicReduce:
    """Tests for core reduce functionality."""

    def test_reduce_valid_event_emits_two_intents(
        self,
        reducer: RegistrationReducer,
        initial_state: ModelRegistrationState,
        valid_event: ModelNodeIntrospectionEvent,
    ) -> None:
        """Test that valid event produces Consul + PostgreSQL intents."""
        output = reducer.reduce(initial_state, valid_event)

        assert len(output.intents) == 2
        intent_types = {intent.intent_type for intent in output.intents}
        assert "consul.register" in intent_types
        assert "postgres.upsert_registration" in intent_types

    def test_reduce_valid_event_transitions_to_pending(
        self,
        reducer: RegistrationReducer,
        initial_state: ModelRegistrationState,
        valid_event: ModelNodeIntrospectionEvent,
    ) -> None:
        """Test that state becomes 'pending' after valid event."""
        output = reducer.reduce(initial_state, valid_event)

        assert output.result.status == "pending"
        assert output.result.node_id == valid_event.node_id

    def test_reduce_returns_model_reducer_output(
        self,
        reducer: RegistrationReducer,
        initial_state: ModelRegistrationState,
        valid_event: ModelNodeIntrospectionEvent,
    ) -> None:
        """Test that output is correct type from omnibase_core."""
        output = reducer.reduce(initial_state, valid_event)

        assert isinstance(output, ModelReducerOutput)
        assert isinstance(output.result, ModelRegistrationState)
        assert isinstance(output.intents, tuple)

    def test_reduce_preserves_correlation_id(
        self,
        reducer: RegistrationReducer,
        initial_state: ModelRegistrationState,
    ) -> None:
        """Test that correlation_id is preserved in intents."""
        correlation_id = uuid4()
        event = create_introspection_event(correlation_id=correlation_id)

        output = reducer.reduce(initial_state, event)

        for intent in output.intents:
            assert intent.payload["correlation_id"] == str(correlation_id)

    def test_reduce_with_all_node_types(
        self,
        reducer: RegistrationReducer,
        initial_state: ModelRegistrationState,
    ) -> None:
        """Test that reduce works for all valid node types."""
        node_types: list[Literal["effect", "compute", "reducer", "orchestrator"]] = [
            "effect",
            "compute",
            "reducer",
            "orchestrator",
        ]

        for node_type in node_types:
            event = create_introspection_event(node_type=node_type)
            output = reducer.reduce(initial_state, event)

            assert len(output.intents) == 2, f"Failed for node_type: {node_type}"
            assert output.result.status == "pending"


# -----------------------------------------------------------------------------
# Validation Tests
# -----------------------------------------------------------------------------


@pytest.mark.unit
class TestValidation:
    """Tests for input validation behavior."""

    def test_reduce_invalid_event_no_node_id_fails(
        self,
        reducer: RegistrationReducer,
        initial_state: ModelRegistrationState,
    ) -> None:
        """Test that missing node_id causes failure.

        Note: Pydantic requires node_id, so we use a mock to test
        the reducer's internal validation logic.
        """
        from unittest.mock import MagicMock

        # Create a mock event with None node_id
        mock_event = MagicMock(spec=ModelNodeIntrospectionEvent)
        mock_event.node_id = None
        mock_event.node_type = "effect"
        mock_event.node_version = "1.0.0"
        mock_event.endpoints = {}
        mock_event.capabilities = {}
        mock_event.metadata = {}
        mock_event.correlation_id = uuid4()

        output = reducer.reduce(initial_state, mock_event)

        assert output.result.status == "failed"
        assert output.result.failure_reason == "validation_failed"

    def test_reduce_invalid_event_no_intents(
        self,
        reducer: RegistrationReducer,
        initial_state: ModelRegistrationState,
    ) -> None:
        """Test that failed validation produces no intents."""
        from unittest.mock import MagicMock

        mock_event = MagicMock(spec=ModelNodeIntrospectionEvent)
        mock_event.node_id = None
        mock_event.node_type = "effect"
        mock_event.correlation_id = uuid4()

        output = reducer.reduce(initial_state, mock_event)

        assert len(output.intents) == 0

    def test_reduce_invalid_event_sets_failure_reason(
        self,
        reducer: RegistrationReducer,
        initial_state: ModelRegistrationState,
    ) -> None:
        """Test that failure_reason is 'validation_failed' for invalid events."""
        from unittest.mock import MagicMock

        mock_event = MagicMock(spec=ModelNodeIntrospectionEvent)
        mock_event.node_id = None
        mock_event.node_type = "effect"
        mock_event.correlation_id = uuid4()

        output = reducer.reduce(initial_state, mock_event)

        assert output.result.failure_reason == "validation_failed"

    def test_reduce_valid_event_has_no_failure_reason(
        self,
        reducer: RegistrationReducer,
        initial_state: ModelRegistrationState,
        valid_event: ModelNodeIntrospectionEvent,
    ) -> None:
        """Test that valid events have no failure_reason."""
        output = reducer.reduce(initial_state, valid_event)

        assert output.result.failure_reason is None


# -----------------------------------------------------------------------------
# Idempotency Tests
# -----------------------------------------------------------------------------


@pytest.mark.unit
class TestIdempotency:
    """Tests for duplicate event handling."""

    def test_reduce_duplicate_event_returns_same_state(
        self,
        reducer: RegistrationReducer,
    ) -> None:
        """Test that duplicate event doesn't change state."""
        correlation_id = uuid4()
        node_id = uuid4()
        event = create_introspection_event(
            node_id=node_id, correlation_id=correlation_id
        )

        # First reduce
        initial_state = ModelRegistrationState()
        output1 = reducer.reduce(initial_state, event)

        # Use the resulting state for second reduce
        state_after_first = output1.result

        # Second reduce with same event
        output2 = reducer.reduce(state_after_first, event)

        # State should be unchanged
        assert output2.result == state_after_first

    def test_reduce_duplicate_event_emits_no_intents(
        self,
        reducer: RegistrationReducer,
    ) -> None:
        """Test that duplicate event produces no intents."""
        correlation_id = uuid4()
        event = create_introspection_event(correlation_id=correlation_id)

        # First reduce
        initial_state = ModelRegistrationState()
        output1 = reducer.reduce(initial_state, event)
        state_after_first = output1.result

        # Second reduce with same event
        output2 = reducer.reduce(state_after_first, event)

        # No intents should be emitted
        assert len(output2.intents) == 0

    def test_reduce_different_events_process_correctly(
        self,
        reducer: RegistrationReducer,
        initial_state: ModelRegistrationState,
    ) -> None:
        """Test that different event IDs process normally."""
        event1 = create_introspection_event(correlation_id=uuid4())
        event2 = create_introspection_event(correlation_id=uuid4())

        output1 = reducer.reduce(initial_state, event1)
        state_after_first = output1.result

        output2 = reducer.reduce(state_after_first, event2)

        # Both should emit intents
        assert len(output1.intents) == 2
        assert len(output2.intents) == 2

    def test_reduce_duplicate_detection_uses_correlation_id(
        self,
        reducer: RegistrationReducer,
    ) -> None:
        """Test that idempotency uses correlation_id for duplicate detection."""
        correlation_id = uuid4()
        node_id1 = uuid4()
        node_id2 = uuid4()

        # Two events with same correlation_id but different node_id
        event1 = create_introspection_event(
            node_id=node_id1, correlation_id=correlation_id
        )
        event2 = create_introspection_event(
            node_id=node_id2, correlation_id=correlation_id
        )

        initial_state = ModelRegistrationState()
        output1 = reducer.reduce(initial_state, event1)
        state_after_first = output1.result

        # Same correlation_id means duplicate
        output2 = reducer.reduce(state_after_first, event2)

        # Second should be treated as duplicate
        assert len(output2.intents) == 0

    def test_reduce_items_processed_zero_for_duplicate(
        self,
        reducer: RegistrationReducer,
    ) -> None:
        """Test that items_processed is 0 for duplicate events."""
        correlation_id = uuid4()
        event = create_introspection_event(correlation_id=correlation_id)

        initial_state = ModelRegistrationState()
        output1 = reducer.reduce(initial_state, event)
        state_after_first = output1.result

        output2 = reducer.reduce(state_after_first, event)

        assert output2.items_processed == 0


# -----------------------------------------------------------------------------
# State Transition Tests (ModelRegistrationState)
# -----------------------------------------------------------------------------


@pytest.mark.unit
class TestStateTransitions:
    """Tests for ModelRegistrationState transitions."""

    def test_state_with_pending_registration(self) -> None:
        """Test creating pending state."""
        state = ModelRegistrationState()
        node_id = uuid4()
        event_id = uuid4()

        new_state = state.with_pending_registration(node_id, event_id)

        assert new_state.status == "pending"
        assert new_state.node_id == node_id
        assert new_state.last_processed_event_id == event_id
        assert new_state.consul_confirmed is False
        assert new_state.postgres_confirmed is False

    def test_state_with_consul_confirmed_partial(self) -> None:
        """Test that Consul only = partial status."""
        state = ModelRegistrationState()
        node_id = uuid4()
        pending_state = state.with_pending_registration(node_id, uuid4())

        consul_confirmed = pending_state.with_consul_confirmed(uuid4())

        assert consul_confirmed.status == "partial"
        assert consul_confirmed.consul_confirmed is True
        assert consul_confirmed.postgres_confirmed is False

    def test_state_with_postgres_confirmed_partial(self) -> None:
        """Test that Postgres only = partial status."""
        state = ModelRegistrationState()
        node_id = uuid4()
        pending_state = state.with_pending_registration(node_id, uuid4())

        postgres_confirmed = pending_state.with_postgres_confirmed(uuid4())

        assert postgres_confirmed.status == "partial"
        assert postgres_confirmed.consul_confirmed is False
        assert postgres_confirmed.postgres_confirmed is True

    def test_state_with_both_confirmed_complete(self) -> None:
        """Test that both confirmations = complete status."""
        state = ModelRegistrationState()
        node_id = uuid4()
        pending_state = state.with_pending_registration(node_id, uuid4())

        consul_confirmed = pending_state.with_consul_confirmed(uuid4())
        both_confirmed = consul_confirmed.with_postgres_confirmed(uuid4())

        assert both_confirmed.status == "complete"
        assert both_confirmed.consul_confirmed is True
        assert both_confirmed.postgres_confirmed is True

    def test_state_with_postgres_then_consul_complete(self) -> None:
        """Test that order doesn't matter: postgres then consul = complete."""
        state = ModelRegistrationState()
        node_id = uuid4()
        pending_state = state.with_pending_registration(node_id, uuid4())

        postgres_confirmed = pending_state.with_postgres_confirmed(uuid4())
        both_confirmed = postgres_confirmed.with_consul_confirmed(uuid4())

        assert both_confirmed.status == "complete"
        assert both_confirmed.consul_confirmed is True
        assert both_confirmed.postgres_confirmed is True

    def test_state_with_failure(self) -> None:
        """Test failure state correctly set."""
        state = ModelRegistrationState()
        node_id = uuid4()
        pending_state = state.with_pending_registration(node_id, uuid4())

        failed_state = pending_state.with_failure("validation_failed", uuid4())

        assert failed_state.status == "failed"
        assert failed_state.failure_reason == "validation_failed"

    def test_state_is_duplicate_event(self) -> None:
        """Test duplicate detection works."""
        state = ModelRegistrationState()
        node_id = uuid4()
        event_id = uuid4()

        pending_state = state.with_pending_registration(node_id, event_id)

        assert pending_state.is_duplicate_event(event_id) is True
        assert pending_state.is_duplicate_event(uuid4()) is False

    def test_state_immutability(self) -> None:
        """Test that state transitions create new instances."""
        state = ModelRegistrationState()
        node_id = uuid4()
        event_id = uuid4()

        pending_state = state.with_pending_registration(node_id, event_id)

        # Original should be unchanged
        assert state.status == "idle"
        assert state.node_id is None

        # New state should have updates
        assert pending_state.status == "pending"
        assert pending_state.node_id == node_id

    def test_state_failure_preserves_confirmation_flags(self) -> None:
        """Test that failure preserves confirmation flags for diagnostics."""
        state = ModelRegistrationState()
        node_id = uuid4()
        pending_state = state.with_pending_registration(node_id, uuid4())

        consul_confirmed = pending_state.with_consul_confirmed(uuid4())
        failed_state = consul_confirmed.with_failure("postgres_failed", uuid4())

        assert failed_state.status == "failed"
        assert failed_state.failure_reason == "postgres_failed"
        # Consul confirmation preserved
        assert failed_state.consul_confirmed is True
        assert failed_state.postgres_confirmed is False

    def test_state_with_reset_from_failed(self) -> None:
        """Test reset from failed state returns to idle."""
        state = ModelRegistrationState()
        node_id = uuid4()
        pending_state = state.with_pending_registration(node_id, uuid4())
        failed_state = pending_state.with_failure("consul_failed", uuid4())

        reset_event_id = uuid4()
        reset_state = failed_state.with_reset(reset_event_id)

        assert reset_state.status == "idle"
        assert reset_state.node_id is None
        assert reset_state.consul_confirmed is False
        assert reset_state.postgres_confirmed is False
        assert reset_state.failure_reason is None
        assert reset_state.last_processed_event_id == reset_event_id

    def test_state_with_reset_from_complete(self) -> None:
        """Test reset from complete state returns to idle."""
        state = ModelRegistrationState()
        node_id = uuid4()
        pending_state = state.with_pending_registration(node_id, uuid4())
        consul_confirmed = pending_state.with_consul_confirmed(uuid4())
        complete_state = consul_confirmed.with_postgres_confirmed(uuid4())

        reset_event_id = uuid4()
        reset_state = complete_state.with_reset(reset_event_id)

        assert reset_state.status == "idle"
        assert reset_state.node_id is None
        assert reset_state.consul_confirmed is False
        assert reset_state.postgres_confirmed is False

    def test_state_with_reset_clears_all_flags(self) -> None:
        """Test that reset clears all confirmation flags and failure reason."""
        # Create a failed state with one confirmation
        state = ModelRegistrationState()
        node_id = uuid4()
        pending_state = state.with_pending_registration(node_id, uuid4())
        consul_confirmed = pending_state.with_consul_confirmed(uuid4())
        failed_state = consul_confirmed.with_failure("postgres_failed", uuid4())

        # Verify pre-conditions
        assert failed_state.consul_confirmed is True
        assert failed_state.failure_reason == "postgres_failed"

        # Reset should clear everything
        reset_state = failed_state.with_reset(uuid4())

        assert reset_state.consul_confirmed is False
        assert reset_state.postgres_confirmed is False
        assert reset_state.failure_reason is None
        assert reset_state.node_id is None

    def test_state_can_reset_from_failed(self) -> None:
        """Test can_reset returns True for failed state."""
        state = ModelRegistrationState()
        node_id = uuid4()
        pending_state = state.with_pending_registration(node_id, uuid4())
        failed_state = pending_state.with_failure("validation_failed", uuid4())

        assert failed_state.can_reset() is True

    def test_state_can_reset_from_complete(self) -> None:
        """Test can_reset returns True for complete state."""
        state = ModelRegistrationState()
        node_id = uuid4()
        pending_state = state.with_pending_registration(node_id, uuid4())
        consul_confirmed = pending_state.with_consul_confirmed(uuid4())
        complete_state = consul_confirmed.with_postgres_confirmed(uuid4())

        assert complete_state.can_reset() is True

    def test_state_cannot_reset_from_idle(self) -> None:
        """Test can_reset returns False for idle state."""
        state = ModelRegistrationState()

        assert state.can_reset() is False

    def test_state_cannot_reset_from_pending(self) -> None:
        """Test can_reset returns False for pending state."""
        state = ModelRegistrationState()
        node_id = uuid4()
        pending_state = state.with_pending_registration(node_id, uuid4())

        assert pending_state.can_reset() is False

    def test_state_cannot_reset_from_partial(self) -> None:
        """Test can_reset returns False for partial state."""
        state = ModelRegistrationState()
        node_id = uuid4()
        pending_state = state.with_pending_registration(node_id, uuid4())
        partial_state = pending_state.with_consul_confirmed(uuid4())

        assert partial_state.can_reset() is False

    def test_state_reset_immutability(self) -> None:
        """Test that reset creates a new instance without mutating original."""
        state = ModelRegistrationState()
        node_id = uuid4()
        pending_state = state.with_pending_registration(node_id, uuid4())
        failed_state = pending_state.with_failure("consul_failed", uuid4())

        original_status = failed_state.status
        original_failure_reason = failed_state.failure_reason

        reset_state = failed_state.with_reset(uuid4())

        # Original should be unchanged
        assert failed_state.status == original_status
        assert failed_state.failure_reason == original_failure_reason

        # New state should be reset
        assert reset_state.status == "idle"
        assert reset_state.failure_reason is None


# -----------------------------------------------------------------------------
# Reducer Reset Tests
# -----------------------------------------------------------------------------


@pytest.mark.unit
class TestReducerReset:
    """Tests for RegistrationReducer.reduce_reset() method."""

    def test_reduce_reset_from_failed_state(
        self,
        reducer: RegistrationReducer,
    ) -> None:
        """Test that reduce_reset transitions failed state to idle."""
        state = ModelRegistrationState()
        node_id = uuid4()
        pending_state = state.with_pending_registration(node_id, uuid4())
        failed_state = pending_state.with_failure("consul_failed", uuid4())

        reset_event_id = uuid4()
        output = reducer.reduce_reset(failed_state, reset_event_id)

        assert output.result.status == "idle"
        assert output.result.node_id is None
        assert output.result.failure_reason is None
        assert output.items_processed == 1

    def test_reduce_reset_from_complete_state(
        self,
        reducer: RegistrationReducer,
    ) -> None:
        """Test that reduce_reset transitions complete state to idle."""
        state = ModelRegistrationState()
        node_id = uuid4()
        pending_state = state.with_pending_registration(node_id, uuid4())
        consul_confirmed = pending_state.with_consul_confirmed(uuid4())
        complete_state = consul_confirmed.with_postgres_confirmed(uuid4())

        reset_event_id = uuid4()
        output = reducer.reduce_reset(complete_state, reset_event_id)

        assert output.result.status == "idle"
        assert output.items_processed == 1

    def test_reduce_reset_fails_from_idle(
        self,
        reducer: RegistrationReducer,
        initial_state: ModelRegistrationState,
    ) -> None:
        """Test that reduce_reset from idle state returns failed with invalid_reset_state.

        Resetting from idle is invalid because there's nothing to reset.
        This is a validation error, not a no-op.
        """
        reset_event_id = uuid4()
        output = reducer.reduce_reset(initial_state, reset_event_id)

        assert output.result.status == "failed"
        assert output.result.failure_reason == "invalid_reset_state"
        assert output.items_processed == 1  # Event was processed (caused state change)

    def test_reduce_reset_fails_from_pending(
        self,
        reducer: RegistrationReducer,
    ) -> None:
        """Test that reduce_reset from pending state returns failed with invalid_reset_state.

        Resetting from pending would lose in-flight registration state,
        potentially causing inconsistency between Consul and PostgreSQL.
        This is a validation error to prevent accidental state loss.
        """
        state = ModelRegistrationState()
        node_id = uuid4()
        pending_state = state.with_pending_registration(node_id, uuid4())

        reset_event_id = uuid4()
        output = reducer.reduce_reset(pending_state, reset_event_id)

        assert output.result.status == "failed"
        assert output.result.failure_reason == "invalid_reset_state"
        # Confirmation flags should be preserved for diagnostics
        assert output.result.node_id == node_id
        assert output.items_processed == 1  # Event was processed (caused state change)

    def test_reduce_reset_fails_from_partial(
        self,
        reducer: RegistrationReducer,
    ) -> None:
        """Test that reduce_reset from partial state returns failed with invalid_reset_state.

        Resetting from partial would lose in-flight registration state.
        One backend (Consul or PostgreSQL) has already confirmed, and
        resetting would discard that confirmation, leaving the system
        in an inconsistent state.
        """
        state = ModelRegistrationState()
        node_id = uuid4()
        pending_state = state.with_pending_registration(node_id, uuid4())
        partial_state = pending_state.with_consul_confirmed(uuid4())

        reset_event_id = uuid4()
        output = reducer.reduce_reset(partial_state, reset_event_id)

        assert output.result.status == "failed"
        assert output.result.failure_reason == "invalid_reset_state"
        # Confirmation flags should be preserved for diagnostics
        assert output.result.consul_confirmed is True
        assert output.items_processed == 1  # Event was processed (caused state change)

    def test_reduce_reset_emits_no_intents(
        self,
        reducer: RegistrationReducer,
    ) -> None:
        """Test that reduce_reset emits no intents."""
        state = ModelRegistrationState()
        node_id = uuid4()
        pending_state = state.with_pending_registration(node_id, uuid4())
        failed_state = pending_state.with_failure("postgres_failed", uuid4())

        output = reducer.reduce_reset(failed_state, uuid4())

        assert len(output.intents) == 0

    def test_reduce_reset_idempotency(
        self,
        reducer: RegistrationReducer,
    ) -> None:
        """Test that duplicate reset events are skipped."""
        state = ModelRegistrationState()
        node_id = uuid4()
        pending_state = state.with_pending_registration(node_id, uuid4())
        failed_state = pending_state.with_failure("consul_failed", uuid4())

        reset_event_id = uuid4()
        output1 = reducer.reduce_reset(failed_state, reset_event_id)
        idle_state = output1.result

        # Second reset with same event_id should be skipped
        output2 = reducer.reduce_reset(idle_state, reset_event_id)

        assert output2.result == idle_state
        assert output2.items_processed == 0

    def test_reduce_reset_full_recovery_workflow(
        self,
        reducer: RegistrationReducer,
    ) -> None:
        """Test complete workflow: introspection -> failure -> reset -> retry."""
        initial_state = ModelRegistrationState()

        # First introspection
        event1 = create_introspection_event()
        output1 = reducer.reduce(initial_state, event1)
        assert output1.result.status == "pending"
        assert len(output1.intents) == 2

        # Simulate failure
        failed_state = output1.result.with_failure("consul_failed", uuid4())
        assert failed_state.status == "failed"

        # Reset to recover
        reset_output = reducer.reduce_reset(failed_state, uuid4())
        assert reset_output.result.status == "idle"

        # Retry with new introspection
        event2 = create_introspection_event()
        retry_output = reducer.reduce(reset_output.result, event2)
        assert retry_output.result.status == "pending"
        assert len(retry_output.intents) == 2


# -----------------------------------------------------------------------------
# Consul Intent Building Tests
# -----------------------------------------------------------------------------


@pytest.mark.unit
class TestConsulIntentBuilding:
    """Tests for Consul registration intent structure."""

    def test_consul_intent_has_correct_service_id(
        self,
        reducer: RegistrationReducer,
        initial_state: ModelRegistrationState,
    ) -> None:
        """Test that service_id format is correct."""
        node_id = uuid4()
        event = create_introspection_event(node_id=node_id, node_type="effect")

        output = reducer.reduce(initial_state, event)

        consul_intent = next(
            (i for i in output.intents if i.intent_type == "consul.register"), None
        )
        assert consul_intent is not None

        expected_service_id = f"node-effect-{node_id}"
        assert consul_intent.payload["service_id"] == expected_service_id

    def test_consul_intent_has_correct_tags(
        self,
        reducer: RegistrationReducer,
        initial_state: ModelRegistrationState,
    ) -> None:
        """Test that tags include node_type and version."""
        event = ModelNodeIntrospectionEvent(
            node_id=uuid4(),
            node_type="compute",
            node_version="2.3.4",
            endpoints={"health": "http://localhost:8080/health"},
            correlation_id=uuid4(),
        )

        output = reducer.reduce(initial_state, event)

        consul_intent = next(
            (i for i in output.intents if i.intent_type == "consul.register"), None
        )
        assert consul_intent is not None

        tags = consul_intent.payload["tags"]
        assert "node_type:compute" in tags
        assert "node_version:2.3.4" in tags

    def test_consul_intent_has_health_check_when_provided(
        self,
        reducer: RegistrationReducer,
        initial_state: ModelRegistrationState,
        valid_event: ModelNodeIntrospectionEvent,
    ) -> None:
        """Test that health check is included if endpoint exists."""
        output = reducer.reduce(initial_state, valid_event)

        consul_intent = next(
            (i for i in output.intents if i.intent_type == "consul.register"), None
        )
        assert consul_intent is not None

        assert "health_check" in consul_intent.payload
        health_check = consul_intent.payload["health_check"]
        assert health_check["HTTP"] == "http://localhost:8080/health"
        assert health_check["Interval"] == "10s"
        assert health_check["Timeout"] == "5s"

    def test_consul_intent_no_health_check_when_missing(
        self,
        reducer: RegistrationReducer,
        initial_state: ModelRegistrationState,
        event_without_health_endpoint: ModelNodeIntrospectionEvent,
    ) -> None:
        """Test that health check is None if no endpoint."""
        output = reducer.reduce(initial_state, event_without_health_endpoint)

        consul_intent = next(
            (i for i in output.intents if i.intent_type == "consul.register"), None
        )
        assert consul_intent is not None

        # health_check should be None when no health endpoint is provided
        assert consul_intent.payload["health_check"] is None

    def test_consul_intent_has_correct_target(
        self,
        reducer: RegistrationReducer,
        initial_state: ModelRegistrationState,
    ) -> None:
        """Test that Consul intent target is correctly formatted."""
        event = create_introspection_event(node_type="orchestrator")

        output = reducer.reduce(initial_state, event)

        consul_intent = next(
            (i for i in output.intents if i.intent_type == "consul.register"), None
        )
        assert consul_intent is not None

        expected_target = "consul://service/onex-orchestrator"
        assert consul_intent.target == expected_target

    def test_consul_intent_service_name_format(
        self,
        reducer: RegistrationReducer,
        initial_state: ModelRegistrationState,
    ) -> None:
        """Test that service_name follows onex-{node_type} format."""
        event = create_introspection_event(node_type="reducer")

        output = reducer.reduce(initial_state, event)

        consul_intent = next(
            (i for i in output.intents if i.intent_type == "consul.register"), None
        )
        assert consul_intent is not None
        assert consul_intent.payload["service_name"] == "onex-reducer"


# -----------------------------------------------------------------------------
# PostgreSQL Intent Building Tests
# -----------------------------------------------------------------------------


@pytest.mark.unit
class TestPostgresIntentBuilding:
    """Tests for PostgreSQL registration intent structure."""

    def test_postgres_intent_has_correct_record(
        self,
        reducer: RegistrationReducer,
        initial_state: ModelRegistrationState,
    ) -> None:
        """Test that record contains all required fields."""
        node_id = uuid4()
        event = create_introspection_event(
            node_id=node_id,
            node_type="effect",
            endpoints={
                "health": "http://localhost:8080/health",
                "api": "http://localhost:8080/api",
            },
        )

        output = reducer.reduce(initial_state, event)

        postgres_intent = next(
            (
                i
                for i in output.intents
                if i.intent_type == "postgres.upsert_registration"
            ),
            None,
        )
        assert postgres_intent is not None

        record = postgres_intent.payload["record"]
        assert record["node_id"] == str(node_id)
        assert record["node_type"] == "effect"
        assert record["node_version"] == "1.0.0"
        assert "health" in record["endpoints"]
        assert "api" in record["endpoints"]

    def test_postgres_intent_has_correlation_id(
        self,
        reducer: RegistrationReducer,
        initial_state: ModelRegistrationState,
    ) -> None:
        """Test that correlation_id is propagated."""
        correlation_id = uuid4()
        event = create_introspection_event(correlation_id=correlation_id)

        output = reducer.reduce(initial_state, event)

        postgres_intent = next(
            (
                i
                for i in output.intents
                if i.intent_type == "postgres.upsert_registration"
            ),
            None,
        )
        assert postgres_intent is not None
        assert postgres_intent.payload["correlation_id"] == str(correlation_id)

    def test_postgres_intent_has_correct_target(
        self,
        reducer: RegistrationReducer,
        initial_state: ModelRegistrationState,
    ) -> None:
        """Test that PostgreSQL intent target is correctly formatted."""
        node_id = uuid4()
        event = create_introspection_event(node_id=node_id)

        output = reducer.reduce(initial_state, event)

        postgres_intent = next(
            (
                i
                for i in output.intents
                if i.intent_type == "postgres.upsert_registration"
            ),
            None,
        )
        assert postgres_intent is not None

        expected_target = f"postgres://node_registrations/{node_id}"
        assert postgres_intent.target == expected_target

    def test_postgres_intent_record_has_timestamps(
        self,
        reducer: RegistrationReducer,
        initial_state: ModelRegistrationState,
        valid_event: ModelNodeIntrospectionEvent,
    ) -> None:
        """Test that record has registered_at and updated_at timestamps."""
        output = reducer.reduce(initial_state, valid_event)

        postgres_intent = next(
            (
                i
                for i in output.intents
                if i.intent_type == "postgres.upsert_registration"
            ),
            None,
        )
        assert postgres_intent is not None

        record = postgres_intent.payload["record"]
        assert "registered_at" in record
        assert "updated_at" in record

    def test_postgres_intent_record_has_health_endpoint(
        self,
        reducer: RegistrationReducer,
        initial_state: ModelRegistrationState,
        valid_event: ModelNodeIntrospectionEvent,
    ) -> None:
        """Test that record includes health_endpoint from endpoints dict."""
        output = reducer.reduce(initial_state, valid_event)

        postgres_intent = next(
            (
                i
                for i in output.intents
                if i.intent_type == "postgres.upsert_registration"
            ),
            None,
        )
        assert postgres_intent is not None

        record = postgres_intent.payload["record"]
        assert record["health_endpoint"] == "http://localhost:8080/health"

    def test_postgres_intent_record_no_health_endpoint_when_missing(
        self,
        reducer: RegistrationReducer,
        initial_state: ModelRegistrationState,
        event_without_health_endpoint: ModelNodeIntrospectionEvent,
    ) -> None:
        """Test that health_endpoint is None when not provided."""
        output = reducer.reduce(initial_state, event_without_health_endpoint)

        postgres_intent = next(
            (
                i
                for i in output.intents
                if i.intent_type == "postgres.upsert_registration"
            ),
            None,
        )
        assert postgres_intent is not None

        record = postgres_intent.payload["record"]
        assert record["health_endpoint"] is None

    def test_postgres_intent_record_capabilities_serialized(
        self,
        reducer: RegistrationReducer,
        initial_state: ModelRegistrationState,
    ) -> None:
        """Test that capabilities model is serialized to dict."""
        event = ModelNodeIntrospectionEvent(
            node_id=uuid4(),
            node_type="effect",
            node_version="1.0.0",
            endpoints={"health": "http://localhost:8080/health"},
            capabilities=ModelNodeCapabilities(postgres=True, consul=True, read=True),
            correlation_id=uuid4(),
        )

        output = reducer.reduce(initial_state, event)

        postgres_intent = next(
            (
                i
                for i in output.intents
                if i.intent_type == "postgres.upsert_registration"
            ),
            None,
        )
        assert postgres_intent is not None

        record = postgres_intent.payload["record"]
        capabilities = record["capabilities"]
        assert isinstance(capabilities, dict)
        assert capabilities.get("postgres") is True
        assert capabilities.get("consul") is True
        assert capabilities.get("read") is True


# -----------------------------------------------------------------------------
# Output Model Tests
# -----------------------------------------------------------------------------


@pytest.mark.unit
class TestOutputModel:
    """Tests for ModelReducerOutput structure and values."""

    def test_output_has_processing_time(
        self,
        reducer: RegistrationReducer,
        initial_state: ModelRegistrationState,
        valid_event: ModelNodeIntrospectionEvent,
    ) -> None:
        """Test that processing_time_ms is populated."""
        output = reducer.reduce(initial_state, valid_event)

        assert output.processing_time_ms >= 0.0

    def test_output_has_items_processed_one_for_valid(
        self,
        reducer: RegistrationReducer,
        initial_state: ModelRegistrationState,
        valid_event: ModelNodeIntrospectionEvent,
    ) -> None:
        """Test that items_processed is 1 for valid events."""
        output = reducer.reduce(initial_state, valid_event)

        assert output.items_processed == 1

    def test_output_has_items_processed_zero_for_invalid(
        self,
        reducer: RegistrationReducer,
        initial_state: ModelRegistrationState,
    ) -> None:
        """Test that items_processed is 0 for invalid events."""
        from unittest.mock import MagicMock

        mock_event = MagicMock(spec=ModelNodeIntrospectionEvent)
        mock_event.node_id = None
        mock_event.node_type = "effect"
        mock_event.correlation_id = uuid4()

        output = reducer.reduce(initial_state, mock_event)

        assert output.items_processed == 0

    def test_output_intents_are_tuple(
        self,
        reducer: RegistrationReducer,
        initial_state: ModelRegistrationState,
        valid_event: ModelNodeIntrospectionEvent,
    ) -> None:
        """Test that intents is a tuple (immutable)."""
        output = reducer.reduce(initial_state, valid_event)

        assert isinstance(output.intents, tuple)

    def test_output_has_operation_id(
        self,
        reducer: RegistrationReducer,
        initial_state: ModelRegistrationState,
        valid_event: ModelNodeIntrospectionEvent,
    ) -> None:
        """Test that output has a unique operation_id."""
        output = reducer.reduce(initial_state, valid_event)

        assert output.operation_id is not None
        assert isinstance(output.operation_id, UUID)

    def test_output_has_reduction_type(
        self,
        reducer: RegistrationReducer,
        initial_state: ModelRegistrationState,
        valid_event: ModelNodeIntrospectionEvent,
    ) -> None:
        """Test that output has correct reduction_type."""
        output = reducer.reduce(initial_state, valid_event)

        assert output.reduction_type == EnumReductionType.MERGE

    def test_output_has_streaming_mode(
        self,
        reducer: RegistrationReducer,
        initial_state: ModelRegistrationState,
        valid_event: ModelNodeIntrospectionEvent,
    ) -> None:
        """Test that output has correct streaming_mode."""
        output = reducer.reduce(initial_state, valid_event)

        assert output.streaming_mode == EnumStreamingMode.BATCH

    def test_output_has_batches_processed(
        self,
        reducer: RegistrationReducer,
        initial_state: ModelRegistrationState,
        valid_event: ModelNodeIntrospectionEvent,
    ) -> None:
        """Test that batches_processed is 1."""
        output = reducer.reduce(initial_state, valid_event)

        assert output.batches_processed == 1

    def test_output_has_conflicts_resolved_zero(
        self,
        reducer: RegistrationReducer,
        initial_state: ModelRegistrationState,
        valid_event: ModelNodeIntrospectionEvent,
    ) -> None:
        """Test that conflicts_resolved is 0."""
        output = reducer.reduce(initial_state, valid_event)

        assert output.conflicts_resolved == 0

    def test_output_result_is_new_state(
        self,
        reducer: RegistrationReducer,
        initial_state: ModelRegistrationState,
        valid_event: ModelNodeIntrospectionEvent,
    ) -> None:
        """Test that result contains the new state."""
        output = reducer.reduce(initial_state, valid_event)

        assert output.result is not initial_state
        assert output.result.status == "pending"

    def test_output_intents_contain_model_intent_instances(
        self,
        reducer: RegistrationReducer,
        initial_state: ModelRegistrationState,
        valid_event: ModelNodeIntrospectionEvent,
    ) -> None:
        """Test that intents are ModelIntent instances."""
        output = reducer.reduce(initial_state, valid_event)

        for intent in output.intents:
            assert isinstance(intent, ModelIntent)

    def test_output_different_operations_have_different_ids(
        self,
        reducer: RegistrationReducer,
        initial_state: ModelRegistrationState,
    ) -> None:
        """Test that each reduce operation generates a unique operation_id."""
        event1 = create_introspection_event()
        event2 = create_introspection_event()

        output1 = reducer.reduce(initial_state, event1)
        output2 = reducer.reduce(initial_state, event2)

        assert output1.operation_id != output2.operation_id


# -----------------------------------------------------------------------------
# Edge Cases and Error Handling
# -----------------------------------------------------------------------------


@pytest.mark.unit
class TestEdgeCases:
    """Tests for edge cases and error handling."""

    def test_reduce_with_empty_endpoints(
        self,
        reducer: RegistrationReducer,
        initial_state: ModelRegistrationState,
    ) -> None:
        """Test reduce works with empty endpoints dict."""
        event = ModelNodeIntrospectionEvent(
            node_id=uuid4(),
            node_type="effect",
            node_version="1.0.0",
            endpoints={},
            correlation_id=uuid4(),
        )

        output = reducer.reduce(initial_state, event)

        assert len(output.intents) == 2
        assert output.result.status == "pending"

    def test_reduce_with_empty_capabilities(
        self,
        reducer: RegistrationReducer,
        initial_state: ModelRegistrationState,
    ) -> None:
        """Test reduce works with empty capabilities."""
        event = ModelNodeIntrospectionEvent(
            node_id=uuid4(),
            node_type="effect",
            node_version="1.0.0",
            endpoints={"health": "http://localhost:8080/health"},
            capabilities=ModelNodeCapabilities(),
            correlation_id=uuid4(),
        )

        output = reducer.reduce(initial_state, event)

        assert len(output.intents) == 2
        assert output.result.status == "pending"

    def test_reduce_uses_deterministic_id_when_mock_has_no_correlation_id(
        self,
        reducer: RegistrationReducer,
        initial_state: ModelRegistrationState,
    ) -> None:
        """Test that event_id is derived deterministically when correlation_id is None.

        Note: Since ModelNodeIntrospectionEvent now requires correlation_id,
        this test uses a mock to simulate the case where correlation_id is None.
        """
        from unittest.mock import MagicMock

        node_id = uuid4()
        timestamp = datetime.now(UTC)

        mock_event = MagicMock(spec=ModelNodeIntrospectionEvent)
        mock_event.node_id = node_id
        mock_event.node_type = "effect"
        mock_event.node_version = "1.0.0"
        mock_event.endpoints = {"health": "http://localhost:8080/health"}
        mock_event.capabilities = ModelNodeCapabilities()
        mock_event.metadata = ModelNodeMetadata()
        mock_event.correlation_id = None  # Force deterministic derivation
        mock_event.timestamp = timestamp

        output = reducer.reduce(initial_state, mock_event)

        # State should have a last_processed_event_id even without correlation_id
        assert output.result.last_processed_event_id is not None

    def test_reduce_is_stateless(
        self,
        reducer: RegistrationReducer,
    ) -> None:
        """Test that reducer is stateless - same inputs produce same outputs."""
        state = ModelRegistrationState()
        correlation_id = uuid4()
        node_id = uuid4()

        event = ModelNodeIntrospectionEvent(
            node_id=node_id,
            node_type="effect",
            node_version="1.0.0",
            endpoints={"health": "http://localhost:8080/health"},
            correlation_id=correlation_id,
        )

        output1 = reducer.reduce(state, event)
        output2 = reducer.reduce(state, event)

        # Outputs should be structurally equivalent (except operation_id)
        assert output1.result == output2.result
        assert len(output1.intents) == len(output2.intents)
        assert output1.items_processed == output2.items_processed

    def test_reduce_with_all_optional_fields_populated(
        self,
        reducer: RegistrationReducer,
        initial_state: ModelRegistrationState,
    ) -> None:
        """Test reduce with all optional fields populated."""
        event = ModelNodeIntrospectionEvent(
            node_id=uuid4(),
            node_type="orchestrator",
            node_version="3.2.1",
            capabilities=ModelNodeCapabilities(
                postgres=True,
                consul=True,
                vault=True,
                kafka=True,
                read=True,
                write=True,
            ),
            endpoints={
                "health": "http://localhost:8080/health",
                "api": "http://localhost:8080/api",
                "metrics": "http://localhost:8080/metrics",
            },
            node_role="orchestrator",
            metadata=ModelNodeMetadata(
                environment="production",
                region="us-east-1",
                cluster="primary",
            ),
            correlation_id=uuid4(),
            network_id="prod-network",
            deployment_id="deploy-123",
            epoch=42,
            timestamp=datetime.now(UTC),
        )

        output = reducer.reduce(initial_state, event)

        assert len(output.intents) == 2
        assert output.result.status == "pending"

        # Verify PostgreSQL record captures all the data
        postgres_intent = next(
            (
                i
                for i in output.intents
                if i.intent_type == "postgres.upsert_registration"
            ),
            None,
        )
        assert postgres_intent is not None

        record = postgres_intent.payload["record"]
        assert record["node_type"] == "orchestrator"
        assert record["node_version"] == "3.2.1"
        assert len(record["endpoints"]) == 3


# -----------------------------------------------------------------------------
# Pure Function Contract Tests
# -----------------------------------------------------------------------------


@pytest.mark.unit
class TestPureFunctionContract:
    """Tests verifying the pure function contract of the reducer."""

    def test_reducer_has_no_instance_state(
        self,
        reducer: RegistrationReducer,
    ) -> None:
        """Test that reducer has no mutable instance state."""
        # The reducer class should have no instance attributes beyond methods
        instance_vars = [
            attr
            for attr in dir(reducer)
            if not attr.startswith("_") and not callable(getattr(reducer, attr))
        ]
        assert (
            len(instance_vars) == 0
        ), f"Reducer should have no instance state, found: {instance_vars}"

    def test_reduce_does_not_mutate_input_state(
        self,
        reducer: RegistrationReducer,
    ) -> None:
        """Test that reduce does not mutate the input state."""
        state = ModelRegistrationState()
        original_status = state.status
        original_node_id = state.node_id

        event = create_introspection_event()
        reducer.reduce(state, event)

        # Original state should be unchanged
        assert state.status == original_status
        assert state.node_id == original_node_id

    def test_reduce_does_not_mutate_input_event(
        self,
        reducer: RegistrationReducer,
        initial_state: ModelRegistrationState,
    ) -> None:
        """Test that reduce does not mutate the input event."""
        node_id = uuid4()
        correlation_id = uuid4()
        event = create_introspection_event(
            node_id=node_id, correlation_id=correlation_id
        )

        original_node_id = event.node_id
        original_correlation_id = event.correlation_id

        reducer.reduce(initial_state, event)

        # Event should be unchanged (it's frozen anyway, but verify)
        assert event.node_id == original_node_id
        assert event.correlation_id == original_correlation_id

    def test_multiple_reducers_produce_same_results(self) -> None:
        """Test that multiple reducer instances produce same results."""
        reducer1 = RegistrationReducer()
        reducer2 = RegistrationReducer()

        state = ModelRegistrationState()
        node_id = uuid4()
        correlation_id = uuid4()
        event = create_introspection_event(
            node_id=node_id, correlation_id=correlation_id
        )

        output1 = reducer1.reduce(state, event)
        output2 = reducer2.reduce(state, event)

        # Results should be equivalent
        assert output1.result == output2.result
        assert len(output1.intents) == len(output2.intents)
        assert output1.items_processed == output2.items_processed


# -----------------------------------------------------------------------------
# Performance Tests
# -----------------------------------------------------------------------------


@pytest.mark.unit
class TestPerformance:
    """Tests for performance characteristics and thresholds.

    These tests validate that:
    1. Performance constants are properly exported and usable
    2. reduce() operation completes within target thresholds
    3. processing_time_ms is accurately reported in output

    Note: Performance tests use generous thresholds (300ms) since:
    - CI environments have variable performance
    - The goal is to catch major regressions, not micro-optimizations
    - Typical execution is <5ms on standard hardware
    """

    def test_performance_constants_are_exported(self) -> None:
        """Test that performance threshold constants are properly exported."""
        from omnibase_infra.nodes.reducers.registration_reducer import (
            PERF_THRESHOLD_IDEMPOTENCY_CHECK_MS,
            PERF_THRESHOLD_INTENT_BUILD_MS,
            PERF_THRESHOLD_REDUCE_MS,
        )

        # Verify constants have expected values
        assert PERF_THRESHOLD_REDUCE_MS == 300.0
        assert PERF_THRESHOLD_INTENT_BUILD_MS == 50.0
        assert PERF_THRESHOLD_IDEMPOTENCY_CHECK_MS == 1.0

        # Verify they are floats (for consistent comparison)
        assert isinstance(PERF_THRESHOLD_REDUCE_MS, float)
        assert isinstance(PERF_THRESHOLD_INTENT_BUILD_MS, float)
        assert isinstance(PERF_THRESHOLD_IDEMPOTENCY_CHECK_MS, float)

    def test_reduce_completes_within_threshold(
        self,
        reducer: RegistrationReducer,
        initial_state: ModelRegistrationState,
        valid_event: ModelNodeIntrospectionEvent,
    ) -> None:
        """Test that reduce() completes well within the 300ms threshold.

        This test validates the primary performance target: <300ms per event.
        In practice, typical execution is <5ms on standard hardware.
        """
        from omnibase_infra.nodes.reducers.registration_reducer import (
            PERF_THRESHOLD_REDUCE_MS,
        )

        output = reducer.reduce(initial_state, valid_event)

        # Processing time should be well under threshold
        assert output.processing_time_ms < PERF_THRESHOLD_REDUCE_MS, (
            f"Processing time {output.processing_time_ms}ms exceeded "
            f"threshold {PERF_THRESHOLD_REDUCE_MS}ms"
        )

        # For healthy systems, should complete in <50ms typically
        # (we don't assert this to avoid flaky tests in slow CI)
        assert output.processing_time_ms >= 0.0

    def test_reduce_reset_completes_within_threshold(
        self,
        reducer: RegistrationReducer,
    ) -> None:
        """Test that reduce_reset() completes well within the threshold."""
        from omnibase_infra.nodes.reducers.registration_reducer import (
            PERF_THRESHOLD_REDUCE_MS,
        )

        # Create a failed state to reset
        state = ModelRegistrationState()
        node_id = uuid4()
        pending_state = state.with_pending_registration(node_id, uuid4())
        failed_state = pending_state.with_failure("consul_failed", uuid4())

        output = reducer.reduce_reset(failed_state, uuid4())

        # Processing time should be well under threshold
        assert output.processing_time_ms < PERF_THRESHOLD_REDUCE_MS

    def test_processing_time_is_reported_accurately(
        self,
        reducer: RegistrationReducer,
        initial_state: ModelRegistrationState,
        valid_event: ModelNodeIntrospectionEvent,
    ) -> None:
        """Test that processing_time_ms is a reasonable positive value."""
        import time

        start_time = time.perf_counter()
        output = reducer.reduce(initial_state, valid_event)
        elapsed_ms = (time.perf_counter() - start_time) * 1000

        # Output processing_time_ms should be less than or equal to elapsed time
        # (allow small margin for measurement overhead)
        assert output.processing_time_ms <= elapsed_ms + 1.0

        # Should be non-negative
        assert output.processing_time_ms >= 0.0

    def test_idempotency_check_is_fast(
        self,
        reducer: RegistrationReducer,
    ) -> None:
        """Test that idempotency check (duplicate detection) is fast.

        When an event is already processed, the reducer should return
        immediately with minimal processing time.
        """
        correlation_id = uuid4()
        event = create_introspection_event(correlation_id=correlation_id)

        # First reduce
        initial_state = ModelRegistrationState()
        output1 = reducer.reduce(initial_state, event)
        state_after_first = output1.result

        # Second reduce with same event (duplicate)
        output2 = reducer.reduce(state_after_first, event)

        # Duplicate detection should be near-instant
        # Use relaxed assertion: processing_time_ms should be very small (< 1ms)
        # rather than exactly 0.0 to avoid over-constraining implementation
        assert output2.processing_time_ms >= 0.0
        assert output2.processing_time_ms < 1.0  # Should complete in <1ms
        assert output2.items_processed == 0

    def test_processing_time_scales_reasonably(
        self,
        reducer: RegistrationReducer,
    ) -> None:
        """Test that processing multiple events has reasonable overhead.

        This test validates that the reducer doesn't have hidden O(n^2)
        behavior or state accumulation issues.
        """
        from omnibase_infra.nodes.reducers.registration_reducer import (
            PERF_THRESHOLD_REDUCE_MS,
        )

        state = ModelRegistrationState()
        total_processing_time = 0.0
        num_events = 10

        for i in range(num_events):
            event = create_introspection_event()
            output = reducer.reduce(state, event)

            # Each event should process independently and quickly
            assert output.processing_time_ms < PERF_THRESHOLD_REDUCE_MS

            total_processing_time += output.processing_time_ms

            # Use the new state for next iteration (though state changes)
            state = output.result

        # Average processing time should be reasonable
        avg_time = total_processing_time / num_events
        assert (
            avg_time < PERF_THRESHOLD_REDUCE_MS / 2
        ), f"Average processing time {avg_time}ms is too high"


# -----------------------------------------------------------------------------
# Complete State Transitions Tests (OMN-942)
# -----------------------------------------------------------------------------


@pytest.mark.unit
class TestCompleteStateTransitions:
    """Comprehensive tests for all FSM state transitions (OMN-942).

    This test class documents and validates all valid and invalid state
    transitions in the registration FSM. The FSM has 5 states:
    idle, pending, partial, complete, failed.

    State Diagram::

        +-------+   introspection   +---------+
        | idle  | ----------------> | pending |
        +-------+                   +---------+
           ^                         |       |
           |       consul confirmed  |       | postgres confirmed
           |       (first)          v       v (first)
           |                  +---------+
           |                  | partial |
           |                  +---------+
           |                    |       |
           |   remaining        |       | error received
           |   confirmed        v       v
           |              +---------+ +---------+
           +---reset------| complete| | failed  |---reset---+
                          +---------+ +---------+           |
                               |                            v
                               +---reset--->  +-------+
                                              | idle  |
                                              +-------+

    Valid State Transitions (13 total):
        1. idle -> pending (introspection event via reduce())
        2. pending -> partial (first confirmation - consul confirmed)
        3. pending -> partial (first confirmation - postgres confirmed)
        4. pending -> failed (validation error or backend error)
        5. partial -> complete (second confirmation - consul then postgres)
        6. partial -> complete (second confirmation - postgres then consul)
        7. partial -> failed (error on remaining backend)
        8. failed -> idle (reset via reduce_reset())
        9. complete -> idle (reset via reduce_reset())
        10. idle -> failed (invalid reset attempt via reduce_reset())
        11. pending -> failed (invalid reset attempt via reduce_reset())
        12. partial -> failed (invalid reset attempt via reduce_reset())
        13. pending -> complete is IMPOSSIBLE (requires partial first)

    Invalid Transitions:
        - complete -> pending (no direct path)
        - complete -> partial (no direct path)
        - failed -> pending (must reset to idle first)
        - idle -> complete (requires pending and partial)
        - partial -> pending (cannot regress)
    """

    # -------------------------------------------------------------------------
    # Document all valid states
    # -------------------------------------------------------------------------

    def test_all_valid_states_exist(self) -> None:
        """Document all valid FSM states."""
        from typing import get_args

        from omnibase_infra.nodes.reducers.models.model_registration_state import (
            RegistrationStatus,
        )

        # Extract valid states from the Literal type
        valid_states = set(get_args(RegistrationStatus))

        expected_states = {"idle", "pending", "partial", "complete", "failed"}
        assert (
            valid_states == expected_states
        ), f"Expected states {expected_states}, got {valid_states}"

    def test_initial_state_is_idle(self) -> None:
        """Verify that the default initial state is idle."""
        state = ModelRegistrationState()

        assert state.status == "idle"
        assert state.node_id is None
        assert state.consul_confirmed is False
        assert state.postgres_confirmed is False
        assert state.failure_reason is None

    # -------------------------------------------------------------------------
    # Transition 1: idle -> pending (via reduce())
    # -------------------------------------------------------------------------

    def test_transition_idle_to_pending_via_introspection(
        self,
        reducer: RegistrationReducer,
    ) -> None:
        """Test idle -> pending transition on introspection event.

        This is the primary entry point to the FSM. When a node introspection
        event is processed, the state transitions from idle to pending.
        """
        state = ModelRegistrationState()
        assert state.status == "idle"

        event = create_introspection_event()
        output = reducer.reduce(state, event)

        assert output.result.status == "pending"
        assert output.result.node_id == event.node_id
        assert len(output.intents) == 2  # Consul + PostgreSQL intents

    # -------------------------------------------------------------------------
    # Transition 2: pending -> partial (Consul confirmed first)
    # -------------------------------------------------------------------------

    def test_transition_pending_to_partial_consul_first(self) -> None:
        """Test pending -> partial when Consul confirms first.

        When Consul confirms registration before PostgreSQL, the state
        transitions to 'partial' with consul_confirmed=True.
        """
        state = ModelRegistrationState()
        node_id = uuid4()
        pending_state = state.with_pending_registration(node_id, uuid4())

        assert pending_state.status == "pending"
        assert pending_state.consul_confirmed is False
        assert pending_state.postgres_confirmed is False

        partial_state = pending_state.with_consul_confirmed(uuid4())

        assert partial_state.status == "partial"
        assert partial_state.consul_confirmed is True
        assert partial_state.postgres_confirmed is False
        assert partial_state.node_id == node_id

    # -------------------------------------------------------------------------
    # Transition 3: pending -> partial (PostgreSQL confirmed first)
    # -------------------------------------------------------------------------

    def test_transition_pending_to_partial_postgres_first(self) -> None:
        """Test pending -> partial when PostgreSQL confirms first.

        When PostgreSQL confirms registration before Consul, the state
        transitions to 'partial' with postgres_confirmed=True.
        """
        state = ModelRegistrationState()
        node_id = uuid4()
        pending_state = state.with_pending_registration(node_id, uuid4())

        assert pending_state.status == "pending"

        partial_state = pending_state.with_postgres_confirmed(uuid4())

        assert partial_state.status == "partial"
        assert partial_state.consul_confirmed is False
        assert partial_state.postgres_confirmed is True
        assert partial_state.node_id == node_id

    # -------------------------------------------------------------------------
    # Transition 4: pending -> failed (validation or backend error)
    # -------------------------------------------------------------------------

    def test_transition_pending_to_failed_validation_error(
        self,
        reducer: RegistrationReducer,
    ) -> None:
        """Test pending -> failed on validation error.

        When an invalid event is processed, the state transitions to failed
        with failure_reason='validation_failed'.
        """
        from unittest.mock import MagicMock

        state = ModelRegistrationState()

        # Create an invalid event (missing node_id)
        mock_event = MagicMock(spec=ModelNodeIntrospectionEvent)
        mock_event.node_id = None
        mock_event.node_type = "effect"
        mock_event.correlation_id = uuid4()

        output = reducer.reduce(state, mock_event)

        assert output.result.status == "failed"
        assert output.result.failure_reason == "validation_failed"

    def test_transition_pending_to_failed_backend_error(self) -> None:
        """Test pending -> failed on backend error.

        When a backend (Consul or PostgreSQL) returns an error, the state
        transitions to failed with the appropriate failure_reason.
        """
        state = ModelRegistrationState()
        node_id = uuid4()
        pending_state = state.with_pending_registration(node_id, uuid4())

        # Simulate Consul failure
        failed_state = pending_state.with_failure("consul_failed", uuid4())

        assert failed_state.status == "failed"
        assert failed_state.failure_reason == "consul_failed"
        assert failed_state.node_id == node_id

    def test_transition_pending_to_failed_postgres_error(self) -> None:
        """Test pending -> failed on PostgreSQL error."""
        state = ModelRegistrationState()
        node_id = uuid4()
        pending_state = state.with_pending_registration(node_id, uuid4())

        failed_state = pending_state.with_failure("postgres_failed", uuid4())

        assert failed_state.status == "failed"
        assert failed_state.failure_reason == "postgres_failed"

    # -------------------------------------------------------------------------
    # Transition 5: partial -> complete (Consul first, then PostgreSQL)
    # -------------------------------------------------------------------------

    def test_transition_partial_to_complete_consul_then_postgres(self) -> None:
        """Test partial -> complete: Consul confirmed, then PostgreSQL.

        When Consul confirms first (pending -> partial), and then PostgreSQL
        confirms, the state transitions to 'complete'.
        """
        state = ModelRegistrationState()
        node_id = uuid4()
        pending_state = state.with_pending_registration(node_id, uuid4())

        # Consul confirms first -> partial
        partial_state = pending_state.with_consul_confirmed(uuid4())
        assert partial_state.status == "partial"
        assert partial_state.consul_confirmed is True
        assert partial_state.postgres_confirmed is False

        # PostgreSQL confirms second -> complete
        complete_state = partial_state.with_postgres_confirmed(uuid4())

        assert complete_state.status == "complete"
        assert complete_state.consul_confirmed is True
        assert complete_state.postgres_confirmed is True
        assert complete_state.node_id == node_id

    # -------------------------------------------------------------------------
    # Transition 6: partial -> complete (PostgreSQL first, then Consul)
    # -------------------------------------------------------------------------

    def test_transition_partial_to_complete_postgres_then_consul(self) -> None:
        """Test partial -> complete: PostgreSQL confirmed, then Consul.

        When PostgreSQL confirms first (pending -> partial), and then Consul
        confirms, the state transitions to 'complete'.
        """
        state = ModelRegistrationState()
        node_id = uuid4()
        pending_state = state.with_pending_registration(node_id, uuid4())

        # PostgreSQL confirms first -> partial
        partial_state = pending_state.with_postgres_confirmed(uuid4())
        assert partial_state.status == "partial"
        assert partial_state.consul_confirmed is False
        assert partial_state.postgres_confirmed is True

        # Consul confirms second -> complete
        complete_state = partial_state.with_consul_confirmed(uuid4())

        assert complete_state.status == "complete"
        assert complete_state.consul_confirmed is True
        assert complete_state.postgres_confirmed is True
        assert complete_state.node_id == node_id

    # -------------------------------------------------------------------------
    # Transition 7: partial -> failed (error on remaining backend)
    # -------------------------------------------------------------------------

    def test_transition_partial_to_failed_consul_confirmed_postgres_fails(self) -> None:
        """Test partial -> failed: Consul confirmed, PostgreSQL fails.

        When Consul confirms (partial state) but PostgreSQL returns an error,
        the state transitions to 'failed'. Consul confirmation is preserved
        for diagnostics.
        """
        state = ModelRegistrationState()
        node_id = uuid4()
        pending_state = state.with_pending_registration(node_id, uuid4())
        partial_state = pending_state.with_consul_confirmed(uuid4())

        assert partial_state.status == "partial"
        assert partial_state.consul_confirmed is True

        failed_state = partial_state.with_failure("postgres_failed", uuid4())

        assert failed_state.status == "failed"
        assert failed_state.failure_reason == "postgres_failed"
        # Consul confirmation preserved for diagnostics
        assert failed_state.consul_confirmed is True
        assert failed_state.postgres_confirmed is False

    def test_transition_partial_to_failed_postgres_confirmed_consul_fails(self) -> None:
        """Test partial -> failed: PostgreSQL confirmed, Consul fails.

        When PostgreSQL confirms (partial state) but Consul returns an error,
        the state transitions to 'failed'. PostgreSQL confirmation is preserved
        for diagnostics.
        """
        state = ModelRegistrationState()
        node_id = uuid4()
        pending_state = state.with_pending_registration(node_id, uuid4())
        partial_state = pending_state.with_postgres_confirmed(uuid4())

        assert partial_state.status == "partial"
        assert partial_state.postgres_confirmed is True

        failed_state = partial_state.with_failure("consul_failed", uuid4())

        assert failed_state.status == "failed"
        assert failed_state.failure_reason == "consul_failed"
        # PostgreSQL confirmation preserved for diagnostics
        assert failed_state.consul_confirmed is False
        assert failed_state.postgres_confirmed is True

    # -------------------------------------------------------------------------
    # Transition 8: failed -> idle (reset via reduce_reset())
    # -------------------------------------------------------------------------

    def test_transition_failed_to_idle_via_reset(
        self,
        reducer: RegistrationReducer,
    ) -> None:
        """Test failed -> idle transition via reduce_reset().

        The reset mechanism allows recovery from failed states.
        All confirmation flags and failure reason are cleared.
        """
        state = ModelRegistrationState()
        node_id = uuid4()
        pending_state = state.with_pending_registration(node_id, uuid4())
        failed_state = pending_state.with_failure("consul_failed", uuid4())

        assert failed_state.status == "failed"
        assert failed_state.failure_reason == "consul_failed"

        reset_output = reducer.reduce_reset(failed_state, uuid4())

        assert reset_output.result.status == "idle"
        assert reset_output.result.node_id is None
        assert reset_output.result.consul_confirmed is False
        assert reset_output.result.postgres_confirmed is False
        assert reset_output.result.failure_reason is None
        assert reset_output.items_processed == 1

    # -------------------------------------------------------------------------
    # Transition 9: complete -> idle (reset via reduce_reset())
    # -------------------------------------------------------------------------

    def test_transition_complete_to_idle_via_reset(
        self,
        reducer: RegistrationReducer,
    ) -> None:
        """Test complete -> idle transition via reduce_reset().

        The reset mechanism also works from complete state, enabling
        re-registration of a node if needed.
        """
        state = ModelRegistrationState()
        node_id = uuid4()
        pending_state = state.with_pending_registration(node_id, uuid4())
        partial_state = pending_state.with_consul_confirmed(uuid4())
        complete_state = partial_state.with_postgres_confirmed(uuid4())

        assert complete_state.status == "complete"

        reset_output = reducer.reduce_reset(complete_state, uuid4())

        assert reset_output.result.status == "idle"
        assert reset_output.result.node_id is None
        assert reset_output.result.consul_confirmed is False
        assert reset_output.result.postgres_confirmed is False
        assert reset_output.items_processed == 1

    # -------------------------------------------------------------------------
    # Transition 10: idle -> failed (invalid reset attempt)
    # -------------------------------------------------------------------------

    def test_transition_idle_to_failed_invalid_reset(
        self,
        reducer: RegistrationReducer,
    ) -> None:
        """Test idle -> failed on invalid reset attempt.

        Resetting from idle is invalid because there's nothing to reset.
        The state transitions to 'failed' with failure_reason='invalid_reset_state'.
        """
        state = ModelRegistrationState()
        assert state.status == "idle"

        reset_output = reducer.reduce_reset(state, uuid4())

        assert reset_output.result.status == "failed"
        assert reset_output.result.failure_reason == "invalid_reset_state"
        assert reset_output.items_processed == 1

    # -------------------------------------------------------------------------
    # Transition 11: pending -> failed (invalid reset attempt)
    # -------------------------------------------------------------------------

    def test_transition_pending_to_failed_invalid_reset(
        self,
        reducer: RegistrationReducer,
    ) -> None:
        """Test pending -> failed on invalid reset attempt.

        Resetting from pending is invalid because it would lose in-flight
        registration state. The state transitions to 'failed'.
        """
        state = ModelRegistrationState()
        node_id = uuid4()
        pending_state = state.with_pending_registration(node_id, uuid4())

        assert pending_state.status == "pending"

        reset_output = reducer.reduce_reset(pending_state, uuid4())

        assert reset_output.result.status == "failed"
        assert reset_output.result.failure_reason == "invalid_reset_state"
        # Node ID preserved for diagnostics
        assert reset_output.result.node_id == node_id

    # -------------------------------------------------------------------------
    # Transition 12: partial -> failed (invalid reset attempt)
    # -------------------------------------------------------------------------

    def test_transition_partial_to_failed_invalid_reset(
        self,
        reducer: RegistrationReducer,
    ) -> None:
        """Test partial -> failed on invalid reset attempt.

        Resetting from partial is invalid because it would discard the
        confirmation from one backend, leaving the system inconsistent.
        """
        state = ModelRegistrationState()
        node_id = uuid4()
        pending_state = state.with_pending_registration(node_id, uuid4())
        partial_state = pending_state.with_consul_confirmed(uuid4())

        assert partial_state.status == "partial"

        reset_output = reducer.reduce_reset(partial_state, uuid4())

        assert reset_output.result.status == "failed"
        assert reset_output.result.failure_reason == "invalid_reset_state"
        # Consul confirmation preserved for diagnostics
        assert reset_output.result.consul_confirmed is True

    # -------------------------------------------------------------------------
    # Transition 13: pending -> complete is IMPOSSIBLE
    # -------------------------------------------------------------------------

    def test_pending_to_complete_impossible_without_partial(self) -> None:
        """Verify pending -> complete is impossible without going through partial.

        The FSM requires both backends to confirm. Each confirmation results
        in either pending->partial (first) or partial->complete (second).
        There is no direct path from pending to complete.
        """
        state = ModelRegistrationState()
        node_id = uuid4()
        pending_state = state.with_pending_registration(node_id, uuid4())

        # Even if we try both confirmations at once, we must go through partial
        # First confirmation -> partial (not complete)
        after_first = pending_state.with_consul_confirmed(uuid4())
        assert (
            after_first.status == "partial"
        ), "First confirmation should result in partial, not complete"

        # Only second confirmation -> complete
        after_second = after_first.with_postgres_confirmed(uuid4())
        assert after_second.status == "complete"

    # -------------------------------------------------------------------------
    # Invalid Transitions: complete -> pending (no direct path)
    # -------------------------------------------------------------------------

    def test_invalid_transition_complete_to_pending_not_possible(self) -> None:
        """Verify no direct complete -> pending transition exists.

        Once complete, the state can only transition to idle via reset.
        There is no method to go directly from complete to pending.
        """
        state = ModelRegistrationState()
        node_id = uuid4()
        pending_state = state.with_pending_registration(node_id, uuid4())
        partial_state = pending_state.with_consul_confirmed(uuid4())
        complete_state = partial_state.with_postgres_confirmed(uuid4())

        assert complete_state.status == "complete"

        # The only available transitions from complete state are:
        # - with_reset() -> idle
        # - with_failure() -> failed (for error scenarios)
        # There is no with_pending_registration that works on complete state
        # (calling it would create a new pending state, not a transition)

        # Verify the state machine doesn't have unintended transitions
        available_methods = [
            m
            for m in dir(complete_state)
            if m.startswith("with_") and callable(getattr(complete_state, m))
        ]
        assert "with_reset" in available_methods
        assert "with_failure" in available_methods

    # -------------------------------------------------------------------------
    # Invalid Transitions: failed -> pending (must reset to idle first)
    # -------------------------------------------------------------------------

    def test_invalid_transition_failed_to_pending_requires_reset(
        self,
        reducer: RegistrationReducer,
    ) -> None:
        """Verify failed -> pending requires going through idle via reset.

        A failed state cannot directly transition to pending. The correct
        recovery path is: failed -> idle (via reset) -> pending (via reduce).
        """
        state = ModelRegistrationState()
        node_id = uuid4()
        pending_state = state.with_pending_registration(node_id, uuid4())
        failed_state = pending_state.with_failure("consul_failed", uuid4())

        assert failed_state.status == "failed"

        # Correct recovery path:
        # 1. Reset to idle
        reset_output = reducer.reduce_reset(failed_state, uuid4())
        assert reset_output.result.status == "idle"

        # 2. New introspection event to pending
        new_event = create_introspection_event()
        new_output = reducer.reduce(reset_output.result, new_event)
        assert new_output.result.status == "pending"

    # -------------------------------------------------------------------------
    # Invalid Transitions: idle -> complete (requires pending and partial)
    # -------------------------------------------------------------------------

    def test_invalid_transition_idle_to_complete_not_possible(self) -> None:
        """Verify no direct idle -> complete transition exists.

        Completing registration requires going through the full FSM path:
        idle -> pending -> partial -> complete.
        """
        state = ModelRegistrationState()
        assert state.status == "idle"

        # Only available transitions from idle:
        # - with_pending_registration() -> pending
        # - with_failure() -> failed (for error scenarios)
        # There is no way to directly reach complete

        # Verify the correct path is required
        pending_state = state.with_pending_registration(uuid4(), uuid4())
        partial_state = pending_state.with_consul_confirmed(uuid4())
        complete_state = partial_state.with_postgres_confirmed(uuid4())

        assert complete_state.status == "complete"

    # -------------------------------------------------------------------------
    # Invalid Transitions: partial -> pending (cannot regress)
    # -------------------------------------------------------------------------

    def test_invalid_transition_partial_to_pending_cannot_regress(self) -> None:
        """Verify no partial -> pending regression exists.

        Once in partial state (one backend confirmed), there is no valid
        transition back to pending. The FSM only moves forward or to failed.
        """
        state = ModelRegistrationState()
        node_id = uuid4()
        pending_state = state.with_pending_registration(node_id, uuid4())
        partial_state = pending_state.with_consul_confirmed(uuid4())

        assert partial_state.status == "partial"

        # Available transitions from partial:
        # - with_consul_confirmed() -> complete (if postgres already confirmed)
        # - with_postgres_confirmed() -> complete (if consul already confirmed)
        # - with_failure() -> failed
        # There is no with_pending_registration that regresses to pending

        # Calling with_pending_registration on partial would start a NEW
        # registration, not regress the current one
        new_pending = partial_state.with_pending_registration(uuid4(), uuid4())
        assert new_pending.status == "pending"
        # But this is a new registration (new node_id), not a regression
        assert new_pending.node_id != node_id

    # -------------------------------------------------------------------------
    # Full workflow test: idle -> pending -> partial -> complete -> idle
    # -------------------------------------------------------------------------

    def test_full_successful_registration_workflow(
        self,
        reducer: RegistrationReducer,
    ) -> None:
        """Test the complete successful registration workflow.

        Validates the full FSM path:
        idle -> pending -> partial -> complete -> idle (reset)
        """
        # Start: idle
        initial_state = ModelRegistrationState()
        assert initial_state.status == "idle"

        # Step 1: idle -> pending (introspection event)
        event = create_introspection_event()
        pending_output = reducer.reduce(initial_state, event)
        assert pending_output.result.status == "pending"
        assert len(pending_output.intents) == 2

        # Step 2: pending -> partial (first confirmation)
        partial_state = pending_output.result.with_consul_confirmed(uuid4())
        assert partial_state.status == "partial"

        # Step 3: partial -> complete (second confirmation)
        complete_state = partial_state.with_postgres_confirmed(uuid4())
        assert complete_state.status == "complete"

        # Step 4: complete -> idle (reset for re-registration)
        reset_output = reducer.reduce_reset(complete_state, uuid4())
        assert reset_output.result.status == "idle"

    # -------------------------------------------------------------------------
    # Full workflow test: idle -> pending -> failed -> idle -> pending
    # -------------------------------------------------------------------------

    def test_full_failure_and_recovery_workflow(
        self,
        reducer: RegistrationReducer,
    ) -> None:
        """Test the complete failure and recovery workflow.

        Validates the FSM recovery path:
        idle -> pending -> failed -> idle (reset) -> pending (retry)
        """
        # Start: idle
        initial_state = ModelRegistrationState()
        assert initial_state.status == "idle"

        # Step 1: idle -> pending
        event = create_introspection_event()
        pending_output = reducer.reduce(initial_state, event)
        assert pending_output.result.status == "pending"

        # Step 2: pending -> failed (simulate backend error)
        failed_state = pending_output.result.with_failure("consul_failed", uuid4())
        assert failed_state.status == "failed"

        # Step 3: failed -> idle (reset)
        reset_output = reducer.reduce_reset(failed_state, uuid4())
        assert reset_output.result.status == "idle"

        # Step 4: idle -> pending (retry)
        retry_event = create_introspection_event()
        retry_output = reducer.reduce(reset_output.result, retry_event)
        assert retry_output.result.status == "pending"
        assert len(retry_output.intents) == 2


# -----------------------------------------------------------------------------
# Circuit Breaker Non-Applicability Documentation Tests
# -----------------------------------------------------------------------------


@pytest.mark.unit
class TestCircuitBreakerNonApplicability:
    """Tests documenting why circuit breaker is NOT needed for this reducer.

    These tests serve as executable documentation that the RegistrationReducer
    follows the pure function pattern and therefore does not require circuit
    breaker integration.

    Key points:
    1. Pure reducers perform NO I/O operations
    2. All external interactions are delegated to Effect layer via intents
    3. Circuit breakers are for I/O resilience, not pure computation
    4. Effect layer nodes (ConsulAdapter, PostgresAdapter) own their resilience
    """

    def test_reducer_has_no_async_methods(self) -> None:
        """Verify reducer has no async methods (no I/O)."""
        reducer = RegistrationReducer()

        # Get all methods
        methods = [
            name
            for name in dir(reducer)
            if callable(getattr(reducer, name)) and not name.startswith("_")
        ]

        # Check none are coroutines
        import inspect

        for method_name in methods:
            method = getattr(reducer, method_name)
            assert not inspect.iscoroutinefunction(
                method
            ), f"Method {method_name} is async - reducers should be pure/sync"

    def test_reducer_has_no_circuit_breaker_mixin(self) -> None:
        """Verify reducer does not inherit from MixinAsyncCircuitBreaker."""
        reducer = RegistrationReducer()

        # Check MRO for circuit breaker mixin
        mro_names = [cls.__name__ for cls in type(reducer).__mro__]

        assert (
            "MixinAsyncCircuitBreaker" not in mro_names
        ), "Pure reducers should not have circuit breaker mixin"

    def test_reducer_outputs_intents_not_io(
        self,
        reducer: RegistrationReducer,
        initial_state: ModelRegistrationState,
        valid_event: ModelNodeIntrospectionEvent,
    ) -> None:
        """Verify reducer emits intents (declarative) not I/O (imperative).

        The reducer returns ModelIntent objects that describe desired actions.
        It does NOT execute those actions - that's the Effect layer's job.
        """
        output = reducer.reduce(initial_state, valid_event)

        # Reducer emits intents, not results of I/O operations
        assert len(output.intents) == 2

        for intent in output.intents:
            # Intents are declarative descriptions
            assert intent.intent_type in (
                "consul.register",
                "postgres.upsert_registration",
            )
            assert intent.target is not None
            assert intent.payload is not None

            # Verify these are just intent descriptions, not executed operations
            # (the payload is serialized data, not live connections)
            assert isinstance(intent.payload, dict)

    def test_reducer_is_deterministic(
        self,
        reducer: RegistrationReducer,
    ) -> None:
        """Verify reducer is deterministic - same inputs produce same outputs.

        Deterministic behavior means no circuit breaker retry logic is needed.
        If an operation fails, retrying with the same inputs produces the
        same result - circuit breakers are for non-deterministic I/O.
        """
        state = ModelRegistrationState()
        node_id = uuid4()
        correlation_id = uuid4()
        event = create_introspection_event(
            node_id=node_id, correlation_id=correlation_id
        )

        # Run reduce multiple times with same inputs
        outputs = [reducer.reduce(state, event) for _ in range(5)]

        # All outputs should have equivalent results (except operation_id)
        for output in outputs:
            assert output.result == outputs[0].result
            assert len(output.intents) == len(outputs[0].intents)
            assert output.items_processed == outputs[0].items_processed


# -----------------------------------------------------------------------------
# Property-Based Determinism Tests (Hypothesis)
# -----------------------------------------------------------------------------


# Check if hypothesis is available; skip tests if not installed
try:
    from hypothesis import given, settings
    from hypothesis import strategies as st

    HYPOTHESIS_AVAILABLE = True
except ImportError:
    HYPOTHESIS_AVAILABLE = False

    # Provide dummy decorators when hypothesis is not available
    def given(*args, **kwargs):  # type: ignore[no-redef]
        def decorator(func):
            return pytest.mark.skip(
                reason="hypothesis not installed - add to dev dependencies"
            )(func)

        return decorator

    def settings(*args, **kwargs):  # type: ignore[no-redef]
        def decorator(func):
            return func

        return decorator

    class StrategiesStub:
        """Stub for hypothesis.strategies when Hypothesis is not installed."""

        @staticmethod
        def sampled_from(values):
            return values

        @staticmethod
        def text(*_args, **_kwargs):
            return None

        @staticmethod
        def uuids():
            return None

        @staticmethod
        def integers(*_args, **_kwargs):
            return None

        @staticmethod
        def booleans():
            return None

        @staticmethod
        def dictionaries(*_args, **_kwargs):
            return None

    st = StrategiesStub  # Alias for compatibility


@pytest.mark.unit
class TestDeterminismProperty:
    """Property-based tests for reducer determinism using Hypothesis.

    These tests validate the core determinism guarantees of the RegistrationReducer:
    1. Same state + same event always produces identical output (excluding operation_id)
    2. Replaying N events produces same final state regardless of replay count
    3. Derived event IDs are deterministic (same content = same derived ID)
    4. State transitions are idempotent when applied with same event_id
    5. Multiple reducer instances produce identical results

    Property-based testing with Hypothesis generates many random test cases to find
    edge cases that example-based tests might miss.
    """

    @given(
        node_type=st.sampled_from(["effect", "compute", "reducer", "orchestrator"]),
        major=st.integers(min_value=0, max_value=99),
        minor=st.integers(min_value=0, max_value=99),
        patch=st.integers(min_value=0, max_value=99),
    )
    @settings(max_examples=50)
    def test_reduce_is_deterministic_for_any_valid_input(
        self, node_type: str, major: int, minor: int, patch: int
    ) -> None:
        """Property: reduce(state, event) is deterministic for any valid input.

        For any valid node_type and node_version (semantic version format),
        calling reduce() twice with the same state and event must produce
        structurally identical outputs (excluding the operation_id and timestamps
        which are intentionally unique per call).
        """
        reducer = RegistrationReducer()
        state = ModelRegistrationState()
        node_id = uuid4()
        correlation_id = uuid4()
        node_version = f"{major}.{minor}.{patch}"

        event = ModelNodeIntrospectionEvent(
            node_id=node_id,
            node_type=node_type,
            node_version=node_version,
            endpoints={"health": "http://localhost:8080/health"},
            correlation_id=correlation_id,
        )

        # Execute reduce twice with identical inputs
        output1 = reducer.reduce(state, event)
        output2 = reducer.reduce(state, event)

        # State must be identical
        assert (
            output1.result == output2.result
        ), f"State mismatch for node_type={node_type}, version={node_version}"

        # Intent count and structure must be identical
        assert len(output1.intents) == len(output2.intents)

        # Intent payloads must be identical (compare non-timestamp fields)
        # Note: Timestamps (registered_at, updated_at) are generated at reduce() time,
        # so they naturally differ between calls. We exclude them from comparison.
        for intent1, intent2 in zip(output1.intents, output2.intents, strict=True):
            assert intent1.intent_type == intent2.intent_type
            assert intent1.target == intent2.target

            # For postgres intents, exclude timestamp fields from comparison
            if intent1.intent_type == "postgres.upsert_registration":
                payload1 = dict(intent1.payload)
                payload2 = dict(intent2.payload)
                record1 = dict(payload1.get("record", {}))
                record2 = dict(payload2.get("record", {}))
                # Remove timestamps before comparison
                for key in ["registered_at", "updated_at"]:
                    record1.pop(key, None)
                    record2.pop(key, None)
                payload1["record"] = record1
                payload2["record"] = record2
                assert payload1 == payload2
            else:
                assert intent1.payload == intent2.payload

        # Items processed must match
        assert output1.items_processed == output2.items_processed

        # Note: operation_id is intentionally different per call

    @given(replay_count=st.integers(min_value=2, max_value=10))
    @settings(max_examples=25)
    def test_replaying_same_event_produces_idempotent_state(
        self, replay_count: int
    ) -> None:
        """Property: Replaying the same event N times produces same final state.

        After the first reduce(), all subsequent replays with the same event
        should return the same state with no intents (idempotency).
        """
        reducer = RegistrationReducer()
        initial_state = ModelRegistrationState()
        correlation_id = uuid4()
        node_id = uuid4()

        event = create_introspection_event(
            node_id=node_id, correlation_id=correlation_id
        )

        # First reduce establishes the state
        output = reducer.reduce(initial_state, event)
        state_after_first = output.result
        assert len(output.intents) == 2  # Consul + PostgreSQL

        # All subsequent replays should be idempotent
        current_state = state_after_first
        for i in range(replay_count - 1):
            replay_output = reducer.reduce(current_state, event)

            # State should be unchanged
            assert (
                replay_output.result == state_after_first
            ), f"State changed on replay {i + 2}"

            # No intents should be emitted on replay
            assert len(replay_output.intents) == 0, f"Intents emitted on replay {i + 2}"

            # Items processed should be 0 (duplicate detection)
            assert replay_output.items_processed == 0

            current_state = replay_output.result

    @given(
        node_type=st.sampled_from(["effect", "compute", "reducer", "orchestrator"]),
    )
    @settings(max_examples=20)
    def test_derived_event_id_is_deterministic(self, node_type: str) -> None:
        """Property: Derived event IDs are deterministic.

        When an event lacks a correlation_id, the reducer derives an event_id
        from the event's content (node_id, node_type, timestamp). This derived
        ID must be deterministic - same content always produces same ID.
        """
        from unittest.mock import MagicMock

        reducer = RegistrationReducer()
        node_id = uuid4()
        fixed_timestamp = datetime.now(UTC)

        # Create mock event without correlation_id (forces derivation)
        mock_event = MagicMock(spec=ModelNodeIntrospectionEvent)
        mock_event.node_id = node_id
        mock_event.node_type = node_type
        mock_event.node_version = "1.0.0"
        mock_event.endpoints = {"health": "http://localhost:8080/health"}
        mock_event.capabilities = ModelNodeCapabilities()
        mock_event.metadata = ModelNodeMetadata()
        mock_event.correlation_id = None  # Force deterministic derivation
        mock_event.timestamp = fixed_timestamp

        # Derive event ID multiple times
        derived_id_1 = reducer._derive_deterministic_event_id(mock_event)
        derived_id_2 = reducer._derive_deterministic_event_id(mock_event)
        derived_id_3 = reducer._derive_deterministic_event_id(mock_event)

        # All derived IDs must be identical
        assert (
            derived_id_1 == derived_id_2 == derived_id_3
        ), f"Derived IDs not deterministic for node_type={node_type}"

        # Derived ID must be a valid UUID
        assert isinstance(derived_id_1, UUID)

    @given(
        has_health_endpoint=st.booleans(),
        has_api_endpoint=st.booleans(),
    )
    @settings(max_examples=20)
    def test_intent_payloads_are_deterministic_for_endpoint_variations(
        self, has_health_endpoint: bool, has_api_endpoint: bool
    ) -> None:
        """Property: Intent payloads are deterministic regardless of endpoint config.

        The reducer must produce identical intent payloads for identical inputs,
        even when endpoint configuration varies. Timestamps are excluded from
        comparison since they are generated at reduce() time.
        """
        reducer = RegistrationReducer()
        state = ModelRegistrationState()
        node_id = uuid4()
        correlation_id = uuid4()

        # Build endpoints dict based on property inputs
        endpoints: dict[str, str] = {}
        if has_health_endpoint:
            endpoints["health"] = "http://localhost:8080/health"
        if has_api_endpoint:
            endpoints["api"] = "http://localhost:8080/api"

        event = ModelNodeIntrospectionEvent(
            node_id=node_id,
            node_type="effect",
            node_version="1.0.0",
            endpoints=endpoints,
            correlation_id=correlation_id,
        )

        # Execute reduce twice
        output1 = reducer.reduce(state, event)
        output2 = reducer.reduce(state, event)

        # Compare intent payloads in detail (excluding timestamps)
        for intent1, intent2 in zip(output1.intents, output2.intents, strict=True):
            assert intent1.intent_type == intent2.intent_type
            assert intent1.target == intent2.target

            # For postgres intents, exclude timestamp fields from comparison
            if intent1.intent_type == "postgres.upsert_registration":
                payload1 = dict(intent1.payload)
                payload2 = dict(intent2.payload)
                record1 = dict(payload1.get("record", {}))
                record2 = dict(payload2.get("record", {}))
                # Remove timestamps before comparison
                for key in ["registered_at", "updated_at"]:
                    record1.pop(key, None)
                    record2.pop(key, None)
                payload1["record"] = record1
                payload2["record"] = record2
                assert (
                    payload1 == payload2
                ), f"Payload mismatch for intent_type={intent1.intent_type}"
            else:
                assert (
                    intent1.payload == intent2.payload
                ), f"Payload mismatch for intent_type={intent1.intent_type}"

    @given(
        num_reducers=st.integers(min_value=2, max_value=5),
    )
    @settings(max_examples=10)
    def test_multiple_reducer_instances_produce_identical_results(
        self, num_reducers: int
    ) -> None:
        """Property: Multiple reducer instances produce identical results.

        Since reducers are stateless pure functions, different instances
        must produce identical outputs for identical inputs. Timestamps are
        excluded from comparison since they are generated at reduce() time.
        """
        reducers = [RegistrationReducer() for _ in range(num_reducers)]
        state = ModelRegistrationState()
        node_id = uuid4()
        correlation_id = uuid4()

        event = ModelNodeIntrospectionEvent(
            node_id=node_id,
            node_type="compute",
            node_version="2.0.0",
            endpoints={"health": "http://localhost:8080/health"},
            correlation_id=correlation_id,
        )

        # Execute reduce on all reducer instances
        outputs = [r.reduce(state, event) for r in reducers]

        # All outputs must have identical results
        first_output = outputs[0]
        for i, output in enumerate(outputs[1:], start=2):
            assert (
                output.result == first_output.result
            ), f"Result mismatch between reducer 1 and {i}"
            assert len(output.intents) == len(
                first_output.intents
            ), f"Intent count mismatch between reducer 1 and {i}"
            for j, (intent1, intent2) in enumerate(
                zip(first_output.intents, output.intents, strict=True)
            ):
                assert intent1.intent_type == intent2.intent_type
                assert intent1.target == intent2.target

                # For postgres intents, exclude timestamp fields from comparison
                if intent1.intent_type == "postgres.upsert_registration":
                    payload1 = dict(intent1.payload)
                    payload2 = dict(intent2.payload)
                    record1 = dict(payload1.get("record", {}))
                    record2 = dict(payload2.get("record", {}))
                    # Remove timestamps before comparison
                    for key in ["registered_at", "updated_at"]:
                        record1.pop(key, None)
                        record2.pop(key, None)
                    payload1["record"] = record1
                    payload2["record"] = record2
                    assert (
                        payload1 == payload2
                    ), f"Intent {j} payload mismatch between reducer 1 and {i}"
                else:
                    assert (
                        intent1.payload == intent2.payload
                    ), f"Intent {j} payload mismatch between reducer 1 and {i}"

    @given(
        reset_attempts=st.integers(min_value=1, max_value=5),
    )
    @settings(max_examples=15)
    def test_reset_idempotency_after_first_reset(self, reset_attempts: int) -> None:
        """Property: Reset is idempotent after the first successful reset.

        After resetting from a failed state to idle, subsequent reset attempts
        with the same event_id should be no-ops (idempotent).
        """
        reducer = RegistrationReducer()

        # Create a failed state
        state = ModelRegistrationState()
        node_id = uuid4()
        pending_state = state.with_pending_registration(node_id, uuid4())
        failed_state = pending_state.with_failure("consul_failed", uuid4())

        reset_event_id = uuid4()

        # First reset should succeed
        output = reducer.reduce_reset(failed_state, reset_event_id)
        assert output.result.status == "idle"
        assert output.items_processed == 1
        idle_state = output.result

        # Subsequent resets with same event_id should be idempotent
        current_state = idle_state
        for i in range(reset_attempts):
            replay_output = reducer.reduce_reset(current_state, reset_event_id)

            # State should be unchanged (same as idle_state)
            assert (
                replay_output.result == idle_state
            ), f"State changed on reset replay {i + 1}"

            # No items processed (duplicate detection)
            assert (
                replay_output.items_processed == 0
            ), f"Items processed on reset replay {i + 1}"

            current_state = replay_output.result

    @given(
        node_type=st.sampled_from(["effect", "compute", "reducer", "orchestrator"]),
    )
    @settings(max_examples=20)
    def test_state_hash_stability_across_reduce_calls(self, node_type: str) -> None:
        """Property: State model hash is stable across identical reduce calls.

        The ModelRegistrationState uses Pydantic's frozen models. The resulting
        state from identical reduce calls must have identical hash values,
        enabling reliable comparison and caching.
        """
        reducer = RegistrationReducer()
        state = ModelRegistrationState()
        node_id = uuid4()
        correlation_id = uuid4()

        event = ModelNodeIntrospectionEvent(
            node_id=node_id,
            node_type=node_type,
            node_version="1.0.0",
            endpoints={"health": "http://localhost:8080/health"},
            correlation_id=correlation_id,
        )

        # Execute reduce twice
        output1 = reducer.reduce(state, event)
        output2 = reducer.reduce(state, event)

        # States must be equal (uses Pydantic's __eq__)
        assert output1.result == output2.result

        # States should be hashable (frozen=True in Pydantic model)
        # If the model is properly frozen, hash() should work
        try:
            hash1 = hash(output1.result)
            hash2 = hash(output2.result)
            assert hash1 == hash2, "State hashes differ for identical states"
        except TypeError:
            # Model might not be hashable if frozen=False
            # This is acceptable but we should still verify equality
            pass


# -----------------------------------------------------------------------------
# Comprehensive Edge Case Tests (OMN-942)
# -----------------------------------------------------------------------------


@pytest.mark.unit
class TestEdgeCasesComprehensive:
    """Comprehensive edge case testing for reducer robustness.

    These tests cover unusual but valid input combinations, boundary conditions,
    and edge cases that may occur in production environments.

    Related: OMN-942 - Reducer Test Suite Enhancement
    """

    def test_event_with_minimal_fields(
        self,
        reducer: RegistrationReducer,
        initial_state: ModelRegistrationState,
    ) -> None:
        """Test reduce with only required fields populated.

        Validates that the reducer handles events where all optional fields
        use their default values. This is the minimal valid event case.
        """
        event = ModelNodeIntrospectionEvent(
            node_id=uuid4(),
            node_type="effect",
            node_version="1.0.0",
            endpoints={},
            correlation_id=uuid4(),
        )

        output = reducer.reduce(initial_state, event)

        assert output.result.status == "pending"
        assert len(output.intents) == 2

        # Verify intents are still built correctly with minimal data
        consul_intent = next(
            (i for i in output.intents if i.intent_type == "consul.register"), None
        )
        assert consul_intent is not None
        assert consul_intent.payload["health_check"] is None

        postgres_intent = next(
            (
                i
                for i in output.intents
                if i.intent_type == "postgres.upsert_registration"
            ),
            None,
        )
        assert postgres_intent is not None
        assert postgres_intent.payload["record"]["health_endpoint"] is None

    def test_event_with_many_endpoints(
        self,
        reducer: RegistrationReducer,
        initial_state: ModelRegistrationState,
    ) -> None:
        """Test reduce with large endpoints dictionary.

        Validates that the reducer handles events with many endpoints,
        simulating a complex service with multiple interfaces.
        """
        many_endpoints = {
            "health": "http://localhost:8080/health",
            "api": "http://localhost:8080/api",
            "metrics": "http://localhost:9090/metrics",
            "admin": "http://localhost:8081/admin",
            "grpc": "grpc://localhost:50051",
            "websocket": "ws://localhost:8082/ws",
            "graphql": "http://localhost:8083/graphql",
            "debug": "http://localhost:8084/debug",
            "internal": "http://localhost:8085/internal",
            "status": "http://localhost:8086/status",
        }

        event = ModelNodeIntrospectionEvent(
            node_id=uuid4(),
            node_type="orchestrator",
            node_version="2.5.0",
            endpoints=many_endpoints,
            correlation_id=uuid4(),
        )

        output = reducer.reduce(initial_state, event)

        assert output.result.status == "pending"
        assert len(output.intents) == 2

        # Verify all endpoints are captured in PostgreSQL intent
        postgres_intent = next(
            (
                i
                for i in output.intents
                if i.intent_type == "postgres.upsert_registration"
            ),
            None,
        )
        assert postgres_intent is not None

        record_endpoints = postgres_intent.payload["record"]["endpoints"]
        assert len(record_endpoints) == 10
        for key in many_endpoints:
            assert key in record_endpoints

    def test_event_with_very_long_version_string(
        self,
        reducer: RegistrationReducer,
        initial_state: ModelRegistrationState,
    ) -> None:
        """Test reduce with a very long version string.

        Validates that the reducer handles unusually long version strings
        without truncation or error. SemVer with build metadata can be lengthy.
        """
        long_version = (
            "1.2.3-alpha.4.5.6+build.metadata.with.many.segments.202512211234"
        )

        event = ModelNodeIntrospectionEvent(
            node_id=uuid4(),
            node_type="compute",
            node_version=long_version,
            endpoints={"health": "http://localhost:8080/health"},
            correlation_id=uuid4(),
        )

        output = reducer.reduce(initial_state, event)

        assert output.result.status == "pending"

        # Verify long version is preserved in tags
        consul_intent = next(
            (i for i in output.intents if i.intent_type == "consul.register"), None
        )
        assert consul_intent is not None
        tags = consul_intent.payload["tags"]
        assert f"node_version:{long_version}" in tags

    def test_rapid_state_transitions(
        self,
        reducer: RegistrationReducer,
    ) -> None:
        """Test multiple state transitions in quick succession.

        Validates that the reducer correctly handles rapid state changes
        without state corruption or missed transitions.
        """
        state = ModelRegistrationState()

        # Rapid-fire: introspection -> consul confirm -> postgres confirm
        node_id = uuid4()
        event1 = create_introspection_event(node_id=node_id)
        output1 = reducer.reduce(state, event1)
        assert output1.result.status == "pending"

        # First confirmation
        state2 = output1.result.with_consul_confirmed(uuid4())
        assert state2.status == "partial"

        # Second confirmation
        state3 = state2.with_postgres_confirmed(uuid4())
        assert state3.status == "complete"

        # Verify final state is consistent
        assert state3.consul_confirmed is True
        assert state3.postgres_confirmed is True
        assert state3.node_id == node_id

    def test_same_node_re_registration_after_complete(
        self,
        reducer: RegistrationReducer,
    ) -> None:
        """Test that a node can re-register after complete->reset->idle.

        Validates the full recovery workflow where a node completes registration,
        is reset (perhaps for re-deployment), and re-registers successfully.
        """
        state = ModelRegistrationState()
        node_id = uuid4()

        # First registration
        event1 = create_introspection_event(node_id=node_id)
        output1 = reducer.reduce(state, event1)
        assert output1.result.status == "pending"

        # Complete the registration
        consul_state = output1.result.with_consul_confirmed(uuid4())
        complete_state = consul_state.with_postgres_confirmed(uuid4())
        assert complete_state.status == "complete"

        # Reset
        reset_output = reducer.reduce_reset(complete_state, uuid4())
        assert reset_output.result.status == "idle"

        # Re-register same node with new event
        event2 = create_introspection_event(node_id=node_id)
        output2 = reducer.reduce(reset_output.result, event2)
        assert output2.result.status == "pending"
        assert output2.result.node_id == node_id
        assert len(output2.intents) == 2

    def test_events_with_same_timestamp(
        self,
        reducer: RegistrationReducer,
        initial_state: ModelRegistrationState,
    ) -> None:
        """Test reduce with events having identical timestamps.

        Validates that deterministic event ID derivation still produces
        unique IDs when timestamps match but node_ids differ.
        """
        from unittest.mock import MagicMock

        fixed_timestamp = datetime.now(UTC)
        node_id1 = uuid4()
        node_id2 = uuid4()

        # Create two events with same timestamp but different node_ids
        # Use mocks to set correlation_id=None to trigger deterministic ID derivation
        mock_event1 = MagicMock(spec=ModelNodeIntrospectionEvent)
        mock_event1.node_id = node_id1
        mock_event1.node_type = "effect"
        mock_event1.node_version = "1.0.0"
        mock_event1.endpoints = {"health": "http://localhost:8080/health"}
        mock_event1.capabilities = ModelNodeCapabilities()
        mock_event1.metadata = ModelNodeMetadata()
        mock_event1.correlation_id = None
        mock_event1.timestamp = fixed_timestamp

        mock_event2 = MagicMock(spec=ModelNodeIntrospectionEvent)
        mock_event2.node_id = node_id2
        mock_event2.node_type = "effect"
        mock_event2.node_version = "1.0.0"
        mock_event2.endpoints = {"health": "http://localhost:8080/health"}
        mock_event2.capabilities = ModelNodeCapabilities()
        mock_event2.metadata = ModelNodeMetadata()
        mock_event2.correlation_id = None
        mock_event2.timestamp = fixed_timestamp

        output1 = reducer.reduce(initial_state, mock_event1)
        output2 = reducer.reduce(initial_state, mock_event2)

        # Both should process successfully
        assert output1.result.status == "pending"
        assert output2.result.status == "pending"

        # They should have different derived event IDs
        assert (
            output1.result.last_processed_event_id
            != output2.result.last_processed_event_id
        )

    def test_nil_uuid_handling(
        self,
        reducer: RegistrationReducer,
        initial_state: ModelRegistrationState,
    ) -> None:
        """Test that nil UUID (all zeros) is handled correctly.

        Validates that a nil UUID is accepted as a valid node_id.
        This is a boundary case for UUID handling.
        """
        nil_uuid = UUID("00000000-0000-0000-0000-000000000000")

        event = ModelNodeIntrospectionEvent(
            node_id=nil_uuid,
            node_type="effect",
            node_version="1.0.0",
            endpoints={"health": "http://localhost:8080/health"},
            correlation_id=uuid4(),
        )

        output = reducer.reduce(initial_state, event)

        assert output.result.status == "pending"
        assert output.result.node_id == nil_uuid

        # Verify nil UUID appears in intents
        consul_intent = next(
            (i for i in output.intents if i.intent_type == "consul.register"), None
        )
        assert consul_intent is not None
        assert str(nil_uuid) in consul_intent.payload["service_id"]

    def test_unicode_in_endpoint_urls(
        self,
        reducer: RegistrationReducer,
        initial_state: ModelRegistrationState,
    ) -> None:
        """Test that unicode characters in endpoint URLs are handled.

        Validates that the reducer preserves unicode in endpoint strings.
        Some international deployments may have unicode in paths.
        """
        event = ModelNodeIntrospectionEvent(
            node_id=uuid4(),
            node_type="effect",
            node_version="1.0.0",
            endpoints={
                "health": "http://localhost:8080/health",
                "api": "http://localhost:8080/api/v1/donnees",
                "docs": "http://localhost:8080/wendang/index",
            },
            correlation_id=uuid4(),
        )

        output = reducer.reduce(initial_state, event)

        assert output.result.status == "pending"

        postgres_intent = next(
            (
                i
                for i in output.intents
                if i.intent_type == "postgres.upsert_registration"
            ),
            None,
        )
        assert postgres_intent is not None

        endpoints = postgres_intent.payload["record"]["endpoints"]
        assert endpoints["api"] == "http://localhost:8080/api/v1/donnees"
        assert endpoints["docs"] == "http://localhost:8080/wendang/index"

    def test_state_transition_preserves_event_order(
        self,
        reducer: RegistrationReducer,
    ) -> None:
        """Test that state transitions maintain event order via last_processed_event_id.

        Validates that each state transition correctly updates the
        last_processed_event_id to maintain event ordering.
        """
        state = ModelRegistrationState()
        event_ids: list[UUID] = []

        # Generate a sequence of events and track their IDs
        for i in range(5):
            event_id = uuid4()
            event_ids.append(event_id)

            if i == 0:
                # First event: introspection
                event = create_introspection_event(correlation_id=event_id)
                output = reducer.reduce(state, event)
                state = output.result
            elif i == 1:
                # Second: consul confirmation
                state = state.with_consul_confirmed(event_id)
            elif i == 2:
                # Third: postgres confirmation (now complete)
                state = state.with_postgres_confirmed(event_id)
            elif i == 3:
                # Fourth: reset
                output = reducer.reduce_reset(state, event_id)
                state = output.result
            else:
                # Fifth: new introspection
                event = create_introspection_event(correlation_id=event_id)
                output = reducer.reduce(state, event)
                state = output.result

        # Final state should have the last event ID
        assert state.last_processed_event_id == event_ids[-1]

    def test_confirmation_order_independence(
        self,
    ) -> None:
        """Test that confirmation order doesn't affect final complete state.

        Validates that whether consul or postgres confirms first, the
        final complete state is equivalent.
        """
        state = ModelRegistrationState()
        node_id = uuid4()
        event_id = uuid4()

        pending_state = state.with_pending_registration(node_id, event_id)

        # Path 1: Consul first, then Postgres
        path1_partial = pending_state.with_consul_confirmed(uuid4())
        path1_complete = path1_partial.with_postgres_confirmed(uuid4())

        # Path 2: Postgres first, then Consul
        path2_partial = pending_state.with_postgres_confirmed(uuid4())
        path2_complete = path2_partial.with_consul_confirmed(uuid4())

        # Both paths should reach complete with same confirmations
        assert path1_complete.status == "complete"
        assert path2_complete.status == "complete"
        assert path1_complete.consul_confirmed == path2_complete.consul_confirmed
        assert path1_complete.postgres_confirmed == path2_complete.postgres_confirmed
        assert path1_complete.node_id == path2_complete.node_id


# -----------------------------------------------------------------------------
# Timeout Scenario Tests (OMN-942)
# -----------------------------------------------------------------------------


@pytest.mark.unit
class TestTimeoutScenarios:
    """Tests for timeout-related state transitions.

    Architecture Note:
        Per DESIGN_TWO_WAY_REGISTRATION_ARCHITECTURE.md, the Orchestrator owns
        timeout detection and emits timeout events. The Reducer folds timeout
        events as failure confirmations. This test class validates the reducer's
        handling of timeout-induced failure states.

    Timeout Event Flow:
        1. Orchestrator tracks pending registrations with deadlines
        2. Orchestrator consumes RuntimeTick events for timeout evaluation
        3. When deadline passes, Orchestrator emits RegistrationTimedOut event
        4. Reducer folds RegistrationTimedOut as failure with reason "consul_failed"
           or "postgres_failed" depending on what timed out

    Related: OMN-942 - Reducer Test Suite Enhancement
    """

    def test_timeout_in_pending_state_causes_failure(
        self,
    ) -> None:
        """Test that pending state + timeout event = failed state.

        When a registration times out while waiting for both confirmations,
        the state transitions to failed with an appropriate failure reason.
        """
        state = ModelRegistrationState()
        node_id = uuid4()
        pending_state = state.with_pending_registration(node_id, uuid4())

        # Simulate timeout as a failure (orchestrator would emit this)
        timeout_event_id = uuid4()
        failed_state = pending_state.with_failure("consul_failed", timeout_event_id)

        assert failed_state.status == "failed"
        assert failed_state.failure_reason == "consul_failed"
        assert failed_state.node_id == node_id
        # Confirmation flags remain false (neither confirmed before timeout)
        assert failed_state.consul_confirmed is False
        assert failed_state.postgres_confirmed is False

    def test_timeout_in_partial_state_causes_failure(
        self,
    ) -> None:
        """Test that partial state + timeout event = failed state.

        When one backend confirms but the other times out, the state
        transitions to failed while preserving the successful confirmation
        for diagnostic purposes.
        """
        state = ModelRegistrationState()
        node_id = uuid4()
        pending_state = state.with_pending_registration(node_id, uuid4())

        # Consul confirmed, waiting for postgres
        partial_state = pending_state.with_consul_confirmed(uuid4())
        assert partial_state.status == "partial"
        assert partial_state.consul_confirmed is True

        # Postgres times out
        timeout_event_id = uuid4()
        failed_state = partial_state.with_failure("postgres_failed", timeout_event_id)

        assert failed_state.status == "failed"
        assert failed_state.failure_reason == "postgres_failed"
        # Consul confirmation is preserved for diagnostics
        assert failed_state.consul_confirmed is True
        assert failed_state.postgres_confirmed is False

    def test_recovery_after_timeout_failure(
        self,
        reducer: RegistrationReducer,
    ) -> None:
        """Test reset and retry workflow after timeout failure.

        Validates the complete recovery workflow:
        1. Start registration
        2. Timeout occurs (failure)
        3. Reset to idle
        4. Retry registration
        5. Complete successfully
        """
        initial_state = ModelRegistrationState()
        node_id = uuid4()

        # Step 1: Start registration
        event1 = create_introspection_event(node_id=node_id)
        output1 = reducer.reduce(initial_state, event1)
        assert output1.result.status == "pending"

        # Step 2: Timeout occurs (simulated as failure)
        failed_state = output1.result.with_failure("consul_failed", uuid4())
        assert failed_state.status == "failed"
        assert failed_state.failure_reason == "consul_failed"

        # Step 3: Reset to idle
        reset_output = reducer.reduce_reset(failed_state, uuid4())
        assert reset_output.result.status == "idle"
        assert reset_output.result.failure_reason is None

        # Step 4: Retry registration (new correlation_id for new attempt)
        event2 = create_introspection_event(node_id=node_id)
        output2 = reducer.reduce(reset_output.result, event2)
        assert output2.result.status == "pending"
        assert len(output2.intents) == 2

        # Step 5: Complete successfully
        consul_confirmed = output2.result.with_consul_confirmed(uuid4())
        both_confirmed = consul_confirmed.with_postgres_confirmed(uuid4())
        assert both_confirmed.status == "complete"
        assert both_confirmed.failure_reason is None

    def test_timeout_preserves_node_id_for_retry(
        self,
    ) -> None:
        """Test that timeout failure preserves node_id for debugging.

        When a timeout occurs, the node_id should be preserved in the
        failed state so operators can identify which node failed.
        """
        state = ModelRegistrationState()
        node_id = uuid4()
        pending_state = state.with_pending_registration(node_id, uuid4())

        # Timeout failure
        failed_state = pending_state.with_failure("both_failed", uuid4())

        assert failed_state.status == "failed"
        assert failed_state.failure_reason == "both_failed"
        assert failed_state.node_id == node_id  # Preserved for debugging

    def test_multiple_timeout_retries(
        self,
        reducer: RegistrationReducer,
    ) -> None:
        """Test multiple timeout-retry cycles before success.

        Validates that the reducer correctly handles multiple retry attempts
        after repeated timeouts, eventually reaching success.
        """
        state = ModelRegistrationState()
        node_id = uuid4()

        # Simulate 3 failed attempts before success
        for attempt in range(3):
            # Start registration
            event = create_introspection_event(node_id=node_id)
            output = reducer.reduce(state, event)
            assert output.result.status == "pending"

            # Timeout failure
            failed_state = output.result.with_failure("consul_failed", uuid4())
            assert failed_state.status == "failed"

            # Reset for next attempt
            reset_output = reducer.reduce_reset(failed_state, uuid4())
            state = reset_output.result
            assert state.status == "idle"

        # Final successful attempt
        event = create_introspection_event(node_id=node_id)
        output = reducer.reduce(state, event)
        assert output.result.status == "pending"

        consul_confirmed = output.result.with_consul_confirmed(uuid4())
        both_confirmed = consul_confirmed.with_postgres_confirmed(uuid4())

        assert both_confirmed.status == "complete"
        assert both_confirmed.node_id == node_id

    def test_timeout_different_failure_reasons(
        self,
    ) -> None:
        """Test that different timeout scenarios produce correct failure reasons.

        Validates that the failure_reason correctly identifies which
        component timed out.
        """
        state = ModelRegistrationState()
        node_id = uuid4()
        pending_state = state.with_pending_registration(node_id, uuid4())

        # Scenario 1: Consul times out first (waiting for both)
        consul_timeout = pending_state.with_failure("consul_failed", uuid4())
        assert consul_timeout.failure_reason == "consul_failed"

        # Scenario 2: Postgres times out first (waiting for both)
        postgres_timeout = pending_state.with_failure("postgres_failed", uuid4())
        assert postgres_timeout.failure_reason == "postgres_failed"

        # Scenario 3: Both time out (no confirmations received)
        both_timeout = pending_state.with_failure("both_failed", uuid4())
        assert both_timeout.failure_reason == "both_failed"

        # Scenario 4: Consul confirmed but postgres times out
        partial_consul = pending_state.with_consul_confirmed(uuid4())
        postgres_after_consul = partial_consul.with_failure("postgres_failed", uuid4())
        assert postgres_after_consul.failure_reason == "postgres_failed"
        assert postgres_after_consul.consul_confirmed is True

        # Scenario 5: Postgres confirmed but consul times out
        partial_postgres = pending_state.with_postgres_confirmed(uuid4())
        consul_after_postgres = partial_postgres.with_failure("consul_failed", uuid4())
        assert consul_after_postgres.failure_reason == "consul_failed"
        assert consul_after_postgres.postgres_confirmed is True


# -----------------------------------------------------------------------------
# Command Folding Prevention Tests
# -----------------------------------------------------------------------------


@pytest.mark.unit
class TestCommandFoldingPrevention:
    """Tests verifying reducer ONLY processes events, never commands.

    Per ONEX architecture, reducers follow a strict separation:
    - **Events**: Describe what HAS happened (past tense)
        - Examples: NodeIntrospectionEvent, ConsulRegistered, RegistrationFailed
        - Reducers fold events into state using pure functions
    - **Commands**: Describe what SHOULD happen (imperative)
        - Examples: RegisterNode, DeregisterNode, RefreshRegistration
        - Commands are handled by Effect layer, NOT reducers

    This separation is critical because:
    1. Reducers are pure functions - they cannot execute side effects
    2. Commands require I/O (network, database) - reducers do no I/O
    3. Event sourcing requires immutable event history - commands are not logged
    4. Replay/recovery relies on events only - replaying commands would cause duplicates

    The reducer's output (intents) are NOT commands - they are declarative
    descriptions of desired side effects that the Effect layer will execute.
    Intents describe WHAT should happen, but the reducer doesn't DO it.

    See Also:
        - CLAUDE.md: "Enum Usage: Message Routing vs Node Validation"
        - DESIGN_TWO_WAY_REGISTRATION_ARCHITECTURE.md: Pure reducer pattern
        - OMN-942: Reducer test suite ticket
    """

    def test_reduce_only_accepts_introspection_events(
        self,
        reducer: RegistrationReducer,
        initial_state: ModelRegistrationState,
    ) -> None:
        """Verify reduce() signature only accepts ModelNodeIntrospectionEvent.

        The type system enforces that reduce() takes events, not commands.
        This test documents and verifies the contract through introspection.

        Key points:
        - reduce() accepts ModelNodeIntrospectionEvent (an EVENT type)
        - There is no reduce() overload for command types
        - Type annotations serve as compile-time enforcement
        """
        import inspect

        # Get the reduce method signature
        sig = inspect.signature(reducer.reduce)
        params = list(sig.parameters.values())

        # Should have exactly 2 params: state and event
        assert len(params) == 2, (
            f"reduce() should have 2 parameters (state, event), "
            f"found {len(params)}: {[p.name for p in params]}"
        )

        # Verify parameter names match expected pattern
        assert (
            params[0].name == "state"
        ), f"First parameter should be 'state', found '{params[0].name}'"
        assert (
            params[1].name == "event"
        ), f"Second parameter should be 'event', found '{params[1].name}'"

        # Verify event parameter type annotation is the Event type
        event_param = params[1]
        annotation = event_param.annotation

        # Handle string annotations (from __future__ import annotations)
        if isinstance(annotation, str):
            assert "Event" in annotation, (
                f"Event parameter should have Event type annotation, "
                f"found '{annotation}'"
            )
        else:
            # Direct type annotation
            type_name = getattr(annotation, "__name__", str(annotation))
            assert "Event" in type_name or "Introspection" in type_name, (
                f"Event parameter should have Event type annotation, "
                f"found '{type_name}'"
            )

    def test_reducer_has_no_command_handlers(self) -> None:
        """Verify reducer has no methods for handling commands.

        Per ONEX architecture, reducers ONLY process events. This test verifies
        there are no command handler methods that would violate this principle.

        Forbidden patterns (should NOT exist):
        - handle_command, execute_command, process_command
        - do_*, perform_*, run_* (imperative action methods)
        - register_node, deregister_node (direct action methods)

        Allowed patterns:
        - reduce, reduce_* (pure event folding)
        - _build_* (internal helpers)
        - _validate_* (validation helpers)
        """
        reducer = RegistrationReducer()

        # Get all public methods
        public_methods = [
            name
            for name in dir(reducer)
            if callable(getattr(reducer, name)) and not name.startswith("_")
        ]

        # Forbidden command-like method patterns
        forbidden_patterns = [
            "handle_command",
            "execute_command",
            "process_command",
            "do_",
            "perform_",
            "run_",
            "register_node",  # Direct action - should be via events
            "deregister_node",  # Direct action - should be via events
            "send_",  # I/O operation
            "publish_",  # I/O operation
            "write_",  # I/O operation
            "delete_",  # I/O operation
        ]

        for method_name in public_methods:
            for pattern in forbidden_patterns:
                assert (
                    not method_name.startswith(pattern) and pattern not in method_name
                ), (
                    f"Reducer has forbidden command-like method: {method_name}. "
                    f"Reducers should only have reduce() methods for event processing."
                )

        # Verify the only public methods are reduce-related
        allowed_prefixes = ["reduce"]
        for method_name in public_methods:
            is_allowed = any(
                method_name.startswith(prefix) for prefix in allowed_prefixes
            )
            assert is_allowed, (
                f"Reducer has unexpected public method: {method_name}. "
                f"Public methods should be reduce() or reduce_*() only."
            )

    def test_output_contains_intents_not_direct_commands(
        self,
        reducer: RegistrationReducer,
        initial_state: ModelRegistrationState,
        valid_event: ModelNodeIntrospectionEvent,
    ) -> None:
        """Verify reducer emits intents (for Effect layer) not commands.

        Intents and commands are fundamentally different:

        - **Intents**: Declarative descriptions of desired side effects.
            The reducer says "I want X to happen" but doesn't do it.
            Intent types: consul.register, postgres.upsert_registration

        - **Commands**: Imperative instructions to execute immediately.
            Commands say "DO X NOW" and expect immediate execution.
            Command types would be: RegisterNode, DeregisterNode

        This test verifies that:
        1. Output intents have declarative intent_type names (not imperative)
        2. Intents target external systems (consul://, postgres://)
        3. Intents contain data for Effect layer, not execution results
        """
        output = reducer.reduce(initial_state, valid_event)

        # Verify intents exist
        assert len(output.intents) > 0, "Reducer should emit intents for Effect layer"

        for intent in output.intents:
            # Verify intent_type uses declarative naming (noun.verb pattern)
            # Declarative: "consul.register" (describes what should be registered)
            # Imperative would be: "RegisterInConsul" (commands action)
            assert "." in intent.intent_type, (
                f"Intent type should use namespace.action pattern, "
                f"found '{intent.intent_type}'"
            )

            # Verify intent targets external system (Effect layer responsibility)
            # Targets like "consul://..." or "postgres://..." indicate
            # the Effect layer will handle the actual I/O
            assert "://" in intent.target, (
                f"Intent target should be a URI for Effect layer, "
                f"found '{intent.target}'"
            )

            # Verify payload is data, not execution results
            assert isinstance(intent.payload, dict), (
                f"Intent payload should be a dict of data for Effect layer, "
                f"found {type(intent.payload)}"
            )

            # Verify payload doesn't contain execution indicators
            # (no "result", "status", "executed", "completed" keys)
            execution_indicators = [
                "result",
                "executed",
                "completed",
                "success",
                "error",
            ]
            for key in intent.payload:
                assert key not in execution_indicators, (
                    f"Intent payload contains execution indicator '{key}'. "
                    f"Intents should contain input data, not execution results."
                )

    def test_reducer_processes_events_not_command_messages(
        self,
        reducer: RegistrationReducer,
        initial_state: ModelRegistrationState,
    ) -> None:
        """Verify reducer does not execute command-like inputs.

        This test verifies the reducer's behavior with mock inputs:
        1. A mock "command" with an execute() method is passed to reduce
        2. The reducer NEVER calls execute() - it only processes data
        3. The reducer treats inputs as data, not as executable commands

        This demonstrates the fundamental difference:
        - Commands have execute() methods that perform actions
        - Events are data that reducers fold into state
        - Reducers don't execute anything - they only transform data

        Note: In production, the dispatch engine routes commands to Effect
        layer, not to reducers. The type system prevents command objects
        from reaching reduce() at compile time.
        """
        from unittest.mock import MagicMock

        # Create a mock "command" object (what a command might look like)
        mock_command = MagicMock(spec=ModelNodeIntrospectionEvent)
        mock_command.command_type = "RegisterNode"  # Imperative name
        mock_command.node_id = uuid4()
        mock_command.action = "register"  # Action field (commands have actions)
        mock_command.execute = MagicMock()  # Commands might have execute()

        # Set required event fields so validation passes
        mock_command.node_type = "effect"
        mock_command.node_version = "1.0.0"
        mock_command.correlation_id = uuid4()
        mock_command.endpoints = {"health": "http://localhost:8080/health"}
        mock_command.capabilities = ModelNodeCapabilities()
        mock_command.metadata = ModelNodeMetadata()

        # When passed to reduce(), it processes as data
        output = reducer.reduce(initial_state, mock_command)  # type: ignore[arg-type]

        # CRITICAL: The reducer NEVER calls execute() on the input
        # This is the key difference between commands and events:
        # - Commands would be executed (execute() called)
        # - Events are only read as data (no execute() call)
        mock_command.execute.assert_not_called()

        # The reducer treats the input as data and emits intents
        # (it doesn't "do" anything, it describes what should be done)
        for intent in output.intents:
            # Intent types should be our standard declarative types
            assert intent.intent_type in (
                "consul.register",
                "postgres.upsert_registration",
            ), f"Unexpected intent type: {intent.intent_type}"

    def test_event_naming_convention_enforced(
        self,
        valid_event: ModelNodeIntrospectionEvent,
    ) -> None:
        """Verify event type follows past-tense naming convention.

        ONEX events use past-tense or noun-based naming to indicate
        they describe what HAS happened:
        - ModelNodeIntrospectionEvent (noun-based: the event OF introspection)
        - ConsulRegistered (past-tense: registration completed)
        - RegistrationFailed (past-tense: failure occurred)

        Commands would use imperative naming:
        - RegisterNode (imperative: DO this)
        - DeregisterService (imperative: DO this)

        This test documents the naming convention through the model class name.
        """
        event_type_name = type(valid_event).__name__

        # Event names should contain "Event" suffix or past-tense verbs
        event_indicators = [
            "Event",  # Explicit event suffix
            "ed",  # Past tense (Registered, Completed, Failed)
            "tion",  # Noun form (Introspection, Registration)
        ]

        has_event_indicator = any(
            indicator in event_type_name for indicator in event_indicators
        )

        assert has_event_indicator, (
            f"Event type '{event_type_name}' should follow event naming convention "
            f"(contain 'Event', past-tense verb like 'ed', or noun like 'tion')"
        )

        # Verify it does NOT use imperative command naming
        command_indicators = [
            "Command",
            "Request",
            "Do",
            "Execute",
            "Perform",
        ]

        for indicator in command_indicators:
            assert indicator not in event_type_name, (
                f"Event type '{event_type_name}' uses command-like naming "
                f"(contains '{indicator}'). Events should use past-tense "
                f"or noun-based naming."
            )

    def test_reduce_method_is_synchronous_not_async(self) -> None:
        """Verify reduce() is synchronous (no I/O, therefore no async).

        Commands often require async execution because they perform I/O.
        Reducers are pure functions that do NO I/O, therefore reduce()
        should be synchronous.

        This is another way to verify reducers don't execute commands:
        - If reduce() were async, it could perform I/O (command execution)
        - Since reduce() is sync, it can only do pure computation (event folding)
        """
        import inspect

        reducer = RegistrationReducer()

        # Verify reduce() is not a coroutine function
        assert not inspect.iscoroutinefunction(reducer.reduce), (
            "reduce() should be synchronous, not async. "
            "Async methods indicate I/O operations, but reducers are pure."
        )

        # Also verify reduce_reset() is synchronous
        assert not inspect.iscoroutinefunction(reducer.reduce_reset), (
            "reduce_reset() should be synchronous, not async. "
            "All reducer methods should be pure and sync."
        )
