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
        """Test that health check is omitted if no endpoint."""
        output = reducer.reduce(initial_state, event_without_health_endpoint)

        consul_intent = next(
            (i for i in output.intents if i.intent_type == "consul.register"), None
        )
        assert consul_intent is not None

        # health_check key should not exist
        assert "health_check" not in consul_intent.payload

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
        )

        output = reducer.reduce(initial_state, event)

        assert len(output.intents) == 2
        assert output.result.status == "pending"

    def test_reduce_generates_event_id_when_correlation_id_missing(
        self,
        reducer: RegistrationReducer,
        initial_state: ModelRegistrationState,
    ) -> None:
        """Test that event_id is generated when correlation_id is None."""
        event = ModelNodeIntrospectionEvent(
            node_id=uuid4(),
            node_type="effect",
            node_version="1.0.0",
            endpoints={"health": "http://localhost:8080/health"},
            correlation_id=None,
        )

        output = reducer.reduce(initial_state, event)

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
        assert len(instance_vars) == 0, (
            f"Reducer should have no instance state, found: {instance_vars}"
        )

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
