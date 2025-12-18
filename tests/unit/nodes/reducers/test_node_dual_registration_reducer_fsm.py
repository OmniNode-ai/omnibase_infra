# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Unit tests for NodeDualRegistrationReducer FSM state transitions.

This test suite validates FSM state transitions in the dual registration
reducer workflow per OMN-889 requirements.

Pure Reducer Architecture:
    The reducer is now a PURE component that emits typed intents instead of
    performing I/O. Tests verify:
    - Correct intent types are emitted (ModelConsulRegisterIntent, ModelPostgresUpsertRegistrationIntent)
    - Intent fields contain correct values
    - FSM state transitions work correctly
    - Validation logic is correct

FSM States Tested:
    - idle (initial): Waiting for introspection events
    - receiving_introspection: Received NODE_INTROSPECTION event
    - validating_payload: Validating introspection data structure
    - registering_parallel: Building typed registration intents
    - aggregating_results: Combining intent emission results
    - registration_complete: Both intents emitted (terminal)
    - partial_failure: One intent could not be built (terminal)
    - registration_failed: No intents emitted (terminal)

Test Organization:
    - TestFSMHappyPath: Successful state transitions
    - TestFSMValidation: Validation failure scenarios
    - TestFSMTerminalStates: Terminal state behavior
    - TestFSMResetCycles: State reset and full workflow cycles
    - TestFSMErrorHandling: Error handling during transitions
    - TestFSMMetrics: Metrics tracking during transitions
    - TestIntentEmission: Intent emission verification

Coverage Goals:
    - All FSM state transitions covered
    - All terminal states verified
    - Reset behavior validated
    - Full workflow cycles tested
    - Intent emission verified
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import Literal
from unittest.mock import MagicMock, patch
from uuid import UUID, uuid4

import pytest
from omnibase_core.models.intents import (
    ModelConsulRegisterIntent,
    ModelPostgresUpsertRegistrationIntent,
)

from omnibase_infra.models.registration import (
    ModelNodeCapabilities,
    ModelNodeIntrospectionEvent,
    ModelNodeMetadata,
)
from omnibase_infra.nodes.reducers.node_dual_registration_reducer import (
    EnumFSMState,
    EnumFSMTrigger,
    ModelReducerMetrics,
    NodeDualRegistrationReducer,
)

# -----------------------------------------------------------------------------
# Fixtures
# -----------------------------------------------------------------------------


@pytest.fixture
def sample_fsm_contract_yaml() -> str:
    """Create sample FSM contract YAML content.

    Note: This fixture includes all transitions from the production contract
    (contracts/fsm/dual_registration_reducer_fsm.yaml) because the reducer
    builds its transition map dynamically from the contract at initialization.

    Field names must match the Pydantic models in model_fsm_contract.py:
    - states use 'state_name' (not 'name')
    - error_handling uses 'default_error_state' (not 'default_action')
    """
    return """
contract_version: "1.0.0"
name: "dual_registration_reducer_fsm"
description: "FSM for dual registration workflow"

state_transitions:
  initial_state: "idle"
  states:
    - state_name: "idle"
      description: "Waiting for introspection events"
    - state_name: "receiving_introspection"
      description: "Parsing NODE_INTROSPECTION event"
    - state_name: "validating_payload"
      description: "Validating event structure"
    - state_name: "registering_parallel"
      description: "Building typed registration intents"
    - state_name: "aggregating_results"
      description: "Combining intent emission outcomes"
    - state_name: "registration_complete"
      description: "Both intents emitted"
    - state_name: "partial_failure"
      description: "One intent could not be built"
    - state_name: "registration_failed"
      description: "No intents emitted"
  transitions:
    - from: "idle"
      to: "receiving_introspection"
      trigger: "introspection_event_received"
    - from: "receiving_introspection"
      to: "validating_payload"
      trigger: "event_parsed"
    - from: "validating_payload"
      to: "registering_parallel"
      trigger: "validation_passed"
    - from: "validating_payload"
      to: "registration_failed"
      trigger: "validation_failed"
    - from: "registering_parallel"
      to: "aggregating_results"
      trigger: "registration_attempts_complete"
    - from: "aggregating_results"
      to: "registration_complete"
      trigger: "all_backends_succeeded"
    - from: "aggregating_results"
      to: "partial_failure"
      trigger: "partial_success"
    - from: "aggregating_results"
      to: "registration_failed"
      trigger: "all_backends_failed"
    - from: "registration_complete"
      to: "idle"
      trigger: "result_emitted"
    - from: "partial_failure"
      to: "idle"
      trigger: "partial_result_emitted"
    - from: "registration_failed"
      to: "idle"
      trigger: "failure_result_emitted"

error_handling:
  default_error_state: "registration_failed"
"""


@pytest.fixture
def fsm_contract_path(sample_fsm_contract_yaml: str, tmp_path: Path) -> Path:
    """Create temporary FSM contract file.

    Uses pytest's tmp_path fixture for automatic cleanup.
    """
    contract_file = tmp_path / "fsm_contract.yaml"
    contract_file.write_text(sample_fsm_contract_yaml)
    return contract_file


@pytest.fixture
def sample_introspection_event() -> ModelNodeIntrospectionEvent:
    """Create sample ModelNodeIntrospectionEvent for testing.

    Returns:
        Valid introspection event with all required fields.
    """
    return ModelNodeIntrospectionEvent(
        node_id=uuid4(),
        node_type="effect",
        node_version="1.0.0",
        capabilities=ModelNodeCapabilities(postgres=True, read=True, write=True),
        endpoints={"health": "http://localhost:8080/health"},
        node_role="adapter",
        metadata=ModelNodeMetadata(environment="test"),
        correlation_id=uuid4(),
        network_id="test-network",
        deployment_id="test-deployment",
        epoch=1,
        timestamp=datetime.now(UTC),
    )


@pytest.fixture
def dual_registration_reducer(
    fsm_contract_path: Path,
) -> NodeDualRegistrationReducer:
    """Create NodeDualRegistrationReducer for testing.

    Pure Reducer Design:
        The reducer no longer accepts ConsulHandler or DbAdapter because
        it emits typed intents instead of performing I/O operations.

    Args:
        fsm_contract_path: Path to FSM contract YAML.

    Returns:
        Configured reducer instance (not yet initialized).
    """
    return NodeDualRegistrationReducer(
        fsm_contract_path=fsm_contract_path,
    )


def create_introspection_event(
    node_type: Literal["effect", "compute", "reducer", "orchestrator"] = "effect",
    node_id: UUID | None = None,
    correlation_id: UUID | None = None,
) -> ModelNodeIntrospectionEvent:
    """Helper factory for creating introspection events.

    Args:
        node_type: Type of node (default: "effect").
        node_id: Optional node ID (generates if not provided).
        correlation_id: Optional correlation ID (generates if not provided).

    Returns:
        Configured ModelNodeIntrospectionEvent instance.
    """
    return ModelNodeIntrospectionEvent(
        node_id=node_id or uuid4(),
        node_type=node_type,
        node_version="1.0.0",
        capabilities=ModelNodeCapabilities(postgres=True, read=True),
        endpoints={"health": "http://localhost:8080/health"},
        correlation_id=correlation_id or uuid4(),
    )


# -----------------------------------------------------------------------------
# Happy Path Tests
# -----------------------------------------------------------------------------


@pytest.mark.unit
@pytest.mark.asyncio
class TestFSMHappyPath:
    """Tests for successful FSM state transitions."""

    async def test_fsm_starts_in_idle_state(
        self,
        dual_registration_reducer: NodeDualRegistrationReducer,
    ) -> None:
        """Test that FSM starts in idle state."""
        # Before initialization
        assert dual_registration_reducer.current_state == EnumFSMState.IDLE

    async def test_fsm_idle_to_receiving_on_event(
        self,
        dual_registration_reducer: NodeDualRegistrationReducer,
    ) -> None:
        """Test transition from idle to receiving_introspection when event received."""
        await dual_registration_reducer.initialize()

        # Manually trigger the transition
        await dual_registration_reducer._transition(
            EnumFSMTrigger.INTROSPECTION_EVENT_RECEIVED
        )

        assert (
            dual_registration_reducer.current_state
            == EnumFSMState.RECEIVING_INTROSPECTION
        )

    async def test_fsm_receiving_to_validating(
        self,
        dual_registration_reducer: NodeDualRegistrationReducer,
    ) -> None:
        """Test transition from receiving_introspection to validating_payload."""
        await dual_registration_reducer.initialize()

        # Set up state chain
        await dual_registration_reducer._transition(
            EnumFSMTrigger.INTROSPECTION_EVENT_RECEIVED
        )
        await dual_registration_reducer._transition(EnumFSMTrigger.EVENT_PARSED)

        assert (
            dual_registration_reducer.current_state == EnumFSMState.VALIDATING_PAYLOAD
        )

    async def test_fsm_validating_to_registering_on_valid_payload(
        self,
        dual_registration_reducer: NodeDualRegistrationReducer,
    ) -> None:
        """Test transition from validating_payload to registering_parallel on valid payload."""
        await dual_registration_reducer.initialize()

        # Set up state chain
        await dual_registration_reducer._transition(
            EnumFSMTrigger.INTROSPECTION_EVENT_RECEIVED
        )
        await dual_registration_reducer._transition(EnumFSMTrigger.EVENT_PARSED)
        await dual_registration_reducer._transition(EnumFSMTrigger.VALIDATION_PASSED)

        assert (
            dual_registration_reducer.current_state == EnumFSMState.REGISTERING_PARALLEL
        )

    async def test_fsm_registering_to_aggregating(
        self,
        dual_registration_reducer: NodeDualRegistrationReducer,
    ) -> None:
        """Test transition from registering_parallel to aggregating_results."""
        await dual_registration_reducer.initialize()

        # Set up state chain
        await dual_registration_reducer._transition(
            EnumFSMTrigger.INTROSPECTION_EVENT_RECEIVED
        )
        await dual_registration_reducer._transition(EnumFSMTrigger.EVENT_PARSED)
        await dual_registration_reducer._transition(EnumFSMTrigger.VALIDATION_PASSED)
        await dual_registration_reducer._transition(
            EnumFSMTrigger.REGISTRATION_ATTEMPTS_COMPLETE
        )

        assert (
            dual_registration_reducer.current_state == EnumFSMState.AGGREGATING_RESULTS
        )

    async def test_fsm_aggregating_to_complete_on_success(
        self,
        dual_registration_reducer: NodeDualRegistrationReducer,
    ) -> None:
        """Test transition from aggregating_results to registration_complete on success."""
        await dual_registration_reducer.initialize()

        # Set up state chain to aggregating_results
        await dual_registration_reducer._transition(
            EnumFSMTrigger.INTROSPECTION_EVENT_RECEIVED
        )
        await dual_registration_reducer._transition(EnumFSMTrigger.EVENT_PARSED)
        await dual_registration_reducer._transition(EnumFSMTrigger.VALIDATION_PASSED)
        await dual_registration_reducer._transition(
            EnumFSMTrigger.REGISTRATION_ATTEMPTS_COMPLETE
        )
        await dual_registration_reducer._transition(
            EnumFSMTrigger.ALL_BACKENDS_SUCCEEDED
        )

        assert (
            dual_registration_reducer.current_state
            == EnumFSMState.REGISTRATION_COMPLETE
        )

    async def test_fsm_full_success_workflow(
        self,
        dual_registration_reducer: NodeDualRegistrationReducer,
        sample_introspection_event: ModelNodeIntrospectionEvent,
    ) -> None:
        """Test complete workflow with both intents emitted successfully."""
        await dual_registration_reducer.initialize()

        # Execute full workflow
        result = await dual_registration_reducer.execute(sample_introspection_event)

        # Verify success result - both intents emitted
        assert result.status == "success"
        assert result.consul_intent_emitted is True
        assert result.postgres_intent_emitted is True
        assert result.validation_error is None
        assert len(result.intents) == 2

        # FSM should have returned to idle after RESULT_EMITTED
        assert dual_registration_reducer.current_state == EnumFSMState.IDLE


# -----------------------------------------------------------------------------
# Intent Emission Tests
# -----------------------------------------------------------------------------


@pytest.mark.unit
@pytest.mark.asyncio
class TestIntentEmission:
    """Tests for verifying typed intent emission."""

    async def test_emits_both_intent_types(
        self,
        dual_registration_reducer: NodeDualRegistrationReducer,
        sample_introspection_event: ModelNodeIntrospectionEvent,
    ) -> None:
        """Test that reducer emits both Consul and PostgreSQL intents."""
        await dual_registration_reducer.initialize()

        result = await dual_registration_reducer.execute(sample_introspection_event)

        assert len(result.intents) == 2
        intent_types = {type(intent) for intent in result.intents}
        assert ModelConsulRegisterIntent in intent_types
        assert ModelPostgresUpsertRegistrationIntent in intent_types

    async def test_consul_intent_has_correct_fields(
        self,
        dual_registration_reducer: NodeDualRegistrationReducer,
        sample_introspection_event: ModelNodeIntrospectionEvent,
    ) -> None:
        """Test that Consul intent has correct field values."""
        await dual_registration_reducer.initialize()

        result = await dual_registration_reducer.execute(sample_introspection_event)

        # Find the Consul intent
        consul_intent = next(
            (i for i in result.intents if isinstance(i, ModelConsulRegisterIntent)),
            None,
        )
        assert consul_intent is not None

        # Verify service ID format
        expected_service_id = (
            f"node-{sample_introspection_event.node_type}-"
            f"{sample_introspection_event.node_id}"
        )
        assert consul_intent.service_id == expected_service_id

        # Verify service name format
        expected_service_name = f"onex-{sample_introspection_event.node_type}"
        assert consul_intent.service_name == expected_service_name

        # Verify tags contain node type and version
        assert f"node_type:{sample_introspection_event.node_type}" in consul_intent.tags
        assert (
            f"node_version:{sample_introspection_event.node_version}"
            in consul_intent.tags
        )

        # Verify correlation ID is propagated
        assert consul_intent.correlation_id is not None

    async def test_postgres_intent_has_correct_fields(
        self,
        dual_registration_reducer: NodeDualRegistrationReducer,
        sample_introspection_event: ModelNodeIntrospectionEvent,
    ) -> None:
        """Test that PostgreSQL intent has correct field values."""
        await dual_registration_reducer.initialize()

        result = await dual_registration_reducer.execute(sample_introspection_event)

        # Find the PostgreSQL intent
        postgres_intent = next(
            (
                i
                for i in result.intents
                if isinstance(i, ModelPostgresUpsertRegistrationIntent)
            ),
            None,
        )
        assert postgres_intent is not None

        # Verify record contains correct node data
        record = postgres_intent.record
        assert record.node_id == sample_introspection_event.node_id
        assert record.node_type == sample_introspection_event.node_type
        assert record.node_version == sample_introspection_event.node_version

        # Verify correlation ID is propagated
        assert postgres_intent.correlation_id is not None

    async def test_intent_correlation_id_matches_request(
        self,
        dual_registration_reducer: NodeDualRegistrationReducer,
    ) -> None:
        """Test that intent correlation IDs match the request correlation ID."""
        await dual_registration_reducer.initialize()

        correlation_id = uuid4()
        event = create_introspection_event(correlation_id=correlation_id)

        result = await dual_registration_reducer.execute(event, correlation_id)

        # All intents should have the same correlation ID
        for intent in result.intents:
            assert intent.correlation_id == correlation_id

        # Result should also have the same correlation ID
        assert result.correlation_id == correlation_id

    async def test_consul_intent_has_health_check_when_endpoint_provided(
        self,
        dual_registration_reducer: NodeDualRegistrationReducer,
    ) -> None:
        """Test that Consul intent includes health check when endpoint is provided."""
        await dual_registration_reducer.initialize()

        event = ModelNodeIntrospectionEvent(
            node_id=uuid4(),
            node_type="effect",
            node_version="1.0.0",
            capabilities=ModelNodeCapabilities(),
            endpoints={"health": "http://localhost:8080/health"},
        )

        result = await dual_registration_reducer.execute(event)

        consul_intent = next(
            (i for i in result.intents if isinstance(i, ModelConsulRegisterIntent)),
            None,
        )
        assert consul_intent is not None
        assert consul_intent.health_check is not None
        assert consul_intent.health_check["HTTP"] == "http://localhost:8080/health"

    async def test_output_has_correct_node_id(
        self,
        dual_registration_reducer: NodeDualRegistrationReducer,
    ) -> None:
        """Test that output node_id matches event node_id."""
        await dual_registration_reducer.initialize()

        test_node_id = uuid4()
        event = ModelNodeIntrospectionEvent(
            node_id=test_node_id,
            node_type="effect",
            node_version="1.0.0",
            capabilities=ModelNodeCapabilities(),
            endpoints={"health": "http://localhost:8080/health"},
        )

        result = await dual_registration_reducer.execute(event)

        assert result.node_id == test_node_id

    async def test_processing_time_is_recorded(
        self,
        dual_registration_reducer: NodeDualRegistrationReducer,
        sample_introspection_event: ModelNodeIntrospectionEvent,
    ) -> None:
        """Test that processing time is recorded in output."""
        await dual_registration_reducer.initialize()

        result = await dual_registration_reducer.execute(sample_introspection_event)

        assert result.processing_time_ms >= 0.0


# -----------------------------------------------------------------------------
# Validation Tests
# -----------------------------------------------------------------------------


@pytest.mark.unit
@pytest.mark.asyncio
class TestFSMValidation:
    """Tests for FSM validation state transitions."""

    async def test_fsm_validation_fails_invalid_node_type(
        self,
        dual_registration_reducer: NodeDualRegistrationReducer,
    ) -> None:
        """Test that validation fails for invalid node_type."""
        await dual_registration_reducer.initialize()

        # Create event with valid Literal type but test internal validation
        event = create_introspection_event(node_type="effect")

        # Patch the validation to simulate invalid node_type (since Pydantic
        # prevents invalid literals at construction)
        with patch.object(
            dual_registration_reducer,
            "_validate_payload",
            return_value=(False, "node_type is invalid"),
        ):
            result = await dual_registration_reducer.execute(event)

        assert result.status == "failed"
        assert len(result.intents) == 0

    async def test_fsm_validation_passes_with_required_fields(
        self,
        dual_registration_reducer: NodeDualRegistrationReducer,
        sample_introspection_event: ModelNodeIntrospectionEvent,
    ) -> None:
        """Test that validation passes with all required fields present."""
        await dual_registration_reducer.initialize()

        # Verify internal validation method directly
        correlation_id = uuid4()
        validation_passed, error_message = dual_registration_reducer._validate_payload(
            sample_introspection_event,
            correlation_id,
        )

        assert validation_passed is True
        assert error_message == ""

    async def test_fsm_validation_fails_with_null_node_id(
        self,
        dual_registration_reducer: NodeDualRegistrationReducer,
    ) -> None:
        """Test that validation fails when node_id is None.

        Note: ModelNodeIntrospectionEvent requires node_id via Pydantic,
        so we test the internal validation logic by mocking.
        """
        await dual_registration_reducer.initialize()

        # Create mock event with None node_id
        mock_event = MagicMock(spec=ModelNodeIntrospectionEvent)
        mock_event.node_id = None
        mock_event.node_type = "effect"

        validation_passed, error_message = dual_registration_reducer._validate_payload(
            mock_event, uuid4()
        )
        assert validation_passed is False
        assert "node_id" in error_message

    async def test_pydantic_rejects_invalid_node_type_value(self) -> None:
        """Verify Pydantic Literal type rejects unsupported node_type values.

        node_type validation is enforced at model construction via the
        Literal["effect", "compute", "reducer", "orchestrator"] type annotation.
        The reducer's _validate_payload method no longer performs this validation
        since Pydantic handles it automatically.

        This test documents that invalid node_type values are rejected before
        they can reach the reducer.
        """
        from pydantic import ValidationError

        with pytest.raises(ValidationError) as exc_info:
            ModelNodeIntrospectionEvent(
                node_id=uuid4(),
                node_type="invalid_type",  # Not in Literal
                node_version="1.0.0",
                capabilities=ModelNodeCapabilities(),
                endpoints={"health": "http://localhost:8080/health"},
            )

        # Verify the error mentions node_type
        assert "node_type" in str(exc_info.value)

    async def test_fsm_validation_accepts_all_valid_node_types(
        self,
        dual_registration_reducer: NodeDualRegistrationReducer,
    ) -> None:
        """Test that validation accepts all four valid node types."""
        await dual_registration_reducer.initialize()

        valid_types: list[Literal["effect", "compute", "reducer", "orchestrator"]] = [
            "effect",
            "compute",
            "reducer",
            "orchestrator",
        ]

        for node_type in valid_types:
            event = create_introspection_event(node_type=node_type)
            validation_passed, error_message = (
                dual_registration_reducer._validate_payload(event, uuid4())
            )
            assert validation_passed is True, (
                f"Validation should pass for node_type: {node_type}"
            )
            assert error_message == ""


# -----------------------------------------------------------------------------
# Terminal State Tests
# -----------------------------------------------------------------------------


@pytest.mark.unit
@pytest.mark.asyncio
class TestFSMTerminalStates:
    """Tests for FSM terminal state behavior."""

    async def test_fsm_registration_complete_is_terminal(
        self,
        dual_registration_reducer: NodeDualRegistrationReducer,
    ) -> None:
        """Test that registration_complete is a terminal state.

        Terminal states only accept specific triggers to transition back to idle.
        """
        await dual_registration_reducer.initialize()

        # Reach registration_complete state
        await dual_registration_reducer._transition(
            EnumFSMTrigger.INTROSPECTION_EVENT_RECEIVED
        )
        await dual_registration_reducer._transition(EnumFSMTrigger.EVENT_PARSED)
        await dual_registration_reducer._transition(EnumFSMTrigger.VALIDATION_PASSED)
        await dual_registration_reducer._transition(
            EnumFSMTrigger.REGISTRATION_ATTEMPTS_COMPLETE
        )
        await dual_registration_reducer._transition(
            EnumFSMTrigger.ALL_BACKENDS_SUCCEEDED
        )

        assert (
            dual_registration_reducer.current_state
            == EnumFSMState.REGISTRATION_COMPLETE
        )

        # Only RESULT_EMITTED should be valid from this state
        await dual_registration_reducer._transition(EnumFSMTrigger.RESULT_EMITTED)
        assert dual_registration_reducer.current_state == EnumFSMState.IDLE

    async def test_fsm_partial_failure_is_terminal(
        self,
        dual_registration_reducer: NodeDualRegistrationReducer,
    ) -> None:
        """Test that partial_failure is a terminal state."""
        await dual_registration_reducer.initialize()

        # Reach partial_failure state
        await dual_registration_reducer._transition(
            EnumFSMTrigger.INTROSPECTION_EVENT_RECEIVED
        )
        await dual_registration_reducer._transition(EnumFSMTrigger.EVENT_PARSED)
        await dual_registration_reducer._transition(EnumFSMTrigger.VALIDATION_PASSED)
        await dual_registration_reducer._transition(
            EnumFSMTrigger.REGISTRATION_ATTEMPTS_COMPLETE
        )
        await dual_registration_reducer._transition(EnumFSMTrigger.PARTIAL_SUCCESS)

        assert dual_registration_reducer.current_state == EnumFSMState.PARTIAL_FAILURE

        # Only PARTIAL_RESULT_EMITTED should be valid from this state
        await dual_registration_reducer._transition(
            EnumFSMTrigger.PARTIAL_RESULT_EMITTED
        )
        assert dual_registration_reducer.current_state == EnumFSMState.IDLE

    async def test_fsm_registration_failed_is_terminal(
        self,
        dual_registration_reducer: NodeDualRegistrationReducer,
    ) -> None:
        """Test that registration_failed is a terminal state.

        Unlike success/partial, registration_failed transitions back to idle
        via the FAILURE_RESULT_EMITTED trigger.
        """
        await dual_registration_reducer.initialize()

        # Reach registration_failed via validation failure path
        await dual_registration_reducer._transition(
            EnumFSMTrigger.INTROSPECTION_EVENT_RECEIVED
        )
        await dual_registration_reducer._transition(EnumFSMTrigger.EVENT_PARSED)
        await dual_registration_reducer._transition(EnumFSMTrigger.VALIDATION_FAILED)

        assert (
            dual_registration_reducer.current_state == EnumFSMState.REGISTRATION_FAILED
        )

    async def test_fsm_registration_failed_from_aggregation(
        self,
        dual_registration_reducer: NodeDualRegistrationReducer,
    ) -> None:
        """Test transition to registration_failed from aggregating_results."""
        await dual_registration_reducer.initialize()

        # Reach registration_failed via all_backends_failed path
        await dual_registration_reducer._transition(
            EnumFSMTrigger.INTROSPECTION_EVENT_RECEIVED
        )
        await dual_registration_reducer._transition(EnumFSMTrigger.EVENT_PARSED)
        await dual_registration_reducer._transition(EnumFSMTrigger.VALIDATION_PASSED)
        await dual_registration_reducer._transition(
            EnumFSMTrigger.REGISTRATION_ATTEMPTS_COMPLETE
        )
        await dual_registration_reducer._transition(EnumFSMTrigger.ALL_BACKENDS_FAILED)

        assert (
            dual_registration_reducer.current_state == EnumFSMState.REGISTRATION_FAILED
        )


# -----------------------------------------------------------------------------
# Reset/Cycle Tests
# -----------------------------------------------------------------------------


@pytest.mark.unit
@pytest.mark.asyncio
class TestFSMResetCycles:
    """Tests for FSM reset and full workflow cycles."""

    async def test_fsm_returns_to_idle_after_complete(
        self,
        dual_registration_reducer: NodeDualRegistrationReducer,
        sample_introspection_event: ModelNodeIntrospectionEvent,
    ) -> None:
        """Test that FSM returns to idle after successful completion."""
        await dual_registration_reducer.initialize()

        # Execute workflow
        result = await dual_registration_reducer.execute(sample_introspection_event)

        # Verify success and idle state
        assert result.status == "success"
        assert dual_registration_reducer.current_state == EnumFSMState.IDLE

    async def test_fsm_full_workflow_cycle(
        self,
        dual_registration_reducer: NodeDualRegistrationReducer,
        sample_introspection_event: ModelNodeIntrospectionEvent,
    ) -> None:
        """Test complete end-to-end FSM workflow cycle."""
        await dual_registration_reducer.initialize()

        # First cycle
        result1 = await dual_registration_reducer.execute(sample_introspection_event)
        assert result1.status == "success"
        assert len(result1.intents) == 2
        assert dual_registration_reducer.current_state == EnumFSMState.IDLE

        # Second cycle (FSM should be reusable)
        new_event = create_introspection_event(node_type="compute")
        result2 = await dual_registration_reducer.execute(new_event)
        assert result2.status == "success"
        assert len(result2.intents) == 2
        assert dual_registration_reducer.current_state == EnumFSMState.IDLE

        # Metrics should reflect both intent emissions
        assert dual_registration_reducer.metrics.total_registrations == 2
        assert dual_registration_reducer.metrics.success_count == 2

    async def test_fsm_multiple_cycles_with_mixed_results(
        self,
        dual_registration_reducer: NodeDualRegistrationReducer,
    ) -> None:
        """Test multiple cycles with varying success/failure outcomes."""
        await dual_registration_reducer.initialize()

        # Cycle 1: Success
        event1 = create_introspection_event(node_type="effect")
        result1 = await dual_registration_reducer.execute(event1)
        assert result1.status == "success"
        assert len(result1.intents) == 2
        assert dual_registration_reducer.current_state == EnumFSMState.IDLE

        # Cycle 2: Success (pure reducer always succeeds for valid input)
        event2 = create_introspection_event(node_type="compute")
        result2 = await dual_registration_reducer.execute(event2)
        assert result2.status == "success"
        assert len(result2.intents) == 2
        assert dual_registration_reducer.current_state == EnumFSMState.IDLE

        # Cycle 3: Failed (validation failure with mocked validation)
        event3 = create_introspection_event(node_type="reducer")
        with patch.object(
            dual_registration_reducer,
            "_validate_payload",
            return_value=(False, "Validation failed"),
        ):
            result3 = await dual_registration_reducer.execute(event3)
        assert result3.status == "failed"
        assert len(result3.intents) == 0
        assert dual_registration_reducer.current_state == EnumFSMState.IDLE

        # Verify metrics
        assert dual_registration_reducer.metrics.total_registrations == 3
        assert dual_registration_reducer.metrics.success_count == 2
        assert dual_registration_reducer.metrics.failure_count == 1


# -----------------------------------------------------------------------------
# Error Handling Tests
# -----------------------------------------------------------------------------


@pytest.mark.unit
@pytest.mark.asyncio
class TestFSMErrorHandling:
    """Tests for FSM error handling during transitions."""

    async def test_invalid_transition_raises_error(
        self,
        dual_registration_reducer: NodeDualRegistrationReducer,
    ) -> None:
        """Test that invalid FSM transitions raise RuntimeHostError."""
        from omnibase_infra.errors import RuntimeHostError

        await dual_registration_reducer.initialize()

        # Attempt invalid transition (VALIDATION_PASSED from idle state)
        with pytest.raises(RuntimeHostError) as exc_info:
            await dual_registration_reducer._transition(
                EnumFSMTrigger.VALIDATION_PASSED
            )

        assert "Invalid FSM transition" in str(exc_info.value)

    async def test_execute_requires_initialization(
        self,
        dual_registration_reducer: NodeDualRegistrationReducer,
        sample_introspection_event: ModelNodeIntrospectionEvent,
    ) -> None:
        """Test that execute raises error if reducer not initialized."""
        from omnibase_infra.errors import RuntimeHostError

        # Do not initialize
        with pytest.raises(RuntimeHostError) as exc_info:
            await dual_registration_reducer.execute(sample_introspection_event)

        assert "not initialized" in str(exc_info.value)

    async def test_shutdown_resets_fsm_state(
        self,
        dual_registration_reducer: NodeDualRegistrationReducer,
    ) -> None:
        """Test that shutdown resets FSM to initial state."""
        await dual_registration_reducer.initialize()

        # Advance FSM to a non-idle state
        await dual_registration_reducer._transition(
            EnumFSMTrigger.INTROSPECTION_EVENT_RECEIVED
        )
        assert (
            dual_registration_reducer.current_state
            == EnumFSMState.RECEIVING_INTROSPECTION
        )

        # Shutdown
        await dual_registration_reducer.shutdown()

        # Verify reset
        assert dual_registration_reducer.current_state == EnumFSMState.IDLE


# -----------------------------------------------------------------------------
# Metrics Tests
# -----------------------------------------------------------------------------


@pytest.mark.unit
@pytest.mark.asyncio
class TestFSMMetrics:
    """Tests for FSM metrics tracking during transitions."""

    async def test_metrics_increment_on_success(
        self,
        dual_registration_reducer: NodeDualRegistrationReducer,
        sample_introspection_event: ModelNodeIntrospectionEvent,
    ) -> None:
        """Test that success count increments on successful intent emission."""
        await dual_registration_reducer.initialize()

        # Initial metrics
        assert dual_registration_reducer.metrics.total_registrations == 0
        assert dual_registration_reducer.metrics.success_count == 0

        # Execute success
        result = await dual_registration_reducer.execute(sample_introspection_event)

        assert result.status == "success"
        assert len(result.intents) == 2
        assert dual_registration_reducer.metrics.total_registrations == 1
        assert dual_registration_reducer.metrics.success_count == 1
        assert dual_registration_reducer.metrics.failure_count == 0
        assert dual_registration_reducer.metrics.partial_count == 0

    async def test_metrics_increment_on_failure(
        self,
        dual_registration_reducer: NodeDualRegistrationReducer,
    ) -> None:
        """Test that failure count increments on validation failure."""
        await dual_registration_reducer.initialize()

        event = create_introspection_event()
        with patch.object(
            dual_registration_reducer,
            "_validate_payload",
            return_value=(False, "Validation failed"),
        ):
            result = await dual_registration_reducer.execute(event)

        assert result.status == "failed"
        assert len(result.intents) == 0
        assert dual_registration_reducer.metrics.total_registrations == 1
        assert dual_registration_reducer.metrics.failure_count == 1
        assert dual_registration_reducer.metrics.success_count == 0
        assert dual_registration_reducer.metrics.partial_count == 0


# -----------------------------------------------------------------------------
# FSM Context Tests
# -----------------------------------------------------------------------------


@pytest.mark.unit
@pytest.mark.asyncio
class TestFSMContext:
    """Tests for FSM context management during transitions."""

    async def test_context_initialized_on_execute(
        self,
        dual_registration_reducer: NodeDualRegistrationReducer,
        sample_introspection_event: ModelNodeIntrospectionEvent,
    ) -> None:
        """Test that FSM context is properly initialized on execute."""
        await dual_registration_reducer.initialize()

        result = await dual_registration_reducer.execute(sample_introspection_event)

        # Context should have been set during execution
        # After execution, context reflects final state
        assert result.correlation_id is not None

    async def test_context_correlation_id_propagated(
        self,
        dual_registration_reducer: NodeDualRegistrationReducer,
    ) -> None:
        """Test that correlation_id is propagated through context."""
        await dual_registration_reducer.initialize()

        correlation_id = uuid4()
        event = create_introspection_event(correlation_id=correlation_id)

        result = await dual_registration_reducer.execute(event, correlation_id)

        # Result should have the same correlation_id
        assert result.correlation_id == correlation_id

    async def test_context_node_info_captured(
        self,
        dual_registration_reducer: NodeDualRegistrationReducer,
        sample_introspection_event: ModelNodeIntrospectionEvent,
    ) -> None:
        """Test that node info is captured in context during execution."""
        await dual_registration_reducer.initialize()

        result = await dual_registration_reducer.execute(sample_introspection_event)

        # Result should have node_id from event
        assert result.node_id == sample_introspection_event.node_id


# -----------------------------------------------------------------------------
# Contract Loading Tests
# -----------------------------------------------------------------------------


@pytest.mark.unit
@pytest.mark.asyncio
class TestFSMContractLoading:
    """Tests for FSM contract loading during initialization."""

    async def test_contract_loaded_on_initialize(
        self,
        dual_registration_reducer: NodeDualRegistrationReducer,
    ) -> None:
        """Test that FSM contract is loaded during initialization."""
        await dual_registration_reducer.initialize()

        assert dual_registration_reducer.fsm_contract is not None
        assert (
            dual_registration_reducer.fsm_contract.name
            == "dual_registration_reducer_fsm"
        )
        assert dual_registration_reducer.fsm_contract.initial_state == "idle"

    async def test_initialization_fails_with_missing_contract(
        self,
    ) -> None:
        """Test that initialization fails if contract file is missing."""
        from omnibase_infra.errors import RuntimeHostError

        reducer = NodeDualRegistrationReducer(
            fsm_contract_path=Path("/nonexistent/path/contract.yaml"),
        )

        with pytest.raises(RuntimeHostError) as exc_info:
            await reducer.initialize()

        assert "not found" in str(exc_info.value)

    async def test_initialization_fails_with_invalid_initial_state(
        self,
        tmp_path: Path,
    ) -> None:
        """Test that initialization fails if initial state is invalid.

        Uses pytest's tmp_path fixture for automatic cleanup.
        """
        from omnibase_infra.errors import RuntimeHostError

        # Create contract with invalid initial state
        invalid_contract = """
contract_version: "1.0.0"
name: "invalid_fsm"
description: "FSM with invalid initial state"
state_transitions:
  initial_state: "invalid_state"
  states: []
  transitions: []
"""
        contract_path = tmp_path / "invalid_contract.yaml"
        contract_path.write_text(invalid_contract)

        reducer = NodeDualRegistrationReducer(
            fsm_contract_path=contract_path,
        )

        with pytest.raises(RuntimeHostError) as exc_info:
            await reducer.initialize()

        assert "Invalid initial state" in str(exc_info.value)


# -----------------------------------------------------------------------------
# Output Model Tests
# -----------------------------------------------------------------------------


@pytest.mark.unit
@pytest.mark.asyncio
class TestOutputModel:
    """Tests for output model structure and values."""

    async def test_success_output_structure(
        self,
        dual_registration_reducer: NodeDualRegistrationReducer,
        sample_introspection_event: ModelNodeIntrospectionEvent,
    ) -> None:
        """Test structure of successful intent emission output."""
        await dual_registration_reducer.initialize()

        result = await dual_registration_reducer.execute(sample_introspection_event)

        assert result.status == "success"
        assert result.consul_intent_emitted is True
        assert result.postgres_intent_emitted is True
        assert result.validation_error is None
        assert result.processing_time_ms >= 0.0
        assert result.correlation_id is not None
        assert result.node_id == sample_introspection_event.node_id
        assert len(result.intents) == 2

    async def test_failed_output_has_validation_error(
        self,
        dual_registration_reducer: NodeDualRegistrationReducer,
    ) -> None:
        """Test that failed output includes validation error message."""
        await dual_registration_reducer.initialize()

        event = create_introspection_event()
        error_message = "Custom validation error"
        with patch.object(
            dual_registration_reducer,
            "_validate_payload",
            return_value=(False, error_message),
        ):
            result = await dual_registration_reducer.execute(event)

        assert result.status == "failed"
        assert result.validation_error == error_message
        assert len(result.intents) == 0
        assert result.consul_intent_emitted is False
        assert result.postgres_intent_emitted is False

    async def test_output_intents_are_immutable(
        self,
        dual_registration_reducer: NodeDualRegistrationReducer,
        sample_introspection_event: ModelNodeIntrospectionEvent,
    ) -> None:
        """Test that output intents tuple is immutable."""
        await dual_registration_reducer.initialize()

        result = await dual_registration_reducer.execute(sample_introspection_event)

        # Intents should be a tuple (immutable)
        assert isinstance(result.intents, tuple)
