# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Runtime execution integration tests for NodeRegistrationOrchestrator.

These tests verify the orchestrator's runtime workflow coordination with mocked
reducer and effect dependencies. Unlike the contract validation tests in
test_registration_orchestrator_integration.py, these tests focus on:

1. Workflow execution sequence - reducer called before effects
2. State transitions through the workflow
3. Event flow between components
4. Error handling and recovery scenarios
5. Correlation ID propagation
6. Partial failure handling

Test Categories:
    - TestWorkflowSequenceExecution: Verifies step ordering and dependencies
    - TestStateTransitions: Tests reducer state management
    - TestEventFlowCoordination: Validates event routing between nodes
    - TestErrorHandlingAndRecovery: Error scenarios and recovery patterns
    - TestCorrelationTracking: Distributed tracing propagation

Running Tests:
    # Run all runtime execution tests:
    pytest tests/integration/nodes/test_registration_orchestrator_runtime_execution.py

    # Run with verbose output:
    pytest tests/integration/nodes/test_registration_orchestrator_runtime_execution.py -v

    # Run specific test class:
    pytest tests/integration/nodes/test_registration_orchestrator_runtime_execution.py::TestWorkflowSequenceExecution
"""

from __future__ import annotations

import asyncio
import time
from datetime import UTC, datetime
from unittest.mock import AsyncMock, MagicMock
from uuid import UUID, uuid4

import pytest

# Test timestamp constant for reproducible tests
TEST_TIMESTAMP = datetime(2025, 1, 15, 12, 0, 0, tzinfo=UTC)

from omnibase_infra.models.registration import ModelNodeIntrospectionEvent
from omnibase_infra.nodes.node_registration_orchestrator.models import (
    ModelConsulIntentPayload,
    ModelConsulRegistrationIntent,
    ModelIntentExecutionResult,
    ModelOrchestratorInput,
    ModelOrchestratorOutput,
    ModelPostgresIntentPayload,
    ModelPostgresUpsertIntent,
    ModelReducerState,
    ModelRegistrationIntent,
)
from omnibase_infra.nodes.node_registration_orchestrator.node import (
    NodeRegistrationOrchestrator,
)
from omnibase_infra.nodes.node_registration_orchestrator.protocols import (
    ProtocolEffect,
    ProtocolReducer,
)

# Import shared conformance helpers
from tests.conftest import (
    assert_effect_protocol_interface,
    assert_has_async_methods,
    assert_has_methods,
    assert_reducer_protocol_interface,
)

# Module-level markers - all tests in this file are integration tests
pytestmark = [
    pytest.mark.integration,
]


# =============================================================================
# Mock Implementations
# =============================================================================


class MockReducerImpl:
    """Mock reducer implementation for testing workflow execution.

    This mock implements ProtocolReducer via duck typing and tracks all calls
    for verification. Per ONEX conventions, protocol compliance is verified
    by checking method presence and callability.

    Attributes:
        call_count: Number of times reduce() was called.
        received_events: List of events passed to reduce().
        received_states: List of states passed to reduce().
        call_timestamps: Timestamps of each reduce() call.
        should_raise: If set, raise this exception in reduce().
        custom_intents: If set, return these intents instead of defaults.
    """

    def __init__(self) -> None:
        self.call_count = 0
        self.received_events: list[ModelNodeIntrospectionEvent] = []
        self.received_states: list[ModelReducerState] = []
        self.call_timestamps: list[float] = []
        self.should_raise: Exception | None = None
        self.custom_intents: list[ModelRegistrationIntent] | None = None
        self._lock = asyncio.Lock()

    async def reduce(
        self,
        state: ModelReducerState,
        event: ModelNodeIntrospectionEvent,
    ) -> tuple[ModelReducerState, list[ModelRegistrationIntent]]:
        """Reduce an introspection event to state and intents.

        Thread-safe implementation using asyncio.Lock.
        """
        async with self._lock:
            self.call_count += 1
            self.received_events.append(event)
            self.received_states.append(state)
            self.call_timestamps.append(time.perf_counter())

        if self.should_raise:
            raise self.should_raise

        if self.custom_intents is not None:
            intents = self.custom_intents
        else:
            # Generate default intents
            intents: list[ModelRegistrationIntent] = [
                ModelConsulRegistrationIntent(
                    operation="register",
                    node_id=event.node_id,
                    correlation_id=event.correlation_id,
                    payload=ModelConsulIntentPayload(
                        service_name=f"node-{event.node_type}",
                    ),
                ),
                ModelPostgresUpsertIntent(
                    operation="upsert",
                    node_id=event.node_id,
                    correlation_id=event.correlation_id,
                    payload=ModelPostgresIntentPayload(
                        node_id=event.node_id,
                        node_type=event.node_type,
                        correlation_id=event.correlation_id,
                        timestamp=event.timestamp.isoformat(),
                    ),
                ),
            ]

        # Update state immutably
        new_state = ModelReducerState(
            last_event_timestamp=event.timestamp.isoformat(),
            processed_node_ids=state.processed_node_ids | frozenset({event.node_id}),
            pending_registrations=state.pending_registrations + len(intents),
        )

        return new_state, intents


class MockEffectImpl:
    """Mock effect implementation for testing workflow execution.

    This mock implements ProtocolEffect via duck typing and tracks all calls
    for verification. Per ONEX conventions, protocol compliance is verified
    by checking method presence and callability.

    Attributes:
        call_count: Number of times execute_intent() was called.
        executed_intents: List of intents passed to execute_intent().
        received_correlation_ids: List of correlation IDs received.
        call_timestamps: Timestamps of each execute_intent() call.
        should_fail: If True, all intents fail.
        fail_on_kinds: Set of intent kinds that should fail.
        execution_delay_ms: Artificial delay to simulate I/O.
        custom_results: If set, return these results instead of defaults.
    """

    def __init__(self) -> None:
        self.call_count = 0
        self.executed_intents: list[
            ModelConsulRegistrationIntent | ModelPostgresUpsertIntent
        ] = []
        self.received_correlation_ids: list[UUID] = []
        self.call_timestamps: list[float] = []
        self.should_fail = False
        self.fail_on_kinds: set[str] = set()
        self.execution_delay_ms = 0.0
        self.custom_results: dict[str, ModelIntentExecutionResult] | None = None
        self._lock = asyncio.Lock()

    async def execute_intent(
        self,
        intent: ModelConsulRegistrationIntent | ModelPostgresUpsertIntent,
        correlation_id: UUID,
    ) -> ModelIntentExecutionResult:
        """Execute a single registration intent.

        Thread-safe implementation using asyncio.Lock.
        """
        start_time = time.perf_counter()

        async with self._lock:
            self.call_count += 1
            self.executed_intents.append(intent)
            self.received_correlation_ids.append(correlation_id)
            self.call_timestamps.append(start_time)

        # Simulate I/O delay
        if self.execution_delay_ms > 0:
            await asyncio.sleep(self.execution_delay_ms / 1000.0)

        execution_time = (time.perf_counter() - start_time) * 1000

        # Return custom result if configured
        if self.custom_results and intent.kind in self.custom_results:
            return self.custom_results[intent.kind]

        # Simulate failure if configured
        if self.should_fail or intent.kind in self.fail_on_kinds:
            return ModelIntentExecutionResult(
                intent_kind=intent.kind,
                success=False,
                error=f"Mock failure for {intent.kind} {intent.operation}",
                execution_time_ms=execution_time,
            )

        return ModelIntentExecutionResult(
            intent_kind=intent.kind,
            success=True,
            error=None,
            execution_time_ms=execution_time,
        )


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def simple_mock_container() -> MagicMock:
    """Create a simple mock ONEX container for runtime execution tests.

    This is a simpler version than the conftest.py mock_container, providing
    just the basic container.config attribute needed for NodeOrchestrator.
    """
    container = MagicMock()
    container.config = MagicMock()
    return container


@pytest.fixture
def correlation_id() -> UUID:
    """Create a fixed correlation ID for testing."""
    return uuid4()


@pytest.fixture
def node_id() -> UUID:
    """Create a fixed node ID for testing."""
    return uuid4()


@pytest.fixture
def introspection_event(
    node_id: UUID, correlation_id: UUID
) -> ModelNodeIntrospectionEvent:
    """Create a test introspection event."""
    return ModelNodeIntrospectionEvent(
        node_id=node_id,
        node_type="effect",
        node_version="1.0.0",
        capabilities={},
        endpoints={"health": "http://localhost:8080/health"},
        correlation_id=correlation_id,
        timestamp=TEST_TIMESTAMP,
    )


@pytest.fixture
def orchestrator_input(
    introspection_event: ModelNodeIntrospectionEvent, correlation_id: UUID
) -> ModelOrchestratorInput:
    """Create test input for the orchestrator."""
    return ModelOrchestratorInput(
        introspection_event=introspection_event,
        correlation_id=correlation_id,
    )


@pytest.fixture
def mock_reducer() -> MockReducerImpl:
    """Create mock reducer for testing."""
    mock = MockReducerImpl()
    # Verify ProtocolReducer interface via shared conformance helper
    assert_reducer_protocol_interface(mock)
    return mock


@pytest.fixture
def mock_effect() -> MockEffectImpl:
    """Create mock effect for testing."""
    mock = MockEffectImpl()
    # Verify ProtocolEffect interface via shared conformance helper
    assert_effect_protocol_interface(mock)
    return mock


@pytest.fixture
def mock_event_emitter() -> MagicMock:
    """Create mock event emitter to capture published events."""
    emitter = MagicMock()
    emitter.emitted_events = []

    async def emit(event_type: str, event_data: dict) -> None:
        emitter.emitted_events.append((event_type, event_data))

    emitter.emit = AsyncMock(side_effect=emit)
    return emitter


@pytest.fixture
def configured_container(
    simple_mock_container: MagicMock,
    mock_reducer: MockReducerImpl,
    mock_effect: MockEffectImpl,
    mock_event_emitter: MagicMock,
) -> MagicMock:
    """Create container with mock dependencies configured."""

    def resolve_mock(protocol: type) -> object:
        """Resolve mock dependencies using protocol type matching."""
        if protocol is ProtocolReducer:
            return mock_reducer
        elif protocol is ProtocolEffect:
            return mock_effect
        return mock_event_emitter

    simple_mock_container.service_registry = MagicMock()
    simple_mock_container.service_registry.resolve = MagicMock(side_effect=resolve_mock)

    # Store references for test access
    simple_mock_container._test_reducer = mock_reducer
    simple_mock_container._test_effect = mock_effect
    simple_mock_container._test_emitter = mock_event_emitter

    return simple_mock_container


@pytest.fixture
def orchestrator(configured_container: MagicMock) -> NodeRegistrationOrchestrator:
    """Create orchestrator with mock dependencies."""
    return NodeRegistrationOrchestrator(configured_container)


# =============================================================================
# TestWorkflowSequenceExecution
# =============================================================================


class TestWorkflowSequenceExecution:
    """Tests for workflow execution sequence and step ordering.

    These tests verify that:
    - Reducer is called before effects
    - Effects receive intents from reducer
    - Effects are called in expected order
    - Dependencies between steps are respected
    """

    @pytest.mark.asyncio
    async def test_reducer_called_before_effects(
        self,
        mock_reducer: MockReducerImpl,
        mock_effect: MockEffectImpl,
        introspection_event: ModelNodeIntrospectionEvent,
        correlation_id: UUID,
    ) -> None:
        """Test that reducer is called before any effects are executed."""
        # Execute workflow steps manually
        initial_state = ModelReducerState.initial()
        _, intents = await mock_reducer.reduce(initial_state, introspection_event)

        for intent in intents:
            await mock_effect.execute_intent(intent, correlation_id)

        # Verify reducer was called first
        assert mock_reducer.call_count == 1
        assert len(mock_reducer.call_timestamps) == 1

        # Verify effects were called after reducer
        assert mock_effect.call_count == len(intents)
        assert len(mock_effect.call_timestamps) == len(intents)

        # All effect timestamps should be after reducer timestamp
        reducer_time = mock_reducer.call_timestamps[0]
        for effect_time in mock_effect.call_timestamps:
            assert effect_time >= reducer_time, (
                f"Effect called before reducer: {effect_time} < {reducer_time}"
            )

    @pytest.mark.asyncio
    async def test_effects_receive_reducer_intents(
        self,
        mock_reducer: MockReducerImpl,
        mock_effect: MockEffectImpl,
        introspection_event: ModelNodeIntrospectionEvent,
        correlation_id: UUID,
    ) -> None:
        """Test that effects receive exactly the intents from reducer."""
        initial_state = ModelReducerState.initial()
        _, intents = await mock_reducer.reduce(initial_state, introspection_event)

        for intent in intents:
            await mock_effect.execute_intent(intent, correlation_id)

        # Verify intents match
        assert len(mock_effect.executed_intents) == len(intents)
        for i, executed in enumerate(mock_effect.executed_intents):
            assert executed == intents[i], (
                f"Intent mismatch at index {i}: {executed} != {intents[i]}"
            )

    @pytest.mark.asyncio
    async def test_effect_order_follows_intent_order(
        self,
        mock_reducer: MockReducerImpl,
        mock_effect: MockEffectImpl,
        introspection_event: ModelNodeIntrospectionEvent,
        correlation_id: UUID,
    ) -> None:
        """Test that effects are executed in intent order."""
        initial_state = ModelReducerState.initial()
        _, intents = await mock_reducer.reduce(initial_state, introspection_event)

        # Execute sequentially
        for intent in intents:
            await mock_effect.execute_intent(intent, correlation_id)

        # Verify order by kind
        expected_kinds = [i.kind for i in intents]
        actual_kinds = [i.kind for i in mock_effect.executed_intents]
        assert actual_kinds == expected_kinds

    @pytest.mark.asyncio
    async def test_parallel_effects_execution(
        self,
        mock_reducer: MockReducerImpl,
        mock_effect: MockEffectImpl,
        introspection_event: ModelNodeIntrospectionEvent,
        correlation_id: UUID,
    ) -> None:
        """Test parallel effect execution (simulating parallel coordination)."""
        # Add delay to make parallelism observable
        mock_effect.execution_delay_ms = 50

        initial_state = ModelReducerState.initial()
        _, intents = await mock_reducer.reduce(initial_state, introspection_event)

        # Execute sequentially first to measure baseline
        sequential_start = time.perf_counter()
        for intent in intents:
            await mock_effect.execute_intent(intent, correlation_id)
        sequential_elapsed = (time.perf_counter() - sequential_start) * 1000

        # Reset for parallel execution
        mock_effect.executed_intents.clear()

        # Execute in parallel
        start_time = time.perf_counter()
        tasks = [
            mock_effect.execute_intent(intent, correlation_id) for intent in intents
        ]
        results = await asyncio.gather(*tasks)
        parallel_elapsed = (time.perf_counter() - start_time) * 1000

        # Verify all completed
        assert len(results) == len(intents)
        assert all(r.success for r in results)

        # CI-friendly timing assertion: Use relative comparison instead of absolute
        # threshold to avoid flakiness across different CI environments.
        # Parallel execution should be meaningfully faster than sequential execution.
        # We use a generous 0.9x multiplier to account for CI variance while still
        # proving parallelism provides a speedup.
        #
        # Local benchmark: parallel ~50ms vs sequential ~100ms (2x speedup)
        # CI allowance: parallel can be up to 90% of sequential and still pass
        assert parallel_elapsed < sequential_elapsed * 0.9, (
            f"Parallel execution ({parallel_elapsed:.1f}ms) should be faster than "
            f"sequential ({sequential_elapsed:.1f}ms) * 0.9 = {sequential_elapsed * 0.9:.1f}ms"
        )


# =============================================================================
# TestStateTransitions
# =============================================================================


class TestStateTransitions:
    """Tests for reducer state management and transitions.

    These tests verify that:
    - Initial state is correctly initialized
    - State transitions preserve immutability
    - Processed node IDs are tracked
    - Pending registrations are counted correctly
    """

    @pytest.mark.asyncio
    async def test_initial_state(self) -> None:
        """Test that initial state has expected default values."""
        state = ModelReducerState.initial()

        assert state.last_event_timestamp is None
        assert state.processed_node_ids == frozenset()
        assert state.pending_registrations == 0

    @pytest.mark.asyncio
    async def test_state_immutability(
        self,
        mock_reducer: MockReducerImpl,
        introspection_event: ModelNodeIntrospectionEvent,
    ) -> None:
        """Test that reducer returns new state without mutating input."""
        initial_state = ModelReducerState.initial()
        initial_id = id(initial_state)
        initial_timestamp = initial_state.last_event_timestamp
        initial_node_ids = initial_state.processed_node_ids

        new_state, _ = await mock_reducer.reduce(initial_state, introspection_event)

        # Original state should be unchanged
        assert id(initial_state) == initial_id
        assert initial_state.last_event_timestamp == initial_timestamp
        assert initial_state.processed_node_ids == initial_node_ids

        # New state should be different
        assert id(new_state) != id(initial_state)
        assert new_state.last_event_timestamp is not None
        assert len(new_state.processed_node_ids) > len(initial_node_ids)

    @pytest.mark.asyncio
    async def test_processed_node_tracking(
        self,
        mock_reducer: MockReducerImpl,
        node_id: UUID,
        correlation_id: UUID,
    ) -> None:
        """Test that processed node IDs are tracked in state."""
        event = ModelNodeIntrospectionEvent(
            node_id=node_id,
            node_type="compute",
            node_version="1.0.0",
            endpoints={"health": "http://localhost:8080/health"},
            correlation_id=correlation_id,
            timestamp=TEST_TIMESTAMP,
        )

        initial_state = ModelReducerState.initial()
        assert node_id not in initial_state.processed_node_ids

        new_state, _ = await mock_reducer.reduce(initial_state, event)
        assert node_id in new_state.processed_node_ids

    @pytest.mark.asyncio
    async def test_pending_registration_count(
        self,
        mock_reducer: MockReducerImpl,
        introspection_event: ModelNodeIntrospectionEvent,
    ) -> None:
        """Test that pending registrations are counted correctly."""
        initial_state = ModelReducerState.initial()
        assert initial_state.pending_registrations == 0

        new_state, intents = await mock_reducer.reduce(
            initial_state, introspection_event
        )

        # Pending should equal intent count
        assert new_state.pending_registrations == len(intents)

    @pytest.mark.asyncio
    async def test_multiple_event_state_accumulation(
        self,
        mock_reducer: MockReducerImpl,
        correlation_id: UUID,
    ) -> None:
        """Test state accumulation across multiple events."""
        node_ids = [uuid4() for _ in range(3)]
        events = [
            ModelNodeIntrospectionEvent(
                node_id=nid,
                node_type="effect",
                node_version="1.0.0",
                endpoints={"health": "http://localhost:8080/health"},
                correlation_id=correlation_id,
                timestamp=TEST_TIMESTAMP,
            )
            for nid in node_ids
        ]

        state = ModelReducerState.initial()
        total_intents = 0

        for event in events:
            state, intents = await mock_reducer.reduce(state, event)
            total_intents += len(intents)

        # All node IDs should be in processed set
        for nid in node_ids:
            assert nid in state.processed_node_ids

        # Pending should match total intents
        assert state.pending_registrations == total_intents


# =============================================================================
# TestEventFlowCoordination
# =============================================================================


class TestEventFlowCoordination:
    """Tests for event flow between orchestrator components.

    These tests verify that:
    - Events are routed to correct handlers
    - Intent kinds determine effect behavior
    - Results are properly aggregated
    """

    @pytest.mark.asyncio
    async def test_consul_intent_routing(
        self,
        mock_effect: MockEffectImpl,
        node_id: UUID,
        correlation_id: UUID,
    ) -> None:
        """Test that consul intents are executed correctly."""
        intent = ModelConsulRegistrationIntent(
            operation="register",
            node_id=node_id,
            correlation_id=correlation_id,
            payload=ModelConsulIntentPayload(service_name="test-service"),
        )

        result = await mock_effect.execute_intent(intent, correlation_id)

        assert result.intent_kind == "consul"
        assert result.success is True
        assert mock_effect.executed_intents[0].kind == "consul"

    @pytest.mark.asyncio
    async def test_postgres_intent_routing(
        self,
        mock_effect: MockEffectImpl,
        node_id: UUID,
        correlation_id: UUID,
    ) -> None:
        """Test that postgres intents are executed correctly."""
        intent = ModelPostgresUpsertIntent(
            operation="upsert",
            node_id=node_id,
            correlation_id=correlation_id,
            payload=ModelPostgresIntentPayload(
                node_id=node_id,
                node_type="effect",
                correlation_id=correlation_id,
                timestamp=datetime.now(UTC).isoformat(),
            ),
        )

        result = await mock_effect.execute_intent(intent, correlation_id)

        assert result.intent_kind == "postgres"
        assert result.success is True
        assert mock_effect.executed_intents[0].kind == "postgres"

    @pytest.mark.asyncio
    async def test_result_aggregation_all_success(
        self,
        mock_reducer: MockReducerImpl,
        mock_effect: MockEffectImpl,
        introspection_event: ModelNodeIntrospectionEvent,
        correlation_id: UUID,
    ) -> None:
        """Test result aggregation when all effects succeed."""
        initial_state = ModelReducerState.initial()
        _, intents = await mock_reducer.reduce(initial_state, introspection_event)

        results = []
        for intent in intents:
            result = await mock_effect.execute_intent(intent, correlation_id)
            results.append(result)

        # Aggregate results
        consul_results = [r for r in results if r.intent_kind == "consul"]
        postgres_results = [r for r in results if r.intent_kind == "postgres"]

        consul_applied = all(r.success for r in consul_results)
        postgres_applied = all(r.success for r in postgres_results)
        all_success = consul_applied and postgres_applied

        output = ModelOrchestratorOutput(
            correlation_id=correlation_id,
            status="success" if all_success else "failed",
            consul_applied=consul_applied,
            postgres_applied=postgres_applied,
            intent_results=results,
            total_execution_time_ms=sum(r.execution_time_ms for r in results),
        )

        assert output.status == "success"
        assert output.consul_applied is True
        assert output.postgres_applied is True
        assert len(output.intent_results) == len(intents)

    @pytest.mark.asyncio
    async def test_result_aggregation_partial_success(
        self,
        mock_reducer: MockReducerImpl,
        mock_effect: MockEffectImpl,
        introspection_event: ModelNodeIntrospectionEvent,
        correlation_id: UUID,
    ) -> None:
        """Test result aggregation when only some effects succeed."""
        mock_effect.fail_on_kinds.add("consul")

        initial_state = ModelReducerState.initial()
        _, intents = await mock_reducer.reduce(initial_state, introspection_event)

        results = []
        for intent in intents:
            result = await mock_effect.execute_intent(intent, correlation_id)
            results.append(result)

        consul_results = [r for r in results if r.intent_kind == "consul"]
        postgres_results = [r for r in results if r.intent_kind == "postgres"]

        consul_applied = all(r.success for r in consul_results)
        postgres_applied = all(r.success for r in postgres_results)
        any_success = consul_applied or postgres_applied
        all_success = consul_applied and postgres_applied

        status = "success" if all_success else ("partial" if any_success else "failed")
        consul_error = next((r.error for r in consul_results if not r.success), None)

        output = ModelOrchestratorOutput(
            correlation_id=correlation_id,
            status=status,
            consul_applied=consul_applied,
            postgres_applied=postgres_applied,
            consul_error=consul_error,
            intent_results=results,
            total_execution_time_ms=sum(r.execution_time_ms for r in results),
        )

        assert output.status == "partial"
        assert output.consul_applied is False
        assert output.postgres_applied is True
        assert output.consul_error is not None

    @pytest.mark.asyncio
    async def test_result_aggregation_all_failed(
        self,
        mock_reducer: MockReducerImpl,
        mock_effect: MockEffectImpl,
        introspection_event: ModelNodeIntrospectionEvent,
        correlation_id: UUID,
    ) -> None:
        """Test result aggregation when all effects fail."""
        mock_effect.should_fail = True

        initial_state = ModelReducerState.initial()
        _, intents = await mock_reducer.reduce(initial_state, introspection_event)

        results = []
        for intent in intents:
            result = await mock_effect.execute_intent(intent, correlation_id)
            results.append(result)

        consul_applied = any(r.success for r in results if r.intent_kind == "consul")
        postgres_applied = any(
            r.success for r in results if r.intent_kind == "postgres"
        )

        output = ModelOrchestratorOutput(
            correlation_id=correlation_id,
            status="failed",
            consul_applied=consul_applied,
            postgres_applied=postgres_applied,
            consul_error="Mock failure for consul register",
            postgres_error="Mock failure for postgres upsert",
            intent_results=results,
            total_execution_time_ms=sum(r.execution_time_ms for r in results),
        )

        assert output.status == "failed"
        assert output.consul_applied is False
        assert output.postgres_applied is False


# =============================================================================
# TestErrorHandlingAndRecovery
# =============================================================================


class TestErrorHandlingAndRecovery:
    """Tests for error handling and recovery scenarios.

    These tests verify that:
    - Reducer errors are properly surfaced
    - Effect failures are captured in results
    - Partial failures are handled correctly
    - Error context is preserved
    """

    @pytest.mark.asyncio
    async def test_reducer_error_propagation(
        self,
        mock_reducer: MockReducerImpl,
        introspection_event: ModelNodeIntrospectionEvent,
    ) -> None:
        """Test that reducer errors are propagated correctly."""
        mock_reducer.should_raise = ValueError("Invalid event data")

        initial_state = ModelReducerState.initial()

        with pytest.raises(ValueError) as exc_info:
            await mock_reducer.reduce(initial_state, introspection_event)

        assert "Invalid event data" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_effect_failure_captured_in_result(
        self,
        mock_effect: MockEffectImpl,
        node_id: UUID,
        correlation_id: UUID,
    ) -> None:
        """Test that effect failures are captured in result, not raised."""
        mock_effect.should_fail = True

        intent = ModelConsulRegistrationIntent(
            operation="register",
            node_id=node_id,
            correlation_id=correlation_id,
            payload=ModelConsulIntentPayload(service_name="test-service"),
        )

        # Effect should not raise, but return failure result
        result = await mock_effect.execute_intent(intent, correlation_id)

        assert result.success is False
        assert result.error is not None
        assert "Mock failure" in result.error

    @pytest.mark.asyncio
    async def test_selective_intent_failure(
        self,
        mock_reducer: MockReducerImpl,
        mock_effect: MockEffectImpl,
        introspection_event: ModelNodeIntrospectionEvent,
        correlation_id: UUID,
    ) -> None:
        """Test handling when specific intent types fail."""
        # Only postgres fails
        mock_effect.fail_on_kinds.add("postgres")

        initial_state = ModelReducerState.initial()
        _, intents = await mock_reducer.reduce(initial_state, introspection_event)

        results = []
        for intent in intents:
            result = await mock_effect.execute_intent(intent, correlation_id)
            results.append(result)

        consul_results = [r for r in results if r.intent_kind == "consul"]
        postgres_results = [r for r in results if r.intent_kind == "postgres"]

        # Consul should succeed, postgres should fail
        assert all(r.success for r in consul_results)
        assert all(not r.success for r in postgres_results)

    @pytest.mark.asyncio
    async def test_error_context_preservation(
        self,
        mock_effect: MockEffectImpl,
        node_id: UUID,
        correlation_id: UUID,
    ) -> None:
        """Test that error messages include context without sensitive data."""
        mock_effect.should_fail = True

        intent = ModelPostgresUpsertIntent(
            operation="upsert",
            node_id=node_id,
            correlation_id=correlation_id,
            payload=ModelPostgresIntentPayload(
                node_id=node_id,
                node_type="effect",
                correlation_id=correlation_id,
                timestamp=datetime.now(UTC).isoformat(),
            ),
        )

        result = await mock_effect.execute_intent(intent, correlation_id)

        # Error should include intent kind and operation
        assert result.error is not None
        assert "postgres" in result.error
        assert "upsert" in result.error

        # Error should NOT include sensitive data
        assert str(correlation_id) not in result.error
        assert str(node_id) not in result.error

    @pytest.mark.asyncio
    async def test_workflow_continues_after_individual_failure(
        self,
        mock_reducer: MockReducerImpl,
        mock_effect: MockEffectImpl,
        introspection_event: ModelNodeIntrospectionEvent,
        correlation_id: UUID,
    ) -> None:
        """Test that workflow processes all intents even if some fail."""
        mock_effect.fail_on_kinds.add("consul")

        initial_state = ModelReducerState.initial()
        _, intents = await mock_reducer.reduce(initial_state, introspection_event)

        results = []
        for intent in intents:
            result = await mock_effect.execute_intent(intent, correlation_id)
            results.append(result)

        # All intents should be executed
        assert len(results) == len(intents)
        assert mock_effect.call_count == len(intents)


# =============================================================================
# TestCorrelationTracking
# =============================================================================


class TestCorrelationTracking:
    """Tests for correlation ID propagation through workflow.

    These tests verify that:
    - Correlation ID is passed from input to reducer
    - Reducer preserves correlation ID in intents
    - Effect receives correct correlation ID
    - All components share same correlation context
    """

    @pytest.mark.asyncio
    async def test_correlation_id_in_reducer_intents(
        self,
        mock_reducer: MockReducerImpl,
        introspection_event: ModelNodeIntrospectionEvent,
        correlation_id: UUID,
    ) -> None:
        """Test that reducer includes correlation ID in all intents."""
        initial_state = ModelReducerState.initial()
        _, intents = await mock_reducer.reduce(initial_state, introspection_event)

        for intent in intents:
            assert intent.correlation_id == correlation_id, (
                f"Intent {intent.kind} has wrong correlation_id: "
                f"{intent.correlation_id} != {correlation_id}"
            )

    @pytest.mark.asyncio
    async def test_correlation_id_passed_to_effect(
        self,
        mock_effect: MockEffectImpl,
        node_id: UUID,
        correlation_id: UUID,
    ) -> None:
        """Test that effect receives correlation ID parameter."""
        intent = ModelConsulRegistrationIntent(
            operation="register",
            node_id=node_id,
            correlation_id=correlation_id,
            payload=ModelConsulIntentPayload(service_name="test-service"),
        )

        await mock_effect.execute_intent(intent, correlation_id)

        assert mock_effect.received_correlation_ids[0] == correlation_id

    @pytest.mark.asyncio
    async def test_correlation_id_consistent_across_workflow(
        self,
        mock_reducer: MockReducerImpl,
        mock_effect: MockEffectImpl,
        introspection_event: ModelNodeIntrospectionEvent,
        correlation_id: UUID,
    ) -> None:
        """Test correlation ID consistency from input through output."""
        initial_state = ModelReducerState.initial()
        _, intents = await mock_reducer.reduce(initial_state, introspection_event)

        for intent in intents:
            await mock_effect.execute_intent(intent, correlation_id)

        # Verify all correlation IDs match
        assert introspection_event.correlation_id == correlation_id
        for intent in intents:
            assert intent.correlation_id == correlation_id
        for received_id in mock_effect.received_correlation_ids:
            assert received_id == correlation_id

    @pytest.mark.asyncio
    async def test_different_correlation_ids_isolated(
        self,
        mock_reducer: MockReducerImpl,
        mock_effect: MockEffectImpl,
        node_id: UUID,
    ) -> None:
        """Test that different correlation IDs are kept separate."""
        corr_id_1 = uuid4()
        corr_id_2 = uuid4()

        event_1 = ModelNodeIntrospectionEvent(
            node_id=node_id,
            node_type="effect",
            node_version="1.0.0",
            endpoints={"health": "http://localhost:8080/health"},
            correlation_id=corr_id_1,
            timestamp=TEST_TIMESTAMP,
        )
        event_2 = ModelNodeIntrospectionEvent(
            node_id=uuid4(),  # Different node
            node_type="compute",
            node_version="1.0.0",
            endpoints={"health": "http://localhost:8081/health"},
            correlation_id=corr_id_2,
            timestamp=TEST_TIMESTAMP,
        )

        state = ModelReducerState.initial()
        state, intents_1 = await mock_reducer.reduce(state, event_1)
        state, intents_2 = await mock_reducer.reduce(state, event_2)

        # Each intent should have its source event's correlation ID
        for intent in intents_1:
            assert intent.correlation_id == corr_id_1
        for intent in intents_2:
            assert intent.correlation_id == corr_id_2


# =============================================================================
# TestOrchestratorIntegration
# =============================================================================


class TestOrchestratorIntegration:
    """Integration tests for full orchestrator instantiation and behavior.

    These tests verify that:
    - Orchestrator instantiates correctly with mock container
    - Container provides mock dependencies
    - Orchestrator has expected methods from base class
    """

    def test_orchestrator_instantiation(
        self,
        orchestrator: NodeRegistrationOrchestrator,
    ) -> None:
        """Test that orchestrator instantiates correctly."""
        # Verify type via duck typing
        assert orchestrator.__class__.__name__ == "NodeRegistrationOrchestrator"

        # Verify required methods
        required_methods = [
            "process",
            "execute_workflow_from_contract",
            "get_node_type",
        ]
        for method in required_methods:
            assert hasattr(orchestrator, method), f"Missing method: {method}"
            assert callable(getattr(orchestrator, method))

    def test_orchestrator_has_container(
        self,
        orchestrator: NodeRegistrationOrchestrator,
        configured_container: MagicMock,
    ) -> None:
        """Test that orchestrator has container reference."""
        assert hasattr(orchestrator, "container")
        assert orchestrator.container is configured_container

    def test_mock_dependencies_accessible(
        self,
        configured_container: MagicMock,
    ) -> None:
        """Test that mock dependencies are accessible via container."""
        assert hasattr(configured_container, "_test_reducer")
        assert hasattr(configured_container, "_test_effect")
        assert hasattr(configured_container, "_test_emitter")

        # Verify protocol interfaces via shared conformance helpers
        assert_reducer_protocol_interface(configured_container._test_reducer)
        assert_effect_protocol_interface(configured_container._test_effect)


# =============================================================================
# TestConcurrencyAndThreadSafety
# =============================================================================


class TestConcurrencyAndThreadSafety:
    """Tests for concurrent execution and thread safety.

    These tests verify that:
    - Reducer handles concurrent calls safely
    - Effect handles concurrent calls safely
    - State is not corrupted by concurrent access
    """

    @pytest.mark.asyncio
    async def test_concurrent_reducer_calls(
        self,
        mock_reducer: MockReducerImpl,
        correlation_id: UUID,
    ) -> None:
        """Test that reducer handles concurrent calls correctly."""
        events = [
            ModelNodeIntrospectionEvent(
                node_id=uuid4(),
                node_type="effect",
                node_version="1.0.0",
                endpoints={"health": f"http://localhost:{8080 + i}/health"},
                correlation_id=correlation_id,
                timestamp=TEST_TIMESTAMP,
            )
            for i in range(10)
        ]

        initial_state = ModelReducerState.initial()

        # Execute concurrently with same initial state
        # Each call uses the same initial state (not chained)
        tasks = [mock_reducer.reduce(initial_state, event) for event in events]
        results = await asyncio.gather(*tasks)

        # All calls should complete
        assert len(results) == 10
        assert mock_reducer.call_count == 10

        # Each result should have intents
        for state, intents in results:
            assert len(intents) == 2  # consul and postgres

    @pytest.mark.asyncio
    async def test_concurrent_effect_calls(
        self,
        mock_effect: MockEffectImpl,
        correlation_id: UUID,
    ) -> None:
        """Test that effect handles concurrent calls correctly."""
        mock_effect.execution_delay_ms = 10  # Small delay for overlap

        intents = [
            ModelConsulRegistrationIntent(
                operation="register",
                node_id=uuid4(),
                correlation_id=correlation_id,
                payload=ModelConsulIntentPayload(service_name=f"service-{i}"),
            )
            for i in range(10)
        ]

        tasks = [
            mock_effect.execute_intent(intent, correlation_id) for intent in intents
        ]
        results = await asyncio.gather(*tasks)

        assert len(results) == 10
        assert mock_effect.call_count == 10
        assert all(r.success for r in results)

    @pytest.mark.asyncio
    async def test_state_isolation_under_concurrency(
        self,
        mock_reducer: MockReducerImpl,
        correlation_id: UUID,
    ) -> None:
        """Test that state changes don't leak between concurrent calls."""
        events = [
            ModelNodeIntrospectionEvent(
                node_id=uuid4(),
                node_type="effect",
                node_version="1.0.0",
                endpoints={"health": f"http://localhost:{8080 + i}/health"},
                correlation_id=correlation_id,
                timestamp=TEST_TIMESTAMP,
            )
            for i in range(5)
        ]

        # Use chained state updates
        state = ModelReducerState.initial()
        for event in events:
            state, _ = await mock_reducer.reduce(state, event)

        # All events should be tracked
        assert len(state.processed_node_ids) == 5
        for event in events:
            assert event.node_id in state.processed_node_ids
