# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Integration tests A3 and A4 for OMN-915 mocked E2E registration workflow.

These tests verify the orchestrated registration workflow with ZERO real
infrastructure, using test doubles for all backends.

Test Scenarios:
    A3 - Orchestrated Dual Registration:
        - Orchestrator receives introspection event
        - Reducer computes intents (NO I/O)
        - Effect executes intents (DOES I/O via mocks)
        - Results aggregated correctly
        - Verifies reducer called BEFORE effect

    A4 - Idempotent Replay:
        - First emission processes event
        - Replay same event (same event_id, correlation_id)
        - Logical state identical
        - No duplicate Postgres rows
        - Effect idempotency store tracks completed backends

Design Principles:
    - ZERO real infrastructure: All backends are test doubles
    - Call order tracking: Verifies orchestration order
    - State comparison: Verifies idempotency
    - Mocked backends: Uses StubConsulClient and StubPostgresAdapter

Related:
    - RegistrationReducer: Pure reducer for intent computation
    - NodeRegistryEffect: Effect for backend registration execution
    - NodeRegistrationOrchestrator: Workflow coordinator (contract-driven)
    - OMN-915: Mocked E2E tests ticket
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING
from uuid import uuid4

import pytest
from omnibase_core.enums.enum_node_kind import EnumNodeKind

if TYPE_CHECKING:
    from omnibase_core.models.reducer.model_intent import ModelIntent

from omnibase_infra.models.registration import ModelNodeIntrospectionEvent
from omnibase_infra.nodes.effects.models import ModelRegistryRequest
from omnibase_infra.nodes.reducers.models import ModelRegistrationState

# Import test doubles
from tests.integration.registration.effect.test_doubles import (
    StubConsulClient,
    StubPostgresAdapter,
)

from .conftest import (
    CallOrderTracker,
    TrackedNodeRegistryEffect,
    TrackedRegistrationReducer,
)


def _convert_intents_to_request(
    event: ModelNodeIntrospectionEvent,
    intents: tuple[ModelIntent, ...],
) -> ModelRegistryRequest:
    """Convert reducer output intents to a registry request.

    This helper simulates what the orchestrator does when converting
    reducer intents into an effect request.

    Args:
        event: The introspection event that triggered the workflow.
        intents: Tuple of ModelIntent from the reducer output.

    Returns:
        ModelRegistryRequest for the effect node.
    """
    return ModelRegistryRequest(
        node_id=event.node_id,
        node_type=event.node_type,
        node_version=event.node_version,
        correlation_id=event.correlation_id,
        service_name=f"onex-{event.node_type}",
        endpoints=dict(event.endpoints) if event.endpoints else {},
        tags=[f"node_type:{event.node_type}", f"node_version:{event.node_version}"],
        metadata={},
    )


@pytest.mark.integration
class TestA3OrchestratedDualRegistration:
    """Test A3: Orchestrator calls reducer, then effect.

    Verifies:
        1. Reducer is called BEFORE effect (call order tracking)
        2. Reducer computes intents (NO I/O - verified by mock call counts)
        3. Effect executes intents (DOES I/O - verified by mock call counts)
        4. Results aggregated correctly
    """

    @pytest.mark.asyncio
    async def test_a3_orchestrated_dual_registration(
        self,
        tracked_reducer: TrackedRegistrationReducer,
        tracked_effect: TrackedNodeRegistryEffect,
        call_tracker: CallOrderTracker,
        consul_client: StubConsulClient,
        postgres_adapter: StubPostgresAdapter,
        sample_introspection_event: ModelNodeIntrospectionEvent,
    ) -> None:
        """Orchestrator calls reducer, then effect.

        This test simulates the orchestrator workflow:
        1. Orchestrator receives introspection event
        2. Orchestrator calls reducer to compute intents
        3. Orchestrator calls effect to execute intents
        4. Results are aggregated

        Verification:
        - Call order: reducer BEFORE effect
        - Reducer output: produces intents without I/O
        - Effect execution: performs I/O via mocked backends
        - Final state: registration complete
        """
        # === PHASE 1: Orchestrator receives introspection event ===
        event = sample_introspection_event
        initial_state = ModelRegistrationState()

        # Verify initial state
        assert initial_state.status == "idle"
        assert initial_state.consul_confirmed is False
        assert initial_state.postgres_confirmed is False

        # Record initial mock call counts (should be 0)
        assert consul_client.call_count == 0
        assert postgres_adapter.call_count == 0

        # === PHASE 2: Reducer computes intents (NO I/O) ===
        reducer_output = tracked_reducer.reduce(initial_state, event)

        # Verify reducer was called
        assert tracked_reducer.reduce_call_count == 1

        # Verify reducer output contains new state and intents
        assert reducer_output.result.status == "pending"
        assert reducer_output.result.node_id == event.node_id
        assert len(reducer_output.intents) == 2  # Consul + PostgreSQL intents

        # Verify reducer did NOT perform any I/O
        # Mock call counts should still be 0 after reducer runs
        assert consul_client.call_count == 0, "Reducer should NOT call Consul"
        assert postgres_adapter.call_count == 0, "Reducer should NOT call PostgreSQL"

        # Verify intents are for correct backends
        intent_types = {intent.intent_type for intent in reducer_output.intents}
        assert "consul.register" in intent_types
        assert "postgres.upsert_registration" in intent_types

        # === PHASE 3: Effect executes intents (DOES I/O) ===
        request = _convert_intents_to_request(event, reducer_output.intents)
        effect_response = await tracked_effect.register_node(request)

        # Verify effect was called
        assert tracked_effect.register_node_call_count == 1

        # Verify effect DID perform I/O (mock call counts should be incremented)
        assert consul_client.call_count == 1, "Effect should call Consul"
        assert postgres_adapter.call_count == 1, "Effect should call PostgreSQL"

        # Verify effect response shows success
        assert effect_response.status == "success"
        assert effect_response.consul_result.success is True
        assert effect_response.postgres_result.success is True

        # === PHASE 4: Verify call order ===
        call_order = call_tracker.get_call_order()
        assert call_order == ["reducer", "effect"], (
            f"Expected reducer before effect, got: {call_order}"
        )

        # Verify correlation ID propagation
        reducer_calls = call_tracker.get_reducer_calls()
        effect_calls = call_tracker.get_effect_calls()
        assert reducer_calls[0].correlation_id == event.correlation_id
        assert effect_calls[0].correlation_id == event.correlation_id

        # === PHASE 5: Verify backend registrations ===
        # Consul registration recorded
        assert len(consul_client.registrations) == 1
        consul_reg = consul_client.registrations[0]
        assert f"node-{event.node_type}-{event.node_id}" == consul_reg.service_id

        # PostgreSQL registration recorded
        assert len(postgres_adapter.registrations) == 1
        pg_reg = postgres_adapter.registrations[0]
        assert pg_reg.node_id == event.node_id
        assert pg_reg.node_type == event.node_type

    @pytest.mark.asyncio
    async def test_a3_reducer_produces_typed_intents(
        self,
        tracked_reducer: TrackedRegistrationReducer,
        sample_introspection_event: ModelNodeIntrospectionEvent,
    ) -> None:
        """Verify reducer produces properly typed intents.

        The reducer should produce ModelIntent objects with typed payloads
        for both Consul and PostgreSQL backends.
        """
        initial_state = ModelRegistrationState()
        reducer_output = tracked_reducer.reduce(
            initial_state, sample_introspection_event
        )

        # Verify intent structure
        assert len(reducer_output.intents) == 2

        for intent in reducer_output.intents:
            assert intent.intent_type in (
                "consul.register",
                "postgres.upsert_registration",
            )
            assert intent.target is not None
            assert intent.payload is not None

            if intent.intent_type == "consul.register":
                # Consul intent should have service registration payload
                assert "correlation_id" in intent.payload
                assert "service_id" in intent.payload
                assert "service_name" in intent.payload

            elif intent.intent_type == "postgres.upsert_registration":
                # PostgreSQL intent should have record payload
                assert "correlation_id" in intent.payload
                assert "record" in intent.payload

    @pytest.mark.asyncio
    async def test_a3_multiple_node_types(
        self,
        tracked_reducer: TrackedRegistrationReducer,
        tracked_effect: TrackedNodeRegistryEffect,
        call_tracker: CallOrderTracker,
        consul_client: StubConsulClient,
        postgres_adapter: StubPostgresAdapter,
        introspection_event_factory: Callable[..., ModelNodeIntrospectionEvent],
    ) -> None:
        """Test orchestration works for all ONEX node types.

        Verifies the workflow handles effect, compute, reducer, and
        orchestrator node types correctly.
        """
        node_types: list[EnumNodeKind] = [
            EnumNodeKind.EFFECT,
            EnumNodeKind.COMPUTE,
            EnumNodeKind.REDUCER,
            EnumNodeKind.ORCHESTRATOR,
        ]

        for node_type in node_types:
            # Reset state for each iteration
            call_tracker.clear()
            consul_client.reset()
            postgres_adapter.reset()
            tracked_reducer.reduce_call_count = 0
            tracked_effect.register_node_call_count = 0

            # Create event for this node type
            event = introspection_event_factory(node_type=node_type)
            initial_state = ModelRegistrationState()

            # Execute workflow
            reducer_output = tracked_reducer.reduce(initial_state, event)
            request = _convert_intents_to_request(event, reducer_output.intents)
            effect_response = await tracked_effect.register_node(request)

            # Verify for this node type
            assert reducer_output.result.status == "pending", f"Failed for {node_type}"
            assert effect_response.status == "success", f"Failed for {node_type}"
            assert call_tracker.get_call_order() == [
                "reducer",
                "effect",
            ], f"Wrong order for {node_type}"


@pytest.mark.integration
class TestA4IdempotentReplay:
    """Test A4: Re-emit identical event, state unchanged.

    Verifies:
        1. First emission processes event normally
        2. Replay same event (same event_id, correlation_id)
        3. Logical state identical after replay
        4. No duplicate Postgres rows (via idempotency check)
        5. Effect idempotency store tracks completed backends
    """

    @pytest.mark.asyncio
    async def test_a4_idempotent_replay(
        self,
        tracked_reducer: TrackedRegistrationReducer,
        tracked_effect: TrackedNodeRegistryEffect,
        call_tracker: CallOrderTracker,
        consul_client: StubConsulClient,
        postgres_adapter: StubPostgresAdapter,
        sample_introspection_event: ModelNodeIntrospectionEvent,
    ) -> None:
        """Re-emit identical event, state unchanged.

        This test verifies idempotency at both reducer and effect layers:
        - Reducer: Uses last_processed_event_id to detect duplicates
        - Effect: Uses idempotency store to skip completed backends
        """
        event = sample_introspection_event
        initial_state = ModelRegistrationState()

        # === FIRST EMISSION ===
        # Process through reducer
        reducer_output_1 = tracked_reducer.reduce(initial_state, event)
        state_after_first = reducer_output_1.result

        # Verify first emission produced pending state with intents
        assert state_after_first.status == "pending"
        assert state_after_first.node_id == event.node_id
        assert len(reducer_output_1.intents) == 2

        # Process through effect
        request_1 = _convert_intents_to_request(event, reducer_output_1.intents)
        effect_response_1 = await tracked_effect.register_node(request_1)

        # Verify first emission succeeded
        assert effect_response_1.status == "success"
        first_consul_call_count = consul_client.call_count
        first_postgres_call_count = postgres_adapter.call_count
        first_reducer_call_count = tracked_reducer.reduce_call_count
        assert first_consul_call_count == 1
        assert first_postgres_call_count == 1
        assert first_reducer_call_count == 1

        # === REPLAY SAME EVENT (Reducer Layer Idempotency) ===
        # Use the state AFTER first emission (which has last_processed_event_id set)
        # This simulates the orchestrator loading state from projection store
        reducer_output_2 = tracked_reducer.reduce(state_after_first, event)
        state_after_replay = reducer_output_2.result

        # Verify reducer detected duplicate and returned same state with NO intents
        assert state_after_replay.status == state_after_first.status
        assert state_after_replay.node_id == state_after_first.node_id
        assert len(reducer_output_2.intents) == 0, (
            "Duplicate event should produce no intents"
        )

        # Reducer was called but detected duplicate
        assert tracked_reducer.reduce_call_count == 2

        # === REPLAY SAME EVENT (Effect Layer Idempotency) ===
        # Even if we call effect with same request, idempotency store should skip
        effect_response_2 = await tracked_effect.register_node(request_1)

        # Verify second emission also succeeded (via idempotency)
        assert effect_response_2.status == "success"

        # Verify backends were NOT called again (idempotency store tracked them)
        assert consul_client.call_count == first_consul_call_count, (
            "Consul should NOT be called again (idempotency)"
        )
        assert postgres_adapter.call_count == first_postgres_call_count, (
            "PostgreSQL should NOT be called again (idempotency)"
        )

        # === VERIFY NO DUPLICATE REGISTRATIONS ===
        # Only one registration per backend
        assert len(consul_client.registrations) == 1, (
            "Should have exactly 1 Consul registration"
        )
        assert len(postgres_adapter.registrations) == 1, (
            "Should have exactly 1 PostgreSQL registration"
        )

        # === VERIFY IDEMPOTENCY STORE STATE ===
        completed_backends = await tracked_effect.get_completed_backends(
            event.correlation_id
        )
        assert "consul" in completed_backends
        assert "postgres" in completed_backends

    @pytest.mark.asyncio
    async def test_a4_reducer_idempotency_with_event_id(
        self,
        tracked_reducer: TrackedRegistrationReducer,
        introspection_event_factory: Callable[..., ModelNodeIntrospectionEvent],
    ) -> None:
        """Verify reducer uses event_id for idempotency detection.

        The reducer should detect duplicate events based on correlation_id
        matching last_processed_event_id in the state.
        """
        correlation_id = uuid4()
        node_id = uuid4()

        # Create event with fixed correlation_id
        event = introspection_event_factory(
            correlation_id=correlation_id,
            node_id=node_id,
        )
        initial_state = ModelRegistrationState()

        # First processing
        output_1 = tracked_reducer.reduce(initial_state, event)
        assert output_1.result.status == "pending"
        assert output_1.result.last_processed_event_id == correlation_id
        assert len(output_1.intents) == 2

        # Replay with same event (same correlation_id)
        output_2 = tracked_reducer.reduce(output_1.result, event)
        assert output_2.result.status == "pending"  # State unchanged
        assert len(output_2.intents) == 0  # No new intents

        # State should be logically identical
        assert output_2.result.node_id == output_1.result.node_id
        assert output_2.result.consul_confirmed == output_1.result.consul_confirmed
        assert output_2.result.postgres_confirmed == output_1.result.postgres_confirmed

    @pytest.mark.asyncio
    async def test_a4_effect_idempotency_store_tracks_backends(
        self,
        tracked_effect: TrackedNodeRegistryEffect,
        consul_client: StubConsulClient,
        postgres_adapter: StubPostgresAdapter,
        sample_introspection_event: ModelNodeIntrospectionEvent,
    ) -> None:
        """Verify effect idempotency store correctly tracks completed backends.

        The effect should track which backends have completed for each
        correlation_id, enabling safe retries.
        """
        event = sample_introspection_event
        request = ModelRegistryRequest(
            node_id=event.node_id,
            node_type=event.node_type,
            node_version=event.node_version,
            correlation_id=event.correlation_id,
            endpoints=dict(event.endpoints),
        )

        # Before registration
        completed_before = await tracked_effect.get_completed_backends(
            event.correlation_id
        )
        assert len(completed_before) == 0

        # After registration
        response = await tracked_effect.register_node(request)
        assert response.status == "success"

        # Check completed backends
        completed_after = await tracked_effect.get_completed_backends(
            event.correlation_id
        )
        assert "consul" in completed_after
        assert "postgres" in completed_after

        # Verify call counts
        assert consul_client.call_count == 1
        assert postgres_adapter.call_count == 1

        # Replay - should not call backends again
        response_2 = await tracked_effect.register_node(request)
        assert response_2.status == "success"
        assert consul_client.call_count == 1  # Unchanged
        assert postgres_adapter.call_count == 1  # Unchanged

    @pytest.mark.asyncio
    async def test_a4_different_correlation_ids_processed_independently(
        self,
        tracked_reducer: TrackedRegistrationReducer,
        tracked_effect: TrackedNodeRegistryEffect,
        consul_client: StubConsulClient,
        postgres_adapter: StubPostgresAdapter,
        introspection_event_factory: Callable[..., ModelNodeIntrospectionEvent],
    ) -> None:
        """Verify different correlation IDs are processed independently.

        Events with different correlation IDs should not affect each other's
        idempotency tracking.
        """
        # Create two events with same node_id but different correlation_ids
        node_id = uuid4()
        event_1 = introspection_event_factory(node_id=node_id, correlation_id=uuid4())
        event_2 = introspection_event_factory(node_id=node_id, correlation_id=uuid4())

        # Process first event
        state_1 = ModelRegistrationState()
        output_1 = tracked_reducer.reduce(state_1, event_1)
        request_1 = _convert_intents_to_request(event_1, output_1.intents)
        await tracked_effect.register_node(request_1)

        # Process second event (different correlation_id)
        state_2 = ModelRegistrationState()
        output_2 = tracked_reducer.reduce(state_2, event_2)

        # Second event should produce intents (not detected as duplicate)
        assert len(output_2.intents) == 2

        # Execute second event through effect
        request_2 = _convert_intents_to_request(event_2, output_2.intents)
        await tracked_effect.register_node(request_2)

        # Both should have been processed independently
        assert consul_client.call_count == 2
        assert postgres_adapter.call_count == 2

        # Idempotency stores track them separately
        completed_1 = await tracked_effect.get_completed_backends(
            event_1.correlation_id
        )
        completed_2 = await tracked_effect.get_completed_backends(
            event_2.correlation_id
        )
        assert "consul" in completed_1
        assert "postgres" in completed_1
        assert "consul" in completed_2
        assert "postgres" in completed_2

    @pytest.mark.asyncio
    async def test_a4_state_immutability_on_replay(
        self,
        tracked_reducer: TrackedRegistrationReducer,
        sample_introspection_event: ModelNodeIntrospectionEvent,
    ) -> None:
        """Verify state is immutable and unchanged on replay.

        The ModelRegistrationState is frozen (immutable). Replay should
        return an identical state without modifying the original.
        """
        event = sample_introspection_event
        initial_state = ModelRegistrationState()

        # First processing
        output_1 = tracked_reducer.reduce(initial_state, event)
        state_after_first = output_1.result

        # Capture state values
        original_status = state_after_first.status
        original_node_id = state_after_first.node_id
        original_last_event_id = state_after_first.last_processed_event_id

        # Replay
        output_2 = tracked_reducer.reduce(state_after_first, event)
        state_after_replay = output_2.result

        # Original state should be unchanged (immutable)
        assert state_after_first.status == original_status
        assert state_after_first.node_id == original_node_id
        assert state_after_first.last_processed_event_id == original_last_event_id

        # Returned state should be identical (or same instance for idempotent case)
        assert state_after_replay.status == state_after_first.status
        assert state_after_replay.node_id == state_after_first.node_id
        assert (
            state_after_replay.last_processed_event_id
            == state_after_first.last_processed_event_id
        )


@pytest.mark.integration
class TestOrchestratedWorkflowIntegration:
    """Additional integration tests for the orchestrated workflow.

    These tests verify edge cases and error handling in the workflow.
    """

    @pytest.mark.asyncio
    async def test_workflow_with_partial_failure_then_retry(
        self,
        tracked_reducer: TrackedRegistrationReducer,
        tracked_effect: TrackedNodeRegistryEffect,
        consul_client: StubConsulClient,
        postgres_adapter: StubPostgresAdapter,
        sample_introspection_event: ModelNodeIntrospectionEvent,
    ) -> None:
        """Test workflow handles partial failure and retry correctly.

        Scenario:
        1. First attempt: Consul fails, PostgreSQL succeeds
        2. Retry: Consul fixed, PostgreSQL skipped (idempotency)
        3. Final: Both backends registered
        """
        event = sample_introspection_event
        initial_state = ModelRegistrationState()

        # Configure Consul to fail initially
        consul_client.should_fail = True
        consul_client.failure_error = "Connection refused"

        # First attempt
        reducer_output = tracked_reducer.reduce(initial_state, event)
        request = _convert_intents_to_request(event, reducer_output.intents)
        response_1 = await tracked_effect.register_node(request)

        # Verify partial failure
        assert response_1.status == "partial"
        assert response_1.consul_result.success is False
        assert response_1.postgres_result.success is True

        # PostgreSQL was registered
        assert len(postgres_adapter.registrations) == 1
        assert len(consul_client.registrations) == 0

        postgres_calls_before_retry = postgres_adapter.call_count

        # Fix Consul for retry
        consul_client.should_fail = False

        # Retry with same correlation_id
        response_2 = await tracked_effect.register_node(request)

        # Verify success
        assert response_2.status == "success"
        assert response_2.consul_result.success is True
        assert response_2.postgres_result.success is True

        # Consul was called again (retry)
        assert consul_client.call_count == 2

        # PostgreSQL was NOT called again (idempotency)
        assert postgres_adapter.call_count == postgres_calls_before_retry

        # Both backends now have registrations
        assert len(consul_client.registrations) == 1
        assert len(postgres_adapter.registrations) == 1

    @pytest.mark.asyncio
    async def test_workflow_correlation_id_propagation(
        self,
        tracked_reducer: TrackedRegistrationReducer,
        tracked_effect: TrackedNodeRegistryEffect,
        call_tracker: CallOrderTracker,
        sample_introspection_event: ModelNodeIntrospectionEvent,
    ) -> None:
        """Verify correlation ID is propagated through entire workflow.

        The same correlation_id should be visible in:
        1. Introspection event
        2. Reducer call tracking
        3. Reducer output intents
        4. Effect request
        5. Effect response
        """
        event = sample_introspection_event
        expected_correlation_id = event.correlation_id

        # Process through workflow
        reducer_output = tracked_reducer.reduce(ModelRegistrationState(), event)
        request = _convert_intents_to_request(event, reducer_output.intents)
        response = await tracked_effect.register_node(request)

        # Verify correlation ID in call tracker
        reducer_calls = call_tracker.get_reducer_calls()
        assert len(reducer_calls) == 1
        assert reducer_calls[0].correlation_id == expected_correlation_id

        effect_calls = call_tracker.get_effect_calls()
        assert len(effect_calls) == 1
        assert effect_calls[0].correlation_id == expected_correlation_id

        # Verify correlation ID in intents
        for intent in reducer_output.intents:
            # Payload is a dict with correlation_id - use equality check
            payload_correlation_id = intent.payload.get("correlation_id")
            # Handle both UUID and string representations
            if isinstance(payload_correlation_id, str):
                assert payload_correlation_id == str(expected_correlation_id), (
                    f"Expected correlation_id {expected_correlation_id}, "
                    f"got {payload_correlation_id}"
                )
            else:
                assert payload_correlation_id == expected_correlation_id, (
                    f"Expected correlation_id {expected_correlation_id}, "
                    f"got {payload_correlation_id}"
                )

        # Verify correlation ID in response
        assert response.correlation_id == expected_correlation_id
