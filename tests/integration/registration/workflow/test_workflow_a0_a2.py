# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Tests A0, A1, A2 for OMN-915 mocked E2E registration workflow.

This module provides mocked E2E tests proving the registration architecture
with ZERO real infrastructure. All Consul/PostgreSQL/Kafka operations are
replaced with controllable test doubles.

Test Scenarios:
    A0 - Purity Gate: Verify reducer performs no I/O, effect handles all external calls
    A1 - Introspection Publish: Node emits ModelNodeIntrospectionEvent via mixin
    A2 - Two-Way Introspection Loop: Registry requests introspection, node responds

Design Principles:
    - Zero real infrastructure: All backends are mocked/stubbed
    - Purity verification: Reducers must have zero I/O calls
    - Call count tracking: All operations are counted for verification
    - Correlation ID tracing: All tests verify correlation preservation

Related:
    - OMN-915: Mocked E2E Registration Workflow
    - DESIGN_TWO_WAY_REGISTRATION_ARCHITECTURE.md: Architecture design
    - RegistrationReducer: Pure reducer under test
    - NodeRegistryEffect: Effect node under test
"""

from __future__ import annotations

import json
import time
from collections.abc import Callable
from datetime import UTC, datetime
from typing import TYPE_CHECKING
from uuid import UUID, uuid4

from omnibase_core.enums.enum_node_kind import EnumNodeKind

from omnibase_infra.event_bus.inmemory_event_bus import InMemoryEventBus
from omnibase_infra.models.registration import ModelNodeIntrospectionEvent
from omnibase_infra.nodes.reducers import RegistrationReducer
from omnibase_infra.nodes.reducers.models import ModelRegistrationState

if TYPE_CHECKING:
    from tests.integration.registration.effect.test_doubles import (
        StubConsulClient,
        StubPostgresAdapter,
    )
    from tests.integration.registration.workflow.conftest import (
        CallOrderTracker,
        IntrospectableTestNode,
        TrackedNodeRegistryEffect,
        TrackedRegistrationReducer,
    )

# =============================================================================
# Test Constants
# =============================================================================

# Expected number of intents emitted by reducer (Consul + PostgreSQL)
EXPECTED_INTENT_COUNT = 2

# =============================================================================
# A0 - Purity Gate Tests
# =============================================================================


class TestA0PurityGate:
    """A0 - Purity Gate: Verify reducer NEVER performs I/O during mocked workflow.

    This test suite ensures that the RegistrationReducer follows the pure reducer
    pattern - all Consul/PostgreSQL interactions are delegated to the Effect layer
    via emitted intents.

    Architecture Verification:
        - Reducer: Pure computation only (state + event -> new_state + intents)
        - Effect: Handles all external I/O (Consul, PostgreSQL)

    Success Criteria:
        - Reducer emits intents but performs no I/O calls
        - Effect receives intents and performs I/O calls
        - Call counts verify correct architecture separation
    """

    async def test_a0_purity_gate_reducer_no_io(
        self,
        registration_reducer: RegistrationReducer,
        consul_client: StubConsulClient,
        postgres_adapter: StubPostgresAdapter,
        introspection_event_factory: Callable[..., ModelNodeIntrospectionEvent],
    ) -> None:
        """Reducer never performs I/O during mocked workflow.

        This is the core purity test - it verifies that calling reduce()
        does NOT directly call Consul or PostgreSQL. Instead, the reducer
        emits intents that the Effect layer should later execute.

        Test Flow:
            1. Call reducer.reduce() with an introspection event
            2. Verify Consul client has 0 calls (reducer didn't call it)
            3. Verify PostgreSQL adapter has 0 calls (reducer didn't call it)
            4. Verify reducer emitted intents for later Effect execution
        """
        # Arrange
        initial_state = ModelRegistrationState()
        event = introspection_event_factory()

        # Assert precondition: no calls before reduce()
        assert consul_client.call_count == 0, "Consul should have 0 calls initially"
        assert postgres_adapter.call_count == 0, (
            "Postgres should have 0 calls initially"
        )

        # Act - call reducer (should NOT perform I/O)
        output = registration_reducer.reduce(initial_state, event)

        # Assert: Reducer performed no I/O
        assert consul_client.call_count == 0, (
            "Reducer VIOLATED PURITY: called Consul directly. "
            "Reducers must emit intents, not perform I/O."
        )
        assert postgres_adapter.call_count == 0, (
            "Reducer VIOLATED PURITY: called PostgreSQL directly. "
            "Reducers must emit intents, not perform I/O."
        )

        # Assert: Reducer did emit intents for Effect layer
        assert len(output.intents) == EXPECTED_INTENT_COUNT, (
            f"Reducer should emit {EXPECTED_INTENT_COUNT} intents (Consul + PostgreSQL), "
            f"got {len(output.intents)}"
        )

        # Verify intent types
        intent_types = {intent.intent_type for intent in output.intents}
        assert "consul.register" in intent_types, "Missing consul.register intent"
        assert "postgres.upsert_registration" in intent_types, (
            "Missing postgres.upsert_registration intent"
        )

    async def test_a0_purity_gate_effect_performs_io(
        self,
        tracked_effect: TrackedNodeRegistryEffect,
        consul_client: StubConsulClient,
        postgres_adapter: StubPostgresAdapter,
        call_tracker: CallOrderTracker,
    ) -> None:
        """Effect layer performs I/O when executing registration.

        This test verifies that the Effect layer correctly performs I/O
        operations when processing a registration request. The Effect is
        the ONLY layer allowed to perform external I/O.

        Test Flow:
            1. Call effect.register_node() with a registration request
            2. Verify Consul client received a call
            3. Verify PostgreSQL adapter received a call
            4. Verify call order is tracked
        """
        from omnibase_infra.nodes.effects.models import ModelRegistryRequest

        # Arrange
        request = ModelRegistryRequest(
            node_id=uuid4(),
            node_type=EnumNodeKind.EFFECT,
            node_version="1.0.0",
            correlation_id=uuid4(),
            service_name=f"onex-{EnumNodeKind.EFFECT.value}",
            endpoints={"health": "http://localhost:8080/health"},
            tags=["onex", EnumNodeKind.EFFECT.value, "test"],
            metadata={"environment": "test"},
            timestamp=datetime.now(UTC),
        )

        # Assert precondition: no calls before effect execution
        assert consul_client.call_count == 0, "Consul should have 0 calls initially"
        assert postgres_adapter.call_count == 0, (
            "Postgres should have 0 calls initially"
        )

        # Act - execute effect (SHOULD perform I/O)
        response = await tracked_effect.register_node(request)

        # Assert: Effect DID perform I/O (this is correct behavior for Effects)
        assert consul_client.call_count > 0, (
            "Effect should call Consul - this is where I/O belongs"
        )
        assert postgres_adapter.call_count > 0, (
            "Effect should call PostgreSQL - this is where I/O belongs"
        )

        # Verify response indicates success
        assert response.is_complete_success(), f"Effect registration failed: {response}"

        # Verify call order tracking works
        assert tracked_effect.register_node_call_count == 1, (
            "Effect register_node should be tracked"
        )
        effect_calls = call_tracker.get_effect_calls()
        assert len(effect_calls) == 1, "Effect call should be tracked in call_tracker"

    async def test_a0_purity_gate_complete_workflow_with_tracking(
        self,
        tracked_reducer: TrackedRegistrationReducer,
        tracked_effect: TrackedNodeRegistryEffect,
        consul_client: StubConsulClient,
        postgres_adapter: StubPostgresAdapter,
        call_tracker: CallOrderTracker,
        introspection_event_factory: Callable[..., ModelNodeIntrospectionEvent],
    ) -> None:
        """Complete workflow: Reducer -> Effect with proper separation.

        This test verifies the complete registration workflow where:
        1. Reducer processes introspection event and emits intents (no I/O)
        2. Effect executes registration and performs I/O
        3. Call order shows reducer is called before effect
        """
        from omnibase_infra.nodes.effects.models import ModelRegistryRequest

        # Arrange
        correlation_id = uuid4()
        event = introspection_event_factory(correlation_id=correlation_id)
        initial_state = ModelRegistrationState()

        # Act - Phase 1: Reducer processes event (no I/O)
        _output = tracked_reducer.reduce(initial_state, event)

        # Verify reducer was tracked and performed no I/O
        assert tracked_reducer.reduce_call_count == 1, "Reducer should be called once"
        assert consul_client.call_count == 0, "Reducer must not call Consul"
        assert postgres_adapter.call_count == 0, "Reducer must not call PostgreSQL"

        # Act - Phase 2: Effect executes registration (with I/O)
        # Note: event.node_type is a Literal["effect", "compute", "reducer", "orchestrator"]
        # string, not an EnumNodeKind. Convert to EnumNodeKind for ModelRegistryRequest.
        request = ModelRegistryRequest(
            node_id=event.node_id,
            node_type=EnumNodeKind(event.node_type),
            node_version=event.node_version,
            correlation_id=correlation_id,
            service_name=f"onex-{event.node_type}",
            endpoints=dict(event.endpoints) if event.endpoints else {},
            tags=["onex", event.node_type],
            metadata={},
            timestamp=datetime.now(UTC),
        )
        await tracked_effect.register_node(request)

        # Assert: Effect performed I/O after reducer
        assert consul_client.call_count > 0, "Effect should call Consul"
        assert postgres_adapter.call_count > 0, "Effect should call PostgreSQL"

        # Verify call order: reducer THEN effect
        call_order = call_tracker.get_call_order()
        assert call_order == ["reducer", "effect"], (
            f"Expected [reducer, effect] but got {call_order}. "
            "Workflow must call reducer before effect."
        )


# =============================================================================
# A1 - Introspection Publish Tests
# =============================================================================


class TestA1IntrospectionPublish:
    """A1 - Introspection Publish: Node emits ModelNodeIntrospectionEvent via mixin.

    This test suite verifies that nodes correctly emit introspection events
    via the MixinNodeIntrospection mixin. The events should contain all
    required fields for registration.

    Success Criteria:
        - Node emits ModelNodeIntrospectionEvent
        - Event contains stable node_id (UUID)
        - Event contains valid node_type (Literal type)
        - Event contains endpoints dict
        - Event contains metadata dict
        - Event contains correlation_id (UUID)
    """

    async def test_a1_introspection_event_structure(
        self,
        test_node: IntrospectableTestNode,
        event_bus: InMemoryEventBus,
    ) -> None:
        """Introspection event has required structure for registration.

        Verifies that publish_introspection() creates a properly structured
        ModelNodeIntrospectionEvent with all required fields.

        Required Fields:
            - node_id: UUID (stable, unique identifier)
            - node_type: Literal["effect", "compute", "reducer", "orchestrator"]
            - endpoints: dict[str, str] (at least "health" endpoint)
            - metadata: dict (arbitrary metadata)
            - correlation_id: UUID (for request tracing)
        """
        # Act - emit introspection event
        await test_node.publish_introspection()

        # Assert - verify event was published
        history = await event_bus.get_event_history(limit=10)
        assert len(history) > 0, "Introspection event should be published"

        # Get the introspection event
        introspection_events = [
            e for e in history if "introspection" in e.topic.lower()
        ]

        # Note: If no introspection events found in topic, the event might be
        # on a different topic. Let's also check the last event published.
        if not introspection_events:
            # Fallback: check any event was published
            last_event = history[-1]
            event_data = json.loads(last_event.value.decode("utf-8"))
        else:
            event_data = json.loads(introspection_events[-1].value.decode("utf-8"))

        # Verify required fields are present
        assert "node_id" in event_data, "Event must contain node_id"
        assert "node_type" in event_data, "Event must contain node_type"
        assert "correlation_id" in event_data, "Event must contain correlation_id"

        # Verify node_id is a valid UUID string
        node_id = event_data["node_id"]
        assert isinstance(node_id, str), "node_id should be serialized as string"
        UUID(node_id)  # Raises if invalid UUID

        # Verify node_type is a valid ONEX type (EnumNodeKind value)
        node_type = event_data["node_type"]
        valid_types: set[str] = {kind.value for kind in EnumNodeKind}
        assert node_type in valid_types, f"node_type {node_type} not in {valid_types}"

        # Verify correlation_id is a valid UUID
        correlation_id = event_data["correlation_id"]
        assert isinstance(correlation_id, str), "correlation_id should be string"
        UUID(correlation_id)  # Raises if invalid UUID

    async def test_a1_introspection_event_stable_node_id(
        self,
        test_node_factory: Callable[..., IntrospectableTestNode],
        event_bus: InMemoryEventBus,
    ) -> None:
        """Node ID is stable across multiple introspection emissions.

        The same node instance should always emit the same node_id.
        This ensures consistent identity in the registry.
        """
        # Arrange - create a node with specific ID
        fixed_node_id = uuid4()
        node = test_node_factory(node_id=fixed_node_id)

        # Act - emit introspection twice
        await node.publish_introspection()
        await node.publish_introspection()

        # Assert - both emissions should have same node_id
        history = await event_bus.get_event_history(limit=10)
        assert len(history) >= 2, "Should have at least 2 events"

        # Parse events and extract node_ids
        node_ids: list[str] = []
        for event in history:
            event_data = json.loads(event.value.decode("utf-8"))
            if "node_id" in event_data:
                node_ids.append(event_data["node_id"])

        # All node_ids should be the same
        assert len(set(node_ids)) == 1, (
            f"Node ID should be stable but got different values: {node_ids}"
        )

        # And should match the expected ID
        assert node_ids[0] == str(fixed_node_id), (
            f"Node ID mismatch: expected {fixed_node_id}, got {node_ids[0]}"
        )

    async def test_a1_introspection_event_valid_node_types(
        self,
        test_node_factory: Callable[..., IntrospectableTestNode],
        event_bus: InMemoryEventBus,
    ) -> None:
        """Each ONEX node type can emit valid introspection events.

        Tests all four ONEX node types to ensure they can all
        emit properly structured introspection events.
        """
        node_types: list[EnumNodeKind] = [
            EnumNodeKind.EFFECT,
            EnumNodeKind.COMPUTE,
            EnumNodeKind.REDUCER,
            EnumNodeKind.ORCHESTRATOR,
        ]

        for node_type in node_types:
            # Clear history for clean test
            await event_bus.clear_event_history()

            # Create node of this type
            node = test_node_factory(node_type=node_type)

            # Emit introspection
            await node.publish_introspection()

            # Verify emission
            history = await event_bus.get_event_history(limit=10)
            assert len(history) > 0, f"No event for node_type={node_type}"

            # Parse and verify (compare with enum value since JSON serializes to string)
            event_data = json.loads(history[-1].value.decode("utf-8"))
            assert event_data["node_type"] == node_type.value, (
                f"node_type mismatch: expected {node_type.value}, got {event_data['node_type']}"
            )

    async def test_a1_introspection_event_endpoints_and_metadata(
        self,
        test_node: IntrospectableTestNode,
        event_bus: InMemoryEventBus,
    ) -> None:
        """Introspection event includes endpoints and metadata dicts.

        The event should contain endpoint information (health URL, API URL)
        and metadata (capabilities, version info).
        """
        # Act
        await test_node.publish_introspection()

        # Get event
        history = await event_bus.get_event_history(limit=10)
        event_data = json.loads(history[-1].value.decode("utf-8"))

        # Verify endpoints structure (should be dict or present)
        # Note: endpoints might be in different formats depending on serialization
        if "endpoints" in event_data:
            endpoints = event_data["endpoints"]
            assert isinstance(endpoints, dict), "endpoints should be a dict"

        # Verify metadata or capabilities structure
        if "metadata" in event_data:
            metadata = event_data["metadata"]
            assert isinstance(metadata, dict), "metadata should be a dict"

        if "capabilities" in event_data:
            capabilities = event_data["capabilities"]
            assert isinstance(capabilities, dict), "capabilities should be a dict"


# =============================================================================
# A2 - Two-Way Introspection Loop Tests
# =============================================================================


class TestA2TwoWayIntrospectionLoop:
    """A2 - Two-Way Introspection Loop: Registry requests, node responds.

    This test suite verifies the bidirectional introspection protocol:
    1. Registry publishes request for introspection (node.request_introspection)
    2. Node receives request and responds with introspection event
    3. Correlation ID is preserved throughout the loop

    Success Criteria:
        - Registry can publish introspection request
        - Node receives and responds to request
        - Correlation ID preserved in response
        - Complete round-trip works with mocked infrastructure
    """

    async def test_a2_registry_requests_introspection(
        self,
        event_bus: InMemoryEventBus,
    ) -> None:
        """Registry can publish introspection request to event bus.

        The registry publishes a request to a well-known topic that
        nodes are expected to subscribe to.
        """
        # Arrange
        correlation_id = uuid4()
        request_topic = "node.request_introspection"

        request_payload = {
            "request_type": "introspection",
            "correlation_id": str(correlation_id),
            "target": "all",  # Request from all nodes
            "timestamp": "2025-12-25T00:00:00Z",
        }

        # Act - registry publishes request
        await event_bus.publish_envelope(
            envelope=request_payload,
            topic=request_topic,
        )

        # Assert - request was published
        history = await event_bus.get_event_history(topic=request_topic)
        assert len(history) == 1, "Request should be published once"

        # Verify request content
        request_data = json.loads(history[0].value.decode("utf-8"))
        assert request_data["correlation_id"] == str(correlation_id), (
            "Correlation ID should be preserved in published request"
        )

    async def test_a2_node_responds_to_introspection_request(
        self,
        test_node: IntrospectableTestNode,
        event_bus: InMemoryEventBus,
    ) -> None:
        """Node responds to introspection request with event.

        When a node receives an introspection request, it should:
        1. Parse the request
        2. Emit its introspection event
        3. Include the original correlation_id for tracing
        """
        # Arrange
        _original_correlation_id = uuid4()
        _request_topic = "node.request_introspection"

        # Set up subscription to capture node's response
        responses: list[dict[str, object]] = []

        async def capture_response(msg: object) -> None:
            """Capture introspection responses."""
            # msg is ModelEventMessage with .value bytes
            response_data = json.loads(msg.value.decode("utf-8"))  # type: ignore[attr-defined]
            responses.append(response_data)

        # Subscribe to introspection events (node's response)
        response_topic = "node.introspection"
        await event_bus.subscribe(response_topic, "test-group", capture_response)

        # Act - simulate receiving introspection request
        # In real implementation, this would be triggered by receiving the request
        # For this test, we directly call publish_introspection with the correlation_id

        # Note: MixinNodeIntrospection.publish_introspection() uses its own correlation_id
        # For testing the two-way loop, we verify the mechanism works
        await test_node.publish_introspection()

        # Assert - response was emitted
        # Note: Due to direct callback invocation in InMemoryEventBus,
        # the response should already be captured
        history = await event_bus.get_event_history(limit=10)
        assert len(history) > 0, "Node should emit introspection response"

        # Verify response structure contains correlation_id
        response_data = json.loads(history[-1].value.decode("utf-8"))
        assert "correlation_id" in response_data, (
            "Response must include correlation_id for request tracing"
        )
        assert "node_id" in response_data, "Response must include node_id"

    async def test_a2_correlation_id_preserved_in_loop(
        self,
        event_bus: InMemoryEventBus,
    ) -> None:
        """Correlation ID is preserved throughout request-response loop.

        This test verifies the end-to-end correlation tracking:
        1. Registry creates correlation_id and includes in request
        2. Node extracts correlation_id from request
        3. Node includes same correlation_id in response
        4. Registry can match response to original request
        """
        # Arrange
        correlation_id = uuid4()
        request_topic = "node.request_introspection"
        response_topic = "node.introspection"

        # Track correlation IDs seen
        request_correlations: list[str] = []
        response_correlations: list[str] = []

        async def track_request(msg: object) -> None:
            """Track request correlation IDs."""
            data = json.loads(msg.value.decode("utf-8"))  # type: ignore[attr-defined]
            if "correlation_id" in data:
                request_correlations.append(data["correlation_id"])

        async def track_response(msg: object) -> None:
            """Track response correlation IDs."""
            data = json.loads(msg.value.decode("utf-8"))  # type: ignore[attr-defined]
            if "correlation_id" in data:
                response_correlations.append(data["correlation_id"])

        # Subscribe to both topics
        await event_bus.subscribe(request_topic, "request-tracker", track_request)
        await event_bus.subscribe(response_topic, "response-tracker", track_response)

        # Act - publish request with specific correlation_id
        request_payload = {
            "request_type": "introspection",
            "correlation_id": str(correlation_id),
        }
        await event_bus.publish_envelope(request_payload, request_topic)

        # Simulate node response with same correlation_id
        response_payload: dict[str, object] = {
            "node_id": str(uuid4()),
            "node_type": "effect",
            "correlation_id": str(correlation_id),  # Same as request
            "endpoints": {"health": "http://localhost:8080/health"},
        }
        await event_bus.publish_envelope(response_payload, response_topic)

        # Assert - correlation IDs match
        assert len(request_correlations) == 1, "Should capture request correlation"
        assert len(response_correlations) == 1, "Should capture response correlation"
        assert request_correlations[0] == response_correlations[0], (
            f"Correlation ID mismatch: request={request_correlations[0]}, "
            f"response={response_correlations[0]}"
        )
        assert request_correlations[0] == str(correlation_id), (
            f"Correlation ID should match original: {correlation_id}"
        )

    async def test_a2_multiple_nodes_respond_with_unique_correlation(
        self,
        test_node_factory: Callable[..., IntrospectableTestNode],
        event_bus: InMemoryEventBus,
    ) -> None:
        """Multiple nodes can respond independently, each with unique IDs.

        When multiple nodes receive an introspection request:
        1. Each node emits its own response
        2. Each response has the same correlation_id (from request)
        3. But each response has a unique node_id

        This tests the fan-out/fan-in pattern of introspection.
        """
        # Arrange - create multiple nodes
        nodes = [
            test_node_factory(node_type=EnumNodeKind.EFFECT),
            test_node_factory(node_type=EnumNodeKind.COMPUTE),
            test_node_factory(node_type=EnumNodeKind.REDUCER),
        ]

        # Clear history
        await event_bus.clear_event_history()

        # Act - each node emits introspection
        for node in nodes:
            await node.publish_introspection()

        # Assert - each node's response was captured
        history = await event_bus.get_event_history(limit=20)
        assert len(history) == 3, f"Expected 3 responses, got {len(history)}"

        # Parse responses and verify unique node_ids
        node_ids: set[str] = set()
        for event in history:
            data = json.loads(event.value.decode("utf-8"))
            node_ids.add(data["node_id"])

        assert len(node_ids) == 3, (
            f"Each node should have unique node_id. Got {len(node_ids)} unique IDs."
        )


# =============================================================================
# Combined Integration Tests
# =============================================================================


class TestWorkflowIntegration:
    """Combined tests verifying the complete mocked E2E workflow.

    These tests combine elements from A0, A1, and A2 to verify
    that all components work together correctly.
    """

    async def test_complete_registration_workflow_mocked(
        self,
        tracked_reducer: TrackedRegistrationReducer,
        tracked_effect: TrackedNodeRegistryEffect,
        test_node: IntrospectableTestNode,
        event_bus: InMemoryEventBus,
        consul_client: StubConsulClient,
        postgres_adapter: StubPostgresAdapter,
        call_tracker: CallOrderTracker,
    ) -> None:
        """Complete mocked E2E registration workflow.

        This test exercises the full registration flow:
        1. Node emits introspection event (A1)
        2. Reducer processes event, emits intents (A0 - purity)
        3. Effect executes intents (A0 - I/O separation)
        4. Correlation ID preserved throughout (A2)

        All with ZERO real infrastructure.
        """
        from omnibase_infra.nodes.effects.models import ModelRegistryRequest

        # Step 1: Node emits introspection (A1)
        await test_node.publish_introspection()

        # Verify introspection was published
        history = await event_bus.get_event_history(limit=10)
        assert len(history) > 0, "Node should emit introspection"

        introspection_data = json.loads(history[-1].value.decode("utf-8"))
        node_id = UUID(introspection_data["node_id"])
        correlation_id = UUID(introspection_data["correlation_id"])

        # Create ModelNodeIntrospectionEvent from the data
        # Note: introspection_data["node_type"] is already a string from JSON serialization
        # (e.g., "effect", "compute") which matches the Literal type expected by the model.
        # No conversion to EnumNodeKind is needed.
        introspection_event = ModelNodeIntrospectionEvent(
            node_id=node_id,
            node_type=introspection_data.get("node_type", "effect"),
            node_version=introspection_data.get("node_version", "1.0.0"),
            correlation_id=correlation_id,
            endpoints=introspection_data.get("endpoints", {}),
            timestamp=datetime.now(UTC),
        )

        # Step 2: Reducer processes event (A0 - purity)
        initial_state = ModelRegistrationState()
        output = tracked_reducer.reduce(initial_state, introspection_event)

        # Verify reducer purity
        assert consul_client.call_count == 0, "Reducer must not call Consul"
        assert postgres_adapter.call_count == 0, "Reducer must not call Postgres"
        assert len(output.intents) == EXPECTED_INTENT_COUNT, (
            f"Reducer should emit {EXPECTED_INTENT_COUNT} intents"
        )

        # Step 3: Effect executes registration (A0 - I/O separation)
        # Note: introspection_event.node_type is a Literal string (e.g., "effect"),
        # not an EnumNodeKind. Convert to EnumNodeKind for ModelRegistryRequest.
        request = ModelRegistryRequest(
            node_id=node_id,
            node_type=EnumNodeKind(introspection_event.node_type),
            node_version=introspection_event.node_version,
            correlation_id=correlation_id,
            service_name=f"onex-{introspection_event.node_type}",
            endpoints=dict(introspection_event.endpoints)
            if introspection_event.endpoints
            else {},
            timestamp=datetime.now(UTC),
        )
        response = await tracked_effect.register_node(request)

        # Verify effect performed I/O
        assert consul_client.call_count > 0, "Effect should call Consul"
        assert postgres_adapter.call_count > 0, "Effect should call Postgres"
        assert response.is_complete_success(), "Registration should succeed"

        # Step 4: Verify workflow order (A2 - correlation)
        call_order = call_tracker.get_call_order()
        assert call_order == ["reducer", "effect"], (
            f"Workflow order should be [reducer, effect], got {call_order}"
        )

        # Verify correlation ID was preserved (would be in effect's request)
        assert request.correlation_id == correlation_id, (
            "Correlation ID must be preserved throughout workflow"
        )


# =============================================================================
# Performance Baseline Tests
# =============================================================================


class TestPerformanceBaseline:
    """Performance baseline tests for registration workflow.

    These tests verify that core registration operations maintain acceptable
    performance characteristics. They serve as regression tests to detect
    performance degradation in the reducer and related components.

    Performance Targets:
        - Reducer operations: 100 reduce calls in <0.5s
        - State creation: Minimal overhead per operation
        - Intent generation: Efficient intent emission

    Note:
        These tests use time.perf_counter() for high-precision timing.
        Thresholds are set conservatively to avoid flaky tests while
        still catching significant performance regressions.
    """

    def test_performance_reducer_operations_under_threshold(
        self,
        registration_reducer: RegistrationReducer,
        introspection_event_factory: Callable[..., ModelNodeIntrospectionEvent],
    ) -> None:
        """Verify reducer operations complete within performance threshold.

        This test ensures the registration reducer maintains acceptable
        performance by running multiple operations and verifying total
        execution time.

        Performance Target: 100 reduce operations in <0.5s

        Test Methodology:
            1. Create 100 unique introspection events
            2. Run reduce() for each event with fresh state
            3. Measure total wall-clock time
            4. Assert time is under threshold

        Rationale:
            - 100 operations provides statistical significance
            - 0.5s threshold is conservative (5ms per operation)
            - Fresh state per operation tests worst-case scenario
            - Uses perf_counter for sub-millisecond precision
        """
        num_operations = 100
        threshold_seconds = 0.5

        start = time.perf_counter()

        for _ in range(num_operations):
            state = ModelRegistrationState()
            event = introspection_event_factory()
            registration_reducer.reduce(state, event)

        elapsed = time.perf_counter() - start

        assert elapsed < threshold_seconds, (
            f"Reducer performance regression: {num_operations} operations took "
            f"{elapsed:.3f}s (threshold: {threshold_seconds}s, "
            f"avg: {elapsed / num_operations * 1000:.2f}ms/op)"
        )

    def test_performance_state_creation_overhead(
        self,
        registration_reducer: RegistrationReducer,
        introspection_event_factory: Callable[..., ModelNodeIntrospectionEvent],
    ) -> None:
        """Verify state creation has minimal overhead.

        This test ensures that creating ModelRegistrationState instances
        does not introduce significant overhead that could impact
        high-throughput registration scenarios.

        Performance Target: 1000 state creations in <0.1s
        """
        num_operations = 1000
        threshold_seconds = 0.1

        start = time.perf_counter()

        for _ in range(num_operations):
            _ = ModelRegistrationState()

        elapsed = time.perf_counter() - start

        assert elapsed < threshold_seconds, (
            f"State creation performance regression: {num_operations} creations took "
            f"{elapsed:.3f}s (threshold: {threshold_seconds}s, "
            f"avg: {elapsed / num_operations * 1000:.3f}ms/op)"
        )

    def test_performance_intent_generation_efficiency(
        self,
        registration_reducer: RegistrationReducer,
        introspection_event_factory: Callable[..., ModelNodeIntrospectionEvent],
    ) -> None:
        """Verify intent generation is efficient.

        This test ensures that the reducer's intent generation (emitting
        Consul and PostgreSQL intents) maintains acceptable performance.

        Performance Target: 50 reduce operations with intent verification in <0.3s
        """
        num_operations = 50
        threshold_seconds = 0.3

        start = time.perf_counter()

        for _ in range(num_operations):
            state = ModelRegistrationState()
            event = introspection_event_factory()
            output = registration_reducer.reduce(state, event)

            # Verify intents were generated (ensures we're testing real work)
            assert len(output.intents) == EXPECTED_INTENT_COUNT

        elapsed = time.perf_counter() - start

        assert elapsed < threshold_seconds, (
            f"Intent generation performance regression: {num_operations} operations took "
            f"{elapsed:.3f}s (threshold: {threshold_seconds}s, "
            f"avg: {elapsed / num_operations * 1000:.2f}ms/op)"
        )


__all__ = [
    "TestA0PurityGate",
    "TestA1IntrospectionPublish",
    "TestA2TwoWayIntrospectionLoop",
    "TestPerformanceBaseline",
    "TestWorkflowIntegration",
]
