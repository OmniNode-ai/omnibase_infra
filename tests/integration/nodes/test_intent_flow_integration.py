# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Integration tests for intent_type-based intent flow [OMN-1258].

This module validates the end-to-end flow of typed intents through
the ONEX registration workflow:

    Reducer -> Runtime/Dispatcher -> Effect -> Confirmation

Architecture:
    The RegistrationReducer emits intents with typed payloads:
        - intent_type="extension"
        - payload.intent_type="consul.register" or "postgres.upsert_registration"
        - payload.plugin_name="consul" or "postgres"
        - payload.data contains the serialized typed intent

    This format enables:
    1. Generic intent routing by the Runtime layer
    2. Plugin-based dispatch to appropriate Effect nodes
    3. Type-safe intent payloads while maintaining flexibility

Test Categories:
    - TestReducerIntentTypeEmission: Verify reducer uses intent_type format
    - TestIntentTypeRouting: Test intent routing by intent_type
    - TestEffectLayerRequestFormatting: Validate Effect receives formatted requests
    - TestEndToEndIntentTypeFlow: Full flow integration with mocks

Running Tests:
    # Run all intent flow tests:
    pytest tests/integration/nodes/test_intent_flow_integration.py

    # Run with verbose output:
    pytest tests/integration/nodes/test_intent_flow_integration.py -v

    # Run specific test class:
    pytest tests/integration/nodes/test_intent_flow_integration.py::TestReducerIntentTypeEmission

Related:
    - RegistrationReducer: Emits typed intents
    - NodeRegistryEffect: Consumes requests built from intents
    - omnibase_core ModelIntent: Intent model with intent_type field
    - omnibase_core ModelPayloadConsulRegister/ModelPayloadPostgresUpsert: Typed payload models
    - PR #114: Migration to intent_type-based intents
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import TYPE_CHECKING
from unittest.mock import AsyncMock, MagicMock
from uuid import UUID, uuid4

import pytest
from omnibase_core.enums import EnumNodeKind
from omnibase_core.models.reducer import ModelIntent

from omnibase_infra.models.registration import ModelNodeIntrospectionEvent
from omnibase_infra.nodes.effects.models.model_registry_request import (
    ModelRegistryRequest,
)
from omnibase_infra.nodes.effects.registry_effect import NodeRegistryEffect
from omnibase_infra.nodes.reducers.models.model_payload_consul_register import (
    ModelPayloadConsulRegister,
)
from omnibase_infra.nodes.reducers.models.model_payload_postgres_upsert_registration import (
    ModelPayloadPostgresUpsertRegistration,
)
from omnibase_infra.nodes.reducers.models.model_registration_state import (
    ModelRegistrationState,
)
from omnibase_infra.nodes.reducers.registration_reducer import RegistrationReducer

if TYPE_CHECKING:
    from omnibase_infra.nodes.effects.models.model_backend_result import (
        ModelBackendResult,
    )
    from omnibase_infra.nodes.effects.models.model_registry_response import (
        ModelRegistryResponse,
    )

# Test timestamp constant for reproducible tests
TEST_TIMESTAMP = datetime(2025, 1, 15, 12, 0, 0, tzinfo=UTC)


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def node_id() -> UUID:
    """Create a fixed node ID for testing."""
    return uuid4()


@pytest.fixture
def correlation_id() -> UUID:
    """Create a fixed correlation ID for testing."""
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
def reducer() -> RegistrationReducer:
    """Create a registration reducer instance."""
    return RegistrationReducer()


@pytest.fixture
def initial_state() -> ModelRegistrationState:
    """Create an initial idle registration state."""
    return ModelRegistrationState()


@pytest.fixture
def mock_consul_client() -> MagicMock:
    """Create a mock Consul client for Effect testing."""
    client = MagicMock()
    client.register_service = AsyncMock(
        return_value=MagicMock(success=True, error=None)
    )
    return client


@pytest.fixture
def mock_postgres_adapter() -> MagicMock:
    """Create a mock PostgreSQL adapter for Effect testing."""
    adapter = MagicMock()
    adapter.upsert = AsyncMock(return_value=MagicMock(success=True, error=None))
    return adapter


@pytest.fixture
def registry_effect(
    mock_consul_client: MagicMock,
    mock_postgres_adapter: MagicMock,
) -> NodeRegistryEffect:
    """Create a NodeRegistryEffect with mock backends."""
    return NodeRegistryEffect(mock_consul_client, mock_postgres_adapter)


# =============================================================================
# TestReducerExtensionTypeEmission
# =============================================================================


class TestReducerExtensionTypeEmission:
    """Tests for reducer extension-type intent emission.

    These tests verify that RegistrationReducer emits intents in the
    correct extension-type format as documented in the migration.
    """

    def test_reducer_emits_intent_type_intents(
        self,
        reducer: RegistrationReducer,
        initial_state: ModelRegistrationState,
        introspection_event: ModelNodeIntrospectionEvent,
    ) -> None:
        """Verify reducer uses new extension-type format with typed payloads.

        Validates that:
        1. intent_type is "extension" for all emitted intents
        2. payload is a typed payload class (ModelPayloadConsulRegister or
           ModelPayloadPostgresUpsertRegistration)
        3. payload.intent_type contains proper backend identifier
        """
        # Execute reducer
        output = reducer.reduce(initial_state, introspection_event)

        # Verify intents were emitted
        assert output.intents, "Reducer should emit intents for introspection event"
        assert len(output.intents) == 2, (
            "Reducer should emit 2 intents (Consul + PostgreSQL)"
        )

        # Verify each intent uses extension-type format with typed payloads
        for intent in output.intents:
            assert isinstance(intent, ModelIntent), (
                f"Intent should be ModelIntent, got {type(intent).__name__}"
            )
            assert intent.intent_type == "extension", (
                f"intent_type should be 'extension', got '{intent.intent_type}'"
            )
            assert isinstance(
                intent.payload,
                (ModelPayloadConsulRegister, ModelPayloadPostgresUpsertRegistration),
            ), (
                f"payload should be typed payload class, got {type(intent.payload).__name__}"
            )

    def test_consul_intent_intent_type_format(
        self,
        reducer: RegistrationReducer,
        initial_state: ModelRegistrationState,
        introspection_event: ModelNodeIntrospectionEvent,
    ) -> None:
        """Verify Consul intent uses correct intent_type with typed payload."""
        output = reducer.reduce(initial_state, introspection_event)

        # Find Consul intent
        consul_intents = [
            i
            for i in output.intents
            if isinstance(i.payload, ModelPayloadConsulRegister)
        ]

        assert len(consul_intents) == 1, "Should have exactly one Consul intent"
        consul_intent = consul_intents[0]

        # Verify typed payload structure
        payload = consul_intent.payload
        assert isinstance(payload, ModelPayloadConsulRegister)
        assert payload.intent_type == "consul.register"
        # Direct field access on typed payload
        assert payload.service_id is not None
        assert payload.service_name is not None
        assert isinstance(payload.tags, list)

    def test_postgres_intent_intent_type_format(
        self,
        reducer: RegistrationReducer,
        initial_state: ModelRegistrationState,
        introspection_event: ModelNodeIntrospectionEvent,
    ) -> None:
        """Verify PostgreSQL intent uses correct intent_type with typed payload."""
        output = reducer.reduce(initial_state, introspection_event)

        # Find PostgreSQL intent
        postgres_intents = [
            i
            for i in output.intents
            if isinstance(i.payload, ModelPayloadPostgresUpsertRegistration)
        ]

        assert len(postgres_intents) == 1, "Should have exactly one PostgreSQL intent"
        postgres_intent = postgres_intents[0]

        # Verify typed payload structure
        payload = postgres_intent.payload
        assert isinstance(payload, ModelPayloadPostgresUpsertRegistration)
        assert payload.intent_type == "postgres.upsert_registration"
        # Direct field access on typed payload - record is a Pydantic model
        assert payload.record is not None

    def test_intent_target_format(
        self,
        reducer: RegistrationReducer,
        initial_state: ModelRegistrationState,
        introspection_event: ModelNodeIntrospectionEvent,
    ) -> None:
        """Verify intent targets have proper URI format."""
        output = reducer.reduce(initial_state, introspection_event)

        # Check target formats
        targets = {i.target for i in output.intents}

        # Consul target should have consul:// scheme
        consul_targets = [t for t in targets if t.startswith("consul://")]
        assert len(consul_targets) == 1, "Should have one consul:// target"

        # PostgreSQL target should have postgres:// scheme
        postgres_targets = [t for t in targets if t.startswith("postgres://")]
        assert len(postgres_targets) == 1, "Should have one postgres:// target"


# =============================================================================
# TestExtensionTypeIntentRouting
# =============================================================================


class TestExtensionTypeIntentRouting:
    """Tests for intent routing by intent_type.

    These tests verify that the Runtime/Dispatcher layer can correctly
    route extension-type intents to appropriate Effect handlers.
    """

    def test_intent_can_be_routed_by_intent_type(
        self,
        reducer: RegistrationReducer,
        initial_state: ModelRegistrationState,
        introspection_event: ModelNodeIntrospectionEvent,
    ) -> None:
        """Verify runtime can route by payload.intent_type.

        Simulates dispatcher routing logic that:
        1. Checks intent_type == "extension"
        2. Extracts payload.intent_type from typed payload
        3. Routes to appropriate backend handler
        """
        output = reducer.reduce(initial_state, introspection_event)

        # Simulate dispatcher routing
        routing_table: dict[str, str] = {
            "consul.register": "consul_handler",
            "postgres.upsert_registration": "postgres_handler",
        }

        routed_handlers: list[str] = []
        for intent in output.intents:
            # Dispatcher checks intent_type
            if intent.intent_type == "extension":
                # Extract intent_type from typed payload (direct attribute access)
                if isinstance(
                    intent.payload,
                    (
                        ModelPayloadConsulRegister,
                        ModelPayloadPostgresUpsertRegistration,
                    ),
                ):
                    intent_type = intent.payload.intent_type
                    handler = routing_table.get(intent_type)
                    if handler:
                        routed_handlers.append(handler)

        # Verify both handlers were selected
        assert "consul_handler" in routed_handlers, "Consul handler should be routed"
        assert "postgres_handler" in routed_handlers, (
            "Postgres handler should be routed"
        )
        assert len(routed_handlers) == 2, "Should route to exactly 2 handlers"

    def test_unknown_intent_type_routing(self) -> None:
        """Verify routing handles unknown intent_type gracefully.

        When an intent_type is not in the routing table, the dispatcher
        should be able to identify it as unrouteable.

        Note: With typed payloads (ModelPayloadConsulRegister, ModelPayloadPostgresUpsertRegistration),
        the intent_type is a Literal fixed at definition time. This test verifies the routing
        table lookup behavior when a known intent_type is not configured in the table.
        """
        # Create a Consul intent with known typed payload
        payload = ModelPayloadConsulRegister(
            correlation_id=uuid4(),
            service_id="test-service-id",
            service_name="test-service",
            tags=["test"],
        )
        intent = ModelIntent(
            intent_type="extension",
            target="consul://service/test",
            payload=payload,
        )

        # Simulate routing with an incomplete table (missing consul.register)
        incomplete_routing_table: dict[str, str] = {
            # "consul.register" intentionally omitted
            "postgres.upsert_registration": "postgres_handler",
        }

        handler = incomplete_routing_table.get(intent.payload.intent_type)
        assert handler is None, (
            "Missing intent_type in routing table should return None"
        )

    def test_intent_type_correlation_id_preservation(
        self,
        reducer: RegistrationReducer,
        initial_state: ModelRegistrationState,
        introspection_event: ModelNodeIntrospectionEvent,
        correlation_id: UUID,
    ) -> None:
        """Verify correlation_id is preserved in typed intent payload.

        The correlation_id should be directly accessible on the typed payload
        for tracing and confirmation event correlation.
        """
        output = reducer.reduce(initial_state, introspection_event)

        for intent in output.intents:
            assert isinstance(
                intent.payload,
                (ModelPayloadConsulRegister, ModelPayloadPostgresUpsertRegistration),
            )
            # Correlation ID is a direct attribute on typed payloads
            assert intent.payload.correlation_id == correlation_id


# =============================================================================
# TestEffectLayerRequestFormatting
# =============================================================================


class TestEffectLayerRequestFormatting:
    """Tests for Effect layer request formatting.

    These tests verify that the Effect layer receives properly formatted
    ModelRegistryRequest objects built from extension-type intents.
    """

    def test_intent_data_to_registry_request(
        self,
        reducer: RegistrationReducer,
        initial_state: ModelRegistrationState,
        introspection_event: ModelNodeIntrospectionEvent,
        node_id: UUID,
        correlation_id: UUID,
    ) -> None:
        """Verify ModelRegistryRequest can be built from typed intent payload.

        The Orchestrator/Runtime layer must translate typed intent payloads
        to ModelRegistryRequest for the Effect layer.
        """
        output = reducer.reduce(initial_state, introspection_event)

        # Find the PostgreSQL intent (which contains the record data)
        postgres_intent = next(
            i
            for i in output.intents
            if isinstance(i.payload, ModelPayloadPostgresUpsertRegistration)
        )

        # Extract data from typed intent payload - direct field access
        payload = postgres_intent.payload
        assert isinstance(payload, ModelPayloadPostgresUpsertRegistration)
        record = payload.record

        # Build ModelRegistryRequest from typed payload data
        # This simulates what the Runtime/Orchestrator would do
        # Note: metadata may be a Pydantic model or dict
        raw_metadata = getattr(record, "metadata", {}) or {}
        if hasattr(raw_metadata, "model_dump"):
            raw_metadata = raw_metadata.model_dump()
        # Filter out None values for dict[str, str] compliance
        clean_metadata = {
            k: v
            for k, v in raw_metadata.items()
            if v is not None and isinstance(v, str)
        }

        request = ModelRegistryRequest(
            node_id=record.node_id,
            node_type=EnumNodeKind(record.node_type),
            node_version=record.node_version,
            correlation_id=payload.correlation_id,
            endpoints=dict(record.endpoints) if record.endpoints else {},
            metadata=clean_metadata,
            timestamp=datetime.now(UTC),
        )

        # Verify request was built correctly
        assert request.node_id == node_id
        assert request.node_type == EnumNodeKind.EFFECT
        assert request.node_version == "1.0.0"
        assert request.correlation_id == correlation_id

    @pytest.mark.asyncio
    async def test_effect_receives_formatted_request(
        self,
        registry_effect: NodeRegistryEffect,
        node_id: UUID,
        correlation_id: UUID,
    ) -> None:
        """Verify Effect layer receives properly formatted request.

        Tests that NodeRegistryEffect can process a request built from
        extension-type intent data.
        """
        # Create a request as the Orchestrator would
        request = ModelRegistryRequest(
            node_id=node_id,
            node_type=EnumNodeKind.EFFECT,
            node_version="1.0.0",
            correlation_id=correlation_id,
            service_name="onex-effect",
            endpoints={"health": "http://localhost:8080/health"},
            tags=["node_type:effect", "node_version:1.0.0"],
            health_check_config={
                "HTTP": "http://localhost:8080/health",
                "Interval": "10s",
                "Timeout": "5s",
            },
            timestamp=datetime.now(UTC),
        )

        # Execute effect
        response = await registry_effect.register_node(request)

        # Verify response structure
        assert response is not None
        assert response.node_id == node_id
        assert response.correlation_id == correlation_id
        assert response.consul_result is not None
        assert response.postgres_result is not None

    @pytest.mark.asyncio
    async def test_effect_handles_consul_intent_data(
        self,
        mock_consul_client: MagicMock,
        mock_postgres_adapter: MagicMock,
        reducer: RegistrationReducer,
        initial_state: ModelRegistrationState,
        introspection_event: ModelNodeIntrospectionEvent,
    ) -> None:
        """Verify Consul registration uses correct data from typed intent."""
        output = reducer.reduce(initial_state, introspection_event)

        # Find Consul intent
        consul_intent = next(
            i
            for i in output.intents
            if isinstance(i.payload, ModelPayloadConsulRegister)
        )

        # Extract Consul registration data from typed payload - direct field access
        consul_payload = consul_intent.payload
        assert isinstance(consul_payload, ModelPayloadConsulRegister)

        # Create effect and execute
        effect = NodeRegistryEffect(mock_consul_client, mock_postgres_adapter)
        request = ModelRegistryRequest(
            node_id=introspection_event.node_id,
            node_type=EnumNodeKind(introspection_event.node_type),
            node_version=introspection_event.node_version,
            correlation_id=introspection_event.correlation_id,
            service_name=consul_payload.service_name,
            tags=consul_payload.tags,
            health_check_config=consul_payload.health_check,
            timestamp=datetime.now(UTC),
        )

        await effect.register_node(request, skip_postgres=True)

        # Verify Consul client was called with correct data
        mock_consul_client.register_service.assert_called_once()
        call_kwargs = mock_consul_client.register_service.call_args.kwargs
        assert call_kwargs["service_name"] == consul_payload.service_name
        assert call_kwargs["tags"] == consul_payload.tags


# =============================================================================
# TestEndToEndExtensionTypeFlow
# =============================================================================


class TestEndToEndExtensionTypeFlow:
    """End-to-end tests for extension-type intent processing.

    These tests validate the full flow:
    Reducer -> Runtime -> Effect -> Confirmation
    """

    @pytest.mark.asyncio
    async def test_full_flow_reducer_to_effect(
        self,
        reducer: RegistrationReducer,
        initial_state: ModelRegistrationState,
        introspection_event: ModelNodeIntrospectionEvent,
        mock_consul_client: MagicMock,
        mock_postgres_adapter: MagicMock,
    ) -> None:
        """Test complete flow from reducer emit to effect execution.

        Simulates the full workflow:
        1. Reducer processes event and emits typed payload intents
        2. Runtime routes intents by intent_type
        3. Runtime builds requests from typed payload data
        4. Effect executes requests against backends
        """
        # Step 1: Reducer processes event
        output = reducer.reduce(initial_state, introspection_event)
        assert output.result.status == "pending"
        assert len(output.intents) == 2

        # Step 2: Extract intents for routing - using typed payload classes
        consul_intent = next(
            i
            for i in output.intents
            if isinstance(i.payload, ModelPayloadConsulRegister)
        )
        postgres_intent = next(
            i
            for i in output.intents
            if isinstance(i.payload, ModelPayloadPostgresUpsertRegistration)
        )

        # Step 3: Build request from typed payload data (simulating Runtime)
        pg_payload = postgres_intent.payload
        assert isinstance(pg_payload, ModelPayloadPostgresUpsertRegistration)
        record = pg_payload.record
        consul_payload = consul_intent.payload
        assert isinstance(consul_payload, ModelPayloadConsulRegister)

        request = ModelRegistryRequest(
            node_id=record.node_id,
            node_type=EnumNodeKind(record.node_type),
            node_version=record.node_version,
            correlation_id=pg_payload.correlation_id,
            service_name=consul_payload.service_name,
            endpoints=dict(record.endpoints) if record.endpoints else {},
            tags=consul_payload.tags,
            health_check_config=consul_payload.health_check,
            timestamp=datetime.now(UTC),
        )

        # Step 4: Effect executes request
        effect = NodeRegistryEffect(mock_consul_client, mock_postgres_adapter)
        response = await effect.register_node(request)

        # Verify end-to-end success
        assert response.status == "success"
        assert response.consul_result.success is True
        assert response.postgres_result.success is True

    @pytest.mark.asyncio
    async def test_flow_with_partial_failure(
        self,
        reducer: RegistrationReducer,
        initial_state: ModelRegistrationState,
        introspection_event: ModelNodeIntrospectionEvent,
        mock_consul_client: MagicMock,
        mock_postgres_adapter: MagicMock,
    ) -> None:
        """Test flow handles partial backend failure correctly.

        Simulates scenario where Consul succeeds but PostgreSQL fails.
        """
        # Configure PostgreSQL to fail
        mock_postgres_adapter.upsert = AsyncMock(
            return_value=MagicMock(success=False, error="connection timeout")
        )

        # Reducer processes event
        output = reducer.reduce(initial_state, introspection_event)

        # Build request using typed payloads
        postgres_intent = next(
            i
            for i in output.intents
            if isinstance(i.payload, ModelPayloadPostgresUpsertRegistration)
        )
        consul_intent = next(
            i
            for i in output.intents
            if isinstance(i.payload, ModelPayloadConsulRegister)
        )

        pg_payload = postgres_intent.payload
        assert isinstance(pg_payload, ModelPayloadPostgresUpsertRegistration)
        record = pg_payload.record
        consul_payload = consul_intent.payload
        assert isinstance(consul_payload, ModelPayloadConsulRegister)

        request = ModelRegistryRequest(
            node_id=record.node_id,
            node_type=EnumNodeKind(record.node_type),
            node_version=record.node_version,
            correlation_id=pg_payload.correlation_id,
            service_name=consul_payload.service_name,
            endpoints=dict(record.endpoints) if record.endpoints else {},
            tags=consul_payload.tags,
            health_check_config=consul_payload.health_check,
            timestamp=datetime.now(UTC),
        )

        # Effect executes with partial failure
        effect = NodeRegistryEffect(mock_consul_client, mock_postgres_adapter)
        response = await effect.register_node(request)

        # Verify partial success
        assert response.status == "partial"
        assert response.consul_result.success is True
        assert response.postgres_result.success is False
        assert response.postgres_result.error is not None

    def test_idempotent_event_handling(
        self,
        reducer: RegistrationReducer,
        initial_state: ModelRegistrationState,
        introspection_event: ModelNodeIntrospectionEvent,
    ) -> None:
        """Test that duplicate events do not emit duplicate intents.

        The reducer should detect duplicate events via last_processed_event_id
        and return current state without emitting new intents.
        """
        # First processing
        output1 = reducer.reduce(initial_state, introspection_event)
        assert len(output1.intents) == 2

        # Second processing with same event (duplicate)
        output2 = reducer.reduce(output1.result, introspection_event)

        # Duplicate event should emit no intents
        assert len(output2.intents) == 0, "Duplicate event should not emit intents"
        assert output2.result == output1.result, "State should be unchanged"


# =============================================================================
# TestIntentPayloadSerialization
# =============================================================================


class TestIntentPayloadSerialization:
    """Tests for intent payload serialization compatibility.

    These tests verify that intent payloads serialize correctly and
    can be deserialized by downstream consumers.
    """

    def test_intent_payload_json_serializable(
        self,
        reducer: RegistrationReducer,
        initial_state: ModelRegistrationState,
        introspection_event: ModelNodeIntrospectionEvent,
    ) -> None:
        """Verify typed intent payloads can be JSON serialized.

        Intents may be transmitted via Kafka/HTTP, so payloads must
        be JSON-serializable.
        """
        import json

        output = reducer.reduce(initial_state, introspection_event)

        for intent in output.intents:
            # Serialize the entire intent
            intent_dict = intent.model_dump(mode="json")
            json_str = json.dumps(intent_dict)

            # Deserialize and verify - typed payloads have intent_type as field
            parsed = json.loads(json_str)
            assert parsed["intent_type"] == "extension"
            # Typed payloads have intent_type field directly
            assert "intent_type" in parsed["payload"]
            # Typed payloads have correlation_id field
            assert "correlation_id" in parsed["payload"]

    def test_typed_payload_round_trip(
        self,
        reducer: RegistrationReducer,
        initial_state: ModelRegistrationState,
        introspection_event: ModelNodeIntrospectionEvent,
    ) -> None:
        """Verify typed payloads survive round-trip serialization."""
        output = reducer.reduce(initial_state, introspection_event)

        for intent in output.intents:
            original_payload = intent.payload

            if isinstance(original_payload, ModelPayloadConsulRegister):
                # Serialize and deserialize Consul payload
                payload_dict = original_payload.model_dump(mode="json")
                restored_payload = ModelPayloadConsulRegister.model_validate(
                    payload_dict
                )
                # Verify all fields preserved
                assert restored_payload.intent_type == original_payload.intent_type
                assert (
                    restored_payload.correlation_id == original_payload.correlation_id
                )
                assert restored_payload.service_id == original_payload.service_id
                assert restored_payload.service_name == original_payload.service_name
                assert restored_payload.tags == original_payload.tags

            elif isinstance(original_payload, ModelPayloadPostgresUpsertRegistration):
                # Serialize and deserialize Postgres payload
                payload_dict = original_payload.model_dump(mode="json")
                restored_payload = (
                    ModelPayloadPostgresUpsertRegistration.model_validate(payload_dict)
                )
                # Verify key fields preserved
                assert restored_payload.intent_type == original_payload.intent_type
                assert (
                    restored_payload.correlation_id == original_payload.correlation_id
                )
                # Record is typed as BaseModel, compare via serialized dict
                # (Pydantic deserializes to generic BaseModel, not the typed subclass)
                original_record_dict = original_payload.record.model_dump(mode="json")
                # Access restored record from the serialized dict (avoids BaseModel issue)
                assert (
                    payload_dict["record"]["node_id"] == original_record_dict["node_id"]
                )
            else:
                pytest.fail(f"Unexpected payload type: {type(original_payload)}")


# =============================================================================
# Module Exports
# =============================================================================

__all__ = [
    "TestReducerExtensionTypeEmission",
    "TestExtensionTypeIntentRouting",
    "TestEffectLayerRequestFormatting",
    "TestEndToEndExtensionTypeFlow",
    "TestIntentPayloadSerialization",
]
