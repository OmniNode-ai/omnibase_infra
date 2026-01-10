# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Unit tests for ServiceCapabilityQuery.

This test suite validates the capability query service that provides service
discovery functionality for finding nodes by capability, intent type, or protocol.

Test Organization:
    - TestFindNodesByCapability: Capability tag queries
    - TestFindNodesByIntentType: Intent type queries
    - TestFindNodesByProtocol: Protocol queries
    - TestFindNodesByContractType: Contract type queries
    - TestResolveDependency: Dependency resolution with strategies
    - TestServiceCapabilityQueryErrorHandling: Error scenarios

Note:
    These tests validate the ServiceCapabilityQuery API contract.
    Implementation: omnibase_infra/services/service_capability_query.py

Coverage Goals:
    - >90% code coverage for service
    - All query paths tested
    - All selection strategies tested
    - Error handling validated

Related Tickets:
    - OMN-1135: ServiceCapabilityQuery Implementation
    - OMN-1134: Registry Projection Extensions for Capabilities
"""

from __future__ import annotations

from datetime import UTC, datetime
from unittest.mock import AsyncMock
from uuid import uuid4

import pytest
from omnibase_core.enums import EnumNodeKind
from omnibase_core.models.primitives.model_semver import ModelSemVer

from omnibase_infra.enums import EnumRegistrationState
from omnibase_infra.models.discovery import ModelDependencySpec
from omnibase_infra.models.projection import ModelRegistrationProjection
from omnibase_infra.models.registration.model_node_capabilities import (
    ModelNodeCapabilities,
)
from omnibase_infra.services import EnumSelectionStrategy, ServiceCapabilityQuery

# =============================================================================
# Test Constants
# =============================================================================

DEFAULT_DOMAIN = "registration"
"""Default domain for registration queries."""


# =============================================================================
# Test Helpers
# =============================================================================


def create_mock_projection(
    state: EnumRegistrationState = EnumRegistrationState.ACTIVE,
    node_type: EnumNodeKind = EnumNodeKind.EFFECT,
    capability_tags: list[str] | None = None,
    intent_types: list[str] | None = None,
    protocols: list[str] | None = None,
    contract_type: str | None = "effect",
) -> ModelRegistrationProjection:
    """Create a mock projection with sensible defaults.

    Args:
        state: Registration state (default: ACTIVE)
        node_type: Node kind (default: EFFECT)
        capability_tags: List of capability tags
        intent_types: List of intent types this node handles
        protocols: List of protocols this node implements
        contract_type: Contract type (effect, compute, reducer, orchestrator)

    Returns:
        ModelRegistrationProjection with test data
    """
    now = datetime.now(UTC)
    return ModelRegistrationProjection(
        entity_id=uuid4(),
        domain=DEFAULT_DOMAIN,
        current_state=state,
        node_type=node_type,
        node_version=ModelSemVer.parse("1.0.0"),
        capabilities=ModelNodeCapabilities(),
        capability_tags=capability_tags or [],
        intent_types=intent_types or [],
        protocols=protocols or [],
        contract_type=contract_type,
        last_applied_event_id=uuid4(),
        last_applied_offset=100,
        registered_at=now,
        updated_at=now,
    )


def create_consul_adapter_projection() -> ModelRegistrationProjection:
    """Create a projection representing a Consul adapter node."""
    return create_mock_projection(
        node_type=EnumNodeKind.EFFECT,
        capability_tags=[
            "consul.registration",
            "consul.discovery",
            "service.discovery",
        ],
        intent_types=["consul.register", "consul.deregister", "consul.healthcheck"],
        protocols=["ProtocolServiceDiscovery", "ProtocolHealthCheck"],
        contract_type="effect",
    )


def create_postgres_adapter_projection() -> ModelRegistrationProjection:
    """Create a projection representing a PostgreSQL adapter node."""
    return create_mock_projection(
        node_type=EnumNodeKind.EFFECT,
        capability_tags=["postgres.storage", "database.sql", "transactions"],
        intent_types=["postgres.query", "postgres.upsert", "postgres.delete"],
        protocols=["ProtocolDatabaseAdapter", "ProtocolTransactional"],
        contract_type="effect",
    )


def create_kafka_adapter_projection() -> ModelRegistrationProjection:
    """Create a projection representing a Kafka adapter node."""
    return create_mock_projection(
        node_type=EnumNodeKind.EFFECT,
        capability_tags=["kafka.consumer", "kafka.producer", "event.streaming"],
        intent_types=["kafka.publish", "kafka.consume", "kafka.commit"],
        protocols=["ProtocolEventPublisher", "ProtocolEventConsumer"],
        contract_type="effect",
    )


def create_registration_reducer_projection() -> ModelRegistrationProjection:
    """Create a projection representing a registration reducer node."""
    return create_mock_projection(
        node_type=EnumNodeKind.REDUCER,
        capability_tags=["registration.fsm", "state.management"],
        intent_types=["registration.reduce"],
        protocols=["ProtocolReducer", "ProtocolFSM"],
        contract_type="reducer",
    )


# =============================================================================
# Test Fixtures
# =============================================================================


@pytest.fixture
def mock_projection_reader() -> AsyncMock:
    """Create a mock projection reader with all capability query methods."""
    reader = AsyncMock()
    reader.get_by_capability_tag = AsyncMock(return_value=[])
    reader.get_by_capability_tags_all = AsyncMock(return_value=[])
    reader.get_by_capability_tags_any = AsyncMock(return_value=[])
    reader.get_by_intent_type = AsyncMock(return_value=[])
    reader.get_by_intent_types = AsyncMock(return_value=[])
    reader.get_by_protocol = AsyncMock(return_value=[])
    reader.get_by_contract_type = AsyncMock(return_value=[])
    return reader


@pytest.fixture
def service(mock_projection_reader: AsyncMock) -> ServiceCapabilityQuery:
    """Create service with mocked dependencies."""
    return ServiceCapabilityQuery(projection_reader=mock_projection_reader)


# =============================================================================
# Test Classes
# =============================================================================


@pytest.mark.unit
@pytest.mark.asyncio
class TestFindNodesByCapability:
    """Tests for find_nodes_by_capability method."""

    async def test_find_by_capability_returns_matching_nodes(
        self,
        mock_projection_reader: AsyncMock,
        service: ServiceCapabilityQuery,
    ) -> None:
        """Should return nodes with matching capability tag.

        Given: A projection reader with nodes having 'consul.registration' capability
        When: find_nodes_by_capability is called with 'consul.registration'
        Then: Returns the matching nodes
        """
        consul_projection = create_consul_adapter_projection()
        mock_projection_reader.get_by_capability_tag.return_value = [consul_projection]

        result = await service.find_nodes_by_capability("consul.registration")

        assert len(result) == 1
        assert result[0].entity_id == consul_projection.entity_id
        mock_projection_reader.get_by_capability_tag.assert_called_once_with(
            tag="consul.registration",
            state=EnumRegistrationState.ACTIVE,
        )

    async def test_find_by_capability_with_custom_state(
        self,
        mock_projection_reader: AsyncMock,
        service: ServiceCapabilityQuery,
    ) -> None:
        """Should filter by custom state when specified.

        Given: A projection reader
        When: find_nodes_by_capability is called with a non-ACTIVE state
        Then: The reader is called with that specific state
        """
        pending_projection = create_mock_projection(
            state=EnumRegistrationState.PENDING_REGISTRATION,
            capability_tags=["consul.registration"],
        )
        mock_projection_reader.get_by_capability_tag.return_value = [pending_projection]

        result = await service.find_nodes_by_capability(
            "consul.registration",
            state=EnumRegistrationState.PENDING_REGISTRATION,
        )

        assert len(result) == 1
        mock_projection_reader.get_by_capability_tag.assert_called_once_with(
            tag="consul.registration",
            state=EnumRegistrationState.PENDING_REGISTRATION,
        )

    async def test_find_by_capability_empty_results(
        self,
        mock_projection_reader: AsyncMock,
        service: ServiceCapabilityQuery,
    ) -> None:
        """Should return empty list when no matches found.

        Given: A projection reader with no matching nodes
        When: find_nodes_by_capability is called with non-existent capability
        Then: Returns an empty list
        """
        mock_projection_reader.get_by_capability_tag.return_value = []

        result = await service.find_nodes_by_capability("nonexistent.capability")

        assert result == []

    async def test_find_by_capability_multiple_matches(
        self,
        mock_projection_reader: AsyncMock,
        service: ServiceCapabilityQuery,
    ) -> None:
        """Should return all matching nodes when multiple exist.

        Given: Multiple nodes with the same capability tag
        When: find_nodes_by_capability is called
        Then: Returns all matching nodes
        """
        projections = [
            create_mock_projection(capability_tags=["database.sql"]),
            create_mock_projection(capability_tags=["database.sql"]),
            create_mock_projection(capability_tags=["database.sql"]),
        ]
        mock_projection_reader.get_by_capability_tag.return_value = projections

        result = await service.find_nodes_by_capability("database.sql")

        assert len(result) == 3

    async def test_find_by_capability_defaults_to_active_state(
        self,
        mock_projection_reader: AsyncMock,
        service: ServiceCapabilityQuery,
    ) -> None:
        """Should default to ACTIVE state when no state specified.

        Given: A projection reader
        When: find_nodes_by_capability is called without state parameter
        Then: The reader is called with state=ACTIVE as default
        """
        mock_projection_reader.get_by_capability_tag.return_value = []

        await service.find_nodes_by_capability("any.capability")

        mock_projection_reader.get_by_capability_tag.assert_called_once()
        call_kwargs = mock_projection_reader.get_by_capability_tag.call_args.kwargs
        assert call_kwargs.get("state") == EnumRegistrationState.ACTIVE


@pytest.mark.unit
@pytest.mark.asyncio
class TestFindNodesByIntentType:
    """Tests for find_nodes_by_intent_type method."""

    async def test_find_by_intent_type_returns_effect_nodes(
        self,
        mock_projection_reader: AsyncMock,
        service: ServiceCapabilityQuery,
    ) -> None:
        """Should return effect nodes that handle the specified intent type.

        Given: A PostgreSQL adapter that handles 'postgres.query' intent
        When: find_nodes_by_intent_type is called with 'postgres.query'
        Then: Returns the matching adapter node
        """
        postgres_projection = create_postgres_adapter_projection()
        mock_projection_reader.get_by_intent_type.return_value = [postgres_projection]

        result = await service.find_nodes_by_intent_type("postgres.query")

        assert len(result) == 1
        assert "postgres.query" in result[0].intent_types
        mock_projection_reader.get_by_intent_type.assert_called_once_with(
            intent_type="postgres.query",
            state=EnumRegistrationState.ACTIVE,
        )

    async def test_find_by_intent_type_with_custom_contract_type(
        self,
        mock_projection_reader: AsyncMock,
        service: ServiceCapabilityQuery,
    ) -> None:
        """Should filter by contract type when specified.

        Given: A projection reader
        When: find_nodes_by_intent_type is called with contract_type filter
        Then: Returns only nodes of that contract type
        """
        reducer_projection = create_registration_reducer_projection()
        mock_projection_reader.get_by_intent_type.return_value = [reducer_projection]

        result = await service.find_nodes_by_intent_type(
            "registration.reduce",
            contract_type="reducer",
        )

        assert len(result) == 1
        assert result[0].contract_type == "reducer"

    async def test_find_by_intent_type_empty_results(
        self,
        mock_projection_reader: AsyncMock,
        service: ServiceCapabilityQuery,
    ) -> None:
        """Should return empty list when no nodes handle the intent type."""
        mock_projection_reader.get_by_intent_type.return_value = []

        result = await service.find_nodes_by_intent_type("nonexistent.intent")

        assert result == []

    async def test_find_by_intent_type_multiple_handlers(
        self,
        mock_projection_reader: AsyncMock,
        service: ServiceCapabilityQuery,
    ) -> None:
        """Should return multiple nodes if they all handle the same intent.

        Given: Multiple nodes that handle 'database.query' intent
        When: find_nodes_by_intent_type is called
        Then: Returns all matching nodes
        """
        projections = [
            create_mock_projection(intent_types=["database.query"]),
            create_mock_projection(intent_types=["database.query", "database.upsert"]),
        ]
        mock_projection_reader.get_by_intent_type.return_value = projections

        result = await service.find_nodes_by_intent_type("database.query")

        assert len(result) == 2


@pytest.mark.unit
@pytest.mark.asyncio
class TestFindNodesByIntentTypes:
    """Tests for find_nodes_by_intent_types bulk query method."""

    async def test_find_by_intent_types_returns_matching_nodes(
        self,
        mock_projection_reader: AsyncMock,
        service: ServiceCapabilityQuery,
    ) -> None:
        """Should return nodes matching ANY of the specified intent types.

        Given: A projection reader with nodes handling various postgres intents
        When: find_nodes_by_intent_types is called with multiple intent types
        Then: Returns all matching nodes in a single query
        """
        projections = [
            create_mock_projection(intent_types=["postgres.query", "postgres.upsert"]),
            create_mock_projection(intent_types=["postgres.delete"]),
        ]
        mock_projection_reader.get_by_intent_types.return_value = projections

        result = await service.find_nodes_by_intent_types(
            ["postgres.query", "postgres.delete"]
        )

        assert len(result) == 2
        mock_projection_reader.get_by_intent_types.assert_called_once_with(
            intent_types=["postgres.query", "postgres.delete"],
            state=EnumRegistrationState.ACTIVE,
        )

    async def test_find_by_intent_types_empty_list_returns_empty(
        self,
        mock_projection_reader: AsyncMock,
        service: ServiceCapabilityQuery,
    ) -> None:
        """Should return empty list when intent_types list is empty.

        Given: An empty intent_types list
        When: find_nodes_by_intent_types is called
        Then: Returns empty list without calling the reader
        """
        result = await service.find_nodes_by_intent_types([])

        assert result == []
        mock_projection_reader.get_by_intent_types.assert_not_called()

    async def test_find_by_intent_types_with_contract_type_filter(
        self,
        mock_projection_reader: AsyncMock,
        service: ServiceCapabilityQuery,
    ) -> None:
        """Should filter results by contract_type."""
        effect_projection = create_mock_projection(
            intent_types=["postgres.query"],
            contract_type="effect",
        )
        reducer_projection = create_mock_projection(
            intent_types=["postgres.query"],
            contract_type="reducer",
        )
        mock_projection_reader.get_by_intent_types.return_value = [
            effect_projection,
            reducer_projection,
        ]

        result = await service.find_nodes_by_intent_types(
            ["postgres.query"],
            contract_type="effect",
        )

        # Only effect node should be returned due to contract_type filter
        assert len(result) == 1
        assert result[0].contract_type == "effect"

    async def test_find_by_intent_types_with_custom_state(
        self,
        mock_projection_reader: AsyncMock,
        service: ServiceCapabilityQuery,
    ) -> None:
        """Should filter by custom state when specified."""
        pending_projection = create_mock_projection(
            state=EnumRegistrationState.PENDING_REGISTRATION,
            intent_types=["postgres.query"],
        )
        mock_projection_reader.get_by_intent_types.return_value = [pending_projection]

        result = await service.find_nodes_by_intent_types(
            ["postgres.query"],
            state=EnumRegistrationState.PENDING_REGISTRATION,
        )

        assert len(result) == 1
        mock_projection_reader.get_by_intent_types.assert_called_once_with(
            intent_types=["postgres.query"],
            state=EnumRegistrationState.PENDING_REGISTRATION,
        )

    async def test_find_by_intent_types_defaults_to_active_state(
        self,
        mock_projection_reader: AsyncMock,
        service: ServiceCapabilityQuery,
    ) -> None:
        """Should default to ACTIVE state when no state specified."""
        mock_projection_reader.get_by_intent_types.return_value = []

        await service.find_nodes_by_intent_types(["any.intent"])

        mock_projection_reader.get_by_intent_types.assert_called_once()
        call_kwargs = mock_projection_reader.get_by_intent_types.call_args.kwargs
        assert call_kwargs.get("state") == EnumRegistrationState.ACTIVE


@pytest.mark.unit
@pytest.mark.asyncio
class TestResolveDependencyBulkQuery:
    """Tests for resolve_dependency using bulk intent type queries."""

    async def test_resolve_by_multiple_intent_types_uses_bulk_query(
        self,
        mock_projection_reader: AsyncMock,
        service: ServiceCapabilityQuery,
    ) -> None:
        """Should use bulk query for multiple intent types instead of N queries.

        Given: A dependency spec with multiple intent types
        When: resolve_dependency is called
        Then: Uses single bulk query (get_by_intent_types) instead of N queries
        """
        projection = create_postgres_adapter_projection()
        mock_projection_reader.get_by_intent_types.return_value = [projection]

        spec = ModelDependencySpec(
            name="storage",
            type="node",
            intent_types=["postgres.query", "postgres.upsert", "postgres.delete"],
        )
        result = await service.resolve_dependency(spec)

        assert result is not None
        # Verify bulk query was called once (not 3 times for 3 intent types)
        mock_projection_reader.get_by_intent_types.assert_called_once()
        # Verify all intent types were passed
        call_kwargs = mock_projection_reader.get_by_intent_types.call_args.kwargs
        assert call_kwargs["intent_types"] == [
            "postgres.query",
            "postgres.upsert",
            "postgres.delete",
        ]


@pytest.mark.unit
@pytest.mark.asyncio
class TestFindNodesByProtocol:
    """Tests for find_nodes_by_protocol method."""

    async def test_find_by_protocol_returns_implementing_nodes(
        self,
        mock_projection_reader: AsyncMock,
        service: ServiceCapabilityQuery,
    ) -> None:
        """Should return nodes implementing the specified protocol.

        Given: A Kafka adapter implementing ProtocolEventPublisher
        When: find_nodes_by_protocol is called with 'ProtocolEventPublisher'
        Then: Returns the matching adapter node
        """
        kafka_projection = create_kafka_adapter_projection()
        mock_projection_reader.get_by_protocol.return_value = [kafka_projection]

        result = await service.find_nodes_by_protocol("ProtocolEventPublisher")

        assert len(result) == 1
        assert "ProtocolEventPublisher" in result[0].protocols
        mock_projection_reader.get_by_protocol.assert_called_once_with(
            protocol_name="ProtocolEventPublisher",
            state=EnumRegistrationState.ACTIVE,
        )

    async def test_find_by_protocol_empty_results(
        self,
        mock_projection_reader: AsyncMock,
        service: ServiceCapabilityQuery,
    ) -> None:
        """Should return empty list when no nodes implement the protocol."""
        mock_projection_reader.get_by_protocol.return_value = []

        result = await service.find_nodes_by_protocol("ProtocolNonExistent")

        assert result == []

    async def test_find_by_protocol_with_custom_state(
        self,
        mock_projection_reader: AsyncMock,
        service: ServiceCapabilityQuery,
    ) -> None:
        """Should filter by state when specified."""
        pending_projection = create_mock_projection(
            state=EnumRegistrationState.AWAITING_ACK,
            protocols=["ProtocolHealthCheck"],
        )
        mock_projection_reader.get_by_protocol.return_value = [pending_projection]

        result = await service.find_nodes_by_protocol(
            "ProtocolHealthCheck",
            state=EnumRegistrationState.AWAITING_ACK,
        )

        assert len(result) == 1
        mock_projection_reader.get_by_protocol.assert_called_once_with(
            protocol_name="ProtocolHealthCheck",
            state=EnumRegistrationState.AWAITING_ACK,
        )


@pytest.mark.unit
@pytest.mark.asyncio
class TestFindNodesByContractType:
    """Tests for contract_type filtering.

    Note: ServiceCapabilityQuery does not have a dedicated find_nodes_by_contract_type method.
    Contract type filtering is available as a parameter on other query methods.
    These tests verify contract_type filtering via find_nodes_by_capability.
    """

    async def test_find_by_capability_filters_by_contract_type(
        self,
        mock_projection_reader: AsyncMock,
        service: ServiceCapabilityQuery,
    ) -> None:
        """Should filter results by contract_type when specified."""
        effect_projection = create_consul_adapter_projection()
        mock_projection_reader.get_by_capability_tag.return_value = [effect_projection]

        result = await service.find_nodes_by_capability(
            "consul.registration",
            contract_type="effect",
        )

        assert len(result) == 1
        assert result[0].contract_type == "effect"

    async def test_find_by_capability_excludes_non_matching_contract_type(
        self,
        mock_projection_reader: AsyncMock,
        service: ServiceCapabilityQuery,
    ) -> None:
        """Should exclude nodes with non-matching contract_type."""
        effect_projection = create_consul_adapter_projection()  # contract_type="effect"
        mock_projection_reader.get_by_capability_tag.return_value = [effect_projection]

        result = await service.find_nodes_by_capability(
            "consul.registration",
            contract_type="reducer",  # Filter for reducer, but projection is effect
        )

        # Should be filtered out since contract_type doesn't match
        assert len(result) == 0


@pytest.mark.unit
@pytest.mark.asyncio
class TestResolveDependency:
    """Tests for resolve_dependency method."""

    async def test_resolve_by_capability_returns_single_node(
        self,
        mock_projection_reader: AsyncMock,
        service: ServiceCapabilityQuery,
    ) -> None:
        """Should resolve to a single node when multiple exist using strategy.

        Given: Multiple nodes with the same capability
        When: resolve_dependency is called with capability query
        Then: Returns exactly one node based on selection strategy
        """
        projections = [
            create_mock_projection(capability_tags=["database.sql"]),
            create_mock_projection(capability_tags=["database.sql"]),
        ]
        mock_projection_reader.get_by_capability_tag.return_value = projections

        spec = ModelDependencySpec(
            name="database",
            type="node",
            capability="database.sql",
        )
        result = await service.resolve_dependency(spec)

        assert result is not None
        assert result.entity_id in [p.entity_id for p in projections]

    async def test_resolve_by_intent_type_returns_single_node(
        self,
        mock_projection_reader: AsyncMock,
        service: ServiceCapabilityQuery,
    ) -> None:
        """Should resolve to a single node when querying by intent type."""
        projection = create_postgres_adapter_projection()
        # Now uses bulk query method get_by_intent_types
        mock_projection_reader.get_by_intent_types.return_value = [projection]

        spec = ModelDependencySpec(
            name="storage",
            type="node",
            intent_types=["postgres.query"],
        )
        result = await service.resolve_dependency(spec)

        assert result is not None
        assert result.entity_id == projection.entity_id

    async def test_resolve_by_protocol_returns_single_node(
        self,
        mock_projection_reader: AsyncMock,
        service: ServiceCapabilityQuery,
    ) -> None:
        """Should resolve to a single node when querying by protocol."""
        projection = create_kafka_adapter_projection()
        mock_projection_reader.get_by_protocol.return_value = [projection]

        spec = ModelDependencySpec(
            name="publisher",
            type="protocol",
            protocol="ProtocolEventPublisher",
        )
        result = await service.resolve_dependency(spec)

        assert result is not None
        assert result.entity_id == projection.entity_id

    async def test_resolve_returns_none_when_no_matches(
        self,
        mock_projection_reader: AsyncMock,
        service: ServiceCapabilityQuery,
    ) -> None:
        """Should return None when no matching nodes exist.

        Given: No nodes match the query criteria
        When: resolve_dependency is called
        Then: Returns None instead of raising an error
        """
        mock_projection_reader.get_by_capability_tag.return_value = []

        spec = ModelDependencySpec(
            name="missing",
            type="node",
            capability="nonexistent.capability",
        )
        result = await service.resolve_dependency(spec)

        assert result is None

    async def test_resolve_uses_first_strategy_by_default(
        self,
        mock_projection_reader: AsyncMock,
        service: ServiceCapabilityQuery,
    ) -> None:
        """Should use FIRST selection strategy by default.

        Given: Multiple matching nodes
        When: resolve_dependency is called without strategy
        Then: Returns the first node in the list
        """
        projections = [
            create_mock_projection(capability_tags=["test.cap"]),
            create_mock_projection(capability_tags=["test.cap"]),
        ]
        mock_projection_reader.get_by_capability_tag.return_value = projections

        spec = ModelDependencySpec(
            name="test",
            type="node",
            capability="test.cap",
            # selection_strategy defaults to "first"
        )
        result = await service.resolve_dependency(spec)

        assert result is not None
        assert result.entity_id == projections[0].entity_id

    async def test_resolve_uses_selection_strategy(
        self,
        mock_projection_reader: AsyncMock,
        service: ServiceCapabilityQuery,
    ) -> None:
        """Should use specified selection strategy when provided.

        Given: Multiple matching nodes
        When: resolve_dependency is called with a strategy
        Then: Uses that strategy to select the node
        """
        projections = [
            create_mock_projection(capability_tags=["test.cap"]),
            create_mock_projection(capability_tags=["test.cap"]),
        ]
        mock_projection_reader.get_by_capability_tag.return_value = projections

        spec = ModelDependencySpec(
            name="test",
            type="node",
            capability="test.cap",
            selection_strategy="random",
        )
        result = await service.resolve_dependency(spec)

        assert result is not None
        assert result.entity_id in [p.entity_id for p in projections]

    async def test_resolve_priority_capability_over_intent(
        self,
        mock_projection_reader: AsyncMock,
        service: ServiceCapabilityQuery,
    ) -> None:
        """Should prioritize capability query over intent type when both specified.

        Given: Both capability and intent_types are provided
        When: resolve_dependency is called
        Then: Uses capability query (higher priority)
        """
        cap_projection = create_mock_projection(capability_tags=["cap.a"])
        mock_projection_reader.get_by_capability_tag.return_value = [cap_projection]

        spec = ModelDependencySpec(
            name="test",
            type="node",
            capability="cap.a",
            intent_types=["intent.b"],  # Should be ignored when capability is set
        )
        result = await service.resolve_dependency(spec)

        assert result is not None
        mock_projection_reader.get_by_capability_tag.assert_called_once()
        mock_projection_reader.get_by_intent_type.assert_not_called()


@pytest.mark.unit
@pytest.mark.asyncio
class TestServiceCapabilityQueryDelegation:
    """Tests for proper delegation to projection reader."""

    async def test_find_by_capability_delegates_to_reader(
        self,
        mock_projection_reader: AsyncMock,
        service: ServiceCapabilityQuery,
    ) -> None:
        """Should delegate capability queries to the projection reader."""
        mock_projection_reader.get_by_capability_tag.return_value = []

        await service.find_nodes_by_capability("test.cap")

        mock_projection_reader.get_by_capability_tag.assert_called_once()
        call_kwargs = mock_projection_reader.get_by_capability_tag.call_args.kwargs
        assert call_kwargs.get("tag") == "test.cap"

    async def test_find_by_intent_type_delegates_to_reader(
        self,
        mock_projection_reader: AsyncMock,
        service: ServiceCapabilityQuery,
    ) -> None:
        """Should delegate intent type queries to the projection reader."""
        mock_projection_reader.get_by_intent_type.return_value = []

        await service.find_nodes_by_intent_type("test.intent")

        mock_projection_reader.get_by_intent_type.assert_called_once()
        call_kwargs = mock_projection_reader.get_by_intent_type.call_args.kwargs
        assert call_kwargs.get("intent_type") == "test.intent"


@pytest.mark.unit
@pytest.mark.asyncio
class TestServiceCapabilityQueryErrorHandling:
    """Tests for error handling in ServiceCapabilityQuery."""

    async def test_propagates_connection_errors(
        self,
        mock_projection_reader: AsyncMock,
        service: ServiceCapabilityQuery,
    ) -> None:
        """Should propagate InfraConnectionError from projection reader."""
        from omnibase_infra.errors import InfraConnectionError

        mock_projection_reader.get_by_capability_tag.side_effect = InfraConnectionError(
            "Connection refused"
        )

        with pytest.raises(InfraConnectionError):
            await service.find_nodes_by_capability("test.cap")

    async def test_propagates_timeout_errors(
        self,
        mock_projection_reader: AsyncMock,
        service: ServiceCapabilityQuery,
    ) -> None:
        """Should propagate InfraTimeoutError from projection reader."""
        from omnibase_infra.errors import InfraTimeoutError

        mock_projection_reader.get_by_intent_type.side_effect = InfraTimeoutError(
            "Query timed out"
        )

        with pytest.raises(InfraTimeoutError):
            await service.find_nodes_by_intent_type("test.intent")

    async def test_propagates_circuit_breaker_errors(
        self,
        mock_projection_reader: AsyncMock,
        service: ServiceCapabilityQuery,
    ) -> None:
        """Should propagate InfraUnavailableError from projection reader."""
        from omnibase_infra.errors import InfraUnavailableError

        mock_projection_reader.get_by_protocol.side_effect = InfraUnavailableError(
            "Circuit breaker is open"
        )

        with pytest.raises(InfraUnavailableError):
            await service.find_nodes_by_protocol("ProtocolTest")


__all__: list[str] = []
