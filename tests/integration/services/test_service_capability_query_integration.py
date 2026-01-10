# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Integration tests for ServiceCapabilityQuery resolution flow.

These tests verify the full dependency resolution path through
ServiceCapabilityQuery, including selection strategies and error handling.

Test Organization:
    - TestResolutionFlowWithCapability: Full resolution using capability filter
    - TestResolutionFlowWithIntentTypes: Full resolution using intent types filter
    - TestResolutionFlowWithProtocol: Full resolution using protocol filter
    - TestSelectionStrategies: Selection strategy behavior verification
    - TestEmptyResultsHandling: Edge cases with no matching nodes
    - TestErrorPropagation: Error handling from projection reader

Related Tickets:
    - OMN-1135: ServiceCapabilityQuery for capability-based discovery
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
from omnibase_infra.errors import (
    InfraConnectionError,
    InfraTimeoutError,
    InfraUnavailableError,
)
from omnibase_infra.models.discovery import ModelDependencySpec
from omnibase_infra.models.projection import ModelRegistrationProjection
from omnibase_infra.models.registration.model_node_capabilities import (
    ModelNodeCapabilities,
)
from omnibase_infra.services import ServiceCapabilityQuery
from omnibase_infra.services.service_node_selector import ServiceNodeSelector

# Test markers
pytestmark = [
    pytest.mark.asyncio,
]

# =============================================================================
# Test Constants
# =============================================================================

DEFAULT_DOMAIN = "registration"
"""Default domain for registration queries."""


# =============================================================================
# Test Helpers
# =============================================================================


def create_mock_projection(
    *,
    entity_id_suffix: str | None = None,
    state: EnumRegistrationState = EnumRegistrationState.ACTIVE,
    node_type: EnumNodeKind = EnumNodeKind.EFFECT,
    capability_tags: list[str] | None = None,
    intent_types: list[str] | None = None,
    protocols: list[str] | None = None,
    contract_type: str | None = "effect",
) -> ModelRegistrationProjection:
    """Create a mock projection with sensible defaults.

    Args:
        entity_id_suffix: Optional suffix for deterministic entity IDs in tests.
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


def create_postgres_adapter_projection() -> ModelRegistrationProjection:
    """Create a projection representing a PostgreSQL adapter node."""
    return create_mock_projection(
        node_type=EnumNodeKind.EFFECT,
        capability_tags=["postgres.storage", "database.sql", "transactions"],
        intent_types=["postgres.query", "postgres.upsert", "postgres.delete"],
        protocols=["ProtocolDatabaseAdapter", "ProtocolTransactional"],
        contract_type="effect",
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


def create_kafka_adapter_projection() -> ModelRegistrationProjection:
    """Create a projection representing a Kafka adapter node."""
    return create_mock_projection(
        node_type=EnumNodeKind.EFFECT,
        capability_tags=["kafka.consumer", "kafka.producer", "event.streaming"],
        intent_types=["kafka.publish", "kafka.consume", "kafka.commit"],
        protocols=["ProtocolEventPublisher", "ProtocolEventConsumer"],
        contract_type="effect",
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
def node_selector() -> ServiceNodeSelector:
    """Create a real node selector for integration testing."""
    return ServiceNodeSelector()


@pytest.fixture
def service(
    mock_projection_reader: AsyncMock,
    node_selector: ServiceNodeSelector,
) -> ServiceCapabilityQuery:
    """Create service with mocked reader but real node selector."""
    return ServiceCapabilityQuery(
        projection_reader=mock_projection_reader,
        node_selector=node_selector,
    )


# =============================================================================
# Test Classes - Full Resolution Flow
# =============================================================================


class TestResolutionFlowWithCapability:
    """Integration tests for full resolution using capability filter."""

    async def test_resolve_single_node_by_capability(
        self,
        mock_projection_reader: AsyncMock,
        service: ServiceCapabilityQuery,
    ) -> None:
        """Test full resolution flow: find single node by capability.

        Flow: find_nodes_by_capability -> node_selector.select -> return
        """
        postgres_projection = create_postgres_adapter_projection()
        mock_projection_reader.get_by_capability_tag.return_value = [
            postgres_projection
        ]

        spec = ModelDependencySpec(
            name="storage",
            type="node",
            capability="postgres.storage",
        )
        result = await service.resolve_dependency(spec)

        assert result is not None
        assert result.entity_id == postgres_projection.entity_id
        assert "postgres.storage" in result.capability_tags
        mock_projection_reader.get_by_capability_tag.assert_called_once_with(
            tag="postgres.storage",
            state=EnumRegistrationState.ACTIVE,
        )

    async def test_resolve_multiple_nodes_by_capability_selects_one(
        self,
        mock_projection_reader: AsyncMock,
        service: ServiceCapabilityQuery,
    ) -> None:
        """Test full resolution flow: find multiple nodes, select one.

        When multiple nodes match, the selector should return exactly one.
        """
        projections = [
            create_mock_projection(capability_tags=["database.sql"]),
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

    async def test_resolve_with_contract_type_filter(
        self,
        mock_projection_reader: AsyncMock,
        service: ServiceCapabilityQuery,
    ) -> None:
        """Test resolution filters by contract_type in addition to capability."""
        effect_projection = create_mock_projection(
            capability_tags=["shared.capability"],
            contract_type="effect",
        )
        reducer_projection = create_mock_projection(
            capability_tags=["shared.capability"],
            contract_type="reducer",
        )
        mock_projection_reader.get_by_capability_tag.return_value = [
            effect_projection,
            reducer_projection,
        ]

        spec = ModelDependencySpec(
            name="storage",
            type="node",
            capability="shared.capability",
            contract_type="effect",
        )
        result = await service.resolve_dependency(spec)

        assert result is not None
        assert result.contract_type == "effect"


class TestResolutionFlowWithIntentTypes:
    """Integration tests for full resolution using intent types filter."""

    async def test_resolve_by_single_intent_type(
        self,
        mock_projection_reader: AsyncMock,
        service: ServiceCapabilityQuery,
    ) -> None:
        """Test resolution using single intent type (bulk query path)."""
        postgres_projection = create_postgres_adapter_projection()
        mock_projection_reader.get_by_intent_types.return_value = [postgres_projection]

        spec = ModelDependencySpec(
            name="storage",
            type="node",
            intent_types=["postgres.query"],
        )
        result = await service.resolve_dependency(spec)

        assert result is not None
        assert "postgres.query" in result.intent_types
        mock_projection_reader.get_by_intent_types.assert_called_once()

    async def test_resolve_by_multiple_intent_types_bulk_query(
        self,
        mock_projection_reader: AsyncMock,
        service: ServiceCapabilityQuery,
    ) -> None:
        """Test resolution uses bulk query for multiple intent types.

        This verifies the optimization: N intent types = 1 database query,
        not N database queries.
        """
        postgres_projection = create_postgres_adapter_projection()
        mock_projection_reader.get_by_intent_types.return_value = [postgres_projection]

        spec = ModelDependencySpec(
            name="storage",
            type="node",
            intent_types=["postgres.query", "postgres.upsert", "postgres.delete"],
        )
        result = await service.resolve_dependency(spec)

        assert result is not None
        # Verify bulk query was called exactly once (not 3 times)
        mock_projection_reader.get_by_intent_types.assert_called_once()
        call_kwargs = mock_projection_reader.get_by_intent_types.call_args.kwargs
        assert call_kwargs["intent_types"] == [
            "postgres.query",
            "postgres.upsert",
            "postgres.delete",
        ]


class TestResolutionFlowWithProtocol:
    """Integration tests for full resolution using protocol filter."""

    async def test_resolve_by_protocol(
        self,
        mock_projection_reader: AsyncMock,
        service: ServiceCapabilityQuery,
    ) -> None:
        """Test full resolution flow using protocol filter."""
        kafka_projection = create_kafka_adapter_projection()
        mock_projection_reader.get_by_protocol.return_value = [kafka_projection]

        spec = ModelDependencySpec(
            name="publisher",
            type="protocol",
            protocol="ProtocolEventPublisher",
        )
        result = await service.resolve_dependency(spec)

        assert result is not None
        assert "ProtocolEventPublisher" in result.protocols
        mock_projection_reader.get_by_protocol.assert_called_once_with(
            protocol_name="ProtocolEventPublisher",
            state=EnumRegistrationState.ACTIVE,
        )


# =============================================================================
# Test Classes - Selection Strategies
# =============================================================================


class TestSelectionStrategies:
    """Integration tests for different selection strategies."""

    async def test_first_strategy_returns_first_candidate(
        self,
        mock_projection_reader: AsyncMock,
        service: ServiceCapabilityQuery,
    ) -> None:
        """Test FIRST strategy consistently returns first candidate."""
        projections = [
            create_mock_projection(capability_tags=["test.cap"]),
            create_mock_projection(capability_tags=["test.cap"]),
            create_mock_projection(capability_tags=["test.cap"]),
        ]
        mock_projection_reader.get_by_capability_tag.return_value = projections

        spec = ModelDependencySpec(
            name="test",
            type="node",
            capability="test.cap",
            selection_strategy="first",
        )

        # Multiple calls should return the same (first) node
        results = [await service.resolve_dependency(spec) for _ in range(3)]

        assert all(r is not None for r in results)
        assert all(r.entity_id == projections[0].entity_id for r in results)

    async def test_random_strategy_returns_valid_candidate(
        self,
        mock_projection_reader: AsyncMock,
        service: ServiceCapabilityQuery,
    ) -> None:
        """Test RANDOM strategy returns a valid candidate from the list."""
        projections = [
            create_mock_projection(capability_tags=["test.cap"]),
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

        # Call multiple times and verify all results are valid candidates
        for _ in range(10):
            result = await service.resolve_dependency(spec)
            assert result is not None
            assert result.entity_id in [p.entity_id for p in projections]

    async def test_round_robin_strategy_cycles_through_candidates(
        self,
        mock_projection_reader: AsyncMock,
        service: ServiceCapabilityQuery,
        node_selector: ServiceNodeSelector,
    ) -> None:
        """Test ROUND_ROBIN strategy cycles through all candidates.

        Given 3 candidates, calls should return: 0, 1, 2, 0, 1, 2, ...
        """
        projections = [
            create_mock_projection(capability_tags=["test.cap"]),
            create_mock_projection(capability_tags=["test.cap"]),
            create_mock_projection(capability_tags=["test.cap"]),
        ]
        mock_projection_reader.get_by_capability_tag.return_value = projections

        # Reset round-robin state before test
        await node_selector.reset_round_robin_state()

        spec = ModelDependencySpec(
            name="test",
            type="node",
            capability="test.cap",
            selection_strategy="round_robin",
        )

        # Collect results for 6 calls (2 full cycles)
        results = []
        for _ in range(6):
            result = await service.resolve_dependency(spec)
            assert result is not None
            results.append(result.entity_id)

        # Verify cycling: should see pattern 0,1,2,0,1,2
        expected_ids = [p.entity_id for p in projections] * 2
        assert results == expected_ids

    async def test_round_robin_with_different_selection_keys(
        self,
        mock_projection_reader: AsyncMock,
        service: ServiceCapabilityQuery,
        node_selector: ServiceNodeSelector,
    ) -> None:
        """Test ROUND_ROBIN maintains independent state per dependency name.

        Different dependency specs should have independent round-robin counters.
        """
        projections = [
            create_mock_projection(capability_tags=["shared.cap"]),
            create_mock_projection(capability_tags=["shared.cap"]),
        ]
        mock_projection_reader.get_by_capability_tag.return_value = projections

        # Reset round-robin state before test
        await node_selector.reset_round_robin_state()

        spec_a = ModelDependencySpec(
            name="service_a",
            type="node",
            capability="shared.cap",
            selection_strategy="round_robin",
        )
        spec_b = ModelDependencySpec(
            name="service_b",
            type="node",
            capability="shared.cap",
            selection_strategy="round_robin",
        )

        # Both should start at index 0 (first candidate)
        result_a1 = await service.resolve_dependency(spec_a)
        result_b1 = await service.resolve_dependency(spec_b)

        assert result_a1 is not None
        assert result_b1 is not None
        assert result_a1.entity_id == projections[0].entity_id
        assert result_b1.entity_id == projections[0].entity_id

        # Second call should move each to index 1
        result_a2 = await service.resolve_dependency(spec_a)
        result_b2 = await service.resolve_dependency(spec_b)

        assert result_a2 is not None
        assert result_b2 is not None
        assert result_a2.entity_id == projections[1].entity_id
        assert result_b2.entity_id == projections[1].entity_id


# =============================================================================
# Test Classes - Empty Results Handling
# =============================================================================


class TestEmptyResultsHandling:
    """Integration tests for empty result handling."""

    async def test_resolve_returns_none_when_no_candidates_by_capability(
        self,
        mock_projection_reader: AsyncMock,
        service: ServiceCapabilityQuery,
    ) -> None:
        """Test resolve returns None when no nodes match capability."""
        mock_projection_reader.get_by_capability_tag.return_value = []

        spec = ModelDependencySpec(
            name="missing",
            type="node",
            capability="nonexistent.capability",
        )
        result = await service.resolve_dependency(spec)

        assert result is None

    async def test_resolve_returns_none_when_no_candidates_by_intent_types(
        self,
        mock_projection_reader: AsyncMock,
        service: ServiceCapabilityQuery,
    ) -> None:
        """Test resolve returns None when no nodes match intent types."""
        mock_projection_reader.get_by_intent_types.return_value = []

        spec = ModelDependencySpec(
            name="missing",
            type="node",
            intent_types=["nonexistent.intent"],
        )
        result = await service.resolve_dependency(spec)

        assert result is None

    async def test_resolve_returns_none_when_no_candidates_by_protocol(
        self,
        mock_projection_reader: AsyncMock,
        service: ServiceCapabilityQuery,
    ) -> None:
        """Test resolve returns None when no nodes match protocol."""
        mock_projection_reader.get_by_protocol.return_value = []

        spec = ModelDependencySpec(
            name="missing",
            type="protocol",
            protocol="ProtocolNonExistent",
        )
        result = await service.resolve_dependency(spec)

        assert result is None

    def test_model_validation_rejects_spec_without_filters(
        self,
    ) -> None:
        """Test ModelDependencySpec validates at least one filter is specified.

        Note: This test validates the Pydantic model behavior, not the service.
        The service never receives specs without filters because validation
        happens at model construction time.
        """
        with pytest.raises(ValueError, match="must have at least one discovery filter"):
            ModelDependencySpec(
                name="empty",
                type="node",
                # No capability, intent_types, or protocol
            )

    async def test_resolve_returns_none_when_contract_type_filters_all(
        self,
        mock_projection_reader: AsyncMock,
        service: ServiceCapabilityQuery,
    ) -> None:
        """Test resolve returns None when contract_type filters out all candidates."""
        # Nodes exist but with different contract_type
        effect_projections = [
            create_mock_projection(
                capability_tags=["shared.cap"],
                contract_type="effect",
            ),
        ]
        mock_projection_reader.get_by_capability_tag.return_value = effect_projections

        spec = ModelDependencySpec(
            name="storage",
            type="node",
            capability="shared.cap",
            contract_type="reducer",  # Different from what's available
        )
        result = await service.resolve_dependency(spec)

        assert result is None


# =============================================================================
# Test Classes - Error Propagation
# =============================================================================


class TestErrorPropagation:
    """Integration tests for error propagation from projection reader."""

    async def test_propagates_connection_error_during_resolution(
        self,
        mock_projection_reader: AsyncMock,
        service: ServiceCapabilityQuery,
    ) -> None:
        """Test InfraConnectionError propagates through resolve_dependency."""
        mock_projection_reader.get_by_capability_tag.side_effect = InfraConnectionError(
            "Database connection refused"
        )

        spec = ModelDependencySpec(
            name="storage",
            type="node",
            capability="postgres.storage",
        )

        with pytest.raises(InfraConnectionError) as exc_info:
            await service.resolve_dependency(spec)

        assert "connection refused" in str(exc_info.value).lower()

    async def test_propagates_timeout_error_during_resolution(
        self,
        mock_projection_reader: AsyncMock,
        service: ServiceCapabilityQuery,
    ) -> None:
        """Test InfraTimeoutError propagates through resolve_dependency."""
        mock_projection_reader.get_by_intent_types.side_effect = InfraTimeoutError(
            "Query timed out after 30s"
        )

        spec = ModelDependencySpec(
            name="storage",
            type="node",
            intent_types=["postgres.query"],
        )

        with pytest.raises(InfraTimeoutError) as exc_info:
            await service.resolve_dependency(spec)

        assert "timed out" in str(exc_info.value).lower()

    async def test_propagates_unavailable_error_during_resolution(
        self,
        mock_projection_reader: AsyncMock,
        service: ServiceCapabilityQuery,
    ) -> None:
        """Test InfraUnavailableError (circuit breaker) propagates through resolve."""
        mock_projection_reader.get_by_protocol.side_effect = InfraUnavailableError(
            "Circuit breaker is OPEN for database"
        )

        spec = ModelDependencySpec(
            name="publisher",
            type="protocol",
            protocol="ProtocolEventPublisher",
        )

        with pytest.raises(InfraUnavailableError) as exc_info:
            await service.resolve_dependency(spec)

        assert "circuit breaker" in str(exc_info.value).lower()

    async def test_error_propagation_preserves_correlation_id_in_logs(
        self,
        mock_projection_reader: AsyncMock,
        service: ServiceCapabilityQuery,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """Test that correlation_id is logged when error occurs."""
        import logging

        mock_projection_reader.get_by_capability_tag.side_effect = InfraConnectionError(
            "Connection failed"
        )

        spec = ModelDependencySpec(
            name="storage",
            type="node",
            capability="postgres.storage",
        )

        correlation_id = "test-correlation-123"

        with caplog.at_level(logging.DEBUG):
            with pytest.raises(InfraConnectionError):
                await service.resolve_dependency(spec, correlation_id=correlation_id)

        # Verify correlation_id was used in logging
        # The service should log before the error is raised
        assert any(
            "correlation_id" in record.message or correlation_id in str(record.__dict__)
            for record in caplog.records
        )


# =============================================================================
# Test Classes - Priority Resolution
# =============================================================================


class TestFilterPriority:
    """Integration tests for filter priority in resolve_dependency."""

    async def test_capability_takes_priority_over_intent_types(
        self,
        mock_projection_reader: AsyncMock,
        service: ServiceCapabilityQuery,
    ) -> None:
        """Test capability filter is used when both capability and intent_types set."""
        cap_projection = create_mock_projection(capability_tags=["preferred.cap"])
        mock_projection_reader.get_by_capability_tag.return_value = [cap_projection]

        spec = ModelDependencySpec(
            name="test",
            type="node",
            capability="preferred.cap",
            intent_types=["ignored.intent"],  # Should be ignored
        )
        result = await service.resolve_dependency(spec)

        assert result is not None
        mock_projection_reader.get_by_capability_tag.assert_called_once()
        mock_projection_reader.get_by_intent_types.assert_not_called()

    async def test_capability_takes_priority_over_protocol(
        self,
        mock_projection_reader: AsyncMock,
        service: ServiceCapabilityQuery,
    ) -> None:
        """Test capability filter is used when both capability and protocol set."""
        cap_projection = create_mock_projection(capability_tags=["preferred.cap"])
        mock_projection_reader.get_by_capability_tag.return_value = [cap_projection]

        spec = ModelDependencySpec(
            name="test",
            type="node",
            capability="preferred.cap",
            protocol="IgnoredProtocol",  # Should be ignored
        )
        result = await service.resolve_dependency(spec)

        assert result is not None
        mock_projection_reader.get_by_capability_tag.assert_called_once()
        mock_projection_reader.get_by_protocol.assert_not_called()

    async def test_intent_types_takes_priority_over_protocol(
        self,
        mock_projection_reader: AsyncMock,
        service: ServiceCapabilityQuery,
    ) -> None:
        """Test intent_types filter is used when both intent_types and protocol set."""
        intent_projection = create_mock_projection(intent_types=["preferred.intent"])
        mock_projection_reader.get_by_intent_types.return_value = [intent_projection]

        spec = ModelDependencySpec(
            name="test",
            type="node",
            intent_types=["preferred.intent"],
            protocol="IgnoredProtocol",  # Should be ignored
        )
        result = await service.resolve_dependency(spec)

        assert result is not None
        mock_projection_reader.get_by_intent_types.assert_called_once()
        mock_projection_reader.get_by_protocol.assert_not_called()


__all__: list[str] = []
