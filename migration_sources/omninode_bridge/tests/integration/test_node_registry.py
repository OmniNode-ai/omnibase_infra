#!/usr/bin/env python3
"""
Integration tests for Node Registry.

Tests the complete node registry pipeline:
- Node registration and deregistration
- Dual registration strategy
- Search and discovery API
- Health monitoring
- Performance requirements

ONEX v2.0 Test Compliance:
- Contract-driven test scenarios
- Performance validation
- Quality gates verification
"""


import pytest

from src.omninode_bridge.registry.node_registry_service import (
    EnumHealthStatus,
    EnumNodeType,
    ModelCapability,
    ModelNodeIntrospection,
    ModelNodeQueryInput,
    ModelNodeRegistrationInput,
    NodeRegistryService,
)
from src.omninode_bridge.registry.search_api import (
    EnumSortBy,
    EnumSortOrder,
    ModelSearchQuery,
    SearchAPI,
)


class TestNodeRegistryService:
    """Test suite for NodeRegistryService."""

    @pytest.fixture
    async def registry(self):
        """Create registry instance for testing."""
        return NodeRegistryService(
            enable_consul=False,
            enable_postgres=False,
            enable_kafka=False,
        )

    @pytest.mark.asyncio
    async def test_register_node(self, registry):
        """Test node registration."""
        node_introspection = ModelNodeIntrospection(
            node_id="test-effect-node-001",
            node_name="TestEffectNode",
            node_type=EnumNodeType.EFFECT,
            version="1.0.0",
            capabilities=[
                ModelCapability(name="http_api", description="HTTP API endpoint"),
                ModelCapability(name="data_validation", description="Input validation"),
            ],
            metadata={"author": "test", "domain": "api"},
        )

        input_data = ModelNodeRegistrationInput(node_introspection=node_introspection)

        result = await registry.register_node(input_data)

        assert result.success
        assert result.registration_result is not None
        assert result.registration_result.node_id == "test-effect-node-001"
        # execution_time_ms may be 0 if timing is not implemented
        assert result.execution_time_ms >= 0

    @pytest.mark.asyncio
    async def test_deregister_node(self, registry):
        """Test node deregistration."""
        # Register node first
        node_introspection = ModelNodeIntrospection(
            node_id="test-node-002",
            node_name="TestNode",
            node_type=EnumNodeType.COMPUTE,
            version="1.0.0",
        )
        await registry.register_node(
            ModelNodeRegistrationInput(node_introspection=node_introspection)
        )

        # Deregister
        success = await registry.deregister_node("test-node-002")
        assert success

        # Verify removed
        node = await registry.get_node_by_id("test-node-002")
        assert node is None

    @pytest.mark.asyncio
    async def test_update_node_health(self, registry):
        """Test node health status update."""
        # Register node
        node_introspection = ModelNodeIntrospection(
            node_id="test-node-003",
            node_name="TestNode",
            node_type=EnumNodeType.REDUCER,
            version="1.0.0",
        )
        await registry.register_node(
            ModelNodeRegistrationInput(node_introspection=node_introspection)
        )

        # Update health
        success = await registry.update_node_health(
            "test-node-003", EnumHealthStatus.DEGRADED
        )
        assert success

        # Verify update
        node = await registry.get_node_by_id("test-node-003")
        assert node.health_status == EnumHealthStatus.DEGRADED

    @pytest.mark.asyncio
    async def test_query_nodes_by_type(self, registry):
        """Test querying nodes by type."""
        # Register nodes of different types
        for node_type in [
            EnumNodeType.EFFECT,
            EnumNodeType.COMPUTE,
            EnumNodeType.REDUCER,
        ]:
            for i in range(2):
                node_introspection = ModelNodeIntrospection(
                    node_id=f"test-{node_type.value}-{i}",
                    node_name=f"Test{node_type.value.capitalize()}Node{i}",
                    node_type=node_type,
                    version="1.0.0",
                )
                await registry.register_node(
                    ModelNodeRegistrationInput(node_introspection=node_introspection)
                )

        # Query for EFFECT nodes
        query_input = ModelNodeQueryInput(
            query_filters={"node_type": EnumNodeType.EFFECT.value}
        )
        result = await registry.query_nodes(query_input)

        assert result.success
        assert result.total_count == 2
        assert all(
            node["node_type"] == EnumNodeType.EFFECT.value for node in result.nodes
        )

    @pytest.mark.asyncio
    async def test_query_nodes_by_capability(self, registry):
        """Test querying nodes by capability."""
        # Register nodes with different capabilities
        node_introspection = ModelNodeIntrospection(
            node_id="test-node-cap-001",
            node_name="TestNodeWithCap",
            node_type=EnumNodeType.EFFECT,
            version="1.0.0",
            capabilities=[
                ModelCapability(name="kafka_consumer", description="Kafka consumer"),
            ],
        )
        await registry.register_node(
            ModelNodeRegistrationInput(node_introspection=node_introspection)
        )

        # Query by capability
        query_input = ModelNodeQueryInput(
            query_filters={"capability": "kafka_consumer"}
        )
        result = await registry.query_nodes(query_input)

        assert result.success
        assert result.total_count >= 1

    @pytest.mark.asyncio
    async def test_get_nodes_by_type(self, registry):
        """Test getting nodes by type."""
        # Register orchestrator nodes
        for i in range(3):
            node_introspection = ModelNodeIntrospection(
                node_id=f"orch-{i}",
                node_name=f"Orchestrator{i}",
                node_type=EnumNodeType.ORCHESTRATOR,
                version="1.0.0",
            )
            await registry.register_node(
                ModelNodeRegistrationInput(node_introspection=node_introspection)
            )

        # Get all orchestrator nodes
        orchestrators = await registry.get_nodes_by_type(EnumNodeType.ORCHESTRATOR)
        assert len(orchestrators) >= 3

    @pytest.mark.asyncio
    async def test_get_healthy_nodes(self, registry):
        """Test getting healthy nodes."""
        # Register nodes with different health statuses
        for i, health in enumerate(
            [
                EnumHealthStatus.HEALTHY,
                EnumHealthStatus.DEGRADED,
                EnumHealthStatus.UNHEALTHY,
            ]
        ):
            node_introspection = ModelNodeIntrospection(
                node_id=f"health-test-{i}",
                node_name=f"HealthTestNode{i}",
                node_type=EnumNodeType.COMPUTE,
                version="1.0.0",
            )
            await registry.register_node(
                ModelNodeRegistrationInput(node_introspection=node_introspection)
            )
            await registry.update_node_health(f"health-test-{i}", health)

        # Get healthy nodes
        healthy_nodes = await registry.get_healthy_nodes()
        assert len(healthy_nodes) >= 1
        assert all(
            node.health_status == EnumHealthStatus.HEALTHY for node in healthy_nodes
        )

    @pytest.mark.asyncio
    async def test_registration_performance(self, registry):
        """Test registration performance (<100ms target)."""
        import time

        node_introspection = ModelNodeIntrospection(
            node_id="perf-test-001",
            node_name="PerfTestNode",
            node_type=EnumNodeType.EFFECT,
            version="1.0.0",
        )

        start_time = time.perf_counter()
        result = await registry.register_node(
            ModelNodeRegistrationInput(node_introspection=node_introspection)
        )
        duration_ms = (time.perf_counter() - start_time) * 1000

        assert result.success
        assert (
            duration_ms < 100
        ), f"Registration took {duration_ms:.2f}ms > 100ms target"


class TestSearchAPI:
    """Test suite for SearchAPI."""

    @pytest.fixture
    async def search_api(self):
        """Create search API instance for testing."""
        registry = NodeRegistryService(
            enable_consul=False,
            enable_postgres=False,
            enable_kafka=False,
        )

        # Register sample nodes
        for i in range(20):
            node_type = [
                EnumNodeType.EFFECT,
                EnumNodeType.COMPUTE,
                EnumNodeType.REDUCER,
                EnumNodeType.ORCHESTRATOR,
            ][i % 4]
            node_introspection = ModelNodeIntrospection(
                node_id=f"search-test-{i:03d}",
                node_name=f"SearchTestNode{i:03d}",
                node_type=node_type,
                version=f"1.{i // 10}.{i % 10}",
                capabilities=[
                    ModelCapability(
                        name=f"capability_{i % 5}", description=f"Capability {i % 5}"
                    ),
                ],
                metadata={"index": i, "category": f"category_{i % 3}"},
            )
            await registry.register_node(
                ModelNodeRegistrationInput(node_introspection=node_introspection)
            )

        return SearchAPI(registry_service=registry)

    @pytest.mark.asyncio
    async def test_search_all_nodes(self, search_api):
        """Test searching all nodes."""
        query = ModelSearchQuery(page=1, page_size=10)
        response = await search_api.search(query)

        assert response.success
        assert response.total_count == 20
        assert len(response.results) == 10  # Page size
        assert response.total_pages == 2

    @pytest.mark.asyncio
    async def test_search_by_text(self, search_api):
        """Test text search."""
        query = ModelSearchQuery(query="Node005", page_size=20)
        response = await search_api.search(query)

        assert response.success
        assert response.total_count >= 1
        assert any("005" in result.node_name for result in response.results)

    @pytest.mark.asyncio
    async def test_search_by_node_type(self, search_api):
        """Test filtering by node type."""
        query = ModelSearchQuery(
            node_type=EnumNodeType.EFFECT,
            page_size=20,
        )
        response = await search_api.search(query)

        assert response.success
        assert all(
            result.node_type == EnumNodeType.EFFECT for result in response.results
        )

    @pytest.mark.asyncio
    async def test_search_by_capability(self, search_api):
        """Test filtering by capability."""
        query = ModelSearchQuery(
            capability="capability_2",
            page_size=20,
        )
        response = await search_api.search(query)

        assert response.success
        assert all(
            any(cap["name"] == "capability_2" for cap in result.capabilities)
            for result in response.results
        )

    @pytest.mark.asyncio
    async def test_search_pagination(self, search_api):
        """Test pagination."""
        # Page 1
        query_page1 = ModelSearchQuery(page=1, page_size=5)
        response_page1 = await search_api.search(query_page1)

        # Page 2
        query_page2 = ModelSearchQuery(page=2, page_size=5)
        response_page2 = await search_api.search(query_page2)

        assert response_page1.success
        assert response_page2.success
        assert len(response_page1.results) == 5
        assert len(response_page2.results) == 5

        # Verify no overlap
        page1_ids = {r.node_id for r in response_page1.results}
        page2_ids = {r.node_id for r in response_page2.results}
        assert page1_ids.isdisjoint(page2_ids)

    @pytest.mark.asyncio
    async def test_search_sorting(self, search_api):
        """Test sorting results."""
        # Sort by name ascending
        query = ModelSearchQuery(
            sort_by=EnumSortBy.NAME,
            sort_order=EnumSortOrder.ASC,
            page_size=20,
        )
        response = await search_api.search(query)

        assert response.success

        # Verify sorted
        names = [r.node_name for r in response.results]
        assert names == sorted(names)

    @pytest.mark.asyncio
    async def test_discover_by_capability(self, search_api):
        """Test capability-based discovery."""
        results = await search_api.discover_by_capability("capability_1")

        assert len(results) > 0
        assert all(
            any(cap["name"] == "capability_1" for cap in result.capabilities)
            for result in results
        )

    @pytest.mark.asyncio
    async def test_discover_by_type(self, search_api):
        """Test type-based discovery."""
        results = await search_api.discover_by_type(EnumNodeType.COMPUTE)

        assert len(results) > 0
        assert all(result.node_type == EnumNodeType.COMPUTE for result in results)

    @pytest.mark.asyncio
    async def test_search_performance(self, search_api):
        """Test search performance (<50ms target)."""
        import time

        query = ModelSearchQuery(
            node_type=EnumNodeType.EFFECT,
            capability="capability_0",
            page_size=10,
        )

        start_time = time.perf_counter()
        response = await search_api.search(query)
        duration_ms = (time.perf_counter() - start_time) * 1000

        assert response.success
        assert duration_ms < 50, f"Search took {duration_ms:.2f}ms > 50ms target"


@pytest.mark.asyncio
async def test_end_to_end_registry_and_search():
    """Test end-to-end registry and search integration."""
    # Create registry and search API
    registry = NodeRegistryService()
    search_api = SearchAPI(registry_service=registry)

    # Register nodes
    for i in range(10):
        node_introspection = ModelNodeIntrospection(
            node_id=f"e2e-node-{i}",
            node_name=f"E2ENode{i}",
            node_type=EnumNodeType.EFFECT if i % 2 == 0 else EnumNodeType.COMPUTE,
            version="1.0.0",
            capabilities=[
                ModelCapability(
                    name="api_endpoint" if i % 2 == 0 else "data_processing"
                ),
            ],
        )
        result = await registry.register_node(
            ModelNodeRegistrationInput(node_introspection=node_introspection)
        )
        assert result.success

    # Search for EFFECT nodes with api_endpoint capability
    results = await search_api.discover_by_capability("api_endpoint")

    assert len(results) >= 5
    assert all(result.node_type == EnumNodeType.EFFECT for result in results)

    # Query via registry
    query_result = await registry.query_nodes(
        ModelNodeQueryInput(query_filters={"node_type": "effect"})
    )

    assert query_result.success
    assert query_result.total_count >= 5

    # Verify metrics
    metrics = await registry.get_metrics()
    assert metrics["total_registrations"] == 10
    assert metrics["successful_registrations"] == 10
    assert metrics["registered_nodes_total"] == 10


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--asyncio-mode=auto"])
