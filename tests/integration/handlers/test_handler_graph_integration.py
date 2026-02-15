# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Integration tests for HandlerGraph against real Graph database infrastructure.

These tests validate HandlerGraph behavior against an actual Memgraph or Neo4j
instance via Bolt protocol. They require a running graph database server and
will be skipped gracefully if the server is not available.

CI/CD Graceful Skip Behavior
============================

These tests skip gracefully in CI/CD environments without graph database access:

Skip Conditions:
    - Skips if MEMGRAPH_BOLT_URL environment variable is not set
    - Tests are marked with module-level ``pytestmark`` using ``pytest.mark.skipif``

Example CI/CD Output::

    $ pytest tests/integration/handlers/test_handler_graph_integration.py -v
    test_graph_describe SKIPPED (MEMGRAPH_BOLT_URL not set - Graph integration tests skipped)
    test_graph_full_workflow SKIPPED (MEMGRAPH_BOLT_URL not set - Graph integration tests skipped)

Run with infrastructure::

    $ MEMGRAPH_BOLT_URL=bolt://localhost:7687 uv run pytest tests/integration/handlers/test_handler_graph_integration.py -v

Test Categories
===============

- Handler Metadata Tests: Validate describe functionality and capabilities
- Query Tests: Execute Cypher queries returning results
- Execute Tests: Execute write statements (CREATE, DELETE, etc.)
- Full Workflow Tests: End-to-end graph database operations
- Security Tests: Validate parameterized queries prevent injection

Environment Variables
=====================

    MEMGRAPH_BOLT_URL: Memgraph/Neo4j Bolt URL (required - skip if not set)
        Example: bolt://localhost:7687 or bolt://your-server-ip:7687
    MEMGRAPH_USERNAME: Optional username for authentication
    MEMGRAPH_PASSWORD: Optional password for authentication
    MEMGRAPH_DATABASE: Database name (default: memgraph)

Related Ticket: OMN-1142
"""

from __future__ import annotations

import os
from collections.abc import AsyncGenerator
from typing import TYPE_CHECKING
from uuid import uuid4

import pytest

if TYPE_CHECKING:
    from omnibase_infra.handlers import HandlerGraph

# =============================================================================
# Environment Configuration
# =============================================================================

MEMGRAPH_BOLT_URL = os.getenv("MEMGRAPH_BOLT_URL")
MEMGRAPH_USERNAME = os.getenv("MEMGRAPH_USERNAME", "")
MEMGRAPH_PASSWORD = os.getenv("MEMGRAPH_PASSWORD", "")
MEMGRAPH_DATABASE = os.getenv("MEMGRAPH_DATABASE", "memgraph")

# Check if Memgraph/Graph DB is available based on URL being set
GRAPH_AVAILABLE = MEMGRAPH_BOLT_URL is not None

# =============================================================================
# Test Configuration and Skip Conditions
# =============================================================================

# Module-level markers - skip all tests if graph database is not available
pytestmark = [
    pytest.mark.skipif(
        not GRAPH_AVAILABLE,
        reason="MEMGRAPH_BOLT_URL not set - Graph integration tests skipped",
    ),
]


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def graph_config() -> dict[str, object]:
    """Provide graph database configuration for HandlerGraph.

    Returns:
        Configuration dict for HandlerGraph.initialize()
    """
    config: dict[str, object] = {
        "uri": MEMGRAPH_BOLT_URL,
        "username": MEMGRAPH_USERNAME,
        "password": MEMGRAPH_PASSWORD,
        "database": MEMGRAPH_DATABASE,
        "timeout_seconds": 30.0,
        "max_connection_pool_size": 5,
    }

    return config


@pytest.fixture
async def initialized_graph_handler(
    graph_config: dict[str, object],
) -> AsyncGenerator[HandlerGraph, None]:
    """Provide an initialized HandlerGraph instance with automatic cleanup.

    Creates a HandlerGraph, initializes it with the test configuration,
    yields it for the test, then ensures proper cleanup via shutdown().

    Cleanup Behavior:
        - Calls handler.shutdown() after test completion
        - Closes neo4j driver connection
        - Idempotent: safe to call shutdown() multiple times
        - Ignores cleanup errors to prevent test pollution

    Yields:
        Initialized HandlerGraph ready for graph operations.
    """
    from omnibase_infra.handlers import HandlerGraph

    handler = HandlerGraph()
    await handler.initialize(graph_config)

    yield handler

    # Cleanup: ensure handler is properly shut down
    try:
        await handler.shutdown()
    except Exception:
        pass  # Ignore cleanup errors


@pytest.fixture
def unique_label() -> str:
    """Generate a unique node label for test isolation.

    Returns:
        Unique label prefixed with 'TestNode_'.
    """
    return f"TestNode_{uuid4().hex[:12]}"


# =============================================================================
# Handler Metadata Tests
# =============================================================================


class TestHandlerGraphMetadata:
    """Tests for HandlerGraph metadata and describe functionality."""

    @pytest.mark.asyncio
    async def test_graph_describe(
        self, initialized_graph_handler: HandlerGraph
    ) -> None:
        """Test handler describe returns correct metadata.

        Verifies that:
        - Describe returns supported operations
        - Handler reports correct type and version
        - Handler is initialized
        """
        description = initialized_graph_handler.describe()

        assert description["handler_type"] == "infra_handler"
        assert description["handler_category"] == "effect"
        assert description["initialized"] is True
        assert "graph.query" in description["supported_operations"]
        assert "graph.execute" in description["supported_operations"]


# =============================================================================
# Query Tests
# =============================================================================


class TestHandlerGraphQuery:
    """Tests for HandlerGraph query operations (read-only Cypher)."""

    @pytest.mark.asyncio
    async def test_simple_query(
        self,
        initialized_graph_handler: HandlerGraph,
    ) -> None:
        """Test simple Cypher query returning scalar value.

        Verifies that:
        - Simple queries execute successfully
        - Results are returned in records list
        """
        envelope = {
            "operation": "graph.query",
            "payload": {
                "cypher": "RETURN 1 AS one, 'hello' AS greeting",
                "parameters": {},
            },
            "correlation_id": str(uuid4()),
        }

        result = await initialized_graph_handler.execute(envelope)

        assert result.result.status == "success"
        assert len(result.result.payload.data.records) == 1
        record = result.result.payload.data.records[0]
        assert record.data["one"] == 1
        assert record.data["greeting"] == "hello"

    @pytest.mark.asyncio
    async def test_parameterized_query(
        self,
        initialized_graph_handler: HandlerGraph,
    ) -> None:
        """Test parameterized Cypher query.

        Verifies that:
        - Parameters are properly bound
        - Query returns expected results
        """
        envelope = {
            "operation": "graph.query",
            "payload": {
                "cypher": "RETURN $name AS name, $value AS value",
                "parameters": {"name": "test_param", "value": 42},
            },
            "correlation_id": str(uuid4()),
        }

        result = await initialized_graph_handler.execute(envelope)

        assert result.result.status == "success"
        assert len(result.result.payload.data.records) == 1
        record = result.result.payload.data.records[0]
        assert record.data["name"] == "test_param"
        assert record.data["value"] == 42


# =============================================================================
# Execute Tests
# =============================================================================


class TestHandlerGraphExecute:
    """Tests for HandlerGraph execute operations (write Cypher)."""

    @pytest.mark.asyncio
    async def test_create_and_delete_node(
        self,
        initialized_graph_handler: HandlerGraph,
        unique_label: str,
    ) -> None:
        """Test creating and deleting a node.

        Verifies that:
        - Nodes can be created with labels and properties
        - Counters reflect nodes_created
        - Nodes can be deleted
        - Counters reflect nodes_deleted
        """
        # Create node
        create_envelope = {
            "operation": "graph.execute",
            "payload": {
                "cypher": f"CREATE (n:{unique_label} {{name: $name, age: $age}})",
                "parameters": {"name": "Alice", "age": 30},
            },
            "correlation_id": str(uuid4()),
        }

        result = await initialized_graph_handler.execute(create_envelope)

        assert result.result.status == "success"
        assert result.result.payload.data.counters["nodes_created"] == 1

        # Delete node
        delete_envelope = {
            "operation": "graph.execute",
            "payload": {
                "cypher": f"MATCH (n:{unique_label}) DELETE n",
                "parameters": {},
            },
            "correlation_id": str(uuid4()),
        }

        result = await initialized_graph_handler.execute(delete_envelope)

        assert result.result.status == "success"
        assert result.result.payload.data.counters["nodes_deleted"] == 1

    @pytest.mark.asyncio
    async def test_create_relationship(
        self,
        initialized_graph_handler: HandlerGraph,
        unique_label: str,
    ) -> None:
        """Test creating nodes with relationships.

        Verifies that:
        - Relationships can be created between nodes
        - Counters reflect relationships_created
        """
        # Create two nodes and a relationship
        create_envelope = {
            "operation": "graph.execute",
            "payload": {
                "cypher": f"""
                    CREATE (a:{unique_label} {{name: 'Alice'}})
                    CREATE (b:{unique_label} {{name: 'Bob'}})
                    CREATE (a)-[r:KNOWS]->(b)
                """,
                "parameters": {},
            },
            "correlation_id": str(uuid4()),
        }

        result = await initialized_graph_handler.execute(create_envelope)

        assert result.result.status == "success"
        assert result.result.payload.data.counters["nodes_created"] == 2
        assert result.result.payload.data.counters["relationships_created"] == 1

        # Cleanup: delete nodes and relationships
        cleanup_envelope = {
            "operation": "graph.execute",
            "payload": {
                "cypher": f"MATCH (n:{unique_label}) DETACH DELETE n",
                "parameters": {},
            },
            "correlation_id": str(uuid4()),
        }
        await initialized_graph_handler.execute(cleanup_envelope)


# =============================================================================
# Full Workflow Tests
# =============================================================================


class TestHandlerGraphFullWorkflow:
    """End-to-end workflow tests for HandlerGraph."""

    @pytest.mark.asyncio
    async def test_full_graph_workflow(
        self,
        initialized_graph_handler: HandlerGraph,
        unique_label: str,
    ) -> None:
        """Test complete workflow: create nodes, query, update, delete.

        This test validates the full lifecycle of graph operations:
        1. Create nodes with properties
        2. Query nodes by property
        3. Update node properties
        4. Delete nodes
        5. Verify deletion
        """
        # 1. Create nodes
        create_envelope = {
            "operation": "graph.execute",
            "payload": {
                "cypher": f"""
                    CREATE (n1:{unique_label} {{name: 'Alice', role: 'admin'}})
                    CREATE (n2:{unique_label} {{name: 'Bob', role: 'user'}})
                    CREATE (n3:{unique_label} {{name: 'Charlie', role: 'user'}})
                """,
                "parameters": {},
            },
            "correlation_id": str(uuid4()),
        }
        result = await initialized_graph_handler.execute(create_envelope)
        assert result.result.status == "success"
        assert result.result.payload.data.counters["nodes_created"] == 3

        # 2. Query nodes by role
        query_envelope = {
            "operation": "graph.query",
            "payload": {
                "cypher": f"""
                    MATCH (n:{unique_label})
                    WHERE n.role = $role
                    RETURN n.name AS name
                    ORDER BY n.name
                """,
                "parameters": {"role": "user"},
            },
            "correlation_id": str(uuid4()),
        }
        result = await initialized_graph_handler.execute(query_envelope)
        assert result.result.status == "success"
        assert len(result.result.payload.data.records) == 2
        names = [r.data["name"] for r in result.result.payload.data.records]
        assert "Bob" in names
        assert "Charlie" in names

        # 3. Update node properties
        update_envelope = {
            "operation": "graph.execute",
            "payload": {
                "cypher": f"""
                    MATCH (n:{unique_label})
                    WHERE n.name = $name
                    SET n.role = $new_role
                """,
                "parameters": {"name": "Bob", "new_role": "admin"},
            },
            "correlation_id": str(uuid4()),
        }
        result = await initialized_graph_handler.execute(update_envelope)
        assert result.result.status == "success"
        assert result.result.payload.data.counters["properties_set"] == 1

        # 4. Delete nodes
        delete_envelope = {
            "operation": "graph.execute",
            "payload": {
                "cypher": f"MATCH (n:{unique_label}) DELETE n",
                "parameters": {},
            },
            "correlation_id": str(uuid4()),
        }
        result = await initialized_graph_handler.execute(delete_envelope)
        assert result.result.status == "success"
        assert result.result.payload.data.counters["nodes_deleted"] == 3

        # 5. Verify deletion
        verify_envelope = {
            "operation": "graph.query",
            "payload": {
                "cypher": f"MATCH (n:{unique_label}) RETURN count(n) AS count",
                "parameters": {},
            },
            "correlation_id": str(uuid4()),
        }
        result = await initialized_graph_handler.execute(verify_envelope)
        assert result.result.status == "success"
        assert result.result.payload.data.records[0].data["count"] == 0


# =============================================================================
# Security Tests
# =============================================================================


class TestHandlerGraphSecurity:
    """Tests for security features in HandlerGraph."""

    @pytest.mark.asyncio
    async def test_parameterized_query_prevents_injection(
        self,
        initialized_graph_handler: HandlerGraph,
        unique_label: str,
    ) -> None:
        """Test that parameterized queries prevent Cypher injection.

        Verifies that:
        - Injection attempt is treated as literal string
        - No unintended operations are executed
        """
        # Create a node with an "injection attempt" as the name
        injection_attempt = "Test'; DROP DATABASE--"
        create_envelope = {
            "operation": "graph.execute",
            "payload": {
                "cypher": f"CREATE (n:{unique_label} {{name: $name}})",
                "parameters": {"name": injection_attempt},
            },
            "correlation_id": str(uuid4()),
        }
        result = await initialized_graph_handler.execute(create_envelope)
        assert result.result.status == "success"

        # Query should find the node with the literal string value
        query_envelope = {
            "operation": "graph.query",
            "payload": {
                "cypher": f"""
                    MATCH (n:{unique_label})
                    WHERE n.name = $name
                    RETURN n.name AS name
                """,
                "parameters": {"name": injection_attempt},
            },
            "correlation_id": str(uuid4()),
        }
        result = await initialized_graph_handler.execute(query_envelope)
        assert result.result.status == "success"
        assert len(result.result.payload.data.records) == 1
        assert result.result.payload.data.records[0].data["name"] == injection_attempt

        # Cleanup
        cleanup_envelope = {
            "operation": "graph.execute",
            "payload": {
                "cypher": f"MATCH (n:{unique_label}) DELETE n",
                "parameters": {},
            },
            "correlation_id": str(uuid4()),
        }
        await initialized_graph_handler.execute(cleanup_envelope)


# =============================================================================
# Correlation ID Tests
# =============================================================================


class TestHandlerGraphCorrelationId:
    """Tests for correlation ID handling in HandlerGraph."""

    @pytest.mark.asyncio
    async def test_correlation_id_preserved(
        self,
        initialized_graph_handler: HandlerGraph,
    ) -> None:
        """Verify correlation_id from envelope is preserved in response."""
        test_correlation_id = uuid4()

        envelope = {
            "operation": "graph.query",
            "correlation_id": str(test_correlation_id),
            "payload": {
                "cypher": "RETURN 1 AS one",
                "parameters": {},
            },
        }

        result = await initialized_graph_handler.execute(envelope)

        assert result.correlation_id == test_correlation_id
        assert result.result.correlation_id == test_correlation_id

    @pytest.mark.asyncio
    async def test_correlation_id_generated_if_missing(
        self,
        initialized_graph_handler: HandlerGraph,
    ) -> None:
        """Verify correlation_id is generated when not provided."""
        from uuid import UUID

        envelope = {
            "operation": "graph.query",
            # No correlation_id provided
            "payload": {
                "cypher": "RETURN 1 AS one",
                "parameters": {},
            },
        }

        result = await initialized_graph_handler.execute(envelope)

        # Should have a generated correlation_id
        assert result.correlation_id is not None
        assert isinstance(result.correlation_id, UUID)


# =============================================================================
# Error Handling Tests
# =============================================================================


class TestHandlerGraphErrors:
    """Tests for error handling in HandlerGraph."""

    @pytest.mark.asyncio
    async def test_invalid_cypher_syntax(
        self,
        initialized_graph_handler: HandlerGraph,
    ) -> None:
        """Verify invalid Cypher syntax raises appropriate error."""
        from omnibase_infra.errors import InfraConnectionError

        envelope = {
            "operation": "graph.query",
            "payload": {
                "cypher": "INVALID CYPHER SYNTAX HERE",
                "parameters": {},
            },
            "correlation_id": str(uuid4()),
        }

        with pytest.raises(InfraConnectionError):
            await initialized_graph_handler.execute(envelope)

    @pytest.mark.asyncio
    async def test_execute_not_initialized(
        self,
    ) -> None:
        """Verify execute fails when handler not initialized."""
        from omnibase_infra.errors import RuntimeHostError
        from omnibase_infra.handlers import HandlerGraph

        handler = HandlerGraph()  # Not initialized

        envelope = {
            "operation": "graph.query",
            "payload": {
                "cypher": "RETURN 1",
                "parameters": {},
            },
            "correlation_id": str(uuid4()),
        }

        with pytest.raises(RuntimeHostError, match="not initialized"):
            await handler.execute(envelope)

    @pytest.mark.asyncio
    async def test_unsupported_operation(
        self,
        initialized_graph_handler: HandlerGraph,
    ) -> None:
        """Verify unsupported operation raises appropriate error."""
        from omnibase_infra.errors import RuntimeHostError

        envelope = {
            "operation": "graph.transaction",  # Not supported
            "payload": {
                "cypher": "RETURN 1",
                "parameters": {},
            },
            "correlation_id": str(uuid4()),
        }

        with pytest.raises(RuntimeHostError, match="not supported"):
            await initialized_graph_handler.execute(envelope)

    @pytest.mark.asyncio
    async def test_missing_cypher(
        self,
        initialized_graph_handler: HandlerGraph,
    ) -> None:
        """Verify missing cypher in payload raises appropriate error."""
        from omnibase_infra.errors import RuntimeHostError

        envelope = {
            "operation": "graph.query",
            "payload": {
                # Missing "cypher" key
                "parameters": {},
            },
            "correlation_id": str(uuid4()),
        }

        with pytest.raises(RuntimeHostError, match="cypher"):
            await initialized_graph_handler.execute(envelope)
