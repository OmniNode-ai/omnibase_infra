# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
# mypy: disable-error-code="index, operator, arg-type"
"""Unit tests for HandlerGraph implementing ProtocolGraphDatabaseHandler.

These tests verify the SPI protocol implementation using mocked neo4j AsyncDriver
to validate HandlerGraph behavior without requiring actual graph database infrastructure.
"""

from __future__ import annotations

from typing import TYPE_CHECKING
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

import pytest

from omnibase_infra.errors import (
    InfraAuthenticationError,
    InfraConnectionError,
    InfraUnavailableError,
    RuntimeHostError,
)
from omnibase_infra.handlers.handler_graph import HandlerGraph
from tests.helpers import filter_handler_warnings

if TYPE_CHECKING:
    from collections.abc import Generator


@pytest.fixture
def handler(mock_container: MagicMock) -> HandlerGraph:
    """Create a fresh HandlerGraph instance with mock container."""
    return HandlerGraph(container=mock_container)


@pytest.fixture
def mock_driver() -> MagicMock:
    """Create mock neo4j AsyncDriver fixture."""
    driver = MagicMock()
    driver.verify_connectivity = AsyncMock()
    driver.close = AsyncMock()
    driver.session = MagicMock()
    driver.get_server_info = AsyncMock(return_value=MagicMock(agent="Neo4j/5.0.0"))
    return driver


class TestHandlerGraphProperties:
    """Test HandlerGraph type and capability properties."""

    def test_handler_type_returns_graph_database(self, handler: HandlerGraph) -> None:
        """Test handler_type property returns 'graph_database'."""
        assert handler.handler_type == "graph_database"

    def test_supports_transactions_returns_true(self, handler: HandlerGraph) -> None:
        """Test supports_transactions property returns True."""
        assert handler.supports_transactions is True


class TestHandlerGraphInitialization:
    """Test HandlerGraph initialization and configuration."""

    @pytest.mark.asyncio
    async def test_initialize_success(
        self, handler: HandlerGraph, mock_driver: MagicMock
    ) -> None:
        """Test successful initialization creates driver and verifies connectivity."""
        with patch(
            "omnibase_infra.handlers.handler_graph.AsyncGraphDatabase"
        ) as mock_db:
            mock_db.driver.return_value = mock_driver

            await handler.initialize(
                connection_uri="bolt://localhost:7687",
            )

            assert handler._initialized is True
            assert handler._driver is mock_driver
            mock_db.driver.assert_called_once()
            mock_driver.verify_connectivity.assert_called_once()

            await handler.shutdown()

    @pytest.mark.asyncio
    async def test_initialize_with_credentials(
        self, handler: HandlerGraph, mock_driver: MagicMock
    ) -> None:
        """Test initialization with username and password."""
        with patch(
            "omnibase_infra.handlers.handler_graph.AsyncGraphDatabase"
        ) as mock_db:
            mock_db.driver.return_value = mock_driver

            await handler.initialize(
                connection_uri="bolt://localhost:7687",
                auth=("neo4j", "test-password"),
            )

            assert handler._initialized is True
            mock_db.driver.assert_called_once()

            await handler.shutdown()

    @pytest.mark.asyncio
    async def test_initialize_without_auth(
        self, handler: HandlerGraph, mock_driver: MagicMock
    ) -> None:
        """Test initialization without authentication."""
        with patch(
            "omnibase_infra.handlers.handler_graph.AsyncGraphDatabase"
        ) as mock_db:
            mock_db.driver.return_value = mock_driver

            await handler.initialize(
                connection_uri="bolt://localhost:7687",
                auth=None,
            )

            mock_db.driver.assert_called_once_with(
                "bolt://localhost:7687",
                auth=None,
                max_connection_pool_size=50,
                encrypted=False,
            )

            await handler.shutdown()

    @pytest.mark.asyncio
    async def test_initialize_with_options(
        self, handler: HandlerGraph, mock_driver: MagicMock
    ) -> None:
        """Test initialization with custom options."""
        with patch(
            "omnibase_infra.handlers.handler_graph.AsyncGraphDatabase"
        ) as mock_db:
            mock_db.driver.return_value = mock_driver

            await handler.initialize(
                connection_uri="bolt://localhost:7687",
                options={
                    "max_connection_pool_size": 100,
                    "timeout_seconds": 60.0,
                    "database": "neo4j",
                },
            )

            assert handler._timeout == 60.0
            assert handler._pool_size == 100
            assert handler._database == "neo4j"

            await handler.shutdown()

    @pytest.mark.asyncio
    async def test_initialize_auth_error(self, handler: HandlerGraph) -> None:
        """Test initialization fails with authentication error."""
        with patch(
            "omnibase_infra.handlers.handler_graph.AsyncGraphDatabase"
        ) as mock_db:
            from neo4j.exceptions import AuthError

            mock_driver = AsyncMock()
            mock_db.driver.return_value = mock_driver
            mock_driver.verify_connectivity.side_effect = AuthError(
                "Invalid credentials"
            )

            with pytest.raises(InfraAuthenticationError) as exc_info:
                await handler.initialize(
                    connection_uri="bolt://localhost:7687",
                    auth=("bad-user", "bad-password"),
                )

            assert "authentication" in str(exc_info.value).lower()

    @pytest.mark.asyncio
    async def test_initialize_connection_error(self, handler: HandlerGraph) -> None:
        """Test initialization fails with connection error."""
        with patch(
            "omnibase_infra.handlers.handler_graph.AsyncGraphDatabase"
        ) as mock_db:
            from neo4j.exceptions import ServiceUnavailable

            mock_driver = AsyncMock()
            mock_db.driver.return_value = mock_driver
            mock_driver.verify_connectivity.side_effect = ServiceUnavailable(
                "Cannot connect"
            )

            with pytest.raises(InfraConnectionError) as exc_info:
                await handler.initialize(
                    connection_uri="bolt://bad-host:7687",
                )

            assert "connect" in str(exc_info.value).lower()

    @pytest.mark.asyncio
    async def test_initialize_generic_error(self, handler: HandlerGraph) -> None:
        """Test initialization fails with generic error wrapped as connection error."""
        with patch(
            "omnibase_infra.handlers.handler_graph.AsyncGraphDatabase"
        ) as mock_db:
            mock_driver = AsyncMock()
            mock_db.driver.return_value = mock_driver
            mock_driver.verify_connectivity.side_effect = RuntimeError("Unknown error")

            with pytest.raises(InfraConnectionError) as exc_info:
                await handler.initialize(
                    connection_uri="bolt://localhost:7687",
                )

            assert "RuntimeError" in str(exc_info.value)


class TestHandlerGraphShutdown:
    """Test HandlerGraph shutdown functionality."""

    @pytest.mark.asyncio
    async def test_shutdown_closes_driver(
        self, handler: HandlerGraph, mock_driver: MagicMock
    ) -> None:
        """Test shutdown closes the driver properly."""
        with patch(
            "omnibase_infra.handlers.handler_graph.AsyncGraphDatabase"
        ) as mock_db:
            mock_db.driver.return_value = mock_driver

            await handler.initialize(connection_uri="bolt://localhost:7687")
            await handler.shutdown()

            mock_driver.close.assert_called_once()
            assert handler._driver is None
            assert handler._initialized is False

    @pytest.mark.asyncio
    async def test_shutdown_without_initialize_is_safe(
        self, handler: HandlerGraph
    ) -> None:
        """Test shutdown without initialize doesn't raise error."""
        await handler.shutdown()

        assert handler._initialized is False
        assert handler._driver is None

    @pytest.mark.asyncio
    async def test_multiple_shutdown_calls_safe(
        self, handler: HandlerGraph, mock_driver: MagicMock
    ) -> None:
        """Test multiple shutdown calls are safe (idempotent)."""
        with patch(
            "omnibase_infra.handlers.handler_graph.AsyncGraphDatabase"
        ) as mock_db:
            mock_db.driver.return_value = mock_driver

            await handler.initialize(connection_uri="bolt://localhost:7687")
            await handler.shutdown()
            await handler.shutdown()  # Second call should not raise

            assert handler._initialized is False
            assert handler._driver is None


class TestHandlerGraphExecuteQuery:
    """Test HandlerGraph execute_query operation."""

    def _setup_mock_session(
        self,
        mock_driver: MagicMock,
        records_data: list[dict[str, object]],
        counters: dict[str, int] | None = None,
    ) -> AsyncMock:
        """Set up mock session with result data."""
        if counters is None:
            counters = {
                "nodes_created": 0,
                "nodes_deleted": 0,
                "relationships_created": 0,
                "relationships_deleted": 0,
                "properties_set": 0,
                "labels_added": 0,
                "labels_removed": 0,
            }

        mock_session = AsyncMock()
        mock_result = AsyncMock()
        mock_result.data = AsyncMock(return_value=records_data)

        mock_summary = MagicMock()
        mock_summary.query_type = "r"
        mock_summary.counters = MagicMock()
        mock_summary.counters.contains_updates = counters.get("nodes_created", 0) > 0
        for key, value in counters.items():
            setattr(mock_summary.counters, key, value)
        mock_result.consume = AsyncMock(return_value=mock_summary)

        mock_session.run = AsyncMock(return_value=mock_result)

        mock_session_cm = MagicMock()
        mock_session_cm.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session_cm.__aexit__ = AsyncMock(return_value=None)
        mock_driver.session.return_value = mock_session_cm

        return mock_session

    @pytest.mark.asyncio
    async def test_execute_query_success(
        self, handler: HandlerGraph, mock_driver: MagicMock
    ) -> None:
        """Test successful Cypher query execution."""
        with patch(
            "omnibase_infra.handlers.handler_graph.AsyncGraphDatabase"
        ) as mock_db:
            mock_db.driver.return_value = mock_driver

            records_data = [
                {"name": "Alice", "age": 30},
                {"name": "Bob", "age": 25},
            ]
            mock_session = self._setup_mock_session(mock_driver, records_data)

            await handler.initialize(connection_uri="bolt://localhost:7687")

            result = await handler.execute_query(
                query="MATCH (n:Person) RETURN n.name as name, n.age as age",
                parameters={},
            )

            assert len(result.records) == 2
            assert result.records[0]["name"] == "Alice"
            assert result.records[1]["name"] == "Bob"
            assert result.execution_time_ms >= 0
            mock_session.run.assert_called_once()

            await handler.shutdown()

    @pytest.mark.asyncio
    async def test_execute_query_with_parameters(
        self, handler: HandlerGraph, mock_driver: MagicMock
    ) -> None:
        """Test query with parameters passes them correctly."""
        with patch(
            "omnibase_infra.handlers.handler_graph.AsyncGraphDatabase"
        ) as mock_db:
            mock_db.driver.return_value = mock_driver

            records_data = [{"name": "Alice", "age": 30}]
            mock_session = self._setup_mock_session(mock_driver, records_data)

            await handler.initialize(connection_uri="bolt://localhost:7687")

            await handler.execute_query(
                query="MATCH (n:Person {name: $name}) RETURN n",
                parameters={"name": "Alice"},
            )

            mock_session.run.assert_called_once_with(
                "MATCH (n:Person {name: $name}) RETURN n",
                {"name": "Alice"},
            )

            await handler.shutdown()

    @pytest.mark.asyncio
    async def test_execute_query_empty_result(
        self, handler: HandlerGraph, mock_driver: MagicMock
    ) -> None:
        """Test query returning no records."""
        with patch(
            "omnibase_infra.handlers.handler_graph.AsyncGraphDatabase"
        ) as mock_db:
            mock_db.driver.return_value = mock_driver

            self._setup_mock_session(mock_driver, [])

            await handler.initialize(connection_uri="bolt://localhost:7687")

            result = await handler.execute_query(
                query="MATCH (n:NonExistent) RETURN n",
            )

            assert len(result.records) == 0

            await handler.shutdown()

    @pytest.mark.asyncio
    async def test_execute_query_not_initialized(self, handler: HandlerGraph) -> None:
        """Test execute_query fails when not initialized."""
        with pytest.raises(RuntimeHostError) as exc_info:
            await handler.execute_query(
                query="MATCH (n) RETURN n",
            )

        assert "not initialized" in str(exc_info.value).lower()

    @pytest.mark.asyncio
    async def test_execute_query_database_error(
        self, handler: HandlerGraph, mock_driver: MagicMock
    ) -> None:
        """Test execute_query handles database errors."""
        with patch(
            "omnibase_infra.handlers.handler_graph.AsyncGraphDatabase"
        ) as mock_db:
            mock_db.driver.return_value = mock_driver

            mock_session = AsyncMock()
            from neo4j.exceptions import Neo4jError

            mock_session.run = AsyncMock(side_effect=Neo4jError("Query failed"))
            mock_session_cm = MagicMock()
            mock_session_cm.__aenter__ = AsyncMock(return_value=mock_session)
            mock_session_cm.__aexit__ = AsyncMock(return_value=None)
            mock_driver.session.return_value = mock_session_cm

            await handler.initialize(connection_uri="bolt://localhost:7687")

            with pytest.raises(InfraConnectionError) as exc_info:
                await handler.execute_query(
                    query="INVALID QUERY",
                )

            assert "failed" in str(exc_info.value).lower()

            await handler.shutdown()


class TestHandlerGraphExecuteQueryBatch:
    """Test HandlerGraph execute_query_batch operation."""

    @pytest.mark.asyncio
    async def test_execute_query_batch_success(
        self, handler: HandlerGraph, mock_driver: MagicMock
    ) -> None:
        """Test successful batch query execution."""
        with patch(
            "omnibase_infra.handlers.handler_graph.AsyncGraphDatabase"
        ) as mock_db:
            mock_db.driver.return_value = mock_driver

            # Setup mock for transaction
            mock_session = AsyncMock()
            mock_tx = AsyncMock()

            mock_result = AsyncMock()
            mock_result.data = AsyncMock(return_value=[{"count": 1}])
            mock_summary = MagicMock()
            mock_summary.query_type = "w"
            mock_summary.counters = MagicMock()
            mock_summary.counters.contains_updates = True
            mock_summary.counters.nodes_created = 1
            mock_summary.counters.nodes_deleted = 0
            mock_summary.counters.relationships_created = 0
            mock_summary.counters.relationships_deleted = 0
            mock_summary.counters.properties_set = 2
            mock_summary.counters.labels_added = 1
            mock_summary.counters.labels_removed = 0
            mock_result.consume = AsyncMock(return_value=mock_summary)

            mock_tx.run = AsyncMock(return_value=mock_result)
            mock_tx.commit = AsyncMock()
            mock_tx.rollback = AsyncMock()

            mock_session.begin_transaction = AsyncMock(return_value=mock_tx)

            mock_session_cm = MagicMock()
            mock_session_cm.__aenter__ = AsyncMock(return_value=mock_session)
            mock_session_cm.__aexit__ = AsyncMock(return_value=None)
            mock_driver.session.return_value = mock_session_cm

            await handler.initialize(connection_uri="bolt://localhost:7687")

            queries = [
                ("CREATE (n:Person {name: $name})", {"name": "Alice"}),
                ("CREATE (n:Person {name: $name})", {"name": "Bob"}),
            ]

            result = await handler.execute_query_batch(
                queries=queries,
                transaction=True,
            )

            assert result.success is True
            assert len(result.results) == 2
            assert result.rollback_occurred is False
            mock_tx.commit.assert_called_once()

            await handler.shutdown()

    @pytest.mark.asyncio
    async def test_execute_query_batch_not_initialized(
        self, handler: HandlerGraph
    ) -> None:
        """Test execute_query_batch fails when not initialized."""
        with pytest.raises(RuntimeHostError) as exc_info:
            await handler.execute_query_batch(
                queries=[("MATCH (n) RETURN n", {})],
            )

        assert "not initialized" in str(exc_info.value).lower()


class TestHandlerGraphCreateNode:
    """Test HandlerGraph create_node operation."""

    @pytest.mark.asyncio
    async def test_create_node_success(
        self, handler: HandlerGraph, mock_driver: MagicMock
    ) -> None:
        """Test successful node creation."""
        with patch(
            "omnibase_infra.handlers.handler_graph.AsyncGraphDatabase"
        ) as mock_db:
            mock_db.driver.return_value = mock_driver

            mock_session = AsyncMock()
            mock_result = AsyncMock()

            mock_node = MagicMock()
            mock_node.labels = frozenset(["Person"])
            mock_node.items = MagicMock(return_value=[("name", "Alice"), ("age", 30)])

            mock_record = MagicMock()
            mock_record.__getitem__ = lambda self, key: {
                "n": mock_node,
                "eid": "4:abc:123",
                "nid": 123,
            }[key]

            mock_result.single = AsyncMock(return_value=mock_record)
            mock_result.consume = AsyncMock()

            mock_session.run = AsyncMock(return_value=mock_result)

            mock_session_cm = MagicMock()
            mock_session_cm.__aenter__ = AsyncMock(return_value=mock_session)
            mock_session_cm.__aexit__ = AsyncMock(return_value=None)
            mock_driver.session.return_value = mock_session_cm

            await handler.initialize(connection_uri="bolt://localhost:7687")

            result = await handler.create_node(
                labels=["Person"],
                properties={"name": "Alice", "age": 30},
            )

            assert result.id == "123"
            assert result.element_id == "4:abc:123"
            assert "Person" in result.labels
            assert result.properties["name"] == "Alice"

            await handler.shutdown()

    @pytest.mark.asyncio
    async def test_create_node_not_initialized(self, handler: HandlerGraph) -> None:
        """Test create_node fails when not initialized."""
        with pytest.raises(RuntimeHostError) as exc_info:
            await handler.create_node(
                labels=["Person"],
                properties={"name": "Alice"},
            )

        assert "not initialized" in str(exc_info.value).lower()


class TestHandlerGraphCreateRelationship:
    """Test HandlerGraph create_relationship operation."""

    @pytest.mark.asyncio
    async def test_create_relationship_success(
        self, handler: HandlerGraph, mock_driver: MagicMock
    ) -> None:
        """Test successful relationship creation."""
        with patch(
            "omnibase_infra.handlers.handler_graph.AsyncGraphDatabase"
        ) as mock_db:
            mock_db.driver.return_value = mock_driver

            mock_session = AsyncMock()
            mock_result = AsyncMock()

            mock_rel = MagicMock()
            mock_rel.type = "KNOWS"
            mock_rel.items = MagicMock(return_value=[("since", 2020)])

            mock_record = MagicMock()
            mock_record.__getitem__ = lambda self, key: {
                "r": mock_rel,
                "eid": "5:abc:456",
                "rid": 456,
                "start_eid": "4:abc:123",
                "end_eid": "4:abc:124",
            }[key]

            mock_result.single = AsyncMock(return_value=mock_record)
            mock_result.consume = AsyncMock()

            mock_session.run = AsyncMock(return_value=mock_result)

            mock_session_cm = MagicMock()
            mock_session_cm.__aenter__ = AsyncMock(return_value=mock_session)
            mock_session_cm.__aexit__ = AsyncMock(return_value=None)
            mock_driver.session.return_value = mock_session_cm

            await handler.initialize(connection_uri="bolt://localhost:7687")

            result = await handler.create_relationship(
                from_node_id=123,
                to_node_id=124,
                relationship_type="KNOWS",
                properties={"since": 2020},
            )

            assert result.id == "456"
            assert result.type == "KNOWS"
            assert result.start_node_id == "4:abc:123"
            assert result.end_node_id == "4:abc:124"

            await handler.shutdown()

    @pytest.mark.asyncio
    async def test_create_relationship_not_initialized(
        self, handler: HandlerGraph
    ) -> None:
        """Test create_relationship fails when not initialized."""
        with pytest.raises(RuntimeHostError) as exc_info:
            await handler.create_relationship(
                from_node_id=1,
                to_node_id=2,
                relationship_type="KNOWS",
            )

        assert "not initialized" in str(exc_info.value).lower()


class TestHandlerGraphDeleteNode:
    """Test HandlerGraph delete_node operation."""

    @pytest.mark.asyncio
    async def test_delete_node_success(
        self, handler: HandlerGraph, mock_driver: MagicMock
    ) -> None:
        """Test successful node deletion."""
        with patch(
            "omnibase_infra.handlers.handler_graph.AsyncGraphDatabase"
        ) as mock_db:
            mock_db.driver.return_value = mock_driver

            mock_session = AsyncMock()
            mock_result = AsyncMock()

            mock_record = MagicMock()
            mock_record.__getitem__ = lambda self, key: {"deleted": 1}[key]

            mock_result.single = AsyncMock(return_value=mock_record)
            mock_result.consume = AsyncMock()

            mock_session.run = AsyncMock(return_value=mock_result)

            mock_session_cm = MagicMock()
            mock_session_cm.__aenter__ = AsyncMock(return_value=mock_session)
            mock_session_cm.__aexit__ = AsyncMock(return_value=None)
            mock_driver.session.return_value = mock_session_cm

            await handler.initialize(connection_uri="bolt://localhost:7687")

            result = await handler.delete_node(node_id=123)

            assert result.success is True
            assert result.node_id == "123"
            assert result.execution_time_ms >= 0

            await handler.shutdown()

    @pytest.mark.asyncio
    async def test_delete_node_with_detach(
        self, handler: HandlerGraph, mock_driver: MagicMock
    ) -> None:
        """Test node deletion with detach (delete relationships)."""
        with patch(
            "omnibase_infra.handlers.handler_graph.AsyncGraphDatabase"
        ) as mock_db:
            mock_db.driver.return_value = mock_driver

            # First call for counting relationships
            mock_count_record = MagicMock()
            mock_count_record.__getitem__ = lambda self, key: {"cnt": 3}[key]
            mock_count_result = AsyncMock()
            mock_count_result.single = AsyncMock(return_value=mock_count_record)
            mock_count_result.consume = AsyncMock()

            # Second call for actual delete
            mock_delete_record = MagicMock()
            mock_delete_record.__getitem__ = lambda self, key: {"deleted": 1}[key]
            mock_delete_result = AsyncMock()
            mock_delete_result.single = AsyncMock(return_value=mock_delete_record)
            mock_delete_result.consume = AsyncMock()

            mock_session = AsyncMock()
            mock_session.run = AsyncMock(
                side_effect=[mock_count_result, mock_delete_result]
            )

            mock_session_cm = MagicMock()
            mock_session_cm.__aenter__ = AsyncMock(return_value=mock_session)
            mock_session_cm.__aexit__ = AsyncMock(return_value=None)
            mock_driver.session.return_value = mock_session_cm

            await handler.initialize(connection_uri="bolt://localhost:7687")

            result = await handler.delete_node(node_id=123, detach=True)

            assert result.success is True
            assert result.relationships_deleted == 3

            await handler.shutdown()

    @pytest.mark.asyncio
    async def test_delete_node_not_initialized(self, handler: HandlerGraph) -> None:
        """Test delete_node fails when not initialized."""
        with pytest.raises(RuntimeHostError) as exc_info:
            await handler.delete_node(node_id=123)

        assert "not initialized" in str(exc_info.value).lower()


class TestHandlerGraphDeleteRelationship:
    """Test HandlerGraph delete_relationship operation."""

    @pytest.mark.asyncio
    async def test_delete_relationship_success(
        self, handler: HandlerGraph, mock_driver: MagicMock
    ) -> None:
        """Test successful relationship deletion."""
        with patch(
            "omnibase_infra.handlers.handler_graph.AsyncGraphDatabase"
        ) as mock_db:
            mock_db.driver.return_value = mock_driver

            mock_session = AsyncMock()
            mock_result = AsyncMock()

            mock_record = MagicMock()
            mock_record.__getitem__ = lambda self, key: {"deleted": 1}[key]

            mock_result.single = AsyncMock(return_value=mock_record)
            mock_result.consume = AsyncMock()

            mock_session.run = AsyncMock(return_value=mock_result)

            mock_session_cm = MagicMock()
            mock_session_cm.__aenter__ = AsyncMock(return_value=mock_session)
            mock_session_cm.__aexit__ = AsyncMock(return_value=None)
            mock_driver.session.return_value = mock_session_cm

            await handler.initialize(connection_uri="bolt://localhost:7687")

            result = await handler.delete_relationship(relationship_id=456)

            assert result.success is True
            assert result.relationships_deleted == 1
            assert result.execution_time_ms >= 0

            await handler.shutdown()

    @pytest.mark.asyncio
    async def test_delete_relationship_not_initialized(
        self, handler: HandlerGraph
    ) -> None:
        """Test delete_relationship fails when not initialized."""
        with pytest.raises(RuntimeHostError) as exc_info:
            await handler.delete_relationship(relationship_id=456)

        assert "not initialized" in str(exc_info.value).lower()


class TestHandlerGraphTraverse:
    """Test HandlerGraph traverse operation."""

    @pytest.mark.asyncio
    async def test_traverse_success(
        self, handler: HandlerGraph, mock_driver: MagicMock
    ) -> None:
        """Test successful graph traversal."""
        with patch(
            "omnibase_infra.handlers.handler_graph.AsyncGraphDatabase"
        ) as mock_db:
            mock_db.driver.return_value = mock_driver

            mock_session = AsyncMock()
            mock_result = AsyncMock()

            # Create mock node
            mock_node = MagicMock()
            mock_node.labels = frozenset(["Person"])
            mock_node.items = MagicMock(return_value=[("name", "Bob")])

            # Create mock relationship
            mock_rel = MagicMock()
            mock_rel.id = 789
            mock_rel.element_id = "5:abc:789"
            mock_rel.type = "KNOWS"
            mock_rel.items = MagicMock(return_value=[])
            mock_rel.start_node = MagicMock(element_id="4:abc:123")
            mock_rel.end_node = MagicMock(element_id="4:abc:124")

            records_data = [
                {
                    "n": mock_node,
                    "eid": "4:abc:124",
                    "nid": 124,
                    "rels": [mock_rel],
                    "path_ids": ["4:abc:123", "4:abc:124"],
                }
            ]
            mock_result.data = AsyncMock(return_value=records_data)
            mock_result.consume = AsyncMock()

            mock_session.run = AsyncMock(return_value=mock_result)

            mock_session_cm = MagicMock()
            mock_session_cm.__aenter__ = AsyncMock(return_value=mock_session)
            mock_session_cm.__aexit__ = AsyncMock(return_value=None)
            mock_driver.session.return_value = mock_session_cm

            await handler.initialize(connection_uri="bolt://localhost:7687")

            result = await handler.traverse(
                start_node_id=123,
                relationship_types=["KNOWS"],
                direction="outgoing",
                max_depth=2,
            )

            assert len(result.nodes) == 1
            assert len(result.relationships) == 1
            assert len(result.paths) == 1
            assert result.execution_time_ms >= 0

            await handler.shutdown()

    @pytest.mark.asyncio
    async def test_traverse_not_initialized(self, handler: HandlerGraph) -> None:
        """Test traverse fails when not initialized."""
        with pytest.raises(RuntimeHostError) as exc_info:
            await handler.traverse(
                start_node_id=123,
            )

        assert "not initialized" in str(exc_info.value).lower()


class TestHandlerGraphHealthCheck:
    """Test HandlerGraph health_check operation."""

    @pytest.mark.asyncio
    async def test_health_check_success(
        self, handler: HandlerGraph, mock_driver: MagicMock
    ) -> None:
        """Test successful health check."""
        with patch(
            "omnibase_infra.handlers.handler_graph.AsyncGraphDatabase"
        ) as mock_db:
            mock_db.driver.return_value = mock_driver

            mock_session = AsyncMock()
            mock_result = AsyncMock()
            mock_result.consume = AsyncMock()
            mock_session.run = AsyncMock(return_value=mock_result)

            mock_session_cm = MagicMock()
            mock_session_cm.__aenter__ = AsyncMock(return_value=mock_session)
            mock_session_cm.__aexit__ = AsyncMock(return_value=None)
            mock_driver.session.return_value = mock_session_cm

            await handler.initialize(connection_uri="bolt://localhost:7687")

            # Clear cache to force actual health check
            handler._cached_health = None
            handler._health_cache_time = 0.0

            result = await handler.health_check()

            assert result.healthy is True
            assert result.latency_ms >= 0

            await handler.shutdown()

    @pytest.mark.asyncio
    async def test_health_check_not_initialized(self, handler: HandlerGraph) -> None:
        """Test health check returns unhealthy when not initialized."""
        result = await handler.health_check()

        assert result.healthy is False
        assert result.latency_ms == 0

    @pytest.mark.asyncio
    async def test_health_check_connection_error(
        self, handler: HandlerGraph, mock_driver: MagicMock
    ) -> None:
        """Test health check handles connection errors."""
        with patch(
            "omnibase_infra.handlers.handler_graph.AsyncGraphDatabase"
        ) as mock_db:
            mock_db.driver.return_value = mock_driver

            mock_session = AsyncMock()
            mock_session.run = AsyncMock(side_effect=Exception("Connection lost"))

            mock_session_cm = MagicMock()
            mock_session_cm.__aenter__ = AsyncMock(return_value=mock_session)
            mock_session_cm.__aexit__ = AsyncMock(return_value=None)
            mock_driver.session.return_value = mock_session_cm

            await handler.initialize(connection_uri="bolt://localhost:7687")

            # Clear cache to force actual health check
            handler._cached_health = None
            handler._health_cache_time = 0.0

            result = await handler.health_check()

            assert result.healthy is False

            await handler.shutdown()


class TestHandlerGraphDescribe:
    """Test HandlerGraph describe operation."""

    @pytest.mark.asyncio
    async def test_describe_returns_metadata(self, handler: HandlerGraph) -> None:
        """Test describe returns handler metadata."""
        result = await handler.describe()

        assert result.handler_type == "graph_database"
        assert result.supports_transactions is True
        assert "cypher" in result.capabilities
        assert "transactions" in result.capabilities
        assert "node_crud" in result.capabilities
        assert "relationship_crud" in result.capabilities
        assert "traversal" in result.capabilities

    @pytest.mark.asyncio
    async def test_describe_detects_database_type(
        self, handler: HandlerGraph, mock_driver: MagicMock
    ) -> None:
        """Test describe detects database type from connection URI."""
        with patch(
            "omnibase_infra.handlers.handler_graph.AsyncGraphDatabase"
        ) as mock_db:
            mock_db.driver.return_value = mock_driver

            await handler.initialize(connection_uri="bolt://localhost:7687")

            result = await handler.describe()

            # Default is memgraph
            assert result.database_type == "memgraph"

            await handler.shutdown()


class TestHandlerGraphCircuitBreaker:
    """Test HandlerGraph circuit breaker behavior."""

    @pytest.mark.asyncio
    async def test_circuit_breaker_initialized(
        self, handler: HandlerGraph, mock_driver: MagicMock
    ) -> None:
        """Test circuit breaker is initialized after handler initialization."""
        with patch(
            "omnibase_infra.handlers.handler_graph.AsyncGraphDatabase"
        ) as mock_db:
            mock_db.driver.return_value = mock_driver

            await handler.initialize(connection_uri="bolt://localhost:7687")

            assert hasattr(handler, "_circuit_breaker_lock")
            assert hasattr(handler, "circuit_breaker_threshold")
            assert handler.circuit_breaker_threshold == 5
            assert handler.circuit_breaker_reset_timeout == 60.0
            assert handler.service_name == "graph"

            await handler.shutdown()

    @pytest.mark.asyncio
    async def test_circuit_breaker_records_failure(
        self, handler: HandlerGraph, mock_driver: MagicMock
    ) -> None:
        """Test circuit breaker records failure when query fails."""
        with patch(
            "omnibase_infra.handlers.handler_graph.AsyncGraphDatabase"
        ) as mock_db:
            mock_db.driver.return_value = mock_driver

            from neo4j.exceptions import Neo4jError

            mock_session = AsyncMock()
            mock_session.run = AsyncMock(side_effect=Neo4jError("Query failed"))
            mock_session_cm = MagicMock()
            mock_session_cm.__aenter__ = AsyncMock(return_value=mock_session)
            mock_session_cm.__aexit__ = AsyncMock(return_value=None)
            mock_driver.session.return_value = mock_session_cm

            await handler.initialize(connection_uri="bolt://localhost:7687")

            with pytest.raises(InfraConnectionError):
                await handler.execute_query(query="MATCH (n) RETURN n")

            assert handler._circuit_breaker_failures == 1

            await handler.shutdown()

    @pytest.mark.asyncio
    async def test_circuit_opens_after_threshold(
        self, handler: HandlerGraph, mock_driver: MagicMock
    ) -> None:
        """Test circuit breaker opens after failure threshold."""
        with patch(
            "omnibase_infra.handlers.handler_graph.AsyncGraphDatabase"
        ) as mock_db:
            mock_db.driver.return_value = mock_driver

            from neo4j.exceptions import Neo4jError

            mock_session = AsyncMock()
            mock_session.run = AsyncMock(side_effect=Neo4jError("Connection lost"))
            mock_session_cm = MagicMock()
            mock_session_cm.__aenter__ = AsyncMock(return_value=mock_session)
            mock_session_cm.__aexit__ = AsyncMock(return_value=None)
            mock_driver.session.return_value = mock_session_cm

            await handler.initialize(connection_uri="bolt://localhost:7687")

            # Cause 5 failures to trip the circuit breaker
            for _ in range(5):
                with pytest.raises(InfraConnectionError):
                    await handler.execute_query(query="MATCH (n) RETURN n")

            # Now the circuit should be open
            with pytest.raises(InfraUnavailableError) as exc_info:
                await handler.execute_query(query="MATCH (n) RETURN n")

            assert "circuit breaker is open" in str(exc_info.value).lower()

            await handler.shutdown()


class TestHandlerGraphLifecycle:
    """Test HandlerGraph lifecycle management."""

    @pytest.mark.asyncio
    async def test_execute_after_shutdown_raises_error(
        self, handler: HandlerGraph, mock_driver: MagicMock
    ) -> None:
        """Test operations after shutdown raise error."""
        with patch(
            "omnibase_infra.handlers.handler_graph.AsyncGraphDatabase"
        ) as mock_db:
            mock_db.driver.return_value = mock_driver

            await handler.initialize(connection_uri="bolt://localhost:7687")
            await handler.shutdown()

            with pytest.raises(RuntimeHostError) as exc_info:
                await handler.execute_query(query="MATCH (n) RETURN n")

            assert "not initialized" in str(exc_info.value).lower()

    @pytest.mark.asyncio
    async def test_reinitialize_after_shutdown(
        self, handler: HandlerGraph, mock_driver: MagicMock
    ) -> None:
        """Test handler can be reinitialized after shutdown."""
        with patch(
            "omnibase_infra.handlers.handler_graph.AsyncGraphDatabase"
        ) as mock_db:
            mock_db.driver.return_value = mock_driver

            await handler.initialize(connection_uri="bolt://localhost:7687")
            await handler.shutdown()

            assert handler._initialized is False

            await handler.initialize(connection_uri="bolt://localhost:7687")

            assert handler._initialized is True
            assert handler._driver is not None

            await handler.shutdown()


class TestHandlerGraphLogWarnings:
    """Test suite for log warning assertions."""

    HANDLER_MODULE = "omnibase_infra.handlers.handler_graph"

    def _setup_mock_session(self, mock_driver: MagicMock) -> None:
        """Set up mock session for normal operations."""
        mock_session = AsyncMock()
        mock_result = AsyncMock()
        mock_result.data = AsyncMock(return_value=[{"name": "Alice"}])

        mock_summary = MagicMock()
        mock_summary.query_type = "r"
        mock_summary.counters = MagicMock()
        mock_summary.counters.contains_updates = False
        mock_summary.counters.nodes_created = 0
        mock_summary.counters.nodes_deleted = 0
        mock_summary.counters.relationships_created = 0
        mock_summary.counters.relationships_deleted = 0
        mock_summary.counters.properties_set = 0
        mock_summary.counters.labels_added = 0
        mock_summary.counters.labels_removed = 0
        mock_result.consume = AsyncMock(return_value=mock_summary)

        mock_session.run = AsyncMock(return_value=mock_result)

        mock_session_cm = MagicMock()
        mock_session_cm.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session_cm.__aexit__ = AsyncMock(return_value=None)
        mock_driver.session.return_value = mock_session_cm

    @pytest.mark.asyncio
    async def test_no_unexpected_warnings_during_normal_operation(
        self,
        handler: HandlerGraph,
        mock_driver: MagicMock,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """Test that normal operations produce no unexpected warnings."""
        import logging

        self._setup_mock_session(mock_driver)

        with caplog.at_level(logging.WARNING):
            with patch(
                "omnibase_infra.handlers.handler_graph.AsyncGraphDatabase"
            ) as mock_db:
                mock_db.driver.return_value = mock_driver

                await handler.initialize(connection_uri="bolt://localhost:7687")

                await handler.execute_query(
                    query="MATCH (n) RETURN n.name as name",
                )

                await handler.shutdown()

        handler_warnings = filter_handler_warnings(caplog.records, self.HANDLER_MODULE)
        assert len(handler_warnings) == 0, (
            f"Unexpected warnings: {[w.message for w in handler_warnings]}"
        )


__all__: list[str] = [
    "TestHandlerGraphProperties",
    "TestHandlerGraphInitialization",
    "TestHandlerGraphShutdown",
    "TestHandlerGraphExecuteQuery",
    "TestHandlerGraphExecuteQueryBatch",
    "TestHandlerGraphCreateNode",
    "TestHandlerGraphCreateRelationship",
    "TestHandlerGraphDeleteNode",
    "TestHandlerGraphDeleteRelationship",
    "TestHandlerGraphTraverse",
    "TestHandlerGraphHealthCheck",
    "TestHandlerGraphDescribe",
    "TestHandlerGraphCircuitBreaker",
    "TestHandlerGraphLifecycle",
    "TestHandlerGraphLogWarnings",
]
