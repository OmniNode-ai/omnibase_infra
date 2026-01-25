# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Unit tests for HandlerIntent wrapping HandlerGraph for intent-specific operations.

These tests verify the HandlerIntent implementation using mocked HandlerGraph
to validate intent storage and query behavior without requiring actual graph
database infrastructure.

Test Coverage:
    - Initialization: Valid config, missing/invalid graph_handler
    - Store Operation: Success, type conversion, primitive preservation, not initialized
    - Query Session: Success, empty results, missing session_id, not initialized
    - Query Distribution: Success, empty database, not initialized
    - Execute Routing: Route to correct operation, unsupported/missing operation, invalid payload
    - Shutdown: State clearing
    - Describe: Metadata return
"""

from __future__ import annotations

from datetime import datetime
from typing import TYPE_CHECKING
from unittest.mock import AsyncMock, MagicMock
from uuid import UUID, uuid4

import pytest

from omnibase_core.models.graph import (
    ModelGraphDatabaseNode,
    ModelGraphQueryCounters,
    ModelGraphQueryResult,
    ModelGraphQuerySummary,
)
from omnibase_infra.enums import EnumInfraTransportType
from omnibase_infra.errors import RuntimeHostError
from omnibase_infra.handlers.handler_graph import HandlerGraph
from omnibase_infra.handlers.handler_intent import HANDLER_ID_INTENT, HandlerIntent

if TYPE_CHECKING:
    from collections.abc import Generator


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def mock_graph_handler() -> MagicMock:
    """Create mock HandlerGraph with all required async methods."""
    mock = MagicMock(spec=HandlerGraph)
    mock.create_node = AsyncMock()
    mock.execute_query = AsyncMock()
    return mock


@pytest.fixture
def handler(mock_container: MagicMock) -> HandlerIntent:
    """Create a fresh HandlerIntent instance with mock container."""
    return HandlerIntent(container=mock_container)


@pytest.fixture
async def initialized_handler(
    handler: HandlerIntent,
    mock_graph_handler: MagicMock,
) -> HandlerIntent:
    """Create an initialized HandlerIntent with mock graph handler."""
    await handler.initialize({"graph_handler": mock_graph_handler})
    return handler


# =============================================================================
# Initialization Tests
# =============================================================================


class TestHandlerIntentInitialization:
    """Test HandlerIntent initialization and configuration."""

    @pytest.mark.asyncio
    async def test_initialize_success(
        self,
        handler: HandlerIntent,
        mock_graph_handler: MagicMock,
    ) -> None:
        """Test handler initializes with valid graph_handler in config."""
        await handler.initialize({"graph_handler": mock_graph_handler})

        assert handler._initialized is True
        assert handler._graph_handler is mock_graph_handler

    @pytest.mark.asyncio
    async def test_initialize_missing_graph_handler(
        self,
        handler: HandlerIntent,
    ) -> None:
        """Test raises RuntimeHostError when graph_handler missing from config."""
        with pytest.raises(RuntimeHostError) as exc_info:
            await handler.initialize({})

        assert "graph_handler" in str(exc_info.value).lower()
        assert handler._initialized is False

    @pytest.mark.asyncio
    async def test_initialize_invalid_graph_handler(
        self,
        handler: HandlerIntent,
    ) -> None:
        """Test raises RuntimeHostError when graph_handler is wrong type."""
        with pytest.raises(RuntimeHostError) as exc_info:
            await handler.initialize({"graph_handler": "not_a_handler"})

        assert "graph_handler" in str(exc_info.value).lower()
        assert handler._initialized is False

    @pytest.mark.asyncio
    async def test_initialize_graph_handler_is_none(
        self,
        handler: HandlerIntent,
    ) -> None:
        """Test raises RuntimeHostError when graph_handler is None."""
        with pytest.raises(RuntimeHostError) as exc_info:
            await handler.initialize({"graph_handler": None})

        assert "graph_handler" in str(exc_info.value).lower()


# =============================================================================
# Store Operation Tests
# =============================================================================


class TestHandlerIntentStoreOperation:
    """Test intent.store operation."""

    @pytest.mark.asyncio
    async def test_store_intent_success(
        self,
        initialized_handler: HandlerIntent,
        mock_graph_handler: MagicMock,
    ) -> None:
        """Test successfully stores intent with all properties."""
        # Configure mock to return a node
        mock_node = ModelGraphDatabaseNode(
            id="123",
            element_id="4:abc:123",
            labels=["Intent"],
            properties={
                "intent_type": "test_intent",
                "session_id": "session-abc",
                "correlation_id": "corr-123",
            },
        )
        mock_graph_handler.create_node.return_value = mock_node

        envelope = {
            "operation": "intent.store",
            "payload": {
                "intent_type": "test_intent",
                "session_id": "session-abc",
                "data": {"key": "value"},
            },
            "correlation_id": str(uuid4()),
        }

        result = await initialized_handler.execute(envelope)

        assert result["success"] is True
        assert result["data"]["node_id"] == "123"
        assert result["data"]["element_id"] == "4:abc:123"
        assert "Intent" in result["data"]["labels"]
        mock_graph_handler.create_node.assert_called_once()

        # Verify create_node was called with correct labels
        call_kwargs = mock_graph_handler.create_node.call_args
        assert call_kwargs.kwargs["labels"] == ["Intent"]

    @pytest.mark.asyncio
    async def test_store_intent_converts_complex_types_to_string(
        self,
        initialized_handler: HandlerIntent,
        mock_graph_handler: MagicMock,
    ) -> None:
        """Test UUID and datetime are converted to strings for graph storage."""
        mock_node = ModelGraphDatabaseNode(
            id="123",
            element_id="4:abc:123",
            labels=["Intent"],
            properties={},
        )
        mock_graph_handler.create_node.return_value = mock_node

        test_uuid = uuid4()
        test_datetime = datetime.now()

        envelope = {
            "operation": "intent.store",
            "payload": {
                "intent_type": "test_intent",
                "uuid_field": test_uuid,
                "datetime_field": test_datetime,
                "list_field": [1, 2, 3],
                "dict_field": {"nested": "value"},
            },
        }

        await initialized_handler.execute(envelope)

        # Verify create_node was called with properties
        call_kwargs = mock_graph_handler.create_node.call_args
        properties = call_kwargs.kwargs["properties"]

        # Complex types should be converted to strings
        assert properties["uuid_field"] == str(test_uuid)
        assert properties["datetime_field"] == str(test_datetime)
        assert properties["list_field"] == str([1, 2, 3])
        assert properties["dict_field"] == str({"nested": "value"})

    @pytest.mark.asyncio
    async def test_store_intent_preserves_primitive_types(
        self,
        initialized_handler: HandlerIntent,
        mock_graph_handler: MagicMock,
    ) -> None:
        """Test str, int, float, bool are preserved in graph storage."""
        mock_node = ModelGraphDatabaseNode(
            id="123",
            element_id="4:abc:123",
            labels=["Intent"],
            properties={},
        )
        mock_graph_handler.create_node.return_value = mock_node

        envelope = {
            "operation": "intent.store",
            "payload": {
                "string_field": "test_string",
                "int_field": 42,
                "float_field": 3.14,
                "bool_field": True,
                "none_field": None,
            },
        }

        await initialized_handler.execute(envelope)

        # Verify create_node was called with preserved primitive types
        call_kwargs = mock_graph_handler.create_node.call_args
        properties = call_kwargs.kwargs["properties"]

        assert properties["string_field"] == "test_string"
        assert properties["int_field"] == 42
        assert properties["float_field"] == 3.14
        assert properties["bool_field"] is True
        assert properties["none_field"] is None

    @pytest.mark.asyncio
    async def test_store_intent_not_initialized(
        self,
        handler: HandlerIntent,
    ) -> None:
        """Test raises RuntimeHostError if not initialized."""
        envelope = {
            "operation": "intent.store",
            "payload": {"intent_type": "test"},
        }

        with pytest.raises(RuntimeHostError) as exc_info:
            await handler.execute(envelope)

        assert "not initialized" in str(exc_info.value).lower()


# =============================================================================
# Query Session Tests
# =============================================================================


class TestHandlerIntentQuerySession:
    """Test intent.query_session operation."""

    @pytest.mark.asyncio
    async def test_query_session_success(
        self,
        initialized_handler: HandlerIntent,
        mock_graph_handler: MagicMock,
    ) -> None:
        """Test returns matching intents for session_id."""
        # Configure mock to return query results
        mock_result = ModelGraphQueryResult(
            records=[
                {
                    "i": {"intent_type": "type1", "session_id": "session-abc"},
                    "eid": "4:abc:123",
                    "nid": 123,
                },
                {
                    "i": {"intent_type": "type2", "session_id": "session-abc"},
                    "eid": "4:abc:124",
                    "nid": 124,
                },
            ],
            summary=ModelGraphQuerySummary(
                query_type="r",
                database="memgraph",
                contains_updates=False,
            ),
            counters=ModelGraphQueryCounters(),
            execution_time_ms=10.0,
        )
        mock_graph_handler.execute_query.return_value = mock_result

        envelope = {
            "operation": "intent.query_session",
            "payload": {"session_id": "session-abc"},
        }

        result = await initialized_handler.execute(envelope)

        assert result["success"] is True
        assert result["data"]["session_id"] == "session-abc"
        assert result["data"]["count"] == 2
        assert len(result["data"]["intents"]) == 2
        mock_graph_handler.execute_query.assert_called_once()

    @pytest.mark.asyncio
    async def test_query_session_empty_results(
        self,
        initialized_handler: HandlerIntent,
        mock_graph_handler: MagicMock,
    ) -> None:
        """Test returns empty list when no matches."""
        mock_result = ModelGraphQueryResult(
            records=[],
            summary=ModelGraphQuerySummary(
                query_type="r",
                database="memgraph",
                contains_updates=False,
            ),
            counters=ModelGraphQueryCounters(),
            execution_time_ms=5.0,
        )
        mock_graph_handler.execute_query.return_value = mock_result

        envelope = {
            "operation": "intent.query_session",
            "payload": {"session_id": "nonexistent-session"},
        }

        result = await initialized_handler.execute(envelope)

        assert result["success"] is True
        assert result["data"]["count"] == 0
        assert len(result["data"]["intents"]) == 0

    @pytest.mark.asyncio
    async def test_query_session_missing_session_id(
        self,
        initialized_handler: HandlerIntent,
    ) -> None:
        """Test raises RuntimeHostError when session_id missing."""
        envelope = {
            "operation": "intent.query_session",
            "payload": {},
        }

        with pytest.raises(RuntimeHostError) as exc_info:
            await initialized_handler.execute(envelope)

        assert "session_id" in str(exc_info.value).lower()

    @pytest.mark.asyncio
    async def test_query_session_not_initialized(
        self,
        handler: HandlerIntent,
    ) -> None:
        """Test raises RuntimeHostError if not initialized."""
        envelope = {
            "operation": "intent.query_session",
            "payload": {"session_id": "test-session"},
        }

        with pytest.raises(RuntimeHostError) as exc_info:
            await handler.execute(envelope)

        assert "not initialized" in str(exc_info.value).lower()


# =============================================================================
# Query Distribution Tests
# =============================================================================


class TestHandlerIntentQueryDistribution:
    """Test intent.query_distribution operation."""

    @pytest.mark.asyncio
    async def test_query_distribution_success(
        self,
        initialized_handler: HandlerIntent,
        mock_graph_handler: MagicMock,
    ) -> None:
        """Test returns total count and distribution by intent_type."""
        # Configure mock for count query
        count_result = ModelGraphQueryResult(
            records=[{"total": 100}],
            summary=ModelGraphQuerySummary(
                query_type="r",
                database="memgraph",
                contains_updates=False,
            ),
            counters=ModelGraphQueryCounters(),
            execution_time_ms=5.0,
        )

        # Configure mock for distribution query
        distribution_result = ModelGraphQueryResult(
            records=[
                {"intent_type": "type_a", "count": 60},
                {"intent_type": "type_b", "count": 30},
                {"intent_type": "type_c", "count": 10},
            ],
            summary=ModelGraphQuerySummary(
                query_type="r",
                database="memgraph",
                contains_updates=False,
            ),
            counters=ModelGraphQueryCounters(),
            execution_time_ms=5.0,
        )

        mock_graph_handler.execute_query.side_effect = [
            count_result,
            distribution_result,
        ]

        envelope = {
            "operation": "intent.query_distribution",
            "payload": {},
        }

        result = await initialized_handler.execute(envelope)

        assert result["success"] is True
        assert result["data"]["total_count"] == 100
        assert result["data"]["distribution"]["type_a"] == 60
        assert result["data"]["distribution"]["type_b"] == 30
        assert result["data"]["distribution"]["type_c"] == 10
        assert mock_graph_handler.execute_query.call_count == 2

    @pytest.mark.asyncio
    async def test_query_distribution_empty_database(
        self,
        initialized_handler: HandlerIntent,
        mock_graph_handler: MagicMock,
    ) -> None:
        """Test returns zero counts when no intents exist."""
        count_result = ModelGraphQueryResult(
            records=[{"total": 0}],
            summary=ModelGraphQuerySummary(
                query_type="r",
                database="memgraph",
                contains_updates=False,
            ),
            counters=ModelGraphQueryCounters(),
            execution_time_ms=5.0,
        )

        distribution_result = ModelGraphQueryResult(
            records=[],
            summary=ModelGraphQuerySummary(
                query_type="r",
                database="memgraph",
                contains_updates=False,
            ),
            counters=ModelGraphQueryCounters(),
            execution_time_ms=5.0,
        )

        mock_graph_handler.execute_query.side_effect = [
            count_result,
            distribution_result,
        ]

        envelope = {
            "operation": "intent.query_distribution",
            "payload": {},
        }

        result = await initialized_handler.execute(envelope)

        assert result["success"] is True
        assert result["data"]["total_count"] == 0
        assert result["data"]["distribution"] == {}

    @pytest.mark.asyncio
    async def test_query_distribution_not_initialized(
        self,
        handler: HandlerIntent,
    ) -> None:
        """Test raises RuntimeHostError if not initialized."""
        envelope = {
            "operation": "intent.query_distribution",
            "payload": {},
        }

        with pytest.raises(RuntimeHostError) as exc_info:
            await handler.execute(envelope)

        assert "not initialized" in str(exc_info.value).lower()


# =============================================================================
# Execute Routing Tests
# =============================================================================


class TestHandlerIntentExecuteRouting:
    """Test execute method routing to appropriate handlers."""

    @pytest.mark.asyncio
    async def test_execute_routes_to_store(
        self,
        initialized_handler: HandlerIntent,
        mock_graph_handler: MagicMock,
    ) -> None:
        """Test routes 'intent.store' to _store_intent."""
        mock_node = ModelGraphDatabaseNode(
            id="123",
            element_id="4:abc:123",
            labels=["Intent"],
            properties={},
        )
        mock_graph_handler.create_node.return_value = mock_node

        envelope = {
            "operation": "intent.store",
            "payload": {"intent_type": "test"},
        }

        result = await initialized_handler.execute(envelope)

        assert result["success"] is True
        mock_graph_handler.create_node.assert_called_once()

    @pytest.mark.asyncio
    async def test_execute_routes_to_query_session(
        self,
        initialized_handler: HandlerIntent,
        mock_graph_handler: MagicMock,
    ) -> None:
        """Test routes 'intent.query_session' to _query_session."""
        mock_result = ModelGraphQueryResult(
            records=[],
            summary=ModelGraphQuerySummary(
                query_type="r",
                database="memgraph",
                contains_updates=False,
            ),
            counters=ModelGraphQueryCounters(),
            execution_time_ms=5.0,
        )
        mock_graph_handler.execute_query.return_value = mock_result

        envelope = {
            "operation": "intent.query_session",
            "payload": {"session_id": "test-session"},
        }

        result = await initialized_handler.execute(envelope)

        assert result["success"] is True
        assert "intents" in result["data"]
        mock_graph_handler.execute_query.assert_called_once()

    @pytest.mark.asyncio
    async def test_execute_routes_to_query_distribution(
        self,
        initialized_handler: HandlerIntent,
        mock_graph_handler: MagicMock,
    ) -> None:
        """Test routes 'intent.query_distribution' to _query_distribution."""
        count_result = ModelGraphQueryResult(
            records=[{"total": 0}],
            summary=ModelGraphQuerySummary(
                query_type="r",
                database="memgraph",
                contains_updates=False,
            ),
            counters=ModelGraphQueryCounters(),
            execution_time_ms=5.0,
        )
        distribution_result = ModelGraphQueryResult(
            records=[],
            summary=ModelGraphQuerySummary(
                query_type="r",
                database="memgraph",
                contains_updates=False,
            ),
            counters=ModelGraphQueryCounters(),
            execution_time_ms=5.0,
        )
        mock_graph_handler.execute_query.side_effect = [
            count_result,
            distribution_result,
        ]

        envelope = {
            "operation": "intent.query_distribution",
            "payload": {},
        }

        result = await initialized_handler.execute(envelope)

        assert result["success"] is True
        assert "distribution" in result["data"]

    @pytest.mark.asyncio
    async def test_execute_unsupported_operation(
        self,
        initialized_handler: HandlerIntent,
    ) -> None:
        """Test raises RuntimeHostError for unknown operation."""
        envelope = {
            "operation": "intent.unknown_operation",
            "payload": {},
        }

        with pytest.raises(RuntimeHostError) as exc_info:
            await initialized_handler.execute(envelope)

        error_msg = str(exc_info.value).lower()
        assert "not supported" in error_msg or "unknown_operation" in error_msg

    @pytest.mark.asyncio
    async def test_execute_missing_operation(
        self,
        initialized_handler: HandlerIntent,
    ) -> None:
        """Test raises RuntimeHostError when operation missing."""
        envelope = {
            "payload": {"data": "test"},
        }

        with pytest.raises(RuntimeHostError) as exc_info:
            await initialized_handler.execute(envelope)

        assert "operation" in str(exc_info.value).lower()

    @pytest.mark.asyncio
    async def test_execute_invalid_operation_type(
        self,
        initialized_handler: HandlerIntent,
    ) -> None:
        """Test raises RuntimeHostError when operation is not a string."""
        envelope = {
            "operation": 123,  # Not a string
            "payload": {},
        }

        with pytest.raises(RuntimeHostError) as exc_info:
            await initialized_handler.execute(envelope)

        assert "operation" in str(exc_info.value).lower()

    @pytest.mark.asyncio
    async def test_execute_invalid_payload(
        self,
        initialized_handler: HandlerIntent,
    ) -> None:
        """Test raises RuntimeHostError when payload not dict."""
        envelope = {
            "operation": "intent.store",
            "payload": "not_a_dict",
        }

        with pytest.raises(RuntimeHostError) as exc_info:
            await initialized_handler.execute(envelope)

        assert "payload" in str(exc_info.value).lower()

    @pytest.mark.asyncio
    async def test_execute_missing_payload(
        self,
        initialized_handler: HandlerIntent,
    ) -> None:
        """Test raises RuntimeHostError when payload is missing."""
        envelope = {
            "operation": "intent.store",
        }

        with pytest.raises(RuntimeHostError) as exc_info:
            await initialized_handler.execute(envelope)

        assert "payload" in str(exc_info.value).lower()


# =============================================================================
# Shutdown Tests
# =============================================================================


class TestHandlerIntentShutdown:
    """Test HandlerIntent shutdown functionality."""

    @pytest.mark.asyncio
    async def test_shutdown_clears_state(
        self,
        initialized_handler: HandlerIntent,
    ) -> None:
        """Test graph handler reference cleared, initialized=False."""
        assert initialized_handler._initialized is True
        assert initialized_handler._graph_handler is not None

        await initialized_handler.shutdown()

        assert initialized_handler._initialized is False
        assert initialized_handler._graph_handler is None

    @pytest.mark.asyncio
    async def test_shutdown_without_initialize_is_safe(
        self,
        handler: HandlerIntent,
    ) -> None:
        """Test shutdown without initialize doesn't raise error."""
        await handler.shutdown()

        assert handler._initialized is False
        assert handler._graph_handler is None

    @pytest.mark.asyncio
    async def test_multiple_shutdown_calls_safe(
        self,
        initialized_handler: HandlerIntent,
    ) -> None:
        """Test multiple shutdown calls are safe (idempotent)."""
        await initialized_handler.shutdown()
        await initialized_handler.shutdown()  # Should not raise

        assert initialized_handler._initialized is False


# =============================================================================
# Describe Tests
# =============================================================================


class TestHandlerIntentDescribe:
    """Test HandlerIntent describe operation."""

    def test_describe_returns_metadata(
        self,
        handler: HandlerIntent,
    ) -> None:
        """Test returns handler_id, operations, version."""
        result = handler.describe()

        assert result["handler_id"] == HANDLER_ID_INTENT
        assert result["handler_type"] == "intent_handler"
        assert "supported_operations" in result
        assert "intent.store" in result["supported_operations"]
        assert "intent.query_session" in result["supported_operations"]
        assert "intent.query_distribution" in result["supported_operations"]
        assert "version" in result

    def test_describe_shows_initialized_status(
        self,
        handler: HandlerIntent,
    ) -> None:
        """Test describe shows correct initialized status."""
        # Not initialized
        result = handler.describe()
        assert result["initialized"] is False

    @pytest.mark.asyncio
    async def test_describe_shows_initialized_true_after_init(
        self,
        initialized_handler: HandlerIntent,
    ) -> None:
        """Test describe shows initialized=True after initialization."""
        result = initialized_handler.describe()
        assert result["initialized"] is True


# =============================================================================
# Error Context Tests
# =============================================================================


class TestHandlerIntentErrorContext:
    """Test error context contains correct information."""

    @pytest.mark.asyncio
    async def test_error_context_has_transport_type(
        self,
        handler: HandlerIntent,
    ) -> None:
        """Test error context contains correct transport_type."""
        envelope = {
            "operation": "intent.store",
            "payload": {},
        }

        with pytest.raises(RuntimeHostError) as exc_info:
            await handler.execute(envelope)

        # Verify error has context with transport_type
        # RuntimeHostError stores ModelInfraErrorContext fields in additional_context
        error = exc_info.value
        assert hasattr(error, "context")
        if error.context:
            # Context is nested under additional_context key
            additional_context = error.context.get("additional_context", {})
            assert (
                additional_context.get("transport_type") == EnumInfraTransportType.GRAPH
            )

    @pytest.mark.asyncio
    async def test_error_context_has_operation(
        self,
        initialized_handler: HandlerIntent,
    ) -> None:
        """Test error context contains operation name."""
        envelope = {
            "operation": "intent.query_session",
            "payload": {},  # Missing session_id
        }

        with pytest.raises(RuntimeHostError) as exc_info:
            await initialized_handler.execute(envelope)

        error = exc_info.value
        if error.context:
            # Context is nested under additional_context key
            additional_context = error.context.get("additional_context", {})
            assert additional_context.get("operation") == "intent.query_session"


# =============================================================================
# Module Exports
# =============================================================================


__all__: list[str] = [
    "TestHandlerIntentInitialization",
    "TestHandlerIntentStoreOperation",
    "TestHandlerIntentQuerySession",
    "TestHandlerIntentQueryDistribution",
    "TestHandlerIntentExecuteRouting",
    "TestHandlerIntentShutdown",
    "TestHandlerIntentDescribe",
    "TestHandlerIntentErrorContext",
]
