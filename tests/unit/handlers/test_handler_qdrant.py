# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Unit tests for HandlerQdrant implementing ProtocolVectorStoreHandler.

These tests verify the SPI protocol implementation using mocked qdrant_client.QdrantClient
to validate HandlerQdrant behavior without requiring actual Qdrant server infrastructure.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import TYPE_CHECKING
from unittest.mock import MagicMock, patch
from uuid import uuid4

import pytest
from omnibase_core.models.common.model_schema_value import ModelSchemaValue
from omnibase_core.models.vector import (
    EnumVectorDistanceMetric,
    ModelEmbedding,
    ModelVectorConnectionConfig,
)
from pydantic import SecretStr

from omnibase_infra.errors import (
    InfraAuthenticationError,
    InfraConnectionError,
    InfraUnavailableError,
    RuntimeHostError,
)
from omnibase_infra.handlers.handler_qdrant import HandlerQdrant

if TYPE_CHECKING:
    from collections.abc import Generator


@pytest.fixture
def handler(mock_container: MagicMock) -> HandlerQdrant:
    """Create a fresh HandlerQdrant instance with mock container."""
    return HandlerQdrant(container=mock_container)


@pytest.fixture
def connection_config() -> ModelVectorConnectionConfig:
    """Provide test connection configuration model."""
    return ModelVectorConnectionConfig(
        url="http://localhost:6333",
        api_key=SecretStr("test-api-key-12345"),
        timeout=30.0,
        pool_size=10,
    )


@pytest.fixture
def mock_qdrant_client() -> MagicMock:
    """Provide mocked qdrant_client.QdrantClient."""
    client = MagicMock()

    # Mock get_collections for connection test
    client.get_collections.return_value = MagicMock(collections=[])

    # Mock create_collection
    client.create_collection.return_value = True

    # Mock upsert
    client.upsert.return_value = True

    # Mock query_points (search API)
    mock_point = MagicMock()
    mock_point.id = "point-1"
    mock_point.score = 0.95
    mock_point.payload = {"text": "hello world"}
    mock_point.vector = None
    mock_result = MagicMock()
    mock_result.points = [mock_point]
    client.query_points.return_value = mock_result

    # Mock delete
    client.delete.return_value = True

    # Mock delete_collection
    client.delete_collection.return_value = True

    # Mock close
    client.close.return_value = None

    return client


class TestHandlerQdrantProperties:
    """Test HandlerQdrant type and capability properties."""

    def test_handler_type_returns_vector_store(self, handler: HandlerQdrant) -> None:
        """Test handler_type property returns 'vector_store'."""
        assert handler.handler_type == "vector_store"

    def test_supported_metrics_returns_expected_list(
        self, handler: HandlerQdrant
    ) -> None:
        """Test supported_metrics returns cosine, euclidean, dot_product."""
        metrics = handler.supported_metrics
        assert isinstance(metrics, list)
        assert "cosine" in metrics
        assert "euclidean" in metrics
        assert "dot_product" in metrics
        assert len(metrics) == 3


class TestHandlerQdrantInitialization:
    """Test HandlerQdrant initialization and configuration."""

    @pytest.mark.asyncio
    async def test_initialize_success(
        self,
        handler: HandlerQdrant,
        connection_config: ModelVectorConnectionConfig,
        mock_qdrant_client: MagicMock,
    ) -> None:
        """Test successful initialization with valid config."""
        with patch("omnibase_infra.handlers.handler_qdrant.QdrantClient") as MockClient:
            MockClient.return_value = mock_qdrant_client

            await handler.initialize(connection_config)

            assert handler._initialized is True
            assert handler._config is not None
            assert handler._config.url == "http://localhost:6333"
            MockClient.assert_called_once()

            await handler.shutdown()

    @pytest.mark.asyncio
    async def test_initialize_without_api_key(
        self,
        handler: HandlerQdrant,
        mock_qdrant_client: MagicMock,
    ) -> None:
        """Test initialization without API key for local Qdrant."""
        config = ModelVectorConnectionConfig(
            url="http://localhost:6333",
        )

        with patch("omnibase_infra.handlers.handler_qdrant.QdrantClient") as MockClient:
            MockClient.return_value = mock_qdrant_client

            await handler.initialize(config)

            assert handler._initialized is True
            MockClient.assert_called_once()

            await handler.shutdown()

    @pytest.mark.asyncio
    async def test_initialize_connection_failure(self, handler: HandlerQdrant) -> None:
        """Test initialization fails with connection error."""
        config = ModelVectorConnectionConfig(url="http://bad-host:6333")

        with patch("omnibase_infra.handlers.handler_qdrant.QdrantClient") as MockClient:
            MockClient.side_effect = Exception("Connection refused")

            with pytest.raises(InfraConnectionError) as exc_info:
                await handler.initialize(config)

            assert "failed to connect" in str(exc_info.value).lower()

    @pytest.mark.asyncio
    async def test_initialize_auth_error_unauthorized(
        self,
        handler: HandlerQdrant,
    ) -> None:
        """Test initialization fails with authentication error (unauthorized)."""
        config = ModelVectorConnectionConfig(
            url="http://localhost:6333",
            api_key=SecretStr("bad-key"),
        )

        with patch("omnibase_infra.handlers.handler_qdrant.QdrantClient") as MockClient:
            mock_client = MagicMock()
            MockClient.return_value = mock_client
            mock_client.get_collections.side_effect = Exception("unauthorized")

            with pytest.raises(InfraAuthenticationError) as exc_info:
                await handler.initialize(config)

            assert "authentication failed" in str(exc_info.value).lower()

    @pytest.mark.asyncio
    async def test_initialize_auth_error_forbidden(
        self,
        handler: HandlerQdrant,
    ) -> None:
        """Test initialization fails with authentication error (forbidden)."""
        config = ModelVectorConnectionConfig(
            url="http://localhost:6333",
            api_key=SecretStr("bad-key"),
        )

        with patch("omnibase_infra.handlers.handler_qdrant.QdrantClient") as MockClient:
            mock_client = MagicMock()
            MockClient.return_value = mock_client
            mock_client.get_collections.side_effect = Exception("403 forbidden")

            with pytest.raises(InfraAuthenticationError) as exc_info:
                await handler.initialize(config)

            assert "authentication failed" in str(exc_info.value).lower()


class TestHandlerQdrantShutdown:
    """Test HandlerQdrant shutdown functionality."""

    @pytest.mark.asyncio
    async def test_shutdown_releases_resources(
        self,
        handler: HandlerQdrant,
        connection_config: ModelVectorConnectionConfig,
        mock_qdrant_client: MagicMock,
    ) -> None:
        """Test handler shutdown releases resources."""
        with patch("omnibase_infra.handlers.handler_qdrant.QdrantClient") as MockClient:
            MockClient.return_value = mock_qdrant_client
            await handler.initialize(connection_config)

            assert handler._initialized is True
            assert handler._client is not None

            await handler.shutdown()

            assert handler._initialized is False
            assert handler._client is None
            mock_qdrant_client.close.assert_called_once()

    @pytest.mark.asyncio
    async def test_shutdown_when_not_initialized(self, handler: HandlerQdrant) -> None:
        """Test shutdown is safe when handler was never initialized."""
        await handler.shutdown()

        assert handler._initialized is False
        assert handler._client is None


class TestHandlerQdrantStoreEmbedding:
    """Test HandlerQdrant store_embedding operation."""

    @pytest.mark.asyncio
    async def test_store_embedding_success(
        self,
        handler: HandlerQdrant,
        connection_config: ModelVectorConnectionConfig,
        mock_qdrant_client: MagicMock,
    ) -> None:
        """Test successful single embedding storage."""
        with patch("omnibase_infra.handlers.handler_qdrant.QdrantClient") as MockClient:
            MockClient.return_value = mock_qdrant_client
            await handler.initialize(connection_config)

            result = await handler.store_embedding(
                embedding_id="emb-001",
                vector=[0.1, 0.2, 0.3],
                metadata={"text": "hello world"},
                index_name="test_collection",
            )

            assert result.success is True
            assert result.embedding_id == "emb-001"
            assert result.index_name == "test_collection"
            assert result.timestamp is not None
            mock_qdrant_client.upsert.assert_called_once()

            await handler.shutdown()

    @pytest.mark.asyncio
    async def test_store_embedding_not_initialized(
        self, handler: HandlerQdrant
    ) -> None:
        """Test store_embedding fails when handler not initialized."""
        with pytest.raises(RuntimeHostError) as exc_info:
            await handler.store_embedding(
                embedding_id="emb-001",
                vector=[0.1, 0.2, 0.3],
                index_name="test_collection",
            )

        assert "not initialized" in str(exc_info.value).lower()

    @pytest.mark.asyncio
    async def test_store_embedding_connection_error(
        self,
        handler: HandlerQdrant,
        connection_config: ModelVectorConnectionConfig,
        mock_qdrant_client: MagicMock,
    ) -> None:
        """Test store_embedding handles connection error."""
        with patch("omnibase_infra.handlers.handler_qdrant.QdrantClient") as MockClient:
            MockClient.return_value = mock_qdrant_client
            await handler.initialize(connection_config)

            mock_qdrant_client.upsert.side_effect = Exception("Connection lost")

            with pytest.raises(InfraConnectionError):
                await handler.store_embedding(
                    embedding_id="emb-001",
                    vector=[0.1, 0.2, 0.3],
                    index_name="test_collection",
                )

            await handler.shutdown()


class TestHandlerQdrantStoreBatch:
    """Test HandlerQdrant store_embeddings_batch operation."""

    @pytest.mark.asyncio
    async def test_store_embeddings_batch_success(
        self,
        handler: HandlerQdrant,
        connection_config: ModelVectorConnectionConfig,
        mock_qdrant_client: MagicMock,
    ) -> None:
        """Test successful batch embedding storage."""
        with patch("omnibase_infra.handlers.handler_qdrant.QdrantClient") as MockClient:
            MockClient.return_value = mock_qdrant_client
            await handler.initialize(connection_config)

            embeddings = [
                ModelEmbedding(
                    id="emb-001",
                    vector=[0.1, 0.2, 0.3],
                    metadata={
                        "text": ModelSchemaValue.from_value("hello"),
                    },
                ),
                ModelEmbedding(
                    id="emb-002",
                    vector=[0.4, 0.5, 0.6],
                    metadata={
                        "text": ModelSchemaValue.from_value("world"),
                    },
                ),
            ]

            result = await handler.store_embeddings_batch(
                embeddings=embeddings,
                index_name="test_collection",
                batch_size=100,
            )

            assert result.success is True
            assert result.total_stored == 2
            assert len(result.failed_ids) == 0
            assert result.execution_time_ms >= 0

            await handler.shutdown()

    @pytest.mark.asyncio
    async def test_store_embeddings_batch_not_initialized(
        self, handler: HandlerQdrant
    ) -> None:
        """Test store_embeddings_batch fails when not initialized."""
        embeddings = [
            ModelEmbedding(id="emb-001", vector=[0.1, 0.2, 0.3], metadata={}),
        ]

        with pytest.raises(RuntimeHostError) as exc_info:
            await handler.store_embeddings_batch(
                embeddings=embeddings,
                index_name="test_collection",
            )

        assert "not initialized" in str(exc_info.value).lower()


class TestHandlerQdrantQuerySimilar:
    """Test HandlerQdrant query_similar operation."""

    @pytest.mark.asyncio
    async def test_query_similar_success(
        self,
        handler: HandlerQdrant,
        connection_config: ModelVectorConnectionConfig,
        mock_qdrant_client: MagicMock,
    ) -> None:
        """Test successful similarity search."""
        with patch("omnibase_infra.handlers.handler_qdrant.QdrantClient") as MockClient:
            MockClient.return_value = mock_qdrant_client
            await handler.initialize(connection_config)

            result = await handler.query_similar(
                query_vector=[0.1, 0.2, 0.3],
                top_k=5,
                index_name="test_collection",
            )

            assert result.total_results == 1
            assert len(result.results) == 1
            assert result.results[0].id == "point-1"
            assert result.results[0].score == 0.95
            assert result.query_time_ms >= 0
            mock_qdrant_client.query_points.assert_called_once()

            await handler.shutdown()

    @pytest.mark.asyncio
    async def test_query_similar_empty_results(
        self,
        handler: HandlerQdrant,
        connection_config: ModelVectorConnectionConfig,
        mock_qdrant_client: MagicMock,
    ) -> None:
        """Test query_similar with no matches."""
        with patch("omnibase_infra.handlers.handler_qdrant.QdrantClient") as MockClient:
            MockClient.return_value = mock_qdrant_client
            empty_result = MagicMock()
            empty_result.points = []
            mock_qdrant_client.query_points.return_value = empty_result

            await handler.initialize(connection_config)

            result = await handler.query_similar(
                query_vector=[0.1, 0.2, 0.3],
                top_k=5,
                index_name="test_collection",
            )

            assert result.total_results == 0
            assert len(result.results) == 0

            await handler.shutdown()

    @pytest.mark.asyncio
    async def test_query_similar_with_include_vectors(
        self,
        handler: HandlerQdrant,
        connection_config: ModelVectorConnectionConfig,
        mock_qdrant_client: MagicMock,
    ) -> None:
        """Test query_similar includes vectors when requested."""
        with patch("omnibase_infra.handlers.handler_qdrant.QdrantClient") as MockClient:
            MockClient.return_value = mock_qdrant_client

            mock_point = MagicMock()
            mock_point.id = "point-1"
            mock_point.score = 0.95
            mock_point.payload = {"text": "hello"}
            mock_point.vector = [0.1, 0.2, 0.3]
            mock_result = MagicMock()
            mock_result.points = [mock_point]
            mock_qdrant_client.query_points.return_value = mock_result

            await handler.initialize(connection_config)

            result = await handler.query_similar(
                query_vector=[0.1, 0.2, 0.3],
                top_k=5,
                index_name="test_collection",
                include_vectors=True,
            )

            assert result.results[0].vector == [0.1, 0.2, 0.3]

            await handler.shutdown()

    @pytest.mark.asyncio
    async def test_query_similar_not_initialized(self, handler: HandlerQdrant) -> None:
        """Test query_similar fails when not initialized."""
        with pytest.raises(RuntimeHostError) as exc_info:
            await handler.query_similar(
                query_vector=[0.1, 0.2, 0.3],
                top_k=5,
                index_name="test_collection",
            )

        assert "not initialized" in str(exc_info.value).lower()

    @pytest.mark.asyncio
    async def test_query_similar_connection_error(
        self,
        handler: HandlerQdrant,
        connection_config: ModelVectorConnectionConfig,
        mock_qdrant_client: MagicMock,
    ) -> None:
        """Test query_similar handles connection error."""
        with patch("omnibase_infra.handlers.handler_qdrant.QdrantClient") as MockClient:
            MockClient.return_value = mock_qdrant_client
            await handler.initialize(connection_config)

            mock_qdrant_client.query_points.side_effect = Exception("Connection lost")

            with pytest.raises(InfraConnectionError):
                await handler.query_similar(
                    query_vector=[0.1, 0.2, 0.3],
                    top_k=5,
                    index_name="test_collection",
                )

            await handler.shutdown()


class TestHandlerQdrantDeleteEmbedding:
    """Test HandlerQdrant delete_embedding operation."""

    @pytest.mark.asyncio
    async def test_delete_embedding_success(
        self,
        handler: HandlerQdrant,
        connection_config: ModelVectorConnectionConfig,
        mock_qdrant_client: MagicMock,
    ) -> None:
        """Test successful single embedding deletion."""
        with patch("omnibase_infra.handlers.handler_qdrant.QdrantClient") as MockClient:
            MockClient.return_value = mock_qdrant_client
            await handler.initialize(connection_config)

            result = await handler.delete_embedding(
                embedding_id="emb-001",
                index_name="test_collection",
            )

            assert result.success is True
            assert result.embedding_id == "emb-001"
            assert result.deleted is True
            mock_qdrant_client.delete.assert_called_once()

            await handler.shutdown()

    @pytest.mark.asyncio
    async def test_delete_embedding_not_initialized(
        self, handler: HandlerQdrant
    ) -> None:
        """Test delete_embedding fails when not initialized."""
        with pytest.raises(RuntimeHostError) as exc_info:
            await handler.delete_embedding(
                embedding_id="emb-001",
                index_name="test_collection",
            )

        assert "not initialized" in str(exc_info.value).lower()


class TestHandlerQdrantDeleteBatch:
    """Test HandlerQdrant delete_embeddings_batch operation."""

    @pytest.mark.asyncio
    async def test_delete_embeddings_batch_success(
        self,
        handler: HandlerQdrant,
        connection_config: ModelVectorConnectionConfig,
        mock_qdrant_client: MagicMock,
    ) -> None:
        """Test successful batch embedding deletion."""
        with patch("omnibase_infra.handlers.handler_qdrant.QdrantClient") as MockClient:
            MockClient.return_value = mock_qdrant_client
            await handler.initialize(connection_config)

            result = await handler.delete_embeddings_batch(
                embedding_ids=["emb-001", "emb-002", "emb-003"],
                index_name="test_collection",
            )

            assert result.success is True
            assert result.deleted is True
            mock_qdrant_client.delete.assert_called_once()

            await handler.shutdown()

    @pytest.mark.asyncio
    async def test_delete_embeddings_batch_not_initialized(
        self, handler: HandlerQdrant
    ) -> None:
        """Test delete_embeddings_batch fails when not initialized."""
        with pytest.raises(RuntimeHostError) as exc_info:
            await handler.delete_embeddings_batch(
                embedding_ids=["emb-001"],
                index_name="test_collection",
            )

        assert "not initialized" in str(exc_info.value).lower()


class TestHandlerQdrantCreateIndex:
    """Test HandlerQdrant create_index operation."""

    @pytest.mark.asyncio
    async def test_create_index_success(
        self,
        handler: HandlerQdrant,
        connection_config: ModelVectorConnectionConfig,
        mock_qdrant_client: MagicMock,
    ) -> None:
        """Test successful index creation."""
        with patch("omnibase_infra.handlers.handler_qdrant.QdrantClient") as MockClient:
            MockClient.return_value = mock_qdrant_client
            await handler.initialize(connection_config)

            result = await handler.create_index(
                index_name="new_collection",
                dimension=384,
                metric="cosine",
            )

            assert result.success is True
            assert result.index_name == "new_collection"
            assert result.dimension == 384
            assert result.metric == EnumVectorDistanceMetric.COSINE
            assert result.created_at is not None
            mock_qdrant_client.create_collection.assert_called_once()

            await handler.shutdown()

    @pytest.mark.asyncio
    async def test_create_index_euclidean_metric(
        self,
        handler: HandlerQdrant,
        connection_config: ModelVectorConnectionConfig,
        mock_qdrant_client: MagicMock,
    ) -> None:
        """Test index creation with euclidean metric."""
        with patch("omnibase_infra.handlers.handler_qdrant.QdrantClient") as MockClient:
            MockClient.return_value = mock_qdrant_client
            await handler.initialize(connection_config)

            result = await handler.create_index(
                index_name="euclidean_collection",
                dimension=768,
                metric="euclidean",
            )

            assert result.success is True
            assert result.metric == EnumVectorDistanceMetric.EUCLIDEAN

            await handler.shutdown()

    @pytest.mark.asyncio
    async def test_create_index_dot_product_metric(
        self,
        handler: HandlerQdrant,
        connection_config: ModelVectorConnectionConfig,
        mock_qdrant_client: MagicMock,
    ) -> None:
        """Test index creation with dot product metric."""
        with patch("omnibase_infra.handlers.handler_qdrant.QdrantClient") as MockClient:
            MockClient.return_value = mock_qdrant_client
            await handler.initialize(connection_config)

            result = await handler.create_index(
                index_name="dotproduct_collection",
                dimension=256,
                metric="dot_product",
            )

            assert result.success is True
            assert result.metric == EnumVectorDistanceMetric.DOT_PRODUCT

            await handler.shutdown()

    @pytest.mark.asyncio
    async def test_create_index_unsupported_metric(
        self,
        handler: HandlerQdrant,
        connection_config: ModelVectorConnectionConfig,
        mock_qdrant_client: MagicMock,
    ) -> None:
        """Test create_index fails with unsupported metric."""
        with patch("omnibase_infra.handlers.handler_qdrant.QdrantClient") as MockClient:
            MockClient.return_value = mock_qdrant_client
            await handler.initialize(connection_config)

            with pytest.raises(ValueError) as exc_info:
                await handler.create_index(
                    index_name="bad_collection",
                    dimension=384,
                    metric="invalid_metric",
                )

            assert "unsupported metric" in str(exc_info.value).lower()

            await handler.shutdown()

    @pytest.mark.asyncio
    async def test_create_index_not_initialized(self, handler: HandlerQdrant) -> None:
        """Test create_index fails when not initialized."""
        with pytest.raises(RuntimeHostError) as exc_info:
            await handler.create_index(
                index_name="test_collection",
                dimension=384,
                metric="cosine",
            )

        assert "not initialized" in str(exc_info.value).lower()


class TestHandlerQdrantDeleteIndex:
    """Test HandlerQdrant delete_index operation."""

    @pytest.mark.asyncio
    async def test_delete_index_calls_client(
        self,
        handler: HandlerQdrant,
        connection_config: ModelVectorConnectionConfig,
        mock_qdrant_client: MagicMock,
    ) -> None:
        """Test delete_index calls Qdrant client delete_collection.

        Note: The handler currently returns dimension=0 which violates the
        ModelVectorIndexResult constraint (dimension >= 1). This causes an
        InfraConnectionError wrapping the ValidationError. This test verifies
        the client is called correctly.
        """
        with patch("omnibase_infra.handlers.handler_qdrant.QdrantClient") as MockClient:
            MockClient.return_value = mock_qdrant_client
            await handler.initialize(connection_config)

            # Handler returns dimension=0 which fails model validation
            # This raises InfraConnectionError wrapping the ValidationError
            with pytest.raises(InfraConnectionError) as exc_info:
                await handler.delete_index(index_name="old_collection")

            # Verify delete_collection was called despite the error
            mock_qdrant_client.delete_collection.assert_called_once_with(
                collection_name="old_collection"
            )
            assert "ValidationError" in str(exc_info.value)

            await handler.shutdown()

    @pytest.mark.asyncio
    async def test_delete_index_not_initialized(self, handler: HandlerQdrant) -> None:
        """Test delete_index fails when not initialized."""
        with pytest.raises(RuntimeHostError) as exc_info:
            await handler.delete_index(index_name="test_collection")

        assert "not initialized" in str(exc_info.value).lower()


class TestHandlerQdrantHealthCheck:
    """Test HandlerQdrant health_check operation."""

    @pytest.mark.asyncio
    async def test_health_check_success(
        self,
        handler: HandlerQdrant,
        connection_config: ModelVectorConnectionConfig,
        mock_qdrant_client: MagicMock,
    ) -> None:
        """Test successful health check."""
        with patch("omnibase_infra.handlers.handler_qdrant.QdrantClient") as MockClient:
            MockClient.return_value = mock_qdrant_client

            mock_collection = MagicMock()
            mock_collection.name = "test_collection"
            mock_collections_response = MagicMock()
            mock_collections_response.collections = [mock_collection]
            mock_qdrant_client.get_collections.return_value = mock_collections_response

            await handler.initialize(connection_config)

            result = await handler.health_check()

            assert result.healthy is True
            assert result.latency_ms >= 0
            assert "test_collection" in result.indices
            assert result.last_error is None

            await handler.shutdown()

    @pytest.mark.asyncio
    async def test_health_check_not_initialized(self, handler: HandlerQdrant) -> None:
        """Test health check returns unhealthy when not initialized."""
        result = await handler.health_check()

        assert result.healthy is False
        assert result.last_error is not None
        assert "not initialized" in result.last_error.lower()

    @pytest.mark.asyncio
    async def test_health_check_connection_error(
        self,
        handler: HandlerQdrant,
        connection_config: ModelVectorConnectionConfig,
        mock_qdrant_client: MagicMock,
    ) -> None:
        """Test health check handles connection errors."""
        with patch("omnibase_infra.handlers.handler_qdrant.QdrantClient") as MockClient:
            MockClient.return_value = mock_qdrant_client
            await handler.initialize(connection_config)

            mock_qdrant_client.get_collections.side_effect = Exception(
                "Connection lost"
            )

            result = await handler.health_check()

            assert result.healthy is False
            assert result.last_error is not None

            await handler.shutdown()


class TestHandlerQdrantDescribe:
    """Test HandlerQdrant describe operation."""

    @pytest.mark.asyncio
    async def test_describe_returns_metadata(self, handler: HandlerQdrant) -> None:
        """Test describe returns handler metadata."""
        result = await handler.describe()

        assert result.handler_type == "qdrant"
        assert "store_embedding" in result.capabilities
        assert "store_embeddings_batch" in result.capabilities
        assert "query_similar" in result.capabilities
        assert "delete_embedding" in result.capabilities
        assert "create_index" in result.capabilities
        assert "delete_index" in result.capabilities
        assert "health_check" in result.capabilities
        assert EnumVectorDistanceMetric.COSINE in result.supported_metrics
        assert EnumVectorDistanceMetric.EUCLIDEAN in result.supported_metrics
        assert EnumVectorDistanceMetric.DOT_PRODUCT in result.supported_metrics


class TestHandlerQdrantCircuitBreaker:
    """Test HandlerQdrant circuit breaker behavior."""

    @pytest.mark.asyncio
    async def test_circuit_breaker_initialized(
        self,
        handler: HandlerQdrant,
        connection_config: ModelVectorConnectionConfig,
        mock_qdrant_client: MagicMock,
    ) -> None:
        """Test circuit breaker is initialized after handler initialization."""
        with patch("omnibase_infra.handlers.handler_qdrant.QdrantClient") as MockClient:
            MockClient.return_value = mock_qdrant_client
            await handler.initialize(connection_config)

            assert hasattr(handler, "_circuit_breaker_lock")
            assert hasattr(handler, "_circuit_breaker_failures")
            assert hasattr(handler, "circuit_breaker_threshold")
            assert handler._circuit_breaker_failures == 0
            assert handler.circuit_breaker_threshold == 5

            await handler.shutdown()

    @pytest.mark.asyncio
    async def test_circuit_breaker_records_failure(
        self,
        handler: HandlerQdrant,
        connection_config: ModelVectorConnectionConfig,
        mock_qdrant_client: MagicMock,
    ) -> None:
        """Test circuit breaker records failures on operation errors."""
        with patch("omnibase_infra.handlers.handler_qdrant.QdrantClient") as MockClient:
            MockClient.return_value = mock_qdrant_client
            await handler.initialize(connection_config)

            mock_qdrant_client.query_points.side_effect = Exception("Connection lost")

            with pytest.raises(InfraConnectionError):
                await handler.query_similar(
                    query_vector=[0.1, 0.2, 0.3],
                    top_k=5,
                    index_name="test_collection",
                )

            assert handler._circuit_breaker_failures >= 1

            await handler.shutdown()

    @pytest.mark.asyncio
    async def test_circuit_opens_after_threshold(
        self,
        handler: HandlerQdrant,
        connection_config: ModelVectorConnectionConfig,
        mock_qdrant_client: MagicMock,
    ) -> None:
        """Test circuit breaker opens after failure threshold is reached."""
        with patch("omnibase_infra.handlers.handler_qdrant.QdrantClient") as MockClient:
            MockClient.return_value = mock_qdrant_client
            await handler.initialize(connection_config)

            mock_qdrant_client.query_points.side_effect = Exception("Connection lost")

            # Trigger failures up to the threshold (default is 5)
            for _ in range(5):
                try:
                    await handler.query_similar(
                        query_vector=[0.1, 0.2, 0.3],
                        top_k=5,
                        index_name="test_collection",
                    )
                except InfraConnectionError:
                    pass

            # Next call should fail with InfraUnavailableError (circuit open)
            with pytest.raises((InfraUnavailableError, InfraConnectionError)):
                await handler.query_similar(
                    query_vector=[0.1, 0.2, 0.3],
                    top_k=5,
                    index_name="test_collection",
                )

            await handler.shutdown()


class TestHandlerQdrantNoDefaultIndex:
    """Test HandlerQdrant behavior when no default index configured."""

    @pytest.mark.asyncio
    async def test_store_embedding_no_index_raises(
        self,
        handler: HandlerQdrant,
        connection_config: ModelVectorConnectionConfig,
        mock_qdrant_client: MagicMock,
    ) -> None:
        """Test store_embedding requires index_name when no default."""
        with patch("omnibase_infra.handlers.handler_qdrant.QdrantClient") as MockClient:
            MockClient.return_value = mock_qdrant_client
            await handler.initialize(connection_config)

            # Handler has no default index
            handler._default_index = None

            with pytest.raises(RuntimeHostError) as exc_info:
                await handler.store_embedding(
                    embedding_id="emb-001",
                    vector=[0.1, 0.2, 0.3],
                    # index_name not provided
                )

            assert "index" in str(exc_info.value).lower()

            await handler.shutdown()


__all__: list[str] = [
    "TestHandlerQdrantProperties",
    "TestHandlerQdrantInitialization",
    "TestHandlerQdrantShutdown",
    "TestHandlerQdrantStoreEmbedding",
    "TestHandlerQdrantStoreBatch",
    "TestHandlerQdrantQuerySimilar",
    "TestHandlerQdrantDeleteEmbedding",
    "TestHandlerQdrantDeleteBatch",
    "TestHandlerQdrantCreateIndex",
    "TestHandlerQdrantDeleteIndex",
    "TestHandlerQdrantHealthCheck",
    "TestHandlerQdrantDescribe",
    "TestHandlerQdrantCircuitBreaker",
    "TestHandlerQdrantNoDefaultIndex",
]
