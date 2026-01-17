# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Qdrant Vector Store Handler - implements ProtocolVectorStoreHandler.

This handler provides vector store operations for the Qdrant vector database
using the qdrant-client library with circuit breaker pattern for fault tolerance.

Supported Operations:
    - store_embedding: Store a single embedding vector
    - store_embeddings_batch: Batch store multiple embeddings
    - query_similar: Similarity search with metadata filtering
    - delete_embedding: Delete single embedding by ID
    - delete_embeddings_batch: Batch delete embeddings
    - create_index: Create a new collection/index
    - delete_index: Delete a collection/index
    - health_check: Check Qdrant connectivity and status
    - describe: Return handler capabilities and metadata

Protocol Implementation:
    This handler implements ProtocolVectorStoreHandler from omnibase_spi.protocols.storage.
    All methods use typed models from omnibase_core.models.vector for type safety.

Circuit Breaker Pattern:
    Uses MixinAsyncCircuitBreaker for fault tolerance with configurable
    failure threshold and reset timeout.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Mapping
from datetime import UTC, datetime
from uuid import UUID, uuid4

from qdrant_client import QdrantClient
from qdrant_client.http import models as qdrant_models

from omnibase_core.container import ModelONEXContainer
from omnibase_core.models.common.model_schema_value import ModelSchemaValue
from omnibase_core.models.vector import (
    EnumVectorDistanceMetric,
    EnumVectorFilterOperator,
    ModelEmbedding,
    ModelVectorBatchStoreResult,
    ModelVectorConnectionConfig,
    ModelVectorDeleteResult,
    ModelVectorHandlerMetadata,
    ModelVectorHealthStatus,
    ModelVectorIndexConfig,
    ModelVectorIndexResult,
    ModelVectorMetadataFilter,
    ModelVectorSearchResult,
    ModelVectorSearchResults,
    ModelVectorStoreResult,
)
from omnibase_core.types import JsonType
from omnibase_infra.enums import EnumInfraTransportType
from omnibase_infra.errors import (
    InfraAuthenticationError,
    InfraConnectionError,
    InfraUnavailableError,
    ModelInfraErrorContext,
    RuntimeHostError,
)
from omnibase_infra.mixins import MixinAsyncCircuitBreaker
from omnibase_spi.protocols.storage import ProtocolVectorStoreHandler

logger = logging.getLogger(__name__)

# Default configuration
_DEFAULT_TIMEOUT_SECONDS: float = 30.0
_DEFAULT_BATCH_SIZE: int = 100
_HANDLER_VERSION: str = "1.0.0"
_HEALTH_CACHE_TTL_SECONDS: float = 5.0

# Metric mapping from string to Qdrant distance enum
_METRIC_MAP: dict[str, qdrant_models.Distance] = {
    "cosine": qdrant_models.Distance.COSINE,
    "euclidean": qdrant_models.Distance.EUCLID,
    "dot_product": qdrant_models.Distance.DOT,
}

# Reverse mapping from EnumVectorDistanceMetric to Qdrant
_ENUM_TO_QDRANT_METRIC: dict[EnumVectorDistanceMetric, qdrant_models.Distance] = {
    EnumVectorDistanceMetric.COSINE: qdrant_models.Distance.COSINE,
    EnumVectorDistanceMetric.EUCLIDEAN: qdrant_models.Distance.EUCLID,
    EnumVectorDistanceMetric.DOT_PRODUCT: qdrant_models.Distance.DOT,
}

# Filter operator mapping
_FILTER_OP_MAP: dict[EnumVectorFilterOperator, str] = {
    EnumVectorFilterOperator.EQ: "match",
    EnumVectorFilterOperator.NE: "match_except",
    EnumVectorFilterOperator.GT: "range",
    EnumVectorFilterOperator.GTE: "range",
    EnumVectorFilterOperator.LT: "range",
    EnumVectorFilterOperator.LTE: "range",
    EnumVectorFilterOperator.IN: "match_any",
    EnumVectorFilterOperator.NOT_IN: "match_except_any",
}


class HandlerQdrant(MixinAsyncCircuitBreaker, ProtocolVectorStoreHandler):
    """Qdrant vector store handler implementing ProtocolVectorStoreHandler.

    This handler provides vector storage, similarity search, and index management
    operations for the Qdrant vector database.

    Security Policy:
        API keys are treated as secrets and never logged or exposed in errors.

    Circuit Breaker Pattern:
        Uses MixinAsyncCircuitBreaker for fault tolerance.
        Configurable failure_threshold and reset_timeout.

    Thread Safety:
        This handler is NOT thread-safe. Use separate instances for concurrent
        access from different threads. It is coroutine-safe for async operations.
    """

    def __init__(self, container: ModelONEXContainer) -> None:
        """Initialize HandlerQdrant with ONEX container for dependency injection.

        Args:
            container: ONEX container providing dependency injection for
                services, configuration, and runtime context.

        Note:
            The container is stored for interface compliance with the standard ONEX
            handler pattern (def __init__(self, container: ModelONEXContainer)) and
            to enable future DI-based service resolution (e.g., dispatcher routing,
            metrics integration). Currently, the handler operates independently but
            the container parameter ensures API consistency across all handlers.
        """
        self._container = container
        self._client: QdrantClient | None = None
        self._config: ModelVectorConnectionConfig | None = None
        self._default_index: str | None = None
        self._timeout: float = _DEFAULT_TIMEOUT_SECONDS
        self._initialized: bool = False
        # Health check caching
        self._cached_health: ModelVectorHealthStatus | None = None
        self._health_cache_time: float = 0.0

    @property
    def handler_type(self) -> str:
        """Return the type of handler as a string identifier.

        Returns:
            String identifier "vector_store" for this handler type.
        """
        return "vector_store"

    @property
    def supported_metrics(self) -> list[str]:
        """Return the list of distance metrics supported by this handler.

        Returns:
            List of supported metric names: cosine, euclidean, dot_product.
        """
        return ["cosine", "euclidean", "dot_product"]

    async def initialize(
        self,
        connection_config: ModelVectorConnectionConfig,
    ) -> None:
        """Initialize the Qdrant client with connection configuration.

        Establishes connection to the Qdrant server, validates credentials,
        and prepares the handler for operations.

        Args:
            connection_config: Configuration model containing connection parameters.

        Raises:
            RuntimeHostError: If configuration is invalid.
            InfraConnectionError: If connection to Qdrant server fails.
            InfraAuthenticationError: If API key authentication fails.
        """
        init_correlation_id = uuid4()
        logger.info(
            "Initializing %s",
            self.__class__.__name__,
            extra={
                "handler": self.__class__.__name__,
                "correlation_id": str(init_correlation_id),
            },
        )

        self._config = connection_config
        self._timeout = connection_config.timeout

        # Extract API key safely using SecretStr
        api_key: str | None = None
        if connection_config.api_key is not None:
            api_key = connection_config.api_key.get_secret_value()

        # Create client
        try:
            self._client = QdrantClient(
                url=connection_config.url,
                api_key=api_key,
                timeout=int(self._timeout),
            )
            # Test connection by listing collections
            self._client.get_collections()
            self._initialized = True
            logger.info(
                "%s initialized successfully",
                self.__class__.__name__,
                extra={
                    "handler": self.__class__.__name__,
                    "correlation_id": str(init_correlation_id),
                },
            )
        except Exception as e:
            ctx = ModelInfraErrorContext.with_correlation(
                correlation_id=init_correlation_id,
                transport_type=EnumInfraTransportType.QDRANT,
                operation="initialize",
                target_name="qdrant_handler",
            )
            error_msg = str(e).lower()
            if "unauthorized" in error_msg or "forbidden" in error_msg:
                raise InfraAuthenticationError(
                    "Qdrant authentication failed - check API key", context=ctx
                ) from e
            raise InfraConnectionError(
                "Failed to connect to Qdrant server", context=ctx
            ) from e

        # Initialize circuit breaker
        self._init_circuit_breaker(
            threshold=5,
            reset_timeout=60.0,
            service_name="qdrant",
            transport_type=EnumInfraTransportType.QDRANT,
        )

    async def shutdown(self, timeout_seconds: float = 30.0) -> None:
        """Release resources and close connections to Qdrant.

        Args:
            timeout_seconds: Maximum time to wait for shutdown to complete.
        """
        if self._client is not None:
            self._client.close()
            self._client = None
        self._initialized = False
        # Clear health cache
        self._cached_health = None
        self._health_cache_time = 0.0
        logger.info("HandlerQdrant shutdown complete")

    def _ensure_initialized(
        self, operation: str, correlation_id: UUID | None = None
    ) -> None:
        """Ensure the handler is initialized before operations.

        Args:
            operation: Name of the operation being attempted.
            correlation_id: Optional correlation ID for error context.

        Raises:
            RuntimeHostError: If handler is not initialized.
        """
        if not self._initialized or self._client is None:
            ctx = ModelInfraErrorContext.with_correlation(
                correlation_id=correlation_id,
                transport_type=EnumInfraTransportType.QDRANT,
                operation=operation,
                target_name="qdrant_handler",
            )
            raise RuntimeHostError(
                "HandlerQdrant not initialized. Call initialize() first.",
                context=ctx,
            )

    def _resolve_index_name(self, index_name: str | None) -> str:
        """Resolve index name, using default if not provided.

        Args:
            index_name: Provided index name or None.

        Returns:
            Resolved index name.

        Raises:
            RuntimeHostError: If no index name available.
        """
        if index_name:
            return index_name
        if self._default_index:
            return self._default_index
        raise RuntimeHostError("No index_name provided and no default index configured")

    def _convert_metadata_to_qdrant(
        self, metadata: Mapping[str, JsonType] | None
    ) -> dict[str, object]:
        """Convert metadata mapping to Qdrant payload format.

        Args:
            metadata: Optional metadata mapping with JsonValue types.

        Returns:
            Dictionary suitable for Qdrant payload.
        """
        if metadata is None:
            return {}
        return dict(metadata.items())

    def _convert_model_metadata_to_qdrant(
        self, metadata: dict[str, ModelSchemaValue]
    ) -> dict[str, object]:
        """Convert ModelSchemaValue metadata to Qdrant payload format.

        Args:
            metadata: Metadata with ModelSchemaValue types.

        Returns:
            Dictionary suitable for Qdrant payload.
        """
        return {k: v.to_value() for k, v in metadata.items()}

    def _convert_qdrant_payload_to_metadata(
        self, payload: dict[str, object] | None
    ) -> dict[str, ModelSchemaValue]:
        """Convert Qdrant payload to ModelSchemaValue metadata.

        Args:
            payload: Qdrant point payload.

        Returns:
            Metadata dictionary with ModelSchemaValue types.
        """
        if payload is None:
            return {}
        return {k: ModelSchemaValue.from_value(v) for k, v in payload.items()}

    def _build_qdrant_filter(
        self, filter_metadata: ModelVectorMetadataFilter
    ) -> qdrant_models.Filter:
        """Build Qdrant filter from ModelVectorMetadataFilter.

        Args:
            filter_metadata: Filter specification.

        Returns:
            Qdrant Filter object.
        """
        field = filter_metadata.field
        operator = filter_metadata.operator
        value = filter_metadata.value.to_value()

        conditions: list[qdrant_models.FieldCondition] = []

        if operator == EnumVectorFilterOperator.EQ:
            conditions.append(
                qdrant_models.FieldCondition(
                    key=field,
                    match=qdrant_models.MatchValue(value=value),
                )
            )
        elif operator == EnumVectorFilterOperator.NE:
            # Use must_not for not-equal
            return qdrant_models.Filter(
                must_not=[
                    qdrant_models.FieldCondition(
                        key=field,
                        match=qdrant_models.MatchValue(value=value),
                    )
                ]
            )
        elif operator in (
            EnumVectorFilterOperator.GT,
            EnumVectorFilterOperator.GTE,
            EnumVectorFilterOperator.LT,
            EnumVectorFilterOperator.LTE,
        ):
            range_params: dict[str, float | int | None] = {
                "gt": None,
                "gte": None,
                "lt": None,
                "lte": None,
            }
            if operator == EnumVectorFilterOperator.GT:
                range_params["gt"] = value  # type: ignore[assignment]
            elif operator == EnumVectorFilterOperator.GTE:
                range_params["gte"] = value  # type: ignore[assignment]
            elif operator == EnumVectorFilterOperator.LT:
                range_params["lt"] = value  # type: ignore[assignment]
            elif operator == EnumVectorFilterOperator.LTE:
                range_params["lte"] = value  # type: ignore[assignment]
            conditions.append(
                qdrant_models.FieldCondition(
                    key=field,
                    range=qdrant_models.Range(**range_params),  # type: ignore[arg-type]
                )
            )
        elif operator == EnumVectorFilterOperator.IN:
            if isinstance(value, list):
                conditions.append(
                    qdrant_models.FieldCondition(
                        key=field,
                        match=qdrant_models.MatchAny(any=value),
                    )
                )
        elif operator == EnumVectorFilterOperator.NOT_IN:
            if isinstance(value, list):
                return qdrant_models.Filter(
                    must_not=[
                        qdrant_models.FieldCondition(
                            key=field,
                            match=qdrant_models.MatchAny(any=value),
                        )
                    ]
                )
        elif operator == EnumVectorFilterOperator.CONTAINS:
            conditions.append(
                qdrant_models.FieldCondition(
                    key=field,
                    match=qdrant_models.MatchText(text=str(value)),
                )
            )
        elif operator == EnumVectorFilterOperator.EXISTS:
            conditions.append(
                qdrant_models.FieldCondition(
                    key=field,
                    is_null=qdrant_models.IsNullCondition(is_null=False),
                )
            )

        return qdrant_models.Filter(must=conditions)

    async def store_embedding(
        self,
        embedding_id: str,
        vector: list[float],
        metadata: Mapping[str, JsonType] | None = None,
        index_name: str | None = None,
    ) -> ModelVectorStoreResult:
        """Store a single embedding vector with optional metadata.

        Args:
            embedding_id: Unique identifier for the embedding.
            vector: The embedding vector as a list of floats.
            metadata: Optional metadata mapping to store with the embedding.
            index_name: Name of the index/collection to store in.

        Returns:
            ModelVectorStoreResult containing operation result.

        Raises:
            RuntimeHostError: If handler not initialized.
            InfraConnectionError: If storage operation fails.
        """
        correlation_id = uuid4()
        self._ensure_initialized("store_embedding", correlation_id)

        resolved_index = self._resolve_index_name(index_name)

        # Check circuit breaker
        async with self._circuit_breaker_lock:
            await self._check_circuit_breaker("store_embedding", correlation_id)

        try:
            if self._client is None:
                raise RuntimeHostError("Client is None after initialization check")

            payload = self._convert_metadata_to_qdrant(metadata)
            self._client.upsert(
                collection_name=resolved_index,
                points=[
                    qdrant_models.PointStruct(
                        id=embedding_id,
                        vector=vector,
                        payload=payload,
                    )
                ],
            )

            # Reset circuit breaker on success
            async with self._circuit_breaker_lock:
                await self._reset_circuit_breaker()

            return ModelVectorStoreResult(
                success=True,
                embedding_id=embedding_id,
                index_name=resolved_index,
                timestamp=datetime.now(UTC),
            )
        except (InfraUnavailableError, RuntimeHostError):
            raise
        except Exception as e:
            ctx = ModelInfraErrorContext.with_correlation(
                correlation_id=correlation_id,
                transport_type=EnumInfraTransportType.QDRANT,
                operation="store_embedding",
                target_name=resolved_index,
            )
            async with self._circuit_breaker_lock:
                await self._record_circuit_failure("store_embedding", correlation_id)
            raise InfraConnectionError(
                f"Failed to store embedding: {type(e).__name__}", context=ctx
            ) from e

    async def store_embeddings_batch(
        self,
        embeddings: list[ModelEmbedding],
        index_name: str | None = None,
        batch_size: int = 100,
    ) -> ModelVectorBatchStoreResult:
        """Store multiple embeddings efficiently in a batch operation.

        Args:
            embeddings: List of ModelEmbedding instances.
            index_name: Name of the index/collection to store in.
            batch_size: Number of embeddings to process per batch.

        Returns:
            ModelVectorBatchStoreResult containing batch operation result.

        Raises:
            RuntimeHostError: If handler not initialized.
            InfraConnectionError: If batch operation fails.
        """
        correlation_id = uuid4()
        self._ensure_initialized("store_embeddings_batch", correlation_id)

        resolved_index = self._resolve_index_name(index_name)
        start_time = time.time()

        # Check circuit breaker
        async with self._circuit_breaker_lock:
            await self._check_circuit_breaker("store_embeddings_batch", correlation_id)

        total_stored = 0
        failed_ids: list[str] = []

        try:
            if self._client is None:
                raise RuntimeHostError("Client is None after initialization check")

            # Process in batches
            for i in range(0, len(embeddings), batch_size):
                batch = embeddings[i : i + batch_size]
                points = [
                    qdrant_models.PointStruct(
                        id=emb.id,
                        vector=emb.vector,
                        payload=self._convert_model_metadata_to_qdrant(emb.metadata),
                    )
                    for emb in batch
                ]

                try:
                    self._client.upsert(
                        collection_name=resolved_index,
                        points=points,
                    )
                    total_stored += len(batch)
                except Exception:
                    # Track failed IDs in this batch
                    failed_ids.extend([emb.id for emb in batch])

            # Reset circuit breaker on success
            async with self._circuit_breaker_lock:
                await self._reset_circuit_breaker()

            execution_time_ms = int((time.time() - start_time) * 1000)

            return ModelVectorBatchStoreResult(
                success=len(failed_ids) == 0,
                total_stored=total_stored,
                failed_ids=failed_ids,
                execution_time_ms=execution_time_ms,
            )
        except (InfraUnavailableError, RuntimeHostError):
            raise
        except Exception as e:
            ctx = ModelInfraErrorContext.with_correlation(
                correlation_id=correlation_id,
                transport_type=EnumInfraTransportType.QDRANT,
                operation="store_embeddings_batch",
                target_name=resolved_index,
            )
            async with self._circuit_breaker_lock:
                await self._record_circuit_failure(
                    "store_embeddings_batch", correlation_id
                )
            raise InfraConnectionError(
                f"Batch store failed: {type(e).__name__}", context=ctx
            ) from e

    async def query_similar(
        self,
        query_vector: list[float],
        top_k: int = 10,
        index_name: str | None = None,
        filter_metadata: ModelVectorMetadataFilter | None = None,
        include_metadata: bool = True,
        include_vectors: bool = False,
        score_threshold: float | None = None,
    ) -> ModelVectorSearchResults:
        """Find similar vectors using similarity/distance search.

        Args:
            query_vector: The query embedding vector to search against.
            top_k: Maximum number of results to return.
            index_name: Name of the index/collection to search.
            filter_metadata: Optional metadata filter to restrict search.
            include_metadata: Whether to include metadata in results.
            include_vectors: Whether to include vectors in results.
            score_threshold: Minimum similarity score threshold.

        Returns:
            ModelVectorSearchResults containing search results.

        Raises:
            RuntimeHostError: If handler not initialized.
            InfraConnectionError: If search operation fails.
        """
        correlation_id = uuid4()
        self._ensure_initialized("query_similar", correlation_id)

        resolved_index = self._resolve_index_name(index_name)
        start_time = time.time()

        # Check circuit breaker
        async with self._circuit_breaker_lock:
            await self._check_circuit_breaker("query_similar", correlation_id)

        try:
            if self._client is None:
                raise RuntimeHostError("Client is None after initialization check")

            # Build filter if provided
            qdrant_filter: qdrant_models.Filter | None = None
            if filter_metadata is not None:
                qdrant_filter = self._build_qdrant_filter(filter_metadata)

            # Execute search using query_points API
            query_result = self._client.query_points(
                collection_name=resolved_index,
                query=query_vector,
                limit=top_k,
                query_filter=qdrant_filter,
                with_payload=include_metadata,
                with_vectors=include_vectors,
                score_threshold=score_threshold,
            )

            # Convert results
            results: list[ModelVectorSearchResult] = []
            for point in query_result.points:
                metadata = (
                    self._convert_qdrant_payload_to_metadata(point.payload)
                    if point.payload and include_metadata
                    else {}
                )
                vector_data: list[float] | None = None
                if include_vectors and point.vector is not None:
                    # Handle both list and dict vector formats
                    raw_vector = point.vector
                    if isinstance(raw_vector, list):
                        vector_data = [float(v) for v in raw_vector]  # type: ignore[arg-type]
                    elif isinstance(raw_vector, dict):
                        # For named vectors, get the default one
                        first_vector = next(iter(raw_vector.values()), None)
                        if first_vector is not None and isinstance(first_vector, list):
                            vector_data = [float(v) for v in first_vector]  # type: ignore[arg-type]

                results.append(
                    ModelVectorSearchResult(
                        id=str(point.id),
                        score=float(point.score) if point.score is not None else 0.0,
                        metadata=metadata,
                        vector=vector_data,
                    )
                )

            # Reset circuit breaker on success
            async with self._circuit_breaker_lock:
                await self._reset_circuit_breaker()

            query_time_ms = int((time.time() - start_time) * 1000)

            return ModelVectorSearchResults(
                results=results,
                total_results=len(results),
                query_time_ms=query_time_ms,
            )
        except (InfraUnavailableError, RuntimeHostError):
            raise
        except Exception as e:
            ctx = ModelInfraErrorContext.with_correlation(
                correlation_id=correlation_id,
                transport_type=EnumInfraTransportType.QDRANT,
                operation="query_similar",
                target_name=resolved_index,
            )
            async with self._circuit_breaker_lock:
                await self._record_circuit_failure("query_similar", correlation_id)
            raise InfraConnectionError(
                f"Search failed: {type(e).__name__}", context=ctx
            ) from e

    async def delete_embedding(
        self,
        embedding_id: str,
        index_name: str | None = None,
    ) -> ModelVectorDeleteResult:
        """Remove a single embedding by ID.

        Args:
            embedding_id: Unique identifier of the embedding to delete.
            index_name: Name of the index/collection containing the embedding.

        Returns:
            ModelVectorDeleteResult containing deletion result.

        Raises:
            RuntimeHostError: If handler not initialized.
            InfraConnectionError: If deletion operation fails.
        """
        correlation_id = uuid4()
        self._ensure_initialized("delete_embedding", correlation_id)

        resolved_index = self._resolve_index_name(index_name)

        # Check circuit breaker
        async with self._circuit_breaker_lock:
            await self._check_circuit_breaker("delete_embedding", correlation_id)

        try:
            if self._client is None:
                raise RuntimeHostError("Client is None after initialization check")

            self._client.delete(
                collection_name=resolved_index,
                points_selector=qdrant_models.PointIdsList(points=[embedding_id]),
            )

            # Reset circuit breaker on success
            async with self._circuit_breaker_lock:
                await self._reset_circuit_breaker()

            return ModelVectorDeleteResult(
                success=True,
                embedding_id=embedding_id,
                deleted=True,
            )
        except (InfraUnavailableError, RuntimeHostError):
            raise
        except Exception as e:
            ctx = ModelInfraErrorContext.with_correlation(
                correlation_id=correlation_id,
                transport_type=EnumInfraTransportType.QDRANT,
                operation="delete_embedding",
                target_name=resolved_index,
            )
            async with self._circuit_breaker_lock:
                await self._record_circuit_failure("delete_embedding", correlation_id)
            raise InfraConnectionError(
                f"Delete failed: {type(e).__name__}", context=ctx
            ) from e

    async def delete_embeddings_batch(
        self,
        embedding_ids: list[str],
        index_name: str | None = None,
    ) -> ModelVectorDeleteResult:
        """Remove multiple embeddings by their IDs.

        Args:
            embedding_ids: List of embedding IDs to delete.
            index_name: Name of the index/collection containing the embeddings.

        Returns:
            ModelVectorDeleteResult containing batch deletion result.

        Raises:
            RuntimeHostError: If handler not initialized.
            InfraConnectionError: If batch deletion fails.
        """
        correlation_id = uuid4()
        self._ensure_initialized("delete_embeddings_batch", correlation_id)

        resolved_index = self._resolve_index_name(index_name)

        # Check circuit breaker
        async with self._circuit_breaker_lock:
            await self._check_circuit_breaker("delete_embeddings_batch", correlation_id)

        try:
            if self._client is None:
                raise RuntimeHostError("Client is None after initialization check")

            self._client.delete(
                collection_name=resolved_index,
                points_selector=qdrant_models.PointIdsList(points=embedding_ids),
            )

            # Reset circuit breaker on success
            async with self._circuit_breaker_lock:
                await self._reset_circuit_breaker()

            # Return first ID for the single embedding_id field
            # The protocol uses embedding_id singular even for batch
            return ModelVectorDeleteResult(
                success=True,
                embedding_id=embedding_ids[0] if embedding_ids else "",
                deleted=True,
            )
        except (InfraUnavailableError, RuntimeHostError):
            raise
        except Exception as e:
            ctx = ModelInfraErrorContext.with_correlation(
                correlation_id=correlation_id,
                transport_type=EnumInfraTransportType.QDRANT,
                operation="delete_embeddings_batch",
                target_name=resolved_index,
            )
            async with self._circuit_breaker_lock:
                await self._record_circuit_failure(
                    "delete_embeddings_batch", correlation_id
                )
            raise InfraConnectionError(
                f"Batch delete failed: {type(e).__name__}", context=ctx
            ) from e

    async def create_index(
        self,
        index_name: str,
        dimension: int,
        metric: str = "cosine",
        index_config: ModelVectorIndexConfig | None = None,
    ) -> ModelVectorIndexResult:
        """Create a new vector index/collection.

        Args:
            index_name: Unique name for the new index.
            dimension: Vector dimension.
            metric: Distance metric to use (cosine, euclidean, dot_product).
            index_config: Optional configuration with HNSW and quantization settings.

        Returns:
            ModelVectorIndexResult containing creation result.

        Raises:
            RuntimeHostError: If handler not initialized.
            InfraConnectionError: If index creation fails.
            ValueError: If metric is not supported.
        """
        correlation_id = uuid4()
        self._ensure_initialized("create_index", correlation_id)

        # Validate metric
        metric_lower = metric.lower()
        if metric_lower not in self.supported_metrics:
            raise ValueError(
                f"Unsupported metric '{metric}'. "
                f"Supported: {', '.join(self.supported_metrics)}"
            )

        # Check circuit breaker
        async with self._circuit_breaker_lock:
            await self._check_circuit_breaker("create_index", correlation_id)

        try:
            if self._client is None:
                raise RuntimeHostError("Client is None after initialization check")

            # Map metric string to Qdrant distance enum
            distance_enum = _METRIC_MAP[metric_lower]

            # Build vectors config
            vectors_config = qdrant_models.VectorParams(
                size=dimension,
                distance=distance_enum,
            )

            # Apply HNSW config if provided
            hnsw_config: qdrant_models.HnswConfigDiff | None = None
            if index_config and index_config.hnsw_config:
                hnsw_config = qdrant_models.HnswConfigDiff(
                    m=index_config.hnsw_config.m,
                    ef_construct=index_config.hnsw_config.ef_construction,
                )

            # Create collection
            self._client.create_collection(
                collection_name=index_name,
                vectors_config=vectors_config,
                hnsw_config=hnsw_config,
            )

            # Reset circuit breaker on success
            async with self._circuit_breaker_lock:
                await self._reset_circuit_breaker()

            # Map metric string to enum
            metric_enum = {
                "cosine": EnumVectorDistanceMetric.COSINE,
                "euclidean": EnumVectorDistanceMetric.EUCLIDEAN,
                "dot_product": EnumVectorDistanceMetric.DOT_PRODUCT,
            }.get(metric_lower, EnumVectorDistanceMetric.COSINE)

            return ModelVectorIndexResult(
                success=True,
                index_name=index_name,
                dimension=dimension,
                metric=metric_enum,
                created_at=datetime.now(UTC),
            )
        except (InfraUnavailableError, RuntimeHostError, ValueError):
            raise
        except Exception as e:
            ctx = ModelInfraErrorContext.with_correlation(
                correlation_id=correlation_id,
                transport_type=EnumInfraTransportType.QDRANT,
                operation="create_index",
                target_name=index_name,
            )
            async with self._circuit_breaker_lock:
                await self._record_circuit_failure("create_index", correlation_id)
            raise InfraConnectionError(
                f"Failed to create index: {type(e).__name__}", context=ctx
            ) from e

    async def delete_index(
        self,
        index_name: str,
    ) -> ModelVectorIndexResult:
        """Delete a vector index/collection.

        Args:
            index_name: Name of the index to delete.

        Returns:
            ModelVectorIndexResult containing deletion result.

        Raises:
            RuntimeHostError: If handler not initialized.
            InfraConnectionError: If deletion fails.
        """
        correlation_id = uuid4()
        self._ensure_initialized("delete_index", correlation_id)

        # Check circuit breaker
        async with self._circuit_breaker_lock:
            await self._check_circuit_breaker("delete_index", correlation_id)

        try:
            if self._client is None:
                raise RuntimeHostError("Client is None after initialization check")

            self._client.delete_collection(collection_name=index_name)

            # Reset circuit breaker on success
            async with self._circuit_breaker_lock:
                await self._reset_circuit_breaker()

            return ModelVectorIndexResult(
                success=True,
                index_name=index_name,
                dimension=0,  # Unknown after deletion
                metric=EnumVectorDistanceMetric.COSINE,  # Default
                created_at=None,
            )
        except (InfraUnavailableError, RuntimeHostError):
            raise
        except Exception as e:
            ctx = ModelInfraErrorContext.with_correlation(
                correlation_id=correlation_id,
                transport_type=EnumInfraTransportType.QDRANT,
                operation="delete_index",
                target_name=index_name,
            )
            async with self._circuit_breaker_lock:
                await self._record_circuit_failure("delete_index", correlation_id)
            raise InfraConnectionError(
                f"Failed to delete index: {type(e).__name__}", context=ctx
            ) from e

    async def health_check(self) -> ModelVectorHealthStatus:
        """Check handler health and connectivity to Qdrant.

        Implements caching to avoid overwhelming the backend with frequent
        health checks. Cache TTL is configured via _HEALTH_CACHE_TTL_SECONDS.

        Returns:
            ModelVectorHealthStatus containing health status.
        """
        # Return cached result if recent
        current_time = time.time()
        if (
            self._cached_health is not None
            and current_time - self._health_cache_time < _HEALTH_CACHE_TTL_SECONDS
        ):
            return self._cached_health

        start_time = time.time()

        if not self._initialized or self._client is None:
            return ModelVectorHealthStatus(
                healthy=False,
                latency_ms=0,
                details={},
                indices=[],
                last_error="Handler not initialized",
            )

        try:
            # List collections to verify connectivity
            collections_response = self._client.get_collections()
            collection_names = [c.name for c in collections_response.collections]

            latency_ms = int((time.time() - start_time) * 1000)

            health = ModelVectorHealthStatus(
                healthy=True,
                latency_ms=latency_ms,
                details={
                    "version": ModelSchemaValue.from_value(_HANDLER_VERSION),
                    "backend": ModelSchemaValue.from_value("qdrant"),
                },
                indices=collection_names,
                last_error=None,
            )

            # Cache the result
            self._cached_health = health
            self._health_cache_time = current_time

            return health
        except Exception as e:
            latency_ms = int((time.time() - start_time) * 1000)
            return ModelVectorHealthStatus(
                healthy=False,
                latency_ms=latency_ms,
                details={},
                indices=[],
                last_error=f"Health check failed: {type(e).__name__}",
            )

    async def describe(self) -> ModelVectorHandlerMetadata:  # type: ignore[override]
        """Return handler metadata and capabilities.

        Returns:
            ModelVectorHandlerMetadata containing handler metadata.
        """
        return ModelVectorHandlerMetadata(
            handler_type="qdrant",
            capabilities=[
                "store_embedding",
                "store_embeddings_batch",
                "query_similar",
                "delete_embedding",
                "delete_embeddings_batch",
                "create_index",
                "delete_index",
                "health_check",
                "filter_metadata",
                "score_threshold",
            ],
            supported_metrics=[
                EnumVectorDistanceMetric.COSINE,
                EnumVectorDistanceMetric.EUCLIDEAN,
                EnumVectorDistanceMetric.DOT_PRODUCT,
            ],
        )


__all__: list[str] = ["HandlerQdrant"]
