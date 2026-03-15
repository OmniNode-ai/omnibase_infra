# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Registry Effect Node for PostgreSQL Registration.  # ai-slop-ok: pre-existing docstring opener

Note on ONEX Naming Convention:
    This node class is named `NodeRegistryEffect` following ONEX convention:
    `Node<Name><Type>` where Name="Registry" and Type="Effect".

    The file is named `registry_effect.py` rather than `node.py` because this
    module is organized within a domain-specific effects/ directory. The canonical
    ONEX pattern uses `node.py` for single-node directories (e.g., `nodes/<adapter>/node.py`).
    When multiple effect implementations exist in a shared directory, descriptive
    naming like `registry_effect.py` is acceptable per infrastructure conventions.

    Future refactoring may move this to `nodes/registry_effect/node.py` for full
    ONEX compliance once the effect node requires its own directory with registry/,
    models/, and contract.yaml subdirectories.

This module provides NodeRegistryEffect, an Effect node responsible for executing
registration operations against PostgreSQL.

Intent Format Compatibility (OMN-1258):
    This Effect node receives domain-specific request objects (ModelRegistryRequest),
    NOT raw ModelIntent objects. The intent-to-request translation is handled by the
    Orchestrator layer.

    The RegistrationReducer emits intents with typed payloads:
        - intent_type is set to "postgres.upsert_registration"
          (the sole registration intent after Consul removal in OMN-3540)

    The Orchestrator/Runtime layer is responsible for:
        1. Consuming ModelIntent objects from reducer output
        2. Routing on intent_type to determine target backend
        3. Building a ModelRegistryRequest from payload data
        4. Calling NodeRegistryEffect.register_node(request)

    This design keeps the Effect layer focused on I/O execution without coupling
    to the intent format.

Architecture:
    NodeRegistryEffect follows the ONEX Effect node pattern:
    - Receives registration requests (from Reducer intents)
    - Executes I/O operations against PostgreSQL
    - Returns structured responses with backend results
    - Supports partial failure handling and targeted retries

Partial Failure Handling:
    The effect tracks which backends have already succeeded using an
    IdempotencyStore. The default in-memory store has:
    - Bounded size (max 10,000 entries by default) with LRU eviction
    - TTL-based expiration (1 hour by default)
    - NOT persistent across restarts
    - NOT suitable for distributed deployments

    For production distributed deployments, inject a persistent
    ProtocolEffectIdempotencyStore implementation.

    Memory Characteristics (default config):
        - Max entries: 10,000
        - Per-entry size: ~100 bytes
        - Max memory: ~1MB

Circuit Breaker Integration:
    Backend clients should implement MixinAsyncCircuitBreaker for
    fault tolerance. The Effect propagates circuit breaker errors
    as backend failures.

Related:
    - ModelRegistryRequest: Input request model
    - ModelRegistryResponse: Output response model
    - ModelBackendResult: Per-backend result model
    - ModelEffectIdempotencyConfig: Idempotency store configuration
    - StoreEffectIdempotencyInmemory: Default bounded cache implementation
    - ProtocolEffectIdempotencyStore: Protocol for pluggable backends
    - RegistrationReducer: Emits intents consumed by this Effect
    - OMN-954: Partial failure scenario testing
    - OMN-3540: Remove Consul entirely from omnibase_infra runtime
"""

from __future__ import annotations

import time
from datetime import UTC, datetime
from uuid import UUID

from omnibase_infra.errors import (
    InfraAuthenticationError,
    InfraConnectionError,
    InfraTimeoutError,
)
from omnibase_infra.models.model_backend_result import (
    ModelBackendResult,
)
from omnibase_infra.nodes.node_registry_effect.models.model_effect_idempotency_config import (
    ModelEffectIdempotencyConfig,
)
from omnibase_infra.nodes.node_registry_effect.models.model_registry_request import (
    ModelRegistryRequest,
)
from omnibase_infra.nodes.node_registry_effect.models.model_registry_response import (
    ModelRegistryResponse,
)
from omnibase_infra.nodes.node_registry_effect.protocols.protocol_effect_idempotency_store import (
    ProtocolEffectIdempotencyStore,
)
from omnibase_infra.nodes.node_registry_effect.protocols.protocol_postgres_adapter import (
    ProtocolPostgresAdapter,
)
from omnibase_infra.nodes.node_registry_effect.store_effect_idempotency_inmemory import (
    StoreEffectIdempotencyInmemory,
)
from omnibase_infra.utils import sanitize_backend_error, sanitize_error_message


class NodeRegistryEffect:
    """Effect node for PostgreSQL node registration.

    Executes registration operations against PostgreSQL, with support for
    partial failure handling and targeted retries.

    Idempotency Store:
        Uses a pluggable idempotency store for tracking completed backends.
        The default StoreEffectIdempotencyInmemory provides:
        - Bounded size with LRU eviction (default 10,000 entries)
        - TTL-based expiration (default 1 hour)
        - ~1MB max memory at default settings

        WARNING: The default in-memory store:
        - Does NOT persist across restarts
        - Does NOT work in distributed/multi-instance scenarios
        - Is suitable only for single-instance deployments or testing

        For production distributed deployments, inject a persistent
        ProtocolEffectIdempotencyStore implementation backed by
        Redis, PostgreSQL, or similar.

    Memory Characteristics:
        Default configuration (~1MB total):
        - Max entries: 10,000 correlation IDs
        - Per-entry: ~100 bytes (UUID + backend set + timestamps)
        - Scales linearly: 100K entries = ~10MB

    Performance Characteristics:
        Idempotency checks are O(1):
        - Backend lookup/update: O(1) amortized
        - LRU eviction: O(1) per evicted entry
        - Throughput: >5,000 ops/sec (single worker), >10,000 concurrent

        Actual registration latency dominated by backend I/O:
        - PostgreSQL upsert: typically 1-5ms (network dependent)
        - Idempotency overhead: <0.1ms

    Coroutine Safety:
        This class is async-safe. The underlying idempotency store
        uses asyncio.Lock for coroutine-safe operations.

    Attributes:
        postgres_handler: Handler for PostgreSQL record persistence.
        idempotency_store: Store for tracking completed backends.

    Example:
        >>> from unittest.mock import AsyncMock
        >>> postgres = AsyncMock()
        >>> effect = NodeRegistryEffect(postgres)
        >>> # Configure mocks and call register_node...

        >>> # With custom idempotency config (smaller cache, shorter TTL):
        >>> from omnibase_infra.nodes.node_registry_effect.models import ModelEffectIdempotencyConfig
        >>> config = ModelEffectIdempotencyConfig(
        ...     max_cache_size=1000,
        ...     cache_ttl_seconds=300.0,
        ... )
        >>> effect = NodeRegistryEffect(postgres, idempotency_config=config)

    See Also:
        - README.md: Comprehensive documentation with configuration guide
        - StoreEffectIdempotencyInmemory: Default store implementation details
        - ProtocolEffectIdempotencyStore: Protocol for custom backends
    """

    def __init__(
        self,
        postgres_adapter: ProtocolPostgresAdapter,
        *,
        idempotency_store: ProtocolEffectIdempotencyStore | None = None,
        idempotency_config: ModelEffectIdempotencyConfig | None = None,
    ) -> None:
        """Initialize the NodeRegistryEffect with backend client.

        Args:
            postgres_adapter: Adapter for PostgreSQL record persistence.
            idempotency_store: Optional custom idempotency store.
                If provided, idempotency_config is ignored.
            idempotency_config: Optional configuration for the default
                in-memory idempotency store. Ignored if idempotency_store
                is provided.

        Memory Characteristics (default config):
            - Max entries: 10,000
            - Per-entry size: ~100 bytes
            - Max memory: ~1MB
            - TTL: 1 hour
        """
        self._postgres_adapter = postgres_adapter

        # Use provided store or create default with optional config
        if idempotency_store is not None:
            self._idempotency_store: ProtocolEffectIdempotencyStore = idempotency_store
        else:
            self._idempotency_store = StoreEffectIdempotencyInmemory(
                config=idempotency_config
            )

    async def register_node(
        self,
        request: ModelRegistryRequest,
        *,
        skip_postgres: bool = False,
    ) -> ModelRegistryResponse:
        """Execute node registration against PostgreSQL.

        Registers the node in PostgreSQL (registration record). Supports
        partial failure scenarios.

        Idempotency:
            If the backend has already succeeded for this correlation_id,
            it will be skipped on retry. This enables safe retries after
            partial failures.

            The idempotency store has bounded memory (default 10K entries, ~1MB)
            and TTL-based expiration (default 1 hour). Long-running operations
            may see entries expire. For production, consider:
            - Shorter operation durations than TTL
            - Larger cache size for high-volume scenarios
            - Persistent store for distributed deployments

        Args:
            request: Registration request with node details.
            skip_postgres: If True, skip PostgreSQL upsert (for retry scenarios).

        Returns:
            ModelRegistryResponse with backend result and overall status.
        """
        correlation_id = request.correlation_id

        # Check for already-completed backends (idempotency)
        completed = await self._idempotency_store.get_completed_backends(correlation_id)

        # Execute PostgreSQL upsert if not skipped and not already completed
        if skip_postgres or "postgres" in completed:
            postgres_result = ModelBackendResult(
                success=True,
                duration_ms=0.0,
                backend_id="postgres",
                correlation_id=correlation_id,
            )
        else:
            postgres_result = await self._upsert_postgres(request)
            if postgres_result.success:
                await self._idempotency_store.mark_completed(correlation_id, "postgres")

        return ModelRegistryResponse.from_backend_results(
            node_id=request.node_id,
            correlation_id=correlation_id,
            postgres_result=postgres_result,
            timestamp=datetime.now(UTC),
        )

    async def _upsert_postgres(
        self,
        request: ModelRegistryRequest,
    ) -> ModelBackendResult:
        """Execute PostgreSQL registration record upsert.

        Args:
            request: Registration request with node details.

        Returns:
            ModelBackendResult with operation outcome.
        """
        start_time = time.perf_counter()

        try:
            result = await self._postgres_adapter.upsert(
                node_id=request.node_id,
                node_type=request.node_type,
                node_version=request.node_version,
                endpoints=request.endpoints,
                metadata=request.metadata,
            )

            duration_ms = (time.perf_counter() - start_time) * 1000

            if result.success:
                return ModelBackendResult(
                    success=True,
                    duration_ms=duration_ms,
                    backend_id="postgres",
                    correlation_id=request.correlation_id,
                )
            else:
                # Sanitize backend error to avoid exposing secrets
                # (connection strings, credentials, internal hostnames)
                sanitized_error = sanitize_backend_error("postgres", result.error)
                return ModelBackendResult(
                    success=False,
                    error=sanitized_error,
                    error_code="POSTGRES_UPSERT_ERROR",
                    duration_ms=duration_ms,
                    backend_id="postgres",
                    correlation_id=request.correlation_id,
                )

        except (TimeoutError, InfraTimeoutError) as e:
            # Timeout during upsert - retriable error
            duration_ms = (time.perf_counter() - start_time) * 1000
            sanitized_error = sanitize_error_message(e)
            return ModelBackendResult(
                success=False,
                error=sanitized_error,
                error_code="POSTGRES_TIMEOUT_ERROR",
                duration_ms=duration_ms,
                backend_id="postgres",
                correlation_id=request.correlation_id,
            )

        except InfraAuthenticationError as e:
            # Authentication failure - non-retriable error
            duration_ms = (time.perf_counter() - start_time) * 1000
            sanitized_error = sanitize_error_message(e)
            return ModelBackendResult(
                success=False,
                error=sanitized_error,
                error_code="POSTGRES_AUTH_ERROR",
                duration_ms=duration_ms,
                backend_id="postgres",
                correlation_id=request.correlation_id,
            )

        except InfraConnectionError as e:
            # Connection failure - retriable error
            duration_ms = (time.perf_counter() - start_time) * 1000
            sanitized_error = sanitize_error_message(e)
            return ModelBackendResult(
                success=False,
                error=sanitized_error,
                error_code="POSTGRES_CONNECTION_ERROR",
                duration_ms=duration_ms,
                backend_id="postgres",
                correlation_id=request.correlation_id,
            )

        except Exception as e:  # noqa: BLE001 — boundary: returns degraded response
            # Unknown exception - sanitize to prevent credential exposure
            duration_ms = (time.perf_counter() - start_time) * 1000
            sanitized_error = sanitize_error_message(e)
            return ModelBackendResult(
                success=False,
                error=sanitized_error,
                error_code="POSTGRES_UNKNOWN_ERROR",
                duration_ms=duration_ms,
                backend_id="postgres",
                correlation_id=request.correlation_id,
            )

    async def clear_completed_backends(self, correlation_id: UUID) -> None:
        """Clear completed backends cache for a correlation ID.

        Used for testing or to force re-registration.

        Args:
            correlation_id: The correlation ID to clear.
        """
        await self._idempotency_store.clear(correlation_id)

    async def get_completed_backends(self, correlation_id: UUID) -> set[str]:
        """Get the set of completed backends for a correlation ID.

        Args:
            correlation_id: The correlation ID to check.

        Returns:
            Set of backend names that have completed ("postgres").
        """
        return await self._idempotency_store.get_completed_backends(correlation_id)


__all__ = [
    "NodeRegistryEffect",
]
