# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Registry Effect Node for Dual-Backend Registration.

This module provides RegistryEffect, an Effect node responsible for executing
registration operations against both Consul and PostgreSQL backends.

Architecture:
    RegistryEffect follows the ONEX Effect node pattern:
    - Receives registration requests (from Reducer intents)
    - Executes I/O operations against external backends
    - Returns structured responses with per-backend results
    - Supports partial failure handling and targeted retries

    The Effect node coordinates with:
    - ConsulClient: For service discovery registration
    - PostgresHandler: For registration record persistence

Partial Failure Handling:
    When one backend fails:
    1. The successful backend's result is preserved
    2. The failed backend's error is captured with context
    3. Response status is set to "partial"
    4. Callers can retry only the failed backend

Idempotency:
    For retry scenarios, the Effect tracks which backends have already
    succeeded. If called with an already-completed backend, it skips
    that backend and only executes the pending operation.

Circuit Breaker Integration:
    Backend clients should implement MixinAsyncCircuitBreaker for
    fault tolerance. The Effect propagates circuit breaker errors
    as backend failures.

Related:
    - ModelRegistryRequest: Input request model
    - ModelRegistryResponse: Output response model
    - ModelBackendResult: Per-backend result model
    - RegistrationReducer: Emits intents consumed by this Effect
    - OMN-954: Partial failure scenario testing
"""

from __future__ import annotations

import time
from uuid import UUID

from omnibase_infra.nodes.effects.models.model_backend_result import (
    ModelBackendResult,
)
from omnibase_infra.nodes.effects.models.model_registry_request import (
    ModelRegistryRequest,
)
from omnibase_infra.nodes.effects.models.model_registry_response import (
    ModelRegistryResponse,
)
from omnibase_infra.nodes.effects.protocol_consul_client import ProtocolConsulClient
from omnibase_infra.nodes.effects.protocol_postgres_adapter import (
    ProtocolPostgresAdapter,
)


class RegistryEffect:
    """Effect node for dual-backend node registration.

    Executes registration operations against both Consul and PostgreSQL,
    with support for partial failure handling and targeted retries.

    Thread Safety:
        This class is NOT thread-safe. Each instance should be used
        from a single async context. For concurrent use, create
        separate instances.

    Attributes:
        consul_client: Client for Consul service registration.
        postgres_handler: Handler for PostgreSQL record persistence.

    Example:
        >>> from unittest.mock import AsyncMock
        >>> consul = AsyncMock()
        >>> postgres = AsyncMock()
        >>> effect = RegistryEffect(consul, postgres)
        >>> # Configure mocks and call register_node...
    """

    def __init__(
        self,
        consul_client: ProtocolConsulClient,
        postgres_adapter: ProtocolPostgresAdapter,
    ) -> None:
        """Initialize the RegistryEffect with backend clients.

        Args:
            consul_client: Client for Consul service registration.
            postgres_adapter: Adapter for PostgreSQL record persistence.
        """
        self._consul_client = consul_client
        self._postgres_adapter = postgres_adapter
        # Track completed backends for idempotent retries
        self._completed_backends: dict[UUID, set[str]] = {}

    async def register_node(
        self,
        request: ModelRegistryRequest,
        *,
        skip_consul: bool = False,
        skip_postgres: bool = False,
    ) -> ModelRegistryResponse:
        """Execute dual-backend node registration.

        Registers the node in both Consul (service discovery) and PostgreSQL
        (registration record) backends. Supports partial failure scenarios
        where one backend succeeds and the other fails.

        Idempotency:
            If a backend has already succeeded for this correlation_id,
            it will be skipped on retry. This enables safe retries after
            partial failures.

        Args:
            request: Registration request with node details.
            skip_consul: If True, skip Consul registration (for retry scenarios).
            skip_postgres: If True, skip PostgreSQL upsert (for retry scenarios).

        Returns:
            ModelRegistryResponse with per-backend results and overall status.
        """
        start_time = time.perf_counter()
        correlation_id = request.correlation_id

        # Check for already-completed backends (idempotency)
        completed = self._completed_backends.get(correlation_id, set())

        # Execute Consul registration if not skipped and not already completed
        if skip_consul or "consul" in completed:
            consul_result = ModelBackendResult(
                success=True,
                duration_ms=0.0,
                retries=0,
                correlation_id=correlation_id,
            )
        else:
            consul_result = await self._register_consul(request)
            if consul_result.success:
                completed.add("consul")

        # Execute PostgreSQL upsert if not skipped and not already completed
        if skip_postgres or "postgres" in completed:
            postgres_result = ModelBackendResult(
                success=True,
                duration_ms=0.0,
                retries=0,
                correlation_id=correlation_id,
            )
        else:
            postgres_result = await self._upsert_postgres(request)
            if postgres_result.success:
                completed.add("postgres")

        # Update completed backends cache
        if completed:
            self._completed_backends[correlation_id] = completed

        return ModelRegistryResponse.from_backend_results(
            node_id=request.node_id,
            correlation_id=correlation_id,
            consul_result=consul_result,
            postgres_result=postgres_result,
        )

    async def _register_consul(
        self,
        request: ModelRegistryRequest,
    ) -> ModelBackendResult:
        """Execute Consul service registration.

        Args:
            request: Registration request with node details.

        Returns:
            ModelBackendResult with operation outcome.
        """
        start_time = time.perf_counter()
        retries = 0

        try:
            service_id = f"node-{request.node_type}-{request.node_id}"
            service_name = request.service_name or f"onex-{request.node_type}"

            result = await self._consul_client.register_service(
                service_id=service_id,
                service_name=service_name,
                tags=request.tags,
                health_check=request.health_check_config,
            )

            duration_ms = (time.perf_counter() - start_time) * 1000

            if result.get("success", False):
                return ModelBackendResult(
                    success=True,
                    duration_ms=duration_ms,
                    retries=retries,
                    backend_id="consul",
                    correlation_id=request.correlation_id,
                )
            else:
                return ModelBackendResult(
                    success=False,
                    error=str(result.get("error", "Unknown Consul error")),
                    error_code="CONSUL_REGISTRATION_ERROR",
                    duration_ms=duration_ms,
                    retries=retries,
                    backend_id="consul",
                    correlation_id=request.correlation_id,
                )

        except Exception as e:
            duration_ms = (time.perf_counter() - start_time) * 1000
            return ModelBackendResult(
                success=False,
                error=str(e),
                error_code="CONSUL_CONNECTION_ERROR",
                duration_ms=duration_ms,
                retries=retries,
                backend_id="consul",
                correlation_id=request.correlation_id,
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
        retries = 0

        try:
            result = await self._postgres_adapter.upsert(
                node_id=request.node_id,
                node_type=request.node_type,
                node_version=request.node_version,
                endpoints=request.endpoints,
                metadata=request.metadata,
            )

            duration_ms = (time.perf_counter() - start_time) * 1000

            if result.get("success", False):
                return ModelBackendResult(
                    success=True,
                    duration_ms=duration_ms,
                    retries=retries,
                    backend_id="postgres",
                    correlation_id=request.correlation_id,
                )
            else:
                return ModelBackendResult(
                    success=False,
                    error=str(result.get("error", "Unknown PostgreSQL error")),
                    error_code="POSTGRES_UPSERT_ERROR",
                    duration_ms=duration_ms,
                    retries=retries,
                    backend_id="postgres",
                    correlation_id=request.correlation_id,
                )

        except Exception as e:
            duration_ms = (time.perf_counter() - start_time) * 1000
            return ModelBackendResult(
                success=False,
                error=str(e),
                error_code="POSTGRES_CONNECTION_ERROR",
                duration_ms=duration_ms,
                retries=retries,
                backend_id="postgres",
                correlation_id=request.correlation_id,
            )

    def clear_completed_backends(self, correlation_id: UUID) -> None:
        """Clear completed backends cache for a correlation ID.

        Used for testing or to force re-registration.

        Args:
            correlation_id: The correlation ID to clear.
        """
        self._completed_backends.pop(correlation_id, None)

    def get_completed_backends(self, correlation_id: UUID) -> set[str]:
        """Get the set of completed backends for a correlation ID.

        Args:
            correlation_id: The correlation ID to check.

        Returns:
            Set of backend names that have completed ("consul", "postgres").
        """
        return self._completed_backends.get(correlation_id, set()).copy()


__all__ = [
    "RegistryEffect",
]
