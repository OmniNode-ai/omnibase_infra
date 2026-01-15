# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Protocol for Service Discovery Handler.

This module defines the protocol that service discovery handlers must implement
to be used with capability-oriented nodes.

Protocol Duplication Note:
    This protocol is intentionally separate from the node-level protocol
    at ``nodes/node_service_discovery_effect/protocols/protocol_discovery_operations.py``:

    - **This protocol (handlers/*/protocol_*.py)**: Handler implementation
      contract used by tests for compliance verification. Uses
      ``ModelServiceInfo`` with optional ``correlation_id``.

    - **Node protocol (nodes/*/protocols/)**: Node-level contract used for
      container-based dependency injection and registry binding.

    The separation allows different parameter models and independent evolution
    of handler contracts vs node-level DI contracts.

Concurrency Safety:
    Implementations MUST be safe for concurrent async calls.
    Multiple coroutines may invoke methods simultaneously.
    Implementations should use asyncio.Lock for coroutine-safety
    when protecting shared state.

Related:
    - NodeServiceDiscoveryEffect: Effect node that uses this protocol
    - HandlerServiceDiscoveryConsul: Consul implementation
    - HandlerServiceDiscoveryMock: In-memory mock for testing
    - nodes/node_service_discovery_effect/protocols/: Node-level protocol
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable
from uuid import UUID

from omnibase_infra.handlers.service_discovery.models import (
    ModelDiscoveryResult,
    ModelHandlerRegistrationResult,
    ModelServiceInfo,
)
from omnibase_infra.nodes.node_service_discovery_effect.models import (
    ModelDiscoveryQuery,
    ModelServiceDiscoveryHealthCheckResult,
)


@runtime_checkable
class ProtocolDiscoveryOperations(Protocol):
    """Protocol for service discovery handler implementations.

    Defines the interface that all service discovery handlers must implement.
    Handlers are responsible for service registration, deregistration,
    discovery, and health checking.

    Concurrency Safety:
        Implementations MUST be safe for concurrent async coroutine calls.

        **Guarantees implementers MUST provide:**
            - Concurrent method calls are coroutine-safe
            - Connection pooling (if used) is async-safe
            - Internal state (if any) is protected by asyncio.Lock

        **What callers can assume:**
            - Multiple coroutines can call methods concurrently
            - Each operation is independent
            - Failures in one operation do not affect others

        Note: asyncio.Lock provides coroutine-safety, not thread-safety.
    """

    @property
    def handler_type(self) -> str:
        """Return the handler type identifier.

        Returns:
            Handler type string (e.g., "consul", "mock").
        """
        ...

    async def register_service(
        self,
        service_info: ModelServiceInfo,
        correlation_id: UUID | None = None,
    ) -> ModelHandlerRegistrationResult:
        """Register a service with the discovery backend.

        Args:
            service_info: Service information to register.
            correlation_id: Optional correlation ID for tracing.

        Returns:
            ModelHandlerRegistrationResult with success status, service ID,
            and operation metadata.

        Raises:
            InfraConnectionError: If connection to backend fails.
            InfraTimeoutError: If operation times out.
            InfraUnavailableError: If circuit breaker is open.
        """
        ...

    async def deregister_service(
        self,
        service_id: UUID,
        correlation_id: UUID | None = None,
    ) -> None:
        """Deregister a service from the discovery backend.

        Args:
            service_id: UUID of the service to deregister.
            correlation_id: Optional correlation ID for tracing.

        Raises:
            InfraConnectionError: If connection to backend fails.
            InfraTimeoutError: If operation times out.
            InfraUnavailableError: If circuit breaker is open.
        """
        ...

    async def discover_services(
        self,
        query: ModelDiscoveryQuery,
        correlation_id: UUID | None = None,
    ) -> ModelDiscoveryResult:
        """Discover services matching the query criteria.

        Args:
            query: Query parameters including service_name, tags,
                and health_filter for filtering services.
            correlation_id: Optional correlation ID for tracing.
                If not provided, uses query.correlation_id.

        Returns:
            ModelDiscoveryResult with list of matching services
            and operation metadata.

        Raises:
            InfraConnectionError: If connection to backend fails.
            InfraTimeoutError: If operation times out.
            InfraUnavailableError: If circuit breaker is open.
        """
        ...

    async def health_check(
        self,
        correlation_id: UUID | None = None,
    ) -> ModelServiceDiscoveryHealthCheckResult:
        """Perform a health check on the handler.

        Args:
            correlation_id: Optional correlation ID for tracing.

        Returns:
            ModelServiceDiscoveryHealthCheckResult: Health status including:
                - healthy: bool indicating overall health
                - backend_type: str identifying the backend
                - latency_ms: float connection latency
                - reason: str explaining the health status
                - error_type: str | None exception type if failed
                - details: ModelServiceDiscoveryHealthCheckDetails with typed diagnostics
                - correlation_id: UUID | None for tracing
        """
        ...


__all__ = ["ProtocolDiscoveryOperations"]
