# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Protocol for Service Discovery Handler.

This module defines the protocol that service discovery handlers must implement
to be used with capability-oriented nodes.

Concurrency Safety:
    Implementations MUST be safe for concurrent async calls.
    Multiple coroutines may invoke methods simultaneously.
    Implementations should use asyncio.Lock for coroutine-safety
    when protecting shared state.

Related:
    - NodeServiceDiscoveryEffect: Effect node that uses this protocol
    - ConsulServiceDiscoveryHandler: Consul implementation
    - MockServiceDiscoveryHandler: In-memory mock for testing
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable
from uuid import UUID

from omnibase_infra.handlers.service_discovery.models import (
    ModelDiscoveryResult,
    ModelRegistrationResult,
    ModelServiceInfo,
)


@runtime_checkable
class ProtocolServiceDiscoveryHandler(Protocol):
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
    ) -> ModelRegistrationResult:
        """Register a service with the discovery backend.

        Args:
            service_info: Service information to register.
            correlation_id: Optional correlation ID for tracing.

        Returns:
            ModelRegistrationResult with success status, service ID,
            and operation metadata.

        Raises:
            InfraConnectionError: If connection to backend fails.
            InfraTimeoutError: If operation times out.
            InfraUnavailableError: If circuit breaker is open.
        """
        ...

    async def deregister_service(
        self,
        service_id: str,
        correlation_id: UUID | None = None,
    ) -> None:
        """Deregister a service from the discovery backend.

        Args:
            service_id: ID of the service to deregister.
            correlation_id: Optional correlation ID for tracing.

        Raises:
            InfraConnectionError: If connection to backend fails.
            InfraTimeoutError: If operation times out.
            InfraUnavailableError: If circuit breaker is open.
        """
        ...

    async def discover_services(
        self,
        service_name: str,
        tags: tuple[str, ...] | None = None,
        correlation_id: UUID | None = None,
    ) -> ModelDiscoveryResult:
        """Discover services matching the given criteria.

        Args:
            service_name: Name of the service to discover.
            tags: Optional tags to filter services.
            correlation_id: Optional correlation ID for tracing.

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
    ) -> dict[str, object]:
        """Perform a health check on the handler.

        Args:
            correlation_id: Optional correlation ID for tracing.

        Returns:
            Dict with health status information.
        """
        ...


__all__ = ["ProtocolServiceDiscoveryHandler"]
