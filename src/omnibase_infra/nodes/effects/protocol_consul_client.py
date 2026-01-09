# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Protocol for Consul service registration client.

This module defines the protocol that Consul clients must implement
to be used with the NodeRegistryEffect node.

Concurrency Safety:
    Implementations MUST be safe for concurrent async calls.
    Multiple coroutines may invoke register_service() simultaneously
    for different or identical service registrations. Implementations
    should use asyncio.Lock for coroutine-safety when protecting shared state.

Related:
    - NodeRegistryEffect: Effect node that uses this protocol
    - ProtocolPostgresAdapter: Protocol for PostgreSQL backend
    - ModelBackendResult: Structured result model for backend operations
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from omnibase_infra.nodes.effects.models import ModelBackendResult


@runtime_checkable
class ProtocolConsulClient(Protocol):
    """Protocol for Consul service registration client.

    Implementations must provide async service registration capability.

    Concurrency Safety:
        Implementations MUST be safe for concurrent async coroutine calls.

        **Guarantees implementers MUST provide:**
            - Concurrent register_service() calls are coroutine-safe
            - Connection pooling (if used) is async-safe
            - Internal state (if any) is protected by asyncio.Lock

        **What callers can assume:**
            - Multiple coroutines can call register_service() concurrently
            - Each registration operation is independent
            - Failures in one registration do not affect others

        Note: asyncio.Lock provides coroutine-safety, not thread-safety.
    """

    async def register_service(
        self,
        service_id: str,
        service_name: str,
        tags: list[str],
        health_check: dict[str, str] | None = None,
    ) -> ModelBackendResult:
        """Register a service in Consul.

        Args:
            service_id: Unique identifier for the service instance.
            service_name: Name of the service for discovery.
            tags: List of tags for filtering.
            health_check: Optional health check configuration.

        Returns:
            ModelBackendResult with success status, optional error message,
            timing information, and correlation context.
        """
        ...

    async def deregister_service(
        self,
        service_id: str,
    ) -> ModelBackendResult:
        """Deregister a service from Consul.

        Removes the service registration, stopping health checks and
        removing it from service discovery.

        Args:
            service_id: Unique identifier for the service instance to remove.

        Returns:
            ModelBackendResult with success status, optional error message,
            timing information, and correlation context.
        """
        ...


__all__ = ["ProtocolConsulClient"]
