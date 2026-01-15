# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Protocol for Service Discovery Operations.

This module defines ProtocolDiscoveryOperations, the protocol for
pluggable service discovery backends in the NodeServiceDiscoveryEffect node.

Architecture:
    ProtocolDiscoveryOperations defines the interface for service
    discovery backends. Implementations include:
    - Consul: HashiCorp Consul service discovery
    - Kubernetes: K8s native service discovery
    - Etcd: CoreOS etcd service discovery

    This protocol enables the capability-oriented design principle:
    "I'm interested in what you do, not what you are"

Protocol Duplication Note:
    This protocol is intentionally separate from the handler-level protocol
    at ``handlers/service_discovery/protocol_discovery_operations.py``:

    - **This protocol (nodes/*/protocols/)**: Node-level contract used for
      container-based dependency injection and registry binding. Uses
      ``ModelServiceRegistration`` with required ``correlation_id``.

    - **Handler protocol (handlers/*/protocol_*.py)**: Handler implementation
      contract used by tests for compliance verification. Uses
      ``ModelServiceInfo`` with optional ``correlation_id``.

    The separation allows:
    1. Node-level protocols to evolve independently of handler contracts
    2. Different parameter models appropriate for each layer
    3. Clear ownership boundaries (nodes own their protocols)

    Both protocols share the same name for discoverability, but serve
    different architectural layers.

Protocol Verification:
    Per ONEX conventions, protocol compliance is verified via duck typing
    rather than isinstance checks. The @runtime_checkable decorator enables
    structural subtyping checks.

    However, registries use isinstance() for fail-fast validation at
    registration time. See registry_infra_service_discovery.py for rationale.

Thread Safety:
    Handler implementations may be invoked concurrently.
    Implementations should be designed for thread-safe operation.

Related:
    - NodeServiceDiscoveryEffect: Effect node that uses this protocol
    - ModelServiceRegistration: Input model for registration
    - ModelDiscoveryQuery: Input model for discovery queries
    - ModelDiscoveryResult: Output model for discovery results
    - ModelRegistrationResult: Output model for registration
    - handlers/service_discovery/protocol_discovery_operations.py: Handler protocol
    - OMN-1131: Capability-oriented node architecture
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, runtime_checkable
from uuid import UUID

if TYPE_CHECKING:
    from omnibase_infra.nodes.node_service_discovery_effect.models import (
        ModelDiscoveryQuery,
        ModelDiscoveryResult,
        ModelRegistrationResult,
        ModelServiceDiscoveryHealthCheckResult,
        ModelServiceRegistration,
    )


@runtime_checkable
class ProtocolDiscoveryOperations(Protocol):
    """Protocol for service discovery backends.

    Defines the interface that all service discovery backend handlers
    must implement. This enables pluggable backends (Consul, Kubernetes,
    Etcd) while maintaining a unified capability-oriented interface.

    Protocol Verification:
        Per ONEX conventions, protocol compliance is verified via duck typing:

        .. code-block:: python

            # Verify required properties and methods exist
            if (hasattr(handler, 'handler_type') and
                hasattr(handler, 'register_service') and callable(handler.register_service)):
                registry.register_handler(handler)

    Attributes:
        handler_type: Type identifier for this handler (consul, kubernetes, etcd).

    Methods:
        register_service: Register a service with the backend.
        deregister_service: Remove a service registration.
        discover_services: Query for service instances.
        health_check: Verify backend connectivity.

    Example:
        .. code-block:: python

            class ConsulServiceDiscoveryHandler:
                '''Consul implementation of service discovery.'''

                @property
                def handler_type(self) -> str:
                    return "consul"

                async def register_service(
                    self,
                    registration: ModelServiceRegistration,
                    correlation_id: UUID,
                ) -> ModelRegistrationResult:
                    # Consul-specific registration logic
                    ...

                async def deregister_service(
                    self,
                    service_id: UUID,
                    correlation_id: UUID,
                ) -> None:
                    # Consul-specific deregistration logic
                    ...

                async def discover_services(
                    self,
                    query: ModelDiscoveryQuery,
                    correlation_id: UUID,
                ) -> ModelDiscoveryResult:
                    # Consul-specific discovery logic
                    ...

                async def health_check(
                    self,
                    correlation_id: UUID | None = None,
                ) -> ModelServiceDiscoveryHealthCheckResult:
                    # Check Consul agent connectivity
                    ...

    Note:
        Method bodies in this Protocol use ``...`` (Ellipsis) rather than
        ``raise NotImplementedError()``. This is the standard Python convention
        for ``typing.Protocol`` classes per PEP 544.

    .. versionadded:: 0.6.0
    """

    @property
    def handler_type(self) -> str:
        """Return the type identifier for this handler.

        The handler type identifies the backend implementation:
        - "consul": HashiCorp Consul
        - "kubernetes": Kubernetes native
        - "etcd": CoreOS etcd

        Returns:
            str: Handler type identifier.
        """
        ...

    async def register_service(
        self,
        registration: ModelServiceRegistration,
        correlation_id: UUID,
    ) -> ModelRegistrationResult:
        """Register a service with the backend.

        Creates or updates a service registration in the service
        discovery backend. The operation should be idempotent.

        Args:
            registration: Service registration details including
                service_id, service_name, address, port, tags, etc.
            correlation_id: Correlation ID for distributed tracing.

        Returns:
            ModelRegistrationResult with success status and any errors.

        Raises:
            InfraConnectionError: If connection to backend fails.
            InfraTimeoutError: If operation times out.
            InfraUnavailableError: If circuit breaker is open.
        """
        ...

    async def deregister_service(
        self,
        service_id: UUID,
        correlation_id: UUID,
    ) -> None:
        """Deregister a service from the backend.

        Removes a service registration from the service discovery
        backend. The operation should be idempotent (no error if
        service does not exist).

        Args:
            service_id: ID of the service to deregister.
            correlation_id: Correlation ID for distributed tracing.

        Raises:
            InfraConnectionError: If connection to backend fails.
            InfraTimeoutError: If operation times out.
            InfraUnavailableError: If circuit breaker is open.
        """
        ...

    async def discover_services(
        self,
        query: ModelDiscoveryQuery,
        correlation_id: UUID,
    ) -> ModelDiscoveryResult:
        """Discover services matching query criteria.

        Queries the service discovery backend for services matching
        the specified criteria (service_name, tags, health_filter).

        Args:
            query: Query parameters for filtering services.
            correlation_id: Correlation ID for distributed tracing.

        Returns:
            ModelDiscoveryResult with matching services and query metadata.

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
        """Check health of the service discovery backend.

        Verifies connectivity and availability of the service
        discovery backend.

        Args:
            correlation_id: Optional correlation ID for distributed tracing.
                If not provided, implementations should generate one.

        Returns:
            ModelServiceDiscoveryHealthCheckResult: Health status including:
                - healthy: bool indicating overall health
                - backend_type: str identifying the backend
                - latency_ms: float connection latency
                - reason: str explaining the health status
                - error_type: str | None exception type if failed
                - details: ModelServiceDiscoveryHealthCheckDetails with typed diagnostics
                - correlation_id: UUID | None for tracing

        Example:
            >>> from omnibase_infra.nodes.node_service_discovery_effect.models import (
            ...     ModelServiceDiscoveryHealthCheckDetails,
            ... )
            >>> health = await handler.health_check()
            >>> health
            ModelServiceDiscoveryHealthCheckResult(
                healthy=True,
                backend_type="consul",
                latency_ms=5.2,
                reason="ok",
                details=ModelServiceDiscoveryHealthCheckDetails(
                    agent_address="localhost:8500",
                    service_count=15,
                ),
            )
        """
        ...


__all__ = ["ProtocolDiscoveryOperations"]
