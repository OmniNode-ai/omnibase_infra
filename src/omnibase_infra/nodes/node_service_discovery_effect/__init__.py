# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Service Discovery Effect Node - Capability-Oriented Service Discovery.

This module provides NodeServiceDiscoveryEffect, a capability-oriented effect
node for service discovery operations with pluggable backend support.

Capability-Oriented Design:
    Named by capability (service.discovery) not vendor (consul/k8s/etcd).
    Following the ONEX principle: "I'm interested in what you do, not what you are"

Supported Operations:
    - register_service: Register a service with the backend
    - deregister_service: Remove a service registration
    - discover_services: Query for service instances
    - health_check: Verify backend connectivity

Pluggable Backends:
    - Consul: HashiCorp Consul service discovery
    - Kubernetes: K8s native service discovery
    - Etcd: CoreOS etcd service discovery

Node:
    NodeServiceDiscoveryEffect: Declarative effect node for service discovery.

Models:
    ModelServiceRegistration: Input model for registration operations.
    ModelServiceInfo: Information about a discovered service.
    ModelDiscoveryQuery: Query parameters for discovery operations.
    ModelDiscoveryResult: Result of discovery queries.
    ModelRegistrationResult: Result of registration/deregistration.

Enums:
    EnumHealthStatus: Health status values (HEALTHY, UNHEALTHY, UNKNOWN).

Protocol:
    ProtocolDiscoveryOperations: Protocol for pluggable backends.

Registry:
    RegistryInfraServiceDiscovery: DI registry for node dependencies.

Example:
    .. code-block:: python

        from omnibase_core.models.container import ModelONEXContainer
        from omnibase_infra.nodes.node_service_discovery_effect import (
            NodeServiceDiscoveryEffect,
            ModelServiceRegistration,
            RegistryInfraServiceDiscovery,
        )

        # Setup container with handler
        container = ModelONEXContainer()
        RegistryInfraServiceDiscovery.register_with_handler(
            container,
            handler=consul_handler,
        )

        # Create node
        node = NodeServiceDiscoveryEffect(container)

        # Register a service
        registration = ModelServiceRegistration(
            service_name="user-service",
            address="localhost",
            port=8080,
            tags=("api", "v2"),
        )

Related:
    - contract.yaml: ONEX contract with capabilities and I/O definitions
    - OMN-1131: Capability-oriented node architecture

.. versionadded:: 0.6.0
"""

from .models import (
    EnumHealthStatus,
    ModelDiscoveryQuery,
    ModelDiscoveryResult,
    ModelQueryMetadata,
    ModelRegistrationResult,
    ModelServiceInfo,
    ModelServiceRegistration,
)
from .node import (
    NodeServiceDiscoveryEffect,
)
from .protocols import (
    ProtocolDiscoveryOperations,
)
from .registry import (
    RegistryInfraServiceDiscovery,
)

__all__ = [
    # Node
    "NodeServiceDiscoveryEffect",
    # Models
    "EnumHealthStatus",
    "ModelDiscoveryQuery",
    "ModelDiscoveryResult",
    "ModelQueryMetadata",
    "ModelRegistrationResult",
    "ModelServiceInfo",
    "ModelServiceRegistration",
    # Protocol
    "ProtocolDiscoveryOperations",
    # Registry
    "RegistryInfraServiceDiscovery",
]
