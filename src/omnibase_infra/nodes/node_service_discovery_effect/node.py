# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Service Discovery Effect Node - Capability-Oriented Effect Node.

This node provides capability-oriented service discovery operations following
the ONEX principle: "I'm interested in what you do, not what you are".

Capability-Oriented Design:
    Named by capability (service.discovery) not vendor (consul/k8s/etcd).
    The node provides unified operations for:
    - register_service: Register a service instance
    - deregister_service: Remove a service registration
    - discover_services: Query for service instances
    - health_check: Verify backend connectivity

Pluggable Backend Support:
    This node delegates actual service discovery operations to implementations
    of ProtocolDiscoveryOperations. Supported backends include:
    - Consul: HashiCorp Consul service discovery
    - Kubernetes: K8s native service discovery
    - Etcd: CoreOS etcd service discovery

Declarative Pattern:
    This node follows ONEX declarative conventions:
    - Extends NodeEffect from omnibase_core
    - All behavior defined in contract.yaml
    - No custom routing logic in Python code
    - Handler wired externally via container injection

Usage:
    .. code-block:: python

        from omnibase_core.models.container import ModelONEXContainer
        from omnibase_infra.nodes.node_service_discovery_effect import (
            NodeServiceDiscoveryEffect,
        )

        # Create container with handler dependency
        container = ModelONEXContainer()
        container.register(
            ProtocolDiscoveryOperations,
            consul_handler_instance,
        )

        # Create node
        node = NodeServiceDiscoveryEffect(container)

Related:
    - contract.yaml: ONEX contract with capabilities and I/O definitions
    - ProtocolDiscoveryOperations: Backend handler protocol
    - ModelServiceRegistration: Input model for registration
    - ModelDiscoveryResult: Output model for discovery queries
    - OMN-1131: Capability-oriented node architecture
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from omnibase_core.nodes.node_effect import NodeEffect

if TYPE_CHECKING:
    from omnibase_core.models.container.model_onex_container import ModelONEXContainer


class NodeServiceDiscoveryEffect(NodeEffect):
    """Effect node for service discovery operations.

    Capability: service.discovery
    Supports pluggable handlers: Consul, Kubernetes, Etcd

    This node provides capability-oriented service discovery following
    the ONEX principle of naming by capability rather than vendor.

    All behavior is declarative and defined in contract.yaml:
    - io_operations: register_service, deregister_service, discover_services
    - capabilities: service.discovery, service.registration, etc.
    - dependencies: ProtocolDiscoveryOperations

    Attributes:
        container: ONEX dependency injection container providing access
            to the configured ProtocolDiscoveryOperations implementation.

    Example:
        .. code-block:: python

            from omnibase_core.models.container import ModelONEXContainer

            container = ModelONEXContainer()
            # Register handler implementation (e.g., Consul, K8s, Etcd)
            container.register(handler_protocol, handler_impl)

            node = NodeServiceDiscoveryEffect(container)

    Note:
        This is a declarative node - all behavior is driven by contract.yaml.
        The handler implementation is resolved from the container at runtime.
    """

    def __init__(self, container: ModelONEXContainer) -> None:
        """Initialize the service discovery effect node.

        Args:
            container: ONEX dependency injection container providing
                access to infrastructure dependencies including the
                configured ProtocolDiscoveryOperations implementation.
        """
        super().__init__(container)


__all__ = ["NodeServiceDiscoveryEffect"]
