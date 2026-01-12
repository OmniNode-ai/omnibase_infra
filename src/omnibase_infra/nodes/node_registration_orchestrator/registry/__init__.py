# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Registry module for NodeRegistrationOrchestrator handler wiring.

This module provides the handler registry for the NodeRegistrationOrchestrator,
enabling a static factory method for creating frozen handler registries.

Handlers Wired:
    - HandlerNodeIntrospected: Processes NodeIntrospectionEvent (registration trigger)
    - HandlerRuntimeTick: Processes RuntimeTick (timeout detection)
    - HandlerNodeRegistrationAcked: Processes NodeRegistrationAcked (ack processing)
    - HandlerNodeHeartbeat: Processes NodeHeartbeatEvent (liveness tracking)

Usage:
    ```python
    from omnibase_infra.nodes.node_registration_orchestrator.registry import (
        RegistryInfraNodeRegistrationOrchestrator,
    )

    registry = RegistryInfraNodeRegistrationOrchestrator.create_registry(
        projection_reader=reader,
        projector=projector,
        consul_handler=consul_handler,
    )
    # registry is frozen and thread-safe

    handler = registry.get_handler_by_id("handler-node-introspected")
    result = await handler.handle(envelope)
    ```

Related:
    - contract.yaml: Defines handler_routing with event-to-handler mappings
    - handlers/: Handler implementations
    - OMN-1102: Make NodeRegistrationOrchestrator fully declarative
"""

from omnibase_infra.nodes.node_registration_orchestrator.registry.registry_infra_node_registration_orchestrator import (
    RegistryInfraNodeRegistrationOrchestrator,
)

__all__ = ["RegistryInfraNodeRegistrationOrchestrator"]
