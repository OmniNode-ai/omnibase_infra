# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Registry module for NodeRegistrationOrchestrator handler wiring.

This module provides the handler registry for the NodeRegistrationOrchestrator,
enabling dependency injection and factory methods for all orchestrator handlers.

Handlers Wired:
    - HandlerNodeIntrospected: Processes NodeIntrospectionEvent (registration trigger)
    - HandlerRuntimeTick: Processes RuntimeTick (timeout detection)
    - HandlerNodeRegistrationAcked: Processes NodeRegistrationAcked (ack processing)
    - HandlerNodeHeartbeat: Processes NodeHeartbeatEvent (liveness tracking)

Usage:
    ```python
    from omnibase_core.models.container import ModelONEXContainer
    from omnibase_infra.nodes.node_registration_orchestrator.registry import (
        RegistryInfraNodeRegistrationOrchestrator,
    )

    container = ModelONEXContainer()
    registry = RegistryInfraNodeRegistrationOrchestrator(container)

    # Create individual handlers
    introspection_handler = registry.create_handler_node_introspected()
    tick_handler = registry.create_handler_runtime_tick()

    # Or get handler mapping for orchestrator routing
    handler_map = registry.get_handler_map()
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
