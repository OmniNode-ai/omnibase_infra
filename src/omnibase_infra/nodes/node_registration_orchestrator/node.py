# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Node Registration Orchestrator - Declarative workflow coordinator.

This orchestrator follows the ONEX declarative pattern:
    - DECLARATIVE orchestrator driven by contract.yaml
    - Zero custom routing logic - all behavior from workflow_definition
    - Used for ONEX-compliant runtime execution via RuntimeHostProcess
    - Pattern: "Contract-driven, handlers wired by registry"

Extends NodeOrchestrator from omnibase_core for workflow-driven coordination.
All workflow logic is 100% driven by contract.yaml, not Python code.

Workflow Pattern:
    1. Receive introspection event (consumed_events in contract)
    2. Call reducer to compute intents (workflow_definition.execution_graph)
    3. Execute intents via effect (workflow_definition.execution_graph)
    4. Publish result events (published_events in contract)

All workflow logic, retry policies, and result aggregation are handled
by the NodeOrchestrator base class using contract.yaml configuration.

Handler Routing:
    Handler routing is defined declaratively in contract.yaml under
    handler_routing section. The orchestrator does NOT contain custom
    dispatch logic - the base class routes events based on:
    - routing_strategy: "payload_type_match"
    - handlers: mapping of event_model to handler_class

    Handler routing is initialized via MixinHandlerRouting._init_handler_routing()
    using the registry created by RegistryInfraNodeRegistrationOrchestrator.

Design Decisions:
    - 100% Contract-Driven: All workflow logic in YAML, not Python
    - Zero Custom Methods: Base class handles everything
    - Declarative Execution: Workflow steps defined in execution_graph
    - Retry at Base Class: NodeOrchestrator owns retry policy
    - Contract-Driven Wiring: Handlers wired via handler_routing in contract.yaml
    - Mixin-Based Routing: MixinHandlerRouting provides route_to_handlers()

Coroutine Safety:
    This orchestrator is NOT coroutine-safe. Each instance should handle one
    workflow at a time. For concurrent workflows, create multiple instances.

Related Modules:
    - contract.yaml: Workflow definition, execution graph, and handler routing
    - handlers/: Handler implementations (HandlerNodeIntrospected, etc.)
    - registry/: RegistryInfraNodeRegistrationOrchestrator for handler wiring
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from omnibase_core.models.contracts.subcontracts.model_handler_routing_entry import (
    ModelHandlerRoutingEntry,
)
from omnibase_core.models.contracts.subcontracts.model_handler_routing_subcontract import (
    ModelHandlerRoutingSubcontract,
)
from omnibase_core.models.primitives.model_semver import ModelSemVer
from omnibase_core.nodes.node_orchestrator import NodeOrchestrator

from omnibase_infra.nodes.node_registration_orchestrator.registry import (
    RegistryInfraNodeRegistrationOrchestrator,
)

if TYPE_CHECKING:
    from omnibase_core.models.container import ModelONEXContainer
    from omnibase_core.services.service_handler_registry import ServiceHandlerRegistry

    from omnibase_infra.projectors import ProjectionReaderRegistration


def _create_handler_routing_subcontract() -> ModelHandlerRoutingSubcontract:
    """Create handler routing subcontract from contract.yaml configuration.

    This function creates the handler routing configuration that matches
    the contract.yaml handler_routing section. Eventually this should be
    loaded directly from the YAML contract.

    Returns:
        ModelHandlerRoutingSubcontract with routing entries for:
        - ModelNodeIntrospectionEvent -> handler-node-introspected
        - ModelRuntimeTick -> handler-runtime-tick
        - ModelNodeRegistrationAcked -> handler-node-registration-acked
        - ModelNodeHeartbeatEvent -> handler-node-heartbeat
    """
    return ModelHandlerRoutingSubcontract(
        version=ModelSemVer(major=1, minor=0, patch=0),
        routing_strategy="payload_type_match",
        handlers=[
            ModelHandlerRoutingEntry(
                routing_key="ModelNodeIntrospectionEvent",
                handler_key="handler-node-introspected",
            ),
            ModelHandlerRoutingEntry(
                routing_key="ModelRuntimeTick",
                handler_key="handler-runtime-tick",
            ),
            ModelHandlerRoutingEntry(
                routing_key="ModelNodeRegistrationAcked",
                handler_key="handler-node-registration-acked",
            ),
            ModelHandlerRoutingEntry(
                routing_key="ModelNodeHeartbeatEvent",
                handler_key="handler-node-heartbeat",
            ),
        ],
        default_handler=None,
    )


class NodeRegistrationOrchestrator(NodeOrchestrator):
    """Declarative orchestrator for node registration workflow.

    All behavior is defined in contract.yaml - no custom logic here.
    Handler routing is driven entirely by the contract and initialized
    via MixinHandlerRouting from the base class.

    Example YAML Contract:
        ```yaml
        handler_routing:
          routing_strategy: "payload_type_match"
          handlers:
            - event_model: "ModelNodeIntrospectionEvent"
              handler_class: "HandlerNodeIntrospected"
            - event_model: "ModelRuntimeTick"
              handler_class: "HandlerRuntimeTick"
            - event_model: "ModelNodeHeartbeatEvent"
              handler_class: "HandlerNodeHeartbeat"

        workflow_coordination:
          workflow_definition:
            workflow_metadata:
              workflow_name: node_registration
              workflow_version: {major: 1, minor: 0, patch: 0}
              execution_mode: sequential
              description: "Node registration workflow"

            execution_graph:
              nodes:
                - node_id: "compute_intents"
                  node_type: reducer
                  description: "Compute registration intents"
                - node_id: "execute_consul"
                  node_type: effect
                  description: "Register with Consul"
                - node_id: "execute_postgres"
                  node_type: effect
                  description: "Register in PostgreSQL"

            coordination_rules:
              parallel_execution_allowed: false
              failure_recovery_strategy: retry
              max_retries: 3
              timeout_ms: 30000
        ```

    Usage:
        ```python
        from omnibase_core.models.container import ModelONEXContainer

        # Create and initialize
        container = ModelONEXContainer()
        orchestrator = NodeRegistrationOrchestrator(container)

        # Workflow definition loaded from contract.yaml by runtime
        # Process input
        result = await orchestrator.process(input_data)
        ```

    Handler Routing Initialization:
        The orchestrator initializes handler routing via MixinHandlerRouting:
        1. Creates ModelHandlerRoutingSubcontract from contract config
        2. Creates ServiceHandlerRegistry via RegistryInfraNodeRegistrationOrchestrator
        3. Calls _init_handler_routing() to wire routing table
    """

    def __init__(
        self,
        container: ModelONEXContainer,
        projection_reader: ProjectionReaderRegistration | None = None,
    ) -> None:
        """Initialize with container dependency injection and handler routing.

        Args:
            container: ONEX dependency injection container.
            projection_reader: Optional projection reader for handler dependencies.
                If None, handler routing initialization is deferred until
                projection_reader is available via initialize_handler_routing().
        """
        super().__init__(container)

        # Store projection reader for deferred initialization
        self._projection_reader = projection_reader

        # Initialize handler routing if projection_reader is available
        if projection_reader is not None:
            self._initialize_handler_routing(projection_reader)

    def _initialize_handler_routing(
        self, projection_reader: ProjectionReaderRegistration
    ) -> None:
        """Initialize handler routing with the given projection reader.

        Creates the handler registry and initializes the routing table
        from the contract configuration.

        Args:
            projection_reader: Projection reader for handler dependencies.
        """
        # Create handler routing subcontract from contract config
        handler_routing = _create_handler_routing_subcontract()

        # Create registry with handlers via static factory
        # Note: projector and consul_handler are None - heartbeat handler won't be registered
        registry: ServiceHandlerRegistry = (
            RegistryInfraNodeRegistrationOrchestrator.create_registry(
                projection_reader=projection_reader,
                projector=None,
                consul_handler=None,
            )
        )

        # Initialize routing via MixinHandlerRouting from base class
        # The registry is already frozen by create_registry()
        self._init_handler_routing(handler_routing, registry)

    def initialize_handler_routing(
        self, projection_reader: ProjectionReaderRegistration
    ) -> None:
        """Initialize handler routing with deferred projection reader.

        Call this method after construction if projection_reader was not
        available at construction time.

        Args:
            projection_reader: Projection reader for handler dependencies.

        Raises:
            RuntimeError: If handler routing is already initialized.
        """
        if self.is_routing_initialized:
            raise RuntimeError("Handler routing is already initialized")
        self._projection_reader = projection_reader
        self._initialize_handler_routing(projection_reader)


__all__ = ["NodeRegistrationOrchestrator"]
