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

    Handler routing is initialized by the RUNTIME (not this module) via
    MixinHandlerRouting._init_handler_routing(), using the registry created
    by RegistryInfraNodeRegistrationOrchestrator. This module only provides
    the helper function _create_handler_routing_subcontract() for the runtime.

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

if TYPE_CHECKING:
    from omnibase_core.models.container import ModelONEXContainer


def _create_handler_routing_subcontract() -> ModelHandlerRoutingSubcontract:
    """Create handler routing subcontract from contract.yaml configuration.

    TODO(OMN-1102): Replace hardcoded routing with contract.yaml loading.
        This function hardcodes the handler routing configuration which duplicates
        what is already declared in contract.yaml. Per the Handler Plugin Loader
        pattern (see CLAUDE.md), routing should be loaded directly from contract.yaml
        at runtime using the plugin-based handler loading system.

        Desired approach:
            1. Load handler_routing section from contract.yaml
            2. Use HandlerPluginLoader to resolve handler classes
            3. Build ModelHandlerRoutingSubcontract from loaded config

        This function may also need relocation to a runtime/loader module rather
        than living in the node module itself (see PR #141 nitpick).

        References:
            - PR #141 review comments
            - CLAUDE.md "Handler Plugin Loader Patterns" section
            - docs/patterns/handler_plugin_loader.md

    This function creates the handler routing configuration that matches
    the contract.yaml handler_routing section.

    Note:
        The contract.yaml uses a nested structure::

            handlers:
              - event_model:
                  name: "ModelNodeIntrospectionEvent"
                  module: "omnibase_infra.models..."
                handler:
                  name: "HandlerNodeIntrospected"
                  module: "omnibase_infra.nodes..."

        While ModelHandlerRoutingEntry uses flat fields::

            ModelHandlerRoutingEntry(
                routing_key="ModelNodeIntrospectionEvent",  # from event_model.name
                handler_key="handler-node-introspected",    # adapter ID in registry
            )

        The ``routing_key`` maps to ``event_model.name`` from contract.yaml.
        The ``handler_key`` is the handler's adapter ID in ServiceHandlerRegistry,
        NOT the class name from ``handler.name``.

    Returns:
        ModelHandlerRoutingSubcontract with entries mapping event models to handlers:

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

    Example YAML Contract (contract.yaml format):
        ```yaml
        handler_routing:
          routing_strategy: "payload_type_match"
          handlers:
            - event_model:
                name: "ModelNodeIntrospectionEvent"
                module: "omnibase_infra.models.registration..."
              handler:
                name: "HandlerNodeIntrospected"
                module: "omnibase_infra.nodes...handlers..."
            - event_model:
                name: "ModelRuntimeTick"
                module: "omnibase_infra.runtime.models..."
              handler:
                name: "HandlerRuntimeTick"
                module: "omnibase_infra.nodes...handlers..."

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

    Note on Handler Routing Field Names:
        The contract.yaml uses a nested structure with ``event_model.name`` and
        ``handler.name``, but ModelHandlerRoutingEntry uses flat fields:

        - ``routing_key``: Corresponds to ``event_model.name``
        - ``handler_key``: The handler's adapter ID in ServiceHandlerRegistry
          (e.g., "handler-node-introspected"), NOT the class name

        See ``_create_handler_routing_subcontract()`` for the translation.

    Usage:
        ```python
        from omnibase_core.models.container import ModelONEXContainer

        # Create and initialize
        container = ModelONEXContainer()
        orchestrator = NodeRegistrationOrchestrator(container)

        # Workflow definition loaded from contract.yaml by runtime
        # Handler routing initialized via runtime using registry factory
        # Process input
        result = await orchestrator.process(input_data)
        ```

    Handler Routing:
        Handler routing is initialized by the runtime, not by this class.
        The runtime uses RegistryInfraNodeRegistrationOrchestrator.create_registry()
        to create the handler registry and calls _init_handler_routing() on
        the orchestrator instance.

    Runtime Initialization:
        Handler routing is initialized by RuntimeHostProcess, not by this class.
        The runtime performs the following sequence:

        1. Creates handler registry via
           RegistryInfraNodeRegistrationOrchestrator.create_registry()
        2. Creates handler routing subcontract via
           _create_handler_routing_subcontract()
        3. Calls orchestrator._init_handler_routing(subcontract, registry)

        This separation ensures the orchestrator remains purely declarative
        with no custom initialization logic.
    """

    def __init__(self, container: ModelONEXContainer) -> None:
        """Initialize with container dependency injection.

        Args:
            container: ONEX dependency injection container.
        """
        super().__init__(container)


__all__ = ["NodeRegistrationOrchestrator"]
