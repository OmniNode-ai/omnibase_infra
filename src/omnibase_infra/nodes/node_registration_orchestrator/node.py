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

import logging
from pathlib import Path
from typing import TYPE_CHECKING

from omnibase_core.nodes.node_orchestrator import NodeOrchestrator
from omnibase_infra.models.routing import ModelRoutingSubcontract
from omnibase_infra.runtime.contract_loaders import load_handler_routing_subcontract

if TYPE_CHECKING:
    from omnibase_core.models.container import ModelONEXContainer

logger = logging.getLogger(__name__)


def _create_handler_routing_subcontract() -> ModelRoutingSubcontract:
    """Load handler routing configuration from this node's contract.yaml.

    Thin wrapper around the shared utility load_handler_routing_subcontract()
    that provides the contract path for this node. This maintains the existing
    function signature for internal callers while delegating to the shared
    implementation.

    Part of OMN-1316: Refactored to use shared handler routing loader utility.

    Returns:
        ModelRoutingSubcontract with entries mapping event models to handlers.

    Raises:
        ProtocolConfigurationError: If contract.yaml does not exist, contains invalid
            YAML syntax, is empty, or handler_routing section is missing.
    """
    contract_path = Path(__file__).parent / "contract.yaml"
    return load_handler_routing_subcontract(contract_path)


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
                  node_type: REDUCER_GENERIC
                  description: "Compute registration intents"
                - node_id: "execute_consul"
                  node_type: EFFECT_GENERIC
                  description: "Register with Consul"
                - node_id: "execute_postgres"
                  node_type: EFFECT_GENERIC
                  description: "Register in PostgreSQL"

            coordination_rules:
              parallel_execution_allowed: false
              failure_recovery_strategy: retry
              max_retries: 3
              timeout_ms: 30000
        ```

    Note on Handler Routing Field Names:
        The contract.yaml uses a nested structure with ``event_model.name`` and
        ``handler.name``, but ModelRoutingEntry uses flat fields:

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
