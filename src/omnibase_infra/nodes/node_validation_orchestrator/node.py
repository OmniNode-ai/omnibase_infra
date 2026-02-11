# SPDX-License-Identifier: MIT
# Copyright (c) 2026 OmniNode Team
"""NodeValidationOrchestrator - Declarative ORCHESTRATOR for validation pipeline.

Coordinates the validation pipeline: receives pattern candidates, builds
validation plans, dispatches to executor and adjudicator, publishes results.

All behavior is defined in contract.yaml -- no custom logic here.

This orchestrator follows the ONEX declarative pattern:
    - DECLARATIVE orchestrator driven by contract.yaml
    - Zero custom routing logic - all behavior from workflow_definition
    - Used for ONEX-compliant runtime execution via RuntimeHostProcess
    - Pattern: "Contract-driven, handlers wired by registry"

Workflow Pattern:
    1. Receive pattern candidate (consumed_events in contract)
    2. Build validation plan via HandlerBuildPlan (handler_routing)
    3. Execute checks via executor (workflow_definition.execution_graph)
    4. Adjudicate results (workflow_definition.execution_graph)
    5. Update lifecycle and publish results (published_events in contract)

Workflow: build_plan -> execute_checks -> adjudicate -> update_lifecycle

Handler Routing:
    Handler routing is defined declaratively in contract.yaml under
    handler_routing section. The orchestrator does NOT contain custom
    dispatch logic - the base class routes events based on:
    - routing_strategy: "payload_type_match"
    - handlers: mapping of event_model to handler_class

Design Decisions:
    - 100% Contract-Driven: All workflow logic in YAML, not Python
    - Zero Custom Methods: Base class handles everything
    - Declarative Execution: Workflow steps defined in execution_graph
    - Retry at Base Class: NodeOrchestrator owns retry policy
    - Contract-Driven Wiring: Handlers wired via handler_routing in contract.yaml

Coroutine Safety:
    This orchestrator is NOT coroutine-safe. Each instance should handle one
    workflow at a time. For concurrent workflows, create multiple instances.

Related Modules:
    - contract.yaml: Workflow definition, execution graph, and handler routing
    - handlers/: Handler implementations (HandlerBuildPlan)
    - registry/: RegistryInfraValidationOrchestrator for handler wiring

Ticket: OMN-2147
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

    Returns:
        ModelRoutingSubcontract with entries mapping event models to handlers.

    Raises:
        ProtocolConfigurationError: If contract.yaml does not exist, contains invalid
            YAML syntax, is empty, or handler_routing section is missing.
    """
    contract_path = Path(__file__).parent / "contract.yaml"
    return load_handler_routing_subcontract(contract_path)


class NodeValidationOrchestrator(NodeOrchestrator):
    """Declarative orchestrator for the validation pipeline.

    All behavior is defined in contract.yaml and delegated to handlers.
    This node contains no custom logic.

    Coordinates the validation pipeline: receives pattern candidates, builds
    validation plans, dispatches to executor and adjudicator, publishes results.

    Handler Routing:
        Handler routing is initialized by the runtime, not by this class.
        The runtime uses RegistryInfraValidationOrchestrator.create_orchestrator()
        to create instances with properly configured dependencies.

    Usage:
        >>> from omnibase_core.models.container import ModelONEXContainer
        >>> container = ModelONEXContainer()
        >>> orchestrator = NodeValidationOrchestrator(container)
    """

    def __init__(self, container: ModelONEXContainer) -> None:
        """Initialize with container dependency injection.

        Args:
            container: ONEX dependency injection container.
        """
        super().__init__(container)


__all__: list[str] = ["NodeValidationOrchestrator"]
