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
import re
from pathlib import Path
from typing import TYPE_CHECKING

import yaml
from omnibase_core.models.primitives.model_semver import ModelSemVer
from omnibase_core.nodes.node_orchestrator import NodeOrchestrator

from omnibase_infra.enums import EnumInfraTransportType
from omnibase_infra.errors import ModelInfraErrorContext, ProtocolConfigurationError
from omnibase_infra.models.routing import (
    ModelRoutingEntry,
    ModelRoutingSubcontract,
)

if TYPE_CHECKING:
    from omnibase_core.models.container import ModelONEXContainer

logger = logging.getLogger(__name__)


# TODO(OMN-1315): Consider relocating module-level helper functions to a dedicated
# utility module (e.g., omnibase_infra.utils.handler_utils) if they become shared
# across multiple orchestrators. Currently kept here for locality with the
# _create_handler_routing_subcontract function that uses it.


def _convert_class_to_handler_key(class_name: str) -> str:
    """Convert handler class name to handler_key format (kebab-case).

    Converts CamelCase handler class names to kebab-case handler keys
    as used in ServiceHandlerRegistry.

    Args:
        class_name: Handler class name in CamelCase (e.g., "HandlerNodeIntrospected").

    Returns:
        Handler key in kebab-case (e.g., "handler-node-introspected").

    Example:
        >>> _convert_class_to_handler_key("HandlerNodeIntrospected")
        'handler-node-introspected'
        >>> _convert_class_to_handler_key("HandlerRuntimeTick")
        'handler-runtime-tick'
    """
    # Insert hyphen before uppercase letters that follow lowercase letters
    s1 = re.sub("(.)([A-Z][a-z]+)", r"\1-\2", class_name)
    # Insert hyphen before uppercase letters that follow other uppercase+lowercase sequences
    return re.sub("([a-z0-9])([A-Z])", r"\1-\2", s1).lower()


def _create_handler_routing_subcontract() -> ModelRoutingSubcontract:
    """Load handler routing configuration from contract.yaml.

    Loads the handler_routing section from this node's contract.yaml
    and converts it to ModelRoutingSubcontract format. This follows
    the Handler Plugin Loader pattern (see CLAUDE.md) where routing is
    defined declaratively in contract.yaml, not hardcoded in Python.

    Contract Structure:
        The contract.yaml uses a nested structure::

            handler_routing:
              routing_strategy: "payload_type_match"
              handlers:
                - event_model:
                    name: "ModelNodeIntrospectionEvent"
                    module: "omnibase_infra.models..."
                  handler:
                    name: "HandlerNodeIntrospected"
                    module: "omnibase_infra.nodes..."

        This is converted to ModelRoutingEntry with flat fields::

            ModelRoutingEntry(
                routing_key="ModelNodeIntrospectionEvent",  # from event_model.name
                handler_key="handler-node-introspected",    # kebab-case of handler.name
            )

        The ``routing_key`` maps to ``event_model.name`` from contract.yaml.
        The ``handler_key`` is derived by converting ``handler.name`` to kebab-case,
        matching the handler's adapter ID in ServiceHandlerRegistry.

    Returns:
        ModelRoutingSubcontract with entries mapping event models to handlers.

    Raises:
        ProtocolConfigurationError: If contract.yaml does not exist, contains invalid
            YAML syntax, is empty, or handler_routing section is missing. Error context
            includes operation and target_name for debugging.
    """
    # Load contract.yaml from same directory as this module
    contract_path = Path(__file__).parent / "contract.yaml"

    try:
        with contract_path.open("r", encoding="utf-8") as f:
            contract = yaml.safe_load(f)
    except FileNotFoundError as e:
        ctx = ModelInfraErrorContext(
            transport_type=EnumInfraTransportType.DATABASE,
            operation="load_handler_routing_contract",
            target_name=str(contract_path),
        )
        logger.exception(
            "contract.yaml not found at %s - handler routing cannot be loaded",
            contract_path,
        )
        raise ProtocolConfigurationError(
            f"contract.yaml not found at {contract_path} - handler routing cannot be loaded",
            context=ctx,
        ) from e
    except yaml.YAMLError as e:
        ctx = ModelInfraErrorContext(
            transport_type=EnumInfraTransportType.DATABASE,
            operation="parse_handler_routing_contract",
            target_name=str(contract_path),
        )
        # Sanitize error message - don't include raw YAML error which may contain file contents
        error_type = type(e).__name__
        logger.exception(
            "Invalid YAML syntax in contract.yaml at %s: %s",
            contract_path,
            error_type,
        )
        raise ProtocolConfigurationError(
            f"Invalid YAML syntax in contract.yaml at {contract_path}: {error_type}",
            context=ctx,
        ) from e

    if contract is None:
        ctx = ModelInfraErrorContext(
            transport_type=EnumInfraTransportType.DATABASE,
            operation="validate_handler_routing_contract",
            target_name=str(contract_path),
        )
        msg = f"contract.yaml at {contract_path} is empty"
        logger.error(msg)
        raise ProtocolConfigurationError(msg, context=ctx)

    handler_routing = contract.get("handler_routing")
    if handler_routing is None:
        ctx = ModelInfraErrorContext(
            transport_type=EnumInfraTransportType.DATABASE,
            operation="validate_handler_routing_contract",
            target_name=str(contract_path),
        )
        msg = f"handler_routing section not found in contract.yaml at {contract_path}"
        logger.error(msg)
        raise ProtocolConfigurationError(msg, context=ctx)

    # Build routing entries from contract
    entries: list[ModelRoutingEntry] = []
    handlers_config = handler_routing.get("handlers", [])

    for handler_config in handlers_config:
        event_model = handler_config.get("event_model", {})
        handler = handler_config.get("handler", {})

        event_model_name = event_model.get("name")
        handler_class_name = handler.get("name")

        if not event_model_name:
            logger.warning(
                "Skipping handler entry with missing event_model.name in contract.yaml"
            )
            continue

        if not handler_class_name:
            logger.warning(
                "Skipping handler entry for %s with missing handler.name in contract.yaml",
                event_model_name,
            )
            continue

        entries.append(
            ModelRoutingEntry(
                routing_key=event_model_name,
                handler_key=_convert_class_to_handler_key(handler_class_name),
            )
        )

    logger.debug(
        "Loaded %d handler routing entries from contract.yaml",
        len(entries),
    )

    return ModelRoutingSubcontract(
        version=ModelSemVer(major=1, minor=0, patch=0),
        routing_strategy=handler_routing.get("routing_strategy", "payload_type_match"),
        handlers=entries,
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
