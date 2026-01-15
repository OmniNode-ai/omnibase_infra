# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Handler Routing Loader for Contract-Driven Orchestrators.

This module provides utilities for loading handler routing configuration
from contract.yaml files. It supports the ONEX declarative pattern where
orchestrators define handler routing in YAML rather than Python code.

Part of OMN-1316: Extract handler routing loader to shared utility.

The loader converts contract.yaml handler_routing sections into
ModelRoutingSubcontract instances that can be used by orchestrators
and the runtime for declarative handler dispatch.

Contract Structure:
    The contract.yaml uses a nested structure for handler routing::

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

Usage:
    ```python
    from pathlib import Path
    from omnibase_infra.runtime.contract_loaders import (
        load_handler_routing_subcontract,
        convert_class_to_handler_key,
    )

    # Load routing from contract.yaml
    contract_path = Path("nodes/my_orchestrator/contract.yaml")
    routing = load_handler_routing_subcontract(contract_path)

    # Access routing entries
    for entry in routing.handlers:
        print(f"{entry.routing_key} -> {entry.handler_key}")
    ```

See Also:
    - ModelRoutingSubcontract: Model for routing configuration
    - ModelRoutingEntry: Model for individual routing mappings
    - CLAUDE.md: Handler Plugin Loader patterns
"""

from __future__ import annotations

import logging
import re
from pathlib import Path

import yaml
from omnibase_core.models.primitives.model_semver import ModelSemVer

from omnibase_infra.enums import EnumInfraTransportType
from omnibase_infra.errors import ModelInfraErrorContext, ProtocolConfigurationError
from omnibase_infra.models.routing import (
    ModelRoutingEntry,
    ModelRoutingSubcontract,
)

logger = logging.getLogger(__name__)


def convert_class_to_handler_key(class_name: str) -> str:
    """Convert handler class name to handler_key format (kebab-case).

    Converts CamelCase handler class names to kebab-case handler keys
    as used in ServiceHandlerRegistry.

    Args:
        class_name: Handler class name in CamelCase (e.g., "HandlerNodeIntrospected").

    Returns:
        Handler key in kebab-case (e.g., "handler-node-introspected").

    Example:
        >>> convert_class_to_handler_key("HandlerNodeIntrospected")
        'handler-node-introspected'
        >>> convert_class_to_handler_key("HandlerRuntimeTick")
        'handler-runtime-tick'
        >>> convert_class_to_handler_key("MyHTTPHandler")
        'my-http-handler'
    """
    # Insert hyphen before uppercase letters that follow lowercase letters
    s1 = re.sub("(.)([A-Z][a-z]+)", r"\1-\2", class_name)
    # Insert hyphen before uppercase letters that follow other uppercase+lowercase sequences
    return re.sub("([a-z0-9])([A-Z])", r"\1-\2", s1).lower()


def load_handler_routing_subcontract(contract_path: Path) -> ModelRoutingSubcontract:
    """Load handler routing configuration from contract.yaml.

    Loads the handler_routing section from a contract.yaml file
    and converts it to ModelRoutingSubcontract format. This follows
    the Handler Plugin Loader pattern (see CLAUDE.md) where routing is
    defined declaratively in contract.yaml, not hardcoded in Python.

    Args:
        contract_path: Path to the contract.yaml file to load.

    Returns:
        ModelRoutingSubcontract with entries mapping event models to handlers.
        The version defaults to 1.0.0 if not specified in the contract.
        The routing_strategy defaults to "payload_type_match" if not specified.

    Raises:
        ProtocolConfigurationError: If contract.yaml does not exist, contains invalid
            YAML syntax, is empty, or handler_routing section is missing. Error context
            includes operation and target_name for debugging.

    Example:
        ```python
        from pathlib import Path
        from omnibase_infra.runtime.contract_loaders import (
            load_handler_routing_subcontract,
        )

        contract_path = Path(__file__).parent / "contract.yaml"
        routing = load_handler_routing_subcontract(contract_path)

        # Use routing entries
        for entry in routing.handlers:
            print(f"Route {entry.routing_key} to {entry.handler_key}")
        ```
    """
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
                "Skipping handler entry with missing event_model.name in contract.yaml at %s",
                contract_path,
            )
            continue

        if not handler_class_name:
            logger.warning(
                "Skipping handler entry for %s with missing handler.name in contract.yaml at %s",
                event_model_name,
                contract_path,
            )
            continue

        entries.append(
            ModelRoutingEntry(
                routing_key=event_model_name,
                handler_key=convert_class_to_handler_key(handler_class_name),
            )
        )

    logger.debug(
        "Loaded %d handler routing entries from contract.yaml at %s",
        len(entries),
        contract_path,
    )

    return ModelRoutingSubcontract(
        version=ModelSemVer(major=1, minor=0, patch=0),
        routing_strategy=handler_routing.get("routing_strategy", "payload_type_match"),
        handlers=entries,
        default_handler=None,
    )


__all__ = [
    "convert_class_to_handler_key",
    "load_handler_routing_subcontract",
]
