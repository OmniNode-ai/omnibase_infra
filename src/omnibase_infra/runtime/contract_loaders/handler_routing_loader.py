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

# Maximum allowed file size for contract.yaml files (10MB)
# Security control to prevent memory exhaustion via large YAML files
# Error code: FILE_SIZE_EXCEEDED (HANDLER_LOADER_050)
MAX_CONTRACT_FILE_SIZE_BYTES: int = 10 * 1024 * 1024  # 10MB


def _check_file_size(contract_path: Path, operation: str) -> None:
    """Check that contract file does not exceed maximum allowed size.

    This is a security control to prevent memory exhaustion attacks via
    oversized YAML files. Per CLAUDE.md Handler Plugin Loader security patterns,
    a 10MB file size limit is enforced.

    Args:
        contract_path: Path to the contract.yaml file.
        operation: Name of the operation for error context.

    Raises:
        ProtocolConfigurationError: If file exceeds MAX_CONTRACT_FILE_SIZE_BYTES.
            Error code: FILE_SIZE_EXCEEDED (HANDLER_LOADER_050).
    """
    try:
        file_size = contract_path.stat().st_size
    except FileNotFoundError:
        # Let the caller handle FileNotFoundError with its own error message
        return

    if file_size > MAX_CONTRACT_FILE_SIZE_BYTES:
        ctx = ModelInfraErrorContext.with_correlation(
            transport_type=EnumInfraTransportType.FILESYSTEM,
            operation=operation,
            target_name=str(contract_path),
        )
        logger.error(
            "Contract file exceeds maximum size: %d bytes > %d bytes at %s",
            file_size,
            MAX_CONTRACT_FILE_SIZE_BYTES,
            contract_path,
        )
        raise ProtocolConfigurationError(
            f"Contract file exceeds maximum size: {file_size} bytes > "
            f"{MAX_CONTRACT_FILE_SIZE_BYTES} bytes. "
            f"Reduce the contract.yaml file size or split into multiple contracts. "
            f"Error code: FILE_SIZE_EXCEEDED (HANDLER_LOADER_050)",
            context=ctx,
        )


def _load_and_validate_contract_yaml(
    contract_path: Path,
    operation: str,
) -> tuple[dict, dict]:
    """Load and validate contract.yaml with handler_routing section.

    Private helper that consolidates common YAML loading and validation logic
    used by multiple loader functions. This reduces code duplication for:
    - File existence checking with proper error context
    - YAML parsing with error handling
    - Empty contract validation
    - handler_routing section validation

    Note on direct file operations:
        This function uses direct file I/O rather than a FileRegistry abstraction.
        See handler_contract_source.py for the same pattern and rationale:
        - RegistryFileBased (FileRegistry) does not yet exist in omnibase_core
        - Once available, this should be refactored for consistency
        - See: docs/architecture/RUNTIME_HOST_IMPLEMENTATION_PLAN.md

        TODO [OMN-1352]: Refactor to use RegistryFileBased once available in omnibase_core.
        This will provide consistent file access patterns across all contract loaders.

    Security Note:
        File size is checked BEFORE YAML parsing (line 164) to prevent memory
        exhaustion attacks via oversized files. This ordering is critical -
        _check_file_size() MUST be called before yaml.safe_load().

    Args:
        contract_path: Path to the contract.yaml file to load.
        operation: Name of the calling operation for error context
            (e.g., "load_handler_routing_contract", "load_handler_class_info").

    Returns:
        Tuple of (contract_dict, handler_routing_dict) where:
        - contract_dict: The full parsed contract
        - handler_routing_dict: The handler_routing section

    Raises:
        ProtocolConfigurationError: If contract.yaml does not exist, contains
            invalid YAML syntax, is empty, exceeds maximum file size,
            or handler_routing section is missing.
    """
    # Check file size before loading (security control)
    _check_file_size(contract_path, operation)

    # Load YAML file
    try:
        with contract_path.open("r", encoding="utf-8") as f:
            contract = yaml.safe_load(f)
    except FileNotFoundError as e:
        ctx = ModelInfraErrorContext.with_correlation(
            transport_type=EnumInfraTransportType.FILESYSTEM,
            operation=operation,
            target_name=str(contract_path),
        )
        logger.exception(
            "contract.yaml not found at %s - cannot load contract",
            contract_path,
        )
        raise ProtocolConfigurationError(
            f"Contract file not found: {contract_path}. "
            f"Ensure the contract.yaml exists in the orchestrator directory. "
            f"Error code: CONTRACT_NOT_FOUND (HANDLER_LOADER_020)",
            context=ctx,
        ) from e
    except yaml.YAMLError as e:
        ctx = ModelInfraErrorContext.with_correlation(
            transport_type=EnumInfraTransportType.FILESYSTEM,
            operation=operation,
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
            f"Invalid YAML syntax in contract.yaml at {contract_path}: {error_type}. "
            f"Verify the YAML syntax is correct. "
            f"Error code: YAML_PARSE_ERROR (HANDLER_LOADER_021)",
            context=ctx,
        ) from e

    # Validate contract is not empty
    if contract is None:
        ctx = ModelInfraErrorContext.with_correlation(
            transport_type=EnumInfraTransportType.FILESYSTEM,
            operation=operation,
            target_name=str(contract_path),
        )
        msg = (
            f"Contract file is empty: {contract_path}. "
            f"The contract.yaml must contain valid YAML with a 'handler_routing' section. "
            f"Error code: CONTRACT_EMPTY (HANDLER_LOADER_022)"
        )
        logger.error(msg)
        raise ProtocolConfigurationError(msg, context=ctx)

    # Validate handler_routing section exists
    handler_routing = contract.get("handler_routing")
    if handler_routing is None:
        ctx = ModelInfraErrorContext.with_correlation(
            transport_type=EnumInfraTransportType.FILESYSTEM,
            operation=operation,
            target_name=str(contract_path),
        )
        msg = (
            f"Missing 'handler_routing' section in contract: {contract_path}. "
            f"Add a handler_routing section with routing_strategy and handlers. "
            f"Error code: MISSING_HANDLER_ROUTING (HANDLER_LOADER_023)"
        )
        logger.error(msg)
        raise ProtocolConfigurationError(msg, context=ctx)

    return contract, handler_routing


# Valid routing strategies for handler routing contracts.
# Currently only "payload_type_match" is implemented. Additional strategies
# such as "first_match" or "all_match" may be added in future versions.
# Unknown strategies will trigger a warning and fall back to "payload_type_match".
VALID_ROUTING_STRATEGIES: frozenset[str] = frozenset(
    {
        "payload_type_match",  # Match by payload type (default and only implemented)
    }
)


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
    # Use shared helper for YAML loading and validation
    _contract, handler_routing = _load_and_validate_contract_yaml(
        contract_path, "load_handler_routing_contract"
    )

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

    # IMPORTANT: Validate routing_strategy BEFORE constructing ModelRoutingSubcontract.
    # This ensures invalid strategy values are caught and corrected (with warning)
    # before the model is instantiated, preventing invalid state in the model.
    routing_strategy = handler_routing.get("routing_strategy", "payload_type_match")
    if routing_strategy not in VALID_ROUTING_STRATEGIES:
        logger.warning(
            "Unknown routing_strategy '%s' in contract at %s. "
            "Valid values: %s. Using 'payload_type_match' as default.",
            routing_strategy,
            contract_path,
            ", ".join(sorted(VALID_ROUTING_STRATEGIES)),
        )
        routing_strategy = "payload_type_match"

    # Now construct the model with validated routing_strategy
    return ModelRoutingSubcontract(
        version=ModelSemVer(major=1, minor=0, patch=0),
        routing_strategy=routing_strategy,
        handlers=entries,
        default_handler=None,
    )


def load_handler_class_info_from_contract(
    contract_path: Path,
) -> list[dict[str, str]]:
    """Load handler class and module information from contract.yaml.

    Extracts handler class and module information from the handler_routing
    section of contract.yaml. This is used by registries that need to
    dynamically load and instantiate handler classes.

    This function consolidates YAML loading logic that would otherwise be
    duplicated in registry modules (see OMN-1316).

    Args:
        contract_path: Path to the contract.yaml file.

    Returns:
        List of handler configurations, each containing:
        - handler_class: The handler class name (e.g., "HandlerNodeIntrospected")
        - handler_module: The fully qualified module path

    Raises:
        ProtocolConfigurationError: If the contract file cannot be loaded,
            contains invalid YAML, is empty, or handler_routing section is missing.

    Example:
        ```python
        from pathlib import Path
        from omnibase_infra.runtime.contract_loaders import (
            load_handler_class_info_from_contract,
        )

        contract_path = Path(__file__).parent / "contract.yaml"
        handler_infos = load_handler_class_info_from_contract(contract_path)

        for info in handler_infos:
            handler_cls = importlib.import_module(info["handler_module"])
            handler = getattr(handler_cls, info["handler_class"])
        ```
    """
    # Use shared helper for YAML loading and validation
    _contract, handler_routing = _load_and_validate_contract_yaml(
        contract_path, "load_handler_class_info_from_contract"
    )

    handlers_config = handler_routing.get("handlers", [])
    result: list[dict[str, str]] = []

    for handler_entry in handlers_config:
        handler_info = handler_entry.get("handler", {})
        handler_class = handler_info.get("name")
        handler_module = handler_info.get("module")

        if handler_class and handler_module:
            result.append(
                {
                    "handler_class": handler_class,
                    "handler_module": handler_module,
                }
            )
        else:
            logger.warning(
                "Skipping handler entry with missing name or module in contract.yaml at %s",
                contract_path,
            )

    logger.debug(
        "Loaded %d handler class info entries from contract.yaml at %s",
        len(result),
        contract_path,
    )

    return result


__all__ = [
    "MAX_CONTRACT_FILE_SIZE_BYTES",
    "VALID_ROUTING_STRATEGIES",
    "convert_class_to_handler_key",
    "load_handler_class_info_from_contract",
    "load_handler_routing_subcontract",
]
