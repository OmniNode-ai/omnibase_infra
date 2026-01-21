# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Handler Contract Configuration Loader.

This module provides utilities for loading handler configuration from contract
YAML files. It supports both relative and absolute paths and extracts handler-
specific configuration for use during handler initialization.

Part of the bootstrap handler contract infrastructure.

The loader validates:
- Contract file existence
- YAML syntax validity
- Required contract structure (must be a dict)

Contract File Structure:
    Handler contracts follow this schema (see contracts/handlers/*/handler_contract.yaml):

    ```yaml
    name: handler-consul
    handler_class: omnibase_infra.handlers.handler_consul.HandlerConsul
    handler_type: effect
    tags:
      - consul
      - service-discovery
    security:
      trusted_namespace: omnibase_infra.handlers
      audit_logging: true
      requires_authentication: false  # optional
    ```

Security:
    - Uses yaml.safe_load() to prevent arbitrary code execution
    - Contract files are treated as trusted configuration (see CLAUDE.md security patterns)

See Also:
    - HandlerBootstrapSource: Uses this loader for bootstrap handler contracts
    - handler_plugin_loader.py: Related handler loading infrastructure
    - docs/patterns/handler_plugin_loader.md: Security documentation

.. versionadded:: 0.6.5
    Created for bootstrap handler contract loading.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING

import yaml

from omnibase_infra.errors import ProtocolConfigurationError
from omnibase_infra.models.errors.model_infra_error_context import (
    ModelInfraErrorContext,
)

if TYPE_CHECKING:
    from omnibase_core.types import JsonType

logger = logging.getLogger(__name__)

# Maximum contract file size (10 MB) - matches handler_plugin_loader.py
MAX_CONTRACT_SIZE_BYTES = 10 * 1024 * 1024


def load_handler_contract_config(
    contract_path: str | Path | None,
    handler_id: str,
) -> dict[str, JsonType]:
    """Load handler configuration from contract YAML file.

    Reads and parses a handler contract YAML file, returning the parsed
    dictionary for further processing. The contract path can be either
    absolute or relative (resolved against common base paths).

    Args:
        contract_path: Path to handler_contract.yaml (relative or absolute).
            If None, raises ProtocolConfigurationError.
        handler_id: Handler identifier for error messages and logging.

    Returns:
        Dict containing the parsed contract YAML content.

    Raises:
        ProtocolConfigurationError: If contract_path is None, file not found,
            file too large, YAML syntax error, or contract is not a dict.

    Example:
        >>> contract = load_handler_contract_config(
        ...     "contracts/handlers/consul/handler_contract.yaml",
        ...     "bootstrap.consul",
        ... )
        >>> contract["name"]
        'handler-consul'
    """
    if contract_path is None:
        raise ProtocolConfigurationError(
            f"Handler {handler_id} has no contract_path configured",
            context=ModelInfraErrorContext.with_correlation(
                operation="load_handler_contract",
                target_name=handler_id,
            ),
        )

    path = Path(contract_path)

    # Resolve relative paths against common base directories
    if not path.is_absolute():
        resolved_path = _resolve_contract_path(path)
        if resolved_path is None:
            raise ProtocolConfigurationError(
                f"Contract file not found: {contract_path}",
                context=ModelInfraErrorContext.with_correlation(
                    operation="load_handler_contract",
                    target_name=handler_id,
                ),
            )
        path = resolved_path

    if not path.exists():
        raise ProtocolConfigurationError(
            f"Contract file not found: {contract_path}",
            context=ModelInfraErrorContext.with_correlation(
                operation="load_handler_contract",
                target_name=handler_id,
            ),
        )

    # Check file size before reading (security: prevent memory exhaustion)
    file_size = path.stat().st_size
    if file_size > MAX_CONTRACT_SIZE_BYTES:
        raise ProtocolConfigurationError(
            f"Contract file too large: {file_size} bytes (max {MAX_CONTRACT_SIZE_BYTES})",
            context=ModelInfraErrorContext.with_correlation(
                operation="load_handler_contract",
                target_name=handler_id,
            ),
        )

    try:
        with path.open() as f:
            contract = yaml.safe_load(f)
    except yaml.YAMLError as e:
        raise ProtocolConfigurationError(
            f"Invalid YAML in contract: {e}",
            context=ModelInfraErrorContext.with_correlation(
                operation="load_handler_contract",
                target_name=handler_id,
            ),
        ) from e

    if not isinstance(contract, dict):
        raise ProtocolConfigurationError(
            f"Contract must be a dict, got {type(contract).__name__}",
            context=ModelInfraErrorContext.with_correlation(
                operation="load_handler_contract",
                target_name=handler_id,
            ),
        )

    logger.debug(
        "Loaded handler contract",
        extra={
            "handler_id": handler_id,
            "contract_path": str(path),
            "contract_name": contract.get("name"),
        },
    )

    return contract


def _resolve_contract_path(relative_path: Path) -> Path | None:
    """Resolve a relative contract path against common base directories.

    Tries multiple base directories to find the contract file:
    1. Current working directory
    2. Package root (omnibase_infra source root)
    3. Repository root (three levels up from this file)

    Args:
        relative_path: Relative path to the contract file.

    Returns:
        Resolved absolute path if found, None otherwise.
    """
    # Base directories to try
    base_paths = [
        Path.cwd(),
        # Package source root (src/omnibase_infra -> go up to repo root)
        Path(__file__).parent.parent.parent.parent,
        # Alternative: direct parent chain
        Path(__file__).parent.parent.parent.parent.parent,
    ]

    for base in base_paths:
        full_path = base / relative_path
        if full_path.exists():
            return full_path.resolve()

    return None


def extract_handler_config(
    contract: dict[str, JsonType],
    handler_type: str,
) -> dict[str, JsonType]:
    """Extract handler-specific configuration from parsed contract.

    Extracts configuration values from both basic and rich contract structures
    and flattens them into a dict suitable for handler initialization.

    Supports Two Contract Formats:

    Basic Contract Structure (contracts/handlers/*/handler_contract.yaml):
        - name: Handler name (e.g., "handler-consul")
        - handler_class: Fully qualified class path
        - handler_type: Handler kind (effect, compute, etc.)
        - tags: List of discovery tags
        - security: Security metadata dict
            - trusted_namespace: Required trusted import namespace
            - audit_logging: Whether to enable audit logging
            - requires_authentication: Whether auth is required (optional)

    Rich Contract Structure (src/omnibase_infra/contracts/handlers/*/handler_contract.yaml):
        - handler_id: Unique identifier (e.g., "effect.mcp.handler")
        - name: Handler name
        - version: Semantic version
        - descriptor: Handler descriptor with timeout, retry, circuit breaker
            - handler_kind: Handler behavioral type
            - timeout_ms: Timeout in milliseconds
            - retry_policy: Retry configuration
            - circuit_breaker: Circuit breaker configuration
        - metadata: Additional metadata
            - transport: Transport configuration
                - default_host: Default bind host
                - default_port: Default port
                - default_path: Default URL path
                - stateless: Whether handler is stateless
                - json_response: Whether to use JSON responses
            - security: Security configuration
                - tool_access: Tool access control
                    - max_tools: Maximum number of tools

    Args:
        contract: Parsed contract dict from load_handler_contract_config().
        handler_type: Handler type identifier (e.g., "consul", "db") for
            logging and context.

    Returns:
        Flat dict with extracted configuration values suitable for
        handler.initialize() or similar configuration methods:
            - name: Handler name
            - handler_class: Fully qualified class path
            - handler_kind: Handler behavioral type
            - tags: List of tags
            - trusted_namespace: Security namespace
            - audit_logging: Audit logging flag
            - requires_authentication: Auth requirement flag
            - host: Transport default host (rich contracts)
            - port: Transport default port (rich contracts)
            - path: Transport default path (rich contracts)
            - stateless: Transport stateless flag (rich contracts)
            - json_response: Transport JSON response flag (rich contracts)
            - timeout_seconds: Timeout in seconds (rich contracts)
            - max_tools: Maximum tools for MCP (rich contracts)

    Example:
        >>> # Basic contract
        >>> contract = {
        ...     "name": "handler-consul",
        ...     "handler_class": "omnibase_infra.handlers.handler_consul.HandlerConsul",
        ...     "handler_type": "effect",
        ...     "tags": ["consul", "service-discovery"],
        ...     "security": {
        ...         "trusted_namespace": "omnibase_infra.handlers",
        ...         "audit_logging": True,
        ...     },
        ... }
        >>> config = extract_handler_config(contract, "consul")
        >>> config["name"]
        'handler-consul'
        >>> config["audit_logging"]
        True

        >>> # Rich contract with transport
        >>> rich_contract = {
        ...     "name": "MCP Handler",
        ...     "descriptor": {"handler_kind": "effect", "timeout_ms": 30000},
        ...     "metadata": {
        ...         "transport": {"default_host": "0.0.0.0", "default_port": 8090},
        ...         "security": {"tool_access": {"max_tools": 100}},
        ...     },
        ... }
        >>> config = extract_handler_config(rich_contract, "mcp")
        >>> config["port"]
        8090
        >>> config["max_tools"]
        100
    """
    config: dict[str, JsonType] = {}

    # Extract top-level fields
    if "name" in contract:
        config["name"] = contract["name"]

    if "handler_class" in contract:
        config["handler_class"] = contract["handler_class"]

    # Handler kind can be in handler_type (basic) or descriptor.handler_kind (rich)
    if "handler_type" in contract:
        config["handler_kind"] = contract["handler_type"]

    if "tags" in contract:
        config["tags"] = contract["tags"]

    # Extract descriptor configuration (rich contracts)
    descriptor = contract.get("descriptor", {})
    if isinstance(descriptor, dict):
        # Handler kind from descriptor (rich contracts)
        if "handler_kind" in descriptor and "handler_kind" not in config:
            config["handler_kind"] = descriptor["handler_kind"]

        # Timeout configuration (convert ms to seconds)
        if "timeout_ms" in descriptor:
            timeout_ms = descriptor["timeout_ms"]
            if isinstance(timeout_ms, (int, float)):
                config["timeout_seconds"] = timeout_ms / 1000.0

    # Extract security configuration (basic contracts - top level)
    security = contract.get("security", {})
    if isinstance(security, dict):
        if "trusted_namespace" in security:
            config["trusted_namespace"] = security["trusted_namespace"]

        if "audit_logging" in security:
            config["audit_logging"] = security["audit_logging"]

        if "requires_authentication" in security:
            config["requires_authentication"] = security["requires_authentication"]

    # Extract metadata configuration (rich contracts)
    metadata = contract.get("metadata", {})
    if isinstance(metadata, dict):
        # Transport configuration
        transport = metadata.get("transport", {})
        if isinstance(transport, dict):
            if "default_host" in transport:
                config["host"] = transport["default_host"]

            if "default_port" in transport:
                config["port"] = transport["default_port"]

            if "default_path" in transport:
                config["path"] = transport["default_path"]

            if "stateless" in transport:
                config["stateless"] = transport["stateless"]

            if "json_response" in transport:
                config["json_response"] = transport["json_response"]

        # Security configuration (rich contracts - in metadata.security)
        metadata_security = metadata.get("security", {})
        if isinstance(metadata_security, dict):
            # Tool access configuration (for MCP)
            tool_access = metadata_security.get("tool_access", {})
            if isinstance(tool_access, dict):
                if "max_tools" in tool_access:
                    config["max_tools"] = tool_access["max_tools"]

    logger.debug(
        "Extracted handler config from contract",
        extra={
            "handler_type": handler_type,
            "config_keys": list(config.keys()),
        },
    )

    return config


__all__ = [
    "MAX_CONTRACT_SIZE_BYTES",
    "extract_handler_config",
    "load_handler_contract_config",
]
